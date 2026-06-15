"""Qwen3 with a shared head-level bucket for attention and expert routing.

This is the first falsifiable implementation of the architecture in
``fdong_inverse_kv/docs/design.md``. It intentionally supports training and
full-sequence evaluation first. Bucketed KV-cache decode is a later stage and
is rejected explicitly instead of silently using an inconsistent cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    bucket_ids: torch.Tensor


def exclusive_causal_mean_center(states: torch.Tensor) -> torch.Tensor:
    """Subtract the detached exclusive-prefix mean along the token axis.

    Args:
        states: Tensor shaped ``[batch, heads, sequence, dimension]``.

    The current state keeps its gradient. Historical states used to estimate
    the center are detached, so a later token cannot train earlier tokens by
    manipulating the centering statistic.
    """

    history = states.detach()
    prefix_sum = history.cumsum(dim=2) - history
    counts = torch.arange(states.shape[2], device=states.device, dtype=states.dtype)
    counts = counts.view(1, 1, -1, 1).clamp_min(1)
    return states - prefix_sum / counts


class HeadBucketRouter(nn.Module):
    def __init__(self, num_heads: int, input_size: int, num_experts: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.input_size = input_size
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_heads, num_experts, input_size))
        self.bias = nn.Parameter(torch.zeros(num_heads, num_experts)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=input_size**-0.5)

    def forward(self, states: torch.Tensor) -> RouterOutput:
        # states: [B, H, T, D]
        logits = torch.einsum("bhtd,hed->bhte", states, self.weight)
        if self.bias is not None:
            logits = logits + self.bias[None, :, None, :]
        probabilities = logits.float().softmax(dim=-1).to(logits.dtype)
        bucket_ids = probabilities.argmax(dim=-1)
        return RouterOutput(logits=logits, probabilities=probabilities, bucket_ids=bucket_ids)


class HeadExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = ACT2FN[activation]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(states)) * self.up_proj(states))


class HeadBucketExperts(nn.Module):
    """One independent expert set per hidden-state head.

    Qwen3-0.6B uses attention head dimension 128 but hidden_size / heads = 64.
    The attention bucket is defined by the attention head. The corresponding
    expert consumes the same-index 64-dimensional slice of the normalized
    residual stream. This keeps the FFN input/output width equal to hidden_size.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_experts = int(config.inverse_kv_num_experts)
        self.hidden_size = int(config.hidden_size)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_hidden_size = self.hidden_size // self.num_heads
        intermediate_size = int(config.inverse_kv_expert_intermediate_size)
        self.experts = nn.ModuleList(
            [
                HeadExpertMLP(self.head_hidden_size, intermediate_size, config.hidden_act)
                for _ in range(self.num_heads * self.num_experts)
            ]
        )

    def _expert(self, head_idx: int, expert_idx: int) -> HeadExpertMLP:
        return self.experts[head_idx * self.num_experts + expert_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing: RouterOutput,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        head_states = hidden_states.view(
            batch_size, sequence_length, self.num_heads, self.head_hidden_size
        )
        output = torch.zeros_like(head_states)

        # Hard expert execution in the forward pass. The selected probability
        # has value one through the straight-through scale, but supplies a
        # gradient to the router from the NTP loss.
        for head_idx in range(self.num_heads):
            head_input = head_states[:, :, head_idx, :]
            head_bucket = routing.bucket_ids[:, head_idx, :]
            head_probabilities = routing.probabilities[:, head_idx, :, :]
            head_output = torch.zeros_like(head_input)
            for expert_idx in range(self.num_experts):
                selected = head_bucket == expert_idx
                if not torch.any(selected):
                    continue
                expert_input = head_input[selected]
                expert_output = self._expert(head_idx, expert_idx)(expert_input)
                selected_probability = head_probabilities[..., expert_idx][selected]
                straight_through_scale = 1.0 + selected_probability - selected_probability.detach()
                routed_output = expert_output * straight_through_scale.unsqueeze(-1)
                head_output[selected] = routed_output.to(head_output.dtype)
            output[:, :, head_idx, :] = head_output

        return output.reshape(batch_size, sequence_length, self.hidden_size)


class OrdinaryTop1MoE(nn.Module):
    """Standard token-level top-1 MoE used as the equal-budget baseline."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_experts = int(config.inverse_kv_num_experts)
        self.router = nn.Linear(
            config.hidden_size,
            self.num_experts,
            bias=bool(config.inverse_kv_router_bias),
        )
        self.experts = nn.ModuleList(
            [
                HeadExpertMLP(
                    config.hidden_size,
                    int(config.inverse_kv_expert_intermediate_size),
                    config.hidden_act,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.last_metrics: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _record_metrics(self, routing: RouterOutput) -> None:
        probabilities = routing.probabilities.float()
        load = F.one_hot(routing.bucket_ids, num_classes=self.num_experts).float().mean(dim=(0, 1))
        load_entropy = -(load * load.clamp_min(1e-9).log()).sum()
        sorted_probabilities = probabilities.topk(k=min(2, self.num_experts), dim=-1).values
        margin = sorted_probabilities[..., 0]
        if self.num_experts > 1:
            margin = margin - sorted_probabilities[..., 1]
        token_entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        self.last_metrics = {
            "candidate_ratio": torch.ones((), device=probabilities.device),
            "router_max_probability": probabilities.max(dim=-1).values.mean().detach(),
            "router_margin": margin.mean().detach(),
            "router_token_entropy": (token_entropy.mean() / math.log(self.num_experts)).detach(),
            "router_load_entropy": (load_entropy / math.log(self.num_experts)).detach(),
            "effective_experts": load_entropy.exp().detach(),
            "max_expert_load": load.max().detach(),
            "min_expert_load": load.min().detach(),
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.router(hidden_states)
        probabilities = logits.float().softmax(dim=-1).to(logits.dtype)
        bucket_ids = probabilities.argmax(dim=-1)
        routing = RouterOutput(logits=logits, probabilities=probabilities, bucket_ids=bucket_ids)
        output = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            selected = bucket_ids == expert_idx
            if not torch.any(selected):
                continue
            expert_output = expert(hidden_states[selected])
            selected_probability = probabilities[..., expert_idx][selected]
            straight_through_scale = 1.0 + selected_probability - selected_probability.detach()
            routed_output = expert_output * straight_through_scale.unsqueeze(-1)
            output[selected] = routed_output.to(output.dtype)

        self._record_metrics(routing)
        return output


class InverseKVAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.num_experts = int(config.inverse_kv_num_experts)
        self.router_input = str(config.inverse_kv_router_input)
        self.center_router_input = bool(config.inverse_kv_center_router_input)
        self.router_normalization = str(config.inverse_kv_router_normalization)
        self.local_window = int(config.inverse_kv_local_window)
        self.sink_tokens = int(config.inverse_kv_sink_tokens)

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.router_input == "layer_input":
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads for layer_input routing")
            router_input_size = self.hidden_size // self.num_heads
        elif self.router_input in {"q", "k", "v"}:
            router_input_size = self.head_dim
        else:
            raise ValueError("inverse_kv_router_input must be one of: layer_input, q, k, v")
        self.router = HeadBucketRouter(
            num_heads=self.num_heads,
            input_size=router_input_size,
            num_experts=self.num_experts,
            bias=bool(config.inverse_kv_router_bias),
        )
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def _repeat_router_kv(self, states: torch.Tensor) -> torch.Tensor:
        # [B, H_kv, T, D] -> [B, H, T, D]
        return repeat_kv(states, self.num_key_value_groups)

    def _router_states(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.router_input == "layer_input":
            batch_size, sequence_length, _ = hidden_states.shape
            states = hidden_states.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        elif self.router_input == "q":
            states = query_states
        elif self.router_input == "k":
            states = self._repeat_router_kv(key_states)
        else:
            states = self._repeat_router_kv(value_states)

        if self.center_router_input:
            states = exclusive_causal_mean_center(states)
        if self.router_normalization == "l2":
            states = F.normalize(states, dim=-1, eps=1e-6)
        elif self.router_normalization != "none":
            raise ValueError("inverse_kv_router_normalization must be 'none' or 'l2'")
        return states

    def _bucket_allowed_mask(self, bucket_ids: torch.Tensor, base_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, query_length = bucket_ids.shape
        key_length = query_length
        device = bucket_ids.device

        query_pos = torch.arange(query_length, device=device).view(1, 1, query_length, 1)
        key_pos = torch.arange(key_length, device=device).view(1, 1, 1, key_length)
        causal = key_pos <= query_pos
        same_bucket = bucket_ids.unsqueeze(-1) == bucket_ids.unsqueeze(-2)
        allowed = same_bucket

        if self.local_window > 0:
            allowed = allowed | ((query_pos - key_pos >= 0) & (query_pos - key_pos < self.local_window))
        if self.sink_tokens > 0:
            allowed = allowed | (key_pos < self.sink_tokens)
        allowed = allowed & causal

        base_allowed = causal.expand(batch_size, self.num_heads, -1, -1)
        if base_mask is not None:
            sliced = base_mask[..., :query_length, :key_length]
            if sliced.shape[1] == 1:
                sliced = sliced.expand(-1, self.num_heads, -1, -1)
            finite_allowed = sliced == 0
            allowed = allowed & finite_allowed
            base_allowed = base_allowed & finite_allowed
        return allowed, base_allowed

    @torch.no_grad()
    def _record_metrics(
        self,
        routing: RouterOutput,
        allowed: torch.Tensor,
        base_allowed: torch.Tensor,
    ) -> None:
        probabilities = routing.probabilities.float()
        load = F.one_hot(routing.bucket_ids, num_classes=self.num_experts).float().mean(dim=(0, 2))
        mean_load = load.mean(dim=0)
        load_entropy = -(mean_load * mean_load.clamp_min(1e-9).log()).sum()
        normalized_load_entropy = load_entropy / math.log(self.num_experts)
        sorted_probabilities = probabilities.topk(k=min(2, self.num_experts), dim=-1).values
        margin = sorted_probabilities[..., 0]
        if self.num_experts > 1:
            margin = margin - sorted_probabilities[..., 1]
        token_entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        candidate_ratio = allowed.sum().float() / base_allowed.sum().clamp_min(1).float()
        self.last_metrics = {
            "candidate_ratio": candidate_ratio.detach(),
            "router_max_probability": probabilities.max(dim=-1).values.mean().detach(),
            "router_margin": margin.mean().detach(),
            "router_token_entropy": (token_entropy.mean() / math.log(self.num_experts)).detach(),
            "router_load_entropy": normalized_load_entropy.detach(),
            "effective_experts": load_entropy.exp().detach(),
            "max_expert_load": mean_load.max().detach(),
            "min_expert_load": mean_load.min().detach(),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], RouterOutput]:
        if past_key_value is not None:
            raise NotImplementedError(
                "Bucketed KV-cache decode is not implemented in the training prototype. "
                "Call with use_cache=False."
            )

        input_shape = hidden_states.shape[:-1]
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        router_states = self._router_states(hidden_states, query_states, key_states, value_states)
        routing = self.router(router_states)

        cos, sin = position_embeddings
        query_states, rope_key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(rope_key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        allowed, base_allowed = self._bucket_allowed_mask(routing.bucket_ids, attention_mask)
        min_value = torch.finfo(query_states.dtype).min
        bucket_mask = torch.zeros_like(allowed, dtype=query_states.dtype).masked_fill(~allowed, min_value)
        attention_logits = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attention_logits = attention_logits + bucket_mask
        attention_weights = F.softmax(attention_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = attention_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attention_output = self.o_proj(attention_output)

        self._record_metrics(routing, allowed, base_allowed)
        return attention_output, attention_weights if output_attentions else None, routing


class InverseKVDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = InverseKVAttention(config, layer_idx)
        self.mlp = HeadBucketExperts(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output, attention_weights, routing = self.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        expert_input = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(expert_input, routing)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs


class OrdinaryMoEDecoderLayer(nn.Module):
    """Standard Qwen3 full attention followed by token-level top-1 MoE."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = OrdinaryTop1MoE(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs


def apply_inverse_kv_config_defaults(config: Qwen3Config) -> Qwen3Config:
    defaults = {
        "inverse_kv_architecture": "shared_bucket",
        "inverse_kv_router_input": "k",
        "inverse_kv_center_router_input": True,
        "inverse_kv_router_normalization": "l2",
        "inverse_kv_router_bias": False,
        "inverse_kv_num_experts": 4,
        "inverse_kv_expert_intermediate_size": 3072,
        "inverse_kv_local_window": 32,
        "inverse_kv_sink_tokens": 4,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    config.use_cache = False
    config._attn_implementation = "eager"
    return config


class MyQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM with an ordinary-MoE or shared-bucket decoder."""

    def __init__(self, config: Qwen3Config):
        config = apply_inverse_kv_config_defaults(config)
        super().__init__(config)
        architecture = str(config.inverse_kv_architecture)
        if architecture == "shared_bucket":
            layer_type = InverseKVDecoderLayer
        elif architecture == "ordinary_moe":
            layer_type = OrdinaryMoEDecoderLayer
        else:
            raise ValueError("inverse_kv_architecture must be 'ordinary_moe' or 'shared_bucket'")
        layers = nn.ModuleList([layer_type(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        layers.apply(self._init_weights)
        self.model.layers = layers

    def routing_metrics(self) -> Dict[str, float]:
        per_metric: Dict[str, list[torch.Tensor]] = {}
        for layer in self.model.layers:
            metric_source = layer.self_attn if hasattr(layer.self_attn, "last_metrics") else layer.mlp
            for name, value in metric_source.last_metrics.items():
                per_metric.setdefault(name, []).append(value.float())
        return {
            name: torch.stack(values).mean().item()
            for name, values in per_metric.items()
            if values
        }


__all__ = [
    "MyQwen3ForCausalLM",
    "InverseKVDecoderLayer",
    "InverseKVAttention",
    "OrdinaryMoEDecoderLayer",
    "OrdinaryTop1MoE",
    "HeadBucketRouter",
    "exclusive_causal_mean_center",
    "apply_inverse_kv_config_defaults",
]
