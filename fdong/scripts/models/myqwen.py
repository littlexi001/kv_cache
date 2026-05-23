from functools import partial
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .qwen3config import Qwen3Config


class AnchorOnlyDynamicCache(Cache):
    """
    Dynamic cache variant for mask-based U-Net Transformer inference.

    Attention is computed with the current token's K/V still present, so the
    current position can attend to itself exactly as it did during training.
    After each layer finishes attention, non-anchor K/V entries are removed
    from stride layers because future positions can never attend to them.
    """

    def __init__(self, attention_stride_pattern: List[int]):
        try:
            super().__init__()
        except TypeError:
            pass
        self.attention_stride_pattern = [int(stride) for stride in attention_stride_pattern]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.position_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def _ensure_layer(self, layer_idx: int):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.position_cache.append(None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            start = self._seen_tokens
            cache_position = torch.arange(start, start + key_states.shape[-2], device=key_states.device)
        cache_position = cache_position.to(device=key_states.device, dtype=torch.long)

        if cache_position.numel() > 0:
            self._seen_tokens = max(self._seen_tokens, int(cache_position.max().item()) + 1)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.position_cache[layer_idx] = cache_position
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.position_cache[layer_idx] = torch.cat([self.position_cache[layer_idx], cache_position], dim=0)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def prune_layer(self, layer_idx: int):
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return

        stride = self.attention_stride_pattern[layer_idx]
        if stride == 1:
            return

        positions = self.position_cache[layer_idx]
        keep_mask = (positions + 1) % stride == 0
        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_mask, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_mask, :]
        self.position_cache[layer_idx] = positions[keep_mask]

    def get_layer_positions(self, layer_idx: int) -> Optional[torch.Tensor]:
        if layer_idx >= len(self.position_cache):
            return None
        return self.position_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(self.key_cache[layer_idx].device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(self.value_cache[layer_idx].device))

    def crop(self, max_length: int):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is None:
                continue
            keep_mask = self.position_cache[layer_idx] < max_length
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_mask, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_mask, :]
            self.position_cache[layer_idx] = self.position_cache[layer_idx][keep_mask]
        self._seen_tokens = min(self._seen_tokens, max_length)

    def get_cache_lengths(self) -> Dict[int, int]:
        lengths = {}
        for layer_idx, key_states in enumerate(self.key_cache):
            lengths[layer_idx] = 0 if key_states is None else key_states.shape[-2]
        return lengths


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MyQwen3MLP(nn.Module):
    def __init__(self, config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MyQwen3DimMLP(nn.Module):
    def __init__(self, config, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MyQwen3Router(nn.Module):
    def __init__(self, input_size: int, num_experts: int, config):
        super().__init__()
        self.router_type = str(getattr(config, "moe_router_type", "linear"))
        self.router_bias = bool(getattr(config, "moe_router_bias", False))
        if self.router_type == "linear":
            self.net = nn.Linear(input_size, num_experts, bias=self.router_bias)
        elif self.router_type == "mlp":
            hidden_size = int(getattr(config, "moe_router_hidden_size", input_size))
            act_name = str(getattr(config, "moe_router_act", "silu"))
            if act_name not in ACT2FN:
                raise ValueError(f"Unsupported `moe_router_act`: {act_name}.")
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=self.router_bias),
                ACT2FN[act_name],
                nn.Linear(hidden_size, num_experts, bias=self.router_bias),
            )
        else:
            raise ValueError("`moe_router_type` must be either 'linear' or 'mlp'.")

    def forward(self, hidden_states):
        return self.net(hidden_states)


class MyQwen3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_unique_experts = int(getattr(config, "moe_num_unique_experts", 0))
        self.num_experts_per_tok = int(getattr(config, "moe_num_experts_per_tok", 1))
        self.use_common_expert = bool(getattr(config, "moe_use_common_expert", False))
        self.normalize_topk_prob = bool(getattr(config, "moe_normalize_topk_prob", True))
        self.router_bias = bool(getattr(config, "moe_router_bias", False))
        self.moe_intermediate_size = int(getattr(config, "moe_intermediate_size", config.intermediate_size))
        self.common_intermediate_size = int(
            getattr(config, "moe_common_intermediate_size", self.moe_intermediate_size)
        )

        if self.num_unique_experts < 1:
            raise ValueError("`moe_num_unique_experts` must be >= 1 when `use_moe=True`.")
        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.num_unique_experts:
            raise ValueError(
                "`moe_num_experts_per_tok` must be in [1, moe_num_unique_experts], "
                f"got {self.num_experts_per_tok} and {self.num_unique_experts}."
            )

        self.router = MyQwen3Router(self.hidden_size, self.num_unique_experts, config)
        self.experts = nn.ModuleList(
            [MyQwen3MLP(config, intermediate_size=self.moe_intermediate_size) for _ in range(self.num_unique_experts)]
        )
        self.common_expert = (
            MyQwen3MLP(config, intermediate_size=self.common_intermediate_size)
            if self.use_common_expert
            else None
        )

    def load_balance_loss(self, router_logits, topk_indices):
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        tokens_per_expert = F.one_hot(topk_indices, num_classes=self.num_unique_experts).float()
        tokens_per_expert = tokens_per_expert.mean(dim=0).sum(dim=0) / self.num_experts_per_tok
        router_prob_per_expert = routing_probs.mean(dim=0)
        return self.num_unique_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def router_inhibition_loss(self, router_logits):
        temperature = max(float(getattr(self.config, "moe_router_inhibition_temperature", 1.0)), 1e-6)
        winners = router_logits.detach().argmax(dim=-1)
        return F.cross_entropy(router_logits.float() / temperature, winners)

    def forward(
        self,
        hidden_states,
        output_expert_labels: bool = False,
        output_router_aux_loss: bool = False,
        output_router_inhibition_loss: bool = False,
        output_router_supervision_loss: bool = False,
        router_supervision_detach_input: bool = False,
        router_hidden_states: Optional[torch.Tensor] = None,
        router_logits_override: Optional[torch.Tensor] = None,
        ground_truth_expert_ids: Optional[torch.Tensor] = None,
        router_supervision_expert_ids: Optional[torch.Tensor] = None,
    ):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        if router_hidden_states is None:
            router_hidden_states = hidden_states
        flat_router_states = router_hidden_states.reshape(-1, self.hidden_size)

        router_logits = None
        if ground_truth_expert_ids is not None:
            if self.num_experts_per_tok != 1:
                raise ValueError("Ground-truth MoE routing currently requires `moe_num_experts_per_tok=1`.")
            topk_indices = ground_truth_expert_ids.reshape(-1, 1).to(flat_states.device)
            if topk_indices.numel() != flat_states.shape[0]:
                raise ValueError(
                    "`ground_truth_expert_ids` must align with hidden states, "
                    f"got {tuple(ground_truth_expert_ids.shape)} for hidden shape {tuple(original_shape)}."
                )
            if (topk_indices < 0).any() or (topk_indices >= self.num_unique_experts).any():
                raise ValueError("Ground-truth expert ids must be in [0, moe_num_unique_experts).")
            topk_weights = torch.ones((flat_states.shape[0], 1), dtype=flat_states.dtype, device=flat_states.device)
        else:
            if router_logits_override is None:
                router_logits = self.router(flat_router_states)
            else:
                router_logits = router_logits_override.reshape(-1, self.num_unique_experts).to(flat_states.device)
                if router_logits.shape[0] != flat_states.shape[0]:
                    raise ValueError(
                        "`router_logits_override` must align with hidden states, "
                        f"got {tuple(router_logits_override.shape)} for hidden shape {tuple(original_shape)}."
                    )
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_indices = torch.topk(
                routing_weights,
                k=self.num_experts_per_tok,
                dim=-1,
            )
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_weights = topk_weights.to(flat_states.dtype)

        final_states = torch.zeros_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat_states[token_idx])
            final_states[token_idx] += expert_output * topk_weights[token_idx, slot_idx].unsqueeze(-1)

        if self.common_expert is not None:
            final_states = final_states + self.common_expert(flat_states)

        final_states = final_states.reshape(original_shape)
        if (
            not output_expert_labels
            and not output_router_aux_loss
            and not output_router_inhibition_loss
            and not output_router_supervision_loss
        ):
            return final_states

        outputs = (final_states,)
        if output_expert_labels:
            expert_labels = topk_indices.reshape(*original_shape[:-1], self.num_experts_per_tok)
            outputs += (expert_labels,)
        if output_router_aux_loss:
            if router_logits is None:
                outputs += (hidden_states.new_zeros(()),)
            else:
                outputs += (self.load_balance_loss(router_logits, topk_indices),)
        if output_router_inhibition_loss:
            if router_logits is None:
                outputs += (hidden_states.new_zeros(()),)
            else:
                outputs += (self.router_inhibition_loss(router_logits),)
        if output_router_supervision_loss:
            if router_logits is None:
                raise ValueError("Router supervision requires learned gate routing, not ground-truth dispatch.")
            if router_supervision_expert_ids is None:
                raise ValueError("Router supervision requires `router_supervision_expert_ids`.")
            supervision_router_states = flat_router_states.detach() if router_supervision_detach_input else flat_router_states
            supervision_router_logits = self.router(supervision_router_states)
            target_expert_ids = router_supervision_expert_ids.reshape(-1).to(flat_states.device)
            if target_expert_ids.numel() != flat_states.shape[0]:
                raise ValueError(
                    "`router_supervision_expert_ids` must align with hidden states, "
                    f"got {tuple(router_supervision_expert_ids.shape)} for hidden shape {tuple(original_shape)}."
                )
            if (target_expert_ids < 0).any() or (target_expert_ids >= self.num_unique_experts).any():
                raise ValueError("Router supervision expert ids must be in [0, moe_num_unique_experts).")
            outputs += (F.cross_entropy(supervision_router_logits.float(), target_expert_ids.long()),)
        return outputs


class MyQwen3HeadMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.num_unique_experts = int(getattr(config, "moe_num_unique_experts", 0))
        self.num_experts_per_tok = int(getattr(config, "moe_num_experts_per_tok", 1))
        self.use_common_expert = bool(getattr(config, "moe_use_common_expert", False))
        self.normalize_topk_prob = bool(getattr(config, "moe_normalize_topk_prob", True))
        self.router_bias = bool(getattr(config, "moe_router_bias", False))
        self.moe_intermediate_size = int(getattr(config, "moe_intermediate_size", config.intermediate_size))
        self.common_intermediate_size = int(
            getattr(config, "moe_common_intermediate_size", self.moe_intermediate_size)
        )

        if self.num_unique_experts < 1:
            raise ValueError("`moe_num_unique_experts` must be >= 1 when `use_moe=True`.")
        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.num_unique_experts:
            raise ValueError(
                "`moe_num_experts_per_tok` must be in [1, moe_num_unique_experts], "
                f"got {self.num_experts_per_tok} and {self.num_unique_experts}."
            )

        self.routers = nn.ModuleList(
            [MyQwen3Router(self.head_dim, self.num_unique_experts, config) for _ in range(self.num_heads)]
        )
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MyQwen3DimMLP(config, hidden_size=self.head_dim, intermediate_size=self.moe_intermediate_size)
                        for _ in range(self.num_unique_experts)
                    ]
                )
                for _ in range(self.num_heads)
            ]
        )
        self.common_experts = (
            nn.ModuleList(
                [
                    MyQwen3DimMLP(config, hidden_size=self.head_dim, intermediate_size=self.common_intermediate_size)
                    for _ in range(self.num_heads)
                ]
            )
            if self.use_common_expert
            else None
        )

    def load_balance_loss(self, router_logits, topk_indices):
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        tokens_per_expert = F.one_hot(topk_indices, num_classes=self.num_unique_experts).float()
        tokens_per_expert = tokens_per_expert.mean(dim=0).sum(dim=0) / self.num_experts_per_tok
        router_prob_per_expert = routing_probs.mean(dim=0)
        return self.num_unique_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def router_inhibition_loss(self, router_logits):
        temperature = max(float(getattr(self.config, "moe_router_inhibition_temperature", 1.0)), 1e-6)
        winners = router_logits.detach().argmax(dim=-1)
        return F.cross_entropy(router_logits.float() / temperature, winners)

    def forward(
        self,
        hidden_states,
        output_expert_labels: bool = False,
        output_router_aux_loss: bool = False,
        output_router_inhibition_loss: bool = False,
        output_router_supervision_loss: bool = False,
        router_supervision_detach_input: bool = False,
        router_hidden_states: Optional[torch.Tensor] = None,
    ):
        # hidden_states/router_hidden_states: [batch, seq, heads, head_dim]
        if output_router_supervision_loss:
            raise ValueError("Router supervision is currently implemented for token-level MoE, not head-level MoE.")
        if hidden_states.dim() != 4:
            raise ValueError(f"Head-level MoE expects [batch, seq, heads, head_dim], got {tuple(hidden_states.shape)}.")
        if router_hidden_states is None:
            router_hidden_states = hidden_states
        if router_hidden_states.shape != hidden_states.shape:
            raise ValueError(
                "`router_hidden_states` must match head-level expert input shape, "
                f"got {tuple(router_hidden_states.shape)} vs {tuple(hidden_states.shape)}."
            )

        batch, seq_len, num_heads, head_dim = hidden_states.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"Expected {self.num_heads} heads with dim {self.head_dim}, got {num_heads} heads with dim {head_dim}."
            )

        head_outputs = []
        head_labels = []
        head_aux_losses = []
        head_inhibition_losses = []
        for head_idx in range(self.num_heads):
            head_states = hidden_states[:, :, head_idx, :].reshape(-1, self.head_dim)
            head_router_states = router_hidden_states[:, :, head_idx, :].reshape(-1, self.head_dim)
            router_logits = self.routers[head_idx](head_router_states)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_indices = torch.topk(
                routing_weights,
                k=self.num_experts_per_tok,
                dim=-1,
            )
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_weights = topk_weights.to(head_states.dtype)

            final_states = torch.zeros_like(head_states)
            for expert_idx, expert in enumerate(self.experts[head_idx]):
                token_idx, slot_idx = torch.where(topk_indices == expert_idx)
                if token_idx.numel() == 0:
                    continue
                expert_output = expert(head_states[token_idx])
                final_states[token_idx] += expert_output * topk_weights[token_idx, slot_idx].unsqueeze(-1)

            if self.common_experts is not None:
                final_states = final_states + self.common_experts[head_idx](head_states)

            head_outputs.append(final_states.reshape(batch, seq_len, self.head_dim))
            if output_expert_labels:
                head_labels.append(topk_indices.reshape(batch, seq_len, self.num_experts_per_tok))
            if output_router_aux_loss:
                head_aux_losses.append(self.load_balance_loss(router_logits, topk_indices))
            if output_router_inhibition_loss:
                head_inhibition_losses.append(self.router_inhibition_loss(router_logits))

        output = torch.stack(head_outputs, dim=2)
        if not output_expert_labels and not output_router_aux_loss and not output_router_inhibition_loss:
            return output

        outputs = (output,)
        if output_expert_labels:
            expert_labels = torch.stack(head_labels, dim=2)
            outputs += (expert_labels,)
        if output_router_aux_loss:
            outputs += (torch.stack(head_aux_losses).mean(),)
        if output_router_inhibition_loss:
            outputs += (torch.stack(head_inhibition_losses).mean(),)
        return outputs


def _remove_self_and_renormalize_attention(attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = attn.shape[-1]
    eye = torch.eye(seq_len, device=attn.device, dtype=torch.bool)
    attn = attn.masked_fill(eye.unsqueeze(0), 0.0)
    row_sum = attn.sum(dim=-1, keepdim=True)
    valid_rows = row_sum.squeeze(-1) > 0
    attn = attn / row_sum.clamp_min(1e-8)
    return attn, valid_rows


def _top_mass_attention(attn: torch.Tensor, rho: float) -> torch.Tensor:
    rho = float(rho)
    sorted_vals, sorted_idx = torch.sort(attn, dim=-1, descending=True)
    cumsum = sorted_vals.cumsum(dim=-1)
    keep_sorted = cumsum <= rho
    first_over = (cumsum > rho).to(torch.int64).argmax(dim=-1, keepdim=True)
    keep_sorted.scatter_(-1, first_over, True)
    keep_sorted = keep_sorted & (sorted_vals > 0)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    selected = attn * keep.to(attn.dtype)
    return selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def attention_derived_router_loss(
    router_logits: torch.Tensor,
    attn_weights: torch.Tensor,
    loss_type: str,
    rho: float,
    topk: int,
) -> torch.Tensor:
    if attn_weights is None:
        return router_logits.new_zeros(())
    if router_logits.dim() != 3:
        raise ValueError(f"`router_logits` must be [batch, seq, experts], got {tuple(router_logits.shape)}.")

    router_probs = F.softmax(router_logits.float(), dim=-1)
    attn = attn_weights.float().mean(dim=1)
    attn, valid_rows = _remove_self_and_renormalize_attention(attn)
    selected = _top_mass_attention(attn, rho=rho)
    valid = valid_rows.to(router_probs.dtype)
    valid_count = valid.sum().clamp_min(1.0)

    loss_type = str(loss_type)
    if loss_type == "kl":
        teacher = selected @ router_probs.detach()
        token_loss = F.kl_div(
            router_probs.clamp_min(1e-8).log(),
            teacher.clamp_min(1e-8),
            reduction="none",
        ).sum(dim=-1)
    elif loss_type == "pairwise":
        same_prob = torch.einsum("bqe,bke->bqk", router_probs, router_probs.detach())
        token_loss = -(selected * same_prob.clamp_min(1e-8).log()).sum(dim=-1)
    elif loss_type == "topk_logits":
        k = max(1, min(int(topk), router_logits.shape[-1]))
        logits = router_logits.float()
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        topk_logits = torch.zeros_like(logits)
        topk_logits.scatter_(-1, topk_idx, topk_vals)
        teacher_logits = selected @ topk_logits.detach()
        teacher_vals, teacher_idx = torch.topk(teacher_logits, k=k, dim=-1)
        masked_teacher = torch.full_like(teacher_logits, torch.finfo(teacher_logits.dtype).min)
        masked_teacher.scatter_(-1, teacher_idx, teacher_vals)
        teacher = F.softmax(masked_teacher, dim=-1)
        token_loss = F.kl_div(
            router_probs.clamp_min(1e-8).log(),
            teacher.clamp_min(1e-8),
            reduction="none",
        ).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported attention-derived router loss type: {loss_type}.")

    return (token_loss * valid).sum() / valid_count


def router_entropy_floor_loss(router_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    if router_logits.dim() != 3:
        raise ValueError(f"`router_logits` must be [batch, seq, experts], got {tuple(router_logits.shape)}.")
    num_experts = router_logits.shape[-1]
    if num_experts <= 1:
        return router_logits.new_zeros(())

    router_probs = F.softmax(router_logits.float(), dim=-1)
    usage = router_probs.mean(dim=(0, 1))
    entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
    target_entropy = float(alpha) * math.log(float(num_experts))
    return F.relu(router_logits.new_tensor(target_entropy) - entropy).to(router_logits.dtype)


def pre_router_kv_additive_mask(router_logits: torch.Tensor, topk: int, dtype: torch.dtype) -> torch.Tensor:
    if router_logits.dim() != 3:
        raise ValueError(f"`router_logits` must be [batch, seq, experts], got {tuple(router_logits.shape)}.")
    batch, seq_len, num_experts = router_logits.shape
    k = max(1, min(int(topk), num_experts))
    topk_idx = torch.topk(router_logits.float(), k=k, dim=-1).indices
    overlap = topk_idx[:, :, None, :, None] == topk_idx[:, None, :, None, :]
    same_bucket = overlap.any(dim=(-1, -2))
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=router_logits.device))
    allowed = same_bucket & causal.unsqueeze(0)
    min_dtype = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seq_len, seq_len), dtype=dtype, device=router_logits.device)
    return mask.masked_fill(~allowed[:, None, :, :], min_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def topk_attention_output_from_weights(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    topk: int,
) -> torch.Tensor:
    value_states = repeat_kv(value, num_key_value_groups)
    key_len = attn_weights.shape[-1]
    k = max(1, min(int(topk), key_len))
    top_values, top_indices = torch.topk(attn_weights.float(), k=k, dim=-1)
    sparse_weights = torch.zeros_like(attn_weights, dtype=top_values.dtype)
    sparse_weights.scatter_(-1, top_indices, top_values)
    sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sparse_output = torch.matmul(sparse_weights.to(value_states.dtype), value_states)
    return sparse_output.transpose(1, 2).contiguous()


class MyQwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.moe_expert_input_attention_topk = int(getattr(config, "moe_expert_input_attention_topk", 0))

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(hidden_states.shape)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                pass 
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        head_attn_output = attn_output
        if self.moe_expert_input_attention_topk > 0:
            if attn_weights is None:
                raise ValueError("`moe_expert_input_attention_topk` requires attention weights; use eager attention.")
            expert_head_attn_output = topk_attention_output_from_weights(
                attn_weights,
                value_states,
                self.num_key_value_groups,
                self.moe_expert_input_attention_topk,
            )
        else:
            expert_head_attn_output = head_attn_output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.moe_expert_input_attention_topk > 0:
            expert_attn_output = expert_head_attn_output.reshape(*input_shape, -1).contiguous()
            expert_attn_output = self.o_proj(expert_attn_output)
        else:
            expert_attn_output = attn_output
        return attn_output, attn_weights, head_attn_output, expert_attn_output, expert_head_attn_output


class MyQwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        print(f"init layer {layer_idx}", end="\r")
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.moe_router_input = str(getattr(config, "moe_router_input", "hidden"))
        self.moe_head_level = bool(getattr(config, "moe_head_level", False))
        self.use_pre_router = bool(getattr(config, "use_pre_router", False))
        self.pre_router_input = str(getattr(config, "pre_router_input", "layer_input"))
        self.pre_router_controls_attention = bool(getattr(config, "pre_router_controls_attention", False))
        self.attention_router_loss_type = str(getattr(config, "attention_router_loss_type", "kl"))
        self.attention_router_rho = float(getattr(config, "attention_router_rho", 0.75))
        self.router_entropy_floor_alpha = float(getattr(config, "router_entropy_floor_alpha", 0.5))
        if self.moe_router_input not in {"hidden", "attention_output"}:
            raise ValueError("`moe_router_input` must be either 'hidden' or 'attention_output'.")
        if self.pre_router_input not in {"layer_input", "q", "k", "v"}:
            raise ValueError("`pre_router_input` must be one of 'layer_input', 'q', 'k', or 'v'.")
        self.self_attn = MyQwen3Attention(config=config, layer_idx=layer_idx)
        if bool(getattr(config, "use_moe", False)):
            self.mlp = MyQwen3HeadMoE(config) if self.moe_head_level else MyQwen3MoE(config)
        else:
            self.mlp = MyQwen3MLP(config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_source: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_expert_labels: Optional[bool] = False,
        output_router_aux_loss: Optional[bool] = False,
        output_router_inhibition_loss: Optional[bool] = False,
        output_router_supervision_loss: Optional[bool] = False,
        output_attention_router_loss: Optional[bool] = False,
        output_router_entropy_floor_loss: Optional[bool] = False,
        use_pre_router_kv_mask: Optional[bool] = False,
        router_supervision_detach_input: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        ground_truth_expert_ids: Optional[torch.Tensor] = None,
        router_supervision_expert_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if residual_source is None:
            residual_source = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        pre_router_logits = None
        if self.use_pre_router:
            if not isinstance(self.mlp, MyQwen3MoE) or self.moe_head_level:
                raise ValueError("Pre-router routing is currently implemented for token-level MoE only.")
            if ground_truth_expert_ids is not None or router_supervision_expert_ids is not None:
                raise ValueError("Pre-router routing should not be combined with ground-truth router supervision.")
            if self.pre_router_input == "q":
                pre_router_states = self.self_attn.q_norm(
                    self.self_attn.q_proj(hidden_states).view(
                        *hidden_states.shape[:-1], self.num_attention_heads, self.head_dim
                    )
                ).reshape(*hidden_states.shape[:-1], -1)
            elif self.pre_router_input == "k":
                pre_router_states = self.self_attn.k_norm(
                    self.self_attn.k_proj(hidden_states).view(
                        *hidden_states.shape[:-1], self.num_key_value_heads, self.head_dim
                    )
                )
                pre_router_states = pre_router_states.repeat_interleave(
                    self.self_attn.num_key_value_groups,
                    dim=-2,
                ).reshape(*hidden_states.shape[:-1], -1)
            elif self.pre_router_input == "v":
                pre_router_states = self.self_attn.v_proj(hidden_states).view(
                    *hidden_states.shape[:-1], self.num_key_value_heads, self.head_dim
                )
                pre_router_states = pre_router_states.repeat_interleave(
                    self.self_attn.num_key_value_groups,
                    dim=-2,
                ).reshape(*hidden_states.shape[:-1], -1)
            else:
                pre_router_states = hidden_states
            pre_router_logits = self.mlp.router(pre_router_states)
            if self.pre_router_controls_attention or use_pre_router_kv_mask:
                kv_mask = pre_router_kv_additive_mask(
                    pre_router_logits,
                    topk=self.mlp.num_experts_per_tok,
                    dtype=hidden_states.dtype,
                )
                if attention_mask is not None and kv_mask.shape[-1] != attention_mask.shape[-1]:
                    pad = attention_mask.shape[-1] - kv_mask.shape[-1]
                    if pad < 0:
                        kv_mask = kv_mask[:, :, :, : attention_mask.shape[-1]]
                    else:
                        kv_mask = F.pad(kv_mask, (0, pad), value=0.0)
                attention_mask = kv_mask if attention_mask is None else attention_mask + kv_mask

        # Self Attention
        (
            attn_output_wo_res,
            self_attn_weights,
            head_attn_output,
            expert_attn_output_wo_res,
            expert_head_attn_output,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual_source + attn_output_wo_res
        expert_hidden_states = residual_source + expert_attn_output_wo_res

        # Fully Connected
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(expert_hidden_states)
        expert_labels = None
        router_aux_loss = None
        router_inhibition_loss = None
        router_supervision_loss = None
        attention_router_loss = None
        router_entropy_loss = None
        if isinstance(self.mlp, MyQwen3HeadMoE):
            if ground_truth_expert_ids is not None:
                raise ValueError("Ground-truth expert routing is not implemented for head-level MoE.")
            if router_supervision_expert_ids is not None:
                raise ValueError("Router supervision is not implemented for head-level MoE.")
            residual_heads = residual_source.reshape(
                residual_source.shape[0], residual_source.shape[1], self.num_attention_heads, self.head_dim
            )
            head_expert_input = residual_heads + expert_head_attn_output
            if self.moe_router_input == "attention_output":
                head_router_input = head_attn_output
            else:
                head_router_input = head_expert_input
            head_output = self.mlp(
                head_expert_input,
                output_expert_labels=output_expert_labels,
                output_router_aux_loss=output_router_aux_loss,
                output_router_inhibition_loss=output_router_inhibition_loss,
                output_router_supervision_loss=output_router_supervision_loss,
                router_supervision_detach_input=router_supervision_detach_input,
                router_hidden_states=head_router_input,
            )
            if isinstance(head_output, tuple):
                tuple_idx = 1
                if output_expert_labels:
                    expert_labels = head_output[tuple_idx]
                    tuple_idx += 1
                if output_router_aux_loss:
                    router_aux_loss = head_output[tuple_idx]
                    tuple_idx += 1
                if output_router_inhibition_loss:
                    router_inhibition_loss = head_output[tuple_idx]
                    tuple_idx += 1
                if output_router_supervision_loss:
                    router_supervision_loss = head_output[tuple_idx]
                head_output = head_output[0]
            hidden_states = head_output.reshape(head_output.shape[0], head_output.shape[1], -1)
        elif isinstance(self.mlp, MyQwen3MoE):
            router_input = None
            if self.moe_router_input == "attention_output":
                router_input = self.post_attention_layernorm(attn_output_wo_res)
            hidden_states = self.mlp(
                mlp_input,
                output_expert_labels=output_expert_labels,
                output_router_aux_loss=output_router_aux_loss,
                output_router_inhibition_loss=output_router_inhibition_loss,
                output_router_supervision_loss=output_router_supervision_loss,
                router_supervision_detach_input=router_supervision_detach_input,
                router_hidden_states=router_input,
                router_logits_override=pre_router_logits,
                ground_truth_expert_ids=ground_truth_expert_ids,
                router_supervision_expert_ids=router_supervision_expert_ids,
            )
            if output_attention_router_loss:
                if pre_router_logits is None:
                    attention_router_loss = hidden_states.new_zeros(())
                else:
                    attention_router_loss = attention_derived_router_loss(
                        router_logits=pre_router_logits,
                        attn_weights=self_attn_weights,
                        loss_type=self.attention_router_loss_type,
                        rho=self.attention_router_rho,
                        topk=self.mlp.num_experts_per_tok,
                    )
            if output_router_entropy_floor_loss:
                if pre_router_logits is None:
                    router_entropy_loss = hidden_states.new_zeros(())
                else:
                    router_entropy_loss = router_entropy_floor_loss(
                        router_logits=pre_router_logits,
                        alpha=self.router_entropy_floor_alpha,
                    )
        else:
            hidden_states = self.mlp(mlp_input)

        if isinstance(hidden_states, tuple):
            tuple_idx = 1
            if output_expert_labels:
                expert_labels = hidden_states[tuple_idx]
                tuple_idx += 1
            if output_router_aux_loss:
                router_aux_loss = hidden_states[tuple_idx]
                tuple_idx += 1
            if output_router_inhibition_loss:
                router_inhibition_loss = hidden_states[tuple_idx]
                tuple_idx += 1
            if output_router_supervision_loss:
                router_supervision_loss = hidden_states[tuple_idx]
                tuple_idx += 1
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if output_expert_labels:
            outputs += (expert_labels, )

        if output_router_aux_loss:
            if router_aux_loss is None:
                router_aux_loss = hidden_states.new_zeros(())
            outputs += (router_aux_loss, )

        if output_router_inhibition_loss:
            if router_inhibition_loss is None:
                router_inhibition_loss = hidden_states.new_zeros(())
            outputs += (router_inhibition_loss, )

        if output_router_supervision_loss:
            if router_supervision_loss is None:
                router_supervision_loss = hidden_states.new_zeros(())
            outputs += (router_supervision_loss, )

        if output_attention_router_loss:
            if attention_router_loss is None:
                attention_router_loss = hidden_states.new_zeros(())
            outputs += (attention_router_loss, )

        if output_router_entropy_floor_loss:
            if router_entropy_loss is None:
                router_entropy_loss = hidden_states.new_zeros(())
            outputs += (router_entropy_loss, )

        return outputs


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MyQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        attention_stride_pattern = self._normalize_layer_pattern(
            config.attention_stride_pattern,
            default_value=1,
            name="attention_stride_pattern",
        )
        residual_source_pattern = self._normalize_layer_pattern(
            config.residual_source_pattern,
            default_value=-1,
            name="residual_source_pattern",
        )
        self.attention_stride_pattern = self._validate_attention_stride_pattern(attention_stride_pattern)
        self.residual_source_pattern = self._validate_residual_source_pattern(residual_source_pattern)
        self._uses_stride_attention = any(stride != 1 for stride in self.attention_stride_pattern)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MyQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _normalize_layer_pattern(self, pattern: Optional[List[int]], default_value: int, name: str) -> List[int]:
        if pattern is None:
            return [default_value] * self.config.num_hidden_layers
        if len(pattern) != self.config.num_hidden_layers:
            raise ValueError(
                f"`{name}` must have length {self.config.num_hidden_layers}, got {len(pattern)}."
            )
        return [int(value) for value in pattern]

    def _validate_attention_stride_pattern(self, pattern: List[int]) -> List[int]:
        if len(pattern) != self.config.num_hidden_layers:
            raise ValueError(
                f"`attention_stride_pattern` must have length {self.config.num_hidden_layers}, got {len(pattern)}."
            )
        if any(stride < 1 for stride in pattern):
            raise ValueError("All values in `attention_stride_pattern` must be positive integers.")
        return [int(stride) for stride in pattern]

    def _validate_residual_source_pattern(self, pattern: List[int]) -> List[int]:
        if len(pattern) != self.config.num_hidden_layers:
            raise ValueError(
                f"`residual_source_pattern` must have length {self.config.num_hidden_layers}, got {len(pattern)}."
            )
        normalized = [int(source) for source in pattern]
        for layer_idx, source in enumerate(normalized):
            if source == -1:
                continue
            if source < 0 or source >= layer_idx:
                raise ValueError(
                    "`residual_source_pattern` uses 0-based layer-output indices. "
                    f"Layer {layer_idx} can only use -1 or a source in [0, {layer_idx - 1}], got {source}."
                )
        return normalized

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        anchor_only_kv_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_expert_labels:Optional[bool] = None,
        output_router_aux_loss: Optional[bool] = None,
        output_router_inhibition_loss: Optional[bool] = None,
        output_router_supervision_loss: Optional[bool] = None,
        output_attention_router_loss: Optional[bool] = None,
        output_router_entropy_floor_loss: Optional[bool] = None,
        use_pre_router_kv_mask: Optional[bool] = False,
        router_supervision_detach_input: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ground_truth_expert_ids: Optional[torch.LongTensor] = None,
        router_supervision_expert_ids: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if anchor_only_kv_cache:
                past_key_values = AnchorOnlyDynamicCache(self.attention_stride_pattern)
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_expert_labels = () if output_expert_labels else None
        all_router_aux_losses = () if output_router_aux_loss else None
        all_router_inhibition_losses = () if output_router_inhibition_loss else None
        all_router_supervision_losses = () if output_router_supervision_loss else None
        all_attention_router_losses = () if output_attention_router_loss else None
        all_router_entropy_floor_losses = () if output_router_entropy_floor_loss else None
        layer_hidden_states = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            residual_source_idx = self.residual_source_pattern[layer_idx]
            residual_source = None if residual_source_idx == -1 else layer_hidden_states[residual_source_idx]
            layer_attention_mask = self._apply_attention_stride_mask(
                causal_mask,
                stride=self.attention_stride_pattern[layer_idx],
                cache_position=cache_position,
                key_positions=self._get_layer_key_positions(
                    past_key_values=past_key_values,
                    layer_idx=layer_idx,
                    cache_position=cache_position,
                    target_length=causal_mask.shape[-1] if causal_mask is not None else None,
                    device=inputs_embeds.device,
                ),
                dtype=inputs_embeds.dtype,
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    residual_source,
                    layer_attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_expert_labels,
                    output_router_aux_loss,
                    output_router_inhibition_loss,
                    output_router_supervision_loss,
                    output_attention_router_loss,
                    output_router_entropy_floor_loss,
                    use_pre_router_kv_mask,
                    router_supervision_detach_input,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    ground_truth_expert_ids,
                    router_supervision_expert_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    residual_source=residual_source,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_expert_labels = output_expert_labels,
                    output_router_aux_loss=output_router_aux_loss,
                    output_router_inhibition_loss=output_router_inhibition_loss,
                    output_router_supervision_loss=output_router_supervision_loss,
                    output_attention_router_loss=output_attention_router_loss,
                    output_router_entropy_floor_loss=output_router_entropy_floor_loss,
                    use_pre_router_kv_mask=use_pre_router_kv_mask,
                    router_supervision_detach_input=router_supervision_detach_input,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ground_truth_expert_ids=ground_truth_expert_ids,
                    router_supervision_expert_ids=router_supervision_expert_ids,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]
            layer_hidden_states.append(hidden_states)

            if use_cache and anchor_only_kv_cache and hasattr(past_key_values, "prune_layer"):
                past_key_values.prune_layer(layer_idx)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_expert_labels:
                label_idx = 2 if output_attentions else 1
                all_expert_labels += (layer_outputs[label_idx],)

            if output_router_aux_loss:
                aux_idx = 1
                if output_attentions:
                    aux_idx += 1
                if output_expert_labels:
                    aux_idx += 1
                all_router_aux_losses += (layer_outputs[aux_idx],)

            if output_router_inhibition_loss:
                inhibition_idx = 1
                if output_attentions:
                    inhibition_idx += 1
                if output_expert_labels:
                    inhibition_idx += 1
                if output_router_aux_loss:
                    inhibition_idx += 1
                all_router_inhibition_losses += (layer_outputs[inhibition_idx],)

            if output_router_supervision_loss:
                supervision_idx = 1
                if output_attentions:
                    supervision_idx += 1
                if output_expert_labels:
                    supervision_idx += 1
                if output_router_aux_loss:
                    supervision_idx += 1
                if output_router_inhibition_loss:
                    supervision_idx += 1
                all_router_supervision_losses += (layer_outputs[supervision_idx],)

            if output_attention_router_loss:
                attention_router_idx = 1
                if output_attentions:
                    attention_router_idx += 1
                if output_expert_labels:
                    attention_router_idx += 1
                if output_router_aux_loss:
                    attention_router_idx += 1
                if output_router_inhibition_loss:
                    attention_router_idx += 1
                if output_router_supervision_loss:
                    attention_router_idx += 1
                all_attention_router_losses += (layer_outputs[attention_router_idx],)

            if output_router_entropy_floor_loss:
                entropy_idx = 1
                if output_attentions:
                    entropy_idx += 1
                if output_expert_labels:
                    entropy_idx += 1
                if output_router_aux_loss:
                    entropy_idx += 1
                if output_router_inhibition_loss:
                    entropy_idx += 1
                if output_router_supervision_loss:
                    entropy_idx += 1
                if output_attention_router_loss:
                    entropy_idx += 1
                all_router_entropy_floor_losses += (layer_outputs[entropy_idx],)
                

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output =  BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
        if output_expert_labels:
            output.expert_labels = all_expert_labels

        if output_router_aux_loss:
            output.moe_load_balance_loss = torch.stack(all_router_aux_losses).mean()

        if output_router_inhibition_loss:
            output.moe_router_inhibition_loss = torch.stack(all_router_inhibition_losses).mean()

        if output_router_supervision_loss:
            output.moe_router_supervision_loss = torch.stack(all_router_supervision_losses).mean()

        if output_attention_router_loss:
            output.attention_router_loss = torch.stack(all_attention_router_losses).mean()

        if output_router_entropy_floor_loss:
            output.router_entropy_floor_loss = torch.stack(all_router_entropy_floor_losses).mean()

        return output

    def _apply_attention_stride_mask(
        self,
        causal_mask: Optional[torch.Tensor],
        stride: int,
        cache_position: torch.Tensor,
        key_positions: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if causal_mask is None:
            return causal_mask

        if key_positions is None:
            key_positions = torch.arange(causal_mask.shape[-1], device=causal_mask.device)
            uses_sparse_positions = False
        else:
            key_positions = key_positions.to(causal_mask.device)
            default_positions = torch.arange(key_positions.numel(), device=causal_mask.device)
            uses_sparse_positions = not torch.equal(key_positions, default_positions)

        target_length = key_positions.numel()
        causal_mask = causal_mask[:, :, :, :target_length]
        if uses_sparse_positions:
            min_dtype = torch.finfo(dtype).min
            query_positions = cache_position.reshape(-1, 1).to(causal_mask.device)
            future_mask = key_positions.unsqueeze(0) > query_positions
            causal_mask = torch.zeros_like(causal_mask).masked_fill(future_mask[None, None, :, :], min_dtype)

        if stride == 1:
            return causal_mask

        min_dtype = torch.finfo(dtype).min
        anchor_mask = (key_positions + 1) % stride == 0
        self_mask = key_positions.unsqueeze(0) == cache_position.reshape(-1, 1).to(causal_mask.device)
        allowed_mask = anchor_mask.unsqueeze(0) | self_mask
        allowed_mask = allowed_mask[None, None, :, :]
        return causal_mask.masked_fill(~allowed_mask, min_dtype)

    def _get_layer_key_positions(
        self,
        past_key_values: Optional[Cache],
        layer_idx: int,
        cache_position: torch.Tensor,
        target_length: Optional[int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if isinstance(past_key_values, AnchorOnlyDynamicCache):
            previous_positions = past_key_values.get_layer_positions(layer_idx)
            current_positions = cache_position.to(device=device, dtype=torch.long)
            if previous_positions is None:
                return current_positions
            return torch.cat([previous_positions.to(device=device), current_positions], dim=0)

        if target_length is None:
            return None
        return torch.arange(target_length, device=device, dtype=torch.long)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if self._uses_stride_attention:
                raise ValueError(
                    "Layer-wise stride attention requires an explicit 4D additive mask. "
                    "Please use `attn_implementation='eager'` or `attn_implementation='sdpa'`."
                )
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
            and not self._uses_stride_attention
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class MyQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        anchor_only_kv_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_expert_labels:Optional[bool] = None,
        output_router_aux_loss: Optional[bool] = None,
        output_router_inhibition_loss: Optional[bool] = None,
        output_router_supervision_loss: Optional[bool] = None,
        output_attention_router_loss: Optional[bool] = None,
        output_router_entropy_floor_loss: Optional[bool] = None,
        use_pre_router_kv_mask: Optional[bool] = False,
        router_supervision_detach_input: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        ground_truth_expert_ids: Optional[torch.LongTensor] = None,
        router_supervision_expert_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            anchor_only_kv_cache=anchor_only_kv_cache,
            output_attentions=output_attentions,
            output_expert_labels = output_expert_labels,
            output_router_aux_loss=output_router_aux_loss,
            output_router_inhibition_loss=output_router_inhibition_loss,
            output_router_supervision_loss=output_router_supervision_loss,
            output_attention_router_loss=output_attention_router_loss,
            output_router_entropy_floor_loss=output_router_entropy_floor_loss,
            use_pre_router_kv_mask=use_pre_router_kv_mask,
            router_supervision_detach_input=router_supervision_detach_input,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            ground_truth_expert_ids=ground_truth_expert_ids,
            router_supervision_expert_ids=router_supervision_expert_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        output =  CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if outputs.hidden_states is not None else outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
        
        if output_expert_labels:
            output.expert_labels = outputs.expert_labels

        if output_router_aux_loss:
            output.moe_load_balance_loss = outputs.moe_load_balance_loss

        if output_router_inhibition_loss:
            output.moe_router_inhibition_loss = outputs.moe_router_inhibition_loss

        if output_router_supervision_loss:
            output.moe_router_supervision_loss = outputs.moe_router_supervision_loss

        if output_attention_router_loss:
            output.attention_router_loss = outputs.attention_router_loss

        if output_router_entropy_floor_loss:
            output.router_entropy_floor_loss = outputs.router_entropy_floor_loss

        return output
