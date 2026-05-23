from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RealAttentionClusterConfig:
    attention_top_ratio: float = 0.10
    expert_input_top_ratio: float = 0.10
    include_self: bool = False
    attention_cluster_temperature: float = 1.0
    attention_cluster_detach_attention: bool = True
    attention_cluster_detach_key_router: bool = False
    load_balance_temperature: float = 1.0


def _first_tensor_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    hidden_states = kwargs.get("hidden_states")
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    raise ValueError("Expected hidden_states as the first positional argument or keyword argument.")


def _replace_first_tensor_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    value: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args and isinstance(args[0], torch.Tensor):
        return (value, *args[1:]), kwargs
    if "hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = value
        return args, kwargs
    return (value, *args), kwargs


def _module_config(module: torch.nn.Module):
    current = module
    while current is not None:
        config = getattr(current, "config", None)
        if config is not None:
            return config
        current = getattr(current, "_real_cluster_parent", None)
    return None


def _num_heads(attn: torch.nn.Module) -> int:
    for name in ("num_heads", "num_attention_heads"):
        value = getattr(attn, name, None)
        if value is not None:
            return int(value)
    config = _module_config(attn)
    if config is not None:
        return int(getattr(config, "num_attention_heads"))
    raise AttributeError("Could not infer number of attention heads.")


def _num_key_value_heads(attn: torch.nn.Module) -> int:
    for name in ("num_key_value_heads", "num_kv_heads"):
        value = getattr(attn, name, None)
        if value is not None:
            return int(value)
    config = _module_config(attn)
    if config is not None:
        return int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads")))
    return _num_heads(attn)


def _head_dim(attn: torch.nn.Module) -> int:
    value = getattr(attn, "head_dim", None)
    if value is not None:
        return int(value)
    config = _module_config(attn)
    if config is not None:
        hidden_size = int(getattr(config, "hidden_size"))
        return hidden_size // _num_heads(attn)
    raise AttributeError("Could not infer attention head dimension.")


def _repeat_kv(value_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return value_states
    batch, num_key_value_heads, seq_len, head_dim = value_states.shape
    value_states = value_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        seq_len,
        head_dim,
    )
    return value_states.reshape(batch, num_key_value_heads * num_key_value_groups, seq_len, head_dim)


def _valid_history_mask(seq_len: int, device: torch.device, include_self: bool) -> torch.Tensor:
    diagonal = 0 if include_self else -1
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=diagonal)


def top_ratio_attention_weights(
    attention_weights: torch.Tensor,
    ratio: float,
    include_self: bool,
) -> torch.Tensor:
    if attention_weights.dim() != 4:
        raise ValueError(f"Expected attention weights [batch, heads, query, key], got {tuple(attention_weights.shape)}.")
    ratio = float(ratio)
    if ratio <= 0.0:
        return torch.zeros_like(attention_weights)

    batch, heads, query_len, key_len = attention_weights.shape
    if query_len != key_len:
        raise ValueError("This training patch currently expects full-sequence no-cache attention.")
    valid = _valid_history_mask(query_len, attention_weights.device, include_self)
    valid = valid[None, None, :, :]
    weights = attention_weights.float().masked_fill(~valid, 0.0)

    valid_counts = valid.expand(batch, heads, query_len, key_len).sum(dim=-1)
    keep_counts = torch.ceil(valid_counts.float() * ratio).long().clamp(min=1, max=key_len)
    sorted_values, sorted_indices = torch.sort(weights, dim=-1, descending=True)
    ranks = torch.arange(key_len, device=weights.device)[None, None, None, :]
    keep_sorted = (ranks < keep_counts[..., None]) & (sorted_values > 0)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_indices, keep_sorted)
    selected = weights * keep.to(weights.dtype)

    denom = selected.sum(dim=-1, keepdim=True)
    fallback = denom <= 0
    normalized = selected / denom.clamp_min(1e-8)
    full_denom = weights.sum(dim=-1, keepdim=True)
    full_normalized = weights / full_denom.clamp_min(1e-8)
    return torch.where(fallback, full_normalized, normalized).to(attention_weights.dtype)


def attention_cluster_loss_from_logits(
    router_logits: list[torch.Tensor],
    attentions: list[torch.Tensor],
    cfg: RealAttentionClusterConfig,
) -> torch.Tensor | None:
    losses = []
    temp = max(float(cfg.attention_cluster_temperature), 1e-6)
    for logits, attn in zip(router_logits, attentions):
        if logits.dim() != 3:
            continue
        if cfg.attention_cluster_detach_attention:
            attn = attn.detach()
        weights = top_ratio_attention_weights(attn, cfg.attention_top_ratio, cfg.include_self).float()
        probs_q = F.softmax(logits.float() / temp, dim=-1)
        probs_k = probs_q.detach() if cfg.attention_cluster_detach_key_router else probs_q
        same_expert_prob = torch.einsum("bqe,bke->bqk", probs_q, probs_k)[:, None, :, :]
        denom = weights.sum()
        if float(denom.detach().cpu()) <= 0.0:
            continue
        losses.append(-(weights * same_expert_prob.clamp_min(1e-8).log()).sum() / denom.clamp_min(1e-8))
    if not losses:
        return None
    return torch.stack(losses).mean()


def load_balance_loss_from_logits(
    router_logits: list[torch.Tensor],
    top_k: int,
    temperature: float,
) -> torch.Tensor | None:
    losses = []
    temp = max(float(temperature), 1e-6)
    for logits in router_logits:
        if logits.dim() != 3 or logits.shape[-1] <= 1:
            continue
        flat_logits = logits.reshape(-1, logits.shape[-1]).float()
        routing_probs = F.softmax(flat_logits / temp, dim=-1)
        k = max(1, min(int(top_k), flat_logits.shape[-1]))
        topk_indices = torch.topk(routing_probs, k=k, dim=-1).indices
        tokens_per_expert = F.one_hot(topk_indices, num_classes=flat_logits.shape[-1]).float()
        tokens_per_expert = tokens_per_expert.mean(dim=0).sum(dim=0) / k
        prob_per_expert = routing_probs.mean(dim=0)
        losses.append(flat_logits.shape[-1] * torch.sum(tokens_per_expert * prob_per_expert))
    if not losses:
        return None
    return torch.stack(losses).mean()


class RealAttentionClusterPatch:
    def __init__(self, model: torch.nn.Module, cfg: RealAttentionClusterConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.handles: list[Any] = []
        self.router_logits: list[torch.Tensor] = []
        self.attentions: list[torch.Tensor] = []
        self.num_patched_layers = 0

    def clear(self) -> None:
        self.router_logits.clear()
        self.attentions.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attention_cluster_loss(self) -> torch.Tensor | None:
        return attention_cluster_loss_from_logits(self.router_logits, self.attentions, self.cfg)

    def load_balance_loss(self, top_k: int) -> torch.Tensor | None:
        return load_balance_loss_from_logits(
            self.router_logits,
            top_k=top_k,
            temperature=self.cfg.load_balance_temperature,
        )

    def apply(self) -> None:
        layers = self._find_decoder_layers()
        if not layers:
            raise ValueError("Could not find decoder layers with self_attn and MoE mlp.gate.")
        for layer in layers:
            self._patch_layer(layer)
        self.num_patched_layers = len(layers)

    def _find_decoder_layers(self) -> list[torch.nn.Module]:
        layers = []
        for module in self.model.modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                mlp = getattr(module, "mlp")
                if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                    layers.append(module)
                    getattr(module, "self_attn")._real_cluster_parent = module
                    mlp._real_cluster_parent = module
        return layers

    def _patch_layer(self, layer: torch.nn.Module) -> None:
        self.handles.append(layer.register_forward_pre_hook(self._make_layer_pre_hook(layer), with_kwargs=True))
        self.handles.append(
            layer.self_attn.register_forward_pre_hook(self._make_attn_pre_hook(layer), with_kwargs=True)
        )
        self.handles.append(layer.self_attn.register_forward_hook(self._make_attn_hook(layer), with_kwargs=True))
        self.handles.append(layer.mlp.register_forward_pre_hook(self._make_mlp_pre_hook(layer), with_kwargs=True))
        self.handles.append(layer.mlp.gate.register_forward_pre_hook(self._make_gate_pre_hook(layer), with_kwargs=True))
        self.handles.append(layer.mlp.gate.register_forward_hook(self._make_gate_hook(layer), with_kwargs=True))

    def _make_layer_pre_hook(self, layer: torch.nn.Module):
        def hook(_module, args, kwargs):
            layer._real_cluster_residual_source = _first_tensor_arg(args, kwargs)
            return args, kwargs

        return hook

    def _make_attn_pre_hook(self, layer: torch.nn.Module):
        def hook(_module, args, kwargs):
            layer._real_cluster_attn_input = _first_tensor_arg(args, kwargs)
            return args, kwargs

        return hook

    def _make_attn_hook(self, layer: torch.nn.Module):
        def hook(module, _args, _kwargs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise ValueError("Expected attention output tuple containing attention weights.")
            attn_weights = output[1]
            if attn_weights is None:
                raise ValueError("Attention cluster training requires output_attentions=True and eager attention.")

            attn_input = layer._real_cluster_attn_input
            residual_source = layer._real_cluster_residual_source
            router_states = self._q_router_states(module, attn_input)
            expert_attn = self._sparse_expert_attention(module, attn_input, attn_weights)
            layer.mlp._real_cluster_router_states = router_states
            layer.mlp._real_cluster_expert_input = layer.post_attention_layernorm(residual_source + expert_attn)
            layer.mlp._real_cluster_last_attention = attn_weights
            return output

        return hook

    def _make_mlp_pre_hook(self, layer: torch.nn.Module):
        def hook(module, args, kwargs):
            expert_input = getattr(module, "_real_cluster_expert_input", None)
            if expert_input is None:
                return args, kwargs
            return _replace_first_tensor_arg(args, kwargs, expert_input)

        return hook

    def _make_gate_pre_hook(self, layer: torch.nn.Module):
        def hook(module, args, kwargs):
            router_states = getattr(layer.mlp, "_real_cluster_router_states", None)
            if router_states is None:
                return args, kwargs
            gate_input = _first_tensor_arg(args, kwargs)
            router_states = router_states.reshape(gate_input.shape).to(device=gate_input.device, dtype=gate_input.dtype)
            return _replace_first_tensor_arg(args, kwargs, router_states)

        return hook

    def _make_gate_hook(self, layer: torch.nn.Module):
        def hook(_module, _args, _kwargs, output):
            router_states = getattr(layer.mlp, "_real_cluster_router_states", None)
            if router_states is None:
                return output
            batch, seq_len = router_states.shape[:2]
            logits = output.reshape(batch, seq_len, output.shape[-1])
            self.router_logits.append(logits)
            self.attentions.append(layer.mlp._real_cluster_last_attention)
            return output

        return hook

    def _q_router_states(self, attn: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        num_heads = _num_heads(attn)
        head_dim = _head_dim(attn)
        q_states = attn.q_proj(hidden_states).view(batch, seq_len, num_heads, head_dim)
        q_norm = getattr(attn, "q_norm", None)
        if q_norm is not None:
            q_states = q_norm(q_states)
        return q_states.reshape(batch, seq_len, num_heads * head_dim)

    def _sparse_expert_attention(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        num_kv_heads = _num_key_value_heads(attn)
        num_heads = _num_heads(attn)
        head_dim = _head_dim(attn)
        num_key_value_groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
        value_states = attn.v_proj(hidden_states).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = _repeat_kv(value_states, num_key_value_groups)
        sparse_weights = top_ratio_attention_weights(
            attn_weights,
            self.cfg.expert_input_top_ratio,
            self.cfg.include_self,
        )
        sparse_output = torch.matmul(sparse_weights.to(value_states.dtype), value_states)
        sparse_output = sparse_output.transpose(1, 2).contiguous().reshape(batch, seq_len, num_heads * head_dim)
        return attn.o_proj(sparse_output)
