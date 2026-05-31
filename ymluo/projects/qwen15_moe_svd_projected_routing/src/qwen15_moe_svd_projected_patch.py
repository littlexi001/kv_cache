from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SvdProjectedRoutingConfig:
    projection_source: str = "q"
    svd_refresh_interval: int = 100
    force_refresh_first_forward: bool = True
    group1_experts: int = 16
    group2_experts: int = 24
    group3_experts: int = 8
    group1_topk: int = 2
    group2_topk: int = 3
    group3_topk: int = 1
    normalize_topk_prob: bool = True
    add_shared_expert: bool = True


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


def _linear_weight(attn: nn.Module, source: str) -> torch.Tensor:
    proj = getattr(attn, f"{source}_proj", None)
    if proj is None or not hasattr(proj, "weight"):
        raise AttributeError(f"Could not find attention projection `{source}_proj.weight`.")
    return proj.weight


def _input_singular_basis(weight: torch.Tensor) -> torch.Tensor:
    # torch.nn.Linear stores weight as [out_features, in_features]. For y = x @ W.T,
    # the input-space singular directions are columns of V.
    _u, _s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return vh.transpose(0, 1).contiguous()


def _slice_by_percent(features: torch.Tensor, start: float, end: float) -> torch.Tensor:
    dim = features.shape[-1]
    left = min(dim, max(0, int(math.floor(dim * start))))
    right = min(dim, max(left + 1, int(math.ceil(dim * end))))
    return features[..., left:right]


def _ensure_gate(module: nn.Module, name: str, in_features: int, out_features: int, device, dtype) -> nn.Linear:
    gate = getattr(module, name, None)
    if gate is None or gate.in_features != in_features or gate.out_features != out_features:
        gate = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        nn.init.normal_(gate.weight, mean=0.0, std=0.02)
        setattr(module, name, gate)
    return gate


def _parameter_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters():
        return param.device, param.dtype
    return torch.device("cpu"), torch.float32


def _group_topk(
    logits: torch.Tensor,
    expert_offset: int,
    top_k: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.float(), dim=-1)
    k = max(1, min(int(top_k), logits.shape[-1]))
    weights, local_indices = torch.topk(probs, k=k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return weights, local_indices + int(expert_offset)


def _expert_count(module: nn.Module) -> int:
    experts = getattr(module, "experts", None)
    if experts is None:
        raise AttributeError("Patched MoE module must have an `experts` ModuleList.")
    return len(experts)


def _module_hidden_size(module: nn.Module, hidden_states: torch.Tensor) -> int:
    value = getattr(module, "hidden_size", None)
    if value is not None:
        return int(value)
    return int(hidden_states.shape[-1])


def _call_expert(expert: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    output = expert(hidden_states)
    if isinstance(output, tuple):
        return output[0]
    return output


def _svd_projected_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
    original_shape = hidden_states.shape
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected hidden_states [batch, seq, hidden], got {tuple(hidden_states.shape)}.")

    batch, seq_len, hidden_size = original_shape
    flat_states = hidden_states.reshape(-1, hidden_size)
    layer_input = getattr(self, "_svd_projected_layer_input", hidden_states)
    flat_layer_input = layer_input.reshape(-1, hidden_size)

    basis = getattr(self, "_svd_projected_basis", None)
    if basis is None:
        raise RuntimeError("SVD projected routing basis has not been initialized.")
    features = flat_layer_input.float() @ basis.to(flat_layer_input.device, dtype=torch.float32)

    group1_features = _slice_by_percent(features, 0.02, 0.10)
    group2_features = _slice_by_percent(features, 0.11, 0.70)
    group3_features = _slice_by_percent(features, 0.71, 1.00)
    device = hidden_states.device
    dtype = hidden_states.dtype
    cfg: SvdProjectedRoutingConfig = self._svd_projected_cfg

    logits0 = self.svd_gate_expert0(features[:, :1].to(dtype)).float()
    logits1 = self.svd_gate_group1(group1_features.to(dtype)).float()
    logits2 = self.svd_gate_group2(group2_features.to(dtype)).float()
    logits3 = self.svd_gate_group3(group3_features.to(dtype)).float()

    weights0 = torch.sigmoid(logits0)
    indices0 = torch.zeros((flat_states.shape[0], 1), device=device, dtype=torch.long)
    weights1, indices1 = _group_topk(logits1, 1, cfg.group1_topk, cfg.normalize_topk_prob)
    weights2, indices2 = _group_topk(logits2, 1 + cfg.group1_experts, cfg.group2_topk, cfg.normalize_topk_prob)
    weights3, indices3 = _group_topk(
        logits3,
        1 + cfg.group1_experts + cfg.group2_experts,
        cfg.group3_topk,
        cfg.normalize_topk_prob,
    )

    selected_weights = torch.cat([weights0, weights1, weights2, weights3], dim=-1)
    selected_experts = torch.cat([indices0, indices1, indices2, indices3], dim=-1)
    if cfg.normalize_topk_prob:
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    selected_weights = selected_weights.to(dtype)

    final_states = torch.zeros_like(flat_states)
    experts = getattr(self, "experts")
    for expert_idx, expert in enumerate(experts):
        token_idx, slot_idx = torch.where(selected_experts == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_output = _call_expert(expert, flat_states[token_idx])
        final_states[token_idx] += expert_output * selected_weights[token_idx, slot_idx].unsqueeze(-1)

    if cfg.add_shared_expert:
        shared_expert = getattr(self, "shared_expert", None)
        shared_gate = getattr(self, "shared_expert_gate", None)
        if shared_expert is not None:
            shared_output = _call_expert(shared_expert, flat_states)
            if shared_gate is not None:
                shared_output = torch.sigmoid(shared_gate(flat_states)) * shared_output
            final_states = final_states + shared_output

    router_logits = torch.cat([logits0, logits1, logits2, logits3], dim=-1)
    self._svd_projected_last_logits = router_logits.reshape(batch, seq_len, -1)
    self._svd_projected_last_indices = selected_experts.reshape(batch, seq_len, -1)
    final_states = final_states.reshape(original_shape)
    return final_states, router_logits


class SvdProjectedRoutingPatch:
    def __init__(self, model: nn.Module, cfg: SvdProjectedRoutingConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.handles: list[Any] = []
        self.router_logits: list[torch.Tensor] = []
        self.num_patched_layers = 0

    @property
    def total_experts(self) -> int:
        return 1 + self.cfg.group1_experts + self.cfg.group2_experts + self.cfg.group3_experts

    @property
    def total_active_experts(self) -> int:
        return 1 + self.cfg.group1_topk + self.cfg.group2_topk + self.cfg.group3_topk

    def clear(self) -> None:
        self.router_logits.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def apply(self) -> None:
        layers = self._find_decoder_layers()
        if not layers:
            raise ValueError("Could not find decoder layers with self_attn and MoE mlp.experts.")
        for layer_idx, layer in enumerate(layers):
            self._patch_layer(layer, layer_idx)
        self.num_patched_layers = len(layers)

    def _find_decoder_layers(self) -> list[nn.Module]:
        layers = []
        for module in self.model.modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):
                mlp = getattr(module, "mlp")
                if hasattr(mlp, "experts"):
                    layers.append(module)
        return layers

    def _patch_layer(self, layer: nn.Module, layer_idx: int) -> None:
        mlp = layer.mlp
        if _expert_count(mlp) != self.total_experts:
            raise ValueError(
                f"Layer {layer_idx} has {_expert_count(mlp)} experts, expected {self.total_experts}. "
                "Set config.num_experts to 49 before model initialization."
            )
        mlp._svd_projected_cfg = self.cfg
        mlp._svd_projected_source_attn = layer.self_attn
        mlp._svd_projected_forward_calls = 0
        original_gate = getattr(mlp, "gate", None)
        if original_gate is not None:
            for param in original_gate.parameters():
                param.requires_grad_(False)
        self._init_gates(mlp)
        mlp.forward = types.MethodType(_svd_projected_forward, mlp)
        self.handles.append(layer.register_forward_pre_hook(self._make_layer_pre_hook(mlp), with_kwargs=True))
        self.handles.append(mlp.register_forward_hook(self._make_mlp_hook(mlp), with_kwargs=True))
        self._refresh_basis(mlp, force=True)

    def _init_gates(self, mlp: nn.Module) -> None:
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        group1_dim = _slice_by_percent(torch.empty(1, hidden_size), 0.02, 0.10).shape[-1]
        group2_dim = _slice_by_percent(torch.empty(1, hidden_size), 0.11, 0.70).shape[-1]
        group3_dim = _slice_by_percent(torch.empty(1, hidden_size), 0.71, 1.00).shape[-1]
        device, dtype = _parameter_device_dtype(mlp)
        _ensure_gate(mlp, "svd_gate_expert0", 1, 1, device, dtype)
        _ensure_gate(mlp, "svd_gate_group1", group1_dim, self.cfg.group1_experts, device, dtype)
        _ensure_gate(mlp, "svd_gate_group2", group2_dim, self.cfg.group2_experts, device, dtype)
        _ensure_gate(mlp, "svd_gate_group3", group3_dim, self.cfg.group3_experts, device, dtype)

    def _make_layer_pre_hook(self, mlp: nn.Module):
        def hook(_module, args, kwargs):
            hidden_states = _first_tensor_arg(args, kwargs)
            mlp._svd_projected_layer_input = hidden_states
            mlp._svd_projected_forward_calls += 1
            interval = max(1, int(self.cfg.svd_refresh_interval))
            if mlp._svd_projected_forward_calls % interval == 1:
                self._refresh_basis(mlp, force=False)
            return args, kwargs

        return hook

    def _make_mlp_hook(self, mlp: nn.Module):
        def hook(_module, _args, _kwargs, output):
            logits = getattr(mlp, "_svd_projected_last_logits", None)
            if logits is not None:
                self.router_logits.append(logits)
            return output

        return hook

    def _refresh_basis(self, mlp: nn.Module, force: bool) -> None:
        with torch.no_grad():
            weight = _linear_weight(mlp._svd_projected_source_attn, self.cfg.projection_source)
            basis = _input_singular_basis(weight.detach()).to(weight.device)
            mlp._svd_projected_basis = basis

    def load_balance_loss(self) -> torch.Tensor | None:
        losses = []
        for logits in self.router_logits:
            if logits.dim() != 3:
                continue
            probs = F.softmax(logits.float(), dim=-1)
            expert_prob = probs.mean(dim=(0, 1))
            losses.append(logits.shape[-1] * torch.sum(expert_prob * expert_prob))
        if not losses:
            return None
        return torch.stack(losses).mean()
