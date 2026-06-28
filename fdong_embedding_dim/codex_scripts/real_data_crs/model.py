from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    hidden_size: int = 1_024
    intermediate_size: int = 2_560
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 2_048
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    variant: str = "dense"
    common_scale: float = 1.0
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.variant not in {"dense", "crs"}:
            raise ValueError("variant must be 'dense' or 'crs'")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).to(dtype)


def causal_common_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split each x[:, k] using only x[:, :k].

    The direction estimate is detached. Position zero has no prefix and is sent
    entirely through the residual branch.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [batch, sequence, dim], got {tuple(x.shape)}")
    x_detached = x.detach().float()
    prefix_sum_inclusive = torch.cumsum(x_detached, dim=1)
    zero = torch.zeros_like(prefix_sum_inclusive[:, :1])
    prefix_sum = torch.cat((zero, prefix_sum_inclusive[:, :-1]), dim=1)
    unit = F.normalize(prefix_sum, dim=-1, eps=1e-8).to(x.dtype)
    coefficient = (x * unit).sum(dim=-1, keepdim=True)
    residual = x - coefficient * unit
    return coefficient, residual, unit


def split_statistics(
    x: torch.Tensor,
    coefficient: torch.Tensor,
    residual: torch.Tensor,
    unit: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        x2 = x.float().square().sum(dim=-1)
        common2 = coefficient.float().square().squeeze(-1)
        residual2 = residual.float().square().sum(dim=-1)
        nonzero = x2 > 1e-12
        common_fraction = torch.where(nonzero, common2 / x2.clamp_min(1e-12), torch.zeros_like(x2))
        leakage = (residual.float() * unit.float()).sum(dim=-1).abs()
        leakage = leakage / residual2.sqrt().clamp_min(1e-12)
        valid_unit = unit.float().square().sum(dim=-1) > 0.5
        return {
            "common_energy_fraction": common_fraction[nonzero].mean() if nonzero.any() else x2.new_zeros(()),
            "residual_common_abs_cos": leakage[valid_unit].mean() if valid_unit.any() else x2.new_zeros(()),
        }


class DenseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_std: float) -> None:
        super().__init__(in_features, out_features, bias=False)
        nn.init.normal_(self.weight, mean=0.0, std=init_std)


class CRSLinear(nn.Module):
    """Low-parameter common/residual split linear map.

    For a causal prefix direction u_k and scalar a_k = <x_k, u_k>:

        y_k = alpha * a_k * c_out + W_residual * (x_k - a_k u_k).

    The common expert contains only c_out, so it adds out_features parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_std: float,
        common_scale: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.common_scale = common_scale
        self.residual = DenseLinear(in_features, out_features, init_std)
        with torch.no_grad():
            default_direction = torch.ones(in_features) / math.sqrt(in_features)
            common_init = self.residual.weight.float() @ default_direction
        self.common_out = nn.Parameter(common_init)

    def forward_from_split(
        self,
        coefficient: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        common_output = coefficient * self.common_out.to(coefficient.dtype)
        return self.residual(residual) + self.common_scale * common_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficient, residual, _ = causal_common_split(x)
        return self.forward_from_split(coefficient, residual)

    @property
    def weight(self) -> torch.Tensor:
        return self.residual.weight


def make_linear(config: ModelConfig, in_features: int, out_features: int) -> nn.Module:
    if config.variant == "crs":
        return CRSLinear(in_features, out_features, config.initializer_range, config.common_scale)
    return DenseLinear(in_features, out_features, config.initializer_range)


def apply_shared_input(
    modules: Tuple[nn.Module, ...],
    x: torch.Tensor,
    collect_diagnostics: bool,
) -> Tuple[Tuple[torch.Tensor, ...], Optional[Dict[str, torch.Tensor]]]:
    if not modules or not isinstance(modules[0], CRSLinear):
        return tuple(module(x) for module in modules), None
    coefficient, residual, unit = causal_common_split(x)
    outputs = tuple(module.forward_from_split(coefficient, residual) for module in modules)
    stats = split_statistics(x, coefficient, residual, unit) if collect_diagnostics else None
    return outputs, stats


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        angles = torch.outer(positions, inv_freq)
        embedding = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.register_buffer("cos", embedding.cos(), persistent=False)
        self.register_buffer("sin", embedding.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos[:seq_len].to(device=q.device, dtype=q.dtype)[None, None]
        sin = self.sin[:seq_len].to(device=q.device, dtype=q.dtype)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class QwenAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = make_linear(config, config.hidden_size, q_dim)
        self.k_proj = make_linear(config, config.hidden_size, kv_dim)
        self.v_proj = make_linear(config, config.hidden_size, kv_dim)
        self.o_proj = make_linear(config, q_dim, config.hidden_size)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        (q, k, v), qkv_stats = apply_shared_input(
            (self.q_proj, self.k_proj, self.v_proj), x, collect_diagnostics
        )
        q = q.view(bsz, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, -1)
        if isinstance(self.o_proj, CRSLinear):
            coefficient, residual, unit = causal_common_split(attn)
            output = self.o_proj.forward_from_split(coefficient, residual)
            o_stats = split_statistics(attn, coefficient, residual, unit) if collect_diagnostics else None
        else:
            output = self.o_proj(attn)
            o_stats = None
        stats = {}
        if qkv_stats is not None:
            stats["qkv_input"] = qkv_stats
        if o_stats is not None:
            stats["o_input"] = o_stats
        return output, stats


class QwenMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = make_linear(config, config.hidden_size, config.intermediate_size)
        self.up_proj = make_linear(config, config.hidden_size, config.intermediate_size)
        self.down_proj = make_linear(config, config.intermediate_size, config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        (gate, up), gate_up_stats = apply_shared_input(
            (self.gate_proj, self.up_proj), x, collect_diagnostics
        )
        intermediate = F.silu(gate) * up
        if isinstance(self.down_proj, CRSLinear):
            coefficient, residual, unit = causal_common_split(intermediate)
            output = self.down_proj.forward_from_split(coefficient, residual)
            down_stats = split_statistics(intermediate, coefficient, residual, unit) if collect_diagnostics else None
        else:
            output = self.down_proj(intermediate)
            down_stats = None
        stats = {}
        if gate_up_stats is not None:
            stats["gate_up_input"] = gate_up_stats
        if down_stats is not None:
            stats["down_input"] = down_stats
        return output, stats


class QwenBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = QwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = QwenMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        attn_output, attn_stats = self.self_attn(self.input_layernorm(x), collect_diagnostics)
        x = x + attn_output
        mlp_output, mlp_stats = self.mlp(self.post_attention_layernorm(x), collect_diagnostics)
        x = x + mlp_output
        return x, {**attn_stats, **mlp_stats}


class QwenCRSForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=config.initializer_range)
        self.layers = nn.ModuleList(QwenBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        diagnostics: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for layer_idx, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training and not collect_diagnostics:
                x = checkpoint(lambda value: layer(value, False)[0], x, use_reentrant=False)
            else:
                x, layer_stats = layer(x, collect_diagnostics)
                if collect_diagnostics:
                    diagnostics[f"layer_{layer_idx}"] = layer_stats
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return logits, diagnostics

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    common = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("common_out")
    )
    return {"total": total, "common_expert": common, "non_common": total - common}
