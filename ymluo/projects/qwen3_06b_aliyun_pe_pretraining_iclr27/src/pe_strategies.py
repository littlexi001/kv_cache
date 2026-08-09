from __future__ import annotations

import json
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from io_utils import read_json, write_json


ALLOWED_KINDS = {
    "native",
    "deep_highfreq_drop",
    "uniform_scale",
    "smooth_layer_frequency",
    "smooth_remote_warp",
    "periodic_nope",
    "band_layer_scale",
    "period_aware_scale",
    "phase_diverse_deep",
    "layer_pair_matrix",
}


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    values: dict[str, Any]


def load_strategy(path: str | Path) -> Strategy:
    path = Path(path)
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError("strategy JSON must be an object")
    name = str(value.get("name", "")).strip()
    kind = str(value.get("kind", "")).strip()
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid strategy name: {name!r}")
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unknown strategy kind {kind!r}; allowed={sorted(ALLOWED_KINDS)}")
    return Strategy(name=name, kind=kind, values=value)


def _sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(value)


def pair_scales(
    strategy: Strategy,
    layer_index: int,
    num_layers: int,
    num_pairs: int,
    device: torch.device | str = "cpu",
    inv_freq: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_layers <= 0 or num_pairs <= 0:
        raise ValueError("num_layers and num_pairs must be positive")
    if not 0 <= layer_index < num_layers:
        raise ValueError(f"layer {layer_index} outside [0,{num_layers})")
    dtype = torch.float64
    indices = torch.arange(num_pairs, device=device, dtype=dtype)
    if strategy.kind == "native":
        return torch.ones(num_pairs, device=device, dtype=dtype)
    if strategy.kind == "uniform_scale":
        scale = float(strategy.values["phase_scale"])
        if not 0.0 <= scale <= 1.0:
            raise ValueError("phase_scale must be in [0,1]")
        return torch.full((num_pairs,), scale, device=device, dtype=dtype)
    if strategy.kind == "periodic_nope":
        interval = int(strategy.values["layer_interval"])
        offset = int(strategy.values.get("layer_offset", interval - 1))
        modified = float(strategy.values.get("modified_scale", 0.0))
        if interval <= 0 or not 0 <= offset < interval:
            raise ValueError("periodic_nope requires interval>0 and offset in [0,interval)")
        if not 0.0 <= modified <= 1.0:
            raise ValueError("modified_scale must be in [0,1]")
        scale = modified if layer_index % interval == offset else 1.0
        return torch.full((num_pairs,), scale, device=device, dtype=dtype)
    if strategy.kind == "deep_highfreq_drop":
        deep_fraction = float(strategy.values["deep_layer_fraction"])
        count = int(strategy.values["high_frequency_pairs"])
        modified = float(strategy.values.get("modified_scale", 0.0))
        if not 0.0 < deep_fraction <= 1.0:
            raise ValueError("deep_layer_fraction must be in (0,1]")
        if not 0 <= count <= num_pairs:
            raise ValueError("high_frequency_pairs is outside the rotary dimension")
        first_deep = int(math.ceil(num_layers * (1.0 - deep_fraction)))
        output = torch.ones(num_pairs, device=device, dtype=dtype)
        if layer_index >= first_deep:
            output[:count] = modified
        return output
    if strategy.kind == "band_layer_scale":
        pair_start = int(strategy.values.get("pair_start", 0))
        pair_end = int(strategy.values.get("pair_end_exclusive", -1))
        if pair_end < 0:
            pair_end = num_pairs
        modified = float(strategy.values["modified_scale"])
        if not 0 <= pair_start <= pair_end <= num_pairs:
            raise ValueError(
                f"invalid pair band [{pair_start},{pair_end}) for {num_pairs} pairs"
            )
        if not 0.0 <= modified <= 1.0:
            raise ValueError("modified_scale must be in [0,1]")
        layer_mode = str(strategy.values.get("layer_mode", "hard_deep"))
        if layer_mode == "hard_deep":
            deep_fraction = float(strategy.values["deep_layer_fraction"])
            if not 0.0 < deep_fraction <= 1.0:
                raise ValueError("deep_layer_fraction must be in (0,1]")
            first_deep = int(math.ceil(num_layers * (1.0 - deep_fraction)))
            layer_gate = 1.0 if layer_index >= first_deep else 0.0
        elif layer_mode == "sigmoid_deep":
            center = float(strategy.values["layer_center_fraction"]) * max(1, num_layers - 1)
            temperature = max(
                1e-6, float(strategy.values["layer_temperature_fraction"]) * num_layers
            )
            layer_gate = float(torch.sigmoid(torch.tensor((layer_index - center) / temperature)))
        else:
            raise ValueError(f"unknown band_layer_scale layer_mode {layer_mode!r}")
        output = torch.ones(num_pairs, device=device, dtype=dtype)
        output[pair_start:pair_end] = 1.0 - (1.0 - modified) * layer_gate
        return output
    if strategy.kind in {"smooth_layer_frequency", "smooth_remote_warp"}:
        layer_center = float(strategy.values["layer_center_fraction"]) * max(1, num_layers - 1)
        layer_temperature = max(
            1e-6, float(strategy.values["layer_temperature_fraction"]) * num_layers
        )
        frequency_center = float(strategy.values["frequency_center_pair"])
        frequency_temperature = max(
            1e-6, float(strategy.values["frequency_temperature_pairs"])
        )
        layer_gate = _sigmoid(
            torch.tensor((layer_index - layer_center) / layer_temperature, device=device, dtype=dtype)
        )
        frequency_gate = _sigmoid((frequency_center - indices) / frequency_temperature)
        gate = layer_gate * frequency_gate
        if strategy.kind == "smooth_layer_frequency":
            alpha_min = float(strategy.values["alpha_min"])
            if not 0.0 <= alpha_min <= 1.0:
                raise ValueError("alpha_min must be in [0,1]")
            return 1.0 - (1.0 - alpha_min) * gate
        gate_max = float(strategy.values["gate_max"])
        if not 0.0 <= gate_max <= 1.0:
            raise ValueError("gate_max must be in [0,1]")
        return gate_max * gate
    if strategy.kind == "period_aware_scale":
        if inv_freq is None:
            raise ValueError("period_aware_scale requires the model's live inv_freq")
        live_frequency = inv_freq.to(device=device, dtype=dtype).reshape(-1)
        if live_frequency.numel() != num_pairs:
            raise ValueError("inv_freq length does not match rotary pair count")
        periods = 2.0 * math.pi / live_frequency.abs().clamp_min(1e-30)
        period_center = float(strategy.values["period_center_tokens"])
        log_temperature = max(1e-6, float(strategy.values["period_log_temperature"]))
        layer_center = float(strategy.values["layer_center_fraction"]) * max(1, num_layers - 1)
        layer_temperature = max(
            1e-6, float(strategy.values["layer_temperature_fraction"]) * num_layers
        )
        alpha_min = float(strategy.values["alpha_min"])
        if period_center <= 0 or not 0.0 <= alpha_min <= 1.0:
            raise ValueError("period center must be positive and alpha_min in [0,1]")
        layer_gate = _sigmoid(
            torch.tensor((layer_index - layer_center) / layer_temperature, device=device, dtype=dtype)
        )
        short_period_gate = _sigmoid(
            (math.log(period_center) - periods.log()) / log_temperature
        )
        return 1.0 - (1.0 - alpha_min) * layer_gate * short_period_gate
    if strategy.kind == "phase_diverse_deep":
        start_fraction = float(strategy.values["start_layer_fraction"])
        count = int(strategy.values["high_frequency_pairs"])
        phase_scales = [float(value) for value in strategy.values["phase_scales"]]
        if not 0.0 <= start_fraction < 1.0:
            raise ValueError("start_layer_fraction must be in [0,1)")
        if not 0 <= count <= num_pairs or not phase_scales:
            raise ValueError("invalid high_frequency_pairs or empty phase_scales")
        if any(not 0.0 <= value <= 1.0 for value in phase_scales):
            raise ValueError("every phase-diversity scale must be in [0,1]")
        first_layer = int(math.floor(start_fraction * num_layers))
        output = torch.ones(num_pairs, device=device, dtype=dtype)
        if layer_index >= first_layer:
            scale = phase_scales[(layer_index - first_layer) % len(phase_scales)]
            output[:count] = scale
        return output
    if strategy.kind == "layer_pair_matrix":
        matrix = strategy.values.get("layer_pair_scales")
        if not isinstance(matrix, list) or len(matrix) != num_layers:
            raise ValueError(
                "layer_pair_matrix requires one layer_pair_scales row per model layer"
            )
        row = matrix[layer_index]
        if not isinstance(row, list) or len(row) != num_pairs:
            raise ValueError(
                f"layer_pair_matrix row {layer_index} has {len(row) if isinstance(row, list) else 'invalid'} "
                f"entries; expected {num_pairs}"
            )
        output = torch.tensor(row, device=device, dtype=dtype)
        if not torch.isfinite(output).all():
            raise ValueError("layer_pair_matrix contains a non-finite scale")
        if bool(((output < 0.0) | (output > 1.0)).any()):
            raise ValueError("layer_pair_matrix scales must be in [0,1]")
        return output
    raise AssertionError(strategy.kind)


def effective_positions(
    positions: torch.Tensor,
    strategy: Strategy,
    layer_index: int,
    num_layers: int,
    num_pairs: int,
    inv_freq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a [sequence, rotary_pairs] matrix in float64."""
    flat = positions.reshape(-1).to(dtype=torch.float64)
    if strategy.kind == "smooth_remote_warp":
        gate = pair_scales(
            strategy, layer_index, num_layers, num_pairs, flat.device, inv_freq=inv_freq
        )
        window = float(strategy.values["local_window"])
        tau = float(strategy.values["remote_tau"])
        if window < 0.0 or tau <= 0.0:
            raise ValueError("local_window must be nonnegative and remote_tau positive")
        compressed = torch.where(
            flat <= window,
            flat,
            window + tau * torch.log1p((flat - window) / tau),
        )
        return flat[:, None] - gate[None, :] * (flat - compressed)[:, None]
    scales = pair_scales(
        strategy, layer_index, num_layers, num_pairs, flat.device, inv_freq=inv_freq
    )
    return flat[:, None] * scales[None, :]


def _rotary_module(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "rotary_emb", None),
        getattr(model, "rotary_emb", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "inv_freq"):
            return candidate
    raise RuntimeError("Qwen3 rotary_emb.inv_freq was not found")


def attention_layers(model: Any) -> list[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("expected model.model.layers for Qwen3")
    attentions = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError(f"layer {index} has no self_attn")
        attentions.append(attention)
    return attentions


def _attention_scaling(rotary: Any, device: torch.device) -> torch.Tensor:
    value = getattr(rotary, "attention_scaling", 1.0)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def build_position_embeddings(
    model: Any,
    strategy: Strategy,
    layer_index: int,
    hidden_states: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_cos, _ = reference
    sequence_length = int(hidden_states.shape[1])
    if cache_position is None:
        positions = torch.arange(sequence_length, device=hidden_states.device)
    else:
        positions = cache_position.to(hidden_states.device).reshape(-1)
        if positions.numel() != sequence_length:
            positions = positions[-sequence_length:]
    rotary = _rotary_module(model)
    inv_freq = rotary.inv_freq.to(device=hidden_states.device, dtype=torch.float64).reshape(-1)
    num_pairs = int(inv_freq.numel())
    cached_layers = getattr(model, "_pe_num_layers", None)
    num_layers = int(cached_layers if cached_layers is not None else len(attention_layers(model)))
    effective = effective_positions(
        positions, strategy, layer_index, num_layers, num_pairs, inv_freq=inv_freq
    )
    angles = effective * inv_freq[None, :]
    if not torch.isfinite(angles).all():
        raise FloatingPointError("non-finite custom RoPE phase")
    doubled = torch.cat((angles, angles), dim=-1).to(torch.float32)
    scaling = _attention_scaling(rotary, hidden_states.device)
    cos = (doubled.cos() * scaling).to(dtype=reference_cos.dtype)
    sin = (doubled.sin() * scaling).to(dtype=reference_cos.dtype)
    cos = cos.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
    sin = sin.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
    if cos.shape != reference_cos.shape:
        raise RuntimeError(
            f"custom RoPE shape {tuple(cos.shape)} != reference {tuple(reference_cos.shape)}"
        )
    return cos, sin


def patch_model(model: Any, strategy: Strategy) -> int:
    """Patch only layer-specific position_embeddings; return patched layer count."""
    if strategy.kind == "native":
        return 0
    attentions = attention_layers(model)
    model._pe_num_layers = len(attentions)
    for layer_index, attention in enumerate(attentions):
        if hasattr(attention, "_pe_original_forward"):
            raise RuntimeError("model attention was already patched")
        original = attention.forward
        attention._pe_original_forward = original

        def wrapped_forward(
            this: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            _original: Any = original,
            _layer_index: int = layer_index,
            **kwargs: Any,
        ) -> Any:
            custom = build_position_embeddings(
                model,
                strategy,
                _layer_index,
                hidden_states,
                position_embeddings,
                cache_position,
            )
            return _original(
                hidden_states=hidden_states,
                position_embeddings=custom,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        attention.forward = types.MethodType(wrapped_forward, attention)
    return len(attentions)


def strategy_profile(model: Any, strategy: Strategy) -> dict[str, Any]:
    rotary = _rotary_module(model)
    inv_freq = rotary.inv_freq.detach().float().cpu().reshape(-1)
    layers = attention_layers(model)
    scales = [
        pair_scales(
            strategy, index, len(layers), len(inv_freq), inv_freq=inv_freq
        ).tolist()
        for index in range(len(layers))
    ]
    sample_positions = torch.tensor([0, 512, 2048, 4096, 8192, 16384], dtype=torch.float64)
    samples = {
        str(layer): effective_positions(
            sample_positions,
            strategy,
            layer,
            len(layers),
            len(inv_freq),
            inv_freq=inv_freq,
        )[:, : min(16, len(inv_freq))].tolist()
        for layer in sorted({0, len(layers) // 2, len(layers) - 1})
    }
    return {
        "strategy": strategy.values,
        "num_layers": len(layers),
        "rotary_pairs": len(inv_freq),
        "native_inv_freq": inv_freq.tolist(),
        "native_period_tokens": (2.0 * math.pi / inv_freq.double()).tolist(),
        "pair_scales_or_gates": scales,
        "sample_positions": sample_positions.tolist(),
        "sample_effective_positions_first_pairs": samples,
    }


def save_strategy_profile(model: Any, strategy: Strategy, path: Path) -> None:
    write_json(path, strategy_profile(model, strategy))
