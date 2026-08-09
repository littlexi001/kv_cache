from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def model_geometry(model_root: Path) -> tuple[int, torch.Tensor, dict[str, Any]]:
    """Read the standard Qwen rotary geometry without loading model weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_root), trust_remote_code=True)
    num_layers = int(config.num_hidden_layers)
    head_dim = int(
        getattr(
            config,
            "head_dim",
            int(config.hidden_size) // int(config.num_attention_heads),
        )
    )
    partial = float(getattr(config, "partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"invalid rotary dimension {rotary_dim}")
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", "default")))
        if rope_type not in {"default", "none"}:
            raise ValueError(
                "phase-profile generation currently requires native/default RoPE; "
                f"found rope_scaling={rope_scaling}"
            )
    base = float(getattr(config, "rope_theta", 10000.0))
    indices = torch.arange(0, rotary_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (indices / rotary_dim))
    metadata = {
        "num_hidden_layers": num_layers,
        "head_dim": head_dim,
        "partial_rotary_factor": partial,
        "rotary_dim": rotary_dim,
        "rope_theta": base,
        "rotary_pairs": int(inv_freq.numel()),
    }
    return num_layers, inv_freq, metadata


def bounded_scales(raw: torch.Tensor, alpha_min: float, alpha_max: float) -> torch.Tensor:
    return alpha_min + (alpha_max - alpha_min) * torch.sigmoid(raw)


def inverse_bounded_scale(
    value: torch.Tensor, alpha_min: float, alpha_max: float
) -> torch.Tensor:
    unit = ((value - alpha_min) / (alpha_max - alpha_min)).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(unit)


def coverage_and_local_penalty(
    scales: torch.Tensor,
    inv_freq: torch.Tensor,
    remote_distances: torch.Tensor,
    phase_probes: torch.Tensor,
    local_distances: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [deep_layer, pair, distance, content_phase]
    phase = (
        scales[:, :, None, None]
        * inv_freq[None, :, None, None]
        * remote_distances[None, None, :, None]
        - phase_probes[None, None, None, :]
    )
    response = torch.cos(phase)
    # A normalized smooth maximum estimates the best phase response available
    # among the deep layers without rewarding a method merely for adding layers.
    coverage = temperature * torch.logsumexp(response / temperature, dim=0)
    coverage = coverage - temperature * math.log(scales.shape[0])

    local_error = (
        (scales[:, :, None] - 1.0)
        * inv_freq[None, :, None]
        * local_distances[None, None, :]
    )
    local_penalty = (1.0 - torch.cos(local_error)).mean()
    return coverage.mean(), local_penalty


def fixed_profile_metrics(
    scales: torch.Tensor,
    inv_freq: torch.Tensor,
    remote_distances: torch.Tensor,
    phase_probes: torch.Tensor,
    local_distances: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    with torch.no_grad():
        coverage, local_penalty = coverage_and_local_penalty(
            scales,
            inv_freq,
            remote_distances,
            phase_probes,
            local_distances,
            temperature,
        )
    return {
        "phase_coverage": float(coverage),
        "local_phase_penalty": float(local_penalty),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_mean": float(scales.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-layer-fraction", type=float, default=0.5)
    parser.add_argument("--frequency-pairs", type=int, default=16)
    parser.add_argument("--alpha-min", type=float, default=0.25)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--remote-min", type=float, default=8192.0)
    parser.add_argument("--remote-max", type=float, default=131072.0)
    parser.add_argument("--remote-points", type=int, default=33)
    parser.add_argument("--content-phase-probes", type=int, default=32)
    parser.add_argument("--local-max", type=float, default=2048.0)
    parser.add_argument("--local-points", type=int, default=17)
    parser.add_argument("--local-preservation-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    if not 0.0 <= args.start_layer_fraction < 1.0:
        raise ValueError("start-layer-fraction must be in [0,1)")
    if not 0.0 <= args.alpha_min < args.alpha_max <= 1.0:
        raise ValueError("require 0 <= alpha-min < alpha-max <= 1")
    if args.remote_min <= 0 or args.remote_max <= args.remote_min:
        raise ValueError("remote distance range is invalid")
    if args.local_max <= 0 or args.local_preservation_weight < 0:
        raise ValueError("local-max must be positive and local weight nonnegative")
    if args.steps <= 0 or args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("steps, learning-rate, and temperature must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    num_layers, all_inv_freq, geometry = model_geometry(args.model_root)
    if not 1 <= args.frequency_pairs <= int(all_inv_freq.numel()):
        raise ValueError(
            f"frequency-pairs must be in [1,{all_inv_freq.numel()}]"
        )
    first_layer = int(math.floor(args.start_layer_fraction * num_layers))
    deep_layers = num_layers - first_layer
    if deep_layers < 2:
        raise ValueError("phase complementarity requires at least two modified layers")

    inv_freq = all_inv_freq[: args.frequency_pairs]
    remote_distances = torch.logspace(
        math.log10(args.remote_min),
        math.log10(args.remote_max),
        args.remote_points,
        dtype=torch.float64,
    )
    phase_probes = torch.arange(args.content_phase_probes, dtype=torch.float64)
    phase_probes = phase_probes * (2.0 * math.pi / args.content_phase_probes)
    local_distances = torch.logspace(
        0.0,
        math.log10(args.local_max),
        args.local_points,
        dtype=torch.float64,
    )

    cycle = torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float64)
    initial = cycle[torch.arange(deep_layers) % len(cycle), None]
    initial = initial.expand(deep_layers, args.frequency_pairs).clone()
    raw = torch.nn.Parameter(
        inverse_bounded_scale(initial, args.alpha_min, args.alpha_max)
    )
    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
    trace: list[dict[str, float | int]] = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scales = bounded_scales(raw, args.alpha_min, args.alpha_max)
        coverage, local_penalty = coverage_and_local_penalty(
            scales,
            inv_freq,
            remote_distances,
            phase_probes,
            local_distances,
            args.temperature,
        )
        loss = -coverage + args.local_preservation_weight * local_penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite objective at step {step}")
        if step % 50 == 0 or step == args.steps:
            trace.append(
                {
                    "step": step,
                    "objective": float(loss.detach()),
                    "phase_coverage": float(coverage.detach()),
                    "local_phase_penalty": float(local_penalty.detach()),
                }
            )
        if step == args.steps:
            break
        loss.backward()
        optimizer.step()

    optimized = bounded_scales(raw.detach(), args.alpha_min, args.alpha_max)
    native = torch.ones_like(optimized)
    fixed_cycle = initial
    metrics = {
        "native": fixed_profile_metrics(
            native,
            inv_freq,
            remote_distances,
            phase_probes,
            local_distances,
            args.temperature,
        ),
        "fixed_cycle": fixed_profile_metrics(
            fixed_cycle,
            inv_freq,
            remote_distances,
            phase_probes,
            local_distances,
            args.temperature,
        ),
        "optimized": fixed_profile_metrics(
            optimized,
            inv_freq,
            remote_distances,
            phase_probes,
            local_distances,
            args.temperature,
        ),
    }
    if metrics["optimized"]["phase_coverage"] <= metrics["native"]["phase_coverage"]:
        raise RuntimeError("optimization did not improve the declared phase-coverage objective")

    matrix = torch.ones(
        (num_layers, int(all_inv_freq.numel())), dtype=torch.float64
    )
    matrix[first_layer:, : args.frequency_pairs] = optimized
    payload = {
        "name": args.name,
        "kind": "layer_pair_matrix",
        "description": (
            "Offline-optimized deep-layer phase-complementary RoPE profile"
            + (
                " with an explicit local-phase preservation penalty."
                if args.local_preservation_weight > 0
                else "."
            )
        ),
        "layer_pair_scales": matrix.tolist(),
        "optimization": {
            "algorithm_version": 1,
            "seed": args.seed,
            "geometry": geometry,
            "first_modified_layer": first_layer,
            "modified_frequency_pairs": args.frequency_pairs,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "remote_distance_tokens": [args.remote_min, args.remote_max],
            "remote_points": args.remote_points,
            "content_phase_probes": args.content_phase_probes,
            "local_max_tokens": args.local_max,
            "local_points": args.local_points,
            "local_preservation_weight": args.local_preservation_weight,
            "temperature": args.temperature,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "metrics": metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    debug_path = args.output.with_suffix(".optimization.json")
    debug_path.write_text(
        json.dumps(
            {
                "strategy": args.name,
                "metrics": metrics,
                "trace": trace,
                "optimized_modified_scales": optimized.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "strategy": args.name,
                "output": str(args.output),
                "debug": str(debug_path),
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
