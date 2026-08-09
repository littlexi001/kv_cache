from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build causal short-window averages and native/probe interpolations for a "
            "generated-Q trajectory. The transformed profiles quantify the stability-"
            "selectivity Pareto frontier without changing the K index."
        )
    )
    parser.add_argument("--input_profile", required=True)
    parser.add_argument("--output_profile", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--windows", default="2,4,8")
    parser.add_argument("--native_weights", default="0.25,0.5,0.75")
    parser.add_argument(
        "--probe_field",
        default="evidence_probe_q",
        choices=["state_only_probe_q", "evidence_probe_q"],
    )
    return parser.parse_args()


def normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1).to(torch.float16)


def causal_window_mean(
    source: torch.Tensor,
    mask: torch.Tensor,
    window: int,
) -> torch.Tensor:
    output = torch.zeros_like(source)
    for trajectory_index in range(source.shape[0]):
        count = int(mask[trajectory_index].sum().item())
        for end in range(count):
            start = max(0, end + 1 - window)
            output[trajectory_index, end].copy_(
                normalize(source[trajectory_index, start : end + 1].float().mean(dim=0))
            )
    return output


def valid_norm_summary(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float | bool]:
    valid = value[mask].float()
    return {
        "finite": bool(torch.isfinite(valid).all()),
        "norm_mean": float(valid.norm(dim=-1).mean().item()),
        "norm_min": float(valid.norm(dim=-1).min().item()),
        "norm_max": float(valid.norm(dim=-1).max().item()),
    }


def main() -> None:
    args = parse_args()
    windows = sorted({int(item) for item in args.windows.split(",") if item.strip()})
    native_weights = sorted(
        {float(item) for item in args.native_weights.split(",") if item.strip()}
    )
    if any(window <= 0 for window in windows):
        raise ValueError("windows must be positive")
    if any(weight <= 0.0 or weight >= 1.0 for weight in native_weights):
        raise ValueError("native_weights must be strictly between zero and one")

    source: dict[str, Any] = torch.load(
        args.input_profile, map_location="cpu", weights_only=False
    )
    native = source.get("native_generation_q", source["svd_q"])
    if args.probe_field not in source:
        raise KeyError(f"input profile has no field {args.probe_field!r}")
    probe = source[args.probe_field]
    mask = source["mask"].bool()
    output = dict(source)
    summaries: dict[str, Any] = {}

    for window in windows:
        field = f"native_ema_w{window}"
        output[field] = causal_window_mean(native, mask, window)
        summaries[field] = valid_norm_summary(output[field], mask)

    for native_weight in native_weights:
        percent = int(round(native_weight * 100))
        field = f"native_{args.probe_field}_mix_n{percent}"
        mixed = torch.zeros_like(native)
        mixed[mask] = normalize(
            native_weight * native[mask].float()
            + (1.0 - native_weight) * probe[mask].float()
        )
        output[field] = mixed
        summaries[field] = valid_norm_summary(mixed, mask)

    output["trajectory_transform"] = {
        "source_profile": str(args.input_profile),
        "causal_windows": windows,
        "probe_field": args.probe_field,
        "native_weights": native_weights,
        "normalization": "per-layer per-head L2 normalization in frozen K-SVD space",
    }
    output_path = Path(args.output_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "source": "causal smoothing and native/probe Q interpolation",
        "input_profile": str(args.input_profile),
        "output_profile": str(output_path),
        "contains_synthetic_vectors": False,
        "trajectories": len(source["trajectories"]),
        "states": int(mask.sum().item()),
        "fields": summaries,
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
