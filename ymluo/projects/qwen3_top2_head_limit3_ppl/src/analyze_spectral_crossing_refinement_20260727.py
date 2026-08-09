from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    quantize_groupwise,
    query_int8,
)


HEAD_DIM = 128
CORE_CODE_BITS = 16 * 8 + 32 * 4
TAIL_PLANE_BITS = 80
CORE_SCALE_BITS = 3 * 16
TAIL_SCALE_BITS = 5 * 16
FULL_KV_BITS = 2 * HEAD_DIM * 16


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one value")
    return result


def nested_tail_reconstructions(
    values: torch.Tensor,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-1] % group_size != 0:
        raise ValueError("tail must contain complete groups")
    grouped = values.float().reshape(
        *values.shape[:-1], values.shape[-1] // group_size, group_size
    )
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 3.0
    sign = torch.where(grouped >= 0.0, 1.0, -1.0)
    first_plane = sign * (2.0 * scale)
    magnitude = torch.where(grouped.abs() >= 2.0 * scale, 3.0, 1.0)
    second_plane = sign * magnitude * scale
    return first_plane.reshape_as(values), second_plane.reshape_as(values)


def systematic_sample_indices(
    count: int,
    sample_count: int,
    phase: int,
    device: torch.device,
) -> torch.Tensor:
    actual = min(count, sample_count)
    stride = max(1, count // actual)
    while math.gcd(stride, count) != 1:
        stride += 1
    offsets = torch.arange(actual, device=device)
    return (phase + offsets * stride) % count


def conformal_quantile(values: torch.Tensor, alpha: float) -> torch.Tensor:
    count = values.numel()
    rank = min(count, max(1, math.ceil((count + 1) * (1.0 - alpha))))
    return torch.kthvalue(values.flatten(), rank).values


def interval_candidates(
    approximate: torch.Tensor,
    radius: torch.Tensor,
    top_count: int,
    domain: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    if domain is None:
        domain = torch.ones_like(approximate, dtype=torch.bool)
    domain_indices = torch.nonzero(domain, as_tuple=False).flatten()
    if domain_indices.numel() < top_count:
        raise RuntimeError("interval domain cannot contain the requested top-k")
    lower = approximate[domain_indices] - radius[domain_indices]
    threshold = float(torch.topk(lower, k=top_count).values[-1].item())
    selected = domain & ((approximate + radius) >= threshold)
    return selected, threshold


def candidate_metrics(
    candidate: torch.Tensor,
    true_top: torch.Tensor,
    attention: torch.Tensor,
) -> dict[str, float]:
    recall = float(candidate[true_top].float().mean().item())
    mass = float(attention[candidate].sum().item())
    return {
        "candidate_count": int(candidate.sum().item()),
        "candidate_ratio": float(candidate.float().mean().item()),
        "top2_recall": recall,
        "attention_mass": mass,
    }


def error_denominator(
    exact_query: torch.Tensor,
    approximate_query: torch.Tensor,
    exact_key: torch.Tensor,
    approximate_key: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    residual_norm = torch.linalg.vector_norm(
        exact_key - approximate_key, dim=-1
    )
    approximate_norm = torch.linalg.vector_norm(approximate_key, dim=-1)
    query_norm = torch.linalg.vector_norm(exact_query)
    query_error_norm = torch.linalg.vector_norm(
        exact_query - approximate_query
    )
    return (
        query_norm * residual_norm
        + query_error_norm * approximate_norm
    ) * scaling


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.5).item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "maximum": float(tensor.max().item()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "candidate_ratio",
        "top2_recall",
        "attention_mass",
        "stage0_ratio",
        "stage1_ratio",
        "calibration_multiplier",
        "access_code_bits",
        "access_metadata_bits",
        "access_total_bits",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "cases": len(items),
            "storage_code_bits": float(items[0]["storage_code_bits"]),
            "storage_metadata_bits": float(items[0]["storage_metadata_bits"]),
            "storage_total_bits": float(items[0]["storage_total_bits"]),
            "storage_ratio_of_full_kv": float(items[0]["storage_total_bits"])
            / FULL_KV_BITS,
        }
        for metric in metrics:
            stats = summarize(float(item[metric]) for item in items)
            result.update(
                {f"{metric}_{name}": value for name, value in stats.items()}
            )
        result["access_ratio_of_full_kv_mean"] = (
            result["access_total_bits_mean"] / FULL_KV_BITS
        )
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure deterministic and sample-calibrated top-k crossing "
            "intervals for nested spectral bit-plane refinement."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_samples", type=int, default=256)
    parser.add_argument("--alphas", default="0.01,0.005,0.001")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    alphas = parse_floats(args.alphas)
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError("alphas must be in (0, 1)")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    states: dict[int, dict[str, Any]] = {}

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        raw_key = record.get("key")
        if layer not in states:
            if raw_key is None:
                raise RuntimeError(f"layer {layer} is missing its initial key")
            all_key = raw_key.to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            prepared: list[dict[str, torch.Tensor]] = []
            for kv_head in range(key.shape[0]):
                head_key = key[kv_head]
                basis, _ = covariance_basis(head_key[:: args.sample_stride])
                coefficients = head_key @ basis
                core = torch.cat(
                    (
                        quantize_groupwise(coefficients[:, :16], 8),
                        quantize_groupwise(coefficients[:, 16:48], 4),
                        torch.zeros_like(coefficients[:, 48:]),
                    ),
                    dim=-1,
                )
                tail1, tail2 = nested_tail_reconstructions(
                    coefficients[:, 48:]
                )
                stage1 = core.clone()
                stage1[:, 48:] = tail1
                stage2 = core.clone()
                stage2[:, 48:] = tail2
                prepared.append(
                    {
                        "head_key": head_key,
                        "basis": basis,
                        "coefficients": coefficients,
                        "core": core,
                        "stage1": stage1,
                        "stage2": stage2,
                    }
                )
            states[layer] = {
                "history_count": history_count,
                "kv_heads": int(key.shape[0]),
                "prepared": prepared,
            }

        state = states[layer]
        history_count = int(state["history_count"])
        kv_heads = int(state["kv_heads"])
        query = record["query"].to(device).float()[0, :, 0, :]
        query_heads = int(query.shape[0])
        groups = query_heads // kv_heads
        scaling = float(record["scaling"])
        top_count = max(1, math.ceil(args.top_fraction * history_count))

        for kv_head in range(kv_heads):
            prepared = state["prepared"][kv_head]
            for group in range(groups):
                query_head = kv_head * groups + group
                exact_query = query[query_head] @ prepared["basis"]
                approximate_query = query_int8(exact_query)
                exact_scores = (
                    prepared["coefficients"] @ exact_query
                ) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                true_top = torch.topk(exact_scores, k=top_count).indices
                phase = (
                    record_index * 1009
                    + layer * 131
                    + kv_head * 31
                    + query_head * 17
                ) % history_count
                sample = systematic_sample_indices(
                    history_count,
                    args.calibration_samples,
                    phase,
                    device,
                )

                stage_payloads: list[dict[str, torch.Tensor]] = []
                for name in ("core", "stage1", "stage2"):
                    reconstructed = prepared[name]
                    approximate_scores = (
                        reconstructed @ approximate_query
                    ) * scaling
                    denominator = error_denominator(
                        exact_query,
                        approximate_query,
                        prepared["coefficients"],
                        reconstructed,
                        scaling,
                    ).clamp_min(1.0e-12)
                    normalized_error = (
                        exact_scores - approximate_scores
                    ).abs() / denominator
                    stage_payloads.append(
                        {
                            "scores": approximate_scores,
                            "denominator": denominator,
                            "normalized_error": normalized_error,
                        }
                    )

                for stage_index, stage in enumerate(stage_payloads):
                    deterministic, _ = interval_candidates(
                        stage["scores"],
                        stage["denominator"],
                        top_count,
                    )
                    deterministic_metrics = candidate_metrics(
                        deterministic, true_top, attention
                    )
                    code_bits = (
                        CORE_CODE_BITS
                        + (TAIL_PLANE_BITS if stage_index >= 1 else 0)
                        + (TAIL_PLANE_BITS if stage_index >= 2 else 0)
                    )
                    metadata_bits = (
                        CORE_SCALE_BITS
                        + (TAIL_SCALE_BITS if stage_index >= 1 else 0)
                    )
                    rows.append(
                        {
                            "label": args.label,
                            "record": record_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "method": f"stage{stage_index}_deterministic",
                            "stage0_ratio": (
                                deterministic_metrics["candidate_ratio"]
                                if stage_index == 0
                                else 1.0
                            ),
                            "stage1_ratio": (
                                deterministic_metrics["candidate_ratio"]
                                if stage_index == 1
                                else 1.0
                            ),
                            "calibration_multiplier": 1.0,
                            "storage_code_bits": code_bits,
                            "storage_metadata_bits": metadata_bits,
                            "storage_total_bits": code_bits + metadata_bits,
                            "access_code_bits": code_bits,
                            "access_metadata_bits": metadata_bits,
                            "access_total_bits": code_bits + metadata_bits,
                            **deterministic_metrics,
                        }
                    )

                for alpha in alphas:
                    candidates: list[torch.Tensor] = []
                    multipliers: list[float] = []
                    domain: torch.Tensor | None = None
                    for stage in stage_payloads:
                        multiplier = conformal_quantile(
                            stage["normalized_error"][sample], alpha
                        )
                        candidate, _ = interval_candidates(
                            stage["scores"],
                            multiplier * stage["denominator"],
                            top_count,
                            domain,
                        )
                        candidates.append(candidate)
                        multipliers.append(float(multiplier.item()))
                        domain = candidate

                    for stage_index, candidate in enumerate(candidates):
                        metrics = candidate_metrics(
                            candidate, true_top, attention
                        )
                        code_bits = (
                            CORE_CODE_BITS
                            + (TAIL_PLANE_BITS if stage_index >= 1 else 0)
                            + (TAIL_PLANE_BITS if stage_index >= 2 else 0)
                        )
                        metadata_bits = (
                            CORE_SCALE_BITS
                            + (TAIL_SCALE_BITS if stage_index >= 1 else 0)
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "record": record_index,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "method": f"stage{stage_index}_conformal_a{alpha:g}",
                                "stage0_ratio": candidates[0]
                                .float()
                                .mean()
                                .item(),
                                "stage1_ratio": (
                                    candidates[1].float().mean().item()
                                    if stage_index >= 1
                                    else 1.0
                                ),
                                "calibration_multiplier": multipliers[
                                    stage_index
                                ],
                                "storage_code_bits": code_bits,
                                "storage_metadata_bits": metadata_bits,
                                "storage_total_bits": code_bits + metadata_bits,
                                "access_code_bits": code_bits,
                                "access_metadata_bits": metadata_bits,
                                "access_total_bits": code_bits + metadata_bits,
                                **metrics,
                            }
                        )

                    progressive_metrics = candidate_metrics(
                        candidates[2], true_top, attention
                    )
                    stage0_ratio = float(
                        candidates[0].float().mean().item()
                    )
                    stage1_ratio = float(
                        candidates[1].float().mean().item()
                    )
                    access_code_bits = (
                        CORE_CODE_BITS
                        + TAIL_PLANE_BITS * stage0_ratio
                        + TAIL_PLANE_BITS * stage1_ratio
                    )
                    access_metadata_bits = (
                        CORE_SCALE_BITS + TAIL_SCALE_BITS * stage0_ratio
                    )
                    rows.append(
                        {
                            "label": args.label,
                            "record": record_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "method": f"progressive_conformal_a{alpha:g}",
                            "stage0_ratio": stage0_ratio,
                            "stage1_ratio": stage1_ratio,
                            "calibration_multiplier": max(multipliers),
                            "storage_code_bits": CORE_CODE_BITS
                            + 2 * TAIL_PLANE_BITS,
                            "storage_metadata_bits": CORE_SCALE_BITS
                            + TAIL_SCALE_BITS,
                            "storage_total_bits": CORE_CODE_BITS
                            + 2 * TAIL_PLANE_BITS
                            + CORE_SCALE_BITS
                            + TAIL_SCALE_BITS,
                            "access_code_bits": access_code_bits,
                            "access_metadata_bits": access_metadata_bits,
                            "access_total_bits": access_code_bits
                            + access_metadata_bits,
                            **progressive_metrics,
                        }
                    )

        print(
            json.dumps(
                {
                    "label": args.label,
                    "record": record_index + 1,
                    "records": len(records),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "top_fraction": args.top_fraction,
            "calibration_samples": args.calibration_samples,
            "alphas": alphas,
            "code_bit_accounting": {
                "core": CORE_CODE_BITS,
                "tail_plane": TAIL_PLANE_BITS,
            },
            "metadata_bit_accounting": {
                "core_fp16_group_scales": CORE_SCALE_BITS,
                "tail_fp16_group_scales": TAIL_SCALE_BITS,
            },
        },
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
