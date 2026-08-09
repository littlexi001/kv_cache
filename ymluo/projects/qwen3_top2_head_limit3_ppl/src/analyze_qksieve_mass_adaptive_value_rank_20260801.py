#!/usr/bin/env python
"""Audit a training-free, mass-adaptive Value-sketch rank policy.

The policy uses only request-local numerical quantities available at runtime:
the sampled proxy mass and the per-head reconstruction residual of each Value
rank.  It selects the smallest rank whose estimated tail error certificate is
below a fixed tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_value_sketch_residual_20260801 import (
    block_affine_quantize,
    output_metrics,
    residual_output,
    value_basis,
)


RANKS = (8, 16, 32)
SNR_GATES = (0.25, 0.5, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument("--value_scale_block", type=int, default=256)
    parser.add_argument(
        "--tolerances",
        default="0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.06",
    )
    return parser.parse_args()


def target_count(history_count: int) -> int:
    return min(history_count, max(256, min(math.ceil(0.06 * history_count), 1280)))


def sampled_selection(
    scores: torch.Tensor,
    *,
    row: int,
    sample_count: int,
    selected_fraction: float,
) -> tuple[torch.Tensor, float, float]:
    """Match the deployment kernel's stratified quantile and mass estimate."""
    history_count = int(scores.numel())
    sample_count = min(sample_count, history_count)
    segment = max(1, history_count // sample_count)
    phase = (row * 131 + 17) % segment
    samples = torch.arange(sample_count, device=scores.device, dtype=torch.long)
    centered = ((2 * samples + 1) * history_count) // (2 * sample_count)
    sampled_indices = (centered + phase) % history_count
    sampled_scores = scores.index_select(0, sampled_indices)
    selected_keep = max(
        1,
        min(sample_count, int(round(selected_fraction * (sample_count + 1)))),
    )
    top_values = torch.topk(sampled_scores, selected_keep).values
    threshold = top_values.min()
    indices = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    sample_mass = float(
        torch.softmax(sampled_scores.float(), dim=0)
        .gather(0, torch.topk(sampled_scores, selected_keep).indices)
        .sum()
    )
    return indices, sample_mass, float(threshold)


def correlation(left: list[float], right: list[float]) -> float:
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1.0e-20))


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.value_bits != 4:
        raise ValueError("this deployment audit currently targets INT4")
    traces = tuple(Path(item) for item in args.traces.split(",") if item.strip())
    tolerances = tuple(float(item) for item in args.tolerances.split(","))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    for trace in traces:
        payload = torch.load(trace, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            layer = int(record["layer"])
            query = record["query"].to(device).float()[0, :, 0, :]
            key = record["key"].to(device).float()[0]
            value = record["value"].to(device).float()[0]
            scaling = float(record["scaling"])
            kv_heads, history_count, _ = key.shape
            groups = query.shape[0] // kv_heads
            keep = target_count(history_count)
            selected_fraction = keep / history_count

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                calibration = query[kv_head * groups : (kv_head + 1) * groups]
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.key_sample_stride],
                    calibration,
                    args.query_shrinkage,
                )
                key_coefficients = head_key @ key_factor
                projected_calibration = calibration @ query_factor
                bands = quantized_bands(key_coefficients, projected_calibration)
                key_distortion, _ = distortion_table(
                    key_coefficients,
                    projected_calibration,
                    ZERO_BIT_LEVELS,
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.key_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                key_reconstruction = reconstruct(bands, allocation)

                mean, basis, coefficients, explained = value_basis(
                    head_value,
                    sample_stride=args.value_sample_stride,
                    maximum_rank=max(RANKS),
                )
                reconstructed: dict[int, torch.Tensor] = {}
                residual_ratio: dict[int, float] = {}
                value_rms = head_value.float().square().mean().sqrt().clamp_min(1.0e-12)
                for rank in RANKS:
                    quantized = block_affine_quantize(
                        coefficients[:, :rank],
                        bits=args.value_bits,
                        block_size=args.value_scale_block,
                    )
                    approximation = mean + quantized @ basis[:, :rank].T
                    reconstructed[rank] = approximation
                    residual_ratio[rank] = float(
                        (head_value.float() - approximation).square().mean().sqrt()
                        / value_rms
                    )

                for group in range(groups):
                    query_head = kv_head * groups + group
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (head_key @ query[query_head]) * scaling
                    proxy_scores = (
                        key_reconstruction.float() @ approximate_query.float()
                    ) * scaling
                    indices, sample_mass, threshold = sampled_selection(
                        proxy_scores,
                        row=query_head,
                        sample_count=args.sample_count,
                        selected_fraction=selected_fraction,
                    )
                    exact_probability = torch.softmax(exact_scores.float(), dim=0)
                    proxy_probability = torch.softmax(proxy_scores.float(), dim=0)
                    exact_mass = float(exact_probability.index_select(0, indices).sum())
                    proxy_mass = float(proxy_probability.index_select(0, indices).sum())
                    full_output = torch.sum(
                        exact_probability.unsqueeze(-1) * head_value.float(), dim=0
                    )
                    selected_scores = exact_scores.index_select(0, indices)
                    selected_probability = torch.softmax(
                        selected_scores.float(), dim=0
                    )
                    sparse_output = torch.sum(
                        selected_probability.unsqueeze(-1)
                        * head_value.float().index_select(0, indices),
                        dim=0,
                    )
                    sparse_metrics = output_metrics(sparse_output, full_output)
                    fixed_metrics: dict[int, dict[str, float]] = {}
                    fixed_outputs: dict[int, torch.Tensor] = {}
                    shrink_metrics: dict[int, dict[str, float]] = {}
                    correction_ratio: dict[int, float] = {}
                    correction_snr: dict[int, float] = {}
                    shrink_alpha: dict[int, float] = {}
                    full_norm = full_output.norm().clamp_min(1.0e-12)
                    for rank in RANKS:
                        output, _ = residual_output(
                            exact_scores,
                            proxy_scores,
                            head_value,
                            reconstructed[rank],
                            indices,
                            shared_normalizer=True,
                        )
                        fixed_outputs[rank] = output
                        fixed_metrics[rank] = output_metrics(output, full_output)
                        correction = output - sparse_output
                        correction_ratio[rank] = float(correction.norm() / full_norm)
                        certificate = (1.0 - proxy_mass) * residual_ratio[rank]
                        correction_snr[rank] = correction_ratio[rank] / max(
                            certificate, 1.0e-12
                        )
                        shrink_alpha[rank] = max(
                            0.0,
                            min(
                                1.0,
                                1.0
                                - certificate
                                / max(correction_ratio[rank], 1.0e-12),
                            ),
                        )
                        shrink_output = sparse_output + shrink_alpha[rank] * correction
                        shrink_metrics[rank] = output_metrics(
                            shrink_output, full_output
                        )

                    base = {
                        "trace": trace.stem,
                        "layer": layer,
                        "kv_head": kv_head,
                        "query_head": query_head,
                        "history_count": history_count,
                        "target_count": keep,
                        "candidate_count": int(indices.numel()),
                        "sample_mass": sample_mass,
                        "proxy_mass": proxy_mass,
                        "exact_mass": exact_mass,
                        "threshold": threshold,
                        "sparse_relative_l2": sparse_metrics["relative_l2"],
                        "sparse_cosine": sparse_metrics["cosine"],
                    }
                    for rank in RANKS:
                        base[f"rank{rank}_residual_ratio"] = residual_ratio[rank]
                        base[f"rank{rank}_explained"] = float(explained[:rank].sum())
                        base[f"rank{rank}_relative_l2"] = fixed_metrics[rank]["relative_l2"]
                        base[f"rank{rank}_cosine"] = fixed_metrics[rank]["cosine"]
                        base[f"rank{rank}_correction_ratio"] = correction_ratio[rank]
                        base[f"rank{rank}_correction_snr"] = correction_snr[rank]
                        base[f"rank{rank}_shrink_alpha"] = shrink_alpha[rank]
                        base[f"rank{rank}_shrink_relative_l2"] = shrink_metrics[rank][
                            "relative_l2"
                        ]
                        base[f"rank{rank}_shrink_cosine"] = shrink_metrics[rank][
                            "cosine"
                        ]
                    rank32_correction = fixed_outputs[32] - sparse_output
                    for snr_gate in SNR_GATES:
                        gated_output = (
                            fixed_outputs[32]
                            if correction_snr[32] >= snr_gate
                            else sparse_output
                        )
                        metrics = output_metrics(gated_output, full_output)
                        name = f"rank32_snr{snr_gate:g}"
                        base[f"{name}_active"] = int(
                            correction_snr[32] >= snr_gate
                        )
                        base[f"{name}_relative_l2"] = metrics["relative_l2"]
                        base[f"{name}_cosine"] = metrics["cosine"]
                    base["rank8_rank32_disagreement_ratio"] = float(
                        (fixed_outputs[32] - fixed_outputs[8]).norm()
                        / rank32_correction.norm().clamp_min(1.0e-12)
                    )
                    for tolerance in tolerances:
                        chosen = max(RANKS)
                        for rank in RANKS:
                            certificate = (1.0 - sample_mass) * residual_ratio[rank]
                            if certificate <= tolerance:
                                chosen = rank
                                break
                        name = f"tau_{tolerance:g}"
                        base[f"{name}_rank"] = chosen
                        base[f"{name}_certificate"] = (
                            (1.0 - sample_mass) * residual_ratio[chosen]
                        )
                        base[f"{name}_relative_l2"] = fixed_metrics[chosen]["relative_l2"]
                        base[f"{name}_cosine"] = fixed_metrics[chosen]["cosine"]
                    rows.append(base)
            print(json.dumps({"trace": trace.stem, "layer": layer, "rows": len(rows)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "schema": "qksieve_mass_adaptive_value_rank_v1",
        "cases": len(rows),
        "traces": [str(path) for path in traces],
        "signal": {
            "sample_vs_proxy_mass_pearson": correlation(
                [float(row["sample_mass"]) for row in rows],
                [float(row["proxy_mass"]) for row in rows],
            ),
            "sample_vs_exact_mass_pearson": correlation(
                [float(row["sample_mass"]) for row in rows],
                [float(row["exact_mass"]) for row in rows],
            ),
            "sample_minus_proxy_mass": quantiles(
                [float(row["sample_mass"] - row["proxy_mass"]) for row in rows]
            ),
            "sample_minus_exact_mass": quantiles(
                [float(row["sample_mass"] - row["exact_mass"]) for row in rows]
            ),
            "candidate_count": quantiles(
                [float(row["candidate_count"]) for row in rows]
            ),
        },
        "fixed": {},
        "confidence_control": {},
        "adaptive": {},
        "by_trace": {},
    }
    for rank in RANKS:
        summary["fixed"][f"rank{rank}"] = {
            "relative_l2": quantiles(
                [float(row[f"rank{rank}_relative_l2"]) for row in rows]
            ),
            "cosine": quantiles([float(row[f"rank{rank}_cosine"]) for row in rows]),
            "correction_snr": quantiles(
                [float(row[f"rank{rank}_correction_snr"]) for row in rows]
            ),
            "shrink_alpha": quantiles(
                [float(row[f"rank{rank}_shrink_alpha"]) for row in rows]
            ),
            "shrink_relative_l2": quantiles(
                [float(row[f"rank{rank}_shrink_relative_l2"]) for row in rows]
            ),
        }
    summary["confidence_control"]["sparse_only"] = {
        "relative_l2": quantiles(
            [float(row["sparse_relative_l2"]) for row in rows]
        ),
        "cosine": quantiles([float(row["sparse_cosine"]) for row in rows]),
    }
    for snr_gate in SNR_GATES:
        name = f"rank32_snr{snr_gate:g}"
        summary["confidence_control"][name] = {
            "snr_gate": snr_gate,
            "active_fraction": sum(int(row[f"{name}_active"]) for row in rows)
            / len(rows),
            "relative_l2": quantiles(
                [float(row[f"{name}_relative_l2"]) for row in rows]
            ),
            "cosine": quantiles(
                [float(row[f"{name}_cosine"]) for row in rows]
            ),
        }
    summary["confidence_control"]["rank8_rank32_disagreement_ratio"] = quantiles(
        [float(row["rank8_rank32_disagreement_ratio"]) for row in rows]
    )
    for tolerance in tolerances:
        name = f"tau_{tolerance:g}"
        ranks = [int(row[f"{name}_rank"]) for row in rows]
        summary["adaptive"][name] = {
            "tolerance": tolerance,
            "mean_rank": sum(ranks) / len(ranks),
            "rank_distribution": dict(sorted(Counter(ranks).items())),
            "relative_l2": quantiles(
                [float(row[f"{name}_relative_l2"]) for row in rows]
            ),
            "cosine": quantiles([float(row[f"{name}_cosine"]) for row in rows]),
        }
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trace[str(row["trace"])].append(row)
    for trace, trace_rows in sorted(by_trace.items()):
        summary["by_trace"][trace] = {
            name: {
                "mean_rank": sum(int(row[f"{name}_rank"]) for row in trace_rows)
                / len(trace_rows),
                "relative_l2": quantiles(
                    [float(row[f"{name}_relative_l2"]) for row in trace_rows]
                ),
            }
            for name in summary["adaptive"]
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
