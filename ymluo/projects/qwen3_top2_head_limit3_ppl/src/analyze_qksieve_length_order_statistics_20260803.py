#!/usr/bin/env python
"""Measure how QK top-k order statistics change with nested history length.

The decode query and RoPE-encoded keys are held fixed.  Only a nested suffix of
the candidate pool is enlarged, so changes isolate the candidate-count effect
from query drift.  The study reports exact omitted mass, proxy crossings, score
error extremes, and calibration effects; it is a diagnostic, not a PPL result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
from analyze_qksieve_output_risk_budget_20260803 import affine_calibrate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lengths", default="4096,8192,16384,32768,65536,98304,131008"
    )
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--calibration_samples", type=int, default=256)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, probability))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "maximum": max(values),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.calibration_samples <= 1:
        raise ValueError("top_k must be positive and calibration_samples > 1")
    requested_lengths = sorted(
        {int(item) for item in args.lengths.split(",") if item.strip()}
    )
    if not requested_lengths or requested_lengths[0] <= 1:
        raise ValueError("lengths must contain positive values greater than one")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        args.trace, map_location="cpu", weights_only=False, mmap=True
    )
    state_by_layer: dict[int, dict[str, Any]] = {}
    queries_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    for record in payload["records"]:
        layer = int(record["layer"])
        queries_by_layer[layer].append(record["query"])
        if record.get("key") is not None:
            state_by_layer.setdefault(layer, record)

    rows: list[dict[str, Any]] = []
    for layer in sorted(state_by_layer):
        record = state_by_layer[layer]
        key_all = record["key"].to(device).float()[0]
        scaling = float(record["scaling"])
        decode_query = torch.cat(
            [item.to(device).float()[:, :, 0, :] for item in queries_by_layer[layer]],
            dim=0,
        )
        query_head_count = int(decode_query.shape[1])
        kv_head_count = int(key_all.shape[0])
        query_groups = query_head_count // kv_head_count
        maximum_history = int(key_all.shape[1])
        active_lengths = [
            min(length, maximum_history) for length in requested_lengths
        ]
        active_lengths = sorted(set(active_lengths))

        for length in active_lengths:
            active_top_k = min(args.top_k, length - 1)
            suffix = key_all[:, maximum_history - length :]
            for kv_head in range(kv_head_count):
                key = suffix[kv_head]
                head_slice = slice(
                    kv_head * query_groups, (kv_head + 1) * query_groups
                )
                flat_queries = decode_query[:, head_slice].reshape(
                    -1, decode_query.shape[-1]
                )
                query_factor, key_factor, _ = qk_balanced_factors(
                    key[:: args.key_sample_stride],
                    flat_queries,
                    args.query_shrinkage,
                )
                key_coordinates = key @ key_factor
                projected_queries = flat_queries @ query_factor
                bands = quantized_bands(key_coordinates, projected_queries)
                key_distortion, _ = distortion_table(
                    key_coordinates, projected_queries, ZERO_BIT_LEVELS
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.key_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                reconstructed_key = reconstruct(bands, allocation)
                approximate_queries = torch.stack(
                    [query_int8(query) for query in projected_queries], dim=0
                )
                exact_scores = flat_queries @ key.T * scaling
                proxy_scores = (
                    approximate_queries.float() @ reconstructed_key.T * scaling
                )
                calibrated_scores, slopes, _ = affine_calibrate_scores(
                    exact_scores, proxy_scores, args.calibration_samples
                )

                exact_indices = torch.topk(
                    exact_scores, active_top_k, dim=-1, sorted=False
                ).indices
                exact_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
                exact_mask.scatter_(1, exact_indices, True)
                exact_weights = torch.softmax(exact_scores, dim=-1)
                exact_topk_mass = (
                    exact_weights * exact_mask.to(exact_weights.dtype)
                ).sum(dim=-1)
                exact_boundary = torch.topk(
                    exact_scores,
                    active_top_k + 1,
                    dim=-1,
                    sorted=True,
                ).values
                exact_gap = (
                    exact_boundary[:, active_top_k - 1]
                    - exact_boundary[:, active_top_k]
                )

                for name, candidate_scores in (
                    ("proxy", proxy_scores),
                    ("calibrated", calibrated_scores),
                ):
                    candidate_indices = torch.topk(
                        candidate_scores,
                        active_top_k,
                        dim=-1,
                        sorted=False,
                    ).indices
                    candidate_mask = torch.zeros_like(
                        candidate_scores, dtype=torch.bool
                    )
                    candidate_mask.scatter_(1, candidate_indices, True)
                    intersection = (candidate_mask & exact_mask).sum(dim=-1)
                    selected_mass = (
                        exact_weights
                        * candidate_mask.to(exact_weights.dtype)
                    ).sum(dim=-1)
                    candidate_boundary = torch.topk(
                        candidate_scores,
                        active_top_k,
                        dim=-1,
                        sorted=True,
                    ).values[:, -1]
                    error = candidate_scores - exact_scores
                    rmse = torch.sqrt(error.square().mean(dim=-1))
                    max_error = error.abs().amax(dim=-1)
                    exact_std = exact_scores.std(dim=-1).clamp_min(1.0e-12)
                    near_boundary = (
                        (candidate_scores - candidate_boundary[:, None]).abs()
                        <= rmse[:, None]
                    ).sum(dim=-1)
                    near_boundary_2sigma = (
                        (candidate_scores - candidate_boundary[:, None]).abs()
                        <= 2.0 * rmse[:, None]
                    ).sum(dim=-1)
                    for query_row in range(flat_queries.shape[0]):
                        rows.append(
                            {
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_row": query_row,
                                "length": length,
                                "top_k": active_top_k,
                                "selector": name,
                                "score_rmse": float(rmse[query_row]),
                                "score_rmse_over_std": float(
                                    rmse[query_row] / exact_std[query_row]
                                ),
                                "score_max_abs_error": float(
                                    max_error[query_row]
                                ),
                                "max_error_over_rmse": float(
                                    max_error[query_row]
                                    / rmse[query_row].clamp_min(1.0e-12)
                                ),
                                "gaussian_extreme_reference": math.sqrt(
                                    2.0 * math.log(length)
                                ),
                                "exact_boundary_gap": float(
                                    exact_gap[query_row]
                                ),
                                "normalized_boundary_gap": float(
                                    exact_gap[query_row]
                                    / rmse[query_row].clamp_min(1.0e-12)
                                ),
                                "near_boundary_1sigma": int(
                                    near_boundary[query_row]
                                ),
                                "near_boundary_2sigma": int(
                                    near_boundary_2sigma[query_row]
                                ),
                                "topk_recall": float(
                                    intersection[query_row] / active_top_k
                                ),
                                "selected_exact_mass": float(
                                    selected_mass[query_row]
                                ),
                                "exact_topk_mass": float(
                                    exact_topk_mass[query_row]
                                ),
                                "selected_mass_over_oracle": float(
                                    selected_mass[query_row]
                                    / exact_topk_mass[query_row].clamp_min(1.0e-20)
                                ),
                                "calibration_slope": float(slopes[query_row]),
                            }
                        )
            print(
                json.dumps({"layer": layer, "length": length}), flush=True
            )
        del key_all, decode_query
        torch.cuda.empty_cache()

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["length"]), str(row["selector"]))].append(row)
    metrics = (
        "score_rmse",
        "score_rmse_over_std",
        "score_max_abs_error",
        "max_error_over_rmse",
        "exact_boundary_gap",
        "normalized_boundary_gap",
        "near_boundary_1sigma",
        "near_boundary_2sigma",
        "topk_recall",
        "selected_exact_mass",
        "exact_topk_mass",
        "selected_mass_over_oracle",
        "calibration_slope",
    )
    summary_rows = []
    for (length, selector), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "length": length,
            "selector": selector,
            "cases": len(group),
            "top_k": int(group[0]["top_k"]),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary.update(
                {
                    f"{metric}_{name}": value
                    for name, value in summarize(values).items()
                }
            )
        summary_rows.append(summary)

    report = {
        "schema": "qksieve_length_order_statistics_v1",
        "setup": {
            "trace": str(args.trace),
            "topic": payload.get("config", {}).get("topic", args.trace.stem),
            "fixed_query": True,
            "nested_history": "RoPE-preserving suffix",
            "lengths": sorted({int(row["length"]) for row in rows}),
            "top_k": args.top_k,
            "key_rate_budget": args.key_rate_budget,
            "calibration_samples": args.calibration_samples,
        },
        "claim_boundary": (
            "The query and encoded keys are fixed while the candidate suffix is "
            "expanded. Results isolate candidate-count order statistics and do "
            "not measure end-to-end model quality."
        ),
        "summary": summary_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
