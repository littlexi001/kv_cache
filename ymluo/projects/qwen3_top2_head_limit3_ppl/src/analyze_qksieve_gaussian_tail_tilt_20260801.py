#!/usr/bin/env python
"""Audit a query-conditioned Value-tail estimate that needs no token Value scan.

For proxy coordinates x and Values v, exponential tilting by z=u^T x gives
E[v exp(z)] / E[exp(z)] = mu_v + Cov(v, x) u under a joint Gaussian model.
Conditioning on z below the retrieval threshold adds the inverse-Mills term.
The moments are fit on a strided request-local sample and may be low-rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

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


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one float")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def output_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(output - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        ),
        "cosine": float(F.cosine_similarity(output, reference, dim=0)),
    }


def inverse_mills_left(value: torch.Tensor) -> torch.Tensor:
    log_density = -0.5 * value.square() - 0.5 * math.log(2.0 * math.pi)
    return torch.exp(log_density - torch.special.log_ndtr(value)).clamp_max(1.0e4)


def combine_selected_and_tail(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    tail_mean: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    threshold = proxy_scores[selected].amin()
    tail_mask = torch.ones_like(proxy_scores, dtype=torch.bool)
    tail_mask[selected] = False
    tail_denominator = torch.exp(
        (proxy_scores[tail_mask] - threshold).clamp(min=-80.0, max=0.0)
    ).sum()
    selected_scores = exact_scores[selected]
    maximum = torch.maximum(selected_scores.amax(), threshold)
    selected_weights = torch.exp((selected_scores - maximum).clamp_min(-80.0))
    tail_factor = torch.exp((threshold - maximum).clamp(min=-80.0, max=80.0))
    tail_scale = float(alpha) * tail_factor * tail_denominator
    numerator = selected_weights @ values[selected] + tail_scale * tail_mean
    denominator = selected_weights.sum() + tail_scale
    return numerator / denominator.clamp_min(1.0e-20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--basis_sample_stride", type=int, default=32)
    parser.add_argument("--moment_sample_stride", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--rate_budget", type=int, default=15)
    parser.add_argument("--fractions", default="0.01,0.02")
    parser.add_argument("--ranks", default="4,8,16,32,64,128")
    parser.add_argument("--alphas", default="0.5,1.0")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in args.traces.split(",") if x.strip())
    fractions = parse_floats(args.fractions)
    ranks = parse_ints(args.ranks)
    alphas = parse_floats(args.alphas)
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
            kv_heads, token_count, _ = key.shape
            groups = query.shape[0] // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                calibration = query[kv_head * groups : (kv_head + 1) * groups]
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.basis_sample_stride],
                    calibration,
                    args.query_shrinkage,
                )
                coefficients = head_key @ key_factor
                projected_calibration = calibration @ query_factor
                bands = quantized_bands(coefficients, projected_calibration)
                key_distortion, _ = distortion_table(
                    coefficients,
                    projected_calibration,
                    ZERO_BIT_LEVELS,
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                proxy_coordinates = reconstruct(bands, allocation).float()

                moment_x = proxy_coordinates[:: args.moment_sample_stride]
                moment_v = head_value[:: args.moment_sample_stride]
                mean_x = moment_x.mean(dim=0)
                mean_v = moment_v.mean(dim=0)
                centered_x = moment_x - mean_x
                centered_v = moment_v - mean_v
                normalizer = float(max(1, moment_x.shape[0] - 1))
                covariance_x = centered_x.T @ centered_x / normalizer
                cross_covariance = centered_v.T @ centered_x / normalizer
                left, singular, right_t = torch.linalg.svd(
                    cross_covariance, full_matrices=False
                )

                for group in range(groups):
                    query_head = kv_head * groups + group
                    projected_query = query[query_head] @ query_factor
                    proxy_query = query_int8(projected_query).float() * scaling
                    exact_scores = head_key @ query[query_head] * scaling
                    proxy_scores = proxy_coordinates @ proxy_query
                    full_probability = torch.softmax(exact_scores, dim=0)
                    full_output = full_probability @ head_value
                    score_mean = mean_x @ proxy_query
                    score_variance = (
                        proxy_query @ covariance_x @ proxy_query
                    ).clamp_min(1.0e-12)
                    score_std = score_variance.sqrt()

                    for fraction in fractions:
                        keep = min(token_count, max(1, math.ceil(fraction * token_count)))
                        selected = torch.topk(proxy_scores, k=keep).indices
                        threshold = proxy_scores[selected].amin()
                        tail_mask = torch.ones_like(proxy_scores, dtype=torch.bool)
                        tail_mask[selected] = False
                        tail_weights = torch.exp(
                            (proxy_scores[tail_mask] - threshold).clamp(
                                min=-80.0, max=0.0
                            )
                        )
                        empirical_tail_mean = (
                            tail_weights @ head_value[tail_mask]
                            / tail_weights.sum().clamp_min(1.0e-20)
                        )
                        tilted_boundary = (
                            threshold - score_mean - score_variance
                        ) / score_std
                        truncation_factor = 1.0 - (
                            inverse_mills_left(tilted_boundary) / score_std
                        )
                        exact_selected_mass = float(full_probability[selected].sum())

                        estimators: dict[str, torch.Tensor] = {
                            "global_mean": mean_v,
                            "empirical_proxy_tail": empirical_tail_mean,
                        }
                        for rank in ranks:
                            actual_rank = min(rank, singular.numel())
                            projected_cross = left[:, :actual_rank] @ (
                                singular[:actual_rank]
                                * (right_t[:actual_rank] @ proxy_query)
                            )
                            estimators[f"tilt_r{rank}"] = mean_v + projected_cross
                            estimators[f"truncated_tilt_r{rank}"] = (
                                mean_v + truncation_factor * projected_cross
                            )

                        sparse = torch.softmax(exact_scores[selected], dim=0) @ head_value[selected]
                        rows.append(
                            {
                                "trace": trace.stem,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "fraction": fraction,
                                "alpha": 0.0,
                                "method": "sparse",
                                "rank": 0,
                                "allocation": "-".join(map(str, allocation)),
                                "exact_selected_mass": exact_selected_mass,
                                "proxy_tail_mass": float(
                                    tail_weights.sum()
                                    / (
                                        tail_weights.sum()
                                        + torch.exp(
                                            (proxy_scores[selected] - threshold).clamp(max=70.0)
                                        ).sum()
                                    ).clamp_min(1.0e-20)
                                ),
                                **output_metrics(sparse, full_output),
                            }
                        )
                        for name, tail_mean in estimators.items():
                            rank = int(name.rsplit("r", 1)[-1]) if "_r" in name else 0
                            tail_error = output_metrics(tail_mean, empirical_tail_mean)
                            for alpha in alphas:
                                output = combine_selected_and_tail(
                                    exact_scores,
                                    proxy_scores,
                                    head_value,
                                    selected,
                                    tail_mean,
                                    alpha,
                                )
                                rows.append(
                                    {
                                        "trace": trace.stem,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "fraction": fraction,
                                        "alpha": alpha,
                                        "method": name,
                                        "rank": rank,
                                        "allocation": "-".join(map(str, allocation)),
                                        "exact_selected_mass": exact_selected_mass,
                                        "tail_relative_l2": tail_error["relative_l2"],
                                        "tail_cosine": tail_error["cosine"],
                                        **output_metrics(output, full_output),
                                    }
                                )
            print(
                json.dumps(
                    {"trace": trace.stem, "layer": layer, "rows": len(rows)}
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["fraction"]), float(row["alpha"]))].append(row)
    summary = []
    for (method, fraction, alpha), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "fraction": fraction,
            "alpha": alpha,
            "rank": int(items[0]["rank"]),
            "cases": len(items),
        }
        for metric in ("relative_l2", "cosine", "exact_selected_mass"):
            for statistic, value in summarize(float(x[metric]) for x in items).items():
                result[f"{metric}_{statistic}"] = value
        if all("tail_relative_l2" in item for item in items):
            for metric in ("tail_relative_l2", "tail_cosine"):
                for statistic, value in summarize(float(x[metric]) for x in items).items():
                    result[f"{metric}_{statistic}"] = value
        summary.append(result)
    report = {
        "schema": "qksieve-gaussian-tail-tilt-v1",
        "traces": [str(path) for path in traces],
        "protocol": {
            "basis_sample_stride": args.basis_sample_stride,
            "moment_sample_stride": args.moment_sample_stride,
            "rate_budget": args.rate_budget,
            "quality_boundary": (
                "Offline real-QKV mechanism audit; moments use a strided "
                "request-local sample and no token-level Value sketch."
            ),
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
