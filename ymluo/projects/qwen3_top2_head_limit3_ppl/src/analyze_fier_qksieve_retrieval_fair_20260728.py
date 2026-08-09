#!/usr/bin/env python
"""Held-out retrieval and rate-allocation audit for QKSieve and FIER.

This is a quality/index audit, not a latency benchmark.  FIER follows the
paper's 1-bit RTN definition: per channel, groups of 32 sequence positions
share two FP16 reconstruction levels (equivalently scale and bias).
QKSieve uses its request-local QK-balanced basis and 240-bit automatic
mixed-bit allocation.  Controlled 256-bit Key-PCA/QK-balanced uniform-1bit
variants, a random-rotation control, and Key-MSE-only 240-bit allocations
separate coordinate-system and rate-allocation effects.  All methods receive
exactly the same top-k budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_head_top2_targeted_ppl_20260714 import (
    _deterministic_random_orthogonal_basis,
    _hierarchical_key_rate_allocation,
    _hierarchical_qmse_rate_allocation,
    _hierarchical_quantize_band,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_stride", type=int, default=32)
    parser.add_argument("--fier_group_size", type=int, default=32)
    parser.add_argument("--budgets", default="0.01,0.02,0.04")
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    return parser.parse_args()


def second_moment(values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhnd,bhne->bhde", values, values) / max(
        1, values.shape[-2]
    )


def sensitivity_heterogeneity(
    projected_keys: torch.Tensor,
    projected_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return band score sensitivities and their AM/GM ratio."""
    if projected_keys.shape[-1] != 128:
        raise ValueError("projected Keys must have head dimension 128")
    if projected_queries.shape[-1] != 128:
        raise ValueError("projected Queries must have head dimension 128")
    key_moment = second_moment(projected_keys).reshape(
        *projected_keys.shape[:2], 8, 16, 8, 16
    )
    query_moment = second_moment(projected_queries).reshape(
        *projected_queries.shape[:2], 8, 16, 8, 16
    )
    key_bands = torch.stack(
        [key_moment[..., band, :, band, :] for band in range(8)],
        dim=2,
    )
    query_bands = torch.stack(
        [query_moment[..., band, :, band, :] for band in range(8)],
        dim=2,
    )
    sensitivities = torch.einsum(
        "bhgij,bhgji->bhg",
        query_bands,
        key_bands,
    ).clamp_min(1.0e-20)
    arithmetic_mean = sensitivities.mean(dim=-1)
    geometric_mean = sensitivities.log().mean(dim=-1).exp()
    return sensitivities, arithmetic_mean / geometric_mean


def key_pca_basis(key_covariance: torch.Tensor) -> torch.Tensor:
    _, eigenvectors = torch.linalg.eigh(key_covariance.float())
    return eigenvectors.flip(-1).contiguous()


def pearson(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size < 2:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def ordinal_rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks


def mapped_queries(records: list[dict[str, Any]]) -> torch.Tensor:
    queries = torch.cat(
        [record["query"].float() for record in records], dim=2
    )
    batch, query_heads, steps, head_dim = queries.shape
    kv_heads = int(records[0]["key"].shape[1])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    return (
        queries.reshape(batch, kv_heads, groups, steps, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, steps * groups, head_dim)
        .contiguous()
    )


def fier_rtn1_dequantize(
    key: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Return the two-level RTN reconstruction used by FIER-g32."""
    token_count = int(key.shape[-2])
    group_count = math.ceil(token_count / group_size)
    padded_count = group_count * group_size
    if padded_count != token_count:
        padding = key[..., -1:, :].expand(
            *key.shape[:-2], padded_count - token_count, key.shape[-1]
        )
        working = torch.cat((key, padding), dim=-2)
    else:
        working = key
    grouped = working.reshape(
        *working.shape[:-2], group_count, group_size, working.shape[-1]
    )
    lower = grouped.amin(dim=-2, keepdim=True)
    upper = grouped.amax(dim=-2, keepdim=True)
    midpoint = (lower + upper) * 0.5
    reconstructed = torch.where(grouped >= midpoint, upper, lower)
    return reconstructed.reshape(*working.shape)[..., :token_count, :]


def qksieve_reconstruct(
    projected_key: torch.Tensor, allocation: torch.Tensor
) -> torch.Tensor:
    reconstructed = torch.empty_like(projected_key)
    for head in range(projected_key.shape[1]):
        for band in range(8):
            start = 16 * band
            bits = int(allocation[0, head, band].item())
            reconstructed[:, head, :, start : start + 16] = (
                _hierarchical_quantize_band(
                    projected_key[:, head, :, start : start + 16],
                    bits,
                )
            )
    return reconstructed


def score_metrics(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    scaling: float,
    budget_fraction: float,
) -> dict[str, float]:
    token_count = int(exact_scores.shape[-1])
    budget = min(token_count, max(1, math.ceil(token_count * budget_fraction)))
    exact_scaled = exact_scores.float() * scaling
    proxy_scaled = proxy_scores.float() * scaling
    exact_top = torch.topk(
        exact_scaled, k=budget, dim=-1, sorted=False
    ).indices
    proxy_top = torch.topk(
        proxy_scaled, k=budget, dim=-1, sorted=False
    ).indices
    exact_mask = torch.zeros_like(exact_scaled, dtype=torch.bool)
    exact_mask.scatter_(-1, exact_top, True)
    recall = exact_mask.gather(-1, proxy_top).float().mean(dim=-1)
    probabilities = torch.softmax(exact_scaled, dim=-1)
    mass = probabilities.gather(-1, proxy_top).sum(dim=-1)
    oracle_mass = probabilities.gather(-1, exact_top).sum(dim=-1)
    top1 = (
        exact_scaled.argmax(dim=-1) == proxy_scaled.argmax(dim=-1)
    ).float()
    delta = exact_scaled - proxy_scaled
    centered_delta = delta - delta.mean(dim=-1, keepdim=True)
    centered_rmse = centered_delta.square().mean(dim=-1).sqrt()
    exact_centered = exact_scaled - exact_scaled.mean(dim=-1, keepdim=True)
    proxy_centered = proxy_scaled - proxy_scaled.mean(dim=-1, keepdim=True)
    correlation = (
        (exact_centered * proxy_centered).mean(dim=-1)
        / (
            exact_centered.square().mean(dim=-1).sqrt()
            * proxy_centered.square().mean(dim=-1).sqrt()
        ).clamp_min(1.0e-12)
    )
    return {
        "budget_tokens": budget,
        "topk_recall": float(recall.mean().item()),
        "attention_mass_recall": float(mass.mean().item()),
        "oracle_topk_mass": float(oracle_mass.mean().item()),
        "top1_recall": float(top1.mean().item()),
        "centered_score_rmse": float(centered_rmse.mean().item()),
        "score_correlation": float(correlation.mean().item()),
        "query_heads": int(exact_scores.numel() // token_count),
    }


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total_weight = sum(int(row["query_heads"]) for row in rows)
    return sum(
        float(row[field]) * int(row["query_heads"]) for row in rows
    ) / total_weight


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = tuple(float(item) for item in args.budgets.split(","))
    if any(value <= 0.0 or value > 1.0 for value in budgets):
        raise ValueError("budgets must be in (0, 1]")
    device = torch.device(args.device)
    detail_rows: list[dict[str, Any]] = []
    heterogeneity_rows: list[dict[str, Any]] = []
    allocation_counter: Counter[tuple[int, ...]] = Counter()

    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        trace = torch.load(
            path_text, map_location="cpu", weights_only=False
        )
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in trace["records"]:
            by_layer[int(record["layer"])].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            if len(records) <= args.calibration_steps:
                raise ValueError(
                    f"layer {layer} has no held-out Query records"
                )
            calibration = records[: args.calibration_steps]
            evaluation = records[args.calibration_steps :]
            key = calibration[0]["key"].to(device=device).float()
            kv_heads = int(key.shape[1])
            sampled_key = key[..., :: args.key_stride, :].contiguous()
            calibration_query = mapped_queries(calibration).to(device)
            key_covariance = second_moment(sampled_key)
            query_covariance = second_moment(calibration_query)
            query_basis, key_basis = _qk_metric_projection_factors(
                key_covariance,
                query_covariance,
                projection_dim=key.shape[-1],
                query_shrinkage=args.query_shrinkage,
            )
            projected_sample = torch.einsum(
                "bhnd,bhdm->bhnm", sampled_key, key_basis
            )
            projected_calibration_query = torch.einsum(
                "bhnd,bhdm->bhnm", calibration_query, query_basis
            )
            allocation = _hierarchical_qmse_rate_allocation(
                projected_sample,
                projected_calibration_query,
                bit_budget_per_coordinate=15,
                allow_zero_bits=True,
                include_scale_metadata=True,
            )
            for head_allocation in allocation[0].tolist():
                allocation_counter[tuple(head_allocation)] += 1
            uniform_allocation = torch.ones_like(allocation)
            qk_key_allocation = _hierarchical_key_rate_allocation(
                projected_sample,
                bit_budget_per_coordinate=15,
                allow_zero_bits=True,
                include_scale_metadata=True,
            )
            sensitivities, heterogeneity = sensitivity_heterogeneity(
                projected_sample,
                projected_calibration_query,
            )

            projected_key = torch.einsum(
                "bhnd,bhdm->bhnm", key, key_basis
            )
            qksieve_key = qksieve_reconstruct(
                projected_key, allocation
            )
            qk_uniform_key = qksieve_reconstruct(
                projected_key, uniform_allocation
            )
            qk_key_key = qksieve_reconstruct(
                projected_key, qk_key_allocation
            )
            pca_basis = key_pca_basis(key_covariance)
            projected_key_pca = torch.einsum(
                "bhnd,bhdm->bhnm", key, pca_basis
            )
            projected_sample_pca = torch.einsum(
                "bhnd,bhdm->bhnm", sampled_key, pca_basis
            )
            keypca_key_allocation = _hierarchical_key_rate_allocation(
                projected_sample_pca,
                bit_budget_per_coordinate=15,
                allow_zero_bits=True,
                include_scale_metadata=True,
            )
            keypca_uniform_key = qksieve_reconstruct(
                projected_key_pca, uniform_allocation
            )
            keypca_key_key = qksieve_reconstruct(
                projected_key_pca, keypca_key_allocation
            )
            random_basis = _deterministic_random_orthogonal_basis(
                int(key.shape[0]),
                kv_heads,
                int(key.shape[-1]),
                layer,
                device,
                key.dtype,
            )
            projected_key_random = torch.einsum(
                "bhnd,bhdm->bhnm", key, random_basis
            )
            random_uniform_key = qksieve_reconstruct(
                projected_key_random, uniform_allocation
            )
            fier_key = fier_rtn1_dequantize(
                key, args.fier_group_size
            )
            auto_error_sum = torch.zeros(
                kv_heads, dtype=torch.float64, device=device
            )
            uniform_error_sum = torch.zeros_like(auto_error_sum)
            key_mse_error_sum = torch.zeros_like(auto_error_sum)
            heldout_query_groups = 0

            for record in evaluation:
                query = record["query"].to(device=device).float()
                batch, query_heads, _, head_dim = query.shape
                groups = query_heads // kv_heads
                grouped_query = query[..., 0, :].reshape(
                    batch, kv_heads, groups, head_dim
                )
                projected_query = torch.einsum(
                    "bhgd,bhdm->bhgm", grouped_query, query_basis
                )
                exact_scores = torch.einsum(
                    "bhgd,bhnd->bhgn", grouped_query, key
                )
                qksieve_scores = torch.einsum(
                    "bhgd,bhnd->bhgn", projected_query, qksieve_key
                )
                qk_uniform_scores = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    projected_query,
                    qk_uniform_key,
                )
                qk_key_scores = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    projected_query,
                    qk_key_key,
                )
                keypca_query = torch.einsum(
                    "bhgd,bhdm->bhgm", grouped_query, pca_basis
                )
                keypca_uniform_scores = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    keypca_query,
                    keypca_uniform_key,
                )
                keypca_key_scores = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    keypca_query,
                    keypca_key_key,
                )
                random_query = torch.einsum(
                    "bhgd,bhdm->bhgm", grouped_query, random_basis
                )
                random_uniform_scores = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    random_query,
                    random_uniform_key,
                )
                fier_scores = torch.einsum(
                    "bhgd,bhnd->bhgn", grouped_query, fier_key
                )
                auto_error_sum += (
                    (exact_scores - qksieve_scores)
                    .double()
                    .square()
                    .mean(dim=(0, 2, 3))
                )
                uniform_error_sum += (
                    (exact_scores - qk_uniform_scores)
                    .double()
                    .square()
                    .mean(dim=(0, 2, 3))
                )
                key_mse_error_sum += (
                    (exact_scores - qk_key_scores)
                    .double()
                    .square()
                    .mean(dim=(0, 2, 3))
                )
                heldout_query_groups += 1
                for method, proxy_scores, index_bytes in (
                    ("qksieve", qksieve_scores, 30.0),
                    (
                        "qkbalanced_uniform1",
                        qk_uniform_scores,
                        32.0,
                    ),
                    (
                        "keypca_uniform1",
                        keypca_uniform_scores,
                        32.0,
                    ),
                    (
                        "random_uniform1",
                        random_uniform_scores,
                        32.0,
                    ),
                    (
                        "qkbalanced_keymse",
                        qk_key_scores,
                        30.0,
                    ),
                    (
                        "keypca_keymse",
                        keypca_key_scores,
                        30.0,
                    ),
                    ("fier_rtn1_g32", fier_scores, 32.0),
                ):
                    for budget in budgets:
                        metrics = score_metrics(
                            exact_scores,
                            proxy_scores,
                            float(record["scaling"]),
                            budget,
                        )
                        detail_rows.append(
                            {
                                "topic": topic,
                                "layer": layer,
                                "step": int(record["step"]),
                                "method": method,
                                "index_bytes_per_token_kv_head": index_bytes,
                                "index_ratio_of_fp16_kv": index_bytes / 512.0,
                                "budget_fraction": budget,
                                **metrics,
                            }
                        )
                del (
                    exact_scores,
                    qksieve_scores,
                    qk_uniform_scores,
                    qk_key_scores,
                    keypca_uniform_scores,
                    keypca_key_scores,
                    random_uniform_scores,
                    fier_scores,
                )
            for kv_head in range(kv_heads):
                auto_mse = float(
                    (auto_error_sum[kv_head] / heldout_query_groups).item()
                )
                uniform_mse = float(
                    (
                        uniform_error_sum[kv_head]
                        / heldout_query_groups
                    ).item()
                )
                key_mse = float(
                    (
                        key_mse_error_sum[kv_head]
                        / heldout_query_groups
                    ).item()
                )
                head_sensitivities = sensitivities[0, kv_head]
                heterogeneity_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "heldout_query_steps": len(evaluation),
                        "sensitivity_am_over_gm": float(
                            heterogeneity[0, kv_head].item()
                        ),
                        "sensitivity_min": float(
                            head_sensitivities.min().item()
                        ),
                        "sensitivity_max": float(
                            head_sensitivities.max().item()
                        ),
                        "auto_qmse": auto_mse,
                        "uniform1_qmse": uniform_mse,
                        "keymse_allocation_qmse": key_mse,
                        "uniform_over_auto_qmse": (
                            uniform_mse / max(auto_mse, 1.0e-20)
                        ),
                        "auto_relative_qmse_reduction": (
                            1.0 - auto_mse / max(uniform_mse, 1.0e-20)
                        ),
                        "keymse_over_querymse": (
                            key_mse / max(auto_mse, 1.0e-20)
                        ),
                    }
                )
            del (
                key,
                sampled_key,
                projected_key,
                projected_key_pca,
                projected_sample_pca,
                projected_key_random,
                qksieve_key,
                qk_uniform_key,
                qk_key_key,
                keypca_uniform_key,
                keypca_key_key,
                random_uniform_key,
                fier_key,
            )

    fieldnames = list(detail_rows[0])
    with (output_dir / "detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    with (output_dir / "heterogeneity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(heterogeneity_rows[0])
        )
        writer.writeheader()
        writer.writerows(heterogeneity_rows)

    grouped_rows: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped_rows[(row["method"], row["budget_fraction"])].append(row)
    aggregate = []
    metric_fields = (
        "topk_recall",
        "attention_mass_recall",
        "oracle_topk_mass",
        "top1_recall",
        "centered_score_rmse",
        "score_correlation",
    )
    for (method, budget), rows in sorted(grouped_rows.items()):
        aggregate.append(
            {
                "method": method,
                "budget_fraction": budget,
                "budget_tokens": rows[0]["budget_tokens"],
                "index_bytes_per_token_kv_head": rows[0][
                    "index_bytes_per_token_kv_head"
                ],
                "index_ratio_of_fp16_kv": rows[0][
                    "index_ratio_of_fp16_kv"
                ],
                "conditions": len(rows),
                "query_heads": sum(int(row["query_heads"]) for row in rows),
                **{
                    field: weighted_mean(rows, field)
                    for field in metric_fields
                },
            }
        )

    paired_cluster_bootstrap = []
    rng = np.random.default_rng(20260728)
    paired_by_condition: dict[
        tuple[str, int, int, float], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in detail_rows:
        condition = (
            str(row["topic"]),
            int(row["layer"]),
            int(row["step"]),
            float(row["budget_fraction"]),
        )
        paired_by_condition[condition][str(row["method"])] = row
    for budget in budgets:
        for metric in ("topk_recall", "attention_mass_recall"):
            cluster_deltas: dict[tuple[str, int], list[float]] = defaultdict(
                list
            )
            for (topic, layer, _, condition_budget), methods in (
                paired_by_condition.items()
            ):
                if condition_budget != budget:
                    continue
                cluster_deltas[(topic, layer)].append(
                    float(methods["qksieve"][metric])
                    - float(methods["fier_rtn1_g32"][metric])
                )
            cluster_means = np.asarray(
                [
                    np.mean(values)
                    for values in cluster_deltas.values()
                ],
                dtype=np.float64,
            )
            bootstrap_means = np.mean(
                rng.choice(
                    cluster_means,
                    size=(args.bootstrap_samples, len(cluster_means)),
                    replace=True,
                ),
                axis=1,
            )
            paired_cluster_bootstrap.append(
                {
                    "metric": metric,
                    "budget_fraction": budget,
                    "clusters": len(cluster_means),
                    "qksieve_minus_fier": float(cluster_means.mean()),
                    "ci95_low": float(
                        np.quantile(bootstrap_means, 0.025)
                    ),
                    "ci95_high": float(
                        np.quantile(bootstrap_means, 0.975)
                    ),
                    "cluster_win_rate": float(
                        np.mean(cluster_means > 0.0)
                    ),
                }
            )

    heterogeneity_values = [
        float(row["sensitivity_am_over_gm"])
        for row in heterogeneity_rows
    ]
    observed_ratios = [
        float(row["uniform_over_auto_qmse"])
        for row in heterogeneity_rows
    ]
    heterogeneity_summary = {
        "layer_head_conditions": len(heterogeneity_rows),
        "am_over_gm_mean": float(np.mean(heterogeneity_values)),
        "am_over_gm_median": float(np.median(heterogeneity_values)),
        "uniform_over_auto_qmse_mean": float(
            np.mean(observed_ratios)
        ),
        "uniform_over_auto_qmse_median": float(
            np.median(observed_ratios)
        ),
        "log_pearson": pearson(
            np.log(np.maximum(heterogeneity_values, 1.0e-20)).tolist(),
            np.log(np.maximum(observed_ratios, 1.0e-20)).tolist(),
        ),
        "spearman": pearson(
            ordinal_rank(heterogeneity_values).tolist(),
            ordinal_rank(observed_ratios).tolist(),
        ),
    }
    summary = {
        "schema": "fier_qksieve_heldout_retrieval_v1",
        "traces": args.trace,
        "calibration_steps": args.calibration_steps,
        "heldout_steps_per_layer": sorted(
            {
                len(
                    [
                        row
                        for row in detail_rows
                        if row["topic"] == topic
                        and row["layer"] == layer
                        and row["method"] == "qksieve"
                        and row["budget_fraction"] == budgets[0]
                    ]
                )
                for topic in {row["topic"] for row in detail_rows}
                for layer in {row["layer"] for row in detail_rows}
            }
        ),
        "fier_definition": {
            "bits": 1,
            "group_axis": "sequence",
            "group_size": args.fier_group_size,
            "reconstruction": "per-channel group min/max RTN",
            "index_bytes_per_token_kv_head": 32.0,
        },
        "qksieve_definition": {
            "query_shrinkage": args.query_shrinkage,
            "key_stride": args.key_stride,
            "rate_bits_per_token_kv_head": 240,
            "index_bytes_per_token_kv_head": 30.0,
            "allocation_histogram": {
                "-".join(map(str, allocation)): count
                for allocation, count in allocation_counter.most_common()
            },
        },
        "controlled_uniform1_definitions": {
            "index_bits_per_token_kv_head": 256,
            "keypca": "orthogonal Key-PCA plus 1-bit in all eight bands",
            "qkbalanced": (
                "QK-balanced biorthogonal coordinates plus 1-bit "
                "in all eight bands"
            ),
            "random": (
                "deterministic Haar orthogonal coordinates plus 1-bit "
                "in all eight bands"
            ),
        },
        "key_mse_ablation_definitions": {
            "index_bits_per_token_kv_head": 240,
            "keypca": "Key-PCA coordinates plus Key-MSE allocation",
            "qkbalanced": (
                "QK-balanced coordinates plus Key-MSE allocation; "
                "Query covariance omitted only from the rate objective"
            ),
        },
        "heterogeneity_prediction": heterogeneity_summary,
        "aggregate": aggregate,
        "paired_cluster_bootstrap": paired_cluster_bootstrap,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ALL_COMPLETE").touch()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
