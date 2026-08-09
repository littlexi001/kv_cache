#!/usr/bin/env python
"""Audit low-rank-plus-diagonal approximations to QKSieve Value risk."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from analyze_qksieve_tail_partition_calibration_20260803 import (
    load_output_projection,
    metric_value_basis,
)
from analyze_qksieve_value_sketch_residual_20260801 import (
    block_affine_quantize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--value_rank", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--evaluation_stride", type=int, default=16)
    parser.add_argument("--metric_ranks", default="4,8,16,32,64")
    parser.add_argument("--timing_iterations", type=int, default=30)
    return parser.parse_args()


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        values.numel(), dtype=torch.float32, device=values.device
    )
    return ranks


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left.float() - left.float().mean()
    right_centered = right.float() - right.float().mean()
    denominator = (
        left_centered.square().sum().sqrt()
        * right_centered.square().sum().sqrt()
    ).clamp_min(1.0e-20)
    return float((left_centered * right_centered).sum() / denominator)


def top_fraction_recall(
    approximate: torch.Tensor,
    exact: torch.Tensor,
    fraction: float,
) -> float:
    count = max(1, math.ceil(fraction * exact.numel()))
    exact_indices = torch.topk(exact, count, sorted=False).indices
    approximate_indices = torch.topk(
        approximate, count, sorted=False
    ).indices
    selected = torch.zeros_like(exact, dtype=torch.bool)
    selected.scatter_(0, approximate_indices, True)
    return float(selected.gather(0, exact_indices).float().mean())


def timed_ms(callable_object: Any, iterations: int) -> float:
    for _ in range(3):
        callable_object()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_object()
        stop.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return float(torch.tensor(samples).median())


def risk_metrics(
    approximate_squared: torch.Tensor,
    exact_squared: torch.Tensor,
) -> dict[str, float]:
    approximate = approximate_squared.clamp_min(1.0e-30).sqrt()
    exact = exact_squared.clamp_min(1.0e-30).sqrt()
    log_approximate = approximate.log()
    log_exact = exact.log()
    return {
        "risk_relative_l2": float(
            (approximate - exact).norm() / exact.norm().clamp_min(1.0e-20)
        ),
        "log_risk_rmse": float(
            (log_approximate - log_exact).square().mean().sqrt()
        ),
        "pearson": correlation(log_approximate, log_exact),
        "spearman": correlation(
            rankdata(log_approximate), rankdata(log_exact)
        ),
        **{
            f"top{int(100 * fraction)}_recall": top_fraction_recall(
                log_approximate, log_exact, fraction
            )
            for fraction in (0.01, 0.02, 0.05, 0.10)
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    metric_ranks = tuple(
        sorted({int(item) for item in args.metric_ranks.split(",") if item})
    )
    if not metric_ranks or min(metric_ranks) <= 0:
        raise ValueError("metric ranks must be positive")
    if args.evaluation_stride <= 0 or args.value_sample_stride <= 0:
        raise ValueError("sample strides must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        args.trace, map_location="cpu", weights_only=False, mmap=True
    )
    state_by_layer: dict[int, dict[str, Any]] = {}
    for record in payload["records"]:
        if record.get("key") is not None and record.get("value") is not None:
            state_by_layer.setdefault(int(record["layer"]), record)
    model_root = str(
        args.model_name_or_path
        or payload.get("config", {}).get("model_name_or_path", "")
    )
    topic = str(payload.get("config", {}).get("topic", args.trace.stem))
    rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for layer, record in sorted(state_by_layer.items()):
        value_all = record["value"].to(device).float()[0]
        projection = load_output_projection(model_root, layer, device).float()
        kv_head_count, token_count, head_dimension = value_all.shape
        query_head_count = projection.shape[1] // head_dimension
        if query_head_count % kv_head_count:
            raise ValueError("query heads must be divisible by KV heads")
        query_groups = query_head_count // kv_head_count
        evaluation_indices = torch.arange(
            0, token_count, args.evaluation_stride, device=device
        )

        for kv_head in range(kv_head_count):
            group_gram = torch.zeros(
                head_dimension,
                head_dimension,
                dtype=torch.float32,
                device=device,
            )
            first_query_head = kv_head * query_groups
            for query_head in range(
                first_query_head, first_query_head + query_groups
            ):
                start = query_head * head_dimension
                block = projection[:, start : start + head_dimension]
                group_gram.add_(block.T @ block)

            value = value_all[kv_head]
            mean, vectors, coefficients, _ = metric_value_basis(
                value,
                group_gram,
                sample_stride=args.value_sample_stride,
                maximum_rank=args.value_rank,
            )
            quantized_coefficients = block_affine_quantize(
                coefficients[:, : args.value_rank],
                bits=args.value_bits,
                block_size=256,
            )
            sampled_value = value.index_select(0, evaluation_indices)
            sampled_coefficients = quantized_coefficients.index_select(
                0, evaluation_indices
            )
            reconstructed = (
                mean
                + sampled_coefficients
                @ vectors[:, : args.value_rank].T
            )
            residual = sampled_value - reconstructed.float()
            exact_squared = torch.einsum(
                "nd,de,ne->n", residual, group_gram, residual
            ).clamp_min(0.0)
            diagonal_squared = (
                residual.square() * group_gram.diagonal()
            ).sum(dim=-1).clamp_min(0.0)
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": "diagonal",
                    "metric_rank": 0,
                    "sample_tokens": int(residual.shape[0]),
                    **risk_metrics(diagonal_squared, exact_squared),
                }
            )

            eigenvalues, eigenvectors = torch.linalg.eigh(group_gram)
            eigenvalues = eigenvalues.clamp_min(0.0)
            total_energy = eigenvalues.sum().clamp_min(1.0e-20)
            for metric_rank in metric_ranks:
                active_rank = min(metric_rank, head_dimension)
                active_values = eigenvalues[-active_rank:]
                active_vectors = eigenvectors[:, -active_rank:]
                projected = residual @ active_vectors
                lowrank_squared = (
                    projected.square() * active_values
                ).sum(dim=-1)
                represented_diagonal = (
                    active_vectors.square() * active_values
                ).sum(dim=-1)
                diagonal_remainder = (
                    group_gram.diagonal() - represented_diagonal
                ).clamp_min(0.0)
                approximate_squared = (
                    lowrank_squared
                    + (residual.square() * diagonal_remainder).sum(dim=-1)
                ).clamp_min(0.0)
                rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": "lowrank_plus_diagonal",
                        "metric_rank": active_rank,
                        "sample_tokens": int(residual.shape[0]),
                        **risk_metrics(approximate_squared, exact_squared),
                    }
                )
                spectrum_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "metric_rank": active_rank,
                        "gram_energy": float(active_values.sum() / total_energy),
                    }
                )

            if device.type == "cuda":
                diagonal_metric = group_gram.diagonal()
                timing_rank4 = min(4, head_dimension)
                timing_values4 = eigenvalues[-timing_rank4:]
                timing_vectors4 = eigenvectors[:, -timing_rank4:]
                timing_diagonal4 = (
                    diagonal_metric
                    - (timing_vectors4.square() * timing_values4).sum(dim=-1)
                ).clamp_min(0.0)
                timing_rank = min(16, head_dimension)
                timing_values = eigenvalues[-timing_rank:]
                timing_vectors = eigenvectors[:, -timing_rank:]
                timing_diagonal = (
                    diagonal_metric
                    - (timing_vectors.square() * timing_values).sum(dim=-1)
                ).clamp_min(0.0)
                exact_ms = timed_ms(
                    lambda: torch.einsum(
                        "nd,de,ne->n", residual, group_gram, residual
                    ),
                    args.timing_iterations,
                )
                diagonal_ms = timed_ms(
                    lambda: (
                        residual.square() * diagonal_metric
                    ).sum(dim=-1),
                    args.timing_iterations,
                )
                rank4_ms = timed_ms(
                    lambda: (
                        (residual @ timing_vectors4).square()
                        * timing_values4
                    ).sum(dim=-1)
                    + (residual.square() * timing_diagonal4).sum(dim=-1),
                    args.timing_iterations,
                )
                approximate_ms = timed_ms(
                    lambda: (
                        (residual @ timing_vectors).square()
                        * timing_values
                    ).sum(dim=-1)
                    + (residual.square() * timing_diagonal).sum(dim=-1),
                    args.timing_iterations,
                )
                timing_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "sample_tokens": int(residual.shape[0]),
                        "exact_metric_ms": exact_ms,
                        "diagonal_metric_ms": diagonal_ms,
                        "rank4_plus_diagonal_ms": rank4_ms,
                        "rank16_plus_diagonal_ms": approximate_ms,
                        "diagonal_metric_speedup": exact_ms / diagonal_ms,
                        "rank4_metric_speedup": exact_ms / rank4_ms,
                        "rank16_metric_speedup": exact_ms / approximate_ms,
                    }
                )

        del value_all, projection
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["metric_rank"]))
        grouped.setdefault(key, []).append(row)
    summary = {
        f"{method}_r{rank}": {
            "rows": len(items),
            **{
                metric: fmean(float(item[metric]) for item in items)
                for metric in (
                    "risk_relative_l2",
                    "log_risk_rmse",
                    "pearson",
                    "spearman",
                    "top1_recall",
                    "top2_recall",
                    "top5_recall",
                    "top10_recall",
                )
            },
        }
        for (method, rank), items in sorted(grouped.items())
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "spectrum.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectrum_rows[0]))
        writer.writeheader()
        writer.writerows(spectrum_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "trace": str(args.trace),
                "model_name_or_path": model_root,
                "config": vars(args) | {"trace": str(args.trace), "output_dir": str(args.output_dir)},
                "summary": summary,
                "timing": {
                    "rows": len(timing_rows),
                    "exact_metric_ms_mean": fmean(
                        float(item["exact_metric_ms"]) for item in timing_rows
                    ),
                    "diagonal_metric_ms_mean": fmean(
                        float(item["diagonal_metric_ms"])
                        for item in timing_rows
                    ),
                    "rank4_plus_diagonal_ms_mean": fmean(
                        float(item["rank4_plus_diagonal_ms"])
                        for item in timing_rows
                    ),
                    "rank16_plus_diagonal_ms_mean": fmean(
                        float(item["rank16_plus_diagonal_ms"])
                        for item in timing_rows
                    ),
                    "diagonal_metric_speedup_mean": fmean(
                        float(item["diagonal_metric_speedup"])
                        for item in timing_rows
                    ),
                    "rank4_metric_speedup_mean": fmean(
                        float(item["rank4_metric_speedup"])
                        for item in timing_rows
                    ),
                    "rank16_metric_speedup_mean": fmean(
                        float(item["rank16_metric_speedup"])
                        for item in timing_rows
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ALL_COMPLETE").touch()


if __name__ == "__main__":
    main()
