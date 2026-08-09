#!/usr/bin/env python
"""Matched-read stress test for sampled control-variate sparse attention."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    control_variate_tail_statistics,
    fit_block_models,
    fit_gaussian_tilt_moments,
    gaussian_tilt_block_control_values,
    stratified_uniform_sample_indices,
)
from analyze_qksieve_gaussian_tail_synthetic_20260802 import (
    make_case,
    relative_energy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--score_dim", type=int, default=16)
    parser.add_argument("--conditional_dim", type=int, default=8)
    parser.add_argument("--value_dim", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--total_read_fraction", type=float, default=0.04)
    parser.add_argument("--samples_per_block", type=int, default=2)
    parser.add_argument("--proxy_noise", default="0.0,0.35,0.7")
    parser.add_argument("--seeds", default="3,11,29")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def block_tail_counts(
    token_count: int,
    block_size: int,
    selected: torch.Tensor,
) -> torch.Tensor:
    block_count = math.ceil(token_count / block_size)
    starts = torch.arange(block_count) * block_size
    counts = (token_count - starts).clamp(min=0, max=block_size).float()
    selected_counts = torch.zeros(block_count)
    selected_counts.index_add_(
        0,
        selected // block_size,
        torch.ones_like(selected, dtype=torch.float32),
    )
    return counts - selected_counts


def selected_only_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    return torch.softmax(scores.index_select(0, selected), dim=0) @ values.index_select(
        0, selected
    )


def combine_block_control(
    scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    block_size: int,
    base_denominator: torch.Tensor,
    base_numerator: torch.Tensor,
) -> torch.Tensor:
    tail_counts = block_tail_counts(scores.numel(), block_size, selected)
    return combine_selected_and_tail(
        scores,
        proxy_scores,
        values,
        selected,
        (tail_counts[:, None] * base_numerator).sum(dim=0),
        (tail_counts * base_denominator).sum(),
        1.0,
    )


def evaluate_case(
    case: str,
    seed: int,
    proxy_noise: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    data_generator = torch.Generator().manual_seed(seed)
    x, values = make_case(
        case,
        args.tokens,
        args.score_dim,
        args.value_dim,
        args.block_size,
        data_generator,
    )
    conditional_x = x[:, : args.conditional_dim]
    conditional_model = fit_block_models(
        conditional_x,
        values,
        args.block_size,
        ridge=0.01,
        moment_bits=8,
        linear_group_blocks=0,
    )
    gaussian_model = fit_gaussian_tilt_moments(
        x,
        args.block_size,
        moment_bits=8,
        covariance_mode="diag",
    )
    reservoir = stratified_uniform_sample_indices(
        args.tokens,
        args.block_size,
        args.samples_per_block,
        torch.Generator().manual_seed(seed + 100_003),
    )
    total_reads = max(1, math.ceil(args.tokens * args.total_read_fraction))
    heavy_count = max(1, total_reads - reservoir.numel())
    if heavy_count + reservoir.numel() > args.tokens:
        raise ValueError("matched read budget exceeds history length")
    queries = torch.randn(
        args.groups,
        args.score_dim,
        generator=torch.Generator().manual_seed(seed + 200_003),
    ) / math.sqrt(args.score_dim)
    rows: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        scores = x @ query
        noise = torch.randn(
            args.tokens,
            generator=torch.Generator().manual_seed(
                seed * 1009 + query_id * 9173 + int(proxy_noise * 10_000)
            ),
        )
        proxy_scores = scores + (
            float(proxy_noise) * scores.std().clamp_min(1.0e-8) * noise
        )
        selected_total = torch.topk(
            proxy_scores, total_reads, sorted=False
        ).indices
        selected_heavy = torch.topk(
            proxy_scores, heavy_count, sorted=False
        ).indices
        oracle_total = torch.topk(scores, total_reads, sorted=False).indices
        reference_output = torch.softmax(scores, dim=0) @ values
        proxy_reference = proxy_scores.index_select(0, selected_heavy).amin()
        gaussian_z, gaussian_y = gaussian_tilt_block_control_values(
            query,
            0.0,
            proxy_reference,
            conditional_model,
            gaussian_model,
        )
        mean = gaussian_model["mean"]
        mean_v = conditional_model["mean_v"]
        assert isinstance(mean, torch.Tensor)
        assert isinstance(mean_v, torch.Tensor)
        centroid_z = torch.exp(
            (mean.float() @ query.float() - proxy_reference).clamp(-80.0, 80.0)
        )
        centroid_y = centroid_z[:, None] * mean_v.float()
        cv_z, cv_y, cv_diagnostics = control_variate_tail_statistics(
            scores,
            values,
            selected_heavy,
            reservoir,
            args.block_size,
            gaussian_z,
            gaussian_y,
            proxy_reference,
        )
        zero_z, zero_y, zero_diagnostics = control_variate_tail_statistics(
            scores,
            values,
            selected_heavy,
            reservoir,
            args.block_size,
            torch.zeros_like(gaussian_z),
            torch.zeros_like(gaussian_y),
            proxy_reference,
        )
        outputs = {
            "proxy_topk_matched": selected_only_output(
                scores, values, selected_total
            ),
            "oracle_topk_matched": selected_only_output(
                scores, values, oracle_total
            ),
            "reduced_topk_only": selected_only_output(
                scores, values, selected_heavy
            ),
            "centroid_control_no_sample": combine_block_control(
                scores,
                proxy_scores,
                values,
                selected_heavy,
                args.block_size,
                centroid_z,
                centroid_y,
            ),
            "gaussian_control_no_sample": combine_block_control(
                scores,
                proxy_scores,
                values,
                selected_heavy,
                args.block_size,
                gaussian_z,
                gaussian_y,
            ),
            "uniform_ht": combine_selected_and_tail(
                scores,
                proxy_scores,
                values,
                selected_heavy,
                zero_y,
                zero_z,
                1.0,
            ),
            "gaussian_control_variate": combine_selected_and_tail(
                scores,
                proxy_scores,
                values,
                selected_heavy,
                cv_y,
                cv_z,
                1.0,
            ),
        }
        overlap = int(
            torch.isin(reservoir, selected_heavy).sum().item()
        )
        unique_reads = heavy_count + reservoir.numel() - overlap
        for method, output in outputs.items():
            diagnostics: dict[str, float] = {}
            if method == "uniform_ht":
                diagnostics = zero_diagnostics
            elif method == "gaussian_control_variate":
                diagnostics = cv_diagnostics
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "query_id": query_id,
                    "proxy_noise": proxy_noise,
                    "method": method,
                    "relative_output_error": relative_energy(
                        output, reference_output
                    ),
                    "target_total_reads": total_reads,
                    "heavy_tokens": heavy_count,
                    "reservoir_tokens": int(reservoir.numel()),
                    "unique_reads": (
                        unique_reads
                        if method
                        in ("uniform_ht", "gaussian_control_variate")
                        else (
                            heavy_count
                            if method
                            in (
                                "reduced_topk_only",
                                "centroid_control_no_sample",
                                "gaussian_control_no_sample",
                            )
                            else total_reads
                        )
                    ),
                    "diagnostics": diagnostics,
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["proxy_noise"], row["method"])].append(row)
    aggregate = []
    for (noise, method), group in sorted(grouped.items()):
        errors = torch.tensor(
            [row["relative_output_error"] for row in group],
            dtype=torch.float64,
        )
        aggregate.append(
            {
                "proxy_noise": noise,
                "method": method,
                "conditions": len(group),
                "relative_output_error_mean": float(errors.mean()),
                "relative_output_error_p90": float(torch.quantile(errors, 0.9)),
                "relative_output_error_worst": float(errors.max()),
                "unique_reads_mean": sum(row["unique_reads"] for row in group)
                / len(group),
                "negative_tail_block_fraction_mean": sum(
                    row["diagnostics"].get(
                        "negative_tail_block_fraction", 0.0
                    )
                    for row in group
                )
                / len(group),
            }
        )
    return aggregate


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    noise_levels = [
        float(item) for item in args.proxy_noise.split(",") if item.strip()
    ]
    cases = (
        "gaussian_linear",
        "block_shift",
        "student_t",
        "mixture_outlier",
        "nonlinear_value",
    )
    rows = [
        row
        for noise in noise_levels
        for case in cases
        for seed in seeds
        for row in evaluate_case(case, seed, noise, args)
    ]
    aggregate = aggregate_rows(rows)
    payload = {
        "schema": "qksieve_sampled_control_variate_synthetic_v1",
        "configuration": vars(args) | {"output": str(args.output or "")},
        "aggregate": aggregate,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
