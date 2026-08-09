#!/usr/bin/env python
"""Synthetic test of shared GQA output-risk candidate selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    conditional_tail_numerator,
    fit_block_models,
    selected_conditioned_residual_mean,
    tail_statistics,
)
from analyze_qksieve_gaussian_tail_synthetic_20260802 import make_case
from analyze_qksieve_layerwise_rate_distortion_20260802 import (
    conditional_value_leverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--score_dim", type=int, default=16)
    parser.add_argument("--conditional_dim", type=int, default=8)
    parser.add_argument("--value_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--seeds", default="3,11,29,47,71")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(matrix, ord=2).clamp_min(1.0e-12)


def approximate_layer_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    conditional_coordinates: torch.Tensor,
    conditional_model: dict[str, torch.Tensor | int | float],
    output_projections: torch.Tensor,
    block_size: int,
    balance_selected_residual: bool = False,
) -> torch.Tensor:
    projected = []
    for group in range(scores.shape[0]):
        denominator, weighted_x, _ = tail_statistics(
            scores[group],
            conditional_coordinates,
            values,
            selected,
            block_size,
        )
        numerator = conditional_tail_numerator(
            denominator, weighted_x, conditional_model
        )
        if balance_selected_residual:
            residual_mean = selected_conditioned_residual_mean(
                conditional_coordinates,
                values,
                selected,
                conditional_model,
            )
            numerator = numerator + torch.einsum(
                "b,bd->d", denominator, residual_mean
            )
        head_output = combine_selected_and_tail(
            scores[group],
            scores[group],
            values,
            selected,
            numerator,
            denominator.sum(),
            1.0,
        )
        projected.append(output_projections[group] @ head_output)
    return torch.stack(projected).sum(dim=0)


def evaluate_case(
    case: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    coordinates, values = make_case(
        case,
        args.tokens,
        args.score_dim,
        args.value_dim,
        args.block_size,
        generator,
    )
    conditional_coordinates = coordinates[:, : args.conditional_dim]
    model = fit_block_models(
        conditional_coordinates,
        values,
        args.block_size,
        ridge=0.01,
        moment_bits=8,
        linear_group_blocks=0,
    )
    queries = torch.randn(args.groups, args.score_dim, generator=generator)
    queries /= math.sqrt(args.score_dim)
    invariant_scores = queries @ coordinates.T
    # These offsets must not affect a softmax-aware candidate rule.
    score_offsets = 6.0 * torch.randn(args.groups, 1, generator=generator)
    shifted_scores = invariant_scores + score_offsets
    log_partitions = torch.logsumexp(invariant_scores, dim=-1)
    log_probabilities = invariant_scores - log_partitions[:, None]
    output_projections = torch.randn(
        args.groups, args.output_dim, args.value_dim, generator=generator
    ) / math.sqrt(args.value_dim)
    group_log_gains = torch.stack(
        [spectral_norm(output_projections[group]) for group in range(args.groups)]
    ).log()
    raw_leverage, _ = conditional_value_leverage(
        conditional_coordinates, values, model, bits=16
    )
    int4_leverage, int4_rate = conditional_value_leverage(
        conditional_coordinates, values, model, bits=4
    )
    projection_grams = torch.einsum(
        "god,goe->gde", output_projections, output_projections
    )
    projected_leverage_fp, _ = conditional_value_leverage(
        conditional_coordinates,
        values,
        model,
        bits=16,
        projection_grams=projection_grams,
    )
    projected_leverage_int2, projected_int2_rate = conditional_value_leverage(
        conditional_coordinates,
        values,
        model,
        bits=2,
        projection_grams=projection_grams,
    )
    keep = max(1, math.ceil(args.tokens * args.fraction))
    priorities = {
        "raw_score_max": shifted_scores.amax(dim=0),
        "normalized_probability_max": log_probabilities.amax(dim=0),
        "normalized_probability_sum": torch.logsumexp(
            log_probabilities, dim=0
        ),
        "residual_bound_fp": torch.logsumexp(
            log_probabilities
            + group_log_gains[:, None]
            + raw_leverage[None, :],
            dim=0,
        ),
        "residual_bound_int4": torch.logsumexp(
            log_probabilities
            + group_log_gains[:, None]
            + int4_leverage[None, :],
            dim=0,
        ),
        "residual_bound_fastmax_int4": (
            log_probabilities
            + group_log_gains[:, None]
            + int4_leverage[None, :]
        ).amax(dim=0),
        "projected_residual_bound_fp": torch.logsumexp(
            log_probabilities + projected_leverage_fp,
            dim=0,
        ),
        "projected_residual_bound_int2": torch.logsumexp(
            log_probabilities + projected_leverage_int2,
            dim=0,
        ),
        "projected_residual_fastmax_int2": (
            log_probabilities + projected_leverage_int2
        ).amax(dim=0),
    }
    full_heads = torch.softmax(invariant_scores, dim=-1) @ values
    full_output = torch.einsum("god,gd->o", output_projections, full_heads)
    errors = {}
    selected_sets = {}
    for name, priority in priorities.items():
        selected = torch.topk(priority, keep, sorted=False).indices
        for balanced in (False, True):
            approximate = approximate_layer_output(
                invariant_scores,
                values,
                selected,
                conditional_coordinates,
                model,
                output_projections,
                args.block_size,
                balance_selected_residual=balanced,
            )
            label = f"{name}_balanced" if balanced else name
            errors[label] = float(
                torch.linalg.vector_norm(approximate - full_output)
                / torch.linalg.vector_norm(full_output).clamp_min(1.0e-8)
            )
        selected_sets[name] = selected
    invariant_priority = invariant_scores.amax(dim=0)
    invariant_selected = torch.topk(invariant_priority, keep).indices
    raw_selected = selected_sets["raw_score_max"]
    raw_shift_jaccard = float(
        torch.isin(raw_selected, invariant_selected).float().mean()
    )
    return {
        "case": case,
        "seed": seed,
        "relative_layer_output_error": errors,
        "raw_max_shift_overlap": raw_shift_jaccard,
        "int4_leverage_bits_per_token": int4_rate,
        "projected_int2_leverage_bits_per_token": projected_int2_rate,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    cases = (
        "gaussian_linear",
        "block_shift",
        "student_t",
        "mixture_outlier",
        "nonlinear_value",
    )
    rows = [
        evaluate_case(case, seed, args) for case in cases for seed in seeds
    ]
    methods = tuple(rows[0]["relative_layer_output_error"])
    aggregate = {
        method: {
            "mean": sum(
                row["relative_layer_output_error"][method] for row in rows
            )
            / len(rows),
            "worst": max(
                row["relative_layer_output_error"][method] for row in rows
            ),
        }
        for method in methods
    }
    payload = {
        "configuration": vars(args) | {"output": str(args.output)},
        "aggregate": aggregate,
        "raw_max_shift_overlap_mean": sum(
            row["raw_max_shift_overlap"] for row in rows
        )
        / len(rows),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "aggregate": aggregate,
        "raw_max_shift_overlap_mean": payload["raw_max_shift_overlap_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
