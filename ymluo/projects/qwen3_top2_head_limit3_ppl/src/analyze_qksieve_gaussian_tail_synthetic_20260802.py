#!/usr/bin/env python
"""Synthetic stress test for conditional and exponential-tilt KV tails."""

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
    fit_gaussian_tilt_moments,
    gaussian_tilt_tail_statistics,
    gaussian_tilt_tail_statistics_hybrid,
    gaussian_tilt_tail_statistics_selected_conditioned,
    tail_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--score_dim", type=int, default=16)
    parser.add_argument("--conditional_dim", type=int, default=8)
    parser.add_argument("--value_dim", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--seeds", default="3,11,29")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_case(
    case: str,
    token_count: int,
    score_dim: int,
    value_dim: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_count = math.ceil(token_count / block_size)
    if case == "student_t":
        numerator = torch.randn(token_count, score_dim, generator=generator)
        denominator = torch.randn(token_count, 3, generator=generator).square().sum(-1)
        x = numerator / (denominator / 3.0).sqrt()[:, None]
        x = x / math.sqrt(3.0)
    else:
        x = torch.randn(token_count, score_dim, generator=generator)
    if case in ("block_shift", "mixture_outlier"):
        shifts = 0.55 * torch.randn(block_count, score_dim, generator=generator)
        block_ids = torch.arange(token_count) // block_size
        x = x + shifts.index_select(0, block_ids)
    if case == "mixture_outlier":
        outlier_count = max(1, token_count // 100)
        positions = torch.randperm(token_count, generator=generator)[:outlier_count]
        directions = torch.randn(outlier_count, score_dim, generator=generator)
        directions = directions / torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        x[positions] += 5.0 * directions

    linear = torch.randn(value_dim, score_dim, generator=generator) / math.sqrt(
        score_dim
    )
    value = x @ linear.T
    value += 0.25 * torch.randn(token_count, value_dim, generator=generator)
    if case == "nonlinear_value":
        nonlinear = torch.randn(
            value_dim, score_dim, generator=generator
        ) / math.sqrt(score_dim)
        value += 0.2 * (x.square() - 1.0) @ nonlinear.T
    return x.float(), value.float()


def relative_energy(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1.0e-8)
    )


def evaluate_case(
    case: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    x, value = make_case(
        case,
        args.tokens,
        args.score_dim,
        args.value_dim,
        args.block_size,
        generator,
    )
    conditional_x = x[:, : args.conditional_dim]
    conditional_model = fit_block_models(
        conditional_x,
        value,
        args.block_size,
        ridge=0.01,
        moment_bits=8,
        linear_group_blocks=0,
    )
    gaussian_models = {
        mode: fit_gaussian_tilt_moments(
            x,
            args.block_size,
            moment_bits=8,
            covariance_mode=mode,
        )
        for mode in ("diag", "full")
    }
    queries = torch.randn(args.groups, args.score_dim, generator=generator)
    queries = queries / math.sqrt(args.score_dim)
    keep = max(1, math.ceil(args.tokens * args.fraction))
    block_ids = torch.arange(args.tokens) // args.block_size
    full_outputs = []
    candidate_only_outputs = []
    mean_tail_outputs = []
    conditional_outputs = []
    gaussian_outputs = {"diag": [], "full": []}
    conditioned_outputs = {"diag": [], "full": []}
    hybrid_outputs = {"diag": [], "full": []}
    gaussian_diagnostics = {"diag": [], "full": []}
    mean_v = conditional_model["mean_v"]
    assert isinstance(mean_v, torch.Tensor)

    for query in queries:
        scores = x @ query
        selected = torch.topk(scores, keep, sorted=False).indices
        full_outputs.append(torch.softmax(scores, dim=0) @ value)
        candidate_only_outputs.append(
            torch.softmax(scores.index_select(0, selected), dim=0)
            @ value.index_select(0, selected)
        )
        tail_denominator, weighted_x, _ = tail_statistics(
            scores,
            conditional_x,
            value,
            selected,
            args.block_size,
        )
        conditional_numerator = conditional_tail_numerator(
            tail_denominator, weighted_x, conditional_model
        )
        conditional_outputs.append(
            combine_selected_and_tail(
                scores,
                scores,
                value,
                selected,
                conditional_numerator,
                tail_denominator.sum(),
                1.0,
            )
        )
        threshold = scores.index_select(0, selected).amin()
        omitted = torch.ones(args.tokens, dtype=torch.bool)
        omitted[selected] = False
        omitted_weights = torch.exp(scores - threshold).masked_fill(~omitted, 0.0)
        block_denominator = torch.zeros(
            int(conditional_model["block_count"]), dtype=torch.float32
        )
        block_denominator.index_add_(0, block_ids, omitted_weights)
        mean_numerator = torch.einsum("b,bd->d", block_denominator, mean_v)
        mean_tail_outputs.append(
            combine_selected_and_tail(
                scores,
                scores,
                value,
                selected,
                mean_numerator,
                block_denominator.sum(),
                1.0,
            )
        )
        for mode, gaussian_model in gaussian_models.items():
            denominator, gaussian_x, diagnostics = gaussian_tilt_tail_statistics(
                scores,
                query,
                0.0,
                conditional_x,
                selected,
                gaussian_model,
            )
            numerator = conditional_tail_numerator(
                denominator, gaussian_x, conditional_model
            )
            gaussian_outputs[mode].append(
                combine_selected_and_tail(
                    scores,
                    scores,
                    value,
                    selected,
                    numerator,
                    denominator.sum(),
                    1.0,
                )
            )
            gaussian_diagnostics[mode].append(diagnostics)
            conditioned_denominator, conditioned_x, conditioned_diagnostics = (
                gaussian_tilt_tail_statistics_selected_conditioned(
                    scores,
                    query,
                    0.0,
                    x,
                    conditional_x,
                    selected,
                    gaussian_model,
                )
            )
            conditioned_numerator = conditional_tail_numerator(
                conditioned_denominator,
                conditioned_x,
                conditional_model,
            )
            conditioned_outputs[mode].append(
                combine_selected_and_tail(
                    scores,
                    scores,
                    value,
                    selected,
                    conditioned_numerator,
                    conditioned_denominator.sum(),
                    1.0,
                )
            )
            gaussian_diagnostics[f"{mode}_conditioned"] = (
                gaussian_diagnostics.get(f"{mode}_conditioned", [])
            )
            gaussian_diagnostics[f"{mode}_conditioned"].append(
                conditioned_diagnostics
            )
            hybrid_denominator, hybrid_x, hybrid_diagnostics = (
                gaussian_tilt_tail_statistics_hybrid(
                    scores,
                    query,
                    0.0,
                    x,
                    conditional_x,
                    selected,
                    gaussian_model,
                )
            )
            hybrid_numerator = conditional_tail_numerator(
                hybrid_denominator, hybrid_x, conditional_model
            )
            hybrid_outputs[mode].append(
                combine_selected_and_tail(
                    scores,
                    scores,
                    value,
                    selected,
                    hybrid_numerator,
                    hybrid_denominator.sum(),
                    1.0,
                )
            )
            gaussian_diagnostics[f"{mode}_hybrid"] = (
                gaussian_diagnostics.get(f"{mode}_hybrid", [])
            )
            gaussian_diagnostics[f"{mode}_hybrid"].append(hybrid_diagnostics)

    reference = torch.stack(full_outputs)
    outputs = {
        "candidate_only": torch.stack(candidate_only_outputs),
        "block_mean_tail": torch.stack(mean_tail_outputs),
        "conditional_token_tail": torch.stack(conditional_outputs),
        "gaussian_diag_tail": torch.stack(gaussian_outputs["diag"]),
        "gaussian_full_tail": torch.stack(gaussian_outputs["full"]),
        "gaussian_diag_conditioned_tail": torch.stack(
            conditioned_outputs["diag"]
        ),
        "gaussian_full_conditioned_tail": torch.stack(
            conditioned_outputs["full"]
        ),
        "gaussian_diag_hybrid_tail": torch.stack(hybrid_outputs["diag"]),
        "gaussian_full_hybrid_tail": torch.stack(hybrid_outputs["full"]),
    }
    return {
        "case": case,
        "seed": seed,
        "relative_output_error": {
            name: relative_energy(output, reference)
            for name, output in outputs.items()
        },
        "gaussian_diagnostics": gaussian_diagnostics,
        "rate_bits_per_token": {
            "conditional": float(conditional_model["moment_bits_per_token"]),
            **{
                f"gaussian_{mode}": float(model["moment_bits_per_token"])
                for mode, model in gaussian_models.items()
            },
        },
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
    methods = tuple(rows[0]["relative_output_error"])
    aggregate = {
        method: {
            "mean": sum(
                row["relative_output_error"][method] for row in rows
            )
            / len(rows),
            "worst": max(
                row["relative_output_error"][method] for row in rows
            ),
        }
        for method in methods
    }
    payload = {
        "configuration": {
            "tokens": args.tokens,
            "score_dim": args.score_dim,
            "conditional_dim": args.conditional_dim,
            "value_dim": args.value_dim,
            "groups": args.groups,
            "block_size": args.block_size,
            "fraction": args.fraction,
            "seeds": seeds,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
