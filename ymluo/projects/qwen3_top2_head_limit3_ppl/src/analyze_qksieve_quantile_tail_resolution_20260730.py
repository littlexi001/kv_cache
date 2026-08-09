#!/usr/bin/env python
"""Analyze sampled-quantile rank resolution without model assumptions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tail_anchors", type=float, default=16.0)
    parser.add_argument("--underselect_delta", type=float, default=0.05)
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072,262080,524256",
    )
    return parser.parse_args()


def target_count(history_count: int) -> int:
    return min(
        history_count,
        1280,
        max(256, math.ceil(0.06 * history_count)),
    )


def sample_count(
    history_count: int,
    selected_fraction: float,
    tail_anchors: float,
    maximum: int,
) -> int:
    return min(
        history_count,
        maximum,
        max(256, math.ceil(tail_anchors / selected_fraction)),
    )


def binomial_tail(n: int, probability: float, minimum: int) -> float:
    if minimum <= 0:
        return 1.0
    if minimum > n:
        return 0.0
    log_term = (
        math.lgamma(n + 1)
        - math.lgamma(minimum + 1)
        - math.lgamma(n - minimum + 1)
        + minimum * math.log(probability)
        + (n - minimum) * math.log1p(-probability)
    )
    term = math.exp(log_term)
    total = term
    for count in range(minimum, n):
        term *= (
            (n - count)
            / (count + 1)
            * probability
            / (1.0 - probability)
        )
        total += term
        if term < max(1e-300, total * 1e-15):
            break
    return min(1.0, total)


def order_statistic(
    history_count: int,
    selected_fraction: float,
    samples: int,
    underselect_delta: float,
) -> dict[str, float | int]:
    unbiased_rank = max(
        1,
        min(samples, round(selected_fraction * (samples + 1))),
    )
    conservative_rank = unbiased_rank
    while (
        conservative_rank <= samples
        and binomial_tail(
            samples,
            selected_fraction,
            conservative_rank,
        )
        > underselect_delta
    ):
        conservative_rank += 1
    conservative_rank = min(samples, conservative_rank)

    def moments(rank: int) -> tuple[float, float]:
        mean_fraction = rank / (samples + 1)
        variance = (
            rank
            * (samples + 1 - rank)
            / ((samples + 1) ** 2 * (samples + 2))
        )
        return (
            history_count * mean_fraction,
            history_count * math.sqrt(variance),
        )

    unbiased_mean, unbiased_sd = moments(unbiased_rank)
    conservative_mean, conservative_sd = moments(conservative_rank)
    return {
        "unbiased_rank": unbiased_rank,
        "unbiased_expected_candidates": unbiased_mean,
        "unbiased_candidate_sd": unbiased_sd,
        "unbiased_underselect_probability": binomial_tail(
            samples,
            selected_fraction,
            unbiased_rank,
        ),
        "conservative_rank": conservative_rank,
        "conservative_expected_candidates": conservative_mean,
        "conservative_candidate_sd": conservative_sd,
        "conservative_underselect_probability": binomial_tail(
            samples,
            selected_fraction,
            conservative_rank,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.tail_anchors <= 0.0:
        raise ValueError("tail_anchors must be positive")
    if not 0.0 < args.underselect_delta < 0.5:
        raise ValueError("underselect_delta must be in (0, 0.5)")
    lengths = [int(value) for value in args.lengths.split(",")]
    rows = []
    for history_count in lengths:
        budget = target_count(history_count)
        selected_fraction = budget / history_count
        for profile, maximum in (("old_cap2048", 2048), ("new_cap8192", 8192)):
            samples = sample_count(
                history_count,
                selected_fraction,
                args.tail_anchors,
                maximum,
            )
            rows.append(
                {
                    "history_tokens": history_count,
                    "target_tokens": budget,
                    "target_fraction": selected_fraction,
                    "profile": profile,
                    "sample_count": samples,
                    "expected_target_tail_samples": (
                        samples * selected_fraction
                    ),
                    **order_statistic(
                        history_count,
                        selected_fraction,
                        samples,
                        args.underselect_delta,
                    ),
                }
            )
    payload = {
        "schema": "qksieve_quantile_tail_resolution_v1",
        "tail_anchors": args.tail_anchors,
        "underselect_delta": args.underselect_delta,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
