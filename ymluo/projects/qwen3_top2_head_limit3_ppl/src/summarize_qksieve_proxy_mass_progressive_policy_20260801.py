#!/usr/bin/env python
"""Summarize the training-free proxy-mass Value-rank policy.

The input is the per-head CSV emitted by
``analyze_qksieve_mass_adaptive_value_rank_20260801.py``.  No model is
loaded: this script only evaluates deterministic policies on frozen traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tolerances",
        default="0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1",
    )
    return parser.parse_args()


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p99": quantile(values, 0.99),
        "maximum": max(values),
    }


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / max(left_norm * right_norm, 1.0e-30)


def choose_rank(row: dict[str, str], mass_field: str, tolerance: float) -> int:
    omitted_mass = 1.0 - float(row[mass_field])
    rank8_residual = float(row["rank8_residual_ratio"])
    return 8 if omitted_mass * rank8_residual <= tolerance else 32


def policy_summary(
    rows: list[dict[str, str]], mass_field: str, tolerance: float
) -> dict[str, object]:
    ranks = [choose_rank(row, mass_field, tolerance) for row in rows]
    errors = [
        float(row[f"rank{rank}_relative_l2"])
        for row, rank in zip(rows, ranks)
    ]
    return {
        "mean_rank": sum(ranks) / len(ranks),
        "rank8_cases": ranks.count(8),
        "rank32_cases": ranks.count(32),
        "relative_l2": distribution(errors),
    }


def main() -> None:
    args = parse_args()
    tolerances = [float(item) for item in args.tolerances.split(",")]
    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV contains no rows")

    proxy_mass = [float(row["proxy_mass"]) for row in rows]
    exact_mass = [float(row["exact_mass"]) for row in rows]
    sample_mass = [float(row["sample_mass"]) for row in rows]
    rank8_error = [float(row["rank8_relative_l2"]) for row in rows]
    rank32_error = [float(row["rank32_relative_l2"]) for row in rows]
    risk_score = [
        (1.0 - proxy) * float(row["rank8_residual_ratio"])
        for row, proxy in zip(rows, proxy_mass)
    ]
    ordered_risk = sorted(
        zip(risk_score, rank8_error, rank32_error), key=lambda item: item[0]
    )
    risk_deciles = []
    for decile in range(10):
        start = decile * len(rows) // 10
        stop = (decile + 1) * len(rows) // 10
        bucket = ordered_risk[start:stop]
        risk_deciles.append(
            {
                "decile": decile + 1,
                "cases": len(bucket),
                "risk_mean": sum(item[0] for item in bucket) / len(bucket),
                "rank8_relative_l2_mean": (
                    sum(item[1] for item in bucket) / len(bucket)
                ),
                "rank32_relative_l2_mean": (
                    sum(item[2] for item in bucket) / len(bucket)
                ),
                "rank32_gain_mean": (
                    sum(item[1] - item[2] for item in bucket) / len(bucket)
                ),
            }
        )
    payload: dict[str, object] = {
        "schema": "qksieve_proxy_mass_progressive_policy_v1",
        "input_csv": str(args.input_csv),
        "cases": len(rows),
        "policy": (
            "rank8 iff (1 - proxy_selected_mass) * "
            "rank8_normalized_value_residual <= tolerance; else rank32"
        ),
        "mass_signal": {
            "proxy_vs_exact_pearson": pearson(proxy_mass, exact_mass),
            "sample_vs_exact_pearson": pearson(sample_mass, exact_mass),
            "proxy_minus_exact": distribution(
                [proxy - exact for proxy, exact in zip(proxy_mass, exact_mass)]
            ),
            "sample_minus_exact": distribution(
                [sample - exact for sample, exact in zip(sample_mass, exact_mass)]
            ),
        },
        "risk_signal": {
            "definition": (
                "(1 - proxy_selected_mass) * rank8_normalized_value_residual"
            ),
            "risk_vs_rank8_error_pearson": pearson(risk_score, rank8_error),
            "risk_vs_rank32_gain_pearson": pearson(
                risk_score,
                [left - right for left, right in zip(rank8_error, rank32_error)],
            ),
            "deciles": risk_deciles,
        },
        "fixed": {
            f"rank{rank}": distribution(
                [float(row[f"rank{rank}_relative_l2"]) for row in rows]
            )
            for rank in (8, 32)
        },
        "sweep": {},
    }

    for tolerance in tolerances:
        key = f"tau_{tolerance:g}"
        proxy = policy_summary(rows, "proxy_mass", tolerance)
        exact = policy_summary(rows, "exact_mass", tolerance)
        proxy_ranks = [
            choose_rank(row, "proxy_mass", tolerance) for row in rows
        ]
        exact_ranks = [
            choose_rank(row, "exact_mass", tolerance) for row in rows
        ]
        proxy["decision_agreement_with_exact_mass"] = sum(
            left == right for left, right in zip(proxy_ranks, exact_ranks)
        ) / len(rows)
        proxy["unsafe_rank8_vs_exact_mass"] = sum(
            left == 8 and right == 32
            for left, right in zip(proxy_ranks, exact_ranks)
        )
        proxy["unnecessary_rank32_vs_exact_mass"] = sum(
            left == 32 and right == 8
            for left, right in zip(proxy_ranks, exact_ranks)
        )
        payload["sweep"][key] = {
            "tolerance": tolerance,
            "proxy_mass_policy": proxy,
            "exact_mass_diagnostic": exact,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
