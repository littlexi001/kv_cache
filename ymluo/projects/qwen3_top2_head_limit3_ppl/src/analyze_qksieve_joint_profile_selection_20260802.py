#!/usr/bin/env python
"""Select joint Key/Value profiles on calibration queries and test later rows."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one float")
    return values


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float]:
    materialized = list(values)
    return {
        "mean": sum(materialized) / len(materialized),
        "p50": quantile(materialized, 0.50),
        "p90": quantile(materialized, 0.90),
        "p99": quantile(materialized, 0.99),
        "maximum": max(materialized),
    }


def statistic(values: list[float], name: str) -> float:
    if name == "mean":
        return sum(values) / len(values)
    if name == "rms":
        return math.sqrt(sum(value * value for value in values) / len(values))
    if name == "p90":
        return quantile(values, 0.90)
    if name == "maximum":
        return max(values)
    raise ValueError(f"unsupported statistic {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument(
        "--calibration_statistics", default="mean,rms,p90,maximum"
    )
    parser.add_argument("--tolerances", default="0.02,0.03,0.05,0.075,0.1")
    parser.add_argument("--moment_cost_weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(args.input_glob)
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration fraction must lie in (0, 1)")
    tolerances = parse_floats(args.tolerances)
    statistic_names = tuple(
        value.strip()
        for value in args.calibration_statistics.split(",")
        if value.strip()
    )

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(
                row
                for row in csv.DictReader(handle)
                if row["method"] == "conditional_block_moment"
                and abs(float(row["fraction"]) - args.fraction) < 1.0e-12
            )
    if not rows:
        raise RuntimeError("no practical rows matched the requested fraction")

    head_key = lambda row: (
        row["trace"],
        int(row["layer"]),
        int(row["kv_head"]),
    )
    option_key = lambda row: (row["key_profile"], row["moment_profile"])
    rows_by_head: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_head[head_key(row)].append(row)

    results: list[dict[str, Any]] = []
    for statistic_name in statistic_names:
        for tolerance in tolerances:
            selected_rows: list[dict[str, str]] = []
            choices: list[dict[str, Any]] = []
            for current_head, head_rows in rows_by_head.items():
                steps = sorted({int(row["step"]) for row in head_rows})
                split = max(1, min(len(steps) - 1, math.ceil(
                    args.calibration_fraction * len(steps)
                )))
                calibration_steps = set(steps[:split])
                heldout_steps = set(steps[split:])
                by_option: dict[
                    tuple[str, str], list[dict[str, str]]
                ] = defaultdict(list)
                for row in head_rows:
                    by_option[option_key(row)].append(row)

                candidates: list[tuple[float, float, tuple[str, str]]] = []
                all_options: list[tuple[float, float, tuple[str, str]]] = []
                for option, option_rows in by_option.items():
                    calibration_errors = [
                        float(row["relative_l2"])
                        for row in option_rows
                        if int(row["step"]) in calibration_steps
                    ]
                    calibration_error = statistic(
                        calibration_errors, statistic_name
                    )
                    representative = option_rows[0]
                    key_bits = float(representative["key_bits_per_token"])
                    moment_bits = float(
                        representative["moment_bits_per_token"]
                    )
                    weighted_rate = (
                        key_bits + args.moment_cost_weight * moment_bits
                    )
                    item = (weighted_rate, calibration_error, option)
                    all_options.append(item)
                    if calibration_error <= tolerance:
                        candidates.append(item)

                met = bool(candidates)
                if met:
                    weighted_rate, calibration_error, selected = min(candidates)
                else:
                    weighted_rate, calibration_error, selected = min(
                        all_options,
                        key=lambda item: (item[1], item[0]),
                    )
                option_rows = by_option[selected]
                heldout = [
                    row
                    for row in option_rows
                    if int(row["step"]) in heldout_steps
                ]
                selected_rows.extend(heldout)
                representative = option_rows[0]
                choices.append(
                    {
                        "trace": current_head[0],
                        "layer": current_head[1],
                        "kv_head": current_head[2],
                        "met_tolerance": met,
                        "key_profile": selected[0],
                        "moment_profile": selected[1],
                        "calibration_error": calibration_error,
                        "key_bits_per_token": float(
                            representative["key_bits_per_token"]
                        ),
                        "moment_bits_per_token": float(
                            representative["moment_bits_per_token"]
                        ),
                        "weighted_rate": weighted_rate,
                        "aux_ratio": float(
                            representative["total_aux_ratio_of_full_kv"]
                        ),
                    }
                )

            error_summary = summarize(
                float(row["relative_l2"]) for row in selected_rows
            )
            regret_summary = summarize(
                float(row["oracle_regret_relative_l2"])
                for row in selected_rows
            )
            key_rate_summary = summarize(
                choice["key_bits_per_token"] for choice in choices
            )
            moment_rate_summary = summarize(
                choice["moment_bits_per_token"] for choice in choices
            )
            aux_summary = summarize(choice["aux_ratio"] for choice in choices)
            profile_distribution = Counter(
                f"{choice['key_profile']}+{choice['moment_profile']}"
                for choice in choices
            )
            results.append(
                {
                    "calibration_statistic": statistic_name,
                    "tolerance": tolerance,
                    "heads": len(choices),
                    "heldout_query_heads": len(selected_rows),
                    "met_tolerance_rate": sum(
                        choice["met_tolerance"] for choice in choices
                    )
                    / len(choices),
                    "profile_distribution": dict(
                        profile_distribution.most_common()
                    ),
                    **{
                        f"heldout_relative_l2_{name}": value
                        for name, value in error_summary.items()
                    },
                    **{
                        f"heldout_oracle_regret_{name}": value
                        for name, value in regret_summary.items()
                    },
                    **{
                        f"key_bits_per_token_{name}": value
                        for name, value in key_rate_summary.items()
                    },
                    **{
                        f"moment_bits_per_token_{name}": value
                        for name, value in moment_rate_summary.items()
                    },
                    **{
                        f"aux_ratio_{name}": value
                        for name, value in aux_summary.items()
                    },
                    "choices": choices,
                }
            )

    report = {
        "schema": "qksieve_joint_profile_selection_v1",
        "inputs": [str(path) for path in paths],
        "contract": {
            "fraction": args.fraction,
            "calibration_fraction_of_already_heldout_steps": (
                args.calibration_fraction
            ),
            "basis_queries_are_disjoint_from_both_halves": True,
            "full_fallback": False,
            "router": False,
            "selection_granularity": "request-layer-KV-head",
            "moment_cost_weight": args.moment_cost_weight,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    concise = [
        {
            key: row[key]
            for key in (
                "calibration_statistic",
                "tolerance",
                "met_tolerance_rate",
                "heldout_relative_l2_mean",
                "heldout_relative_l2_p90",
                "heldout_relative_l2_p99",
                "key_bits_per_token_mean",
                "moment_bits_per_token_mean",
                "aux_ratio_mean",
                "profile_distribution",
            )
        }
        for row in results
    ]
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
