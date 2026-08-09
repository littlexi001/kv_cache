#!/usr/bin/env python
"""Aggregate GPU-rotated autoregressive generation-speed runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ORDER = (
    "full",
    "qksieve_no_value_top1280",
    "fier_rtn1_g32_top1280",
    "fier_rtn1_g32_top512",
)


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in ORDER}
    by_round: dict[str, dict[str, dict[str, Any]]] = {}
    by_gpu: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(args.run_root.glob("round*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload["method"])
        if method not in grouped:
            continue
        grouped[method].append(payload)
        by_round.setdefault(path.parent.name, {})[method] = payload
        round_match = re.fullmatch(r"round(\d+)", path.parent.name)
        if round_match is not None:
            round_index = int(round_match.group(1))
            physical_gpu = (ORDER.index(method) + round_index - 1) % len(ORDER)
            by_gpu.setdefault(physical_gpu, {})[method] = payload
    if any(not grouped[name] for name in ORDER):
        missing = [name for name in ORDER if not grouped[name]]
        raise RuntimeError(f"missing methods: {missing}")

    full_steady = median(grouped["full"], "steady_mean_ms_per_token")
    complete_gpus = sorted(
        gpu for gpu, rows in by_gpu.items() if set(rows) == set(ORDER)
    )
    horizons = sorted(
        int(value)
        for value in grouped["full"][0]["horizons"]
    )
    rows = []
    for method in ORDER:
        runs = grouped[method]
        steady_ms = median(runs, "steady_mean_ms_per_token")
        paired_steady_speedups = [
            float(by_gpu[gpu]["full"]["steady_mean_ms_per_token"])
            / float(by_gpu[gpu][method]["steady_mean_ms_per_token"])
            for gpu in complete_gpus
        ]
        latency_ratios_vs_qksieve = [
            float(by_gpu[gpu][method]["steady_mean_ms_per_token"])
            / float(
                by_gpu[gpu]["qksieve_no_value_top1280"][
                    "steady_mean_ms_per_token"
                ]
            )
            for gpu in complete_gpus
        ]
        horizon_rows = {}
        for horizon in horizons:
            key = str(horizon)
            method_ms = statistics.median(
                float(row["horizons"][key]["ms_per_generated_token_including_prebuild"])
                for row in runs
            )
            full_ms = statistics.median(
                float(row["horizons"][key]["ms_per_generated_token_including_prebuild"])
                for row in grouped["full"]
            )
            paired_horizon_speedups = [
                float(
                    by_gpu[gpu]["full"]["horizons"][key][
                        "ms_per_generated_token_including_prebuild"
                    ]
                )
                / float(
                    by_gpu[gpu][method]["horizons"][key][
                        "ms_per_generated_token_including_prebuild"
                    ]
                )
                for gpu in complete_gpus
            ]
            latency_ratios_vs_qksieve_horizon = [
                float(
                    by_gpu[gpu][method]["horizons"][key][
                        "ms_per_generated_token_including_prebuild"
                    ]
                )
                / float(
                    by_gpu[gpu]["qksieve_no_value_top1280"]["horizons"][key][
                        "ms_per_generated_token_including_prebuild"
                    ]
                )
                for gpu in complete_gpus
            ]
            horizon_rows[key] = {
                "median_ms_per_token_including_prebuild": method_ms,
                "paired_to_full_speedup": full_ms / method_ms,
                "same_gpu_speedups_vs_full": paired_horizon_speedups,
                "same_gpu_speedup_vs_full_median": (
                    statistics.median(paired_horizon_speedups)
                    if paired_horizon_speedups
                    else None
                ),
                "same_gpu_latency_ratio_vs_qksieve_median": (
                    statistics.median(latency_ratios_vs_qksieve_horizon)
                    if latency_ratios_vs_qksieve_horizon
                    else None
                ),
            }
        rows.append(
            {
                "method": method,
                "repeats": len(runs),
                "steady_ms_per_token_median": steady_ms,
                "steady_tokens_per_second": 1000.0 / steady_ms,
                "steady_speedup_vs_full": full_steady / steady_ms,
                "same_gpu_speedups_vs_full": paired_steady_speedups,
                "same_gpu_speedup_vs_full_median": (
                    statistics.median(paired_steady_speedups)
                    if paired_steady_speedups
                    else None
                ),
                "same_gpu_speedup_vs_full_geomean": (
                    geometric_mean(paired_steady_speedups)
                    if paired_steady_speedups
                    else None
                ),
                "same_gpu_latency_ratios_vs_qksieve": latency_ratios_vs_qksieve,
                "same_gpu_latency_ratio_vs_qksieve_median": (
                    statistics.median(latency_ratios_vs_qksieve)
                    if latency_ratios_vs_qksieve
                    else None
                ),
                "same_gpu_latency_ratio_vs_qksieve_geomean": (
                    geometric_mean(latency_ratios_vs_qksieve)
                    if latency_ratios_vs_qksieve
                    else None
                ),
                "prebuild_seconds_median": median(runs, "prebuild_wall_seconds"),
                "first_step_ms_median": median(runs, "first_step_ms"),
                "generated_hashes": sorted(
                    {str(row["generated_token_sha256"]) for row in runs}
                ),
                "horizons": horizon_rows,
            }
        )

    agreement = []
    for round_name, round_rows in sorted(by_round.items()):
        if "full" not in round_rows:
            continue
        full_tokens = round_rows["full"]["generated_token_ids"]
        for method in ORDER[1:]:
            if method not in round_rows:
                continue
            candidate = round_rows[method]["generated_token_ids"]
            agreement.append(
                {
                    "round": round_name,
                    "method": method,
                    "longest_common_prefix_tokens": lcp(full_tokens, candidate),
                    "positionwise_token_agreement": sum(
                        int(a == b) for a, b in zip(full_tokens, candidate)
                    )
                    / min(len(full_tokens), len(candidate)),
                }
            )

    result = {
        "schema": "qksieve_fier_autoregressive_speed_summary_v1",
        "history_tokens": grouped["full"][0]["history_tokens"],
        "generation_steps": grouped["full"][0]["generation_steps"],
        "complete_physical_gpu_pairs": complete_gpus,
        "rows": rows,
        "sequence_agreement": agreement,
    }
    (args.run_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
