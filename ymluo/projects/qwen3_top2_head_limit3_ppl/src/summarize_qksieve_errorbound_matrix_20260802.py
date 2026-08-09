#!/usr/bin/env python
"""Summarize QKSieve output-bound and Gaussian-tail matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def maximum(values: list[float]) -> float:
    return max(values) if values else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.run_root.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if not results:
            continue
        test_errors: list[float] = []
        calibration_errors: list[float] = []
        auxiliary_ratios: list[float] = []
        exact_errors: list[float] = []
        negative_blocks: list[float] = []
        mass_deficits: list[float] = []
        repaired_blocks: list[float] = []
        partition_errors: list[float] = []
        for result in results:
            uniforms = result.get("uniform_profiles", [])
            if len(uniforms) != 1:
                raise ValueError(
                    f"{path} must contain exactly one uniform profile per layer"
                )
            uniform = uniforms[0]
            test_errors.append(float(uniform["test_relative_l2"]))
            calibration_errors.append(
                float(uniform["calibration_relative_l2"])
            )
            auxiliary_ratios.append(float(uniform["aux_ratio"]))
            exact_errors.append(float(result["exact_topk"]["test_relative_l2"]))
            for key, target in (
                ("gaussian_negative_block_fraction", negative_blocks),
                ("gaussian_selected_mass_deficit_ratio", mass_deficits),
                ("gaussian_repaired_block_fraction", repaired_blocks),
            ):
                summary = result.get(key, {})
                if summary:
                    target.append(float(summary.get("mean", 0.0)))
            partition = result.get("sampled_log_partition_error", {})
            if partition:
                partition_errors.append(abs(float(partition.get("mean", 0.0))))
        contract = payload.get("contract", {})
        rows.append(
            {
                "run": path.stem,
                "key_profile": results[0]["uniform_profiles"][0]["profile"].split(
                    "+", 1
                )[0],
                "candidate_policy": contract.get("candidate_policy"),
                "tail_estimator": contract.get("tail_estimator", "token_exact"),
                "selection_priority": contract.get("selection_priority"),
                "value_leverage_space": contract.get("value_leverage_space"),
                "value_leverage_bits": contract.get("value_leverage_bits"),
                "output_group_gain": contract.get("output_group_gain"),
                "test_output_relative_l2_mean": mean(test_errors),
                "test_output_relative_l2_worst": maximum(test_errors),
                "calibration_output_relative_l2_mean": mean(calibration_errors),
                "exact_topk_relative_l2_mean": mean(exact_errors),
                "aux_ratio_mean": mean(auxiliary_ratios),
                "gaussian_negative_block_fraction_mean": mean(negative_blocks),
                "gaussian_mass_deficit_ratio_mean": mean(mass_deficits),
                "gaussian_repaired_block_fraction_mean": mean(
                    repaired_blocks
                ),
                "sampled_log_partition_abs_error_mean": mean(partition_errors),
                "layer_trace_count": len(results),
            }
        )
    rows.sort(
        key=lambda row: (
            row["test_output_relative_l2_mean"],
            row["aux_ratio_mean"],
            row["run"],
        )
    )
    output_json = args.run_root / "summary.json"
    output_csv = args.run_root / "summary.csv"
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
