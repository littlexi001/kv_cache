#!/usr/bin/env python3
"""Estimate wall-clock upper bound for batched PCIC-CR sentinel probes.

Uses existing conffast_s8 CSVs. Current implementation charges sentinel gate
seconds as the sum of all non-selected candidate prefix runs. A batch-row budget
implementation would ideally charge roughly one prefix run instead of N prefix
runs for N candidates. This script estimates several conservative speed models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = [
    ("War off8192", "war", "off8192_b4"),
    ("War off16384", "war", "off16384_b4"),
    ("War off24576", "war", "off24576_b4"),
    ("War off32768", "war", "off32768_b4"),
    ("Monte off8192", "monte", "off8192_b4"),
    ("Monte off16384", "monte", "off16384_b4"),
    ("Monte off24576", "monte", "off24576_b4"),
    ("Monte off32768", "monte", "off32768_b4"),
    ("War b8", "war", "b8"),
    ("Monte b8", "monte", "b8"),
    ("War eval128", "war", "b4_eval128"),
    ("Monte eval128", "monte", "b4_eval128"),
]


def run_dir(dataset: str, key: str) -> str:
    if key.startswith("off"):
        return f"server_pcic_r3_{dataset}_{key}_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"
    return f"server_pcic_r3_{dataset}_{key}_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    print("| run | blocks | ppl_delta | current_ratio | oracle_batch_ratio | conservative_batch_ratio | triggered | mean_candidates |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    aggregate = []
    for label, dataset, key in RUNS:
        path = args.outputs_root / run_dir(dataset, key) / "pcic_r_blockwise_results.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
        if not evals:
            continue
        ppl_delta = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
        baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
        current_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
        oracle_seconds = 0.0
        conservative_seconds = 0.0
        triggered = 0
        candidate_counts = []
        for row in evals:
            method_seconds = float(row.get("method_seconds") or row.get("seconds") or 0.0)
            rule = json.loads(row.get("rescue_rule") or "{}")
            gate_seconds = float(rule.get("sentinel_gate_seconds") or 0.0)
            selected_prefix = float(rule.get("sentinel_selected_prefix_seconds") or 0.0)
            all_seconds = [float(value) for value in (rule.get("sentinel_all_seconds") or {}).values()]
            if all_seconds:
                triggered += 1
                candidate_counts.append(len(all_seconds))
                ideal_gate = max(all_seconds) - selected_prefix
                conservative_gate = (sum(all_seconds) / len(all_seconds)) - selected_prefix
                oracle_seconds += method_seconds - gate_seconds + max(0.0, ideal_gate)
                conservative_seconds += method_seconds - gate_seconds + max(0.0, conservative_gate)
            else:
                oracle_seconds += method_seconds
                conservative_seconds += method_seconds
        current_ratio = current_seconds / max(baseline_seconds, 1e-9)
        oracle_ratio = oracle_seconds / max(baseline_seconds, 1e-9)
        conservative_ratio = conservative_seconds / max(baseline_seconds, 1e-9)
        mean_candidates = sum(candidate_counts) / max(1, len(candidate_counts))
        aggregate.append((ppl_delta, current_ratio, oracle_ratio, conservative_ratio))
        print(
            f"| {label} | {len(evals)} | {ppl_delta:.6f} | {current_ratio:.3f} | "
            f"{oracle_ratio:.3f} | {conservative_ratio:.3f} | {triggered} | {mean_candidates:.2f} |"
        )

    if aggregate:
        print()
        print("| aggregate | mean_ppl_delta | mean_current_ratio | mean_oracle_batch_ratio | mean_conservative_batch_ratio |")
        print("|---|---:|---:|---:|---:|")
        print(
            f"| all | {sum(item[0] for item in aggregate) / len(aggregate):.6f} | "
            f"{sum(item[1] for item in aggregate) / len(aggregate):.3f} | "
            f"{sum(item[2] for item in aggregate) / len(aggregate):.3f} | "
            f"{sum(item[3] for item in aggregate) / len(aggregate):.3f} |"
        )


if __name__ == "__main__":
    main()
