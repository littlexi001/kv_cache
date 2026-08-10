#!/usr/bin/env python
"""Verify the fast shrinkage grid against the scalar reference analyzer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


KEYS = (
    "label",
    "layer",
    "heldout_step",
    "kv_head",
    "query_head",
    "method",
    "selected_fraction_target",
)
METRIC_LIMITS = {
    "top2_recall": (0.002, 2.0e-6),
    "selected_attention_mass": (5.0e-4, 2.0e-6),
    "oracle_top2_attention_mass": (2.0e-6, 2.0e-6),
    "top2_attention_mass_recall": (5.0e-4, 2.0e-6),
    "score_pearson": (2.0e-6, 2.0e-6),
    "score_rmse": (2.0e-6, 2.0e-6),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "qk_balanced":
                continue
            key = tuple(row[name] for name in KEYS)
            if key in rows:
                raise AssertionError(f"duplicate qk_balanced condition: {key}")
            rows[key] = row
    if not rows:
        raise AssertionError(f"no qk_balanced rows: {path}")
    return rows


def read_allocations(path: Path) -> dict[tuple[str, str, str], str]:
    rows: dict[tuple[str, str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "qk_balanced":
                continue
            key = (row["layer"], row["kv_head"], row["method"])
            if key in rows:
                raise AssertionError(f"duplicate qk_balanced allocation: {key}")
            rows[key] = row["allocation"]
    if not rows:
        raise AssertionError(f"no qk_balanced allocations: {path}")
    return rows


def verify(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_path = reference_dir / "per_head.csv"
    candidate_path = candidate_dir / "per_head.csv"
    reference = read_rows(reference_path)
    candidate = read_rows(candidate_path)
    if set(reference) != set(candidate):
        raise AssertionError(
            "fast-grid condition keys differ from scalar reference: "
            f"missing={len(set(reference) - set(candidate))}, "
            f"extra={len(set(candidate) - set(reference))}"
        )
    reference_allocations = read_allocations(reference_dir / "allocations.csv")
    candidate_allocations = read_allocations(candidate_dir / "allocations.csv")
    if reference_allocations != candidate_allocations:
        raise AssertionError("fast-grid qMSE allocations differ from reference")

    metrics = {}
    for metric, (maximum_limit, mean_limit) in METRIC_LIMITS.items():
        differences = [
            abs(float(reference[key][metric]) - float(candidate[key][metric]))
            for key in reference
        ]
        maximum = max(differences)
        average = sum(differences) / len(differences)
        if maximum > maximum_limit or average > mean_limit:
            raise AssertionError(
                f"{metric} exceeds equivalence tolerance: "
                f"max={maximum}, mean={average}"
            )
        metrics[metric] = {
            "max_abs_difference": maximum,
            "mean_abs_difference": average,
            "max_tolerance": maximum_limit,
            "mean_tolerance": mean_limit,
        }
    return {
        "schema": "qksieve_shrinkage_fast_grid_equivalence_v1",
        "passed": True,
        "conditions": len(reference),
        "allocation_conditions": len(reference_allocations),
        "condition_keys_identical": True,
        "allocations_identical": True,
        "metrics": metrics,
        "source_sha256": {
            str(reference_path): sha256(reference_path),
            str(candidate_path): sha256(candidate_path),
            str(reference_dir / "allocations.csv"): sha256(
                reference_dir / "allocations.csv"
            ),
            str(candidate_dir / "allocations.csv"): sha256(
                candidate_dir / "allocations.csv"
            ),
        },
        "claim_boundary": (
            "Batched GEMM may exchange one token at a numerical top-k tie; "
            "the rate allocation and paired condition grid must be exact."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True, type=Path)
    parser.add_argument("--candidate_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify(args.reference_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
