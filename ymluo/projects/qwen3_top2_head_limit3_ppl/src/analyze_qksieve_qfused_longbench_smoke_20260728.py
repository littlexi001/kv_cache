#!/usr/bin/env python
"""Verify that the experimental qfused path really executes in LongBench.

This is a paired execution and quality smoke, not a publishable latency run:
attention diagnostics are enabled so every sparse layer exposes whether the
fused Query preparation path was requested and executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FULL = "full_kv"
FROZEN = "qksieve_fullprompt_auto_plain_fulltopk"
QFUSED = "qksieve_fullprompt_auto_plain_qfused_fulltopk"
EXPECTED_METHODS = (FULL, FROZEN, QFUSED)
FROZEN_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
)
QFUSED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_packed_fulltopk"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--validation_matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_prediction_match", type=float, default=0.875)
    parser.add_argument("--max_mean_score_delta", type=float, default=0.01)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_key(row: dict[str, str]) -> tuple[str, str]:
    return row["task"], row["sample_id"]


def as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name)
    if value in (None, ""):
        raise ValueError(f"missing numeric field {name!r}")
    return float(value)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def analyze(
    rows: list[dict[str, str]],
    validation: dict[str, Any],
    *,
    min_prediction_match: float,
    max_mean_score_delta: float,
) -> dict[str, Any]:
    method_counts = Counter(row["method"] for row in rows)
    unexpected_methods = sorted(set(method_counts) - set(EXPECTED_METHODS))
    by_method = {
        method: {
            sample_key(row): row
            for row in rows
            if row["method"] == method
        }
        for method in EXPECTED_METHODS
    }
    key_sets = [set(by_method[method]) for method in EXPECTED_METHODS]
    paired_keys = set.intersection(*key_sets) if key_sets else set()
    union_keys = set.union(*key_sets) if key_sets else set()
    complete_triplets = (
        not unexpected_methods
        and bool(union_keys)
        and all(keys == union_keys for keys in key_sets)
        and len(rows) == 3 * len(union_keys)
    )

    sparse_contract_errors: list[str] = []
    frozen_rows = by_method[FROZEN]
    qfused_rows = by_method[QFUSED]
    for key in sorted(paired_keys):
        frozen = frozen_rows[key]
        qfused = qfused_rows[key]
        if frozen.get("executed_path") != FROZEN:
            sparse_contract_errors.append(f"{key}: frozen executed_path")
        if qfused.get("executed_path") != QFUSED:
            sparse_contract_errors.append(f"{key}: qfused executed_path")
        if frozen.get("configured_score_mode") != FROZEN_SCORE_MODE:
            sparse_contract_errors.append(f"{key}: frozen score mode")
        if qfused.get("configured_score_mode") != QFUSED_SCORE_MODE:
            sparse_contract_errors.append(f"{key}: qfused score mode")
        if frozen.get("diagnostics_enabled", "").lower() not in {
            "true",
            "1",
        }:
            sparse_contract_errors.append(f"{key}: frozen diagnostics")
        if qfused.get("diagnostics_enabled", "").lower() not in {
            "true",
            "1",
        }:
            sparse_contract_errors.append(f"{key}: qfused diagnostics")
        for field in (
            "configured_attention_fraction",
            "configured_attention_tokens",
            "configured_candidate_fraction",
            "configured_projection_dim",
            "packed_qmse_index_bits_per_token",
        ):
            if as_float(frozen, field) != as_float(qfused, field):
                sparse_contract_errors.append(f"{key}: mismatch {field}")

    frozen_requested = [
        as_float(row, "packed_qmse_fused_query_prepare_requested")
        for row in frozen_rows.values()
    ]
    frozen_executed = [
        as_float(row, "packed_qmse_fused_query_prepare_executed")
        for row in frozen_rows.values()
    ]
    qfused_requested = [
        as_float(row, "packed_qmse_fused_query_prepare_requested")
        for row in qfused_rows.values()
    ]
    qfused_executed = [
        as_float(row, "packed_qmse_fused_query_prepare_executed")
        for row in qfused_rows.values()
    ]
    qfused_frozen_before = [
        as_float(row, "packed_qmse_allocation_frozen_before_query")
        for row in qfused_rows.values()
    ]
    execution_proven = bool(
        qfused_rows
        and max(frozen_requested, default=1.0) == 0.0
        and max(frozen_executed, default=1.0) == 0.0
        and min(qfused_requested, default=0.0) >= 1.0 - 1.0e-6
        and min(qfused_executed, default=0.0) >= 1.0 - 1.0e-6
        and min(qfused_frozen_before, default=0.0) >= 1.0 - 1.0e-6
    )

    prediction_matches = [
        float(frozen_rows[key]["prediction"] == qfused_rows[key]["prediction"])
        for key in sorted(paired_keys)
    ]
    frozen_scores = [
        as_float(frozen_rows[key], "score") for key in sorted(paired_keys)
    ]
    qfused_scores = [
        as_float(qfused_rows[key], "score") for key in sorted(paired_keys)
    ]
    full_scores = [
        as_float(by_method[FULL][key], "score") for key in sorted(paired_keys)
    ]
    prediction_match_rate = mean(prediction_matches) if prediction_matches else 0.0
    mean_score_delta = (
        mean(qfused_scores) - mean(frozen_scores) if paired_keys else 0.0
    )
    quality_passed = bool(
        paired_keys
        and prediction_match_rate >= min_prediction_match
        and abs(mean_score_delta) <= max_mean_score_delta
    )

    timing = {}
    for method in EXPECTED_METHODS:
        method_rows = list(by_method[method].values())
        timing[method] = {
            "mean_query_seconds": (
                mean([as_float(row, "query_seconds") for row in method_rows])
                if method_rows
                else None
            ),
            "mean_decode_seconds": (
                mean([as_float(row, "decode_seconds") for row in method_rows])
                if method_rows
                else None
            ),
            "mean_online_seconds": (
                mean([as_float(row, "online_seconds") for row in method_rows])
                if method_rows
                else None
            ),
        }

    validation_passed = validation.get("all_passed") is True
    promotion_smoke_passed = bool(
        validation_passed
        and complete_triplets
        and not sparse_contract_errors
        and execution_proven
        and quality_passed
    )
    return {
        "schema": "qksieve_qfused_longbench_smoke_v1",
        "validation_matrix_passed": validation_passed,
        "row_count": len(rows),
        "method_counts": dict(sorted(method_counts.items())),
        "paired_triplets": len(paired_keys),
        "tasks": sorted({key[0] for key in paired_keys}),
        "complete_triplets": complete_triplets,
        "sparse_contract_errors": sparse_contract_errors,
        "execution": {
            "frozen_requested_mean": (
                mean(frozen_requested) if frozen_requested else None
            ),
            "frozen_executed_mean": (
                mean(frozen_executed) if frozen_executed else None
            ),
            "qfused_requested_mean": (
                mean(qfused_requested) if qfused_requested else None
            ),
            "qfused_executed_mean": (
                mean(qfused_executed) if qfused_executed else None
            ),
            "qfused_allocation_frozen_before_query_mean": (
                mean(qfused_frozen_before) if qfused_frozen_before else None
            ),
            "proven": execution_proven,
        },
        "quality": {
            "full_mean_score": mean(full_scores) if full_scores else None,
            "frozen_mean_score": mean(frozen_scores) if frozen_scores else None,
            "qfused_mean_score": mean(qfused_scores) if qfused_scores else None,
            "qfused_minus_frozen_mean_score": mean_score_delta,
            "prediction_exact_match_rate": prediction_match_rate,
            "min_prediction_match": min_prediction_match,
            "max_abs_mean_score_delta": max_mean_score_delta,
            "passed": quality_passed,
        },
        "timing_with_diagnostics": timing,
        "timing_is_promotion_evidence": False,
        "promotion_smoke_passed": promotion_smoke_passed,
    }


def main() -> None:
    args = parse_args()
    validation = json.loads(
        args.validation_matrix.read_text(encoding="utf-8")
    )
    report = analyze(
        read_rows(args.results),
        validation,
        min_prediction_match=args.min_prediction_match,
        max_mean_score_delta=args.max_mean_score_delta,
    )
    report["source_sha256"] = {
        args.results.name: sha256(args.results),
        args.validation_matrix.name: sha256(args.validation_matrix),
        Path(__file__).name: sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["promotion_smoke_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
