#!/usr/bin/env python
"""Summarize paired shrinkage sensitivity under the frozen QKSieve protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)
PRODUCTION_SHRINKAGE = 0.75


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item) for item in text.split(",") if item}))
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def lambda_tag(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise AssertionError("metric values are empty or non-finite")
    return float(array.mean())


def read_run(
    run_root: Path,
    label: str,
    shrinkage: float,
    method: str,
    fractions: tuple[float, ...],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, str]]:
    root = run_root / "analysis" / label / lambda_tag(shrinkage)
    summary_path = root / "summary.json"
    rows_path = root / "per_head.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        raise AssertionError(f"missing shrinkage artifact: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    if not math.isclose(
        float(config.get("query_shrinkage", -1.0)),
        shrinkage,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError(f"shrinkage metadata drifted: {summary_path}")
    if config.get("calibration_source") != "prefill_tail":
        raise AssertionError(f"non-production calibration source: {summary_path}")

    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    with rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != method:
                continue
            fraction = float(row["selected_fraction_target"])
            if not any(math.isclose(fraction, target) for target in fractions):
                continue
            key = (
                label,
                int(row["layer"]),
                int(row["heldout_step"]),
                int(row["kv_head"]),
                int(row["query_head"]),
                fraction,
            )
            if key in rows:
                raise AssertionError(f"duplicate paired condition: {key}")
            parsed = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in parsed.values()):
                raise AssertionError(f"non-finite metric in condition: {key}")
            rows[key] = parsed
    if not rows:
        raise AssertionError(f"no {method} rows in {rows_path}")
    return rows, {
        str(summary_path): sha256(summary_path),
        str(rows_path): sha256(rows_path),
    }


def paired_cluster_interval(
    left: dict[tuple[Any, ...], dict[str, Any]],
    right: dict[tuple[Any, ...], dict[str, Any]],
    metric: str,
    fraction: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    cluster_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key in left:
        if not math.isclose(float(key[-1]), fraction):
            continue
        cluster_values[(str(key[0]), int(key[1]))].append(
            float(left[key][metric]) - float(right[key][metric])
        )
    clusters = np.asarray(
        [mean(values) for values in cluster_values.values()], dtype=np.float64
    )
    if not clusters.size:
        raise AssertionError("paired interval has no clusters")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        clusters,
        size=(bootstrap_samples, clusters.size),
        replace=True,
    ).mean(axis=1)
    return {
        "clusters": int(clusters.size),
        "delta_vs_production": float(clusters.mean()),
        "ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }


def summarize(
    run_root: Path,
    *,
    labels: tuple[str, ...],
    shrinkages: tuple[float, ...],
    fractions: tuple[float, ...],
    method: str = "qk_balanced",
    bootstrap_samples: int = 10000,
    seed: int = 20260810,
) -> dict[str, Any]:
    if PRODUCTION_SHRINKAGE not in shrinkages:
        raise AssertionError("production shrinkage 0.75 is absent")
    all_rows: dict[float, dict[tuple[Any, ...], dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for shrinkage in shrinkages:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for label in labels:
            rows, hashes = read_run(
                run_root, label, shrinkage, method, fractions
            )
            overlap = set(merged).intersection(rows)
            if overlap:
                raise AssertionError(f"duplicate labels in paired rows: {overlap}")
            merged.update(rows)
            sources.update(hashes)
        all_rows[shrinkage] = merged

    expected_keys = set(all_rows[PRODUCTION_SHRINKAGE])
    for shrinkage, rows in all_rows.items():
        if set(rows) != expected_keys:
            missing = len(expected_keys - set(rows))
            extra = len(set(rows) - expected_keys)
            raise AssertionError(
                f"unpaired shrinkage {shrinkage}: missing={missing}, extra={extra}"
            )

    production = all_rows[PRODUCTION_SHRINKAGE]
    aggregate_rows: list[dict[str, Any]] = []
    per_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fraction in fractions:
        for shrinkage in shrinkages:
            rows = all_rows[shrinkage]
            selected = {
                key: value
                for key, value in rows.items()
                if math.isclose(float(key[-1]), fraction)
            }
            row: dict[str, Any] = {
                "shrinkage": shrinkage,
                "selected_fraction": fraction,
                "conditions": len(selected),
            }
            for metric in METRICS:
                row[metric] = mean(value[metric] for value in selected.values())
                row[f"{metric}_paired"] = paired_cluster_interval(
                    rows,
                    production,
                    metric,
                    fraction,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + int(round(10000 * fraction)) + int(100 * shrinkage),
                )
            aggregate_rows.append(row)

            for label in labels:
                label_values = [
                    value
                    for key, value in selected.items()
                    if str(key[0]) == label
                ]
                per_label[label].append(
                    {
                        "shrinkage": shrinkage,
                        "selected_fraction": fraction,
                        "conditions": len(label_values),
                        **{
                            metric: mean(value[metric] for value in label_values)
                            for metric in METRICS
                        },
                    }
                )

    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for fraction in fractions:
        rows = [
            row
            for row in aggregate_rows
            if math.isclose(float(row["selected_fraction"]), fraction)
        ]
        production_row = next(
            row
            for row in rows
            if math.isclose(float(row["shrinkage"]), PRODUCTION_SHRINKAGE)
        )
        best_mass = max(float(row["selected_attention_mass"]) for row in rows)
        best_recall = max(float(row["top2_recall"]) for row in rows)
        best_rmse = min(float(row["score_rmse"]) for row in rows)
        check = {
            "selected_fraction": fraction,
            "production_mass_regret": (
                best_mass - float(production_row["selected_attention_mass"])
            ),
            "production_recall_regret": (
                best_recall - float(production_row["top2_recall"])
            ),
            "production_rmse_ratio_to_best": (
                float(production_row["score_rmse"]) / max(best_rmse, 1.0e-12)
            ),
        }
        checks.append(check)
        if check["production_mass_regret"] > 0.01:
            failures.append(f"mass regret exceeds 1 point at {fraction:.1%}")
        if check["production_recall_regret"] > 0.02:
            failures.append(f"recall regret exceeds 2 points at {fraction:.1%}")
        if check["production_rmse_ratio_to_best"] > 1.10:
            failures.append(f"RMSE exceeds 1.10x best at {fraction:.1%}")

    return {
        "schema": "qksieve_shrinkage_sensitivity_v1",
        "complete": True,
        "method": method,
        "calibration_source": "prefill_tail",
        "production_shrinkage": PRODUCTION_SHRINKAGE,
        "labels": list(labels),
        "shrinkages": list(shrinkages),
        "selected_fractions": list(fractions),
        "strict_paired_conditions": len(expected_keys),
        "bootstrap_samples": bootstrap_samples,
        "aggregate": aggregate_rows,
        "per_label": dict(per_label),
        "acceptance": {
            "passed": not failures,
            "thresholds": {
                "mass_regret_max": 0.01,
                "top2_recall_regret_max": 0.02,
                "score_rmse_ratio_to_best_max": 1.10,
            },
            "checks": checks,
            "failures": failures,
        },
        "source_sha256": sources,
        "claim_boundary": (
            "This is a paired selector-mechanism sensitivity test. It does not "
            "replace downstream LongBench, RULER, or PPL quality evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--shrinkages", default="0,0.25,0.5,0.75,0.9")
    parser.add_argument("--fractions", default="0.01,0.02,0.04")
    parser.add_argument("--method", default="qk_balanced")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(
        args.run_root,
        labels=tuple(item for item in args.labels.split(",") if item),
        shrinkages=parse_float_list(args.shrinkages),
        fractions=parse_float_list(args.fractions),
        method=args.method,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
