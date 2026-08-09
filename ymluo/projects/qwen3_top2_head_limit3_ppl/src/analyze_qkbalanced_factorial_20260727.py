from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FULL = "full_kv"
AUTO_PLAIN = "countcap_fullprompt_qkbalanced_packed_direct"
FIXED_PLAIN = "countcap_fullprompt_qkbalanced_fixed4421_packed_direct"
AUTO_QSCALE = "countcap_fullprompt_qkbalanced_qscale_packed_direct"
FIXED_QSCALE = (
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_packed_direct"
)
AUTO_QSCALE_OAS = (
    "countcap_fullprompt_qkbalanced_qscale_oas_packed_direct"
)
FIXED_QSCALE_OAS = (
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct"
)
METHODS = (FULL, AUTO_PLAIN, FIXED_PLAIN, AUTO_QSCALE, FIXED_QSCALE)
OAS_METHODS = (AUTO_QSCALE_OAS, FIXED_QSCALE_OAS)
FACTORIAL_METHODS = (FIXED_PLAIN, AUTO_QSCALE, FIXED_QSCALE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the 2x2 QK-balanced bit-allocation by scale factorial."
        )
    )
    parser.add_argument("--reference_glob", required=True)
    parser.add_argument("--factorial_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bootstrap_replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--include_qscale_oas", action="store_true")
    return parser.parse_args()


def read_csv_glob(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _float(row: dict[str, str], field: str) -> float:
    return float(row.get(field, "") or 0.0)


def _index_rows(
    rows: list[dict[str, str]],
    allowed_methods: set[str],
) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in allowed_methods:
            continue
        key = (str(row["task"]), str(row["sample_id"]), method)
        if key in indexed:
            raise ValueError(f"duplicate row: {key}")
        indexed[key] = row
    return indexed


def _macro_scores(
    samples: dict[tuple[str, str], dict[str, dict[str, str]]],
    methods: tuple[str, ...],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (task, _), methods in samples.items():
        by_task[task].append(methods)
    task_rows: list[dict[str, Any]] = []
    for task, task_samples in sorted(by_task.items()):
        row: dict[str, Any] = {"task": task, "samples": len(task_samples)}
        for method in methods:
            row[method] = float(
                np.mean([_float(sample[method], "score") for sample in task_samples])
            )
        task_rows.append(row)
    macro = {
        method: float(np.mean([row[method] for row in task_rows]))
        for method in methods
    }
    return macro, task_rows


def factorial_contrasts(scores: dict[str, float]) -> dict[str, float]:
    allocation_plain = scores[FIXED_PLAIN] - scores[AUTO_PLAIN]
    allocation_qscale = scores[FIXED_QSCALE] - scores[AUTO_QSCALE]
    qscale_auto = scores[AUTO_QSCALE] - scores[AUTO_PLAIN]
    qscale_fixed = scores[FIXED_QSCALE] - scores[FIXED_PLAIN]
    contrasts = {
        "fixed_minus_auto_plain": allocation_plain,
        "fixed_minus_auto_qscale": allocation_qscale,
        "qscale_minus_plain_auto": qscale_auto,
        "qscale_minus_plain_fixed": qscale_fixed,
        "allocation_x_qscale_interaction": (
            allocation_qscale - allocation_plain
        ),
    }
    if AUTO_QSCALE_OAS in scores and FIXED_QSCALE_OAS in scores:
        allocation_qscale_oas = (
            scores[FIXED_QSCALE_OAS] - scores[AUTO_QSCALE_OAS]
        )
        contrasts.update(
            {
                "fixed_minus_auto_qscale_oas": allocation_qscale_oas,
                "qscale_oas_minus_plain_auto": (
                    scores[AUTO_QSCALE_OAS] - scores[AUTO_PLAIN]
                ),
                "qscale_oas_minus_plain_fixed": (
                    scores[FIXED_QSCALE_OAS] - scores[FIXED_PLAIN]
                ),
                "qscale_oas_minus_raw_auto": (
                    scores[AUTO_QSCALE_OAS] - scores[AUTO_QSCALE]
                ),
                "qscale_oas_minus_raw_fixed": (
                    scores[FIXED_QSCALE_OAS] - scores[FIXED_QSCALE]
                ),
                "allocation_x_qscale_oas_interaction": (
                    allocation_qscale_oas - allocation_plain
                ),
            }
        )
    return contrasts


def _interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "lower_2p5": float(low),
        "median": float(median),
        "upper_97p5": float(high),
        "probability_ge_zero": float(np.mean(values >= 0.0)),
    }


def analyze_rows(
    reference_rows: list[dict[str, str]],
    factorial_rows: list[dict[str, str]],
    *,
    bootstrap_replicates: int,
    seed: int,
    include_qscale_oas: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    methods = METHODS + (OAS_METHODS if include_qscale_oas else ())
    factorial_methods = FACTORIAL_METHODS + (
        OAS_METHODS if include_qscale_oas else ()
    )
    factorial_index = _index_rows(
        factorial_rows,
        set(factorial_methods),
    )
    factorial_keys = {
        (task, sample_id)
        for task, sample_id, _ in factorial_index
    }
    if not factorial_keys:
        raise ValueError("no factorial samples")
    reference_index = _index_rows(
        reference_rows,
        {FULL, AUTO_PLAIN},
    )
    samples: dict[
        tuple[str, str],
        dict[str, dict[str, str]],
    ] = defaultdict(dict)
    for (task, sample_id, method), row in factorial_index.items():
        samples[(task, sample_id)][method] = row
    for task, sample_id in factorial_keys:
        for method in (FULL, AUTO_PLAIN):
            key = (task, sample_id, method)
            if key not in reference_index:
                raise ValueError(f"missing reference row: {key}")
            samples[(task, sample_id)][method] = reference_index[key]
    incomplete = [
        key for key, rows in samples.items() if set(rows) != set(methods)
    ]
    if incomplete:
        raise ValueError(f"incomplete factorial cells: {incomplete[:5]}")

    macro, task_rows = _macro_scores(samples, methods)
    contrasts = factorial_contrasts(macro)
    counts = Counter(
        row["method"]
        for methods in samples.values()
        for row in methods.values()
    )
    online_speedup: dict[str, float] = {}
    index_ratio: dict[str, float | None] = {}
    index_bits: dict[str, float | None] = {}
    prediction_exact: dict[str, float] = {}
    full_online = sum(
        _float(methods[FULL], "online_seconds")
        for methods in samples.values()
    )
    for method in methods:
        method_rows = [rows[method] for rows in samples.values()]
        online = sum(_float(row, "online_seconds") for row in method_rows)
        online_speedup[method] = full_online / online if online > 0.0 else 0.0
        measured_ratios = [
            _float(row, "packed_index_ratio_of_full_kv")
            for row in method_rows
            if _float(row, "packed_index_ratio_of_full_kv") > 0.0
        ]
        measured_bits = [
            _float(row, "packed_qmse_index_bits_per_token")
            for row in method_rows
            if _float(row, "packed_qmse_index_bits_per_token") > 0.0
        ]
        index_ratio[method] = (
            float(np.mean(measured_ratios))
            if measured_ratios
            else None
        )
        index_bits[method] = (
            float(np.mean(measured_bits))
            if measured_bits
            else None
        )
        prediction_exact[method] = float(
            np.mean(
                [
                    row.get("prediction", "")
                    == methods[FULL].get("prediction", "")
                    for methods, row in (
                        (methods, methods[method])
                        for methods in samples.values()
                    )
                ]
            )
        )

    by_task: dict[str, list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (task, _), methods in samples.items():
        by_task[task].append(methods)
    task_names = sorted(by_task)
    rng = np.random.default_rng(seed)
    bootstrap = {
        name: np.empty(bootstrap_replicates, dtype=np.float64)
        for name in contrasts
    }
    for replicate in range(bootstrap_replicates):
        sampled_tasks = rng.choice(task_names, size=len(task_names), replace=True)
        replicate_scores: dict[str, list[float]] = {
            method: [] for method in methods
        }
        for task in sampled_tasks:
            task_samples = by_task[str(task)]
            indices = rng.integers(0, len(task_samples), len(task_samples))
            for method in methods:
                replicate_scores[method].append(
                    float(
                        np.mean(
                            [
                                _float(task_samples[index][method], "score")
                                for index in indices
                            ]
                        )
                    )
                )
        replicate_macro = {
            method: float(np.mean(values))
            for method, values in replicate_scores.items()
        }
        replicate_contrasts = factorial_contrasts(replicate_macro)
        for name, value in replicate_contrasts.items():
            bootstrap[name][replicate] = value

    summary = {
        "protocol": {
            "samples": len(samples),
            "tasks": len(task_names),
            "methods": list(methods),
            "counts": dict(counts),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap": "resample tasks, then paired samples within task",
            "seed": seed,
        },
        "macro_scores": macro,
        "quality_retention_vs_full": {
            method: macro[method] / macro[FULL] if macro[FULL] > 0.0 else 0.0
            for method in methods
        },
        "factorial_contrasts": contrasts,
        "factorial_contrasts_bootstrap_95_percent": {
            name: _interval(values) for name, values in bootstrap.items()
        },
        "paired_online_speedup_vs_full": online_speedup,
        "mean_index_ratio_of_full_kv": index_ratio,
        "mean_index_bits_per_token": index_bits,
        "index_measurement_note": (
            "null means attention diagnostics were disabled; use the "
            "physical-cache benchmark for memory accounting"
        ),
        "prediction_exact_rate_vs_full": prediction_exact,
    }
    return summary, task_rows


def main() -> None:
    args = parse_args()
    summary, task_rows = analyze_rows(
        read_csv_glob(args.reference_glob),
        read_csv_glob(args.factorial_glob),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        include_qscale_oas=args.include_qscale_oas,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_task.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
