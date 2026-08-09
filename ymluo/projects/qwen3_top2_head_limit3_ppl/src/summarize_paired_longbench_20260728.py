from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TIMING_FIELDS = (
    "prefill_seconds",
    "query_seconds",
    "decode_seconds",
    "online_seconds",
    "total_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly summarize paired multi-method LongBench rows."
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--reference_method", default="full_kv")
    parser.add_argument("--expected_pairs", type=int, default=3750)
    parser.add_argument("--expected_tasks", type=int, default=16)
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _load_rows(run_root: Path) -> list[dict[str, str]]:
    paths = sorted(run_root.glob("shard[0-9]*/sample_results.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no shard sample_results.csv files under {run_root}"
        )
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
    reference_method: str,
    expected_pairs: int,
    expected_tasks: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if reference_method not in methods:
        raise ValueError("reference method must be one of methods")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    expected_counts = Counter(
        {method: expected_pairs for method in methods}
    )
    counts = Counter(row["method"] for row in rows)
    if counts != expected_counts:
        raise AssertionError(
            f"method counts differ: expected={expected_counts}, got={counts}"
        )

    by_method: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for method in methods:
        selected: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            if row["method"] != method:
                continue
            key = (row["task"], row["sample_id"])
            if key in selected:
                raise AssertionError(
                    f"duplicate row for {method}: {key}"
                )
            selected[key] = row
        by_method[method] = selected
    reference_keys = set(by_method[reference_method])
    if len(reference_keys) != expected_pairs:
        raise AssertionError(
            f"strict pair count differs: {len(reference_keys)}"
        )
    for method in methods:
        if set(by_method[method]) != reference_keys:
            missing = reference_keys - set(by_method[method])
            extra = set(by_method[method]) - reference_keys
            raise AssertionError(
                f"{method} is not strictly paired: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    tasks = sorted({task for task, _ in reference_keys})
    if len(tasks) != expected_tasks:
        raise AssertionError(
            f"task count differs: expected={expected_tasks}, got={len(tasks)}"
        )

    keys_by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(reference_keys):
        keys_by_task[key[0]].append(key)
    per_task: dict[str, Any] = {}
    for task in tasks:
        task_summary: dict[str, Any] = {
            "samples": len(keys_by_task[task])
        }
        for method in methods:
            task_rows = [
                by_method[method][key] for key in keys_by_task[task]
            ]
            method_summary = {
                "score": _mean(
                    [float(row["score"]) for row in task_rows]
                )
            }
            for field in TIMING_FIELDS:
                if field in task_rows[0]:
                    method_summary[field] = _mean(
                        [float(row[field]) for row in task_rows]
                    )
            task_summary[method] = method_summary
        reference_score = task_summary[reference_method]["score"]
        for method in methods:
            score = task_summary[method]["score"]
            task_summary[method]["quality_retention"] = (
                score / reference_score if reference_score else None
            )
            for field in TIMING_FIELDS:
                if field not in task_summary[method]:
                    continue
                method_time = task_summary[method][field]
                reference_time = task_summary[reference_method][field]
                task_summary[method][field.replace("_seconds", "_speedup")] = (
                    reference_time / method_time
                    if method_time > 0
                    else None
                )
        per_task[task] = task_summary

    method_summary: dict[str, Any] = {}
    reference_macro = _mean(
        [per_task[task][reference_method]["score"] for task in tasks]
    )
    for method in methods:
        macro = _mean(
            [per_task[task][method]["score"] for task in tasks]
        )
        all_rows = list(by_method[method].values())
        summary: dict[str, Any] = {
            "macro_score": macro,
            "quality_retention": (
                macro / reference_macro if reference_macro else None
            ),
        }
        for field in TIMING_FIELDS:
            if field not in all_rows[0]:
                continue
            mean_time = _mean([float(row[field]) for row in all_rows])
            reference_time = _mean(
                [
                    float(row[field])
                    for row in by_method[reference_method].values()
                ]
            )
            summary[field] = mean_time
            summary[field.replace("_seconds", "_speedup")] = (
                reference_time / mean_time if mean_time > 0 else None
            )
        method_summary[method] = summary

    if bootstrap_resamples > 0:
        rng = np.random.default_rng(seed)
        arrays = {
            task: {
                method: np.asarray(
                    [
                        float(by_method[method][key]["score"])
                        for key in keys_by_task[task]
                    ],
                    dtype=np.float64,
                )
                for method in methods
            }
            for task in tasks
        }
        differences = {
            method: [] for method in methods if method != reference_method
        }
        retentions = {
            method: [] for method in methods if method != reference_method
        }
        for _ in range(bootstrap_resamples):
            macros = {method: [] for method in methods}
            for task in tasks:
                sample_count = len(keys_by_task[task])
                indices = rng.integers(
                    0, sample_count, size=sample_count
                )
                for method in methods:
                    macros[method].append(
                        float(arrays[task][method][indices].mean())
                    )
            macro_values = {
                method: _mean(values)
                for method, values in macros.items()
            }
            boot_reference = macro_values[reference_method]
            for method in differences:
                differences[method].append(
                    macro_values[method] - boot_reference
                )
                retentions[method].append(
                    macro_values[method] / boot_reference
                )
        for method in differences:
            method_summary[method]["macro_difference_95ci"] = [
                float(np.quantile(differences[method], 0.025)),
                float(np.quantile(differences[method], 0.975)),
            ]
            method_summary[method]["quality_retention_95ci"] = [
                float(np.quantile(retentions[method], 0.025)),
                float(np.quantile(retentions[method], 0.975)),
            ]

    return {
        "strict_pairs": len(reference_keys),
        "tasks": len(tasks),
        "reference_method": reference_method,
        "counts": dict(counts),
        "methods": method_summary,
        "per_task": per_task,
    }


def main() -> None:
    args = parse_args()
    methods = tuple(
        item.strip() for item in args.methods.split(",") if item.strip()
    )
    summary = summarize(
        rows=_load_rows(args.run_root),
        methods=methods,
        reference_method=args.reference_method,
        expected_pairs=args.expected_pairs,
        expected_tasks=args.expected_tasks,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output = args.output or args.run_root / "paired_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
