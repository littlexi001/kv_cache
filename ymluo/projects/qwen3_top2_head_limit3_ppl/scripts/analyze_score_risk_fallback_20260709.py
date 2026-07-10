#!/usr/bin/env python3
"""Summarize score-risk gates and fallback behavior from task_results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def as_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None) -> int:
    return int(as_float(value, 0.0))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories containing task_results.csv")
    parser.add_argument(
        "--tasks",
        default="narrativeqa,multifieldqa_en,2wikimqa",
        help="Comma-separated task names to print in detail.",
    )
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    for run in args.runs:
        path = Path(run) / "task_results.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        print(f"run={Path(run).name} rows={len(rows)}")
        for task in tasks:
            task_rows = [row for row in rows if row.get("task") == task]
            if not task_rows:
                continue
            risk_values = [as_float(row.get("ours_score_risk_linear_value"), -99.0) for row in task_rows]
            scored = [
                (
                    as_float(row.get("ours_score_risk_linear_value"), -99.0),
                    as_float(row.get("score")),
                    as_int(row.get("ours_score_risk_triggered")),
                    as_int(row.get("ours_consistency_full_fallback_active")),
                    as_float(row.get("keep_fraction")),
                    row.get("sample_id", "")[:12],
                )
                for row in task_rows
            ]
            scored.sort()
            print(
                ",".join(
                    [
                        task,
                        f"n={len(task_rows)}",
                        f"score={mean([as_float(row.get('score')) for row in task_rows]):.6f}",
                        f"keep={mean([as_float(row.get('keep_fraction')) for row in task_rows]):.6f}",
                        f"risk_trigger={sum(as_int(row.get('ours_score_risk_triggered')) for row in task_rows)}",
                        f"consistency_full={sum(as_int(row.get('ours_consistency_full_fallback_active')) for row in task_rows)}",
                        f"risk_min={min(risk_values):.6f}",
                        f"risk_max={max(risk_values):.6f}",
                    ]
                )
            )
            print("  lowest")
            for item in scored[: args.limit]:
                print("   risk={:.6f},score={:.6f},trigger={},full={},keep={:.6f},id={}".format(*item))
            print("  highest")
            for item in scored[-args.limit :]:
                print("   risk={:.6f},score={:.6f},trigger={},full={},keep={:.6f},id={}".format(*item))


if __name__ == "__main__":
    main()
