#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def flag(row: dict[str, str], key: str) -> int:
    return int(fnum(row.get(key), 0.0) != 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()

    print(
        "run,task,samples,score,keep,online,full_keep,"
        "task_full,output_fb,grounding_fb,support_window_fb,retry_full,"
        "consistency_full,coverage_trigger,coverage_active,support_window_active"
    )
    for output in args.outputs:
        out = Path(output)
        path = out / "task_results.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_task[row.get("task", "")].append(row)
            by_task["ALL"].append(row)
        for task, subset in sorted(by_task.items()):
            n = max(1, len(subset))
            full_keep = [
                1
                for row in subset
                if fnum(row.get("kept_prefix_tokens")) >= fnum(row.get("raw_prefix_tokens")) - 1
            ]
            print(
                f"{out.name},{task},{len(subset)},"
                f"{mean([fnum(row.get('score')) for row in subset]):.6f},"
                f"{mean([fnum(row.get('keep_fraction')) for row in subset]):.6f},"
                f"{mean([fnum(row.get('online_seconds')) for row in subset]):.3f},"
                f"{len(full_keep) / n:.6f},"
                f"{sum(flag(row, 'ours_full_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_output_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_grounding_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_support_window_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_retry_full_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_consistency_full_fallback_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_coverage_risk_triggered') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_coverage_risk_active') for row in subset) / n:.6f},"
                f"{sum(flag(row, 'ours_support_window_verifier_active') for row in subset) / n:.6f}"
            )


if __name__ == "__main__":
    main()
