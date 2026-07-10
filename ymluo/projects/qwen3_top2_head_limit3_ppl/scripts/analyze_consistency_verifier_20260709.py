#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with (path / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row.get("task", ""), row.get("sample_id", ""))] = row
    return rows


def summarize(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    n = len(rows)
    print(
        f"{name},n={n},"
        f"base={sum(row['base'] for row in rows)/n:.6f},"
        f"candidate={sum(row['cand'] for row in rows)/n:.6f},"
        f"full={sum(row['full'] for row in rows)/n:.6f},"
        f"gain_vs_base={sum(row['cand']-row['base'] for row in rows)/n:.6f},"
        f"gap_to_full={sum(row['full']-row['cand'] for row in rows)/n:.6f},"
        f"keep={sum(row['keep'] for row in rows)/n:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--full", required=True)
    args = parser.parse_args()

    base = read_rows(Path(args.base))
    candidate = read_rows(Path(args.candidate))
    full = read_rows(Path(args.full))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    triggered: list[dict[str, Any]] = []
    untriggered: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for key, row in candidate.items():
        if key not in base or key not in full:
            continue
        item = {
            "task": key[0],
            "base": fnum(base[key].get("score")),
            "cand": fnum(row.get("score")),
            "full": fnum(full[key].get("score")),
            "keep": fnum(row.get("keep_fraction")),
            "triggered": fnum(row.get("ours_consistency_full_fallback_active")) > 0.5,
            "checked": fnum(row.get("ours_consistency_check_active")) > 0.5,
        }
        all_rows.append(item)
        by_task[key[0]].append(item)
        if item["triggered"]:
            triggered.append(item)
        else:
            untriggered.append(item)

    summarize("ALL", all_rows)
    summarize("TRIGGERED", triggered)
    summarize("UNTRIGGERED", untriggered)
    print("BY_TASK")
    for task in sorted(by_task):
        rows = by_task[task]
        trigger_rate = sum(1 for row in rows if row["triggered"]) / max(1, len(rows))
        check_rate = sum(1 for row in rows if row["checked"]) / max(1, len(rows))
        print(f"task={task},check_rate={check_rate:.6f},trigger_rate={trigger_rate:.6f}")
        summarize(f"  {task}", rows)


if __name__ == "__main__":
    main()
