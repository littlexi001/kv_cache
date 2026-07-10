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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", help="Output dirs with task_results.csv")
    args = parser.parse_args()

    print(
        "run,task,samples,score,keep,online,coverage_active,coverage_trigger,"
        "trigger_rate,initial_terms,initial_recall,final_recall,triggered_score,untriggered_score"
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
            active_rows = [row for row in subset if int(fnum(row.get("ours_coverage_risk_active"))) == 1]
            triggered_rows = [row for row in subset if int(fnum(row.get("ours_coverage_risk_triggered"))) == 1]
            untriggered_rows = [
                row
                for row in subset
                if int(fnum(row.get("ours_coverage_risk_active"))) == 1
                and int(fnum(row.get("ours_coverage_risk_triggered"))) == 0
            ]
            print(
                f"{out.name},{task},{len(subset)},"
                f"{mean([fnum(row.get('score')) for row in subset]):.6f},"
                f"{mean([fnum(row.get('keep_fraction')) for row in subset]):.6f},"
                f"{mean([fnum(row.get('online_seconds')) for row in subset]):.3f},"
                f"{len(active_rows)},"
                f"{len(triggered_rows)},"
                f"{len(triggered_rows) / max(1, len(active_rows)):.6f},"
                f"{mean([fnum(row.get('ours_coverage_risk_initial_terms')) for row in active_rows]):.3f},"
                f"{mean([fnum(row.get('ours_coverage_risk_initial_recall')) for row in active_rows]):.6f},"
                f"{mean([fnum(row.get('ours_query_coverage_recall')) for row in active_rows]):.6f},"
                f"{mean([fnum(row.get('score')) for row in triggered_rows]):.6f},"
                f"{mean([fnum(row.get('score')) for row in untriggered_rows]):.6f}"
            )


if __name__ == "__main__":
    main()
