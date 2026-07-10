#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def read_calibration(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["task"]: row for row in csv.DictReader(handle)}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy", required=True, type=Path)
    parser.add_argument("--calibration_csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min_beneficial", type=int, default=1)
    parser.add_argument("--max_trigger_rate", type=float, default=0.75)
    parser.add_argument("--default_budget", type=int, default=2048)
    parser.add_argument("--min_terms", type=int, default=3)
    args = parser.parse_args()

    policy = json.loads(args.base_policy.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("base policy must be a JSON object")
    out_policy = deepcopy(policy)
    calibration = read_calibration(args.calibration_csv)

    for task, row in calibration.items():
        threshold = row.get("threshold")
        if threshold in (None, "", "None"):
            continue
        beneficial = int(fnum(row.get("beneficial")))
        trigger_rate = fnum(row.get("trigger_rate"), 1.0)
        if beneficial < args.min_beneficial or trigger_rate > args.max_trigger_rate:
            continue
        task_cfg = out_policy.setdefault(task, {})
        if not isinstance(task_cfg, dict):
            task_cfg = {}
            out_policy[task] = task_cfg
        task_cfg["coverage_risk"] = True
        task_cfg["coverage_risk_min_recall"] = float(threshold)
        task_cfg["coverage_risk_min_terms"] = int(args.min_terms)
        task_cfg["coverage_risk_budget_tokens"] = int(
            task_cfg.get("coverage_risk_budget_tokens", args.default_budget)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
