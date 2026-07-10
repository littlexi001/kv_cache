#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def load_task_results(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    task_results = path / "task_results.csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with task_results.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["task"], row["sample_id"])] = row
    return rows


def mean(rows: list[dict[str, str]], column: str) -> float:
    return sum(fnum(row.get(column)) for row in rows) / max(1, len(rows))


def score_risk(row: dict[str, str], gap2: float, entropy: float, gap3: float, top_score: float) -> bool:
    if gap2 >= 0.0 and fnum(row.get("ours_score_gap2")) > gap2:
        return False
    if entropy <= 1.0 and fnum(row.get("ours_score_entropy")) < entropy:
        return False
    if gap3 >= 0.0 and fnum(row.get("ours_score_gap3")) > gap3:
        return False
    if top_score >= 0.0 and fnum(row.get("ours_score_max")) > top_score:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate selective counterfactual verification by using a base first-action run for gated tasks "
            "and a reference verifier run for triggered or non-gated tasks."
        )
    )
    parser.add_argument("--base", required=True, type=Path, help="First-action or compact policy output directory.")
    parser.add_argument("--reference", required=True, type=Path, help="Verifier/full-policy output directory.")
    parser.add_argument("--consistency_tasks", default="narrativeqa,multifieldqa_en,2wikimqa")
    parser.add_argument("--gap2", default="0.08,0.10,0.12,0.15,0.18,0.20")
    parser.add_argument("--entropy", default="0.90,0.93,0.95,0.97,1.01")
    parser.add_argument("--gap3", type=float, default=-1.0)
    parser.add_argument("--top_score", type=float, default=-1.0)
    args = parser.parse_args()

    base = load_task_results(args.base)
    reference = load_task_results(args.reference)
    keys = sorted(set(base) & set(reference))
    if not keys:
        raise SystemExit("No overlapping samples between base and reference outputs.")

    consistency_tasks = {item.strip() for item in args.consistency_tasks.split(",") if item.strip()}
    print(
        "score,keep_fraction,online_seconds,trigger_rate,consistency_check_rate,"
        "consistency_disagreement_rate,consistency_full_fallback_rate,gap2,entropy,gap3,top_score"
    )
    for gap2 in [float(item) for item in args.gap2.split(",") if item.strip()]:
        for entropy in [float(item) for item in args.entropy.split(",") if item.strip()]:
            stitched: list[dict[str, str]] = []
            trigger_count = 0
            for key in keys:
                task, _ = key
                if task in consistency_tasks:
                    triggered = score_risk(base[key], gap2, entropy, args.gap3, args.top_score)
                    if triggered:
                        trigger_count += 1
                    stitched.append(reference[key] if triggered else base[key])
                else:
                    stitched.append(reference[key])
            print(
                f"{mean(stitched, 'score'):.6f},"
                f"{mean(stitched, 'keep_fraction'):.6f},"
                f"{mean(stitched, 'online_seconds'):.6f},"
                f"{trigger_count / max(1, len(keys)):.6f},"
                f"{mean(stitched, 'ours_consistency_check_active'):.6f},"
                f"{mean(stitched, 'ours_consistency_disagreement_active'):.6f},"
                f"{mean(stitched, 'ours_consistency_full_fallback_active'):.6f},"
                f"{gap2:.6f},{entropy:.6f},{args.gap3:.6f},{args.top_score:.6f}"
            )


if __name__ == "__main__":
    main()
