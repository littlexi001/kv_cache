#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def load_rows(base_dir: Path, full_dir: Path, tasks: set[str]) -> list[dict[str, Any]]:
    full_scores: dict[tuple[str, str], float] = {}
    with (full_dir / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            full_scores[(row["task"], row["sample_id"])] = fnum(row.get("score"))

    rows: list[dict[str, Any]] = []
    with (base_dir / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task = row.get("task", "")
            if tasks and task not in tasks:
                continue
            key = (task, row.get("sample_id", ""))
            if key not in full_scores:
                continue
            if row.get("ours_score_gap2", "") == "" or row.get("ours_score_entropy", "") == "":
                continue
            rows.append(
                {
                    "task": task,
                    "sample_id": key[1],
                    "base_score": fnum(row.get("score")),
                    "full_score": full_scores[key],
                    "gap2": fnum(row.get("ours_score_gap2")),
                    "gap3": fnum(row.get("ours_score_gap3")),
                    "entropy": fnum(row.get("ours_score_entropy")),
                    "top_score": fnum(row.get("ours_score_max")),
                    "kept_context_tokens": fnum(row.get("kept_context_tokens")),
                    "raw_prefix_tokens": fnum(row.get("raw_prefix_tokens")),
                }
            )
    return rows


def triggered(row: dict[str, Any], gap2: float, entropy: float, gap3: float, top_score: float) -> bool:
    if gap2 >= 0.0 and row["gap2"] > gap2:
        return False
    if entropy <= 1.0 and row["entropy"] < entropy:
        return False
    if gap3 >= 0.0 and row["gap3"] > gap3:
        return False
    if top_score >= 0.0 and row["top_score"] > top_score:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--tasks", default="narrativeqa,2wikimqa,qasper")
    parser.add_argument("--gap2", default="0.08,0.10,0.12,0.15,0.20")
    parser.add_argument("--entropy", default="0.90,0.93,0.95,0.97,0.98")
    parser.add_argument("--gap3", type=float, default=-1.0)
    parser.add_argument("--top_score", type=float, default=-1.0)
    parser.add_argument("--positive_margin", type=float, default=0.05)
    args = parser.parse_args()

    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    rows = load_rows(args.base, args.full, tasks)
    if not rows:
        raise SystemExit("No calibrated rows found.")

    base_mean = sum(row["base_score"] for row in rows) / len(rows)
    full_mean = sum(row["full_score"] for row in rows) / len(rows)
    full_tokens = sum(max(1.0, row["raw_prefix_tokens"]) for row in rows)
    print(f"tasks={','.join(sorted(tasks)) or 'ALL'} samples={len(rows)}")
    print(f"base_mean={base_mean:.6f} full_mean={full_mean:.6f}")
    print(
        "estimated_score,estimated_kv_ratio,trigger_rate,trigger_count,"
        "precision,recall,gap2,entropy,gap3,top_score"
    )

    positives = [row for row in rows if row["full_score"] - row["base_score"] > args.positive_margin]
    for gap2 in [float(item) for item in args.gap2.split(",") if item.strip()]:
        for entropy in [float(item) for item in args.entropy.split(",") if item.strip()]:
            active = [row for row in rows if triggered(row, gap2, entropy, args.gap3, args.top_score)]
            if not active:
                continue
            active_keys = {(row["task"], row["sample_id"]) for row in active}
            score = sum(
                row["full_score"] if (row["task"], row["sample_id"]) in active_keys else row["base_score"]
                for row in rows
            ) / len(rows)
            kept_tokens = sum(
                row["raw_prefix_tokens"] if (row["task"], row["sample_id"]) in active_keys else row["kept_context_tokens"]
                for row in rows
            )
            true_positive = sum(1 for row in active if row["full_score"] - row["base_score"] > args.positive_margin)
            precision = true_positive / max(1, len(active))
            recall = true_positive / max(1, len(positives))
            print(
                f"{score:.6f},{kept_tokens / full_tokens:.6f},{len(active) / len(rows):.6f},"
                f"{len(active)},{precision:.6f},{recall:.6f},{gap2:.6f},{entropy:.6f},"
                f"{args.gap3:.6f},{args.top_score:.6f}"
            )


if __name__ == "__main__":
    main()
