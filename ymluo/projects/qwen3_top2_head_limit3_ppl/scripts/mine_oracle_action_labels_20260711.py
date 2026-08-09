#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def label_for_dir(path: Path) -> str:
    name = path.name
    if name.startswith("riskkv_v19_"):
        name = name[len("riskkv_v19_") :]
    return name


def sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""))


def load_actions(candidate_dirs: list[Path]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for directory in candidate_dirs:
        label = label_for_dir(directory)
        for row in read_csv(directory / "task_results.csv"):
            key = sample_key(row)
            if not key[1] or not key[2]:
                continue
            by_key[key].append(
                {
                    "label": label,
                    "score": fnum(row, "score"),
                    "kv_keep": fnum(row, "keep_fraction"),
                    "online_seconds": fnum(row, "online_seconds"),
                    "total_seconds": fnum(row, "total_seconds"),
                    "raw": row,
                }
            )
    return by_key


def load_baseline(directory: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(directory / "task_results.csv"):
        key = sample_key(row)
        if key[1] and key[2]:
            out[key] = row
    return out


def choose_min_safe(
    actions: list[dict[str, Any]],
    threshold: float,
    max_kv: float,
) -> dict[str, Any] | None:
    feasible = [
        action
        for action in actions
        if float(action["score"]) >= threshold and float(action["kv_keep"]) <= max_kv
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda item: (float(item["kv_keep"]), float(item["online_seconds"]), -float(item["score"])))
    return feasible[0]


def choose_best(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not actions:
        return None
    return max(actions, key=lambda item: (float(item["score"]), -float(item["kv_keep"]), -float(item["online_seconds"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--reference_label", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_ratio", type=float, default=0.95)
    parser.add_argument("--score_slack", type=float, default=0.0)
    parser.add_argument("--max_kv", type=float, default=0.30)
    parser.add_argument("--allow_missing_full", action="store_true")
    parser.add_argument("candidate_dirs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = load_baseline(Path(args.full_dir))
    actions_by_key = load_actions([Path(item) for item in args.candidate_dirs])

    rows: list[dict[str, Any]] = []
    for key in sorted(actions_by_key):
        full_row = full_rows.get(key)
        if full_row is None and not args.allow_missing_full:
            continue
        full_score = fnum(full_row, "score")
        full_online = fnum(full_row, "online_seconds")
        threshold = max(0.0, full_score * float(args.score_ratio) - float(args.score_slack))
        actions = actions_by_key[key]
        best = choose_best(actions)
        min_safe = choose_min_safe(actions, threshold, float(args.max_kv))
        reference = None
        if args.reference_label:
            reference = next((action for action in actions if action["label"] == args.reference_label), None)
        row: dict[str, Any] = {
            "benchmark": key[0],
            "task": key[1],
            "sample_id": key[2],
            "full_score": full_score,
            "full_online_seconds": full_online,
            "safe_threshold": threshold,
            "candidate_count": len(actions),
            "best_label": best["label"] if best else "",
            "best_score": best["score"] if best else "",
            "best_kv_keep": best["kv_keep"] if best else "",
            "best_online_seconds": best["online_seconds"] if best else "",
            "min_safe_label": min_safe["label"] if min_safe else "full_kv_required",
            "min_safe_score": min_safe["score"] if min_safe else full_score,
            "min_safe_kv_keep": min_safe["kv_keep"] if min_safe else 1.0,
            "min_safe_online_seconds": min_safe["online_seconds"] if min_safe else full_online,
            "has_safe_sparse_action": int(min_safe is not None),
        }
        if reference is not None:
            row.update(
                {
                    "reference_label": reference["label"],
                    "reference_score": reference["score"],
                    "reference_kv_keep": reference["kv_keep"],
                    "reference_online_seconds": reference["online_seconds"],
                    "reference_is_dangerous": int(float(reference["score"]) < threshold),
                }
            )
        rows.append(row)

    write_csv(output_dir / "oracle_action_labels.csv", rows)
    (output_dir / "oracle_action_labels.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    summary: list[dict[str, Any]] = []
    for task, subset in sorted(by_task.items()):
        safe = sum(int(row["has_safe_sparse_action"]) for row in subset)
        summary.append(
            {
                "task": task,
                "samples": len(subset),
                "safe_sparse_rate": safe / max(1, len(subset)),
                "avg_min_safe_kv": sum(float(row["min_safe_kv_keep"]) for row in subset) / max(1, len(subset)),
                "avg_best_score": sum(float(row["best_score"] or 0.0) for row in subset) / max(1, len(subset)),
            }
        )
    write_csv(output_dir / "oracle_action_summary.csv", summary)
    print(json.dumps({"labels": len(rows), "summary_rows": len(summary), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
