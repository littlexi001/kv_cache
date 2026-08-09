#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {
    "b128_p16": {"budget": 128, "page": 16},
    "b256_p16": {"budget": 256, "page": 16},
    "b256_p64": {"budget": 256, "page": 64},
    "b512_p64": {"budget": 512, "page": 64},
    "b512_p128": {"budget": 512, "page": 128},
    "b1024_p128": {"budget": 1024, "page": 128},
    "b2048_p256": {"budget": 2048, "page": 256},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["task"], row["sample_id"]): row
        for row in rows
        if row.get("task") and row.get("sample_id")
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize_rows(
    rows: list[dict[str, Any]],
    action: str,
    task: str,
) -> dict[str, Any]:
    score = mean([fnum(row, "score") for row in rows])
    full_score = mean([fnum(row, "full_score") for row in rows])
    kv = mean([fnum(row, "keep_fraction") for row in rows])
    online = mean([fnum(row, "online_seconds") for row in rows])
    total = mean([fnum(row, "total_seconds") for row in rows])
    full_online = mean([fnum(row, "full_online_seconds") for row in rows])
    full_total = mean([fnum(row, "full_total_seconds") for row in rows])
    return {
        "action": action,
        "task": task,
        "samples": len(rows),
        "score": score,
        "full_score": full_score,
        "score_vs_full": score / full_score if full_score > 0 else "",
        "kv": kv,
        "online_seconds": online,
        "total_seconds": total,
        "online_speed_vs_full": full_online / online if online > 0 else "",
        "total_speed_vs_full": full_total / total if total > 0 else "",
        "direct_used": sum(int(fnum(row, "direct_used") > 0) for row in rows),
    }


def is_dominated(row: dict[str, Any], other: dict[str, Any]) -> bool:
    return (
        float(other["score"]) >= float(row["score"])
        and float(other["kv"]) <= float(row["kv"])
        and float(other["total_seconds"]) <= float(row["total_seconds"])
        and (
            float(other["score"]) > float(row["score"])
            or float(other["kv"]) < float(row["kv"])
            or float(other["total_seconds"]) < float(row["total_seconds"])
        )
    )


def choose_oracle(
    candidates: list[dict[str, Any]],
    full_row: dict[str, Any],
    quality_ratio: float,
    target_mode: str,
) -> dict[str, Any]:
    full_score = fnum(full_row, "score")
    best_score = max([full_score, *[fnum(row, "score") for row in candidates]])
    if target_mode == "full95" and full_score > 0:
        target = quality_ratio * full_score
    else:
        target = quality_ratio * best_score
    actions = [
        *candidates,
        {
            **full_row,
            "action": "full",
            "keep_fraction": 1.0,
            "direct_used": 0,
        },
    ]
    eligible = [row for row in actions if fnum(row, "score") + 1e-12 >= target]
    selected = min(
        eligible or actions,
        key=lambda row: (
            fnum(row, "keep_fraction"),
            fnum(row, "total_seconds"),
            -fnum(row, "score"),
        ),
    )
    return {**selected, "quality_target": target, "best_available_score": best_score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v441_pure_action_frontier_analysis_20260712")
    args = parser.parse_args()

    root = Path(args.root)
    full_table = by_key(read_csv(root / args.full_results))
    candidate_tables: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    missing: list[str] = []
    for action in ACTIONS:
        path = root / (
            f"outputs/riskkv_v19_v441_purefront_{action}_20260712_purefront_"
            f"m{args.samples}_bDyn_pDyn/task_results.csv"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        candidate_tables[action] = by_key(read_csv(path))
    if missing:
        raise FileNotFoundError("Missing frontier results:\n" + "\n".join(missing))

    common_keys = set(full_table)
    for table in candidate_tables.values():
        common_keys &= set(table)
    if not common_keys:
        raise RuntimeError("No common samples across Full and frontier actions")

    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_sample: list[dict[str, Any]] = []
    oracle_rows: dict[str, list[dict[str, Any]]] = {"full95": [], "best95": []}
    for key in sorted(common_keys):
        task, sample_id = key
        full = full_table[key]
        candidates: list[dict[str, Any]] = []
        for action, table in candidate_tables.items():
            row = table[key]
            item = {
                **row,
                "action": action,
                "budget": ACTIONS[action]["budget"],
                "page": ACTIONS[action]["page"],
                "direct_used": fnum(row, "ours_direct_structured_answer_used"),
                "full_score": fnum(full, "score"),
                "full_online_seconds": fnum(full, "online_seconds"),
                "full_total_seconds": fnum(full, "total_seconds"),
            }
            normalized[action].append(item)
            candidates.append(item)
        record: dict[str, Any] = {
            "task": task,
            "sample_id": sample_id,
            "full_score": fnum(full, "score"),
            "full_online_seconds": fnum(full, "online_seconds"),
            "full_total_seconds": fnum(full, "total_seconds"),
        }
        for mode in ["full95", "best95"]:
            selected = choose_oracle(candidates, full, args.quality_ratio, mode)
            record[f"{mode}_action"] = selected.get("action", "")
            record[f"{mode}_score"] = fnum(selected, "score")
            record[f"{mode}_kv"] = fnum(selected, "keep_fraction")
            record[f"{mode}_online_seconds"] = fnum(selected, "online_seconds")
            record[f"{mode}_total_seconds"] = fnum(selected, "total_seconds")
            record[f"{mode}_quality_target"] = fnum(selected, "quality_target")
            record[f"{mode}_best_available_score"] = fnum(selected, "best_available_score")
            oracle_rows[mode].append(
                {
                    **record,
                    "action": record[f"{mode}_action"],
                    "score": record[f"{mode}_score"],
                    "keep_fraction": record[f"{mode}_kv"],
                    "online_seconds": record[f"{mode}_online_seconds"],
                    "total_seconds": record[f"{mode}_total_seconds"],
                    "full_score": record["full_score"],
                }
            )
        per_sample.append(record)

    summary_rows: list[dict[str, Any]] = []
    tasks = sorted({task for task, _sample_id in common_keys})
    for action, rows in normalized.items():
        summary_rows.append(summarize_rows(rows, action, "ALL"))
        for task in tasks:
            summary_rows.append(summarize_rows([row for row in rows if row["task"] == task], action, task))
    for mode, rows in oracle_rows.items():
        enriched = [
            {
                **row,
                "full_online_seconds": next(
                    item["full_online_seconds"] for item in per_sample
                    if item["task"] == row["task"] and item["sample_id"] == row["sample_id"]
                ),
                "full_total_seconds": next(
                    item["full_total_seconds"] for item in per_sample
                    if item["task"] == row["task"] and item["sample_id"] == row["sample_id"]
                ),
                "direct_used": 0,
            }
            for row in rows
        ]
        summary_rows.append(summarize_rows(enriched, f"oracle_{mode}", "ALL"))
        for task in tasks:
            summary_rows.append(
                summarize_rows([row for row in enriched if row["task"] == task], f"oracle_{mode}", task)
            )

    global_fixed = [row for row in summary_rows if row["task"] == "ALL" and row["action"] in ACTIONS]
    pareto_actions = {
        str(row["action"])
        for row in global_fixed
        if not any(is_dominated(row, other) for other in global_fixed if other is not row)
    }
    win_counts = Counter(str(row["best95_action"]) for row in per_sample)
    min_wins = max(2, round(0.02 * len(per_sample)))
    selected_actions = sorted(
        pareto_actions
        | {action for action, count in win_counts.items() if action in ACTIONS and count >= min_wins},
        key=lambda action: (ACTIONS[action]["budget"], ACTIONS[action]["page"]),
    )

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "frontier_summary.csv", summary_rows)
    write_csv(output_dir / "oracle_labels.csv", per_sample)
    selection = {
        "samples": len(per_sample),
        "quality_ratio": args.quality_ratio,
        "pareto_actions": sorted(pareto_actions),
        "best95_oracle_win_counts": dict(win_counts),
        "minimum_wins": min_wins,
        "selected_actions_for_m50": selected_actions,
        "all_actions": ACTIONS,
    }
    (output_dir / "selected_actions.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    for row in summary_rows:
        if row["task"] == "ALL":
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
