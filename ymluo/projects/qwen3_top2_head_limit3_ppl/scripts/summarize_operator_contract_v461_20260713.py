#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows))


def task_score(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(float(row["score"]))
    return {task: sum(values) / len(values) for task, values in grouped.items()}


def macro_score(rows: list[dict[str, str]]) -> float:
    scores = task_score(rows)
    return sum(scores.values()) / max(1, len(scores))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ours = read_csv(Path(args.ours))
    full_all = read_csv(Path(args.full))
    full_index = {(row["task"], row["sample_id"]): row for row in full_all}
    full = [full_index[(row["task"], row["sample_id"])] for row in ours if (row["task"], row["sample_id"]) in full_index]
    if len(full) != len(ours):
        raise RuntimeError(f"matched Full rows {len(full)} != ours rows {len(ours)}")

    ours_tasks = task_score(ours)
    full_tasks = task_score(full)
    task_rows: list[dict[str, Any]] = []
    for task in sorted(ours_tasks):
        ours_subset = [row for row in ours if row["task"] == task]
        full_subset = [row for row in full if row["task"] == task]
        task_rows.append(
            {
                "task": task,
                "samples": len(ours_subset),
                "operator": Counter(row.get("ours_operator_mode", "") for row in ours_subset).most_common(1)[0][0],
                "ours_score": ours_tasks[task],
                "full_score": full_tasks[task],
                "score_over_full": ours_tasks[task] / full_tasks[task] if full_tasks[task] > 0 else "",
                "kv_ratio": mean(ours_subset, "keep_fraction"),
                "online_speed": sum(float(row["online_seconds"]) for row in full_subset)
                / max(1e-12, sum(float(row["online_seconds"]) for row in ours_subset)),
                "total_speed": sum(float(row["total_seconds"]) for row in full_subset)
                / max(1e-12, sum(float(row["total_seconds"]) for row in ours_subset)),
                "direct_rate": sum(int(float(row.get("ours_direct_structured_answer_used", 0) or 0)) for row in ours_subset)
                / max(1, len(ours_subset)),
            }
        )

    ours_macro = macro_score(ours)
    full_macro = macro_score(full)
    overall = {
        "samples": len(ours),
        "tasks": len(ours_tasks),
        "ours_macro_score": ours_macro,
        "full_macro_score": full_macro,
        "score_over_full": ours_macro / full_macro if full_macro > 0 else None,
        "mean_kv_ratio": mean(ours, "keep_fraction"),
        "online_speed": sum(float(row["online_seconds"]) for row in full)
        / max(1e-12, sum(float(row["online_seconds"]) for row in ours)),
        "total_speed": sum(float(row["total_seconds"]) for row in full)
        / max(1e-12, sum(float(row["total_seconds"]) for row in ours)),
        "mean_prefill_seconds": mean(ours, "prefill_seconds"),
        "mean_gather_seconds": mean(ours, "kv_gather_seconds"),
        "mean_query_seconds": mean(ours, "query_seconds"),
        "mean_decode_seconds": mean(ours, "decode_seconds"),
        "direct_rate": sum(int(float(row.get("ours_direct_structured_answer_used", 0) or 0)) for row in ours)
        / max(1, len(ours)),
        "operator_counts": dict(Counter(row.get("ours_operator_mode", "") for row in ours)),
        "route_errors": sum(bool(row.get("ours_operator_fallback_reason", "").startswith("operator_router_error")) for row in ours),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "matched_task_summary.csv", task_rows)
    (output_dir / "matched_overall.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2, sort_keys=True))
    for row in task_rows:
        ratio = row["score_over_full"]
        ratio_text = "NA" if ratio == "" else f"{float(ratio):.3f}"
        print(
            f"{row['task']:24s} op={row['operator']:10s} ours={row['ours_score']:.4f} "
            f"full={row['full_score']:.4f} ratio={ratio_text} kv={row['kv_ratio']:.3f} "
            f"total={row['total_speed']:.2f}x"
        )


if __name__ == "__main__":
    main()
