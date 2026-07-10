#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("benchmark") != "longbench":
                continue
            key = (str(row["task"]), str(row["sample_id"]))
            rows[key] = row
    return rows


def parse_pages(spec: str) -> set[int]:
    out: set[int] = set()
    for item in str(spec).split(","):
        item = item.strip()
        if item.isdigit():
            out.add(int(item))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_bridge_results", required=True)
    parser.add_argument("--bridge_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_delta", type=float, default=1e-9)
    args = parser.parse_args()

    no_rows = read_rows(Path(args.no_bridge_results))
    bridge_rows = read_rows(Path(args.bridge_results))
    keys = sorted(set(no_rows) & set(bridge_rows))
    if not keys:
        raise SystemExit("No paired rows found")

    label_rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task, sample_id in keys:
        no = no_rows[(task, sample_id)]
        bridge = bridge_rows[(task, sample_id)]
        no_score = float(no.get("score") or 0.0)
        bridge_score = float(bridge.get("score") or 0.0)
        delta = bridge_score - no_score
        no_pages = parse_pages(no.get("selected_pages", ""))
        bridge_pages = parse_pages(bridge.get("selected_pages", ""))
        union = no_pages | bridge_pages
        row = {
            "task": task,
            "sample_id": sample_id,
            "metric": no.get("metric", ""),
            "raw_prefix_tokens": no.get("raw_prefix_tokens", ""),
            "page_count": no.get("page_count", ""),
            "no_bridge_score": fmt(no_score),
            "bridge_score": fmt(bridge_score),
            "delta": fmt(delta),
            "label_bridge": int(delta > args.min_delta),
            "label_or_tie_bridge": int(delta >= -args.min_delta),
            "selected_page_jaccard": fmt(len(no_pages & bridge_pages) / max(1, len(union))),
            "no_selected_pages": no.get("selected_pages", ""),
            "bridge_selected_pages": bridge.get("selected_pages", ""),
        }
        label_rows.append(row)
        by_task[task].append(row)

    task_policy_rows: list[dict[str, Any]] = []
    task_policy_score = 0.0
    no_score_sum = 0.0
    bridge_score_sum = 0.0
    oracle_score_sum = 0.0
    for task, rows in sorted(by_task.items()):
        no_scores = [float(row["no_bridge_score"]) for row in rows]
        bridge_scores = [float(row["bridge_score"]) for row in rows]
        deltas = [float(row["delta"]) for row in rows]
        use_bridge = mean(deltas) > args.min_delta
        task_policy_rows.append(
            {
                "task": task,
                "n": len(rows),
                "no_bridge_mean": fmt(mean(no_scores)),
                "bridge_mean": fmt(mean(bridge_scores)),
                "mean_delta": fmt(mean(deltas)),
                "bridge_win_rate": fmt(sum(delta > args.min_delta for delta in deltas) / len(deltas)),
                "policy_action": "bridge" if use_bridge else "no_bridge",
            }
        )
        task_policy_score += sum(bridge_scores if use_bridge else no_scores)
        no_score_sum += sum(no_scores)
        bridge_score_sum += sum(bridge_scores)
        oracle_score_sum += sum(max(a, b) for a, b in zip(no_scores, bridge_scores))

    n = len(label_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "bridge_gate_labels.csv", label_rows)
    write_csv(output_dir / "bridge_gate_task_policy.csv", task_policy_rows)

    summary = {
        "pairs": n,
        "no_bridge_score": no_score_sum / n,
        "all_bridge_score": bridge_score_sum / n,
        "task_policy_score": task_policy_score / n,
        "sample_oracle_score": oracle_score_sum / n,
        "min_delta": args.min_delta,
    }
    (output_dir / "bridge_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Bridge Gate Distillation Report",
        "",
        f"- pairs: {n}",
        f"- no_bridge_score: {summary['no_bridge_score']:.6f}",
        f"- all_bridge_score: {summary['all_bridge_score']:.6f}",
        f"- task_policy_score: {summary['task_policy_score']:.6f}",
        f"- sample_oracle_score: {summary['sample_oracle_score']:.6f}",
        "",
        "| Task | n | No bridge | Bridge | Delta | Win rate | Action |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in task_policy_rows:
        lines.append(
            "| {task} | {n} | {no_bridge_mean} | {bridge_mean} | {mean_delta} | {bridge_win_rate} | {policy_action} |".format(
                **row
            )
        )
    (output_dir / "bridge_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
