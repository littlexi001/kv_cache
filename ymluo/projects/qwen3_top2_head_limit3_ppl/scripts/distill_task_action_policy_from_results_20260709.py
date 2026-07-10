#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def parse_result_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected action=path, got {spec}")
        action, path = spec.split("=", 1)
        action = action.strip()
        if not action:
            raise SystemExit(f"Empty action in {spec}")
        out[action] = Path(path.strip())
    if len(out) < 2:
        raise SystemExit("Need at least two --result action=path entries")
    return out


def load_action_rows(paths: dict[str, Path]) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    action_rows: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for action, path in paths.items():
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row in read_csv(path):
            if row.get("benchmark") != "longbench":
                continue
            rows[(str(row.get("task", "")), str(row.get("sample_id", "")))] = row
        action_rows[action] = rows
    return action_rows


def choose_best(candidates: dict[str, dict[str, str]], min_delta: float, priority: dict[str, int]) -> str:
    best_action = ""
    best_score = -1e9
    best_keep = 1e9
    best_online = 1e9
    best_priority = 10**9
    for action, row in candidates.items():
        score = fnum(row.get("score"))
        keep = fnum(row.get("keep_fraction"), 1.0)
        online = fnum(row.get("online_seconds"), 1e9)
        action_priority = priority.get(action, 10**9)
        if score > best_score + min_delta:
            best_action, best_score, best_keep, best_online, best_priority = (
                action,
                score,
                keep,
                online,
                action_priority,
            )
        elif abs(score - best_score) <= min_delta and (keep, online, action_priority) < (
            best_keep,
            best_online,
            best_priority,
        ):
            best_action, best_score, best_keep, best_online, best_priority = (
                action,
                score,
                keep,
                online,
                action_priority,
            )
    return best_action


def score_actions(rows: list[dict[str, Any]], action_by_key: dict[tuple[str, str], str]) -> float:
    total = 0.0
    for row in rows:
        total += fnum(row[f"score_{action_by_key[(row['task'], row['sample_id'])]}"])
    return total / max(1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", default=[], help="Action result as action=task_results.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_delta", type=float, default=1e-9)
    args = parser.parse_args()

    result_paths = parse_result_specs(args.result)
    priority = {action: idx for idx, action in enumerate(result_paths)}
    action_rows = load_action_rows(result_paths)
    common_keys = set.intersection(*(set(rows) for rows in action_rows.values()))
    if not common_keys:
        raise SystemExit("No common longbench task/sample_id rows across actions")

    label_rows: list[dict[str, Any]] = []
    for task, sample_id in sorted(common_keys):
        candidates = {action: rows[(task, sample_id)] for action, rows in action_rows.items()}
        best = choose_best(candidates, args.min_delta, priority)
        row: dict[str, Any] = {
            "task": task,
            "sample_id": sample_id,
            "metric": candidates[best].get("metric", ""),
            "best_action": best,
        }
        for action, item in sorted(candidates.items()):
            row[f"score_{action}"] = f"{fnum(item.get('score')):.6f}"
            row[f"keep_{action}"] = f"{fnum(item.get('keep_fraction'), 1.0):.6f}"
            row[f"online_{action}"] = f"{fnum(item.get('online_seconds'), 0.0):.6f}"
            row[f"budget_{action}"] = item.get("budget_tokens", "")
            row[f"page_tokens_{action}"] = item.get("page_tokens", "")
            row[f"scorer_{action}"] = item.get("ours_scorer", "")
        label_rows.append(row)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        by_task[str(row["task"])].append(row)

    task_policy: dict[tuple[str, str], str] = {}
    task_rows: list[dict[str, Any]] = []
    for task, rows in sorted(by_task.items()):
        means = {}
        keeps = {}
        for action in result_paths:
            means[action] = mean(fnum(row[f"score_{action}"]) for row in rows)
            keeps[action] = mean(fnum(row[f"keep_{action}"], 1.0) for row in rows)
        best = sorted(result_paths, key=lambda action: (-means[action], keeps[action], priority[action]))[0]
        for row in rows:
            task_policy[(str(row["task"]), str(row["sample_id"]))] = best
        counts = Counter(str(row["best_action"]) for row in rows)
        task_rows.append(
            {
                "task": task,
                "n": len(rows),
                "task_policy_action": best,
                "sample_oracle_majority": counts.most_common(1)[0][0],
                "sample_oracle_agree": f"{counts[best] / len(rows):.6f}",
                **{f"mean_score_{action}": f"{means[action]:.6f}" for action in sorted(result_paths)},
                **{f"mean_keep_{action}": f"{keeps[action]:.6f}" for action in sorted(result_paths)},
            }
        )

    sample_oracle_actions = {(str(row["task"]), str(row["sample_id"])): str(row["best_action"]) for row in label_rows}
    summary = {
        "actions": sorted(result_paths),
        "examples": len(label_rows),
        "sample_oracle_score": score_actions(label_rows, sample_oracle_actions),
        "task_policy_score": score_actions(label_rows, task_policy),
        "action_scores": {
            action: mean(fnum(row[f"score_{action}"]) for row in label_rows) for action in sorted(result_paths)
        },
        "action_keep": {
            action: mean(fnum(row[f"keep_{action}"], 1.0) for row in label_rows) for action in sorted(result_paths)
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "action_labels.csv", label_rows)
    write_csv(output_dir / "task_action_policy.csv", task_rows)
    (output_dir / "action_policy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Task Action Policy Distillation",
        "",
        f"- examples: {summary['examples']}",
        f"- sample_oracle_score: {summary['sample_oracle_score']:.6f}",
        f"- task_policy_score: {summary['task_policy_score']:.6f}",
        "",
        "| Action | Mean score | Mean keep |",
        "|---|---:|---:|",
    ]
    for action in sorted(result_paths):
        lines.append(f"| {action} | {summary['action_scores'][action]:.6f} | {summary['action_keep'][action]:.6f} |")
    lines.extend(
        [
            "",
            "| Task | n | Policy | Oracle majority | Agree |",
            "|---|---:|---|---|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| {row['task']} | {row['n']} | {row['task_policy_action']} | "
            f"{row['sample_oracle_majority']} | {row['sample_oracle_agree']} |"
        )
    (output_dir / "action_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
