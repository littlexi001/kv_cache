#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")

CANDIDATES = {
    "policy_v368": {
        "results": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_confirm_m100_20260712_lowkv_confirm_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json",
    },
    "policy_v381": {
        "results": "outputs/riskkv_v19_v381_policy_multiclass_nopost_20260712_policy_multiclass_nopost_v381_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v381_policy_multiclass_nopost_20260712.json",
    },
    "policy_v389": {
        "results": "outputs/riskkv_v19_v389_m100_task_knapsack_v2_20260712_m100_task_knapsack_v2_v389_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json",
    },
    "policy_v393": {
        "results": "outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v393_m100_task_knapsack_v385_20260712.json",
    },
    "policy_v395": {
        "results": "outputs/riskkv_v19_v395_m100_task_knapsack075_exact_20260712_m100_task_knapsack075_exact_v395_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v395_m100_task_knapsack075_exact_20260712.json",
    },
    "policy_v396": {
        "results": "outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json",
    },
}


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_by_task(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out = {}
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        subset = [row for row in rows if row["task"] == task]
        out[task] = {
            "score": sum(fnum(row, "score") for row in subset) / len(subset),
            "kv": sum(fnum(row, "keep_fraction") for row in subset) / len(subset),
            "online": sum(fnum(row, "online_seconds") for row in subset) / len(subset),
        }
    return out


def non_dominated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                float(other["score"]) >= float(row["score"]) - 1e-12
                and float(other["kv"]) <= float(row["kv"]) + 1e-12
                and (
                    float(other["score"]) > float(row["score"]) + 1e-12
                    or float(other["kv"]) < float(row["kv"]) - 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return sorted(kept, key=lambda item: (float(item["kv"]), -float(item["score"]), str(item["action"])))


def solve(per_task: dict[str, list[dict[str, Any]]], kv_limit: float, kv_scale: int) -> dict[str, Any]:
    tasks = sorted(per_task)
    limit = int(round(kv_limit * len(tasks) * kv_scale))
    dp: dict[int, tuple[float, float, list[dict[str, Any]]]] = {0: (0.0, 0.0, [])}
    for task in tasks:
        next_dp: dict[int, tuple[float, float, list[dict[str, Any]]]] = {}
        for used, (score_sum, online_sum, choices) in dp.items():
            for cand in per_task[task]:
                units = int(round(float(cand["kv"]) * kv_scale))
                new_used = used + units
                if new_used > limit:
                    continue
                new_score = score_sum + float(cand["score"])
                new_online = online_sum + float(cand["online"])
                choice = {**cand, "task": task}
                old = next_dp.get(new_used)
                if old is None or (new_score, -new_online) > (old[0], -old[1]):
                    next_dp[new_used] = (new_score, new_online, choices + [choice])
        if not next_dp:
            raise RuntimeError(f"No feasible knapsack state after {task}")
        best_seen = -1.0
        pruned = {}
        for used in sorted(next_dp):
            if next_dp[used][0] > best_seen + 1e-10:
                pruned[used] = next_dp[used]
                best_seen = next_dp[used][0]
        dp = pruned
    used, best = max(dp.items(), key=lambda item: (item[1][0], -item[1][1], -item[0]))
    return {
        "score": best[0] / len(tasks),
        "kv": used / (kv_scale * len(tasks)),
        "online": best[1] / len(tasks),
        "choices": best[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--kv-limit", type=float, default=0.06)
    parser.add_argument("--kv-scale", type=int, default=10000)
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v407_task_knapsack06_exact_20260712.json")
    parser.add_argument("--summary-out", default="outputs/riskkv_v19_v407_task_knapsack06_exact_20260712/summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    tables = {}
    for action, spec in CANDIDATES.items():
        path = root / spec["results"]
        config = root / spec["config"]
        if path.exists() and config.exists():
            rows = read_csv(path)
            if len(rows) >= 1600:
                tables[action] = aggregate_by_task(rows)
    if "policy_v396" not in tables:
        raise FileNotFoundError("policy_v396 is required as the low-KV base")
    common_tasks = sorted(set.intersection(*[set(table) for table in tables.values()]))
    per_task = {}
    for task in common_tasks:
        candidates = []
        for action, table in tables.items():
            row = table[task]
            candidates.append({"action": action, **row})
        per_task[task] = non_dominated(candidates)

    selected = solve(per_task, args.kv_limit, args.kv_scale)
    config = {
        "__extends": "riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json",
        "__comment": (
            f"v407: exact task-level frontier at avg KV<={args.kv_limit:.1%}. "
            "Built from completed M100 task aggregates; intended as a stable point between v396 and v395."
        ),
        "__task_sources": {
            choice["task"]: {
                "policy": Path(CANDIDATES[choice["action"]]["config"]).name,
            }
            for choice in selected["choices"]
            if choice["action"] != "policy_v396"
        },
    }
    payload = {
        "kv_limit": args.kv_limit,
        "selected": selected,
        "available_actions": sorted(tables),
        "per_task_candidates": per_task,
        "config_out": args.config_out,
    }
    config_path = root / args.config_out
    summary_path = root / args.summary_out
    config_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(config_path)
    print(summary_path)
    print(json.dumps({"score": selected["score"], "kv": selected["kv"], "online": selected["online"], "sources": config["__task_sources"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
