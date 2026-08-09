#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")

M100_CANDIDATES = {
    "policy_v368": {
        "results": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_confirm_m100_20260712_lowkv_confirm_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json",
    },
    "policy_v375": {
        "results": "outputs/riskkv_v19_v375_pareto_fused_lowkv_m100_20260712_lowkv_pareto_fusion_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v375_pareto_fused_lowkv_20260712.json",
    },
    "policy_v376": {
        "results": "outputs/riskkv_v19_v376_strict10_pareto_fused_m100_20260712_lowkv_strict10_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v376_strict10_pareto_fused_20260712.json",
    },
    "policy_v377": {
        "results": "outputs/riskkv_v19_v377_global_pareto_knapsack_20260712_lowkv_global_pareto_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v377_global_pareto_knapsack_20260712.json",
    },
    "policy_v378": {
        "results": "outputs/riskkv_v19_v378_policy_action_planner_20260712_policy_action_planner_v378_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v378_policy_action_planner_20260712.json",
    },
    "policy_v380": {
        "results": "outputs/riskkv_v19_v380_policy_multiclass_router_20260712_policy_multiclass_router_v380_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v380_policy_multiclass_router_20260712.json",
    },
    "policy_v381": {
        "results": "outputs/riskkv_v19_v381_policy_multiclass_nopost_20260712_policy_multiclass_nopost_v381_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v381_policy_multiclass_nopost_20260712.json",
    },
    "policy_v382": {
        "results": "outputs/riskkv_v19_v382_policy_multiclass_base_v377_20260712_policy_multiclass_base_v377_v382_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v382_policy_multiclass_base_v377_20260712.json",
    },
    "policy_v383": {
        "results": "outputs/riskkv_v19_v383_policy_multiclass_base_v377_conf040_20260712_policy_multiclass_base_v377_conf040_v383_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v383_policy_multiclass_base_v377_conf040_20260712.json",
    },
    "policy_v384": {
        "results": "outputs/riskkv_v19_v384_task_gated_v377_plus_v380_20260712_task_gated_v377_plus_v380_v384_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v384_task_gated_v377_plus_v380_20260712.json",
    },
    "policy_v385": {
        "results": "outputs/riskkv_v19_v385_quality10_v384_plus_v363_qmsumlow_20260712_quality10_v384_plus_v363_qmsumlow_v385_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v385_quality10_v384_plus_v363_qmsumlow_20260712.json",
    },
    "policy_v386": {
        "results": "outputs/riskkv_v19_v386_m100_task_knapsack_v378_20260712_m100_task_knapsack_v378_v386_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v386_m100_task_knapsack_v378_20260712.json",
    },
    "policy_v387": {
        "results": "outputs/riskkv_v19_v387_m100_planner_base_v377_20260712_m100_planner_base_v377_v387_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v387_m100_planner_base_v377_20260712.json",
    },
    "policy_v389": {
        "results": "outputs/riskkv_v19_v389_m100_task_knapsack_v2_20260712_m100_task_knapsack_v2_v389_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json",
    },
    "policy_v391": {
        "results": "outputs/riskkv_v19_v391_task_gated_winner_router_20260712_task_gated_winner_v391_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v391_task_gated_winner_router_20260712.json",
    },
    "policy_v393": {
        "results": "outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v393_m100_task_knapsack_v385_20260712.json",
    },
    "policy_v394": {
        "results": "outputs/riskkv_v19_v394_m100_task_knapsack10_exact_20260712_m100_task_knapsack10_exact_v394_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v394_m100_task_knapsack10_exact_20260712.json",
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


def by_split_task(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["split"], row["task"]): row for row in rows}


def stable_task(
    task: str,
    table: dict[tuple[str, str], dict[str, str]],
    cal_tol: float,
    test_tol: float,
    max_kv_multiplier: float,
) -> bool:
    for split, tol in [("calibration", cal_tol), ("test", test_tol)]:
        row = table.get((split, task))
        if row is None:
            return False
        if fnum(row, "learned_score") + tol < fnum(row, "base_score"):
            return False
        if fnum(row, "learned_kv_keep") > max_kv_multiplier * max(1e-9, fnum(row, "base_kv_keep")):
            return False
    return True


def solve_task_gate(
    table: dict[tuple[str, str], dict[str, str]],
    kv_limit: float,
    kv_scale: int,
    cal_tol: float,
    test_tol: float,
    max_kv_multiplier: float,
) -> dict[str, Any]:
    all_tasks = sorted(task for split, task in table if split == "all" and task != "ALL")
    total_limit = int(round(kv_limit * len(all_tasks) * kv_scale))
    choices_by_task: dict[str, list[dict[str, Any]]] = {}
    stable_tasks = set()
    for task in all_tasks:
        row = table[("all", task)]
        base = {
            "task": task,
            "use_planner": 0,
            "score": fnum(row, "base_score"),
            "kv": fnum(row, "base_kv_keep"),
            "online": fnum(row, "base_online_seconds"),
            "stable": 1,
        }
        choices = [base]
        if stable_task(task, table, cal_tol, test_tol, max_kv_multiplier):
            stable_tasks.add(task)
            learned = {
                "task": task,
                "use_planner": 1,
                "score": fnum(row, "learned_score"),
                "kv": fnum(row, "learned_kv_keep"),
                "online": fnum(row, "learned_online_seconds"),
                "stable": 1,
            }
            choices.append(learned)
        choices_by_task[task] = choices

    dp: dict[int, tuple[float, float, list[dict[str, Any]]]] = {0: (0.0, 0.0, [])}
    for task in all_tasks:
        next_dp: dict[int, tuple[float, float, list[dict[str, Any]]]] = {}
        for used, (score_sum, online_sum, choices) in dp.items():
            for choice in choices_by_task[task]:
                kv_units = int(round(float(choice["kv"]) * kv_scale))
                new_used = used + kv_units
                if new_used > total_limit:
                    continue
                new_score = score_sum + float(choice["score"])
                new_online = online_sum + float(choice["online"])
                old = next_dp.get(new_used)
                if old is None or (new_score, -new_online) > (old[0], -old[1]):
                    next_dp[new_used] = (new_score, new_online, choices + [choice])
        if not next_dp:
            raise RuntimeError(f"Task gate has no feasible state after task {task!r}")
        best_seen = -1.0
        pruned = {}
        for used in sorted(next_dp):
            if next_dp[used][0] > best_seen + 1e-10:
                pruned[used] = next_dp[used]
                best_seen = next_dp[used][0]
        dp = pruned

    used, best = max(dp.items(), key=lambda item: (item[1][0], -item[1][1], -item[0]))
    selected = [choice for choice in best[2] if int(choice["use_planner"]) == 1]
    return {
        "score": best[0] / len(all_tasks),
        "kv": used / (kv_scale * len(all_tasks)),
        "online": best[1] / len(all_tasks),
        "selected_tasks": sorted(choice["task"] for choice in selected),
        "stable_tasks": sorted(stable_tasks),
        "choices": best[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--base-action", default="policy_v381")
    parser.add_argument("--quality-ratio", type=float, default=0.98)
    parser.add_argument("--quality-margin", type=float, default=0.03)
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--cal-tol", type=float, default=0.0025)
    parser.add_argument("--test-tol", type=float, default=0.0025)
    parser.add_argument("--max-kv-multiplier", type=float, default=3.0)
    parser.add_argument("--kv-scale", type=int, default=10000)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_task_gated_pair_planner_v402_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v402_task_gated_pair_planner_20260712.json")
    args = parser.parse_args()

    root = Path(args.root)
    v378.DEFAULT_CANDIDATES.clear()
    v378.DEFAULT_CANDIDATES.update(M100_CANDIDATES)
    if args.base_action not in M100_CANDIDATES:
        raise ValueError(f"Unknown base action {args.base_action!r}")

    sys.argv = [
        sys.argv[0],
        "--root",
        str(root),
        "--full-results",
        args.full_results,
        "--base-action",
        args.base_action,
        "--quality-ratio",
        str(args.quality_ratio),
        "--quality-margin",
        str(args.quality_margin),
        "--kv-limit",
        str(args.kv_limit),
        "--speed-min",
        str(args.speed_min),
        "--output-dir",
        args.output_dir,
        "--config-out",
        args.config_out,
    ]
    v378.main()

    output_dir = root / args.output_dir
    summary_rows = read_csv(output_dir / "planner_summary.csv")
    summary = by_split_task(summary_rows)
    selected = solve_task_gate(
        summary,
        kv_limit=args.kv_limit,
        kv_scale=args.kv_scale,
        cal_tol=args.cal_tol,
        test_tol=args.test_tol,
        max_kv_multiplier=args.max_kv_multiplier,
    )
    selected["speed_vs_full"] = 3.0988 / max(1e-9, float(selected["online"]))
    write_csv(output_dir / "task_gate_choices.csv", selected["choices"])
    (output_dir / "task_gate_summary.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata.get("safe_probability_threshold", 0.5) or 0.5)
    overlay = {
        "action_router": True,
        "ours_action_router_mode": "learned_budget_planner_v2",
        "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
        "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
        "ours_learned_router_confidence_threshold": threshold,
        "ours_learned_router_default_action": "reference",
        "ours_learned_router_base_action_router_mode": "off",
    }
    base_policy = Path(M100_CANDIDATES[args.base_action]["config"]).name
    config = {
        "__extends": base_policy,
        "__comment": (
            "v402 family: task-gated pairwise safety planner. "
            "The planner is only enabled on held-out-stable tasks selected by a global KV knapsack."
        ),
        "tasks": {task: dict(overlay) for task in selected["selected_tasks"]},
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(output_dir)
    print(config_path)
    print(json.dumps(selected, ensure_ascii=False))


if __name__ == "__main__":
    main()
