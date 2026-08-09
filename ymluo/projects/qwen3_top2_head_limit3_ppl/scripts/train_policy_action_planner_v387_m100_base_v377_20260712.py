#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
CONFIG_OUT = "configs/riskkv_task_policy_v387_m100_planner_base_v377_20260712.json"
OUTPUT_DIR = "outputs/riskkv_v19_policy_action_planner_v387_m100_base_v377_20260712"

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
}


def main() -> None:
    v378.DEFAULT_CANDIDATES.clear()
    v378.DEFAULT_CANDIDATES.update(M100_CANDIDATES)
    sys.argv = [
        sys.argv[0],
        "--root",
        str(ROOT),
        "--base-action",
        "policy_v377",
        "--quality-ratio",
        "0.95",
        "--quality-margin",
        "0.05",
        "--kv-limit",
        "0.10",
        "--speed-min",
        "2.5",
        "--output-dir",
        OUTPUT_DIR,
        "--config-out",
        CONFIG_OUT,
    ]
    v378.main()

    config_path = ROOT / CONFIG_OUT
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["__extends"] = "riskkv_task_policy_v377_global_pareto_knapsack_20260712.json"
    overlay = config.setdefault("__overlay_all_tasks", {})
    overlay["ours_learned_router_base_action_router_mode"] = "off"
    overlay["ours_learned_router_default_action"] = "reference"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(config_path)
    print(json.dumps(config, ensure_ascii=False))


if __name__ == "__main__":
    main()
