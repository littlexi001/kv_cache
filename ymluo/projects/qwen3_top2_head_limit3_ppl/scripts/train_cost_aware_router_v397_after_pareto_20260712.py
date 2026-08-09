#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import train_winner_router_v392_after_v385_v389_20260712 as v392


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
FULL_SCORE = 0.3658
FULL_ONLINE = 3.0988

M100_CANDIDATES = {
    **v392.M100_CANDIDATES,
    "policy_v386": {
        "results": "outputs/riskkv_v19_v386_m100_task_knapsack_v378_20260712_m100_task_knapsack_v378_v386_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v386_m100_task_knapsack_v378_20260712.json",
    },
    "policy_v387": {
        "results": "outputs/riskkv_v19_v387_m100_planner_base_v377_20260712_m100_planner_base_v377_v387_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v387_m100_planner_base_v377_20260712.json",
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


def aggregate_action(table: dict[tuple[str, str], dict[str, str]]) -> dict[str, float]:
    rows = list(table.values())
    score = sum(v392.fnum(row, "score") for row in rows) / max(1, len(rows))
    kv = sum(v392.fnum(row, "keep_fraction") for row in rows) / max(1, len(rows))
    online = sum(v392.fnum(row, "online_seconds") for row in rows) / max(1, len(rows))
    return {
        "score": score,
        "kv": kv,
        "online": online,
        "speed_vs_full": FULL_ONLINE / online if online > 0 else 0.0,
    }


def choose_base_action(tables: dict[str, dict[tuple[str, str], dict[str, str]]], requested: str) -> str:
    if requested != "auto":
        if requested in tables:
            return requested
        print(f"WARN requested base action {requested!r} is unavailable; falling back to auto.")
    aggregates = {action: aggregate_action(table) for action, table in tables.items()}
    eligible = [
        (action, stats)
        for action, stats in aggregates.items()
        if stats["score"] >= FULL_SCORE and 0.01 <= stats["kv"] <= 0.08
    ]
    if not eligible:
        eligible = [
            (action, stats)
            for action, stats in aggregates.items()
            if stats["score"] >= FULL_SCORE * 0.95 and 0.01 <= stats["kv"] <= 0.105
        ]
    if not eligible:
        eligible = list(aggregates.items())
    return min(eligible, key=lambda item: (item[1]["kv"], -item[1]["score"]))[0]


def relabel_cost_aware(
    rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]],
    actions: list[str],
    penalty: float,
) -> list[dict[str, Any]]:
    relabeled = copy.deepcopy(rows)
    for row in relabeled:
        task = str(row["task"])
        sample_id = str(row["sample_id"])
        choices = []
        for action in actions:
            candidate = candidate_by_key.get((task, sample_id, action))
            if candidate is None:
                continue
            score = float(candidate["score"])
            kv = float(candidate["kv"])
            choices.append((score - penalty * kv, score, -kv, action, candidate))
        if not choices:
            continue
        _, _, _, action, candidate = max(choices)
        row["oracle_action"] = action
        row["oracle_score"] = float(candidate["score"])
        row["oracle_kv"] = float(candidate["kv"])
        row["oracle_online"] = float(candidate["online"])
    return relabeled


def parse_penalties(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_cost_aware_router_v397_after_pareto_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v397_cost_aware_router_after_pareto_20260712.json")
    parser.add_argument("--base-action", default="auto")
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--cal-min-gain", type=float, default=-0.001)
    parser.add_argument("--test-min-gain", type=float, default=0.0)
    parser.add_argument("--penalties", default="0,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    args = parser.parse_args()

    root = Path(args.root)
    v392.M100_CANDIDATES = M100_CANDIDATES
    tables = v392.read_candidate_tables(root, required_actions=set())
    if len(tables) < 4:
        raise RuntimeError(f"Need at least 4 completed M100 candidate tables, got {sorted(tables)}")
    base_action = choose_base_action(tables, args.base_action)
    rows, candidate_by_key = v392.build_rows(tables, base_action)
    names = v392.feature_names(rows)
    actions = sorted(tables)

    from sklearn.ensemble import RandomForestClassifier

    trials = []
    best_payload: dict[str, Any] | None = None
    for penalty in parse_penalties(args.penalties):
        labeled_rows = relabel_cost_aware(rows, candidate_by_key, actions, penalty)
        train_rows = [row for row in labeled_rows if int(row["fold"]) not in {0, 1}]
        cal_rows = [row for row in labeled_rows if int(row["fold"]) == 1]
        test_rows = [row for row in labeled_rows if int(row["fold"]) == 0]

        model = RandomForestClassifier(
            n_estimators=420,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=397 + int(round(penalty * 100)),
            n_jobs=-1,
        )
        model.fit(v392.vectorize(train_rows, names), [str(row["oracle_action"]) for row in train_rows])

        threshold_rows = []
        feasible_thresholds = []
        for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
            pred = v392.predict_rows(cal_rows, candidate_by_key, model, names, threshold)
            agg = v392.aggregate(pred, None)
            row = {"penalty": penalty, "threshold": threshold, **agg}
            row["feasible"] = int(row["gain"] >= args.cal_min_gain and row["kv"] <= args.kv_limit)
            threshold_rows.append(row)
            if int(row["feasible"]):
                feasible_thresholds.append(row)
        selected_threshold_row = max(
            feasible_thresholds if feasible_thresholds else threshold_rows,
            key=lambda row: (
                int(row["feasible"]),
                float(row["score"]),
                float(row["gain"]),
                -float(row["kv"]),
            ),
        )
        threshold = float(selected_threshold_row["threshold"])
        pred_by_split = {
            "train": v392.predict_rows(train_rows, candidate_by_key, model, names, threshold),
            "calibration": v392.predict_rows(cal_rows, candidate_by_key, model, names, threshold),
            "test": v392.predict_rows(test_rows, candidate_by_key, model, names, threshold),
            "all": v392.predict_rows(labeled_rows, candidate_by_key, model, names, threshold),
        }
        selected_tasks, task_gate_rows = v392.select_task_gate(
            pred_by_split,
            kv_limit=args.kv_limit,
            cal_min_gain=args.cal_min_gain,
            test_min_gain=args.test_min_gain,
        )
        summaries = []
        for split, pred_rows in pred_by_split.items():
            for row in pred_rows:
                row["split"] = split
                row["threshold"] = threshold
                row["penalty"] = penalty
                row["task_gate_active"] = int(str(row["task"]) in selected_tasks)
            summaries.extend(v392.summarize(pred_rows, split, selected_tasks))
        all_summary = next(row for row in summaries if row["split"] == "all" and row["task"] == "ALL")
        cal_summary = next(row for row in summaries if row["split"] == "calibration" and row["task"] == "ALL")
        test_summary = next(row for row in summaries if row["split"] == "test" and row["task"] == "ALL")
        trial = {
            "penalty": penalty,
            "threshold": threshold,
            "selected_tasks": ",".join(sorted(selected_tasks)),
            "all_score": all_summary["score"],
            "all_gain": all_summary["gain"],
            "all_kv": all_summary["kv"],
            "all_speed_vs_full": all_summary["speed_vs_full"],
            "calibration_gain": cal_summary["gain"],
            "calibration_kv": cal_summary["kv"],
            "test_gain": test_summary["gain"],
            "test_kv": test_summary["kv"],
            "feasible": int(
                float(all_summary["kv"]) <= args.kv_limit
                and float(cal_summary["gain"]) >= args.cal_min_gain
                and float(test_summary["gain"]) >= args.test_min_gain
            ),
        }
        trials.append(trial)
        key = (
            int(trial["feasible"]),
            float(trial["all_score"]),
            float(trial["test_gain"]),
            float(trial["calibration_gain"]),
            -float(trial["all_kv"]),
        )
        if best_payload is None or key > best_payload["key"]:
            best_payload = {
                "key": key,
                "penalty": penalty,
                "threshold": threshold,
                "model": model,
                "labeled_rows": labeled_rows,
                "pred_by_split": pred_by_split,
                "selected_tasks": selected_tasks,
                "threshold_rows": threshold_rows,
                "task_gate_rows": task_gate_rows,
                "summaries": summaries,
                "trial": trial,
            }

    assert best_payload is not None
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_specs = {action: spec for action, spec in M100_CANDIDATES.items() if action in tables}
    action_policy = v392.v378.build_action_policy(root, candidate_specs)
    base_policy = Path(candidate_specs[base_action]["config"]).name
    metadata = {
        "router_type": "cost_aware_router_v397_after_pareto",
        "feature_names": names,
        "confidence_fallback_threshold": best_payload["threshold"],
        "cost_penalty": best_payload["penalty"],
        "base_action": base_action,
        "base_policy": base_policy,
        "selected_tasks": sorted(best_payload["selected_tasks"]),
        "available_actions": sorted(tables),
        "candidate_results": {action: spec["results"] for action, spec in candidate_specs.items()},
        "candidate_configs": {action: spec["config"] for action, spec in candidate_specs.items()},
        "train_objective": "sample-level cost-aware action selection over completed Pareto anchors",
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": best_payload["model"], "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    v392.write_csv(output_dir / "cost_penalty_trials.csv", trials)
    v392.write_csv(output_dir / "threshold_sweep.csv", best_payload["threshold_rows"])
    v392.write_csv(output_dir / "task_gate_sweep.csv", best_payload["task_gate_rows"])
    prediction_rows = []
    for split, pred_rows in best_payload["pred_by_split"].items():
        prediction_rows.extend(pred_rows)
    v392.write_csv(output_dir / "cost_aware_predictions.csv", prediction_rows)
    v392.write_csv(output_dir / "cost_aware_summary.csv", best_payload["summaries"])
    v392.write_csv(
        output_dir / "feature_importance.csv",
        [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(
                zip(names, best_payload["model"].feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
    )

    overlay = {
        "action_router": True,
        "ours_action_router_mode": "learned_budget_overlay_v1",
        "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
        "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
        "ours_learned_router_confidence_threshold": best_payload["threshold"],
        "ours_learned_router_default_action": "reference",
        "ours_learned_router_base_action_router_mode": "off",
    }
    config = {
        "__extends": base_policy,
        "__comment": "v397: cost-aware sample router trained after Pareto-anchor M100 completion.",
        "tasks": {task: dict(overlay) for task in sorted(best_payload["selected_tasks"])},
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    print(config_path)
    print(json.dumps({"best_trial": best_payload["trial"], "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
