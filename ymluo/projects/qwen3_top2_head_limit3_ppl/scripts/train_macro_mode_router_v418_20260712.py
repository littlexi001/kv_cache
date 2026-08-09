#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
FULL_SCORE = 0.36581658127460975
FULL_ONLINE = 3.0988

MACRO_CANDIDATES = {
    "operator_direct": {
        "results": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_confirm_m100_20260712_lowkv_confirm_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json",
    },
    "frontier_10": {
        "results": "outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v393_m100_task_knapsack_v385_20260712.json",
    },
    "frontier_075": {
        "results": "outputs/riskkv_v19_v395_m100_task_knapsack075_exact_20260712_m100_task_knapsack075_exact_v395_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v395_m100_task_knapsack075_exact_20260712.json",
    },
    "frontier_05": {
        "results": "outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json",
    },
    "legacy_sparse_128": {
        "results": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b128/task_results.csv",
        "config": "configs/riskkv_task_policy_v413_legacy_b128_20260712.json",
    },
    "legacy_sparse_256": {
        "results": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b256/task_results.csv",
        "config": "configs/riskkv_task_policy_v413_legacy_b256_20260712.json",
    },
    "legacy_sparse_512": {
        "results": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b512/task_results.csv",
        "config": "configs/riskkv_task_policy_v413_legacy_b512_20260712.json",
    },
}


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


def read_table(root: Path, rel_path: str) -> dict[tuple[str, str], dict[str, str]]:
    path = root / rel_path
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["task"], row["sample_id"]): row for row in csv.DictReader(handle)}


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return (
        v378.RUNTIME_NUMERIC_FEATURES
        + [f"family={family}" for family in sorted(set(v378.FAMILY_BY_TASK.values()) | {"other"})]
    )


def vectorize(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        family = v378.task_family(str(row["task"]))
        vector = []
        for name in names:
            if name in row:
                vector.append(float(row.get(name, 0.0) or 0.0))
            elif name.startswith("family="):
                vector.append(1.0 if name == f"family={family}" else 0.0)
            else:
                vector.append(0.0)
        matrix.append(vector)
    return matrix


def build_rows(
    root: Path,
    full_results: str,
    base_action: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, dict[str, str]]]:
    full = read_table(root, full_results)
    tables = {}
    specs = {}
    for action, spec in MACRO_CANDIDATES.items():
        if (root / spec["results"]).exists() and (root / spec["config"]).exists():
            tables[action] = read_table(root, spec["results"])
            specs[action] = spec
    if base_action not in tables:
        raise FileNotFoundError(f"base action {base_action!r} is unavailable")
    common = set(full) & set(tables[base_action])
    for table in tables.values():
        common &= set(table)
    rows: list[dict[str, Any]] = []
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in sorted(common):
        task, sample_id = key
        base_row = tables[base_action][key]
        full_row = full[key]
        row = {
            "task": task,
            "sample_id": sample_id,
            "fold": v378.fold_for_key(task, sample_id),
            "full_score": v378.fnum(full_row, "score"),
            "base_action": base_action,
            "base_score": v378.fnum(base_row, "score"),
            "base_kv": v378.fnum(base_row, "keep_fraction"),
            "base_online": v378.fnum(base_row, "online_seconds"),
        }
        for feature in v378.RUNTIME_NUMERIC_FEATURES:
            row[feature] = v378.fnum(base_row, feature)
        rows.append(row)
        for action, table in tables.items():
            cand = table[key]
            candidate_by_key[(task, sample_id, action)] = {
                "action": action,
                "score": v378.fnum(cand, "score"),
                "kv": v378.fnum(cand, "keep_fraction"),
                "online": v378.fnum(cand, "online_seconds"),
            }
    return rows, candidate_by_key, specs


def relabel(
    rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]],
    actions: list[str],
    penalty: float,
    min_score_floor: float,
) -> list[dict[str, Any]]:
    out = copy.deepcopy(rows)
    for row in out:
        task = str(row["task"])
        sample_id = str(row["sample_id"])
        choices = []
        relaxed = []
        for action in actions:
            cand = candidate_by_key[(task, sample_id, action)]
            score = float(cand["score"])
            kv = float(cand["kv"])
            item = (score - penalty * kv, score, -kv, action, cand)
            if score + 1e-12 >= min_score_floor * float(row["full_score"]):
                choices.append(item)
            relaxed.append(item)
        if not choices:
            choices = relaxed
        _, _, _, action, cand = max(choices)
        row["oracle_action"] = action
        row["oracle_score"] = float(cand["score"])
        row["oracle_kv"] = float(cand["kv"])
        row["oracle_online"] = float(cand["online"])
    return out


def predict(
    rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]],
    model: Any,
    names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    vectors = vectorize(rows, names)
    actions = [str(item) for item in model.predict(vectors)]
    probs = model.predict_proba(vectors) if hasattr(model, "predict_proba") else None
    out = []
    for idx, (row, action) in enumerate(zip(rows, actions)):
        confidence = 1.0
        if probs is not None and len(probs) and len(probs[idx]):
            confidence = float(max(probs[idx]))
        if confidence < threshold:
            learned_action = "reference"
            score = float(row["base_score"])
            kv = float(row["base_kv"])
            online = float(row["base_online"])
        else:
            cand = candidate_by_key.get((str(row["task"]), str(row["sample_id"]), action))
            if cand is None:
                learned_action = "reference"
                score = float(row["base_score"])
                kv = float(row["base_kv"])
                online = float(row["base_online"])
            else:
                learned_action = action
                score = float(cand["score"])
                kv = float(cand["kv"])
                online = float(cand["online"])
        out.append(
            {
                **row,
                "learned_action": learned_action,
                "confidence": confidence,
                "learned_score": score,
                "learned_kv": kv,
                "learned_online": online,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["ALL"].append(row)
        grouped[str(row["task"])].append(row)
    out = []
    for task, subset in sorted(grouped.items(), key=lambda item: (item[0] != "ALL", item[0])):
        full_score = mean([float(row["full_score"]) for row in subset])
        learned_score = mean([float(row["learned_score"]) for row in subset])
        base_score = mean([float(row["base_score"]) for row in subset])
        learned_kv = mean([float(row["learned_kv"]) for row in subset])
        base_kv = mean([float(row["base_kv"]) for row in subset])
        learned_online = mean([float(row["learned_online"]) for row in subset])
        base_online = mean([float(row["base_online"]) for row in subset])
        out.append(
            {
                "split": split,
                "task": task,
                "samples": len(subset),
                "full_score": full_score,
                "base_score": base_score,
                "learned_score": learned_score,
                "learned_vs_full": learned_score / max(1e-9, full_score),
                "base_vs_full": base_score / max(1e-9, full_score),
                "base_kv": base_kv,
                "learned_kv": learned_kv,
                "base_online": base_online,
                "learned_online": learned_online,
                "base_speed_vs_full": FULL_ONLINE / max(1e-9, base_online),
                "learned_speed_vs_full": FULL_ONLINE / max(1e-9, learned_online),
                "fallback_rate": mean([1.0 if row["learned_action"] == "reference" else 0.0 for row in subset]),
                "oracle_match_rate": mean([1.0 if row["learned_action"] == row["oracle_action"] else 0.0 for row in subset]),
                "mean_confidence": mean([float(row["confidence"]) for row in subset]),
            }
        )
    return out


def parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v418_macro_mode_router_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v418_macro_mode_router_20260712.json")
    parser.add_argument("--base-action", default="frontier_05")
    parser.add_argument("--kv-limit", type=float, default=0.045)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--min-score-floor", type=float, default=0.90)
    parser.add_argument("--penalties", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0")
    args = parser.parse_args()

    root = Path(args.root)
    rows, candidate_by_key, specs = build_rows(root, args.full_results, args.base_action)
    actions = sorted(specs)
    names = feature_names(rows)
    train_rows = [row for row in rows if int(row["fold"]) not in {0, 1}]
    cal_base = [row for row in rows if int(row["fold"]) == 1]
    test_base = [row for row in rows if int(row["fold"]) == 0]

    from sklearn.ensemble import RandomForestClassifier

    trials = []
    best: dict[str, Any] | None = None
    for penalty in parse_float_list(args.penalties):
        labeled = relabel(rows, candidate_by_key, actions, penalty, args.min_score_floor)
        train = [row for row in labeled if int(row["fold"]) not in {0, 1}]
        cal = [row for row in labeled if int(row["fold"]) == 1]
        test = [row for row in labeled if int(row["fold"]) == 0]
        model = RandomForestClassifier(
            n_estimators=360,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=418 + int(round(penalty * 100)),
            n_jobs=-1,
        )
        model.fit(vectorize(train, names), [str(row["oracle_action"]) for row in train])
        for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
            pred_cal = predict(cal, candidate_by_key, model, names, threshold)
            cal_summary = next(row for row in summarize(pred_cal, "calibration") if row["task"] == "ALL")
            feasible = (
                float(cal_summary["learned_vs_full"]) >= args.quality_ratio
                and float(cal_summary["learned_kv"]) <= args.kv_limit
            )
            trial = {
                "penalty": penalty,
                "threshold": threshold,
                "cal_score": cal_summary["learned_score"],
                "cal_vs_full": cal_summary["learned_vs_full"],
                "cal_kv": cal_summary["learned_kv"],
                "cal_speed": cal_summary["learned_speed_vs_full"],
                "cal_fallback": cal_summary["fallback_rate"],
                "feasible": int(feasible),
            }
            trials.append(trial)
            key = (
                int(feasible),
                -float(cal_summary["learned_kv"]),
                float(cal_summary["learned_score"]),
                float(cal_summary["oracle_match_rate"]),
            )
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "penalty": penalty,
                    "threshold": threshold,
                    "model": model,
                    "labeled": labeled,
                    "train": train,
                    "cal": cal,
                    "test": test,
                }

    assert best is not None
    split_rows = {
        "train": [row for row in best["labeled"] if int(row["fold"]) not in {0, 1}],
        "calibration": [row for row in best["labeled"] if int(row["fold"]) == 1],
        "test": [row for row in best["labeled"] if int(row["fold"]) == 0],
        "all": best["labeled"],
    }
    predictions = []
    summaries = []
    for split, split_base in split_rows.items():
        pred = predict(split_base, candidate_by_key, best["model"], names, float(best["threshold"]))
        for row in pred:
            row["split"] = split
            row["penalty"] = best["penalty"]
            row["threshold"] = best["threshold"]
        predictions.extend(pred)
        summaries.extend(summarize(pred, split))

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = v378.build_action_policy(root, specs)
    metadata = {
        "router_type": "macro_mode_router_v418",
        "feature_names": names,
        "confidence_fallback_threshold": best["threshold"],
        "cost_penalty": best["penalty"],
        "base_action": args.base_action,
        "base_policy": Path(specs[args.base_action]["config"]).name,
        "available_actions": actions,
        "candidate_results": {action: spec["results"] for action, spec in specs.items()},
        "candidate_configs": {action: spec["config"] for action, spec in specs.items()},
        "uses_task_one_hot": False,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": best["model"], "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "penalty_threshold_trials.csv", trials)
    write_csv(output_dir / "router_predictions.csv", predictions)
    write_csv(output_dir / "router_summary.csv", summaries)
    write_csv(
        output_dir / "feature_importance.csv",
        [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(
                zip(names, best["model"].feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
    )
    config = {
        "__extends": Path(specs[args.base_action]["config"]).name,
        "__comment": (
            "v418: task-one-hot-free macro-mode router over operator/direct, low-KV frontier, "
            "and legacy sparse retrieval actions. Trained with cost-aware labels and confidence fallback."
        ),
        "__overlay_all_tasks": {
            "action_router": True,
            "ours_action_router_mode": "learned_budget_overlay_v1",
            "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
            "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
            "ours_learned_router_confidence_threshold": best["threshold"],
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "off",
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    all_summary = next(row for row in summaries if row["split"] == "all" and row["task"] == "ALL")
    cal_summary = next(row for row in summaries if row["split"] == "calibration" and row["task"] == "ALL")
    test_summary = next(row for row in summaries if row["split"] == "test" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps({"all": all_summary, "calibration": cal_summary, "test": test_summary, "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
