#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")

FULL_ONLINE = 3.0988
V389_STATIC_SCORE = 0.39060259707091227
V389_STATIC_KV = 0.09928865774409969

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
}

BASE_BY_TASK = {
    "2wikimqa": "policy_v380",
    "gov_report": "policy_v376",
    "hotpotqa": "policy_v375",
    "lcc": "policy_v380",
    "multi_news": "policy_v375",
    "multifieldqa_en": "policy_v377",
    "musique": "policy_v381",
    "narrativeqa": "policy_v380",
    "passage_count": "policy_v368",
    "passage_retrieval_en": "policy_v381",
    "qasper": "policy_v377",
    "qmsum": "policy_v375",
    "repobench-p": "policy_v368",
    "samsum": "policy_v375",
    "trec": "policy_v375",
    "triviaqa": "policy_v368",
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


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def read_candidate_tables(root: Path) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    tables: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for action, spec in M100_CANDIDATES.items():
        path = root / spec["results"]
        config = root / spec["config"]
        if not path.exists() or not config.exists():
            continue
        rows = v378.read_csv(path)
        if len(rows) < 1600:
            continue
        tables[action] = {(row["task"], row["sample_id"]): row for row in rows}
    return tables


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return (
        v378.RUNTIME_NUMERIC_FEATURES
        + [f"family={family}" for family in sorted(set(v378.FAMILY_BY_TASK.values()) | {"other"})]
        + [f"task={task}" for task in sorted({str(row["task"]) for row in rows})]
    )


def vectorize(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        task = str(row["task"])
        family = v378.task_family(task)
        vector = []
        for name in names:
            if name in row:
                vector.append(float(row.get(name, 0.0) or 0.0))
            elif name.startswith("family="):
                vector.append(1.0 if name == f"family={family}" else 0.0)
            elif name.startswith("task="):
                vector.append(1.0 if name == f"task={task}" else 0.0)
            else:
                vector.append(0.0)
        matrix.append(vector)
    return matrix


def build_rows(tables: dict[str, dict[tuple[str, str], dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    common = set.intersection(*[set(table) for table in tables.values()])
    rows: list[dict[str, Any]] = []
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task, sample_id in sorted(common):
        base_action = BASE_BY_TASK[task]
        base_row = tables[base_action][(task, sample_id)]
        candidates = []
        for action, table in tables.items():
            row = table[(task, sample_id)]
            candidate = {
                "task": task,
                "sample_id": sample_id,
                "action": action,
                "score": fnum(row, "score"),
                "kv": fnum(row, "keep_fraction"),
                "online": fnum(row, "online_seconds"),
            }
            candidates.append(candidate)
            candidate_by_key[(task, sample_id, action)] = candidate
        oracle = max(candidates, key=lambda item: (float(item["score"]), -float(item["kv"])))
        record: dict[str, Any] = {
            "task": task,
            "task_family": v378.task_family(task),
            "sample_id": sample_id,
            "fold": v378.fold_for_key(task, sample_id),
            "base_action": base_action,
            "oracle_action": oracle["action"],
            "base_score": fnum(base_row, "score"),
            "base_kv": fnum(base_row, "keep_fraction"),
            "base_online": fnum(base_row, "online_seconds"),
            "oracle_score": oracle["score"],
            "oracle_kv": oracle["kv"],
            "oracle_online": oracle["online"],
        }
        for feature in v378.RUNTIME_NUMERIC_FEATURES:
            record[feature] = fnum(base_row, feature)
        rows.append(record)
    return rows, candidate_by_key


def predict_rows(
    rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]],
    model: Any,
    names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    vectors = vectorize(rows, names)
    actions = [str(item) for item in model.predict(vectors)]
    probabilities = model.predict_proba(vectors) if hasattr(model, "predict_proba") else None
    out = []
    for idx, (row, action) in enumerate(zip(rows, actions)):
        confidence = 1.0
        if probabilities is not None:
            confidence = float(max(probabilities[idx]))
        if confidence < threshold:
            chosen_action = "reference"
            score = float(row["base_score"])
            kv = float(row["base_kv"])
            online = float(row["base_online"])
        else:
            candidate = candidate_by_key.get((str(row["task"]), str(row["sample_id"]), action))
            if candidate is None:
                chosen_action = "reference"
                score = float(row["base_score"])
                kv = float(row["base_kv"])
                online = float(row["base_online"])
            else:
                chosen_action = action
                score = float(candidate["score"])
                kv = float(candidate["kv"])
                online = float(candidate["online"])
        out.append(
            {
                **row,
                "predicted_action": action,
                "chosen_action": chosen_action,
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
        score = mean([float(row["learned_score"]) for row in subset])
        kv = mean([float(row["learned_kv"]) for row in subset])
        online = mean([float(row["learned_online"]) for row in subset])
        base_score = mean([float(row["base_score"]) for row in subset])
        oracle_score = mean([float(row["oracle_score"]) for row in subset])
        out.append(
            {
                "split": split,
                "task": task,
                "samples": len(subset),
                "base_score": base_score,
                "learned_score": score,
                "oracle_score": oracle_score,
                "score_gain_vs_base": score - base_score,
                "oracle_gap_remaining": oracle_score - score,
                "base_kv": mean([float(row["base_kv"]) for row in subset]),
                "learned_kv": kv,
                "oracle_kv": mean([float(row["oracle_kv"]) for row in subset]),
                "learned_online": online,
                "learned_speed_vs_full": FULL_ONLINE / online if online > 0 else "",
                "fallback_rate": mean([1.0 if row["chosen_action"] == "reference" else 0.0 for row in subset]),
                "oracle_match_rate": mean([1.0 if row["chosen_action"] == row["oracle_action"] else 0.0 for row in subset]),
                "action_counts": json.dumps(Counter(str(row["chosen_action"]) for row in subset), sort_keys=True),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_winner_router_v390_m100_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v390_winner_router_m100_20260712.json")
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=V389_STATIC_SCORE)
    args = parser.parse_args()

    root = Path(args.root)
    tables = read_candidate_tables(root)
    if len(tables) < len(M100_CANDIDATES):
        missing = sorted(set(M100_CANDIDATES) - set(tables))
        print(f"WARNING missing candidates: {missing}")
    rows, candidate_by_key = build_rows(tables)
    names = feature_names(rows)
    train_rows = [row for row in rows if int(row["fold"]) not in {0, 1}]
    cal_rows = [row for row in rows if int(row["fold"]) == 1]
    test_rows = [row for row in rows if int(row["fold"]) == 0]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=320,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=390,
        n_jobs=-1,
    )
    model.fit(vectorize(train_rows, names), [str(row["oracle_action"]) for row in train_rows])

    threshold_rows = []
    feasible = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        pred = predict_rows(cal_rows, candidate_by_key, model, names, threshold)
        row = next(item for item in summarize(pred, "calibration") if item["task"] == "ALL")
        row["threshold"] = threshold
        row["feasible"] = int(float(row["learned_score"]) >= args.min_score and float(row["learned_kv"]) <= args.kv_limit)
        threshold_rows.append(row)
        if int(row["feasible"]):
            feasible.append(row)
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["learned_score"]),
                float(row["oracle_match_rate"]),
                -float(row["learned_kv"]),
            ),
        )
    else:
        selected = max(
            threshold_rows,
            key=lambda row: (
                float(row["learned_score"]),
                -abs(float(row["learned_kv"]) - args.kv_limit),
                float(row["oracle_match_rate"]),
            ),
        )
    threshold = float(selected["threshold"])

    prediction_rows = []
    summary_rows = []
    for split, split_rows in [("train", train_rows), ("calibration", cal_rows), ("test", test_rows), ("all", rows)]:
        pred = predict_rows(split_rows, candidate_by_key, model, names, threshold)
        for row in pred:
            row["split"] = split
            row["threshold"] = threshold
        prediction_rows.extend(pred)
        for row in summarize(pred, split):
            row["threshold"] = threshold
            summary_rows.append(row)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_specs = {action: spec for action, spec in M100_CANDIDATES.items() if action in tables}
    action_policy = v378.build_action_policy(root, candidate_specs)
    metadata = {
        "router_type": "winner_router_v390_m100",
        "feature_names": names,
        "confidence_fallback_threshold": threshold,
        "base_policy": "riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json",
        "candidate_results": {action: spec["results"] for action, spec in candidate_specs.items()},
        "candidate_configs": {action: spec["config"] for action, spec in candidate_specs.items()},
        "base_by_task": BASE_BY_TASK,
        "train_objective": "predict best completed-M100 action by sample score; fallback to v389 static on low confidence",
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "winner_predictions.csv", prediction_rows)
    write_csv(output_dir / "winner_summary.csv", summary_rows)
    write_csv(
        output_dir / "feature_importance.csv",
        [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(
                zip(names, model.feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
    )

    config = {
        "__extends": "riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json",
        "__overlay_all_tasks": {
            "action_router": True,
            "ours_action_router_mode": "learned_budget_overlay_v1",
            "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
            "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
            "ours_learned_router_confidence_threshold": threshold,
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "off",
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    all_summary = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    cal_summary = next(row for row in summary_rows if row["split"] == "calibration" and row["task"] == "ALL")
    test_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps({"all": all_summary, "calibration": cal_summary, "test": test_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
