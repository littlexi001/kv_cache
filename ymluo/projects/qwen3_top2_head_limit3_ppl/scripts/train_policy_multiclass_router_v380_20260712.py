#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import train_policy_action_planner_v378_20260712 as v378


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


def make_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return (
        v378.RUNTIME_NUMERIC_FEATURES
        + [f"family={family}" for family in sorted(set(v378.FAMILY_BY_TASK.values()) | {"other"})]
        + [f"task={task}" for task in sorted({str(row["task"]) for row in rows})]
    )


def vectorize(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    matrix = []
    for row in rows:
        task = str(row["task"])
        family = v378.task_family(task)
        vector = []
        for name in feature_names:
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


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize(rows: list[dict[str, Any]], split: str, full_online: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["ALL"].append(row)
        grouped[str(row["task"])].append(row)
    out = []
    for task, subset in sorted(grouped.items(), key=lambda item: (item[0] != "ALL", item[0])):
        full_score = mean([float(row["full_score"]) for row in subset])
        learned_score = mean([float(row["learned_score"]) for row in subset])
        oracle_score = mean([float(row["oracle_score"]) for row in subset])
        base_score = mean([float(row["base_score"]) for row in subset])
        learned_kv = mean([float(row["learned_kv_keep"]) for row in subset])
        oracle_kv = mean([float(row["oracle_kv_keep"]) for row in subset])
        base_kv = mean([float(row["base_kv_keep"]) for row in subset])
        learned_online = mean([float(row["learned_online_seconds"]) for row in subset])
        oracle_online = mean([float(row["oracle_online_seconds"]) for row in subset])
        base_online = mean([float(row["base_online_seconds"]) for row in subset])
        out.append(
            {
                "split": split,
                "task": task,
                "samples": len(subset),
                "full_score": full_score,
                "base_score": base_score,
                "learned_score": learned_score,
                "oracle_score": oracle_score,
                "learned_vs_full": learned_score / full_score if full_score > 0 else "",
                "oracle_vs_full": oracle_score / full_score if full_score > 0 else "",
                "base_vs_full": base_score / full_score if full_score > 0 else "",
                "base_kv_keep": base_kv,
                "learned_kv_keep": learned_kv,
                "oracle_kv_keep": oracle_kv,
                "base_online_seconds": base_online,
                "learned_online_seconds": learned_online,
                "oracle_online_seconds": oracle_online,
                "learned_speed_vs_full": full_online / learned_online if learned_online > 0 else "",
                "oracle_speed_vs_full": full_online / oracle_online if oracle_online > 0 else "",
                "base_speed_vs_full": full_online / base_online if base_online > 0 else "",
                "fallback_rate": mean([1.0 if row["learned_action"] == "reference" else 0.0 for row in subset]),
                "oracle_match_rate": mean([1.0 if row["learned_action"] == row["oracle_action"] else 0.0 for row in subset]),
                "mean_confidence": mean([float(row["confidence"]) for row in subset]),
            }
        )
    return out


def build_training_rows(
    root: Path,
    full_results: str,
    base_action: str,
    quality_ratio: float,
    quality_margin: float,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, dict[str, str]],
]:
    full_rows = v378.by_key(v378.read_csv(root / full_results))
    candidate_tables = {}
    candidate_specs = {}
    for action, spec in v378.DEFAULT_CANDIDATES.items():
        path = root / spec["results"]
        config_path = root / spec["config"]
        if path.exists() and config_path.exists():
            table = v378.by_key(v378.read_csv(path))
            if table:
                candidate_tables[action] = table
                candidate_specs[action] = spec
    if base_action not in candidate_tables:
        raise FileNotFoundError(f"Missing base action results: {base_action}")
    common_keys = set(full_rows) & set(candidate_tables[base_action])
    for table in candidate_tables.values():
        common_keys &= set(table)
    candidate_costs = {
        action: v378.candidate_cost_tokens([row for key, row in table.items() if key in common_keys])
        for action, table in candidate_tables.items()
    }
    actions_by_cost = sorted(candidate_tables, key=lambda action: (candidate_costs[action], action))
    base_rows = []
    pair_by_key = {}
    for key in sorted(common_keys):
        base_row = candidate_tables[base_action][key]
        full_row = full_rows[key]
        base = v378.feature_base(base_row, full_row, quality_ratio, quality_margin)
        pairs = []
        for action in actions_by_cost:
            cand = candidate_tables[action][key]
            pair = {
                "task": base["task"],
                "sample_id": base["sample_id"],
                "action": action,
                "candidate_score": v378.fnum(cand, "score"),
                "candidate_kv_keep": v378.fnum(cand, "keep_fraction"),
                "candidate_online_seconds": v378.fnum(cand, "online_seconds"),
                "is_safe": int(v378.fnum(cand, "score") + 1e-12 >= float(base["quality_target"])),
            }
            pairs.append(pair)
            pair_by_key[(str(base["task"]), str(base["sample_id"]), action)] = pair
        safe_pairs = [row for row in pairs if int(row["is_safe"]) == 1]
        if safe_pairs:
            oracle = min(safe_pairs, key=lambda row: (float(row["candidate_kv_keep"]), -float(row["candidate_score"])))
        else:
            oracle = max(pairs, key=lambda row: (float(row["candidate_score"]), -float(row["candidate_kv_keep"])))
        base.update(
            {
                "oracle_action": str(oracle["action"]),
                "oracle_score": float(oracle["candidate_score"]),
                "oracle_kv_keep": float(oracle["candidate_kv_keep"]),
                "oracle_online_seconds": float(oracle["candidate_online_seconds"]),
            }
        )
        base_rows.append(base)
    return base_rows, pair_by_key, candidate_specs


def predict_rows(
    rows: list[dict[str, Any]],
    pair_by_key: dict[tuple[str, str, str], dict[str, Any]],
    model: Any,
    feature_names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    vectors = vectorize(rows, feature_names)
    actions = [str(item) for item in model.predict(vectors)]
    probabilities = model.predict_proba(vectors)
    out = []
    for row, action, probs in zip(rows, actions, probabilities):
        confidence = float(max(probs))
        if confidence < threshold:
            learned_action = "reference"
            learned_score = float(row["base_score"])
            learned_kv = float(row["base_kv_keep"])
            learned_online = float(row["base_online_seconds"])
        else:
            pair = pair_by_key.get((str(row["task"]), str(row["sample_id"]), action))
            if pair is None:
                learned_action = "reference"
                learned_score = float(row["base_score"])
                learned_kv = float(row["base_kv_keep"])
                learned_online = float(row["base_online_seconds"])
            else:
                learned_action = action
                learned_score = float(pair["candidate_score"])
                learned_kv = float(pair["candidate_kv_keep"])
                learned_online = float(pair["candidate_online_seconds"])
        out.append(
            {
                **row,
                "learned_action": learned_action,
                "confidence": confidence,
                "learned_score": learned_score,
                "learned_kv_keep": learned_kv,
                "learned_online_seconds": learned_online,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v378.ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--base-action", default="policy_v365")
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--quality-margin", type=float, default=0.05)
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--full-online", type=float, default=3.0988)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_policy_multiclass_router_v380_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v380_policy_multiclass_router_20260712.json")
    args = parser.parse_args()

    root = Path(args.root)
    base_rows, pair_by_key, candidate_specs = build_training_rows(
        root,
        args.full_results,
        args.base_action,
        args.quality_ratio,
        args.quality_margin,
    )
    feature_names = make_feature_names(base_rows)
    train_rows = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    cal_rows = [row for row in base_rows if int(row["fold"]) == 1]
    test_rows = [row for row in base_rows if int(row["fold"]) == 0]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=23,
        n_jobs=-1,
    )
    model.fit(vectorize(train_rows, feature_names), [str(row["oracle_action"]) for row in train_rows])

    threshold_rows = []
    feasible = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        pred = predict_rows(cal_rows, pair_by_key, model, feature_names, threshold)
        row = next(item for item in summarize(pred, "calibration", args.full_online) if item["task"] == "ALL")
        learned_vs_full = float(row["learned_vs_full"] or 0.0)
        learned_kv = float(row["learned_kv_keep"])
        learned_speed = float(row["learned_speed_vs_full"] or 0.0)
        row = {
            **row,
            "threshold": threshold,
            "feasible": int(learned_vs_full >= args.quality_ratio and learned_kv <= args.kv_limit and learned_speed >= args.speed_min),
        }
        threshold_rows.append(row)
        if int(row["feasible"]):
            feasible.append(row)
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["learned_score"]),
                float(row["oracle_match_rate"]),
                -float(row["learned_kv_keep"]),
            ),
        )
    else:
        selected = max(
            threshold_rows,
            key=lambda row: (
                float(row["learned_score"]),
                -float(row["learned_kv_keep"]),
                float(row["oracle_match_rate"]),
            ),
        )
    threshold = float(selected["threshold"])

    prediction_rows = []
    summary_rows = []
    for split, rows in [("train", train_rows), ("calibration", cal_rows), ("test", test_rows), ("all", base_rows)]:
        pred = predict_rows(rows, pair_by_key, model, feature_names, threshold)
        for row in pred:
            row["split"] = split
            row["threshold"] = threshold
        prediction_rows.extend(pred)
        for row in summarize(pred, split, args.full_online):
            row["threshold"] = threshold
            summary_rows.append(row)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = v378.build_action_policy(root, candidate_specs)
    metadata = {
        "router_type": "policy_multiclass_router_v380",
        "feature_names": feature_names,
        "confidence_fallback_threshold": threshold,
        "quality_ratio": args.quality_ratio,
        "quality_margin": args.quality_margin,
        "kv_limit": args.kv_limit,
        "speed_min": args.speed_min,
        "base_action": args.base_action,
        "candidate_results": {action: spec["results"] for action, spec in candidate_specs.items()},
        "candidate_configs": {action: spec["config"] for action, spec in candidate_specs.items()},
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "router_predictions.csv", prediction_rows)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(
        output_dir / "feature_importance.csv",
        [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(zip(feature_names, model.feature_importances_), key=lambda item: float(item[1]), reverse=True)
        ],
    )

    config = {
        "__extends": "riskkv_task_policy_v365_ultra_skeleton_all_20260712.json",
        "__overlay_all_tasks": {
            "action_router": True,
            "ours_action_router_mode": "learned_budget_overlay_v1",
            "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
            "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
            "ours_learned_router_confidence_threshold": threshold,
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "v293_rules_after_learned",
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    all_summary = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps(all_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
