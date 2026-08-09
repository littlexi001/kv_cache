#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import train_policy_action_planner_v378_20260712 as v378
import train_policy_multiclass_router_v380_20260712 as v380


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v378.ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--base-action", default="policy_v377")
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--quality-margin", type=float, default=0.05)
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--full-online", type=float, default=3.0988)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_policy_multiclass_router_v382_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v382_policy_multiclass_base_v377_20260712.json")
    args = parser.parse_args()

    root = Path(args.root)
    base_rows, pair_by_key, candidate_specs = v380.build_training_rows(
        root,
        args.full_results,
        args.base_action,
        args.quality_ratio,
        args.quality_margin,
    )
    if args.base_action not in candidate_specs:
        raise FileNotFoundError(f"Missing base action config: {args.base_action}")

    feature_names = v380.make_feature_names(base_rows)
    train_rows = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    cal_rows = [row for row in base_rows if int(row["fold"]) == 1]
    test_rows = [row for row in base_rows if int(row["fold"]) == 0]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=29,
        n_jobs=-1,
    )
    model.fit(v380.vectorize(train_rows, feature_names), [str(row["oracle_action"]) for row in train_rows])

    threshold_rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        pred = v380.predict_rows(cal_rows, pair_by_key, model, feature_names, threshold)
        row = next(item for item in v380.summarize(pred, "calibration", args.full_online) if item["task"] == "ALL")
        learned_vs_full = float(row["learned_vs_full"] or 0.0)
        learned_kv = float(row["learned_kv_keep"])
        learned_speed = float(row["learned_speed_vs_full"] or 0.0)
        row = {
            **row,
            "threshold": threshold,
            "feasible": int(
                learned_vs_full >= args.quality_ratio
                and learned_kv <= args.kv_limit
                and learned_speed >= args.speed_min
            ),
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

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, rows in [("train", train_rows), ("calibration", cal_rows), ("test", test_rows), ("all", base_rows)]:
        pred = v380.predict_rows(rows, pair_by_key, model, feature_names, threshold)
        for row in pred:
            row["split"] = split
            row["threshold"] = threshold
        prediction_rows.extend(pred)
        for row in v380.summarize(pred, split, args.full_online):
            row["threshold"] = threshold
            summary_rows.append(row)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = v378.build_action_policy(root, candidate_specs)
    base_config = Path(candidate_specs[args.base_action]["config"]).name
    metadata = {
        "router_type": "policy_multiclass_router_v382_base_v377_nopost",
        "feature_names": feature_names,
        "confidence_fallback_threshold": threshold,
        "quality_ratio": args.quality_ratio,
        "quality_margin": args.quality_margin,
        "kv_limit": args.kv_limit,
        "speed_min": args.speed_min,
        "base_action": args.base_action,
        "base_config": base_config,
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
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
    )

    config = {
        "__extends": base_config,
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
    test_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps({"all": all_summary, "test": test_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
