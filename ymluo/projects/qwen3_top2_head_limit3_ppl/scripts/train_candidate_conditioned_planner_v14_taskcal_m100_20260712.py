#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_candidate_conditioned_planner_v13_m100_20260712 import (
    ROOT,
    OUT_ROOT,
    TASKS,
    build_dataset,
    feature_names,
    fit_model,
    make_pair_tables,
    summarize,
)


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


def choose_action(pairs: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    safe = [row for row in pairs if float(row["safe_probability"]) >= threshold]
    if not safe:
        return None
    return min(safe, key=lambda item: (float(item["candidate_keep_fraction"]), float(item["candidate_budget_tokens"])))


def predict_with_task_thresholds(
    base_rows: list[dict[str, Any]],
    pair_table: dict[str, list[dict[str, Any]]],
    thresholds_by_task: dict[str, float],
    tag: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for base in base_rows:
        task = str(base["task"])
        threshold = thresholds_by_task.get(task, 1.01)
        pairs = pair_table[str(base["key"])]
        selected = choose_action(pairs, threshold)
        if selected is None:
            action = "reference"
            learned_score = float(base["reference_score"])
            learned_kv = float(base["reference_kv_keep"])
            learned_online = float(base["reference_online_seconds"])
            prob = max(float(item["safe_probability"]) for item in pairs)
        else:
            action = str(selected["action"])
            learned_score = float(selected["candidate_score"])
            learned_kv = float(selected["candidate_keep_fraction"])
            learned_online = float(selected["candidate_online_seconds"])
            prob = float(selected["safe_probability"])
        predictions.append(
            {
                "task": task,
                "task_family": base["task_family"],
                "sample_id": base["sample_id"],
                "fold": base["fold"],
                "oracle_action": base["oracle_action"],
                "learned_action": action,
                "learned_safe_probability": prob,
                "reference_score": base["reference_score"],
                "learned_score": learned_score,
                "oracle_score": base["oracle_score"],
                "quality_target": base["quality_target"],
                "reference_kv_keep": base["reference_kv_keep"],
                "learned_kv_keep": learned_kv,
                "oracle_kv_keep": base["oracle_kv_keep"],
                "reference_online_seconds": base["reference_online_seconds"],
                "learned_online_seconds": learned_online,
                "oracle_online_seconds": base["oracle_online_seconds"],
                "threshold_tag": tag,
            }
        )
    return predictions, summarize(predictions, "tmp", tag)


def select_task_thresholds(
    cal_rows: list[dict[str, Any]],
    pair_table: dict[str, list[dict[str, Any]]],
    min_score_ratio: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cal_rows:
        grouped[str(row["task"])].append(row)
    thresholds: dict[str, float] = {}
    selected_rows: list[dict[str, Any]] = []
    for task, rows in grouped.items():
        candidates: list[dict[str, Any]] = []
        for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
            _preds, summaries = predict_with_task_thresholds(rows, pair_table, {task: threshold}, f"{task}_{threshold}")
            summary = next(row for row in summaries if row["task"] == "ALL")
            candidates.append({**summary, "task_calibrated": task, "threshold": threshold})
        feasible = [row for row in candidates if float(row["learned_vs_reference"]) >= min_score_ratio]
        selected = min(
            feasible or candidates,
            key=lambda row: (
                float(row["learned_kv_keep"]),
                -float(row["learned_vs_reference"]),
                -float(row["safe_rate"]),
            ),
        )
        thresholds[task] = float(selected["threshold"])
        selected_rows.append(selected)
    return thresholds, selected_rows


def run_one(
    quality_ratio: float,
    model_kind: str,
    class_weight: str,
    max_depth: int,
    min_leaf: int,
    min_score_ratio: float,
) -> dict[str, Any]:
    base_rows, pair_rows = build_dataset(quality_ratio)
    names = feature_names(pair_rows)
    train_pairs = [row for row in pair_rows if int(row["fold"]) not in {0, 1}]
    train_base = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    cal_base = [row for row in base_rows if int(row["fold"]) == 1]
    test_base = [row for row in base_rows if int(row["fold"]) == 0]
    model = fit_model(model_kind, class_weight, max_depth, min_leaf, train_pairs, names)
    pair_table = make_pair_tables(pair_rows, model, names)
    thresholds, task_cal_rows = select_task_thresholds(cal_base, pair_table, min_score_ratio)
    tag = (
        f"v14_taskcal_q{str(quality_ratio).replace('.', '')}_{model_kind}_{class_weight}_"
        f"d{max_depth}_l{min_leaf}_cal{str(min_score_ratio).replace('.', '')}"
    )
    out_dir = OUT_ROOT / f"riskkv_v19_candidate_conditioned_planner_{tag}_20260712"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, rows in [
        ("train", train_base),
        ("calibration", cal_base),
        ("test", test_base),
        ("all", base_rows),
    ]:
        preds, summaries = predict_with_task_thresholds(rows, pair_table, thresholds, tag)
        for row in preds:
            row["split"] = split
        for row in summaries:
            row["split"] = split
        prediction_rows.extend(preds)
        summary_rows.extend(summaries)
    write_csv(out_dir / "planner_predictions.csv", prediction_rows)
    write_csv(out_dir / "planner_summary.csv", summary_rows)
    write_csv(out_dir / "task_calibration_thresholds.csv", task_cal_rows)
    metadata = {
        "router_type": "candidate_conditioned_planner_v14_taskcal",
        "quality_ratio": quality_ratio,
        "model_kind": model_kind,
        "class_weight": class_weight,
        "max_depth": max_depth if max_depth > 0 else "none",
        "min_leaf": min_leaf,
        "min_score_ratio": min_score_ratio,
        "thresholds_by_task": thresholds,
        "feature_names": names,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    test_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    all_summary = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    return {
        "router": out_dir.name,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "quality_ratio": quality_ratio,
        "model_kind": model_kind,
        "class_weight": class_weight,
        "max_depth": max_depth if max_depth > 0 else "none",
        "min_leaf": min_leaf,
        "min_score_ratio": min_score_ratio,
        "thresholds_json": json.dumps(thresholds, sort_keys=True),
        "test_score_ratio": test_summary["learned_vs_reference"],
        "test_kv_relative": test_summary["kv_relative"],
        "test_speed_vs_reference": test_summary["learned_speed_vs_reference"],
        "test_safe_rate": test_summary["safe_rate"],
        "test_fallback_rate": test_summary["fallback_rate"],
        "all_score_ratio": all_summary["learned_vs_reference"],
        "all_kv_relative": all_summary["kv_relative"],
        "all_speed_vs_reference": all_summary["learned_speed_vs_reference"],
        "all_safe_rate": all_summary["safe_rate"],
        "all_fallback_rate": all_summary["fallback_rate"],
        "oracle_score_ratio": test_summary["oracle_vs_reference"],
        "oracle_kv_relative": test_summary["oracle_kv_relative"],
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for quality_ratio in [1.0, 0.99, 0.95]:
        for model_kind in ["rf", "extra"]:
            for class_weight in ["none"]:
                for max_depth, min_leaf in [(8, 4), (0, 4)]:
                    for min_score_ratio in [1.0, 0.995, 0.99, 0.95]:
                        row = run_one(quality_ratio, model_kind, class_weight, max_depth, min_leaf, min_score_ratio)
                        rows.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = OUT_ROOT / "riskkv_v19_candidate_conditioned_planner_v14_taskcal_compare_summary_20260712.csv"
    write_csv(out, rows)
    rows.sort(
        key=lambda row: (
            float(row["all_score_ratio"]),
            -float(row["all_kv_relative"]),
            float(row["test_score_ratio"]),
        ),
        reverse=True,
    )
    (OUT_ROOT / "riskkv_v19_candidate_conditioned_planner_v14_taskcal_top20_20260712.json").write_text(
        json.dumps(rows[:20], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(out)
    print(json.dumps(rows[:20], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
