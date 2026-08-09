#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
BUDGETS = [256, 384, 512, 768, 1024, 1536, 2048, 3072]
REFERENCE_DIR = ROOT / "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
FULL_DIR = ROOT / "outputs/riskkv_fullkv_m100_same_samples_20260710"

PRESELECTION_NUMERIC_FEATURES = [
    "raw_prefix_tokens",
    "raw_prompt_tokens",
    "context_length_field",
    "page_count",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
]

BUDGET_NUMERIC_FEATURES = [
    "candidate_budget_tokens",
    "candidate_budget_log2",
    "candidate_budget_fraction_of_context",
    "candidate_budget_fraction_of_prompt",
    "candidate_budget_pages",
    "candidate_budget_page_fraction",
    "candidate_budget_rank",
    "candidate_budget_rank_fraction",
]

FAMILY_BY_TASK = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "fewshot",
    "triviaqa": "fewshot",
    "samsum": "fewshot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}


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


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task = row.get("task", "")
        sample_id = row.get("sample_id", "")
        if task and sample_id:
            out[(task, sample_id)] = row
    return out


def fold_for_key(task: str, sample_id: str, folds: int = 5) -> int:
    digest = hashlib.md5(f"{task}\t{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, folds)


def task_family(task: str) -> str:
    return FAMILY_BY_TASK.get(task, "other")


def quality_target(reference_score: float, ratio: float, margin: float) -> float:
    return max(0.0, ratio * reference_score, reference_score - margin)


def candidate_output_dir(budget: int) -> Path:
    return ROOT / "outputs" / f"riskkv_v19_budget_sweep_b{budget}_20260711_budget_sweep_m100_m100_bDyn_pDyn"


def require_inputs() -> None:
    paths = [REFERENCE_DIR / "task_results.csv", FULL_DIR / "task_results.csv"]
    paths.extend(candidate_output_dir(budget) / "task_results.csv" for budget in BUDGETS)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required M100 sweep files:\n" + "\n".join(str(path) for path in missing))


def feature_value(row: dict[str, str], key: str) -> float:
    if key == "raw_prefix_tokens":
        return fnum(row, key)
    return fnum(row, key)


def build_base_record(
    key: tuple[str, str],
    ref_row: dict[str, str],
    quality_ratio: float,
    quality_margin: float,
) -> dict[str, Any]:
    task, sample_id = key
    reference_score = fnum(ref_row, "score")
    record: dict[str, Any] = {
        "task": task,
        "task_family": task_family(task),
        "sample_id": sample_id,
        "fold": fold_for_key(task, sample_id),
        "reference_score": reference_score,
        "reference_kv_keep": fnum(ref_row, "keep_fraction"),
        "reference_online_seconds": fnum(ref_row, "online_seconds"),
        "quality_target": quality_target(reference_score, quality_ratio, quality_margin),
    }
    for feature in PRESELECTION_NUMERIC_FEATURES:
        record[feature] = feature_value(ref_row, feature)
    return record


def budget_feature_values(base: dict[str, Any], budget: int, rank: int, count: int) -> dict[str, float]:
    raw_prefix = max(1.0, float(base.get("raw_prefix_tokens", 0.0) or 0.0))
    raw_prompt = max(1.0, float(base.get("raw_prompt_tokens", 0.0) or 0.0))
    page_count = max(1.0, float(base.get("page_count", 0.0) or 0.0))
    page_tokens = 128.0
    budget_value = float(max(0, budget))
    return {
        "candidate_budget_tokens": budget_value,
        "candidate_budget_log2": math.log2(max(1.0, budget_value)),
        "candidate_budget_fraction_of_context": budget_value / raw_prefix,
        "candidate_budget_fraction_of_prompt": budget_value / raw_prompt,
        "candidate_budget_pages": budget_value / page_tokens,
        "candidate_budget_page_fraction": budget_value / max(1.0, page_count * page_tokens),
        "candidate_budget_rank": float(rank),
        "candidate_budget_rank_fraction": float(rank) / max(1.0, float(count - 1)),
    }


def build_feature_names(pair_rows: list[dict[str, Any]], task_encoding: str) -> list[str]:
    categories: set[str] = set()
    if task_encoding in {"family", "both"}:
        for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}):
            categories.add(f"family={family}")
    if task_encoding in {"task", "both"}:
        for row in pair_rows:
            categories.add(f"task={row['task']}")
    for action in sorted({str(row["action"]) for row in pair_rows}):
        categories.add(f"action={action}")
    return PRESELECTION_NUMERIC_FEATURES + BUDGET_NUMERIC_FEATURES + sorted(categories)


def make_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        task = str(row["task"])
        family = task_family(task)
        action = str(row["action"])
        vector: list[float] = []
        for name in feature_names:
            if name in row:
                vector.append(float(row.get(name, 0.0) or 0.0))
            elif name.startswith("family="):
                vector.append(1.0 if name == f"family={family}" else 0.0)
            elif name.startswith("task="):
                vector.append(1.0 if name == f"task={task}" else 0.0)
            elif name.startswith("action="):
                vector.append(1.0 if name == f"action={action}" else 0.0)
            else:
                vector.append(0.0)
        matrix.append(vector)
    return matrix


def safe_probability(model: Any, vector: list[float]) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([vector])
        classes = [str(item) for item in getattr(model, "classes_", [])]
        if len(probs) and len(probs[0]):
            if "1" in classes:
                return float(probs[0][classes.index("1")])
            return float(max(probs[0]))
    return 1.0 if int(model.predict([vector])[0]) == 1 else 0.0


def summarize_predictions(rows: list[dict[str, Any]], split: str, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["ALL"].append(row)
        grouped[str(row["task"])].append(row)
    out: list[dict[str, Any]] = []
    for task, subset in sorted(grouped.items(), key=lambda item: (item[0] != "ALL", item[0])):
        ref_score = mean([float(row["reference_score"]) for row in subset])
        learned_score = mean([float(row["learned_score"]) for row in subset])
        oracle_score = mean([float(row["oracle_score"]) for row in subset])
        ref_kv = mean([float(row["reference_kv_keep"]) for row in subset])
        learned_kv = mean([float(row["learned_kv_keep"]) for row in subset])
        oracle_kv = mean([float(row["oracle_kv_keep"]) for row in subset])
        ref_online = mean([float(row["reference_online_seconds"]) for row in subset])
        learned_online = mean([float(row["learned_online_seconds"]) for row in subset])
        oracle_online = mean([float(row["oracle_online_seconds"]) for row in subset])
        safe = [row for row in subset if float(row["learned_score"]) + 1e-12 >= float(row["quality_target"])]
        fallback = [row for row in subset if row["learned_action"] == "reference"]
        out.append(
            {
                "split": split,
                "task": task,
                "threshold": threshold,
                "samples": len(subset),
                "reference_score": ref_score,
                "learned_score": learned_score,
                "oracle_score": oracle_score,
                "learned_vs_reference": learned_score / ref_score if ref_score > 0 else "",
                "oracle_vs_reference": oracle_score / ref_score if ref_score > 0 else "",
                "reference_kv_keep": ref_kv,
                "learned_kv_keep": learned_kv,
                "oracle_kv_keep": oracle_kv,
                "kv_relative": learned_kv / ref_kv if ref_kv > 0 else "",
                "oracle_kv_relative": oracle_kv / ref_kv if ref_kv > 0 else "",
                "reference_online_seconds": ref_online,
                "learned_online_seconds": learned_online,
                "oracle_online_seconds": oracle_online,
                "learned_speed_vs_reference": ref_online / learned_online if learned_online > 0 else "",
                "oracle_speed_vs_reference": ref_online / oracle_online if oracle_online > 0 else "",
                "safe_rate": len(safe) / max(1, len(subset)),
                "fallback_rate": len(fallback) / max(1, len(subset)),
                "mean_safe_probability": mean([float(row["learned_safe_probability"]) for row in subset]),
            }
        )
    return out


def build_records(quality_ratio: float, quality_margin: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_rows = by_key(read_csv(REFERENCE_DIR / "task_results.csv"))
    candidate_tables = {
        f"budget_b{budget}": by_key(read_csv(candidate_output_dir(budget) / "task_results.csv"))
        for budget in BUDGETS
    }
    base_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    action_budget = {f"budget_b{budget}": budget for budget in BUDGETS}
    for key, ref_row in sorted(reference_rows.items()):
        if any(key not in table for table in candidate_tables.values()):
            continue
        base = build_base_record(key, ref_row, quality_ratio, quality_margin)
        available_pairs: list[dict[str, Any]] = []
        for rank, budget in enumerate(BUDGETS):
            action = f"budget_b{budget}"
            cand = candidate_tables[action][key]
            pair: dict[str, Any] = {
                **base,
                "action": action,
                "candidate_budget_tokens": float(budget),
                "candidate_score": fnum(cand, "score"),
                "candidate_kv_keep": fnum(cand, "keep_fraction"),
                "candidate_online_seconds": fnum(cand, "online_seconds"),
                "is_safe": int(fnum(cand, "score") + 1e-12 >= float(base["quality_target"])),
            }
            pair.update(budget_feature_values(base, budget, rank, len(BUDGETS)))
            available_pairs.append(pair)
            pair_rows.append(pair)
        safe_pairs = [row for row in available_pairs if int(row["is_safe"]) == 1]
        if safe_pairs:
            oracle = min(safe_pairs, key=lambda row: (float(row["candidate_kv_keep"]), -float(row["candidate_score"])))
            oracle_action = str(oracle["action"])
            oracle_score = float(oracle["candidate_score"])
            oracle_kv = float(oracle["candidate_kv_keep"])
            oracle_online = float(oracle["candidate_online_seconds"])
        else:
            oracle_action = "reference"
            oracle_score = float(base["reference_score"])
            oracle_kv = float(base["reference_kv_keep"])
            oracle_online = float(base["reference_online_seconds"])
        base.update(
            {
                "oracle_action": oracle_action,
                "oracle_score": oracle_score,
                "oracle_kv_keep": oracle_kv,
                "oracle_online_seconds": oracle_online,
                "candidate_actions": ",".join(sorted(action_budget, key=action_budget.get)),
            }
        )
        base_rows.append(base)
    return base_rows, pair_rows


def select_predictions(
    base_rows: list[dict[str, Any]],
    pair_by_key: dict[tuple[str, str, str], dict[str, Any]],
    model: Any,
    feature_names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for base in base_rows:
        task = str(base["task"])
        sample_id = str(base["sample_id"])
        selected: dict[str, Any] | None = None
        selected_probability = 0.0
        best_probability = 0.0
        best_action = "reference"
        for budget in BUDGETS:
            action = f"budget_b{budget}"
            pair = pair_by_key[(task, sample_id, action)]
            probability = safe_probability(model, make_matrix([pair], feature_names)[0])
            if probability > best_probability:
                best_probability = probability
                best_action = action
            if probability >= threshold and selected is None:
                selected = pair
                selected_probability = probability
                break
        if selected is None:
            learned_action = "reference"
            learned_score = float(base["reference_score"])
            learned_kv = float(base["reference_kv_keep"])
            learned_online = float(base["reference_online_seconds"])
            selected_probability = best_probability
            fallback_reason = f"no_safe_candidate:{best_action}"
        else:
            learned_action = str(selected["action"])
            learned_score = float(selected["candidate_score"])
            learned_kv = float(selected["candidate_kv_keep"])
            learned_online = float(selected["candidate_online_seconds"])
            fallback_reason = ""
        out.append(
            {
                "task": task,
                "task_family": base["task_family"],
                "sample_id": sample_id,
                "fold": base["fold"],
                "threshold": threshold,
                "oracle_action": base["oracle_action"],
                "learned_action": learned_action,
                "learned_safe_probability": selected_probability,
                "fallback_reason": fallback_reason,
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
            }
        )
    return out


def train_one(
    base_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    class_weight_mode: str,
    max_depth: int,
    min_samples_leaf: int,
    calibration_score_ratio: float,
    calibration_min_safe_rate: float,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    tag = (
        f"riskkv_v19_budget_planner_v12_m100_rf_"
        f"{class_weight_mode}_d{max_depth}_l{min_samples_leaf}_"
        f"score{str(calibration_score_ratio).replace('.', '')}_safe{str(calibration_min_safe_rate).replace('.', '')}_20260711"
    )
    output_dir = ROOT / "outputs" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = [row for row in pair_rows if int(row["fold"]) not in {0, 1}]
    cal_base = [row for row in base_rows if int(row["fold"]) == 1]
    test_base = [row for row in base_rows if int(row["fold"]) == 0]
    train_base = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    all_base = list(base_rows)
    feature_names = build_feature_names(pair_rows, "both")
    x_train = make_matrix(train_pairs, feature_names)
    y_train = [int(row["is_safe"]) for row in train_pairs]
    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=13,
        class_weight="balanced_subsample" if class_weight_mode == "balanced" else None,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pair_by_key = {
        (str(row["task"]), str(row["sample_id"]), str(row["action"])): row
        for row in pair_rows
    }
    threshold_rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        cal_predictions = select_predictions(cal_base, pair_by_key, model, feature_names, threshold)
        summary = summarize_predictions(cal_predictions, "calibration", threshold)
        row = next(item for item in summary if item["task"] == "ALL")
        row = {**row, "feasible": 0}
        if float(row["learned_vs_reference"]) >= calibration_score_ratio and float(row["safe_rate"]) >= calibration_min_safe_rate:
            row["feasible"] = 1
            feasible.append(row)
        threshold_rows.append(row)
    if feasible:
        selected_threshold_row = min(
            feasible,
            key=lambda row: (
                float(row["learned_kv_keep"]),
                -float(row["learned_vs_reference"]),
                -float(row["safe_rate"]),
            ),
        )
    else:
        selected_threshold_row = threshold_rows[-1]
    threshold = float(selected_threshold_row["threshold"])

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, rows in [
        ("train", train_base),
        ("calibration", cal_base),
        ("test", test_base),
        ("all", all_base),
    ]:
        predictions = select_predictions(rows, pair_by_key, model, feature_names, threshold)
        for row in predictions:
            row["split"] = split
        prediction_rows.extend(predictions)
        summary_rows.extend(summarize_predictions(predictions, split, threshold))

    action_policy = {
        "actions": {
            f"budget_b{budget}": {
                "budget_tokens": budget,
            }
            for budget in BUDGETS
        }
    }
    candidate_budget_tokens = {f"budget_b{budget}": budget for budget in BUDGETS}
    metadata = {
        "router_type": "budget_pair_planner_v12",
        "candidate_actions": [f"budget_b{budget}" for budget in BUDGETS],
        "candidate_budget_tokens": candidate_budget_tokens,
        "safe_probability_threshold": threshold,
        "calibration_score_ratio": calibration_score_ratio,
        "calibration_min_safe_rate": calibration_min_safe_rate,
        "class_weight_mode": class_weight_mode,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "feature_names": feature_names,
        "reference_dir": str(REFERENCE_DIR.relative_to(ROOT)),
        "full_dir": str(FULL_DIR.relative_to(ROOT)),
        "quality_ratio": 1.0,
        "quality_margin": 0.0,
        "folds": 5,
        "test_fold": 0,
        "calibration_fold": 1,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "planner_predictions.csv", prediction_rows)
    write_csv(output_dir / "planner_summary.csv", summary_rows)
    feature_rows = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    ]
    write_csv(output_dir / "feature_importance.csv", feature_rows)
    all_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    return {
        "router": tag,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "threshold": threshold,
        "test_score_ratio": all_summary["learned_vs_reference"],
        "test_learned_score": all_summary["learned_score"],
        "test_reference_score": all_summary["reference_score"],
        "test_kv_relative": all_summary["kv_relative"],
        "test_learned_kv": all_summary["learned_kv_keep"],
        "test_speed_vs_reference": all_summary["learned_speed_vs_reference"],
        "test_safe_rate": all_summary["safe_rate"],
        "test_fallback_rate": all_summary["fallback_rate"],
        "oracle_kv_relative": all_summary["oracle_kv_relative"],
        "oracle_score_ratio": all_summary["oracle_vs_reference"],
        "class_weight_mode": class_weight_mode,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "calibration_score_ratio": calibration_score_ratio,
        "calibration_min_safe_rate": calibration_min_safe_rate,
    }


def main() -> None:
    require_inputs()
    base_rows, pair_rows = build_records(quality_ratio=1.0, quality_margin=0.0)
    if not base_rows:
        raise ValueError("No joined M100 rows for v12 planner training.")
    comparison_rows: list[dict[str, Any]] = []
    for class_weight_mode in ["balanced", "none"]:
        for max_depth, min_leaf in [(4, 12), (5, 8), (6, 6), (7, 4)]:
            for calibration_score_ratio in [1.0, 0.9975, 0.995]:
                row = train_one(
                    base_rows,
                    pair_rows,
                    class_weight_mode=class_weight_mode,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    calibration_score_ratio=calibration_score_ratio,
                    calibration_min_safe_rate=0.0,
                )
                comparison_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    comparison_rows.sort(
        key=lambda row: (
            float(row["test_score_ratio"]),
            -float(row["test_kv_relative"]),
            float(row["test_speed_vs_reference"]),
        ),
        reverse=True,
    )
    out_path = ROOT / "outputs/riskkv_v19_budget_planner_v12_m100_compare_summary_20260711.csv"
    write_csv(out_path, comparison_rows)
    print(out_path)
    print(json.dumps(comparison_rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
