#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
OUT_ROOT = ROOT / "outputs"
CONFIG_ROOT = ROOT / "configs"
REFERENCE_DIR = OUT_ROOT / "riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
BUDGETS = [256, 384, 512, 768, 1024, 1536, 2048, 3072]

BASE_FEATURES = [
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

BUDGET_FEATURES = [
    "candidate_budget_tokens",
    "candidate_budget_log2",
    "candidate_budget_fraction_of_context",
    "candidate_budget_fraction_of_prompt",
    "candidate_budget_pages",
    "candidate_budget_page_fraction",
    "candidate_budget_rank",
    "candidate_budget_rank_fraction",
]

TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
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


def fnum(row: dict[str, str] | dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("task", ""), row.get("sample_id", ""))
        if key[0] and key[1]:
            out[key] = row
    return out


def fold_for_key(task: str, sample_id: str) -> int:
    digest = hashlib.md5(f"{task}\t{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 5


def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def budget_dir(budget: int) -> Path:
    return OUT_ROOT / f"riskkv_v19_budget_sweep_b{budget}_20260711_budget_sweep_m100_m100_bDyn_pDyn"


def build_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    reference = by_key(read_csv(REFERENCE_DIR / "task_results.csv"))
    candidates = {
        f"budget_b{budget}": by_key(read_csv(budget_dir(budget) / "task_results.csv"))
        for budget in BUDGETS
    }
    base_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    actions_by_key: dict[str, list[str]] = {}
    for key, ref in sorted(reference.items()):
        if any(key not in table for table in candidates.values()):
            continue
        task, sample_id = key
        base: dict[str, Any] = {
            "task": task,
            "task_family": FAMILY_BY_TASK.get(task, "other"),
            "sample_id": sample_id,
            "key": f"{task}\t{sample_id}",
            "fold": fold_for_key(task, sample_id),
            "reference_score": fnum(ref, "score"),
            "reference_kv_keep": fnum(ref, "keep_fraction"),
            "reference_online_seconds": fnum(ref, "online_seconds"),
            "quality_target": fnum(ref, "score"),
        }
        for feature in BASE_FEATURES:
            base[feature] = fnum(ref, feature)
        safe_pairs: list[dict[str, Any]] = []
        actions: list[str] = []
        for rank, budget in enumerate(BUDGETS):
            action = f"budget_b{budget}"
            row = candidates[action][key]
            raw_prefix = max(1.0, float(base["raw_prefix_tokens"]))
            raw_prompt = max(1.0, float(base["raw_prompt_tokens"]))
            page_count = max(1.0, float(base["page_count"]))
            page_tokens = 128.0
            pair = {
                **base,
                "action": action,
                "candidate_budget_tokens": float(budget),
                "candidate_budget_log2": math.log2(float(budget)),
                "candidate_budget_fraction_of_context": float(budget) / raw_prefix,
                "candidate_budget_fraction_of_prompt": float(budget) / raw_prompt,
                "candidate_budget_pages": float(budget) / page_tokens,
                "candidate_budget_page_fraction": float(budget) / max(1.0, page_count * page_tokens),
                "candidate_budget_rank": float(rank),
                "candidate_budget_rank_fraction": float(rank) / float(len(BUDGETS) - 1),
                "candidate_score": fnum(row, "score"),
                "candidate_kv_keep": fnum(row, "keep_fraction"),
                "candidate_online_seconds": fnum(row, "online_seconds"),
            }
            pair["is_safe"] = int(pair["candidate_score"] + 1e-12 >= base["quality_target"])
            if pair["is_safe"]:
                safe_pairs.append(pair)
            pair_rows.append(pair)
            actions.append(action)
        if safe_pairs:
            oracle = min(safe_pairs, key=lambda item: (item["candidate_kv_keep"], -item["candidate_score"]))
            base["oracle_action"] = oracle["action"]
            base["oracle_score"] = oracle["candidate_score"]
            base["oracle_kv_keep"] = oracle["candidate_kv_keep"]
            base["oracle_online_seconds"] = oracle["candidate_online_seconds"]
        else:
            base["oracle_action"] = "reference"
            base["oracle_score"] = base["reference_score"]
            base["oracle_kv_keep"] = base["reference_kv_keep"]
            base["oracle_online_seconds"] = base["reference_online_seconds"]
        actions_by_key[base["key"]] = actions
        base_rows.append(base)
    return base_rows, pair_rows, actions_by_key


def feature_names(pair_rows: list[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}):
        categories.append(f"family={family}")
    for task in sorted({str(row["task"]) for row in pair_rows}):
        categories.append(f"task={task}")
    for action in sorted({str(row["action"]) for row in pair_rows}):
        categories.append(f"action={action}")
    return BASE_FEATURES + BUDGET_FEATURES + categories


def make_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        task = str(row["task"])
        family = FAMILY_BY_TASK.get(task, "other")
        action = str(row.get("action", ""))
        for j, name in enumerate(names):
            if name in row:
                matrix[i, j] = float(row.get(name, 0.0) or 0.0)
            elif name == f"family={family}" or name == f"task={task}" or name == f"action={action}":
                matrix[i, j] = 1.0
    return matrix


def safe_probabilities(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = [str(item) for item in model.classes_]
    if "1" in classes:
        return probs[:, classes.index("1")]
    return probs[:, -1]


def evaluate_threshold(
    base_rows: list[dict[str, Any]],
    pair_rows_by_key: dict[str, list[dict[str, Any]]],
    probabilities_by_key: dict[str, np.ndarray],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for base in base_rows:
        key = str(base["key"])
        pairs = pair_rows_by_key[key]
        probs = probabilities_by_key[key]
        selected_idx = next((idx for idx, prob in enumerate(probs) if prob >= threshold), None)
        if selected_idx is None:
            action = "reference"
            learned_score = float(base["reference_score"])
            learned_kv = float(base["reference_kv_keep"])
            learned_online = float(base["reference_online_seconds"])
            safe_probability = float(np.max(probs)) if len(probs) else 0.0
        else:
            pair = pairs[selected_idx]
            action = str(pair["action"])
            learned_score = float(pair["candidate_score"])
            learned_kv = float(pair["candidate_kv_keep"])
            learned_online = float(pair["candidate_online_seconds"])
            safe_probability = float(probs[selected_idx])
        predictions.append(
            {
                "task": base["task"],
                "task_family": base["task_family"],
                "sample_id": base["sample_id"],
                "fold": base["fold"],
                "threshold": threshold,
                "oracle_action": base["oracle_action"],
                "learned_action": action,
                "learned_safe_probability": safe_probability,
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
    ref_score = np.array([float(row["reference_score"]) for row in predictions])
    learned_score = np.array([float(row["learned_score"]) for row in predictions])
    oracle_score = np.array([float(row["oracle_score"]) for row in predictions])
    ref_kv = np.array([float(row["reference_kv_keep"]) for row in predictions])
    learned_kv = np.array([float(row["learned_kv_keep"]) for row in predictions])
    oracle_kv = np.array([float(row["oracle_kv_keep"]) for row in predictions])
    ref_online = np.array([float(row["reference_online_seconds"]) for row in predictions])
    learned_online = np.array([float(row["learned_online_seconds"]) for row in predictions])
    oracle_online = np.array([float(row["oracle_online_seconds"]) for row in predictions])
    safe = np.array([
        float(row["learned_score"]) + 1e-12 >= float(row["quality_target"])
        for row in predictions
    ])
    fallback = np.array([row["learned_action"] == "reference" for row in predictions])
    summary = {
        "samples": len(predictions),
        "reference_score": mean(ref_score),
        "learned_score": mean(learned_score),
        "oracle_score": mean(oracle_score),
        "learned_vs_reference": mean(learned_score) / mean(ref_score) if mean(ref_score) > 0 else "",
        "oracle_vs_reference": mean(oracle_score) / mean(ref_score) if mean(ref_score) > 0 else "",
        "reference_kv_keep": mean(ref_kv),
        "learned_kv_keep": mean(learned_kv),
        "oracle_kv_keep": mean(oracle_kv),
        "kv_relative": mean(learned_kv) / mean(ref_kv) if mean(ref_kv) > 0 else "",
        "oracle_kv_relative": mean(oracle_kv) / mean(ref_kv) if mean(ref_kv) > 0 else "",
        "reference_online_seconds": mean(ref_online),
        "learned_online_seconds": mean(learned_online),
        "oracle_online_seconds": mean(oracle_online),
        "learned_speed_vs_reference": mean(ref_online) / mean(learned_online) if mean(learned_online) > 0 else "",
        "oracle_speed_vs_reference": mean(ref_online) / mean(oracle_online) if mean(oracle_online) > 0 else "",
        "safe_rate": float(np.mean(safe)) if len(safe) else 0.0,
        "fallback_rate": float(np.mean(fallback)) if len(fallback) else 0.0,
    }
    return predictions, summary


def grouped_summary(predictions: list[dict[str, Any]], split: str, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ["ALL", *TASKS]:
        subset = predictions if task == "ALL" else [row for row in predictions if row["task"] == task]
        if not subset:
            continue
        _, summary = evaluate_threshold([], {}, {}, threshold) if False else ([], {})
        ref_score = np.array([float(row["reference_score"]) for row in subset])
        learned_score = np.array([float(row["learned_score"]) for row in subset])
        oracle_score = np.array([float(row["oracle_score"]) for row in subset])
        ref_kv = np.array([float(row["reference_kv_keep"]) for row in subset])
        learned_kv = np.array([float(row["learned_kv_keep"]) for row in subset])
        oracle_kv = np.array([float(row["oracle_kv_keep"]) for row in subset])
        ref_online = np.array([float(row["reference_online_seconds"]) for row in subset])
        learned_online = np.array([float(row["learned_online_seconds"]) for row in subset])
        oracle_online = np.array([float(row["oracle_online_seconds"]) for row in subset])
        safe = np.array([
            float(row["learned_score"]) + 1e-12 >= float(row["quality_target"])
            for row in subset
        ])
        fallback = np.array([row["learned_action"] == "reference" for row in subset])
        out.append(
            {
                "split": split,
                "task": task,
                "threshold": threshold,
                "samples": len(subset),
                "reference_score": mean(ref_score),
                "learned_score": mean(learned_score),
                "oracle_score": mean(oracle_score),
                "learned_vs_reference": mean(learned_score) / mean(ref_score) if mean(ref_score) > 0 else "",
                "oracle_vs_reference": mean(oracle_score) / mean(ref_score) if mean(ref_score) > 0 else "",
                "reference_kv_keep": mean(ref_kv),
                "learned_kv_keep": mean(learned_kv),
                "oracle_kv_keep": mean(oracle_kv),
                "kv_relative": mean(learned_kv) / mean(ref_kv) if mean(ref_kv) > 0 else "",
                "oracle_kv_relative": mean(oracle_kv) / mean(ref_kv) if mean(ref_kv) > 0 else "",
                "reference_online_seconds": mean(ref_online),
                "learned_online_seconds": mean(learned_online),
                "oracle_online_seconds": mean(oracle_online),
                "learned_speed_vs_reference": mean(ref_online) / mean(learned_online) if mean(learned_online) > 0 else "",
                "oracle_speed_vs_reference": mean(ref_online) / mean(oracle_online) if mean(oracle_online) > 0 else "",
                "safe_rate": float(np.mean(safe)),
                "fallback_rate": float(np.mean(fallback)),
            }
        )
    return out


def run_one(
    base_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    names: list[str],
    max_depth: int,
    min_leaf: int,
    class_weight: str,
    calibration_score_ratio: float,
) -> dict[str, Any]:
    tag = (
        f"riskkv_v19_budget_planner_v12_fast_m100_rf_{class_weight}_"
        f"d{max_depth}_l{min_leaf}_score{str(calibration_score_ratio).replace('.', '')}_20260712"
    )
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        pair_rows_by_key.setdefault(str(row["key"]), []).append(row)
    for rows in pair_rows_by_key.values():
        rows.sort(key=lambda item: float(item["candidate_budget_tokens"]))

    train_pairs = [row for row in pair_rows if int(row["fold"]) not in {0, 1}]
    x_train = make_matrix(train_pairs, names)
    y_train = np.array([int(row["is_safe"]) for row in train_pairs])
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=13,
        class_weight="balanced_subsample" if class_weight == "balanced" else None,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    all_pair_matrix = make_matrix(pair_rows, names)
    all_probs = safe_probabilities(model, all_pair_matrix)
    probabilities_by_key: dict[str, np.ndarray] = {}
    offset = 0
    for key, rows in pair_rows_by_key.items():
        count = len(rows)
        probabilities_by_key[key] = all_probs[offset : offset + count]
        offset += count

    cal_base = [row for row in base_rows if int(row["fold"]) == 1]
    test_base = [row for row in base_rows if int(row["fold"]) == 0]
    train_base = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    thresholds = [round(i / 100, 2) for i in range(0, 101)] + [1.01]
    threshold_rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in thresholds:
        _pred, summary = evaluate_threshold(cal_base, pair_rows_by_key, probabilities_by_key, threshold)
        row = {"threshold": threshold, **summary}
        row["feasible"] = int(float(row["learned_vs_reference"]) >= calibration_score_ratio)
        threshold_rows.append(row)
        if row["feasible"]:
            feasible.append(row)
    selected = min(
        feasible or threshold_rows,
        key=lambda row: (
            float(row["learned_kv_keep"]),
            -float(row["learned_vs_reference"]),
            -float(row["safe_rate"]),
        ),
    )
    threshold = float(selected["threshold"])
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split, rows in [
        ("train", train_base),
        ("calibration", cal_base),
        ("test", test_base),
        ("all", base_rows),
    ]:
        predictions, _summary = evaluate_threshold(rows, pair_rows_by_key, probabilities_by_key, threshold)
        for row in predictions:
            row["split"] = split
        prediction_rows.extend(predictions)
        summary_rows.extend(grouped_summary(predictions, split, threshold))

    metadata = {
        "router_type": "budget_pair_planner_v12",
        "candidate_actions": [f"budget_b{budget}" for budget in BUDGETS],
        "candidate_budget_tokens": {f"budget_b{budget}": budget for budget in BUDGETS},
        "safe_probability_threshold": threshold,
        "feature_names": names,
        "class_weight_mode": class_weight,
        "max_depth": max_depth,
        "min_samples_leaf": min_leaf,
        "calibration_score_ratio": calibration_score_ratio,
        "folds": 5,
        "test_fold": 0,
        "calibration_fold": 1,
    }
    with (out_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "action_policy.json").write_text(
        json.dumps({"actions": {f"budget_b{budget}": {"budget_tokens": budget} for budget in BUDGETS}}, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(out_dir / "planner_predictions.csv", prediction_rows)
    write_csv(out_dir / "planner_summary.csv", summary_rows)
    test_summary = next(row for row in summary_rows if row["split"] == "test" and row["task"] == "ALL")
    cal_summary = next(row for row in summary_rows if row["split"] == "calibration" and row["task"] == "ALL")
    all_summary = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    return {
        "router": tag,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "threshold": threshold,
        "test_score_ratio": test_summary["learned_vs_reference"],
        "test_kv_relative": test_summary["kv_relative"],
        "test_speed_vs_reference": test_summary["learned_speed_vs_reference"],
        "test_safe_rate": test_summary["safe_rate"],
        "test_fallback_rate": test_summary["fallback_rate"],
        "calibration_score_ratio_measured": cal_summary["learned_vs_reference"],
        "calibration_kv_relative_measured": cal_summary["kv_relative"],
        "all_score_ratio_measured": all_summary["learned_vs_reference"],
        "all_kv_relative_measured": all_summary["kv_relative"],
        "oracle_score_ratio": test_summary["oracle_vs_reference"],
        "oracle_kv_relative": test_summary["oracle_kv_relative"],
        "class_weight": class_weight,
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "calibration_score_ratio": calibration_score_ratio,
    }


def write_selected_policy(best: dict[str, Any]) -> Path:
    policy = {
        "__extends": "riskkv_task_policy_v300_action_router_extra50_robust_20260711.json",
        "tasks": {
            "*": {
                "ours_learned_router_model_path": f"{best['output_dir']}/model.pkl",
                "ours_learned_router_action_policy_json": f"{best['output_dir']}/action_policy.json",
                "ours_learned_router_confidence_threshold": -1,
                "ours_learned_router_default_action": "reference",
                "ours_learned_router_base_action_router_mode": "v293_rules",
            }
        },
    }
    for task in TASKS:
        policy["tasks"][task] = {
            "action_router": True,
            "action_router_mode": "learned_budget_planner_v2",
        }
    path = CONFIG_ROOT / "riskkv_task_policy_v352_budget_planner_v12_fast_m100_20260712.json"
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    base_rows, pair_rows, _actions_by_key = build_dataset()
    names = feature_names(pair_rows)
    rows: list[dict[str, Any]] = []
    for class_weight in ["none", "balanced"]:
        for max_depth, min_leaf in [(4, 12), (5, 8), (6, 6), (7, 4)]:
            for score_ratio in [1.0, 0.9975, 0.995, 0.99]:
                row = run_one(base_rows, pair_rows, names, max_depth, min_leaf, class_weight, score_ratio)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(
        key=lambda row: (
            float(row["calibration_score_ratio_measured"]) >= 1.0,
            -float(row["calibration_kv_relative_measured"]),
            float(row["test_score_ratio"]),
            -float(row["test_kv_relative"]),
        ),
        reverse=True,
    )
    write_csv(OUT_ROOT / "riskkv_v19_budget_planner_v12_fast_m100_compare_summary_20260712.csv", rows)
    feasible = [row for row in rows if float(row["calibration_score_ratio_measured"]) >= 0.9975]
    best = min(
        feasible or rows,
        key=lambda row: (
            float(row["calibration_kv_relative_measured"]),
            -float(row["calibration_score_ratio_measured"]),
            -float(row["test_score_ratio"]),
        ),
    )
    selected_path = OUT_ROOT / "riskkv_v19_budget_planner_v12_fast_selected_20260712.json"
    selected_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    policy_path = write_selected_policy(best)
    print(json.dumps({"selected": best, "policy": str(policy_path.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
