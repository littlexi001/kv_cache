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

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
OUT_ROOT = ROOT / "outputs"
REFERENCE_DIR = OUT_ROOT / "riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
BUDGETS = [256, 384, 512, 768, 1024, 1536, 2048, 3072]

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

PAIR_NUMERIC_FEATURES = [
    *BASE_FEATURES,
    "candidate_budget_tokens",
    "candidate_budget_log2",
    "candidate_budget_rank",
    "candidate_budget_rank_fraction",
    "candidate_effective_budget_tokens",
    "candidate_page_tokens",
    "candidate_keep_fraction",
    "candidate_kv_relative_to_reference",
    "candidate_kv_saving_vs_reference",
    "candidate_kept_context_tokens",
    "candidate_kept_context_fraction",
    "candidate_kept_context_relative_to_reference",
    "candidate_query_terms",
    "candidate_query_covered",
    "candidate_query_recall",
    "candidate_query_missing",
    "candidate_query_recall_per_kv",
    "candidate_query_covered_per_kv",
    "candidate_selected_page_count",
    "candidate_selected_page_fraction",
    "candidate_selected_span_pages",
    "candidate_selected_span_fraction",
    "candidate_selected_density",
    "candidate_selected_mean_gap",
    "candidate_selected_max_gap",
    "candidate_selected_run_count",
    "candidate_selected_run_fraction",
    "candidate_selected_edge_fraction",
    "candidate_selected_recent_fraction",
    "candidate_score_max",
    "candidate_score_mean",
    "candidate_score_gap2",
    "candidate_score_gap3",
    "candidate_score_entropy",
    "candidate_score_positive_fraction",
    "delta_query_recall_vs_reference",
    "delta_selected_density_vs_reference",
    "delta_score_gap2_vs_reference",
    "delta_score_entropy_vs_reference",
]


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


def fnum(row: dict[str, str] | dict[str, Any] | None, key: str) -> float:
    if row is None:
        return 0.0
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


def mean(values: list[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def budget_dir(budget: int) -> Path:
    return OUT_ROOT / f"riskkv_v19_budget_sweep_b{budget}_20260711_budget_sweep_m100_m100_bDyn_pDyn"


def parse_pages(text: str) -> list[int]:
    pages: list[int] = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            pages.append(int(item))
        except ValueError:
            continue
    return sorted(set(pages))


def selected_page_features(row: dict[str, str], prefix: str) -> dict[str, float]:
    pages = parse_pages(row.get("selected_pages", ""))
    page_count = max(1.0, fnum(row, "page_count"))
    if not pages:
        return {
            f"{prefix}_selected_page_count": 0.0,
            f"{prefix}_selected_page_fraction": 0.0,
            f"{prefix}_selected_span_pages": 0.0,
            f"{prefix}_selected_span_fraction": 0.0,
            f"{prefix}_selected_density": 0.0,
            f"{prefix}_selected_mean_gap": 0.0,
            f"{prefix}_selected_max_gap": 0.0,
            f"{prefix}_selected_run_count": 0.0,
            f"{prefix}_selected_run_fraction": 0.0,
            f"{prefix}_selected_edge_fraction": 0.0,
            f"{prefix}_selected_recent_fraction": 0.0,
        }
    gaps = [pages[idx + 1] - pages[idx] for idx in range(len(pages) - 1)]
    run_count = 1 + sum(1 for gap in gaps if gap > 1)
    span_pages = pages[-1] - pages[0] + 1
    edge = sum(1 for page in pages if page <= 1 or page >= page_count - 2)
    recent_cut = max(0.0, page_count - 4.0)
    recent = sum(1 for page in pages if float(page) >= recent_cut)
    return {
        f"{prefix}_selected_page_count": float(len(pages)),
        f"{prefix}_selected_page_fraction": float(len(pages)) / page_count,
        f"{prefix}_selected_span_pages": float(span_pages),
        f"{prefix}_selected_span_fraction": float(span_pages) / page_count,
        f"{prefix}_selected_density": float(len(pages)) / max(1.0, float(span_pages)),
        f"{prefix}_selected_mean_gap": mean(gaps) if gaps else 0.0,
        f"{prefix}_selected_max_gap": float(max(gaps)) if gaps else 0.0,
        f"{prefix}_selected_run_count": float(run_count),
        f"{prefix}_selected_run_fraction": float(run_count) / max(1.0, float(len(pages))),
        f"{prefix}_selected_edge_fraction": float(edge) / max(1.0, float(len(pages))),
        f"{prefix}_selected_recent_fraction": float(recent) / max(1.0, float(len(pages))),
    }


def base_record(key: tuple[str, str], row: dict[str, str]) -> dict[str, Any]:
    task, sample_id = key
    record: dict[str, Any] = {
        "task": task,
        "task_family": FAMILY_BY_TASK.get(task, "other"),
        "sample_id": sample_id,
        "key": f"{task}\t{sample_id}",
        "fold": fold_for_key(task, sample_id),
        "reference_score": fnum(row, "score"),
        "reference_kv_keep": fnum(row, "keep_fraction"),
        "reference_online_seconds": fnum(row, "online_seconds"),
        "reference_kept_context_tokens": fnum(row, "kept_context_tokens"),
        "reference_query_recall": fnum(row, "ours_query_coverage_recall"),
        "reference_score_gap2": fnum(row, "ours_score_gap2"),
        "reference_score_entropy": fnum(row, "ours_score_entropy"),
    }
    for feature in BASE_FEATURES:
        record[feature] = fnum(row, feature)
    record.update(selected_page_features(row, "reference"))
    return record


def pair_record(
    base: dict[str, Any],
    action: str,
    budget: int,
    rank: int,
    row: dict[str, str],
    quality_ratio: float,
) -> dict[str, Any]:
    ref_kv = max(1e-8, float(base["reference_kv_keep"]))
    ref_context = max(1.0, float(base["reference_kept_context_tokens"]))
    raw_prefix = max(1.0, float(base["raw_prefix_tokens"]))
    query_terms = fnum(row, "ours_query_coverage_terms")
    query_covered = fnum(row, "ours_query_coverage_covered")
    query_recall = fnum(row, "ours_query_coverage_recall")
    keep_fraction = fnum(row, "keep_fraction")
    kept_context = fnum(row, "kept_context_tokens")
    pair: dict[str, Any] = {
        **base,
        "action": action,
        "candidate_budget_tokens": float(budget),
        "candidate_budget_log2": math.log2(float(max(1, budget))),
        "candidate_budget_rank": float(rank),
        "candidate_budget_rank_fraction": float(rank) / max(1.0, float(len(BUDGETS) - 1)),
        "candidate_effective_budget_tokens": fnum(row, "budget_tokens"),
        "candidate_page_tokens": fnum(row, "page_tokens"),
        "candidate_keep_fraction": keep_fraction,
        "candidate_kv_relative_to_reference": keep_fraction / ref_kv,
        "candidate_kv_saving_vs_reference": 1.0 - keep_fraction / ref_kv,
        "candidate_kept_context_tokens": kept_context,
        "candidate_kept_context_fraction": kept_context / raw_prefix,
        "candidate_kept_context_relative_to_reference": kept_context / ref_context,
        "candidate_query_terms": query_terms,
        "candidate_query_covered": query_covered,
        "candidate_query_recall": query_recall,
        "candidate_query_missing": max(0.0, query_terms - query_covered),
        "candidate_query_recall_per_kv": query_recall / max(1e-8, keep_fraction),
        "candidate_query_covered_per_kv": query_covered / max(1e-8, keep_fraction),
        "candidate_score_max": fnum(row, "ours_score_max"),
        "candidate_score_mean": fnum(row, "ours_score_mean"),
        "candidate_score_gap2": fnum(row, "ours_score_gap2"),
        "candidate_score_gap3": fnum(row, "ours_score_gap3"),
        "candidate_score_entropy": fnum(row, "ours_score_entropy"),
        "candidate_score_positive_fraction": fnum(row, "ours_score_positive_fraction"),
        "candidate_score": fnum(row, "score"),
        "candidate_online_seconds": fnum(row, "online_seconds"),
    }
    pair.update(selected_page_features(row, "candidate"))
    pair["delta_query_recall_vs_reference"] = pair["candidate_query_recall"] - float(base["reference_query_recall"])
    pair["delta_selected_density_vs_reference"] = (
        pair["candidate_selected_density"] - float(base["reference_selected_density"])
    )
    pair["delta_score_gap2_vs_reference"] = pair["candidate_score_gap2"] - float(base["reference_score_gap2"])
    pair["delta_score_entropy_vs_reference"] = pair["candidate_score_entropy"] - float(base["reference_score_entropy"])
    target = quality_ratio * float(base["reference_score"])
    pair["quality_target"] = target
    pair["is_safe"] = int(pair["candidate_score"] + 1e-12 >= target)
    return pair


def build_dataset(quality_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = by_key(read_csv(REFERENCE_DIR / "task_results.csv"))
    candidates = {
        f"budget_b{budget}": by_key(read_csv(budget_dir(budget) / "task_results.csv"))
        for budget in BUDGETS
    }
    base_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key, ref_row in sorted(reference.items()):
        if any(key not in table for table in candidates.values()):
            continue
        base = base_record(key, ref_row)
        safe_pairs: list[dict[str, Any]] = []
        for rank, budget in enumerate(BUDGETS):
            action = f"budget_b{budget}"
            pair = pair_record(base, action, budget, rank, candidates[action][key], quality_ratio)
            if pair["is_safe"]:
                safe_pairs.append(pair)
            pair_rows.append(pair)
        if safe_pairs:
            oracle = min(safe_pairs, key=lambda item: (item["candidate_keep_fraction"], -item["candidate_score"]))
            base["oracle_action"] = oracle["action"]
            base["oracle_score"] = oracle["candidate_score"]
            base["oracle_kv_keep"] = oracle["candidate_keep_fraction"]
            base["oracle_online_seconds"] = oracle["candidate_online_seconds"]
        else:
            base["oracle_action"] = "reference"
            base["oracle_score"] = base["reference_score"]
            base["oracle_kv_keep"] = base["reference_kv_keep"]
            base["oracle_online_seconds"] = base["reference_online_seconds"]
        base["quality_target"] = quality_ratio * float(base["reference_score"])
        base_rows.append(base)
    return base_rows, pair_rows


def feature_names(pair_rows: list[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}):
        categories.append(f"family={family}")
    for task in sorted({str(row["task"]) for row in pair_rows}):
        categories.append(f"task={task}")
    for action in sorted({str(row["action"]) for row in pair_rows}):
        categories.append(f"action={action}")
    return PAIR_NUMERIC_FEATURES + categories


def make_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        task = str(row["task"])
        family = str(row["task_family"])
        action = str(row.get("action", ""))
        for j, name in enumerate(names):
            if name in row:
                matrix[i, j] = float(row.get(name, 0.0) or 0.0)
            elif name == f"family={family}" or name == f"task={task}" or name == f"action={action}":
                matrix[i, j] = 1.0
    return matrix


def fit_model(kind: str, class_weight: str, max_depth: int, min_leaf: int, rows: list[dict[str, Any]], names: list[str]):
    x_train = make_matrix(rows, names)
    y_train = np.array([int(row["is_safe"]) for row in rows])
    common = {
        "n_estimators": 140,
        "max_depth": max_depth if max_depth > 0 else None,
        "min_samples_leaf": min_leaf,
        "random_state": 13,
        "class_weight": "balanced_subsample" if class_weight == "balanced" else None,
        "n_jobs": -1,
    }
    if kind == "extra":
        model = ExtraTreesClassifier(**common)
    else:
        model = RandomForestClassifier(**common)
    model.fit(x_train, y_train)
    return model


def safe_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = [str(item) for item in model.classes_]
    if "1" in classes:
        return probs[:, classes.index("1")]
    return probs[:, -1]


def make_pair_tables(pair_rows: list[dict[str, Any]], model: Any, names: list[str]) -> dict[str, list[dict[str, Any]]]:
    x_all = make_matrix(pair_rows, names)
    probs = safe_probabilities(model, x_all)
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, prob in zip(pair_rows, probs):
        item = dict(row)
        item["safe_probability"] = float(prob)
        by_key[str(row["key"])].append(item)
    for rows in by_key.values():
        rows.sort(key=lambda item: float(item["candidate_keep_fraction"]))
    return by_key


def choose_action(pairs: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    safe = [row for row in pairs if float(row["safe_probability"]) >= threshold]
    if not safe:
        return None
    return min(safe, key=lambda item: (float(item["candidate_keep_fraction"]), float(item["candidate_budget_tokens"])))


def summarize(predictions: list[dict[str, Any]], split: str, threshold_tag: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ["ALL", *TASKS]:
        subset = predictions if task == "ALL" else [row for row in predictions if row["task"] == task]
        if not subset:
            continue
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
                "threshold_tag": threshold_tag,
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


def predict_with_thresholds(
    base_rows: list[dict[str, Any]],
    pair_table: dict[str, list[dict[str, Any]]],
    thresholds_by_family: dict[str, float],
    threshold_tag: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for base in base_rows:
        threshold = thresholds_by_family.get(str(base["task_family"]), thresholds_by_family.get("*", 1.01))
        selected = choose_action(pair_table[str(base["key"])], threshold)
        if selected is None:
            action = "reference"
            learned_score = float(base["reference_score"])
            learned_kv = float(base["reference_kv_keep"])
            learned_online = float(base["reference_online_seconds"])
            prob = max(float(item["safe_probability"]) for item in pair_table[str(base["key"])])
        else:
            action = str(selected["action"])
            learned_score = float(selected["candidate_score"])
            learned_kv = float(selected["candidate_keep_fraction"])
            learned_online = float(selected["candidate_online_seconds"])
            prob = float(selected["safe_probability"])
        predictions.append(
            {
                "task": base["task"],
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
                "threshold_tag": threshold_tag,
            }
        )
    return predictions, summarize(predictions, "tmp", threshold_tag)


def select_global_threshold(
    cal_rows: list[dict[str, Any]],
    pair_table: dict[str, list[dict[str, Any]]],
    min_score_ratio: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        preds, rows = predict_with_thresholds(cal_rows, pair_table, {"*": threshold}, f"global_{threshold}")
        summary = next(row for row in rows if row["task"] == "ALL")
        summary = {**summary, "threshold": threshold, "thresholds_json": json.dumps({"*": threshold})}
        candidates.append(summary)
    feasible = [row for row in candidates if float(row["learned_vs_reference"]) >= min_score_ratio]
    selected = min(
        feasible or candidates,
        key=lambda row: (
            float(row["learned_kv_keep"]),
            -float(row["learned_vs_reference"]),
            -float(row["safe_rate"]),
        ),
    )
    return {"*": float(selected["threshold"])}, selected


def select_family_thresholds(
    cal_rows: list[dict[str, Any]],
    pair_table: dict[str, list[dict[str, Any]]],
    min_score_ratio: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cal_rows:
        family_rows[str(row["task_family"])].append(row)
    for family, rows in family_rows.items():
        family_candidates: list[dict[str, Any]] = []
        for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
            _preds, summaries = predict_with_thresholds(rows, pair_table, {family: threshold}, f"{family}_{threshold}")
            summary = next(row for row in summaries if row["task"] == "ALL")
            summary = {**summary, "threshold": threshold}
            family_candidates.append(summary)
        feasible = [row for row in family_candidates if float(row["learned_vs_reference"]) >= min_score_ratio]
        selected = min(
            feasible or family_candidates,
            key=lambda row: (
                float(row["learned_kv_keep"]),
                -float(row["learned_vs_reference"]),
                -float(row["safe_rate"]),
            ),
        )
        thresholds[family] = float(selected["threshold"])
    preds, summaries = predict_with_thresholds(cal_rows, pair_table, thresholds, "family")
    summary = next(row for row in summaries if row["task"] == "ALL")
    summary = {**summary, "threshold": "", "thresholds_json": json.dumps(thresholds, sort_keys=True)}
    return thresholds, summary


def run_one(
    quality_ratio: float,
    base_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    names: list[str],
    model_kind: str,
    class_weight: str,
    max_depth: int,
    min_leaf: int,
    min_score_ratio: float,
) -> dict[str, Any]:
    train_pairs = [row for row in pair_rows if int(row["fold"]) not in {0, 1}]
    train_base = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    cal_base = [row for row in base_rows if int(row["fold"]) == 1]
    test_base = [row for row in base_rows if int(row["fold"]) == 0]
    model = fit_model(model_kind, class_weight, max_depth, min_leaf, train_pairs, names)
    pair_table = make_pair_tables(pair_rows, model, names)

    outputs: list[dict[str, Any]] = []
    threshold_specs = []
    global_thresholds, global_cal = select_global_threshold(cal_base, pair_table, min_score_ratio)
    threshold_specs.append(("global", global_thresholds, global_cal))
    family_thresholds, family_cal = select_family_thresholds(cal_base, pair_table, min_score_ratio)
    threshold_specs.append(("family", family_thresholds, family_cal))

    for calibration_mode, thresholds, cal_summary in threshold_specs:
        tag = (
            f"v13_q{str(quality_ratio).replace('.', '')}_{model_kind}_{class_weight}_"
            f"d{max_depth}_l{min_leaf}_cal{str(min_score_ratio).replace('.', '')}_{calibration_mode}"
        )
        all_predictions: list[dict[str, Any]] = []
        all_summaries: list[dict[str, Any]] = []
        for split, rows in [
            ("train", train_base),
            ("calibration", cal_base),
            ("test", test_base),
            ("all", base_rows),
        ]:
            preds, summaries = predict_with_thresholds(rows, pair_table, thresholds, tag)
            for row in preds:
                row["split"] = split
            for row in summaries:
                row["split"] = split
            all_predictions.extend(preds)
            all_summaries.extend(summaries)
        out_dir = OUT_ROOT / f"riskkv_v19_candidate_conditioned_planner_{tag}_20260712"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "planner_predictions.csv", all_predictions)
        write_csv(out_dir / "planner_summary.csv", all_summaries)
        write_csv(out_dir / "calibration_selected_summary.csv", [cal_summary])
        metadata = {
            "router_type": "candidate_conditioned_planner_v13",
            "quality_ratio": quality_ratio,
            "model_kind": model_kind,
            "class_weight": class_weight,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "min_score_ratio": min_score_ratio,
            "calibration_mode": calibration_mode,
            "thresholds": thresholds,
            "feature_names": names,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out_dir / "model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "metadata": metadata}, handle)
        feature_rows = []
        if hasattr(model, "feature_importances_"):
            for name, importance in sorted(
                zip(names, model.feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            ):
                feature_rows.append({"feature": name, "importance": float(importance)})
        write_csv(out_dir / "feature_importance.csv", feature_rows)
        test_summary = next(row for row in all_summaries if row["split"] == "test" and row["task"] == "ALL")
        all_summary = next(row for row in all_summaries if row["split"] == "all" and row["task"] == "ALL")
        outputs.append(
            {
                "router": out_dir.name,
                "output_dir": str(out_dir.relative_to(ROOT)),
                "quality_ratio": quality_ratio,
                "model_kind": model_kind,
                "class_weight": class_weight,
                "max_depth": max_depth,
                "min_leaf": min_leaf,
                "min_score_ratio": min_score_ratio,
                "calibration_mode": calibration_mode,
                "thresholds_json": json.dumps(thresholds, sort_keys=True),
                "calibration_score_ratio_measured": cal_summary["learned_vs_reference"],
                "calibration_kv_relative_measured": cal_summary["kv_relative"],
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
        )
    return {"rows": outputs}


def main() -> None:
    rows: list[dict[str, Any]] = []
    for quality_ratio in [1.0, 0.99, 0.95]:
        base_rows, pair_rows = build_dataset(quality_ratio)
        names = feature_names(pair_rows)
        for model_kind in ["rf", "extra"]:
            for class_weight in ["none", "balanced"]:
                for max_depth, min_leaf in [(5, 8), (8, 4), (0, 4)]:
                    for min_score_ratio in [1.0, 0.995, 0.99, 0.95]:
                        result = run_one(
                            quality_ratio=quality_ratio,
                            base_rows=base_rows,
                            pair_rows=pair_rows,
                            names=names,
                            model_kind=model_kind,
                            class_weight=class_weight,
                            max_depth=max_depth,
                            min_leaf=min_leaf,
                            min_score_ratio=min_score_ratio,
                        )
                        for row in result["rows"]:
                            if max_depth == 0:
                                row["max_depth"] = "none"
                            rows.append(row)
                        print(json.dumps(rows[-2:], ensure_ascii=False), flush=True)
    out_path = OUT_ROOT / "riskkv_v19_candidate_conditioned_planner_v13_compare_summary_20260712.csv"
    write_csv(out_path, rows)
    rows.sort(
        key=lambda row: (
            float(row["test_score_ratio"]),
            -float(row["test_kv_relative"]),
            float(row["all_score_ratio"]),
            -float(row["all_kv_relative"]),
        ),
        reverse=True,
    )
    selected_path = OUT_ROOT / "riskkv_v19_candidate_conditioned_planner_v13_top20_20260712.json"
    selected_path.write_text(json.dumps(rows[:20], indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    print(json.dumps(rows[:20], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
