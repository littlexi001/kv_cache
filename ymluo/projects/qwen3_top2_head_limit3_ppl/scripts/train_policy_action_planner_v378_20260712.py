#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")

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

RUNTIME_NUMERIC_FEATURES = [
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
    "ours_query_coverage_terms",
    "ours_query_coverage_covered",
    "ours_query_coverage_recall",
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

DEFAULT_CANDIDATES = {
    "policy_v360": {
        "results": "outputs/riskkv_v19_v360_lowkv_certificate_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v360_lowkv_certificate_20260712.json",
    },
    "policy_v363": {
        "results": "outputs/riskkv_v19_v363_taskwise_lowkv_mix_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v363_taskwise_lowkv_mix_20260712.json",
    },
    "policy_v365": {
        "results": "outputs/riskkv_v19_v365_ultra_skeleton_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v365_ultra_skeleton_all_20260712.json",
    },
    "policy_v368": {
        "results": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json",
    },
    "policy_v373": {
        "results": "outputs/riskkv_v19_v373_selective_direct_ladder_all_20260712_lowkv_extreme_1to10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v373_selective_direct_ladder_20260712.json",
    },
    "policy_v375": {
        "results": "outputs/riskkv_v19_v375_pareto_fused_lowkv_m100_20260712_lowkv_pareto_fusion_m100_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v375_pareto_fused_lowkv_20260712.json",
    },
    "policy_v376": {
        "results": "outputs/riskkv_v19_v376_strict10_pareto_fused_m20_20260712_lowkv_strict10_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v376_strict10_pareto_fused_20260712.json",
    },
    "policy_v377": {
        "results": "outputs/riskkv_v19_v377_global_pareto_knapsack_20260712_lowkv_global_pareto_m20_bDyn_pDyn/task_results.csv",
        "config": "configs/riskkv_task_policy_v377_global_pareto_knapsack_20260712.json",
    },
}

SHORT_KEY_MAP = {
    "scorer": "ours_scorer",
    "coverage_mmr_weight": "ours_coverage_mmr_weight",
    "coverage_mmr_max_terms": "ours_coverage_mmr_max_terms",
    "mmr_lambda": "ours_mmr_lambda",
    "coverage_certificate_budget_fraction": "ours_coverage_certificate_budget_fraction",
    "coverage_certificate_min_terms": "ours_coverage_certificate_min_terms",
    "coverage_risk_min_recall": "ours_coverage_risk_min_recall",
    "coverage_risk_min_terms": "ours_coverage_risk_min_terms",
    "coverage_risk_budget_tokens": "ours_coverage_risk_budget_tokens",
    "anchor_window_tokens": "ours_anchor_window_tokens",
    "span_repack_window_tokens": "ours_span_repack_window_tokens",
    "span_repack_budget_fraction": "ours_span_repack_budget_fraction",
    "span_repack_top_pages": "ours_span_repack_top_pages",
    "span_repack_min_score": "ours_span_repack_min_score",
    "span_repack_score_mode": "ours_span_repack_score_mode",
    "retry_budget_tokens": "ours_retry_budget_tokens",
    "retry_full_fallback_tasks": "ours_retry_full_fallback_tasks",
    "consistency_budget_tokens": "ours_consistency_budget_tokens",
    "consistency_probe_max_tokens": "ours_consistency_probe_max_tokens",
    "consistency_requires_score_risk": "ours_consistency_requires_score_risk",
    "score_risk_budget_tokens": "ours_score_risk_budget_tokens",
    "score_risk_min_gap2": "ours_score_risk_min_gap2",
    "score_risk_min_gap3": "ours_score_risk_min_gap3",
    "score_risk_max_gap2": "ours_score_risk_max_gap2",
    "score_risk_max_gap3": "ours_score_risk_max_gap3",
    "score_risk_max_entropy": "ours_score_risk_max_entropy",
    "score_risk_entropy_at_most": "ours_score_risk_entropy_at_most",
    "score_risk_min_top_score": "ours_score_risk_min_top_score",
    "score_risk_mean_at_least": "ours_score_risk_mean_at_least",
    "score_risk_mean_at_most": "ours_score_risk_mean_at_most",
    "score_risk_raw_prefix_at_most": "ours_score_risk_raw_prefix_at_most",
    "score_risk_raw_prefix_at_least": "ours_score_risk_raw_prefix_at_least",
    "score_risk_linear_threshold": "ours_score_risk_linear_threshold",
    "score_risk_gap2_weight": "ours_score_risk_gap2_weight",
    "score_risk_gap3_weight": "ours_score_risk_gap3_weight",
    "score_risk_top_score_weight": "ours_score_risk_top_score_weight",
    "score_safe_min_gap2": "ours_score_safe_min_gap2",
    "score_safe_min_gap3": "ours_score_safe_min_gap3",
    "score_safe_max_entropy": "ours_score_safe_max_entropy",
    "score_safe_min_top_score": "ours_score_safe_min_top_score",
    "score_safe_mean_at_least": "ours_score_safe_mean_at_least",
    "score_safe_raw_prefix_at_most": "ours_score_safe_raw_prefix_at_most",
    "score_safe_raw_prefix_at_least": "ours_score_safe_raw_prefix_at_least",
    "score_safe_linear_threshold": "ours_score_safe_linear_threshold",
    "budget_ladder_tokens": "ours_budget_ladder_tokens",
    "budget_ladder_gap2_thresholds": "ours_budget_ladder_gap2_thresholds",
    "budget_ladder_entropy_thresholds": "ours_budget_ladder_entropy_thresholds",
    "budget_ladder_top_score_thresholds": "ours_budget_ladder_top_score_thresholds",
    "graph_bridge_budget_fraction": "ours_graph_bridge_budget_fraction",
    "graph_bridge_seed_pages": "ours_graph_bridge_seed_pages",
    "graph_bridge_max_terms": "ours_graph_bridge_max_terms",
    "graph_bridge_min_score": "ours_graph_bridge_min_score",
    "coarse_to_fine_group_pages": "ours_coarse_to_fine_group_pages",
    "coarse_to_fine_candidate_multiplier": "ours_coarse_to_fine_candidate_multiplier",
    "coarse_to_fine_neighbor_groups": "ours_coarse_to_fine_neighbor_groups",
    "layer_router_mode": "ours_layer_router_mode",
    "layer_router_low_fraction": "ours_layer_router_low_fraction",
    "layer_router_low_budget_tokens": "ours_layer_router_low_budget_tokens",
    "action_router_mode": "ours_action_router_mode",
    "learned_router_base_action_router_mode": "ours_learned_router_base_action_router_mode",
    "passage_closure_budget_fraction": "ours_passage_closure_budget_fraction",
    "passage_closure_radius_pages": "ours_passage_closure_radius_pages",
    "structured_fingerprint_budget_fraction": "ours_structured_fingerprint_budget_fraction",
    "direct_summary_max_words": "ours_direct_summary_max_words",
    "short_decode_max_tokens": "ours_short_decode_max_tokens",
    "support_window_radius_words": "ours_support_window_radius_words",
    "support_window_min_query_terms": "ours_support_window_min_query_terms",
}

TOGGLE_TASK_KEYS = {
    "bridge": "ours_bridge_tasks",
    "graph_bridge": "ours_graph_bridge_tasks",
    "coarse_to_fine": "ours_coarse_to_fine_tasks",
    "anchor_window": "ours_anchor_window_tasks",
    "span_repack": "ours_span_repack_tasks",
    "layer_router": "ours_layer_router_tasks",
    "action_router": "ours_action_router_tasks",
    "score_safe": "ours_score_safe_tasks",
    "full_fallback": "ours_full_fallback_tasks",
    "retry_full_fallback": "ours_retry_full_fallback_tasks",
    "label_support": "ours_label_support_tasks",
    "passage_closure": "ours_passage_closure_tasks",
    "structured_fingerprint": "ours_structured_fingerprint_tasks",
    "direct_structured_answer": "ours_direct_structured_answer_tasks",
    "short_decode": "ours_short_decode_tasks",
    "output_verifier": "ours_output_verifier_tasks",
    "score_risk": "ours_score_risk_tasks",
    "budget_ladder": "ours_budget_ladder_tasks",
    "coverage_risk": "ours_coverage_risk_tasks",
    "coverage_certificate": "ours_coverage_certificate_tasks",
    "consistency_verifier": "ours_consistency_verifier_tasks",
    "grounding_verifier": "ours_grounding_verifier_tasks",
    "title_anchor": "ours_title_anchor_tasks",
    "support_window_verifier": "ours_support_window_verifier_tasks",
}

DIRECT_KEYS = {
    "budget_tokens",
    "sink_tokens",
    "recent_tokens",
    "page_tokens",
    "ours_scorer",
    "ours_coverage_mmr_weight",
    "ours_coverage_mmr_max_terms",
    "ours_mmr_lambda",
    "ours_coverage_certificate_budget_fraction",
    "ours_coverage_certificate_min_terms",
    "ours_coverage_risk_min_recall",
    "ours_coverage_risk_min_terms",
    "ours_coverage_risk_budget_tokens",
    "ours_anchor_window_tokens",
    "ours_span_repack_window_tokens",
    "ours_span_repack_budget_fraction",
    "ours_span_repack_top_pages",
    "ours_span_repack_min_score",
    "ours_span_repack_score_mode",
    "ours_retry_budget_tokens",
    "ours_retry_full_fallback_tasks",
    "ours_consistency_budget_tokens",
    "ours_consistency_probe_max_tokens",
    "ours_consistency_requires_score_risk",
    "ours_score_risk_budget_tokens",
    "ours_score_risk_min_gap2",
    "ours_score_risk_min_gap3",
    "ours_score_risk_max_gap2",
    "ours_score_risk_max_gap3",
    "ours_score_risk_max_entropy",
    "ours_score_risk_entropy_at_most",
    "ours_score_risk_min_top_score",
    "ours_score_risk_mean_at_least",
    "ours_score_risk_mean_at_most",
    "ours_score_risk_raw_prefix_at_most",
    "ours_score_risk_raw_prefix_at_least",
    "ours_score_risk_linear_threshold",
    "ours_score_risk_gap2_weight",
    "ours_score_risk_gap3_weight",
    "ours_score_risk_top_score_weight",
    "ours_score_safe_min_gap2",
    "ours_score_safe_min_gap3",
    "ours_score_safe_max_entropy",
    "ours_score_safe_min_top_score",
    "ours_score_safe_mean_at_least",
    "ours_score_safe_raw_prefix_at_most",
    "ours_score_safe_raw_prefix_at_least",
    "ours_score_safe_linear_threshold",
    "ours_budget_ladder_tokens",
    "ours_budget_ladder_gap2_thresholds",
    "ours_budget_ladder_entropy_thresholds",
    "ours_budget_ladder_top_score_thresholds",
    "ours_graph_bridge_budget_fraction",
    "ours_graph_bridge_seed_pages",
    "ours_graph_bridge_max_terms",
    "ours_graph_bridge_min_score",
    "ours_coarse_to_fine_group_pages",
    "ours_coarse_to_fine_candidate_multiplier",
    "ours_coarse_to_fine_neighbor_groups",
    "ours_layer_router_mode",
    "ours_layer_router_low_fraction",
    "ours_layer_router_low_budget_tokens",
    "ours_action_router_mode",
    "ours_learned_router_model_path",
    "ours_learned_router_action_policy_json",
    "ours_learned_router_confidence_threshold",
    "ours_learned_router_default_action",
    "ours_learned_router_base_action_router_mode",
    "ours_passage_closure_budget_fraction",
    "ours_passage_closure_radius_pages",
    "ours_structured_fingerprint_budget_fraction",
    "ours_direct_summary_max_words",
    "ours_short_decode_max_tokens",
    "ours_support_window_radius_words",
    "ours_support_window_min_query_terms",
    *TOGGLE_TASK_KEYS.values(),
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
    out = {}
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


def parse_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "__extends" in payload:
        parent_path = Path(str(payload["__extends"]))
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = parse_policy(parent_path)
        merged = {key: dict(value) if isinstance(value, dict) else value for key, value in parent.items()}
        overlay_all = payload.get("__overlay_all_tasks", {})
        if overlay_all and not isinstance(overlay_all, dict):
            raise ValueError(f"{path}: __overlay_all_tasks must be a dict")
        child_tasks = dict(payload.get("tasks", {}))
        task_sources = payload.get("__task_sources", {})
        if task_sources and not isinstance(task_sources, dict):
            raise ValueError(f"{path}: __task_sources must be a dict")
        for target_task, source in task_sources.items():
            if isinstance(source, str):
                source_policy_spec = source
                source_task = str(target_task)
            elif isinstance(source, dict):
                source_policy_spec = str(source.get("policy", ""))
                source_task = str(source.get("task", target_task))
            else:
                raise ValueError(f"{path}: __task_sources values must be strings or dicts")
            if not source_policy_spec:
                raise ValueError(f"{path}: __task_sources entries require a policy path")
            source_path = Path(source_policy_spec)
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            source_policy = parse_policy(source_path)
            source_value = source_policy.get(source_task)
            if source_value is None:
                raise ValueError(f"{path}: task source {source_policy_spec!r} has no task {source_task!r}")
            if isinstance(source_value, int):
                source_value = {"budget_tokens": source_value}
            if not isinstance(source_value, dict):
                raise ValueError(f"{path}: __task_sources resolved values must be dicts or ints")
            child_tasks[str(target_task)] = dict(source_value)
        for key, value in payload.items():
            if key.startswith("__") or key == "tasks":
                continue
            child_tasks[key] = value
        if overlay_all:
            keys = [key for key in merged if key != "*"]
            keys.extend(key for key in child_tasks if key != "*" and key not in keys)
            for key in keys:
                current = merged.get(key)
                merged[key] = dict(current) if isinstance(current, dict) else {}
                merged[key].update(overlay_all)
        if "*" in child_tasks and isinstance(child_tasks["*"], dict):
            current = merged.get("*")
            merged["*"] = dict(current) if isinstance(current, dict) else {}
            merged["*"].update(child_tasks["*"])
        for key, value in child_tasks.items():
            if key == "*":
                continue
            if isinstance(value, int):
                value = {"budget_tokens": value}
            current = merged.get(key)
            merged[key] = dict(current) if isinstance(current, dict) else {}
            if isinstance(value, dict):
                merged[key].update(value)
        return merged
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict):
        return payload["tasks"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"{path}: invalid policy JSON")


def fragment_for_task(policy: dict[str, Any], task: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(policy.get("*"), dict):
        merged.update(policy["*"])
    if isinstance(policy.get(task), dict):
        merged.update(policy[task])
    return merged


def normalize_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in fragment.items():
        if key in TOGGLE_TASK_KEYS:
            continue
        out_key = SHORT_KEY_MAP.get(key, key)
        if out_key in DIRECT_KEYS:
            normalized[out_key] = value
    for shortcut, task_key in TOGGLE_TASK_KEYS.items():
        normalized[task_key] = "__TASK__" if bool(fragment.get(shortcut, False)) else "__EMPTY__"
    if bool(fragment.get("bridge", False)):
        normalized.setdefault("ours_scorer", "hybrid_late_mmr_multiscale_task_bridge_flow")
    return normalized


def build_action_policy(root: Path, candidate_specs: dict[str, dict[str, str]]) -> dict[str, Any]:
    actions: dict[str, Any] = {}
    for action, spec in candidate_specs.items():
        config_path = root / spec["config"]
        if not config_path.exists():
            continue
        policy = parse_policy(config_path)
        action_entry: dict[str, Any] = {}
        for task in TASKS:
            fragment = normalize_fragment(fragment_for_task(policy, task))
            if fragment:
                action_entry[task] = fragment
        if action_entry:
            actions[action] = action_entry
    return {"metadata": {"description": "v378 sample-level policy action fragments."}, "actions": actions}


def feature_base(row: dict[str, str], full_row: dict[str, str], quality_ratio: float, quality_margin: float) -> dict[str, Any]:
    task = row["task"]
    sample_id = row["sample_id"]
    full_score = fnum(full_row, "score")
    target = max(0.0, quality_ratio * full_score, full_score - quality_margin)
    record: dict[str, Any] = {
        "task": task,
        "task_family": task_family(task),
        "sample_id": sample_id,
        "fold": fold_for_key(task, sample_id),
        "full_score": full_score,
        "quality_target": target,
        "base_score": fnum(row, "score"),
        "base_kv_keep": fnum(row, "keep_fraction"),
        "base_online_seconds": fnum(row, "online_seconds"),
    }
    for feature in RUNTIME_NUMERIC_FEATURES:
        record[feature] = fnum(row, feature)
    return record


def candidate_cost_tokens(rows: list[dict[str, str]]) -> int:
    values = [fnum(row, "kept_context_tokens") or fnum(row, "kept_prefix_tokens") for row in rows]
    return max(1, int(round(mean(values))))


def budget_features(base: dict[str, Any], cost_tokens: int, rank: int, count: int) -> dict[str, float]:
    raw_prefix = max(1.0, float(base.get("raw_prefix_tokens", 0.0) or 0.0))
    raw_prompt = max(1.0, float(base.get("raw_prompt_tokens", 0.0) or 0.0))
    page_count = max(1.0, float(base.get("page_count", 0.0) or 0.0))
    page_tokens = 128.0
    budget = float(max(1, cost_tokens))
    return {
        "candidate_budget_tokens": budget,
        "candidate_budget_log2": math.log2(budget),
        "candidate_budget_fraction_of_context": budget / raw_prefix,
        "candidate_budget_fraction_of_prompt": budget / raw_prompt,
        "candidate_budget_pages": budget / page_tokens,
        "candidate_budget_page_fraction": budget / max(1.0, page_count * page_tokens),
        "candidate_budget_rank": float(rank),
        "candidate_budget_rank_fraction": float(rank) / max(1.0, float(count - 1)),
    }


def make_feature_names(pair_rows: list[dict[str, Any]]) -> list[str]:
    categories = []
    categories.extend(f"family={family}" for family in sorted(set(FAMILY_BY_TASK.values()) | {"other"}))
    categories.extend(f"task={task}" for task in sorted({str(row["task"]) for row in pair_rows}))
    categories.extend(f"action={action}" for action in sorted({str(row["action"]) for row in pair_rows}))
    return RUNTIME_NUMERIC_FEATURES + BUDGET_FEATURES + categories


def vectorize(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    matrix = []
    for row in rows:
        task = str(row["task"])
        family = task_family(task)
        action = str(row["action"])
        vector = []
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
    probabilities = model.predict_proba([vector])
    classes = [str(item) for item in getattr(model, "classes_", [])]
    if "1" in classes:
        return float(probabilities[0][classes.index("1")])
    return float(max(probabilities[0]))


def summarize_predictions(rows: list[dict[str, Any]], split: str, full_online: float) -> list[dict[str, Any]]:
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
                "safe_rate": mean([1.0 if float(row["learned_score"]) + 1e-12 >= float(row["quality_target"]) else 0.0 for row in subset]),
                "mean_safe_probability": mean([float(row["safe_probability"]) for row in subset]),
            }
        )
    return out


def choose_predictions(
    base_rows: list[dict[str, Any]],
    pair_rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    actions_by_cost: list[str],
    model: Any | None,
    feature_names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for base in base_rows:
        task = str(base["task"])
        sample_id = str(base["sample_id"])
        selected: dict[str, Any] | None = None
        selected_probability = 0.0
        best_action = "reference"
        best_probability = 0.0
        for action in actions_by_cost:
            pair = pair_rows_by_key.get((task, sample_id, action))
            if pair is None:
                continue
            if "safe_probability_model" in pair:
                probability = float(pair["safe_probability_model"])
            elif model is not None:
                probability = safe_probability(model, vectorize([pair], feature_names)[0])
            else:
                probability = 0.0
            if probability > best_probability:
                best_probability = probability
                best_action = action
            if probability >= threshold:
                selected = pair
                selected_probability = probability
                break
        if selected is None:
            learned_action = "reference"
            learned_score = float(base["base_score"])
            learned_kv = float(base["base_kv_keep"])
            learned_online = float(base["base_online_seconds"])
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
                "learned_action": learned_action,
                "oracle_action": base["oracle_action"],
                "safe_probability": selected_probability,
                "fallback_reason": fallback_reason,
                "full_score": base["full_score"],
                "base_score": base["base_score"],
                "learned_score": learned_score,
                "oracle_score": base["oracle_score"],
                "quality_target": base["quality_target"],
                "base_kv_keep": base["base_kv_keep"],
                "learned_kv_keep": learned_kv,
                "oracle_kv_keep": base["oracle_kv_keep"],
                "base_online_seconds": base["base_online_seconds"],
                "learned_online_seconds": learned_online,
                "oracle_online_seconds": base["oracle_online_seconds"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--base-action", default="policy_v365")
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--quality-margin", type=float, default=0.05)
    parser.add_argument("--kv-limit", type=float, default=0.10)
    parser.add_argument("--speed-min", type=float, default=2.5)
    parser.add_argument("--full-online", type=float, default=3.0988)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_policy_action_planner_v378_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v378_policy_action_planner_20260712.json")
    args = parser.parse_args()

    root = Path(args.root)
    full_rows = by_key(read_csv(root / args.full_results))
    candidate_tables: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    candidate_specs: dict[str, dict[str, str]] = {}
    for action, spec in DEFAULT_CANDIDATES.items():
        path = root / spec["results"]
        config_path = root / spec["config"]
        if path.exists() and config_path.exists():
            table = by_key(read_csv(path))
            if table:
                candidate_tables[action] = table
                candidate_specs[action] = spec

    if args.base_action not in candidate_tables:
        raise FileNotFoundError(f"Base action {args.base_action} is unavailable; completed candidate results are required.")
    if not candidate_tables:
        raise FileNotFoundError("No completed candidate task_results.csv files found.")

    common_keys = set(full_rows) & set(candidate_tables[args.base_action])
    for table in candidate_tables.values():
        common_keys &= set(table)
    if not common_keys:
        raise ValueError("No common samples across full, base, and candidate tables.")

    candidate_costs = {
        action: candidate_cost_tokens([row for key, row in table.items() if key in common_keys])
        for action, table in candidate_tables.items()
    }
    actions_by_cost = sorted(candidate_tables, key=lambda action: (candidate_costs[action], action))

    base_rows = []
    pair_rows = []
    for key in sorted(common_keys):
        task, sample_id = key
        base_row = candidate_tables[args.base_action][key]
        full_row = full_rows[key]
        base = feature_base(base_row, full_row, args.quality_ratio, args.quality_margin)
        candidate_pairs = []
        for rank, action in enumerate(actions_by_cost):
            cand = candidate_tables[action][key]
            pair = {
                **base,
                "action": action,
                "candidate_score": fnum(cand, "score"),
                "candidate_kv_keep": fnum(cand, "keep_fraction"),
                "candidate_online_seconds": fnum(cand, "online_seconds"),
                "candidate_cost_tokens": candidate_costs[action],
                "is_safe": int(fnum(cand, "score") + 1e-12 >= float(base["quality_target"])),
            }
            pair.update(budget_features(base, candidate_costs[action], rank, len(actions_by_cost)))
            pair_rows.append(pair)
            candidate_pairs.append(pair)
        safe_pairs = [row for row in candidate_pairs if int(row["is_safe"]) == 1]
        if safe_pairs:
            oracle = min(safe_pairs, key=lambda row: (float(row["candidate_kv_keep"]), -float(row["candidate_score"])))
        else:
            oracle = max(candidate_pairs, key=lambda row: (float(row["candidate_score"]), -float(row["candidate_kv_keep"])))
        base.update(
            {
                "oracle_action": str(oracle["action"]),
                "oracle_score": float(oracle["candidate_score"]),
                "oracle_kv_keep": float(oracle["candidate_kv_keep"]),
                "oracle_online_seconds": float(oracle["candidate_online_seconds"]),
            }
        )
        base_rows.append(base)

    from sklearn.ensemble import RandomForestClassifier

    train_pairs = [row for row in pair_rows if int(row["fold"]) not in {0, 1}]
    cal_base = [row for row in base_rows if int(row["fold"]) == 1]
    test_base = [row for row in base_rows if int(row["fold"]) == 0]
    train_base = [row for row in base_rows if int(row["fold"]) not in {0, 1}]
    feature_names = make_feature_names(pair_rows)
    model = RandomForestClassifier(
        n_estimators=96,
        max_depth=6,
        min_samples_leaf=4,
        random_state=17,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(vectorize(train_pairs, feature_names), [int(row["is_safe"]) for row in train_pairs])
    probability_matrix = vectorize(pair_rows, feature_names)
    probabilities = model.predict_proba(probability_matrix)
    classes = [str(item) for item in getattr(model, "classes_", [])]
    safe_index = classes.index("1") if "1" in classes else int(probabilities.shape[1] - 1)
    for row, probability in zip(pair_rows, probabilities[:, safe_index]):
        row["safe_probability_model"] = float(probability)

    pair_by_key = {(str(row["task"]), str(row["sample_id"]), str(row["action"])): row for row in pair_rows}
    threshold_rows = []
    feasible = []
    for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
        predictions = choose_predictions(cal_base, pair_by_key, actions_by_cost, model, feature_names, threshold)
        summary = next(row for row in summarize_predictions(predictions, "calibration", args.full_online) if row["task"] == "ALL")
        learned_vs_full = float(summary["learned_vs_full"] or 0.0)
        learned_kv = float(summary["learned_kv_keep"])
        learned_speed = float(summary["learned_speed_vs_full"] or 0.0)
        row = {
            **summary,
            "threshold": threshold,
            "feasible": int(
                learned_vs_full >= args.quality_ratio
                and learned_kv <= args.kv_limit
                and learned_speed >= args.speed_min
            ),
        }
        if int(row["feasible"]):
            feasible.append(row)
        threshold_rows.append(row)
    if feasible:
        selected_threshold = max(
            feasible,
            key=lambda row: (
                float(row["learned_score"]),
                -float(row["learned_kv_keep"]),
                float(row["learned_speed_vs_full"] or 0.0),
            ),
        )
    else:
        selected_threshold = max(
            threshold_rows,
            key=lambda row: (
                float(row["learned_vs_full"] or 0.0) >= args.quality_ratio,
                float(row["learned_kv_keep"]) <= args.kv_limit,
                float(row["learned_score"]),
                -float(row["learned_kv_keep"]),
            ),
        )
    threshold = float(selected_threshold["threshold"])

    prediction_rows = []
    summary_rows = []
    for split, rows in [
        ("train", train_base),
        ("calibration", cal_base),
        ("test", test_base),
        ("all", base_rows),
    ]:
        predictions = choose_predictions(rows, pair_by_key, actions_by_cost, model, feature_names, threshold)
        for row in predictions:
            row["split"] = split
            row["threshold"] = threshold
        prediction_rows.extend(predictions)
        for row in summarize_predictions(predictions, split, args.full_online):
            row["threshold"] = threshold
            summary_rows.append(row)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = build_action_policy(root, candidate_specs)
    candidate_budget_tokens = {action: candidate_costs[action] for action in actions_by_cost}
    metadata = {
        "router_type": "budget_pair_planner_v12",
        "planner_name": "policy_action_planner_v378",
        "candidate_actions": actions_by_cost,
        "candidate_budget_tokens": candidate_budget_tokens,
        "safe_probability_threshold": threshold,
        "feature_names": feature_names,
        "quality_ratio": args.quality_ratio,
        "quality_margin": args.quality_margin,
        "kv_limit": args.kv_limit,
        "speed_min": args.speed_min,
        "base_action": args.base_action,
        "candidate_results": {action: spec["results"] for action, spec in candidate_specs.items()},
        "candidate_configs": {action: spec["config"] for action, spec in candidate_specs.items()},
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

    config = {
        "__extends": "riskkv_task_policy_v365_ultra_skeleton_all_20260712.json",
        "__overlay_all_tasks": {
            "action_router": True,
            "ours_action_router_mode": "learned_budget_planner_v2",
            "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
            "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
            "ours_learned_router_confidence_threshold": threshold,
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "v293_rules",
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    feature_rows = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(feature_names, model.feature_importances_), key=lambda item: float(item[1]), reverse=True)
    ]
    write_csv(output_dir / "feature_importance.csv", feature_rows)

    selected_all = next(row for row in summary_rows if row["split"] == "all" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps(selected_all, ensure_ascii=False))


if __name__ == "__main__":
    main()
