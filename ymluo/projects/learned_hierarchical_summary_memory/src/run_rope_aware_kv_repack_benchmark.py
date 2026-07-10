from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    LONG_BENCH_PROMPTS,
    SUMMARY_TASKS,
    BenchCase,
    exact_match_any,
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    parse_int_tuple,
    rouge_l_f1,
)

from run_position_mode_planner_from_repack_results import (  # noqa: E402
    TEXT_FEATURE_NAMES,
    MLP,
    content_words,
)
from run_output_level_risk_verifier_from_repack_results import output_features  # noqa: E402


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    longbench_data_dir: str
    ruler_data_dir: str
    longbench_tasks: tuple[str, ...]
    ruler_tasks: tuple[str, ...]
    ruler_context_lengths: tuple[int, ...]
    max_examples_per_task: int
    case_start: int
    case_limit: int
    runtime_methods: tuple[str, ...]
    max_context_tokens: int
    page_tokens: int
    top_k: int
    max_new_tokens_exact: int
    max_new_tokens_summary: int
    dtype: str
    attn_implementation: str
    two_stage_planner_path: str
    two_stage_threshold_full: float
    two_stage_threshold_k3: float
    variable_budget_planner_path: str
    variable_budget_policy: str
    variable_budget_tail_threshold: float
    variable_budget_temperature: float
    variable_budget_min_budget: int
    variable_budget_source: str
    consistency_probe_budgets: tuple[int, ...]
    consistency_probe_kl_threshold: float
    consistency_probe_require_top1_agree: bool
    teacher_verifier_budgets: tuple[int, ...]
    teacher_verifier_fallback_nll: float
    output_verifier_path: str
    output_verifier_threshold: float
    output_verifier_source: str
    output_verifier_budgets: tuple[int, ...]
    output_verifier_mode: str
    output_verifier_min_budget: int
    output_verifier_long_ruler_min_budget: int
    output_verifier_long_ruler_context_threshold: int
    seed: int


@dataclass
class ResultRow:
    benchmark: str
    task: str
    case_id: str
    method: str
    context_tokens: int
    active_kv_tokens: int
    query_tokens: int
    selected_pages: str
    planner_action: str
    planner_seconds: float
    prefill_seconds: float
    full_prefill_seconds: float
    gather_seconds: float
    repack_seconds: float
    query_seconds: float
    decode_seconds: float
    total_online_seconds: float
    speedup_vs_full_online: float
    end_to_end_seconds: float
    speedup_vs_full_e2e: float
    amortized4_speedup_vs_full_e2e: float
    amortized16_speedup_vs_full_e2e: float
    prediction: str
    answers: str
    exact_correct: int
    rouge_l: float
    score: float
    answer_nll: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Small benchmark for RoPE-aware cache-native KV page repacking.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--longbench_data_dir", default="ymluo/external/KVCache-Factory/data/LongBench")
    parser.add_argument("--ruler_data_dir", default="ymluo/external/KVCache-Factory/data/RULER")
    parser.add_argument("--longbench_tasks", default="hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count")
    parser.add_argument("--ruler_tasks", default="niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe")
    parser.add_argument("--ruler_context_lengths", default="4096")
    parser.add_argument("--max_examples_per_task", type=int, default=1)
    parser.add_argument("--case_start", type=int, default=0)
    parser.add_argument("--case_limit", type=int, default=0)
    parser.add_argument(
        "--runtime_methods",
        default="all",
        help="Comma-separated method rows to run; 'all' preserves the full benchmark suite.",
    )
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--page_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--max_new_tokens_exact", type=int, default=48)
    parser.add_argument("--max_new_tokens_summary", type=int, default=120)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--two_stage_planner_path", default="")
    parser.add_argument("--two_stage_threshold_full", type=float, default=0.01)
    parser.add_argument("--two_stage_threshold_k3", type=float, default=0.01)
    parser.add_argument("--variable_budget_planner_path", default="")
    parser.add_argument("--variable_budget_policy", choices=["argmax", "tail_risk"], default="tail_risk")
    parser.add_argument("--variable_budget_tail_threshold", type=float, default=0.35)
    parser.add_argument("--variable_budget_temperature", type=float, default=1.0)
    parser.add_argument("--variable_budget_min_budget", type=int, default=0)
    parser.add_argument(
        "--variable_budget_source",
        default="auto",
        help="Source one-hot to use for variable-budget checkpoints; 'auto' infers from benchmark/example count.",
    )
    parser.add_argument(
        "--consistency_probe_budgets",
        default="",
        help="Comma-separated compact budgets for full-cache teacher logit-consistency probing; empty disables it.",
    )
    parser.add_argument("--consistency_probe_kl_threshold", type=float, default=0.05)
    parser.add_argument("--consistency_probe_require_top1_agree", action="store_true")
    parser.add_argument(
        "--teacher_verifier_budgets",
        default="",
        help="Comma-separated compact budgets for full-cache teacher likelihood verification; empty disables it.",
    )
    parser.add_argument(
        "--teacher_verifier_fallback_nll",
        type=float,
        default=float("inf"),
        help="Fallback to full decode if the best compact candidate has teacher NLL above this value.",
    )
    parser.add_argument("--output_verifier_path", default="")
    parser.add_argument("--output_verifier_threshold", type=float, default=0.7)
    parser.add_argument(
        "--output_verifier_source",
        default="auto",
        help="Source one-hot to use for output-level verifier checkpoints; 'auto' follows the training groups.",
    )
    parser.add_argument(
        "--output_verifier_budgets",
        default="",
        help="Optional compact budgets to evaluate for output-level verifier; empty uses checkpoint actions.",
    )
    parser.add_argument(
        "--output_verifier_mode",
        choices=["all", "prefix"],
        default="all",
        help="all decodes every candidate before verifying; prefix stops at the first safe candidate.",
    )
    parser.add_argument("--output_verifier_min_budget", type=int, default=1)
    parser.add_argument("--output_verifier_long_ruler_min_budget", type=int, default=0)
    parser.add_argument("--output_verifier_long_ruler_context_threshold", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026070701)
    args = parser.parse_args()
    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        longbench_data_dir=args.longbench_data_dir,
        ruler_data_dir=args.ruler_data_dir,
        longbench_tasks=parse_csv_tuple(args.longbench_tasks),
        ruler_tasks=parse_csv_tuple(args.ruler_tasks),
        ruler_context_lengths=parse_int_tuple(args.ruler_context_lengths),
        max_examples_per_task=args.max_examples_per_task,
        case_start=args.case_start,
        case_limit=args.case_limit,
        runtime_methods=parse_csv_tuple(args.runtime_methods),
        max_context_tokens=args.max_context_tokens,
        page_tokens=args.page_tokens,
        top_k=args.top_k,
        max_new_tokens_exact=args.max_new_tokens_exact,
        max_new_tokens_summary=args.max_new_tokens_summary,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        two_stage_planner_path=args.two_stage_planner_path,
        two_stage_threshold_full=args.two_stage_threshold_full,
        two_stage_threshold_k3=args.two_stage_threshold_k3,
        variable_budget_planner_path=args.variable_budget_planner_path,
        variable_budget_policy=args.variable_budget_policy,
        variable_budget_tail_threshold=args.variable_budget_tail_threshold,
        variable_budget_temperature=args.variable_budget_temperature,
        variable_budget_min_budget=args.variable_budget_min_budget,
        variable_budget_source=args.variable_budget_source,
        consistency_probe_budgets=parse_int_tuple(args.consistency_probe_budgets),
        consistency_probe_kl_threshold=args.consistency_probe_kl_threshold,
        consistency_probe_require_top1_agree=args.consistency_probe_require_top1_agree,
        teacher_verifier_budgets=parse_int_tuple(args.teacher_verifier_budgets),
        teacher_verifier_fallback_nll=args.teacher_verifier_fallback_nll,
        output_verifier_path=args.output_verifier_path,
        output_verifier_threshold=args.output_verifier_threshold,
        output_verifier_source=args.output_verifier_source,
        output_verifier_budgets=parse_int_tuple(args.output_verifier_budgets),
        output_verifier_mode=args.output_verifier_mode,
        output_verifier_min_budget=args.output_verifier_min_budget,
        output_verifier_long_ruler_min_budget=args.output_verifier_long_ruler_min_budget,
        output_verifier_long_ruler_context_threshold=args.output_verifier_long_ruler_context_threshold,
        seed=args.seed,
    )


def bench_config(config: Config) -> BenchConfig:
    return BenchConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        adapter_path="",
        longbench_data_dir=config.longbench_data_dir,
        ruler_data_dir=config.ruler_data_dir,
        longbench_tasks=config.longbench_tasks,
        ruler_tasks=config.ruler_tasks,
        ruler_context_lengths=config.ruler_context_lengths,
        methods=(),
        max_examples_per_task=config.max_examples_per_task,
        block_tokens=config.page_tokens,
        recent_tokens=0,
        max_input_tokens=config.max_context_tokens,
        summary10_words=10,
        summary100_words=100,
        summary1000_words=900,
        max_new_tokens_exact=config.max_new_tokens_exact,
        max_new_tokens_summary=config.max_new_tokens_summary,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        device_map="cuda",
        cuda_visible_devices="",
        router_path="",
        seed=config.seed,
    )


def method_enabled(config: Config, method: str) -> bool:
    methods = set(config.runtime_methods)
    return not methods or "all" in methods or method in methods


def any_method_enabled(config: Config, methods: tuple[str, ...]) -> bool:
    return any(method_enabled(config, method) for method in methods)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def cache_from_legacy(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


def clone_cache(cache: Any) -> Any:
    return cache_from_legacy(tuple((key.clone(), value.clone()) for key, value in legacy_cache(cache)))


def cache_len(cache: Any) -> int:
    return int(legacy_cache(cache)[0][0].shape[2])


def words(text: str) -> set[str]:
    terms = re.findall(r"[A-Za-z0-9_\-]{3,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "what", "which", "answer", "question"}
    return {term for term in terms if term not in stop}


def query_codes(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]+[-_][A-Za-z0-9_\-]+|\d+", text))


def query_text(case: BenchCase) -> str:
    if case.benchmark.startswith("ruler"):
        return case.query
    template = LONG_BENCH_PROMPTS.get(case.task, "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
    return template.format(context="", input=case.query)


def tokenize_context(tokenizer: Any, case: BenchCase, max_tokens: int) -> list[int]:
    ids = tokenizer(case.context, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


def page_text(tokenizer: Any, context_ids: list[int], page: int, page_tokens: int) -> str:
    return tokenizer.decode(context_ids[page * page_tokens : min(len(context_ids), (page + 1) * page_tokens)], skip_special_tokens=True)


def lexical_page_scores(
    tokenizer: Any,
    context_ids: list[int],
    query: str,
    page_tokens: int,
) -> tuple[list[float], float]:
    query_words = words(query)
    codes = query_codes(query)
    scores: list[float] = []
    for start in range(0, len(context_ids), page_tokens):
        text = tokenizer.decode(context_ids[start : start + page_tokens], skip_special_tokens=True)
        score = float(len(query_words & words(text)))
        score += 4.0 * len(codes & query_codes(text))
        scores.append(score)
    denom = max(1.0, float(len(query_words) + 4 * len(codes)))
    return scores, denom


def top_pages_from_scores(scores: list[float], top_k: int) -> list[int]:
    scored: list[tuple[float, int]] = []
    for page, score in enumerate(scores):
        scored.append((float(score), page))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [page for _, page in scored[:top_k]]
    return sorted(selected)


def lexical_pages(tokenizer: Any, context_ids: list[int], query: str, page_tokens: int, top_k: int) -> list[int]:
    scores, _ = lexical_page_scores(tokenizer, context_ids, query, page_tokens)
    return top_pages_from_scores(scores, top_k)


def runtime_text_features(query: str, scores: list[float], denom: float, pages: list[int]) -> list[float]:
    if not scores:
        return [0.0] * len(TEXT_FEATURE_NAMES)
    sorted_scores = sorted(scores, reverse=True)
    top1 = sorted_scores[0] if sorted_scores else 0.0
    top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    top3 = sorted_scores[2] if len(sorted_scores) > 2 else 0.0
    selected = [scores[page] for page in pages if 0 <= page < len(scores)]
    if selected:
        mean = sum(selected) / len(selected)
        var = sum((score - mean) ** 2 for score in selected) / len(selected)
        smin = min(selected)
        smax = max(selected)
        ssum = sum(selected)
        sstd = math.sqrt(var)
    else:
        mean = smin = smax = ssum = sstd = 0.0
    best_page = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
    codes = query_codes(query)
    return [
        math.log1p(len(content_words(query))),
        math.log1p(len(codes)),
        top1,
        top2,
        top3,
        top1 - top2,
        top2 - top3,
        math.log1p(sum(1 for score in scores if score > 0)),
        top1 / denom,
        (top1 - top2) / denom,
        mean,
        smin,
        smax,
        ssum,
        1.0 if best_page in pages else 0.0,
        sstd,
    ]


def page_layout_features(context_tokens: int, page_tokens: int, pages: list[int]) -> list[float]:
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    denom = max(1.0, float(total_pages - 1))
    if not pages:
        return [0.0] * 9
    sorted_pages = sorted(pages)
    gaps = [b - a for a, b in zip(sorted_pages, sorted_pages[1:])]
    return [
        float(len(sorted_pages)),
        float(sorted_pages[0]) / denom,
        float(sorted_pages[-1]) / denom,
        sum(sorted_pages) / len(sorted_pages) / denom,
        float(sorted_pages[-1] - sorted_pages[0]) / denom,
        float(max(gaps)) / denom if gaps else 0.0,
        1.0 if 0 in sorted_pages else 0.0,
        1.0 if gaps and all(gap == 1 for gap in gaps) else (1.0 if len(sorted_pages) <= 1 else 0.0),
        float(len(sorted_pages) * page_tokens) / max(1.0, float(context_tokens)),
    ]


def compact_action_budget(action: str) -> int:
    if action == "full":
        return 10**9
    if action.startswith("k") and action.endswith("_compact"):
        return int(action[1 : action.index("_")])
    raise ValueError(action)


def apply_variable_budget_floor(action: str, available_budgets: list[int], min_budget: int) -> str:
    floor = max(0, int(min_budget))
    if floor <= 0 or action == "full":
        return action
    selected_budget = compact_action_budget(action)
    if selected_budget >= floor:
        return action
    eligible_budgets = [budget for budget in sorted(available_budgets) if budget >= floor]
    if not eligible_budgets:
        return action
    return f"k{eligible_budgets[0]}_compact"


class RuntimeTwoStagePlanner:
    def __init__(self, checkpoint_path: str, threshold_full: float, threshold_k3: float) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.feature_names = list(payload["feature_names"])
        self.mean = [float(item) for item in payload["mean"]]
        self.std = [float(item) for item in payload["std"]]
        self.label_to_id = {str(key): int(value) for key, value in payload["label_to_id"].items()}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.threshold_full = threshold_full
        self.threshold_k3 = threshold_k3
        self.model = MLP(int(payload["input_dim"]), int(payload["hidden_dim"]), len(self.label_to_id))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def build_feature_dict(
        self,
        *,
        benchmark: str,
        task: str,
        context_tokens: int,
        query_tokens: int,
        page_tokens: int,
        query: str,
        scores: list[float],
        denom: float,
        k2_pages: list[int],
        k3_pages: list[int],
    ) -> dict[str, float]:
        k2_set = set(k2_pages)
        k3_set = set(k3_pages)
        union = k2_set | k3_set
        values: dict[str, float] = {
            "is_longbench": 1.0 if benchmark == "longbench" else 0.0,
            "is_ruler": 1.0 if benchmark.startswith("ruler") else 0.0,
            "context_tokens_log": math.log1p(context_tokens),
            "query_tokens_log": math.log1p(query_tokens),
            "page_tokens_log": math.log1p(page_tokens),
            "k2_subset_k3": 1.0 if k2_set and k2_set.issubset(k3_set) else 0.0,
            "page_jaccard": float(len(k2_set & k3_set)) / max(1.0, float(len(union))),
            "k3_added_pages": float(len(k3_set - k2_set)),
            "k3_added_page0": 1.0 if 0 in (k3_set - k2_set) else 0.0,
            f"task={task}": 1.0,
            f"benchmark={benchmark}": 1.0,
        }
        layout_names = [
            "pages",
            "first_page_norm",
            "last_page_norm",
            "mean_page_norm",
            "span_width_norm",
            "max_gap_norm",
            "has_page0",
            "all_pages_adjacent",
            "selected_kv_ratio",
        ]
        for prefix, pages in (("k2", k2_pages), ("k3", k3_pages)):
            for name, value in zip(layout_names, page_layout_features(context_tokens, page_tokens, pages)):
                values[f"{prefix}_{name}"] = float(value)
            for name, value in zip(TEXT_FEATURE_NAMES, runtime_text_features(query, scores, denom, pages)):
                values[f"{prefix}_{name}"] = float(value)
        return values

    def predict(
        self,
        *,
        benchmark: str,
        task: str,
        context_tokens: int,
        query_tokens: int,
        page_tokens: int,
        query: str,
        scores: list[float],
        denom: float,
        k2_pages: list[int],
        k3_pages: list[int],
    ) -> tuple[str, dict[str, float]]:
        values = self.build_feature_dict(
            benchmark=benchmark,
            task=task,
            context_tokens=context_tokens,
            query_tokens=query_tokens,
            page_tokens=page_tokens,
            query=query,
            scores=scores,
            denom=denom,
            k2_pages=k2_pages,
            k3_pages=k3_pages,
        )
        features = [values.get(name, 0.0) for name in self.feature_names]
        normed = [(value - self.mean[idx]) / max(self.std[idx], 1e-6) for idx, value in enumerate(features)]
        x = torch.tensor([normed], dtype=torch.float32)
        with torch.inference_mode():
            probs_tensor = torch.softmax(self.model(x), dim=-1)[0]
        probs = {label: float(probs_tensor[idx]) for idx, label in self.id_to_label.items()}
        if probs.get("full", 0.0) >= self.threshold_full:
            return "full", probs
        if probs.get("k3_compact", 0.0) >= self.threshold_k3:
            return "k3_compact", probs
        return "k2_compact", probs


class RuntimeVariableBudgetPlanner:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        policy: str,
        tail_threshold: float,
        temperature: float,
        source_name: str,
        max_examples_per_task: int,
    ) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.feature_names = list(payload["feature_names"])
        self.mean = [float(item) for item in payload["mean"]]
        self.std = [float(item) for item in payload["std"]]
        self.label_to_id = {str(key): int(value) for key, value in payload["label_to_id"].items()}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.policy = policy
        checkpoint_tau = payload.get("selected_tau")
        if tail_threshold < 0.0 and checkpoint_tau is not None:
            self.tail_threshold = float(checkpoint_tau)
        else:
            self.tail_threshold = float(tail_threshold)
        self.temperature = max(1e-6, float(temperature))
        self.source_name = source_name
        self.max_examples_per_task = int(max_examples_per_task)
        self.source_features = [name.removeprefix("source=") for name in self.feature_names if name.startswith("source=")]
        self.budgets = sorted({compact_action_budget(label) for label in self.label_to_id if label != "full"})
        cfg = payload.get("config", {})
        self.use_text_features = bool(cfg.get("use_text_features", False))
        self.model = MLP(int(payload["input_dim"]), int(payload["hidden_dim"]), len(self.label_to_id))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def resolve_source(self, benchmark: str) -> str:
        if self.source_name != "auto":
            return self.source_name
        if not self.source_features:
            return ""
        if len(self.source_features) == 1:
            return self.source_features[0]
        if benchmark.startswith("ruler"):
            return self.source_features[0]
        if benchmark == "longbench" and self.max_examples_per_task > 4:
            return self.source_features[-1]
        return self.source_features[0]

    def build_feature_dict(
        self,
        *,
        benchmark: str,
        task: str,
        context_tokens: int,
        query_tokens: int,
        page_tokens: int,
        query: str,
        scores: list[float],
        denom: float,
        pages_by_budget: dict[int, list[int]],
    ) -> dict[str, float]:
        available_budgets = sorted(pages_by_budget)
        values: dict[str, float] = {
            "is_longbench": 1.0 if benchmark == "longbench" else 0.0,
            "is_ruler": 1.0 if benchmark.startswith("ruler") else 0.0,
            "context_tokens_log": math.log1p(context_tokens),
            "query_tokens_log": math.log1p(query_tokens),
            "page_tokens_log": math.log1p(page_tokens),
            "num_budgets": float(len(available_budgets)),
            "min_budget": float(min(available_budgets)) if available_budgets else 0.0,
            "max_budget": float(max(available_budgets)) if available_budgets else 0.0,
            f"task={task}": 1.0,
            f"benchmark={benchmark}": 1.0,
        }
        source = self.resolve_source(benchmark)
        if source:
            values[f"source={source}"] = 1.0
        layout_names = [
            "pages",
            "first_page_norm",
            "last_page_norm",
            "mean_page_norm",
            "span_width_norm",
            "max_gap_norm",
            "has_page0",
            "all_pages_adjacent",
            "selected_kv_ratio",
        ]
        page_sets: dict[int, set[int]] = {}
        for top_k in self.budgets:
            pages = pages_by_budget.get(top_k, [])
            page_sets[top_k] = set(pages)
            values[f"k{top_k}_available"] = 1.0 if top_k in pages_by_budget else 0.0
            for name, value in zip(layout_names, page_layout_features(context_tokens, page_tokens, pages)):
                values[f"k{top_k}_{name}"] = float(value)
            text_values = (
                runtime_text_features(query, scores, denom, pages)
                if self.use_text_features
                else [0.0] * len(TEXT_FEATURE_NAMES)
            )
            for name, value in zip(TEXT_FEATURE_NAMES, text_values):
                values[f"k{top_k}_{name}"] = float(value)
        for left, right in zip(self.budgets, self.budgets[1:]):
            left_set = page_sets.get(left, set())
            right_set = page_sets.get(right, set())
            union = left_set | right_set
            values[f"delta_k{left}_to_k{right}_page_jaccard"] = (
                float(len(left_set & right_set)) / max(1.0, float(len(union)))
            )
            values[f"delta_k{left}_to_k{right}_added_pages"] = float(len(right_set - left_set))
        return values

    def choose_tail_risk(self, probs: dict[str, float]) -> str:
        labels = sorted(self.label_to_id, key=lambda label: (compact_action_budget(label), label))
        for label in labels:
            tail = sum(
                probs.get(other, 0.0)
                for other in labels
                if compact_action_budget(other) > compact_action_budget(label)
            )
            if tail <= self.tail_threshold:
                return label
        return labels[-1]

    def predict(
        self,
        *,
        benchmark: str,
        task: str,
        context_tokens: int,
        query_tokens: int,
        page_tokens: int,
        query: str,
        scores: list[float],
        denom: float,
        pages_by_budget: dict[int, list[int]],
    ) -> tuple[str, dict[str, float]]:
        values = self.build_feature_dict(
            benchmark=benchmark,
            task=task,
            context_tokens=context_tokens,
            query_tokens=query_tokens,
            page_tokens=page_tokens,
            query=query,
            scores=scores,
            denom=denom,
            pages_by_budget=pages_by_budget,
        )
        features = [values.get(name, 0.0) for name in self.feature_names]
        normed = [(value - self.mean[idx]) / max(self.std[idx], 1e-6) for idx, value in enumerate(features)]
        x = torch.tensor([normed], dtype=torch.float32)
        with torch.inference_mode():
            probs_tensor = torch.softmax(self.model(x) / self.temperature, dim=-1)[0]
        probs = {label: float(probs_tensor[idx]) for idx, label in self.id_to_label.items()}
        if self.policy == "argmax":
            return max(probs, key=lambda label: (probs[label], -compact_action_budget(label))), probs
        return self.choose_tail_risk(probs), probs


class RuntimeOutputLevelRiskVerifier:
    OUTPUT_FEATURE_NAMES = [
        "prediction_chars_log",
        "prediction_words_log",
        "prediction_unique_word_ratio",
        "prediction_repeated_bigram_ratio",
        "prediction_contains_passage",
        "prediction_contains_question",
        "prediction_contains_answer",
        "prediction_contains_only_give",
        "prediction_ends_question_mark",
        "prediction_same_as_smaller_budget",
        "prediction_same_as_larger_budget",
        "prediction_same_as_any_budget",
    ]

    def __init__(
        self,
        checkpoint_path: str,
        *,
        threshold: float,
        source_name: str,
        max_examples_per_task: int,
        budget_override: tuple[int, ...],
    ) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.feature_names = list(payload["feature_names"])
        self.mean = [float(item) for item in payload["mean"]]
        self.std = [float(item) for item in payload["std"]]
        self.compact_actions = [str(item) for item in payload["compact_actions"]]
        if budget_override:
            allowed = {f"k{budget}_compact" for budget in budget_override}
            self.compact_actions = [action for action in self.compact_actions if action in allowed]
        self.budgets = sorted({compact_action_budget(action) for action in self.compact_actions})
        self.threshold = float(threshold)
        self.source_name = source_name
        self.max_examples_per_task = int(max_examples_per_task)
        self.source_features = [name.removeprefix("source=") for name in self.feature_names if name.startswith("source=")]
        cfg = payload.get("config", {})
        self.use_text_features = bool(cfg.get("use_text_features", False))
        self.model = MLP(int(payload["input_dim"]), int(payload["hidden_dim"]), 2)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def resolve_source(self, benchmark: str) -> str:
        if self.source_name != "auto":
            return self.source_name
        if not self.source_features:
            return ""
        if len(self.source_features) == 1:
            return self.source_features[0]
        if benchmark.startswith("ruler"):
            return self.source_features[0]
        if benchmark == "longbench" and self.max_examples_per_task > 4:
            return self.source_features[-1]
        return self.source_features[0]

    def build_base_values(
        self,
        *,
        benchmark: str,
        task: str,
        context_tokens: int,
        query_tokens: int,
        page_tokens: int,
        query: str,
        scores: list[float],
        denom: float,
        pages_by_budget: dict[int, list[int]],
    ) -> dict[str, float]:
        available_budgets = sorted(pages_by_budget)
        values: dict[str, float] = {
            "is_longbench": 1.0 if benchmark == "longbench" else 0.0,
            "is_ruler": 1.0 if benchmark.startswith("ruler") else 0.0,
            "context_tokens_log": math.log1p(context_tokens),
            "query_tokens_log": math.log1p(query_tokens),
            "page_tokens_log": math.log1p(page_tokens),
            "num_budgets": float(len(available_budgets)),
            "min_budget": float(min(available_budgets)) if available_budgets else 0.0,
            "max_budget": float(max(available_budgets)) if available_budgets else 0.0,
            f"task={task}": 1.0,
            f"benchmark={benchmark}": 1.0,
        }
        source = self.resolve_source(benchmark)
        if source:
            values[f"source={source}"] = 1.0
        layout_names = [
            "pages",
            "first_page_norm",
            "last_page_norm",
            "mean_page_norm",
            "span_width_norm",
            "max_gap_norm",
            "has_page0",
            "all_pages_adjacent",
            "selected_kv_ratio",
        ]
        page_sets: dict[int, set[int]] = {}
        for top_k in self.budgets:
            pages = pages_by_budget.get(top_k, [])
            page_sets[top_k] = set(pages)
            values[f"k{top_k}_available"] = 1.0 if top_k in pages_by_budget else 0.0
            for name, value in zip(layout_names, page_layout_features(context_tokens, page_tokens, pages)):
                values[f"k{top_k}_{name}"] = float(value)
            text_values = (
                runtime_text_features(query, scores, denom, pages)
                if self.use_text_features
                else [0.0] * len(TEXT_FEATURE_NAMES)
            )
            for name, value in zip(TEXT_FEATURE_NAMES, text_values):
                values[f"k{top_k}_{name}"] = float(value)
        for left, right in zip(self.budgets, self.budgets[1:]):
            left_set = page_sets.get(left, set())
            right_set = page_sets.get(right, set())
            union = left_set | right_set
            values[f"delta_k{left}_to_k{right}_page_jaccard"] = (
                float(len(left_set & right_set)) / max(1.0, float(len(union)))
            )
            values[f"delta_k{left}_to_k{right}_added_pages"] = float(len(right_set - left_set))
        return values

    def action_values(
        self,
        *,
        base_values: dict[str, float],
        action: str,
        rank: int,
        active_kv_ratio: float,
        predictions: dict[str, str],
    ) -> dict[str, float]:
        values = dict(base_values)
        budget = compact_action_budget(action)
        values.update(
            {
                "candidate_budget_log": math.log1p(budget),
                "candidate_budget_rank": float(rank) / max(1.0, float(len(self.compact_actions) - 1)),
                "candidate_kv_ratio": active_kv_ratio,
                "candidate_is_k1": 1.0 if budget == 1 else 0.0,
                "candidate_is_k2": 1.0 if budget == 2 else 0.0,
                "candidate_is_k3": 1.0 if budget == 3 else 0.0,
                "candidate_is_k4": 1.0 if budget == 4 else 0.0,
                "candidate_is_k6": 1.0 if budget == 6 else 0.0,
                "candidate_is_k8": 1.0 if budget == 8 else 0.0,
            }
        )
        for candidate in self.compact_actions:
            values[f"candidate={candidate}"] = 1.0 if candidate == action else 0.0
        for name, value in zip(self.OUTPUT_FEATURE_NAMES, output_features(predictions.get(action, ""), predictions, action)):
            values[name] = float(value)
        return values

    def safe_probability(
        self,
        *,
        base_values: dict[str, float],
        action: str,
        rank: int,
        active_kv_ratio: float,
        predictions: dict[str, str],
    ) -> float:
        values = self.action_values(
            base_values=base_values,
            action=action,
            rank=rank,
            active_kv_ratio=active_kv_ratio,
            predictions=predictions,
        )
        features = [values.get(name, 0.0) for name in self.feature_names]
        normed = [(value - self.mean[idx]) / max(self.std[idx], 1e-6) for idx, value in enumerate(features)]
        x = torch.tensor([normed], dtype=torch.float32)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=-1)[0]
        return float(probs[1])


def page_indices(pages: list[int], context_len: int, page_tokens: int, device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for page in pages:
        start = page * page_tokens
        end = min(context_len, start + page_tokens)
        indices.extend(range(start, end))
    if not indices:
        raise ValueError("empty page selection")
    return torch.tensor(indices, dtype=torch.long, device=device)


def selected_context_text(tokenizer: Any, context_ids: list[int], pages: list[int], page_tokens: int) -> str:
    return "\n\n".join(page_text(tokenizer, context_ids, page, page_tokens) for page in pages)


def gather_cache(cache: Any, indices: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    gathered = []
    for key, value in legacy_cache(cache):
        gathered.append((key.index_select(2, indices), value.index_select(2, indices)))
    return tuple(gathered)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary_inv_freq(model: Any, head_dim: int, device: torch.device) -> torch.Tensor:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    if inv_freq is not None:
        return inv_freq.to(device=device, dtype=torch.float32)
    theta = float(getattr(model.config, "rope_theta", 10000.0))
    rotary_dim = int(getattr(model.config, "head_dim", head_dim))
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))


def apply_rope_delta_to_key(
    key: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    rot_dim = min(key.shape[-1], int(inv_freq.numel() * 2))
    if rot_dim <= 0:
        return key
    key_rot = key[..., :rot_dim]
    key_pass = key[..., rot_dim:]
    delta = (new_positions.to(torch.float32) - old_positions.to(torch.float32)).to(inv_freq.device)
    freqs = torch.outer(delta, inv_freq[: rot_dim // 2])
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=key.dtype).view(1, 1, -1, rot_dim)
    sin = emb.sin().to(dtype=key.dtype).view(1, 1, -1, rot_dim)
    repacked = key_rot * cos + rotate_half(key_rot) * sin
    return torch.cat((repacked, key_pass), dim=-1) if key_pass.numel() else repacked


def gather_and_rope_repack_cache(
    model: Any,
    cache: Any,
    indices: torch.Tensor,
    new_positions: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    legacy = legacy_cache(cache)
    head_dim = int(legacy[0][0].shape[-1])
    inv_freq = rotary_inv_freq(model, head_dim, indices.device)
    repacked = []
    for key, value in legacy:
        gathered_key = key.index_select(2, indices)
        gathered_value = value.index_select(2, indices)
        repacked_key = apply_rope_delta_to_key(gathered_key, indices, new_positions, inv_freq)
        repacked.append((repacked_key, gathered_value))
    return tuple(repacked)


def prefill(model: Any, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor, float]:
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    synchronize()
    return out.past_key_values, out.logits[:, -1, :], time.perf_counter() - start


def run_query_on_cache(
    model: Any,
    query_ids: torch.Tensor,
    past_key_values: Any,
    position_start: int,
    past_len: int,
) -> tuple[Any, torch.Tensor, float]:
    device = query_ids.device
    q_len = int(query_ids.shape[1])
    attention_mask = torch.ones((1, past_len + q_len), dtype=torch.long, device=device)
    position_ids = torch.arange(position_start, position_start + q_len, device=device).view(1, -1)
    cache_position = torch.arange(position_start, position_start + q_len, device=device)
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        try:
            out = model(
                input_ids=query_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
        except TypeError:
            out = model(
                input_ids=query_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
    synchronize()
    return out.past_key_values, out.logits[:, -1, :], time.perf_counter() - start


def greedy_decode(model: Any, tokenizer: Any, logits: torch.Tensor, past_key_values: Any, steps: int) -> tuple[str, float]:
    if steps <= 0:
        return "", 0.0
    device = logits.device
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated: list[int] = []
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            generated.append(int(next_token.item()))
            out = model(input_ids=next_token.to(device), past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).to(device)
    synchronize()
    return tokenizer.decode(generated, skip_special_tokens=True), time.perf_counter() - start


def answer_nll(model: Any, tokenizer: Any, answers: tuple[str, ...], logits: torch.Tensor, past_key_values: Any) -> float:
    if not answers:
        return 0.0
    answer_ids = tokenizer(answers[0], add_special_tokens=False, return_tensors="pt").input_ids.to(logits.device)
    if answer_ids.numel() == 0:
        return 0.0
    total = 0.0
    count = 0
    current_logits = logits
    cache = past_key_values
    with torch.inference_mode():
        for idx in range(int(answer_ids.shape[1])):
            target = answer_ids[:, idx]
            log_probs = torch.log_softmax(current_logits, dim=-1)
            total -= float(log_probs.gather(1, target.view(1, 1)).item())
            count += 1
            out = model(input_ids=target.view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            current_logits = out.logits[:, -1, :]
    return total / max(1, count)


def candidate_nll(model: Any, tokenizer: Any, candidate: str, logits: torch.Tensor, past_key_values: Any) -> float:
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(logits.device)
    if candidate_ids.numel() == 0:
        return float("inf")
    total = 0.0
    count = 0
    current_logits = logits
    cache = past_key_values
    with torch.inference_mode():
        for idx in range(int(candidate_ids.shape[1])):
            target = candidate_ids[:, idx]
            log_probs = torch.log_softmax(current_logits, dim=-1)
            total -= float(log_probs.gather(1, target.view(1, 1)).item())
            count += 1
            out = model(input_ids=target.view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            current_logits = out.logits[:, -1, :]
    return total / max(1, count)


def logit_probe_metrics(full_logits: torch.Tensor, compact_logits: torch.Tensor) -> dict[str, float]:
    full = full_logits.float()
    compact = compact_logits.float()
    full_log_probs = torch.log_softmax(full, dim=-1)
    compact_log_probs = torch.log_softmax(compact, dim=-1)
    full_probs = full_log_probs.exp()
    kl = torch.sum(full_probs * (full_log_probs - compact_log_probs), dim=-1)
    full_top1 = torch.argmax(full, dim=-1)
    compact_top1 = torch.argmax(compact, dim=-1)
    top1_agree = float(torch.eq(full_top1, compact_top1).float().mean().item())
    top5_full = torch.topk(full, k=min(5, full.shape[-1]), dim=-1).indices[0].tolist()
    top5_compact = set(torch.topk(compact, k=min(5, compact.shape[-1]), dim=-1).indices[0].tolist())
    top5_overlap = len([idx for idx in top5_full if idx in top5_compact]) / max(1, len(top5_full))
    return {
        "kl_full_to_compact": float(kl.mean().item()),
        "top1_agree": top1_agree,
        "top5_overlap": float(top5_overlap),
        "full_top1": float(int(full_top1[0].item())),
        "compact_top1": float(int(compact_top1[0].item())),
    }


def logit_probe_passes(metrics: dict[str, float], config: Config) -> bool:
    if metrics["kl_full_to_compact"] > config.consistency_probe_kl_threshold:
        return False
    if config.consistency_probe_require_top1_agree and metrics["top1_agree"] < 1.0:
        return False
    return True


def score_prediction(case: BenchCase, prediction: str) -> tuple[int, float, float]:
    exact = exact_match_any(prediction, case.answers)
    rouge = rouge_l_f1(prediction, case.answers)
    score = rouge if case.task in SUMMARY_TASKS else float(exact)
    return exact, rouge, score


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_row(
    *,
    case: BenchCase,
    method: str,
    context_tokens: int,
    active_kv_tokens: int,
    query_tokens: int,
    selected_pages_value: str,
    planner_action: str = "",
    planner_seconds: float = 0.0,
    prefill_seconds: float = 0.0,
    full_prefill_seconds: float = 0.0,
    gather_seconds: float = 0.0,
    repack_seconds: float = 0.0,
    query_seconds: float = 0.0,
    decode_seconds: float = 0.0,
    full_online_seconds: float = 0.0,
    prediction: str = "",
    nll: float = 0.0,
) -> ResultRow:
    exact, rouge, score = score_prediction(case, prediction)
    total = planner_seconds + gather_seconds + repack_seconds + query_seconds + decode_seconds
    if method.startswith("prompt"):
        total = prefill_seconds + decode_seconds
    full_e2e = full_prefill_seconds + full_online_seconds
    if method.startswith("prompt"):
        end_to_end = total
    else:
        end_to_end = full_prefill_seconds + total

    def amortized_speedup(num_queries: int) -> float:
        full_total = full_prefill_seconds + num_queries * full_online_seconds
        if method.startswith("prompt"):
            method_total = num_queries * total
        else:
            method_total = full_prefill_seconds + num_queries * total
        return full_total / method_total if method_total > 0 else 0.0

    return ResultRow(
        benchmark=case.benchmark,
        task=case.task,
        case_id=case.case_id,
        method=method,
        context_tokens=context_tokens,
        active_kv_tokens=active_kv_tokens,
        query_tokens=query_tokens,
        selected_pages=selected_pages_value,
        planner_action=planner_action,
        planner_seconds=planner_seconds,
        prefill_seconds=prefill_seconds,
        full_prefill_seconds=full_prefill_seconds,
        gather_seconds=gather_seconds,
        repack_seconds=repack_seconds,
        query_seconds=query_seconds,
        decode_seconds=decode_seconds,
        total_online_seconds=total,
        speedup_vs_full_online=full_online_seconds / total if total > 0 else 0.0,
        end_to_end_seconds=end_to_end,
        speedup_vs_full_e2e=full_e2e / end_to_end if end_to_end > 0 else 0.0,
        amortized4_speedup_vs_full_e2e=amortized_speedup(4),
        amortized16_speedup_vs_full_e2e=amortized_speedup(16),
        prediction=prediction.replace("\n", " ")[:500],
        answers=json.dumps(case.answers, ensure_ascii=False),
        exact_correct=exact,
        rouge_l=rouge,
        score=score,
        answer_nll=nll,
    )


def run_cache_method(
    *,
    rows: list[ResultRow],
    case: BenchCase,
    method: str,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    cache: Any,
    context_tokens: int,
    active_kv_tokens: int,
    selected_pages_value: str,
    planner_action: str = "",
    planner_seconds: float = 0.0,
    position_start: int = 0,
    prefill_seconds: float = 0.0,
    full_prefill_seconds: float = 0.0,
    gather_seconds: float = 0.0,
    repack_seconds: float = 0.0,
    decode_steps: int = 0,
    full_online_seconds: float = 0.0,
) -> None:
    q_cache, logits, query_seconds = run_query_on_cache(
        model, query_ids, cache, position_start=position_start, past_len=active_kv_tokens
    )
    prediction, decode_seconds = greedy_decode(model, tokenizer, logits, q_cache, decode_steps)
    nll = answer_nll(model, tokenizer, case.answers, logits, q_cache)
    rows.append(
        make_row(
            case=case,
            method=method,
            context_tokens=context_tokens,
            active_kv_tokens=active_kv_tokens,
            query_tokens=int(query_ids.shape[1]),
            selected_pages_value=selected_pages_value,
            planner_action=planner_action,
            planner_seconds=planner_seconds,
            prefill_seconds=prefill_seconds,
            full_prefill_seconds=full_prefill_seconds,
            gather_seconds=gather_seconds,
            repack_seconds=repack_seconds,
            query_seconds=query_seconds,
            decode_seconds=decode_seconds,
            full_online_seconds=full_online_seconds,
            prediction=prediction,
            nll=nll,
        )
    )


def run_consistency_probe_method(
    *,
    rows: list[ResultRow],
    case: BenchCase,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    full_cache: Any,
    full_logits: torch.Tensor,
    full_query_seconds: float,
    full_decode: float,
    full_prediction: str,
    full_nll: float,
    full_prefill: float,
    full_online: float,
    context_tokens: int,
    page_tokens: int,
    page_scores: list[float],
    lexical_planner_seconds: float,
    decode_steps: int,
    config: Config,
) -> None:
    if not config.consistency_probe_budgets:
        return
    device = query_ids.device
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    probe_log: list[dict[str, Any]] = []
    accumulated_probe_seconds = lexical_planner_seconds + full_query_seconds

    for budget in sorted(set(config.consistency_probe_budgets)):
        if budget >= total_pages:
            break
        selected_pages = top_pages_from_scores(page_scores, budget)
        selected_indices = page_indices(selected_pages, context_tokens, page_tokens, device)
        synchronize()
        start = time.perf_counter()
        selected_positions = torch.arange(int(selected_indices.numel()), dtype=torch.long, device=device)
        compact_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, selected_indices, selected_positions))
        synchronize()
        repack_seconds = time.perf_counter() - start
        active_tokens = cache_len(compact_cache)
        q_cache, compact_logits, query_seconds = run_query_on_cache(
            model,
            query_ids,
            compact_cache,
            position_start=active_tokens,
            past_len=active_tokens,
        )
        metrics = logit_probe_metrics(full_logits, compact_logits)
        passed = logit_probe_passes(metrics, config)
        probe_record = {
            "action": f"k{budget}_compact",
            "budget": budget,
            "pages": selected_pages,
            "active_kv_tokens": active_tokens,
            "active_kv_ratio_vs_full": active_tokens / max(1, context_tokens),
            "repack_seconds": repack_seconds,
            "query_seconds": query_seconds,
            "passed": passed,
            **metrics,
        }
        probe_log.append(probe_record)
        if passed:
            prediction, decode_seconds = greedy_decode(model, tokenizer, compact_logits, q_cache, decode_steps)
            nll = answer_nll(model, tokenizer, case.answers, compact_logits, q_cache)
            rows.append(
                make_row(
                    case=case,
                    method="consistency_probe_kv_planner",
                    context_tokens=context_tokens,
                    active_kv_tokens=active_tokens,
                    query_tokens=int(query_ids.shape[1]),
                    selected_pages_value=json.dumps(
                        {
                            "action": f"k{budget}_compact",
                            "threshold": config.consistency_probe_kl_threshold,
                            "require_top1_agree": config.consistency_probe_require_top1_agree,
                            "probes": probe_log,
                        },
                        ensure_ascii=False,
                    ),
                    planner_action=f"k{budget}_compact",
                    planner_seconds=accumulated_probe_seconds,
                    prefill_seconds=0.0,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=0.0,
                    repack_seconds=repack_seconds,
                    query_seconds=query_seconds,
                    decode_seconds=decode_seconds,
                    full_online_seconds=full_online,
                    prediction=prediction,
                    nll=nll,
                )
            )
            return
        accumulated_probe_seconds += repack_seconds + query_seconds
        del compact_cache, q_cache, compact_logits

    rows.append(
        make_row(
            case=case,
            method="consistency_probe_kv_planner",
            context_tokens=context_tokens,
            active_kv_tokens=context_tokens,
            query_tokens=int(query_ids.shape[1]),
            selected_pages_value=json.dumps(
                {
                    "action": "full",
                    "threshold": config.consistency_probe_kl_threshold,
                    "require_top1_agree": config.consistency_probe_require_top1_agree,
                    "probes": probe_log,
                },
                ensure_ascii=False,
            ),
            planner_action="full",
            planner_seconds=accumulated_probe_seconds,
            prefill_seconds=0.0,
            full_prefill_seconds=full_prefill,
            gather_seconds=0.0,
            repack_seconds=0.0,
            query_seconds=0.0,
            decode_seconds=full_decode,
            full_online_seconds=full_online,
            prediction=full_prediction,
            nll=full_nll,
        )
    )


def run_teacher_verifier_method(
    *,
    rows: list[ResultRow],
    case: BenchCase,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    full_cache: Any,
    full_decode: float,
    full_prediction: str,
    full_nll: float,
    full_prefill: float,
    full_online: float,
    context_tokens: int,
    page_tokens: int,
    page_scores: list[float],
    lexical_planner_seconds: float,
    decode_steps: int,
    config: Config,
) -> None:
    if not config.teacher_verifier_budgets:
        return
    device = query_ids.device
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    synchronize()
    teacher_cache, teacher_logits, teacher_query_seconds = run_query_on_cache(
        model,
        query_ids,
        full_cache,
        position_start=context_tokens,
        past_len=context_tokens,
    )
    candidates: list[dict[str, Any]] = []
    method_seconds = lexical_planner_seconds + teacher_query_seconds

    for budget in sorted(set(config.teacher_verifier_budgets)):
        if budget >= total_pages:
            continue
        selected_pages = top_pages_from_scores(page_scores, budget)
        selected_indices = page_indices(selected_pages, context_tokens, page_tokens, device)
        synchronize()
        start = time.perf_counter()
        selected_positions = torch.arange(int(selected_indices.numel()), dtype=torch.long, device=device)
        compact_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, selected_indices, selected_positions))
        synchronize()
        repack_seconds = time.perf_counter() - start
        active_tokens = cache_len(compact_cache)
        q_cache, compact_logits, query_seconds = run_query_on_cache(
            model,
            query_ids,
            compact_cache,
            position_start=active_tokens,
            past_len=active_tokens,
        )
        prediction, decode_seconds = greedy_decode(model, tokenizer, compact_logits, q_cache, decode_steps)
        synchronize()
        score_start = time.perf_counter()
        teacher_nll = candidate_nll(model, tokenizer, prediction, teacher_logits, clone_cache(teacher_cache))
        synchronize()
        teacher_score_seconds = time.perf_counter() - score_start
        candidate = {
            "action": f"k{budget}_compact",
            "budget": budget,
            "pages": selected_pages,
            "active_kv_tokens": active_tokens,
            "active_kv_ratio_vs_full": active_tokens / max(1, context_tokens),
            "prediction": prediction.replace("\n", " ")[:200],
            "teacher_nll": teacher_nll,
            "repack_seconds": repack_seconds,
            "query_seconds": query_seconds,
            "decode_seconds": decode_seconds,
            "teacher_score_seconds": teacher_score_seconds,
        }
        candidates.append(candidate)
        method_seconds += repack_seconds + query_seconds + decode_seconds + teacher_score_seconds
        del compact_cache, q_cache, compact_logits

    if not candidates:
        selected: dict[str, Any] = {
            "action": "full",
            "teacher_nll": full_nll,
            "fallback_reason": "no_compact_candidates",
        }
        rows.append(
            make_row(
                case=case,
                method="teacher_likelihood_kv_planner",
                context_tokens=context_tokens,
                active_kv_tokens=context_tokens,
                query_tokens=int(query_ids.shape[1]),
                selected_pages_value=json.dumps({"selected": selected, "candidates": candidates}, ensure_ascii=False),
                planner_action="full",
                planner_seconds=method_seconds,
                prefill_seconds=0.0,
                full_prefill_seconds=full_prefill,
                decode_seconds=full_decode,
                full_online_seconds=full_online,
                prediction=full_prediction,
                nll=full_nll,
            )
        )
        return

    best = min(
        candidates,
        key=lambda item: (
            item["teacher_nll"],
            item["active_kv_ratio_vs_full"],
            item["budget"],
        ),
    )
    if best["teacher_nll"] > config.teacher_verifier_fallback_nll:
        selected = {
            "action": "full",
            "teacher_nll": full_nll,
            "fallback_reason": "teacher_nll_threshold",
            "best_compact": best,
            "fallback_threshold": config.teacher_verifier_fallback_nll,
        }
        rows.append(
            make_row(
                case=case,
                method="teacher_likelihood_kv_planner",
                context_tokens=context_tokens,
                active_kv_tokens=context_tokens,
                query_tokens=int(query_ids.shape[1]),
                selected_pages_value=json.dumps({"selected": selected, "candidates": candidates}, ensure_ascii=False),
                planner_action="full",
                planner_seconds=method_seconds,
                prefill_seconds=0.0,
                full_prefill_seconds=full_prefill,
                decode_seconds=full_decode,
                full_online_seconds=full_online,
                prediction=full_prediction,
                nll=full_nll,
            )
        )
        return

    rows.append(
        make_row(
            case=case,
            method="teacher_likelihood_kv_planner",
            context_tokens=context_tokens,
            active_kv_tokens=int(best["active_kv_tokens"]),
            query_tokens=int(query_ids.shape[1]),
            selected_pages_value=json.dumps({"selected": best, "candidates": candidates}, ensure_ascii=False),
            planner_action=str(best["action"]),
            planner_seconds=method_seconds,
            prefill_seconds=0.0,
            full_prefill_seconds=full_prefill,
            full_online_seconds=full_online,
            prediction=str(best["prediction"]),
            nll=float(best["teacher_nll"]),
        )
    )


def run_output_level_verifier_method(
    *,
    rows: list[ResultRow],
    case: BenchCase,
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    full_cache: Any,
    full_query_seconds: float,
    full_decode: float,
    full_prediction: str,
    full_nll: float,
    full_prefill: float,
    full_online: float,
    context_tokens: int,
    page_tokens: int,
    page_scores: list[float],
    score_denom: float,
    q_text: str,
    lexical_planner_seconds: float,
    decode_steps: int,
    verifier: RuntimeOutputLevelRiskVerifier | None,
    mode: str,
    min_budget: int,
) -> None:
    if verifier is None:
        return
    device = query_ids.device
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    pages_by_budget = {
        budget: top_pages_from_scores(page_scores, budget)
        for budget in verifier.budgets
        if budget <= total_pages
    }
    candidates: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}
    probs: dict[str, float] = {}
    method_seconds = lexical_planner_seconds
    base_values = verifier.build_base_values(
        benchmark=case.benchmark,
        task=case.task,
        context_tokens=context_tokens,
        query_tokens=int(query_ids.shape[1]),
        page_tokens=page_tokens,
        query=q_text,
        scores=page_scores,
        denom=score_denom,
        pages_by_budget=pages_by_budget,
    )

    eligible_actions = [
        action
        for action in sorted(verifier.compact_actions, key=lambda item: (compact_action_budget(item), item))
        if compact_action_budget(action) >= min_budget
    ]
    if not eligible_actions:
        eligible_actions = sorted(verifier.compact_actions, key=lambda item: (compact_action_budget(item), item))

    for action in eligible_actions:
        budget = compact_action_budget(action)
        if budget not in pages_by_budget:
            continue
        selected_pages = pages_by_budget[budget]
        if budget >= total_pages:
            prediction = full_prediction
            nll = full_nll
            active_tokens = context_tokens
            repack_seconds = 0.0
            query_seconds = full_query_seconds
            decode_seconds = full_decode
            method_seconds += query_seconds + decode_seconds
        else:
            selected_indices = page_indices(selected_pages, context_tokens, page_tokens, device)
            synchronize()
            start = time.perf_counter()
            selected_positions = torch.arange(int(selected_indices.numel()), dtype=torch.long, device=device)
            compact_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, selected_indices, selected_positions))
            synchronize()
            repack_seconds = time.perf_counter() - start
            active_tokens = cache_len(compact_cache)
            q_cache, compact_logits, query_seconds = run_query_on_cache(
                model,
                query_ids,
                compact_cache,
                position_start=active_tokens,
                past_len=active_tokens,
            )
            prediction, decode_seconds = greedy_decode(model, tokenizer, compact_logits, q_cache, decode_steps)
            nll = answer_nll(model, tokenizer, case.answers, compact_logits, q_cache)
            method_seconds += repack_seconds + query_seconds + decode_seconds
            del compact_cache, q_cache, compact_logits
        predictions[action] = prediction
        candidates.append(
            {
                "action": action,
                "budget": budget,
                "pages": selected_pages,
                "active_kv_tokens": active_tokens,
                "active_kv_ratio_vs_full": active_tokens / max(1, context_tokens),
                "prediction": prediction.replace("\n", " ")[:200],
                "answer_nll": nll,
                "repack_seconds": repack_seconds,
                "query_seconds": query_seconds,
                "decode_seconds": decode_seconds,
            }
        )

        if mode == "prefix":
            rank = verifier.compact_actions.index(action)
            probs[action] = verifier.safe_probability(
                base_values=base_values,
                action=action,
                rank=rank,
                active_kv_ratio=active_tokens / max(1, context_tokens),
                predictions=predictions,
            )
            if probs[action] >= verifier.threshold:
                selected = candidates[-1]
                selected_cost = (
                    float(selected["repack_seconds"])
                    + float(selected["query_seconds"])
                    + float(selected["decode_seconds"])
                )
                payload = {
                    "mode": mode,
                    "min_budget": min_budget,
                    "selected": selected,
                    "threshold": verifier.threshold,
                    "safe_probs": probs,
                    "candidates": candidates,
                }
                rows.append(
                    make_row(
                        case=case,
                        method="output_level_risk_kv_planner",
                        context_tokens=context_tokens,
                        active_kv_tokens=int(selected["active_kv_tokens"]),
                        query_tokens=int(query_ids.shape[1]),
                        selected_pages_value=json.dumps(payload, ensure_ascii=False),
                        planner_action=str(selected["action"]),
                        planner_seconds=max(0.0, method_seconds - selected_cost),
                        prefill_seconds=0.0,
                        full_prefill_seconds=full_prefill,
                        repack_seconds=float(selected["repack_seconds"]),
                        query_seconds=float(selected["query_seconds"]),
                        decode_seconds=float(selected["decode_seconds"]),
                        full_online_seconds=full_online,
                        prediction=str(selected["prediction"]),
                        nll=float(selected["answer_nll"]),
                    )
                )
                return

    for rank, action in enumerate(verifier.compact_actions):
        if action not in eligible_actions:
            continue
        candidate = next((item for item in candidates if item["action"] == action), None)
        if candidate is None:
            continue
        probs[action] = verifier.safe_probability(
            base_values=base_values,
            action=action,
            rank=rank,
            active_kv_ratio=float(candidate["active_kv_ratio_vs_full"]),
            predictions=predictions,
        )

    selected = None
    for candidate in sorted(candidates, key=lambda item: (item["budget"], item["action"])):
        if probs.get(str(candidate["action"]), 0.0) >= verifier.threshold:
            selected = candidate
            break

    if selected is None:
        payload = {
            "mode": mode,
            "min_budget": min_budget,
            "selected": {
                "action": "full",
                "fallback_reason": "no_candidate_above_threshold",
                "threshold": verifier.threshold,
            },
            "safe_probs": probs,
            "candidates": candidates,
        }
        rows.append(
            make_row(
                case=case,
                method="output_level_risk_kv_planner",
                context_tokens=context_tokens,
                active_kv_tokens=context_tokens,
                query_tokens=int(query_ids.shape[1]),
                selected_pages_value=json.dumps(payload, ensure_ascii=False),
                planner_action="full",
                planner_seconds=method_seconds,
                prefill_seconds=0.0,
                full_prefill_seconds=full_prefill,
                query_seconds=full_query_seconds,
                decode_seconds=full_decode,
                full_online_seconds=full_online,
                prediction=full_prediction,
                nll=full_nll,
            )
        )
        return

    selected_cost = (
        float(selected["repack_seconds"])
        + float(selected["query_seconds"])
        + float(selected["decode_seconds"])
    )
    payload = {
        "mode": mode,
        "min_budget": min_budget,
        "selected": selected,
        "threshold": verifier.threshold,
        "safe_probs": probs,
        "candidates": candidates,
    }
    rows.append(
        make_row(
            case=case,
            method="output_level_risk_kv_planner",
            context_tokens=context_tokens,
            active_kv_tokens=int(selected["active_kv_tokens"]),
            query_tokens=int(query_ids.shape[1]),
            selected_pages_value=json.dumps(payload, ensure_ascii=False),
            planner_action=str(selected["action"]),
            planner_seconds=max(0.0, method_seconds - selected_cost),
            prefill_seconds=0.0,
            full_prefill_seconds=full_prefill,
            repack_seconds=float(selected["repack_seconds"]),
            query_seconds=float(selected["query_seconds"]),
            decode_seconds=float(selected["decode_seconds"]),
            full_online_seconds=full_online,
            prediction=str(selected["prediction"]),
            nll=float(selected["answer_nll"]),
        )
    )


def add_oracle_rows(rows: list[ResultRow], case: BenchCase) -> None:
    case_rows = [row for row in rows if row.benchmark == case.benchmark and row.task == case.task and row.case_id == case.case_id]
    sparse = [
        row
        for row in case_rows
        if row.method
        in {
            "naive_kv_gather_absolute_query_pos",
            "naive_kv_gather_compact_query_pos",
            "rope_delta_repack_compact_query_pos",
            "rope_delta_repack_shifted_query_pos",
        }
    ]
    if sparse:
        best = max(sparse, key=lambda row: (row.score, -row.answer_nll, -row.active_kv_tokens))
        rows.append(ResultRow(**{**asdict(best), "method": "position_mode_oracle_sparse"}))
    full = [row for row in case_rows if row.method == "full_kv_cache"]
    if sparse and full:
        best_fb = max(sparse + full, key=lambda row: (row.score, -row.answer_nll, -row.active_kv_tokens))
        rows.append(ResultRow(**{**asdict(best_fb), "method": "position_mode_oracle_with_full"}))


def summarize(rows: list[ResultRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[ResultRow]] = {}
    for row in rows:
        bench_group = row.benchmark if row.benchmark == "longbench" else row.benchmark
        groups.setdefault(("__overall__", "__overall__", row.method), []).append(row)
        groups.setdefault((bench_group, "__all__", row.method), []).append(row)
        groups.setdefault((row.benchmark, row.task, row.method), []).append(row)
    full_tokens: dict[tuple[str, str], float] = {}
    for (bench, task, method), items in groups.items():
        if method == "full_kv_cache":
            full_tokens[(bench, task)] = statistics.mean(item.active_kv_tokens for item in items)
    out: list[dict[str, Any]] = []
    for (bench, task, method), items in sorted(groups.items()):
        ft = full_tokens.get((bench, task), statistics.mean(item.context_tokens for item in items))
        out.append(
            {
                "benchmark": bench,
                "task": task,
                "method": method,
                "samples": len(items),
                "avg_score": statistics.mean(item.score for item in items),
                "exact_accuracy": statistics.mean(item.exact_correct for item in items),
                "avg_rouge_l": statistics.mean(item.rouge_l for item in items),
                "avg_answer_nll": statistics.mean(item.answer_nll for item in items),
                "avg_active_kv_tokens": statistics.mean(item.active_kv_tokens for item in items),
                "active_kv_ratio_vs_full": statistics.mean(item.active_kv_tokens for item in items) / ft if ft else 0.0,
                "avg_planner_seconds": statistics.mean(item.planner_seconds for item in items),
                "avg_prefill_seconds": statistics.mean(item.prefill_seconds for item in items),
                "avg_full_prefill_seconds": statistics.mean(item.full_prefill_seconds for item in items),
                "avg_gather_seconds": statistics.mean(item.gather_seconds for item in items),
                "avg_repack_seconds": statistics.mean(item.repack_seconds for item in items),
                "avg_query_seconds": statistics.mean(item.query_seconds for item in items),
                "avg_decode_seconds": statistics.mean(item.decode_seconds for item in items),
                "avg_total_online_seconds": statistics.mean(item.total_online_seconds for item in items),
                "avg_speedup_vs_full_online": statistics.mean(item.speedup_vs_full_online for item in items),
                "avg_end_to_end_seconds": statistics.mean(item.end_to_end_seconds for item in items),
                "avg_speedup_vs_full_e2e": statistics.mean(item.speedup_vs_full_e2e for item in items),
                "avg_amortized4_speedup_vs_full_e2e": statistics.mean(
                    item.amortized4_speedup_vs_full_e2e for item in items
                ),
                "avg_amortized16_speedup_vs_full_e2e": statistics.mean(
                    item.amortized16_speedup_vs_full_e2e for item in items
                ),
            }
        )
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(config.dtype),
        attn_implementation=config.attn_implementation,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    two_stage_planner = None
    if config.two_stage_planner_path:
        planner_path = Path(config.two_stage_planner_path)
        if not planner_path.exists():
            raise FileNotFoundError(f"two-stage planner checkpoint not found: {planner_path}")
        two_stage_planner = RuntimeTwoStagePlanner(
            str(planner_path),
            threshold_full=config.two_stage_threshold_full,
            threshold_k3=config.two_stage_threshold_k3,
        )
    variable_budget_planner = None
    if config.variable_budget_planner_path:
        planner_path = Path(config.variable_budget_planner_path)
        if not planner_path.exists():
            raise FileNotFoundError(f"variable-budget planner checkpoint not found: {planner_path}")
        variable_budget_planner = RuntimeVariableBudgetPlanner(
            str(planner_path),
            policy=config.variable_budget_policy,
            tail_threshold=config.variable_budget_tail_threshold,
            temperature=config.variable_budget_temperature,
            source_name=config.variable_budget_source,
            max_examples_per_task=config.max_examples_per_task,
        )
    output_verifier = None
    if config.output_verifier_path:
        verifier_path = Path(config.output_verifier_path)
        if not verifier_path.exists():
            raise FileNotFoundError(f"output-level verifier checkpoint not found: {verifier_path}")
        output_verifier = RuntimeOutputLevelRiskVerifier(
            str(verifier_path),
            threshold=config.output_verifier_threshold,
            source_name=config.output_verifier_source,
            max_examples_per_task=config.max_examples_per_task,
            budget_override=config.output_verifier_budgets,
        )

    bcfg = bench_config(config)
    cases = load_longbench_cases(bcfg) + load_ruler_cases(bcfg)
    case_start = max(0, int(config.case_start))
    if case_start:
        cases = cases[case_start:]
    if int(config.case_limit) > 0:
        cases = cases[: int(config.case_limit)]
    rows: list[ResultRow] = []

    for case_idx, case in enumerate(cases):
        context_ids = tokenize_context(tokenizer, case, config.max_context_tokens)
        if not context_ids:
            continue
        q_text = query_text(case)
        query_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).view(1, -1)
        planner_start = time.perf_counter()
        page_scores, score_denom = lexical_page_scores(tokenizer, context_ids, q_text, config.page_tokens)
        lexical_planner_seconds = time.perf_counter() - planner_start
        pages = top_pages_from_scores(page_scores, config.top_k)
        k2_pages = top_pages_from_scores(page_scores, 2)
        k3_pages = top_pages_from_scores(page_scores, 3)
        planner_action = ""
        planner_probs: dict[str, float] = {}
        planner_seconds = 0.0
        if two_stage_planner is not None:
            two_stage_start = time.perf_counter()
            planner_action, planner_probs = two_stage_planner.predict(
                benchmark=case.benchmark,
                task=case.task,
                context_tokens=len(context_ids),
                query_tokens=int(query_ids.shape[1]),
                page_tokens=config.page_tokens,
                query=q_text,
                scores=page_scores,
                denom=score_denom,
                k2_pages=k2_pages,
                k3_pages=k3_pages,
            )
            planner_seconds = lexical_planner_seconds + (time.perf_counter() - two_stage_start)
        variable_budget_action = ""
        variable_budget_probs: dict[str, float] = {}
        variable_budget_pages_by_budget: dict[int, list[int]] = {}
        variable_budget_planner_seconds = 0.0
        if variable_budget_planner is not None:
            variable_budget_pages_by_budget = {
                budget: top_pages_from_scores(page_scores, budget)
                for budget in variable_budget_planner.budgets
            }
            variable_budget_start = time.perf_counter()
            variable_budget_action, variable_budget_probs = variable_budget_planner.predict(
                benchmark=case.benchmark,
                task=case.task,
                context_tokens=len(context_ids),
                query_tokens=int(query_ids.shape[1]),
                page_tokens=config.page_tokens,
                query=q_text,
                scores=page_scores,
                denom=score_denom,
                pages_by_budget=variable_budget_pages_by_budget,
            )
            raw_variable_budget_action = variable_budget_action
            variable_budget_action = apply_variable_budget_floor(
                variable_budget_action,
                variable_budget_planner.budgets,
                config.variable_budget_min_budget,
            )
            variable_budget_planner_seconds = lexical_planner_seconds + (time.perf_counter() - variable_budget_start)
        indices = page_indices(pages, len(context_ids), config.page_tokens, device)
        pages_json = json.dumps(pages)
        decode_steps = config.max_new_tokens_summary if case.task in SUMMARY_TASKS else config.max_new_tokens_exact

        full_cache, _, full_prefill = prefill(model, context_tensor)
        full_q_cache, full_logits, full_query_seconds = run_query_on_cache(
            model, query_ids, full_cache, position_start=len(context_ids), past_len=len(context_ids)
        )
        full_prediction, full_decode = greedy_decode(model, tokenizer, full_logits, full_q_cache, decode_steps)
        full_nll = answer_nll(model, tokenizer, case.answers, full_logits, full_q_cache)
        full_online = full_query_seconds + full_decode
        rows.append(
            make_row(
                case=case,
                method="full_kv_cache",
                context_tokens=len(context_ids),
                active_kv_tokens=len(context_ids),
                query_tokens=int(query_ids.shape[1]),
                selected_pages_value="all",
                planner_action="full",
                prefill_seconds=full_prefill,
                full_prefill_seconds=full_prefill,
                gather_seconds=0.0,
                repack_seconds=0.0,
                query_seconds=full_query_seconds,
                decode_seconds=full_decode,
                full_online_seconds=full_online,
                prediction=full_prediction,
                nll=full_nll,
            )
        )
        naive_legacy = compact_cache = shifted_cache = prompt_cache = None
        two_stage_cache = None
        variable_budget_cache = None

        if any_method_enabled(
            config,
            ("naive_kv_gather_absolute_query_pos", "naive_kv_gather_compact_query_pos"),
        ):
            synchronize()
            start = time.perf_counter()
            naive_legacy = gather_cache(full_cache, indices)
            synchronize()
            gather_seconds = time.perf_counter() - start

            if method_enabled(config, "naive_kv_gather_absolute_query_pos"):
                run_cache_method(
                    rows=rows,
                    case=case,
                    method="naive_kv_gather_absolute_query_pos",
                    model=model,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    cache=cache_from_legacy(naive_legacy),
                    context_tokens=len(context_ids),
                    active_kv_tokens=int(indices.numel()),
                    selected_pages_value=pages_json,
                    position_start=len(context_ids),
                    prefill_seconds=0.0,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=gather_seconds,
                    repack_seconds=0.0,
                    decode_steps=decode_steps,
                    full_online_seconds=full_online,
                )
            if method_enabled(config, "naive_kv_gather_compact_query_pos"):
                run_cache_method(
                    rows=rows,
                    case=case,
                    method="naive_kv_gather_compact_query_pos",
                    model=model,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    cache=cache_from_legacy(naive_legacy),
                    context_tokens=len(context_ids),
                    active_kv_tokens=int(indices.numel()),
                    selected_pages_value=pages_json,
                    position_start=int(indices.numel()),
                    prefill_seconds=0.0,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=0.0,
                    repack_seconds=0.0,
                    decode_steps=decode_steps,
                    full_online_seconds=full_online,
                )

        if method_enabled(config, "rope_delta_repack_compact_query_pos"):
            synchronize()
            start = time.perf_counter()
            compact_positions = torch.arange(int(indices.numel()), dtype=torch.long, device=device)
            compact_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, indices, compact_positions))
            synchronize()
            compact_repack = time.perf_counter() - start
            run_cache_method(
                rows=rows,
                case=case,
                method="rope_delta_repack_compact_query_pos",
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                cache=compact_cache,
                context_tokens=len(context_ids),
                active_kv_tokens=cache_len(compact_cache),
                selected_pages_value=pages_json,
                position_start=cache_len(compact_cache),
                prefill_seconds=0.0,
                full_prefill_seconds=full_prefill,
                gather_seconds=0.0,
                repack_seconds=compact_repack,
                decode_steps=decode_steps,
                full_online_seconds=full_online,
            )

        if method_enabled(config, "rope_delta_repack_shifted_query_pos"):
            synchronize()
            start = time.perf_counter()
            shifted_positions = indices - int(indices.min().item())
            shifted_cache = cache_from_legacy(gather_and_rope_repack_cache(model, full_cache, indices, shifted_positions))
            synchronize()
            shifted_repack = time.perf_counter() - start
            run_cache_method(
                rows=rows,
                case=case,
                method="rope_delta_repack_shifted_query_pos",
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                cache=shifted_cache,
                context_tokens=len(context_ids),
                active_kv_tokens=cache_len(shifted_cache),
                selected_pages_value=pages_json,
                position_start=len(context_ids) - int(indices.min().item()),
                prefill_seconds=0.0,
                full_prefill_seconds=full_prefill,
                gather_seconds=0.0,
                repack_seconds=shifted_repack,
                decode_steps=decode_steps,
                full_online_seconds=full_online,
            )

        if method_enabled(config, "prompt_rebuild_selected_pages"):
            selected_text = selected_context_text(tokenizer, context_ids, pages, config.page_tokens)
            prompt_text = selected_text + "\n\n" + q_text
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            prompt_cache, prompt_logits, prompt_prefill = prefill(model, prompt_ids)
            prompt_prediction, prompt_decode = greedy_decode(model, tokenizer, prompt_logits, prompt_cache, decode_steps)
            prompt_nll = answer_nll(model, tokenizer, case.answers, prompt_logits, prompt_cache)
            rows.append(
                make_row(
                    case=case,
                    method="prompt_rebuild_selected_pages",
                    context_tokens=len(context_ids),
                    active_kv_tokens=int(prompt_ids.shape[1]),
                    query_tokens=int(query_ids.shape[1]),
                    selected_pages_value=pages_json,
                    planner_action="prompt_rebuild",
                    prefill_seconds=prompt_prefill,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=0.0,
                    repack_seconds=0.0,
                    query_seconds=0.0,
                    decode_seconds=prompt_decode,
                    full_online_seconds=full_online,
                    prediction=prompt_prediction,
                    nll=prompt_nll,
                )
            )

        if two_stage_planner is not None and method_enabled(config, "two_stage_calibrated_kv_planner"):
            two_stage_selected = {
                "action": planner_action,
                "k2_pages": k2_pages,
                "k3_pages": k3_pages,
                "probs": planner_probs,
            }
            if planner_action == "full":
                rows.append(
                    make_row(
                        case=case,
                        method="two_stage_calibrated_kv_planner",
                        context_tokens=len(context_ids),
                        active_kv_tokens=len(context_ids),
                        query_tokens=int(query_ids.shape[1]),
                        selected_pages_value=json.dumps(two_stage_selected),
                        planner_action=planner_action,
                        planner_seconds=planner_seconds,
                        prefill_seconds=0.0,
                        full_prefill_seconds=full_prefill,
                        gather_seconds=0.0,
                        repack_seconds=0.0,
                        query_seconds=full_query_seconds,
                        decode_seconds=full_decode,
                        full_online_seconds=full_online,
                        prediction=full_prediction,
                        nll=full_nll,
                    )
                )
            else:
                selected_pages = k3_pages if planner_action == "k3_compact" else k2_pages
                selected_indices = page_indices(selected_pages, len(context_ids), config.page_tokens, device)
                synchronize()
                start = time.perf_counter()
                selected_positions = torch.arange(int(selected_indices.numel()), dtype=torch.long, device=device)
                two_stage_cache = cache_from_legacy(
                    gather_and_rope_repack_cache(model, full_cache, selected_indices, selected_positions)
                )
                synchronize()
                two_stage_repack = time.perf_counter() - start
                run_cache_method(
                    rows=rows,
                    case=case,
                    method="two_stage_calibrated_kv_planner",
                    model=model,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    cache=two_stage_cache,
                    context_tokens=len(context_ids),
                    active_kv_tokens=cache_len(two_stage_cache),
                    selected_pages_value=json.dumps(two_stage_selected),
                    planner_action=planner_action,
                    planner_seconds=planner_seconds,
                    position_start=cache_len(two_stage_cache),
                    prefill_seconds=0.0,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=0.0,
                    repack_seconds=two_stage_repack,
                    decode_steps=decode_steps,
                    full_online_seconds=full_online,
                )

        if variable_budget_planner is not None and method_enabled(config, "variable_budget_kv_planner"):
            variable_budget_selected = {
                "action": variable_budget_action,
                "raw_action": raw_variable_budget_action,
                "min_budget": int(config.variable_budget_min_budget),
                "policy": config.variable_budget_policy,
                "tail_threshold": config.variable_budget_tail_threshold,
                "temperature": config.variable_budget_temperature,
                "source": variable_budget_planner.resolve_source(case.benchmark),
                "pages_by_budget": {str(k): v for k, v in sorted(variable_budget_pages_by_budget.items())},
                "probs": variable_budget_probs,
            }
            if variable_budget_action == "full":
                rows.append(
                    make_row(
                        case=case,
                        method="variable_budget_kv_planner",
                        context_tokens=len(context_ids),
                        active_kv_tokens=len(context_ids),
                        query_tokens=int(query_ids.shape[1]),
                        selected_pages_value=json.dumps(variable_budget_selected),
                        planner_action=variable_budget_action,
                        planner_seconds=variable_budget_planner_seconds,
                        prefill_seconds=0.0,
                        full_prefill_seconds=full_prefill,
                        gather_seconds=0.0,
                        repack_seconds=0.0,
                        query_seconds=full_query_seconds,
                        decode_seconds=full_decode,
                        full_online_seconds=full_online,
                        prediction=full_prediction,
                        nll=full_nll,
                    )
                )
            else:
                selected_budget = compact_action_budget(variable_budget_action)
                selected_pages = variable_budget_pages_by_budget.get(
                    selected_budget,
                    top_pages_from_scores(page_scores, selected_budget),
                )
                selected_indices = page_indices(selected_pages, len(context_ids), config.page_tokens, device)
                synchronize()
                start = time.perf_counter()
                selected_positions = torch.arange(int(selected_indices.numel()), dtype=torch.long, device=device)
                variable_budget_cache = cache_from_legacy(
                    gather_and_rope_repack_cache(model, full_cache, selected_indices, selected_positions)
                )
                synchronize()
                variable_budget_repack = time.perf_counter() - start
                run_cache_method(
                    rows=rows,
                    case=case,
                    method="variable_budget_kv_planner",
                    model=model,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    cache=variable_budget_cache,
                    context_tokens=len(context_ids),
                    active_kv_tokens=cache_len(variable_budget_cache),
                    selected_pages_value=json.dumps(variable_budget_selected),
                    planner_action=variable_budget_action,
                    planner_seconds=variable_budget_planner_seconds,
                    position_start=cache_len(variable_budget_cache),
                    prefill_seconds=0.0,
                    full_prefill_seconds=full_prefill,
                    gather_seconds=0.0,
                    repack_seconds=variable_budget_repack,
                    decode_steps=decode_steps,
                    full_online_seconds=full_online,
                )

        if method_enabled(config, "consistency_probe_kv_planner"):
            run_consistency_probe_method(
                rows=rows,
                case=case,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                full_cache=full_cache,
                full_logits=full_logits,
                full_query_seconds=full_query_seconds,
                full_decode=full_decode,
                full_prediction=full_prediction,
                full_nll=full_nll,
                full_prefill=full_prefill,
                full_online=full_online,
                context_tokens=len(context_ids),
                page_tokens=config.page_tokens,
                page_scores=page_scores,
                lexical_planner_seconds=lexical_planner_seconds,
                decode_steps=decode_steps,
                config=config,
            )

        if method_enabled(config, "teacher_likelihood_kv_planner"):
            run_teacher_verifier_method(
                rows=rows,
                case=case,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                full_cache=full_cache,
                full_decode=full_decode,
                full_prediction=full_prediction,
                full_nll=full_nll,
                full_prefill=full_prefill,
                full_online=full_online,
                context_tokens=len(context_ids),
                page_tokens=config.page_tokens,
                page_scores=page_scores,
                lexical_planner_seconds=lexical_planner_seconds,
                decode_steps=decode_steps,
                config=config,
            )

        if output_verifier is not None and method_enabled(config, "output_level_risk_kv_planner"):
            del full_q_cache, naive_legacy, compact_cache, shifted_cache, prompt_cache
            full_q_cache = naive_legacy = compact_cache = shifted_cache = prompt_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_verifier_min_budget = max(1, int(config.output_verifier_min_budget))
        benchmark_context_tokens = len(context_ids)
        ruler_match = re.match(r"ruler_(\d+)", case.benchmark)
        if ruler_match is not None:
            benchmark_context_tokens = max(benchmark_context_tokens, int(ruler_match.group(1)))
        if (
            case.benchmark.startswith("ruler")
            and benchmark_context_tokens >= int(config.output_verifier_long_ruler_context_threshold)
            and int(config.output_verifier_long_ruler_min_budget) > 0
        ):
            output_verifier_min_budget = max(output_verifier_min_budget, int(config.output_verifier_long_ruler_min_budget))
        if method_enabled(config, "output_level_risk_kv_planner"):
            run_output_level_verifier_method(
                rows=rows,
                case=case,
                model=model,
                tokenizer=tokenizer,
                query_ids=query_ids,
                full_cache=full_cache,
                full_query_seconds=full_query_seconds,
                full_decode=full_decode,
                full_prediction=full_prediction,
                full_nll=full_nll,
                full_prefill=full_prefill,
                full_online=full_online,
                context_tokens=len(context_ids),
                page_tokens=config.page_tokens,
                page_scores=page_scores,
                score_denom=score_denom,
                q_text=q_text,
                lexical_planner_seconds=lexical_planner_seconds,
                decode_steps=decode_steps,
                verifier=output_verifier,
                mode=config.output_verifier_mode,
                min_budget=output_verifier_min_budget,
            )

        add_oracle_rows(rows, case)
        write_csv(output_dir / "results.partial.csv", [asdict(row) for row in rows])
        print(f"finished {case_idx + 1}/{len(cases)} {case.benchmark}/{case.task}/{case.case_id}", flush=True)

        del full_cache, full_q_cache, naive_legacy, compact_cache, shifted_cache, prompt_cache
        if two_stage_cache is not None:
            del two_stage_cache
        if variable_budget_cache is not None:
            del variable_budget_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(output_dir / "results.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "num_cases": len(cases), "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("benchmark,task,method,samples,score,exact,nll,kv_ratio,online_speedup,e2e_speedup,amort16_speedup")
    for row in summary:
        if row["benchmark"] == "__overall__":
            print(
                f"{row['benchmark']},{row['task']},{row['method']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['exact_accuracy']:.4f},{row['avg_answer_nll']:.4f},"
                f"{row['active_kv_ratio_vs_full']:.4f},{row['avg_speedup_vs_full_online']:.3f},"
                f"{row['avg_speedup_vs_full_e2e']:.3f},{row['avg_amortized16_speedup_vs_full_e2e']:.3f}"
            )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
