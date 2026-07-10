from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import random
import re
import string
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    resolve_dtype,
)


LONG_BENCH_PROMPTS = {
    "narrativeqa": {
        "prefix": (
            "You are given a story, which can be either a novel or a movie script, and a question. "
            "Answer the question asconcisely as you can, using a single phrase if possible. "
            "Do not provide any explanation.\n\nStory: "
        ),
        "suffix": (
            "\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. "
            "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
        ),
        "max_new_tokens": 128,
        "metric": "qa_f1",
    },
    "qasper": {
        "prefix": (
            "You are given a scientific article and a question. Answer the question as concisely as you can, "
            "using a single phrase or sentence if possible. If the question cannot be answered based on the "
            'information in the article, write "unanswerable". If the question is a yes/no question, answer '
            '"yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: '
        ),
        "suffix": (
            '\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. '
            'If the question cannot be answered based on the information in the article, write "unanswerable". '
            'If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
            "Question: {input}\n\nAnswer:"
        ),
        "max_new_tokens": 128,
        "metric": "qa_f1",
    },
    "multifieldqa_en": {
        "prefix": "Read the following text and answer briefly.\n\n",
        "suffix": (
            "\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 64,
        "metric": "qa_f1",
    },
    "hotpotqa": {
        "prefix": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
        "suffix": (
            "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 32,
        "metric": "qa_f1",
    },
    "2wikimqa": {
        "prefix": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
        "suffix": (
            "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 32,
        "metric": "qa_f1",
    },
    "musique": {
        "prefix": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
        "suffix": (
            "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "Question: {input}\nAnswer:"
        ),
        "max_new_tokens": 32,
        "metric": "qa_f1",
    },
    "qmsum": {
        "prefix": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n",
        "suffix": "\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "max_new_tokens": 512,
        "metric": "rouge_l",
        "global_task": True,
    },
    "trec": {
        "prefix": "Please determine the type of the question below. Here are some examples of questions.\n\n",
        "suffix": "\n{input}",
        "max_new_tokens": 64,
        "metric": "classification",
        "no_chat": True,
    },
    "triviaqa": {
        "prefix": (
            "Answer the question based on the given passage. Only give me the answer and do not output any other words. "
            "The following are some examples.\n\n"
        ),
        "suffix": "\n\n{input}",
        "max_new_tokens": 32,
        "metric": "qa_f1",
        "no_chat": True,
    },
    "samsum": {
        "prefix": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n",
        "suffix": "\n\n{input}",
        "max_new_tokens": 128,
        "metric": "rouge_l",
        "global_task": True,
        "no_chat": True,
    },
    "passage_retrieval_en": {
        "prefix": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n",
        "suffix": (
            '\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. '
            'The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
        ),
        "max_new_tokens": 32,
        "metric": "retrieval",
    },
    "passage_count": {
        "prefix": (
            "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
            "Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. "
            "In other words, how many non-repeating paragraphs are there in total?\n\n"
        ),
        "suffix": (
            "\n\nPlease enter the final count of unique paragraphs after removing duplicates. "
            "The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
        ),
        "max_new_tokens": 32,
        "metric": "count",
    },
    "gov_report": {
        "prefix": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n",
        "suffix": "\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "max_new_tokens": 512,
        "metric": "rouge_l",
        "global_task": True,
    },
    "multi_news": {
        "prefix": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n",
        "suffix": "\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "max_new_tokens": 512,
        "metric": "rouge_l",
        "global_task": True,
    },
    "lcc": {
        "prefix": "Please complete the code given below. \n",
        "suffix": "Next line of code:\n",
        "max_new_tokens": 64,
        "metric": "code_sim",
        "no_chat": True,
    },
    "repobench-p": {
        "prefix": "Please complete the code given below. \n",
        "suffix": "{input}Next line of code:\n",
        "max_new_tokens": 64,
        "metric": "code_sim",
        "no_chat": True,
    },
}


STOPWORDS = {
    "about",
    "above",
    "after",
    "all",
    "also",
    "and",
    "answer",
    "are",
    "based",
    "below",
    "between",
    "can",
    "determine",
    "does",
    "from",
    "given",
    "have",
    "how",
    "into",
    "only",
    "paragraph",
    "passage",
    "please",
    "question",
    "read",
    "should",
    "some",
    "story",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "words",
    "would",
}


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    output_dir: str
    benchmarks: str
    longbench_tasks: str
    ruler_tasks: str
    max_samples_per_task: int
    max_context_tokens: int
    max_new_tokens_override: int
    seed: int
    methods: str
    budget_tokens: int
    sink_tokens: int
    recent_tokens: int
    page_tokens: int
    prefill_chunk_tokens: int
    obs_window_tokens: int
    snap_pool_kernel: int
    ours_scorer: str
    semantic_embed_max_tokens: int
    semantic_weight: float
    lexical_weight: float
    entity_weight: float
    structural_weight: float
    coverage_weight: float
    ours_mmr_lambda: float
    ours_coverage_mmr_weight: float
    ours_coverage_mmr_max_terms: int
    ours_coverage_certificate_tasks: str
    ours_coverage_certificate_budget_fraction: float
    ours_coverage_certificate_min_terms: int
    ours_coverage_risk_tasks: str
    ours_coverage_risk_min_recall: float
    ours_coverage_risk_min_terms: int
    ours_coverage_risk_budget_tokens: int
    anchor_pages_per_key: int
    ours_flow_neighbor_radius: int
    ours_flow_neighbor_budget_fraction: float
    ours_flow_neighbor_min_score: float
    ours_flow_score_smooth_weight: float
    ours_flow_anchor_boost: float
    ours_multiscale_group_pages: int
    ours_multiscale_weight: float
    ours_idf_mix: float
    ours_spread_budget_fraction: float
    ours_spread_gap_threshold: float
    ours_spread_bins: int
    ours_spread_min_score: float
    ours_bridge_budget_fraction: float
    ours_bridge_min_score: float
    ours_bridge_max_terms: int
    ours_bridge_tasks: str
    ours_graph_bridge_tasks: str
    ours_graph_bridge_budget_fraction: float
    ours_graph_bridge_seed_pages: int
    ours_graph_bridge_max_terms: int
    ours_graph_bridge_min_score: float
    ours_task_policy_json: str
    ours_full_fallback_tasks: str
    ours_label_support_tasks: str
    ours_label_backtrack_pages: int
    ours_label_budget_fraction: float
    ours_passage_closure_tasks: str
    ours_passage_closure_budget_fraction: float
    ours_passage_closure_radius_pages: int
    ours_structured_fingerprint_tasks: str
    ours_structured_fingerprint_budget_fraction: float
    ours_direct_structured_answer_tasks: str
    ours_short_decode_tasks: str
    ours_short_decode_max_tokens: int
    ours_output_verifier_tasks: str
    ours_output_probe_max_tokens: int
    ours_retry_budget_tokens: int
    ours_score_risk_tasks: str
    ours_score_risk_budget_tokens: int
    ours_score_risk_min_gap2: float
    ours_score_risk_min_gap3: float
    ours_score_risk_max_gap2: float
    ours_score_risk_max_gap3: float
    ours_score_risk_max_entropy: float
    ours_score_risk_entropy_at_most: float
    ours_score_risk_min_top_score: float
    ours_score_risk_raw_prefix_at_most: int
    ours_score_risk_raw_prefix_at_least: int
    ours_score_risk_linear_threshold: float
    ours_score_risk_gap2_weight: float
    ours_score_risk_gap3_weight: float
    ours_score_risk_top_score_weight: float
    ours_budget_ladder_tasks: str
    ours_budget_ladder_tokens: str
    ours_budget_ladder_gap2_thresholds: str
    ours_budget_ladder_entropy_thresholds: str
    ours_budget_ladder_top_score_thresholds: str
    ours_consistency_verifier_tasks: str
    ours_consistency_budget_tokens: int
    ours_consistency_probe_max_tokens: int
    ours_consistency_requires_score_risk: int
    ours_title_anchor_tasks: str
    ours_grounding_verifier_tasks: str
    ours_support_window_verifier_tasks: str
    ours_support_window_radius_words: int
    ours_support_window_min_query_terms: int
    ours_bm25_mix: float
    ours_bm25_k1: float
    ours_bm25_b: float
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    prompt_wrapper: str
    force_no_chat_tasks: str
    longbench_zip_path: str
    hf_cache_dir: str
    lm_eval_path: str
    ruler_lengths: str
    log_every: int


@dataclass
class Example:
    benchmark: str
    task: str
    sample_id: str
    context: str
    query: str
    answers: list[str]
    prefix_template: str
    suffix_template: str
    metric: str
    max_new_tokens: int
    length: int
    all_classes: list[str]
    no_chat: bool = False


@dataclass
class Page:
    page_id: int
    text: str
    token_start: int
    token_end: int
    score: float = 0.0


@dataclass
class PromptBundle:
    input_ids: torch.Tensor
    prefix_token_count: int
    context_token_start: int
    query_start: int
    suffix_token_count: int
    page_spans: dict[int, tuple[int, int]]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled public LongBench/RULER KV-cache benchmark. The runner keeps model, prompts, budgets, "
            "generation settings, timing, and sampled IDs fixed across full KV and sparse KV methods."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", default="longbench,ruler")
    parser.add_argument("--longbench_tasks", default="passage_retrieval_en,hotpotqa")
    parser.add_argument("--ruler_tasks", default="niah_single_1")
    parser.add_argument("--max_samples_per_task", type=int, default=2)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument(
        "--max_new_tokens_override",
        type=int,
        default=0,
        help="Use the benchmark default when 0; otherwise cap all generations to this length.",
    )
    parser.add_argument("--seed", type=int, default=2026070302)
    parser.add_argument(
        "--methods",
        default="full_kv,streamingllm_sink_recent,h2o_observe,snapkv_observe,ours_page_gather",
    )
    parser.add_argument("--budget_tokens", type=int, default=512)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=256)
    parser.add_argument("--page_tokens", type=int, default=256)
    parser.add_argument(
        "--prefill_chunk_tokens",
        type=int,
        default=2048,
        help="Chunk prefix prefill to reduce activation memory. Use 0 to prefill the whole prefix at once.",
    )
    parser.add_argument("--obs_window_tokens", type=int, default=64)
    parser.add_argument("--snap_pool_kernel", type=int, default=7)
    parser.add_argument(
        "--ours_scorer",
        choices=[
            "lexical",
            "semantic",
            "late_interaction",
            "hybrid",
            "hybrid_mmr",
            "hybrid_late_mmr",
            "hybrid_late_mmr_bm25_flow",
            "hybrid_late_mmr_bridge_flow",
            "hybrid_late_mmr_flow",
            "hybrid_late_mmr_multiscale_bm25_flow",
            "hybrid_late_mmr_multiscale_bridge_flow",
            "hybrid_late_mmr_multiscale_task_bridge_flow",
            "hybrid_late_mmr_multiscale_flow",
            "hybrid_late_mmr_idf_flow",
            "hybrid_late_mmr_multiscale_idf_flow",
            "hybrid_late_mmr_idf_spread_flow",
            "hybrid_late_mmr_multiscale_idf_spread_flow",
        ],
        default="hybrid_late_mmr",
        help=(
            "Scorer for ours_page_gather. semantic/hybrid use the LM input embedding table without downloading an "
            "extra model; late_interaction uses query-token to page-token MaxSim as a lightweight reranker."
        ),
    )
    parser.add_argument("--semantic_embed_max_tokens", type=int, default=192)
    parser.add_argument("--semantic_weight", type=float, default=0.55)
    parser.add_argument("--lexical_weight", type=float, default=0.25)
    parser.add_argument("--entity_weight", type=float, default=0.15)
    parser.add_argument("--structural_weight", type=float, default=0.05)
    parser.add_argument(
        "--coverage_weight",
        type=float,
        default=0.15,
        help="Extra position-coverage weight for global summarization style tasks.",
    )
    parser.add_argument("--ours_mmr_lambda", type=float, default=0.82)
    parser.add_argument(
        "--ours_coverage_mmr_weight",
        type=float,
        default=0.0,
        help="Bonus for MMR candidates that cover previously uncovered query words, entities, or numbers.",
    )
    parser.add_argument(
        "--ours_coverage_mmr_max_terms",
        type=int,
        default=32,
        help="Maximum number of query coverage terms used by coverage-aware MMR.",
    )
    parser.add_argument(
        "--ours_coverage_certificate_tasks",
        default="",
        help="Comma-separated task names that reserve a small budget for hard query-term coverage pages.",
    )
    parser.add_argument(
        "--ours_coverage_certificate_budget_fraction",
        type=float,
        default=0.22,
        help="Maximum fraction of the sparse budget spent on hard query-term coverage certificate pages.",
    )
    parser.add_argument(
        "--ours_coverage_certificate_min_terms",
        type=int,
        default=3,
        help="Minimum query terms required before hard coverage certificate selection is active.",
    )
    parser.add_argument(
        "--ours_coverage_risk_tasks",
        default="",
        help="Comma-separated task names that enable pre-decode query-coverage risk escalation.",
    )
    parser.add_argument(
        "--ours_coverage_risk_min_recall",
        type=float,
        default=-1.0,
        help="Escalate the memory action when selected pages cover less than this fraction of query terms.",
    )
    parser.add_argument(
        "--ours_coverage_risk_min_terms",
        type=int,
        default=3,
        help="Minimum number of query coverage terms required before coverage-risk escalation can fire.",
    )
    parser.add_argument(
        "--ours_coverage_risk_budget_tokens",
        type=int,
        default=0,
        help="Expanded sparse budget for coverage-risk escalation. Set >= prefix length to use full KV.",
    )
    parser.add_argument(
        "--anchor_pages_per_key",
        type=int,
        default=2,
        help="For typed-anchor queries, reserve up to this many exact-anchor pages per query key before MMR fill.",
    )
    parser.add_argument(
        "--ours_flow_neighbor_radius",
        type=int,
        default=1,
        help="For hybrid_late_mmr_flow, add local evidence-support pages around a selected evidence page.",
    )
    parser.add_argument(
        "--ours_flow_neighbor_budget_fraction",
        type=float,
        default=0.22,
        help="Maximum fraction of context budget reserved for evidence-flow neighbor support pages.",
    )
    parser.add_argument(
        "--ours_flow_neighbor_min_score",
        type=float,
        default=0.18,
        help="Minimum flow score for neighbor support pages unless the neighbor has an exact anchor/entity hit.",
    )
    parser.add_argument(
        "--ours_flow_score_smooth_weight",
        type=float,
        default=0.18,
        help="Amount of local neighbor evidence mixed into each page score for hybrid_late_mmr_flow.",
    )
    parser.add_argument(
        "--ours_flow_anchor_boost",
        type=float,
        default=0.22,
        help="Score boost for pages with exact query anchor or entity hits in hybrid_late_mmr_flow.",
    )
    parser.add_argument(
        "--ours_multiscale_group_pages",
        type=int,
        default=4,
        help="For hybrid_late_mmr_multiscale_flow, number of fine pages per coarse evidence group.",
    )
    parser.add_argument(
        "--ours_multiscale_weight",
        type=float,
        default=0.20,
        help="For hybrid_late_mmr_multiscale_flow, weight of coarse group support mixed into each page score.",
    )
    parser.add_argument(
        "--ours_idf_mix",
        type=float,
        default=0.65,
        help="For IDF flow scorers, mix this fraction of document-local IDF overlap into the lexical component.",
    )
    parser.add_argument(
        "--ours_spread_budget_fraction",
        type=float,
        default=0.18,
        help="For spread-flow scorers, reserve this budget fraction for position-diverse evidence rescue pages.",
    )
    parser.add_argument(
        "--ours_spread_gap_threshold",
        type=float,
        default=0.12,
        help="Trigger spread rescue when top evidence scores are closer than this threshold.",
    )
    parser.add_argument(
        "--ours_spread_bins",
        type=int,
        default=4,
        help="Number of context position bins used by spread-flow evidence rescue.",
    )
    parser.add_argument(
        "--ours_spread_min_score",
        type=float,
        default=0.08,
        help="Minimum page score considered by spread-flow evidence rescue.",
    )
    parser.add_argument(
        "--ours_bridge_budget_fraction",
        type=float,
        default=0.16,
        help="For bridge-flow scorers, reserve this budget fraction for entity-chain expansion pages.",
    )
    parser.add_argument(
        "--ours_bridge_min_score",
        type=float,
        default=0.0,
        help="Minimum base page score for bridge expansion candidates; entity overlap is still required.",
    )
    parser.add_argument(
        "--ours_bridge_max_terms",
        type=int,
        default=24,
        help="Maximum rare entity terms extracted from an evidence center for bridge expansion.",
    )
    parser.add_argument(
        "--ours_bridge_tasks",
        default="qasper,musique,passage_retrieval_en,hotpotqa",
        help="Comma-separated task names that activate task-adaptive bridge expansion.",
    )
    parser.add_argument(
        "--ours_graph_bridge_tasks",
        default="",
        help="Comma-separated task names that add query-seed to evidence-link page pairs before regular fill.",
    )
    parser.add_argument(
        "--ours_graph_bridge_budget_fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of the sparse budget reserved for two-hop graph bridge page pairs.",
    )
    parser.add_argument(
        "--ours_graph_bridge_seed_pages",
        type=int,
        default=4,
        help="Number of high-scoring query-seed pages used by graph bridge expansion.",
    )
    parser.add_argument(
        "--ours_graph_bridge_max_terms",
        type=int,
        default=24,
        help="Maximum rare entity terms extracted from each seed page for graph bridge expansion.",
    )
    parser.add_argument(
        "--ours_graph_bridge_min_score",
        type=float,
        default=0.0,
        help="Minimum base page score for graph bridge candidate pages.",
    )
    parser.add_argument(
        "--ours_task_policy_json",
        default="",
        help=(
            "Optional JSON string or JSON file path with per-task action overrides. Each task may override "
            "budget_tokens, sink_tokens, recent_tokens, page_tokens, ours_scorer, ours_bridge_tasks, bridge, "
            "full_fallback, label_support, output_verifier, and scalar ours_* routing parameters. Use "
            "bridge=true to activate task bridge only for that task; use full_fallback=true for the "
            "risk-aware minimum-safe action; use label_support=true for structured Paragraph-k support pages."
        ),
    )
    parser.add_argument(
        "--ours_full_fallback_tasks",
        default="",
        help="Comma-separated task names for which ours_page_gather keeps the full prefix as a safety fallback.",
    )
    parser.add_argument(
        "--ours_label_support_tasks",
        default="",
        help=(
            "Comma-separated task names that add structured label/header support pages, e.g. Paragraph-k pages, "
            "when a selected evidence page is a continuation chunk."
        ),
    )
    parser.add_argument(
        "--ours_label_backtrack_pages",
        type=int,
        default=4,
        help="Maximum number of previous pages searched for a structured label/header page.",
    )
    parser.add_argument(
        "--ours_label_budget_fraction",
        type=float,
        default=0.16,
        help="Maximum fraction of context budget used by structured label/header support pages.",
    )
    parser.add_argument(
        "--ours_passage_closure_tasks",
        default="",
        help="Comma-separated task names that reserve budget for neighboring chunks from the same Passage block.",
    )
    parser.add_argument(
        "--ours_passage_closure_budget_fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of context budget used by same-Passage closure pages.",
    )
    parser.add_argument(
        "--ours_passage_closure_radius_pages",
        type=int,
        default=2,
        help="Maximum page distance from a selected page for same-Passage closure.",
    )
    parser.add_argument(
        "--ours_structured_fingerprint_tasks",
        default="",
        help="Comma-separated task names that reserve one structural fingerprint page per Paragraph/Passage label.",
    )
    parser.add_argument(
        "--ours_structured_fingerprint_budget_fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of context budget used by structural fingerprint pages.",
    )
    parser.add_argument(
        "--ours_direct_structured_answer_tasks",
        default="",
        help=(
            "Comma-separated task names that answer directly from selected structural labels after sparse "
            "memory selection, e.g. Paragraph-k retrieval tasks."
        ),
    )
    parser.add_argument(
        "--ours_short_decode_tasks",
        default="",
        help="Comma-separated task names that cap the first sparse decode to a short format-focused answer.",
    )
    parser.add_argument(
        "--ours_short_decode_max_tokens",
        type=int,
        default=0,
        help="Maximum new tokens for short-decode tasks. Use 0 to keep the benchmark task max_new_tokens.",
    )
    parser.add_argument(
        "--ours_output_verifier_tasks",
        default="",
        help=(
            "Comma-separated task names that trigger a full-KV retry when the sparse answer violates the task "
            "output contract, e.g. missing Paragraph-k for passage retrieval."
        ),
    )
    parser.add_argument(
        "--ours_output_probe_max_tokens",
        type=int,
        default=0,
        help=(
            "When output verifier is active, cap the first sparse decode to this many tokens before deciding "
            "whether to retry with full KV. Use 0 to keep the benchmark task max_new_tokens."
        ),
    )
    parser.add_argument(
        "--ours_retry_budget_tokens",
        type=int,
        default=0,
        help=(
            "When an output/grounding verifier fails, first retry with this expanded sparse budget before "
            "falling back to full KV. Use 0 to keep the original direct full retry behavior."
        ),
    )
    parser.add_argument(
        "--ours_score_risk_tasks",
        default="",
        help=(
            "Comma-separated task names that use pre-decode page-score confidence to expand the memory action "
            "before generating."
        ),
    )
    parser.add_argument(
        "--ours_score_risk_budget_tokens",
        type=int,
        default=0,
        help=(
            "Expanded context budget used when score-risk confidence fires. Set larger than the raw prefix "
            "length to make the risky action a full-cache action."
        ),
    )
    parser.add_argument(
        "--ours_score_risk_min_gap2",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when top1-top2 page-score gap is at most this value.",
    )
    parser.add_argument(
        "--ours_score_risk_min_gap3",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when top1-top3 page-score gap is at most this value.",
    )
    parser.add_argument(
        "--ours_score_risk_max_gap2",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when top1-top2 page-score gap is at least this value.",
    )
    parser.add_argument(
        "--ours_score_risk_max_gap3",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when top1-top3 page-score gap is at least this value.",
    )
    parser.add_argument(
        "--ours_score_risk_max_entropy",
        type=float,
        default=2.0,
        help="If <=1, score-risk fires only when normalized page-score entropy is at least this value.",
    )
    parser.add_argument(
        "--ours_score_risk_entropy_at_most",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when normalized page-score entropy is at most this value.",
    )
    parser.add_argument(
        "--ours_score_risk_min_top_score",
        type=float,
        default=-1.0,
        help="If >=0, score-risk fires only when the top page score is below this value.",
    )
    parser.add_argument(
        "--ours_score_risk_raw_prefix_at_most",
        type=int,
        default=-1,
        help="If >=0, score-risk fires only when the raw prefix length is at most this many tokens.",
    )
    parser.add_argument(
        "--ours_score_risk_raw_prefix_at_least",
        type=int,
        default=-1,
        help="If >=0, score-risk fires only when the raw prefix length is at least this many tokens.",
    )
    parser.add_argument(
        "--ours_score_risk_linear_threshold",
        type=float,
        default=-1.0,
        help=(
            "If >=0, score-risk also fires only when entropy - w2*gap2 - w3*gap3 - wt*top_score "
            "is at least this conformal risk threshold."
        ),
    )
    parser.add_argument(
        "--ours_score_risk_gap2_weight",
        type=float,
        default=1.0,
        help="Gap2 weight used by the optional linear score-risk gate.",
    )
    parser.add_argument(
        "--ours_score_risk_gap3_weight",
        type=float,
        default=0.0,
        help="Gap3 weight used by the optional linear score-risk gate.",
    )
    parser.add_argument(
        "--ours_score_risk_top_score_weight",
        type=float,
        default=0.0,
        help="Top-page-score weight used by the optional linear score-risk gate.",
    )
    parser.add_argument(
        "--ours_budget_ladder_tasks",
        default="",
        help=(
            "Comma-separated task names that select the smallest safe context budget from a configured ladder "
            "using pre-decode page-score risk signals."
        ),
    )
    parser.add_argument(
        "--ours_budget_ladder_tokens",
        default="",
        help="Comma-separated sparse budgets, e.g. 512,1024,2048,4096. Empty disables the ladder.",
    )
    parser.add_argument(
        "--ours_budget_ladder_gap2_thresholds",
        default="",
        help=(
            "Comma-separated gap2 thresholds for ladder upgrades. Values should be ordered from mild to severe "
            "risk, e.g. 0.14,0.10,0.06 for progressively smaller top1-top2 gaps."
        ),
    )
    parser.add_argument(
        "--ours_budget_ladder_entropy_thresholds",
        default="",
        help="Comma-separated entropy thresholds for ladder upgrades, e.g. 0.85,0.90,0.95.",
    )
    parser.add_argument(
        "--ours_budget_ladder_top_score_thresholds",
        default="",
        help=(
            "Comma-separated top-score thresholds for ladder upgrades. Smaller top-page scores are riskier, "
            "e.g. 0.20,0.15,0.10."
        ),
    )
    parser.add_argument(
        "--ours_consistency_verifier_tasks",
        default="",
        help=(
            "Comma-separated task names that run a second sparse memory action and fall back to full KV if "
            "the two sparse answers disagree."
        ),
    )
    parser.add_argument(
        "--ours_consistency_budget_tokens",
        type=int,
        default=0,
        help=(
            "Expanded sparse budget for the action-consistency verifier. Use 0 to disable the second action."
        ),
    )
    parser.add_argument(
        "--ours_consistency_probe_max_tokens",
        type=int,
        default=0,
        help=(
            "When action-consistency verifier is active, cap the second sparse decode to this many tokens. "
            "Use 0 to keep the benchmark task max_new_tokens."
        ),
    )
    parser.add_argument(
        "--ours_consistency_requires_score_risk",
        type=int,
        default=0,
        help=(
            "When nonzero, run the action-consistency verifier only if the pre-decode score-risk gate fires."
        ),
    )
    parser.add_argument(
        "--ours_title_anchor_tasks",
        default="",
        help=(
            "Comma-separated task names that enable title-like query anchors such as 'Miller v. California' "
            "or 'Here Comes the Boom'."
        ),
    )
    parser.add_argument(
        "--ours_grounding_verifier_tasks",
        default="",
        help=(
            "Comma-separated extractive QA task names that trigger a full-KV retry when the generated short "
            "answer is not lexically grounded in the retained sparse context."
        ),
    )
    parser.add_argument(
        "--ours_support_window_verifier_tasks",
        default="",
        help=(
            "Comma-separated extractive QA task names that require the generated short answer to co-occur "
            "with query evidence terms in a local retained-context window."
        ),
    )
    parser.add_argument(
        "--ours_support_window_radius_words",
        type=int,
        default=80,
        help="Word radius around an answer candidate used by the support-window verifier.",
    )
    parser.add_argument(
        "--ours_support_window_min_query_terms",
        type=int,
        default=1,
        help="Minimum query evidence terms required near the answer candidate.",
    )
    parser.add_argument(
        "--ours_bm25_mix",
        type=float,
        default=0.70,
        help="For BM25-flow scorers, mix this fraction of context-local BM25 into the lexical component.",
    )
    parser.add_argument("--ours_bm25_k1", type=float, default=1.2)
    parser.add_argument("--ours_bm25_b", type=float, default=0.75)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument(
        "--prompt_wrapper",
        choices=["none", "llama3"],
        default="none",
        help="Wrap the full prompt while preserving context page spans. Use llama3 to match KVCache-Factory LongBench.",
    )
    parser.add_argument(
        "--force_no_chat_tasks",
        default="",
        help="Comma-separated task names that disable chat wrapping even when --prompt_wrapper llama3 is active.",
    )
    parser.add_argument("--longbench_zip_path", default="")
    parser.add_argument("--hf_cache_dir", default="/home/fdong/ymluo/hf_cache")
    parser.add_argument("--lm_eval_path", default="/home/fdong/lm-evaluation-harness")
    parser.add_argument("--ruler_lengths", default="4096")
    parser.add_argument("--log_every", type=int, default=1)
    return Config(**vars(parser.parse_args()))


def parse_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def cache_to_legacy(past_key_values: Any) -> tuple[Any, ...]:
    if past_key_values is None:
        return tuple()
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if isinstance(past_key_values, tuple):
        return past_key_values
    raise TypeError(f"Unsupported cache type: {type(past_key_values)!r}")


def legacy_to_cache_like(legacy: tuple[Any, ...], template: Any) -> Any:
    from_legacy_cache = getattr(type(template), "from_legacy_cache", None)
    if callable(from_legacy_cache):
        return from_legacy_cache(legacy)
    if isinstance(template, list):
        return list(legacy)
    return legacy


def gather_past_key_values(past_key_values: Any, keep_indices: list[int]) -> Any:
    legacy = cache_to_legacy(past_key_values)
    gathered_layers = []
    for layer_cache in legacy:
        key_states, value_states = layer_cache[:2]
        idx = torch.tensor(keep_indices, dtype=torch.long, device=key_states.device)
        gathered_key = key_states.index_select(2, idx).contiguous()
        gathered_value = value_states.index_select(2, idx).contiguous()
        if len(layer_cache) > 2:
            gathered_layers.append((gathered_key, gathered_value, *layer_cache[2:]))
        else:
            gathered_layers.append((gathered_key, gathered_value))
    return legacy_to_cache_like(tuple(gathered_layers), past_key_values)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(truth_tokens))
    return 2 * precision * recall / max(precision + recall, 1e-12)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    target = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for number in numbers if str(number) == str(target)) / len(numbers)


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for number in numbers if str(number) == str(ground_truth)) / len(numbers)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    truth = normalize_answer(ground_truth).split()
    if not pred or not truth:
        return 0.0
    previous = [0] * (len(truth) + 1)
    for token in pred:
        current = [0]
        for j, truth_token in enumerate(truth, start=1):
            if token == truth_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ruler_string_match(prediction: str, answers: list[str]) -> float:
    lowered = prediction.lower()
    return sum(1.0 if answer.lower() in lowered else 0.0 for answer in answers) / max(1, len(answers))


def ruler_string_match_part(prediction: str, answers: list[str]) -> float:
    lowered = prediction.lower()
    return 1.0 if any(answer.lower() in lowered for answer in answers) else 0.0


def code_sim_score(prediction: str, ground_truth: str) -> float:
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return SequenceMatcher(None, line, ground_truth).ratio()
    return 0.0


def classification_score(prediction: str, ground_truth: str, all_classes: list[str]) -> float:
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [match for match in matches if not (match in ground_truth and match != ground_truth)]
    return 1.0 / len(matches) if ground_truth in matches and matches else 0.0


def score_prediction(metric: str, prediction: str, answers: list[str], all_classes: list[str] | None = None) -> float:
    if metric == "ruler_string_match":
        return ruler_string_match(prediction, answers)
    if metric == "ruler_string_match_part":
        return ruler_string_match_part(prediction, answers)
    if metric in {"classification", "qa_f1", "rouge_l"}:
        prediction = prediction.lstrip("\n").split("\n")[0] if metric == "classification" else prediction
    scores = []
    for answer in answers:
        if metric == "qa_f1":
            scores.append(qa_f1_score(prediction, answer))
        elif metric == "retrieval":
            scores.append(retrieval_score(prediction, answer))
        elif metric == "count":
            scores.append(count_score(prediction, answer))
        elif metric == "rouge_l":
            scores.append(rouge_l_score(prediction, answer))
        elif metric == "classification":
            scores.append(classification_score(prediction, answer, all_classes or []))
        elif metric == "code_sim":
            scores.append(code_sim_score(prediction, answer))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return max(scores) if scores else 0.0


def ensure_longbench_zip(config: Config) -> Path:
    if config.longbench_zip_path:
        path = Path(config.longbench_zip_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=config.hf_cache_dir,
        )
    )


def load_longbench_examples(config: Config) -> list[Example]:
    zip_path = ensure_longbench_zip(config)
    rng = random.Random(config.seed)
    examples: list[Example] = []
    with zipfile.ZipFile(zip_path) as archive:
        for task in parse_list(config.longbench_tasks):
            if task not in LONG_BENCH_PROMPTS:
                raise ValueError(f"Unsupported LongBench task {task}; supported={sorted(LONG_BENCH_PROMPTS)}")
            info = LONG_BENCH_PROMPTS[task]
            name = f"data/{task}.jsonl"
            if name not in archive.namelist():
                raise FileNotFoundError(f"{name} not found in {zip_path}")
            rows = [json.loads(line) for line in archive.open(name).read().decode("utf-8").splitlines() if line.strip()]
            if len(rows) > config.max_samples_per_task:
                rows = rows[: config.max_samples_per_task]
            rng.shuffle(rows)
            rows = sorted(rows, key=lambda row: str(row.get("_id", "")))[: config.max_samples_per_task]
            for row in rows:
                max_new = int(info["max_new_tokens"])
                if config.max_new_tokens_override > 0:
                    max_new = min(max_new, config.max_new_tokens_override)
                examples.append(
                    Example(
                        benchmark="longbench",
                        task=task,
                        sample_id=str(row.get("_id", len(examples))),
                        context=str(row["context"]),
                        query=str(row["input"]),
                        answers=[str(answer) for answer in row["answers"]],
                        prefix_template=str(info["prefix"]),
                        suffix_template=str(info["suffix"]),
                        metric=str(info["metric"]),
                        max_new_tokens=max_new,
                        length=int(row.get("length", 0) or 0),
                        all_classes=[str(item) for item in (row.get("all_classes") or [])],
                        no_chat=bool(info.get("no_chat", False)),
                    )
                )
    return examples


def load_ruler_examples(config: Config, tokenizer_name: str) -> list[Example]:
    lm_eval_path = Path(config.lm_eval_path)
    if not lm_eval_path.exists():
        raise FileNotFoundError(f"lm-evaluation-harness path not found: {lm_eval_path}")
    if str(lm_eval_path) not in sys.path:
        sys.path.insert(0, str(lm_eval_path))

    from lm_eval.tasks.ruler.common_utils import get_tokenizer  # noqa: E402
    from lm_eval.tasks.ruler.cwe_utils import sys_word_pair_random  # noqa: E402
    from lm_eval.tasks.ruler.fwe_utils import sys_kwext  # noqa: E402
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE  # noqa: E402
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack  # noqa: E402
    from lm_eval.tasks.ruler.qa_utils import generate_samples as generate_qa_samples  # noqa: E402
    from lm_eval.tasks.ruler.qa_utils import read_hotpotqa, read_squad  # noqa: E402
    from lm_eval.tasks.ruler.vt_utils import sys_vartrack_w_noise_random  # noqa: E402

    lengths = [int(item) for item in parse_list(config.ruler_lengths)]
    tokenizer = get_tokenizer(pretrained=tokenizer_name)
    examples: list[Example] = []

    def split_ruler_prompt(text: str, gen_prefix: str = "") -> tuple[str, str]:
        question_markers = ["\nQuestion:", " Question:", "\nWhat is ", "\nWhat are "]
        split_at = -1
        for marker in question_markers:
            split_at = max(split_at, text.rfind(marker))
        if split_at >= 0:
            context_prompt = text[:split_at]
            suffix = text[split_at:]
        else:
            last_line_start = text.rfind("\n")
            last_line = text[last_line_start + 1 :] if last_line_start >= 0 else text
            if last_line_start >= 0 and "?" in last_line:
                context_prompt = text[:last_line_start]
                suffix = text[last_line_start:]
            else:
                context_prompt = text
                suffix = ""
        if gen_prefix:
            suffix = suffix.rstrip() + "\n" + gen_prefix.strip()
        return context_prompt, suffix

    def append_rows(task: str, length: int, rows: list[dict[str, Any]], max_new: int, metric: str) -> None:
        if config.max_new_tokens_override > 0:
            max_new = min(max_new, config.max_new_tokens_override)
        for idx, row in enumerate(rows[: config.max_samples_per_task]):
            context_prompt, suffix = split_ruler_prompt(str(row["input"]), str(row.get("gen_prefix", "")))
            outputs = row["outputs"]
            if isinstance(outputs, str):
                answers = [outputs]
            else:
                answers = [str(answer) for answer in outputs]
            examples.append(
                Example(
                    benchmark="ruler",
                    task=f"{task}_{length}",
                    sample_id=f"{task}_{length}_{idx}",
                    context=context_prompt,
                    query=suffix,
                    answers=answers,
                    prefix_template="",
                    suffix_template=suffix,
                    metric=metric,
                    max_new_tokens=max_new,
                    length=length,
                    all_classes=[],
                )
            )

    for task in parse_list(config.ruler_tasks):
        for length in lengths:
            if task == "niah_single_1":
                rows = generate_samples(
                    get_haystack(type_haystack="repeat"),
                    max_seq_length=length,
                    template=TEMPLATE,
                    type_haystack="repeat",
                    type_needle_k="words",
                    type_needle_v="numbers",
                    num_samples=config.max_samples_per_task,
                    TOKENIZER=tokenizer,
                )
                append_rows(task, length, rows, 128, "ruler_string_match")
            elif task == "niah_single_2":
                rows = generate_samples(
                    get_haystack(type_haystack="essay"),
                    max_seq_length=length,
                    template=TEMPLATE,
                    type_haystack="essay",
                    type_needle_k="words",
                    type_needle_v="numbers",
                    num_samples=config.max_samples_per_task,
                    TOKENIZER=tokenizer,
                )
                append_rows(task, length, rows, 128, "ruler_string_match")
            elif task == "niah_multikey_1":
                rows = generate_samples(
                    get_haystack(type_haystack="essay"),
                    max_seq_length=length,
                    template=TEMPLATE,
                    type_haystack="essay",
                    type_needle_k="words",
                    type_needle_v="numbers",
                    num_needle_k=4,
                    num_samples=config.max_samples_per_task,
                    TOKENIZER=tokenizer,
                )
                append_rows(task, length, rows, 128, "ruler_string_match")
            elif task == "niah_multivalue":
                rows = generate_samples(
                    get_haystack(type_haystack="essay"),
                    max_seq_length=length,
                    template=TEMPLATE,
                    type_haystack="essay",
                    type_needle_k="words",
                    type_needle_v="numbers",
                    num_needle_v=4,
                    num_samples=config.max_samples_per_task,
                    TOKENIZER=tokenizer,
                )
                append_rows(task, length, rows, 128, "ruler_string_match")
            elif task == "niah_multiquery":
                rows = generate_samples(
                    get_haystack(type_haystack="essay"),
                    max_seq_length=length,
                    template=TEMPLATE,
                    type_haystack="essay",
                    type_needle_k="words",
                    type_needle_v="numbers",
                    num_needle_q=4,
                    num_samples=config.max_samples_per_task,
                    TOKENIZER=tokenizer,
                )
                append_rows(task, length, rows, 128, "ruler_string_match")
            elif task == "vt":
                icl_example = sys_vartrack_w_noise_random(
                    tokenizer=tokenizer,
                    num_samples=1,
                    max_seq_length=500,
                    incremental=5,
                )[0]
                rows = sys_vartrack_w_noise_random(
                    tokenizer=tokenizer,
                    num_samples=config.max_samples_per_task,
                    max_seq_length=length,
                    icl_example=icl_example,
                )
                append_rows(task, length, rows, 30, "ruler_string_match")
            elif task == "cwe":
                rows = sys_word_pair_random(
                    num_samples=config.max_samples_per_task,
                    max_seq_length=length,
                    tokenizer=tokenizer,
                )
                append_rows(task, length, rows, 120, "ruler_string_match")
            elif task == "fwe":
                rows = sys_kwext(
                    tokenizer=tokenizer,
                    max_seq_length=length,
                    num_samples=config.max_samples_per_task,
                )
                append_rows(task, length, rows, 50, "ruler_string_match")
            elif task == "qa_squad":
                try:
                    qas, docs = read_squad()
                except Exception as exc:
                    print(f"[skip-ruler-task] qa_squad download/read failed: {exc}", flush=True)
                    continue
                rows = generate_qa_samples(
                    tokenizer=tokenizer,
                    docs=docs,
                    qas=qas,
                    num_samples=config.max_samples_per_task,
                    tokens_to_generate=32,
                    max_seq_length=length,
                )
                append_rows(task, length, rows, 32, "ruler_string_match_part")
            elif task == "qa_hotpot":
                try:
                    qas, docs = read_hotpotqa()
                except Exception as exc:
                    print(f"[skip-ruler-task] qa_hotpot download/read failed: {exc}", flush=True)
                    continue
                rows = generate_qa_samples(
                    tokenizer=tokenizer,
                    docs=docs,
                    qas=qas,
                    num_samples=config.max_samples_per_task,
                    tokens_to_generate=32,
                    max_seq_length=length,
                )
                append_rows(task, length, rows, 32, "ruler_string_match_part")
            else:
                raise ValueError(
                    "This runner supports ruler tasks: niah_single_1, niah_single_2, niah_multikey_1, "
                    "niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_squad, qa_hotpot"
                )
    return examples


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def trim_context_to_tokens(tokenizer: Any, context: str, max_context_tokens: int) -> str:
    ids = token_ids(tokenizer, context)
    if max_context_tokens <= 0 or len(ids) <= max_context_tokens:
        return context
    half = max_context_tokens // 2
    kept = ids[:half] + ids[-(max_context_tokens - half) :]
    return tokenizer.decode(kept, skip_special_tokens=False)


def split_units(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    pieces = [piece.strip() for piece in re.split(r"(?<=[.;!?])\s+", text) if piece.strip()]
    return pieces if pieces else [text]


def make_pages(tokenizer: Any, context: str, prefix_token_count: int, page_tokens: int) -> list[Page]:
    pages: list[Page] = []
    current_text: list[str] = []
    current_ids: list[int] = []
    cursor = prefix_token_count

    def flush_current() -> None:
        nonlocal current_text, current_ids, cursor
        if not current_ids:
            return
        text = "\n\n".join(current_text)
        pages.append(Page(len(pages), text, cursor, cursor + len(current_ids)))
        cursor += len(current_ids)
        current_text = []
        current_ids = []

    for unit in split_units(context):
        unit_ids = token_ids(tokenizer, unit + "\n\n")
        if len(unit_ids) > page_tokens:
            flush_current()
            for start in range(0, len(unit_ids), page_tokens):
                chunk_ids = unit_ids[start : start + page_tokens]
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                pages.append(Page(len(pages), chunk_text, cursor, cursor + len(chunk_ids)))
                cursor += len(chunk_ids)
            continue
        if current_ids and len(current_ids) + len(unit_ids) > page_tokens:
            flush_current()
        current_text.append(unit)
        current_ids.extend(unit_ids)
    flush_current()
    if not pages:
        context_ids = token_ids(tokenizer, context)
        pages.append(Page(0, context, cursor, cursor + len(context_ids)))
    return pages


def build_bundle(tokenizer: Any, example: Example, config: Config) -> tuple[PromptBundle, list[Page], str, str, str]:
    context = trim_context_to_tokens(tokenizer, example.context, config.max_context_tokens)
    prefix_text = example.prefix_template
    suffix_text = example.suffix_template.format(input=example.query)
    force_no_chat = example.task in {item.strip() for item in config.force_no_chat_tasks.split(",") if item.strip()}
    if config.prompt_wrapper == "llama3" and not example.no_chat and not force_no_chat:
        prefix_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prefix_text
        suffix_text = suffix_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prefix_ids = token_ids(tokenizer, prefix_text)
    context_ids = token_ids(tokenizer, context)
    suffix_ids = token_ids(tokenizer, suffix_text)
    prefix_context_ids = prefix_ids + context_ids
    prompt_ids = prefix_context_ids + suffix_ids
    pages = make_pages(tokenizer, context, len(prefix_ids), config.page_tokens)
    page_spans = {page.page_id: (max(len(prefix_ids), page.token_start), min(len(prefix_context_ids), page.token_end)) for page in pages}
    bundle = PromptBundle(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        prefix_token_count=len(prefix_ids),
        context_token_start=len(prefix_ids),
        query_start=len(prefix_context_ids),
        suffix_token_count=len(suffix_ids),
        page_spans=page_spans,
    )
    return bundle, pages, prefix_text, context, suffix_text


def word_counter(text: str) -> Counter[str]:
    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
    return Counter(word for word in words if word not in STOPWORDS)


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Z0-9_-]{2,}\b", text):
        entities.add(match.group(0).lower())
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text):
        span = match.group(0).lower()
        if span not in STOPWORDS:
            entities.add(span)
    for match in re.finditer(r"\b\d{2,}\b", text):
        entities.add(match.group(0))
    return entities


def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@torch.inference_mode()
def static_text_embeddings(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_tokens: int,
) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    vectors = []
    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )["input_ids"].to(device)
        if ids.numel() == 0:
            vectors.append(torch.zeros(embedding.weight.shape[-1], dtype=torch.float32, device=device))
            continue
        token_embeddings = embedding(ids).float()
        vector = token_embeddings.mean(dim=1).squeeze(0)
        vector = F.normalize(vector, dim=0)
        vectors.append(vector)
    return torch.stack(vectors, dim=0)


@torch.inference_mode()
def late_interaction_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    query: str,
    page_texts: list[str],
    max_tokens: int,
) -> list[float]:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device

    def token_vectors(text: str, limit: int) -> torch.Tensor:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=limit,
            return_tensors="pt",
        )["input_ids"].to(device)
        if ids.numel() == 0:
            return torch.empty(0, embedding.weight.shape[-1], dtype=torch.float32, device=device)
        vectors = embedding(ids).float().squeeze(0)
        return F.normalize(vectors, dim=-1)

    query_vectors = token_vectors(query, min(64, max_tokens))
    if query_vectors.numel() == 0:
        return [0.0 for _ in page_texts]

    scores: list[float] = []
    for text in page_texts:
        page_vectors = token_vectors(text, max_tokens)
        if page_vectors.numel() == 0:
            scores.append(0.0)
            continue
        sim = torch.matmul(query_vectors, page_vectors.transpose(0, 1))
        scores.append(float(sim.max(dim=-1).values.mean().item()))
    return scores


def task_is_global(example: Example) -> bool:
    info = LONG_BENCH_PROMPTS.get(example.task, {})
    return bool(info.get("global_task", False)) or example.metric == "rouge_l"


def extract_query_anchors(query: str, include_title_phrases: bool = False) -> list[str]:
    anchors: list[str] = []
    patterns = [
        r"\bfor\s+(.+?)\s+mentioned\b",
        r"\bvalue\s+([A-Za-z0-9_-]+)\b",
        r"\bassigned\s+the\s+value\s+([A-Za-z0-9_-]+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE | re.DOTALL):
            raw = match.group(1)
            raw = re.sub(r"\bin\s+the\s+provided\s+text\b.*", "", raw, flags=re.IGNORECASE)
            for part in re.split(r",|\band\b|/|;", raw):
                part = part.strip(" .?\"'():\n\t")
                if 2 <= len(part) <= 80:
                    anchors.append(part.lower())
    if include_title_phrases:
        title_phrase_pattern = (
            r"\b(?:[A-Z][A-Za-z0-9'-]*|[A-Z]{2,})"
            r"(?:\s+(?:v\.?|vs\.?|of|the|and|for|in|on|at|to|from|by|with|"
            r"[A-Z][A-Za-z0-9'-]*|[A-Z]{2,})){1,7}\b"
        )
        question_starts = {"who", "what", "when", "where", "which", "why", "how"}
        for match in re.finditer(title_phrase_pattern, query):
            phrase = match.group(0).strip(" .?\"'():\n\t")
            pieces = [phrase]
            split_parts = [
                part.strip(" .?\"'():\n\t")
                for part in re.split(r"\b(?:in|from|by|with|to)\b", phrase)
                if part.strip(" .?\"'():\n\t")
            ]
            if len(split_parts) > 1:
                pieces.extend(split_parts)
            for piece in pieces:
                words = [word.lower().strip(".") for word in re.findall(r"[A-Za-z0-9'-]+", piece)]
                if not words or words[0] in question_starts:
                    continue
                content = [word for word in words if word not in STOPWORDS]
                if len(content) < 2:
                    continue
                if 4 <= len(piece) <= 96:
                    anchors.append(piece.lower())
    for match in re.finditer(r"\b[A-Z]{3,}\b", query):
        anchors.append(match.group(0).lower())
    deduped: list[str] = []
    for anchor in anchors:
        if anchor not in deduped:
            deduped.append(anchor)
    return deduped[:12]


def flow_enabled(config: Config) -> bool:
    return config.ours_scorer in {
        "hybrid_late_mmr_flow",
        "hybrid_late_mmr_bm25_flow",
        "hybrid_late_mmr_bridge_flow",
        "hybrid_late_mmr_multiscale_flow",
        "hybrid_late_mmr_multiscale_bm25_flow",
        "hybrid_late_mmr_multiscale_bridge_flow",
        "hybrid_late_mmr_multiscale_task_bridge_flow",
        "hybrid_late_mmr_idf_flow",
        "hybrid_late_mmr_multiscale_idf_flow",
        "hybrid_late_mmr_idf_spread_flow",
        "hybrid_late_mmr_multiscale_idf_spread_flow",
    }


def multiscale_enabled(config: Config) -> bool:
    return config.ours_scorer in {
        "hybrid_late_mmr_multiscale_flow",
        "hybrid_late_mmr_multiscale_bm25_flow",
        "hybrid_late_mmr_multiscale_bridge_flow",
        "hybrid_late_mmr_multiscale_task_bridge_flow",
        "hybrid_late_mmr_multiscale_idf_flow",
        "hybrid_late_mmr_multiscale_idf_spread_flow",
    }


def idf_enabled(config: Config) -> bool:
    return config.ours_scorer in {
        "hybrid_late_mmr_idf_flow",
        "hybrid_late_mmr_multiscale_idf_flow",
        "hybrid_late_mmr_idf_spread_flow",
        "hybrid_late_mmr_multiscale_idf_spread_flow",
    }


def spread_enabled(config: Config) -> bool:
    return config.ours_scorer in {"hybrid_late_mmr_idf_spread_flow", "hybrid_late_mmr_multiscale_idf_spread_flow"}


TASK_POLICY_KEYS = {
    "budget_tokens",
    "sink_tokens",
    "recent_tokens",
    "page_tokens",
    "ours_scorer",
    "ours_coverage_mmr_weight",
    "ours_coverage_mmr_max_terms",
    "ours_coverage_certificate_tasks",
    "ours_coverage_certificate_budget_fraction",
    "ours_coverage_certificate_min_terms",
    "ours_coverage_risk_tasks",
    "ours_coverage_risk_min_recall",
    "ours_coverage_risk_min_terms",
    "ours_coverage_risk_budget_tokens",
    "anchor_pages_per_key",
    "ours_bridge_tasks",
    "ours_flow_neighbor_radius",
    "ours_flow_neighbor_budget_fraction",
    "ours_flow_neighbor_min_score",
    "ours_flow_score_smooth_weight",
    "ours_flow_anchor_boost",
    "ours_multiscale_group_pages",
    "ours_multiscale_weight",
    "ours_idf_mix",
    "ours_spread_budget_fraction",
    "ours_spread_gap_threshold",
    "ours_spread_bins",
    "ours_spread_min_score",
    "ours_bridge_budget_fraction",
    "ours_bridge_min_score",
    "ours_bridge_max_terms",
    "ours_graph_bridge_tasks",
    "ours_graph_bridge_budget_fraction",
    "ours_graph_bridge_seed_pages",
    "ours_graph_bridge_max_terms",
    "ours_graph_bridge_min_score",
    "ours_full_fallback_tasks",
    "ours_label_support_tasks",
    "ours_label_backtrack_pages",
    "ours_label_budget_fraction",
    "ours_passage_closure_tasks",
    "ours_passage_closure_budget_fraction",
    "ours_passage_closure_radius_pages",
    "ours_structured_fingerprint_tasks",
    "ours_structured_fingerprint_budget_fraction",
    "ours_direct_structured_answer_tasks",
    "ours_short_decode_tasks",
    "ours_short_decode_max_tokens",
    "ours_output_verifier_tasks",
    "ours_output_probe_max_tokens",
    "ours_retry_budget_tokens",
    "ours_score_risk_tasks",
    "ours_score_risk_budget_tokens",
    "ours_score_risk_min_gap2",
    "ours_score_risk_min_gap3",
    "ours_score_risk_max_gap2",
    "ours_score_risk_max_gap3",
    "ours_score_risk_max_entropy",
    "ours_score_risk_entropy_at_most",
    "ours_score_risk_min_top_score",
    "ours_score_risk_raw_prefix_at_most",
    "ours_score_risk_raw_prefix_at_least",
    "ours_score_risk_linear_threshold",
    "ours_score_risk_gap2_weight",
    "ours_score_risk_gap3_weight",
    "ours_score_risk_top_score_weight",
    "ours_budget_ladder_tasks",
    "ours_budget_ladder_tokens",
    "ours_budget_ladder_gap2_thresholds",
    "ours_budget_ladder_entropy_thresholds",
    "ours_budget_ladder_top_score_thresholds",
    "ours_consistency_verifier_tasks",
    "ours_consistency_budget_tokens",
    "ours_consistency_probe_max_tokens",
    "ours_consistency_requires_score_risk",
    "ours_title_anchor_tasks",
    "ours_grounding_verifier_tasks",
    "ours_support_window_verifier_tasks",
    "ours_support_window_radius_words",
    "ours_support_window_min_query_terms",
    "ours_bm25_mix",
    "ours_bm25_k1",
    "ours_bm25_b",
}


@lru_cache(maxsize=32)
def parse_task_policy_spec(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        return {}
    text = spec
    try:
        path = Path(spec)
        if path.exists():
            text = path.read_text(encoding="utf-8")
    except OSError:
        text = spec
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict):
        payload = payload["tasks"]
    if not isinstance(payload, dict):
        raise ValueError("--ours_task_policy_json must decode to a dict or {'tasks': dict}")
    return payload


def config_for_example(config: Config, example: Example) -> Config:
    policy = parse_task_policy_spec(config.ours_task_policy_json)
    if not policy:
        return config
    merged: dict[str, Any] = {}
    matched_keys: list[str] = []
    if "*" in policy:
        matched_keys.append("*")
    for key in policy:
        if key in {"*", example.task}:
            continue
        if any(char in key for char in "*?[]") and fnmatch.fnmatchcase(example.task, key):
            matched_keys.append(key)
    if example.task in policy:
        matched_keys.append(example.task)
    for key in matched_keys:
        value = policy.get(key)
        if isinstance(value, int):
            value = {"budget_tokens": value}
        if isinstance(value, dict):
            merged.update(value)
    if not merged:
        return config

    overrides: dict[str, Any] = {}
    for key, value in merged.items():
        if key == "scorer":
            normalized_key = "ours_scorer"
        elif key == "coverage_mmr_weight":
            normalized_key = "ours_coverage_mmr_weight"
        elif key == "coverage_mmr_max_terms":
            normalized_key = "ours_coverage_mmr_max_terms"
        elif key == "coverage_certificate_budget_fraction":
            normalized_key = "ours_coverage_certificate_budget_fraction"
        elif key == "coverage_certificate_min_terms":
            normalized_key = "ours_coverage_certificate_min_terms"
        elif key == "coverage_risk_min_recall":
            normalized_key = "ours_coverage_risk_min_recall"
        elif key == "coverage_risk_min_terms":
            normalized_key = "ours_coverage_risk_min_terms"
        elif key == "coverage_risk_budget_tokens":
            normalized_key = "ours_coverage_risk_budget_tokens"
        elif key == "retry_budget_tokens":
            normalized_key = "ours_retry_budget_tokens"
        elif key == "consistency_budget_tokens":
            normalized_key = "ours_consistency_budget_tokens"
        elif key == "consistency_probe_max_tokens":
            normalized_key = "ours_consistency_probe_max_tokens"
        elif key == "consistency_requires_score_risk":
            normalized_key = "ours_consistency_requires_score_risk"
        elif key == "score_risk_budget_tokens":
            normalized_key = "ours_score_risk_budget_tokens"
        elif key == "score_risk_min_gap2":
            normalized_key = "ours_score_risk_min_gap2"
        elif key == "score_risk_min_gap3":
            normalized_key = "ours_score_risk_min_gap3"
        elif key == "score_risk_max_gap2":
            normalized_key = "ours_score_risk_max_gap2"
        elif key == "score_risk_max_gap3":
            normalized_key = "ours_score_risk_max_gap3"
        elif key == "score_risk_max_entropy":
            normalized_key = "ours_score_risk_max_entropy"
        elif key == "score_risk_entropy_at_most":
            normalized_key = "ours_score_risk_entropy_at_most"
        elif key == "score_risk_min_top_score":
            normalized_key = "ours_score_risk_min_top_score"
        elif key == "score_risk_raw_prefix_at_most":
            normalized_key = "ours_score_risk_raw_prefix_at_most"
        elif key == "score_risk_raw_prefix_at_least":
            normalized_key = "ours_score_risk_raw_prefix_at_least"
        elif key == "score_risk_linear_threshold":
            normalized_key = "ours_score_risk_linear_threshold"
        elif key == "score_risk_gap2_weight":
            normalized_key = "ours_score_risk_gap2_weight"
        elif key == "score_risk_gap3_weight":
            normalized_key = "ours_score_risk_gap3_weight"
        elif key == "score_risk_top_score_weight":
            normalized_key = "ours_score_risk_top_score_weight"
        elif key == "budget_ladder_tokens":
            normalized_key = "ours_budget_ladder_tokens"
        elif key == "budget_ladder_gap2_thresholds":
            normalized_key = "ours_budget_ladder_gap2_thresholds"
        elif key == "budget_ladder_entropy_thresholds":
            normalized_key = "ours_budget_ladder_entropy_thresholds"
        elif key == "budget_ladder_top_score_thresholds":
            normalized_key = "ours_budget_ladder_top_score_thresholds"
        elif key == "graph_bridge_budget_fraction":
            normalized_key = "ours_graph_bridge_budget_fraction"
        elif key == "graph_bridge_seed_pages":
            normalized_key = "ours_graph_bridge_seed_pages"
        elif key == "graph_bridge_max_terms":
            normalized_key = "ours_graph_bridge_max_terms"
        elif key == "graph_bridge_min_score":
            normalized_key = "ours_graph_bridge_min_score"
        elif key == "passage_closure_budget_fraction":
            normalized_key = "ours_passage_closure_budget_fraction"
        elif key == "passage_closure_radius_pages":
            normalized_key = "ours_passage_closure_radius_pages"
        elif key == "structured_fingerprint_budget_fraction":
            normalized_key = "ours_structured_fingerprint_budget_fraction"
        elif key == "short_decode_max_tokens":
            normalized_key = "ours_short_decode_max_tokens"
        elif key == "support_window_radius_words":
            normalized_key = "ours_support_window_radius_words"
        elif key == "support_window_min_query_terms":
            normalized_key = "ours_support_window_min_query_terms"
        else:
            normalized_key = key
        if normalized_key not in TASK_POLICY_KEYS:
            continue
        current = getattr(config, normalized_key)
        if isinstance(current, int):
            overrides[normalized_key] = int(value)
        elif isinstance(current, float):
            overrides[normalized_key] = float(value)
        else:
            overrides[normalized_key] = str(value)

    if "bridge" in merged:
        if bool(merged["bridge"]):
            overrides.setdefault("ours_scorer", "hybrid_late_mmr_multiscale_task_bridge_flow")
            overrides["ours_bridge_tasks"] = example.task
        else:
            if overrides.get("ours_scorer", config.ours_scorer) == "hybrid_late_mmr_multiscale_task_bridge_flow":
                overrides["ours_bridge_tasks"] = ""

    if "graph_bridge" in merged:
        overrides["ours_graph_bridge_tasks"] = example.task if bool(merged["graph_bridge"]) else ""

    if "full_fallback" in merged:
        overrides["ours_full_fallback_tasks"] = example.task if bool(merged["full_fallback"]) else ""

    if "label_support" in merged:
        overrides["ours_label_support_tasks"] = example.task if bool(merged["label_support"]) else ""

    if "passage_closure" in merged:
        overrides["ours_passage_closure_tasks"] = example.task if bool(merged["passage_closure"]) else ""

    if "structured_fingerprint" in merged:
        overrides["ours_structured_fingerprint_tasks"] = example.task if bool(merged["structured_fingerprint"]) else ""

    if "direct_structured_answer" in merged:
        overrides["ours_direct_structured_answer_tasks"] = (
            example.task if bool(merged["direct_structured_answer"]) else ""
        )

    if "short_decode" in merged:
        overrides["ours_short_decode_tasks"] = example.task if bool(merged["short_decode"]) else ""

    if "output_verifier" in merged:
        overrides["ours_output_verifier_tasks"] = example.task if bool(merged["output_verifier"]) else ""

    if "score_risk" in merged:
        overrides["ours_score_risk_tasks"] = example.task if bool(merged["score_risk"]) else ""

    if "budget_ladder" in merged:
        overrides["ours_budget_ladder_tasks"] = example.task if bool(merged["budget_ladder"]) else ""

    if "coverage_risk" in merged:
        overrides["ours_coverage_risk_tasks"] = example.task if bool(merged["coverage_risk"]) else ""

    if "coverage_certificate" in merged:
        overrides["ours_coverage_certificate_tasks"] = example.task if bool(merged["coverage_certificate"]) else ""

    if "consistency_verifier" in merged:
        overrides["ours_consistency_verifier_tasks"] = example.task if bool(merged["consistency_verifier"]) else ""

    if "grounding_verifier" in merged:
        overrides["ours_grounding_verifier_tasks"] = example.task if bool(merged["grounding_verifier"]) else ""

    if "title_anchor" in merged:
        overrides["ours_title_anchor_tasks"] = example.task if bool(merged["title_anchor"]) else ""

    if "support_window_verifier" in merged:
        overrides["ours_support_window_verifier_tasks"] = (
            example.task if bool(merged["support_window_verifier"]) else ""
        )

    return replace(config, **overrides)


def bridge_enabled(config: Config) -> bool:
    return config.ours_scorer in {
        "hybrid_late_mmr_bridge_flow",
        "hybrid_late_mmr_multiscale_bridge_flow",
        "hybrid_late_mmr_multiscale_task_bridge_flow",
    }


def bridge_active_for_example(config: Config, example: Example) -> bool:
    if not bridge_enabled(config):
        return False
    if config.ours_scorer != "hybrid_late_mmr_multiscale_task_bridge_flow":
        return True
    tasks = {item.strip() for item in config.ours_bridge_tasks.split(",") if item.strip()}
    return example.task in tasks


def graph_bridge_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_graph_bridge_tasks.split(",") if item.strip()}
    return example.task in tasks


def full_fallback_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_full_fallback_tasks.split(",") if item.strip()}
    return example.task in tasks


STRUCTURED_LABEL_RE = re.compile(r"\b(?:Paragraph|Passage)\s+\d+\b", re.IGNORECASE)
PASSAGE_HEADER_RE = re.compile(r"\bPassage\s+(\d+)\s*:", re.IGNORECASE)


def label_support_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_label_support_tasks.split(",") if item.strip()}
    return example.task in tasks


def passage_closure_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_passage_closure_tasks.split(",") if item.strip()}
    return example.task in tasks


def structured_fingerprint_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_structured_fingerprint_tasks.split(",") if item.strip()}
    return example.task in tasks


def direct_structured_answer_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_direct_structured_answer_tasks.split(",") if item.strip()}
    return example.task in tasks


def short_decode_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_short_decode_tasks.split(",") if item.strip()}
    return example.task in tasks


def page_has_structured_label(page: Page) -> bool:
    return STRUCTURED_LABEL_RE.search(page.text) is not None


def output_verifier_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_output_verifier_tasks.split(",") if item.strip()}
    return example.task in tasks


def title_anchor_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_title_anchor_tasks.split(",") if item.strip()}
    return example.task in tasks


def grounding_verifier_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_grounding_verifier_tasks.split(",") if item.strip()}
    return example.task in tasks


def support_window_verifier_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_support_window_verifier_tasks.split(",") if item.strip()}
    return example.task in tasks


def consistency_verifier_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_consistency_verifier_tasks.split(",") if item.strip()}
    return example.task in tasks


def score_risk_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_score_risk_tasks.split(",") if item.strip()}
    return example.task in tasks


def coverage_risk_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_coverage_risk_tasks.split(",") if item.strip()}
    return example.task in tasks


def coverage_certificate_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_coverage_certificate_tasks.split(",") if item.strip()}
    return example.task in tasks


def budget_ladder_active_for_example(config: Config, example: Example) -> bool:
    tasks = {item.strip() for item in config.ours_budget_ladder_tasks.split(",") if item.strip()}
    return example.task in tasks


def parse_int_csv(spec: str) -> list[int]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(float(item)))
    return out


def parse_float_csv(spec: str) -> list[float]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def ladder_level_le(value: float, thresholds: list[float]) -> int:
    level = 0
    for idx, threshold in enumerate(thresholds):
        if value <= threshold:
            level = max(level, idx + 1)
    return level


def ladder_level_ge(value: float, thresholds: list[float]) -> int:
    level = 0
    for idx, threshold in enumerate(thresholds):
        if value >= threshold:
            level = max(level, idx + 1)
    return level


def budget_ladder_decision(
    example: Example,
    bundle: PromptBundle,
    pages: list[Page],
    config: Config,
) -> dict[str, Any]:
    budgets = sorted({budget for budget in parse_int_csv(config.ours_budget_ladder_tokens) if budget > 0})
    if not budgets:
        return {
            "active": 0,
            "selected_budget": config.budget_tokens,
            "level": 0,
            "reasons": "",
        }
    stats = page_score_stats(pages)
    gap_level = ladder_level_le(stats["gap2"], parse_float_csv(config.ours_budget_ladder_gap2_thresholds))
    entropy_level = ladder_level_ge(stats["entropy"], parse_float_csv(config.ours_budget_ladder_entropy_thresholds))
    top_score_level = ladder_level_le(stats["max"], parse_float_csv(config.ours_budget_ladder_top_score_thresholds))
    level = max(gap_level, entropy_level, top_score_level)
    level = min(level, len(budgets) - 1)
    reasons = []
    if gap_level:
        reasons.append(f"gap2:{stats['gap2']:.4f}->L{gap_level}")
    if entropy_level:
        reasons.append(f"entropy:{stats['entropy']:.4f}->L{entropy_level}")
    if top_score_level:
        reasons.append(f"top:{stats['max']:.4f}->L{top_score_level}")
    selected_budget = budgets[level]
    if selected_budget >= bundle.query_start:
        selected_budget = bundle.query_start
    return {
        "active": 1,
        "selected_budget": selected_budget,
        "level": level,
        "reasons": ";".join(reasons),
        "tokens": ",".join(str(item) for item in budgets),
        "gap2_level": gap_level,
        "entropy_level": entropy_level,
        "top_score_level": top_score_level,
    }


def score_risk_linear_value(stats: dict[str, float], config: Config) -> float:
    return (
        stats["entropy"]
        - config.ours_score_risk_gap2_weight * stats["gap2"]
        - config.ours_score_risk_gap3_weight * stats["gap3"]
        - config.ours_score_risk_top_score_weight * stats["max"]
    )


def score_risk_triggered_for_stats(stats: dict[str, float], config: Config, raw_prefix_tokens: int = -1) -> bool:
    conditions: list[bool] = []
    if config.ours_score_risk_linear_threshold >= 0.0:
        conditions.append(score_risk_linear_value(stats, config) >= config.ours_score_risk_linear_threshold)
    if config.ours_score_risk_min_gap2 >= 0.0:
        conditions.append(stats["gap2"] <= config.ours_score_risk_min_gap2)
    if config.ours_score_risk_min_gap3 >= 0.0:
        conditions.append(stats["gap3"] <= config.ours_score_risk_min_gap3)
    if config.ours_score_risk_max_gap2 >= 0.0:
        conditions.append(stats["gap2"] >= config.ours_score_risk_max_gap2)
    if config.ours_score_risk_max_gap3 >= 0.0:
        conditions.append(stats["gap3"] >= config.ours_score_risk_max_gap3)
    if config.ours_score_risk_max_entropy <= 1.0:
        conditions.append(stats["entropy"] >= config.ours_score_risk_max_entropy)
    if config.ours_score_risk_entropy_at_most >= 0.0:
        conditions.append(stats["entropy"] <= config.ours_score_risk_entropy_at_most)
    if config.ours_score_risk_min_top_score >= 0.0:
        conditions.append(stats["max"] <= config.ours_score_risk_min_top_score)
    if config.ours_score_risk_raw_prefix_at_most >= 0 and raw_prefix_tokens >= 0:
        conditions.append(raw_prefix_tokens <= config.ours_score_risk_raw_prefix_at_most)
    if config.ours_score_risk_raw_prefix_at_least >= 0 and raw_prefix_tokens >= 0:
        conditions.append(raw_prefix_tokens >= config.ours_score_risk_raw_prefix_at_least)
    return bool(conditions) and all(conditions)


def score_risk_triggered_for_pages(pages: list[Page], config: Config, raw_prefix_tokens: int = -1) -> bool:
    stats = page_score_stats(pages)
    return score_risk_triggered_for_stats(stats, config, raw_prefix_tokens=raw_prefix_tokens)


def output_contract_failed(example: Example, prediction: str) -> bool:
    text = prediction.strip()
    if example.task == "passage_retrieval_en":
        return re.search(r"\bParagraph\s+\d+\b", text, flags=re.IGNORECASE) is None
    if example.task == "passage_count":
        return re.search(r"\b\d+\b", text[:64]) is None
    if example.metric == "classification" and example.all_classes:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        first_lower = first_line.lower()
        return not any(str(label).lower() in first_lower for label in example.all_classes)
    if example.metric == "code_sim" or example.task == "repobench-p":
        lowered = text[:256].lower()
        if text.startswith("```"):
            return True
        natural_language_failures = [
            "i'll be happy",
            "i will be happy",
            "happy to help",
            "i can help",
            "i cannot",
            "i can't",
            "not clear",
            "not enough",
            "there are not",
            "however, i",
            "complete the code",
            "stop here",
        ]
        return any(pattern in lowered for pattern in natural_language_failures)
    if example.metric == "qa_f1" and example.task != "qasper":
        lowered = text.lower()
        abstention_patterns = [
            "there is no information",
            "not enough information",
            "cannot determine",
            "can't determine",
            "could not determine",
            "cannot be determined",
            "not specified",
            "not mentioned",
            "not provided",
            "no mention",
            "i couldn't find",
            "i could not find",
            "the passages do not",
            "the passage does not",
            "the context does not",
            "the text does not",
        ]
        if any(pattern in lowered for pattern in abstention_patterns):
            return True
        word_count = len(re.findall(r"[A-Za-z0-9]+", text))
        if word_count > 48 and example.max_new_tokens <= 64:
            return True
    return False


def extract_grounding_candidate(prediction: str) -> str:
    text = prediction.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^(?:answer|the answer is|it is|it's|this is)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\b(?:because|since|as shown|according to)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip(" .;:,\"'()[]{}")
    return text


def selected_context_text(bundle: PromptBundle, pages: list[Page], keep_indices: list[int]) -> str:
    keep = set(keep_indices)
    chunks: list[str] = []
    for page in pages:
        start, end = bundle.page_spans[page.page_id]
        if any(idx in keep for idx in range(start, end)):
            chunks.append(page.text)
    return "\n\n".join(chunks)


def grounding_contract_failed(example: Example, prediction: str, retained_context: str) -> bool:
    if example.metric != "qa_f1" or example.task == "qasper":
        return False
    candidate = extract_grounding_candidate(prediction)
    norm_candidate = normalize_answer(candidate)
    if not norm_candidate:
        return False
    if norm_candidate in {"yes", "no", "unanswerable", "unknown", "none"}:
        return False
    tokens = norm_candidate.split()
    if len(tokens) > 12:
        return False
    if len(tokens) == 1 and (len(tokens[0]) < 3 or tokens[0].isdigit()):
        return False
    norm_context = normalize_answer(retained_context)
    if norm_candidate in norm_context:
        return False
    content_tokens = [token for token in tokens if token not in STOPWORDS and len(token) >= 3]
    if len(content_tokens) >= 2 and all(re.search(rf"\b{re.escape(token)}\b", norm_context) for token in content_tokens):
        return False
    return True


def answer_consistency_failed(example: Example, first_prediction: str, second_prediction: str) -> bool:
    if example.metric != "qa_f1":
        return False
    first = normalize_answer(extract_grounding_candidate(first_prediction))
    second = normalize_answer(extract_grounding_candidate(second_prediction))
    if not first or not second:
        return False
    if first == second:
        return False
    if first in {"yes", "no"} or second in {"yes", "no"}:
        return first != second
    first_tokens = [token for token in first.split() if token not in STOPWORDS]
    second_tokens = [token for token in second.split() if token not in STOPWORDS]
    if not first_tokens or not second_tokens:
        return False
    first_set = set(first_tokens)
    second_set = set(second_tokens)
    overlap = len(first_set & second_set)
    precision = overlap / max(1, len(second_set))
    recall = overlap / max(1, len(first_set))
    f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
    if f1 >= 0.67:
        return False
    if len(first_tokens) >= 2 and len(second_tokens) >= 2 and (first in second or second in first):
        return False
    return True


def bm25_enabled(config: Config) -> bool:
    return config.ours_scorer in {"hybrid_late_mmr_bm25_flow", "hybrid_late_mmr_multiscale_bm25_flow"}


def page_has_anchor_hit(page: Page, anchors: list[str], q_entities: set[str]) -> bool:
    lowered = page.text.lower()
    if any(
        anchor in lowered or anchor.replace("-", " ") in lowered or anchor.replace(" ", "-") in lowered
        for anchor in anchors
    ):
        return True
    if q_entities and q_entities & extract_entities(page.text):
        return True
    return False


def query_coverage_terms_for_example(example: Example, config: Config) -> list[str]:
    query = example.query or example.suffix_template
    auxiliary_terms = {
        "did",
        "do",
        "does",
        "done",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
    }
    raw_terms: list[str] = []
    raw_terms.extend(
        extract_query_anchors(
            query,
            include_title_phrases=title_anchor_active_for_example(config, example),
        )
    )
    raw_terms.extend(sorted(extract_entities(query), key=lambda term: (len(term), term), reverse=True))
    raw_terms.extend(re.findall(r"\b\d+\b", query))
    word_items = list(word_counter(query).items())
    word_items.sort(key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    raw_terms.extend(word for word, _ in word_items)

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        term = raw.lower().strip(" .?\"'():\n\t")
        if not term:
            continue
        if term in STOPWORDS or term in auxiliary_terms:
            continue
        if not term.isdigit() and len(term) < 3:
            continue
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
        if len(deduped) >= max(1, config.ours_coverage_mmr_max_terms):
            break
    return deduped


def page_query_coverage_terms(page: Page, query_terms: list[str]) -> set[str]:
    if not query_terms:
        return set()
    lowered = page.text.lower()
    matched: set[str] = set()
    for term in query_terms:
        variants = {term, term.replace("-", " "), term.replace(" ", "-")}
        if any(" " in variant for variant in variants):
            if any(variant in lowered for variant in variants):
                matched.add(term)
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", lowered):
            matched.add(term)
    return matched


def support_window_contract_failed(example: Example, prediction: str, retained_context: str, config: Config) -> bool:
    if example.metric != "qa_f1" or example.task == "qasper":
        return False
    candidate = extract_grounding_candidate(prediction)
    norm_candidate = normalize_answer(candidate)
    if not norm_candidate or norm_candidate in {"yes", "no", "unanswerable", "unknown", "none"}:
        return False
    candidate_tokens = [token for token in norm_candidate.split() if token not in STOPWORDS and len(token) >= 3]
    if len(candidate_tokens) > 12:
        return False
    if len(candidate_tokens) == 1 and (len(candidate_tokens[0]) < 3 or candidate_tokens[0].isdigit()):
        return False
    query_terms: list[str] = []
    candidate_set = set(candidate_tokens)
    for term in query_coverage_terms_for_example(example, config):
        for token in re.findall(r"[a-z0-9]+", normalize_answer(term)):
            if token in STOPWORDS or token in candidate_set:
                continue
            if not token.isdigit() and len(token) < 3:
                continue
            query_terms.append(token)
    if len(query_terms) < max(1, config.ours_support_window_min_query_terms):
        return False
    context_tokens = re.findall(r"[a-z0-9]+", normalize_answer(retained_context))
    if not context_tokens:
        return False
    query_set = set(query_terms)
    radius = max(8, int(config.ours_support_window_radius_words))
    required = max(1, int(config.ours_support_window_min_query_terms))
    best_overlap = 0
    for idx, token in enumerate(context_tokens):
        if token not in candidate_set:
            continue
        left = max(0, idx - radius)
        right = min(len(context_tokens), idx + radius + 1)
        window = set(context_tokens[left:right])
        overlap = len(query_set & window)
        best_overlap = max(best_overlap, overlap)
        if overlap >= required:
            return False
    # If the answer is not present, the lexical grounding verifier handles it. This verifier only rejects
    # answer mentions that are not locally tied to query evidence.
    if best_overlap == 0 and any(token in context_tokens for token in candidate_set):
        return True
    return False


def apply_flow_scores(example: Example, pages: list[Page], config: Config) -> None:
    if not pages:
        return
    base_scores = [float(page.score) for page in pages]
    anchors = extract_query_anchors(
        example.query or example.suffix_template,
        include_title_phrases=title_anchor_active_for_example(config, example),
    )
    q_entities = extract_entities(example.query or example.suffix_template)
    radius = max(1, config.ours_flow_neighbor_radius)
    smooth = min(0.9, max(0.0, config.ours_flow_score_smooth_weight))
    for idx, page in enumerate(pages):
        left = max(0, idx - radius)
        right = min(len(pages), idx + radius + 1)
        local_scores = base_scores[left:right]
        local_max = max(local_scores) if local_scores else base_scores[idx]
        score = (1.0 - smooth) * base_scores[idx] + smooth * local_max
        if page_has_anchor_hit(page, anchors, q_entities):
            score += max(0.0, config.ours_flow_anchor_boost)
        page.score = float(score)


def apply_multiscale_scores(pages: list[Page], config: Config) -> None:
    if not pages:
        return
    group_pages = max(2, config.ours_multiscale_group_pages)
    weight = min(0.9, max(0.0, config.ours_multiscale_weight))
    base_scores = [float(page.score) for page in pages]
    coarse_scores: list[float] = []
    for start in range(0, len(pages), group_pages):
        group = base_scores[start : start + group_pages]
        if not group:
            continue
        # Max support keeps a fine page if any page in its coarse neighborhood is strong.
        coarse_scores.extend([max(group)] * len(group))
    for page, base, coarse in zip(pages, base_scores, coarse_scores):
        page.score = float((1.0 - weight) * base + weight * coarse)


def add_flow_neighbor_pages(
    keep: set[int],
    bundle: PromptBundle,
    example: Example,
    pages: list[Page],
    center: Page,
    remaining: int,
    config: Config,
    support_tokens_used: int,
) -> tuple[int, int, list[Page]]:
    if remaining <= 0 or config.ours_flow_neighbor_radius <= 0:
        return 0, support_tokens_used, []
    support_cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_flow_neighbor_budget_fraction))
    if support_tokens_used >= support_cap:
        return 0, support_tokens_used, []
    anchors = extract_query_anchors(
        example.query or example.suffix_template,
        include_title_phrases=title_anchor_active_for_example(config, example),
    )
    q_entities = extract_entities(example.query or example.suffix_template)
    candidates: list[tuple[float, int, Page]] = []
    for offset in range(1, config.ours_flow_neighbor_radius + 1):
        for page_id in (center.page_id - offset, center.page_id + offset):
            if page_id < 0 or page_id >= len(pages):
                continue
            page = pages[page_id]
            new_count = sum(
                1
                for idx in range(*bundle.page_spans[page.page_id])
                if bundle.context_token_start <= idx < bundle.query_start and idx not in keep
            )
            if new_count <= 0:
                continue
            if page.score < config.ours_flow_neighbor_min_score and not page_has_anchor_hit(page, anchors, q_entities):
                continue
            candidates.append((page.score, -offset, page))
    candidates.sort(key=lambda item: (item[0], item[1], -item[2].page_id), reverse=True)
    total_added = 0
    added_pages: list[Page] = []
    for _, _, page in candidates:
        if remaining <= 0 or support_tokens_used >= support_cap:
            break
        allowed = min(remaining, support_cap - support_tokens_used)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            continue
        remaining -= added
        total_added += added
        support_tokens_used += added
        added_pages.append(page)
    return total_added, support_tokens_used, added_pages


def add_structured_label_pages(
    keep: set[int],
    bundle: PromptBundle,
    pages: list[Page],
    center: Page,
    remaining: int,
    config: Config,
    label_tokens_used: int,
) -> tuple[int, int, list[Page]]:
    if remaining <= 0 or page_has_structured_label(center):
        return 0, label_tokens_used, []
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_label_budget_fraction))
    if cap <= 0 or label_tokens_used >= cap:
        return 0, label_tokens_used, []
    max_backtrack = max(0, config.ours_label_backtrack_pages)
    candidates: list[Page] = []
    for offset in range(1, max_backtrack + 1):
        page_id = center.page_id - offset
        if page_id < 0:
            break
        page = pages[page_id]
        if page_has_structured_label(page):
            candidates.append(page)
            break
    total_added = 0
    added_pages: list[Page] = []
    for page in candidates:
        if remaining <= 0 or label_tokens_used >= cap:
            break
        allowed = min(remaining, cap - label_tokens_used)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            continue
        remaining -= added
        total_added += added
        label_tokens_used += added
        added_pages.append(page)
    return total_added, label_tokens_used, added_pages


def infer_passage_ids(pages: list[Page]) -> list[int | None]:
    passage_ids: list[int | None] = []
    current: int | None = None
    for page in pages:
        match = PASSAGE_HEADER_RE.search(page.text)
        if match:
            try:
                current = int(match.group(1))
            except ValueError:
                current = None
        passage_ids.append(current)
    return passage_ids


def add_passage_closure_pages(
    keep: set[int],
    bundle: PromptBundle,
    pages: list[Page],
    center: Page,
    remaining: int,
    config: Config,
    closure_tokens_used: int,
    passage_ids: list[int | None],
    selected_pages: list[Page],
) -> tuple[int, int, list[Page]]:
    if remaining <= 0 or center.page_id >= len(passage_ids):
        return 0, closure_tokens_used, []
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_passage_closure_budget_fraction))
    if cap <= 0 or closure_tokens_used >= cap:
        return 0, closure_tokens_used, []
    passage_id = passage_ids[center.page_id]
    if passage_id is None:
        return 0, closure_tokens_used, []
    selected_ids = {page.page_id for page in selected_pages}
    selected_ids.add(center.page_id)
    radius = max(0, int(config.ours_passage_closure_radius_pages))
    candidates: list[Page] = []
    for page in pages:
        if page.page_id in selected_ids or page.page_id >= len(passage_ids):
            continue
        if passage_ids[page.page_id] != passage_id:
            continue
        if radius > 0 and abs(page.page_id - center.page_id) > radius:
            continue
        candidates.append(page)
    candidates.sort(key=lambda page: (abs(page.page_id - center.page_id), -page.score, page.page_id))

    total_added = 0
    added_pages: list[Page] = []
    for page in candidates:
        if remaining <= 0 or closure_tokens_used >= cap:
            break
        allowed = min(remaining, cap - closure_tokens_used)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            selected_ids.add(page.page_id)
            continue
        remaining -= added
        total_added += added
        closure_tokens_used += added
        selected_ids.add(page.page_id)
        added_pages.append(page)
    return total_added, closure_tokens_used, added_pages


def add_structured_fingerprint_pages(
    keep: set[int],
    bundle: PromptBundle,
    pages: list[Page],
    remaining: int,
    config: Config,
    selected_pages: list[Page],
) -> tuple[int, list[Page], dict[str, float]]:
    stats = {
        "structured_fingerprint_labels": 0.0,
        "structured_fingerprint_tokens": 0.0,
    }
    if remaining <= 0:
        return 0, [], stats
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_structured_fingerprint_budget_fraction))
    if cap <= 0:
        return 0, [], stats
    selected_ids = {page.page_id for page in selected_pages}
    label_pages: dict[str, Page] = {}
    for page in pages:
        match = STRUCTURED_LABEL_RE.search(page.text)
        if not match:
            continue
        label = match.group(0).lower()
        if label not in label_pages:
            label_pages[label] = page
    candidates = sorted(
        label_pages.values(),
        key=lambda page: (
            int(re.search(r"\d+", page.text).group(0)) if re.search(r"\d+", page.text) else page.page_id,
            page.page_id,
        ),
    )
    total_added = 0
    added_pages: list[Page] = []
    for page in candidates:
        if remaining <= 0 or total_added >= cap:
            break
        if page.page_id in selected_ids:
            continue
        allowed = min(remaining, cap - total_added)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            selected_ids.add(page.page_id)
            continue
        remaining -= added
        total_added += added
        selected_ids.add(page.page_id)
        added_pages.append(page)
    stats["structured_fingerprint_labels"] = float(len(candidates))
    stats["structured_fingerprint_tokens"] = float(total_added)
    return total_added, added_pages, stats


def add_coverage_certificate_pages(
    keep: set[int],
    bundle: PromptBundle,
    example: Example,
    pages: list[Page],
    remaining: int,
    config: Config,
    selected_pages: list[Page],
) -> tuple[int, list[Page], dict[str, float]]:
    stats = {
        "coverage_certificate_terms": 0.0,
        "coverage_certificate_covered": 0.0,
        "coverage_certificate_recall": 0.0,
        "coverage_certificate_tokens": 0.0,
    }
    if remaining <= 0 or not coverage_certificate_active_for_example(config, example):
        return 0, [], stats
    terms = query_coverage_terms_for_example(example, config)
    stats["coverage_certificate_terms"] = float(len(terms))
    if len(terms) < max(1, config.ours_coverage_certificate_min_terms):
        return 0, [], stats
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_coverage_certificate_budget_fraction))
    if cap <= 0:
        return 0, [], stats

    selected_ids = {page.page_id for page in selected_pages}
    covered: set[str] = set()
    added_pages: list[Page] = []
    total_added = 0
    coverage_by_page = {page.page_id: page_query_coverage_terms(page, terms) for page in pages}
    while remaining > 0 and total_added < cap and len(covered) < len(terms):
        candidates: list[tuple[float, float, int, Page, set[str]]] = []
        for page in pages:
            if page.page_id in selected_ids:
                continue
            new_terms = coverage_by_page.get(page.page_id, set()) - covered
            if not new_terms:
                continue
            new_count = sum(
                1
                for idx in range(*bundle.page_spans[page.page_id])
                if bundle.context_token_start <= idx < bundle.query_start and idx not in keep
            )
            if new_count <= 0:
                continue
            # Prefer pages that certify multiple uncovered query terms, then use the base evidence score.
            candidates.append((float(len(new_terms)), page.score, -new_count, page, new_terms))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2], -item[3].page_id), reverse=True)
        _, _, _, page, new_terms = candidates[0]
        allowed = min(remaining, cap - total_added)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            selected_ids.add(page.page_id)
            continue
        remaining -= added
        total_added += added
        covered.update(new_terms)
        selected_ids.add(page.page_id)
        added_pages.append(page)

    stats["coverage_certificate_covered"] = float(len(covered))
    stats["coverage_certificate_recall"] = len(covered) / max(1, len(terms))
    stats["coverage_certificate_tokens"] = float(total_added)
    return total_added, added_pages, stats


def evidence_score_gap(pages: list[Page], rank: int = 3) -> float:
    scores = sorted((float(page.score) for page in pages), reverse=True)
    if len(scores) <= 1:
        return 1.0
    idx = min(max(1, rank - 1), len(scores) - 1)
    return scores[0] - scores[idx]


def add_spread_rescue_pages(
    keep: set[int],
    bundle: PromptBundle,
    pages: list[Page],
    remaining: int,
    config: Config,
    selected_pages: list[Page],
) -> tuple[int, list[Page]]:
    if remaining <= 0 or not pages:
        return 0, []
    gap = evidence_score_gap(pages)
    if gap > max(0.0, config.ours_spread_gap_threshold):
        return 0, []
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_spread_budget_fraction))
    if cap <= 0:
        return 0, []
    selected_ids = {page.page_id for page in selected_pages}
    bins = max(2, config.ours_spread_bins)
    bin_size = max(1, math.ceil(len(pages) / bins))
    candidates: list[Page] = []
    for start in range(0, len(pages), bin_size):
        group = [
            page
            for page in pages[start : start + bin_size]
            if page.page_id not in selected_ids and page.score >= config.ours_spread_min_score
        ]
        if not group:
            continue
        candidates.append(max(group, key=lambda page: (page.score, -page.page_id)))
    candidates.sort(key=lambda page: (page.score, -abs(page.page_id - len(pages) / 2.0)), reverse=True)
    total_added = 0
    added_pages: list[Page] = []
    for page in candidates:
        if remaining <= 0 or total_added >= cap:
            break
        allowed = min(remaining, cap - total_added)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            continue
        remaining -= added
        total_added += added
        added_pages.append(page)
        selected_ids.add(page.page_id)
    return total_added, added_pages


def bridge_allowed_for_example(example: Example) -> bool:
    return example.metric in {"qa_f1", "retrieval_score", "count_score", "ruler_string_match", "ruler_string_match_part"}


def build_bridge_cache(example: Example, pages: list[Page]) -> tuple[list[set[str]], Counter[str], dict[str, float]] | None:
    if not bridge_allowed_for_example(example) or not pages:
        return None
    q_entities = extract_entities(example.query or example.suffix_template)
    entity_sets: list[set[str]] = []
    doc_freq: Counter[str] = Counter()
    for page in pages:
        terms = {
            term
            for term in extract_entities(page.text)
            if term not in q_entities and len(term) >= 3 and term not in STOPWORDS
        }
        entity_sets.append(terms)
        doc_freq.update(terms)
    page_count = max(1, len(pages))
    idf = {
        term: 1.0 + math.log((1.0 + page_count) / (1.0 + float(freq)))
        for term, freq in doc_freq.items()
        if freq >= 2
    }
    return entity_sets, doc_freq, idf


def add_bridge_pages(
    keep: set[int],
    bundle: PromptBundle,
    pages: list[Page],
    center: Page,
    remaining: int,
    config: Config,
    bridge_tokens_used: int,
    selected_pages: list[Page],
    bridge_cache: tuple[list[set[str]], Counter[str], dict[str, float]] | None,
) -> tuple[int, int, list[Page]]:
    if remaining <= 0 or bridge_cache is None:
        return 0, bridge_tokens_used, []
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_bridge_budget_fraction))
    if cap <= 0 or bridge_tokens_used >= cap or center.page_id >= len(pages):
        return 0, bridge_tokens_used, []
    entity_sets, doc_freq, idf = bridge_cache
    center_terms = [
        term
        for term in entity_sets[center.page_id]
        if doc_freq.get(term, 0) >= 2 and term in idf
    ]
    if not center_terms:
        return 0, bridge_tokens_used, []
    center_terms.sort(key=lambda term: (idf.get(term, 0.0), -doc_freq.get(term, 0), -len(term)), reverse=True)
    center_set = set(center_terms[: max(1, config.ours_bridge_max_terms)])
    selected_ids = {page.page_id for page in selected_pages}
    candidates: list[tuple[float, float, Page]] = []
    for page in pages:
        if page.page_id in selected_ids or page.page_id == center.page_id or page.score < config.ours_bridge_min_score:
            continue
        overlap = center_set & entity_sets[page.page_id]
        if not overlap:
            continue
        bridge_score = sum(idf.get(term, 0.0) for term in overlap)
        candidates.append((bridge_score, page.score, page))
    candidates.sort(key=lambda item: (item[0], item[1], -abs(item[2].page_id - center.page_id)), reverse=True)
    total_added = 0
    added_pages: list[Page] = []
    for _, _, page in candidates:
        if remaining <= 0 or bridge_tokens_used >= cap:
            break
        allowed = min(remaining, cap - bridge_tokens_used)
        added = add_page_to_keep(keep, bundle, page, allowed)
        if added <= 0:
            continue
        remaining -= added
        total_added += added
        bridge_tokens_used += added
        added_pages.append(page)
        selected_ids.add(page.page_id)
    return total_added, bridge_tokens_used, added_pages


def add_graph_bridge_pages(
    keep: set[int],
    bundle: PromptBundle,
    example: Example,
    pages: list[Page],
    remaining: int,
    config: Config,
    selected_pages: list[Page],
    bridge_cache: tuple[list[set[str]], Counter[str], dict[str, float]] | None,
) -> tuple[int, list[Page], dict[str, float]]:
    stats = {
        "graph_bridge_pairs": 0.0,
        "graph_bridge_tokens": 0.0,
    }
    if remaining <= 0 or bridge_cache is None:
        return 0, [], stats
    cap = int(max(0, config.budget_tokens) * max(0.0, config.ours_graph_bridge_budget_fraction))
    if cap <= 0:
        return 0, [], stats
    entity_sets, doc_freq, idf = bridge_cache
    selected_ids = {page.page_id for page in selected_pages}
    anchors = extract_query_anchors(
        example.query or example.suffix_template,
        include_title_phrases=title_anchor_active_for_example(config, example),
    )
    q_entities = extract_entities(example.query or example.suffix_template)
    seeds = [
        page
        for page in sorted(
            pages,
            key=lambda item: (
                int(page_has_anchor_hit(item, anchors, q_entities)),
                item.score,
                -(item.token_end - item.token_start),
                -item.page_id,
            ),
            reverse=True,
        )
        if page.page_id < len(entity_sets)
    ][: max(1, config.ours_graph_bridge_seed_pages)]
    candidates: list[tuple[float, float, float, Page, Page]] = []
    for seed in seeds:
        seed_terms = [
            term
            for term in entity_sets[seed.page_id]
            if doc_freq.get(term, 0) >= 2 and term in idf
        ]
        seed_terms.sort(key=lambda term: (idf.get(term, 0.0), -doc_freq.get(term, 0), -len(term)), reverse=True)
        seed_set = set(seed_terms[: max(1, config.ours_graph_bridge_max_terms)])
        if not seed_set:
            continue
        for page in pages:
            if page.page_id == seed.page_id or page.page_id >= len(entity_sets):
                continue
            if page.score < config.ours_graph_bridge_min_score:
                continue
            overlap = seed_set & entity_sets[page.page_id]
            if not overlap:
                continue
            link_score = sum(idf.get(term, 0.0) for term in overlap)
            pair_score = link_score + 0.5 * seed.score + 0.5 * page.score
            candidates.append((pair_score, link_score, page.score, seed, page))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -abs(item[3].page_id - item[4].page_id)), reverse=True)

    total_added = 0
    added_pages: list[Page] = []
    pair_count = 0
    for _, _, _, seed, page in candidates:
        if remaining <= 0 or total_added >= cap:
            break
        pair_added = 0
        for candidate in (seed, page):
            if candidate.page_id in selected_ids:
                continue
            allowed = min(remaining, cap - total_added)
            if allowed <= 0:
                break
            added = add_page_to_keep(keep, bundle, candidate, allowed)
            if added <= 0:
                selected_ids.add(candidate.page_id)
                continue
            remaining -= added
            total_added += added
            pair_added += added
            selected_ids.add(candidate.page_id)
            added_pages.append(candidate)
        if pair_added > 0:
            pair_count += 1
    stats["graph_bridge_pairs"] = float(pair_count)
    stats["graph_bridge_tokens"] = float(total_added)
    return total_added, added_pages, stats


def score_pages(
    example: Example,
    pages: list[Page],
    config: Config,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[int, torch.Tensor]:
    query = example.query or example.suffix_template
    q_words = word_counter(query)
    q_entities = extract_entities(query)
    lexical_raw: list[float] = []
    idf_lexical_raw: list[float] = []
    bm25_raw: list[float] = []
    entity_raw: list[float] = []
    structural_raw: list[float] = []
    coverage_raw: list[float] = []
    page_word_counters = [word_counter(page.text) for page in pages]
    doc_freq: Counter[str] = Counter()
    for p_words in page_word_counters:
        doc_freq.update(p_words.keys())
    page_count = max(1, len(pages))
    page_lengths = [sum(p_words.values()) for p_words in page_word_counters]
    avg_page_len = max(1.0, sum(page_lengths) / max(1, len(page_lengths)))
    bm25_k1 = max(0.01, config.ours_bm25_k1)
    bm25_b = min(1.0, max(0.0, config.ours_bm25_b))
    for page, p_words in zip(pages, page_word_counters):
        p_entities = extract_entities(page.text)
        lexical_raw.append(float(sum(min(count, p_words.get(word, 0)) for word, count in q_words.items())))
        idf_lexical_raw.append(
            float(
                sum(
                    min(count, p_words.get(word, 0))
                    * (1.0 + math.log((1.0 + page_count) / (1.0 + float(doc_freq.get(word, 0)))))
                    for word, count in q_words.items()
                )
            )
        )
        page_len = max(1.0, float(sum(p_words.values())))
        bm25_score = 0.0
        for word in q_words:
            tf = float(p_words.get(word, 0))
            if tf <= 0.0:
                continue
            df = float(doc_freq.get(word, 0))
            idf = math.log(1.0 + (page_count - df + 0.5) / (df + 0.5))
            denom = tf + bm25_k1 * (1.0 - bm25_b + bm25_b * page_len / avg_page_len)
            bm25_score += idf * (tf * (bm25_k1 + 1.0)) / max(denom, 1e-6)
        bm25_raw.append(float(bm25_score))
        entity_raw.append(float(len(q_entities & p_entities)))
        structural_raw.append(float(page.text.count(":") + page.text.count("|") + len(re.findall(r"\bParagraph\s+\d+", page.text))))
        if task_is_global(example):
            if len(pages) <= 1:
                coverage_raw.append(1.0)
            else:
                pos = page.page_id / max(1, len(pages) - 1)
                # Favor a broad beginning/middle/end spread for summarization-like tasks.
                anchors = (0.08, 0.50, 0.92)
                coverage_raw.append(float(max(1.0 - abs(pos - anchor) / 0.22 for anchor in anchors)))
        else:
            coverage_raw.append(0.0)

    lexical = normalize_values(lexical_raw)
    idf_lexical = normalize_values(idf_lexical_raw)
    bm25_lexical = normalize_values(bm25_raw)
    entity = normalize_values(entity_raw)
    structural = normalize_values(structural_raw)
    coverage = normalize_values(coverage_raw)
    semantic_vectors: dict[int, torch.Tensor] = {}
    semantic = [0.0 for _ in pages]
    late = [0.0 for _ in pages]
    needs_mean_semantic = config.ours_scorer in {
        "semantic",
        "hybrid",
        "hybrid_mmr",
        "hybrid_late_mmr",
        "hybrid_late_mmr_bm25_flow",
        "hybrid_late_mmr_bridge_flow",
        "hybrid_late_mmr_flow",
        "hybrid_late_mmr_multiscale_bm25_flow",
        "hybrid_late_mmr_multiscale_bridge_flow",
        "hybrid_late_mmr_multiscale_task_bridge_flow",
        "hybrid_late_mmr_multiscale_flow",
        "hybrid_late_mmr_idf_flow",
        "hybrid_late_mmr_multiscale_idf_flow",
        "hybrid_late_mmr_idf_spread_flow",
        "hybrid_late_mmr_multiscale_idf_spread_flow",
    }
    needs_late_interaction = config.ours_scorer in {
        "late_interaction",
        "hybrid_late_mmr",
        "hybrid_late_mmr_bm25_flow",
        "hybrid_late_mmr_bridge_flow",
        "hybrid_late_mmr_flow",
        "hybrid_late_mmr_multiscale_bm25_flow",
        "hybrid_late_mmr_multiscale_bridge_flow",
        "hybrid_late_mmr_multiscale_task_bridge_flow",
        "hybrid_late_mmr_multiscale_flow",
        "hybrid_late_mmr_idf_flow",
        "hybrid_late_mmr_multiscale_idf_flow",
        "hybrid_late_mmr_idf_spread_flow",
        "hybrid_late_mmr_multiscale_idf_spread_flow",
    }
    if needs_mean_semantic and model is not None and tokenizer is not None:
        query_vec = static_text_embeddings(model, tokenizer, [query], config.semantic_embed_max_tokens)[0]
        page_vecs = static_text_embeddings(model, tokenizer, [page.text for page in pages], config.semantic_embed_max_tokens)
        semantic_raw = torch.matmul(page_vecs, query_vec).detach().float().cpu().tolist()
        semantic = normalize_values([float(value) for value in semantic_raw])
        semantic_vectors = {page.page_id: page_vecs[idx].detach().float().cpu() for idx, page in enumerate(pages)}
    if needs_late_interaction and model is not None and tokenizer is not None:
        late_raw = late_interaction_scores(
            model,
            tokenizer,
            query,
            [page.text for page in pages],
            config.semantic_embed_max_tokens,
        )
        late = normalize_values(late_raw)

    for idx, page in enumerate(pages):
        if config.ours_scorer == "lexical":
            score = lexical[idx] + config.entity_weight * entity[idx] + config.structural_weight * structural[idx]
        elif config.ours_scorer == "semantic":
            score = semantic[idx] + config.coverage_weight * coverage[idx]
        elif config.ours_scorer == "late_interaction":
            score = late[idx] + config.coverage_weight * coverage[idx]
        else:
            semantic_component = (
                late[idx]
                if config.ours_scorer
                in {
                    "hybrid_late_mmr",
                    "hybrid_late_mmr_bm25_flow",
                    "hybrid_late_mmr_bridge_flow",
                    "hybrid_late_mmr_flow",
                    "hybrid_late_mmr_multiscale_bm25_flow",
                    "hybrid_late_mmr_multiscale_bridge_flow",
                    "hybrid_late_mmr_multiscale_task_bridge_flow",
                    "hybrid_late_mmr_multiscale_flow",
                    "hybrid_late_mmr_idf_flow",
                    "hybrid_late_mmr_multiscale_idf_flow",
                    "hybrid_late_mmr_idf_spread_flow",
                    "hybrid_late_mmr_multiscale_idf_spread_flow",
                }
                else semantic[idx]
            )
            lexical_component = lexical[idx]
            if idf_enabled(config):
                mix = min(1.0, max(0.0, config.ours_idf_mix))
                lexical_component = (1.0 - mix) * lexical[idx] + mix * idf_lexical[idx]
            if bm25_enabled(config):
                mix = min(1.0, max(0.0, config.ours_bm25_mix))
                lexical_component = (1.0 - mix) * lexical[idx] + mix * bm25_lexical[idx]
            score = (
                config.semantic_weight * semantic_component
                + config.lexical_weight * lexical_component
                + config.entity_weight * entity[idx]
                + config.structural_weight * structural[idx]
                + config.coverage_weight * coverage[idx]
            )
        page.score = float(score)
    return semantic_vectors


def add_page_to_keep(
    keep: set[int],
    bundle: PromptBundle,
    page: Page,
    remaining: int,
) -> int:
    start, end = bundle.page_spans[page.page_id]
    page_indices = [idx for idx in range(start, end) if bundle.context_token_start <= idx < bundle.query_start and idx not in keep]
    if not page_indices or remaining <= 0:
        return 0
    page_indices = page_indices[:remaining]
    keep.update(page_indices)
    return len(page_indices)


def base_context_keep_indices(bundle: PromptBundle, config: Config) -> set[int]:
    keep: set[int] = set(range(bundle.prefix_token_count))
    context_start = bundle.context_token_start
    context_end = bundle.query_start
    context_len = max(0, context_end - context_start)
    sink_end = min(context_end, context_start + max(0, config.sink_tokens))
    keep.update(range(context_start, sink_end))
    recent_start = max(context_start, context_end - max(0, config.recent_tokens))
    keep.update(range(recent_start, context_end))
    return keep


def fit_context_budget(indices: set[int], bundle: PromptBundle, budget_tokens: int) -> list[int]:
    prefix = set(range(bundle.prefix_token_count))
    context = sorted(idx for idx in indices if bundle.context_token_start <= idx < bundle.query_start)
    if budget_tokens > 0 and len(context) > budget_tokens:
        # When sink + recent already exceed the budget, keep both ends instead of truncating away recency.
        chosen: set[int] = set()
        left = 0
        right = len(context) - 1
        while len(chosen) < budget_tokens and left <= right:
            chosen.add(context[left])
            left += 1
            if len(chosen) >= budget_tokens or right < left:
                break
            chosen.add(context[right])
            right -= 1
        context = sorted(chosen)
    return sorted(prefix | set(context))


def keep_full(bundle: PromptBundle, _: Example, __: list[Page], config: Config, ___: Any = None) -> list[int]:
    return list(range(bundle.query_start))


def keep_streaming(bundle: PromptBundle, _: Example, __: list[Page], config: Config, ___: Any = None) -> list[int]:
    keep = base_context_keep_indices(bundle, config)
    return fit_context_budget(keep, bundle, config.budget_tokens)


def keep_ours_page(bundle: PromptBundle, example: Example, pages: list[Page], config: Config, extra: Any = None) -> list[int]:
    if full_fallback_active_for_example(config, example):
        return keep_full(bundle, example, pages, config, extra)
    if extra is None:
        extra = {}
    extra.setdefault("coverage_risk_triggered", 0)
    extra.setdefault("coverage_risk_initial_terms", "")
    extra.setdefault("coverage_risk_initial_recall", "")
    extra.setdefault("coverage_risk_escalation_budget", "")
    extra.setdefault("budget_ladder_active", 0)
    extra.setdefault("budget_ladder_tokens", "")
    extra.setdefault("budget_ladder_selected_budget", "")
    extra.setdefault("budget_ladder_level", "")
    extra.setdefault("budget_ladder_reasons", "")
    extra.setdefault("budget_ladder_gap2_level", "")
    extra.setdefault("budget_ladder_entropy_level", "")
    extra.setdefault("budget_ladder_top_score_level", "")
    extra.setdefault("graph_bridge_pairs", "")
    extra.setdefault("graph_bridge_tokens", "")
    semantic_vectors = score_pages(
        example,
        pages,
        config,
        model=extra.get("model"),
        tokenizer=extra.get("tokenizer"),
    )
    if multiscale_enabled(config):
        apply_multiscale_scores(pages, config)
    if flow_enabled(config):
        apply_flow_scores(example, pages, config)
    if budget_ladder_active_for_example(config, example):
        ladder = budget_ladder_decision(example, bundle, pages, config)
        extra["budget_ladder_active"] = ladder.get("active", 0)
        extra["budget_ladder_tokens"] = ladder.get("tokens", "")
        extra["budget_ladder_selected_budget"] = ladder.get("selected_budget", "")
        extra["budget_ladder_level"] = ladder.get("level", "")
        extra["budget_ladder_reasons"] = ladder.get("reasons", "")
        extra["budget_ladder_gap2_level"] = ladder.get("gap2_level", "")
        extra["budget_ladder_entropy_level"] = ladder.get("entropy_level", "")
        extra["budget_ladder_top_score_level"] = ladder.get("top_score_level", "")
        selected_budget = int(ladder.get("selected_budget", config.budget_tokens))
        if selected_budget >= bundle.query_start:
            return keep_full(bundle, example, pages, config, extra)
        if selected_budget > 0 and selected_budget != config.budget_tokens:
            config = replace(config, budget_tokens=selected_budget)
    if (
        score_risk_active_for_example(config, example)
        and config.ours_score_risk_budget_tokens > config.budget_tokens
        and score_risk_triggered_for_pages(pages, config, raw_prefix_tokens=bundle.query_start)
    ):
        if config.ours_score_risk_budget_tokens >= bundle.query_start:
            return keep_full(bundle, example, pages, config, extra)
        config = replace(config, budget_tokens=config.ours_score_risk_budget_tokens)
    bridge_active = bridge_active_for_example(config, example)
    graph_bridge_active = graph_bridge_active_for_example(config, example)
    label_support_active = label_support_active_for_example(config, example)
    passage_closure_active = passage_closure_active_for_example(config, example)
    structured_fingerprint_active = structured_fingerprint_active_for_example(config, example)
    bridge_cache = build_bridge_cache(example, pages) if bridge_active or graph_bridge_active else None
    passage_ids = infer_passage_ids(pages) if passage_closure_active else []
    keep = base_context_keep_indices(bundle, config)
    selected_context_tokens = sum(1 for idx in keep if bundle.context_token_start <= idx < bundle.query_start)
    remaining = max(0, config.budget_tokens - selected_context_tokens)
    selected_pages: list[Page] = []
    flow_support_tokens = 0
    bridge_tokens_used = 0
    label_tokens_used = 0
    passage_closure_tokens_used = 0
    coverage_certificate_stats = {
        "coverage_certificate_terms": 0.0,
        "coverage_certificate_covered": 0.0,
        "coverage_certificate_recall": 0.0,
        "coverage_certificate_tokens": 0.0,
    }
    structured_fingerprint_stats = {
        "structured_fingerprint_labels": 0.0,
        "structured_fingerprint_tokens": 0.0,
    }

    anchors = extract_query_anchors(
        example.query or example.suffix_template,
        include_title_phrases=title_anchor_active_for_example(config, example),
    )
    coverage_mmr_weight = max(0.0, config.ours_coverage_mmr_weight)
    coverage_query_terms = query_coverage_terms_for_example(example, config) if coverage_mmr_weight > 0.0 else []
    coverage_terms_by_page = (
        {page.page_id: page_query_coverage_terms(page, coverage_query_terms) for page in pages}
        if coverage_query_terms
        else {}
    )
    if structured_fingerprint_active and remaining > 0:
        fingerprint_added, fingerprint_pages, structured_fingerprint_stats = add_structured_fingerprint_pages(
            keep,
            bundle,
            pages,
            remaining,
            config,
            selected_pages,
        )
        if fingerprint_added > 0:
            remaining -= fingerprint_added
            selected_pages.extend(fingerprint_pages)
    certificate_added, certificate_pages, coverage_certificate_stats = add_coverage_certificate_pages(
        keep,
        bundle,
        example,
        pages,
        remaining,
        config,
        selected_pages,
    )
    if certificate_added > 0:
        remaining -= certificate_added
        selected_pages.extend(certificate_pages)
    if graph_bridge_active and remaining > 0:
        graph_added, graph_pages, graph_stats = add_graph_bridge_pages(
            keep,
            bundle,
            example,
            pages,
            remaining,
            config,
            selected_pages,
            bridge_cache,
        )
        extra["graph_bridge_pairs"] = graph_stats.get("graph_bridge_pairs", "")
        extra["graph_bridge_tokens"] = graph_stats.get("graph_bridge_tokens", "")
        if graph_added > 0:
            remaining -= graph_added
            selected_pages.extend(graph_pages)
    if passage_closure_active and remaining > 0 and selected_pages:
        for page in list(selected_pages):
            closure_added, passage_closure_tokens_used, closure_pages = add_passage_closure_pages(
                keep,
                bundle,
                pages,
                page,
                remaining,
                config,
                passage_closure_tokens_used,
                passage_ids,
                selected_pages,
            )
            if closure_added > 0:
                remaining -= closure_added
                selected_pages.extend(closure_pages)
            if remaining <= 0:
                break
    if anchors and remaining > 0 and config.anchor_pages_per_key > 0:
        lowered_by_page = {page.page_id: page.text.lower() for page in pages}
        for anchor in anchors:
            matches = [
                page
                for page in pages
                if anchor in lowered_by_page[page.page_id]
                or anchor.replace("-", " ") in lowered_by_page[page.page_id]
                or anchor.replace(" ", "-") in lowered_by_page[page.page_id]
            ]
            matches.sort(key=lambda page: (page.score, -(page.token_end - page.token_start), -page.page_id), reverse=True)
            for page in matches[: config.anchor_pages_per_key]:
                added = add_page_to_keep(keep, bundle, page, remaining)
                if added > 0:
                    remaining -= added
                    selected_pages.append(page)
                    if label_support_active and remaining > 0:
                        label_added, label_tokens_used, label_pages = add_structured_label_pages(
                            keep,
                            bundle,
                            pages,
                            page,
                            remaining,
                            config,
                            label_tokens_used,
                        )
                        if label_added > 0:
                            remaining -= label_added
                            selected_pages.extend(label_pages)
                    if flow_enabled(config):
                        support_added, flow_support_tokens, support_pages = add_flow_neighbor_pages(
                            keep,
                            bundle,
                            example,
                            pages,
                            page,
                            remaining,
                            config,
                            flow_support_tokens,
                        )
                        if support_added > 0:
                            remaining -= support_added
                            selected_pages.extend(support_pages)
                    if bridge_active and remaining > 0:
                        bridge_added, bridge_tokens_used, bridge_pages = add_bridge_pages(
                            keep,
                            bundle,
                            pages,
                            page,
                            remaining,
                            config,
                            bridge_tokens_used,
                            selected_pages,
                            bridge_cache,
                        )
                        if bridge_added > 0:
                            remaining -= bridge_added
                            selected_pages.extend(bridge_pages)
                    if passage_closure_active and remaining > 0:
                        closure_added, passage_closure_tokens_used, closure_pages = add_passage_closure_pages(
                            keep,
                            bundle,
                            pages,
                            page,
                            remaining,
                            config,
                            passage_closure_tokens_used,
                            passage_ids,
                            selected_pages,
                        )
                        if closure_added > 0:
                            remaining -= closure_added
                            selected_pages.extend(closure_pages)
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
    if spread_enabled(config) and remaining > 0:
        spread_added, spread_pages = add_spread_rescue_pages(keep, bundle, pages, remaining, config, selected_pages)
        if spread_added > 0:
            remaining -= spread_added
            selected_pages.extend(spread_pages)
            if label_support_active and remaining > 0:
                for page in spread_pages:
                    label_added, label_tokens_used, label_pages = add_structured_label_pages(
                        keep,
                        bundle,
                        pages,
                        page,
                        remaining,
                        config,
                        label_tokens_used,
                    )
                    if label_added > 0:
                        remaining -= label_added
                        selected_pages.extend(label_pages)
                    if remaining <= 0:
                        break
    selected_page_id_set = {page.page_id for page in selected_pages}
    candidate_pages = [page for page in pages if page.page_id not in selected_page_id_set]
    while candidate_pages and remaining > 0:
        if selected_pages and (
            (
                config.ours_scorer
                in {
                    "hybrid_mmr",
                    "hybrid_late_mmr",
                    "hybrid_late_mmr_bm25_flow",
                    "hybrid_late_mmr_bridge_flow",
                    "hybrid_late_mmr_flow",
                    "hybrid_late_mmr_multiscale_bm25_flow",
                    "hybrid_late_mmr_multiscale_bridge_flow",
                    "hybrid_late_mmr_multiscale_task_bridge_flow",
                    "hybrid_late_mmr_multiscale_flow",
                    "hybrid_late_mmr_idf_flow",
                    "hybrid_late_mmr_multiscale_idf_flow",
                    "hybrid_late_mmr_idf_spread_flow",
                    "hybrid_late_mmr_multiscale_idf_spread_flow",
                }
                and semantic_vectors
            )
            or coverage_mmr_weight > 0.0
        ):
            selected_vecs = [semantic_vectors[page.page_id] for page in selected_pages if page.page_id in semantic_vectors]
            covered_terms: set[str] = set()
            if coverage_terms_by_page:
                for selected in selected_pages:
                    covered_terms.update(coverage_terms_by_page.get(selected.page_id, set()))
            coverage_denominator = max(1, len(coverage_query_terms))
            reranked = []
            for page in candidate_pages:
                redundancy = 0.0
                if page.page_id in semantic_vectors and selected_vecs:
                    page_vec = semantic_vectors[page.page_id]
                    redundancy = max(float(torch.dot(page_vec, vec).item()) for vec in selected_vecs)
                coverage_bonus = 0.0
                if coverage_terms_by_page:
                    new_terms = coverage_terms_by_page.get(page.page_id, set()) - covered_terms
                    coverage_bonus = len(new_terms) / coverage_denominator
                mmr_score = (
                    config.ours_mmr_lambda * page.score
                    - (1.0 - config.ours_mmr_lambda) * redundancy
                    + coverage_mmr_weight * coverage_bonus
                )
                reranked.append((mmr_score, coverage_bonus, page))
            reranked.sort(key=lambda item: (item[0], item[1], item[2].score, -item[2].page_id), reverse=True)
            page = reranked[0][2]
        else:
            page = max(candidate_pages, key=lambda item: (item.score, -(item.token_end - item.token_start), -item.page_id))
        new_count = sum(
            1
            for idx in range(*bundle.page_spans[page.page_id])
            if bundle.context_token_start <= idx < bundle.query_start and idx not in keep
        )
        if new_count <= 0:
            candidate_pages = [candidate for candidate in candidate_pages if candidate.page_id != page.page_id]
            continue
        added = add_page_to_keep(keep, bundle, page, remaining)
        remaining -= added
        selected_pages.append(page)
        if label_support_active and remaining > 0:
            label_added, label_tokens_used, label_pages = add_structured_label_pages(
                keep,
                bundle,
                pages,
                page,
                remaining,
                config,
                label_tokens_used,
            )
            if label_added > 0:
                remaining -= label_added
                selected_pages.extend(label_pages)
        if flow_enabled(config):
            support_added, flow_support_tokens, support_pages = add_flow_neighbor_pages(
                keep,
                bundle,
                example,
                pages,
                page,
                remaining,
                config,
                flow_support_tokens,
            )
            if support_added > 0:
                remaining -= support_added
                selected_pages.extend(support_pages)
        if bridge_active and remaining > 0:
            bridge_added, bridge_tokens_used, bridge_pages = add_bridge_pages(
                keep,
                bundle,
                pages,
                page,
                remaining,
                config,
                bridge_tokens_used,
                selected_pages,
                bridge_cache,
            )
            if bridge_added > 0:
                remaining -= bridge_added
                selected_pages.extend(bridge_pages)
        if passage_closure_active and remaining > 0:
            closure_added, passage_closure_tokens_used, closure_pages = add_passage_closure_pages(
                keep,
                bundle,
                pages,
                page,
                remaining,
                config,
                passage_closure_tokens_used,
                passage_ids,
                selected_pages,
            )
            if closure_added > 0:
                remaining -= closure_added
                selected_pages.extend(closure_pages)
        selected_page_id_set = {selected.page_id for selected in selected_pages}
        candidate_pages = [candidate for candidate in candidate_pages if candidate.page_id not in selected_page_id_set]
    fitted_keep = fit_context_budget(keep, bundle, config.budget_tokens)
    extra.update(coverage_certificate_stats)
    extra.update(structured_fingerprint_stats)
    extra["passage_closure_tokens"] = float(passage_closure_tokens_used)
    if (
        coverage_risk_active_for_example(config, example)
        and config.ours_coverage_risk_min_recall >= 0.0
        and config.ours_coverage_risk_budget_tokens > config.budget_tokens
    ):
        selected_ids = selected_page_ids(bundle, fitted_keep)
        coverage = selected_query_coverage_stats(example, pages, selected_ids, config)
        extra["coverage_risk_initial_terms"] = coverage["terms"]
        extra["coverage_risk_initial_recall"] = coverage["recall"]
        enough_terms = coverage["terms"] >= max(1, float(config.ours_coverage_risk_min_terms))
        if enough_terms and coverage["recall"] < config.ours_coverage_risk_min_recall:
            extra["coverage_risk_triggered"] = 1
            extra["coverage_risk_escalation_budget"] = config.ours_coverage_risk_budget_tokens
            if config.ours_coverage_risk_budget_tokens >= bundle.query_start:
                return keep_full(bundle, example, pages, config, extra)
            escalation_config = replace(
                config,
                budget_tokens=config.ours_coverage_risk_budget_tokens,
                ours_coverage_risk_tasks="",
            )
            return keep_ours_page(bundle, example, pages, escalation_config, extra)
    return fitted_keep


def attention_scores_from_suffix(
    model: torch.nn.Module,
    bundle: PromptBundle,
    full_prefix_cache: Any,
    input_device: torch.device,
) -> torch.Tensor:
    suffix_ids = bundle.input_ids[:, bundle.query_start :].to(input_device)
    if suffix_ids.shape[-1] == 0:
        return torch.zeros(bundle.query_start, dtype=torch.float32, device=input_device)
    outputs = model_forward(
        model,
        {
            "input_ids": suffix_ids,
            "past_key_values": clone_past_key_values(full_prefix_cache),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "output_hidden_states": False,
            "cache_position": torch.arange(bundle.query_start, bundle.query_start + suffix_ids.shape[-1], device=input_device),
        },
    )
    scores = torch.zeros(bundle.query_start, dtype=torch.float32, device=input_device)
    attentions = outputs.attentions or []
    for attn in attentions:
        # attn: [batch, heads, suffix_tokens, prefix + suffix_tokens]
        prefix_attn = attn[0, :, -min(attn.shape[-2], suffix_ids.shape[-1]) :, : bundle.query_start]
        scores += prefix_attn.float().sum(dim=(0, 1))
    return scores


def keep_attention_topk(
    bundle: PromptBundle,
    config: Config,
    scores: torch.Tensor,
    pool_kernel: int = 1,
) -> list[int]:
    keep = base_context_keep_indices(bundle, config)
    selected_context_tokens = sum(1 for idx in keep if bundle.context_token_start <= idx < bundle.query_start)
    remaining = max(0, config.budget_tokens - selected_context_tokens)
    if remaining <= 0:
        return fit_context_budget(keep, bundle, config.budget_tokens)
    context_scores = scores[bundle.context_token_start : bundle.query_start].detach().float()
    if context_scores.numel() == 0:
        return fit_context_budget(keep, bundle, config.budget_tokens)
    if pool_kernel > 1 and context_scores.numel() >= pool_kernel:
        pad = pool_kernel // 2
        pooled = F.max_pool1d(context_scores.view(1, 1, -1), kernel_size=pool_kernel, stride=1, padding=pad)
        context_scores = pooled.view(-1)[: context_scores.numel()]
    banned = torch.zeros_like(context_scores, dtype=torch.bool)
    for idx in keep:
        if bundle.context_token_start <= idx < bundle.query_start:
            banned[idx - bundle.context_token_start] = True
    context_scores = context_scores.masked_fill(banned, -float("inf"))
    topk = min(remaining, int(torch.isfinite(context_scores).sum().item()))
    if topk > 0:
        selected = torch.topk(context_scores, k=topk).indices.detach().cpu().tolist()
        keep.update(bundle.context_token_start + int(idx) for idx in selected)
    return fit_context_budget(keep, bundle, config.budget_tokens)


def keep_h2o(bundle: PromptBundle, _: Example, __: list[Page], config: Config, scores: torch.Tensor) -> list[int]:
    return keep_attention_topk(bundle, config, scores, pool_kernel=1)


def keep_snapkv(bundle: PromptBundle, _: Example, __: list[Page], config: Config, scores: torch.Tensor) -> list[int]:
    return keep_attention_topk(bundle, config, scores, pool_kernel=max(1, config.snap_pool_kernel))


METHODS = {
    "full_kv": keep_full,
    "streamingllm_sink_recent": keep_streaming,
    "h2o_observe": keep_h2o,
    "snapkv_observe": keep_snapkv,
    "ours_page_gather": keep_ours_page,
}


@torch.inference_mode()
def prefill_prefix(
    model: torch.nn.Module,
    bundle: PromptBundle,
    input_device: torch.device,
    chunk_tokens: int,
) -> tuple[Any, float]:
    ids = bundle.input_ids[:, : bundle.query_start].to(input_device)
    started = time.perf_counter()
    if chunk_tokens > 0 and ids.shape[-1] > chunk_tokens:
        past_key_values = None
        for start in range(0, ids.shape[-1], chunk_tokens):
            end = min(start + chunk_tokens, ids.shape[-1])
            outputs = model_forward(
                model,
                {
                    "input_ids": ids[:, start:end],
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "cache_position": torch.arange(start, end, device=input_device),
                },
            )
            past_key_values = outputs.past_key_values
        return past_key_values, time.perf_counter() - started
    outputs = model_forward(
        model,
        {
            "input_ids": ids,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(bundle.query_start, device=input_device),
        },
    )
    return outputs.past_key_values, time.perf_counter() - started


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    position_start: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor, float]:
    ids = input_ids.to(input_device)
    if ids.shape[-1] == 0:
        raise ValueError("empty token segment")
    started = time.perf_counter()
    outputs = model_forward(
        model,
        {
            "input_ids": ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(position_start, position_start + ids.shape[-1], device=input_device),
        },
    )
    return outputs.past_key_values, outputs.logits[:, -1, :].detach(), time.perf_counter() - started


@torch.inference_mode()
def generate_with_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    bundle: PromptBundle,
    prefix_cache: Any,
    max_new_tokens: int,
    input_device: torch.device,
) -> tuple[str, list[int], float, float]:
    suffix_ids = bundle.input_ids[:, bundle.query_start :].to(input_device)
    query_cache, prev_logits, query_seconds = run_tokens(model, suffix_ids, prefix_cache, bundle.query_start, input_device)
    generated: list[int] = []
    decode_started = time.perf_counter()
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for step in range(max_new_tokens):
        next_id = int(torch.argmax(prev_logits.float(), dim=-1).item())
        if next_id in eos_ids:
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], dtype=torch.long, device=input_device)
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": query_cache,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.tensor([bundle.query_start + bundle.suffix_token_count + step], device=input_device),
            },
        )
        query_cache = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    decode_seconds = time.perf_counter() - decode_started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated, query_seconds, decode_seconds


def selected_page_ids(bundle: PromptBundle, keep_indices: list[int]) -> list[int]:
    keep = set(keep_indices)
    out = []
    for page_id, (start, end) in bundle.page_spans.items():
        total = max(1, end - start)
        kept = sum(1 for idx in range(start, end) if idx in keep)
        if kept / total >= 0.5:
            out.append(page_id)
    return out


TREC_HEAD_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "s",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "your",
}


def trec_head_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in TREC_HEAD_STOP_WORDS and len(term) > 1
    }


def trec_question_text(query: str) -> str:
    if "Question:" in query:
        query = query.split("Question:", 1)[1]
    if "\nType:" in query:
        query = query.split("\nType:", 1)[0]
    return query.strip()


def trec_context_examples(context: str) -> list[tuple[str, str]]:
    return [
        (question.strip(), label.strip())
        for question, label in re.findall(r"Question:\s*(.*?)\nType:\s*([^\n]+)", context, flags=re.S)
    ]


def trec_nearest_examples(question: str, context: str, k: int = 18) -> list[tuple[str, str]]:
    question_terms = trec_head_terms(question)
    question_first = question.lower().split()[:1]
    scored: list[tuple[float, str, str]] = []
    for example_question, label in trec_context_examples(context):
        example_terms = trec_head_terms(example_question)
        score = len(question_terms & example_terms) / max(1.0, len(question_terms) + 0.25)
        if question_first and question_first == example_question.lower().split()[:1]:
            score += 0.05
        scored.append((score, example_question, label))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(question_text, label) for _score, question_text, label in scored[:k]]


def normalize_trec_head_prediction(text: str, all_classes: list[str]) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    lowered = first_line.lower()
    for class_name in all_classes:
        if lowered == class_name.lower():
            return class_name
    for class_name in all_classes:
        class_lower = class_name.lower()
        if class_lower in lowered or lowered in class_lower:
            return class_name
    aliases = {
        "color": "Color",
        "date": "Date",
        "definition": "Definition of something",
        "individual": "Individual",
        "location": "Other location",
        "number": "Number of something",
        "person": "Individual",
        "price": "Price",
    }
    for marker, class_name in aliases.items():
        if marker in lowered and class_name in all_classes:
            return class_name
    return first_line


@torch.inference_mode()
def direct_trec_classification_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: Example,
) -> tuple[str, list[int], float, float]:
    question = trec_question_text(example.query)
    examples = trec_nearest_examples(question, example.context)
    example_text = "\n".join(f"Question: {item_question}\nType: {label}" for item_question, label in examples)
    label_text = "\n".join(f"- {label}" for label in example.all_classes)
    prompt = (
        "You are a question type classifier. Answer with exactly one label from the label set.\n"
        f"Label set:\n{label_text}\n\n"
        f"Examples:\n{example_text}\n\n"
        f"Question: {question}\nType:"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(prompt, return_tensors="pt").to(input_device)
    started = time.perf_counter()
    output_ids = model.generate(
        **encoded,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    generated_ids = output_ids[0, encoded["input_ids"].shape[-1] :].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return normalize_trec_head_prediction(generated_text, example.all_classes), generated_ids, elapsed, 0.0


def direct_structured_answer(
    example: Example,
    pages: list[Page],
    selected_ids: list[int],
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
    input_device: torch.device | None = None,
) -> tuple[str, list[int], float, float]:
    if example.task == "trec" and model is not None and tokenizer is not None and input_device is not None:
        return direct_trec_classification_answer(model, tokenizer, input_device, example)
    if example.task != "passage_retrieval_en":
        return "", [], 0.0, 0.0
    selected = set(selected_ids)
    candidates: list[tuple[float, int, str]] = []
    for page in pages:
        if page.page_id not in selected:
            continue
        match = STRUCTURED_LABEL_RE.search(page.text)
        if not match:
            continue
        label = match.group(0)
        number_match = re.search(r"\d+", label)
        if not number_match:
            continue
        candidates.append((float(page.score), -abs(page.page_id), f"Paragraph {number_match.group(0)}"))
    if not candidates:
        return "", [], 0.0, 0.0
    return max(candidates)[2], [], 0.0, 0.0


def selected_query_coverage_stats(example: Example, pages: list[Page], selected_ids: list[int], config: Config) -> dict[str, float]:
    terms = query_coverage_terms_for_example(example, config)
    selected_set = set(selected_ids)
    covered: set[str] = set()
    if terms and selected_set:
        for page in pages:
            if page.page_id in selected_set:
                covered.update(page_query_coverage_terms(page, terms))
    return {
        "terms": float(len(terms)),
        "covered": float(len(covered)),
        "recall": len(covered) / max(1, len(terms)),
    }


def page_score_stats(pages: list[Page]) -> dict[str, float]:
    if not pages:
        return {
            "max": 0.0,
            "mean": 0.0,
            "gap2": 0.0,
            "gap3": 0.0,
            "entropy": 0.0,
            "positive_fraction": 0.0,
        }
    scores = [max(0.0, float(page.score)) for page in pages]
    ordered = sorted(scores, reverse=True)
    total = sum(scores)
    if total > 0.0:
        probs = [score / total for score in scores if score > 0.0]
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs) / max(1e-12, math.log(max(2, len(scores))))
    else:
        entropy = 0.0
    return {
        "max": ordered[0],
        "mean": sum(scores) / max(1, len(scores)),
        "gap2": ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0),
        "gap3": ordered[0] - (ordered[2] if len(ordered) > 2 else 0.0),
        "entropy": entropy,
        "positive_fraction": sum(1 for score in scores if score > 0.0) / max(1, len(scores)),
    }


def evaluate_method(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    example: Example,
    bundle: PromptBundle,
    pages: list[Page],
    full_prefix_cache: Any,
    full_prefill_seconds: float,
    method: str,
    config: Config,
    attention_scores: torch.Tensor | None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; available={sorted(METHODS)}")
    selector = METHODS[method]
    selector_extra: dict[str, Any] = {}
    if method in {"h2o_observe", "snapkv_observe"}:
        if attention_scores is None:
            raise ValueError(f"{method} requires attention_scores")
        keep_indices = selector(bundle, example, pages, config, attention_scores)
    elif method == "ours_page_gather":
        selector_extra = {"model": model, "tokenizer": tokenizer}
        keep_indices = selector(bundle, example, pages, config, selector_extra)
    else:
        keep_indices = selector(bundle, example, pages, config, None)
    full_fallback_active = full_fallback_active_for_example(config, example) if method == "ours_page_gather" else False
    score_stats = page_score_stats(pages) if method == "ours_page_gather" and not full_fallback_active else None
    bridge_active = (
        bridge_active_for_example(config, example) if method == "ours_page_gather" and not full_fallback_active else False
    )
    graph_bridge_active = (
        graph_bridge_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    label_support_active = (
        label_support_active_for_example(config, example) if method == "ours_page_gather" and not full_fallback_active else False
    )
    passage_closure_active = (
        passage_closure_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    structured_fingerprint_active = (
        structured_fingerprint_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    direct_structured_answer_active = (
        direct_structured_answer_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    short_decode_active = (
        short_decode_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    output_verifier_active = (
        output_verifier_active_for_example(config, example) if method == "ours_page_gather" and not full_fallback_active else False
    )
    grounding_verifier_active = (
        grounding_verifier_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    support_window_verifier_active = (
        support_window_verifier_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    consistency_verifier_active = (
        consistency_verifier_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    score_risk_active = (
        score_risk_active_for_example(config, example) if method == "ours_page_gather" and not full_fallback_active else False
    )
    budget_ladder_active = (
        budget_ladder_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    coverage_risk_active = (
        coverage_risk_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    coverage_certificate_active = (
        coverage_certificate_active_for_example(config, example)
        if method == "ours_page_gather" and not full_fallback_active
        else False
    )
    score_risk_value = score_risk_linear_value(score_stats, config) if score_stats else 0.0
    score_risk_triggered = (
        score_risk_active
        and bool(score_stats)
        and score_risk_triggered_for_stats(score_stats, config, raw_prefix_tokens=bundle.query_start)
    )
    if consistency_verifier_active and config.ours_consistency_requires_score_risk and not score_risk_triggered:
        consistency_verifier_active = False
    uses_full_prefix_cache = method == "full_kv" or len(keep_indices) >= bundle.query_start
    gather_started = time.perf_counter()
    sparse_cache = full_prefix_cache if uses_full_prefix_cache else gather_past_key_values(full_prefix_cache, keep_indices)
    gather_seconds = 0.0 if uses_full_prefix_cache else time.perf_counter() - gather_started
    first_max_new_tokens = example.max_new_tokens
    if short_decode_active and not uses_full_prefix_cache and config.ours_short_decode_max_tokens > 0:
        first_max_new_tokens = min(first_max_new_tokens, config.ours_short_decode_max_tokens)
    if output_verifier_active and not uses_full_prefix_cache and config.ours_output_probe_max_tokens > 0:
        first_max_new_tokens = min(first_max_new_tokens, config.ours_output_probe_max_tokens)
    if direct_structured_answer_active and example.task == "trec" and not uses_full_prefix_cache:
        first_max_new_tokens = 0
    direct_structured_prediction = ""
    if (
        direct_structured_answer_active
        and not uses_full_prefix_cache
        and example.task in {"passage_retrieval_en", "trec"}
    ):
        (
            candidate_direct_prediction,
            candidate_direct_generated_ids,
            candidate_direct_query_seconds,
            candidate_direct_decode_seconds,
        ) = direct_structured_answer(
            example,
            pages,
            selected_page_ids(bundle, keep_indices),
            model,
            tokenizer,
            input_device,
        )
        if candidate_direct_prediction:
            direct_structured_prediction = candidate_direct_prediction
            prediction = candidate_direct_prediction
            generated_ids = candidate_direct_generated_ids
            query_seconds = candidate_direct_query_seconds
            decode_seconds = candidate_direct_decode_seconds
        else:
            prediction, generated_ids, query_seconds, decode_seconds = generate_with_cache(
                model,
                tokenizer,
                bundle,
                sparse_cache,
                first_max_new_tokens,
                input_device,
            )
    else:
        prediction, generated_ids, query_seconds, decode_seconds = generate_with_cache(
            model,
            tokenizer,
            bundle,
            sparse_cache,
            first_max_new_tokens,
            input_device,
        )
    output_fallback_active = False
    grounding_fallback_active = False
    support_window_fallback_active = False
    retry_fallback_active = False
    retry_full_fallback_active = False
    consistency_check_active = False
    consistency_disagreement_active = False
    consistency_full_fallback_active = False
    contract_failed = (output_verifier_active or grounding_verifier_active) and output_contract_failed(example, prediction)
    grounding_failed = False
    if grounding_verifier_active and not uses_full_prefix_cache and not contract_failed:
        grounding_failed = grounding_contract_failed(example, prediction, selected_context_text(bundle, pages, keep_indices))
    support_window_failed = False
    if (
        support_window_verifier_active
        and not uses_full_prefix_cache
        and not contract_failed
        and not grounding_failed
    ):
        support_window_failed = support_window_contract_failed(
            example,
            prediction,
            selected_context_text(bundle, pages, keep_indices),
            config,
        )
    consistency_failed = False
    if (
        consistency_verifier_active
        and not uses_full_prefix_cache
        and not contract_failed
        and not grounding_failed
        and not support_window_failed
        and config.ours_consistency_budget_tokens > config.budget_tokens
    ):
        consistency_config = replace(
            config,
            budget_tokens=config.ours_consistency_budget_tokens,
            ours_retry_budget_tokens=0,
        )
        consistency_keep_indices = selector(
            bundle,
            example,
            pages,
            consistency_config,
            {"model": model, "tokenizer": tokenizer},
        )
        consistency_started = time.perf_counter()
        consistency_uses_full_prefix_cache = len(consistency_keep_indices) >= bundle.query_start
        consistency_cache = (
            full_prefix_cache
            if consistency_uses_full_prefix_cache
            else gather_past_key_values(full_prefix_cache, consistency_keep_indices)
        )
        gather_seconds += 0.0 if consistency_uses_full_prefix_cache else time.perf_counter() - consistency_started
        consistency_max_new_tokens = example.max_new_tokens
        if config.ours_consistency_probe_max_tokens > 0:
            consistency_max_new_tokens = min(example.max_new_tokens, config.ours_consistency_probe_max_tokens)
        (
            consistency_prediction,
            _consistency_generated_ids,
            consistency_query_seconds,
            consistency_decode_seconds,
        ) = generate_with_cache(
            model,
            tokenizer,
            bundle,
            consistency_cache,
            consistency_max_new_tokens,
            input_device,
        )
        query_seconds += consistency_query_seconds
        decode_seconds += consistency_decode_seconds
        consistency_check_active = True
        consistency_contract_failed = (output_verifier_active or grounding_verifier_active) and output_contract_failed(
            example, consistency_prediction
        )
        consistency_grounding_failed = False
        if grounding_verifier_active and not consistency_uses_full_prefix_cache and not consistency_contract_failed:
            consistency_grounding_failed = grounding_contract_failed(
                example,
                consistency_prediction,
                selected_context_text(bundle, pages, consistency_keep_indices),
            )
        consistency_support_window_failed = False
        if (
            support_window_verifier_active
            and not consistency_uses_full_prefix_cache
            and not consistency_contract_failed
            and not consistency_grounding_failed
        ):
            consistency_support_window_failed = support_window_contract_failed(
                example,
                consistency_prediction,
                selected_context_text(bundle, pages, consistency_keep_indices),
                config,
            )
        consistency_failed = (
            consistency_contract_failed
            or consistency_grounding_failed
            or consistency_support_window_failed
            or answer_consistency_failed(example, prediction, consistency_prediction)
        )
        consistency_disagreement_active = bool(consistency_failed)
    if not uses_full_prefix_cache and (contract_failed or grounding_failed or support_window_failed):
        retry_accepted = False
        retry_budget = max(0, int(config.ours_retry_budget_tokens))
        if retry_budget > config.budget_tokens and retry_budget < bundle.query_start:
            retry_config = replace(config, budget_tokens=retry_budget)
            retry_keep_indices = selector(bundle, example, pages, retry_config, {"model": model, "tokenizer": tokenizer})
            retry_started = time.perf_counter()
            retry_uses_full_prefix_cache = len(retry_keep_indices) >= bundle.query_start
            retry_cache = (
                full_prefix_cache
                if retry_uses_full_prefix_cache
                else gather_past_key_values(full_prefix_cache, retry_keep_indices)
            )
            gather_seconds += 0.0 if retry_uses_full_prefix_cache else time.perf_counter() - retry_started
            retry_prediction, retry_generated_ids, retry_query_seconds, retry_decode_seconds = generate_with_cache(
                model,
                tokenizer,
                bundle,
                retry_cache,
                example.max_new_tokens,
                input_device,
            )
            query_seconds += retry_query_seconds
            decode_seconds += retry_decode_seconds
            retry_contract_failed = (output_verifier_active or grounding_verifier_active) and output_contract_failed(
                example, retry_prediction
            )
            retry_grounding_failed = False
            if grounding_verifier_active and not retry_uses_full_prefix_cache and not retry_contract_failed:
                retry_grounding_failed = grounding_contract_failed(
                    example,
                    retry_prediction,
                    selected_context_text(bundle, pages, retry_keep_indices),
                )
            retry_support_window_failed = False
            if (
                support_window_verifier_active
                and not retry_uses_full_prefix_cache
                and not retry_contract_failed
                and not retry_grounding_failed
            ):
                retry_support_window_failed = support_window_contract_failed(
                    example,
                    retry_prediction,
                    selected_context_text(bundle, pages, retry_keep_indices),
                    config,
                )
            retry_fallback_active = True
            if not retry_contract_failed and not retry_grounding_failed and not retry_support_window_failed:
                prediction = retry_prediction
                generated_ids = retry_generated_ids
                keep_indices = retry_keep_indices
                retry_accepted = True
            else:
                grounding_failed = grounding_failed or retry_grounding_failed
                support_window_failed = support_window_failed or retry_support_window_failed
        if not retry_accepted:
            fallback_prediction, fallback_generated_ids, fallback_query_seconds, fallback_decode_seconds = generate_with_cache(
                model,
                tokenizer,
                bundle,
                full_prefix_cache,
                example.max_new_tokens,
                input_device,
            )
            prediction = fallback_prediction
            generated_ids = fallback_generated_ids
            query_seconds += fallback_query_seconds
            decode_seconds += fallback_decode_seconds
            retry_full_fallback_active = retry_fallback_active
            keep_indices = list(range(bundle.query_start))
        output_fallback_active = True
        grounding_fallback_active = bool(grounding_failed)
        support_window_fallback_active = bool(support_window_failed)
    elif not uses_full_prefix_cache and consistency_failed:
        fallback_prediction, fallback_generated_ids, fallback_query_seconds, fallback_decode_seconds = generate_with_cache(
            model,
            tokenizer,
            bundle,
            full_prefix_cache,
            example.max_new_tokens,
            input_device,
        )
        prediction = fallback_prediction
        generated_ids = fallback_generated_ids
        query_seconds += fallback_query_seconds
        decode_seconds += fallback_decode_seconds
        consistency_full_fallback_active = True
        keep_indices = list(range(bundle.query_start))
    if direct_structured_answer_active and not direct_structured_prediction:
        (
            direct_structured_prediction,
            direct_structured_generated_ids,
            direct_structured_query_seconds,
            direct_structured_decode_seconds,
        ) = direct_structured_answer(
            example,
            pages,
            selected_page_ids(bundle, keep_indices),
            model,
            tokenizer,
            input_device,
        )
        if direct_structured_prediction:
            prediction = direct_structured_prediction
            generated_ids = direct_structured_generated_ids
            query_seconds += direct_structured_query_seconds
            decode_seconds += direct_structured_decode_seconds
    score = score_prediction(example.metric, prediction, example.answers, example.all_classes)
    context_kept = sum(1 for idx in keep_indices if bundle.context_token_start <= idx < bundle.query_start)
    selected_ids = selected_page_ids(bundle, keep_indices)
    query_coverage = (
        selected_query_coverage_stats(example, pages, selected_ids, config)
        if method == "ours_page_gather"
        else {"terms": 0.0, "covered": 0.0, "recall": 0.0}
    )
    return {
        "benchmark": example.benchmark,
        "task": example.task,
        "sample_id": example.sample_id,
        "method": method,
        "metric": example.metric,
        "score": score,
        "prediction": prediction.replace("\n", "\\n")[:500],
        "answers": json.dumps(example.answers, ensure_ascii=False),
        "generated_tokens": len(generated_ids),
        "prefill_seconds": full_prefill_seconds,
        "kv_gather_seconds": gather_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "online_seconds": gather_seconds + query_seconds + decode_seconds,
        "total_seconds": full_prefill_seconds + gather_seconds + query_seconds + decode_seconds,
        "raw_prefix_tokens": bundle.query_start,
        "raw_prompt_tokens": int(bundle.input_ids.shape[-1]),
        "kept_prefix_tokens": len(keep_indices),
        "kept_context_tokens": context_kept,
        "keep_fraction": len(keep_indices) / max(1, bundle.query_start),
        "budget_tokens": config.budget_tokens,
        "sink_tokens": config.sink_tokens,
        "recent_tokens": config.recent_tokens,
        "page_tokens": config.page_tokens,
        "ours_scorer": config.ours_scorer if method == "ours_page_gather" else "",
        "ours_bridge_active": int(bridge_active) if method == "ours_page_gather" else "",
        "ours_bridge_tasks": config.ours_bridge_tasks if method == "ours_page_gather" else "",
        "ours_graph_bridge_active": int(graph_bridge_active) if method == "ours_page_gather" else "",
        "ours_graph_bridge_tasks": config.ours_graph_bridge_tasks if method == "ours_page_gather" else "",
        "ours_graph_bridge_budget_fraction": config.ours_graph_bridge_budget_fraction
        if method == "ours_page_gather"
        else "",
        "ours_graph_bridge_seed_pages": config.ours_graph_bridge_seed_pages if method == "ours_page_gather" else "",
        "ours_graph_bridge_max_terms": config.ours_graph_bridge_max_terms if method == "ours_page_gather" else "",
        "ours_graph_bridge_pairs": selector_extra.get("graph_bridge_pairs", "")
        if method == "ours_page_gather"
        else "",
        "ours_graph_bridge_tokens": selector_extra.get("graph_bridge_tokens", "")
        if method == "ours_page_gather"
        else "",
        "ours_task_policy_active": int(bool(config.ours_task_policy_json)) if method == "ours_page_gather" else "",
        "ours_full_fallback_active": int(full_fallback_active) if method == "ours_page_gather" else "",
        "ours_full_fallback_tasks": config.ours_full_fallback_tasks if method == "ours_page_gather" else "",
        "ours_label_support_active": int(label_support_active) if method == "ours_page_gather" else "",
        "ours_label_support_tasks": config.ours_label_support_tasks if method == "ours_page_gather" else "",
        "ours_passage_closure_active": int(passage_closure_active) if method == "ours_page_gather" else "",
        "ours_passage_closure_tasks": config.ours_passage_closure_tasks if method == "ours_page_gather" else "",
        "ours_passage_closure_budget_fraction": config.ours_passage_closure_budget_fraction
        if method == "ours_page_gather"
        else "",
        "ours_passage_closure_radius_pages": config.ours_passage_closure_radius_pages
        if method == "ours_page_gather"
        else "",
        "ours_passage_closure_tokens": selector_extra.get("passage_closure_tokens", "")
        if method == "ours_page_gather"
        else "",
        "ours_structured_fingerprint_active": int(structured_fingerprint_active) if method == "ours_page_gather" else "",
        "ours_structured_fingerprint_tasks": config.ours_structured_fingerprint_tasks
        if method == "ours_page_gather"
        else "",
        "ours_structured_fingerprint_budget_fraction": config.ours_structured_fingerprint_budget_fraction
        if method == "ours_page_gather"
        else "",
        "ours_structured_fingerprint_labels": selector_extra.get("structured_fingerprint_labels", "")
        if method == "ours_page_gather"
        else "",
        "ours_structured_fingerprint_tokens": selector_extra.get("structured_fingerprint_tokens", "")
        if method == "ours_page_gather"
        else "",
        "ours_direct_structured_answer_active": int(direct_structured_answer_active)
        if method == "ours_page_gather"
        else "",
        "ours_direct_structured_answer_tasks": config.ours_direct_structured_answer_tasks
        if method == "ours_page_gather"
        else "",
        "ours_direct_structured_answer_used": int(bool(direct_structured_prediction))
        if method == "ours_page_gather"
        else "",
        "ours_short_decode_active": int(short_decode_active) if method == "ours_page_gather" else "",
        "ours_short_decode_tasks": config.ours_short_decode_tasks if method == "ours_page_gather" else "",
        "ours_short_decode_max_tokens": config.ours_short_decode_max_tokens if method == "ours_page_gather" else "",
        "first_max_new_tokens": first_max_new_tokens,
        "ours_output_verifier_active": int(output_verifier_active) if method == "ours_page_gather" else "",
        "ours_output_fallback_active": int(output_fallback_active) if method == "ours_page_gather" else "",
        "ours_output_verifier_tasks": config.ours_output_verifier_tasks if method == "ours_page_gather" else "",
        "ours_output_probe_max_tokens": config.ours_output_probe_max_tokens if method == "ours_page_gather" else "",
        "ours_retry_budget_tokens": config.ours_retry_budget_tokens if method == "ours_page_gather" else "",
        "ours_retry_fallback_active": int(retry_fallback_active) if method == "ours_page_gather" else "",
        "ours_retry_full_fallback_active": int(retry_full_fallback_active) if method == "ours_page_gather" else "",
        "ours_score_risk_active": int(score_risk_active) if method == "ours_page_gather" else "",
        "ours_score_risk_triggered": int(score_risk_triggered) if method == "ours_page_gather" else "",
        "ours_score_risk_tasks": config.ours_score_risk_tasks if method == "ours_page_gather" else "",
        "ours_score_risk_budget_tokens": config.ours_score_risk_budget_tokens if method == "ours_page_gather" else "",
        "ours_score_risk_min_gap2": config.ours_score_risk_min_gap2 if method == "ours_page_gather" else "",
        "ours_score_risk_min_gap3": config.ours_score_risk_min_gap3 if method == "ours_page_gather" else "",
        "ours_score_risk_max_gap2": config.ours_score_risk_max_gap2 if method == "ours_page_gather" else "",
        "ours_score_risk_max_gap3": config.ours_score_risk_max_gap3 if method == "ours_page_gather" else "",
        "ours_score_risk_max_entropy": config.ours_score_risk_max_entropy if method == "ours_page_gather" else "",
        "ours_score_risk_entropy_at_most": config.ours_score_risk_entropy_at_most
        if method == "ours_page_gather"
        else "",
        "ours_score_risk_min_top_score": config.ours_score_risk_min_top_score if method == "ours_page_gather" else "",
        "ours_score_risk_raw_prefix_at_most": config.ours_score_risk_raw_prefix_at_most
        if method == "ours_page_gather"
        else "",
        "ours_score_risk_raw_prefix_at_least": config.ours_score_risk_raw_prefix_at_least
        if method == "ours_page_gather"
        else "",
        "ours_score_risk_linear_threshold": config.ours_score_risk_linear_threshold if method == "ours_page_gather" else "",
        "ours_score_risk_gap2_weight": config.ours_score_risk_gap2_weight if method == "ours_page_gather" else "",
        "ours_score_risk_gap3_weight": config.ours_score_risk_gap3_weight if method == "ours_page_gather" else "",
        "ours_score_risk_top_score_weight": config.ours_score_risk_top_score_weight if method == "ours_page_gather" else "",
        "ours_score_risk_linear_value": score_risk_value if method == "ours_page_gather" else "",
        "ours_budget_ladder_active": int(budget_ladder_active) if method == "ours_page_gather" else "",
        "ours_budget_ladder_tasks": config.ours_budget_ladder_tasks if method == "ours_page_gather" else "",
        "ours_budget_ladder_tokens": config.ours_budget_ladder_tokens if method == "ours_page_gather" else "",
        "ours_budget_ladder_selected_budget": selector_extra.get("budget_ladder_selected_budget", "")
        if method == "ours_page_gather"
        else "",
        "ours_budget_ladder_level": selector_extra.get("budget_ladder_level", "")
        if method == "ours_page_gather"
        else "",
        "ours_budget_ladder_reasons": selector_extra.get("budget_ladder_reasons", "")
        if method == "ours_page_gather"
        else "",
        "ours_budget_ladder_gap2_level": selector_extra.get("budget_ladder_gap2_level", "")
        if method == "ours_page_gather"
        else "",
        "ours_budget_ladder_entropy_level": selector_extra.get("budget_ladder_entropy_level", "")
        if method == "ours_page_gather"
        else "",
        "ours_budget_ladder_top_score_level": selector_extra.get("budget_ladder_top_score_level", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_mmr_weight": config.ours_coverage_mmr_weight if method == "ours_page_gather" else "",
        "ours_coverage_mmr_max_terms": config.ours_coverage_mmr_max_terms if method == "ours_page_gather" else "",
        "ours_coverage_certificate_active": int(coverage_certificate_active) if method == "ours_page_gather" else "",
        "ours_coverage_certificate_tasks": config.ours_coverage_certificate_tasks if method == "ours_page_gather" else "",
        "ours_coverage_certificate_budget_fraction": config.ours_coverage_certificate_budget_fraction
        if method == "ours_page_gather"
        else "",
        "ours_coverage_certificate_min_terms": config.ours_coverage_certificate_min_terms
        if method == "ours_page_gather"
        else "",
        "ours_coverage_certificate_terms": selector_extra.get("coverage_certificate_terms", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_certificate_covered": selector_extra.get("coverage_certificate_covered", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_certificate_recall": selector_extra.get("coverage_certificate_recall", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_certificate_tokens": selector_extra.get("coverage_certificate_tokens", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_risk_active": int(coverage_risk_active) if method == "ours_page_gather" else "",
        "ours_coverage_risk_triggered": selector_extra.get("coverage_risk_triggered", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_risk_initial_terms": selector_extra.get("coverage_risk_initial_terms", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_risk_initial_recall": selector_extra.get("coverage_risk_initial_recall", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_risk_escalation_budget": selector_extra.get("coverage_risk_escalation_budget", "")
        if method == "ours_page_gather"
        else "",
        "ours_coverage_risk_min_recall": config.ours_coverage_risk_min_recall if method == "ours_page_gather" else "",
        "ours_coverage_risk_min_terms": config.ours_coverage_risk_min_terms if method == "ours_page_gather" else "",
        "ours_coverage_risk_budget_tokens": config.ours_coverage_risk_budget_tokens if method == "ours_page_gather" else "",
        "ours_consistency_verifier_active": int(consistency_verifier_active) if method == "ours_page_gather" else "",
        "ours_consistency_check_active": int(consistency_check_active) if method == "ours_page_gather" else "",
        "ours_consistency_disagreement_active": int(consistency_disagreement_active) if method == "ours_page_gather" else "",
        "ours_consistency_full_fallback_active": int(consistency_full_fallback_active)
        if method == "ours_page_gather"
        else "",
        "ours_consistency_verifier_tasks": config.ours_consistency_verifier_tasks if method == "ours_page_gather" else "",
        "ours_consistency_budget_tokens": config.ours_consistency_budget_tokens if method == "ours_page_gather" else "",
        "ours_consistency_probe_max_tokens": config.ours_consistency_probe_max_tokens
        if method == "ours_page_gather"
        else "",
        "ours_consistency_requires_score_risk": config.ours_consistency_requires_score_risk
        if method == "ours_page_gather"
        else "",
        "ours_grounding_verifier_active": int(grounding_verifier_active) if method == "ours_page_gather" else "",
        "ours_grounding_fallback_active": int(grounding_fallback_active) if method == "ours_page_gather" else "",
        "ours_grounding_verifier_tasks": config.ours_grounding_verifier_tasks if method == "ours_page_gather" else "",
        "ours_support_window_verifier_active": int(support_window_verifier_active)
        if method == "ours_page_gather"
        else "",
        "ours_support_window_fallback_active": int(support_window_fallback_active)
        if method == "ours_page_gather"
        else "",
        "ours_support_window_verifier_tasks": config.ours_support_window_verifier_tasks
        if method == "ours_page_gather"
        else "",
        "ours_support_window_radius_words": config.ours_support_window_radius_words
        if method == "ours_page_gather"
        else "",
        "ours_support_window_min_query_terms": config.ours_support_window_min_query_terms
        if method == "ours_page_gather"
        else "",
        "ours_score_max": score_stats["max"] if score_stats else "",
        "ours_score_mean": score_stats["mean"] if score_stats else "",
        "ours_score_gap2": score_stats["gap2"] if score_stats else "",
        "ours_score_gap3": score_stats["gap3"] if score_stats else "",
        "ours_score_entropy": score_stats["entropy"] if score_stats else "",
        "ours_score_positive_fraction": score_stats["positive_fraction"] if score_stats else "",
        "ours_query_coverage_terms": query_coverage["terms"] if method == "ours_page_gather" else "",
        "ours_query_coverage_covered": query_coverage["covered"] if method == "ours_page_gather" else "",
        "ours_query_coverage_recall": query_coverage["recall"] if method == "ours_page_gather" else "",
        "selected_pages": ",".join(str(page_id) for page_id in selected_ids),
        "page_count": len(pages),
        "context_length_field": example.length,
        "force_no_chat_tasks": config.force_no_chat_tasks,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark"]), str(row["task"]), str(row["method"]))].append(row)
        grouped[(str(row["benchmark"]), "ALL", str(row["method"]))].append(row)
        grouped[("ALL", "ALL", str(row["method"]))].append(row)
    summary = []
    for (benchmark, task, method), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        summary.append(
            {
                "benchmark": benchmark,
                "task": task,
                "method": method,
                "samples": len(subset),
                "score": sum(float(row["score"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_online_seconds": sum(float(row["online_seconds"]) for row in subset) / n,
                "mean_prefill_seconds": sum(float(row["prefill_seconds"]) for row in subset) / n,
                "mean_kv_gather_seconds": sum(float(row["kv_gather_seconds"]) for row in subset) / n,
                "mean_query_seconds": sum(float(row["query_seconds"]) for row in subset) / n,
                "mean_decode_seconds": sum(float(row["decode_seconds"]) for row in subset) / n,
                "mean_raw_prefix_tokens": sum(int(row["raw_prefix_tokens"]) for row in subset) / n,
                "mean_kept_prefix_tokens": sum(int(row["kept_prefix_tokens"]) for row in subset) / n,
                "mean_kept_context_tokens": sum(int(row["kept_context_tokens"]) for row in subset) / n,
                "mean_keep_fraction": sum(float(row["keep_fraction"]) for row in subset) / n,
            }
        )
    return summary


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    examples: list[Example] = []
    benchmarks = set(parse_list(config.benchmarks))
    if "longbench" in benchmarks:
        examples.extend(load_longbench_examples(config))
    if "ruler" in benchmarks:
        examples.extend(load_ruler_examples(config, config.model_name_or_path))
    sampled_ids = [
        {
            "benchmark": example.benchmark,
            "task": example.task,
            "sample_id": example.sample_id,
            "length": example.length,
        }
        for example in examples
    ]
    write_csv(output_dir / "sampled_ids.csv", sampled_ids)
    (output_dir / "sampled_ids.json").write_text(json.dumps(sampled_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    methods = parse_list(config.methods)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; available={sorted(METHODS)}")
    # Full KV generation can extend cache objects in place. Run it last so sparse selectors
    # always see the original prefix cache without requiring a full long-context clone.
    if "full_kv" in methods:
        methods = [method for method in methods if method != "full_kv"] + ["full_kv"]

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    needs_attention = any(method in {"h2o_observe", "snapkv_observe"} for method in methods)
    for idx, example in enumerate(examples):
        example_config = config_for_example(config, example)
        bundle, pages, _, _, _ = build_bundle(tokenizer, example, example_config)
        if (idx + 1) % config.log_every == 0 or idx == 0:
            print(
                f"[{idx + 1}/{len(examples)}] {example.benchmark}/{example.task}/{example.sample_id} "
                f"prefix_tokens={bundle.query_start} pages={len(pages)} "
                f"budget={example_config.budget_tokens} page_tokens={example_config.page_tokens} "
                f"scorer={example_config.ours_scorer}",
                flush=True,
            )
        full_prefix_cache, prefill_seconds = prefill_prefix(
            model,
            bundle,
            input_device,
            example_config.prefill_chunk_tokens,
        )
        attention_scores = None
        if needs_attention:
            attention_scores = attention_scores_from_suffix(model, bundle, full_prefix_cache, input_device)
        for method in methods:
            row = evaluate_method(
                model,
                tokenizer,
                input_device,
                example,
                bundle,
                pages,
                full_prefix_cache,
                prefill_seconds,
                method,
                example_config,
                attention_scores,
            )
            rows.append(row)
            print(
                f"  {method}: score={row['score']:.3f} kept={row['kept_prefix_tokens']}/{row['raw_prefix_tokens']} "
                f"online={row['online_seconds']:.3f}s pred={row['prediction'][:80]}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del full_prefix_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(output_dir / "task_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "examples": len(examples),
        "methods": methods,
        "benchmarks": sorted(benchmarks),
        "labeling": "controlled_public_longbench_ruler_generation_kv_gather",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
