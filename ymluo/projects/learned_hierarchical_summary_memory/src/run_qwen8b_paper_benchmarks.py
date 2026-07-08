from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_static_summary_ppl_speed import content_words, resolve_dtype, split_sentences, word_tokens  # noqa: E402

try:
    from memory_policy_router_runtime import load_router  # noqa: E402
except Exception:
    load_router = None


LONG_BENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story and a question. Answer the question as concisely as possible.\n\n"
        "Story:\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer concisely. If unanswerable, write unanswerable.\n\n"
        "Article:\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "passage_retrieval_en": (
        "Here are paragraphs from Wikipedia and an abstract. Determine which paragraph the abstract is from.\n\n"
        "{context}\n\nAbstract:\n{input}\n\nAnswer:"
    ),
    "passage_count": (
        "There are paragraphs below. Determine how many unique paragraphs there are after removing duplicates.\n\n"
        "{context}\n\nThe final answer is:"
    ),
    "gov_report": "Write a concise summary of this government report.\n\nReport:\n{context}\n\nSummary:",
    "multi_news": "Write a concise summary of all news passages.\n\nNews:\n{context}\n\nSummary:",
}

SUMMARY_TASKS = {"gov_report", "multi_news"}
EXACT_LONG_BENCH_TASKS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "passage_retrieval_en",
    "passage_count",
}
RULER_TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_multikey_1",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
}

CALIBRATED_FLOOR_ACTIONS = {
    "longbench": "recent_plus_b128_span_top12_b0_a0",
    "ruler_4096": "recent_plus_b256_span_top3_b0_a0",
    "ruler_8192": "recent_plus_b256_span_top3_b0_a0",
    "ruler_16384": "recent_plus_b512_span_top3_b0_a0",
}


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    adapter_path: str
    longbench_data_dir: str
    ruler_data_dir: str
    longbench_tasks: tuple[str, ...]
    ruler_tasks: tuple[str, ...]
    ruler_context_lengths: tuple[int, ...]
    methods: tuple[str, ...]
    max_examples_per_task: int
    case_ids: tuple[str, ...]
    block_tokens: int
    recent_tokens: int
    max_input_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_new_tokens_exact: int
    max_new_tokens_summary: int
    dtype: str
    attn_implementation: str
    device_map: str
    cuda_visible_devices: str
    router_path: str
    seed: int


@dataclass(frozen=True)
class BenchCase:
    benchmark: str
    task: str
    case_id: str
    context: str
    query: str
    answers: tuple[str, ...]
    length: int


@dataclass
class TrialRow:
    benchmark: str
    task: str
    case_id: str
    method: str
    routed_action: str
    prompt_tokens: int
    token_ratio_vs_full_raw: float
    output_tokens: int
    seconds: float
    prediction: str
    answers: str
    exact_correct: int
    rouge_l: float
    score: float


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Qwen3-8B paper-style benchmarks for task-adaptive summary memory.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_paper_benchmarks")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_path", default="", help="Optional PEFT LoRA adapter directory to load on top of the base model.")
    parser.add_argument("--longbench_data_dir", default="ymluo/external/KVCache-Factory/data/LongBench")
    parser.add_argument("--ruler_data_dir", default="ymluo/external/KVCache-Factory/data/RULER")
    parser.add_argument(
        "--longbench_tasks",
        default="hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,qasper,gov_report,multi_news",
    )
    parser.add_argument(
        "--ruler_tasks",
        default="niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe",
    )
    parser.add_argument("--ruler_context_lengths", default="4096,8192,16384")
    parser.add_argument(
        "--methods",
        default="full_raw,summary10,summary100,summary1000,summary1_8,summary1_4,summary1_2,static_hier,retrieval_raw_k1,retrieval_raw_k2,router,router_conservative",
    )
    parser.add_argument("--max_examples_per_task", type=int, default=6)
    parser.add_argument("--case_ids", default="", help="Optional comma-separated case ids to run instead of the first N cases.")
    parser.add_argument("--block_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=24000)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_new_tokens_exact", type=int, default=48)
    parser.add_argument("--max_new_tokens_summary", type=int, default=160)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument(
        "--router_path",
        default="ymluo/projects/learned_hierarchical_summary_memory/outputs/memory_policy_router_10texts_s4_20260704/router.pt",
    )
    parser.add_argument("--seed", type=int, default=2026070403)
    args = parser.parse_args()
    return Config(
        **{
            **vars(args),
            "longbench_tasks": parse_csv_tuple(args.longbench_tasks),
            "ruler_tasks": parse_csv_tuple(args.ruler_tasks),
            "ruler_context_lengths": parse_int_tuple(args.ruler_context_lengths),
            "methods": parse_csv_tuple(args.methods),
            "case_ids": parse_csv_tuple(args.case_ids),
        }
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match_any(prediction: str, answers: tuple[str, ...]) -> int:
    pred = normalize_answer(prediction)
    for answer in answers:
        ans = normalize_answer(answer)
        if ans and ans in pred:
            return 1
    return 0


def rouge_l_f1(prediction: str, answers: tuple[str, ...]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        if not gold:
            continue
        dp = [0] * (len(gold) + 1)
        for token in pred_tokens:
            prev = 0
            for idx, gold_token in enumerate(gold, start=1):
                tmp = dp[idx]
                if token == gold_token:
                    dp[idx] = prev + 1
                else:
                    dp[idx] = max(dp[idx], dp[idx - 1])
                prev = tmp
        lcs = dp[-1]
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        best = max(best, f1)
    return best


def load_longbench_cases(config: Config) -> list[BenchCase]:
    cases: list[BenchCase] = []
    data_dir = Path(config.longbench_data_dir)
    for task in config.longbench_tasks:
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                case_id = str(idx)
                if not config.case_ids and idx >= config.max_examples_per_task:
                    break
                row = json.loads(line)
                case_id = str(row.get("_id") or idx)
                if config.case_ids and case_id not in config.case_ids:
                    continue
                cases.append(
                    BenchCase(
                        benchmark="longbench",
                        task=task,
                        case_id=case_id,
                        context=str(row.get("context", "")),
                        query=str(row.get("input", "")),
                        answers=tuple(str(item) for item in row.get("answers", [])),
                        length=int(row.get("length", 0) or 0),
                    )
                )
    return cases


def split_ruler_input(text: str) -> tuple[str, str]:
    markers = [
        "What is",
        "Question:",
        "Now",
        "What are",
        "Find",
        "Please",
        "Tell me",
        "Output",
    ]
    tail_window = text[-1600:]
    best = -1
    for marker in markers:
        pos = tail_window.rfind(marker)
        if pos > best:
            best = pos
    if best >= 0:
        cut = len(text) - len(tail_window) + best
        return text[:cut], text[cut:]
    cut = max(0, len(text) - 900)
    return text[:cut], text[cut:]


def load_ruler_cases(config: Config) -> list[BenchCase]:
    cases: list[BenchCase] = []
    base = Path(config.ruler_data_dir)
    for context_length in config.ruler_context_lengths:
        for task in config.ruler_tasks:
            path = base / str(context_length) / f"{task}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    case_id = str(idx)
                    if not config.case_ids and idx >= config.max_examples_per_task:
                        break
                    row = json.loads(line)
                    case_id = str(row.get("index", idx))
                    if config.case_ids and case_id not in config.case_ids:
                        continue
                    memory, query = split_ruler_input(str(row["input"]))
                    cases.append(
                        BenchCase(
                            benchmark=f"ruler_{context_length}",
                            task=task,
                            case_id=case_id,
                            context=memory,
                            query=query,
                            answers=tuple(str(item) for item in row.get("outputs", [])),
                            length=int(row.get("length", 0) or 0),
                        )
                    )
    return cases


def token_blocks(tokenizer: Any, text: str, block_tokens: int) -> list[str]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = []
    for idx in range(0, len(ids), block_tokens):
        blocks.append(tokenizer.decode(ids[idx : idx + block_tokens], skip_special_tokens=True))
    return blocks


def token_id_blocks(tokenizer: Any, text: str, block_tokens: int) -> list[list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [ids[idx : idx + block_tokens] for idx in range(0, len(ids), block_tokens)]


def fit_words(items: list[str], max_words: int) -> str:
    words: list[str] = []
    for item in items:
        for word in item.split():
            if len(words) >= max_words:
                return " ".join(words)
            words.append(word)
    return " ".join(words)


def summarize_block(text: str, word_budget: int) -> str:
    if not text.strip():
        return ""
    if word_budget <= 12:
        counts = Counter(content_words(text))
        return " ".join(word for word, _ in counts.most_common(max(1, word_budget)))
    sentences = split_sentences(text)
    if not sentences:
        return fit_words([text], word_budget)
    terms = {word for word, _ in Counter(content_words(text)).most_common(24)}
    scored = []
    for sentence in sentences:
        score = sum(1 for word in content_words(sentence) if word in terms)
        score += 0.1 * min(len(sentence.split()), 40)
        scored.append((score, sentence))
    scored.sort(reverse=True)
    return fit_words([sentence for _, sentence in scored], word_budget)


def summarize_block_to_token_budget(tokenizer: Any, text: str, token_budget: int) -> str:
    if not text.strip() or token_budget <= 0:
        return ""
    if token_budget <= 24:
        counts = Counter(content_words(text))
        short = " ".join(word for word, _ in counts.most_common(max(1, token_budget)))
        ids = tokenizer(short, add_special_tokens=False)["input_ids"][:token_budget]
        return tokenizer.decode(ids, skip_special_tokens=True)

    sentences = split_sentences(text)
    if not sentences:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:token_budget]
        return tokenizer.decode(ids, skip_special_tokens=True)

    terms = {word for word, _ in Counter(content_words(text)).most_common(32)}
    scored = []
    for sentence in sentences:
        words = content_words(sentence)
        score = sum(1 for word in words if word in terms)
        score += 0.05 * min(len(words), 80)
        scored.append((score, sentence))
    scored.sort(reverse=True)

    chosen: list[str] = []
    used = 0
    for _, sentence in scored:
        ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if used + len(ids) > token_budget and chosen:
            continue
        remain = token_budget - used
        if remain <= 0:
            break
        if len(ids) > remain:
            ids = ids[:remain]
            sentence = tokenizer.decode(ids, skip_special_tokens=True)
        chosen.append(sentence)
        used += len(ids)
        if used >= token_budget:
            break
    return " ".join(chosen)


def build_summary_memory(tokenizer: Any, context: str, config: Config, level: str) -> str:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    if level == "summary10":
        budget = config.summary10_words
    elif level == "summary100":
        budget = config.summary100_words
    elif level == "summary1000":
        budget = config.summary1000_words
    else:
        raise ValueError(level)
    summaries = [summarize_block(block, budget) for block in blocks]
    return "\n".join(f"[block {idx}] {summary}" for idx, summary in enumerate(summaries) if summary)


def summary_ratio(level: str) -> float | None:
    return {
        "summary1_8": 1.0 / 8.0,
        "summary1_4": 1.0 / 4.0,
        "summary1_2": 1.0 / 2.0,
    }.get(level)


def build_ratio_summary_memory(tokenizer: Any, context: str, config: Config, level: str) -> str:
    ratio = summary_ratio(level)
    if ratio is None:
        raise ValueError(level)
    blocks = token_id_blocks(tokenizer, context, config.block_tokens)
    summaries = []
    for idx, block_ids in enumerate(blocks):
        block_text = tokenizer.decode(block_ids, skip_special_tokens=True)
        budget = max(8, int(round(len(block_ids) * ratio)))
        summary = summarize_block_to_token_budget(tokenizer, block_text, budget)
        if summary:
            summaries.append(f"[block {idx}] {summary}")
    return "\n".join(summaries)


def retrieve_blocks(tokenizer: Any, context: str, query: str, config: Config, top_k: int) -> str:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    query_terms = set(content_words(query))
    scored = []
    for idx, block in enumerate(blocks):
        block_terms = set(content_words(block))
        overlap = len(query_terms & block_terms)
        # Numbers and exact identifiers matter for RULER / retrieval tasks.
        query_numbers = set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", query))
        overlap += 3 * len(query_numbers & set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", block)))
        scored.append((overlap, -idx, idx, block))
    scored.sort(reverse=True)
    chosen = [(idx, block) for score, _, idx, block in scored[:top_k] if score > 0]
    if not chosen:
        chosen = [(idx, block) for _, _, idx, block in scored[:top_k]]
    return "\n\n".join(f"[raw block {idx}]\n{block}" for idx, block in chosen)


def score_blocks(tokenizer: Any, context: str, query: str, config: Config) -> list[tuple[int, int, str]]:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    query_terms = set(content_words(query))
    query_numbers = set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", query))
    scored = []
    for idx, block in enumerate(blocks):
        block_terms = set(content_words(block))
        overlap = len(query_terms & block_terms)
        overlap += 3 * len(query_numbers & set(re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", block)))
        scored.append((overlap, idx, block))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored


def retrieve_span_blocks(
    tokenizer: Any,
    context: str,
    query: str,
    config: Config,
    before: int,
    after: int,
    max_spans: int = 1,
) -> str:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    if not blocks:
        return ""
    scored = score_blocks(tokenizer, context, query, config)
    centers = [idx for score, idx, _ in scored[:max_spans] if score > 0]
    if not centers:
        centers = [scored[0][1]]
    selected: set[int] = set()
    for center in centers:
        for idx in range(max(0, center - before), min(len(blocks) - 1, center + after) + 1):
            selected.add(idx)
    return "\n\n".join(f"[raw span block {idx}]\n{blocks[idx]}" for idx in sorted(selected))


def retrieve_prefix_to_evidence(tokenizer: Any, context: str, query: str, config: Config, top_k: int = 1) -> str:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    if not blocks:
        return ""
    scored = score_blocks(tokenizer, context, query, config)
    centers = [idx for score, idx, _ in scored[:top_k] if score > 0]
    if not centers:
        centers = [scored[0][1]]
    center = max(centers)
    return "\n\n".join(f"[raw prefix block {idx}]\n{blocks[idx]}" for idx in range(center + 1))


def kv_safe_rule_action(case: BenchCase, tokenizer: Any, config: Config) -> str:
    context_tokens = len(tokenizer(case.context, add_special_tokens=False)["input_ids"])
    lower_query = case.query.lower()
    exact = case.benchmark.startswith("ruler") or case.task in EXACT_LONG_BENCH_TASKS
    global_needles = (
        "all ",
        "all the",
        "every",
        "list",
        "count",
        "how many",
        "number of",
        "most common",
        "frequently appeared",
        "top 10",
        "three most",
        "summary",
        "summarize",
    )
    multi_evidence = (
        "all ",
        "every",
        "variables",
        "multi",
        "compare",
        "and ",
        " or ",
        "count",
        "how many",
    )
    if case.task in SUMMARY_TASKS:
        return "recent_plus_summary1_8" if context_tokens >= 12000 else "recent_plus_summary1_4"
    if any(phrase in lower_query for phrase in global_needles):
        if context_tokens >= 12000 or case.task in {"passage_count", "cwe", "fwe", "vt"}:
            return "recent_plus_full_old_raw"
        return "recent_plus_prefix_to_farthest_top3"
    if case.benchmark.startswith("ruler"):
        if any(phrase in lower_query for phrase in multi_evidence):
            return "recent_plus_prefix_to_farthest_top2"
        return "recent_plus_span_top2_b0_a0"
    if exact and context_tokens >= 12000:
        return "recent_plus_prefix_to_farthest_top2"
    if exact:
        return "recent_plus_span_top2_b1_a0"
    return "recent_plus_summary1_8"


def recent_text(tokenizer: Any, context: str, recent_tokens: int) -> str:
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[-recent_tokens:], skip_special_tokens=True) if ids else ""


def static_hier_memory(tokenizer: Any, context: str, config: Config) -> str:
    blocks = token_blocks(tokenizer, context, config.block_tokens)
    parts = []
    for idx, block in enumerate(blocks):
        distance = len(blocks) - idx
        if distance == 1:
            summary = summarize_block(block, config.summary1000_words)
        elif distance <= 3:
            summary = summarize_block(block, config.summary100_words)
        else:
            summary = summarize_block(block, config.summary10_words)
        parts.append(f"[block {idx}] {summary}")
    return "\n".join(parts)


def router_features(tokenizer: Any, case: BenchCase, config: Config) -> tuple[list[float], str]:
    exact = case.benchmark.startswith("ruler") or case.task in EXACT_LONG_BENCH_TASKS
    task_family = "exact" if exact else "generation"
    prefix_ids = tokenizer(case.context, add_special_tokens=False)["input_ids"]
    recent_len = min(config.recent_tokens, len(prefix_ids))
    older_tokens = max(0, len(prefix_ids) - recent_len)
    num_blocks = math.ceil(older_tokens / config.block_tokens) if config.block_tokens else 0
    query = case.query
    query_terms = set(content_words(query))
    blocks = token_blocks(tokenizer, case.context, config.block_tokens)
    scored_blocks = []
    for idx, block in enumerate(blocks):
        scored_blocks.append((float(len(query_terms & set(content_words(block)))), idx))
    scored_blocks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    scores = [score for score, _ in scored_blocks]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top3 = scores[2] if len(scores) > 2 else 0.0
    top1_idx = scored_blocks[0][1] if scored_blocks else 0
    top2_idx = scored_blocks[1][1] if len(scored_blocks) > 1 else top1_idx
    block_denom = max(1.0, float(len(blocks) - 1))
    top1_pos = float(top1_idx) / block_denom
    top2_pos = float(top2_idx) / block_denom
    recent_block_threshold = max(0, len(blocks) - math.ceil(max(1, recent_len) / max(1, config.block_tokens)) - 1)
    top1_is_recent = 1.0 if top1_idx >= recent_block_threshold else 0.0
    gap = top1 - top2
    positive = float(sum(1 for score in scores if score > 0))
    query_words = word_tokens(query)
    sample_text = tokenizer.decode(prefix_ids[: min(len(prefix_ids), 2048)], skip_special_tokens=True)
    sample_words = word_tokens(sample_text)
    unique_ratio = len({word.lower() for word in sample_words}) / max(1, len(sample_words))
    nums_log = math.log1p(len(re.findall(r"\d+", sample_text)))
    caps_log = math.log1p(len(re.findall(r"\b[A-Z][a-z]{2,}\b", sample_text)))
    lower = query.lower()
    exact_kw = sum(1 for word in ("exact", "code", "access", "private", "value", "answer with only") if word in lower)
    quote_kw = sum(1 for word in ("quote", "verbatim", "span", "sentence") if word in lower)
    count_kw = sum(1 for word in ("count", "how many", "number of", "total") if word in lower)
    list_kw = sum(1 for word in ("list", "all", "every", "enumerate") if word in lower)
    compare_kw = sum(1 for word in ("compare", "contrast", "difference", "before", "after") if word in lower)
    denom = max(1.0, float(len(query_terms)))
    return (
        [
            0.0 if exact else 1.0,
            1.0 if exact else 0.0,
            float(len(query)),
            float(len(query_words)),
            1.0 if "?" in query else 0.0,
            float(exact_kw),
            float(quote_kw),
            float(count_kw),
            float(list_kw),
            float(compare_kw),
            float(len(re.findall(r"\d+", query))),
            1.0 if re.search(r"\ball\b", lower) else 0.0,
            float(len(prefix_ids)),
            float(older_tokens),
            float(recent_len),
            float(config.block_tokens),
            float(num_blocks),
            float(config.summary10_words),
            float(config.summary100_words),
            float(config.summary1000_words),
            top1,
            top2,
            top3,
            gap,
            positive,
            top1 / denom,
            top2 / denom,
            gap / denom,
            top1_pos,
            top2_pos,
            top1_is_recent,
            unique_ratio,
            nums_log,
            caps_log,
        ],
        task_family,
    )


def rule_router_action(case: BenchCase) -> str:
    if case.benchmark.startswith("ruler"):
        return "retrieval_raw_k2"
    if case.task in SUMMARY_TASKS:
        return "summary100"
    if case.task in {"passage_count", "passage_retrieval_en", "hotpotqa", "2wikimqa", "musique"}:
        return "retrieval_raw_k2"
    return "retrieval_raw_k1"


def safety_override_action(case: BenchCase, action: str) -> str:
    exact = case.benchmark.startswith("ruler") or case.task in EXACT_LONG_BENCH_TASKS
    if not exact:
        return action
    old_action = action.removeprefix("recent_plus_") if action.startswith("recent_plus_") else action
    retrieval_prefix = "recent_plus_retrieval_raw_k" if action.startswith("recent_plus_") else "retrieval_raw_k"
    lower = case.query.lower()
    high_risk_multi = any(
        phrase in lower
        for phrase in (
            "all the",
            "all ",
            "variables",
            "assigned the value",
            "most common",
            "frequently appeared",
            "top 10",
            "three most",
            "list all",
        )
    )
    unsafe_compressed = {
        "recent_only",
        "summary10",
        "summary100",
        "summary1000",
        "summary1_8",
        "summary1_4",
        "summary1_2",
        "static_hier",
    }
    if case.benchmark.startswith("ruler"):
        if any(phrase in lower for phrase in ("most common", "frequently appeared", "top 10", "three most")):
            return f"{retrieval_prefix}1"
        if high_risk_multi or old_action in unsafe_compressed or old_action == "retrieval_raw_k1":
            return f"{retrieval_prefix}2"
        return action
    if old_action in unsafe_compressed:
        return f"{retrieval_prefix}1"
    return action


def kv_safe_router_override(case: BenchCase, action: str) -> str:
    exact = case.benchmark.startswith("ruler") or case.task in EXACT_LONG_BENCH_TASKS
    old_action = action.removeprefix("recent_plus_") if action.startswith("recent_plus_") else action
    lower = case.query.lower()
    length_match = re.search(r"ruler_(\d+)", case.benchmark)
    ruler_length = int(length_match.group(1)) if length_match else 0
    compressed = {
        "recent_only",
        "summary10",
        "summary100",
        "summary1000",
        "summary1_8",
        "summary1_4",
        "summary1_2",
        "static_hier",
    }
    if case.task in SUMMARY_TASKS:
        return action
    if not exact:
        return action
    if old_action in compressed:
        if case.benchmark.startswith("ruler"):
            return "recent_plus_span_top3_b0_a0"
        return "recent_plus_retrieval_raw_k2"
    if case.benchmark.startswith("ruler") and case.task in {"vt", "cwe", "fwe"}:
        if old_action in {"span_top2_b0_a0", "span_b0_a0", "prefix_to_evidence"}:
            return "recent_plus_span_top3_b0_a0"
    if case.benchmark.startswith("ruler") and ruler_length >= 16384:
        if case.task == "niah_multikey_1" and old_action.startswith("span_"):
            return "recent_plus_full_old_raw"
        if case.task in {"niah_multiquery", "niah_multivalue"} and old_action.startswith("span_"):
            return "recent_plus_prefix_to_farthest_top3"
    if any(phrase in lower for phrase in ("all ", "all the", "every", "list all", "most common", "frequently appeared")):
        if old_action in {"span_top2_b0_a0", "span_b0_a0"}:
            return "recent_plus_span_top3_b0_a0"
    return action


def router_prediction_margin(prediction: Any) -> float:
    probabilities = getattr(prediction, "probabilities", None)
    if not probabilities:
        return 1.0
    values = sorted((float(value) for value in probabilities.values()), reverse=True)
    if len(values) < 2:
        return values[0] if values else 0.0
    return values[0] - values[1]


def kv_safe_router_override_v5(case: BenchCase, prediction: Any) -> str:
    action = getattr(prediction, "raw_action", prediction)
    confidence = float(getattr(prediction, "confidence", 1.0))
    margin = router_prediction_margin(prediction)
    exact = case.benchmark.startswith("ruler") or case.task in EXACT_LONG_BENCH_TASKS
    old_action = action.removeprefix("recent_plus_") if action.startswith("recent_plus_") else action
    lower = case.query.lower()
    length_match = re.search(r"ruler_(\d+)", case.benchmark)
    ruler_length = int(length_match.group(1)) if length_match else 0
    compressed = {
        "recent_only",
        "summary10",
        "summary100",
        "summary1000",
        "summary1_8",
        "summary1_4",
        "summary1_2",
        "static_hier",
    }
    if case.task in SUMMARY_TASKS:
        return action
    if not exact:
        return action
    if old_action in compressed:
        if case.benchmark.startswith("ruler"):
            return "recent_plus_span_top3_b0_a0"
        return "recent_plus_retrieval_raw_k2"
    if case.benchmark.startswith("ruler") and case.task in {"vt", "cwe", "fwe"}:
        if old_action in {"span_top2_b0_a0", "span_b0_a0", "prefix_to_evidence"}:
            return "recent_plus_span_top3_b0_a0"
    if case.benchmark.startswith("ruler") and ruler_length >= 16384:
        if case.task == "niah_multikey_1" and old_action.startswith("span_"):
            return "recent_plus_full_old_raw"
        uncertain = confidence < 0.55 or margin < 0.20
        if case.task in {"niah_multiquery", "niah_multivalue"} and old_action.startswith("span_") and uncertain:
            return "recent_plus_prefix_to_farthest_top3"
    if any(phrase in lower for phrase in ("all ", "all the", "every", "list all", "most common", "frequently appeared")):
        if old_action in {"span_top2_b0_a0", "span_b0_a0"}:
            return "recent_plus_span_top3_b0_a0"
    return action


def parse_recent_plus_block_action(action: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"recent_plus_b(\d+)_span_top(\d+)_b0_a0", action)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def enforce_block_floor(action: str, floor_action: str) -> str:
    parsed = parse_recent_plus_block_action(action)
    floor = parse_recent_plus_block_action(floor_action)
    if parsed is None or floor is None:
        return floor_action
    block, top_k = parsed
    floor_block, floor_top_k = floor
    if block < floor_block or top_k < floor_top_k:
        return floor_action
    return action


def blocksize_floor_v1_action(case: BenchCase, action: str) -> str:
    """Risk floor for the aggressive small-block router.

    The m3 oracle often selects 32/64-token blocks, but m10 partial runs show
    that unconstrained small-block routing is unsafe on multi-evidence RULER.
    This floor keeps the router useful for easy cases while enforcing the
    smallest empirically safe action for hard benchmark families.
    """
    if case.benchmark.startswith("ruler_"):
        length_match = re.search(r"ruler_(\d+)", case.benchmark)
        ruler_length = int(length_match.group(1)) if length_match else 0
        if ruler_length >= 16384:
            return enforce_block_floor(action, "recent_plus_b512_span_top3_b0_a0")
        return enforce_block_floor(action, "recent_plus_b256_span_top3_b0_a0")
    if case.task in SUMMARY_TASKS:
        return enforce_block_floor(action, "recent_plus_b128_span_top12_b0_a0")
    if case.task in EXACT_LONG_BENCH_TASKS:
        return enforce_block_floor(action, "recent_plus_b128_span_top6_b0_a0")
    return action


def calibrated_floor_for_case(case: BenchCase) -> str:
    if case.benchmark.startswith("ruler_"):
        return CALIBRATED_FLOOR_ACTIONS.get(case.benchmark, "")
    if case.benchmark == "longbench":
        return CALIBRATED_FLOOR_ACTIONS["longbench"]
    return ""


def blocksize_lattice_floor_action(case: BenchCase, action: str) -> str:
    """Legacy floor that treats larger block/topK as a monotone upgrade."""
    floor_action = calibrated_floor_for_case(case)
    if floor_action:
        return enforce_block_floor(action, floor_action)
    return action


def blocksize_calibrated_floor_action(case: BenchCase, action: str) -> str:
    """Exact calibrated risk floor selected from block-size/topK sweeps.

    Calibration is over complete actions, not just lower bounds: changing block
    size can change evidence ranking, so a larger block is not always safer.
    """
    floor_action = calibrated_floor_for_case(case)
    return floor_action or action


def blocksize_floor_v2_action(case: BenchCase, action: str) -> str:
    return blocksize_lattice_floor_action(case, action)


def resolve_action(method: str, tokenizer: Any, case: BenchCase, config: Config, router: Any | None) -> str:
    if method == "recent_plus_kv_safe_rule_v0":
        return kv_safe_rule_action(case, tokenizer, config)
    if method == "router":
        if router is None:
            return rule_router_action(case)
        features, task_family = router_features(tokenizer, case, config)
        return router.predict(features, task_family=task_family).raw_action
    if method == "router_safe":
        if router is None:
            action = rule_router_action(case)
        else:
            features, task_family = router_features(tokenizer, case, config)
            action = router.predict(features, task_family=task_family).raw_action
        return kv_safe_router_override(case, action)
    if method == "router_safe_v5":
        if router is None:
            return kv_safe_router_override_v5(case, rule_router_action(case))
        features, task_family = router_features(tokenizer, case, config)
        prediction = router.predict(features, task_family=task_family)
        return kv_safe_router_override_v5(case, prediction)
    if method == "router_blocksize":
        if router is None:
            return "recent_plus_span_top3_b0_a0"
        features, task_family = router_features(tokenizer, case, config)
        return router.predict(features, task_family=task_family).raw_action
    if method == "router_blocksize_floor_v1":
        if router is None:
            action = "recent_plus_span_top3_b0_a0"
        else:
            features, task_family = router_features(tokenizer, case, config)
            action = router.predict(features, task_family=task_family).raw_action
        return blocksize_floor_v1_action(case, action)
    if method == "router_blocksize_floor_v2":
        if router is None:
            action = "recent_plus_span_top3_b0_a0"
        else:
            features, task_family = router_features(tokenizer, case, config)
            action = router.predict(features, task_family=task_family).raw_action
        return blocksize_floor_v2_action(case, action)
    if method in {"router_blocksize_calibrated", "riskkv_block_calibrated"}:
        if router is None:
            action = "recent_plus_span_top3_b0_a0"
        else:
            features, task_family = router_features(tokenizer, case, config)
            action = router.predict(features, task_family=task_family).raw_action
        return blocksize_calibrated_floor_action(case, action)
    if method in {"blocksize_calibrated_floor_only", "riskkv_block_floor_only"}:
        return blocksize_calibrated_floor_action(case, "full_raw")
    if method == "router_conservative":
        if router is None:
            action = rule_router_action(case)
        else:
            features, task_family = router_features(tokenizer, case, config)
            action = router.predict(features, task_family=task_family).action
        if action == "summary10" and not case.benchmark.startswith("ruler") and case.task not in EXACT_LONG_BENCH_TASKS:
            action = "summary100"
        return safety_override_action(case, action)
    return method


def build_memory_for_action(action: str, tokenizer: Any, case: BenchCase, config: Config) -> str:
    context = case.context
    if action == "full_raw":
        return context
    if action.startswith("recent_plus_"):
        old_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        old_cut = max(0, len(old_ids) - config.recent_tokens)
        old_context = tokenizer.decode(old_ids[:old_cut], skip_special_tokens=True)
        recent = tokenizer.decode(old_ids[old_cut:], skip_special_tokens=True) if old_ids else ""
        old_case = BenchCase(
            benchmark=case.benchmark,
            task=case.task,
            case_id=case.case_id,
            context=old_context,
            query=case.query,
            answers=case.answers,
            length=case.length,
        )
        old_action = action.removeprefix("recent_plus_")
        action_config = config
        block_match = re.fullmatch(r"b(\d+)_(.+)", old_action)
        if block_match:
            action_config = replace(config, block_tokens=int(block_match.group(1)))
            old_action = block_match.group(2)
        if old_action == "full_old_raw":
            old_memory = old_context
        elif old_action == "static_hier":
            old_memory = static_hier_memory(tokenizer, old_context, action_config)
        elif summary_ratio(old_action) is not None:
            old_memory = build_ratio_summary_memory(tokenizer, old_context, action_config, old_action)
        elif old_action.startswith("retrieval_raw_k"):
            top_k = int(old_action.removeprefix("retrieval_raw_k"))
            old_memory = (
                "Old summary memory:\n"
                f"{static_hier_memory(tokenizer, old_context, action_config)}\n\n"
                "Retrieved old raw evidence:\n"
                f"{retrieve_blocks(tokenizer, old_context, case.query, action_config, top_k)}"
            )
        elif old_action == "prefix_to_evidence":
            old_memory = retrieve_prefix_to_evidence(tokenizer, old_context, case.query, action_config)
        elif old_action.startswith("prefix_to_farthest_top"):
            top_k = int(old_action.removeprefix("prefix_to_farthest_top"))
            old_memory = retrieve_prefix_to_evidence(tokenizer, old_context, case.query, action_config, top_k=top_k)
        elif old_action.startswith("span_b"):
            match = re.fullmatch(r"span_b(\d+)_a(\d+)", old_action)
            if not match:
                raise ValueError(action)
            before, after = int(match.group(1)), int(match.group(2))
            old_memory = retrieve_span_blocks(tokenizer, old_context, case.query, action_config, before=before, after=after)
        elif old_action.startswith("span_top"):
            match = re.fullmatch(r"span_top(\d+)_b(\d+)_a(\d+)", old_action)
            if not match:
                raise ValueError(action)
            max_spans, before, after = int(match.group(1)), int(match.group(2)), int(match.group(3))
            old_memory = retrieve_span_blocks(
                tokenizer,
                old_context,
                case.query,
                action_config,
                before=before,
                after=after,
                max_spans=max_spans,
            )
        else:
            raise ValueError(action)
        return (
            "Old memory:\n"
            f"{old_memory}\n\n"
            "Recent raw context:\n"
            f"{recent}"
        )
    if action == "recent_only":
        return recent_text(tokenizer, context, config.recent_tokens)
    if action == "summary10":
        return build_summary_memory(tokenizer, context, config, "summary10")
    if action == "summary100":
        return build_summary_memory(tokenizer, context, config, "summary100")
    if action == "summary1000":
        return build_summary_memory(tokenizer, context, config, "summary1000")
    if summary_ratio(action) is not None:
        return build_ratio_summary_memory(tokenizer, context, config, action)
    if action == "static_hier":
        return static_hier_memory(tokenizer, context, config)
    if action.startswith("retrieval_raw_k"):
        top_k = int(action.removeprefix("retrieval_raw_k"))
        return (
            "Summary memory:\n"
            f"{static_hier_memory(tokenizer, context, config)}\n\n"
            "Retrieved raw evidence:\n"
            f"{retrieve_blocks(tokenizer, context, case.query, config, top_k)}\n\n"
            "Recent raw:\n"
            f"{recent_text(tokenizer, context, config.recent_tokens)}"
        )
    raise ValueError(action)


def build_prompt(tokenizer: Any, case: BenchCase, memory: str, config: Config) -> str:
    if case.benchmark.startswith("ruler"):
        prompt = f"{memory}\n\n{case.query}"
    else:
        template = LONG_BENCH_PROMPTS.get(case.task, "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
        prompt = template.format(context=memory, input=case.query)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= config.max_input_tokens:
        return prompt
    # Keep the question and the tail of memory if a method still exceeds the limit.
    query_text = case.query
    keep = max(256, config.max_input_tokens - 512)
    trimmed_memory = tokenizer.decode(tokenizer(memory, add_special_tokens=False)["input_ids"][-keep:], skip_special_tokens=True)
    if case.benchmark.startswith("ruler"):
        return f"{trimmed_memory}\n\n{query_text}"
    template = LONG_BENCH_PROMPTS.get(case.task, "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
    return template.format(context=trimmed_memory, input=query_text)


def generate_prediction(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    input_device = next(param.device for param in model.parameters() if param.device.type != "meta")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(input_device)
    attention_mask = inputs.attention_mask.to(input_device) if "attention_mask" in inputs else torch.ones_like(input_ids)
    prompt_tokens = int(input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    new_ids = output_ids[0, input_ids.shape[1] :]
    prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return prediction, prompt_tokens, int(new_ids.shape[0]), seconds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[TrialRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[TrialRow]] = {}
    for row in rows:
        grouped.setdefault((row.benchmark, row.task, row.method), []).append(row)
        grouped.setdefault(("__overall__", "__overall__", row.method), []).append(row)
    full_tokens: dict[tuple[str, str], float] = {}
    full_seconds: dict[tuple[str, str], float] = {}
    for (bench, task, method), items in grouped.items():
        if method == "full_raw":
            full_tokens[(bench, task)] = statistics.mean(item.prompt_tokens for item in items)
            full_seconds[(bench, task)] = statistics.mean(item.seconds for item in items)
    out = []
    for (bench, task, method), items in sorted(grouped.items()):
        key = (bench, task)
        avg_tokens = statistics.mean(item.prompt_tokens for item in items)
        avg_seconds = statistics.mean(item.seconds for item in items)
        ft = full_tokens.get(key, avg_tokens)
        fs = full_seconds.get(key, avg_seconds)
        out.append(
            {
                "benchmark": bench,
                "task": task,
                "method": method,
                "samples": len(items),
                "exact_accuracy": statistics.mean(item.exact_correct for item in items),
                "avg_rouge_l": statistics.mean(item.rouge_l for item in items),
                "avg_score": statistics.mean(item.score for item in items),
                "avg_prompt_tokens": avg_tokens,
                "token_ratio_vs_full_raw": avg_tokens / ft if ft else 0.0,
                "avg_seconds": avg_seconds,
                "speedup_vs_full_raw": fs / avg_seconds if avg_seconds else 0.0,
            }
        )
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.cuda_visible_devices:
        # The launcher should set CUDA_VISIBLE_DEVICES before process start; this is only recorded in metadata.
        pass

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(config.dtype, torch),
    }
    if config.device_map not in {"", "none", "cuda"}:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    if config.device_map in {"", "none", "cuda"} and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    router = None
    if load_router is not None and config.router_path and Path(config.router_path).exists():
        router = load_router(config.router_path, conservative_generation_upgrade="summary100")

    cases = load_longbench_cases(config) + load_ruler_cases(config)
    rows: list[TrialRow] = []
    for case_idx, case in enumerate(cases):
        full_memory = build_memory_for_action("full_raw", tokenizer, case, config)
        full_prompt = build_prompt(tokenizer, case, full_memory, config)
        full_prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
        for method in config.methods:
            action = resolve_action(method, tokenizer, case, config, router)
            memory = build_memory_for_action(action, tokenizer, case, config)
            prompt = build_prompt(tokenizer, case, memory, config)
            max_new = config.max_new_tokens_summary if case.task in SUMMARY_TASKS else config.max_new_tokens_exact
            try:
                prediction, prompt_tokens, output_tokens, seconds = generate_prediction(model, tokenizer, prompt, max_new)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                prediction = f"ERROR: {type(exc).__name__}: {str(exc)[:200]}"
                prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                output_tokens = 0
                seconds = 0.0
            exact = exact_match_any(prediction, case.answers)
            rouge = rouge_l_f1(prediction, case.answers)
            score = rouge if case.task in SUMMARY_TASKS else float(exact)
            rows.append(
                TrialRow(
                    benchmark=case.benchmark,
                    task=case.task,
                    case_id=case.case_id,
                    method=method,
                    routed_action=action,
                    prompt_tokens=prompt_tokens,
                    token_ratio_vs_full_raw=prompt_tokens / full_prompt_tokens if full_prompt_tokens else 0.0,
                    output_tokens=output_tokens,
                    seconds=seconds,
                    prediction=prediction.replace("\n", " ")[:500],
                    answers=json.dumps(case.answers, ensure_ascii=False),
                    exact_correct=exact,
                    rouge_l=rouge,
                    score=score,
                )
            )
            write_csv(output_dir / "trials.partial.csv", [asdict(row) for row in rows])
        print(f"finished case {case_idx + 1}/{len(cases)} {case.benchmark}/{case.task}/{case.case_id}", flush=True)

    summary = summarize(rows)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    payload = {"config": asdict(config), "num_cases": len(cases), "summary": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("benchmark,task,method,samples,score,tokens_vs_full,speedup")
    for row in summary:
        print(
            f"{row['benchmark']},{row['task']},{row['method']},{row['samples']},"
            f"{row['avg_score']:.4f},{row['token_ratio_vs_full_raw']:.4f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
