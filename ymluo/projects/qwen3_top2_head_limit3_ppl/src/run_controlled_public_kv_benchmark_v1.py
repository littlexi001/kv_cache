from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import string
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
    "triviaqa": {
        "prefix": (
            "Answer the question based on the given passage. Only give me the answer and do not output any other words. "
            "The following are some examples.\n\n"
        ),
        "suffix": "\n\n{input}",
        "max_new_tokens": 32,
        "metric": "qa_f1",
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
    anchor_pages_per_key: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str
    prompt_wrapper: str
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
    parser.add_argument("--obs_window_tokens", type=int, default=64)
    parser.add_argument("--snap_pool_kernel", type=int, default=7)
    parser.add_argument(
        "--ours_scorer",
        choices=["lexical", "semantic", "late_interaction", "hybrid", "hybrid_mmr", "hybrid_late_mmr"],
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
        "--anchor_pages_per_key",
        type=int,
        default=2,
        help="For typed-anchor queries, reserve up to this many exact-anchor pages per query key before MMR fill.",
    )
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


def score_prediction(metric: str, prediction: str, answers: list[str]) -> float:
    if metric == "ruler_string_match":
        return ruler_string_match(prediction, answers)
    if metric == "ruler_string_match_part":
        return ruler_string_match_part(prediction, answers)
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
    if config.prompt_wrapper == "llama3":
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


def extract_query_anchors(query: str) -> list[str]:
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
    for match in re.finditer(r"\b[A-Z]{3,}\b", query):
        anchors.append(match.group(0).lower())
    deduped: list[str] = []
    for anchor in anchors:
        if anchor not in deduped:
            deduped.append(anchor)
    return deduped[:12]


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
    entity_raw: list[float] = []
    structural_raw: list[float] = []
    coverage_raw: list[float] = []
    for page in pages:
        p_words = word_counter(page.text)
        p_entities = extract_entities(page.text)
        lexical_raw.append(float(sum(min(count, p_words.get(word, 0)) for word, count in q_words.items())))
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
    entity = normalize_values(entity_raw)
    structural = normalize_values(structural_raw)
    coverage = normalize_values(coverage_raw)
    semantic_vectors: dict[int, torch.Tensor] = {}
    semantic = [0.0 for _ in pages]
    late = [0.0 for _ in pages]
    needs_mean_semantic = config.ours_scorer in {"semantic", "hybrid", "hybrid_mmr", "hybrid_late_mmr"}
    needs_late_interaction = config.ours_scorer in {"late_interaction", "hybrid_late_mmr"}
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
            semantic_component = late[idx] if config.ours_scorer == "hybrid_late_mmr" else semantic[idx]
            score = (
                config.semantic_weight * semantic_component
                + config.lexical_weight * lexical[idx]
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
    extra = extra or {}
    semantic_vectors = score_pages(
        example,
        pages,
        config,
        model=extra.get("model"),
        tokenizer=extra.get("tokenizer"),
    )
    keep = base_context_keep_indices(bundle, config)
    selected_context_tokens = sum(1 for idx in keep if bundle.context_token_start <= idx < bundle.query_start)
    remaining = max(0, config.budget_tokens - selected_context_tokens)
    selected_pages: list[Page] = []

    anchors = extract_query_anchors(example.query or example.suffix_template)
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
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
    selected_page_ids = {page.page_id for page in selected_pages}
    candidate_pages = [page for page in pages if page.page_id not in selected_page_ids]
    while candidate_pages and remaining > 0:
        if config.ours_scorer in {"hybrid_mmr", "hybrid_late_mmr"} and semantic_vectors and selected_pages:
            selected_vecs = [semantic_vectors[page.page_id] for page in selected_pages if page.page_id in semantic_vectors]
            reranked = []
            for page in candidate_pages:
                redundancy = 0.0
                if page.page_id in semantic_vectors and selected_vecs:
                    page_vec = semantic_vectors[page.page_id]
                    redundancy = max(float(torch.dot(page_vec, vec).item()) for vec in selected_vecs)
                mmr_score = config.ours_mmr_lambda * page.score - (1.0 - config.ours_mmr_lambda) * redundancy
                reranked.append((mmr_score, page))
            reranked.sort(key=lambda item: (item[0], item[1].score, -item[1].page_id), reverse=True)
            page = reranked[0][1]
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
        candidate_pages = [candidate for candidate in candidate_pages if candidate.page_id != page.page_id]
    return fit_context_budget(keep, bundle, config.budget_tokens)


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
def prefill_prefix(model: torch.nn.Module, bundle: PromptBundle, input_device: torch.device) -> tuple[Any, float]:
    ids = bundle.input_ids[:, : bundle.query_start].to(input_device)
    started = time.perf_counter()
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
    if method in {"h2o_observe", "snapkv_observe"}:
        if attention_scores is None:
            raise ValueError(f"{method} requires attention_scores")
        keep_indices = selector(bundle, example, pages, config, attention_scores)
    elif method == "ours_page_gather":
        keep_indices = selector(bundle, example, pages, config, {"model": model, "tokenizer": tokenizer})
    else:
        keep_indices = selector(bundle, example, pages, config, None)
    gather_started = time.perf_counter()
    sparse_cache = full_prefix_cache if method == "full_kv" else gather_past_key_values(full_prefix_cache, keep_indices)
    gather_seconds = 0.0 if method == "full_kv" else time.perf_counter() - gather_started
    prediction, generated_ids, query_seconds, decode_seconds = generate_with_cache(
        model,
        tokenizer,
        bundle,
        sparse_cache,
        example.max_new_tokens,
        input_device,
    )
    score = score_prediction(example.metric, prediction, example.answers)
    context_kept = sum(1 for idx in keep_indices if bundle.context_token_start <= idx < bundle.query_start)
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
        "ours_scorer": config.ours_scorer if method == "ours_page_gather" else "",
        "selected_pages": ",".join(str(page_id) for page_id in selected_page_ids(bundle, keep_indices)),
        "page_count": len(pages),
        "context_length_field": example.length,
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

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    needs_attention = any(method in {"h2o_observe", "snapkv_observe"} for method in methods)
    for idx, example in enumerate(examples):
        bundle, pages, _, _, _ = build_bundle(tokenizer, example, config)
        if (idx + 1) % config.log_every == 0 or idx == 0:
            print(
                f"[{idx + 1}/{len(examples)}] {example.benchmark}/{example.task}/{example.sample_id} "
                f"prefix_tokens={bundle.query_start} pages={len(pages)}",
                flush=True,
            )
        full_prefix_cache, prefill_seconds = prefill_prefix(model, bundle, input_device)
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
                clone_past_key_values(full_prefix_cache),
                prefill_seconds,
                method,
                config,
                attention_scores,
            )
            rows.append(row)
            print(
                f"  {method}: score={row['score']:.3f} kept={row['kept_prefix_tokens']}/{row['raw_prefix_tokens']} "
                f"online={row['online_seconds']:.3f}s pred={row['prediction'][:80]}",
                flush=True,
            )
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
