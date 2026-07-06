from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_policy_router_runtime import TinyMemoryRouter  # noqa: E402
from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    BenchCase,
    SUMMARY_TASKS,
    build_memory_for_action,
    build_prompt,
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    parse_int_tuple,
    retrieve_blocks,
    router_features,
)
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES, write_csv  # noqa: E402


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    benchmark_output_dir: str
    candidate_methods: tuple[str, ...]
    cases_per_dataset: int
    prefill_tokens: int
    prefill_token_lengths: tuple[int, ...]
    sample_stride_tokens: int
    eval_start_tokens: int
    block_tokens: int
    recent_tokens: int
    max_text_tokens: int
    max_input_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    label_mode: str
    seed: int


@dataclass(frozen=True)
class SyntheticCase:
    benchmark: str
    task: str
    case_id: str
    dataset: str
    kind: str
    context: str
    query: str
    answers: tuple[str, ...]
    old_target_blocks: tuple[int, ...]
    recent_answers: tuple[str, ...]
    success_methods: tuple[str, ...] | None


@dataclass
class SyntheticTrial:
    benchmark: str
    task: str
    case_id: str
    dataset: str
    kind: str
    method: str
    success: int
    prompt_tokens: int
    token_ratio_vs_full_raw: float
    label_candidate: int


@dataclass
class RouterExample:
    benchmark: str
    task: str
    case_id: str
    dataset: str
    kind: str
    task_family: str
    label: str
    oracle_token_ratio: float
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    benchmark: str
    task: str
    case_id: str
    dataset: str
    kind: str
    task_family: str
    oracle_label: str
    predicted_label: str
    label_correct: int
    synthetic_success: int
    token_ratio_vs_full_raw: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Distill a router from non-benchmark synthetic exact/retrieval data.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--benchmark_output_dir", default="")
    parser.add_argument(
        "--candidate_methods",
        default="full_raw,recent_only,static_hier,summary1_8,summary1_4,summary1_2,retrieval_raw_k1,retrieval_raw_k2,retrieval_raw_k3,retrieval_raw_k4,retrieval_raw_k8",
    )
    parser.add_argument("--cases_per_dataset", type=int, default=240)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument(
        "--prefill_token_lengths",
        default="",
        help="Optional comma-separated prefix lengths. When set, cases_per_dataset is generated for each length.",
    )
    parser.add_argument("--sample_stride_tokens", type=int, default=768)
    parser.add_argument("--eval_start_tokens", type=int, default=20000)
    parser.add_argument("--block_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_text_tokens", type=int, default=260000)
    parser.add_argument("--max_input_tokens", type=int, default=24000)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--label_mode", choices=["cheapest", "length_aware"], default="length_aware")
    parser.add_argument("--seed", type=int, default=2026070501)
    args = parser.parse_args()
    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    prefill_token_lengths = parse_int_tuple(args.prefill_token_lengths) if args.prefill_token_lengths else (args.prefill_tokens,)
    return Config(
        **{
            **vars(args),
            "text_paths": text_paths,
            "dataset_names": dataset_names,
            "candidate_methods": parse_csv_tuple(args.candidate_methods),
            "prefill_token_lengths": prefill_token_lengths,
        }
    )


def bench_config(config: Config, output_dir: str = "") -> BenchConfig:
    return BenchConfig(
        output_dir=output_dir or config.output_dir,
        model_name_or_path=config.model_name_or_path,
        adapter_path="",
        longbench_data_dir="",
        ruler_data_dir="",
        longbench_tasks=(),
        ruler_tasks=(),
        ruler_context_lengths=(),
        methods=config.candidate_methods,
        max_examples_per_task=0,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        max_input_tokens=config.max_input_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_new_tokens_exact=48,
        max_new_tokens_summary=160,
        dtype="float16",
        attn_implementation="sdpa",
        device_map="auto",
        cuda_visible_devices="",
        router_path="",
        seed=config.seed,
    )


def load_text_ids(tokenizer: Any, config: Config) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for dataset, path in zip(config.dataset_names, config.text_paths):
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        out[dataset] = tokenizer(text, add_special_tokens=False)["input_ids"][: config.max_text_tokens]
    return out


def make_record(dataset: str, case_idx: int, slot: int, rng: random.Random) -> tuple[str, str, str]:
    key = f"{dataset.upper()}-MEM-{case_idx:04d}-{slot}-{rng.randrange(1000, 9999)}"
    answer = f"{rng.choice(('lumen', 'atlas', 'cedar', 'violet', 'ember', 'quartz'))}-{rng.randrange(100, 999)}-{rng.choice(('cipher', 'anchor', 'relay', 'ledger', 'vector'))}"
    record = (
        "\n\nSPECIAL MEMORY RECORD.\n"
        f"Private access key: {key}\n"
        f"Private access code: {answer}\n"
        "The code must be recalled exactly.\n\n"
    )
    return key, answer, record


def pick_old_blocks(num_blocks: int, count: int, rng: random.Random) -> list[int]:
    usable = max(1, num_blocks - 1)
    if count >= usable:
        return list(range(usable))
    return sorted(rng.sample(range(usable), count))


def case_plan(kind: str) -> tuple[str, str, int, int, tuple[str, ...] | None]:
    if kind == "magic_single_old":
        return "ruler_synthetic", "niah_single_1", 1, 0, None
    if kind == "magic_single_recent":
        return "ruler_synthetic", "niah_single_2", 0, 1, None
    if kind == "magic_multiquery":
        return "ruler_synthetic", "niah_multiquery", 3, 0, None
    if kind == "magic_multivalue":
        return "ruler_synthetic", "niah_multivalue", 4, 0, None
    if kind == "vt_k2":
        return "ruler_synthetic", "vt", 3, 0, None
    if kind == "cwe_k1":
        return "ruler_synthetic", "cwe", 1, 0, None
    if kind == "fwe_k1":
        return "ruler_synthetic", "fwe", 1, 0, None
    if kind == "single_old":
        return "ruler_synthetic", "niah_single_1", 1, 0, None
    if kind == "single_recent":
        return "ruler_synthetic", "niah_single_2", 0, 1, None
    if kind == "two_old":
        return "ruler_synthetic", "niah_multikey_1", 2, 0, None
    if kind == "four_old":
        return "ruler_synthetic", "niah_multiquery", 4, 0, None
    if kind == "old_recent":
        return "ruler_synthetic", "niah_multivalue", 1, 1, None
    if kind == "three_old":
        return "ruler_synthetic", "niah_multiquery", 3, 0, None
    if kind == "natural_single_old":
        return "synthetic_longbench", "hotpotqa", 1, 0, None
    if kind == "natural_two_old":
        return "synthetic_longbench", "2wikimqa", 2, 0, None
    if kind == "natural_three_old":
        return "synthetic_longbench", "musique", 3, 0, None
    if kind == "summary_brief":
        return "synthetic_generation", "gov_report", 0, 0, ("summary1_8", "summary1_4", "summary1_2", "full_raw")
    if kind == "summary_detailed":
        return "synthetic_generation", "multi_news", 0, 0, ("summary1_4", "summary1_2", "full_raw")
    if kind == "recent_generation":
        return "synthetic_generation", "multi_news", 0, 0, ("recent_only", "full_raw")
    if kind == "full_context":
        return "synthetic_generation", "gov_report", 0, 0, ("full_raw",)
    raise ValueError(kind)


def build_synthetic_cases(tokenizer: Any, token_ids: dict[str, list[int]], config: Config) -> list[SyntheticCase]:
    rng = random.Random(config.seed)
    kinds = (
        "magic_single_old",
        "magic_single_old",
        "magic_single_recent",
        "magic_multiquery",
        "magic_multiquery",
        "magic_multivalue",
        "vt_k2",
        "cwe_k1",
        "fwe_k1",
        "summary_brief",
        "recent_generation",
        "magic_single_old",
        "magic_multiquery",
        "summary_brief",
        "magic_multivalue",
        "three_old",
        "summary_detailed",
        "recent_generation",
        "full_context",
        "single_old",
        "natural_single_old",
        "two_old",
        "natural_two_old",
        "natural_three_old",
        "four_old",
    )
    cases: list[SyntheticCase] = []
    global_idx = 0
    for dataset, ids in token_ids.items():
        for prefix_len in config.prefill_token_lengths:
            max_start = len(ids) - prefix_len
            if max_start <= 0:
                continue
            for local_idx in range(config.cases_per_dataset):
                kind = kinds[local_idx % len(kinds)]
                benchmark, task, old_count, recent_count, success_methods = case_plan(kind)
                start = min(max_start, config.eval_start_tokens + local_idx * config.sample_stride_tokens)
                raw_ids = ids[start : start + prefix_len]
                raw_blocks = [
                    raw_ids[idx : idx + config.block_tokens]
                    for idx in range(0, len(raw_ids), config.block_tokens)
                ]
                if len(raw_blocks) < 3:
                    continue
                block_texts = [tokenizer.decode(block, skip_special_tokens=True) for block in raw_blocks]
                old_blocks = pick_old_blocks(len(block_texts), old_count, rng)
                answers: list[str] = []
                keys: list[str] = []
                for slot, block_id in enumerate(old_blocks):
                    key, answer, record = make_record(dataset, global_idx, slot, rng)
                    if kind.startswith("magic_"):
                        key = f"{rng.choice(('wandering', 'abject', 'fair', 'used', 'squealing', 'annoying', 'depressed'))}-{rng.choice(('age', 'antler', 'sprout', 'commotion', 'dibble', 'decimal', 'tweet'))}-{global_idx}-{slot}"
                        if kind == "magic_multivalue" and keys:
                            key = keys[0]
                        answer = str(rng.randrange(1000000, 9999999))
                        record = f"\n\nThe special magic number for {key} mentioned in the provided text is {answer}.\n\n"
                    elif kind == "vt_k2":
                        key = f"value-{global_idx}-{slot}"
                        answer = rng.choice(("FITJT", "VGCAO", "ZJQUQ", "TYFAD", "DROFS", "QPLMN"))
                        record = f"\n\nVariable assignment chain: {answer} is assigned the value 15311 in this text.\n\n"
                    elif kind in {"cwe_k1", "fwe_k1"}:
                        key = f"freq-{global_idx}-{slot}"
                        words = [rng.choice(("arthur", "kilt", "activity", "fire", "appliance", "forest", "meter", "behalf", "authenticity", "fkmgoo", "quqtyf", "rsrvqx")) for _ in range(6)]
                        answer = words[0]
                        record = "\n\nFrequency evidence: " + " ".join(words * 8) + "\n\n"
                    block_texts[block_id] = record + block_texts[block_id]
                    keys.append(key)
                    answers.append(answer)
                recent_answers: list[str] = []
                recent_keys: list[str] = []
                recent_records: list[str] = []
                for slot in range(recent_count):
                    key, answer, record = make_record(dataset, global_idx, old_count + slot, rng)
                    if kind == "magic_single_recent":
                        key = f"{rng.choice(('wandering', 'abject', 'fair'))}-{rng.choice(('age', 'antler', 'sprout'))}-{global_idx}-{slot}"
                        answer = str(rng.randrange(1000000, 9999999))
                        record = f"\n\nThe special magic number for {key} mentioned in the provided text is {answer}.\n\n"
                    recent_keys.append(key)
                    recent_answers.append(answer)
                    answers.append(answer)
                    keys.append(key)
                    recent_records.append(record)
                context = "\n".join(block_texts + recent_records)
                if kind.startswith("summary"):
                    query = "Write a concise summary of the important information in the document."
                    if kind == "summary_detailed":
                        query = "Write a detailed summary that preserves major entities, events, and relationships."
                elif kind == "recent_generation":
                    query = "Continue from the most recent passage using the local context."
                elif kind == "full_context":
                    query = "Compare the main themes across the whole document and mention evidence from the beginning and the end."
                elif kind in {"magic_single_old", "magic_single_recent"}:
                    query = f"What is the special magic number for {keys[0]} mentioned in the provided text? The special magic number for {keys[0]} mentioned in the provided text is"
                elif kind == "magic_multiquery":
                    joined = ", ".join(keys)
                    query = f"What are all the special magic numbers for {joined} mentioned in the provided text? The special magic numbers for {joined} mentioned in the provided text are"
                elif kind == "magic_multivalue":
                    query = f"What are all the special magic numbers for {keys[0]} mentioned in the provided text? The special magic numbers for {keys[0]} mentioned in the provided text are"
                elif kind == "natural_single_old":
                    query = f"According to the document, which private access code is associated with {keys[0]}?"
                elif kind in {"natural_two_old", "natural_three_old"}:
                    joined = ", ".join(keys)
                    query = f"Using the document, identify the private access codes associated with these records: {joined}."
                elif kind == "vt_k2":
                    query = "Find all variables that are assigned the value 15311 in the text above. Answer: According to the chain(s) of variable assignment in the text above, variables assigned the value 15311 are:"
                elif kind == "cwe_k1":
                    query = "What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:"
                elif kind == "fwe_k1":
                    query = "What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:"
                elif len(keys) == 1:
                    query = f"What is the private access code for {keys[0]}? Answer with only the code."
                else:
                    joined = ", ".join(keys)
                    query = f"List all private access codes for these keys: {joined}. Answer with only the codes."
                cases.append(
                    SyntheticCase(
                        benchmark=benchmark,
                        task=task,
                        case_id=f"{dataset}-{prefix_len}-{global_idx:06d}-{kind}",
                        dataset=dataset,
                        kind=kind,
                        context=context,
                        query=query,
                        answers=tuple(answers),
                        old_target_blocks=tuple(old_blocks),
                        recent_answers=tuple(recent_answers),
                        success_methods=success_methods,
                    )
                )
                global_idx += 1
    return cases


def as_bench_case(case: SyntheticCase) -> BenchCase:
    return BenchCase(
        benchmark=case.benchmark,
        task=case.task,
        case_id=case.case_id,
        context=case.context,
        query=case.query,
        answers=case.answers,
        length=len(case.context),
    )


def retrieved_contains_answers(action: str, tokenizer: Any, case: SyntheticCase, config: Config) -> bool:
    if not action.startswith("retrieval_raw_k"):
        return False
    top_k = int(action.removeprefix("retrieval_raw_k"))
    cfg = bench_config(config)
    raw = retrieve_blocks(tokenizer, case.context, case.query, cfg, top_k)
    raw_and_recent = raw + "\n" + build_memory_for_action("recent_only", tokenizer, as_bench_case(case), cfg)
    return all(answer in raw_and_recent for answer in case.answers)


def action_success(action: str, tokenizer: Any, case: SyntheticCase, config: Config) -> bool:
    if action.startswith("recent_plus_"):
        old_action = action.removeprefix("recent_plus_")
        if case.kind == "recent_generation":
            return True
        if case.recent_answers and all(answer in case.recent_answers for answer in case.answers):
            return True
        if old_action == "full_old_raw":
            return True
        return action_success(old_action, tokenizer, case, config)
    if case.success_methods is not None:
        if action.startswith("retrieval_raw_k"):
            action_k = int(action.removeprefix("retrieval_raw_k"))
            for method in case.success_methods:
                if method.startswith("retrieval_raw_k") and action_k >= int(method.removeprefix("retrieval_raw_k")):
                    return True
        return action in case.success_methods
    if action == "full_raw":
        return True
    if action == "recent_only":
        return bool(case.recent_answers) and all(answer in case.recent_answers for answer in case.answers)
    if action.startswith("retrieval_raw_k"):
        return retrieved_contains_answers(action, tokenizer, case, config)
    return False


def target_budget_ratio(full_prompt_tokens: int) -> float:
    if full_prompt_tokens <= 5_000:
        return 0.80
    if full_prompt_tokens <= 9_000:
        return 0.50
    if full_prompt_tokens <= 18_000:
        return 0.30
    return 0.22


def robustness_rank(method: str) -> tuple[int, int]:
    if method.startswith("recent_plus_"):
        method = method.removeprefix("recent_plus_")
        if method == "full_old_raw":
            method = "full_raw"
    if method == "full_raw":
        return (4, 99)
    if method.startswith("retrieval_raw_k"):
        return (3, int(method.removeprefix("retrieval_raw_k")))
    if method in {"summary1_2", "summary1_4", "summary1_8", "static_hier"}:
        return (2, {"summary1_2": 3, "summary1_4": 2, "static_hier": 1, "summary1_8": 0}.get(method, 0))
    if method == "recent_only":
        return (1, 0)
    return (0, 0)


def choose_synthetic_label(case: SyntheticCase, case_trials: list[SyntheticTrial], full_prompt_tokens: int, config: Config) -> SyntheticTrial:
    successful = [row for row in case_trials if row.success]
    if not successful:
        successful = [row for row in case_trials if row.method == "full_raw"]
    if config.label_mode == "cheapest" or case.task in SUMMARY_TASKS or case.kind in {"summary_brief", "summary_detailed", "recent_generation"}:
        return min(successful, key=lambda row: (row.token_ratio_vs_full_raw, row.prompt_tokens, row.method))

    budget = target_budget_ratio(full_prompt_tokens)
    within_budget = [row for row in successful if row.token_ratio_vs_full_raw <= budget + 1e-12]
    if within_budget:
        # Exact tasks need robust evidence coverage. For short contexts, cost is less important,
        # so choose the most robust action still inside the length-aware budget.
        return max(
            within_budget,
            key=lambda row: (robustness_rank(row.method), -row.token_ratio_vs_full_raw, -row.prompt_tokens),
        )
    # If the target budget is too tight, fall back to the cheapest successful action.
    return min(successful, key=lambda row: (row.token_ratio_vs_full_raw, row.prompt_tokens, row.method))


def build_trials(tokenizer: Any, cases: list[SyntheticCase], config: Config) -> tuple[list[SyntheticTrial], list[RouterExample]]:
    cfg = bench_config(config)
    trials: list[SyntheticTrial] = []
    examples: list[RouterExample] = []
    for case in cases:
        bench_case = as_bench_case(case)
        full_prompt = build_prompt(tokenizer, bench_case, build_memory_for_action("full_raw", tokenizer, bench_case, cfg), cfg)
        full_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
        case_trials: list[SyntheticTrial] = []
        for method in config.candidate_methods:
            memory = build_memory_for_action(method, tokenizer, bench_case, cfg)
            prompt = build_prompt(tokenizer, bench_case, memory, cfg)
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            success = int(action_success(method, tokenizer, case, config))
            row = SyntheticTrial(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                dataset=case.dataset,
                kind=case.kind,
                method=method,
                success=success,
                prompt_tokens=prompt_tokens,
                token_ratio_vs_full_raw=prompt_tokens / full_tokens if full_tokens else 0.0,
                label_candidate=0,
            )
            case_trials.append(row)
        chosen = choose_synthetic_label(case, case_trials, full_tokens, config)
        for row in case_trials:
            row.label_candidate = int(row.method == chosen.method)
        features, task_family = router_features(tokenizer, bench_case, cfg)
        examples.append(
            RouterExample(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                dataset=case.dataset,
                kind=case.kind,
                task_family=task_family,
                label=chosen.method,
                oracle_token_ratio=chosen.token_ratio_vs_full_raw,
                features=features,
            )
        )
        trials.extend(case_trials)
    return trials, examples


def split_indices(examples: list[RouterExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_key: dict[tuple[str, str], list[int]] = {}
    for idx, example in enumerate(examples):
        by_key.setdefault((example.kind, example.label), []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in by_key.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        test_count = max(1, min(len(indices) - 1, round(len(indices) * config.test_fraction)))
        test.extend(indices[:test_count])
        train.extend(indices[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def normalize(examples: list[RouterExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        mean.append(float(m))
        std.append(float(sd))
    return mean, std


def tensorize(
    examples: list[RouterExample],
    indices: list[int],
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for idx in indices:
        example = examples[idx]
        xs.append([(val - mean[col]) / max(std[col], 1e-6) for col, val in enumerate(example.features)])
        ys.append(label_to_id[example.label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_router(
    examples: list[RouterExample],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    label_names = sorted({example.label for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    mean, std = normalize(examples, train_indices)
    train_x, train_y = tensorize(examples, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize(examples, test_indices, label_to_id, mean, std) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = TinyMemoryRouter(train_x.shape[1], config.hidden_dim, len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights.mean().clamp_min(1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_acc = float((model(train_x).argmax(-1) == train_y).float().mean())
                test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_label_accuracy": train_acc,
                    "test_label_accuracy": test_acc,
                }
            )
    return model, label_to_id, mean, std, history


def evaluate_synthetic_split(
    model: TinyMemoryRouter,
    examples: list[RouterExample],
    trials: list[SyntheticTrial],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    trial_lookup = {(row.case_id, row.method): row for row in trials}
    x, _ = tensorize(examples, indices, label_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        pred_ids = model(x).argmax(-1).tolist()
    rows: list[PredictionRow] = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        pred = id_to_label[int(pred_ids[local_idx])]
        trial = trial_lookup[(example.case_id, pred)]
        rows.append(
            PredictionRow(
                split=split,
                benchmark=example.benchmark,
                task=example.task,
                case_id=example.case_id,
                dataset=example.dataset,
                kind=example.kind,
                task_family=example.task_family,
                oracle_label=example.label,
                predicted_label=pred,
                label_correct=int(pred == example.label),
                synthetic_success=trial.success,
                token_ratio_vs_full_raw=trial.token_ratio_vs_full_raw,
            )
        )
    return rows


def summarize_predictions(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PredictionRow]] = {}
    for row in rows:
        groups.setdefault((row.split, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.task_family), []).append(row)
        groups.setdefault((row.split, row.kind), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "group": group,
            "samples": len(items),
            "label_accuracy": sum(row.label_correct for row in items) / len(items),
            "synthetic_success": sum(row.synthetic_success for row in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
        }
        counts: dict[str, int] = {}
        for row in items:
            counts[row.predicted_label] = counts.get(row.predicted_label, 0) + 1
        for label, count in sorted(counts.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        out.append(payload)
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_case_metadata(case: SyntheticCase) -> dict[str, Any]:
    return {
        "benchmark": case.benchmark,
        "task": case.task,
        "case_id": case.case_id,
        "dataset": case.dataset,
        "kind": case.kind,
        "context_chars": len(case.context),
        "query": case.query,
        "answers": "|".join(case.answers),
        "old_target_blocks": "|".join(str(item) for item in case.old_target_blocks),
        "recent_answers": "|".join(case.recent_answers),
        "success_methods": "|".join(case.success_methods or ()),
    }


def bench_config_from_summary(path: Path) -> BenchConfig:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    raw = payload["config"]
    return BenchConfig(
        output_dir=raw["output_dir"],
        model_name_or_path=raw["model_name_or_path"],
        adapter_path=raw.get("adapter_path", ""),
        longbench_data_dir=raw["longbench_data_dir"],
        ruler_data_dir=raw["ruler_data_dir"],
        longbench_tasks=tuple(raw["longbench_tasks"]),
        ruler_tasks=tuple(raw["ruler_tasks"]),
        ruler_context_lengths=tuple(int(item) for item in raw["ruler_context_lengths"]),
        methods=tuple(raw["methods"]),
        max_examples_per_task=int(raw["max_examples_per_task"]),
        block_tokens=int(raw["block_tokens"]),
        recent_tokens=int(raw["recent_tokens"]),
        max_input_tokens=int(raw["max_input_tokens"]),
        summary10_words=int(raw["summary10_words"]),
        summary100_words=int(raw["summary100_words"]),
        summary1000_words=int(raw["summary1000_words"]),
        max_new_tokens_exact=int(raw["max_new_tokens_exact"]),
        max_new_tokens_summary=int(raw["max_new_tokens_summary"]),
        dtype=raw["dtype"],
        attn_implementation=raw["attn_implementation"],
        device_map=raw["device_map"],
        cuda_visible_devices=raw.get("cuda_visible_devices", ""),
        router_path=raw.get("router_path", ""),
        seed=int(raw["seed"]),
    )


def evaluate_heldout_benchmark(
    model: TinyMemoryRouter,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    tokenizer: Any,
    benchmark_output_dir: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not benchmark_output_dir:
        return [], []
    bench_dir = Path(benchmark_output_dir)
    if not (bench_dir / "trials.csv").exists():
        return [], []
    bench_cfg = bench_config_from_summary(bench_dir)
    trials = read_csv(bench_dir / "trials.csv")
    trial_lookup = {(row["benchmark"], row["task"], row["case_id"], row["method"]): row for row in trials}
    cases = load_longbench_cases(bench_cfg) + load_ruler_cases(bench_cfg)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    rows: list[dict[str, Any]] = []
    for case in cases:
        features, task_family = router_features(tokenizer, case, bench_cfg)
        x = torch.tensor(
            [[(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]],
            dtype=torch.float32,
        )
        model.eval()
        with torch.inference_mode():
            pred = id_to_label[int(model(x).argmax(-1).item())]
        trial = trial_lookup.get((case.benchmark, case.task, case.case_id, pred))
        full = trial_lookup.get((case.benchmark, case.task, case.case_id, "full_raw"))
        if trial is None or full is None:
            continue
        score = float(trial["score"])
        full_score = float(full["score"])
        rows.append(
            {
                "benchmark": case.benchmark,
                "task": case.task,
                "case_id": case.case_id,
                "task_family": task_family,
                "predicted_label": pred,
                "score": score,
                "full_score": full_score,
                "relative_to_full": score / full_score if abs(full_score) > 1e-12 else "",
                "exact_correct": int(trial["exact_correct"]),
                "token_ratio_vs_full_raw": float(trial["token_ratio_vs_full_raw"]),
                "seconds": float(trial["seconds"]),
            }
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault("__overall__", []).append(row)
        groups.setdefault(row["task_family"], []).append(row)
        groups.setdefault(row["benchmark"], []).append(row)
    summary: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "group": group,
            "samples": len(items),
            "avg_score": sum(float(row["score"]) for row in items) / len(items),
            "avg_full_score": sum(float(row["full_score"]) for row in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(float(row["token_ratio_vs_full_raw"]) for row in items) / len(items),
            "avg_seconds": sum(float(row["seconds"]) for row in items) / len(items),
        }
        payload["relative_to_full"] = payload["avg_score"] / payload["avg_full_score"] if payload["avg_full_score"] else ""
        counts: dict[str, int] = {}
        for row in items:
            counts[row["predicted_label"]] = counts.get(row["predicted_label"], 0) + 1
        for label, count in sorted(counts.items()):
            payload[f"select_{label}"] = count
            payload[f"select_{label}_rate"] = count / len(items)
        summary.append(payload)
    return rows, summary


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids = load_text_ids(tokenizer, config)
    cases = build_synthetic_cases(tokenizer, token_ids, config)
    trials, examples = build_trials(tokenizer, cases, config)
    train_indices, test_indices = split_indices(examples, config)
    model, label_to_id, mean, std, history = train_router(examples, train_indices, test_indices, config)
    pred_rows: list[PredictionRow] = []
    pred_rows.extend(evaluate_synthetic_split(model, examples, trials, train_indices, "train", label_to_id, mean, std))
    pred_rows.extend(evaluate_synthetic_split(model, examples, trials, test_indices, "test", label_to_id, mean, std))
    prediction_summary = summarize_predictions(pred_rows)
    heldout_rows, heldout_summary = evaluate_heldout_benchmark(
        model,
        label_to_id,
        mean,
        std,
        tokenizer,
        config.benchmark_output_dir,
    )
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    label_names = [id_to_label[idx] for idx in range(len(id_to_label))]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "label_names": label_names,
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "router.pt",
    )
    write_csv(output_dir / "synthetic_cases.csv", [synthetic_case_metadata(row) for row in cases])
    write_csv(output_dir / "synthetic_trials.csv", [asdict(row) for row in trials])
    write_csv(output_dir / "examples.csv", [asdict(row) for row in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in pred_rows])
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "train_history.csv", history)
    write_csv(output_dir / "heldout_benchmark_predictions.csv", heldout_rows)
    write_csv(output_dir / "heldout_benchmark_summary.csv", heldout_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "label_names": label_names,
                "synthetic_cases": len(cases),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "history_tail": history[-5:],
                "prediction_summary": prediction_summary,
                "heldout_benchmark_summary": heldout_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("SYNTHETIC")
    print("split,group,samples,label_accuracy,synthetic_success,avg_token_ratio_vs_full_raw")
    for row in prediction_summary:
        print(
            f"{row['split']},{row['group']},{row['samples']},"
            f"{row['label_accuracy']:.4f},{row['synthetic_success']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    if heldout_summary:
        print("HELDOUT_BENCHMARK_OFFLINE")
        print("group,samples,avg_score,avg_full_score,relative_to_full,avg_token_ratio_vs_full_raw")
        for row in heldout_summary:
            print(
                f"{row['group']},{row['samples']},{row['avg_score']:.4f},"
                f"{row['avg_full_score']:.4f},{row['relative_to_full']:.4f},"
                f"{row['avg_token_ratio_vs_full_raw']:.4f}"
            )
    print(f"saved router to {output_dir / 'router.pt'}")


if __name__ == "__main__":
    main()
