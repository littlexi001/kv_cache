from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    LONG_BENCH_PROMPTS,
    BenchCase,
    load_longbench_cases,
    load_ruler_cases,
)


MODE_TO_METHOD = {
    "absolute": "naive_kv_gather_absolute_query_pos",
    "compact": "rope_delta_repack_compact_query_pos",
    "shifted": "rope_delta_repack_shifted_query_pos",
    "full": "full_kv_cache",
    "prompt": "prompt_rebuild_selected_pages",
}

SPARSE_MODES = ("absolute", "compact", "shifted")
DEPLOY_MODES = ("absolute", "compact", "shifted", "full")


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    label_target: str
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    use_text_features: bool
    split_by_case: bool
    seed: int


@dataclass
class PlannerExample:
    source: str
    benchmark: str
    task: str
    case_id: str
    context_tokens: int
    query_tokens: int
    top_k: int
    page_tokens: int
    selected_pages: str
    label_sparse: str
    label_with_full: str
    label_safe_vs_full: str
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    policy: str
    source: str
    benchmark: str
    task: str
    case_id: str
    target_label: str
    predicted_mode: str
    method: str
    score: float
    exact_correct: int
    answer_nll: float
    active_kv_tokens: int
    active_kv_ratio_vs_full: float
    speedup_vs_full_online: float
    label_correct: int


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a position-mode planner from RoPE KV repack benchmark results.")
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_target", choices=["sparse", "with_full", "safe_vs_full"], default="with_full")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--split_by_case", action="store_true")
    parser.add_argument("--seed", type=int, default=2026070702)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        label_target=args.label_target,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        use_text_features=args.use_text_features,
        split_by_case=args.split_by_case,
        seed=args.seed,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def load_run_config(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return dict(payload.get("config", {}))


def bench_config_from_run(raw: dict[str, Any]) -> BenchConfig:
    def tuple_from_raw(name: str) -> tuple[str, ...]:
        value = raw.get(name, ())
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return tuple(str(item) for item in value)

    def int_tuple_from_raw(name: str) -> tuple[int, ...]:
        value = raw.get(name, ())
        if isinstance(value, str):
            return tuple(int(item.strip()) for item in value.split(",") if item.strip())
        return tuple(int(item) for item in value)

    return BenchConfig(
        output_dir=raw.get("output_dir", ""),
        model_name_or_path=raw.get("model_name_or_path", ""),
        adapter_path="",
        longbench_data_dir=raw.get("longbench_data_dir", ""),
        ruler_data_dir=raw.get("ruler_data_dir", ""),
        longbench_tasks=tuple_from_raw("longbench_tasks"),
        ruler_tasks=tuple_from_raw("ruler_tasks"),
        ruler_context_lengths=int_tuple_from_raw("ruler_context_lengths"),
        methods=(),
        max_examples_per_task=int(raw.get("max_examples_per_task", 1)),
        block_tokens=int(raw.get("page_tokens", 512)),
        recent_tokens=0,
        max_input_tokens=int(raw.get("max_context_tokens", 4096)),
        summary10_words=10,
        summary100_words=100,
        summary1000_words=900,
        max_new_tokens_exact=int(raw.get("max_new_tokens_exact", 48)),
        max_new_tokens_summary=int(raw.get("max_new_tokens_summary", 120)),
        dtype=raw.get("dtype", "float16"),
        attn_implementation=raw.get("attn_implementation", "sdpa"),
        device_map="cuda",
        cuda_visible_devices="",
        router_path="",
        seed=int(raw.get("seed", 0)),
    )


def query_text(case: BenchCase) -> str:
    if case.benchmark.startswith("ruler"):
        return case.query
    template = LONG_BENCH_PROMPTS.get(case.task, "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
    return template.format(context="", input=case.query)


def content_words(text: str) -> set[str]:
    terms = re.findall(r"[A-Za-z0-9_\-]{3,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "what", "which", "answer", "question"}
    return {term for term in terms if term not in stop}


TEXT_FEATURE_NAMES = [
    "query_terms_log",
    "query_code_terms_log",
    "retriever_top1_score",
    "retriever_top2_score",
    "retriever_top3_score",
    "retriever_gap12",
    "retriever_gap23",
    "retriever_positive_pages_log",
    "retriever_top1_norm",
    "retriever_gap12_norm",
    "selected_score_mean",
    "selected_score_min",
    "selected_score_max",
    "selected_score_sum",
    "selected_contains_top1",
    "selected_score_std",
]


def zero_text_features() -> list[float]:
    return [0.0] * len(TEXT_FEATURE_NAMES)


def tokenize_context(tokenizer: Any, case: BenchCase, max_tokens: int) -> list[int]:
    return tokenizer(case.context, add_special_tokens=False)["input_ids"][:max_tokens]


def page_scores(tokenizer: Any, context_ids: list[int], query: str, page_tokens: int) -> tuple[list[float], float]:
    query_terms = content_words(query)
    query_codes = set(re.findall(r"[A-Za-z]+[-_][A-Za-z0-9_\-]+|\d+", query))
    scores: list[float] = []
    for start in range(0, len(context_ids), page_tokens):
        text = tokenizer.decode(context_ids[start : start + page_tokens], skip_special_tokens=True)
        score = float(len(query_terms & content_words(text)))
        score += 4.0 * len(query_codes & set(re.findall(r"[A-Za-z]+[-_][A-Za-z0-9_\-]+|\d+", text)))
        scores.append(score)
    denom = max(1.0, float(len(query_terms) + 4 * len(query_codes)))
    return scores, denom


def text_features_for_case(
    tokenizer: Any,
    case: BenchCase,
    max_context_tokens: int,
    page_tokens: int,
    pages: list[int],
) -> list[float]:
    context_ids = tokenize_context(tokenizer, case, max_context_tokens)
    query = query_text(case)
    scores, denom = page_scores(tokenizer, context_ids, query, page_tokens)
    if not scores:
        return zero_text_features()
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
    query_codes = set(re.findall(r"[A-Za-z]+[-_][A-Za-z0-9_\-]+|\d+", query))
    return [
        math.log1p(len(content_words(query))),
        math.log1p(len(query_codes)),
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


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["benchmark"], row["task"], row["case_id"])][row["method"]] = row
    return grouped


def parse_pages(value: str) -> list[int]:
    try:
        payload = json.loads(value)
    except Exception:
        return []
    if isinstance(payload, list):
        return [int(item) for item in payload]
    return []


def row_score(row: dict[str, str]) -> float:
    return float(row["score"])


def row_nll(row: dict[str, str]) -> float:
    return float(row["answer_nll"])


def row_kv(row: dict[str, str]) -> int:
    return int(float(row["active_kv_tokens"]))


def choose_best(candidates: list[tuple[str, dict[str, str]]]) -> tuple[str, dict[str, str]]:
    return max(candidates, key=lambda item: (row_score(item[1]), -row_nll(item[1]), -row_kv(item[1])))


def choose_min_safe_vs_full(by_method: dict[str, dict[str, str]]) -> tuple[str, dict[str, str]]:
    full = by_method[MODE_TO_METHOD["full"]]
    full_score = row_score(full)
    mode_order = {"compact": 0, "absolute": 1, "shifted": 2, "full": 3}
    candidates = [
        (mode, by_method[MODE_TO_METHOD[mode]])
        for mode in DEPLOY_MODES
        if row_score(by_method[MODE_TO_METHOD[mode]]) + 1e-12 >= full_score
    ]
    if not candidates:
        return "full", full
    return min(candidates, key=lambda item: (row_kv(item[1]), mode_order[item[0]], row_nll(item[1])))


def feature_names(task_to_id: dict[str, int], benchmark_to_id: dict[str, int], source_to_id: dict[str, int]) -> list[str]:
    return (
        [
            "is_longbench",
            "is_ruler",
            "context_tokens_log",
            "query_tokens_log",
            "top_k",
            "page_tokens_log",
            "selected_pages",
            "first_page_norm",
            "last_page_norm",
            "mean_page_norm",
            "span_width_norm",
            "max_gap_norm",
            "min_gap_norm",
            "has_page0",
            "all_pages_adjacent",
            "selected_kv_ratio",
        ]
        + TEXT_FEATURE_NAMES
        + [f"task={name}" for name in sorted(task_to_id, key=task_to_id.get)]
        + [f"benchmark={name}" for name in sorted(benchmark_to_id, key=benchmark_to_id.get)]
        + [f"source={name}" for name in sorted(source_to_id, key=source_to_id.get)]
    )


def one_hot(name: str, mapping: dict[str, int]) -> list[float]:
    values = [0.0] * len(mapping)
    if name in mapping:
        values[mapping[name]] = 1.0
    return values


def build_feature_vector(
    *,
    source: str,
    benchmark: str,
    task: str,
    context_tokens: int,
    query_tokens: int,
    top_k: int,
    page_tokens: int,
    pages: list[int],
    text_features: list[float],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    source_to_id: dict[str, int],
) -> list[float]:
    total_pages = max(1, math.ceil(context_tokens / max(1, page_tokens)))
    norm_denom = max(1.0, float(total_pages - 1))
    if pages:
        sorted_pages = sorted(pages)
        gaps = [b - a for a, b in zip(sorted_pages, sorted_pages[1:])]
        first = float(sorted_pages[0]) / norm_denom
        last = float(sorted_pages[-1]) / norm_denom
        mean = sum(sorted_pages) / len(sorted_pages) / norm_denom
        span = float(sorted_pages[-1] - sorted_pages[0]) / norm_denom
        max_gap = float(max(gaps)) / norm_denom if gaps else 0.0
        min_gap = float(min(gaps)) / norm_denom if gaps else 0.0
        adjacent = 1.0 if gaps and all(gap == 1 for gap in gaps) else (1.0 if len(sorted_pages) <= 1 else 0.0)
        has_page0 = 1.0 if 0 in sorted_pages else 0.0
    else:
        first = last = mean = span = max_gap = min_gap = adjacent = has_page0 = 0.0
    base = [
        1.0 if benchmark == "longbench" else 0.0,
        1.0 if benchmark.startswith("ruler") else 0.0,
        math.log1p(context_tokens),
        math.log1p(query_tokens),
        float(top_k),
        math.log1p(page_tokens),
        float(len(pages)),
        first,
        last,
        mean,
        span,
        max_gap,
        min_gap,
        has_page0,
        adjacent,
        float(len(pages) * page_tokens) / max(1.0, float(context_tokens)),
    ]
    return (
        base
        + text_features
        + one_hot(task, task_to_id)
        + one_hot(benchmark, benchmark_to_id)
        + one_hot(source, source_to_id)
    )


def build_examples(config: Config) -> tuple[list[PlannerExample], dict[tuple[str, str, str, str], dict[str, str]], list[str]]:
    raw_cases: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    tokenizer_cache: dict[str, Any] = {}
    auto_tokenizer: Any | None = None
    for directory_text in config.benchmark_output_dirs:
        directory = Path(directory_text)
        source = directory.name
        run_cfg = load_run_config(directory)
        case_lookup: dict[tuple[str, str, str], BenchCase] = {}
        tokenizer: Any | None = None
        if config.use_text_features:
            if auto_tokenizer is None:
                from transformers import AutoTokenizer

                auto_tokenizer = AutoTokenizer
            bcfg = bench_config_from_run(run_cfg)
            case_lookup = {
                (case.benchmark, case.task, case.case_id): case
                for case in (load_longbench_cases(bcfg) + load_ruler_cases(bcfg))
            }
            model_path = str(run_cfg.get("model_name_or_path", ""))
            if model_path:
                if model_path not in tokenizer_cache:
                    tokenizer_cache[model_path] = auto_tokenizer.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = tokenizer_cache[model_path]
        rows = read_csv(directory / "results.csv")
        grouped = group_rows(rows)
        top_k = int(run_cfg.get("top_k", 0) or 0)
        page_tokens = int(run_cfg.get("page_tokens", 0) or 0)
        max_context_tokens = int(run_cfg.get("max_context_tokens", 4096) or 4096)
        for key, by_method in grouped.items():
            if not all(MODE_TO_METHOD[mode] in by_method for mode in ("absolute", "compact", "shifted", "full", "prompt")):
                continue
            sparse = [(mode, by_method[MODE_TO_METHOD[mode]]) for mode in SPARSE_MODES]
            label_sparse, _ = choose_best(sparse)
            deploy = [(mode, by_method[MODE_TO_METHOD[mode]]) for mode in DEPLOY_MODES]
            label_with_full, _ = choose_best(deploy)
            label_safe_vs_full, _ = choose_min_safe_vs_full(by_method)
            ref = by_method["full_kv_cache"]
            pages = parse_pages(by_method[MODE_TO_METHOD["compact"]]["selected_pages"])
            text_features = zero_text_features()
            case = case_lookup.get(key)
            if tokenizer is not None and case is not None:
                text_features = text_features_for_case(tokenizer, case, max_context_tokens, page_tokens, pages)
            source_key = f"{source}::{key[0]}::{key[1]}::{key[2]}"
            raw_cases.append(
                {
                    "source": source,
                    "source_key": source_key,
                    "benchmark": key[0],
                    "task": key[1],
                    "case_id": key[2],
                    "context_tokens": int(float(ref["context_tokens"])),
                    "query_tokens": int(float(ref["query_tokens"])),
                    "top_k": top_k,
                    "page_tokens": page_tokens,
                    "selected_pages": json.dumps(pages),
                    "pages": pages,
                    "text_features": text_features,
                    "label_sparse": label_sparse,
                    "label_with_full": label_with_full,
                    "label_safe_vs_full": label_safe_vs_full,
                }
            )
            for method, row in by_method.items():
                lookup[(source_key, key[0], key[1], key[2], method)] = row
    if not raw_cases:
        raise ValueError("no planner examples were built")

    task_to_id = {task: idx for idx, task in enumerate(sorted({case["task"] for case in raw_cases}))}
    benchmark_to_id = {
        bench: idx for idx, bench in enumerate(sorted({case["benchmark"] for case in raw_cases}))
    }
    source_to_id = {source: idx for idx, source in enumerate(sorted({case["source"] for case in raw_cases}))}
    names = feature_names(task_to_id, benchmark_to_id, source_to_id)
    examples = [
        PlannerExample(
            source=case["source_key"],
            benchmark=case["benchmark"],
            task=case["task"],
            case_id=case["case_id"],
            context_tokens=case["context_tokens"],
            query_tokens=case["query_tokens"],
            top_k=case["top_k"],
            page_tokens=case["page_tokens"],
            selected_pages=case["selected_pages"],
            label_sparse=case["label_sparse"],
            label_with_full=case["label_with_full"],
            label_safe_vs_full=case["label_safe_vs_full"],
            features=build_feature_vector(
                source=case["source"],
                benchmark=case["benchmark"],
                task=case["task"],
                context_tokens=case["context_tokens"],
                query_tokens=case["query_tokens"],
                top_k=case["top_k"],
                page_tokens=case["page_tokens"],
                pages=case["pages"],
                text_features=case["text_features"],
                task_to_id=task_to_id,
                benchmark_to_id=benchmark_to_id,
                source_to_id=source_to_id,
            ),
        )
        for case in raw_cases
    ]
    return examples, lookup, names


def split_indices(examples: list[PlannerExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)

    def example_label(example: PlannerExample) -> str:
        if config.label_target == "sparse":
            return example.label_sparse
        if config.label_target == "safe_vs_full":
            return example.label_safe_vs_full
        return example.label_with_full

    if config.split_by_case:
        case_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, example in enumerate(examples):
            case_groups[(example.benchmark, example.task, example.case_id)].append(idx)
        buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for key, indices in case_groups.items():
            group_labels = [
                example_label(examples[idx])
                for idx in indices
            ]
            label = Counter(group_labels).most_common(1)[0][0]
            buckets[(key[1], label)].append(key)
        train: list[int] = []
        test: list[int] = []
        for group_keys in buckets.values():
            rng.shuffle(group_keys)
            if len(group_keys) == 1:
                train.extend(idx for key in group_keys for idx in case_groups[key])
                continue
            n_test = max(1, min(len(group_keys) - 1, round(len(group_keys) * config.test_fraction)))
            for key in group_keys[:n_test]:
                test.extend(case_groups[key])
            for key in group_keys[n_test:]:
                train.extend(case_groups[key])
        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        label = example_label(example)
        buckets[(example.task, label)].append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in buckets.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        n_test = max(1, min(len(indices) - 1, round(len(indices) * config.test_fraction)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def normalize(examples: list[PlannerExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        m = sum(vals) / max(1, len(vals))
        var = sum((value - m) ** 2 for value in vals) / max(1, len(vals))
        mean.append(m)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def norm_features(features: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]


def labels(config: Config, examples: list[PlannerExample]) -> list[str]:
    if config.label_target == "sparse":
        return [ex.label_sparse for ex in examples]
    if config.label_target == "safe_vs_full":
        return [ex.label_safe_vs_full for ex in examples]
    return [ex.label_with_full for ex in examples]


def train_model(
    examples: list[PlannerExample],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    config: Config,
) -> tuple[MLP, dict[str, int], list[dict[str, Any]]]:
    label_names = sorted(set(labels(config, examples)))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    def xy(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [norm_features(examples[idx].features, mean, std) for idx in indices]
        ys = [label_to_id[labels(config, examples)[idx]] for idx in indices]
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = xy(train_indices)
    test_x, test_y = xy(test_indices) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = MLP(train_x.shape[1], config.hidden_dim, len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    if torch.any(weights > 0):
        weights = weights / weights[weights > 0].mean().clamp_min(1e-6)
    else:
        weights = None
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
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
    return model, label_to_id, history


def row_for_mode(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    example: PlannerExample,
    mode: str,
) -> dict[str, str]:
    method = MODE_TO_METHOD[mode]
    return lookup[(example.source, example.benchmark, example.task, example.case_id, method)]


def full_kv_tokens(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    example: PlannerExample,
) -> int:
    return row_kv(row_for_mode(lookup, example, "full"))


def add_prediction(
    rows: list[PredictionRow],
    split: str,
    policy: str,
    example: PlannerExample,
    target: str,
    predicted_mode: str,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    row = row_for_mode(lookup, example, predicted_mode)
    full_tokens = full_kv_tokens(lookup, example)
    active = row_kv(row)
    rows.append(
        PredictionRow(
            split=split,
            policy=policy,
            source=example.source,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            target_label=target,
            predicted_mode=predicted_mode,
            method=MODE_TO_METHOD[predicted_mode],
            score=row_score(row),
            exact_correct=int(float(row["exact_correct"])),
            answer_nll=row_nll(row),
            active_kv_tokens=active,
            active_kv_ratio_vs_full=active / full_tokens if full_tokens else 0.0,
            speedup_vs_full_online=float(row["speedup_vs_full_online"]),
            label_correct=int(predicted_mode == target),
        )
    )


def oracle_sparse_mode(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    example: PlannerExample,
) -> str:
    mode, _ = choose_best([(mode, row_for_mode(lookup, example, mode)) for mode in SPARSE_MODES])
    return mode


def oracle_with_full_mode(
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    example: PlannerExample,
) -> str:
    mode, _ = choose_best([(mode, row_for_mode(lookup, example, mode)) for mode in DEPLOY_MODES])
    return mode


def evaluate(
    model: MLP,
    examples: list[PlannerExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    out: list[PredictionRow] = []
    model.eval()
    for idx in indices:
        example = examples[idx]
        if config.label_target == "sparse":
            target = example.label_sparse
        elif config.label_target == "safe_vs_full":
            target = example.label_safe_vs_full
        else:
            target = example.label_with_full
        for mode in DEPLOY_MODES:
            add_prediction(out, split, f"fixed_{mode}", example, target, mode, lookup)
        add_prediction(out, split, "prompt_rebuild", example, target, "prompt", lookup)
        add_prediction(out, split, "oracle_sparse", example, target, oracle_sparse_mode(lookup, example), lookup)
        add_prediction(out, split, "oracle_with_full", example, target, oracle_with_full_mode(lookup, example), lookup)
        x = torch.tensor([norm_features(example.features, mean, std)], dtype=torch.float32)
        with torch.inference_mode():
            pred = id_to_label[int(model(x).argmax(-1).item())]
        add_prediction(out, split, "learned_planner", example, target, pred, lookup)
    return out


def summarize(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, row.benchmark)].append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "group": group,
            "samples": len(items),
            "avg_score": sum(item.score for item in items) / len(items),
            "exact_accuracy": sum(item.exact_correct for item in items) / len(items),
            "avg_answer_nll": sum(item.answer_nll for item in items) / len(items),
            "avg_active_kv_ratio_vs_full": sum(item.active_kv_ratio_vs_full for item in items) / len(items),
            "avg_speedup_vs_full_online": sum(item.speedup_vs_full_online for item in items) / len(items),
            "label_accuracy": sum(item.label_correct for item in items) / len(items),
        }
        counts = Counter(item.predicted_mode for item in items)
        for mode, count in sorted(counts.items()):
            payload[f"select_{mode}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, lookup, names = build_examples(config)
    train_indices, test_indices = split_indices(examples, config)
    mean, std = normalize(examples, train_indices)
    model, label_to_id, history = train_model(examples, train_indices, test_indices, mean, std, config)
    rows = evaluate(model, examples, train_indices, "train", label_to_id, mean, std, lookup, config)
    rows += evaluate(model, examples, test_indices, "test", label_to_id, mean, std, lookup, config)
    summary = summarize(rows)
    write_csv(output_dir / "examples.csv", [asdict(example) for example in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(examples[0].features),
            "hidden_dim": config.hidden_dim,
            "label_to_id": label_to_id,
            "feature_names": names,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "position_mode_planner.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "examples": len(examples),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "label_counts_sparse": dict(Counter(example.label_sparse for example in examples)),
                "label_counts_with_full": dict(Counter(example.label_with_full for example in examples)),
                "label_counts_safe_vs_full": dict(Counter(example.label_safe_vs_full for example in examples)),
                "history_tail": history[-5:],
                "prediction_summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("split,policy,group,samples,score,kv_ratio,label_acc")
    for row in summary:
        if row["split"] == "test" and row["group"] == "__overall__":
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['avg_active_kv_ratio_vs_full']:.4f},{row['label_accuracy']:.4f}"
            )
    print(f"saved planner to {output_dir / 'position_mode_planner.pt'}")


if __name__ == "__main__":
    main()
