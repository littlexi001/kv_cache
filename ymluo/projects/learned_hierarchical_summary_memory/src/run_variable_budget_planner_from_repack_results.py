from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_position_mode_planner_from_repack_results as base  # noqa: E402


COMPACT_METHOD = "rope_delta_repack_compact_query_pos"
FULL_METHOD = "full_kv_cache"
PROMPT_METHOD = "prompt_rebuild_selected_pages"


@dataclass(frozen=True)
class Config:
    benchmark_dirs: tuple[str, ...]
    benchmark_groups: tuple[tuple[str, ...], ...]
    output_dir: str
    label_target: str
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    confidence_penalty: float
    ce_loss_weight: float
    expected_cost_weight: float
    unsafe_cost_weight: float
    best_gap_cost_weight: float
    kv_cost_weight: float
    include_full_action: bool
    test_fraction: float
    use_text_features: bool
    split_by_case: bool
    risk_thresholds: tuple[float, ...]
    holdout_tasks: tuple[str, ...]
    holdout_benchmarks: tuple[str, ...]
    seed: int


@dataclass
class VariableBudgetExample:
    source: str
    benchmark: str
    task: str
    case_id: str
    context_tokens: int
    query_tokens: int
    page_tokens: int
    available_budgets: str
    selected_pages_by_budget: str
    label_min_safe: str
    label_best: str
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
    predicted_action: str
    method: str
    score: float
    exact_correct: int
    answer_nll: float
    active_kv_tokens: int
    active_kv_ratio_vs_full: float
    speedup_vs_full_online: float
    label_correct: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a variable-budget risk-constrained KV planner.")
    parser.add_argument("--benchmark_dirs", default="", help="Comma-separated RoPE repack benchmark output dirs.")
    parser.add_argument(
        "--benchmark_groups",
        default="",
        help="Groups of comma-separated benchmark dirs separated by ';' or '@@'; each group is intersected independently.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_target", choices=["min_safe", "best"], default="min_safe")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--confidence_penalty", type=float, default=0.0)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--expected_cost_weight", type=float, default=0.0)
    parser.add_argument("--unsafe_cost_weight", type=float, default=2.0)
    parser.add_argument("--best_gap_cost_weight", type=float, default=1.0)
    parser.add_argument("--kv_cost_weight", type=float, default=0.05)
    parser.add_argument("--include_full_action", action="store_true")
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--split_by_case", action="store_true")
    parser.add_argument(
        "--risk_thresholds",
        default="0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.8,1.0",
        help="Comma-separated tail-risk thresholds for calibrated budget promotion.",
    )
    parser.add_argument("--holdout_tasks", default="", help="Comma-separated tasks used only for test.")
    parser.add_argument("--holdout_benchmarks", default="", help="Comma-separated benchmarks used only for test.")
    parser.add_argument("--seed", type=int, default=2026070714)
    args = parser.parse_args()
    benchmark_dirs = base.parse_csv_tuple(args.benchmark_dirs)
    benchmark_groups = parse_benchmark_groups(args.benchmark_groups)
    if not benchmark_groups and benchmark_dirs:
        benchmark_groups = (benchmark_dirs,)
    if not benchmark_groups:
        raise ValueError("provide --benchmark_dirs or --benchmark_groups")
    return Config(
        benchmark_dirs=benchmark_dirs,
        benchmark_groups=benchmark_groups,
        output_dir=args.output_dir,
        label_target=args.label_target,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        confidence_penalty=args.confidence_penalty,
        ce_loss_weight=args.ce_loss_weight,
        expected_cost_weight=args.expected_cost_weight,
        unsafe_cost_weight=args.unsafe_cost_weight,
        best_gap_cost_weight=args.best_gap_cost_weight,
        kv_cost_weight=args.kv_cost_weight,
        include_full_action=args.include_full_action,
        test_fraction=args.test_fraction,
        use_text_features=args.use_text_features,
        split_by_case=args.split_by_case,
        risk_thresholds=tuple(float(value) for value in base.parse_csv_tuple(args.risk_thresholds)),
        holdout_tasks=base.parse_csv_tuple(args.holdout_tasks),
        holdout_benchmarks=base.parse_csv_tuple(args.holdout_benchmarks),
        seed=args.seed,
    )


def parse_benchmark_groups(value: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    normalized = value.replace("@@", ";")
    for raw_group in normalized.split(";"):
        group = base.parse_csv_tuple(raw_group)
        if group:
            groups.append(group)
    return tuple(groups)


def row_score(row: dict[str, str]) -> float:
    return float(row["score"])


def row_nll(row: dict[str, str]) -> float:
    return float(row["answer_nll"])


def row_kv(row: dict[str, str]) -> int:
    return int(float(row["active_kv_tokens"]))


def load_dir(directory: Path) -> tuple[int, dict[str, Any], dict[tuple[str, str, str], dict[str, dict[str, str]]]]:
    run_cfg = base.load_run_config(directory)
    top_k = int(run_cfg.get("top_k", 0) or 0)
    if top_k <= 0:
        raise ValueError(f"cannot infer top_k from {directory}")
    rows = base.read_csv(directory / "results.csv")
    return top_k, run_cfg, base.group_rows(rows)


def action_name_for_k(top_k: int) -> str:
    return f"k{top_k}_compact"


def action_budget(action: str) -> int:
    if action == "full":
        return 10**9
    if action.startswith("k") and action.endswith("_compact"):
        return int(action[1 : action.index("_")])
    raise ValueError(action)


def action_row(
    case_payload: dict[str, Any],
    action: str,
) -> dict[str, str]:
    if action == "full":
        return case_payload["full"]
    top_k = action_budget(action)
    return case_payload["by_k"][top_k][COMPACT_METHOD]


def prompt_action_row(case_payload: dict[str, Any], top_k: int) -> dict[str, str]:
    return case_payload["by_k"][top_k][PROMPT_METHOD]


def available_actions(case_payload: dict[str, Any]) -> list[str]:
    return [action_name_for_k(k) for k in sorted(case_payload["by_k"])] + ["full"]


def choose_min_safe(case_payload: dict[str, Any]) -> str:
    full_score = row_score(case_payload["full"])
    for action in sorted(available_actions(case_payload), key=lambda name: (action_budget(name), row_nll(action_row(case_payload, name)))):
        if row_score(action_row(case_payload, action)) + 1e-12 >= full_score:
            return action
    return "full"


def choose_best(case_payload: dict[str, Any]) -> str:
    return max(
        available_actions(case_payload),
        key=lambda action: (
            row_score(action_row(case_payload, action)),
            -row_kv(action_row(case_payload, action)),
            -row_nll(action_row(case_payload, action)),
        ),
    )


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


def feature_names(
    budgets: list[int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    source_to_id: dict[str, int],
) -> list[str]:
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
    names = [
        "is_longbench",
        "is_ruler",
        "context_tokens_log",
        "query_tokens_log",
        "page_tokens_log",
        "num_budgets",
        "min_budget",
        "max_budget",
    ]
    for top_k in budgets:
        names.append(f"k{top_k}_available")
        names.extend(f"k{top_k}_{name}" for name in layout_names)
        names.extend(f"k{top_k}_{name}" for name in base.TEXT_FEATURE_NAMES)
    names.extend(f"delta_k{left}_to_k{right}_page_jaccard" for left, right in zip(budgets, budgets[1:]))
    names.extend(f"delta_k{left}_to_k{right}_added_pages" for left, right in zip(budgets, budgets[1:]))
    names.extend([f"task={name}" for name in sorted(task_to_id, key=task_to_id.get)])
    names.extend([f"benchmark={name}" for name in sorted(benchmark_to_id, key=benchmark_to_id.get)])
    names.extend([f"source={name}" for name in sorted(source_to_id, key=source_to_id.get)])
    return names


def one_hot(name: str, mapping: dict[str, int]) -> list[float]:
    values = [0.0] * len(mapping)
    if name in mapping:
        values[mapping[name]] = 1.0
    return values


def build_examples(
    config: Config,
) -> tuple[
    list[VariableBudgetExample],
    dict[tuple[str, str, str, str], dict[str, Any]],
    list[str],
]:
    tokenizer_cache: dict[str, Any] = {}
    raw_cases: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    all_budgets: set[int] = set()
    auto_tokenizer: Any | None = None
    if config.use_text_features:
        from transformers import AutoTokenizer

        auto_tokenizer = AutoTokenizer

    for group_idx, benchmark_group in enumerate(config.benchmark_groups):
        loaded: list[tuple[int, Path, dict[str, Any], dict[tuple[str, str, str], dict[str, dict[str, str]]]]] = []
        for directory_text in benchmark_group:
            directory = Path(directory_text)
            top_k, run_cfg, grouped = load_dir(directory)
            loaded.append((top_k, directory, run_cfg, grouped))
        loaded.sort(key=lambda item: item[0])
        group_budgets = sorted({top_k for top_k, _, _, _ in loaded})
        all_budgets.update(group_budgets)
        source = f"group{group_idx}__" + "__".join(f"k{top_k}" for top_k in group_budgets)

        common_keys = set(loaded[0][3])
        for _, _, _, grouped in loaded[1:]:
            common_keys &= set(grouped)

        case_lookup: dict[tuple[str, str, str], base.BenchCase] = {}
        tokenizer: Any | None = None
        first_cfg = loaded[0][2]
        page_tokens = int(first_cfg.get("page_tokens", 512) or 512)
        max_context_tokens = int(first_cfg.get("max_context_tokens", 4096) or 4096)
        if config.use_text_features and auto_tokenizer is not None:
            bcfg = base.bench_config_from_run(first_cfg)
            case_lookup = {
                (case.benchmark, case.task, case.case_id): case
                for case in (base.load_longbench_cases(bcfg) + base.load_ruler_cases(bcfg))
            }
            model_path = str(first_cfg.get("model_name_or_path", ""))
            if model_path:
                if model_path not in tokenizer_cache:
                    tokenizer_cache[model_path] = auto_tokenizer.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = tokenizer_cache[model_path]

        for key in sorted(common_keys):
            by_k: dict[int, dict[str, dict[str, str]]] = {}
            full_row: dict[str, str] | None = None
            for top_k, _, _, grouped in loaded:
                by_method = grouped[key]
                if COMPACT_METHOD not in by_method or FULL_METHOD not in by_method or PROMPT_METHOD not in by_method:
                    by_k = {}
                    break
                by_k[top_k] = by_method
                full_row = full_row or by_method[FULL_METHOD]
            if not by_k or full_row is None:
                continue
            case_payload = {"by_k": by_k, "full": full_row}
            label_min_safe = choose_min_safe(case_payload)
            label_best = choose_best(case_payload)
            pages_by_k = {
                str(top_k): base.parse_pages(by_k[top_k][COMPACT_METHOD]["selected_pages"])
                for top_k in group_budgets
                if top_k in by_k
            }
            text_features_by_k = {str(top_k): base.zero_text_features() for top_k in group_budgets}
            case = case_lookup.get(key)
            if tokenizer is not None and case is not None:
                for top_k, pages in pages_by_k.items():
                    text_features_by_k[top_k] = base.text_features_for_case(
                        tokenizer, case, max_context_tokens, page_tokens, pages
                    )
            source_key = f"{source}::{key[0]}::{key[1]}::{key[2]}"
            raw_cases.append(
                {
                    "source": source,
                    "source_key": source_key,
                    "benchmark": key[0],
                    "task": key[1],
                    "case_id": key[2],
                    "context_tokens": int(float(full_row["context_tokens"])),
                    "query_tokens": int(float(full_row["query_tokens"])),
                    "page_tokens": page_tokens,
                    "pages_by_k": pages_by_k,
                    "text_features_by_k": text_features_by_k,
                    "label_min_safe": label_min_safe,
                    "label_best": label_best,
                }
            )
            lookup[(source_key, key[0], key[1], key[2])] = case_payload
    if not raw_cases:
        raise ValueError("no variable-budget examples were built")

    budgets = sorted(all_budgets)
    task_to_id = {task: idx for idx, task in enumerate(sorted({case["task"] for case in raw_cases}))}
    benchmark_to_id = {bench: idx for idx, bench in enumerate(sorted({case["benchmark"] for case in raw_cases}))}
    source_to_id = {source_name: idx for idx, source_name in enumerate(sorted({case["source"] for case in raw_cases}))}
    names = feature_names(budgets, task_to_id, benchmark_to_id, source_to_id)

    examples: list[VariableBudgetExample] = []
    for case in raw_cases:
        values: dict[str, float] = {
            "is_longbench": 1.0 if case["benchmark"] == "longbench" else 0.0,
            "is_ruler": 1.0 if case["benchmark"].startswith("ruler") else 0.0,
            "context_tokens_log": math.log1p(case["context_tokens"]),
            "query_tokens_log": math.log1p(case["query_tokens"]),
            "page_tokens_log": math.log1p(case["page_tokens"]),
            "num_budgets": float(len(budgets)),
            "min_budget": float(min(budgets)),
            "max_budget": float(max(budgets)),
            f"task={case['task']}": 1.0,
            f"benchmark={case['benchmark']}": 1.0,
            f"source={case['source']}": 1.0,
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
        page_sets: dict[int, set[int]] = {}
        for top_k in budgets:
            pages = case["pages_by_k"].get(str(top_k), [])
            page_sets[top_k] = set(pages)
            values[f"k{top_k}_available"] = 1.0 if str(top_k) in case["pages_by_k"] else 0.0
            for name, value in zip(
                layout_names, page_layout_features(case["context_tokens"], case["page_tokens"], pages)
            ):
                values[f"k{top_k}_{name}"] = float(value)
            for name, value in zip(base.TEXT_FEATURE_NAMES, case["text_features_by_k"].get(str(top_k), base.zero_text_features())):
                values[f"k{top_k}_{name}"] = float(value)
        for left, right in zip(budgets, budgets[1:]):
            union = page_sets[left] | page_sets[right]
            values[f"delta_k{left}_to_k{right}_page_jaccard"] = (
                float(len(page_sets[left] & page_sets[right])) / max(1.0, float(len(union)))
            )
            values[f"delta_k{left}_to_k{right}_added_pages"] = float(len(page_sets[right] - page_sets[left]))

        features = [values.get(name, 0.0) for name in names]
        examples.append(
            VariableBudgetExample(
                source=case["source_key"],
                benchmark=case["benchmark"],
                task=case["task"],
                case_id=case["case_id"],
                context_tokens=case["context_tokens"],
                query_tokens=case["query_tokens"],
                page_tokens=case["page_tokens"],
                available_budgets=json.dumps(budgets),
                selected_pages_by_budget=json.dumps(case["pages_by_k"]),
                label_min_safe=case["label_min_safe"],
                label_best=case["label_best"],
                features=features,
            )
        )
    return examples, lookup, names


def label_for(config: Config, example: VariableBudgetExample) -> str:
    return example.label_best if config.label_target == "best" else example.label_min_safe


def split_indices(examples: list[VariableBudgetExample], config: Config) -> tuple[list[int], list[int]]:
    holdout_tasks = set(config.holdout_tasks)
    holdout_benchmarks = set(config.holdout_benchmarks)
    if holdout_tasks or holdout_benchmarks:
        train: list[int] = []
        test: list[int] = []
        for idx, example in enumerate(examples):
            is_holdout = (
                (example.task in holdout_tasks if holdout_tasks else False)
                or (example.benchmark in holdout_benchmarks if holdout_benchmarks else False)
            )
            if is_holdout:
                test.append(idx)
            else:
                train.append(idx)
        if not train or not test:
            raise ValueError("--holdout_tasks/--holdout_benchmarks produced an empty train or test split")
        rng = random.Random(config.seed)
        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    rng = random.Random(config.seed)
    if config.split_by_case:
        groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, example in enumerate(examples):
            groups[(example.benchmark, example.task, example.case_id)].append(idx)
        buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for key, indices in groups.items():
            label = Counter(label_for(config, examples[idx]) for idx in indices).most_common(1)[0][0]
            buckets[(key[1], label)].append(key)
        train: list[int] = []
        test: list[int] = []
        for keys in buckets.values():
            rng.shuffle(keys)
            if len(keys) == 1:
                train.extend(idx for key in keys for idx in groups[key])
                continue
            n_test = max(1, min(len(keys) - 1, round(len(keys) * config.test_fraction)))
            for key in keys[:n_test]:
                test.extend(groups[key])
            for key in keys[n_test:]:
                train.extend(groups[key])
        rng.shuffle(train)
        rng.shuffle(test)
        return train, test

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        buckets[(example.task, label_for(config, example))].append(idx)
    train = []
    test = []
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


def normalize(examples: list[VariableBudgetExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
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


def train_model(
    examples: list[VariableBudgetExample],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    config: Config,
) -> tuple[base.MLP, dict[str, int], list[dict[str, Any]]]:
    labels = [label_for(config, example) for example in examples]
    if config.expected_cost_weight > 0 or config.include_full_action:
        action_names: set[str] = set(labels)
        for example in examples:
            case_payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
            action_names.update(available_actions(case_payload))
        if config.include_full_action:
            action_names.add("full")
        label_names = sorted(action_names, key=lambda name: (action_budget(name), name))
    else:
        label_names = sorted(set(labels), key=lambda name: (action_budget(name), name))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    def xy(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [norm_features(examples[idx].features, mean, std) for idx in indices]
        ys = [label_to_id[label_for(config, examples[idx])] for idx in indices]
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = xy(train_indices)
    test_x, test_y = xy(test_indices) if test_indices else (train_x, train_y)
    train_costs = action_cost_matrix(examples, train_indices, label_names, lookup, config)
    test_costs = action_cost_matrix(examples, test_indices, label_names, lookup, config) if test_indices else train_costs
    torch.manual_seed(config.seed)
    model = base.MLP(train_x.shape[1], config.hidden_dim, len(label_names))
    counts = torch.bincount(train_y, minlength=len(label_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    if torch.any(weights > 0):
        weights = weights / weights[weights > 0].mean().clamp_min(1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        ce_loss = F.cross_entropy(logits, train_y, weight=weights, label_smoothing=config.label_smoothing)
        loss = config.ce_loss_weight * ce_loss
        expected_cost = torch.tensor(0.0, dtype=torch.float32)
        if config.expected_cost_weight > 0:
            probs = torch.softmax(logits, dim=-1)
            expected_cost = (probs * train_costs).sum(dim=-1).mean()
            loss = loss + config.expected_cost_weight * expected_cost
        if config.confidence_penalty > 0:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            loss = loss - config.confidence_penalty * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_logits = model(train_x)
                test_logits = model(test_x)
                train_acc = float((train_logits.argmax(-1) == train_y).float().mean())
                test_acc = float((test_logits.argmax(-1) == test_y).float().mean())
                train_expected_cost = float((torch.softmax(train_logits, dim=-1) * train_costs).sum(dim=-1).mean())
                test_expected_cost = float((torch.softmax(test_logits, dim=-1) * test_costs).sum(dim=-1).mean())
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "ce_loss": float(ce_loss.detach()),
                    "expected_cost": float(expected_cost.detach()),
                    "train_label_accuracy": train_acc,
                    "test_label_accuracy": test_acc,
                    "train_expected_cost": train_expected_cost,
                    "test_expected_cost": test_expected_cost,
                }
            )
    return model, label_to_id, history


def action_cost_matrix(
    examples: list[VariableBudgetExample],
    indices: list[int],
    label_names: list[str],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    config: Config,
) -> torch.Tensor:
    rows: list[list[float]] = []
    for idx in indices:
        example = examples[idx]
        case_payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        full_score = row_score(case_payload["full"])
        best_score = row_score(action_row(case_payload, choose_best(case_payload)))
        ft = max(1, full_tokens(case_payload))
        available = set(available_actions(case_payload))
        costs: list[float] = []
        for action in label_names:
            if action not in available:
                costs.append(100.0)
                continue
            row = action_row(case_payload, action)
            score = row_score(row)
            kv_ratio = row_kv(row) / ft
            unsafe_gap = max(0.0, full_score - score)
            best_gap = max(0.0, best_score - score)
            costs.append(
                config.unsafe_cost_weight * unsafe_gap
                + config.best_gap_cost_weight * best_gap
                + config.kv_cost_weight * kv_ratio
            )
        rows.append(costs)
    return torch.tensor(rows, dtype=torch.float32)


def full_tokens(case_payload: dict[str, Any]) -> int:
    return row_kv(case_payload["full"])


def add_prediction(
    rows: list[PredictionRow],
    split: str,
    policy: str,
    example: VariableBudgetExample,
    target: str,
    action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    case_payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
    row = action_row(case_payload, action)
    ft = full_tokens(case_payload)
    rows.append(
        PredictionRow(
            split=split,
            policy=policy,
            source=example.source,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            target_label=target,
            predicted_action=action,
            method=FULL_METHOD if action == "full" else COMPACT_METHOD,
            score=row_score(row),
            exact_correct=int(float(row["exact_correct"])),
            answer_nll=row_nll(row),
            active_kv_tokens=row_kv(row),
            active_kv_ratio_vs_full=row_kv(row) / ft if ft else 0.0,
            speedup_vs_full_online=float(row.get("speedup_vs_full_online", 0.0)),
            label_correct=int(action == target),
        )
    )


def evaluate(
    model: base.MLP,
    examples: list[VariableBudgetExample],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    config: Config,
) -> list[PredictionRow]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    out: list[PredictionRow] = []
    model.eval()
    for idx in indices:
        example = examples[idx]
        target = label_for(config, example)
        case_payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        for action in available_actions(case_payload):
            add_prediction(out, split, f"fixed_{action}", example, target, action, lookup)
        add_prediction(out, split, "oracle_min_safe", example, target, choose_min_safe(case_payload), lookup)
        add_prediction(out, split, "oracle_best", example, target, choose_best(case_payload), lookup)
        x = torch.tensor([norm_features(example.features, mean, std)], dtype=torch.float32)
        with torch.inference_mode():
            logits = model(x)
            pred = id_to_label[int(logits.argmax(-1).item())]
            probs = torch.softmax(logits, dim=-1)[0]
        add_prediction(out, split, "learned_planner", example, target, pred, lookup)
        for tau in config.risk_thresholds:
            risk_action = choose_tail_risk_action(probs, id_to_label, tau)
            add_prediction(out, split, f"risk_tail_tau_{tau:g}", example, target, risk_action, lookup)
    return out


def choose_tail_risk_action(probs: torch.Tensor, id_to_label: dict[int, str], tau: float) -> str:
    labels = sorted(id_to_label.values(), key=lambda name: (action_budget(name), name))
    prob_by_label = {id_to_label[idx]: float(probs[idx]) for idx in id_to_label}
    for label in labels:
        tail_risk = sum(
            prob_by_label[other]
            for other in labels
            if action_budget(other) > action_budget(label)
        )
        if tail_risk <= tau:
            return label
    return labels[-1]


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
        counts = Counter(item.predicted_action for item in items)
        for action, count in sorted(counts.items(), key=lambda pair: (action_budget(pair[0]), pair[0])):
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


def threshold_summary(summary: list[dict[str, Any]]) -> dict[str, Any]:
    test_overall = [
        row for row in summary
        if row["split"] == "test" and row["group"] == "__overall__"
    ]
    by_policy = {row["policy"]: row for row in test_overall}
    full_row = by_policy.get("fixed_full")
    risk_rows = [
        row for row in test_overall
        if row["policy"].startswith("risk_tail_tau_")
    ]
    if not full_row or not risk_rows:
        return {}
    full_score = float(full_row["avg_score"])

    def with_tau(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out["tau"] = float(str(row["policy"]).removeprefix("risk_tail_tau_"))
        return out

    risk_rows_with_tau = [with_tau(row) for row in risk_rows]
    best_score = max(
        risk_rows_with_tau,
        key=lambda row: (row["avg_score"], -row["avg_active_kv_ratio_vs_full"], row["label_accuracy"]),
    )
    full_level_candidates = [
        row for row in risk_rows_with_tau
        if row["avg_score"] + 1e-12 >= full_score
    ]
    if full_level_candidates:
        min_kv_at_full = min(
            full_level_candidates,
            key=lambda row: (row["avg_active_kv_ratio_vs_full"], -row["avg_score"], -row["label_accuracy"]),
        )
    else:
        min_kv_at_full = None
    within_one_point = [
        row for row in risk_rows_with_tau
        if row["avg_score"] + 0.01 + 1e-12 >= full_score
    ]
    min_kv_within_one_point = (
        min(
            within_one_point,
            key=lambda row: (row["avg_active_kv_ratio_vs_full"], -row["avg_score"], -row["label_accuracy"]),
        )
        if within_one_point else None
    )
    return {
        "full_score": full_score,
        "fixed_full": full_row,
        "best_score_then_low_kv": best_score,
        "min_kv_at_full_score": min_kv_at_full,
        "min_kv_within_one_point_of_full": min_kv_within_one_point,
        "risk_rows": risk_rows_with_tau,
    }


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


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, lookup, names = build_examples(config)
    train_indices, test_indices = split_indices(examples, config)
    mean, std = normalize(examples, train_indices)
    model, label_to_id, history = train_model(examples, train_indices, test_indices, mean, std, lookup, config)
    rows = evaluate(model, examples, train_indices, "train", label_to_id, mean, std, lookup, config)
    rows += evaluate(model, examples, test_indices, "test", label_to_id, mean, std, lookup, config)
    summary = summarize(rows)
    write_csv(output_dir / "examples.csv", [asdict(example) for example in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    risk_summary = threshold_summary(summary)
    write_csv(output_dir / "risk_threshold_sweep.csv", risk_summary.get("risk_rows", []))
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
        output_dir / "variable_budget_planner.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "examples": len(examples),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "label_counts_min_safe": dict(Counter(example.label_min_safe for example in examples)),
                "label_counts_best": dict(Counter(example.label_best for example in examples)),
                "history_tail": history[-5:],
                "prediction_summary": summary,
                "risk_threshold_summary": risk_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "risk_threshold_summary.json").write_text(
        json.dumps(risk_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("split,policy,group,samples,score,kv_ratio,label_acc")
    for row in summary:
        if row["split"] == "test" and row["group"] == "__overall__":
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['avg_active_kv_ratio_vs_full']:.4f},{row['label_accuracy']:.4f}"
            )
    print(f"saved planner to {output_dir / 'variable_budget_planner.pt'}")


if __name__ == "__main__":
    main()
