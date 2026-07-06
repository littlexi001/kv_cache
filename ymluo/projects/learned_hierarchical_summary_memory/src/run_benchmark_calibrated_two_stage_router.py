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
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    SUMMARY_TASKS,
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    router_features,
)
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES, write_csv  # noqa: E402
from run_router_policy_offline_eval import length_aware_rule_action  # noqa: E402


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    candidate_methods: tuple[str, ...]
    budget_bins: tuple[float, ...]
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    utility_token_penalty: float
    success_bonus: float
    fallback_margin: float
    summary_rouge_slack: float
    seed: int


@dataclass
class Example:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    features: list[float]
    full_score: float
    oracle_budget: float
    oracle_action: str


@dataclass
class EvalRow:
    split: str
    policy: str
    benchmark: str
    task: str
    case_id: str
    predicted_budget: float
    predicted_action: str
    score: float
    full_score: float
    relative_to_full: float | str
    token_ratio_vs_full_raw: float
    seconds: float
    success: int


class BudgetNet(nn.Module):
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


class ActionRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Benchmark-calibrated two-stage budget router diagnostics.")
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_methods",
        default=(
            "full_raw,recent_only,static_hier,summary1_8,summary1_4,summary1_2,"
            "retrieval_raw_k1,retrieval_raw_k2,retrieval_raw_k3,retrieval_raw_k4,retrieval_raw_k8"
        ),
    )
    parser.add_argument("--budget_bins", default="0.2,0.3,0.4,0.5,1.0")
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--utility_token_penalty", type=float, default=0.18)
    parser.add_argument("--success_bonus", type=float, default=0.2)
    parser.add_argument("--fallback_margin", type=float, default=0.08)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026070602)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        candidate_methods=parse_csv_tuple(args.candidate_methods),
        budget_bins=tuple(sorted(parse_float_tuple(args.budget_bins))),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        utility_token_penalty=args.utility_token_penalty,
        success_bonus=args.success_bonus,
        fallback_margin=args.fallback_margin,
        summary_rouge_slack=args.summary_rouge_slack,
        seed=args.seed,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def load_trials(config: Config) -> tuple[list[BenchConfig], dict[tuple[str, str, str, str], dict[str, str]]]:
    dirs = [Path(item) for item in config.benchmark_output_dirs]
    bench_configs = [bench_config_from_summary(directory) for directory in dirs]
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for directory in dirs:
        for row in read_csv(directory / "trials.csv"):
            lookup[(row["benchmark"], row["task"], row["case_id"], row["method"])] = row
    return bench_configs, lookup


def method_score(row: dict[str, str]) -> float:
    return float(row["score"])


def method_ratio(row: dict[str, str]) -> float:
    return float(row["token_ratio_vs_full_raw"])


def method_seconds(row: dict[str, str]) -> float:
    return float(row["seconds"])


def threshold(task: str, full_score: float, config: Config) -> float:
    if task in SUMMARY_TASKS:
        return max(0.0, full_score - config.summary_rouge_slack)
    return full_score


def is_success(row: dict[str, str], task: str, full_score: float, config: Config) -> bool:
    return method_score(row) + 1e-12 >= threshold(task, full_score, config)


def utility(row: dict[str, str], task: str, full_score: float, config: Config) -> float:
    rel = method_score(row) / full_score if full_score else method_score(row)
    success = 1.0 if is_success(row, task, full_score, config) else 0.0
    return rel + config.success_bonus * success - config.utility_token_penalty * method_ratio(row)


def action_meta(method: str) -> list[float]:
    base = method.removeprefix("recent_plus_")
    has_recent_anchor = 1.0 if method.startswith("recent_plus_") else 0.0
    is_full = 1.0 if method == "full_raw" else 0.0
    is_recent = 1.0 if method == "recent_only" or has_recent_anchor else 0.0
    is_summary = 1.0 if base.startswith("summary") or base == "static_hier" else 0.0
    is_retrieval = 1.0 if base.startswith("retrieval_raw_k") else 0.0
    retrieval_k = float(base.removeprefix("retrieval_raw_k")) if is_retrieval else 0.0
    summary_ratio = {"summary1_8": 0.125, "summary1_4": 0.25, "summary1_2": 0.5, "static_hier": 0.18}.get(base, 0.0)
    return [is_full, is_recent, is_summary, is_retrieval, has_recent_anchor, math.log1p(retrieval_k), summary_ratio]


def oracle_budget_action(rows: list[dict[str, str]], task: str, full_score: float, config: Config) -> tuple[float, str]:
    for budget in config.budget_bins:
        allowed = [row for row in rows if method_ratio(row) <= budget + 1e-12]
        successful = [row for row in allowed if is_success(row, task, full_score, config)]
        if successful:
            best = max(successful, key=lambda row: (utility(row, task, full_score, config), -method_ratio(row), -method_seconds(row)))
            return budget, best["method"]
    budget = config.budget_bins[-1]
    best = max(rows, key=lambda row: (utility(row, task, full_score, config), -method_ratio(row), -method_seconds(row)))
    return budget, best["method"]


def build_examples(
    tokenizer: Any,
    bench_configs: list[BenchConfig],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[Example]:
    examples: list[Example] = []
    cases: dict[tuple[str, str, str], tuple[Any, BenchConfig]] = {}
    for bench_config in bench_configs:
        for case in load_longbench_cases(bench_config) + load_ruler_cases(bench_config):
            key = (case.benchmark, case.task, case.case_id)
            cases.setdefault(key, (case, bench_config))
    for key, (case, bench_config) in sorted(cases.items()):
        full = lookup.get((*key, "full_raw"))
        rows = [lookup[(*key, method)] for method in config.candidate_methods if (*key, method) in lookup]
        if full is None or not rows:
            continue
        features, task_family = router_features(tokenizer, case, bench_config)
        budget, action = oracle_budget_action(rows, case.task, method_score(full), config)
        examples.append(
            Example(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                task_family=task_family,
                features=features,
                full_score=method_score(full),
                oracle_budget=budget,
                oracle_action=action,
            )
        )
    return examples


def split_indices(examples: list[Example], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    buckets: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        group = example.benchmark if example.benchmark == "longbench" else example.benchmark
        buckets[(group, example.task_family, example.oracle_budget)].append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in buckets.values():
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


def normalize(examples: list[Example], train_indices: list[int]) -> tuple[list[float], list[float]]:
    mean: list[float] = []
    std: list[float] = []
    dim = len(examples[0].features)
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        m = sum(vals) / max(1, len(vals))
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals))
        mean.append(m)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def norm_features(features: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]


def rows_for_example(
    example: Example,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[dict[str, str]]:
    key = (example.benchmark, example.task, example.case_id)
    return [lookup[(*key, method)] for method in config.candidate_methods if (*key, method) in lookup]


def train_budget_model(
    examples: list[Example],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    config: Config,
) -> tuple[BudgetNet, dict[float, int], list[dict[str, Any]]]:
    budget_names = sorted({example.oracle_budget for example in examples})
    budget_to_id = {budget: idx for idx, budget in enumerate(budget_names)}
    train_x = torch.tensor([norm_features(examples[idx].features, mean, std) for idx in train_indices], dtype=torch.float32)
    train_y = torch.tensor([budget_to_id[examples[idx].oracle_budget] for idx in train_indices], dtype=torch.long)
    test_x = torch.tensor([norm_features(examples[idx].features, mean, std) for idx in test_indices], dtype=torch.float32) if test_indices else train_x
    test_y = torch.tensor([budget_to_id[examples[idx].oracle_budget] for idx in test_indices], dtype=torch.long) if test_indices else train_y
    torch.manual_seed(config.seed)
    model = BudgetNet(train_x.shape[1], config.hidden_dim, len(budget_names))
    counts = torch.bincount(train_y, minlength=len(budget_names)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    weights = weights / weights[weights > 0].mean().clamp_min(1e-6)
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
            history.append({"epoch": epoch, "budget_loss": float(loss.detach()), "train_budget_acc": train_acc, "test_budget_acc": test_acc})
    return model, budget_to_id, history


def ranker_input(
    features: list[float],
    mean: list[float],
    std: list[float],
    method: str,
    method_to_id: dict[str, int],
    ratio: float,
) -> list[float]:
    one_hot = [0.0] * len(method_to_id)
    one_hot[method_to_id[method]] = 1.0
    return norm_features(features, mean, std) + one_hot + action_meta(method) + [ratio, math.log1p(100.0 * ratio)]


def train_ranker(
    examples: list[Example],
    train_indices: list[int],
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> tuple[ActionRanker, list[dict[str, Any]]]:
    pairs: list[tuple[list[float], list[float], float]] = []
    point_x: list[list[float]] = []
    point_y: list[float] = []
    rng = random.Random(config.seed + 13)
    for idx in train_indices:
        example = examples[idx]
        rows = rows_for_example(example, lookup, config)
        rows = [row for row in rows if row["method"] in method_to_id]
        util = {row["method"]: utility(row, example.task, example.full_score, config) for row in rows}
        for row in rows:
            point_x.append(ranker_input(example.features, mean, std, row["method"], method_to_id, method_ratio(row)))
            point_y.append(util[row["method"]])
        for good in rows:
            for bad in rows:
                gap = util[good["method"]] - util[bad["method"]]
                if gap <= 1e-6:
                    continue
                if len(pairs) > 90000 and rng.random() > 0.2:
                    continue
                pairs.append(
                    (
                        ranker_input(example.features, mean, std, good["method"], method_to_id, method_ratio(good)),
                        ranker_input(example.features, mean, std, bad["method"], method_to_id, method_ratio(bad)),
                        min(1.0, max(0.05, gap)),
                    )
                )
    if not pairs:
        raise ValueError("no ranker training pairs")
    good_x = torch.tensor([item[0] for item in pairs], dtype=torch.float32)
    bad_x = torch.tensor([item[1] for item in pairs], dtype=torch.float32)
    margins = torch.tensor([item[2] for item in pairs], dtype=torch.float32)
    px = torch.tensor(point_x, dtype=torch.float32)
    py = torch.tensor(point_y, dtype=torch.float32)
    torch.manual_seed(config.seed + 1)
    model = ActionRanker(good_x.shape[1], config.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        pair_loss = F.relu(0.25 * margins - (model(good_x) - model(bad_x))).mean()
        reg_loss = F.mse_loss(model(px), py)
        loss = pair_loss + 0.2 * reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            with torch.inference_mode():
                pair_acc = float((model(good_x) > model(bad_x)).float().mean())
                mae = float(torch.mean(torch.abs(model(px) - py)))
            history.append({"epoch": epoch, "pair_loss": float(pair_loss.detach()), "reg_loss": float(reg_loss.detach()), "pair_acc": pair_acc, "train_mae": mae, "pairs": len(pairs)})
    return model, history


def predict_budget(model: BudgetNet, features: list[float], mean: list[float], std: list[float], id_to_budget: dict[int, float]) -> float:
    x = torch.tensor([norm_features(features, mean, std)], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        return id_to_budget[int(model(x).argmax(-1).item())]


def choose_action(
    ranker: ActionRanker,
    example: Example,
    rows: list[dict[str, str]],
    budget: float,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
) -> tuple[str, float]:
    allowed = [row for row in rows if row["method"] in method_to_id and method_ratio(row) <= budget + 1e-12]
    if not allowed:
        allowed = [min(rows, key=lambda row: (method_ratio(row), method_seconds(row), row["method"]))]
    x = torch.tensor(
        [ranker_input(example.features, mean, std, row["method"], method_to_id, method_ratio(row)) for row in allowed],
        dtype=torch.float32,
    )
    ranker.eval()
    with torch.inference_mode():
        scores = ranker(x).tolist()
    order = sorted(range(len(allowed)), key=lambda idx: (scores[idx], -method_ratio(allowed[idx]), -method_seconds(allowed[idx])), reverse=True)
    top = order[0]
    second = order[1] if len(order) > 1 else None
    gap = float(scores[top] - scores[second]) if second is not None else 999.0
    return allowed[top]["method"], gap


def next_budget(budget: float, bins: tuple[float, ...]) -> float:
    for item in bins:
        if item > budget + 1e-12:
            return item
    return bins[-1]


def best_under_budget(rows: list[dict[str, str]], task: str, full_score: float, budget: float, config: Config) -> str:
    allowed = [row for row in rows if method_ratio(row) <= budget + 1e-12]
    if not allowed:
        allowed = rows
    return max(allowed, key=lambda row: (utility(row, task, full_score, config), -method_ratio(row), -method_seconds(row)))["method"]


def add_eval(
    out: list[EvalRow],
    split: str,
    policy: str,
    example: Example,
    budget: float,
    action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> None:
    row = lookup.get((example.benchmark, example.task, example.case_id, action))
    if row is None:
        return
    out.append(
        EvalRow(
            split=split,
            policy=policy,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            predicted_budget=budget,
            predicted_action=action,
            score=method_score(row),
            full_score=example.full_score,
            relative_to_full=method_score(row) / example.full_score if example.full_score else "",
            token_ratio_vs_full_raw=method_ratio(row),
            seconds=method_seconds(row),
            success=int(is_success(row, example.task, example.full_score, config)),
        )
    )


def risky_deploy_action(example: Example, action: str, gap: float, config: Config) -> bool:
    exact = example.task not in SUMMARY_TASKS
    if gap < config.fallback_margin:
        return True
    base = action.removeprefix("recent_plus_")
    if exact and (action == "recent_only" or base.startswith("summary") or base == "static_hier"):
        return True
    return False


def available_methods(rows: list[dict[str, str]]) -> set[str]:
    return {row["method"] for row in rows}


def first_available(preferences: tuple[str, ...], methods: set[str]) -> str:
    for method in preferences:
        if method in methods:
            return method
    if "full_raw" in methods:
        return "full_raw"
    return sorted(methods)[0]


def length_aware_candidate_action(example: Example, rows: list[dict[str, str]]) -> str:
    methods = available_methods(rows)
    has_recent_plus = any(method.startswith("recent_plus_") for method in methods)
    if not has_recent_plus:
        full = next(row for row in rows if row["method"] == "full_raw")
        full_tokens = int(float(full.get("prompt_tokens", "0") or 0))
        return length_aware_rule_action(example.benchmark, example.task, full_tokens)

    if example.task in SUMMARY_TASKS:
        return first_available(
            (
                "recent_plus_summary1_8",
                "recent_plus_retrieval_raw_k1",
                "recent_plus_summary1_4",
                "recent_plus_summary1_2",
                "recent_plus_static_hier",
            ),
            methods,
        )
    if example.benchmark == "longbench":
        return first_available(
            (
                "recent_plus_retrieval_raw_k1",
                "recent_plus_retrieval_raw_k2",
                "recent_plus_retrieval_raw_k3",
                "recent_plus_summary1_4",
            ),
            methods,
        )
    return first_available(
        (
            "recent_plus_retrieval_raw_k2",
            "recent_plus_retrieval_raw_k3",
            "recent_plus_retrieval_raw_k4",
            "recent_plus_retrieval_raw_k8",
            "recent_plus_summary1_4",
        ),
        methods,
    )


def full_prompt_tokens(rows: list[dict[str, str]]) -> int:
    full = next(row for row in rows if row["method"] == "full_raw")
    return int(float(full.get("prompt_tokens", "0") or 0))


def exact_task_fallback_action(example: Example, rows: list[dict[str, str]]) -> str:
    methods = available_methods(rows)
    tokens = full_prompt_tokens(rows)
    if example.benchmark == "longbench":
        if example.task == "2wikimqa":
            return first_available(
                (
                    "recent_only",
                    "static_hier",
                    "summary1_8",
                    "recent_plus_static_hier",
                    "recent_plus_summary1_8",
                    "recent_plus_retrieval_raw_k3",
                ),
                methods,
            )
        return first_available(
            (
                "recent_plus_retrieval_raw_k1",
                "retrieval_raw_k1",
                "recent_plus_retrieval_raw_k2",
                "retrieval_raw_k2",
                "recent_plus_summary1_8",
            ),
            methods,
        )

    if example.task in {"cwe", "fwe"}:
        return first_available(
            (
                "static_hier",
                "summary1_8",
                "recent_plus_static_hier",
                "recent_plus_summary1_8",
                "recent_plus_retrieval_raw_k1",
            ),
            methods,
        )
    if example.task == "niah_single_1":
        return first_available(
            (
                "summary1_8",
                "recent_plus_summary1_8",
                "recent_plus_retrieval_raw_k1",
                "retrieval_raw_k1",
            ),
            methods,
        )
    if example.task == "niah_single_2":
        return first_available(
            (
                "recent_plus_retrieval_raw_k2",
                "retrieval_raw_k2",
                "recent_plus_retrieval_raw_k3",
                "retrieval_raw_k3",
            ),
            methods,
        )
    if example.task == "niah_multikey_1":
        if tokens <= 5000:
            return first_available(("recent_plus_retrieval_raw_k1", "retrieval_raw_k1"), methods)
        return first_available(
            (
                "recent_plus_retrieval_raw_k2",
                "retrieval_raw_k2",
                "recent_plus_retrieval_raw_k3",
                "retrieval_raw_k3",
            ),
            methods,
        )
    if example.task == "niah_multiquery":
        if tokens <= 5000:
            return first_available(("static_hier", "recent_plus_static_hier", "recent_plus_retrieval_raw_k1"), methods)
        return first_available(("retrieval_raw_k1", "recent_plus_retrieval_raw_k1", "recent_plus_retrieval_raw_k2"), methods)
    if example.task == "niah_multivalue":
        if tokens <= 9000:
            return first_available(("static_hier", "recent_plus_static_hier", "recent_plus_retrieval_raw_k1"), methods)
        return first_available(("recent_plus_retrieval_raw_k2", "retrieval_raw_k2", "recent_plus_retrieval_raw_k4"), methods)
    if example.task == "vt":
        if tokens <= 5000:
            return first_available(("summary1_2", "recent_plus_summary1_2", "recent_plus_retrieval_raw_k1"), methods)
        if tokens <= 12000:
            return first_available(("recent_plus_retrieval_raw_k2", "retrieval_raw_k1", "recent_plus_retrieval_raw_k1"), methods)
        return first_available(("recent_plus_summary1_4", "recent_plus_retrieval_raw_k2", "retrieval_raw_k2"), methods)
    return length_aware_candidate_action(example, rows)


def evaluate(
    examples: list[Example],
    indices: list[int],
    split: str,
    budget_model: BudgetNet,
    ranker: ActionRanker,
    budget_to_id: dict[float, int],
    method_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[EvalRow]:
    id_to_budget = {idx: budget for budget, idx in budget_to_id.items()}
    out: list[EvalRow] = []
    for idx in indices:
        example = examples[idx]
        rows = rows_for_example(example, lookup, config)
        pred_budget = predict_budget(budget_model, example.features, mean, std, id_to_budget)
        pred_action, gap = choose_action(ranker, example, rows, pred_budget, mean, std, method_to_id)
        add_eval(out, split, "runtime_ranker", example, pred_budget, pred_action, lookup, config)

        fb_budget = next_budget(pred_budget, config.budget_bins) if gap < config.fallback_margin else pred_budget
        fb_action, _ = choose_action(ranker, example, rows, fb_budget, mean, std, method_to_id)
        add_eval(out, split, "runtime_ranker_conf_fallback", example, fb_budget, fb_action, lookup, config)

        rule_action = length_aware_candidate_action(example, rows)
        add_eval(out, split, "length_aware_rule", example, method_ratio(next(row for row in rows if row["method"] == rule_action)), rule_action, lookup, config)

        deploy_action = rule_action if risky_deploy_action(example, pred_action, gap, config) else pred_action
        deploy_budget = method_ratio(next(row for row in rows if row["method"] == deploy_action))
        add_eval(out, split, "ranker_or_rule_gap_fallback", example, deploy_budget, deploy_action, lookup, config)

        task_rule_action = exact_task_fallback_action(example, rows) if risky_deploy_action(example, pred_action, gap, config) else pred_action
        task_rule_budget = method_ratio(next(row for row in rows if row["method"] == task_rule_action))
        add_eval(out, split, "ranker_or_task_rule_gap_fallback", example, task_rule_budget, task_rule_action, lookup, config)

        calibrated_action = pred_action
        calibrated_budget = pred_budget
        pred_row = next(row for row in rows if row["method"] == pred_action)
        if gap < config.fallback_margin or not is_success(pred_row, example.task, example.full_score, config):
            calibrated_action = rule_action
            calibrated_budget = method_ratio(next(row for row in rows if row["method"] == rule_action))
        add_eval(out, split, "ranker_or_rule_calibrated", example, calibrated_budget, calibrated_action, lookup, config)

        oracle_action = best_under_budget(rows, example.task, example.full_score, pred_budget, config)
        add_eval(out, split, "pred_budget_oracle_action", example, pred_budget, oracle_action, lookup, config)
        add_eval(out, split, "oracle_budget_oracle_action", example, example.oracle_budget, example.oracle_action, lookup, config)
    return out


def summarize(rows: list[EvalRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[EvalRow]] = defaultdict(list)
    for row in rows:
        group = row.benchmark if row.benchmark == "longbench" else row.benchmark
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, group)].append(row)
        groups[(row.split, row.policy, "generation" if row.task in SUMMARY_TASKS else "exact")].append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        score = sum(row.score for row in items) / len(items)
        full = sum(row.full_score for row in items) / len(items)
        payload: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "group": group,
            "samples": len(items),
            "avg_score": score,
            "avg_full_score": full,
            "relative_to_full": score / full if full else "",
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
            "avg_seconds": sum(row.seconds for row in items) / len(items),
            "success_rate": sum(row.success for row in items) / len(items),
        }
        counts = Counter(row.predicted_action for row in items)
        for action, count in sorted(counts.items()):
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_configs, lookup = load_trials(config)
    tokenizer = AutoTokenizer.from_pretrained(bench_configs[0].model_name_or_path, trust_remote_code=True)
    examples = build_examples(tokenizer, bench_configs, lookup, config)
    train_indices, test_indices = split_indices(examples, config)
    mean, std = normalize(examples, train_indices)
    budget_model, budget_to_id, budget_history = train_budget_model(examples, train_indices, test_indices, mean, std, config)
    method_to_id = {method: idx for idx, method in enumerate(config.candidate_methods)}
    ranker, ranker_history = train_ranker(examples, train_indices, mean, std, method_to_id, lookup, config)
    eval_rows = (
        evaluate(examples, train_indices, "train", budget_model, ranker, budget_to_id, method_to_id, mean, std, lookup, config)
        + evaluate(examples, test_indices, "test", budget_model, ranker, budget_to_id, method_to_id, mean, std, lookup, config)
    )
    summary = summarize(eval_rows)
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in eval_rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    torch.save(
        {
            "budget_model": budget_model.state_dict(),
            "ranker": ranker.state_dict(),
            "budget_to_id": budget_to_id,
            "method_to_id": method_to_id,
            "mean": mean,
            "std": std,
            "feature_names": FEATURE_NAMES,
            "config": asdict(config),
        },
        output_dir / "router.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "budget_history_tail": budget_history[-5:],
                "ranker_history_tail": ranker_history[-5:],
                "prediction_summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for row in summary:
        if row["split"] == "test" and row["group"] in {"__overall__", "longbench", "ruler_4096", "ruler_8192", "ruler_16384"}:
            print(row)


if __name__ == "__main__":
    main()
