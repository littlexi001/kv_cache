from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
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
    parse_int_tuple,
    router_features,
)
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES, write_csv  # noqa: E402
from run_synthetic_router_distillation import (  # noqa: E402
    Config as SyntheticConfig,
    SyntheticTrial,
    as_bench_case,
    bench_config as synthetic_bench_config,
    build_synthetic_cases,
    build_trials,
    load_text_ids,
)
from run_router_policy_offline_eval import length_aware_rule_action  # noqa: E402


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    benchmark_output_dirs: tuple[str, ...]
    candidate_methods: tuple[str, ...]
    budget_bins: tuple[float, ...]
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
    margin: float
    fallback_margin: float
    token_penalty: float
    summary_rouge_slack: float
    seed: int


@dataclass
class PairwiseExample:
    case_id: str
    task_family: str
    kind: str
    oracle_budget: float
    features: list[float]


@dataclass
class EvalRow:
    split: str
    policy: str
    benchmark: str
    task: str
    case_id: str
    task_family: str
    predicted_budget: float
    predicted_action: str
    score: float
    full_score: float
    relative_to_full: float | str
    success: int | str
    token_ratio_vs_full_raw: float
    seconds: float


class BudgetClassifier(nn.Module):
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


class PairwiseActionRanker(nn.Module):
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
    parser = argparse.ArgumentParser(description="Two-stage budget router with synthetic pairwise stage-2 ranker.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument(
        "--candidate_methods",
        default=(
            "full_raw,recent_only,static_hier,summary1_8,summary1_4,summary1_2,"
            "retrieval_raw_k1,retrieval_raw_k2,retrieval_raw_k3,retrieval_raw_k4,retrieval_raw_k8"
        ),
    )
    parser.add_argument("--budget_bins", default="0.2,0.3,0.4,0.5,1.0")
    parser.add_argument("--cases_per_dataset", type=int, default=420)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--prefill_token_lengths", default="4096,8192,16384")
    parser.add_argument("--sample_stride_tokens", type=int, default=512)
    parser.add_argument("--eval_start_tokens", type=int, default=20000)
    parser.add_argument("--block_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_text_tokens", type=int, default=280000)
    parser.add_argument("--max_input_tokens", type=int, default=24000)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--fallback_margin", type=float, default=0.12)
    parser.add_argument("--token_penalty", type=float, default=0.25)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026070522)
    args = parser.parse_args()
    text_paths = parse_csv_tuple(args.text_paths)
    dataset_names = parse_csv_tuple(args.dataset_names)
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(
        **{
            **vars(args),
            "text_paths": text_paths,
            "dataset_names": dataset_names,
            "benchmark_output_dirs": parse_csv_tuple(args.benchmark_output_dirs),
            "candidate_methods": parse_csv_tuple(args.candidate_methods),
            "budget_bins": tuple(sorted(parse_float_tuple(args.budget_bins))),
            "prefill_token_lengths": parse_int_tuple(args.prefill_token_lengths),
        }
    )


def to_synthetic_config(config: Config) -> SyntheticConfig:
    return SyntheticConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        benchmark_output_dir="",
        candidate_methods=config.candidate_methods,
        cases_per_dataset=config.cases_per_dataset,
        prefill_tokens=config.prefill_tokens,
        prefill_token_lengths=config.prefill_token_lengths,
        sample_stride_tokens=config.sample_stride_tokens,
        eval_start_tokens=config.eval_start_tokens,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        max_text_tokens=config.max_text_tokens,
        max_input_tokens=config.max_input_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        test_fraction=config.test_fraction,
        label_mode="length_aware",
        seed=config.seed,
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


def load_benchmark_trials(config: Config) -> tuple[BenchConfig, dict[tuple[str, str, str, str], dict[str, str]]]:
    dirs = [Path(item) for item in config.benchmark_output_dirs]
    bench_config = bench_config_from_summary(dirs[0])
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for directory in dirs:
        for row in read_csv(directory / "trials.csv"):
            lookup[(row["benchmark"], row["task"], row["case_id"], row["method"])] = row
    return bench_config, lookup


def split_indices(examples: list[PairwiseExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        groups[(example.kind, example.oracle_budget)].append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in groups.values():
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


def normalize(examples: list[PairwiseExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        mean.append(float(statistics.mean(vals)))
        std.append(float(statistics.pstdev(vals) or 1.0))
    return mean, std


def norm_features(features: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]


def action_meta(method: str) -> list[float]:
    is_full = 1.0 if method == "full_raw" else 0.0
    is_recent = 1.0 if method == "recent_only" else 0.0
    is_summary = 1.0 if method.startswith("summary") or method == "static_hier" else 0.0
    is_retrieval = 1.0 if method.startswith("retrieval_raw_k") else 0.0
    retrieval_k = float(method.removeprefix("retrieval_raw_k")) if is_retrieval else 0.0
    summary_ratio = {
        "summary1_8": 0.125,
        "summary1_4": 0.25,
        "summary1_2": 0.5,
        "static_hier": 0.18,
    }.get(method, 0.0)
    return [is_full, is_recent, is_summary, is_retrieval, math.log1p(retrieval_k), summary_ratio]


def ranker_input(
    features: list[float],
    mean: list[float],
    std: list[float],
    method: str,
    method_to_id: dict[str, int],
    token_ratio: float,
) -> list[float]:
    one_hot = [0.0] * len(method_to_id)
    one_hot[method_to_id[method]] = 1.0
    return norm_features(features, mean, std) + one_hot + action_meta(method) + [token_ratio, math.log1p(token_ratio * 100.0)]


def synthetic_case_groups(trials: list[SyntheticTrial]) -> dict[str, list[SyntheticTrial]]:
    groups: dict[str, list[SyntheticTrial]] = defaultdict(list)
    for row in trials:
        groups[row.case_id].append(row)
    return groups


def min_success_budget(rows: list[SyntheticTrial], budget_bins: tuple[float, ...]) -> float:
    successful = [row for row in rows if row.success]
    if not successful:
        return budget_bins[-1]
    for budget in budget_bins:
        if any(row.token_ratio_vs_full_raw <= budget + 1e-12 for row in successful):
            return budget
    return budget_bins[-1]


def train_budget_classifier(
    examples: list[PairwiseExample],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    config: Config,
) -> tuple[BudgetClassifier, dict[float, int], list[dict[str, Any]]]:
    budget_names = sorted({example.oracle_budget for example in examples})
    budget_to_id = {budget: idx for idx, budget in enumerate(budget_names)}
    train_x = torch.tensor([norm_features(examples[idx].features, mean, std) for idx in train_indices], dtype=torch.float32)
    train_y = torch.tensor([budget_to_id[examples[idx].oracle_budget] for idx in train_indices], dtype=torch.long)
    test_x = torch.tensor([norm_features(examples[idx].features, mean, std) for idx in test_indices], dtype=torch.float32) if test_indices else train_x
    test_y = torch.tensor([budget_to_id[examples[idx].oracle_budget] for idx in test_indices], dtype=torch.long) if test_indices else train_y

    torch.manual_seed(config.seed)
    model = BudgetClassifier(train_x.shape[1], config.hidden_dim, len(budget_names))
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


def utility(row: SyntheticTrial, config: Config) -> float:
    return (1.0 if row.success else 0.0) - config.token_penalty * row.token_ratio_vs_full_raw


def train_pairwise_ranker(
    examples: list[PairwiseExample],
    trials: list[SyntheticTrial],
    train_indices: list[int],
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    config: Config,
) -> tuple[PairwiseActionRanker, list[dict[str, Any]]]:
    example_by_id = {example.case_id: example for example in examples}
    train_case_ids = {examples[idx].case_id for idx in train_indices}
    groups = synthetic_case_groups(trials)
    pairs: list[tuple[list[float], list[float], float]] = []
    point_x: list[list[float]] = []
    point_y: list[float] = []
    rng = random.Random(config.seed + 11)
    for case_id in train_case_ids:
        example = example_by_id[case_id]
        rows = [row for row in groups[case_id] if row.method in method_to_id]
        for row in rows:
            point_x.append(ranker_input(example.features, mean, std, row.method, method_to_id, row.token_ratio_vs_full_raw))
            point_y.append(utility(row, config))
        ordered = sorted(rows, key=lambda row: utility(row, config), reverse=True)
        for good in ordered:
            for bad in ordered:
                if utility(good, config) <= utility(bad, config) + 1e-6:
                    continue
                if len(pairs) > 60000 and rng.random() > 0.15:
                    continue
                pairs.append(
                    (
                        ranker_input(example.features, mean, std, good.method, method_to_id, good.token_ratio_vs_full_raw),
                        ranker_input(example.features, mean, std, bad.method, method_to_id, bad.token_ratio_vs_full_raw),
                        min(1.0, max(0.05, utility(good, config) - utility(bad, config))),
                    )
                )
    if not pairs:
        raise ValueError("no pairwise training pairs were built")
    good_x = torch.tensor([item[0] for item in pairs], dtype=torch.float32)
    bad_x = torch.tensor([item[1] for item in pairs], dtype=torch.float32)
    margins = torch.tensor([item[2] for item in pairs], dtype=torch.float32)
    px = torch.tensor(point_x, dtype=torch.float32)
    py = torch.tensor(point_y, dtype=torch.float32)
    torch.manual_seed(config.seed + 1)
    model = PairwiseActionRanker(good_x.shape[1], config.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        good_score = model(good_x)
        bad_score = model(bad_x)
        pair_loss = F.relu(config.margin * margins - (good_score - bad_score)).mean()
        reg_loss = F.mse_loss(model(px), py)
        loss = pair_loss + 0.25 * reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            with torch.inference_mode():
                pair_acc = float((model(good_x) > model(bad_x)).float().mean())
                mae = float(torch.mean(torch.abs(model(px) - py)))
            history.append({"epoch": epoch, "pair_loss": float(pair_loss.detach()), "reg_loss": float(reg_loss.detach()), "pair_acc": pair_acc, "train_mae": mae, "pairs": len(pairs)})
    return model, history


def predict_budget(model: BudgetClassifier, features: list[float], mean: list[float], std: list[float], id_to_budget: dict[int, float]) -> float:
    x = torch.tensor([norm_features(features, mean, std)], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        idx = int(model(x).argmax(-1).item())
    return id_to_budget[idx]


def score_actions(
    ranker: PairwiseActionRanker,
    features: list[float],
    rows: list[Any],
    get_method: Any,
    get_ratio: Any,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
) -> list[tuple[float, Any]]:
    xs = [ranker_input(features, mean, std, get_method(row), method_to_id, get_ratio(row)) for row in rows]
    x = torch.tensor(xs, dtype=torch.float32)
    ranker.eval()
    with torch.inference_mode():
        scores = ranker(x).tolist()
    return [(float(score), row) for score, row in zip(scores, rows)]


def choose_by_ranker(
    ranker: PairwiseActionRanker,
    features: list[float],
    rows: list[Any],
    budget: float,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    get_method: Any,
    get_ratio: Any,
) -> tuple[str, float, float]:
    allowed = [row for row in rows if get_method(row) in method_to_id and get_ratio(row) <= budget + 1e-12]
    if not allowed:
        allowed = [min(rows, key=lambda row: (get_ratio(row), get_method(row)))]
    scored = score_actions(ranker, features, allowed, get_method, get_ratio, mean, std, method_to_id)
    scored.sort(key=lambda item: (item[0], -get_ratio(item[1])), reverse=True)
    top = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else top[0] - 999.0
    return get_method(top[1]), top[0], top[0] - second_score


def next_budget(budget: float, bins: tuple[float, ...]) -> float:
    for item in bins:
        if item > budget + 1e-12:
            return item
    return bins[-1]


def conservative_action(action: str, rows: list[dict[str, str]], budget: float) -> str:
    available = {row["method"]: row for row in rows}
    def allowed(method: str) -> bool:
        row = available.get(method)
        return row is not None and float(row["token_ratio_vs_full_raw"]) <= budget + 1e-12

    if action == "retrieval_raw_k1":
        for method in ("retrieval_raw_k2", "retrieval_raw_k3", "retrieval_raw_k4", "retrieval_raw_k8"):
            if allowed(method):
                return method
    if action == "retrieval_raw_k2":
        for method in ("retrieval_raw_k3", "retrieval_raw_k4", "retrieval_raw_k8"):
            if allowed(method):
                return method
    if action == "summary1_8":
        for method in ("summary1_4", "summary1_2"):
            if allowed(method):
                return method
    return action


def successful_under_budget_benchmark(rows: list[dict[str, str]], budget: float, full_score: float, task: str, config: Config) -> str:
    threshold = max(0.0, full_score - config.summary_rouge_slack) if task in SUMMARY_TASKS else full_score
    allowed = [row for row in rows if float(row["token_ratio_vs_full_raw"]) <= budget + 1e-12]
    successful = [row for row in allowed if float(row["score"]) + 1e-12 >= threshold]
    if successful:
        return min(successful, key=lambda row: (float(row["token_ratio_vs_full_raw"]), float(row["seconds"]), row["method"]))["method"]
    if allowed:
        best = max(float(row["score"]) for row in allowed)
        return min([row for row in allowed if abs(float(row["score"]) - best) < 1e-12], key=lambda row: (float(row["token_ratio_vs_full_raw"]), row["method"]))["method"]
    return min(rows, key=lambda row: float(row["token_ratio_vs_full_raw"]))["method"]


def add_benchmark_eval(
    out: list[EvalRow],
    split: str,
    policy: str,
    case: Any,
    budget: float,
    action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    row = lookup.get((case.benchmark, case.task, case.case_id, action))
    full = lookup.get((case.benchmark, case.task, case.case_id, "full_raw"))
    if row is None or full is None:
        return
    score = float(row["score"])
    full_score = float(full["score"])
    out.append(
        EvalRow(
            split=split,
            policy=policy,
            benchmark=case.benchmark,
            task=case.task,
            case_id=case.case_id,
            task_family="generation" if case.task in SUMMARY_TASKS else "exact",
            predicted_budget=budget,
            predicted_action=action,
            score=score,
            full_score=full_score,
            relative_to_full=score / full_score if full_score else "",
            success="",
            token_ratio_vs_full_raw=float(row["token_ratio_vs_full_raw"]),
            seconds=float(row["seconds"]),
        )
    )


def risky_pairwise_action(action: str, case: Any, gap: float, config: Config) -> bool:
    exact = case.benchmark.startswith("ruler") or case.task not in SUMMARY_TASKS
    if gap < config.fallback_margin:
        return True
    if exact and action in {"recent_only", "summary1_8", "summary1_4", "summary1_2", "static_hier"}:
        return True
    if exact and action == "full_raw":
        return True
    return False


def evaluate_benchmark(
    budget_model: BudgetClassifier,
    ranker: PairwiseActionRanker,
    bench_config: BenchConfig,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    tokenizer: Any,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    id_to_budget: dict[int, float],
    config: Config,
) -> list[EvalRow]:
    cases = load_longbench_cases(bench_config) + load_ruler_cases(bench_config)
    rows_out: list[EvalRow] = []
    for case in cases:
        features, _ = router_features(tokenizer, case, bench_config)
        case_rows = [lookup[(case.benchmark, case.task, case.case_id, method)] for method in config.candidate_methods if (case.benchmark, case.task, case.case_id, method) in lookup]
        full = lookup.get((case.benchmark, case.task, case.case_id, "full_raw"))
        if not case_rows or full is None:
            continue
        budget = predict_budget(budget_model, features, mean, std, id_to_budget)
        action, _, gap = choose_by_ranker(
            ranker,
            features,
            case_rows,
            budget,
            mean,
            std,
            method_to_id,
            lambda row: row["method"],
            lambda row: float(row["token_ratio_vs_full_raw"]),
        )
        add_benchmark_eval(rows_out, "heldout", "runtime_pairwise", case, budget, action, lookup)
        fallback_budget = next_budget(budget, config.budget_bins) if gap < config.fallback_margin else budget
        fallback_action, _, _ = choose_by_ranker(
            ranker,
            features,
            case_rows,
            fallback_budget,
            mean,
            std,
            method_to_id,
            lambda row: row["method"],
            lambda row: float(row["token_ratio_vs_full_raw"]),
        )
        if gap < config.fallback_margin:
            fallback_action = conservative_action(fallback_action, case_rows, fallback_budget)
        add_benchmark_eval(rows_out, "heldout", "runtime_pairwise_fallback", case, fallback_budget, fallback_action, lookup)
        full_tokens = int(full.get("prompt_tokens", "0") or 0)
        lenaware_action = length_aware_rule_action(case.benchmark, case.task, full_tokens)
        blend_action = lenaware_action if risky_pairwise_action(action, case, gap, config) else action
        blend_budget = float(next((row["token_ratio_vs_full_raw"] for row in case_rows if row["method"] == blend_action), budget))
        add_benchmark_eval(rows_out, "heldout", "runtime_pairwise_lenaware_fallback", case, blend_budget, blend_action, lookup)
        oracle_action = successful_under_budget_benchmark(case_rows, budget, float(full["score"]), case.task, config)
        add_benchmark_eval(rows_out, "heldout", "pred_budget_oracle_action", case, budget, oracle_action, lookup)
    return rows_out


def evaluate_synthetic(
    budget_model: BudgetClassifier,
    ranker: PairwiseActionRanker,
    examples: list[PairwiseExample],
    trials: list[SyntheticTrial],
    indices: list[int],
    split: str,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    id_to_budget: dict[int, float],
    config: Config,
) -> list[EvalRow]:
    groups = synthetic_case_groups(trials)
    rows_out: list[EvalRow] = []
    for idx in indices:
        example = examples[idx]
        case_rows = groups[example.case_id]
        budget = predict_budget(budget_model, example.features, mean, std, id_to_budget)
        action, _, gap = choose_by_ranker(
            ranker,
            example.features,
            case_rows,
            budget,
            mean,
            std,
            method_to_id,
            lambda row: row.method,
            lambda row: row.token_ratio_vs_full_raw,
        )
        selected = next(row for row in case_rows if row.method == action)
        rows_out.append(
            EvalRow(
                split=split,
                policy="runtime_pairwise",
                benchmark="synthetic",
                task=example.kind,
                case_id=example.case_id,
                task_family=example.task_family,
                predicted_budget=budget,
                predicted_action=action,
                score=float(selected.success),
                full_score=1.0,
                relative_to_full=float(selected.success),
                success=int(selected.success),
                token_ratio_vs_full_raw=selected.token_ratio_vs_full_raw,
                seconds=0.0,
            )
        )
        fallback_budget = next_budget(budget, config.budget_bins) if gap < config.fallback_margin else budget
        fallback_action, _, _ = choose_by_ranker(
            ranker,
            example.features,
            case_rows,
            fallback_budget,
            mean,
            std,
            method_to_id,
            lambda row: row.method,
            lambda row: row.token_ratio_vs_full_raw,
        )
        selected_fb = next(row for row in case_rows if row.method == fallback_action)
        rows_out.append(
            EvalRow(
                split=split,
                policy="runtime_pairwise_fallback",
                benchmark="synthetic",
                task=example.kind,
                case_id=example.case_id,
                task_family=example.task_family,
                predicted_budget=fallback_budget,
                predicted_action=fallback_action,
                score=float(selected_fb.success),
                full_score=1.0,
                relative_to_full=float(selected_fb.success),
                success=int(selected_fb.success),
                token_ratio_vs_full_raw=selected_fb.token_ratio_vs_full_raw,
                seconds=0.0,
            )
        )
    return rows_out


def summarize(rows: list[EvalRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[EvalRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, row.task_family)].append(row)
        if row.benchmark != "synthetic":
            groups[(row.split, row.policy, row.benchmark)].append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        score = sum(float(row.score) for row in items) / len(items)
        full = sum(float(row.full_score) for row in items) / len(items)
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
        }
        successes = [row.success for row in items if isinstance(row.success, int)]
        if successes:
            payload["success_rate"] = sum(successes) / len(successes)
        action_counts = Counter(row.predicted_action for row in items)
        budget_counts = Counter(row.predicted_budget for row in items)
        for budget, count in sorted(budget_counts.items()):
            payload[f"select_budget_{budget:g}_rate"] = count / len(items)
        for action, count in sorted(action_counts.items()):
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    synthetic_config = to_synthetic_config(config)
    token_ids = load_text_ids(tokenizer, synthetic_config)
    cases = build_synthetic_cases(tokenizer, token_ids, synthetic_config)
    trials, router_examples = build_trials(tokenizer, cases, synthetic_config)
    trial_groups = synthetic_case_groups(trials)
    examples: list[PairwiseExample] = []
    synth_cfg = synthetic_bench_config(synthetic_config)
    case_by_id = {case.case_id: case for case in cases}
    for rex in router_examples:
        examples.append(
            PairwiseExample(
                case_id=rex.case_id,
                task_family=rex.task_family,
                kind=rex.kind,
                oracle_budget=min_success_budget(trial_groups[rex.case_id], config.budget_bins),
                features=router_features(tokenizer, as_bench_case(case_by_id[rex.case_id]), synth_cfg)[0],
            )
        )
    train_indices, test_indices = split_indices(examples, config)
    mean, std = normalize(examples, train_indices)
    budget_model, budget_to_id, budget_history = train_budget_classifier(examples, train_indices, test_indices, mean, std, config)
    method_to_id = {method: idx for idx, method in enumerate(config.candidate_methods)}
    ranker, ranker_history = train_pairwise_ranker(examples, trials, train_indices, mean, std, method_to_id, config)
    id_to_budget = {idx: budget for budget, idx in budget_to_id.items()}

    eval_rows: list[EvalRow] = []
    eval_rows.extend(evaluate_synthetic(budget_model, ranker, examples, trials, train_indices, "synthetic_train", mean, std, method_to_id, id_to_budget, config))
    eval_rows.extend(evaluate_synthetic(budget_model, ranker, examples, trials, test_indices, "synthetic_test", mean, std, method_to_id, id_to_budget, config))
    bench_config, benchmark_lookup = load_benchmark_trials(config)
    eval_rows.extend(evaluate_benchmark(budget_model, ranker, bench_config, benchmark_lookup, tokenizer, mean, std, method_to_id, id_to_budget, config))
    summary = summarize(eval_rows)

    torch.save(
        {
            "state_dict": budget_model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "budget_names": [id_to_budget[idx] for idx in range(len(id_to_budget))],
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "budget_router.pt",
    )
    torch.save(
        {
            "state_dict": ranker.state_dict(),
            "input_dim": len(FEATURE_NAMES) + len(method_to_id) + 8,
            "hidden_dim": config.hidden_dim,
            "method_names": config.candidate_methods,
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "pairwise_action_ranker.pt",
    )
    write_csv(output_dir / "synthetic_cases.csv", [asdict(row) for row in cases])
    write_csv(output_dir / "synthetic_trials.csv", [asdict(row) for row in trials])
    write_csv(output_dir / "pairwise_examples.csv", [asdict(row) for row in examples])
    write_csv(output_dir / "eval_rows.csv", [asdict(row) for row in eval_rows])
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "budget_history.csv", budget_history)
    write_csv(output_dir / "ranker_history.csv", ranker_history)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "synthetic_cases": len(cases),
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "budget_history_tail": budget_history[-5:],
                "ranker_history_tail": ranker_history[-5:],
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("split,policy,group,samples,avg_score,relative_to_full,success_rate,avg_token_ratio")
    for row in summary:
        success = row.get("success_rate", "")
        success_text = f"{success:.4f}" if isinstance(success, float) else ""
        rel = row["relative_to_full"]
        rel_text = f"{rel:.4f}" if isinstance(rel, float) else ""
        print(
            f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
            f"{row['avg_score']:.4f},{rel_text},{success_text},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print(f"saved budget router to {output_dir / 'budget_router.pt'}")
    print(f"saved pairwise ranker to {output_dir / 'pairwise_action_ranker.pt'}")


if __name__ == "__main__":
    main()
