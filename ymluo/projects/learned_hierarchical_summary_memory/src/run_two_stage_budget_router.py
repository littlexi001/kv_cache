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
    budget_loss_weight: float
    action_loss_weight: float
    summary_rouge_slack: float
    seed: int


@dataclass
class BudgetExample:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    oracle_budget: float
    oracle_action: str
    full_score: float
    features: list[float]


@dataclass
class PredictionRow:
    split: str
    policy: str
    benchmark: str
    task: str
    case_id: str
    task_family: str
    oracle_budget: float
    predicted_budget: float
    oracle_action: str
    predicted_action: str
    budget_correct: int
    action_correct: int
    score: float
    full_score: float
    relative_to_full: float | str
    token_ratio_vs_full_raw: float
    seconds: float


class TwoStageBudgetRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, budget_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.budget_head = nn.Linear(hidden_dim, budget_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.budget_head(hidden), self.action_head(hidden)


class ActionScoreRanker(nn.Module):
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
    parser = argparse.ArgumentParser(description="Train/evaluate a two-stage budget router from precomputed trials.")
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
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--budget_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026070508)
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
        budget_loss_weight=args.budget_loss_weight,
        action_loss_weight=args.action_loss_weight,
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


def case_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["benchmark"], row["task"], row["case_id"])


def method_score(row: dict[str, str]) -> float:
    return float(row["score"])


def method_ratio(row: dict[str, str]) -> float:
    return float(row["token_ratio_vs_full_raw"])


def load_trials(config: Config) -> tuple[BenchConfig, dict[tuple[str, str, str, str], dict[str, str]]]:
    dirs = [Path(item) for item in config.benchmark_output_dirs]
    bench_config = bench_config_from_summary(dirs[0])
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for directory in dirs:
        for row in read_csv(directory / "trials.csv"):
            lookup[(row["benchmark"], row["task"], row["case_id"], row["method"])] = row
    return bench_config, lookup


def full_threshold(task: str, full_score: float, config: Config) -> float:
    if task in SUMMARY_TASKS:
        return max(0.0, full_score - config.summary_rouge_slack)
    return full_score


def successful_action_under_budget(rows: list[dict[str, str]], budget: float, threshold: float) -> str:
    under_budget = [row for row in rows if method_ratio(row) <= budget + 1e-12]
    successful = [row for row in under_budget if method_score(row) + 1e-12 >= threshold]
    if successful:
        selected = min(successful, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))
        return selected["method"]
    if under_budget:
        best_score = max(method_score(row) for row in under_budget)
        selected = min(
            [row for row in under_budget if abs(method_score(row) - best_score) < 1e-12],
            key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]),
        )
        return selected["method"]
    selected = min(rows, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))
    return selected["method"]


def oracle_budget_action(rows: list[dict[str, str]], config: Config) -> tuple[float, str]:
    full = next((row for row in rows if row["method"] == "full_raw"), None)
    full_score = method_score(full) if full is not None else max(method_score(row) for row in rows)
    threshold = full_threshold(rows[0]["task"], full_score, config)
    for budget in config.budget_bins:
        action = successful_action_under_budget(rows, budget, threshold)
        action_row = next(row for row in rows if row["method"] == action)
        if method_score(action_row) + 1e-12 >= threshold:
            return budget, action
    budget = config.budget_bins[-1]
    return budget, successful_action_under_budget(rows, budget, threshold)


def build_examples(
    tokenizer: Any,
    bench_config: BenchConfig,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[BudgetExample]:
    cases = load_longbench_cases(bench_config) + load_ruler_cases(bench_config)
    examples: list[BudgetExample] = []
    for case in cases:
        key = (case.benchmark, case.task, case.case_id)
        rows = [lookup[(*key, method)] for method in config.candidate_methods if (*key, method) in lookup]
        full = lookup.get((*key, "full_raw"))
        if not rows or full is None:
            continue
        budget, action = oracle_budget_action(rows, config)
        features, task_family = router_features(tokenizer, case, bench_config)
        examples.append(
            BudgetExample(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                task_family=task_family,
                oracle_budget=budget,
                oracle_action=action,
                full_score=method_score(full),
                features=features,
            )
        )
    if not examples:
        raise ValueError("no two-stage router examples were built")
    return examples


def split_indices(examples: list[BudgetExample], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    buckets: dict[tuple[float, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        buckets[(example.oracle_budget, example.task_family)].append(idx)
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


def normalize(examples: list[BudgetExample], train_indices: list[int]) -> tuple[list[float], list[float]]:
    dim = len(examples[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        vals = [examples[idx].features[col] for idx in train_indices]
        m = sum(vals) / max(1, len(vals))
        var = sum((val - m) ** 2 for val in vals) / max(1, len(vals))
        mean.append(m)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def tensorize(
    examples: list[BudgetExample],
    indices: list[int],
    budget_to_id: dict[float, int],
    action_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    budget_ys = []
    action_ys = []
    for idx in indices:
        example = examples[idx]
        xs.append([(val - mean[col]) / max(std[col], 1e-6) for col, val in enumerate(example.features)])
        budget_ys.append(budget_to_id[example.oracle_budget])
        action_ys.append(action_to_id[example.oracle_action])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(budget_ys, dtype=torch.long), torch.tensor(action_ys, dtype=torch.long)


def class_weights(labels: torch.Tensor, class_count: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=class_count).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    return weights / weights[weights > 0].mean().clamp_min(1e-6)


def train_router(
    examples: list[BudgetExample],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TwoStageBudgetRouter, dict[float, int], dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    budget_names = sorted({example.oracle_budget for example in examples})
    action_names = sorted({example.oracle_action for example in examples})
    budget_to_id = {budget: idx for idx, budget in enumerate(budget_names)}
    action_to_id = {action: idx for idx, action in enumerate(action_names)}
    mean, std = normalize(examples, train_indices)
    train_x, train_budget_y, train_action_y = tensorize(examples, train_indices, budget_to_id, action_to_id, mean, std)
    if test_indices:
        test_x, test_budget_y, test_action_y = tensorize(examples, test_indices, budget_to_id, action_to_id, mean, std)
    else:
        test_x, test_budget_y, test_action_y = train_x, train_budget_y, train_action_y

    torch.manual_seed(config.seed)
    model = TwoStageBudgetRouter(train_x.shape[1], config.hidden_dim, len(budget_names), len(action_names))
    budget_weights = class_weights(train_budget_y, len(budget_names))
    action_weights = class_weights(train_action_y, len(action_names))
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        budget_logits, action_logits = model(train_x)
        loss = (
            config.budget_loss_weight * F.cross_entropy(budget_logits, train_budget_y, weight=budget_weights)
            + config.action_loss_weight * F.cross_entropy(action_logits, train_action_y, weight=action_weights)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_budget_logits, train_action_logits = model(train_x)
                test_budget_logits, test_action_logits = model(test_x)
                history.append(
                    {
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "train_budget_acc": float((train_budget_logits.argmax(-1) == train_budget_y).float().mean()),
                        "train_action_acc": float((train_action_logits.argmax(-1) == train_action_y).float().mean()),
                        "test_budget_acc": float((test_budget_logits.argmax(-1) == test_budget_y).float().mean()),
                        "test_action_acc": float((test_action_logits.argmax(-1) == test_action_y).float().mean()),
                    }
                )
    return model, budget_to_id, action_to_id, mean, std, history


def ranker_input(
    features: list[float],
    mean: list[float],
    std: list[float],
    action: str,
    action_to_id: dict[str, int],
    token_ratio: float,
) -> list[float]:
    normalized = [(val - mean[col]) / max(std[col], 1e-6) for col, val in enumerate(features)]
    one_hot = [0.0] * len(action_to_id)
    one_hot[action_to_id[action]] = 1.0
    return normalized + one_hot + [token_ratio]


def train_score_ranker(
    examples: list[BudgetExample],
    train_indices: list[int],
    all_action_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> tuple[ActionScoreRanker, list[dict[str, Any]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for idx in train_indices:
        example = examples[idx]
        for row in candidate_rows_for_example(example, lookup, config):
            action = row["method"]
            xs.append(ranker_input(example.features, mean, std, action, all_action_to_id, method_ratio(row)))
            score = method_score(row)
            ys.append(score / example.full_score if example.full_score else score)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    torch.manual_seed(config.seed + 17)
    model = ActionScoreRanker(x.shape[1], config.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            with torch.inference_mode():
                mae = torch.mean(torch.abs(model(x) - y))
            history.append({"epoch": epoch, "ranker_mse": float(loss.detach()), "ranker_train_mae": float(mae)})
    return model, history


def candidate_rows_for_example(
    example: BudgetExample,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> list[dict[str, str]]:
    key = (example.benchmark, example.task, example.case_id)
    return [lookup[(*key, method)] for method in config.candidate_methods if (*key, method) in lookup]


def masked_action_from_logits(
    action_logits: torch.Tensor,
    id_to_action: dict[int, str],
    rows: list[dict[str, str]],
    budget: float,
) -> str:
    available = {row["method"]: row for row in rows}
    allowed = [
        (idx, action)
        for idx, action in id_to_action.items()
        if action in available and method_ratio(available[action]) <= budget + 1e-12
    ]
    if not allowed:
        cheapest = min(rows, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))
        return cheapest["method"]
    return max(allowed, key=lambda item: float(action_logits[item[0]]))[1]


def budget_rule_action(rows: list[dict[str, str]], budget: float, task: str) -> str:
    available = {row["method"]: row for row in rows}
    under_budget = [row for row in rows if method_ratio(row) <= budget + 1e-12]
    if not under_budget:
        return min(rows, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))["method"]
    if task in SUMMARY_TASKS:
        for method in ("summary1_8", "summary1_4", "summary1_2", "static_hier", "recent_only", "full_raw"):
            row = available.get(method)
            if row is not None and method_ratio(row) <= budget + 1e-12:
                return method
        return min(under_budget, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))["method"]
    retrievals = [
        row
        for row in under_budget
        if row["method"].startswith("retrieval_raw_k")
    ]
    if retrievals:
        return max(
            retrievals,
            key=lambda row: (int(row["method"].removeprefix("retrieval_raw_k")), -float(row["seconds"])),
        )["method"]
    for method in ("recent_only", "static_hier", "summary1_2", "summary1_4", "summary1_8", "full_raw"):
        row = available.get(method)
        if row is not None and method_ratio(row) <= budget + 1e-12:
            return method
    return max(under_budget, key=lambda row: method_score(row))["method"]


def score_ranker_action(
    ranker: ActionScoreRanker,
    example: BudgetExample,
    rows: list[dict[str, str]],
    budget: float,
    all_action_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> str:
    allowed = [row for row in rows if method_ratio(row) <= budget + 1e-12 and row["method"] in all_action_to_id]
    if not allowed:
        allowed = [min(rows, key=lambda row: (method_ratio(row), float(row["seconds"]), row["method"]))]
    xs = [
        ranker_input(example.features, mean, std, row["method"], all_action_to_id, method_ratio(row))
        for row in allowed
    ]
    x = torch.tensor(xs, dtype=torch.float32)
    ranker.eval()
    with torch.inference_mode():
        preds = ranker(x).tolist()
    best_idx = max(range(len(allowed)), key=lambda idx: (float(preds[idx]), -method_ratio(allowed[idx]), -float(allowed[idx]["seconds"])))
    return allowed[best_idx]["method"]


def add_prediction(
    out: list[PredictionRow],
    split: str,
    policy: str,
    example: BudgetExample,
    predicted_budget: float,
    predicted_action: str,
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    row = lookup.get((example.benchmark, example.task, example.case_id, predicted_action))
    if row is None:
        return
    score = method_score(row)
    full_score = example.full_score
    out.append(
        PredictionRow(
            split=split,
            policy=policy,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            task_family=example.task_family,
            oracle_budget=example.oracle_budget,
            predicted_budget=predicted_budget,
            oracle_action=example.oracle_action,
            predicted_action=predicted_action,
            budget_correct=int(abs(predicted_budget - example.oracle_budget) < 1e-12),
            action_correct=int(predicted_action == example.oracle_action),
            score=score,
            full_score=full_score,
            relative_to_full=score / full_score if full_score else "",
            token_ratio_vs_full_raw=method_ratio(row),
            seconds=float(row["seconds"]),
        )
    )


def evaluate_split(
    model: TwoStageBudgetRouter,
    examples: list[BudgetExample],
    indices: list[int],
    split: str,
    budget_to_id: dict[float, int],
    action_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
    ranker: ActionScoreRanker,
    all_action_to_id: dict[str, int],
) -> list[PredictionRow]:
    id_to_budget = {idx: budget for budget, idx in budget_to_id.items()}
    id_to_action = {idx: action for action, idx in action_to_id.items()}
    x, _, _ = tensorize(examples, indices, budget_to_id, action_to_id, mean, std)
    model.eval()
    with torch.inference_mode():
        budget_logits, action_logits = model(x)
    rows_out: list[PredictionRow] = []
    for local_idx, example_idx in enumerate(indices):
        example = examples[example_idx]
        rows = candidate_rows_for_example(example, lookup, config)
        pred_budget = id_to_budget[int(budget_logits[local_idx].argmax())]
        pred_action_masked = masked_action_from_logits(action_logits[local_idx], id_to_action, rows, pred_budget)
        oracle_budget_action_name = successful_action_under_budget(
            rows,
            example.oracle_budget,
            full_threshold(example.task, example.full_score, config),
        )
        pred_budget_oracle_action = successful_action_under_budget(
            rows,
            pred_budget,
            full_threshold(example.task, example.full_score, config),
        )
        action_head_unmasked = id_to_action[int(action_logits[local_idx].argmax())]
        pred_budget_rule_action = budget_rule_action(rows, pred_budget, example.task)
        oracle_budget_rule_action = budget_rule_action(rows, example.oracle_budget, example.task)
        pred_budget_ranker_action = score_ranker_action(ranker, example, rows, pred_budget, all_action_to_id, mean, std)
        oracle_budget_ranker_action = score_ranker_action(ranker, example, rows, example.oracle_budget, all_action_to_id, mean, std)
        add_prediction(rows_out, split, "two_stage_masked", example, pred_budget, pred_action_masked, lookup)
        add_prediction(rows_out, split, "action_head_unmasked", example, pred_budget, action_head_unmasked, lookup)
        add_prediction(rows_out, split, "oracle_budget_action_head", example, example.oracle_budget, masked_action_from_logits(action_logits[local_idx], id_to_action, rows, example.oracle_budget), lookup)
        add_prediction(rows_out, split, "pred_budget_oracle_action", example, pred_budget, pred_budget_oracle_action, lookup)
        add_prediction(rows_out, split, "oracle_budget_oracle_action", example, example.oracle_budget, oracle_budget_action_name, lookup)
        add_prediction(rows_out, split, "pred_budget_rule_action", example, pred_budget, pred_budget_rule_action, lookup)
        add_prediction(rows_out, split, "oracle_budget_rule_action", example, example.oracle_budget, oracle_budget_rule_action, lookup)
        add_prediction(rows_out, split, "pred_budget_score_ranker", example, pred_budget, pred_budget_ranker_action, lookup)
        add_prediction(rows_out, split, "oracle_budget_score_ranker", example, example.oracle_budget, oracle_budget_ranker_action, lookup)
    return rows_out


def summarize_predictions(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, row.task_family)].append(row)
        groups[(row.split, row.policy, row.benchmark if row.benchmark == "longbench" else row.benchmark)].append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        score = sum(row.score for row in items) / len(items)
        full = sum(row.full_score for row in items) / len(items)
        payload: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "group": group,
            "samples": len(items),
            "budget_accuracy": sum(row.budget_correct for row in items) / len(items),
            "action_accuracy": sum(row.action_correct for row in items) / len(items),
            "avg_score": score,
            "avg_full_score": full,
            "relative_to_full": score / full if full else "",
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
            "avg_seconds": sum(row.seconds for row in items) / len(items),
        }
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
    bench_config, lookup = load_trials(config)
    tokenizer = AutoTokenizer.from_pretrained(bench_config.model_name_or_path, trust_remote_code=True)
    examples = build_examples(tokenizer, bench_config, lookup, config)
    train_indices, test_indices = split_indices(examples, config)
    model, budget_to_id, action_to_id, mean, std, history = train_router(examples, train_indices, test_indices, config)
    all_action_names = sorted(
        {
            row["method"]
            for idx in train_indices
            for row in candidate_rows_for_example(examples[idx], lookup, config)
        }
    )
    all_action_to_id = {action: idx for idx, action in enumerate(all_action_names)}
    ranker, ranker_history = train_score_ranker(examples, train_indices, all_action_to_id, mean, std, lookup, config)

    rows: list[PredictionRow] = []
    rows.extend(evaluate_split(model, examples, train_indices, "train", budget_to_id, action_to_id, mean, std, lookup, config, ranker, all_action_to_id))
    rows.extend(evaluate_split(model, examples, test_indices, "test", budget_to_id, action_to_id, mean, std, lookup, config, ranker, all_action_to_id))
    summary = summarize_predictions(rows)
    id_to_budget = {idx: budget for budget, idx in budget_to_id.items()}
    id_to_action = {idx: action for action, idx in action_to_id.items()}

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "budget_names": [id_to_budget[idx] for idx in range(len(id_to_budget))],
            "action_names": [id_to_action[idx] for idx in range(len(id_to_action))],
            "all_action_names": all_action_names,
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "two_stage_budget_router.pt",
    )
    torch.save(
        {
            "state_dict": ranker.state_dict(),
            "input_dim": len(FEATURE_NAMES) + len(all_action_names) + 1,
            "hidden_dim": config.hidden_dim,
            "all_action_names": all_action_names,
            "feature_names": FEATURE_NAMES,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "action_score_ranker.pt",
    )
    write_csv(output_dir / "examples.csv", [asdict(example) for example in examples])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    write_csv(output_dir / "ranker_history.csv", ranker_history)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "bench_config": asdict(bench_config),
                "budget_names": [id_to_budget[idx] for idx in range(len(id_to_budget))],
                "action_names": [id_to_action[idx] for idx in range(len(id_to_action))],
                "all_action_names": all_action_names,
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "history_tail": history[-5:],
                "ranker_history_tail": ranker_history[-5:],
                "prediction_summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("split,policy,group,samples,budget_acc,action_acc,avg_score,relative_to_full,avg_token_ratio")
    for row in summary:
        print(
            f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
            f"{row['budget_accuracy']:.4f},{row['action_accuracy']:.4f},"
            f"{row['avg_score']:.4f},{row['relative_to_full']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print(f"saved router to {output_dir / 'two_stage_budget_router.pt'}")


if __name__ == "__main__":
    main()
