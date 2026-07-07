from __future__ import annotations

import argparse
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

from run_benchmark_calibrated_two_stage_router import (  # noqa: E402
    Example,
    action_meta,
    bench_config_from_summary,
    build_examples,
    load_trials,
    method_ratio,
    method_score,
    method_seconds,
    norm_features,
    normalize,
    rows_for_example,
)
from run_qwen8b_paper_benchmarks import SUMMARY_TASKS, parse_csv_tuple  # noqa: E402
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES, write_csv  # noqa: E402


DEFAULT_METHODS = (
    "full_raw,recent_only,static_hier,summary1_8,summary1_4,summary1_2,"
    "retrieval_raw_k1,retrieval_raw_k2,retrieval_raw_k3,retrieval_raw_k4,retrieval_raw_k8,"
    "recent_plus_static_hier,recent_plus_summary1_8,recent_plus_summary1_4,recent_plus_summary1_2,"
    "recent_plus_retrieval_raw_k1,recent_plus_retrieval_raw_k2,recent_plus_retrieval_raw_k3,"
    "recent_plus_retrieval_raw_k4,recent_plus_retrieval_raw_k8"
)


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
    summary_rouge_slack: float
    risk_threshold: float
    seed: int


@dataclass
class CaseLabel:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    full_score: float
    safety_threshold: float
    max_candidate_score: float
    has_safe_action: int
    min_safe_action: str
    min_safe_token_ratio: float
    min_safe_seconds: float
    safe_actions: str
    unsafe_actions: str


@dataclass
class ActionLabel:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    method: str
    score: float
    full_score: float
    safety_threshold: float
    token_ratio_vs_full_raw: float
    seconds: float
    dangerous: int
    is_min_safe_action: int


@dataclass
class PredictionRow:
    split: str
    policy: str
    benchmark: str
    task: str
    case_id: str
    task_family: str
    has_safe_action: int
    target_min_safe_action: str
    raw_predicted_action: str
    predicted_action: str
    predicted_danger_prob: float | str
    score: float
    full_score: float
    safety_threshold: float
    relative_to_full: float | str
    token_ratio_vs_full_raw: float
    seconds: float
    success: int
    dangerous: int


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


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train risk-aware router v2 from benchmark trials. "
            "It predicts whether a candidate action is dangerous and the minimal safe action per case."
        )
    )
    parser.add_argument("--benchmark_output_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_methods", default=DEFAULT_METHODS)
    parser.add_argument("--budget_bins", default="0.15,0.2,0.25,0.3,0.4,0.5,1.0")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--utility_token_penalty", type=float, default=0.18)
    parser.add_argument("--success_bonus", type=float, default=0.2)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--risk_threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026070604)
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
        summary_rouge_slack=args.summary_rouge_slack,
        risk_threshold=args.risk_threshold,
        seed=args.seed,
    )


def label_threshold(example: Example, rows: list[dict[str, str]], config: Config) -> float:
    if example.task in SUMMARY_TASKS:
        return max(0.0, example.full_score - config.summary_rouge_slack)
    return example.full_score


def action_is_safe(row: dict[str, str], threshold: float) -> bool:
    return method_score(row) + 1e-12 >= threshold


def choose_min_safe_action(
    example: Example,
    rows: list[dict[str, str]],
    config: Config,
) -> tuple[CaseLabel, list[ActionLabel]]:
    threshold = label_threshold(example, rows, config)
    safe_rows = [row for row in rows if action_is_safe(row, threshold)]
    max_score = max(method_score(row) for row in rows)
    if safe_rows:
        chosen = min(safe_rows, key=lambda row: (method_ratio(row), method_seconds(row), row["method"]))
        has_safe = 1
    else:
        fallback_rows = [row for row in rows if row["method"] == "full_raw"] or rows
        chosen = max(fallback_rows, key=lambda row: (method_score(row), -method_ratio(row), -method_seconds(row)))
        has_safe = 0
    safe_names = sorted(row["method"] for row in safe_rows)
    unsafe_names = sorted(row["method"] for row in rows if row not in safe_rows)
    case_label = CaseLabel(
        benchmark=example.benchmark,
        task=example.task,
        case_id=example.case_id,
        task_family=example.task_family,
        full_score=example.full_score,
        safety_threshold=threshold,
        max_candidate_score=max_score,
        has_safe_action=has_safe,
        min_safe_action=chosen["method"],
        min_safe_token_ratio=method_ratio(chosen),
        min_safe_seconds=method_seconds(chosen),
        safe_actions=";".join(safe_names),
        unsafe_actions=";".join(unsafe_names),
    )
    action_labels = [
        ActionLabel(
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            task_family=example.task_family,
            method=row["method"],
            score=method_score(row),
            full_score=example.full_score,
            safety_threshold=threshold,
            token_ratio_vs_full_raw=method_ratio(row),
            seconds=method_seconds(row),
            dangerous=int(not action_is_safe(row, threshold)),
            is_min_safe_action=int(row["method"] == chosen["method"]),
        )
        for row in rows
    ]
    return case_label, action_labels


def build_labels(
    examples: list[Example],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    config: Config,
) -> tuple[list[CaseLabel], list[ActionLabel], dict[tuple[str, str, str], CaseLabel]]:
    case_labels: list[CaseLabel] = []
    action_labels: list[ActionLabel] = []
    label_lookup: dict[tuple[str, str, str], CaseLabel] = {}
    for example in examples:
        rows = rows_for_example(example, lookup, config)
        if not rows:
            continue
        case_label, per_action = choose_min_safe_action(example, rows, config)
        case_labels.append(case_label)
        action_labels.extend(per_action)
        label_lookup[(example.benchmark, example.task, example.case_id)] = case_label
    if not case_labels:
        raise ValueError("no case labels were built")
    return case_labels, action_labels, label_lookup


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def retriever_derived_features(features: list[float]) -> list[float]:
    top1 = features[feature_index("retriever_top1_overlap")]
    top2 = features[feature_index("retriever_top2_overlap")]
    top3 = features[feature_index("retriever_top3_overlap")]
    gap = features[feature_index("retriever_score_gap")]
    positive = features[feature_index("retriever_positive_blocks")]
    num_blocks = max(1.0, features[feature_index("num_older_blocks")])
    top1_pos = features[feature_index("retriever_top1_position")]
    top2_pos = features[feature_index("retriever_top2_position")]
    denom = max(1e-6, top1)
    return [
        top2 / denom,
        top3 / denom,
        gap / denom,
        positive / num_blocks,
        abs(top1_pos - top2_pos),
        1.0 if gap <= 0.0 else 0.0,
        1.0 if positive >= 2.0 else 0.0,
        1.0 if positive >= 3.0 else 0.0,
    ]


DERIVED_FEATURE_NAMES = [
    "retriever_top2_over_top1",
    "retriever_top3_over_top1",
    "retriever_gap_over_top1",
    "retriever_positive_block_density",
    "retriever_top1_top2_position_distance",
    "retriever_no_gap",
    "retriever_has_two_positive_blocks",
    "retriever_has_three_positive_blocks",
]


def case_feature_names(task_to_id: dict[str, int], benchmark_to_id: dict[str, int], family_to_id: dict[str, int]) -> list[str]:
    return (
        FEATURE_NAMES
        + DERIVED_FEATURE_NAMES
        + [f"task={name}" for name in sorted(task_to_id, key=task_to_id.get)]
        + [f"benchmark={name}" for name in sorted(benchmark_to_id, key=benchmark_to_id.get)]
        + [f"task_family={name}" for name in sorted(family_to_id, key=family_to_id.get)]
    )


def action_feature_names(
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
    method_to_id: dict[str, int],
) -> list[str]:
    return (
        case_feature_names(task_to_id, benchmark_to_id, family_to_id)
        + [f"method={name}" for name in sorted(method_to_id, key=method_to_id.get)]
        + [
            "action_is_full",
            "action_has_recent",
            "action_is_summary",
            "action_is_retrieval",
            "action_has_recent_anchor",
            "action_log_retrieval_k",
            "action_summary_ratio",
            "action_token_ratio",
            "action_log100_token_ratio",
        ]
    )


def vocabularies(examples: list[Example]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    tasks = sorted({example.task for example in examples})
    benchmarks = sorted({example.benchmark for example in examples})
    families = sorted({example.task_family for example in examples})
    return (
        {task: idx for idx, task in enumerate(tasks)},
        {benchmark: idx for idx, benchmark in enumerate(benchmarks)},
        {family: idx for idx, family in enumerate(families)},
    )


def one_hot(name: str, mapping: dict[str, int]) -> list[float]:
    values = [0.0] * len(mapping)
    if name in mapping:
        values[mapping[name]] = 1.0
    return values


def case_input(
    example: Example,
    mean: list[float],
    std: list[float],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
) -> list[float]:
    return (
        norm_features(example.features, mean, std)
        + retriever_derived_features(example.features)
        + one_hot(example.task, task_to_id)
        + one_hot(example.benchmark, benchmark_to_id)
        + one_hot(example.task_family, family_to_id)
    )


def action_input(
    example: Example,
    method: str,
    ratio: float,
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
) -> list[float]:
    method_bits = [0.0] * len(method_to_id)
    method_bits[method_to_id[method]] = 1.0
    return (
        case_input(example, mean, std, task_to_id, benchmark_to_id, family_to_id)
        + method_bits
        + action_meta(method)
        + [ratio, math.log1p(100.0 * ratio)]
    )


def split_indices_from_labels(
    examples: list[Example],
    label_lookup: dict[tuple[str, str, str], CaseLabel],
    config: Config,
) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
        label = label_lookup[(example.benchmark, example.task, example.case_id)]
        buckets[(example.benchmark, example.task, label.min_safe_action)].append(idx)
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


def train_safe_action_model(
    examples: list[Example],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    label_lookup: dict[tuple[str, str, str], CaseLabel],
    action_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
    config: Config,
) -> tuple[MLP, list[dict[str, Any]]]:
    def xy(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[list[float]] = []
        ys: list[int] = []
        for idx in indices:
            example = examples[idx]
            label = label_lookup[(example.benchmark, example.task, example.case_id)]
            xs.append(case_input(example, mean, std, task_to_id, benchmark_to_id, family_to_id))
            ys.append(action_to_id[label.min_safe_action])
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = xy(train_indices)
    test_x, test_y = xy(test_indices) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = MLP(train_x.shape[1], config.hidden_dim, len(action_to_id))
    counts = torch.bincount(train_y, minlength=len(action_to_id)).float()
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
            history.append(
                {
                    "epoch": epoch,
                    "safe_action_loss": float(loss.detach()),
                    "train_safe_action_acc": train_acc,
                    "test_safe_action_acc": test_acc,
                }
            )
    return model, history


def train_risk_model(
    examples: list[Example],
    train_indices: list[int],
    test_indices: list[int],
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    label_lookup: dict[tuple[str, str, str], CaseLabel],
    config: Config,
) -> tuple[MLP, list[dict[str, Any]]]:
    def xy(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[list[float]] = []
        ys: list[float] = []
        for idx in indices:
            example = examples[idx]
            label = label_lookup[(example.benchmark, example.task, example.case_id)]
            for row in rows_for_example(example, lookup, config):
                if row["method"] not in method_to_id:
                    continue
                xs.append(
                    action_input(
                        example,
                        row["method"],
                        method_ratio(row),
                        mean,
                        std,
                        method_to_id,
                        task_to_id,
                        benchmark_to_id,
                        family_to_id,
                    )
                )
                ys.append(float(not action_is_safe(row, label.safety_threshold)))
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    train_x, train_y = xy(train_indices)
    test_x, test_y = xy(test_indices) if test_indices else (train_x, train_y)
    if train_x.numel() == 0:
        raise ValueError("no risk training rows were built")
    torch.manual_seed(config.seed + 1)
    model = MLP(train_x.shape[1], config.hidden_dim, 1)
    pos = train_y.sum()
    neg = float(train_y.numel()) - float(pos)
    if float(pos) == 0.0 or neg == 0.0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([neg / float(pos)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_prob = torch.sigmoid(model(train_x).squeeze(-1))
                test_prob = torch.sigmoid(model(test_x).squeeze(-1))
                train_pred = (train_prob >= 0.5).float()
                test_pred = (test_prob >= 0.5).float()
                train_acc = float((train_pred == train_y).float().mean())
                test_acc = float((test_pred == test_y).float().mean())
                train_recall = recall_at_threshold(train_prob, train_y, 0.5)
                test_recall = recall_at_threshold(test_prob, test_y, 0.5)
            history.append(
                {
                    "epoch": epoch,
                    "risk_loss": float(loss.detach()),
                    "train_risk_acc": train_acc,
                    "test_risk_acc": test_acc,
                    "train_danger_recall": train_recall,
                    "test_danger_recall": test_recall,
                    "train_actions": int(train_y.numel()),
                    "train_danger_rate": float(train_y.mean()),
                }
            )
    return model, history


def recall_at_threshold(prob: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    positives = target >= 0.5
    if int(positives.sum()) == 0:
        return 1.0
    pred_pos = prob >= threshold
    return float((pred_pos[positives]).float().mean())


def predict_safe_action(
    model: MLP,
    example: Example,
    mean: list[float],
    std: list[float],
    action_id_to_name: dict[int, str],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
) -> str:
    x = torch.tensor([case_input(example, mean, std, task_to_id, benchmark_to_id, family_to_id)], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        pred_id = int(model(x).argmax(-1).item())
    return action_id_to_name[pred_id]


def predict_danger_prob(
    model: MLP,
    example: Example,
    row: dict[str, str],
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
) -> float:
    x = torch.tensor(
        [
            action_input(
                example,
                row["method"],
                method_ratio(row),
                mean,
                std,
                method_to_id,
                task_to_id,
                benchmark_to_id,
                family_to_id,
            )
        ],
        dtype=torch.float32,
    )
    model.eval()
    with torch.inference_mode():
        return float(torch.sigmoid(model(x).squeeze(-1)).item())


def row_by_method(rows: list[dict[str, str]], method: str) -> dict[str, str] | None:
    for row in rows:
        if row["method"] == method:
            return row
    return None


def resolve_available_action(rows: list[dict[str, str]], preferred: str) -> str:
    if row_by_method(rows, preferred) is not None:
        return preferred
    if row_by_method(rows, "full_raw") is not None:
        return "full_raw"
    return min(rows, key=lambda row: (method_ratio(row), method_seconds(row), row["method"]))["method"]


def choose_cheapest_predicted_safe(
    risk_model: MLP,
    example: Example,
    rows: list[dict[str, str]],
    mean: list[float],
    std: list[float],
    method_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
    config: Config,
) -> tuple[str, float]:
    scored: list[tuple[dict[str, str], float]] = []
    for row in rows:
        if row["method"] not in method_to_id:
            continue
        prob = predict_danger_prob(risk_model, example, row, mean, std, method_to_id, task_to_id, benchmark_to_id, family_to_id)
        scored.append((row, prob))
    if not scored:
        row = min(rows, key=lambda item: (method_ratio(item), method_seconds(item), item["method"]))
        return row["method"], 1.0
    safe = [(row, prob) for row, prob in scored if prob <= config.risk_threshold]
    if safe:
        row, prob = min(safe, key=lambda item: (method_ratio(item[0]), method_seconds(item[0]), item[0]["method"]))
        return row["method"], prob
    row, prob = min(scored, key=lambda item: (item[1], method_ratio(item[0]), item[0]["method"]))
    return row["method"], prob


def add_prediction(
    out: list[PredictionRow],
    split: str,
    policy: str,
    example: Example,
    raw_action: str,
    action: str,
    danger_prob: float | str,
    rows: list[dict[str, str]],
    label: CaseLabel,
) -> None:
    row = row_by_method(rows, action)
    if row is None:
        return
    score = method_score(row)
    success = int(action_is_safe(row, label.safety_threshold))
    out.append(
        PredictionRow(
            split=split,
            policy=policy,
            benchmark=example.benchmark,
            task=example.task,
            case_id=example.case_id,
            task_family=example.task_family,
            has_safe_action=label.has_safe_action,
            target_min_safe_action=label.min_safe_action,
            raw_predicted_action=raw_action,
            predicted_action=action,
            predicted_danger_prob=danger_prob,
            score=score,
            full_score=example.full_score,
            safety_threshold=label.safety_threshold,
            relative_to_full=score / example.full_score if example.full_score else "",
            token_ratio_vs_full_raw=method_ratio(row),
            seconds=method_seconds(row),
            success=success,
            dangerous=1 - success,
        )
    )


def evaluate(
    examples: list[Example],
    indices: list[int],
    split: str,
    safe_model: MLP,
    risk_model: MLP,
    mean: list[float],
    std: list[float],
    action_id_to_name: dict[int, str],
    method_to_id: dict[str, int],
    task_to_id: dict[str, int],
    benchmark_to_id: dict[str, int],
    family_to_id: dict[str, int],
    lookup: dict[tuple[str, str, str, str], dict[str, str]],
    label_lookup: dict[tuple[str, str, str], CaseLabel],
    config: Config,
) -> list[PredictionRow]:
    out: list[PredictionRow] = []
    for idx in indices:
        example = examples[idx]
        label = label_lookup[(example.benchmark, example.task, example.case_id)]
        rows = rows_for_example(example, lookup, config)
        if not rows:
            continue

        oracle_action = resolve_available_action(rows, label.min_safe_action)
        add_prediction(out, split, "oracle_min_safe", example, label.min_safe_action, oracle_action, "", rows, label)

        raw_safe = predict_safe_action(
            safe_model,
            example,
            mean,
            std,
            action_id_to_name,
            task_to_id,
            benchmark_to_id,
            family_to_id,
        )
        safe_action = resolve_available_action(rows, raw_safe)
        safe_row = row_by_method(rows, safe_action)
        safe_prob: float | str = ""
        if safe_row is not None and safe_action in method_to_id:
            safe_prob = predict_danger_prob(
                risk_model, example, safe_row, mean, std, method_to_id, task_to_id, benchmark_to_id, family_to_id
            )
        add_prediction(out, split, "safe_action_classifier", example, raw_safe, safe_action, safe_prob, rows, label)

        cheap_action, cheap_prob = choose_cheapest_predicted_safe(
            risk_model,
            example,
            rows,
            mean,
            std,
            method_to_id,
            task_to_id,
            benchmark_to_id,
            family_to_id,
            config,
        )
        add_prediction(out, split, "risk_filtered_cheapest", example, cheap_action, cheap_action, cheap_prob, rows, label)

        if safe_row is not None and isinstance(safe_prob, float) and safe_prob <= config.risk_threshold:
            final_action = safe_action
            final_prob = safe_prob
            raw_final = raw_safe
        else:
            final_action = cheap_action
            final_prob = cheap_prob
            raw_final = cheap_action
        add_prediction(out, split, "safe_classifier_then_risk_filter", example, raw_final, final_action, final_prob, rows, label)

        full_action = resolve_available_action(rows, "full_raw")
        add_prediction(out, split, "full_raw", example, "full_raw", full_action, "", rows, label)
    return out


def summarize(rows: list[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        benchmark_group = row.benchmark if row.benchmark == "longbench" else row.benchmark
        groups[(row.split, row.policy, "__overall__")].append(row)
        groups[(row.split, row.policy, benchmark_group)].append(row)
        groups[(row.split, row.policy, row.task_family)].append(row)
        groups[(row.split, row.policy, row.task)].append(row)
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
            "danger_rate": sum(row.dangerous for row in items) / len(items),
            "no_safe_case_rate": sum(1 - row.has_safe_action for row in items) / len(items),
        }
        counts = Counter(row.predicted_action for row in items)
        for action, count in sorted(counts.items()):
            payload[f"select_{action}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    config = parse_args()
    from transformers import AutoTokenizer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_configs, lookup = load_trials(config)
    tokenizer = AutoTokenizer.from_pretrained(bench_configs[0].model_name_or_path, trust_remote_code=True)
    examples = build_examples(tokenizer, bench_configs, lookup, config)
    case_labels, action_labels, label_lookup = build_labels(examples, lookup, config)
    train_indices, test_indices = split_indices_from_labels(examples, label_lookup, config)
    mean, std = normalize(examples, train_indices)
    task_to_id, benchmark_to_id, family_to_id = vocabularies(examples)
    action_names = sorted({label.min_safe_action for label in case_labels})
    action_to_id = {name: idx for idx, name in enumerate(action_names)}
    action_id_to_name = {idx: name for name, idx in action_to_id.items()}
    method_to_id = {method: idx for idx, method in enumerate(config.candidate_methods)}

    safe_model, safe_history = train_safe_action_model(
        examples,
        train_indices,
        test_indices,
        mean,
        std,
        label_lookup,
        action_to_id,
        task_to_id,
        benchmark_to_id,
        family_to_id,
        config,
    )
    risk_model, risk_history = train_risk_model(
        examples,
        train_indices,
        test_indices,
        mean,
        std,
        method_to_id,
        task_to_id,
        benchmark_to_id,
        family_to_id,
        lookup,
        label_lookup,
        config,
    )
    prediction_rows = evaluate(
        examples,
        train_indices,
        "train",
        safe_model,
        risk_model,
        mean,
        std,
        action_id_to_name,
        method_to_id,
        task_to_id,
        benchmark_to_id,
        family_to_id,
        lookup,
        label_lookup,
        config,
    ) + evaluate(
        examples,
        test_indices,
        "test",
        safe_model,
        risk_model,
        mean,
        std,
        action_id_to_name,
        method_to_id,
        task_to_id,
        benchmark_to_id,
        family_to_id,
        lookup,
        label_lookup,
        config,
    )
    prediction_summary = summarize(prediction_rows)
    case_names = case_feature_names(task_to_id, benchmark_to_id, family_to_id)
    risk_names = action_feature_names(task_to_id, benchmark_to_id, family_to_id, method_to_id)
    first_rows = rows_for_example(examples[0], lookup, config)

    write_csv(output_dir / "case_labels.csv", [asdict(row) for row in case_labels])
    write_csv(output_dir / "action_labels.csv", [asdict(row) for row in action_labels])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in prediction_rows])
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "safe_action_history.csv", safe_history)
    write_csv(output_dir / "risk_history.csv", risk_history)

    torch.save(
        {
            "safe_action_model": safe_model.state_dict(),
            "risk_model": risk_model.state_dict(),
            "case_input_dim": len(case_input(examples[0], mean, std, task_to_id, benchmark_to_id, family_to_id)),
            "risk_input_dim": len(
                action_input(
                    examples[0],
                    first_rows[0]["method"],
                    method_ratio(first_rows[0]),
                    mean,
                    std,
                    method_to_id,
                    task_to_id,
                    benchmark_to_id,
                    family_to_id,
                )
            ),
            "hidden_dim": config.hidden_dim,
            "method_to_id": method_to_id,
            "action_to_id": action_to_id,
            "task_to_id": task_to_id,
            "benchmark_to_id": benchmark_to_id,
            "family_to_id": family_to_id,
            "mean": mean,
            "std": std,
            "feature_names": FEATURE_NAMES,
            "derived_feature_names": DERIVED_FEATURE_NAMES,
            "case_feature_names": case_names,
            "risk_feature_names": risk_names,
            "config": asdict(config),
        },
        output_dir / "risk_router_v2.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "benchmark_dirs": [asdict(bench_config) for bench_config in bench_configs],
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "case_labels": len(case_labels),
                "action_labels": len(action_labels),
                "min_safe_action_counts": dict(Counter(row.min_safe_action for row in case_labels)),
                "safe_action_history_tail": safe_history[-5:],
                "risk_history_tail": risk_history[-5:],
                "prediction_summary": prediction_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("split,policy,group,samples,success_rate,avg_token_ratio_vs_full_raw,relative_to_full")
    for row in prediction_summary:
        if row["split"] == "test" and row["group"] in {"__overall__", "longbench", "ruler_4096", "ruler_8192", "ruler_16384"}:
            rel = row["relative_to_full"]
            rel_text = f"{rel:.4f}" if isinstance(rel, float) else ""
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['success_rate']:.4f},{row['avg_token_ratio_vs_full_raw']:.4f},{rel_text}"
            )
    print(f"saved risk-aware router v2 to {output_dir / 'risk_router_v2.pt'}")


if __name__ == "__main__":
    main()
