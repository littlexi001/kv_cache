from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
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
from run_blocksize_router_distill_from_sweeps import (  # noqa: E402
    finite_float,
    load_sweep_rows,
    threshold_for,
    write_csv,
)
from run_qwen8b_paper_benchmarks import load_longbench_cases, load_ruler_cases, parse_csv_tuple, router_features  # noqa: E402
from run_qwen8b_router_distill_from_trials import FEATURE_NAMES  # noqa: E402


ACTION_FEATURE_NAMES = [
    "action_is_full_raw",
    "action_block_tokens",
    "action_topk",
    "action_log2_block",
    "action_log2_topk",
    "action_est_selected_tokens",
    "action_est_token_ratio",
    "action_est_old_coverage",
    "action_est_block_fraction",
]


@dataclass(frozen=True)
class Config:
    benchmark_output_dirs: tuple[str, ...]
    output_dir: str
    feature_block_tokens: int
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    summary_rouge_slack: float
    quality_mode: str
    allowed_label_regex: str
    risk_threshold: float
    seed: int


@dataclass
class CaseLabel:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    full_score: float
    max_score: float
    safety_threshold: float
    min_safe_action: str
    min_safe_score: float
    min_safe_token_ratio: float
    min_safe_seconds: float
    safe_actions: str
    unsafe_actions: str
    features: list[float]


@dataclass
class ActionLabel:
    benchmark: str
    task: str
    case_id: str
    task_family: str
    action: str
    score: float
    token_ratio_vs_full_raw: float
    seconds: float
    safety_threshold: float
    dangerous: int
    is_min_safe_action: int
    action_features: list[float]


@dataclass
class CasePrediction:
    split: str
    policy: str
    benchmark: str
    task: str
    case_id: str
    task_family: str
    target_min_safe_action: str
    raw_predicted_action: str
    predicted_action: str
    upgraded: int
    raw_danger_prob: float
    final_danger_prob: float
    label_correct: int
    final_is_safe: int
    score: float
    token_ratio_vs_full_raw: float
    seconds: float


@dataclass
class DangerPrediction:
    split: str
    benchmark: str
    task: str
    case_id: str
    action: str
    target_dangerous: int
    predicted_dangerous: int
    danger_prob: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a block-size risk router from model-aware topK sweeps. "
            "The router predicts a cheapest safe action and a separate danger head."
        )
    )
    parser.add_argument("--benchmark_output_dirs", required=True, help="Comma-separated sweep output directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_block_tokens", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--summary_rouge_slack", type=float, default=0.03)
    parser.add_argument("--quality_mode", choices=["full", "best", "best_or_full"], default="best_or_full")
    parser.add_argument(
        "--allowed_label_regex",
        default="",
        help="Optional full-match regex for non-full candidate actions.",
    )
    parser.add_argument("--risk_threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026070805)
    args = parser.parse_args()
    return Config(
        benchmark_output_dirs=parse_csv_tuple(args.benchmark_output_dirs),
        output_dir=args.output_dir,
        feature_block_tokens=args.feature_block_tokens,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        summary_rouge_slack=args.summary_rouge_slack,
        quality_mode=args.quality_mode,
        allowed_label_regex=args.allowed_label_regex,
        risk_threshold=args.risk_threshold,
        seed=args.seed,
    )


def case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["benchmark"], row["task"], row["case_id"]


def dedupe_rows_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["label"])
        old = best.get(label)
        if old is None:
            best[label] = row
            continue
        old_key = (
            finite_float(old.get("score")),
            -finite_float(old.get("token_ratio_vs_full_raw"), 1.0),
            -finite_float(old.get("seconds"), 0.0),
        )
        new_key = (
            finite_float(row.get("score")),
            -finite_float(row.get("token_ratio_vs_full_raw"), 1.0),
            -finite_float(row.get("seconds"), 0.0),
        )
        if new_key > old_key:
            best[label] = row
    return list(best.values())


def allowed_candidate_rows(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["label"] != "full_raw"]
    if config.allowed_label_regex:
        allowed = [row for row in candidates if re.fullmatch(config.allowed_label_regex, str(row["label"]))]
        if allowed:
            candidates = allowed
    full = [row for row in rows if row["label"] == "full_raw"]
    return candidates + full[:1]


def parse_block_action(action: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"recent_plus_b(\d+)_span_top(\d+)_b0_a0", action)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def action_estimate_features(action: str, case_features: list[float]) -> list[float]:
    prefix_tokens = max(1.0, float(case_features[12]))
    older_tokens = max(0.0, float(case_features[13]))
    recent_tokens = max(0.0, float(case_features[14]))
    if action == "full_raw":
        selected = prefix_tokens
        return [
            1.0,
            older_tokens,
            max(1.0, math.ceil(older_tokens / 512.0)),
            math.log2(max(2.0, older_tokens)),
            math.log2(max(2.0, older_tokens / 512.0)),
            selected,
            1.0,
            1.0,
            1.0,
        ]
    parsed = parse_block_action(action)
    if parsed is None:
        return [0.0] * len(ACTION_FEATURE_NAMES)
    block_tokens, top_k = parsed
    old_blocks = max(1.0, math.ceil(older_tokens / max(1, block_tokens)))
    selected_old = min(older_tokens, float(block_tokens * min(top_k, old_blocks)))
    selected = recent_tokens + selected_old
    return [
        0.0,
        float(block_tokens),
        float(top_k),
        math.log2(float(block_tokens)),
        math.log2(float(max(1, top_k))),
        selected,
        min(1.0, selected / prefix_tokens),
        min(1.0, selected_old / max(1.0, older_tokens)),
        min(1.0, float(block_tokens) / max(1.0, older_tokens)),
    ]


def normalize_matrix(rows: list[list[float]], train_indices: list[int]) -> tuple[list[float], list[float]]:
    mean: list[float] = []
    std: list[float] = []
    dim = len(rows[0])
    for col in range(dim):
        values = [rows[idx][col] for idx in train_indices]
        m = sum(values) / len(values)
        var = sum((value - m) ** 2 for value in values) / len(values)
        mean.append(float(m))
        std.append(float(math.sqrt(var)) if var > 1e-12 else 1.0)
    return mean, std


def apply_norm(values: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(values)]


def build_labels(tokenizer: Any, config: Config) -> tuple[
    list[CaseLabel],
    list[ActionLabel],
    dict[tuple[str, str, str, str], dict[str, Any]],
]:
    rows, lookup, bench_configs = load_sweep_rows(config)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(case_key(row), []).append(row)

    case_lookup = {}
    for bench_config in bench_configs:
        for case in load_longbench_cases(bench_config) + load_ruler_cases(bench_config):
            case_lookup[(case.benchmark, case.task, case.case_id)] = case

    feature_config = bench_configs[0]
    case_labels: list[CaseLabel] = []
    action_labels: list[ActionLabel] = []
    for key, raw_group in sorted(grouped.items()):
        case = case_lookup.get(key)
        if case is None:
            continue
        group = dedupe_rows_by_label(
            [
                row
                for row in raw_group
                if math.isfinite(finite_float(row.get("score")))
                and math.isfinite(finite_float(row.get("token_ratio_vs_full_raw")))
            ]
        )
        if not group:
            continue
        threshold, full_score, max_score = threshold_for(group, config)
        candidates = allowed_candidate_rows(group, config)
        if not candidates:
            continue
        safe = [row for row in candidates if finite_float(row["score"]) + 1e-12 >= threshold]
        if not safe:
            safe = [row for row in candidates if row["label"] == "full_raw"] or [
                max(candidates, key=lambda row: finite_float(row["score"]))
            ]
        chosen = min(
            safe,
            key=lambda row: (
                finite_float(row["token_ratio_vs_full_raw"], 1.0),
                finite_float(row["seconds"], 0.0),
                str(row["label"]),
            ),
        )
        features, task_family = router_features(tokenizer, case, feature_config)
        safe_names = sorted(str(row["label"]) for row in candidates if finite_float(row["score"]) + 1e-12 >= threshold)
        unsafe_names = sorted(str(row["label"]) for row in candidates if finite_float(row["score"]) + 1e-12 < threshold)
        case_labels.append(
            CaseLabel(
                benchmark=key[0],
                task=key[1],
                case_id=key[2],
                task_family=task_family,
                full_score=full_score,
                max_score=max_score,
                safety_threshold=threshold,
                min_safe_action=str(chosen["label"]),
                min_safe_score=finite_float(chosen["score"]),
                min_safe_token_ratio=finite_float(chosen["token_ratio_vs_full_raw"], 1.0),
                min_safe_seconds=finite_float(chosen["seconds"], 0.0),
                safe_actions=";".join(safe_names),
                unsafe_actions=";".join(unsafe_names),
                features=features,
            )
        )
        for row in candidates:
            action = str(row["label"])
            action_labels.append(
                ActionLabel(
                    benchmark=key[0],
                    task=key[1],
                    case_id=key[2],
                    task_family=task_family,
                    action=action,
                    score=finite_float(row["score"]),
                    token_ratio_vs_full_raw=finite_float(row["token_ratio_vs_full_raw"], 1.0),
                    seconds=finite_float(row["seconds"], 0.0),
                    safety_threshold=threshold,
                    dangerous=int(finite_float(row["score"]) + 1e-12 < threshold),
                    is_min_safe_action=int(action == str(chosen["label"])),
                    action_features=features + action_estimate_features(action, features),
                )
            )
    if not case_labels:
        raise ValueError("no case labels were built")
    return case_labels, action_labels, lookup


def split_case_indices(labels: list[CaseLabel], config: Config) -> tuple[list[int], list[int]]:
    rng = random.Random(config.seed)
    by_group: dict[tuple[str, str, str], list[int]] = {}
    for idx, label in enumerate(labels):
        by_group.setdefault((label.benchmark, label.task, label.min_safe_action), []).append(idx)
    train: list[int] = []
    test: list[int] = []
    for indices in by_group.values():
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


def split_action_indices(labels: list[ActionLabel], test_case_keys: set[tuple[str, str, str]]) -> tuple[list[int], list[int]]:
    train: list[int] = []
    test: list[int] = []
    for idx, label in enumerate(labels):
        key = (label.benchmark, label.task, label.case_id)
        if key in test_case_keys:
            test.append(idx)
        else:
            train.append(idx)
    return train, test


def tensorize_cases(
    labels: list[CaseLabel],
    indices: list[int],
    label_to_id: dict[str, int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for idx in indices:
        row = labels[idx]
        xs.append(apply_norm(row.features, mean, std))
        ys.append(label_to_id[row.min_safe_action])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def tensorize_actions(
    labels: list[ActionLabel],
    indices: list[int],
    mean: list[float],
    std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for idx in indices:
        row = labels[idx]
        xs.append(apply_norm(row.action_features, mean, std))
        ys.append(float(row.dangerous))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).view(-1, 1)


def train_action_model(
    labels: list[CaseLabel],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, dict[str, int], list[float], list[float], list[dict[str, Any]]]:
    label_names = sorted({row.min_safe_action for row in labels})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    feature_rows = [row.features for row in labels]
    mean, std = normalize_matrix(feature_rows, train_indices)
    train_x, train_y = tensorize_cases(labels, train_indices, label_to_id, mean, std)
    test_x, test_y = tensorize_cases(labels, test_indices, label_to_id, mean, std) if test_indices else (train_x, train_y)
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


def train_danger_model(
    labels: list[ActionLabel],
    train_indices: list[int],
    test_indices: list[int],
    config: Config,
) -> tuple[TinyMemoryRouter, list[float], list[float], list[dict[str, Any]]]:
    feature_rows = [row.action_features for row in labels]
    mean, std = normalize_matrix(feature_rows, train_indices)
    train_x, train_y = tensorize_actions(labels, train_indices, mean, std)
    test_x, test_y = tensorize_actions(labels, test_indices, mean, std) if test_indices else (train_x, train_y)
    torch.manual_seed(config.seed + 17)
    model = TinyMemoryRouter(train_x.shape[1], config.hidden_dim, 1)
    positives = train_y.sum().clamp_min(1.0)
    negatives = (float(train_y.numel()) - train_y.sum()).clamp_min(1.0)
    pos_weight = negatives / positives
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_prob = torch.sigmoid(model(train_x))
                test_prob = torch.sigmoid(model(test_x))
                train_pred = (train_prob >= config.risk_threshold).float()
                test_pred = (test_prob >= config.risk_threshold).float()
                train_acc = float((train_pred == train_y).float().mean())
                test_acc = float((test_pred == test_y).float().mean())
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_danger_accuracy": train_acc,
                    "test_danger_accuracy": test_acc,
                }
            )
    return model, mean, std, history


def danger_prob_for_action(
    danger_model: TinyMemoryRouter,
    case_features: list[float],
    action: str,
    danger_mean: list[float],
    danger_std: list[float],
) -> float:
    features = case_features + action_estimate_features(action, case_features)
    x = torch.tensor([apply_norm(features, danger_mean, danger_std)], dtype=torch.float32)
    danger_model.eval()
    with torch.inference_mode():
        return float(torch.sigmoid(danger_model(x)).item())


def select_with_danger_head(
    raw_action: str,
    case_features: list[float],
    candidate_actions: list[str],
    danger_model: TinyMemoryRouter,
    danger_mean: list[float],
    danger_std: list[float],
    risk_threshold: float,
) -> tuple[str, float, float, int]:
    raw_prob = danger_prob_for_action(danger_model, case_features, raw_action, danger_mean, danger_std)
    if raw_prob < risk_threshold:
        return raw_action, raw_prob, raw_prob, 0
    raw_ratio = action_estimate_features(raw_action, case_features)[6]
    ordered = sorted(
        [
            action for action in candidate_actions
            if action_estimate_features(action, case_features)[6] + 1e-12 >= raw_ratio
        ],
        key=lambda action: (
            action_estimate_features(action, case_features)[6],
            action_estimate_features(action, case_features)[1],
            action_estimate_features(action, case_features)[2],
            action,
        ),
    )
    for action in ordered:
        prob = danger_prob_for_action(danger_model, case_features, action, danger_mean, danger_std)
        if prob < risk_threshold:
            return action, raw_prob, prob, int(action != raw_action)
    return "full_raw", raw_prob, danger_prob_for_action(danger_model, case_features, "full_raw", danger_mean, danger_std), int(raw_action != "full_raw")


def evaluate_cases(
    action_model: TinyMemoryRouter,
    danger_model: TinyMemoryRouter,
    labels: list[CaseLabel],
    indices: list[int],
    split: str,
    label_to_id: dict[str, int],
    action_mean: list[float],
    action_std: list[float],
    danger_mean: list[float],
    danger_std: list[float],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    candidate_actions: list[str],
    config: Config,
) -> list[CasePrediction]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    x, _ = tensorize_cases(labels, indices, label_to_id, action_mean, action_std)
    action_model.eval()
    with torch.inference_mode():
        probs = torch.softmax(action_model(x), dim=-1)
        raw_ids = probs.argmax(-1).tolist()
    rows: list[CasePrediction] = []
    for local_idx, case_idx in enumerate(indices):
        row = labels[case_idx]
        raw = id_to_label[int(raw_ids[local_idx])]
        for policy in ("raw_action_head", "risk_gated"):
            if policy == "risk_gated":
                final, raw_prob, final_prob, upgraded = select_with_danger_head(
                    raw,
                    row.features,
                    candidate_actions,
                    danger_model,
                    danger_mean,
                    danger_std,
                    config.risk_threshold,
                )
            else:
                final = raw
                raw_prob = danger_prob_for_action(danger_model, row.features, raw, danger_mean, danger_std)
                final_prob = raw_prob
                upgraded = 0
            trial = lookup.get((row.benchmark, row.task, row.case_id, final))
            score = finite_float(trial.get("score") if trial else None, 0.0)
            ratio = finite_float(trial.get("token_ratio_vs_full_raw") if trial else None, 1.0)
            seconds = finite_float(trial.get("seconds") if trial else None, 0.0)
            rows.append(
                CasePrediction(
                    split=split,
                    policy=policy,
                    benchmark=row.benchmark,
                    task=row.task,
                    case_id=row.case_id,
                    task_family=row.task_family,
                    target_min_safe_action=row.min_safe_action,
                    raw_predicted_action=raw,
                    predicted_action=final,
                    upgraded=upgraded,
                    raw_danger_prob=raw_prob,
                    final_danger_prob=final_prob,
                    label_correct=int(final == row.min_safe_action),
                    final_is_safe=int(score + 1e-12 >= row.safety_threshold),
                    score=score,
                    token_ratio_vs_full_raw=ratio,
                    seconds=seconds,
                )
            )
    return rows


def evaluate_danger(
    model: TinyMemoryRouter,
    labels: list[ActionLabel],
    indices: list[int],
    split: str,
    mean: list[float],
    std: list[float],
    config: Config,
) -> list[DangerPrediction]:
    x, y = tensorize_actions(labels, indices, mean, std)
    model.eval()
    with torch.inference_mode():
        prob = torch.sigmoid(model(x)).view(-1).tolist()
    rows: list[DangerPrediction] = []
    for local_idx, label_idx in enumerate(indices):
        row = labels[label_idx]
        danger_prob = float(prob[local_idx])
        rows.append(
            DangerPrediction(
                split=split,
                benchmark=row.benchmark,
                task=row.task,
                case_id=row.case_id,
                action=row.action,
                target_dangerous=row.dangerous,
                predicted_dangerous=int(danger_prob >= config.risk_threshold),
                danger_prob=danger_prob,
            )
        )
    return rows


def summarize_case_predictions(rows: list[CasePrediction]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[CasePrediction]] = {}
    for row in rows:
        groups.setdefault((row.split, row.policy, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.policy, row.benchmark), []).append(row)
        groups.setdefault((row.split, row.policy, row.task_family), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, policy, group), items in sorted(groups.items()):
        payload: dict[str, Any] = {
            "split": split,
            "policy": policy,
            "group": group,
            "samples": len(items),
            "label_accuracy": sum(row.label_correct for row in items) / len(items),
            "safe_rate": sum(row.final_is_safe for row in items) / len(items),
            "upgrade_rate": sum(row.upgraded for row in items) / len(items),
            "avg_score": sum(row.score for row in items) / len(items),
            "avg_token_ratio_vs_full_raw": sum(row.token_ratio_vs_full_raw for row in items) / len(items),
            "avg_seconds": sum(row.seconds for row in items) / len(items),
            "avg_final_danger_prob": sum(row.final_danger_prob for row in items) / len(items),
        }
        for row in items:
            key = f"select_{row.predicted_action}"
            payload[key] = payload.get(key, 0) + 1
        out.append(payload)
    return out


def summarize_danger(rows: list[DangerPrediction]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[DangerPrediction]] = {}
    for row in rows:
        groups.setdefault((row.split, "__overall__"), []).append(row)
        groups.setdefault((row.split, row.benchmark), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, group), items in sorted(groups.items()):
        tp = sum(1 for row in items if row.target_dangerous and row.predicted_dangerous)
        tn = sum(1 for row in items if not row.target_dangerous and not row.predicted_dangerous)
        fp = sum(1 for row in items if not row.target_dangerous and row.predicted_dangerous)
        fn = sum(1 for row in items if row.target_dangerous and not row.predicted_dangerous)
        out.append(
            {
                "split": split,
                "group": group,
                "samples": len(items),
                "accuracy": (tp + tn) / len(items),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "false_positive_rate": fp / max(1, fp + tn),
                "false_negative_rate": fn / max(1, fn + tp),
                "danger_rate": sum(row.target_dangerous for row in items) / len(items),
            }
        )
    return out


def main() -> None:
    from transformers import AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, _, bench_configs = load_sweep_rows(config)
    tokenizer = AutoTokenizer.from_pretrained(bench_configs[0].model_name_or_path, trust_remote_code=True)
    case_labels, action_labels, lookup = build_labels(tokenizer, config)
    train_indices, test_indices = split_case_indices(case_labels, config)
    test_keys = {(case_labels[idx].benchmark, case_labels[idx].task, case_labels[idx].case_id) for idx in test_indices}
    action_train_indices, action_test_indices = split_action_indices(action_labels, test_keys)

    action_model, label_to_id, action_mean, action_std, action_history = train_action_model(
        case_labels, train_indices, test_indices, config
    )
    danger_model, danger_mean, danger_std, danger_history = train_danger_model(
        action_labels, action_train_indices, action_test_indices, config
    )
    candidate_actions = sorted({row.action for row in action_labels})
    case_predictions = evaluate_cases(
        action_model,
        danger_model,
        case_labels,
        train_indices,
        "train",
        label_to_id,
        action_mean,
        action_std,
        danger_mean,
        danger_std,
        lookup,
        candidate_actions,
        config,
    )
    case_predictions += evaluate_cases(
        action_model,
        danger_model,
        case_labels,
        test_indices,
        "test",
        label_to_id,
        action_mean,
        action_std,
        danger_mean,
        danger_std,
        lookup,
        candidate_actions,
        config,
    )
    danger_predictions = evaluate_danger(danger_model, action_labels, action_train_indices, "train", danger_mean, danger_std, config)
    danger_predictions += evaluate_danger(danger_model, action_labels, action_test_indices, "test", danger_mean, danger_std, config)
    case_summary = summarize_case_predictions(case_predictions)
    danger_summary = summarize_danger(danger_predictions)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    label_names = [id_to_label[idx] for idx in range(len(id_to_label))]

    compatible_router = {
        "state_dict": action_model.state_dict(),
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": config.hidden_dim,
        "label_names": label_names,
        "feature_names": FEATURE_NAMES,
        "mean": action_mean,
        "std": action_std,
        "config": asdict(config),
    }
    torch.save(compatible_router, output_dir / "router.pt")
    torch.save(
        {
            "router_kind": "blocksize_risk_router",
            "action_state_dict": action_model.state_dict(),
            "danger_state_dict": danger_model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "action_input_dim": len(FEATURE_NAMES),
            "danger_input_dim": len(FEATURE_NAMES) + len(ACTION_FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "label_names": label_names,
            "candidate_actions": candidate_actions,
            "feature_names": FEATURE_NAMES,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "mean": action_mean,
            "std": action_std,
            "danger_mean": danger_mean,
            "danger_std": danger_std,
            "risk_threshold": config.risk_threshold,
            "config": asdict(config),
        },
        output_dir / "risk_router.pt",
    )
    write_csv(output_dir / "case_labels.csv", [asdict(row) for row in case_labels])
    write_csv(output_dir / "action_labels.csv", [asdict(row) for row in action_labels])
    write_csv(output_dir / "case_predictions.csv", [asdict(row) for row in case_predictions])
    write_csv(output_dir / "danger_predictions.csv", [asdict(row) for row in danger_predictions])
    write_csv(output_dir / "case_prediction_summary.csv", case_summary)
    write_csv(output_dir / "danger_prediction_summary.csv", danger_summary)
    write_csv(output_dir / "action_train_history.csv", action_history)
    write_csv(output_dir / "danger_train_history.csv", danger_history)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "input_sweep_rows": len(rows),
                "case_labels": len(case_labels),
                "action_labels": len(action_labels),
                "train_cases": len(train_indices),
                "test_cases": len(test_indices),
                "label_names": label_names,
                "candidate_actions": candidate_actions,
                "action_history_tail": action_history[-5:],
                "danger_history_tail": danger_history[-5:],
                "case_prediction_summary": case_summary,
                "danger_prediction_summary": danger_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("case split,policy,group,samples,label_acc,safe_rate,token")
    for row in case_summary:
        print(
            f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
            f"{row['label_accuracy']:.4f},{row['safe_rate']:.4f},{row['avg_token_ratio_vs_full_raw']:.4f}"
        )
    print("danger split,group,samples,acc,precision,recall,fnr")
    for row in danger_summary:
        print(
            f"{row['split']},{row['group']},{row['samples']},{row['accuracy']:.4f},"
            f"{row['precision']:.4f},{row['recall']:.4f},{row['false_negative_rate']:.4f}"
        )
    print(f"saved compatible router to {output_dir / 'router.pt'}")
    print(f"saved risk router to {output_dir / 'risk_router.pt'}")


if __name__ == "__main__":
    main()
