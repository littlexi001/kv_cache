from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_variable_budget_planner_from_repack_results as vb  # noqa: E402


@dataclass(frozen=True)
class Config:
    benchmark_dirs: tuple[str, ...]
    benchmark_groups: tuple[tuple[str, ...], ...]
    output_dir: str
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float
    test_fraction: float
    use_text_features: bool
    split_by_case: bool
    safety_thresholds: tuple[float, ...]
    seed: int


@dataclass
class CandidateExample:
    case_index: int
    action: str
    safe_vs_full: int
    action_kv_ratio: float
    features: list[float]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train an output-level risk verifier for KV budget candidates.")
    parser.add_argument("--benchmark_dirs", default="")
    parser.add_argument("--benchmark_groups", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--use_text_features", action="store_true")
    parser.add_argument("--split_by_case", action="store_true")
    parser.add_argument("--safety_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--seed", type=int, default=2026070796)
    args = parser.parse_args()
    benchmark_dirs = vb.base.parse_csv_tuple(args.benchmark_dirs)
    benchmark_groups = vb.parse_benchmark_groups(args.benchmark_groups)
    if not benchmark_groups and benchmark_dirs:
        benchmark_groups = (benchmark_dirs,)
    if not benchmark_groups:
        raise ValueError("provide --benchmark_dirs or --benchmark_groups")
    return Config(
        benchmark_dirs=benchmark_dirs,
        benchmark_groups=benchmark_groups,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_fraction=args.test_fraction,
        use_text_features=args.use_text_features,
        split_by_case=args.split_by_case,
        safety_thresholds=tuple(float(item) for item in vb.base.parse_csv_tuple(args.safety_thresholds)),
        seed=args.seed,
    )


def as_vb_config(config: Config) -> vb.Config:
    return vb.Config(
        benchmark_dirs=config.benchmark_dirs,
        benchmark_groups=config.benchmark_groups,
        output_dir=config.output_dir,
        label_target="min_safe",
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        label_smoothing=0.0,
        confidence_penalty=0.0,
        ce_loss_weight=1.0,
        expected_cost_weight=0.0,
        unsafe_cost_weight=2.0,
        best_gap_cost_weight=1.0,
        kv_cost_weight=0.05,
        include_full_action=False,
        test_fraction=config.test_fraction,
        use_text_features=config.use_text_features,
        split_by_case=config.split_by_case,
        risk_thresholds=config.safety_thresholds,
        holdout_tasks=(),
        holdout_benchmarks=(),
        seed=config.seed,
    )


def compact_actions_for_examples(
    examples: list[vb.VariableBudgetExample],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[str]:
    actions: set[str] = set()
    for example in examples:
        payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        actions.update(action for action in vb.available_actions(payload) if action != "full")
    return sorted(actions, key=lambda action: (vb.action_budget(action), action))


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9_\- ]+", "", value)
    return value.strip()


def output_features(prediction: str, neighbors: dict[str, str], action: str) -> list[float]:
    normalized = normalize_text(prediction)
    words = normalized.split()
    unique_words = set(words)
    bigrams = list(zip(words, words[1:]))
    repeated_bigram_ratio = 0.0
    if bigrams:
        repeated_bigram_ratio = 1.0 - (len(set(bigrams)) / max(1, len(bigrams)))
    budget = vb.action_budget(action)
    smaller = [text for name, text in neighbors.items() if vb.action_budget(name) < budget]
    larger = [text for name, text in neighbors.items() if vb.action_budget(name) > budget]
    same_as_smaller = any(normalized and normalized == normalize_text(text) for text in smaller)
    same_as_larger = any(normalized and normalized == normalize_text(text) for text in larger)
    same_as_any = any(normalized and normalized == normalize_text(text) for name, text in neighbors.items() if name != action)
    return [
        math.log1p(len(prediction)),
        math.log1p(len(words)),
        len(unique_words) / max(1.0, float(len(words))),
        repeated_bigram_ratio,
        1.0 if "passage" in normalized else 0.0,
        1.0 if "question" in normalized else 0.0,
        1.0 if "answer" in normalized else 0.0,
        1.0 if "only give" in normalized else 0.0,
        1.0 if normalized.endswith("?") else 0.0,
        1.0 if same_as_smaller else 0.0,
        1.0 if same_as_larger else 0.0,
        1.0 if same_as_any else 0.0,
    ]


def candidate_feature_names(base_names: list[str], compact_actions: list[str]) -> list[str]:
    names = list(base_names)
    names.extend(
        [
            "candidate_budget_log",
            "candidate_budget_rank",
            "candidate_kv_ratio",
            "candidate_is_k1",
            "candidate_is_k2",
            "candidate_is_k3",
            "candidate_is_k4",
            "candidate_is_k6",
            "candidate_is_k8",
        ]
    )
    names.extend(f"candidate={action}" for action in compact_actions)
    names.extend(
        [
            "prediction_chars_log",
            "prediction_words_log",
            "prediction_unique_word_ratio",
            "prediction_repeated_bigram_ratio",
            "prediction_contains_passage",
            "prediction_contains_question",
            "prediction_contains_answer",
            "prediction_contains_only_give",
            "prediction_ends_question_mark",
            "prediction_same_as_smaller_budget",
            "prediction_same_as_larger_budget",
            "prediction_same_as_any_budget",
        ]
    )
    return names


def build_candidate_examples(
    examples: list[vb.VariableBudgetExample],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    compact_actions: list[str],
) -> list[CandidateExample]:
    out: list[CandidateExample] = []
    for case_idx, example in enumerate(examples):
        payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        full_score = vb.row_score(payload["full"])
        ft = max(1, vb.full_tokens(payload))
        available = set(vb.available_actions(payload))
        predictions = {
            action: vb.action_row(payload, action).get("prediction", "")
            for action in compact_actions
            if action in available
        }
        for rank, action in enumerate(compact_actions):
            if action not in available:
                continue
            row = vb.action_row(payload, action)
            budget = vb.action_budget(action)
            kv_ratio = vb.row_kv(row) / ft
            suffix = [
                math.log1p(budget),
                float(rank) / max(1.0, float(len(compact_actions) - 1)),
                kv_ratio,
                1.0 if budget == 1 else 0.0,
                1.0 if budget == 2 else 0.0,
                1.0 if budget == 3 else 0.0,
                1.0 if budget == 4 else 0.0,
                1.0 if budget == 6 else 0.0,
                1.0 if budget == 8 else 0.0,
            ]
            suffix.extend(1.0 if candidate == action else 0.0 for candidate in compact_actions)
            suffix.extend(output_features(row.get("prediction", ""), predictions, action))
            score = vb.row_score(row)
            out.append(
                CandidateExample(
                    case_index=case_idx,
                    action=action,
                    safe_vs_full=int(score + 1e-12 >= full_score),
                    action_kv_ratio=kv_ratio,
                    features=example.features + suffix,
                )
            )
    return out


def normalize_features(items: list[CandidateExample], train_cases: set[int]) -> tuple[list[float], list[float]]:
    train_items = [item for item in items if item.case_index in train_cases]
    dim = len(items[0].features)
    mean: list[float] = []
    std: list[float] = []
    for col in range(dim):
        values = [item.features[col] for item in train_items]
        m = sum(values) / max(1, len(values))
        var = sum((value - m) ** 2 for value in values) / max(1, len(values))
        mean.append(m)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def norm(features: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(features)]


def train_model(
    items: list[CandidateExample],
    train_cases: list[int],
    test_cases: list[int],
    mean: list[float],
    std: list[float],
    config: Config,
) -> tuple[vb.base.MLP, list[dict[str, Any]]]:
    train_set = set(train_cases)
    test_set = set(test_cases)

    def xy(case_set: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        selected = [item for item in items if item.case_index in case_set]
        xs = [norm(item.features, mean, std) for item in selected]
        ys = [item.safe_vs_full for item in selected]
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = xy(train_set)
    test_x, test_y = xy(test_set) if test_set else (train_x, train_y)
    torch.manual_seed(config.seed)
    model = vb.base.MLP(train_x.shape[1], config.hidden_dim, 2)
    counts = torch.bincount(train_y, minlength=2).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    if torch.any(weights > 0):
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
                    "loss": float(loss.detach()),
                    "train_candidate_accuracy": train_acc,
                    "test_candidate_accuracy": test_acc,
                }
            )
    return model, history


def safe_probability(
    model: vb.base.MLP,
    item: CandidateExample,
    mean: list[float],
    std: list[float],
) -> float:
    x = torch.tensor([norm(item.features, mean, std)], dtype=torch.float32)
    with torch.inference_mode():
        return float(torch.softmax(model(x), dim=-1)[0, 1])


def evaluate(
    model: vb.base.MLP,
    examples: list[vb.VariableBudgetExample],
    case_indices: list[int],
    split: str,
    compact_actions: list[str],
    items: list[CandidateExample],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    mean: list[float],
    std: list[float],
    config: Config,
) -> list[vb.PredictionRow]:
    rows: list[vb.PredictionRow] = []
    item_by_key = {(item.case_index, item.action): item for item in items}
    for case_idx in case_indices:
        example = examples[case_idx]
        target = example.label_min_safe
        payload = lookup[(example.source, example.benchmark, example.task, example.case_id)]
        for action in vb.available_actions(payload):
            vb.add_prediction(rows, split, f"fixed_{action}", example, target, action, lookup)
        vb.add_prediction(rows, split, "oracle_min_safe", example, target, vb.choose_min_safe(payload), lookup)
        vb.add_prediction(rows, split, "oracle_best", example, target, vb.choose_best(payload), lookup)
        probs = {
            action: safe_probability(model, item_by_key[(case_idx, action)], mean, std)
            for action in compact_actions
            if (case_idx, action) in item_by_key
        }
        for tau in config.safety_thresholds:
            chosen = "full"
            for action in sorted(probs, key=lambda value: (vb.action_budget(value), value)):
                if probs[action] >= tau:
                    chosen = action
                    break
            vb.add_prediction(rows, split, f"output_verifier_tau_{tau:g}", example, target, chosen, lookup)
    return rows


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
    vb_config = as_vb_config(config)
    examples, lookup, base_names = vb.build_examples(vb_config)
    train_indices, test_indices = vb.split_indices(examples, vb_config)
    compact_actions = compact_actions_for_examples(examples, lookup)
    items = build_candidate_examples(examples, lookup, compact_actions)
    mean, std = normalize_features(items, set(train_indices))
    model, history = train_model(items, train_indices, test_indices, mean, std, config)
    rows = evaluate(model, examples, train_indices, "train", compact_actions, items, lookup, mean, std, config)
    rows += evaluate(model, examples, test_indices, "test", compact_actions, items, lookup, mean, std, config)
    summary = vb.summarize(rows)
    feature_names = candidate_feature_names(base_names, compact_actions)
    write_csv(output_dir / "candidate_examples.csv", [asdict(item) for item in items])
    write_csv(output_dir / "predictions.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "prediction_summary.csv", summary)
    write_csv(output_dir / "train_history.csv", history)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(items[0].features),
            "hidden_dim": config.hidden_dim,
            "compact_actions": compact_actions,
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "config": asdict(config),
        },
        output_dir / "output_level_risk_verifier.pt",
    )
    payload = {
        "config": asdict(config),
        "case_examples": len(examples),
        "candidate_examples": len(items),
        "train_cases": len(train_indices),
        "test_cases": len(test_indices),
        "safe_label_counts": dict(Counter(item.safe_vs_full for item in items)),
        "history_tail": history[-5:],
        "prediction_summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("split,policy,group,samples,score,kv_ratio,label_acc")
    for row in summary:
        if row["split"] == "test" and row["group"] == "__overall__":
            print(
                f"{row['split']},{row['policy']},{row['group']},{row['samples']},"
                f"{row['avg_score']:.4f},{row['avg_active_kv_ratio_vs_full']:.4f},{row['label_accuracy']:.4f}"
            )
    print(f"saved verifier to {output_dir / 'output_level_risk_verifier.pt'}")


if __name__ == "__main__":
    main()
