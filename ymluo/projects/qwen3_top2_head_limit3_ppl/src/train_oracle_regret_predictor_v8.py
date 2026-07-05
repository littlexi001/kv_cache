from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEPLOYABLE_MODES = [
    "full_kv_cache",
    "recent_kv_gather_topk",
    "lexical_kv_gather_topk",
    "learned_causal_kv_gather_topk",
    "set_utility_kv_gather_v7",
]
SLA_NAMES = ["quality", "balanced", "speed"]


@dataclass(frozen=True)
class Config:
    input_dir: str
    output_dir: str
    train_split: str
    eval_split: str
    epochs: int
    lr: float
    weight_decay: float
    hidden_dim: int
    seed: int
    feature_policy: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight oracle-regret predictor from V7 CSV outputs. "
            "Input: query/page aggregate features + candidate expert stats. "
            "Target: oracle objective regret per expert."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026070308)
    parser.add_argument(
        "--feature_policy",
        choices=["online_proxy", "learned_causal_proxy", "oracle_debug"],
        default="online_proxy",
        help=(
            "online_proxy keeps only selection-time proxy features. "
            "learned_causal_proxy first trains a page influence predictor from train-split causal labels "
            "and uses predicted page coverage as deployable candidate features. "
            "oracle_debug additionally uses teacher/gold outcome fields and is for ablation only."
        ),
    )
    return Config(**vars(parser.parse_args()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(round(f(row, key, float(default))))


def group_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["variant"]), int(row["task_id"]), int(row["budget"])


def page_task_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["variant"]), int(row["task_id"])


def parse_page_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    out: set[int] = set()
    for piece in re.split(r"[\s,;]+", text):
        if not piece:
            continue
        try:
            out.add(int(piece))
        except ValueError:
            continue
    return out


class PageInfluenceMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def binary_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(1 for value in labels if value > 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in ranked[index:end] if label > 0.5)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / max(1.0, positives * negatives)


def train_page_influence_predictor(page_rows: list[dict[str, str]], config: Config) -> dict[str, Any]:
    feature_names = sorted(
        key.removeprefix("feature_")
        for row in page_rows[:1]
        for key in row
        if key.startswith("feature_")
    )
    train_rows = [row for row in page_rows if str(row.get("split", "")) == config.train_split]
    if not train_rows:
        raise ValueError(f"no page train rows for split={config.train_split}")

    train_causal_positive = sum(i(row, "causal_label") for row in train_rows)
    target_key = "causal_label" if train_causal_positive > 0 else "weak_positive_label"

    def vectorize_pages(rows: list[dict[str, str]]) -> torch.Tensor:
        return torch.tensor(
            [[f(row, f"feature_{name}") for name in feature_names] for row in rows],
            dtype=torch.float32,
        )

    x = vectorize_pages(train_rows)
    y = torch.tensor([f(row, target_key) for row in train_rows], dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std

    torch.manual_seed(config.seed + 17)
    model = PageInfluenceMLP(len(feature_names), config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    positive = float(y.sum().item())
    negative = float(len(train_rows) - positive)
    pos_weight = torch.tensor([min(30.0, negative / max(1.0, positive))], dtype=torch.float32)
    epochs = max(120, min(config.epochs, 600))
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_norm)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    all_x = vectorize_pages(page_rows)
    all_x_norm = (all_x - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(model(all_x_norm)).detach().cpu().tolist()

    prediction_rows: list[dict[str, Any]] = []
    page_pred_by_task: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    labels_by_split: dict[str, list[float]] = defaultdict(list)
    scores_by_split: dict[str, list[float]] = defaultdict(list)
    for row, prob in zip(page_rows, probs):
        split = str(row.get("split", ""))
        label = f(row, target_key)
        labels_by_split[split].append(label)
        scores_by_split[split].append(float(prob))
        task_key = page_task_key(row)
        page_id = i(row, "page_id", -1)
        if page_id >= 0:
            page_pred_by_task[task_key][page_id] = float(prob)
        prediction_rows.append(
            {
                "variant": task_key[0],
                "task_id": task_key[1],
                "split": split,
                "page_id": page_id,
                "causal_label": i(row, "causal_label"),
                "weak_positive_label": i(row, "weak_positive_label"),
                "target_label": label,
                "predicted_influence_prob": float(prob),
            }
        )

    split_summary = {}
    for split, labels in sorted(labels_by_split.items()):
        scores = scores_by_split[split]
        split_summary[split] = {
            "rows": len(labels),
            "positive_rate": sum(1 for value in labels if value > 0.5) / max(1, len(labels)),
            "auc": binary_auc(labels, scores),
            "mean_predicted_prob": sum(scores) / max(1, len(scores)),
        }

    return {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_key": target_key,
        "train_rows": len(train_rows),
        "train_positive_rate": positive / max(1, len(train_rows)),
        "epochs": epochs,
        "split_summary": split_summary,
        "page_pred_by_task": dict(page_pred_by_task),
        "prediction_rows": prediction_rows,
    }


def selected_influence_features(
    strategy: dict[str, str],
    page_pred_by_task: dict[tuple[str, int], dict[int, float]],
) -> dict[str, float]:
    task_key = (str(strategy["variant"]), int(strategy["task_id"]))
    page_probs = page_pred_by_task.get(task_key, {})
    if not page_probs:
        return {}
    selected = parse_page_ids(strategy.get("selected_page_ids", ""))
    if str(strategy.get("mode", "")) == "full_kv_cache" and not selected:
        selected = set(page_probs)
    total = sum(max(0.0, prob) for prob in page_probs.values())
    selected_values = [max(0.0, page_probs[page_id]) for page_id in selected if page_id in page_probs]
    selected_sum = sum(selected_values)
    ranked = sorted(page_probs.items(), key=lambda item: item[1], reverse=True)
    top3 = {page_id for page_id, _ in ranked[:3]}
    top5 = {page_id for page_id, _ in ranked[:5]}
    top10 = {page_id for page_id, _ in ranked[:10]}
    top5_sum = sum(max(0.0, prob) for _, prob in ranked[:5])
    selected_top5_sum = sum(max(0.0, page_probs[page_id]) for page_id in selected & top5)
    unselected_values = [max(0.0, prob) for page_id, prob in page_probs.items() if page_id not in selected]
    selected_count = len(selected_values)
    return {
        "candidate_pred_influence_selected_sum": selected_sum,
        "candidate_pred_influence_selected_mean": selected_sum / max(1, selected_count),
        "candidate_pred_influence_selected_max": max(selected_values) if selected_values else 0.0,
        "candidate_pred_influence_recall": selected_sum / max(1e-8, total),
        "candidate_pred_influence_precision": selected_sum / max(1, selected_count),
        "candidate_pred_top3_coverage": len(selected & top3) / max(1, len(top3)),
        "candidate_pred_top5_coverage": len(selected & top5) / max(1, len(top5)),
        "candidate_pred_top10_coverage": len(selected & top10) / max(1, len(top10)),
        "candidate_pred_top5_mass_recall": selected_top5_sum / max(1e-8, top5_sum),
        "candidate_pred_top5_mass_miss": max(0.0, top5_sum - selected_top5_sum) / max(1e-8, top5_sum),
        "candidate_pred_selected_vs_unselected_max_margin": (max(selected_values) if selected_values else 0.0)
        - (max(unselected_values) if unselected_values else 0.0),
    }


def page_feature_aggregates(
    page_rows: list[dict[str, str]],
    feature_policy: str,
    page_pred_by_task: dict[tuple[str, int], dict[int, float]] | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in page_rows:
        grouped[page_task_key(row)].append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    include_oracle_debug = feature_policy == "oracle_debug"
    include_learned_causal = feature_policy == "learned_causal_proxy"
    feature_names = sorted(
        key.removeprefix("feature_")
        for row in page_rows[:1]
        for key in row
        if key.startswith("feature_")
    )
    for key, rows in grouped.items():
        n = max(1, len(rows))
        agg: dict[str, float] = {
            "page_count": float(n),
            "mean_page_tokens": sum(f(row, "page_token_count") for row in rows) / n,
        }
        positives: list[dict[str, str]] = []
        if include_oracle_debug:
            positives = [row for row in rows if i(row, "causal_label") == 1]
            answer_hits = [row for row in rows if i(row, "answer_text_hit") == 1]
            deltas = [f(row, "loss_delta") for row in rows]
            sorted_deltas = sorted(deltas, reverse=True)
            top_delta_sum = sum(max(0.0, value) for value in sorted_deltas[:5])
            total_pos_delta = sum(max(0.0, value) for value in deltas)
            agg.update(
                {
                    "causal_positive_rate": len(positives) / n,
                    "answer_hit_rate": len(answer_hits) / n,
                    "mean_loss_delta": sum(deltas) / n,
                    "max_loss_delta": max(deltas) if deltas else 0.0,
                    "top5_positive_delta_sum": top_delta_sum,
                    "total_positive_delta": total_pos_delta,
                    "positive_delta_concentration": top_delta_sum / max(1e-8, total_pos_delta),
            }
        )
        if include_learned_causal and page_pred_by_task:
            probs = [page_pred_by_task.get(key, {}).get(i(row, "page_id", -1), 0.0) for row in rows]
            sorted_probs = sorted((max(0.0, value) for value in probs), reverse=True)
            total_prob = sum(sorted_probs)
            top3_sum = sum(sorted_probs[:3])
            top5_sum = sum(sorted_probs[:5])
            top10_sum = sum(sorted_probs[:10])
            agg.update(
                {
                    "pred_influence_mean": total_prob / n,
                    "pred_influence_max": sorted_probs[0] if sorted_probs else 0.0,
                    "pred_influence_top3_sum": top3_sum,
                    "pred_influence_top5_sum": top5_sum,
                    "pred_influence_top10_sum": top10_sum,
                    "pred_influence_total": total_prob,
                    "pred_influence_top5_concentration": top5_sum / max(1e-8, total_prob),
                    "pred_influence_top10_concentration": top10_sum / max(1e-8, total_prob),
                    "pred_influence_prob_over_50_rate": sum(1 for value in probs if value >= 0.5) / n,
                }
            )
        for name in feature_names:
            values = [f(row, f"feature_{name}") for row in rows]
            agg[f"page_mean_{name}"] = sum(values) / n
            agg[f"page_max_{name}"] = max(values) if values else 0.0
            if include_oracle_debug:
                if positives:
                    agg[f"pos_mean_{name}"] = sum(f(row, f"feature_{name}") for row in positives) / len(positives)
                else:
                    agg[f"pos_mean_{name}"] = 0.0
        out[key] = agg
    return out


def build_dataset(
    strategy_rows: list[dict[str, str]],
    regret_rows: list[dict[str, str]],
    page_aggs: dict[tuple[str, int], dict[str, float]],
    sla: str,
    feature_policy: str,
    page_pred_by_task: dict[tuple[str, int], dict[int, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    strategy_by_key_mode = {(group_key(row), str(row["mode"])): row for row in strategy_rows if str(row["mode"]) in DEPLOYABLE_MODES}
    regret_by_key_mode = {
        (group_key(row), str(row["mode"])): row
        for row in regret_rows
        if str(row["sla"]) == sla and str(row["mode"]) in DEPLOYABLE_MODES
    }
    rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
    for (key, mode), strategy in strategy_by_key_mode.items():
        regret = regret_by_key_mode.get((key, mode))
        if regret is None:
            continue
        variant, task_id, budget = key
        page_agg = page_aggs.get((variant, task_id), {})
        page_count = max(1.0, f(strategy, "page_count", page_agg.get("page_count", 1.0)))
        selected_pages = f(strategy, "selected_pages")
        keep_fraction = f(strategy, "keep_fraction", 1.0)
        features: dict[str, float] = {
            "budget_log": math.log1p(budget),
            "candidate_keep_fraction": keep_fraction,
            "candidate_selected_page_ratio": selected_pages / page_count,
            "candidate_visible_tokens_log": math.log1p(f(strategy, "visible_tokens")),
            "candidate_estimated_causal_recall": f(strategy, "estimated_causal_recall"),
            "candidate_selected_pages": selected_pages,
            "candidate_selected_page_tokens_log": math.log1p(f(strategy, "selected_page_tokens")),
            "page_count_log": math.log1p(page_count),
            "query_tokens_log": math.log1p(f(strategy, "query_tokens")),
            "raw_prefix_tokens_log": math.log1p(f(strategy, "raw_prefix_tokens")),
            "raw_prompt_tokens_log": math.log1p(f(strategy, "raw_prompt_tokens")),
        }
        if feature_policy == "oracle_debug":
            features.update(
                {
                    "candidate_correct_proxy": f(strategy, "correct"),
                    "candidate_margin": f(strategy, "margin"),
                    "candidate_online_seconds": f(strategy, "online_seconds"),
                    "candidate_query_eval_seconds": f(strategy, "query_eval_seconds"),
                    "candidate_kv_gather_seconds": f(strategy, "kv_gather_seconds"),
                    "candidate_evidence_hit": f(strategy, "evidence_hit"),
                    "candidate_causal_label_recall": f(strategy, "causal_label_recall"),
                    "candidate_causal_label_precision": f(strategy, "causal_label_precision"),
                    "candidate_causal_mass_recall": f(strategy, "causal_positive_mass_recall"),
                    "teacher_correct": f(strategy, "teacher_correct"),
                    "teacher_ppl_log": math.log(max(1e-8, f(strategy, "teacher_gold_label_ppl", 1.0))),
                }
            )
        if feature_policy == "learned_causal_proxy" and page_pred_by_task:
            features.update(selected_influence_features(strategy, page_pred_by_task))
        for candidate in DEPLOYABLE_MODES:
            features[f"mode_{candidate}"] = float(mode == candidate)
        for variant_name in sorted({row["variant"] for row in strategy_rows}):
            features[f"variant_{variant_name}"] = float(variant == variant_name)
        for name, value in page_agg.items():
            features[f"agg_{name}"] = value

        feature_names.update(features)
        rows.append(
            {
                "variant": variant,
                "task_id": task_id,
                "budget": budget,
                "split": strategy["split"],
                "sla": sla,
                "mode": mode,
                "features": features,
                "objective_regret": f(regret, "deployable_objective_regret"),
                "is_oracle": i(regret, "is_deployable_oracle"),
                "actual_row": strategy,
                "oracle_mode": regret["deployable_oracle_mode"],
            }
        )
    return rows, sorted(feature_names)


class RegretMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def vectorize(rows: list[dict[str, Any]], feature_names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=torch.float32,
    )


def train_model(rows: list[dict[str, Any]], feature_names: list[str], config: Config) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == config.train_split]
    if not train_rows:
        raise ValueError(f"no train rows for split={config.train_split}")
    x = vectorize(train_rows, feature_names)
    y_raw = torch.tensor([float(row["objective_regret"]) for row in train_rows], dtype=torch.float32)
    y = torch.log1p(torch.clamp(y_raw, min=0.0))
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std

    torch.manual_seed(config.seed)
    model = RegretMLP(len(feature_names), config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_norm)
        # Weight oracle and near-oracle rows a bit more; ranking among good candidates matters most.
        weights = 1.0 + 2.0 * (y_raw <= 1e-6).float() + 1.0 * (y_raw < 1.0).float()
        loss = (weights * F.smooth_l1_loss(pred, y, reduction="none")).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = model(x_norm)
        train_mae = float((train_pred - y).abs().mean().item())
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "train_rows": len(train_rows),
        "train_mae_log1p_regret": train_mae,
    }


def predict_rows(rows: list[dict[str, Any]], feature_names: list[str], model_info: dict[str, Any]) -> None:
    model: RegretMLP = model_info["model"]
    x = vectorize(rows, feature_names)
    x_norm = (x - model_info["mean"]) / model_info["std"]
    with torch.no_grad():
        pred = model(x_norm).detach().cpu().tolist()
    for row, value in zip(rows, pred):
        row["predicted_log_regret"] = float(value)
        row["predicted_regret"] = math.expm1(max(-20.0, min(20.0, float(value))))


def summarize_selected(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = max(1, len(rows))
    token_count = sum(i(row["actual_row"], "gold_label_tokens", 1) for row in rows)
    loss = sum(f(row["actual_row"], "gold_label_loss") * i(row["actual_row"], "gold_label_tokens", 1) for row in rows)
    return {
        "selector": label,
        "cases": len(rows),
        "accuracy": sum(i(row["actual_row"], "correct") for row in rows) / n,
        "gold_label_ppl": math.exp(min(80.0, loss / max(1, token_count))),
        "mean_keep_fraction": sum(f(row["actual_row"], "keep_fraction", 1.0) for row in rows) / n,
        "mean_online_seconds": sum(f(row["actual_row"], "online_seconds") for row in rows) / n,
        "mean_total_seconds": sum(f(row["actual_row"], "total_seconds") for row in rows) / n,
        "mean_actual_regret": sum(float(row["objective_regret"]) for row in rows) / n,
        "oracle_match_rate": sum(int(row["is_oracle"]) for row in rows) / n,
        "selection_mode_counts": json.dumps(
            dict(sorted((mode, sum(1 for row in rows if row["mode"] == mode)) for mode in DEPLOYABLE_MODES)),
            sort_keys=True,
        ),
    }


def evaluate_selector(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[(row["variant"], row["task_id"], row["budget"], row["sla"])].append(row)

    selected_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for key, candidates in sorted(grouped.items()):
        selected = min(candidates, key=lambda row: (row["predicted_regret"], row["mode"]))
        oracle = min(candidates, key=lambda row: (row["objective_regret"], row["mode"]))
        selected_rows.append(selected)
        case_rows.append(
            {
                "variant": key[0],
                "task_id": key[1],
                "budget": key[2],
                "sla": key[3],
                "selected_mode": selected["mode"],
                "oracle_mode": oracle["mode"],
                "selected_correct": i(selected["actual_row"], "correct"),
                "oracle_correct": i(oracle["actual_row"], "correct"),
                "selected_ppl": f(selected["actual_row"], "gold_label_ppl"),
                "selected_keep_fraction": f(selected["actual_row"], "keep_fraction", 1.0),
                "selected_online_seconds": f(selected["actual_row"], "online_seconds"),
                "selected_actual_regret": selected["objective_regret"],
                "selected_predicted_regret": selected["predicted_regret"],
                "oracle_match": int(selected["mode"] == oracle["mode"]),
            }
        )
    return selected_rows, case_rows


def baseline_rows(rows: list[dict[str, Any]], split: str, mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split and row["mode"] == mode]


def oracle_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[(row["variant"], row["task_id"], row["budget"], row["sla"])].append(row)
    return [min(candidates, key=lambda row: (row["objective_regret"], row["mode"])) for candidates in grouped.values()]


def main() -> None:
    config = parse_args()
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    strategy_rows = read_csv(input_dir / "strategy_results.csv")
    regret_rows = read_csv(input_dir / "oracle_regret_labels.csv")
    page_rows = read_csv(input_dir / "causal_page_influence_labels.csv")
    page_influence_summary: dict[str, Any] | None = None
    page_pred_by_task: dict[tuple[str, int], dict[int, float]] = {}
    if config.feature_policy == "learned_causal_proxy":
        page_influence_info = train_page_influence_predictor(page_rows, config)
        page_pred_by_task = page_influence_info["page_pred_by_task"]
        write_csv(output_dir / "learned_causal_page_predictions.csv", page_influence_info["prediction_rows"])
        page_influence_summary = {
            key: value
            for key, value in page_influence_info.items()
            if key not in {"page_pred_by_task", "prediction_rows", "feature_names"}
        }
    page_aggs = page_feature_aggregates(page_rows, config.feature_policy, page_pred_by_task)

    all_eval_rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    feature_names_by_sla: dict[str, list[str]] = {}

    for sla in SLA_NAMES:
        rows, feature_names = build_dataset(
            strategy_rows,
            regret_rows,
            page_aggs,
            sla,
            config.feature_policy,
            page_pred_by_task,
        )
        model_info = train_model(rows, feature_names, config)
        predict_rows(rows, feature_names, model_info)
        selected, case_rows = evaluate_selector(rows, config.eval_split)
        all_eval_rows.extend({**row, "selector": "predicted_regret_v8"} for row in selected)
        all_case_rows.extend(case_rows)
        feature_names_by_sla[sla] = feature_names
        model_summaries[sla] = {
            "train_rows": model_info["train_rows"],
            "train_mae_log1p_regret": model_info["train_mae_log1p_regret"],
            "feature_count": len(feature_names),
        }
        if page_influence_summary is not None:
            model_summaries["page_influence_predictor"] = page_influence_summary

        for mode in DEPLOYABLE_MODES:
            all_eval_rows.extend({**row, "selector": mode} for row in baseline_rows(rows, config.eval_split, mode))
        all_eval_rows.extend({**row, "selector": "deployable_oracle"} for row in oracle_rows(rows, config.eval_split))

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_eval_rows:
        grouped[(row["selector"], row["sla"])].append(row)
    for (selector, sla), rows in sorted(grouped.items()):
        summary = summarize_selected(rows, selector)
        summary["sla"] = sla
        summary["feature_policy"] = config.feature_policy
        summary_rows.append(summary)

    write_csv(output_dir / "predicted_regret_cases.csv", all_case_rows)
    write_csv(output_dir / "predicted_regret_summary.csv", summary_rows)
    (output_dir / "model_summary.json").write_text(json.dumps(model_summaries, indent=2), encoding="utf-8")
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names_by_sla, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows, "models": model_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
