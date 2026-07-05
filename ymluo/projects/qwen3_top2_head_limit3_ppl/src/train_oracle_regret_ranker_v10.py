from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_oracle_regret_predictor_v8 as v8  # noqa: E402


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
    pair_margin: float
    pair_min_gap: float
    listwise_weight: float
    pointwise_weight: float
    full_fallback_margins: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "V10 oracle-regret ranker. It reuses V8 online/learned causal features, "
            "but trains with within-query pairwise/listwise expert ranking and evaluates "
            "risk-calibrated full fallback margins."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026070410)
    parser.add_argument(
        "--feature_policy",
        choices=["online_proxy", "learned_causal_proxy", "oracle_debug"],
        default="learned_causal_proxy",
    )
    parser.add_argument(
        "--pair_margin",
        type=float,
        default=0.20,
        help="Ranking margin in score units. Lower scores are better.",
    )
    parser.add_argument(
        "--pair_min_gap",
        type=float,
        default=1e-6,
        help="Only form training pairs when objective regret differs by at least this value.",
    )
    parser.add_argument("--listwise_weight", type=float, default=0.25)
    parser.add_argument("--pointwise_weight", type=float, default=0.05)
    parser.add_argument(
        "--full_fallback_margins",
        default="0,0.05,0.1,0.2,0.35,0.5",
        help=(
            "Comma-separated fallback margins. If the best non-full candidate is not better "
            "than full by this predicted score margin, select full."
        ),
    )
    return Config(**vars(parser.parse_args()))


class RankerMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(float(piece))
    return out or [0.0]


def vectorize(rows: list[dict[str, Any]], feature_names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=torch.float32,
    )


def row_group_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return str(row["variant"]), int(row["task_id"]), int(row["budget"]), str(row["sla"])


def make_train_pairs(rows: list[dict[str, Any]], min_gap: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_good: list[int] = []
    pair_bad: list[int] = []
    pair_weight: list[float] = []
    grouped: dict[tuple[str, int, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row_group_key(row)].append(index)

    for indices in grouped.values():
        ordered = sorted(indices, key=lambda idx: (float(rows[idx]["objective_regret"]), rows[idx]["mode"]))
        for left_pos, good_idx in enumerate(ordered):
            good_regret = float(rows[good_idx]["objective_regret"])
            for bad_idx in ordered[left_pos + 1 :]:
                bad_regret = float(rows[bad_idx]["objective_regret"])
                gap = bad_regret - good_regret
                if gap < min_gap:
                    continue
                pair_good.append(good_idx)
                pair_bad.append(bad_idx)
                pair_weight.append(max(1.0, min(8.0, math.log1p(gap))))
    if not pair_good:
        raise ValueError("no train pairs were formed; check objective_regret labels")
    return (
        torch.tensor(pair_good, dtype=torch.long),
        torch.tensor(pair_bad, dtype=torch.long),
        torch.tensor(pair_weight, dtype=torch.float32),
    )


def make_group_indices(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[tuple[str, int, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row_group_key(row)].append(index)
    return list(grouped.values())


def train_ranker(rows: list[dict[str, Any]], feature_names: list[str], config: Config) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == config.train_split]
    if not train_rows:
        raise ValueError(f"no train rows for split={config.train_split}")

    x = vectorize(train_rows, feature_names)
    y_regret = torch.tensor([float(row["objective_regret"]) for row in train_rows], dtype=torch.float32)
    y_log = torch.log1p(torch.clamp(y_regret, min=0.0))
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std

    good_idx, bad_idx, pair_weight = make_train_pairs(train_rows, config.pair_min_gap)
    groups = make_group_indices(train_rows)

    torch.manual_seed(config.seed)
    model = RankerMLP(len(feature_names), config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        scores = model(x_norm)
        pair_loss = (
            pair_weight
            * F.softplus(scores[good_idx] - scores[bad_idx] + float(config.pair_margin))
        ).mean()
        loss = pair_loss
        if config.listwise_weight > 0:
            list_losses = []
            for indices in groups:
                idx = torch.tensor(indices, dtype=torch.long)
                target = torch.softmax(-y_log[idx], dim=0)
                log_prob = torch.log_softmax(-scores[idx], dim=0)
                list_losses.append(-(target * log_prob).sum())
            loss = loss + config.listwise_weight * torch.stack(list_losses).mean()
        if config.pointwise_weight > 0:
            centered = scores - scores.mean()
            target = y_log - y_log.mean()
            loss = loss + config.pointwise_weight * F.smooth_l1_loss(centered, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores = model(x_norm)
        pair_acc = float((scores[good_idx] < scores[bad_idx]).float().mean().item())
        best_match = 0
        for indices in groups:
            pred_idx = min(indices, key=lambda idx: (float(scores[idx].item()), train_rows[idx]["mode"]))
            oracle_idx = min(indices, key=lambda idx: (float(train_rows[idx]["objective_regret"]), train_rows[idx]["mode"]))
            best_match += int(pred_idx == oracle_idx)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "train_rows": len(train_rows),
        "train_pairs": int(len(good_idx)),
        "train_pair_accuracy": pair_acc,
        "train_group_oracle_match": best_match / max(1, len(groups)),
    }


def predict_rows(rows: list[dict[str, Any]], feature_names: list[str], model_info: dict[str, Any]) -> None:
    model: RankerMLP = model_info["model"]
    x = vectorize(rows, feature_names)
    x_norm = (x - model_info["mean"]) / model_info["std"]
    with torch.no_grad():
        scores = model(x_norm).detach().cpu().tolist()
    for row, score in zip(rows, scores):
        row["predicted_rank_score"] = float(score)


def select_with_optional_full_fallback(
    candidates: list[dict[str, Any]],
    full_fallback_margin: float | None,
) -> dict[str, Any]:
    selected = min(candidates, key=lambda row: (float(row["predicted_rank_score"]), row["mode"]))
    if full_fallback_margin is None:
        return selected
    full = next((row for row in candidates if row["mode"] == "full_kv_cache"), None)
    if full is None or selected["mode"] == "full_kv_cache":
        return selected
    full_score = float(full["predicted_rank_score"])
    selected_score = float(selected["predicted_rank_score"])
    if full_score - selected_score <= full_fallback_margin:
        return full
    return selected


def evaluate_selector(
    rows: list[dict[str, Any]],
    split: str,
    selector: str,
    full_fallback_margin: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row_group_key(row)].append(row)

    selected_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for key, candidates in sorted(grouped.items()):
        selected = select_with_optional_full_fallback(candidates, full_fallback_margin)
        oracle = min(candidates, key=lambda row: (float(row["objective_regret"]), row["mode"]))
        full = next((row for row in candidates if row["mode"] == "full_kv_cache"), None)
        selected_rows.append(selected)
        case_rows.append(
            {
                "variant": key[0],
                "task_id": key[1],
                "budget": key[2],
                "sla": key[3],
                "selector": selector,
                "selected_mode": selected["mode"],
                "oracle_mode": oracle["mode"],
                "selected_correct": v8.i(selected["actual_row"], "correct"),
                "oracle_correct": v8.i(oracle["actual_row"], "correct"),
                "selected_ppl": v8.f(selected["actual_row"], "gold_label_ppl"),
                "selected_keep_fraction": v8.f(selected["actual_row"], "keep_fraction", 1.0),
                "selected_online_seconds": v8.f(selected["actual_row"], "online_seconds"),
                "selected_actual_regret": selected["objective_regret"],
                "selected_rank_score": selected["predicted_rank_score"],
                "full_rank_score": full["predicted_rank_score"] if full else "",
                "full_fallback_margin": "" if full_fallback_margin is None else full_fallback_margin,
                "oracle_match": int(selected["mode"] == oracle["mode"]),
            }
        )
    return selected_rows, case_rows


def summarize_selected(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    summary = v8.summarize_selected(rows, selector)
    return summary


def baseline_rows(rows: list[dict[str, Any]], split: str, mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split and row["mode"] == mode]


def oracle_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row_group_key(row)].append(row)
    return [min(candidates, key=lambda row: (float(row["objective_regret"]), row["mode"])) for candidates in grouped.values()]


def prepare_page_features(config: Config, output_dir: Path) -> tuple[
    dict[tuple[str, int], dict[str, float]],
    dict[tuple[str, int], dict[int, float]],
    dict[str, Any] | None,
]:
    input_dir = Path(config.input_dir)
    page_rows = v8.read_csv(input_dir / "causal_page_influence_labels.csv")
    page_pred_by_task: dict[tuple[str, int], dict[int, float]] = {}
    page_influence_summary: dict[str, Any] | None = None
    if config.feature_policy == "learned_causal_proxy":
        page_config = v8.Config(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            train_split=config.train_split,
            eval_split=config.eval_split,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
            feature_policy=config.feature_policy,
        )
        page_info = v8.train_page_influence_predictor(page_rows, page_config)
        page_pred_by_task = page_info["page_pred_by_task"]
        v8.write_csv(output_dir / "learned_causal_page_predictions.csv", page_info["prediction_rows"])
        page_influence_summary = {
            key: value
            for key, value in page_info.items()
            if key not in {"page_pred_by_task", "prediction_rows", "feature_names"}
        }
    page_aggs = v8.page_feature_aggregates(page_rows, config.feature_policy, page_pred_by_task)
    return page_aggs, page_pred_by_task, page_influence_summary


def main() -> None:
    config = parse_args()
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    strategy_rows = v8.read_csv(input_dir / "strategy_results.csv")
    regret_rows = v8.read_csv(input_dir / "oracle_regret_labels.csv")
    page_aggs, page_pred_by_task, page_influence_summary = prepare_page_features(config, output_dir)
    fallback_margins = parse_float_list(config.full_fallback_margins)

    all_eval_rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    feature_names_by_sla: dict[str, list[str]] = {}

    for sla in v8.SLA_NAMES:
        rows, feature_names = v8.build_dataset(
            strategy_rows,
            regret_rows,
            page_aggs,
            sla,
            config.feature_policy,
            page_pred_by_task,
        )
        model_info = train_ranker(rows, feature_names, config)
        predict_rows(rows, feature_names, model_info)
        feature_names_by_sla[sla] = feature_names
        model_summaries[sla] = {
            "train_rows": model_info["train_rows"],
            "train_pairs": model_info["train_pairs"],
            "train_pair_accuracy": model_info["train_pair_accuracy"],
            "train_group_oracle_match": model_info["train_group_oracle_match"],
            "feature_count": len(feature_names),
        }
        if page_influence_summary is not None:
            model_summaries["page_influence_predictor"] = page_influence_summary

        selected, case_rows = evaluate_selector(rows, config.eval_split, "pairwise_ranker_v10", None)
        all_eval_rows.extend({**row, "selector": "pairwise_ranker_v10"} for row in selected)
        all_case_rows.extend(case_rows)
        for margin in fallback_margins:
            selector = f"pairwise_ranker_v10_fullfb_{margin:g}"
            selected_fb, case_rows_fb = evaluate_selector(rows, config.eval_split, selector, margin)
            all_eval_rows.extend({**row, "selector": selector} for row in selected_fb)
            all_case_rows.extend(case_rows_fb)

        for mode in v8.DEPLOYABLE_MODES:
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

    v8.write_csv(output_dir / "pairwise_ranker_cases.csv", all_case_rows)
    v8.write_csv(output_dir / "pairwise_ranker_summary.csv", summary_rows)
    (output_dir / "model_summary.json").write_text(json.dumps(model_summaries, indent=2), encoding="utf-8")
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names_by_sla, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows, "models": model_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
