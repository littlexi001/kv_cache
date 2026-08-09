from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_stepwise_set_utility import mcnemar_exact_p
from rerank_sparse_candidate_blocks_svd import rank_ids, target_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny pairwise passage head over BM25 and per-channel QK scores."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--train_splits",
        default="train",
        help="Comma-separated splits used to fit the head; test remains evaluation-only.",
    )
    parser.add_argument(
        "--export_method",
        choices=["", "bm25", "full128", "svd", "both"],
        default="",
    )
    parser.add_argument("--export_split", default="test")
    parser.add_argument("--ranked_rows_path", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def column_zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    return (matrix - mean) / np.maximum(scale, 1.0e-8)


def candidate_features(row: dict[str, Any], method: str) -> np.ndarray:
    bm25 = column_zscore(np.asarray(row["bm25_scores"], dtype=np.float64)[:, None])
    if method == "bm25":
        return bm25
    if method == "both":
        profiles = np.concatenate(
            [
                np.asarray(row["full128_profile_scores"], dtype=np.float64),
                np.asarray(row["svd_profile_scores"], dtype=np.float64),
            ],
            axis=1,
        )
    else:
        profiles = np.asarray(row[f"{method}_profile_scores"], dtype=np.float64)
    return np.concatenate([bm25, column_zscore(profiles)], axis=1)


def pairwise_examples(
    rows: list[dict[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, int]:
    examples = []
    labels = []
    reachable = 0
    for row in rows:
        candidates = [int(item) for item in row["candidate_candidates"]]
        target = int(row["target_block_id"])
        if target not in candidates:
            continue
        reachable += 1
        features = candidate_features(row, method)
        positive = candidates.index(target)
        for negative in range(len(candidates)):
            if negative == positive:
                continue
            difference = features[positive] - features[negative]
            examples.extend([difference, -difference])
            labels.extend([1, 0])
    if not examples:
        raise ValueError("no reachable pairwise train examples")
    return np.stack(examples), np.asarray(labels, dtype=np.int64), reachable


def runtime_passage_scores(
    features: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    mean = np.asarray(parameters["feature_mean"], dtype=np.float64)
    scale = np.asarray(parameters["feature_scale"], dtype=np.float64)
    weight = np.asarray(parameters["linear_weight"], dtype=np.float64)
    return (
        ((features - mean) / np.maximum(scale, 1.0e-12)) @ weight
        + float(parameters.get("linear_intercept", 0.0))
    )


def evaluate(
    rows: list[dict[str, Any]], method: str, model: Any
) -> dict[str, Any]:
    baseline_hits = []
    learned_hits = []
    learned_ranks = []
    for row in rows:
        candidates = [int(item) for item in row["candidate_candidates"]]
        scores = model.decision_function(candidate_features(row, method))
        ranked = rank_ids(candidates, scores.tolist())
        target = int(row["target_block_id"])
        baseline_rank = int(row["candidate_rank"])
        learned_rank = target_rank(ranked, target)
        baseline_hits.append(0 < baseline_rank <= 3)
        learned_hits.append(0 < learned_rank <= 3)
        learned_ranks.append(learned_rank)
    wins = sum(
        learned and not baseline
        for baseline, learned in zip(baseline_hits, learned_hits, strict=True)
    )
    losses = sum(
        baseline and not learned
        for baseline, learned in zip(baseline_hits, learned_hits, strict=True)
    )
    reachable = [rank for rank in learned_ranks if rank > 0]
    return {
        "steps": len(rows),
        "bm25_recall_at_3": statistics.fmean(baseline_hits),
        "learned_recall_at_1": statistics.fmean(rank == 1 for rank in learned_ranks),
        "learned_recall_at_3": statistics.fmean(learned_hits),
        "learned_recall_at_16": statistics.fmean(rank > 0 for rank in learned_ranks),
        "learned_conditional_mrr": (
            statistics.fmean(1.0 / rank for rank in reachable) if reachable else 0.0
        ),
        "wins_losses": [wins, losses],
        "mcnemar_p": mcnemar_exact_p(wins, losses),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_path))
    train_splits = {
        item.strip() for item in args.train_splits.split(",") if item.strip()
    }
    if not train_splits or "test" in train_splits:
        raise ValueError("train_splits must be non-empty and must not include test")
    payload: dict[str, Any] = {
        "source": "train-only pairwise linear passage head over coarse and internal QK scores",
        "selection_uses_gold": False,
        "train_labels_used_for_pairwise_head_only": True,
        "train_splits": sorted(train_splits),
        "methods": {},
    }
    step_types = sorted({str(row["step_type"]) for row in rows})
    for method in ("bm25", "full128", "svd", "both"):
        method_payload: dict[str, Any] = {}
        for step_type in step_types:
            groups = {
                split: [
                    row
                    for row in rows
                    if str(row["split"]) == split
                    and str(row["step_type"]) == step_type
                ]
                for split in ("train", "dev", "test")
            }
            fit_rows = [
                row for split in sorted(train_splits) for row in groups.get(split, [])
            ]
            train_x, train_y, reachable = pairwise_examples(fit_rows, method)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    fit_intercept=False,
                    max_iter=1000,
                    random_state=17,
                ),
            )
            model.fit(train_x, train_y)
            scaler = model.named_steps["standardscaler"]
            coefficients = model.named_steps["logisticregression"].coef_[0]
            method_payload[step_type] = {
                "train_reachable_queries": reachable,
                "pairwise_examples": int(len(train_y)),
                "standardized_coefficients": [float(item) for item in coefficients],
                "runtime_parameters": {
                    "feature_mean": [float(item) for item in scaler.mean_],
                    "feature_scale": [float(item) for item in scaler.scale_],
                    "linear_weight": [float(item) for item in coefficients],
                    "linear_intercept": 0.0,
                },
                "evaluations": {
                    split: evaluate(group, method, model)
                    for split, group in groups.items()
                },
            }
        payload["methods"][method] = method_payload
    if bool(args.export_method) != bool(args.ranked_rows_path):
        raise ValueError("export_method and ranked_rows_path must be provided together")
    if args.export_method:
        export_rows = []
        for row in rows:
            if str(row["split"]) != args.export_split:
                continue
            method = args.export_method
            step_type = str(row["step_type"])
            parameters = payload["methods"][method][step_type]["runtime_parameters"]
            candidates = [int(item) for item in row["candidate_candidates"]]
            scores = runtime_passage_scores(
                candidate_features(row, method), parameters
            )
            ranked = rank_ids(candidates, scores.tolist())
            export_rows.append(
                {
                    "query_id": int(row["query_id"]),
                    "step_index": int(row["step_index"]),
                    "split": str(row["split"]),
                    "step_type": str(row["step_type"]),
                    "selection_uses_gold": False,
                    "training_uses_gold": True,
                    "passage_head_method": method,
                    "passage_head_candidates": ranked,
                    "passage_head_scores": [
                        float(scores[candidates.index(block_id)]) for block_id in ranked
                    ],
                    "target_block_id": int(row["target_block_id"]),
                    "passage_head_rank": target_rank(
                        ranked, int(row["target_block_id"])
                    ),
                }
            )
        ranked_rows_path = Path(args.ranked_rows_path)
        ranked_rows_path.parent.mkdir(parents=True, exist_ok=True)
        with ranked_rows_path.open("w", encoding="utf-8") as handle:
            for row in export_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload["export"] = {
            "method": args.export_method,
            "split": args.export_split,
            "rows": len(export_rows),
            "ranked_rows_path": str(ranked_rows_path),
        }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
