from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_stepwise_set_utility import mcnemar_exact_p
from rerank_sparse_candidate_blocks_svd import rank_ids, target_rank
from train_pairwise_qk_passage_head import column_zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dev-only diagnostic head for dynamic bridge-state pools."
    )
    parser.add_argument("--train_rows_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dynamic_candidate_features(row: dict[str, Any]) -> np.ndarray:
    passage = np.asarray(row["hypothesis_passage_scores"], dtype=np.float64)
    bm25 = np.asarray(row["hypothesis_bm25_scores"], dtype=np.float64)
    ranks = np.asarray(row["hypothesis_candidate_ranks"], dtype=np.float64)
    selected = int(row["selected_index"])
    bm25_z = np.stack([column_zscore(values[:, None])[:, 0] for values in bm25])
    reciprocal_rank = np.sum(1.0 / (60.0 + ranks), axis=0)
    top3_support = np.mean(ranks <= 3, axis=0)
    top16_support = np.mean(ranks <= 16, axis=0)
    return np.stack(
        [
            passage[selected],
            passage.max(axis=0),
            passage.mean(axis=0),
            passage.std(axis=0),
            bm25_z[selected],
            bm25_z.max(axis=0),
            bm25_z.mean(axis=0),
            reciprocal_rank,
            top3_support,
            top16_support,
        ],
        axis=1,
    )


def pairwise_examples(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, int]:
    examples = []
    labels = []
    reachable = 0
    for row in rows:
        candidates = [int(item) for item in row["candidate_pool"]]
        target = int(row["target_block_id"])
        if target not in candidates:
            continue
        reachable += 1
        features = dynamic_candidate_features(row)
        positive = candidates.index(target)
        for negative in range(len(candidates)):
            if negative == positive:
                continue
            difference = features[positive] - features[negative]
            examples.extend([difference, -difference])
            labels.extend([1, 0])
    return np.asarray(examples), np.asarray(labels), reachable


def evaluate(
    rows: Sequence[dict[str, Any]], model: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_rows = []
    for row in rows:
        candidates = [int(item) for item in row["candidate_pool"]]
        scores = model.decision_function(dynamic_candidate_features(row))
        ranked = rank_ids(candidates, scores.tolist())
        output_rows.append(
            {
                "query_id": int(row["query_id"]),
                "target_block_id": int(row["target_block_id"]),
                "selected_bm25_rank": int(row["selected_bm25_rank"]),
                "dynamic_head_rank": target_rank(ranked, int(row["target_block_id"])),
                "dynamic_head_candidates": ranked,
            }
        )
    baseline = [0 < row["selected_bm25_rank"] <= 3 for row in output_rows]
    learned = [0 < row["dynamic_head_rank"] <= 3 for row in output_rows]
    wins = sum(new and not old for new, old in zip(learned, baseline))
    losses = sum(old and not new for new, old in zip(learned, baseline))
    summary = {
        "queries": len(output_rows),
        "selected_bm25_recall_at_3": sum(baseline) / len(output_rows),
        "dynamic_head_recall_at_1": sum(
            row["dynamic_head_rank"] == 1 for row in output_rows
        )
        / len(output_rows),
        "dynamic_head_recall_at_3": sum(learned) / len(output_rows),
        "dynamic_head_recall_at_16": sum(
            0 < row["dynamic_head_rank"] <= 16 for row in output_rows
        )
        / len(output_rows),
        "wins_losses_vs_selected_bm25": [wins, losses],
        "mcnemar_p_vs_selected_bm25": mcnemar_exact_p(wins, losses),
    }
    return summary, output_rows


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(Path(args.train_rows_path))
    test_rows = read_jsonl(Path(args.test_rows_path))
    train_x, train_y, reachable = pairwise_examples(train_rows)
    if not len(train_y):
        raise ValueError("no reachable training candidates")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=17),
    )
    model.fit(train_x, train_y)
    train_summary, _train_output = evaluate(train_rows, model)
    test_summary, test_output = evaluate(test_rows, model)
    logistic = model.named_steps["logisticregression"]
    payload = {
        "source": "dev-trained pairwise diagnostic over dynamic bridge-state features",
        "selection_uses_test_labels": False,
        "train_reachable_queries": reachable,
        "pairwise_examples": int(len(train_y)),
        "standardized_coefficients": [
            float(item) for item in logistic.coef_[0]
        ],
        "feature_names": [
            "selected_passage",
            "max_passage",
            "mean_passage",
            "std_passage",
            "selected_bm25_z",
            "max_bm25_z",
            "mean_bm25_z",
            "reciprocal_rank_consensus",
            "top3_support",
            "top16_support",
        ],
        "train": train_summary,
        "test": test_summary,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows_path = output_path.with_name(f"{output_path.stem}_test_rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in test_output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
