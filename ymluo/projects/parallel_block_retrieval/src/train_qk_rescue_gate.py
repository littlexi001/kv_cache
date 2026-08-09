from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a train-only confidence gate for rare QK rescue actions."
    )
    parser.add_argument("--rerank_rows_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_gap(scores: Sequence[float], left: int, right: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) <= max(left, right):
        return 0.0
    return float((values[left] - values[right]) / max(float(values.std()), 1.0e-8))


def ranked_gap(scores: Sequence[float], left: int, right: int) -> float:
    return normalized_gap(sorted(scores, reverse=True), left, right)


def top_overlap(left: Sequence[int], right: Sequence[int], count: int) -> float:
    return len(set(left[:count]) & set(right[:count])) / float(count)


def order_correlation(left: Sequence[int], right: Sequence[int]) -> float:
    right_rank = {int(item): rank for rank, item in enumerate(right)}
    left_values = np.arange(len(left), dtype=np.float64)
    right_values = np.asarray([right_rank[int(item)] for item in left], dtype=np.float64)
    if len(left) <= 1:
        return 1.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


FEATURE_NAMES = [
    "bm25_gap_1_2",
    "bm25_gap_3_4",
    "qk_gap_1_2",
    "qk_gap_3_4",
    "svd_gap_1_2",
    "bm25_qk_top1_agree",
    "bm25_qk_top3_overlap",
    "bm25_svd_top3_overlap",
    "qk_svd_top1_agree",
    "qk_svd_top3_overlap",
    "bm25_qk_order_corr",
    "qk_svd_order_corr",
    "qk_top1_bm25_rank",
    "query_term_count",
    "anchor_top3_overlap",
    "anchor_top16_overlap",
]


def feature_vector(
    row: dict[str, Any], source: dict[str, Any], method: str
) -> np.ndarray:
    bm25 = [int(item) for item in row["candidate_candidates"]]
    qk = [int(item) for item in row[f"{method}_candidates"]]
    svd = [int(item) for item in row["svd_candidates"]]
    bm25_rank = {block_id: rank + 1 for rank, block_id in enumerate(bm25)}
    values = [
        normalized_gap(row["bm25_scores"], 0, 1),
        normalized_gap(row["bm25_scores"], 2, 3),
        ranked_gap(row[f"{method}_scores"], 0, 1),
        ranked_gap(row[f"{method}_scores"], 2, 3),
        ranked_gap(row["svd_scores"], 0, 1),
        float(bm25[0] == qk[0]),
        top_overlap(bm25, qk, 3),
        top_overlap(bm25, svd, 3),
        float(qk[0] == svd[0]),
        top_overlap(qk, svd, 3),
        order_correlation(bm25, qk),
        order_correlation(qk, svd),
        float(bm25_rank[qk[0]]) / max(1, len(bm25)),
        float(source.get("query_term_count", 0)),
        float(source.get("anchor_lexical_top3_overlap", 0)) / 3.0,
        float(source.get("anchor_lexical_top16_overlap", 0)) / 16.0,
    ]
    return np.asarray(values, dtype=np.float64)


def is_hit(rank: int) -> bool:
    return 0 < rank <= 3


def choose_threshold(
    probabilities: np.ndarray,
    baseline_hits: np.ndarray,
    qk_hits: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    thresholds = sorted(set([0.0, 1.0, *probabilities.tolist()]))
    sweep = []
    for threshold in thresholds:
        switch = probabilities >= threshold
        selected = np.where(switch, qk_hits, baseline_hits)
        sweep.append(
            {
                "threshold": float(threshold),
                "recall_at_3": float(selected.mean()),
                "switch_rate": float(switch.mean()),
            }
        )
    best = max(
        sweep,
        key=lambda item: (
            item["recall_at_3"],
            -item["switch_rate"],
            item["threshold"],
        ),
    )
    return float(best["threshold"]), sweep


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    if len(set(labels.tolist())) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def evaluate(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    method: str,
) -> dict[str, Any]:
    baseline_hits = np.asarray([is_hit(int(row["candidate_rank"])) for row in rows])
    qk_hits = np.asarray([is_hit(int(row[f"{method}_rank"])) for row in rows])
    labels = qk_hits & ~baseline_hits
    switch = probabilities >= threshold
    selected = np.where(switch, qk_hits, baseline_hits)
    wins = int(np.sum(selected & ~baseline_hits))
    losses = int(np.sum(baseline_hits & ~selected))
    return {
        "steps": len(rows),
        "beneficial_actions": int(labels.sum()),
        "baseline_recall_at_3": float(baseline_hits.mean()),
        "qk_recall_at_3": float(qk_hits.mean()),
        "gated_recall_at_3": float(selected.mean()),
        "switch_rate": float(switch.mean()),
        "wins_losses": [wins, losses],
        **safe_auc(labels.astype(np.int64), probabilities),
    }


def main() -> None:
    args = parse_args()
    rerank_rows = read_jsonl(Path(args.rerank_rows_path))
    source_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.candidate_rows_path))
    }
    payload: dict[str, Any] = {
        "source": "OOF train-only logistic confidence gate for QK rescue",
        "selection_uses_gold": False,
        "train_labels_used_for_gate_only": True,
        "features": FEATURE_NAMES,
        "methods": {},
    }
    for method in ("full128", "svd"):
        method_payload: dict[str, Any] = {}
        for step_type in sorted({str(row["step_type"]) for row in rerank_rows}):
            groups = {
                split: [
                    row
                    for row in rerank_rows
                    if str(row["split"]) == split
                    and str(row["step_type"]) == step_type
                ]
                for split in ("train", "dev", "test")
            }
            train = groups["train"]
            train_x = np.stack(
                [
                    feature_vector(
                        row,
                        source_by_key[(int(row["query_id"]), int(row["step_index"]))],
                        method,
                    )
                    for row in train
                ]
            )
            baseline_hits = np.asarray(
                [is_hit(int(row["candidate_rank"])) for row in train]
            )
            qk_hits = np.asarray([is_hit(int(row[f"{method}_rank"])) for row in train])
            train_y = (qk_hits & ~baseline_hits).astype(np.int64)
            positives = int(train_y.sum())
            folds = min(5, positives, int(len(train_y) - positives))
            if folds < 2:
                raise ValueError(f"not enough positive/negative train actions for {step_type}")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=17,
                ),
            )
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)
            oof = cross_val_predict(
                model, train_x, train_y, cv=cv, method="predict_proba"
            )[:, 1]
            threshold, threshold_sweep = choose_threshold(
                oof, baseline_hits, qk_hits
            )
            model.fit(train_x, train_y)
            evaluations = {}
            for split, rows in groups.items():
                matrix = np.stack(
                    [
                        feature_vector(
                            row,
                            source_by_key[
                                (int(row["query_id"]), int(row["step_index"]))
                            ],
                            method,
                        )
                        for row in rows
                    ]
                )
                probabilities = (
                    oof if split == "train" else model.predict_proba(matrix)[:, 1]
                )
                evaluations[split] = evaluate(
                    rows, probabilities, threshold, method
                )
            coefficients = model.named_steps["logisticregression"].coef_[0]
            method_payload[step_type] = {
                "train_positive_actions": positives,
                "cv_folds": folds,
                "selected_threshold": threshold,
                "threshold_sweep": threshold_sweep,
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(FEATURE_NAMES, coefficients, strict=True)
                },
                "evaluations": evaluations,
            }
        payload["methods"][method] = method_payload
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
