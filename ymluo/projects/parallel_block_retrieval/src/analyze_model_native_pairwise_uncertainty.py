from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


CLASSIFICATION_PAIRS = (
    (
        "model_qk_mean_logistic",
        "geometry_logistic",
    ),
    (
        "geometry_e5_dense_novelty_logistic",
        "geometry_logistic",
    ),
    (
        "model_qk_mean_logistic",
        "geometry_e5_dense_novelty_logistic",
    ),
    (
        "geometry_e5_model_compact_logistic",
        "geometry_e5_dense_novelty_logistic",
    ),
    (
        "geometry_e5_model_mean_forest",
        "geometry_e5_dense_novelty_logistic",
    ),
)
REGRESSION_PAIRS = (
    (
        "geometry_e5_model_compact_ridge",
        "geometry_e5_dense_novelty_ridge",
    ),
    (
        "geometry_e5_model_compact_ridge",
        "geometry_ridge",
    ),
    (
        "model_value_mean_ridge",
        "geometry_ridge",
    ),
    (
        "model_qk_mean_ridge",
        "geometry_e5_dense_novelty_ridge",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap paired uncertainty for model-native utility predictors."
    )
    parser.add_argument("--prediction_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_pairs(
    rows: list[dict[str, Any]],
    *,
    pairs: tuple[tuple[str, str], ...],
    prediction_type: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    labels = np.asarray([bool(row["expansion_helped"]) for row in rows], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in rows])
    groups = np.asarray([int(row["query_id"]) for row in rows], dtype=np.int64)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    predictions = {
        name: np.asarray([float(row[prediction_type][name]) for row in rows])
        for pair in pairs
        for name in pair
    }
    output = []
    for pair_index, (left, right) in enumerate(pairs):
        left_values = predictions[left]
        right_values = predictions[right]
        if prediction_type == "classification":
            observed_left = float(roc_auc_score(labels, left_values))
            observed_right = float(roc_auc_score(labels, right_values))
        else:
            observed_left = float(spearmanr(left_values, utility).statistic)
            observed_right = float(spearmanr(right_values, utility).statistic)
        left_samples: list[float] = []
        right_samples: list[float] = []
        differences: list[float] = []
        for _ in range(samples):
            sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
            indices = np.concatenate([group_indices[group] for group in sampled_groups])
            if prediction_type == "classification":
                sampled_labels = labels[indices]
                if len(np.unique(sampled_labels)) < 2:
                    continue
                left_metric = float(roc_auc_score(sampled_labels, left_values[indices]))
                right_metric = float(roc_auc_score(sampled_labels, right_values[indices]))
            else:
                left_metric = float(spearmanr(left_values[indices], utility[indices]).statistic)
                right_metric = float(spearmanr(right_values[indices], utility[indices]).statistic)
                if not math.isfinite(left_metric) or not math.isfinite(right_metric):
                    continue
            left_samples.append(left_metric)
            right_samples.append(right_metric)
            differences.append(left_metric - right_metric)
        output.append(
            {
                "left": left,
                "right": right,
                "metric": "auc" if prediction_type == "classification" else "spearman",
                "observed_left": observed_left,
                "observed_right": observed_right,
                "observed_delta_left_minus_right": observed_left - observed_right,
                "left_95": interval(left_samples),
                "right_95": interval(right_samples),
                "delta_95": interval(differences),
                "bootstrap_probability_left_better": float(
                    np.mean(np.asarray(differences) > 0)
                ),
                "bootstrap_samples": len(differences),
                "pair_index": pair_index,
            }
        )
    return output


def ranking_budgets(
    rows: list[dict[str, Any]], *, prediction_type: str, methods: list[str]
) -> list[dict[str, Any]]:
    labels = np.asarray([bool(row["expansion_helped"]) for row in rows], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in rows])
    output = []
    for method in methods:
        predictions = np.asarray([float(row[prediction_type][method]) for row in rows])
        order = np.argsort(-predictions)
        for fraction in (0.1, 0.2, 0.4, 0.6):
            take = max(1, round(len(rows) * fraction))
            selected = order[:take]
            output.append(
                {
                    "prediction_type": prediction_type,
                    "method": method,
                    "top_fraction": fraction,
                    "selected_examples": len(selected),
                    "positive_precision": float(labels[selected].mean()),
                    "positive_recall": float(labels[selected].sum() / labels.sum()),
                    "mean_marginal_nll_gain": float(utility[selected].mean()),
                    "median_marginal_nll_gain": float(np.median(utility[selected])),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.prediction_rows)
    if len(rows) != 180 or not all(bool(row["out_of_fold"]) for row in rows):
        raise RuntimeError("requires 180 out-of-fold transition predictions")
    output = {
        "source": "paired query-cluster uncertainty for fair-state model-native predictors",
        "protocol": {
            "examples": len(rows),
            "query_groups": len({int(row["query_id"]) for row in rows}),
            "bootstrap_samples": args.bootstrap_samples,
            "resampling_unit": "query_id",
            "predictions_are_out_of_fold": True,
            "selection_uses_target": False,
        },
        "classification_pairwise": bootstrap_pairs(
            rows,
            pairs=CLASSIFICATION_PAIRS,
            prediction_type="classification",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "regression_pairwise": bootstrap_pairs(
            rows,
            pairs=REGRESSION_PAIRS,
            prediction_type="regression",
            samples=args.bootstrap_samples,
            seed=args.seed + 100_000,
        ),
        "fixed_ranking_budget_diagnostics": ranking_budgets(
            rows,
            prediction_type="classification",
            methods=[
                "geometry_logistic",
                "geometry_e5_dense_novelty_logistic",
                "model_qk_mean_logistic",
                "geometry_e5_model_compact_logistic",
            ],
        )
        + ranking_budgets(
            rows,
            prediction_type="regression",
            methods=[
                "geometry_ridge",
                "geometry_e5_dense_novelty_ridge",
                "geometry_e5_model_compact_ridge",
            ],
        ),
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
