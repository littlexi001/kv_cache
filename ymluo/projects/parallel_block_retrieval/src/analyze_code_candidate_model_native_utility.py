from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from analyze_candidate_complementarity_utility import key_families as text_key_families
from analyze_scope_marginal_utility_stop import (
    fit_predict,
    forest_factory,
    logistic_factory,
    ridge_regression_factory,
)
from analyze_zero_extra_forward_utility import fdr_bh


RETRIEVERS = ("bm25", "e5", "bm25_e5_rrf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain grouped evaluation of text and model-native signals on "
            "LongBench-v2 code candidate future utility."
        )
    )
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--model_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_prediction_rows", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None


def row_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["query_id"]), int(row["candidate_id"])


def model_families(keys: list[str]) -> dict[str, list[str]]:
    mean_keys = sorted(key for key in keys if key.endswith("_layer_mean"))
    compact_keys = sorted(
        key
        for key in keys
        if key.endswith(("_layer_mean", "_late_mean", "_layer_max"))
    )
    return {
        "qk_mean": [key for key in mean_keys if key.startswith("model_qk_")],
        "value_mean": [key for key in mean_keys if key.startswith("model_value_")],
        "model_mean": mean_keys,
        "model_compact": compact_keys,
    }


def rank_features(row: dict[str, Any]) -> list[float]:
    origins = row["origins"]
    output = [float(row["retriever_count"])]
    ranks = []
    for method in RETRIEVERS:
        rank = int(origins[method]) if method in origins else 64
        present = float(method in origins)
        ranks.append(rank)
        output.extend([present, 1.0 / rank, math.log1p(rank)])
    output.extend([1.0 / min(ranks), math.log1p(min(ranks))])
    return output


def scope_features(row: dict[str, Any]) -> list[float]:
    return [float(row["same_scope_any"]), float(row["same_scope_fraction"])]


def feature_values(
    examples: list[dict[str, Any]], source: str, keys: list[str]
) -> np.ndarray:
    return np.asarray(
        [[float(row[source]["features"][key]) for key in keys] for row in examples],
        dtype=np.float64,
    )


def classification_metrics(
    labels: np.ndarray, utility: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    correlation = spearmanr(predictions, utility)
    return {
        "auc": safe_auc(labels, predictions),
        "brier": float(brier_score_loss(labels, predictions)),
        "spearman_probability_vs_future_gain": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
    }


def regression_metrics(utility: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    correlation = spearmanr(predictions, utility)
    return {
        "spearman": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "mae": float(mean_absolute_error(utility, predictions)),
        "sign_auc": safe_auc((utility > 0).astype(np.int64), predictions),
    }


def grouped_bootstrap(
    *,
    labels: np.ndarray,
    utility: np.ndarray,
    groups: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray | None,
    prediction_type: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metrics = []
    deltas = []
    for _ in range(samples):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        if prediction_type == "classification":
            if len(np.unique(labels[indices])) < 2:
                continue
            metric = float(roc_auc_score(labels[indices], predictions[indices]))
            baseline_metric = (
                float(roc_auc_score(labels[indices], baseline[indices]))
                if baseline is not None
                else None
            )
        else:
            metric = float(spearmanr(predictions[indices], utility[indices]).statistic)
            baseline_metric = (
                float(spearmanr(baseline[indices], utility[indices]).statistic)
                if baseline is not None
                else None
            )
            if not math.isfinite(metric) or (
                baseline_metric is not None and not math.isfinite(baseline_metric)
            ):
                continue
        metrics.append(metric)
        if baseline_metric is not None:
            deltas.append(metric - baseline_metric)

    def interval(values: list[float]) -> list[float] | None:
        return (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else None
        )

    return {
        "query_cluster_bootstrap_samples": len(metrics),
        "metric_95": interval(metrics),
        "delta_vs_rank_scope_95": interval(deltas),
        "probability_better_than_rank_scope": (
            float(np.mean(np.asarray(deltas) > 0)) if deltas else None
        ),
    }


def bootstrap_mean(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def selection_quality(
    examples: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    method: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = []
    for query_id in sorted({int(row["query_id"]) for row in examples}):
        indices = [
            index
            for index, row in enumerate(examples)
            if int(row["query_id"]) == query_id
        ]
        selected.append(max(indices, key=lambda index: float(scores[index])))
    gains = [float(examples[index]["delta_nll_future_b"]) for index in selected]
    return {
        "method": method,
        "queries": len(selected),
        "mean_future_nll_gain": mean(gains),
        "gain_bootstrap95": bootstrap_mean(gains, samples=samples, seed=seed),
        "positive_future_utility_rate": mean([float(value > 0) for value in gains]),
        "mean_source_overlap": mean(
            [float(examples[index]["source_overlap"]) for index in selected]
        ),
        "selected_candidate_ids": [
            int(examples[index]["candidate_id"]) for index in selected
        ],
    }


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidate_rows)
    text_lookup = {row_key(row): row for row in read_jsonl(args.text_rows)}
    model_lookup = {row_key(row): row for row in read_jsonl(args.model_rows)}
    examples = []
    for candidate in candidates:
        key = row_key(candidate)
        examples.append(
            {
                **candidate,
                "text": text_lookup[key],
                "model_native": model_lookup[key],
            }
        )
    if len(examples) != len(text_lookup) or len(examples) != len(model_lookup):
        raise RuntimeError("candidate/text/model rows do not align")
    if any(
        bool(row["future_target_used"])
        or bool(row["selection_uses_target"])
        or bool(row["model_native"]["expanded_workset_reader_forward_used"])
        for row in examples
    ):
        raise RuntimeError("feature protocol contains future or expanded-reader leakage")

    text_keys = sorted(examples[0]["text"]["features"])
    text_families = text_key_families(text_keys)
    model_keys = sorted(examples[0]["model_native"]["features"])
    native_families = model_families(model_keys)
    rank = np.asarray([rank_features(row) for row in examples], dtype=np.float64)
    scope = np.asarray([scope_features(row) for row in examples], dtype=np.float64)
    rank_scope = np.column_stack([rank, scope])
    probe = np.asarray([[float(row["delta_nll_observed_a"])] for row in examples])
    text_dense = feature_values(examples, "text", text_families["dense_query"])
    text_all = feature_values(examples, "text", text_families["all_candidate"])
    qk_mean = feature_values(examples, "model_native", native_families["qk_mean"])
    value_mean = feature_values(examples, "model_native", native_families["value_mean"])
    model_mean = feature_values(examples, "model_native", native_families["model_mean"])
    model_compact = feature_values(
        examples, "model_native", native_families["model_compact"]
    )
    matrices = {
        "rank": rank,
        "rank_scope": rank_scope,
        "text_dense": text_dense,
        "model_qk_mean": qk_mean,
        "model_value_mean": value_mean,
        "model_mean": model_mean,
        "rank_scope_text": np.column_stack([rank_scope, text_all]),
        "rank_scope_model": np.column_stack([rank_scope, model_compact]),
        "rank_scope_text_model": np.column_stack(
            [rank_scope, text_all, model_compact]
        ),
        "rank_scope_probe": np.column_stack([rank_scope, probe]),
        "rank_scope_text_model_probe": np.column_stack(
            [rank_scope, text_all, model_compact, probe]
        ),
    }
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("non-finite feature matrix")
    labels = np.asarray(
        [float(row["delta_nll_future_b"]) > 0 for row in examples], dtype=np.int64
    )
    utility = np.asarray([float(row["delta_nll_future_b"]) for row in examples])
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)

    classifiers: dict[str, tuple[np.ndarray, Callable[[int], Any]]] = {
        f"{name}_logistic": (matrix, logistic_factory) for name, matrix in matrices.items()
    }
    for name in (
        "rank_scope_text",
        "rank_scope_model",
        "rank_scope_text_model",
        "rank_scope_probe",
        "rank_scope_text_model_probe",
    ):
        classifiers[f"{name}_forest"] = (matrices[name], forest_factory)
    regressors = {
        f"{name}_ridge": (matrix, ridge_regression_factory)
        for name, matrix in matrices.items()
    }
    classification_predictions = {
        name: np.zeros(len(examples), dtype=np.float64) for name in classifiers
    }
    regression_predictions = {
        name: np.zeros(len(examples), dtype=np.float64) for name in regressors
    }
    folds = np.full(len(examples), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(examples)), groups=groups)
    ):
        folds[test_indices] = fold
        for name, (matrix, factory) in classifiers.items():
            classification_predictions[name][test_indices] = fit_predict(
                factory,
                matrix[train_indices],
                labels[train_indices],
                matrix[test_indices],
                seed=args.seed + fold,
            )
        for name, (matrix, factory) in regressors.items():
            model = factory(args.seed + fold)
            model.fit(matrix[train_indices], utility[train_indices])
            regression_predictions[name][test_indices] = model.predict(matrix[test_indices])
    if np.any(folds < 0):
        raise RuntimeError("missing out-of-fold predictions")

    baseline_classification = classification_predictions["rank_scope_logistic"]
    baseline_regression = regression_predictions["rank_scope_ridge"]
    bootstrap_names = {
        "rank_scope_logistic",
        "text_dense_logistic",
        "model_qk_mean_logistic",
        "model_value_mean_logistic",
        "rank_scope_text_forest",
        "rank_scope_model_forest",
        "rank_scope_text_model_forest",
        "rank_scope_probe_forest",
        "rank_scope_text_model_probe_forest",
    }
    classification_quality = {}
    for offset, (name, predictions) in enumerate(classification_predictions.items()):
        classification_quality[name] = {
            **classification_metrics(labels, utility, predictions),
            "features": int(classifiers[name][0].shape[1]),
        }
        if name in bootstrap_names:
            classification_quality[name]["uncertainty_95"] = grouped_bootstrap(
                labels=labels,
                utility=utility,
                groups=groups,
                predictions=predictions,
                baseline=(
                    None if name == "rank_scope_logistic" else baseline_classification
                ),
                prediction_type="classification",
                samples=args.bootstrap_samples,
                seed=args.seed + 1000 * (offset + 1),
            )
    regression_bootstrap_names = {
        "rank_scope_ridge",
        "text_dense_ridge",
        "model_qk_mean_ridge",
        "model_value_mean_ridge",
        "rank_scope_text_model_ridge",
        "rank_scope_probe_ridge",
        "rank_scope_text_model_probe_ridge",
    }
    regression_quality = {}
    for offset, (name, predictions) in enumerate(regression_predictions.items()):
        regression_quality[name] = {
            **regression_metrics(utility, predictions),
            "features": int(regressors[name][0].shape[1]),
        }
        if name in regression_bootstrap_names:
            regression_quality[name]["uncertainty_95"] = grouped_bootstrap(
                labels=labels,
                utility=utility,
                groups=groups,
                predictions=predictions,
                baseline=None if name == "rank_scope_ridge" else baseline_regression,
                prediction_type="regression",
                samples=args.bootstrap_samples,
                seed=args.seed + 100_000 + 1000 * (offset + 1),
            )

    individual = []
    for key in model_keys:
        values = np.asarray(
            [float(row["model_native"]["features"][key]) for row in examples]
        )
        correlation = spearmanr(values, utility)
        if math.isfinite(float(correlation.pvalue)):
            individual.append(
                {
                    "feature": key,
                    "spearman": float(correlation.statistic),
                    "pvalue": float(correlation.pvalue),
                }
            )
    adjusted = fdr_bh([row["pvalue"] for row in individual])
    for row, qvalue in zip(individual, adjusted):
        row["fdr_bh_qvalue"] = qvalue
    individual.sort(key=lambda row: (-abs(row["spearman"]), row["feature"]))

    selection_specs = {
        "static_best_retriever_rank": -np.asarray(
            [float(row["best_retriever_rank"]) for row in examples]
        ),
        "observed_probe_delta_a": np.asarray(
            [float(row["delta_nll_observed_a"]) for row in examples]
        ),
        "oof_rank_scope": regression_predictions["rank_scope_ridge"],
        "oof_text": regression_predictions["rank_scope_text_ridge"],
        "oof_model": regression_predictions["rank_scope_model_ridge"],
        "oof_text_model": regression_predictions["rank_scope_text_model_ridge"],
        "oof_probe_model": regression_predictions[
            "rank_scope_text_model_probe_ridge"
        ],
        "oracle_future_b": utility,
    }
    selection = [
        selection_quality(
            examples,
            scores,
            method=method,
            samples=50_000,
            seed=args.seed + 5000 + index,
        )
        for index, (method, scores) in enumerate(selection_specs.items())
    ]

    prediction_rows = []
    for index, example in enumerate(examples):
        prediction_rows.append(
            {
                "query_id": int(example["query_id"]),
                "candidate_id": int(example["candidate_id"]),
                "fold": int(folds[index]),
                "delta_nll_observed_a": float(example["delta_nll_observed_a"]),
                "delta_nll_future_b": float(utility[index]),
                "future_utility_positive": bool(labels[index]),
                "classification": {
                    name: float(values[index])
                    for name, values in classification_predictions.items()
                },
                "regression": {
                    name: float(values[index])
                    for name, values in regression_predictions.items()
                },
                "out_of_fold": True,
                "future_target_used_for_features": False,
                "selection_uses_target": False,
            }
        )

    output = {
        "source": "LongBench-v2 code candidate future utility cross-domain analysis",
        "protocol": {
            "queries": len(np.unique(groups)),
            "candidate_windows": len(examples),
            "window_tokens": 256,
            "observed_segment_a_tokens": 64,
            "future_segment_b_tokens": 64,
            "candidate_depth_per_retriever": 32,
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "current_workset_reader_forward_used": True,
            "expanded_workset_reader_forward_used": False,
            "future_target_used_for_features": False,
            "selection_uses_target": False,
        },
        "target_statistics": {
            "positive_future_utility_rate": float(labels.mean()),
            "mean_future_nll_gain": float(utility.mean()),
        },
        "text_feature_families": text_families,
        "model_feature_families": native_families,
        "feature_set_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "classification_quality": classification_quality,
        "regression_quality": regression_quality,
        "selection_quality": selection,
        "top_individual_model_correlations": individual[:30],
        "fdr_significant_model_features": [
            row for row in individual if row["fdr_bh_qvalue"] < 0.05
        ],
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    prediction_path = Path(args.output_prediction_rows)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
