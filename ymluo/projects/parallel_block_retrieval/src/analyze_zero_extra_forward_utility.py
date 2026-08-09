from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from analyze_scope_marginal_utility_stop import (
    DEPTHS,
    TRANSITIONS,
    common_features,
    fit_predict,
    forest_factory,
    full_features,
    logistic_factory,
    mean,
    prediction_metrics,
    read_jsonl,
    regression_metrics,
    ridge_regression_factory,
    summarize_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether one normal current-workset forward predicts future marginal "
            "scope utility without evaluating an expanded workset."
        )
    )
    parser.add_argument("--signal_rows", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--ppl128_rows", required=True)
    parser.add_argument("--ppl512_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_policy_rows", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def infer_state_suffix(row: dict[str, Any]) -> int:
    suffix = (
        int(row["model_input_tokens"])
        - int(row["retrieved_tokens"])
        - int(row["target_tokens"])
    )
    if suffix not in (128, 512):
        raise ValueError(f"unexpected state suffix: {suffix}")
    return suffix


def select_signal_keys(keys: list[str]) -> dict[str, list[str]]:
    distribution = sorted(
        key
        for key in keys
        if key.startswith(("predictive_entropy_", "top1_margin_", "max_probability_"))
    )
    observed_surprisal = sorted(
        key
        for key in keys
        if key.startswith(("observed_surprisal_", "observed_token_probability_"))
    )
    hidden = sorted(
        key
        for key in keys
        if key.startswith("hidden_checkpoint")
        and any(
            marker in key
            for marker in (
                "adjacent_cosine_mean",
                "dispersion",
                "shift_8",
                "shift_16",
                "shift_32",
            )
        )
    )
    attention = sorted(
        key
        for key in keys
        if key.startswith("attention_")
        and not key.endswith("_max")
        and any(
            marker in key
            for marker in (
                "norm_ratio_mean",
                "residual_alignment_mean",
                "adjacent_cosine_mean",
                "dispersion",
                "shift_8",
                "shift_16",
                "shift_32",
            )
        )
    )
    if not all((distribution, observed_surprisal, hidden, attention)):
        raise RuntimeError("one or more signal families are empty")
    return {
        "distribution": distribution,
        "observed_surprisal": observed_surprisal,
        "hidden": hidden,
        "attention_response": attention,
    }


def signal_values(example: dict[str, Any], keys: list[str]) -> list[float]:
    features = example["current_signal"]["signal_features"]
    values = [float(features[key]) for key in keys]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite signal feature")
    return values


def grouped_bootstrap_metrics(
    examples: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    baseline_predictions: np.ndarray | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    aucs = []
    correlations = []
    auc_deltas = []
    for _ in range(samples):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(sampled_labels, predictions[indices]))
        aucs.append(auc)
        correlation = spearmanr(predictions[indices], utility[indices]).statistic
        if math.isfinite(float(correlation)):
            correlations.append(float(correlation))
        if baseline_predictions is not None:
            baseline_auc = float(
                roc_auc_score(sampled_labels, baseline_predictions[indices])
            )
            auc_deltas.append(auc - baseline_auc)

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "query_cluster_bootstrap_samples": len(aucs),
        "auc_95": interval(aucs),
        "spearman_95": interval(correlations),
        "auc_delta_vs_geometry_95": interval(auc_deltas),
    }


def fdr_bh(pvalues: list[float]) -> list[float]:
    array = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(array) - reverse_rank + 1
        running = min(running, float(array[index]) * len(array) / rank)
        adjusted[index] = running
    return adjusted.clip(0.0, 1.0).tolist()


def main() -> None:
    args = parse_args()
    signal_rows = read_jsonl(args.signal_rows)
    signal_lookup = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["current_scope_depth"]),
        ): row
        for row in signal_rows
    }
    if len(signal_lookup) != 30 * 2 * 3:
        raise RuntimeError(f"expected 180 signal rows, found {len(signal_lookup)}")
    if any(
        bool(row["expanded_workset_forward_used"])
        or bool(row["future_target_used"])
        or bool(row["selection_uses_target"])
        or bool(row["retrieval_query_uses_observed_probe_tokens"])
        for row in signal_rows
    ):
        raise RuntimeError("signal protocol contains future or expanded-workset leakage")

    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != 100_000_000:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth_text = method.removeprefix("hier_bm25_scope")
        if not depth_text.isdigit():
            continue
        depth = int(depth_text)
        suffix = int(row["prefix_tokens"])
        if depth in DEPTHS and suffix in (128, 512):
            retrieval_lookup[(int(row["query_id"]), suffix, depth)] = row

    ppl_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.ppl128_rows) + read_jsonl(args.ppl512_rows):
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth_text = method.removeprefix("hier_bm25_scope")
        if not depth_text.isdigit():
            continue
        depth = int(depth_text)
        if depth in DEPTHS:
            ppl_lookup[(int(row["query_id"]), infer_state_suffix(row), depth)] = row

    examples = []
    for query_id in range(30):
        for suffix in (128, 512):
            for previous_depth, expanded_depth in TRANSITIONS:
                previous_nll = float(
                    ppl_lookup[(query_id, suffix, previous_depth)]["mean_nll"]
                )
                expanded_nll = float(
                    ppl_lookup[(query_id, suffix, expanded_depth)]["mean_nll"]
                )
                marginal_gain = previous_nll - expanded_nll
                examples.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": suffix,
                        "previous_depth": previous_depth,
                        "expanded_depth": expanded_depth,
                        "marginal_nll_gain": marginal_gain,
                        "expansion_helped": marginal_gain > 0,
                        "previous_retrieval": retrieval_lookup[
                            (query_id, suffix, previous_depth)
                        ],
                        "expanded_retrieval": retrieval_lookup[
                            (query_id, suffix, expanded_depth)
                        ],
                        "current_signal": signal_lookup[
                            (query_id, suffix, previous_depth)
                        ],
                    }
                )

    all_signal_keys = sorted(signal_rows[0]["signal_features"])
    key_families = select_signal_keys(all_signal_keys)
    state_keys = sorted({key for values in key_families.values() for key in values})
    uncertainty_keys = sorted(
        set(key_families["distribution"] + key_families["observed_surprisal"])
    )

    common_matrix = np.asarray([common_features(row) for row in examples])
    geometry_matrix = np.asarray([full_features(row) for row in examples])

    def family_matrix(keys: list[str]) -> np.ndarray:
        return np.asarray([signal_values(row, keys) for row in examples])

    matrices = {
        "structural": common_matrix,
        "geometry_churn": geometry_matrix,
        "uncertainty": np.column_stack(
            [common_matrix, family_matrix(uncertainty_keys)]
        ),
        "hidden": np.column_stack(
            [common_matrix, family_matrix(key_families["hidden"])]
        ),
        "attention_response": np.column_stack(
            [common_matrix, family_matrix(key_families["attention_response"])]
        ),
        "state_signals": np.column_stack(
            [common_matrix, family_matrix(state_keys)]
        ),
        "geometry_plus_state": np.column_stack(
            [geometry_matrix, family_matrix(state_keys)]
        ),
    }
    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)

    classifiers: dict[str, tuple[np.ndarray, Callable[[int], Any]]] = {}
    regressors: dict[str, tuple[np.ndarray, Callable[[int], Any]]] = {}
    for family, matrix in matrices.items():
        classifiers[f"{family}_logistic"] = (matrix, logistic_factory)
        regressors[f"{family}_ridge"] = (matrix, ridge_regression_factory)
    classifiers["geometry_plus_state_forest"] = (
        matrices["geometry_plus_state"],
        forest_factory,
    )

    classification_predictions = {
        name: np.zeros(len(examples), dtype=np.float64) for name in classifiers
    }
    regression_predictions = {
        name: np.zeros(len(examples), dtype=np.float64) for name in regressors
    }
    folds = np.full(len(examples), -1, dtype=np.int64)
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
            regression_predictions[name][test_indices] = model.predict(
                matrix[test_indices]
            )
    if np.any(folds < 0):
        raise RuntimeError("missing out-of-fold predictions")

    geometry_predictions = classification_predictions["geometry_churn_logistic"]
    prediction_quality = {}
    for offset, (name, predictions) in enumerate(classification_predictions.items()):
        prediction_quality[name] = {
            **prediction_metrics(examples, predictions),
            "features": int(classifiers[name][0].shape[1]),
            "uncertainty_95": grouped_bootstrap_metrics(
                examples,
                predictions,
                baseline_predictions=(
                    None if name == "geometry_churn_logistic" else geometry_predictions
                ),
                samples=args.bootstrap_samples,
                seed=args.seed + 1000 * (offset + 1),
            ),
        }
    regression_quality = {
        name: {
            **regression_metrics(examples, predictions),
            "features": int(regressors[name][0].shape[1]),
        }
        for name, predictions in regression_predictions.items()
    }

    individual = []
    for key in all_signal_keys:
        values = np.asarray([signal_values(row, [key])[0] for row in examples])
        correlation = spearmanr(values, utility)
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

    transition_index = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["previous_depth"]),
            int(row["expanded_depth"]),
        ): index
        for index, row in enumerate(examples)
    }
    policy_specs = {
        "zero_extra_geometry_state_logistic_t50": (
            classification_predictions["geometry_plus_state_logistic"],
            0.5,
            "probability",
        ),
        "zero_extra_geometry_state_ridge_g0": (
            regression_predictions["geometry_plus_state_ridge"],
            0.0,
            "gain",
        ),
        "zero_extra_state_logistic_t50": (
            classification_predictions["state_signals_logistic"],
            0.5,
            "probability",
        ),
    }
    policy_rows = []
    for method, (predictions, threshold, score_type) in policy_specs.items():
        for query_id in range(30):
            for suffix in (128, 512):
                chosen_depth = DEPTHS[0]
                decisions = []
                for previous_depth, expanded_depth in TRANSITIONS:
                    if chosen_depth != previous_depth:
                        break
                    index = transition_index[
                        (query_id, suffix, previous_depth, expanded_depth)
                    ]
                    score = float(predictions[index])
                    expand = score >= threshold if score_type == "probability" else score > threshold
                    decisions.append(
                        {
                            "previous_depth": previous_depth,
                            "expanded_depth": expanded_depth,
                            "score": score,
                            "score_type": score_type,
                            "threshold": threshold,
                            "expand": expand,
                        }
                    )
                    if not expand:
                        break
                    chosen_depth = expanded_depth
                selected = retrieval_lookup[(query_id, suffix, chosen_depth)]
                policy_rows.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": suffix,
                        "method": method,
                        "chosen_scope_depth": chosen_depth,
                        "decisions": decisions,
                        "candidate_blocks": int(selected["candidate_blocks"]),
                        "top_block_ids": selected["top_block_ids"],
                        "out_of_fold": True,
                        "grouped_by_query_id": True,
                        "one_normal_current_workset_forward": True,
                        "expanded_workset_forward_used": False,
                        "future_target_used_for_features": False,
                        "train_labels_use_future_nll": True,
                        "selection_uses_target": False,
                    }
                )

    output = {
        "source": "zero-extra-candidate-forward future scope utility analysis",
        "protocol": {
            "queries": 30,
            "states": [128, 512],
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "transition_examples": len(examples),
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "retrieval_excludes_observed_64_tokens": True,
            "features_use_one_normal_current_workset_forward": True,
            "expanded_workset_forward_used": False,
            "future_target_used_for_features": False,
            "train_labels_use_future_nll": True,
            "teacher_forced_observed_tokens": True,
            "selection_uses_target": False,
        },
        "signal_feature_families": key_families,
        "feature_set_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "expansion_statistics": {
            "positive_rate": float(labels.mean()),
            "mean_marginal_nll_gain": float(utility.mean()),
        },
        "prediction_quality": prediction_quality,
        "regression_quality": regression_quality,
        "top_individual_signal_correlations": individual[:30],
        "fdr_significant_individual_signals": [
            row for row in individual if row["fdr_bh_qvalue"] < 0.05
        ],
        "policy_quality": summarize_policy(
            policy_rows,
            ppl_lookup=ppl_lookup,
            retrieval_lookup=retrieval_lookup,
            bootstrap_samples=50_000,
            seed=args.seed,
        ),
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    policy_path = Path(args.output_policy_rows)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for row in policy_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
