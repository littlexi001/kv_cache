from __future__ import annotations

import argparse
import json
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
    prediction_metrics,
    read_jsonl,
    regression_metrics,
    ridge_regression_factory,
    summarize_policy,
)
from analyze_zero_extra_forward_utility import grouped_bootstrap_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether no-reader candidate complementarity predicts future marginal "
            "scope-expansion utility."
        )
    )
    parser.add_argument("--candidate_rows", required=True)
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
    suffix = int(row["model_input_tokens"]) - int(row["retrieved_tokens"]) - int(
        row["target_tokens"]
    )
    if suffix not in (128, 512):
        raise ValueError(f"unexpected state suffix: {suffix}")
    return suffix


def key_families(keys: list[str]) -> dict[str, list[str]]:
    dense_query = sorted(
        key
        for key in keys
        if key.startswith(
            (
                "dense_query_affinity_",
                "dense_centroid_query_",
                "dense_projection_query_",
            )
        )
    )
    dense_novelty = sorted(
        key
        for key in keys
        if key.startswith(
            (
                "dense_added_",
                "dense_set_similarity_",
                "dense_centroid_shift",
            )
        )
    )
    lexical = sorted(key for key in keys if key.startswith("lexical_"))
    scope = sorted(key for key in keys if key.startswith("scope_"))
    structure = sorted(key for key in keys if key.startswith("set_"))
    if not all((dense_query, dense_novelty, lexical, scope, structure)):
        raise RuntimeError("one or more candidate feature families are empty")
    return {
        "structure": structure,
        "lexical": lexical,
        "dense_query": dense_query,
        "dense_novelty": dense_novelty,
        "scope": scope,
        "all_candidate": keys,
    }


def main() -> None:
    args = parse_args()
    candidate_rows = read_jsonl(args.candidate_rows)
    candidate_lookup = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["previous_depth"]),
            int(row["expanded_depth"]),
        ): row
        for row in candidate_rows
    }
    if len(candidate_lookup) != 30 * 2 * 3:
        raise RuntimeError(f"expected 180 candidate rows, found {len(candidate_lookup)}")
    if any(
        bool(row["reader_forward_used"])
        or bool(row["expanded_workset_reader_forward_used"])
        or bool(row["future_target_used"])
        or bool(row["selection_uses_target"])
        for row in candidate_rows
    ):
        raise RuntimeError("candidate feature protocol contains reader or future leakage")

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
        if depth_text.isdigit() and int(depth_text) in DEPTHS:
            ppl_lookup[(int(row["query_id"]), infer_state_suffix(row), int(depth_text))] = row

    examples = []
    for query_id in range(30):
        for suffix in (128, 512):
            for previous_depth, expanded_depth in TRANSITIONS:
                previous_nll = float(ppl_lookup[(query_id, suffix, previous_depth)]["mean_nll"])
                expanded_nll = float(ppl_lookup[(query_id, suffix, expanded_depth)]["mean_nll"])
                gain = previous_nll - expanded_nll
                examples.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": suffix,
                        "previous_depth": previous_depth,
                        "expanded_depth": expanded_depth,
                        "marginal_nll_gain": gain,
                        "expansion_helped": gain > 0,
                        "previous_retrieval": retrieval_lookup[(query_id, suffix, previous_depth)],
                        "expanded_retrieval": retrieval_lookup[(query_id, suffix, expanded_depth)],
                        "candidate": candidate_lookup[
                            (query_id, suffix, previous_depth, expanded_depth)
                        ],
                    }
                )

    all_keys = sorted(candidate_rows[0]["features"])
    families = key_families(all_keys)
    common_matrix = np.asarray([common_features(row) for row in examples], dtype=np.float64)
    geometry_matrix = np.asarray([full_features(row) for row in examples], dtype=np.float64)

    def candidate_matrix(keys: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(row["candidate"]["features"][key]) for key in keys] for row in examples],
            dtype=np.float64,
        )

    matrices: dict[str, np.ndarray] = {
        "structural": common_matrix,
        "geometry_churn": geometry_matrix,
    }
    for family in ("lexical", "dense_query", "dense_novelty", "scope", "all_candidate"):
        matrices[family] = np.column_stack([common_matrix, candidate_matrix(families[family])])
        matrices[f"geometry_plus_{family}"] = np.column_stack(
            [geometry_matrix, candidate_matrix(families[family])]
        )
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("non-finite analysis matrix")

    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)

    classifiers: dict[str, tuple[np.ndarray, Callable[[int], Any]]] = {
        f"{name}_logistic": (matrix, logistic_factory) for name, matrix in matrices.items()
    }
    classifiers["geometry_plus_all_candidate_forest"] = (
        matrices["geometry_plus_all_candidate"],
        forest_factory,
    )
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
    for key in all_keys:
        values = np.asarray(
            [float(row["candidate"]["features"][key]) for row in examples]
        )
        correlation = spearmanr(values, utility)
        individual.append(
            {
                "feature": key,
                "spearman": float(correlation.statistic),
                "pvalue": float(correlation.pvalue),
            }
        )
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
        "candidate_complementarity_logistic_t50": (
            classification_predictions["geometry_plus_all_candidate_logistic"],
            0.5,
            "probability",
        ),
        "candidate_complementarity_ridge_g0": (
            regression_predictions["geometry_plus_all_candidate_ridge"],
            0.0,
            "gain",
        ),
        "candidate_dense_novelty_logistic_t50": (
            classification_predictions["geometry_plus_dense_novelty_logistic"],
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
                    index = transition_index[(query_id, suffix, previous_depth, expanded_depth)]
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
                        "candidate_texts_observed": True,
                        "reader_forward_used": False,
                        "future_target_used_for_features": False,
                        "train_labels_use_future_nll": True,
                        "selection_uses_target": False,
                    }
                )

    output = {
        "source": "candidate-conditioned no-reader complementarity utility analysis",
        "protocol": {
            "queries": 30,
            "states": [128, 512],
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "transition_examples": len(examples),
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "retrieval_excludes_observed_64_tokens": True,
            "candidate_texts_observed": True,
            "reader_forward_used": False,
            "expanded_workset_reader_forward_used": False,
            "future_target_used_for_features": False,
            "train_labels_use_future_nll": True,
            "selection_uses_target": False,
        },
        "candidate_feature_families": families,
        "feature_set_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "expansion_statistics": {
            "positive_rate": float(labels.mean()),
            "mean_marginal_nll_gain": float(utility.mean()),
        },
        "prediction_quality": prediction_quality,
        "regression_quality": regression_quality,
        "top_individual_candidate_correlations": individual[:30],
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
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    policy_path = Path(args.output_policy_rows)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for row in policy_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
