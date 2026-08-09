from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from analyze_candidate_complementarity_utility import key_families as candidate_key_families
from analyze_scope_marginal_utility_stop import (
    DEPTHS,
    TRANSITIONS,
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
from analyze_zero_extra_forward_utility import fdr_bh, grouped_bootstrap_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare text complementarity with model-native QK/Value response for "
            "future scope-expansion utility."
        )
    )
    parser.add_argument("--model_rows", required=True)
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--ppl128_rows", required=True)
    parser.add_argument("--ppl512_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_prediction_rows", required=True)
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


def model_families(keys: list[str]) -> dict[str, list[str]]:
    mean_keys = sorted(key for key in keys if key.endswith("_layer_mean"))
    compact_keys = sorted(
        key
        for key in keys
        if key.endswith(("_layer_mean", "_late_mean", "_layer_max"))
    )
    output = {
        "qk_mean": [key for key in mean_keys if key.startswith("model_qk_")],
        "value_mean": [key for key in mean_keys if key.startswith("model_value_")],
        "model_mean": mean_keys,
        "qk_compact": [key for key in compact_keys if key.startswith("model_qk_")],
        "value_compact": [
            key for key in compact_keys if key.startswith("model_value_")
        ],
        "model_compact": compact_keys,
        "model_all": keys,
    }
    if not all(output.values()):
        raise RuntimeError("one or more model-native feature families are empty")
    return output


def main() -> None:
    args = parse_args()
    model_rows = read_jsonl(args.model_rows)
    candidate_rows = read_jsonl(args.candidate_rows)
    transition_key = lambda row: (
        int(row["query_id"]),
        int(row["state_suffix_tokens"]),
        int(row["previous_depth"]),
        int(row["expanded_depth"]),
    )
    model_lookup = {transition_key(row): row for row in model_rows}
    candidate_lookup = {transition_key(row): row for row in candidate_rows}
    expected = 30 * 2 * 3
    if len(model_lookup) != expected or len(candidate_lookup) != expected:
        raise RuntimeError("expected 180 model and candidate rows")
    candidate_query_offsets = sorted(
        {
            int(row.get("candidate_query_end_offset_tokens", -1))
            for row in candidate_rows
        }
    )
    if any(
        bool(row["expanded_workset_reader_forward_used"])
        or bool(row["future_target_used"])
        or bool(row["selection_uses_target"])
        for row in model_rows
    ):
        raise RuntimeError("model-native protocol contains expanded-reader or future leakage")

    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != 100_000_000:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth_text = method.removeprefix("hier_bm25_scope")
        if depth_text.isdigit() and int(depth_text) in DEPTHS:
            suffix = int(row["prefix_tokens"])
            if suffix in (128, 512):
                retrieval_lookup[(int(row["query_id"]), suffix, int(depth_text))] = row
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
                key = (query_id, suffix, previous_depth, expanded_depth)
                gain = float(ppl_lookup[(query_id, suffix, previous_depth)]["mean_nll"]) - float(
                    ppl_lookup[(query_id, suffix, expanded_depth)]["mean_nll"]
                )
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
                        "candidate": candidate_lookup[key],
                        "model_native": model_lookup[key],
                    }
                )

    candidate_keys = sorted(candidate_rows[0]["features"])
    candidate_families = candidate_key_families(candidate_keys)
    model_keys = sorted(model_rows[0]["features"])
    native_families = model_families(model_keys)
    geometry = np.asarray([full_features(row) for row in examples], dtype=np.float64)

    def values(source: str, keys: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(row[source]["features"][key]) for key in keys] for row in examples],
            dtype=np.float64,
        )

    e5_dense_novelty = values("candidate", candidate_families["dense_novelty"])
    e5_all = values("candidate", candidate_families["all_candidate"])
    model_matrices = {
        name: values("model_native", keys) for name, keys in native_families.items()
    }
    matrices = {
        "geometry": geometry,
        "geometry_e5_dense_novelty": np.column_stack([geometry, e5_dense_novelty]),
        "geometry_e5_all": np.column_stack([geometry, e5_all]),
        "model_qk_mean": model_matrices["qk_mean"],
        "model_value_mean": model_matrices["value_mean"],
        "model_mean": model_matrices["model_mean"],
        "geometry_model_qk_mean": np.column_stack(
            [geometry, model_matrices["qk_mean"]]
        ),
        "geometry_model_value_mean": np.column_stack(
            [geometry, model_matrices["value_mean"]]
        ),
        "geometry_model_mean": np.column_stack(
            [geometry, model_matrices["model_mean"]]
        ),
        "geometry_model_compact": np.column_stack(
            [geometry, model_matrices["model_compact"]]
        ),
        "geometry_e5_model_mean": np.column_stack(
            [geometry, e5_all, model_matrices["model_mean"]]
        ),
        "geometry_e5_model_compact": np.column_stack(
            [geometry, e5_all, model_matrices["model_compact"]]
        ),
    }
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("non-finite feature matrix")

    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)
    classifiers: dict[str, tuple[np.ndarray, Callable[[int], Any]]] = {
        f"{name}_logistic": (matrix, logistic_factory) for name, matrix in matrices.items()
    }
    for name in (
        "geometry_model_compact",
        "geometry_e5_model_mean",
        "geometry_e5_model_compact",
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

    baseline = classification_predictions["geometry_logistic"]
    prediction_quality = {}
    for offset, (name, predictions) in enumerate(classification_predictions.items()):
        prediction_quality[name] = {
            **prediction_metrics(examples, predictions),
            "features": int(classifiers[name][0].shape[1]),
            "uncertainty_95": grouped_bootstrap_metrics(
                examples,
                predictions,
                baseline_predictions=None if name == "geometry_logistic" else baseline,
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
    for key in model_keys:
        feature_values = np.asarray(
            [float(row["model_native"]["features"][key]) for row in examples]
        )
        correlation = spearmanr(feature_values, utility)
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

    prediction_rows = []
    for index, example in enumerate(examples):
        prediction_rows.append(
            {
                "query_id": int(example["query_id"]),
                "state_suffix_tokens": int(example["state_suffix_tokens"]),
                "previous_depth": int(example["previous_depth"]),
                "expanded_depth": int(example["expanded_depth"]),
                "fold": int(folds[index]),
                "marginal_nll_gain": float(utility[index]),
                "expansion_helped": bool(labels[index]),
                "classification": {
                    name: float(values[index])
                    for name, values in classification_predictions.items()
                },
                "regression": {
                    name: float(values[index])
                    for name, values in regression_predictions.items()
                },
                "out_of_fold": True,
                "selection_uses_target": False,
            }
        )

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
        "model_native_logistic_t50": (
            classification_predictions["geometry_model_compact_logistic"],
            0.5,
            "probability",
        ),
        "model_native_ridge_g0": (
            regression_predictions["geometry_model_compact_ridge"],
            0.0,
            "gain",
        ),
        "e5_model_forest_t50": (
            classification_predictions["geometry_e5_model_compact_forest"],
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
                        "current_workset_reader_forward_used": True,
                        "expanded_workset_reader_forward_used": False,
                        "future_target_used_for_features": False,
                        "train_labels_use_future_nll": True,
                        "selection_uses_target": False,
                    }
                )

    output = {
        "source": "model-native candidate-conditioned scope utility analysis",
        "protocol": {
            "queries": 30,
            "states": [128, 512],
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "examples": len(examples),
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "current_workset_reader_forward_used": True,
            "expanded_workset_reader_forward_used": False,
            "candidate_sidecar_is_block_local": True,
            "candidate_query_end_offset_tokens": candidate_query_offsets,
            "candidate_features_include_observed_64_tokens": (
                candidate_query_offsets == [0]
            ),
            "future_target_used_for_features": False,
            "train_labels_use_future_nll": True,
            "selection_uses_target": False,
        },
        "model_feature_families": native_families,
        "feature_set_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "expansion_statistics": {
            "positive_rate": float(labels.mean()),
            "mean_marginal_nll_gain": float(utility.mean()),
        },
        "prediction_quality": prediction_quality,
        "regression_quality": regression_quality,
        "top_individual_model_correlations": individual[:30],
        "fdr_significant_model_features": [
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
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    prediction_path = Path(args.output_prediction_rows)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    policy_path = Path(args.output_policy_rows)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for row in policy_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
