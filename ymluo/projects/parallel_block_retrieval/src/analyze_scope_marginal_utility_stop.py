from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEPTHS = (3, 8, 16, 32)
TRANSITIONS = tuple(zip(DEPTHS[:-1], DEPTHS[1:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test no-target predictors for marginal scope expansion utility."
    )
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--ppl128_rows", required=True)
    parser.add_argument("--ppl512_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_policy_rows", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--bootstrap_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_thresholds(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0 or max(values) >= 1:
        raise ValueError("thresholds must lie in (0, 1)")
    return values


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def top8_jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(int(item) for item in left[:8])
    right_set = set(int(item) for item in right[:8])
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def common_features(example: dict[str, Any]) -> list[float]:
    previous = example["previous_retrieval"]
    expanded = example["expanded_retrieval"]
    return [
        math.log(float(example["state_suffix_tokens"])),
        math.log(float(example["previous_depth"])),
        math.log(float(example["expanded_depth"])),
        math.log1p(float(previous["candidate_blocks"])),
        math.log(
            max(
                float(expanded["candidate_blocks"])
                / max(float(previous["candidate_blocks"]), 1.0),
                1.0e-8,
            )
        ),
        math.log1p(float(previous["scope_query_features"])),
        math.log1p(float(previous["active_scopes"])),
    ]


def score_geometry_features(example: dict[str, Any]) -> list[float]:
    row = example["previous_retrieval"]
    top1 = max(abs(float(row["scope_top1_score"])), 1.0e-8)
    return [
        *common_features(example),
        math.log1p(float(row["positive_scope_scores"])),
        float(row["scope_top1_score"]) / max(float(row["scope_query_features"]), 1.0),
        float(row["scope_normalized_margin_1_2"]),
        float(row["scope_margin_3_4"]) / top1,
        float(row["scope_margin_8_9"]) / top1,
        float(row["scope_margin_16_17"]) / top1,
        float(row["scope_margin_32_33"]) / top1,
        float(row["scope_top1_positive_share"]),
        float(row["scope_top3_positive_share"]),
        float(row["scope_top8_positive_share"]),
        float(row["scope_top16_positive_share"]),
        float(row["scope_top32_positive_share"]),
        float(row["scope_score_normalized_entropy"]),
        float(row["scope_score_hhi"]),
        float(row["scope_top1_z"]),
    ]


def full_features(example: dict[str, Any]) -> list[float]:
    previous = example["previous_retrieval"]
    expanded = example["expanded_retrieval"]
    previous_depth = int(example["previous_depth"])
    expanded_depth = int(example["expanded_depth"])
    share_keys = {
        3: "scope_top3_positive_share",
        8: "scope_top8_positive_share",
        16: "scope_top16_positive_share",
        32: "scope_top32_positive_share",
    }
    mass_increment = float(previous[share_keys[expanded_depth]]) - float(
        previous[share_keys[previous_depth]]
    )
    jaccard = top8_jaccard(
        previous["top_block_ids"], expanded["top_block_ids"]
    )
    return [
        *score_geometry_features(example),
        mass_increment,
        jaccard,
        1.0 - jaccard,
        float(previous["candidate_fraction"]),
        float(expanded["candidate_fraction"]),
    ]


def logistic_factory(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )


def forest_factory(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def ridge_regression_factory(seed: int) -> Any:
    del seed
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def forest_regression_factory(seed: int) -> Any:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def fit_predict(
    factory: Callable[[int], Any],
    x_train: np.ndarray,
    labels_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    classes = np.unique(labels_train)
    if len(classes) == 1:
        return np.full(len(x_test), float(classes[0]), dtype=np.float64)
    model = factory(seed)
    model.fit(x_train, labels_train)
    return model.predict_proba(x_test)[:, 1]


def prediction_metrics(
    examples: list[dict[str, Any]], probabilities: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    correlation = spearmanr(probabilities, utility)
    output: dict[str, Any] = {
        "examples": len(examples),
        "positive_rate": float(labels.mean()),
        "auc": safe_auc(labels, probabilities),
        "brier": float(brier_score_loss(labels, probabilities)),
        "spearman_probability_vs_nll_gain": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "by_state": {},
        "by_transition": {},
    }
    for suffix in (128, 512):
        mask = np.asarray(
            [int(row["state_suffix_tokens"]) == suffix for row in examples]
        )
        output["by_state"][str(suffix)] = {
            "examples": int(mask.sum()),
            "positive_rate": float(labels[mask].mean()),
            "auc": safe_auc(labels[mask], probabilities[mask]),
        }
    for previous_depth, expanded_depth in TRANSITIONS:
        label = f"{previous_depth}_to_{expanded_depth}"
        mask = np.asarray(
            [
                int(row["previous_depth"]) == previous_depth
                and int(row["expanded_depth"]) == expanded_depth
                for row in examples
            ]
        )
        output["by_transition"][label] = {
            "examples": int(mask.sum()),
            "positive_rate": float(labels[mask].mean()),
            "auc": safe_auc(labels[mask], probabilities[mask]),
        }
    return output


def regression_metrics(
    examples: list[dict[str, Any]], predictions: np.ndarray
) -> dict[str, Any]:
    utility = np.asarray([float(row["marginal_nll_gain"]) for row in examples])
    labels = (utility > 0).astype(np.int64)
    correlation = spearmanr(predictions, utility)
    output: dict[str, Any] = {
        "examples": len(examples),
        "mae": float(mean_absolute_error(utility, predictions)),
        "spearman_predicted_vs_observed_gain": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "sign_auc": safe_auc(labels, predictions),
        "by_state": {},
    }
    for suffix in (128, 512):
        mask = np.asarray(
            [int(row["state_suffix_tokens"]) == suffix for row in examples]
        )
        state_correlation = spearmanr(predictions[mask], utility[mask])
        output["by_state"][str(suffix)] = {
            "examples": int(mask.sum()),
            "mae": float(mean_absolute_error(utility[mask], predictions[mask])),
            "spearman": float(state_correlation.statistic),
            "spearman_pvalue": float(state_correlation.pvalue),
            "sign_auc": safe_auc(labels[mask], predictions[mask]),
        }
    return output


def summarize_policy(
    policy_rows: list[dict[str, Any]],
    *,
    ppl_lookup: dict[tuple[int, int, int], dict[str, Any]],
    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for suffix in (128, 512):
        for method in sorted({str(row["method"]) for row in policy_rows}):
            group = [
                row
                for row in policy_rows
                if row["method"] == method
                and int(row["state_suffix_tokens"]) == suffix
            ]
            selected_nll = [
                float(
                    ppl_lookup[
                        (
                            int(row["query_id"]),
                            suffix,
                            int(row["chosen_scope_depth"]),
                        )
                    ]["mean_nll"]
                )
                for row in group
            ]
            selected_retrieval = [
                retrieval_lookup[
                    (
                        int(row["query_id"]),
                        suffix,
                        int(row["chosen_scope_depth"]),
                    )
                ]
                for row in group
            ]
            item = {
                "state_suffix_tokens": suffix,
                "method": method,
                "queries": len(group),
                "mean_chosen_scope_depth": mean(
                    [float(row["chosen_scope_depth"]) for row in group]
                ),
                "chosen_depth_counts": {
                    str(depth): sum(
                        int(row["chosen_scope_depth"]) == depth for row in group
                    )
                    for depth in DEPTHS
                },
                "ppl": math.exp(mean(selected_nll)),
                "mean_nll": mean(selected_nll),
                "mean_candidate_blocks": mean(
                    [float(row["candidate_blocks"]) for row in selected_retrieval]
                ),
                "same_scope_any_at_8": mean(
                    [float(row["same_scope_any_at_8"]) for row in selected_retrieval]
                ),
                "same_scope_fraction_at_8": mean(
                    [float(row["same_scope_fraction_at_8"]) for row in selected_retrieval]
                ),
                "paired_vs_fixed": {},
            }
            for fixed_depth in (3, 8):
                differences = [
                    selected_nll[index]
                    - float(
                        ppl_lookup[
                            (int(row["query_id"]), suffix, fixed_depth)
                        ]["mean_nll"]
                    )
                    for index, row in enumerate(group)
                ]
                item["paired_vs_fixed"][str(fixed_depth)] = {
                    "meaning": "negative favors policy",
                    "mean_nll_policy_minus_fixed": mean(differences),
                    "bootstrap95": bootstrap_mean_ci(
                        differences,
                        samples=bootstrap_samples,
                        seed=seed + suffix + fixed_depth,
                    ),
                }
            output.append(item)
    return output


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    retrieval_rows = read_jsonl(args.retrieval_rows)
    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in retrieval_rows:
        if int(row["memory_tokens"]) != 100_000_000:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        if depth not in DEPTHS:
            continue
        retrieval_lookup[
            (int(row["query_id"]), int(row["prefix_tokens"]), depth)
        ] = row

    ppl_rows = read_jsonl(args.ppl128_rows) + read_jsonl(args.ppl512_rows)
    ppl_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in ppl_rows:
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        if depth not in DEPTHS:
            continue
        suffix = (
            int(row["model_input_tokens"])
            - int(row["retrieved_tokens"])
            - int(row["target_tokens"])
        )
        if suffix not in (128, 512):
            raise ValueError(f"unexpected state suffix inferred from PPL row: {suffix}")
        ppl_lookup[(int(row["query_id"]), suffix, depth)] = row

    examples = []
    for query_id in range(30):
        for suffix in (128, 512):
            for previous_depth, expanded_depth in TRANSITIONS:
                previous_ppl = ppl_lookup[(query_id, suffix, previous_depth)]
                expanded_ppl = ppl_lookup[(query_id, suffix, expanded_depth)]
                marginal_gain = float(previous_ppl["mean_nll"]) - float(
                    expanded_ppl["mean_nll"]
                )
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
                    }
                )
    if len(examples) != 180:
        raise RuntimeError(f"expected 180 transition examples, found {len(examples)}")

    feature_sets = {
        "structural_logistic": (
            np.asarray([common_features(row) for row in examples]),
            logistic_factory,
        ),
        "score_geometry_logistic": (
            np.asarray([score_geometry_features(row) for row in examples]),
            logistic_factory,
        ),
        "geometry_churn_logistic": (
            np.asarray([full_features(row) for row in examples]),
            logistic_factory,
        ),
        "geometry_churn_forest": (
            np.asarray([full_features(row) for row in examples]),
            forest_factory,
        ),
    }
    labels = np.asarray([bool(row["expansion_helped"]) for row in examples], dtype=np.int64)
    utility_targets = np.asarray(
        [float(row["marginal_nll_gain"]) for row in examples], dtype=np.float64
    )
    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)
    predictions = {
        name: np.zeros(len(examples), dtype=np.float64) for name in feature_sets
    }
    regression_features = np.asarray([full_features(row) for row in examples])
    regression_factories = {
        "geometry_churn_ridge": ridge_regression_factory,
        "geometry_churn_regression_forest": forest_regression_factory,
    }
    regression_predictions = {
        name: np.zeros(len(examples), dtype=np.float64)
        for name in regression_factories
    }
    folds = np.full(len(examples), -1, dtype=np.int64)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(examples)), groups=groups)
    ):
        folds[test_indices] = fold
        for feature_name, (features, factory) in feature_sets.items():
            predictions[feature_name][test_indices] = fit_predict(
                factory,
                features[train_indices],
                labels[train_indices],
                features[test_indices],
                seed=args.seed + fold,
            )
        for model_name, factory in regression_factories.items():
            model = factory(args.seed + fold)
            model.fit(regression_features[train_indices], utility_targets[train_indices])
            regression_predictions[model_name][test_indices] = model.predict(
                regression_features[test_indices]
            )
    if np.any(folds < 0):
        raise RuntimeError("missing out-of-fold predictions")

    transition_index = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["previous_depth"]),
            int(row["expanded_depth"]),
        ): index
        for index, row in enumerate(examples)
    }
    policy_rows = []
    for predictor_name in ("geometry_churn_logistic", "geometry_churn_forest"):
        predictor_label = "logistic" if predictor_name.endswith("logistic") else "forest"
        primary_predictions = predictions[predictor_name]
        for threshold in thresholds:
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
                        probability = float(primary_predictions[index])
                        expand = probability >= threshold
                        decisions.append(
                            {
                                "previous_depth": previous_depth,
                                "expanded_depth": expanded_depth,
                                "predicted_help_probability": probability,
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
                            "method": (
                                f"utility_stop_{predictor_label}_"
                                f"t{int(round(threshold * 100)):02d}"
                            ),
                            "threshold": threshold,
                            "chosen_scope_depth": chosen_depth,
                            "decisions": decisions,
                            "candidate_blocks": int(selected["candidate_blocks"]),
                            "top_block_ids": selected["top_block_ids"],
                            "out_of_fold": True,
                            "grouped_by_query_id": True,
                            "features_use_target": False,
                            "train_labels_use_future_nll": True,
                            "selection_uses_target": False,
                        }
                    )

    regression_gain_thresholds = (0.0, 0.0025, 0.005, 0.01)
    for predictor_name, predicted_gains in regression_predictions.items():
        predictor_label = "ridge" if predictor_name.endswith("ridge") else "regforest"
        for gain_threshold in regression_gain_thresholds:
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
                        predicted_gain = float(predicted_gains[index])
                        expand = predicted_gain > gain_threshold
                        decisions.append(
                            {
                                "previous_depth": previous_depth,
                                "expanded_depth": expanded_depth,
                                "predicted_marginal_nll_gain": predicted_gain,
                                "gain_threshold": gain_threshold,
                                "expand": expand,
                            }
                        )
                        if not expand:
                            break
                        chosen_depth = expanded_depth
                    selected = retrieval_lookup[(query_id, suffix, chosen_depth)]
                    threshold_label = int(round(gain_threshold * 10_000))
                    policy_rows.append(
                        {
                            "query_id": query_id,
                            "state_suffix_tokens": suffix,
                            "method": (
                                f"utility_stop_{predictor_label}_"
                                f"g{threshold_label:03d}"
                            ),
                            "gain_threshold": gain_threshold,
                            "chosen_scope_depth": chosen_depth,
                            "decisions": decisions,
                            "candidate_blocks": int(selected["candidate_blocks"]),
                            "top_block_ids": selected["top_block_ids"],
                            "out_of_fold": True,
                            "grouped_by_query_id": True,
                            "features_use_target": False,
                            "train_labels_use_future_nll": True,
                            "selection_uses_target": False,
                        }
                    )

    oracle_policy_rows = []
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
                expand = bool(examples[index]["expansion_helped"])
                decisions.append(
                    {
                        "previous_depth": previous_depth,
                        "expanded_depth": expanded_depth,
                        "oracle_expand": expand,
                    }
                )
                if not expand:
                    break
                chosen_depth = expanded_depth
            selected = retrieval_lookup[(query_id, suffix, chosen_depth)]
            oracle_policy_rows.append(
                {
                    "query_id": query_id,
                    "state_suffix_tokens": suffix,
                    "method": "oracle_utility_stop",
                    "chosen_scope_depth": chosen_depth,
                    "decisions": decisions,
                    "candidate_blocks": int(selected["candidate_blocks"]),
                    "top_block_ids": selected["top_block_ids"],
                    "selection_uses_target": True,
                    "diagnostic_only": True,
                }
            )

    output = {
        "source": "grouped out-of-fold marginal scope utility STOP probe",
        "protocol": {
            "queries": 30,
            "states": [128, 512],
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "transition_examples": len(examples),
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "features_use_target": False,
            "train_labels_use_future_nll": True,
            "selection_uses_target": False,
            "thresholds": thresholds,
        },
        "expansion_statistics": {
            "positive_rate": float(labels.mean()),
            "mean_marginal_nll_gain": mean(
                [float(row["marginal_nll_gain"]) for row in examples]
            ),
            "by_state_transition": {
                f"{suffix}_{left}_to_{right}": {
                    "positive_rate": mean(
                        [
                            float(row["expansion_helped"])
                            for row in examples
                            if int(row["state_suffix_tokens"]) == suffix
                            and int(row["previous_depth"]) == left
                            and int(row["expanded_depth"]) == right
                        ]
                    ),
                    "mean_marginal_nll_gain": mean(
                        [
                            float(row["marginal_nll_gain"])
                            for row in examples
                            if int(row["state_suffix_tokens"]) == suffix
                            and int(row["previous_depth"]) == left
                            and int(row["expanded_depth"]) == right
                        ]
                    ),
                }
                for suffix in (128, 512)
                for left, right in TRANSITIONS
            },
        },
        "prediction_quality": {
            name: prediction_metrics(examples, probabilities)
            for name, probabilities in predictions.items()
        },
        "regression_quality": {
            name: regression_metrics(examples, predicted_gains)
            for name, predicted_gains in regression_predictions.items()
        },
        "policy_quality": summarize_policy(
            policy_rows,
            ppl_lookup=ppl_lookup,
            retrieval_lookup=retrieval_lookup,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "oracle_utility_stop_quality": summarize_policy(
            oracle_policy_rows,
            ppl_lookup=ppl_lookup,
            retrieval_lookup=retrieval_lookup,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10_000,
        ),
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    policy_path = Path(args.output_policy_rows)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for row in policy_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
