from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DOMAINS = ("xsum", "pg19", "code")
POLICY_BUDGETS = (0.10, 0.20, 1.0 / 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate target-free incremental refresh gates with query-grouped and "
            "leave-one-domain-out generalization."
        )
    )
    parser.add_argument("--xsum_rows", required=True)
    parser.add_argument("--pg19_rows", required=True)
    parser.add_argument("--code_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else None


def feature_families(names: list[str]) -> dict[str, list[str]]:
    state = [
        "previous_prefix_tokens",
        "current_prefix_tokens",
        "prefix_ratio",
        "new_tokens",
        "new_unique_tokens",
        "query_token_set_jaccard",
        "query_e5_cosine",
    ]
    agreement = state + [
        "current_bm25_e5_top8_jaccard",
        "current_bm25_e5_top64_jaccard",
        "temporal_rrf_top8_jaccard",
        "temporal_rrf_top64_jaccard",
    ]
    compact = agreement + [
        "bm25_positive_fraction",
        "bm25_normalized_entropy",
        "bm25_top1_z",
        "bm25_top8_softmax_mass",
        "bm25_score_temporal_spearman",
        "bm25_top1_score_delta_previous_std",
        "e5_normalized_entropy",
        "e5_top1_z",
        "e5_top8_softmax_mass",
        "e5_score_temporal_spearman",
        "e5_top1_score_delta_previous_std",
        "rrf_normalized_entropy",
        "rrf_top1_z",
        "rrf_top8_softmax_mass",
        "rrf_score_temporal_spearman",
        "rrf_top1_score_delta_previous_std",
    ]
    for family in (state, agreement, compact):
        missing = set(family) - set(names)
        if missing:
            raise ValueError(f"missing refresh features: {sorted(missing)}")
    return {
        "state": state,
        "agreement": agreement,
        "compact": compact,
        "all": names,
    }


def matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    result = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ValueError("non-finite refresh feature matrix")
    return result


def logistic_factory(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        ),
    )


def forest_classifier_factory(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=12,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def ridge_factory(seed: int) -> Any:
    del seed
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def forest_regressor_factory(seed: int) -> Any:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=12,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "auc": safe_auc(labels, predictions),
        "average_precision": float(average_precision_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, predictions)),
        "positive_rate": float(labels.mean()),
    }


def regression_metrics(target: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    correlation = spearmanr(target, predictions)
    return {
        "spearman": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "mae": float(mean_absolute_error(target, predictions)),
    }


def grouped_bootstrap_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if metric == "auc":
            if np.unique(labels[indices]).size < 2:
                continue
            values.append(float(roc_auc_score(labels[indices], predictions[indices])))
        elif metric == "spearman":
            value = float(spearmanr(labels[indices], predictions[indices]).statistic)
            if math.isfinite(value):
                values.append(value)
        else:
            raise ValueError(metric)
    return interval(np.asarray(values, dtype=np.float64))


def fit_grouped_predictions(
    matrices: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    *,
    classification: bool,
    folds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    factories: dict[str, Callable[[int], Any]]
    if classification:
        factories = {
            "logistic": logistic_factory,
            "forest": forest_classifier_factory,
        }
    else:
        factories = {
            "ridge": ridge_factory,
            "forest": forest_regressor_factory,
        }
    output = {
        f"{family}_{model}": np.zeros(len(target), dtype=np.float64)
        for family in matrices
        for model in factories
    }
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(target)), target, groups)
    ):
        for family, values in matrices.items():
            for model_name, factory in factories.items():
                model = factory(seed + 1000 * fold)
                model.fit(values[train], target[train])
                key = f"{family}_{model_name}"
                if classification:
                    output[key][test] = model.predict_proba(values[test])[:, 1]
                else:
                    output[key][test] = model.predict(values[test])
    return output


def fit_lodo_predictions(
    matrices: dict[str, np.ndarray],
    target: np.ndarray,
    domains: np.ndarray,
    *,
    classification: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    factories: dict[str, Callable[[int], Any]]
    if classification:
        factories = {"logistic": logistic_factory, "forest": forest_classifier_factory}
    else:
        factories = {"ridge": ridge_factory, "forest": forest_regressor_factory}
    output = {
        f"{family}_{model}": np.zeros(len(target), dtype=np.float64)
        for family in matrices
        for model in factories
    }
    for domain_index, domain in enumerate(DOMAINS):
        train = np.flatnonzero(domains != domain)
        test = np.flatnonzero(domains == domain)
        for family, values in matrices.items():
            for model_name, factory in factories.items():
                model = factory(seed + 10_000 * domain_index)
                model.fit(values[train], target[train])
                key = f"{family}_{model_name}"
                if classification:
                    output[key][test] = model.predict_proba(values[test])[:, 1]
                else:
                    output[key][test] = model.predict(values[test])
    return output


def model_quality(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    domains: np.ndarray,
    *,
    classification: bool,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for index, (name, values) in enumerate(predictions.items()):
        item = (
            classification_metrics(target, values)
            if classification
            else regression_metrics(target, values)
        )
        item["query_cluster_bootstrap95"] = grouped_bootstrap_metric(
            target,
            values,
            groups,
            metric="auc" if classification else "spearman",
            samples=samples,
            seed=seed + 1000 * (index + 1),
        )
        item["by_domain"] = {}
        for domain in DOMAINS:
            selected = domains == domain
            item["by_domain"][domain] = (
                classification_metrics(target[selected], values[selected])
                if classification
                else regression_metrics(target[selected], values[selected])
            )
        output[name] = item
    return output


def top_budget_mask(
    scores: np.ndarray, domains: np.ndarray, budget: float
) -> np.ndarray:
    output = np.zeros(len(scores), dtype=bool)
    for domain in DOMAINS:
        indices = np.flatnonzero(domains == domain)
        count = max(1, int(math.ceil(budget * len(indices))))
        order = indices[np.argsort(-scores[indices], kind="stable")]
        output[order[:count]] = True
    return output


def policy_quality(
    rows: list[dict[str, Any]],
    refresh: np.ndarray,
    groups: np.ndarray,
    *,
    method: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    full_any = np.asarray(
        [float(row["labels"]["full_source_any_at_8"]) for row in rows]
    )
    restricted_any = np.asarray(
        [float(row["labels"]["restricted_source_any_at_8"]) for row in rows]
    )
    full_recall = np.asarray(
        [float(row["labels"]["full_source_recall_at_8"]) for row in rows]
    )
    restricted_recall = np.asarray(
        [float(row["labels"]["restricted_source_recall_at_8"]) for row in rows]
    )
    gain = full_any - restricted_any
    selected_any = np.where(refresh, full_any, restricted_any)
    selected_recall = np.where(refresh, full_recall, restricted_recall)
    memory_blocks = np.asarray([float(row["memory_blocks"]) for row in rows])
    frontier_blocks = np.asarray([float(row["frontier_blocks"]) for row in rows])
    accessed = frontier_blocks + refresh.astype(np.float64) * memory_blocks
    unique = np.unique(groups)
    by_group_delta = np.asarray(
        [
            float((selected_any[groups == group] - restricted_any[groups == group]).mean())
            for group in unique
        ]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(unique), size=(samples, len(unique))
    )
    positive_need = gain > 0
    return {
        "method": method,
        "refresh_rate": float(refresh.mean()),
        "source_any_at_8": float(selected_any.mean()),
        "source_recall_at_8": float(selected_recall.mean()),
        "source_any_delta_vs_never_refresh": float(
            selected_any.mean() - restricted_any.mean()
        ),
        "source_any_delta_query_bootstrap95": interval(
            by_group_delta[indices].mean(axis=1)
        ),
        "strict_source_need_recall": float(refresh[positive_need].mean())
        if positive_need.any()
        else None,
        "harmful_refresh_rate_among_refreshed": float((gain[refresh] < 0).mean())
        if refresh.any()
        else 0.0,
        "mean_index_blocks_accessed_per_event": float(accessed.mean()),
        "index_block_access_reduction_vs_every_global_search": float(
            memory_blocks.mean() / accessed.mean()
        ),
    }


def build_policies(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    domains: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
    prefix: str,
) -> list[dict[str, Any]]:
    count = len(rows)
    current_prefix = np.asarray([int(row["current_prefix_tokens"]) for row in rows])
    source_gain = np.asarray(
        [float(row["labels"]["global_refresh_source_any_gain"]) for row in rows]
    )
    specs: dict[str, np.ndarray] = {
        "never": np.zeros(count, dtype=bool),
        "always_global": np.ones(count, dtype=bool),
        "fixed_at_16": current_prefix == 16,
        "fixed_at_32": current_prefix == 32,
        "fixed_at_64": current_prefix == 64,
        "oracle_positive_source_gain": source_gain > 0,
    }
    query_drift = -np.asarray([float(row["features"]["query_e5_cosine"]) for row in rows])
    for budget in POLICY_BUDGETS:
        suffix = int(round(100 * budget))
        specs[f"query_drift_top{suffix}pct"] = top_budget_mask(
            query_drift, domains, budget
        )
        for target_name in ("miss25", "source_need"):
            for family in ("state_logistic", "compact_logistic", "all_forest"):
                key = f"{target_name}_{family}"
                specs[f"{key}_top{suffix}pct"] = top_budget_mask(
                    predictions[key], domains, budget
                )
    return [
        policy_quality(
            rows,
            refresh,
            groups,
            method=f"{prefix}{name}",
            samples=samples,
            seed=seed + index * 1000,
        )
        for index, (name, refresh) in enumerate(specs.items())
    ]


def main() -> None:
    args = parse_args()
    rows = (
        read_jsonl(args.xsum_rows)
        + read_jsonl(args.pg19_rows)
        + read_jsonl(args.code_rows)
    )
    if any(
        bool(row["online_features_use_current_global_ranking"])
        or bool(row["online_features_use_target"])
        or not bool(row["full_global_ranking_used_only_for_labels"])
        for row in rows
    ):
        raise ValueError("refresh feature protocol contains leakage")
    names = sorted(rows[0]["features"])
    if any(sorted(row["features"]) != names for row in rows):
        raise ValueError("refresh feature schema differs across domains")
    families = feature_families(names)
    matrices = {name: matrix(rows, keys) for name, keys in families.items()}
    domains = np.asarray([str(row["dataset"]) for row in rows])
    groups = np.asarray(
        [f"{row['dataset']}:{int(row['query_id'])}" for row in rows]
    )
    miss_fraction = np.asarray(
        [float(row["labels"]["full_top8_frontier_miss_fraction"]) for row in rows]
    )
    miss25 = np.asarray(
        [bool(row["labels"]["frontier_miss_above_25pct"]) for row in rows],
        dtype=np.int64,
    )
    source_need = np.asarray(
        [
            bool(row["labels"]["global_refresh_strictly_needed_for_source"])
            for row in rows
        ],
        dtype=np.int64,
    )

    oof_miss = fit_grouped_predictions(
        matrices,
        miss25,
        groups,
        classification=True,
        folds=args.folds,
        seed=args.seed,
    )
    oof_source = fit_grouped_predictions(
        matrices,
        source_need,
        groups,
        classification=True,
        folds=args.folds,
        seed=args.seed + 100_000,
    )
    oof_regression = fit_grouped_predictions(
        matrices,
        miss_fraction,
        groups,
        classification=False,
        folds=args.folds,
        seed=args.seed + 200_000,
    )
    lodo_miss = fit_lodo_predictions(
        matrices,
        miss25,
        domains,
        classification=True,
        seed=args.seed + 300_000,
    )
    lodo_source = fit_lodo_predictions(
        matrices,
        source_need,
        domains,
        classification=True,
        seed=args.seed + 400_000,
    )
    lodo_regression = fit_lodo_predictions(
        matrices,
        miss_fraction,
        domains,
        classification=False,
        seed=args.seed + 500_000,
    )

    oof_policy_predictions = {
        **{f"miss25_{key}": value for key, value in oof_miss.items()},
        **{f"source_need_{key}": value for key, value in oof_source.items()},
    }
    lodo_policy_predictions = {
        **{f"miss25_{key}": value for key, value in lodo_miss.items()},
        **{f"source_need_{key}": value for key, value in lodo_source.items()},
    }
    output = {
        "source": "target-free incremental frontier refresh gate",
        "protocol": {
            "events": len(rows),
            "queries": len(np.unique(groups)),
            "events_by_domain": {
                domain: int((domains == domain).sum()) for domain in DOMAINS
            },
            "feature_count": len(names),
            "feature_families": families,
            "online_features_use_current_global_ranking": False,
            "online_features_use_target": False,
            "full_global_ranking_used_only_for_labels": True,
            "query_grouped_folds": args.folds,
            "leave_one_domain_out": True,
            "policy_budgets_predeclared": list(POLICY_BUDGETS),
        },
        "target_statistics": {
            "frontier_miss_above_25pct_rate": float(miss25.mean()),
            "strict_source_refresh_need_rate": float(source_need.mean()),
            "mean_frontier_miss_fraction": float(miss_fraction.mean()),
            "by_domain": {
                domain: {
                    "events": int((domains == domain).sum()),
                    "frontier_miss_above_25pct_rate": float(
                        miss25[domains == domain].mean()
                    ),
                    "strict_source_refresh_need_rate": float(
                        source_need[domains == domain].mean()
                    ),
                    "mean_frontier_miss_fraction": float(
                        miss_fraction[domains == domain].mean()
                    ),
                }
                for domain in DOMAINS
            },
        },
        "query_grouped_quality": {
            "miss25_classification": model_quality(
                oof_miss,
                miss25,
                groups,
                domains,
                classification=True,
                samples=args.bootstrap_samples,
                seed=args.seed + 600_000,
            ),
            "strict_source_need_classification": model_quality(
                oof_source,
                source_need,
                groups,
                domains,
                classification=True,
                samples=args.bootstrap_samples,
                seed=args.seed + 700_000,
            ),
            "miss_fraction_regression": model_quality(
                oof_regression,
                miss_fraction,
                groups,
                domains,
                classification=False,
                samples=args.bootstrap_samples,
                seed=args.seed + 800_000,
            ),
        },
        "leave_one_domain_out_quality": {
            "miss25_classification": model_quality(
                lodo_miss,
                miss25,
                groups,
                domains,
                classification=True,
                samples=args.bootstrap_samples,
                seed=args.seed + 900_000,
            ),
            "strict_source_need_classification": model_quality(
                lodo_source,
                source_need,
                groups,
                domains,
                classification=True,
                samples=args.bootstrap_samples,
                seed=args.seed + 1_000_000,
            ),
            "miss_fraction_regression": model_quality(
                lodo_regression,
                miss_fraction,
                groups,
                domains,
                classification=False,
                samples=args.bootstrap_samples,
                seed=args.seed + 1_100_000,
            ),
        },
        "query_grouped_policies": build_policies(
            rows,
            oof_policy_predictions,
            domains,
            groups,
            samples=args.bootstrap_samples,
            seed=args.seed + 1_200_000,
            prefix="oof_",
        ),
        "leave_one_domain_out_policies": build_policies(
            rows,
            lodo_policy_predictions,
            domains,
            groups,
            samples=args.bootstrap_samples,
            seed=args.seed + 1_300_000,
            prefix="lodo_",
        ),
    }
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    prediction_path = Path(args.output_predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "dataset": row["dataset"],
                        "query_id": int(row["query_id"]),
                        "prefix_transition": row["prefix_transition"],
                        "labels": row["labels"],
                        "query_grouped_predictions": {
                            "miss25": {
                                key: float(value[index]) for key, value in oof_miss.items()
                            },
                            "source_need": {
                                key: float(value[index])
                                for key, value in oof_source.items()
                            },
                            "miss_fraction": {
                                key: float(value[index])
                                for key, value in oof_regression.items()
                            },
                        },
                        "lodo_predictions": {
                            "miss25": {
                                key: float(value[index]) for key, value in lodo_miss.items()
                            },
                            "source_need": {
                                key: float(value[index])
                                for key, value in lodo_source.items()
                            },
                            "miss_fraction": {
                                key: float(value[index])
                                for key, value in lodo_regression.items()
                            },
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
