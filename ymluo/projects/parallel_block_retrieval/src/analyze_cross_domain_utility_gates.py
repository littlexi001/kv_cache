from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METHODS = ("bm25", "e5", "bm25_e5_rrf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cross-domain future-utility predictability and bounded "
            "counterfactual KEEP policies on PG19 and code continuations."
        )
    )
    parser.add_argument("--pg19_rows", required=True)
    parser.add_argument("--pg19_data_dir", required=True)
    parser.add_argument("--pg19_scope_ids", required=True)
    parser.add_argument("--code_rows", required=True)
    parser.add_argument("--code_data_dir", required=True)
    parser.add_argument("--code_scope_ids", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--probe_budgets", default="1,2,4,8,16,32,64")
    parser.add_argument("--keep_thresholds", default="0,0.05,0.1")
    parser.add_argument("--relative_keep_thresholds", default="0.01,0.02,0.05")
    parser.add_argument("--bootstrap_samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("probe budgets must be positive")
    return values


def parse_floats(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("keep thresholds cannot be empty")
    return values


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def prepare_domain(
    name: str,
    rows_path: str,
    data_dir: str,
    scope_ids_path: str,
    metadata_scope_key: str,
) -> dict[str, Any]:
    rows = read_jsonl(rows_path)
    metadata = read_jsonl(Path(data_dir) / "metadata.jsonl")
    query_scope = {
        int(row["query_id"]): int(row[metadata_scope_key]) for row in metadata
    }
    block_scope_ids = np.load(scope_ids_path, mmap_mode="r")
    base_count = len(block_scope_ids)
    by_query = {
        query_id: [row for row in rows if int(row["query_id"]) == query_id]
        for query_id in sorted(query_scope)
    }
    retrieval_rows = []
    random95_by_query = {}
    for query_id, group in by_query.items():
        random_rows = [row for row in group if "random" in row["origins"]]
        if not random_rows:
            raise RuntimeError(f"{name} query {query_id} has no random windows")
        random95 = float(
            np.quantile([float(row["delta_nll_b"]) for row in random_rows], 0.95)
        )
        random95_by_query[query_id] = random95
        for row in group:
            origins = row["origins"]
            if not any(method in origins for method in METHODS):
                continue
            scopes = set()
            for block_id in row["block_ids"]:
                block_id = int(block_id)
                if block_id < base_count:
                    scope_id = int(block_scope_ids[block_id])
                    if scope_id >= 0:
                        scopes.add(scope_id)
                else:
                    scopes.add(query_scope[query_id])
            ranks = {
                method: int(origins[method]) if method in origins else None
                for method in METHODS
            }
            agreement = sum(value is not None for value in ranks.values())
            item = dict(row)
            item.update(
                {
                    "domain": name,
                    "same_scope": query_scope[query_id] in scopes,
                    "rank_bm25_feature": 1.0 / ranks["bm25"] if ranks["bm25"] else 0.0,
                    "rank_e5_feature": 1.0 / ranks["e5"] if ranks["e5"] else 0.0,
                    "rank_hybrid_feature": (
                        1.0 / ranks["bm25_e5_rrf"]
                        if ranks["bm25_e5_rrf"]
                        else 0.0
                    ),
                    "origin_agreement": agreement / len(METHODS),
                    "bm25_e5_agreement": (
                        ranks["bm25"] is not None and ranks["e5"] is not None
                    ),
                    "observed_delta_nll_a": float(row["delta_nll_a"]),
                    "observed_positive_a": float(row["delta_nll_a"]) > 0,
                    "future_utility_event": float(row["delta_nll_b"]) > random95,
                }
            )
            retrieval_rows.append(item)
    return {
        "name": name,
        "rows": retrieval_rows,
        "query_ids": sorted(query_scope),
        "random95_by_query": random95_by_query,
        "scope_is_observed_metadata": True,
    }


FEATURE_SETS = {
    "rank_only": [
        "rank_bm25_feature",
        "rank_e5_feature",
        "rank_hybrid_feature",
        "origin_agreement",
        "bm25_e5_agreement",
    ],
    "scope_only": ["same_scope"],
    "rank_plus_scope": [
        "rank_bm25_feature",
        "rank_e5_feature",
        "rank_hybrid_feature",
        "origin_agreement",
        "bm25_e5_agreement",
        "same_scope",
    ],
    "observed_utility_only": ["observed_delta_nll_a", "observed_positive_a"],
    "rank_scope_observed_utility": [
        "rank_bm25_feature",
        "rank_e5_feature",
        "rank_hybrid_feature",
        "origin_agreement",
        "bm25_e5_agreement",
        "same_scope",
        "observed_delta_nll_a",
        "observed_positive_a",
    ],
}


def matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[feature]) for feature in features] for row in rows],
        dtype=np.float64,
    )


def selection_quality(
    selected: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    improvements = [float(row["delta_nll_b"]) for row in selected]
    nll = mean(float(row["mean_nll_b"]) for row in selected)
    baseline_nll = mean(float(row["baseline_nll_b"]) for row in selected)
    return {
        "queries": len(selected),
        "mean_nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "baseline_ppl": math.exp(min(baseline_nll, 20.0)),
        "mean_nll_improvement_vs_query_only": mean(improvements),
        "improvement_bootstrap95": bootstrap_ci(improvements, samples, seed),
        "positive_future_utility_rate": mean(value > 0 for value in improvements),
        "future_event_rate": mean(bool(row["future_utility_event"]) for row in selected),
        "same_scope_rate": mean(bool(row["same_scope"]) for row in selected),
    }


def static_selection(domain: dict[str, Any], method: str) -> list[dict[str, Any]]:
    selected = []
    for query_id in domain["query_ids"]:
        group = [
            row
            for row in domain["rows"]
            if int(row["query_id"]) == query_id and method in row["origins"]
        ]
        selected.append(min(group, key=lambda row: int(row["origins"][method])))
    return selected


def cross_domain_models(
    train_domain: dict[str, Any],
    test_domain: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    train_rows = train_domain["rows"]
    test_rows = test_domain["rows"]
    train_y = np.asarray(
        [bool(row["future_utility_event"]) for row in train_rows], dtype=np.int64
    )
    test_y = np.asarray(
        [bool(row["future_utility_event"]) for row in test_rows], dtype=np.int64
    )
    output = []
    for index, (feature_name, features) in enumerate(FEATURE_SETS.items()):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        model.fit(matrix(train_rows, features), train_y)
        probabilities = model.predict_proba(matrix(test_rows, features))[:, 1]
        selected = []
        for query_id in test_domain["query_ids"]:
            indices = [
                row_index
                for row_index, row in enumerate(test_rows)
                if int(row["query_id"]) == query_id
            ]
            best_index = max(
                indices,
                key=lambda item: (
                    float(probabilities[item]),
                    float(test_rows[item]["rank_hybrid_feature"]),
                    float(test_rows[item]["rank_e5_feature"]),
                    float(test_rows[item]["rank_bm25_feature"]),
                ),
            )
            selected.append(test_rows[best_index])
        output.append(
            {
                "train_domain": train_domain["name"],
                "test_domain": test_domain["name"],
                "feature_set": feature_name,
                "features": features,
                "requires_candidate_A_probe": "observed_delta_nll_a" in features,
                "scope_feature_uses_observed_metadata": "same_scope" in features,
                "candidate_positive_rate": float(test_y.mean()),
                "candidate_roc_auc": float(roc_auc_score(test_y, probabilities)),
                "candidate_average_precision": float(
                    average_precision_score(test_y, probabilities)
                ),
                "top1_selection_on_future_B": selection_quality(
                    selected, samples=samples, seed=seed + index
                ),
            }
        )
    return output


def probe_policy_quality(
    domain: dict[str, Any],
    *,
    method: str,
    budgets: list[int],
    thresholds: list[float],
    relative_thresholds: list[float],
    ordering: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    static = static_selection(domain, method)
    static_lookup = {
        int(row["query_id"]): float(row["mean_nll_b"]) for row in static
    }
    output = []
    for budget in budgets:
        best_a_selected = []
        best_a_probes = []
        early_selected: dict[float, list[dict[str, Any]]] = {
            threshold: [] for threshold in thresholds
        }
        early_probes: dict[float, list[int]] = {
            threshold: [] for threshold in thresholds
        }
        relative_selected: dict[float, list[dict[str, Any]]] = {
            threshold: [] for threshold in relative_thresholds
        }
        relative_probes: dict[float, list[int]] = {
            threshold: [] for threshold in relative_thresholds
        }
        for query_id in domain["query_ids"]:
            ranked = sorted(
                [
                    row
                    for row in domain["rows"]
                    if int(row["query_id"]) == query_id and method in row["origins"]
                ],
                key=(
                    (lambda row: (not bool(row["same_scope"]), int(row["origins"][method])))
                    if ordering == "same_scope_then_rank"
                    else (lambda row: (False, int(row["origins"][method])))
                ),
            )
            probed = ranked[: min(budget, len(ranked))]
            best_a_selected.append(
                max(probed, key=lambda row: float(row["delta_nll_a"]))
            )
            best_a_probes.append(len(probed))
            for threshold in thresholds:
                chosen = None
                used = 0
                for used, row in enumerate(probed, start=1):
                    if float(row["delta_nll_a"]) >= threshold:
                        chosen = row
                        break
                if chosen is None:
                    chosen = max(probed, key=lambda row: float(row["delta_nll_a"]))
                    used = len(probed)
                early_selected[threshold].append(chosen)
                early_probes[threshold].append(used)
            for threshold in relative_thresholds:
                chosen = None
                used = 0
                for used, row in enumerate(probed, start=1):
                    relative_gain = float(row["delta_nll_a"]) / max(
                        float(row["baseline_nll_a"]), 1.0e-8
                    )
                    if relative_gain >= threshold:
                        chosen = row
                        break
                if chosen is None:
                    chosen = max(probed, key=lambda row: float(row["delta_nll_a"]))
                    used = len(probed)
                relative_selected[threshold].append(chosen)
                relative_probes[threshold].append(used)

        def summarize_policy(
            policy: str,
            selected: list[dict[str, Any]],
            probes: list[int],
            policy_seed: int,
        ) -> dict[str, Any]:
            quality = selection_quality(selected, samples=samples, seed=policy_seed)
            improvement_vs_static = [
                static_lookup[int(row["query_id"])] - float(row["mean_nll_b"])
                for row in selected
            ]
            return {
                "domain": domain["name"],
                "method": method,
                "candidate_ordering": ordering,
                "policy": policy,
                "max_probe_budget": budget,
                "mean_A_probes": mean(probes),
                "selection_uses_future_B": False,
                "quality_on_future_B": quality,
                "mean_nll_improvement_over_static_top1": mean(
                    improvement_vs_static
                ),
                "improvement_over_static_bootstrap95": bootstrap_ci(
                    improvement_vs_static, samples, policy_seed + 1000
                ),
                "wins_over_static_top1": sum(
                    value > 0 for value in improvement_vs_static
                ),
                "losses_to_static_top1": sum(
                    value < 0 for value in improvement_vs_static
                ),
                "ties_with_static_top1": sum(
                    value == 0 for value in improvement_vs_static
                ),
            }

        output.append(
            summarize_policy(
                "best_observed_A_within_budget",
                best_a_selected,
                best_a_probes,
                seed + budget,
            )
        )
        for threshold_index, threshold in enumerate(thresholds):
            output.append(
                summarize_policy(
                    f"first_delta_A_ge_{threshold:g}_else_best_seen",
                    early_selected[threshold],
                    early_probes[threshold],
                    seed + budget * 10 + threshold_index,
                )
            )
        for threshold_index, threshold in enumerate(relative_thresholds):
            output.append(
                summarize_policy(
                    f"first_relative_delta_A_ge_{threshold:g}_else_best_seen",
                    relative_selected[threshold],
                    relative_probes[threshold],
                    seed + budget * 20 + threshold_index,
                )
            )
    return output


def main() -> None:
    args = parse_args()
    budgets = parse_ints(args.probe_budgets)
    thresholds = parse_floats(args.keep_thresholds)
    relative_thresholds = parse_floats(args.relative_keep_thresholds)
    pg19 = prepare_domain(
        "pg19_books",
        args.pg19_rows,
        args.pg19_data_dir,
        args.pg19_scope_ids,
        "book_index",
    )
    code = prepare_domain(
        "code_repositories",
        args.code_rows,
        args.code_data_dir,
        args.code_scope_ids,
        "repo_index",
    )
    cross_domain = cross_domain_models(
        pg19,
        code,
        samples=args.bootstrap_samples,
        seed=args.seed,
    ) + cross_domain_models(
        code,
        pg19,
        samples=args.bootstrap_samples,
        seed=args.seed + 100,
    )
    probe_policies = []
    for domain_index, domain in enumerate((pg19, code)):
        for method_index, method in enumerate(METHODS):
            for ordering_index, ordering in enumerate(
                ("global_rank", "same_scope_then_rank")
            ):
                probe_policies.extend(
                    probe_policy_quality(
                        domain,
                        method=method,
                        budgets=budgets,
                        thresholds=thresholds,
                        relative_thresholds=relative_thresholds,
                        ordering=ordering,
                        samples=args.bootstrap_samples,
                        seed=(
                            args.seed
                            + domain_index * 1000
                            + method_index * 100
                            + ordering_index * 10_000
                        ),
                    )
                )

    static_baselines = []
    for domain_index, domain in enumerate((pg19, code)):
        for method_index, method in enumerate(METHODS):
            selected = static_selection(domain, method)
            static_baselines.append(
                {
                    "domain": domain["name"],
                    "method": method,
                    "quality_on_future_B": selection_quality(
                        selected,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 5000 + domain_index * 10 + method_index,
                    ),
                }
            )
    output = {
        "source": "cross-domain future utility gates on real PG19 and code memories",
        "protocol": {
            "future_B_is_never_used_as_a_feature": True,
            "scope_is_observed_metadata": True,
            "rank_scope_models_require_no_candidate_reader_probe": True,
            "observed_utility_models_require_A_probe_for_each_scored_candidate": True,
            "bounded_probe_policies_only_observe_at_most_the_named_budget": True,
            "relative_keep_threshold_is_delta_A_over_query_only_A_NLL": True,
            "contains_synthetic_text": False,
        },
        "domains": {
            pg19["name"]: {
                "queries": len(pg19["query_ids"]),
                "retrieval_candidate_rows": len(pg19["rows"]),
            },
            code["name"]: {
                "queries": len(code["query_ids"]),
                "retrieval_candidate_rows": len(code["rows"]),
            },
        },
        "cross_domain_models": cross_domain,
        "static_baselines": static_baselines,
        "bounded_probe_policies": probe_policies,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
