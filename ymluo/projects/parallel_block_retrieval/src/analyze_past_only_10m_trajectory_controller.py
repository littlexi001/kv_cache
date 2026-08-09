#!/usr/bin/env python3
"""Compare static and trajectory-conditioned retrieval controllers on real 10M data."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from analyze_past_only_10m_dynamic_controller import (
    EVIDENCE,
    METHODS,
    ONLINE_SCOPE_FEATURES,
    method_config,
    paired_policy,
    read_jsonl,
)


CANONICAL_ROUTER_METHOD = "multilevel_bm25_book8_segment32"
STATES = (128, 256, 512)
TOP_SCOPE_DEPTHS = (3, 8, 16, 32, 64)
TOP_BLOCK_DEPTHS = (8, 64, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_all_states_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--reader_rows",
        type=Path,
        nargs="+",
        default=[
            EVIDENCE / f"pg19_past_only_multilevel_10m_ppl_s{state}_rows_20260715.jsonl"
            for state in (64, 128, 256, 512)
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_trajectory_controller_20260715.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def jaccard(left: list[int], right: list[int], depth: int) -> float:
    left_set = set(left[:depth])
    right_set = set(right[:depth])
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def coverage(left: list[int], right: list[int], left_depth: int, right_depth: int) -> float:
    left_set = set(left[:left_depth])
    right_set = set(right[:right_depth])
    return float(len(left_set & right_set) / len(left_set)) if left_set else 1.0


def score_vector(row: dict[str, Any]) -> dict[int, float]:
    return {
        int(scope): float(score)
        for scope, score in zip(row["scope_top64_rows"], row["scope_top64_scores"])
    }


def vector_cosine(left: dict[int, float], right: dict[int, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_values = np.asarray([left.get(key, 0.0) for key in keys], dtype=np.float64)
    right_values = np.asarray([right.get(key, 0.0) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    return float(np.dot(left_values, right_values) / denominator) if denominator else 0.0


def action_features(row: dict[str, Any]) -> tuple[list[float], list[str]]:
    method = str(row["method"])
    one_hot = [float(method == candidate) for candidate in METHODS]
    values = [
        *method_config(method),
        *one_hot,
        math.log1p(float(row["candidate_books"])),
        math.log1p(float(row["candidate_segments"])),
        math.log1p(float(row["candidate_blocks"])),
        math.log1p(float(row.get("selected_segments", row["candidate_segments"]))),
    ]
    names = [
        "is_global",
        "is_flat_book",
        "configured_book_depth",
        "configured_segment_depth",
        *[f"action_{method_name}" for method_name in METHODS],
        "log_candidate_books",
        "log_candidate_segments",
        "log_candidate_blocks",
        "log_selected_segments",
    ]
    return values, names


def static_router_features(
    current_router: dict[str, Any], action_row: dict[str, Any]
) -> tuple[list[float], list[str]]:
    action, action_names = action_features(action_row)
    values = [math.log2(float(current_router["prefix_tokens"])), *action]
    values.extend(float(current_router[field]) for field in ONLINE_SCOPE_FEATURES)
    names = ["log2_state_tokens", *action_names, *ONLINE_SCOPE_FEATURES]
    return values, names


def router_trajectory_features(
    current_router: dict[str, Any],
    previous_router: dict[str, Any],
    action_row: dict[str, Any],
) -> tuple[list[float], list[str]]:
    values, names = static_router_features(current_router, action_row)
    for field in ONLINE_SCOPE_FEATURES:
        values.append(float(current_router[field]) - float(previous_router[field]))
        names.append(f"delta_{field}")
    current_scopes = [int(item) for item in current_router["scope_top64_rows"]]
    previous_scopes = [int(item) for item in previous_router["scope_top64_rows"]]
    for depth in TOP_SCOPE_DEPTHS:
        values.append(jaccard(current_scopes, previous_scopes, depth))
        names.append(f"scope_top{depth}_jaccard_previous")
    values.extend(
        [
            coverage(current_scopes, previous_scopes, 8, 64),
            coverage(previous_scopes, current_scopes, 8, 64),
            vector_cosine(score_vector(current_router), score_vector(previous_router)),
        ]
    )
    names.extend(
        [
            "current_top8_covered_by_previous_top64",
            "previous_top8_covered_by_current_top64",
            "scope_top64_score_cosine",
        ]
    )
    return values, names


def frontier_trajectory_features(
    current_router: dict[str, Any],
    previous_router: dict[str, Any],
    current_action: dict[str, Any],
    previous_action: dict[str, Any],
) -> tuple[list[float], list[str]]:
    values, names = router_trajectory_features(
        current_router, previous_router, current_action
    )
    current_blocks = [int(item) for item in current_action["top_block_ids"]]
    previous_blocks = [int(item) for item in previous_action["top_block_ids"]]
    for depth in TOP_BLOCK_DEPTHS:
        values.append(jaccard(current_blocks, previous_blocks, depth))
        names.append(f"block_top{depth}_jaccard_previous")
    values.extend(
        [
            coverage(current_blocks, previous_blocks, 8, 512),
            coverage(previous_blocks, current_blocks, 8, 512),
            coverage(current_blocks, previous_blocks, 64, 512),
            coverage(previous_blocks, current_blocks, 64, 512),
        ]
    )
    names.extend(
        [
            "current_top8_covered_by_previous_top512",
            "previous_top8_covered_by_current_top512",
            "current_top64_covered_by_previous_top512",
            "previous_top64_covered_by_current_top512",
        ]
    )
    return values, names


def fit_oof(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    predictions = np.full_like(target, np.nan)
    importance = np.zeros(features.shape[1], dtype=np.float64)
    splits = list(GroupKFold(n_splits=folds).split(features, target, groups))
    for fold, (train, test) in enumerate(splits):
        model = RandomForestRegressor(
            n_estimators=trees,
            max_depth=5,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed + fold,
        )
        model.fit(features[train], target[train])
        predictions[test] = model.predict(features[test])
        importance += model.feature_importances_ / folds
    if np.any(~np.isfinite(predictions)):
        raise RuntimeError("incomplete OOF predictions")
    return predictions, importance, splits


def summarize_policy(
    choices: dict[tuple[int, int], str],
    nll_lookup: dict[tuple[int, int, str], float],
    query_only: dict[tuple[int, int], float],
    retrieval_lookup: dict[tuple[int, int, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    event_nll: dict[tuple[int, int], float] = {}
    retrieval_seconds: list[float] = []
    candidate_blocks: list[float] = []
    for event, method in choices.items():
        if method == "query_only":
            event_nll[event] = query_only[event]
            retrieval_seconds.append(0.0)
            candidate_blocks.append(0.0)
        else:
            event_nll[event] = nll_lookup[(*event, method)]
            row = retrieval_lookup[(*event, method)]
            retrieval_seconds.append(float(row["query_seconds"]))
            candidate_blocks.append(float(row["candidate_blocks"]))
    delta = float(np.mean([event_nll[event] - query_only[event] for event in choices]))
    per_state = {}
    for state in STATES:
        state_events = [event for event in choices if event[1] == state]
        state_delta = float(
            np.mean([event_nll[event] - query_only[event] for event in state_events])
        )
        per_state[str(state)] = {
            "events": len(state_events),
            "mean_delta_nll_vs_query_only": state_delta,
            "selection_counts": dict(Counter(choices[event] for event in state_events)),
        }
    return (
        {
            "events": len(choices),
            "mean_delta_nll_vs_query_only": delta,
            "geometric_mean_ppl_ratio": float(math.exp(delta)),
            "selection_counts": dict(Counter(choices.values())),
            "mean_selected_action_retrieval_seconds": float(np.mean(retrieval_seconds)),
            "mean_selected_action_candidate_blocks": float(np.mean(candidate_blocks)),
            "per_state": per_state,
        },
        event_nll,
    )


def controller_diagnostics(
    predictions: np.ndarray,
    target: np.ndarray,
    action_keys: list[tuple[int, int, str]],
    choices: dict[tuple[int, int], str],
    nll_lookup: dict[tuple[int, int, str], float],
    query_only: dict[tuple[int, int], float],
) -> dict[str, Any]:
    event_predictions: dict[tuple[int, int], list[float]] = defaultdict(list)
    event_targets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key, prediction, actual in zip(action_keys, predictions, target):
        event = (key[0], key[1])
        event_predictions[event].append(float(prediction))
        event_targets[event].append(float(actual))
    rank_correlations = []
    for event in sorted(event_predictions):
        prediction_values = np.asarray(event_predictions[event], dtype=np.float64)
        target_values = np.asarray(event_targets[event], dtype=np.float64)
        if (
            len(np.unique(np.round(prediction_values, 12))) > 1
            and len(np.unique(np.round(target_values, 12))) > 1
        ):
            value = float(spearmanr(prediction_values, target_values).statistic)
            if np.isfinite(value):
                rank_correlations.append(value)
    exact_best = 0
    within_005 = 0
    regrets = []
    for event, chosen in choices.items():
        candidates = {
            "query_only": query_only[event],
            **{method: nll_lookup[(*event, method)] for method in METHODS},
        }
        oracle_value = min(candidates.values())
        chosen_value = candidates[chosen]
        regret = chosen_value - oracle_value
        regrets.append(regret)
        exact_best += int(abs(regret) < 1e-12)
        within_005 += int(regret <= 0.005)
    overall = spearmanr(predictions, target)
    return {
        "action_row_mae_delta_nll": float(np.mean(np.abs(predictions - target))),
        "action_row_spearman_descriptive": float(overall.statistic),
        "mean_within_event_action_rank_spearman": float(np.mean(rank_correlations)),
        "median_within_event_action_rank_spearman": float(np.median(rank_correlations)),
        "events_with_nonconstant_actions": len(rank_correlations),
        "exact_oracle_action_rate": exact_best / len(choices),
        "within_0.005_nll_of_oracle_rate": within_005 / len(choices),
        "mean_action_regret_vs_oracle": float(np.mean(regrets)),
    }


def top_importance(names: list[str], values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:15]
    ]


def main() -> None:
    args = parse_args()
    retrieval_rows = [
        row for row in read_jsonl(args.retrieval_rows) if str(row["method"]) in METHODS
    ]
    retrieval_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in retrieval_rows
    }
    router_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"])): row
        for row in retrieval_rows
        if row["method"] == CANONICAL_ROUTER_METHOD
    }

    nll_lookup: dict[tuple[int, int, str], float] = {}
    query_only: dict[tuple[int, int], float] = {}
    selected_contexts: dict[tuple[int, int, str], tuple[int, ...]] = {}
    for path in args.reader_rows:
        match = re.search(r"ppl_s(\d+)", path.name)
        if not match:
            raise ValueError(f"cannot infer state from {path}")
        state = int(match.group(1))
        for row in read_jsonl(path):
            event = (int(row["query_id"]), state)
            method = str(row["method"])
            if method == "query_only":
                query_only[event] = float(row["mean_nll"])
            elif method in METHODS:
                key = (*event, method)
                nll_lookup[key] = float(row["mean_nll"])
                selected_contexts[key] = tuple(int(item) for item in row["selected_block_ids"])

    events = sorted(event for event in query_only if event[1] in STATES)
    previous_state = {128: 64, 256: 128, 512: 256}
    static_rows: list[list[float]] = []
    router_rows: list[list[float]] = []
    frontier_rows: list[list[float]] = []
    target: list[float] = []
    groups: list[int] = []
    action_keys: list[tuple[int, int, str]] = []
    static_names: list[str] = []
    router_names: list[str] = []
    frontier_names: list[str] = []
    for query_id, state in events:
        current_router = router_lookup[(query_id, state)]
        previous_router = router_lookup[(query_id, previous_state[state])]
        for method in METHODS:
            key = (query_id, state, method)
            current_action = retrieval_lookup[key]
            previous_action = retrieval_lookup[(query_id, previous_state[state], method)]
            static, static_names = static_router_features(current_router, current_action)
            router, router_names = router_trajectory_features(
                current_router, previous_router, current_action
            )
            frontier, frontier_names = frontier_trajectory_features(
                current_router, previous_router, current_action, previous_action
            )
            static_rows.append(static)
            router_rows.append(router)
            frontier_rows.append(frontier)
            target.append(nll_lookup[key] - query_only[(query_id, state)])
            groups.append(query_id)
            action_keys.append(key)

    matrices = {
        "static_router": np.asarray(static_rows, dtype=np.float64),
        "router_trajectory": np.asarray(router_rows, dtype=np.float64),
        "frontier_trajectory": np.asarray(frontier_rows, dtype=np.float64),
    }
    names = {
        "static_router": static_names,
        "router_trajectory": router_names,
        "frontier_trajectory": frontier_names,
    }
    target_array = np.asarray(target, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=np.int64)

    predictions: dict[str, np.ndarray] = {}
    importances: dict[str, np.ndarray] = {}
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    for index, (name, matrix) in enumerate(matrices.items()):
        prediction, importance, current_splits = fit_oof(
            matrix,
            target_array,
            groups_array,
            folds=args.folds,
            trees=args.trees,
            seed=args.seed + index * 100,
        )
        predictions[name] = prediction
        importances[name] = importance
        if splits is None:
            splits = current_splits

    assert splits is not None
    action_prediction = {
        name: {key: float(value) for key, value in zip(action_keys, prediction)}
        for name, prediction in predictions.items()
    }
    learned_choices: dict[str, dict[tuple[int, int], str]] = {}
    for name, lookup in action_prediction.items():
        choices = {}
        for event in events:
            method = min(METHODS, key=lambda candidate: lookup[(*event, candidate)])
            choices[event] = method if lookup[(*event, method)] < 0.0 else "query_only"
        learned_choices[name] = choices

    fold_fixed_choices: dict[tuple[int, int], str] = {}
    state_only_choices: dict[tuple[int, int], str] = {}
    fold_schedules = []
    for fold, (train, test) in enumerate(splits):
        train_values: dict[tuple[int, str], list[float]] = defaultdict(list)
        method_values: dict[str, list[float]] = defaultdict(list)
        for index in train:
            _, state, method = action_keys[index]
            value = float(target_array[index])
            train_values[(state, method)].append(value)
            method_values[method].append(value)
        schedule = {
            state: min(
                METHODS, key=lambda method: float(np.mean(train_values[(state, method)]))
            )
            for state in STATES
        }
        fixed_method = min(
            METHODS, key=lambda method: float(np.mean(method_values[method]))
        )
        test_events = sorted({(action_keys[index][0], action_keys[index][1]) for index in test})
        for event in test_events:
            state_only_choices[event] = schedule[event[1]]
            fold_fixed_choices[event] = fixed_method
        fold_schedules.append(
            {
                "fold": fold,
                "state_schedule": {str(state): method for state, method in schedule.items()},
                "fixed_method": fixed_method,
            }
        )

    policies: dict[str, dict[tuple[int, int], str]] = {
        "fold_selected_fixed": fold_fixed_choices,
        "state_only": state_only_choices,
        **learned_choices,
    }
    oracle_choices = {
        event: min(
            ["query_only", *METHODS],
            key=lambda method: (
                query_only[event]
                if method == "query_only"
                else nll_lookup[(*event, method)]
            ),
        )
        for event in events
    }
    summaries: dict[str, Any] = {}
    event_nll: dict[str, dict[tuple[int, int], float]] = {}
    diagnostics: dict[str, Any] = {}
    for name, choices in policies.items():
        summaries[name], event_nll[name] = summarize_policy(
            choices, nll_lookup, query_only, retrieval_lookup
        )
        if name in predictions:
            diagnostics[name] = controller_diagnostics(
                predictions[name],
                target_array,
                action_keys,
                choices,
                nll_lookup,
                query_only,
            )
    oracle_summary, _ = summarize_policy(
        oracle_choices, nll_lookup, query_only, retrieval_lookup
    )

    rng = np.random.default_rng(args.seed)
    paired = {}
    for name in ("router_trajectory", "frontier_trajectory"):
        paired[name] = {
            baseline: paired_policy(
                event_nll[name],
                event_nll[baseline],
                samples=args.bootstrap_samples,
                rng=rng,
            )
            for baseline in ("static_router", "state_only", "fold_selected_fixed")
        }
        paired[name]["query_only"] = paired_policy(
            event_nll[name], query_only, samples=args.bootstrap_samples, rng=rng
        )

    unique_contexts_by_state = {}
    selection_opportunities_by_state = {}
    for state in STATES:
        counts = []
        nll_ranges = []
        for query_id, event_state in events:
            if event_state != state:
                continue
            counts.append(
                len({selected_contexts[(query_id, state, method)] for method in METHODS})
            )
            action_nll = [nll_lookup[(query_id, state, method)] for method in METHODS]
            nll_ranges.append(max(action_nll) - min(action_nll))
        unique_contexts_by_state[str(state)] = {
            "mean_unique_contexts_among_7_actions": float(np.mean(counts)),
            "min": int(min(counts)),
            "max": int(max(counts)),
        }
        selection_opportunities_by_state[str(state)] = {
            "mean_action_nll_range": float(np.mean(nll_ranges)),
            "events_with_more_than_one_context": float(np.mean(np.asarray(counts) > 1)),
            "events_with_action_nll_range_gt_0.005": float(
                np.mean(np.asarray(nll_ranges) > 0.005)
            ),
            "events_with_action_nll_range_gt_0.01": float(
                np.mean(np.asarray(nll_ranges) > 0.01)
            ),
        }

    all_action_observation_cost = float(
        np.mean(
            [
                sum(retrieval_lookup[(*event, method)]["query_seconds"] for method in METHODS)
                for event in events
            ]
        )
    )
    payload = {
        "source": "real strict past-only PG19 9.9M trajectory-conditioned action selection",
        "protocol": {
            "events": len(events),
            "query_groups": len({query_id for query_id, _ in events}),
            "states_with_previous_state": list(STATES),
            "candidate_actions": METHODS,
            "final_reader_tokens_per_retrieval_action": 512,
            "future_target_tokens": 128,
            "cross_validation": f"{args.folds}-fold grouped by query_id",
            "online_target_or_scope_labels_used": False,
            "offline_training_label": "future reader delta NLL",
            "static_router_features": "current state, shared scope score geometry, action configuration",
            "router_trajectory_features": "static plus previous-to-current scope rank/score-set changes",
            "frontier_trajectory_features": "router trajectory plus action-specific block frontier churn",
        },
        "policy_summary": summaries,
        "controller_diagnostics": diagnostics,
        "oracle_action_upper_bound_diagnostic": oracle_summary,
        "paired_trajectory_policies": paired,
        "feature_importance": {
            name: top_importance(names[name], importances[name]) for name in matrices
        },
        "fold_schedules": fold_schedules,
        "action_context_redundancy": unique_contexts_by_state,
        "selection_opportunities": selection_opportunities_by_state,
        "cost_boundary": {
            "mean_seconds_if_all_7_action_rankings_are_materialized": all_action_observation_cost,
            "note": (
                "router_trajectory can be computed before action selection; frontier_trajectory currently "
                "requires maintaining or materializing action-specific rankings, so its quality must be "
                "discounted by this observation cost"
            ),
        },
        "limitations": [
            "Only 30 independent queries and 90 state transitions are available.",
            "All data are PG19; cross-domain trajectory-controller transfer is not established.",
            "The random-forest hyperparameters are fixed, but the action set was designed from prior experiments.",
            "Retrieval timings are CPU prototype timings and exclude asynchronous overlap and storage I/O.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "policy_delta_nll": {
                    name: summary["mean_delta_nll_vs_query_only"]
                    for name, summary in summaries.items()
                },
                "paired": paired,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
