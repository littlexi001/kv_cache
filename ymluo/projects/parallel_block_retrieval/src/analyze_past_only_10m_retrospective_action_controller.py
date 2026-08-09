#!/usr/bin/env python3
"""Test whether causal retrospective reader loss selects useful future 10M actions."""

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_past_only_10m_dynamic_controller import (
    EVIDENCE,
    METHODS,
    method_config,
    paired_policy,
    read_jsonl,
)


STATES = (128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrospective_rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_retrospective_action64_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--future_reader_rows",
        type=Path,
        nargs="+",
        default=[
            EVIDENCE / f"pg19_past_only_multilevel_10m_ppl_s{state}_rows_20260715.jsonl"
            for state in STATES
        ],
    )
    parser.add_argument(
        "--retrieval_rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_all_states_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_retrospective_action_controller_20260715.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def safe_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        len(left_array) < 3
        or len(np.unique(np.round(left_array, 12))) < 2
        or len(np.unique(np.round(right_array, 12))) < 2
    ):
        return {"n": len(left_array), "rho": None, "p_descriptive": None}
    value = spearmanr(left_array, right_array)
    return {
        "n": len(left_array),
        "rho": float(value.statistic),
        "p_descriptive": float(value.pvalue),
    }


def action_features(
    event: tuple[int, int],
    method: str,
    retrospective: dict[tuple[int, int, str], float],
    retrospective_query_only: dict[tuple[int, int], float],
) -> tuple[list[float], list[str]]:
    action_values = np.asarray(
        [retrospective[(*event, candidate)] for candidate in METHODS], dtype=np.float64
    )
    current = retrospective[(*event, method)]
    delta = current - retrospective_query_only[event]
    centered = current - float(action_values.mean())
    rank = float(np.sum(action_values < current) / max(len(METHODS) - 1, 1))
    sorted_values = np.sort(action_values)
    best_margin = float(sorted_values[1] - sorted_values[0])
    one_hot = [float(method == candidate) for candidate in METHODS]
    values = [
        math.log2(event[1]),
        *method_config(method),
        *one_hot,
        retrospective_query_only[event],
        current,
        delta,
        centered,
        rank,
        float(action_values.std()),
        float(action_values.max() - action_values.min()),
        best_margin,
    ]
    names = [
        "log2_state_tokens",
        "is_global",
        "is_flat_book",
        "configured_book_depth",
        "configured_segment_depth",
        *[f"action_{candidate}" for candidate in METHODS],
        "retrospective_query_only_nll",
        "retrospective_action_nll",
        "retrospective_delta_vs_query_only",
        "retrospective_action_centered",
        "retrospective_action_rank",
        "retrospective_action_std",
        "retrospective_action_range",
        "retrospective_best_second_margin",
    ]
    return values, names


def summarize_policy(
    choices: dict[tuple[int, int], str],
    future: dict[tuple[int, int, str], float],
    future_query_only: dict[tuple[int, int], float],
    retrieval: dict[tuple[int, int, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    event_nll = {}
    costs = []
    blocks = []
    for event, method in choices.items():
        if method == "query_only":
            event_nll[event] = future_query_only[event]
            costs.append(0.0)
            blocks.append(0.0)
        else:
            event_nll[event] = future[(*event, method)]
            costs.append(float(retrieval[(*event, method)]["query_seconds"]))
            blocks.append(float(retrieval[(*event, method)]["candidate_blocks"]))
    delta = float(np.mean([event_nll[event] - future_query_only[event] for event in choices]))
    return (
        {
            "events": len(choices),
            "mean_delta_nll_vs_query_only": delta,
            "geometric_mean_ppl_ratio": float(math.exp(delta)),
            "selection_counts": dict(Counter(choices.values())),
            "mean_selected_retrieval_seconds": float(np.mean(costs)),
            "mean_selected_candidate_blocks": float(np.mean(blocks)),
            "per_state": {
                str(state): {
                    "mean_delta_nll_vs_query_only": float(
                        np.mean(
                            [
                                event_nll[event] - future_query_only[event]
                                for event in choices
                                if event[1] == state
                            ]
                        )
                    ),
                    "selection_counts": dict(
                        Counter(
                            method for event, method in choices.items() if event[1] == state
                        )
                    ),
                }
                for state in STATES
            },
        },
        event_nll,
    )


def prediction_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    action_keys: list[tuple[int, int, str]],
) -> dict[str, Any]:
    centered_prediction = []
    centered_target = []
    within_event = []
    by_event: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(action_keys):
        by_event[(key[0], key[1])].append(index)
    for indices in by_event.values():
        pred = prediction[indices]
        actual = target[indices]
        centered_prediction.extend((pred - pred.mean()).tolist())
        centered_target.extend((actual - actual.mean()).tolist())
        correlation = safe_spearman(pred.tolist(), actual.tolist())
        if correlation["rho"] is not None:
            within_event.append(float(correlation["rho"]))
    return {
        "action_row_mae": float(np.mean(np.abs(prediction - target))),
        "pooled_action_row_spearman": safe_spearman(prediction.tolist(), target.tolist()),
        "within_event_centered_spearman": safe_spearman(
            centered_prediction, centered_target
        ),
        "mean_event_action_rank_spearman": float(np.mean(within_event)),
        "median_event_action_rank_spearman": float(np.median(within_event)),
        "events_with_defined_rank_correlation": len(within_event),
    }


def main() -> None:
    args = parse_args()
    retrospective_rows = read_jsonl(args.retrospective_rows)
    retrospective = {
        (int(row["query_id"]), int(row["state_suffix_tokens"]), str(row["method"])): float(
            row["mean_nll"]
        )
        for row in retrospective_rows
        if row["method"] in METHODS
    }
    retrospective_query_only = {
        (int(row["query_id"]), int(row["state_suffix_tokens"])): float(row["mean_nll"])
        for row in retrospective_rows
        if row["method"] == "query_only"
    }

    future: dict[tuple[int, int, str], float] = {}
    future_query_only: dict[tuple[int, int], float] = {}
    for path in args.future_reader_rows:
        match = re.search(r"ppl_s(\d+)", path.name)
        if not match:
            raise ValueError(f"cannot infer state from {path}")
        state = int(match.group(1))
        for row in read_jsonl(path):
            event = (int(row["query_id"]), state)
            method = str(row["method"])
            if method == "query_only":
                future_query_only[event] = float(row["mean_nll"])
            elif method in METHODS:
                future[(*event, method)] = float(row["mean_nll"])

    retrieval_rows = [
        row
        for row in read_jsonl(args.retrieval_rows)
        if int(row["prefix_tokens"]) in STATES and row["method"] in METHODS
    ]
    retrieval = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in retrieval_rows
    }
    events = sorted(future_query_only)
    expected = len(events) * len(METHODS)
    if len(retrospective) != expected or len(future) != expected:
        raise ValueError("retrospective or future action matrix is incomplete")

    retrospective_deltas = []
    future_deltas = []
    retrospective_centered = []
    future_centered = []
    event_correlations = []
    for event in events:
        retro_values = np.asarray(
            [retrospective[(*event, method)] for method in METHODS], dtype=np.float64
        )
        future_values = np.asarray(
            [future[(*event, method)] for method in METHODS], dtype=np.float64
        )
        retrospective_deltas.extend(
            (retro_values - retrospective_query_only[event]).tolist()
        )
        future_deltas.extend((future_values - future_query_only[event]).tolist())
        retrospective_centered.extend((retro_values - retro_values.mean()).tolist())
        future_centered.extend((future_values - future_values.mean()).tolist())
        correlation = safe_spearman(retro_values.tolist(), future_values.tolist())
        if correlation["rho"] is not None:
            event_correlations.append(float(correlation["rho"]))

    feature_rows = []
    target = []
    groups = []
    action_keys = []
    feature_names = []
    for event in events:
        for method in METHODS:
            values, feature_names = action_features(
                event, method, retrospective, retrospective_query_only
            )
            feature_rows.append(values)
            target.append(future[(*event, method)] - future_query_only[event])
            groups.append(event[0])
            action_keys.append((*event, method))
    features = np.asarray(feature_rows, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=np.int64)

    models = {
        "retrospective_ridge": lambda fold: make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)
        ),
        "retrospective_forest": lambda fold: RandomForestRegressor(
            n_estimators=args.trees,
            max_depth=5,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.seed + fold,
        ),
    }
    predictions = {name: np.full_like(target_array, np.nan) for name in models}
    splits = list(GroupKFold(n_splits=args.folds).split(features, target_array, groups_array))
    state_only_choices: dict[tuple[int, int], str] = {}
    fold_fixed_choices: dict[tuple[int, int], str] = {}
    shrinkage_choices: dict[tuple[int, int], str] = {}
    shrinkage_folds = []
    shrinkage_grid = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]

    retrospective_z: dict[tuple[int, int, str], float] = {}
    for event in events:
        values = np.asarray(
            [retrospective[(*event, method)] for method in METHODS], dtype=np.float64
        )
        scale = float(values.std())
        normalized = (values - values.mean()) / scale if scale > 1e-12 else np.zeros_like(values)
        for method, value in zip(METHODS, normalized):
            retrospective_z[(*event, method)] = float(value)

    for fold, (train, test) in enumerate(splits):
        for name, factory in models.items():
            model = factory(fold)
            model.fit(features[train], target_array[train])
            predictions[name][test] = model.predict(features[test])
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
        fixed = min(METHODS, key=lambda method: float(np.mean(method_values[method])))
        state_method_prior = {
            (state, method): float(np.mean(train_values[(state, method)]))
            for state in STATES
            for method in METHODS
        }
        train_events = sorted({(action_keys[index][0], action_keys[index][1]) for index in train})

        def shrinkage_choice(event: tuple[int, int], weight: float) -> str:
            method = min(
                METHODS,
                key=lambda candidate: state_method_prior[(event[1], candidate)]
                + weight * retrospective_z[(*event, candidate)],
            )
            score = state_method_prior[(event[1], method)] + weight * retrospective_z[
                (*event, method)
            ]
            return method if score < 0.0 else "query_only"

        train_losses = {}
        for weight in shrinkage_grid:
            values = []
            for event in train_events:
                method = shrinkage_choice(event, weight)
                values.append(
                    0.0
                    if method == "query_only"
                    else future[(*event, method)] - future_query_only[event]
                )
            train_losses[weight] = float(np.mean(values))
        selected_weight = min(shrinkage_grid, key=lambda weight: train_losses[weight])
        test_events = {(action_keys[index][0], action_keys[index][1]) for index in test}
        for event in test_events:
            state_only_choices[event] = schedule[event[1]]
            fold_fixed_choices[event] = fixed
            shrinkage_choices[event] = shrinkage_choice(event, selected_weight)
        shrinkage_folds.append(
            {
                "fold": fold,
                "selected_weight": selected_weight,
                "train_mean_delta_by_weight": {
                    str(weight): value for weight, value in train_losses.items()
                },
            }
        )

    direct_choices = {
        event: min(
            ["query_only", *METHODS],
            key=lambda method: (
                retrospective_query_only[event]
                if method == "query_only"
                else retrospective[(*event, method)]
            ),
        )
        for event in events
    }
    learned_choices = {}
    for name, prediction in predictions.items():
        lookup = {key: float(value) for key, value in zip(action_keys, prediction)}
        choices = {}
        for event in events:
            method = min(METHODS, key=lambda candidate: lookup[(*event, candidate)])
            choices[event] = method if lookup[(*event, method)] < 0.0 else "query_only"
        learned_choices[name] = choices
    oracle_choices = {
        event: min(
            ["query_only", *METHODS],
            key=lambda method: (
                future_query_only[event]
                if method == "query_only"
                else future[(*event, method)]
            ),
        )
        for event in events
    }
    policies = {
        "fold_selected_fixed": fold_fixed_choices,
        "state_only": state_only_choices,
        "retrospective_prior_shrinkage": shrinkage_choices,
        "direct_retrospective_argmin": direct_choices,
        **learned_choices,
        "oracle_future_diagnostic": oracle_choices,
    }
    summaries = {}
    event_nll = {}
    for name, choices in policies.items():
        summaries[name], event_nll[name] = summarize_policy(
            choices, future, future_query_only, retrieval
        )

    rng = np.random.default_rng(args.seed)
    paired = {
        name: {
            baseline: paired_policy(
                event_nll[name],
                event_nll[baseline],
                samples=args.bootstrap_samples,
                rng=rng,
            )
            for baseline in (
                "fold_selected_fixed",
                "state_only",
                "direct_retrospective_argmin",
            )
            if baseline != name
        }
        for name in ("retrospective_ridge", "retrospective_forest")
    }
    for name in ("direct_retrospective_argmin", "retrospective_ridge", "retrospective_forest"):
        paired.setdefault(name, {})["query_only"] = paired_policy(
            event_nll[name], future_query_only, samples=args.bootstrap_samples, rng=rng
        )
    paired["retrospective_prior_shrinkage"] = {
        baseline: paired_policy(
            event_nll["retrospective_prior_shrinkage"],
            event_nll[baseline],
            samples=args.bootstrap_samples,
            rng=rng,
        )
        for baseline in (
            "state_only",
            "fold_selected_fixed",
            "direct_retrospective_argmin",
        )
    }
    paired["retrospective_prior_shrinkage"]["query_only"] = paired_policy(
        event_nll["retrospective_prior_shrinkage"],
        future_query_only,
        samples=args.bootstrap_samples,
        rng=rng,
    )

    retrospective_compute_seconds = float(
        np.mean(
            [
                sum(
                    float(row["forward_seconds"])
                    for row in retrospective_rows
                    if int(row["query_id"]) == event[0]
                    and int(row["state_suffix_tokens"]) == event[1]
                )
                for event in events
            ]
        )
    )
    payload = {
        "source": "real strict past-only PG19 9.9M causal retrospective action selection",
        "protocol": {
            "events": len(events),
            "query_groups": len({event[0] for event in events}),
            "states": list(STATES),
            "actions": METHODS,
            "retrospective_observation_tokens": 64,
            "future_evaluation_tokens": 128,
            "final_reader_tokens_per_retrieval_action": 512,
            "selection_uses_future_target": False,
            "retrospective_window_is_observed_at_decision_time": True,
            "cross_validation": f"{args.folds}-fold grouped by query_id",
        },
        "retrospective_signal": {
            "pooled_delta_spearman_with_future_delta": safe_spearman(
                retrospective_deltas, future_deltas
            ),
            "within_event_centered_spearman": safe_spearman(
                retrospective_centered, future_centered
            ),
            "mean_event_action_rank_spearman": float(np.mean(event_correlations)),
            "median_event_action_rank_spearman": float(np.median(event_correlations)),
            "events_with_defined_action_rank": len(event_correlations),
        },
        "policy_summary": summaries,
        "paired_policies": paired,
        "retrospective_prior_shrinkage_folds": shrinkage_folds,
        "oof_prediction_diagnostics": {
            name: prediction_diagnostics(prediction, target_array, action_keys)
            for name, prediction in predictions.items()
        },
        "feature_names": feature_names,
        "compute": {
            "mean_sequential_seconds_to_score_query_only_plus_7_actions": retrospective_compute_seconds,
            "mean_candidate_forward_seconds": float(
                np.mean(
                    [
                        float(row["forward_seconds"])
                        for row in retrospective_rows
                        if row["method"] in METHODS
                    ]
                )
            ),
            "note": "current implementation scores actions sequentially per query; candidate actions can be batched",
        },
        "limitations": [
            "The current retrieval ranking uses the same observed window later replayed for loss, which is causal but can overfit recent lexical content.",
            "Only 30 query groups from PG19 are available.",
            "Seven-way retrospective argmin has multiple-comparison selection bias.",
            "The extra reader forwards are not yet amortized or batched in an online generation system.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "signal": payload["retrospective_signal"],
                "policy_delta": {
                    name: summary["mean_delta_nll_vs_query_only"]
                    for name, summary in summaries.items()
                },
                "compute": payload["compute"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
