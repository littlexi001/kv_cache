#!/usr/bin/env python3
"""Target-free, query-grouped controller for 10M multilevel retrieval actions."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "doc" / "1b_context_search_research_exploration" / "evidence"
METHODS = [
    "global_bm25_unigram",
    "flat_book_bm25_depth8",
    "multilevel_bm25_book8_segment8",
    "multilevel_bm25_book8_segment32",
    "multilevel_bm25_book8_segment128",
    "multilevel_bm25_book32_segment8",
    "multilevel_bm25_book32_segment32",
]
ONLINE_SCOPE_FEATURES = [
    "active_scopes",
    "positive_scope_scores",
    "scope_query_features",
    "scope_top1_score",
    "scope_margin_1_2",
    "scope_margin_3_4",
    "scope_margin_8_9",
    "scope_margin_16_17",
    "scope_margin_32_33",
    "scope_normalized_margin_1_2",
    "scope_top1_positive_share",
    "scope_top3_positive_share",
    "scope_top8_positive_share",
    "scope_top16_positive_share",
    "scope_top32_positive_share",
    "scope_score_normalized_entropy",
    "scope_score_hhi",
    "scope_top1_z",
]


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
        default=EVIDENCE / "pg19_past_only_multilevel_10m_dynamic_controller_20260715.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_config(method: str) -> list[float]:
    book = re.search(r"book(\d+)", method)
    flat = re.search(r"depth(\d+)", method)
    segment = re.search(r"segment(\d+)", method)
    return [
        float(method == "global_bm25_unigram"),
        float(method.startswith("flat_book")),
        float(book.group(1) if book else flat.group(1) if flat else 135),
        float(segment.group(1) if segment else 2477),
    ]


def row_features(row: dict[str, Any]) -> list[float]:
    state = int(row["prefix_tokens"])
    config = method_config(str(row["method"]))
    values = [
        math.log2(state),
        *config,
        math.log1p(float(row["candidate_books"])),
        math.log1p(float(row["candidate_segments"])),
        math.log1p(float(row["candidate_blocks"])),
        math.log1p(float(row.get("selected_segments", row["candidate_segments"]))),
    ]
    values.extend(float(row.get(field, 0.0)) for field in ONLINE_SCOPE_FEATURES)
    return values


def bootstrap_query_ci(
    values_by_query: dict[int, list[float]], *, samples: int, rng: np.random.Generator
) -> list[float]:
    query_ids = sorted(values_by_query)
    matrix = np.asarray(
        [[float(np.mean(values_by_query[qid]))] for qid in query_ids], dtype=np.float64
    ).reshape(-1)
    indices = rng.integers(0, len(matrix), size=(samples, len(matrix)))
    means = matrix[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_policy(
    candidate: dict[tuple[int, int], float],
    baseline: dict[tuple[int, int], float],
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    keys = sorted(set(candidate) & set(baseline))
    deltas_by_query: dict[int, list[float]] = defaultdict(list)
    for query_id, state in keys:
        deltas_by_query[query_id].append(candidate[(query_id, state)] - baseline[(query_id, state)])
    query_deltas = {
        query_id: float(np.mean(values)) for query_id, values in deltas_by_query.items()
    }
    wins = sum(value < -1e-12 for value in query_deltas.values())
    losses = sum(value > 1e-12 for value in query_deltas.values())
    ties = len(query_deltas) - wins - losses
    return {
        "events": len(keys),
        "query_groups": len(query_deltas),
        "delta_definition": "candidate NLL minus baseline NLL; negative favors candidate",
        "mean_delta_nll": float(
            np.mean([candidate[key] - baseline[key] for key in keys])
        ),
        "query_cluster_bootstrap95": bootstrap_query_ci(
            deltas_by_query, samples=samples, rng=rng
        ),
        "query_group_wins": wins,
        "query_group_losses": losses,
        "query_group_ties": ties,
        "query_group_exact_sign_p": (
            float(binomtest(wins, wins + losses, p=0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def summarize_policy(
    choices: dict[tuple[int, int], str],
    nll_lookup: dict[tuple[int, int, str], float],
    query_only: dict[tuple[int, int], float],
) -> dict[str, Any]:
    deltas = {
        key: nll_lookup[(*key, method)] - query_only[key]
        if method != "query_only"
        else 0.0
        for key, method in choices.items()
    }
    per_state: dict[str, Any] = {}
    for state in sorted({state for _, state in choices}):
        state_keys = [key for key in choices if key[1] == state]
        state_delta = float(np.mean([deltas[key] for key in state_keys]))
        per_state[str(state)] = {
            "events": len(state_keys),
            "mean_delta_nll_vs_query_only": state_delta,
            "geometric_mean_ppl_ratio": float(math.exp(state_delta)),
            "selection_counts": dict(Counter(choices[key] for key in state_keys)),
        }
    mean_delta = float(np.mean(list(deltas.values())))
    return {
        "events": len(choices),
        "mean_delta_nll_vs_query_only": mean_delta,
        "geometric_mean_ppl_ratio": float(math.exp(mean_delta)),
        "selection_counts": dict(Counter(choices.values())),
        "per_state": per_state,
        "event_nll": {
            f"{query_id}:{state}": (
                query_only[(query_id, state)] + deltas[(query_id, state)]
            )
            for query_id, state in sorted(choices)
        },
    }


def main() -> None:
    args = parse_args()
    retrieval_rows = [
        row for row in read_jsonl(args.retrieval_rows) if str(row["method"]) in METHODS
    ]
    retrieval_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in retrieval_rows
    }

    nll_lookup: dict[tuple[int, int, str], float] = {}
    query_only: dict[tuple[int, int], float] = {}
    for path in args.reader_rows:
        match = re.search(r"ppl_s(\d+)", path.name)
        if not match:
            raise ValueError(f"cannot infer state length from {path}")
        state = int(match.group(1))
        for row in read_jsonl(path):
            key = (int(row["query_id"]), state)
            method = str(row["method"])
            if method == "query_only":
                query_only[key] = float(row["mean_nll"])
            elif method in METHODS:
                nll_lookup[(*key, method)] = float(row["mean_nll"])

    events = sorted(query_only)
    missing = [
        (*key, method)
        for key in events
        for method in METHODS
        if (*key, method) not in nll_lookup or (*key, method) not in retrieval_lookup
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} method/event rows; first={missing[0]}")

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    action_keys: list[tuple[int, int, str]] = []
    groups: list[int] = []
    for query_id, state in events:
        for method in METHODS:
            key = (query_id, state, method)
            feature_rows.append(row_features(retrieval_lookup[key]))
            targets.append(nll_lookup[key] - query_only[(query_id, state)])
            action_keys.append(key)
            groups.append(query_id)
    features = np.asarray(feature_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=np.int64)

    predictions = np.full_like(target, np.nan)
    feature_importance = np.zeros(features.shape[1], dtype=np.float64)
    state_only_choices: dict[tuple[int, int], str] = {}
    fold_fixed_choices: dict[tuple[int, int], str] = {}
    fold_schedules: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train, test) in enumerate(splitter.split(features, target, groups_array)):
        model = RandomForestRegressor(
            n_estimators=args.trees,
            max_depth=5,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.seed + fold,
        )
        model.fit(features[train], target[train])
        predictions[test] = model.predict(features[test])
        feature_importance += model.feature_importances_ / args.folds

        train_keys = [action_keys[index] for index in train]
        train_targets = target[train]
        train_values: dict[tuple[int, str], list[float]] = defaultdict(list)
        method_values: dict[str, list[float]] = defaultdict(list)
        for (_, state, method), value in zip(train_keys, train_targets):
            train_values[(state, method)].append(float(value))
            method_values[method].append(float(value))
        state_schedule = {
            state: min(
                METHODS,
                key=lambda method: float(np.mean(train_values[(state, method)])),
            )
            for state in sorted({key[1] for key in action_keys})
        }
        fold_fixed = min(METHODS, key=lambda method: float(np.mean(method_values[method])))
        test_events = sorted({(action_keys[index][0], action_keys[index][1]) for index in test})
        for event in test_events:
            state_only_choices[event] = state_schedule[event[1]]
            fold_fixed_choices[event] = fold_fixed
        fold_schedules.append(
            {
                "fold": fold,
                "held_out_query_groups": sorted({event[0] for event in test_events}),
                "state_schedule": {str(key): value for key, value in state_schedule.items()},
                "fold_fixed_method": fold_fixed,
            }
        )
    if np.any(~np.isfinite(predictions)):
        raise RuntimeError("out-of-fold predictions are incomplete")

    action_prediction = {key: float(value) for key, value in zip(action_keys, predictions)}
    learned_choices: dict[tuple[int, int], str] = {}
    for event in events:
        method = min(METHODS, key=lambda item: action_prediction[(*event, item)])
        learned_choices[event] = (
            method if action_prediction[(*event, method)] < 0.0 else "query_only"
        )

    fixed_choices = {
        method: {event: method for event in events} for method in METHODS
    }
    fixed_summary = {
        method: summarize_policy(choices, nll_lookup, query_only)
        for method, choices in fixed_choices.items()
    }
    fixed_best = min(
        METHODS, key=lambda method: fixed_summary[method]["mean_delta_nll_vs_query_only"]
    )
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
    learned_summary = summarize_policy(learned_choices, nll_lookup, query_only)
    state_only_summary = summarize_policy(state_only_choices, nll_lookup, query_only)
    fold_fixed_summary = summarize_policy(fold_fixed_choices, nll_lookup, query_only)
    oracle_summary = summarize_policy(oracle_choices, nll_lookup, query_only)

    learned_nll = {
        event: (
            query_only[event]
            if learned_choices[event] == "query_only"
            else nll_lookup[(*event, learned_choices[event])]
        )
        for event in events
    }
    rng = np.random.default_rng(args.seed)
    paired = {
        method: paired_policy(
            learned_nll,
            {event: nll_lookup[(*event, method)] for event in events},
            samples=args.bootstrap_samples,
            rng=rng,
        )
        for method in METHODS
    }
    paired["query_only"] = paired_policy(
        learned_nll,
        query_only,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    state_only_nll = {
        event: nll_lookup[(*event, state_only_choices[event])] for event in events
    }
    state_only_paired = {
        "fold_selected_fixed_policy": paired_policy(
            state_only_nll,
            {event: nll_lookup[(*event, fold_fixed_choices[event])] for event in events},
            samples=args.bootstrap_samples,
            rng=rng,
        ),
        "query_only": paired_policy(
            state_only_nll,
            query_only,
            samples=args.bootstrap_samples,
            rng=rng,
        ),
    }

    feature_names = [
        "log2_state_tokens",
        "is_global",
        "is_flat_book",
        "configured_book_depth",
        "configured_segment_depth",
        "log_candidate_books",
        "log_candidate_segments",
        "log_candidate_blocks",
        "log_selected_segments",
        *ONLINE_SCOPE_FEATURES,
    ]
    top_features = sorted(
        zip(feature_names, feature_importance), key=lambda item: item[1], reverse=True
    )
    payload = {
        "source": "real strict past-only PG19 9.9M, four generation-state lengths",
        "protocol": {
            "events": len(events),
            "query_groups": len({query_id for query_id, _ in events}),
            "state_suffix_tokens": sorted({state for _, state in events}),
            "candidate_actions": METHODS,
            "final_reader_tokens_for_every_retrieval_action": 512,
            "selection_uses_target_online": False,
            "offline_training_label": "paired future-128-token reader delta NLL",
            "cross_validation": f"{args.folds}-fold grouped by query_id",
            "model": "random forest regressor, fixed hyperparameters",
            "forbidden_online_fields": [
                "future target tokens",
                "true_scope_rank",
                "book_router_hit",
                "segment_router_hit",
                "same_scope metrics",
            ],
        },
        "fixed_policies": fixed_summary,
        "fixed_best_over_all_events_diagnostic": fixed_best,
        "fold_selected_fixed_oof_policy": fold_fixed_summary,
        "state_only_oof_policy": state_only_summary,
        "state_only_oof_fold_schedules": fold_schedules,
        "paired_state_only_oof": state_only_paired,
        "learned_target_free_oof_policy": learned_summary,
        "paired_learned_vs_baselines": paired,
        "oracle_action_upper_bound_diagnostic": oracle_summary,
        "oof_regression": {
            "mae_delta_nll": float(np.mean(np.abs(predictions - target))),
            "spearman_note": "action-level rows share events; use only as descriptive fit quality",
            "top_feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in top_features[:12]
            ],
        },
        "limitations": [
            "Only 30 independent query groups are available.",
            "Offline controller labels come from Qwen3-0.6B on PG19 and require cross-domain validation.",
            "The fixed-best and oracle policies are diagnostics, not leakage-free online policies.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "fixed_best": fixed_best,
                "fixed_best_delta": fixed_summary[fixed_best]["mean_delta_nll_vs_query_only"],
                "fold_fixed_delta": fold_fixed_summary["mean_delta_nll_vs_query_only"],
                "state_only_delta": state_only_summary["mean_delta_nll_vs_query_only"],
                "learned_delta": learned_summary["mean_delta_nll_vs_query_only"],
                "learned_selection": learned_summary["selection_counts"],
                "oracle_delta": oracle_summary["mean_delta_nll_vs_query_only"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
