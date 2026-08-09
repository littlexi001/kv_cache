from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_natural_operator_library import ridge_predict, standardize_apply, standardize_fit, write_csv


SCORE_FEATURES = [
    "qk_top1",
    "qk_margin12",
    "qk_margin14",
    "qk_std",
    "qk_entropy",
    "lexical_top1",
    "lexical_margin12",
    "lexical_margin14",
    "lexical_std",
    "lexical_entropy",
    "lexical_nonzero_fraction",
    "qk_lexical_topk_jaccard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a query-disjoint, conformal per-head operator router from exact "
            "attention-output-distortion teacher labels."
        )
    )
    parser.add_argument("--distortion_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--relative_error_threshold", type=float, default=0.05)
    parser.add_argument("--conformal_quantile", type=float, default=0.95)
    parser.add_argument(
        "--conformal_scope",
        choices=["global", "layer", "head"],
        default="head",
    )
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--gqa_group_size", type=int, default=2)
    parser.add_argument(
        "--interaction_mode",
        choices=["none", "layer_score"],
        default="none",
        help="Optional structured interactions for context-dependent score slopes.",
    )
    return parser.parse_args()


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "query_id": int(raw["query_id"]),
                    "layer": int(raw["layer"]),
                    "query_head": int(raw["query_head"]),
                    "selected_blocks": int(raw["selected_blocks"]),
                    "selected_block_ids": [
                        int(item) for item in json.loads(raw["selected_block_ids"])
                    ],
                    "relative_output_l2": float(raw["relative_output_l2"]),
                    **{name: float(raw[name]) for name in SCORE_FEATURES},
                }
            )
    return rows


def query_split(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    dataset_by_query: dict[int, str] = {}
    for row in rows:
        dataset_by_query[row["query_id"]] = str(row["dataset"])
    output: dict[int, str] = {}
    for dataset in sorted(set(dataset_by_query.values())):
        query_ids = sorted(
            query_id for query_id, current in dataset_by_query.items() if current == dataset
        )
        for local_index, query_id in enumerate(query_ids):
            if local_index % 2 == 1:
                output[query_id] = "test"
            elif (local_index // 2) % 2 == 0:
                output[query_id] = "fit"
            else:
                output[query_id] = "conformal"
    return output


def feature_vector(
    row: dict[str, Any],
    num_layers: int,
    num_heads: int,
    interaction_mode: str = "none",
) -> np.ndarray:
    continuous = np.asarray(
        [
            row["layer"] / max(num_layers - 1, 1),
            row["query_head"] / max(num_heads - 1, 1),
            *[row[name] for name in SCORE_FEATURES],
        ],
        dtype=np.float64,
    )
    identity = np.zeros(num_layers * num_heads, dtype=np.float64)
    identity[row["layer"] * num_heads + row["query_head"]] = 1.0
    parts = [continuous, identity]
    if interaction_mode == "layer_score":
        score_values = np.asarray([row[name] for name in SCORE_FEATURES], dtype=np.float64)
        interaction = np.zeros(num_layers * len(SCORE_FEATURES), dtype=np.float64)
        start = row["layer"] * len(SCORE_FEATURES)
        interaction[start : start + len(SCORE_FEATURES)] = score_values
        parts.append(interaction)
    elif interaction_mode != "none":
        raise ValueError(f"Unknown interaction_mode={interaction_mode!r}")
    return np.concatenate(parts)


def choose_alpha_group_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    query_ids: np.ndarray,
    alphas: Sequence[float],
) -> float:
    unique_queries = np.asarray(sorted(set(query_ids.tolist())))
    best_alpha = float(alphas[0])
    best_loss = float("inf")
    for alpha in alphas:
        losses: list[float] = []
        for parity in [0, 1]:
            heldout_queries = set(unique_queries[parity::2].tolist())
            test = np.asarray([query_id in heldout_queries for query_id in query_ids])
            train = ~test
            if not train.any() or not test.any():
                continue
            train_x, mean, scale = standardize_fit(matrix[train])
            test_x = standardize_apply(matrix[test], mean, scale)
            prediction, _ = ridge_predict(
                train_x, target[train, None], test_x, float(alpha)
            )
            losses.extend(np.square(prediction[:, 0] - target[test]).tolist())
        mean_loss = statistics.fmean(losses) if losses else float("inf")
        if (mean_loss, float(alpha)) < (best_loss, best_alpha):
            best_loss, best_alpha = mean_loss, float(alpha)
    return best_alpha


def choose_alphas_group_cv_multi(
    matrix: np.ndarray,
    targets: np.ndarray,
    query_ids: np.ndarray,
    action_names: Sequence[str],
    alphas: Sequence[float],
) -> dict[str, float]:
    """Select each action's ridge alpha while sharing every matrix solve."""
    unique_queries = np.asarray(sorted(set(query_ids.tolist())))
    losses_by_action = {
        action: {float(alpha): [] for alpha in alphas} for action in action_names
    }
    for alpha in alphas:
        for parity in [0, 1]:
            heldout_queries = set(unique_queries[parity::2].tolist())
            test = np.asarray([query_id in heldout_queries for query_id in query_ids])
            train = ~test
            if not train.any() or not test.any():
                continue
            train_x, mean, scale = standardize_fit(matrix[train])
            test_x = standardize_apply(matrix[test], mean, scale)
            prediction, _ = ridge_predict(
                train_x, targets[train], test_x, float(alpha)
            )
            squared = np.square(prediction - targets[test])
            for action_index, action in enumerate(action_names):
                losses_by_action[action][float(alpha)].extend(
                    squared[:, action_index].tolist()
                )
    output: dict[str, float] = {}
    for action in action_names:
        output[action] = min(
            (float(alpha) for alpha in alphas),
            key=lambda alpha: (
                statistics.fmean(losses_by_action[action][alpha])
                if losses_by_action[action][alpha]
                else float("inf"),
                alpha,
            ),
        )
    return output


def higher_quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    ordered = np.sort(values)
    index = min(int(np.ceil(quantile * len(ordered))) - 1, len(ordered) - 1)
    return float(ordered[max(index, 0)])


def conformal_corrections(
    residual: np.ndarray,
    keys: Sequence[tuple[int, int, int, int]],
    target_keys: Sequence[tuple[int, int, int, int]],
    quantile: float,
    scope: str,
) -> tuple[np.ndarray, list[float]]:
    if scope == "global":
        correction = higher_quantile(residual, quantile)
        return np.full(len(target_keys), correction), [correction]

    def group_key(key: tuple[int, int, int, int]) -> tuple[int, ...]:
        if scope == "layer":
            return (key[1],)
        if scope == "head":
            return (key[1], key[2])
        raise ValueError(f"unknown conformal scope {scope}")

    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for key, value in zip(keys, residual):
        grouped[group_key(key)].append(float(value))
    global_fallback = higher_quantile(residual, quantile)
    by_group = {
        key: higher_quantile(np.asarray(values), quantile)
        for key, values in grouped.items()
    }
    corrections = np.asarray(
        [by_group.get(group_key(key), global_fallback) for key in target_keys],
        dtype=np.float64,
    )
    return corrections, list(by_group.values())


def summarize_policy(
    name: str,
    selected_actions: Sequence[str],
    selected_blocks: np.ndarray,
    selected_error: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return {
        "policy": name,
        "samples": len(selected_actions),
        "mean_selected_blocks": float(selected_blocks.mean()),
        "p95_selected_blocks": float(np.quantile(selected_blocks, 0.95)),
        "mean_relative_output_l2": float(selected_error.mean()),
        "p95_relative_output_l2": float(np.quantile(selected_error, 0.95)),
        "violation_rate": float(np.mean(selected_error > threshold)),
        "action_counts": dict(sorted(Counter(selected_actions).items())),
    }


def gqa_physical_summary(
    keys: Sequence[tuple[int, int, int, int]],
    grouped: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]],
    selected_actions: Sequence[str],
    gqa_group_size: int,
) -> dict[str, float]:
    unions: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    full_blocks: dict[tuple[int, int, int, int], int] = {}
    for key, action in zip(keys, selected_actions):
        query_id, layer, query_head, query_position = key
        physical_key = (query_id, layer, query_head // gqa_group_size, query_position)
        unions[physical_key].update(grouped[key][action]["selected_block_ids"])
        full_blocks[physical_key] = grouped[key]["full"]["selected_blocks"]
    union_values = np.asarray([len(value) for value in unions.values()], dtype=np.float64)
    full_values = np.asarray([full_blocks[key] for key in unions], dtype=np.float64)
    return {
        "mean_physical_gqa_blocks": float(union_values.mean()),
        "p95_physical_gqa_blocks": float(np.quantile(union_values, 0.95)),
        "mean_full_physical_gqa_blocks": float(full_values.mean()),
        "physical_gqa_saving_rate": float(1.0 - union_values.sum() / full_values.sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.distortion_rows))
    split = query_split(rows)
    deployable = ["streaming", "lexical_blocks", "uniform", "qk_top_blocks"]
    actions = [*deployable, "full"]
    grouped: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["query_id"], row["layer"], row["query_head"], int(row["query_position"]))
        grouped[key][str(row["action"])] = row
    keys = sorted(grouped)
    num_layers = max(row["layer"] for row in rows) + 1
    num_heads = max(row["query_head"] for row in rows) + 1
    features = np.stack(
        [
            feature_vector(
                grouped[key]["full"],
                num_layers,
                num_heads,
                interaction_mode=args.interaction_mode,
            )
            for key in keys
        ]
    )
    query_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    split_values = np.asarray([split[key[0]] for key in keys])
    fit = split_values == "fit"
    conformal = split_values == "conformal"
    test = split_values == "test"
    if not fit.any() or not conformal.any() or not test.any():
        raise ValueError("fit/conformal/test split is empty")

    predictions: dict[str, np.ndarray] = {}
    upper_bounds: dict[str, np.ndarray] = {}
    model_rows: list[dict[str, Any]] = []
    deployment_models: dict[str, Any] = {}
    target_matrix = np.column_stack(
        [
            np.asarray(
                [grouped[key][action]["relative_output_l2"] for key in keys]
            )
            for action in deployable
        ]
    )
    alphas_by_action = choose_alphas_group_cv_multi(
        features[fit],
        target_matrix[fit],
        query_ids[fit],
        deployable,
        parse_floats(args.alphas),
    )
    fit_x, mean, scale = standardize_fit(features[fit])
    conformal_x = standardize_apply(features[conformal], mean, scale)
    test_x = standardize_apply(features[test], mean, scale)
    for action_index, action in enumerate(deployable):
        target = target_matrix[:, action_index]
        alpha = alphas_by_action[action]
        conformal_prediction, _ = ridge_predict(
            fit_x, target[fit, None], conformal_x, alpha
        )
        test_prediction, _ = ridge_predict(fit_x, target[fit, None], test_x, alpha)
        residual = target[conformal] - conformal_prediction[:, 0]
        conformal_keys = [key for key, keep in zip(keys, conformal) if keep]
        test_keys_for_correction = [key for key, keep in zip(keys, test) if keep]
        correction, correction_values = conformal_corrections(
            residual,
            conformal_keys,
            test_keys_for_correction,
            args.conformal_quantile,
            args.conformal_scope,
        )
        predictions[action] = test_prediction[:, 0]
        # Relative L2 error is non-negative. Ridge extrapolation can be negative;
        # clipping keeps the one-sided conformal quantity a valid error scale and
        # prevents negative values from influencing equal-budget tie breaks.
        upper_bounds[action] = np.maximum(0.0, test_prediction[:, 0] + correction)
        design = np.column_stack([np.ones(len(fit_x)), fit_x])
        penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        weights = (
            np.linalg.pinv(design.T @ design + penalty)
            @ design.T
            @ target[fit, None]
        )[:, 0]
        global_correction = higher_quantile(residual, args.conformal_quantile)
        correction_by_group: dict[str, float] = {}
        if args.conformal_scope != "global":
            grouped_residuals: dict[tuple[int, ...], list[float]] = defaultdict(list)
            for residual_key, value in zip(conformal_keys, residual):
                if args.conformal_scope == "layer":
                    group_key = (residual_key[1],)
                else:
                    group_key = (residual_key[1], residual_key[2])
                grouped_residuals[group_key].append(float(value))
            correction_by_group = {
                ":".join(str(item) for item in group_key): higher_quantile(
                    np.asarray(values), args.conformal_quantile
                )
                for group_key, values in grouped_residuals.items()
            }
        deployment_models[action] = {
            "alpha": alpha,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "weights": weights.tolist(),
            "global_correction": global_correction,
            "correction_by_group": correction_by_group,
        }
        model_rows.append(
            {
                "action": action,
                "alpha": alpha,
                "conformal_scope": args.conformal_scope,
                "mean_conformal_correction": statistics.fmean(correction_values),
                "p95_conformal_correction": float(np.quantile(correction_values, 0.95)),
                "max_conformal_correction": max(correction_values),
                "fit_queries": len(set(query_ids[fit].tolist())),
                "conformal_queries": len(set(query_ids[conformal].tolist())),
            }
        )

    test_keys = [key for key, is_test in zip(keys, test) if is_test]
    learned_actions: list[str] = []
    learned_blocks: list[int] = []
    learned_errors: list[float] = []
    static_actions: list[str] = []
    static_blocks: list[int] = []
    static_errors: list[float] = []
    oracle_actions: list[str] = []
    oracle_blocks: list[int] = []
    oracle_errors: list[float] = []

    calibration = fit | conformal
    static_by_head: dict[tuple[int, int], str] = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            head_mask = calibration & np.asarray(
                [(key[1], key[2]) == (layer, head) for key in keys]
            )
            feasible = []
            for action in deployable:
                errors = np.asarray(
                    [grouped[key][action]["relative_output_l2"] for key in keys]
                )[head_mask]
                if len(errors) and float(np.quantile(errors, 0.95)) <= args.relative_error_threshold:
                    feasible.append(action)
            static_by_head[(layer, head)] = min(
                feasible,
                key=lambda action: (
                    statistics.fmean(
                        grouped[key][action]["selected_blocks"]
                        for key, keep in zip(keys, head_mask)
                        if keep
                    ),
                    action,
                ),
            ) if feasible else "full"

    test_index = 0
    output_rows: list[dict[str, Any]] = []
    for key in test_keys:
        action_rows = grouped[key]
        feasible_learned = [
            action
            for action in deployable
            if upper_bounds[action][test_index] <= args.relative_error_threshold
        ]
        learned = min(
            feasible_learned,
            key=lambda action: (action_rows[action]["selected_blocks"], upper_bounds[action][test_index], action),
        ) if feasible_learned else "full"
        feasible_oracle = [
            action
            for action in deployable
            if action_rows[action]["relative_output_l2"] <= args.relative_error_threshold
        ]
        oracle = min(
            feasible_oracle,
            key=lambda action: (
                action_rows[action]["selected_blocks"],
                action_rows[action]["relative_output_l2"],
                action,
            ),
        ) if feasible_oracle else "full"
        static = static_by_head[(key[1], key[2])]
        for action, names, blocks_list, errors_list in [
            (learned, learned_actions, learned_blocks, learned_errors),
            (static, static_actions, static_blocks, static_errors),
            (oracle, oracle_actions, oracle_blocks, oracle_errors),
        ]:
            names.append(action)
            blocks_list.append(action_rows[action]["selected_blocks"])
            errors_list.append(action_rows[action]["relative_output_l2"])
        output_rows.append(
            {
                "query_id": key[0],
                "layer": key[1],
                "query_head": key[2],
                "query_position": key[3],
                "learned_action": learned,
                "learned_blocks": action_rows[learned]["selected_blocks"],
                "learned_error": action_rows[learned]["relative_output_l2"],
                "static_action": static,
                "static_blocks": action_rows[static]["selected_blocks"],
                "static_error": action_rows[static]["relative_output_l2"],
                "oracle_action": oracle,
                "oracle_blocks": action_rows[oracle]["selected_blocks"],
                "oracle_error": action_rows[oracle]["relative_output_l2"],
            }
        )
        test_index += 1

    summaries = [
        summarize_policy(
            "learned_conformal",
            learned_actions,
            np.asarray(learned_blocks),
            np.asarray(learned_errors),
            args.relative_error_threshold,
        ),
        summarize_policy(
            "static_head_prior",
            static_actions,
            np.asarray(static_blocks),
            np.asarray(static_errors),
            args.relative_error_threshold,
        ),
        summarize_policy(
            "test_oracle",
            oracle_actions,
            np.asarray(oracle_blocks),
            np.asarray(oracle_errors),
            args.relative_error_threshold,
        ),
    ]
    for action in actions:
        action_errors = np.asarray(
            [grouped[key][action]["relative_output_l2"] for key in test_keys]
        )
        action_blocks = np.asarray(
            [grouped[key][action]["selected_blocks"] for key in test_keys]
        )
        summaries.append(
            summarize_policy(
                f"fixed_{action}",
                [action] * len(test_keys),
                action_blocks,
                action_errors,
                args.relative_error_threshold,
            )
        )
    policy_actions = {
        "learned_conformal": learned_actions,
        "static_head_prior": static_actions,
        "test_oracle": oracle_actions,
        **{f"fixed_{action}": [action] * len(test_keys) for action in actions},
    }
    for summary_row in summaries:
        summary_row.update(
            gqa_physical_summary(
                test_keys,
                grouped,
                policy_actions[summary_row["policy"]],
                args.gqa_group_size,
            )
        )
    write_csv(output_dir / "models.csv", model_rows)
    write_csv(output_dir / "test_rows.csv", output_rows)
    write_csv(output_dir / "policy_summary.csv", summaries)
    summary = {
        "source": "query-disjoint conformal head-distortion router",
        "queries": {
            "fit": sorted({key[0] for key, keep in zip(keys, fit) if keep}),
            "conformal": sorted({key[0] for key, keep in zip(keys, conformal) if keep}),
            "test": sorted({key[0] for key, keep in zip(keys, test) if keep}),
        },
        "relative_error_threshold": args.relative_error_threshold,
        "conformal_quantile": args.conformal_quantile,
        "conformal_scope": args.conformal_scope,
        "upper_bound_floor": 0.0,
        "feature_count": int(features.shape[1]),
        "gqa_group_size": args.gqa_group_size,
        "interaction_mode": args.interaction_mode,
        "model_fits": model_rows,
        "policies": summaries,
        "note": (
            "The learned router sees only head identity and QK/lexical score signatures. "
            "The test oracle is diagnostic and never used for fitting."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    deployment_bundle = {
        "source": "query-disjoint conformal head-distortion router",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "feature_count": int(features.shape[1]),
        "relative_error_threshold": args.relative_error_threshold,
        "conformal_quantile": args.conformal_quantile,
        "conformal_scope": args.conformal_scope,
        "upper_bound_floor": 0.0,
        "interaction_mode": args.interaction_mode,
        "deployable_actions": deployable,
        "fit_query_ids": sorted({key[0] for key, keep in zip(keys, fit) if keep}),
        "conformal_query_ids": sorted(
            {key[0] for key, keep in zip(keys, conformal) if keep}
        ),
        "test_query_ids": sorted({key[0] for key, keep in zip(keys, test) if keep}),
        "models": deployment_models,
    }
    (output_dir / "deployment_bundle.json").write_text(
        json.dumps(deployment_bundle), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
