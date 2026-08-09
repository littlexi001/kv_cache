from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_natural_operator_library import (
    cluster_bootstrap_ci,
    load_candidate,
    ridge_predict,
    route_from_predictions,
    standardize_apply,
    standardize_fit,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer-free self-verification proxies for natural KV operator routing."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as alias=mode=/path/to/answer_nll_rows.csv.",
    )
    parser.add_argument("--proxy_rows", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prior_betas", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--fallback_thresholds", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument("--regret_alphas", default="0,0.1,1,10,100")
    parser.add_argument("--regret_thresholds", default="0,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--regret_risk_zs", default="0,0.5,1")
    parser.add_argument("--tail_weight", type=float, default=0.1)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_proxy_rows(paths: Sequence[Path]) -> dict[tuple[int, str], dict[str, float]]:
    rows: dict[tuple[int, str], dict[str, float]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                key = (int(raw["query_id"]), raw["mode"])
                row = {
                    "question_nll": float(raw["question_nll"]),
                    "question_last_token_nll": float(raw["question_last_token_nll"]),
                    "answer_prefix_entropy": float(raw["answer_prefix_entropy"]),
                    "answer_prefix_top2_margin": float(raw["answer_prefix_top2_margin"]),
                    "elapsed_seconds": float(raw["elapsed_seconds"]),
                }
                if key in rows and rows[key] != row:
                    raise ValueError(f"conflicting proxy rows for {key}")
                rows[key] = row
    return rows


def within_query_zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1.0e-8] = 1.0
    return (values - mean) / scale


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    correlations = np.zeros(len(left), dtype=np.float64)
    for index in range(len(left)):
        left_rank = rankdata(left[index])
        right_rank = rankdata(right[index])
        if left_rank.std() < 1.0e-8 or right_rank.std() < 1.0e-8:
            correlations[index] = 0.0
        else:
            correlations[index] = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return correlations


PROXY_CONFIGS: dict[str, np.ndarray] = {
    "question_nll": np.asarray([1.0, 0.0, 0.0, 0.0]),
    "last_token_nll": np.asarray([0.0, 1.0, 0.0, 0.0]),
    "answer_entropy": np.asarray([0.0, 0.0, 1.0, 0.0]),
    "answer_margin": np.asarray([0.0, 0.0, 0.0, 1.0]),
    "question_plus_last": np.asarray([1.0, 1.0, 0.0, 0.0]),
    "question_plus_entropy": np.asarray([1.0, 0.0, 1.0, 0.0]),
    "question_plus_margin": np.asarray([1.0, 0.0, 0.0, 1.0]),
    "question_entropy_margin": np.asarray([1.0, 0.0, 1.0, 1.0]),
    "all_proxies": np.asarray([1.0, 1.0, 1.0, 1.0]),
}


def route_proxy(
    proxy_z: np.ndarray,
    action_prior: np.ndarray,
    weights: np.ndarray,
    prior_beta: float,
    fallback_threshold: float,
) -> np.ndarray:
    # The fourth proxy is a confidence margin, so larger is better.
    signed = proxy_z.copy()
    signed[:, :, 3] *= -1.0
    normalized_weights = weights / max(float(weights.sum()), 1.0)
    score = signed @ normalized_weights + prior_beta * action_prior[None, :]
    proposed = np.argmin(score, axis=1)
    fallback = int(np.argmin(action_prior))
    routed = np.full(len(score), fallback, dtype=np.int64)
    gain = score[:, fallback] - score[np.arange(len(score)), proposed]
    accepted = (proposed != fallback) & (gain >= fallback_threshold)
    routed[accepted] = proposed[accepted]
    return routed


def tune_proxy(
    proxy_z: np.ndarray,
    nll: np.ndarray,
    prior_betas: Sequence[float],
    thresholds: Sequence[float],
    tail_weight: float,
) -> dict[str, Any]:
    action_prior_raw = nll.mean(axis=0)
    prior_scale = max(float(action_prior_raw.std()), 1.0e-8)
    action_prior = (action_prior_raw - action_prior_raw.mean()) / prior_scale
    fallback = int(np.argmin(action_prior))
    best: dict[str, Any] | None = None
    for config_name, weights in PROXY_CONFIGS.items():
        for prior_beta in prior_betas:
            for threshold in thresholds:
                routed = route_proxy(
                    proxy_z, action_prior, weights, prior_beta, threshold
                )
                selected = nll[np.arange(len(nll)), routed]
                baseline = nll[:, fallback]
                regret = selected - baseline
                positive_tail = max(float(np.quantile(regret, 0.95)), 0.0)
                objective = float(selected.mean() + tail_weight * positive_tail)
                row = {
                    "config": config_name,
                    "prior_beta": float(prior_beta),
                    "fallback_threshold": float(threshold),
                    "objective": objective,
                    "mean_nll": float(selected.mean()),
                    "p95_regret": float(np.quantile(regret, 0.95)),
                }
                if best is None or (
                    row["objective"], row["fallback_threshold"], row["prior_beta"], row["config"]
                ) < (
                    best["objective"],
                    best["fallback_threshold"],
                    best["prior_beta"],
                    best["config"],
                ):
                    best = row
    if best is None:
        raise ValueError("empty proxy grid")
    return best


def actionwise_proxy_regret_predict(
    train_proxy_z: np.ndarray,
    train_nll: np.ndarray,
    test_proxy_z: np.ndarray,
    baseline_index: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed_train = train_proxy_z.copy()
    signed_test = test_proxy_z.copy()
    signed_train[:, :, 3] *= -1.0
    signed_test[:, :, 3] *= -1.0
    predictions = np.zeros((len(test_proxy_z), train_nll.shape[1]), dtype=np.float64)
    residual_scales = np.zeros(train_nll.shape[1], dtype=np.float64)
    for action_index in range(train_nll.shape[1]):
        if action_index == baseline_index:
            continue
        train_features = (
            signed_train[:, action_index, :] - signed_train[:, baseline_index, :]
        )
        test_features = (
            signed_test[:, action_index, :] - signed_test[:, baseline_index, :]
        )
        train_features, mean, scale = standardize_fit(train_features)
        test_features = standardize_apply(test_features, mean, scale)
        target = train_nll[:, action_index] - train_nll[:, baseline_index]
        prediction, residual = ridge_predict(
            train_features, target[:, None], test_features, alpha
        )
        predictions[:, action_index] = prediction[:, 0]
        residual_scales[action_index] = residual[0]
    return predictions, residual_scales


def inner_actionwise_proxy_route(
    proxy_z: np.ndarray,
    nll: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    threshold: float,
    risk_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected_nll = np.zeros(len(nll), dtype=np.float64)
    baseline_nll = np.zeros(len(nll), dtype=np.float64)
    for heldout in sorted(set(groups.tolist())):
        train = groups != heldout
        test = ~train
        baseline = int(np.argmin(nll[train].mean(axis=0)))
        prediction, residual = actionwise_proxy_regret_predict(
            proxy_z[train], nll[train], proxy_z[test], baseline, alpha
        )
        actions = route_from_predictions(
            prediction, residual, baseline, threshold, risk_z
        )
        test_rows = np.flatnonzero(test)
        selected_nll[test_rows] = nll[test_rows, actions]
        baseline_nll[test_rows] = nll[test_rows, baseline]
    return selected_nll, baseline_nll


def tune_actionwise_proxy_regret(
    proxy_z: np.ndarray,
    nll: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    thresholds: Sequence[float],
    risk_zs: Sequence[float],
    tail_weight: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for alpha in alphas:
        for threshold in thresholds:
            for risk_z in risk_zs:
                selected, baseline = inner_actionwise_proxy_route(
                    proxy_z, nll, groups, alpha, threshold, risk_z
                )
                regret = selected - baseline
                positive_tail = max(float(np.quantile(regret, 0.95)), 0.0)
                objective = float(selected.mean() + tail_weight * positive_tail)
                row = {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "risk_z": float(risk_z),
                    "objective": objective,
                    "mean_nll": float(selected.mean()),
                    "p95_regret": float(np.quantile(regret, 0.95)),
                }
                if best is None or (
                    row["objective"], row["threshold"], row["risk_z"], row["alpha"]
                ) < (
                    best["objective"],
                    best["threshold"],
                    best["risk_z"],
                    best["alpha"],
                ):
                    best = row
    if best is None:
        raise ValueError("empty actionwise regret grid")
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [load_candidate(spec) for spec in args.candidate]
    aliases = [alias for alias, _, _ in payloads]
    modes = [mode for _, mode, _ in payloads]
    query_sets = [set(rows) for _, _, rows in payloads]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("candidates do not cover identical queries")
    query_ids = sorted(query_sets[0])
    dataset = np.asarray([payloads[0][2][query_id]["dataset"] for query_id in query_ids])
    nll = np.asarray(
        [[rows[query_id]["answer_nll"] for _, _, rows in payloads] for query_id in query_ids],
        dtype=np.float64,
    )
    proxy_rows = load_proxy_rows([Path(path) for path in args.proxy_rows])
    proxy_names = [
        "question_nll",
        "question_last_token_nll",
        "answer_prefix_entropy",
        "answer_prefix_top2_margin",
    ]
    proxy = np.asarray(
        [
            [
                [proxy_rows[(query_id, mode)][name] for name in proxy_names]
                for mode in modes
            ]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )
    proxy_z = np.stack(
        [within_query_zscore(proxy[:, :, metric]) for metric in range(proxy.shape[2])],
        axis=2,
    )
    prior_betas = parse_float_list(args.prior_betas)
    thresholds = parse_float_list(args.fallback_thresholds)
    regret_alphas = parse_float_list(args.regret_alphas)
    regret_thresholds = parse_float_list(args.regret_thresholds)
    regret_risk_zs = parse_float_list(args.regret_risk_zs)

    routed = np.zeros(len(query_ids), dtype=np.int64)
    baseline_actions = np.zeros(len(query_ids), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    regret_routed = np.zeros(len(query_ids), dtype=np.int64)
    regret_baseline_actions = np.zeros(len(query_ids), dtype=np.int64)
    regret_fold_rows: list[dict[str, Any]] = []
    for heldout in sorted(set(dataset.tolist())):
        train = dataset != heldout
        test = ~train
        tuned = tune_proxy(
            proxy_z[train], nll[train], prior_betas, thresholds, args.tail_weight
        )
        prior_raw = nll[train].mean(axis=0)
        prior = (prior_raw - prior_raw.mean()) / max(float(prior_raw.std()), 1.0e-8)
        baseline = int(np.argmin(prior))
        actions = route_proxy(
            proxy_z[test],
            prior,
            PROXY_CONFIGS[tuned["config"]],
            tuned["prior_beta"],
            tuned["fallback_threshold"],
        )
        test_rows = np.flatnonzero(test)
        routed[test_rows] = actions
        baseline_actions[test_rows] = baseline
        selected = nll[test_rows, actions]
        baseline_nll = nll[test_rows, baseline]
        fold_rows.append(
            {
                "heldout_dataset": heldout,
                "train_queries": int(train.sum()),
                "test_queries": int(test.sum()),
                "baseline_action": aliases[baseline],
                "proxy_config": tuned["config"],
                "prior_beta": tuned["prior_beta"],
                "fallback_threshold": tuned["fallback_threshold"],
                "test_proxy_mean_nll": float(selected.mean()),
                "test_baseline_mean_nll": float(baseline_nll.mean()),
                "test_mean_delta": float((selected - baseline_nll).mean()),
                "fallback_rate": float(np.mean(actions == baseline)),
            }
        )

        regret_hyper = tune_actionwise_proxy_regret(
            proxy_z[train],
            nll[train],
            dataset[train],
            regret_alphas,
            regret_thresholds,
            regret_risk_zs,
            args.tail_weight,
        )
        regret_baseline = int(np.argmin(nll[train].mean(axis=0)))
        prediction, residual = actionwise_proxy_regret_predict(
            proxy_z[train],
            nll[train],
            proxy_z[test],
            regret_baseline,
            regret_hyper["alpha"],
        )
        regret_actions = route_from_predictions(
            prediction,
            residual,
            regret_baseline,
            regret_hyper["threshold"],
            regret_hyper["risk_z"],
        )
        regret_routed[test_rows] = regret_actions
        regret_baseline_actions[test_rows] = regret_baseline
        regret_selected = nll[test_rows, regret_actions]
        regret_baseline_nll = nll[test_rows, regret_baseline]
        regret_fold_rows.append(
            {
                "heldout_dataset": heldout,
                "train_queries": int(train.sum()),
                "test_queries": int(test.sum()),
                "baseline_action": aliases[regret_baseline],
                "alpha": regret_hyper["alpha"],
                "threshold": regret_hyper["threshold"],
                "risk_z": regret_hyper["risk_z"],
                "test_regret_router_mean_nll": float(regret_selected.mean()),
                "test_baseline_mean_nll": float(regret_baseline_nll.mean()),
                "test_mean_delta": float((regret_selected - regret_baseline_nll).mean()),
                "fallback_rate": float(np.mean(regret_actions == regret_baseline)),
            }
        )

    routed_nll = nll[np.arange(len(nll)), routed]
    baseline_nll = nll[np.arange(len(nll)), baseline_actions]
    best_global_index = int(np.argmin(nll.mean(axis=0)))
    best_global = nll[:, best_global_index]
    oracle_actions = np.argmin(nll, axis=1)
    oracle_nll = nll[np.arange(len(nll)), oracle_actions]
    regret_routed_nll = nll[np.arange(len(nll)), regret_routed]
    regret_baseline_nll = nll[np.arange(len(nll)), regret_baseline_actions]
    simple_rows: list[dict[str, Any]] = []
    signs = [1.0, 1.0, 1.0, -1.0]
    for metric_index, name in enumerate(proxy_names):
        actions = np.argmin(proxy[:, :, metric_index] * signs[metric_index], axis=1)
        selected = nll[np.arange(len(nll)), actions]
        simple_rows.append(
            {
                "proxy": name,
                "mean_answer_nll": float(selected.mean()),
                "oracle_action_accuracy": float(np.mean(actions == oracle_actions)),
                "selected_action_counts": json.dumps(
                    dict(sorted(Counter(aliases[index] for index in actions).items()))
                ),
            }
        )

    correlations: list[dict[str, Any]] = []
    for metric_index, name in enumerate(proxy_names):
        signed_proxy = proxy[:, :, metric_index] * signs[metric_index]
        correlations.append(
            {
                "proxy": name,
                "mean_querywise_spearman": float(
                    rowwise_spearman(signed_proxy, nll).mean()
                ),
                "pooled_pearson": float(
                    np.corrcoef(signed_proxy.reshape(-1), nll.reshape(-1))[0, 1]
                ),
            }
        )

    rng = np.random.default_rng(args.seed)
    delta_baseline = routed_nll - baseline_nll
    delta_global = routed_nll - best_global
    baseline_ci = cluster_bootstrap_ci(
        delta_baseline, dataset, args.bootstrap_samples, rng
    )
    global_ci = cluster_bootstrap_ci(
        delta_global, dataset, args.bootstrap_samples, rng
    )
    regret_delta_baseline = regret_routed_nll - regret_baseline_nll
    regret_delta_global = regret_routed_nll - best_global
    regret_baseline_ci = cluster_bootstrap_ci(
        regret_delta_baseline, dataset, args.bootstrap_samples, rng
    )
    regret_global_ci = cluster_bootstrap_ci(
        regret_delta_global, dataset, args.bootstrap_samples, rng
    )
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        rows.append(
            {
                "query_id": query_id,
                "dataset": dataset[index],
                "selected_action": aliases[routed[index]],
                "baseline_action": aliases[baseline_actions[index]],
                "oracle_action": aliases[oracle_actions[index]],
                "routed_nll": routed_nll[index],
                "baseline_nll": baseline_nll[index],
                "best_global_nll": best_global[index],
                "oracle_nll": oracle_nll[index],
            }
        )

    summary = {
        "source": "answer-free operator self-verification proxy audit",
        "queries": len(query_ids),
        "actions": aliases,
        "best_global_action": aliases[best_global_index],
        "best_global_mean_nll": float(best_global.mean()),
        "oracle_mean_nll": float(oracle_nll.mean()),
        "lodo_proxy_router": {
            "mean_nll": float(routed_nll.mean()),
            "fold_static_baseline_mean_nll": float(baseline_nll.mean()),
            "mean_delta_vs_fold_baseline": float(delta_baseline.mean()),
            "dataset_cluster_ci95_vs_fold_baseline": list(baseline_ci),
            "mean_delta_vs_global_best": float(delta_global.mean()),
            "dataset_cluster_ci95_vs_global_best": list(global_ci),
            "oracle_action_accuracy": float(np.mean(routed == oracle_actions)),
            "selected_action_counts": dict(
                sorted(Counter(aliases[index] for index in routed).items())
            ),
        },
        "lodo_actionwise_proxy_regret_router": {
            "mean_nll": float(regret_routed_nll.mean()),
            "fold_static_baseline_mean_nll": float(regret_baseline_nll.mean()),
            "mean_delta_vs_fold_baseline": float(regret_delta_baseline.mean()),
            "dataset_cluster_ci95_vs_fold_baseline": list(regret_baseline_ci),
            "mean_delta_vs_global_best": float(regret_delta_global.mean()),
            "dataset_cluster_ci95_vs_global_best": list(regret_global_ci),
            "oracle_action_accuracy": float(np.mean(regret_routed == oracle_actions)),
            "selected_action_counts": dict(
                sorted(Counter(aliases[index] for index in regret_routed).items())
            ),
        },
        "proxy_compute": {
            "mean_seconds_per_candidate_query": float(proxy[:, :, 0].size * 0.0)
            + float(
                np.mean(
                    [
                        proxy_rows[(query_id, mode)]["elapsed_seconds"]
                        for query_id in query_ids
                        for mode in modes
                    ]
                )
            ),
            "warning": (
                "Full-model proxy scoring over every candidate is a mechanism probe, not an "
                "acceptable final runtime. A paper system must distill or early-exit this signal."
            ),
        },
        "interpretation": (
            "A successful proxy route would show that answer-free test-time evidence can expose "
            "operator regret. Failure indicates that question likelihood/confidence is not a "
            "sufficient routing target."
        ),
    }
    write_csv(output_dir / "simple_proxy_routes.csv", simple_rows)
    write_csv(output_dir / "proxy_correlations.csv", correlations)
    write_csv(output_dir / "lodo_proxy_folds.csv", fold_rows)
    write_csv(output_dir / "lodo_actionwise_regret_folds.csv", regret_fold_rows)
    write_csv(output_dir / "lodo_proxy_rows.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
