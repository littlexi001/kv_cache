from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_natural_operator_library import cluster_bootstrap_ci
from analyze_proxy_route import (
    PROXY_CONFIGS,
    actionwise_proxy_regret_predict,
    load_proxy_rows,
    route_proxy,
    tune_actionwise_proxy_regret,
    tune_proxy,
    within_query_zscore,
)
from analyze_natural_operator_library import route_from_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an answer-free risk router on calibration queries and freeze it on holdout."
    )
    parser.add_argument(
        "--action",
        action="append",
        required=True,
        help=(
            "Action as alias=calibration_mode=calibration_nll.csv="
            "target_mode=target_nll.csv."
        ),
    )
    parser.add_argument("--calibration_proxy_rows", action="append", required=True)
    parser.add_argument("--target_proxy_rows", action="append", required=True)
    parser.add_argument("--calibration_queries_jsonl", required=True)
    parser.add_argument("--target_queries_jsonl", required=True)
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


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_action(spec: str) -> tuple[str, str, Path, str, Path]:
    pieces = spec.split("=", 4)
    if len(pieces) != 5:
        raise ValueError(
            "action must be alias=calibration_mode=calibration_nll.csv="
            "target_mode=target_nll.csv"
        )
    return pieces[0], pieces[1], Path(pieces[2]), pieces[3], Path(pieces[4])


def read_nll(path: Path, mode: str) -> tuple[dict[int, float], dict[int, str]]:
    values: dict[int, float] = {}
    datasets: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != mode:
                continue
            query_id = int(row["query_id"])
            values[query_id] = float(row["answer_nll"])
            datasets[query_id] = row["dataset"]
    if not values:
        raise ValueError(f"no NLL rows for {mode} in {path}")
    return values, datasets


def read_uids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {str(json.loads(line)["record_uid"]) for line in handle if line.strip()}


def build_proxy_tensor(
    query_ids: list[int],
    modes: list[str],
    rows: dict[tuple[int, str], dict[str, float]],
) -> np.ndarray:
    names = [
        "question_nll",
        "question_last_token_nll",
        "answer_prefix_entropy",
        "answer_prefix_top2_margin",
    ]
    raw = np.asarray(
        [
            [[rows[(query_id, mode)][name] for name in names] for mode in modes]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )
    return np.stack(
        [within_query_zscore(raw[:, :, index]) for index in range(raw.shape[2])],
        axis=2,
    )


def paired_summary(
    selected: np.ndarray,
    baseline: np.ndarray,
    dataset: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    delta = selected - baseline
    query_bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(delta), len(delta))
        query_bootstrap[index] = delta[sample].mean()
    cluster_ci = cluster_bootstrap_ci(delta, dataset, bootstrap_samples, rng)
    return {
        "mean_nll": float(selected.mean()),
        "baseline_mean_nll": float(baseline.mean()),
        "mean_delta_vs_baseline": float(delta.mean()),
        "query_ci95": [
            float(np.quantile(query_bootstrap, 0.025)),
            float(np.quantile(query_bootstrap, 0.975)),
        ],
        "dataset_cluster_ci95": list(cluster_ci),
        "win_rate": float(np.mean(delta < 0.0)),
        "tie_rate": float(np.mean(delta == 0.0)),
        "p95_regret": float(np.quantile(delta, 0.95)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_specs = [parse_action(spec) for spec in args.action]
    aliases = [item[0] for item in action_specs]
    calibration_modes = [item[1] for item in action_specs]
    target_modes = [item[3] for item in action_specs]
    calibration_payloads = [read_nll(item[2], item[1]) for item in action_specs]
    target_payloads = [read_nll(item[4], item[3]) for item in action_specs]
    calibration_sets = [set(values) for values, _ in calibration_payloads]
    target_sets = [set(values) for values, _ in target_payloads]
    if any(query_set != calibration_sets[0] for query_set in calibration_sets[1:]):
        raise ValueError("calibration actions do not cover identical queries")
    if any(query_set != target_sets[0] for query_set in target_sets[1:]):
        raise ValueError("target actions do not cover identical queries")
    if read_uids(Path(args.calibration_queries_jsonl)) & read_uids(Path(args.target_queries_jsonl)):
        raise ValueError("calibration and target query record_uids overlap")
    calibration_ids = sorted(calibration_sets[0])
    target_ids = sorted(target_sets[0])
    calibration_nll = np.asarray(
        [[values[query_id] for values, _ in calibration_payloads] for query_id in calibration_ids]
    )
    target_nll = np.asarray(
        [[values[query_id] for values, _ in target_payloads] for query_id in target_ids]
    )
    calibration_dataset = np.asarray(
        [calibration_payloads[0][1][query_id] for query_id in calibration_ids]
    )
    target_dataset = np.asarray(
        [target_payloads[0][1][query_id] for query_id in target_ids]
    )
    calibration_proxy = build_proxy_tensor(
        calibration_ids,
        calibration_modes,
        load_proxy_rows([Path(path) for path in args.calibration_proxy_rows]),
    )
    target_proxy = build_proxy_tensor(
        target_ids,
        target_modes,
        load_proxy_rows([Path(path) for path in args.target_proxy_rows]),
    )
    baseline_index = int(np.argmin(calibration_nll.mean(axis=0)))
    baseline_target = target_nll[:, baseline_index]

    heuristic = tune_proxy(
        calibration_proxy,
        calibration_nll,
        parse_floats(args.prior_betas),
        parse_floats(args.fallback_thresholds),
        args.tail_weight,
    )
    prior_raw = calibration_nll.mean(axis=0)
    prior = (prior_raw - prior_raw.mean()) / max(float(prior_raw.std()), 1.0e-8)
    heuristic_actions = route_proxy(
        target_proxy,
        prior,
        PROXY_CONFIGS[heuristic["config"]],
        heuristic["prior_beta"],
        heuristic["fallback_threshold"],
    )
    heuristic_nll = target_nll[np.arange(len(target_nll)), heuristic_actions]

    regret_hyper = tune_actionwise_proxy_regret(
        calibration_proxy,
        calibration_nll,
        calibration_dataset,
        parse_floats(args.regret_alphas),
        parse_floats(args.regret_thresholds),
        parse_floats(args.regret_risk_zs),
        args.tail_weight,
    )
    prediction, residual = actionwise_proxy_regret_predict(
        calibration_proxy,
        calibration_nll,
        target_proxy,
        baseline_index,
        regret_hyper["alpha"],
    )
    regret_actions = route_from_predictions(
        prediction,
        residual,
        baseline_index,
        regret_hyper["threshold"],
        regret_hyper["risk_z"],
    )
    regret_nll = target_nll[np.arange(len(target_nll)), regret_actions]
    rng = np.random.default_rng(args.seed)
    oracle = target_nll.min(axis=1)
    summary = {
        "source": "frozen answer-free router on zero-overlap query holdout",
        "calibration_queries": len(calibration_ids),
        "target_queries": len(target_ids),
        "query_uid_overlap": 0,
        "actions": aliases,
        "calibration_baseline": aliases[baseline_index],
        "target_baseline_mean_nll": float(baseline_target.mean()),
        "target_oracle_mean_nll": float(oracle.mean()),
        "target_oracle_headroom": float(baseline_target.mean() - oracle.mean()),
        "heuristic": {
            "hyperparameters": heuristic,
            "selected_action_counts": dict(
                sorted(Counter(aliases[index] for index in heuristic_actions).items())
            ),
            **paired_summary(
                heuristic_nll,
                baseline_target,
                target_dataset,
                args.bootstrap_samples,
                rng,
            ),
        },
        "actionwise_regret": {
            "hyperparameters": regret_hyper,
            "selected_action_counts": dict(
                sorted(Counter(aliases[index] for index in regret_actions).items())
            ),
            **paired_summary(
                regret_nll,
                baseline_target,
                target_dataset,
                args.bootstrap_samples,
                rng,
            ),
        },
        "warning": (
            "Proxy scoring currently runs the full model once per candidate and is a "
            "mechanism probe, not a deployable runtime."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

