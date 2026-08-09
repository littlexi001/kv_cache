from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analyze_zero_extra_forward_utility import fdr_bh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-utility signals with the query, rather than each "
            "candidate row, as the independent statistical unit."
        )
    )
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--model_rows", required=True)
    parser.add_argument("--selection_summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--sign_flip_samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def bootstrap_mean(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return interval(values[indices].mean(axis=1))


def query_spearman(values: np.ndarray, utility: np.ndarray) -> float:
    if len(values) < 4 or np.unique(values).size < 2 or np.unique(utility).size < 2:
        return math.nan
    statistic = float(spearmanr(values, utility).statistic)
    return statistic if math.isfinite(statistic) else math.nan


def sign_flip_pvalues(
    query_correlations: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 5_000,
) -> np.ndarray:
    valid = np.isfinite(query_correlations)
    filled = np.nan_to_num(query_correlations, nan=0.0)
    counts = valid.sum(axis=0)
    observed = np.divide(
        filled.sum(axis=0), counts, out=np.zeros(filled.shape[1]), where=counts > 0
    )
    exceed = np.zeros(filled.shape[1], dtype=np.int64)
    generated = 0
    while generated < samples:
        current = min(batch_size, samples - generated)
        signs = rng.choice((-1.0, 1.0), size=(current, filled.shape[0]))
        null = signs @ filled
        null = np.divide(
            null,
            counts,
            out=np.zeros_like(null),
            where=counts[None, :] > 0,
        )
        exceed += (np.abs(null) >= np.abs(observed)[None, :]).sum(axis=0)
        generated += current
    return (exceed + 1.0) / (samples + 1.0)


def feature_statistics(
    candidate_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    sign_flip_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    candidate_lookup = {
        (int(row["query_id"]), int(row["candidate_id"])): row
        for row in candidate_rows
    }
    model_lookup = {
        (int(row["query_id"]), int(row["candidate_id"])): row
        for row in model_rows
    }
    if candidate_lookup.keys() != model_lookup.keys():
        raise ValueError("candidate and model-native rows do not align")

    feature_keys = sorted(next(iter(model_lookup.values()))["features"])
    query_ids = sorted({key[0] for key in candidate_lookup})
    per_query = np.full((len(query_ids), len(feature_keys)), np.nan, dtype=np.float64)
    for query_index, query_id in enumerate(query_ids):
        keys = sorted(key for key in candidate_lookup if key[0] == query_id)
        utility = np.asarray(
            [float(candidate_lookup[key]["delta_nll_future_b"]) for key in keys]
        )
        matrix = np.asarray(
            [
                [float(model_lookup[key]["features"][name]) for name in feature_keys]
                for key in keys
            ],
            dtype=np.float64,
        )
        for feature_index in range(len(feature_keys)):
            per_query[query_index, feature_index] = query_spearman(
                matrix[:, feature_index], utility
            )

    pvalues = sign_flip_pvalues(
        per_query, samples=sign_flip_samples, rng=rng
    )
    qvalues = fdr_bh(pvalues.tolist())
    output = []
    for feature_index, name in enumerate(feature_keys):
        values = per_query[:, feature_index]
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        indices = rng.integers(
            0, len(values), size=(bootstrap_samples, len(values))
        )
        bootstrap = values[indices].mean(axis=1)
        output.append(
            {
                "feature": name,
                "queries": int(len(values)),
                "mean_within_query_spearman": float(values.mean()),
                "median_within_query_spearman": float(np.median(values)),
                "query_bootstrap95": interval(bootstrap),
                "positive_query_fraction": float((values > 0).mean()),
                "negative_query_fraction": float((values < 0).mean()),
                "query_sign_flip_pvalue": float(pvalues[feature_index]),
                "fdr_bh_qvalue": float(qvalues[feature_index]),
            }
        )
    output.sort(
        key=lambda row: (-abs(row["mean_within_query_spearman"]), row["feature"])
    )
    return output


def selection_statistics(
    candidate_rows: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["query_id"]), int(row["candidate_id"])): float(
            row["delta_nll_future_b"]
        )
        for row in candidate_rows
    }
    query_ids = sorted({key[0] for key in lookup})
    gains: dict[str, np.ndarray] = {}
    for row in selection_summary["selection_quality"]:
        ids = row["selected_candidate_ids"]
        if len(ids) != len(query_ids):
            raise ValueError(f"selection length mismatch for {row['method']}")
        gains[row["method"]] = np.asarray(
            [lookup[(query_id, int(candidate_id))] for query_id, candidate_id in zip(query_ids, ids)],
            dtype=np.float64,
        )

    baseline_name = "oof_rank_scope"
    baseline = gains[baseline_name]
    output = []
    for method, values in gains.items():
        delta = values - baseline
        indices = rng.integers(
            0, len(values), size=(bootstrap_samples, len(values))
        )
        output.append(
            {
                "method": method,
                "queries": len(values),
                "mean_future_nll_gain": float(values.mean()),
                "gain_query_bootstrap95": interval(values[indices].mean(axis=1)),
                "positive_future_utility_rate": float((values > 0).mean()),
                "paired_baseline": baseline_name,
                "mean_gain_delta_vs_baseline": float(delta.mean()),
                "paired_delta_query_bootstrap95": interval(
                    delta[indices].mean(axis=1)
                ),
                "wins_ties_losses_vs_baseline": [
                    int((delta > 1e-12).sum()),
                    int((np.abs(delta) <= 1e-12).sum()),
                    int((delta < -1e-12).sum()),
                ],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    candidate_rows = read_jsonl(args.candidate_rows)
    model_rows = read_jsonl(args.model_rows)
    selection_summary = json.loads(
        Path(args.selection_summary).read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(args.seed)
    features = feature_statistics(
        candidate_rows,
        model_rows,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
        rng=rng,
    )
    selections = selection_statistics(
        candidate_rows,
        selection_summary,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    output = {
        "source": "LongBench-v2 code candidate utility, query-cluster robust analysis",
        "protocol": {
            "independent_unit": "query",
            "queries": len({int(row["query_id"]) for row in candidate_rows}),
            "candidate_windows": len(candidate_rows),
            "within_query_spearman_then_equal_query_average": True,
            "query_sign_flip_samples": args.sign_flip_samples,
            "query_bootstrap_samples": args.bootstrap_samples,
            "future_target_used_for_features": False,
        },
        "top_query_cluster_model_signals": features[:30],
        "fdr_significant_query_cluster_model_signals": [
            row for row in features if row["fdr_bh_qvalue"] < 0.05
        ],
        "selection_quality_paired_by_query": selections,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
