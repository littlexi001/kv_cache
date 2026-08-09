from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_natural_operator_library import cluster_bootstrap_ci, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired audit of frozen KV policies on a disjoint-query holdout."
    )
    parser.add_argument("--nll_rows", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.baseline not in methods:
        raise ValueError("baseline must be included in methods")
    nll: dict[tuple[int, str], float] = {}
    dataset: dict[int, str] = {}
    blocks: dict[str, set[int]] = {method: set() for method in methods}
    tokens: dict[str, set[int]] = {method: set() for method in methods}
    with Path(args.nll_rows).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["mode"]
            if method not in methods:
                continue
            query_id = int(row["query_id"])
            nll[(query_id, method)] = float(row["answer_nll"])
            dataset[query_id] = row["dataset"]
            blocks[method].add(int(row["context_blocks"]))
            tokens[method].add(int(row["context_tokens"]))
    query_ids = sorted(dataset)
    if any((query_id, method) not in nll for query_id in query_ids for method in methods):
        raise ValueError("methods do not cover identical query IDs")
    if len({tuple(sorted(values)) for values in blocks.values()}) != 1:
        raise ValueError(f"unequal block budgets: {blocks}")
    if len({tuple(sorted(values)) for values in tokens.values()}) != 1:
        raise ValueError(f"unequal token budgets: {tokens}")
    groups = np.asarray([dataset[query_id] for query_id in query_ids])
    baseline = np.asarray([nll[(query_id, args.baseline)] for query_id in query_ids])
    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, Any]] = []
    for method in methods:
        values = np.asarray([nll[(query_id, method)] for query_id in query_ids])
        delta = values - baseline
        query_bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
        for index in range(args.bootstrap_samples):
            sample = rng.integers(0, len(delta), len(delta))
            query_bootstrap[index] = delta[sample].mean()
        cluster_ci = cluster_bootstrap_ci(
            delta, groups, args.bootstrap_samples, rng
        )
        comparisons.append(
            {
                "method": method,
                "queries": len(query_ids),
                "mean_nll": float(values.mean()),
                "mean_delta_vs_baseline": float(delta.mean()),
                "query_ci95_low": float(np.quantile(query_bootstrap, 0.025)),
                "query_ci95_high": float(np.quantile(query_bootstrap, 0.975)),
                "dataset_cluster_ci95_low": cluster_ci[0],
                "dataset_cluster_ci95_high": cluster_ci[1],
                "win_rate_vs_baseline": float(np.mean(delta < 0.0)),
                "tie_rate_vs_baseline": float(np.mean(delta == 0.0)),
            }
        )
    per_dataset: list[dict[str, Any]] = []
    for task in sorted(set(groups.tolist())):
        ids = [query_id for query_id in query_ids if dataset[query_id] == task]
        for method in methods:
            values = [nll[(query_id, method)] for query_id in ids]
            reference = [nll[(query_id, args.baseline)] for query_id in ids]
            per_dataset.append(
                {
                    "dataset": task,
                    "method": method,
                    "queries": len(ids),
                    "mean_nll": float(np.mean(values)),
                    "mean_delta_vs_baseline": float(np.mean(np.asarray(values) - reference)),
                }
            )
    matrix = np.asarray(
        [[nll[(query_id, method)] for method in methods] for query_id in query_ids]
    )
    oracle_actions = np.argmin(matrix, axis=1)
    oracle_nll = matrix[np.arange(len(matrix)), oracle_actions]
    summary = {
        "source": "frozen-policy disjoint-query holdout audit",
        "queries": len(query_ids),
        "datasets": dict(sorted(Counter(groups.tolist()).items())),
        "physical_budget": {
            "context_blocks": sorted(next(iter(blocks.values()))),
            "context_tokens": sorted(next(iter(tokens.values()))),
        },
        "baseline": args.baseline,
        "comparisons": comparisons,
        "per_query_oracle": {
            "mean_nll": float(oracle_nll.mean()),
            "headroom_vs_baseline": float(baseline.mean() - oracle_nll.mean()),
            "action_counts": dict(
                sorted(Counter(methods[index] for index in oracle_actions).items())
            ),
        },
        "interpretation": (
            "All actions and quotas were frozen before this zero-overlap query holdout. "
            "A positive delta means regression relative to the frozen baseline."
        ),
    }
    write_csv(output_dir / "comparisons.csv", comparisons)
    write_csv(output_dir / "per_dataset.csv", per_dataset)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

