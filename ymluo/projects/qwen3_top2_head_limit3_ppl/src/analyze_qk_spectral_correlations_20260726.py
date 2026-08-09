#!/usr/bin/env python3
"""Summarize descriptive per-head correlations in the QK spectrum audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIRS = (
    (
        "softmax_invariant_row_mean_energy_fraction",
        "qk_rank1_energy_fraction",
        "row_mean_vs_raw_rank1",
    ),
    (
        "centered_key_query_covariance_commutator_ratio",
        "centered_qk_uncentered_key_pca_optimality_gap",
        "centered_commutator_vs_key_pca_gap",
    ),
    (
        "production_prefix_pca_subspace_overlap",
        "centered_production_prefix_pca_qk_fidelity",
        "prefix_overlap_vs_centered_fidelity",
    ),
    (
        "centered_qk_effective_rank",
        "centered_production_prefix_pca_qk_fidelity",
        "centered_effective_rank_vs_prefix_fidelity",
    ),
    (
        "qk_lambda_rank_over_next_ratio",
        "centered_production_prefix_pca_qk_fidelity",
        "rank48_gap_vs_prefix_fidelity",
    ),
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks with average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(
        np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]
    )
    return pearson, spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    output_rows: list[dict[str, object]] = []
    for model, model_rows in sorted(grouped.items()):
        for x_key, y_key, label in PAIRS:
            x = np.asarray(
                [float(row[x_key]) for row in model_rows], dtype=np.float64
            )
            y = np.asarray(
                [float(row[y_key]) for row in model_rows], dtype=np.float64
            )
            pearson, spearman = correlation(x, y)
            output_rows.append(
                {
                    "model": model,
                    "samples": len(model_rows),
                    "comparison": label,
                    "x_metric": x_key,
                    "y_metric": y_key,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"rows": output_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output_json)
    print(args.output_csv)


if __name__ == "__main__":
    main()
