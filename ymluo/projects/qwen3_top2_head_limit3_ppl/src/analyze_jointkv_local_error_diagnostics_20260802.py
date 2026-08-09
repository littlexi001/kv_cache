#!/usr/bin/env python
"""Summarize local attention errors and their relation to token-level NLL drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    records = row["budget_records"]
    if not records:
        raise ValueError(f"{path} does not contain raw budget records")
    keys = (
        "local_attention_relative_l2",
        "selected_attention_mass",
        "attention_entropy",
        "maximum_attention_probability",
        "selected_remote_fraction",
    )
    arrays = {
        key: np.asarray([float(record[key]) for record in records], dtype=np.float64)
        for key in keys
    }
    local_error = arrays["local_attention_relative_l2"]
    correlations = {}
    for key, values in arrays.items():
        if key == "local_attention_relative_l2":
            continue
        correlations[key] = {
            "pearson": pearson(local_error, values),
            "spearman": pearson(rank(local_error), rank(values)),
        }

    layers = sorted({int(record["layer"]) for record in records})
    per_layer = {}
    for layer in layers:
        mask = np.asarray([int(record["layer"]) == layer for record in records])
        per_layer[str(layer)] = {
            "local_error": distribution(local_error[mask]),
            "selected_mass_mean": float(arrays["selected_attention_mass"][mask].mean()),
            "selected_fraction_mean": float(
                arrays["selected_remote_fraction"][mask].mean()
            ),
            "entropy_mean": float(arrays["attention_entropy"][mask].mean()),
        }

    histories = sorted({int(record["history_tokens"]) for record in records})
    full_losses = np.asarray(row["full_token_nll"], dtype=np.float64)
    sparse_losses = np.asarray(row["sparse_token_nll"], dtype=np.float64)
    if len(histories) != full_losses.size:
        raise ValueError("decode histories do not align with token losses")
    step_mean_error = []
    step_p90_error = []
    step_max_error = []
    for history in histories:
        mask = np.asarray(
            [int(record["history_tokens"]) == history for record in records]
        )
        step_mean_error.append(float(local_error[mask].mean()))
        step_p90_error.append(float(np.quantile(local_error[mask], 0.90)))
        step_max_error.append(float(local_error[mask].max()))
    nll_delta = sparse_losses - full_losses
    step_correlations = {
        "mean_local_error": pearson(np.asarray(step_mean_error), nll_delta),
        "p90_local_error": pearson(np.asarray(step_p90_error), nll_delta),
        "max_local_error": pearson(np.asarray(step_max_error), nll_delta),
    }

    worst_layers = sorted(
        per_layer.items(),
        key=lambda item: item[1]["local_error"]["mean"],
        reverse=True,
    )[:8]
    return {
        "name": path.stem,
        "aggregate": payload["aggregate"],
        "record_count": len(records),
        "local_error": distribution(local_error),
        "selected_mass": distribution(arrays["selected_attention_mass"]),
        "attention_entropy": distribution(arrays["attention_entropy"]),
        "correlations_with_local_error": correlations,
        "token_nll_delta": distribution(nll_delta),
        "token_nll_correlations": step_correlations,
        "worst_layers": [
            {"layer": int(layer), **statistics} for layer, statistics in worst_layers
        ],
        "per_layer": per_layer,
    }


def main() -> None:
    args = parse_args()
    output = {path.stem: summarize(path) for path in args.inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, summary in output.items():
        aggregate = summary["aggregate"]
        print(
            name,
            f"quality={100 * aggregate['quality_ratio']:.3f}%",
            f"KL={aggregate['full_to_sparse_kl']:.6f}",
            f"local={summary['local_error']['mean']:.4f}",
            f"mass={summary['selected_mass']['mean']:.4f}",
            f"corr_nll_p90={summary['token_nll_correlations']['p90_local_error']:.3f}",
        )


if __name__ == "__main__":
    main()
