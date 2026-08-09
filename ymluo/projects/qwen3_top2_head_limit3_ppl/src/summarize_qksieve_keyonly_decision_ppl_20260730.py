#!/usr/bin/env python3
"""Summarize the paired Key-PCA versus QK-balanced PPL decision run."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


METHODS = {
    "keypca_keymse": {
        "transform": "key_pca",
        "allocation": "key_mse",
        "score_mode": "pca_hierarchical_autokeytotal15z_packed_fulltopk",
    },
    "qkbalanced_keymse": {
        "transform": "qk_balanced",
        "allocation": "key_mse",
        "score_mode": (
            "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk"
        ),
    },
}


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    valid = [row for row in rows if row.get(field) is not None]
    if not valid:
        return None
    total_tokens = sum(int(row["tokens"]) for row in valid)
    return (
        sum(float(row[field]) * int(row["tokens"]) for row in valid)
        / total_tokens
    )


def _strict_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["topic"]), int(row["window"]))
        method = str(row["method"])
        if method in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate row for {key} and {method}")
        grouped[key][method] = row

    pairs = []
    for key in sorted(grouped):
        methods = grouped[key]
        if set(methods) != {"full_attention", "direct_countcap"}:
            raise ValueError(f"incomplete pair for {key}: {sorted(methods)}")
        pairs.append((methods["full_attention"], methods["direct_countcap"]))
    return pairs


def _aggregate_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    full_rows = [pair[0] for pair in pairs]
    sparse_rows = [pair[1] for pair in pairs]
    full_nll = _weighted_mean(full_rows, "nll")
    sparse_nll = _weighted_mean(sparse_rows, "nll")
    if full_nll is None or sparse_nll is None:
        raise ValueError("NLL is required for all rows")
    full_step = _weighted_mean(full_rows, "steady_sparse_seconds_per_step")
    sparse_step = _weighted_mean(
        sparse_rows, "steady_sparse_seconds_per_step"
    )
    if full_step is None or sparse_step is None:
        raise ValueError("steady decode time is required for all rows")

    return {
        "pair_count": len(pairs),
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "delta_nll": sparse_nll - full_nll,
        "quality_retention": math.exp(full_nll - sparse_nll),
        "top1_agreement": _weighted_mean(sparse_rows, "top1_agreement"),
        "kl_full_to_sparse": _weighted_mean(
            sparse_rows, "kl_full_to_sparse_mean"
        ),
        "js_divergence": _weighted_mean(
            sparse_rows, "js_divergence_mean"
        ),
        "attention_tokens_mean": _weighted_mean(
            sparse_rows, "actual_attention_tokens_mean"
        ),
        "attention_token_ratio": (
            _weighted_mean(sparse_rows, "actual_attention_tokens_mean")
            / _weighted_mean(sparse_rows, "history_tokens")
        ),
        "packed_index_ratio_of_full_kv": _weighted_mean(
            sparse_rows, "packed_index_ratio_of_full_kv"
        ),
        "steady_decode_ms_per_token": 1000.0 * sparse_step,
        "steady_decode_speedup": full_step / sparse_step,
        "fixed_index_overhead_seconds": _weighted_mean(
            sparse_rows, "fixed_sparse_overhead_seconds"
        ),
    }


def _bootstrap_difference(
    left: list[tuple[dict[str, Any], dict[str, Any]]],
    right: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    left_lookup = {
        (str(full["topic"]), int(full["window"])): (full, sparse)
        for full, sparse in left
    }
    right_lookup = {
        (str(full["topic"]), int(full["window"])): (full, sparse)
        for full, sparse in right
    }
    if set(left_lookup) != set(right_lookup):
        raise ValueError("method runs do not contain the same topic/windows")
    keys = sorted(left_lookup)

    per_pair_delta = []
    for key in keys:
        left_full, left_sparse = left_lookup[key]
        right_full, right_sparse = right_lookup[key]
        left_loss = float(left_sparse["nll"]) - float(left_full["nll"])
        right_loss = float(right_sparse["nll"]) - float(right_full["nll"])
        per_pair_delta.append(left_loss - right_loss)

    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [
            per_pair_delta[rng.randrange(len(per_pair_delta))]
            for _ in per_pair_delta
        ]
        samples.append(sum(selected) / len(selected))
    samples.sort()

    def percentile(p: float) -> float:
        index = min(len(samples) - 1, max(0, round(p * (len(samples) - 1))))
        return samples[index]

    return {
        "metric": (
            "delta_nll(Key-PCA+Key-MSE) - "
            "delta_nll(QK-balanced+Key-MSE)"
        ),
        "paired_window_count": len(keys),
        "mean": sum(per_pair_delta) / len(per_pair_delta),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "probability_keypca_lower_delta_nll": (
            sum(value < 0.0 for value in samples) / len(samples)
        ),
        "interpretation": (
            "negative favors Key-PCA; positive favors QK-balanced"
        ),
    }


def summarize(
    run_root: Path,
    *,
    expected_pairs: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "scope": (
            "Qwen3-4B, 32K history, six topics, two held-out windows/topic"
        ),
        "shared_attention_budget": (
            "min(N, 1280, max(256, ceil(0.06*N)))"
        ),
        "shared_index_budget_bits_per_token_per_kv_head": 240,
        "shared_exact_sparse_attention": True,
        "rerank": False,
        "fallback": False,
        "methods": {},
    }
    pair_sets: dict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = {}

    for tag, metadata in METHODS.items():
        source = run_root / tag / "case_summary.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        pairs = _strict_pairs(rows)
        if len(pairs) != expected_pairs:
            raise ValueError(
                f"{tag}: expected {expected_pairs} pairs, got {len(pairs)}"
            )
        pair_sets[tag] = pairs
        topics: dict[str, Any] = {}
        for topic in sorted({str(pair[0]["topic"]) for pair in pairs}):
            topic_pairs = [
                pair for pair in pairs if str(pair[0]["topic"]) == topic
            ]
            topics[topic] = _aggregate_pairs(topic_pairs)
        output["methods"][tag] = {
            **metadata,
            "aggregate": _aggregate_pairs(pairs),
            "by_topic": topics,
        }

    output["paired_comparison"] = _bootstrap_difference(
        pair_sets["keypca_keymse"],
        pair_sets["qkbalanced_keymse"],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--expected_pairs", type=int, default=12)
    parser.add_argument("--bootstrap_iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = summarize(
        args.run_root,
        expected_pairs=args.expected_pairs,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    output = args.output or args.run_root / "summary.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
