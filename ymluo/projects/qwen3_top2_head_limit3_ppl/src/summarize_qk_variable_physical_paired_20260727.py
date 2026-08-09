from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


SPARSE_METHODS = (
    ("qk_fulltopk", "qkphysical"),
    ("qk_sampled_ragged", "qksampled"),
    ("pca64_fulltopk", "pca64physical"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--topics",
        default="sports,medicine,computer,religion",
    )
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clustered_bootstrap_retention(
    paired_nll: dict[str, list[tuple[float, float]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    topics = sorted(paired_nll)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_topics = [rng.choice(topics) for _ in topics]
        full_values: list[float] = []
        sparse_values: list[float] = []
        for topic in sampled_topics:
            pairs = paired_nll[topic]
            sampled_pairs = [rng.choice(pairs) for _ in pairs]
            full_values.extend(pair[0] for pair in sampled_pairs)
            sparse_values.extend(pair[1] for pair in sampled_pairs)
        mean_delta = (
            sum(sparse_values) / len(sparse_values)
            - sum(full_values) / len(full_values)
        )
        estimates.append(100.0 * math.exp(-mean_delta))
    return {
        "retention_percent_ci95_low": percentile(estimates, 0.025),
        "retention_percent_ci95_high": percentile(estimates, 0.975),
        "probability_retention_ge_95_percent": sum(
            value >= 95.0 for value in estimates
        )
        / len(estimates),
        "probability_retention_ge_98_percent": sum(
            value >= 98.0 for value in estimates
        )
        / len(estimates),
    }


def main() -> None:
    args = parse_args()
    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    rows: list[dict[str, Any]] = []
    paired_nll_by_method: dict[
        str, dict[str, list[tuple[float, float]]]
    ] = {label: {} for label, _ in SPARSE_METHODS}
    for topic in topics:
        full = read_json(args.input_dir / f"{topic}_full.json")
        full_token_nll = [float(value) for value in full["token_nll"]]
        row: dict[str, Any] = {
            "topic": topic,
            "eval_tokens": len(full_token_nll),
            "full_ppl": float(full["ppl"]),
            "full_online_seconds": float(
                full["synchronized_model_forward_seconds"]
            ),
        }
        for label, suffix in SPARSE_METHODS:
            sparse = read_json(args.input_dir / f"{topic}_{suffix}.json")
            if full["target_token_ids"] != sparse["target_token_ids"]:
                raise RuntimeError(
                    f"target token mismatch for {topic}/{label}"
                )
            sparse_token_nll = [
                float(value) for value in sparse["token_nll"]
            ]
            if len(full_token_nll) != len(sparse_token_nll):
                raise RuntimeError(
                    f"token NLL length mismatch for {topic}/{label}"
                )
            paired_nll_by_method[label][topic] = list(
                zip(full_token_nll, sparse_token_nll)
            )
            row.update(
                {
                    f"{label}_ppl": float(sparse["ppl"]),
                    f"{label}_quality_retention_percent": (
                        100.0 * float(full["ppl"]) / float(sparse["ppl"])
                    ),
                    f"{label}_online_seconds": float(
                        sparse["synchronized_model_forward_seconds"]
                    ),
                    f"{label}_online_speedup": (
                        float(full["synchronized_model_forward_seconds"])
                        / float(sparse["synchronized_model_forward_seconds"])
                    ),
                    f"{label}_gpu_kv_ratio": float(
                        sparse["hierarchical_over_final_length_full_kv"]
                    ),
                    f"{label}_index_plus_hot_bytes": int(
                        sparse["hierarchical_persistent_gpu_bytes"]
                    ),
                    f"{label}_pinned_host_bytes": int(
                        sparse["pinned_host_bytes"]
                    ),
                    f"{label}_cache_hit_rate": float(
                        sparse["mean_cache_hit_rate"]
                    ),
                    f"{label}_prefill_seconds": float(
                        sparse["prefill_seconds"]
                    ),
                    f"{label}_dense_query_seconds": float(
                        sparse["dense_query_seconds"]
                    ),
                    f"{label}_conversion_seconds": float(
                        sparse["cache_conversion_seconds"]
                    ),
                    f"{label}_sampled_candidate_count": sparse.get(
                        "mean_sampled_candidate_count"
                    ),
                    f"{label}_sampled_overflow_rate": sparse.get(
                        "mean_sampled_overflow_rate"
                    ),
                    f"{label}_sampled_clipped_fraction": sparse.get(
                        "mean_sampled_clipped_fraction"
                    ),
                }
            )
        rows.append(row)

    first_method_pairs = paired_nll_by_method[SPARSE_METHODS[0][0]]
    all_full_nll = [
        pair[0] for pairs in first_method_pairs.values() for pair in pairs
    ]
    full_mean_nll = sum(all_full_nll) / len(all_full_nll)
    summary: dict[str, Any] = {
        "topics": topics,
        "paired_eval_tokens": len(all_full_nll),
        "aggregate_full_ppl": math.exp(full_mean_nll),
    }
    for method_index, (label, _) in enumerate(SPARSE_METHODS):
        paired_nll = paired_nll_by_method[label]
        all_sparse_nll = [
            pair[1] for pairs in paired_nll.values() for pair in pairs
        ]
        sparse_mean_nll = sum(all_sparse_nll) / len(all_sparse_nll)
        bootstrap = clustered_bootstrap_retention(
            paired_nll,
            resamples=args.bootstrap_resamples,
            seed=args.seed + method_index,
        )
        summary[label] = {
            "aggregate_ppl": math.exp(sparse_mean_nll),
            "aggregate_quality_retention_percent": (
                100.0 * math.exp(full_mean_nll - sparse_mean_nll)
            ),
            "macro_quality_retention_percent": sum(
                float(row[f"{label}_quality_retention_percent"])
                for row in rows
            )
            / len(rows),
            "aggregate_online_speedup": (
                sum(float(row["full_online_seconds"]) for row in rows)
                / sum(
                    float(row[f"{label}_online_seconds"])
                    for row in rows
                )
            ),
            "mean_gpu_kv_ratio": sum(
                float(row[f"{label}_gpu_kv_ratio"]) for row in rows
            )
            / len(rows),
            "mean_cache_hit_rate": sum(
                float(row[f"{label}_cache_hit_rate"]) for row in rows
            )
            / len(rows),
            **bootstrap,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_topic.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
