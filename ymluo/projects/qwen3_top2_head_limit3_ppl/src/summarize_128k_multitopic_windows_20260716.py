from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_cases", type=int, default=18)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(
    values: list[float], samples: int, rng: random.Random
) -> dict[str, float]:
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for name in glob.glob(str(args.input_dir / "*_w*_*.json")):
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        method = "full" if name.endswith("_full.json") else "sparse"
        key = (str(data["topic"]), int(data["window_index"]))
        pairs.setdefault(key, {})[method] = data
    complete = {key: value for key, value in pairs.items() if set(value) == {"full", "sparse"}}
    if len(complete) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} complete pairs, found {len(complete)}"
        )

    rows = []
    position_rows = []
    for (topic, window), pair in sorted(complete.items()):
        full = pair["full"]
        sparse = pair["sparse"]
        if full["target_token_ids"] != sparse["target_token_ids"]:
            raise ValueError(f"target mismatch for {topic} window {window}")
        full_token_nll = list(map(float, full["token_nll"]))
        sparse_token_nll = list(map(float, sparse["token_nll"]))
        if not (
            len(full_token_nll)
            == len(sparse_token_nll)
            == len(full["target_token_ids"])
        ):
            raise ValueError(f"token NLL length mismatch for {topic} window {window}")
        for target_index, (full_nll, sparse_nll) in enumerate(
            zip(full_token_nll, sparse_token_nll)
        ):
            gap = sparse_nll - full_nll
            position_rows.append(
                {
                    "topic": topic,
                    "window": window,
                    "target_index": target_index,
                    "token_id": int(full["target_token_ids"][target_index]),
                    "token_text": str(full["target_token_texts"][target_index]),
                    "full_nll": full_nll,
                    "sparse_nll": sparse_nll,
                    "delta_nll": gap,
                    "token_quality_retention": math.exp(-gap),
                }
            )
        quality_retention = math.exp(-(float(sparse["nll"]) - float(full["nll"])))
        full_total = float(full["prefill_seconds"]) + float(
            full["synchronized_model_forward_seconds"]
        )
        sparse_total = (
            float(sparse["prefill_seconds"])
            + float(sparse["cache_conversion_seconds"])
            + float(sparse["online_seconds"])
        )
        rows.append(
            {
                "topic": topic,
                "window": window,
                "target_tokens": len(full["target_token_ids"]),
                "full_ppl": float(full["ppl"]),
                "sparse_ppl": float(sparse["ppl"]),
                "delta_nll": float(sparse["nll"]) - float(full["nll"]),
                "quality_retention": quality_retention,
                "logical_attention_token_ratio": float(sparse["attention_fraction"]),
                "logical_candidate_token_ratio": float(sparse["candidate_fraction"]),
                "kv_ratio": float(sparse["hierarchical_over_final_length_full_kv"]),
                "persistent_gpu_tensor_bytes_ratio": float(
                    sparse["hierarchical_over_final_length_full_kv"]
                ),
                "pinned_host_over_original_full_kv": float(sparse["pinned_host_bytes"])
                / float(sparse["original_remote_full_gpu_kv_bytes"]),
                "cache_hit_rate": float(sparse["mean_cache_hit_rate"]),
                "decode_speedup": float(full["synchronized_model_forward_seconds"])
                / float(sparse["online_seconds"]),
                "protocol_speedup": full_total / sparse_total,
                "full_peak_gpu_bytes": int(
                    full["process_peak_gpu_allocated_during_prefill_decode"]
                ),
                "sparse_peak_gpu_bytes": int(
                    sparse["process_peak_gpu_allocated_during_prefill_conversion"]
                ),
                "peak_gpu_bytes_ratio": int(
                    sparse["process_peak_gpu_allocated_during_prefill_conversion"]
                )
                / int(full["process_peak_gpu_allocated_during_prefill_decode"]),
            }
        )

    positions: dict[int, list[float]] = {}
    for row in position_rows:
        positions.setdefault(int(row["target_index"]), []).append(
            float(row["delta_nll"])
        )
    position_summary = []
    for target_index, gaps in sorted(positions.items()):
        mean_gap = sum(gaps) / len(gaps)
        position_summary.append(
            {
                "target_index": target_index,
                "cases": len(gaps),
                "mean_delta_nll": mean_gap,
                "median_delta_nll": percentile(gaps, 0.5),
                "p05_delta_nll": percentile(gaps, 0.05),
                "p95_delta_nll": percentile(gaps, 0.95),
                "min_delta_nll": min(gaps),
                "max_delta_nll": max(gaps),
                "mean_position_quality_retention": math.exp(-mean_gap),
            }
        )

    rng = random.Random(args.seed)
    metrics = {}
    for key in (
        "delta_nll",
        "quality_retention",
        "logical_attention_token_ratio",
        "persistent_gpu_tensor_bytes_ratio",
        "kv_ratio",
        "decode_speedup",
        "protocol_speedup",
        "peak_gpu_bytes_ratio",
    ):
        metrics[key] = bootstrap_ci(
            [float(row[key]) for row in rows], args.bootstrap_samples, rng
        )
    metrics["worst_quality_retention"] = min(
        float(row["quality_retention"]) for row in rows
    )
    metrics["worst_case"] = min(rows, key=lambda row: row["quality_retention"])
    metrics["cases"] = len(rows)
    metrics["target_tokens_per_case"] = sorted({row["target_tokens"] for row in rows})
    metrics["worst_mean_position"] = max(
        position_summary, key=lambda row: row["mean_delta_nll"]
    )
    metrics["worst_individual_position"] = max(
        position_rows, key=lambda row: row["delta_nll"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_cases.csv", rows)
    write_csv(args.output_dir / "position_nll_gaps.csv", position_rows)
    write_csv(args.output_dir / "position_summary.csv", position_summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
