"""Summarize paired synthetic RoPE retrieval runs produced by QKSieve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        action="append",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap input must not be empty")
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(
            values[generator.randrange(count)] for _ in range(count)
        )
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def load_rows(run_roots: list[Path]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    paths = sorted(
        path
        for run_root in run_roots
        for path in run_root.glob("*/seed*/summary.json")
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("synthetic_rope")
        if not isinstance(metadata, dict):
            continue
        length = int(payload["history_tokens"])
        seed = int(metadata["seed"])
        for row in payload["rows"]:
            if "synthetic_gold_nll" not in row:
                continue
            variant = str(row["variant"])
            record = {
                "history_tokens": length,
                "seed": seed,
                "variant": variant,
                "gold_nll": float(row["synthetic_gold_nll"]),
                "gold_ppl": float(row["synthetic_gold_ppl"]),
                "gold_probability": float(
                    row["synthetic_gold_probability"]
                ),
                "correct": int(row.get("synthetic_gold_correct", 0)),
                "actual_attention_ratio": float(
                    row["actual_attention_tokens_mean"]
                )
                / float(length),
                "packed_index_ratio_of_full_kv": float(
                    row.get("packed_index_ratio_of_full_kv", 0.0)
                ),
                "steady_seconds_per_step": float(
                    row.get("steady_sparse_seconds_per_step", 0.0)
                ),
                "fixed_overhead_seconds": float(
                    row.get("fixed_sparse_overhead_seconds", 0.0)
                ),
                "source": str(path),
            }
            key = (length, seed, variant)
            existing = rows_by_key.get(key)
            if existing is not None:
                if not math.isclose(
                    float(existing["gold_nll"]),
                    float(record["gold_nll"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                ):
                    raise RuntimeError(
                        "conflicting duplicate for "
                        f"length={length}, seed={seed}, variant={variant}"
                    )
                continue
            rows_by_key[key] = record
    return list(rows_by_key.values())


def summarize(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["history_tokens"]), str(row["variant"])),
            [],
        ).append(row)

    summaries: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    lengths = sorted({int(row["history_tokens"]) for row in rows})
    for length in lengths:
        full_rows = {
            int(row["seed"]): row
            for row in grouped.get((length, "full_attention"), [])
        }
        for (candidate_length, variant), subset in sorted(grouped.items()):
            if candidate_length != length:
                continue
            mean_nll = statistics.fmean(
                float(row["gold_nll"]) for row in subset
            )
            summaries.append(
                {
                    "history_tokens": length,
                    "variant": variant,
                    "sample_count": len(subset),
                    "mean_gold_nll": mean_nll,
                    "geometric_gold_ppl": math.exp(mean_nll),
                    "mean_gold_probability": statistics.fmean(
                        float(row["gold_probability"]) for row in subset
                    ),
                    "correct_rate": statistics.fmean(
                        int(row["correct"]) for row in subset
                    ),
                    "mean_attention_ratio": statistics.fmean(
                        float(row["actual_attention_ratio"])
                        for row in subset
                    ),
                    "mean_index_ratio_of_full_kv": statistics.fmean(
                        float(row["packed_index_ratio_of_full_kv"])
                        for row in subset
                    ),
                    "mean_steady_seconds_per_step": statistics.fmean(
                        float(row["steady_seconds_per_step"])
                        for row in subset
                    ),
                    "mean_fixed_overhead_seconds": statistics.fmean(
                        float(row["fixed_overhead_seconds"])
                        for row in subset
                    ),
                }
            )
            if variant == "full_attention":
                continue
            matched = [
                (
                    full_rows[int(row["seed"])],
                    row,
                )
                for row in subset
                if int(row["seed"]) in full_rows
            ]
            deltas = [
                float(full["gold_nll"]) - float(method["gold_nll"])
                for full, method in matched
            ]
            if not deltas:
                continue
            low, high = bootstrap_mean_interval(
                deltas,
                bootstrap_samples,
                bootstrap_seed + length + sum(map(ord, variant)),
            )
            paired.append(
                {
                    "history_tokens": length,
                    "variant": variant,
                    "paired_sample_count": len(deltas),
                    "mean_full_minus_method_nll": statistics.fmean(deltas),
                    "quality_ratio_vs_full": math.exp(
                        statistics.fmean(deltas)
                    ),
                    "delta_nll_ci_low": low,
                    "delta_nll_ci_high": high,
                    "quality_ratio_ci_low": math.exp(low),
                    "quality_ratio_ci_high": math.exp(high),
                    "ci_lower_exceeds_full": math.exp(low) > 1.0,
                    "ci_lower_exceeds_110pct": math.exp(low) > 1.1,
                    "improved_seed_fraction": statistics.fmean(
                        value > 0.0 for value in deltas
                    ),
                }
            )
    return {
        "row_count": len(rows),
        "summary": summaries,
        "paired_vs_full": paired,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_root)
    if not rows:
        raise RuntimeError(
            f"no completed synthetic rows found in {args.run_root}"
        )
    result = summarize(
        rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    result["run_roots"] = [str(path) for path in args.run_root]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "rows.csv", rows)
    write_csv(args.output_dir / "summary.csv", result["summary"])
    write_csv(
        args.output_dir / "paired_vs_full.csv",
        result["paired_vs_full"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
