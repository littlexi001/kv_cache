from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CASES = (
    ("mixed_a", 2),
    ("mixed_a", 3),
    ("mixed_b", 2),
    ("mixed_b", 3),
)
VARIANTS = (
    ("full_topk", "qkphysical"),
    ("sampled_compact", "qksampled"),
    ("fixed4421_sampled_compact", "qkfixed4421sampled"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def configured_candidate_count(payload: dict[str, Any]) -> int:
    history = int(payload["final_cache_length"]) - 1
    count = max(
        int(payload["candidate_min_tokens"]),
        math.ceil(float(payload["candidate_fraction"]) * history),
    )
    maximum = payload.get("candidate_max_tokens")
    if maximum is not None:
        count = min(count, int(maximum))
    return min(history, count)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    all_full_nll: list[float] = []
    all_logical_nll: list[float] = []
    all_physical_nll: dict[str, list[float]] = {
        variant: [] for variant, _ in VARIANTS
    }
    for topic, window in CASES:
        stem = f"{topic}_w{window}"
        reference_rows = read_csv(
            args.reference_root / stem / "token_results.csv"
        )
        full_rows = [
            row for row in reference_rows if row["method"] == "full_attention"
        ]
        logical_rows = [
            row for row in reference_rows if row["method"] == "direct_countcap"
        ]
        full_rows.sort(key=lambda row: int(row["target_index"]))
        logical_rows.sort(key=lambda row: int(row["target_index"]))
        full_ids = [int(row["token_id"]) for row in full_rows]
        logical_ids = [int(row["token_id"]) for row in logical_rows]
        if full_ids != logical_ids:
            raise RuntimeError(f"reference token mismatch for {stem}")
        full_nll = [float(row["nll"]) for row in full_rows]
        logical_nll = [float(row["nll"]) for row in logical_rows]
        all_full_nll.extend(full_nll)
        all_logical_nll.extend(logical_nll)
        case_summary = read_json(
            args.reference_root / stem / "case_summary.json"
        )
        full_summary = next(
            row for row in case_summary if row["method"] == "full_attention"
        )
        logical_summary = next(
            row for row in case_summary if row["method"] == "direct_countcap"
        )
        full_ppl = math.exp(mean(full_nll))
        logical_ppl = math.exp(mean(logical_nll))
        for variant, suffix in VARIANTS:
            physical = read_json(
                args.physical_root / f"{stem}_{suffix}.json"
            )
            physical_ids = [
                int(value) for value in physical["target_token_ids"]
            ]
            if physical_ids != full_ids:
                raise RuntimeError(
                    f"target token mismatch for {stem}/{variant}"
                )
            physical_nll = [
                float(value) for value in physical["token_nll"]
            ]
            all_physical_nll[variant].extend(physical_nll)
            physical_ppl = math.exp(mean(physical_nll))
            rows.append(
                {
                    "case": stem,
                    "variant": variant,
                    "tokens": len(physical_nll),
                    "full_ppl": full_ppl,
                    "logical_qk_ppl": logical_ppl,
                    "physical_qk_ppl": physical_ppl,
                    "physical_quality_retention_percent": (
                        100.0 * full_ppl / physical_ppl
                    ),
                    "physical_vs_logical_retention_percent": (
                        100.0 * logical_ppl / physical_ppl
                    ),
                    "full_online_seconds": float(
                        full_summary["sparse_decode_seconds"]
                    ),
                    "logical_qk_online_seconds": float(
                        logical_summary["sparse_decode_seconds"]
                    ),
                    "physical_qk_online_seconds": float(
                        physical["synchronized_model_forward_seconds"]
                    ),
                    "physical_online_speedup_vs_full": (
                        float(full_summary["sparse_decode_seconds"])
                        / float(
                            physical[
                                "synchronized_model_forward_seconds"
                            ]
                        )
                    ),
                    "logical_online_speedup_vs_full": (
                        float(full_summary["sparse_decode_seconds"])
                        / float(
                            logical_summary["sparse_decode_seconds"]
                        )
                    ),
                    "physical_gpu_kv_ratio": float(
                        physical[
                            "hierarchical_over_final_length_full_kv"
                        ]
                    ),
                    "physical_pinned_host_bytes": int(
                        physical["pinned_host_bytes"]
                    ),
                    "physical_cache_hit_rate": float(
                        physical["mean_cache_hit_rate"]
                    ),
                    "physical_prefill_seconds": float(
                        physical["prefill_seconds"]
                    ),
                    "physical_dense_query_seconds": float(
                        physical["dense_query_seconds"]
                    ),
                    "physical_conversion_seconds": float(
                        physical["cache_conversion_seconds"]
                    ),
                    "actual_attention_tokens_mean": float(
                        physical["mean_sampled_candidate_count"]
                        if physical["mean_sampled_candidate_count"]
                        is not None
                        else configured_candidate_count(physical)
                    ),
                    "candidate_overflow_rate": float(
                        physical["mean_sampled_overflow_rate"]
                        if physical["mean_sampled_overflow_rate"]
                        is not None
                        else 0.0
                    ),
                }
            )

    full_mean_nll = mean(all_full_nll)
    logical_mean_nll = mean(all_logical_nll)
    variant_summaries: dict[str, dict[str, float | int]] = {}
    for variant, _ in VARIANTS:
        variant_rows = [
            row for row in rows if row["variant"] == variant
        ]
        physical_mean_nll = mean(all_physical_nll[variant])
        variant_summaries[variant] = {
            "tokens": len(all_physical_nll[variant]),
            "aggregate_physical_qk_ppl": math.exp(physical_mean_nll),
            "aggregate_physical_quality_retention_percent": (
                100.0 * math.exp(full_mean_nll - physical_mean_nll)
            ),
            "aggregate_physical_vs_logical_retention_percent": (
                100.0 * math.exp(logical_mean_nll - physical_mean_nll)
            ),
            "aggregate_physical_online_speedup_vs_full": (
                sum(
                    float(row["full_online_seconds"])
                    for row in variant_rows
                )
                / sum(
                    float(row["physical_qk_online_seconds"])
                    for row in variant_rows
                )
            ),
            "mean_physical_gpu_kv_ratio": mean(
                [
                    float(row["physical_gpu_kv_ratio"])
                    for row in variant_rows
                ]
            ),
            "mean_physical_cache_hit_rate": mean(
                [
                    float(row["physical_cache_hit_rate"])
                    for row in variant_rows
                ]
            ),
            "mean_actual_attention_tokens": mean(
                [
                    float(row["actual_attention_tokens_mean"])
                    for row in variant_rows
                ]
            ),
            "mean_candidate_overflow_rate": mean(
                [
                    float(row["candidate_overflow_rate"])
                    for row in variant_rows
                ]
            ),
        }
    summary = {
        "cases": len(rows),
        "tokens_per_variant": len(next(iter(all_physical_nll.values()))),
        "history_tokens": 128_000,
        "aggregate_full_ppl": math.exp(full_mean_nll),
        "aggregate_logical_qk_ppl": math.exp(logical_mean_nll),
        "aggregate_logical_online_speedup_vs_full": (
            sum(
                float(row["full_online_seconds"])
                for row in rows
                if row["variant"] == VARIANTS[0][0]
            )
            / sum(
                float(row["logical_qk_online_seconds"])
                for row in rows
                if row["variant"] == VARIANTS[0][0]
            )
        ),
        "variants": variant_summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open(
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
