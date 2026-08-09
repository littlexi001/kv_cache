#!/usr/bin/env python
"""Aggregate paired long-context QKSieve PPL windows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(int(row["tokens"]) for row in rows)
    return sum(float(row[field]) * int(row["tokens"]) for row in rows) / total


def main() -> None:
    args = parse_args()
    cases: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.input_root.glob("seed*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (path.parent / "ALL_COMPLETE").exists():
            continue
        cases.append(
            {
                "case": path.parent.name,
                "history_tokens": int(payload["history_tokens"]),
                "eval_tokens": int(payload["eval_tokens"]),
                "shared_prefill_seconds": float(
                    payload["shared_prefill_seconds"]
                ),
            }
        )
        for row in payload["rows"]:
            grouped[str(row["variant"])].append(row)

    if not cases:
        raise RuntimeError(f"no completed cases under {args.input_root}")

    full_nll = weighted_mean(grouped["full_attention"], "nll")
    methods: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        nll = weighted_mean(rows, "nll")
        item: dict[str, Any] = {
            "variant": variant,
            "windows": len(rows),
            "tokens": sum(int(row["tokens"]) for row in rows),
            "nll": nll,
            "ppl": math.exp(nll),
            "quality_retention_vs_full": math.exp(full_nll - nll),
            "steady_ms_per_token": 1000.0
            * weighted_mean(rows, "steady_sparse_seconds_per_step"),
            "peak_gpu_allocated_gib_max": max(
                float(row["peak_gpu_allocated_bytes_max_device"])
                / (1024.0**3)
                for row in rows
            ),
        }
        if variant != "full_attention":
            item.update(
                {
                    "top1_agreement": weighted_mean(
                        rows, "top1_agreement"
                    ),
                    "kl_full_to_sparse_mean": weighted_mean(
                        rows, "kl_full_to_sparse_mean"
                    ),
                    "actual_attention_tokens_mean": weighted_mean(
                        rows, "actual_attention_tokens_mean"
                    ),
                    "packed_index_ratio_of_full_kv": weighted_mean(
                        rows, "packed_index_ratio_of_full_kv"
                    ),
                    "packed_quantile_sample_count": weighted_mean(
                        rows, "packed_quantile_sample_count"
                    ),
                }
            )
        methods.append(item)

    full_ms = next(
        row["steady_ms_per_token"]
        for row in methods
        if row["variant"] == "full_attention"
    )
    for row in methods:
        row["decode_speedup_vs_full"] = full_ms / row["steady_ms_per_token"]

    result = {
        "schema": "qksieve_longcontext_quality_aggregate_v1",
        "input_root": str(args.input_root),
        "cases": cases,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
