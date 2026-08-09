#!/usr/bin/env python
"""Summarize isolated, GPU-rotated QKSieve/FIER measurements."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from summarize_qksieve_fier_budget_ab_20260808 import DISPLAY_NAMES


def main() -> None:
    root = Path(sys.argv[1])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(root.glob("round*/*/summary.json")):
        payload = json.loads(summary_path.read_text())
        full = next(row for row in payload["rows"] if row["variant"] == "full_attention")
        sparse = next(row for row in payload["rows"] if row["variant"] != "full_attention")
        full_step = float(full["steady_sparse_seconds_per_step"])
        sparse_step = float(sparse["steady_sparse_seconds_per_step"])
        grouped[sparse["variant"]].append(
            {
                "full_ms": 1000.0 * full_step,
                "sparse_ms": 1000.0 * sparse_step,
                "speedup": full_step / sparse_step,
                "quality": math.exp(float(full["nll"]) - float(sparse["nll"])),
                "ppl": float(sparse["ppl"]),
                "tokens": float(sparse["actual_attention_tokens_mean"]),
                "top1": float(sparse.get("top1_agreement", 0.0)),
                "fixed_seconds": float(sparse["fixed_sparse_overhead_seconds"]),
                "index_ratio": float(sparse["packed_index_ratio_of_full_kv"]),
                "value_ratio": float(sparse["packed_value_sketch_ratio_of_full_kv"]),
            }
        )
    if set(grouped) != set(DISPLAY_NAMES):
        raise AssertionError(f"unexpected variants: {sorted(grouped)}")

    rows = []
    for variant, display_name in DISPLAY_NAMES.items():
        repeats = grouped[variant]
        rows.append(
            {
                "variant": variant,
                "display_name": display_name,
                "repeats": len(repeats),
                "quality_retention_median": median(row["quality"] for row in repeats),
                "ppl_median": median(row["ppl"] for row in repeats),
                "full_ms_per_token_median": median(row["full_ms"] for row in repeats),
                "sparse_ms_per_token_median": median(row["sparse_ms"] for row in repeats),
                "paired_speedup_median": median(row["speedup"] for row in repeats),
                "actual_tokens_per_head_median": median(row["tokens"] for row in repeats),
                "top1_agreement_median": median(row["top1"] for row in repeats),
                "fixed_overhead_seconds_median": median(
                    row["fixed_seconds"] for row in repeats
                ),
                "index_ratio_of_full_kv_median": median(
                    row["index_ratio"] for row in repeats
                ),
                "value_sketch_ratio_of_full_kv_median": median(
                    row["value_ratio"] for row in repeats
                ),
                "repeat_rows": repeats,
            }
        )
    result = {
        "schema": "qksieve_fier_budget_isolated_64k_v1",
        "history_tokens": 65536,
        "value_sketch_disabled": True,
        "gpu_rotation": True,
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Isolated 64K QKSieve/FIER budget ablation",
        "",
        "Each sparse variant runs in its own process; variants rotate over four GPUs.",
        "",
        "| Variant | Quality | Top-1 | Tokens/head | Full ms/tok | Sparse ms/tok | Paired speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} "
            f"| {100.0 * row['quality_retention_median']:.3f}% "
            f"| {100.0 * row['top1_agreement_median']:.3f}% "
            f"| {row['actual_tokens_per_head_median']:.1f} "
            f"| {row['full_ms_per_token_median']:.3f} "
            f"| {row['sparse_ms_per_token_median']:.3f} "
            f"| {row['paired_speedup_median']:.3f}x |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
