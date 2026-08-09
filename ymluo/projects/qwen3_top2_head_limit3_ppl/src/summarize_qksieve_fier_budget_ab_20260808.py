#!/usr/bin/env python
"""Summarize repeated frozen-driver QKSieve/FIER budget ablations."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DISPLAY_NAMES = {
    "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280": (
        "QKSieve top1280, no ValueSketch"
    ),
    "qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k1280": (
        "FIER RTN-1 g32 top1280"
    ),
    "qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k512": (
        "FIER RTN-1 g32 top512"
    ),
    "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k512": (
        "QKSieve top512, no ValueSketch"
    ),
}


def main() -> None:
    root = Path(sys.argv[1])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(root.glob("r*/summary.json")):
        payload = json.loads(summary_path.read_text())
        full = next(row for row in payload["rows"] if row["variant"] == "full_attention")
        for row in payload["rows"]:
            variant = row["variant"]
            if variant == "full_attention":
                continue
            grouped[variant].append(
                {
                    "ppl": float(row["ppl"]),
                    "quality_retention": math.exp(float(full["nll"]) - float(row["nll"])),
                    "full_ms": 1000.0 * float(full["steady_sparse_seconds_per_step"]),
                    "sparse_ms": 1000.0 * float(row["steady_sparse_seconds_per_step"]),
                    "speedup": float(full["steady_sparse_seconds_per_step"])
                    / float(row["steady_sparse_seconds_per_step"]),
                    "actual_tokens": float(row["actual_attention_tokens_mean"]),
                    "fixed_overhead_seconds": float(row["fixed_sparse_overhead_seconds"]),
                    "auxiliary_ratio": float(row["packed_total_auxiliary_ratio_of_full_kv"]),
                    "value_sketch_ratio": float(row["packed_value_sketch_ratio_of_full_kv"]),
                    "top1_agreement": float(row.get("top1_agreement", 0.0)),
                    "score_mode": row["score_mode"],
                }
            )
    if set(grouped) != set(DISPLAY_NAMES):
        raise AssertionError(f"unexpected completed variants: {sorted(grouped)}")

    rows = []
    for variant, display_name in DISPLAY_NAMES.items():
        repeats = grouped[variant]
        result = {
            "variant": variant,
            "display_name": display_name,
            "repeats": len(repeats),
            "ppl_median": median(row["ppl"] for row in repeats),
            "quality_retention_median": median(
                row["quality_retention"] for row in repeats
            ),
            "full_ms_per_token_median": median(row["full_ms"] for row in repeats),
            "sparse_ms_per_token_median": median(row["sparse_ms"] for row in repeats),
            "decode_speedup_median": median(row["speedup"] for row in repeats),
            "actual_tokens_per_head_median": median(
                row["actual_tokens"] for row in repeats
            ),
            "fixed_overhead_seconds_median": median(
                row["fixed_overhead_seconds"] for row in repeats
            ),
            "auxiliary_ratio_of_full_kv_median": median(
                row["auxiliary_ratio"] for row in repeats
            ),
            "value_sketch_ratio_of_full_kv_median": median(
                row["value_sketch_ratio"] for row in repeats
            ),
            "top1_agreement_median": median(
                row["top1_agreement"] for row in repeats
            ),
            "score_mode": repeats[0]["score_mode"],
        }
        rows.append(result)

    payload = {
        "schema": "qksieve_fier_budget_ab_64k_v2",
        "history_tokens": 65536,
        "value_sketch_disabled": True,
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# 64K QKSieve/FIER budget ablation",
        "",
        "ValueSketch is disabled for all four sparse variants.",
        "",
        "| Variant | Quality | Tokens/head | Full ms/tok | Sparse ms/tok | Speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} "
            f"| {100.0 * row['quality_retention_median']:.3f}% "
            f"| {row['actual_tokens_per_head_median']:.1f} "
            f"| {row['full_ms_per_token_median']:.3f} "
            f"| {row['sparse_ms_per_token_median']:.3f} "
            f"| {row['decode_speedup_median']:.3f}x |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
