#!/usr/bin/env python
"""Summarize paired six-topic PPL for per-head cold skipping."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail: list[dict[str, Any]] = []
    known_modes = ("baseline", "skip50", "skip60")
    for path in sorted(args.root.glob("*/case_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        methods = {row["method"]: row for row in payload}
        if set(methods) != {"full_attention", "direct_countcap"}:
            raise ValueError(f"unexpected methods in {path}")
        full = methods["full_attention"]
        sparse = methods["direct_countcap"]
        matching_modes = [
            mode for mode in known_modes if path.parent.name.endswith(mode)
        ]
        if len(matching_modes) != 1:
            raise ValueError(f"cannot infer mode from {path.parent.name}")
        mode = matching_modes[0]
        detail.append(
            {
                "mode": mode,
                "topic": sparse["topic"],
                "tokens": int(sparse["tokens"]),
                "full_nll": float(full["nll"]),
                "sparse_nll": float(sparse["nll"]),
                "full_ppl": float(full["ppl"]),
                "sparse_ppl": float(sparse["ppl"]),
                "quality_retention": float(sparse["quality_retention"]),
                "top1_agreement": float(sparse["top1_agreement"]),
                "kl_full_to_sparse": float(
                    sparse["kl_full_to_sparse_mean"]
                ),
                "hard_skip_pool_fraction": float(
                    sparse.get(
                        "frequency_hard_skip_pool_fraction_mean", 0.0
                    )
                ),
                "hard_skip_state_count": int(
                    sparse.get("frequency_hard_skip_state_count", 0)
                ),
                "prefill_query_tokens": int(
                    sparse.get("packed_prefill_query_tokens", 0)
                ),
                "measured_quality_simulation_speedup": float(
                    full["sparse_seconds_per_step"]
                    / sparse["sparse_seconds_per_step"]
                ),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[row["mode"]].append(row)
    aggregate = []
    for mode, rows in sorted(grouped.items()):
        full_nll = float(np.mean([row["full_nll"] for row in rows]))
        sparse_nll = float(np.mean([row["sparse_nll"] for row in rows]))
        aggregate.append(
            {
                "mode": mode,
                "topics": len(rows),
                "tokens": sum(row["tokens"] for row in rows),
                "full_mean_nll": full_nll,
                "sparse_mean_nll": sparse_nll,
                "full_geometric_ppl": math.exp(full_nll),
                "sparse_geometric_ppl": math.exp(sparse_nll),
                "geometric_quality_retention": math.exp(
                    full_nll - sparse_nll
                ),
                "macro_top1_agreement": float(
                    np.mean([row["top1_agreement"] for row in rows])
                ),
                "macro_kl_full_to_sparse": float(
                    np.mean([row["kl_full_to_sparse"] for row in rows])
                ),
                "worst_topic_quality_retention": min(
                    row["quality_retention"] for row in rows
                ),
                "mean_hard_skip_pool_fraction": float(
                    np.mean(
                        [row["hard_skip_pool_fraction"] for row in rows]
                    )
                ),
                "minimum_hard_skip_state_count": min(
                    row["hard_skip_state_count"] for row in rows
                ),
                "minimum_prefill_query_tokens": min(
                    row["prefill_query_tokens"] for row in rows
                ),
                "quality_simulation_speedup": float(
                    np.mean(
                        [
                            row["measured_quality_simulation_speedup"]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    baseline_rows = [
        row for row in aggregate if row["mode"] == "baseline"
    ]
    if len(baseline_rows) != 1:
        raise ValueError("exactly one QKSieve baseline aggregate is required")
    baseline_nll = baseline_rows[0]["sparse_mean_nll"]
    for row in aggregate:
        row["quality_retention_vs_qksieve_baseline"] = math.exp(
            baseline_nll - row["sparse_mean_nll"]
        )

    full_by_topic: dict[str, list[float]] = defaultdict(list)
    for row in detail:
        full_by_topic[row["topic"]].append(row["full_nll"])
    maximum_full_repeat_error = max(
        max(values) - min(values) for values in full_by_topic.values()
    )
    output = {
        "schema": "qksieve_per_head_cold_skip_ppl_v1",
        "detail": detail,
        "aggregate": aggregate,
        "maximum_repeated_full_nll_error": maximum_full_repeat_error,
        "speed_caveat": (
            "These runs materialize full proxy scores and mask them for PPL "
            "quality validation. Their wall-clock speed is not the compact "
            "cold-skip CUDA speed; use the dedicated CUDA benchmark."
        ),
    }
    (args.root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.root / "task_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
