#!/usr/bin/env python3
"""Measure the exact legacy QK solver under small-matrix CPU thread counts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import _qk_metric_projection_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_tokens", type=int, default=4096)
    parser.add_argument("--query_tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def solve(
    key_covariance: torch.Tensor,
    query_covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _qk_metric_projection_factors(
        key_covariance,
        query_covariance,
        128,
        0.75,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    key = torch.randn(
        1, 8, args.sample_tokens, 128, device="cuda", dtype=torch.float32
    )
    query = torch.randn(
        1, 8, args.query_tokens, 128, device="cuda", dtype=torch.float32
    )
    key_covariance = torch.einsum("bhkd,bhke->bhde", key, key) / args.sample_tokens
    query_covariance = (
        torch.einsum("bhqd,bhqe->bhde", query, query) / args.query_tokens
    )

    original_threads = torch.get_num_threads()
    torch.set_num_threads(original_threads)
    reference = solve(key_covariance, query_covariance)
    rows = []
    for thread_count in (1, 2, 4, 8, 16, 32):
        torch.set_num_threads(thread_count)
        for _ in range(2):
            solve(key_covariance, query_covariance)
        samples = []
        candidate = None
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            candidate = solve(key_covariance, query_covariance)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        assert candidate is not None
        rows.append(
            {
                "threads": thread_count,
                "median_seconds": float(statistics.median(samples)),
                "min_seconds": float(min(samples)),
                "query_factor_bitwise_equal": bool(
                    torch.equal(reference[0], candidate[0])
                ),
                "key_factor_bitwise_equal": bool(
                    torch.equal(reference[1], candidate[1])
                ),
                "query_factor_max_abs_error": float(
                    (reference[0] - candidate[0]).abs().max().item()
                ),
                "key_factor_max_abs_error": float(
                    (reference[1] - candidate[1]).abs().max().item()
                ),
            }
        )
    baseline = next(row for row in rows if row["threads"] == 32)
    for row in rows:
        row["speedup_vs_32_threads"] = (
            baseline["median_seconds"] / row["median_seconds"]
        )
    output = {
        "schema": "qksieve_qk_cpu_thread_benchmark_v1",
        "hardware": torch.cuda.get_device_name(0),
        "cpu_count": __import__("os").cpu_count(),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
