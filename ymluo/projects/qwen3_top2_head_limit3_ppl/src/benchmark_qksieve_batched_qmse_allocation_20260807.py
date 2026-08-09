#!/usr/bin/env python
"""Benchmark exact qMSE allocation across multiple decoder layers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import run_head_top2_targeted_ppl_20260714 as qksieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--key_samples", type=int, default=1024)
    parser.add_argument("--query_samples", type=int, default=32)
    parser.add_argument("--chunks", default="1,2,4,6,9,12,18,36")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    chunks = sorted(
        {
            int(item)
            for item in args.chunks.split(",")
            if item.strip() and 0 < int(item) <= args.layers
        }
    )
    if not chunks:
        raise ValueError("at least one valid chunk size is required")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    projected_keys = torch.randn(
        args.layers,
        args.kv_heads,
        args.key_samples,
        128,
        dtype=torch.float16,
        device=device,
    )
    projected_queries = torch.randn(
        args.layers,
        args.kv_heads,
        args.query_samples,
        128,
        dtype=torch.float16,
        device=device,
    )
    def solve(chunk_size: int) -> torch.Tensor:
        rows = []
        for start in range(0, args.layers, chunk_size):
            stop = min(args.layers, start + chunk_size)
            rows.append(
                qksieve._hierarchical_qmse_rate_allocation(
                    projected_keys[start:stop],
                    projected_queries[start:stop],
                    bit_budget_per_coordinate=15,
                    allow_zero_bits=True,
                    include_scale_metadata=True,
                    query_covariance_shrinkage="oas",
                    metric_scale_quantization=False,
                )
            )
        return torch.cat(rows, dim=0)

    outputs: dict[int, torch.Tensor] = {}
    timings: dict[int, list[float]] = {chunk: [] for chunk in chunks}
    for chunk in chunks:
        outputs[chunk] = solve(chunk)
    for repeat in range(args.repeats):
        order = chunks if repeat % 2 == 0 else list(reversed(chunks))
        for chunk in order:
            synchronize(device)
            start = time.perf_counter()
            outputs[chunk] = solve(chunk)
            synchronize(device)
            timings[chunk].append(time.perf_counter() - start)
    reference = outputs[chunks[0]]
    rows = []
    reference_median = statistics.median(timings[chunks[0]])
    for chunk in chunks:
        median = statistics.median(timings[chunk])
        rows.append(
            {
                "chunk_layers": chunk,
                "seconds": timings[chunk],
                "median_seconds": median,
                "speedup_vs_layer_serial": reference_median / median,
                "allocation_exact": bool(torch.equal(reference, outputs[chunk])),
            }
        )
    result = {
        "schema": "qksieve_batched_qmse_allocation_benchmark_v1",
        "device": str(device),
        "layers": args.layers,
        "kv_heads": args.kv_heads,
        "key_samples": args.key_samples,
        "query_samples": args.query_samples,
        "repeats": args.repeats,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
