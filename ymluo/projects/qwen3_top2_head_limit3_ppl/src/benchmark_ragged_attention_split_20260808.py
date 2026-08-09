#!/usr/bin/env python
"""Measure ragged exact-attention split choices at fixed candidate budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=65536)
    parser.add_argument("--budgets", default="512,1280")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260808)
    device = torch.device("cuda")
    query = torch.randn(1, 32, 128, dtype=torch.float16, device=device)
    key = torch.randn(
        1, 8, args.history_tokens + 1, 128, dtype=torch.float16, device=device
    )
    value = torch.randn_like(key)
    rows = []
    for budget in [int(item) for item in args.budgets.split(",") if item]:
        indices = torch.randint(
            0,
            args.history_tokens,
            (1, 32, budget),
            dtype=torch.long,
            device=device,
        )
        counts = torch.full((1, 32), budget, dtype=torch.long, device=device)

        def unsplit():
            return kernels.final_attention_ragged_self(
                query, key, value, indices, counts, 128.0**-0.5
            )

        timings = {
            "split1": measure(unsplit, args.warmup, args.iterations)
        }
        for split in (2, 4, 8, 16):
            def split_attention(split: int = split):
                return kernels.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    indices,
                    counts,
                    128.0**-0.5,
                    split,
                )

            timings[f"split{split}"] = measure(
                split_attention, args.warmup, args.iterations
            )
        best = min(timings, key=timings.get)
        row = {
            "history_tokens": args.history_tokens,
            "budget": budget,
            "timings_ms": timings,
            "best": best,
            "best_ms": timings[best],
            "speedup_vs_split1": timings["split1"] / timings[best],
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "schema": "ragged_attention_split_benchmark_v1",
        "device": torch.cuda.get_device_name(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
