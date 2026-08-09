#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def event_time_ms(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
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
    return float(start.elapsed_time(stop)) / float(iterations)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    lengths = [
        int(item.strip()) for item in args.lengths.split(",") if item.strip()
    ]
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    rows = []
    for history_count in lengths:
        torch.cuda.empty_cache()
        key = torch.randn(
            1,
            8,
            history_count,
            128,
            device=device,
            dtype=dtype,
        )
        query = torch.randn(
            1,
            32,
            128,
            device=device,
            dtype=dtype,
        )
        packed = torch.empty(
            1,
            8,
            history_count,
            8,
            device=device,
            dtype=torch.uint8,
        )

        torch.cuda.synchronize()
        build_start = torch.cuda.Event(enable_timing=True)
        build_stop = torch.cuda.Event(enable_timing=True)
        build_start.record()
        qabs_cuda_kernels.pre_rope_lowfreq_int2_fixed_pack_into(
            key,
            packed,
            0,
            10_000.0,
            1.5,
        )
        build_stop.record()
        torch.cuda.synchronize()
        build_ms = float(build_start.elapsed_time(build_stop))

        keep_count = min(
            history_count,
            1280,
            max(256, math.ceil(0.06 * history_count)),
        )

        def score() -> torch.Tensor:
            return qabs_cuda_kernels.pre_rope_lowfreq_int2_fixed_scores(
                query,
                packed,
                history_count,
                history_count,
                10_000.0,
            )

        scores = score()

        def select() -> torch.Tensor:
            return torch.topk(
                scores,
                k=keep_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices

        def score_and_select() -> torch.Tensor:
            current_scores = score()
            return torch.topk(
                current_scores,
                k=keep_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices

        score_ms = event_time_ms(
            score,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        select_ms = event_time_ms(
            select,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        total_ms = event_time_ms(
            score_and_select,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        selected = score_and_select()
        rows.append(
            {
                "history_tokens": history_count,
                "keep_tokens_per_head": keep_count,
                "selected_fraction": keep_count / float(history_count),
                "index_bits_per_token_kv_head": 64,
                "index_ratio_of_full_fp16_kv": 64.0 / 4096.0,
                "build_ms": build_ms,
                "score_ms": score_ms,
                "torch_topk_ms": select_ms,
                "score_plus_topk_ms": total_ms,
                "candidate_shape": list(selected.shape),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    payload = {
        "schema": "prerope_lowfreq_int2_benchmark_v1",
        "device": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
