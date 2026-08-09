#!/usr/bin/env python
"""Benchmark exact layer-parallel request-local QK factor construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _qk_metric_projection_factors_with_key_spectrum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def solve_one(
    moments: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _qk_metric_projection_factors_with_key_spectrum(
        moments[0],
        moments[1],
        projection_dim=moments[0].shape[-1],
        query_shrinkage=0.75,
    )


def timed_run(
    moments: list[tuple[torch.Tensor, torch.Tensor]],
    workers: int,
) -> tuple[float, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if moments[0][0].is_cuda:
        torch.cuda.synchronize(moments[0][0].device)
    start = time.perf_counter()
    if workers == 1:
        outputs = [solve_one(value) for value in moments]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(solve_one, moments))
    if moments[0][0].is_cuda:
        torch.cuda.synchronize(moments[0][0].device)
    return time.perf_counter() - start, outputs


def main() -> None:
    args = parse_args()
    if args.layers <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("matrix dimensions must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    moments = []
    for _ in range(args.layers):
        keys = torch.randn(
            1,
            args.heads,
            args.samples,
            args.head_dim,
            generator=generator,
            device=device,
        )
        queries = torch.randn(
            1,
            args.heads,
            args.samples,
            args.head_dim,
            generator=generator,
            device=device,
        )
        key_moment = torch.einsum("bhnd,bhne->bhde", keys, keys) / float(
            args.samples
        )
        query_moment = torch.einsum(
            "bhnd,bhne->bhde", queries, queries
        ) / float(args.samples)
        moments.append((key_moment, query_moment))

    # Warm LAPACK and establish the exact sequential reference.
    _, reference = timed_run(moments, 1)
    reference_hashes = [
        tuple(tensor_hash(tensor) for tensor in layer) for layer in reference
    ]
    rows = []
    for workers in (1, 2, 4, 8, 12):
        if workers > args.layers:
            continue
        elapsed = []
        exact = []
        for _ in range(args.repeats):
            seconds, outputs = timed_run(moments, workers)
            elapsed.append(seconds)
            hashes = [
                tuple(tensor_hash(tensor) for tensor in layer)
                for layer in outputs
            ]
            exact.append(hashes == reference_hashes)
        rows.append(
            {
                "workers": workers,
                "seconds": elapsed,
                "median_seconds": float(torch.tensor(elapsed).median()),
                "all_outputs_bitwise_equal": all(exact),
            }
        )
    baseline = rows[0]["median_seconds"]
    for row in rows:
        row["speedup_vs_sequential"] = baseline / row["median_seconds"]
    payload = {
        "layers": args.layers,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "samples": args.samples,
        "torch_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
