from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import (
    _pack_nested_int4_high2,
    _pack_projected_int4_logscale16,
)


def benchmark(call: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark nested high-2-bit QK retrieval against INT4."
    )
    parser.add_argument("--tokens", type=int, default=128_000)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--baseline_candidate_fraction", type=float, default=0.06)
    parser.add_argument("--bitplane_candidate_fractions", default="0.08,0.10,0.12")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    query_heads = args.kv_heads * args.groups
    projected_key = torch.randn(
        1,
        args.kv_heads,
        args.tokens,
        args.rank,
        dtype=torch.float16,
        device=device,
    )
    packed, scales, exponents = _pack_projected_int4_logscale16(projected_key)
    packed = (
        packed.reshape(
            1,
            args.kv_heads,
            args.tokens,
            args.rank // 16,
            8,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    packed_high2 = _pack_nested_int4_high2(packed)
    query_codes = torch.randint(
        -127,
        128,
        (1, args.kv_heads, args.groups, args.rank),
        dtype=torch.int8,
        device=device,
    )
    raw_query = torch.randn(
        1, query_heads, args.head_dim, dtype=torch.float16, device=device
    )
    raw_key = torch.randn(
        1,
        args.kv_heads,
        args.tokens,
        args.head_dim,
        dtype=torch.float16,
        device=device,
    )

    def int4_scan() -> torch.Tensor:
        return qabs_cuda_kernels.pca_int4_chunked_logscale16_prefix_scores(
            query_codes,
            packed,
            scales,
            exponents,
            args.tokens,
            args.rank,
        )

    def bitplane_scan() -> torch.Tensor:
        return qabs_cuda_kernels.pca_nested_int2_logscale16_prefix_scores(
            query_codes,
            packed_high2,
            scales,
            exponents,
            args.tokens,
            args.rank,
        )

    int4_scores = int4_scan()
    bitplane_scores = bitplane_scan()
    rows: dict[str, float | int | dict[str, float]] = {
        "tokens": args.tokens,
        "rank": args.rank,
        "int4_scan_ms": benchmark(int4_scan, args.warmup, args.iterations),
        "bitplane_scan_ms": benchmark(
            bitplane_scan, args.warmup, args.iterations
        ),
    }

    fractions = [args.baseline_candidate_fraction] + [
        float(value) for value in args.bitplane_candidate_fractions.split(",")
    ]
    fraction_rows: dict[str, dict[str, float]] = {}
    for fraction in fractions:
        candidate_count = math.ceil(fraction * args.tokens)
        source = (
            int4_scores
            if math.isclose(fraction, args.baseline_candidate_fraction)
            else bitplane_scores
        )

        def select() -> torch.Tensor:
            return torch.topk(
                source, candidate_count, dim=-1, largest=True, sorted=False
            ).indices

        indices = select()

        def exact() -> torch.Tensor:
            return qabs_cuda_kernels.candidate_compact_scores(
                raw_query,
                raw_key,
                indices,
                args.head_dim**-0.5,
            )

        def pipeline() -> torch.Tensor:
            scores = int4_scan() if source is int4_scores else bitplane_scan()
            selected = torch.topk(
                scores, candidate_count, dim=-1, largest=True, sorted=False
            ).indices
            exact_scores = qabs_cuda_kernels.candidate_compact_scores(
                raw_query,
                raw_key,
                selected,
                args.head_dim**-0.5,
            )
            return torch.topk(
                exact_scores,
                math.ceil(0.02 * args.tokens),
                dim=-1,
                largest=True,
                sorted=False,
            ).indices

        fraction_rows[f"{fraction:g}"] = {
            "topk_ms": benchmark(select, args.warmup, args.iterations),
            "exact_qk_ms": benchmark(exact, args.warmup, args.iterations),
            "scan_topk_exact_finaltopk_ms": benchmark(
                pipeline, args.warmup, args.iterations
            ),
        }
    rows["fractions"] = fraction_rows
    rows["scan_speedup"] = float(rows["int4_scan_ms"]) / float(
        rows["bitplane_scan_ms"]
    )
    baseline_pipeline = fraction_rows[f"{args.baseline_candidate_fraction:g}"][
        "scan_topk_exact_finaltopk_ms"
    ]
    rows["pipeline_speedups_vs_int4_c6"] = {
        fraction: baseline_pipeline / values["scan_topk_exact_finaltopk_ms"]
        for fraction, values in fraction_rows.items()
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
