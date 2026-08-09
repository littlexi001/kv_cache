from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark residual-sentinel retrieval against matched PCA and full SDPA."
    )
    parser.add_argument("--history_tokens", type=int, default=128_000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--query_heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--attention_fraction", type=float, default=0.02)
    parser.add_argument("--candidate_fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    output = function()
    for _ in range(warmup):
        output = function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        output = function()
    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000.0 / iterations
    return elapsed, output


def persistent_index_bytes(state: dict[str, object]) -> int:
    names = {
        "basis",
        "packed",
        "packed_chunked",
        "scales",
        "error_radius_codes",
        "error_radius_scale",
        "indexed_key_norm_max",
    }
    return sum(
        value.numel() * value.element_size()
        for name, value in state.items()
        if name in names and isinstance(value, torch.Tensor)
    )


def main() -> None:
    args = parse_args()
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16
    scaling = args.head_dim**-0.5
    key_count = args.history_tokens + 1
    query = torch.randn(
        1, args.query_heads, 1, args.head_dim, dtype=dtype, device=device
    )
    key = torch.randn(
        1, args.kv_heads, key_count, args.head_dim, dtype=dtype, device=device
    )
    value = torch.randn_like(key)
    group_count = args.query_heads // args.kv_heads
    expanded_key = key.repeat_interleave(group_count, dim=1)
    expanded_value = value.repeat_interleave(group_count, dim=1)

    states: dict[str, dict[str, object]] = {
        "pca64_top2": {},
        "pca64_overfetch3": {},
        "pca64_overfetch8": {},
        "progressive_cascade8": {},
        "two_stage16_8": {},
        "two_stage32_8": {},
        "two_stage48_8": {},
        "residual_sentinel3": {},
    }

    def sparse_call(
        state: dict[str, object],
        score_mode: str,
        candidate_fraction: float,
        skip_rerank: bool,
    ) -> torch.Tensor:
        output, _ = qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=1.0e-6,
            budget_fractions=(args.attention_fraction,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=candidate_fraction,
            use_cuda_kernels=True,
            skip_candidate_rerank=skip_rerank,
            score_mode=score_mode,
            projection_dim=args.projection_dim,
            pca_state=state,
        )
        return output

    methods: dict[str, Callable[[], torch.Tensor]] = {
        "full_sdpa": lambda: F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            is_causal=False,
            scale=scaling,
        ),
        "pca64_top2": lambda: sparse_call(
            states["pca64_top2"], "pca_int4", args.attention_fraction, True
        ),
        "pca64_overfetch3": lambda: sparse_call(
            states["pca64_overfetch3"],
            "pca_int4",
            args.candidate_fraction,
            False,
        ),
        "pca64_overfetch8": lambda: sparse_call(
            states["pca64_overfetch8"],
            "pca_int4",
            0.08,
            False,
        ),
        "progressive_cascade8": lambda: sparse_call(
            states["progressive_cascade8"],
            "pca_int4_progressive_cascade",
            0.08,
            False,
        ),
        "two_stage16_8": lambda: sparse_call(
            states["two_stage16_8"], "pca_int4_two_stage16", 0.08, False
        ),
        "two_stage32_8": lambda: sparse_call(
            states["two_stage32_8"], "pca_int4_two_stage32", 0.08, False
        ),
        "two_stage48_8": lambda: sparse_call(
            states["two_stage48_8"], "pca_int4_two_stage48", 0.08, False
        ),
        "residual_sentinel3": lambda: sparse_call(
            states["residual_sentinel3"],
            "pca_int4_residual_sentinel",
            args.candidate_fraction,
            False,
        ),
    }

    build_ms: dict[str, float] = {}
    for name in (
        "pca64_top2",
        "pca64_overfetch3",
        "pca64_overfetch8",
        "progressive_cascade8",
        "two_stage16_8",
        "two_stage32_8",
        "two_stage48_8",
        "residual_sentinel3",
    ):
        synchronize(device)
        start = time.perf_counter()
        methods[name]()
        synchronize(device)
        build_ms[name] = (time.perf_counter() - start) * 1000.0

    timings: dict[str, float] = {}
    outputs: dict[str, torch.Tensor] = {}
    for name, function in methods.items():
        timings[name], outputs[name] = measure_ms(
            function, device, args.warmup, args.iterations
        )

    full_ms = timings["full_sdpa"]
    full_kv_bytes = key.numel() * key.element_size() * 2
    results = []
    for name in methods:
        state = states.get(name)
        index_bytes = persistent_index_bytes(state) if state is not None else 0
        results.append(
            {
                "method": name,
                "steady_ms": timings[name],
                "speedup_vs_full_sdpa": full_ms / timings[name],
                "index_build_ms": build_ms.get(name, 0.0),
                "persistent_index_bytes": index_bytes,
                "persistent_index_ratio_vs_compact_fp16_kv": index_bytes
                / full_kv_bytes,
                "output_norm": float(outputs[name].float().norm()),
            }
        )

    payload = {
        "history_tokens": args.history_tokens,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "attention_fraction": args.attention_fraction,
        "candidate_fraction": args.candidate_fraction,
        "iterations": args.iterations,
        "results": results,
        "theoretical_full_precision_read_ratio": {
            "pca64_top2": args.attention_fraction,
            "pca64_overfetch3": (
                args.candidate_fraction + args.attention_fraction
            )
            / 2.0,
            "pca64_overfetch8": (0.08 + args.attention_fraction) / 2.0,
            "progressive_cascade8": (0.08 + args.attention_fraction) / 2.0,
            "two_stage16_8": (0.08 + args.attention_fraction) / 2.0,
            "two_stage32_8": (0.08 + args.attention_fraction) / 2.0,
            "two_stage48_8": (0.08 + args.attention_fraction) / 2.0,
            "residual_sentinel3": (
                args.candidate_fraction + args.attention_fraction
            )
            / 2.0,
        },
        "full_precision_read_breakdown": {
            "pca64_top2": {
                "candidate_k": 0.0,
                "attention_k": args.attention_fraction,
                "attention_v": args.attention_fraction,
            },
            "pca64_overfetch3": {
                "candidate_k": args.candidate_fraction,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "pca64_overfetch8": {
                "candidate_k": 0.08,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "progressive_cascade8": {
                "candidate_k": 0.08,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "two_stage16_8": {
                "candidate_k": 0.08,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "two_stage32_8": {
                "candidate_k": 0.08,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "two_stage48_8": {
                "candidate_k": 0.08,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
            "residual_sentinel3": {
                "candidate_k": args.candidate_fraction,
                "attention_k": 0.0,
                "attention_v": args.attention_fraction,
            },
        },
        "theoretical_attention_compute_speedup": 1.0
        / args.attention_fraction,
        "progressive_cascade_normalized_index_work": 0.385,
        "candidate_count": math.ceil(args.candidate_fraction * args.history_tokens),
        "selected_count": math.ceil(args.attention_fraction * args.history_tokens),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
