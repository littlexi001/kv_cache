from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def timed_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--query_heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--budget_fraction", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    scaling = args.head_dim**-0.5
    rows = []

    with torch.inference_mode():
        for history_count in args.history_lengths:
            key_count = history_count + 1
            query = torch.randn(
                (1, args.query_heads, 1, args.head_dim), device=device, dtype=dtype
            )
            key = torch.randn(
                (1, args.kv_heads, key_count, args.head_dim), device=device, dtype=dtype
            )
            value = torch.randn_like(key)
            groups = args.query_heads // args.kv_heads
            expanded_key = key.repeat_interleave(groups, dim=1)
            expanded_value = value.repeat_interleave(groups, dim=1)

            def full_attention() -> torch.Tensor:
                scores = torch.matmul(query, expanded_key.transpose(2, 3)) * scaling
                weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
                return torch.matmul(weights, expanded_value)

            state: dict[str, object] = {}

            def pca_attention(active_state=state) -> torch.Tensor:
                return qabs_sampled_head_adaptive_attention(
                    query,
                    key,
                    value,
                    None,
                    scaling,
                    1.0e-6,
                    (args.budget_fraction,),
                    0.0025,
                    16,
                    args.budget_fraction,
                    use_cuda_kernels=True,
                    skip_candidate_rerank=True,
                    score_mode="pca_int4",
                    projection_dim=args.projection_dim,
                    pca_state=active_state,
                )[0]

            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            init_start.record()
            pca_attention()
            init_end.record()
            torch.cuda.synchronize()
            initialization_ms = float(init_start.elapsed_time(init_end))
            full_ms = timed_ms(full_attention, args.warmup, args.repeats)
            pca_ms = timed_ms(pca_attention, args.warmup, args.repeats)
            index_bytes = (
                state["packed"].numel() * state["packed"].element_size()
                + state["scales"].numel() * state["scales"].element_size()
                + state["basis"].numel() * state["basis"].element_size()
            )
            full_kv_bytes = (
                key.numel() * key.element_size() + value.numel() * value.element_size()
            )
            row = {
                "history_count": history_count,
                "budget_fraction": args.budget_fraction,
                "projection_dim": args.projection_dim,
                "initialization_ms": initialization_ms,
                "full_ms": full_ms,
                "pca_int4_ms": pca_ms,
                "full_over_pca_int4": full_ms / pca_ms,
                "index_mib": index_bytes / (1024**2),
                "index_fraction_of_full_kv": index_bytes / full_kv_bytes,
                "logical_index_plus_selected_kv_fraction": (
                    index_bytes / full_kv_bytes + args.budget_fraction
                ),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
            del query, key, value, expanded_key, expanded_value, state
            torch.cuda.empty_cache()

    payload = {"config": vars(args) | {"output": str(args.output)}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
