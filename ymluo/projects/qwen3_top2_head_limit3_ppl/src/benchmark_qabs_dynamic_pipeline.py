from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import qabs_sampled_head_adaptive_attention
import qabs_cuda_kernels as kernels


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
    parser.add_argument("--head_count", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--qabs_dim_count", type=int, default=16)
    parser.add_argument("--candidate_fraction", type=float, default=0.07)
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--mass_threshold", type=float, default=0.75)
    parser.add_argument("--use_dim_major_index", action="store_true")
    parser.add_argument("--use_int2_index", action="store_true")
    parser.add_argument("--skip_candidate_rerank", action="store_true")
    parser.add_argument("--budget_fractions", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.04])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    scaling = args.head_dim**-0.5
    rows = []

    with torch.inference_mode():
        for history_count in args.history_lengths:
            key_count = history_count + 1
            query = torch.randn((1, args.head_count, 1, args.head_dim), device=device, dtype=dtype)
            key = torch.randn((1, args.head_count, key_count, args.head_dim), device=device, dtype=dtype)
            value = torch.randn_like(key)
            key_dim_major = None
            key_int2 = None
            if args.use_dim_major_index:
                key_dim_major = key[..., :history_count, :].transpose(2, 3).contiguous()
            if args.use_int2_index:
                if args.use_dim_major_index:
                    raise ValueError("choose only one auxiliary index")
                key_int2 = kernels.pack_int2(key[..., :history_count, :])

            def full_attention() -> torch.Tensor:
                scores = torch.matmul(query, key.transpose(2, 3)) * scaling
                weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
                return torch.matmul(weights, value)

            def dynamic_attention(diagnostics=None) -> torch.Tensor:
                return qabs_sampled_head_adaptive_attention(
                    query,
                    key,
                    value,
                    None,
                    scaling,
                    args.mass_threshold,
                    tuple(args.budget_fractions),
                    args.sample_fraction,
                    args.qabs_dim_count,
                    args.candidate_fraction,
                    use_cuda_kernels=True,
                    diagnostics=diagnostics,
                    qabs_key_dim_major=key_dim_major,
                    qabs_key_int2=key_int2,
                    skip_candidate_rerank=args.skip_candidate_rerank,
                )[0]

            diagnostics = {}
            dynamic_attention(diagnostics)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            dynamic_attention()
            torch.cuda.synchronize()
            dynamic_peak_bytes = torch.cuda.max_memory_allocated(device)
            full_ms = timed_ms(full_attention, args.warmup, args.repeats)
            dynamic_ms = timed_ms(dynamic_attention, args.warmup, args.repeats)
            selected_fraction = float(diagnostics["selected_history_fraction"].float().mean().item())
            selected_counts = diagnostics["selected_history_count"].reshape(-1)
            rung_histogram = {
                str(fraction): int(
                    (selected_counts == max(1, math.ceil(history_count * fraction))).sum().item()
                )
                for fraction in args.budget_fractions
            }
            row = {
                "history_count": history_count,
                "selected_history_fraction": selected_fraction,
                "rung_histogram": rung_histogram,
                "full_ms": full_ms,
                "dynamic_ms": dynamic_ms,
                "full_over_dynamic": full_ms / dynamic_ms,
                "dynamic_peak_gib": dynamic_peak_bytes / (1024**3),
                "index_gib": (
                    key_dim_major.numel() * key_dim_major.element_size() / (1024**3)
                    if key_dim_major is not None
                    else 0.0
                    if key_int2 is None
                    else sum(tensor.numel() * tensor.element_size() for tensor in key_int2) / (1024**3)
                ),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
            del query, key, value, key_dim_major, key_int2, diagnostics
            torch.cuda.empty_cache()

    payload = {"config": vars(args) | {"output": str(args.output)}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
