from __future__ import annotations

import argparse
from pathlib import Path

import torch

from run_head_top2_targeted_ppl_20260714 import qabs_sampled_head_adaptive_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_length", type=int, default=32768)
    parser.add_argument("--mass_threshold", type=float, default=0.75)
    parser.add_argument("--use_dim_major_index", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    head_count = 32
    head_dim = 128
    query = torch.randn((1, head_count, 1, head_dim), device=device, dtype=torch.float16)
    key = torch.randn((1, head_count, args.history_length + 1, head_dim), device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    key_dim_major = None
    if args.use_dim_major_index:
        key_dim_major = key[..., : args.history_length, :].transpose(2, 3).contiguous()

    def run() -> None:
        qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            None,
            head_dim**-0.5,
            args.mass_threshold,
            (0.0025, 0.005, 0.01, 0.02, 0.04),
            0.0025,
            16,
            0.07,
            use_cuda_kernels=True,
            qabs_key_dim_major=key_dim_major,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        run()
    torch.cuda.synchronize()
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    print(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
