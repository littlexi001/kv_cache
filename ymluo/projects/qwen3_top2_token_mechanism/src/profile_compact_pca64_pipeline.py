from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--score_mode",
        default="pca_int4_chunked_logscale16",
        choices=(
            "pca_int4_chunked_logscale16",
            "pca_int4_chunked_logscale16_qkmetric_autosplit",
            "pca_int4_qkmetric_microblock8_o24_autosplit",
            "pca_int4_qkmetric_microblock8_o32_autosplit",
            "pca_int4_chunked_logscale16_split8",
        ),
    )
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260720)
    history = args.history_tokens
    query = torch.randn((1, 32, 1, 128), device="cuda", dtype=torch.float16)
    key = torch.randn((1, 8, history + 1, 128), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    state: dict[str, object] = {"rope_theta": 5_000_000.0}

    def step() -> torch.Tensor:
        return qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=128**-0.5,
            mass_threshold=0.75,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=args.candidate_fraction,
            use_cuda_kernels=True,
            skip_candidate_rerank=False,
            score_mode=args.score_mode,
            projection_dim=args.projection_dim,
            pca_state=state,
        )[0]

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
    torch.cuda.synchronize()
    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=40
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
