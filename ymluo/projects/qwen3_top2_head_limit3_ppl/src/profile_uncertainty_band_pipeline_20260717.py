from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    args = parser.parse_args()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    query = torch.randn(
        1, query_heads, 1, head_dim, dtype=torch.float16, device=device
    )
    key = torch.randn(
        1,
        kv_heads,
        args.history_tokens + 1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value = torch.randn_like(key)
    state: dict[str, object] = {}

    def call() -> torch.Tensor:
        output, _ = qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=head_dim**-0.5,
            mass_threshold=1.0,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=0.08,
            use_cuda_kernels=True,
            skip_candidate_rerank=False,
            score_mode="pca_int4_uncertainty_band",
            projection_dim=64,
            pca_state=state,
        )
        return output

    for _ in range(20):
        call()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        for _ in range(10):
            with record_function("uncertainty_band_iteration"):
                call()
    torch.cuda.synchronize()
    print(
        profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=40
        )
    )
    print("\n=== grouped by input shape ===")
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=60
        )
    )


if __name__ == "__main__":
    main()
