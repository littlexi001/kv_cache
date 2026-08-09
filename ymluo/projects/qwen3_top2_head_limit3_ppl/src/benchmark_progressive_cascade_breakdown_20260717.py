from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable

import torch

import qabs_cuda_kernels
from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=128_000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def measure_ms(
    function: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> float:
    output = function()
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = function()
    torch.cuda.synchronize()
    if output.numel() == 0:
        raise RuntimeError("benchmark produced an empty tensor")
    return (time.perf_counter() - start) * 1000.0 / iterations


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    batch_count = 1
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    projection_dim = 64
    groups = query_heads // kv_heads
    history_count = args.history_tokens
    key_count = history_count + 1
    scaling = head_dim**-0.5
    query = torch.randn(
        batch_count,
        query_heads,
        1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    key = torch.randn(
        batch_count,
        kv_heads,
        key_count,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value = torch.randn_like(key)
    q_raw = query[..., 0, :]
    state: dict[str, object] = {}
    _pca_int4_partial_scores(
        q_raw,
        key[..., :history_count, :],
        state,
        projection_dim,
        basis_descending=True,
        score_prefix_dim=16,
    )
    query_codes = state["last_projected_query_codes"].contiguous()
    query_scale = state["last_projected_query_scale"].reshape(
        batch_count, query_heads, 1
    )

    def prefix_scores() -> torch.Tensor:
        return qabs_cuda_kernels.pca_int4_prefix_scores(
            query_codes,
            state["packed"],
            state["scales"],
            history_count,
            16,
        ) * query_scale

    def full_scores() -> torch.Tensor:
        return qabs_cuda_kernels.pca_int4_scores(
            query_codes,
            state["packed"],
            state["scales"],
            history_count,
        ) * query_scale

    prefix = prefix_scores()
    full = full_scores()
    count30 = math.ceil(0.30 * history_count)
    count12 = math.ceil(0.12 * history_count)
    count8 = math.ceil(0.08 * history_count)
    count2 = math.ceil(0.02 * history_count)
    _, indices30 = torch.topk(prefix, k=count30, dim=-1, sorted=False)

    def range_scores(
        indices: torch.Tensor, start_dim: int, end_dim: int
    ) -> torch.Tensor:
        return qabs_cuda_kernels.pca_int4_candidate_range_scores(
            query_codes,
            state["packed"],
            state["scales"],
            indices,
            history_count,
            start_dim,
            end_dim,
        ) * query_scale

    scores32 = torch.gather(prefix, dim=-1, index=indices30) + range_scores(
        indices30, 16, 32
    )
    scores12, local12 = torch.topk(scores32, k=count12, dim=-1, sorted=False)
    indices12 = torch.gather(indices30, dim=-1, index=local12)
    scores64 = scores12 + range_scores(indices12, 32, 64)
    _, local8 = torch.topk(scores64, k=count8, dim=-1, sorted=False)
    cascade_indices8 = torch.gather(indices12, dim=-1, index=local8)
    _, full_indices8 = torch.topk(full, k=count8, dim=-1, sorted=False)

    def exact_candidate_scores(indices: torch.Tensor) -> torch.Tensor:
        return qabs_cuda_kernels.candidate_compact_scores(
            q_raw,
            key[..., :history_count, :],
            indices,
            scaling,
        )

    cascade_exact = exact_candidate_scores(cascade_indices8)
    selected_scores, cascade_local2 = torch.topk(
        cascade_exact, k=count2, dim=-1, sorted=True
    )
    selected = torch.gather(cascade_indices8, dim=-1, index=cascade_local2)
    packed_indices = torch.cat(
        (
            selected,
            torch.full(
                (batch_count, query_heads, 1),
                history_count,
                dtype=torch.long,
                device=device,
            ),
        ),
        dim=-1,
    )
    selected_counts = torch.full(
        (batch_count, query_heads),
        count2 + 1,
        dtype=torch.long,
        device=device,
    )
    current_key = key[..., -1, :].repeat_interleave(groups, dim=1)
    self_scores = (
        q_raw.float() * current_key.float()
    ).sum(dim=-1, keepdim=True) * scaling
    packed_scores = torch.cat((selected_scores, self_scores), dim=-1)

    timings = {
        "prefix16_scan": measure_ms(prefix_scores, args.warmup, args.iterations),
        "prefix_top30": measure_ms(
            lambda: torch.topk(prefix, k=count30, dim=-1, sorted=False).indices,
            args.warmup,
            args.iterations,
        ),
        "candidate_16_32": measure_ms(
            lambda: range_scores(indices30, 16, 32),
            args.warmup,
            args.iterations,
        ),
        "candidate_top12": measure_ms(
            lambda: torch.topk(scores32, k=count12, dim=-1, sorted=False).indices,
            args.warmup,
            args.iterations,
        ),
        "candidate_32_64": measure_ms(
            lambda: range_scores(indices12, 32, 64),
            args.warmup,
            args.iterations,
        ),
        "candidate_top8": measure_ms(
            lambda: torch.topk(scores64, k=count8, dim=-1, sorted=False).indices,
            args.warmup,
            args.iterations,
        ),
        "full64_scan": measure_ms(full_scores, args.warmup, args.iterations),
        "full_top8": measure_ms(
            lambda: torch.topk(full, k=count8, dim=-1, sorted=False).indices,
            args.warmup,
            args.iterations,
        ),
        "exact_k8": measure_ms(
            lambda: exact_candidate_scores(cascade_indices8),
            args.warmup,
            args.iterations,
        ),
        "exact_top2": measure_ms(
            lambda: torch.topk(cascade_exact, k=count2, dim=-1, sorted=True).indices,
            args.warmup,
            args.iterations,
        ),
        "final_attention2": measure_ms(
            lambda: qabs_cuda_kernels.final_attention_ragged(
                q_raw,
                key,
                value,
                packed_indices,
                selected_counts,
                scaling,
            ),
            args.warmup,
            args.iterations,
        ),
        "final_attention2_reuse_scores": measure_ms(
            lambda: qabs_cuda_kernels.final_attention_from_scores_ragged(
                value,
                packed_indices,
                packed_scores,
                selected_counts,
            ),
            args.warmup,
            args.iterations,
        ),
        "full_exact_k8": measure_ms(
            lambda: exact_candidate_scores(full_indices8),
            args.warmup,
            args.iterations,
        ),
    }
    payload = {
        "history_tokens": history_count,
        "counts": {"stage1": count30, "stage2": count12, "candidate": count8, "final": count2},
        "timings_ms": timings,
        "cascade_index_stage_sum_ms": sum(
            timings[name]
            for name in (
                "prefix16_scan",
                "prefix_top30",
                "candidate_16_32",
                "candidate_top12",
                "candidate_32_64",
                "candidate_top8",
            )
        ),
        "full64_index_stage_sum_ms": timings["full64_scan"]
        + timings["full_top8"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
