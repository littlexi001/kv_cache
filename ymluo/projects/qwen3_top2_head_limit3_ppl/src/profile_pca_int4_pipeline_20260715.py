from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels
from run_head_top2_targeted_ppl_20260714 import (
    _pca_int4_partial_scores,
    qabs_sampled_head_adaptive_attention,
)


def timed_ms(function, warmup: int = 10, repeats: int = 100) -> float:
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
    parser.add_argument("--history_count", type=int, default=131072)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--budget_fractions", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    query_heads = 32
    kv_heads = 8
    groups = query_heads // kv_heads
    head_dim = 128
    key_count = args.history_count + 1
    scaling = head_dim**-0.5
    torch.manual_seed(20260715)
    query = torch.randn((1, query_heads, 1, head_dim), device=device, dtype=dtype)
    key = torch.randn((1, kv_heads, key_count, head_dim), device=device, dtype=dtype)
    value = torch.randn_like(key)
    state: dict[str, object] = {}
    qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        None,
        scaling,
        1.0e-6,
        (args.budget_fractions[-1],),
        0.0025,
        16,
        args.budget_fractions[-1],
        use_cuda_kernels=True,
        skip_candidate_rerank=True,
        score_mode="pca_int4",
        projection_dim=args.projection_dim,
        pca_state=state,
    )
    q_raw = query[..., 0, :]
    grouped_query = q_raw.reshape(1, kv_heads, groups, head_dim)

    def project_query() -> torch.Tensor:
        projected = torch.einsum("bhgd,bhdm->bhgm", grouped_query, state["basis"])
        scale = projected.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
        return torch.round(projected.float() / scale).clamp(-127, 127).to(torch.int8)

    projected_query = project_query()

    def scan() -> torch.Tensor:
        return kernels.pca_int4_scores(
            projected_query,
            state["packed"],
            state["scales"],
            args.history_count,
        )

    scores = scan()
    rows = []
    for fraction in args.budget_fractions:
        keep_count = max(1, math.ceil(fraction * args.history_count))

        def select() -> torch.Tensor:
            return torch.topk(scores, k=keep_count, dim=-1, sorted=True).indices

        indices = select()
        packed_indices = torch.cat(
            (
                indices,
                torch.full(
                    (*indices.shape[:-1], 1),
                    key_count - 1,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=-1,
        )
        counts = torch.full(
            packed_indices.shape[:-1],
            keep_count + 1,
            dtype=torch.long,
            device=device,
        )

        def final_attention() -> torch.Tensor:
            return kernels.final_attention_ragged(
                q_raw,
                key,
                value,
                packed_indices,
                counts,
                scaling,
            )

        def composed() -> torch.Tensor:
            current_scores = scan()
            current_indices = torch.topk(
                current_scores, k=keep_count, dim=-1, sorted=True
            ).indices
            current_packed = torch.cat(
                (
                    current_indices,
                    torch.full(
                        (*current_indices.shape[:-1], 1),
                        key_count - 1,
                        dtype=torch.long,
                        device=device,
                    ),
                ),
                dim=-1,
            )
            return kernels.final_attention_ragged(
                q_raw, key, value, current_packed, counts, scaling
            )

        row = {
            "history_count": args.history_count,
            "budget_fraction": fraction,
            "keep_count": keep_count,
            "query_projection_ms": timed_ms(project_query),
            "packed_scan_ms": timed_ms(scan),
            "torch_topk_ms": timed_ms(select),
            "final_sparse_attention_ms": timed_ms(final_attention),
            "composed_ms": timed_ms(composed),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
