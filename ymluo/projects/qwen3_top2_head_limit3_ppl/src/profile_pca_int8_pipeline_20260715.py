from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import qabs_cuda_kernels as kernels


def timed_ms(function, warmup: int = 20, repeats: int = 200) -> float:
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
    parser.add_argument("--projection_dims", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--budget_fractions", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument(
        "--candidate_modes",
        nargs="+",
        choices=["independent", "shared_mean"],
        default=["independent"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    batch = 1
    kv_heads = 8
    groups = 4
    query_heads = kv_heads * groups
    head_dim = 128
    key_count = args.history_count + 1
    capacity = args.history_count + 2048
    scaling = head_dim**-0.5
    query = torch.randn((batch, query_heads, head_dim), device=device, dtype=dtype)
    key = torch.randn((batch, kv_heads, key_count, head_dim), device=device, dtype=dtype)
    value = torch.randn_like(key)
    rows = []

    for projection_dim in args.projection_dims:
        basis = torch.randn(
            (batch, kv_heads, head_dim, projection_dim), device=device, dtype=dtype
        )
        query_codes = torch.randint(
            -127,
            128,
            (batch, kv_heads, groups, projection_dim),
            dtype=torch.int8,
            device=device,
        )
        key_codes = torch.randint(
            -127,
            128,
            (batch, kv_heads, capacity, projection_dim),
            dtype=torch.int8,
            device=device,
        )
        key_scales = torch.rand(
            (batch, kv_heads, capacity, 1), dtype=dtype, device=device
        )
        padded_query_codes = torch.zeros(
            (batch, kv_heads, 16, projection_dim), dtype=torch.int8, device=device
        )
        padded_query_codes[..., :groups, :].copy_(query_codes)

        def prepare_query() -> torch.Tensor:
            grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
            projected_query = torch.einsum("bhgd,bhdm->bhgm", grouped_query, basis)
            query_scales = (
                projected_query.float()
                .abs()
                .amax(dim=-1, keepdim=True)
                .clamp_min(1.0e-8)
                / 127.0
            )
            current_codes = (
                torch.round(projected_query.float() / query_scales)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            padded_query_codes[..., :groups, :].copy_(current_codes)
            return padded_query_codes

        def gemm() -> torch.Tensor:
            return kernels.pca_int8_scores(query_codes, key_codes, args.history_count)

        integer_scores = gemm()

        def scaled_scan() -> torch.Tensor:
            scores = gemm().float() * key_scales[
                ..., : args.history_count, 0
            ].unsqueeze(2).float()
            return scores.reshape(batch, query_heads, args.history_count)

        def wmma_scan() -> torch.Tensor:
            return kernels.pca_int8_wmma_scores(
                padded_query_codes,
                key_codes,
                key_scales,
                args.history_count,
                groups,
            )

        scores = wmma_scan()
        for candidate_mode in args.candidate_modes:
            def select_from_scores(current_scores: torch.Tensor, keep_count: int) -> torch.Tensor:
                if candidate_mode == "independent":
                    return torch.topk(
                        current_scores, k=keep_count, dim=-1, sorted=True
                    ).indices
                grouped_scores = current_scores.reshape(
                    batch, kv_heads, groups, args.history_count
                )
                shared_indices = torch.topk(
                    grouped_scores.sum(dim=2),
                    k=keep_count,
                    dim=-1,
                    sorted=False,
                ).indices
                return (
                    shared_indices.unsqueeze(2)
                    .expand(-1, -1, groups, -1)
                    .reshape(batch, query_heads, keep_count)
                    .contiguous()
                )

            for fraction in args.budget_fractions:
                keep_count = max(1, math.ceil(fraction * args.history_count))

                def select() -> torch.Tensor:
                    return select_from_scores(scores, keep_count)

                indices = select()
                feature_candidate_scores = torch.gather(scores, dim=-1, index=indices)
                previous_probe = torch.roll(indices[..., :32], shifts=1, dims=-1)

                def retrieval_features() -> torch.Tensor:
                    return kernels.retrieval_metrics(
                        feature_candidate_scores, indices, previous_probe, 32
                    )
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
                        query, key, value, packed_indices, counts, scaling
                    )

                def composed() -> torch.Tensor:
                    current_scores = wmma_scan()
                    current_indices = select_from_scores(current_scores, keep_count)
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
                        query, key, value, current_packed, counts, scaling
                    )

                def online_composed() -> torch.Tensor:
                    prepare_query()
                    return composed()

                row = {
                    "history_count": args.history_count,
                    "projection_dim": projection_dim,
                    "candidate_mode": candidate_mode,
                    "budget_fraction": fraction,
                    "int8_gemm_ms": timed_ms(gemm),
                    "gemm_plus_scale_ms": timed_ms(scaled_scan),
                    "wmma_scan_ms": timed_ms(wmma_scan),
                    "query_projection_quantization_ms": timed_ms(prepare_query),
                    "retrieval_feature_ms": timed_ms(retrieval_features),
                    "torch_topk_ms": timed_ms(select),
                    "final_sparse_attention_ms": timed_ms(final_attention),
                    "composed_ms": timed_ms(composed),
                    "online_composed_ms": timed_ms(online_composed),
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
        del basis, query_codes, key_codes, key_scales, integer_scores, scores
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
