"""Direct CUDA benchmark for BinaryPC overfetch plus exact-QK reranking.

The complete-path number is measured as one CUDA call sequence.  Stage sums
are never used to derive a speedup.  Exact candidate logits are reused by the
softmax-AV consumer, avoiding a second read of candidate Keys.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import qabs_cuda_kernels as sparse_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binarypc_repo", type=Path, required=True)
    parser.add_argument("--projection_path", type=Path, required=True)
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument("--overfetch_factors", default="1,1.25,1.5,2,3")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_lengths(spec: str) -> list[int]:
    values = sorted({int(item) for item in spec.split(",") if item.strip()})
    if not values or any(value <= 0 or value % 256 for value in values):
        raise ValueError("lengths must be positive multiples of 256")
    return values


def parse_factors(spec: str) -> list[float]:
    values = sorted({float(item) for item in spec.split(",") if item.strip()})
    if not values or any(value < 1.0 for value in values):
        raise ValueError("overfetch factors must be at least one")
    return values


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def attention_from_scores(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    count = int(indices.shape[-1])
    if count >= 1280:
        return sparse_cuda.final_attention_from_scores_split(
            value, indices, scores, counts, split_count=4
        )
    if count >= 900:
        return sparse_cuda.final_attention_from_scores_split(
            value, indices, scores, counts, split_count=2
        )
    return sparse_cuda.final_attention_from_scores_ragged(
        value, indices, scores, counts
    )


def sparse_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    count = int(indices.shape[-1])
    if count >= 1280:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 4
        )
    if count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query, key, value, indices, counts, scaling, 2
        )
    return sparse_cuda.final_attention_ragged_self(
        query, key, value, indices, counts, scaling
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.binarypc_repo))
    from bpc.bpc.bpc import (  # noqa: PLC0415
        _load_cuda_kernel,
        binary_project_cuda,
        compute_errors_cuda,
        compute_hashscores_cuda,
    )

    if not _load_cuda_kernel(0):
        raise RuntimeError("official BinaryPC CUDA extension failed to load")
    projections = torch.load(
        args.projection_path, map_location="cpu", weights_only=False
    )
    if not isinstance(projections, dict) or 16 not in projections:
        raise ValueError("projection checkpoint must contain layer 16")
    projection = projections[16].to(device="cuda", dtype=torch.bfloat16)
    if tuple(projection.shape) != (8, 64, 128):
        raise ValueError("expected Llama-3.1-8B projection [8,64,128]")
    projection_t = projection.transpose(-1, -2).contiguous()

    torch.manual_seed(args.seed)
    scaling = 128.0**-0.5
    rows: list[dict[str, Any]] = []
    for history_tokens in parse_lengths(args.lengths):
        count = selected_count(history_tokens)
        error_count = max(1, int(0.1 * count))
        iterations = min(args.iterations, 20 if history_tokens >= 65536 else args.iterations)
        warmup = min(args.warmup, iterations)
        query = torch.randn(
            1, 32, 128, dtype=torch.bfloat16, device="cuda"
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        key = torch.randn(
            1,
            8,
            history_tokens,
            128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        value = torch.randn_like(key)
        hashcodes = binary_project_cuda(key, projection)
        errors = compute_errors_cuda(key, hashcodes, projection)
        error_indices = torch.topk(
            errors, k=error_count, dim=-1, sorted=False
        ).indices
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), full_key, full_value, is_causal=False
            )

        dense_iterations = min(10, iterations)
        dense_warmup = min(5, warmup)
        full_ms = measure_ms(full_attention, dense_warmup, dense_iterations)

        def query_probe() -> torch.Tensor:
            return grouped_query @ projection_t

        probe = query_probe().contiguous()

        def hash_scan() -> torch.Tensor:
            return compute_hashscores_cuda(
                probe, hashcodes, history_tokens, history_tokens
            )

        hash_scores = hash_scan()

        def rescue() -> torch.Tensor:
            current = hash_scores.clone()
            current.scatter_(2, error_indices, 10000)
            return current

        rescued = rescue()

        for factor in parse_factors(args.overfetch_factors):
            coarse_count = min(
                history_tokens, max(count, math.ceil(factor * count))
            )
            counts = torch.full(
                (1, 32), count, dtype=torch.long, device="cuda"
            )

            def coarse_topk() -> torch.Tensor:
                return torch.topk(
                    rescued, k=coarse_count, dim=-1, sorted=False
                ).indices

            coarse_kv_indices = coarse_topk()
            coarse_indices = coarse_kv_indices.repeat_interleave(
                4, dim=1
            ).contiguous()

            def exact_candidate_scores() -> torch.Tensor:
                return sparse_cuda.candidate_compact_scores(
                    query, key, coarse_indices, scaling
                )

            candidate_scores = exact_candidate_scores()

            def final_select() -> tuple[torch.Tensor, torch.Tensor]:
                selected_scores, local_indices = torch.topk(
                    candidate_scores, k=count, dim=-1, sorted=False
                )
                selected_indices = torch.gather(
                    coarse_indices, dim=-1, index=local_indices
                )
                return selected_indices.contiguous(), selected_scores.contiguous()

            final_indices, final_scores = final_select()

            def value_consumer() -> torch.Tensor:
                return attention_from_scores(
                    value, final_indices, final_scores, counts
                )

            def complete_attention() -> torch.Tensor:
                current_probe = (grouped_query @ projection_t).contiguous()
                current_hash_scores = compute_hashscores_cuda(
                    current_probe,
                    hashcodes,
                    history_tokens,
                    history_tokens,
                )
                current_hash_scores.scatter_(2, error_indices, 10000)
                current_kv = torch.topk(
                    current_hash_scores,
                    k=coarse_count,
                    dim=-1,
                    sorted=False,
                ).indices
                current_candidates = current_kv.repeat_interleave(
                    4, dim=1
                ).contiguous()
                current_exact = sparse_cuda.candidate_compact_scores(
                    query, key, current_candidates, scaling
                )
                current_scores, current_local = torch.topk(
                    current_exact, k=count, dim=-1, sorted=False
                )
                current_indices = torch.gather(
                    current_candidates, dim=-1, index=current_local
                ).contiguous()
                return attention_from_scores(
                    value,
                    current_indices,
                    current_scores.contiguous(),
                    counts,
                )

            reused_output = value_consumer()
            reference_output = sparse_attention_reference(
                query,
                key,
                value,
                final_indices,
                counts,
                scaling,
            )
            max_error = float(
                (reused_output.float() - reference_output.float())
                .abs()
                .max()
                .item()
            )
            complete_ms = measure_ms(
                complete_attention, warmup, iterations
            )
            row = {
                "history_tokens": history_tokens,
                "target_tokens_per_query_head": count,
                "target_fraction": count / history_tokens,
                "overfetch_factor": factor,
                "coarse_tokens_per_kv_head": coarse_count,
                "coarse_fraction": coarse_count / history_tokens,
                "error_rescue_tokens_per_kv_head": error_count,
                "logical_index_bits_per_token_per_kv_head": 64,
                "full_preexpanded_sdpa_direct_ms": full_ms,
                "query_probe_direct_ms": measure_ms(
                    query_probe, warmup, iterations
                ),
                "fused_hash_scan_direct_ms": measure_ms(
                    hash_scan, warmup, iterations
                ),
                "error_rescue_direct_ms": measure_ms(
                    rescue, warmup, iterations
                ),
                "coarse_topk_direct_ms": measure_ms(
                    coarse_topk, warmup, iterations
                ),
                "exact_candidate_qk_direct_ms": measure_ms(
                    exact_candidate_scores, warmup, iterations
                ),
                "final_topk_and_index_direct_ms": measure_ms(
                    final_select, warmup, iterations
                ),
                "score_reuse_softmax_av_direct_ms": measure_ms(
                    value_consumer, warmup, iterations
                ),
                "attention_complete_direct_ms": complete_ms,
                "attention_speedup_vs_full_preexpanded_sdpa": (
                    full_ms / complete_ms
                ),
                "score_reuse_max_abs_error_vs_exact_sparse": max_error,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        del query, grouped_query, key, value, hashcodes, errors
        del error_indices, full_key, full_value, probe, hash_scores, rescued
        torch.cuda.empty_cache()

    result = {
        "schema": "binarypc_exact_rerank_direct_cuda_stages_v1",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "gqa_group_size": 4,
            "head_dimension": 128,
            "dtype": "bfloat16",
            "candidate_schedule": (
                "min(N,1280,max(256,ceil(0.06*N)))"
            ),
            "coarse_selector": (
                "released BinaryPC-64 fused scan with 10% error rescue"
            ),
            "refinement": (
                "exact-QK rerank of an overfetched GQA-shared candidate set"
            ),
            "consumer": (
                "exact softmax-AV reusing candidate QK logits; no second Key read"
            ),
            "timing": (
                "CUDA events; every stage and complete path measured directly"
            ),
            "stage_sums_used_for_speedup": False,
            "historical_index_build_in_attention_speedup": False,
            "full_fallback": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
