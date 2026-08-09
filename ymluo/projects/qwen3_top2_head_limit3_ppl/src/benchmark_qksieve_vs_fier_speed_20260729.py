#!/usr/bin/env python
"""Fair fused-retrieval speed comparison between QKSieve and FIER RTN1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import benchmark_qksieve_mixedblock_cuda_20260729 as mixed_bench
import fier_rtn1_cuda_20260728 as fier_cuda
import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,131072")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--hot_fraction", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def threshold_fraction(fraction: float, sample_count: int) -> float:
    rank = max(
        1,
        min(sample_count, int(round(fraction * (sample_count + 1)))),
    )
    return (rank - 0.5) / sample_count


def sample_count_for(fraction: float) -> int:
    return min(2048, max(256, math.ceil(16.0 / fraction)))


def capacity_for(history_count: int, fraction: float) -> int:
    samples = sample_count_for(fraction)
    capacity_fraction = min(
        1.0,
        max(
            0.06,
            fraction
            + 6.0
            * math.sqrt(fraction * (1.0 - fraction) / samples),
        ),
    )
    return min(
        history_count,
        max(1, math.ceil(capacity_fraction * history_count)),
    )


def reference_sampled_thresholds(
    scores: torch.Tensor,
    sample_count: int,
    selected_fraction: float,
) -> torch.Tensor:
    history_count = int(scores.shape[-1])
    rank = max(
        1,
        min(
            sample_count,
            math.ceil(selected_fraction * sample_count),
        ),
    )
    flat = scores.reshape(-1, history_count)
    sample = torch.arange(
        sample_count, dtype=torch.long, device=scores.device
    )
    centered = ((2 * sample + 1) * history_count) // (2 * sample_count)
    segment = max(1, history_count // sample_count)
    rows = torch.arange(
        flat.shape[0], dtype=torch.long, device=scores.device
    )
    phase = (rows * 131 + 17) % segment
    indices = (
        centered.unsqueeze(0) + phase.unsqueeze(1)
    ) % history_count
    sampled = flat.gather(1, indices)
    return torch.topk(
        sampled, k=rank, dim=-1, sorted=False
    ).values.amin(dim=-1).reshape(scores.shape[:-1])


def allocate_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        ),
        torch.empty(1, 32, dtype=torch.long, device="cuda"),
        torch.empty(1, 32, dtype=torch.float32, device="cuda"),
        torch.empty(1, 32, dtype=torch.bool, device="cuda"),
    )


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    indices, counts, _, _ = outputs
    split_count = 8 if indices.shape[-1] <= 4096 else 4
    return sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        indices,
        counts,
        128.0**-0.5,
        split_count,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = []
    lengths = sorted(
        {int(item) for item in args.lengths.split(",") if item}
    )
    for history_count in lengths:
        q_fraction = mixed_bench.selected_fraction(history_count)
        fier_equal_active_fraction = q_fraction
        fier_equal_quality_fraction = min(1.0, 2.0 * q_fraction)
        q_samples = sample_count_for(q_fraction)
        fier_active_samples = sample_count_for(fier_equal_active_fraction)
        fier_quality_samples = sample_count_for(fier_equal_quality_fraction)
        q_threshold_fraction = threshold_fraction(q_fraction, q_samples)
        fier_active_threshold_fraction = threshold_fraction(
            fier_equal_active_fraction, fier_active_samples
        )
        fier_quality_threshold_fraction = threshold_fraction(
            fier_equal_quality_fraction, fier_quality_samples
        )

        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        query_basis = torch.randn(
            1, 8, 128, 128, dtype=torch.float16, device="cuda"
        )
        projected_query = torch.einsum(
            "bhgd,bhdm->bhgm", grouped_query, query_basis
        )
        query_codes, query_scales = (
            varbit_cuda.quantize_projected_query(projected_query)
        )
        key = torch.randn(
            1,
            8,
            history_count + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        q_meta, q_code_count, q_scale_count = mixed_bench.mixed_metadata(
            history_count,
            args.block_size,
            args.hot_fraction,
            mixed_bench.LOW_PROFILES["fixed84"],
        )
        q_codes = torch.randint(
            0,
            256,
            (q_code_count,),
            dtype=torch.uint8,
            device="cuda",
        )
        q_scales = torch.rand(
            q_scale_count, dtype=torch.float16, device="cuda"
        )
        q_outputs = allocate_outputs(capacity_for(history_count, q_fraction))

        fier_index = fier_cuda.allocate_packed_index(
            1, 8, history_count, key.device
        )
        fier_cuda.update_packed_index(
            key[..., :history_count, :],
            fier_index,
            history_count,
        )
        fier_active_outputs = allocate_outputs(
            capacity_for(history_count, fier_equal_active_fraction)
        )
        fier_quality_outputs = allocate_outputs(
            capacity_for(history_count, fier_equal_quality_fraction)
        )

        def q_retrieve(
            codes: torch.Tensor = query_codes,
            scales: torch.Tensor = query_scales,
        ) -> tuple[torch.Tensor, ...]:
            return mixed_cuda.sampled_threshold_compact_gqa4_indices_out(
                codes,
                scales,
                q_codes,
                q_scales,
                q_meta,
                *q_outputs,
                history_count,
                q_samples,
                q_threshold_fraction,
            )

        def q_retrieve_with_prepare() -> tuple[torch.Tensor, ...]:
            codes, scales = query_cuda.project_quantize(
                grouped_query, query_basis
            )
            return q_retrieve(codes, scales)

        def q_complete() -> torch.Tensor:
            q_retrieve_with_prepare()
            return sparse_attention(query, key, value, q_outputs)

        def fier_active_retrieve() -> tuple[torch.Tensor, ...]:
            return fier_cuda.sampled_threshold_compact_out(
                query,
                fier_index,
                *fier_active_outputs,
                history_count,
                fier_active_samples,
                fier_active_threshold_fraction,
            )

        def fier_quality_retrieve() -> tuple[torch.Tensor, ...]:
            return fier_cuda.sampled_threshold_compact_out(
                query,
                fier_index,
                *fier_quality_outputs,
                history_count,
                fier_quality_samples,
                fier_quality_threshold_fraction,
            )

        def fier_active_complete() -> torch.Tensor:
            fier_active_retrieve()
            return sparse_attention(
                query, key, value, fier_active_outputs
            )

        def fier_quality_complete() -> torch.Tensor:
            fier_quality_retrieve()
            return sparse_attention(
                query, key, value, fier_quality_outputs
            )

        def fier_scores() -> torch.Tensor:
            return fier_cuda.scores(query, fier_index, history_count)

        materialized_fier_scores = fier_scores()
        fier_target_count = min(
            history_count,
            max(1, math.ceil(fier_equal_active_fraction * history_count)),
        )

        def fier_topk() -> torch.Tensor:
            return torch.topk(
                materialized_fier_scores,
                k=fier_target_count,
                dim=-1,
                sorted=False,
            ).indices

        def fier_current_selection() -> torch.Tensor:
            return torch.topk(
                fier_scores(),
                k=fier_target_count,
                dim=-1,
                sorted=False,
            ).indices

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), full_key, full_value
            )

        q_retrieve()
        fier_active_retrieve()
        fier_quality_retrieve()
        iterations = min(
            args.iterations,
            20 if history_count >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        q_retrieval_ms = measure_ms(
            q_retrieve_with_prepare, warmup, iterations
        )
        q_complete_ms = measure_ms(q_complete, warmup, iterations)
        fier_active_retrieval_ms = measure_ms(
            fier_active_retrieve, warmup, iterations
        )
        fier_active_complete_ms = measure_ms(
            fier_active_complete, warmup, iterations
        )
        fier_quality_retrieval_ms = measure_ms(
            fier_quality_retrieve, warmup, iterations
        )
        fier_quality_complete_ms = measure_ms(
            fier_quality_complete, warmup, iterations
        )
        fier_score_ms = measure_ms(fier_scores, warmup, iterations)
        fier_topk_ms = measure_ms(fier_topk, warmup, iterations)
        fier_current_selection_ms = measure_ms(
            fier_current_selection, warmup, iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(5, warmup),
            min(10, iterations),
        )

        q_counts = q_outputs[1].float()
        fier_active_counts = fier_active_outputs[1].float()
        fier_quality_counts = fier_quality_outputs[1].float()
        reference_threshold = reference_sampled_thresholds(
            materialized_fier_scores,
            fier_active_samples,
            fier_active_threshold_fraction,
        )
        expected_mask = (
            materialized_fier_scores >= reference_threshold.unsqueeze(-1)
        )
        expected_counts = expected_mask.sum(dim=-1)
        candidate_set_recalls = []
        candidate_set_precisions = []
        flat_indices = fier_active_outputs[0].reshape(32, -1)
        flat_counts = fier_active_outputs[1].reshape(-1)
        flat_expected = expected_mask.reshape(32, history_count)
        for head in range(32):
            actual_count = int(flat_counts[head].item())
            actual_set = set(
                flat_indices[
                    head, :actual_count
                ].detach().cpu().tolist()
            )
            expected_set = set(
                torch.nonzero(
                    flat_expected[head], as_tuple=False
                ).flatten().detach().cpu().tolist()
            )
            intersection = len(actual_set & expected_set)
            candidate_set_recalls.append(
                intersection / max(1, len(expected_set))
            )
            candidate_set_precisions.append(
                intersection / max(1, len(actual_set))
            )
        row = {
            "history_count": history_count,
            "qksieve_target_fraction": q_fraction,
            "fier_equal_active_target_fraction": fier_equal_active_fraction,
            "fier_equal_quality_target_fraction": fier_equal_quality_fraction,
            "qksieve_actual_fraction": float(
                (q_counts / history_count).mean().item()
            ),
            "fier_equal_active_actual_fraction": float(
                (fier_active_counts / history_count).mean().item()
            ),
            "fier_equal_quality_actual_fraction": float(
                (fier_quality_counts / history_count).mean().item()
            ),
            "qksieve_query_prepare_retrieval_ms": q_retrieval_ms,
            "qksieve_complete_ms": q_complete_ms,
            "fier_equal_active_retrieval_ms": fier_active_retrieval_ms,
            "fier_equal_active_complete_ms": fier_active_complete_ms,
            "fier_equal_quality_retrieval_ms": fier_quality_retrieval_ms,
            "fier_equal_quality_complete_ms": fier_quality_complete_ms,
            "fier_materialized_score_ms": fier_score_ms,
            "fier_torch_topk_ms": fier_topk_ms,
            "fier_current_score_topk_ms": fier_current_selection_ms,
            "full_sdpa_ms": full_ms,
            "qksieve_attention_speedup": full_ms / q_complete_ms,
            "fier_equal_active_attention_speedup": (
                full_ms / fier_active_complete_ms
            ),
            "fier_equal_quality_attention_speedup": (
                full_ms / fier_quality_complete_ms
            ),
            "qksieve_vs_fier_equal_active_speedup": (
                fier_active_complete_ms / q_complete_ms
            ),
            "qksieve_vs_fier_equal_quality_speedup": (
                fier_quality_complete_ms / q_complete_ms
            ),
            "qksieve_index_ratio_of_full_kv": (
                (
                    q_code_count
                    + 2 * q_scale_count
                    + 2 * q_meta["block_hot_prefix"].numel()
                    + 8 * q_meta["head_code_bases"].numel()
                    + 8 * q_meta["head_scale_bases"].numel()
                )
                / (8 * history_count * 512)
            ),
            "fier_index_ratio_of_full_kv": (
                fier_cuda.allocated_bytes(fier_index)
                / (8 * history_count * 512)
            ),
            "qksieve_overflow": bool(q_outputs[3].any().item()),
            "fier_equal_active_overflow": bool(
                fier_active_outputs[3].any().item()
            ),
            "fier_equal_quality_overflow": bool(
                fier_quality_outputs[3].any().item()
            ),
            "fier_sampled_threshold_max_abs_error": float(
                (
                    fier_active_outputs[2] - reference_threshold
                ).abs().max().item()
            ),
            "fier_sampled_count_max_abs_error": int(
                (
                    fier_active_outputs[1] - expected_counts
                ).abs().max().item()
            ),
            "fier_sampled_candidate_set_recall": float(
                sum(candidate_set_recalls) / len(candidate_set_recalls)
            ),
            "fier_sampled_candidate_set_precision": float(
                sum(candidate_set_precisions)
                / len(candidate_set_precisions)
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            query,
            grouped_query,
            query_basis,
            projected_query,
            query_codes,
            query_scales,
            key,
            value,
            full_key,
            full_value,
            q_codes,
            q_scales,
            fier_index,
            materialized_fier_scores,
        )
        torch.cuda.empty_cache()

    output = {
        "scope": (
            "One Qwen-style GQA decode attention layer on RTX-class CUDA. "
            "All fused paths include threshold estimation, complete index "
            "scan, candidate compaction, exact sparse QK/softmax/AV, and V "
            "aggregation. QKSieve additionally includes fused Query "
            "projection and quantization. Historical index construction and "
            "per-token index append are excluded. FIER equal-quality uses "
            "twice the QKSieve candidate fraction, motivated by the held-out "
            "32K attention-mass frontier; it is a speed scenario rather than "
            "a claim that the 2x rate transfers to every length."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
