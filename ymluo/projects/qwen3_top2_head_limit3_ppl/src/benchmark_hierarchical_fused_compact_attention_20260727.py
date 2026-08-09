from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import hierarchical_spectral_cuda_20260727 as hierarchical_cuda
import qabs_cuda_kernels as qabs_cuda
import sparse_score_self_cuda_20260727 as sparse_score_self_cuda
from benchmark_hierarchical_spectral_cuda_20260727 import make_inputs


DEFAULT_SCHEDULE = {
    8192: 0.06,
    16384: 0.06,
    32768: 0.04,
    65536: 0.02,
    131072: 0.01,
}


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("expected at least one integer")
    return values


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


def selected_fraction_for_length(history_tokens: int) -> float:
    if history_tokens in DEFAULT_SCHEDULE:
        return DEFAULT_SCHEDULE[history_tokens]
    return min(0.06, 1280.0 / history_tokens)


def append_self(
    query: torch.Tensor,
    key: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_indices = torch.cat(
        (indices, torch.zeros_like(indices[..., :1])), dim=-1
    )
    packed_scores = torch.cat(
        (scores, torch.full_like(scores[..., :1], -torch.inf)), dim=-1
    )
    positions = counts.unsqueeze(-1)
    self_indices = torch.full_like(positions, key.shape[2] - 1)
    kv_heads = int(key.shape[1])
    groups = int(query.shape[1]) // kv_heads
    self_key = key[:, :, -1, :].repeat_interleave(groups, dim=1)
    self_scores = (
        (query.float() * self_key.float()).sum(dim=-1, keepdim=True)
        * scaling
    )
    packed_indices.scatter_(-1, positions, self_indices)
    packed_scores.scatter_(-1, positions, self_scores)
    return packed_indices, packed_scores, counts + 1


@torch.inference_mode()
def validate_compaction() -> dict[str, float | int | bool]:
    history_tokens = 4096
    tensors = make_inputs(history_tokens, 0.30)
    capacity = math.ceil(0.15 * history_tokens)
    (
        indices,
        scores,
        counts,
        thresholds,
        overflow,
    ) = hierarchical_cuda.sampled_threshold_compact(
        tensors["query_codes"],
        tensors["query_scales"],
        tensors["core_int8"],
        tensors["middle_int4"],
        tensors["tail_sign"],
        tensors["key_scales"],
        256,
        0.30,
        0.06,
        capacity,
    )
    core = hierarchical_cuda.core_scores(
        tensors["query_codes"],
        tensors["query_scales"],
        tensors["core_int8"],
        tensors["middle_int4"],
        tensors["key_scales"],
    )
    all_indices = torch.arange(
        history_tokens, device="cuda", dtype=torch.long
    ).view(1, 1, -1).expand(1, 32, -1).contiguous()
    full = core + hierarchical_cuda.tail_candidate_scores(
        tensors["query_codes"],
        tensors["query_scales"],
        tensors["tail_sign"],
        tensors["key_scales"],
        all_indices,
    )
    expected = (
        (core >= thresholds[..., 0:1])
        & (full >= thresholds[..., 1:2])
    )
    count_error = int(
        (
            counts
            - expected.sum(dim=-1).clamp_max(capacity)
        )
        .abs()
        .max()
        .item()
    )
    score_error = 0.0
    membership_error = 0
    for head in range(32):
        count = int(counts[0, head].item())
        selected = indices[0, head, :count]
        membership_error = max(
            membership_error,
            int((~expected[0, head, selected]).sum().item()),
        )
        if count:
            score_error = max(
                score_error,
                float(
                    (
                        scores[0, head, :count]
                        - full[0, head, selected]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
    return {
        "count_error": count_error,
        "membership_error": membership_error,
        "max_score_error": score_error,
        "overflow": bool(overflow.any().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused sampled-threshold 8/4/1 retrieval followed by "
            "exact split-parallel sparse attention."
        )
    )
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--candidate_fraction", type=float, default=0.30)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--capacity_multiplier", type=float, default=2.0)
    parser.add_argument("--minimum_capacity_fraction", type=float, default=0.04)
    parser.add_argument("--split_count", type=int, default=16)
    parser.add_argument(
        "--consumer_mode",
        choices=(
            "staged",
            "fused_ragged_self_split",
            "score_self_split",
            "proxy_score_self_split",
        ),
        default="staged",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--full_iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260727)
    correctness = validate_compaction()
    if (
        correctness["count_error"] != 0
        or correctness["membership_error"] != 0
        or correctness["max_score_error"] > 1.0e-4
    ):
        raise RuntimeError(f"fused compaction correctness failure: {correctness}")

    rows: list[dict[str, float | int | bool]] = []
    for history_tokens in parse_ints(args.lengths):
        selected_fraction = selected_fraction_for_length(history_tokens)
        capacity_fraction = min(
            1.0,
            max(
                args.minimum_capacity_fraction,
                args.capacity_multiplier * selected_fraction,
            ),
        )
        capacity = min(
            history_tokens,
            max(1, math.ceil(capacity_fraction * history_tokens)),
        )
        tensors = make_inputs(history_tokens, args.candidate_fraction)
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        scaling = 128.0**-0.5

        def retrieve() -> tuple[torch.Tensor, ...]:
            return hierarchical_cuda.sampled_threshold_compact(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["core_int8"],
                tensors["middle_int4"],
                tensors["tail_sign"],
                tensors["key_scales"],
                args.sample_count,
                args.candidate_fraction,
                selected_fraction,
                capacity,
            )

        (
            candidate_indices,
            candidate_proxy_scores,
            candidate_counts,
            _,
            overflow,
        ) = retrieve()

        def exact_scores() -> torch.Tensor:
            return qabs_cuda.candidate_compact_scores_ragged(
                query,
                key,
                candidate_indices,
                candidate_counts,
                scaling,
            )

        candidate_exact_scores = exact_scores()
        (
            indices_with_self,
            scores_with_self,
            counts_with_self,
        ) = append_self(
            query,
            key,
            candidate_indices,
            candidate_exact_scores,
            candidate_counts,
            scaling,
        )

        def value_attention() -> torch.Tensor:
            return qabs_cuda.final_attention_from_scores_split(
                value,
                indices_with_self,
                scores_with_self,
                counts_with_self,
                min(args.split_count, 16),
            )

        def candidate_consumption() -> torch.Tensor:
            if args.consumer_mode == "proxy_score_self_split":
                return sparse_score_self_cuda.attention(
                    query,
                    key,
                    value,
                    candidate_indices,
                    candidate_proxy_scores * scaling,
                    candidate_counts,
                    scaling,
                    args.split_count,
                )
            if args.consumer_mode == "fused_ragged_self_split":
                return qabs_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    candidate_indices,
                    candidate_counts,
                    scaling,
                    args.split_count,
                )
            current_scores = exact_scores()
            if args.consumer_mode == "score_self_split":
                return sparse_score_self_cuda.attention(
                    query,
                    key,
                    value,
                    candidate_indices,
                    current_scores,
                    candidate_counts,
                    scaling,
                    args.split_count,
                )
            (
                current_indices,
                current_scores_with_self,
                current_counts,
            ) = append_self(
                query,
                key,
                candidate_indices,
                current_scores,
                candidate_counts,
                scaling,
            )
            return qabs_cuda.final_attention_from_scores_split(
                value,
                current_indices,
                current_scores_with_self,
                current_counts,
                args.split_count,
            )

        def complete_attention() -> torch.Tensor:
            (
                current_indices,
                current_proxy_scores,
                current_counts,
                _,
                _,
            ) = retrieve()
            if args.consumer_mode == "proxy_score_self_split":
                return sparse_score_self_cuda.attention(
                    query,
                    key,
                    value,
                    current_indices,
                    current_proxy_scores * scaling,
                    current_counts,
                    scaling,
                    args.split_count,
                )
            if args.consumer_mode == "fused_ragged_self_split":
                return qabs_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    current_counts,
                    scaling,
                    args.split_count,
                )
            current_scores = qabs_cuda.candidate_compact_scores_ragged(
                query,
                key,
                current_indices,
                current_counts,
                scaling,
            )
            if args.consumer_mode == "score_self_split":
                return sparse_score_self_cuda.attention(
                    query,
                    key,
                    value,
                    current_indices,
                    current_scores,
                    current_counts,
                    scaling,
                    args.split_count,
                )
            (
                current_indices_with_self,
                current_scores_with_self,
                current_counts_with_self,
            ) = append_self(
                query,
                key,
                current_indices,
                current_scores,
                current_counts,
                scaling,
            )
            return qabs_cuda.final_attention_from_scores_split(
                value,
                current_indices_with_self,
                current_scores_with_self,
                current_counts_with_self,
                args.split_count,
            )

        if args.consumer_mode == "proxy_score_self_split":
            (
                proxy_indices_with_self,
                proxy_scores_with_self,
                proxy_counts_with_self,
            ) = append_self(
                query,
                key,
                candidate_indices,
                candidate_proxy_scores * scaling,
                candidate_counts,
                scaling,
            )
            reference_attention = qabs_cuda.final_attention_from_scores_split(
                value,
                proxy_indices_with_self,
                proxy_scores_with_self,
                proxy_counts_with_self,
                min(args.split_count, 16),
            )
        else:
            reference_attention = qabs_cuda.final_attention_ragged_self_warp(
                query,
                key,
                value,
                candidate_indices,
                candidate_counts,
                scaling,
            )
        tested_attention = candidate_consumption()
        consumer_max_abs_error = float(
            (tested_attention - reference_attention).abs().max().item()
        )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key,
                value,
                enable_gqa=True,
            )

        measured_iterations = (
            min(args.iterations, 20)
            if history_tokens >= 65536
            else args.iterations
        )
        retrieval_ms = measure_ms(
            retrieve, args.warmup, measured_iterations
        )
        exact_score_ms = measure_ms(
            exact_scores, args.warmup, measured_iterations
        )
        value_ms = measure_ms(
            value_attention, args.warmup, measured_iterations
        )
        consume_ms = measure_ms(
            candidate_consumption, args.warmup, measured_iterations
        )
        complete_ms = measure_ms(
            complete_attention, args.warmup, measured_iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(args.warmup, 10),
            min(args.full_iterations, measured_iterations),
        )
        mean_selected_fraction = float(
            (candidate_counts.float() / history_tokens).mean().item()
        )
        row: dict[str, float | int | bool] = {
            "history_tokens": history_tokens,
            "selected_fraction_target": selected_fraction,
            "selected_fraction_mean": mean_selected_fraction,
            "selected_fraction_max": float(
                (candidate_counts.float() / history_tokens).max().item()
            ),
            "capacity_fraction": capacity / history_tokens,
            "overflow": bool(overflow.any().item()),
            "fused_retrieval_ms": retrieval_ms,
            "exact_candidate_qk_ms": exact_score_ms,
            "value_attention_ms": value_ms,
            "candidate_consumption_ms": consume_ms,
            "candidate_consumer_mode": args.consumer_mode,
            "candidate_consumer_max_abs_error": consumer_max_abs_error,
            "complete_attention_ms": complete_ms,
            "sum_retrieval_and_consumption_ms": (
                retrieval_ms + consume_ms
            ),
            "full_sdpa_ms": full_ms,
            "full_sdpa_over_complete_attention": full_ms / complete_ms,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
        del (
            tensors,
            query,
            key,
            value,
            candidate_indices,
            candidate_proxy_scores,
            candidate_counts,
            candidate_exact_scores,
            indices_with_self,
            scores_with_self,
            counts_with_self,
            overflow,
        )
        torch.cuda.empty_cache()

    output = {
        "correctness": correctness,
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
            "selected_fraction_schedule": DEFAULT_SCHEDULE,
        },
        "scope": (
            "One decode attention layer, 32 query heads, 8 KV heads, "
            "head dimension 128. Includes sampled thresholds, packed 8/4/1 "
            "scan, direct final-candidate compaction, the current token, and "
            "split-parallel value attention. Exact candidate QK is omitted "
            "only in proxy_score_self_split mode. Excludes index construction "
            "and all non-attention model work."
        ),
        "rows": rows,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
