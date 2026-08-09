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
from benchmark_hierarchical_spectral_cuda_20260727 import make_inputs


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the complete 8/4/1 progressive attention path: packed "
            "core scan, sampled threshold compaction, tail refinement, final "
            "top-k, and exact ragged K/V attention."
        )
    )
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--candidate_fraction", type=float, default=0.30)
    parser.add_argument("--candidate_capacity_fraction", type=float, default=0.40)
    parser.add_argument("--selected_fraction", type=float, default=0.06)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--full_iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not (
        0.0
        < args.selected_fraction
        <= args.candidate_fraction
        < args.candidate_capacity_fraction
        <= 1.0
    ):
        raise ValueError(
            "fractions must satisfy 0 < selected <= candidate < capacity <= 1"
        )
    if not 1 <= args.sample_count <= 256:
        raise ValueError("sample count must be in [1, 256]")
    torch.manual_seed(20260727)
    rows: list[dict[str, float | int | bool]] = []

    for history_tokens in parse_ints(args.lengths):
        tensors = make_inputs(history_tokens, args.candidate_fraction)
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1, 8, history_tokens, 128, dtype=torch.float16, device="cuda"
        )
        value = torch.randn_like(key)
        scaling = 128.0**-0.5
        candidate_capacity = min(
            history_tokens,
            max(
                1,
                math.ceil(
                    args.candidate_capacity_fraction * history_tokens
                ),
            ),
        )
        selected_count = min(
            history_tokens,
            max(1, math.ceil(args.selected_fraction * history_tokens)),
        )
        candidate_rank = torch.arange(
            candidate_capacity, device="cuda"
        ).view(1, 1, -1)
        final_counts = torch.full(
            (1, 32),
            selected_count,
            dtype=torch.long,
            device="cuda",
        )

        def core_scan() -> torch.Tensor:
            return hierarchical_cuda.core_scores(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["core_int8"],
                tensors["middle_int4"],
                tensors["key_scales"],
            )

        core_scores = core_scan()

        def candidate_compaction() -> tuple[torch.Tensor, ...]:
            return qabs_cuda.sampled_quantile_candidates(
                core_scores,
                args.sample_count,
                args.candidate_fraction,
                candidate_capacity,
            )

        (
            candidate_indices,
            candidate_counts,
            _,
            candidate_overflow,
        ) = candidate_compaction()

        def tail_refinement() -> torch.Tensor:
            return hierarchical_cuda.tail_candidate_scores(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["tail_sign"],
                tensors["key_scales"],
                candidate_indices,
            )

        tail_scores = tail_refinement()
        candidate_core_scores = torch.gather(
            core_scores, -1, candidate_indices
        )
        candidate_valid = candidate_rank < candidate_counts.unsqueeze(-1)

        def final_selection() -> torch.Tensor:
            refined = torch.where(
                candidate_valid,
                candidate_core_scores + tail_scores,
                -torch.inf,
            )
            local_indices = torch.topk(
                refined, k=selected_count, dim=-1, sorted=False
            ).indices
            return torch.gather(candidate_indices, -1, local_indices)

        final_indices = final_selection()

        def sparse_attention() -> torch.Tensor:
            return qabs_cuda.final_attention_ragged(
                query,
                key,
                value,
                final_indices,
                final_counts,
                scaling,
            )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                key,
                value,
                enable_gqa=True,
            )

        def complete_progressive_attention() -> torch.Tensor:
            current_core_scores = core_scan()
            (
                current_candidate_indices,
                current_candidate_counts,
                _,
                _,
            ) = qabs_cuda.sampled_quantile_candidates(
                current_core_scores,
                args.sample_count,
                args.candidate_fraction,
                candidate_capacity,
            )
            current_tail_scores = hierarchical_cuda.tail_candidate_scores(
                tensors["query_codes"],
                tensors["query_scales"],
                tensors["tail_sign"],
                tensors["key_scales"],
                current_candidate_indices,
            )
            current_core_candidates = torch.gather(
                current_core_scores, -1, current_candidate_indices
            )
            current_valid = (
                candidate_rank < current_candidate_counts.unsqueeze(-1)
            )
            current_refined = torch.where(
                current_valid,
                current_core_candidates + current_tail_scores,
                -torch.inf,
            )
            local_indices = torch.topk(
                current_refined,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices
            current_final_indices = torch.gather(
                current_candidate_indices, -1, local_indices
            )
            return qabs_cuda.final_attention_ragged(
                query,
                key,
                value,
                current_final_indices,
                final_counts,
                scaling,
            )

        measured_iterations = (
            min(args.iterations, 20)
            if history_tokens >= 65536
            else args.iterations
        )
        core_ms = measure_ms(core_scan, args.warmup, measured_iterations)
        compact_ms = measure_ms(
            candidate_compaction, args.warmup, measured_iterations
        )
        tail_ms = measure_ms(
            tail_refinement, args.warmup, measured_iterations
        )
        select_ms = measure_ms(
            final_selection, args.warmup, measured_iterations
        )
        sparse_ms = measure_ms(
            sparse_attention, args.warmup, measured_iterations
        )
        progressive_ms = measure_ms(
            complete_progressive_attention,
            args.warmup,
            measured_iterations,
        )
        full_ms = measure_ms(
            full_attention,
            min(args.warmup, 10),
            min(args.full_iterations, measured_iterations),
        )
        mean_candidate_ratio = float(
            (candidate_counts.float() / history_tokens).mean().item()
        )
        max_candidate_ratio = float(
            (candidate_counts.float() / history_tokens).max().item()
        )
        row: dict[str, float | int | bool] = {
            "history_tokens": history_tokens,
            "target_candidate_fraction": args.candidate_fraction,
            "mean_candidate_fraction": mean_candidate_ratio,
            "max_candidate_fraction": max_candidate_ratio,
            "candidate_capacity_fraction": (
                candidate_capacity / history_tokens
            ),
            "selected_fraction": selected_count / history_tokens,
            "candidate_overflow": bool(candidate_overflow.any().item()),
            "core_scan_ms": core_ms,
            "candidate_compaction_ms": compact_ms,
            "tail_refinement_ms": tail_ms,
            "final_topk_ms": select_ms,
            "exact_sparse_attention_ms": sparse_ms,
            "sum_of_stages_ms": (
                core_ms + compact_ms + tail_ms + select_ms + sparse_ms
            ),
            "progressive_attention_ms": progressive_ms,
            "full_sdpa_ms": full_ms,
            "full_sdpa_over_progressive_attention": (
                full_ms / progressive_ms
            ),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
        del (
            tensors,
            query,
            key,
            value,
            core_scores,
            candidate_indices,
            candidate_counts,
            candidate_overflow,
            tail_scores,
            candidate_core_scores,
            candidate_valid,
            final_indices,
            candidate_rank,
            final_counts,
        )
        torch.cuda.empty_cache()

    output = {
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
        },
        "scope": (
            "One decode attention layer with Llama-style 32 query heads, "
            "8 KV heads, and head dimension 128. Includes packed hierarchical "
            "scan, sampled threshold candidate compaction, tail refinement, "
            "final top-k, and exact ragged K/V attention. Excludes Q/K/V/O "
            "projections, MLP, index construction, and the rest of the model."
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
