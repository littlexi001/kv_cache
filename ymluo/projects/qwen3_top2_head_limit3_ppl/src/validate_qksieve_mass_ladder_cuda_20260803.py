from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)


def measure_ms(function: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop) / repeats)


def packed_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, torch.Tensor | int],
    history_count: int,
) -> torch.Tensor:
    return varbit_cuda.scores(
        query_codes,
        query_scales,
        index["packed_codes"],
        index["key_scales"],
        index["bit_allocations"],
        index["code_offsets"],
        index["scale_offsets"],
        index["code_bases"],
        index["scale_bases"],
        index["code_strides"],
        index["scale_strides"],
        history_count,
        score_bias=index.get("score_bias"),
    ).reshape(1, 32, history_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    parser.add_argument("--floor_k", type=int, default=1280)
    parser.add_argument("--sample_count", type=int, default=1024)
    parser.add_argument("--growth", type=float, default=2.0)
    parser.add_argument("--target_mass", type=float, default=0.90)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    dtype = torch.bfloat16
    scaling = 128.0**-0.5
    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    torch.manual_seed(20260803)
    rows: list[dict[str, object]] = []

    for history_count in lengths:
        minimum_fraction = min(1.0, args.floor_k / history_count)
        rung_count = 1
        fraction = minimum_fraction
        while fraction < 1.0 and rung_count < 8:
            fraction *= args.growth
            rung_count += 1

        query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
        grouped_query = query.reshape(1, 8, 4, 128)
        basis = torch.randn(1, 8, 128, 128, dtype=dtype, device="cuda")
        query_codes, query_scales = query_cuda.project_quantize_active(
            grouped_query, basis, allocation
        )
        index = varbit_cuda.allocate_packed_index(
            allocation, history_count, dtype
        )
        index["packed_codes"].random_(0, 256)
        index["key_scales"].uniform_(0.01, 0.1)

        ladder_thresholds = torch.empty(
            rung_count, 1, 32, dtype=torch.float32, device="cuda"
        )
        mass_bins = torch.empty(
            rung_count + 1, 1, 32, dtype=torch.float32, device="cuda"
        )
        chosen_thresholds = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )
        chosen_rungs = torch.empty(
            1, 32, dtype=torch.long, device="cuda"
        )
        chosen_mass = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )

        def ladder() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.plain_mass_ladder_thresholds_out(
                query_codes,
                query_scales,
                index,
                ladder_thresholds,
                mass_bins,
                chosen_thresholds,
                chosen_rungs,
                chosen_mass,
                history_count,
                args.sample_count,
                minimum_fraction,
                args.growth,
                args.target_mass,
                scaling,
            )

        ladder()
        torch.cuda.synchronize()
        scores = packed_scores(
            query_codes, query_scales, index, history_count
        )
        probabilities = torch.softmax(scores.float() * scaling, dim=-1)
        selected_mask = scores.float() >= chosen_thresholds.unsqueeze(-1)
        measured_mass = probabilities.masked_fill(~selected_mask, 0.0).sum(-1)
        candidate_counts = selected_mask.sum(-1)

        def full_score_sort() -> torch.Tensor:
            active_scores = packed_scores(
                query_codes, query_scales, index, history_count
            ).float() * scaling
            sorted_scores = torch.sort(
                active_scores, dim=-1, descending=True
            ).values
            weights = torch.exp(sorted_scores - sorted_scores[..., :1])
            cumulative = torch.cumsum(weights, dim=-1)
            return torch.sum(
                cumulative < args.target_mass * cumulative[..., -1:],
                dim=-1,
            ) + 1

        ladder_ms = measure_ms(ladder, args.warmup, args.repeats)
        full_score_sort_ms = measure_ms(
            full_score_sort, args.warmup, args.repeats
        )
        rung_histogram = torch.bincount(
            chosen_rungs.flatten(), minlength=rung_count
        ).cpu().tolist()
        rows.append(
            {
                "history_tokens": history_count,
                "floor_k": args.floor_k,
                "minimum_fraction": minimum_fraction,
                "rung_count": rung_count,
                "rung_histogram": rung_histogram,
                "candidate_count_mean": float(candidate_counts.float().mean()),
                "candidate_count_maximum": int(candidate_counts.max()),
                "candidate_fraction_mean": float(
                    candidate_counts.float().mean() / history_count
                ),
                "kernel_reported_mass_minimum": float(chosen_mass.min()),
                "kernel_reported_mass_mean": float(chosen_mass.mean()),
                "recomputed_mass_minimum": float(measured_mass.min()),
                "recomputed_mass_mean": float(measured_mass.mean()),
                "mass_absolute_error_maximum": float(
                    (chosen_mass - measured_mass).abs().max()
                ),
                "mass_ladder_prepass_ms": ladder_ms,
                "materialize_and_full_sort_ms": full_score_sort_ms,
                "prepass_speedup_over_full_sort": full_score_sort_ms / ladder_ms,
            }
        )

    payload = {
        "schema": "qksieve_mass_ladder_cuda_v1",
        "setup": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
