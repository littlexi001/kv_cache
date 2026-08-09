#!/usr/bin/env python
"""Validate and benchmark the GQA4 adapter on the production packed index."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

import benchmark_variablebit_spectral_attention_20260727 as varbit_bench
import mixedblock_spectral_cuda_20260729 as mixed_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from run_head_top2_targeted_ppl_20260714 import (
    _packed_qmse_gqa4_adapter_metadata,
)


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,32768,131072")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260729)
    rows = []
    allocation = varbit_bench.ALLOCATION_PROFILES[
        "qmse_total_b15"
    ].unsqueeze(0).cuda()
    for history_count in sorted(
        {int(item) for item in args.lengths.split(",") if item}
    ):
        selected_fraction = min(0.06, 1280.0 / history_count)
        sample_count = min(
            2048, max(256, math.ceil(16.0 / selected_fraction))
        )
        rank = max(
            1,
            min(
                sample_count,
                int(round(selected_fraction * (sample_count + 1))),
            ),
        )
        threshold_fraction = (rank - 0.5) / sample_count
        capacity_fraction = min(
            1.0,
            max(
                0.06,
                selected_fraction
                + 6.0
                * math.sqrt(
                    selected_fraction
                    * (1.0 - selected_fraction)
                    / sample_count
                ),
            ),
        )
        candidate_capacity = min(
            history_count,
            max(1, math.ceil(capacity_fraction * history_count)),
        )
        packed_index = varbit_cuda.allocate_packed_index(
            allocation, history_count, torch.float16
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.001, 0.03)
        state: dict[str, object] = {}
        adapter = _packed_qmse_gqa4_adapter_metadata(
            packed_index, state
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = (
            varbit_cuda.quantize_projected_query(projected_query)
        )
        shape = (1, 32, candidate_capacity)
        generic_outputs = (
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.float32, device="cuda"),
            torch.empty(1, 32, dtype=torch.long, device="cuda"),
            torch.empty(1, 32, dtype=torch.float32, device="cuda"),
            torch.empty(1, 32, dtype=torch.bool, device="cuda"),
        )
        gqa4_outputs = (
            torch.empty_like(generic_outputs[0]),
            torch.empty_like(generic_outputs[2]),
            torch.empty_like(generic_outputs[3]),
            torch.empty_like(generic_outputs[4]),
        )

        def generic() -> tuple[torch.Tensor, ...]:
            return varbit_cuda.sampled_threshold_compact_out(
                query_codes,
                query_scales,
                packed_index["packed_codes"],
                packed_index["key_scales"],
                packed_index["bit_allocations"],
                packed_index["code_offsets"],
                packed_index["scale_offsets"],
                packed_index["code_bases"],
                packed_index["scale_bases"],
                packed_index["code_strides"],
                packed_index["scale_strides"],
                *generic_outputs,
                history_count,
                sample_count,
                threshold_fraction,
                score_bias=packed_index["score_bias"],
            )

        def gqa4() -> tuple[torch.Tensor, ...]:
            return mixed_cuda.sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index["packed_codes"],
                packed_index["key_scales"],
                adapter,
                *gqa4_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        generic()
        gqa4()
        generic_ms = measure_ms(
            generic, args.warmup, args.iterations
        )
        gqa4_ms = measure_ms(gqa4, args.warmup, args.iterations)
        generic()
        gqa4()
        set_recalls = []
        set_precisions = []
        for row in range(32):
            generic_count = int(
                generic_outputs[2].reshape(-1)[row].item()
            )
            gqa4_count = int(gqa4_outputs[1].reshape(-1)[row].item())
            generic_set = set(
                generic_outputs[0]
                .reshape(32, -1)[row, :generic_count]
                .cpu()
                .tolist()
            )
            gqa4_set = set(
                gqa4_outputs[0]
                .reshape(32, -1)[row, :gqa4_count]
                .cpu()
                .tolist()
            )
            intersection = len(generic_set & gqa4_set)
            set_recalls.append(intersection / max(1, len(generic_set)))
            set_precisions.append(intersection / max(1, len(gqa4_set)))
        result = {
            "history_count": history_count,
            "selected_fraction": selected_fraction,
            "sample_count": sample_count,
            "generic_ms": generic_ms,
            "gqa4_ms": gqa4_ms,
            "speedup": generic_ms / gqa4_ms,
            "threshold_max_abs_error": float(
                (
                    generic_outputs[3] - gqa4_outputs[2]
                ).abs().max().item()
            ),
            "count_max_abs_error": int(
                (
                    generic_outputs[2] - gqa4_outputs[1]
                ).abs().max().item()
            ),
            "candidate_set_recall": sum(set_recalls) / len(set_recalls),
            "candidate_set_precision": (
                sum(set_precisions) / len(set_precisions)
            ),
            "generic_overflow": bool(generic_outputs[4].any().item()),
            "gqa4_overflow": bool(gqa4_outputs[3].any().item()),
        }
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del packed_index, projected_query, query_codes, query_scales
        torch.cuda.empty_cache()
    output = {
        "scope": (
            "Production variable-bit packed index; identical Query codes, "
            "threshold estimator, candidate capacity, and selected fraction."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
