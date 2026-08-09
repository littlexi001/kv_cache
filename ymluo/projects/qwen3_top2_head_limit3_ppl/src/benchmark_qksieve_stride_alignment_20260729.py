from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import mixedblock_spectral_cuda_20260729 as mixedblock_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_qksieve_mixedblock_cuda_20260729 import (
    HIGH_ALLOCATIONS,
    measure_ms,
    sample_count_for,
    selected_fraction,
    unbiased_fraction,
)
from benchmark_qksieve_plain_gqa4_20260729 import (
    allocate_outputs,
    capacity_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--alignments", default="2,4,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def align_index(
    packed_index: dict[str, Any],
    alignment: int,
) -> dict[str, Any]:
    capacity = int(packed_index["capacity"])
    old_strides = packed_index["code_strides"].to("cpu")
    new_strides = (
        (old_strides.to(torch.int64) + alignment - 1)
        // alignment
        * alignment
    ).to(torch.int16)
    new_bases = torch.zeros_like(
        packed_index["code_bases"], device="cpu"
    )
    cursor = 0
    for batch_index in range(old_strides.shape[0]):
        for head_index in range(old_strides.shape[1]):
            new_bases[batch_index, head_index] = cursor
            cursor += (
                capacity
                * int(new_strides[batch_index, head_index].item())
            )
    aligned_codes = torch.zeros(
        cursor,
        dtype=torch.uint8,
        device=packed_index["packed_codes"].device,
    )
    for batch_index in range(old_strides.shape[0]):
        for head_index in range(old_strides.shape[1]):
            old_stride = int(
                old_strides[batch_index, head_index].item()
            )
            new_stride = int(
                new_strides[batch_index, head_index].item()
            )
            old_base = int(
                packed_index["code_bases"][
                    batch_index, head_index
                ].item()
            )
            new_base = int(
                new_bases[batch_index, head_index].item()
            )
            source = packed_index["packed_codes"][
                old_base : old_base + capacity * old_stride
            ].reshape(capacity, old_stride)
            target = aligned_codes[
                new_base : new_base + capacity * new_stride
            ].reshape(capacity, new_stride)
            target[:, :old_stride] = source
    aligned = dict(packed_index)
    aligned["packed_codes"] = aligned_codes
    aligned["code_bases"] = new_bases.to(
        device=packed_index["packed_codes"].device
    )
    aligned["code_strides"] = new_strides.to(
        device=packed_index["packed_codes"].device
    )
    aligned["total_code_bytes"] = int(cursor)
    return aligned


def candidate_sets_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    for row in range(32):
        left_count = int(left[1].reshape(-1)[row].item())
        right_count = int(right[1].reshape(-1)[row].item())
        if left_count != right_count:
            return False
        left_indices = torch.sort(
            left[0].reshape(32, -1)[row, :left_count]
        ).values
        right_indices = torch.sort(
            right[0].reshape(32, -1)[row, :right_count]
        ).values
        if not torch.equal(left_indices, right_indices):
            return False
    return True


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    mixedblock_cuda.load_extension()
    alignments = sorted(
        {int(value) for value in args.alignments.split(",")}
    )
    rows: list[dict[str, float | int | bool]] = []
    for history_count in sorted(
        {int(value) for value in args.lengths.split(",")}
    ):
        fraction = selected_fraction(history_count)
        sample_count = sample_count_for(fraction)
        threshold_fraction = unbiased_fraction(fraction, sample_count)
        candidate_capacity = capacity_for(
            history_count, fraction, sample_count
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
        source_index = varbit_cuda.allocate_packed_index(
            HIGH_ALLOCATIONS.unsqueeze(0).cuda(),
            history_count,
            torch.float16,
        )
        source_index["packed_codes"].random_(0, 256)
        source_index["key_scales"].uniform_(0.05, 1.0)
        indexes = {
            alignment: align_index(source_index, alignment)
            for alignment in alignments
        }
        outputs = {
            alignment: allocate_outputs(candidate_capacity)
            for alignment in alignments
        }

        def call(alignment: int) -> None:
            mixedblock_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                indexes[alignment],
                *outputs[alignment],
                history_count,
                sample_count,
                threshold_fraction,
            )

        for alignment in alignments:
            call(alignment)
        torch.cuda.synchronize()
        reference_alignment = min(alignments)
        iterations = min(
            args.iterations,
            50 if history_count >= 96_000 else args.iterations,
        )
        latencies = {
            alignment: measure_ms(
                lambda alignment=alignment: call(alignment),
                args.warmup,
                iterations,
            )
            for alignment in alignments
        }
        reference_bytes = int(
            indexes[reference_alignment]["total_code_bytes"]
        )
        for alignment in alignments:
            rows.append(
                {
                    "history_count": history_count,
                    "alignment": alignment,
                    "latency_ms": latencies[alignment],
                    "speedup_vs_alignment2": (
                        latencies[reference_alignment]
                        / latencies[alignment]
                    ),
                    "candidate_sets_equal": candidate_sets_equal(
                        outputs[reference_alignment],
                        outputs[alignment],
                    ),
                    "code_bytes_ratio_vs_alignment2": (
                        int(indexes[alignment]["total_code_bytes"])
                        / reference_bytes
                    ),
                    "mean_code_stride": float(
                        indexes[alignment]["code_strides"]
                        .float()
                        .mean()
                        .item()
                    ),
                }
            )
        del source_index, indexes, outputs
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
