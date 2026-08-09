from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import bandmajor_spectral_cuda_20260729 as bandmajor_cuda
import mixedblock_spectral_cuda_20260729 as plain_cuda
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    plain_cuda.load_extension()
    bandmajor_cuda.load_extension()
    rows: list[dict[str, float | int | bool]] = []
    for history_count in sorted(
        {int(value) for value in args.lengths.split(",")}
    ):
        fraction = selected_fraction(history_count)
        sample_count = sample_count_for(fraction)
        threshold_fraction = unbiased_fraction(fraction, sample_count)
        capacity = capacity_for(history_count, fraction, sample_count)
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = varbit_cuda.quantize_projected_query(
            projected_query
        )
        packed_index = varbit_cuda.allocate_packed_index(
            HIGH_ALLOCATIONS.unsqueeze(0).cuda(),
            history_count,
            torch.float16,
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.05, 1.0)
        bandmajor_index = bandmajor_cuda.repack_bandmajor(packed_index)
        plain_outputs = allocate_outputs(capacity)
        bandmajor_outputs = allocate_outputs(capacity)

        def plain_call() -> None:
            plain_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index,
                *plain_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        def bandmajor_call() -> None:
            bandmajor_cuda.sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                bandmajor_index,
                *bandmajor_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )

        plain_call()
        bandmajor_call()
        torch.cuda.synchronize()
        iterations = min(
            args.iterations,
            50 if history_count >= 96_000 else args.iterations,
        )
        plain_ms = measure_ms(plain_call, args.warmup, iterations)
        bandmajor_ms = measure_ms(
            bandmajor_call, args.warmup, iterations
        )
        rows.append(
            {
                "history_count": history_count,
                "sample_count": sample_count,
                "candidate_count_mean": float(
                    plain_outputs[1].float().mean().item()
                ),
                "candidate_count_max_abs_diff": int(
                    (plain_outputs[1] - bandmajor_outputs[1])
                    .abs()
                    .max()
                    .item()
                ),
                "threshold_max_abs_diff": float(
                    (plain_outputs[2] - bandmajor_outputs[2])
                    .abs()
                    .max()
                    .item()
                ),
                "candidate_sets_equal": candidate_sets_equal(
                    plain_outputs, bandmajor_outputs
                ),
                "plain_ms": plain_ms,
                "bandmajor_ms": bandmajor_ms,
                "speedup": plain_ms / bandmajor_ms,
            }
        )
        del packed_index, bandmajor_index
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
