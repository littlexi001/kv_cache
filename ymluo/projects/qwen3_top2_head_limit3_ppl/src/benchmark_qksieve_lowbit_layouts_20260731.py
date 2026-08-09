from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import bandmajor_spectral_cuda_20260729 as bandmajor_cuda
import mixedblock_spectral_cuda_20260729 as plain_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_qksieve_bandmajor_20260729 import candidate_sets_equal
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


PROFILES = {
    "auto240_reference": HIGH_ALLOCATIONS,
    "fixed441_b192": torch.tensor(
        [[4, 4, 1, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed4221_b208": torch.tensor(
        [[4, 2, 2, 1, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed4421_b240": torch.tensor(
        [[4, 4, 2, 1, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed440_b160": torch.tensor(
        [[4, 4, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed420_b128": torch.tensor(
        [[4, 2, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed410_b112": torch.tensor(
        [[4, 1, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed211_b112": torch.tensor(
        [[2, 1, 1, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed220_b96": torch.tensor(
        [[2, 2, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed400_b80": torch.tensor(
        [[4, 0, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
    "fixed200_b48": torch.tensor(
        [[2, 0, 0, 0, 0, 0, 0, 0]] * 8,
        dtype=torch.int8,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def physical_bits(allocations: torch.Tensor) -> float:
    return float(
        (
            16.0 * allocations.float().sum(dim=-1)
            + 16.0 * (allocations > 0).float().sum(dim=-1)
        )
        .mean()
        .item()
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    names = [part.strip() for part in args.profiles.split(",") if part.strip()]
    unknown = sorted(set(names) - set(PROFILES))
    if not names or unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    lengths = sorted({int(part) for part in args.lengths.split(",")})

    torch.manual_seed(args.seed)
    plain_cuda.load_extension()
    bandmajor_cuda.load_extension()
    rows: list[dict[str, float | int | bool | str]] = []
    for history_count in lengths:
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
        for name in names:
            allocations = PROFILES[name].unsqueeze(0).cuda()
            packed_index = varbit_cuda.allocate_packed_index(
                allocations,
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
            validation_plain_outputs = allocate_outputs(history_count)
            validation_bandmajor_outputs = allocate_outputs(history_count)

            plain_cuda.plain_sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                packed_index,
                *validation_plain_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )
            bandmajor_cuda.sampled_threshold_compact_gqa4_indices_out(
                query_codes,
                query_scales,
                bandmajor_index,
                *validation_bandmajor_outputs,
                history_count,
                sample_count,
                threshold_fraction,
            )
            iterations = min(
                args.iterations,
                50 if history_count >= 96_000 else args.iterations,
            )
            plain_ms = measure_ms(plain_call, args.warmup, iterations)
            bandmajor_ms = measure_ms(
                bandmajor_call,
                args.warmup,
                iterations,
            )
            rows.append(
                {
                    "profile": name,
                    "history_count": history_count,
                    "physical_bits_per_token_head": physical_bits(
                        allocations
                    ),
                    "index_ratio_of_full_kv": physical_bits(allocations)
                    / 4096.0,
                    "sample_count": sample_count,
                    "candidate_count_mean": float(
                        plain_outputs[1].float().mean().item()
                    ),
                    "candidate_sets_equal": candidate_sets_equal(
                        plain_outputs,
                        bandmajor_outputs,
                    ),
                    "timing_plain_overflow_rows": int(
                        plain_outputs[3].sum().item()
                    ),
                    "timing_bandmajor_overflow_rows": int(
                        bandmajor_outputs[3].sum().item()
                    ),
                    "untruncated_candidate_sets_equal": (
                        candidate_sets_equal(
                            validation_plain_outputs,
                            validation_bandmajor_outputs,
                        )
                    ),
                    "untruncated_counts_equal": bool(
                        torch.equal(
                            validation_plain_outputs[1],
                            validation_bandmajor_outputs[1],
                        )
                    ),
                    "untruncated_threshold_max_abs_diff": float(
                        (
                            validation_plain_outputs[2]
                            - validation_bandmajor_outputs[2]
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "plain_ms": plain_ms,
                    "bandmajor_ms": bandmajor_ms,
                    "bandmajor_vs_plain_speedup": plain_ms / bandmajor_ms,
                }
            )
            del (
                packed_index,
                bandmajor_index,
                validation_plain_outputs,
                validation_bandmajor_outputs,
            )
            torch.cuda.empty_cache()

    reference = {
        int(row["history_count"]): float(row["bandmajor_ms"])
        for row in rows
        if row["profile"] == "auto240_reference"
    }
    for row in rows:
        row["bandmajor_speedup_vs_auto240"] = (
            reference[int(row["history_count"])]
            / float(row["bandmajor_ms"])
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
