from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import _get_build_directory

import mixedblock_spectral_cuda_20260729 as specialized_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


BASELINE_EXTENSION = "qksieve_mixedblock_spectral_20260729_v16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=120_000)
    parser.add_argument("--max_tokens", type=int, default=1_280)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_baseline_extension() -> object:
    build_directory = _get_build_directory(
        BASELINE_EXTENSION, verbose=False
    )
    sys.path.insert(0, build_directory)
    return importlib.import_module(BASELINE_EXTENSION)


def measure_ms(callable_, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def sample_configuration(
    history_tokens: int,
    max_tokens: int,
) -> tuple[float, int, float, int]:
    fraction = min(0.06, max_tokens / history_tokens)
    sample_count = min(2048, max(256, math.ceil(16.0 / fraction)))
    rank = max(
        1,
        min(sample_count, round(fraction * (sample_count + 1))),
    )
    threshold_fraction = (rank - 0.5) / sample_count
    standard_error = math.sqrt(
        fraction * (1.0 - fraction) / sample_count
    )
    capacity_fraction = min(
        1.0, max(0.06, fraction + 6.0 * standard_error)
    )
    capacity = max(1, math.ceil(capacity_fraction * history_tokens))
    return fraction, sample_count, threshold_fraction, capacity


def allocate_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(1, 32, capacity, dtype=torch.long, device="cuda"),
        torch.empty(1, 32, dtype=torch.long, device="cuda"),
        torch.empty(1, 32, dtype=torch.float32, device="cuda"),
        torch.empty(1, 32, dtype=torch.bool, device="cuda"),
    )


def invoke(
    extension: object,
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    outputs: tuple[torch.Tensor, ...],
    history_tokens: int,
    sample_count: int,
    threshold_fraction: float,
) -> None:
    extension.plain_sampled_compact_gqa4_indices_out(
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
        *outputs,
        history_tokens,
        sample_count,
        threshold_fraction,
    )


def candidate_sets_equal(
    baseline: tuple[torch.Tensor, ...],
    specialized: tuple[torch.Tensor, ...],
) -> bool:
    baseline_counts = baseline[1].reshape(-1)
    specialized_counts = specialized[1].reshape(-1)
    if not torch.equal(baseline_counts, specialized_counts):
        return False
    baseline_indices = baseline[0].reshape(32, -1)
    specialized_indices = specialized[0].reshape(32, -1)
    for row in range(32):
        count = int(baseline_counts[row].item())
        if not torch.equal(
            torch.sort(baseline_indices[row, :count]).values,
            torch.sort(specialized_indices[row, :count]).values,
        ):
            return False
    return True


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens <= 0 or args.max_tokens <= 0:
        raise ValueError("token counts must be positive")
    torch.manual_seed(args.seed)
    baseline = load_baseline_extension()
    specialized = specialized_cuda.load_extension()
    template = torch.load(
        args.template, map_location="cpu", weights_only=False
    )
    (
        selected_fraction,
        sample_count,
        threshold_fraction,
        capacity,
    ) = sample_configuration(args.history_tokens, args.max_tokens)

    rows: list[dict[str, object]] = []
    for layer_index in sorted(template):
        allocation = template[layer_index]["allocation"].to(
            device="cuda", dtype=torch.int8
        )
        projected_query = torch.randn(
            1, 8, 4, 128, dtype=torch.float16, device="cuda"
        )
        query_codes, query_scales = (
            varbit_cuda.quantize_projected_query(projected_query)
        )
        packed_index = varbit_cuda.allocate_packed_index(
            allocation,
            args.history_tokens,
            torch.float16,
        )
        packed_index["packed_codes"].random_(0, 256)
        packed_index["key_scales"].uniform_(0.05, 1.0)
        baseline_outputs = allocate_outputs(capacity)
        specialized_outputs = allocate_outputs(capacity)

        baseline_call = lambda: invoke(
            baseline,
            query_codes,
            query_scales,
            packed_index,
            baseline_outputs,
            args.history_tokens,
            sample_count,
            threshold_fraction,
        )
        specialized_call = lambda: invoke(
            specialized,
            query_codes,
            query_scales,
            packed_index,
            specialized_outputs,
            args.history_tokens,
            sample_count,
            threshold_fraction,
        )
        baseline_call()
        specialized_call()
        torch.cuda.synchronize()
        equal = candidate_sets_equal(
            baseline_outputs, specialized_outputs
        )
        threshold_error = float(
            (
                baseline_outputs[2] - specialized_outputs[2]
            )
            .abs()
            .max()
            .item()
        )
        baseline_ms = measure_ms(
            baseline_call, args.warmup, args.iterations
        )
        specialized_ms = measure_ms(
            specialized_call, args.warmup, args.iterations
        )
        rows.append(
            {
                "layer": int(layer_index),
                "allocation_patterns": [
                    [int(value) for value in row]
                    for row in allocation.reshape(-1, 8).tolist()
                ],
                "candidate_sets_equal": equal,
                "threshold_max_abs_diff": threshold_error,
                "baseline_ms": baseline_ms,
                "specialized_ms": specialized_ms,
                "speedup": baseline_ms / specialized_ms,
            }
        )
        del packed_index, baseline_outputs, specialized_outputs

    baseline_total = sum(float(row["baseline_ms"]) for row in rows)
    specialized_total = sum(
        float(row["specialized_ms"]) for row in rows
    )
    result = {
        "history_tokens": args.history_tokens,
        "max_tokens": args.max_tokens,
        "selected_fraction": selected_fraction,
        "sample_count": sample_count,
        "candidate_capacity": capacity,
        "all_candidate_sets_equal": all(
            bool(row["candidate_sets_equal"]) for row in rows
        ),
        "threshold_max_abs_diff": max(
            float(row["threshold_max_abs_diff"]) for row in rows
        ),
        "baseline_layer_sum_ms": baseline_total,
        "specialized_layer_sum_ms": specialized_total,
        "speedup": baseline_total / specialized_total,
        "layers": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
