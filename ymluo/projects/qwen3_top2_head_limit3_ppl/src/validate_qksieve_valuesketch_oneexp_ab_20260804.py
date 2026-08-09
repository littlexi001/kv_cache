#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import qksieve_valuesketch_cuda_20260801 as sketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131_072)
    parser.add_argument("--rank", type=int, choices=(8, 12, 16, 32), default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--old_extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_named_extension(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocate_outputs(
    history: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.empty(1, 32, history, dtype=torch.long, device=device),
        "counts": torch.empty(1, 32, dtype=torch.long, device=device),
        "thresholds": torch.empty(1, 32, dtype=torch.float32, device=device),
        "overflow": torch.empty(1, 32, dtype=torch.bool, device=device),
        "selected_denominator": torch.empty(
            1, 32, dtype=torch.float32, device=device
        ),
        "tail_denominator": torch.empty(
            1, 32, dtype=torch.float32, device=device
        ),
        "tail_coefficients": torch.empty(
            1, 32, rank, dtype=torch.float32, device=device
        ),
    }


def launch(
    extension: object,
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, torch.Tensor | int],
    value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    history: int,
    sample_count: int,
    selected_fraction: float,
    rank: int,
    block_size: int,
    scaling: float,
) -> None:
    extension.plain_sampled_compact_gqa4_valuesketch_out(
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
        value_codes,
        value_minimum,
        value_scale,
        outputs["indices"],
        outputs["counts"],
        outputs["thresholds"],
        outputs["overflow"],
        outputs["selected_denominator"],
        outputs["tail_denominator"],
        outputs["tail_coefficients"],
        history,
        sample_count,
        selected_fraction,
        rank,
        block_size,
        scaling,
    )


def measure_ms(function: object, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iterations


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    history = args.history_tokens
    rank = args.rank
    block_size = 256
    scaling = 128.0**-0.5
    selected = min(history, 1280, max(256, math.ceil(0.06 * history)))
    selected_fraction = selected / history
    sample_count = min(2048, max(256, math.ceil(16.0 / selected_fraction)))

    query = torch.randn(1, 32, 128, dtype=dtype, device=device)
    grouped_query = query.reshape(1, 8, 4, 128)
    key_basis = torch.randn(1, 8, 128, 128, dtype=dtype, device=device)
    query_codes, query_scales = query_cuda.project_quantize(
        grouped_query, key_basis
    )
    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    index = varbit_cuda.allocate_packed_index(allocation, history, dtype)
    index["packed_codes"].random_(0, 256)
    index["key_scales"].uniform_(0.01, 0.1)
    block_count = math.ceil(history / block_size)
    value_codes = torch.randint(
        0,
        256,
        (1, 8, history, rank // 2),
        dtype=torch.uint8,
        device=device,
    )
    value_minimum = torch.randn(
        1, 8, block_count, rank, dtype=dtype, device=device
    )
    value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(0.01)
    old_outputs = allocate_outputs(history, rank, device)
    new_outputs = allocate_outputs(history, rank, device)

    old_name = args.old_extension.stem
    old_extension = load_named_extension(old_name, args.old_extension)
    new_extension = mixed_cuda.load_extension()
    old_call = lambda: launch(  # noqa: E731
        old_extension,
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        old_outputs,
        history,
        sample_count,
        selected_fraction,
        rank,
        block_size,
        scaling,
    )
    new_call = lambda: launch(  # noqa: E731
        new_extension,
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        new_outputs,
        history,
        sample_count,
        selected_fraction,
        rank,
        block_size,
        scaling,
    )
    old_call()
    new_call()
    torch.cuda.synchronize()

    count_equal = bool(torch.equal(old_outputs["counts"], new_outputs["counts"]))
    threshold_max_error = float(
        (old_outputs["thresholds"] - new_outputs["thresholds"])
        .abs()
        .max()
        .item()
    )
    candidate_sets_equal = True
    for row in range(32):
        old_count = int(old_outputs["counts"].reshape(-1)[row].item())
        new_count = int(new_outputs["counts"].reshape(-1)[row].item())
        if old_count != new_count:
            candidate_sets_equal = False
            break
        old_indices = old_outputs["indices"].reshape(32, history)[row, :old_count]
        new_indices = new_outputs["indices"].reshape(32, history)[row, :new_count]
        if not torch.equal(old_indices.sort().values, new_indices.sort().values):
            candidate_sets_equal = False
            break
    denominator_relative_error = float(
        (old_outputs["tail_denominator"] - new_outputs["tail_denominator"])
        .abs()
        .max()
        .item()
    ) / max(1.0, float(old_outputs["tail_denominator"].abs().max().item()))
    coefficient_relative_error = float(
        (old_outputs["tail_coefficients"] - new_outputs["tail_coefficients"])
        .abs()
        .max()
        .item()
    ) / max(1.0, float(old_outputs["tail_coefficients"].abs().max().item()))
    old_ms = measure_ms(old_call, args.warmup, args.iterations)
    new_ms = measure_ms(new_call, args.warmup, args.iterations)
    result = {
        "history_tokens": history,
        "rank": rank,
        "count_equal": count_equal,
        "candidate_sets_equal": candidate_sets_equal,
        "threshold_max_abs_error": threshold_max_error,
        "tail_denominator_relative_error": denominator_relative_error,
        "tail_coefficient_relative_error": coefficient_relative_error,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "kernel_speedup": old_ms / new_ms,
    }
    if not count_equal or not candidate_sets_equal or threshold_max_error != 0.0:
        raise AssertionError(json.dumps(result, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
