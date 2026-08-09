#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from validate_qksieve_valuesketch_oneexp_ab_20260804 import (
    allocate_outputs,
    launch,
    measure_ms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=32_768)
    parser.add_argument("--sample_count", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def launch_deterministic(
    extension: object,
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    index: dict[str, torch.Tensor | int],
    value_codes: torch.Tensor,
    value_minimum: torch.Tensor,
    value_scale: torch.Tensor,
    selection_masks: torch.Tensor,
    tail_partials: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    history: int,
    sample_count: int,
    selected_fraction: float,
    scaling: float,
) -> None:
    extension.plain_sampled_compact_gqa4_valuesketch_deterministic_out(
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
        selection_masks,
        tail_partials,
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
        16,
        256,
        scaling,
    )


def active_indices(
    outputs: dict[str, torch.Tensor],
    row: int,
    capacity: int,
) -> torch.Tensor:
    count = int(outputs["counts"].reshape(-1)[row].item())
    return outputs["indices"].reshape(-1, capacity)[row, :count]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    history = args.history_tokens
    rank = 16
    block_size = 256
    scaling = 128.0**-0.5
    selected = min(history, 1280, max(256, math.ceil(0.06 * history)))
    selected_fraction = selected / history

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
    atomic_outputs = allocate_outputs(history, rank, device)
    deterministic_outputs = allocate_outputs(history, rank, device)
    selection_masks = torch.empty(
        1,
        32,
        block_count * 8,
        dtype=torch.int32,
        device=device,
    )
    tail_partials = torch.empty(
        1,
        32,
        block_count,
        rank + 2,
        dtype=torch.float32,
        device=device,
    )
    extension = mixed_cuda.load_extension()

    atomic_call = lambda: launch(  # noqa: E731
        extension,
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        atomic_outputs,
        history,
        args.sample_count,
        selected_fraction,
        rank,
        block_size,
        scaling,
    )
    deterministic_call = lambda: launch_deterministic(  # noqa: E731
        extension,
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        selection_masks,
        tail_partials,
        deterministic_outputs,
        history,
        args.sample_count,
        selected_fraction,
        scaling,
    )

    atomic_call()
    deterministic_call()
    torch.cuda.synchronize()
    count_equal = bool(
        torch.equal(atomic_outputs["counts"], deterministic_outputs["counts"])
    )
    threshold_max_abs_error = float(
        (
            atomic_outputs["thresholds"]
            - deterministic_outputs["thresholds"]
        )
        .abs()
        .max()
        .item()
    )
    candidate_sets_equal = True
    deterministic_ordered = True
    for row in range(32):
        atomic_indices = active_indices(atomic_outputs, row, history)
        deterministic_indices = active_indices(
            deterministic_outputs, row, history
        )
        if not torch.equal(
            atomic_indices.sort().values,
            deterministic_indices,
        ):
            candidate_sets_equal = False
        if deterministic_indices.numel() > 1 and not bool(
            torch.all(deterministic_indices[1:] > deterministic_indices[:-1])
        ):
            deterministic_ordered = False

    tail_denominator_relative_error = float(
        (
            atomic_outputs["tail_denominator"]
            - deterministic_outputs["tail_denominator"]
        )
        .abs()
        .max()
        .item()
    ) / max(1.0, float(atomic_outputs["tail_denominator"].abs().max().item()))
    tail_coefficient_relative_error = float(
        (
            atomic_outputs["tail_coefficients"]
            - deterministic_outputs["tail_coefficients"]
        )
        .abs()
        .max()
        .item()
    ) / max(
        1.0,
        float(atomic_outputs["tail_coefficients"].abs().max().item()),
    )

    deterministic_call()
    torch.cuda.synchronize()
    reference = {
        name: tensor.clone() for name, tensor in deterministic_outputs.items()
    }
    deterministic_mismatch_runs = 0
    for _ in range(args.repeats - 1):
        deterministic_call()
        torch.cuda.synchronize()
        if any(
            not torch.equal(reference[name], tensor)
            for name, tensor in deterministic_outputs.items()
        ):
            deterministic_mismatch_runs += 1

    atomic_ms = measure_ms(atomic_call, args.warmup, args.iterations)
    deterministic_ms = measure_ms(
        deterministic_call, args.warmup, args.iterations
    )
    workspace_bytes = (
        selection_masks.numel() * selection_masks.element_size()
        + tail_partials.numel() * tail_partials.element_size()
    )
    result = {
        "history_tokens": history,
        "sample_count": args.sample_count,
        "target_selected_tokens": selected,
        "count_equal": count_equal,
        "candidate_sets_equal": candidate_sets_equal,
        "deterministic_candidate_ordered": deterministic_ordered,
        "threshold_max_abs_error": threshold_max_abs_error,
        "tail_denominator_relative_error": tail_denominator_relative_error,
        "tail_coefficient_relative_error": tail_coefficient_relative_error,
        "deterministic_repeats": args.repeats,
        "deterministic_mismatch_runs": deterministic_mismatch_runs,
        "atomic_ms": atomic_ms,
        "deterministic_ms": deterministic_ms,
        "deterministic_over_atomic_speedup": atomic_ms / deterministic_ms,
        "workspace_bytes": workspace_bytes,
        "workspace_mib": workspace_bytes / (1024**2),
    }
    if not (
        count_equal
        and candidate_sets_equal
        and deterministic_ordered
        and threshold_max_abs_error == 0.0
        and deterministic_mismatch_runs == 0
    ):
        raise AssertionError(json.dumps(result, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
