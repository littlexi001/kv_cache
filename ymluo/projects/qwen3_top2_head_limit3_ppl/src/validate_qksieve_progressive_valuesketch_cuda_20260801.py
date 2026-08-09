#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
)
from validate_qksieve_valuesketch_cuda_20260801 import unpack_int4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def outputs(history: int) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.empty(1, 32, history, dtype=torch.long, device="cuda"),
        "counts": torch.empty(1, 32, dtype=torch.long, device="cuda"),
        "thresholds": torch.empty(1, 32, dtype=torch.float32, device="cuda"),
        "overflow": torch.empty(1, 32, dtype=torch.bool, device="cuda"),
        "selected_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "tail_denominator": torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        ),
        "tail_coefficients": torch.empty(
            1, 32, 32, dtype=torch.float32, device="cuda"
        ),
        "refinement_flags": torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 256:
        raise ValueError("history_tokens must be at least 256")
    torch.manual_seed(args.seed)
    history = args.history_tokens
    dtype = torch.bfloat16
    block_size = 256
    value_stride = math.ceil((history + 513) / block_size) * block_size
    value_blocks = value_stride // block_size
    scaling = 128.0**-0.5
    selected = min(history, 1280, max(256, math.ceil(0.06 * history)))
    fraction = selected / history
    samples = min(2048, max(256, math.ceil(16.0 / fraction)))

    query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
    grouped_query = query.reshape(1, 8, 4, 128)
    key_basis = torch.randn(1, 8, 128, 128, dtype=dtype, device="cuda")
    query_codes, query_scales = query_cuda.project_quantize(
        grouped_query, key_basis
    )
    allocation = ALLOCATION_PROFILES["qmse_total_b15"].unsqueeze(0).cuda()
    index = varbit_cuda.allocate_packed_index(allocation, history, dtype)
    index["packed_codes"].random_(0, 256)
    index["key_scales"].uniform_(0.01, 0.1)

    value_codes = torch.randint(
        0,
        256,
        (1, 8, value_stride, 16),
        dtype=torch.uint8,
        device="cuda",
    )
    value_minimum = torch.randn(
        1, 8, value_blocks, 32, dtype=dtype, device="cuda"
    )
    value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(0.01)
    coefficients = unpack_int4(
        value_codes, value_minimum, value_scale, block_size
    )[..., :history, :]

    proxy_scores = varbit_cuda.scores(
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
        history,
    ).reshape(1, 32, history).float()

    run_results: dict[str, dict[str, float | int]] = {}
    retained: dict[str, dict[str, torch.Tensor]] = {}
    for name, residual, tolerance in (
        ("base8", torch.zeros(1, 8, device="cuda"), 1.0),
        ("refine32", torch.ones(1, 8, device="cuda"), 0.0),
    ):
        result = outputs(history)
        mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_progressive_out(
            query_codes,
            query_scales,
            index,
            value_codes,
            value_minimum,
            value_scale,
            residual,
            result["indices"],
            result["counts"],
            result["thresholds"],
            result["overflow"],
            result["selected_denominator"],
            result["tail_denominator"],
            result["tail_coefficients"],
            result["refinement_flags"],
            history,
            samples,
            fraction,
            block_size,
            scaling,
            tolerance,
        )
        torch.cuda.synchronize()
        tail = proxy_scores < result["thresholds"].unsqueeze(-1)
        shifted_scores = (
            (proxy_scores - result["thresholds"].unsqueeze(-1)) * scaling
        )
        expected_tail = torch.exp(shifted_scores).masked_fill(~tail, 0.0)
        # The fused kernel clamps only selected logits. Tail logits are
        # non-positive by construction, while synthetic validation inputs can
        # make selected logits overflow even though real attention logits do not.
        expected_selected = torch.exp(shifted_scores.clamp_max(70.0)).masked_fill(
            tail, 0.0
        )
        grouped_tail = expected_tail.reshape(1, 8, 4, history)
        expected_coefficients = torch.einsum(
            "bhgn,bhnr->bhgr", grouped_tail, coefficients
        ).reshape(1, 32, 32)
        active_rank = 8 if name == "base8" else 32
        coefficient_error = float(
            (
                result["tail_coefficients"][..., :active_rank]
                - expected_coefficients[..., :active_rank]
            )
            .abs()
            .max()
            .item()
        )
        coefficient_relative = coefficient_error / max(
            1.0,
            float(expected_coefficients[..., :active_rank].abs().max().item()),
        )
        denominator_relative = float(
            (
                result["tail_denominator"] - expected_tail.sum(-1)
            ).abs().max().item()
        ) / max(1.0, float(expected_tail.sum(-1).abs().max().item()))
        selected_relative = float(
            (
                result["selected_denominator"] - expected_selected.sum(-1)
            ).abs().max().item()
        ) / max(1.0, float(expected_selected.sum(-1).abs().max().item()))
        inactive_maximum = float(
            result["tail_coefficients"][..., active_rank:].abs().max().item()
        ) if active_rank < 32 else 0.0
        if coefficient_relative > 1.0e-2:
            raise AssertionError(f"{name} coefficient error {coefficient_relative}")
        if denominator_relative > 1.0e-2 or selected_relative > 1.0e-2:
            raise AssertionError(f"{name} denominator mismatch")
        if inactive_maximum != 0.0:
            raise AssertionError("rank-8 path wrote inactive coefficients")
        expected_flags = 0 if name == "base8" else 32
        flag_count = int(result["refinement_flags"].sum().item())
        if flag_count != expected_flags:
            raise AssertionError(
                f"{name} refinement count {flag_count} != {expected_flags}"
            )
        run_results[name] = {
            "coefficient_relative_error": coefficient_relative,
            "tail_denominator_relative_error": denominator_relative,
            "selected_denominator_relative_error": selected_relative,
            "refinement_heads": flag_count,
            "candidate_count_mean": float(result["counts"].float().mean()),
        }
        retained[name] = result

    standard = outputs(history)
    mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        standard["indices"],
        standard["counts"],
        standard["thresholds"],
        standard["overflow"],
        standard["selected_denominator"],
        standard["tail_denominator"],
        standard["tail_coefficients"],
        history,
        samples,
        fraction,
        block_size,
        scaling,
    )
    torch.cuda.synchronize()
    progressive_standard_relative = float(
        (
            retained["refine32"]["tail_coefficients"]
            - standard["tail_coefficients"]
        ).abs().max().item()
    ) / max(1.0, float(standard["tail_coefficients"].abs().max().item()))
    if progressive_standard_relative > 1.0e-2:
        raise AssertionError(
            "progressive all-refine path disagrees with fixed rank32"
        )

    print(
        "QKSIEVE_PROGRESSIVE_VALUESKETCH_CUDA_VALIDATION_OK",
        {
            "history_tokens": history,
            "padded_value_stride": value_stride,
            "progressive_vs_fixed32_relative_error": (
                progressive_standard_relative
            ),
            **run_results,
        },
    )


if __name__ == "__main__":
    main()
