#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

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
    parser.add_argument("--history_tokens", type=int, default=4096)
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def unpack_int4(
    packed: torch.Tensor,
    minimum: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    rank = packed.shape[-1] * 2
    codes = torch.empty(
        *packed.shape[:-1], rank, dtype=torch.float32, device=packed.device
    )
    codes[..., 0::2] = (packed & 0x0F).float()
    codes[..., 1::2] = (packed >> 4).float()
    block_ids = torch.arange(packed.shape[2], device=packed.device) // block_size
    return (
        codes * scale.float().index_select(2, block_ids)
        + minimum.float().index_select(2, block_ids)
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 256:
        raise ValueError("history_tokens must be at least 256")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    history = args.history_tokens
    rank = args.rank
    block_size = 256
    selected = min(history, 1280, max(256, math.ceil(0.06 * history)))
    selected_fraction = selected / history
    sample_count = min(
        2048, max(256, math.ceil(16.0 / selected_fraction))
    )
    capacity = history
    scaling = 128.0**-0.5

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
        (1, 8, history, rank // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    block_count = math.ceil(history / block_size)
    value_minimum = torch.randn(
        1, 8, block_count, rank, dtype=dtype, device="cuda"
    )
    value_scale = torch.rand_like(value_minimum).mul_(0.1).add_(0.01)
    coefficients = unpack_int4(
        value_codes, value_minimum, value_scale, block_size
    )
    value_mean = torch.randn(1, 8, 128, dtype=dtype, device="cuda")
    value_basis = torch.randn(
        1, 8, 128, rank, dtype=dtype, device="cuda"
    ).mul_(0.1)

    candidate_indices = torch.empty(
        1, 32, capacity, dtype=torch.long, device="cuda"
    )
    candidate_counts = torch.empty(1, 32, dtype=torch.long, device="cuda")
    thresholds = torch.empty(1, 32, dtype=torch.float32, device="cuda")
    overflow = torch.empty(1, 32, dtype=torch.bool, device="cuda")
    selected_denominator = torch.empty(
        1, 32, dtype=torch.float32, device="cuda"
    )
    tail_denominator = torch.empty(
        1, 32, dtype=torch.float32, device="cuda"
    )
    tail_coefficients = torch.empty(
        1, 32, rank, dtype=torch.float32, device="cuda"
    )
    mixed_cuda.plain_sampled_threshold_compact_gqa4_valuesketch_out(
        query_codes,
        query_scales,
        index,
        value_codes,
        value_minimum,
        value_scale,
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_coefficients,
        history,
        sample_count,
        selected_fraction,
        block_size,
        scaling,
    )
    torch.cuda.synchronize()
    if bool(overflow.any().item()):
        raise AssertionError("candidate output overflowed")

    mass_indices = torch.empty_like(candidate_indices)
    mass_counts = torch.empty_like(candidate_counts)
    mass_thresholds = torch.empty_like(thresholds)
    mass_overflow = torch.empty_like(overflow)
    mass_selected_denominator = torch.empty_like(selected_denominator)
    mass_tail_denominator = torch.empty_like(tail_denominator)
    mixed_cuda.plain_sampled_threshold_compact_gqa4_mass_out(
        query_codes,
        query_scales,
        index,
        mass_indices,
        mass_counts,
        mass_thresholds,
        mass_overflow,
        mass_selected_denominator,
        mass_tail_denominator,
        history,
        sample_count,
        selected_fraction,
        scaling,
    )
    torch.cuda.synchronize()
    if bool(mass_overflow.any().item()):
        raise AssertionError("mass-only candidate output overflowed")

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
    tail_mask = proxy_scores < thresholds.unsqueeze(-1)
    expected_counts = (~tail_mask).sum(dim=-1)
    candidate_count_error = int(
        (candidate_counts - expected_counts).abs().max().item()
    )
    if candidate_count_error > 2:
        raise AssertionError(
            f"candidate counts differ from materialized proxy: {candidate_count_error}"
        )
    tail_weights = torch.exp(
        (proxy_scores - thresholds.unsqueeze(-1)) * scaling
    ).masked_fill(~tail_mask, 0.0)
    selected_weights = torch.exp(
        torch.clamp(
            (proxy_scores - thresholds.unsqueeze(-1)) * scaling,
            max=70.0,
        )
    ).masked_fill(tail_mask, 0.0)
    expected_selected_denominator = selected_weights.sum(dim=-1)
    expected_denominator = tail_weights.sum(dim=-1)
    grouped_weights = tail_weights.reshape(1, 8, 4, history)
    expected_coefficients = torch.einsum(
        "bhgn,bhnr->bhgr", grouped_weights, coefficients
    ).reshape(1, 32, rank)
    denominator_error = float(
        (tail_denominator - expected_denominator).abs().max().item()
    )
    selected_denominator_error = float(
        (
            selected_denominator - expected_selected_denominator
        ).abs().max().item()
    )
    coefficient_error = float(
        (tail_coefficients - expected_coefficients).abs().max().item()
    )
    denominator_relative = denominator_error / max(
        1.0, float(expected_denominator.abs().max().item())
    )
    selected_denominator_relative = selected_denominator_error / max(
        1.0, float(expected_selected_denominator.abs().max().item())
    )
    coefficient_relative = coefficient_error / max(
        1.0, float(expected_coefficients.abs().max().item())
    )
    mass_threshold_error = float(
        (mass_thresholds - thresholds).abs().max().item()
    )
    mass_count_error = int((mass_counts - candidate_counts).abs().max().item())
    mass_selected_relative = float(
        (mass_selected_denominator - expected_selected_denominator)
        .abs()
        .max()
        .item()
    ) / max(1.0, float(expected_selected_denominator.abs().max().item()))
    mass_tail_relative = float(
        (mass_tail_denominator - expected_denominator).abs().max().item()
    ) / max(1.0, float(expected_denominator.abs().max().item()))
    if (
        denominator_relative > 1.0e-2
        or selected_denominator_relative > 1.0e-2
        or coefficient_relative > 1.0e-2
        or mass_threshold_error > 1.0e-5
        or mass_count_error > 0
        or mass_selected_relative > 1.0e-2
        or mass_tail_relative > 1.0e-2
    ):
        raise AssertionError(
            "tail scan mismatch: "
            f"selected_den={selected_denominator_relative:.6g}, "
            f"tail_den={denominator_relative:.6g}, "
            f"coeff={coefficient_relative:.6g}, "
            f"mass_threshold={mass_threshold_error:.6g}, "
            f"mass_count={mass_count_error}, "
            f"mass_selected={mass_selected_relative:.6g}, "
            f"mass_tail={mass_tail_relative:.6g}"
        )

    key = torch.randn(1, 8, history + 1, 128, dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    output = sketch_cuda.exact_selected_plus_tail(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        scaling,
    ).squeeze(1).float()
    tail_alpha = 0.5
    alpha_output = sketch_cuda.exact_selected_plus_tail(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_coefficients,
        value_mean,
        value_basis,
        scaling,
        tail_alpha,
    ).squeeze(1).float()
    zero_coefficients = torch.zeros(
        1, 32, 1, dtype=torch.float32, device="cuda"
    )
    zero_basis = torch.zeros(1, 8, 128, 1, dtype=dtype, device="cuda")
    mean_tail_output = sketch_cuda.exact_selected_plus_tail(
        query,
        key,
        value,
        mass_indices,
        mass_counts,
        mass_thresholds,
        mass_tail_denominator,
        zero_coefficients,
        value_mean,
        zero_basis,
        scaling,
        tail_alpha,
    ).squeeze(1).float()

    reference = torch.empty_like(output)
    alpha_reference = torch.empty_like(output)
    mean_tail_reference = torch.empty_like(output)
    for row in range(32):
        kv_head = row // 4
        count = int(candidate_counts[0, row].item())
        indices = candidate_indices[0, row, :count]
        exact_indices = torch.cat(
            (indices, torch.tensor([history], device="cuda"))
        )
        exact_key = key[0, kv_head].index_select(0, exact_indices).float()
        exact_value = value[0, kv_head].index_select(0, exact_indices).float()
        logits = exact_key @ query[0, row].float() * scaling
        exact_maximum = logits.max()
        exact_weights = torch.exp(logits - exact_maximum)
        factor = torch.exp(
            thresholds[0, row] * scaling - exact_maximum
        )
        tail_vector = (
            tail_denominator[0, row] * value_mean[0, kv_head].float()
            + value_basis[0, kv_head].float()
            @ tail_coefficients[0, row]
        )
        reference[0, row] = (
            exact_weights @ exact_value + factor * tail_vector
        ) / (exact_weights.sum() + factor * tail_denominator[0, row])
        alpha_reference[0, row] = (
            exact_weights @ exact_value
            + tail_alpha * factor * tail_vector
        ) / (
            exact_weights.sum()
            + tail_alpha * factor * tail_denominator[0, row]
        )
        mass_count = int(mass_counts[0, row].item())
        mass_row_indices = mass_indices[0, row, :mass_count]
        mass_exact_indices = torch.cat(
            (mass_row_indices, torch.tensor([history], device="cuda"))
        )
        mass_key = key[0, kv_head].index_select(
            0, mass_exact_indices
        ).float()
        mass_value = value[0, kv_head].index_select(
            0, mass_exact_indices
        ).float()
        mass_logits = mass_key @ query[0, row].float() * scaling
        mass_maximum = mass_logits.max()
        mass_weights = torch.exp(mass_logits - mass_maximum)
        mass_factor = torch.exp(
            mass_thresholds[0, row] * scaling - mass_maximum
        )
        mean_tail_reference[0, row] = (
            mass_weights @ mass_value
            + tail_alpha
            * mass_factor
            * mass_tail_denominator[0, row]
            * value_mean[0, kv_head].float()
        ) / (
            mass_weights.sum()
            + tail_alpha * mass_factor * mass_tail_denominator[0, row]
        )
    output_error = float((output - reference).abs().max().item())
    if output_error > 0.04:
        raise AssertionError(f"attention output mismatch: {output_error:.6g}")
    alpha_output_error = float(
        (alpha_output - alpha_reference).abs().max().item()
    )
    if alpha_output_error > 0.04:
        raise AssertionError(
            f"alpha-fused attention mismatch: {alpha_output_error:.6g}"
        )
    mean_tail_output_error = float(
        (mean_tail_output - mean_tail_reference).abs().max().item()
    )
    if mean_tail_output_error > 0.04:
        raise AssertionError(
            f"mean-tail attention mismatch: {mean_tail_output_error:.6g}"
        )

    value_storage = torch.randn(
        1, 8, history + 7, 128, dtype=dtype, device="cuda"
    )
    strided_history = value_storage[..., :history, :]
    appended_codes = value_codes.clone()
    append_start = history - 1
    sketch_cuda.append_int4_out(
        strided_history,
        value_mean,
        value_basis,
        value_minimum,
        value_scale,
        appended_codes,
        append_start,
        history,
        block_size,
    )
    block = append_start // block_size
    expected_append_coefficients = torch.einsum(
        "bhd,bhdr->bhr",
        strided_history[..., append_start, :].float()
        - value_mean.float(),
        value_basis.float(),
    )
    append_minimum = value_minimum[..., block, :].float()
    append_scale = value_scale[..., block, :].float()
    expected_append_codes = torch.round(
        (expected_append_coefficients - append_minimum) / append_scale
    ).clamp(0, 15).to(torch.uint8)
    expected_append_packed = (
        expected_append_codes[..., 0::2]
        | (expected_append_codes[..., 1::2] << 4)
    )
    append_code_error = int(
        (
            appended_codes[..., append_start, :].to(torch.int16)
            - expected_append_packed.to(torch.int16)
        )
        .abs()
        .max()
        .item()
    )
    if append_code_error:
        raise AssertionError(
            f"fused Value append code mismatch: {append_code_error}"
        )
    print(
        "QKSIEVE_VALUESKETCH_CUDA_VALIDATION_OK",
        {
            "history_tokens": history,
            "rank": rank,
            "candidate_count_mean": float(candidate_counts.float().mean()),
            "candidate_count_max_error": candidate_count_error,
            "tail_denominator_relative_error": denominator_relative,
            "tail_coefficient_relative_error": coefficient_relative,
            "output_max_abs_error": output_error,
            "alpha_fused_output_max_abs_error": alpha_output_error,
            "mass_only_threshold_max_abs_error": mass_threshold_error,
            "mass_only_count_max_error": mass_count_error,
            "mass_only_selected_denominator_relative_error": mass_selected_relative,
            "mass_only_tail_denominator_relative_error": mass_tail_relative,
            "mean_tail_output_max_abs_error": mean_tail_output_error,
            "append_code_max_error": append_code_error,
        },
    )


if __name__ == "__main__":
    main()
