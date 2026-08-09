#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch

import mixedblock_spectral_cuda_20260729 as mixed_cuda
import qksieve_valuesketch_cuda_20260801 as sketch_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=4096)
    parser.add_argument("--moment_block_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = float((actual.float() - expected.float()).abs().max().item())
    denominator = max(1.0, float(expected.float().abs().max().item()))
    return numerator / denominator


def decode_first_eight(
    index: dict[str, object], history: int, bits: int
) -> torch.Tensor:
    decoded = torch.empty(
        1, 8, history, 8, dtype=torch.float32, device="cuda"
    )
    token_ids = torch.arange(history, dtype=torch.long, device="cuda")
    for head in range(8):
        code_base = int(index["code_bases"][0, head].item())
        code_stride = int(index["code_strides"][0, head].item())
        code_offset = int(index["code_offsets"][0, head, 0].item())
        scale_base = int(index["scale_bases"][0, head].item())
        scale_stride = int(index["scale_strides"][0, head].item())
        scale_offset = int(index["scale_offsets"][0, head, 0].item())
        byte_count = bits
        addresses = (
            code_base
            + token_ids[:, None] * code_stride
            + code_offset
            + torch.arange(byte_count, device="cuda")[None, :]
        )
        packed = index["packed_codes"].index_select(
            0, addresses.reshape(-1)
        ).reshape(history, byte_count)
        if bits == 8:
            codes = packed[:, :8].view(torch.int8).float()
        else:
            codes = torch.empty(
                history, 8, dtype=torch.float32, device="cuda"
            )
            low = (packed[:, :4] & 0x0F).to(torch.int16)
            high = (packed[:, :4] >> 4).to(torch.int16)
            low = torch.where(low < 8, low, low - 16)
            high = torch.where(high < 8, high, high - 16)
            codes[:, 0::2] = low.float()
            codes[:, 1::2] = high.float()
        scale_addresses = (
            scale_base + token_ids * scale_stride + scale_offset
        )
        scales = index["key_scales"].index_select(
            0, scale_addresses
        ).float()
        decoded[0, head] = codes * scales[:, None]
    return decoded


@torch.inference_mode()
def validate_one(history: int, block_size: int, bits: int) -> dict[str, float]:
    dtype = torch.bfloat16
    scaling = 128.0**-0.5
    block_count = math.ceil(history / block_size)
    selected_fraction = min(0.06, 1280.0 / history)
    sample_count = min(
        history,
        min(2048, max(256, math.ceil(16.0 / selected_fraction))),
    )
    allocation = torch.tensor(
        [[[bits, 0, 0, 0, 0, 0, 0, 0]] * 8],
        dtype=torch.int8,
        device="cuda",
    )
    projected_keys = torch.randn(
        1, 8, history, 128, dtype=dtype, device="cuda"
    )
    projected_query = torch.randn(
        1, 8, 4, 128, dtype=dtype, device="cuda"
    )
    query_codes, query_scales = varbit_cuda.quantize_projected_query(
        projected_query
    )
    index = varbit_cuda.allocate_packed_index(allocation, history, dtype)
    varbit_cuda.encode_projected_keys_into(projected_keys, index, 0)

    candidate_indices = torch.empty(
        1, 32, history, dtype=torch.long, device="cuda"
    )
    candidate_counts = torch.empty(
        1, 32, dtype=torch.long, device="cuda"
    )
    thresholds = torch.empty(1, 32, dtype=torch.float32, device="cuda")
    overflow = torch.empty(1, 32, dtype=torch.bool, device="cuda")
    selected_denominator = torch.empty_like(thresholds)
    tail_denominator = torch.empty_like(thresholds)
    tail_block_denominator = torch.empty(
        1, 32, block_count, dtype=torch.float32, device="cuda"
    )
    tail_weighted_x = torch.empty(
        1, 32, 8, dtype=torch.float32, device="cuda"
    )
    mixed_cuda.plain_sampled_threshold_compact_gqa4_condtail_out(
        query_codes,
        query_scales,
        index,
        candidate_indices,
        candidate_counts,
        thresholds,
        overflow,
        selected_denominator,
        tail_denominator,
        tail_block_denominator,
        tail_weighted_x,
        history,
        sample_count,
        selected_fraction,
        block_size,
        scaling,
    )
    torch.cuda.synchronize()
    if bool(overflow.any().item()):
        raise AssertionError("conditional-tail candidate output overflowed")

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
    keep = proxy_scores >= thresholds.unsqueeze(-1)
    expected_counts = keep.sum(dim=-1)
    count_error = int((candidate_counts - expected_counts).abs().max().item())
    if count_error:
        raise AssertionError(f"candidate count mismatch: {count_error}")
    relative_scores = (proxy_scores - thresholds.unsqueeze(-1)) * scaling
    expected_selected_weights = torch.exp(
        relative_scores.clamp(max=70.0)
    ).masked_fill(~keep, 0.0)
    expected_tail_weights = torch.exp(relative_scores).masked_fill(keep, 0.0)
    expected_selected_denominator = expected_selected_weights.sum(dim=-1)
    expected_tail_denominator = expected_tail_weights.sum(dim=-1)

    padded_tokens = block_count * block_size
    padded_tail_weights = torch.nn.functional.pad(
        expected_tail_weights, (0, padded_tokens - history)
    )
    expected_block_denominator = padded_tail_weights.reshape(
        1, 32, block_count, block_size
    ).sum(dim=-1)
    decoded_x = decode_first_eight(index, history, bits)
    query_head_x = decoded_x.repeat_interleave(4, dim=1)
    expected_weighted_x = torch.einsum(
        "bqn,bqnr->bqr", expected_tail_weights, query_head_x
    )

    errors = {
        "selected_denominator": relative_error(
            selected_denominator, expected_selected_denominator
        ),
        "tail_denominator": relative_error(
            tail_denominator, expected_tail_denominator
        ),
        "block_denominator": relative_error(
            tail_block_denominator, expected_block_denominator
        ),
        "weighted_x": relative_error(tail_weighted_x, expected_weighted_x),
    }
    if max(errors.values()) > 1.0e-2:
        raise AssertionError(f"conditional scan mismatch: {errors}")

    mean_x = torch.randn(
        1, 8, block_count, 8, dtype=dtype, device="cuda"
    ).mul_(0.1)
    mean_v = torch.randn(
        1, 8, block_count, 128, dtype=dtype, device="cuda"
    ).mul_(0.1)
    linear_map = torch.randn(
        1, 8, 128, 8, dtype=dtype, device="cuda"
    ).mul_(0.1)
    tail_numerator = sketch_cuda.reduce_conditional_tail_moments(
        tail_block_denominator,
        tail_weighted_x,
        mean_x,
        mean_v,
        linear_map,
    )
    query_mean_x = mean_x.repeat_interleave(4, dim=1).float()
    query_mean_v = mean_v.repeat_interleave(4, dim=1).float()
    query_map = linear_map.repeat_interleave(4, dim=1).float()
    expected_centered_x = expected_weighted_x - torch.einsum(
        "bqc,bqcr->bqr", expected_block_denominator, query_mean_x
    )
    expected_tail_numerator = torch.einsum(
        "bqc,bqcd->bqd", expected_block_denominator, query_mean_v
    ) + torch.einsum("bqr,bqdr->bqd", expected_centered_x, query_map)
    errors["tail_numerator"] = relative_error(
        tail_numerator, expected_tail_numerator
    )
    if errors["tail_numerator"] > 1.0e-2:
        raise AssertionError(f"conditional reducer mismatch: {errors}")

    query = torch.randn(1, 32, 128, dtype=dtype, device="cuda")
    key = torch.randn(1, 8, history + 1, 128, dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    output = sketch_cuda.exact_selected_plus_conditional_tail(
        query,
        key,
        value,
        candidate_indices,
        candidate_counts,
        thresholds,
        tail_denominator,
        tail_numerator,
        scaling,
    ).squeeze(1).float()
    reference = torch.empty_like(output)
    current = torch.tensor([history], dtype=torch.long, device="cuda")
    for row in range(32):
        kv_head = row // 4
        count = int(candidate_counts[0, row].item())
        indices = torch.cat((candidate_indices[0, row, :count], current))
        exact_key = key[0, kv_head].index_select(0, indices).float()
        exact_value = value[0, kv_head].index_select(0, indices).float()
        logits = exact_key @ query[0, row].float() * scaling
        maximum = max(
            float(logits.max().item()),
            float(thresholds[0, row].item()) * scaling,
        )
        exact_weights = torch.exp(logits - maximum)
        tail_factor = math.exp(
            float(thresholds[0, row].item()) * scaling - maximum
        )
        reference[0, row] = (
            exact_weights @ exact_value
            + tail_factor * tail_numerator[0, row]
        ) / (
            exact_weights.sum()
            + tail_factor * tail_denominator[0, row]
        )
    errors["attention_output_max_abs"] = float(
        (output - reference).abs().max().item()
    )
    if errors["attention_output_max_abs"] > 0.04:
        raise AssertionError(f"conditional attention mismatch: {errors}")

    # The shared GQA consumer must match the existing per-query-head consumer
    # exactly when all four query heads receive the same candidate set.
    shared_counts = candidate_counts[:, ::4].contiguous()
    shared_capacity = max(1, int(shared_counts.max().item()))
    shared_indices = candidate_indices[
        :, ::4, :shared_capacity
    ].contiguous()
    repeated_indices = shared_indices.repeat_interleave(4, dim=1)
    repeated_counts = shared_counts.repeat_interleave(4, dim=1)
    zero_denominator = torch.zeros_like(tail_denominator)
    zero_numerator = torch.zeros_like(tail_numerator)
    exact_per_head = sketch_cuda.exact_selected_plus_conditional_tail(
        query,
        key,
        value,
        repeated_indices,
        repeated_counts,
        thresholds,
        zero_denominator,
        zero_numerator,
        scaling,
    )
    exact_shared = sketch_cuda.exact_shared_gqa_selected_plus_conditional_tail(
        query,
        key,
        value,
        shared_indices,
        shared_counts,
        thresholds,
        zero_denominator,
        zero_numerator,
        scaling,
    )
    errors["shared_gqa_consumer_max_abs"] = float(
        (exact_shared.float() - exact_per_head.float()).abs().max().item()
    )
    if errors["shared_gqa_consumer_max_abs"] > 0.04:
        raise AssertionError(f"shared GQA consumer mismatch: {errors}")
    return errors


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 256:
        raise ValueError("history_tokens must be at least 256")
    if args.moment_block_size < 256 or args.moment_block_size % 256:
        raise ValueError("moment_block_size must be a multiple of 256")
    torch.manual_seed(args.seed)
    results = {
        bits: validate_one(
            args.history_tokens, args.moment_block_size, bits
        )
        for bits in (4, 8)
    }
    print(
        "QKSIEVE_CONDTAIL_CUDA_VALIDATION_OK",
        {
            "history_tokens": args.history_tokens,
            "moment_block_size": args.moment_block_size,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
