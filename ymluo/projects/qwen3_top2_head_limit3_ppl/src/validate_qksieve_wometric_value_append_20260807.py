#!/usr/bin/env python
"""Compare the fused W_o-metric Value append with the current torch path."""

from __future__ import annotations

import argparse
import json

import torch

import qksieve_valuesketch_cuda_20260801 as value_sketch_cuda


def packed_reference(
    values: torch.Tensor,
    mean: torch.Tensor,
    encoder: torch.Tensor,
    minimum: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    coefficients = torch.einsum(
        "bhnd,bhdr->bhnr",
        values.float() - mean.float().unsqueeze(2),
        encoder.float(),
    )
    codes = torch.round(
        (coefficients - minimum.float().unsqueeze(2))
        / scale.float().unsqueeze(2).clamp_min(1.0e-12)
    ).clamp(0, 15).to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


@torch.inference_mode()
def run_case(dtype: torch.dtype, seed: int, append_tokens: int) -> dict:
    torch.manual_seed(seed)
    batch_count = 1
    head_count = 8
    head_dim = 128
    rank = 16
    block_size = 256
    history_count = block_size + append_tokens
    capacity = 2 * block_size
    values = torch.randn(
        batch_count,
        head_count,
        history_count,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    mean = torch.randn(
        batch_count,
        head_count,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    encoder = (
        torch.randn(
            batch_count,
            head_count,
            head_dim,
            rank,
            dtype=dtype,
            device="cuda",
        )
        * 0.1
    )
    initial_coefficients = torch.einsum(
        "bhnd,bhdr->bhnr",
        values[..., :block_size, :].float() - mean.float().unsqueeze(2),
        encoder.float(),
    )
    minimum = initial_coefficients.amin(dim=2)
    maximum = initial_coefficients.amax(dim=2)
    scale = ((maximum - minimum) / 15.0).clamp_min(1.0e-12)
    metadata_minimum = torch.empty(
        batch_count,
        head_count,
        2,
        rank,
        dtype=dtype,
        device="cuda",
    )
    metadata_scale = torch.empty_like(metadata_minimum)
    metadata_minimum[:, :, 0] = minimum.to(dtype)
    metadata_scale[:, :, 0] = scale.to(dtype)
    metadata_minimum[:, :, 1] = metadata_minimum[:, :, 0]
    metadata_scale[:, :, 1] = metadata_scale[:, :, 0]
    packed = torch.empty(
        batch_count,
        head_count,
        capacity,
        rank // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    value_sketch_cuda.append_int4_out(
        values,
        mean,
        encoder,
        metadata_minimum,
        metadata_scale,
        packed,
        block_size,
        history_count,
        block_size,
    )
    torch.cuda.synchronize()
    reference = packed_reference(
        values[..., block_size:history_count, :],
        mean,
        encoder,
        metadata_minimum[:, :, 1],
        metadata_scale[:, :, 1],
    )
    candidate = packed[..., block_size:history_count, :]
    mismatches = int((candidate != reference).sum().item())
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "append_tokens": append_tokens,
        "packed_values": candidate.numel(),
        "packed_mismatches": mismatches,
        "packed_mismatch_rate": mismatches / candidate.numel(),
        "bitwise_equal": mismatches == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        run_case(dtype, seed, append_tokens)
        for dtype in (torch.float16, torch.bfloat16)
        for seed in (20260807, 20260808, 20260809)
        for append_tokens in (1, 7, 64)
    ]
    payload = {
        "schema": "qksieve_wometric_value_append_validation_v1",
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows),
        "total_mismatches": sum(row["packed_mismatches"] for row in rows),
        "total_packed_values": sum(row["packed_values"] for row in rows),
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
