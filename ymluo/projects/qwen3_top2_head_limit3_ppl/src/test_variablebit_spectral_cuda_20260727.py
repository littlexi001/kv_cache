from __future__ import annotations

import json

import torch

import variablebit_spectral_cuda_20260727 as variablebit


ALLOCATION = torch.tensor(
    [
        [8, 4, 0, 0, 0, 0, 0, 0],
        [8, 1, 1, 1, 0, 0, 0, 0],
        [4, 4, 4, 0, 0, 0, 0, 0],
        [8, 1, 1, 1, 0, 0, 0, 0],
        [4, 4, 4, 0, 0, 0, 0, 0],
        [8, 4, 0, 0, 0, 0, 0, 0],
        [8, 1, 1, 1, 0, 0, 0, 0],
        [4, 2, 2, 1, 1, 0, 0, 0],
    ],
    dtype=torch.int8,
)


def encode_torch_reference(
    projected: torch.Tensor,
    allocation: torch.Tensor,
    capacity: int,
) -> dict[str, object]:
    reference = variablebit.allocate_packed_index(
        allocation,
        capacity,
        projected.dtype,
    )
    reference["packed_codes"].zero_()
    reference["key_scales"].zero_()
    for head_index in range(projected.shape[1]):
        code_stride = int(reference["code_strides"][0, head_index].item())
        scale_stride = int(reference["scale_strides"][0, head_index].item())
        code_base = int(reference["code_bases"][0, head_index].item())
        scale_base = int(reference["scale_bases"][0, head_index].item())
        head_codes = reference["packed_codes"][
            code_base : code_base + capacity * code_stride
        ].reshape(capacity, code_stride)
        head_scales = reference["key_scales"][
            scale_base : scale_base + capacity * scale_stride
        ].reshape(capacity, scale_stride)
        for band_index in range(8):
            bits = int(allocation[0, head_index, band_index].item())
            if bits == 0:
                continue
            band = projected[
                0,
                head_index,
                :,
                16 * band_index : 16 * (band_index + 1),
            ].float()
            if bits == 1:
                scales = band.abs().mean(dim=-1).clamp_min(1.0e-8)
                codes = torch.where(
                    band >= 0.0,
                    torch.ones_like(band),
                    -torch.ones_like(band),
                ).to(torch.int8)
            else:
                maximum = (1 << (bits - 1)) - 1
                scales = (
                    band.abs().amax(dim=-1).clamp_min(1.0e-8)
                    / float(maximum)
                )
                codes = torch.round(
                    band / scales.unsqueeze(-1)
                ).clamp(-maximum, maximum).to(torch.int8)
            packed = variablebit._pack_band_codes(codes, bits)
            code_offset = int(
                reference["code_offsets"][0, head_index, band_index].item()
            )
            scale_offset = int(
                reference["scale_offsets"][0, head_index, band_index].item()
            )
            head_codes[
                : projected.shape[-2],
                code_offset : code_offset + 2 * bits,
            ].copy_(packed)
            head_scales[: projected.shape[-2], scale_offset].copy_(
                scales.to(head_scales.dtype)
            )
    return reference


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260727)
    token_count = 257
    capacity = 320
    allocation_cuda = ALLOCATION.unsqueeze(0).cuda()
    projected_cuda = torch.randn(
        1,
        8,
        token_count,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    tested = variablebit.allocate_packed_index(
        allocation_cuda,
        capacity,
        torch.float16,
    )
    tested["packed_codes"].zero_()
    tested["key_scales"].zero_()
    variablebit.encode_projected_keys_into(projected_cuda, tested, 0)

    expected = encode_torch_reference(
        projected_cuda,
        allocation_cuda,
        capacity,
    )
    tested_codes_cpu = tested["packed_codes"].cpu()
    expected_codes_cpu = expected["packed_codes"].cpu()
    code_difference = tested_codes_cpu != expected_codes_cpu
    code_equal = not bool(code_difference.any().item())
    mismatch_indices = torch.nonzero(code_difference).flatten()
    scale_error = float(
        (
            tested["key_scales"].float()
            - expected["key_scales"].float()
        )
        .abs()
        .max()
        .item()
    )

    projected_query = torch.randn(
        1,
        8,
        4,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    query_codes, query_scales = variablebit.quantize_projected_query(
        projected_query
    )
    scores = variablebit.scores(
        query_codes,
        query_scales,
        tested["packed_codes"],
        tested["key_scales"],
        tested["bit_allocations"],
        tested["code_offsets"],
        tested["scale_offsets"],
        tested["code_bases"],
        tested["scale_bases"],
        tested["code_strides"],
        tested["scale_strides"],
        token_count,
    )
    expected_scores = variablebit.scores(
        query_codes,
        query_scales,
        expected["packed_codes"],
        expected["key_scales"],
        tested["bit_allocations"],
        tested["code_offsets"],
        tested["scale_offsets"],
        tested["code_bases"],
        tested["scale_bases"],
        tested["code_strides"],
        tested["scale_strides"],
        token_count,
    )
    score_error = float((scores - expected_scores).abs().max().item())
    candidates = variablebit.sampled_threshold_compact(
        query_codes,
        query_scales,
        tested["packed_codes"],
        tested["key_scales"],
        tested["bit_allocations"],
        tested["code_offsets"],
        tested["scale_offsets"],
        tested["code_bases"],
        tested["scale_bases"],
        tested["code_strides"],
        tested["scale_strides"],
        token_count,
        256,
        0.06,
        64,
    )
    candidate_counts = candidates[2]
    output = {
        "packed_code_equal": code_equal,
        "packed_code_mismatch_count": int(mismatch_indices.numel()),
        "first_mismatch_indices": mismatch_indices[:8].tolist(),
        "first_tested_bytes": (
            tested_codes_cpu[mismatch_indices[:8]].tolist()
            if mismatch_indices.numel()
            else []
        ),
        "first_expected_bytes": (
            expected_codes_cpu[mismatch_indices[:8]].tolist()
            if mismatch_indices.numel()
            else []
        ),
        "maximum_scale_error": scale_error,
        "score_shape": list(scores.shape),
        "score_finite": bool(torch.isfinite(scores).all().item()),
        "maximum_score_error": score_error,
        "candidate_count_min": int(candidate_counts.min().item()),
        "candidate_count_max": int(candidate_counts.max().item()),
        "candidate_overflow_count": int(candidates[4].sum().item()),
    }
    print(json.dumps(output, indent=2))
    if scale_error > 2.0e-4 or score_error > 0.05:
        raise RuntimeError(f"packed encoder mismatch: {output}")


if __name__ == "__main__":
    main()
