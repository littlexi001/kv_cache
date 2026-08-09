from __future__ import annotations

import json

import torch

import variablebit_spectral_cuda_20260727 as variablebit


ALLOCATION = torch.tensor(
    [
        [8, 4, 0, 0, 0, 0, 0, 0],
        [4, 4, 2, 1, 0, 0, 0, 0],
    ],
    dtype=torch.int8,
).unsqueeze(0)


def unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    packed = packed.to(torch.int16)
    if bits == 8:
        return torch.where(packed >= 128, packed - 256, packed)
    if bits == 4:
        unsigned = torch.stack(
            (packed & 0xF, (packed >> 4) & 0xF),
            dim=-1,
        ).reshape(packed.shape[0], 16)
        return torch.where(unsigned >= 8, unsigned - 16, unsigned)
    if bits == 2:
        unsigned = torch.stack(
            tuple((packed >> shift) & 0x3 for shift in (0, 2, 4, 6)),
            dim=-1,
        ).reshape(packed.shape[0], 16)
        return torch.where(unsigned >= 2, unsigned - 4, unsigned)
    if bits == 1:
        return torch.stack(
            tuple(
                torch.where(
                    ((packed >> shift) & 1) > 0,
                    torch.ones_like(packed),
                    -torch.ones_like(packed),
                )
                for shift in range(8)
            ),
            dim=-1,
        ).reshape(packed.shape[0], 16)
    raise ValueError(bits)


def metric_scale_reference_from_packed(
    projected: torch.Tensor,
    metrics: torch.Tensor,
    packed_index: dict[str, object],
) -> torch.Tensor:
    packed_codes = packed_index["packed_codes"].cpu()
    expected = torch.zeros_like(packed_index["key_scales"].cpu())
    capacity = int(packed_index["capacity"])
    for head in range(projected.shape[1]):
        code_stride = int(packed_index["code_strides"][0, head].item())
        scale_stride = int(packed_index["scale_strides"][0, head].item())
        code_base = int(packed_index["code_bases"][0, head].item())
        scale_base = int(packed_index["scale_bases"][0, head].item())
        head_codes = packed_codes[
            code_base : code_base + capacity * code_stride
        ].reshape(capacity, code_stride)
        head_scales = expected[
            scale_base : scale_base + capacity * scale_stride
        ].reshape(capacity, scale_stride)
        for band in range(8):
            bits = int(ALLOCATION[0, head, band].item())
            if bits == 0:
                continue
            code_offset = int(
                packed_index["code_offsets"][0, head, band].item()
            )
            scale_offset = int(
                packed_index["scale_offsets"][0, head, band].item()
            )
            codes = unpack_codes(
                head_codes[
                    : projected.shape[-2],
                    code_offset : code_offset + 2 * bits,
                ],
                bits,
            ).float()
            values = projected[
                0,
                head,
                :,
                band * 16 : (band + 1) * 16,
            ].cpu().float()
            metric = metrics[0, head, band].cpu().float()
            weighted_codes = codes @ metric
            scales = (
                (weighted_codes * values).sum(dim=-1)
                / (
                    (weighted_codes * codes)
                    .sum(dim=-1)
                    .clamp_min(1.0e-12)
                )
            ).clamp_min(0.0)
            head_scales[
                : projected.shape[-2],
                scale_offset,
            ] = scales.to(head_scales.dtype)
    return expected


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260727)
    token_count = 257
    capacity = 320
    projected = torch.randn(
        1,
        2,
        token_count,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    queries = torch.randn(
        1,
        2,
        19,
        8,
        16,
        dtype=torch.float32,
        device="cuda",
    )
    metrics = torch.einsum(
        "bhqgd,bhqge->bhgde",
        queries,
        queries,
    ) / queries.shape[2]

    tested = variablebit.allocate_packed_index(
        ALLOCATION.cuda(),
        capacity,
        torch.float16,
    )
    tested["packed_codes"].zero_()
    tested["key_scales"].zero_()
    variablebit.encode_projected_keys_into(
        projected,
        tested,
        0,
        scale_metrics=metrics,
    )
    optimized = variablebit.allocate_packed_index(
        ALLOCATION.cuda(),
        capacity,
        torch.float16,
    )
    optimized["packed_codes"].zero_()
    optimized["key_scales"].zero_()
    precomputed_scales = (
        variablebit.metric_optimal_projected_key_scales(
            projected,
            ALLOCATION.cuda(),
            metrics,
        )
    )
    variablebit.encode_projected_keys_into(
        projected,
        optimized,
        0,
        precomputed_scales=precomputed_scales,
    )
    baseline = variablebit.allocate_packed_index(
        ALLOCATION.cuda(),
        capacity,
        torch.float16,
    )
    baseline["packed_codes"].zero_()
    baseline["key_scales"].zero_()
    variablebit.encode_projected_keys_into(
        projected,
        baseline,
        0,
    )

    expected = variablebit.allocate_packed_index(
        ALLOCATION,
        capacity,
        torch.float16,
    )
    expected["packed_codes"].zero_()
    expected["key_scales"].zero_()
    variablebit.encode_projected_keys_into(
        projected.cpu(),
        expected,
        0,
        scale_metrics=metrics.cpu(),
    )

    tested_codes = tested["packed_codes"].cpu()
    expected_codes = expected["packed_codes"]
    baseline_codes = baseline["packed_codes"].cpu()
    mismatch = torch.nonzero(tested_codes != expected_codes).flatten()
    code_equal = mismatch.numel() == 0
    packed_scale_reference = metric_scale_reference_from_packed(
        projected,
        metrics,
        tested,
    )
    scale_error = float(
        (
            tested["key_scales"].cpu().float()
            - packed_scale_reference.float()
        )
        .abs()
        .max()
        .item()
    )
    result = {
        "code_equal": code_equal,
        "metric_vs_baseline_code_equal": torch.equal(
            tested_codes,
            baseline_codes,
        ),
        "precomputed_vs_direct_code_equal": torch.equal(
            optimized["packed_codes"].cpu(),
            tested_codes,
        ),
        "precomputed_vs_direct_scale_max_abs_error": float(
            (
                optimized["key_scales"].float()
                - tested["key_scales"].float()
            )
            .abs()
            .max()
            .item()
        ),
        "baseline_vs_cpu_code_equal": torch.equal(
            baseline_codes,
            expected_codes,
        ),
        "code_mismatch_count": int(mismatch.numel()),
        "first_mismatch_indices": mismatch[:8].tolist(),
        "first_tested_bytes": tested_codes[mismatch[:8]].tolist(),
        "first_expected_bytes": expected_codes[mismatch[:8]].tolist(),
        "scale_max_abs_error": scale_error,
        "extension_version": "v13",
    }
    print(json.dumps(result, indent=2))
    if (
        not result["metric_vs_baseline_code_equal"]
        or not result["precomputed_vs_direct_code_equal"]
        or result["precomputed_vs_direct_scale_max_abs_error"] > 2.0e-3
        or scale_error > 2.0e-3
    ):
        raise AssertionError(result)


if __name__ == "__main__":
    main()
