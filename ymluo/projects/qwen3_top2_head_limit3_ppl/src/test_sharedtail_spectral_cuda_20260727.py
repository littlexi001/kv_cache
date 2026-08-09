from __future__ import annotations

import argparse
import json

import torch

import variablebit_spectral_cuda_20260727 as spectral_cuda


def unpack_int4(values: torch.Tensor) -> torch.Tensor:
    low = (values & 0x0F).to(torch.int16)
    high = (values >> 4).to(torch.int16)
    low = torch.where(low < 8, low, low - 16)
    high = torch.where(high < 8, high, high - 16)
    return torch.stack((low, high), dim=-1).reshape(*values.shape[:-1], -1)


def unpack_signs(values: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, device=values.device, dtype=torch.uint8)
    bits = ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return (2.0 * bits - 1.0).reshape(*values.shape[:-1], -1)


def reference_scores(
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict,
    history_count: int,
) -> torch.Tensor:
    key_codes = packed_index["packed_codes"][..., :history_count, :]
    key_scales = packed_index["key_scales"][..., :history_count, :].float()
    first = unpack_int4(key_codes[..., :8]).float()
    second = unpack_int4(key_codes[..., 8:16]).float()
    tail = unpack_signs(key_codes[..., 16:24])
    query_codes_float = query_codes.float()
    query_scales_float = query_scales.float()
    output = []
    batch_count, kv_head_count, group_count, _ = query_codes.shape
    for batch_index in range(batch_count):
        heads = []
        for kv_head in range(kv_head_count):
            for group in range(group_count):
                core0 = (
                    first[batch_index, kv_head]
                    @ query_codes_float[batch_index, kv_head, group, :16]
                )
                core0 *= (
                    key_scales[batch_index, kv_head, :, 0]
                    * query_scales_float[batch_index, kv_head, group, 0]
                )
                core1 = (
                    second[batch_index, kv_head]
                    @ query_codes_float[
                        batch_index, kv_head, group, 16:32
                    ]
                )
                core1 *= (
                    key_scales[batch_index, kv_head, :, 1]
                    * query_scales_float[batch_index, kv_head, group, 1]
                )
                tail_score = torch.zeros_like(core0)
                for band in range(4):
                    start = 16 * band
                    stop = start + 16
                    tail_score += (
                        tail[batch_index, kv_head, :, start:stop]
                        @ query_codes_float[
                            batch_index,
                            kv_head,
                            group,
                            32 + start : 32 + stop,
                        ]
                    ) * query_scales_float[
                        batch_index, kv_head, group, 2 + band
                    ]
                tail_score *= key_scales[batch_index, kv_head, :, 2]
                heads.append(core0 + core1 + tail_score)
        output.append(torch.stack(heads))
    return torch.stack(output)


def reference_query_quantization(
    projected_query: torch.Tensor,
    coordinate_amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = projected_query.float().reshape(
        *projected_query.shape[:-1], 8, 16
    )
    scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127)
    reconstructed_tail = (
        codes[..., 2:6, :] * scales[..., 2:6].unsqueeze(-1)
    )
    amplitudes = coordinate_amplitude[..., 32:96].reshape(
        *coordinate_amplitude.shape[:-1], 4, 16
    )
    weighted = reconstructed_tail * amplitudes.unsqueeze(-3)
    weighted_scales = weighted.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    weighted_codes = torch.round(
        weighted / weighted_scales.unsqueeze(-1)
    ).clamp(-127, 127)
    return (
        torch.cat(
            (
                codes[..., :2, :].reshape(*projected_query.shape[:-1], 32),
                weighted_codes.reshape(*projected_query.shape[:-1], 64),
            ),
            dim=-1,
        ).to(torch.int8),
        torch.cat((scales[..., :2], weighted_scales), dim=-1).to(
            projected_query.dtype
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this correctness test requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    batch_count = 1
    kv_head_count = 2
    query_groups = 4
    keys = torch.randn(
        batch_count,
        kv_head_count,
        args.tokens,
        128,
        device=device,
        dtype=torch.float16,
    )
    queries = torch.randn(
        batch_count,
        kv_head_count,
        query_groups,
        128,
        device=device,
        dtype=torch.float16,
    )
    coordinate_rms, coordinate_amplitude = (
        spectral_cuda.sharedtail_calibration_parameters(keys)
    )
    packed_index = spectral_cuda.allocate_sharedtail_index(
        batch_count,
        kv_head_count,
        args.tokens + 17,
        keys.dtype,
        device,
        coordinate_rms,
        coordinate_amplitude,
    )
    spectral_cuda.encode_sharedtail_projected_keys_into(
        keys,
        packed_index,
        0,
    )
    query_codes, query_scales = (
        spectral_cuda.quantize_sharedtail_projected_query(
            queries,
            packed_index["coordinate_amplitude"],
        )
    )
    standard_query_codes, standard_query_scales = (
        spectral_cuda.quantize_projected_query(queries)
    )
    grouped_query = queries.float().reshape(
        batch_count,
        kv_head_count,
        query_groups,
        8,
        16,
    )
    standard_reference_scales = (
        grouped_query.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    )
    standard_reference_codes = torch.round(
        grouped_query / standard_reference_scales.unsqueeze(-1)
    ).clamp(-127, 127).to(torch.int8).reshape_as(queries)
    if not torch.equal(standard_query_codes, standard_reference_codes):
        raise AssertionError("fused variable-bit query codes are inconsistent")
    standard_query_scale_error = float(
        (
            standard_query_scales
            - standard_reference_scales.to(queries.dtype)
        ).abs().max().item()
    )
    if standard_query_scale_error > 1.0e-6:
        raise AssertionError(
            "variable-bit query scale mismatch: "
            f"{standard_query_scale_error}"
        )
    reference_query_codes, reference_query_scales = (
        reference_query_quantization(
            queries,
            packed_index["coordinate_amplitude"],
        )
    )
    if not torch.equal(query_codes, reference_query_codes):
        raise AssertionError("fused shared-tail query codes are inconsistent")
    query_scale_error = float(
        (query_scales - reference_query_scales).abs().max().item()
    )
    if query_scale_error > 1.0e-6:
        raise AssertionError(
            f"shared-tail query scale mismatch: {query_scale_error}"
        )
    observed = spectral_cuda.sharedtail_scores(
        query_codes,
        query_scales,
        packed_index,
        args.tokens,
    )
    expected = reference_scores(
        query_codes,
        query_scales,
        packed_index,
        args.tokens,
    )
    torch.cuda.synchronize()
    absolute = (observed - expected).abs()
    max_absolute_error = float(absolute.max().item())
    mean_absolute_error = float(absolute.mean().item())
    if max_absolute_error > 2.0e-3:
        raise AssertionError(
            f"shared-tail score mismatch: {max_absolute_error}"
        )

    candidate_capacity = min(args.tokens, 96)
    row_shape = (batch_count, kv_head_count * query_groups)
    candidate_indices = torch.empty(
        *row_shape,
        candidate_capacity,
        dtype=torch.long,
        device=device,
    )
    candidate_scores = torch.empty(
        *row_shape,
        candidate_capacity,
        dtype=torch.float32,
        device=device,
    )
    candidate_counts = torch.empty(
        row_shape,
        dtype=torch.long,
        device=device,
    )
    thresholds = torch.empty(
        row_shape,
        dtype=torch.float32,
        device=device,
    )
    overflow = torch.empty(
        row_shape,
        dtype=torch.bool,
        device=device,
    )
    sample_count = min(256, args.tokens)
    spectral_cuda.sharedtail_sampled_threshold_compact_out(
        query_codes,
        query_scales,
        packed_index,
        candidate_indices,
        candidate_scores,
        candidate_counts,
        thresholds,
        overflow,
        args.tokens,
        sample_count,
        0.06,
    )
    torch.cuda.synchronize()
    expected_counts = (
        observed >= thresholds.unsqueeze(-1)
    ).sum(dim=-1)
    if bool((expected_counts > candidate_capacity).any().item()):
        if not bool(overflow.any().item()):
            raise AssertionError("shared-tail compaction failed to mark overflow")
    else:
        if bool(overflow.any().item()):
            raise AssertionError("unexpected shared-tail compaction overflow")
        if not torch.equal(candidate_counts, expected_counts):
            raise AssertionError("shared-tail candidate counts are inconsistent")
        for row in range(candidate_counts.numel()):
            batch_index = row // row_shape[1]
            head_index = row % row_shape[1]
            count = int(candidate_counts[batch_index, head_index].item())
            indices = candidate_indices[batch_index, head_index, :count]
            scores = candidate_scores[batch_index, head_index, :count]
            reference = observed[batch_index, head_index, indices]
            if float((scores - reference).abs().max().item()) > 2.0e-3:
                raise AssertionError("compacted scores differ from full scan")

    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "index_bits_per_head_token": 240,
                "index_ratio_of_full_fp16_kv": 240 / 4096,
                "max_absolute_error": max_absolute_error,
                "mean_absolute_error": mean_absolute_error,
                "query_scale_max_absolute_error": query_scale_error,
                "standard_query_scale_max_absolute_error": (
                    standard_query_scale_error
                ),
                "candidate_count_min": int(candidate_counts.min().item()),
                "candidate_count_max": int(candidate_counts.max().item()),
                "overflow": bool(overflow.any().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
