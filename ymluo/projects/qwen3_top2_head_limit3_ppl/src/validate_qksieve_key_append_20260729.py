from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import qksieve_query_cuda_20260728 as qksieve_cuda
import variablebit_spectral_cuda_20260727 as variablebit_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def elapsed_ms(function, warmup: int, iterations: int) -> float:
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
    return float(start.elapsed_time(stop) / iterations)


def clear_index(index: dict) -> None:
    index["packed_codes"].zero_()
    index["key_scales"].zero_()
    index["indexed_count"] = 0


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260729)
    device = torch.device("cuda")
    dtype = torch.float16
    batch_count = 1
    kv_head_count = 8
    capacity = 64
    start = 17
    allocations = torch.tensor(
        [
            [8, 4, 2, 1, 0, 0, 0, 0],
            [4, 4, 4, 2, 1, 0, 0, 0],
            [8, 4, 4, 1, 1, 0, 0, 0],
            [4, 4, 2, 2, 2, 1, 0, 0],
            [8, 2, 2, 2, 1, 1, 0, 0],
            [4, 4, 4, 1, 1, 1, 0, 0],
            [8, 4, 2, 2, 1, 0, 0, 0],
            [4, 4, 2, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int8,
        device=device,
    ).unsqueeze(0)
    key_storage = torch.randn(
        batch_count,
        kv_head_count,
        capacity,
        128,
        dtype=dtype,
        device=device,
    )
    key = key_storage[..., start : start + 1, :]
    if key.is_contiguous():
        raise RuntimeError("validation Key must exercise a strided cache view")
    gaussian = torch.randn(
        batch_count,
        kv_head_count,
        128,
        128,
        dtype=torch.float32,
        device=device,
    )
    basis = torch.linalg.qr(gaussian).Q.to(dtype).contiguous()
    reference = variablebit_cuda.allocate_packed_index(
        allocations, capacity, dtype
    )
    fused = variablebit_cuda.allocate_packed_index(
        allocations, capacity, dtype
    )
    joint_reference = variablebit_cuda.allocate_packed_index(
        allocations, capacity, dtype
    )
    joint = variablebit_cuda.allocate_packed_index(
        allocations, capacity, dtype
    )
    clear_index(reference)
    clear_index(fused)
    clear_index(joint_reference)
    clear_index(joint)

    projected = torch.einsum("bhkd,bhdm->bhkm", key, basis)
    variablebit_cuda.encode_projected_keys_into(
        projected.contiguous(), reference, start
    )
    qksieve_cuda.project_encode_append(key, basis, fused, start)
    torch.cuda.synchronize()

    code_difference_count = int(
        (reference["packed_codes"] != fused["packed_codes"]).sum().item()
    )
    scale_max_abs_error = float(
        (
            reference["key_scales"].float()
            - fused["key_scales"].float()
        )
        .abs()
        .max()
        .item()
    )
    query = torch.randn(
        batch_count,
        kv_head_count,
        4,
        128,
        dtype=dtype,
        device=device,
    )
    query_codes, query_scales = variablebit_cuda.quantize_projected_query(
        query
    )

    def score(index: dict) -> torch.Tensor:
        return variablebit_cuda.scores(
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
            start + 1,
            score_bias=index["score_bias"],
        )

    reference_scores = score(reference)
    fused_scores = score(fused)
    appended_score_error = (
        reference_scores[..., start] - fused_scores[..., start]
    ).abs()
    raw_query = torch.randn(
        batch_count,
        kv_head_count,
        4,
        128,
        dtype=dtype,
        device=device,
    )
    separate_query_codes, separate_query_scales = (
        qksieve_cuda.project_quantize(raw_query, basis)
    )
    active_query_codes, active_query_scales = (
        qksieve_cuda.project_quantize_active(
            raw_query, basis, allocations
        )
    )
    qksieve_cuda.project_encode_append(
        key, basis, joint_reference, start
    )
    joint_query_codes, joint_query_scales = (
        qksieve_cuda.project_append_quantize(
            key, raw_query, basis, joint, start
        )
    )
    torch.cuda.synchronize()
    joint_code_difference_count = int(
        (
            joint_reference["packed_codes"]
            != joint["packed_codes"]
        )
        .sum()
        .item()
    )
    joint_scale_max_abs_error = float(
        (
            joint_reference["key_scales"].float()
            - joint["key_scales"].float()
        )
        .abs()
        .max()
        .item()
    )
    joint_query_code_difference_count = int(
        (separate_query_codes != joint_query_codes).sum().item()
    )
    joint_query_scale_max_abs_error = float(
        (
            separate_query_scales.float()
            - joint_query_scales.float()
        )
        .abs()
        .max()
        .item()
    )
    active_band_mask = (allocations > 0).unsqueeze(2)
    active_coordinate_mask = active_band_mask.repeat_interleave(
        16, dim=-1
    )
    active_query_code_difference_count = int(
        (
            (separate_query_codes != active_query_codes)
            & active_coordinate_mask
        )
        .sum()
        .item()
    )
    active_query_scale_max_abs_error = float(
        torch.where(
            active_band_mask,
            (
                separate_query_scales.float()
                - active_query_scales.float()
            ).abs(),
            torch.zeros_like(active_query_scales.float()),
        )
        .max()
        .item()
    )

    def scores_for_query(
        codes: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        return variablebit_cuda.scores(
            codes,
            scales,
            joint_reference["packed_codes"],
            joint_reference["key_scales"],
            joint_reference["bit_allocations"],
            joint_reference["code_offsets"],
            joint_reference["scale_offsets"],
            joint_reference["code_bases"],
            joint_reference["scale_bases"],
            joint_reference["code_strides"],
            joint_reference["scale_strides"],
            start + 1,
            score_bias=joint_reference["score_bias"],
        )

    active_query_score_max_abs_error = float(
        (
            scores_for_query(
                separate_query_codes, separate_query_scales
            )
            - scores_for_query(
                active_query_codes, active_query_scales
            )
        )
        .abs()
        .max()
        .item()
    )

    def baseline_append() -> None:
        projected_key = torch.einsum("bhkd,bhdm->bhkm", key, basis)
        variablebit_cuda.encode_projected_keys_into(
            projected_key.contiguous(), reference, start
        )

    def fused_append() -> None:
        qksieve_cuda.project_encode_append(key, basis, fused, start)

    def separate_prepare() -> None:
        qksieve_cuda.project_encode_append(
            key, basis, joint_reference, start
        )
        qksieve_cuda.project_quantize(raw_query, basis)

    def joint_prepare() -> None:
        qksieve_cuda.project_append_quantize(
            key, raw_query, basis, joint, start
        )

    def normal_query_prepare() -> None:
        qksieve_cuda.project_quantize(raw_query, basis)

    def active_query_prepare() -> None:
        qksieve_cuda.project_quantize_active(
            raw_query, basis, allocations
        )

    baseline_ms = elapsed_ms(
        baseline_append, args.warmup, args.iterations
    )
    fused_ms = elapsed_ms(fused_append, args.warmup, args.iterations)
    separate_prepare_ms = elapsed_ms(
        separate_prepare, args.warmup, args.iterations
    )
    joint_prepare_ms = elapsed_ms(
        joint_prepare, args.warmup, args.iterations
    )
    normal_query_ms = elapsed_ms(
        normal_query_prepare, args.warmup, args.iterations
    )
    active_query_ms = elapsed_ms(
        active_query_prepare, args.warmup, args.iterations
    )
    result = {
        "code_difference_count": code_difference_count,
        "total_code_bytes": int(reference["packed_codes"].numel()),
        "scale_max_abs_error": scale_max_abs_error,
        "appended_score_max_abs_error": float(
            appended_score_error.max().item()
        ),
        "appended_score_mean_abs_error": float(
            appended_score_error.mean().item()
        ),
        "baseline_project_encode_ms": baseline_ms,
        "fused_project_encode_ms": fused_ms,
        "speedup": baseline_ms / fused_ms,
        "joint_code_difference_count": joint_code_difference_count,
        "joint_scale_max_abs_error": joint_scale_max_abs_error,
        "joint_query_code_difference_count": (
            joint_query_code_difference_count
        ),
        "joint_query_scale_max_abs_error": (
            joint_query_scale_max_abs_error
        ),
        "separate_fused_prepare_ms": separate_prepare_ms,
        "joint_prepare_ms": joint_prepare_ms,
        "joint_prepare_speedup": separate_prepare_ms / joint_prepare_ms,
        "active_query_code_difference_count": (
            active_query_code_difference_count
        ),
        "active_query_scale_max_abs_error": (
            active_query_scale_max_abs_error
        ),
        "active_query_score_max_abs_error": (
            active_query_score_max_abs_error
        ),
        "normal_query_prepare_ms": normal_query_ms,
        "active_query_prepare_ms": active_query_ms,
        "active_query_prepare_speedup": (
            normal_query_ms / active_query_ms
        ),
    }
    print(json.dumps(result, sort_keys=True))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
