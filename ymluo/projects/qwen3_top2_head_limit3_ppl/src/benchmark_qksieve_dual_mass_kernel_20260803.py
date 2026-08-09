from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from qksieve_dual_mass_cuda_20260803 import (
    dual_mass_candidates,
    dual_mass_candidates_with_value_tail,
    dual_mass_candidates_with_value_tail_out,
)
from run_head_top2_targeted_ppl_20260714 import (
    _head_local_dual_mass_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", default="4096,32000,65536,131072")
    parser.add_argument("--query_heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--target_mass", type=float, default=0.975)
    parser.add_argument("--floor_fraction", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def timed_ms(callable_object, iterations: int) -> tuple[float, list[float]]:
    for _ in range(3):
        callable_object()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_object()
        stop.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return statistics.median(samples), samples


def selected_mask(
    indices: torch.Tensor,
    counts: torch.Tensor,
    history_count: int,
) -> torch.Tensor:
    valid = (
        torch.arange(indices.shape[-1], device=indices.device)[None, None, :]
        < counts.unsqueeze(-1)
    )
    mask = torch.zeros(
        *indices.shape[:2],
        history_count,
        dtype=torch.bool,
        device=indices.device,
    )
    safe_indices = torch.where(valid, indices, 0)
    mask.scatter_(2, safe_indices, valid)
    return mask


def main() -> None:
    args = parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    rows = []
    for history_count in [int(value) for value in args.lengths.split(",")]:
        floor_k = max(1, round(args.floor_fraction * history_count))
        # A shared salience term creates the concentrated long-context score
        # distribution observed in real attention while retaining head noise.
        salience = torch.randn(1, 1, history_count, device=device) * 2.5
        proxy = salience + torch.randn(
            1, args.query_heads, history_count, device=device
        ) * 1.25
        risk = torch.randn(
            1, args.kv_heads, history_count, device=device
        ) * 0.65
        slope = 0.75 + 0.5 * torch.rand(
            1, args.query_heads, device=device
        )
        intercept = torch.randn(1, args.query_heads, device=device) * 0.2
        calibrated = slope.unsqueeze(-1) * proxy + intercept.unsqueeze(-1)
        expanded_risk = risk.repeat_interleave(
            args.query_heads // args.kv_heads, dim=1
        )
        value_block_size = 256
        value_block_count = (
            history_count + value_block_size - 1
        ) // value_block_size
        packed_value_codes = torch.randint(
            0,
            256,
            (1, args.kv_heads, history_count, 8),
            dtype=torch.uint8,
            device=device,
        )
        value_minimum = torch.randn(
            1, args.kv_heads, value_block_count, 16, device=device
        ) * 0.15
        value_scale = 0.01 + 0.05 * torch.rand(
            1, args.kv_heads, value_block_count, 16, device=device
        )

        kernel_result = dual_mass_candidates(
            proxy,
            risk,
            slope,
            intercept,
            args.target_mass,
            floor_k,
        )
        kernel_indices, kernel_counts, _, _, overflow = kernel_result
        tail_result = dual_mass_candidates_with_value_tail(
            proxy,
            risk,
            slope,
            intercept,
            packed_value_codes,
            value_minimum,
            value_scale,
            args.target_mass,
            floor_k,
            value_block_size=value_block_size,
        )
        persistent_outputs = (
            torch.empty_like(tail_result[0]),
            torch.empty_like(tail_result[1]),
            torch.empty_like(tail_result[2]),
            torch.empty_like(tail_result[3]),
            torch.empty_like(tail_result[4]),
            torch.empty_like(tail_result[5]),
            torch.empty_like(tail_result[6]),
            torch.empty_like(tail_result[7]),
        )
        persistent_result = dual_mass_candidates_with_value_tail_out(
            proxy,
            risk,
            slope,
            intercept,
            packed_value_codes,
            value_minimum,
            value_scale,
            *persistent_outputs,
            args.target_mass,
            floor_k,
            value_block_size,
        )
        (
            tail_indices,
            tail_counts,
            _,
            _,
            tail_overflow,
            tail_anchor,
            tail_denominator,
            tail_coefficients,
        ) = tail_result
        reference_indices, reference_counts = (
            _head_local_dual_mass_candidates(
                calibrated,
                expanded_risk,
                floor_k,
                args.target_mass,
            )
        )
        kernel_mask = selected_mask(
            kernel_indices, kernel_counts, history_count
        )
        tail_mask = selected_mask(
            tail_indices, tail_counts, history_count
        )
        persistent_mask = selected_mask(
            persistent_result[0], persistent_result[1], history_count
        )
        reference_mask = selected_mask(
            reference_indices, reference_counts, history_count
        )
        attention_mass = (
            torch.softmax(calibrated, dim=-1) * kernel_mask
        ).sum(dim=-1)
        risk_mass = (
            torch.softmax(calibrated + expanded_risk, dim=-1) * kernel_mask
        ).sum(dim=-1)
        intersection = (kernel_mask & reference_mask).sum(dim=-1)
        union = (kernel_mask | reference_mask).sum(dim=-1).clamp_min(1)

        packed = packed_value_codes[0, 0]
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
            history_count, 16
        )
        block_ids = torch.arange(history_count, device=device) // 256
        decoded_coefficients = (
            codes.float() * value_scale[0, 0].index_select(0, block_ids)
            + value_minimum[0, 0].index_select(0, block_ids)
        )
        tail_weights = torch.exp(calibrated[0, 0] - tail_anchor[0, 0])
        tail_weights = tail_weights * (~tail_mask[0, 0])
        expected_denominator = tail_weights.sum()
        expected_coefficients = torch.einsum(
            "n,nr->r", tail_weights, decoded_coefficients
        )
        denominator_relative_error = float(
            (
                (tail_denominator[0, 0] - expected_denominator).abs()
                / expected_denominator.abs().clamp_min(1.0e-8)
            ).item()
        )
        coefficient_relative_error = float(
            (
                (tail_coefficients[0, 0] - expected_coefficients).norm()
                / expected_coefficients.norm().clamp_min(1.0e-8)
            ).item()
        )

        iterations = 30 if history_count <= 32_000 else 15
        kernel_median_ms, kernel_samples = timed_ms(
            lambda: dual_mass_candidates(
                proxy,
                risk,
                slope,
                intercept,
                args.target_mass,
                floor_k,
            ),
            iterations,
        )
        tail_kernel_median_ms, tail_kernel_samples = timed_ms(
            lambda: dual_mass_candidates_with_value_tail(
                proxy,
                risk,
                slope,
                intercept,
                packed_value_codes,
                value_minimum,
                value_scale,
                args.target_mass,
                floor_k,
                value_block_size=value_block_size,
            ),
            iterations,
        )
        persistent_tail_median_ms, persistent_tail_samples = timed_ms(
            lambda: dual_mass_candidates_with_value_tail_out(
                proxy,
                risk,
                slope,
                intercept,
                packed_value_codes,
                value_minimum,
                value_scale,
                *persistent_outputs,
                args.target_mass,
                floor_k,
                value_block_size,
            ),
            iterations,
        )
        reference_median_ms, reference_samples = timed_ms(
            lambda: _head_local_dual_mass_candidates(
                calibrated,
                expanded_risk,
                floor_k,
                args.target_mass,
            ),
            max(5, iterations // 3),
        )
        rows.append(
            {
                "history_count": history_count,
                "floor_k": floor_k,
                "target_mass": args.target_mass,
                "kernel_selected_ratio_mean": float(
                    kernel_counts.float().mean().item() / history_count
                ),
                "reference_selected_ratio_mean": float(
                    reference_counts.float().mean().item() / history_count
                ),
                "attention_mass_min": float(attention_mass.min().item()),
                "attention_mass_mean": float(attention_mass.mean().item()),
                "risk_mass_min": float(risk_mass.min().item()),
                "risk_mass_mean": float(risk_mass.mean().item()),
                "set_jaccard_mean": float(
                    (intersection.float() / union.float()).mean().item()
                ),
                "overflow_count": int(overflow.sum().item()),
                "tail_overflow_count": int(tail_overflow.sum().item()),
                "tail_denominator_relative_error": (
                    denominator_relative_error
                ),
                "tail_coefficient_relative_error": (
                    coefficient_relative_error
                ),
                "kernel_median_ms": kernel_median_ms,
                "kernel_p90_ms": float(
                    torch.tensor(kernel_samples).quantile(0.9).item()
                ),
                "reference_sort_median_ms": reference_median_ms,
                "reference_sort_p90_ms": float(
                    torch.tensor(reference_samples).quantile(0.9).item()
                ),
                "selection_speedup": reference_median_ms / kernel_median_ms,
                "tail_kernel_median_ms": tail_kernel_median_ms,
                "tail_kernel_p90_ms": float(
                    torch.tensor(tail_kernel_samples).quantile(0.9).item()
                ),
                "persistent_tail_kernel_median_ms": (
                    persistent_tail_median_ms
                ),
                "persistent_tail_kernel_p90_ms": float(
                    torch.tensor(persistent_tail_samples)
                    .quantile(0.9)
                    .item()
                ),
                "persistent_output_speedup": (
                    tail_kernel_median_ms / persistent_tail_median_ms
                ),
                "persistent_candidate_jaccard": float(
                    (
                        (persistent_mask & tail_mask).sum(dim=-1).float()
                        / (persistent_mask | tail_mask)
                        .sum(dim=-1)
                        .clamp_min(1)
                    ).mean().item()
                ),
                "persistent_tail_denominator_relative_error": float(
                    (
                        (persistent_result[6] - tail_result[6]).norm()
                        / tail_result[6].norm().clamp_min(1.0e-8)
                    ).item()
                ),
                "persistent_tail_coefficient_relative_error": float(
                    (
                        (persistent_result[7] - tail_result[7]).norm()
                        / tail_result[7].norm().clamp_min(1.0e-8)
                    ).item()
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    payload = {
        "schema": "qksieve_dual_mass_kernel_benchmark_v1",
        "device": torch.cuda.get_device_name(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
