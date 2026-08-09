from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import variablebit_spectral_cuda_20260727 as varbit_cuda
from benchmark_variablebit_spectral_attention_20260727 import (
    ALLOCATION_PROFILES,
    make_metadata,
    measure_ms,
    quantize_query,
    selected_fraction,
)


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def adaptive_sample_count(
    selected_fraction_target: float,
    minimum: int,
    maximum: int,
    tail_samples: int,
) -> int:
    return min(
        maximum,
        max(
            minimum,
            math.ceil(tail_samples / selected_fraction_target),
        ),
    )


def statistics(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=True).item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure candidate-fraction variance of fixed and "
            "binomial-variance-controlled sampled quantiles."
        )
    )
    parser.add_argument(
        "--lengths", default="8192,32768,65536,131072"
    )
    parser.add_argument(
        "--allocation_profile",
        choices=sorted(ALLOCATION_PROFILES),
        default="qmse_total_b15",
    )
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--minimum_sample_count", type=int, default=256)
    parser.add_argument("--maximum_sample_count", type=int, default=2048)
    parser.add_argument("--target_tail_samples", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260727)
    rows = []
    allocations = ALLOCATION_PROFILES[
        args.allocation_profile
    ].unsqueeze(0)
    for history_tokens in parse_ints(args.lengths):
        target = selected_fraction(history_tokens)
        sample_counts = sorted(
            {
                args.minimum_sample_count,
                adaptive_sample_count(
                    target,
                    args.minimum_sample_count,
                    args.maximum_sample_count,
                    args.target_tail_samples,
                ),
                args.maximum_sample_count,
            }
        )
        metadata = make_metadata(allocations, history_tokens)
        packed_codes = torch.randint(
            0,
            256,
            (int(metadata["total_code_bytes"]),),
            dtype=torch.uint8,
            device="cuda",
        )
        key_scales = (
            torch.rand(
                int(metadata["total_scale_values"]),
                dtype=torch.float16,
                device="cuda",
            )
            * 0.03
            + 0.001
        )
        capacity = max(
            1,
            math.ceil(
                max(0.20, 4.0 * target) * history_tokens
            ),
        )
        candidate_indices = torch.empty(
            1, 32, capacity, dtype=torch.long, device="cuda"
        )
        candidate_proxy_scores = torch.empty(
            1, 32, capacity, dtype=torch.float32, device="cuda"
        )
        candidate_counts = torch.empty(
            1, 32, dtype=torch.long, device="cuda"
        )
        candidate_thresholds = torch.empty(
            1, 32, dtype=torch.float32, device="cuda"
        )
        candidate_overflow = torch.empty(
            1, 32, dtype=torch.bool, device="cuda"
        )
        for sample_count in sample_counts:
            trial_fractions = []
            timing_query_codes = None
            timing_query_scales = None
            for _ in range(args.trials):
                query = torch.randn(
                    1, 8, 4, 128, dtype=torch.float16, device="cuda"
                )
                query_codes, query_scales = quantize_query(query)
                varbit_cuda.sampled_threshold_compact_out(
                    query_codes,
                    query_scales,
                    packed_codes,
                    key_scales,
                    metadata["bit_allocations"],
                    metadata["code_offsets"],
                    metadata["scale_offsets"],
                    metadata["code_bases"],
                    metadata["scale_bases"],
                    metadata["code_strides"],
                    metadata["scale_strides"],
                    candidate_indices,
                    candidate_proxy_scores,
                    candidate_counts,
                    candidate_thresholds,
                    candidate_overflow,
                    history_tokens,
                    sample_count,
                    target,
                )
                if bool(candidate_overflow.any().item()):
                    raise RuntimeError("candidate buffer overflow")
                trial_fractions.append(
                    candidate_counts.float().cpu() / history_tokens
                )
                timing_query_codes = query_codes
                timing_query_scales = query_scales
            fraction_tensor = torch.stack(trial_fractions)

            def retrieve() -> tuple[torch.Tensor, ...]:
                return varbit_cuda.sampled_threshold_compact_out(
                    timing_query_codes,
                    timing_query_scales,
                    packed_codes,
                    key_scales,
                    metadata["bit_allocations"],
                    metadata["code_offsets"],
                    metadata["scale_offsets"],
                    metadata["code_bases"],
                    metadata["scale_bases"],
                    metadata["code_strides"],
                    metadata["scale_strides"],
                    candidate_indices,
                    candidate_proxy_scores,
                    candidate_counts,
                    candidate_thresholds,
                    candidate_overflow,
                    history_tokens,
                    sample_count,
                    target,
                )

            retrieval_ms = measure_ms(
                retrieve, args.warmup, args.iterations
            )
            row = {
                "history_tokens": history_tokens,
                "selected_fraction_target": target,
                "sample_count": sample_count,
                "expected_tail_samples": sample_count * target,
                "retrieval_ms": retrieval_ms,
                **{
                    f"selected_fraction_{name}": value
                    for name, value in statistics(fraction_tensor).items()
                },
                "mean_absolute_fraction_error": float(
                    (fraction_tensor - target).abs().mean().item()
                ),
                "relative_mean_absolute_error": float(
                    (fraction_tensor - target).abs().mean().item()
                    / target
                ),
                "under_target_rate": float(
                    (fraction_tensor < target).float().mean().item()
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del (
            metadata,
            packed_codes,
            key_scales,
            candidate_indices,
            candidate_proxy_scores,
            candidate_counts,
            candidate_thresholds,
            candidate_overflow,
        )
        torch.cuda.empty_cache()
    output = {
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
        },
        "derivation": (
            "For a target upper-tail probability p and S uniform samples, "
            "the tail count has coefficient of variation approximately "
            "1/sqrt(S*p). Requiring at least target_tail_samples expected "
            "tail observations chooses S >= target_tail_samples/p."
        ),
        "rows": rows,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
