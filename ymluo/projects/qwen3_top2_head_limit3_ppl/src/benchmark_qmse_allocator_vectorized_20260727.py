from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _hierarchical_qmse_rate_allocation_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--key-count", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--new-repeats", type=int, default=20)
    parser.add_argument("--old-repeats", type=int, default=3)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(
    function,
    arguments: tuple[object, ...],
    keyword_arguments: dict[str, object],
    repeats: int,
    device: torch.device,
) -> list[float]:
    measurements = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function(*arguments, **keyword_arguments)
        synchronize(device)
        measurements.append((time.perf_counter() - start) * 1000.0)
    return measurements


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(20260727)
    projected_sample = torch.randn(
        1,
        args.kv_heads,
        args.key_count,
        128,
        dtype=torch.float16,
        device=device,
    )
    projected_queries = torch.randn(
        1,
        args.kv_heads,
        args.query_count,
        128,
        dtype=torch.float16,
        device=device,
    )
    cases = [
        {
            "allow_zero_bits": True,
            "include_scale_metadata": True,
            "query_covariance_shrinkage": "none",
        },
        {
            "allow_zero_bits": True,
            "include_scale_metadata": True,
            "query_covariance_shrinkage": "oas",
        },
        {
            "allow_zero_bits": False,
            "include_scale_metadata": False,
            "query_covariance_shrinkage": "none",
        },
    ]
    comparisons = []
    for case in cases:
        budget = 15 if case["allow_zero_bits"] else 24
        reference = _hierarchical_qmse_rate_allocation_reference(
            projected_sample,
            projected_queries,
            budget,
            **case,
        )
        vectorized = _hierarchical_qmse_rate_allocation(
            projected_sample,
            projected_queries,
            budget,
            **case,
        )
        comparisons.append(
            {
                **case,
                "budget": budget,
                "equal": bool(torch.equal(reference, vectorized)),
                "mismatch_count": int((reference != vectorized).sum().item()),
                "reference_allocations": reference.cpu().tolist(),
                "vectorized_allocations": vectorized.cpu().tolist(),
            }
        )

    benchmark_case = cases[0]
    benchmark_arguments = (
        projected_sample,
        projected_queries,
        15,
    )
    _hierarchical_qmse_rate_allocation(
        *benchmark_arguments,
        **benchmark_case,
    )
    new_times = measure_ms(
        _hierarchical_qmse_rate_allocation,
        benchmark_arguments,
        benchmark_case,
        args.new_repeats,
        device,
    )
    old_times = measure_ms(
        _hierarchical_qmse_rate_allocation_reference,
        benchmark_arguments,
        benchmark_case,
        args.old_repeats,
        device,
    )
    output = {
        "device": str(device),
        "key_count": args.key_count,
        "query_count": args.query_count,
        "kv_heads": args.kv_heads,
        "comparisons": comparisons,
        "all_allocations_equal": all(item["equal"] for item in comparisons),
        "reference_median_ms": statistics.median(old_times),
        "vectorized_median_ms": statistics.median(new_times),
        "allocator_speedup": (
            statistics.median(old_times) / statistics.median(new_times)
        ),
        "reference_times_ms": old_times,
        "vectorized_times_ms": new_times,
    }
    print(json.dumps(output, indent=2))
    if not output["all_allocations_equal"]:
        raise RuntimeError("vectorized allocation changed a bit allocation")


if __name__ == "__main__":
    main()
