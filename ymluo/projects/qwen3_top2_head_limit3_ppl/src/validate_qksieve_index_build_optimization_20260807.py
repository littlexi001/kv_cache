#!/usr/bin/env python3
"""Validate quality-preserving QKSieve request-index build optimizations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

import run_head_top2_targeted_ppl_20260714 as optimized
import variablebit_spectral_cuda_20260727 as variablebit_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_module", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=32768)
    parser.add_argument("--prefill_query_tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("qksieve_legacy_index_build", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def elapsed_seconds(function: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    left_float = left.float()
    right_float = right.float()
    max_abs_error = (
        float((left_float - right_float).abs().max().item())
        if shape_equal and left.numel() > 0
        else (0.0 if shape_equal else float("inf"))
    )
    return {
        "shape_equal": shape_equal,
        "dtype_equal": left.dtype == right.dtype,
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs_error": max_abs_error,
    }


def state_config(prefill_queries: torch.Tensor) -> dict[str, Any]:
    return {
        "packed_qmse_transform": "qk_metric",
        "packed_qmse_allocation_objective": "qmse",
        "packed_qmse_bit_budget": 15,
        "packed_qmse_covariance_shrinkage": "oas",
        "packed_qmse_prefill_queries": prefill_queries,
    }


def compare_packed_indices(
    left: dict[str, Any],
    right: dict[str, Any],
    history_count: int,
) -> dict[str, Any]:
    def active_payload(index: dict[str, Any], name: str) -> torch.Tensor:
        if name == "score_bias":
            return index[name][..., :history_count]
        prefix = "code" if name == "packed_codes" else "scale"
        values = index[name]
        pieces = []
        bases = index[f"{prefix}_bases"]
        strides = index[f"{prefix}_strides"]
        for batch_index in range(int(bases.shape[0])):
            for head_index in range(int(bases.shape[1])):
                base = int(bases[batch_index, head_index].item())
                stride = int(strides[batch_index, head_index].item())
                pieces.append(values[base : base + history_count * stride])
        return torch.cat(pieces) if pieces else values[:0]

    output: dict[str, Any] = {}
    for name in sorted(set(left) | set(right)):
        left_value = left.get(name)
        right_value = right.get(name)
        if isinstance(left_value, torch.Tensor) and isinstance(
            right_value, torch.Tensor
        ):
            if name in {"packed_codes", "key_scales", "score_bias"}:
                left_value = active_payload(left, name)
                right_value = active_payload(right, name)
            output[name] = tensor_comparison(left_value, right_value)
        elif isinstance(left_value, (int, float, str, bool, type(None))) and isinstance(
            right_value, (int, float, str, bool, type(None))
        ):
            output[name] = {"equal": left_value == right_value}
        else:
            output[name] = {
                "comparable": False,
                "left_type": type(left_value).__name__,
                "right_type": type(right_value).__name__,
            }
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    legacy = load_module(args.legacy_module)
    torch.manual_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    key = torch.randn(
        1,
        8,
        args.history_tokens,
        128,
        dtype=dtype,
        device="cuda",
    )
    prefill_queries = torch.randn(
        1,
        32,
        args.prefill_query_tokens,
        128,
        dtype=dtype,
        device="cuda",
    )
    sampled_key = key[..., ::32, :].float().contiguous()
    variablebit_cuda.load_extension()
    warm_key = key[..., :1024, :]
    warm_queries = prefill_queries.clone()
    legacy._packed_qmse_initialize(
        warm_key,
        state_config(warm_queries.clone()),
    )
    optimized._packed_qmse_initialize(
        warm_key,
        state_config(warm_queries.clone()),
    )
    optimized._hierarchical_key_rate_allocation(
        sampled_key,
        15,
        allow_zero_bits=True,
        include_scale_metadata=True,
    )
    torch.cuda.synchronize()

    allocator_rows = []
    for budget in (10, 12, 15, 18, 26):
        for allow_zero in (False, True):
            minimum_physical_bits = 8 * (
                1 + (1 if not allow_zero else 0)
            )
            if not allow_zero and budget < minimum_physical_bits:
                continue
            reference, reference_seconds = elapsed_seconds(
                lambda: optimized._hierarchical_key_rate_allocation_reference(
                    sampled_key,
                    budget,
                    allow_zero_bits=allow_zero,
                    include_scale_metadata=True,
                )
            )
            vectorized, vectorized_seconds = elapsed_seconds(
                lambda: optimized._hierarchical_key_rate_allocation(
                    sampled_key,
                    budget,
                    allow_zero_bits=allow_zero,
                    include_scale_metadata=True,
                )
            )
            allocator_rows.append(
                {
                    "budget": budget,
                    "allow_zero_bits": allow_zero,
                    "reference_seconds": reference_seconds,
                    "vectorized_seconds": vectorized_seconds,
                    "speedup": reference_seconds / vectorized_seconds,
                    **tensor_comparison(reference, vectorized),
                }
            )

    key_second_moment = torch.einsum(
        "bhkd,bhke->bhde", sampled_key, sampled_key
    ) / float(sampled_key.shape[-2])
    grouped_queries = (
        prefill_queries.reshape(1, 8, 4, args.prefill_query_tokens, 128)
        .permute(0, 1, 3, 2, 4)
        .reshape(1, 8, args.prefill_query_tokens * 4, 128)
    )
    query_float = grouped_queries.float()
    query_second_moment = torch.einsum(
        "bhnd,bhne->bhde", query_float, query_float
    ) / float(query_float.shape[-2])
    legacy_factors, legacy_factor_seconds = elapsed_seconds(
        lambda: legacy._qk_metric_projection_factors(
            key_second_moment,
            query_second_moment,
            128,
            0.5,
        )
    )
    optimized_factors, optimized_factor_seconds = elapsed_seconds(
        lambda: optimized._qk_metric_projection_factors_with_key_spectrum(
            key_second_moment,
            query_second_moment,
            128,
            0.5,
        )
    )
    legacy_key_eigenvalues = legacy._small_matrix_eigh(key_second_moment)[0]
    factor_comparison = {
        "legacy_seconds": legacy_factor_seconds,
        "optimized_with_spectrum_seconds": optimized_factor_seconds,
        "query_factor": tensor_comparison(
            legacy_factors[0], optimized_factors[0]
        ),
        "key_factor": tensor_comparison(
            legacy_factors[1], optimized_factors[1]
        ),
        "key_spectrum": tensor_comparison(
            legacy_key_eigenvalues, optimized_factors[2]
        ),
    }

    projected_sample = torch.einsum(
        "bhkd,bhdm->bhkm",
        sampled_key.to(dtype),
        legacy_factors[1].to(dtype),
    )
    projected_queries = torch.einsum(
        "bhnd,bhdm->bhnm",
        grouped_queries,
        legacy_factors[0].to(dtype),
    )
    qmse_allocator_rows = []
    for shrinkage in ("none", "oas"):
        for metric_scale in (False, True):
            legacy_allocation, legacy_seconds = elapsed_seconds(
                lambda: legacy._hierarchical_qmse_rate_allocation(
                    projected_sample,
                    projected_queries,
                    15,
                    allow_zero_bits=True,
                    include_scale_metadata=True,
                    query_covariance_shrinkage=shrinkage,
                    metric_scale_quantization=metric_scale,
                )
            )
            optimized_allocation, optimized_seconds = elapsed_seconds(
                lambda: optimized._hierarchical_qmse_rate_allocation(
                    projected_sample,
                    projected_queries,
                    15,
                    allow_zero_bits=True,
                    include_scale_metadata=True,
                    query_covariance_shrinkage=shrinkage,
                    metric_scale_quantization=metric_scale,
                )
            )
            qmse_allocator_rows.append(
                {
                    "query_covariance_shrinkage": shrinkage,
                    "metric_scale_quantization": metric_scale,
                    "legacy_seconds": legacy_seconds,
                    "optimized_seconds": optimized_seconds,
                    "speedup": legacy_seconds / optimized_seconds,
                    **tensor_comparison(
                        legacy_allocation,
                        optimized_allocation,
                    ),
                }
            )

    legacy_state = state_config(prefill_queries.clone())
    optimized_state = state_config(prefill_queries.clone())
    _, legacy_init_seconds = elapsed_seconds(
        lambda: legacy._packed_qmse_initialize(key, legacy_state)
    )
    _, optimized_init_seconds = elapsed_seconds(
        lambda: optimized._packed_qmse_initialize(key, optimized_state)
    )
    state_tensor_names = (
        "basis",
        "query_basis",
        "spectral_weights",
        "packed_qmse_allocation",
        "packed_qmse_key_second_moment",
        "packed_qmse_query_second_moment",
    )
    state_comparison = {
        name: tensor_comparison(legacy_state[name], optimized_state[name])
        for name in state_tensor_names
    }
    packed_comparison = compare_packed_indices(
        legacy_state["packed_qmse_index"],
        optimized_state["packed_qmse_index"],
        args.history_tokens,
    )
    all_allocator_exact = all(row["bitwise_equal"] for row in allocator_rows)
    all_qmse_allocator_exact = all(
        row["bitwise_equal"] for row in qmse_allocator_rows
    )
    all_state_exact = all(
        comparison["bitwise_equal"] for comparison in state_comparison.values()
    )
    all_packed_exact = all(
        comparison.get("bitwise_equal", comparison.get("equal", True))
        for comparison in packed_comparison.values()
    )
    output = {
        "schema": "qksieve_index_build_optimization_validation_v1",
        "hardware": torch.cuda.get_device_name(0),
        "config": {
            "legacy_module": str(args.legacy_module),
            "history_tokens": args.history_tokens,
            "prefill_query_tokens": args.prefill_query_tokens,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "key_allocator": allocator_rows,
        "qmse_allocator": qmse_allocator_rows,
        "qk_factor": factor_comparison,
        "full_qmse_index_build": {
            "legacy_seconds": legacy_init_seconds,
            "optimized_seconds": optimized_init_seconds,
            "speedup": legacy_init_seconds / optimized_init_seconds,
            "state": state_comparison,
            "packed_index": packed_comparison,
        },
        "gates": {
            "all_allocator_allocations_bitwise_equal": all_allocator_exact,
            "all_qmse_allocations_bitwise_equal": all_qmse_allocator_exact,
            "all_initialization_state_tensors_bitwise_equal": all_state_exact,
            "all_packed_index_fields_equal": all_packed_exact,
            "passed": all_allocator_exact
            and all_qmse_allocator_exact
            and all_state_exact
            and all_packed_exact,
        },
    }
    rendered = json.dumps(output, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
