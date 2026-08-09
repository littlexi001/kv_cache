#!/usr/bin/env python
"""Benchmark compact per-KV-head cold skipping against full-index QKSieve."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import mixedblock_spectral_cuda_20260729 as specialized_cuda
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--lengths", default="32768,65536,120000")
    parser.add_argument("--pool_fractions", default="0.62605,0.70063")
    parser.add_argument("--max_fraction", type=float, default=0.06)
    parser.add_argument("--min_tokens", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1280)
    parser.add_argument("--target_tail_count", type=int, default=16)
    parser.add_argument("--sample_alignment", type=int, default=256)
    parser.add_argument("--sample_cap", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def target_count(args: argparse.Namespace, history_tokens: int) -> int:
    return min(
        history_tokens,
        args.max_tokens,
        max(args.min_tokens, math.ceil(args.max_fraction * history_tokens)),
    )


def sample_configuration(
    history_tokens: int,
    selected_tokens: int,
    target_tail_count: int,
    sample_alignment: int,
    sample_cap: int,
) -> tuple[int, float, int]:
    fraction = min(1.0, selected_tokens / history_tokens)
    raw_sample_count = math.ceil(target_tail_count / fraction)
    sample_count = min(
        sample_cap,
        max(
            sample_alignment,
            sample_alignment
            * math.ceil(raw_sample_count / sample_alignment),
        ),
    )
    rank = max(
        1,
        min(sample_count, round(fraction * (sample_count + 1))),
    )
    threshold_fraction = (rank - 0.5) / sample_count
    standard_error = math.sqrt(
        fraction * (1.0 - fraction) / sample_count
    )
    capacity_fraction = min(
        1.0, max(0.06, fraction + 6.0 * standard_error)
    )
    capacity = min(
        history_tokens,
        max(1, math.ceil(capacity_fraction * history_tokens)),
    )
    return sample_count, threshold_fraction, capacity


def allocate_outputs(capacity: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, 32, capacity, dtype=torch.long, device="cuda"),
        torch.zeros(1, 32, dtype=torch.long, device="cuda"),
        torch.zeros(1, 32, dtype=torch.float32, device="cuda"),
        torch.zeros(1, 32, dtype=torch.bool, device="cuda"),
    )


def ragged_attention_split_count(candidate_capacity: int) -> int:
    if candidate_capacity <= 0:
        raise ValueError("candidate_capacity must be positive")
    if candidate_capacity <= 4096:
        return 8
    required_splits = math.ceil(candidate_capacity / 11_000)
    if required_splits > 16:
        raise RuntimeError(
            "ragged attention candidate capacity requires more "
            "than 16 splits"
        )
    return next(
        split for split in (4, 8, 16) if split >= required_splits
    )


def invoke_retrieval(
    extension: object,
    query_codes: torch.Tensor,
    query_scales: torch.Tensor,
    packed_index: dict[str, torch.Tensor | int],
    outputs: tuple[torch.Tensor, ...],
    history_tokens: int,
    sample_count: int,
    threshold_fraction: float,
) -> tuple[torch.Tensor, ...]:
    extension.plain_sampled_compact_gqa4_indices_out(
        query_codes,
        query_scales,
        packed_index["packed_codes"],
        packed_index["key_scales"],
        packed_index["bit_allocations"],
        packed_index["code_offsets"],
        packed_index["scale_offsets"],
        packed_index["code_bases"],
        packed_index["scale_bases"],
        packed_index["code_strides"],
        packed_index["scale_strides"],
        *outputs,
        history_tokens,
        sample_count,
        threshold_fraction,
    )
    return outputs


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    candidate_capacity = int(indices.shape[-1])
    split_count = ragged_attention_split_count(candidate_capacity)
    return sparse_cuda.final_attention_ragged_self_split(
        query,
        key,
        value,
        indices,
        counts,
        128.0**-0.5,
        split_count,
    )


def deterministic_token_map(
    history_tokens: int,
    pool_tokens: int,
) -> torch.Tensor:
    maps = []
    base = torch.arange(history_tokens, device="cuda")
    for head in range(8):
        multiplier = 2 * head + 1
        if math.gcd(multiplier, history_tokens) != 1:
            multiplier = 1
        priority = (base * multiplier + 104729 * head) % history_tokens
        maps.append(torch.topk(priority, k=pool_tokens, largest=False).indices)
    return (
        torch.stack(maps)
        .sort(dim=-1)
        .values[None]
        .repeat_interleave(4, dim=1)
        .contiguous()
    )


def hf_repeat_kv(
    hidden_states: torch.Tensor,
    repetition_count: int,
) -> torch.Tensor:
    """Match Transformers repeat_kv, including its materialization cost."""
    batch, kv_heads, tokens, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, kv_heads, repetition_count, tokens, head_dim)
        .reshape(batch, kv_heads * repetition_count, tokens, head_dim)
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.target_tail_count <= 0:
        raise ValueError("target_tail_count must be positive")
    if args.sample_alignment <= 0:
        raise ValueError("sample_alignment must be positive")
    if (
        args.sample_cap < args.sample_alignment
        or args.sample_cap % args.sample_alignment != 0
    ):
        raise ValueError(
            "sample_cap must be an aligned positive upper bound"
        )
    torch.manual_seed(args.seed)
    extension = specialized_cuda.load_extension()
    template = torch.load(
        args.template, map_location="cpu", weights_only=False
    )
    lengths = sorted(
        {int(value) for value in args.lengths.split(",") if value}
    )
    pool_fractions = sorted(
        {float(value) for value in args.pool_fractions.split(",") if value}
    )
    layer_ids = sorted(template)
    rows: list[dict[str, object]] = []

    for history_tokens in lengths:
        selected_tokens = target_count(args, history_tokens)
        full_sample_count, full_threshold, full_capacity = (
            sample_configuration(
                history_tokens,
                selected_tokens,
                args.target_tail_count,
                args.sample_alignment,
                args.sample_cap,
            )
        )
        query = torch.randn(
            1, 32, 128, dtype=torch.float16, device="cuda"
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        query_basis = torch.randn(
            1, 8, 128, 128, dtype=torch.float16, device="cuda"
        )
        key = torch.randn(
            1,
            8,
            history_tokens + 1,
            128,
            dtype=torch.float16,
            device="cuda",
        )
        value = torch.randn_like(key)
        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        def full_attention_preexpanded() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2), full_key, full_value
            )

        iterations = min(
            args.iterations,
            10 if history_tokens >= 100_000 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        full_sdpa_ms = measure_ms(
            full_attention_preexpanded,
            min(3, warmup),
            min(5, iterations),
        )
        del full_key, full_value
        torch.cuda.empty_cache()

        def full_attention_hf_gqa() -> torch.Tensor:
            expanded_key = hf_repeat_kv(key, 4)
            expanded_value = hf_repeat_kv(value, 4)
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                expanded_key,
                expanded_value,
            )

        full_hf_gqa_ms = measure_ms(
            full_attention_hf_gqa,
            min(3, warmup),
            min(5, iterations),
        )
        per_fraction: dict[float, list[dict[str, float]]] = {
            fraction: [] for fraction in pool_fractions
        }

        for layer_id in layer_ids:
            allocation = template[layer_id]["allocation"].to(
                device="cuda", dtype=torch.int8
            )
            full_index = varbit_cuda.allocate_packed_index(
                allocation, history_tokens, torch.float16
            )
            full_index["packed_codes"].random_(0, 256)
            full_index["key_scales"].uniform_(0.05, 1.0)
            full_outputs = allocate_outputs(full_capacity)

            def prepare_query() -> tuple[torch.Tensor, torch.Tensor]:
                return query_cuda.project_quantize_wmma(
                    grouped_query, query_basis
                )

            def full_retrieve() -> tuple[torch.Tensor, ...]:
                query_codes, query_scales = prepare_query()
                return invoke_retrieval(
                    extension,
                    query_codes,
                    query_scales,
                    full_index,
                    full_outputs,
                    history_tokens,
                    full_sample_count,
                    full_threshold,
                )

            def full_complete() -> torch.Tensor:
                indices, counts, _, _ = full_retrieve()
                return sparse_attention(
                    query, key, value, indices, counts
                )

            full_retrieval_ms = measure_ms(
                full_retrieve, warmup, iterations
            )
            full_complete_ms = measure_ms(
                full_complete, warmup, iterations
            )

            for fraction in pool_fractions:
                pool_tokens = min(
                    history_tokens,
                    max(selected_tokens, math.ceil(fraction * history_tokens)),
                )
                pool_sample_count, pool_threshold, pool_capacity = (
                    sample_configuration(
                        pool_tokens,
                        selected_tokens,
                        args.target_tail_count,
                        args.sample_alignment,
                        args.sample_cap,
                    )
                )
                compact_index = varbit_cuda.allocate_packed_index(
                    allocation, pool_tokens, torch.float16
                )
                compact_index["packed_codes"].random_(0, 256)
                compact_index["key_scales"].uniform_(0.05, 1.0)
                local_outputs = allocate_outputs(pool_capacity)
                token_map = deterministic_token_map(
                    history_tokens, pool_tokens
                )
                mapped_indices = torch.empty_like(local_outputs[0])

                def compact_retrieve_local() -> tuple[torch.Tensor, ...]:
                    query_codes, query_scales = prepare_query()
                    return invoke_retrieval(
                        extension,
                        query_codes,
                        query_scales,
                        compact_index,
                        local_outputs,
                        pool_tokens,
                        pool_sample_count,
                        pool_threshold,
                    )

                def map_indices() -> torch.Tensor:
                    return torch.gather(
                        token_map,
                        2,
                        local_outputs[0],
                        out=mapped_indices,
                    )

                def compact_retrieve_mapped() -> tuple[torch.Tensor, ...]:
                    compact_retrieve_local()
                    map_indices()
                    return (
                        mapped_indices,
                        local_outputs[1],
                        local_outputs[2],
                        local_outputs[3],
                    )

                def compact_complete() -> torch.Tensor:
                    indices, counts, _, _ = compact_retrieve_mapped()
                    return sparse_attention(
                        query, key, value, indices, counts
                    )

                compact_retrieve_local()
                map_indices()
                map_ms = measure_ms(map_indices, warmup, iterations)
                compact_retrieval_ms = measure_ms(
                    compact_retrieve_mapped, warmup, iterations
                )
                compact_complete_ms = measure_ms(
                    compact_complete, warmup, iterations
                )
                candidate_count = float(
                    local_outputs[1].float().mean().item()
                )
                per_fraction[fraction].append(
                    {
                        "full_retrieval_ms": full_retrieval_ms,
                        "full_complete_ms": full_complete_ms,
                        "compact_map_ms": map_ms,
                        "compact_retrieval_ms": compact_retrieval_ms,
                        "compact_complete_ms": compact_complete_ms,
                        "candidate_count": candidate_count,
                        "pool_tokens": float(pool_tokens),
                        "runtime_index_bytes": float(
                            int(compact_index["total_code_bytes"])
                            + int(compact_index["total_scale_values"]) * 2
                            + token_map.numel() * token_map.element_size()
                        ),
                        "target_index_bytes": float(
                            int(compact_index["total_code_bytes"])
                            + int(compact_index["total_scale_values"]) * 2
                            + 4 * pool_tokens * 8
                        ),
                    }
                )
                del (
                    compact_index,
                    local_outputs,
                    token_map,
                    mapped_indices,
                )
            del full_index, full_outputs

        for fraction, layer_rows in per_fraction.items():
            full_retrieval_sum = sum(
                row["full_retrieval_ms"] for row in layer_rows
            )
            full_complete_sum = sum(
                row["full_complete_ms"] for row in layer_rows
            )
            compact_retrieval_sum = sum(
                row["compact_retrieval_ms"] for row in layer_rows
            )
            compact_complete_sum = sum(
                row["compact_complete_ms"] for row in layer_rows
            )
            rows.append(
                {
                    "history_tokens": history_tokens,
                    "selected_tokens": selected_tokens,
                    "target_tail_count": args.target_tail_count,
                    "quantile_sample_count": full_sample_count,
                    "requested_pool_fraction": fraction,
                    "actual_pool_fraction": (
                        layer_rows[0]["pool_tokens"] / history_tokens
                    ),
                    "layers": len(layer_rows),
                    "candidate_tokens_mean": float(
                        sum(row["candidate_count"] for row in layer_rows)
                        / len(layer_rows)
                    ),
                    "full_sdpa_layer_ms": full_sdpa_ms,
                    "full_sdpa_model_attention_ms": (
                        full_sdpa_ms * len(layer_rows)
                    ),
                    "full_hf_gqa_layer_ms": full_hf_gqa_ms,
                    "full_hf_gqa_model_attention_ms": (
                        full_hf_gqa_ms * len(layer_rows)
                    ),
                    "current_qksieve_retrieval_model_ms": (
                        full_retrieval_sum
                    ),
                    "coldskip_retrieval_model_ms": compact_retrieval_sum,
                    "retrieval_speedup_vs_current": (
                        full_retrieval_sum / compact_retrieval_sum
                    ),
                    "current_qksieve_attention_model_ms": full_complete_sum,
                    "coldskip_attention_model_ms": compact_complete_sum,
                    "attention_speedup_vs_current": (
                        full_complete_sum / compact_complete_sum
                    ),
                    "current_qksieve_speedup_vs_full": (
                        full_sdpa_ms * len(layer_rows) / full_complete_sum
                    ),
                    "current_qksieve_speedup_vs_hf_gqa_full": (
                        full_hf_gqa_ms
                        * len(layer_rows)
                        / full_complete_sum
                    ),
                    "coldskip_speedup_vs_full": (
                        full_sdpa_ms
                        * len(layer_rows)
                        / compact_complete_sum
                    ),
                    "coldskip_speedup_vs_hf_gqa_full": (
                        full_hf_gqa_ms
                        * len(layer_rows)
                        / compact_complete_sum
                    ),
                    "coldskip_map_model_ms": sum(
                        row["compact_map_ms"] for row in layer_rows
                    ),
                    "runtime_int64_index_ratio_vs_full_kv": (
                        sum(
                            row["runtime_index_bytes"]
                            for row in layer_rows
                        )
                        / len(layer_rows)
                        / (history_tokens * 8 * 128 * 4)
                    ),
                    "target_uint32_index_ratio_vs_full_kv": (
                        sum(
                            row["target_index_bytes"] for row in layer_rows
                        )
                        / len(layer_rows)
                        / (history_tokens * 8 * 128 * 4)
                    ),
                }
            )
        del query, grouped_query, query_basis, key, value
        torch.cuda.empty_cache()

    output = {
        "scope": (
            "Qwen3-4B-style 36-layer GQA attention sum. Current QKSieve and "
            "cold-skip include WMMA Query projection/INT8, sampled-quantile "
            "selection, and exact sparse QK-softmax-AV. Cold-skip additionally "
            "includes per-query-head local-to-global token-ID mapping. Dense "
            "baselines report both pre-expanded SDPA and HF-style repeat_kv "
            "plus SDPA. Index construction and non-attention model work are "
            "excluded."
        ),
        "template": str(args.template),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
