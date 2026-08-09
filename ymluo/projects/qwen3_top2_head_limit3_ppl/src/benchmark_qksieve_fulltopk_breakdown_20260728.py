from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import benchmark_variablebit_spectral_attention_20260727 as varbit_bench
import qabs_cuda_kernels as sparse_cuda
import qksieve_query_cuda_20260728 as qksieve_query_cuda
import variablebit_spectral_cuda_20260727 as varbit_cuda


def parse_ints(spec: str) -> list[int]:
    values = [int(item) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one length")
    return values


def target_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def measure_ms(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the frozen QKSieve full-index top-k decode path."
    )
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260728)
    allocations = varbit_bench.ALLOCATION_PROFILES[
        "qmse_total_b15"
    ].unsqueeze(0)
    rows: list[dict[str, float | int]] = []

    for history_tokens in parse_ints(args.lengths):
        selected_count = target_count(history_tokens)
        metadata = varbit_bench.make_metadata(allocations, history_tokens)
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
        scaling = 128.0**-0.5
        counts = torch.full(
            (1, 32),
            selected_count,
            dtype=torch.long,
            device="cuda",
        )

        def project_query() -> torch.Tensor:
            return torch.einsum(
                "bhgd,bhdm->bhgm",
                grouped_query,
                query_basis,
            )

        projected_query = project_query()

        def prepare_query() -> tuple[torch.Tensor, torch.Tensor]:
            return varbit_cuda.quantize_projected_query(projected_query)

        query_codes, query_scales = prepare_query()

        def fused_prepare_query() -> tuple[torch.Tensor, torch.Tensor]:
            return qksieve_query_cuda.project_quantize(
                grouped_query,
                query_basis,
            )

        fused_query_codes, fused_query_scales = fused_prepare_query()

        def scan() -> torch.Tensor:
            return varbit_cuda.scores(
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
                history_tokens,
            ).reshape(1, 32, history_tokens)

        approximate_scores = scan()

        def select_topk() -> torch.Tensor:
            return torch.topk(
                approximate_scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        candidate_indices = select_topk().contiguous()
        approximate_scores_fp16 = approximate_scores.half()
        gather_indices = candidate_indices.unsqueeze(-1).expand(
            1, 32, selected_count, 128
        )

        def explicit_gather() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                full_key.gather(2, gather_indices),
                full_value.gather(2, gather_indices),
            )

        gathered_key, gathered_value = explicit_gather()

        def gathered_sdpa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                gathered_key,
                gathered_value,
                is_causal=False,
                scale=scaling,
            )

        production_allocation = metadata["bit_allocations"]
        build_index = varbit_cuda.allocate_packed_index(
            production_allocation,
            history_tokens + 1,
            torch.float16,
        )

        def project_encode_history() -> torch.Tensor:
            chunk_tokens = 4096
            for chunk_start in range(0, history_tokens, chunk_tokens):
                chunk_stop = min(
                    history_tokens, chunk_start + chunk_tokens
                )
                projected_key = torch.einsum(
                    "bhkd,bhdm->bhkm",
                    key[..., chunk_start:chunk_stop, :],
                    query_basis,
                )
                varbit_cuda.encode_projected_keys_into(
                    projected_key.contiguous(),
                    build_index,
                    chunk_start,
                )
            return build_index["packed_codes"]

        append_index = varbit_cuda.allocate_packed_index(
            production_allocation,
            history_tokens + 1,
            torch.float16,
        )

        def project_encode_append() -> torch.Tensor:
            projected_key = torch.einsum(
                "bhkd,bhdm->bhkm",
                key[..., history_tokens : history_tokens + 1, :],
                query_basis,
            )
            varbit_cuda.encode_projected_keys_into(
                projected_key.contiguous(),
                append_index,
                history_tokens,
            )
            return append_index["packed_codes"]

        def select_topk_fp16() -> torch.Tensor:
            return torch.topk(
                approximate_scores_fp16,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        candidate_indices_fp16 = select_topk_fp16().contiguous()

        def exact_sparse_attention() -> torch.Tensor:
            if selected_count >= 1280:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    candidate_indices,
                    counts,
                    scaling,
                    4,
                )
            if selected_count >= 900:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    candidate_indices,
                    counts,
                    scaling,
                    2,
                )
            return sparse_cuda.final_attention_ragged_self(
                query,
                key,
                value,
                candidate_indices,
                counts,
                scaling,
            )

        def selection_path() -> torch.Tensor:
            current_projected = project_query()
            current_codes, current_scales = (
                varbit_cuda.quantize_projected_query(current_projected)
            )
            current_scores = varbit_cuda.scores(
                current_codes,
                current_scales,
                packed_codes,
                key_scales,
                metadata["bit_allocations"],
                metadata["code_offsets"],
                metadata["scale_offsets"],
                metadata["code_bases"],
                metadata["scale_bases"],
                metadata["code_strides"],
                metadata["scale_strides"],
                history_tokens,
            ).reshape(1, 32, history_tokens)
            return torch.topk(
                current_scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        def fused_selection_path() -> torch.Tensor:
            current_codes, current_scales = fused_prepare_query()
            current_scores = varbit_cuda.scores(
                current_codes,
                current_scales,
                packed_codes,
                key_scales,
                metadata["bit_allocations"],
                metadata["code_offsets"],
                metadata["scale_offsets"],
                metadata["code_bases"],
                metadata["scale_bases"],
                metadata["code_strides"],
                metadata["scale_strides"],
                history_tokens,
            ).reshape(1, 32, history_tokens)
            return torch.topk(
                current_scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        def fp16_selection_path() -> torch.Tensor:
            current_scores = scan().half()
            return torch.topk(
                current_scores,
                k=selected_count,
                dim=-1,
                sorted=False,
            ).indices

        def complete_sparse_path() -> torch.Tensor:
            current_indices = selection_path().contiguous()
            if selected_count >= 1280:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    4,
                )
            if selected_count >= 900:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    2,
                )
            return sparse_cuda.final_attention_ragged_self(
                query,
                key,
                value,
                current_indices,
                counts,
                scaling,
            )

        def complete_fused_sparse_path() -> torch.Tensor:
            current_indices = fused_selection_path().contiguous()
            if selected_count >= 1280:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    4,
                )
            if selected_count >= 900:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    2,
                )
            return sparse_cuda.final_attention_ragged_self(
                query,
                key,
                value,
                current_indices,
                counts,
                scaling,
            )

        def complete_fp16_sparse_path() -> torch.Tensor:
            current_indices = fp16_selection_path().contiguous()
            if selected_count >= 1280:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    4,
                )
            if selected_count >= 900:
                return sparse_cuda.final_attention_ragged_self_split(
                    query,
                    key,
                    value,
                    current_indices,
                    counts,
                    scaling,
                    2,
                )
            return sparse_cuda.final_attention_ragged_self(
                query,
                key,
                value,
                current_indices,
                counts,
                scaling,
            )

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
            )

        iterations = min(
            args.iterations,
            15 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        project_ms = measure_ms(project_query, warmup, iterations)
        query_quant_ms = measure_ms(prepare_query, warmup, iterations)
        fused_query_prepare_ms = measure_ms(
            fused_prepare_query, warmup, iterations
        )
        scan_ms = measure_ms(scan, warmup, iterations)
        topk_ms = measure_ms(select_topk, warmup, iterations)
        topk_fp16_ms = measure_ms(
            select_topk_fp16, warmup, iterations
        )
        selection_ms = measure_ms(selection_path, warmup, iterations)
        sparse_attention_ms = measure_ms(
            exact_sparse_attention, warmup, iterations
        )
        explicit_gather_ms = measure_ms(
            explicit_gather, warmup, iterations
        )
        gathered_sdpa_ms = measure_ms(
            gathered_sdpa, warmup, iterations
        )
        index_append_ms = measure_ms(
            project_encode_append, warmup, iterations
        )
        index_build_iterations = min(
            3, max(1, iterations // 4)
        )
        index_build_ms = measure_ms(
            project_encode_history,
            1,
            index_build_iterations,
        )
        complete_ms = measure_ms(
            complete_sparse_path, warmup, iterations
        )
        fused_selection_ms = measure_ms(
            fused_selection_path, warmup, iterations
        )
        fp16_selection_ms = measure_ms(
            fp16_selection_path, warmup, iterations
        )
        fused_complete_ms = measure_ms(
            complete_fused_sparse_path, warmup, iterations
        )
        fp16_complete_ms = measure_ms(
            complete_fp16_sparse_path, warmup, iterations
        )
        full_ms = measure_ms(
            full_attention,
            min(5, warmup),
            min(10, iterations),
        )
        sum_stages_ms = (
            project_ms
            + query_quant_ms
            + scan_ms
            + topk_ms
            + sparse_attention_ms
        )
        complete_with_append_ms = complete_ms + index_append_ms
        fused_complete_with_append_ms = (
            fused_complete_ms + index_append_ms
        )
        rows.append(
            {
                "history_tokens": history_tokens,
                "selected_tokens": selected_count,
                "selected_fraction": selected_count / history_tokens,
                "query_projection_ms": project_ms,
                "query_quantization_ms": query_quant_ms,
                "fused_query_prepare_ms": fused_query_prepare_ms,
                "fused_query_code_agreement": float(
                    (
                        fused_query_codes == query_codes
                    ).float().mean().item()
                ),
                "fused_query_scale_max_abs_error": float(
                    (
                        fused_query_scales - query_scales
                    ).abs().max().item()
                ),
                "packed_scan_ms": scan_ms,
                "torch_topk_ms": topk_ms,
                "torch_topk_fp16_ms": topk_fp16_ms,
                "fp16_topk_set_recall": float(
                    (
                        candidate_indices.unsqueeze(-1)
                        == candidate_indices_fp16.unsqueeze(-2)
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),
                "selection_path_ms": selection_ms,
                "fused_selection_path_ms": fused_selection_ms,
                "fp16_selection_path_ms": fp16_selection_ms,
                "exact_sparse_attention_ms": sparse_attention_ms,
                "explicit_kv_gather_ms": explicit_gather_ms,
                "gathered_sdpa_ms": gathered_sdpa_ms,
                "explicit_gather_plus_sdpa_ms": (
                    explicit_gather_ms + gathered_sdpa_ms
                ),
                "historical_index_project_encode_ms": index_build_ms,
                "per_token_index_project_encode_ms": index_append_ms,
                "sum_independent_stages_ms": sum_stages_ms,
                "sum_independent_stages_with_index_append_ms": (
                    sum_stages_ms + index_append_ms
                ),
                "complete_sparse_path_ms": complete_ms,
                "complete_fused_sparse_path_ms": fused_complete_ms,
                "complete_fp16_sparse_path_ms": fp16_complete_ms,
                "complete_sparse_path_with_index_append_ms": (
                    complete_with_append_ms
                ),
                "complete_fused_sparse_path_with_index_append_ms": (
                    fused_complete_with_append_ms
                ),
                "full_sdpa_ms": full_ms,
                "attention_speedup": full_ms / complete_ms,
                "fused_attention_speedup": full_ms / fused_complete_ms,
                "fp16_attention_speedup": full_ms / fp16_complete_ms,
                "attention_speedup_including_index_append": (
                    full_ms / complete_with_append_ms
                ),
                "fused_attention_speedup_including_index_append": (
                    full_ms / fused_complete_with_append_ms
                ),
                "topk_fraction_of_sparse": topk_ms / complete_ms,
                "scan_fraction_of_sparse": scan_ms / complete_ms,
                "exact_attention_fraction_of_sparse": (
                    sparse_attention_ms / complete_ms
                ),
                "fused_gather_attention_speedup_vs_explicit": (
                    (explicit_gather_ms + gathered_sdpa_ms)
                    / sparse_attention_ms
                ),
                "index_bytes": int(metadata["packed_bytes"]),
                "index_ratio_of_full_kv": float(
                    metadata["index_ratio_of_full_kv"]
                ),
            }
        )
        del (
            packed_codes,
            key_scales,
            query,
            grouped_query,
            query_basis,
            key,
            value,
            full_key,
            full_value,
            counts,
            projected_query,
            query_codes,
            query_scales,
            fused_query_codes,
            fused_query_scales,
            approximate_scores,
            candidate_indices,
            gather_indices,
            gathered_key,
            gathered_value,
            build_index,
            append_index,
            approximate_scores_fp16,
            candidate_indices_fp16,
        )
        torch.cuda.empty_cache()

    output = {
        "scope": (
            "One Qwen-style GQA layer: batch=1, 32 Query heads, 8 KV "
            "heads, d=128. Includes Query projection, Query INT8, full "
            "packed-index scan, torch.topk, and exact sparse attention. "
            "The production sparse-attention kernel fuses selected-K/V "
            "gather with exact QK--softmax--AV; explicit gather plus SDPA is "
            "reported only as a diagnostic decomposition. Historical index "
            "projection/encoding and one-token index append are measured "
            "separately. Excludes basis/allocation construction, ordinary "
            "exact-K/V cache append, and non-attention model work."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
