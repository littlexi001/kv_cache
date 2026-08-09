"""Direct CUDA timing of the official BinaryPC-64 retrieval path.

Every reported stage and complete path is enclosed by its own CUDA events.
Complete-path speedups never use a sum of separately timed stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import qabs_cuda_kernels as sparse_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binarypc_repo", type=Path, required=True)
    parser.add_argument("--projection_path", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        default="8192,16384,32768,65536,131072",
    )
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16",),
        default="bfloat16",
        help="The released BinaryPC fused scan kernel requires BF16 probes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_lengths(spec: str) -> list[int]:
    lengths = sorted({int(item) for item in spec.split(",") if item.strip()})
    if not lengths or any(length <= 0 or length % 256 for length in lengths):
        raise ValueError("BinaryPC lengths must be positive multiples of 256")
    return lengths


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


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    count = int(indices.shape[-1])
    scaling = 128.0**-0.5
    if count >= 1280:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
            4,
        )
    if count >= 900:
        return sparse_cuda.final_attention_ragged_self_split(
            query,
            key,
            value,
            indices,
            counts,
            scaling,
            2,
        )
    return sparse_cuda.final_attention_ragged_self(
        query,
        key,
        value,
        indices,
        counts,
        scaling,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.binarypc_repo))
    from bpc.bpc.bpc import (  # noqa: PLC0415
        _load_cuda_kernel,
        binary_project_cuda,
        compute_errors_cuda,
        compute_hashscores_cuda,
    )

    if not _load_cuda_kernel(0):
        raise RuntimeError("official BinaryPC CUDA extension failed to load")
    projections = torch.load(
        args.projection_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(projections, dict) or 16 not in projections:
        raise ValueError("projection checkpoint must contain layer 16")
    dtype = torch.bfloat16
    projection = projections[16].to(device="cuda", dtype=dtype)
    if projection.shape != (8, 64, 128):
        raise ValueError("expected Llama-3.1-8B projection [8,64,128]")

    torch.manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    for history_tokens in parse_lengths(args.lengths):
        count = selected_count(history_tokens)
        error_count = max(1, int(0.1 * count))
        iterations = min(
            args.iterations,
            20 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        query = torch.randn(
            1,
            32,
            128,
            dtype=dtype,
            device="cuda",
        )
        grouped_query = query.reshape(1, 8, 4, 128)
        key = torch.randn(
            1,
            8,
            history_tokens,
            128,
            dtype=dtype,
            device="cuda",
        )
        value = torch.randn_like(key)
        hashcodes = torch.randint(
            -(2**63),
            2**63 - 1,
            (1, 8, history_tokens),
            dtype=torch.int64,
            device="cuda",
        )
        error_indices = torch.stack(
            [
                torch.randperm(history_tokens, device="cuda")[:error_count]
                for _ in range(8)
            ],
            dim=0,
        ).unsqueeze(0)
        counts = torch.full(
            (1, 32),
            count,
            dtype=torch.long,
            device="cuda",
        )
        projection_t = projection.transpose(-1, -2).contiguous()

        def query_probe() -> torch.Tensor:
            return grouped_query @ projection_t

        probe = query_probe().contiguous()

        def hash_scan() -> torch.Tensor:
            return compute_hashscores_cuda(
                probe,
                hashcodes,
                history_tokens,
                history_tokens,
            )

        scores = hash_scan()

        def error_rescue() -> torch.Tensor:
            rescued = scores.clone()
            rescued.scatter_(2, error_indices, 10000)
            return rescued

        rescued_scores = error_rescue()

        def topk() -> torch.Tensor:
            return torch.topk(
                rescued_scores,
                k=count,
                dim=-1,
                sorted=False,
            ).indices

        kv_indices = topk()
        query_indices = kv_indices.repeat_interleave(4, dim=1).contiguous()

        def exact_attention() -> torch.Tensor:
            return sparse_attention(
                query,
                key,
                value,
                query_indices,
                counts,
            )

        def selection_complete() -> torch.Tensor:
            current_probe = (grouped_query @ projection_t).contiguous()
            current_scores = compute_hashscores_cuda(
                current_probe,
                hashcodes,
                history_tokens,
                history_tokens,
            )
            current_scores.scatter_(2, error_indices, 10000)
            return torch.topk(
                current_scores,
                k=count,
                dim=-1,
                sorted=False,
            ).indices

        def attention_complete() -> torch.Tensor:
            current_kv_indices = selection_complete()
            current_indices = current_kv_indices.repeat_interleave(
                4,
                dim=1,
            ).contiguous()
            return sparse_attention(
                query,
                key,
                value,
                current_indices,
                counts,
            )

        full_key = key.repeat_interleave(4, dim=1)
        full_value = value.repeat_interleave(4, dim=1)

        def full_preexpanded_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.unsqueeze(2),
                full_key,
                full_value,
                is_causal=False,
            )

        def index_build() -> tuple[torch.Tensor, torch.Tensor]:
            current_hashes = binary_project_cuda(key, projection)
            errors = compute_errors_cuda(key, current_hashes, projection)
            rescue = torch.topk(
                errors,
                k=error_count,
                dim=-1,
                sorted=False,
            ).indices
            return current_hashes, rescue

        dense_iterations = min(iterations, 10)
        dense_warmup = min(warmup, 5)
        full_ms = measure_ms(
            full_preexpanded_attention,
            dense_warmup,
            dense_iterations,
        )
        row = {
            "history_tokens": history_tokens,
            "selected_tokens_per_kv_head": count,
            "selected_fraction": count / history_tokens,
            "error_rescue_tokens_per_kv_head": error_count,
            "logical_index_bits_per_token_per_kv_head": 64,
            "full_preexpanded_sdpa_direct_ms": full_ms,
            "query_probe_direct_ms": measure_ms(
                query_probe,
                warmup,
                iterations,
            ),
            "fused_quantized_hash_scan_direct_ms": measure_ms(
                hash_scan,
                warmup,
                iterations,
            ),
            "error_rescue_direct_ms": measure_ms(
                error_rescue,
                warmup,
                iterations,
            ),
            "torch_topk_direct_ms": measure_ms(
                topk,
                warmup,
                iterations,
            ),
            "exact_sparse_attention_direct_ms": measure_ms(
                exact_attention,
                warmup,
                iterations,
            ),
            "selection_complete_direct_ms": measure_ms(
                selection_complete,
                warmup,
                iterations,
            ),
            "attention_complete_direct_ms": measure_ms(
                attention_complete,
                warmup,
                iterations,
            ),
            "historical_index_build_direct_ms": measure_ms(
                index_build,
                0,
                1,
            ),
        }
        row["attention_speedup_vs_full_preexpanded_sdpa"] = (
            full_ms / row["attention_complete_direct_ms"]
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            query,
            grouped_query,
            key,
            value,
            hashcodes,
            error_indices,
            counts,
            probe,
            scores,
            rescued_scores,
            kv_indices,
            query_indices,
            full_key,
            full_value,
        )
        torch.cuda.empty_cache()

    output = {
        "schema": "binarypc_official_direct_cuda_stages_v1",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "gqa_group_size": 4,
            "head_dimension": 128,
            "dtype": args.dtype,
            "candidate_schedule": "min(N,1280,max(256,ceil(0.06*N)))",
            "binarypc_selector": (
                "official fused 64-bit scan; GQA-shared max; "
                "10% reconstruction-error rescue"
            ),
            "final_consumer": "same exact ragged sparse QK-softmax-AV CUDA kernel",
            "timing": "CUDA events; every stage and complete path measured directly",
            "stage_sums_used_for_speedup": False,
            "historical_index_build_in_attention_speedup": False,
            "native_first_two_full_layers_included": False,
            "quality_claim": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
