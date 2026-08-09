"""Direct CUDA benchmark of the official Louver TA index.

The complete path includes Query normalization, sampled budget-threshold
estimation, range filtering, and sparse attention. Complete-path speedups are
measured directly and are never reconstructed from separately timed stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--louver_repo", type=Path, required=True)
    parser.add_argument(
        "--lengths", default="8192,16384,32768,65536,131072"
    )
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_lengths(spec: str) -> list[int]:
    values = sorted({int(item) for item in spec.split(",") if item.strip()})
    if not values or any(value <= 0 or value % 256 for value in values):
        raise ValueError("lengths must be positive multiples of 256")
    return values


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def sampled_threshold_rank(
    history_tokens: int,
    target_count: int,
    sample_count: int,
) -> int:
    if not 0 < target_count <= history_tokens:
        raise ValueError("target_count must be within the history")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return min(
        sample_count,
        max(1, math.ceil(target_count * sample_count / history_tokens)),
    )


def measure_ms(
    function: Callable[[], object], warmup: int, iterations: int
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


def install_louver_import(repo: Path) -> None:
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    module = types.ModuleType("hira")
    module.__path__ = [str(repo)]
    module.__package__ = "hira"
    sys.modules["hira"] = module


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    install_louver_import(args.louver_repo)
    from benchmark_area.kernel_impl.TA_filter_alg.index import (  # noqa: PLC0415
        TAIndex,
        TAIndexConfig,
    )

    torch.manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    for history_tokens in parse_lengths(args.lengths):
        target_count = selected_count(history_tokens)
        sample_count = min(args.sample_count, history_tokens)
        threshold_rank = sampled_threshold_rank(
            history_tokens,
            target_count,
            sample_count,
        )
        iterations = min(
            args.iterations,
            12 if history_tokens >= 65536 else args.iterations,
        )
        warmup = min(args.warmup, iterations)
        query = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        key = torch.randn(
            8, history_tokens, 128, device="cuda", dtype=torch.float16
        )
        value = torch.randn_like(key)
        q_head_to_kv = (
            torch.arange(32, device="cuda", dtype=torch.long) // 4
        )
        sample_positions = torch.linspace(
            0,
            history_tokens - 1,
            sample_count,
            device="cuda",
        ).round().long()
        sampled_key = key.index_select(1, sample_positions).contiguous()

        index = TAIndex(
            TAIndexConfig(
                n_growth=8192,
                refine_iter=5,
                parallel_update=False,
            )
        )
        torch.cuda.synchronize()
        build_started = time.perf_counter()
        index.build(key, value)
        torch.cuda.synchronize()
        build_ms = (time.perf_counter() - build_started) * 1000.0

        expanded_key = key.repeat_interleave(4, dim=0)
        expanded_value = value.repeat_interleave(4, dim=0)
        scaling = 128.0**-0.5

        def full_attention() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query.reshape(1, 32, 1, 128),
                expanded_key.reshape(1, 32, history_tokens, 128),
                expanded_value.reshape(1, 32, history_tokens, 128),
                is_causal=False,
                scale=scaling,
            )

        def normalize_query() -> torch.Tensor:
            return (
                query
                / query.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            ).contiguous()

        normalized_query = normalize_query()

        def estimate_threshold() -> torch.Tensor:
            grouped_query = normalized_query.reshape(8, 4, 128).float()
            scores = torch.einsum(
                "hgd,hmd->hgm", grouped_query, sampled_key.float()
            ).reshape(32, sample_count)
            return torch.topk(
                scores,
                k=threshold_rank,
                dim=-1,
                largest=True,
                sorted=True,
            ).values[:, -1].contiguous()

        threshold = estimate_threshold()

        def range_attention_only() -> torch.Tensor:
            return index.attend(
                normalized_query,
                threshold,
                q_head_to_kv=q_head_to_kv,
                scale=scaling,
            )

        def complete_attention() -> torch.Tensor:
            current_query = (
                query
                / query.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            ).contiguous()
            grouped_query = current_query.reshape(8, 4, 128).float()
            sample_scores = torch.einsum(
                "hgd,hmd->hgm", grouped_query, sampled_key.float()
            ).reshape(32, sample_count)
            current_threshold = torch.topk(
                sample_scores,
                k=threshold_rank,
                dim=-1,
                largest=True,
                sorted=True,
            ).values[:, -1].contiguous()
            return index.attend(
                current_query,
                current_threshold,
                q_head_to_kv=q_head_to_kv,
                scale=scaling,
            )

        output = complete_attention()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("Louver returned non-finite output")
        if index._ws is None:
            raise RuntimeError("Louver workspace was not materialized")
        live_counts = index._ws["live_count"].float()
        full_ms = measure_ms(
            full_attention,
            min(4, warmup),
            min(10, iterations),
        )
        complete_ms = measure_ms(
            complete_attention,
            warmup,
            iterations,
        )
        row = {
            "history_tokens": history_tokens,
            "target_tokens_per_query_head": target_count,
            "target_fraction": target_count / history_tokens,
            "threshold_sample_count_per_kv_head": sample_count,
            "threshold_sample_rank": threshold_rank,
            "actual_live_tokens_per_query_head_mean": float(
                live_counts.mean().item()
            ),
            "actual_live_tokens_per_query_head_max": int(
                live_counts.max().item()
            ),
            "actual_live_fraction_mean": float(
                live_counts.mean().item() / history_tokens
            ),
            "index_build_wall_ms": build_ms,
            "index_memory_bytes_including_exact_kv": index.memory_bytes(),
            "full_preexpanded_sdpa_direct_ms": full_ms,
            "query_normalization_direct_ms": measure_ms(
                normalize_query, warmup, iterations
            ),
            "sampled_threshold_direct_ms": measure_ms(
                estimate_threshold, warmup, iterations
            ),
            "range_filter_sparse_attention_direct_ms": measure_ms(
                range_attention_only, warmup, iterations
            ),
            "attention_complete_direct_ms": complete_ms,
            "attention_speedup_vs_full_preexpanded_sdpa": full_ms / complete_ms,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del query, key, value, sampled_key, expanded_key, expanded_value, index
        torch.cuda.empty_cache()

    result = {
        "schema": "louver_official_direct_cuda_stages_v1",
        "official_commit": "06d3a319ecc5e267a9a45a4b8402523ef79fd448",
        "hardware": torch.cuda.get_device_name(0),
        "contract": {
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "FP16",
            "target_schedule": "min(N,1280,max(256,ceil(0.06*N)))",
            "threshold": (
                "256 systematic resident Keys; sampled budget quantile"
            ),
            "complete_path_includes": [
                "query normalization",
                "sampled threshold estimation",
                "official TA range filter",
                "official sparse attention",
            ],
            "complete_path_measured_directly": True,
            "stage_sum_used": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
