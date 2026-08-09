from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def extend_history(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    repeats = math.ceil(target_count / tensor.shape[-2])
    return tensor.repeat(1, 1, repeats, 1)[..., :target_count, :].contiguous()


def tensor_bytes(state: dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def measure_ms(
    callback: Callable[[torch.Tensor], torch.Tensor],
    queries: list[torch.Tensor],
    warmup_cycles: int,
    measure_cycles: int,
) -> float:
    for _ in range(warmup_cycles):
        for query in queries:
            callback(query)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measure_cycles):
        for query in queries:
            callback(query)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / (measure_cycles * len(queries)))


@torch.inference_mode()
def benchmark_length(
    records: list[dict[str, Any]],
    layer: int,
    history_count: int,
    warmup_cycles: int,
    measure_cycles: int,
    query_steps: int,
    rope_theta: float,
) -> dict[str, Any]:
    selected = [row for row in records if int(row["layer"]) == layer]
    first = next(row for row in selected if row.get("key") is not None)
    queries = [
        row["query"].cuda().half().contiguous()
        for row in selected[:query_steps]
    ]
    key = extend_history(first["key"].cuda().half(), history_count + 1)
    value = extend_history(first["value"].cuda().half(), history_count + 1)
    groups = queries[0].shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    scaling = float(first["scaling"])

    def full(query: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            is_causal=False,
            scale=scaling,
        )

    full_ms = measure_ms(full, queries, warmup_cycles, measure_cycles)

    def full_hf_sdpa(query: torch.Tensor) -> torch.Tensor:
        batch, kv_heads, sequence, head_dim = key.shape
        repeated_key = (
            key[:, :, None, :, :]
            .expand(batch, kv_heads, groups, sequence, head_dim)
            .reshape(batch, kv_heads * groups, sequence, head_dim)
            .contiguous()
        )
        repeated_value = (
            value[:, :, None, :, :]
            .expand(batch, kv_heads, groups, sequence, head_dim)
            .reshape(batch, kv_heads * groups, sequence, head_dim)
            .contiguous()
        )
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            repeated_key,
            repeated_value,
            is_causal=False,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous()

    full_hf_ms = measure_ms(
        full_hf_sdpa,
        queries,
        warmup_cycles,
        measure_cycles,
    )
    rows: list[dict[str, Any]] = [
        {
            "method": "full_sdpa",
            "pipeline_ms": full_ms,
            "speedup_vs_full_sdpa": 1.0,
            "speedup_vs_hf_full_sdpa": full_hf_ms / full_ms,
            "state_mib": 0.0,
        },
        {
            "method": "full_hf_sdpa_with_repeat_kv",
            "pipeline_ms": full_hf_ms,
            "speedup_vs_full_sdpa": full_ms / full_hf_ms,
            "speedup_vs_hf_full_sdpa": 1.0,
            "state_mib": 0.0,
        },
    ]
    methods = {
        "qkmetric48_topk6": (
            "pca_int4_chunked_logscale16_qkmetric_autosplit",
            0.06,
            0.02,
        ),
        "qkmetric48_sampleq256_cap12": (
            "pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit",
            0.06,
            0.02,
        ),
        "qkmetric48_sampleq256_dp4a_cap12": (
            "pca_int4_chunked_logscale16_qkmetric_sampleq_dp4a_autosplit",
            0.06,
            0.02,
        ),
        "qkmetric48_sampleq1024_cap8": (
            "pca_int4_chunked_logscale16_qkmetric_sampleq1024_autosplit",
            0.06,
            0.02,
        ),
        "qkmetric48_sampleq256_candidate4_top1": (
            "pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit",
            0.04,
            0.01,
        ),
    }
    for name, (mode, method_candidate_fraction, method_budget_fraction) in methods.items():
        state: dict[str, Any] = {"rope_theta": rope_theta}

        def sparse(query: torch.Tensor) -> torch.Tensor:
            return qabs_sampled_head_adaptive_attention(
                query,
                key,
                value,
                attention_mask=None,
                scaling=scaling,
                mass_threshold=0.75,
                budget_fractions=(method_budget_fraction,),
                sample_fraction=0.0025,
                qabs_dim_count=8,
                candidate_fraction=method_candidate_fraction,
                use_cuda_kernels=True,
                skip_candidate_rerank=False,
                score_mode=mode,
                projection_dim=48,
                pca_state=state,
            )[0]

        elapsed = measure_ms(sparse, queries, warmup_cycles, measure_cycles)
        sampled_counts = state.get("last_sampled_candidate_counts")
        sampled_overflow = state.get("last_sampled_candidate_overflow")
        rows.append(
            {
                "method": name,
                "pipeline_ms": elapsed,
                "speedup_vs_full_sdpa": full_ms / elapsed,
                "speedup_vs_hf_full_sdpa": full_hf_ms / elapsed,
                "state_mib": tensor_bytes(state) / (1024**2),
                "sampled_candidate_fraction_mean": (
                    float(sampled_counts.float().mean().item()) / history_count
                    if isinstance(sampled_counts, torch.Tensor)
                    else None
                ),
                "sampled_candidate_fraction_max": (
                    float(sampled_counts.max().item()) / history_count
                    if isinstance(sampled_counts, torch.Tensor)
                    else None
                ),
                "sampled_overflow_heads": (
                    int(sampled_overflow.sum().item())
                    if isinstance(sampled_overflow, torch.Tensor)
                    else 0
                ),
            }
        )
        del state
        torch.cuda.empty_cache()

    del queries, key, value, expanded_key, expanded_value
    torch.cuda.empty_cache()
    return {"history_tokens": history_count, "layer": layer, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="32768,131072")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--query_steps", type=int, default=8)
    parser.add_argument("--warmup_cycles", type=int, default=3)
    parser.add_argument("--measure_cycles", type=int, default=10)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    args = parser.parse_args()

    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    report = {
        "hardware": torch.cuda.get_device_name(),
        "trace": str(args.trace),
        "results": [
            benchmark_length(
                payload["records"],
                args.layer,
                int(length),
                args.warmup_cycles,
                args.measure_cycles,
                args.query_steps,
                args.rope_theta,
            )
            for length in args.lengths.split(",")
            if length.strip()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
