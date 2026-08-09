from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def measure_ms(callback: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def tensor_bytes(state: dict[str, object]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


@torch.inference_mode()
def benchmark_length(
    history_count: int,
    warmup: int,
    repeats: int,
    theta: float,
) -> dict[str, object]:
    batch, query_heads, kv_heads, head_dim = 1, 32, 8, 128
    groups = query_heads // kv_heads
    device = torch.device("cuda")
    dtype = torch.float16
    scaling = head_dim**-0.5
    query = torch.randn(batch, query_heads, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, kv_heads, history_count + 1, head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    states = {
        "pca64_candidate6": {"rope_theta": theta},
        "pca64_candidate8": {"rope_theta": theta},
        "pca64_candidate8_split8": {"rope_theta": theta},
        "pca64_candidate8_split16": {"rope_theta": theta},
        "pca64_candidate8_autosplit": {"rope_theta": theta},
        "pca64_tailvalue005": {"rope_theta": theta},
        "pca64_tailvalue005_shrink50": {"rope_theta": theta},
        "pca64_tailvalue005_shrink50_mass95": {"rope_theta": theta},
        "pca64_tailvalue005_reliability": {"rope_theta": theta},
        "dual_space_candidate8": {"rope_theta": theta},
        "dual_space_int2_refresh4": {"rope_theta": theta},
        "dual_space_int2_oldest50_refresh4": {"rope_theta": theta},
    }
    call_counts = {name: 0 for name in states}

    def sparse(name: str, mode: str, candidate_fraction: float) -> torch.Tensor:
        if name in {
            "dual_space_int2_refresh4",
            "dual_space_int2_oldest50_refresh4",
        }:
            if call_counts[name] % 4 == 0:
                states[name]["last_lowfreq_rescue_refresh"] = history_count - 4
            call_counts[name] += 1
        return qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=0.75,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=candidate_fraction,
            use_cuda_kernels=True,
            skip_candidate_rerank=False,
            score_mode=mode,
            projection_dim=64,
            pca_state=states[name],
        )[0]

    methods: dict[str, Callable[[], torch.Tensor]] = {
        "full_sdpa": lambda: F.scaled_dot_product_attention(
            query, expanded_key, expanded_value, is_causal=False, scale=scaling
        ),
        "pca64_candidate6": lambda: sparse(
            "pca64_candidate6", "pca_int4_chunked_logscale16", 0.06
        ),
        "pca64_candidate8": lambda: sparse(
            "pca64_candidate8", "pca_int4_chunked_logscale16", 0.08
        ),
        "pca64_candidate8_split8": lambda: sparse(
            "pca64_candidate8_split8",
            "pca_int4_chunked_logscale16_split8",
            0.08,
        ),
        "pca64_candidate8_split16": lambda: sparse(
            "pca64_candidate8_split16",
            "pca_int4_chunked_logscale16_split16",
            0.08,
        ),
        "pca64_candidate8_autosplit": lambda: sparse(
            "pca64_candidate8_autosplit",
            "pca_int4_chunked_logscale16_autosplit",
            0.08,
        ),
        "pca64_tailvalue005": lambda: sparse(
            "pca64_tailvalue005",
            "pca_int4_chunked_logscale16_tailvalue005",
            0.08,
        ),
        "pca64_tailvalue005_shrink50": lambda: sparse(
            "pca64_tailvalue005_shrink50",
            "pca_int4_chunked_logscale16_tailvalue005_shrink50",
            0.08,
        ),
        "pca64_tailvalue005_shrink50_mass95": lambda: sparse(
            "pca64_tailvalue005_shrink50_mass95",
            "pca_int4_chunked_logscale16_tailvalue005_shrink50_mass95",
            0.08,
        ),
        "pca64_tailvalue005_reliability": lambda: sparse(
            "pca64_tailvalue005_reliability",
            "pca_int4_chunked_logscale16_tailvalue005_reliability",
            0.08,
        ),
        "dual_space_candidate8": lambda: sparse(
            "dual_space_candidate8",
            "pca_int4_chunked_logscale16_lowfreq32_rescue005",
            0.08,
        ),
        "dual_space_int2_refresh4": lambda: sparse(
            "dual_space_int2_refresh4",
            "pca_int4_chunked_logscale16_lowfreq32_int2_union005_refresh4",
            0.08,
        ),
        "dual_space_int2_oldest50_refresh4": lambda: sparse(
            "dual_space_int2_oldest50_refresh4",
            "pca_int4_chunked_logscale16_lowfreq32_int2_oldest50_union005_refresh4",
            0.08,
        ),
    }
    build_ms: dict[str, float] = {}
    for name in states:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        methods[name]()
        end.record()
        torch.cuda.synchronize()
        build_ms[name] = float(start.elapsed_time(end))
    timings = {
        name: measure_ms(callback, warmup, repeats)
        for name, callback in methods.items()
    }
    full_kv_bytes = key.numel() * key.element_size() * 2
    rows = []
    for name, elapsed in timings.items():
        index_bytes = tensor_bytes(states[name]) if name in states else 0
        rows.append(
            {
                "method": name,
                "steady_ms": elapsed,
                "speedup_vs_full_sdpa": timings["full_sdpa"] / elapsed,
                "build_ms": build_ms.get(name, 0.0),
                "state_mib": index_bytes / 1024**2,
                "state_ratio_vs_full_kv": index_bytes / full_kv_bytes,
                "selected_split_count": int(
                    states.get(name, {}).get("last_auto_split_count", 0)
                ),
            }
        )
    del query, key, value, expanded_key, expanded_value, states
    torch.cuda.empty_cache()
    return {"history_tokens": history_count, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="8192,32768,65536,131072")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260720)
    report = {
        "hardware": torch.cuda.get_device_name(),
        "results": [
            benchmark_length(int(length), args.warmup, args.repeats, args.rope_theta)
            for length in args.lengths.split(",")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
