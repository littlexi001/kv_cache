from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark chunk-major PCA INT4 projection/candidate frontiers."
    )
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def measure_ms(function, warmup: int, iterations: int) -> tuple[float, torch.Tensor]:
    output = function()
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations, output


def state_bytes(state: dict[str, object]) -> int:
    names = {"basis", "packed_chunked", "scales"}
    return sum(
        value.numel() * value.element_size()
        for name, value in state.items()
        if name in names and isinstance(value, torch.Tensor)
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    dtype = torch.float16
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    key_count = args.history_tokens + 1
    scaling = head_dim**-0.5
    query = torch.randn(
        1, query_heads, 1, head_dim, dtype=dtype, device=device
    )
    key = torch.randn(1, kv_heads, key_count, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)
    expanded_key = key.repeat_interleave(query_heads // kv_heads, dim=1)
    expanded_value = value.repeat_interleave(query_heads // kv_heads, dim=1)

    configs = (
        ("pca64_top2", 64, 0.02, True, "pca_int4_chunked", 1.0e-6),
        ("pca64_c3", 64, 0.03, False, "pca_int4_chunked", 1.0e-6),
        ("pca64_c4", 64, 0.04, False, "pca_int4_chunked", 1.0e-6),
        ("pca64_c6", 64, 0.06, False, "pca_int4_chunked", 1.0e-6),
        ("pca64_c8", 64, 0.08, False, "pca_int4_chunked", 1.0e-6),
        (
            "pca64_calibrated_z050",
            64,
            0.08,
            False,
            "pca_int4_sample_calibrated",
            0.50,
        ),
        (
            "pca64_calibrated_z075",
            64,
            0.08,
            False,
            "pca_int4_sample_calibrated",
            0.75,
        ),
        (
            "pca64_calibrated_z100",
            64,
            0.08,
            False,
            "pca_int4_sample_calibrated",
            1.00,
        ),
        (
            "pca64_uncertainty_z100",
            64,
            0.08,
            False,
            "pca_int4_uncertainty_band",
            1.00,
        ),
        (
            "pca64_direct_uncertainty_z100",
            64,
            0.08,
            False,
            "pca_int4_direct_uncertainty",
            1.00,
        ),
        ("pca96_top2", 96, 0.02, True, "pca_int4_chunked", 1.0e-6),
        ("pca128_top2", 128, 0.02, True, "pca_int4_chunked", 1.0e-6),
    )
    states = {name: {} for name, *_ in configs}

    def sparse_call(
        name: str,
        projection_dim: int,
        candidate_fraction: float,
        skip: bool,
        score_mode: str,
        confidence_threshold: float,
    ) -> torch.Tensor:
        output, _ = qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=confidence_threshold,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=candidate_fraction,
            use_cuda_kernels=True,
            skip_candidate_rerank=skip,
            score_mode=score_mode,
            projection_dim=projection_dim,
            pca_state=states[name],
        )
        return output

    methods = {
        "full_sdpa": lambda: F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            is_causal=False,
            scale=scaling,
        )
    }
    for name, projection_dim, candidate_fraction, skip, score_mode, threshold in configs:
        methods[name] = lambda n=name, d=projection_dim, c=candidate_fraction, s=skip, m=score_mode, t=threshold: (
            sparse_call(n, d, c, s, m, t)
        )

    build_ms = {}
    for name, function in methods.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        build_ms[name] = (time.perf_counter() - start) * 1000.0
    timings = {}
    outputs = {}
    for name, function in methods.items():
        timings[name], outputs[name] = measure_ms(
            function, args.warmup, args.iterations
        )
    full_ms = timings["full_sdpa"]
    full_kv_bytes = key.numel() * key.element_size() * 2
    results = []
    for name in methods:
        persistent_bytes = state_bytes(states[name]) if name in states else 0
        results.append(
            {
                "method": name,
                "steady_ms": timings[name],
                "speedup_vs_full_sdpa": full_ms / timings[name],
                "index_build_ms": build_ms[name],
                "persistent_index_bytes": persistent_bytes,
                "persistent_index_ratio_vs_compact_fp16_kv": (
                    persistent_bytes / full_kv_bytes
                ),
                "output_norm": float(outputs[name].float().norm().item()),
                "average_candidate_fraction": (
                    float(
                        states[name]["last_calibrated_candidate_counts"]
                        .float()
                        .mean()
                        .item()
                    )
                    / args.history_tokens
                    if name in states
                    and "last_calibrated_candidate_counts" in states[name]
                    else None
                ),
            }
        )
    payload = {
        "history_tokens": args.history_tokens,
        "attention_fraction": 0.02,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
