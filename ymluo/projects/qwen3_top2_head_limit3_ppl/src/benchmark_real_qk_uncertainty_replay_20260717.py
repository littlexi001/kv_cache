from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay real Q/K traces at a controlled history length."
    )
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--record_index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
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


def repeat_history(
    source_key: torch.Tensor, history_tokens: int
) -> torch.Tensor:
    source_history = source_key[..., :-1, :]
    current_key = source_key[..., -1:, :]
    repeats = math.ceil(history_tokens / source_history.shape[-2])
    history = source_history.repeat(1, 1, repeats, 1)[..., :history_tokens, :]
    return torch.cat((history, current_key), dim=-2).contiguous()


def state_bytes(state: dict[str, object]) -> int:
    names = {"basis", "packed_chunked", "scales"}
    return sum(
        value.numel() * value.element_size()
        for name, value in state.items()
        if name in names and isinstance(value, torch.Tensor)
    )


@torch.inference_mode()
def benchmark_trace(args: argparse.Namespace, trace_path: Path) -> dict[str, object]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    record = payload["records"][args.record_index]
    device = torch.device("cuda")
    query = record["query"].to(device=device, dtype=torch.float16).contiguous()
    source_key = record["key"].to(device=device, dtype=torch.float16)
    key = repeat_history(source_key, args.history_tokens)
    value = torch.randn_like(key)
    group_count = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(group_count, dim=1)
    expanded_value = value.repeat_interleave(group_count, dim=1)
    scaling = float(record["scaling"])

    configs = (
        ("fixed_c4", "pca_int4_chunked", 0.04, 1.0e-6, 1.0),
        ("fixed_c6", "pca_int4_chunked", 0.06, 1.0e-6, 1.0),
        ("uncertainty_z075", "pca_int4_uncertainty_band", 0.08, 0.75, 1.0),
        ("uncertainty_z100", "pca_int4_uncertainty_band", 0.08, 1.00, 1.0),
        ("uncertainty_z075_v95", "pca_int4_uncertainty_band", 0.08, 0.75, 0.95),
        ("uncertainty_z075_v995", "pca_int4_uncertainty_band", 0.08, 0.75, 0.995),
        ("uncertainty_z075_v99", "pca_int4_uncertainty_band", 0.08, 0.75, 0.99),
        ("direct_uncertainty_z100", "pca_int4_direct_uncertainty", 0.08, 1.00, 1.0),
    )
    states: dict[str, dict[str, object]] = {name: {} for name, *_ in configs}

    def sparse_call(
        name: str,
        score_mode: str,
        candidate_fraction: float,
        confidence_width: float,
        value_mass_threshold: float,
    ) -> torch.Tensor:
        output, _ = qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=confidence_width,
            budget_fractions=(0.02,),
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=candidate_fraction,
            use_cuda_kernels=True,
            skip_candidate_rerank=False,
            score_mode=score_mode,
            projection_dim=64,
            pca_state=states[name],
            value_mass_threshold=value_mass_threshold,
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
    for name, mode, fraction, width, value_threshold in configs:
        methods[name] = (
            lambda n=name, m=mode, f=fraction, w=width, v=value_threshold: sparse_call(
                n, m, f, w, v
            )
        )

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
        state = states.get(name, {})
        candidate_counts = state.get("last_calibrated_candidate_counts")
        overflow = state.get("last_direct_candidate_overflow")
        persistent_bytes = state_bytes(state)
        results.append(
            {
                "method": name,
                "steady_ms": timings[name],
                "speedup_vs_full_sdpa": full_ms / timings[name],
                "average_candidate_fraction": (
                    float(candidate_counts.float().mean().item())
                    / args.history_tokens
                    if isinstance(candidate_counts, torch.Tensor)
                    else None
                ),
                "candidate_overflow": (
                    bool(overflow.any().item())
                    if isinstance(overflow, torch.Tensor)
                    else None
                ),
                "persistent_index_bytes": persistent_bytes,
                "persistent_index_ratio_vs_compact_fp16_kv": (
                    persistent_bytes / full_kv_bytes
                ),
                "output_norm": float(outputs[name].float().norm().item()),
            }
        )
    return {
        "trace": trace_path.stem,
        "record_index": args.record_index,
        "layer": int(record["layer"]),
        "source_history_tokens": int(source_key.shape[-2] - 1),
        "replay_history_tokens": args.history_tokens,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260717)
    traces = []
    for trace_path in args.trace_paths:
        traces.append(benchmark_trace(args, trace_path))
        torch.cuda.empty_cache()
    payload = {"traces": traces}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
