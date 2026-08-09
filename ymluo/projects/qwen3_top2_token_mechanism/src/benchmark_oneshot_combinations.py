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


def tensor_bytes(state: dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def measure_sequence_ms(
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


def extend_history(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    repeat_count = math.ceil(target_count / tensor.shape[-2])
    return tensor.repeat(1, 1, repeat_count, 1)[..., :target_count, :].contiguous()


@torch.inference_mode()
def benchmark_layer(
    records: list[dict[str, Any]],
    layer: int,
    history_count: int,
    warmup_cycles: int,
    measure_cycles: int,
    theta: float,
) -> dict[str, Any]:
    selected = [row for row in records if int(row["layer"]) == layer]
    first = next(row for row in selected if row["key"] is not None)
    queries = [row["query"].cuda().half().contiguous() for row in selected]
    key = extend_history(first["key"].cuda().half(), history_count + 1)
    value = extend_history(first["value"].cuda().half(), history_count + 1)
    groups = queries[0].shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    scaling = float(first["scaling"])

    configurations = {
        "fixed_pca64_top2_autosplit": {
            "mode": "pca_int4_chunked_logscale16_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 64,
            "candidate_fraction": 0.08,
        },
        "fixed_pca48_top2_autosplit": {
            "mode": "pca_int4_chunked_logscale16_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 48,
            "candidate_fraction": 0.08,
        },
        "oneshot_scan_fixed_top2_autosplit": {
            "mode": "pca_int4_logscale16_oneshot95_fixed2_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 64,
        },
        "oneshot_scan_dynamic_budget_autosplit": {
            "mode": "pca_int4_logscale16_oneshot95_budget_autosplit",
            "budgets": (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08),
            "overfetch": 2,
            "projection_dim": 64,
        },
    }
    for candidate_fraction in (0.05, 0.06, 0.07):
        configurations[f"fixed_pca64_candidate{candidate_fraction:.2f}"] = {
            "mode": "pca_int4_chunked_logscale16_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 64,
            "candidate_fraction": candidate_fraction,
        }
    configurations["qkmetric_pca64_candidate0.06"] = {
        "mode": "pca_int4_chunked_logscale16_qkmetric_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 64,
        "candidate_fraction": 0.06,
    }
    configurations["qkmetric_pca48_candidate0.06"] = {
        "mode": "pca_int4_chunked_logscale16_qkmetric_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 48,
        "candidate_fraction": 0.06,
    }
    configurations["qkmetric_pca64_candidate0.04"] = {
        "mode": "pca_int4_chunked_logscale16_qkmetric_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 64,
        "candidate_fraction": 0.04,
    }
    configurations["qkmetric_microblock8_o24_r48_candidate0.06"] = {
        "mode": "pca_int4_qkmetric_microblock8_o24_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 48,
        "candidate_fraction": 0.06,
    }
    configurations["qkmetric_microblock8_o32_r48_candidate0.06"] = {
        "mode": "pca_int4_qkmetric_microblock8_o32_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 48,
        "candidate_fraction": 0.06,
    }
    for outer in (16, 20, 24):
        configurations[f"qkmetric_microblock8_q8_o{outer}_r48_candidate0.06"] = {
            "mode": f"pca_int4_qkmetric_microblock8_q8_o{outer}_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 48,
            "candidate_fraction": 0.06,
        }
    for outer in (16, 20):
        configurations[f"qkmetric_microblock8_q8_o{outer}_r48_candidate0.04"] = {
            "mode": f"pca_int4_qkmetric_microblock8_q8_o{outer}_autosplit",
            "budgets": (0.02,),
            "overfetch": 0,
            "projection_dim": 48,
            "candidate_fraction": 0.04,
        }
    configurations["qkmetric_microblock8_q8_o20_r48_direct2"] = {
        "mode": "pca_int4_qkmetric_microblock8_q8_o20_autosplit",
        "budgets": (0.02,),
        "overfetch": 0,
        "projection_dim": 48,
        "candidate_fraction": 0.02,
        "skip_rerank": True,
    }
    states = {
        name: {
            "rope_theta": theta,
            "sampled_quantile_selected_fraction": config.get(
                "sampleq_selected_fraction", 0.10
            ),
            "sampled_quantile_capacity_fraction": config.get(
                "sampleq_capacity_fraction", 0.15
            ),
        }
        for name, config in configurations.items()
    }
    last_diagnostics: dict[str, dict[str, Any]] = {}

    def full(query: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            is_causal=False,
            scale=scaling,
        )

    def sparse(
        name: str, query: torch.Tensor, *, collect_diagnostics: bool = False
    ) -> torch.Tensor:
        config = configurations[name]
        diagnostics: dict[str, Any] | None = {} if collect_diagnostics else None
        output, _ = qabs_sampled_head_adaptive_attention(
            query,
            key,
            value,
            attention_mask=None,
            scaling=scaling,
            mass_threshold=0.75,
            budget_fractions=config["budgets"],
            sample_fraction=0.0025,
            qabs_dim_count=8,
            candidate_fraction=float(config.get("candidate_fraction", 0.08)),
            use_cuda_kernels=True,
            skip_candidate_rerank=bool(config.get("skip_rerank", False)),
            score_mode=config["mode"],
            projection_dim=int(config["projection_dim"]),
            pca_state=states[name],
            partition_ucb_z=0.0,
            partition_overfetch_factor=config["overfetch"],
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            last_diagnostics[name] = diagnostics
        return output

    callbacks: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"full_sdpa": full}
    callbacks.update(
        {
            name: lambda query, method=name: sparse(method, query)
            for name in configurations
        }
    )

    build_ms: dict[str, float] = {}
    for name in configurations:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callbacks[name](queries[0])
        end.record()
        torch.cuda.synchronize()
        build_ms[name] = float(start.elapsed_time(end))

    timings = {
        name: measure_sequence_ms(callback, queries, warmup_cycles, measure_cycles)
        for name, callback in callbacks.items()
    }
    for name in configurations:
        sparse(name, queries[-1], collect_diagnostics=True)
    torch.cuda.synchronize()
    full_kv_bytes = key.numel() * key.element_size() * 2
    rows = []
    for name, elapsed in timings.items():
        state = states.get(name, {})
        diagnostics = last_diagnostics.get(name, {})
        sampled_counts = state.get("last_sampled_candidate_counts")
        sampled_overflow = state.get("last_sampled_candidate_overflow")
        selected_fraction = diagnostics.get("selected_history_fraction", 1.0)
        if isinstance(selected_fraction, torch.Tensor):
            selected_fraction = float(selected_fraction.float().mean().item())
        rows.append(
            {
                "method": name,
                "pipeline_ms": elapsed,
                "speedup_vs_full_sdpa": timings["full_sdpa"] / elapsed,
                "build_ms": build_ms.get(name, 0.0),
                "state_mib": tensor_bytes(state) / 1024**2,
                "state_ratio_vs_full_kv": tensor_bytes(state) / full_kv_bytes,
                "attention_link_ratio": float(selected_fraction),
                "scan_dimension_fraction": float(
                    diagnostics.get("transport_scan_dimension_fraction", 1.0)
                ),
                "selected_split_count": int(state.get("last_auto_split_count", 0)),
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
                "sampled_quantile_fallback": bool(
                    state.get("last_sampled_quantile_fallback", False)
                ),
                "qk_metric_active": bool(state.get("qk_metric_active", False)),
                "qk_metric_query_count": int(
                    state.get("qk_metric_query_count", 0)
                ),
                "qk_metric_rebuild_count": int(
                    state.get("qk_metric_rebuild_count", 0)
                ),
                "microblock_outer_fraction": float(
                    state.get("last_microblock_outer_fraction", 0.0)
                ),
            }
        )

    del queries, key, value, expanded_key, expanded_value, states
    torch.cuda.empty_cache()
    return {"layer": layer, "query_steps": len(selected), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--warmup_cycles", type=int, default=2)
    parser.add_argument("--measure_cycles", type=int, default=5)
    parser.add_argument("--rope_theta", type=float, default=5_000_000.0)
    args = parser.parse_args()
    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    layers = [int(value) for value in args.layers.split(",")]
    report = {
        "hardware": torch.cuda.get_device_name(),
        "trace": str(args.trace),
        "history_tokens": args.history_tokens,
        "layers": [
            benchmark_layer(
                payload["records"],
                layer,
                args.history_tokens,
                args.warmup_cycles,
                args.measure_cycles,
                args.rope_theta,
            )
            for layer in layers
        ],
    }
    method_names = [row["method"] for row in report["layers"][0]["rows"]]
    report["macro"] = [
        {
            "method": name,
            **{
                key: sum(
                    next(row for row in layer["rows"] if row["method"] == name)[key]
                    for layer in report["layers"]
                )
                / len(report["layers"])
                for key in (
                    "pipeline_ms",
                    "build_ms",
                    "state_mib",
                    "state_ratio_vs_full_kv",
                    "attention_link_ratio",
                    "scan_dimension_fraction",
                    "selected_split_count",
                )
            },
        }
        for name in method_names
    ]
    full_ms = next(row["pipeline_ms"] for row in report["macro"] if row["method"] == "full_sdpa")
    for row in report["macro"]:
        row["speedup_vs_full_sdpa"] = full_ms / row["pipeline_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
