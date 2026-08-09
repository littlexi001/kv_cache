#!/usr/bin/env python
"""Measure real greedy generation latency for Full, QKSieve, and FIER."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import run_direct_countcap_denseprompt_ppl_20260725 as direct
from run_critical_position_budget_probe_20260715 import run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    consume_speed_stage_timings,
    install_resident_value_sketch_cache,
    install_llama_head_top_fraction_patch,
    load_model,
    prebuild_resident_value_sketch_cache,
    preload_qksieve_qmse_rate_tables,
    preload_qksieve_runtime_extensions,
    prefill_query_tail_mode,
    seed_packed_qmse_prefill_queries,
)


METHODS = {
    "full": (None, 65536),
    "full_native_gqa": (None, 65536),
    "qksieve_no_value_top1280": (
        direct.PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE,
        1280,
    ),
    "qksieve_valuesketch_top1280": (
        direct.PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE,
        1280,
    ),
    "fier_rtn1_g32_top1280": (
        direct.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
        1280,
    ),
    "fier_rtn1_g32_top512": (
        direct.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
        512,
    ),
}
_NATIVE_GQA_CALLS = 0


def native_gqa_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Call PyTorch SDPA with compact KV heads instead of HF repeat_kv."""
    global _NATIVE_GQA_CALLS
    _NATIVE_GQA_CALLS += 1
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if is_causal is None:
        is_causal = bool(
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2).contiguous(), None


def install_native_gqa_sdpa() -> None:
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register("sdpa", native_gqa_sdpa_attention_forward)
    modeling_qwen3.ALL_ATTENTION_FUNCTIONS["sdpa"] = (
        native_gqa_sdpa_attention_forward
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--history_tokens", type=int, default=65536)
    parser.add_argument("--generation_steps", type=int, default=256)
    parser.add_argument("--steady_start", type=int, default=16)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=22.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--original_max_position_embeddings", type=int, default=0)
    parser.add_argument("--global_max_position", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def token_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def cuda_memory_snapshot(*, peak: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes_per_device": [],
            "reserved_bytes_per_device": [],
            "allocated_bytes_total": 0,
            "reserved_bytes_total": 0,
            "allocated_bytes_max_device": 0,
            "reserved_bytes_max_device": 0,
        }
    allocated_reader = (
        torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    )
    reserved_reader = (
        torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved
    )
    allocated = [
        int(allocated_reader(device_index))
        for device_index in range(torch.cuda.device_count())
    ]
    reserved = [
        int(reserved_reader(device_index))
        for device_index in range(torch.cuda.device_count())
    ]
    return {
        "allocated_bytes_per_device": allocated,
        "reserved_bytes_per_device": reserved,
        "allocated_bytes_total": sum(allocated),
        "reserved_bytes_total": sum(reserved),
        "allocated_bytes_max_device": max(allocated, default=0),
        "reserved_bytes_max_device": max(reserved, default=0),
    }


def repeated_stream(tokenizer: Any, path: Path, count: int) -> list[int]:
    tokens = tokenizer(
        path.read_text(encoding="utf-8"),
        add_special_tokens=False,
    )["input_ids"]
    if not tokens:
        raise RuntimeError(f"empty token stream: {path}")
    return (tokens * math.ceil(count / len(tokens)))[:count]


def configure_sparse_args(args: argparse.Namespace, score_mode: str, budget: int) -> None:
    args.model = args.model
    args.direct_score_mode = score_mode
    args.packed_qmse_template_in = None
    args.packed_qmse_template_out = None
    args.direct_fraction = 0.06
    args.exact_fraction = 0.02
    args.direct_min_tokens = min(256, budget)
    args.direct_max_tokens = budget
    args.projection_dim = 128
    args.sample_count = 256
    args.candidate_overfetch = 1.0
    args.qk_metric_query_shrinkage = 0.75
    args.protect_recent_tokens = 0
    args.cache_mode = "preallocated"
    args.preallocated_cache_min_tokens = 1


def capture_prefill_queries_for_score_mode(score_mode: str | None) -> bool:
    return bool(score_mode in direct.PACKED_PREFILL_QUERY_SCORE_MODES)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 2:
        raise ValueError("history_tokens must be at least two")
    if args.generation_steps < 2:
        raise ValueError("generation_steps must be at least two")
    if not 0 <= args.steady_start < args.generation_steps:
        raise ValueError("steady_start must be inside generation_steps")

    value_sketch_enabled = args.method == "qksieve_valuesketch_top1280"
    os.environ["QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH"] = (
        "0" if value_sketch_enabled else "1"
    )
    if value_sketch_enabled:
        os.environ.setdefault("QKSIEVE_VALUE_SKETCH_TAIL_ALPHA", "0.5")

    torch.manual_seed(args.seed)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    args.model = model

    score_mode, budget = METHODS[args.method]
    is_sparse = score_mode is not None
    if is_sparse:
        initial_budget = direct.direct_countcap_target_count(
            args.history_tokens,
            0.06,
            min(256, budget),
            budget,
        )
        final_budget = direct.direct_countcap_target_count(
            args.history_tokens + args.generation_steps - 1,
            0.06,
            min(256, budget),
            budget,
        )
    else:
        initial_budget = args.history_tokens
        final_budget = args.history_tokens + args.generation_steps - 1
    preload_requested = os.environ.get(
        "QKSIEVE_PRELOAD_EXTENSIONS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if is_sparse:
        configure_sparse_args(args, score_mode, budget)
        preload_timings = (
            preload_qksieve_runtime_extensions()
            if preload_requested
            else {}
        )
        rate_table_timings = (
            preload_qksieve_qmse_rate_tables(model)
            if capture_prefill_queries_for_score_mode(score_mode)
            else {}
        )
    else:
        preload_timings = {}
        rate_table_timings = {}

    history = repeated_stream(tokenizer, args.text_file, args.history_tokens)
    capture_prefill_queries = bool(
        is_sparse and capture_prefill_queries_for_score_mode(score_mode)
    )
    prefill_context = (
        prefill_query_tail_mode(8)
        if capture_prefill_queries
        else direct.nullcontext({})
    )
    sync(input_device)
    reset_cuda_peak_memory()
    prefill_wall_start = time.perf_counter()
    with prefill_context as prefill_queries:
        cache, previous_logits, measured_prefill_seconds = direct.dense_prompt(
            model,
            tokenizer,
            history,
            input_device,
            args.prefill_chunk_tokens,
            "preallocated",
            1,
            args.generation_steps + 2,
        )
    sync(input_device)
    prefill_wall_seconds = time.perf_counter() - prefill_wall_start
    post_prefill_memory = cuda_memory_snapshot(peak=False)
    if int(cache.get_seq_length()) != args.history_tokens:
        raise RuntimeError("prefill cache has the wrong sequence length")

    if args.method == "full_native_gqa":
        # PyTorch's native GQA backend can fall back to a high-memory kernel
        # for multi-token prefill. The benchmark targets decode over an
        # already materialized compact-GQA cache, so patch only decode.
        install_native_gqa_sdpa()

    # The common dense prefill predicts this seed. Every measured path starts
    # from the same seed and then feeds its own greedy predictions back in.
    current_token = int(previous_logits.reshape(-1).argmax().item())
    common_seed_token = current_token
    generated: list[int] = []
    step_seconds: list[float] = []
    cumulative_seconds: list[float] = []
    qk_prebuild: dict[str, Any] = {}
    value_prebuild: dict[str, Any] = {}
    value_install: dict[str, Any] = {}

    method_name = "direct_countcap" if is_sparse else "full_attention"
    with direct.sparse_context(args, method_name):
        if capture_prefill_queries:
            seed_packed_qmse_prefill_queries(prefill_queries)

        sync(input_device)
        online_start = time.perf_counter()
        if capture_prefill_queries:
            workers = int(os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "12"))
            if workers > 0:
                qk_prebuild = (
                    direct.sparse_attention.precompute_active_packed_qmse_qk_factors(
                        cache,
                        max_workers=workers,
                    )
                )
        if value_sketch_enabled:
            value_workers = int(
                os.environ.get("QKSIEVE_PARALLEL_VALUE_WORKERS", "12")
            )
            value_prebuild = prebuild_resident_value_sketch_cache(
                cache,
                model,
                max_workers=max(1, value_workers),
            )
            value_install = install_resident_value_sketch_cache(cache)
        sync(input_device)
        prebuild_seconds = time.perf_counter() - online_start
        post_prebuild_memory = cuda_memory_snapshot(peak=False)

        for step in range(args.generation_steps):
            cache, logits, elapsed, _ = run_one_token(
                model,
                current_token,
                cache,
                args.history_tokens + step,
                input_device,
                collect_attention_stats=False,
            )
            current_token = int(logits.reshape(-1).argmax().item())
            generated.append(current_token)
            step_seconds.append(float(elapsed))
            cumulative_seconds.append(time.perf_counter() - online_start)

    sync(input_device)
    peak_memory = cuda_memory_snapshot(peak=True)
    stage_totals_ms = consume_speed_stage_timings()
    stage_mean_ms_per_token = {
        name: total_ms / args.generation_steps
        for name, total_ms in stage_totals_ms.items()
    }

    steady = step_seconds[args.steady_start :]
    horizons = sorted(
        set(
            value
            for value in (16, 32, 64, 128, 256, args.generation_steps)
            if value <= args.generation_steps
        )
    )
    result = {
        "schema": "qksieve_fier_autoregressive_speed_v1",
        "method": args.method,
        "score_mode": score_mode,
        "budget_tokens_per_head": initial_budget,
        "budget_cap_tokens_per_head": budget if is_sparse else None,
        "final_budget_tokens_per_head": final_budget,
        "budget_policy": (
            "min(cap, max(256, ceil(0.06 * history_tokens)))"
            if is_sparse
            else "full_history"
        ),
        "initial_budget_ratio": initial_budget / args.history_tokens,
        "final_budget_ratio": final_budget
        / (args.history_tokens + args.generation_steps - 1),
        "history_tokens": args.history_tokens,
        "generation_steps": args.generation_steps,
        "steady_start": args.steady_start,
        "generation_contract": "greedy_argmax_feedback_fixed_steps",
        "value_sketch_disabled": os.environ.get(
            "QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH", "0"
        ) == "1",
        "value_sketch_tail_alpha": (
            float(os.environ["QKSIEVE_VALUE_SKETCH_TAIL_ALPHA"])
            if value_sketch_enabled
            else None
        ),
        "fier_attention_split_override": int(
            os.environ.get("QKSIEVE_FIER_ATTENTION_SPLIT_OVERRIDE", "0")
        ),
        "native_gqa_attention_calls": _NATIVE_GQA_CALLS,
        "prefill_wall_seconds": prefill_wall_seconds,
        "measured_prefill_seconds": measured_prefill_seconds,
        "prebuild_wall_seconds": prebuild_seconds,
        "post_prefill_memory": post_prefill_memory,
        "post_prebuild_memory": post_prebuild_memory,
        "peak_memory": peak_memory,
        "qk_prebuild": qk_prebuild,
        "value_prebuild": value_prebuild,
        "value_install": value_install,
        "cuda_stage_total_ms": stage_totals_ms,
        "cuda_stage_mean_ms_per_token": stage_mean_ms_per_token,
        "first_step_ms": 1000.0 * step_seconds[0],
        "steady_mean_ms_per_token": 1000.0 * statistics.fmean(steady),
        "steady_median_ms_per_token": 1000.0 * statistics.median(steady),
        "steady_tokens_per_second": 1.0 / statistics.fmean(steady),
        "all_step_mean_ms_per_token": 1000.0 * statistics.fmean(step_seconds),
        "online_wall_seconds": cumulative_seconds[-1],
        "horizons": {
            str(horizon): {
                "online_seconds_including_prebuild": cumulative_seconds[horizon - 1],
                "ms_per_generated_token_including_prebuild": (
                    1000.0 * cumulative_seconds[horizon - 1] / horizon
                ),
                "tokens_per_second_including_prebuild": (
                    horizon / cumulative_seconds[horizon - 1]
                ),
            }
            for horizon in horizons
        },
        "common_seed_token_id": common_seed_token,
        "generated_token_ids": generated,
        "generated_token_sha256": token_hash(generated),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "runtime_extension_preload": preload_timings,
        "qmse_rate_table_preload": rate_table_timings,
        "gpu_name": (
            torch.cuda.get_device_name(input_device)
            if input_device.type == "cuda"
            else "cpu"
        ),
        "model_name_or_path": args.model_name_or_path,
        "model_type": getattr(model.config, "model_type", None),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        "dtype": args.dtype,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
