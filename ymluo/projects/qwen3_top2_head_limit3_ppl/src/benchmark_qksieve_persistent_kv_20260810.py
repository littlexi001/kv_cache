#!/usr/bin/env python
"""Benchmark cold and persistent-KV QKSieve request lifecycles."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import run_direct_countcap_denseprompt_ppl_20260725 as direct
import run_qksieve_fier_autoregressive_speed_20260808 as speed
from run_critical_position_budget_probe_20260715 import run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    active_qksieve_persistent_state_signature,
    install_llama_head_top_fraction_patch,
    install_resident_value_sketch_cache,
    prebuild_resident_value_sketch_cache,
    prefill_query_tail_mode,
    rewind_active_qksieve_cache,
    seed_packed_qmse_prefill_queries,
)


METHODS = ("full", "qksieve_robust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--history_tokens", type=int, default=65536)
    parser.add_argument("--branch_count", type=int, default=4)
    parser.add_argument("--branch_steps", type=int, default=32)
    parser.add_argument("--append_steps", type=int, default=128)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=22.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--original_max_position_embeddings", type=int, default=0)
    parser.add_argument("--global_max_position", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def crop_for_next_branch(cache: Any, prefix_length: int, sparse: bool) -> dict[str, int]:
    if sparse:
        return rewind_active_qksieve_cache(cache, prefix_length)
    crop = getattr(cache, "crop", None)
    if not callable(crop):
        raise TypeError("persistent benchmark requires a crop-capable cache")
    crop(prefix_length)
    return {"active_length": prefix_length, "key_layers": 0, "value_layers": 0}


def stable_index_signature(snapshot: dict[str, Any]) -> list[tuple[int | None, ...]]:
    return [
        (
            layer["key_rebuild_count"],
            layer["key_code_ptr"],
            layer["key_scale_ptr"],
            layer["value_code_ptr"],
            layer["value_minimum_ptr"],
            layer["value_scale_ptr"],
        )
        for layer in snapshot["layers"]
    ]


@torch.inference_mode()
def run_branch(
    model: torch.nn.Module,
    cache: Any,
    seed_token: int,
    prefix_length: int,
    steps: int,
    input_device: torch.device,
) -> dict[str, Any]:
    current_token = int(seed_token)
    generated: list[int] = []
    step_seconds: list[float] = []
    sync(input_device)
    wall_start = time.perf_counter()
    for step in range(steps):
        cache, logits, elapsed, _ = run_one_token(
            model,
            current_token,
            cache,
            prefix_length + step,
            input_device,
            collect_attention_stats=False,
        )
        current_token = int(logits.reshape(-1).argmax().item())
        generated.append(current_token)
        step_seconds.append(float(elapsed))
    sync(input_device)
    wall_seconds = time.perf_counter() - wall_start
    return {
        "seed_token_id": int(seed_token),
        "generated_token_ids": generated,
        "generated_token_sha256": speed.token_hash(generated),
        "wall_seconds": wall_seconds,
        "mean_ms_per_token": 1000.0 * wall_seconds / steps,
        "cuda_synchronized_step_mean_ms": 1000.0 * statistics.fmean(step_seconds),
        "first_step_ms": 1000.0 * step_seconds[0],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 2:
        raise ValueError("history_tokens must be at least two")
    if args.branch_count < 2:
        raise ValueError("branch_count must be at least two")
    if args.branch_steps < 2 or args.append_steps < 2:
        raise ValueError("branch_steps and append_steps must be at least two")

    sparse = args.method == "qksieve_robust"
    os.environ["QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH"] = "0" if sparse else "1"
    if sparse:
        os.environ.setdefault("QKSIEVE_VALUE_SKETCH_TAIL_ALPHA", "0.5")

    torch.manual_seed(args.seed)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = speed.load_model(args)
    args.model = model

    score_mode, budget = speed.METHODS[
        "qksieve_valuesketch_top1280" if sparse else "full"
    ]
    if sparse:
        speed.configure_sparse_args(args, score_mode, budget)
        rate_table_timings = speed.preload_qksieve_qmse_rate_tables(model)
    else:
        rate_table_timings = {}

    history = speed.repeated_stream(tokenizer, args.text_file, args.history_tokens)
    prefill_context = prefill_query_tail_mode(8) if sparse else direct.nullcontext({})
    sync(input_device)
    prefill_start = time.perf_counter()
    with prefill_context as prefill_queries:
        cache, previous_logits, measured_prefill_seconds = direct.dense_prompt(
            model,
            tokenizer,
            history,
            input_device,
            args.prefill_chunk_tokens,
            "preallocated",
            1,
            max(args.branch_steps, args.append_steps) + 2,
        )
    sync(input_device)
    prefill_wall_seconds = time.perf_counter() - prefill_start
    prefix_length = int(cache.get_seq_length())
    if prefix_length != args.history_tokens:
        raise RuntimeError("prefill cache has the wrong sequence length")

    branch_seed_ids = [
        int(value)
        for value in torch.topk(
            previous_logits.reshape(-1),
            k=args.branch_count,
        ).indices.tolist()
    ]
    qk_prebuild: dict[str, Any] = {}
    value_prebuild: dict[str, Any] = {}
    value_install: dict[str, Any] = {}
    branches: list[dict[str, Any]] = []
    rewind_records: list[dict[str, int]] = []
    persistent_snapshots: list[dict[str, Any]] = []
    method_name = "direct_countcap" if sparse else "full_attention"

    with direct.sparse_context(args, method_name):
        if sparse:
            seed_packed_qmse_prefill_queries(prefill_queries)
        sync(input_device)
        prebuild_start = time.perf_counter()
        if sparse:
            qk_prebuild = direct.sparse_attention.precompute_active_packed_qmse_qk_factors(
                cache,
                max_workers=int(os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "12")),
            )
            value_prebuild = prebuild_resident_value_sketch_cache(
                cache,
                model,
                max_workers=max(
                    1,
                    int(os.environ.get("QKSIEVE_PARALLEL_VALUE_WORKERS", "12")),
                ),
            )
            value_install = install_resident_value_sketch_cache(cache)
        sync(input_device)
        prebuild_wall_seconds = time.perf_counter() - prebuild_start

        for branch_index, seed_token in enumerate(branch_seed_ids):
            if branch_index:
                rewind_records.append(
                    crop_for_next_branch(cache, prefix_length, sparse)
                )
            branch = run_branch(
                model,
                cache,
                seed_token,
                prefix_length,
                args.branch_steps,
                input_device,
            )
            branch["branch_index"] = branch_index
            branches.append(branch)
            if sparse:
                persistent_snapshots.append(
                    active_qksieve_persistent_state_signature()
                )

        rewind_records.append(crop_for_next_branch(cache, prefix_length, sparse))
        repeated_branch = run_branch(
            model,
            cache,
            branch_seed_ids[0],
            prefix_length,
            args.branch_steps,
            input_device,
        )
        repeated_branch["branch_index"] = "repeat_0"
        branches.append(repeated_branch)
        if sparse:
            persistent_snapshots.append(
                active_qksieve_persistent_state_signature()
            )

        rewind_records.append(crop_for_next_branch(cache, prefix_length, sparse))
        append_only = run_branch(
            model,
            cache,
            branch_seed_ids[0],
            prefix_length,
            args.append_steps,
            input_device,
        )
        if sparse:
            persistent_snapshots.append(
                active_qksieve_persistent_state_signature()
            )

    reuse_tokens_equal = (
        branches[0]["generated_token_ids"]
        == repeated_branch["generated_token_ids"]
    )
    warm_branches = branches[1:]
    shared_prefix_warm_mean_ms = statistics.fmean(
        branch["mean_ms_per_token"] for branch in warm_branches
    )
    all_branch_decode_seconds = sum(
        float(branch["wall_seconds"]) for branch in branches[:-1]
    )
    stable_signatures = [
        stable_index_signature(snapshot) for snapshot in persistent_snapshots
    ]
    index_buffers_reused = (
        not sparse
        or all(
            signature == stable_signatures[0]
            for signature in stable_signatures[1:]
        )
    )
    result = {
        "schema": "qksieve_persistent_kv_lifecycle_v1",
        "method": args.method,
        "score_mode": score_mode,
        "history_tokens": args.history_tokens,
        "branch_count": args.branch_count,
        "branch_steps": args.branch_steps,
        "append_steps": args.append_steps,
        "budget_policy": (
            "min(1280, max(256, ceil(0.06 * history_tokens)))"
            if sparse
            else "full_history"
        ),
        "prefill_wall_seconds": prefill_wall_seconds,
        "measured_prefill_seconds": measured_prefill_seconds,
        "prebuild_wall_seconds": prebuild_wall_seconds,
        "cold_persistent_request_seconds": (
            prebuild_wall_seconds + float(branches[0]["wall_seconds"])
        ),
        "cold_persistent_request_ms_per_token": 1000.0
        * (prebuild_wall_seconds + float(branches[0]["wall_seconds"]))
        / args.branch_steps,
        "shared_prefix_warm_mean_ms_per_token": shared_prefix_warm_mean_ms,
        "shared_prefix_amortized_ms_per_token": 1000.0
        * (prebuild_wall_seconds + all_branch_decode_seconds)
        / (args.branch_count * args.branch_steps),
        "append_only_ms_per_token": append_only["mean_ms_per_token"],
        "reuse_tokens_equal": reuse_tokens_equal,
        "reuse_hash_equal": (
            branches[0]["generated_token_sha256"]
            == repeated_branch["generated_token_sha256"]
        ),
        "index_buffers_reused_without_rebuild": index_buffers_reused,
        "branch_seed_token_ids": branch_seed_ids,
        "branches": branches,
        "append_only": append_only,
        "rewinds": rewind_records,
        "persistent_state_snapshots": persistent_snapshots,
        "qk_prebuild": qk_prebuild,
        "value_prebuild": value_prebuild,
        "value_install": value_install,
        "qmse_rate_table_preload": rate_table_timings,
        "value_sketch_tail_alpha": (
            float(os.environ["QKSIEVE_VALUE_SKETCH_TAIL_ALPHA"])
            if sparse
            else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(input_device)
            if input_device.type == "cuda"
            else "cpu"
        ),
        "model_name_or_path": args.model_name_or_path,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not reuse_tokens_equal:
        raise RuntimeError("shared-prefix rewind changed repeated-branch tokens")
    if not index_buffers_reused:
        raise RuntimeError("persistent branch execution rebuilt an index buffer")


if __name__ == "__main__":
    main()
