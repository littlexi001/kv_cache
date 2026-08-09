from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

import run_controlled_public_kv_benchmark_v1 as lb
from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    hierarchical_attention_mode,
)
from run_head_top2_targeted_ppl_20260714 import load_model
from run_hierarchical_physical_cache_ppl_20260715 import (
    cuda_memory_by_device,
    empty_cuda_caches,
    reset_cuda_peak_memory_stats,
    synchronize_cuda_devices,
)
from run_multitopic_lpcm_ppl_20260714 import TOPICS, encode_topic_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="True same-cache batch throughput for FullKV and hierarchical PCA KV."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--topics", default=",".join(TOPICS))
    parser.add_argument("--history_tokens", type=int, default=32000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--candidate_fraction", type=float, default=0.015)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--stream_group_size", type=int, default=2)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn"
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def cache_bytes(cache: Any) -> int:
    return sum(
        (key.numel() + value.numel()) * key.element_size()
        for key, value in zip(cache.key_cache, cache.value_cache)
    )


@torch.inference_mode()
def batched_step(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    cache: Any,
    position: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor, float]:
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    started = time.perf_counter()
    outputs = lb.model_forward(
        model,
        {
            "input_ids": token_ids.reshape(-1, 1).to(input_device),
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.tensor([position], device=input_device),
        },
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    return (
        outputs.past_key_values,
        outputs.logits[:, -1, :].detach(),
        time.perf_counter() - started,
    )


@torch.inference_mode()
def evaluate_cache(
    model: torch.nn.Module,
    cache: Any,
    query_ids: torch.Tensor,
    target_ids: torch.Tensor,
    remote_length: int,
    input_device: torch.device,
) -> tuple[Any, list[float], float]:
    previous_logits: torch.Tensor | None = None
    forward_seconds = 0.0
    for offset in range(query_ids.shape[1]):
        cache, previous_logits, elapsed = batched_step(
            model,
            query_ids[:, offset],
            cache,
            remote_length + offset,
            input_device,
        )
        forward_seconds += elapsed
    assert previous_logits is not None
    nll_sums = torch.zeros(query_ids.shape[0], dtype=torch.float64)
    for offset in range(target_ids.shape[1]):
        labels = target_ids[:, offset].to(previous_logits.device)
        nll_sums += F.cross_entropy(
            previous_logits.float(), labels, reduction="none"
        ).double().cpu()
        if offset + 1 < target_ids.shape[1]:
            cache, previous_logits, elapsed = batched_step(
                model,
                target_ids[:, offset],
                cache,
                remote_length + query_ids.shape[1] + offset,
                input_device,
            )
            forward_seconds += elapsed
    return (
        cache,
        (nll_sums / target_ids.shape[1]).tolist(),
        forward_seconds,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    topics = [item.strip() for item in args.topics.split(",") if item.strip()]
    if len(topics) < args.batch_size or any(topic not in TOPICS for topic in topics):
        raise ValueError("topics must contain at least batch_size known topics")
    tokenizer, model, input_device = load_model(
        SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
        )
    )

    remote_rows = []
    query_rows = []
    target_rows = []
    selected_topics = topics[: args.batch_size]
    for index, topic in enumerate(selected_topics):
        stream = encode_topic_stream(
            tokenizer,
            TOPICS[topic],
            args.history_tokens + args.eval_tokens,
            args.dataset_cache_dir,
            args.seed + index,
        )
        history = stream[: args.history_tokens]
        remote_rows.append(history[: -args.query_tokens])
        query_rows.append(history[-args.query_tokens :])
        target_rows.append(
            stream[args.history_tokens : args.history_tokens + args.eval_tokens]
        )
    remote_ids = torch.tensor(remote_rows, dtype=torch.long)
    query_ids = torch.tensor(query_rows, dtype=torch.long)
    target_ids = torch.tensor(target_rows, dtype=torch.long)
    remote_length = remote_ids.shape[1]
    decode_steps = args.query_tokens + args.eval_tokens - 1

    if torch.cuda.is_available():
        reset_cuda_peak_memory_stats()
    full_cache, _, full_prefill_seconds = lb.run_token_segment(
        model,
        remote_ids,
        None,
        0,
        input_device,
        args.prefill_chunk_tokens,
    )
    full_initial_bytes = cache_bytes(full_cache)
    full_cache, full_nll, full_forward_seconds = evaluate_cache(
        model,
        full_cache,
        query_ids,
        target_ids,
        remote_length,
        input_device,
    )
    full_peak = (
        sum(cuda_memory_by_device("max_memory_allocated"))
        if torch.cuda.is_available()
        else 0
    )
    del full_cache
    if torch.cuda.is_available():
        empty_cuda_caches()
        reset_cuda_peak_memory_stats()

    source_cache, _, sparse_prefill_seconds = lb.run_token_segment(
        model,
        remote_ids,
        None,
        0,
        input_device,
        args.prefill_chunk_tokens,
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_started = time.perf_counter()
    sparse_cache = HierarchicalPCACache.from_dynamic_cache(
        source_cache,
        projection_dim=args.projection_dim,
        index_bits=args.index_bits,
        candidate_fraction=args.candidate_fraction,
        attention_fraction=args.candidate_fraction,
        exact_cache_fraction=args.exact_cache_fraction,
        max_new_tokens=args.query_tokens + args.eval_tokens + 8,
        candidate_selection_mode="per_head_stream",
        stream_group_size=args.stream_group_size,
        directory_backend="fused",
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_seconds = time.perf_counter() - conversion_started
    del source_cache
    if torch.cuda.is_available():
        empty_cuda_caches()
    with hierarchical_attention_mode(model):
        sparse_cache, sparse_nll, sparse_forward_seconds = evaluate_cache(
            model,
            sparse_cache,
            query_ids,
            target_ids,
            remote_length,
            input_device,
        )
    sparse_peak = (
        sum(cuda_memory_by_device("max_memory_allocated"))
        if torch.cuda.is_available()
        else 0
    )
    final_length = sparse_cache.get_seq_length()
    bytes_per_token = sparse_cache.original_gpu_bytes / remote_length
    sparse_bytes = sparse_cache.persistent_gpu_bytes()
    result = {
        "status": "true_same_cache_batch_throughput",
        "batch_size": args.batch_size,
        "topics": selected_topics,
        "history_tokens": args.history_tokens,
        "query_tokens": args.query_tokens,
        "eval_tokens": args.eval_tokens,
        "decode_steps_per_sequence": decode_steps,
        "full_mean_ppl": math.exp(sum(full_nll) / len(full_nll)),
        "sparse_mean_ppl": math.exp(sum(sparse_nll) / len(sparse_nll)),
        "full_nll_by_sequence": full_nll,
        "sparse_nll_by_sequence": sparse_nll,
        "full_prefill_seconds": full_prefill_seconds,
        "sparse_prefill_seconds": sparse_prefill_seconds,
        "sparse_conversion_seconds": conversion_seconds,
        "full_forward_seconds": full_forward_seconds,
        "sparse_forward_seconds": sparse_forward_seconds,
        "full_tokens_per_second": args.batch_size
        * decode_steps
        / full_forward_seconds,
        "sparse_tokens_per_second": args.batch_size
        * decode_steps
        / sparse_forward_seconds,
        "throughput_speedup": full_forward_seconds / sparse_forward_seconds,
        "full_initial_gpu_kv_bytes": full_initial_bytes,
        "sparse_persistent_gpu_bytes": sparse_bytes,
        "sparse_over_final_full_kv": sparse_bytes
        / (bytes_per_token * final_length),
        "full_peak_gpu_allocated_bytes": full_peak,
        "sparse_peak_gpu_allocated_bytes": sparse_peak,
        "mean_cache_hit_rate": sparse_cache.mean_cache_hit_rate(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
