from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_controlled_public_kv_benchmark_v1 as lb
from run_head_top2_targeted_ppl_20260714 import load_model
from run_hierarchical_physical_cache_ppl_20260715 import (
    cuda_memory_by_device,
    reset_cuda_peak_memory_stats,
    run_synchronized_one_token,
)
from run_multitopic_lpcm_ppl_20260714 import TOPICS, encode_topic_stream, make_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", choices=sorted(TOPICS), default="religion")
    parser.add_argument("--history_tokens", type=int, default=32000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--window_stride_tokens", type=int, default=128_512)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def cache_bytes(cache) -> int:
    return sum(
        (key.numel() + value.numel()) * key.element_size()
        for key, value in zip(cache.key_cache, cache.value_cache)
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 < args.query_tokens < args.history_tokens:
        raise ValueError("query_tokens must be in (0, history_tokens)")
    if args.window_index < 0 or args.window_stride_tokens <= 0:
        raise ValueError("window index must be non-negative and stride positive")
    tokenizer, model, input_device = load_model(
        SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
        )
    )
    window_start = args.window_index * args.window_stride_tokens
    stream = encode_topic_stream(
        tokenizer,
        TOPICS[args.topic],
        window_start + args.history_tokens + args.eval_tokens,
        args.dataset_cache_dir,
        args.seed,
    )
    window_stream = stream[
        window_start : window_start + args.history_tokens + args.eval_tokens
    ]
    history = window_stream[: args.history_tokens]
    remote_ids = history[: -args.query_tokens]
    query_ids = history[-args.query_tokens :]
    target_ids = window_stream[
        args.history_tokens : args.history_tokens + args.eval_tokens
    ]
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
    if torch.cuda.is_available():
        reset_cuda_peak_memory_stats()
    cache, prefill_seconds = lb.prefill_prefix(
        model, bundle, input_device, args.prefill_chunk_tokens
    )
    initial_bytes = cache_bytes(cache)

    online_seconds = 0.0
    previous_logits: torch.Tensor | None = None
    for offset, token_id in enumerate(query_ids):
        cache, previous_logits, elapsed, _ = run_synchronized_one_token(
            model,
            int(token_id),
            cache,
            len(remote_ids) + offset,
            input_device,
        )
        online_seconds += elapsed
    if previous_logits is None:
        raise RuntimeError("query warm-up produced no logits")

    nll_sum = 0.0
    token_nll = []
    for offset, label_id in enumerate(target_ids):
        log_probs = F.log_softmax(previous_logits[0].float(), dim=-1)
        nll = -float(log_probs[int(label_id)].item())
        nll_sum += nll
        token_nll.append(nll)
        if offset + 1 < len(target_ids):
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                int(label_id),
                cache,
                args.history_tokens + offset,
                input_device,
            )
            online_seconds += elapsed

    mean_nll = nll_sum / len(target_ids)
    allocated_by_device = (
        cuda_memory_by_device("memory_allocated") if torch.cuda.is_available() else []
    )
    reserved_by_device = (
        cuda_memory_by_device("memory_reserved") if torch.cuda.is_available() else []
    )
    peak_by_device = (
        cuda_memory_by_device("max_memory_allocated")
        if torch.cuda.is_available()
        else []
    )
    result = {
        "cache_backend": "full_gpu_dynamic_cache",
        "attention_implementation": model.config._attn_implementation,
        "topic": args.topic,
        "window_index": args.window_index,
        "window_start_token": window_start,
        "window_stride_tokens": args.window_stride_tokens,
        "history_tokens": args.history_tokens,
        "remote_tokens": len(remote_ids),
        "query_tokens": len(query_ids),
        "eval_tokens": len(target_ids),
        "final_cache_length": cache.get_seq_length(),
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "prefill_seconds": prefill_seconds,
        "synchronized_model_forward_seconds": online_seconds,
        "initial_gpu_kv_bytes": initial_bytes,
        "final_gpu_kv_bytes": cache_bytes(cache),
        "process_gpu_allocated_after_decode": sum(allocated_by_device),
        "process_gpu_reserved_after_decode": sum(reserved_by_device),
        "process_peak_gpu_allocated_during_prefill_decode": sum(peak_by_device),
        "process_gpu_allocated_by_device_after_decode": allocated_by_device,
        "process_gpu_reserved_by_device_after_decode": reserved_by_device,
        "process_peak_gpu_allocated_by_device": peak_by_device,
        "target_token_ids": [int(token_id) for token_id in target_ids],
        "target_token_texts": [
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in target_ids
        ],
        "token_nll": token_nll,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact_result = {
        key: value
        for key, value in result.items()
        if key not in {"target_token_ids", "target_token_texts", "token_nll"}
    }
    print(json.dumps(compact_result, sort_keys=True))


if __name__ == "__main__":
    main()
