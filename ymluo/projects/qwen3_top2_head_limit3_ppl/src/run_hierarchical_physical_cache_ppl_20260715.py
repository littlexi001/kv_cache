from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_controlled_public_kv_benchmark_v1 as lb
from run_direct_countcap_denseprompt_ppl_20260725 import (
    MIXED_TOPIC_POOLS,
    encode_evaluation_stream,
)
from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    hierarchical_attention_mode,
)
from offloaded_prefill_cache_20260716 import (
    OffloadedExactPrefillCache,
    QuantizedOffloadedExactPrefillCache,
)
from run_critical_position_budget_probe_20260715 import (
    causal_logit_features,
    run_one_token,
    token_shape,
)
from run_head_top2_targeted_ppl_20260714 import (
    install_llama_head_top_fraction_patch,
    load_model,
    prefill_query_tail_mode,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    make_bundle,
)


def parse_fixed_bit_allocation(spec: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in spec.split(","))
    if len(values) != 8:
        raise argparse.ArgumentTypeError(
            "fixed bit allocation must contain eight comma-separated bands"
        )
    if any(bits not in {0, 1, 2, 4, 8} for bits in values):
        raise argparse.ArgumentTypeError(
            "fixed bit allocation values must be selected from 0,1,2,4,8"
        )
    if not any(values):
        raise argparse.ArgumentTypeError(
            "fixed bit allocation cannot be all zero"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topic",
        choices=sorted(set(TOPICS) | set(MIXED_TOPIC_POOLS)),
        default="religion",
    )
    parser.add_argument("--history_tokens", type=int, default=32000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--window_stride_tokens", type=int, default=128_512)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--index_mode",
        choices=("pca_fixed", "qk_variable"),
        default="pca_fixed",
    )
    parser.add_argument(
        "--qk_metric_query_shrinkage",
        type=float,
        default=0.75,
    )
    parser.add_argument("--variable_rate_budget", type=int, default=15)
    parser.add_argument(
        "--fixed_bit_allocation",
        type=parse_fixed_bit_allocation,
    )
    parser.add_argument("--candidate_fraction", type=float, default=0.02)
    parser.add_argument("--candidate_min_tokens", type=int, default=1)
    parser.add_argument("--candidate_max_tokens", type=int)
    parser.add_argument(
        "--retrieval_backend",
        choices=("full_topk", "sampled_compact"),
        default="full_topk",
    )
    parser.add_argument(
        "--sampled_candidate_multiplier",
        type=float,
        default=1.5,
    )
    parser.add_argument("--attention_fraction", type=float)
    parser.add_argument(
        "--selection_mode",
        choices=("shared_sum", "shared_max", "head_balanced"),
        default="shared_sum",
    )
    parser.add_argument(
        "--candidate_selection_mode",
        choices=(
            "shared_sum",
            "head_balanced",
            "per_head",
            "per_head_union",
            "per_head_stream",
        ),
    )
    parser.add_argument(
        "--rerank_selection_mode",
        choices=("shared_sum", "shared_max", "head_balanced"),
    )
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--stream_group_size", type=int, default=1)
    parser.add_argument("--candidate_refresh_interval", type=int, default=1)
    parser.add_argument(
        "--host_append_mode", choices=("async", "sync"), default="async"
    )
    parser.add_argument(
        "--conversion_mode", choices=("async", "sync"), default="async"
    )
    parser.add_argument("--recent_fraction", type=float, default=0.0)
    parser.add_argument("--debug_directory", action="store_true")
    parser.add_argument("--collect_router_features", action="store_true")
    parser.add_argument("--record_candidate_overlap", action="store_true")
    parser.add_argument(
        "--directory_backend", choices=("sorted", "fused"), default="sorted"
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--prefill_cache_mode",
        choices=(
            "dynamic",
            "offloaded_exact",
            "quantized_offloaded_exact",
        ),
        default="dynamic",
    )
    parser.add_argument(
        "--prefill_quantization_bits",
        type=int,
        choices=(4, 8),
        default=4,
    )
    parser.add_argument(
        "--prefill_quantization_group_size",
        type=int,
        default=0,
        help="Dimensions per transient K/V quantization scale; 0 uses one scale per head.",
    )
    parser.add_argument(
        "--prefill_conversion_source",
        choices=("exact_host", "transient_quantized_key"),
        default="exact_host",
        help="Build the final index from exact host K or the transient quantized GPU K.",
    )
    parser.add_argument(
        "--dense_query_before_conversion",
        action="store_true",
    )
    parser.add_argument("--known_reference_ppl", type=float, default=12.341219940643663)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def synchronize_cuda_devices() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory_stats() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)


def empty_cuda_caches() -> None:
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def cuda_memory_by_device(metric: str) -> list[int]:
    function = getattr(torch.cuda, metric)
    return [int(function(device)) for device in range(torch.cuda.device_count())]


def run_synchronized_one_token(
    model: torch.nn.Module,
    token_id: int,
    cache,
    position: int,
    input_device: torch.device,
):
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    started = time.perf_counter()
    cache, logits, _, features = run_one_token(
        model, token_id, cache, position, input_device
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    return cache, logits, time.perf_counter() - started, features


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 < args.query_tokens < args.history_tokens:
        raise ValueError("query_tokens must be in (0, history_tokens)")
    if args.window_index < 0 or args.window_stride_tokens <= 0:
        raise ValueError("window index must be non-negative and stride positive")
    if args.index_mode == "qk_variable":
        install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(
        SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
        )
    )
    window_start = args.window_index * args.window_stride_tokens
    required_tokens = window_start + args.history_tokens + args.eval_tokens
    stream = encode_evaluation_stream(
        tokenizer,
        args.topic,
        required_tokens,
        args.dataset_cache_dir,
        args.seed,
    )
    window_stream = stream[
        window_start : window_start + args.history_tokens + args.eval_tokens
    ]
    history = window_stream[: args.history_tokens]
    history_counts = Counter(int(token_id) for token_id in history)
    remote_ids = history[: -args.query_tokens]
    query_ids = history[-args.query_tokens :]
    target_ids = window_stream[
        args.history_tokens : args.history_tokens + args.eval_tokens
    ]
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
    if torch.cuda.is_available():
        reset_cuda_peak_memory_stats()
    max_new_tokens = args.query_tokens + args.eval_tokens + 8
    dense_query_logits: torch.Tensor | None = None
    captured_queries: dict[int, torch.Tensor] | None = None
    if args.prefill_cache_mode in {
        "offloaded_exact",
        "quantized_offloaded_exact",
    }:
        cache_capacity = len(remote_ids) + max_new_tokens
        source_cache: object
        if args.prefill_cache_mode == "offloaded_exact":
            source_cache = OffloadedExactPrefillCache(
                capacity=cache_capacity
            )
        else:
            source_cache = QuantizedOffloadedExactPrefillCache(
                capacity=cache_capacity,
                bits=args.prefill_quantization_bits,
                group_size=args.prefill_quantization_group_size,
            )
        source_cache, _, prefill_seconds = lb.run_token_segment(
            model,
            bundle.input_ids[:, : bundle.query_start],
            source_cache,
            0,
            input_device,
            args.prefill_chunk_tokens,
        )
    else:
        source_cache, prefill_seconds = lb.prefill_prefix(
            model, bundle, input_device, args.prefill_chunk_tokens
        )
    dense_query_seconds = 0.0
    dense_query_before_conversion = (
        args.index_mode == "qk_variable"
        or args.dense_query_before_conversion
    )
    if dense_query_before_conversion:
        query_tensor = torch.tensor(
            [query_ids],
            dtype=torch.long,
            device=input_device,
        )
        query_capture = (
            prefill_query_tail_mode(8)
            if args.index_mode == "qk_variable"
            else nullcontext(None)
        )
        with query_capture as captured_queries:
            source_cache, dense_query_logits, dense_query_seconds = (
                lb.run_token_segment(
                    model,
                    query_tensor,
                    source_cache,
                    len(remote_ids),
                    input_device,
                    args.prefill_chunk_tokens,
                )
            )
        prefill_seconds += dense_query_seconds
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_started = time.perf_counter()
    conversion_max_new_tokens = (
        args.eval_tokens + 8
        if dense_query_before_conversion
        else max_new_tokens
    )
    conversion_kwargs = dict(
        projection_dim=args.projection_dim,
        index_bits=args.index_bits,
        index_mode=args.index_mode,
        query_tail_by_layer=captured_queries,
        qk_metric_query_shrinkage=args.qk_metric_query_shrinkage,
        variable_rate_budget=args.variable_rate_budget,
        fixed_bit_allocation=args.fixed_bit_allocation,
        candidate_fraction=args.candidate_fraction,
        candidate_min_tokens=args.candidate_min_tokens,
        candidate_max_tokens=args.candidate_max_tokens,
        retrieval_backend=args.retrieval_backend,
        sampled_candidate_multiplier=args.sampled_candidate_multiplier,
        attention_fraction=args.attention_fraction,
        selection_mode=args.selection_mode,
        candidate_selection_mode=args.candidate_selection_mode,
        rerank_selection_mode=args.rerank_selection_mode,
        exact_cache_fraction=args.exact_cache_fraction,
        stream_group_size=args.stream_group_size,
        candidate_refresh_interval=args.candidate_refresh_interval,
        async_host_append=args.host_append_mode == "async",
        async_conversion=args.conversion_mode == "async",
        max_new_tokens=conversion_max_new_tokens,
        directory_backend=args.directory_backend,
        recent_fraction=args.recent_fraction,
        debug_directory=args.debug_directory,
        record_traces=args.record_candidate_overlap,
    )
    if isinstance(
        source_cache,
        (
            OffloadedExactPrefillCache,
            QuantizedOffloadedExactPrefillCache,
        ),
    ):
        if (
            args.prefill_conversion_source == "transient_quantized_key"
            and not isinstance(
                source_cache,
                QuantizedOffloadedExactPrefillCache,
            )
        ):
            raise ValueError(
                "transient_quantized_key conversion requires "
                "quantized_offloaded_exact prefill"
            )
        physical_cache = HierarchicalPCACache.from_offloaded_prefill_cache(
            source_cache,
            use_transient_quantized_key=(
                args.prefill_conversion_source == "transient_quantized_key"
            ),
            **conversion_kwargs,
        )
    else:
        physical_cache = HierarchicalPCACache.from_dynamic_cache(
            source_cache, **conversion_kwargs
        )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_seconds = time.perf_counter() - conversion_started
    del source_cache
    if torch.cuda.is_available():
        empty_cuda_caches()
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
    process_gpu_allocated_after_conversion = sum(allocated_by_device)
    process_gpu_reserved_after_conversion = sum(reserved_by_device)
    process_peak_gpu_allocated = sum(peak_by_device)

    online_seconds = 0.0
    previous_logits: torch.Tensor | None = dense_query_logits
    with hierarchical_attention_mode(model):
        if not dense_query_before_conversion:
            for offset, token_id in enumerate(query_ids):
                physical_cache, previous_logits, elapsed, _ = (
                    run_synchronized_one_token(
                        model,
                        int(token_id),
                        physical_cache,
                        len(remote_ids) + offset,
                        input_device,
                    )
                )
                online_seconds += elapsed
        if previous_logits is None:
            raise RuntimeError("query warm-up produced no logits")

        nll_sum = 0.0
        token_nll = []
        router_features: list[dict[str, object]] = []
        current_input_id = int(query_ids[-1])
        for offset, label_id in enumerate(target_ids):
            if args.collect_router_features:
                causal_features = causal_logit_features(previous_logits)
                top1_id = int(causal_features["top1_id"])
                top1_text = tokenizer.decode(
                    [top1_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                input_text = tokenizer.decode(
                    [current_input_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                router_features.append(
                    {
                        "target_index": offset,
                        "prediction_position": args.history_tokens + offset,
                        "history_tokens": args.history_tokens,
                        **causal_features,
                        "top1_text": top1_text,
                        "top1_history_frequency": history_counts[top1_id],
                        **{
                            f"top1_{key}": value
                            for key, value in token_shape(top1_text).items()
                        },
                        "input_id": current_input_id,
                        "input_text": input_text,
                        "input_history_frequency": history_counts[current_input_id],
                        **{
                            f"input_{key}": value
                            for key, value in token_shape(input_text).items()
                        },
                    }
                )
            log_probs = F.log_softmax(previous_logits[0].float(), dim=-1)
            nll = -float(log_probs[int(label_id)].item())
            nll_sum += nll
            token_nll.append(nll)
            if offset + 1 < len(target_ids):
                physical_cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                    model,
                    int(label_id),
                    physical_cache,
                    args.history_tokens + offset,
                    input_device,
                )
                online_seconds += elapsed
                current_input_id = int(label_id)

    mean_nll = nll_sum / len(target_ids)
    ppl = math.exp(mean_nll)
    initial_full_bytes = physical_cache.original_gpu_bytes
    initial_length = physical_cache.states[0].initial_length
    capacity = physical_cache.states[0].capacity
    final_length = physical_cache.get_seq_length()
    bytes_per_token = initial_full_bytes / initial_length
    persistent_bytes = physical_cache.persistent_gpu_bytes()
    result = {
        "topic": args.topic,
        "window_index": args.window_index,
        "window_start_token": window_start,
        "window_stride_tokens": args.window_stride_tokens,
        "history_tokens": args.history_tokens,
        "remote_tokens": len(remote_ids),
        "query_tokens": len(query_ids),
        "eval_tokens": len(target_ids),
        "final_cache_length": final_length,
        "allocated_capacity": capacity,
        "index_mode": args.index_mode,
        "projection_dim": physical_cache.projection_dim,
        "index_bits": args.index_bits,
        "qk_metric_query_shrinkage": args.qk_metric_query_shrinkage,
        "variable_rate_budget": args.variable_rate_budget,
        "fixed_bit_allocation": args.fixed_bit_allocation,
        "candidate_fraction": args.candidate_fraction,
        "candidate_min_tokens": args.candidate_min_tokens,
        "candidate_max_tokens": args.candidate_max_tokens,
        "retrieval_backend": args.retrieval_backend,
        "sampled_candidate_multiplier": args.sampled_candidate_multiplier,
        "attention_fraction": (
            args.candidate_fraction
            if args.attention_fraction is None
            else args.attention_fraction
        ),
        "candidate_selection_mode": (
            args.selection_mode
            if args.candidate_selection_mode is None
            else args.candidate_selection_mode
        ),
        "rerank_selection_mode": (
            args.selection_mode
            if args.rerank_selection_mode is None
            else args.rerank_selection_mode
        ),
        "exact_cache_fraction": args.exact_cache_fraction,
        "stream_group_size": args.stream_group_size,
        "candidate_refresh_interval": args.candidate_refresh_interval,
        "record_candidate_overlap": args.record_candidate_overlap,
        "host_append_mode": args.host_append_mode,
        "conversion_mode": args.conversion_mode,
        "prefill_cache_mode": args.prefill_cache_mode,
        "prefill_quantization_bits": (
            args.prefill_quantization_bits
            if args.prefill_cache_mode == "quantized_offloaded_exact"
            else None
        ),
        "prefill_quantization_group_size": (
            args.prefill_quantization_group_size
            if args.prefill_cache_mode == "quantized_offloaded_exact"
            else None
        ),
        "prefill_conversion_source": args.prefill_conversion_source,
        "dense_query_before_conversion": dense_query_before_conversion,
        "stream_physical_mode": (
            "direct_flatten"
            if args.candidate_selection_mode == "per_head_stream"
            else None
        ),
        "recent_fraction": args.recent_fraction,
        "directory_backend": args.directory_backend,
        "nll": mean_nll,
        "ppl": ppl,
        "known_reference_ppl": args.known_reference_ppl,
        "physical_over_known_reference_ppl": ppl / args.known_reference_ppl,
        "prefill_seconds": prefill_seconds,
        "dense_query_seconds": dense_query_seconds,
        "cache_conversion_seconds": conversion_seconds,
        "prefill_plus_conversion_seconds": prefill_seconds + conversion_seconds,
        "online_seconds": online_seconds,
        "synchronized_model_forward_seconds": online_seconds,
        "timing_is_synchronized_per_token": True,
        "mean_cache_hit_rate": physical_cache.mean_cache_hit_rate(),
        "mean_sampled_candidate_count": (
            physical_cache.mean_sampled_candidate_count()
        ),
        "mean_sampled_overflow_rate": (
            physical_cache.mean_sampled_overflow_rate()
        ),
        "mean_sampled_clipped_fraction": (
            physical_cache.mean_sampled_clipped_fraction()
        ),
        "mean_candidate_union_fraction": (
            physical_cache.mean_candidate_union_fraction()
        ),
        "max_candidate_union_fraction": physical_cache.max_candidate_union_fraction(),
        "original_remote_full_gpu_kv_bytes": initial_full_bytes,
        "hierarchical_persistent_gpu_bytes": persistent_bytes,
        "hierarchical_over_initial_remote_full_kv": persistent_bytes
        / initial_full_bytes,
        "hierarchical_over_capacity_equivalent_full_kv": persistent_bytes
        / (bytes_per_token * capacity),
        "hierarchical_over_final_length_full_kv": persistent_bytes
        / (bytes_per_token * final_length),
        "pinned_host_bytes": physical_cache.pinned_host_bytes(),
        "process_gpu_allocated_after_conversion": process_gpu_allocated_after_conversion,
        "process_gpu_reserved_after_conversion": process_gpu_reserved_after_conversion,
        "process_peak_gpu_allocated_during_prefill_conversion": process_peak_gpu_allocated,
        "process_gpu_allocated_by_device_after_conversion": allocated_by_device,
        "process_gpu_reserved_by_device_after_conversion": reserved_by_device,
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
        "causal_router_features": router_features,
        "token_nll": token_nll,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact_result = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "causal_router_features",
            "target_token_ids",
            "target_token_texts",
            "token_nll",
        }
    }
    print(json.dumps(compact_result, sort_keys=True))


if __name__ == "__main__":
    main()
