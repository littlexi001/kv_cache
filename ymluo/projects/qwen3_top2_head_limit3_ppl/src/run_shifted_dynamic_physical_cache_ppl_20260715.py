from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter
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
from run_critical_position_budget_probe_20260715 import (
    causal_logit_features,
    token_shape,
)
from run_head_top2_targeted_ppl_20260714 import load_model
from run_hierarchical_physical_cache_ppl_20260715 import (
    cuda_memory_by_device,
    empty_cuda_caches,
    reset_cuda_peak_memory_stats,
    run_synchronized_one_token,
    synchronize_cuda_devices,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    encode_topic_stream,
    make_bundle,
)
from train_shifted_physical_budget_router_20260715 import (
    FEATURE_NAMES,
    shifted_feature_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a shifted-causal 1%/1.5% physical KV budget router."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--router_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", choices=sorted(TOPICS), default="computer")
    parser.add_argument("--history_tokens", type=int, default=128000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--window_stride_tokens", type=int, default=128_512)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--candidate_refresh_interval", type=int, default=1)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--known_reference_ppl", type=float, default=0.0)
    parser.add_argument(
        "--query_action", choices=("low", "mid", "high"), default="low"
    )
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def load_router(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if list(artifact.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("router feature schema does not match runtime schema")
    if not artifact.get("shifted_causal", False):
        raise ValueError("router artifact is not shifted-causal")
    if (
        artifact.get("counterfactual_onpolicy_teacher", False)
        and artifact.get("teacher_reference") != "full_kv_nll"
    ):
        raise ValueError("counterfactual router labels are not FullKV-referenced")
    return artifact


def previous_logit_features(
    tokenizer: Any,
    logits: torch.Tensor,
    history_counts: Counter[int],
    retrieval_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    features = causal_logit_features(logits)
    retrieval_features = retrieval_features or {}
    top1_id = int(features["top1_id"])
    top1_text = tokenizer.decode(
        [top1_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        **features,
        "retrieval_feature_valid": float(
            retrieval_features.get("retrieval_feature_valid", 0.0)
        ),
        "retrieval_score_spread": float(
            retrieval_features.get("retrieval_score_spread", 0.0)
        ),
        "retrieval_candidate_stability": float(
            retrieval_features.get("retrieval_candidate_stability", 0.0)
        ),
        "retrieval_refreshed_fraction": float(
            retrieval_features.get("retrieval_refreshed_fraction", 0.0)
        ),
        "top1_history_frequency": history_counts[top1_id],
        **{
            f"top1_{key}": value
            for key, value in token_shape(top1_text).items()
        },
    }


def current_input_features(
    tokenizer: Any,
    input_id: int,
    history_counts: Counter[int],
    history_tokens: int,
) -> dict[str, Any]:
    text = tokenizer.decode(
        [input_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "history_tokens": history_tokens,
        "input_history_frequency": history_counts[input_id],
        **{
            f"input_{key}": value
            for key, value in token_shape(text).items()
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    artifact = load_router(args.router_path)
    if args.window_index < 0 or args.window_stride_tokens <= 0:
        raise ValueError("window index must be non-negative and stride positive")
    low_fraction = float(artifact["low_fraction"])
    three_action = bool(artifact.get("ordinal_three_action", False))
    if three_action:
        mid_fraction = float(artifact["mid_fraction"])
        high_fraction = float(artifact["high_fraction"])
        low_stream_group_size = int(artifact["low_stream_group_size"])
        mid_stream_group_size = int(artifact["mid_stream_group_size"])
        high_stream_group_size = int(artifact["high_stream_group_size"])
        construction_stream_group_size = high_stream_group_size
    else:
        mid_fraction = None
        high_fraction = float(artifact["high_fraction"])
        low_stream_group_size = int(artifact["stream_group_size"])
        mid_stream_group_size = low_stream_group_size
        high_stream_group_size = low_stream_group_size
        construction_stream_group_size = low_stream_group_size
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
    history_counts = Counter(int(token_id) for token_id in history)
    remote_ids = history[: -args.query_tokens]
    query_ids = history[-args.query_tokens :]
    target_ids = window_stream[
        args.history_tokens : args.history_tokens + args.eval_tokens
    ]
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
    if torch.cuda.is_available():
        reset_cuda_peak_memory_stats()
    source_cache, prefill_seconds = lb.prefill_prefix(
        model, bundle, input_device, args.prefill_chunk_tokens
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_started = time.perf_counter()
    cache = HierarchicalPCACache.from_dynamic_cache(
        source_cache,
        projection_dim=args.projection_dim,
        index_bits=args.index_bits,
        candidate_fraction=high_fraction,
        attention_fraction=high_fraction,
        exact_cache_fraction=args.exact_cache_fraction,
        max_new_tokens=args.query_tokens + args.eval_tokens + 8,
        candidate_selection_mode="per_head_stream",
        stream_group_size=construction_stream_group_size,
        candidate_refresh_interval=args.candidate_refresh_interval,
        directory_backend="fused",
        collect_retrieval_features=True,
    )
    if args.query_action == "high":
        query_fraction = high_fraction
        query_stream_group_size = high_stream_group_size
    elif args.query_action == "mid":
        query_fraction = (
            float(mid_fraction) if mid_fraction is not None else high_fraction
        )
        query_stream_group_size = mid_stream_group_size
    else:
        query_fraction = low_fraction
        query_stream_group_size = low_stream_group_size
    cache.set_runtime_action(
        candidate_fraction=query_fraction,
        stream_group_size=query_stream_group_size,
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

    model_forward_seconds = 0.0
    router_seconds = 0.0
    previous_logits: torch.Tensor | None = None
    token_nll: list[float] = []
    action_fractions: list[float] = []
    action_stream_group_sizes: list[int] = []
    risk_scores: list[float | None] = []
    with hierarchical_attention_mode(model):
        for offset, token_id in enumerate(query_ids):
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                int(token_id),
                cache,
                len(remote_ids) + offset,
                input_device,
            )
            model_forward_seconds += elapsed
        if previous_logits is None:
            raise RuntimeError("query warm-up produced no logits")

        action_fractions.append(query_fraction)
        action_stream_group_sizes.append(query_stream_group_size)
        risk_scores.append(None)
        for target_index, label_id in enumerate(target_ids):
            log_probs = F.log_softmax(previous_logits[0].float(), dim=-1)
            token_nll.append(-float(log_probs[int(label_id)].item()))
            if target_index + 1 >= len(target_ids):
                continue
            router_started = time.perf_counter()
            vector = shifted_feature_vector(
                previous_logit_features(
                    tokenizer,
                    previous_logits,
                    history_counts,
                    cache.retrieval_features(),
                ),
                current_input_features(
                    tokenizer,
                    int(label_id),
                    history_counts,
                    args.history_tokens,
                ),
                target_index + 1,
                len(target_ids),
            )
            risk = float(artifact["model"].predict([vector])[0])
            if three_action and risk >= float(artifact["high_threshold"]):
                selected_fraction = high_fraction
                selected_stream_group_size = high_stream_group_size
            elif three_action and risk >= float(artifact["mid_threshold"]):
                selected_fraction = float(mid_fraction)
                selected_stream_group_size = mid_stream_group_size
            elif not three_action and risk >= float(artifact["threshold"]):
                selected_fraction = high_fraction
                selected_stream_group_size = high_stream_group_size
            else:
                selected_fraction = low_fraction
                selected_stream_group_size = low_stream_group_size
            cache.set_runtime_action(
                candidate_fraction=selected_fraction,
                stream_group_size=selected_stream_group_size,
            )
            router_seconds += time.perf_counter() - router_started
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                int(label_id),
                cache,
                args.history_tokens + target_index,
                input_device,
            )
            model_forward_seconds += elapsed
            action_fractions.append(selected_fraction)
            action_stream_group_sizes.append(selected_stream_group_size)
            risk_scores.append(risk)

    mean_nll = sum(token_nll) / len(token_nll)
    persistent_bytes = cache.persistent_gpu_bytes()
    bytes_per_token = cache.original_gpu_bytes / len(remote_ids)
    final_length = cache.get_seq_length()
    result = {
        "status": "real_shifted_causal_mixed_trajectory",
        "topic": args.topic,
        "window_index": args.window_index,
        "window_start_token": window_start,
        "window_stride_tokens": args.window_stride_tokens,
        "history_tokens": args.history_tokens,
        "query_tokens": len(query_ids),
        "eval_tokens": len(target_ids),
        "router_path": str(args.router_path),
        "router_threshold": (
            None if three_action else float(artifact["threshold"])
        ),
        "router_mid_threshold": (
            float(artifact["mid_threshold"]) if three_action else None
        ),
        "router_high_threshold": (
            float(artifact["high_threshold"]) if three_action else None
        ),
        "low_fraction": low_fraction,
        "mid_fraction": mid_fraction,
        "high_fraction": high_fraction,
        "query_action": args.query_action,
        "query_fraction": query_fraction,
        "candidate_refresh_interval": args.candidate_refresh_interval,
        "mid_action_count": (
            0
            if mid_fraction is None
            else sum(fraction == mid_fraction for fraction in action_fractions)
        ),
        "high_action_count": sum(
            fraction == high_fraction for fraction in action_fractions
        ),
        "high_action_rate": sum(
            fraction == high_fraction for fraction in action_fractions
        )
        / len(action_fractions),
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "known_reference_ppl": args.known_reference_ppl,
        "quality_retention": (
            args.known_reference_ppl / math.exp(mean_nll)
            if args.known_reference_ppl > 0
            else None
        ),
        "prefill_seconds": prefill_seconds,
        "cache_conversion_seconds": conversion_seconds,
        "model_forward_seconds": model_forward_seconds,
        "router_seconds": router_seconds,
        "online_seconds": model_forward_seconds + router_seconds,
        "hierarchical_persistent_gpu_bytes": persistent_bytes,
        "hierarchical_over_final_length_full_kv": persistent_bytes
        / (bytes_per_token * final_length),
        "mean_cache_hit_rate": cache.mean_cache_hit_rate(),
        "process_gpu_allocated_after_conversion": sum(allocated_by_device),
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
        "action_fractions": action_fractions,
        "action_stream_group_sizes": action_stream_group_sizes,
        "risk_scores": risk_scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "action_fractions",
            "action_stream_group_sizes",
            "risk_scores",
            "target_token_ids",
            "target_token_texts",
            "token_nll",
        }
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
