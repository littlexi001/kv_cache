from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_controlled_public_kv_benchmark_v1 as lb
from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    hierarchical_attention_mode,
)
from run_head_top2_targeted_ppl_20260714 import load_model
from run_hierarchical_physical_cache_ppl_20260715 import (
    empty_cuda_caches,
    run_synchronized_one_token,
    synchronize_cuda_devices,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    encode_topic_stream,
    make_bundle,
)
from run_shifted_dynamic_physical_cache_ppl_20260715 import (
    current_input_features,
    previous_logit_features,
)
from train_shifted_physical_budget_router_20260715 import (
    FEATURE_NAMES,
    shifted_feature_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect causal counterfactual action labels on one committed high-budget "
            "trajectory. Probe timings are intentionally not benchmark results."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--full_reference",
        type=Path,
        help="Aligned FullKV result used to define the absolute safe-NLL threshold.",
    )
    parser.add_argument("--topic", choices=sorted(TOPICS), default="computer")
    parser.add_argument("--history_tokens", type=int, default=128000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_index", type=int, default=0)
    parser.add_argument("--window_stride_tokens", type=int, default=128_512)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--low_fraction", type=float, default=0.01)
    parser.add_argument("--mid_fraction", type=float, default=0.015)
    parser.add_argument("--high_fraction", type=float, default=0.02)
    parser.add_argument("--low_stream_group_size", type=int, default=2)
    parser.add_argument("--mid_stream_group_size", type=int, default=2)
    parser.add_argument("--high_stream_group_size", type=int, default=1)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--safe_nll_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--behavior_action", choices=("high", "teacher"), default="high"
    )
    parser.add_argument("--known_reference_ppl", type=float, default=0.0)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn"
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def minimal_safe_action(
    action_nll: list[float],
    tolerance: float,
    reference_nll: float | None = None,
) -> int:
    if len(action_nll) != 3:
        raise ValueError("expected low, mid, and high action losses")
    high_limit = (
        action_nll[2] + tolerance
        if reference_nll is None
        else float(reference_nll) + tolerance
    )
    for action, nll in enumerate(action_nll):
        if nll <= high_limit:
            return action
    return 2


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fractions = [args.low_fraction, args.mid_fraction, args.high_fraction]
    group_sizes = [
        args.low_stream_group_size,
        args.mid_stream_group_size,
        args.high_stream_group_size,
    ]
    if not 0 < fractions[0] < fractions[1] < fractions[2] < args.exact_cache_fraction:
        raise ValueError(
            "expected 0 < low < mid < high < exact_cache_fraction"
        )
    if min(group_sizes) < 1:
        raise ValueError("stream group sizes must be positive")
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
    history_counts = Counter(int(token_id) for token_id in history)
    remote_ids = history[: -args.query_tokens]
    query_ids = history[-args.query_tokens :]
    target_ids = window_stream[
        args.history_tokens : args.history_tokens + args.eval_tokens
    ]
    full_reference_nll: list[float] | None = None
    if args.full_reference is not None:
        full_reference = json.loads(args.full_reference.read_text(encoding="utf-8"))
        if (
            str(full_reference["topic"]) != args.topic
            or int(full_reference["window_index"]) != args.window_index
            or list(map(int, full_reference["target_token_ids"]))
            != list(map(int, target_ids))
        ):
            raise ValueError("FullKV reference is not aligned with this router case")
        full_reference_nll = list(map(float, full_reference["token_nll"]))
    bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
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
        candidate_fraction=args.high_fraction,
        attention_fraction=args.high_fraction,
        exact_cache_fraction=args.exact_cache_fraction,
        max_new_tokens=args.query_tokens + args.eval_tokens + 8,
        candidate_selection_mode="per_head_stream",
        stream_group_size=args.high_stream_group_size,
        directory_backend="fused",
        collect_retrieval_features=True,
    )
    del source_cache
    if torch.cuda.is_available():
        empty_cuda_caches()
        synchronize_cuda_devices()
    conversion_seconds = time.perf_counter() - conversion_started

    cache.set_runtime_action(
        candidate_fraction=args.high_fraction,
        stream_group_size=args.high_stream_group_size,
    )
    previous_logits: torch.Tensor | None = None
    behavior_token_nll: list[float] = []
    feature_vectors: list[list[float]] = []
    counterfactual_nll: list[list[float]] = []
    counterfactual_reference_nll: list[float] = []
    teacher_actions: list[int] = []
    probe_seconds = [0.0, 0.0, 0.0]
    commit_seconds = 0.0

    with hierarchical_attention_mode(model):
        for offset, token_id in enumerate(query_ids):
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                int(token_id),
                cache,
                len(remote_ids) + offset,
                input_device,
            )
            commit_seconds += elapsed
        if previous_logits is None:
            raise RuntimeError("query warm-up produced no logits")

        for target_index, label_id in enumerate(target_ids):
            current_nll = -float(
                F.log_softmax(previous_logits[0].float(), dim=-1)[int(label_id)].item()
            )
            behavior_token_nll.append(current_nll)
            if target_index + 1 >= len(target_ids):
                continue

            feature_vectors.append(
                shifted_feature_vector(
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
            )
            checkpoint = cache.sequence_checkpoint()
            retrieval_checkpoint = cache.retrieval_diagnostic_checkpoint()
            next_label = int(target_ids[target_index + 1])
            action_losses: list[float] = []
            high_probe_logits: torch.Tensor | None = None
            high_probe_seconds = 0.0
            for action, (fraction, group_size) in enumerate(
                zip(fractions, group_sizes)
            ):
                cache.restore_retrieval_diagnostic(retrieval_checkpoint)
                cache.set_runtime_action(
                    candidate_fraction=fraction,
                    stream_group_size=group_size,
                )
                cache, probe_logits, elapsed, _ = run_synchronized_one_token(
                    model,
                    int(label_id),
                    cache,
                    args.history_tokens + target_index,
                    input_device,
                )
                probe_seconds[action] += elapsed
                action_losses.append(
                    -float(
                        F.log_softmax(probe_logits[0].float(), dim=-1)[
                            next_label
                        ].item()
                    )
                )
                if action < 2:
                    cache.restore_sequence_length(checkpoint)
                else:
                    high_probe_logits = probe_logits
                    high_probe_seconds = elapsed

            counterfactual_nll.append(action_losses)
            reference_nll = (
                None
                if full_reference_nll is None
                else full_reference_nll[target_index + 1]
            )
            counterfactual_reference_nll.append(
                action_losses[2] if reference_nll is None else reference_nll
            )
            teacher_action = minimal_safe_action(
                action_losses,
                args.safe_nll_tolerance,
                reference_nll=reference_nll,
            )
            teacher_actions.append(teacher_action)
            if high_probe_logits is None:
                raise RuntimeError("high action probe did not produce logits")
            if args.behavior_action == "high" or teacher_action == 2:
                previous_logits = high_probe_logits
                commit_seconds += high_probe_seconds
            else:
                cache.restore_sequence_length(checkpoint)
                cache.set_runtime_action(
                    candidate_fraction=fractions[teacher_action],
                    stream_group_size=group_sizes[teacher_action],
                )
                cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                    model,
                    int(label_id),
                    cache,
                    args.history_tokens + target_index,
                    input_device,
                )
                commit_seconds += elapsed

    mean_nll = sum(behavior_token_nll) / len(behavior_token_nll)
    action_counts = {
        name: teacher_actions.count(action)
        for action, name in enumerate(("low", "mid", "high"))
    }
    result = {
        "status": "onpolicy_counterfactual_labels",
        "timing_status": "probe_warmed_not_valid_for_speed_claims",
        "topic": args.topic,
        "window_index": args.window_index,
        "window_start_token": window_start,
        "window_stride_tokens": args.window_stride_tokens,
        "history_tokens": args.history_tokens,
        "query_tokens": len(query_ids),
        "eval_tokens": len(target_ids),
        "feature_names": FEATURE_NAMES,
        "action_fractions": fractions,
        "action_stream_group_sizes": group_sizes,
        "safe_nll_tolerance": args.safe_nll_tolerance,
        "behavior_action": args.behavior_action,
        "behavior_ppl": math.exp(mean_nll),
        "known_reference_ppl": args.known_reference_ppl,
        "behavior_quality_retention": (
            args.known_reference_ppl / math.exp(mean_nll)
            if args.known_reference_ppl > 0
            else None
        ),
        "teacher_action_counts": action_counts,
        "teacher_high_rate": action_counts["high"] / len(teacher_actions),
        "teacher_reference": (
            "full_kv_nll" if full_reference_nll is not None else "high_action_nll"
        ),
        "full_reference_path": (
            str(args.full_reference) if args.full_reference is not None else None
        ),
        "prefill_seconds": prefill_seconds,
        "cache_conversion_seconds": conversion_seconds,
        "probe_seconds_by_action": probe_seconds,
        "commit_seconds": commit_seconds,
        "feature_vectors": feature_vectors,
        "counterfactual_nll": counterfactual_nll,
        "counterfactual_reference_nll": counterfactual_reference_nll,
        "teacher_actions": teacher_actions,
        "behavior_token_nll": behavior_token_nll,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "feature_vectors",
            "counterfactual_nll",
            "counterfactual_reference_nll",
            "teacher_actions",
            "behavior_token_nll",
        }
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
