from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from riskkv_universal_controller import (  # noqa: E402
    UniversalControllerConfig,
    UniversalKVController,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    AutoModelForCausalLM,
    AutoTokenizer,
    encode_topic_stream,
    evaluate_target_ppl,
    make_bundle,
    pick_input_device,
    resolve_dtype,
    run_tokens_all_logits,
    selector_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal rolling echo retrieval PPL prototype.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--windows_per_topic", type=int, default=3)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--budget_tokens", type=int, default=2048)
    parser.add_argument("--page_tokens", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=32)
    parser.add_argument("--recent_tokens", type=int, default=1536)
    parser.add_argument("--echo_match_tokens", type=int, default=8)
    parser.add_argument("--echo_confirmation_tokens", type=int, default=8)
    parser.add_argument("--echo_stability_matches", type=int, default=3)
    parser.add_argument("--echo_refresh_tokens", type=int, default=16)
    parser.add_argument("--replay_chunk_tokens", type=int, default=64)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def build_echo_index(remote_ids: list[int], match_tokens: int) -> dict[tuple[int, ...], list[int]]:
    index: dict[tuple[int, ...], list[int]] = {}
    for start in range(0, len(remote_ids) - match_tokens + 1):
        index.setdefault(tuple(remote_ids[start : start + match_tokens]), []).append(start)
    return index


def find_echo_match(
    observed_ids: list[int],
    echo_index: dict[tuple[int, ...], list[int]],
    match_tokens: int,
) -> int | None:
    if len(observed_ids) < match_tokens:
        return None
    starts = echo_index.get(tuple(observed_ids[-match_tokens:]))
    if not starts:
        return None
    return starts[-1]


def find_echo_match_with_backward_span(
    observed_ids: list[int],
    remote_ids: list[int],
    echo_index: dict[tuple[int, ...], list[int]],
    match_tokens: int,
    max_confirmation_tokens: int = 64,
) -> tuple[int | None, int]:
    if len(observed_ids) < match_tokens:
        return None, 0
    starts = echo_index.get(tuple(observed_ids[-match_tokens:]))
    if not starts:
        return None, 0
    best_start: int | None = None
    best_span = 0
    observed_end = len(observed_ids)
    for start in starts:
        span = match_tokens
        while span < max_confirmation_tokens:
            extra = span - match_tokens + 1
            remote_index = start - extra
            observed_index = observed_end - match_tokens - extra
            if remote_index < 0 or observed_index < 0:
                break
            if remote_ids[remote_index] != observed_ids[observed_index]:
                break
            span += 1
        if span > best_span or (span == best_span and (best_start is None or start > best_start)):
            best_start = start
            best_span = span
    return best_start, best_span


def update_aligned_match_run(
    previous_match_start: int | None,
    match_start: int | None,
    previous_run: int,
) -> int:
    if match_start is None:
        return 0
    if previous_match_start is not None and match_start == previous_match_start + 1:
        return previous_run + 1
    return 1


def echo_keep_indices(
    bundle: lb.PromptBundle,
    config: lb.Config,
    base_keep: list[int],
    match_start: int | None,
    match_tokens: int,
) -> list[int]:
    if match_start is None:
        return base_keep
    keep = lb.base_context_keep_indices(bundle, config)
    remaining = max(0, config.budget_tokens - len(keep))
    echo_start = max(bundle.context_token_start, match_start)
    echo_end = min(bundle.query_start, echo_start + match_tokens + remaining)
    for index in range(echo_start, echo_end):
        if len(keep) >= config.budget_tokens:
            break
        keep.add(index)
    if len(keep) < config.budget_tokens:
        for index in base_keep:
            if len(keep) >= config.budget_tokens:
                break
            keep.add(index)
    return lb.fit_context_budget(keep, bundle, config.budget_tokens)


def expanded_recent_keep_indices(
    bundle: lb.PromptBundle,
    budget_tokens: int,
    sink_tokens: int,
) -> list[int]:
    keep = set(
        range(
            bundle.context_token_start,
            min(bundle.query_start, bundle.context_token_start + sink_tokens),
        )
    )
    recent_tokens = max(0, budget_tokens - len(keep))
    keep.update(range(max(bundle.context_token_start, bundle.query_start - recent_tokens), bundle.query_start))
    return lb.fit_context_budget(keep, bundle, budget_tokens)


@torch.inference_mode()
def evaluate_causal_echo_ppl(
    model: torch.nn.Module,
    full_prefix_cache: Any,
    bundle: lb.PromptBundle,
    config: lb.Config,
    base_keep: list[int],
    query_ids: list[int],
    target_ids: list[int],
    input_device: torch.device,
    match_tokens: int,
    refresh_tokens: int,
    replay_chunk_tokens: int,
    stability_matches: int = 1,
    confirmation_tokens: int | None = None,
    enable_echo: bool = True,
    timing_stats: dict[str, float] | None = None,
) -> tuple[float, float, float, int, list[dict[str, int]]]:
    remote_ids = bundle.input_ids[0].tolist()
    index_started = time.perf_counter()
    echo_index = build_echo_index(remote_ids, match_tokens) if enable_echo else {}
    index_seconds = time.perf_counter() - index_started
    loss_sum = 0.0
    token_count = 0
    hash_seconds = 0.0
    gather_seconds = 0.0
    replay_seconds = 0.0
    decode_seconds = 0.0
    rebuilds = 0
    matches: list[dict[str, int]] = []
    cache: Any | None = None
    active_echo_start: int | None = None
    previous_match_start: int | None = None
    aligned_match_run = 0
    required_confirmation = confirmation_tokens or match_tokens
    local_keep = lb.base_context_keep_indices(bundle, config) if config is not None else set(base_keep)
    local_tokens = len(local_keep)
    echo_capacity = max(0, config.budget_tokens - local_tokens) if config is not None else 0
    for target_start in range(0, len(target_ids), refresh_tokens):
        target_end = min(len(target_ids), target_start + refresh_tokens)
        observed_ids = query_ids + target_ids[:target_start]
        lookup_started = time.perf_counter()
        if enable_echo:
            match_start, backward_match_span = find_echo_match_with_backward_span(
                observed_ids,
                remote_ids,
                echo_index,
                match_tokens,
                max(required_confirmation, 64),
            )
        else:
            match_start, backward_match_span = None, 0
        hash_seconds += time.perf_counter() - lookup_started
        aligned_match_run = update_aligned_match_run(
            previous_match_start, match_start, aligned_match_run
        )
        previous_match_start = match_start
        stable_match = (
            match_start is not None
            and backward_match_span >= required_confirmation
            and aligned_match_run >= stability_matches
        )
        match_is_local = match_start is not None and all(
            match_start + offset in local_keep for offset in range(match_tokens)
        )
        actionable_match_start = match_start if stable_match else None
        new_echo_episode = actionable_match_start is not None and not match_is_local and (
            active_echo_start is None
            or not (active_echo_start <= actionable_match_start < active_echo_start + echo_capacity)
        )
        if match_start is not None:
            matches.append(
                {
                    "target_start": target_start,
                    "remote_match_start": match_start,
                    "distance_to_query": len(remote_ids) - match_start,
                    "local_match": int(match_is_local),
                    "aligned_match_run": aligned_match_run,
                    "backward_match_span": backward_match_span,
                    "stable_match": int(stable_match),
                    "cache_rebuilt": int(new_echo_episode),
                }
            )
        if cache is None or new_echo_episode:
            if cache is not None:
                del cache
            keep_indices = echo_keep_indices(
                bundle, config, base_keep, actionable_match_start, match_tokens
            )
            gather_started = time.perf_counter()
            cache = lb.gather_past_key_values(full_prefix_cache, keep_indices)
            gather_seconds += time.perf_counter() - gather_started
            observed = torch.tensor([observed_ids], dtype=torch.long, device=input_device)
            cache, previous_logits, seconds = lb.run_token_segment(
                model,
                observed,
                cache,
                len(remote_ids),
                input_device,
                replay_chunk_tokens,
                True,
            )
            replay_seconds += seconds
            if new_echo_episode:
                active_echo_start = actionable_match_start
                rebuilds += 1
        else:
            previous_token = torch.tensor(
                [[target_ids[target_start - 1]]], dtype=torch.long, device=input_device
            )
            cache, previous_logits, seconds = lb.run_tokens(
                model,
                previous_token,
                cache,
                len(remote_ids) + len(query_ids) + target_start - 1,
                input_device,
                True,
            )
            decode_seconds += seconds
        first_label = torch.tensor([target_ids[target_start]], dtype=torch.long, device=previous_logits.device)
        loss_sum += float(F.cross_entropy(previous_logits.float(), first_label, reduction="sum").item())
        token_count += 1
        if target_end - target_start > 1:
            current_ids = target_ids[target_start : target_end - 1]
            labels = target_ids[target_start + 1 : target_end]
            current = torch.tensor([current_ids], dtype=torch.long, device=input_device)
            cache, logits, seconds = run_tokens_all_logits(
                model,
                current,
                cache,
                len(remote_ids) + len(observed_ids),
                input_device,
                True,
            )
            label_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
            loss_sum += float(F.cross_entropy(logits[0].float(), label_tensor, reduction="sum").item())
            token_count += len(labels)
            decode_seconds += seconds
    if cache is not None:
        del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nll = loss_sum / max(1, token_count)
    elapsed = hash_seconds + gather_seconds + replay_seconds + decode_seconds
    if timing_stats is not None:
        timing_stats.update(
            {
                "index_build_seconds": index_seconds,
                "hash_lookup_seconds": hash_seconds,
                "gather_seconds": gather_seconds,
                "replay_seconds": replay_seconds,
                "decode_seconds": decode_seconds,
                "online_seconds": elapsed,
                "cache_rebuilds": float(rebuilds),
            }
        )
    return nll, math.exp(min(20.0, nll)), elapsed, token_count, matches


@torch.inference_mode()
def evaluate_universal_controller_ppl(
    model: torch.nn.Module,
    full_prefix_cache: Any,
    bundle: lb.PromptBundle,
    config: lb.Config,
    base_keep: list[int],
    query_ids: list[int],
    target_ids: list[int],
    input_device: torch.device,
    match_tokens: int,
    stability_matches: int,
    confirmation_tokens: int,
    replay_chunk_tokens: int,
    timing_stats: dict[str, float] | None = None,
) -> tuple[float, float, float, int, list[dict[str, Any]]]:
    remote_ids = bundle.input_ids[0].tolist()
    controller = UniversalKVController(
        UniversalControllerConfig(
            base_budget_tokens=config.budget_tokens,
            sink_tokens=config.sink_tokens,
            base_recent_tokens=config.recent_tokens,
            expanded_budget_tokens=max(2816, config.budget_tokens + 768),
            echo_match_tokens=match_tokens,
        )
    )
    index_started = time.perf_counter()
    echo_index = build_echo_index(remote_ids, match_tokens)
    index_seconds = time.perf_counter() - index_started
    cache: Any | None = None
    active_action = controller.base_action()
    active_echo_start: int | None = None
    previous_match_start: int | None = None
    aligned_match_run = 0
    loss_sum = 0.0
    token_count = 0
    hash_seconds = 0.0
    gather_seconds = 0.0
    replay_seconds = 0.0
    decode_seconds = 0.0
    rebuilds = 0
    active_budget_sum = 0
    peak_budget = config.budget_tokens
    traces: list[dict[str, Any]] = []
    for target_start, target_id in enumerate(target_ids):
        observed_ids = query_ids + target_ids[:target_start]
        lookup_started = time.perf_counter()
        match_start, backward_match_span = find_echo_match_with_backward_span(
            observed_ids,
            remote_ids,
            echo_index,
            match_tokens,
            max(confirmation_tokens, 64),
        )
        hash_seconds += time.perf_counter() - lookup_started
        aligned_match_run = update_aligned_match_run(
            previous_match_start, match_start, aligned_match_run
        )
        previous_match_start = match_start
        stable_match = (
            match_start is not None
            and backward_match_span >= confirmation_tokens
            and aligned_match_run >= stability_matches
        )
        proposed_action = controller.choose_action(
            match_start,
            len(remote_ids),
            base_keep,
            stable_match=stable_match,
        )
        rebuild = False
        keep_indices = base_keep
        if proposed_action.name == "expanded_recent_2p8k":
            if active_action.name != proposed_action.name:
                rebuild = True
                keep_indices = expanded_recent_keep_indices(
                    bundle, proposed_action.budget_tokens, proposed_action.sink_tokens
                )
                active_action = proposed_action
                active_echo_start = None
        elif proposed_action.name == "recurrence_echo_2k":
            echo_capacity = proposed_action.echo_tokens
            new_episode = (
                active_action.name != proposed_action.name
                or active_echo_start is None
                or match_start is None
                or not (active_echo_start <= match_start < active_echo_start + echo_capacity)
            )
            if new_episode:
                rebuild = True
                keep_indices = echo_keep_indices(bundle, config, base_keep, match_start, match_tokens)
                active_action = proposed_action
                active_echo_start = match_start

        if cache is None or rebuild:
            if cache is not None:
                del cache
                cache = None
            if cache is None and not rebuild:
                keep_indices = base_keep
            gather_started = time.perf_counter()
            cache = lb.gather_past_key_values(full_prefix_cache, keep_indices)
            gather_seconds += time.perf_counter() - gather_started
            observed = torch.tensor([observed_ids], dtype=torch.long, device=input_device)
            cache, previous_logits, seconds = lb.run_token_segment(
                model,
                observed,
                cache,
                len(remote_ids),
                input_device,
                replay_chunk_tokens,
                True,
            )
            replay_seconds += seconds
            if rebuild:
                rebuilds += 1
        else:
            previous_token = torch.tensor(
                [[target_ids[target_start - 1]]], dtype=torch.long, device=input_device
            )
            cache, previous_logits, seconds = lb.run_tokens(
                model,
                previous_token,
                cache,
                len(remote_ids) + len(query_ids) + target_start - 1,
                input_device,
                True,
            )
            decode_seconds += seconds
        label = torch.tensor([target_id], dtype=torch.long, device=previous_logits.device)
        loss_sum += float(F.cross_entropy(previous_logits.float(), label, reduction="sum").item())
        token_count += 1
        active_budget_sum += active_action.budget_tokens
        peak_budget = max(peak_budget, active_action.budget_tokens)
        if match_start is not None or rebuild:
            traces.append(
                {
                    "target_start": target_start,
                    "remote_match_start": match_start,
                    "distance_to_query": len(remote_ids) - match_start if match_start is not None else None,
                    "aligned_match_run": aligned_match_run,
                    "backward_match_span": backward_match_span,
                    "stable_match": int(stable_match),
                    "proposed_action": proposed_action.name,
                    "active_action": active_action.name,
                    "cache_rebuilt": int(rebuild),
                }
            )
    if cache is not None:
        del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nll = loss_sum / max(1, token_count)
    elapsed = hash_seconds + gather_seconds + replay_seconds + decode_seconds
    if timing_stats is not None:
        timing_stats.update(
            {
                "index_build_seconds": index_seconds,
                "hash_lookup_seconds": hash_seconds,
                "gather_seconds": gather_seconds,
                "replay_seconds": replay_seconds,
                "decode_seconds": decode_seconds,
                "online_seconds": elapsed,
                "cache_rebuilds": float(rebuilds),
                "mean_active_budget_tokens": active_budget_sum / max(1, token_count),
                "peak_budget_tokens": float(peak_budget),
            }
        )
    return nll, math.exp(min(20.0, nll)), elapsed, token_count, traces


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = selector_config(args)
    lb.install_llama_layerwise_attention_mask_patch()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    required_tokens = (
        (args.windows_per_topic - 1) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    )
    rows: list[dict[str, Any]] = []
    for topic in topics:
        stream = encode_topic_stream(
            tokenizer,
            TOPICS[topic],
            required_tokens,
            args.dataset_cache_dir,
            args.seed,
        )
        for window in range(args.windows_per_topic):
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            query_text = tokenizer.decode(query_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            bundle, pages = make_bundle(tokenizer, remote_ids, args.page_tokens)
            example = lb.Example(
                benchmark="multitopic_ppl",
                task=topic,
                sample_id=f"{topic}_{window}",
                context="",
                query=query_text,
                answers=[],
                prefix_template="",
                suffix_template="",
                metric="qa_f1",
                max_new_tokens=0,
                length=len(history),
                all_classes=[],
            )
            full_cache, prefill_seconds = lb.prefill_prefix(
                model, bundle, input_device, config.prefill_chunk_tokens
            )
            selector_started = time.perf_counter()
            base_keep = lb.keep_ours_page(bundle, example, pages, config, {"model": model, "tokenizer": tokenizer})
            selector_seconds = time.perf_counter() - selector_started
            recent_config = replace(
                config,
                sink_tokens=args.sink_tokens,
                recent_tokens=max(0, args.budget_tokens - args.sink_tokens),
            )
            recent_keep = lb.keep_streaming(bundle, example, pages, recent_config, None)
            method_specs = [
                ("hybrid_static", base_keep),
                ("sink_recent", recent_keep),
            ]
            for method, keep_indices in method_specs:
                cache = lb.gather_past_key_values(full_cache, keep_indices)
                nll, ppl, seconds, count = evaluate_target_ppl(
                    model,
                    cache,
                    query_ids,
                    target_ids,
                    len(remote_ids),
                    input_device,
                    args.replay_chunk_tokens,
                    True,
                )
                rows.append(
                    {
                        "topic": topic,
                        "window": window,
                        "method": method,
                        "nll": nll,
                        "ppl": ppl,
                        "tokens": count,
                        "kv_ratio": (len(keep_indices) + len(query_ids)) / len(history),
                        "seconds": seconds,
                        "selector_seconds": selector_seconds,
                        "prefill_seconds": prefill_seconds,
                        "echo_matches": [],
                    }
                )
                del cache
            tokenwise_stats: dict[str, float] = {}
            nll, ppl, seconds, count, _ = evaluate_causal_echo_ppl(
                model,
                full_cache,
                bundle,
                config,
                base_keep,
                query_ids,
                target_ids,
                input_device,
                args.echo_match_tokens,
                args.echo_refresh_tokens,
                args.replay_chunk_tokens,
                stability_matches=args.echo_stability_matches,
                confirmation_tokens=args.echo_confirmation_tokens,
                enable_echo=False,
                timing_stats=tokenwise_stats,
            )
            rows.append(
                {
                    "topic": topic,
                    "window": window,
                    "method": "tokenwise_static",
                    "nll": nll,
                    "ppl": ppl,
                    "tokens": count,
                    "kv_ratio": (args.budget_tokens + len(query_ids)) / len(history),
                    "seconds": seconds,
                    "selector_seconds": selector_seconds,
                    "prefill_seconds": prefill_seconds,
                    "timing": tokenwise_stats,
                    "echo_matches": [],
                }
            )
            echo_stats: dict[str, float] = {}
            nll, ppl, seconds, count, matches = evaluate_causal_echo_ppl(
                model,
                full_cache,
                bundle,
                config,
                base_keep,
                query_ids,
                target_ids,
                input_device,
                args.echo_match_tokens,
                args.echo_refresh_tokens,
                args.replay_chunk_tokens,
                stability_matches=args.echo_stability_matches,
                confirmation_tokens=args.echo_confirmation_tokens,
                enable_echo=True,
                timing_stats=echo_stats,
            )
            rows.append(
                {
                    "topic": topic,
                    "window": window,
                    "method": "causal_echo",
                    "nll": nll,
                    "ppl": ppl,
                    "tokens": count,
                    "kv_ratio": (args.budget_tokens + len(query_ids)) / len(history),
                    "seconds": seconds,
                    "selector_seconds": selector_seconds,
                    "prefill_seconds": prefill_seconds,
                    "timing": echo_stats,
                    "echo_matches": matches,
                }
            )
            controller_stats: dict[str, float] = {}
            nll, ppl, seconds, count, traces = evaluate_universal_controller_ppl(
                model,
                full_cache,
                bundle,
                config,
                base_keep,
                query_ids,
                target_ids,
                input_device,
                args.echo_match_tokens,
                args.echo_stability_matches,
                args.echo_confirmation_tokens,
                args.replay_chunk_tokens,
                timing_stats=controller_stats,
            )
            mean_budget = controller_stats["mean_active_budget_tokens"]
            rows.append(
                {
                    "topic": topic,
                    "window": window,
                    "method": "universal_controller",
                    "nll": nll,
                    "ppl": ppl,
                    "tokens": count,
                    "kv_ratio": (mean_budget + len(query_ids)) / len(history),
                    "peak_kv_ratio": (
                        controller_stats["peak_budget_tokens"] + len(query_ids)
                    )
                    / len(history),
                    "seconds": seconds,
                    "selector_seconds": selector_seconds,
                    "prefill_seconds": prefill_seconds,
                    "timing": controller_stats,
                    "echo_matches": traces,
                }
            )
            nll, ppl, seconds, count = evaluate_target_ppl(
                model,
                full_cache,
                query_ids,
                target_ids,
                len(remote_ids),
                input_device,
                args.replay_chunk_tokens,
                False,
            )
            rows.append(
                {
                    "topic": topic,
                    "window": window,
                    "method": "full_kv",
                    "nll": nll,
                    "ppl": ppl,
                    "tokens": count,
                    "kv_ratio": 1.0,
                    "seconds": seconds,
                    "selector_seconds": selector_seconds,
                    "prefill_seconds": prefill_seconds,
                    "echo_matches": [],
                }
            )
            del full_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            args.output_dir.joinpath("results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"{topic}/{window}: "
                + " ".join(
                    f"{row['method']}={row['ppl']:.3f}"
                    for row in rows
                    if row["topic"] == topic and row["window"] == window
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
