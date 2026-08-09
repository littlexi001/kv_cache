#!/usr/bin/env python
"""Extract real QKSieve candidate IDs for CPU-offload locality analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch

import run_direct_countcap_denseprompt_ppl_20260725 as direct
import run_qksieve_fier_autoregressive_speed_20260808 as speed
from run_critical_position_budget_probe_20260715 import run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    install_llama_head_top_fraction_patch,
    load_model,
    prefill_query_tail_mode,
    preload_qksieve_qmse_rate_tables,
    preload_qksieve_runtime_extensions,
    seed_packed_qmse_prefill_queries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history_tokens", type=int, default=65536)
    parser.add_argument("--generation_steps", type=int, default=4)
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


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize_trace(
    records: list[dict[str, Any]],
    *,
    query_heads: int,
    kv_heads: int,
    page_sizes: tuple[int, ...],
) -> dict[str, Any]:
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    groups = query_heads // kv_heads
    union_factors: list[float] = []
    common_fractions: list[float] = []
    page_expansions = {page: [] for page in page_sizes}
    candidate_counts: list[float] = []
    temporal_jaccards: list[float] = []
    previous: dict[tuple[int, int], set[int]] = {}

    for record in records:
        indices = record["indices"].long()
        counts = record["counts"].long()
        layer = int(record["layer"])
        if indices.shape[0] != query_heads:
            raise ValueError("trace query-head count does not match model config")
        head_sets: list[set[int]] = []
        for head in range(query_heads):
            count = int(counts[head].item())
            selected = set(indices[head, :count].tolist())
            head_sets.append(selected)
            candidate_counts.append(float(len(selected)))
            key = (layer, head)
            if key in previous:
                old = previous[key]
                temporal_jaccards.append(
                    len(old & selected) / max(1, len(old | selected))
                )
            previous[key] = selected

        for kv_head in range(kv_heads):
            group_sets = head_sets[kv_head * groups : (kv_head + 1) * groups]
            union = set().union(*group_sets)
            intersection = set.intersection(*group_sets)
            average_head_count = statistics.fmean(len(row) for row in group_sets)
            union_factors.append(len(union) / max(1.0, average_head_count))
            common_fractions.append(len(intersection) / max(1.0, average_head_count))
            for page_size in page_sizes:
                fetched_pages = {token // page_size for token in union}
                fetched_tokens = min(
                    int(record["history_tokens"]),
                    len(fetched_pages) * page_size,
                )
                page_expansions[page_size].append(
                    fetched_tokens / max(1, len(union))
                )

    return {
        "records": len(records),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "gqa_group_size": groups,
        "candidate_count_per_query_head_mean": mean(candidate_counts),
        "gqa_union_factor_mean": mean(union_factors),
        "gqa_union_factor_median": float(statistics.median(union_factors)),
        "four_head_intersection_fraction_mean": mean(common_fractions),
        "temporal_candidate_jaccard_mean": mean(temporal_jaccards),
        "temporal_candidate_jaccard_median": (
            float(statistics.median(temporal_jaccards))
            if temporal_jaccards
            else 0.0
        ),
        "page_transfer_expansion_mean": {
            str(page): mean(values) for page, values in page_expansions.items()
        },
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    args.model = model

    score_mode, budget = speed.METHODS["qksieve_no_value_top1280"]
    assert score_mode is not None
    speed.configure_sparse_args(args, score_mode, budget)
    if os.environ.get("QKSIEVE_PRELOAD_EXTENSIONS", "0") == "1":
        preload_qksieve_runtime_extensions()
    preload_qksieve_qmse_rate_tables(model)

    history = speed.repeated_stream(
        tokenizer, args.text_file, args.history_tokens
    )
    with prefill_query_tail_mode(8) as prefill_queries:
        cache, previous_logits, _ = direct.dense_prompt(
            model,
            tokenizer,
            history,
            input_device,
            args.prefill_chunk_tokens,
            "preallocated",
            1,
            args.generation_steps + 2,
        )
    current_token = int(previous_logits.reshape(-1).argmax().item())
    records: list[dict[str, Any]] = []

    with direct.sparse_context(args, "direct_countcap"):
        seed_packed_qmse_prefill_queries(prefill_queries)
        workers = int(os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "12"))
        if workers > 0:
            direct.sparse_attention.precompute_active_packed_qmse_qk_factors(
                cache, max_workers=workers
            )
        for step in range(args.generation_steps):
            cache, logits, _, _ = run_one_token(
                model,
                current_token,
                cache,
                args.history_tokens + step,
                input_device,
                collect_attention_stats=False,
            )
            current_token = int(logits.reshape(-1).argmax().item())
            states = direct.sparse_attention._ACTIVE_QABS_PCA_STATES
            if not isinstance(states, dict):
                raise RuntimeError("QKSieve states are unavailable inside sparse context")
            for layer, state in sorted(states.items()):
                indices = state.get("packed_qmse_candidate_indices")
                counts = state.get("packed_qmse_candidate_counts")
                if not isinstance(indices, torch.Tensor) or not isinstance(
                    counts, torch.Tensor
                ):
                    continue
                records.append(
                    {
                        "step": step,
                        "layer": int(layer),
                        "history_tokens": args.history_tokens + step,
                        "indices": indices[0].detach().to("cpu", torch.int32),
                        "counts": counts[0].detach().to("cpu", torch.int16),
                    }
                )

    config = model.config
    summary = summarize_trace(
        records,
        query_heads=int(config.num_attention_heads),
        kv_heads=int(config.num_key_value_heads),
        page_sizes=(1, 2, 4, 8, 16, 32, 64),
    )
    payload = {
        "schema": "qksieve_candidate_trace_v1",
        "history_tokens": args.history_tokens,
        "generation_steps": args.generation_steps,
        "score_mode": score_mode,
        "budget": budget,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
