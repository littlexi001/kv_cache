#!/usr/bin/env python3
"""Score current 10M retrieval actions on an already-observed causal token window."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from evaluate_past_only_10m_retrieval_ppl import selected_context
from evaluate_xsum_news_ppl_retrieval import target_nll
from profile_real_qk import resolve_dtype


METHODS = [
    "global_bm25_unigram",
    "flat_book_bm25_depth8",
    "multilevel_bm25_book8_segment8",
    "multilevel_bm25_book8_segment32",
    "multilevel_bm25_book8_segment128",
    "multilevel_bm25_book32_segment8",
    "multilevel_bm25_book32_segment32",
]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--states", default="128,256,512")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--observation_tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_only = {
        (int(row["query_id"]), int(row["state_suffix_tokens"])): float(row["mean_nll"])
        for row in rows
        if row["method"] == "query_only"
    }
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["state_suffix_tokens"]), str(row["method"]))].append(row)
    output = []
    for (state, method), group in sorted(groups.items()):
        values = [float(row["mean_nll"]) for row in group]
        deltas = [
            float(row["mean_nll"]) - query_only[(int(row["query_id"]), state)]
            for row in group
        ]
        output.append(
            {
                "state_suffix_tokens": state,
                "method": method,
                "queries": len(group),
                "mean_observed_nll": statistics.fmean(values),
                "observed_window_ppl": math.exp(min(statistics.fmean(values), 20.0)),
                "mean_delta_nll_vs_query_only": statistics.fmean(deltas),
                "mean_forward_seconds": statistics.fmean(
                    float(row["forward_seconds"]) for row in group
                ),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("strict past-only data without predefined source blocks required")
    states = parse_ints(args.states)
    if min(states) <= args.observation_tokens:
        raise ValueError("every state must be longer than the retrospective observation window")
    methods = [item for item in args.methods.split(",") if item]
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    retrieval_rows = [
        row
        for row in read_jsonl(args.retrieval_rows)
        if int(row["prefix_tokens"]) in states and str(row["method"]) in methods
    ]
    retrieval_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in retrieval_rows
    }
    expected = len(queries) * len(states) * len(methods)
    if len(retrieval_lookup) != expected:
        raise ValueError(f"retrieval matrix has {len(retrieval_lookup)} rows, expected {expected}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_query_ids = [query_id for query_id in range(len(queries)) if query_id % world_size == rank]
    rows = []
    for query_id in local_query_ids:
        for state in states:
            current_state = np.asarray(queries[query_id, -state:], dtype=np.int32)
            observed = current_state[-args.observation_tokens :]
            prefix = current_state[: -args.observation_tokens]
            selections: dict[str, list[int]] = {"query_only": []}
            for method in methods:
                ranking = retrieval_lookup[(query_id, state, method)]["top_block_ids"]
                selections[method] = [int(item) for item in ranking[: args.retrieval_blocks]]
            for method, selection in selections.items():
                context = selected_context(selection, base_blocks)
                mean_nll, total_nll, target_tokens, seconds, model_input_tokens = target_nll(
                    model, context, prefix, observed, device
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": state,
                        "method": method,
                        "selected_block_ids": selection,
                        "retrieved_tokens": len(selection) * int(data_summary["block_tokens"]),
                        "prefix_tokens_before_observation": len(prefix),
                        "observation_tokens": len(observed),
                        "model_input_tokens": model_input_tokens,
                        "mean_nll": mean_nll,
                        "total_nll": total_nll,
                        "target_tokens": target_tokens,
                        "forward_seconds": seconds,
                        "selection_uses_future_target": False,
                        "observation_is_available_at_decision_time": True,
                    }
                )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    barrier(world_size)
    if rank == 0:
        all_rows = [
            row
            for shard in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard:03d}.jsonl")
        ]
        all_rows.sort(
            key=lambda row: (
                int(row["query_id"]),
                int(row["state_suffix_tokens"]),
                str(row["method"]),
            )
        )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row) + "\n")
        summary = {
            "source": "real strict past-only PG19 9.9M retrospective action utility",
            "data_summary": data_summary,
            "states": states,
            "methods": methods,
            "retrieval_blocks": args.retrieval_blocks,
            "retrieval_tokens": args.retrieval_blocks * int(data_summary["block_tokens"]),
            "observation_tokens": args.observation_tokens,
            "world_size": world_size,
            "selection_uses_future_target": False,
            "observation_is_available_at_decision_time": True,
            "quality": summarize(all_rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
