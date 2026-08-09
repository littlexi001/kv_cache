from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from evaluate_past_only_10m_retrieval_ppl import (
    barrier,
    read_jsonl,
    selected_context,
    setup_distributed,
)
from evaluate_xsum_news_ppl_retrieval import synchronize
from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score recently observed state tokens under alternative scope depths."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=100_000_000)
    parser.add_argument("--state_suffix_tokens", default="128,512")
    parser.add_argument("--scope_depths", default="3,8,16,32")
    parser.add_argument("--max_probe_tokens", type=int, default=64)
    parser.add_argument("--probe_windows", default="8,16,32,64")
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


@torch.inference_mode()
def observed_token_losses(
    model: AutoModelForCausalLM,
    context_ids: np.ndarray,
    state_ids: np.ndarray,
    *,
    probe_tokens: int,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    state = np.asarray(state_ids, dtype=np.int64)
    if probe_tokens >= len(state):
        raise ValueError("probe_tokens must be smaller than state length")
    context = torch.from_numpy(np.asarray(context_ids, dtype=np.int64))
    prefix = torch.from_numpy(state[:-probe_tokens])
    probe = torch.from_numpy(state[-probe_tokens:])
    prompt = torch.cat([context, prefix], dim=0)
    input_ids = torch.cat([prompt, probe], dim=0)[None, :].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_tokens = int(prompt.numel())
    synchronize(device)
    started = time.perf_counter()
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    positions = torch.arange(
        prompt_tokens - 1,
        prompt_tokens + probe_tokens - 1,
        device=device,
        dtype=torch.long,
    )
    hidden = outputs.last_hidden_state[0].index_select(0, positions)
    logits = model.lm_head(hidden).float()
    targets = input_ids[0, prompt_tokens : prompt_tokens + probe_tokens]
    losses = F.cross_entropy(logits, targets, reduction="none")
    synchronize(device)
    elapsed = time.perf_counter() - started
    return losses.cpu().numpy().astype(np.float32), elapsed, int(input_ids.shape[1])


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    suffixes = parse_ints(args.state_suffix_tokens)
    depths = parse_ints(args.scope_depths)
    windows = parse_ints(args.probe_windows)
    if max(windows) > args.max_probe_tokens:
        raise ValueError("probe window exceeds max_probe_tokens")
    if min(suffixes) <= args.max_probe_tokens:
        raise ValueError("every state must leave non-empty prefix before the probe")

    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("requires past-only data without predefined source blocks")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != args.memory_tokens:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        if depth not in depths or int(row["prefix_tokens"]) not in suffixes:
            continue
        retrieval_lookup[
            (int(row["query_id"]), int(row["prefix_tokens"]), depth)
        ] = row

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_query_ids = [
        query_id for query_id in range(len(queries)) if query_id % world_size == rank
    ]
    rows = []
    for query_id in local_query_ids:
        for suffix in suffixes:
            state = np.asarray(queries[query_id, -suffix:], dtype=np.int32)
            for depth in depths:
                retrieval = retrieval_lookup[(query_id, suffix, depth)]
                retrieval_query_end_offset = int(
                    retrieval.get("query_end_offset_tokens", 0)
                )
                selection = [
                    int(item) for item in retrieval["top_block_ids"][: args.retrieval_blocks]
                ]
                context = selected_context(selection, base_blocks)
                losses, seconds, model_input_tokens = observed_token_losses(
                    model,
                    context,
                    state,
                    probe_tokens=args.max_probe_tokens,
                    device=device,
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "memory_tokens": args.memory_tokens,
                        "state_suffix_tokens": suffix,
                        "scope_depth": depth,
                        "method": f"hier_bm25_scope{depth}",
                        "selected_block_ids": selection,
                        "retrieved_tokens": len(selection)
                        * int(data_summary["block_tokens"]),
                        "max_probe_tokens": args.max_probe_tokens,
                        "probe_token_losses": losses.astype(float).tolist(),
                        "probe_mean_nll": {
                            str(window): float(losses[-window:].mean())
                            for window in windows
                        },
                        "forward_seconds": seconds,
                        "model_input_tokens": model_input_tokens,
                        "retrieval_query_end_offset_tokens": retrieval_query_end_offset,
                        "retrieval_query_uses_observed_probe_tokens": (
                            retrieval_query_end_offset < args.max_probe_tokens
                        ),
                        "probe_uses_only_already_observed_state": True,
                        "probe_uses_future_target": False,
                        "selection_uses_target": False,
                    }
                )
        print(f"rank={rank} completed query={query_id}", flush=True)

    rank_path = output_dir / f"rows_rank{rank}.jsonl"
    with rank_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    barrier(world_size)
    if rank == 0:
        all_rows = []
        for worker_rank in range(world_size):
            all_rows.extend(read_jsonl(output_dir / f"rows_rank{worker_rank}.jsonl"))
        all_rows.sort(
            key=lambda row: (
                int(row["query_id"]),
                int(row["state_suffix_tokens"]),
                int(row["scope_depth"]),
            )
        )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": "retrospective observed-state counterfactual scope probe",
            "protocol": {
                "queries": len(queries),
                "memory_tokens": args.memory_tokens,
                "state_suffix_tokens": suffixes,
                "scope_depths": depths,
                "max_probe_tokens": args.max_probe_tokens,
                "probe_windows": windows,
                "retrieval_blocks": args.retrieval_blocks,
                "retrieval_query_end_offset_tokens": sorted(
                    {
                        int(row["retrieval_query_end_offset_tokens"])
                        for row in all_rows
                    }
                ),
                "retrieval_query_uses_observed_probe_tokens": any(
                    bool(row["retrieval_query_uses_observed_probe_tokens"])
                    for row in all_rows
                ),
                "probe_uses_only_already_observed_state": True,
                "probe_uses_future_target": False,
                "selection_uses_target": False,
                "world_size": world_size,
            },
            "rows": len(all_rows),
            "mean_forward_seconds": statistics.fmean(
                float(row["forward_seconds"]) for row in all_rows
            ),
            "mean_forward_seconds_by_state": {
                str(suffix): statistics.fmean(
                    float(row["forward_seconds"])
                    for row in all_rows
                    if int(row["state_suffix_tokens"]) == suffix
                )
                for suffix in suffixes
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)


if __name__ == "__main__":
    main()
