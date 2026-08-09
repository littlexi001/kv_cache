from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate answer NLL on original and retrieved real-text contexts."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument(
        "--retrieval_results",
        action="append",
        required=True,
        help="Retrieval query_results.csv; may be repeated.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default="full128,svd32,svd32_rerank")
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--splits", default="", help="Optional comma-separated data splits.")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument(
        "--query_ids",
        default="",
        help="Optional comma-separated query IDs evaluated before max_queries.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--skip_original_context", action="store_true")
    parser.add_argument("--skip_source_oracle", action="store_true")
    return parser.parse_args()


def setup_distributed(device_mode: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_mode == "cuda" or (
        device_mode == "auto" and torch.cuda.is_available()
    )
    if device_mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return rank, world_size, local_rank, device


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_retrieval(path: Path, methods: set[str]) -> dict[tuple[str, int], list[int]]:
    selected: dict[tuple[str, int], list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method in methods:
                selected[(method, int(row["query_id"]))] = [
                    int(item) for item in json.loads(row["selected_block_ids"])
                ]
    return selected


def answer_nll(
    model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> tuple[float, int, float]:
    prompt = torch.cat([context_ids, suffix_ids], dim=0)
    input_ids = torch.cat([prompt, answer_ids], dim=0)[None, :].to(device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_tokens = int(prompt.numel())
    answer_tokens = int(answer_ids.numel())
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(
            prompt_tokens - 1,
            prompt_tokens + answer_tokens - 1,
            device=device,
            dtype=torch.long,
        )
        hidden = outputs.last_hidden_state[0].index_select(0, positions)
        logits = model.lm_head(hidden).float()
        targets = input_ids[0, prompt_tokens : prompt_tokens + answer_tokens]
        losses = F.cross_entropy(logits, targets, reduction="none")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return float(losses.mean().item()), answer_tokens, elapsed


def source_oracle_blocks(query: dict[str, Any], target_blocks: int) -> list[int]:
    block_start = int(query["block_start"])
    block_count = int(query["block_count"])
    source_ids = list(range(block_start, block_start + block_count))
    if len(source_ids) <= target_blocks:
        return source_ids
    gold = [
        int(block_id)
        for block_id in query.get("gold_block_ids", [])
        if block_start <= int(block_id) < block_start + block_count
    ]
    selected = set(gold[:target_blocks])
    remaining = target_blocks - len(selected)
    if remaining > 0:
        positions = np.linspace(0, block_count - 1, num=remaining, dtype=np.int64).tolist()
        selected.update(source_ids[int(position)] for position in positions)
    if len(selected) < target_blocks:
        for block_id in source_ids:
            selected.add(block_id)
            if len(selected) == target_blocks:
                break
    if len(selected) > target_blocks:
        non_gold = sorted(selected - set(gold), reverse=True)
        while len(selected) > target_blocks and non_gold:
            selected.remove(non_gold.pop(0))
    return sorted(selected)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed(args.device)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    method_set = set(methods)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    selected: dict[tuple[str, int], list[int]] = {}
    for retrieval_path in args.retrieval_results:
        current = read_retrieval(Path(retrieval_path), method_set)
        overlap = selected.keys() & current.keys()
        if overlap:
            raise ValueError(f"duplicate retrieval rows for {sorted(overlap)[:3]}")
        selected.update(current)
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    if allowed_splits:
        queries = [query for query in queries if str(query.get("split", "")) in allowed_splits]
    allowed_query_ids = {
        int(item.strip()) for item in args.query_ids.split(",") if item.strip()
    }
    if allowed_query_ids:
        queries = [
            query for query in queries if int(query["query_id"]) in allowed_query_ids
        ]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_rows: list[dict[str, Any]] = []
    modes: list[str] = []
    if not args.skip_original_context:
        modes.append("original_context")
    if not args.skip_source_oracle:
        modes.append("source_oracle_10k")
    modes.extend(methods)
    for query in queries:
        query_id = int(query["query_id"])
        if query_id % world_size != rank:
            continue
        answers = [str(item).strip() for item in query.get("answers", []) if str(item).strip()]
        if not answers:
            continue
        answer = answers[0]
        suffix_text = f"\nQuestion: {query['question']}\nAnswer:"
        suffix_ids = torch.tensor(
            tokenizer(suffix_text, add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        answer_ids = torch.tensor(
            tokenizer(" " + answer, add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        for mode in modes:
            if mode == "original_context":
                block_start = int(query["block_start"])
                block_count = int(query["block_count"])
                block_ids = list(range(block_start, block_start + block_count))
            elif mode == "source_oracle_10k":
                block_ids = source_oracle_blocks(query, args.target_blocks)
            else:
                block_ids = selected[(mode, query_id)][: args.target_blocks]
            context_array = np.asarray(blocks[block_ids], dtype=np.int64).reshape(-1)
            context_ids = torch.from_numpy(context_array)
            nll, answer_tokens, elapsed = answer_nll(
                model,
                context_ids,
                suffix_ids,
                answer_ids,
                device,
            )
            local_rows.append(
                {
                    "query_id": query_id,
                    "dataset": query["dataset"],
                    "mode": mode,
                    "context_blocks": len(block_ids),
                    "context_tokens": int(context_ids.numel()),
                    "answer_tokens": answer_tokens,
                    "answer_nll": nll,
                    "answer_ppl": math.exp(min(nll, 20.0)),
                    "elapsed_seconds": elapsed,
                    "answer": answer,
                }
            )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "query_id": query_id,
                        "mode": mode,
                        "answer_nll": nll,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as f:
        for row in local_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        rows: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            rows.extend(read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl"))
        reference_mode = (
            "original_context" if "original_context" in modes else "source_oracle_10k"
        )
        if reference_mode not in modes:
            raise ValueError("at least one reference mode is required")
        baseline = {
            int(row["query_id"]): float(row["answer_nll"])
            for row in rows
            if row["mode"] == reference_mode
        }
        for row in rows:
            delta = float(row["answer_nll"]) - baseline[int(row["query_id"])]
            row["reference_mode"] = reference_mode
            row["nll_delta_vs_reference"] = delta
            row["nll_delta_vs_original"] = delta if reference_mode == "original_context" else ""
        rows.sort(key=lambda row: (int(row["query_id"]), str(row["mode"])))

        fields = [
            "query_id",
            "dataset",
            "mode",
            "context_blocks",
            "context_tokens",
            "answer_tokens",
            "answer_nll",
            "answer_ppl",
            "reference_mode",
            "nll_delta_vs_reference",
            "nll_delta_vs_original",
            "elapsed_seconds",
            "answer",
        ]
        with (output_dir / "answer_nll_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        summary_rows: list[dict[str, Any]] = []
        for mode in modes:
            group = [row for row in rows if row["mode"] == mode]
            summary_rows.append(
                {
                    "mode": mode,
                    "queries": len(group),
                    "mean_answer_nll": statistics.fmean(float(row["answer_nll"]) for row in group),
                    "median_answer_nll": statistics.median(float(row["answer_nll"]) for row in group),
                    "reference_mode": reference_mode,
                    "mean_nll_delta_vs_reference": statistics.fmean(
                        float(row["nll_delta_vs_reference"]) for row in group
                    ),
                    "nll_non_increase_rate": statistics.fmean(
                        float(row["nll_delta_vs_reference"]) <= 0.0 for row in group
                    ),
                    "mean_elapsed_seconds": statistics.fmean(
                        float(row["elapsed_seconds"]) for row in group
                    ),
                }
            )
        with (output_dir / "answer_nll_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "source": "Qwen3 answer-token NLL on original or retrieved contexts",
                    "contains_synthetic_vectors": False,
                    "world_size": world_size,
                    "reference_mode": reference_mode,
                    "methods": summary_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
