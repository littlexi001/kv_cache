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
        description=(
            "Compute answer-free, test-time proxy losses for selecting among equal-budget "
            "KV retrieval operators."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--retrieval_results", action="append", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("operator proxy evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, device


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_retrieval(
    paths: list[str], methods: set[str]
) -> dict[tuple[str, int], list[int]]:
    selected: dict[tuple[str, int], list[int]] = {}
    for raw_path in paths:
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["method"]
                if method not in methods:
                    continue
                key = (method, int(row["query_id"]))
                block_ids = [int(item) for item in json.loads(row["selected_block_ids"])]
                if key in selected and selected[key] != block_ids:
                    raise ValueError(f"conflicting retrieval rows for {key}")
                selected[key] = block_ids
    return selected


def proxy_metrics(
    model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
    question_prefix_ids: torch.Tensor,
    question_ids: torch.Tensor,
    answer_prefix_ids: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float, float, int, float]:
    prefix = torch.cat([context_ids, question_prefix_ids], dim=0)
    prompt = torch.cat([prefix, question_ids, answer_prefix_ids], dim=0)
    input_ids = prompt[None, :].to(device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    question_start = int(prefix.numel())
    question_tokens = int(question_ids.numel())
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        question_positions = torch.arange(
            question_start - 1,
            question_start + question_tokens - 1,
            device=device,
            dtype=torch.long,
        )
        question_hidden = outputs.last_hidden_state[0].index_select(0, question_positions)
        question_logits = model.lm_head(question_hidden).float()
        question_targets = input_ids[0, question_start : question_start + question_tokens]
        question_losses = F.cross_entropy(question_logits, question_targets, reduction="none")

        next_logits = model.lm_head(outputs.last_hidden_state[0, -1]).float()
        next_log_probs = F.log_softmax(next_logits, dim=-1)
        next_probs = next_log_probs.exp()
        entropy = -(next_probs * next_log_probs).sum()
        top2 = torch.topk(next_log_probs, k=2).values
        margin = top2[0] - top2[1]
        confidence = top2[0]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        float(question_losses.mean().item()),
        float(question_losses[-1].item()),
        float(entropy.item()),
        float(margin.item()),
        question_tokens,
        elapsed,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    selected = read_retrieval(args.retrieval_results, set(methods))
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    for query in queries:
        query_id = int(query["query_id"])
        for method in methods:
            if (method, query_id) not in selected:
                raise ValueError(f"missing retrieval selection for {(method, query_id)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    question_prefix_ids = torch.tensor(
        tokenizer("\nQuestion: ", add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    answer_prefix_ids = torch.tensor(
        tokenizer("\nAnswer:", add_special_tokens=False)["input_ids"], dtype=torch.long
    )

    local_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = int(query["query_id"])
        if query_id % world_size != rank:
            continue
        question_ids = torch.tensor(
            tokenizer(str(query["question"]), add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        for method in methods:
            block_ids = selected[(method, query_id)][: args.target_blocks]
            context_array = np.asarray(blocks[block_ids], dtype=np.int64).reshape(-1)
            metrics = proxy_metrics(
                model,
                torch.from_numpy(context_array),
                question_prefix_ids,
                question_ids,
                answer_prefix_ids,
                device,
            )
            row = {
                "query_id": query_id,
                "dataset": query["dataset"],
                "mode": method,
                "context_blocks": len(block_ids),
                "context_tokens": int(context_array.size),
                "question_tokens": metrics[4],
                "question_nll": metrics[0],
                "question_last_token_nll": metrics[1],
                "answer_prefix_entropy": metrics[2],
                "answer_prefix_top2_margin": metrics[3],
                "elapsed_seconds": metrics[5],
            }
            local_rows.append(row)
            print(json.dumps({"rank": rank, **row}), flush=True)

    shard = output_dir / f"proxy_rank{rank:03d}.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row) + "\n")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        rows: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            rows.extend(read_jsonl(output_dir / f"proxy_rank{shard_rank:03d}.jsonl"))
        rows.sort(key=lambda row: (int(row["query_id"]), str(row["mode"])))
        fields = list(rows[0])
        with (output_dir / "proxy_rows.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary: list[dict[str, Any]] = []
        for method in methods:
            group = [row for row in rows if row["mode"] == method]
            summary.append(
                {
                    "mode": method,
                    "queries": len(group),
                    "mean_question_nll": statistics.fmean(
                        float(row["question_nll"]) for row in group
                    ),
                    "mean_answer_prefix_entropy": statistics.fmean(
                        float(row["answer_prefix_entropy"]) for row in group
                    ),
                    "mean_elapsed_seconds": statistics.fmean(
                        float(row["elapsed_seconds"]) for row in group
                    ),
                }
            )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

