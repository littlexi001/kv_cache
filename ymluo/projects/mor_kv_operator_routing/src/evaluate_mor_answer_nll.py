from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out answer NLL for MoR-KV retrieved block sets."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--retrieval_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--methods",
        default="bm25_b4,single_hybrid_b4,mor_kv_b4,wrong_router_mor_b4",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max_queries", type=int, default=0)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_retrieval(
    path: Path, methods: set[str], split: str
) -> dict[tuple[str, int], list[int]]:
    output: dict[tuple[str, int], list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in methods or row.get("split") != split:
                continue
            output[(row["method"], int(row["query_id"]))] = [
                int(item) for item in json.loads(row["selected_block_ids"])
            ]
    return output


def answer_nll(
    model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> tuple[float, int, float]:
    prompt = torch.cat([context_ids, suffix_ids])
    input_ids = torch.cat([prompt, answer_ids])[None, :].to(device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
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
    torch.cuda.synchronize(device)
    return float(losses.mean().item()), answer_tokens, time.perf_counter() - started


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    method_set = set(methods)
    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = [row for row in read_jsonl(corpus_dir / "queries.jsonl") if row["split"] == args.split]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    retrieval = read_retrieval(Path(args.retrieval_results), method_set, args.split)
    for query in queries:
        for method in methods:
            key = (method, int(query["query_id"]))
            if key not in retrieval:
                raise KeyError(f"Missing retrieval result: {key}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    local_rows: list[dict[str, Any]] = []
    for local_index, query in enumerate(queries):
        query_id = int(query["query_id"])
        if local_index % world_size != rank:
            continue
        answers = [str(item).strip() for item in query.get("answers", []) if str(item).strip()]
        if not answers:
            continue
        suffix = f"\nQuestion: {query['question']}\nAnswer:"
        suffix_ids = torch.tensor(
            tokenizer(suffix, add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        answer_ids = torch.tensor(
            tokenizer(" " + answers[0], add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        gold = {int(item) for item in query["gold_block_ids"]}
        negatives = {int(item) for item in query.get("hard_negative_block_ids", [])}
        for method in methods:
            selected = retrieval[(method, query_id)]
            # KV blocks must be restored in causal order after retrieval.
            ordered = sorted({int(item) for item in selected})
            context = torch.from_numpy(np.asarray(blocks[ordered], dtype=np.int64).reshape(-1))
            nll, answer_tokens, elapsed = answer_nll(
                model, context, suffix_ids, answer_ids, device
            )
            selected_set = set(ordered)
            local_rows.append(
                {
                    "query_id": query_id,
                    "task_type": query["task_type"],
                    "method": method,
                    "context_blocks": len(ordered),
                    "context_tokens": int(context.numel()),
                    "gold_hits": len(gold & selected_set),
                    "hard_negative_hits": len(negatives & selected_set),
                    "answer_tokens": answer_tokens,
                    "answer_nll": nll,
                    "answer_ppl": math.exp(min(nll, 20.0)),
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                json.dumps(
                    {"rank": rank, "query_id": query_id, "method": method, "nll": nll}
                ),
                flush=True,
            )

    shard = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row) + "\n")
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        rows: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            rows.extend(read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl"))
        baseline = {
            int(row["query_id"]): float(row["answer_nll"])
            for row in rows
            if row["method"] == methods[0]
        }
        for row in rows:
            row["nll_delta_vs_first_method"] = float(row["answer_nll"]) - baseline[
                int(row["query_id"])
            ]
        rows.sort(key=lambda row: (int(row["query_id"]), str(row["method"])))
        fields = list(rows[0])
        write_csv(output_dir / "answer_nll_rows.csv", rows, fields)

        summary_rows: list[dict[str, Any]] = []
        for method in methods:
            for task in ["all", *sorted({str(row["task_type"]) for row in rows})]:
                group = [
                    row
                    for row in rows
                    if row["method"] == method
                    and (task == "all" or row["task_type"] == task)
                ]
                if not group:
                    continue
                summary_rows.append(
                    {
                        "method": method,
                        "task_type": task,
                        "queries": len(group),
                        "mean_answer_nll": statistics.fmean(
                            float(row["answer_nll"]) for row in group
                        ),
                        "mean_nll_delta_vs_first_method": statistics.fmean(
                            float(row["nll_delta_vs_first_method"]) for row in group
                        ),
                        "win_rate_vs_first_method": statistics.fmean(
                            float(row["nll_delta_vs_first_method"]) < 0.0 for row in group
                        ),
                        "mean_gold_hits": statistics.fmean(
                            int(row["gold_hits"]) for row in group
                        ),
                        "mean_hard_negative_hits": statistics.fmean(
                            int(row["hard_negative_hits"]) for row in group
                        ),
                        "mean_elapsed_seconds": statistics.fmean(
                            float(row["elapsed_seconds"]) for row in group
                        ),
                    }
                )
        write_csv(output_dir / "answer_nll_summary.csv", summary_rows, list(summary_rows[0]))
        summary = {
            "split": args.split,
            "queries": len(queries),
            "first_method_baseline": methods[0],
            "methods": [row for row in summary_rows if row["task_type"] == "all"],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
