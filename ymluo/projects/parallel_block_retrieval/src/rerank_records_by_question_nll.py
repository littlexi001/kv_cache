from __future__ import annotations

import argparse
import csv
import json
import os
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
        description="Rerank BM25 records by Qwen question likelihood without using answers."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--lexical_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_records", type=int, default=5)
    parser.add_argument("--record_margin_threshold", type=float, default=0.04)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Question-likelihood reranking requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def question_nll(
    model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    question_ids: torch.Tensor,
    device: torch.device,
) -> float:
    prompt = torch.cat([context_ids, prefix_ids])
    input_ids = torch.cat([prompt, question_ids])[None].to(device=device, dtype=torch.long)
    prompt_tokens = int(prompt.numel())
    question_tokens = int(question_ids.numel())
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(
            prompt_tokens - 1,
            prompt_tokens + question_tokens - 1,
            device=device,
            dtype=torch.long,
        )
        hidden = outputs.last_hidden_state[0].index_select(0, positions)
        logits = model.lm_head(hidden).float()
        targets = input_ids[0, prompt_tokens : prompt_tokens + question_tokens]
        losses = F.cross_entropy(logits, targets, reduction="none")
    return float(losses.mean().item())


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    corpus_dir = Path(args.corpus_dir)
    lexical_dir = Path(args.lexical_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    records = read_jsonl(corpus_dir / "records.jsonl")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    record_scores = np.load(lexical_dir / "record_scores.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    source_record_by_start = {
        int(record["block_start"]): record_id for record_id, record in enumerate(records)
    }
    prefix_ids = torch.tensor(
        tokenizer("\nQuestion:", add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    local_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query_index, query in enumerate(queries):
        if query_index % world_size != rank:
            continue
        ids = np.arange(len(records), dtype=np.int64)
        ranked_records = np.lexsort((ids, -record_scores[query_index]))[: args.top_records]
        top_score = float(record_scores[query_index, ranked_records[0]])
        margin = float(
            record_scores[query_index, ranked_records[0]]
            - record_scores[query_index, ranked_records[1]]
        )
        relative_margin = margin / max(abs(top_score), 1.0e-6)
        question_ids = torch.tensor(
            tokenizer(" " + str(query["question"]), add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        source_record = source_record_by_start[int(query["block_start"])]
        for bm25_rank, record_id_value in enumerate(ranked_records, start=1):
            record_id = int(record_id_value)
            record = records[record_id]
            block_start = int(record["block_start"])
            block_end = block_start + int(record["block_count"])
            context_array = np.array(
                blocks[block_start:block_end], dtype=np.int64, copy=True
            ).reshape(-1)
            context = torch.from_numpy(context_array)
            nll = question_nll(model, context, prefix_ids, question_ids, device)
            local_rows.append(
                {
                    "query_id": query_index,
                    "dataset": query["dataset"],
                    "record_id": record_id,
                    "bm25_rank": bm25_rank,
                    "bm25_score": float(record_scores[query_index, record_id]),
                    "question_nll": nll,
                    "source_record": source_record,
                    "is_source_record": float(record_id == source_record),
                    "relative_bm25_margin": relative_margin,
                }
            )
        print(
            json.dumps(
                {"rank": rank, "query_id": query_index, "relative_margin": relative_margin}
            ),
            flush=True,
        )

    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered[0] = local_rows
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if rank == 0:
        rows = [row for part in gathered if part is not None for row in part]
        rows.sort(key=lambda row: (row["query_id"], row["bm25_rank"]))
        with (output_dir / "record_scores.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["query_id"]), []).append(row)
        bm25_hits = 0
        likelihood_hits = 0
        risk_hits = 0
        route_rows: list[dict[str, Any]] = []
        for query_id, group in sorted(grouped.items()):
            bm25_choice = min(group, key=lambda row: int(row["bm25_rank"]))
            likelihood_choice = min(
                group, key=lambda row: (float(row["question_nll"]), int(row["record_id"]))
            )
            use_likelihood = (
                float(bm25_choice["relative_bm25_margin"])
                < args.record_margin_threshold
            )
            risk_choice = likelihood_choice if use_likelihood else bm25_choice
            source = int(bm25_choice["source_record"])
            bm25_hits += int(int(bm25_choice["record_id"]) == source)
            likelihood_hits += int(int(likelihood_choice["record_id"]) == source)
            risk_hits += int(int(risk_choice["record_id"]) == source)
            route_rows.append(
                {
                    "query_id": query_id,
                    "source_record": source,
                    "bm25_record": int(bm25_choice["record_id"]),
                    "likelihood_record": int(likelihood_choice["record_id"]),
                    "risk_record": int(risk_choice["record_id"]),
                    "relative_bm25_margin": float(
                        bm25_choice["relative_bm25_margin"]
                    ),
                    "used_likelihood": use_likelihood,
                }
            )
        with (output_dir / "routing.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(route_rows[0]))
            writer.writeheader()
            writer.writerows(route_rows)
        summary = {
            "queries": len(grouped),
            "top_records": args.top_records,
            "record_margin_threshold": args.record_margin_threshold,
            "bm25_top1_recall": bm25_hits / len(grouped),
            "question_likelihood_top1_recall": likelihood_hits / len(grouped),
            "risk_route_top1_recall": risk_hits / len(grouped),
            "likelihood_fallback_queries": sum(row["used_likelihood"] for row in route_rows),
            "elapsed_seconds": elapsed,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
