from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_single_query_dynamic_kv_generation import (
    DynamicKVController,
    advance_token,
    answer_hit,
    build_chat_prompt_parts,
    build_prompt,
    prefill,
    read_jsonl,
    resolve_dtype,
)


METHODS = [
    "question_only",
    "full_source",
    "static_k3",
    "dynamic_c1k3",
    "dynamic_c3k3",
]
TARGET_DATASETS = ["2wikimqa", "hotpotqa", "musique"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corpus-disjoint multi-sample evaluation of online dynamic KV retrieval."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--queries_per_dataset", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--prompt_style", choices=["reasoning_v2", "longbench"], default="reasoning_v2"
    )
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--prefill_chunk", type=int, default=1024)
    parser.add_argument("--full_correct_f1", type=float, default=0.8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def normalize_longbench(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_longbench(prediction).split()
    truth_tokens = normalize_longbench(ground_truth).split()
    if not prediction_tokens or not truth_tokens:
        return float(prediction_tokens == truth_tokens)
    overlap = Counter(prediction_tokens) & Counter(truth_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_f1(prediction: str, answers: Sequence[str]) -> float:
    return max((token_f1(prediction, answer) for answer in answers), default=0.0)


def extract_first_final_answer(text: str) -> str:
    match = re.search(r"final\s+answer\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def method_spec(method: str) -> tuple[str, int, bool]:
    if method == "question_only":
        return "question_only", 0, False
    if method == "full_source":
        return "full_source", 0, False
    if method == "static_k3":
        return "dynamic", 0, True
    if method == "dynamic_c1k3":
        return "dynamic", 1, False
    if method == "dynamic_c3k3":
        return "dynamic", 3, False
    raise ValueError(f"unknown method: {method}")


def event_metrics(
    events: Sequence[dict[str, Any]], gold_blocks: set[int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    first_gold_generation_token = None
    for event in events:
        key = (str(event["phase"]), int(event["phase_token"]))
        selected = {int(item) for item in event["selected_global_block_ids"]}
        grouped[key].update(selected)
        if (
            event["phase"] == "generation"
            and selected & gold_blocks
            and first_gold_generation_token is None
        ):
            first_gold_generation_token = int(event["phase_token"])
    generation_unions = [
        len(selected) for (phase, _), selected in grouped.items() if phase == "generation"
    ]
    prompt_tokens = [token for (phase, token) in grouped if phase == "prompt"]
    final_prompt_layers_with_gold = 0
    if prompt_tokens:
        final_prompt = max(prompt_tokens)
        final_prompt_layers_with_gold = sum(
            bool(gold_blocks & {int(item) for item in event["selected_global_block_ids"]})
            for event in events
            if event["phase"] == "prompt" and int(event["phase_token"]) == final_prompt
        )
    return {
        "refresh_events": len(grouped),
        "mean_generation_physical_union": (
            statistics.fmean(generation_unions) if generation_unions else 0.0
        ),
        "max_generation_physical_union": max(generation_unions, default=0),
        "first_gold_generation_token": first_gold_generation_token,
        "final_prompt_layers_with_gold": final_prompt_layers_with_gold,
    }


@torch.inference_mode()
def run_method(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    blocks: np.ndarray,
    query: dict[str, Any],
    method: str,
    max_new_tokens: int,
    block_tokens: int,
    prefill_chunk: int,
    prompt_style: str,
    use_chat_template: bool,
    device: torch.device,
) -> dict[str, Any]:
    mode, interval, static_generation = method_spec(method)
    block_start = int(query["block_start"])
    block_count = int(query["block_count"])
    context_ids = np.asarray(
        blocks[block_start : block_start + block_count], dtype=np.int64
    ).reshape(-1).tolist()
    if use_chat_template:
        chat_prefix_ids, prompt_ids = build_chat_prompt_parts(
            tokenizer, str(query["question"]), prompt_style
        )
    else:
        chat_prefix_ids = []
        prompt_ids = tokenizer(
            build_prompt(str(query["question"]), prompt_style),
            add_special_tokens=False,
        )["input_ids"]
    answers = [str(item) for item in query["answers"]]
    started = time.perf_counter()
    controller = None
    if mode == "question_only":
        initial_ids = chat_prefix_ids + prompt_ids
        cache, logits = prefill(model, initial_ids, prefill_chunk, device)
        position = len(initial_ids)
    else:
        memory_ids = chat_prefix_ids + context_ids
        cache, _ = prefill(model, memory_ids, prefill_chunk, device)
        position = len(memory_ids)
        if mode == "dynamic":
            controller = DynamicKVController(
                model,
                context_length=len(context_ids),
                context_start=len(chat_prefix_ids),
                context_block_start=block_start,
                block_tokens=block_tokens,
                blocks_per_refresh=3,
                retrieval_interval=max(interval, 1),
            )
        logits = torch.empty(0, device=device)
        for prompt_index, token_id in enumerate(prompt_ids):
            if controller is not None:
                controller.set_step("prompt", prompt_index, force_refresh=True)
            cache, logits = advance_token(model, token_id, cache, position, device)
            position += 1

    generated: list[int] = []
    first_hit_generation_token = None
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    try:
        for generated_index in range(max_new_tokens):
            token_id = int(torch.argmax(logits).item())
            if token_id in eos_ids:
                break
            generated.append(token_id)
            current_text = tokenizer.decode(generated, skip_special_tokens=True)
            if first_hit_generation_token is None and answer_hit(current_text, answers):
                first_hit_generation_token = generated_index + 1
            if controller is not None:
                refresh = False if static_generation else generated_index % interval == 0
                controller.set_step("generation", generated_index, force_refresh=refresh)
            cache, logits = advance_token(model, token_id, cache, position, device)
            position += 1
    finally:
        if controller is not None:
            controller.close(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    final_answer = extract_first_final_answer(generated_text)
    events = controller.events if controller is not None else []
    unique_global = sorted(
        {
            int(block_id)
            for event in events
            for block_id in event["selected_global_block_ids"]
        }
    )
    gold = {int(item) for item in query["gold_block_ids"]}
    row = {
        "query_id": int(query["query_id"]),
        "dataset": str(query["dataset"]),
        "method": method,
        "question": str(query["question"]),
        "answers": answers,
        "context_blocks": 0 if mode == "question_only" else block_count,
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "first_final_answer": final_answer,
        "answer_hit_128": answer_hit(generated_text, answers),
        "first_hit_generation_token": first_hit_generation_token,
        "raw_generation_f1": best_f1(generated_text, answers),
        "structured_final_f1": best_f1(final_answer, answers),
        "gold_ever_retrieved": bool(gold & set(unique_global)),
        "unique_block_count": len(unique_global),
        "elapsed_seconds": time.perf_counter() - started,
        **event_metrics(events, gold),
    }
    del cache, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def summarize(
    rows: Sequence[dict[str, Any]], eligible: set[int], methods: Sequence[str]
) -> list[dict[str, Any]]:
    summary = []
    for scope, query_ids in (("all", None), ("full_source_f1_ge_0.8", eligible)):
        for method in methods:
            group = [
                row
                for row in rows
                if row["method"] == method
                and (query_ids is None or int(row["query_id"]) in query_ids)
            ]
            summary.append(
                {
                    "scope": scope,
                    "method": method,
                    "queries": len(group),
                    "answer_hit_128": mean(group, "answer_hit_128"),
                    "structured_final_f1": mean(group, "structured_final_f1"),
                    "raw_generation_f1": mean(group, "raw_generation_f1"),
                    "gold_ever_retrieved": mean(group, "gold_ever_retrieved"),
                    "unique_block_count": mean(group, "unique_block_count"),
                    "mean_generation_physical_union": mean(
                        group, "mean_generation_physical_union"
                    ),
                    "elapsed_seconds": mean(group, "elapsed_seconds"),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be selected from {METHODS}")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    corpus_dir = Path(args.corpus_dir)
    all_queries = read_jsonl(corpus_dir / "queries.jsonl")
    selected_queries = []
    for dataset in TARGET_DATASETS:
        selected_queries.extend(
            [query for query in all_queries if query["dataset"] == dataset][
                : args.queries_per_dataset
            ]
        )
    selected_queries.sort(key=lambda row: int(row["query_id"]))
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    local_rows = []
    for query_index, query in enumerate(selected_queries):
        if query_index % world_size != rank:
            continue
        for method in methods:
            row = run_method(
                model=model,
                tokenizer=tokenizer,
                blocks=blocks,
                query=query,
                method=method,
                max_new_tokens=args.max_new_tokens,
                block_tokens=args.block_tokens,
                prefill_chunk=args.prefill_chunk,
                prompt_style=args.prompt_style,
                use_chat_template=True,
                device=device,
            )
            local_rows.append(row)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "query_id": row["query_id"],
                        "method": method,
                        "hit": row["answer_hit_128"],
                        "final_f1": row["structured_final_f1"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    shard = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        rows = []
        for shard_rank in range(world_size):
            rows.extend(read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl"))
        rows.sort(key=lambda row: (int(row["query_id"]), methods.index(row["method"])))
        full_rows = {int(row["query_id"]): row for row in rows if row["method"] == "full_source"}
        eligible = {
            query_id
            for query_id, row in full_rows.items()
            if float(row["structured_final_f1"]) >= args.full_correct_f1
        }
        summary_rows = summarize(rows, eligible, methods)
        (output_dir / "rows.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = {
            "queries": len(selected_queries),
            "datasets": TARGET_DATASETS,
            "queries_per_dataset": args.queries_per_dataset,
            "methods": methods,
            "prompt_style": args.prompt_style,
            "use_chat_template": True,
            "full_correct_f1_threshold": args.full_correct_f1,
            "full_correct_query_ids": sorted(eligible),
            "full_correct_queries": len(eligible),
            "summary": summary_rows,
            "note": (
                "Static and dynamic methods use the same per-layer K=3 budget. "
                "No cumulative block-read limit is enforced."
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
