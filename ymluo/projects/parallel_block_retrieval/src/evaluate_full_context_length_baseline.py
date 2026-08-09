from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


POSITION_PATTERNS = (
    (0.10, 0.50),
    (0.10, 0.90),
    (0.50, 0.10),
    (0.50, 0.90),
    (0.90, 0.10),
    (0.90, 0.50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate end-to-end two-hop QA with full attention over controlled "
            "10K/20K/40K contexts containing both gold evidence blocks."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--context_lengths", default="10000,20000,40000")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--attn_implementation", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--model_parallel",
        action="store_true",
        help="Shard one reader across all CUDA_VISIBLE_DEVICES with Accelerate.",
    )
    parser.add_argument("--manual_shard_rank", type=int, default=0)
    parser.add_argument("--manual_num_shards", type=int, default=1)
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge completed manual shard files without loading the model.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_strict(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_hit(text: str, answers: Sequence[str]) -> bool:
    normalized = normalize_strict(text)
    return any(
        normalize_strict(answer) in normalized
        for answer in answers
        if normalize_strict(answer)
    )


def exact_match(text: str, answers: Sequence[str]) -> bool:
    normalized = normalize_answer(text)
    return any(
        normalized == normalize_answer(answer)
        for answer in answers
        if normalize_answer(answer)
    )


def token_f1(text: str, answer: str) -> float:
    predicted = normalize_answer(text).split()
    gold = normalize_answer(answer).split()
    if not predicted or not gold:
        return float(predicted == gold)
    predicted_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in predicted:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in gold:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(
        min(count, gold_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def setup_distributed(device_name: str) -> tuple[int, int, torch.device]:
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device_name == "cuda" else "gloo")
        rank = dist.get_rank()
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", str(rank)))
    else:
        rank = 0
        local_rank = 0
    if device_name == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def query_seed(seed: int, query_id: int) -> int:
    return (seed * 1_000_003 + query_id * 97_409) & 0xFFFFFFFF


def distractor_order(
    num_blocks: int, gold_ids: set[int], seed: int, query_id: int
) -> list[int]:
    candidates = [block_id for block_id in range(num_blocks) if block_id not in gold_ids]
    random.Random(query_seed(seed, query_id)).shuffle(candidates)
    return candidates


def build_context(
    blocks: np.ndarray,
    separator_ids: list[int],
    target_tokens: int,
    gold_ids: list[int],
    distractors: list[int],
    pattern: tuple[float, float],
) -> tuple[list[int], list[int], list[int]]:
    block_tokens = int(blocks.shape[1])
    separator_tokens = len(separator_ids)
    slots = max(
        3,
        math.ceil((target_tokens + separator_tokens) / (block_tokens + separator_tokens)),
    )
    gold_slots = []
    occupied: set[int] = set()
    for fraction in pattern:
        slot = min(slots - 2, max(0, round(fraction * (slots - 1))))
        while slot in occupied:
            slot = min(slots - 2, slot + 1)
        occupied.add(slot)
        gold_slots.append(slot)

    selected: list[int] = []
    distractor_index = 0
    for slot in range(slots):
        if slot == gold_slots[0]:
            selected.append(gold_ids[0])
        elif slot == gold_slots[1]:
            selected.append(gold_ids[1])
        else:
            selected.append(distractors[distractor_index])
            distractor_index += 1

    context_ids: list[int] = []
    gold_starts: list[int] = []
    for slot, block_id in enumerate(selected):
        if slot:
            context_ids.extend(separator_ids)
        if block_id in gold_ids:
            gold_starts.append(len(context_ids))
        context_ids.extend(int(token) for token in blocks[block_id])
    if len(context_ids) < target_tokens:
        while len(context_ids) < target_tokens:
            context_ids.extend(separator_ids)
            context_ids.extend(int(token) for token in blocks[distractors[distractor_index]])
            distractor_index += 1
    context_ids = context_ids[:target_tokens]
    if len(gold_starts) != 2 or any(start + block_tokens > target_tokens for start in gold_starts):
        raise RuntimeError("both complete gold blocks must remain inside the context")
    return context_ids, selected, gold_starts


def prompt_parts(tokenizer: Any, question: str) -> tuple[list[int], list[int]]:
    marker = "<FULL_CONTEXT_TOKENS>"
    content = (
        "Context:\n"
        f"{marker}\n\n"
        f"Question: {question}\n"
        "Answer the question using only the context. Resolve all required relations. "
        "Return only the shortest exact answer, with no explanation."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count(marker) != 1:
        raise RuntimeError("context marker was changed by the chat template")
    prefix, suffix = rendered.split(marker)
    return (
        tokenizer.encode(prefix, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int, float]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated_ids = output[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), len(generated_ids), elapsed


def wilson_interval(successes: int, count: int) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [center - radius, center + radius]


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * (0.5**discordant)
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]], context_lengths: list[int]) -> dict[str, Any]:
    by_length = []
    for context_length in context_lengths:
        group = [row for row in rows if row["context_tokens"] == context_length]
        hits = sum(bool(row["answer_hit"]) for row in group)
        by_length.append(
            {
                "context_tokens": context_length,
                "queries": len(group),
                "answer_hit_rate": hits / len(group),
                "answer_hit_95ci": wilson_interval(hits, len(group)),
                "exact_match_rate": statistics.fmean(row["exact_match"] for row in group),
                "mean_token_f1": statistics.fmean(row["token_f1"] for row in group),
                "mean_prompt_tokens": statistics.fmean(row["prompt_tokens"] for row in group),
                "mean_generation_seconds": statistics.fmean(row["generation_seconds"] for row in group),
                "median_generation_seconds": statistics.median(row["generation_seconds"] for row in group),
                "by_position_pattern": [
                    {
                        "pattern": pattern_index,
                        "target_relative_positions": list(POSITION_PATTERNS[pattern_index]),
                        "queries": len(pattern_group),
                        "answer_hit_rate": statistics.fmean(
                            row["answer_hit"] for row in pattern_group
                        ),
                    }
                    for pattern_index in range(len(POSITION_PATTERNS))
                    if (
                        pattern_group := [
                            row
                            for row in group
                            if int(row["position_pattern"]) == pattern_index
                        ]
                    )
                ],
            }
        )
    paired = []
    by_key = {(row["query_id"], row["context_tokens"]): row for row in rows}
    for left, right in itertools.combinations(context_lengths, 2):
        wins = losses = ties = 0
        for query_id in sorted({row["query_id"] for row in rows}):
            left_hit = bool(by_key[(query_id, left)]["answer_hit"])
            right_hit = bool(by_key[(query_id, right)]["answer_hit"])
            if right_hit and not left_hit:
                wins += 1
            elif left_hit and not right_hit:
                losses += 1
            else:
                ties += 1
        paired.append(
            {
                "left_context_tokens": left,
                "right_context_tokens": right,
                "right_wins": wins,
                "right_losses": losses,
                "ties": ties,
                "mcnemar_exact_p": exact_mcnemar_pvalue(wins, losses),
            }
        )
    return {"by_length": by_length, "paired_answer_hit": paired}


def merge_rows(
    output_dir: Path,
    num_shards: int,
    context_lengths: list[int],
    metadata: dict[str, Any],
) -> None:
    rows = [
        row
        for shard_rank in range(num_shards)
        for row in read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl")
    ]
    rows.sort(key=lambda row: (row["context_tokens"], row["query_id"]))
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {**metadata, **summarize(rows, context_lengths)}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.manual_num_shards <= 0:
        raise ValueError("manual_num_shards must be positive")
    if not 0 <= args.manual_shard_rank < args.manual_num_shards:
        raise ValueError("manual_shard_rank must be inside manual_num_shards")
    context_lengths = sorted(
        {int(item.strip()) for item in args.context_lengths.split(",") if item.strip()}
    )
    output_dir = Path(args.output_dir)
    queries = [
        row
        for row in read_jsonl(Path(args.corpus_dir) / "queries.jsonl")
        if str(row["split"]) == args.split
    ]
    queries.sort(key=lambda row: int(row["query_id"]))
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    metadata = {
        "source": "full-attention end-to-end MuSiQue two-hop length baseline",
        "model": args.model_name_or_path,
        "split": args.split,
        "queries": len(queries),
        "context_lengths": context_lengths,
        "world_size": args.manual_num_shards,
        "seed": args.seed,
        "contains_both_gold_blocks": True,
        "distractor_pool": "real blocks from the shared aligned 10M MuSiQue corpus",
        "nested_distractors": True,
        "position_control": "six balanced early/middle/late ordered pairs",
    }
    if args.merge_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        merge_rows(
            output_dir, args.manual_num_shards, context_lengths, metadata
        )
        return

    distributed_world = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if args.model_parallel and distributed_world > 1:
        raise ValueError("model_parallel cannot be combined with torchrun data parallelism")
    dist_rank, dist_world_size, device = setup_distributed(args.device)
    if args.manual_num_shards > 1:
        if dist_world_size > 1:
            raise ValueError("manual sharding cannot be combined with torchrun")
        rank = args.manual_shard_rank
        world_size = args.manual_num_shards
    else:
        rank = dist_rank
        world_size = dist_world_size
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist_world_size > 1:
        dist.barrier()

    jobs = [
        (query, context_length)
        for context_length in context_lengths
        for query in queries
    ]
    local_jobs = [job for offset, job in enumerate(jobs) if offset % world_size == rank]

    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, local_files_only=True
    )
    separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    model_kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if args.model_parallel:
        model_kwargs["device_map"] = "balanced"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
        device = torch.device("cuda", 0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs
        ).to(device)
    model.eval()

    local_rows = []
    for job_index, (query, context_length) in enumerate(local_jobs, start=1):
        query_id = int(query["query_id"])
        gold_ids = [int(item) for item in query["step_target_blocks"]]
        if len(gold_ids) != 2 or len(set(gold_ids)) != 2:
            raise ValueError(f"query {query_id} does not have two distinct gold blocks")
        distractors = distractor_order(
            int(blocks.shape[0]), set(gold_ids), args.seed, query_id
        )
        pattern_index = query_id % len(POSITION_PATTERNS)
        context_ids, selected_ids, gold_starts = build_context(
            blocks,
            separator_ids,
            context_length,
            gold_ids,
            distractors,
            POSITION_PATTERNS[pattern_index],
        )
        prefix_ids, suffix_ids = prompt_parts(tokenizer, str(query["question"]))
        prompt_ids = prefix_ids + context_ids + suffix_ids
        if len(prompt_ids) > int(model.config.max_position_embeddings):
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, exceeding model limit "
                f"{model.config.max_position_embeddings}"
            )
        generated, generated_tokens, elapsed = generate(
            model, tokenizer, prompt_ids, args.max_new_tokens, device
        )
        answers = [str(query["answer"])]
        row = {
            "query_id": query_id,
            "source_id": str(query["source_id"]),
            "split": str(query["split"]),
            "question": str(query["question"]),
            "answer": str(query["answer"]),
            "context_tokens": context_length,
            "prompt_tokens": len(prompt_ids),
            "gold_block_ids": gold_ids,
            "gold_context_starts": gold_starts,
            "gold_relative_positions": [start / context_length for start in gold_starts],
            "position_pattern": pattern_index,
            "selected_block_ids": selected_ids,
            "generated_text": generated,
            "generated_tokens": generated_tokens,
            "generation_seconds": elapsed,
            "answer_hit": answer_hit(generated, answers),
            "exact_match": exact_match(generated, answers),
            "token_f1": token_f1(generated, answers[0]),
        }
        local_rows.append(row)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "job": job_index,
                    "query_id": query_id,
                    "context_tokens": context_length,
                    "prompt_tokens": len(prompt_ids),
                    "hit": row["answer_hit"],
                    "seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if dist_world_size > 1:
        dist.barrier()
    if args.manual_num_shards == 1 and rank == 0:
        metadata["world_size"] = world_size
        merge_rows(output_dir, world_size, context_lengths, metadata)
    else:
        print(
            json.dumps(
                {"manual_shard_complete": rank, "rows": len(local_rows)}
            ),
            flush=True,
        )
    if dist_world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
