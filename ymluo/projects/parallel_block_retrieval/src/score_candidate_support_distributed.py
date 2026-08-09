from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import resolve_dtype, setup_distributed
from evaluate_transition_support_verifier import (
    answer_likelihood_prompt,
    score_answer_logprob,
    score_yes_no,
    support_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed frozen support scoring for per-block candidate answers."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--generation_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_group(tokenizer: Any, values: tuple[str, ...]) -> list[int]:
    return list(
        dict.fromkeys(
            tokenizer(value, add_special_tokens=False)["input_ids"][0]
            for value in values
        )
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rank, world_size, _local_rank, device = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_answer_from_bridge"
    }
    generations = [
        row
        for row in read_jsonl(Path(args.generation_rows_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_answer_from_bridge"
    ]
    generations.sort(key=lambda row: int(row["query_id"]))
    local = [row for offset, row in enumerate(generations) if offset % world_size == rank]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    positive_ids = token_group(tokenizer, ("Yes", " Yes", "yes"))
    negative_ids = token_group(tokenizer, ("No", " No", "no"))

    support_prompts = []
    likelihood_prompts = []
    answers = []
    metadata = []
    for generation in local:
        query_id = int(generation["query_id"])
        question = str(steps[query_id]["step_question"])
        for branch in generation["branches"]:
            answer = str(branch.get("state_text", branch["generated_text"])).strip()
            evidence = str(branch["memory_text"])
            support_prompts.append(support_prompt(question, evidence, answer))
            likelihood_prompts.append(answer_likelihood_prompt(question, evidence))
            answers.append(answer)
        metadata.append(generation)

    yes_no_scores = score_yes_no(
        model,
        tokenizer,
        support_prompts,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        batch_size=args.batch_size,
        device=device,
    )
    likelihood_scores = score_answer_logprob(
        model,
        tokenizer,
        likelihood_prompts,
        answers,
        batch_size=args.batch_size,
        device=device,
    )
    branch_count = len(local[0]["branches"]) if local else 0
    rows = []
    for offset, generation in enumerate(metadata):
        start = offset * branch_count
        end = start + branch_count
        yes = yes_no_scores[start:end]
        likelihood = likelihood_scores[start:end]
        rows.append(
            {
                "query_id": int(generation["query_id"]),
                "yes_no_scores": yes,
                "answer_logprob_scores": likelihood,
                "yes_no_index": max(range(branch_count), key=yes.__getitem__),
                "answer_logprob_index": max(
                    range(branch_count), key=likelihood.__getitem__
                ),
                "branch_target_hits": [
                    bool(branch["target_hit"]) for branch in generation["branches"]
                ],
                "generated_answers": [
                    str(branch.get("state_text", branch["generated_text"]))
                    for branch in generation["branches"]
                ],
            }
        )
    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged = [
            row
            for shard_rank in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl")
        ]
        merged.sort(key=lambda row: int(row["query_id"]))
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        def accuracy(field: str) -> float:
            return statistics.fmean(
                bool(row["branch_target_hits"][int(row[field])]) for row in merged
            )

        summary = {
            "source": "distributed frozen 8B support and candidate-likelihood scoring",
            "selection_uses_gold": False,
            "queries": len(merged),
            "branches_per_query": branch_count,
            "world_size": world_size,
            "yes_no_accuracy": accuracy("yes_no_index"),
            "answer_logprob_accuracy": accuracy("answer_logprob_index"),
            "oracle_any_branch_accuracy": statistics.fmean(
                any(row["branch_target_hits"]) for row in merged
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
