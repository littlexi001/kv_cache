from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import answer_nll, resolve_dtype, setup_distributed
from evaluate_stepwise_set_utility import generate_step, render_prompt
from run_dynamic_kv_multisample import token_f1
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one reasoning step from KV-selected token spans."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", default="target_plus_negative")
    parser.add_argument("--method", default="operator_specialist_mean")
    parser.add_argument("--exclude_query_ids", default="375")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument(
        "--max_retrieval_branches", type=int, choices=[1, 2, 3, 4], default=1
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and str(row["task_type"]) == "multihop"
        and int(row["query_id"]) not in excluded_ids
    ]
    retrieval_rows = [
        row
        for row in read_jsonl(Path(args.retrieval_rows_path))
        if str(row["split"]) == args.split
        and str(row["scenario"]) == args.scenario
        and str(row["method"]) == args.method
        and int(row["query_id"]) not in excluded_ids
    ]
    retrieval_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in retrieval_rows
    }
    expected_keys = {(int(row["query_id"]), int(row["step_index"])) for row in steps}
    if set(retrieval_by_key) != expected_keys:
        raise ValueError("retrieval rows do not exactly cover the requested steps")

    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    local_rows = []
    for offset, step in enumerate(steps):
        if offset % world_size != rank:
            continue
        retrieval = retrieval_by_key[(int(step["query_id"]), int(step["step_index"]))]
        compact_state = [str(item) for item in step["compact_state_before"]]
        target = str(step["target_output"])
        target_ids = torch.tensor(
            tokenizer(" " + target, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        candidates = retrieval.get("top_candidates") or [
            {
                "rank": 1,
                "block_id": int(retrieval["selected_block"]),
                "start": int(retrieval["selected_start"]),
                "end": int(retrieval["selected_end"]),
                "target_overlap": 1.0 if retrieval["target_span_hit_at_1"] else 0.0,
            }
        ]
        branches = []
        for candidate in candidates[: args.max_retrieval_branches]:
            selected_tokens = blocks[int(candidate["block_id"])][
                int(candidate["start"]) : int(candidate["end"])
            ].tolist()
            memory = tokenizer.decode(selected_tokens, skip_special_tokens=True)
            prompt_ids = render_prompt(tokenizer, memory, step, compact_state)
            nll, target_tokens, nll_seconds = answer_nll(
                model,
                torch.tensor(prompt_ids, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                target_ids,
                device,
            )
            generated, generated_tokens, generation_seconds = generate_step(
                model,
                tokenizer,
                prompt_ids,
                args.max_new_tokens,
                device,
            )
            branches.append(
                {
                    "rank": int(candidate["rank"]),
                    "selected_block": int(candidate["block_id"]),
                    "selected_start": int(candidate["start"]),
                    "selected_end": int(candidate["end"]),
                    "retrieval_target_span_hit": float(candidate["target_overlap"])
                    >= 0.999,
                    "memory_text": memory,
                    "context_tokens": len(selected_tokens),
                    "target_tokens": target_tokens,
                    "target_nll": nll,
                    "target_ppl": math.exp(min(nll, 20.0)),
                    "nll_seconds": nll_seconds,
                    "generated_text": generated,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": generation_seconds,
                    "target_hit": answer_hit(generated, [target]),
                    "target_f1": token_f1(generated, target),
                }
            )
        top = branches[0]
        local_rows.append(
            {
                "query_id": int(step["query_id"]),
                "split": str(step["split"]),
                "step_index": int(step["step_index"]),
                "step_type": str(step["step_type"]),
                "scenario": args.scenario,
                "method": args.method,
                "selected_block": top["selected_block"],
                "selected_start": top["selected_start"],
                "selected_end": top["selected_end"],
                "retrieval_target_span_hit": top["retrieval_target_span_hit"],
                "retrieval_target_span_hit_at_2": any(
                    branch["retrieval_target_span_hit"] for branch in branches[:2]
                ),
                "retrieval_target_span_hit_at_k": any(
                    branch["retrieval_target_span_hit"] for branch in branches
                ),
                "memory_text": top["memory_text"],
                "context_tokens": top["context_tokens"],
                "target_output": target,
                "target_tokens": top["target_tokens"],
                "target_nll": top["target_nll"],
                "target_ppl": top["target_ppl"],
                "nll_seconds": top["nll_seconds"],
                "generated_text": top["generated_text"],
                "generated_tokens": top["generated_tokens"],
                "generation_seconds": top["generation_seconds"],
                "target_hit": top["target_hit"],
                "target_f1": top["target_f1"],
                "any_branch_target_hit": any(branch["target_hit"] for branch in branches),
                "total_branch_generation_seconds": sum(
                    branch["generation_seconds"] for branch in branches
                ),
                "parallel_branch_critical_seconds": max(
                    branch["generation_seconds"] for branch in branches
                ),
                "branches": branches,
            }
        )

    shard_path = output_dir / f"rows_rank{rank:03d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        rows = [
            row
            for shard_rank in range(world_size)
            for row in read_jsonl(output_dir / f"rows_rank{shard_rank:03d}.jsonl")
        ]
        rows.sort(key=lambda row: (row["query_id"], row["step_index"]))
        summaries = []
        for step_type in sorted({row["step_type"] for row in rows}):
            group = [row for row in rows if row["step_type"] == step_type]
            retrieved = [row for row in group if row["retrieval_target_span_hit"]]
            summaries.append(
                {
                    "step_type": step_type,
                    "steps": len(group),
                    "retrieval_target_span_recall": statistics.fmean(
                        row["retrieval_target_span_hit"] for row in group
                    ),
                    "retrieval_target_span_recall_at_2": statistics.fmean(
                        row["retrieval_target_span_hit_at_2"] for row in group
                    ),
                    "retrieval_target_span_recall_at_k": statistics.fmean(
                        row["retrieval_target_span_hit_at_k"] for row in group
                    ),
                    "end_to_end_target_hit_rate": statistics.fmean(
                        row["target_hit"] for row in group
                    ),
                    "target_hit_given_retrieval": statistics.fmean(
                        row["target_hit"] for row in retrieved
                    )
                    if retrieved
                    else 0.0,
                    "oracle_any_branch_target_hit_rate": statistics.fmean(
                        row["any_branch_target_hit"] for row in group
                    ),
                    "mean_target_nll": statistics.fmean(row["target_nll"] for row in group),
                    "mean_target_f1": statistics.fmean(row["target_f1"] for row in group),
                    "mean_generation_seconds": statistics.fmean(
                        row["generation_seconds"] for row in group
                    ),
                    "mean_total_branch_generation_seconds": statistics.fmean(
                        row["total_branch_generation_seconds"] for row in group
                    ),
                    "mean_parallel_branch_critical_seconds": statistics.fmean(
                        row["parallel_branch_critical_seconds"] for row in group
                    ),
                }
            )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {
            "source": "Qwen generation from real step-Q/K selected token spans",
            "contains_synthetic_vectors": False,
            "selection_uses_test_gold": False,
            "world_size": world_size,
            "split": args.split,
            "scenario": args.scenario,
            "method": args.method,
            "max_retrieval_branches": args.max_retrieval_branches,
            "excluded_query_ids": sorted(excluded_ids),
            "steps": len(rows),
            "selected_span_tokens": sorted(
                {int(row["context_tokens"]) for row in rows}
            ),
            "summaries": summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
