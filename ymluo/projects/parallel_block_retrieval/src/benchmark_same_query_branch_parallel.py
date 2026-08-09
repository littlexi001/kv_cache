from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import resolve_dtype, setup_distributed
from evaluate_stepwise_set_utility import generate_step, render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure true same-query branch wall time across distributed GPUs."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--step_types", default="resolve_bridge,resolve_answer_from_bridge")
    parser.add_argument("--exclude_query_ids", default="375")
    parser.add_argument("--max_queries", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda"], default="cuda")
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
    step_types = {item.strip() for item in args.step_types.split(",") if item.strip()}
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) in step_types
        and int(row["query_id"]) not in excluded_ids
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_queries > 0:
        steps = steps[: args.max_queries]
    retrieval_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.retrieval_rows_path))
    }
    if any(
        (int(step["query_id"]), int(step["step_index"])) not in retrieval_by_key
        for step in steps
    ):
        raise ValueError("retrieval rows do not cover all benchmark steps")

    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    for step in steps:
        key = (int(step["query_id"]), int(step["step_index"]))
        candidates = retrieval_by_key[key]["branch_candidates"][: args.branches]
        compact_state = [str(item) for item in step["compact_state_before"]]
        if world_size > 1:
            dist.barrier()
        wall_started = time.perf_counter()
        local_results = []
        for branch_index, candidate in enumerate(candidates):
            if branch_index % world_size != rank:
                continue
            selected_tokens = blocks[int(candidate["block_id"])][
                int(candidate["start"]) : int(candidate["end"])
            ].tolist()
            memory = tokenizer.decode(selected_tokens, skip_special_tokens=True)
            prompt_ids = render_prompt(tokenizer, memory, step, compact_state)
            generated, generated_tokens, generation_seconds = generate_step(
                model,
                tokenizer,
                prompt_ids,
                args.max_new_tokens,
                device,
            )
            local_results.append(
                {
                    "branch_index": branch_index,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": generated_tokens,
                    "generated_text": generated,
                }
            )
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gathered, local_results)
            dist.barrier()
        else:
            gathered[0] = local_results
        wall_seconds = time.perf_counter() - wall_started
        if rank != 0:
            continue
        branch_results = sorted(
            [item for group in gathered if group for item in group],
            key=lambda item: int(item["branch_index"]),
        )
        serial_sum = sum(float(item["generation_seconds"]) for item in branch_results)
        rows.append(
            {
                "query_id": key[0],
                "step_index": key[1],
                "step_type": str(step["step_type"]),
                "world_size": world_size,
                "branches": len(branch_results),
                "wall_seconds": wall_seconds,
                "serial_branch_seconds_sum": serial_sum,
                "effective_speedup": serial_sum / max(wall_seconds, 1.0e-9),
                "branch_results": branch_results,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    if rank == 0:
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": "true same-query branch distribution across GPUs",
            "world_size": world_size,
            "steps": len(rows),
            "branches": args.branches,
            "max_new_tokens": args.max_new_tokens,
            "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in rows),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
            "mean_effective_speedup": statistics.fmean(
                row["effective_speedup"] for row in rows
            ),
            "mean_generated_tokens_per_second": sum(
                item["generated_tokens"] for row in rows for item in row["branch_results"]
            )
            / sum(row["wall_seconds"] for row in rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
