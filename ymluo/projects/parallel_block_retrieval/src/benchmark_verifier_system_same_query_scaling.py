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

from evaluate_global_step_branch_generation import (
    concat_candidate_memories,
    render_atomic_prompt,
)
from evaluate_retrieved_answer_nll import resolve_dtype, setup_distributed
from evaluate_stepwise_set_utility import generate_step
from evaluate_transition_support_verifier import score_yes_no, support_prompt
from run_single_query_dynamic_kv_generation import answer_hit, normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure true per-request GPU scaling for the strongest 10M verifier "
            "trace: one bridge generation, 16 answer branches, and Yes/No selection."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--bridge_steps_path", required=True)
    parser.add_argument("--bridge_retrieval_rows_path", required=True)
    parser.add_argument("--answer_steps_path", required=True)
    parser.add_argument("--answer_generation_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup_queries", type=int, default=2)
    parser.add_argument("--max_queries", type=int, default=30)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--verifier_batch_size", type=int, default=8)
    parser.add_argument("--retrieval_seconds", type=float, default=0.0425)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def token_group(tokenizer: Any, values: tuple[str, ...]) -> list[int]:
    return list(
        dict.fromkeys(
            tokenizer(value, add_special_tokens=False)["input_ids"][0]
            for value in values
        )
    )


def timed_barrier(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()


def load_model_sequentially(
    rank: int,
    world_size: int,
    model_name_or_path: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    model = None
    for loading_rank in range(world_size):
        if rank == loading_rank:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(device)
            model.eval()
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        if dist.is_initialized():
            dist.barrier()
    if model is None:
        raise RuntimeError("model was not loaded")
    return model


def run_query(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    model: Any,
    tokenizer: Any,
    blocks: np.ndarray,
    bridge_step: dict[str, Any],
    bridge_retrieval: dict[str, Any],
    answer_step: dict[str, Any],
    answer_trace: dict[str, Any],
    branches: int,
    max_new_tokens: int,
    verifier_batch_size: int,
    positive_ids: list[int],
    negative_ids: list[int],
    retrieval_seconds: float,
) -> dict[str, Any] | None:
    timed_barrier(device)
    total_started = time.perf_counter()

    bridge_payload: list[dict[str, Any] | None] = [None]
    bridge_started = time.perf_counter()
    if rank == 0:
        bridge_candidates = bridge_retrieval["branch_candidates"][:3]
        memory, _context_tokens, _sources = concat_candidate_memories(
            bridge_candidates, blocks, tokenizer
        )
        prompt_ids = render_atomic_prompt(tokenizer, memory, bridge_step, [])
        generated, generated_tokens, generation_seconds = generate_step(
            model, tokenizer, prompt_ids, max_new_tokens, device
        )
        bridge_payload[0] = {
            "generated": generated.strip(),
            "generated_tokens": generated_tokens,
            "generation_seconds": generation_seconds,
        }
    if dist.is_initialized():
        dist.broadcast_object_list(bridge_payload, src=0)
    timed_barrier(device)
    bridge_wall = time.perf_counter() - bridge_started
    bridge_result = bridge_payload[0]
    if bridge_result is None:
        raise RuntimeError("bridge result was not broadcast")

    frozen_bridge = str(answer_step["compact_state_before"][0]).split(
        ":", maxsplit=1
    )[-1].strip()
    generated_bridge = str(bridge_result["generated"])
    bridge_matches_frozen = normalize_answer(generated_bridge) == normalize_answer(
        frozen_bridge
    )

    branch_inputs = answer_trace["branches"][:branches]
    generation_started = time.perf_counter()
    local_generations = []
    for branch_index, branch in enumerate(branch_inputs):
        if branch_index % world_size != rank:
            continue
        prompt_ids = render_atomic_prompt(
            tokenizer,
            str(branch["memory_text"]),
            answer_step,
            [str(item) for item in answer_step["compact_state_before"]],
        )
        generated, generated_tokens, generation_seconds = generate_step(
            model, tokenizer, prompt_ids, max_new_tokens, device
        )
        local_generations.append(
            {
                "branch_index": branch_index,
                "generated": generated.strip(),
                "generated_tokens": generated_tokens,
                "generation_seconds": generation_seconds,
            }
        )
    gathered_generations: list[list[dict[str, Any]] | None] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered_generations, local_generations)
    else:
        gathered_generations[0] = local_generations
    timed_barrier(device)
    generation_wall = time.perf_counter() - generation_started
    generations = sorted(
        [
            item
            for group in gathered_generations
            if group is not None
            for item in group
        ],
        key=lambda item: int(item["branch_index"]),
    )
    if len(generations) != branches:
        raise RuntimeError("answer generation did not return every branch")

    verifier_started = time.perf_counter()
    local_indices = [
        branch_index
        for branch_index in range(branches)
        if branch_index % world_size == rank
    ]
    local_prompts = [
        support_prompt(
            str(answer_step["step_question"]),
            str(branch_inputs[branch_index]["memory_text"]),
            str(generations[branch_index]["generated"]),
        )
        for branch_index in local_indices
    ]
    local_scores = score_yes_no(
        model,
        tokenizer,
        local_prompts,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        batch_size=verifier_batch_size,
        device=device,
    )
    local_verifier = [
        {"branch_index": branch_index, "score": score}
        for branch_index, score in zip(local_indices, local_scores)
    ]
    gathered_verifier: list[list[dict[str, Any]] | None] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered_verifier, local_verifier)
    else:
        gathered_verifier[0] = local_verifier
    timed_barrier(device)
    verifier_wall = time.perf_counter() - verifier_started
    total_model_wall = time.perf_counter() - total_started

    if rank != 0:
        return None
    verifier_rows = sorted(
        [
            item
            for group in gathered_verifier
            if group is not None
            for item in group
        ],
        key=lambda item: int(item["branch_index"]),
    )
    scores = [float(item["score"]) for item in verifier_rows]
    selected_index = max(range(branches), key=scores.__getitem__)
    selected_answer = str(generations[selected_index]["generated"])
    target = str(answer_step["target_output"])
    return {
        "query_id": int(answer_step["query_id"]),
        "world_size": world_size,
        "branches": branches,
        "bridge_wall_seconds": bridge_wall,
        "answer_generation_wall_seconds": generation_wall,
        "verifier_wall_seconds": verifier_wall,
        "model_wall_seconds": total_model_wall,
        "retrieval_seconds": retrieval_seconds,
        "estimated_online_wall_seconds": total_model_wall + retrieval_seconds,
        "bridge_generated": generated_bridge,
        "bridge_frozen": frozen_bridge,
        "bridge_matches_frozen": bridge_matches_frozen,
        "selected_index": selected_index,
        "selected_answer": selected_answer,
        "target_output": target,
        "answer_hit": answer_hit(selected_answer, [target]),
        "any_branch_answer_hit": any(
            answer_hit(str(item["generated"]), [target]) for item in generations
        ),
        "total_generated_tokens": int(bridge_result["generated_tokens"])
        + sum(int(item["generated_tokens"]) for item in generations),
    }


def main() -> None:
    args = parse_args()
    if args.max_queries <= 0 or args.warmup_queries < 0:
        raise ValueError("query counts must be positive")
    if args.branches <= 0 or args.verifier_batch_size <= 0:
        raise ValueError("branch and batch sizes must be positive")
    rank, world_size, _local_rank, device = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    bridge_steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.bridge_steps_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_bridge"
    }
    bridge_retrieval = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.bridge_retrieval_rows_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_bridge"
    }
    answer_steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.answer_steps_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_answer_from_bridge"
    }
    answer_traces = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.answer_generation_rows_path))
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_answer_from_bridge"
    }
    common_ids = sorted(
        set(bridge_steps) & set(bridge_retrieval) & set(answer_steps) & set(answer_traces)
    )
    requested = args.warmup_queries + args.max_queries
    if len(common_ids) < requested:
        raise ValueError("not enough aligned queries for warmup and measurement")
    query_ids = common_ids[:requested]

    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, local_files_only=True
    )
    model = load_model_sequentially(
        rank,
        world_size,
        args.model_name_or_path,
        resolve_dtype(args.dtype),
        device,
    )
    positive_ids = token_group(tokenizer, ("Yes", " Yes", "yes"))
    negative_ids = token_group(tokenizer, ("No", " No", "no"))

    rows = []
    for query_offset, query_id in enumerate(query_ids):
        row = run_query(
            rank=rank,
            world_size=world_size,
            device=device,
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            bridge_step=bridge_steps[query_id],
            bridge_retrieval=bridge_retrieval[query_id],
            answer_step=answer_steps[query_id],
            answer_trace=answer_traces[query_id],
            branches=args.branches,
            max_new_tokens=args.max_new_tokens,
            verifier_batch_size=args.verifier_batch_size,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            retrieval_seconds=args.retrieval_seconds,
        )
        if rank != 0 or row is None:
            continue
        row["warmup"] = query_offset < args.warmup_queries
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not row["warmup"]:
            rows.append(row)

    if rank == 0:
        if len(rows) != args.max_queries:
            raise RuntimeError("measured row count does not match max_queries")
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        def stage_summary(field: str) -> dict[str, float]:
            values = [float(row[field]) for row in rows]
            return {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
            }

        summary = {
            "source": "same-request strongest 10M verifier trace GPU scaling",
            "strict_wall_clock": True,
            "model_loading_included": False,
            "routing_trace_frozen": True,
            "retrieval_measured_separately": True,
            "retrieval_seconds_per_query": args.retrieval_seconds,
            "world_size": world_size,
            "queries": len(rows),
            "warmup_queries": args.warmup_queries,
            "branches": args.branches,
            "bridge": stage_summary("bridge_wall_seconds"),
            "answer_generation": stage_summary("answer_generation_wall_seconds"),
            "verifier": stage_summary("verifier_wall_seconds"),
            "model_total": stage_summary("model_wall_seconds"),
            "estimated_online_total": stage_summary("estimated_online_wall_seconds"),
            "answer_accuracy": statistics.fmean(row["answer_hit"] for row in rows),
            "oracle_any_branch_accuracy": statistics.fmean(
                row["any_branch_answer_hit"] for row in rows
            ),
            "bridge_replay_match_rate": statistics.fmean(
                row["bridge_matches_frozen"] for row in rows
            ),
            "mean_generated_tokens_per_second": sum(
                int(row["total_generated_tokens"]) for row in rows
            )
            / sum(float(row["model_wall_seconds"]) for row in rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
