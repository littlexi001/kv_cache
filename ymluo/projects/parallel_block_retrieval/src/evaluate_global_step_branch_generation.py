from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import resolve_dtype, setup_distributed
from evaluate_stepwise_set_utility import generate_step, render_prompt
from run_dynamic_kv_multisample import token_f1
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one verified reasoning step from global block/span branches."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--step_types",
        default="",
        help="Optional comma-separated step types to evaluate.",
    )
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--max_retrieval_branches", type=int, default=9)
    parser.add_argument(
        "--branch_mode",
        choices=["independent", "concat"],
        default="independent",
        help=(
            "Generate once per candidate block, or concatenate all selected blocks "
            "into one evidence set and generate once."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--prompt_mode",
        choices=[
            "legacy",
            "atomic",
            "adaptive",
            "extractive",
            "adaptive_extract",
            "bridge_reasoned",
            "support_extract",
        ],
        default="legacy",
        help="Use the existing chain prompt or a minimal atomic extraction prompt.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_atomic_prompt(
    tokenizer: Any,
    memory: str,
    step: dict[str, Any],
    compact_state: list[str],
) -> list[int]:
    facts = [
        item.split(":", maxsplit=1)[-1].strip()
        if item.startswith("BRIDGE_ENTITY:")
        else item
        for item in compact_state
    ]
    known = ", ".join(facts) if facts else "(none)"
    content = (
        "Evidence:\n"
        f"{memory}\n\n"
        f"Question: {step['step_question']}\n"
        f"Known verified entities: {known}\n"
        "Return only the shortest exact answer supported by the evidence. "
        "Ignore unrelated text."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_extractive_prompt(
    tokenizer: Any,
    memory: str,
    step: dict[str, Any],
    compact_state: list[str],
) -> list[int]:
    facts = [
        item.split(":", maxsplit=1)[-1].strip()
        if item.startswith("BRIDGE_ENTITY:")
        else item
        for item in compact_state
    ]
    known = ", ".join(facts) if facts else "(none)"
    content = (
        "Evidence:\n"
        f"{memory}\n\n"
        f"Question: {step['step_question']}\n"
        f"Known verified entities: {known}\n"
        "Copy the shortest exact contiguous answer span from the evidence. "
        "Output only that span, with no explanation or label. Do not repeat a known "
        "entity unless it is itself the answer."
    )


def render_bridge_reasoned_prompt(
    tokenizer: Any,
    memory: str,
    step: dict[str, Any],
    compact_state: list[str],
) -> list[int]:
    facts = [
        item.split(":", maxsplit=1)[-1].strip()
        if item.startswith("BRIDGE_ENTITY:")
        else item
        for item in compact_state
    ]
    known = ", ".join(facts) if facts else "(none)"
    content = (
        "Evidence:\n"
        f"{memory}\n\n"
        f"Question: {step['step_question']}\n"
        f"Known verified entities: {known}\n"
        "Resolve the missing relation using only the evidence. Output exactly two lines:\n"
        "Relevant fact: <copy one shortest evidence sentence that answers the question>\n"
        "Bridge entity: <only the shortest missing entity or value>"
    )


def render_support_extract_prompt(
    tokenizer: Any,
    memory: str,
    step: dict[str, Any],
    compact_state: list[str],
) -> list[int]:
    facts = [
        item.split(":", maxsplit=1)[-1].strip()
        if item.startswith("BRIDGE_ENTITY:")
        else item
        for item in compact_state
    ]
    known = ", ".join(facts) if facts else "(none)"
    content = (
        "Candidate evidence:\n"
        f"{memory}\n\n"
        f"Atomic question: {step['step_question']}\n"
        f"Known verified entities: {known}\n"
        "Decide whether this candidate evidence directly supports one unambiguous "
        "answer to the atomic question. Use only this evidence. Output exactly one line:\n"
        "SUPPORTED: <copy only the shortest exact answer span from the evidence>\n"
        "or\n"
        "NOT_SUPPORTED"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generated_state_text(generated: str, prompt_mode: str) -> str:
    if prompt_mode == "support_extract":
        if re.search(r"\bNOT[_ ]SUPPORTED\b", generated, flags=re.IGNORECASE):
            return ""
        matches = re.findall(
            r"(?:^|\n)\s*SUPPORTED\s*:\s*(.+?)\s*(?=\n|$)",
            generated,
            flags=re.IGNORECASE,
        )
        return matches[-1].strip() if matches else ""
    if prompt_mode != "bridge_reasoned":
        return generated
    matches = re.findall(
        r"(?:^|\n)\s*Bridge\s+entity\s*:\s*(.+?)\s*(?=\n|$)",
        generated,
        flags=re.IGNORECASE,
    )
    return matches[-1].strip() if matches else ""


def candidate_segments(candidate: dict[str, Any]) -> list[list[int]]:
    return [
        [int(start), int(end)]
        for start, end in candidate.get(
            "segments",
            [[int(candidate["start"]), int(candidate["end"])]],
        )
    ]


def candidate_memory(
    candidate: dict[str, Any], blocks: np.ndarray, tokenizer: Any
) -> tuple[list[list[int]], list[list[int]], str]:
    segments = candidate_segments(candidate)
    segment_tokens = [
        blocks[int(candidate["block_id"])][start:end].tolist()
        for start, end in segments
    ]
    memory = "\n".join(
        tokenizer.decode(segment, skip_special_tokens=True)
        for segment in segment_tokens
    )
    return segments, segment_tokens, memory


def concat_candidate_memories(
    candidates: list[dict[str, Any]], blocks: np.ndarray, tokenizer: Any
) -> tuple[str, int, list[dict[str, Any]]]:
    memories = []
    sources = []
    context_tokens = 0
    for evidence_index, candidate in enumerate(candidates, start=1):
        segments, segment_tokens, memory = candidate_memory(
            candidate, blocks, tokenizer
        )
        context_tokens += sum(len(segment) for segment in segment_tokens)
        memories.append(f"[Evidence {evidence_index}]\n{memory}")
        sources.append(
            {
                "rank": int(candidate["rank"]),
                "block_rank": int(candidate["block_rank"]),
                "span_rank": int(candidate["span_rank"]),
                "block_id": int(candidate["block_id"]),
                "segments": segments,
                "target_span_hit": float(candidate["target_overlap"]) >= 0.8,
            }
        )
    return "\n\n".join(memories), context_tokens, sources
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    args = parse_args()
    if args.max_retrieval_branches <= 0:
        raise ValueError("max_retrieval_branches must be positive")
    rank, world_size, _local_rank, device = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    allowed_step_types = {
        item.strip() for item in args.step_types.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and str(row["task_type"]) == "multihop"
        and int(row["query_id"]) not in excluded_ids
        and (
            not allowed_step_types
            or str(row["step_type"]) in allowed_step_types
        )
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    expected = {(int(row["query_id"]), int(row["step_index"])) for row in steps}
    retrieval_by_key = {
        key: row
        for row in read_jsonl(Path(args.retrieval_rows_path))
        if str(row["split"]) == args.split
        and int(row["query_id"]) not in excluded_ids
        and (key := (int(row["query_id"]), int(row["step_index"]))) in expected
    }
    if set(retrieval_by_key) != expected:
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
        key = (int(step["query_id"]), int(step["step_index"]))
        retrieval = retrieval_by_key[key]
        compact_state = [str(item) for item in step["compact_state_before"]]
        target = str(step["target_output"])
        candidates = retrieval["branch_candidates"][: args.max_retrieval_branches]
        branches = []
        generation_inputs: list[tuple[dict[str, Any], str, int, list[dict[str, Any]]]]
        if args.branch_mode == "concat":
            memory, context_tokens, sources = concat_candidate_memories(
                candidates, blocks, tokenizer
            )
            generation_inputs = [(candidates[0], memory, context_tokens, sources)]
        else:
            generation_inputs = []
            for candidate in candidates:
                segments, segment_tokens, memory = candidate_memory(
                    candidate, blocks, tokenizer
                )
                context_tokens = sum(len(segment) for segment in segment_tokens)
                sources = [
                    {
                        "rank": int(candidate["rank"]),
                        "block_rank": int(candidate["block_rank"]),
                        "span_rank": int(candidate["span_rank"]),
                        "block_id": int(candidate["block_id"]),
                        "segments": segments,
                        "target_span_hit": float(candidate["target_overlap"]) >= 0.8,
                    }
                ]
                generation_inputs.append(
                    (candidate, memory, context_tokens, sources)
                )
        for candidate, memory, context_tokens, sources in generation_inputs:
            if args.prompt_mode == "support_extract":
                prompt_ids = render_support_extract_prompt(
                    tokenizer, memory, step, compact_state
                )
            elif args.prompt_mode == "bridge_reasoned":
                prompt_ids = render_bridge_reasoned_prompt(
                    tokenizer, memory, step, compact_state
                )
            elif args.prompt_mode == "extractive" or (
                args.prompt_mode == "adaptive_extract"
                and step["step_type"] == "resolve_answer_from_bridge"
            ):
                prompt_ids = render_extractive_prompt(
                    tokenizer, memory, step, compact_state
                )
            elif args.prompt_mode == "atomic" or (
                args.prompt_mode in {"adaptive", "adaptive_extract"}
                and step["step_type"] == "resolve_bridge"
            ):
                prompt_ids = render_atomic_prompt(tokenizer, memory, step, compact_state)
            else:
                prompt_ids = render_prompt(tokenizer, memory, step, compact_state)
            generated, generated_tokens, generation_seconds = generate_step(
                model,
                tokenizer,
                prompt_ids,
                args.max_new_tokens,
                device,
            )
            state_text = generated_state_text(generated, args.prompt_mode)
            branches.append(
                {
                    "rank": int(candidate["rank"]),
                    "block_rank": int(candidate["block_rank"]),
                    "span_rank": int(candidate["span_rank"]),
                    "selected_block": int(candidate["block_id"]),
                    "selected_start": int(candidate["start"]),
                    "selected_end": int(candidate["end"]),
                    "selected_segments": [
                        [int(start), int(end)]
                        for start, end in candidate_segments(candidate)
                    ],
                    "selected_sources": sources,
                    "retrieval_target_span_hit": any(
                        source["target_span_hit"] for source in sources
                    ),
                    "memory_text": memory,
                    "context_tokens": context_tokens,
                    "generated_text": generated,
                    "state_text": state_text,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": generation_seconds,
                    "target_hit": answer_hit(state_text, [target]),
                    "target_f1": token_f1(state_text, target),
                }
            )
        top = branches[0]
        local_rows.append(
            {
                "query_id": key[0],
                "split": str(step["split"]),
                "step_index": key[1],
                "step_type": str(step["step_type"]),
                "selection_uses_gold": bool(
                    retrieval.get("selection_uses_gold", False)
                ),
                "target_output": target,
                "retrieval_target_span_hit": top["retrieval_target_span_hit"],
                "retrieval_target_span_hit_at_k": any(
                    branch["retrieval_target_span_hit"] for branch in branches
                ),
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
        print(
            json.dumps(
                {
                    "rank": rank,
                    "query_id": key[0],
                    "step_index": key[1],
                    "branches": len(branches),
                    "retrieval_hit": local_rows[-1]["retrieval_target_span_hit_at_k"],
                    "generation_hit": local_rows[-1]["any_branch_target_hit"],
                }
            ),
            flush=True,
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
            summaries.append(
                {
                    "step_type": step_type,
                    "steps": len(group),
                    "retrieval_target_span_recall_at_k": statistics.fmean(
                        row["retrieval_target_span_hit_at_k"] for row in group
                    ),
                    "top1_target_hit_rate": statistics.fmean(
                        row["target_hit"] for row in group
                    ),
                    "oracle_any_branch_target_hit_rate": statistics.fmean(
                        row["any_branch_target_hit"] for row in group
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
            "source": "Qwen step generation from 10M anchor-block/specialist-span branches",
            "contains_synthetic_vectors": False,
            "selection_uses_gold": any(
                bool(row["selection_uses_gold"]) for row in rows
            ),
            "world_size": world_size,
        "split": args.split,
        "step_types": sorted(allowed_step_types) if allowed_step_types else None,
        "max_retrieval_branches": args.max_retrieval_branches,
        "branch_mode": args.branch_mode,
        "prompt_mode": args.prompt_mode,
            "steps": len(rows),
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
