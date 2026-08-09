from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import answer_nll, resolve_dtype, setup_distributed
from run_dynamic_kv_multisample import token_f1
from run_single_query_dynamic_kv_generation import answer_hit


STEP0_MODES = (
    "fact_only",
    "target_only",
    "target_span",
    "target_plus_negative",
    "negative_only",
    "full_record",
)
STEP1_MODES = (
    "fact_with_state",
    "fact_with_full_state",
    "fact_no_state",
    "target_with_state",
    "target_span_with_state",
    "target_with_full_state",
    "target_span_with_full_state",
    "target_no_state",
    "target_span_no_state",
    "target_plus_previous_with_state",
    "target_plus_negative_with_state",
    "previous_with_state",
    "full_record_no_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-step minimal evidence and compact-state sufficiency."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_ids(*groups: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(item) for group in groups for item in group))


def lexical_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def select_evidence_span(memory: str, query_text: str, max_sentences: int = 1) -> str:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", memory)
        if item.strip()
    ]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    sentence_terms = [lexical_tokens(sentence) for sentence in sentences]
    query_terms = lexical_tokens(query_text)
    document_frequency = Counter(
        term for terms in sentence_terms for term in terms if term in query_terms
    )
    count = len(sentences)
    scored = []
    for index, terms in enumerate(sentence_terms):
        overlap = terms & query_terms
        score = sum(
            math.log((count + 1.0) / (document_frequency[term] + 0.5))
            for term in overlap
        )
        score /= max(1.0, len(terms) ** 0.2)
        scored.append((score, index))
    selected = sorted(
        (index for _score, index in sorted(scored, reverse=True)[:max_sentences])
    )
    return " ".join(sentences[index] for index in selected)


def build_span_query(step: dict[str, Any], compact_state: Sequence[str]) -> str:
    return " ".join(
        [
            str(step.get("lookup_key", "")),
            str(step["step_question"]),
            str(step["question"]),
            *[str(item) for item in compact_state],
        ]
    )


def mode_inputs(step: dict[str, Any], mode: str) -> tuple[list[int], list[str]]:
    target = [int(item) for item in step["target_block_ids"]]
    previous = [int(item) for item in step["previous_evidence_block_ids"]]
    negatives = [int(item) for item in step["hard_negative_block_ids"][:1]]
    record = list(
        range(
            int(step["block_start"]),
            int(step["block_start"]) + int(step["block_count"]),
        )
    )
    state = [str(item) for item in step["compact_state_before"]]
    full_state = [str(item) for item in step.get("full_state_before", state)]
    if mode == "fact_only":
        return [], []
    if mode == "fact_with_state":
        return [], state
    if mode == "fact_with_full_state":
        return [], full_state
    if mode == "fact_no_state":
        return [], []
    if mode == "target_only":
        return target, []
    if mode == "target_span":
        return target, []
    if mode == "target_plus_negative":
        return unique_ids(target, negatives), []
    if mode == "negative_only":
        return negatives, []
    if mode == "full_record":
        return record, []
    if mode == "target_with_state":
        return target, state
    if mode == "target_span_with_state":
        return target, state
    if mode == "target_with_full_state":
        return target, full_state
    if mode == "target_span_with_full_state":
        return target, full_state
    if mode == "target_no_state":
        return target, []
    if mode == "target_span_no_state":
        return target, []
    if mode == "target_plus_previous_with_state":
        return unique_ids(target, previous), state
    if mode == "target_plus_negative_with_state":
        return unique_ids(target, negatives), state
    if mode == "previous_with_state":
        return previous, state
    if mode == "full_record_no_state":
        return record, []
    raise ValueError(f"unknown mode: {mode}")


def mode_memory(
    tokenizer: Any,
    blocks: np.ndarray,
    step: dict[str, Any],
    mode: str,
    block_ids: Sequence[int],
    compact_state: Sequence[str],
) -> tuple[str, str]:
    if mode in {
        "fact_only",
        "fact_with_state",
        "fact_with_full_state",
        "fact_no_state",
    }:
        return str(step["target_fact"]), "exact_target_fact"
    decoded = "\n\n".join(
        tokenizer.decode(blocks[block_id].tolist(), skip_special_tokens=True)
        for block_id in block_ids
    )
    if mode.startswith("target_span"):
        query_text = build_span_query(step, compact_state)
        return select_evidence_span(decoded, query_text), "automatic_evidence_span"
    return decoded, "decoded_blocks"


def render_prompt(
    tokenizer: Any,
    memory: str,
    step: dict[str, Any],
    compact_state: Sequence[str],
) -> list[int]:
    state_text = "\n".join(compact_state) if compact_state else "(none)"
    if step["step_type"] == "resolve_bridge":
        lookup_key = str(step["lookup_key"])
        instruction = (
            "Output only the linked entity name. The answer must be different from "
            f"the known identifier {lookup_key}; do not repeat that identifier."
        )
    elif step["step_type"] == "resolve_answer_from_bridge":
        instruction = (
            "Use the verified compact state and current memory to resolve the remaining "
            "relation. Output only the final value."
        )
    else:
        instruction = "Resolve the current step and output only its value."
    original_question = (
        ""
        if step["step_type"] == "resolve_bridge"
        else f"Original question: {step['question']}\n"
    )
    content = (
        "Memory:\n"
        f"{memory}\n\n"
        f"{original_question}"
        f"Current reasoning subproblem: {step['step_question']}\n"
        "Verified compact state from earlier steps:\n"
        f"{state_text}\n\n"
        f"{instruction}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def generate_step(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int, float]:
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated), elapsed


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    selected_query_ids = {
        int(item.strip()) for item in args.query_ids.split(",") if item.strip()
    }
    excluded_query_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    requested_modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    steps = [
        item
        for item in read_jsonl(Path(args.step_queries_path))
        if (not allowed_splits or str(item["split"]) in allowed_splits)
        and (not allowed_tasks or str(item["task_type"]) in allowed_tasks)
        and (not selected_query_ids or int(item["query_id"]) in selected_query_ids)
        and int(item["query_id"]) not in excluded_query_ids
    ]
    steps.sort(key=lambda item: (int(item["query_id"]), int(item["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
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
    for step_offset, step in enumerate(steps):
        if step_offset % world_size != rank:
            continue
        default_modes = STEP0_MODES if int(step["step_index"]) == 0 else STEP1_MODES
        modes = [mode for mode in default_modes if not requested_modes or mode in requested_modes]
        for mode in modes:
            block_ids, compact_state = mode_inputs(step, mode)
            memory, memory_kind = mode_memory(
                tokenizer,
                blocks,
                step,
                mode,
                block_ids,
                compact_state,
            )
            memory_tokens = len(tokenizer(memory, add_special_tokens=False)["input_ids"])
            prompt_ids = render_prompt(tokenizer, memory, step, compact_state)
            target = str(step["target_output"])
            target_ids = torch.tensor(
                tokenizer(" " + target, add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
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
            row = {
                "query_id": int(step["query_id"]),
                "split": str(step["split"]),
                "task_type": str(step["task_type"]),
                "step_index": int(step["step_index"]),
                "step_type": str(step["step_type"]),
                "mode": mode,
                "memory_kind": memory_kind,
                "memory_text": memory if memory_kind != "decoded_blocks" else None,
                "block_ids": block_ids,
                "context_blocks": len(block_ids),
                "context_tokens": memory_tokens,
                "compact_state_facts": len(compact_state),
                "target_output": target,
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
            local_rows.append(row)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "query_id": row["query_id"],
                        "step": row["step_index"],
                        "mode": mode,
                        "nll": nll,
                        "hit": row["target_hit"],
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
        rows.sort(key=lambda row: (row["query_id"], row["step_index"], row["mode"]))
        summaries = []
        keys = sorted({(row["step_type"], row["mode"]) for row in rows})
        for step_type, mode in keys:
            group = [
                row for row in rows if row["step_type"] == step_type and row["mode"] == mode
            ]
            summaries.append(
                {
                    "step_type": step_type,
                    "mode": mode,
                    "steps": len(group),
                    "mean_target_nll": statistics.fmean(row["target_nll"] for row in group),
                    "median_target_nll": statistics.median(row["target_nll"] for row in group),
                    "target_hit_rate": statistics.fmean(row["target_hit"] for row in group),
                    "mean_target_f1": statistics.fmean(row["target_f1"] for row in group),
                    "mean_context_blocks": statistics.fmean(
                        row["context_blocks"] for row in group
                    ),
                    "mean_generation_seconds": statistics.fmean(
                        row["generation_seconds"] for row in group
                    ),
                }
            )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {
            "source": "real Qwen step NLL and generation on controlled text blocks",
            "contains_synthetic_vectors": False,
            "world_size": world_size,
            "splits": sorted(allowed_splits),
            "task_types": sorted(allowed_tasks),
            "query_ids": sorted(selected_query_ids),
            "excluded_query_ids": sorted(excluded_query_ids),
            "steps": len(steps),
            "evaluations": len(rows),
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
