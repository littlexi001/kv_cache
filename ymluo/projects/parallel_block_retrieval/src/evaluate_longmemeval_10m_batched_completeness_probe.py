from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_longmemeval_10m_pairwise_set_utility_probe import (
    completeness_prompt,
    format_pages,
    ordered_difference,
    read_jsonl,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate static and dynamic working-set completeness in one batched model "
            "forward while preserving the validated independent scoring protocol."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selection_rows", required=True)
    parser.add_argument("--state_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


@torch.inference_mode()
def batched_log_odds(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    yes_id: int,
    no_id: int,
    device: torch.device,
) -> tuple[list[float], float, list[int], int]:
    sequences = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    lengths = [len(sequence) for sequence in sequences]
    batch = tokenizer.pad(
        {"input_ids": sequences}, padding=True, return_tensors="pt"
    ).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits[:, -1]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    scores = (
        logits[:, yes_id].float() - logits[:, no_id].float()
    ).detach().cpu().tolist()
    return [float(score) for score in scores], seconds, lengths, int(batch["input_ids"].numel())


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(data_dir / "queries.jsonl")
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    query_ids = {int(row["query_id"]) for row in queries}
    states = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.state_rows))
        if int(row["query_id"]) in query_ids
    }
    selections: dict[str, dict[int, dict[str, Any]]] = {
        "static_top12": {},
        "evidence_state_dynamic_top12": {},
    }
    for row in read_jsonl(Path(args.selection_rows)):
        method = str(row["method"])
        query_id = int(row["query_id"])
        if method in selections and query_id in query_ids:
            selections[method][query_id] = row
    if any(set(rows) != query_ids for rows in selections.values()) or set(states) != query_ids:
        raise ValueError("states and selections must cover every query")

    all_block_ids: set[int] = set()
    for query_id in query_ids:
        all_block_ids.update(map(int, states[query_id]["initial_block_ids"]))
        for method in selections:
            all_block_ids.update(
                map(int, selections[method][query_id]["top_block_ids"])
            )
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    block_texts = {
        block_id: tokenizer.decode(
            np.asarray(base_blocks[block_id], dtype=np.int64), skip_special_tokens=True
        )
        for block_id in sorted(all_block_ids)
    }
    yes = tokenizer("YES", add_special_tokens=False)["input_ids"]
    no = tokenizer("NO", add_special_tokens=False)["input_ids"]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError(f"expected single-token labels, got YES={yes}, NO={no}")

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    for index, query in enumerate(queries):
        query_id = int(query["query_id"])
        state = states[query_id]
        initial_ids = list(map(int, state["initial_block_ids"]))
        initial_set = set(initial_ids)
        static_ids = list(map(int, selections["static_top12"][query_id]["top_block_ids"]))
        dynamic_ids = list(
            map(
                int,
                selections["evidence_state_dynamic_top12"][query_id]["top_block_ids"],
            )
        )
        static_extra = ordered_difference(static_ids, initial_set)
        dynamic_extra = ordered_difference(dynamic_ids, initial_set)
        identical = static_extra == dynamic_extra
        if identical:
            static_score = 0.0
            dynamic_score = 0.0
            seconds = 0.0
            prompt_lengths = [0, 0]
            padded_tokens = 0
        else:
            static_context = format_pages(
                initial_ids + static_extra, block_texts, label="Working-set"
            )
            dynamic_context = format_pages(
                initial_ids + dynamic_extra, block_texts, label="Working-set"
            )
            scores, seconds, prompt_lengths, padded_tokens = batched_log_odds(
                model,
                tokenizer,
                [
                    completeness_prompt(
                        str(query["question"]),
                        str(state["state_text"]),
                        static_context,
                    ),
                    completeness_prompt(
                        str(query["question"]),
                        str(state["state_text"]),
                        dynamic_context,
                    ),
                ],
                yes_id=int(yes[0]),
                no_id=int(no[0]),
                device=device,
            )
            static_score, dynamic_score = scores
        rows.append(
            {
                "query_id": query_id,
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "sets_identical": identical,
                "batched_static_completeness_log_odds": static_score,
                "batched_dynamic_completeness_log_odds": dynamic_score,
                "batched_completeness_utility_score": dynamic_score - static_score,
                "batch_seconds": seconds,
                "static_prompt_tokens": int(prompt_lengths[0]),
                "dynamic_prompt_tokens": int(prompt_lengths[1]),
                "batch_padded_tokens": padded_tokens,
                "selection_uses_answer": False,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "queries": len(queries),
                    "query_id": query_id,
                    "score": dynamic_score - static_score,
                    "seconds": seconds,
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    changed = [row for row in rows if not row["sets_identical"]]
    summary = {
        "source": "Qwen3-8B batched independent-completeness probe on LongMemEval 10M",
        "protocol": {
            "selection_uses_answer": False,
            "probe_generates_answer": False,
            "logical_scores": 2,
            "gpu_model_calls_per_edit": 1,
        },
        "model_name_or_path": args.model_name_or_path,
        "queries": len(rows),
        "changed_candidate_sets": len(changed),
        "mean_batch_seconds_changed": mean(row["batch_seconds"] for row in changed),
        "mean_logical_prompt_tokens_changed": mean(
            row["static_prompt_tokens"] + row["dynamic_prompt_tokens"] for row in changed
        ),
        "mean_padded_tokens_changed": mean(row["batch_padded_tokens"] for row in changed),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
