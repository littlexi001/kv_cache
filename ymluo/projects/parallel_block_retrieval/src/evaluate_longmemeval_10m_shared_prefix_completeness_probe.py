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
    binary_log_odds,
    completeness_prompt,
    format_pages,
    ordered_difference,
    read_jsonl,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score two independent completeness branches after computing their exact "
            "common token prefix once and reusing its KV cache."
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
    parser.add_argument(
        "--also_sequential",
        action="store_true",
        help="Run the original two sequential forwards in the same process for paired timing.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def common_prefix_length(first: list[int], second: list[int]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


@torch.inference_mode()
def shared_prefix_log_odds(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    yes_id: int,
    no_id: int,
    device: torch.device,
) -> tuple[list[float], dict[str, float | int | list[int]]]:
    if len(prompts) != 2:
        raise ValueError("the shared-prefix probe expects exactly two branches")
    sequences = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    prefix_length = common_prefix_length(sequences[0], sequences[1])
    suffixes = [sequence[prefix_length:] for sequence in sequences]
    if prefix_length == 0 or not all(suffixes):
        raise RuntimeError(
            f"invalid branch split prefix={prefix_length}, suffixes={list(map(len, suffixes))}"
        )
    prefix_ids = torch.tensor([sequences[0][:prefix_length]], dtype=torch.long, device=device)
    prefix_mask = torch.ones_like(prefix_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_started = time.perf_counter()
    prefix_started = total_started
    prefix_output = model(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefix_seconds = time.perf_counter() - prefix_started

    cache = prefix_output.past_key_values
    cache.batch_repeat_interleave(2)
    max_suffix = max(map(len, suffixes))
    pad_id = int(tokenizer.pad_token_id)
    suffix_ids = torch.full(
        (2, max_suffix), pad_id, dtype=torch.long, device=device
    )
    suffix_mask = torch.zeros((2, max_suffix), dtype=torch.long, device=device)
    for branch, suffix in enumerate(suffixes):
        length = len(suffix)
        suffix_ids[branch, :length] = torch.tensor(
            suffix, dtype=torch.long, device=device
        )
        suffix_mask[branch, :length] = 1
    full_mask = torch.cat(
        [
            torch.ones((2, prefix_length), dtype=torch.long, device=device),
            suffix_mask,
        ],
        dim=1,
    )
    suffix_started = time.perf_counter()
    suffix_output = model(
        input_ids=suffix_ids,
        attention_mask=full_mask,
        past_key_values=cache,
        use_cache=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    suffix_seconds = time.perf_counter() - suffix_started
    total_seconds = time.perf_counter() - total_started
    scores = []
    for branch, suffix in enumerate(suffixes):
        logits = suffix_output.logits[branch, len(suffix) - 1]
        scores.append(float((logits[yes_id] - logits[no_id]).float().item()))
    return scores, {
        "prefix_tokens": prefix_length,
        "suffix_tokens": list(map(len, suffixes)),
        "padded_suffix_tokens": int(suffix_ids.numel()),
        "logical_prompt_tokens": int(sum(map(len, sequences))),
        "executed_tokens": int(prefix_length + suffix_ids.numel()),
        "prefix_seconds": prefix_seconds,
        "suffix_seconds": suffix_seconds,
        "total_seconds": total_seconds,
    }


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
            metrics: dict[str, float | int | list[int]] = {
                "prefix_tokens": 0,
                "suffix_tokens": [0, 0],
                "padded_suffix_tokens": 0,
                "logical_prompt_tokens": 0,
                "executed_tokens": 0,
                "prefix_seconds": 0.0,
                "suffix_seconds": 0.0,
                "total_seconds": 0.0,
            }
            sequential_static_score = 0.0
            sequential_dynamic_score = 0.0
            sequential_seconds = 0.0
        else:
            static_context = format_pages(
                initial_ids + static_extra, block_texts, label="Working-set"
            )
            dynamic_context = format_pages(
                initial_ids + dynamic_extra, block_texts, label="Working-set"
            )
            prompts = [
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
            ]

            def run_shared() -> tuple[list[float], dict[str, float | int | list[int]]]:
                return shared_prefix_log_odds(
                    model,
                    tokenizer,
                    prompts,
                    yes_id=int(yes[0]),
                    no_id=int(no[0]),
                    device=device,
                )

            def run_sequential() -> tuple[list[float], float]:
                sequential_scores = []
                elapsed = 0.0
                for prompt in prompts:
                    score, seconds, _ = binary_log_odds(
                        model,
                        tokenizer,
                        prompt,
                        positive_label_id=int(yes[0]),
                        negative_label_id=int(no[0]),
                        device=device,
                    )
                    sequential_scores.append(score)
                    elapsed += seconds
                return sequential_scores, elapsed

            sequential_scores = [0.0, 0.0]
            sequential_seconds = 0.0
            if args.also_sequential and index % 2 == 1:
                sequential_scores, sequential_seconds = run_sequential()
                scores, metrics = run_shared()
            else:
                scores, metrics = run_shared()
                if args.also_sequential:
                    sequential_scores, sequential_seconds = run_sequential()
            static_score, dynamic_score = scores
            sequential_static_score, sequential_dynamic_score = sequential_scores
        rows.append(
            {
                "query_id": query_id,
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "sets_identical": identical,
                "shared_static_completeness_log_odds": static_score,
                "shared_dynamic_completeness_log_odds": dynamic_score,
                "shared_completeness_utility_score": dynamic_score - static_score,
                "paired_sequential_static_log_odds": sequential_static_score,
                "paired_sequential_dynamic_log_odds": sequential_dynamic_score,
                "paired_sequential_utility_score": (
                    sequential_dynamic_score - sequential_static_score
                ),
                "paired_sequential_seconds": sequential_seconds,
                "paired_order": (
                    "sequential_first"
                    if args.also_sequential and index % 2 == 1
                    else "shared_first"
                ),
                **metrics,
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
                    "seconds": metrics["total_seconds"],
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    changed = [row for row in rows if not row["sets_identical"]]
    summary = {
        "source": "Qwen3-8B shared-prefix independent-completeness probe on LongMemEval 10M",
        "protocol": {
            "selection_uses_answer": False,
            "probe_generates_answer": False,
            "common_prefix_model_calls": 1,
            "batched_branch_model_calls": 1,
            "fixed_pages": 8,
            "also_sequential": args.also_sequential,
        },
        "model_name_or_path": args.model_name_or_path,
        "queries": len(rows),
        "changed_candidate_sets": len(changed),
        "mean_total_seconds_changed": mean(row["total_seconds"] for row in changed),
        "mean_prefix_tokens_changed": mean(row["prefix_tokens"] for row in changed),
        "mean_suffix_tokens_changed": mean(
            sum(row["suffix_tokens"]) for row in changed
        ),
        "mean_logical_prompt_tokens_changed": mean(
            row["logical_prompt_tokens"] for row in changed
        ),
        "mean_executed_tokens_changed": mean(row["executed_tokens"] for row in changed),
        "mean_paired_sequential_seconds_changed": (
            mean(row["paired_sequential_seconds"] for row in changed)
            if args.also_sequential
            else None
        ),
        "paired_wall_clock_speedup": (
            mean(row["paired_sequential_seconds"] for row in changed)
            / mean(row["total_seconds"] for row in changed)
            if args.also_sequential
            else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
