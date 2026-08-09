from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dynamic_kv_multisample import token_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen reader quality for fixed LongMemEval retrieval selections."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selection_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default="static_top12,evidence_state_dynamic_top12")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--answer_max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--question_types",
        default="",
        help="Optional comma-separated LongMemEval question types.",
    )
    parser.add_argument("--include_page_dates", action="store_true")
    parser.add_argument(
        "--page_order",
        choices=["retrieval", "chronological", "latest_first"],
        default="retrieval",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def answer_contains(generation: str, reference: str) -> bool:
    answer = normalize(reference)
    return len(answer) >= 2 and answer in normalize(generation)


def is_refusal(text: str) -> bool:
    value = normalize(text)
    patterns = (
        "not enough information",
        "insufficient information",
        "cannot determine",
        "can t determine",
        "cannot be determined",
        "not provided",
        "do not have enough",
        "unable to determine",
    )
    return any(pattern in value for pattern in patterns)


def format_date_minutes(value: int) -> str:
    if value < 0:
        return "unknown"
    ordinal, minute_of_day = divmod(value, 24 * 60)
    return (
        datetime.fromordinal(ordinal) + timedelta(minutes=minute_of_day)
    ).strftime("%Y-%m-%d %H:%M")


def format_context(
    block_ids: list[int],
    block_texts: list[str],
    block_dates: np.ndarray | None,
) -> str:
    return "\n\n".join(
        (
            f"[Memory page {index + 1}; session time: "
            f"{format_date_minutes(int(block_dates[block_id]))}]\n{block_texts[block_id]}"
            if block_dates is not None
            else f"[Memory page {index + 1}]\n{block_texts[block_id]}"
        )
        for index, block_id in enumerate(block_ids)
    )


def reader_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the memory pages below. Resolve dates, updates, "
        "and comparisons from the provided records. If the pages do not contain enough "
        "information, say that there is not enough information. Give only a concise "
        "final answer, without explaining retrieval.\n\n"
        f"Memory pages:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@torch.inference_mode()
def reference_nll(
    model: Any,
    tokenizer: Any,
    prompt: str,
    reference: str,
    device: torch.device,
) -> tuple[float, int, float]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    answer_ids = tokenizer(
        reference, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)
    answer_ids = torch.cat([answer_ids, eos], dim=1)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[1] :] = answer_ids
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(output.loss.item()), int(answer_ids.shape[1]), time.perf_counter() - started


@torch.inference_mode()
def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int, float]:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
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
    generated_ids = output[0, input_ids.shape[1] :]
    return (
        tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        int(generated_ids.numel()),
        elapsed,
    )


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("at least one method is required")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(data_dir / "queries.jsonl")
    allowed_types = {
        item.strip() for item in args.question_types.split(",") if item.strip()
    }
    if allowed_types:
        queries = [
            row for row in queries if str(row["question_type"]) in allowed_types
        ]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    query_ids = {int(row["query_id"]) for row in queries}
    selections = [
        row
        for row in read_jsonl(Path(args.selection_rows))
        if int(row["query_id"]) in query_ids and str(row["method"]) in methods
    ]
    selection_by_key = {
        (int(row["query_id"]), str(row["method"])): row for row in selections
    }
    expected = {(int(row["query_id"]), method) for row in queries for method in methods}
    if set(selection_by_key) != expected:
        raise ValueError("selection rows do not cover every query and method")

    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    ordering_dates = (
        np.load(data_dir / "base_block_date_minutes.npy", mmap_mode="r")
        if args.include_page_dates or args.page_order != "retrieval"
        else None
    )
    context_dates = ordering_dates if args.include_page_dates else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    unique_blocks = sorted(
        {
            int(block_id)
            for row in selections
            for block_id in row["top_block_ids"]
        }
    )
    block_texts = [""] * len(base_blocks)
    for block_id in unique_blocks:
        block_texts[block_id] = tokenizer.decode(
            np.asarray(base_blocks[block_id], dtype=np.int64),
            skip_special_tokens=True,
        )

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    for query_index, query in enumerate(queries):
        question = str(query["question"])
        reference = str(query["answer"])
        for method in methods:
            selection = selection_by_key[(int(query["query_id"]), method)]
            block_ids = list(map(int, selection["top_block_ids"]))
            if args.page_order == "chronological":
                if ordering_dates is None:
                    raise ValueError("chronological order requires block dates")
                block_ids.sort(key=lambda item: (int(ordering_dates[item]), item))
            elif args.page_order == "latest_first":
                if ordering_dates is None:
                    raise ValueError("latest-first order requires block dates")
                block_ids.sort(key=lambda item: (-int(ordering_dates[item]), item))
            prompt = reader_prompt(
                question, format_context(block_ids, block_texts, context_dates)
            )
            nll, reference_tokens, nll_seconds = reference_nll(
                model, tokenizer, prompt, reference, device
            )
            generation, generated_tokens, generation_seconds = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.answer_max_new_tokens,
                device=device,
            )
            rows.append(
                {
                    "query_id": int(query["query_id"]),
                    "question_id": str(query["question_id"]),
                    "question_type": str(query["question_type"]),
                    "is_abstention": bool(query["is_abstention"]),
                    "method": method,
                    "selected_blocks": len(block_ids),
                    "working_set_tokens": len(block_ids)
                    * int(base_blocks.shape[1]),
                    "reader_block_ids": block_ids,
                    "include_page_dates": args.include_page_dates,
                    "page_order": args.page_order,
                    "reference_nll": nll,
                    "reference_tokens": reference_tokens,
                    "nll_seconds": nll_seconds,
                    "generation": generation,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": generation_seconds,
                    "token_f1": float(token_f1(generation, reference)),
                    "normalized_exact_match": normalize(generation)
                    == normalize(reference),
                    "answer_contains": answer_contains(generation, reference),
                    "predicted_refusal": is_refusal(generation),
                    "selection_uses_answer": False,
                }
            )
        print(
            json.dumps(
                {
                    "completed": query_index + 1,
                    "queries": len(queries),
                    "query_id": int(query["query_id"]),
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    quality = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        positive = [row for row in group if not row["is_abstention"]]
        abstention = [row for row in group if row["is_abstention"]]
        quality.append(
            {
                "method": method,
                "queries": len(group),
                "positive_queries": len(positive),
                "abstention_queries": len(abstention),
                "mean_working_set_tokens": mean(
                    float(row["working_set_tokens"]) for row in group
                ),
                "mean_reference_nll": mean(
                    float(row["reference_nll"]) for row in group
                ),
                "mean_positive_token_f1": mean(
                    float(row["token_f1"]) for row in positive
                ),
                "positive_exact_match": mean(
                    float(row["normalized_exact_match"]) for row in positive
                ),
                "positive_answer_contains": mean(
                    float(row["answer_contains"]) for row in positive
                ),
                "abstention_refusal_accuracy": mean(
                    float(row["predicted_refusal"]) for row in abstention
                ),
                "mean_nll_seconds": mean(float(row["nll_seconds"]) for row in group),
                "mean_generation_seconds": mean(
                    float(row["generation_seconds"]) for row in group
                ),
            }
        )
    summary = {
        "source": "Qwen3-8B reader on static versus evidence-conditioned LongMemEval 10M selections",
        "protocol": {
            "selection_uses_answer": False,
            "reference_used_only_after_retrieval_for_scoring": True,
            "teacher_forced_reference_nll_is_primary": True,
            "token_f1_is_exploratory_not_official_longmemeval_judge": True,
            "methods": methods,
            "question_types": sorted(allowed_types),
            "include_page_dates": args.include_page_dates,
            "page_order": args.page_order,
        },
        "model_name_or_path": args.model_name_or_path,
        "queries": len(queries),
        "quality": quality,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
