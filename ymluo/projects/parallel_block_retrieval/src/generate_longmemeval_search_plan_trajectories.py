from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answer-free memory-search plans whose token prefixes define a "
            "retrieval trajectory for LongMemEval."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_for(question: str) -> str:
    return (
        "Write a terse search plan for retrieving personal memory records needed to "
        "answer the question below. Identify entities, relations, time constraints, "
        "comparisons, and missing facts. Break multi-record questions into atomic "
        "lookups. Do not answer the question, guess values, or claim that a fact is "
        "known. Output only search terms and subquestions.\n\n"
        f"Question: {question}\n"
        "Search plan:"
    )


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def answer_overlap(generation: str, answer: str, question: str) -> bool:
    normalized_answer = normalize(answer)
    normalized_generation = normalize(generation)
    normalized_question = normalize(question)
    if len(normalized_answer) < 3:
        return False
    # Repeating a choice already present in the question is not answer leakage.
    return (
        normalized_answer in normalized_generation
        and normalized_answer not in normalized_question
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], float]:
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
    generated = output[0, input_ids.shape[1] :].tolist()
    while generated and generated[-1] in {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }:
        generated.pop()
    return [int(item) for item in generated], elapsed


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(data_dir / "queries.jsonl")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = []
    for index, query in enumerate(queries):
        question = str(query["question"])
        prompt = prompt_for(question)
        token_ids, seconds = generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        generation = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        overlap = answer_overlap(generation, str(query["answer"]), question)
        row = {
            "query_id": int(query["query_id"]),
            "question_id": str(query["question_id"]),
            "question_type": str(query["question_type"]),
            "is_abstention": bool(query["is_abstention"]),
            "model_name_or_path": args.model_name_or_path,
            "prompt_uses_memory": False,
            "prompt_uses_answer": False,
            "answer_used_only_for_posthoc_overlap_audit": True,
            "answer_overlap_posthoc": overlap,
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "generation": generation,
            "generation_seconds": seconds,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "queries": len(queries),
                    "query_id": row["query_id"],
                    "tokens": row["generated_tokens"],
                    "answer_overlap": overlap,
                    "seconds": round(seconds, 4),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "LongMemEval question-only generated memory-search trajectories",
        "queries": len(rows),
        "model_name_or_path": args.model_name_or_path,
        "max_new_tokens": args.max_new_tokens,
        "prompt_uses_memory": False,
        "prompt_uses_answer": False,
        "answer_used_only_for_posthoc_overlap_audit": True,
        "answer_overlap_queries": sum(bool(row["answer_overlap_posthoc"]) for row in rows),
        "mean_generated_tokens": sum(int(row["generated_tokens"]) for row in rows)
        / len(rows),
        "mean_generation_seconds": sum(float(row["generation_seconds"]) for row in rows)
        / len(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
