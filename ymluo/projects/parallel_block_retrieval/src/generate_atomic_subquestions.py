from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_retrieved_answer_nll import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate relation-agnostic atomic lookup questions from reasoning state."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_prompt(step: dict[str, Any]) -> str:
    question = str(step["question"])
    if str(step["step_type"]) == "resolve_bridge":
        entity = str(step["lookup_key"])
        examples = (
            "Examples:\n"
            "Original: Where was the wife of Person A born? Known entity: Person A\n"
            "NEXT: Who is Person A's wife?\n"
            "Original: Who is the spouse of the director of Film B? Known entity: Film B\n"
            "NEXT: Who directed Film B?\n"
            "Original: Who was the paternal grandfather of Person C? Known entity: Person C\n"
            "NEXT: Who was Person C's father?\n"
        )
        state = f"Known starting entity: {entity}"
        requirement = (
            "Write the first atomic lookup question. Its answer must be one new "
            "intermediate entity, not the final answer."
        )
    else:
        compact_state = [str(item) for item in step.get("compact_state_before", [])]
        entity = compact_state[-1].split(":", 1)[-1].strip() if compact_state else "unknown"
        examples = (
            "Examples:\n"
            "Original: Where was the wife of Person A born? Intermediate entity: Person D\n"
            "NEXT: Where was Person D born?\n"
            "Original: Who is the spouse of the director of Film B? Intermediate entity: Person E\n"
            "NEXT: Who is Person E's spouse?\n"
            "Original: Who was the paternal grandfather of Person C? Intermediate entity: Person F\n"
            "NEXT: Who was Person F's father?\n"
        )
        state = f"Verified intermediate entity: {entity}"
        requirement = (
            "Write the next atomic lookup question about the verified entity. Its answer "
            "must directly answer the original question."
        )
    return (
        "Decompose a multi-hop question one lookup at a time. Do not answer it.\n"
        f"{examples}\n"
        f"Original question: {question}\n"
        f"{state}\n"
        f"{requirement}\n"
        "Output exactly one line beginning with `NEXT:`."
    )


def clean_subquestion(text: str) -> str:
    match = re.search(r"(?:^|\n)\s*NEXT\s*:\s*(.+)", text, flags=re.IGNORECASE)
    value = match.group(1) if match else text.splitlines()[0]
    value = " ".join(value.strip().split()).strip(" `\"'")
    if not value:
        raise ValueError("empty generated subquestion")
    return value


@torch.inference_mode()
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    steps = [
        dict(step)
        for step in read_jsonl(Path(args.step_queries_path))
        if str(step["split"]) in allowed_splits
    ]
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    traces = []
    for index, step in enumerate(steps):
        prompt = atomic_prompt(step)
        raw, tokens, seconds = generate(
            model, tokenizer, prompt, args.max_new_tokens, device
        )
        subquestion = clean_subquestion(raw)
        step["step_question"] = subquestion
        step["step_operator"] = "model_atomic_subquestion"
        step["subquestion_source"] = "question_and_verified_state_only"
        if str(step["step_type"]) == "resolve_answer_from_bridge":
            step.pop("step_question_template", None)
        traces.append(
            {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "prompt_uses_gold": False,
                "raw_generation": raw,
                "subquestion": subquestion,
                "generated_tokens": tokens,
                "generation_seconds": seconds,
            }
        )
        print(
            json.dumps(
                {
                    "step": index + 1,
                    "steps": len(steps),
                    "query_id": int(step["query_id"]),
                    "step_index": int(step["step_index"]),
                    "subquestion": subquestion,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for name, rows in (("step_queries.jsonl", steps), ("traces.jsonl", traces)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "model-generated atomic subquestions from question and verified state",
        "selection_uses_gold": False,
        "prompt_uses_gold": False,
        "steps": len(steps),
        "mean_generation_seconds": sum(row["generation_seconds"] for row in traces)
        / len(traces),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in traces)
        / len(traces),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
