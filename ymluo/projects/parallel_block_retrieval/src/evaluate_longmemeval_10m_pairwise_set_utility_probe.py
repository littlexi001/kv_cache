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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare static-depth and state-innovated candidate page sets conditional on "
            "the current LongMemEval working set, without using reference answers."
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


def ordered_difference(values: Iterable[int], excluded: set[int]) -> list[int]:
    output = []
    seen = set()
    for value in values:
        item = int(value)
        if item not in excluded and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def format_pages(
    block_ids: list[int], block_texts: dict[int, str], *, label: str
) -> str:
    if not block_ids:
        return f"[{label}: no additional pages]"
    return "\n\n".join(
        f"[{label} page {index + 1}]\n{block_texts[block_id]}"
        for index, block_id in enumerate(block_ids)
    )


def probe_prompt(
    question: str,
    state_text: str,
    fixed_context: str,
    candidate_a: str,
    candidate_b: str,
) -> str:
    return (
        "You manage a bounded external-memory working set. Compare two candidate page "
        "sets conditional on the pages already loaded. Decide which candidate set adds "
        "more of the still-missing evidence needed to answer the question. Check entity "
        "relations, dates, updates, comparisons, and whether all required facts are jointly "
        "supported. Penalize redundant, irrelevant, stale, or conflicting pages. Do not "
        "answer the question and do not use outside knowledge. Reply with exactly A or B.\n\n"
        f"Question:\n{question}\n\n"
        f"Current retrieval state:\n{state_text}\n\n"
        f"Pages already fixed in the working set:\n{fixed_context}\n\n"
        f"Candidate set A:\n{candidate_a}\n\n"
        f"Candidate set B:\n{candidate_b}\n\n"
        "Which candidate set has higher conditional evidence utility?\nChoice:"
    )


def completeness_prompt(
    question: str,
    state_text: str,
    context: str,
) -> str:
    return (
        "Judge whether the bounded memory working set contains all evidence required to "
        "answer the question using only these pages. Every required entity relation, date, "
        "update, comparison, and intermediate fact must be explicitly supported. If any "
        "required slot is unresolved, or the pages are merely related but insufficient, "
        "choose NO. Do not answer the question and do not use outside knowledge. Reply "
        "with exactly YES or NO.\n\n"
        f"Question:\n{question}\n\n"
        f"Current retrieval state:\n{state_text}\n\n"
        f"Complete candidate working set:\n{context}\n\n"
        "Are all required evidence slots explicitly supported?\nDecision:"
    )


@torch.inference_mode()
def binary_log_odds(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    positive_label_id: int,
    negative_label_id: int,
    device: torch.device,
) -> tuple[float, float, int]:
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
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits[0, -1]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    log_odds = float(
        (logits[positive_label_id] - logits[negative_label_id]).float().item()
    )
    return log_odds, elapsed, int(input_ids.shape[1])


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

    all_block_ids = set()
    for query_id in query_ids:
        all_block_ids.update(int(value) for value in states[query_id]["initial_block_ids"])
        for method in selections:
            all_block_ids.update(
                int(value) for value in selections[method][query_id]["top_block_ids"]
            )
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    block_texts = {
        block_id: tokenizer.decode(
            np.asarray(base_blocks[block_id], dtype=np.int64), skip_special_tokens=True
        )
        for block_id in sorted(all_block_ids)
    }
    label_a = tokenizer("A", add_special_tokens=False)["input_ids"]
    label_b = tokenizer("B", add_special_tokens=False)["input_ids"]
    if len(label_a) != 1 or len(label_b) != 1:
        raise ValueError(f"expected single-token labels, got A={label_a}, B={label_b}")
    label_yes = tokenizer("YES", add_special_tokens=False)["input_ids"]
    label_no = tokenizer("NO", add_special_tokens=False)["input_ids"]
    if len(label_yes) != 1 or len(label_no) != 1:
        raise ValueError(
            f"expected single-token labels, got YES={label_yes}, NO={label_no}"
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
        if static_extra == dynamic_extra:
            forward_score = 0.0
            reverse_score = 0.0
            utility_score = 0.0
            forward_seconds = 0.0
            reverse_seconds = 0.0
            forward_tokens = 0
            reverse_tokens = 0
            static_complete_score = 0.0
            dynamic_complete_score = 0.0
            completeness_utility_score = 0.0
            static_complete_seconds = 0.0
            dynamic_complete_seconds = 0.0
            static_complete_tokens = 0
            dynamic_complete_tokens = 0
        else:
            fixed = format_pages(initial_ids, block_texts, label="Fixed")
            dynamic = format_pages(dynamic_extra, block_texts, label="Dynamic candidate")
            static = format_pages(static_extra, block_texts, label="Static candidate")
            forward_prompt = probe_prompt(
                str(query["question"]),
                str(state["state_text"]),
                fixed,
                dynamic,
                static,
            )
            forward_score, forward_seconds, forward_tokens = binary_log_odds(
                model,
                tokenizer,
                forward_prompt,
                positive_label_id=int(label_a[0]),
                negative_label_id=int(label_b[0]),
                device=device,
            )
            reverse_prompt = probe_prompt(
                str(query["question"]),
                str(state["state_text"]),
                fixed,
                static,
                dynamic,
            )
            reverse_raw, reverse_seconds, reverse_tokens = binary_log_odds(
                model,
                tokenizer,
                reverse_prompt,
                positive_label_id=int(label_a[0]),
                negative_label_id=int(label_b[0]),
                device=device,
            )
            reverse_score = -reverse_raw
            utility_score = 0.5 * (forward_score + reverse_score)
            static_full = format_pages(
                initial_ids + static_extra, block_texts, label="Working-set"
            )
            dynamic_full = format_pages(
                initial_ids + dynamic_extra, block_texts, label="Working-set"
            )
            static_complete_score, static_complete_seconds, static_complete_tokens = (
                binary_log_odds(
                    model,
                    tokenizer,
                    completeness_prompt(
                        str(query["question"]), str(state["state_text"]), static_full
                    ),
                    positive_label_id=int(label_yes[0]),
                    negative_label_id=int(label_no[0]),
                    device=device,
                )
            )
            dynamic_complete_score, dynamic_complete_seconds, dynamic_complete_tokens = (
                binary_log_odds(
                    model,
                    tokenizer,
                    completeness_prompt(
                        str(query["question"]), str(state["state_text"]), dynamic_full
                    ),
                    positive_label_id=int(label_yes[0]),
                    negative_label_id=int(label_no[0]),
                    device=device,
                )
            )
            completeness_utility_score = dynamic_complete_score - static_complete_score
        rows.append(
            {
                "query_id": query_id,
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "initial_block_ids": initial_ids,
                "static_extra_block_ids": static_extra,
                "dynamic_extra_block_ids": dynamic_extra,
                "sets_identical": static_extra == dynamic_extra,
                "forward_dynamic_log_odds": forward_score,
                "reverse_dynamic_log_odds": reverse_score,
                "pairwise_dynamic_utility_score": utility_score,
                "static_completeness_log_odds": static_complete_score,
                "dynamic_completeness_log_odds": dynamic_complete_score,
                "completeness_dynamic_utility_score": completeness_utility_score,
                "order_sign_agreement": bool(
                    forward_score == 0
                    or reverse_score == 0
                    or (forward_score > 0) == (reverse_score > 0)
                ),
                "forward_seconds": forward_seconds,
                "reverse_seconds": reverse_seconds,
                "forward_prompt_tokens": forward_tokens,
                "reverse_prompt_tokens": reverse_tokens,
                "static_completeness_seconds": static_complete_seconds,
                "dynamic_completeness_seconds": dynamic_complete_seconds,
                "static_completeness_prompt_tokens": static_complete_tokens,
                "dynamic_completeness_prompt_tokens": dynamic_complete_tokens,
                "selection_uses_answer": False,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "queries": len(queries),
                    "query_id": query_id,
                    "score": utility_score,
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    changed = [row for row in rows if not row["sets_identical"]]
    summary = {
        "source": "Qwen3-8B pairwise conditional set-utility probe on LongMemEval 10M",
        "protocol": {
            "selection_uses_answer": False,
            "probe_generates_answer": False,
            "fixed_pages": 8,
            "compared_actions": "static rank 9-12 versus state-innovated extra pages",
            "order_counterbalanced": True,
        },
        "model_name_or_path": args.model_name_or_path,
        "queries": len(rows),
        "changed_candidate_sets": len(changed),
        "mean_pairwise_seconds_changed": mean(
            float(row["forward_seconds"]) + float(row["reverse_seconds"])
            for row in changed
        ),
        "mean_completeness_seconds_changed": mean(
            float(row["static_completeness_seconds"])
            + float(row["dynamic_completeness_seconds"])
            for row in changed
        ),
        "mean_prompt_tokens_changed": mean(
            0.5
            * (float(row["forward_prompt_tokens"]) + float(row["reverse_prompt_tokens"]))
            for row in changed
        ),
        "order_sign_agreement_changed": mean(
            float(row["order_sign_agreement"]) for row in changed
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
