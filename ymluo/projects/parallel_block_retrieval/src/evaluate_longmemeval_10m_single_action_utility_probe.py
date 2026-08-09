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
            "Score a static-to-dynamic LongMemEval working-set edit with one YES/NO "
            "forward, with a no-fixed-context ablation and order controls."
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


def format_pages(block_ids: list[int], block_texts: dict[int, str], label: str) -> str:
    if not block_ids:
        return f"[{label}: no pages]"
    return "\n\n".join(
        f"[{label} {index + 1}]\n{block_texts[block_id]}"
        for index, block_id in enumerate(block_ids)
    )


def replacement_prompt(
    question: str,
    state_text: str,
    fixed_context: str | None,
    old_pages: str,
    new_pages: str,
    *,
    new_first: bool,
) -> str:
    page_sections = (
        f"NEW replacement pages:\n{new_pages}\n\n"
        f"OLD pages that would be removed:\n{old_pages}"
        if new_first
        else f"OLD pages that would be removed:\n{old_pages}\n\n"
        f"NEW replacement pages:\n{new_pages}"
    )
    fixed_section = (
        f"FIXED pages that remain loaded:\n{fixed_context}\n\n"
        if fixed_context is not None
        else "The fixed pages are intentionally hidden for this ablation.\n\n"
    )
    return (
        "You control one edit to a bounded external-memory working set. Decide whether "
        "replacing the OLD pages with the NEW pages increases the evidence completeness "
        "for the question. Check all required entity relations, dates, updates, "
        "comparisons, and intermediate facts. Count redundancy, stale facts, conflicts, "
        "and evidence lost by removing OLD pages. Use only the supplied pages and state. "
        "Do not answer the question. Reply YES only when the replacement has positive "
        "conditional utility; otherwise reply NO. Reply with exactly YES or NO.\n\n"
        f"Question:\n{question}\n\n"
        f"Current retrieval state:\n{state_text}\n\n"
        f"{fixed_section}{page_sections}\n\n"
        "Should NEW replace OLD?\nDecision:"
    )


@torch.inference_mode()
def yes_no_log_odds(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    yes_id: int,
    no_id: int,
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
    seconds = time.perf_counter() - started
    score = float((logits[yes_id] - logits[no_id]).float().item())
    return score, seconds, int(input_ids.shape[1])


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
        old_ids = ordered_difference(static_ids, initial_set)
        new_ids = ordered_difference(dynamic_ids, initial_set)
        identical = old_ids == new_ids
        result: dict[str, Any] = {}
        if identical:
            for prefix in (
                "full_old_first",
                "full_new_first",
                "delta_old_first",
                "delta_new_first",
            ):
                result[f"{prefix}_log_odds"] = 0.0
                result[f"{prefix}_seconds"] = 0.0
                result[f"{prefix}_prompt_tokens"] = 0
        else:
            fixed = format_pages(initial_ids, block_texts, "Fixed page")
            old = format_pages(old_ids, block_texts, "Old page")
            new = format_pages(new_ids, block_texts, "New page")
            for prefix, fixed_context, new_first in (
                ("full_old_first", fixed, False),
                ("full_new_first", fixed, True),
                ("delta_old_first", None, False),
                ("delta_new_first", None, True),
            ):
                score, seconds, tokens = yes_no_log_odds(
                    model,
                    tokenizer,
                    replacement_prompt(
                        str(query["question"]),
                        str(state["state_text"]),
                        fixed_context,
                        old,
                        new,
                        new_first=new_first,
                    ),
                    yes_id=int(yes[0]),
                    no_id=int(no[0]),
                    device=device,
                )
                result[f"{prefix}_log_odds"] = score
                result[f"{prefix}_seconds"] = seconds
                result[f"{prefix}_prompt_tokens"] = tokens
        result["full_order_average_log_odds"] = 0.5 * (
            float(result["full_old_first_log_odds"])
            + float(result["full_new_first_log_odds"])
        )
        result["delta_order_average_log_odds"] = 0.5 * (
            float(result["delta_old_first_log_odds"])
            + float(result["delta_new_first_log_odds"])
        )
        result["full_order_sign_agreement"] = bool(
            result["full_old_first_log_odds"] == 0
            or result["full_new_first_log_odds"] == 0
            or (result["full_old_first_log_odds"] > 0)
            == (result["full_new_first_log_odds"] > 0)
        )
        result["delta_order_sign_agreement"] = bool(
            result["delta_old_first_log_odds"] == 0
            or result["delta_new_first_log_odds"] == 0
            or (result["delta_old_first_log_odds"] > 0)
            == (result["delta_new_first_log_odds"] > 0)
        )
        rows.append(
            {
                "query_id": query_id,
                "question_id": str(query["question_id"]),
                "question_type": str(query["question_type"]),
                "is_abstention": bool(query["is_abstention"]),
                "initial_block_ids": initial_ids,
                "old_extra_block_ids": old_ids,
                "new_extra_block_ids": new_ids,
                "sets_identical": identical,
                **result,
                "selection_uses_answer": False,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "queries": len(queries),
                    "query_id": query_id,
                    "full_score": result["full_old_first_log_odds"],
                }
            ),
            flush=True,
        )

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    changed = [row for row in rows if not row["sets_identical"]]
    summary = {
        "source": "Qwen3-8B single-action conditional set-utility probe on LongMemEval 10M",
        "protocol": {
            "selection_uses_answer": False,
            "probe_generates_answer": False,
            "fixed_pages": 8,
            "action": "replace static rank 9-12 with state-innovated pages",
            "deployment_score": "one full_old_first YES/NO forward",
            "order_controls_are_offline_only": True,
        },
        "model_name_or_path": args.model_name_or_path,
        "queries": len(rows),
        "changed_candidate_sets": len(changed),
        "mean_seconds_changed": {
            prefix: mean(float(row[f"{prefix}_seconds"]) for row in changed)
            for prefix in ("full_old_first", "delta_old_first")
        },
        "mean_prompt_tokens_changed": {
            prefix: mean(float(row[f"{prefix}_prompt_tokens"]) for row in changed)
            for prefix in ("full_old_first", "delta_old_first")
        },
        "order_sign_agreement_changed": {
            "full": mean(float(row["full_order_sign_agreement"]) for row in changed),
            "delta_only": mean(
                float(row["delta_order_sign_agreement"]) for row in changed
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
