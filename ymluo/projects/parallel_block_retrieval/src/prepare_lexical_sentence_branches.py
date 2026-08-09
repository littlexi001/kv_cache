from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from evaluate_stepwise_set_utility import build_span_query, lexical_tokens
from run_step_state_kv_span_retrieval import (
    find_text_subsequence,
    sentence_token_spans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select query-relevant sentence segments inside retrieved blocks."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--step_types", default="")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--sentences_per_block", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_sentence_spans(
    token_ids: Sequence[int],
    spans: Sequence[Sequence[int]],
    query_text: str,
    tokenizer: Any,
) -> list[tuple[int, int]]:
    texts = [
        tokenizer.decode(token_ids[int(start) : int(end)], skip_special_tokens=True)
        for start, end in spans
    ]
    terms = [lexical_tokens(text) for text in texts]
    query_terms = lexical_tokens(query_text)
    document_frequency = Counter(
        term for sentence_terms in terms for term in sentence_terms if term in query_terms
    )
    count = len(spans)
    scored = []
    for index, (text, sentence_terms) in enumerate(zip(texts, terms, strict=True)):
        overlap = sentence_terms & query_terms
        score = sum(
            math.log((count + 1.0) / (document_frequency[term] + 0.5))
            for term in overlap
        )
        score /= max(1.0, len(sentence_terms) ** 0.2)
        if re.match(r"^\s*Passage\s*:", text, flags=re.IGNORECASE):
            score -= 5.0
        scored.append((score, -int(spans[index][0]), index))
    return [
        (int(spans[index][0]), int(spans[index][1]))
        for _score, _start, index in sorted(scored, reverse=True)
    ]


def overlap_fraction(span: Sequence[int], target: Sequence[int]) -> float:
    start, end = int(span[0]), int(span[1])
    target_start, target_end = int(target[0]), int(target[1])
    return max(0, min(end, target_end) - max(start, target_start)) / max(
        1, target_end - target_start
    )


def main() -> None:
    args = parse_args()
    if args.sentences_per_block <= 0:
        raise ValueError("sentences_per_block must be positive")
    allowed_types = {
        item.strip() for item in args.step_types.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split
        and (not allowed_types or str(row["step_type"]) in allowed_types)
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    step_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row for row in steps
    }
    retrieval_rows = [
        row
        for row in read_jsonl(Path(args.retrieval_rows_path))
        if (int(row["query_id"]), int(row["step_index"])) in step_by_key
    ]
    if len(retrieval_rows) != len(steps):
        raise ValueError("retrieval rows do not exactly cover requested steps")

    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    output_rows = []
    selected_tokens = []
    for retrieval in retrieval_rows:
        key = (int(retrieval["query_id"]), int(retrieval["step_index"]))
        step = step_by_key[key]
        query_text = build_span_query(
            step, [str(item) for item in step.get("compact_state_before", [])]
        )
        candidates = []
        for candidate in retrieval["branch_candidates"]:
            block_id = int(candidate["block_id"])
            token_ids = blocks[block_id].tolist()
            try:
                spans = sentence_token_spans(token_ids, tokenizer)
            except ValueError:
                spans = [(0, len(token_ids))]
            ranked = rank_sentence_spans(token_ids, spans, query_text, tokenizer)
            selected = sorted(ranked[: args.sentences_per_block])
            target_span = find_text_subsequence(
                token_ids, str(step["target_output"]), tokenizer
            )
            coverage = max(
                (overlap_fraction(span, target_span) for span in selected),
                default=0.0,
            )
            context_tokens = sum(end - start for start, end in selected)
            selected_tokens.append(context_tokens)
            candidates.append(
                {
                    **candidate,
                    "start": selected[0][0],
                    "end": selected[-1][1],
                    "segments": [[start, end] for start, end in selected],
                    "target_overlap": coverage,
                    "sentence_selector": "query_idf_overlap",
                }
            )
        output_rows.append(
            {
                **retrieval,
                "sentence_selector_uses_gold": False,
                "branch_candidates": candidates,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload: dict[str, Any] = {
        "source": "query-only lexical sentence selection within retrieved blocks",
        "selection_uses_gold": any(
            bool(row.get("selection_uses_gold", False)) for row in output_rows
        ),
        "sentence_selector_uses_gold": False,
        "steps": len(output_rows),
        "sentences_per_block": args.sentences_per_block,
        "target_span_recall": statistics.fmean(
            any(float(item["target_overlap"]) >= 0.8 for item in row["branch_candidates"])
            for row in output_rows
        ),
        "mean_selected_tokens_per_branch": statistics.fmean(selected_tokens),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
