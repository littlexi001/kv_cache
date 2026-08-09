from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from transformers import AutoTokenizer

from prepare_musique_official_10m import (
    find_answer_span,
    first_subject,
    paragraph_key,
    read_jsonl,
    render_atomic_question,
    valid_two_hop,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paragraph-aligned 10M MuSiQue corpus with sufficient gold blocks."
    )
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seq_tokens", type=int, default=10_000_000)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--train_queries", type=int, default=200)
    parser.add_argument("--dev_queries", type=int, default=100)
    parser.add_argument("--test_queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def paragraph_text(paragraph: dict[str, Any]) -> str:
    return f"\nPassage: {paragraph['title']}\n{paragraph['paragraph_text']}\n"


class FillerTokenStream:
    def __init__(
        self,
        keys: Sequence[str],
        encode: Callable[[str], list[int]],
    ) -> None:
        self.keys = list(keys)
        self.encode = encode
        self.key_index = 0
        self.token_offset = 0
        self.used_keys: set[str] = set()

    def take(self, count: int) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            if self.key_index >= len(self.keys):
                raise RuntimeError("real MuSiQue distractor stream was exhausted")
            key = self.keys[self.key_index]
            values = self.encode(key)
            self.used_keys.add(key)
            available = len(values) - self.token_offset
            take = min(count - len(output), available)
            output.extend(values[self.token_offset : self.token_offset + take])
            self.token_offset += take
            if self.token_offset == len(values):
                self.key_index += 1
                self.token_offset = 0
        return output


def main() -> None:
    args = parse_args()
    target_tokens = (args.seq_tokens // args.block_tokens) * args.block_tokens
    if target_tokens <= 0:
        raise ValueError("seq_tokens must contain at least one complete block")
    num_blocks = target_tokens // args.block_tokens
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_rows = read_jsonl(Path(args.train_path))
    dev_rows = read_jsonl(Path(args.dev_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = max(
        int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000
    )

    paragraph_by_key: dict[str, dict[str, Any]] = {}
    for row in [*train_rows, *dev_rows]:
        for paragraph in row["paragraphs"]:
            paragraph_by_key.setdefault(paragraph_key(paragraph), paragraph)
    encoded: dict[str, list[int]] = {}

    def encode(key: str) -> list[int]:
        if key not in encoded:
            encoded[key] = tokenizer(
                paragraph_text(paragraph_by_key[key]), add_special_tokens=False
            )["input_ids"]
        return encoded[key]

    def aligned_row(row: dict[str, Any]) -> bool:
        if not valid_two_hop(row):
            return False
        support_keys = []
        for step in row["question_decomposition"]:
            paragraph = row["paragraphs"][int(step["paragraph_support_idx"])]
            key = paragraph_key(paragraph)
            values = encode(key)
            if len(values) > args.block_tokens:
                return False
            try:
                find_answer_span(values, str(step["answer"]), tokenizer)
            except ValueError:
                return False
            support_keys.append(key)
        return len(set(support_keys)) == 2

    def select(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        candidates = [row for row in rows if aligned_row(row)]
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"only {len(candidates)} aligned rows for requested {count}")
        return candidates[:count]

    selected = [("train", row) for row in select(train_rows, args.train_queries)]
    heldout = select(dev_rows, args.dev_queries + args.test_queries)
    selected.extend(("dev", row) for row in heldout[: args.dev_queries])
    selected.extend(("test", row) for row in heldout[args.dev_queries :])
    support_keys = {
        paragraph_key(row["paragraphs"][int(step["paragraph_support_idx"])])
        for _split, row in selected
        for step in row["question_decomposition"]
    }
    if len(support_keys) >= num_blocks:
        raise RuntimeError("support paragraphs exceed the block budget")
    filler_keys = [key for key in paragraph_by_key if key not in support_keys]
    rng.shuffle(filler_keys)
    filler = FillerTokenStream(filler_keys, encode)

    raw_blocks = np.empty((num_blocks, args.block_tokens), dtype=np.int32)
    old_support_block: dict[str, int] = {}
    shuffled_support = list(support_keys)
    rng.shuffle(shuffled_support)
    for old_block, key in enumerate(shuffled_support):
        support_tokens = encode(key)
        fill = filler.take(args.block_tokens - len(support_tokens))
        raw_blocks[old_block] = np.asarray([*support_tokens, *fill], dtype=np.int32)
        old_support_block[key] = old_block
    for old_block in range(len(shuffled_support), num_blocks):
        raw_blocks[old_block] = np.asarray(
            filler.take(args.block_tokens), dtype=np.int32
        )

    permutation = list(range(num_blocks))
    rng.shuffle(permutation)
    blocks = raw_blocks[np.asarray(permutation, dtype=np.int64)]
    inverse = np.empty(num_blocks, dtype=np.int64)
    for new_block, old_block in enumerate(permutation):
        inverse[old_block] = new_block
    support_block = {
        key: int(inverse[old_block]) for key, old_block in old_support_block.items()
    }
    np.save(output_dir / "blocks.npy", blocks)
    support_by_new_block = {block_id: key for key, block_id in support_block.items()}
    write_jsonl(
        output_dir / "blocks.jsonl",
        (
            {
                "block_id": block_id,
                "token_start": block_id * args.block_tokens,
                "token_end": (block_id + 1) * args.block_tokens,
                "support_paragraph_key": support_by_new_block.get(block_id),
            }
            for block_id in range(num_blocks)
        ),
    )

    step_rows = []
    query_rows = []
    for query_id, (split, row) in enumerate(selected):
        answers_so_far: list[str] = []
        query_steps = []
        for step_index, decomposition_step in enumerate(row["question_decomposition"]):
            answer = str(decomposition_step["answer"])
            paragraph_index = int(decomposition_step["paragraph_support_idx"])
            paragraph = row["paragraphs"][paragraph_index]
            key = paragraph_key(paragraph)
            target_block = support_block[key]
            _start, _end, aligned_answer = find_answer_span(
                blocks[target_block].tolist(), answer, tokenizer
            )
            raw_question = str(decomposition_step["question"])
            atomic_question = render_atomic_question(raw_question, answers_so_far)
            lookup_key = first_subject(raw_question) if step_index == 0 else answers_so_far[-1]
            common = {
                "query_id": query_id,
                "dataset": "musique_official_2hop_aligned",
                "task_type": "multihop",
                "split": split,
                "question": str(row["question"]),
                "record_id": -1,
                "block_start": 0,
                "block_count": num_blocks,
                "hard_negative_block_ids": [],
                "annotation_uses_answer": True,
                "selection_uses_answer": False,
                "official_decomposition": True,
                "gold_block_is_full_support_paragraph": True,
            }
            step = {
                **common,
                "step_index": step_index,
                "step_type": (
                    "resolve_bridge" if step_index == 0 else "resolve_answer_from_bridge"
                ),
                "step_operator": "official_atomic_decomposition",
                "lookup_key": lookup_key,
                "step_question": atomic_question,
                "retrieval_state": " ".join([*answers_so_far, atomic_question]),
                "compact_state_before": [
                    f"BRIDGE_ENTITY: {value}" for value in answers_so_far
                ],
                "full_state_before": [],
                "target_fact": aligned_answer,
                "target_output": answer,
                "target_block_ids": [target_block],
                "previous_evidence_block_ids": [
                    int(item["target_block_ids"][0]) for item in query_steps
                ],
                "minimal_sufficient_block_ids": [target_block],
                "official_raw_step_question": raw_question,
                "official_support_paragraph_idx": paragraph_index,
                "official_support_title": str(paragraph["title"]),
                "support_paragraph_tokens": len(encode(key)),
            }
            if step_index == 1:
                raw_template = raw_question.replace("#1", "{bridge}")
                if ">>" in raw_template:
                    subject, relation = [
                        item.strip() for item in raw_template.split(">>", maxsplit=1)
                    ]
                    step["step_question_template"] = f"What is the {relation} of {subject}?"
            query_steps.append(step)
            answers_so_far.append(answer)
        step_rows.extend(query_steps)
        query_rows.append(
            {
                "query_id": query_id,
                "source_id": str(row["id"]),
                "split": split,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "step_answers": answers_so_far,
                "step_target_blocks": [
                    int(item["target_block_ids"][0]) for item in query_steps
                ],
            }
        )
    write_jsonl(output_dir / "step_queries.jsonl", step_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    support_lengths = [len(encode(key)) for key in support_keys]
    summary = {
        "source": "official MuSiQue with paragraph-aligned sufficient support blocks",
        "contains_synthetic_vectors": False,
        "contains_synthetic_text": False,
        "contains_padding_tokens": False,
        "block_construction": (
            "each selected support paragraph is wholly contained in one block; remaining "
            "tokens and all other blocks come from real MuSiQue distractor paragraphs"
        ),
        "requested_tokens": args.seq_tokens,
        "num_tokens": target_tokens,
        "num_blocks": num_blocks,
        "block_tokens": args.block_tokens,
        "support_blocks": len(support_keys),
        "real_filler_paragraphs_used": len(filler.used_keys),
        "queries": len(query_rows),
        "steps": len(step_rows),
        "split_queries": {
            split: sum(row["split"] == split for row in query_rows)
            for split in ("train", "dev", "test")
        },
        "mean_support_paragraph_tokens": float(np.mean(support_lengths)),
        "p95_support_paragraph_tokens": float(np.quantile(support_lengths, 0.95)),
        "max_support_paragraph_tokens": max(support_lengths),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
