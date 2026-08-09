from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer

from run_step_state_kv_span_retrieval import find_text_subsequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 10M real-text MuSiQue corpus with official two-hop step labels."
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    return f" {normalized(phrase)} " in f" {normalized(text)} "


def paragraph_key(paragraph: dict[str, Any]) -> str:
    payload = f"{paragraph['title']}\n{paragraph['paragraph_text']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def valid_two_hop(row: dict[str, Any]) -> bool:
    decomposition = row.get("question_decomposition", [])
    paragraphs = row.get("paragraphs", [])
    if not bool(row.get("answerable", True)) or len(decomposition) != 2:
        return False
    if "#" in str(decomposition[0].get("question", "")):
        return False
    if "#1" not in str(decomposition[1].get("question", "")):
        return False
    for step in decomposition:
        paragraph_index = int(step.get("paragraph_support_idx", -1))
        if paragraph_index < 0 or paragraph_index >= len(paragraphs):
            return False
        answer = str(step.get("answer", "")).strip()
        if not answer or not contains_phrase(
            str(paragraphs[paragraph_index]["paragraph_text"]), answer
        ):
            return False
    return True


def choose_rows(
    rows: Sequence[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if valid_two_hop(row)]
    rng.shuffle(eligible)
    if len(eligible) < count:
        raise RuntimeError(f"only {len(eligible)} eligible rows for requested {count}")
    return eligible[:count]


def render_atomic_question(raw: str, prior_answers: Sequence[str]) -> str:
    value = str(raw)
    for index, answer in enumerate(prior_answers, start=1):
        value = value.replace(f"#{index}", answer)
    if ">>" not in value:
        return value.strip()
    subject, relation = [item.strip() for item in value.split(">>", maxsplit=1)]
    return f"What is the {relation} of {subject}?"


def first_subject(raw: str) -> str:
    return str(raw).split(">>", maxsplit=1)[0].strip()


def find_answer_span(
    tokens: Sequence[int], answer: str, tokenizer: Any
) -> tuple[int, int, str]:
    variants = [answer, answer.strip(" \t\r\n.,;:!?()[]{}\"'")]
    for variant in dict.fromkeys(item for item in variants if item):
        try:
            start, end = find_text_subsequence(tokens, variant, tokenizer)
            return start, end, variant
        except ValueError:
            continue
    raise ValueError(f"official answer could not be aligned: {answer!r}")


def main() -> None:
    args = parse_args()
    target_tokens = (args.seq_tokens // args.block_tokens) * args.block_tokens
    if target_tokens <= 0:
        raise ValueError("seq_tokens must contain at least one complete block")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_rows = read_jsonl(Path(args.train_path))
    dev_rows = read_jsonl(Path(args.dev_path))
    selected = [
        ("train", row)
        for row in choose_rows(train_rows, args.train_queries, rng)
    ]
    heldout = choose_rows(dev_rows, args.dev_queries + args.test_queries, rng)
    selected.extend(("dev", row) for row in heldout[: args.dev_queries])
    selected.extend(("test", row) for row in heldout[args.dev_queries :])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = max(
        int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000
    )
    selected_keys = {
        paragraph_key(paragraph)
        for _split, row in selected
        for paragraph in row["paragraphs"]
    }
    paragraph_by_key: dict[str, dict[str, Any]] = {}
    for row in [*train_rows, *dev_rows]:
        for paragraph in row["paragraphs"]:
            paragraph_by_key.setdefault(paragraph_key(paragraph), paragraph)
    distractor_keys = [key for key in paragraph_by_key if key not in selected_keys]
    rng.shuffle(distractor_keys)

    encoded: dict[str, list[int]] = {}

    def encode(key: str) -> list[int]:
        if key not in encoded:
            paragraph = paragraph_by_key[key]
            text = f"\nPassage: {paragraph['title']}\n{paragraph['paragraph_text']}\n"
            encoded[key] = tokenizer(text, add_special_tokens=False)["input_ids"]
        return encoded[key]

    core_keys = list(selected_keys)
    rng.shuffle(core_keys)
    core_tokens = sum(len(encode(key)) for key in core_keys)
    if core_tokens >= target_tokens:
        raise RuntimeError("selected MuSiQue paragraphs alone exceed the token budget")
    filler_key = None
    for key in distractor_keys:
        values = encode(key)
        if core_tokens + len(values) >= target_tokens:
            filler_key = key
            break
        core_keys.append(key)
        core_tokens += len(values)
    if filler_key is None:
        raise RuntimeError("MuSiQue paragraphs did not fill the requested token budget")
    rng.shuffle(core_keys)
    ordered_keys = [*core_keys, filler_key]

    stream = np.empty(target_tokens, dtype=np.int32)
    paragraph_ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key in ordered_keys:
        values = encode(key)
        count = min(len(values), target_tokens - cursor)
        if count <= 0:
            break
        stream[cursor : cursor + count] = np.asarray(values[:count], dtype=np.int32)
        paragraph_ranges[key] = (cursor, cursor + count)
        cursor += count
    if cursor != target_tokens:
        raise RuntimeError(f"only wrote {cursor}/{target_tokens} tokens")
    blocks = stream.reshape(-1, args.block_tokens)
    np.save(output_dir / "blocks.npy", blocks)

    block_paragraphs: list[list[str]] = [[] for _ in range(blocks.shape[0])]
    for key, (paragraph_start, paragraph_end) in paragraph_ranges.items():
        first_block = paragraph_start // args.block_tokens
        last_block = (paragraph_end - 1) // args.block_tokens
        for block_id in range(first_block, last_block + 1):
            block_paragraphs[block_id].append(key)
    block_rows = []
    for block_id in range(blocks.shape[0]):
        start = block_id * args.block_tokens
        end = start + args.block_tokens
        block_rows.append(
            {
                "block_id": block_id,
                "token_start": start,
                "token_end": end,
                "paragraph_keys": block_paragraphs[block_id],
            }
        )
    write_jsonl(output_dir / "blocks.jsonl", block_rows)

    step_rows = []
    query_rows = []
    alignment_failures = []
    for query_id, (split, row) in enumerate(selected):
        decomposition = row["question_decomposition"]
        answers_so_far: list[str] = []
        query_steps = []
        try:
            for step_index, decomposition_step in enumerate(decomposition):
                answer = str(decomposition_step["answer"])
                paragraph_index = int(decomposition_step["paragraph_support_idx"])
                paragraph = row["paragraphs"][paragraph_index]
                key = paragraph_key(paragraph)
                paragraph_start, paragraph_end = paragraph_ranges[key]
                paragraph_tokens = stream[paragraph_start:paragraph_end].tolist()
                local_start, _local_end, aligned_answer = find_answer_span(
                    paragraph_tokens, answer, tokenizer
                )
                target_block = (paragraph_start + local_start) // args.block_tokens
                raw_question = str(decomposition_step["question"])
                atomic_question = render_atomic_question(raw_question, answers_so_far)
                lookup_key = (
                    first_subject(raw_question)
                    if step_index == 0
                    else answers_so_far[-1]
                )
                common = {
                "query_id": query_id,
                "dataset": "musique_official_2hop",
                "task_type": "multihop",
                "split": split,
                "question": str(row["question"]),
                "record_id": -1,
                "block_start": 0,
                "block_count": int(blocks.shape[0]),
                "hard_negative_block_ids": [],
                "annotation_uses_answer": True,
                "selection_uses_answer": False,
                "official_decomposition": True,
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
                    f"BRIDGE_ENTITY: {answer_value}" for answer_value in answers_so_far
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
            }
                if step_index == 1:
                    raw_template = raw_question.replace("#1", "{bridge}")
                    if ">>" in raw_template:
                        subject, relation = [
                            item.strip() for item in raw_template.split(">>", maxsplit=1)
                        ]
                        step["step_question_template"] = (
                            f"What is the {relation} of {subject}?"
                        )
                query_steps.append(step)
                answers_so_far.append(answer)
        except ValueError as error:
            alignment_failures.append(
                {"source_id": str(row["id"]), "split": split, "error": str(error)}
            )
            continue
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
    summary = {
        "source": "official MuSiQue answerable train/dev with supplied question decomposition",
        "contains_synthetic_vectors": False,
        "contains_synthetic_text": False,
        "selection_uses_answer": False,
        "annotation_uses_answer": True,
        "requested_tokens": args.seq_tokens,
        "num_tokens": int(stream.shape[0]),
        "num_blocks": int(blocks.shape[0]),
        "block_tokens": args.block_tokens,
        "paragraphs": len(paragraph_ranges),
        "queries": len(query_rows),
        "steps": len(step_rows),
        "alignment_failures": len(alignment_failures),
        "split_queries": {
            split: sum(row["split"] == split for row in query_rows)
            for split in ("train", "dev", "test")
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output_dir / "alignment_failures.jsonl", alignment_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
