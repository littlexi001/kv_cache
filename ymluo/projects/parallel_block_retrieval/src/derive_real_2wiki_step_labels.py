from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from run_step_state_kv_span_retrieval import find_text_subsequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive high-confidence ordered two-hop labels from structured 2Wiki passages."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--queries_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--decomposition_mode",
        choices=["typed_rules", "generic_state"],
        default="typed_rules",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalized_phrase(text)} "
    normalized_target = normalized_phrase(phrase)
    return bool(normalized_target) and f" {normalized_target} " in normalized_text


def parse_passages(context: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"(?:^|\n)Passage\s+\d+:\s*\n", context))
    passages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        payload = context[match.end() : end].strip()
        lines = payload.splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if title and body:
            passages.append({"title": title, "body": body})
    return passages


def sentence_containing(text: str, phrase: str) -> str | None:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if item.strip()
    ]
    matches = [sentence for sentence in sentences if contains_phrase(sentence, phrase)]
    if not matches:
        return None
    return min(matches, key=lambda item: (len(item), item))


def derive_chain(
    question: str, answers: Sequence[str], context: str
) -> tuple[dict[str, str] | None, str]:
    passages = parse_passages(context)
    if len(passages) < 2:
        return None, "fewer_than_two_passages"
    seed_candidates = [
        passage
        for passage in passages
        if len(normalized_phrase(passage["title"])) >= 4
        and contains_phrase(question, passage["title"])
    ]
    if not seed_candidates:
        return None, "no_seed_title_in_question"
    seed_candidates.sort(key=lambda item: (-len(normalized_phrase(item["title"])), item["title"]))
    answer_matches = []
    for passage in passages:
        for answer in answers:
            if len(normalized_phrase(answer)) >= 2 and contains_phrase(passage["body"], answer):
                answer_matches.append((passage, answer))
    if not answer_matches:
        return None, "no_answer_passage"
    candidates = []
    for seed in seed_candidates:
        for answer_passage, answer in answer_matches:
            if seed["title"] == answer_passage["title"]:
                continue
            bridge = answer_passage["title"]
            if contains_phrase(question, bridge):
                continue
            bridge_fact = sentence_containing(seed["body"], bridge)
            answer_fact = sentence_containing(answer_passage["body"], answer)
            if bridge_fact and answer_fact:
                candidates.append(
                    {
                        "lookup_key": seed["title"],
                        "bridge": bridge,
                        "answer": answer,
                        "bridge_fact": bridge_fact,
                        "answer_fact": answer_fact,
                    }
                )
    unique = {
        (
            item["lookup_key"],
            item["bridge"],
            item["answer"],
            item["bridge_fact"],
            item["answer_fact"],
        ): item
        for item in candidates
    }
    if len(unique) != 1:
        return None, "ambiguous_or_missing_path"
    return next(iter(unique.values())), "ok"


def split_for_source_index(index: int) -> str:
    remainder = index % 5
    if remainder == 0:
        return "test"
    if remainder == 1:
        return "dev"
    return "train"


def decompose_question(question: str, lookup_key: str) -> dict[str, str] | None:
    normalized = " ".join(question.strip().split())
    spouse_director = re.fullmatch(
        r"Who is the spouse of the director of film .+\?",
        normalized,
        flags=re.IGNORECASE,
    )
    if spouse_director and contains_phrase(question, lookup_key):
        return {
            "step0_operator": "resolve_director",
            "step0_question": f"Who directed the film {lookup_key}?",
            "step1_operator": "resolve_spouse",
            "step1_question_template": "Who is the spouse of {bridge}?",
        }
    father_death = re.fullmatch(
        r"When did .+['’]s father die\?",
        normalized,
        flags=re.IGNORECASE,
    )
    if father_death and contains_phrase(question, lookup_key):
        return {
            "step0_operator": "resolve_father",
            "step0_question": f"Who is the father of {lookup_key}?",
            "step1_operator": "resolve_death_date",
            "step1_question_template": "When did {bridge} die?",
        }
    return None


def find_fact_block(
    blocks: np.ndarray,
    block_start: int,
    block_count: int,
    fact: str,
    tokenizer: Any,
) -> int | None:
    matches = []
    for block_id in range(block_start, block_start + block_count):
        try:
            find_text_subsequence(blocks[block_id].tolist(), fact, tokenizer)
            matches.append(block_id)
        except ValueError:
            continue
    return matches[0] if len(matches) == 1 else None


def generic_decomposition(question: str, lookup_key: str) -> dict[str, str]:
    return {
        "step0_operator": "generic_link",
        "step0_question": (
            f"Original question: {question} Starting from {lookup_key}, which new entity "
            "stated in the memory is needed for the next lookup?"
        ),
        "step1_operator": "generic_answer",
        "step1_question_template": (
            f"Original question: {question} The verified intermediate entity is "
            "{bridge}. What value about {bridge} answers the original question?"
        ),
    }


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    queries_path = Path(args.queries_path) if args.queries_path else corpus_dir / "queries.jsonl"
    queries = [
        row
        for row in read_jsonl(queries_path)
        if str(row["dataset"]) == "2wikimqa"
    ]
    steps = []
    examples = []
    reasons: Counter[str] = Counter()
    for query in queries:
        chain, reason = derive_chain(
            str(query["question"]),
            [str(item) for item in query["answers"]],
            str(query["context"]),
        )
        if chain is None:
            reasons[reason] += 1
            continue
        decomposition = (
            generic_decomposition(str(query["question"]), chain["lookup_key"])
            if args.decomposition_mode == "generic_state"
            else decompose_question(str(query["question"]), chain["lookup_key"])
        )
        if decomposition is None:
            reasons["unsupported_relation_decomposition"] += 1
            continue
        block_start = int(query["block_start"])
        block_count = int(query["block_count"])
        bridge_block = find_fact_block(
            blocks, block_start, block_count, chain["bridge_fact"], tokenizer
        )
        answer_block = find_fact_block(
            blocks, block_start, block_count, chain["answer_fact"], tokenizer
        )
        if bridge_block is None or answer_block is None or bridge_block == answer_block:
            reasons["fact_block_alignment_failed"] += 1
            continue
        query_id = int(query["query_id"])
        split = split_for_source_index(int(query["source_index"]))
        common = {
            "query_id": query_id,
            "dataset": "real_2wikimqa_derived",
            "task_type": "multihop",
            "split": split,
            "question": str(query["question"]),
            "record_id": int(query.get("record_id", -1)),
            "block_start": block_start,
            "block_count": block_count,
            "hard_negative_block_ids": [],
            "annotation_uses_answer": True,
            "selection_uses_answer": False,
        }
        steps.extend(
            [
                {
                    **common,
                    "step_index": 0,
                    "step_type": "resolve_bridge",
                    "step_operator": decomposition["step0_operator"],
                    "lookup_key": chain["lookup_key"],
                    "step_question": decomposition["step0_question"],
                    "retrieval_state": str(query["question"]),
                    "compact_state_before": [],
                    "full_state_before": [],
                    "target_fact": chain["bridge_fact"],
                    "target_output": chain["bridge"],
                    "target_block_ids": [bridge_block],
                    "previous_evidence_block_ids": [],
                    "minimal_sufficient_block_ids": [bridge_block],
                },
                {
                    **common,
                    "step_index": 1,
                    "step_type": "resolve_answer_from_bridge",
                    "step_operator": decomposition["step1_operator"],
                    "lookup_key": chain["lookup_key"],
                    "step_question": decomposition["step1_question_template"].format(
                        bridge=chain["bridge"]
                    ),
                    "step_question_template": decomposition[
                        "step1_question_template"
                    ],
                    "retrieval_state": f"{chain['bridge']} {query['question']}",
                    "compact_state_before": [f"BRIDGE_ENTITY: {chain['bridge']}"],
                    "full_state_before": [chain["bridge_fact"]],
                    "target_fact": chain["answer_fact"],
                    "target_output": chain["answer"],
                    "target_block_ids": [answer_block],
                    "previous_evidence_block_ids": [bridge_block],
                    "minimal_sufficient_block_ids": [answer_block],
                },
            ]
        )
        examples.append(
            {
                "query_id": query_id,
                "split": split,
                **chain,
                "bridge_block": bridge_block,
                "answer_block": answer_block,
            }
        )
        reasons["ok"] += 1
    write_jsonl(output_dir / "step_queries.jsonl", steps)
    write_jsonl(output_dir / "examples.jsonl", examples)
    summary = {
        "source": "deterministic title-link path derived from real LongBench 2WikiMQA",
        "annotation_uses_answer": True,
        "selection_uses_answer": False,
        "decomposition_mode": args.decomposition_mode,
        "queries_path": str(queries_path),
        "input_queries": len(queries),
        "derived_examples": len(examples),
        "steps": len(steps),
        "split_counts": dict(Counter(item["split"] for item in examples)),
        "reason_counts": dict(reasons),
        "step_queries_path": str(output_dir / "step_queries.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
