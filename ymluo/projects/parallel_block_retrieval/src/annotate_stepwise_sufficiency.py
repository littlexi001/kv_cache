from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add explicit per-step bridge and minimal-evidence labels."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_step_rows(query: dict[str, Any]) -> list[dict[str, Any]]:
    gold_ids = [int(item) for item in query["gold_block_ids"]]
    evidence = [str(item) for item in query["evidence_texts"]]
    negatives = [int(item) for item in query.get("hard_negative_block_ids", [])]
    common = {
        "query_id": int(query["query_id"]),
        "dataset": str(query["dataset"]),
        "task_type": str(query["task_type"]),
        "split": str(query["split"]),
        "question": str(query["question"]),
        "record_id": int(query["record_id"]),
        "block_start": int(query["block_start"]),
        "block_count": int(query["block_count"]),
        "hard_negative_block_ids": negatives,
    }
    if query["task_type"] != "multihop":
        if len(gold_ids) != 1 or len(evidence) != 1:
            raise ValueError("single-hop query must contain one evidence block")
        return [
            {
                **common,
                "step_index": 0,
                "step_type": "direct_answer",
                "step_question": str(query["question"]),
                "retrieval_state": str(query["question"]),
                "compact_state_before": [],
                "full_state_before": [],
                "target_fact": evidence[0],
                "target_output": str(query["answers"][0]),
                "target_block_ids": [gold_ids[0]],
                "previous_evidence_block_ids": [],
                "minimal_sufficient_block_ids": [gold_ids[0]],
            }
        ]
    if len(gold_ids) != 2 or len(evidence) != 2:
        raise ValueError("multihop query must contain exactly two ordered evidence blocks")
    bridge = str(query["entity"])
    alias_match = re.search(r"\b[A-Z][A-Za-z]+-\d{4}\b", str(query["question"]))
    if alias_match is None:
        raise ValueError("multihop question does not expose its identifier")
    alias = alias_match.group(0)
    return [
        {
            **common,
            "step_index": 0,
            "step_type": "resolve_bridge",
            "step_operator": "resolve_identifier",
            "lookup_key": alias,
            "step_question": f"Which different entity does Memory link to {alias}?",
            "retrieval_state": str(query["question"]),
            "compact_state_before": [],
            "full_state_before": [],
            "target_fact": evidence[0],
            "target_output": bridge,
            "target_block_ids": [gold_ids[0]],
            "previous_evidence_block_ids": [],
            "minimal_sufficient_block_ids": [gold_ids[0]],
        },
        {
            **common,
            "step_index": 1,
            "step_type": "resolve_answer_from_bridge",
            "step_operator": "read_remaining_property",
            "step_question": (
                "For the entity in the verified mapping, what value of the remaining "
                "property requested by the original question is stated?"
            ),
            "retrieval_state": f"{bridge} {query['question']}",
            "compact_state_before": [f"BRIDGE_ENTITY: {bridge}"],
            "full_state_before": [evidence[0]],
            "target_fact": evidence[1],
            "target_output": str(query["answers"][0]),
            "target_block_ids": [gold_ids[1]],
            "previous_evidence_block_ids": [gold_ids[0]],
            "minimal_sufficient_block_ids": [gold_ids[1]],
        },
    ]


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    step_rows = [step for query in queries for step in build_step_rows(query)]

    missing_target_output = 0
    missing_target_fact = 0
    answer_leaks_in_step0 = 0
    query_by_id = {int(item["query_id"]): item for item in queries}
    for step in step_rows:
        texts = [
            tokenizer.decode(blocks[block_id].tolist(), skip_special_tokens=True)
            for block_id in step["target_block_ids"]
        ]
        joined = " ".join(texts).casefold()
        if str(step["target_output"]).casefold() not in joined:
            missing_target_output += 1
        if str(step["target_fact"]).casefold() not in joined:
            missing_target_fact += 1
        if step["step_type"] == "resolve_bridge":
            answer = str(query_by_id[int(step["query_id"])] ["answers"][0]).casefold()
            if answer in joined:
                answer_leaks_in_step0 += 1

    write_jsonl(output_dir / "step_queries.jsonl", step_rows)
    counts = Counter((row["split"], row["step_type"]) for row in step_rows)
    summary = {
        "source": "explicit step labels derived from controlled synthetic evidence order",
        "corpus_dir": str(corpus_dir),
        "queries": len(queries),
        "steps": len(step_rows),
        "counts": {f"{split}/{step_type}": count for (split, step_type), count in counts.items()},
        "audit": {
            "steps_missing_target_output_in_target_block": missing_target_output,
            "steps_missing_exact_target_fact_in_target_block": missing_target_fact,
            "bridge_steps_leaking_final_answer": answer_leaks_in_step0,
        },
        "step_queries_path": str(output_dir / "step_queries.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
