from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from evaluate_pg19_candidate_utility_landscape import context_for_window, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode held-out PG19 utility-persistence successes and failures."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--examples_per_group", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=1600)
    return parser.parse_args()


def words(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z]{3,}", text)}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = read_jsonl(data_dir / "metadata.jsonl")
    rows = read_jsonl(args.rows)
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    block_tokens = int(summary["block_tokens"])
    total_blocks = max(max(int(item) for item in row["block_ids"]) for row in rows) + 1
    source_count = int(summary["source_blocks"])
    base_count = total_blocks - source_count
    window_blocks = len(rows[0]["block_ids"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    retrieval_methods = {"bm25", "e5", "bm25_e5_rrf"}
    selected = []
    for query_id in sorted({int(row["query_id"]) for row in rows}):
        group = [
            row
            for row in rows
            if int(row["query_id"]) == query_id
            and int(row["source_overlap"]) == 0
            and retrieval_methods & set(row["origins"])
        ]
        selected.append(max(group, key=lambda row: float(row["delta_nll_a"])))

    def decode(row: dict[str, Any]) -> dict[str, Any]:
        query_id = int(row["query_id"])
        context_ids = context_for_window(
            int(row["window_start"]),
            window_blocks=window_blocks,
            base_blocks=base_blocks,
            source_blocks=source_blocks[query_id],
            base_count=base_count,
        )
        window_text = tokenizer.decode(context_ids.tolist(), skip_special_tokens=True)
        query_text = tokenizer.decode(
            np.asarray(queries[query_id], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        observed_text = tokenizer.decode(
            np.asarray(targets[query_id, :64], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        future_text = tokenizer.decode(
            np.asarray(targets[query_id, 64:], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        state_words = words(query_text + " " + observed_text)
        shared = sorted(words(window_text) & state_words)
        return {
            "query_id": query_id,
            "query_book_title": metadata[query_id]["book_title"],
            "origins": row["origins"],
            "delta_nll_on_observed_A": row["delta_nll_a"],
            "delta_nll_on_future_B": row["delta_nll_b"],
            "shared_words_with_observed_state": shared[:40],
            "candidate_window": window_text[: args.max_chars],
            "initial_query": query_text[: args.max_chars],
            "observed_segment_A": observed_text[: args.max_chars],
            "held_out_future_B": future_text[: args.max_chars],
            "selection_rule": "best delta-NLL on A among non-source retrieval candidates",
            "group_order_uses_B_for_qualitative_inspection_only": True,
        }

    successes = sorted(selected, key=lambda row: float(row["delta_nll_b"]), reverse=True)
    failures = sorted(selected, key=lambda row: float(row["delta_nll_b"]))
    output = {
        "source": "qualitative audit of real PG19 non-source utility persistence",
        "selection_uses_future_B": False,
        "group_order_uses_future_B": True,
        "persistent_successes": [
            decode(row) for row in successes[: args.examples_per_group]
        ],
        "retention_failures": [
            decode(row) for row in failures[: args.examples_per_group]
        ],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
