from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from evaluate_global_step_hybrid_candidates import (
    AnchorInvertedIndex,
    rank_or_zero,
    step_anchor_text,
)
from profile_real_qk import read_jsonl
from profile_step_state_q import step_state_text
from run_lexical_block_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast anchor-only global candidates for generated reasoning states."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--anchor_alias_fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    build_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    index = AnchorInvertedIndex(block_texts)
    build_seconds = time.perf_counter() - build_started
    rows = []
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        step
        for step in read_jsonl(Path(args.step_queries_path))
        if str(step["split"]) in allowed_splits
        and str(step["task_type"]) in allowed_tasks
        and int(step["query_id"]) not in excluded_ids
    ]
    for step in steps:
        anchor_text = step_anchor_text(step)
        started = time.perf_counter()
        use_alias = args.anchor_alias_fallback and str(step["step_type"]) == "resolve_answer_from_bridge"
        candidates = (
            index.search_alias_aware(
                anchor_text, step_state_text(step), args.candidate_blocks
            )
            if use_alias
            else index.search(anchor_text, step_state_text(step), args.candidate_blocks)
        )
        query_seconds = time.perf_counter() - started
        target = int(step["target_block_ids"][0])
        rows.append(
            {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "selection_uses_gold": False,
                "anchor_text": anchor_text,
                "anchor_seconds": query_seconds,
                "anchor_rank": rank_or_zero(candidates, target),
                "anchor_candidates": candidates,
                "target_block_id": target,
            }
        )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "fast exact-anchor inverted index over generated step state",
        "selection_uses_gold": False,
        "num_blocks": int(blocks.shape[0]),
        "steps": len(rows),
        "build_seconds": build_seconds,
        "mean_query_milliseconds": statistics.fmean(row["anchor_seconds"] for row in rows)
        * 1e3,
        "anchor_alias_fallback": args.anchor_alias_fallback,
        "recall_at_1": statistics.fmean(0 < row["anchor_rank"] <= 1 for row in rows),
        "recall_at_3": statistics.fmean(0 < row["anchor_rank"] <= 3 for row in rows),
        "recall_at_16": statistics.fmean(0 < row["anchor_rank"] <= 16 for row in rows),
        "recall_at_512": statistics.fmean(0 < row["anchor_rank"] <= 512 for row in rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
