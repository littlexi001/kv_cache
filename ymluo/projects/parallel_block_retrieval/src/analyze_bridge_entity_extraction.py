from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from run_global_bridge_controller_single import (
    extract_novel_entity_from_memory,
    lexical_window,
    memory_text,
)
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic bridge entities extracted from retrieved K3."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target_blocks", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quota_merge(
    primary: list[int], secondary: list[int], primary_quota: int, target_blocks: int
) -> list[int]:
    selected: list[int] = []
    for block_id in primary[:primary_quota]:
        if block_id not in selected:
            selected.append(block_id)
    for block_id in secondary:
        if block_id not in selected:
            selected.append(block_id)
        if len(selected) >= target_blocks:
            return selected
    for block_id in primary[primary_quota:]:
        if block_id not in selected:
            selected.append(block_id)
        if len(selected) >= target_blocks:
            break
    return selected


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    rows = read_jsonl(Path(args.results_path))
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    build_started = time.perf_counter()
    documents = decode_blocks(tokenizer, blocks)
    bm25 = BM25Index(documents, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    build_seconds = time.perf_counter() - build_started

    output_rows = []
    for row in rows:
        question = str(row["question"])
        selected = [int(item) for item in row["hop1_selected"]]
        memory = memory_text(tokenizer, blocks, selected)
        entity = extract_novel_entity_from_memory(memory, question)
        deterministic_selected: list[int] = []
        model_selected = [
            int(item) for item in row.get("search_trace", [{}])[0].get("selected_blocks", [])
        ]
        gold_hit = False
        gold_rank = None
        if entity is not None:
            query_text = f"{entity} {question}"
            scores = bm25.score([query_text])[0]
            deterministic_selected, ranked, _policy = lexical_window(
                scores,
                args.target_blocks,
                documents=documents,
                focus_entity=entity,
            )
            gold_ids = {int(item) for item in row["gold_block_ids"]}
            gold_hit = any(item in gold_ids for item in deterministic_selected)
            gold_rank = min(ranked.index(item) + 1 for item in gold_ids)
        gold_ids = {int(item) for item in row["gold_block_ids"]}
        model2_det1 = quota_merge(
            model_selected, deterministic_selected, 2, args.target_blocks
        )
        det2_model1 = quota_merge(
            deterministic_selected, model_selected, 2, args.target_blocks
        )
        union_k6 = list(dict.fromkeys([*model_selected, *deterministic_selected]))
        output_rows.append(
            {
                "query_id": int(row["query_id"]),
                "dataset": str(row["dataset"]),
                "question": question,
                "deterministic_entity": entity,
                "deterministic_selected": deterministic_selected,
                "deterministic_gold_hit": gold_hit,
                "deterministic_gold_rank": gold_rank,
                "model_bridge_gold_hit": bool(row.get("any_search_gold_hit")),
                "model_selected": model_selected,
                "model2_det1_selected": model2_det1,
                "model2_det1_gold_hit": any(item in gold_ids for item in model2_det1),
                "det2_model1_selected": det2_model1,
                "det2_model1_gold_hit": any(item in gold_ids for item in det2_model1),
                "union_k6_gold_hit": any(item in gold_ids for item in union_k6),
            }
        )

    payload = {
        "source": "deterministic relation-aware entity extraction from initial K3",
        "queries": len(output_rows),
        "bm25_build_seconds": build_seconds,
        "entity_extraction_rate": sum(
            item["deterministic_entity"] is not None for item in output_rows
        )
        / len(output_rows),
        "deterministic_gold_recall_at_3": sum(
            item["deterministic_gold_hit"] for item in output_rows
        )
        / len(output_rows),
        "model_bridge_gold_recall_at_3": sum(
            item["model_bridge_gold_hit"] for item in output_rows
        )
        / len(output_rows),
        "model2_det1_gold_recall_at_3": sum(
            item["model2_det1_gold_hit"] for item in output_rows
        )
        / len(output_rows),
        "det2_model1_gold_recall_at_3": sum(
            item["det2_model1_gold_hit"] for item in output_rows
        )
        / len(output_rows),
        "union_k6_gold_recall": sum(item["union_k6_gold_hit"] for item in output_rows)
        / len(output_rows),
        "median_deterministic_gold_rank": statistics.median(
            item["deterministic_gold_rank"]
            for item in output_rows
            if item["deterministic_gold_rank"] is not None
        ),
        "rows": output_rows,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
