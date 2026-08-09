from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from profile_real_qk import read_jsonl


WINDOW_MODES = ("sum", "max_mean", "top2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent BM25 blocks with contiguous block windows."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_blocks", type=int, default=3)
    return parser.parse_args()


def independent_topk(scores: np.ndarray, target_blocks: int) -> list[int]:
    ids = np.arange(scores.shape[0], dtype=np.int64)
    return [int(item) for item in np.lexsort((ids, -scores))[:target_blocks]]


def window_score(values: np.ndarray, mode: str) -> float:
    if mode == "sum":
        return float(values.sum())
    if mode == "max_mean":
        return float(values.max() + 0.25 * values.mean())
    if mode == "top2":
        count = min(2, values.shape[0])
        return float(np.partition(values, -count)[-count:].sum())
    raise ValueError(f"unknown window mode: {mode}")


def best_contiguous_window(
    scores: np.ndarray,
    records: Sequence[dict[str, Any]],
    target_blocks: int,
    mode: str,
) -> list[int]:
    best_key: tuple[float, int] | None = None
    best_ids: list[int] = []
    for record in records:
        block_start = int(record["block_start"])
        block_end = block_start + int(record["block_count"])
        for start in range(block_start, block_end):
            end = min(start + target_blocks, block_end)
            key = (window_score(scores[start:end], mode), -start)
            if best_key is None or key > best_key:
                best_key = key
                best_ids = list(range(start, end))
    return best_ids


def evaluate_selection(
    selected: Sequence[int], query: dict[str, Any]
) -> tuple[bool, bool]:
    block_start = int(query["block_start"])
    block_end = block_start + int(query["block_count"])
    gold = {int(item) for item in query["gold_block_ids"]}
    return (
        any(block_start <= int(item) < block_end for item in selected),
        any(int(item) in gold for item in selected),
    )


def main() -> None:
    args = parse_args()
    if args.target_blocks < 1:
        raise ValueError("target_blocks must be positive")
    corpus_dir = Path(args.corpus_dir)
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    records = read_jsonl(corpus_dir / "records.jsonl")
    scores = np.load(args.scores_path, mmap_mode="r")
    if scores.shape[0] != len(queries):
        raise ValueError("score rows must align with queries.jsonl")

    rows: list[dict[str, Any]] = []
    methods = ("independent", *WINDOW_MODES)
    for query, query_scores in zip(queries, scores, strict=True):
        for method in methods:
            selected = (
                independent_topk(query_scores, args.target_blocks)
                if method == "independent"
                else best_contiguous_window(
                    query_scores, records, args.target_blocks, method
                )
            )
            record_hit, gold_hit = evaluate_selection(selected, query)
            rows.append(
                {
                    "method": method,
                    "query_id": int(query["query_id"]),
                    "dataset": str(query["dataset"]),
                    "selected_block_ids": selected,
                    "record_hit": record_hit,
                    "gold_hit": gold_hit,
                }
            )

    summaries = []
    for method in methods:
        method_rows = [item for item in rows if item["method"] == method]
        summaries.append(
            {
                "method": method,
                "queries": len(method_rows),
                "record_recall_at_k": sum(item["record_hit"] for item in method_rows)
                / len(method_rows),
                "gold_recall_at_k": sum(item["gold_hit"] for item in method_rows)
                / len(method_rows),
            }
        )
    payload = {
        "source": "saved real-text BM25 block scores",
        "target_blocks": args.target_blocks,
        "summaries": summaries,
        "rows": rows,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
