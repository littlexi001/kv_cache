from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25 block and record routing over the clean real-text corpus."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument("--record_allocations", default="20,30,39")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.98)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decode_blocks(tokenizer: AutoTokenizer, blocks: np.ndarray, batch_size: int = 512) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(blocks), batch_size):
        batch = np.asarray(blocks[start : start + batch_size], dtype=np.int64)
        texts.extend(tokenizer.batch_decode(batch, skip_special_tokens=True))
    return texts


def bm25_matrix(
    documents: list[str],
    queries: list[str],
    *,
    min_df: int,
    max_df: float,
    k1: float,
    b: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
    document_count = counts.shape[0]
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(
        (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)

    document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = max(float(document_lengths.mean()), 1.0e-6)
    row_ids = np.repeat(np.arange(document_count), np.diff(counts.indptr))
    term_frequency = counts.data
    denominator = term_frequency + k1 * (
        1.0 - b + b * document_lengths[row_ids] / average_length
    )
    counts.data = (
        inverse_document_frequency[counts.indices]
        * term_frequency
        * (k1 + 1.0)
        / denominator
    )

    query_counts = vectorizer.transform(queries).tocsr().astype(np.float32)
    query_counts.data.fill(1.0)
    scores = (query_counts @ counts.transpose()).toarray().astype(np.float32, copy=False)
    elapsed = time.perf_counter() - started
    return scores, {
        "documents": document_count,
        "features": counts.shape[1],
        "nonzero_weights": int(counts.nnz),
        "build_and_score_seconds": elapsed,
        "average_terms_per_document": average_length,
    }


def descending_ids(scores: np.ndarray) -> list[int]:
    ids = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((ids, -scores)).tolist()


def group_for_context(
    ranked_ids: list[int], block_to_record: np.ndarray
) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    record_order: list[int] = []
    for block_id in ranked_ids:
        record_id = int(block_to_record[block_id])
        if record_id not in groups:
            record_order.append(record_id)
        groups[record_id].append(block_id)
    output: list[int] = []
    for record_id in record_order:
        output.extend(sorted(groups[record_id]))
    return output


def select_global_blocks(scores: np.ndarray, target_blocks: int) -> tuple[list[int], list[int]]:
    ranked = descending_ids(scores)[:target_blocks]
    return ranked, ranked


def select_record_then_global(
    *,
    block_scores: np.ndarray,
    record_scores: np.ndarray,
    records: list[dict[str, Any]],
    block_to_record: np.ndarray,
    target_blocks: int,
    record_allocation: int,
) -> tuple[list[int], list[int], int, float]:
    ranked_records = descending_ids(record_scores)
    predicted_record = ranked_records[0]
    margin = float(record_scores[ranked_records[0]] - record_scores[ranked_records[1]])
    record = records[predicted_record]
    block_start = int(record["block_start"])
    block_count = int(record["block_count"])
    record_ids = list(range(block_start, block_start + block_count))
    record_ids.sort(key=lambda block_id: (-float(block_scores[block_id]), block_id))

    ranked = record_ids[: min(record_allocation, target_blocks)]
    selected = set(ranked)
    for block_id in descending_ids(block_scores):
        if len(ranked) >= target_blocks:
            break
        if block_id not in selected:
            ranked.append(block_id)
            selected.add(block_id)
    return ranked, group_for_context(ranked, block_to_record), predicted_record, margin


def evaluate_selection(
    *,
    method: str,
    query: dict[str, Any],
    ranked_ids: list[int],
    context_ids: list[int],
    predicted_record: int,
    source_record: int,
    record_margin: float,
) -> dict[str, Any]:
    gold = set(int(item) for item in query.get("gold_block_ids", []))
    gold_ranks = [rank + 1 for rank, block_id in enumerate(ranked_ids) if block_id in gold]
    block_start = int(query["block_start"])
    block_end = block_start + int(query["block_count"])
    source_hit = any(block_start <= block_id < block_end for block_id in context_ids)
    return {
        "method": method,
        "query_id": int(query["query_id"]),
        "dataset": query["dataset"],
        "source_record_recall": float(source_hit),
        "record_top1_recall": float(predicted_record == source_record),
        "answer_block_recall": float(bool(gold_ranks)),
        "answer_block_mrr": 1.0 / min(gold_ranks) if gold_ranks else 0.0,
        "gold_block_count": len(gold),
        "record_margin": record_margin,
        "selected_block_ids": json.dumps(context_ids),
        "ranked_block_ids": json.dumps(ranked_ids),
    }


def main() -> None:
    args = parse_args()
    if args.target_blocks <= 0:
        raise ValueError("target_blocks must be positive")
    allocations = sorted(
        {int(item) for item in args.record_allocations.split(",") if item.strip()}
    )
    if any(item <= 0 for item in allocations):
        raise ValueError("record allocations must be positive")

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    records = read_jsonl(corpus_dir / "records.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    decode_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    decode_seconds = time.perf_counter() - decode_started
    questions = [str(query["question"]) for query in queries]
    block_scores, block_index = bm25_matrix(
        block_texts,
        questions,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )

    record_texts: list[str] = []
    block_to_record = np.empty(len(blocks), dtype=np.int32)
    source_record_by_start: dict[int, int] = {}
    for record_id, record in enumerate(records):
        block_start = int(record["block_start"])
        block_count = int(record["block_count"])
        block_end = block_start + block_count
        record_texts.append("\n".join(block_texts[block_start:block_end]))
        block_to_record[block_start:block_end] = record_id
        source_record_by_start[block_start] = record_id
    record_scores, record_index = bm25_matrix(
        record_texts,
        questions,
        min_df=args.min_df,
        max_df=args.max_df,
        k1=args.k1,
        b=args.b,
    )

    rows: list[dict[str, Any]] = []
    retrieval_started = time.perf_counter()
    for query_index, query in enumerate(queries):
        source_record = source_record_by_start[int(query["block_start"])]
        ranked, _ = select_global_blocks(block_scores[query_index], args.target_blocks)
        context = group_for_context(ranked, block_to_record)
        record_order = descending_ids(record_scores[query_index])
        rows.append(
            evaluate_selection(
                method="bm25_block",
                query=query,
                ranked_ids=ranked,
                context_ids=context,
                predicted_record=record_order[0],
                source_record=source_record,
                record_margin=float(
                    record_scores[query_index, record_order[0]]
                    - record_scores[query_index, record_order[1]]
                ),
            )
        )
        for allocation in allocations:
            ranked, context, predicted_record, margin = select_record_then_global(
                block_scores=block_scores[query_index],
                record_scores=record_scores[query_index],
                records=records,
                block_to_record=block_to_record,
                target_blocks=args.target_blocks,
                record_allocation=allocation,
            )
            rows.append(
                evaluate_selection(
                    method=f"bm25_record{allocation}",
                    query=query,
                    ranked_ids=ranked,
                    context_ids=context,
                    predicted_record=predicted_record,
                    source_record=source_record,
                    record_margin=margin,
                )
            )
    retrieval_seconds = time.perf_counter() - retrieval_started

    methods = sorted({row["method"] for row in rows})
    summaries: list[dict[str, Any]] = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "queries": len(group),
                "source_record_recall": statistics.fmean(
                    row["source_record_recall"] for row in group
                ),
                "record_top1_recall": statistics.fmean(
                    row["record_top1_recall"] for row in group
                ),
                "answer_block_recall": statistics.fmean(
                    row["answer_block_recall"] for row in group
                ),
                "answer_block_mrr": statistics.fmean(
                    row["answer_block_mrr"] for row in group
                ),
            }
        )

    row_fields = list(rows[0])
    write_csv(output_dir / "query_results.csv", rows, row_fields)
    write_csv(output_dir / "method_summary.csv", summaries, list(summaries[0]))
    np.save(output_dir / "block_scores.npy", block_scores)
    np.save(output_dir / "record_scores.npy", record_scores)
    summary = {
        "source": "clean real-text corpus",
        "retriever": "BM25 over word unigrams and bigrams",
        "contains_synthetic_vectors": False,
        "num_blocks": len(blocks),
        "num_records": len(records),
        "num_queries": len(queries),
        "target_blocks": args.target_blocks,
        "record_allocations": allocations,
        "decode_seconds": decode_seconds,
        "block_index": block_index,
        "record_index": record_index,
        "block_scores_path": str(output_dir / "block_scores.npy"),
        "record_scores_path": str(output_dir / "record_scores.npy"),
        "retrieval_seconds": retrieval_seconds,
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
