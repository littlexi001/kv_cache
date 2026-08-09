from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from profile_step_state_q import step_state_text
from run_iterative_condition_retrieval import BM25Index
from run_lexical_block_retrieval import decode_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile warm BM25 latency over the full block corpus."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_sizes", default="1,8,32")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def score_topk(
    index: BM25Index, queries: list[str], top_k: int, *, postings: bool
) -> np.ndarray:
    scores = index.score_postings(queries) if postings else index.score(queries)
    k = min(top_k, scores.shape[1])
    return np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmup < 0 or args.top_k <= 0:
        raise ValueError("repeats/top_k must be positive and warmup non-negative")
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("batch sizes must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    decode_started = time.perf_counter()
    block_texts = decode_blocks(tokenizer, blocks)
    decode_seconds = time.perf_counter() - decode_started
    build_started = time.perf_counter()
    index = BM25Index(block_texts, min_df=1, max_df=1.0, k1=1.2, b=0.75)
    build_seconds = time.perf_counter() - build_started

    queries = [step_state_text(row) for row in read_jsonl(Path(args.step_queries_path))]
    if not queries:
        raise ValueError("no queries found")
    summaries = []
    for mode in ("matrix", "postings"):
        for batch_size in batch_sizes:
            batches = [
                [queries[(offset + item) % len(queries)] for item in range(batch_size)]
                for offset in range(args.repeats + args.warmup)
            ]
            for batch in batches[: args.warmup]:
                score_topk(index, batch, args.top_k, postings=mode == "postings")
            elapsed = []
            for batch in batches[args.warmup :]:
                started = time.perf_counter()
                score_topk(index, batch, args.top_k, postings=mode == "postings")
                elapsed.append(time.perf_counter() - started)
            summaries.append(
                {
                    "mode": mode,
                    "batch_size": batch_size,
                    "repeats": args.repeats,
                    "mean_batch_ms": 1000.0 * statistics.fmean(elapsed),
                    "median_batch_ms": 1000.0 * statistics.median(elapsed),
                    "p95_batch_ms": 1000.0 * percentile(elapsed, 95),
                    "mean_query_ms": 1000.0 * statistics.fmean(elapsed) / batch_size,
                    "queries_per_second": batch_size / statistics.fmean(elapsed),
                }
            )
    check_queries = queries[: min(8, len(queries))]
    matrix_scores = index.score(check_queries)
    postings_scores = index.score_postings(check_queries)
    max_abs_score_error = float(np.max(np.abs(matrix_scores - postings_scores)))
    payload = {
        "source": "warm BM25 full-corpus query plus Top-K selection latency",
        "num_blocks": int(len(blocks)),
        "top_k": args.top_k,
        "decode_seconds": decode_seconds,
        "build_seconds": build_seconds,
        "features": index.features,
        "warmup_batches": args.warmup,
        "max_abs_score_error": max_abs_score_error,
        "summaries": summaries,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
