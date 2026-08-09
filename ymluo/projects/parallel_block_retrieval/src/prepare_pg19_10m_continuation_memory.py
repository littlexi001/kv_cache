from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-filtered PG19 continuation queries in nested real-text memories."
    )
    parser.add_argument("--pg19_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--query_samples", type=int, default=30)
    parser.add_argument("--query_offset_tokens", type=int, default=2048)
    parser.add_argument("--memory_scales", default="40000,1000000,10000000")
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--source_tokens", type=int, default=512)
    parser.add_argument("--query_tokens", type=int, default=64)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def parse_scales(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("memory_scales must contain positive values")
    return values


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.memory_scales)
    if args.source_tokens % args.block_tokens:
        raise ValueError("source_tokens must be block aligned")
    if any(scale % args.block_tokens for scale in scales):
        raise ValueError("every memory scale must be block aligned")
    if min(scales) <= args.source_tokens:
        raise ValueError("every memory scale must exceed source_tokens")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    table = pq.read_table(
        args.pg19_parquet,
        columns=["short_book_title", "publication_date", "url", "text"],
    )
    rows = table.to_pylist()
    tokenized: list[dict[str, Any]] = []
    for book_index, row in enumerate(rows):
        token_ids = tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]
        tokenized.append(
            {
                "book_index": book_index,
                "title": str(row["short_book_title"]),
                "publication_date": row["publication_date"],
                "url": str(row["url"]),
                "token_ids": token_ids,
            }
        )

    required = (
        args.query_offset_tokens
        + args.source_tokens
        + args.query_tokens
        + args.target_tokens
    )
    eligible = [book for book in tokenized if len(book["token_ids"]) >= required]
    if len(eligible) < args.query_samples:
        raise RuntimeError(f"only {len(eligible)} books contain {required} tokens")
    # Reserve the shortest eligible books for queries so the remaining disjoint books
    # maximize the amount of unique distractor text.
    query_books = sorted(
        eligible, key=lambda book: (len(book["token_ids"]), int(book["book_index"]))
    )[: args.query_samples]
    query_ids = {int(book["book_index"]) for book in query_books}

    sources = []
    queries = []
    targets = []
    metadata = []
    for query_id, book in enumerate(query_books):
        token_ids = book["token_ids"]
        source_start = args.query_offset_tokens
        query_start = source_start + args.source_tokens
        target_start = query_start + args.query_tokens
        sources.append(
            np.asarray(
                token_ids[source_start : source_start + args.source_tokens],
                dtype=np.int32,
            )
        )
        queries.append(
            np.asarray(
                token_ids[query_start : query_start + args.query_tokens],
                dtype=np.int32,
            )
        )
        targets.append(
            np.asarray(
                token_ids[target_start : target_start + args.target_tokens],
                dtype=np.int32,
            )
        )
        metadata.append(
            {
                "query_id": query_id,
                "book_index": int(book["book_index"]),
                "book_title": book["title"],
                "publication_date": book["publication_date"],
                "url": book["url"],
                "source_start_token": source_start,
                "book_tokens": len(token_ids),
                "selection_uses_target": False,
            }
        )

    reserved_end = (
        args.query_offset_tokens
        + args.source_tokens
        + args.query_tokens
        + args.target_tokens
    )
    distractor_segments: list[tuple[int, int, list[int]]] = []
    for book in tokenized:
        book_index = int(book["book_index"])
        token_ids = book["token_ids"]
        if book_index in query_ids:
            before = token_ids[: args.query_offset_tokens]
            after = token_ids[reserved_end:]
            if before:
                distractor_segments.append((book_index, 0, before))
            if after:
                distractor_segments.append((book_index, 1, after))
        else:
            distractor_segments.append((book_index, 0, token_ids))
    random.Random(args.seed).shuffle(distractor_segments)

    max_base_tokens = max(scales) - args.source_tokens
    base_stream: list[int] = []
    used_book_ids: set[int] = set()
    used_segments = 0
    for book_index, _, segment in distractor_segments:
        base_stream.extend(segment)
        base_stream.extend(separator)
        used_book_ids.add(book_index)
        used_segments += 1
        if len(base_stream) >= max_base_tokens:
            break
    if len(base_stream) < max_base_tokens:
        raise RuntimeError(
            f"leakage-filtered book segments provide {len(base_stream)} tokens, need {max_base_tokens}"
        )
    base = np.asarray(base_stream[:max_base_tokens], dtype=np.int32)
    np.save(output_dir / "base_blocks.npy", base.reshape(-1, args.block_tokens))
    np.save(
        output_dir / "source_blocks.npy",
        np.stack(sources).reshape(args.query_samples, -1, args.block_tokens),
    )
    np.save(output_dir / "queries.npy", np.stack(queries))
    np.save(output_dir / "targets.npy", np.stack(targets))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "real PG19 book continuation memory",
        "pg19_parquet": args.pg19_parquet,
        "books_total": len(tokenized),
        "query_books": args.query_samples,
        "distractor_segments_available": len(distractor_segments),
        "distractor_segments_used": used_segments,
        "distractor_books_used": len(used_book_ids),
        "query_distractor_books_disjoint": False,
        "reserved_source_query_target_spans_removed_from_base": True,
        "query_samples": args.query_samples,
        "memory_scales_tokens": scales,
        "memory_scales_blocks": [scale // args.block_tokens for scale in scales],
        "max_base_tokens": max_base_tokens,
        "max_base_blocks": max_base_tokens // args.block_tokens,
        "block_tokens": args.block_tokens,
        "source_tokens": args.source_tokens,
        "source_blocks": args.source_tokens // args.block_tokens,
        "query_tokens": args.query_tokens,
        "target_tokens": args.target_tokens,
        "query_offset_tokens": args.query_offset_tokens,
        "virtual_memory_contract": (
            "nested unique-book-text base with every query's reserved source/query/target "
            "span removed, plus query-specific contiguous source; total tokens exactly "
            "equal the named scale"
        ),
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
