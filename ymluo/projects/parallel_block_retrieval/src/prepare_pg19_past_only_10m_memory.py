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
        description="Prepare a shared causal PG19 memory containing no query-book future text."
    )
    parser.add_argument("--pg19_parquet", required=True)
    parser.add_argument(
        "--extra_text_dir",
        help="Optional disjoint PG19 plain-text books used only as distractors.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--query_samples", type=int, default=30)
    parser.add_argument("--memory_tokens", type=int, default=9_900_032)
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--local_context_tokens", type=int, default=512)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--min_external_history_tokens", type=int, default=32_768)
    parser.add_argument("--max_external_history_tokens", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.memory_tokens % args.block_tokens:
        raise ValueError("memory_tokens must be block aligned")
    if args.local_context_tokens <= 0 or args.target_tokens <= 0:
        raise ValueError("local_context_tokens and target_tokens must be positive")
    if args.min_external_history_tokens < args.block_tokens:
        raise ValueError("min_external_history_tokens is too small")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    separator = np.asarray(
        tokenizer("\n\n", add_special_tokens=False)["input_ids"], dtype=np.int32
    )
    table = pq.read_table(
        args.pg19_parquet,
        columns=["short_book_title", "publication_date", "url", "text"],
    )
    books: list[dict[str, Any]] = []
    for book_index, row in enumerate(table.to_pylist()):
        token_ids = np.asarray(
            tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"],
            dtype=np.int32,
        )
        books.append(
            {
                "book_index": book_index,
                "title": str(row["short_book_title"]),
                "publication_date": row["publication_date"],
                "url": str(row["url"]),
                "token_ids": token_ids,
            }
        )
    primary_books_total = len(books)

    required = (
        args.min_external_history_tokens
        + args.local_context_tokens
        + args.target_tokens
    )
    eligible = [book for book in books if len(book["token_ids"]) >= required]
    if len(eligible) < args.query_samples:
        raise RuntimeError(f"only {len(eligible)} books contain {required} tokens")
    rng = random.Random(args.seed)
    query_books = rng.sample(eligible, args.query_samples)
    query_book_ids = {int(book["book_index"]) for book in query_books}

    query_segments = []
    queries = []
    targets = []
    metadata = []
    for query_id, book in enumerate(query_books):
        token_ids = book["token_ids"]
        max_history = min(
            args.max_external_history_tokens,
            len(token_ids) - args.local_context_tokens - args.target_tokens,
        )
        low_block = math_ceil_div(args.min_external_history_tokens, args.block_tokens)
        high_block = max_history // args.block_tokens
        if high_block < low_block:
            raise RuntimeError(f"book {book['book_index']} has no valid aligned split")
        history_tokens = rng.randint(low_block, high_block) * args.block_tokens
        local_start = history_tokens
        target_start = local_start + args.local_context_tokens
        query_segments.append(
            {
                "scope_id": int(book["book_index"]),
                "scope_type": "query_book_past",
                "original_start_token": 0,
                "tokens": token_ids[:history_tokens],
            }
        )
        queries.append(token_ids[local_start:target_start])
        targets.append(token_ids[target_start : target_start + args.target_tokens])
        metadata.append(
            {
                "query_id": query_id,
                "book_index": int(book["book_index"]),
                "book_title": book["title"],
                "publication_date": book["publication_date"],
                "url": book["url"],
                "external_history_tokens": history_tokens,
                "local_context_start_token": local_start,
                "target_start_token": target_start,
                "book_tokens": len(token_ids),
                "memory_contains_query_book_future": False,
                "predefined_source": False,
                "selection_uses_target": False,
            }
        )

    if args.extra_text_dir:
        extra_paths = sorted(Path(args.extra_text_dir).rglob("*.txt"))
        for extra_index, path in enumerate(extra_paths):
            text = path.read_text(encoding="utf-8")
            token_ids = np.asarray(
                tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int32
            )
            books.append(
                {
                    "book_index": primary_books_total + extra_index,
                    "title": path.stem,
                    "publication_date": None,
                    "url": str(path),
                    "token_ids": token_ids,
                }
            )

    distractors = [
        {
            "scope_id": int(book["book_index"]),
            "scope_type": "distractor_book",
            "original_start_token": 0,
            "tokens": book["token_ids"],
        }
        for book in books
        if int(book["book_index"]) not in query_book_ids
    ]
    rng.shuffle(query_segments)
    rng.shuffle(distractors)

    # Every query-book history is included in full. Distractor books fill the rest.
    segments = list(query_segments)
    used_distractors = []
    current_tokens = sum(len(item["tokens"]) + len(separator) for item in segments)
    for segment in distractors:
        if current_tokens >= args.memory_tokens:
            break
        segments.append(segment)
        used_distractors.append(int(segment["scope_id"]))
        current_tokens += len(segment["tokens"]) + len(separator)
    if current_tokens < args.memory_tokens:
        raise RuntimeError(
            f"PG19 provides only {current_tokens} memory tokens, need {args.memory_tokens}"
        )

    # Randomize complete segments, but put a distractor last so truncation cannot remove
    # any required query-book history.
    complete_segments = segments[:-1]
    final_segment = segments[-1]
    rng.shuffle(complete_segments)
    ordered_segments = complete_segments + [final_segment]

    base = np.empty(args.memory_tokens, dtype=np.int32)
    token_scope_ids = np.full(args.memory_tokens, -1, dtype=np.int16)
    token_original_positions = np.full(args.memory_tokens, -1, dtype=np.int64)
    manifest = []
    cursor = 0
    query_scopes_written = set()
    for segment in ordered_segments:
        if cursor >= args.memory_tokens:
            break
        token_ids = segment["tokens"]
        take = min(len(token_ids), args.memory_tokens - cursor)
        start = cursor
        base[start : start + take] = token_ids[:take]
        token_scope_ids[start : start + take] = int(segment["scope_id"])
        original_start = int(segment["original_start_token"])
        token_original_positions[start : start + take] = np.arange(
            original_start, original_start + take, dtype=np.int64
        )
        cursor += take
        if segment["scope_type"] == "query_book_past" and take == len(token_ids):
            query_scopes_written.add(int(segment["scope_id"]))
        manifest.append(
            {
                "scope_id": int(segment["scope_id"]),
                "scope_type": str(segment["scope_type"]),
                "start_token": start,
                "end_token": cursor,
                "original_start_token": original_start,
                "original_end_token": original_start + take,
                "truncated": take < len(token_ids),
            }
        )
        if cursor >= args.memory_tokens:
            break
        separator_take = min(len(separator), args.memory_tokens - cursor)
        base[cursor : cursor + separator_take] = separator[:separator_take]
        cursor += separator_take
    if cursor != args.memory_tokens:
        raise RuntimeError(f"constructed {cursor} tokens, need {args.memory_tokens}")
    if query_scopes_written != query_book_ids:
        missing = sorted(query_book_ids - query_scopes_written)
        raise RuntimeError(f"query-book histories were truncated or omitted: {missing}")

    block_tokens = args.block_tokens
    scope_matrix = token_scope_ids.reshape(-1, block_tokens)
    position_matrix = token_original_positions.reshape(-1, block_tokens)
    block_scope_ids = np.full(len(scope_matrix), -1, dtype=np.int16)
    block_original_centers = np.full(len(scope_matrix), -1, dtype=np.int64)
    mixed_scope_blocks = 0
    for block_id, (scopes, positions) in enumerate(zip(scope_matrix, position_matrix)):
        valid_scopes = scopes[scopes >= 0]
        if not len(valid_scopes):
            continue
        unique, counts = np.unique(valid_scopes, return_counts=True)
        scope_id = int(unique[int(np.argmax(counts))])
        block_scope_ids[block_id] = scope_id
        valid_positions = positions[(scopes == scope_id) & (positions >= 0)]
        block_original_centers[block_id] = int(np.median(valid_positions))
        if len(unique) > 1:
            mixed_scope_blocks += 1

    np.save(output_dir / "base_blocks.npy", base.reshape(-1, block_tokens))
    np.save(output_dir / "base_block_scope_ids.npy", block_scope_ids)
    np.save(output_dir / "base_block_original_centers.npy", block_original_centers)
    np.save(
        output_dir / "source_blocks.npy",
        np.empty((args.query_samples, 0, block_tokens), dtype=np.int32),
    )
    np.save(output_dir / "queries.npy", np.stack(queries))
    np.save(output_dir / "targets.npy", np.stack(targets))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "segment_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    causal_violations = 0
    for row in metadata:
        scope = int(row["book_index"])
        limit = int(row["local_context_start_token"])
        mask = block_scope_ids == scope
        causal_violations += int(np.sum(block_original_centers[mask] >= limit))
    summary = {
        "source": "real PG19 past-only causal continuation memory",
        "pg19_parquet": args.pg19_parquet,
        "books_total": len(books),
        "primary_test_books": primary_books_total,
        "extra_disjoint_distractor_books": len(books) - primary_books_total,
        "query_samples": args.query_samples,
        "query_books": len(query_book_ids),
        "distractor_books_used": len(set(used_distractors)),
        "memory_scales_tokens": [args.memory_tokens],
        "memory_scales_blocks": [args.memory_tokens // block_tokens],
        "memory_tokens": args.memory_tokens,
        "max_base_tokens": args.memory_tokens,
        "max_base_blocks": args.memory_tokens // block_tokens,
        "block_tokens": block_tokens,
        "source_tokens": 0,
        "source_blocks": 0,
        "local_context_tokens": args.local_context_tokens,
        "query_tokens": args.local_context_tokens,
        "target_tokens": args.target_tokens,
        "min_external_history_tokens": args.min_external_history_tokens,
        "max_external_history_tokens": args.max_external_history_tokens,
        "past_only": True,
        "predefined_source": False,
        "memory_contains_query_book_future": False,
        "query_book_future_block_violations": causal_violations,
        "all_query_histories_included_fully": True,
        "scope_type": "book",
        "mixed_scope_blocks": mixed_scope_blocks,
        "mixed_scope_block_rate": mixed_scope_blocks / len(block_scope_ids),
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def math_ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


if __name__ == "__main__":
    main()
