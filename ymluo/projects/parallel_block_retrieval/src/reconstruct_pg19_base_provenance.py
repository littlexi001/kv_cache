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
        description="Reconstruct and verify book provenance for an existing PG19 memory."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    table = pq.read_table(summary["pg19_parquet"], columns=["text"])
    tokenized: list[dict[str, Any]] = []
    for book_index, text in enumerate(table.column("text").to_pylist()):
        tokenized.append(
            {
                "book_index": book_index,
                "token_ids": tokenizer(str(text), add_special_tokens=False)["input_ids"],
            }
        )

    required = (
        int(summary["query_offset_tokens"])
        + int(summary["source_tokens"])
        + int(summary["query_tokens"])
        + int(summary["target_tokens"])
    )
    eligible = [book for book in tokenized if len(book["token_ids"]) >= required]
    query_books = sorted(
        eligible, key=lambda book: (len(book["token_ids"]), int(book["book_index"]))
    )[: int(summary["query_samples"])]
    query_ids = {int(book["book_index"]) for book in query_books}
    reserved_end = required
    segments: list[tuple[int, int, list[int]]] = []
    for book in tokenized:
        book_index = int(book["book_index"])
        token_ids = book["token_ids"]
        if book_index in query_ids:
            before = token_ids[: int(summary["query_offset_tokens"])]
            after = token_ids[reserved_end:]
            if before:
                segments.append((book_index, 0, before))
            if after:
                segments.append((book_index, 1, after))
        else:
            segments.append((book_index, 0, token_ids))
    random.Random(int(summary["seed"])).shuffle(segments)

    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    flat_base = base_blocks.reshape(-1)
    max_tokens = int(summary["max_base_tokens"])
    if len(flat_base) != max_tokens:
        raise RuntimeError("stored base length does not match summary")
    token_book_ids = np.full(max_tokens, -1, dtype=np.int16)
    manifest = []
    cursor = 0
    for book_index, part, segment in segments:
        if cursor >= max_tokens:
            break
        take = min(len(segment), max_tokens - cursor)
        expected = np.asarray(segment[:take], dtype=flat_base.dtype)
        if not np.array_equal(np.asarray(flat_base[cursor : cursor + take]), expected):
            raise RuntimeError(f"base token mismatch at offset {cursor}")
        token_book_ids[cursor : cursor + take] = book_index
        manifest.append(
            {
                "book_index": book_index,
                "part": part,
                "start_token": cursor,
                "end_token": cursor + take,
            }
        )
        cursor += take
        if cursor >= max_tokens:
            break
        separator_take = min(len(separator), max_tokens - cursor)
        expected_separator = np.asarray(separator[:separator_take], dtype=flat_base.dtype)
        if not np.array_equal(
            np.asarray(flat_base[cursor : cursor + separator_take]), expected_separator
        ):
            raise RuntimeError(f"separator mismatch at offset {cursor}")
        cursor += separator_take
    if cursor != max_tokens:
        raise RuntimeError(f"only reconstructed {cursor} of {max_tokens} tokens")

    block_tokens = int(summary["block_tokens"])
    token_matrix = token_book_ids.reshape(-1, block_tokens)
    block_book_ids = np.full(len(token_matrix), -1, dtype=np.int16)
    mixed_blocks = 0
    for block_id, values in enumerate(token_matrix):
        valid = values[values >= 0]
        if not len(valid):
            continue
        unique, counts = np.unique(valid, return_counts=True)
        block_book_ids[block_id] = unique[int(np.argmax(counts))]
        if len(unique) > 1:
            mixed_blocks += 1
    np.save(output_dir / "base_block_book_ids.npy", block_book_ids)
    with (output_dir / "segment_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    provenance_summary = {
        "source": "verified PG19 base block provenance",
        "base_tokens_verified_exactly": max_tokens,
        "base_blocks": len(block_book_ids),
        "books_in_base": len(set(int(item) for item in block_book_ids if item >= 0)),
        "mixed_book_blocks": mixed_blocks,
        "mixed_book_block_rate": mixed_blocks / len(block_book_ids),
        "separator_only_blocks": int((block_book_ids < 0).sum()),
        "query_book_indices": sorted(query_ids),
        "segments": len(manifest),
        "contains_synthetic_text": False,
    }
    (output_dir / "provenance_summary.json").write_text(
        json.dumps(provenance_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
