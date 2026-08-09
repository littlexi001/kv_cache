from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare causal 40K-context XSum news continuation samples."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", default="xsum")
    parser.add_argument("--split", default="test")
    parser.add_argument("--natural_samples", type=int, default=20)
    parser.add_argument("--delayed_samples", type=int, default=20)
    parser.add_argument("--history_tokens", type=int, default=40_000)
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--query_tokens", type=int, default=64)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--source_tokens", type=int, default=512)
    parser.add_argument("--min_blocks_after_source", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def tokenize_documents(
    rows: Any,
    tokenizer: Any,
    separator: list[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    documents: list[dict[str, Any]] = []
    stream: list[int] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("document", "")).strip()
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        document_id = str(row.get("id", row_index))
        documents.append(
            {
                "document_id": document_id,
                "token_ids": token_ids,
                "text_chars": len(text),
            }
        )
        stream.extend(token_ids)
        stream.extend(separator)
    return documents, stream


def distractor_stream_without_document(
    documents: list[dict[str, Any]],
    *,
    excluded_document_id: str,
    start_index: int,
    length: int,
    separator: list[int],
) -> list[int]:
    output: list[int] = []
    document_index = start_index % len(documents)
    visited = 0
    while len(output) < length:
        document = documents[document_index]
        document_index = (document_index + 1) % len(documents)
        visited += 1
        if str(document["document_id"]) == excluded_document_id:
            continue
        output.extend(document["token_ids"])
        output.extend(separator)
        if visited > len(documents) * 2 and not output:
            raise RuntimeError("could not construct a target-free distractor stream")
    return output[:length]


def main() -> None:
    args = parse_args()
    if args.history_tokens % args.block_tokens:
        raise ValueError("history_tokens must be divisible by block_tokens")
    if args.source_tokens % args.block_tokens:
        raise ValueError("source_tokens must be divisible by block_tokens")
    if min(
        args.natural_samples,
        args.delayed_samples,
        args.history_tokens,
        args.block_tokens,
        args.query_tokens,
        args.target_tokens,
        args.source_tokens,
    ) <= 0:
        raise ValueError("all sample and token counts must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    dataset = load_dataset(args.dataset_name, split=args.split)
    documents, stream = tokenize_documents(dataset, tokenizer, separator)

    natural_span = args.history_tokens + args.query_tokens + args.target_tokens
    required_natural = natural_span * args.natural_samples
    if len(stream) < required_natural:
        raise RuntimeError(
            f"dataset stream has {len(stream)} tokens, fewer than required {required_natural}"
        )

    histories: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for sample_index in range(args.natural_samples):
        start = sample_index * natural_span
        span = stream[start : start + natural_span]
        histories.append(np.asarray(span[: args.history_tokens], dtype=np.int32))
        queries.append(
            np.asarray(
                span[args.history_tokens : args.history_tokens + args.query_tokens],
                dtype=np.int32,
            )
        )
        targets.append(np.asarray(span[-args.target_tokens :], dtype=np.int32))
        metadata.append(
            {
                "sample_id": len(metadata),
                "protocol": "natural_stream",
                "stream_start": start,
                "oracle_block_ids": [],
                "source_document_id": None,
                "selection_uses_target": False,
            }
        )

    delayed_needed = args.source_tokens + args.query_tokens + args.target_tokens
    delayed_documents = [
        document for document in documents if len(document["token_ids"]) >= delayed_needed
    ]
    if len(delayed_documents) < args.delayed_samples:
        raise RuntimeError(
            f"only {len(delayed_documents)} documents have at least {delayed_needed} tokens"
        )

    rng = random.Random(args.seed)
    rng.shuffle(delayed_documents)
    total_blocks = args.history_tokens // args.block_tokens
    source_blocks = args.source_tokens // args.block_tokens
    latest_insert = total_blocks - source_blocks - args.min_blocks_after_source
    if latest_insert <= args.min_blocks_after_source:
        raise ValueError("history is too short for the requested delayed-source margins")
    distractor_tokens = args.history_tokens - args.source_tokens

    for delayed_index, document in enumerate(delayed_documents[: args.delayed_samples]):
        token_ids = document["token_ids"]
        source = token_ids[: args.source_tokens]
        query = token_ids[
            args.source_tokens : args.source_tokens + args.query_tokens
        ]
        target_start = args.source_tokens + args.query_tokens
        target = token_ids[target_start : target_start + args.target_tokens]

        distractor_start = (delayed_index + args.natural_samples) * 97
        distractors = distractor_stream_without_document(
            documents,
            excluded_document_id=str(document["document_id"]),
            start_index=distractor_start,
            length=distractor_tokens,
            separator=separator,
        )
        insert_block = rng.randint(args.min_blocks_after_source, latest_insert)
        insert_token = insert_block * args.block_tokens
        history = distractors[:insert_token] + source + distractors[insert_token:]
        if len(history) != args.history_tokens:
            raise AssertionError("delayed history length changed during source insertion")

        histories.append(np.asarray(history, dtype=np.int32))
        queries.append(np.asarray(query, dtype=np.int32))
        targets.append(np.asarray(target, dtype=np.int32))
        metadata.append(
            {
                "sample_id": len(metadata),
                "protocol": "delayed_article",
                "stream_start": distractor_start,
                "oracle_block_ids": list(
                    range(insert_block, insert_block + source_blocks)
                ),
                "source_document_id": document["document_id"],
                "source_text_chars": document["text_chars"],
                "source_insert_block": insert_block,
                "selection_uses_target": False,
            }
        )

    history_array = np.stack(histories)
    query_array = np.stack(queries)
    target_array = np.stack(targets)
    np.save(output_dir / "histories.npy", history_array)
    np.save(output_dir / "queries.npy", query_array)
    np.save(output_dir / "targets.npy", target_array)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "real XSum BBC news text",
        "dataset_name": args.dataset_name,
        "dataset_split": args.split,
        "model_tokenizer": args.model_name_or_path,
        "samples": len(metadata),
        "natural_stream_samples": args.natural_samples,
        "delayed_article_samples": args.delayed_samples,
        "history_tokens": args.history_tokens,
        "block_tokens": args.block_tokens,
        "history_blocks": total_blocks,
        "query_tokens": args.query_tokens,
        "target_tokens": args.target_tokens,
        "source_tokens": args.source_tokens,
        "retrieval_blocks": 8,
        "retrieval_tokens": 8 * args.block_tokens,
        "dataset_documents": len(documents),
        "dataset_stream_tokens": len(stream),
        "eligible_delayed_documents": len(delayed_documents),
        "contains_synthetic_text": False,
        "delayed_protocol_changes_only_document_order": True,
        "selection_uses_target": False,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
