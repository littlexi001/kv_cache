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
        description=(
            "Prepare a shared real-news memory for causal continuation retrieval at "
            "nested 40K, 1M, and 10M scales."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", default="xsum")
    parser.add_argument("--distractor_split", default="train")
    parser.add_argument("--query_split", default="test")
    parser.add_argument("--query_samples", type=int, default=100)
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
        raise ValueError("memory_scales must contain positive integers")
    return values


def tokenize_document(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    text = str(row.get("document", "")).strip()
    if not text:
        return []
    return tokenizer(text, add_special_tokens=False)["input_ids"]


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
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

    query_dataset = load_dataset(args.dataset_name, split=args.query_split)
    query_order = list(range(len(query_dataset)))
    random.Random(args.seed).shuffle(query_order)
    required = args.source_tokens + args.query_tokens + args.target_tokens
    sources: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for row_index in query_order:
        row = query_dataset[int(row_index)]
        token_ids = tokenize_document(tokenizer, row)
        if len(token_ids) < required:
            continue
        sources.append(np.asarray(token_ids[: args.source_tokens], dtype=np.int32))
        queries.append(
            np.asarray(
                token_ids[args.source_tokens : args.source_tokens + args.query_tokens],
                dtype=np.int32,
            )
        )
        targets.append(np.asarray(token_ids[required - args.target_tokens : required], dtype=np.int32))
        metadata.append(
            {
                "query_id": len(metadata),
                "source_document_id": str(row.get("id", row_index)),
                "query_split_row": int(row_index),
                "source_chars": len(str(row.get("document", ""))),
                "selection_uses_target": False,
            }
        )
        if len(metadata) >= args.query_samples:
            break
    if len(metadata) < args.query_samples:
        raise RuntimeError(
            f"only {len(metadata)} query documents contain at least {required} tokens"
        )

    max_base_tokens = max(scales) - args.source_tokens
    distractor_dataset = load_dataset(args.dataset_name, split=args.distractor_split)
    distractor_order = list(range(len(distractor_dataset)))
    random.Random(args.seed + 1).shuffle(distractor_order)
    base_stream: list[int] = []
    used_documents = 0
    for row_index in distractor_order:
        token_ids = tokenize_document(tokenizer, distractor_dataset[int(row_index)])
        if not token_ids:
            continue
        base_stream.extend(token_ids)
        base_stream.extend(separator)
        used_documents += 1
        if len(base_stream) >= max_base_tokens:
            break
    if len(base_stream) < max_base_tokens:
        raise RuntimeError(
            f"distractor split produced {len(base_stream)} tokens, need {max_base_tokens}"
        )
    base = np.asarray(base_stream[:max_base_tokens], dtype=np.int32)
    if len(base) % args.block_tokens:
        raise AssertionError("base memory is not block aligned")

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
        "source": "real XSum BBC news continuation memory",
        "dataset_name": args.dataset_name,
        "distractor_split": args.distractor_split,
        "query_split": args.query_split,
        "split_disjoint": args.distractor_split != args.query_split,
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
        "distractor_documents_used": used_documents,
        "virtual_memory_contract": (
            "at each scale, use the nested base prefix plus the query-specific source "
            "blocks; total tokens exactly equal the named scale"
        ),
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
