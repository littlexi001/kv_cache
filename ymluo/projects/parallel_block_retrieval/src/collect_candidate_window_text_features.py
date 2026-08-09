from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from collect_candidate_complementarity_features import (
    build_features,
    decode_selected_blocks,
    encode_texts,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect E5/text complementarity features for materialized candidate windows."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--embedding_model", default="intfloat/e5-base-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--embedding_max_length", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(args.candidate_rows)
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_scope_ids = np.asarray(
        np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    block_ids = sorted(
        {
            int(item)
            for row in candidates
            for field in ("previous_block_ids", "expanded_block_ids")
            for item in row[field]
        }
    )
    decode_tokenizer = AutoTokenizer.from_pretrained(args.decode_tokenizer, use_fast=True)
    started = time.perf_counter()
    block_texts = decode_selected_blocks(decode_tokenizer, base_blocks, block_ids)
    query_ids = sorted({int(row["query_id"]) for row in candidates})
    query_texts = {
        query_id: decode_tokenizer.decode(
            np.asarray(queries[query_id], dtype=np.int64).tolist(),
            skip_special_tokens=True,
        )
        for query_id in query_ids
    }
    decode_seconds = time.perf_counter() - started

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, use_fast=True)
    model = AutoModel.from_pretrained(args.embedding_model, torch_dtype=dtype).eval().to(device)
    started = time.perf_counter()
    block_vectors = encode_texts(
        model,
        tokenizer,
        [block_texts[item] for item in block_ids],
        prefix="passage: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    )
    tokenizer.truncation_side = "left"
    query_vectors = encode_texts(
        model,
        tokenizer,
        [query_texts[item] for item in query_ids],
        prefix="query: ",
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=device,
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    embedding_seconds = time.perf_counter() - started
    block_embeddings = {item: block_vectors[index] for index, item in enumerate(block_ids)}
    query_embeddings = {
        item: query_vectors[index] for index, item in enumerate(query_ids)
    }

    rows = []
    started = time.perf_counter()
    for candidate in candidates:
        query_id = int(candidate["query_id"])
        previous_ids = [int(item) for item in candidate["previous_block_ids"]]
        expanded_ids = [int(item) for item in candidate["expanded_block_ids"]]
        rows.append(
            {
                "query_id": query_id,
                "candidate_id": int(candidate["candidate_id"]),
                "features": build_features(
                    query_text=query_texts[query_id],
                    query_embedding=query_embeddings[query_id],
                    previous_ids=previous_ids,
                    expanded_ids=expanded_ids,
                    block_texts=block_texts,
                    block_embeddings=block_embeddings,
                    block_scope_ids=block_scope_ids,
                    previous_scope_ids=[],
                    expanded_scope_ids=sorted(
                        {int(block_scope_ids[item]) for item in expanded_ids}
                    ),
                ),
                "candidate_texts_observed": True,
                "reader_forward_used": False,
                "future_target_used": False,
                "selection_uses_target": False,
            }
        )
    feature_seconds = time.perf_counter() - started
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "source": "candidate-window E5/text complementarity features",
        "rows": len(rows),
        "queries": len(query_ids),
        "unique_candidate_blocks": len(block_ids),
        "feature_count": len(rows[0]["features"]),
        "decode_seconds": decode_seconds,
        "embedding_seconds": embedding_seconds,
        "feature_seconds": feature_seconds,
        "reader_forward_used": False,
        "future_target_used": False,
        "selection_uses_target": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
