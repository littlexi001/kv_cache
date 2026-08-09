from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace sparse real-filler blocks with controlled blocks in a 10M corpus."
    )
    parser.add_argument("--filler_corpus_dir", required=True)
    parser.add_argument("--controlled_corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_block_mapping(
    filler_blocks: int, controlled_blocks: int, seed: int
) -> dict[int, int]:
    if controlled_blocks <= 0 or controlled_blocks > filler_blocks:
        raise ValueError("controlled block count must be within the filler corpus")
    rng = random.Random(seed)
    destinations = rng.sample(range(filler_blocks), controlled_blocks)
    source_ids = list(range(controlled_blocks))
    rng.shuffle(source_ids)
    return {
        int(source_id): int(destination)
        for source_id, destination in zip(source_ids, destinations, strict=True)
    }


def main() -> None:
    args = parse_args()
    filler_dir = Path(args.filler_corpus_dir)
    controlled_dir = Path(args.controlled_corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filler = np.load(filler_dir / "blocks.npy", mmap_mode="r")
    controlled = np.load(controlled_dir / "blocks.npy", mmap_mode="r")
    if filler.ndim != 2 or controlled.ndim != 2:
        raise ValueError("both corpora must contain rank-2 block arrays")
    if int(filler.shape[1]) != int(controlled.shape[1]):
        raise ValueError("filler and controlled block sizes differ")
    mapping = build_block_mapping(int(filler.shape[0]), int(controlled.shape[0]), args.seed)

    output_blocks = np.lib.format.open_memmap(
        output_dir / "blocks.npy",
        mode="w+",
        dtype=filler.dtype,
        shape=filler.shape,
    )
    copy_chunk = 1_024
    for start in range(0, int(filler.shape[0]), copy_chunk):
        end = min(int(filler.shape[0]), start + copy_chunk)
        output_blocks[start:end] = filler[start:end]
    for source_id, destination in mapping.items():
        output_blocks[destination] = controlled[source_id]
    output_blocks.flush()

    filler_blocks_metadata = read_jsonl(filler_dir / "blocks.jsonl")
    controlled_blocks_metadata = {
        int(row["block_id"]): row for row in read_jsonl(controlled_dir / "blocks.jsonl")
    }
    destination_to_source = {destination: source for source, destination in mapping.items()}
    mixed_block_rows = []
    for row in filler_blocks_metadata:
        output = dict(row)
        block_id = int(row["block_id"])
        if block_id in destination_to_source:
            controlled_id = destination_to_source[block_id]
            controlled_row = controlled_blocks_metadata[controlled_id]
            output.update(
                {
                    "contains_controlled_text": True,
                    "controlled_block_id": controlled_id,
                    "controlled_query_ids": controlled_row.get("synthetic_query_ids", []),
                    "controlled_roles": controlled_row.get("synthetic_roles", []),
                }
            )
        mixed_block_rows.append(output)
    write_jsonl(output_dir / "blocks.jsonl", mixed_block_rows)

    records = read_jsonl(filler_dir / "records.jsonl")
    for record in records:
        record["source_file"] = (
            f"mixed:{filler_dir}/blocks.npy+{controlled_dir}/blocks.npy"
        )
    write_jsonl(output_dir / "records.jsonl", records)

    queries = read_jsonl(controlled_dir / "queries.jsonl")
    mixed_queries = []
    for query in queries:
        row = dict(query)
        row["controlled_gold_block_ids"] = [
            int(item) for item in query["gold_block_ids"]
        ]
        row["controlled_hard_negative_block_ids"] = [
            int(item) for item in query.get("hard_negative_block_ids", [])
        ]
        row["gold_block_ids"] = [mapping[int(item)] for item in query["gold_block_ids"]]
        row["hard_negative_block_ids"] = [
            mapping[int(item)] for item in query.get("hard_negative_block_ids", [])
        ]
        row["block_start"] = 0
        row["block_count"] = int(filler.shape[0])
        row["mixed_global_scope"] = True
        mixed_queries.append(row)
    write_jsonl(output_dir / "queries.jsonl", mixed_queries)

    mapping_rows = [
        {"controlled_block_id": source, "mixed_block_id": destination}
        for source, destination in sorted(mapping.items())
    ]
    write_jsonl(output_dir / "controlled_block_mapping.jsonl", mapping_rows)
    summary = {
        "source": "real 10M LongBench filler with sparse controlled-text block replacement",
        "contains_synthetic_vectors": False,
        "contains_controlled_synthetic_text": True,
        "selection_uses_mapping": False,
        "seed": args.seed,
        "filler_corpus_dir": str(filler_dir),
        "controlled_corpus_dir": str(controlled_dir),
        "num_blocks": int(filler.shape[0]),
        "block_tokens": int(filler.shape[1]),
        "num_tokens": int(filler.size),
        "controlled_blocks": int(controlled.shape[0]),
        "controlled_tokens": int(controlled.size),
        "controlled_token_fraction": float(controlled.size / filler.size),
        "real_filler_blocks": int(filler.shape[0] - controlled.shape[0]),
        "queries": len(mixed_queries),
        "records": len(records),
        "mapping_path": str(output_dir / "controlled_block_mapping.jsonl"),
        "blocks_path": str(output_dir / "blocks.npy"),
        "queries_path": str(output_dir / "queries.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
