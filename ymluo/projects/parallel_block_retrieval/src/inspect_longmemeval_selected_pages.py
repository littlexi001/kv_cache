from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect decoded LongMemEval selected pages.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selection_rows", required=True)
    parser.add_argument("--question_id", required=True)
    parser.add_argument("--methods", default="static_top12,evidence_state_dynamic_top12")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    query = next(
        row
        for row in read_jsonl(data_dir / "queries.jsonl")
        if str(row["question_id"]) == args.question_id
    )
    selections = [
        row
        for row in read_jsonl(Path(args.selection_rows))
        if str(row["question_id"]) == args.question_id and str(row["method"]) in methods
    ]
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    block_dates = np.load(data_dir / "base_block_date_minutes.npy", mmap_mode="r")
    block_sessions = np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    output = {
        "query": query,
        "methods": [],
    }
    for selection in selections:
        pages = []
        for rank, block_id in enumerate(selection["top_block_ids"], start=1):
            block_id = int(block_id)
            pages.append(
                {
                    "rank": rank,
                    "block_id": block_id,
                    "session_row": int(block_sessions[block_id]),
                    "date_minutes": int(block_dates[block_id]),
                    "is_positive_block": block_id
                    in set(map(int, query["positive_block_ids"])),
                    "is_latest_positive_block": block_id
                    in set(map(int, query["latest_positive_block_ids"])),
                    "text": tokenizer.decode(
                        np.asarray(base_blocks[block_id], dtype=np.int64),
                        skip_special_tokens=True,
                    ),
                }
            )
        output["methods"].append({"method": selection["method"], "pages": pages})
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
