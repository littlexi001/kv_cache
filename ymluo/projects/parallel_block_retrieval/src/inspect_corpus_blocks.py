from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode selected corpus block IDs.")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--block_ids", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_characters", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_ids = [int(item.strip()) for item in args.block_ids.split(",") if item.strip()]
    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    for block_id in block_ids:
        text = tokenizer.decode(blocks[block_id].tolist(), skip_special_tokens=True)
        print(
            json.dumps(
                {
                    "block_id": block_id,
                    "text": " ".join(text.split())[: args.max_characters],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
