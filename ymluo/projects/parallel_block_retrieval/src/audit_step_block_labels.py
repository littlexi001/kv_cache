from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from run_step_state_kv_span_retrieval import find_text_subsequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit that every declared step target span is contained by its target block."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    blocks = np.load(Path(args.corpus_dir) / "blocks.npy", mmap_mode="r")
    steps = read_jsonl(Path(args.step_queries_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    failures = []
    split_steps: Counter[str] = Counter()
    step_types: Counter[str] = Counter()
    target_blocks: dict[int, list[int]] = {}
    for step in steps:
        query_id = int(step["query_id"])
        step_index = int(step["step_index"])
        split_steps[str(step["split"])] += 1
        step_types[str(step["step_type"])] += 1
        target = int(step["target_block_ids"][0])
        target_blocks.setdefault(query_id, []).append(target)
        try:
            find_text_subsequence(
                blocks[target].tolist(), str(step["target_fact"]), tokenizer
            )
        except ValueError as error:
            failures.append(
                {
                    "query_id": query_id,
                    "step_index": step_index,
                    "target_block": target,
                    "target_fact": str(step["target_fact"]),
                    "error": str(error),
                }
            )
    payload = {
        "steps": len(steps),
        "target_span_contained": len(steps) - len(failures),
        "target_span_failures": len(failures),
        "split_steps": dict(split_steps),
        "step_types": dict(step_types),
        "same_target_block_two_hop_queries": sum(
            len(values) == 2 and values[0] == values[1]
            for values in target_blocks.values()
        ),
        "failures": failures,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
