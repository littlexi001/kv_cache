from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from evaluate_stepwise_set_utility import build_span_query, select_evidence_span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark automatic within-block evidence span selection."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--exclude_query_ids", default="375")
    parser.add_argument("--timing_repeats", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_ids(*groups: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(item) for group in groups for item in group))


def scenario_blocks(step: dict[str, Any]) -> dict[str, list[int]]:
    target = [int(item) for item in step["target_block_ids"]]
    negative = [int(item) for item in step["hard_negative_block_ids"][:1]]
    previous = [int(item) for item in step["previous_evidence_block_ids"]]
    scenarios = {
        "target_block": target,
        "target_plus_negative": unique_ids(target, negative),
    }
    if previous:
        scenarios["target_plus_previous"] = unique_ids(target, previous)
    return scenarios


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if row["task_type"] == "multihop"
        and str(row["split"]) in allowed_splits
        and int(row["query_id"]) not in excluded_ids
    ]
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    decode_started = time.perf_counter()
    decoded_blocks = [
        tokenizer.decode(block.tolist(), skip_special_tokens=True) for block in blocks
    ]
    decode_seconds = time.perf_counter() - decode_started

    rows = []
    timings: dict[tuple[str, str], list[float]] = defaultdict(list)
    for step in steps:
        compact_state = [str(item) for item in step["compact_state_before"]]
        query_text = build_span_query(step, compact_state)
        for scenario, block_ids in scenario_blocks(step).items():
            memory = "\n\n".join(decoded_blocks[block_id] for block_id in block_ids)
            selected = select_evidence_span(memory, query_text)
            for _ in range(args.timing_repeats):
                started = time.perf_counter()
                select_evidence_span(memory, query_text)
                timings[(str(step["step_type"]), scenario)].append(
                    time.perf_counter() - started
                )
            rows.append(
                {
                    "query_id": int(step["query_id"]),
                    "split": str(step["split"]),
                    "step_type": str(step["step_type"]),
                    "scenario": scenario,
                    "candidate_blocks": len(block_ids),
                    "candidate_tokens": len(block_ids) * int(blocks.shape[1]),
                    "selected_tokens": len(
                        tokenizer(selected, add_special_tokens=False)["input_ids"]
                    ),
                    "target_fact_selected": str(step["target_fact"]).casefold()
                    in selected.casefold(),
                    "target_output_selected": str(step["target_output"]).casefold()
                    in selected.casefold(),
                    "selected_text": selected,
                }
            )

    summaries = []
    keys = sorted({(row["step_type"], row["scenario"]) for row in rows})
    for key in keys:
        group = [
            row for row in rows if (row["step_type"], row["scenario"]) == key
        ]
        elapsed = timings[key]
        summaries.append(
            {
                "step_type": key[0],
                "scenario": key[1],
                "steps": len(group),
                "target_fact_selection_rate": statistics.fmean(
                    row["target_fact_selected"] for row in group
                ),
                "target_output_selection_rate": statistics.fmean(
                    row["target_output_selected"] for row in group
                ),
                "mean_candidate_tokens": statistics.fmean(
                    row["candidate_tokens"] for row in group
                ),
                "mean_selected_tokens": statistics.fmean(
                    row["selected_tokens"] for row in group
                ),
                "mean_selector_microseconds": statistics.fmean(elapsed) * 1e6,
                "median_selector_microseconds": statistics.median(elapsed) * 1e6,
                "p95_selector_microseconds": float(np.quantile(elapsed, 0.95)) * 1e6,
            }
        )
    payload = {
        "source": "gold-containing candidate diagnostic; selector does not read gold",
        "contains_synthetic_vectors": False,
        "corpus_dir": str(corpus_dir),
        "splits": sorted(allowed_splits),
        "excluded_query_ids": sorted(excluded_ids),
        "steps": len(steps),
        "block_decode_seconds_one_time": decode_seconds,
        "timing_repeats": args.timing_repeats,
        "summaries": summaries,
        "rows": rows,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
