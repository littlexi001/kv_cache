from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from transformers import AutoTokenizer

import run_controlled_public_kv_benchmark_v1 as lb
import run_sample_calibrated_longbench_20260717 as runner


DEFAULT_TASKS = ",".join(lb.LONG_BENCH_PROMPTS)


def flat_token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    values = encoded["input_ids"]
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def official_middle_ids(
    tokenizer: Any,
    example: lb.Example,
    max_prompt_tokens: int,
    prompt_wrapper: str,
) -> tuple[list[int], int, bool]:
    prompt = (
        example.prefix_template
        + example.context
        + example.suffix_template.format(input=example.query)
    )
    raw_ids = flat_token_ids(
        tokenizer,
        prompt,
        add_special_tokens=True,
    )
    working_prompt = prompt
    truncated = len(raw_ids) > max_prompt_tokens
    if truncated:
        half = max_prompt_tokens // 2
        working_prompt = tokenizer.decode(
            raw_ids[:half],
            skip_special_tokens=True,
        ) + tokenizer.decode(
            raw_ids[-half:],
            skip_special_tokens=True,
        )

    wrapped = not example.no_chat and prompt_wrapper != "none"
    if wrapped and prompt_wrapper == "llama3":
        working_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + working_prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        final_ids = flat_token_ids(
            tokenizer,
            working_prompt,
            add_special_tokens=False,
        )
    elif wrapped and prompt_wrapper == "qwen3":
        working_prompt = (
            "<|im_start|>user\n"
            + working_prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        final_ids = flat_token_ids(
            tokenizer,
            working_prompt,
            add_special_tokens=False,
        )
    elif truncated:
        final_ids = flat_token_ids(
            tokenizer,
            working_prompt,
            add_special_tokens=True,
        )
    else:
        final_ids = raw_ids
    return final_ids, len(raw_ids), truncated


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def common_suffix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_value, right_value in zip(reversed(left), reversed(right)):
        if left_value != right_value:
            break
        count += 1
    return count


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["ALL"] = rows
    for row in rows:
        groups[str(row["task"])].append(row)
    output = {}
    for task, items in sorted(groups.items()):
        output[task] = {
            "samples": len(items),
            "raw_over_limit": sum(
                int(row["raw_over_limit"]) for row in items
            ),
            "exact_token_match_rate": mean(
                float(row["exact_token_match"]) for row in items
            ),
            "current_length_mean": mean(
                float(row["current_tokens"]) for row in items
            ),
            "official_length_mean": mean(
                float(row["official_tokens"]) for row in items
            ),
            "official_over_limit_rate": mean(
                float(row["official_tokens"] > row["max_prompt_tokens"])
                for row in items
            ),
            "edge_agreement_ratio_mean": mean(
                float(row["edge_agreement_ratio"]) for row in items
            ),
            "common_suffix_tokens_mean": mean(
                float(row["common_suffix_tokens"]) for row in items
            ),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit token-level differences between the query-preserving "
            "LongBench prompt split and KVCache-Factory whole-prompt "
            "middle truncation."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_prompt_tokens", type=int, default=7500)
    parser.add_argument("--prompt_wrapper", default="llama3")
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    examples = runner.load_examples(
        SimpleNamespace(
            tasks=args.tasks,
            longbench_data_dir=args.longbench_data_dir,
            sample_offset_per_task=0,
            max_samples_per_task=args.max_samples_per_task,
            num_shards=1,
            shard_index=0,
            max_new_tokens_override=0,
        )
    )
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        current = runner.build_prompt_limited_bundle(
            tokenizer,
            example,
            args.max_prompt_tokens,
            args.prompt_wrapper,
        )
        current_ids = [
            int(value) for value in current.input_ids[0].tolist()
        ]
        official_ids, raw_tokens, raw_over_limit = official_middle_ids(
            tokenizer,
            example,
            args.max_prompt_tokens,
            args.prompt_wrapper,
        )
        common_prefix = common_prefix_length(
            current_ids,
            official_ids,
        )
        common_suffix = common_suffix_length(
            current_ids,
            official_ids,
        )
        denominator = max(1, max(len(current_ids), len(official_ids)))
        rows.append(
            {
                "task": example.task,
                "sample_id": example.sample_id,
                "max_prompt_tokens": args.max_prompt_tokens,
                "raw_tokens": raw_tokens,
                "raw_over_limit": int(raw_over_limit),
                "current_tokens": len(current_ids),
                "official_tokens": len(official_ids),
                "current_query_start": int(current.query_start),
                "current_suffix_tokens": int(current.suffix_token_count),
                "exact_token_match": int(current_ids == official_ids),
                "common_prefix_tokens": common_prefix,
                "common_suffix_tokens": common_suffix,
                "edge_agreement_ratio": min(
                    1.0,
                    (common_prefix + common_suffix) / denominator,
                ),
            }
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(examples),
                    }
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "model_name_or_path": args.model_name_or_path,
            "tasks": len({row["task"] for row in rows}),
            "samples": len(rows),
            "max_prompt_tokens": args.max_prompt_tokens,
            "prompt_wrapper": args.prompt_wrapper,
        },
        "results": aggregate(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
