from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoConfig, AutoTokenizer


def parse_csv(spec: str) -> set[str]:
    return {item.strip() for item in spec.split(",") if item.strip()}


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def prompt_token_count(
    tokenizer: Any,
    row: dict[str, Any],
    prompt_wrapper: str,
    force_no_chat_tasks: set[str],
) -> int:
    prefix = str(row.get("prefix_template", ""))
    context = str(row["context"])
    suffix_template = str(row.get("suffix_template", ""))
    query = str(row.get("query", ""))
    suffix = suffix_template.format(input=query)
    task = str(row["task"])
    no_chat = bool(row.get("no_chat", False)) or task in force_no_chat_tasks

    if prompt_wrapper == "llama3" and not no_chat:
        prefix = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prefix
        )
        suffix += (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif prompt_wrapper == "qwen3" and not no_chat:
        prefix = "<|im_start|>user\n" + prefix
        suffix += "<|im_end|>\n<|im_start|>assistant\n"

    # The benchmark runner tokenizes these three regions independently.
    return sum(token_count(tokenizer, text) for text in (prefix, context, suffix))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield path, json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error


def summarize_lengths(
    rows: Iterable[tuple[Path, dict[str, Any]]],
    tokenizer: Any,
    prompt_wrapper: str,
    force_no_chat_tasks: set[str],
    max_sequence_tokens: int,
) -> tuple[dict[str, Any], int]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    for path, row in rows:
        prompt_tokens = prompt_token_count(
            tokenizer, row, prompt_wrapper, force_no_chat_tasks
        )
        max_new_tokens = int(row.get("max_new_tokens", 0))
        total_tokens = prompt_tokens + max_new_tokens
        grouped[int(row["length"])].append(
            {
                "task": str(row["task"]),
                "sample_id": str(row["sample_id"]),
                "source": str(path),
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "total_tokens": total_tokens,
            }
        )
        total_rows += 1

    per_length: dict[str, Any] = {}
    for length, items in sorted(grouped.items()):
        prompt_counts = [item["prompt_tokens"] for item in items]
        total_counts = [item["total_tokens"] for item in items]
        offenders = [
            item for item in items if item["total_tokens"] > max_sequence_tokens
        ]
        per_length[str(length)] = {
            "rows": len(items),
            "min_prompt_tokens": min(prompt_counts),
            "mean_prompt_tokens": sum(prompt_counts) / len(prompt_counts),
            "max_prompt_tokens": max(prompt_counts),
            "max_prompt_plus_generation_tokens": max(total_counts),
            "minimum_remaining_tokens": min(
                max_sequence_tokens - count for count in total_counts
            ),
            "overflow_rows": len(offenders),
            "overflow_examples": offenders[:10],
        }
    return per_length, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen RULER prompt lengths before GPU evaluation."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--examples_jsonl", required=True, type=Path, action="append"
    )
    parser.add_argument(
        "--prompt_wrapper", choices=("llama3", "qwen3", "none"), default="none"
    )
    parser.add_argument("--force_no_chat_tasks", default="")
    parser.add_argument("--expected_rows", type=int, default=0)
    parser.add_argument("--max_sequence_tokens", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for path in args.examples_jsonl:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, local_files_only=True, trust_remote_code=False
    )
    model_limit = int(getattr(config, "max_position_embeddings", 0))
    max_sequence_tokens = args.max_sequence_tokens or model_limit
    if max_sequence_tokens <= 0:
        raise RuntimeError("model/config does not define a positive sequence limit")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    per_length, total_rows = summarize_lengths(
        iter_rows(args.examples_jsonl),
        tokenizer,
        args.prompt_wrapper,
        parse_csv(args.force_no_chat_tasks),
        max_sequence_tokens,
    )
    overflow_rows = sum(int(item["overflow_rows"]) for item in per_length.values())
    expected_rows_ok = args.expected_rows <= 0 or total_rows == args.expected_rows
    report = {
        "schema": "qksieve_ruler_prompt_length_audit_v1",
        "model_name_or_path": str(args.model_name_or_path),
        "model_max_position_embeddings": model_limit,
        "audited_max_sequence_tokens": max_sequence_tokens,
        "prompt_wrapper": args.prompt_wrapper,
        "force_no_chat_tasks": sorted(parse_csv(args.force_no_chat_tasks)),
        "input_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in args.examples_jsonl
        ],
        "expected_rows": args.expected_rows,
        "observed_rows": total_rows,
        "expected_rows_ok": expected_rows_ok,
        "overflow_rows": overflow_rows,
        "all_within_model_limit": overflow_rows == 0,
        "per_requested_length": per_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))

    if not expected_rows_ok:
        raise RuntimeError(
            f"observed {total_rows} rows, expected {args.expected_rows}"
        )
    if overflow_rows:
        raise RuntimeError(
            f"{overflow_rows} RULER prompts exceed {max_sequence_tokens} tokens"
        )


if __name__ == "__main__":
    main()
