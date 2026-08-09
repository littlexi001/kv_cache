from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-filtered LongBench-v2 code-repository continuation memory."
    )
    parser.add_argument("--longbench_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--query_samples", type=int, default=30)
    parser.add_argument("--query_offset_tokens", type=int, default=4096)
    parser.add_argument("--memory_scales", default="40000,1000000,10000000")
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--source_tokens", type=int, default=512)
    parser.add_argument("--query_tokens", type=int, default=64)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def parse_scales(spec: str) -> list[int]:
    scales = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not scales or min(scales) <= 0:
        raise ValueError("memory_scales must contain positive values")
    return scales


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.memory_scales)
    if args.source_tokens % args.block_tokens:
        raise ValueError("source_tokens must be block aligned")
    if any(scale % args.block_tokens for scale in scales):
        raise ValueError("every memory scale must be block aligned")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    separator = np.asarray(
        tokenizer("\n\n", add_special_tokens=False)["input_ids"], dtype=np.int32
    )
    raw = json.loads(Path(args.longbench_json).read_text(encoding="utf-8"))
    code_rows = [
        row for row in raw if str(row.get("domain")) == "Code Repository Understanding"
    ]
    del raw
    tokenized: list[dict[str, Any]] = []
    for repo_index, row in enumerate(code_rows):
        token_ids = np.asarray(
            tokenizer(str(row["context"]), add_special_tokens=False)["input_ids"],
            dtype=np.int32,
        )
        tokenized.append(
            {
                "repo_index": repo_index,
                "sample_id": str(row["_id"]),
                "original_question": str(row["question"]),
                "token_ids": token_ids,
            }
        )

    required = (
        args.query_offset_tokens
        + args.source_tokens
        + args.query_tokens
        + args.target_tokens
    )
    eligible = [repo for repo in tokenized if len(repo["token_ids"]) >= required]
    if len(eligible) < args.query_samples:
        raise RuntimeError(f"only {len(eligible)} repositories contain {required} tokens")
    query_repos = random.Random(args.seed).sample(eligible, args.query_samples)
    query_repo_ids = {int(repo["repo_index"]) for repo in query_repos}

    sources = []
    queries = []
    targets = []
    metadata = []
    reserved_end = required
    for query_id, repo in enumerate(query_repos):
        token_ids = repo["token_ids"]
        source_start = args.query_offset_tokens
        query_start = source_start + args.source_tokens
        target_start = query_start + args.query_tokens
        sources.append(token_ids[source_start:query_start])
        queries.append(token_ids[query_start:target_start])
        targets.append(token_ids[target_start : target_start + args.target_tokens])
        metadata.append(
            {
                "query_id": query_id,
                "repo_index": int(repo["repo_index"]),
                "sample_id": repo["sample_id"],
                "original_longbench_question": repo["original_question"],
                "source_start_token": source_start,
                "repo_tokens": len(token_ids),
                "selection_uses_target": False,
            }
        )

    segments: list[tuple[int, int, np.ndarray]] = []
    for repo in tokenized:
        repo_index = int(repo["repo_index"])
        token_ids = repo["token_ids"]
        if repo_index in query_repo_ids:
            before = token_ids[: args.query_offset_tokens]
            after = token_ids[reserved_end:]
            if len(before):
                segments.append((repo_index, 0, before))
            if len(after):
                segments.append((repo_index, 1, after))
        else:
            segments.append((repo_index, 0, token_ids))
    random.Random(args.seed + 1).shuffle(segments)

    max_base_tokens = max(scales) - args.source_tokens
    base = np.empty(max_base_tokens, dtype=np.int32)
    token_scope_ids = np.full(max_base_tokens, -1, dtype=np.int16)
    manifest = []
    cursor = 0
    for repo_index, part, segment in segments:
        if cursor >= max_base_tokens:
            break
        take = min(len(segment), max_base_tokens - cursor)
        base[cursor : cursor + take] = segment[:take]
        token_scope_ids[cursor : cursor + take] = repo_index
        manifest.append(
            {
                "repo_index": repo_index,
                "part": part,
                "start_token": cursor,
                "end_token": cursor + take,
            }
        )
        cursor += take
        if cursor >= max_base_tokens:
            break
        separator_take = min(len(separator), max_base_tokens - cursor)
        base[cursor : cursor + separator_take] = separator[:separator_take]
        cursor += separator_take
    if cursor != max_base_tokens:
        raise RuntimeError(
            f"leakage-filtered repositories provide {cursor} tokens, need {max_base_tokens}"
        )

    token_scope_matrix = token_scope_ids.reshape(-1, args.block_tokens)
    block_scope_ids = np.full(len(token_scope_matrix), -1, dtype=np.int16)
    mixed_scope_blocks = 0
    for block_id, values in enumerate(token_scope_matrix):
        valid = values[values >= 0]
        if not len(valid):
            continue
        unique, counts = np.unique(valid, return_counts=True)
        block_scope_ids[block_id] = unique[int(np.argmax(counts))]
        if len(unique) > 1:
            mixed_scope_blocks += 1

    np.save(output_dir / "base_blocks.npy", base.reshape(-1, args.block_tokens))
    np.save(output_dir / "base_block_scope_ids.npy", block_scope_ids)
    np.save(
        output_dir / "source_blocks.npy",
        np.stack(sources).reshape(args.query_samples, -1, args.block_tokens),
    )
    np.save(output_dir / "queries.npy", np.stack(queries))
    np.save(output_dir / "targets.npy", np.stack(targets))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "segment_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "real LongBench-v2 code repository continuation memory",
        "longbench_json": args.longbench_json,
        "repositories_total": len(tokenized),
        "query_repositories": args.query_samples,
        "query_repository_selection": "seeded random among eligible repositories",
        "query_distractor_repositories_disjoint": False,
        "reserved_source_query_target_spans_removed_from_base": True,
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
        "query_offset_tokens": args.query_offset_tokens,
        "scope_type": "repository_context",
        "base_scope_ids_file": "base_block_scope_ids.npy",
        "mixed_scope_blocks": mixed_scope_blocks,
        "mixed_scope_block_rate": mixed_scope_blocks / len(block_scope_ids),
        "virtual_memory_contract": (
            "nested unique repository-text base with every query's reserved source/query/target "
            "span removed, plus query-specific contiguous source; total tokens exactly equal the "
            "named scale"
        ),
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
