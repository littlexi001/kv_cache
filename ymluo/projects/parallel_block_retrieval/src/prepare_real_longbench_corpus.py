from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


DEFAULT_DATASETS = (
    "hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en,qmsum,gov_report"
)
DEFAULT_QUERY_DATASETS = (
    "hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en"
)
EMBEDDED_QA_MARKER = re.compile(r"(?m)^(Passage|Question|Answer):[ \t]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real 10M-token block corpus and answer-block labels from LongBench."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--longbench_dir", required=True)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--query_datasets", default=DEFAULT_QUERY_DATASETS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_tokens", type=int, default=10_000_000)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--num_queries", type=int, default=64)
    parser.add_argument("--max_blocks_per_record", type=int, default=128)
    parser.add_argument(
        "--exclude_queries_jsonl",
        action="append",
        default=[],
        help="Exclude query record_uids listed in one or more existing queries.jsonl files.",
    )
    parser.add_argument(
        "--allow_embedded_qa_templates",
        action="store_true",
        help=(
            "Allow records whose context or input contains standalone Passage/Question/Answer "
            "markers. Disabled by default because LongBench few-shot tasks use these as demos."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def answer_char_spans(context: str, answers: Iterable[str]) -> list[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue
        for match in re.finditer(re.escape(answer), context, flags=re.IGNORECASE):
            spans.add((match.start(), match.end()))
    return sorted(spans)


def embedded_qa_markers(text: str) -> list[str]:
    return [match.group(1) for match in EMBEDDED_QA_MARKER.finditer(text)]


def spans_to_block_ids(
    spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
    *,
    block_start: int,
    block_count: int,
    block_tokens: int,
) -> list[int]:
    if not spans or not offsets or block_count <= 0:
        return []
    starts = [int(item[0]) for item in offsets]
    ends = [int(item[1]) for item in offsets]
    written_tokens = block_count * block_tokens
    gold: set[int] = set()
    for char_start, char_end in spans:
        token_start = bisect.bisect_right(ends, char_start)
        token_end = bisect.bisect_left(starts, char_end)
        token_start = min(token_start, written_tokens)
        token_end = min(max(token_end, token_start + 1), written_tokens)
        if token_start >= written_tokens:
            continue
        first = token_start // block_tokens
        last = (token_end - 1) // block_tokens
        for local_block in range(first, min(last + 1, block_count)):
            gold.add(block_start + local_block)
    return sorted(gold)


def choose_queries(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[str(candidate["dataset"])].append(candidate)
    for rows in by_dataset.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    names = sorted(by_dataset)
    while len(selected) < count:
        made_progress = False
        for name in names:
            if by_dataset[name]:
                selected.append(by_dataset[name].pop())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    for query_id, row in enumerate(selected):
        row["query_id"] = query_id
    return selected


def main() -> None:
    args = parse_args()
    if args.seq_tokens < args.block_tokens:
        raise ValueError("seq_tokens must be at least block_tokens")
    if args.block_tokens <= 0 or args.max_blocks_per_record <= 0:
        raise ValueError("block sizes must be positive")

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    query_dataset_names = {
        item.strip() for item in args.query_datasets.split(",") if item.strip()
    }
    unknown_query_datasets = query_dataset_names - set(dataset_names)
    if unknown_query_datasets:
        raise ValueError(
            f"query_datasets must be a subset of datasets; unknown: {sorted(unknown_query_datasets)}"
        )
    longbench_dir = Path(args.longbench_dir)
    source_paths = {name: longbench_dir / f"{name}.jsonl" for name in dataset_names}
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LongBench files: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_blocks = args.seq_tokens // args.block_tokens
    actual_tokens = target_blocks * args.block_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000)

    blocks = np.lib.format.open_memmap(
        output_dir / "blocks.npy",
        mode="w+",
        dtype=np.int32,
        shape=(target_blocks, args.block_tokens),
    )

    handles = {name: source_paths[name].open("r", encoding="utf-8") for name in dataset_names}
    source_indices = Counter()
    accepted_records = Counter()
    accepted_corpus_only_records = Counter()
    accepted_template_records = Counter()
    accepted_template_markers = Counter()
    rejected_template_records = Counter()
    rejected_template_markers = Counter()
    query_candidates: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    block_cursor = 0

    block_meta_path = output_dir / "blocks.jsonl"
    record_meta_path = output_dir / "records.jsonl"
    with block_meta_path.open("w", encoding="utf-8") as block_file, record_meta_path.open(
        "w", encoding="utf-8"
    ) as record_file:
        active = set(dataset_names)
        while block_cursor < target_blocks and active:
            for dataset in dataset_names:
                if dataset not in active or block_cursor >= target_blocks:
                    continue
                line = handles[dataset].readline()
                if not line:
                    active.remove(dataset)
                    continue
                source_index = source_indices[dataset]
                source_indices[dataset] += 1
                row = json.loads(line)
                context = str(row.get("context", ""))
                question = str(row.get("input", "")).strip()
                answers = [str(item) for item in row.get("answers", []) if str(item).strip()]
                if not context:
                    continue
                markers = embedded_qa_markers(context) + embedded_qa_markers(question)
                if markers:
                    if not args.allow_embedded_qa_templates:
                        rejected_template_records[dataset] += 1
                        rejected_template_markers.update(markers)
                        continue
                    accepted_template_records[dataset] += 1
                    accepted_template_markers.update(markers)
                digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
                if digest in seen_contexts:
                    continue
                seen_contexts.add(digest)

                encoded = tokenizer(
                    context,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                )
                token_ids = encoded["input_ids"]
                offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
                available_blocks = len(token_ids) // args.block_tokens
                record_blocks = min(
                    available_blocks,
                    args.max_blocks_per_record,
                    target_blocks - block_cursor,
                )
                if record_blocks <= 0:
                    continue

                block_start = block_cursor
                written_tokens = record_blocks * args.block_tokens
                token_array = np.asarray(token_ids[:written_tokens], dtype=np.int32).reshape(
                    record_blocks, args.block_tokens
                )
                blocks[block_start : block_start + record_blocks] = token_array

                record_uid = str(row.get("_id") or f"{dataset}:{source_index}")
                spans = answer_char_spans(context, answers)
                gold_block_ids = spans_to_block_ids(
                    spans,
                    offsets,
                    block_start=block_start,
                    block_count=record_blocks,
                    block_tokens=args.block_tokens,
                )

                for local_block in range(record_blocks):
                    block_id = block_start + local_block
                    block_file.write(
                        json.dumps(
                            {
                                "block_id": block_id,
                                "dataset": dataset,
                                "record_uid": record_uid,
                                "source_index": source_index,
                                "block_in_record": local_block,
                                "token_start_in_record": local_block * args.block_tokens,
                                "token_end_in_record": (local_block + 1) * args.block_tokens,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                record_row = {
                    "dataset": dataset,
                    "record_uid": record_uid,
                    "source_file": str(source_paths[dataset]),
                    "source_index": source_index,
                    "block_start": block_start,
                    "block_count": record_blocks,
                    "source_token_count": len(token_ids),
                    "written_token_count": written_tokens,
                    "question": question,
                    "answers": answers,
                    "gold_block_ids": gold_block_ids,
                }
                record_file.write(json.dumps(record_row, ensure_ascii=False) + "\n")
                if dataset in query_dataset_names and gold_block_ids and answers and question:
                    query_candidates.append({**record_row, "context": context})
                if dataset not in query_dataset_names:
                    accepted_corpus_only_records[dataset] += 1

                block_cursor += record_blocks
                accepted_records[dataset] += 1
                if block_cursor % 1024 < record_blocks:
                    print(
                        json.dumps(
                            {
                                "blocks": block_cursor,
                                "target_blocks": target_blocks,
                                "tokens": block_cursor * args.block_tokens,
                                "query_candidates": len(query_candidates),
                            }
                        ),
                        flush=True,
                    )

    for handle in handles.values():
        handle.close()
    blocks.flush()
    if block_cursor != target_blocks:
        raise RuntimeError(f"Only built {block_cursor}/{target_blocks} blocks from the requested datasets")

    excluded_record_uids: set[str] = set()
    for raw_path in args.exclude_queries_jsonl:
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    excluded_record_uids.add(str(json.loads(line)["record_uid"]))
    eligible_query_candidates = [
        row
        for row in query_candidates
        if str(row["record_uid"]) not in excluded_record_uids
    ]
    queries = choose_queries(eligible_query_candidates, args.num_queries, args.seed)
    if len(queries) < args.num_queries:
        raise RuntimeError(f"Only found {len(queries)} answer-aligned queries; requested {args.num_queries}")
    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "LongBench real contexts",
        "model_tokenizer": args.model_name_or_path,
        "datasets": dataset_names,
        "query_datasets": sorted(query_dataset_names),
        "requested_seq_tokens": args.seq_tokens,
        "actual_seq_tokens": actual_tokens,
        "block_tokens": args.block_tokens,
        "num_blocks": target_blocks,
        "num_queries": len(queries),
        "query_candidates": len(query_candidates),
        "eligible_query_candidates": len(eligible_query_candidates),
        "excluded_query_record_uids": len(excluded_record_uids),
        "accepted_records_by_dataset": dict(accepted_records),
        "accepted_corpus_only_records_by_dataset": dict(accepted_corpus_only_records),
        "source_records_seen_by_dataset": dict(source_indices),
        "accepted_template_records_by_dataset": dict(accepted_template_records),
        "accepted_template_markers": dict(accepted_template_markers),
        "rejected_template_records_by_dataset": dict(rejected_template_records),
        "rejected_template_markers": dict(rejected_template_markers),
        "allows_embedded_qa_templates": args.allow_embedded_qa_templates,
        "contains_embedded_qa_templates": bool(accepted_template_records),
        "corpus_policy": (
            "unique real document contexts; standalone Passage/Question/Answer demo templates "
            "rejected; queries require an exact reference-answer occurrence"
            if not args.allow_embedded_qa_templates
            else "embedded QA templates allowed"
        ),
        "contains_synthetic_vectors": False,
        "gold_definition": "block overlaps a case-insensitive exact occurrence of a reference answer in its source context",
        "blocks_path": str(output_dir / "blocks.npy"),
        "blocks_metadata_path": str(block_meta_path),
        "records_metadata_path": str(record_meta_path),
        "queries_path": str(output_dir / "queries.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
