from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 10M-token shared LongMemEval memory with owner/session/turn/time "
            "metadata and exact evidence spans."
        )
    )
    parser.add_argument("--longmemeval_s", required=True)
    parser.add_argument("--longmemeval_oracle", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=10_000_000)
    parser.add_argument("--block_tokens", type=int, default=64)
    parser.add_argument("--query_samples", type=int, default=64)
    parser.add_argument("--min_per_stratum", type=int, default=4)
    parser.add_argument("--partition_count", type=int, default=1)
    parser.add_argument("--partition_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list: {path}")
    return data


def encode(tokenizer: Any, text: str) -> np.ndarray:
    return np.asarray(
        tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int32
    )


def parse_date(value: str) -> int:
    parsed = dt.datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")
    return parsed.toordinal() * 1440 + parsed.hour * 60 + parsed.minute


def stratum(row: dict[str, Any]) -> tuple[str, bool]:
    return str(row["question_type"]), str(row["question_id"]).endswith("_abs")


def stratified_sample(
    rows: list[dict[str, Any]], *, samples: int, minimum: int, seed: int
) -> list[int]:
    if samples > len(rows):
        raise ValueError("query_samples exceeds dataset size")
    groups: dict[tuple[str, bool], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[stratum(row)].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[int] = []
    for key in sorted(groups):
        take = min(minimum, len(groups[key]))
        selected.extend(groups[key][:take])
        groups[key] = groups[key][take:]
    if len(selected) > samples:
        raise ValueError("query_samples is smaller than the requested stratum minimum")

    remaining = [index for values in groups.values() for index in values]
    rng.shuffle(remaining)
    selected.extend(remaining[: samples - len(selected)])
    rng.shuffle(selected)
    return selected


def estimated_history_size(row: dict[str, Any]) -> int:
    return sum(
        len(str(message.get("content", ""))) + len(str(message.get("role", ""))) + 4
        for session in row["haystack_sessions"]
        for message in session
    )


def balanced_stratified_partitions(
    rows: list[dict[str, Any]], *, partitions: int, seed: int
) -> list[list[int]]:
    if partitions <= 0:
        raise ValueError("partition_count must be positive")
    groups: dict[tuple[str, bool], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[stratum(row)].append(index)
    rng = random.Random(seed)
    output = [[] for _ in range(partitions)]
    estimated_sizes = [0 for _ in range(partitions)]
    stratum_counts = [collections.Counter() for _ in range(partitions)]
    for key in sorted(groups):
        values = groups[key]
        rng.shuffle(values)
        values.sort(key=lambda index: estimated_history_size(rows[index]), reverse=True)
        for index in values:
            shard = min(
                range(partitions),
                key=lambda item: (
                    stratum_counts[item][key],
                    estimated_sizes[item],
                    len(output[item]),
                    item,
                ),
            )
            output[shard].append(index)
            stratum_counts[shard][key] += 1
            estimated_sizes[shard] += estimated_history_size(rows[index])
    for values in output:
        values.sort()
    return output


def mode_or_negative(values: np.ndarray) -> int:
    values = values[values >= 0]
    if not len(values):
        return -1
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def blocks_overlapping(
    intervals: list[tuple[int, int]], block_tokens: int
) -> list[int]:
    blocks: set[int] = set()
    for start, end in intervals:
        if end <= start:
            continue
        blocks.update(range(start // block_tokens, (end - 1) // block_tokens + 1))
    return sorted(blocks)


def main() -> None:
    args = parse_args()
    if args.memory_tokens % args.block_tokens:
        raise ValueError("memory_tokens must be divisible by block_tokens")
    if args.query_samples <= 0 or args.min_per_stratum <= 0:
        raise ValueError("query_samples and min_per_stratum must be positive")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")

    rows = read_json(args.longmemeval_s)
    oracle_rows = read_json(args.longmemeval_oracle)
    oracle_by_id = {str(row["question_id"]): row for row in oracle_rows}
    row_ids = {str(row["question_id"]) for row in rows}
    if row_ids != set(oracle_by_id):
        raise ValueError("LongMemEval S and oracle question IDs do not match")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    if args.partition_count == 1:
        selected_owner_rows = stratified_sample(
            rows,
            samples=args.query_samples,
            minimum=args.min_per_stratum,
            seed=args.seed,
        )
    else:
        selected_owner_rows = balanced_stratified_partitions(
            rows, partitions=args.partition_count, seed=args.seed
        )[args.partition_index]
    selected_set = set(selected_owner_rows)
    rng = random.Random(args.seed)

    # Session content is stable across occurrences; timestamps are occurrence-specific.
    content_cache: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}

    def content_tokens(
        session_id: str, messages: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        cached = content_cache.get(session_id)
        if cached is not None:
            return cached
        token_parts = []
        turn_parts = []
        spans = []
        cursor = 0
        for turn_id, message in enumerate(messages):
            role = str(message.get("role", "unknown")).strip().title()
            text = f"{role}: {str(message.get('content', ''))}\n"
            tokens = encode(tokenizer, text)
            token_parts.append(tokens)
            turn_parts.append(np.full(len(tokens), turn_id, dtype=np.int16))
            spans.append((cursor, cursor + len(tokens)))
            cursor += len(tokens)
        body = np.concatenate(token_parts) if token_parts else np.empty(0, np.int32)
        turns = np.concatenate(turn_parts) if turn_parts else np.empty(0, np.int16)
        cached = body, turns, spans
        content_cache[session_id] = cached
        return cached

    occurrence_cache: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int]
    ] = {}

    def occurrence_tokens(
        session_id: str, date: str, messages: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int]:
        key = session_id, date
        cached = occurrence_cache.get(key)
        if cached is not None:
            return cached
        header = encode(tokenizer, f"Session date: {date}\n")
        body, body_turns, body_spans = content_tokens(session_id, messages)
        footer = encode(tokenizer, "\nEnd of session.\n\n")
        tokens = np.concatenate([header, body, footer])
        turns = np.concatenate(
            [
                np.full(len(header), -1, dtype=np.int16),
                body_turns,
                np.full(len(footer), -1, dtype=np.int16),
            ]
        )
        spans = [(start + len(header), end + len(header)) for start, end in body_spans]
        cached = tokens, turns, spans, parse_date(date)
        occurrence_cache[key] = cached
        return cached

    def owner_length(owner_row: int) -> int:
        row = rows[owner_row]
        total = 0
        for session_id, date, messages in zip(
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
        ):
            total += len(occurrence_tokens(str(session_id), str(date), messages)[0])
        return total

    selected_lengths = {owner: owner_length(owner) for owner in selected_owner_rows}
    selected_tokens = sum(selected_lengths.values())
    if selected_tokens >= args.memory_tokens:
        raise RuntimeError(
            f"selected histories require {selected_tokens:,} tokens, exceeding memory; "
            "reduce query_samples"
        )

    distractor_rows = [index for index in range(len(rows)) if index not in selected_set]
    rng.shuffle(distractor_rows)
    selected_order = list(selected_owner_rows)
    rng.shuffle(selected_order)
    owner_order = selected_order + distractor_rows

    base = np.empty(args.memory_tokens, dtype=np.int32)
    token_owner_ids = np.full(args.memory_tokens, -1, dtype=np.int16)
    token_session_rows = np.full(args.memory_tokens, -1, dtype=np.int32)
    token_turn_ids = np.full(args.memory_tokens, -1, dtype=np.int16)
    token_date_minutes = np.full(args.memory_tokens, -1, dtype=np.int32)
    session_manifest: list[dict[str, Any]] = []
    owner_manifest: list[dict[str, Any]] = []
    occurrence_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    cursor = 0
    complete_selected_owners: set[int] = set()
    complete_distractor_owners: set[int] = set()

    for owner_row in owner_order:
        if cursor >= args.memory_tokens:
            break
        row = rows[owner_row]
        owner_start = cursor
        owner_complete = True
        for session_index, (session_id, date, messages) in enumerate(
            zip(
                row["haystack_session_ids"],
                row["haystack_dates"],
                row["haystack_sessions"],
            )
        ):
            if cursor >= args.memory_tokens:
                owner_complete = False
                break
            session_id = str(session_id)
            date = str(date)
            tokens, turns, turn_spans, date_minutes = occurrence_tokens(
                session_id, date, messages
            )
            take = min(len(tokens), args.memory_tokens - cursor)
            start = cursor
            end = start + take
            session_row = len(session_manifest)
            base[start:end] = tokens[:take]
            token_owner_ids[start:end] = owner_row
            token_session_rows[start:end] = session_row
            token_turn_ids[start:end] = turns[:take]
            token_date_minutes[start:end] = date_minutes
            clipped_turn_spans = [
                (start + left, min(start + right, end))
                for left, right in turn_spans
                if start + left < end
            ]
            manifest_row = {
                "session_row": session_row,
                "owner_row": owner_row,
                "owner_question_id": str(row["question_id"]),
                "session_index": session_index,
                "session_id": session_id,
                "date": date,
                "date_minutes": date_minutes,
                "start_token": start,
                "end_token": end,
                "turn_spans": clipped_turn_spans,
                "turns": len(messages),
                "truncated": take < len(tokens),
            }
            session_manifest.append(manifest_row)
            occurrence_lookup[(owner_row, session_id)] = manifest_row
            cursor = end
            if take < len(tokens):
                owner_complete = False
                break
        owner_manifest.append(
            {
                "owner_row": owner_row,
                "owner_question_id": str(row["question_id"]),
                "selected_for_evaluation": owner_row in selected_set,
                "start_token": owner_start,
                "end_token": cursor,
                "complete": owner_complete,
                "sessions_written": sum(
                    int(item["owner_row"] == owner_row) for item in session_manifest
                ),
            }
        )
        if owner_complete:
            if owner_row in selected_set:
                complete_selected_owners.add(owner_row)
            else:
                complete_distractor_owners.add(owner_row)

    if cursor != args.memory_tokens:
        raise RuntimeError(f"constructed {cursor:,} tokens, need {args.memory_tokens:,}")
    if complete_selected_owners != selected_set:
        missing = sorted(selected_set - complete_selected_owners)
        raise RuntimeError(f"selected histories were truncated or omitted: {missing}")

    block_tokens = args.block_tokens
    block_owner_ids = np.empty(args.memory_tokens // block_tokens, dtype=np.int16)
    block_session_rows = np.empty(args.memory_tokens // block_tokens, dtype=np.int32)
    block_date_minutes = np.empty(args.memory_tokens // block_tokens, dtype=np.int32)
    mixed_owner_blocks = 0
    mixed_session_blocks = 0
    for block_id in range(len(block_owner_ids)):
        start = block_id * block_tokens
        end = start + block_tokens
        owners = token_owner_ids[start:end]
        sessions = token_session_rows[start:end]
        block_owner_ids[block_id] = mode_or_negative(owners)
        block_session_rows[block_id] = mode_or_negative(sessions)
        valid_dates = token_date_minutes[start:end]
        valid_dates = valid_dates[valid_dates >= 0]
        block_date_minutes[block_id] = (
            int(np.median(valid_dates)) if len(valid_dates) else -1
        )
        mixed_owner_blocks += int(len(np.unique(owners[owners >= 0])) > 1)
        mixed_session_blocks += int(len(np.unique(sessions[sessions >= 0])) > 1)

    query_rows = []
    missing_evidence = []
    for query_id, owner_row in enumerate(selected_owner_rows):
        row = rows[owner_row]
        question_id = str(row["question_id"])
        is_abstention = question_id.endswith("_abs")
        answer_session_ids = [str(item) for item in row["answer_session_ids"]]
        answer_sessions = [
            occurrence_lookup.get((owner_row, session_id))
            for session_id in answer_session_ids
        ]
        if any(item is None for item in answer_sessions):
            missing_evidence.append(question_id)
            continue
        intervals = []
        intervals_by_session: dict[int, list[tuple[int, int]]] = {}
        answer_session_rows = []
        fallback_sessions = 0
        for session_id, session_row in zip(answer_session_ids, answer_sessions):
            assert session_row is not None
            answer_session_rows.append(int(session_row["session_row"]))
            original_index = row["haystack_session_ids"].index(session_id)
            messages = row["haystack_sessions"][original_index]
            flagged_turns = [
                index for index, message in enumerate(messages) if message.get("has_answer")
            ]
            if flagged_turns:
                for turn_id in flagged_turns:
                    if turn_id < len(session_row["turn_spans"]):
                        interval = tuple(session_row["turn_spans"][turn_id])
                        intervals.append(interval)
                        intervals_by_session.setdefault(
                            int(session_row["session_row"]), []
                        ).append(interval)
            else:
                fallback_sessions += 1
                interval = (
                    int(session_row["start_token"]),
                    int(session_row["end_token"]),
                )
                intervals.append(interval)
                intervals_by_session.setdefault(
                    int(session_row["session_row"]), []
                ).append(interval)
        exact_blocks = blocks_overlapping(intervals, block_tokens)
        latest_date = max(
            int(session_manifest[session_row]["date_minutes"])
            for session_row in answer_session_rows
        )
        latest_answer_session_rows = [
            session_row
            for session_row in answer_session_rows
            if int(session_manifest[session_row]["date_minutes"]) == latest_date
        ]
        latest_intervals = [
            interval
            for session_row in latest_answer_session_rows
            for interval in intervals_by_session[session_row]
        ]
        latest_exact_blocks = blocks_overlapping(latest_intervals, block_tokens)
        positive_blocks = [] if is_abstention else exact_blocks
        hard_negative_blocks = exact_blocks if is_abstention else []
        positive_session_rows = [] if is_abstention else answer_session_rows
        latest_positive_session_rows = (
            [] if is_abstention else latest_answer_session_rows
        )
        latest_positive_block_ids = [] if is_abstention else latest_exact_blocks
        hard_negative_session_rows = answer_session_rows if is_abstention else []
        oracle = oracle_by_id[question_id]
        if set(map(str, oracle["answer_session_ids"])) != set(answer_session_ids):
            raise RuntimeError(f"oracle answer sessions disagree for {question_id}")
        query_rows.append(
            {
                "query_id": query_id,
                "owner_row": owner_row,
                "question_id": question_id,
                "question_type": str(row["question_type"]),
                "is_abstention": is_abstention,
                "question": str(row["question"]),
                "question_date": str(row["question_date"]),
                "question_date_minutes": parse_date(str(row["question_date"])),
                "answer": str(row["answer"]),
                "answer_session_ids": answer_session_ids,
                "positive_session_rows": positive_session_rows,
                "latest_positive_session_rows": latest_positive_session_rows,
                "hard_negative_session_rows": hard_negative_session_rows,
                "positive_block_ids": positive_blocks,
                "latest_positive_block_ids": latest_positive_block_ids,
                "hard_negative_block_ids": hard_negative_blocks,
                "evidence_intervals": intervals,
                "evidence_fallback_sessions": fallback_sessions,
                "history_sessions": len(row["haystack_sessions"]),
                "history_fully_in_memory": True,
                "selection_uses_answer": False,
            }
        )
    if missing_evidence:
        raise RuntimeError(f"answer sessions missing from memory: {missing_evidence}")

    np.save(output_dir / "base_blocks.npy", base.reshape(-1, block_tokens))
    np.save(output_dir / "base_block_owner_ids.npy", block_owner_ids)
    np.save(output_dir / "base_block_session_rows.npy", block_session_rows)
    np.save(output_dir / "base_block_date_minutes.npy", block_date_minutes)
    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in query_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "session_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in session_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "owner_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in owner_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    non_abstention = [row for row in query_rows if not row["is_abstention"]]
    summary = {
        "source": "LongMemEval-S shared multi-owner 10M memory",
        "raw_dataset": args.longmemeval_s,
        "oracle_dataset": args.longmemeval_oracle,
        "memory_tokens": args.memory_tokens,
        "memory_blocks": len(block_owner_ids),
        "block_tokens": block_tokens,
        "query_samples": len(query_rows),
        "non_abstention_queries": len(non_abstention),
        "abstention_queries": len(query_rows) - len(non_abstention),
        "question_types": dict(collections.Counter(row["question_type"] for row in query_rows)),
        "selected_history_tokens": selected_tokens,
        "owners_in_memory": len(owner_manifest),
        "complete_distractor_owners": len(complete_distractor_owners),
        "sessions_in_memory": len(session_manifest),
        "selected_histories_fully_in_memory": True,
        "all_answer_sessions_present": True,
        "mean_positive_sessions_non_abstention": float(
            np.mean([len(row["positive_session_rows"]) for row in non_abstention])
        ),
        "mean_exact_positive_blocks_non_abstention": float(
            np.mean([len(row["positive_block_ids"]) for row in non_abstention])
        ),
        "mixed_owner_blocks": mixed_owner_blocks,
        "mixed_owner_block_rate": mixed_owner_blocks / len(block_owner_ids),
        "mixed_session_blocks": mixed_session_blocks,
        "mixed_session_block_rate": mixed_session_blocks / len(block_owner_ids),
        "scope_hierarchy": ["owner", "session", "turn", "timestamp", "block"],
        "contains_real_sharegpt_ultrachat_fillers": True,
        "contains_simulated_memory_sessions": True,
        "fully_natural_corpus": False,
        "selection_uses_answer": False,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "partition_covers_full_dataset": args.partition_count > 1,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
