from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_fixed300_age_distractor_qk_8b as age
import run_local_rule_failure_boundary as base


PARTITION_ORDER = (
    "gold_other",
    "gold_age",
    "distractor_other",
    "distractor_ages",
    "irrelevant_periods",
    "query",
)

PARTITION_LABELS = {
    "gold_other": "Gold evidence sentence excluding nine",
    "gold_age": "Gold age token (nine)",
    "distractor_other": "Distractor sentences excluding their age tokens",
    "distractor_ages": "Distractor age tokens",
    "irrelevant_periods": "Irrelevant period tokens",
    "query": "Question and answer-instruction tokens",
}

NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Kate", "Liam", "Mary", "Noah", "Olivia", "Peter",
    "Quinn", "Rose", "Sarah", "Thomas", "Uma", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zoe", "Aaron", "Bella", "Caleb", "Diana", "Ethan", "Fiona",
    "Gavin", "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Mason", "Nora",
    "Owen", "Paula", "Ryan", "Sophia", "Tyler", "Violet", "Wyatt", "Amber",
    "Blake", "Chloe", "Dylan", "Ella", "Felix", "Georgia", "Hugo", "Isabel",
    "Jason", "Kara", "Leo", "Maya", "Neil", "Opal", "Parker", "Ruby",
    "Simon", "Tara", "Vera", "Will",
)

WRONG_AGES = ("one", "two", "three", "four", "five", "six", "seven", "eight", "ten")


def parse_points(value: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split(":")
        if len(fields) != 2:
            raise ValueError(f"point must be TOTAL_TOKENS:DISTRACTOR_COUNT, got {raw!r}")
        point = (int(fields[0]), int(fields[1]))
        if point[0] <= 0 or point[1] < 0:
            raise ValueError(f"invalid point: {point}")
        points.append(point)
    if not points:
        raise ValueError("at least one point is required")
    return list(dict.fromkeys(points))


def build_distractor_pool(tokenizer: Any) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for name_index, name in enumerate(NAMES):
        for age_index, age_word in enumerate(WRONG_AGES):
            # Rotate the age assignment so a prefix of the pool remains balanced.
            rotated_age = WRONG_AGES[(age_index + name_index) % len(WRONG_AGES)]
            text = f"{name}'s age is {rotated_age} years.\n"
            ids = age.token_ids(tokenizer, text)
            age_span = age.local_word_span(tokenizer, text, rotated_age)
            pool.append(
                {
                    "name": name,
                    "age": rotated_age,
                    "text": text,
                    "ids": ids,
                    "age_local_span": age_span,
                }
            )
    if not pool:
        raise RuntimeError("no valid distractor sentences")
    return pool


def append_span_positions(
    output: list[int],
    start: int,
    end: int,
    excluded: tuple[int, int] | None = None,
) -> None:
    for position in range(start, end):
        if excluded is None or not (excluded[0] <= position < excluded[1]):
            output.append(position)


def build_case(
    tokenizer: Any,
    distractor_pool: Sequence[dict[str, Any]],
    total_tokens: int,
    distractor_count: int,
    filler_text: str = ".",
) -> dict[str, Any]:
    period_id = age.one_token_id(
        tokenizer,
        filler_text,
        "irrelevant filler",
    )
    gold_ids = age.token_ids(tokenizer, age.GOLD_TEXT)
    query_ids = age.token_ids(tokenizer, age.QUERY_TEXT)
    gold_age_local = age.local_word_span(tokenizer, age.GOLD_TEXT, "nine")

    selected = [
        distractor_pool[index % len(distractor_pool)]
        for index in range(distractor_count)
    ]
    fixed_tokens = len(gold_ids) + len(query_ids) + sum(len(row["ids"]) for row in selected)
    filler_count = total_tokens - fixed_tokens
    if filler_count < 0:
        raise ValueError(
            f"{distractor_count} distractors require {fixed_tokens} tokens, "
            f"which does not fit total_tokens={total_tokens}"
        )
    gap_counts = age.spread_counts(filler_count, distractor_count + 1)

    prompt_ids: list[int] = []
    positions: dict[str, list[int]] = {category: [] for category in PARTITION_ORDER}
    distractor_spans: list[tuple[int, int]] = []
    distractor_age_spans: list[tuple[int, int]] = []

    gold_start = len(prompt_ids)
    prompt_ids.extend(gold_ids)
    gold_end = len(prompt_ids)
    gold_age_span = (
        gold_start + gold_age_local[0],
        gold_start + gold_age_local[1],
    )
    append_span_positions(positions["gold_other"], gold_start, gold_end, gold_age_span)
    append_span_positions(positions["gold_age"], *gold_age_span)

    def append_filler(count: int) -> None:
        start = len(prompt_ids)
        prompt_ids.extend([period_id] * count)
        append_span_positions(positions["irrelevant_periods"], start, start + count)

    append_filler(gap_counts[0])
    age_histogram: collections.Counter[str] = collections.Counter()
    for index, record in enumerate(selected):
        start = len(prompt_ids)
        prompt_ids.extend(record["ids"])
        end = len(prompt_ids)
        age_local = record["age_local_span"]
        age_span = (start + age_local[0], start + age_local[1])
        distractor_spans.append((start, end))
        distractor_age_spans.append(age_span)
        append_span_positions(positions["distractor_other"], start, end, age_span)
        append_span_positions(positions["distractor_ages"], *age_span)
        age_histogram[record["age"]] += 1
        append_filler(gap_counts[index + 1])

    query_start = len(prompt_ids)
    prompt_ids.extend(query_ids)
    query_span = (query_start, len(prompt_ids))
    append_span_positions(positions["query"], *query_span)

    if len(prompt_ids) != total_tokens:
        raise AssertionError(f"constructed {len(prompt_ids)} tokens, expected {total_tokens}")
    partition_count = sum(len(positions[category]) for category in PARTITION_ORDER)
    if partition_count != total_tokens:
        raise AssertionError(
            f"partition covers {partition_count} tokens, expected {total_tokens}"
        )
    seen = [
        position
        for category in PARTITION_ORDER
        for position in positions[category]
    ]
    if len(set(seen)) != total_tokens:
        raise AssertionError("partition categories overlap")

    digest = hashlib.sha256()
    for token_id in prompt_ids:
        digest.update(int(token_id).to_bytes(4, byteorder="little", signed=False))

    preview_count = min(8, distractor_count)
    preview_rows = []
    for index in list(range(preview_count)) + list(
        range(max(preview_count, distractor_count - preview_count), distractor_count)
    ):
        record = selected[index]
        preview_rows.append(
            {
                "index": index,
                "name": record["name"],
                "age": record["age"],
                "text": record["text"].rstrip(),
                "span": list(distractor_spans[index]),
                "age_span": list(distractor_age_spans[index]),
            }
        )

    return {
        "total_tokens": total_tokens,
        "distractor_count": distractor_count,
        "filler_text": filler_text,
        "filler_token_id": period_id,
        "prompt_ids": prompt_ids,
        "category_positions": positions,
        "category_counts": {
            category: len(positions[category])
            for category in PARTITION_ORDER
        },
        "filler_count": filler_count,
        "gap_count": len(gap_counts),
        "gap_min": min(gap_counts),
        "gap_max": max(gap_counts),
        "gap_mean": age.rounded(sum(gap_counts) / len(gap_counts)),
        "gold_span": [gold_start, gold_end],
        "gold_age_span": list(gold_age_span),
        "query_span": list(query_span),
        "age_histogram": dict(sorted(age_histogram.items())),
        "distractor_preview": preview_rows,
        "prompt_sha256": digest.hexdigest(),
        "prompt_prefix_text": tokenizer.decode(
            prompt_ids[: min(80, total_tokens)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "prompt_suffix_text": tokenizer.decode(
            prompt_ids[max(0, total_tokens - 100) :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"prompt_ids", "category_positions"}
    }


def point_filename(total_tokens: int, distractor_count: int) -> str:
    return f"tokens_{total_tokens:06d}_distractors_{distractor_count:05d}.json"


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    points: Sequence[tuple[int, int]],
    completed: Sequence[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    age.write_json_atomic(
        path,
        {
            "schema_version": 1,
            "experiment": "age_distractor_length_count_failure_boundary",
            "model_name_or_path": args.model_name_or_path,
            "gold_evidence": age.GOLD_TEXT.rstrip(),
            "query": age.QUERY_TEXT.strip(),
            "gold_answer": "nine",
            "partition_order": list(PARTITION_ORDER),
            "partition_labels": PARTITION_LABELS,
            "requested_points": [
                {"total_tokens": total, "distractor_count": count}
                for total, count in points
            ],
            "completed_count": len(completed),
            "completed": list(completed),
            "elapsed_seconds": age.rounded(elapsed_seconds),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Length × distractor-count failure-boundary scan for Qwen3-8B"
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--points",
        required=True,
        help="Comma-separated TOTAL_TOKENS:DISTRACTOR_COUNT points",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=32768)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Skip per-layer/head QK aggregation during coarse boundary search",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first full-vocabulary top-1 failure",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = parse_points(args.points)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    answer_token_ids = age.validate_answer_vocabulary(tokenizer)
    distractor_pool = build_distractor_pool(tokenizer)
    cases = [
        build_case(tokenizer, distractor_pool, total_tokens, distractor_count)
        for total_tokens, distractor_count in points
    ]
    design = {
        "schema_version": 1,
        "experiment": "age_distractor_length_count_failure_boundary",
        "gold_evidence": age.GOLD_TEXT.rstrip(),
        "query": age.QUERY_TEXT.strip(),
        "gold_answer": "nine",
        "answer_token_ids": answer_token_ids,
        "partition_order": list(PARTITION_ORDER),
        "partition_labels": PARTITION_LABELS,
        "distractor_pool_size": len(distractor_pool),
        "cases": [public_case(case) for case in cases],
    }
    age.write_json_atomic(output_dir / "design.json", design)
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return

    max_position = max(total for total, _ in points)
    max_factor = base.rope_factor_for_length(
        max_position,
        args.original_max_position_embeddings,
    )
    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        max_position,
        max_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("Tokenizer changed between validation and model loading")
    del tokenizer
    tokenizer = model_tokenizer

    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        total_tokens = case["total_tokens"]
        distractor_count = case["distractor_count"]
        file_name = point_filename(total_tokens, distractor_count)
        point_started = time.perf_counter()
        prompt = torch.tensor(case["prompt_ids"], dtype=torch.long).view(1, -1)
        base_cache, prefill_seconds = base.prefill_sequence(
            model,
            prompt[:, :-1],
            args.prefill_chunk_size,
        )
        query_cache = base.cache_from_legacy(base_cache)
        del base_cache
        if args.answer_only:
            base.synchronize()
            query_started = time.perf_counter()
            with torch.inference_mode():
                query_output = base.forward_with_cache(
                    model,
                    prompt[:, -1:].to(base.input_device(model)),
                    query_cache,
                    total_tokens - 1,
                )
            base.synchronize()
            query_seconds = time.perf_counter() - query_started
            captured_queries = None
        else:
            query_output, captured_queries, query_seconds = attention_runner.capture_query_states(
                model,
                query_cache,
                prompt[:, -1:],
                total_tokens - 1,
            )
        answer = age.score_answer(tokenizer, query_output, answer_token_ids)
        attention = (
            None
            if captured_queries is None
            else age.summarize_categories(
                model,
                query_output,
                captured_queries,
                case["category_positions"],
                category_order=PARTITION_ORDER,
            )
        )
        result = {
            "schema_version": 1,
            "experiment": "age_distractor_length_count_failure_boundary",
            "model_name_or_path": args.model_name_or_path,
            "case": public_case(case),
            "answer": answer,
            "attention": attention,
            "answer_only": bool(args.answer_only),
            "timing": {
                "prefill_seconds": age.rounded(prefill_seconds),
                "query_seconds": age.rounded(query_seconds),
                "point_seconds": age.rounded(time.perf_counter() - point_started),
            },
        }
        age.write_json_atomic(output_dir / file_name, result)
        row = {
            "total_tokens": total_tokens,
            "distractor_count": distractor_count,
            "distractor_density_tokens": age.rounded(
                case["category_counts"]["distractor_other"]
                + case["category_counts"]["distractor_ages"]
            ),
            "file": file_name,
            "gold_ppl": answer["gold_ppl"],
            "gold_probability": answer["gold_probability"],
            "full_vocab_margin": answer["full_vocab_margin"],
            "full_vocab_correct": answer["full_vocab_correct"],
            "candidate_margin": answer["candidate_margin"],
            "candidate_correct": answer["candidate_correct"],
            "candidate_prediction": answer["candidate_prediction"],
            "top_token": answer["top_token"],
            "point_seconds": result["timing"]["point_seconds"],
        }
        completed.append(row)
        write_manifest(
            output_dir / "manifest.json",
            args,
            points,
            completed,
            time.perf_counter() - started,
        )
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")), flush=True)
        del prompt, query_cache, query_output, captured_queries, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.stop_on_failure and not bool(answer["full_vocab_correct"]):
            break

    write_manifest(
        output_dir / "manifest.json",
        args,
        points,
        completed,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
