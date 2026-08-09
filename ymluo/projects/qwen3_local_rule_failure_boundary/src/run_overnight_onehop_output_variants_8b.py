from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_age_distractor_failure_boundary_8b as boundary
import run_fixed300_age_distractor_qk_8b as age
import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_twohop_age_distractor_failure_boundary_8b as twohop


START_TOTAL = 136 * 1024
END_TOTAL = 144 * 1024
MAX_DISTRACTORS = 4607

VARIANTS = (
    {
        "name": "period_with_distractors",
        "filler_text": ".",
        "distractor_count": MAX_DISTRACTORS,
        "stride": 8,
    },
    {
        "name": "comma_with_distractors",
        "filler_text": ",",
        "distractor_count": MAX_DISTRACTORS,
        "stride": 8,
    },
    {
        "name": "question_with_distractors",
        "filler_text": "?",
        "distractor_count": MAX_DISTRACTORS,
        "stride": 8,
    },
    {
        "name": "semicolon_with_distractors",
        "filler_text": ";",
        "distractor_count": MAX_DISTRACTORS,
        "stride": 8,
    },
    {
        "name": "period_without_distractors",
        "filler_text": ".",
        "distractor_count": 0,
        "stride": 1,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overnight one-hop output experiments: punctuation fillers, "
            "no-distractor tokenwise scan, and multi-token generation."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--existing-points-csv", required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=40960,
    )
    parser.add_argument("--fixed-rope-factor", type=float, default=4.0)
    parser.add_argument(
        "--fixed-max-position-embeddings",
        type=int,
        default=END_TOTAL,
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"prompt_ids", "category_positions"}
    }


def prepare_incremental_case(
    tokenizer: Any,
    distractor_pool: Sequence[dict[str, Any]],
    *,
    filler_text: str,
    distractor_count: int,
) -> dict[str, Any]:
    case = boundary.build_case(
        tokenizer,
        distractor_pool,
        END_TOTAL,
        distractor_count,
        filler_text=filler_text,
    )
    query_start, query_end = case["query_span"]
    query_ids = case["prompt_ids"][query_start:query_end]
    query_length = len(query_ids)
    base_body_length = START_TOTAL - query_length
    max_body = case["prompt_ids"][:query_start]
    if len(max_body) != END_TOTAL - query_length:
        raise AssertionError("unexpected maximum body length")
    base_body = max_body[:base_body_length]
    continuation = max_body[base_body_length:]
    if len(continuation) != END_TOTAL - START_TOTAL:
        raise AssertionError("unexpected continuation length")
    category_lookup = incremental.build_category_lookup(
        END_TOTAL,
        case["category_positions"],
    )
    continuation_categories = category_lookup[
        base_body_length : base_body_length + len(continuation)
    ]
    return {
        "case": case,
        "query_ids": query_ids,
        "query_length": query_length,
        "base_body_length": base_body_length,
        "base_body": base_body,
        "continuation": continuation,
        "continuation_categories": continuation_categories,
    }


def prefill_body(
    model: Any,
    body_ids: Sequence[int],
    chunk_size: int,
) -> tuple[Any, float]:
    prompt = torch.tensor(
        list(body_ids),
        dtype=torch.long,
    ).view(1, -1)
    legacy_cache, seconds = base.prefill_sequence(
        model,
        prompt,
        chunk_size,
    )
    del prompt
    return base.cache_from_legacy(legacy_cache), seconds


def checkpoint_variant(
    path: Path,
    *,
    name: str,
    filler_text: str,
    distractor_count: int,
    stride: int,
    rows: list[dict[str, Any]],
    prefill_seconds: float,
    started: float,
) -> None:
    competitor_counts = collections.Counter(
        f'{row["strongest_competitor_token_id"]}:'
        f'{row["strongest_competitor_token_label"]}'
        for row in rows
    )
    failures = sum(not bool(row["top_is_gold"]) for row in rows)
    write_json(
        path,
        {
            "schema_version": 1,
            "experiment": "onehop_output_variant",
            "name": name,
            "filler_text": filler_text,
            "distractor_count": distractor_count,
            "stride": stride,
            "start_total_tokens": START_TOTAL,
            "end_total_tokens": END_TOTAL,
            "single_shared_prefix_prefill": True,
            "prefill_count": 1,
            "prefill_seconds": first_token.rounded(prefill_seconds),
            "completed_points": len(rows),
            "failure_count": failures,
            "failure_rate": failures / len(rows) if rows else None,
            "strongest_competitor_counts": dict(
                competitor_counts.most_common()
            ),
            "last_point": rows[-1] if rows else None,
            "elapsed_seconds": first_token.rounded(
                time.perf_counter() - started
            ),
            "complete": bool(
                rows
                and rows[-1]["total_tokens"] == END_TOTAL
            ),
        },
    )


def run_variant(
    model: Any,
    tokenizer: Any,
    answer_variants: dict[str, list[int]],
    distractor_pool: Sequence[dict[str, Any]],
    attention_mask_buffer: torch.Tensor,
    output_root: Path,
    args: argparse.Namespace,
    variant: dict[str, Any],
) -> dict[str, Any]:
    name = str(variant["name"])
    filler_text = str(variant["filler_text"])
    distractor_count = int(variant["distractor_count"])
    stride = int(variant["stride"])
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_incremental_case(
        tokenizer,
        distractor_pool,
        filler_text=filler_text,
        distractor_count=distractor_count,
    )
    write_json(
        output_dir / "design.json",
        {
            "schema_version": 1,
            "experiment": "onehop_output_variant",
            "name": name,
            "filler_text": filler_text,
            "distractor_count": distractor_count,
            "stride": stride,
            "single_shared_prefix_prefill": True,
            "rope_scaling": {
                "type": "yarn",
                "factor": args.fixed_rope_factor,
                "original_max_position_embeddings": (
                    args.original_max_position_embeddings
                ),
                "max_position_embeddings": (
                    args.fixed_max_position_embeddings
                ),
            },
            "case": public_case(prepared["case"]),
            "base_body_tokens": prepared["base_body_length"],
            "query_tokens": prepared["query_length"],
            "continuation_tokens": len(prepared["continuation"]),
        },
    )

    started = time.perf_counter()
    cache, prefill_seconds = prefill_body(
        model,
        prepared["base_body"],
        args.prefill_chunk_size,
    )
    query_ids = torch.tensor(
        prepared["query_ids"],
        dtype=torch.long,
    ).view(1, -1)
    body_length = int(prepared["base_body_length"])
    rows: list[dict[str, Any]] = []
    csv_handle = (output_dir / "points.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    )
    writer: csv.DictWriter | None = None
    try:
        for added in range(len(prepared["continuation"]) + 1):
            if added > 0:
                next_id = int(prepared["continuation"][added - 1])
                with torch.inference_mode():
                    body_output = incremental.forward_with_shared_mask(
                        model,
                        torch.tensor([[next_id]], dtype=torch.long),
                        cache,
                        body_length,
                        attention_mask_buffer,
                        with_logits=False,
                    )
                cache = body_output.past_key_values
                body_length += 1
                del body_output

            if (
                added % stride != 0
                and added != len(prepared["continuation"])
            ):
                continue

            point_started = time.perf_counter()
            with torch.inference_mode():
                query_output = incremental.forward_with_shared_mask(
                    model,
                    query_ids,
                    cache,
                    body_length,
                    attention_mask_buffer,
                    with_logits=True,
                )
            score = first_token.score_first_token(
                tokenizer,
                query_output.logits[0, -1],
                answer_variants,
            )
            score.pop("top12_json", None)
            point = {
                "added_tokens": added,
                "total_tokens": START_TOTAL + added,
                "kib_tokens": first_token.rounded(
                    (START_TOTAL + added) / 1024,
                    6,
                ),
                "added_token_id": (
                    ""
                    if added == 0
                    else int(prepared["continuation"][added - 1])
                ),
                "added_token_text": (
                    ""
                    if added == 0
                    else token_text(
                        tokenizer,
                        int(prepared["continuation"][added - 1]),
                    )
                ),
                "added_token_category": (
                    "baseline"
                    if added == 0
                    else prepared["continuation_categories"][added - 1]
                ),
                **score,
                "point_seconds": first_token.rounded(
                    time.perf_counter() - point_started
                ),
            }
            if writer is None:
                writer = csv.DictWriter(
                    csv_handle,
                    fieldnames=list(point),
                )
                writer.writeheader()
            writer.writerow(point)
            csv_handle.flush()
            rows.append(point)
            cache = incremental.crop_cache(
                query_output.past_key_values,
                body_length,
            )
            del query_output

            if (
                len(rows) % args.checkpoint_every == 0
                or added == len(prepared["continuation"])
            ):
                checkpoint_variant(
                    output_dir / "manifest.json",
                    name=name,
                    filler_text=filler_text,
                    distractor_count=distractor_count,
                    stride=stride,
                    rows=rows,
                    prefill_seconds=prefill_seconds,
                    started=started,
                )
                print(
                    json.dumps(
                        {
                            "variant": name,
                            "added": added,
                            "points": len(rows),
                            "top": point["top_token_label"],
                            "competitor": point[
                                "strongest_competitor_token_label"
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    finally:
        csv_handle.close()

    checkpoint_variant(
        output_dir / "manifest.json",
        name=name,
        filler_text=filler_text,
        distractor_count=distractor_count,
        stride=stride,
        rows=rows,
        prefill_seconds=prefill_seconds,
        started=started,
    )
    summary = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    del cache, rows, prepared, query_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def quantile_sample(values: Sequence[int], count: int) -> list[int]:
    if not values:
        return []
    if len(values) <= count:
        return list(values)
    return [
        int(values[round(index * (len(values) - 1) / (count - 1))])
        for index in range(count)
    ]


def select_generation_points(
    path: Path,
    output_path: Path,
) -> list[int]:
    by_top: dict[int, list[int]] = collections.defaultdict(list)
    correct: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            added = int(row["added_tokens"])
            top_id = int(row["top_token_id"])
            is_correct = row["full_vocab_correct"].lower() == "true"
            if is_correct:
                correct.append(added)
            else:
                by_top[top_id].append(added)

    selected = {0, END_TOTAL - START_TOTAL}
    for token_id, values in by_top.items():
        count = 14 if token_id == 4710 else 8
        selected.update(quantile_sample(values, count))
    selected.update(quantile_sample(correct, 6))
    ordered = sorted(selected)
    write_json(
        output_path,
        {
            "source": str(path),
            "selected_added_tokens": ordered,
            "failed_top_token_counts": {
                str(token_id): len(values)
                for token_id, values in sorted(by_top.items())
            },
        },
    )
    return ordered


def generate_greedy(
    model: Any,
    tokenizer: Any,
    query_output: Any,
    prompt_tokens: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], Any]:
    cache = query_output.past_key_values
    logits = query_output.logits[:, -1, :]
    generated: list[int] = []
    past_length = prompt_tokens
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[0]).item())
        generated.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
        token = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=base.input_device(model),
        )
        with torch.inference_mode():
            output = base.forward_with_cache(
                model,
                token,
                cache,
                past_length,
            )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        past_length += 1
        del output
    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    matches = re.findall(
        r"\b(" + "|".join(age.NUMBER_WORDS) + r")\b",
        normalized,
    )
    return (
        {
            "token_ids": generated,
            "text": text,
            "normalized": normalized,
            "number_words": matches,
            "first_number_word": matches[0] if matches else None,
            "contains_nine": "nine" in matches,
            "first_answer_correct": bool(
                matches and matches[0] == "nine"
            ),
        },
        cache,
    )


def run_generation_probe(
    model: Any,
    tokenizer: Any,
    answer_variants: dict[str, list[int]],
    distractor_pool: Sequence[dict[str, Any]],
    attention_mask_buffer: torch.Tensor,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = output_root / "period_multitoken_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_generation_points(
        Path(args.existing_points_csv),
        output_dir / "selection.json",
    )
    selected_set = set(selected)
    prepared = prepare_incremental_case(
        tokenizer,
        distractor_pool,
        filler_text=".",
        distractor_count=MAX_DISTRACTORS,
    )
    started = time.perf_counter()
    cache, prefill_seconds = prefill_body(
        model,
        prepared["base_body"],
        args.prefill_chunk_size,
    )
    query_ids = torch.tensor(
        prepared["query_ids"],
        dtype=torch.long,
    ).view(1, -1)
    body_length = int(prepared["base_body_length"])
    rows: list[dict[str, Any]] = []

    for added in range(len(prepared["continuation"]) + 1):
        if added > 0:
            next_id = int(prepared["continuation"][added - 1])
            with torch.inference_mode():
                body_output = incremental.forward_with_shared_mask(
                    model,
                    torch.tensor([[next_id]], dtype=torch.long),
                    cache,
                    body_length,
                    attention_mask_buffer,
                    with_logits=False,
                )
            cache = body_output.past_key_values
            body_length += 1
            del body_output
        if added not in selected_set:
            continue

        with torch.inference_mode():
            query_output = incremental.forward_with_shared_mask(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                with_logits=True,
            )
        score = first_token.score_first_token(
            tokenizer,
            query_output.logits[0, -1],
            answer_variants,
        )
        score.pop("top12_json", None)
        generation, generated_cache = generate_greedy(
            model,
            tokenizer,
            query_output,
            START_TOTAL + added,
            args.generation_max_new_tokens,
        )
        cache = incremental.crop_cache(
            generated_cache,
            body_length,
        )
        rows.append(
            {
                "added_tokens": added,
                "total_tokens": START_TOTAL + added,
                **score,
                "generation": generation,
            }
        )
        write_json(
            output_dir / "results.json",
            {
                "schema_version": 1,
                "experiment": "period_multitoken_generation",
                "selected_count": len(selected),
                "completed_count": len(rows),
                "prefill_count": 1,
                "prefill_seconds": first_token.rounded(
                    prefill_seconds
                ),
                "max_new_tokens": args.generation_max_new_tokens,
                "rows": rows,
                "elapsed_seconds": first_token.rounded(
                    time.perf_counter() - started
                ),
                "complete": len(rows) == len(selected),
            },
        )
        print(
            json.dumps(
                {
                    "generation_added": added,
                    "top": score["top_token_label"],
                    "first_number": generation["first_number_word"],
                    "contains_nine": generation["contains_nine"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        del query_output

    result = json.loads(
        (output_dir / "results.json").read_text(encoding="utf-8")
    )
    del cache, prepared, query_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    if args.dry_run:
        designs = []
        for variant in VARIANTS:
            prepared = prepare_incremental_case(
                tokenizer,
                distractor_pool,
                filler_text=str(variant["filler_text"]),
                distractor_count=int(variant["distractor_count"]),
            )
            designs.append(
                {
                    **variant,
                    "filler_token_id": prepared["case"][
                        "filler_token_id"
                    ],
                    "base_body_tokens": prepared[
                        "base_body_length"
                    ],
                    "query_tokens": prepared["query_length"],
                    "continuation_tokens": len(
                        prepared["continuation"]
                    ),
                    "total_tokens_at_end": (
                        len(prepared["case"]["prompt_ids"])
                    ),
                }
            )
        write_json(
            output_root / "dry_run.json",
            {
                "schema_version": 1,
                "model_name_or_path": args.model_name_or_path,
                "variants": designs,
            },
        )
        print(json.dumps(designs, ensure_ascii=False, indent=2))
        return
    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("tokenizer changed while loading model")
    tokenizer = model_tokenizer
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=base.input_device(model),
    )

    summaries = []
    for variant in VARIANTS:
        summaries.append(
            run_variant(
                model,
                tokenizer,
                answer_variants,
                distractor_pool,
                attention_mask_buffer,
                output_root,
                args,
                variant,
            )
        )
    generation = run_generation_probe(
        model,
        tokenizer,
        answer_variants,
        distractor_pool,
        attention_mask_buffer,
        output_root,
        args,
    )
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "overnight_onehop_output_variants",
            "gpu_scope": "CUDA_VISIBLE_DEVICES=6,7",
            "rope_scaling": {
                "type": "yarn",
                "factor": args.fixed_rope_factor,
                "original_max_position_embeddings": (
                    args.original_max_position_embeddings
                ),
                "max_position_embeddings": (
                    args.fixed_max_position_embeddings
                ),
            },
            "variant_summaries": summaries,
            "generation_summary": {
                "selected_count": generation["selected_count"],
                "completed_count": generation["completed_count"],
                "complete": generation["complete"],
            },
            "complete": all(
                bool(summary["complete"]) for summary in summaries
            )
            and bool(generation["complete"]),
        },
    )


if __name__ == "__main__":
    main()
