from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_age_distractor_failure_boundary_8b as boundary
import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_twohop_age_distractor_failure_boundary_8b as twohop


MAX_TOTAL = 136 * 1024
COARSE_STRIDE = 256
MEDIUM_STRIDE = 16
COARSE_WINDOW_TOKENS = 2048
MEDIUM_WINDOW_TOKENS = 512
EXACT_LEFT_CONTEXT = 2048
EXACT_RIGHT_CONTEXT = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-prefix staged search for the first frequent-failure "
            "boundary in the no-age-distractor, period-filler one-hop case."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
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
        default=144 * 1024,
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_case(
    tokenizer: Any,
) -> tuple[dict[str, Any], list[int], list[int]]:
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    case = boundary.build_case(
        tokenizer,
        distractor_pool,
        MAX_TOTAL,
        0,
        filler_text=".",
    )
    query_start, query_end = case["query_span"]
    query_ids = case["prompt_ids"][query_start:query_end]
    body_ids = case["prompt_ids"][:query_start]
    if len(body_ids) + len(query_ids) != MAX_TOTAL:
        raise AssertionError("unexpected prompt partition")
    return case, body_ids, query_ids


def build_cache(
    model: Any,
    body_ids: Sequence[int],
    chunk_size: int,
) -> tuple[Any, float]:
    started = time.perf_counter()
    prompt = torch.tensor(
        list(body_ids),
        dtype=torch.long,
    ).view(1, -1)
    legacy, _ = base.prefill_sequence(
        model,
        prompt,
        chunk_size,
    )
    del prompt
    return (
        base.cache_from_legacy(legacy),
        time.perf_counter() - started,
    )


def extend_cache(
    model: Any,
    cache: Any,
    body_ids: Sequence[int],
    current_length: int,
    target_length: int,
    attention_mask_buffer: torch.Tensor,
    chunk_size: int,
) -> tuple[Any, int]:
    while current_length < target_length:
        end = min(target_length, current_length + chunk_size)
        chunk = body_ids[current_length:end]
        with torch.inference_mode():
            output = incremental.forward_with_shared_mask(
                model,
                torch.tensor([chunk], dtype=torch.long),
                cache,
                current_length,
                attention_mask_buffer,
                with_logits=False,
            )
        cache = output.past_key_values
        current_length = end
        del output
    return cache, current_length


def score_at_length(
    model: Any,
    tokenizer: Any,
    answer_variants: dict[str, list[int]],
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    clear_allocator()
    started = time.perf_counter()
    with torch.inference_mode():
        output = incremental.forward_with_shared_mask(
            model,
            query_ids,
            cache,
            body_length,
            attention_mask_buffer,
            with_logits=True,
        )
    score = first_token.score_first_token(
        tokenizer,
        output.logits[0, -1],
        answer_variants,
    )
    score.pop("top12_json", None)
    cache = incremental.crop_cache(
        output.past_key_values,
        body_length,
    )
    del output
    score["point_seconds"] = first_token.rounded(
        time.perf_counter() - started
    )
    return cache, score


def checkpoint_totals(
    start_total: int,
    end_total: int,
    stride: int,
) -> list[int]:
    values = list(range(start_total, end_total + 1, stride))
    if values[-1] != end_total:
        values.append(end_total)
    return values


def first_majority_window(
    rows: Sequence[dict[str, Any]],
    window_points: int,
) -> dict[str, Any] | None:
    if len(rows) < window_points:
        return None
    failures = [not bool(row["top_is_gold"]) for row in rows]
    count = sum(failures[:window_points])
    for end_index in range(window_points - 1, len(rows)):
        if end_index >= window_points:
            count += int(failures[end_index])
            count -= int(failures[end_index - window_points])
        if count / window_points > 0.5:
            start_index = end_index - window_points + 1
            return {
                "start_total_tokens": int(
                    rows[start_index]["total_tokens"]
                ),
                "end_total_tokens": int(
                    rows[end_index]["total_tokens"]
                ),
                "failure_count": int(count),
                "window_points": int(window_points),
                "failure_rate": count / window_points,
            }
    return None


def run_stage(
    *,
    stage: str,
    model: Any,
    tokenizer: Any,
    answer_variants: dict[str, list[int]],
    body_ids: Sequence[int],
    query_ids: torch.Tensor,
    query_length: int,
    attention_mask_buffer: torch.Tensor,
    output_dir: Path,
    start_total: int,
    end_total: int,
    stride: int,
    prefill_chunk_size: int,
    extension_chunk_size: int,
    checkpoint_every: int,
) -> tuple[list[dict[str, Any]], float]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    start_body = start_total - query_length
    end_body = end_total - query_length
    if start_body < 1 or end_body > len(body_ids):
        raise ValueError("stage body range is invalid")

    cache, prefill_seconds = build_cache(
        model,
        body_ids[:start_body],
        prefill_chunk_size,
    )
    current_body = start_body
    totals = checkpoint_totals(start_total, end_total, stride)
    rows: list[dict[str, Any]] = []
    csv_path = stage_dir / "points.csv"
    started = time.perf_counter()
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer: csv.DictWriter | None = None
        for point_index, total in enumerate(totals):
            target_body = total - query_length
            cache, current_body = extend_cache(
                model,
                cache,
                body_ids,
                current_body,
                target_body,
                attention_mask_buffer,
                extension_chunk_size,
            )
            cache, score = score_at_length(
                model,
                tokenizer,
                answer_variants,
                query_ids,
                cache,
                current_body,
                attention_mask_buffer,
            )
            row = {
                "stage": stage,
                "total_tokens": total,
                "kib_tokens": first_token.rounded(
                    total / 1024,
                    6,
                ),
                **score,
            }
            if writer is None:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(row),
                )
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if (
                len(rows) % checkpoint_every == 0
                or point_index == len(totals) - 1
            ):
                write_json(
                    stage_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "start_total_tokens": start_total,
                        "end_total_tokens": end_total,
                        "stride": stride,
                        "prefill_chunk_size": prefill_chunk_size,
                        "extension_chunk_size": extension_chunk_size,
                        "single_shared_prefix": True,
                        "prefill_seconds": prefill_seconds,
                        "completed_points": len(rows),
                        "last_total_tokens": total,
                        "elapsed_seconds": (
                            time.perf_counter() - started
                        ),
                        "complete": (
                            point_index == len(totals) - 1
                        ),
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "total": total,
                            "points": len(rows),
                            "top": row["top_token_label"],
                            "margin": row[
                                "gold_exact_vs_competitor_margin"
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    del cache
    clear_allocator()
    return rows, prefill_seconds


def exact_boundaries(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [not bool(row["top_is_gold"]) for row in rows]
    first_failure = next(
        (
            int(row["total_tokens"])
            for row, failure in zip(rows, failures)
            if failure
        ),
        None,
    )

    def first_consecutive(count: int) -> int | None:
        run = 0
        for row, failure in zip(rows, failures):
            run = run + 1 if failure else 0
            if run >= count:
                return int(row["total_tokens"])
        return None

    return {
        "first_failure_total_tokens": first_failure,
        "first_5_consecutive_failure_end": first_consecutive(5),
        "first_64_token_window_majority_failure": (
            first_majority_window(rows, 64)
        ),
        "first_512_token_window_majority_failure": (
            first_majority_window(rows, 512)
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    case, body_ids, query_ids_list = prepare_case(tokenizer)
    query_length = len(query_ids_list)
    minimum_total = int(case["gold_span"][1]) + query_length
    if minimum_total > 64:
        raise RuntimeError("minimum prompt is unexpectedly long")
    minimum_total = max(32, minimum_total)
    if args.dry_run:
        write_json(
            output_dir / "dry_run.json",
            {
                "minimum_total": minimum_total,
                "maximum_total": MAX_TOTAL,
                "query_tokens": query_length,
                "body_tokens": len(body_ids),
                "coarse_stride": COARSE_STRIDE,
                "medium_stride": MEDIUM_STRIDE,
                "single_shared_prefix_per_stage": True,
                "distractor_count": 0,
                "filler_text": ".",
            },
        )
        return

    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("tokenizer changed while loading model")
    tokenizer = model_tokenizer
    query_ids = torch.tensor(
        query_ids_list,
        dtype=torch.long,
    ).view(1, -1)
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=base.input_device(model),
    )
    started = time.perf_counter()

    coarse_rows, coarse_prefill = run_stage(
        stage="coarse",
        model=model,
        tokenizer=tokenizer,
        answer_variants=answer_variants,
        body_ids=body_ids,
        query_ids=query_ids,
        query_length=query_length,
        attention_mask_buffer=attention_mask_buffer,
        output_dir=output_dir,
        start_total=minimum_total,
        end_total=MAX_TOTAL,
        stride=COARSE_STRIDE,
        prefill_chunk_size=args.prefill_chunk_size,
        extension_chunk_size=args.prefill_chunk_size,
        checkpoint_every=args.checkpoint_every,
    )
    coarse_window_points = max(
        2,
        COARSE_WINDOW_TOKENS // COARSE_STRIDE,
    )
    coarse_boundary = first_majority_window(
        coarse_rows,
        coarse_window_points,
    )
    if coarse_boundary is None:
        raise RuntimeError("coarse scan found no frequent-failure region")

    medium_center = int(coarse_boundary["end_total_tokens"])
    medium_start = max(minimum_total, medium_center - 4096)
    medium_end = min(MAX_TOTAL, medium_center + 2048)
    medium_rows, medium_prefill = run_stage(
        stage="medium",
        model=model,
        tokenizer=tokenizer,
        answer_variants=answer_variants,
        body_ids=body_ids,
        query_ids=query_ids,
        query_length=query_length,
        attention_mask_buffer=attention_mask_buffer,
        output_dir=output_dir,
        start_total=medium_start,
        end_total=medium_end,
        stride=MEDIUM_STRIDE,
        prefill_chunk_size=args.prefill_chunk_size,
        extension_chunk_size=1,
        checkpoint_every=args.checkpoint_every,
    )
    medium_window_points = max(
        2,
        MEDIUM_WINDOW_TOKENS // MEDIUM_STRIDE,
    )
    medium_boundary = first_majority_window(
        medium_rows,
        medium_window_points,
    )
    if medium_boundary is None:
        raise RuntimeError("medium scan found no frequent-failure region")

    exact_center = int(medium_boundary["end_total_tokens"])
    exact_start = max(
        minimum_total,
        exact_center - EXACT_LEFT_CONTEXT,
    )
    exact_end = min(
        MAX_TOTAL,
        exact_center + EXACT_RIGHT_CONTEXT,
    )
    exact_rows, exact_prefill = run_stage(
        stage="exact",
        model=model,
        tokenizer=tokenizer,
        answer_variants=answer_variants,
        body_ids=body_ids,
        query_ids=query_ids,
        query_length=query_length,
        attention_mask_buffer=attention_mask_buffer,
        output_dir=output_dir,
        start_total=exact_start,
        end_total=exact_end,
        stride=1,
        prefill_chunk_size=args.prefill_chunk_size,
        extension_chunk_size=1,
        checkpoint_every=args.checkpoint_every,
    )
    boundaries = exact_boundaries(exact_rows)
    exact_window = boundaries[
        "first_512_token_window_majority_failure"
    ]
    if (
        exact_window is not None
        and exact_window["start_total_tokens"] == exact_start
    ):
        boundaries["left_truncated_warning"] = True
    else:
        boundaries["left_truncated_warning"] = False

    summary = {
        "schema_version": 1,
        "experiment": "no_distractor_failure_boundary_search",
        "model": args.model_name_or_path,
        "distractor_count": 0,
        "filler_text": ".",
        "minimum_total": minimum_total,
        "maximum_total": MAX_TOTAL,
        "query_tokens": query_length,
        "single_shared_prefix_per_stage": True,
        "coarse": {
            "stride": COARSE_STRIDE,
            "points": len(coarse_rows),
            "prefill_seconds": coarse_prefill,
            "boundary": coarse_boundary,
        },
        "medium": {
            "start_total": medium_start,
            "end_total": medium_end,
            "stride": MEDIUM_STRIDE,
            "points": len(medium_rows),
            "prefill_seconds": medium_prefill,
            "boundary": medium_boundary,
        },
        "exact": {
            "start_total": exact_start,
            "end_total": exact_end,
            "stride": 1,
            "points": len(exact_rows),
            "prefill_seconds": exact_prefill,
            "boundaries": boundaries,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            summary["exact"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
