from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_age_distractor_failure_boundary_8b as boundary
import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_no_distractor_failure_boundary_search_8b as one_token
import run_twohop_age_distractor_failure_boundary_8b as twohop


COARSE_STRIDE = 256
MEDIUM_STRIDE = 16
COARSE_WINDOW_TOKENS = 2048
SEMANTIC_WINDOW_TOKENS = 512
MEDIUM_CONTEXT_TOKENS = 2048
EXACT_CONTEXT_TOKENS = 512
NUMBER_PATTERN = re.compile(
    r"\b("
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|[0-9]+"
    r")\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Staged shared-prefix search for the first frequent semantic "
            "failure when up to 32 tokens may be generated."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-total", type=int, default=147_392)
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
        default=147456,
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_case(
    tokenizer: Any,
    max_total: int,
) -> tuple[dict[str, Any], list[int], list[int]]:
    distractor_pool = boundary.build_distractor_pool(tokenizer)
    case = boundary.build_case(
        tokenizer,
        distractor_pool,
        max_total,
        0,
        filler_text=".",
    )
    query_start, query_end = case["query_span"]
    query_ids = case["prompt_ids"][query_start:query_end]
    body_ids = case["prompt_ids"][:query_start]
    if len(body_ids) + len(query_ids) != max_total:
        raise AssertionError("unexpected prompt partition")
    return case, body_ids, query_ids


def age_mentions(text: str) -> list[str]:
    values = [
        match.group(1).lower()
        for match in NUMBER_PATTERN.finditer(text)
    ]
    return ["nine" if value == "9" else value for value in values]


def semantic_score_at_length(
    *,
    model: Any,
    tokenizer: Any,
    answer_variants: dict[str, list[int]],
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    max_new_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
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

    generated: list[int] = []
    generated_text = ""
    first_age: str | None = None
    generated_cache = query_output.past_key_values
    logits = query_output.logits[:, -1, :]
    past_length = body_length + query_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[0]).item())
        generated.append(next_id)
        generated_text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        mentions = age_mentions(generated_text)
        if mentions:
            first_age = mentions[0]
            break
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
                generated_cache,
                past_length,
            )
        generated_cache = output.past_key_values
        logits = output.logits[:, -1, :]
        past_length += 1
        del output

    cache = incremental.crop_cache(
        generated_cache,
        body_length,
    )
    del query_output
    return (
        cache,
        {
            **score,
            "generated_token_count": len(generated),
            "generated_token_ids": json.dumps(
                generated,
                separators=(",", ":"),
            ),
            "generated_text": generated_text,
            "first_age": first_age,
            "semantic_correct": first_age == "nine",
            "no_age_within_budget": first_age is None,
            "point_seconds": first_token.rounded(
                time.perf_counter() - started
            ),
        },
    )


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
    failures = [
        not bool(row["semantic_correct"]) for row in rows
    ]
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
    attention_mask_buffer: torch.Tensor,
    output_dir: Path,
    start_total: int,
    end_total: int,
    stride: int,
    max_new_tokens: int,
    prefill_chunk_size: int,
    checkpoint_every: int,
) -> tuple[list[dict[str, Any]], float]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    query_length = int(query_ids.shape[1])
    start_body = start_total - query_length
    end_body = end_total - query_length
    if start_body < 1 or end_body > len(body_ids):
        raise ValueError("stage body range is invalid")

    cache, prefill_seconds = one_token.build_cache(
        model,
        body_ids[:start_body],
        prefill_chunk_size,
    )
    current_body = start_body
    totals = checkpoint_totals(
        start_total,
        end_total,
        stride,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    csv_path = stage_dir / "points.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer: csv.DictWriter | None = None
        for index, total in enumerate(totals):
            target_body = total - query_length
            cache, current_body = one_token.extend_cache(
                model,
                cache,
                body_ids,
                current_body,
                target_body,
                attention_mask_buffer,
                prefill_chunk_size,
            )
            cache, result = semantic_score_at_length(
                model=model,
                tokenizer=tokenizer,
                answer_variants=answer_variants,
                query_ids=query_ids,
                cache=cache,
                body_length=current_body,
                attention_mask_buffer=attention_mask_buffer,
                max_new_tokens=max_new_tokens,
            )
            row = {
                "stage": stage,
                "total_tokens": total,
                "kib_tokens": first_token.rounded(
                    total / 1024,
                    6,
                ),
                **result,
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
                or index == len(totals) - 1
            ):
                write_json(
                    stage_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "start_total_tokens": start_total,
                        "end_total_tokens": end_total,
                        "stride": stride,
                        "max_new_tokens": max_new_tokens,
                        "single_shared_prefix": True,
                        "prefill_seconds": prefill_seconds,
                        "completed_points": len(rows),
                        "last_total_tokens": total,
                        "semantic_failure_count": sum(
                            not bool(item["semantic_correct"])
                            for item in rows
                        ),
                        "elapsed_seconds": (
                            time.perf_counter() - started
                        ),
                        "complete": index == len(totals) - 1,
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "completed": len(rows),
                            "total": total,
                            "first_age": result["first_age"],
                            "tokens": result[
                                "generated_token_count"
                            ],
                            "semantic_correct": result[
                                "semantic_correct"
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, prefill_seconds


def first_failure(
    rows: Sequence[dict[str, Any]],
) -> int | None:
    return next(
        (
            int(row["total_tokens"])
            for row in rows
            if not bool(row["semantic_correct"])
        ),
        None,
    )


def main() -> None:
    args = parse_args()
    if (
        args.max_total + args.max_new_tokens
        > args.fixed_max_position_embeddings
    ):
        raise ValueError(
            "prompt plus generation exceeds fixed position limit"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    case, body_ids, query_ids_list = prepare_case(
        tokenizer,
        args.max_total,
    )
    query_length = len(query_ids_list)
    minimum_total = max(
        32,
        int(case["gold_span"][1]) + query_length,
    )
    dry = {
        "minimum_total": minimum_total,
        "maximum_total": args.max_total,
        "query_tokens": query_length,
        "body_tokens": len(body_ids),
        "max_new_tokens": args.max_new_tokens,
        "coarse_stride": COARSE_STRIDE,
        "semantic_window_tokens": SEMANTIC_WINDOW_TOKENS,
        "distractor_count": 0,
        "filler_text": ".",
        "early_stop_after_first_age": True,
    }
    write_json(output_dir / "config.json", dry)
    if args.dry_run:
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
        attention_mask_buffer=attention_mask_buffer,
        output_dir=output_dir,
        start_total=minimum_total,
        end_total=args.max_total,
        stride=COARSE_STRIDE,
        max_new_tokens=args.max_new_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
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
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "no_distractor_multitoken_boundary",
        **dry,
        "coarse": {
            "points": len(coarse_rows),
            "prefill_seconds": coarse_prefill,
            "first_semantic_failure": first_failure(coarse_rows),
            "boundary": coarse_boundary,
        },
        "medium": None,
        "exact": None,
        "complete": False,
    }
    write_json(output_dir / "summary.json", summary)

    if coarse_boundary is not None:
        medium_start = max(
            minimum_total,
            int(coarse_boundary["start_total_tokens"])
            - MEDIUM_CONTEXT_TOKENS,
        )
        medium_end = min(
            args.max_total,
            int(coarse_boundary["end_total_tokens"])
            + MEDIUM_CONTEXT_TOKENS,
        )
        medium_rows, medium_prefill = run_stage(
            stage="medium",
            model=model,
            tokenizer=tokenizer,
            answer_variants=answer_variants,
            body_ids=body_ids,
            query_ids=query_ids,
            attention_mask_buffer=attention_mask_buffer,
            output_dir=output_dir,
            start_total=medium_start,
            end_total=medium_end,
            stride=MEDIUM_STRIDE,
            max_new_tokens=args.max_new_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            checkpoint_every=args.checkpoint_every,
        )
        medium_window_points = max(
            2,
            SEMANTIC_WINDOW_TOKENS // MEDIUM_STRIDE,
        )
        medium_boundary = first_majority_window(
            medium_rows,
            medium_window_points,
        )
        summary["medium"] = {
            "start_total": medium_start,
            "end_total": medium_end,
            "stride": MEDIUM_STRIDE,
            "points": len(medium_rows),
            "prefill_seconds": medium_prefill,
            "first_semantic_failure": first_failure(medium_rows),
            "boundary": medium_boundary,
        }
        write_json(output_dir / "summary.json", summary)

        if medium_boundary is not None:
            exact_start = max(
                minimum_total,
                int(medium_boundary["start_total_tokens"])
                - EXACT_CONTEXT_TOKENS,
            )
            exact_end = min(
                args.max_total,
                int(medium_boundary["end_total_tokens"])
                + EXACT_CONTEXT_TOKENS,
            )
            exact_rows, exact_prefill = run_stage(
                stage="exact",
                model=model,
                tokenizer=tokenizer,
                answer_variants=answer_variants,
                body_ids=body_ids,
                query_ids=query_ids,
                attention_mask_buffer=attention_mask_buffer,
                output_dir=output_dir,
                start_total=exact_start,
                end_total=exact_end,
                stride=1,
                max_new_tokens=args.max_new_tokens,
                prefill_chunk_size=args.prefill_chunk_size,
                checkpoint_every=args.checkpoint_every,
            )
            exact_boundary = first_majority_window(
                exact_rows,
                SEMANTIC_WINDOW_TOKENS,
            )
            summary["exact"] = {
                "start_total": exact_start,
                "end_total": exact_end,
                "stride": 1,
                "points": len(exact_rows),
                "prefill_seconds": exact_prefill,
                "first_semantic_failure": first_failure(
                    exact_rows
                ),
                "boundary": exact_boundary,
            }

    summary["elapsed_seconds"] = (
        time.perf_counter() - started
    )
    summary["complete"] = True
    write_json(output_dir / "summary.json", summary)
    del model, query_ids, attention_mask_buffer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
