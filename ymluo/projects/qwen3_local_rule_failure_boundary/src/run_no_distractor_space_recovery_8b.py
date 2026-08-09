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

import run_incremental_nine_newline_boundary_8b as incremental
import run_incremental_twohop_first_token_8b as first_token
import run_local_rule_failure_boundary as base
import run_no_distractor_failure_boundary_search_8b as no_distractor
import run_twohop_age_distractor_failure_boundary_8b as twohop


DEFAULT_START_TOTAL = 17_802
DEFAULT_END_TOTAL = 18_800
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
            "Continue generation after the single-space first-token "
            "failure in the no-distractor one-hop experiment."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--start-total",
        type=int,
        default=DEFAULT_START_TOTAL,
    )
    parser.add_argument(
        "--end-total",
        type=int,
        default=DEFAULT_END_TOTAL,
    )
    parser.add_argument(
        "--continuation-tokens-after-first",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="0 means all eligible first-token failures.",
    )
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
    parser.add_argument(
        "--fixed-rope-factor",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--fixed-max-position-embeddings",
        type=int,
        default=144 * 1024,
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_eligible_points(
    path: Path,
    start_total: int,
    end_total: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            total = int(raw["total_tokens"])
            if not start_total <= total <= end_total:
                continue
            if raw["top_is_gold"].lower() == "true":
                continue
            rows.append(
                {
                    "total_tokens": total,
                    "source_top_token_id": int(raw["top_token_id"]),
                    "source_top_token_label": raw["top_token_label"],
                    "source_gold_probability": float(
                        raw["gold_exact_probability"]
                    ),
                    "source_competitor_probability": float(
                        raw["strongest_competitor_probability"]
                    ),
                    "source_margin": float(
                        raw["gold_exact_vs_competitor_margin"]
                    ),
                }
            )
    if not rows:
        raise RuntimeError("no eligible first-token failures found")
    return rows


def quantile_sample(
    rows: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return list(rows)
    indices = {
        round(index * (len(rows) - 1) / (count - 1))
        for index in range(count)
    }
    return [rows[index] for index in sorted(indices)]


def age_mentions(text: str) -> list[str]:
    mentions = [
        match.group(1).lower()
        for match in NUMBER_PATTERN.finditer(text)
    ]
    return ["nine" if value == "9" else value for value in mentions]


def generate_after_query(
    model: Any,
    tokenizer: Any,
    query_output: Any,
    prompt_tokens: int,
    continuation_after_first: int,
) -> tuple[dict[str, Any], Any]:
    cache = query_output.past_key_values
    logits = query_output.logits[:, -1, :]
    generated: list[int] = []
    past_length = prompt_tokens
    eos_id = tokenizer.eos_token_id
    maximum = 1 + continuation_after_first
    for _ in range(maximum):
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

    full_text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    post_first_text = tokenizer.decode(
        generated[1:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    full_mentions = age_mentions(full_text)
    post_first_mentions = age_mentions(post_first_text)
    first_token_text = tokenizer.decode(
        generated[:1],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return (
        {
            "token_ids": generated,
            "first_token_id": generated[0],
            "first_token_text": first_token_text,
            "first_token_is_single_space": first_token_text == " ",
            "generated_token_count": len(generated),
            "text": full_text,
            "post_first_text": post_first_text,
            "age_mentions": full_mentions,
            "post_first_age_mentions": post_first_mentions,
            "contains_nine": "nine" in full_mentions,
            "contains_nine_after_first": (
                "nine" in post_first_mentions
            ),
            "first_age": (
                full_mentions[0] if full_mentions else None
            ),
            "first_age_after_first": (
                post_first_mentions[0]
                if post_first_mentions
                else None
            ),
            "first_age_correct": bool(
                full_mentions and full_mentions[0] == "nine"
            ),
            "first_age_after_first_correct": bool(
                post_first_mentions
                and post_first_mentions[0] == "nine"
            ),
        },
        cache,
    )


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    single_space = [
        row
        for row in rows
        if row["generation"]["first_token_is_single_space"]
    ]
    recovered = [
        row
        for row in single_space
        if row["generation"]["contains_nine_after_first"]
    ]
    strict = [
        row
        for row in single_space
        if row["generation"]["first_age_after_first_correct"]
    ]
    no_age = [
        row
        for row in single_space
        if row["generation"]["first_age_after_first"] is None
    ]
    wrong_age_counts: dict[str, int] = {}
    for row in single_space:
        value = row["generation"]["first_age_after_first"]
        if value is None or value == "nine":
            continue
        wrong_age_counts[value] = wrong_age_counts.get(value, 0) + 1
    denominator = max(1, len(single_space))
    return {
        "completed_points": count,
        "single_space_first_token_points": len(single_space),
        "contains_nine_after_space_count": len(recovered),
        "contains_nine_after_space_rate": (
            len(recovered) / denominator
        ),
        "first_age_after_space_is_nine_count": len(strict),
        "first_age_after_space_is_nine_rate": (
            len(strict) / denominator
        ),
        "no_age_mention_after_space_count": len(no_age),
        "no_age_mention_after_space_rate": (
            len(no_age) / denominator
        ),
        "wrong_first_age_counts": dict(
            sorted(
                wrong_age_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = read_eligible_points(
        Path(args.points_csv),
        args.start_total,
        args.end_total,
    )
    selected = quantile_sample(eligible, args.max_points)
    selection = {
        "source_points_csv": args.points_csv,
        "eligible_failure_points": len(eligible),
        "selected_points": len(selected),
        "start_total": args.start_total,
        "end_total": args.end_total,
        "continuation_tokens_after_first": (
            args.continuation_tokens_after_first
        ),
        "selected_totals": [
            row["total_tokens"] for row in selected
        ],
    }
    write_json(output_dir / "selection.json", selection)
    if args.dry_run:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    _, body_ids, query_ids_list = no_distractor.prepare_case(
        tokenizer
    )
    query_length = len(query_ids_list)
    first_body = selected[0]["total_tokens"] - query_length
    last_body = selected[-1]["total_tokens"] - query_length
    if first_body < 1 or last_body > len(body_ids):
        raise ValueError("selected range is outside prepared case")

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
    cache, prefill_seconds = no_distractor.build_cache(
        model,
        body_ids[:first_body],
        args.prefill_chunk_size,
    )
    current_body = first_body
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for index, source in enumerate(selected):
        target_body = source["total_tokens"] - query_length
        cache, current_body = no_distractor.extend_cache(
            model,
            cache,
            body_ids,
            current_body,
            target_body,
            attention_mask_buffer,
            args.prefill_chunk_size,
        )
        with torch.inference_mode():
            query_output = incremental.forward_with_shared_mask(
                model,
                query_ids,
                cache,
                current_body,
                attention_mask_buffer,
                with_logits=True,
            )
        score = first_token.score_first_token(
            tokenizer,
            query_output.logits[0, -1],
            answer_variants,
        )
        score.pop("top12_json", None)
        generation, generated_cache = generate_after_query(
            model,
            tokenizer,
            query_output,
            source["total_tokens"],
            args.continuation_tokens_after_first,
        )
        cache = incremental.crop_cache(
            generated_cache,
            current_body,
        )
        rows.append(
            {
                **source,
                "rerun_score": score,
                "generation": generation,
            }
        )
        if (
            len(rows) % args.checkpoint_every == 0
            or index == len(selected) - 1
        ):
            payload = {
                "schema_version": 1,
                "experiment": "no_distractor_space_recovery",
                **selection,
                "completed_points": len(rows),
                "prefill_count": 1,
                "prefill_seconds": first_token.rounded(
                    prefill_seconds
                ),
                "elapsed_seconds": first_token.rounded(
                    time.perf_counter() - started
                ),
                "summary": summarize(rows),
                "rows": rows,
                "complete": len(rows) == len(selected),
            }
            write_json(output_dir / "results.json", payload)
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "selected": len(selected),
                        "total": source["total_tokens"],
                        "first": generation["first_token_text"],
                        "first_age_after": generation[
                            "first_age_after_first"
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        del query_output

    del cache, model, query_ids, attention_mask_buffer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
