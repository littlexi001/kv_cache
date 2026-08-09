from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

import run_incremental_nine_newline_boundary_8b as incremental
import run_local_rule_failure_boundary as base
import run_no_distractor_failure_boundary_search_8b as one_token
import run_no_distractor_multitoken_boundary_8b as multitoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export next-token probabilities after forcing the first "
            "single-space token in the no-distractor age probe."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-total", type=int, default=136_000)
    parser.add_argument("--end-total", type=int, default=146_000)
    parser.add_argument("--stride", type=int, default=256)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def single_token_id(
    tokenizer: Any,
    text: str,
) -> int:
    values = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if len(values) != 1:
        raise RuntimeError(
            f"{text!r} is not one token: {values}"
        )
    return int(values[0])


def score_distribution(
    logits: torch.Tensor,
    correct_id: int,
) -> dict[str, Any]:
    probabilities = torch.softmax(
        logits.float(),
        dim=-1,
    )
    correct_probability = float(
        probabilities[correct_id].item()
    )
    competitor_probabilities = probabilities.clone()
    competitor_probabilities[correct_id] = -1.0
    competitor_id = int(
        torch.argmax(competitor_probabilities).item()
    )
    competitor_probability = float(
        probabilities[competitor_id].item()
    )
    top_id = int(torch.argmax(probabilities).item())
    return {
        "correct_probability": correct_probability,
        "strongest_competitor_token_id": competitor_id,
        "strongest_competitor_probability": (
            competitor_probability
        ),
        "correct_vs_competitor_log_probability_margin": (
            float(
                torch.log(
                    probabilities[correct_id].clamp_min(1e-30)
                ).item()
                - torch.log(
                    probabilities[competitor_id].clamp_min(1e-30)
                ).item()
            )
        ),
        "top_token_id": top_id,
        "top_probability": float(
            probabilities[top_id].item()
        ),
        "top_is_correct": top_id == correct_id,
    }


def main() -> None:
    args = parse_args()
    if args.end_total + 1 > args.fixed_max_position_embeddings:
        raise ValueError("second decoding step exceeds position limit")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    _, body_ids, query_ids_list = multitoken.prepare_case(
        tokenizer,
        args.end_total,
    )
    query_length = len(query_ids_list)
    totals = list(
        range(
            args.start_total,
            args.end_total + 1,
            args.stride,
        )
    )
    if totals[-1] != args.end_total:
        totals.append(args.end_total)
    space_id = single_token_id(tokenizer, " ")
    digit_nine_id = single_token_id(tokenizer, "9")
    leading_space_nine_id = single_token_id(tokenizer, " nine")
    config = {
        "start_total": args.start_total,
        "end_total": args.end_total,
        "stride": args.stride,
        "points": len(totals),
        "query_tokens": query_length,
        "conditioned_first_token": {
            "token_id": space_id,
            "text": " ",
        },
        "correct_second_token": {
            "token_id": digit_nine_id,
            "text": "9",
            "meaning": "age nine",
        },
        "original_one_token_answer": {
            "token_id": leading_space_nine_id,
            "text": " nine",
        },
    }
    write_json(output_dir / "config.json", config)
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
    first_body = args.start_total - query_length
    cache, prefill_seconds = one_token.build_cache(
        model,
        body_ids[:first_body],
        args.prefill_chunk_size,
    )
    current_body = first_body
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for total in totals:
        target_body = total - query_length
        cache, current_body = one_token.extend_cache(
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
        first_score = score_distribution(
            query_output.logits[0, -1],
            leading_space_nine_id,
        )
        with torch.inference_mode():
            second_output = base.forward_with_cache(
                model,
                torch.tensor(
                    [[space_id]],
                    dtype=torch.long,
                    device=base.input_device(model),
                ),
                query_output.past_key_values,
                total,
            )
        second_score = score_distribution(
            second_output.logits[0, -1],
            digit_nine_id,
        )
        competitor_id = int(
            second_score["strongest_competitor_token_id"]
        )
        top_id = int(second_score["top_token_id"])
        cache = incremental.crop_cache(
            second_output.past_key_values,
            current_body,
        )
        row = {
            "total_tokens_before_generation": total,
            "k_tokens": total / 1000,
            "conditioned_first_token_id": space_id,
            "conditioned_first_token_text": " ",
            "first_step_original_nine_probability": (
                first_score["correct_probability"]
            ),
            "first_step_space_probability": (
                query_output.logits[0, -1]
                .float()
                .softmax(dim=-1)[space_id]
                .item()
            ),
            "correct_second_token_id": digit_nine_id,
            "correct_second_token_text": "9",
            "correct_second_token_probability": second_score[
                "correct_probability"
            ],
            "strongest_competitor_token_id": competitor_id,
            "strongest_competitor_token_text": token_text(
                tokenizer,
                competitor_id,
            ),
            "strongest_competitor_probability": second_score[
                "strongest_competitor_probability"
            ],
            "correct_vs_competitor_log_probability_margin": (
                second_score[
                    "correct_vs_competitor_log_probability_margin"
                ]
            ),
            "top_token_id": top_id,
            "top_token_text": token_text(tokenizer, top_id),
            "top_probability": second_score[
                "top_probability"
            ],
            "top_is_correct": second_score["top_is_correct"],
        }
        rows.append(row)
        del query_output, second_output
        print(
            json.dumps(
                {
                    "completed": len(rows),
                    "total": total,
                    "p9": row[
                        "correct_second_token_probability"
                    ],
                    "competitor": row[
                        "strongest_competitor_token_text"
                    ],
                    "top_is_9": row["top_is_correct"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )

    csv_path = output_dir / "post_space_logits.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "summary.json",
        {
            **config,
            "completed_points": len(rows),
            "prefill_count": 1,
            "prefill_seconds": prefill_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "correct_second_token_count": sum(
                bool(row["top_is_correct"]) for row in rows
            ),
            "complete": True,
        },
    )
    del cache, model, query_ids, attention_mask_buffer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
