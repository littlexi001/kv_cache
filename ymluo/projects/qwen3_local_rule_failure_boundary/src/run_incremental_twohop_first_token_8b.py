from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

import run_incremental_nine_newline_boundary_8b as incremental
import run_local_rule_failure_boundary as base
import run_twohop_age_distractor_failure_boundary_8b as twohop


DEFAULT_START_TOTAL = 98 * 1024
DEFAULT_END_TOTAL = 120 * 1024
DEFAULT_MAX_DISTRACTORS = DEFAULT_END_TOTAL // 32 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Token-by-token two-hop first-token scan. Prefill the shared body "
            "once, append one body token per step, temporarily evaluate the "
            "fixed query, and crop the query from the cache."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--start-total-tokens",
        type=int,
        default=DEFAULT_START_TOTAL,
    )
    parser.add_argument(
        "--end-total-tokens",
        type=int,
        default=DEFAULT_END_TOTAL,
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=DEFAULT_MAX_DISTRACTORS,
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
        default=32768,
    )
    parser.add_argument("--fixed-rope-factor", type=float, default=8.0)
    parser.add_argument(
        "--fixed-max-position-embeddings",
        type=int,
        default=163840,
    )
    parser.add_argument(
        "--max-added-tokens",
        type=int,
        help="Optional short-run cap used for validation.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


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


def token_label(text: str) -> str:
    if text == "":
        return "∅"
    return (
        text.replace(" ", "␠")
        .replace("\r", "␍")
        .replace("\n", "↵")
        .replace("\t", "⇥")
    )


def prompt_digest(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(
            int(token_id).to_bytes(
                4,
                byteorder="little",
                signed=False,
            )
        )
    return digest.hexdigest()


def score_first_token(
    tokenizer: Any,
    logits: torch.Tensor,
    answer_variants: dict[str, list[int]],
) -> dict[str, Any]:
    logits = logits.float()
    log_normalizer = torch.logsumexp(logits, dim=-1)
    gold_ids = [int(value) for value in answer_variants[twohop.GOLD_ANSWER]]
    gold_index = torch.tensor(
        gold_ids,
        dtype=torch.long,
        device=logits.device,
    )
    gold_logits = logits[gold_index]
    best_gold_offset = int(torch.argmax(gold_logits).item())
    best_gold_id = gold_ids[best_gold_offset]
    exact_gold_id = gold_ids[0]

    competitor_logits = logits.clone()
    competitor_logits[gold_index] = -torch.inf
    competitor_id = int(torch.argmax(competitor_logits).item())
    top_id = int(torch.argmax(logits).item())

    exact_gold_log_probability = float(
        logits[exact_gold_id].item() - log_normalizer.item()
    )
    semantic_gold_log_probability = float(
        torch.logsumexp(gold_logits, dim=-1).item()
        - log_normalizer.item()
    )
    competitor_log_probability = float(
        logits[competitor_id].item() - log_normalizer.item()
    )
    top_log_probability = float(
        logits[top_id].item() - log_normalizer.item()
    )

    top_values, top_indices = torch.topk(
        logits,
        k=min(12, int(logits.shape[-1])),
        largest=True,
        sorted=True,
    )
    top_rows = []
    for value, token_id in zip(
        top_values.tolist(),
        top_indices.tolist(),
    ):
        text = token_text(tokenizer, int(token_id))
        top_rows.append(
            {
                "token_id": int(token_id),
                "token_text": text,
                "token_label": token_label(text),
                "probability": rounded(
                    math.exp(float(value) - log_normalizer.item())
                ),
            }
        )

    age_candidate_rows = []
    for word, ids in answer_variants.items():
        index = torch.tensor(
            ids,
            dtype=torch.long,
            device=logits.device,
        )
        log_probability = float(
            torch.logsumexp(logits[index], dim=-1).item()
            - log_normalizer.item()
        )
        age_candidate_rows.append(
            {
                "word": word,
                "probability": math.exp(log_probability),
                "log_probability": log_probability,
            }
        )
    age_candidate_rows.sort(
        key=lambda row: float(row["probability"]),
        reverse=True,
    )

    top_text = token_text(tokenizer, top_id)
    competitor_text = token_text(tokenizer, competitor_id)
    best_gold_text = token_text(tokenizer, best_gold_id)
    return {
        "gold_exact_token_id": exact_gold_id,
        "gold_exact_probability": rounded(
            math.exp(exact_gold_log_probability)
        ),
        "gold_exact_ppl": rounded(
            math.exp(-exact_gold_log_probability)
        ),
        "gold_semantic_probability": rounded(
            math.exp(semantic_gold_log_probability)
        ),
        "gold_semantic_ppl": rounded(
            math.exp(-semantic_gold_log_probability)
        ),
        "best_gold_token_id": best_gold_id,
        "best_gold_token_text": best_gold_text,
        "best_gold_token_label": token_label(best_gold_text),
        "top_token_id": top_id,
        "top_token_text": top_text,
        "top_token_label": token_label(top_text),
        "top_probability": rounded(math.exp(top_log_probability)),
        "top_is_gold": top_id in set(gold_ids),
        "strongest_competitor_token_id": competitor_id,
        "strongest_competitor_token_text": competitor_text,
        "strongest_competitor_token_label": token_label(
            competitor_text
        ),
        "strongest_competitor_probability": rounded(
            math.exp(competitor_log_probability)
        ),
        "gold_exact_vs_competitor_margin": rounded(
            logits[exact_gold_id].item() - logits[competitor_id].item()
        ),
        "best_gold_vs_competitor_margin": rounded(
            logits[best_gold_id].item() - logits[competitor_id].item()
        ),
        "age_candidate_prediction": age_candidate_rows[0]["word"],
        "age_candidate_probability": rounded(
            float(age_candidate_rows[0]["probability"])
        ),
        "age_candidate_margin": rounded(
            semantic_gold_log_probability
            - float(
                next(
                    row["log_probability"]
                    for row in age_candidate_rows
                    if row["word"] != twohop.GOLD_ANSWER
                )
            )
        ),
        "top12_json": json.dumps(
            top_rows,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def checkpoint_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    competitor_counts: collections.Counter[str],
    prediction_counts: collections.Counter[str],
    prefill_seconds: float,
    started: float,
    requested_end: int,
    continuation_counts: dict[str, int],
) -> None:
    first_failure = next(
        (row for row in rows if not bool(row["top_is_gold"])),
        None,
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "incremental_twohop_first_token",
            "model_name_or_path": args.model_name_or_path,
            "start_total_tokens": args.start_total_tokens,
            "end_total_tokens": requested_end,
            "completed_points": len(rows),
            "expected_points": (
                requested_end - args.start_total_tokens + 1
            ),
            "single_shared_prefix_prefill": True,
            "prefill_count": 1,
            "prefill_seconds": rounded(prefill_seconds),
            "elapsed_seconds": rounded(time.perf_counter() - started),
            "first_failure": first_failure,
            "last_point": rows[-1] if rows else None,
            "strongest_competitor_counts": dict(
                competitor_counts.most_common()
            ),
            "top_prediction_counts": dict(
                prediction_counts.most_common()
            ),
            "continuation_category_counts": continuation_counts,
            "complete": len(rows)
            == requested_end - args.start_total_tokens + 1,
        },
    )


def main() -> None:
    args = parse_args()
    if args.start_total_tokens <= 0:
        raise ValueError("start-total-tokens must be positive")
    if args.end_total_tokens < args.start_total_tokens:
        raise ValueError(
            "end-total-tokens must be at least start-total-tokens"
        )
    if args.fixed_max_position_embeddings < args.end_total_tokens:
        raise ValueError(
            "fixed-max-position-embeddings is shorter than the scan"
        )
    requested_added = args.end_total_tokens - args.start_total_tokens
    if args.max_added_tokens is not None:
        if args.max_added_tokens < 0:
            raise ValueError("max-added-tokens must be non-negative")
        requested_added = min(
            requested_added,
            args.max_added_tokens,
        )
    requested_end = args.start_total_tokens + requested_added

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    distractor_pool = twohop.onehop.build_distractor_pool(tokenizer)
    max_case = twohop.build_case(
        tokenizer,
        distractor_pool,
        args.end_total_tokens,
        args.max_distractors,
    )
    query_start, query_end = max_case["query_span"]
    query_ids_list = max_case["prompt_ids"][query_start:query_end]
    query_length = len(query_ids_list)
    base_body_length = args.start_total_tokens - query_length
    max_body_ids = max_case["prompt_ids"][:query_start]
    if base_body_length <= 0:
        raise ValueError("query is longer than the starting prompt")
    if len(max_body_ids) != args.end_total_tokens - query_length:
        raise AssertionError("unexpected maximum body length")
    base_body_ids = max_body_ids[:base_body_length]
    continuation_ids = max_body_ids[
        base_body_length : base_body_length + requested_added
    ]
    if len(continuation_ids) != requested_added:
        raise AssertionError("continuation length mismatch")

    category_lookup = incremental.build_category_lookup(
        args.end_total_tokens,
        max_case["category_positions"],
    )
    continuation_categories = category_lookup[
        base_body_length : base_body_length + requested_added
    ]
    continuation_counts = {
        category: continuation_categories.count(category)
        for category in twohop.PARTITION_ORDER
    }
    base_prompt_ids = base_body_ids + query_ids_list

    design = {
        "schema_version": 1,
        "experiment": "incremental_twohop_first_token",
        "model_name_or_path": args.model_name_or_path,
        "single_shared_prefix_prefill": True,
        "prefill_count": 1,
        "start_total_tokens": args.start_total_tokens,
        "start_kib_tokens": args.start_total_tokens / 1024,
        "end_total_tokens": requested_end,
        "end_kib_tokens": requested_end / 1024,
        "full_case_total_tokens": args.end_total_tokens,
        "base_body_tokens": base_body_length,
        "query_tokens": query_length,
        "continuation_tokens": requested_added,
        "max_distractors": args.max_distractors,
        "relation_evidence": twohop.RELATION_TEXT.rstrip(),
        "age_evidence": twohop.AGE_TEXT.rstrip(),
        "query": twohop.QUERY_TEXT.strip(),
        "gold_answer": twohop.GOLD_ANSWER,
        "gold_variant_token_ids": answer_variants[
            twohop.GOLD_ANSWER
        ],
        "fixed_rope_factor": args.fixed_rope_factor,
        "fixed_max_position_embeddings": (
            args.fixed_max_position_embeddings
        ),
        "original_max_position_embeddings": (
            args.original_max_position_embeddings
        ),
        "base_prompt_sha256": prompt_digest(base_prompt_ids),
        "max_prompt_sha256": max_case["prompt_sha256"],
        "base_prompt_prefix": tokenizer.decode(
            base_prompt_ids[:100],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "base_body_suffix": tokenizer.decode(
            base_body_ids[-100:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "continuation_prefix": tokenizer.decode(
            continuation_ids[:100],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "continuation_suffix": tokenizer.decode(
            continuation_ids[-100:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "continuation_category_counts": continuation_counts,
    }
    write_json(output_dir / "design.json", design)
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return

    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError(
            "Tokenizer changed between validation and model loading"
        )
    del tokenizer
    tokenizer = model_tokenizer

    input_device = base.input_device(model)
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=input_device,
    )
    body_prompt = torch.tensor(
        base_body_ids,
        dtype=torch.long,
    ).view(1, -1)
    query_ids = torch.tensor(
        query_ids_list,
        dtype=torch.long,
    ).view(1, -1)

    started = time.perf_counter()
    legacy_cache, prefill_seconds = base.prefill_sequence(
        model,
        body_prompt,
        args.prefill_chunk_size,
    )
    cache = base.cache_from_legacy(legacy_cache)
    del legacy_cache, body_prompt

    csv_path = output_dir / "points.csv"
    rows: list[dict[str, Any]] = []
    competitor_counts: collections.Counter[str] = collections.Counter()
    prediction_counts: collections.Counter[str] = collections.Counter()
    body_length = base_body_length
    point_times: list[float] = []

    csv_handle = csv_path.open("w", encoding="utf-8", newline="")
    writer: csv.DictWriter | None = None
    try:
        for added in range(requested_added + 1):
            point_started = time.perf_counter()
            if added > 0:
                next_id = int(continuation_ids[added - 1])
                with torch.inference_mode():
                    body_output = incremental.forward_with_shared_mask(
                        model,
                        torch.tensor(
                            [[next_id]],
                            dtype=torch.long,
                        ),
                        cache,
                        body_length,
                        attention_mask_buffer,
                        with_logits=False,
                    )
                cache = body_output.past_key_values
                body_length += 1
                del body_output

            base.synchronize()
            query_started = time.perf_counter()
            with torch.inference_mode():
                query_output = incremental.forward_with_shared_mask(
                    model,
                    query_ids,
                    cache,
                    body_length,
                    attention_mask_buffer,
                    with_logits=True,
                )
            base.synchronize()
            query_seconds = time.perf_counter() - query_started
            score = score_first_token(
                tokenizer,
                query_output.logits[0, -1],
                answer_variants,
            )
            added_token_text = (
                ""
                if added == 0
                else token_text(
                    tokenizer,
                    int(continuation_ids[added - 1]),
                )
            )
            point = {
                "added_tokens": added,
                "total_tokens": args.start_total_tokens + added,
                "kib_tokens": rounded(
                    (args.start_total_tokens + added) / 1024,
                    6,
                ),
                "body_tokens": body_length,
                "query_position": (
                    body_length + query_length - 1
                ),
                "added_token_id": (
                    ""
                    if added == 0
                    else int(continuation_ids[added - 1])
                ),
                "added_token_text": added_token_text,
                "added_token_label": (
                    "baseline"
                    if added == 0
                    else token_label(added_token_text)
                ),
                "added_token_category": (
                    "baseline"
                    if added == 0
                    else continuation_categories[added - 1]
                ),
                **score,
                "query_seconds": rounded(query_seconds),
                "point_seconds": rounded(
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
            point_times.append(float(point["point_seconds"]))
            competitor_counts[
                f'{point["strongest_competitor_token_id"]}:'
                f'{point["strongest_competitor_token_label"]}'
            ] += 1
            prediction_counts[
                f'{point["top_token_id"]}:'
                f'{point["top_token_label"]}'
            ] += 1

            cache = incremental.crop_cache(
                query_output.past_key_values,
                body_length,
            )
            del query_output

            if (
                added % args.checkpoint_every == 0
                or added == requested_added
            ):
                checkpoint_manifest(
                    output_dir,
                    args,
                    rows,
                    competitor_counts,
                    prediction_counts,
                    prefill_seconds,
                    started,
                    requested_end,
                    continuation_counts,
                )
                recent = point_times[
                    -min(
                        len(point_times),
                        args.checkpoint_every,
                    ) :
                ]
                seconds_per_point = sum(recent) / len(recent)
                remaining = requested_added - added
                print(
                    json.dumps(
                        {
                            "added": added,
                            "total_tokens": (
                                args.start_total_tokens + added
                            ),
                            "top_token": point["top_token_label"],
                            "top_is_gold": point["top_is_gold"],
                            "gold_probability": point[
                                "gold_exact_probability"
                            ],
                            "strongest_competitor": point[
                                "strongest_competitor_token_label"
                            ],
                            "competitor_probability": point[
                                "strongest_competitor_probability"
                            ],
                            "margin": point[
                                "gold_exact_vs_competitor_margin"
                            ],
                            "seconds_per_point_recent": rounded(
                                seconds_per_point
                            ),
                            "eta_seconds": rounded(
                                remaining * seconds_per_point
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    finally:
        csv_handle.close()

    checkpoint_manifest(
        output_dir,
        args,
        rows,
        competitor_counts,
        prediction_counts,
        prefill_seconds,
        started,
        requested_end,
        continuation_counts,
    )


if __name__ == "__main__":
    main()
