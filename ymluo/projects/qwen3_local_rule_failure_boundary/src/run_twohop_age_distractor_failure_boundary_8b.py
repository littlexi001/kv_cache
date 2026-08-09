from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_age_distractor_failure_boundary_8b as onehop
import run_attention_confidence_sweep_8b as attention_runner
import run_fixed300_age_distractor_qk_8b as age
import run_local_rule_failure_boundary as base


RELATION_TEXT = (
    "Xiaohong has exactly one older brother named Xiaoming.\n"
)
AGE_TEXT = age.GOLD_TEXT
QUERY_TEXT = (
    "\nQuestion: What is the age of Xiaohong's only older brother? "
    "Reply with exactly one English number word and nothing else. Answer:"
)
GOLD_ANSWER = "nine"

PARTITION_ORDER = (
    "relation",
    "gold_age_other",
    "gold_age",
    "distractor_other",
    "distractor_ages",
    "irrelevant_periods",
    "query",
)

PARTITION_LABELS = {
    "relation": "First-hop relation: Xiaohong -> Xiaoming",
    "gold_age_other": "Second-hop age sentence excluding nine",
    "gold_age": "Gold age token (nine)",
    "distractor_other": "Distractor age sentences excluding age tokens",
    "distractor_ages": "Distractor age tokens",
    "irrelevant_periods": "Irrelevant period tokens",
    "query": "Question and answer-instruction tokens",
}


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
) -> dict[str, Any]:
    period_id = age.one_token_id(tokenizer, ".", "irrelevant period")
    relation_ids = age.token_ids(tokenizer, RELATION_TEXT)
    age_ids = age.token_ids(tokenizer, AGE_TEXT)
    query_ids = age.token_ids(tokenizer, QUERY_TEXT)
    gold_age_local = age.local_word_span(tokenizer, AGE_TEXT, GOLD_ANSWER)

    selected = [
        distractor_pool[index % len(distractor_pool)]
        for index in range(distractor_count)
    ]
    fixed_tokens = (
        len(relation_ids)
        + len(age_ids)
        + len(query_ids)
        + sum(len(row["ids"]) for row in selected)
    )
    filler_count = total_tokens - fixed_tokens
    if filler_count < 0:
        raise ValueError(
            f"{distractor_count} distractors require {fixed_tokens} tokens, "
            f"which does not fit total_tokens={total_tokens}"
        )
    gap_counts = age.spread_counts(filler_count, distractor_count + 1)

    prompt_ids: list[int] = []
    positions: dict[str, list[int]] = {
        category: [] for category in PARTITION_ORDER
    }
    distractor_spans: list[tuple[int, int]] = []
    distractor_age_spans: list[tuple[int, int]] = []

    relation_start = len(prompt_ids)
    prompt_ids.extend(relation_ids)
    relation_end = len(prompt_ids)
    append_span_positions(
        positions["relation"],
        relation_start,
        relation_end,
    )

    age_start = len(prompt_ids)
    prompt_ids.extend(age_ids)
    age_end = len(prompt_ids)
    gold_age_span = (
        age_start + gold_age_local[0],
        age_start + gold_age_local[1],
    )
    append_span_positions(
        positions["gold_age_other"],
        age_start,
        age_end,
        gold_age_span,
    )
    append_span_positions(positions["gold_age"], *gold_age_span)

    def append_filler(count: int) -> None:
        start = len(prompt_ids)
        prompt_ids.extend([period_id] * count)
        append_span_positions(
            positions["irrelevant_periods"],
            start,
            start + count,
        )

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
        append_span_positions(
            positions["distractor_other"],
            start,
            end,
            age_span,
        )
        append_span_positions(positions["distractor_ages"], *age_span)
        age_histogram[record["age"]] += 1
        append_filler(gap_counts[index + 1])

    query_start = len(prompt_ids)
    prompt_ids.extend(query_ids)
    query_span = (query_start, len(prompt_ids))
    append_span_positions(positions["query"], *query_span)

    if len(prompt_ids) != total_tokens:
        raise AssertionError(
            f"constructed {len(prompt_ids)} tokens, expected {total_tokens}"
        )
    partition_count = sum(
        len(positions[category]) for category in PARTITION_ORDER
    )
    if partition_count != total_tokens:
        raise AssertionError(
            f"partition covers {partition_count} tokens, "
            f"expected {total_tokens}"
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
        digest.update(
            int(token_id).to_bytes(4, byteorder="little", signed=False)
        )

    preview_indices = list(range(min(6, distractor_count)))
    preview_indices += list(
        range(max(6, distractor_count - 6), distractor_count)
    )
    preview_rows = []
    for index in dict.fromkeys(preview_indices):
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
        "relation_span": [relation_start, relation_end],
        "age_evidence_span": [age_start, age_end],
        "gold_age_span": list(gold_age_span),
        "query_span": list(query_span),
        "age_histogram": dict(sorted(age_histogram.items())),
        "distractor_preview": preview_rows,
        "prompt_sha256": digest.hexdigest(),
        "prompt_prefix_text": tokenizer.decode(
            prompt_ids[: min(100, total_tokens)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "prompt_suffix_text": tokenizer.decode(
            prompt_ids[max(0, total_tokens - 120) :],
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


def validate_answer_variants(
    tokenizer: Any,
) -> dict[str, list[int]]:
    variants: dict[str, list[int]] = {}
    for word in age.NUMBER_WORDS:
        ids: list[int] = []
        for surface in (word, word.title()):
            token_id = age.one_token_id(
                tokenizer,
                f" {surface}",
                f"answer variant {surface}",
            )
            if token_id not in ids:
                ids.append(token_id)
        variants[word] = ids
    return variants


@torch.inference_mode()
def score_semantic_answer(
    tokenizer: Any,
    query_output: Any,
    answer_variants: dict[str, list[int]],
) -> dict[str, Any]:
    logits = query_output.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    gold_ids = answer_variants[GOLD_ANSWER]
    gold_id_set = set(gold_ids)

    top_values, top_indices = torch.topk(
        log_probs,
        k=12,
        largest=True,
        sorted=True,
    )
    top_rows = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "log_probability": age.rounded(float(value)),
            "probability": age.rounded(math.exp(float(value))),
        }
        for value, token_id in zip(
            top_values.tolist(),
            top_indices.tolist(),
        )
    ]
    best_non_gold = next(
        row for row in top_rows if row["token_id"] not in gold_id_set
    )
    best_gold_id = max(
        gold_ids,
        key=lambda token_id: float(log_probs[token_id].item()),
    )
    best_gold_log_probability = float(log_probs[best_gold_id].item())
    semantic_gold_probability = float(
        probabilities[
            torch.tensor(
                gold_ids,
                dtype=torch.long,
                device=probabilities.device,
            )
        ]
        .sum()
        .item()
    )

    candidate_rows = []
    for word, token_ids in answer_variants.items():
        probability = float(
            probabilities[
                torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=probabilities.device,
                )
            ]
            .sum()
            .item()
        )
        candidate_rows.append(
            {
                "word": word,
                "token_ids": token_ids,
                "probability": age.rounded(probability),
                "log_probability": age.rounded(math.log(probability)),
            }
        )
    candidate_rows.sort(
        key=lambda row: float(row["probability"]),
        reverse=True,
    )
    strongest_wrong_candidate = next(
        row for row in candidate_rows if row["word"] != GOLD_ANSWER
    )

    top_id = int(top_indices[0].item())
    exact_lower_id = answer_variants[GOLD_ANSWER][0]
    exact_lower_log_probability = float(log_probs[exact_lower_id].item())
    return {
        "gold_answer": GOLD_ANSWER,
        "gold_variant_token_ids": gold_ids,
        "semantic_gold_probability": age.rounded(
            semantic_gold_probability
        ),
        "semantic_gold_nll": age.rounded(
            -math.log(semantic_gold_probability)
        ),
        "semantic_gold_ppl": age.rounded(
            1.0 / semantic_gold_probability
        ),
        "semantic_full_vocab_margin": age.rounded(
            best_gold_log_probability
            - float(best_non_gold["log_probability"])
        ),
        "semantic_full_vocab_correct": top_id in gold_id_set,
        "top_token_id": top_id,
        "top_token": top_rows[0]["token"],
        "top_probability": top_rows[0]["probability"],
        "strongest_non_gold": best_non_gold,
        "next_token_top12": top_rows,
        "semantic_candidate_margin": age.rounded(
            math.log(semantic_gold_probability)
            - float(strongest_wrong_candidate["log_probability"])
        ),
        "semantic_candidate_correct": (
            candidate_rows[0]["word"] == GOLD_ANSWER
        ),
        "semantic_candidate_prediction": candidate_rows[0]["word"],
        "strongest_wrong_candidate": strongest_wrong_candidate,
        "candidate_scores": candidate_rows,
        "exact_lowercase_probability": age.rounded(
            math.exp(exact_lower_log_probability)
        ),
        "exact_lowercase_ppl": age.rounded(
            math.exp(-exact_lower_log_probability)
        ),
        "exact_lowercase_top1_correct": top_id == exact_lower_id,
    }


@torch.inference_mode()
def greedy_generate_from_output(
    model: Any,
    tokenizer: Any,
    query_output: Any,
    prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if max_new_tokens <= 0:
        return None
    cache = query_output.past_key_values
    logits = query_output.logits[:, -1, :]
    generated_ids: list[int] = []
    past_len = prompt_tokens
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[0]).item())
        generated_ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
        token = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=base.input_device(model),
        )
        output = base.forward_with_cache(
            model,
            token,
            cache,
            past_len,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        past_len += 1
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    pattern = r"\b(" + "|".join(age.NUMBER_WORDS) + r")\b"
    match = re.search(pattern, normalized)
    extracted = match.group(1) if match else None
    return {
        "token_ids": generated_ids,
        "text": text,
        "normalized": normalized,
        "first_number_word": extracted,
        "answer_correct": extracted == GOLD_ANSWER,
    }


def point_filename(total_tokens: int, distractor_count: int) -> str:
    return (
        f"tokens_{total_tokens:06d}_"
        f"distractors_{distractor_count:05d}.json"
    )


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
            "experiment": "twohop_age_distractor_failure_boundary",
            "model_name_or_path": args.model_name_or_path,
            "relation_evidence": RELATION_TEXT.rstrip(),
            "age_evidence": AGE_TEXT.rstrip(),
            "query": QUERY_TEXT.strip(),
            "gold_answer": GOLD_ANSWER,
            "fixed_rope_factor": args.fixed_rope_factor,
            "fixed_max_position_embeddings": (
                args.fixed_max_position_embeddings
            ),
            "partition_order": list(PARTITION_ORDER),
            "partition_labels": PARTITION_LABELS,
            "requested_points": [
                {
                    "total_tokens": total,
                    "distractor_count": count,
                }
                for total, count in points
            ],
            "completed_count": len(completed),
            "completed": list(completed),
            "elapsed_seconds": age.rounded(elapsed_seconds),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-hop age QA failure-boundary scan with the same distractor "
            "sentences and density used by the prior one-hop experiment."
        )
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
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--fixed-rope-factor",
        type=float,
        help=(
            "Keep one RoPE/YaRN scaling configuration across all bisection "
            "rounds instead of deriving it from each round's longest point."
        ),
    )
    parser.add_argument(
        "--fixed-max-position-embeddings",
        type=int,
        help="Fixed configured context capacity used with --fixed-rope-factor.",
    )
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=0,
        help=(
            "Optionally continue greedy decoding and score the first "
            "generated English number word."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = onehop.parse_points(args.points)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    answer_variants = validate_answer_variants(tokenizer)
    distractor_pool = onehop.build_distractor_pool(tokenizer)
    cases = [
        build_case(
            tokenizer,
            distractor_pool,
            total_tokens,
            distractor_count,
        )
        for total_tokens, distractor_count in points
    ]
    design = {
        "schema_version": 1,
        "experiment": "twohop_age_distractor_failure_boundary",
        "relation_evidence": RELATION_TEXT.rstrip(),
        "age_evidence": AGE_TEXT.rstrip(),
        "query": QUERY_TEXT.strip(),
        "gold_answer": GOLD_ANSWER,
        "fixed_rope_factor": args.fixed_rope_factor,
        "fixed_max_position_embeddings": (
            args.fixed_max_position_embeddings
        ),
        "answer_variants": answer_variants,
        "partition_order": list(PARTITION_ORDER),
        "partition_labels": PARTITION_LABELS,
        "distractor_pool_size": len(distractor_pool),
        "cases": [public_case(case) for case in cases],
    }
    age.write_json_atomic(output_dir / "design.json", design)
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return

    requested_max_position = max(total for total, _ in points)
    max_position = (
        args.fixed_max_position_embeddings
        if args.fixed_max_position_embeddings is not None
        else requested_max_position
    )
    if max_position < requested_max_position:
        raise ValueError(
            "fixed-max-position-embeddings is shorter than a requested point"
        )
    max_factor = (
        args.fixed_rope_factor
        if args.fixed_rope_factor is not None
        else base.rope_factor_for_length(
            max_position,
            args.original_max_position_embeddings,
        )
    )
    if max_factor <= 0:
        raise ValueError("fixed-rope-factor must be positive")
    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        max_position,
        max_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError(
            "Tokenizer changed between validation and model loading"
        )
    del tokenizer
    tokenizer = model_tokenizer

    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        total_tokens = int(case["total_tokens"])
        distractor_count = int(case["distractor_count"])
        file_name = point_filename(total_tokens, distractor_count)
        point_started = time.perf_counter()
        prompt = torch.tensor(
            case["prompt_ids"],
            dtype=torch.long,
        ).view(1, -1)
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
            (
                query_output,
                captured_queries,
                query_seconds,
            ) = attention_runner.capture_query_states(
                model,
                query_cache,
                prompt[:, -1:],
                total_tokens - 1,
            )
        answer = score_semantic_answer(
            tokenizer,
            query_output,
            answer_variants,
        )
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
        generation = greedy_generate_from_output(
            model,
            tokenizer,
            query_output,
            total_tokens,
            args.generation_max_new_tokens,
        )
        result = {
            "schema_version": 1,
            "experiment": "twohop_age_distractor_failure_boundary",
            "model_name_or_path": args.model_name_or_path,
            "case": public_case(case),
            "answer": answer,
            "generation": generation,
            "attention": attention,
            "answer_only": bool(args.answer_only),
            "timing": {
                "prefill_seconds": age.rounded(prefill_seconds),
                "query_seconds": age.rounded(query_seconds),
                "point_seconds": age.rounded(
                    time.perf_counter() - point_started
                ),
            },
        }
        age.write_json_atomic(output_dir / file_name, result)
        row = {
            "total_tokens": total_tokens,
            "distractor_count": distractor_count,
            "filler_count": case["filler_count"],
            "file": file_name,
            "semantic_gold_ppl": answer["semantic_gold_ppl"],
            "semantic_gold_probability": answer[
                "semantic_gold_probability"
            ],
            "semantic_full_vocab_margin": answer[
                "semantic_full_vocab_margin"
            ],
            "semantic_full_vocab_correct": answer[
                "semantic_full_vocab_correct"
            ],
            "semantic_candidate_margin": answer[
                "semantic_candidate_margin"
            ],
            "semantic_candidate_correct": answer[
                "semantic_candidate_correct"
            ],
            "semantic_candidate_prediction": answer[
                "semantic_candidate_prediction"
            ],
            "exact_lowercase_top1_correct": answer[
                "exact_lowercase_top1_correct"
            ],
            "top_token": answer["top_token"],
            "generation_answer": (
                None
                if generation is None
                else generation["first_number_word"]
            ),
            "generation_correct": (
                None
                if generation is None
                else generation["answer_correct"]
            ),
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
        print(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        del (
            prompt,
            query_cache,
            query_output,
            captured_queries,
            result,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (
            args.stop_on_failure
            and not bool(answer["semantic_full_vocab_correct"])
        ):
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
