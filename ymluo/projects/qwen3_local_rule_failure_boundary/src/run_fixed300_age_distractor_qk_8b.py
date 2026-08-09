from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_local_rule_failure_boundary as base


CATEGORY_ORDER = (
    "gold_line",
    "gold_age",
    "distractor_lines",
    "distractor_ages",
    "irrelevant_periods",
)

CATEGORY_LABELS = {
    "gold_line": "Gold evidence sentence",
    "gold_age": "Gold age token (nine)",
    "distractor_lines": "Distractor sentences",
    "distractor_ages": "Distractor age tokens",
    "irrelevant_periods": "Irrelevant period tokens",
}

NUMBER_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

DISTRACTORS = (
    ("Alice", "four"),
    ("Bob", "seven"),
    ("Carol", "two"),
    ("David", "six"),
    ("Emma", "eight"),
    ("Frank", "three"),
    ("Grace", "five"),
    ("Henry", "one"),
    ("Irene", "ten"),
)

GOLD_TEXT = "Xiaoming's age is nine years.\n"
QUERY_TEXT = (
    "\nQuestion: What is Xiaoming's age? "
    "Reply with exactly one English number word and nothing else. Answer:"
)


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{value:.{digits}g}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]


def one_token_id(tokenizer: Any, text: str, label: str) -> int:
    ids = token_ids(tokenizer, text)
    if len(ids) != 1:
        raise RuntimeError(f"{label} must be exactly one token: {text!r} -> {ids}")
    return ids[0]


def local_word_span(tokenizer: Any, text: str, word: str) -> tuple[int, int]:
    start = text.find(word)
    if start < 0:
        raise ValueError(f"{word!r} not found in {text!r}")
    end = start + len(word)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("A fast tokenizer with offset_mapping is required")
    overlapping = [
        index
        for index, (left, right) in enumerate(offsets)
        if int(right) > start and int(left) < end
    ]
    if len(overlapping) != 1:
        raise RuntimeError(
            f"age word must occupy exactly one token in its sentence: "
            f"{word!r} -> token indices {overlapping}"
        )
    return overlapping[0], overlapping[0] + 1


def spread_counts(total: int, buckets: int) -> list[int]:
    if total < 0:
        raise ValueError(f"cannot distribute a negative token count: {total}")
    base_count, remainder = divmod(total, buckets)
    return [base_count + (1 if index < remainder else 0) for index in range(buckets)]


def build_case(tokenizer: Any, distractor_count: int, total_tokens: int) -> dict[str, Any]:
    if not 0 <= distractor_count <= len(DISTRACTORS):
        raise ValueError(f"distractor_count must be 0..{len(DISTRACTORS)}")

    period_id = one_token_id(tokenizer, ".", "irrelevant period")
    gold_ids = token_ids(tokenizer, GOLD_TEXT)
    query_ids = token_ids(tokenizer, QUERY_TEXT)
    gold_age_local = local_word_span(tokenizer, GOLD_TEXT, "nine")

    distractor_records: list[dict[str, Any]] = []
    distractor_ids: list[list[int]] = []
    for name, age in DISTRACTORS[:distractor_count]:
        text = f"{name}'s age is {age} years.\n"
        ids = token_ids(tokenizer, text)
        age_span = local_word_span(tokenizer, text, age)
        distractor_records.append(
            {
                "name": name,
                "age": age,
                "text": text,
                "token_count": len(ids),
                "age_local_span": list(age_span),
            }
        )
        distractor_ids.append(ids)

    fixed_tokens = len(gold_ids) + len(query_ids) + sum(len(ids) for ids in distractor_ids)
    filler_count = total_tokens - fixed_tokens
    if filler_count < 0:
        raise RuntimeError(
            f"fixed content needs {fixed_tokens} tokens, exceeding total_tokens={total_tokens}"
        )
    gaps = spread_counts(filler_count, distractor_count + 1)

    prompt_ids: list[int] = []
    token_categories: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {category: [] for category in CATEGORY_ORDER}

    gold_start = len(prompt_ids)
    prompt_ids.extend(gold_ids)
    token_categories.extend(["gold_line"] * len(gold_ids))
    gold_end = len(prompt_ids)
    spans["gold_line"].append((gold_start, gold_end))
    spans["gold_age"].append(
        (gold_start + gold_age_local[0], gold_start + gold_age_local[1])
    )

    def append_filler(count: int) -> None:
        start = len(prompt_ids)
        prompt_ids.extend([period_id] * count)
        token_categories.extend(["irrelevant_periods"] * count)
        if count:
            spans["irrelevant_periods"].append((start, start + count))

    append_filler(gaps[0])
    for index, ids in enumerate(distractor_ids):
        start = len(prompt_ids)
        prompt_ids.extend(ids)
        token_categories.extend(["distractor_lines"] * len(ids))
        end = len(prompt_ids)
        spans["distractor_lines"].append((start, end))
        age_local = distractor_records[index]["age_local_span"]
        age_span = (start + int(age_local[0]), start + int(age_local[1]))
        spans["distractor_ages"].append(age_span)
        distractor_records[index]["span"] = [start, end]
        distractor_records[index]["age_span"] = list(age_span)
        append_filler(gaps[index + 1])

    query_start = len(prompt_ids)
    prompt_ids.extend(query_ids)
    token_categories.extend(["query"] * len(query_ids))
    query_span = (query_start, len(prompt_ids))

    if len(prompt_ids) != total_tokens:
        raise AssertionError(f"constructed {len(prompt_ids)} tokens, expected {total_tokens}")
    if spans["gold_line"][0][0] != 0:
        raise AssertionError("gold evidence must start at position zero")
    if query_span[1] != total_tokens:
        raise AssertionError("question must end at the final prompt token")

    category_positions = {
        category: [
            position
            for start, end in spans[category]
            for position in range(start, end)
        ]
        for category in CATEGORY_ORDER
    }
    category_counts = {
        category: len(category_positions[category])
        for category in CATEGORY_ORDER
    }

    decoded_tokens = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in prompt_ids
    ]
    return {
        "distractor_count": distractor_count,
        "total_tokens": total_tokens,
        "prompt_ids": prompt_ids,
        "prompt_text": tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "decoded_tokens": decoded_tokens,
        "token_categories": token_categories,
        "category_spans": {
            category: [list(span) for span in spans[category]]
            for category in CATEGORY_ORDER
        },
        "category_positions": category_positions,
        "category_counts": category_counts,
        "gold_text": GOLD_TEXT,
        "gold_span": list(spans["gold_line"][0]),
        "gold_age_span": list(spans["gold_age"][0]),
        "distractors": distractor_records,
        "filler_token_id": period_id,
        "filler_token_text": tokenizer.decode(
            [period_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "filler_count": filler_count,
        "filler_gap_counts": gaps,
        "query_text": QUERY_TEXT,
        "query_span": list(query_span),
    }


def validate_answer_vocabulary(tokenizer: Any) -> dict[str, int]:
    token_ids_by_word: dict[str, int] = {}
    for word in NUMBER_WORDS:
        token_ids_by_word[word] = one_token_id(
            tokenizer,
            f" {word}",
            f"answer word {word}",
        )
    if len(set(token_ids_by_word.values())) != len(token_ids_by_word):
        raise RuntimeError(f"answer words are not token-id unique: {token_ids_by_word}")
    return token_ids_by_word


@torch.inference_mode()
def score_answer(
    tokenizer: Any,
    query_output: Any,
    answer_token_ids: dict[str, int],
) -> dict[str, Any]:
    logits = query_output.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    gold_id = answer_token_ids["nine"]
    gold_log_probability = float(log_probs[gold_id].item())

    full_values, full_indices = torch.topk(log_probs, k=10, largest=True, sorted=True)
    full_rows = []
    for value, token_id in zip(full_values.tolist(), full_indices.tolist()):
        full_rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "log_probability": rounded(float(value)),
                "probability": rounded(math.exp(float(value))),
            }
        )
    best_non_gold = max(
        (row for row in full_rows if row["token_id"] != gold_id),
        key=lambda row: row["log_probability"],
    )

    candidate_rows = []
    for word, token_id in answer_token_ids.items():
        candidate_rows.append(
            {
                "word": word,
                "token_id": token_id,
                "log_probability": rounded(float(log_probs[token_id].item())),
                "probability": rounded(float(probabilities[token_id].item())),
            }
        )
    candidate_rows.sort(key=lambda row: row["log_probability"], reverse=True)
    strongest_wrong_candidate = next(row for row in candidate_rows if row["word"] != "nine")

    return {
        "gold_answer": "nine",
        "gold_token_id": gold_id,
        "gold_probability": rounded(math.exp(gold_log_probability)),
        "gold_nll": rounded(-gold_log_probability),
        "gold_ppl": rounded(math.exp(-gold_log_probability)),
        "full_vocab_margin": rounded(
            gold_log_probability - float(best_non_gold["log_probability"])
        ),
        "full_vocab_correct": int(full_indices[0].item()) == gold_id,
        "top_token_id": int(full_indices[0].item()),
        "top_token": full_rows[0]["token"],
        "top_probability": full_rows[0]["probability"],
        "strongest_non_gold": best_non_gold,
        "next_token_top10": full_rows,
        "candidate_margin": rounded(
            gold_log_probability - float(strongest_wrong_candidate["log_probability"])
        ),
        "candidate_correct": candidate_rows[0]["word"] == "nine",
        "candidate_prediction": candidate_rows[0]["word"],
        "strongest_wrong_candidate": strongest_wrong_candidate,
        "candidate_scores": candidate_rows,
    }


@torch.inference_mode()
def summarize_categories(
    model: Any,
    query_output: Any,
    captured_queries: dict[int, torch.Tensor],
    category_positions: dict[str, Sequence[int]],
    category_order: Sequence[str] = CATEGORY_ORDER,
) -> dict[str, Any]:
    layers = list(model.model.layers)
    cache = base.legacy_cache(query_output.past_key_values)
    key_length = int(cache[0][0].shape[2])

    head_mass: list[list[list[float]]] = []
    head_mean_attention: list[list[list[float]]] = []
    head_enrichment: list[list[list[float]]] = []
    head_mean_logit: list[list[list[float]]] = []
    head_category_max_logit: list[list[list[float]]] = []
    head_logsumexp_category: list[list[list[float]]] = []
    head_best_rank: list[list[list[int]]] = []
    head_entropy: list[list[float]] = []
    head_effective_tokens: list[list[float]] = []
    head_logsumexp: list[list[float]] = []
    head_max_logit: list[list[float]] = []

    for layer_index, layer in enumerate(layers):
        query = captured_queries[layer_index][0]
        key = cache[layer_index][0][0]
        num_heads = int(query.shape[0])
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        scale = float(getattr(layer.self_attn, "scaling", query.shape[-1] ** -0.5))
        indices = {
            category: torch.tensor(
                list(category_positions[category]),
                dtype=torch.long,
                device=key.device,
            )
            for category in category_order
        }

        layer_mass: list[list[float]] = []
        layer_mean_attention: list[list[float]] = []
        layer_enrichment: list[list[float]] = []
        layer_mean_logit: list[list[float]] = []
        layer_category_max_logit: list[list[float]] = []
        layer_logsumexp_category: list[list[float]] = []
        layer_best_rank: list[list[int]] = []
        layer_entropy: list[float] = []
        layer_effective_tokens: list[float] = []
        layer_logsumexp: list[float] = []
        layer_head_max_logit: list[float] = []

        for head_index in range(num_heads):
            kv_index = min(kv_heads - 1, head_index // groups)
            logits = torch.matmul(key[kv_index].float(), query[head_index].float()) * scale
            probabilities = torch.softmax(logits, dim=-1)
            entropy = float(
                (-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()).item()
            )

            category_mass: list[float] = []
            category_mean_attention: list[float] = []
            category_enrichment: list[float] = []
            category_mean_logit: list[float] = []
            category_max_logit: list[float] = []
            category_lse: list[float] = []
            category_rank: list[int] = []
            for category in category_order:
                category_indices = indices[category]
                count = int(category_indices.numel())
                if count == 0:
                    category_mass.append(0.0)
                    category_mean_attention.append(0.0)
                    category_enrichment.append(0.0)
                    category_mean_logit.append(0.0)
                    category_max_logit.append(0.0)
                    category_lse.append(0.0)
                    category_rank.append(key_length + 1)
                    continue
                selected_probabilities = probabilities.index_select(0, category_indices)
                selected_logits = logits.index_select(0, category_indices)
                mass = float(selected_probabilities.sum().item())
                maximum = selected_logits.max()
                category_mass.append(rounded(mass))
                category_mean_attention.append(rounded(mass / count))
                category_enrichment.append(rounded(mass * key_length / count))
                category_mean_logit.append(rounded(float(selected_logits.mean().item())))
                category_max_logit.append(rounded(float(maximum.item())))
                category_lse.append(rounded(float(torch.logsumexp(selected_logits, dim=-1).item())))
                category_rank.append(int((logits > maximum).sum().item()) + 1)

            layer_mass.append(category_mass)
            layer_mean_attention.append(category_mean_attention)
            layer_enrichment.append(category_enrichment)
            layer_mean_logit.append(category_mean_logit)
            layer_category_max_logit.append(category_max_logit)
            layer_logsumexp_category.append(category_lse)
            layer_best_rank.append(category_rank)
            layer_entropy.append(rounded(entropy))
            layer_effective_tokens.append(rounded(math.exp(entropy)))
            layer_logsumexp.append(rounded(float(torch.logsumexp(logits, dim=-1).item())))
            layer_head_max_logit.append(rounded(float(logits.max().item())))

        head_mass.append(layer_mass)
        head_mean_attention.append(layer_mean_attention)
        head_enrichment.append(layer_enrichment)
        head_mean_logit.append(layer_mean_logit)
        head_category_max_logit.append(layer_category_max_logit)
        head_logsumexp_category.append(layer_logsumexp_category)
        head_best_rank.append(layer_best_rank)
        head_entropy.append(layer_entropy)
        head_effective_tokens.append(layer_effective_tokens)
        head_logsumexp.append(layer_logsumexp)
        head_max_logit.append(layer_head_max_logit)

    return {
        "key_length": key_length,
        "num_layers": len(head_mass),
        "num_attention_heads": len(head_mass[0]) if head_mass else 0,
        "category_order": list(category_order),
        "head_category_mass": head_mass,
        "head_category_mean_attention": head_mean_attention,
        "head_category_enrichment": head_enrichment,
        "head_category_mean_logit": head_mean_logit,
        "head_category_max_logit": head_category_max_logit,
        "head_category_logsumexp": head_logsumexp_category,
        "head_category_best_rank": head_best_rank,
        "head_entropy": head_entropy,
        "head_effective_tokens": head_effective_tokens,
        "head_logsumexp": head_logsumexp,
        "head_max_logit": head_max_logit,
    }


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"prompt_ids", "category_positions"}
    }


def manifest_payload(
    args: argparse.Namespace,
    answer_token_ids: dict[str, int],
    cases: Sequence[dict[str, Any]],
    completed: Sequence[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "fixed300_age_distractor_qk_attention",
        "model_name_or_path": args.model_name_or_path,
        "total_tokens": args.total_tokens,
        "gold_evidence": GOLD_TEXT.rstrip(),
        "query": QUERY_TEXT.strip(),
        "gold_answer": "nine",
        "answer_token_ids": answer_token_ids,
        "number_words": list(NUMBER_WORDS),
        "category_order": list(CATEGORY_ORDER),
        "category_labels": CATEGORY_LABELS,
        "case_count": len(cases),
        "completed_count": len(completed),
        "elapsed_seconds": rounded(elapsed_seconds),
        "cases": [
            {
                "distractor_count": case["distractor_count"],
                "filler_count": case["filler_count"],
                "category_counts": case["category_counts"],
                "filler_gap_counts": case["filler_gap_counts"],
                "file": f"case_{case['distractor_count']:02d}.json",
            }
            for case in cases
        ],
        "completed": list(completed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-300-token age distractor QK/attention experiment for Qwen3-8B"
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-tokens", type=int, default=300)
    parser.add_argument("--distractor-counts", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--original-max-position-embeddings", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distractor_counts = sorted(
        {
            int(value.strip())
            for value in args.distractor_counts.split(",")
            if value.strip()
        }
    )
    if distractor_counts != list(range(10)):
        raise ValueError("This controlled experiment requires distractor counts exactly 0..9")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    answer_token_ids = validate_answer_vocabulary(tokenizer)
    cases = [build_case(tokenizer, count, args.total_tokens) for count in distractor_counts]

    design = {
        "schema_version": 1,
        "experiment": "fixed300_age_distractor_qk_attention",
        "total_tokens": args.total_tokens,
        "gold_evidence": GOLD_TEXT.rstrip(),
        "query": QUERY_TEXT.strip(),
        "gold_answer": "nine",
        "answer_token_ids": answer_token_ids,
        "category_order": list(CATEGORY_ORDER),
        "category_labels": CATEGORY_LABELS,
        "cases": [public_case(case) for case in cases],
    }
    write_json_atomic(output_dir / "design.json", design)
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return

    max_factor = base.rope_factor_for_length(
        args.total_tokens,
        args.original_max_position_embeddings,
    )
    model, model_tokenizer = base.load_model_and_tokenizer(
        args,
        args.total_tokens,
        max_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("Tokenizer changed between validation and model loading")
    del tokenizer
    tokenizer = model_tokenizer

    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        case_started = time.perf_counter()
        prompt = torch.tensor(case["prompt_ids"], dtype=torch.long).view(1, -1)
        base_cache, prefill_seconds = base.prefill_sequence(
            model,
            prompt[:, :-1],
            args.prefill_chunk_size,
        )
        query_cache = base.cache_from_legacy(base_cache)
        del base_cache
        query_output, captured_queries, query_seconds = attention_runner.capture_query_states(
            model,
            query_cache,
            prompt[:, -1:],
            args.total_tokens - 1,
        )
        answer = score_answer(tokenizer, query_output, answer_token_ids)
        attention = summarize_categories(
            model,
            query_output,
            captured_queries,
            case["category_positions"],
        )
        result = {
            "schema_version": 1,
            "experiment": "fixed300_age_distractor_qk_attention",
            "model_name_or_path": args.model_name_or_path,
            "case": public_case(case),
            "answer": answer,
            "attention": attention,
            "timing": {
                "prefill_seconds": rounded(prefill_seconds),
                "query_seconds": rounded(query_seconds),
                "case_seconds": rounded(time.perf_counter() - case_started),
            },
        }
        result_path = output_dir / f"case_{case['distractor_count']:02d}.json"
        write_json_atomic(result_path, result)
        completed.append(
            {
                "distractor_count": case["distractor_count"],
                "file": result_path.name,
                "gold_ppl": answer["gold_ppl"],
                "full_vocab_margin": answer["full_vocab_margin"],
                "full_vocab_correct": answer["full_vocab_correct"],
                "candidate_margin": answer["candidate_margin"],
                "candidate_correct": answer["candidate_correct"],
                "top_token": answer["top_token"],
                "candidate_prediction": answer["candidate_prediction"],
                "case_seconds": result["timing"]["case_seconds"],
            }
        )
        write_json_atomic(
            output_dir / "manifest.json",
            manifest_payload(
                args,
                answer_token_ids,
                cases,
                completed,
                time.perf_counter() - started,
            ),
        )
        print(
            json.dumps(
                completed[-1],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        del prompt, query_cache, query_output, captured_queries, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json_atomic(
        output_dir / "manifest.json",
        manifest_payload(
            args,
            answer_token_ids,
            cases,
            completed,
            time.perf_counter() - started,
        ),
    )


if __name__ == "__main__":
    main()
