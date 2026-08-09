from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_length_causal_mechanism_20260717 as length_runner
import run_local_rule_failure_boundary as base


FREQUENCY_BINS = {
    "common": (4.50, 5.00, 4.75),
    "medium": (3.50, 4.00, 3.75),
    "tail": (2.50, 3.00, 2.75),
}

STOP_WORDS = {
    "about", "above", "after", "again", "against", "almost", "along", "also",
    "among", "another", "around", "because", "before", "being", "below", "between",
    "both", "could", "does", "doing", "down", "during", "each", "either", "enough",
    "every", "first", "from", "further", "have", "having", "here", "hers", "himself",
    "into", "itself", "just", "least", "many", "might", "more", "most", "much",
    "must", "neither", "never", "other", "otherwise", "ours", "rather", "same",
    "should", "since", "some", "such", "than", "that", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "those", "through", "under",
    "until", "very", "what", "when", "where", "which", "while", "whom", "whose",
    "with", "within", "without", "would", "your", "yours",
}


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def stable_hash(seed: int, label: str, word: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{word}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def filler_vocabulary() -> set[str]:
    text = " ".join(base.FILLER_PARAGRAPHS).lower()
    words = set(re.findall(r"[a-z]+", text))
    words.update(
        {
            "answer", "code", "exactly", "explanation", "final", "follow", "leads",
            "only", "return", "rule", "start", "task", "transition", "transitions",
            "two", "verified", "bombard", "essen", "florida", "mexico", "scape", "sites",
            "technical", "treat",
        }
    )
    return words


def ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def find_all(values: Sequence[int], target: Sequence[int]) -> list[int]:
    if not target:
        return []
    return [
        start
        for start in range(len(values) - len(target) + 1)
        if list(values[start : start + len(target)]) == list(target)
    ]


def stable_single_token_word(tokenizer: Any, word: str) -> tuple[bool, int | None]:
    target = ids(tokenizer, " " + word)
    if len(target) != 1:
        return False, None
    token_id = target[0]
    contexts = (
        f"VERIFIED RULE 1: CODE: {word} LEADS TO CODE: {word}.\n",
        f"Start CODE: {word}\n",
        "FINAL CODE:" + " " + word,
    )
    for text in contexts:
        if token_id not in ids(tokenizer, text):
            return False, None
    return True, token_id


def select_words(tokenizer: Any, count: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    try:
        from wordfreq import top_n_list, zipf_frequency
    except ImportError as exc:
        raise RuntimeError("wordfreq is required: pip install wordfreq==3.1.1") from exc
    try:
        from nltk.corpus import wordnet as wn
    except (ImportError, LookupError) as exc:
        raise RuntimeError("NLTK WordNet is required: python -m nltk.downloader wordnet") from exc

    forbidden = filler_vocabulary() | STOP_WORDS
    concrete_noun_classes = {
        "noun.animal", "noun.artifact", "noun.body", "noun.food", "noun.location",
        "noun.object", "noun.plant", "noun.shape", "noun.substance",
    }
    candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in FREQUENCY_BINS}
    for word in top_n_list("en", 100_000):
        if not re.fullmatch(r"[a-z]{5,10}", word) or word in forbidden:
            continue
        if word.endswith(("ing", "edly", "ously")):
            continue
        synsets = wn.synsets(word)
        if not synsets or synsets[0].pos() != "n" or synsets[0].lexname() not in concrete_noun_classes:
            continue
        ok, token_id = stable_single_token_word(tokenizer, word)
        if not ok:
            continue
        frequency = float(zipf_frequency(word, "en"))
        for label, (lower, upper, target) in FREQUENCY_BINS.items():
            if lower <= frequency < upper:
                candidates[label].append(
                    {
                        "word": word,
                        "token_id": int(token_id),
                        "zipf": frequency,
                        "selection_distance": abs(frequency - target),
                        "tie_break": stable_hash(seed, label, word),
                    }
                )
                break

    selected: dict[str, list[dict[str, Any]]] = {}
    for label, rows in candidates.items():
        rows.sort(key=lambda row: (row["selection_distance"], row["tie_break"]))
        if len(rows) < count:
            raise RuntimeError(f"only {len(rows)} valid words in bin {label}; need {count}")
        selected[label] = [
            {key: value for key, value in row.items() if key not in {"selection_distance", "tie_break"}}
            for row in rows[:count]
        ]
    return selected


def chat_wrapper_without_answer_space(tokenizer: Any) -> tuple[list[int], list[int]]:
    marker = "CODEX_COMMON_TAIL_CONTENT_MARKER_20260721"
    kwargs = {
        "conversation": [{"role": "user", "content": marker}],
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        rendered = list(tokenizer.apply_chat_template(**kwargs, enable_thinking=False))
    except TypeError:
        rendered = list(tokenizer.apply_chat_template(**kwargs))
    marker_ids = ids(tokenizer, marker)
    start = length_runner.find_subsequence(rendered, marker_ids)
    return rendered[:start], rendered[start + len(marker_ids) :] + ids(tokenizer, "FINAL CODE:")


def rule_text(words: Sequence[str]) -> str:
    return (
        f"VERIFIED RULE 1: CODE: {words[0]} LEADS TO CODE: {words[1]}.\n"
        f"VERIFIED RULE 2: CODE: {words[1]} LEADS TO CODE: {words[2]}.\n"
    )


def query_text(start_word: str) -> str:
    return (
        "\nTWO-STEP RETRIEVAL TASK\n"
        f"Start CODE: {start_word}\n"
        "Follow exactly two VERIFIED RULE transitions.\n"
        "Return only the final CODE, with no explanation.\n"
    )


def build_pair(
    tokenizer: Any,
    wrapper_prefix: Sequence[int],
    wrapper_suffix: Sequence[int],
    words: Sequence[str],
    target_length: int,
    seed: int,
    placement: str,
    recent_gap: int,
) -> dict[str, Any]:
    filler = base.build_filler_ids(tokenizer, target_length, seed)
    evidence_body = list(filler)
    block = ids(tokenizer, rule_text(words))
    if len(block) >= target_length:
        raise ValueError(f"rule block ({len(block)}) does not fit body ({target_length})")
    if placement == "middle":
        block_start = (target_length - len(block)) // 2
    elif placement == "fixed_recent":
        block_start = target_length - len(block) - recent_gap
        if block_start < 0:
            raise ValueError(
                f"rule block ({len(block)}) plus recent gap ({recent_gap}) does not fit body ({target_length})"
            )
    else:
        raise ValueError(f"unknown placement: {placement}")
    evidence_body[block_start : block_start + len(block)] = block

    query_ids = ids(tokenizer, query_text(words[0]))
    evidence_prompt = list(wrapper_prefix) + evidence_body + query_ids + list(wrapper_suffix)
    control_prompt = list(wrapper_prefix) + filler + query_ids + list(wrapper_suffix)
    if len(evidence_prompt) != len(control_prompt):
        raise AssertionError("evidence/control prompts must have identical token length")

    absolute_start = len(wrapper_prefix) + block_start
    line1 = ids(tokenizer, f"VERIFIED RULE 1: CODE: {words[0]} LEADS TO CODE: {words[1]}.\n")
    line2 = ids(tokenizer, f"VERIFIED RULE 2: CODE: {words[1]} LEADS TO CODE: {words[2]}.\n")
    word_token_ids = [ids(tokenizer, " " + word)[0] for word in words]
    final_offsets = find_all(line2, [word_token_ids[2]])
    if not final_offsets:
        raise AssertionError(f"cannot locate final word {words[2]!r} in rule 2")
    code_positions: list[int] = []
    for token_id in set(word_token_ids):
        code_positions.extend(absolute_start + offset for offset in find_all(block, [token_id]))
    code_positions = sorted(set(code_positions))
    final_position = absolute_start + len(line1) + final_offsets[-1]
    return {
        "evidence_prompt": evidence_prompt,
        "control_prompt": control_prompt,
        "spans": {
            "rule1": (absolute_start, absolute_start + len(line1)),
            "rule2": (absolute_start + len(line1), absolute_start + len(block)),
            "code_positions": code_positions,
            "final_position": final_position,
        },
        "rule_block_tokens": len(block),
        "rule_start": absolute_start,
        "relative_distance": len(evidence_prompt) - 1 - final_position,
    }


def score_output(
    tokenizer: Any,
    output: Any,
    gold_token_id: int,
    candidate_token_ids: Sequence[int],
) -> dict[str, Any]:
    logits = output.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    candidate = torch.tensor(candidate_token_ids, dtype=torch.long, device=logits.device)
    candidate_values = log_probs[candidate]
    candidate_index = int(torch.argmax(candidate_values).item())
    gold_candidate_index = list(candidate_token_ids).index(gold_token_id)
    sorted_values, _ = torch.sort(candidate_values, descending=True)
    gold_logprob = float(log_probs[gold_token_id].item())
    greedy_token_id = int(torch.argmax(logits).item())
    return {
        "gold_logprob": gold_logprob,
        "gold_probability": math.exp(gold_logprob),
        "gold_ppl": math.exp(-gold_logprob),
        "candidate_correct": candidate_index == gold_candidate_index,
        "candidate_prediction_token_id": int(candidate_token_ids[candidate_index]),
        "candidate_margin": float(candidate_values[gold_candidate_index].item() - sorted_values[1].item())
        if candidate_index == gold_candidate_index and len(candidate_token_ids) > 1
        else float(candidate_values[gold_candidate_index].item() - sorted_values[0].item()),
        "greedy_correct": greedy_token_id == gold_token_id,
        "greedy_token_id": greedy_token_id,
        "greedy_token": tokenizer.decode([greedy_token_id], clean_up_tokenization_spaces=False),
    }


def span_mass(probabilities: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    return probabilities[:, span[0] : span[1]].sum(dim=1)


@torch.inference_mode()
def summarize_evidence_attention(
    model: Any,
    output: Any,
    captured_queries: dict[int, torch.Tensor],
    spans: dict[str, Any],
) -> dict[str, Any]:
    cache = base.legacy_cache(output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    top_count = max(1, int(math.ceil(0.02 * key_length)))
    top20_count = min(20, key_length)
    rule1 = tuple(spans["rule1"])
    rule2 = tuple(spans["rule2"])
    code_positions = torch.tensor(spans["code_positions"], dtype=torch.long)
    final_position = int(spans["final_position"])
    scale = float(model.model.layers[0].self_attn.scaling)
    layer_rows: list[dict[str, float]] = []

    for layer_index, layer_cache in enumerate(cache):
        keys = layer_cache[0][0]
        queries = captured_queries[layer_index][0]
        q_heads = int(queries.shape[0])
        kv_heads = int(keys.shape[0])
        group_size = q_heads // kv_heads
        metrics: dict[str, list[torch.Tensor]] = {
            "rule_mass": [], "code_mass": [], "final_mass": [], "both_rules_top2": [],
            "code_recall_top2": [], "final_rank_fraction": [], "final_logit": [],
            "final_cosine": [], "background_log_partition": [], "background_max_logit": [],
            "needle_log_odds": [], "residual_after_top20_mass": [],
        }
        local_code_positions = code_positions.to(keys.device)
        for kv_index in range(kv_heads):
            first_head = kv_index * group_size
            q = queries[first_head : first_head + group_size].float()
            k = keys[kv_index].float()
            logits = torch.matmul(q, k.transpose(0, 1)) * scale
            probabilities = torch.softmax(logits, dim=1)
            line1_mass = span_mass(probabilities, rule1)
            line2_mass = span_mass(probabilities, rule2)
            code_mass = probabilities[:, local_code_positions].sum(dim=1)
            final_mass = probabilities[:, final_position]
            top_indices = torch.topk(logits, k=top_count, dim=1).indices
            top20_values = torch.topk(probabilities, k=top20_count, dim=1).values
            hit1 = ((top_indices >= rule1[0]) & (top_indices < rule1[1])).any(dim=1)
            hit2 = ((top_indices >= rule2[0]) & (top_indices < rule2[1])).any(dim=1)
            code_hits = (top_indices.unsqueeze(2) == local_code_positions.view(1, 1, -1)).any(dim=1)
            final_logits = logits[:, final_position]
            rank = 1 + (logits > final_logits.unsqueeze(1)).sum(dim=1)
            final_key = k[final_position]
            cosine = torch.nn.functional.cosine_similarity(q, final_key.view(1, -1), dim=1)
            masked = logits.clone()
            masked[:, rule1[0] : rule1[1]] = -torch.inf
            masked[:, rule2[0] : rule2[1]] = -torch.inf
            background_lse = torch.logsumexp(masked, dim=1)
            evidence_lse = torch.logsumexp(
                torch.cat((logits[:, rule1[0] : rule1[1]], logits[:, rule2[0] : rule2[1]]), dim=1),
                dim=1,
            )
            metrics["rule_mass"].append(line1_mass + line2_mass)
            metrics["code_mass"].append(code_mass)
            metrics["final_mass"].append(final_mass)
            metrics["both_rules_top2"].append((hit1 & hit2).float())
            metrics["code_recall_top2"].append(code_hits.float().mean(dim=1))
            metrics["final_rank_fraction"].append(rank.float() / key_length)
            metrics["final_logit"].append(final_logits)
            metrics["final_cosine"].append(cosine)
            metrics["background_log_partition"].append(background_lse)
            metrics["background_max_logit"].append(torch.max(masked, dim=1).values)
            metrics["needle_log_odds"].append(evidence_lse - background_lse)
            metrics["residual_after_top20_mass"].append(1.0 - top20_values.sum(dim=1))

        layer_rows.append(
            {
                name: float(torch.cat(values).mean().item())
                for name, values in metrics.items()
            }
        )

    return {
        "key_length": key_length,
        "top2_count": top_count,
        "layer_mean": layer_rows,
        "model_mean": {
            name: sum(row[name] for row in layer_rows) / len(layer_rows)
            for name in layer_rows[0]
        },
    }


def release_cuda(*objects: Any) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_prompt(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    gold_token_id: int,
    candidate_token_ids: Sequence[int],
    chunk_size: int,
    spans: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, float]]:
    prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
    prefix = prompt[:, :-1]
    last = prompt[:, -1:]
    cache, prefill_seconds = base.prefill_sequence(model, prefix, chunk_size)
    if spans is None:
        started = time.perf_counter()
        with torch.inference_mode():
            output = base.forward_with_cache(
                model,
                last.to(base.input_device(model)),
                base.cache_from_legacy(cache),
                int(prefix.shape[1]),
            )
        query_seconds = time.perf_counter() - started
        captured = None
    else:
        output, captured, query_seconds = attention_runner.capture_query_states(
            model, base.cache_from_legacy(cache), last, int(prefix.shape[1])
        )
    scores = score_output(tokenizer, output, gold_token_id, candidate_token_ids)
    attention = summarize_evidence_attention(model, output, captured, spans) if spans is not None else None
    del output, cache, captured, prompt, prefix, last
    release_cuda()
    return scores, attention, {"prefill_seconds": prefill_seconds, "query_seconds": query_seconds}


def assign_shards(specs: Sequence[dict[str, Any]], shard_count: int) -> dict[str, int]:
    loads = [0] * shard_count
    assignment: dict[str, int] = {}
    for spec in sorted(specs, key=lambda item: (-item["length"], item["case_id"])):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assignment[spec["case_id"]] = shard
        loads[shard] += int(spec["length"])
    return assignment


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(json.loads(line)["case_id"])
    return result


def build_specs(
    words: dict[str, list[dict[str, Any]]], lengths: Sequence[int], samples_per_bin: int, placement: str
) -> list[dict[str, Any]]:
    specs = []
    for sample_index in range(samples_per_bin):
        for label, rows in words.items():
            chain = [rows[(sample_index + offset) % len(rows)] for offset in range(3)]
            for length in lengths:
                specs.append(
                    {
                        "case_id": f"{placement}_{label}_s{sample_index:02d}_n{length}",
                        "bin": label,
                        "sample_index": sample_index,
                        "length": length,
                        "chain": chain,
                    }
                )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure common-vs-tail needle failure boundaries.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="8192,32768,65536,127500")
    parser.add_argument("--samples_per_bin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--placement", choices=("fixed_recent", "middle"), default="fixed_recent")
    parser.add_argument("--recent_gap", type=int, default=256)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_label", default="shard0")
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "auto"), default="float16")
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--global_max_position", type=int, default=130000)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    words = select_words(tokenizer, args.samples_per_bin, args.seed)
    lengths = parse_ints(args.lengths)
    wrapper_prefix, wrapper_suffix = chat_wrapper_without_answer_space(tokenizer)
    specs = build_specs(words, lengths, args.samples_per_bin, args.placement)
    assignments = assign_shards(specs, args.num_shards)
    design = {
        "definition": "wordfreq English Zipf frequency; all code words are one leading-space Qwen token",
        "frequency_bins": FREQUENCY_BINS,
        "words": words,
        "lengths": lengths,
        "samples_per_bin": args.samples_per_bin,
        "placement": args.placement,
        "recent_gap": args.recent_gap,
        "cases": len(specs),
        "paired_conditions": ["evidence", "matched_no_evidence"],
        "answer_boundary": "FINAL CODE:<leading-space-token>",
    }
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return

    row_path = output_dir / f"rows_{args.shard_label}.jsonl"
    completed = existing_case_ids(row_path)
    manifest_path = output_dir / f"design_{args.shard_label}.json"
    manifest_path.write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")

    max_factor = base.rope_factor_for_length(args.global_max_position, args.original_max_position_embeddings)
    model, tokenizer = base.load_model_and_tokenizer(args, args.global_max_position, max_factor)
    selected_specs = [
        spec for spec in specs
        if assignments[spec["case_id"]] == args.shard_index and spec["case_id"] not in completed
    ]
    for case_number, spec in enumerate(selected_specs, start=1):
        chain = spec["chain"]
        chain_words = [row["word"] for row in chain]
        pair = build_pair(
            tokenizer,
            wrapper_prefix,
            wrapper_suffix,
            chain_words,
            spec["length"],
            args.seed + spec["sample_index"],
            args.placement,
            args.recent_gap,
        )
        candidate_rows = words[spec["bin"]]
        candidate_ids = [int(row["token_id"]) for row in candidate_rows]
        gold_token_id = int(chain[2]["token_id"])
        started = time.perf_counter()
        evidence, attention, evidence_timing = run_prompt(
            model,
            tokenizer,
            pair["evidence_prompt"],
            gold_token_id,
            candidate_ids,
            args.prefill_chunk_size,
            pair["spans"],
        )
        control, _, control_timing = run_prompt(
            model,
            tokenizer,
            pair["control_prompt"],
            gold_token_id,
            candidate_ids,
            args.prefill_chunk_size,
            None,
        )
        evidence["candidate_prediction"] = next(
            row["word"] for row in candidate_rows
            if row["token_id"] == evidence["candidate_prediction_token_id"]
        )
        control["candidate_prediction"] = next(
            row["word"] for row in candidate_rows
            if row["token_id"] == control["candidate_prediction_token_id"]
        )
        row = {
            "case_id": spec["case_id"],
            "bin": spec["bin"],
            "sample_index": spec["sample_index"],
            "target_context_tokens": spec["length"],
            "placement": args.placement,
            "prompt_tokens": len(pair["evidence_prompt"]),
            "chain": chain,
            "gold_word": chain_words[2],
            "rule_block_tokens": pair["rule_block_tokens"],
            "rule_start": pair["rule_start"],
            "relative_distance": pair["relative_distance"],
            "evidence": evidence,
            "control": control,
            "evidence_lift_nats": evidence["gold_logprob"] - control["gold_logprob"],
            "evidence_probability_ratio": math.exp(evidence["gold_logprob"] - control["gold_logprob"]),
            "attention": attention,
            "timing": {
                "evidence": evidence_timing,
                "control": control_timing,
                "total_seconds": time.perf_counter() - started,
            },
        }
        append_jsonl(row_path, row)
        print(
            json.dumps(
                {
                    "progress": f"{case_number}/{len(selected_specs)}",
                    "case_id": spec["case_id"],
                    "gold_ppl": round(evidence["gold_ppl"], 4),
                    "lift": round(row["evidence_lift_nats"], 4),
                    "candidate_correct": evidence["candidate_correct"],
                    "seconds": round(row["timing"]["total_seconds"], 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    (output_dir / f"{args.shard_label}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")


if __name__ == "__main__":
    main()
