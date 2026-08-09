from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_local_rule_failure_boundary as base
import run_semantic_common_tail_pilot_8b as semantic_base


CONCEPTS = semantic_base.CONCEPTS
LABELS = semantic_base.LABELS
CONDITIONS = ("plain", "distractor", "conflict")
PLACEMENTS = ("remote", "fixed_recent")

# Each item is a near semantic neighbor of the concept at the same index.  It is
# related enough to compete for attention, but it is not a contradictory claim.
NEAR_NEIGHBORS = {
    "horse.n.01": "a sturdy donkey used to carry loads on rural paths",
    "mongoose.n.01": "a slender ferret that hunts small burrowing animals",
    "sword.n.01": "a short utility knife used for cutting rope and leather",
    "scraper.n.01": "a stiff wire brush used to clean residue from a surface",
    "bottle.n.01": "a wide-mouthed jar used to preserve dry food",
    "toolbox.n.01": "a portable suitcase used to carry clothes while traveling",
    "salt.n.01": "a crystalline sugar used to sweeten food and drinks",
    "gypsum.n.01": "a soft white chalk used for writing on a board",
}

PLAIN_CONTROLS = {
    "horse.n.01": "a municipal calendar listing street-cleaning dates",
    "mongoose.n.01": "an office memo describing printer maintenance",
    "sword.n.01": "a train timetable for weekday commuter service",
    "scraper.n.01": "a receipt recording the purchase of desk lamps",
    "bottle.n.01": "a weather table listing monthly rainfall totals",
    "toolbox.n.01": "a library notice about revised opening hours",
    "salt.n.01": "a meeting agenda for the regional planning committee",
    "gypsum.n.01": "a parking permit issued for a delivery vehicle",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(mean(math.log(max(value, 1e-300)) for value in values))


def wrong_labels(mapping: dict[str, str]) -> dict[str, str]:
    ordered = [mapping[concept["concept_id"]] for concept in CONCEPTS]
    return {
        concept["concept_id"]: ordered[(index + 1) % len(ordered)]
        for index, concept in enumerate(CONCEPTS)
    }


def line_with_label(
    tokenizer: Any,
    prefix: str,
    label: str,
    suffix: str = ".\n",
) -> tuple[list[int], int]:
    prefix_ids = encode(tokenizer, prefix)
    label_ids = encode(tokenizer, " " + label)
    if len(label_ids) != 1:
        raise AssertionError(f"label must be one token: {label!r} -> {label_ids}")
    return prefix_ids + label_ids + encode(tokenizer, suffix), len(prefix_ids)


def build_special_block(
    tokenizer: Any,
    condition: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    wrong = wrong_labels(mapping)
    token_ids = encode(
        tokenizer,
        "\nUNVERIFIED ARCHIVE BLOCK\n"
        "These archived annotations are not part of the VERIFIED REFERENCE CATALOG.\n",
    )
    concept_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    label_positions: dict[str, list[int]] = defaultdict(list)
    for index, concept in enumerate(CONCEPTS, start=1):
        concept_id = concept["concept_id"]
        if condition == "plain":
            description = PLAIN_CONTROLS[concept_id]
        elif condition == "distractor":
            description = NEAR_NEIGHBORS[concept_id]
        else:
            description = concept["query_clue"]
        line, label_offset = line_with_label(
            tokenizer,
            f"ARCHIVE ITEM {index}: {description}. ARCHIVED LABEL:",
            wrong[concept_id],
        )
        start = len(token_ids)
        token_ids.extend(line)
        concept_spans[concept_id].append((start, len(token_ids)))
        label_positions[concept_id].append(start + label_offset)
    token_ids.extend(encode(tokenizer, "END UNVERIFIED ARCHIVE BLOCK\n"))
    return {
        "token_ids": token_ids,
        "concept_spans": dict(concept_spans),
        "label_positions": dict(label_positions),
        "wrong_labels": wrong,
    }


def build_condition_filler(
    tokenizer: Any,
    target_length: int,
    seed: int,
    condition: str,
    mapping: dict[str, str],
    period: int,
    block_budget: int,
) -> dict[str, Any]:
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    filler = base.build_filler_ids(tokenizer, target_length, seed) if target_length else []
    block = build_special_block(tokenizer, condition, mapping)
    if len(block["token_ids"]) > block_budget:
        raise AssertionError("special block does not fit the shared block budget")
    all_spans: list[tuple[int, int]] = []
    concept_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    label_positions: dict[str, list[int]] = defaultdict(list)
    if target_length >= block_budget:
        # A whole reserved block is overwritten at identical offsets in all
        # conditions.  Unused reserved tokens stay as neutral prose, keeping the
        # total context length and evidence/query positions exactly matched.
        first = max(0, period // 2 - block_budget // 2)
        starts = list(range(first, target_length - block_budget + 1, period))
        if not starts:
            starts = [(target_length - block_budget) // 2]
        for start in starts:
            content = block["token_ids"]
            filler[start : start + len(content)] = content
            all_spans.append((start, start + len(content)))
            for concept_id, spans in block["concept_spans"].items():
                concept_spans[concept_id].extend(
                    (start + span_start, start + span_end) for span_start, span_end in spans
                )
            for concept_id, positions in block["label_positions"].items():
                label_positions[concept_id].extend(start + position for position in positions)
    return {
        "token_ids": filler,
        "all_spans": all_spans,
        "concept_spans": dict(concept_spans),
        "label_positions": dict(label_positions),
        "wrong_labels": block["wrong_labels"],
        "block_content_tokens": len(block["token_ids"]),
        "block_budget": block_budget,
    }


def shift_spans(spans: Iterable[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
    return [(offset + start, offset + end) for start, end in spans]


def build_shared_prefix(
    tokenizer: Any,
    wrapper_prefix: Sequence[int],
    catalog: dict[str, Any],
    filler_length: int,
    condition: str,
    placement: str,
    seed: int,
    period: int,
    block_budget: int,
    recent_gap: int,
) -> dict[str, Any]:
    filler = build_condition_filler(
        tokenizer, filler_length, seed, condition, catalog["mapping"], period, block_budget
    )
    gap_ids = base.build_filler_ids(tokenizer, recent_gap, seed + 17) if recent_gap else []
    if placement == "remote":
        catalog_start = len(wrapper_prefix)
        filler_start = catalog_start + len(catalog["token_ids"])
        body = list(catalog["token_ids"]) + filler["token_ids"] + gap_ids
    elif placement == "fixed_recent":
        filler_start = len(wrapper_prefix)
        catalog_start = filler_start + len(filler["token_ids"])
        body = filler["token_ids"] + list(catalog["token_ids"]) + gap_ids
    else:
        raise ValueError(f"unknown placement: {placement}")

    prefix_ids = list(wrapper_prefix) + body
    catalog_spans: dict[str, dict[str, Any]] = {}
    for concept_id, relative in catalog["spans"].items():
        catalog_spans[concept_id] = {
            **relative,
            "entry": tuple(catalog_start + value for value in relative["entry"]),
            "definition": tuple(catalog_start + value for value in relative["definition"]),
            "label_span": tuple(catalog_start + value for value in relative["label_span"]),
        }
    special_by_concept = {
        concept_id: shift_spans(spans, filler_start)
        for concept_id, spans in filler["concept_spans"].items()
    }
    special_labels = {
        concept_id: [filler_start + position for position in positions]
        for concept_id, positions in filler["label_positions"].items()
    }
    return {
        "prefix_ids": prefix_ids,
        "catalog_spans": catalog_spans,
        "all_catalog_spans": [value["entry"] for value in catalog_spans.values()],
        "special_spans": shift_spans(filler["all_spans"], filler_start),
        "special_by_concept": special_by_concept,
        "special_label_positions": special_labels,
        "wrong_labels": filler["wrong_labels"],
        "catalog_start": catalog_start,
        "filler_start": filler_start,
        "actual_prefix_tokens": len(prefix_ids),
        "special_occurrences": len(filler["all_spans"]),
        "special_content_tokens": sum(end - start for start, end in filler["all_spans"]),
    }


def query_suffix(tokenizer: Any, wrapper_suffix: Sequence[int], concept: dict[str, Any]) -> list[int]:
    text = (
        "\n\nRETRIEVAL QUESTION\n"
        "Use only the VERIFIED REFERENCE CATALOG. Ignore every UNVERIFIED ARCHIVE annotation.\n"
        f'Which verified entry matches this clue: "{concept["query_clue"]}"?\n'
        "Return only its one-letter ARBITRARY LABEL, with no explanation."
    )
    return encode(tokenizer, text) + list(wrapper_suffix)


def mask_from_spans(length: int, spans: Iterable[tuple[int, int]], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for start, end in spans:
        if start < 0 or end > length or start >= end:
            raise AssertionError(f"invalid span {(start, end)} for key length {length}")
        mask[start:end] = True
    return mask


def safe_lse(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if not bool(mask.any().item()):
        return None
    return torch.logsumexp(values[:, mask], dim=1)


@torch.inference_mode()
def summarize_attention(
    model: Any,
    output: Any,
    captured_queries: dict[int, torch.Tensor],
    target_entry: tuple[int, int],
    target_label_position: int,
    all_catalog_spans: Sequence[tuple[int, int]],
    special_spans: Sequence[tuple[int, int]],
    target_special_spans: Sequence[tuple[int, int]],
    target_special_label_positions: Sequence[int],
    save_head_rows: bool,
) -> dict[str, Any]:
    cache = base.legacy_cache(output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    top20_count = min(20, key_length)
    top2_count = max(1, int(math.ceil(0.02 * key_length)))
    layer_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []

    for layer_index, layer_cache in enumerate(cache):
        keys = layer_cache[0][0]
        queries = captured_queries[layer_index][0]
        q_heads = int(queries.shape[0])
        kv_heads = int(keys.shape[0])
        group_size = q_heads // kv_heads
        scale = float(model.model.layers[layer_index].self_attn.scaling)
        device = keys.device
        evidence_mask = mask_from_spans(key_length, [target_entry], device)
        catalog_mask = mask_from_spans(key_length, all_catalog_spans, device)
        other_catalog_mask = catalog_mask & ~evidence_mask
        special_mask = mask_from_spans(key_length, special_spans, device)
        target_special_mask = mask_from_spans(key_length, target_special_spans, device)
        non_evidence_mask = ~evidence_mask
        ordinary_background_mask = ~(evidence_mask | special_mask | other_catalog_mask)
        layer_heads: list[dict[str, float]] = []

        for kv_index in range(kv_heads):
            first_head = kv_index * group_size
            q = queries[first_head : first_head + group_size].float()
            k = keys[kv_index].float()
            logits = torch.matmul(q, k.transpose(0, 1)) * scale
            probs = torch.softmax(logits, dim=1)
            evidence_lse = torch.logsumexp(logits[:, evidence_mask], dim=1)
            non_evidence_lse = torch.logsumexp(logits[:, non_evidence_mask], dim=1)
            other_catalog_lse = safe_lse(logits, other_catalog_mask)
            special_lse = safe_lse(logits, special_mask)
            target_special_lse = safe_lse(logits, target_special_mask)
            ordinary_background_lse = safe_lse(logits, ordinary_background_mask)
            evidence_mass = probs[:, evidence_mask].sum(dim=1)
            other_catalog_mass = probs[:, other_catalog_mask].sum(dim=1)
            special_mass = probs[:, special_mask].sum(dim=1)
            target_special_mass = probs[:, target_special_mask].sum(dim=1)
            ordinary_background_mass = probs[:, ordinary_background_mask].sum(dim=1)
            label_logits = logits[:, target_label_position]
            label_rank = 1 + (logits > label_logits.unsqueeze(1)).sum(dim=1)
            top20 = torch.topk(logits, k=top20_count, dim=1).indices
            top2 = torch.topk(logits, k=top2_count, dim=1).indices
            evidence_hit20 = evidence_mask[top20].any(dim=1)
            evidence_hit2 = evidence_mask[top2].any(dim=1)
            top20_mass = torch.topk(probs, k=top20_count, dim=1).values.sum(dim=1)
            if target_special_label_positions:
                special_label_mass = probs[:, list(target_special_label_positions)].sum(dim=1)
            else:
                special_label_mass = torch.zeros(q.shape[0], device=device)

            for local_index in range(group_size):
                row = {
                    "layer": float(layer_index),
                    "head": float(first_head + local_index),
                    "evidence_logsumexp": float(evidence_lse[local_index].item()),
                    "non_evidence_logsumexp": float(non_evidence_lse[local_index].item()),
                    "evidence_vs_non_evidence": float(
                        (evidence_lse[local_index] - non_evidence_lse[local_index]).item()
                    ),
                    "other_catalog_logsumexp": float(other_catalog_lse[local_index].item())
                    if other_catalog_lse is not None
                    else 0.0,
                    "special_logsumexp": float(special_lse[local_index].item())
                    if special_lse is not None
                    else 0.0,
                    "target_special_logsumexp": float(target_special_lse[local_index].item())
                    if target_special_lse is not None
                    else 0.0,
                    "ordinary_background_logsumexp": float(ordinary_background_lse[local_index].item())
                    if ordinary_background_lse is not None
                    else 0.0,
                    "evidence_mass": float(evidence_mass[local_index].item()),
                    "other_catalog_mass": float(other_catalog_mass[local_index].item()),
                    "special_mass": float(special_mass[local_index].item()),
                    "target_special_mass": float(target_special_mass[local_index].item()),
                    "ordinary_background_mass": float(ordinary_background_mass[local_index].item()),
                    "evidence_label_mass": float(probs[local_index, target_label_position].item()),
                    "special_label_mass": float(special_label_mass[local_index].item()),
                    "evidence_label_rank_fraction": float(label_rank[local_index].item() / key_length),
                    "evidence_hit_top20": float(evidence_hit20[local_index].item()),
                    "evidence_hit_top2pct": float(evidence_hit2[local_index].item()),
                    "outside_top20_mass": float((1.0 - top20_mass[local_index]).item()),
                }
                if target_special_lse is not None:
                    row["evidence_vs_target_special"] = float(
                        (evidence_lse[local_index] - target_special_lse[local_index]).item()
                    )
                else:
                    row["evidence_vs_target_special"] = 0.0
                layer_heads.append(row)
                if save_head_rows:
                    head_rows.append(row)

        metric_names = [name for name in layer_heads[0] if name not in {"layer", "head"}]
        layer_rows.append(
            {"layer": layer_index, **{name: mean(row[name] for row in layer_heads) for name in metric_names}}
        )

    metric_names = [name for name in layer_rows[0] if name != "layer"]
    result = {
        "key_length": key_length,
        "top20_count": top20_count,
        "top2_count": top2_count,
        "model_mean": {name: mean(row[name] for row in layer_rows) for name in metric_names},
        "layer_mean": layer_rows,
    }
    if save_head_rows:
        result["head_rows"] = head_rows
    return result


def score_logits(
    tokenizer: Any,
    logits: torch.Tensor,
    gold_token_id: int,
    candidate_ids: Sequence[int],
    condition_label_id: int,
) -> dict[str, Any]:
    values = logits[0, -1].float()
    log_probs = torch.log_softmax(values, dim=-1)
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=values.device)
    candidate_values = log_probs[candidate_tensor]
    gold_index = list(candidate_ids).index(gold_token_id)
    other = torch.cat((candidate_values[:gold_index], candidate_values[gold_index + 1 :]))
    prediction_index = int(torch.argmax(candidate_values).item())
    gold_logprob = float(log_probs[gold_token_id].item())
    greedy_id = int(torch.argmax(values).item())
    return {
        "gold_logprob": gold_logprob,
        "gold_probability": math.exp(gold_logprob),
        "gold_ppl": math.exp(-gold_logprob),
        "candidate_correct": prediction_index == gold_index,
        "candidate_prediction": tokenizer.decode([int(candidate_ids[prediction_index])]).strip(),
        "candidate_margin": float((candidate_values[gold_index] - other.max()).item()),
        "condition_label_margin": float(
            (log_probs[gold_token_id] - log_probs[condition_label_id]).item()
        ),
        "greedy_correct": greedy_id == gold_token_id,
        "greedy_token_id": greedy_id,
        "greedy_token": tokenizer.decode([greedy_id], clean_up_tokenization_spaces=False),
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(args: argparse.Namespace, maximum_length: int) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    factor = base.rope_factor_for_length(maximum_length, args.original_max_position_embeddings)
    if factor > 1.0:
        config.max_position_embeddings = maximum_length
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(args.original_max_position_embeddings),
        }
    kwargs = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, {
        "model": args.model_name_or_path,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(base.input_device(model)),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_position_embeddings": maximum_length,
        "rope_factor": factor,
        "attn_implementation": args.attn_implementation,
    }


def extend_cache(
    model: Any,
    cache: Any,
    token_ids: Sequence[int],
    start_position: int,
    chunk_size: int,
) -> float:
    started = time.perf_counter()
    past_len = start_position
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        chunk = torch.tensor(chunk_ids, dtype=torch.long, device=base.input_device(model)).view(1, -1)
        with torch.inference_mode():
            output = base.forward_with_cache(model, chunk, cache, past_len)
        cache = output.past_key_values
        past_len += len(chunk_ids)
        del output, chunk
    base.synchronize()
    return time.perf_counter() - started


def run_concept_branch(
    model: Any,
    tokenizer: Any,
    cache: Any,
    base_length: int,
    suffix_ids: Sequence[int],
    gold_token_id: int,
    candidate_ids: Sequence[int],
    condition_label_id: int,
    chunk_size: int,
    span_data: dict[str, Any],
    concept_id: str,
    save_head_rows: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    if len(suffix_ids) < 2:
        raise ValueError("query suffix is too short")
    extend_seconds = extend_cache(model, cache, suffix_ids[:-1], base_length, chunk_size)
    prompt_len_minus_one = base_length + len(suffix_ids) - 1
    last_id = torch.tensor([suffix_ids[-1]], dtype=torch.long).view(1, 1)
    output, captured, query_seconds = attention_runner.capture_query_states(
        model, cache, last_id, prompt_len_minus_one
    )
    relevant = span_data["catalog_spans"][concept_id]
    scores = score_logits(
        tokenizer, output.logits, gold_token_id, candidate_ids, condition_label_id
    )
    attention = summarize_attention(
        model=model,
        output=output,
        captured_queries=captured,
        target_entry=tuple(relevant["entry"]),
        target_label_position=int(relevant["label_span"][0]),
        all_catalog_spans=span_data["all_catalog_spans"],
        special_spans=span_data["special_spans"],
        target_special_spans=span_data["special_by_concept"].get(concept_id, []),
        target_special_label_positions=span_data["special_label_positions"].get(concept_id, []),
        save_head_rows=save_head_rows,
    )
    output.past_key_values.crop(base_length)
    if int(output.past_key_values.get_seq_length()) != base_length:
        raise AssertionError("failed to restore the shared prefix cache")
    del output, captured, last_id
    release_cuda()
    return scores, attention, {"extend_seconds": extend_seconds, "query_seconds": query_seconds}


def completed_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str, int, str]]:
    return {
        (row["placement"], row["condition"], int(row["filler_length"]), row["concept_id"])
        for row in rows
    }


def run_group(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    catalog: dict[str, Any],
    wrapper_prefix: Sequence[int],
    wrapper_suffix: Sequence[int],
    output_dir: Path,
    placement: str,
    condition: str,
    filler_length: int,
    period: int,
    block_budget: int,
) -> None:
    rows_path = output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path)
    done = completed_keys(existing)
    selected_concepts = set(parse_csv(args.concept_ids))
    concepts = [
        concept
        for concept in CONCEPTS
        if not selected_concepts or concept["concept_id"] in selected_concepts
        if (placement, condition, filler_length, concept["concept_id"]) not in done
    ]
    if not concepts:
        return
    span_data = build_shared_prefix(
        tokenizer=tokenizer,
        wrapper_prefix=wrapper_prefix,
        catalog=catalog,
        filler_length=filler_length,
        condition=condition,
        placement=placement,
        seed=args.seed,
        period=period,
        block_budget=block_budget,
        recent_gap=args.recent_gap,
    )
    prefix = torch.tensor(span_data["prefix_ids"], dtype=torch.long).view(1, -1)
    cache, prefill_seconds = base.prefill_sequence(model, prefix, args.prefill_chunk_size)
    cache = base.cache_from_legacy(cache)
    # Labels inside catalog prose carry a leading-space token.  The assistant
    # generation prompt already ends with newlines, so the answer's first token
    # is the no-leading-space form.  Mixing these two token IDs makes PPL
    # meaningless even when relative candidate ranking looks plausible.
    candidate_ids = [encode(tokenizer, label)[0] for label in LABELS]
    print(
        f"prefill placement={placement} condition={condition} filler={filler_length} "
        f"prefix={len(span_data['prefix_ids'])} seconds={prefill_seconds:.3f}",
        flush=True,
    )
    for concept in concepts:
        concept_id = concept["concept_id"]
        relevant = span_data["catalog_spans"][concept_id]
        suffix = query_suffix(tokenizer, wrapper_suffix, concept)
        gold_token_id = encode(tokenizer, relevant["label"])[0]
        wrong_label = span_data["wrong_labels"][concept_id]
        wrong_token_id = encode(tokenizer, wrong_label)[0]
        scores, attention, timings = run_concept_branch(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            base_length=len(span_data["prefix_ids"]),
            suffix_ids=suffix,
            gold_token_id=gold_token_id,
            candidate_ids=candidate_ids,
            condition_label_id=wrong_token_id,
            chunk_size=args.prefill_chunk_size,
            span_data=span_data,
            concept_id=concept_id,
            save_head_rows=args.save_head_rows,
        )
        row = {
            "placement": placement,
            "condition": condition,
            "filler_length": filler_length,
            "concept_id": concept_id,
            "bin": concept["bin"],
            "pair": concept["pair"],
            "lemma": concept["lemma"],
            "query_clue": concept["query_clue"],
            "gold_label": relevant["label"],
            "condition_label": wrong_label,
            "actual_prefix_tokens": len(span_data["prefix_ids"]),
            "actual_prompt_tokens": len(span_data["prefix_ids"]) + len(suffix),
            "evidence_query_distance": len(span_data["prefix_ids"]) + len(suffix) - int(relevant["entry"][1]),
            "special_occurrences": span_data["special_occurrences"],
            "special_content_tokens": span_data["special_content_tokens"],
            "scores": scores,
            "attention": attention,
            "timings": {"shared_prefill_seconds": prefill_seconds, **timings},
        }
        append_jsonl(rows_path, row)
        print(
            f"  {concept['lemma']:<9} candidate={int(scores['candidate_correct'])} "
            f"greedy={int(scores['greedy_correct'])} margin={scores['candidate_margin']:+.3f} "
            f"ppl={scores['gold_ppl']:.3f} mass={attention['model_mean']['evidence_mass']:.6f}",
            flush=True,
        )
    del cache, prefix
    release_cuda()


def aggregate(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attention_keys = (
        "evidence_mass",
        "special_mass",
        "target_special_mass",
        "ordinary_background_mass",
        "evidence_logsumexp",
        "non_evidence_logsumexp",
        "special_logsumexp",
        "target_special_logsumexp",
        "ordinary_background_logsumexp",
        "evidence_vs_non_evidence",
        "evidence_vs_target_special",
        "evidence_hit_top20",
        "evidence_hit_top2pct",
        "outside_top20_mass",
    )
    result = {
        "n": len(group),
        "candidate_accuracy": mean(float(row["scores"]["candidate_correct"]) for row in group),
        "greedy_accuracy": mean(float(row["scores"]["greedy_correct"]) for row in group),
        "gold_ppl_geomean": geometric_mean(float(row["scores"]["gold_ppl"]) for row in group),
        "candidate_margin": mean(float(row["scores"]["candidate_margin"]) for row in group),
        "condition_label_margin": mean(float(row["scores"]["condition_label_margin"]) for row in group),
    }
    for key in attention_keys:
        result[key] = mean(float(row["attention"]["model_mean"][key]) for row in group)
    return result


def grouped_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["placement"], row["condition"], int(row["filler_length"]))].append(row)
    summaries = []
    for (placement, condition, filler_length), group in sorted(groups.items()):
        summaries.append(
            {
                "placement": placement,
                "condition": condition,
                "filler_length": filler_length,
                **aggregate(group),
            }
        )
    by_key = {(row["placement"], row["condition"], row["filler_length"]): row for row in summaries}
    for row in summaries:
        plain = by_key.get((row["placement"], "plain", row["filler_length"]))
        if plain is None:
            continue
        delta = row["non_evidence_logsumexp"] - plain["non_evidence_logsumexp"]
        row["delta_non_evidence_logsumexp_vs_plain"] = delta
        row["typical_head_competitor_multiplier_vs_plain"] = math.exp(max(-50.0, min(50.0, delta)))
        row["evidence_mass_ratio_vs_plain"] = row["evidence_mass"] / max(plain["evidence_mass"], 1e-300)
        row["ppl_ratio_vs_plain"] = row["gold_ppl_geomean"] / max(plain["gold_ppl_geomean"], 1e-300)
    return summaries


def failure_boundaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["placement"], row["condition"], row["concept_id"])].append(row)
    result = []
    for (placement, condition, concept_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["filler_length"]))
        base_ppl = float(ordered[0]["scores"]["gold_ppl"])
        failures = [row for row in ordered if not row["scores"]["candidate_correct"]]
        doubled = [row for row in ordered if float(row["scores"]["gold_ppl"]) >= 2.0 * base_ppl]
        sustained = None
        for index, row in enumerate(ordered):
            if all(not later["scores"]["candidate_correct"] for later in ordered[index:]):
                sustained = int(row["filler_length"])
                break
        result.append(
            {
                "placement": placement,
                "condition": condition,
                "concept_id": concept_id,
                "lemma": ordered[0]["lemma"],
                "bin": ordered[0]["bin"],
                "first_candidate_failure": int(failures[0]["filler_length"]) if failures else None,
                "sustained_candidate_failure": sustained,
                "first_ppl_2x": int(doubled[0]["filler_length"]) if doubled else None,
                "minimum_margin": min(float(row["scores"]["candidate_margin"]) for row in ordered),
                "maximum_tested_length": max(int(row["filler_length"]) for row in ordered),
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, model_metadata: dict[str, Any] | None = None) -> None:
    rows = read_jsonl(output_dir / "rows.jsonl")
    summaries = grouped_summaries(rows)
    boundaries = failure_boundaries(rows)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "failure_boundaries.csv", boundaries)
    lines = [
        "# Local Qwen softmax needle failure-boundary experiment",
        "",
        f"Completed rows: **{len(rows)}**",
        "",
        "## Definitions",
        "",
        "- `remote`: the verified catalog stays before the growing filler; evidence-query distance grows.",
        "- `fixed_recent`: filler grows before the verified catalog; evidence-query distance stays fixed and isolates denominator competition.",
        "- `plain`: unrelated archive records; `distractor`: semantic neighbors; `conflict`: the target meaning is assigned a wrong label in explicitly unverified records.",
        "- `evidence_vs_non_evidence = logsumexp(gold evidence logits) - logsumexp(all other logits)`. Its sigmoid equals the evidence attention mass when the two sets partition the history.",
        "",
        "## Aggregate results",
        "",
        "| Placement | Condition | Filler | N | Candidate | Greedy | Gold PPL | Margin | Evidence mass | Non-evidence LSE | Evidence log-odds | Top-20 recall | Outside top-20 | Competitor × plain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        multiplier = row.get("typical_head_competitor_multiplier_vs_plain", 1.0)
        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['filler_length']} | {row['n']} | "
            f"{row['candidate_accuracy']:.1%} | {row['greedy_accuracy']:.1%} | "
            f"{row['gold_ppl_geomean']:.3f} | {row['candidate_margin']:+.3f} | "
            f"{row['evidence_mass']:.6f} | {row['non_evidence_logsumexp']:.3f} | "
            f"{row['evidence_vs_non_evidence']:+.3f} | {row['evidence_hit_top20']:.1%} | "
            f"{row['outside_top20_mass']:.1%} | {multiplier:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-needle observed boundaries",
            "",
            "| Placement | Condition | Needle | Bin | First candidate failure | Sustained failure | First PPL ×2 | Min margin |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in boundaries:
        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['lemma']} | {row['bin']} | "
            f"{row['first_candidate_failure']} | {row['sustained_candidate_failure']} | "
            f"{row['first_ppl_2x']} | {row['minimum_margin']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Candidate accuracy means the gold A-H label is best among A-H; greedy accuracy means it is best in the full vocabulary.",
            "- The reported competitor multiplier is `exp(mean_head(S_nonE_condition - S_nonE_plain))`: a typical-head geometric multiplier, not a model-level probability ratio.",
            "- A first failure can recover later. `sustained failure` is therefore reported separately and should be trusted only after enough larger lengths are tested.",
            "- Arbitrary one-token labels prevent answer-token familiarity from being confused with semantic retrieval.",
        ]
    )
    if model_metadata:
        lines.extend(["", "## Runtime", "", "```json", json.dumps(model_metadata, ensure_ascii=False, indent=2), "```"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refinement_requests(rows: Sequence[dict[str, Any]], minimum_step: int) -> set[tuple[str, str, int]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["placement"], row["condition"], row["concept_id"])].append(row)
    requests: set[tuple[str, str, int]] = set()
    for (placement, condition, _), group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["filler_length"]))
        for left, right in zip(ordered, ordered[1:]):
            left_pass = bool(left["scores"]["candidate_correct"])
            right_pass = bool(right["scores"]["candidate_correct"])
            low = int(left["filler_length"])
            high = int(right["filler_length"])
            if left_pass == right_pass or high - low <= minimum_step:
                continue
            midpoint = ((low + high) // (2 * minimum_step)) * minimum_step
            midpoint = max(low + minimum_step, min(high - minimum_step, midpoint))
            if low < midpoint < high:
                requests.add((placement, condition, midpoint))
    return requests


def validate_design(tokenizer: Any, catalog: dict[str, Any]) -> dict[str, Any]:
    evidence_labels = {label: encode(tokenizer, " " + label) for label in LABELS}
    answer_labels = {label: encode(tokenizer, label) for label in LABELS}
    if any(len(ids) != 1 for ids in evidence_labels.values()):
        raise AssertionError(f"not all evidence labels are one token: {evidence_labels}")
    if any(len(ids) != 1 for ids in answer_labels.values()):
        raise AssertionError(f"not all answer labels are one token: {answer_labels}")
    blocks = {condition: build_special_block(tokenizer, condition, catalog["mapping"]) for condition in CONDITIONS}
    block_budget = max(len(value["token_ids"]) for value in blocks.values()) + 16
    return {
        "evidence_label_token_ids": evidence_labels,
        "answer_label_token_ids": answer_labels,
        "catalog_mapping": catalog["mapping"],
        "special_block_tokens": {key: len(value["token_ids"]) for key, value in blocks.items()},
        "block_budget": block_budget,
        "concepts": [
            {
                "concept_id": concept["concept_id"],
                "bin": concept["bin"],
                "lemma": concept["lemma"],
                "definition": concept["definition"],
                "query_clue": concept["query_clue"],
            }
            for concept in CONCEPTS
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen softmax needle boundary experiment.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--filler_lengths", default="0,1024,4096,16384,32768,65536")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--placements", default=",".join(PLACEMENTS))
    parser.add_argument(
        "--concept_ids",
        default="",
        help="Optional comma-separated concept IDs to evaluate; the shared catalog and archive stay unchanged.",
    )
    parser.add_argument("--special_period", type=int, default=4096)
    parser.add_argument("--recent_gap", type=int, default=128)
    parser.add_argument("--prefill_chunk_size", type=int, default=256)
    parser.add_argument("--adaptive_rounds", type=int, default=2)
    parser.add_argument("--minimum_refinement_step", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--save_head_rows", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    catalog = semantic_base.build_catalog(tokenizer, args.seed)
    wrapper_prefix, wrapper_suffix = semantic_base.chat_wrapper(tokenizer)
    design = validate_design(tokenizer, catalog)
    design.update(
        {
            "conditions": parse_csv(args.conditions),
            "placements": parse_csv(args.placements),
            "concept_ids": parse_csv(args.concept_ids),
            "filler_lengths": parse_ints(args.filler_lengths),
            "special_period": args.special_period,
            "recent_gap": args.recent_gap,
            "seed": args.seed,
        }
    )
    (output_dir / "design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return
    if args.summarize_only:
        metadata_path = output_dir / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        write_report(output_dir, metadata)
        return

    lengths = sorted(set(parse_ints(args.filler_lengths)))
    conditions = parse_csv(args.conditions)
    placements = parse_csv(args.placements)
    selected_concepts = set(parse_csv(args.concept_ids))
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"conditions must be a subset of {CONDITIONS}")
    if any(placement not in PLACEMENTS for placement in placements):
        raise ValueError(f"placements must be a subset of {PLACEMENTS}")
    known_concepts = {concept["concept_id"] for concept in CONCEPTS}
    unknown_concepts = selected_concepts - known_concepts
    if unknown_concepts:
        raise ValueError(f"unknown concept IDs: {sorted(unknown_concepts)}")
    maximum_length = max(lengths) + 4096
    model, tokenizer, model_metadata = load_model(args, maximum_length)
    (output_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    block_budget = int(design["block_budget"])

    for placement in placements:
        for condition in conditions:
            for filler_length in lengths:
                run_group(
                    args, model, tokenizer, catalog, wrapper_prefix, wrapper_suffix,
                    output_dir, placement, condition, filler_length,
                    args.special_period, block_budget,
                )
                write_report(output_dir, model_metadata)

    for round_index in range(args.adaptive_rounds):
        requests = refinement_requests(read_jsonl(output_dir / "rows.jsonl"), args.minimum_refinement_step)
        if not requests:
            break
        print(f"adaptive round {round_index + 1}: {sorted(requests)}", flush=True)
        for placement, condition, filler_length in sorted(requests):
            run_group(
                args, model, tokenizer, catalog, wrapper_prefix, wrapper_suffix,
                output_dir, placement, condition, filler_length,
                args.special_period, block_budget,
            )
            write_report(output_dir, model_metadata)

    write_report(output_dir, model_metadata)
    (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
