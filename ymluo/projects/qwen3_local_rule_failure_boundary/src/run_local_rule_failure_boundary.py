from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


FILLER_PARAGRAPHS = [
    "The archive describes schedules, room numbers, supply orders, weather notes, and maintenance logs.",
    "A committee reviewed forms, labels, map references, and delivery times without changing any rule.",
    "The report lists storage shelves, cable colors, visitor badges, and ordinary office procedures.",
    "During the workshop, participants discussed examples, formatting choices, and checklist updates.",
    "The building guide mentions elevators, exits, lighting, equipment rooms, and cleaning routines.",
    "A neutral memo records timestamps, batch identifiers, temperature readings, and inspection status.",
    "The catalog entry contains a title, a short description, a location tag, and a revision date.",
    "An operations note explains who opened the room, which tools were returned, and where boxes were placed.",
    "The training document covers policy names, review cadence, note taking, and general communication.",
    "A travel notice summarizes platform changes, route labels, gate numbers, and parking instructions.",
    "The ledger includes invoice IDs, folder names, department codes, and unrelated reference numbers.",
    "A classroom handout discusses outlines, examples, paragraph order, and proofreading reminders.",
]


@dataclass(frozen=True)
class RuleEvent:
    kind: str
    label: str
    text: str
    start_token: int
    end_token: int
    antecedent: str
    consequent: str
    step: int


@dataclass(frozen=True)
class BuiltCase:
    case_id: str
    model_label: str
    target_context_tokens: int
    actual_prompt_tokens: int
    depth_percent: float
    seed: int
    distractor_count: int
    distractor_similarity: str
    rule_gap_tokens: int
    actual_rule_gap_tokens: int
    chain_length: int
    competitor_count: int
    start_code: str
    gold_answer: str
    prompt_suffix_tokens: int
    rope_factor: float
    max_position_embeddings: int
    relevant_rule_count: int
    distractor_rule_count: int
    conflict_rule_count: int
    competitor_rule_count: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def rope_factor_for_length(token_count: int, original_max_position_embeddings: int) -> float:
    if token_count <= original_max_position_embeddings:
        return 1.0
    return float(2 ** math.ceil(math.log2(token_count / original_max_position_embeddings)))


def build_filler_ids(tokenizer: Any, target_length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    ids: list[int] = []
    templates = list(FILLER_PARAGRAPHS)
    while len(ids) < target_length + 512:
        rng.shuffle(templates)
        text = "\n".join(templates) + "\n"
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids[:target_length]


def make_code(rng: random.Random, family: str, index: int) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{family}{letters[index % len(letters)]}{rng.randrange(10, 99)}-{rng.randrange(100, 999)}"


def mutate_code(code: str, rng: random.Random) -> str:
    chars = list(code)
    positions = [idx for idx, ch in enumerate(chars) if ch.isalnum()]
    pos = rng.choice(positions)
    if chars[pos].isdigit():
        chars[pos] = str((int(chars[pos]) + rng.randrange(1, 9)) % 10)
    else:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        chars[pos] = rng.choice([ch for ch in alphabet if ch != chars[pos]])
    return "".join(chars)


def overlap(span_a: tuple[int, int], span_b: tuple[int, int], buffer_tokens: int = 4) -> bool:
    return not (span_a[1] + buffer_tokens <= span_b[0] or span_b[1] + buffer_tokens <= span_a[0])


def find_free_offset(
    length: int,
    target_context_tokens: int,
    occupied: list[tuple[int, int]],
    rng: random.Random,
) -> int:
    if length >= target_context_tokens:
        raise ValueError(f"event length {length} does not fit target length {target_context_tokens}")
    for _ in range(2000):
        start = rng.randrange(0, target_context_tokens - length)
        span = (start, start + length)
        if all(not overlap(span, existing) for existing in occupied):
            return start
    for start in range(0, target_context_tokens - length):
        span = (start, start + length)
        if all(not overlap(span, existing) for existing in occupied):
            return start
    raise ValueError("could not place non-overlapping event; reduce distractor count or increase context length")


def place_event(
    context_ids: list[int],
    event_ids: list[int],
    start: int,
    event: RuleEvent,
) -> RuleEvent:
    end = start + len(event_ids)
    if end > len(context_ids):
        raise ValueError(f"event {event.label} overflows context")
    context_ids[start:end] = event_ids
    return RuleEvent(
        kind=event.kind,
        label=event.label,
        text=event.text,
        start_token=start,
        end_token=end,
        antecedent=event.antecedent,
        consequent=event.consequent,
        step=event.step,
    )


def rule_line(prefix: str, label: str, antecedent: str, consequent: str) -> str:
    return f"{prefix} {label}: IF {antecedent} IS ACTIVE THEN {consequent} BECOMES ACTIVE.\n"


def note_line(label: str, code: str) -> str:
    return f"NOTE {label}: The catalog mentions code {code}, but this note is not an active rule.\n"


def build_prompt_suffix(start_code: str, chain_length: int) -> str:
    return (
        "\n\nTask: Follow symbolic rules exactly.\n"
        "Use only lines beginning with VERIFIED RULE.\n"
        "Ignore DECOY RULE and NOTE lines even if they look similar.\n"
        f"Start code: {start_code}\n"
        f"Apply exactly {chain_length} valid rule step(s), one step at a time.\n"
        "What is the final active code?\n"
        "Answer with the code only.\n"
        "Answer: "
    )


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_case(
    tokenizer: Any,
    *,
    model_label: str,
    target_context_tokens: int,
    depth_percent: float,
    seed: int,
    distractor_count: int,
    distractor_similarity: str,
    rule_gap_tokens: int,
    chain_length: int,
    competitor_count: int,
    max_new_tokens: int,
    original_max_position_embeddings: int,
) -> tuple[BuiltCase, torch.Tensor, list[RuleEvent], list[str]]:
    rng = random.Random(
        (
            seed * 1000003
            + target_context_tokens * 9176
            + int(depth_percent * 13)
            + distractor_count * 193
            + rule_gap_tokens * 17
            + chain_length * 31
            + competitor_count * 47
            + sum(ord(ch) for ch in distractor_similarity)
        )
    )
    context_ids = build_filler_ids(tokenizer, target_context_tokens, seed)
    family = chr(ord("A") + (seed + chain_length + distractor_count) % 20)
    chain_codes = [make_code(rng, family, idx) for idx in range(chain_length + 1)]
    start_code = chain_codes[0]
    gold_answer = chain_codes[-1]

    relevant_events: list[tuple[RuleEvent, list[int]]] = []
    for step in range(chain_length):
        text = rule_line("VERIFIED RULE", f"T{step}", chain_codes[step], chain_codes[step + 1])
        event = RuleEvent("relevant", f"T{step}", text, -1, -1, chain_codes[step], chain_codes[step + 1], step)
        relevant_events.append((event, token_ids(tokenizer, text)))

    relevant_total_len = sum(len(ids) for _, ids in relevant_events)
    if chain_length > 1:
        max_gap = max(0, (target_context_tokens - relevant_total_len) // (chain_length - 1))
        actual_gap = min(rule_gap_tokens, max_gap)
    else:
        actual_gap = 0
    relevant_block_len = relevant_total_len + actual_gap * max(0, chain_length - 1)
    if relevant_block_len > target_context_tokens:
        raise ValueError("relevant rule block does not fit context")
    max_start = max(0, target_context_tokens - relevant_block_len)
    relevant_start = int(round(max_start * depth_percent / 100.0))

    occupied: list[tuple[int, int]] = []
    placed_events: list[RuleEvent] = []
    cursor = relevant_start
    for event, ids in relevant_events:
        placed = place_event(context_ids, ids, cursor, event)
        placed_events.append(placed)
        occupied.append((placed.start_token, placed.end_token))
        cursor = placed.end_token + actual_gap

    conflict_answers: list[str] = []
    distractor_answers: list[str] = []
    competitor_answers: list[str] = []

    competitor_event_count = 0
    for comp_idx in range(competitor_count):
        comp_family = chr(ord("U") + (comp_idx % 5))
        comp_codes = [make_code(rng, comp_family, comp_idx * 7 + step) for step in range(chain_length + 1)]
        competitor_answers.append(comp_codes[-1])
        for step in range(chain_length):
            text = rule_line("VERIFIED RULE", f"C{comp_idx}_{step}", comp_codes[step], comp_codes[step + 1])
            ids = token_ids(tokenizer, text)
            start = find_free_offset(len(ids), target_context_tokens, occupied, rng)
            event = RuleEvent(
                "competitor",
                f"C{comp_idx}_{step}",
                text,
                -1,
                -1,
                comp_codes[step],
                comp_codes[step + 1],
                step,
            )
            placed = place_event(context_ids, ids, start, event)
            placed_events.append(placed)
            occupied.append((placed.start_token, placed.end_token))
            competitor_event_count += 1

    for idx in range(distractor_count):
        step = idx % max(1, chain_length)
        if distractor_similarity == "low":
            fake_code = make_code(rng, "L", idx)
            text = note_line(f"L{idx}", fake_code)
            antecedent = fake_code
            consequent = fake_code
            kind = "distractor"
        elif distractor_similarity == "medium":
            antecedent = make_code(rng, "M", idx)
            consequent = make_code(rng, "N", idx)
            text = rule_line("VERIFIED RULE", f"D{idx}", antecedent, consequent)
            kind = "distractor"
            distractor_answers.append(consequent)
        elif distractor_similarity == "high":
            antecedent = mutate_code(chain_codes[step], rng)
            consequent = mutate_code(chain_codes[min(step + 1, chain_length)], rng)
            text = rule_line("VERIFIED RULE", f"H{idx}", antecedent, consequent)
            kind = "distractor"
            distractor_answers.append(consequent)
        elif distractor_similarity == "conflict":
            antecedent = chain_codes[step]
            consequent = mutate_code(chain_codes[min(step + 1, chain_length)], rng)
            text = rule_line("DECOY RULE", f"X{idx}", antecedent, consequent)
            kind = "conflict"
            conflict_answers.append(consequent)
        else:
            raise ValueError(f"unknown distractor similarity: {distractor_similarity}")
        ids = token_ids(tokenizer, text)
        start = find_free_offset(len(ids), target_context_tokens, occupied, rng)
        event = RuleEvent(kind, f"D{idx}", text, -1, -1, antecedent, consequent, step)
        placed = place_event(context_ids, ids, start, event)
        placed_events.append(placed)
        occupied.append((placed.start_token, placed.end_token))

    suffix = build_prompt_suffix(start_code, chain_length)
    suffix_ids = token_ids(tokenizer, suffix)
    prompt_ids = context_ids + suffix_ids
    rope_factor = rope_factor_for_length(len(prompt_ids) + max_new_tokens + 8, original_max_position_embeddings)
    max_pos = max(
        len(prompt_ids) + max_new_tokens + 8,
        int(original_max_position_embeddings * rope_factor),
    )
    case_id = (
        f"len{target_context_tokens}_d{int(depth_percent)}_seed{seed}_"
        f"dist{distractor_count}_{distractor_similarity}_gap{rule_gap_tokens}_"
        f"chain{chain_length}_comp{competitor_count}"
    )
    case = BuiltCase(
        case_id=case_id,
        model_label=model_label,
        target_context_tokens=target_context_tokens,
        actual_prompt_tokens=len(prompt_ids),
        depth_percent=depth_percent,
        seed=seed,
        distractor_count=distractor_count,
        distractor_similarity=distractor_similarity,
        rule_gap_tokens=rule_gap_tokens,
        actual_rule_gap_tokens=actual_gap,
        chain_length=chain_length,
        competitor_count=competitor_count,
        start_code=start_code,
        gold_answer=gold_answer,
        prompt_suffix_tokens=len(suffix_ids),
        rope_factor=rope_factor,
        max_position_embeddings=max_pos,
        relevant_rule_count=chain_length,
        distractor_rule_count=sum(1 for item in placed_events if item.kind == "distractor"),
        conflict_rule_count=sum(1 for item in placed_events if item.kind == "conflict"),
        competitor_rule_count=competitor_event_count,
    )
    candidates = [gold_answer] + conflict_answers + competitor_answers + distractor_answers
    while len(candidates) < 8:
        candidates.append(make_code(rng, "Z", len(candidates)))
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    rng.shuffle(deduped)
    if gold_answer not in deduped:
        deduped.insert(0, gold_answer)
    max_candidates = 16
    if gold_answer in deduped[:max_candidates]:
        final_candidates = deduped[:max_candidates]
    else:
        final_candidates = [gold_answer] + [candidate for candidate in deduped if candidate != gold_answer][
            : max_candidates - 1
        ]
    return case, torch.tensor(prompt_ids, dtype=torch.long).view(1, -1), placed_events, final_candidates


def load_model_and_tokenizer(args: argparse.Namespace, max_case_position: int, max_factor: float) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if max_factor > 1.0:
        config.max_position_embeddings = max_case_position
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(max_factor),
            "original_max_position_embeddings": int(args.original_max_position_embeddings),
        }
    elif max_case_position > int(getattr(config, "max_position_embeddings", 0)):
        config.max_position_embeddings = max_case_position

    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.dtype),
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(args.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    return tuple(cache)


def cache_from_legacy(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


def forward_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any | None,
    past_len: int,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": True,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    if past_key_values is not None:
        q_len = int(input_ids.shape[1])
        device = input_ids.device
        kwargs["past_key_values"] = past_key_values
        kwargs["attention_mask"] = torch.ones((1, past_len + q_len), dtype=torch.long, device=device)
        kwargs["position_ids"] = torch.arange(past_len, past_len + q_len, device=device).view(1, -1)
        kwargs["cache_position"] = torch.arange(past_len, past_len + q_len, device=device)
    return model(**kwargs)


def prefill_sequence(model: Any, prompt_prefix: torch.Tensor, chunk_size: int) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], float]:
    device = input_device(model)
    ids = prompt_prefix.to(device)
    past = None
    past_len = 0
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(ids.shape[1]), chunk_size):
            chunk = ids[:, start : start + chunk_size]
            out = forward_with_cache(model, chunk, past, past_len)
            past = out.past_key_values
            past_len += int(chunk.shape[1])
    synchronize()
    return legacy_cache(past), time.perf_counter() - started


def score_candidate(
    model: Any,
    tokenizer: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    candidate: str,
) -> dict[str, Any]:
    device = input_device(model)
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    total_nll = 0.0
    token_count = int(candidate_ids.shape[1])
    with torch.inference_mode():
        out = forward_with_cache(model, last_prompt_id.to(device), cache_from_legacy(base_cache), prompt_len_minus_one)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        past_len = prompt_len_minus_one + 1
        for idx in range(token_count):
            target = candidate_ids[:, idx]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            total_nll -= float(log_probs.gather(1, target.view(1, 1)).item())
            out = forward_with_cache(model, target.view(1, 1), cache, past_len)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            past_len += 1
    mean_nll = total_nll / max(1, token_count)
    return {
        "candidate": candidate,
        "candidate_token_count": token_count,
        "candidate_total_nll": total_nll,
        "candidate_mean_nll": mean_nll,
        "candidate_ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
    }


def score_candidates(
    model: Any,
    tokenizer: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    candidates: list[str],
    gold_answer: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        score_candidate(model, tokenizer, base_cache, last_prompt_id, prompt_len_minus_one, candidate)
        for candidate in candidates
    ]
    rows.sort(key=lambda row: (float(row["candidate_mean_nll"]), float(row["candidate_total_nll"])))
    pred = str(rows[0]["candidate"]) if rows else ""
    gold_rows = [row for row in rows if row["candidate"] == gold_answer]
    gold = gold_rows[0] if gold_rows else {"candidate_mean_nll": float("inf"), "candidate_ppl": float("inf")}
    runner_up = rows[1] if len(rows) > 1 and pred == gold_answer else rows[0] if rows else None
    margin = (
        float(runner_up["candidate_mean_nll"]) - float(gold["candidate_mean_nll"])
        if runner_up is not None
        else float("nan")
    )
    probs = []
    if rows:
        min_nll = min(float(row["candidate_mean_nll"]) for row in rows)
        weights = [math.exp(-(float(row["candidate_mean_nll"]) - min_nll)) for row in rows]
        denom = sum(weights)
        probs = [value / denom for value in weights]
    entropy = -sum(prob * math.log(max(prob, 1e-30)) for prob in probs)
    return (
        {
            "candidate_prediction": pred,
            "candidate_correct": int(pred == gold_answer),
            "gold_candidate_mean_nll": gold["candidate_mean_nll"],
            "gold_candidate_ppl": gold["candidate_ppl"],
            "candidate_margin": margin,
            "candidate_entropy": entropy,
            "candidate_count": len(rows),
        },
        rows,
    )


def normalize_generated(text: str) -> str:
    text = text.replace("\\n", "\n")
    text = re.split(r"[\s,.;:，。；：]", text.strip(), maxsplit=1)[0]
    return re.sub(r"[^A-Za-z0-9-]", "", text).upper()


def compact_code(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def known_code_kinds(gold: str, events: list[RuleEvent], candidates: list[str]) -> dict[str, str]:
    kinds: dict[str, str] = {compact_code(gold): "gold"}
    for event in events:
        key = compact_code(event.consequent)
        if key == compact_code(gold):
            kinds[key] = "gold"
        else:
            kinds.setdefault(key, event.kind)
    for candidate in candidates:
        key = compact_code(candidate)
        if key == compact_code(gold):
            kinds[key] = "gold"
        else:
            kinds.setdefault(key, "random")
    return kinds


def extract_known_code_mentions(
    generated_text: str,
    gold: str,
    events: list[RuleEvent],
    candidates: list[str],
) -> list[dict[str, Any]]:
    kinds = known_code_kinds(gold, events, candidates)
    canonical_by_compact: dict[str, str] = {}
    canonical_by_compact[compact_code(gold)] = gold.upper()
    for event in events:
        canonical_by_compact.setdefault(compact_code(event.consequent), event.consequent.upper())
    for candidate in candidates:
        canonical_by_compact.setdefault(compact_code(candidate), candidate.upper())

    mentions: list[dict[str, Any]] = []
    pattern = re.compile(r"\b[A-Za-z]{2}\d{2}\s*-?\s*\d{3}\b")
    for match in pattern.finditer(generated_text):
        key = compact_code(match.group(0))
        if key not in kinds:
            continue
        mentions.append(
            {
                "answer": canonical_by_compact.get(key, match.group(0).upper().replace(" ", "")),
                "answer_compact": key,
                "answer_class": kinds[key],
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )
    return mentions


def classify_answer_text(
    raw_answer: str,
    gold: str,
    events: list[RuleEvent],
    candidates: list[str],
) -> dict[str, Any]:
    kinds = known_code_kinds(gold, events, candidates)
    canonical_by_compact: dict[str, str] = {compact_code(gold): gold.upper()}
    for event in events:
        canonical_by_compact.setdefault(compact_code(event.consequent), event.consequent.upper())
    for candidate in candidates:
        canonical_by_compact.setdefault(compact_code(candidate), candidate.upper())

    key = compact_code(raw_answer)
    cleaned = re.sub(r"[^A-Za-z0-9-]", "", raw_answer).upper()
    if key in kinds:
        return {
            "answer": canonical_by_compact.get(key, cleaned),
            "answer_compact": key,
            "answer_class": kinds[key],
        }
    return {
        "answer": cleaned,
        "answer_compact": key,
        "answer_class": "wrong" if key else "miss",
    }


def extract_explicit_answers(
    generated_text: str,
    gold: str,
    events: list[RuleEvent],
    candidates: list[str],
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(?i)\b(?:final\s+(?:active\s+)?(?:code|answer)|answer)\s*(?:is|:)?\s*"
        r"([A-Za-z]{0,2}\d{2}\s*-?\s*\d{3}|\d{3,6})"
    )
    for match in pattern.finditer(generated_text):
        item = classify_answer_text(match.group(1), gold, events, candidates)
        item.update(
            {
                "raw_answer": match.group(1),
                "start_char": match.start(1),
                "end_char": match.end(1),
            }
        )
        answers.append(item)
    return answers


def classify_generation(generated_text: str, gold: str, events: list[RuleEvent]) -> str:
    normalized = normalize_generated(generated_text)
    if not normalized:
        return "miss"
    if normalized == gold.upper():
        return "correct"
    by_kind: dict[str, set[str]] = {}
    for event in events:
        by_kind.setdefault(event.kind, set()).add(event.consequent.upper())
    for kind in ("conflict", "competitor", "distractor", "relevant"):
        if normalized in by_kind.get(kind, set()):
            return kind
    return "wrong"


def generate_answer(
    model: Any,
    tokenizer: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    max_new_tokens: int,
    gold_answer: str,
    events: list[RuleEvent],
    candidates: list[str],
) -> dict[str, Any]:
    device = input_device(model)
    generated: list[int] = []
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = forward_with_cache(model, last_prompt_id.to(device), cache_from_legacy(base_cache), prompt_len_minus_one)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        past_len = prompt_len_minus_one + 1
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break
            generated.append(token_id)
            out = forward_with_cache(model, next_token.to(device), cache, past_len)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            past_len += 1
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
    synchronize()
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    generation_class = classify_generation(generated_text, gold_answer, events)
    mentions = extract_known_code_mentions(generated_text, gold_answer, events, candidates)
    explicit_answers = extract_explicit_answers(generated_text, gold_answer, events, candidates)
    first_mention = mentions[0] if mentions else None
    last_known_mention = mentions[-1] if mentions else None
    final_answer = explicit_answers[-1] if explicit_answers else last_known_mention
    final_source = "explicit" if explicit_answers else "last_known_mention" if last_known_mention else "none"
    mentioned_classes = [str(item["answer_class"]) for item in mentions]
    contains_gold = any(item["answer_class"] == "gold" for item in mentions)
    contains_wrong = any(item["answer_class"] in {"conflict", "competitor", "distractor", "random"} for item in mentions)
    contains_wrong = contains_wrong or any(item["answer_class"] == "wrong" for item in explicit_answers)
    final_class = str(final_answer["answer_class"]) if final_answer else "miss"
    first_class = str(first_mention["answer_class"]) if first_mention else "miss"
    return {
        "generated_text": generated_text.replace("\n", "\\n"),
        "generated_normalized": normalize_generated(generated_text),
        "generation_class": generation_class,
        "generation_correct": int(generation_class == "correct"),
        "generation_contains_gold": int(contains_gold),
        "generation_contains_wrong": int(contains_wrong),
        "generation_mentioned_codes": " ".join(str(item["answer"]) for item in mentions),
        "generation_mentioned_classes": " ".join(mentioned_classes),
        "generation_explicit_answers": " ".join(str(item["answer"]) for item in explicit_answers),
        "generation_explicit_classes": " ".join(str(item["answer_class"]) for item in explicit_answers),
        "generation_first_answer": "" if first_mention is None else first_mention["answer"],
        "generation_first_class": first_class,
        "generation_first_correct": int(first_class == "gold"),
        "generation_final_answer": "" if final_answer is None else final_answer["answer"],
        "generation_final_class": final_class,
        "generation_final_correct": int(final_class == "gold"),
        "generation_final_source": final_source,
        "generation_seconds": time.perf_counter() - started,
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_to_q(q: torch.Tensor, position_embeddings: Any) -> torch.Tensor:
    if position_embeddings is None:
        return q
    cos, sin = position_embeddings
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    sin = sin.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    return (q * cos) + (rotate_half(q) * sin)


def span_numer_from_logits(logits_f: torch.Tensor, spans: list[tuple[int, int]], max_value: torch.Tensor) -> torch.Tensor:
    numer = torch.zeros((), device=logits_f.device, dtype=torch.float32)
    for start, end in spans:
        if end > start:
            numer = numer + torch.exp(logits_f[start:end] - max_value).sum()
    return numer


def attention_selectivity(
    model: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    events: list[RuleEvent],
    page_tokens: int,
) -> dict[str, Any]:
    device = input_device(model)
    spans_by_kind: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        spans_by_kind.setdefault(event.kind, []).append((event.start_token, event.end_token))
    relevant_spans = spans_by_kind.get("relevant", [])
    non_gold_spans = (
        spans_by_kind.get("distractor", [])
        + spans_by_kind.get("conflict", [])
        + spans_by_kind.get("competitor", [])
    )
    try:
        layers = list(getattr(getattr(model, "model", None), "layers", []))
        if not layers:
            raise RuntimeError("cannot locate model.model.layers")
        captured_q: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                position_embeddings = kwargs.get("position_embeddings")
                if position_embeddings is None and len(args) >= 2:
                    position_embeddings = args[1]
                if hidden_states is None:
                    return
                q = module.q_proj(hidden_states)
                batch, q_len, _ = q.shape
                head_dim = int(getattr(module, "head_dim"))
                num_heads = int(q.shape[-1] // head_dim)
                q = q.view(batch, q_len, num_heads, head_dim).transpose(1, 2)
                q = apply_rope_to_q(q, position_embeddings)
                captured_q[layer_idx] = q[:, :, -1, :].detach()

            return hook

        for idx, layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))
        synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                out = forward_with_cache(model, last_prompt_id.to(device), cache_from_legacy(base_cache), prompt_len_minus_one)
        finally:
            for handle in handles:
                handle.remove()
        synchronize()
        cache = legacy_cache(out.past_key_values)
        if not captured_q:
            raise RuntimeError("failed to capture query states")

        relevant_values: list[float] = []
        nongold_values: list[float] = []
        conflict_values: list[float] = []
        competitor_values: list[float] = []
        distractor_values: list[float] = []
        selectivity_values: list[float] = []
        page_scores_accum: list[float] | None = None
        kv_len = int(cache[-1][0].shape[2])

        for layer_idx, layer in enumerate(layers):
            q = captured_q[layer_idx][0]
            key = cache[layer_idx][0][0]
            num_heads = int(q.shape[0])
            kv_heads = int(key.shape[0])
            groups = max(1, num_heads // kv_heads)
            scale = float(getattr(layer.self_attn, "scaling", q.shape[-1] ** -0.5))
            for head_idx in range(num_heads):
                kv_idx = min(kv_heads - 1, head_idx // groups)
                logits = torch.matmul(key[kv_idx].float(), q[head_idx].float()) * scale
                logits_f = logits.float()
                max_value = logits_f.max()
                denom = torch.exp(logits_f - max_value).sum().clamp_min(1e-30)
                relevant = float((span_numer_from_logits(logits_f, relevant_spans, max_value) / denom).item())
                nongold = float((span_numer_from_logits(logits_f, non_gold_spans, max_value) / denom).item())
                conflict = float((span_numer_from_logits(logits_f, spans_by_kind.get("conflict", []), max_value) / denom).item())
                competitor = float((span_numer_from_logits(logits_f, spans_by_kind.get("competitor", []), max_value) / denom).item())
                distractor = float((span_numer_from_logits(logits_f, spans_by_kind.get("distractor", []), max_value) / denom).item())
                relevant_values.append(relevant)
                nongold_values.append(nongold)
                conflict_values.append(conflict)
                competitor_values.append(competitor)
                distractor_values.append(distractor)
                selectivity_values.append(relevant / max(relevant + nongold, 1e-30))
                if page_scores_accum is None:
                    page_scores_accum = [0.0 for _ in range((kv_len + page_tokens - 1) // page_tokens)]
                probs = torch.exp(logits_f - max_value)
                probs = probs / probs.sum().clamp_min(1e-30)
                for page_idx, page_start in enumerate(range(0, kv_len, page_tokens)):
                    page_end = min(kv_len, page_start + page_tokens)
                    page_scores_accum[page_idx] += float(probs[page_start:page_end].sum().item())

        def mean(values: list[float]) -> float:
            return sum(values) / max(1, len(values))

        page_scores = page_scores_accum or []
        ranked_pages = sorted(range(len(page_scores)), key=lambda idx: page_scores[idx], reverse=True)
        relevant_pages = sorted({start // page_tokens for start, _ in relevant_spans})
        ranks = [ranked_pages.index(page) + 1 for page in relevant_pages if page in ranked_pages]
        expected_relevant = sum(max(0, end - start) for start, end in relevant_spans) / max(1, kv_len)
        return {
            "attention_available": 1,
            "attention_seconds": time.perf_counter() - started,
            "gold_rule_mass_mean": mean(relevant_values),
            "gold_rule_mass_top_head": max(relevant_values) if relevant_values else 0.0,
            "non_gold_rule_mass_mean": mean(nongold_values),
            "conflict_rule_mass_mean": mean(conflict_values),
            "competitor_rule_mass_mean": mean(competitor_values),
            "distractor_rule_mass_mean": mean(distractor_values),
            "rule_attention_selectivity": mean(selectivity_values),
            "normalized_gold_rule_mass": mean(relevant_values) / max(expected_relevant, 1e-30),
            "gold_rule_best_page_rank": min(ranks) if ranks else -1,
            "gold_rule_worst_page_rank": max(ranks) if ranks else -1,
            "page_count": len(page_scores),
        }
    except Exception as exc:
        return {
            "attention_available": 0,
            "attention_error": repr(exc),
            "attention_seconds": 0.0,
            "gold_rule_mass_mean": "",
            "gold_rule_mass_top_head": "",
            "non_gold_rule_mass_mean": "",
            "conflict_rule_mass_mean": "",
            "competitor_rule_mass_mean": "",
            "distractor_rule_mass_mean": "",
            "rule_attention_selectivity": "",
            "normalized_gold_rule_mass": "",
            "gold_rule_best_page_rank": "",
            "gold_rule_worst_page_rank": "",
            "page_count": "",
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        return None
    return None


def summarize(rows: list[dict[str, Any]], boundary_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_keys = [
        "model_label",
        "target_context_tokens",
        "depth_percent",
        "distractor_count",
        "distractor_similarity",
        "rule_gap_tokens",
        "chain_length",
        "competitor_count",
    ]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        n = len(bucket)
        cand_acc = sum(int(row["candidate_correct"]) for row in bucket) / max(1, n)
        gen_acc = sum(int(row["generation_correct"]) for row in bucket) / max(1, n)
        contains_gold_acc = sum(int(row.get("generation_contains_gold", 0)) for row in bucket) / max(1, n)
        first_answer_acc = sum(int(row.get("generation_first_correct", 0)) for row in bucket) / max(1, n)
        final_answer_acc = sum(int(row.get("generation_final_correct", 0)) for row in bucket) / max(1, n)
        wrong_contam_rate = sum(int(row.get("generation_contains_wrong", 0)) for row in bucket) / max(1, n)
        margins = [safe_float(row.get("candidate_margin")) for row in bucket]
        margins = [item for item in margins if item is not None]
        selectivity = [safe_float(row.get("rule_attention_selectivity")) for row in bucket]
        selectivity = [item for item in selectivity if item is not None]
        gold_mass = [safe_float(row.get("gold_rule_mass_mean")) for row in bucket]
        gold_mass = [item for item in gold_mass if item is not None]
        row = dict(zip(group_keys, key))
        row.update(
            {
                "cases": n,
                "candidate_accuracy": f"{cand_acc:.6f}",
                "generation_accuracy": f"{gen_acc:.6f}",
                "contains_gold_accuracy": f"{contains_gold_acc:.6f}",
                "first_answer_accuracy": f"{first_answer_acc:.6f}",
                "final_answer_accuracy": f"{final_answer_acc:.6f}",
                "wrong_answer_contamination": f"{wrong_contam_rate:.6f}",
                "mean_candidate_margin": "" if not margins else f"{sum(margins) / len(margins):.6f}",
                "mean_rule_attention_selectivity": "" if not selectivity else f"{sum(selectivity) / len(selectivity):.6f}",
                "mean_gold_rule_mass": "" if not gold_mass else f"{sum(gold_mass) / len(gold_mass):.8f}",
            }
        )
        summary_rows.append(row)

    boundary_group_keys = [
        "model_label",
        "depth_percent",
        "distractor_similarity",
        "rule_gap_tokens",
        "chain_length",
        "competitor_count",
        "distractor_count",
    ]
    boundary_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summary_rows:
        boundary_buckets.setdefault(tuple(row[key] for key in boundary_group_keys), []).append(row)
    boundary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(boundary_buckets.items(), key=lambda item: item[0]):
        ordered = sorted(bucket, key=lambda row: int(row["target_context_tokens"]))
        fail = [row for row in ordered if float(row["candidate_accuracy"]) < boundary_threshold]
        best_before_fail = [row for row in ordered if float(row["candidate_accuracy"]) >= boundary_threshold]
        out = dict(zip(boundary_group_keys, key))
        out.update(
            {
                "boundary_threshold": boundary_threshold,
                "first_fail_context_tokens": "" if not fail else fail[0]["target_context_tokens"],
                "first_fail_accuracy": "" if not fail else fail[0]["candidate_accuracy"],
                "last_pass_context_tokens": "" if not best_before_fail else best_before_fail[-1]["target_context_tokens"],
                "last_pass_accuracy": "" if not best_before_fail else best_before_fail[-1]["candidate_accuracy"],
                "observed_lengths": ",".join(str(row["target_context_tokens"]) for row in ordered),
                "observed_accuracies": ",".join(str(row["candidate_accuracy"]) for row in ordered),
            }
        )
        boundary_rows.append(out)
    return summary_rows, boundary_rows


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Local Rule Failure Boundary Summary",
        "",
        "## Lowest Candidate Accuracy Conditions",
        "",
        "| model | length | depth | dist | sim | gap | chain | comp | cases | cand acc | gen acc | contains gold | final acc | wrong contam | selectivity |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    worst = sorted(summary_rows, key=lambda row: float(row["candidate_accuracy"]))[:30]
    for row in worst:
        lines.append(
            f"| {row['model_label']} | {row['target_context_tokens']} | {row['depth_percent']} | "
            f"{row['distractor_count']} | {row['distractor_similarity']} | {row['rule_gap_tokens']} | "
            f"{row['chain_length']} | {row['competitor_count']} | {row['cases']} | "
            f"{row['candidate_accuracy']} | {row['generation_accuracy']} | {row['contains_gold_accuracy']} | "
            f"{row['final_answer_accuracy']} | {row['wrong_answer_contamination']} | "
            f"{row['mean_rule_attention_selectivity']} |"
        )
    lines.extend(
        [
            "",
            "## Earliest Failure Boundaries",
            "",
            "| model | sim | chain | comp | dist | depth | gap | first fail | acc | last pass | acc |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    failed = [row for row in boundary_rows if str(row["first_fail_context_tokens"]) != ""]
    failed = sorted(failed, key=lambda row: int(row["first_fail_context_tokens"]))[:40]
    for row in failed:
        lines.append(
            f"| {row['model_label']} | {row['distractor_similarity']} | {row['chain_length']} | "
            f"{row['competitor_count']} | {row['distractor_count']} | {row['depth_percent']} | "
            f"{row['rule_gap_tokens']} | {row['first_fail_context_tokens']} | {row['first_fail_accuracy']} | "
            f"{row['last_pass_context_tokens']} | {row['last_pass_accuracy']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def should_compute_attention(args: argparse.Namespace, row: dict[str, Any], case_index: int) -> bool:
    if not args.compute_attention or args.attention_mode == "none":
        return False
    if int(row["actual_prompt_tokens"]) > args.max_attention_prompt_tokens:
        return False
    if args.attention_mode == "all":
        return True
    if args.attention_mode == "failures":
        return int(row.get("candidate_correct", 0)) == 0 or int(row.get("generation_correct", 0)) == 0
    if args.attention_mode == "sampled":
        rng = random.Random(args.case_order_seed + case_index * 997)
        return rng.random() < args.attention_sample_rate
    raise ValueError(f"unknown attention mode: {args.attention_mode}")


def build_case_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for length in parse_csv_ints(args.lengths):
        for depth in parse_csv_floats(args.depths):
            for seed in parse_csv_ints(args.seeds):
                for distractor_count in parse_csv_ints(args.distractor_counts):
                    for similarity in parse_csv_strs(args.distractor_similarities):
                        for gap in parse_csv_ints(args.rule_gap_tokens):
                            for chain_length in parse_csv_ints(args.chain_lengths):
                                for competitor_count in parse_csv_ints(args.competitor_counts):
                                    specs.append(
                                        {
                                            "target_context_tokens": length,
                                            "depth_percent": depth,
                                            "seed": seed,
                                            "distractor_count": distractor_count,
                                            "distractor_similarity": similarity,
                                            "rule_gap_tokens": gap,
                                            "chain_length": chain_length,
                                            "competitor_count": competitor_count,
                                        }
                                    )
    if args.case_order == "shuffled":
        rng = random.Random(args.case_order_seed)
        rng.shuffle(specs)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local if-then rule failure-boundary diagnostics.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--model_label", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="1024,4096,8192")
    parser.add_argument("--depths", default="10,50,90")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--distractor_counts", default="0,16,64")
    parser.add_argument("--distractor_similarities", default="low,high,conflict")
    parser.add_argument("--rule_gap_tokens", default="0,512")
    parser.add_argument("--chain_lengths", default="1,2,4")
    parser.add_argument("--competitor_counts", default="0,4")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--case_order", choices=["sequential", "shuffled"], default="shuffled")
    parser.add_argument("--case_order_seed", type=int, default=20260709)
    parser.add_argument("--dry_run_cases", type=str2bool, default=False)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--compute_attention", type=str2bool, default=True)
    parser.add_argument("--attention_mode", choices=["all", "failures", "sampled", "none"], default="failures")
    parser.add_argument("--attention_sample_rate", type=float, default=0.10)
    parser.add_argument("--max_attention_prompt_tokens", type=int, default=65536)
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--boundary_threshold", type=float, default=0.90)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = args.model_label or Path(args.model_name_or_path.rstrip("/")).name

    specs = build_case_specs(args)
    if not specs:
        raise SystemExit("no cases selected")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    preview_cases: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    valid_specs: list[dict[str, Any]] = []
    max_case_position = 0
    max_factor = 1.0
    for spec in specs:
        try:
            case, _, events, candidates = build_case(
                tokenizer,
                model_label=model_label,
                max_new_tokens=args.max_new_tokens,
                original_max_position_embeddings=args.original_max_position_embeddings,
                **spec,
            )
        except ValueError as exc:
            skipped_cases.append({**spec, "skip_reason": str(exc)})
            continue
        valid_specs.append(spec)
        max_case_position = max(max_case_position, case.max_position_embeddings)
        max_factor = max(max_factor, case.rope_factor)
        preview = asdict(case)
        preview["events"] = [asdict(event) for event in events]
        preview["candidates"] = candidates
        preview_cases.append(preview)
        if args.max_cases > 0 and len(valid_specs) >= args.max_cases:
            break

    specs = valid_specs
    if not specs:
        raise SystemExit("no valid cases selected after placement filtering")
    (output_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in preview_cases) + "\n",
        encoding="utf-8",
    )
    (output_dir / "skipped_cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in skipped_cases) + ("\n" if skipped_cases else ""),
        encoding="utf-8",
    )
    if args.dry_run_cases:
        env = {
            "dry_run": True,
            "case_count": len(preview_cases),
            "skipped_case_count": len(skipped_cases),
            "args": vars(args),
            "model_label": model_label,
            "max_case_position": max_case_position,
            "max_rope_factor": max_factor,
        }
        (output_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"dry-run cases written: {output_dir / 'cases.jsonl'}", flush=True)
        return

    model, tokenizer = load_model_and_tokenizer(args, max_case_position, max_factor)
    device = input_device(model)

    env = {
        "git_commit": git_commit(repo_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_name_or_path": args.model_name_or_path,
        "model_label": model_label,
        "case_count": len(specs),
        "skipped_case_count": len(skipped_cases),
        "effective_config": {
            "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
            "rope_scaling": getattr(model.config, "rope_scaling", None),
            "rope_theta": getattr(model.config, "rope_theta", None),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
        },
        "args": vars(args),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(device)
        env["gpu_count_visible"] = torch.cuda.device_count()
    (output_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    for case_index, spec in enumerate(specs):
        case, prompt_ids, events, candidates = build_case(
            tokenizer,
            model_label=model_label,
            max_new_tokens=args.max_new_tokens,
            original_max_position_embeddings=args.original_max_position_embeddings,
            **spec,
        )
        print(
            f"=== {case_index + 1}/{len(specs)} {case.case_id} prompt_tokens={case.actual_prompt_tokens} ===",
            flush=True,
        )
        prompt_prefix = prompt_ids[:, :-1]
        last_prompt_id = prompt_ids[:, -1:]
        base_cache, prefill_seconds = prefill_sequence(model, prompt_prefix, args.prefill_chunk_size)
        cand_summary, cand_rows = score_candidates(
            model,
            tokenizer,
            base_cache,
            last_prompt_id,
            case.actual_prompt_tokens - 1,
            candidates,
            case.gold_answer,
        )
        gen = generate_answer(
            model,
            tokenizer,
            base_cache,
            last_prompt_id,
            case.actual_prompt_tokens - 1,
            args.max_new_tokens,
            case.gold_answer,
            events,
            candidates,
        )
        row: dict[str, Any] = {
            **asdict(case),
            "prefill_seconds": f"{prefill_seconds:.6f}",
            **cand_summary,
            **gen,
        }
        if should_compute_attention(args, row, case_index):
            attn = attention_selectivity(
                model,
                base_cache,
                last_prompt_id,
                case.actual_prompt_tokens - 1,
                events,
                args.page_tokens,
            )
        else:
            attn = {"attention_available": 0, "attention_seconds": 0.0}
        row.update(attn)
        result_rows.append(row)
        for rank, score_row in enumerate(cand_rows):
            score_rows.append({**asdict(case), "rank": rank + 1, **score_row})
        if int(attn.get("attention_available", 0)) == 1:
            attention_rows.append(row)

        write_csv(output_dir / "results.csv", result_rows)
        write_csv(output_dir / "candidate_scores.csv", score_rows)
        write_csv(output_dir / "attention_selectivity.csv", attention_rows)
        summary_rows, boundary_rows = summarize(result_rows, args.boundary_threshold)
        write_csv(output_dir / "summary_by_condition.csv", summary_rows)
        write_csv(output_dir / "failure_boundary.csv", boundary_rows)
        write_markdown_summary(output_dir / "summary.md", summary_rows, boundary_rows)
        print(
            f"done {case.case_id}: cand={cand_summary['candidate_prediction']} "
            f"correct={cand_summary['candidate_correct']} gen={gen['generation_class']} "
            f"margin={cand_summary['candidate_margin']:.4f} "
            f"selectivity={attn.get('rule_attention_selectivity', '')}",
            flush=True,
        )
        del base_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
