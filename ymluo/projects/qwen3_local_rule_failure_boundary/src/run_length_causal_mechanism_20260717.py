from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import run_local_rule_failure_boundary as base


CONDITIONS = (
    "clean",
    "low16",
    "high4",
    "conflict1",
    "competitor2",
    "mixed",
)

PLACEMENTS = ("prefix", "middle", "recent")

QUERY_SPECS = (
    ("full2", "legacy", True),
    ("hop1", "cloze", False),
    ("oracle_hop2", "cloze", False),
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def make_event(
    kind: str,
    label: str,
    prefix: str,
    antecedent: str,
    consequent: str,
    step: int,
) -> base.RuleEvent:
    return base.RuleEvent(
        kind,
        label,
        base.rule_line(prefix, label, antecedent, consequent),
        -1,
        -1,
        antecedent,
        consequent,
        step,
    )


def build_bundle(seed: int) -> dict[str, Any]:
    """Build every potential evidence item once so candidates stay fixed across conditions."""

    rng = random.Random(2026071701 + seed * 1009)
    gold_codes = [base.make_code(rng, "G", index) for index in range(3)]
    conflict_codes = [gold_codes[0], base.make_code(rng, "X", 1), base.make_code(rng, "X", 2)]

    gold_events = [
        make_event("relevant", f"T{step}", "VERIFIED RULE", gold_codes[step], gold_codes[step + 1], step)
        for step in range(2)
    ]
    conflict_events = [
        make_event(
            "conflict",
            f"X{step}",
            "DECOY RULE",
            conflict_codes[step],
            conflict_codes[step + 1],
            step,
        )
        for step in range(2)
    ]

    high_events: list[base.RuleEvent] = []
    for index in range(4):
        step = index % 2
        antecedent = base.mutate_code(gold_codes[step], rng)
        consequent = base.mutate_code(gold_codes[step + 1], rng)
        high_events.append(
            make_event("distractor", f"H{index}", "VERIFIED RULE", antecedent, consequent, step)
        )

    competitor_chains: list[list[base.RuleEvent]] = []
    competitor_finals: list[str] = []
    for chain_index, family in enumerate(("U", "V")):
        codes = [base.make_code(rng, family, chain_index * 10 + step) for step in range(3)]
        competitor_finals.append(codes[-1])
        competitor_chains.append(
            [
                make_event(
                    "competitor",
                    f"C{chain_index}_{step}",
                    "VERIFIED RULE",
                    codes[step],
                    codes[step + 1],
                    step,
                )
                for step in range(2)
            ]
        )

    low_events: list[base.RuleEvent] = []
    for index in range(16):
        code = base.make_code(rng, "L", index)
        low_events.append(
            base.RuleEvent(
                "distractor",
                f"L{index}",
                base.note_line(f"L{index}", code),
                -1,
                -1,
                code,
                code,
                index % 2,
            )
        )

    candidates = unique(
        [
            gold_codes[-1],
            gold_codes[1],
            gold_codes[0],
            conflict_codes[-1],
            conflict_codes[1],
            *(event.consequent for event in high_events),
            *competitor_finals,
        ]
    )
    # The start code is intentionally retained.  Excluding it made candidate
    # accuracy look correct even when greedy generation merely copied the key.
    while len(candidates) < 13:
        candidates.append(base.make_code(rng, "Z", len(candidates)))
    candidates = candidates[:13]
    roles = {
        gold_codes[-1]: "gold_final",
        gold_codes[1]: "gold_intermediate",
        gold_codes[0]: "gold_start",
        conflict_codes[-1]: "conflict_final",
        conflict_codes[1]: "conflict_intermediate",
    }
    for index, event in enumerate(high_events):
        roles[event.consequent] = f"high_distractor_{index}"
    for index, answer in enumerate(competitor_finals):
        roles[answer] = f"competitor_final_{index}"
    for candidate in candidates:
        roles.setdefault(candidate, "random")

    return {
        "gold_codes": gold_codes,
        "conflict_codes": conflict_codes,
        "gold_events": gold_events,
        "conflict_events": conflict_events,
        "high_events": high_events,
        "competitor_chains": competitor_chains,
        "low_events": low_events,
        "candidates": candidates,
        "candidate_roles": roles,
    }


def encode_event_block(
    tokenizer: Any, events: Sequence[base.RuleEvent], start: int
) -> tuple[list[int], list[base.RuleEvent]]:
    block: list[int] = []
    placed: list[base.RuleEvent] = []
    cursor = start
    for event in events:
        ids = base.token_ids(tokenizer, event.text)
        placed.append(
            base.RuleEvent(
                event.kind,
                event.label,
                event.text,
                cursor,
                cursor + len(ids),
                event.antecedent,
                event.consequent,
                event.step,
            )
        )
        block.extend(ids)
        cursor += len(ids)
    return block, placed


def span_free(start: int, length: int, occupied: Sequence[tuple[int, int]], buffer: int = 4) -> bool:
    end = start + length
    return all(end + buffer <= left or start >= right + buffer for left, right in occupied)


def nearest_free_start(
    preferred: int,
    block_length: int,
    total_length: int,
    occupied: Sequence[tuple[int, int]],
) -> int:
    max_start = total_length - block_length
    if max_start < 0:
        raise ValueError(f"block length {block_length} does not fit body length {total_length}")
    center = max(0, min(max_start, preferred))
    for radius in range(max_start + 1):
        options = (center - radius, center + radius) if radius else (center,)
        for candidate in options:
            if 0 <= candidate <= max_start and span_free(candidate, block_length, occupied):
                return candidate
    raise ValueError("could not place non-overlapping event block")


def insert_event_block(
    body: list[int],
    tokenizer: Any,
    events: Sequence[base.RuleEvent],
    preferred_start: int,
    occupied: list[tuple[int, int]],
    placed: list[base.RuleEvent],
) -> None:
    encoded, _ = encode_event_block(tokenizer, events, 0)
    start = nearest_free_start(preferred_start, len(encoded), len(body), occupied)
    body[start : start + len(encoded)] = encoded
    _, placed_events = encode_event_block(tokenizer, events, start)
    placed.extend(placed_events)
    occupied.append((start, start + len(encoded)))


def condition_blocks(bundle: dict[str, Any], condition: str) -> list[tuple[Sequence[base.RuleEvent], float]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    blocks: list[tuple[Sequence[base.RuleEvent], float]] = []
    # Keep low-information notes as their own density control.  Adding them to
    # mixed would change both plausible competition and occupied-token density,
    # and does not fit cleanly in the 1k condition.
    if condition == "low16":
        for index, event in enumerate(bundle["low_events"]):
            blocks.append(([event], (index + 1) / 18.0))
    if condition in {"high4", "mixed"}:
        for event, fraction in zip(bundle["high_events"], (0.15, 0.35, 0.65, 0.85)):
            blocks.append(([event], fraction))
    if condition in {"conflict1", "mixed"}:
        blocks.append((bundle["conflict_events"], 0.25))
    if condition in {"competitor2", "mixed"}:
        blocks.extend(
            (chain, fraction)
            for chain, fraction in zip(bundle["competitor_chains"], (0.10, 0.90))
        )
    return blocks


def build_body(
    tokenizer: Any,
    *,
    seed: int,
    target_context_tokens: int,
    condition: str,
    placement: str,
) -> dict[str, Any]:
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement: {placement}")
    bundle = build_bundle(seed)
    extra_blocks = condition_blocks(bundle, condition)
    placed: list[base.RuleEvent] = []

    if target_context_tokens == 0:
        ordered_blocks: list[Sequence[base.RuleEvent]] = [bundle["gold_events"]]
        ordered_blocks.extend(events for events, _ in extra_blocks)
        if seed % 2 == 1:
            ordered_blocks = list(reversed(ordered_blocks))
        body: list[int] = []
        cursor = 0
        for events in ordered_blocks:
            encoded, placed_events = encode_event_block(tokenizer, events, cursor)
            body.extend(encoded)
            placed.extend(placed_events)
            cursor += len(encoded)
    else:
        body = base.build_filler_ids(tokenizer, target_context_tokens, 1_700_000 + seed)
        occupied: list[tuple[int, int]] = []
        gold_ids, _ = encode_event_block(tokenizer, bundle["gold_events"], 0)
        if placement == "prefix":
            gold_preferred = min(256, max(0, target_context_tokens - len(gold_ids)))
        elif placement == "recent":
            gold_preferred = max(0, target_context_tokens - len(gold_ids) - 256)
        else:
            gold_preferred = target_context_tokens // 2 - len(gold_ids) // 2
        insert_event_block(
            body,
            tokenizer,
            bundle["gold_events"],
            gold_preferred,
            occupied,
            placed,
        )
        ordered_extra = extra_blocks if seed % 2 == 0 else list(reversed(extra_blocks))
        for events, fraction in ordered_extra:
            encoded, _ = encode_event_block(tokenizer, events, 0)
            preferred = int(round((target_context_tokens - len(encoded)) * fraction))
            insert_event_block(body, tokenizer, events, preferred, occupied, placed)

    placed.sort(key=lambda event: event.start_token)
    return {
        "seed": seed,
        "condition": condition,
        "placement": placement,
        "target_context_tokens": target_context_tokens,
        "body_ids": torch.tensor(body, dtype=torch.long).view(1, -1),
        "body_tokens": len(body),
        "events": placed,
        **bundle,
    }


def build_suffix(style: str, start_code: str, steps: int, query_mode: str) -> str:
    if style == "cloze":
        if query_mode not in {"hop1", "oracle_hop2"}:
            raise ValueError("cloze is only defined for one-transition probes")
        return (
            "\nVERIFIED RULE LOOKUP\n"
            "Complete the consequent of this exact applicable VERIFIED RULE.\n"
            f"IF {start_code} IS ACTIVE THEN "
        )
    if style == "legacy":
        return base.build_prompt_suffix(start_code, steps)
    if style != "chat_concise":
        raise ValueError(f"unknown prompt style: {style}")
    if query_mode in {"hop1", "oracle_hop2"}:
        return (
            "\nEVIDENCE LOOKUP\n"
            f"Lookup key: {start_code}\n"
            f"Find the line containing exactly 'IF {start_code} IS ACTIVE THEN'.\n"
            "It must begin with VERIFIED RULE. Copy the complete code immediately after THEN.\n"
            "Do one lookup only. Do not follow the copied code into a second rule.\n"
            "Ignore NOTE and DECOY RULE lines. Do not output markdown or an explanation.\n"
            "Respond with one complete code only."
        )
    return (
        "\nTWO-STEP TASK\n"
        f"Start code: {start_code}\n"
        f"Required state transitions: {steps}\n"
        "Use only applicable VERIFIED RULE lines. Ignore every NOTE and DECOY RULE line.\n"
        "The answer is the code reached after two transitions, not the start or one-step code.\n"
        "Do not output markdown, an explanation, or any repeated rules.\n"
        "Respond with one complete final code only."
    )


def find_subsequence(values: Sequence[int], target: Sequence[int]) -> int:
    if not target:
        raise ValueError("empty target subsequence")
    for start in range(0, len(values) - len(target) + 1):
        if list(values[start : start + len(target)]) == list(target):
            return start
    raise ValueError("marker tokens not found in chat template")


def chat_wrapper_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Split a zero-shot native chat template around the target content.

    The fixed assistant prefix turns free generation into a regular field
    completion without leaking a reusable answer from a demonstration.
    """

    marker = "CODEX_USER_CONTENT_MARKER_20260717"
    kwargs = {
        "conversation": [{"role": "user", "content": marker}],
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        rendered = tokenizer.apply_chat_template(**kwargs, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(**kwargs)
    rendered_ids = list(rendered)
    marker_ids = base.token_ids(tokenizer, marker)
    start = find_subsequence(rendered_ids, marker_ids)
    answer_prefix_ids = base.token_ids(tokenizer, "FINAL CODE: ")
    return (
        rendered_ids[:start],
        rendered_ids[start + len(marker_ids) :] + answer_prefix_ids,
    )


def query_parameters(body: dict[str, Any], query_mode: str) -> tuple[str, int, str]:
    gold_codes = body["gold_codes"]
    if query_mode == "full2":
        return gold_codes[0], 2, gold_codes[2]
    if query_mode == "hop1":
        return gold_codes[0], 1, gold_codes[1]
    if query_mode == "oracle_hop2":
        return gold_codes[1], 1, gold_codes[2]
    raise ValueError(f"unknown query mode: {query_mode}")


def extend_cache(
    model: Any,
    starting_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    starting_length: int,
    token_ids: torch.Tensor,
    chunk_size: int,
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], float]:
    device = base.input_device(model)
    ids = token_ids.to(device)
    past = base.cache_from_legacy(starting_cache)
    past_len = starting_length
    base.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(ids.shape[1]), chunk_size):
            chunk = ids[:, start : start + chunk_size]
            if int(chunk.shape[1]) == 0:
                continue
            out = base.forward_with_cache(model, chunk, past, past_len)
            past = out.past_key_values
            past_len += int(chunk.shape[1])
    base.synchronize()
    return base.legacy_cache(past), time.perf_counter() - started


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generation_metrics(generation: dict[str, Any], gold_answer: str) -> dict[str, Any]:
    text = str(generation["generated_text"]).replace("\\n", "\n")
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return {
        "generation_first_line": first_line,
        "generation_first_line_correct": int(base.compact_code(first_line) == base.compact_code(gold_answer)),
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["target_context_tokens"]),
            row["placement"],
            row["condition"],
            row["query_mode"],
            row["prompt_style"],
        )
        buckets[key].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: (item[0], item[1], item[2], item[3], item[4])):
        selected = buckets[key]
        n = len(selected)
        candidate_acc = statistics.mean(int(row["candidate_correct"]) for row in selected)
        start_excluded_acc = statistics.mean(
            int(row["start_excluded_candidate_correct"]) for row in selected
        )
        total_nll_acc = statistics.mean(
            int(row["total_nll_candidate_correct"]) for row in selected
        )
        generation_rows = [row for row in selected if "generation_first_line_correct" in row]
        output.append(
            {
                "target_context_tokens": key[0],
                "placement": key[1],
                "condition": key[2],
                "query_mode": key[3],
                "prompt_style": key[4],
                "sample_count": n,
                "candidate_accuracy": candidate_acc,
                "candidate_accuracy_sem": math.sqrt(candidate_acc * (1.0 - candidate_acc) / n),
                "start_excluded_candidate_accuracy": start_excluded_acc,
                "total_nll_candidate_accuracy": total_nll_acc,
                "mean_gold_nll": statistics.mean(float(row["gold_candidate_mean_nll"]) for row in selected),
                "mean_candidate_margin": statistics.mean(float(row["candidate_margin"]) for row in selected),
                "generation_sample_count": len(generation_rows),
                "generation_first_line_accuracy": (
                    statistics.mean(int(row["generation_first_line_correct"]) for row in generation_rows)
                    if generation_rows
                    else ""
                ),
                "generation_first_known_accuracy": (
                    statistics.mean(int(row["generation_first_correct"]) for row in generation_rows)
                    if generation_rows
                    else ""
                ),
                "generation_final_known_accuracy": (
                    statistics.mean(int(row["generation_final_correct"]) for row in generation_rows)
                    if generation_rows
                    else ""
                ),
                "generation_contains_gold_rate": (
                    statistics.mean(int(row["generation_contains_gold"]) for row in generation_rows)
                    if generation_rows
                    else ""
                ),
            }
        )
    return output


def mechanism_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[tuple[int, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        by_case[key][str(row["query_mode"])] = row
    detailed: list[dict[str, Any]] = []
    for (seed, length, placement, condition), modes in sorted(by_case.items()):
        if not {"full2", "hop1", "oracle_hop2"}.issubset(modes):
            continue
        full = int(modes["full2"].get("generation_final_correct", 0))
        full_candidate = int(modes["full2"]["candidate_correct"])
        full_candidate_conditioned = int(
            modes["full2"]["start_excluded_candidate_correct"]
        )
        strict_hop1 = int(modes["hop1"]["candidate_correct"])
        strict_oracle = int(modes["oracle_hop2"]["candidate_correct"])
        hop1 = int(modes["hop1"]["start_excluded_candidate_correct"])
        oracle = int(modes["oracle_hop2"]["start_excluded_candidate_correct"])
        if full:
            label = "success"
        elif not hop1:
            label = "first_hop_access_or_binding_failure"
        elif not oracle:
            label = "second_rule_access_failure"
        else:
            label = "composition_or_state_update_failure"
        detailed.append(
            {
                "seed": seed,
                "target_context_tokens": length,
                "placement": placement,
                "condition": condition,
                "full2_correct": full,
                "full2_candidate_correct": full_candidate,
                "full2_candidate_start_excluded_correct": full_candidate_conditioned,
                "hop1_cloze_correct": hop1,
                "oracle_hop2_cloze_correct": oracle,
                "hop1_cloze_strict_correct": strict_hop1,
                "oracle_hop2_cloze_strict_correct": strict_oracle,
                "failure_mechanism": label,
                "full2_prediction_role": modes["full2"]["candidate_prediction_role"],
                "hop1_prediction_role": modes["hop1"]["candidate_prediction_role"],
                "oracle_hop2_prediction_role": modes["oracle_hop2"]["candidate_prediction_role"],
            }
        )
    grouped: dict[tuple[int, str, str, str], int] = defaultdict(int)
    totals: dict[tuple[int, str, str], int] = defaultdict(int)
    for row in detailed:
        key = (
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        grouped[(key[0], key[1], key[2], str(row["failure_mechanism"]))] += 1
        totals[key] += 1
    summary = [
        {
            "target_context_tokens": length,
            "placement": placement,
            "condition": condition,
            "failure_mechanism": mechanism,
            "count": count,
            "fraction": count / totals[(length, placement, condition)],
            "sample_count": totals[(length, placement, condition)],
        }
        for (length, placement, condition, mechanism), count in sorted(grouped.items())
    ]
    return detailed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired causal diagnostic for long-context evidence access, binding, and composition."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=4)
    parser.add_argument("--lengths", default="0,1024,8192,32768,65536")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--placements", default="middle")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--fast_preflight",
        action="store_true",
        help="Skip rebuilding every body before model load; cases are written incrementally.",
    )
    args = parser.parse_args()

    lengths = parse_csv_ints(args.lengths)
    conditions = parse_csv_strs(args.conditions)
    placements = parse_csv_strs(args.placements)
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    unknown_placements = sorted(set(placements) - set(PLACEMENTS))
    if unknown_placements:
        raise ValueError(f"unknown placements: {unknown_placements}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    scores_jsonl = output_dir / "candidate_scores.jsonl"
    cases_jsonl = output_dir / "cases.jsonl"
    for path in (results_jsonl, scores_jsonl, cases_jsonl):
        if path.exists():
            path.unlink()
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    specs = [
        (seed, length, placement, condition)
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
        for length in lengths
        for placement in placements
        for condition in conditions
    ]
    previews: list[dict[str, Any]] = []
    if args.fast_preflight and not args.dry_run:
        # The largest filler body is exact.  A code-only bundle is enough to
        # measure every suffix without tokenizing all bodies.  Keeping this
        # bound tight matters at 64K: even a ~1K conservative overestimate can
        # push SDPA's repeated KV tensors over a 24GB card's limit.
        sample = build_bundle(args.seed_start)
        max_suffix = max(
            len(
                base.token_ids(
                    tokenizer,
                    build_suffix(
                        style,
                        *query_parameters(sample, mode)[:2],
                        mode,
                    ),
                )
            )
            for mode, style, _ in QUERY_SPECS
        )
        max_body = max(max(lengths, default=0), 2048 if 0 in lengths else 0)
        max_position = max_body + max_suffix + args.max_new_tokens + 8
    else:
        max_position = 0
        for seed, length, placement, condition in specs:
            body = build_body(
                tokenizer,
                seed=seed,
                target_context_tokens=length,
                condition=condition,
                placement=placement,
            )
            max_suffix = 0
            for mode, style, _ in QUERY_SPECS:
                suffix_length = len(
                    base.token_ids(
                        tokenizer,
                        build_suffix(style, *query_parameters(body, mode)[:2], mode),
                    )
                )
                max_suffix = max(max_suffix, suffix_length)
            max_position = max(
                max_position,
                int(body["body_tokens"]) + max_suffix + args.max_new_tokens + 8,
            )
            previews.append(
                {
                    "seed": seed,
                    "target_context_tokens": length,
                    "condition": condition,
                    "placement": placement,
                    "body_tokens": body["body_tokens"],
                    "gold_codes": body["gold_codes"],
                    "conflict_codes": body["conflict_codes"],
                    "candidates": body["candidates"],
                    "candidate_roles": body["candidate_roles"],
                    "events": [asdict(event) for event in body["events"]],
                }
            )
        append_jsonl(cases_jsonl, previews)
    if args.dry_run:
        print(f"wrote {len(previews)} body cases; max_position={max_position}", flush=True)
        return

    max_factor = base.rope_factor_for_length(max_position, args.original_max_position_embeddings)
    model, tokenizer = base.load_model_and_tokenizer(args, max_position, max_factor)
    result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for body_index, (seed, length, placement, condition) in enumerate(specs, start=1):
        body = build_body(
            tokenizer,
            seed=seed,
            target_context_tokens=length,
            condition=condition,
            placement=placement,
        )
        if args.fast_preflight:
            append_jsonl(
                cases_jsonl,
                [
                    {
                        "seed": seed,
                        "target_context_tokens": length,
                        "condition": condition,
                        "placement": placement,
                        "body_tokens": body["body_tokens"],
                        "gold_codes": body["gold_codes"],
                        "conflict_codes": body["conflict_codes"],
                        "candidates": body["candidates"],
                        "candidate_roles": body["candidate_roles"],
                        "events": [asdict(event) for event in body["events"]],
                    }
                ],
            )
        body_result_rows: list[dict[str, Any]] = []
        body_score_rows: list[dict[str, Any]] = []
        # Legacy answer queries and exact cloze probes share the identical raw
        # document prefix, so one long-context prefill serves all three.
        wrapped_body_ids = body["body_ids"]
        wrapped_body_tokens = int(wrapped_body_ids.shape[1])
        body_cache, body_prefill_seconds = base.prefill_sequence(
            model, wrapped_body_ids, args.prefill_chunk_size
        )
        for query_mode, prompt_style, run_generation in QUERY_SPECS:
                start_code, steps, gold_answer = query_parameters(body, query_mode)
                suffix = build_suffix(prompt_style, start_code, steps, query_mode)
                suffix_tokens = base.token_ids(tokenizer, suffix)
                suffix_ids = torch.tensor(suffix_tokens, dtype=torch.long).view(1, -1)
                query_cache, suffix_prefill_seconds = extend_cache(
                    model,
                    body_cache,
                    wrapped_body_tokens,
                    suffix_ids[:, :-1],
                    args.prefill_chunk_size,
                )
                prompt_len_minus_one = wrapped_body_tokens + int(suffix_ids.shape[1]) - 1
                # Keep the start key in the primary candidate set.  Copying the
                # query key is a real failure mode and must not be hidden by the
                # evaluator.  We also report the historical start-excluded
                # score below so old and corrected protocols remain comparable.
                query_candidates = list(body["candidates"])
                if gold_answer not in query_candidates or len(query_candidates) != 13:
                    raise ValueError("fixed candidate set is invalid")
                candidate_summary, candidate_rows = base.score_candidates(
                    model,
                    tokenizer,
                    query_cache,
                    suffix_ids[:, -1:],
                    prompt_len_minus_one,
                    query_candidates,
                    gold_answer,
                )
                prediction = str(candidate_summary["candidate_prediction"])
                total_nll_prediction = str(
                    min(
                        candidate_rows,
                        key=lambda candidate_row: float(candidate_row["candidate_total_nll"]),
                    )["candidate"]
                )
                start_excluded_rows = [
                    candidate_row
                    for candidate_row in candidate_rows
                    if base.compact_code(str(candidate_row["candidate"]))
                    != base.compact_code(start_code)
                ]
                start_excluded_prediction = (
                    str(start_excluded_rows[0]["candidate"])
                    if start_excluded_rows
                    else ""
                )
                generation: dict[str, Any] = {}
                if run_generation:
                    generation = base.generate_answer(
                        model,
                        tokenizer,
                        query_cache,
                        suffix_ids[:, -1:],
                        prompt_len_minus_one,
                        args.max_new_tokens,
                        gold_answer,
                        body["events"],
                        query_candidates,
                    )
                    generation.update(generation_metrics(generation, gold_answer))
                row = {
                    "seed": seed,
                    "target_context_tokens": length,
                    "condition": condition,
                    "placement": placement,
                    "query_mode": query_mode,
                    "prompt_style": prompt_style,
                    "start_code": start_code,
                    "required_steps": steps,
                    "gold_answer": gold_answer,
                    "gold_final": body["gold_codes"][-1],
                    "gold_intermediate": body["gold_codes"][1],
                    "conflict_final": body["conflict_codes"][-1],
                    "conflict_intermediate": body["conflict_codes"][1],
                    "body_tokens": body["body_tokens"],
                    "prompt_tokens": prompt_len_minus_one + 1,
                    "body_prefill_seconds": body_prefill_seconds,
                    "suffix_prefill_seconds": suffix_prefill_seconds,
                    "candidate_prediction_role": body["candidate_roles"].get(prediction, "unknown"),
                    "total_nll_candidate_prediction": total_nll_prediction,
                    "total_nll_candidate_prediction_role": body["candidate_roles"].get(
                        total_nll_prediction, "unknown"
                    ),
                    "total_nll_candidate_correct": int(total_nll_prediction == gold_answer),
                    "start_excluded_candidate_prediction": start_excluded_prediction,
                    "start_excluded_candidate_prediction_role": body["candidate_roles"].get(
                        start_excluded_prediction, "unknown"
                    ),
                    "start_excluded_candidate_correct": int(
                        start_excluded_prediction == gold_answer
                    ),
                    **candidate_summary,
                    **generation,
                }
                body_result_rows.append(row)
                for rank, candidate_row in enumerate(candidate_rows, start=1):
                    candidate = str(candidate_row["candidate"])
                    body_score_rows.append(
                        {
                            "seed": seed,
                            "target_context_tokens": length,
                            "condition": condition,
                            "placement": placement,
                            "query_mode": query_mode,
                            "prompt_style": prompt_style,
                            "gold_answer": gold_answer,
                            "candidate": candidate,
                            "candidate_role": body["candidate_roles"].get(candidate, "unknown"),
                            "rank": rank,
                            **{key: value for key, value in candidate_row.items() if key != "candidate"},
                        }
                    )
                del query_cache
        del body_cache
        result_rows.extend(body_result_rows)
        score_rows.extend(body_score_rows)
        append_jsonl(results_jsonl, body_result_rows)
        append_jsonl(scores_jsonl, body_score_rows)
        print(
            f"[{body_index}/{len(specs)}] seed={seed} length={length} placement={placement} "
            f"condition={condition} "
            + " ".join(
                f"{row['prompt_style']}/{row['query_mode']}="
                f"{row['candidate_prediction_role']}"
                for row in body_result_rows
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(result_rows)
    mechanism_detail, mechanism_summary = mechanism_rows(result_rows)
    write_csv(output_dir / "results.csv", result_rows)
    write_csv(output_dir / "candidate_scores.csv", score_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "mechanism_detail.csv", mechanism_detail)
    write_csv(output_dir / "mechanism_summary.csv", mechanism_summary)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "max_position": max_position,
                "rope_factor": max_factor,
                "body_case_count": len(specs),
                "query_result_count": len(result_rows),
                "candidate_score_count": len(score_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
