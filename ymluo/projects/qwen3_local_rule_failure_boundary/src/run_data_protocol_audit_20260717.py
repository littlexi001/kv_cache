from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base


PROTOCOLS = ("typed", "source", "temporal", "scope", "ambiguous")


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def transform_event(
    event: base.RuleEvent, protocol: str, role: str, neutral_index: int
) -> base.RuleEvent:
    antecedent = event.antecedent
    consequent = event.consequent
    if protocol == "typed":
        text = event.text
        label = event.label
    elif protocol == "source":
        source = "OFFICIAL RECORD" if role == "gold" else "FORUM CLAIM"
        label = f"R{neutral_index}"
        text = (
            f"{source} {label}: WHEN {antecedent} IS ACTIVE, "
            f"THE NEXT CODE IS {consequent}.\n"
        )
    elif protocol == "temporal":
        date = "2026-07-17" if role == "gold" else "2025-01-01"
        label = f"R{neutral_index}"
        text = (
            f"RECORD {label} [DATE {date}]: WHEN {antecedent} IS ACTIVE, "
            f"THE NEXT CODE IS {consequent}.\n"
        )
    elif protocol == "scope":
        scope = "PROJECT ALPHA" if role == "gold" else "PROJECT BETA"
        label = f"R{neutral_index}"
        text = (
            f"{scope} RECORD {label}: WHEN {antecedent} IS ACTIVE, "
            f"THE NEXT CODE IS {consequent}.\n"
        )
    elif protocol == "ambiguous":
        label = f"R{neutral_index}"
        text = (
            f"RULE {label}: WHEN {antecedent} IS ACTIVE, "
            f"THE NEXT CODE IS {consequent}.\n"
        )
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return base.RuleEvent(
        event.kind,
        label,
        text,
        -1,
        -1,
        antecedent,
        consequent,
        event.step,
    )


def protocol_events(seed: int, protocol: str) -> tuple[dict[str, Any], list[base.RuleEvent], list[base.RuleEvent]]:
    bundle = causal.build_bundle(seed)
    gold = [
        transform_event(event, protocol, "gold", index)
        for index, event in enumerate(bundle["gold_events"])
    ]
    conflict = [
        transform_event(event, protocol, "conflict", index + len(gold))
        for index, event in enumerate(bundle["conflict_events"])
    ]
    return bundle, gold, conflict


def build_body(
    tokenizer: Any,
    *,
    seed: int,
    target_context_tokens: int,
    condition: str,
    protocol: str,
) -> dict[str, Any]:
    if condition not in {"clean", "conflict1"}:
        raise ValueError(f"unknown condition: {condition}")
    bundle, gold, conflict = protocol_events(seed, protocol)
    blocks: list[Sequence[base.RuleEvent]] = [gold]
    if condition == "conflict1":
        blocks.append(conflict)
    placed: list[base.RuleEvent] = []
    if target_context_tokens == 0:
        if seed % 2:
            blocks.reverse()
        body: list[int] = []
        cursor = 0
        for events in blocks:
            encoded, placed_events = causal.encode_event_block(tokenizer, events, cursor)
            body.extend(encoded)
            placed.extend(placed_events)
            cursor += len(encoded)
    else:
        body = base.build_filler_ids(tokenizer, target_context_tokens, 2_700_000 + seed)
        occupied: list[tuple[int, int]] = []
        gold_ids, _ = causal.encode_event_block(tokenizer, gold, 0)
        causal.insert_event_block(
            body,
            tokenizer,
            gold,
            target_context_tokens // 2 - len(gold_ids) // 2,
            occupied,
            placed,
        )
        if condition == "conflict1":
            conflict_ids, _ = causal.encode_event_block(tokenizer, conflict, 0)
            fraction = 0.25 if seed % 2 == 0 else 0.75
            preferred = int((target_context_tokens - len(conflict_ids)) * fraction)
            causal.insert_event_block(
                body,
                tokenizer,
                conflict,
                preferred,
                occupied,
                placed,
            )
    placed.sort(key=lambda event: event.start_token)
    return {
        **bundle,
        "body_ids": torch.tensor(body, dtype=torch.long).view(1, -1),
        "body_tokens": len(body),
        "events": placed,
        "protocol": protocol,
        "condition": condition,
        "target_context_tokens": target_context_tokens,
    }


def suffix(protocol: str, start_code: str, required_steps: int) -> str:
    if protocol == "typed":
        return base.build_prompt_suffix(start_code, required_steps)
    authority = {
        "source": "Use OFFICIAL RECORD lines only. FORUM CLAIM lines are not authoritative.",
        "temporal": "Use only the newest dated records (2026-07-17), not older records.",
        "scope": "Answer for PROJECT ALPHA only. PROJECT BETA records are out of scope.",
        "ambiguous": "Use the applicable RULE lines.",
    }[protocol]
    return (
        f"\n{required_steps}-STEP RECORD TASK\n"
        f"Start code: {start_code}\n"
        f"{authority}\n"
        f"Apply exactly {required_steps} transition(s) by matching the current code to WHEN and "
        "copying NEXT CODE.\n"
        f"Return the code after exactly {required_steps} transition(s). Do not return the start code.\n"
        "Answer with one complete code only."
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit synthetic conflict-resolution protocols.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=16)
    parser.add_argument("--lengths", default="0,8192")
    parser.add_argument("--conditions", default="clean,conflict1")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--required_steps", type=int, choices=[1, 2], default=2)
    parser.add_argument("--prefill_chunk_size", type=int, default=512)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    lengths = [int(item) for item in csv_items(args.lengths)]
    conditions = csv_items(args.conditions)
    protocols = csv_items(args.protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"unknown protocols: {unknown}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    specs = [
        (seed, length, condition, protocol)
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
        for length in lengths
        for condition in conditions
        for protocol in protocols
    ]
    previews = []
    max_position = 0
    for seed, length, condition, protocol in specs:
        body = build_body(
            tokenizer,
            seed=seed,
            target_context_tokens=length,
            condition=condition,
            protocol=protocol,
        )
        suffix_tokens = len(
            base.token_ids(
                tokenizer,
                suffix(protocol, body["gold_codes"][0], args.required_steps),
            )
        )
        max_position = max(max_position, body["body_tokens"] + suffix_tokens + args.max_new_tokens + 8)
        previews.append(
            {
                "seed": seed,
                "target_context_tokens": length,
                "condition": condition,
                "protocol": protocol,
                "identifiable": int(protocol != "ambiguous"),
                "required_steps": args.required_steps,
                "gold_codes": body["gold_codes"],
                "conflict_codes": body["conflict_codes"],
                "events": [asdict(event) for event in body["events"]],
            }
        )
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in previews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(f"cases={len(previews)} max_position={max_position}")
        return

    factor = base.rope_factor_for_length(max_position, args.original_max_position_embeddings)
    model, tokenizer = base.load_model_and_tokenizer(args, max_position, factor)
    rows: list[dict[str, Any]] = []
    for index, (seed, length, condition, protocol) in enumerate(specs, start=1):
        body = build_body(
            tokenizer,
            seed=seed,
            target_context_tokens=length,
            condition=condition,
            protocol=protocol,
        )
        prompt_suffix = suffix(protocol, body["gold_codes"][0], args.required_steps)
        suffix_ids = torch.tensor(
            base.token_ids(tokenizer, prompt_suffix), dtype=torch.long
        ).view(1, -1)
        body_cache, prefill_seconds = base.prefill_sequence(
            model, body["body_ids"], args.prefill_chunk_size
        )
        query_cache, suffix_seconds = causal.extend_cache(
            model,
            body_cache,
            body["body_tokens"],
            suffix_ids[:, :-1],
            args.prefill_chunk_size,
        )
        prompt_len_minus_one = body["body_tokens"] + int(suffix_ids.shape[1]) - 1
        candidates = list(body["candidates"])
        gold = body["gold_codes"][args.required_steps]
        candidate_summary, candidate_rows = base.score_candidates(
            model,
            tokenizer,
            query_cache,
            suffix_ids[:, -1:],
            prompt_len_minus_one,
            candidates,
            gold,
        )
        start = body["gold_codes"][0]
        no_start = [
            row
            for row in candidate_rows
            if base.compact_code(str(row["candidate"])) != base.compact_code(start)
        ]
        generation = base.generate_answer(
            model,
            tokenizer,
            query_cache,
            suffix_ids[:, -1:],
            prompt_len_minus_one,
            args.max_new_tokens,
            gold,
            body["events"],
            candidates,
        )
        prediction = str(candidate_summary["candidate_prediction"])
        conditioned_prediction = str(no_start[0]["candidate"])
        row = {
            "seed": seed,
            "target_context_tokens": length,
            "condition": condition,
            "protocol": protocol,
            "identifiable": int(protocol != "ambiguous"),
            "required_steps": args.required_steps,
            "body_tokens": body["body_tokens"],
            "gold_final": gold,
            "conflict_final": body["conflict_codes"][-1],
            "candidate_prediction_role": body["candidate_roles"].get(prediction, "unknown"),
            "start_excluded_candidate_prediction": conditioned_prediction,
            "start_excluded_candidate_prediction_role": body["candidate_roles"].get(
                conditioned_prediction, "unknown"
            ),
            "start_excluded_candidate_correct": int(conditioned_prediction == gold),
            "body_prefill_seconds": prefill_seconds,
            "suffix_prefill_seconds": suffix_seconds,
            **candidate_summary,
            **generation,
        }
        rows.append(row)
        print(
            f"[{index}/{len(specs)}] seed={seed} length={length} condition={condition} "
            f"protocol={protocol} candidate={row['candidate_prediction_role']} "
            f"final={row['generation_final_correct']}",
            flush=True,
        )
        del query_cache, body_cache

    write_csv(output_dir / "results.csv", rows)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["target_context_tokens"], row["condition"], row["protocol"])].append(row)
    summary = []
    for (length, condition, protocol), selected in sorted(buckets.items()):
        roles = Counter(str(row["candidate_prediction_role"]) for row in selected)
        summary.append(
            {
                "target_context_tokens": length,
                "condition": condition,
                "protocol": protocol,
                "identifiable": int(protocol != "ambiguous"),
                "required_steps": args.required_steps,
                "sample_count": len(selected),
                "generation_final_accuracy_or_arbitrary_gold_rate": statistics.mean(
                    int(row["generation_final_correct"]) for row in selected
                ),
                "start_excluded_candidate_accuracy_or_arbitrary_gold_rate": statistics.mean(
                    int(row["start_excluded_candidate_correct"]) for row in selected
                ),
                "strict_candidate_accuracy_or_arbitrary_gold_rate": statistics.mean(
                    int(row["candidate_correct"]) for row in selected
                ),
                "mean_gold_nll": statistics.mean(
                    float(row["gold_candidate_mean_nll"]) for row in selected
                ),
                "mean_margin": statistics.mean(float(row["candidate_margin"]) for row in selected),
                "prediction_roles": json.dumps(roles, ensure_ascii=False, sort_keys=True),
            }
        )
    write_csv(output_dir / "summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
