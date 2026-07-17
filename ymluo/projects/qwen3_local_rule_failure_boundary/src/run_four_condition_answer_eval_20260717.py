from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

import run_local_rule_failure_boundary as base


CONDITIONS = (
    "gold_only",
    "gold_plus_conflict",
    "filler_plus_gold",
    "filler_plus_gold_plus_conflict",
)


def build_events(seed: int, chain_length: int) -> tuple[list[base.RuleEvent], list[base.RuleEvent], list[str]]:
    rng = random.Random(20260717 + seed * 1009)
    gold_codes = [base.make_code(rng, "G", index) for index in range(chain_length + 1)]
    conflict_codes = [gold_codes[0]] + [
        base.make_code(rng, "X", index + 1) for index in range(chain_length)
    ]
    gold_events: list[base.RuleEvent] = []
    conflict_events: list[base.RuleEvent] = []
    for step in range(chain_length):
        gold_text = base.rule_line(
            "VERIFIED RULE", f"T{step}", gold_codes[step], gold_codes[step + 1]
        )
        conflict_text = base.rule_line(
            "DECOY RULE", f"X{step}", conflict_codes[step], conflict_codes[step + 1]
        )
        gold_events.append(
            base.RuleEvent(
                "relevant", f"T{step}", gold_text, -1, -1,
                gold_codes[step], gold_codes[step + 1], step,
            )
        )
        conflict_events.append(
            base.RuleEvent(
                "conflict", f"X{step}", conflict_text, -1, -1,
                conflict_codes[step], conflict_codes[step + 1], step,
            )
        )
    candidates = [gold_codes[-1], conflict_codes[-1], gold_codes[1], conflict_codes[1]]
    while len(candidates) < 8:
        candidates.append(base.make_code(rng, "Z", len(candidates)))
    return gold_events, conflict_events, candidates


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
                event.kind, event.label, event.text, cursor, cursor + len(ids),
                event.antecedent, event.consequent, event.step,
            )
        )
        block.extend(ids)
        cursor += len(ids)
    return block, placed


def build_variant(
    tokenizer: Any,
    *,
    seed: int,
    condition: str,
    filler_tokens: int,
    chain_length: int,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    with_filler = condition.startswith("filler_")
    with_conflict = condition.endswith("conflict")
    gold_events, conflict_events, candidates = build_events(seed, chain_length)
    gold_block, _ = encode_event_block(tokenizer, gold_events, 0)
    conflict_block, _ = encode_event_block(tokenizer, conflict_events, 0)

    placed: list[base.RuleEvent] = []
    if with_filler:
        body = base.build_filler_ids(tokenizer, filler_tokens, 900_000 + seed)
        gold_start = max(0, min(filler_tokens - len(gold_block), filler_tokens // 2 - len(gold_block) // 2))
        body[gold_start : gold_start + len(gold_block)] = gold_block
        _, placed_gold = encode_event_block(tokenizer, gold_events, gold_start)
        placed.extend(placed_gold)
        if with_conflict:
            preferred = filler_tokens // 4 if seed % 2 == 0 else (3 * filler_tokens) // 4
            conflict_start = max(0, min(filler_tokens - len(conflict_block), preferred))
            gold_span = (gold_start, gold_start + len(gold_block))
            conflict_span = (conflict_start, conflict_start + len(conflict_block))
            if base.overlap(gold_span, conflict_span, buffer_tokens=8):
                conflict_start = max(0, gold_start - len(conflict_block) - 16)
            body[conflict_start : conflict_start + len(conflict_block)] = conflict_block
            _, placed_conflict = encode_event_block(tokenizer, conflict_events, conflict_start)
            placed.extend(placed_conflict)
    else:
        blocks: list[tuple[list[int], Sequence[base.RuleEvent]]] = [(gold_block, gold_events)]
        if with_conflict:
            conflict_item = (conflict_block, conflict_events)
            blocks = [conflict_item, blocks[0]] if seed % 2 == 0 else [blocks[0], conflict_item]
        body = []
        cursor = 0
        for block, events in blocks:
            body.extend(block)
            _, placed_block = encode_event_block(tokenizer, events, cursor)
            placed.extend(placed_block)
            cursor += len(block)

    start_code = gold_events[0].antecedent
    gold_answer = gold_events[-1].consequent
    suffix_ids = base.token_ids(tokenizer, base.build_prompt_suffix(start_code, chain_length))
    prompt_ids = torch.tensor(body + suffix_ids, dtype=torch.long).view(1, -1)
    return {
        "seed": seed,
        "condition": condition,
        "with_filler": int(with_filler),
        "with_conflict": int(with_conflict),
        "prompt_ids": prompt_ids,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "gold_answer": gold_answer,
        "conflict_answer": conflict_events[-1].consequent,
        "events": placed,
        "candidates": candidates,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        n = len(selected)
        candidate_accuracy = statistics.mean(int(row["candidate_correct"]) for row in selected)
        strict_accuracy = statistics.mean(int(row["generation_correct"]) for row in selected)
        final_accuracy = statistics.mean(int(row["generation_final_correct"]) for row in selected)
        contains_gold = statistics.mean(int(row["generation_contains_gold"]) for row in selected)
        nlls = [float(row["gold_candidate_mean_nll"]) for row in selected]
        best_wrong_nlls = [
            float(row["gold_candidate_mean_nll"]) + float(row["candidate_margin"])
            for row in selected
        ]
        mean_nll = statistics.mean(nlls)
        mean_best_wrong_nll = statistics.mean(best_wrong_nlls)
        nll_sem = statistics.stdev(nlls) / math.sqrt(n) if n > 1 else 0.0
        output.append(
            {
                "condition": condition,
                "sample_count": n,
                "candidate_correct_count": sum(int(row["candidate_correct"]) for row in selected),
                "generation_final_correct_count": sum(
                    int(row["generation_final_correct"]) for row in selected
                ),
                "mean_prompt_tokens": statistics.mean(int(row["prompt_tokens"]) for row in selected),
                "candidate_accuracy": candidate_accuracy,
                "candidate_accuracy_sem": math.sqrt(candidate_accuracy * (1 - candidate_accuracy) / n),
                "generation_strict_accuracy": strict_accuracy,
                "generation_final_accuracy": final_accuracy,
                "generation_final_accuracy_sem": math.sqrt(final_accuracy * (1 - final_accuracy) / n),
                "generation_contains_gold_rate": contains_gold,
                "mean_gold_answer_nll": mean_nll,
                "gold_answer_nll_sem": nll_sem,
                "gold_answer_ppl": math.exp(mean_nll),
                "mean_best_wrong_nll": mean_best_wrong_nll,
                "best_wrong_ppl": math.exp(mean_best_wrong_nll),
                "mean_candidate_margin": statistics.mean(float(row["candidate_margin"]) for row in selected),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generation_budget_metrics(
    tokenizer: Any,
    generation: dict[str, Any],
    budget: int,
    gold_answer: str,
    events: Sequence[base.RuleEvent],
    candidates: Sequence[str],
) -> dict[str, Any]:
    """Re-score a prefix of one long greedy generation without another model pass."""

    raw_text = str(generation["generated_text"]).replace("\\n", "\n")
    generated_ids = tokenizer(raw_text, add_special_tokens=False).input_ids[:budget]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    mentions = base.extract_known_code_mentions(text, gold_answer, list(events), list(candidates))
    explicit = base.extract_explicit_answers(text, gold_answer, list(events), list(candidates))
    last_known = mentions[-1] if mentions else None
    final_answer = explicit[-1] if explicit else last_known
    final_class = str(final_answer["answer_class"]) if final_answer else "miss"
    return {
        f"generation_{budget}_text": text.replace("\n", "\\n"),
        f"generation_{budget}_contains_gold": int(
            any(item["answer_class"] == "gold" for item in mentions)
        ),
        f"generation_{budget}_final_answer": "" if final_answer is None else final_answer["answer"],
        f"generation_{budget}_final_class": final_class,
        f"generation_{budget}_final_correct": int(final_class == "gold"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the exact conflict x filler 2x2 synthetic design.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=16)
    parser.add_argument("--filler_tokens", type=int, default=8192)
    parser.add_argument("--chain_length", type=int, default=2)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--report_generation_budgets",
        default="",
        help="Comma-separated prefixes to score from the single max_new_tokens generation, e.g. 16,128.",
    )
    parser.add_argument("--prefill_chunk_size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    generation_budgets = [
        int(item.strip())
        for item in args.report_generation_budgets.split(",")
        if item.strip()
    ]
    if any(budget <= 0 or budget > args.max_new_tokens for budget in generation_budgets):
        raise ValueError("report_generation_budgets must be positive and <= max_new_tokens")
    if args.candidate_count != 8:
        raise ValueError("this paired protocol currently fixes candidate_count=8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    seeds = range(args.seed_start, args.seed_start + args.num_seeds)
    variants = [
        build_variant(
            tokenizer,
            seed=seed,
            condition=condition,
            filler_tokens=args.filler_tokens,
            chain_length=args.chain_length,
        )
        for seed in seeds
        for condition in CONDITIONS
    ]
    preview = [
        {
            key: value
            for key, value in variant.items()
            if key not in {"prompt_ids", "events"}
        }
        | {"events": [asdict(event) for event in variant["events"]]}
        for variant in variants
    ]
    (output_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in preview) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(f"wrote {len(variants)} dry-run variants", flush=True)
        return

    max_position = max(int(variant["prompt_tokens"]) + args.max_new_tokens + 8 for variant in variants)
    model, tokenizer = base.load_model_and_tokenizer(args, max_position, 1.0)
    result_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        prompt_ids = variant["prompt_ids"]
        base_cache, prefill_seconds = base.prefill_sequence(
            model, prompt_ids[:, :-1], args.prefill_chunk_size
        )
        candidate_summary, _ = base.score_candidates(
            model,
            tokenizer,
            base_cache,
            prompt_ids[:, -1:],
            int(prompt_ids.shape[1]) - 1,
            variant["candidates"],
            variant["gold_answer"],
        )
        generation = base.generate_answer(
            model,
            tokenizer,
            base_cache,
            prompt_ids[:, -1:],
            int(prompt_ids.shape[1]) - 1,
            args.max_new_tokens,
            variant["gold_answer"],
            variant["events"],
            variant["candidates"],
        )
        budget_metrics: dict[str, Any] = {}
        for budget in generation_budgets:
            budget_metrics.update(
                generation_budget_metrics(
                    tokenizer,
                    generation,
                    budget,
                    variant["gold_answer"],
                    variant["events"],
                    variant["candidates"],
                )
            )
        row = {
            "seed": variant["seed"],
            "condition": variant["condition"],
            "with_filler": variant["with_filler"],
            "with_conflict": variant["with_conflict"],
            "prompt_tokens": variant["prompt_tokens"],
            "gold_answer": variant["gold_answer"],
            "conflict_answer": variant["conflict_answer"],
            "prefill_seconds": prefill_seconds,
            **candidate_summary,
            **generation,
            **budget_metrics,
        }
        result_rows.append(row)
        write_csv(output_dir / "results.csv", result_rows)
        print(
            f"[{index}/{len(variants)}] seed={variant['seed']} condition={variant['condition']} "
            f"candidate={candidate_summary['candidate_correct']} "
            f"final={generation['generation_final_correct']} "
            f"gold_nll={candidate_summary['gold_candidate_mean_nll']:.4f}",
            flush=True,
        )
        del base_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(result_rows)
    write_csv(output_dir / "summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
