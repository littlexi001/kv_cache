from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_local_rule_failure_boundary import (  # noqa: E402
    BuiltCase,
    RuleEvent,
    apply_rope_to_q,
    build_case,
    cache_from_legacy,
    forward_with_cache,
    input_device,
    legacy_cache,
    load_model_and_tokenizer,
    mutate_code,
    parse_csv_floats,
    parse_csv_ints,
    prefill_sequence,
    score_candidates,
    synchronize,
    token_ids,
)


@dataclass(frozen=True)
class PairedVariant:
    pair_id: str
    condition: str
    case: BuiltCase
    prompt_ids: torch.Tensor
    events: list[RuleEvent]
    candidates: list[str]
    prompt_token_diff_count: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def spans_for_kinds(events: list[RuleEvent], kinds: set[str]) -> list[tuple[int, int]]:
    return [(event.start_token, event.end_token) for event in events if event.kind in kinds]


def span_token_count(spans: list[tuple[int, int]]) -> int:
    return sum(max(0, end - start) for start, end in spans)


def span_mass(probabilities: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    value = torch.zeros((), device=probabilities.device, dtype=torch.float32)
    for start, end in spans:
        if end > start:
            value += probabilities[start:end].sum()
    return value


def span_mask(length: int, spans: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for start, end in spans:
        if end > start:
            mask[start:end] = True
    return mask


def find_length_preserving_nonconflict_text(
    tokenizer: Any,
    event: RuleEvent,
    forbidden_codes: set[str],
    rng: random.Random,
) -> tuple[str, str, list[int]]:
    expected_length = event.end_token - event.start_token
    tried: set[str] = set()
    for _ in range(5000):
        candidate = mutate_code(event.antecedent, rng)
        if candidate in tried or candidate in forbidden_codes or candidate == event.antecedent:
            continue
        tried.add(candidate)
        text = event.text.replace(event.antecedent, candidate, 1)
        ids = token_ids(tokenizer, text)
        if len(ids) == expected_length:
            return candidate, text, ids
    raise ValueError(
        f"Could not create a token-length-matched nonconflict antecedent for {event.label}; "
        f"expected {expected_length} tokens."
    )


def make_nonconflict_variant(
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    events: list[RuleEvent],
    seed: int,
) -> tuple[torch.Tensor, list[RuleEvent], int]:
    """Remove logical conflicts while preserving every event span and prompt length."""

    variant = prompt_ids.clone()
    rng = random.Random(seed)
    forbidden_codes = {
        code
        for event in events
        if event.kind == "relevant"
        for code in (event.antecedent, event.consequent)
    }
    converted_events: list[RuleEvent] = []
    for event in events:
        if event.kind != "conflict":
            converted_events.append(event)
            continue
        antecedent, text, ids = find_length_preserving_nonconflict_text(
            tokenizer, event, forbidden_codes, rng
        )
        start, end = event.start_token, event.end_token
        variant[0, start:end] = torch.tensor(ids, dtype=variant.dtype)
        converted_events.append(
            replace(event, kind="decoy", text=text, antecedent=antecedent)
        )
    difference_count = int((variant != prompt_ids).sum().item())
    if int(variant.numel()) != int(prompt_ids.numel()):
        raise AssertionError("Paired prompt length changed.")
    if any(
        event.kind == "decoy" and event.antecedent in forbidden_codes
        for event in converted_events
    ):
        raise AssertionError("Nonconflict antecedent still matches the gold chain.")
    return variant, converted_events, difference_count


def build_paired_variants(
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[PairedVariant]:
    variants: list[PairedVariant] = []
    pair_count = 0
    for length in parse_csv_ints(args.lengths):
        for depth in parse_csv_floats(args.depths):
            for seed in parse_csv_ints(args.seeds):
                for distractor_count in parse_csv_ints(args.distractor_counts):
                    for gap in parse_csv_ints(args.rule_gap_tokens):
                        for chain_length in parse_csv_ints(args.chain_lengths):
                            for competitor_count in parse_csv_ints(args.competitor_counts):
                                case, prompt_ids, events, candidates = build_case(
                                    tokenizer,
                                    model_label=args.model_label,
                                    target_context_tokens=length,
                                    depth_percent=depth,
                                    seed=seed,
                                    distractor_count=distractor_count,
                                    distractor_similarity="conflict",
                                    rule_gap_tokens=gap,
                                    chain_length=chain_length,
                                    competitor_count=competitor_count,
                                    max_new_tokens=1,
                                    original_max_position_embeddings=args.original_max_position_embeddings,
                                )
                                pair_id = case.case_id.replace("_conflict_", "_paired_")
                                conflict_case = replace(
                                    case,
                                    case_id=f"{pair_id}_conditionconflict",
                                )
                                non_prompt, non_events, difference_count = make_nonconflict_variant(
                                    tokenizer,
                                    prompt_ids,
                                    events,
                                    seed=(args.pair_mutation_seed + seed * 1009 + competitor_count * 9173),
                                )
                                non_case = replace(
                                    case,
                                    case_id=f"{pair_id}_conditionnonconflict",
                                    distractor_similarity="paired_nonconflict",
                                    distractor_rule_count=case.distractor_rule_count + case.conflict_rule_count,
                                    conflict_rule_count=0,
                                )
                                pair_variants = [
                                    PairedVariant(
                                        pair_id,
                                        "conflict",
                                        conflict_case,
                                        prompt_ids,
                                        events,
                                        candidates,
                                        difference_count,
                                    ),
                                    PairedVariant(
                                        pair_id,
                                        "nonconflict",
                                        non_case,
                                        non_prompt,
                                        non_events,
                                        candidates,
                                        difference_count,
                                    ),
                                ]
                                if (seed + competitor_count) % 2 == 0:
                                    pair_variants.reverse()
                                variants.extend(pair_variants)
                                pair_count += 1
                                if args.max_pairs > 0 and pair_count >= args.max_pairs:
                                    return variants
    return variants


@torch.inference_mode()
def attention_by_head(
    model: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    events: list[RuleEvent],
    top_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = input_device(model)
    layers = list(getattr(getattr(model, "model", None), "layers", []))
    if not layers:
        raise RuntimeError("Cannot locate model.model.layers.")
    captured_q: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None:
                return
            projected = module.q_proj(hidden_states)
            batch, q_len, _ = projected.shape
            head_dim = int(getattr(module, "head_dim"))
            num_heads = int(projected.shape[-1] // head_dim)
            q = projected.view(batch, q_len, num_heads, head_dim)
            q_norm = getattr(module, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            q = q.transpose(1, 2)
            q = apply_rope_to_q(q, position_embeddings)
            captured_q[layer_idx] = q[:, :, -1, :].detach()

        return hook

    for layer_idx, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
    synchronize()
    try:
        out = forward_with_cache(
            model,
            last_prompt_id.to(device),
            cache_from_legacy(base_cache),
            prompt_len_minus_one,
        )
    finally:
        for handle in handles:
            handle.remove()
    synchronize()
    cache = legacy_cache(out.past_key_values)
    if len(captured_q) != len(layers):
        raise RuntimeError(f"Captured {len(captured_q)} query layers, expected {len(layers)}.")

    relevant_spans = spans_for_kinds(events, {"relevant"})
    decoy_spans = spans_for_kinds(events, {"conflict", "decoy", "distractor"})
    competitor_spans = spans_for_kinds(events, {"competitor"})
    non_gold_spans = decoy_spans + competitor_spans
    gold_tokens = span_token_count(relevant_spans)
    decoy_tokens = span_token_count(decoy_spans)
    competitor_tokens = span_token_count(competitor_spans)
    head_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for layer_idx, layer in enumerate(layers):
        q = captured_q[layer_idx][0]
        key = cache[layer_idx][0][0]
        num_heads = int(q.shape[0])
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        kv_len = int(key.shape[1])
        scale = float(getattr(layer.self_attn, "scaling", q.shape[-1] ** -0.5))
        gold_mask = span_mask(kv_len, relevant_spans, key.device)
        decoy_mask = span_mask(kv_len, decoy_spans, key.device)
        competitor_mask = span_mask(kv_len, competitor_spans, key.device)
        top_count = min(kv_len, max(1, math.ceil(top_fraction * kv_len)))

        for head_idx in range(num_heads):
            kv_idx = min(kv_heads - 1, head_idx // groups)
            logits = torch.matmul(key[kv_idx].float(), q[head_idx].float()) * scale
            probabilities = torch.softmax(logits.float(), dim=-1)
            gold_mass = float(span_mass(probabilities, relevant_spans))
            decoy_mass = float(span_mass(probabilities, decoy_spans))
            competitor_mass = float(span_mass(probabilities, competitor_spans))
            non_gold_mass = decoy_mass + competitor_mass
            background_mass = max(0.0, 1.0 - gold_mass - non_gold_mass)
            top_indices = torch.topk(logits, k=top_count, largest=True).indices
            top_mask = torch.zeros(kv_len, dtype=torch.bool, device=logits.device)
            top_mask[top_indices] = True
            gold_top_tokens = int((top_mask & gold_mask).sum())
            decoy_top_tokens = int((top_mask & decoy_mask).sum())
            competitor_top_tokens = int((top_mask & competitor_mask).sum())
            gold_top_mass = float(probabilities[top_mask & gold_mask].sum())
            gold_density = gold_mass / max(gold_tokens, 1)
            decoy_density = decoy_mass / max(decoy_tokens, 1)
            background_tokens = max(1, kv_len - gold_tokens - decoy_tokens - competitor_tokens)
            background_density = background_mass / background_tokens
            step_masses: dict[str, float] = {}
            for event in events:
                if event.kind == "relevant":
                    step_masses[f"gold_step_{event.step}_mass"] = float(
                        probabilities[event.start_token : event.end_token].sum()
                    )
            sorted_indices = torch.argsort(logits, descending=True)
            ranks = torch.empty_like(sorted_indices)
            ranks[sorted_indices] = torch.arange(1, kv_len + 1, device=logits.device)
            gold_ranks = ranks[gold_mask]
            head_rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "kv_head": kv_idx,
                    "kv_length": kv_len,
                    "top_fraction": top_fraction,
                    "top_token_count": top_count,
                    "gold_rule_tokens": gold_tokens,
                    "decoy_rule_tokens": decoy_tokens,
                    "competitor_rule_tokens": competitor_tokens,
                    "gold_rule_mass": gold_mass,
                    "decoy_rule_mass": decoy_mass,
                    "competitor_rule_mass": competitor_mass,
                    "non_gold_rule_mass": non_gold_mass,
                    "background_mass": background_mass,
                    "gold_rule_selectivity": gold_mass / max(gold_mass + non_gold_mass, 1e-30),
                    "gold_uniform_enrichment": gold_mass / max(gold_tokens / kv_len, 1e-30),
                    "gold_vs_decoy_density_ratio": gold_density / max(decoy_density, 1e-30),
                    "gold_vs_decoy_log2_density_ratio": math.log2(
                        max(gold_density, 1e-30) / max(decoy_density, 1e-30)
                    ),
                    "gold_vs_background_density_ratio": gold_density / max(background_density, 1e-30),
                    "gold_top2_tokens": gold_top_tokens,
                    "decoy_top2_tokens": decoy_top_tokens,
                    "competitor_top2_tokens": competitor_top_tokens,
                    "gold_top2_token_recall": gold_top_tokens / max(gold_tokens, 1),
                    "gold_top2_token_precision": gold_top_tokens / top_count,
                    "gold_top2_mass_recall": gold_top_mass / max(gold_mass, 1e-30),
                    "gold_best_token_rank": int(gold_ranks.min()) if gold_ranks.numel() else -1,
                    "gold_mean_token_rank": float(gold_ranks.float().mean()) if gold_ranks.numel() else -1,
                    **step_masses,
                }
            )

            for event in events:
                event_mass = float(probabilities[event.start_token : event.end_token].sum())
                event_top_tokens = int(top_mask[event.start_token : event.end_token].sum())
                event_tokens = event.end_token - event.start_token
                event_rows.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "kv_head": kv_idx,
                        "event_kind": event.kind,
                        "event_label": event.label,
                        "event_step": event.step,
                        "event_start_token": event.start_token,
                        "event_end_token": event.end_token,
                        "event_tokens": event_tokens,
                        "event_attention_mass": event_mass,
                        "event_uniform_enrichment": event_mass / max(event_tokens / kv_len, 1e-30),
                        "event_top2_tokens": event_top_tokens,
                        "event_top2_token_recall": event_top_tokens / max(event_tokens, 1),
                    }
                )
    return head_rows, event_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired per-head attention study on gold symbolic rules.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--model_label", default="qwen3_0p6b")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="8192")
    parser.add_argument("--depths", default="50")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--distractor_counts", default="16")
    parser.add_argument("--rule_gap_tokens", default="512")
    parser.add_argument("--chain_lengths", default="2")
    parser.add_argument("--competitor_counts", default="0,4")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--pair_mutation_seed", type=int, default=20260714)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    variants = build_paired_variants(tokenizer, args)
    if not variants:
        raise SystemExit("No paired variants were built.")
    preview_rows = []
    for variant in variants:
        preview_rows.append(
            {
                "pair_id": variant.pair_id,
                "condition": variant.condition,
                "case": asdict(variant.case),
                "prompt_token_diff_count": variant.prompt_token_diff_count,
                "events": [asdict(event) for event in variant.events],
                "candidates": variant.candidates,
            }
        )
    (output_dir / "paired_cases.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in preview_rows) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"wrote {len(variants)} paired variants to {output_dir}", flush=True)
        return

    max_position = max(variant.case.max_position_embeddings for variant in variants)
    max_factor = max(variant.case.rope_factor for variant in variants)
    model, tokenizer = load_model_and_tokenizer(args, max_position, max_factor)
    env = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "model_name_or_path": args.model_name_or_path,
        "variant_count": len(variants),
        "pair_count": len(variants) // 2,
        "args": vars(args),
    }
    (output_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    all_head_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants, start=1):
        case = variant.case
        print(
            f"[{variant_index}/{len(variants)}] pair={variant.pair_id} condition={variant.condition} "
            f"competitors={case.competitor_count}",
            flush=True,
        )
        prompt_prefix = variant.prompt_ids[:, :-1]
        last_prompt_id = variant.prompt_ids[:, -1:]
        base_cache, prefill_seconds = prefill_sequence(model, prompt_prefix, args.prefill_chunk_size)
        candidate_summary, scores = score_candidates(
            model,
            tokenizer,
            base_cache,
            last_prompt_id,
            case.actual_prompt_tokens - 1,
            variant.candidates,
            case.gold_answer,
        )
        started = time.perf_counter()
        head_rows, event_rows = attention_by_head(
            model,
            base_cache,
            last_prompt_id,
            case.actual_prompt_tokens - 1,
            variant.events,
            args.top_fraction,
        )
        attention_seconds = time.perf_counter() - started
        metadata = {
            "pair_id": variant.pair_id,
            "condition": variant.condition,
            "case_id": case.case_id,
            "seed": case.seed,
            "target_context_tokens": case.target_context_tokens,
            "depth_percent": case.depth_percent,
            "distractor_count": case.distractor_count,
            "rule_gap_tokens": case.rule_gap_tokens,
            "chain_length": case.chain_length,
            "competitor_count": case.competitor_count,
            "prompt_token_diff_count": variant.prompt_token_diff_count,
            **candidate_summary,
        }
        for row in head_rows:
            all_head_rows.append({**metadata, **row})
        for row in event_rows:
            all_event_rows.append({**metadata, **row})
        case_rows.append(
            {
                **metadata,
                "actual_prompt_tokens": case.actual_prompt_tokens,
                "prefill_seconds": prefill_seconds,
                "attention_seconds": attention_seconds,
                "mean_gold_rule_mass": mean([float(row["gold_rule_mass"]) for row in head_rows]),
                "mean_gold_rule_selectivity": mean(
                    [float(row["gold_rule_selectivity"]) for row in head_rows]
                ),
                "mean_gold_uniform_enrichment": mean(
                    [float(row["gold_uniform_enrichment"]) for row in head_rows]
                ),
                "mean_gold_top2_token_recall": mean(
                    [float(row["gold_top2_token_recall"]) for row in head_rows]
                ),
            }
        )
        for rank, score in enumerate(scores, start=1):
            candidate_rows.append({**metadata, "rank": rank, **score})
        write_csv(output_dir / "case_results.csv", case_rows)
        write_csv(output_dir / "candidate_scores.csv", candidate_rows)
        write_csv(output_dir / "head_attention.csv", all_head_rows)
        write_csv(output_dir / "head_event_attention.csv", all_event_rows)
        print(
            f"done condition={variant.condition} correct={candidate_summary['candidate_correct']} "
            f"margin={candidate_summary['candidate_margin']:.4f} "
            f"gold_mass={case_rows[-1]['mean_gold_rule_mass']:.6f} "
            f"selectivity={case_rows[-1]['mean_gold_rule_selectivity']:.4f}",
            flush=True,
        )
        del base_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
