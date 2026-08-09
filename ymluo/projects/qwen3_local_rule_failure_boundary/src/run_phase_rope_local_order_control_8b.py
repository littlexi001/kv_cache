from __future__ import annotations

"""Frozen-Qwen3 control separating local order from remote retrieval.

The experiment is intentionally small and diagnostic.  It reuses the exact
attention interventions in ``run_phase_coherent_rope_probe_8b`` and evaluates
two task families:

* ``local_order``: a counterfactual pair has exactly the same four words and
  query, but swaps the two words after the anchor.  The correct successor must
  therefore change solely because local order changed.  The sequence is put
  immediately before the question, even when a long filler prefix is used.
* ``remote_retrieval``: the controlled two-hop rule task used by the main RoPE
  probe, with evidence near the beginning and the query at the end.

Only the final query-token attention is changed.  The model is frozen and the
prefix is always encoded with its native RoPE, so this is an inference-time
local-safety/remote-benefit control rather than an end-to-end training result.
"""

import argparse
import csv
import gc
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

import run_length_causal_mechanism_20260717 as causal
import run_phase_coherent_rope_probe_8b as phase
import run_native_phase_envelope_rollback_8b as npe
import run_rope_retrieval_repair_8b as rope_repair


runner = phase.runner
base = runner.base

DEFAULT_VARIANTS = (
    "full_rope",
    "rope_top2",
    "exact_pre_top2_postscore",
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25",
    "strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_masspreserve",
    "npe_native_pre_top2",
    "npe_rollback_pre_top2",
    "npe_rollback_masspreserve_pre_top2",
)
TASK_FAMILIES = ("local_order", "remote_retrieval")


def uses_shared_exact_pre_support(variant: str) -> bool:
    """Whether the variant uses the same exact-pre sparse token selector."""

    return phase._uses_exact_pre_selector(variant) or variant in npe.NPE_VARIANTS


def install_attention_adapter() -> None:
    """Install NPE dispatch while retaining phase/strict fallback variants.

    ``native_phase_envelope_attention_forward`` delegates every non-NPE
    variant to the phase runner it captured at import time.  Installing this
    adapter explicitly before model patching avoids relying on incidental
    module-import order.  Restore the NPE fallback explicitly as well because
    other experiment modules legitimately replace that mutable dispatch while
    sharing the same Python process (for example, in the full test suite).
    """

    npe._PHASE_FORWARD = phase.phase_kernel_attention_forward
    runner.local_global_attention_forward = npe.native_phase_envelope_attention_forward


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive comma-separated integers, got {text!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Qwen3-8B control: preserve a nearby order-sensitive relation "
            "while improving remote two-hop retrieval."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-lengths", default="8192,32768")
    parser.add_argument("--remote-lengths", default="8192,32768")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument(
        "--task-families",
        default=",".join(TASK_FAMILIES),
        help="Comma-separated subset of local_order,remote_retrieval.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated attention variants registered by the phase runner.",
    )
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--minimum-keep-tokens", type=int, default=0)
    parser.add_argument("--maximum-keep-tokens", type=int, default=0)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=70000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_order_pair(words: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a same-bag counterfactual pair whose correct successor changes."""

    if len(words) != 4 or len(set(words)) != 4:
        raise ValueError("local order control needs exactly four distinct words")
    first, anchor, left, right = words
    pair_id = "__".join(words)
    return (
        {
            "pair_id": pair_id,
            "pair_member": 0,
            "sequence_words": (first, anchor, left, right),
            "anchor": anchor,
            "gold": left,
            "candidates": tuple(words),
        },
        {
            "pair_id": pair_id,
            "pair_member": 1,
            "sequence_words": (first, anchor, right, left),
            "anchor": anchor,
            "gold": right,
            "candidates": tuple(words),
        },
    )


def _leading_word_id(tokenizer: Any, word: str) -> int:
    ids = base.token_ids(tokenizer, f" {word}")
    if len(ids) != 1:
        raise RuntimeError(f"local answer is not one leading-space token: {word} -> {ids}")
    return int(ids[0])


def _local_word_pool(tokenizer: Any, required: int = 32) -> list[str]:
    candidates = causal.build_english_single_token_code_pool(
        tokenizer, required=max(64, required)
    )
    output: list[str] = []
    for word in candidates:
        try:
            _leading_word_id(tokenizer, word)
        except RuntimeError:
            continue
        output.append(word)
        if len(output) >= required:
            return output
    raise RuntimeError(f"found only {len(output)} local-order answer words")


def encode_order_block(
    tokenizer: Any, sequence_words: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Encode a readable sequence while retaining exact positions of its words."""

    block = base.token_ids(
        tokenizer,
        "\nLOCAL ORDER RECORD\nRead the following exact sequence from left to right:",
    )
    positions: list[int] = []
    for index, word in enumerate(sequence_words):
        positions.append(len(block))
        block.append(_leading_word_id(tokenizer, word))
        if index + 1 < len(sequence_words):
            block.extend(base.token_ids(tokenizer, ","))
    block.extend(base.token_ids(tokenizer, ".\n"))
    return block, positions


def build_local_order_case(
    tokenizer: Any,
    *,
    target_context_tokens: int,
    seed: int,
    pair_member: int,
    word_pool: Sequence[str],
) -> dict[str, Any]:
    rng = random.Random(2026080107 + seed * 1009)
    chosen = list(word_pool)
    rng.shuffle(chosen)
    spec = make_order_pair(chosen[:4])[pair_member]
    block, word_positions = encode_order_block(tokenizer, spec["sequence_words"])
    if len(block) > target_context_tokens:
        raise ValueError(
            f"local order block has {len(block)} tokens, exceeds target "
            f"length {target_context_tokens}"
        )
    body = base.build_filler_ids(
        tokenizer, target_context_tokens, 3_100_000 + seed
    )
    block_start = target_context_tokens - len(block)
    body[block_start:] = block
    question = (
        "\nLOCAL ORDER QUESTION\n"
        f"Which word occurs immediately after {spec['anchor']} in the exact "
        "sequence above? Use the sequence order, ignore all earlier prose, and "
        "reply with one word only."
    )
    wrapper_prefix, wrapper_suffix = causal.chat_wrapper_ids(tokenizer)
    prompt_ids = (
        wrapper_prefix
        + body
        + base.token_ids(tokenizer, question)
        + wrapper_suffix
    )
    offset = len(wrapper_prefix) + block_start
    anchor_index = spec["sequence_words"].index(spec["anchor"])
    gold_index = spec["sequence_words"].index(spec["gold"])
    evidence_positions = (
        offset + word_positions[anchor_index],
        offset + word_positions[gold_index],
    )
    query_position = len(prompt_ids) - 1
    return {
        "case_id": (
            f"local_order_L{target_context_tokens}_S{seed}_P{pair_member}"
        ),
        "task_family": "local_order",
        "target_context_tokens": target_context_tokens,
        "seed": seed,
        "pair_id": spec["pair_id"],
        "pair_member": pair_member,
        "prompt": torch.tensor(prompt_ids, dtype=torch.long).view(1, -1),
        "gold": spec["gold"],
        "candidates": spec["candidates"],
        "anchor": spec["anchor"],
        "sequence_words": spec["sequence_words"],
        # Two one-token spans force a head to retain both the anchor and its
        # successor for the strict chain-complete metric.
        "evidence_spans": tuple((position, position + 1) for position in evidence_positions),
        "minimum_evidence_query_distance": min(
            query_position - position for position in evidence_positions
        ),
        "maximum_evidence_query_distance": max(
            query_position - position for position in evidence_positions
        ),
    }


def build_remote_case(tokenizer: Any, *, target_context_tokens: int, seed: int) -> dict[str, Any]:
    case = runner.seeded_case(tokenizer, target_context_tokens, seed)
    query_position = int(case["prompt"].shape[1]) - 1
    distances = [
        query_position - position
        for start, end in case["evidence_spans"]
        for position in range(start, end)
    ]
    return {
        "case_id": f"remote_retrieval_L{target_context_tokens}_S{seed}",
        "task_family": "remote_retrieval",
        "target_context_tokens": target_context_tokens,
        "seed": seed,
        "pair_id": "",
        "pair_member": -1,
        "prompt": case["prompt"],
        "gold": case["codes"][-1],
        "candidates": tuple(case["codes"]),
        "anchor": case["codes"][0],
        "sequence_words": tuple(case["codes"]),
        "evidence_spans": case["evidence_spans"],
        "minimum_evidence_query_distance": min(distances),
        "maximum_evidence_query_distance": max(distances),
    }


def choice_metrics(
    tokenizer: Any,
    logits: torch.Tensor,
    gold: str,
    candidates: Sequence[str],
) -> dict[str, Any]:
    ids = [_leading_word_id(tokenizer, word) for word in candidates]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"candidate token ids are not unique: {candidates} -> {ids}")
    gold_id = _leading_word_id(tokenizer, gold)
    if gold_id not in ids:
        raise ValueError(f"gold {gold!r} is not in candidates {candidates}")
    final_logits = logits[0, -1].float()
    candidate_logits = final_logits[torch.tensor(ids, device=final_logits.device)]
    winning_index = int(candidate_logits.argmax().item())
    wrong = candidate_logits[
        torch.tensor(
            [index for index, token_id in enumerate(ids) if token_id != gold_id],
            device=final_logits.device,
        )
    ]
    gold_logit = float(final_logits[gold_id].item())
    strongest_wrong = float(wrong.max().item()) if wrong.numel() else -math.inf
    return {
        "candidate_correct": int(ids[winning_index] == gold_id),
        "candidate_prediction": candidates[winning_index],
        "candidate_margin": gold_logit - strongest_wrong,
        "gold_logit": gold_logit,
        "strongest_wrong_candidate_logit": strongest_wrong,
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def nontrigger_noop_error(
    variant: str,
    attention_metrics: dict[str, Any],
) -> float:
    """Return the variant-specific exact no-op audit in common units."""

    if variant in phase.STRICT_MPR_VARIANTS:
        return float(attention_metrics["strict_phase_nontrigger_noop_max"])
    if variant in npe.NPE_VARIANTS:
        return float(attention_metrics["npe_unmodified_native_max_error"])
    # Baselines do not have an intervention trigger; treating them as exact
    # no-ops makes the unified column directly comparable.
    return 0.0


def _local_pair_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(int(row["seed"]), str(row["pair_id"]))].append(row)
    complete = [pair for pair in by_pair.values() if len(pair) == 2]
    if not complete:
        return {
            "counterfactual_pair_accuracy": 0.0,
            "prediction_changes_with_order_rate": 0.0,
            "complete_counterfactual_pairs": 0.0,
        }
    return {
        "counterfactual_pair_accuracy": statistics.fmean(
            float(all(int(row["candidate_correct"]) == 1 for row in pair))
            for pair in complete
        ),
        "prediction_changes_with_order_rate": statistics.fmean(
            float(len({str(row["candidate_prediction"]) for row in pair}) == 2)
            for pair in complete
        ),
        "complete_counterfactual_pairs": float(len(complete)),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (
                str(row["task_family"]),
                int(row["target_context_tokens"]),
                str(row["variant"]),
            )
            for row in rows
        }
    )
    for family, length, variant in groups:
        selected = [
            row
            for row in rows
            if str(row["task_family"]) == family
            and int(row["target_context_tokens"]) == length
            and str(row["variant"]) == variant
        ]
        mean_nll = _mean(selected, "gold_nll")
        item: dict[str, Any] = {
            "task_family": family,
            "target_context_tokens": length,
            "variant": variant,
            "sample_count": len(selected),
            "candidate_accuracy": _mean(selected, "candidate_correct"),
            "next_token_accuracy": _mean(selected, "next_token_correct"),
            "mean_candidate_margin": _mean(selected, "candidate_margin"),
            "mean_gold_nll": mean_nll,
            "gold_ppl": math.exp(mean_nll),
            "gold_evidence_token_recall": _mean(
                selected, "gold_evidence_token_recall"
            ),
            "gold_evidence_line_hit_rate": _mean(
                selected, "gold_evidence_line_hit_rate"
            ),
            "gold_chain_complete_rate": _mean(
                selected, "gold_chain_complete_rate"
            ),
            "gold_evidence_attention_mass": _mean(
                selected, "gold_evidence_attention_mass"
            ),
            "shared_exact_pre_support": int(
                uses_shared_exact_pre_support(variant)
            ),
            "nontrigger_exact_noop_max": max(
                float(row.get("nontrigger_exact_noop_max", 0.0))
                for row in selected
            ),
            "mean_query_seconds": _mean(selected, "query_seconds"),
        }
        if family == "local_order":
            item.update(_local_pair_metrics(selected))
        else:
            item.update(
                {
                    "counterfactual_pair_accuracy": "",
                    "prediction_changes_with_order_rate": "",
                    "complete_counterfactual_pairs": "",
                }
            )
        output.append(item)

    baselines = {
        (str(row["task_family"]), int(row["target_context_tokens"])): row
        for row in output
        if row["variant"] == "full_rope"
    }
    for row in output:
        baseline = baselines.get(
            (str(row["task_family"]), int(row["target_context_tokens"]))
        )
        if baseline is None:
            row.update(
                {
                    "candidate_accuracy_delta_vs_full": "",
                    "gold_nll_delta_vs_full": "",
                    "evidence_mass_delta_vs_full": "",
                }
            )
            continue
        row.update(
            {
                "candidate_accuracy_delta_vs_full": float(
                    row["candidate_accuracy"]
                )
                - float(baseline["candidate_accuracy"]),
                "gold_nll_delta_vs_full": float(row["mean_gold_nll"])
                - float(baseline["mean_gold_nll"]),
                "evidence_mass_delta_vs_full": float(
                    row["gold_evidence_attention_mass"]
                )
                - float(baseline["gold_evidence_attention_mass"]),
            }
        )
    return output


def protocol() -> dict[str, Any]:
    return {
        "model_state": "frozen; no training or parameter updates",
        "intervention_scope": (
            "final query-token attention only; every prefix representation uses "
            "native pretrained RoPE"
        ),
        "matched_support": (
            "exact_pre_top2_postscore, strict-MPR, and all NPE variants use the "
            "same exact pre-RoPE local/global Top-2% token support"
        ),
        "no_op_audit": (
            "nontrigger_exact_noop_max must be exactly zero for strict-MPR and "
            "NPE, including their trigger-only mass-preserving variants"
        ),
        "local_order": {
            "construction": (
                "two prompts have the same four-word multiset and identical "
                "question; swapping the two post-anchor words changes the gold successor"
            ),
            "primary_metrics": [
                "counterfactual_pair_accuracy",
                "candidate_accuracy",
                "mean_candidate_margin",
                "gold_ppl",
                "nontrigger_exact_noop_max",
            ],
            "safety_criterion": (
                "candidate accuracy/pair accuracy should not decrease versus full_rope"
            ),
        },
        "remote_retrieval": {
            "construction": "controlled two-hop VERIFIED RULE chain near the prefix",
            "primary_metrics": [
                "gold_evidence_token_recall",
                "gold_chain_complete_rate",
                "gold_evidence_attention_mass",
                "gold_ppl",
                "candidate_accuracy",
            ],
        },
    }


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_stream(
    tokenizer: Any,
    families: Sequence[str],
    local_lengths: Sequence[int],
    remote_lengths: Sequence[int],
    seed_start: int,
    num_seeds: int,
) -> Iterable[dict[str, Any]]:
    word_pool = _local_word_pool(tokenizer) if "local_order" in families else []
    for seed in range(seed_start, seed_start + num_seeds):
        if "local_order" in families:
            for length in local_lengths:
                for pair_member in (0, 1):
                    yield build_local_order_case(
                        tokenizer,
                        target_context_tokens=length,
                        seed=seed,
                        pair_member=pair_member,
                        word_pool=word_pool,
                    )
        if "remote_retrieval" in families:
            for length in remote_lengths:
                yield build_remote_case(
                    tokenizer, target_context_tokens=length, seed=seed
                )


def main() -> None:
    args = parse_args()
    local_lengths = parse_int_list(args.local_lengths)
    remote_lengths = parse_int_list(args.remote_lengths)
    families = [
        item.strip() for item in args.task_families.split(",") if item.strip()
    ]
    unknown_families = sorted(set(families) - set(TASK_FAMILIES))
    if not families or unknown_families:
        raise ValueError(f"unknown task families: {unknown_families}")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown_variants = sorted(set(variants) - set(runner.VARIANTS))
    if not variants or unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive")
    if args.minimum_keep_tokens < 0 or args.maximum_keep_tokens < 0:
        raise ValueError("keep-token bounds must be non-negative")
    if (
        args.minimum_keep_tokens > 0
        and args.maximum_keep_tokens > 0
        and args.minimum_keep_tokens > args.maximum_keep_tokens
    ):
        raise ValueError("minimum keep tokens cannot exceed maximum")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "resolved_local_lengths": local_lengths,
        "resolved_remote_lengths": remote_lengths,
        "resolved_task_families": families,
        "resolved_variants": variants,
        "cuda_visible_devices": __import__("os").environ.get(
            "CUDA_VISIBLE_DEVICES", ""
        ),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "protocol.json", protocol())
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    install_attention_adapter()
    model, tokenizer = runner.load_model(args)
    runner.patch_model(model)
    rows_path = output_dir / "rows.jsonl"
    existing = _read_rows(rows_path)
    completed = {
        (str(row["case_id"]), str(row["variant"])) for row in existing
    }

    for case in _case_stream(
        tokenizer,
        families,
        local_lengths,
        remote_lengths,
        args.seed_start,
        args.num_seeds,
    ):
        missing = [
            variant
            for variant in variants
            if (str(case["case_id"]), variant) not in completed
        ]
        if not missing:
            print(f"{case['case_id']} already complete", flush=True)
            continue
        if (
            case["task_family"] == "local_order"
            and int(case["maximum_evidence_query_distance"]) > args.local_window
        ):
            raise RuntimeError(
                f"{case['case_id']} is not local: maximum evidence-query "
                f"distance {case['maximum_evidence_query_distance']} exceeds "
                f"local window {args.local_window}"
            )
        prompt = case["prompt"]
        prefix_length = int(prompt.shape[1]) - 1
        base.synchronize()
        legacy, prefill_seconds = base.prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size
        )
        cache = base.cache_from_legacy(legacy)
        del legacy
        case_rows: list[dict[str, Any]] = []
        for variant in missing:
            controller = runner.Controller(
                variant=variant,
                ratio=args.ratio,
                minimum_keep_tokens=args.minimum_keep_tokens,
                maximum_keep_tokens=args.maximum_keep_tokens,
                local_window=args.local_window,
                sink_tokens=args.sink_tokens,
                evidence_spans=case["evidence_spans"],
            )
            base.synchronize()
            started = time.perf_counter()
            with runner.activate(controller), torch.inference_mode():
                output = base.forward_with_cache(
                    model,
                    prompt[:, -1:].to(base.input_device(model)),
                    cache,
                    prefix_length,
                )
            base.synchronize()
            query_seconds = time.perf_counter() - started
            answer = runner.answer_metrics(tokenizer, output.logits, case["gold"])
            choices = choice_metrics(
                tokenizer, output.logits, case["gold"], case["candidates"]
            )
            attention_metrics = controller.metrics.summary()
            row = {
                "case_id": case["case_id"],
                "task_family": case["task_family"],
                "target_context_tokens": case["target_context_tokens"],
                "prompt_tokens": int(prompt.shape[1]),
                "seed": case["seed"],
                "pair_id": case["pair_id"],
                "pair_member": case["pair_member"],
                "variant": variant,
                "anchor": case["anchor"],
                "gold": case["gold"],
                "sequence_or_chain": " -> ".join(case["sequence_words"]),
                "candidate_words": "|".join(case["candidates"]),
                "minimum_evidence_query_distance": case[
                    "minimum_evidence_query_distance"
                ],
                "maximum_evidence_query_distance": case[
                    "maximum_evidence_query_distance"
                ],
                **attention_metrics,
                "shared_exact_pre_support": int(
                    uses_shared_exact_pre_support(variant)
                ),
                "nontrigger_exact_noop_max": nontrigger_noop_error(
                    variant, attention_metrics
                ),
                **answer,
                **choices,
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
            }
            case_rows.append(row)
            completed.add((str(case["case_id"]), variant))
            del output
            rope_repair.reset_dynamic_cache(cache, prefix_length)

        with rows_path.open("a", encoding="utf-8") as handle:
            for row in case_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_rows = _read_rows(rows_path)
        summary = summarize(all_rows)
        write_csv(output_dir / "rows.csv", all_rows)
        write_csv(output_dir / "summary.csv", summary)
        write_json(output_dir / "summary.json", summary)
        print(
            f"{case['case_id']} prompt={prompt.shape[1]} "
            f"prefill={prefill_seconds:.2f}s",
            flush=True,
        )
        for row in case_rows:
            print(
                f"  {row['variant']}: candidate={row['candidate_correct']} "
                f"margin={row['candidate_margin']:.3f} ppl={row['gold_ppl']:.3f} "
                f"evidence_mass={row['gold_evidence_attention_mass']:.6f}",
                flush=True,
            )
        del cache, prompt
        clear_allocator()

    all_rows = _read_rows(rows_path)
    summary = summarize(all_rows)
    write_csv(output_dir / "rows.csv", all_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
