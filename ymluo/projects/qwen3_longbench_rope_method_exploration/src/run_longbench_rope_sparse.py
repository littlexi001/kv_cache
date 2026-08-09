from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
RULE_SRC = PROJECTS / "qwen3_local_rule_failure_boundary" / "src"
ORACLE_SRC = PROJECTS / "qwen3_longbench_oracle_evidence" / "src"
for directory in (RULE_SRC, ORACLE_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_local_global_rope_probe_8b as local_global  # noqa: E402
import run_local_rule_failure_boundary as cache_runner  # noqa: E402
import run_rope_retrieval_repair_8b as rope_repair  # noqa: E402
import run_hotpot_oracle_pilot as oracle  # noqa: E402


VARIANTS = (
    "native_full",
    "full_rope_replay",
    "rope_top2",
    "semantic_top2_postscore",
    "local_global_postscore",
    "local_global_blend25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Position-preserving LongBench comparison for RoPE-aware sparse retrieval."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--longbench-jsonl", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--evidence-mapping", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
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


def contiguous_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    limit = len(haystack) - len(needle) + 1
    for start in range(limit):
        if haystack[start] == first and list(haystack[start : start + len(needle)]) == list(needle):
            return start, start + len(needle)
    return None


def align_evidence_spans(
    tokenizer: Any,
    prompt_text: str,
    context: str,
    mapping: dict[str, Any],
) -> tuple[list[int], tuple[tuple[int, int], ...], list[dict[str, Any]]]:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    context_start = prompt_text.find(context)
    if context_start < 0:
        raise RuntimeError("Full LongBench context is not a literal substring of the chat prompt")

    passages = oracle.parse_longbench_passages(context)
    by_title: dict[str, list[Any]] = {}
    for passage in passages:
        by_title.setdefault(oracle.normalize_title(passage.title), []).append(passage)

    spans: list[tuple[int, int]] = []
    audits: list[dict[str, Any]] = []
    for record in mapping["support_alignment_records"]:
        title = str(record["title"])
        title_matches = by_title.get(oracle.normalize_title(title), [])
        if len(title_matches) != 1:
            raise RuntimeError(f"support title does not map uniquely: {title!r}")
        passage = title_matches[0]
        matched_text = str(record["matched_text"])
        region = context[int(passage.char_start) : int(passage.char_end)]
        local_char = region.find(matched_text)
        token_span: tuple[int, int] | None = None
        alignment_mode = "character_exact"
        if local_char >= 0:
            char_start = context_start + int(passage.char_start) + local_char
            char_end = char_start + len(matched_text)
            positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < char_end and end > char_start
            ]
            if positions:
                token_span = (min(positions), max(positions) + 1)
        if token_span is None:
            # Boundary whitespace can change the first subword. Search both
            # surface forms inside the token range of the uniquely titled passage.
            passage_start = context_start + int(passage.char_start)
            passage_end = context_start + int(passage.char_end)
            passage_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < passage_end and end > passage_start
            ]
            if not passage_positions:
                raise RuntimeError(f"no prompt tokens overlap support passage {title!r}")
            search_start, search_end = min(passage_positions), max(passage_positions) + 1
            for surface in (matched_text, " " + matched_text):
                needle = oracle.token_ids(tokenizer, surface)
                found = contiguous_subsequence(prompt_ids[search_start:search_end], needle)
                if found is not None:
                    token_span = (search_start + found[0], search_start + found[1])
                    alignment_mode = "token_subsequence"
                    break
        if token_span is None:
            raise RuntimeError(f"could not align accepted support text in prompt: {title!r}")
        spans.append(token_span)
        audits.append(
            {
                "title": title,
                "match_type": record.get("match_type"),
                "prompt_token_start": token_span[0],
                "prompt_token_end": token_span[1],
                "prompt_token_count": token_span[1] - token_span[0],
                "alignment_mode": alignment_mode,
            }
        )
    if not spans:
        raise RuntimeError("sample has no aligned support spans")
    return prompt_ids, tuple(spans), audits


class AuditedController(local_global.Controller):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.collect_metrics = True
        self.support_observations = 0
        self.support_budget_violations = 0
        self.duplicate_support_violations = 0
        self.support_size_min: int | None = None
        self.support_size_max: int | None = None

    def record(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        key_count: int,
        remote_mask: torch.Tensor | None,
    ) -> None:
        if not self.collect_metrics:
            return
        selected = int(positions.shape[-1])
        expected = int(key_count) if self.variant == "full_rope" else min(
            int(key_count), max(1, int(math.ceil(self.ratio * int(key_count))))
        )
        self.support_observations += int(positions.shape[0])
        if selected != expected:
            self.support_budget_violations += int(positions.shape[0])
        duplicate_rows = int((positions.sort(dim=-1).values.diff(dim=-1) == 0).any(dim=-1).sum().item())
        self.duplicate_support_violations += duplicate_rows
        self.support_size_min = selected if self.support_size_min is None else min(self.support_size_min, selected)
        self.support_size_max = selected if self.support_size_max is None else max(self.support_size_max, selected)
        super().record(positions, weights, key_count, remote_mask)

    def record_semantic_gap(
        self,
        post_scores: torch.Tensor,
        calibrated_pre_scores: torch.Tensor,
        remote_mask: torch.Tensor,
    ) -> None:
        if self.collect_metrics:
            super().record_semantic_gap(post_scores, calibrated_pre_scores, remote_mask)

    def audit_summary(self) -> dict[str, Any]:
        denominator = max(1, self.support_observations)
        return {
            "support_observations": self.support_observations,
            "support_budget_violation_fraction": self.support_budget_violations / denominator,
            "duplicate_support_violation_fraction": self.duplicate_support_violations / denominator,
            "support_size_min": self.support_size_min,
            "support_size_max": self.support_size_max,
        }


def make_controller(
    variant: str,
    args: argparse.Namespace,
    evidence_spans: tuple[tuple[int, int], ...],
) -> AuditedController | None:
    if variant == "native_full":
        return None
    internal = "full_rope" if variant == "full_rope_replay" else variant
    return AuditedController(
        variant=internal,
        ratio=float(args.ratio),
        minimum_keep_tokens=0,
        maximum_keep_tokens=0,
        local_window=int(args.local_window),
        sink_tokens=int(args.sink_tokens),
        evidence_spans=evidence_spans,
    )


def run_last_prompt_token(
    model: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
    controller: AuditedController | None,
) -> tuple[torch.Tensor, Any, float]:
    cache_runner.synchronize()
    started = time.perf_counter()
    with local_global.activate(controller), torch.inference_mode():
        output = cache_runner.forward_with_cache(
            model,
            prompt[:, -1:].to(cache_runner.input_device(model)),
            cache,
            prefix_length,
        )
    cache_runner.synchronize()
    return output.logits[:, -1, :].float(), output.past_key_values, time.perf_counter() - started


def eos_ids(tokenizer: Any) -> set[int]:
    values: set[int] = set()
    if tokenizer.eos_token_id is not None:
        values.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            values.add(token_id)
    return values


@torch.inference_mode()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    first_logits: torch.Tensor,
    cache: Any,
    prompt_length: int,
    max_new_tokens: int,
    controller: AuditedController | None,
) -> tuple[str, list[int], float]:
    started = time.perf_counter()
    generated: list[int] = []
    stops = eos_ids(tokenizer)
    logits = first_logits
    past_length = int(prompt_length)
    for step in range(int(max_new_tokens)):
        next_id = int(logits.argmax(dim=-1).item())
        if next_id in stops:
            break
        generated.append(next_id)
        if step + 1 == int(max_new_tokens):
            break
        token = torch.tensor([[next_id]], dtype=torch.long, device=cache_runner.input_device(model))
        with local_global.activate(controller):
            output = cache_runner.forward_with_cache(model, token, cache, past_length)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        past_length += 1
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated, time.perf_counter() - started


def answer_candidates(tokenizer: Any, answer: str) -> list[list[int]]:
    surfaces = [answer]
    if oracle.normalize_answer(answer) in {"yes", "no"}:
        surfaces.append(answer.strip().capitalize())
    output: list[list[int]] = []
    for surface in dict.fromkeys(surfaces):
        ids = oracle.token_ids(tokenizer, surface)
        if ids and ids not in output:
            output.append(ids)
    return output


@torch.inference_mode()
def gold_answer_nll(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
    answer: str,
    args: argparse.Namespace,
    variant: str,
    evidence_spans: tuple[tuple[int, int], ...],
) -> tuple[float, int, float]:
    candidates = answer_candidates(tokenizer, answer)
    if not candidates:
        return float("nan"), 0, 0.0
    controller = make_controller(variant, args, evidence_spans)
    if controller is not None:
        controller.collect_metrics = False
    started = time.perf_counter()
    first_logits, cache, _ = run_last_prompt_token(model, prompt, cache, prefix_length, controller)
    if all(len(ids) == 1 for ids in candidates):
        losses = [-float(F.log_softmax(first_logits, dim=-1)[0, ids[0]].item()) for ids in candidates]
        return min(losses), 1, time.perf_counter() - started
    ids = candidates[0]
    logits = first_logits
    losses: list[float] = []
    past_length = int(prompt.shape[1])
    for index, token_id in enumerate(ids):
        losses.append(-float(F.log_softmax(logits, dim=-1)[0, token_id].item()))
        if index + 1 < len(ids):
            token = torch.tensor([[token_id]], dtype=torch.long, device=cache_runner.input_device(model))
            with local_global.activate(controller):
                output = cache_runner.forward_with_cache(model, token, cache, past_length)
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            past_length += 1
    return sum(losses) / len(losses), len(ids), time.perf_counter() - started


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        mean_nll = sum(float(row["gold_answer_mean_nll"]) for row in selected) / len(selected)
        def mean(name: str) -> float | None:
            values = [float(row[name]) for row in selected if row.get(name) is not None]
            return sum(values) / len(values) if values else None
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "qa_f1_percent": 100.0 * mean("official_qa_f1"),
                "em_percent": 100.0 * mean("normalized_exact_match"),
                "gold_answer_mean_nll": mean_nll,
                "gold_answer_ppl": math.exp(min(mean_nll, 30.0)),
                "first_token_accuracy_percent": 100.0 * mean("first_token_correct"),
                "gold_evidence_token_recall": mean("gold_evidence_token_recall"),
                "gold_chain_complete_rate": mean("gold_chain_complete_rate"),
                "gold_evidence_attention_mass": mean("gold_evidence_attention_mass"),
                "mean_query_seconds": mean("query_seconds"),
                "mean_generation_seconds": mean("generation_seconds"),
                "support_budget_violation_fraction": mean("support_budget_violation_fraction"),
                "duplicate_support_violation_fraction": mean("duplicate_support_violation_fraction"),
                "dense_replay_max_logit_error": mean("dense_replay_max_logit_error"),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or not variants:
        raise ValueError(f"unknown or empty variants: {unknown}")
    if not (0.0 < args.ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard-index must be in [0, shard-count)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = read_jsonl(args.frozen_manifest)
    if args.max_samples > 0:
        manifests = manifests[: args.max_samples]
    shard = [row for index, row in enumerate(manifests) if index % args.shard_count == args.shard_index]
    evidence = {row["sample_id"]: row for row in read_jsonl(args.evidence_mapping)}
    full_predictions = {
        row["sample_id"]: row
        for row in read_jsonl(args.frozen_predictions)
        if row.get("condition") == "full"
    }
    longbench = {str(row.get("_id", index)): row for index, row in enumerate(read_jsonl(args.longbench_jsonl))}
    missing = [row["sample_id"] for row in shard if row["sample_id"] not in longbench]
    if missing:
        raise RuntimeError(f"frozen samples missing from LongBench JSONL: {missing}")

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_variants": variants,
        "frozen_sample_count": len(manifests),
        "shard_sample_count": len(shard),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "selector_uses_gold": False,
        "consumer": "native positions + native post-RoPE QK + original V",
    }
    write_json(args.output_dir / "config.json", config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=False,
    )
    model, tokenizer = local_global.load_model(load_args)
    local_global.patch_model(model)

    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {str(row["sample_id"]) for row in existing}
    for item in shard:
        sample_id = str(item["sample_id"])
        if sample_id in completed:
            print(f"sample={sample_id} already complete", flush=True)
            continue
        source = longbench[sample_id]
        context = str(source["context"])
        question = str(source["input"])
        answers = [str(value) for value in source["answers"]]
        prompt_text = oracle.chat_prompt(tokenizer, question, context)
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        expected_hash = str(full_predictions[sample_id]["prompt_sha256"])
        if prompt_hash != expected_hash:
            raise RuntimeError(f"prompt hash mismatch for {sample_id}: {prompt_hash} != {expected_hash}")
        prompt_ids, evidence_spans, alignment_audit = align_evidence_spans(
            tokenizer, prompt_text, context, evidence[sample_id]
        )
        if len(prompt_ids) != int(item["full_prompt_tokens"]):
            raise RuntimeError(
                f"prompt token mismatch for {sample_id}: {len(prompt_ids)} != {item['full_prompt_tokens']}"
            )
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prefix_length = int(prompt.shape[1]) - 1
        legacy, prefill_seconds = cache_runner.prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size
        )
        cache = cache_runner.cache_from_legacy(legacy)
        del legacy
        sample_rows: list[dict[str, Any]] = []
        native_logits: torch.Tensor | None = None
        for variant in variants:
            controller = make_controller(variant, args, evidence_spans)
            logits, cache, query_seconds = run_last_prompt_token(
                model, prompt, cache, prefix_length, controller
            )
            if variant == "native_full":
                native_logits = logits.detach().cpu()
            replay_error = None
            if variant == "full_rope_replay" and native_logits is not None:
                replay_error = float((logits.detach().cpu() - native_logits).abs().max().item())
            first_gold_ids = answer_candidates(tokenizer, answers[0])[0]
            first_token_id = int(first_gold_ids[0])
            first_token_nll = -float(F.log_softmax(logits, dim=-1)[0, first_token_id].item())
            first_token_correct = int(int(logits.argmax(dim=-1).item()) == first_token_id)
            metrics = controller.metrics.summary() if controller is not None else {}
            audits = controller.audit_summary() if controller is not None else {}
            if controller is not None:
                controller.collect_metrics = False
            prediction, generated_ids, generation_seconds = greedy_generate(
                model,
                tokenizer,
                logits,
                cache,
                int(prompt.shape[1]),
                args.max_new_tokens,
                controller,
            )
            rope_repair.reset_dynamic_cache(cache, prefix_length)
            nll, answer_tokens, nll_seconds = gold_answer_nll(
                model,
                tokenizer,
                prompt,
                cache,
                prefix_length,
                answers[0],
                args,
                variant,
                evidence_spans,
            )
            rope_repair.reset_dynamic_cache(cache, prefix_length)
            if float(audits.get("support_budget_violation_fraction", 0.0)) != 0.0:
                raise RuntimeError(f"support budget violation in {sample_id}/{variant}")
            if float(audits.get("duplicate_support_violation_fraction", 0.0)) != 0.0:
                raise RuntimeError(f"duplicate support violation in {sample_id}/{variant}")
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "question": question,
                    "answers": answers,
                    "prediction": prediction,
                    "generated_token_ids": generated_ids,
                    "official_qa_f1": oracle.official_score(prediction, answers),
                    "normalized_exact_match": int(oracle.normalized_exact_match(prediction, answers)),
                    "prediction_contains_answer": int(oracle.contains_answer(prediction, answers)),
                    "prompt_tokens": len(prompt_ids),
                    "prompt_sha256": prompt_hash,
                    "evidence_position_bin": item["evidence_position_bin"],
                    "hotpot_type": item["hotpot_type"],
                    "aligned_evidence_spans": [list(value) for value in evidence_spans],
                    "alignment_audit": alignment_audit,
                    "ratio": args.ratio,
                    "expected_keep_tokens_at_first_query": int(math.ceil(args.ratio * len(prompt_ids))),
                    "local_window": args.local_window,
                    "sink_tokens": args.sink_tokens,
                    "first_token_id": first_token_id,
                    "first_token_nll": first_token_nll,
                    "first_token_correct": first_token_correct,
                    "gold_answer_mean_nll": nll,
                    "gold_answer_ppl": math.exp(min(nll, 30.0)),
                    "gold_answer_tokens": answer_tokens,
                    "prefill_seconds": prefill_seconds,
                    "query_seconds": query_seconds,
                    "generation_seconds": generation_seconds,
                    "nll_seconds": nll_seconds,
                    "dense_replay_max_logit_error": replay_error,
                    **metrics,
                    **audits,
                }
            )
            print(
                f"sample={sample_id[:8]} variant={variant} f1={sample_rows[-1]['official_qa_f1']:.3f} "
                f"em={sample_rows[-1]['normalized_exact_match']} nll={nll:.4f} "
                f"recall={sample_rows[-1].get('gold_evidence_token_recall')}",
                flush=True,
            )
        append_jsonl(rows_path, sample_rows)
        all_rows = read_jsonl(rows_path)
        write_csv(args.output_dir / "rows.csv", all_rows)
        write_csv(args.output_dir / "summary.csv", summarize(all_rows))
        del cache, prompt
        local_global.clear_allocator()

    all_rows = read_jsonl(rows_path) if rows_path.exists() else []
    write_csv(args.output_dir / "rows.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()

