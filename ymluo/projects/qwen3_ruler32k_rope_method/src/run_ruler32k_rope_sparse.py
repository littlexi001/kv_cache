from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
LONG_ROPE_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
PUBLIC_SRC = PROJECTS / "qwen3_top2_head_limit3_ppl" / "src"
for directory in (LONG_ROPE_SRC, PUBLIC_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_longbench_rope_sparse as core  # noqa: E402
import run_controlled_public_kv_benchmark_v1 as public  # noqa: E402


DEFAULT_VARIANTS = (
    "native_full",
    "rope_top2",
    "local_global_postscore",
    "local_global_blend25",
)
ALLOWED_VARIANTS = set(core.VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired Qwen3-8B RULER-32K query-time RoPE sparse retrieval pilot."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--target-length", type=int, default=32768)
    parser.add_argument("--max-samples-per-task", type=int, default=2)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_task(task: str) -> str:
    stem, separator, suffix = task.rpartition("_")
    if not separator or not suffix.isdigit():
        raise ValueError(f"RULER task does not end in a numeric length: {task}")
    return stem


def select_examples(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[public.Example]:
    requested = {item.strip() for item in args.tasks.split(",") if item.strip()}
    counts: dict[str, int] = {}
    selected: list[public.Example] = []
    for row in rows:
        example = public.Example(**row)
        task_name = base_task(example.task)
        if int(example.length) != int(args.target_length):
            continue
        if requested and task_name not in requested:
            continue
        if counts.get(task_name, 0) >= int(args.max_samples_per_task):
            continue
        counts[task_name] = counts.get(task_name, 0) + 1
        selected.append(example)
    selected.sort(key=lambda value: (base_task(value.task), value.sample_id))
    return [
        example
        for index, example in enumerate(selected)
        if index % int(args.shard_count) == int(args.shard_index)
    ]


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(map(int, tokenizer(text, add_special_tokens=False)["input_ids"]))


def all_subsequences(haystack: Sequence[int], needle: Sequence[int]) -> list[tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return []
    output: list[tuple[int, int]] = []
    for start in range(len(haystack) - len(needle) + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            output.append((start, start + len(needle)))
    return output


def merge_spans(spans: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(spans))
    if not ordered:
        return ()
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def answer_evidence_spans(
    tokenizer: Any,
    context_ids: Sequence[int],
    answers: Sequence[str],
    task_name: str,
) -> tuple[tuple[int, int], ...]:
    if not (task_name.startswith("niah_") or task_name.startswith("qa_")):
        return ()
    spans: set[tuple[int, int]] = set()
    for answer in answers:
        for surface in (answer, " " + answer):
            spans.update(all_subsequences(context_ids, token_ids(tokenizer, surface)))
    return merge_spans(spans)


def first_answer_token_candidates(tokenizer: Any, answer: str) -> list[int]:
    candidates: list[int] = []
    for surface in (answer, " " + answer, "\n" + answer):
        ids = token_ids(tokenizer, surface)
        if ids and ids[0] not in candidates:
            candidates.append(ids[0])
    return candidates


def official_answer_coverage(prediction: str, answers: Sequence[str]) -> float:
    lowered = prediction.lower()
    return sum(float(answer.lower() in lowered) for answer in answers) / max(1, len(answers))


def make_prompt(tokenizer: Any, example: public.Example) -> tuple[torch.Tensor, int, str]:
    prompt_text = example.context + example.query
    context = token_ids(tokenizer, example.context)
    prompt = token_ids(tokenizer, prompt_text)
    if prompt[: len(context)] != context:
        # Boundary tokenization can merge across context/query. The experiment
        # still uses the exact full prompt; this count only bounds evidence search.
        context = prompt[: max(0, len(prompt) - len(token_ids(tokenizer, example.query)))]
    return torch.tensor(prompt, dtype=torch.long).view(1, -1), len(context), prompt_text


def controller_for(
    variant: str,
    args: argparse.Namespace,
    evidence_spans: tuple[tuple[int, int], ...],
) -> core.AuditedController | None:
    return core.make_controller(variant, args, evidence_spans)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - ALLOWED_VARIANTS)
    if unknown or not variants:
        raise ValueError(f"unknown or empty variants: {unknown}")
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in (0,1]")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard count/index")

    examples = select_examples(read_jsonl(args.examples_jsonl), args)
    if not examples:
        raise RuntimeError("no examples selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_variants": variants,
        "selected_examples": [asdict(example) | {"context": "<omitted>"} for example in examples],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "selector_uses_gold": False,
        "consumer": "native positions + original V; native post-RoPE scores except blend25",
    }
    core.write_json(args.output_dir / "config.json", config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
        return

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model, tokenizer = core.local_global.load_model(load_args)
    core.local_global.patch_model(model)

    rows_path = args.output_dir / "rows.jsonl"
    existing = core.read_jsonl(rows_path) if rows_path.exists() else []
    completed = {(str(row["sample_id"]), str(row["variant"])) for row in existing}

    for sample_index, example in enumerate(examples, start=1):
        task_name = base_task(example.task)
        prompt, context_count, prompt_text = make_prompt(tokenizer, example)
        prompt_tokens = int(prompt.shape[-1])
        # RULER's task generators reserve different output/template margins;
        # FWE is normally around 28.8K even when max_seq_length is 32K.
        if prompt_tokens < int(0.85 * args.target_length) or prompt_tokens > args.target_length + 64:
            raise RuntimeError(
                f"length_mismatch {example.sample_id}: prompt={prompt_tokens}, target={args.target_length}"
            )
        evidence_spans = answer_evidence_spans(
            tokenizer,
            prompt[0, :context_count].tolist(),
            example.answers,
            task_name,
        )
        prefix_length = prompt_tokens - 1
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        legacy, prefill_seconds = core.cache_runner.prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size
        )
        cache = core.cache_runner.cache_from_legacy(legacy)
        del legacy
        print(
            f"[{sample_index}/{len(examples)}] {example.sample_id} tokens={prompt_tokens} "
            f"evidence_spans={len(evidence_spans)} prefill={prefill_seconds:.2f}s",
            flush=True,
        )

        native_logits: torch.Tensor | None = None
        for variant in variants:
            if (example.sample_id, variant) in completed:
                print(f"  {variant}: already complete", flush=True)
                continue
            controller = controller_for(variant, args, evidence_spans)
            logits, cache, query_seconds = core.run_last_prompt_token(
                model, prompt, cache, prefix_length, controller
            )
            replay_error = None
            if variant == "native_full":
                native_logits = logits.detach().cpu()
            elif variant == "full_rope_replay" and native_logits is not None:
                replay_error = float((logits.detach().cpu() - native_logits).abs().max().item())

            first_ids = first_answer_token_candidates(tokenizer, example.answers[0])
            log_probs = F.log_softmax(logits.float(), dim=-1)[0]
            first_nll = min(-float(log_probs[token_id].item()) for token_id in first_ids)
            first_correct = int(int(logits.argmax(dim=-1).item()) in set(first_ids))
            metrics = controller.metrics.summary() if controller is not None else {}
            audits = controller.audit_summary() if controller is not None else {}
            if controller is not None:
                controller.collect_metrics = False
            max_new = min(int(example.max_new_tokens), int(args.max_new_tokens_cap))
            prediction, generated_ids, generation_seconds = core.greedy_generate(
                model,
                tokenizer,
                logits,
                cache,
                prompt_tokens,
                max_new,
                controller,
            )
            core.rope_repair.reset_dynamic_cache(cache, prefix_length)
            if float(audits.get("support_budget_violation_fraction", 0.0)) != 0.0:
                raise RuntimeError(f"support_budget_violation {example.sample_id}/{variant}")
            if float(audits.get("duplicate_support_violation_fraction", 0.0)) != 0.0:
                raise RuntimeError(f"duplicate_support {example.sample_id}/{variant}")

            score = public.score_prediction(
                example.metric,
                prediction,
                example.answers,
                example.all_classes,
                task=task_name,
            )
            row = {
                "sample_id": example.sample_id,
                "task": task_name,
                "requested_length": int(example.length),
                "prompt_tokens": prompt_tokens,
                "variant": variant,
                "metric": example.metric,
                "answers": example.answers,
                "prediction": prediction,
                "generated_token_ids": generated_ids,
                "generated_tokens": len(generated_ids),
                "official_score": score,
                "answer_coverage": official_answer_coverage(prediction, example.answers),
                "first_answer_next_token_nll": first_nll,
                "first_answer_next_token_ppl": math.exp(min(first_nll, 30.0)),
                "first_answer_next_token_correct": first_correct,
                "answer_evidence_span_count": len(evidence_spans),
                "answer_evidence_spans": [list(span) for span in evidence_spans],
                "ratio": float(args.ratio),
                "expected_keep_tokens_at_first_query": int(math.ceil(args.ratio * prompt_tokens)),
                "local_window": int(args.local_window),
                "sink_tokens": int(args.sink_tokens),
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "generation_seconds": generation_seconds,
                "dense_replay_max_logit_error": replay_error,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
                ),
                **metrics,
                **audits,
            }
            append_jsonl(rows_path, [row])
            completed.add((example.sample_id, variant))
            print(
                f"  {variant}: score={score:.3f} first_nll={first_nll:.3f} "
                f"recall={row.get('gold_evidence_token_recall')} pred={prediction[:100]!r}",
                flush=True,
            )

        del cache, prompt
        core.local_global.clear_allocator()

    core.write_json(args.output_dir / "status.json", {"complete": True, "rows": len(core.read_jsonl(rows_path))})
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
