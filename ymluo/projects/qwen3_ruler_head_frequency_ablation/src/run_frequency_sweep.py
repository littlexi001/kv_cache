from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
PROJECTS = HERE.parents[2]
RNOPE_SRC = PROJECTS / "qwen3_inference_rnope" / "src"
RULER_SRC = PROJECTS / "qwen3_ruler32k_rope_method" / "src"
LONG_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
for directory in (PROJECT / "src", RNOPE_SRC, RULER_SRC, LONG_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from head_frequency_intervention import HeadFrequencyIntervention  # noqa: E402
import run_inference_rnope_ruler as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--specs-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--target-length", type=int, default=32768)
    parser.add_argument("--max-new-tokens-cap", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        help="Transformers/Accelerate placement policy for one or more visible GPUs.",
    )
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--spec-shard-count", type=int, default=1)
    parser.add_argument("--spec-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    return parser.parse_args()


def load_specs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value["specs"] if isinstance(value, dict) else value
    if not isinstance(specs, list) or not specs:
        raise ValueError("specs file must contain a non-empty list")
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("spec names must be unique")
    return specs


def resolve_query_tail_spec(spec: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    """Resolve sample-relative query-tail gates into absolute token positions."""
    resolved = copy.deepcopy(spec)
    for atom in resolved.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) != "relative_distance":
            continue
        tail_tokens = atom.get("query_tail_tokens")
        if tail_tokens is None:
            continue
        tail_tokens = int(tail_tokens)
        if tail_tokens <= 0:
            raise ValueError("query_tail_tokens must be positive")
        atom["query_position_start"] = max(0, int(prompt_tokens) - tail_tokens)
        atom["query_position_end"] = int(prompt_tokens)
    return resolved


def select_examples(path: Path, sample_ids: str, target_length: int) -> list[Any]:
    rows = base.ruler.read_jsonl(path)
    requested = [value.strip() for value in sample_ids.split(",") if value.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    if requested:
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"missing sample ids: {missing}")
        selected_rows = [by_id[value] for value in requested]
    else:
        selected_rows = [row for row in rows if int(row["length"]) == int(target_length)]
        selected_rows.sort(key=lambda row: str(row["sample_id"]))
    return [base.ruler.public.Example(**row) for row in selected_rows]


def ensure_dynamic_cache(model: Any, cache: Any) -> Any:
    """Normalize legacy tuple caches emitted by patched attention on newer Transformers."""
    if cache is None or hasattr(cache, "get_mask_sizes"):
        return cache
    from transformers.cache_utils import DynamicCache

    try:
        converted = DynamicCache.from_legacy_cache(cache)
    except Exception:
        converted = DynamicCache(ddp_cache_data=cache, config=model.config)
    if not hasattr(converted, "get_mask_sizes"):
        raise TypeError(f"cache conversion returned {type(converted)!r}")
    return converted


def prefill_sequence_compat(
    model: Any, prompt_prefix: torch.Tensor, chunk_size: int
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], float]:
    device = base.core.cache_runner.input_device(model)
    ids = prompt_prefix.to(device)
    past = None
    past_len = 0
    base.core.cache_runner.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(ids.shape[1]), chunk_size):
            chunk = ids[:, start : start + chunk_size]
            output = base.core.cache_runner.forward_with_cache(model, chunk, past, past_len)
            past = ensure_dynamic_cache(model, output.past_key_values)
            past_len += int(chunk.shape[1])
    base.core.cache_runner.synchronize()
    return base.core.cache_runner.legacy_cache(past), time.perf_counter() - started


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.spec_shard_count <= 0 or not 0 <= args.spec_shard_index < args.spec_shard_count:
        raise ValueError("invalid spec shard")
    all_specs = load_specs(args.specs_json)
    specs = [
        spec for index, spec in enumerate(all_specs)
        if index % args.spec_shard_count == args.spec_shard_index
    ]
    if not specs:
        raise RuntimeError("this shard has no specs")
    examples = select_examples(args.examples_jsonl, args.sample_ids, args.target_length)
    if args.sample_shard_count <= 0:
        raise ValueError("sample-shard-count must be positive")
    if not 0 <= args.sample_shard_index < args.sample_shard_count:
        raise ValueError("sample-shard-index must lie in [0, sample-shard-count)")
    examples = examples[args.sample_shard_index :: args.sample_shard_count]
    if not examples:
        raise ValueError("sample shard selected no examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=bool(args.load_in_4bit),
        device_map=args.device_map,
    )
    model, tokenizer = base.core.local_global.load_model(load_args)
    intervention = HeadFrequencyIntervention(model)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_specs": specs,
        "selected_examples": [asdict(example) | {"context": "<omitted>"} for example in examples],
        "model_shape": {
            "layers": intervention.num_layers,
            "query_heads": intervention.num_query_heads,
            "kv_heads": intervention.num_kv_heads,
            "head_dim": intervention.head_dim,
        },
        "weights_frozen": True,
        "intervention": (
            "spec-driven layer/head-group/frequency RoPE intervention; supports "
            "absolute phase scaling and locality-preserving relative-distance score repair"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    base.write_json(args.output_dir / "config.json", config)

    rows_path = args.output_dir / "rows.jsonl"
    existing = base.read_jsonl(rows_path) if rows_path.exists() else []
    completed = {(str(row["sample_id"]), str(row["variant"])) for row in existing}
    original_logits: dict[str, torch.Tensor] = {}

    for spec_index, spec in enumerate(specs, start=1):
        variant = str(spec["name"])
        print(f"spec [{spec_index}/{len(specs)}] {variant}", flush=True)
        for sample_index, example in enumerate(examples, start=1):
            if (example.sample_id, variant) in completed:
                continue
            task_name = base.ruler.base_task(example.task)
            prompt, _, _ = base.ruler.make_prompt(tokenizer, example)
            prompt_tokens = int(prompt.shape[-1])
            prefix_length = prompt_tokens - 1
            active_spec = resolve_query_tail_spec(spec, prompt_tokens)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            with intervention.activate(active_spec):
                started = time.perf_counter()
                legacy, prefill_seconds = prefill_sequence_compat(
                    model, prompt[:, :-1], args.prefill_chunk_size
                )
                cache = ensure_dynamic_cache(model, legacy)
                del legacy
                logits, cache, query_seconds = base.run_last(model, prompt, cache, prefix_length)
                if not base.finite_logits(logits):
                    raise RuntimeError(f"non-finite logits: {example.sample_id}/{variant}")
                if bool(spec.get("reference_original", False)):
                    original_logits[example.sample_id] = logits.detach().cpu()
                replay_error = None
                if bool(spec.get("compare_to_original", False)) and example.sample_id in original_logits:
                    replay_error = float(
                        (logits.detach().cpu() - original_logits[example.sample_id]).abs().max().item()
                    )
                first_nll, first_correct = base.first_answer_stats(
                    tokenizer, logits, example.answers[0]
                )
                query_legacy = base.core.cache_runner.legacy_cache(cache)
                max_new = min(int(example.max_new_tokens), int(args.max_new_tokens_cap))
                prediction, generated_ids, generation_seconds = base.greedy_generate(
                    model,
                    tokenizer,
                    logits,
                    ensure_dynamic_cache(model, query_legacy),
                    prompt_tokens,
                    max_new,
                )
                gold_nll, gold_tokens, nll_seconds = base.gold_answer_nll(
                    model,
                    tokenizer,
                    prompt,
                    ensure_dynamic_cache(model, query_legacy),
                    prefix_length,
                    example.answers[0],
                )
                official_score = base.ruler.public.score_prediction(
                    example.metric,
                    prediction,
                    example.answers,
                    example.all_classes,
                    task=task_name,
                )
                elapsed = time.perf_counter() - started

            row = {
                "sample_id": example.sample_id,
                "task": task_name,
                "prompt_tokens": prompt_tokens,
                "variant": variant,
                "spec": active_spec,
                "metric": example.metric,
                "answers": example.answers,
                "prediction": prediction,
                "generated_token_ids": generated_ids,
                "official_score": official_score,
                "first_answer_next_token_nll": first_nll,
                "first_answer_next_token_correct": first_correct,
                "gold_answer_mean_nll": gold_nll,
                "gold_answer_ppl": math.exp(min(gold_nll, 30.0)),
                "gold_answer_tokens": gold_tokens,
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "generation_seconds": generation_seconds,
                "nll_seconds": nll_seconds,
                "elapsed_seconds": elapsed,
                "patched_vs_original_max_logit_error": replay_error,
                "finite_logits": True,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
            base.append_jsonl(rows_path, [row])
            completed.add((example.sample_id, variant))
            print(
                f"  [{sample_index}/{len(examples)}] {example.sample_id} "
                f"score={official_score:.3f} nll={gold_nll:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            del cache, prompt
            base.core.local_global.clear_allocator()

    base.write_json(args.output_dir / "status.json", {"complete": True, "rows": len(completed)})
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
