from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
PROJECTS = HERE.parents[2]
LONG_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
for directory in (PROJECT / "src", LONG_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from head_frequency_intervention import HeadFrequencyIntervention  # noqa: E402
import run_longbench_rope_sparse as longbench  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense LongBench evaluation for layer/head/frequency RoPE scaling specs."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--longbench-jsonl", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--frozen-predictions", required=True, type=Path)
    parser.add_argument("--specs-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_specs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value["specs"] if isinstance(value, dict) else value
    if not isinstance(specs, list) or not specs:
        raise ValueError("specs-json must contain a non-empty specs list")
    names = [str(spec["name"]) for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("spec names must be unique")
    return specs


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_last(
    model: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
) -> tuple[torch.Tensor, Any, float]:
    longbench.cache_runner.synchronize()
    started = time.perf_counter()
    output = longbench.cache_runner.forward_with_cache(
        model,
        prompt[:, -1:].to(longbench.cache_runner.input_device(model)),
        cache,
        prefix_length,
    )
    longbench.cache_runner.synchronize()
    return output.logits[:, -1, :].float(), output.past_key_values, time.perf_counter() - started


@torch.inference_mode()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    first_logits: torch.Tensor,
    cache: Any,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[str, list[int], float]:
    started = time.perf_counter()
    generated: list[int] = []
    stop_ids = longbench.eos_ids(tokenizer)
    logits = first_logits
    past_length = int(prompt_length)
    for step in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        if step + 1 == max_new_tokens:
            break
        token = torch.tensor(
            [[token_id]], dtype=torch.long, device=longbench.cache_runner.input_device(model)
        )
        output = longbench.cache_runner.forward_with_cache(model, token, cache, past_length)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        past_length += 1
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, time.perf_counter() - started


@torch.inference_mode()
def gold_answer_nll(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
    answer: str,
) -> tuple[float, int, float]:
    candidates = longbench.answer_candidates(tokenizer, answer)
    if not candidates:
        return float("nan"), 0, 0.0
    started = time.perf_counter()
    first_logits, cache, _ = run_last(model, prompt, cache, prefix_length)
    if all(len(candidate) == 1 for candidate in candidates):
        losses = [
            -float(F.log_softmax(first_logits, dim=-1)[0, candidate[0]].item())
            for candidate in candidates
        ]
        return min(losses), 1, time.perf_counter() - started
    tokens = candidates[0]
    logits = first_logits
    losses: list[float] = []
    past_length = int(prompt.shape[1])
    for index, token_id in enumerate(tokens):
        losses.append(-float(F.log_softmax(logits, dim=-1)[0, token_id].item()))
        if index + 1 < len(tokens):
            token = torch.tensor(
                [[token_id]], dtype=torch.long, device=longbench.cache_runner.input_device(model)
            )
            output = longbench.cache_runner.forward_with_cache(model, token, cache, past_length)
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            past_length += 1
    return sum(losses) / len(losses), len(tokens), time.perf_counter() - started


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        mean_nll = mean(float(row["gold_answer_mean_nll"]) for row in selected)
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "qa_f1_percent": 100.0 * mean(float(row["official_qa_f1"]) for row in selected),
                "em_percent": 100.0 * mean(float(row["normalized_exact_match"]) for row in selected),
                "contains_answer_percent": 100.0
                * mean(float(row["prediction_contains_answer"]) for row in selected),
                "first_token_accuracy_percent": 100.0
                * mean(float(row["first_token_correct"]) for row in selected),
                "gold_answer_mean_nll": mean_nll,
                "gold_answer_ppl": math.exp(min(mean_nll, 30.0)),
                "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in selected),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = read_specs(args.specs_json)
    manifests = read_jsonl(args.frozen_manifest)
    if args.max_samples > 0:
        manifests = manifests[: args.max_samples]
    longbench_rows = {
        str(row.get("_id", index)): row
        for index, row in enumerate(read_jsonl(args.longbench_jsonl))
    }
    frozen_predictions = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.frozen_predictions)
        if row.get("condition") == "full"
    }
    cases = [
        (spec, manifest)
        for spec in specs
        for manifest in manifests
    ]
    cases = [case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index]
    if not cases:
        raise RuntimeError("this shard has no cases")

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model, tokenizer = longbench.local_global.load_model(load_args)
    intervention = HeadFrequencyIntervention(model)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "specs": specs,
        "case_count": len(cases),
        "weights_frozen": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(args.output_dir / "config.json", config)
    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {(str(row["variant"]), str(row["sample_id"])) for row in existing}

    for case_index, (spec, manifest) in enumerate(cases, start=1):
        variant = str(spec["name"])
        sample_id = str(manifest["sample_id"])
        if (variant, sample_id) in completed:
            continue
        source = longbench_rows[sample_id]
        answers = [str(value) for value in source["answers"]]
        prompt_text = longbench.oracle.chat_prompt(
            tokenizer, str(source["input"]), str(source["context"])
        )
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        expected_hash = str(frozen_predictions[sample_id]["prompt_sha256"])
        if prompt_hash != expected_hash:
            raise RuntimeError(f"prompt hash mismatch for {sample_id}")
        prompt_ids = longbench.oracle.token_ids(tokenizer, prompt_text)
        if len(prompt_ids) != int(manifest["full_prompt_tokens"]):
            raise RuntimeError(
                f"prompt token mismatch for {sample_id}: {len(prompt_ids)} != "
                f"{manifest['full_prompt_tokens']}"
            )
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prefix_length = int(prompt.shape[1]) - 1
        with intervention.activate(spec):
            started = time.perf_counter()
            legacy, prefill_seconds = longbench.cache_runner.prefill_sequence(
                model, prompt[:, :-1], args.prefill_chunk_size
            )
            cache = longbench.cache_runner.cache_from_legacy(legacy)
            del legacy
            logits, cache, query_seconds = run_last(model, prompt, cache, prefix_length)
            first_ids = longbench.answer_candidates(tokenizer, answers[0])[0]
            first_token_id = int(first_ids[0])
            first_token_nll = -float(F.log_softmax(logits, dim=-1)[0, first_token_id].item())
            first_token_correct = int(int(logits.argmax(dim=-1).item()) == first_token_id)
            prediction, generated_ids, generation_seconds = greedy_generate(
                model, tokenizer, logits, cache, len(prompt_ids), args.max_new_tokens
            )
            longbench.rope_repair.reset_dynamic_cache(cache, prefix_length)
            gold_nll, gold_tokens, nll_seconds = gold_answer_nll(
                model, tokenizer, prompt, cache, prefix_length, answers[0]
            )
            longbench.rope_repair.reset_dynamic_cache(cache, prefix_length)
            elapsed = time.perf_counter() - started
        row = {
            "sample_id": sample_id,
            "variant": variant,
            "spec": spec,
            "answers": answers,
            "prediction": prediction,
            "generated_token_ids": generated_ids,
            "official_qa_f1": longbench.oracle.official_score(prediction, answers),
            "normalized_exact_match": int(longbench.oracle.normalized_exact_match(prediction, answers)),
            "prediction_contains_answer": int(longbench.oracle.contains_answer(prediction, answers)),
            "prompt_tokens": len(prompt_ids),
            "prompt_sha256": prompt_hash,
            "first_token_id": first_token_id,
            "first_token_nll": first_token_nll,
            "first_token_correct": first_token_correct,
            "gold_answer_mean_nll": gold_nll,
            "gold_answer_ppl": math.exp(min(gold_nll, 30.0)),
            "gold_answer_tokens": gold_tokens,
            "prefill_seconds": prefill_seconds,
            "query_seconds": query_seconds,
            "generation_seconds": generation_seconds,
            "nll_seconds": nll_seconds,
            "elapsed_seconds": elapsed,
        }
        append_jsonl(rows_path, [row])
        completed.add((variant, sample_id))
        print(
            f"[{case_index}/{len(cases)}] variant={variant} sample={sample_id[:10]} "
            f"f1={row['official_qa_f1']:.3f} nll={gold_nll:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del cache, prompt
        longbench.local_global.clear_allocator()

    all_rows = read_jsonl(rows_path)
    write_csv(args.output_dir / "rows.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "status.json", {"complete": True, "rows": len(all_rows)})
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
