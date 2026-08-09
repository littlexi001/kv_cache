from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    hierarchical_attention_mode,
)
from offloaded_prefill_cache_20260716 import OffloadedExactPrefillCache
from run_head_top2_targeted_ppl_20260714 import load_model
from run_hierarchical_physical_cache_ppl_20260715 import (
    empty_cuda_caches,
    run_synchronized_one_token,
    synchronize_cuda_devices,
)


DEFAULT_TASKS = "narrativeqa,hotpotqa,passage_retrieval_en,lcc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned FullKV versus hierarchical PCA per-head LongBench probe."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples_per_task", type=int, default=5)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=7500)
    parser.add_argument("--max_new_tokens_override", type=int, default=64)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--candidate_fraction", type=float, default=0.025)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--stream_group_size", type=int, default=1)
    parser.add_argument("--candidate_refresh_interval", type=int, default=1)
    parser.add_argument(
        "--host_append_mode", choices=("async", "sync"), default="async"
    )
    parser.add_argument(
        "--conversion_mode", choices=("async", "sync"), default="async"
    )
    parser.add_argument(
        "--hierarchical_prompt_mode",
        choices=("prefix_sparse_suffix", "full_prompt_then_compress"),
        default="prefix_sparse_suffix",
        help=(
            "prefix_sparse_suffix converts after the context prefix and replays the "
            "question with sparse attention. full_prompt_then_compress prefills the "
            "entire prompt densely, then compresses once before generation."
        ),
    )
    parser.add_argument(
        "--prefill_cache_mode",
        choices=("dynamic", "offloaded_exact"),
        default="dynamic",
        help=(
            "dynamic keeps the model-wide prefill KV on GPU until conversion. "
            "offloaded_exact preserves exact prefill attention while materializing "
            "only one full KV layer on GPU at a time."
        ),
    )
    parser.add_argument("--minimum_sparse_prefix_tokens", type=int, default=0)
    parser.add_argument(
        "--prompt_wrapper", choices=("llama3", "qwen3", "none"), default="llama3"
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def parse_tasks(spec: str) -> list[str]:
    tasks = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in lb.LONG_BENCH_PROMPTS]
    if unknown:
        raise ValueError(f"unsupported LongBench tasks: {unknown}")
    return tasks


def load_examples(args: argparse.Namespace) -> list[lb.Example]:
    examples: list[lb.Example] = []
    for task in parse_tasks(args.tasks):
        info = lb.LONG_BENCH_PROMPTS[task]
        path = args.longbench_data_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.max_samples_per_task > 0:
            rows = rows[: args.max_samples_per_task]
        for row_index, row in enumerate(rows):
            if row_index % args.num_shards != args.shard_index:
                continue
            max_new_tokens = int(info["max_new_tokens"])
            if args.max_new_tokens_override > 0:
                max_new_tokens = min(
                    max_new_tokens, args.max_new_tokens_override
                )
            examples.append(
                lb.Example(
                    benchmark="longbench",
                    task=task,
                    sample_id=str(row.get("_id", row_index)),
                    context=str(row["context"]),
                    query=str(row["input"]),
                    answers=[str(answer) for answer in row["answers"]],
                    prefix_template=str(info["prefix"]),
                    suffix_template=str(info["suffix"]),
                    metric=str(info["metric"]),
                    max_new_tokens=max_new_tokens,
                    length=int(row.get("length", 0) or 0),
                    all_classes=[
                        str(item) for item in (row.get("all_classes") or [])
                    ],
                    no_chat=bool(info.get("no_chat", False)),
                )
            )
    return examples


def build_bundle(
    tokenizer: Any, example: lb.Example, args: argparse.Namespace
) -> lb.PromptBundle:
    config = SimpleNamespace(
        max_context_tokens=args.max_context_tokens,
        page_tokens=128,
        force_no_chat_tasks="",
        prompt_wrapper=args.prompt_wrapper,
    )
    bundle, _, _, _, _ = lb.build_bundle(tokenizer, example, config)
    return bundle


@torch.inference_mode()
def generate_full(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: lb.PromptBundle,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
) -> dict[str, Any]:
    prefix_cache, prefill_seconds = lb.prefill_prefix(
        model, bundle, input_device, prefill_chunk_tokens
    )
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    prediction, generated_ids, query_seconds, decode_seconds = (
        lb.generate_with_cache(
            model,
            tokenizer,
            bundle,
            prefix_cache,
            max_new_tokens,
            input_device,
        )
    )
    return {
        "prediction": prediction,
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "conversion_seconds": 0.0,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "kv_ratio": 1.0,
        "cache_hit_rate": None,
    }


@torch.inference_mode()
def generate_hierarchical(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: lb.PromptBundle,
    max_new_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    previous_logits: torch.Tensor | None = None
    if args.hierarchical_prompt_mode == "full_prompt_then_compress":
        conversion_decode_reserve = max_new_tokens + 8
        if args.prefill_cache_mode == "offloaded_exact":
            source_cache: Any = OffloadedExactPrefillCache(
                capacity=int(bundle.input_ids.shape[-1]) + conversion_decode_reserve
            )
        else:
            source_cache = None
        source_cache, previous_logits, prefill_seconds = lb.run_token_segment(
            model,
            bundle.input_ids,
            source_cache,
            0,
            input_device,
            args.prefill_chunk_tokens,
        )
        source_length = int(bundle.input_ids.shape[-1])
        suffix_ids: list[int] = []
    else:
        if args.prefill_cache_mode != "dynamic":
            raise ValueError(
                "offloaded_exact prefill requires full_prompt_then_compress"
            )
        source_cache, prefill_seconds = lb.prefill_prefix(
            model, bundle, input_device, args.prefill_chunk_tokens
        )
        source_length = bundle.query_start
        suffix_ids = bundle.input_ids[0, bundle.query_start :].tolist()
        conversion_decode_reserve = bundle.suffix_token_count + max_new_tokens + 8
    if torch.cuda.is_available():
        synchronize_cuda_devices()
    conversion_started = time.perf_counter()
    conversion_kwargs = dict(
        projection_dim=args.projection_dim,
        index_bits=args.index_bits,
        candidate_fraction=args.candidate_fraction,
        attention_fraction=args.candidate_fraction,
        exact_cache_fraction=args.exact_cache_fraction,
        max_new_tokens=conversion_decode_reserve,
        candidate_selection_mode="per_head_stream",
        stream_group_size=args.stream_group_size,
        candidate_refresh_interval=args.candidate_refresh_interval,
        async_host_append=args.host_append_mode == "async",
        async_conversion=args.conversion_mode == "async",
        directory_backend="fused",
    )
    if isinstance(source_cache, OffloadedExactPrefillCache):
        cache = HierarchicalPCACache.from_offloaded_prefill_cache(
            source_cache, **conversion_kwargs
        )
    else:
        cache = HierarchicalPCACache.from_dynamic_cache(
            source_cache, **conversion_kwargs
        )
    del source_cache
    if torch.cuda.is_available():
        empty_cuda_caches()
        synchronize_cuda_devices()
    conversion_seconds = time.perf_counter() - conversion_started

    query_seconds = 0.0
    decode_seconds = 0.0
    generated_ids: list[int] = []
    eos_ids = (
        {int(tokenizer.eos_token_id)}
        if tokenizer.eos_token_id is not None
        else set()
    )
    with hierarchical_attention_mode(model):
        for offset, token_id in enumerate(suffix_ids):
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                int(token_id),
                cache,
                bundle.query_start + offset,
                input_device,
            )
            query_seconds += elapsed
        if previous_logits is None:
            raise RuntimeError("LongBench suffix produced no logits")
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(previous_logits.float(), dim=-1).item())
            if next_id in eos_ids:
                break
            generated_ids.append(next_id)
            if step + 1 == max_new_tokens:
                break
            cache, previous_logits, elapsed, _ = run_synchronized_one_token(
                model,
                next_id,
                cache,
                int(bundle.input_ids.shape[-1]) + step,
                input_device,
            )
            decode_seconds += elapsed

    persistent_bytes = cache.persistent_gpu_bytes()
    final_length = cache.get_seq_length()
    full_bytes_per_token = cache.original_gpu_bytes / max(1, source_length)
    return {
        "prediction": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "conversion_seconds": conversion_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "kv_ratio": persistent_bytes / (full_bytes_per_token * final_length),
        "cache_hit_rate": cache.mean_cache_hit_rate(),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        task_scores: list[float] = []
        for task in sorted({str(row["task"]) for row in subset}):
            task_subset = [row for row in subset if row["task"] == task]
            task_score = sum(float(row["score"]) for row in task_subset) / len(
                task_subset
            )
            task_scores.append(task_score)
            output.append(
                {
                    "task": task,
                    "method": method,
                    "samples": len(task_subset),
                    "score": task_score,
                    "mean_prompt_tokens": sum(
                        int(row["prompt_tokens"]) for row in task_subset
                    )
                    / len(task_subset),
                    "mean_kv_ratio": sum(
                        float(row["kv_ratio"]) for row in task_subset
                    )
                    / len(task_subset),
                    "mean_online_seconds": sum(
                        float(row["online_seconds"]) for row in task_subset
                    )
                    / len(task_subset),
                }
            )
        output.append(
            {
                "task": "ALL",
                "method": method,
                "samples": len(subset),
                "score": sum(task_scores) / len(task_scores),
                "mean_prompt_tokens": sum(
                    int(row["prompt_tokens"]) for row in subset
                )
                / len(subset),
                "mean_kv_ratio": sum(float(row["kv_ratio"]) for row in subset)
                / len(subset),
                "mean_online_seconds": sum(
                    float(row["online_seconds"]) for row in subset
                )
                / len(subset),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if not 0 < args.candidate_fraction < args.exact_cache_fraction < 1:
        raise ValueError("expected candidate_fraction < exact_cache_fraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    tokenizer, model, input_device = load_model(args)
    examples = load_examples(args)
    results_path = args.output_dir / "sample_results.csv"
    rows = read_csv(results_path)
    completed = {
        (str(row["task"]), str(row["sample_id"]), str(row["method"]))
        for row in rows
    }
    if rows:
        print(
            f"resuming shard {args.shard_index}/{args.num_shards}: "
            f"loaded {len(rows)} completed method rows",
            flush=True,
        )
    for index, example in enumerate(examples, start=1):
        bundle = build_bundle(tokenizer, example, args)
        print(
            f"[{index}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prefix={bundle.query_start} suffix={bundle.suffix_token_count}",
            flush=True,
        )
        methods = ["full_kv"]
        if bundle.query_start >= args.minimum_sparse_prefix_tokens:
            methods.append("hierarchical_pca_perhead")
        for method in methods:
            result_key = (example.task, example.sample_id, method)
            if result_key in completed:
                print(f"  {method}: already complete, skipping", flush=True)
                continue
            if method == "full_kv":
                result = generate_full(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                )
            else:
                result = generate_hierarchical(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args,
                )
            score = lb.score_prediction(
                example.metric,
                result["prediction"],
                example.answers,
                example.all_classes,
                task=example.task,
            )
            row = {
                "task": example.task,
                "sample_id": example.sample_id,
                "method": method,
                "metric": example.metric,
                "score": score,
                "prediction": result["prediction"].replace("\n", "\\n")[:500],
                "answers": json.dumps(example.answers, ensure_ascii=False),
                "prompt_tokens": int(bundle.input_ids.shape[-1]),
                "prefix_tokens": bundle.query_start,
                "suffix_tokens": bundle.suffix_token_count,
                "generated_tokens": len(result["generated_ids"]),
                "kv_ratio": result["kv_ratio"],
                "cache_hit_rate": result["cache_hit_rate"],
                "host_append_mode": args.host_append_mode,
                "conversion_mode": args.conversion_mode,
                "prefill_seconds": result["prefill_seconds"],
                "conversion_seconds": result["conversion_seconds"],
                "query_seconds": result["query_seconds"],
                "decode_seconds": result["decode_seconds"],
                "online_seconds": result["conversion_seconds"]
                + result["query_seconds"]
                + result["decode_seconds"],
                "total_seconds": result["prefill_seconds"]
                + result["conversion_seconds"]
                + result["query_seconds"]
                + result["decode_seconds"],
            }
            rows.append(row)
            append_csv_row(results_path, row)
            completed.add(result_key)
            print(
                f"  {method}: score={score:.4f} kv={float(result['kv_ratio']):.4f} "
                f"online={row['online_seconds']:.3f}s pred={result['prediction'][:80]!r}",
                flush=True,
            )
            if torch.cuda.is_available():
                empty_cuda_caches()

    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
