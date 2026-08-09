from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import run_controlled_public_kv_benchmark_v1 as lb
import run_hierarchical_longbench_probe_20260715 as probe
from run_hierarchical_physical_cache_ppl_20260715 import empty_cuda_caches


DEFAULT_TASKS = (
    "niah_single_1,niah_single_2,niah_single_3,"
    "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
    "niah_multivalue,niah_multiquery,"
    "vt,cwe,fwe,qa_squad,qa_hotpot"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned FullKV versus hierarchical PCA per-head RULER probe."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lm_eval_path", required=True)
    parser.add_argument("--examples_jsonl", type=Path)
    parser.add_argument(
        "--ruler_hotpot_parquet",
        default=(
            "/home/fdong/ymluo/datasets/ruler_sources/hotpotqa/distractor/"
            "validation-00000-of-00001.parquet"
        ),
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--ruler_tasks", default=DEFAULT_TASKS)
    parser.add_argument("--ruler_lengths", default="4096,8192,16384,32768")
    parser.add_argument("--max_samples_per_task", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens_override", type=int, default=0)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--candidate_fraction", type=float, default=0.015)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--stream_group_size", type=int, choices=(1, 2, 4), default=2)
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
        default="full_prompt_then_compress",
    )
    parser.add_argument(
        "--prefill_cache_mode",
        choices=("dynamic", "offloaded_exact"),
        default="dynamic",
    )
    parser.add_argument("--minimum_sparse_prefix_tokens", type=int, default=0)
    parser.add_argument(
        "--prompt_wrapper", choices=("llama3", "qwen3", "none"), default="none"
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> list[lb.Example]:
    if args.max_samples_per_task <= 0:
        raise ValueError("max_samples_per_task must be positive for generated RULER data")
    if args.examples_jsonl is not None:
        if not args.examples_jsonl.is_file():
            raise FileNotFoundError(args.examples_jsonl)
        examples = []
        with args.examples_jsonl.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index % args.num_shards != args.shard_index:
                    continue
                examples.append(lb.Example(**json.loads(line)))
        return examples
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = SimpleNamespace(
        lm_eval_path=args.lm_eval_path,
        ruler_tasks=args.ruler_tasks,
        ruler_lengths=args.ruler_lengths,
        max_samples_per_task=args.max_samples_per_task,
        max_new_tokens_override=args.max_new_tokens_override,
        ruler_hotpot_parquet=args.ruler_hotpot_parquet,
    )
    examples = lb.load_ruler_examples(config, args.model_name_or_path)
    return [
        example
        for index, example in enumerate(examples)
        if index % args.num_shards == args.shard_index
    ]


def write_examples_jsonl(path: Path, examples: list[lb.Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = probe.summarize(rows)
    methods = sorted({str(row["method"]) for row in rows})
    lengths = sorted({int(row["requested_length"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for length in lengths:
            subset = [
                row
                for row in method_rows
                if int(row["requested_length"]) == length
            ]
            if not subset:
                continue
            task_scores = []
            for task in sorted({str(row["base_task"]) for row in subset}):
                task_rows = [row for row in subset if row["base_task"] == task]
                task_scores.append(
                    sum(float(row["score"]) for row in task_rows) / len(task_rows)
                )
            summary.append(
                {
                    "task": f"LENGTH_{length}",
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
    return summary


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

    examples = load_examples(args)
    tokenizer, model, input_device = probe.load_model(args)
    results_path = args.output_dir / "sample_results.csv"
    rows = probe.read_csv(results_path)
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
        bundle = probe.build_bundle(tokenizer, example, args)
        base_task, _, requested_length = example.task.rpartition("_")
        if not requested_length.isdigit():
            raise ValueError(f"unexpected RULER task name: {example.task}")
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
                result = probe.generate_full(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                )
            else:
                result = probe.generate_hierarchical(
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
            )
            row = {
                "task": example.task,
                "base_task": base_task,
                "requested_length": int(requested_length),
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
            probe.append_csv_row(results_path, row)
            completed.add(result_key)
            print(
                f"  {method}: score={score:.4f} kv={float(result['kv_ratio']):.4f} "
                f"online={row['online_seconds']:.3f}s pred={result['prediction'][:80]!r}",
                flush=True,
            )
            if torch.cuda.is_available():
                empty_cuda_caches()

    summary = summarize(rows)
    probe.write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
