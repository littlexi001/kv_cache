from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
import run_hierarchical_ruler_probe_20260716 as ruler_data
import run_sample_calibrated_longbench_20260717 as probe
from run_head_top2_targeted_ppl_20260714 import (
    install_llama_head_top_fraction_patch,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aligned FullKV, global partition, and query-gated temporal "
            "partition retrieval on frozen RULER examples."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--examples_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--methods", default="full_kv,global_partition,qgate_partition"
    )
    parser.add_argument("--ruler_tasks", default="niah_single_1")
    parser.add_argument("--ruler_lengths", default="65536,131072")
    parser.add_argument("--max_samples_per_task", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=0)
    parser.add_argument("--max_prompt_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens_override", type=int, default=32)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt_wrapper", choices=("llama3", "qwen3", "none"), default="none"
    )
    parser.add_argument("--minimum_sparse_prefix_tokens", type=int, default=0)
    parser.add_argument("--collect_attention_stats", action="store_true")
    parser.add_argument("--mass_threshold", type=float, default=0.75)
    parser.add_argument(
        "--budget_fractions",
        default=",".join(str(value) for value in probe.FROZEN_BUDGET_FRACTIONS),
    )
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--partition_ucb_z", type=float, default=0.0)
    parser.add_argument("--partition_overfetch_factor", type=int, default=2)
    parser.add_argument("--value_mass_threshold", type=float, default=1.0)
    parser.add_argument(
        "--qk_metric_query_shrinkage",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--sampled_quantile_sample_count",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--sampled_quantile_target_tail_count",
        type=int,
        default=0,
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    # Compatibility fields required by the shared frozen-data loader.
    parser.add_argument("--lm_eval_path", default="")
    parser.add_argument("--ruler_hotpot_parquet", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> list[lb.Example]:
    requested_tasks = set(probe.parse_csv_values(args.ruler_tasks))
    requested_lengths = {
        int(value) for value in probe.parse_csv_values(args.ruler_lengths)
    }
    # Apply the per-task sample limit before sharding. Sharding first would turn
    # an m1 request into up to one sample per shard for every task/length pair.
    loader_args = argparse.Namespace(**vars(args))
    loader_args.num_shards = 1
    loader_args.shard_index = 0
    loaded = ruler_data.load_examples(loader_args)
    counts: dict[tuple[str, int], int] = {}
    examples: list[lb.Example] = []
    for example in loaded:
        base_task, _, length_text = example.task.rpartition("_")
        if not length_text.isdigit():
            raise ValueError(f"unexpected RULER task name: {example.task}")
        length = int(length_text)
        if base_task not in requested_tasks or length not in requested_lengths:
            continue
        key = (base_task, length)
        if counts.get(key, 0) >= args.max_samples_per_task:
            continue
        if args.max_new_tokens_override > 0:
            example.max_new_tokens = min(
                example.max_new_tokens, args.max_new_tokens_override
            )
        counts[key] = counts.get(key, 0) + 1
        examples.append(example)
    if not examples:
        raise ValueError("no frozen RULER examples matched the requested filters")
    return [
        example
        for index, example in enumerate(examples)
        if index % args.num_shards == args.shard_index
    ]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(row)
    if not needs_header:
        with path.open("r", encoding="utf-8", newline="") as existing:
            existing_header = csv.DictReader(existing).fieldnames
        if not existing_header:
            raise RuntimeError(f"existing CSV has no header: {path}")
        fieldnames = existing_header
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def method_audit_fields(
    method: str,
    method_config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "executed_path": method,
        "configured_sampled_quantile_sample_count": method_config.get(
            "sampled_quantile_sample_count", 0
        ),
        "configured_index_bits_per_token": (
            probe.configured_index_bits_per_token(method_config["score_mode"])
        ),
        "index_build_seconds": result.get("index_build_seconds", 0.0),
        "qk_prebuild_seconds": result.get("qk_prebuild_seconds", 0.0),
        "qk_prebuild_layers": result.get("qk_prebuild_layers", 0),
        "qk_batched_allocation_layers": result.get(
            "qk_batched_allocation_layers", 0
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = probe.summarize(rows)
    methods = sorted({str(row["method"]) for row in rows})
    lengths = sorted({int(row["requested_length"]) for row in rows})
    for method in methods:
        for length in lengths:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and int(row["requested_length"]) == length
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
                    "mean_prompt_tokens": probe.mean(subset, "prompt_tokens"),
                    "mean_generated_tokens": probe.mean(subset, "generated_tokens"),
                    "mean_configured_attention_fraction": probe.mean(
                        subset, "configured_attention_fraction"
                    ),
                    "mean_configured_candidate_fraction": probe.mean(
                        subset, "configured_candidate_fraction"
                    ),
                    "mean_attention_link_ratio": probe.mean(
                        subset, "attention_link_ratio"
                    ),
                    "mean_exact_qk_ratio": probe.mean(subset, "exact_qk_ratio"),
                    "mean_temporal_reuse_rate": probe.mean(
                        subset, "temporal_reuse_rate"
                    ),
                    "mean_gpu_kv_storage_ratio": probe.mean(
                        subset, "gpu_kv_storage_ratio"
                    ),
                    "mean_scan_dimension_fraction": probe.mean(
                        subset, "scan_dimension_fraction"
                    ),
                    "mean_online_seconds": probe.mean(subset, "online_seconds"),
                    "paired_online_speedup": "",
                }
            )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if not 0.0 <= args.qk_metric_query_shrinkage <= 1.0:
        raise ValueError("qk_metric_query_shrinkage must be in [0, 1]")
    methods = probe.parse_methods(args.methods)
    budgets = probe.parse_budget_fractions(args.budget_fractions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    examples = load_examples(args)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    preload_extensions = (
        os.environ.get("QKSIEVE_PRELOAD_EXTENSIONS", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    if preload_extensions and probe.QKSIEVE_FROZEN_C64_METHOD in methods:
        probe.preload_qksieve_runtime_extensions()
        if (
            os.environ.get("QKSIEVE_PRELOAD_QMSE_RATE_TABLES", "1")
            .strip()
            .lower()
            not in {"0", "false", "no"}
        ):
            probe.preload_qksieve_qmse_rate_tables(model)
        probe.precompute_qksieve_value_metric_grams(model)
    results_path = args.output_dir / "sample_results.csv"
    rows = read_csv(results_path)
    completed = {
        (str(row["task"]), str(row["sample_id"]), str(row["method"]))
        for row in rows
    }

    for index, example in enumerate(examples, start=1):
        bundle = probe.build_bundle(tokenizer, example, args)
        base_task, _, length_text = example.task.rpartition("_")
        requested_length = int(length_text)
        eos_token_ids = probe.longbench_stop_token_ids(
            tokenizer, base_task
        )
        print(
            f"[{index}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prefix={bundle.query_start} suffix={bundle.suffix_token_count}",
            flush=True,
        )
        active_methods = list(methods)
        if bundle.query_start < args.minimum_sparse_prefix_tokens:
            active_methods = [
                method
                for method in active_methods
                if method
                not in {
                    "global_partition",
                    "qgate_partition",
                    "ec_bandef",
                    "ec_bandef_budget",
                    "oneshot_bandef_budget",
                    "countcap",
                }
            ]
        for method in active_methods:
            result_key = (example.task, example.sample_id, method)
            if result_key in completed:
                print(f"  {method}: already complete", flush=True)
                continue
            if method == "full_kv":
                method_config = {
                    "budget_fractions": (1.0,),
                    "candidate_fraction": 1.0,
                    "projection_dim": 0,
                    "score_mode": "full_kv",
                    "attention_tokens": bundle.query_start,
                }
                result = probe.generate_full(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                    eos_token_ids,
                )
            else:
                method_config = probe.sparse_method_config(
                    method,
                    bundle.query_start,
                    budgets,
                    args,
                )
                result = probe.generate_global_partition(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                    method_config["budget_fractions"],
                    args,
                    method_config["score_mode"],
                    candidate_fraction=method_config["candidate_fraction"],
                    projection_dim=method_config["projection_dim"],
                    dense_suffix=probe.uses_dense_prompt_suffix(method),
                    eos_token_ids=eos_token_ids,
                    sampled_quantile_sample_count=method_config.get(
                        "sampled_quantile_sample_count"
                    ),
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
                "requested_length": requested_length,
                "sample_id": example.sample_id,
                "method": method,
                **method_audit_fields(method, method_config, result),
                "metric": example.metric,
                "score": score,
                "prediction": result["prediction"].replace("\n", "\\n"),
                "answers": json.dumps(example.answers, ensure_ascii=False),
                "prompt_tokens": int(bundle.input_ids.shape[-1]),
                "prefix_tokens": bundle.query_start,
                "suffix_tokens": bundle.suffix_token_count,
                "generated_tokens": len(result["generated_ids"]),
                "configured_attention_fraction": method_config["budget_fractions"][-1],
                "configured_attention_tokens": method_config["attention_tokens"],
                "configured_candidate_fraction": method_config["candidate_fraction"],
                "configured_projection_dim": method_config["projection_dim"],
                "configured_score_mode": method_config["score_mode"],
                "attention_link_ratio": result["attention_link_ratio"],
                "exact_qk_ratio": result["exact_qk_ratio"],
                "estimated_retained_mass": result["estimated_retained_mass"],
                "temporal_reuse_rate": result["temporal_reuse_rate"],
                "gpu_kv_storage_ratio": result["gpu_kv_storage_ratio"],
                "scan_dimension_fraction": result["scan_dimension_fraction"],
                "diagnostics_enabled": args.collect_attention_stats,
                "prefill_seconds": result["prefill_seconds"],
                "query_seconds": result["query_seconds"],
                "decode_seconds": result["decode_seconds"],
                "online_seconds": result["query_seconds"] + result["decode_seconds"],
                "total_seconds": result["prefill_seconds"]
                + result["query_seconds"]
                + result["decode_seconds"],
            }
            rows.append(row)
            append_csv_row(results_path, row)
            completed.add(result_key)
            print(
                f"  {method}: score={score:.4f} "
                f"links={float(result['attention_link_ratio']):.4f} "
                f"exact_qk={float(result['exact_qk_ratio']):.4f} "
                f"online={row['online_seconds']:.3f}s "
                f"pred={result['prediction'][:80]!r}",
                flush=True,
            )
            probe.empty_cuda_caches()

    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
