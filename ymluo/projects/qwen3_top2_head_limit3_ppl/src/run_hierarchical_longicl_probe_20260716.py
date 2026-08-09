from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import run_controlled_public_kv_benchmark_v1 as lb
import run_hierarchical_longbench_probe_20260715 as physical
from run_head_top2_targeted_ppl_20260714 import load_model


LONG_ICL_DOMAIN = "Long In-context Learning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired FullKV/physical hierarchical KV LongBench-v2 Long ICL probe."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_v2_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=32000)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--index_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--candidate_fraction", type=float, default=0.015)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--stream_group_size", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--candidate_refresh_interval", type=int, default=1)
    parser.add_argument("--host_append_mode", choices=("async", "sync"), default="async")
    parser.add_argument("--conversion_mode", choices=("async", "sync"), default="async")
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
    parser.add_argument("--prompt_wrapper", choices=("llama3", "qwen3", "none"), default="llama3")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> list[lb.Example]:
    payload = json.loads(args.longbench_v2_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LongBench-v2 split must contain a JSON list")
    rows = [
        row
        for row in payload
        if isinstance(row, dict) and str(row.get("domain", "")) == LONG_ICL_DOMAIN
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    examples: list[lb.Example] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        query = (
            f"What is the correct answer to this question: {question}\n"
            "Choices:\n"
            f"(A) {str(row.get('choice_A', '')).strip()}\n"
            f"(B) {str(row.get('choice_B', '')).strip()}\n"
            f"(C) {str(row.get('choice_C', '')).strip()}\n"
            f"(D) {str(row.get('choice_D', '')).strip()}"
        )
        examples.append(
            lb.Example(
                benchmark="longbench_v2",
                task="lbv2_long_in_context_learning",
                sample_id=str(row.get("_id", len(examples))),
                context=str(row.get("context", "")),
                query=query,
                answers=[str(row.get("answer", "")).strip().upper()],
                prefix_template=(
                    "Please read the following text and answer the question below.\n\n"
                    "<text>\n"
                ),
                suffix_template=(
                    "\n</text>\n\n{input}\n\n"
                    'Format your response as follows: "The correct answer is (insert answer here)".'
                ),
                metric="longbench_v2_mc",
                max_new_tokens=args.max_new_tokens,
                length=0,
                all_classes=[],
                domain=LONG_ICL_DOMAIN,
                sub_domain=str(row.get("sub_domain", "")),
                difficulty=str(row.get("difficulty", "")),
                length_category=str(row.get("length", "")),
            )
        )
    return [
        example
        for index, example in enumerate(examples)
        if index % args.num_shards == args.shard_index
    ]


def build_bundle(tokenizer: Any, example: lb.Example, args: argparse.Namespace) -> lb.PromptBundle:
    config = SimpleNamespace(
        max_context_tokens=args.max_context_tokens,
        page_tokens=128,
        force_no_chat_tasks="",
        prompt_wrapper=args.prompt_wrapper,
    )
    bundle, _, _, _, _ = lb.build_bundle(tokenizer, example, config)
    return bundle


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if not 0 < args.candidate_fraction < args.exact_cache_fraction < 1:
        raise ValueError("expected candidate_fraction < exact_cache_fraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    tokenizer, model, input_device = load_model(args)
    examples = load_examples(args)
    result_path = args.output_dir / "sample_results.csv"
    rows = read_rows(result_path)
    completed = {
        (row["sample_id"], row["method"])
        for row in rows
    }
    for index, example in enumerate(examples, start=1):
        bundle = build_bundle(tokenizer, example, args)
        print(
            f"[{index}/{len(examples)}] {example.sample_id} "
            f"prompt={bundle.input_ids.shape[-1]}",
            flush=True,
        )
        for method in ("full_kv", "hierarchical_pca_perhead"):
            key = (example.sample_id, method)
            if key in completed:
                continue
            if method == "full_kv":
                result = physical.generate_full(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                )
            else:
                result = physical.generate_hierarchical(
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
                "benchmark": example.benchmark,
                "task": example.task,
                "sample_id": example.sample_id,
                "domain": example.domain,
                "sub_domain": example.sub_domain,
                "difficulty": example.difficulty,
                "length_category": example.length_category,
                "method": method,
                "metric": example.metric,
                "score": score,
                "prediction": result["prediction"].replace("\n", "\\n")[:500],
                "answers": json.dumps(example.answers),
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
            append_row(result_path, row)
            rows.append(row)
            completed.add(key)
            print(
                f"  {method}: score={score:.3f} kv={result['kv_ratio']:.4f} "
                f"total={row['total_seconds']:.3f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                physical.empty_cuda_caches()

    summary = physical.summarize(rows)
    physical.write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
