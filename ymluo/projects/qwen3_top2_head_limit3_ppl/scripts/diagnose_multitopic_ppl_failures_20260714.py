from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    AutoModelForCausalLM,
    AutoTokenizer,
    encode_topic_stream,
    make_bundle,
    resolve_dtype,
    selector_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose multi-topic causal PPL failure windows.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--windows_per_topic", type=int, default=3)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--budget_tokens", type=int, default=2048)
    parser.add_argument("--page_tokens", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=32)
    parser.add_argument("--recent_tokens", type=int, default=64)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return runs


def repeated_target_coverage(
    remote_ids: list[int],
    target_ids: list[int],
    keep_indices: list[int],
    ngram: int,
) -> dict[str, Any]:
    locations: dict[tuple[int, ...], list[int]] = {}
    for start in range(0, len(remote_ids) - ngram + 1):
        locations.setdefault(tuple(remote_ids[start : start + ngram]), []).append(start)
    keep = set(keep_indices)
    matched: set[int] = set()
    retained: set[int] = set()
    examples: list[dict[str, int]] = []
    for target_start in range(0, len(target_ids) - ngram + 1):
        remote_starts = locations.get(tuple(target_ids[target_start : target_start + ngram]), [])
        if not remote_starts:
            continue
        matched.update(range(target_start, target_start + ngram))
        retained_starts = [
            remote_start
            for remote_start in remote_starts
            if all(remote_start + offset in keep for offset in range(ngram))
        ]
        if retained_starts:
            retained.update(range(target_start, target_start + ngram))
        if len(examples) < 8:
            examples.append(
                {
                    "target_start": target_start,
                    "first_remote_start": remote_starts[0],
                    "first_distance_to_query": len(remote_ids) - remote_starts[0],
                    "retained_occurrences": len(retained_starts),
                }
            )
    return {
        "ngram": ngram,
        "matched_target_fraction": len(matched) / max(1, len(target_ids)),
        "retained_target_fraction": len(retained) / max(1, len(target_ids)),
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_json.parent
    config = selector_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()

    topic_list = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    required_tokens = (
        (args.windows_per_topic - 1) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    )
    records: list[dict[str, Any]] = []
    for topic in topic_list:
        stream = encode_topic_stream(
            tokenizer,
            TOPICS[topic],
            required_tokens,
            args.dataset_cache_dir,
            args.seed,
        )
        for window in range(args.windows_per_topic):
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            query_text = tokenizer.decode(query_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            target_text = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            bundle, pages = make_bundle(tokenizer, remote_ids, args.page_tokens)
            example = lb.Example(
                benchmark="multitopic_ppl",
                task=topic,
                sample_id=f"{topic}_{window}",
                context="",
                query=query_text,
                answers=[],
                prefix_template="",
                suffix_template="",
                metric="qa_f1",
                max_new_tokens=0,
                length=len(history),
                all_classes=[],
            )
            extra = {"model": model, "tokenizer": tokenizer}
            keep_indices = lb.keep_ours_page(bundle, example, pages, config, extra)
            context_keep = sorted(index for index in keep_indices if 0 <= index < len(remote_ids))
            selected_ids = set(lb.selected_page_ids(bundle, context_keep))
            runs = contiguous_runs(context_keep)
            recency = {
                str(tokens): sum(index >= len(remote_ids) - tokens for index in context_keep)
                for tokens in (64, 128, 256, 512, 1024, 2048, 4096)
            }
            selected_pages = sorted(
                (page for page in pages if page.page_id in selected_ids),
                key=lambda page: (page.score, page.page_id),
                reverse=True,
            )
            records.append(
                {
                    "topic": topic,
                    "window": window,
                    "remote_tokens": len(remote_ids),
                    "kept_tokens": len(context_keep),
                    "selected_pages": len(selected_ids),
                    "contiguous_runs": len(runs),
                    "median_run_tokens": sorted(end - begin for begin, end in runs)[len(runs) // 2],
                    "max_run_tokens": max(end - begin for begin, end in runs),
                    "recency_tokens_retained": recency,
                    "repeat32": repeated_target_coverage(remote_ids, target_ids, context_keep, 32),
                    "query_text": query_text,
                    "target_text": target_text,
                    "top_selected_pages": [
                        {
                            "page_id": page.page_id,
                            "score": page.score,
                            "distance_to_query": len(remote_ids) - page.token_start,
                            "text": page.text,
                        }
                        for page in selected_pages[:20]
                    ],
                }
            )
            print(
                f"{topic}/{window}: runs={len(runs)} recent2k={recency['2048']} "
                f"repeat32={records[-1]['repeat32']['matched_target_fraction']:.3f} "
                f"retained={records[-1]['repeat32']['retained_target_fraction']:.3f}",
                flush=True,
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
