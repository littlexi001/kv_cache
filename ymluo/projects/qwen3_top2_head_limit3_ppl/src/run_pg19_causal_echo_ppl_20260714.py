from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from run_causal_echo_ppl_20260714 import (  # noqa: E402
    evaluate_causal_echo_ppl,
    evaluate_universal_controller_ppl,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    evaluate_target_ppl,
    make_bundle,
    pick_input_device,
    resolve_dtype,
    selector_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal KV retrieval PPL on individual PG19 books.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--pg19_parquet", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--book_indices", default="0,1,2,3,4,5")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--windows_per_book", type=int, default=1)
    parser.add_argument("--book_offset_tokens", type=int, default=2048)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--budget_tokens", type=int, default=2048)
    parser.add_argument("--page_tokens", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=32)
    parser.add_argument("--recent_tokens", type=int, default=1536)
    parser.add_argument("--echo_match_tokens", type=int, default=8)
    parser.add_argument("--echo_confirmation_tokens", type=int, default=8)
    parser.add_argument("--echo_stability_matches", type=int, default=3)
    parser.add_argument("--echo_refresh_tokens", type=int, default=1)
    parser.add_argument("--replay_chunk_tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def parse_indices(spec: str) -> list[int]:
    indices = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not indices or min(indices) < 0:
        raise ValueError("book_indices must contain non-negative integers")
    return indices


def tokenize_book_prefix(tokenizer: Any, text: str, required_tokens: int) -> list[int]:
    # PG19 books are long; tokenize a growing character prefix instead of every full book.
    chars = min(len(text), max(200_000, required_tokens * 6))
    while True:
        ids = tokenizer(text[:chars], add_special_tokens=False)["input_ids"]
        if len(ids) >= required_tokens or chars >= len(text):
            return ids
        chars = min(len(text), chars * 2)


def load_pg19_rows(path: Path, indices: list[int]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["short_book_title", "publication_date", "url", "text"])
    if max(indices) >= table.num_rows:
        raise IndexError(f"PG19 has {table.num_rows} rows, requested {max(indices)}")
    rows = table.take(indices).to_pylist()
    return [{"book_index": index, **row} for index, row in zip(indices, rows)]


def append_result(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    method: str,
    nll: float,
    ppl: float,
    seconds: float,
    count: int,
    kv_ratio: float,
    selector_seconds: float,
    prefill_seconds: float,
    timing: dict[str, float] | None = None,
    matches: list[dict[str, int]] | None = None,
) -> None:
    rows.append(
        {
            **metadata,
            "method": method,
            "nll": nll,
            "ppl": ppl,
            "tokens": count,
            "kv_ratio": kv_ratio,
            "seconds": seconds,
            "selector_seconds": selector_seconds,
            "prefill_seconds": prefill_seconds,
            "timing": timing or {},
            "echo_matches": matches or [],
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = selector_config(args)
    lb.install_llama_layerwise_attention_mask_patch()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    book_indices = parse_indices(args.book_indices)
    books = load_pg19_rows(args.pg19_parquet, book_indices)
    required_tokens = (
        args.book_offset_tokens
        + (args.windows_per_book - 1) * args.window_stride_tokens
        + args.history_tokens
        + args.eval_tokens
    )
    rows: list[dict[str, Any]] = []
    for book in books:
        stream = tokenize_book_prefix(tokenizer, book["text"], required_tokens)
        if len(stream) < required_tokens:
            print(
                f"skip book {book['book_index']}: only {len(stream)} tokens, need {required_tokens}",
                flush=True,
            )
            continue
        for window in range(args.windows_per_book):
            start = args.book_offset_tokens + window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            target_ids = stream[start + args.history_tokens : start + args.history_tokens + args.eval_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            query_text = tokenizer.decode(
                query_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            bundle, pages = make_bundle(tokenizer, remote_ids, args.page_tokens)
            metadata = {
                "corpus": "pg19",
                "book_index": book["book_index"],
                "book_title": book["short_book_title"],
                "publication_date": book["publication_date"],
                "window": window,
                "window_start": start,
            }
            example = lb.Example(
                benchmark="pg19_ppl",
                task="book_continuation",
                sample_id=f"pg19_{book['book_index']}_{window}",
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
            full_cache, prefill_seconds = lb.prefill_prefix(
                model, bundle, input_device, config.prefill_chunk_tokens
            )
            selector_started = time.perf_counter()
            base_keep = lb.keep_ours_page(
                bundle, example, pages, config, {"model": model, "tokenizer": tokenizer}
            )
            selector_seconds = time.perf_counter() - selector_started
            recent_config = replace(
                config,
                sink_tokens=args.sink_tokens,
                recent_tokens=max(0, args.budget_tokens - args.sink_tokens),
            )
            recent_keep = lb.keep_streaming(bundle, example, pages, recent_config, None)
            for method, keep_indices in (("hybrid_static", base_keep), ("sink_recent", recent_keep)):
                cache = lb.gather_past_key_values(full_cache, keep_indices)
                nll, ppl, seconds, count = evaluate_target_ppl(
                    model,
                    cache,
                    query_ids,
                    target_ids,
                    len(remote_ids),
                    input_device,
                    args.replay_chunk_tokens,
                    True,
                )
                append_result(
                    rows,
                    metadata,
                    method,
                    nll,
                    ppl,
                    seconds,
                    count,
                    (len(keep_indices) + len(query_ids)) / len(history),
                    selector_seconds,
                    prefill_seconds,
                )
                del cache

            for method, enable_echo in (("tokenwise_static", False), ("causal_echo", True)):
                timing: dict[str, float] = {}
                nll, ppl, seconds, count, matches = evaluate_causal_echo_ppl(
                    model,
                    full_cache,
                    bundle,
                    config,
                    base_keep,
                    query_ids,
                    target_ids,
                    input_device,
                    args.echo_match_tokens,
                    args.echo_refresh_tokens,
                    args.replay_chunk_tokens,
                    stability_matches=args.echo_stability_matches,
                    confirmation_tokens=args.echo_confirmation_tokens,
                    enable_echo=enable_echo,
                    timing_stats=timing,
                )
                append_result(
                    rows,
                    metadata,
                    method,
                    nll,
                    ppl,
                    seconds,
                    count,
                    (args.budget_tokens + len(query_ids)) / len(history),
                    selector_seconds,
                    prefill_seconds,
                    timing,
                    matches,
                )

            controller_stats: dict[str, float] = {}
            nll, ppl, seconds, count, traces = evaluate_universal_controller_ppl(
                model,
                full_cache,
                bundle,
                config,
                base_keep,
                query_ids,
                target_ids,
                input_device,
                args.echo_match_tokens,
                args.echo_stability_matches,
                args.echo_confirmation_tokens,
                args.replay_chunk_tokens,
                timing_stats=controller_stats,
            )
            append_result(
                rows,
                metadata,
                "universal_controller",
                nll,
                ppl,
                seconds,
                count,
                (controller_stats["mean_active_budget_tokens"] + len(query_ids)) / len(history),
                selector_seconds,
                prefill_seconds,
                controller_stats,
                traces,
            )

            nll, ppl, seconds, count = evaluate_target_ppl(
                model,
                full_cache,
                query_ids,
                target_ids,
                len(remote_ids),
                input_device,
                args.replay_chunk_tokens,
                False,
            )
            append_result(
                rows,
                metadata,
                "full_kv",
                nll,
                ppl,
                seconds,
                count,
                1.0,
                selector_seconds,
                prefill_seconds,
            )
            del full_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            args.output_dir.joinpath("results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            current = [row for row in rows if row["book_index"] == book["book_index"] and row["window"] == window]
            print(
                f"book={book['book_index']} window={window}: "
                + " ".join(f"{row['method']}={row['ppl']:.3f}" for row in current),
                flush=True,
            )


if __name__ == "__main__":
    main()
