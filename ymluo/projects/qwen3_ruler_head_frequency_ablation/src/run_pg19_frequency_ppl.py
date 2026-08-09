from __future__ import annotations

import argparse
import csv
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
        description="Paired PG19 held-out perplexity for RoPE frequency-scaling specs."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--pg19-parquet", required=True, type=Path)
    parser.add_argument("--specs-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--books-per-length", type=int, default=4)
    parser.add_argument("--token-offset", type=int, default=512)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def read_specs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value["specs"] if isinstance(value, dict) else value
    if not isinstance(specs, list) or not specs:
        raise ValueError("specs-json must contain a non-empty specs list")
    return specs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def load_pg19_cases(
    parquet_path: Path,
    tokenizer: Any,
    lengths: Sequence[int],
    books_per_length: int,
    token_offset: int,
) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    selected: dict[int, list[dict[str, Any]]] = {int(length): [] for length in lengths}
    maximum = max(lengths) + token_offset + 1
    for book_index, row in frame.iterrows():
        token_ids = tokenizer(
            str(row["text"]), add_special_tokens=False, truncation=False
        )["input_ids"]
        if len(token_ids) <= token_offset + min(lengths):
            continue
        for length in lengths:
            if len(selected[length]) >= books_per_length:
                continue
            end = token_offset + length + 1
            if len(token_ids) >= end:
                selected[length].append(
                    {
                        "case_id": f"pg19_book{book_index}_offset{token_offset}_len{length}",
                        "book_index": int(book_index),
                        "book_title": str(row["short_book_title"]),
                        "length": int(length),
                        "token_ids": [int(value) for value in token_ids[token_offset:end]],
                    }
                )
        if all(len(selected[length]) >= books_per_length for length in lengths):
            break
        if len(token_ids) > maximum:
            del token_ids[maximum:]
    missing = {length: books_per_length - len(rows) for length, rows in selected.items() if len(rows) < books_per_length}
    if missing:
        raise RuntimeError(f"not enough PG19 books for requested lengths: {missing}")
    return [row for length in lengths for row in selected[length]]


@torch.inference_mode()
def score_tokens(
    model: Any,
    token_ids: Sequence[int],
    chunk_size: int,
) -> tuple[float, int, float]:
    from transformers import DynamicCache

    device = longbench.cache_runner.input_device(model)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    cache: Any = DynamicCache()
    loss_sum = 0.0
    scored = 0
    started = time.perf_counter()
    for start in range(0, len(token_ids) - 1, chunk_size):
        end = min(start + chunk_size, len(token_ids) - 1)
        inputs = tokens[start:end].view(1, -1).to(device)
        cache_position = torch.arange(start, end, dtype=torch.long, device=device)
        output = model(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = output.logits[0].float()
        targets = tokens[start + 1 : end + 1].to(logits.device)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        scored += int(targets.numel())
        cache = output.past_key_values
        del inputs, output, logits, targets
    longbench.cache_runner.synchronize()
    return loss_sum, scored, time.perf_counter() - started


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(str(row["variant"]), int(row["context_length"])) for row in rows})
    for variant, length in groups:
        selected = [
            row for row in rows
            if row["variant"] == variant and int(row["context_length"]) == length
        ]
        token_count = sum(int(row["scored_tokens"]) for row in selected)
        nll = sum(float(row["loss_sum"]) for row in selected) / token_count
        output.append(
            {
                "variant": variant,
                "context_length": length,
                "book_count": len(selected),
                "scored_tokens": token_count,
                "mean_nll": nll,
                "perplexity": math.exp(min(nll, 30.0)),
                "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in selected),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")
    lengths = sorted({int(value) for value in args.lengths.split(",") if value.strip()})
    if not lengths or min(lengths) < 2:
        raise ValueError("lengths must contain positive integers >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = read_specs(args.specs_json)
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
    cases = load_pg19_cases(
        args.pg19_parquet,
        tokenizer,
        lengths,
        args.books_per_length,
        args.token_offset,
    )
    pairs = [(spec, case) for spec in specs for case in cases]
    pairs = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    if not pairs:
        raise RuntimeError("this shard has no cases")
    write_json(
        args.output_dir / "config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "specs": specs,
            "selected_cases": [{key: value for key, value in case.items() if key != "token_ids"} for case in cases],
            "pair_count": len(pairs),
            "weights_frozen": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )
    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {(str(row["variant"]), str(row["case_id"])) for row in existing}
    for index, (spec, case) in enumerate(pairs, start=1):
        variant = str(spec["name"])
        case_id = str(case["case_id"])
        if (variant, case_id) in completed:
            continue
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        with intervention.activate(spec):
            loss_sum, scored_tokens, elapsed = score_tokens(
                model, case["token_ids"], args.score_chunk_size
            )
        mean_nll = loss_sum / scored_tokens
        row = {
            "variant": variant,
            "spec": spec,
            "case_id": case_id,
            "book_index": case["book_index"],
            "book_title": case["book_title"],
            "context_length": case["length"],
            "scored_tokens": scored_tokens,
            "loss_sum": loss_sum,
            "mean_nll": mean_nll,
            "perplexity": math.exp(min(mean_nll, 30.0)),
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        append_jsonl(rows_path, [row])
        completed.add((variant, case_id))
        print(
            f"[{index}/{len(pairs)}] variant={variant} case={case_id} "
            f"ppl={row['perplexity']:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )
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
