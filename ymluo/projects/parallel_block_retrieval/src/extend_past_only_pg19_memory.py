from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


TRAIN_LIST_URL = (
    "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/train_files.txt"
)
ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend a causal PG19 memory with disjoint real PG19 train books."
    )
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=100_000_000)
    parser.add_argument("--download_workers", type=int, default=16)
    parser.add_argument("--download_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--train_list_url", default=TRAIN_LIST_URL)
    parser.add_argument("--asset_root_url", default=ASSET_ROOT_URL)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_text(url: str, path: Path, retries: int = 3) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "parallel-block-retrieval/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
            return path
        except Exception as error:  # network retries need the original exception
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def download_batch(
    file_names: list[str], cache_dir: Path, asset_root_url: str, workers: int
) -> list[tuple[str, Path]]:
    completed: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_text,
                asset_root_url.rstrip("/") + "/" + file_name,
                cache_dir / file_name,
            ): file_name
            for file_name in file_names
        }
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                completed[file_name] = future.result()
            except Exception as error:
                print(f"download warning: {file_name}: {error}", flush=True)
    return [(name, completed[name]) for name in file_names if name in completed]


def copy_small_files(input_dir: Path, output_dir: Path) -> None:
    for name in (
        "queries.npy",
        "targets.npy",
        "source_blocks.npy",
        "metadata.jsonl",
    ):
        shutil.copy2(input_dir / name, output_dir / name)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if not input_summary.get("past_only"):
        raise ValueError("input memory must be past-only")
    if input_summary.get("source_blocks") != 0:
        raise ValueError("input memory must not contain predefined source blocks")
    block_tokens = int(input_summary["block_tokens"])
    if args.memory_tokens % block_tokens:
        raise ValueError("memory_tokens must be block aligned")

    input_blocks = np.load(input_dir / "base_blocks.npy", mmap_mode="r")
    input_scopes = np.load(input_dir / "base_block_scope_ids.npy", mmap_mode="r")
    input_centers = np.load(
        input_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    target_blocks = args.memory_tokens // block_tokens
    if target_blocks <= len(input_blocks):
        raise ValueError("target memory must be larger than input memory")

    base_output = np.lib.format.open_memmap(
        output_dir / "base_blocks.npy",
        mode="w+",
        dtype=np.int32,
        shape=(target_blocks, block_tokens),
    )
    scope_output = np.lib.format.open_memmap(
        output_dir / "base_block_scope_ids.npy",
        mode="w+",
        dtype=np.int32,
        shape=(target_blocks,),
    )
    center_output = np.lib.format.open_memmap(
        output_dir / "base_block_original_centers.npy",
        mode="w+",
        dtype=np.int64,
        shape=(target_blocks,),
    )
    base_output[: len(input_blocks)] = input_blocks
    scope_output[: len(input_blocks)] = input_scopes
    center_output[: len(input_blocks)] = input_centers
    cursor = len(input_blocks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    train_list_path = download_text(args.train_list_url, cache_dir / "train_files.txt")
    train_files = [
        line.strip()
        for line in train_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.Random(args.seed).shuffle(train_files)
    next_scope = max(int(np.max(input_scopes)), 0) + 1
    added_books: list[dict[str, Any]] = []
    attempted = 0
    started = time.perf_counter()

    while cursor < target_blocks and attempted < len(train_files):
        batch_names = train_files[attempted : attempted + args.download_batch]
        attempted += len(batch_names)
        downloaded = download_batch(
            batch_names, cache_dir, args.asset_root_url, args.download_workers
        )
        texts = [
            path.read_text(encoding="utf-8", errors="replace")
            for _, path in downloaded
        ]
        batch_token_ids = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        for (file_name, _), ids in zip(downloaded, batch_token_ids):
            if cursor >= target_blocks:
                break
            token_ids = np.asarray(ids, dtype=np.int32)
            available_blocks = len(token_ids) // block_tokens
            take_blocks = min(available_blocks, target_blocks - cursor)
            if take_blocks <= 0:
                continue
            take_tokens = take_blocks * block_tokens
            start = cursor
            end = cursor + take_blocks
            base_output[start:end] = token_ids[:take_tokens].reshape(
                take_blocks, block_tokens
            )
            scope_output[start:end] = next_scope
            center_output[start:end] = np.arange(
                block_tokens // 2,
                take_tokens,
                block_tokens,
                dtype=np.int64,
            )
            added_books.append(
                {
                    "scope_id": next_scope,
                    "split": "train",
                    "file": file_name,
                    "start_block": start,
                    "end_block": end,
                    "written_tokens": take_tokens,
                    "original_tokens": len(token_ids),
                    "truncated": take_blocks < available_blocks,
                }
            )
            cursor = end
            next_scope += 1
        written_tokens = cursor * block_tokens
        print(
            f"progress tokens={written_tokens:,}/{args.memory_tokens:,} "
            f"books={len(added_books)} attempted={attempted}",
            flush=True,
        )

    if cursor != target_blocks:
        raise RuntimeError(
            f"only constructed {cursor * block_tokens:,} tokens from PG19 train"
        )
    base_output.flush()
    scope_output.flush()
    center_output.flush()
    copy_small_files(input_dir, output_dir)
    with (output_dir / "added_train_books.jsonl").open("w", encoding="utf-8") as handle:
        for row in added_books:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    old_scopes = set(int(item) for item in np.unique(input_scopes) if int(item) >= 0)
    summary = {
        **input_summary,
        "source": "strict past-only PG19 test histories plus disjoint PG19 train distractors",
        "memory_tokens": args.memory_tokens,
        "max_base_tokens": args.memory_tokens,
        "memory_scales_tokens": [
            int(input_summary["memory_tokens"]),
            20_000_000,
            50_000_000,
            args.memory_tokens,
        ],
        "memory_scales_blocks": [
            int(input_summary["memory_tokens"]) // block_tokens,
            20_000_000 // block_tokens,
            50_000_000 // block_tokens,
            target_blocks,
        ],
        "max_base_blocks": target_blocks,
        "books_total": len(old_scopes) + len(added_books),
        "added_disjoint_train_books": len(added_books),
        "pg19_train_files_attempted": attempted,
        "input_memory_tokens": int(input_summary["memory_tokens"]),
        "contains_synthetic_text": False,
        "contains_repeated_distractor_text": False,
        "query_and_added_distractor_splits_disjoint": True,
        "selection_uses_target": False,
        "construction_seconds": time.perf_counter() - started,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
