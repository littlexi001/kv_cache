from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_selected_head_debiased_retrieval import read_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack the layer/KV-head channels required by a train-only head "
            "selection into contiguous arrays without changing values."
        )
    )
    parser.add_argument("--source_profile_dir", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--block_chunk", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def pack_file(task: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(task["source_path"])
    output_path = Path(task["output_path"])
    kv_heads = [int(item) for item in task["kv_heads"]]
    block_chunk = int(task["block_chunk"])
    started = time.perf_counter()
    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 4:
        raise ValueError(f"unexpected source shape: {source.shape}")
    shape = (source.shape[0], source.shape[1], len(kv_heads), source.shape[3])
    target = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=source.dtype, shape=shape
    )
    digest = hashlib.sha256()
    for offset in range(0, source.shape[0], block_chunk):
        end = min(source.shape[0], offset + block_chunk)
        chunk = np.take(
            source[offset:end], np.asarray(kv_heads, dtype=np.int64), axis=2
        )
        contiguous = np.ascontiguousarray(chunk)
        target[offset:end] = contiguous
        digest.update(contiguous.view(np.uint8))
    target.flush()
    del target
    with output_path.open("rb") as handle:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(
                handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
            )
    return {
        "layer": int(task["layer"]),
        "rank": int(task["rank"]),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "kv_heads": kv_heads,
        "shape": list(shape),
        "dtype": str(source.dtype),
        "data_sha256": digest.hexdigest(),
        "bytes": int(np.prod(shape) * source.dtype.itemsize),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.block_chunk <= 0 or args.workers <= 0:
        raise ValueError("block_chunk and workers must be positive")
    source_dir = Path(args.source_profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    num_query_heads = int(source_summary["num_query_heads"])
    num_kv_heads = int(source_summary["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    union_flat_heads = sorted(
        {head for heads in selected_by_fold.values() for head in heads}
    )
    query_heads_by_layer: dict[int, list[int]] = {}
    for flat_head in union_flat_heads:
        layer_index, query_head = divmod(flat_head, num_query_heads)
        query_heads_by_layer.setdefault(layer_index, []).append(query_head)
    kv_heads_by_layer = {
        layer_index: sorted(
            {query_head // repeat_groups for query_head in query_heads}
        )
        for layer_index, query_heads in query_heads_by_layer.items()
    }

    tasks: list[dict[str, Any]] = []
    packed_shards: list[dict[str, Any]] = []
    for shard in source_summary["shards"]:
        rank = int(shard["rank"])
        packed_paths: dict[str, str] = {}
        for layer_index, kv_heads in sorted(kv_heads_by_layer.items()):
            layer = int(source_summary["layers"][layer_index])
            source_path = source_dir / Path(
                shard["layer_k_paths"][str(layer)]
            ).name
            filename = f"selected_k_layer{layer:03d}_rank{rank:03d}.npy"
            output_path = output_dir / filename
            packed_paths[str(layer)] = filename
            tasks.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "kv_heads": kv_heads,
                    "block_chunk": args.block_chunk,
                }
            )
        packed_shards.append(
            {
                "rank": rank,
                "block_start": int(shard["block_start"]),
                "block_end": int(shard["block_end"]),
                "layer_k_paths": packed_paths,
            }
        )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(pack_file, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: (row["layer"], row["rank"]))
    packed_bytes = sum(int(row["bytes"]) for row in results)
    source_bytes = sum(
        (source_dir / Path(path).name).stat().st_size
        for shard in source_summary["shards"]
        for path in shard["layer_k_paths"].values()
    )
    summary = {
        "experiment": "lossless_selected_layer_kv_profile_pack",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "source_profile_dir": str(source_dir),
        "source_profile_bytes": source_bytes,
        "packed_profile_bytes": packed_bytes,
        "packed_fraction_of_full": packed_bytes / source_bytes,
        "gate_feature": args.gate_feature,
        "heads_per_fold": args.heads_per_fold,
        "folds": len(selected_by_fold),
        "num_blocks": int(source_summary["num_blocks"]),
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "layers": [int(item) for item in source_summary["layers"]],
        "selected_layers": [
            int(source_summary["layers"][index])
            for index in sorted(query_heads_by_layer)
        ],
        "selected_query_heads_by_layer": {
            str(source_summary["layers"][index]): sorted(query_heads)
            for index, query_heads in query_heads_by_layer.items()
        },
        "selected_kv_heads_by_layer": {
            str(source_summary["layers"][index]): kv_heads
            for index, kv_heads in kv_heads_by_layer.items()
        },
        "union_flat_heads": union_flat_heads,
        "selected_query_head_channels": len(union_flat_heads),
        "selected_layer_kv_channels": sum(
            len(heads) for heads in kv_heads_by_layer.values()
        ),
        "shards": packed_shards,
        "files": results,
        "workers": args.workers,
        "total_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
