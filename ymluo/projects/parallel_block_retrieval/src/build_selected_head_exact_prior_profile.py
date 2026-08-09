from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_selected_head_debiased_retrieval import read_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only exact-QK block mean/std profiles for the selected "
            "LODO fold-head models."
        )
    )
    parser.add_argument("--packed_profile_dir", required=True)
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--query_batch", type=int, default=8)
    parser.add_argument("--block_chunk", type=int, default=64)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--std_epsilon", type=float, default=1e-4)
    parser.add_argument("--save_raw_scores", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("exact prior construction requires CUDA")
    if args.query_batch <= 0 or args.block_chunk <= 0 or args.std_epsilon <= 0:
        raise ValueError("batch sizes and epsilon must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.packed_profile_dir)
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    query_vectors = payload["svd_q"]
    query_mask = payload["mask"]
    with np.load(Path(args.reference_npz)) as reference:
        fold_ids_np = reference["fold_ids"].astype(np.int64)
        layers = reference["layers"].astype(np.int64)

    num_blocks = int(profile["num_blocks"])
    num_query_heads = int(profile["num_query_heads"])
    num_kv_heads = int(profile["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    model_rows: list[dict[str, Any]] = []
    for fold in sorted(selected_by_fold):
        for flat_head in selected_by_fold[fold]:
            layer_index, query_head = divmod(flat_head, num_query_heads)
            model_rows.append(
                {
                    "model_index": len(model_rows),
                    "fold": fold,
                    "flat_head": int(flat_head),
                    "layer_index": layer_index,
                    "layer": int(layers[layer_index]),
                    "query_head": query_head,
                    "train_queries": int(np.sum(fold_ids_np != fold)),
                }
            )

    union_flat_heads = sorted({int(row["flat_head"]) for row in model_rows})
    flat_head_to_output = {
        flat_head: index for index, flat_head in enumerate(union_flat_heads)
    }

    means_path = output_dir / "exact_train_mean.npy"
    stds_path = output_dir / "exact_train_std.npy"
    means_partial = output_dir / "exact_train_mean.partial.npy"
    stds_partial = output_dir / "exact_train_std.partial.npy"
    means_partial.unlink(missing_ok=True)
    stds_partial.unlink(missing_ok=True)
    means = np.lib.format.open_memmap(
        means_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(model_rows), num_blocks),
    )
    stds = np.lib.format.open_memmap(
        stds_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(model_rows), num_blocks),
    )
    raw_scores_path = output_dir / "exact_raw_selected_scores.npy"
    raw_scores_partial = output_dir / "exact_raw_selected_scores.partial.npy"
    raw_scores: np.memmap | None = None
    if args.save_raw_scores:
        raw_scores_partial.unlink(missing_ok=True)
        raw_scores = np.lib.format.open_memmap(
            raw_scores_partial,
            mode="w+",
            dtype=np.float32,
            shape=(len(query_vectors), len(union_flat_heads), num_blocks),
        )

    fold_ids = torch.from_numpy(fold_ids_np).to(device=device)
    mask = query_mask.to(device=device, non_blocking=True)
    valid = mask.sum(dim=1).clamp_min(1).float()
    layer_rows: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for layer_index in sorted({row["layer_index"] for row in model_rows}):
        layer_started = time.perf_counter()
        layer = int(layers[layer_index])
        layer_models = [
            row for row in model_rows if row["layer_index"] == layer_index
        ]
        query_heads = sorted({int(row["query_head"]) for row in layer_models})
        query_head_to_local = {
            query_head: index for index, query_head in enumerate(query_heads)
        }
        packed_kv_heads = [
            int(item)
            for item in profile["selected_kv_heads_by_layer"][str(layer)]
        ]
        stored_kv_indices = [
            packed_kv_heads.index(query_head // repeat_groups)
            for query_head in query_heads
        ]
        layer_queries = query_vectors[:, :, layer_index, query_heads].to(
            device=device, non_blocking=True
        )

        for shard in profile["shards"]:
            shard_start = int(shard["block_start"])
            shard_end = int(shard["block_end"])
            source = np.load(
                profile_dir / Path(shard["layer_k_paths"][str(layer)]).name,
                mmap_mode="r",
            )
            for local_start in range(0, shard_end - shard_start, args.block_chunk):
                local_end = min(
                    shard_end - shard_start, local_start + args.block_chunk
                )
                key_array = np.take(
                    source[local_start:local_end],
                    np.asarray(stored_kv_indices, dtype=np.int64),
                    axis=2,
                )
                keys = torch.from_numpy(np.array(key_array, copy=True)).to(
                    device=device, non_blocking=True
                )
                keys = keys[:, args.exclude_block_prefix_tokens :]
                chunk_scores = torch.empty(
                    (
                        len(query_vectors),
                        len(query_heads),
                        local_end - local_start,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                for query_start in range(0, len(query_vectors), args.query_batch):
                    query_end = min(
                        len(query_vectors), query_start + args.query_batch
                    )
                    similarities = torch.einsum(
                        "qihd,bthd->qihbt",
                        layer_queries[query_start:query_end],
                        keys,
                    )
                    token_max = similarities.amax(dim=-1).float()
                    chunk_scores[query_start:query_end] = (
                        token_max
                        * mask[query_start:query_end, :, None, None]
                    ).sum(dim=1) / valid[query_start:query_end, None, None]

                global_start = shard_start + local_start
                global_end = shard_start + local_end
                if raw_scores is not None:
                    output_indices = [
                        flat_head_to_output[layer_index * num_query_heads + head]
                        for head in query_heads
                    ]
                    raw_scores[
                        :, output_indices, global_start:global_end
                    ] = chunk_scores.cpu().numpy()
                for row in layer_models:
                    train = fold_ids != int(row["fold"])
                    values = chunk_scores[
                        train, query_head_to_local[int(row["query_head"])]
                    ]
                    mean = values.mean(dim=0)
                    variance = (
                        values.square().mean(dim=0) - mean.square()
                    ).clamp_min(0)
                    means[int(row["model_index"]), global_start:global_end] = (
                        mean.cpu().numpy()
                    )
                    stds[int(row["model_index"]), global_start:global_end] = (
                        variance.sqrt().clamp_min(args.std_epsilon).cpu().numpy()
                    )

        torch.cuda.synchronize(device)
        layer_row = {
            "layer": layer,
            "layer_index": layer_index,
            "query_heads": query_heads,
            "fold_head_models": len(layer_models),
            "seconds": time.perf_counter() - layer_started,
        }
        layer_rows.append(layer_row)
        print(json.dumps(layer_row), flush=True)

    means.flush()
    stds.flush()
    if raw_scores is not None:
        raw_scores.flush()
    del means, stds
    if raw_scores is not None:
        del raw_scores
    means_partial.replace(means_path)
    stds_partial.replace(stds_path)
    if args.save_raw_scores:
        raw_scores_partial.replace(raw_scores_path)
        np.save(
            output_dir / "selected_flat_heads.npy",
            np.asarray(union_flat_heads, dtype=np.int32),
        )
    write_csv(output_dir / "models.csv", model_rows)
    write_csv(output_dir / "layer_runtime.csv", layer_rows)
    summary = {
        "experiment": "selected_head_exact_train_prior_profile",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "selection_uses_heldout_queries": False,
        "profile_source": str(profile_dir),
        "queries": int(len(query_vectors)),
        "blocks": num_blocks,
        "fold_head_models": len(model_rows),
        "selected_layers": len(layer_rows),
        "mean_dtype": "float32",
        "std_dtype": "float32",
        "std_epsilon": args.std_epsilon,
        "total_bytes": means_path.stat().st_size + stds_path.stat().st_size,
        "raw_scores_saved": args.save_raw_scores,
        "raw_scores_bytes": raw_scores_path.stat().st_size
        if args.save_raw_scores
        else 0,
        "total_wall_seconds": time.perf_counter() - total_started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
