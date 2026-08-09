from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce sharded full-record raw K profiles into block centroid indices."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--segments", type=int, default=4)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    block_count = int(source["num_blocks"])
    block_tokens = int(source["block_tokens"])
    profile_count = len(source["pair_specs"])
    head_dim = int(source["head_dim"])
    if block_tokens % args.segments:
        raise ValueError("block token count must be divisible by segments")
    segment_tokens = block_tokens // args.segments

    means = np.lib.format.open_memmap(
        output_dir / "pre_k_mean.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count, head_dim),
    )
    segments = np.lib.format.open_memmap(
        output_dir / "pre_k_segment_mean.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, args.segments, profile_count, head_dim),
    )
    concentration = np.lib.format.open_memmap(
        output_dir / "pre_concentration.npy",
        mode="w+",
        dtype=np.float16,
        shape=(block_count, profile_count),
    )
    started = time.perf_counter()
    for shard in source["shards"]:
        start = int(shard["block_start"])
        end = int(shard["block_end"])
        raw_path = Path(str(shard["raw_k_path"]))
        if not raw_path.exists():
            raw_path = profile_dir / raw_path.name
        raw = np.load(raw_path, mmap_mode="r")
        values = np.asarray(raw, dtype=np.float32)
        batch_mean = values.mean(axis=1)
        means[start:end] = batch_mean.astype(np.float16)
        segments[start:end] = values.reshape(
            end - start,
            args.segments,
            segment_tokens,
            profile_count,
            head_dim,
        ).mean(axis=2).astype(np.float16)
        mean_norm = np.linalg.norm(batch_mean, axis=-1)
        token_norm = np.linalg.norm(values, axis=-1).mean(axis=1)
        concentration[start:end] = (
            mean_norm / np.maximum(token_norm, 1.0e-8)
        ).astype(np.float16)
        print(json.dumps({"blocks": end, "total": block_count}), flush=True)

    for array in (means, segments, concentration):
        array.flush()
    elapsed = time.perf_counter() - started
    payload = {
        "source": "real full-record causal prefill K centroids reduced from raw shards",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "model": source["model_name_or_path"],
        "profile_dir": str(profile_dir),
        "corpus_dir": source.get("corpus_dir"),
        "profile_space": source["profile_space"],
        "context_mode": "full_record_causal_prefill",
        "blocks": block_count,
        "block_tokens": block_tokens,
        "tokens": block_count * block_tokens,
        "pair_specs": source["pair_specs"],
        "profiles": profile_count,
        "head_dim": head_dim,
        "segments": args.segments,
        "dtype": "float16",
        "pre_rope_concentration": distribution(concentration),
        "elapsed_seconds": elapsed,
        "pre_k_mean_path": "pre_k_mean.npy",
        "pre_k_segment_mean_path": "pre_k_segment_mean.npy",
        "post_local_k_mean_path": None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
