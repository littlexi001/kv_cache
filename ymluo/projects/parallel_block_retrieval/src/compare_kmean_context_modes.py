from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare block-local and full-record K centroid directions."
    )
    parser.add_argument("--block_local_index_dir", required=True)
    parser.add_argument("--record_context_index_dir", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    args = parse_args()
    local = np.asarray(
        np.load(Path(args.block_local_index_dir) / "pre_k_mean.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    record = np.asarray(
        np.load(Path(args.record_context_index_dir) / "pre_k_mean.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    if local.shape != record.shape:
        raise ValueError("centroid indices do not align")
    local_unit = local / np.maximum(np.linalg.norm(local, axis=2, keepdims=True), 1.0e-8)
    record_unit = record / np.maximum(
        np.linalg.norm(record, axis=2, keepdims=True), 1.0e-8
    )
    raw_cosine = np.sum(local_unit * record_unit, axis=2)
    local_residual = local - local.mean(axis=0, keepdims=True)
    record_residual = record - record.mean(axis=0, keepdims=True)
    local_residual /= np.maximum(
        np.linalg.norm(local_residual, axis=2, keepdims=True), 1.0e-8
    )
    record_residual /= np.maximum(
        np.linalg.norm(record_residual, axis=2, keepdims=True), 1.0e-8
    )
    residual_cosine = np.sum(local_residual * record_residual, axis=2)
    payload = {
        "source": "paired block-local versus full-record K centroid stability",
        "contains_synthetic_vectors": False,
        "blocks": int(local.shape[0]),
        "profiles": [
            {
                "profile": profile,
                "raw_centroid_cosine": stats(raw_cosine[:, profile]),
                "global_centered_centroid_cosine": stats(
                    residual_cosine[:, profile]
                ),
            }
            for profile in range(local.shape[1])
        ],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
