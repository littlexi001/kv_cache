from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lossless center-radius upper-bound summaries for selected "
            "K channels at several token-segment resolutions."
        )
    )
    parser.add_argument("--packed_profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--segments", default="1,2,4,8,16")
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=16)
    parser.add_argument("--block_chunk", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--radius_relative_margin", type=float, default=1e-6)
    parser.add_argument("--radius_absolute_margin", type=float, default=1e-6)
    return parser.parse_args()


def segment_slices(token_count: int, segments: int) -> list[tuple[int, int]]:
    if segments <= 0 or segments > token_count:
        raise ValueError("segments must be between 1 and the retained token count")
    quotient, remainder = divmod(token_count, segments)
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(segments):
        length = quotient + int(index < remainder)
        result.append((start, start + length))
        start += length
    return result


def build_file(task: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(task["source_path"])
    output_dir = Path(task["output_dir"])
    segment_counts = [int(item) for item in task["segments"]]
    prefix = int(task["exclude_block_prefix_tokens"])
    block_chunk = int(task["block_chunk"])
    relative_margin = float(task["radius_relative_margin"])
    absolute_margin = float(task["radius_absolute_margin"])
    source = np.load(source_path, mmap_mode="r")
    retained_tokens = int(source.shape[1]) - prefix
    if retained_tokens <= 0:
        raise ValueError("prefix exclusion removes every token")
    layer = int(task["layer"])
    rank = int(task["rank"])
    outputs: dict[int, tuple[np.memmap, np.memmap, Path, Path]] = {}
    for segments in segment_counts:
        center_path = output_dir / (
            f"centers_s{segments:02d}_layer{layer:03d}_rank{rank:03d}.npy"
        )
        radius_path = output_dir / (
            f"radii_s{segments:02d}_layer{layer:03d}_rank{rank:03d}.npy"
        )
        centers = np.lib.format.open_memmap(
            center_path,
            mode="w+",
            dtype=np.float32,
            shape=(source.shape[0], source.shape[2], segments, source.shape[3]),
        )
        radii = np.lib.format.open_memmap(
            radius_path,
            mode="w+",
            dtype=np.float32,
            shape=(source.shape[0], source.shape[2], segments),
        )
        outputs[segments] = (centers, radii, center_path, radius_path)

    slices = {
        segments: segment_slices(retained_tokens, segments)
        for segments in segment_counts
    }
    started = time.perf_counter()
    for offset in range(0, source.shape[0], block_chunk):
        end = min(source.shape[0], offset + block_chunk)
        keys = np.asarray(source[offset:end, prefix:], dtype=np.float32)
        for segments in segment_counts:
            centers, radii, _center_path, _radius_path = outputs[segments]
            for segment_index, (start, stop) in enumerate(slices[segments]):
                values = keys[:, start:stop]
                center = values.mean(axis=1, dtype=np.float32)
                residual = values - center[:, None]
                radius = np.sqrt(
                    np.sum(residual * residual, axis=-1, dtype=np.float32)
                ).max(axis=1)
                radius = radius * (1.0 + relative_margin) + absolute_margin
                centers[offset:end, :, segment_index] = center
                radii[offset:end, :, segment_index] = np.nextafter(
                    radius, np.float32(np.inf)
                )

    files: list[dict[str, Any]] = []
    for segments in segment_counts:
        centers, radii, center_path, radius_path = outputs[segments]
        centers.flush()
        radii.flush()
        center_shape = list(centers.shape)
        radius_shape = list(radii.shape)
        del centers, radii
        for path in (center_path, radius_path):
            with path.open("rb") as handle:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(
                        handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                    )
        files.append(
            {
                "segments": segments,
                "center_path": center_path.name,
                "radius_path": radius_path.name,
                "center_shape": center_shape,
                "radius_shape": radius_shape,
                "bytes": center_path.stat().st_size + radius_path.stat().st_size,
            }
        )
    return {
        "layer": layer,
        "rank": rank,
        "source_path": str(source_path),
        "source_shape": list(source.shape),
        "retained_tokens": retained_tokens,
        "files": files,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    segments = sorted({int(item) for item in args.segments.split(",")})
    if args.block_chunk <= 0 or args.workers <= 0:
        raise ValueError("block_chunk and workers must be positive")
    profile_dir = Path(args.packed_profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for shard in profile["shards"]:
        for layer, relative_path in shard["layer_k_paths"].items():
            tasks.append(
                {
                    "layer": int(layer),
                    "rank": int(shard["rank"]),
                    "source_path": str(profile_dir / Path(relative_path).name),
                    "output_dir": str(output_dir),
                    "segments": segments,
                    "exclude_block_prefix_tokens": args.exclude_block_prefix_tokens,
                    "block_chunk": args.block_chunk,
                    "radius_relative_margin": args.radius_relative_margin,
                    "radius_absolute_margin": args.radius_absolute_margin,
                }
            )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_file, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "layer": result["layer"],
                        "rank": result["rank"],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda row: (row["layer"], row["rank"]))
    output_bytes = sum(
        int(file["bytes"])
        for result in results
        for file in result["files"]
    )
    summary = {
        "experiment": "selected_kv_center_radius_support_bounds",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "bound": (
            "max_t q dot k_t <= max_j(q dot center_j + "
            "norm(q) * radius_j)"
        ),
        "packed_profile_dir": str(profile_dir),
        "num_blocks": int(profile["num_blocks"]),
        "num_query_heads": int(profile["num_query_heads"]),
        "num_kv_heads": int(profile["num_kv_heads"]),
        "selected_query_heads_by_layer": profile[
            "selected_query_heads_by_layer"
        ],
        "selected_kv_heads_by_layer": profile["selected_kv_heads_by_layer"],
        "segments": segments,
        "exclude_block_prefix_tokens": args.exclude_block_prefix_tokens,
        "retained_tokens": 256 - args.exclude_block_prefix_tokens,
        "center_dtype": "float32",
        "radius_dtype": "float32",
        "radius_relative_margin": args.radius_relative_margin,
        "radius_absolute_margin": args.radius_absolute_margin,
        "output_bytes": output_bytes,
        "files": results,
        "workers": args.workers,
        "total_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
