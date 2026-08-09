from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


OFFSETS = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure global-centered K locality within real LongBench records."
    )
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--blocks_metadata_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--random_pairs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit_length(offsets: list[int], correlations: list[float]) -> float | None:
    x = np.asarray(offsets, dtype=np.float64)
    y = np.asarray(correlations, dtype=np.float64)
    valid = y > 0.01
    if valid.sum() < 2:
        return None
    slope = float(np.polyfit(x[valid], np.log(y[valid]), 1)[0])
    return -1.0 / slope if slope < 0 else None


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    values = np.asarray(
        np.load(index_dir / "pre_k_mean.npy", mmap_mode="r"), dtype=np.float32
    )
    metadata = read_jsonl(Path(args.blocks_metadata_path))
    if len(metadata) != len(values):
        raise ValueError("block metadata and centroid index do not align")
    values = values - values.mean(axis=0, keepdims=True)
    values = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1.0e-8)
    datasets = np.asarray([str(row["dataset"]) for row in metadata])
    records = np.asarray([str(row["record_uid"]) for row in metadata])
    rng = np.random.default_rng(args.seed)

    results = []
    for dataset in sorted(set(datasets.tolist())):
        dataset_ids = np.flatnonzero(datasets == dataset)
        if len(dataset_ids) < 2:
            continue
        profile_rows = []
        for profile in range(values.shape[1]):
            offset_means = []
            offset_counts = []
            for offset in OFFSETS:
                left = dataset_ids[dataset_ids + offset < len(values)]
                right = left + offset
                valid = (datasets[right] == dataset) & (records[left] == records[right])
                left = left[valid]
                right = right[valid]
                cosine = np.sum(values[left, profile] * values[right, profile], axis=1)
                offset_means.append(float(cosine.mean()) if len(cosine) else float("nan"))
                offset_counts.append(int(len(cosine)))

            sample_count = min(args.random_pairs, len(dataset_ids) * 10)
            left = rng.choice(dataset_ids, size=sample_count, replace=True)
            right = rng.choice(dataset_ids, size=sample_count, replace=True)
            different = records[left] != records[right]
            left = left[different]
            right = right[different]
            random_cosine = np.sum(
                values[left, profile] * values[right, profile], axis=1
            )
            finite = [
                (offset, mean)
                for offset, mean in zip(OFFSETS, offset_means)
                if np.isfinite(mean)
            ]
            random_mean = float(random_cosine.mean())
            excess = [(offset, mean - random_mean) for offset, mean in finite]
            profile_rows.append(
                {
                    "profile": profile,
                    "offset_cosine": {
                        str(offset): {"mean": mean, "pairs": count}
                        for offset, mean, count in zip(
                            OFFSETS, offset_means, offset_counts
                        )
                    },
                    "different_record_random_cosine_mean": random_mean,
                    "exponential_correlation_length_blocks": fit_length(
                        [item[0] for item in finite], [item[1] for item in finite]
                    ),
                    "excess_over_cross_record_correlation_length_blocks": fit_length(
                        [item[0] for item in excess], [item[1] for item in excess]
                    ),
                }
            )
        results.append(
            {
                "dataset": dataset,
                "blocks": int(len(dataset_ids)),
                "records": int(len(set(records[dataset_ids].tolist()))),
                "mean_adjacent_cosine_across_profiles": float(
                    np.mean(
                        [row["offset_cosine"]["1"]["mean"] for row in profile_rows]
                    )
                ),
                "mean_different_record_random_cosine_across_profiles": float(
                    np.mean(
                        [
                            row["different_record_random_cosine_mean"]
                            for row in profile_rows
                        ]
                    )
                ),
                "mean_correlation_length_blocks": float(
                    np.mean(
                        [
                            row["exponential_correlation_length_blocks"]
                            for row in profile_rows
                            if row["exponential_correlation_length_blocks"] is not None
                        ]
                    )
                ),
                "mean_excess_correlation_length_blocks": float(
                    np.mean(
                        [
                            row["excess_over_cross_record_correlation_length_blocks"]
                            for row in profile_rows
                            if row[
                                "excess_over_cross_record_correlation_length_blocks"
                            ]
                            is not None
                        ]
                    )
                ),
                "profiles": profile_rows,
            }
        )

    payload = {
        "source": "within-record locality of global-centered real K block centroids",
        "contains_synthetic_vectors": False,
        "index_dir": str(index_dir),
        "blocks_metadata_path": args.blocks_metadata_path,
        "offsets": list(OFFSETS),
        "datasets": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
