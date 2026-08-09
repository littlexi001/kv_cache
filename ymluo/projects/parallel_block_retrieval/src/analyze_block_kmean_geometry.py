from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure anisotropy and block separation in a K-mean index."
    )
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--random_pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def spectrum_metrics(matrix: np.ndarray) -> dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)[::-1]
    probability = eigenvalues / max(float(eigenvalues.sum()), 1.0e-30)
    cumulative = np.cumsum(probability)
    entropy = -float(np.sum(probability * np.log(probability + 1.0e-30)))
    return {
        "energy_at_rank": {
            str(rank): float(cumulative[min(rank, len(cumulative)) - 1])
            for rank in (1, 4, 8, 16, 32, 64)
        },
        "rank_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "rank_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "effective_rank_entropy": float(np.exp(entropy)),
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / max(float(np.square(eigenvalues).sum()), 1.0e-30)
        ),
    }


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-8)


def offset_cosines(matrix: np.ndarray, offsets: tuple[int, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset in offsets:
        if offset >= len(matrix):
            continue
        values = np.sum(matrix[:-offset] * matrix[offset:], axis=1)
        output[str(offset)] = stats(values)
    return output


def group_coherence(matrix: np.ndarray, group_sizes: tuple[int, ...]) -> dict[str, float]:
    output = {}
    token_energy = np.square(matrix).sum(axis=1)
    for group_size in group_sizes:
        usable = (len(matrix) // group_size) * group_size
        groups = matrix[:usable].reshape(-1, group_size, matrix.shape[1])
        mean_energy = np.square(groups.mean(axis=1)).sum(axis=1)
        base_energy = token_energy[:usable].reshape(-1, group_size).mean(axis=1)
        output[str(group_size)] = float(
            np.mean(mean_energy / np.maximum(base_energy, 1.0e-12))
        )
    return output


def main() -> None:
    args = parse_args()
    if args.random_pairs <= 0:
        raise ValueError("random_pairs must be positive")
    index_dir = Path(args.index_dir)
    summary = json.loads((index_dir / "summary.json").read_text(encoding="utf-8"))
    pre = np.asarray(
        np.load(index_dir / "pre_k_mean.npy", mmap_mode="r"), dtype=np.float32
    )
    post_path = index_dir / "post_local_k_mean.npy"
    post = (
        np.asarray(np.load(post_path, mmap_mode="r"), dtype=np.float32)
        if post_path.exists()
        else pre
    )
    segments = np.asarray(
        np.load(index_dir / "pre_k_segment_mean.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    rng = np.random.default_rng(args.seed)
    left = rng.integers(0, len(pre), size=args.random_pairs)
    right = rng.integers(0, len(pre), size=args.random_pairs)
    right = np.where(right == left, (right + 1) % len(pre), right)

    profiles = []
    for profile, spec in enumerate(summary["pair_specs"]):
        pre_values = pre[:, profile]
        post_values = post[:, profile]
        pre_unit = normalized(pre_values)
        post_unit = normalized(post_values)
        residual_values = pre_values - pre_values.mean(axis=0, keepdims=True)
        residual_unit = normalized(residual_values)
        segment_values = segments[:, :, profile]
        segment_unit = segment_values / np.maximum(
            np.linalg.norm(segment_values, axis=2, keepdims=True), 1.0e-8
        )
        pair_cos = np.sum(pre_unit[left] * pre_unit[right], axis=1)
        adjacent_cos = np.sum(pre_unit[:-1] * pre_unit[1:], axis=1)
        residual_pair_cos = np.sum(
            residual_unit[left] * residual_unit[right], axis=1
        )
        residual_adjacent_cos = np.sum(
            residual_unit[:-1] * residual_unit[1:], axis=1
        )
        post_pair_cos = np.sum(post_unit[left] * post_unit[right], axis=1)
        segment_pair_cos = np.concatenate(
            [
                np.sum(segment_unit[:, i] * segment_unit[:, j], axis=1)
                for i in range(segment_unit.shape[1])
                for j in range(i + 1, segment_unit.shape[1])
            ]
        )
        profiles.append(
            {
                "profile": profile,
                **spec,
                "global_mean_direction_norm": float(
                    np.linalg.norm(pre_unit.mean(axis=0))
                ),
                "random_block_cosine": stats(pair_cos),
                "adjacent_block_cosine": stats(adjacent_cos),
                "adjacent_minus_random_mean": float(adjacent_cos.mean() - pair_cos.mean()),
                "global_centered_random_block_cosine": stats(residual_pair_cos),
                "global_centered_adjacent_block_cosine": stats(residual_adjacent_cos),
                "global_centered_adjacent_minus_random_mean": float(
                    residual_adjacent_cos.mean() - residual_pair_cos.mean()
                ),
                "global_centered_cosine_by_block_offset": offset_cosines(
                    residual_unit, (1, 2, 4, 8, 16, 32, 64, 128)
                ),
                "global_centered_group_coherence": group_coherence(
                    residual_values, (2, 4, 8, 16, 32, 64)
                ),
                "post_local_random_block_cosine": stats(post_pair_cos),
                "within_block_segment_cosine": stats(segment_pair_cos),
                "centroid_spectrum": spectrum_metrics(pre_values),
                "unit_centroid_spectrum": spectrum_metrics(pre_unit),
            }
        )

    payload = {
        "source": "unsupervised geometry of real block K centroids",
        "contains_synthetic_vectors": False,
        "index_dir": str(index_dir),
        "blocks": len(pre),
        "random_pairs": args.random_pairs,
        "post_local_centroids_available": post_path.exists(),
        "profiles": profiles,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
