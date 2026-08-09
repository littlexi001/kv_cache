from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from benchmark_selected_head_debiased_retrieval import read_selection
from run_all_head_prior_debiased_retrieval import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether train-only selected-head Q directions form a "
            "low-dimensional or cluster-coverable manifold on held-out datasets."
        )
    )
    parser.add_argument("--query_profiles", required=True)
    parser.add_argument("--selection_csv", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--fold_reference_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_feature", default="raw_top1_block_diversity")
    parser.add_argument("--heads_per_fold", type=int, default=16)
    parser.add_argument("--prototype_counts", default="8,32,128")
    parser.add_argument("--subspace_ranks", default="4,8,16,24")
    parser.add_argument("--max_train_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def valid_vectors(
    q: np.ndarray,
    mask: np.ndarray,
    query_indices: np.ndarray,
    layer_index: int,
    query_head: int,
) -> np.ndarray:
    values = q[query_indices, :, layer_index, query_head].astype(np.float32)
    valid = mask[query_indices].astype(bool)
    return values[valid]


def energy_rank(eigenvalues: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(eigenvalues)
    return int(np.searchsorted(cumulative, threshold * cumulative[-1]) + 1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_counts = sorted(
        {int(item) for item in args.prototype_counts.split(",")}
    )
    subspace_ranks = sorted(
        {int(item) for item in args.subspace_ranks.split(",")}
    )
    selected_by_fold = read_selection(
        Path(args.selection_csv), args.gate_feature, args.heads_per_fold
    )
    payload = torch.load(
        Path(args.query_profiles), map_location="cpu", weights_only=False
    )
    q = payload["svd_q"].numpy()
    mask = payload["mask"].numpy()
    queries = read_jsonl(Path(args.queries_jsonl))
    with np.load(Path(args.fold_reference_npz)) as reference:
        fold_ids = reference["fold_ids"].astype(np.int64)
        layers = reference["layers"].astype(np.int64)
    datasets = np.asarray([str(query["dataset"]) for query in queries])
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    for fold in sorted(selected_by_fold):
        train_queries = np.flatnonzero(fold_ids != fold)
        test_queries = np.flatnonzero(fold_ids == fold)
        heldout_dataset = str(np.unique(datasets[test_queries])[0])
        for flat_head in selected_by_fold[fold]:
            layer_index, query_head = divmod(flat_head, q.shape[3])
            train = valid_vectors(
                q, mask, train_queries, layer_index, query_head
            )
            test = valid_vectors(q, mask, test_queries, layer_index, query_head)
            train_directions = normalize_rows(train)
            test_directions = normalize_rows(test)
            if len(train_directions) > args.max_train_tokens:
                chosen = rng.choice(
                    len(train_directions),
                    size=args.max_train_tokens,
                    replace=False,
                )
                train_fit = train_directions[chosen]
            else:
                train_fit = train_directions

            covariance = train_fit.T @ train_fit / len(train_fit)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            eigenvectors = eigenvectors[:, order]
            probability = eigenvalues / eigenvalues.sum()
            effective_rank = float(np.exp(-(probability * np.log(
                np.maximum(probability, 1e-30)
            )).sum()))
            row: dict[str, Any] = {
                "fold": fold,
                "heldout_dataset": heldout_dataset,
                "flat_head": flat_head,
                "layer": int(layers[layer_index]),
                "query_head": query_head,
                "train_queries": len(train_queries),
                "test_queries": len(test_queries),
                "train_tokens": len(train_directions),
                "test_tokens": len(test_directions),
                "direction_rank90": energy_rank(eigenvalues, 0.90),
                "direction_rank95": energy_rank(eigenvalues, 0.95),
                "direction_effective_rank": effective_rank,
            }
            for subspace_rank in subspace_ranks:
                basis = eigenvectors[:, :subspace_rank]
                projection_energy = np.sum((test_directions @ basis) ** 2, axis=1)
                residual = np.sqrt(np.maximum(0.0, 1.0 - projection_energy))
                row[f"rank{subspace_rank}_test_residual_mean"] = float(
                    residual.mean()
                )
                row[f"rank{subspace_rank}_test_residual_p95"] = float(
                    np.percentile(residual, 95)
                )

            for prototypes in prototype_counts:
                model = MiniBatchKMeans(
                    n_clusters=prototypes,
                    batch_size=min(1024, len(train_fit)),
                    max_iter=100,
                    n_init=1,
                    random_state=(
                        args.seed + fold * 1000 + flat_head * 17 + prototypes
                    ),
                    reassignment_ratio=0.01,
                )
                model.fit(train_fit)
                centers = normalize_rows(model.cluster_centers_.astype(np.float32))
                nearest_cosine = np.max(test_directions @ centers.T, axis=1)
                nearest_cosine = np.clip(nearest_cosine, -1.0, 1.0)
                chord_distance = np.sqrt(2.0 - 2.0 * nearest_cosine)
                row[f"proto{prototypes}_nearest_cosine_mean"] = float(
                    nearest_cosine.mean()
                )
                row[f"proto{prototypes}_nearest_cosine_p05"] = float(
                    np.percentile(nearest_cosine, 5)
                )
                row[f"proto{prototypes}_chord_mean"] = float(
                    chord_distance.mean()
                )
                row[f"proto{prototypes}_chord_p95"] = float(
                    np.percentile(chord_distance, 95)
                )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "dataset": heldout_dataset,
                        "flat_head": flat_head,
                    }
                ),
                flush=True,
            )

    write_csv(output_dir / "fold_head_metrics.csv", rows)
    numeric_fields = [
        key
        for key in rows[0]
        if key
        not in {
            "fold",
            "heldout_dataset",
            "flat_head",
            "layer",
            "query_head",
            "train_queries",
            "test_queries",
            "train_tokens",
            "test_tokens",
        }
    ]
    aggregate = {
        field: {
            "mean": float(np.mean([float(row[field]) for row in rows])),
            "median": float(np.median([float(row[field]) for row in rows])),
            "p05": float(np.percentile([float(row[field]) for row in rows], 5)),
            "p95": float(np.percentile([float(row[field]) for row in rows], 95)),
        }
        for field in numeric_fields
    }
    dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted(set(str(row["heldout_dataset"]) for row in rows)):
        subset = [row for row in rows if row["heldout_dataset"] == dataset]
        item: dict[str, Any] = {"heldout_dataset": dataset, "heads": len(subset)}
        for field in numeric_fields:
            item[field] = float(np.mean([float(row[field]) for row in subset]))
        dataset_rows.append(item)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    summary = {
        "experiment": "lodo_selected_head_query_direction_manifold",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "selection_uses_heldout_queries": False,
        "queries": len(queries),
        "fold_head_pairs": len(rows),
        "svd_dimension": q.shape[-1],
        "prototype_counts": prototype_counts,
        "subspace_ranks": subspace_ranks,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
