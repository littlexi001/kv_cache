from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SparseAffinityGraph:
    prior: np.ndarray
    prior_order: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray


def build_affinity_graph(
    train_indices: np.ndarray,
    token_count: int,
    *,
    neighbor_count: int = 32,
    smoothing: float = 0.5,
) -> SparseAffinityGraph:
    """Learn positive conditional excess and retain a bounded adjacency list."""
    values = np.asarray(train_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("train_indices must have shape [observations, budget].")
    if values.shape[0] == 0:
        raise ValueError("at least one training observation is required.")
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if values.size and (values.min() < 0 or values.max() >= token_count):
        raise ValueError("selection index is outside token_count.")

    observations = values.shape[0]
    rows = np.repeat(np.arange(observations, dtype=np.int32), values.shape[1])
    columns = values.reshape(-1).astype(np.int32, copy=False)
    incidence = csr_matrix(
        (np.ones(columns.size, dtype=np.float32), (rows, columns)),
        shape=(observations, token_count),
    )
    incidence.data[:] = 1.0
    incidence.eliminate_zeros()
    count = np.asarray(incidence.sum(axis=0)).reshape(-1)
    prior = (count + smoothing) / (observations + 2.0 * smoothing)
    cooccurrence = (incidence.T @ incidence).tocsr()
    retained = min(max(1, int(neighbor_count)), max(1, token_count - 1))
    neighbor_indices = np.zeros((token_count, retained), dtype=np.int32)
    neighbor_weights = np.zeros((token_count, retained), dtype=np.float32)
    for token in range(token_count):
        start, end = cooccurrence.indptr[token : token + 2]
        candidates = cooccurrence.indices[start:end]
        pair_count = cooccurrence.data[start:end]
        keep = candidates != token
        candidates = candidates[keep]
        pair_count = pair_count[keep]
        if candidates.size == 0:
            continue
        weights = np.maximum(
            (pair_count + smoothing * prior[candidates]) / (count[token] + smoothing)
            - prior[candidates],
            0.0,
        )
        positive = weights > 0.0
        candidates = candidates[positive]
        weights = weights[positive]
        if candidates.size == 0:
            continue
        take = min(retained, candidates.size)
        if candidates.size > take:
            selected = np.argpartition(weights, -take)[-take:]
            candidates = candidates[selected]
            weights = weights[selected]
        order = np.argsort(-weights)
        neighbor_indices[token, :take] = candidates[order].astype(np.int32)
        neighbor_weights[token, :take] = weights[order].astype(np.float32)

    return SparseAffinityGraph(
        prior=prior.astype(np.float32),
        prior_order=np.argsort(-prior).astype(np.int32),
        neighbors=neighbor_indices,
        weights=neighbor_weights,
    )


def _deduplicate(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _fill_with_prior(selected: list[int], graph: SparseAffinityGraph, count: int) -> np.ndarray:
    selected = _deduplicate(selected)
    seen = set(selected)
    if len(selected) < count:
        for token in graph.prior_order:
            item = int(token)
            if item not in seen:
                selected.append(item)
                seen.add(item)
                if len(selected) == count:
                    break
    return np.asarray(selected[:count], dtype=np.int32)


def expand_from_seeds(
    seeds: np.ndarray,
    graph: SparseAffinityGraph,
    candidate_count: int,
    *,
    reduction: str = "sum",
) -> np.ndarray:
    """Expand seeds through sparse edges without scanning all historical tokens."""
    seed_values = np.unique(np.asarray(seeds, dtype=np.int64))
    if candidate_count < seed_values.size:
        raise ValueError("candidate_count cannot be smaller than the number of seeds.")
    if reduction not in {"sum", "max"}:
        raise ValueError("reduction must be 'sum' or 'max'.")

    seed_set = {int(value) for value in seed_values}
    scores: dict[int, float] = {}
    for seed in seed_values:
        for neighbor, weight in zip(graph.neighbors[seed], graph.weights[seed]):
            item = int(neighbor)
            if item in seed_set or weight <= 0.0:
                continue
            if reduction == "sum":
                scores[item] = scores.get(item, 0.0) + float(weight)
            else:
                scores[item] = max(scores.get(item, 0.0), float(weight))

    ranked = sorted(scores, key=lambda item: (-scores[item], item))
    return _fill_with_prior([*seed_values.tolist(), *ranked], graph, candidate_count)


def frequency_from_seeds(
    seeds: np.ndarray,
    graph: SparseAffinityGraph,
    candidate_count: int,
) -> np.ndarray:
    return _fill_with_prior(np.unique(seeds).tolist(), graph, candidate_count)


def local_from_seeds(
    seeds: np.ndarray,
    graph: SparseAffinityGraph,
    candidate_count: int,
    *,
    token_count: int,
    radius: int = 16,
) -> np.ndarray:
    seed_values = np.unique(np.asarray(seeds, dtype=np.int64))
    votes: dict[int, int] = defaultdict(int)
    seed_set = {int(value) for value in seed_values}
    for seed in seed_values:
        for token in range(max(0, int(seed) - radius), min(token_count, int(seed) + radius + 1)):
            if token not in seed_set:
                votes[token] += 1
    ranked = sorted(votes, key=lambda item: (-votes[item], item))
    return _fill_with_prior([*seed_values.tolist(), *ranked], graph, candidate_count)


def recall(candidate: np.ndarray, target: np.ndarray) -> float:
    return len(set(np.asarray(candidate).tolist()) & set(np.asarray(target).tolist())) / len(target)


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", maxsplit=1)
    return label, Path(path)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def evaluate_file(
    label: str,
    path: Path,
    *,
    train_observations: int,
    seed_fractions: list[float],
    candidate_multipliers: list[float],
    neighbor_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    payload = np.load(path)
    indices = payload["indices"]
    token_count = int(payload["context_token_ids"].shape[0])
    budget = int(payload["budget"][0])
    if not 0 < train_observations < indices.shape[2]:
        raise ValueError("train_observations must leave at least one held-out query.")

    totals: dict[tuple[str, float, str, float], float] = defaultdict(float)
    counts: dict[tuple[str, float, str, float], int] = defaultdict(int)
    rng = np.random.default_rng(random_seed)

    for layer_slot in range(indices.shape[0]):
        for head_slot in range(indices.shape[1]):
            train = indices[layer_slot, head_slot, :train_observations]
            test = indices[layer_slot, head_slot, train_observations:]
            graph = build_affinity_graph(train, token_count, neighbor_count=neighbor_count)

            for query_slot, target in enumerate(test):
                previous = train[-1] if query_slot == 0 else test[query_slot - 1]
                candidate_counts = {
                    multiplier: min(token_count, max(budget, int(math.ceil(multiplier * budget))))
                    for multiplier in candidate_multipliers
                }
                maximum_candidate_count = max(candidate_counts.values())
                causal_candidates = {
                    "previous_frequency": frequency_from_seeds(previous, graph, maximum_candidate_count),
                    "previous_graph": expand_from_seeds(previous, graph, maximum_candidate_count),
                }
                for multiplier in candidate_multipliers:
                    candidate_count = candidate_counts[multiplier]
                    for method, candidate in causal_candidates.items():
                        key = ("causal_previous", 1.0, method, multiplier)
                        totals[key] += recall(candidate[:candidate_count], target)
                        counts[key] += 1

                for seed_fraction in seed_fractions:
                    seed_count = min(budget, max(1, int(math.ceil(seed_fraction * budget))))
                    seeds = rng.choice(target, size=seed_count, replace=False)
                    methods = {
                        "frequency": frequency_from_seeds(seeds, graph, maximum_candidate_count),
                        "local16": local_from_seeds(
                            seeds,
                            graph,
                            maximum_candidate_count,
                            token_count=token_count,
                        ),
                        "graph_sum": expand_from_seeds(
                            seeds,
                            graph,
                            maximum_candidate_count,
                            reduction="sum",
                        ),
                        "graph_max": expand_from_seeds(
                            seeds,
                            graph,
                            maximum_candidate_count,
                            reduction="max",
                        ),
                    }
                    for multiplier in candidate_multipliers:
                        candidate_count = candidate_counts[multiplier]
                        for method, candidate in methods.items():
                            key = ("current_oracle_seed", seed_fraction, method, multiplier)
                            totals[key] += recall(candidate[:candidate_count], target)
                            counts[key] += 1

    rows: list[dict[str, object]] = []
    for key in sorted(totals):
        mode, seed_fraction, method, multiplier = key
        rows.append(
            {
                "dataset": label,
                "context_tokens": token_count,
                "layers": int(indices.shape[0]),
                "heads": int(indices.shape[1]),
                "train_queries": train_observations,
                "test_queries": int(indices.shape[2] - train_observations),
                "top2_budget": budget,
                "mode": mode,
                "seed_fraction_of_top2": seed_fraction,
                "seed_fraction_of_history": seed_fraction * budget / token_count,
                "method": method,
                "candidate_multiplier_of_top2": multiplier,
                "candidate_fraction_of_history": min(1.0, multiplier * budget / token_count),
                "top2_position_recall": totals[key] / counts[key],
                "observations": counts[key],
                "graph_neighbors_per_token": neighbor_count,
            }
        )
    return rows


def write_rows(rows: list[dict[str, object]], output_dir: Path, config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "results": rows}, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out evaluation of sparse Top-2% co-selection expansion.")
    parser.add_argument("--input", action="append", required=True, help="LABEL=selection_indices.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_observations", type=int, default=256)
    parser.add_argument("--seed_fractions", default="0.125,0.25,0.5")
    parser.add_argument("--candidate_multipliers", default="1,2,4")
    parser.add_argument("--neighbor_count", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=20260718)
    args = parser.parse_args()

    seed_fractions = parse_float_list(args.seed_fractions)
    candidate_multipliers = parse_float_list(args.candidate_multipliers)
    labeled_paths = [parse_labeled_path(value) for value in args.input]
    rows: list[dict[str, object]] = []
    for label, path in labeled_paths:
        rows.extend(
            evaluate_file(
                label,
                path,
                train_observations=args.train_observations,
                seed_fractions=seed_fractions,
                candidate_multipliers=candidate_multipliers,
                neighbor_count=args.neighbor_count,
                random_seed=args.random_seed,
            )
        )

    config = {
        "inputs": {label: str(path) for label, path in labeled_paths},
        "train_observations": args.train_observations,
        "seed_fractions": seed_fractions,
        "candidate_multipliers": candidate_multipliers,
        "neighbor_count": args.neighbor_count,
        "random_seed": args.random_seed,
        "split_contract": "first train_observations queries build graph; all later queries are held out",
    }
    write_rows(rows, Path(args.output_dir), config)
    for row in rows:
        if (
            row["mode"] == "current_oracle_seed"
            and row["seed_fraction_of_top2"] == 0.25
            and row["method"] in {"frequency", "graph_sum"}
        ):
            print(
                f"{row['dataset']} {row['method']} candidate={row['candidate_multiplier_of_top2']}x "
                f"recall={100.0 * float(row['top2_position_recall']):.2f}%"
            )


if __name__ == "__main__":
    main()
