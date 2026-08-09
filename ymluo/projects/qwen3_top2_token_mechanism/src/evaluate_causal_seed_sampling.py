from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_coselection_expansion import (
    build_affinity_graph,
    expand_from_seeds,
    frequency_from_seeds,
    parse_labeled_path,
    recall,
)


def weighted_sample_without_replacement(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    uniform = np.clip(rng.random(values.size), 1.0e-12, 1.0 - 1.0e-12)
    priority = np.log(np.maximum(values, 1.0e-12)) - np.log(-np.log(uniform))
    return np.argpartition(priority, -count)[-count:].astype(np.int32)


def evaluate(
    label: str,
    path: Path,
    *,
    train_observations: int,
    seed_fraction: float,
    candidate_multipliers: tuple[float, ...],
    neighbor_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    payload = np.load(path)
    indices = payload["indices"]
    token_count = int(payload["context_token_ids"].shape[0])
    budget = int(payload["budget"][0])
    seed_count = max(1, int(math.ceil(seed_fraction * token_count)))
    if seed_count > budget:
        raise ValueError("causal seed budget cannot exceed the previous Top-2% budget")
    rng = np.random.default_rng(random_seed)
    totals: dict[tuple[str, str, float], float] = defaultdict(float)
    counts: dict[tuple[str, str, float], int] = defaultdict(int)

    for layer_slot in range(indices.shape[0]):
        for head_slot in range(indices.shape[1]):
            train = indices[layer_slot, head_slot, :train_observations]
            test = indices[layer_slot, head_slot, train_observations:]
            graph = build_affinity_graph(train, token_count, neighbor_count=neighbor_count)
            candidate_counts = {
                multiplier: min(token_count, max(budget, int(math.ceil(multiplier * budget))))
                for multiplier in candidate_multipliers
            }
            maximum_count = max(candidate_counts.values())

            for query_slot, target in enumerate(test):
                previous = train[-1] if query_slot == 0 else test[query_slot - 1]
                seed_sets = {
                    "uniform": rng.choice(token_count, size=seed_count, replace=False).astype(np.int32),
                    "history_weighted": weighted_sample_without_replacement(
                        graph.prior, seed_count, rng
                    ),
                    "previous_subset": rng.choice(
                        previous, size=seed_count, replace=False
                    ).astype(np.int32),
                }
                for seed_method, seeds in seed_sets.items():
                    candidates = {
                        "frequency": frequency_from_seeds(seeds, graph, maximum_count),
                        "graph": expand_from_seeds(seeds, graph, maximum_count),
                    }
                    for fill_method, candidate in candidates.items():
                        for multiplier, candidate_count in candidate_counts.items():
                            key = (seed_method, fill_method, multiplier)
                            totals[key] += recall(candidate[:candidate_count], target)
                            counts[key] += 1

    rows: list[dict[str, object]] = []
    for key in sorted(totals):
        seed_method, fill_method, multiplier = key
        rows.append(
            {
                "dataset": label,
                "context_tokens": token_count,
                "layers": int(indices.shape[0]),
                "heads": int(indices.shape[1]),
                "train_queries": train_observations,
                "test_queries": int(indices.shape[2] - train_observations),
                "seed_method": seed_method,
                "seed_fraction_of_history": seed_fraction,
                "seed_count": seed_count,
                "fill_method": fill_method,
                "candidate_multiplier_of_top2": multiplier,
                "candidate_fraction_of_history": multiplier * budget / token_count,
                "top2_position_recall": totals[key] / counts[key],
                "observations": counts[key],
                "graph_neighbors_per_token": neighbor_count,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out causal seed sampling evaluation.")
    parser.add_argument("--input", action="append", required=True, help="LABEL=selection_indices.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_observations", type=int, default=256)
    parser.add_argument("--seed_fraction", type=float, default=0.005)
    parser.add_argument("--candidate_multipliers", default="2,4")
    parser.add_argument("--neighbor_count", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=20260718)
    args = parser.parse_args()

    multipliers = tuple(float(item) for item in args.candidate_multipliers.split(",") if item)
    inputs = [parse_labeled_path(value) for value in args.input]
    rows: list[dict[str, object]] = []
    for label, path in inputs:
        rows.extend(
            evaluate(
                label,
                path,
                train_observations=args.train_observations,
                seed_fraction=args.seed_fraction,
                candidate_multipliers=multipliers,
                neighbor_count=args.neighbor_count,
                random_seed=args.random_seed,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in rows:
        print(
            row["dataset"],
            row["seed_method"],
            row["fill_method"],
            f"candidate={row['candidate_multiplier_of_top2']}x",
            f"recall={100.0 * float(row['top2_position_recall']):.2f}%",
        )


if __name__ == "__main__":
    main()
