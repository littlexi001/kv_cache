from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_coselection_expansion import build_affinity_graph, parse_labeled_path


def union_query_heads(indices: np.ndarray) -> list[np.ndarray]:
    if indices.ndim != 3:
        raise ValueError("indices must have shape [query_heads, queries, budget]")
    return [
        np.unique(indices[:, query].reshape(-1)).astype(np.int32)
        for query in range(indices.shape[1])
    ]


def complete_query_head_groups(
    selected_heads: np.ndarray, group_size: int
) -> list[np.ndarray]:
    selected = np.asarray(selected_heads, dtype=np.int32)
    slot_by_head = {int(head): slot for slot, head in enumerate(selected)}
    groups: list[np.ndarray] = []
    for group_start in sorted({int(head) // group_size * group_size for head in selected}):
        heads = range(group_start, group_start + group_size)
        if all(head in slot_by_head for head in heads):
            groups.append(
                np.asarray([slot_by_head[head] for head in heads], dtype=np.int32)
            )
    return groups


def padded_rows(rows: list[np.ndarray]) -> np.ndarray:
    width = max(len(row) for row in rows)
    result = np.empty((len(rows), width), dtype=np.int32)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row
        result[index, len(row) :] = row[0]
    return result


def coaccess_physical_positions(
    training_token_sets: list[np.ndarray],
    token_count: int,
    *,
    page_size: int,
    microblock_size: int,
    neighbor_count: int,
) -> np.ndarray:
    if page_size % microblock_size != 0:
        raise ValueError("page_size must be divisible by microblock_size")
    node_count = math.ceil(token_count / microblock_size)
    page_capacity = page_size // microblock_size
    training_nodes = [np.unique(row // microblock_size) for row in training_token_sets]
    graph = build_affinity_graph(
        padded_rows(training_nodes), node_count, neighbor_count=neighbor_count
    )

    assigned = np.zeros(node_count, dtype=bool)
    physical_order: list[int] = []
    chronological_cursor = 0
    for root_value in graph.prior_order:
        root = int(root_value)
        if assigned[root]:
            continue
        page_nodes: list[int] = []
        frontier: dict[int, float] = {root: float("inf")}
        while len(page_nodes) < page_capacity:
            selected = -1
            while frontier:
                candidate = max(frontier, key=lambda item: (frontier[item], -item))
                frontier.pop(candidate)
                if not assigned[candidate]:
                    selected = candidate
                    break
            if selected < 0:
                while chronological_cursor < node_count and assigned[chronological_cursor]:
                    chronological_cursor += 1
                if chronological_cursor == node_count:
                    break
                selected = chronological_cursor
                chronological_cursor += 1

            assigned[selected] = True
            page_nodes.append(selected)
            for neighbor, weight in zip(
                graph.neighbors[selected], graph.weights[selected]
            ):
                neighbor_value = int(neighbor)
                if weight > 0.0 and not assigned[neighbor_value]:
                    frontier[neighbor_value] = frontier.get(neighbor_value, 0.0) + float(weight)
        physical_order.extend(page_nodes)

    if len(physical_order) != node_count or len(set(physical_order)) != node_count:
        raise RuntimeError("co-access packing did not produce a node permutation")
    inverse_node = np.empty(node_count, dtype=np.int32)
    inverse_node[np.asarray(physical_order, dtype=np.int32)] = np.arange(
        node_count, dtype=np.int32
    )
    positions = np.empty(token_count, dtype=np.int32)
    for token in range(token_count):
        positions[token] = (
            int(inverse_node[token // microblock_size]) * microblock_size
            + token % microblock_size
        )
    if np.unique(positions).size != token_count:
        raise RuntimeError("token physical positions are not a permutation")
    return positions


def page_counts(
    token_sets: list[np.ndarray], positions: np.ndarray, page_size: int
) -> np.ndarray:
    return np.asarray(
        [np.unique(positions[row] // page_size).size for row in token_sets],
        dtype=np.float64,
    )


def evaluate_file(
    label: str,
    path: Path,
    *,
    train_observations: int,
    group_size: int,
    page_sizes: tuple[int, ...],
    microblock_sizes: tuple[int, ...],
    neighbor_count: int,
) -> list[dict[str, object]]:
    payload = np.load(path)
    indices = payload["indices"]
    token_count = int(payload["context_token_ids"].shape[0])
    selected_heads = payload["selected_heads"]
    query_head_groups = complete_query_head_groups(selected_heads, group_size)
    if not query_head_groups:
        raise ValueError("shared KV-head evaluation requires at least one complete query-head group")

    aggregate: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    build_seconds: dict[tuple[int, int], float] = defaultdict(float)
    group_observations: dict[tuple[int, int], int] = defaultdict(int)
    for layer_slot in range(indices.shape[0]):
        for group_slots in query_head_groups:
            token_sets = union_query_heads(indices[layer_slot, group_slots])
            training = token_sets[:train_observations]
            test = token_sets[train_observations:]
            selected_counts = np.asarray([len(row) for row in test], dtype=np.float64)
            for page_size in page_sizes:
                chronological = page_counts(test, np.arange(token_count), page_size)
                for microblock_size in microblock_sizes:
                    if page_size % microblock_size != 0:
                        continue
                    started = time.perf_counter()
                    positions = coaccess_physical_positions(
                        training,
                        token_count,
                        page_size=page_size,
                        microblock_size=microblock_size,
                        neighbor_count=neighbor_count,
                    )
                    build_seconds[(page_size, microblock_size)] += time.perf_counter() - started
                    coaccess = page_counts(test, positions, page_size)
                    key = (page_size, microblock_size)
                    aggregate[(page_size, microblock_size, "chronological_pages")].extend(
                        chronological.tolist()
                    )
                    aggregate[(page_size, microblock_size, "coaccess_pages")].extend(
                        coaccess.tolist()
                    )
                    aggregate[(page_size, microblock_size, "selected_tokens")].extend(
                        selected_counts.tolist()
                    )
                    group_observations[key] += len(test)

    rows: list[dict[str, object]] = []
    for page_size in page_sizes:
        for microblock_size in microblock_sizes:
            key = (page_size, microblock_size)
            if (page_size, microblock_size, "coaccess_pages") not in aggregate:
                continue
            chronological = np.asarray(
                aggregate[(page_size, microblock_size, "chronological_pages")]
            )
            coaccess = np.asarray(aggregate[(page_size, microblock_size, "coaccess_pages")])
            selected = np.asarray(aggregate[(page_size, microblock_size, "selected_tokens")])
            rows.append(
                {
                    "dataset": label,
                    "context_tokens": token_count,
                    "layers": int(indices.shape[0]),
                    "kv_heads": len(query_head_groups),
                    "selected_query_heads": selected_heads.tolist(),
                    "complete_query_head_groups": [
                        selected_heads[slots].tolist() for slots in query_head_groups
                    ],
                    "gqa_group_size": group_size,
                    "train_queries": train_observations,
                    "test_queries": int(indices.shape[2] - train_observations),
                    "page_tokens": page_size,
                    "microblock_tokens": microblock_size,
                    "mean_selected_tokens": float(selected.mean()),
                    "chronological_mean_pages": float(chronological.mean()),
                    "coaccess_mean_pages": float(coaccess.mean()),
                    "page_reduction_fraction": float(1.0 - coaccess.mean() / chronological.mean()),
                    "chronological_p90_pages": float(np.quantile(chronological, 0.90)),
                    "coaccess_p90_pages": float(np.quantile(coaccess, 0.90)),
                    "chronological_read_amplification": float(
                        np.mean(chronological * page_size / selected)
                    ),
                    "coaccess_read_amplification": float(
                        np.mean(coaccess * page_size / selected)
                    ),
                    "layout_build_seconds": build_seconds[key],
                    "head_query_observations": group_observations[key],
                    "quality_contract": "exact token IDs unchanged; only physical positions are permuted",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate co-access-aware physical KV page layouts.")
    parser.add_argument("--input", action="append", required=True, help="LABEL=selection_indices.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_observations", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--page_sizes", default="8,16,32")
    parser.add_argument("--microblock_sizes", default="1,4,8")
    parser.add_argument("--neighbor_count", type=int, default=16)
    args = parser.parse_args()

    page_sizes = tuple(int(item) for item in args.page_sizes.split(",") if item)
    microblock_sizes = tuple(int(item) for item in args.microblock_sizes.split(",") if item)
    inputs = [parse_labeled_path(item) for item in args.input]
    rows: list[dict[str, object]] = []
    for label, path in inputs:
        rows.extend(
            evaluate_file(
                label,
                path,
                train_observations=args.train_observations,
                group_size=args.group_size,
                page_sizes=page_sizes,
                microblock_sizes=microblock_sizes,
                neighbor_count=args.neighbor_count,
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
            f"page={row['page_tokens']}",
            f"micro={row['microblock_tokens']}",
            f"pages={row['chronological_mean_pages']:.3f}->{row['coaccess_mean_pages']:.3f}",
            f"reduction={100.0 * row['page_reduction_fraction']:.2f}%",
        )


if __name__ == "__main__":
    main()
