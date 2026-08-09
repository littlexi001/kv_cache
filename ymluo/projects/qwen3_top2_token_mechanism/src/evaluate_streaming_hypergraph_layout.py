from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_coaccess_kv_layout import (
    complete_query_head_groups,
    page_counts,
    union_query_heads,
)
from evaluate_coselection_expansion import parse_labeled_path


def streaming_hypergraph_positions(
    training_token_sets: list[np.ndarray], token_count: int
) -> np.ndarray:
    """Pack by most recent prefill co-access in linear incidence complexity."""
    frequency = np.zeros(token_count, dtype=np.int32)
    for tokens in training_token_sets:
        frequency[tokens] += 1

    assigned = np.zeros(token_count, dtype=bool)
    physical_order: list[int] = []
    for tokens in reversed(training_token_sets):
        unassigned = tokens[~assigned[tokens]]
        if unassigned.size == 0:
            continue
        order = np.lexsort((unassigned, -frequency[unassigned]))
        selected = unassigned[order]
        assigned[selected] = True
        physical_order.extend(selected.tolist())
    physical_order.extend(np.flatnonzero(~assigned).tolist())

    if len(physical_order) != token_count or len(set(physical_order)) != token_count:
        raise RuntimeError("streaming hypergraph packing did not produce a permutation")
    positions = np.empty(token_count, dtype=np.int32)
    positions[np.asarray(physical_order, dtype=np.int32)] = np.arange(
        token_count, dtype=np.int32
    )
    return positions


def evaluate_file(
    label: str,
    path: Path,
    *,
    train_observations: int,
    group_size: int,
    page_sizes: tuple[int, ...],
) -> list[dict[str, object]]:
    payload = np.load(path)
    indices = payload["indices"]
    token_count = int(payload["context_token_ids"].shape[0])
    selected_heads = payload["selected_heads"]
    query_head_groups = complete_query_head_groups(selected_heads, group_size)
    if not query_head_groups:
        raise ValueError("shared KV-head evaluation requires at least one complete query-head group")

    aggregate: dict[tuple[int, str], list[float]] = defaultdict(list)
    build_seconds = 0.0
    observations = 0
    for layer_slot in range(indices.shape[0]):
        for group_slots in query_head_groups:
            token_sets = union_query_heads(indices[layer_slot, group_slots])
            training = token_sets[:train_observations]
            test = token_sets[train_observations:]
            started = time.perf_counter()
            positions = streaming_hypergraph_positions(training, token_count)
            build_seconds += time.perf_counter() - started
            selected = np.asarray([len(tokens) for tokens in test], dtype=np.float64)
            for page_size in page_sizes:
                chronological = page_counts(test, np.arange(token_count), page_size)
                packed = page_counts(test, positions, page_size)
                aggregate[(page_size, "chronological")].extend(chronological.tolist())
                aggregate[(page_size, "packed")].extend(packed.tolist())
                aggregate[(page_size, "selected")].extend(selected.tolist())
            observations += len(test)

    rows: list[dict[str, object]] = []
    for page_size in page_sizes:
        chronological = np.asarray(aggregate[(page_size, "chronological")])
        packed = np.asarray(aggregate[(page_size, "packed")])
        selected = np.asarray(aggregate[(page_size, "selected")])
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
                "prefill_probe_queries": train_observations,
                "heldout_decode_queries": int(indices.shape[2] - train_observations),
                "page_tokens": page_size,
                "mean_selected_tokens": float(selected.mean()),
                "chronological_mean_pages": float(chronological.mean()),
                "packed_mean_pages": float(packed.mean()),
                "page_reduction_fraction": float(1.0 - packed.mean() / chronological.mean()),
                "ideal_page_read_speedup": float(chronological.mean() / packed.mean()),
                "chronological_p90_pages": float(np.quantile(chronological, 0.9)),
                "packed_p90_pages": float(np.quantile(packed, 0.9)),
                "layout_build_seconds_python_all_groups": build_seconds,
                "head_query_observations": observations,
                "build_complexity": "O(prefill probes * selected tokens + context tokens)",
                "test_contract": "prompt-tail probes only; held-out continuation is never used",
                "quality_contract": "exact token IDs unchanged; only physical positions are permuted",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate streaming prefill-hypergraph KV packing.")
    parser.add_argument("--input", action="append", required=True, help="LABEL=selection_indices.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_observations", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--page_sizes", default="8,16,32")
    args = parser.parse_args()

    page_sizes = tuple(int(value) for value in args.page_sizes.split(",") if value)
    rows: list[dict[str, object]] = []
    for label, path in (parse_labeled_path(value) for value in args.input):
        rows.extend(
            evaluate_file(
                label,
                path,
                train_observations=args.train_observations,
                group_size=args.group_size,
                page_sizes=page_sizes,
            )
        )
    report = {"method": "streaming prompt-tail hypergraph packing", "results": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(
            row["dataset"],
            f"page={row['page_tokens']}",
            f"pages={row['chronological_mean_pages']:.3f}->{row['packed_mean_pages']:.3f}",
            f"reduction={100.0 * row['page_reduction_fraction']:.2f}%",
            f"build={row['layout_build_seconds_python_all_groups']:.3f}s",
        )


if __name__ == "__main__":
    main()
