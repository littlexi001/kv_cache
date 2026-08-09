from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project exact compiled query-head actions onto physical GQA KV unions."
    )
    parser.add_argument("--distortion_rows", required=True)
    parser.add_argument("--compiled_actions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gqa_group_size", type=int, required=True)
    return parser.parse_args()


def physical_union_summary(
    compiled_rows: Sequence[dict[str, Any]],
    selected_by_action: dict[tuple[int, int, int, int, str], set[int]],
    full_by_head: dict[tuple[int, int, int, int], set[int]],
    gqa_group_size: int,
) -> dict[str, float | int]:
    physical: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    full_physical: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    logical_blocks: list[int] = []
    for row in compiled_rows:
        query_id = int(row["query_id"])
        layer = int(row["layer"])
        query_head = int(row["query_head"])
        query_position = int(row["query_position"])
        action = str(row["chosen_action"])
        logical_blocks.append(int(row["selected_blocks"]))
        action_key = (query_id, layer, query_head, query_position, action)
        head_key = (query_id, layer, query_head, query_position)
        physical_key = (
            query_id,
            layer,
            query_head // gqa_group_size,
            query_position,
        )
        physical[physical_key].update(selected_by_action[action_key])
        full_physical[physical_key].update(full_by_head[head_key])
    selected = np.asarray([len(value) for value in physical.values()], dtype=np.float64)
    full = np.asarray(
        [len(full_physical[key]) for key in physical], dtype=np.float64
    )
    return {
        "decisions": len(compiled_rows),
        "physical_groups": len(physical),
        "mean_logical_blocks": float(np.mean(logical_blocks)),
        "mean_physical_gqa_blocks": float(selected.mean()),
        "mean_full_physical_gqa_blocks": float(full.mean()),
        "physical_gqa_saving_rate": float(1.0 - selected.sum() / full.sum()),
    }


def structured_gqa_oracle_summary(
    action_rows: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    threshold: float,
    gqa_group_size: int,
) -> dict[str, Any]:
    grouped_heads: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for query_id, layer, query_head, query_position in action_rows:
        group_key = (
            query_id,
            layer,
            query_head // gqa_group_size,
            query_position,
        )
        grouped_heads[group_key].append(query_head)

    selected_sizes: list[int] = []
    full_sizes: list[int] = []
    logical_sizes: list[float] = []
    action_counts: Counter[str] = Counter()
    for (query_id, layer, _kv_head, query_position), query_heads in grouped_heads.items():
        # Dynamic programming over the small GQA group. States with an identical
        # physical union retain the lowest aggregate error/logical cost path.
        states: dict[frozenset[int], tuple[float, int, tuple[str, ...]]] = {
            frozenset(): (0.0, 0, ())
        }
        full_union: set[int] = set()
        for query_head in sorted(query_heads):
            candidates = action_rows[(query_id, layer, query_head, query_position)]
            feasible = [
                row
                for row in candidates
                if row["action"] != "mass_oracle_blocks"
                and (
                    row["action"] == "full"
                    or float(row["relative_output_l2"]) <= threshold
                )
            ]
            full_union.update(
                next(row["selected_block_ids"] for row in candidates if row["action"] == "full")
            )
            updated: dict[frozenset[int], tuple[float, int, tuple[str, ...]]] = {}
            for current_union, (error, logical, actions) in states.items():
                for row in feasible:
                    union = frozenset(current_union.union(row["selected_block_ids"]))
                    candidate = (
                        error + float(row["relative_output_l2"]),
                        logical + int(row["selected_blocks"]),
                        (*actions, str(row["action"])),
                    )
                    if union not in updated or candidate < updated[union]:
                        updated[union] = candidate
            states = updated
        best_union, best = min(
            states.items(),
            key=lambda item: (len(item[0]), item[1][0], item[1][1], item[1][2]),
        )
        selected_sizes.append(len(best_union))
        full_sizes.append(len(full_union))
        logical_sizes.append(best[1] / len(query_heads))
        action_counts.update(best[2])
    selected_array = np.asarray(selected_sizes, dtype=np.float64)
    full_array = np.asarray(full_sizes, dtype=np.float64)
    return {
        "physical_groups": len(grouped_heads),
        "mean_logical_blocks": float(np.mean(logical_sizes)),
        "mean_physical_gqa_blocks": float(selected_array.mean()),
        "mean_full_physical_gqa_blocks": float(full_array.mean()),
        "physical_gqa_saving_rate": float(
            1.0 - selected_array.sum() / full_array.sum()
        ),
        "action_counts": dict(sorted(action_counts.items())),
    }


def main() -> None:
    args = parse_args()
    selected_by_action: dict[tuple[int, int, int, int, str], set[int]] = {}
    full_by_head: dict[tuple[int, int, int, int], set[int]] = {}
    action_rows: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    with Path(args.distortion_rows).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            head_key = (
                int(row["query_id"]),
                int(row["layer"]),
                int(row["query_head"]),
                int(row["query_position"]),
            )
            selected = set(int(item) for item in json.loads(row["selected_block_ids"]))
            selected_by_action[(*head_key, str(row["action"]))] = selected
            action_rows[head_key].append(
                {
                    "action": str(row["action"]),
                    "selected_blocks": int(row["selected_blocks"]),
                    "selected_block_ids": selected,
                    "relative_output_l2": float(row["relative_output_l2"]),
                }
            )
            if row["action"] == "full":
                full_by_head[head_key] = selected

    compiled_by_threshold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(args.compiled_actions).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            compiled_by_threshold[str(row["threshold"])].append(row)
    summary = {
        "gqa_group_size": args.gqa_group_size,
        "thresholds": {
            threshold: {
                "independent_head_oracle": physical_union_summary(
                    rows, selected_by_action, full_by_head, args.gqa_group_size
                ),
                "structured_physical_union_oracle": structured_gqa_oracle_summary(
                    action_rows, float(threshold), args.gqa_group_size
                ),
            }
            for threshold, rows in sorted(
                compiled_by_threshold.items(), key=lambda item: float(item[0])
            )
        },
        "note": (
            "The independent oracle minimizes each query head before union. The structured "
            "oracle jointly minimizes the physical union under the same per-head error "
            "constraints. Both are capacity statistics, not learned-router results."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
