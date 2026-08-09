from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-quota, equal-budget portfolios from complementary KV operators."
    )
    parser.add_argument(
        "--action",
        action="append",
        required=True,
        help="Action as alias=mode=/path/to/query_results.csv.",
    )
    parser.add_argument(
        "--portfolio",
        action="append",
        required=True,
        help="Portfolio as name=alias:quota,alias:quota.",
    )
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_action(spec: str) -> tuple[str, str, Path]:
    pieces = spec.split("=", 2)
    if len(pieces) != 3:
        raise ValueError("action must be alias=mode=/path/to/query_results.csv")
    return pieces[0], pieces[1], Path(pieces[2])


def load_action(spec: str) -> tuple[str, dict[int, dict[str, Any]]]:
    alias, mode, path = parse_action(spec)
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["method"] != mode:
                continue
            query_id = int(raw["query_id"])
            rows[query_id] = {
                "dataset": raw["dataset"],
                "ranked": [int(item) for item in json.loads(raw["ranked_block_ids"])],
            }
    if not rows:
        raise ValueError(f"no rows for {mode} in {path}")
    return alias, rows


def parse_portfolio(spec: str) -> tuple[str, list[tuple[str, int]]]:
    if "=" not in spec:
        raise ValueError("portfolio must be name=alias:quota,alias:quota")
    name, raw_components = spec.split("=", 1)
    components: list[tuple[str, int]] = []
    for raw_component in raw_components.split(","):
        alias, raw_quota = raw_component.rsplit(":", 1)
        components.append((alias, int(raw_quota)))
    if not name or not components:
        raise ValueError("portfolio name and components must be non-empty")
    return name, components


def allocate_portfolio(
    rankings: dict[str, Sequence[int]],
    components: Sequence[tuple[str, int]],
    target_blocks: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()

    def extend(source: Iterable[int], count: int) -> None:
        added = 0
        for raw_block_id in source:
            if len(selected) >= target_blocks or added >= count:
                break
            block_id = int(raw_block_id)
            if block_id not in seen:
                selected.append(block_id)
                seen.add(block_id)
                added += 1

    for alias, quota in components:
        extend(rankings[alias], quota)
    # Deduplication can leave capacity; cycle through all operators in priority order.
    while len(selected) < target_blocks:
        previous = len(selected)
        for alias, _ in components:
            extend(rankings[alias], target_blocks - len(selected))
        if len(selected) == previous:
            break
    return selected


def build_block_to_record(records: Sequence[dict[str, Any]]) -> np.ndarray:
    num_blocks = max(
        int(record["block_start"]) + int(record["block_count"]) for record in records
    )
    mapping = np.full(num_blocks, -1, dtype=np.int32)
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        mapping[start : start + int(record["block_count"])] = record_id
    if np.any(mapping < 0):
        raise ValueError("records do not cover all blocks")
    return mapping


def group_for_context(ranked: Sequence[int], block_to_record: np.ndarray) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    order: list[int] = []
    for raw_block_id in ranked:
        block_id = int(raw_block_id)
        record_id = int(block_to_record[block_id])
        if record_id not in groups:
            order.append(record_id)
        groups[record_id].append(block_id)
    return [block_id for record_id in order for block_id in sorted(groups[record_id])]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = dict(load_action(spec) for spec in args.action)
    query_sets = [set(rows) for rows in actions.values()]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("actions do not cover identical query IDs")
    query_ids = sorted(query_sets[0])
    portfolios = [parse_portfolio(spec) for spec in args.portfolio]
    for name, components in portfolios:
        unknown = [alias for alias, _ in components if alias not in actions]
        if unknown:
            raise ValueError(f"portfolio {name} has unknown actions {unknown}")
        if sum(quota for _, quota in components) > args.target_blocks:
            raise ValueError(f"portfolio {name} quotas exceed target budget")
    block_to_record = build_block_to_record(read_jsonl(Path(args.records_jsonl)))

    output_rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        datasets = {rows[query_id]["dataset"] for rows in actions.values()}
        if len(datasets) != 1:
            raise ValueError(f"dataset disagreement for query {query_id}")
        rankings = {alias: rows[query_id]["ranked"] for alias, rows in actions.items()}
        for name, components in portfolios:
            ranked = allocate_portfolio(rankings, components, args.target_blocks)
            if len(ranked) != args.target_blocks:
                raise ValueError(f"portfolio {name} produced only {len(ranked)} blocks")
            output_rows.append(
                {
                    "method": name,
                    "query_id": query_id,
                    "dataset": next(iter(datasets)),
                    "source_record_recall": 0.0,
                    "record_top1_recall": 0.0,
                    "answer_block_recall": 0.0,
                    "answer_block_mrr": 0.0,
                    "gold_block_count": 0,
                    "record_margin": 0.0,
                    "selected_block_ids": json.dumps(
                        group_for_context(ranked, block_to_record)
                    ),
                    "ranked_block_ids": json.dumps(ranked),
                }
            )
    output_rows.sort(key=lambda row: (row["method"], int(row["query_id"])))
    with (output_dir / "query_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "source": "fixed-quota equal-budget operator portfolios",
        "queries": len(query_ids),
        "target_blocks": args.target_blocks,
        "actions": sorted(actions),
        "portfolios": [
            {"method": name, "components": components} for name, components in portfolios
        ],
        "note": "No answer, NLL, dataset identity, or gold-block feature is used to build a query selection.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

