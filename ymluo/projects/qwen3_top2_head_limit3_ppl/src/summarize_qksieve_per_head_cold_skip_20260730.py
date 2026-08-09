#!/usr/bin/env python
"""Merge sharded per-head cold-skip trace experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


GROUP_FIELDS = (
    "hot_fraction",
    "recent_tokens",
    "cold_shards",
    "carry_previous",
)
METRIC_FIELDS = (
    "pool_fraction",
    "attention_mass",
    "baseline_qksieve_mass",
    "mass_retention",
    "oracle_topk_recall",
    "baseline_selection_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def normalize(row: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    output["hot_fraction"] = float(row["hot_fraction"])
    output["recent_tokens"] = int(row["recent_tokens"])
    output["cold_shards"] = int(row["cold_shards"])
    output["carry_previous"] = int(row["carry_previous"])
    for field in METRIC_FIELDS:
        output[field] = float(row[field])
    return output


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "minimum": float(array.min()),
    }


def main() -> None:
    args = parse_args()
    detail_paths = sorted(args.root.glob("*/step_detail.csv"))
    if not detail_paths:
        raise FileNotFoundError(f"no shard detail under {args.root}")
    rows: list[dict[str, Any]] = []
    for path in detail_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(normalize(row) for row in csv.DictReader(handle))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    aggregate: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(GROUP_FIELDS, key))
        item["conditions"] = len(group)
        for field in METRIC_FIELDS:
            summary = summarize_values(
                [float(row[field]) for row in group]
            )
            item[field] = summary["mean"]
            item[f"{field}_p05"] = summary["p05"]
            item[f"{field}_min"] = summary["minimum"]
        item["scan_speedup_upper_bound"] = 1.0 / item["pool_fraction"]
        aggregate.append(item)

    feasible_mean = [
        row for row in aggregate if row["mass_retention"] >= 0.995
    ]
    feasible_strict = [
        row
        for row in feasible_mean
        if row["mass_retention_p05"] >= 0.98
    ]
    for values in (feasible_mean, feasible_strict):
        values.sort(
            key=lambda row: (
                row["pool_fraction"],
                -row["mass_retention"],
            )
        )

    task_summary: dict[str, dict[str, Any]] = {}
    if feasible_mean:
        winner = feasible_mean[0]
        winner_key = tuple(winner[field] for field in GROUP_FIELDS)
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in grouped[winner_key]:
            by_topic[str(row["topic"])].append(row)
        for topic, topic_rows in sorted(by_topic.items()):
            task_summary[topic] = {
                "conditions": len(topic_rows),
                "pool_fraction": summarize_values(
                    [float(row["pool_fraction"]) for row in topic_rows]
                ),
                "mass_retention": summarize_values(
                    [float(row["mass_retention"]) for row in topic_rows]
                ),
            }

    output = {
        "schema": "qksieve_per_head_cold_skip_multimodel_v1",
        "shards": [str(path.parent) for path in detail_paths],
        "conditions_total": len(rows),
        "aggregate": aggregate,
        "best_mean_mass_ge_99_5": (
            feasible_mean[0] if feasible_mean else None
        ),
        "best_mean_ge_99_5_and_p05_ge_98": (
            feasible_strict[0] if feasible_strict else None
        ),
        "best_mean_configuration_by_topic": task_summary,
        "speed_note": (
            "The reciprocal pool fraction is a scan-only upper bound, not "
            "measured CUDA or end-to-end speed."
        ),
    }
    (args.root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.root / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.root / "ALL_COMPLETE").touch()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
