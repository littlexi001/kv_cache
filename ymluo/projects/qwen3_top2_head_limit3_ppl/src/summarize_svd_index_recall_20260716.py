from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_svd_index_recall_20260716 import summarize, write_csv


METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "oracle_top2_attention_mass",
    "top2_attention_mass_recall",
)


def read_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(glob.glob(pattern)):
        with Path(name).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no rows matched {pattern}")
    return rows


def aggregate(
    rows: list[dict[str, Any]], group_fields: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        result: dict[str, Any] = dict(zip(group_fields, key))
        result["cases"] = len(items)
        for metric in METRICS:
            stats = summarize([float(item[metric]) for item in items])
            result.update({f"{metric}_{name}": value for name, value in stats.items()})
        output.append(result)
    return output


def add_paired_deltas(rows: list[dict[str, Any]]) -> None:
    baseline: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["scheme"] == "current_sampled_pca":
            key = (
                row["topic"],
                row["record_index"],
                row["layer"],
                row["query_head"],
                row["rank"] + ":" + row["precision"],
            )
            baseline[key] = row
    for row in rows:
        key = (
            row["topic"],
            row["record_index"],
            row["layer"],
            row["query_head"],
            row["rank"] + ":" + row["precision"],
        )
        reference = baseline[key]
        row["top2_recall_delta_vs_current"] = float(row["top2_recall"]) - float(
            reference["top2_recall"]
        )
        row["mass_recall_delta_vs_current"] = float(
            row["top2_attention_mass_recall"]
        ) - float(reference["top2_attention_mass_recall"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    rows = read_rows(args.input_glob)
    add_paired_deltas(rows)
    overall = aggregate(rows, ["scheme", "rank", "precision"])
    by_topic = aggregate(rows, ["topic", "scheme", "rank", "precision"])
    by_layer = aggregate(rows, ["layer", "scheme", "rank", "precision"])

    delta_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        delta_groups[(row["scheme"], row["rank"], row["precision"])].append(row)
    paired = []
    for key, items in sorted(delta_groups.items()):
        paired.append(
            {
                "scheme": key[0],
                "rank": key[1],
                "precision": key[2],
                "cases": len(items),
                "top2_recall_delta_vs_current_mean": sum(
                    float(item["top2_recall_delta_vs_current"]) for item in items
                )
                / len(items),
                "mass_recall_delta_vs_current_mean": sum(
                    float(item["mass_recall_delta_vs_current"]) for item in items
                )
                / len(items),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_overall.csv", overall, list(overall[0]))
    write_csv(args.output_dir / "summary_by_topic.csv", by_topic, list(by_topic[0]))
    write_csv(args.output_dir / "summary_by_layer.csv", by_layer, list(by_layer[0]))
    write_csv(args.output_dir / "paired_delta_vs_current.csv", paired, list(paired[0]))
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"cases": len(rows), "overall": overall, "paired_deltas": paired},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(rows), "paired_deltas": paired}, sort_keys=True))


if __name__ == "__main__":
    main()
