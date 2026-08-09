#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MEAN_FIELDS = [
    "total_seconds",
    "online_seconds",
    "prefill_seconds",
    "kv_gather_seconds",
    "query_seconds",
    "decode_seconds",
    "raw_prefix_tokens",
    "kept_prefix_tokens",
    "kept_context_tokens",
    "keep_fraction",
]

SUMMARY_FIELDS = [
    "benchmark",
    "task",
    "method",
    "samples",
    "score",
    "mean_total_seconds",
    "mean_online_seconds",
    "mean_prefill_seconds",
    "mean_kv_gather_seconds",
    "mean_query_seconds",
    "mean_decode_seconds",
    "mean_raw_prefix_tokens",
    "mean_kept_prefix_tokens",
    "mean_kept_context_tokens",
    "mean_keep_fraction",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    return 0.0 if raw == "" else float(raw)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        benchmark = str(row["benchmark"])
        task = str(row["task"])
        method = str(row["method"])
        for key in [
            ("ALL", "ALL", method),
            (benchmark, "ALL", method),
            (benchmark, task, method),
        ]:
            groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0] != "ALL", item[0], item[1] != "ALL", item[1], item[2])):
        subset = groups[key]
        count = len(subset)
        row: dict[str, Any] = {
            "benchmark": key[0],
            "task": key[1],
            "method": key[2],
            "samples": count,
            "score": sum(as_float(item, "score") for item in subset) / max(1, count),
        }
        for field in MEAN_FIELDS:
            row[f"mean_{field}"] = sum(as_float(item, field) for item in subset) / max(1, count)
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--replace_tasks", required=True, help="Comma-separated task names to replace from base_dir.")
    parser.add_argument("--label", default="replace_tasks")
    parser.add_argument(
        "--ignore_extra_replacement_tasks",
        action="store_true",
        help="Filter replacement rows to --replace_tasks when a replacement dir contains additional tasks.",
    )
    parser.add_argument("replacement_dirs", nargs="+")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replace_tasks = {item.strip() for item in args.replace_tasks.split(",") if item.strip()}
    if not replace_tasks:
        raise ValueError("--replace_tasks must name at least one task")

    base_rows = read_csv(base_dir / "task_results.csv")
    rows = [row for row in base_rows if row.get("task") not in replace_tasks]
    replacement_rows: list[dict[str, Any]] = []
    for directory_text in args.replacement_dirs:
        replacement_rows.extend(read_csv(Path(directory_text) / "task_results.csv"))
    unexpected = sorted({row.get("task", "") for row in replacement_rows} - replace_tasks)
    if unexpected:
        if not args.ignore_extra_replacement_tasks:
            raise ValueError(f"replacement dirs contain unexpected tasks: {unexpected}")
        replacement_rows = [row for row in replacement_rows if row.get("task") in replace_tasks]
    rows.extend(replacement_rows)
    rows.sort(key=lambda row: (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""), row.get("method", "")))

    summary = summarize(rows)
    write_csv(output_dir / "task_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "base_dir": str(base_dir),
        "label": args.label,
        "replace_tasks": sorted(replace_tasks),
        "replacement_dirs": [str(path) for path in args.replacement_dirs],
        "examples": len(rows),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
