#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
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
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


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
        n = len(subset)
        out: dict[str, Any] = {
            "benchmark": key[0],
            "task": key[1],
            "method": key[2],
            "samples": n,
            "score": sum(f(row, "score") for row in subset) / max(1, n),
        }
        for field in MEAN_FIELDS:
            out[f"mean_{field}"] = sum(f(row, field) for row in subset) / max(1, n)
        summary.append(out)
    return summary


def wait_for_inputs(input_dirs: list[Path], interval_seconds: int) -> None:
    while True:
        missing = [path / "task_results.csv" for path in input_dirs if not (path / "task_results.csv").exists()]
        if not missing:
            return
        print("WAIT_SPLIT_RESULTS " + " ".join(str(path) for path in missing), flush=True)
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait_interval_seconds", type=int, default=120)
    parser.add_argument("input_dirs", nargs="+")
    args = parser.parse_args()

    input_dirs = [Path(item) for item in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait:
        wait_for_inputs(input_dirs, args.wait_interval_seconds)

    rows: list[dict[str, Any]] = []
    for directory in input_dirs:
        rows.extend(read_csv(directory / "task_results.csv"))
    rows.sort(key=lambda row: (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""), row.get("method", "")))
    summary = summarize(rows)

    write_csv(output_dir / "task_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "combined_from": [str(path) for path in input_dirs],
        "examples": len(rows),
        "labeling": "combined_split_task_results",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
