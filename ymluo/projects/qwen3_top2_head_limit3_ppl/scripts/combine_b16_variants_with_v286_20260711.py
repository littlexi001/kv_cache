#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


QA_TASKS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
}

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
        for row in rows:
            writer.writerow(row)


def value(row: dict[str, Any], key: str) -> float:
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
        out: dict[str, Any] = {
            "benchmark": key[0],
            "task": key[1],
            "method": key[2],
            "samples": count,
            "score": sum(value(row, "score") for row in subset) / max(1, count),
        }
        for field in MEAN_FIELDS:
            out[f"mean_{field}"] = sum(value(row, field) for row in subset) / max(1, count)
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant_label", required=True)
    parser.add_argument("variant_task_dirs", nargs="+")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_csv(base_dir / "task_results.csv")
    rows = [row for row in base_rows if row.get("task") not in QA_TASKS]
    for directory_text in args.variant_task_dirs:
        rows.extend(read_csv(Path(directory_text) / "task_results.csv"))
    rows.sort(key=lambda row: (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""), row.get("method", "")))
    summary = summarize(rows)

    write_csv(output_dir / "task_results.csv", rows)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "base_dir": str(base_dir),
        "variant_label": args.variant_label,
        "variant_task_dirs": [str(path) for path in args.variant_task_dirs],
        "replaced_tasks": sorted(QA_TASKS),
        "examples": len(rows),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
