#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmark = str(row.get("benchmark", ""))
        task = str(row.get("task", ""))
        method = str(row.get("method", ""))
        grouped[(benchmark, task, method)].append(row)
        grouped[(benchmark, "ALL", method)].append(row)
        grouped[("ALL", "ALL", method)].append(row)
    out = []
    for (benchmark, task, method), subset in sorted(grouped.items()):
        n = max(1, len(subset))
        out.append(
            {
                "benchmark": benchmark,
                "task": task,
                "method": method,
                "samples": len(subset),
                "score": sum(fnum(row.get("score")) for row in subset) / n,
                "mean_total_seconds": sum(fnum(row.get("total_seconds")) for row in subset) / n,
                "mean_online_seconds": sum(fnum(row.get("online_seconds")) for row in subset) / n,
                "mean_prefill_seconds": sum(fnum(row.get("prefill_seconds")) for row in subset) / n,
                "mean_kv_gather_seconds": sum(fnum(row.get("kv_gather_seconds")) for row in subset) / n,
                "mean_query_seconds": sum(fnum(row.get("query_seconds")) for row in subset) / n,
                "mean_decode_seconds": sum(fnum(row.get("decode_seconds")) for row in subset) / n,
                "mean_raw_prefix_tokens": sum(fnum(row.get("raw_prefix_tokens")) for row in subset) / n,
                "mean_kept_prefix_tokens": sum(fnum(row.get("kept_prefix_tokens")) for row in subset) / n,
                "mean_kept_context_tokens": sum(fnum(row.get("kept_context_tokens")) for row in subset) / n,
                "mean_keep_fraction": sum(fnum(row.get("keep_fraction")) for row in subset) / n,
            }
        )
    return out


def parse_override(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Override must be task=path, got {spec!r}")
    task, path = spec.split("=", 1)
    task = task.strip()
    if not task:
        raise ValueError(f"Override task is empty in {spec!r}")
    return task, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base experiment output directory.")
    parser.add_argument("--output_dir", required=True, help="Stitched output directory.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override task results as task=/path/to/experiment. Can be repeated.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(base / "task_results.csv")
    override_tasks: dict[str, list[dict[str, str]]] = {}
    manifest = {"base": str(base), "overrides": {}}
    for spec in args.override:
        task, path = parse_override(spec)
        task_rows = [row for row in read_csv(path / "task_results.csv") if row.get("task") == task]
        if not task_rows:
            raise ValueError(f"No rows found for task={task!r} in {path}")
        override_tasks[task] = task_rows
        manifest["overrides"][task] = str(path)

    stitched_rows = [row for row in rows if row.get("task") not in override_tasks]
    for task_rows in override_tasks.values():
        stitched_rows.extend(task_rows)
    stitched_rows.sort(key=lambda row: (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""), row.get("method", "")))

    write_csv(output_dir / "task_results.csv", stitched_rows)
    write_csv(output_dir / "summary.csv", summarize(stitched_rows))
    (output_dir / "stitch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
