#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--min_keep", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--prefix", default="", help="Only print fields whose name starts with this prefix.")
    args = parser.parse_args()

    rows = []
    with (Path(args.input_dir) / "task_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") != args.task:
                continue
            try:
                keep = float(row.get("keep_fraction", "") or 0.0)
            except ValueError:
                keep = 0.0
            if keep >= args.min_keep:
                rows.append(row)
    rows.sort(key=lambda row: float(row.get("online_seconds", "") or 0.0), reverse=True)

    for idx, row in enumerate(rows[: args.limit], start=1):
        print(f"ROW {idx} task={row.get('task')} sample_id={row.get('sample_id')} keep={row.get('keep_fraction')} score={row.get('score')}")
        for key in row:
            if args.prefix and not key.startswith(args.prefix):
                continue
            value = row.get(key, "")
            if value not in {"", "0", "0.0", "-1", "-1.0"}:
                print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    main()
