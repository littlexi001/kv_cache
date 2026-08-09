#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--domains", default="")
    parser.add_argument("--parts", required=True, type=int)
    args = parser.parse_args()

    if args.parts <= 0:
        raise ValueError("--parts must be positive")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    domains = {item.strip() for item in args.domains.split(",") if item.strip()}
    selected = [row for row in rows if not domains or str(row.get("domain", "")) in domains]
    selected.sort(key=lambda row: str(row.get("_id", "")))
    partitions = [selected[index :: args.parts] for index in range(args.parts)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    for index, partition in enumerate(partitions, start=1):
        path = args.output_dir / f"part{index}.json"
        path.write_text(json.dumps(partition, indent=2), encoding="utf-8")
        part_ids = {str(row.get("_id", "")) for row in partition}
        if ids & part_ids:
            raise RuntimeError("partition ID overlap")
        ids.update(part_ids)
        print(f"{path}: {len(partition)}")
    if len(ids) != len(selected):
        raise RuntimeError(f"partition coverage mismatch: {len(ids)} != {len(selected)}")


if __name__ == "__main__":
    main()
