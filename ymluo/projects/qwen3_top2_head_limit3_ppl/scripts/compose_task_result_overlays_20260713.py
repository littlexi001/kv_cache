#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import combine_split_task_results_20260711 as combine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--overlay", nargs="*", default=[])
    args = parser.parse_args()

    indexed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for source in [*args.base, *args.overlay]:
        path = Path(source)
        if path.is_dir():
            path = path / "task_results.csv"
        for row in combine.read_csv(path):
            key = (row["benchmark"], row["task"], row["sample_id"], row["method"])
            indexed[key] = row

    rows = [indexed[key] for key in sorted(indexed)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combine.write_csv(output_dir / "task_results.csv", rows)
    summary = combine.summarize(rows)
    combine.write_csv(output_dir / "summary.csv", summary, combine.SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "base": args.base,
        "overlay": args.overlay,
        "examples": len(rows),
        "overlay_precedence": "last source wins by benchmark/task/sample/method",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
