from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_sample_calibrated_longbench_20260717 as longbench
import run_sample_calibrated_ruler_20260717 as ruler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("longbench", "ruler"), required=True)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    import glob

    paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(args.input_glob)
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(longbench.read_csv(path))
    if not rows:
        raise RuntimeError("no sample rows found")

    summary = (
        longbench.summarize(rows)
        if args.kind == "longbench"
        else ruler.summarize(rows)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    longbench.write_csv(args.output_dir / "sample_results.csv", rows)
    longbench.write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"merged {len(rows)} rows from {len(paths)} shards into {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
