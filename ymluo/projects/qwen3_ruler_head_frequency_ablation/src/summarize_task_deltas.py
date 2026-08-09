from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run.glob("shard*/rows.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_run"] = str(run)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--baseline", default="native_rope")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [row for run in args.run for row in read_rows(run)]
    by_case = {
        (str(row["_run"]), str(row["sample_id"]), str(row["variant"])): row
        for row in rows
    }
    variants = sorted({str(row["variant"]) for row in rows if row["variant"] != args.baseline})
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (run, sample_id, variant), row in by_case.items():
        if variant == args.baseline:
            continue
        native = by_case.get((run, sample_id, args.baseline))
        if native is None:
            continue
        grouped[(variant, str(row["task"]))].append(
            (
                float(row["official_score"]) - float(native["official_score"]),
                float(native["gold_answer_mean_nll"]) - float(row["gold_answer_mean_nll"]),
            )
        )

    output: list[dict[str, Any]] = []
    for variant in variants:
        for task in sorted({key[1] for key in grouped if key[0] == variant}):
            values = grouped[(variant, task)]
            output.append(
                {
                    "variant": variant,
                    "task": task,
                    "samples": len(values),
                    "official_delta": mean(value[0] for value in values),
                    "nll_improvement": mean(value[1] for value in values),
                    "nll_improved": sum(value[1] > 0 for value in values),
                    "nll_degraded": sum(value[1] < 0 for value in values),
                }
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
