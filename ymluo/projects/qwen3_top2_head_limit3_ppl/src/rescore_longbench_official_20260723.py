from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_sample_calibrated_longbench_20260717 as runner


# Classification rows were already first-line scored online and the compact CSV
# does not retain all_classes. Only these two non-classification tasks need a
# post-hoc correction.
FIRST_LINE_TASKS = {"triviaqa", "samsum"}


def stored_prediction(row: dict[str, Any]) -> str:
    prediction = str(row.get("prediction", ""))
    if str(row.get("task", "")) in FIRST_LINE_TASKS:
        # Harness CSVs escape generated newlines before truncating the stored
        # prediction. Only the first line is needed for these official metrics.
        return prediction.split("\\n", 1)[0].lstrip("\\n")
    return prediction


def rescore_row(row: dict[str, Any]) -> dict[str, Any]:
    task = str(row["task"])
    if task not in FIRST_LINE_TASKS:
        return dict(row)
    updated = dict(row)
    updated["score"] = runner.lb.score_prediction(
        str(row["metric"]),
        stored_prediction(row),
        [str(item) for item in json.loads(str(row["answers"]))],
        task=task,
    )
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score escaped CountCap LongBench CSVs with official task rules."
    )
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(args.input_glob)

    source_rows: list[dict[str, Any]] = []
    for path in paths:
        source_rows.extend(runner.read_csv(path))
    if args.expected_rows and len(source_rows) != args.expected_rows:
        raise RuntimeError(
            f"expected {args.expected_rows} rows, found {len(source_rows)}"
        )

    keys = [
        (str(row["task"]), str(row["sample_id"]), str(row["method"]))
        for row in source_rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate task/sample/method rows found")

    rows = [rescore_row(row) for row in source_rows]
    changed = [
        (source, updated)
        for source, updated in zip(source_rows, rows)
        if float(source["score"]) != float(updated["score"])
    ]
    summary = runner.summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner.write_csv(args.output_dir / "sample_results.csv", rows)
    runner.write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit = {
        "input_files": [str(path) for path in paths],
        "rows": len(rows),
        "changed_rows": len(changed),
        "task_counts": Counter(str(row["task"]) for row in rows),
        "method_counts": Counter(str(row["method"]) for row in rows),
    }
    (args.output_dir / "rescore_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"officially re-scored {len(rows)} rows; changed {len(changed)} rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
