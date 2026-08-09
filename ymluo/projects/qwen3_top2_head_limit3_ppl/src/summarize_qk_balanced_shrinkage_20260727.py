from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "top2_recall_mean",
    "selected_attention_mass_mean",
    "top2_attention_mass_recall_mean",
    "score_pearson_mean",
    "score_rmse_mean",
    "index_ratio_of_full_kv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        shrinkage = float(payload["config"]["query_shrinkage"])
        method = next(
            row
            for row in payload["methods"]
            if row["method"] == "qk_balanced"
            and float(row["selected_fraction_target"]) == 0.01
        )
        rows.append(
            {
                "label": payload["config"]["label"],
                "query_shrinkage": shrinkage,
                **method,
            }
        )
    if not rows:
        raise ValueError("no shrinkage summaries found")
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["query_shrinkage"])].append(row)
    aggregate = []
    for shrinkage, items in sorted(grouped.items()):
        cases = sum(int(row["cases"]) for row in items)
        output: dict[str, Any] = {
            "query_shrinkage": shrinkage,
            "traces": len(items),
            "cases": cases,
        }
        for metric in METRICS:
            output[metric] = sum(
                float(row[metric]) * int(row["cases"]) for row in items
            ) / cases
        output["worst_trace_top2_recall"] = min(
            float(row["top2_recall_mean"]) for row in items
        )
        output["worst_trace_mass_recall"] = min(
            float(row["top2_attention_mass_recall_mean"])
            for row in items
        )
        aggregate.append(output)
    summary = {
        "trace_count": len({str(row["label"]) for row in rows}),
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_trace.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
