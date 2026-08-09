from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    metric_rows: list[dict[str, str]] = []
    allocation_rows: list[dict[str, str]] = []
    trace_means: dict[tuple[str, str], dict[str, float]] = {}
    traces = []
    for summary_path in sorted(args.input_root.glob("*/summary.json")):
        trace = summary_path.parent.name
        traces.append(trace)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload["methods"]:
            trace_means[(trace, str(row["method"]))] = {
                metric: float(row[f"{metric}_mean"]) for metric in METRICS
            }
        metric_rows.extend(read_csv(summary_path.parent / "per_head.csv"))
        allocation_rows.extend(
            read_csv(summary_path.parent / "allocations.csv")
        )
    if not metric_rows:
        raise ValueError(f"no completed trace outputs under {args.input_root}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["method"])].append(row)
    allocations: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in allocation_rows:
        allocations[str(row["method"])].append(row)

    methods: list[dict[str, Any]] = []
    for method, rows in sorted(grouped.items()):
        method_allocations = allocations[method]
        trace_metric_rows = [
            (trace, trace_means[(trace, method)])
            for trace in traces
            if (trace, method) in trace_means
        ]
        result: dict[str, Any] = {
            "method": method,
            "cases": len(rows),
            "traces": len(trace_metric_rows),
            "total_index_bits_mean": sum(
                float(row["total_index_bits"])
                for row in method_allocations
            )
            / len(method_allocations),
            "allocation_histogram": dict(
                Counter(
                    str(row["allocation"]) for row in method_allocations
                ).most_common()
            ),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = sum(values) / len(values)
            worst_trace, worst_values = min(
                trace_metric_rows,
                key=lambda item: (
                    item[1][metric]
                    if metric != "score_rmse"
                    else -item[1][metric]
                ),
            )
            result[f"{metric}_worst_trace"] = worst_trace
            result[f"{metric}_worst_trace_mean"] = worst_values[metric]
        methods.append(result)

    output = {
        "input_root": str(args.input_root),
        "completed_traces": traces,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
