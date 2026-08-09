from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


METRIC_FIELDS = (
    "top2_recall",
    "selected_attention_mass",
    "oracle_top2_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_head_rows: list[dict[str, str]] = []
    allocation_rows: list[dict[str, str]] = []
    case_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for summary_path in sorted(args.input_root.glob("*/summary.json")):
        case = summary_path.parent.name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        case_summaries[case] = {
            str(row["method"]): row for row in payload["methods"]
        }
        per_head_rows.extend(read_csv(summary_path.parent / "per_head.csv"))
        allocation_rows.extend(
            read_csv(summary_path.parent / "allocations.csv")
        )
    if not per_head_rows:
        raise ValueError(f"no completed cases under {args.input_root}")

    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in per_head_rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    allocations_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in allocation_rows:
        allocations_by_method[str(row["method"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (method, fraction), rows in sorted(grouped.items()):
        allocations = allocations_by_method[method]
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(rows),
            "traces": len(
                {str(row["label"]) for row in rows}
            ),
            "total_index_bits_mean": sum(
                int(row["total_index_bits"]) for row in allocations
            )
            / len(allocations),
            "index_ratio_of_full_kv": sum(
                float(row["index_ratio_of_full_kv"])
                for row in allocations
            )
            / len(allocations),
            "shared_envelope_layer_head_fraction": sum(
                str(row["uses_shared_envelope"]).lower() == "true"
                for row in allocations
            )
            / len(allocations),
        }
        for field in METRIC_FIELDS:
            for statistic, value in summarize(
                float(row[field]) for row in rows
            ).items():
                result[f"{field}_{statistic}"] = value
        trace_values = []
        for case, methods in case_summaries.items():
            row = methods.get(method)
            if row is None:
                continue
            trace_values.append(
                {
                    "case": case,
                    "top2_attention_mass_recall_mean": float(
                        row["top2_attention_mass_recall_mean"]
                    ),
                }
            )
        result["worst_trace"] = min(
            trace_values,
            key=lambda row: row["top2_attention_mass_recall_mean"],
        )["case"]
        result["worst_trace_top2_attention_mass_recall"] = min(
            row["top2_attention_mass_recall_mean"]
            for row in trace_values
        )
        aggregate_rows.append(result)

    output = {
        "input_root": str(args.input_root),
        "completed_traces": sorted(case_summaries),
        "methods": aggregate_rows,
        "allocation_histograms": {
            method: dict(
                Counter(
                    str(row["allocation"])
                    for row in allocations
                ).most_common()
            )
            for method, allocations in sorted(
                allocations_by_method.items()
            )
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
