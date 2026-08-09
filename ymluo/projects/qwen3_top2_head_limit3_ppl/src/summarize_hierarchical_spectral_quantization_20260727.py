from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


QUALITY_FIELDS = (
    "top2_recall_mean",
    "selected_attention_mass_mean",
    "top2_attention_mass_recall_mean",
    "score_pearson_mean",
    "candidate_top2_recall_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-trace hierarchical spectral quantization summaries and "
            "identify quality/access-cost Pareto configurations."
        )
    )
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--expected_labels",
        default="",
        help="Optional comma-separated labels required before aggregation.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["top2_recall_weighted"] >= right["top2_recall_weighted"]
        and left["top2_attention_mass_recall_weighted"]
        >= right["top2_attention_mass_recall_weighted"]
        and left["mean_scan_bits_per_history_token"]
        <= right["mean_scan_bits_per_history_token"]
    )
    strictly_better = (
        left["top2_recall_weighted"] > right["top2_recall_weighted"]
        or left["top2_attention_mass_recall_weighted"]
        > right["top2_attention_mass_recall_weighted"]
        or left["mean_scan_bits_per_history_token"]
        < right["mean_scan_bits_per_history_token"]
    )
    return no_worse and strictly_better


def main() -> None:
    args = parse_args()
    expected = {
        value.strip() for value in args.expected_labels.split(",") if value.strip()
    }
    summary_paths = sorted(args.input_root.glob("*/summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"no */summary.csv under {args.input_root}")

    by_configuration: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    labels: set[str] = set()
    for path in summary_paths:
        label = path.parent.name
        labels.add(label)
        for raw in read_csv(path):
            row: dict[str, Any] = dict(raw)
            row["label"] = label
            row["cases"] = int(raw["cases"])
            row["selected_fraction_target"] = float(
                raw["selected_fraction_target"]
            )
            for field in (
                "logical_index_bits_per_token",
                "logical_index_ratio_of_full_kv",
                "mean_scan_bits_per_history_token",
                "mean_scan_ratio_of_full_kv",
                *QUALITY_FIELDS,
            ):
                row[field] = float(raw[field])
            key = (raw["method"], row["selected_fraction_target"])
            by_configuration[key].append(row)

    missing = sorted(expected - labels)
    if missing:
        raise RuntimeError(f"missing expected labels: {missing}")

    combined: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for (method, selected_fraction), items in sorted(by_configuration.items()):
        total_cases = sum(item["cases"] for item in items)
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": selected_fraction,
            "datasets": len(items),
            "cases": total_cases,
            "logical_index_bits_per_token": items[0][
                "logical_index_bits_per_token"
            ],
            "logical_index_ratio_of_full_kv": items[0][
                "logical_index_ratio_of_full_kv"
            ],
            "mean_scan_bits_per_history_token": items[0][
                "mean_scan_bits_per_history_token"
            ],
            "mean_scan_ratio_of_full_kv": items[0][
                "mean_scan_ratio_of_full_kv"
            ],
        }
        for field in QUALITY_FIELDS:
            prefix = field.removesuffix("_mean")
            weighted = (
                sum(item[field] * item["cases"] for item in items) / total_cases
            )
            macro = sum(item[field] for item in items) / len(items)
            worst = min(items, key=lambda item: item[field])
            result[f"{prefix}_weighted"] = weighted
            result[f"{prefix}_macro"] = macro
            result[f"{prefix}_worst"] = worst[field]
            result[f"{prefix}_worst_label"] = worst["label"]
        combined.append(result)
        for item in items:
            details.append(
                {
                    "method": method,
                    "selected_fraction_target": selected_fraction,
                    "label": item["label"],
                    "cases": item["cases"],
                    **{field: item[field] for field in QUALITY_FIELDS},
                }
            )

    for row in combined:
        comparable = [
            other
            for other in combined
            if other["selected_fraction_target"]
            == row["selected_fraction_target"]
        ]
        row["quality_scan_pareto"] = not any(
            dominates(other, row) for other in comparable if other is not row
        )

    pareto = [row for row in combined if row["quality_scan_pareto"]]
    combined.sort(
        key=lambda row: (
            row["selected_fraction_target"],
            -row["top2_attention_mass_recall_weighted"],
            -row["top2_recall_weighted"],
            row["mean_scan_bits_per_history_token"],
        )
    )
    pareto.sort(
        key=lambda row: (
            row["selected_fraction_target"],
            row["mean_scan_bits_per_history_token"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_summary.csv", combined)
    write_csv(args.output_dir / "per_dataset_summary.csv", details)
    write_csv(args.output_dir / "quality_scan_pareto.csv", pareto)
    payload = {
        "input_root": str(args.input_root),
        "labels": sorted(labels),
        "summary_files": [str(path) for path in summary_paths],
        "configurations": len(combined),
        "pareto_configurations": len(pareto),
        "combined": combined,
    }
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "labels": sorted(labels),
                "configurations": len(combined),
                "pareto_configurations": len(pareto),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
