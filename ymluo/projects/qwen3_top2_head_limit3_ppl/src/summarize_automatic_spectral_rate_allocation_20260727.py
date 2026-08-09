from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "top2_recall_mean",
    "selected_attention_mass_mean",
    "top2_attention_mass_recall_mean",
    "score_pearson_mean",
)


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


def weighted_mean(values: Iterable[tuple[float, int]]) -> float:
    pairs = list(values)
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / max(1, total_weight)


def scope_for_label(label: str) -> str:
    return "96k" if "96k" in label.lower() else "32k"


def aggregate_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scopes = (str(row["scope"]), "all")
        for scope in scopes:
            grouped[
                (
                    scope,
                    str(row["method"]),
                    float(row["selected_fraction_target"]),
                )
            ].append(row)

    output: list[dict[str, Any]] = []
    for (scope, method, fraction), items in sorted(grouped.items()):
        cases = sum(int(item["cases"]) for item in items)
        result: dict[str, Any] = {
            "scope": scope,
            "method": method,
            "selected_fraction_target": fraction,
            "datasets": len(items),
            "cases": cases,
            "calibration_steps_min": min(
                int(item["calibration_steps"]) for item in items
            ),
            "calibration_steps_max": max(
                int(item["calibration_steps"]) for item in items
            ),
            "code_bits_mean": weighted_mean(
                (float(item["code_bits_mean"]), int(item["cases"]))
                for item in items
            ),
            "metadata_bits_mean": weighted_mean(
                (float(item["metadata_bits_mean"]), int(item["cases"]))
                for item in items
            ),
            "index_ratio_of_full_kv": weighted_mean(
                (float(item["index_ratio_of_full_kv"]), int(item["cases"]))
                for item in items
            ),
        }
        for metric in METRICS:
            result[metric] = weighted_mean(
                (float(item[metric]), int(item["cases"])) for item in items
            )
            result[f"{metric}_worst_dataset"] = min(
                float(item[metric]) for item in items
            )
        output.append(result)
    return output


def aggregate_allocations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), str(row["method"]))].append(row)
        grouped[("all", str(row["method"]))].append(row)

    output: list[dict[str, Any]] = []
    for (scope, method), items in sorted(grouped.items()):
        allocation_counts = Counter(str(item["allocation"]) for item in items)
        mode, mode_count = allocation_counts.most_common(1)[0]
        result: dict[str, Any] = {
            "scope": scope,
            "method": method,
            "layer_kv_heads": len(items),
            "distinct_allocations": len(allocation_counts),
            "mode_allocation": mode,
            "mode_rate": mode_count / len(items),
            "code_bits_mean": sum(float(item["code_bits"]) for item in items)
            / len(items),
            "metadata_bits_mean": sum(
                float(item["metadata_bits"]) for item in items
            )
            / len(items),
        }
        for index in range(8):
            result[f"group{index}_bits_mean"] = sum(
                float(item[f"group{index}_bits"]) for item in items
            ) / len(items)
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate automatic spectral rate-allocation experiments."
    )
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, str]] = []
    for path in sorted(args.input_root.glob("*/summary.csv")):
        label = path.parent.name
        scope = scope_for_label(label)
        for row in read_csv(path):
            summary_rows.append({"label": label, "scope": scope, **row})
        allocation_path = path.parent / "allocations.csv"
        if allocation_path.exists():
            for row in read_csv(allocation_path):
                allocation_rows.append({"scope": scope, **row})
    if not summary_rows:
        raise ValueError(f"no experiment summaries found under {args.input_root}")

    combined = aggregate_summaries(summary_rows)
    allocations = aggregate_allocations(allocation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_summary.csv", combined)
    write_csv(args.output_dir / "combined_allocations.csv", allocations)
    output = {
        "input_root": str(args.input_root),
        "dataset_count": len({str(row["label"]) for row in summary_rows}),
        "combined_summary": combined,
        "combined_allocations": allocations,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
