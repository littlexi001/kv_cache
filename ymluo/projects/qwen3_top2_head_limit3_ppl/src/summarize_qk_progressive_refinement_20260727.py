from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "topk_recall",
    "selected_attention_mass",
    "oracle_mass_recall",
    "refinement_ratio",
    "interval_full_proxy_topk_recall",
    "sampled_radius_token_coverage",
    "access_rate_units",
    "access_ratio_of_full_index",
    "access_ratio_of_full_kv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.input_root.glob("*/per_query.csv"))
    if not files:
        raise ValueError(f"no per-query files under {args.input_root}")
    rows: list[dict[str, str]] = []
    for path in files:
        rows.extend(read_csv(path))

    full_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        if row["method"] != "full_packed_proxy":
            continue
        key = (
            row["label"],
            row["layer"],
            row["kv_head"],
            row["query_head"],
            row["step"],
            row["selected_fraction"],
        )
        full_by_key[key] = row

    grouped: dict[tuple[str, int, float, float], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                row["method"],
                int(row["base_rate_budget"]),
                float(row["selected_fraction"]),
                float(row["alpha"]),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (method, budget, fraction, alpha), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "base_rate_budget": budget,
            "selected_fraction": fraction,
            "alpha": alpha,
            "cases": len(items),
        }
        for metric in METRICS:
            values = np.asarray(
                [float(item[metric]) for item in items],
                dtype=np.float64,
            )
            for statistic, value in interval(values).items():
                result[f"{metric}_{statistic}"] = value

        paired_mass_retention = []
        paired_recall_delta = []
        for item in items:
            key = (
                item["label"],
                item["layer"],
                item["kv_head"],
                item["query_head"],
                item["step"],
                item["selected_fraction"],
            )
            full = full_by_key[key]
            paired_mass_retention.append(
                float(item["selected_attention_mass"])
                / max(1.0e-12, float(full["selected_attention_mass"]))
            )
            paired_recall_delta.append(
                float(item["topk_recall"])
                - float(full["topk_recall"])
            )
        result["paired_full_proxy_mass_retention_mean"] = float(
            np.mean(paired_mass_retention)
        )
        result["paired_full_proxy_mass_retention_p10"] = float(
            np.quantile(paired_mass_retention, 0.10)
        )
        result["paired_full_proxy_topk_recall_delta_mean"] = float(
            np.mean(paired_recall_delta)
        )
        summaries.append(result)

    progressive = [
        row for row in summaries if row["method"] == "progressive_interval"
    ]
    safe = [
        row
        for row in progressive
        if row["interval_full_proxy_topk_recall_mean"] >= 0.999
        and row["paired_full_proxy_mass_retention_mean"] >= 0.999
    ]
    best_by_fraction: dict[str, dict[str, Any] | None] = {}
    for fraction in sorted(
        {float(row["selected_fraction"]) for row in progressive}
    ):
        eligible = [
            row
            for row in safe
            if float(row["selected_fraction"]) == fraction
        ]
        best_by_fraction[str(fraction)] = (
            min(
                eligible,
                key=lambda row: (
                    float(row["access_ratio_of_full_index_mean"]),
                    -float(row["paired_full_proxy_mass_retention_mean"]),
                ),
            )
            if eligible
            else None
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_summary.csv", summaries)
    payload = {
        "schema": "qk_progressive_refinement_summary_v1",
        "input_files": [str(path) for path in files],
        "rows": len(rows),
        "safe_definition": {
            "mean_full_proxy_topk_containment": 0.999,
            "mean_attention_mass_retention_vs_full_proxy": 0.999,
        },
        "best_safe_by_selected_fraction": best_by_fraction,
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
