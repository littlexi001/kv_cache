from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


METRICS = (
    "positive_excess_pair_mass_fraction",
    "significant_positive_pairs",
    "significant_fraction_of_coobserved_pairs",
    "significant_conditional_median",
    "significant_lift_median",
    "largest_component_tokens",
    "distance_le_16_enrichment",
    "cluster_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-head Top-2% co-selection across corpora.")
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated label=/path/to/run entries.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_head_count", type=int, default=50)
    return parser.parse_args()


def parse_runs(spec: str) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for part in spec.split(","):
        label, separator, path = part.strip().partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Invalid run entry: {part!r}")
        runs.append((label, Path(path)))
    if len(runs) < 2:
        raise ValueError("At least two runs are required.")
    return runs


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["layer"]), int(row["head"])): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_keys(rows: dict[tuple[int, int], dict[str, Any]], count: int) -> set[tuple[int, int]]:
    ordered = sorted(rows, key=lambda key: float(rows[key]["cluster_score"]), reverse=True)
    return set(ordered[: min(count, len(ordered))])


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.runs)
    loaded = {
        label: read_rows(path / "head_coselection_summary.csv")
        for label, path in runs
    }
    common_keys = set.intersection(*(set(rows) for rows in loaded.values()))
    if not common_keys:
        raise RuntimeError("Runs have no common layer/head rows.")

    comparison_rows: list[dict[str, Any]] = []
    for layer, head in sorted(common_keys):
        row: dict[str, Any] = {"layer": layer, "head": head}
        for label, _ in runs:
            source = loaded[label][(layer, head)]
            for metric in METRICS:
                row[f"{label}_{metric}"] = source[metric]
        comparison_rows.append(row)

    summary: dict[str, Any] = {
        "common_heads": len(common_keys),
        "runs": {},
        "pairwise": {},
    }
    for label, _ in runs:
        rows = loaded[label]
        summary["runs"][label] = {
            "heads": len(rows),
            "heads_with_significant_pairs": sum(
                int(float(row["significant_positive_pairs"])) > 0 for row in rows.values()
            ),
            "median_significant_pairs": float(
                np.median([float(row["significant_positive_pairs"]) for row in rows.values()])
            ),
            "median_positive_excess_pair_mass_fraction": float(
                np.median([float(row["positive_excess_pair_mass_fraction"]) for row in rows.values()])
            ),
            "median_conditional_probability": float(
                np.median([float(row["significant_conditional_median"]) for row in rows.values()])
            ),
            "median_distance_le_16_enrichment": float(
                np.median([float(row["distance_le_16_enrichment"]) for row in rows.values()])
            ),
        }

    for left_index, (left_label, _) in enumerate(runs):
        for right_label, _ in runs[left_index + 1 :]:
            pair_key = f"{left_label}_vs_{right_label}"
            pair_summary: dict[str, Any] = {}
            for metric in METRICS:
                left = np.asarray(
                    [float(loaded[left_label][key][metric]) for key in sorted(common_keys)],
                    dtype=np.float64,
                )
                right = np.asarray(
                    [float(loaded[right_label][key][metric]) for key in sorted(common_keys)],
                    dtype=np.float64,
                )
                correlation = spearmanr(left, right).statistic
                pair_summary[f"{metric}_spearman"] = (
                    float(correlation) if np.isfinite(correlation) else None
                )
            left_top = top_keys(loaded[left_label], args.top_head_count)
            right_top = top_keys(loaded[right_label], args.top_head_count)
            union = left_top | right_top
            pair_summary["top_head_count"] = args.top_head_count
            pair_summary["top_head_intersection"] = len(left_top & right_top)
            pair_summary["top_head_jaccard"] = len(left_top & right_top) / len(union) if union else 0.0
            summary["pairwise"][pair_key] = pair_summary

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cross_corpus_head_comparison.csv", comparison_rows)
    (output_dir / "cross_corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
