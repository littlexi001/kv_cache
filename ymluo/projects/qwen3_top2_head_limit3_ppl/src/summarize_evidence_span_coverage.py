from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evidence span coverage outputs.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    task_rows = read_csv(output_dir / "task_results.csv")
    overall_rows = read_csv(output_dir / "coverage_by_task_overall.csv")
    layer_rows = read_csv(output_dir / "coverage_by_task_layer_head.csv")

    accuracy_rows: list[dict[str, Any]] = []
    variants = sorted({row["variant"] for row in task_rows})
    for variant in variants:
        for mode in ["baseline", "qabs8cand5reuse"]:
            subset = [row for row in task_rows if row["variant"] == variant and row["mode"] == mode]
            correct = sum(int(row["correct"]) for row in subset)
            accuracy_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "correct": correct,
                    "total": len(subset),
                    "accuracy": correct / max(1, len(subset)),
                }
            )

    coverage_totals: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    coverage_by_correct: dict[tuple[str, int, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in overall_rows:
        key = (row["variant"], row["mask"], row["span"], row["metric"])
        coverage_totals[key][0] += int(row["hit_count"])
        coverage_totals[key][1] += int(row["query_count"])
        corr_key = (row["variant"], int(row["correct"]), row["mask"], row["span"], row["metric"])
        coverage_by_correct[corr_key][0] += int(row["hit_count"])
        coverage_by_correct[corr_key][1] += int(row["query_count"])

    coverage_summary: list[dict[str, Any]] = []
    for key, (hits, queries) in sorted(coverage_totals.items()):
        variant, mask, span, metric = key
        coverage_summary.append(
            {
                "variant": variant,
                "mask": mask,
                "span": span,
                "metric": metric,
                "hit_count": hits,
                "query_count": queries,
                "coverage": hits / queries if queries else 0.0,
            }
        )

    correctness_summary: list[dict[str, Any]] = []
    for key, (hits, queries) in sorted(coverage_by_correct.items()):
        variant, correct, mask, span, metric = key
        correctness_summary.append(
            {
                "variant": variant,
                "correct": correct,
                "mask": mask,
                "span": span,
                "metric": metric,
                "hit_count": hits,
                "query_count": queries,
                "coverage": hits / queries if queries else 0.0,
            }
        )

    layer_totals: dict[tuple[str, int, int, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in layer_rows:
        key = (
            row["variant"],
            int(row["layer"]),
            int(row["head"]),
            row["mask"],
            row["span"],
            row["metric"],
        )
        layer_totals[key][0] += int(row["hit_count"])
        layer_totals[key][1] += int(row["query_count"])

    top_layer_rows: list[dict[str, Any]] = []
    for variant in variants:
        for span, metric in [("key", "any"), ("key", "all"), ("label", "any"), ("label", "all")]:
            scored = []
            for (v, layer, head, mask, s, m), (hits, queries) in layer_totals.items():
                if v == variant and mask == "final" and s == span and m == metric and queries:
                    scored.append((hits / queries, layer, head, hits, queries))
            for coverage, layer, head, hits, queries in sorted(scored, reverse=True)[:10]:
                top_layer_rows.append(
                    {
                        "variant": variant,
                        "mask": "final",
                        "span": span,
                        "metric": metric,
                        "layer": layer,
                        "head": head,
                        "hit_count": hits,
                        "query_count": queries,
                        "coverage": coverage,
                    }
                )

    write_csv(output_dir / "accuracy_summary.csv", accuracy_rows, ["variant", "mode", "correct", "total", "accuracy"])
    write_csv(
        output_dir / "coverage_summary.csv",
        coverage_summary,
        ["variant", "mask", "span", "metric", "hit_count", "query_count", "coverage"],
    )
    write_csv(
        output_dir / "coverage_by_correctness_summary.csv",
        correctness_summary,
        ["variant", "correct", "mask", "span", "metric", "hit_count", "query_count", "coverage"],
    )
    write_csv(
        output_dir / "top_layer_head_coverage.csv",
        top_layer_rows,
        ["variant", "mask", "span", "metric", "layer", "head", "hit_count", "query_count", "coverage"],
    )
    (output_dir / "coverage_report.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy_rows,
                "coverage_summary": coverage_summary,
                "coverage_by_correctness": correctness_summary,
                "top_layer_head_coverage": top_layer_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"accuracy": accuracy_rows, "top_layer_head_coverage": top_layer_rows[:12]}, indent=2))


if __name__ == "__main__":
    main()
