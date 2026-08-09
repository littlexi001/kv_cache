from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "top1_agreement",
    "margin_certificate_satisfied",
    "full_top1_margin",
    "shift_invariant_logit_delta_range",
    "kl_full_to_sparse",
    "kl_range_upper_bound",
    "kl_range_bound_satisfied",
    "js_divergence",
    "target_nll_delta",
    "target_nll_range_bound_satisfied",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired Full/CountCap token-logit stability."
    )
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sparse_rows = [
        row
        for row in rows
        if row.get("method") == "direct_countcap"
        and row.get("top1_agreement", "") != ""
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sparse_rows:
        grouped[row["topic"]].append(row)
        grouped["ALL"].append(row)

    summaries = []
    for topic, subset in sorted(
        grouped.items(),
        key=lambda item: (item[0] == "ALL", item[0]),
    ):
        summary: dict[str, Any] = {
            "topic": topic,
            "tokens": len(subset),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_p50"] = percentile(values, 0.50)
            summary[f"{metric}_p90"] = percentile(values, 0.90)
            summary[f"{metric}_p99"] = percentile(values, 0.99)
            summary[f"{metric}_max"] = max(values)

        certified = [
            row
            for row in subset
            if int(float(row["margin_certificate_satisfied"])) == 1
        ]
        uncertified = [
            row
            for row in subset
            if int(float(row["margin_certificate_satisfied"])) == 0
        ]
        summary["certified_violations"] = sum(
            int(float(row["top1_agreement"])) == 0 for row in certified
        )
        summary["uncertified_flip_rate"] = (
            sum(
                int(float(row["top1_agreement"])) == 0
                for row in uncertified
            )
            / len(uncertified)
            if uncertified
            else 0.0
        )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(args.input_glob)
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    summaries = summarize_rows(rows)
    if not summaries:
        raise RuntimeError("no direct_countcap logit-stability rows found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
