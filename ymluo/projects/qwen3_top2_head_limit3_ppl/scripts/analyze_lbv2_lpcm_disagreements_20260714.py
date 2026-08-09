#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FEATURE_KEYS = (
    "keep_fraction",
    "budget_tokens",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
    "ours_query_coverage_terms",
    "ours_query_coverage_covered",
    "ours_query_coverage_recall",
    "ours_graph_bridge_pairs",
    "ours_graph_bridge_tokens",
)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    full_score = mean([item["full_score"] for item in items])
    ours_score = mean([item["ours_score"] for item in items])
    return {
        "samples": len(items),
        "full_score": full_score,
        "ours_score": ours_score,
        "score_over_full": ours_score / full_score if full_score else None,
        "prediction_agreement": mean([float(item["prediction_agreement"]) for item in items]),
        "full_only": sum(item["outcome"] == "full_only" for item in items),
        "ours_only": sum(item["outcome"] == "ours_only" for item in items),
        "both_correct": sum(item["outcome"] == "both_correct" for item in items),
        "both_wrong": sum(item["outcome"] == "both_wrong" for item in items),
        "mean_kv_ratio": mean(
            [value for item in items if (value := item["features"].get("keep_fraction")) is not None]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.results)
    full = {row["sample_id"]: row for row in rows if row.get("method") == "full_kv"}
    ours = {row["sample_id"]: row for row in rows if row.get("method") == "ours_page_gather"}
    sample_ids = sorted(full.keys() & ours.keys())
    paired: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        full_row = full[sample_id]
        ours_row = ours[sample_id]
        full_score = float(number(full_row, "score") or 0.0)
        ours_score = float(number(ours_row, "score") or 0.0)
        full_correct = full_score > 0.5
        ours_correct = ours_score > 0.5
        if full_correct and ours_correct:
            outcome = "both_correct"
        elif full_correct:
            outcome = "full_only"
        elif ours_correct:
            outcome = "ours_only"
        else:
            outcome = "both_wrong"
        features = {key: number(ours_row, key) for key in FEATURE_KEYS}
        paired.append(
            {
                "sample_id": sample_id,
                "domain": full_row.get("domain", ""),
                "sub_domain": full_row.get("sub_domain", ""),
                "difficulty": full_row.get("difficulty", ""),
                "length_category": full_row.get("length_category", ""),
                "operator_mode": ours_row.get("ours_operator_mode", ""),
                "selected_action": ours_row.get("ours_action_router_selected_action", ""),
                "budget_tokens": int(float(ours_row.get("budget_tokens", "0") or 0)),
                "full_prediction": full_row.get("longbench_v2_pred", ""),
                "ours_prediction": ours_row.get("longbench_v2_pred", ""),
                "answer": ours_row.get("answers", ""),
                "full_score": full_score,
                "ours_score": ours_score,
                "prediction_agreement": full_row.get("longbench_v2_pred", "")
                == ours_row.get("longbench_v2_pred", ""),
                "outcome": outcome,
                "features": features,
            }
        )

    groups: dict[str, dict[str, Any]] = {}
    group_specs = {
        "domain": lambda item: item["domain"],
        "operator_mode": lambda item: item["operator_mode"],
        "budget_tokens": lambda item: str(item["budget_tokens"]),
        "outcome": lambda item: item["outcome"],
    }
    for group_name, key_fn in group_specs.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in paired:
            buckets[key_fn(item)].append(item)
        groups[group_name] = {key: summarize_group(items) for key, items in sorted(buckets.items())}

    feature_by_outcome: dict[str, dict[str, float | None]] = {}
    for outcome in ("both_correct", "full_only", "ours_only", "both_wrong"):
        items = [item for item in paired if item["outcome"] == outcome]
        feature_by_outcome[outcome] = {
            key: mean([value for item in items if (value := item["features"].get(key)) is not None])
            for key in FEATURE_KEYS
        }

    payload = {
        "overall": summarize_group(paired),
        "outcome_counts": dict(Counter(item["outcome"] for item in paired)),
        "groups": groups,
        "feature_means_by_outcome": feature_by_outcome,
        "prediction_disagreements": [item for item in paired if not item["prediction_agreement"]],
        "full_only_samples": [item for item in paired if item["outcome"] == "full_only"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_path = args.output.with_suffix(".csv")
    fieldnames = [
        "sample_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length_category",
        "operator_mode",
        "selected_action",
        "budget_tokens",
        "full_prediction",
        "ours_prediction",
        "answer",
        "full_score",
        "ours_score",
        "prediction_agreement",
        "outcome",
        *FEATURE_KEYS,
    ]
    with flat_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in paired:
            writer.writerow({**{key: item.get(key, "") for key in fieldnames}, **item["features"]})
    print(json.dumps({"overall": payload["overall"], "outcome_counts": payload["outcome_counts"], "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
