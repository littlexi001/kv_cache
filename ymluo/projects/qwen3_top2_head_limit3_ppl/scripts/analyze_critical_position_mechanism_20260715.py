from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from train_provisional_criticality_router_20260715 import (
    ACTIONS,
    load_dataset,
    summarize_policy,
)


FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "hundred",
    "thousand",
    "million",
    "billion",
}


def read_metadata(root: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    output = {}
    for path in sorted(root.glob("*/metadata.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["topic"], int(row["window"]), int(row["target_index"]))
                output[key] = row
    return output


def lexical_class(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if re.search(r"\d", stripped) or lowered in NUMBER_WORDS:
        return "number"
    if lowered in FUNCTION_WORDS:
        return "function_word"
    if stripped and not any(character.isalnum() for character in stripped):
        return "punctuation"
    if stripped[:1].isupper():
        return "capitalized"
    return "other_content"


def confidence_bin(probability: float) -> str:
    if probability < 0.1:
        return "top1_prob_[0,0.1)"
    if probability < 0.3:
        return "top1_prob_[0.1,0.3)"
    if probability < 0.6:
        return "top1_prob_[0.3,0.6)"
    return "top1_prob_[0.6,1]"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {"tokens": len(rows)}
    for action in ACTIONS:
        deltas = [float(row["deltas"][action]) for row in rows]
        mean_delta = sum(deltas) / len(deltas)
        output[f"delta_nll_{action}"] = mean_delta
        output[f"ppl_ratio_{action}"] = math.exp(mean_delta)
        output[f"danger_rate_{action}_delta_gt_0p05"] = sum(
            delta > 0.05 for delta in deltas
        ) / len(deltas)
        output[f"danger_rate_{action}_delta_gt_0p10"] = sum(
            delta > 0.10 for delta in deltas
        ) / len(deltas)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--hidden_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    records = load_dataset(args.scan_root, args.reference_root)
    metadata = read_metadata(args.hidden_root)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    enriched = []
    for record in records:
        row = metadata[record["key"]]
        label_id = int(row["label_id"])
        top1_id = int(row["top1_id"])
        item = {
            **record,
            "lexical_class": lexical_class(row["label_text"]),
            "confidence_bin": confidence_bin(float(row["top1_probability"])),
            "top1_correct": label_id == top1_id,
        }
        enriched.append(item)
        grouped[f"lexical:{item['lexical_class']}"] .append(item)
        grouped[f"confidence:{item['confidence_bin']}"] .append(item)
        grouped[f"top1_correct:{item['top1_correct']}"] .append(item)
        grouped[f"window:{record['key'][1]}"] .append(item)

    report = {
        "all": summarize(enriched),
        "groups": {
            name: summarize(rows)
            for name, rows in sorted(grouped.items())
            if len(rows) >= 8
        },
        "note": (
            "top1 confidence is measured after the full low-budget forward and is only a "
            "mechanism diagnostic; it is not a causal online router feature."
        ),
    }
    oracle_actions = []
    for record in enriched:
        selected = "rerank"
        for action in ACTIONS:
            if float(record["deltas"][action]) <= 0.05:
                selected = action
                break
        oracle_actions.append(selected)
    report["diagnostic_oracle_min_safe_action"] = summarize_policy(
        enriched, oracle_actions
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
