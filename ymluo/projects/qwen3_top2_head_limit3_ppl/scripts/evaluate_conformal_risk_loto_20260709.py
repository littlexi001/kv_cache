#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def load_task_results(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with (path / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["task"], row["sample_id"])] = row
    return rows


def risk_score(row: dict[str, str], gap2_weight: float, gap3_weight: float, top_score_weight: float) -> float:
    return (
        fnum(row.get("ours_score_entropy"))
        - gap2_weight * fnum(row.get("ours_score_gap2"))
        - gap3_weight * fnum(row.get("ours_score_gap3"))
        - top_score_weight * fnum(row.get("ours_score_max"))
    )


def is_danger(
    base_row: dict[str, str],
    reference_row: dict[str, str],
    full_row: dict[str, str] | None,
    label_mode: str,
    full_gain_margin: float,
    reference_gain_margin: float,
) -> bool:
    labels: list[bool] = []
    if label_mode in {"consistency", "union"}:
        labels.append(
            fnum(reference_row.get("ours_consistency_disagreement_active")) > 0.0
            or fnum(reference_row.get("ours_consistency_full_fallback_active")) > 0.0
        )
    if label_mode in {"reference_gain", "union"}:
        labels.append(fnum(reference_row.get("score")) - fnum(base_row.get("score")) > reference_gain_margin)
    if label_mode in {"full_gain", "union"} and full_row is not None:
        labels.append(fnum(full_row.get("score")) - fnum(base_row.get("score")) > full_gain_margin)
    return any(labels)


def threshold_for_recall(scores: list[float], target_recall: float) -> float:
    if not scores:
        return float("inf")
    ordered = sorted(scores)
    miss_fraction = max(0.0, min(1.0, 1.0 - target_recall))
    index = int(miss_fraction * (len(ordered) - 1))
    return ordered[index]


def mean(rows: list[dict[str, str]], column: str) -> float:
    return sum(fnum(row.get(column)) for row in rows) / max(1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--full", type=Path, default=None)
    parser.add_argument("--tasks", default="narrativeqa,multifieldqa_en,2wikimqa")
    parser.add_argument("--target_recall", type=float, default=0.80)
    parser.add_argument(
        "--label_mode",
        choices=["consistency", "reference_gain", "full_gain", "union"],
        default="consistency",
    )
    parser.add_argument("--full_gain_margin", type=float, default=0.05)
    parser.add_argument("--reference_gain_margin", type=float, default=0.01)
    parser.add_argument("--gap2_weight", type=float, default=1.0)
    parser.add_argument("--gap3_weight", type=float, default=0.0)
    parser.add_argument("--top_score_weight", type=float, default=0.0)
    args = parser.parse_args()

    base = load_task_results(args.base)
    reference = load_task_results(args.reference)
    full = load_task_results(args.full) if args.full else {}
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    common_keys = sorted(set(base) & set(reference))
    if full:
        common_keys = [key for key in common_keys if key in full]

    print(
        "heldout_task,threshold,train_samples,train_danger,heldout_samples,heldout_danger,"
        "heldout_trigger_rate,heldout_precision,heldout_recall,heldout_score,heldout_keep_fraction,"
        "heldout_online_seconds,overall_score,overall_keep_fraction,overall_online_seconds,"
        "overall_consistency_check_rate"
    )
    for heldout in tasks:
        train_scores: list[float] = []
        train_danger = 0
        train_samples = 0
        for key in common_keys:
            task, _ = key
            if task == heldout or task not in tasks:
                continue
            train_samples += 1
            label = is_danger(
                base[key],
                reference[key],
                full.get(key),
                args.label_mode,
                args.full_gain_margin,
                args.reference_gain_margin,
            )
            if label:
                train_danger += 1
                train_scores.append(risk_score(base[key], args.gap2_weight, args.gap3_weight, args.top_score_weight))
        threshold = threshold_for_recall(train_scores, args.target_recall)

        heldout_rows: list[dict[str, str]] = []
        heldout_danger = 0
        heldout_trigger = 0
        heldout_true_positive = 0
        overall_rows: list[dict[str, str]] = []
        for key in common_keys:
            task, _ = key
            if task in tasks:
                triggered = risk_score(base[key], args.gap2_weight, args.gap3_weight, args.top_score_weight) >= threshold
                row = reference[key] if triggered else base[key]
            else:
                row = reference[key]
            overall_rows.append(row)
            if task != heldout:
                continue
            label = is_danger(
                base[key],
                reference[key],
                full.get(key),
                args.label_mode,
                args.full_gain_margin,
                args.reference_gain_margin,
            )
            if label:
                heldout_danger += 1
            if triggered:
                heldout_trigger += 1
            if label and triggered:
                heldout_true_positive += 1
            heldout_rows.append(row)

        heldout_precision = heldout_true_positive / max(1, heldout_trigger)
        heldout_recall = heldout_true_positive / max(1, heldout_danger)
        print(
            f"{heldout},{threshold:.12f},{train_samples},{train_danger},"
            f"{len(heldout_rows)},{heldout_danger},"
            f"{heldout_trigger / max(1, len(heldout_rows)):.6f},"
            f"{heldout_precision:.6f},{heldout_recall:.6f},"
            f"{mean(heldout_rows, 'score'):.6f},"
            f"{mean(heldout_rows, 'keep_fraction'):.6f},"
            f"{mean(heldout_rows, 'online_seconds'):.6f},"
            f"{mean(overall_rows, 'score'):.6f},"
            f"{mean(overall_rows, 'keep_fraction'):.6f},"
            f"{mean(overall_rows, 'online_seconds'):.6f},"
            f"{mean(overall_rows, 'ours_consistency_check_active'):.6f}"
        )


if __name__ == "__main__":
    main()
