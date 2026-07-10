#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    out: dict[tuple[str, str], dict[str, str]] = {}
    with (path / "task_results.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["task"], row["sample_id"])] = row
    return out


def mean(rows: list[dict[str, str]], column: str) -> float:
    return sum(fnum(row.get(column)) for row in rows) / max(1, len(rows))


def risk_score(
    row: dict[str, str],
    gap2_weight: float,
    gap3_weight: float,
    top_score_weight: float,
) -> float:
    return (
        fnum(row.get("ours_score_entropy"))
        - gap2_weight * fnum(row.get("ours_score_gap2"))
        - gap3_weight * fnum(row.get("ours_score_gap3"))
        - top_score_weight * fnum(row.get("ours_score_max"))
    )


def danger_label(
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


def policy_patch(
    tasks: set[str],
    threshold: float,
    consistency_budget_tokens: int,
    gap2_weight: float,
    gap3_weight: float,
    top_score_weight: float,
) -> dict[str, Any]:
    return {
        task: {
            "score_risk": True,
            "score_risk_linear_threshold": threshold,
            "score_risk_gap2_weight": gap2_weight,
            "score_risk_gap3_weight": gap3_weight,
            "score_risk_top_score_weight": top_score_weight,
            "consistency_verifier": True,
            "consistency_budget_tokens": consistency_budget_tokens,
            "consistency_requires_score_risk": 1,
        }
        for task in sorted(tasks)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--full", type=Path, default=None)
    parser.add_argument("--tasks", default="narrativeqa,multifieldqa_en,2wikimqa")
    parser.add_argument("--target_recall", type=float, default=0.90)
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
    parser.add_argument("--consistency_budget_tokens", type=int, default=2048)
    parser.add_argument("--write_policy_patch", type=Path, default=None)
    args = parser.parse_args()

    base = load_task_results(args.base)
    reference = load_task_results(args.reference)
    full = load_task_results(args.full) if args.full else {}
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    keys = sorted(set(base) & set(reference))
    if full:
        keys = [key for key in keys if key in full]
    rows = [key for key in keys if key[0] in tasks]
    if not rows:
        raise SystemExit("No overlapping task rows for calibration.")

    scored: list[dict[str, Any]] = []
    for key in rows:
        base_row = base[key]
        reference_row = reference[key]
        full_row = full.get(key)
        score = risk_score(base_row, args.gap2_weight, args.gap3_weight, args.top_score_weight)
        label = danger_label(
            base_row,
            reference_row,
            full_row,
            args.label_mode,
            args.full_gain_margin,
            args.reference_gain_margin,
        )
        scored.append({"key": key, "risk_score": score, "danger": label})

    danger_scores = [row["risk_score"] for row in scored if row["danger"]]
    threshold = threshold_for_recall(danger_scores, args.target_recall)
    triggered_keys = {row["key"] for row in scored if row["risk_score"] >= threshold}
    dangerous = [row for row in scored if row["danger"]]
    true_positive = [row for row in scored if row["danger"] and row["key"] in triggered_keys]

    stitched: list[dict[str, str]] = []
    for key in sorted(set(base) & set(reference)):
        if key[0] in tasks:
            stitched.append(reference[key] if key in triggered_keys else base[key])
        else:
            stitched.append(reference[key])

    precision = len(true_positive) / max(1, len(triggered_keys))
    recall = len(true_positive) / max(1, len(dangerous))
    summary = {
        "tasks": ",".join(sorted(tasks)),
        "label_mode": args.label_mode,
        "target_recall": args.target_recall,
        "threshold": threshold,
        "gap2_weight": args.gap2_weight,
        "gap3_weight": args.gap3_weight,
        "top_score_weight": args.top_score_weight,
        "calibration_samples": len(scored),
        "danger_count": len(dangerous),
        "trigger_count": len(triggered_keys),
        "trigger_rate_on_calibration_tasks": len(triggered_keys) / max(1, len(scored)),
        "precision": precision,
        "recall": recall,
        "stitched_score": mean(stitched, "score"),
        "stitched_keep_fraction": mean(stitched, "keep_fraction"),
        "stitched_online_seconds": mean(stitched, "online_seconds"),
        "stitched_consistency_check_rate": mean(stitched, "ours_consistency_check_active"),
        "stitched_consistency_full_fallback_rate": mean(stitched, "ours_consistency_full_fallback_active"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.write_policy_patch is not None:
        payload = policy_patch(
            tasks,
            threshold,
            args.consistency_budget_tokens,
            args.gap2_weight,
            args.gap3_weight,
            args.top_score_weight,
        )
        args.write_policy_patch.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
