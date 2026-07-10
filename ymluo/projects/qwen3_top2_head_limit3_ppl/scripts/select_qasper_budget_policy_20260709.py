#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    label: str
    output_dir: str
    policy: str
    m50_name: str


DEFAULT_CANDIDATES = [
    Candidate(
        "v81_qasper2048",
        "outputs/riskkv_v81_v72_qasper_budgeted_m20_20260709",
        "configs/riskkv_task_policy_v81_v72_qasper_budgeted_20260709.json",
        "riskkv_v81_v72_qasper_budgeted_m50_20260709",
    ),
    Candidate(
        "v82_qasper1024",
        "outputs/riskkv_v82_v72_qasper1024_m20_retry_20260709",
        "configs/riskkv_task_policy_v82_v72_qasper1024_20260709.json",
        "riskkv_v82_v72_qasper1024_m50_20260709",
    ),
    Candidate(
        "v83_qasper1536",
        "outputs/riskkv_v83_v72_qasper1536_m20_retry_20260709",
        "configs/riskkv_task_policy_v83_v72_qasper1536_20260709.json",
        "riskkv_v83_v72_qasper1536_m50_20260709",
    ),
    Candidate(
        "v84_qasper3072",
        "outputs/riskkv_v84_v72_qasper3072_m20_retry_20260709",
        "configs/riskkv_task_policy_v84_v72_qasper3072_20260709.json",
        "riskkv_v84_v72_qasper3072_m50_20260709",
    ),
    Candidate(
        "v85_qasper_adaptive1024_2048",
        "outputs/riskkv_v85_v72_qasper_adaptive1024_2048_m20_20260709",
        "configs/riskkv_task_policy_v85_v72_qasper_adaptive1024_2048_20260709.json",
        "riskkv_v85_v72_qasper_adaptive1024_2048_m50_20260709",
    ),
]


def read_all_row(output_dir: str) -> tuple[float, float, int] | None:
    path = Path(output_dir) / "summary.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("benchmark") == "ALL" and row.get("task") == "ALL":
                return float(row["score"]), float(row["keep_fraction"]), int(float(row["samples"]))
    return None


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="outputs/riskkv_v72_2wiki_grounded_only_qasper_full_m20_20260709",
        help="Baseline output dir used for the quality tolerance.",
    )
    parser.add_argument(
        "--score_tolerance",
        type=float,
        default=0.002,
        help="A candidate within baseline_score - tolerance is selected by lowest KV ratio.",
    )
    parser.add_argument("--shell", action="store_true", help="Print shell assignments for the selected candidate.")
    args = parser.parse_args()

    baseline = read_all_row(args.baseline)
    if baseline is None:
        raise SystemExit(f"missing baseline summary: {args.baseline}/summary.csv")
    baseline_score, baseline_keep, _baseline_samples = baseline

    rows = []
    missing = []
    for candidate in DEFAULT_CANDIDATES:
        metrics = read_all_row(candidate.output_dir)
        if metrics is None:
            missing.append(candidate.output_dir)
            continue
        score, keep, samples = metrics
        rows.append((candidate, score, keep, samples))
    if missing:
        raise SystemExit("missing candidate summaries:\n" + "\n".join(missing))
    if not rows:
        raise SystemExit("no candidates")

    quality_floor = baseline_score - args.score_tolerance
    eligible = [row for row in rows if row[1] >= quality_floor]
    if eligible:
        selected = min(eligible, key=lambda row: (row[2], -row[1]))
        reason = "lowest_kv_within_quality_floor"
    else:
        selected = max(rows, key=lambda row: (row[1], -row[2]))
        reason = "highest_score_no_candidate_met_floor"

    if args.shell:
        candidate, score, keep, samples = selected
        print(f"SELECTED_LABEL={shell_quote(candidate.label)}")
        print(f"SELECTED_POLICY={shell_quote(candidate.policy)}")
        print(f"SELECTED_M50_NAME={shell_quote(candidate.m50_name)}")
        print(f"SELECTED_SCORE={score:.9f}")
        print(f"SELECTED_KEEP={keep:.9f}")
        print(f"SELECTED_SAMPLES={samples}")
        print(f"SELECTION_REASON={shell_quote(reason)}")
        print(f"BASELINE_SCORE={baseline_score:.9f}")
        print(f"BASELINE_KEEP={baseline_keep:.9f}")
        return

    print("label,samples,score,keep,baseline_score,baseline_keep,eligible,selected,reason")
    selected_label = selected[0].label
    for candidate, score, keep, samples in rows:
        eligible_flag = int(score >= quality_floor)
        selected_flag = int(candidate.label == selected_label)
        print(
            f"{candidate.label},{samples},{score:.9f},{keep:.9f},"
            f"{baseline_score:.9f},{baseline_keep:.9f},{eligible_flag},{selected_flag},{reason}"
        )


if __name__ == "__main__":
    main()
