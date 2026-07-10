#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def summarize_task_results(path: Path) -> list[dict[str, Any]]:
    rows = []
    task_results = path / "task_results.csv"
    if not task_results.exists():
        return rows
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with task_results.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row.get("task", "ALL")].append(row)
            groups["ALL"].append(row)
    for task, subset in sorted(groups.items()):
        n = max(1, len(subset))
        rows.append(
            {
                "experiment": path.name,
                "task": task,
                "samples": len(subset),
                "score": sum(fnum(row.get("score")) for row in subset) / n,
                "keep_fraction": sum(fnum(row.get("keep_fraction")) for row in subset) / n,
                "online_seconds": sum(fnum(row.get("online_seconds")) for row in subset) / n,
                "full_fallback_rate": sum(fnum(row.get("ours_full_fallback_active")) for row in subset) / n,
                "output_fallback_rate": sum(fnum(row.get("ours_output_fallback_active")) for row in subset) / n,
                "retry_fallback_rate": sum(fnum(row.get("ours_retry_fallback_active")) for row in subset) / n,
                "retry_full_fallback_rate": sum(fnum(row.get("ours_retry_full_fallback_active")) for row in subset) / n,
                "score_risk_active_rate": sum(fnum(row.get("ours_score_risk_active")) for row in subset) / n,
                "score_risk_trigger_rate": sum(fnum(row.get("ours_score_risk_triggered")) for row in subset) / n,
                "score_risk_linear_value": sum(fnum(row.get("ours_score_risk_linear_value")) for row in subset) / n,
                "score_risk_linear_threshold": sum(
                    fnum(row.get("ours_score_risk_linear_threshold"), default=-1.0) for row in subset
                )
                / n,
                "budget_ladder_active_rate": sum(fnum(row.get("ours_budget_ladder_active")) for row in subset) / n,
                "budget_ladder_selected_budget": sum(
                    fnum(row.get("ours_budget_ladder_selected_budget")) for row in subset
                )
                / n,
                "budget_ladder_level": sum(fnum(row.get("ours_budget_ladder_level")) for row in subset) / n,
                "coverage_certificate_active_rate": sum(
                    fnum(row.get("ours_coverage_certificate_active")) for row in subset
                )
                / n,
                "coverage_certificate_recall": sum(
                    fnum(row.get("ours_coverage_certificate_recall")) for row in subset
                )
                / n,
                "coverage_certificate_tokens": sum(
                    fnum(row.get("ours_coverage_certificate_tokens")) for row in subset
                )
                / n,
                "graph_bridge_active_rate": sum(fnum(row.get("ours_graph_bridge_active")) for row in subset) / n,
                "graph_bridge_pairs": sum(fnum(row.get("ours_graph_bridge_pairs")) for row in subset) / n,
                "graph_bridge_tokens": sum(fnum(row.get("ours_graph_bridge_tokens")) for row in subset) / n,
                "structured_fingerprint_active_rate": sum(
                    fnum(row.get("ours_structured_fingerprint_active")) for row in subset
                )
                / n,
                "structured_fingerprint_labels": sum(
                    fnum(row.get("ours_structured_fingerprint_labels")) for row in subset
                )
                / n,
                "structured_fingerprint_tokens": sum(
                    fnum(row.get("ours_structured_fingerprint_tokens")) for row in subset
                )
                / n,
                "consistency_check_rate": sum(fnum(row.get("ours_consistency_check_active")) for row in subset) / n,
                "consistency_disagreement_rate": sum(
                    fnum(row.get("ours_consistency_disagreement_active")) for row in subset
                )
                / n,
                "consistency_full_fallback_rate": sum(
                    fnum(row.get("ours_consistency_full_fallback_active")) for row in subset
                )
                / n,
                "consistency_requires_score_risk_rate": sum(
                    fnum(row.get("ours_consistency_requires_score_risk")) for row in subset
                )
                / n,
                "grounding_fallback_rate": sum(fnum(row.get("ours_grounding_fallback_active")) for row in subset) / n,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Experiment output directories.")
    args = parser.parse_args()

    print(
        "experiment,task,samples,score,keep_fraction,online_seconds,"
        "full_fallback_rate,output_fallback_rate,retry_fallback_rate,"
        "retry_full_fallback_rate,score_risk_active_rate,score_risk_trigger_rate,"
        "score_risk_linear_value,score_risk_linear_threshold,"
        "budget_ladder_active_rate,budget_ladder_selected_budget,budget_ladder_level,"
        "coverage_certificate_active_rate,coverage_certificate_recall,coverage_certificate_tokens,"
        "graph_bridge_active_rate,graph_bridge_pairs,graph_bridge_tokens,"
        "structured_fingerprint_active_rate,structured_fingerprint_labels,structured_fingerprint_tokens,"
        "consistency_check_rate,consistency_disagreement_rate,"
        "consistency_full_fallback_rate,consistency_requires_score_risk_rate,grounding_fallback_rate"
    )
    for raw_path in args.paths:
        path = Path(raw_path)
        for row in summarize_task_results(path):
            print(
                f"{row['experiment']},{row['task']},{row['samples']},"
                f"{row['score']:.6f},{row['keep_fraction']:.6f},{row['online_seconds']:.6f},"
                f"{row['full_fallback_rate']:.6f},{row['output_fallback_rate']:.6f},"
                f"{row['retry_fallback_rate']:.6f},{row['retry_full_fallback_rate']:.6f},"
                f"{row['score_risk_active_rate']:.6f},{row['score_risk_trigger_rate']:.6f},"
                f"{row['score_risk_linear_value']:.6f},{row['score_risk_linear_threshold']:.6f},"
                f"{row['budget_ladder_active_rate']:.6f},{row['budget_ladder_selected_budget']:.6f},"
                f"{row['budget_ladder_level']:.6f},"
                f"{row['coverage_certificate_active_rate']:.6f},{row['coverage_certificate_recall']:.6f},"
                f"{row['coverage_certificate_tokens']:.6f},"
                f"{row['graph_bridge_active_rate']:.6f},{row['graph_bridge_pairs']:.6f},"
                f"{row['graph_bridge_tokens']:.6f},"
                f"{row['structured_fingerprint_active_rate']:.6f},"
                f"{row['structured_fingerprint_labels']:.6f},"
                f"{row['structured_fingerprint_tokens']:.6f},"
                f"{row['consistency_check_rate']:.6f},{row['consistency_disagreement_rate']:.6f},"
                f"{row['consistency_full_fallback_rate']:.6f},{row['consistency_requires_score_risk_rate']:.6f},"
                f"{row['grounding_fallback_rate']:.6f}"
            )


if __name__ == "__main__":
    main()
