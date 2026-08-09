from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

from io_utils import read_json, write_csv, write_json


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else None


def flatten_summary(path: Path, strategy_name: str) -> dict[str, Any]:
    summary = read_json(path)
    ppl = summary.get("validation_ppl", {})
    controlled = summary.get("controlled", [])
    longbench = summary.get("longbench", [])
    row = {
        "strategy": strategy_name if summary["label"] != "base" else "base_checkpoint",
        "label": summary["label"],
        "step": int(summary["step"]),
        "critical_complete": bool(summary.get("critical_complete", False)),
        "failure_count": int(summary.get("failure_count", 0)),
        "validation_ppl": ppl.get("ppl"),
        "validation_nll": ppl.get("mean_nll"),
        "controlled_qa_f1_percent": average(controlled, "qa_f1_percent"),
        "controlled_exact_match_percent": average(controlled, "exact_match_percent"),
        "controlled_contains_answer_percent": average(controlled, "contains_answer_percent"),
        "controlled_gold_nll": average(controlled, "gold_answer_mean_nll"),
        "longbench_status": summary.get("longbench_status"),
        "longbench_qa_f1_percent": average(longbench, "qa_f1_percent"),
        "longbench_exact_match_percent": average(longbench, "exact_match_percent"),
        "longbench_contains_answer_percent": average(longbench, "contains_answer_percent"),
        "longbench_gold_nll": average(longbench, "gold_answer_mean_nll"),
        "evaluation_elapsed_seconds": summary.get("elapsed_seconds"),
        "summary_path": str(path),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strategy-name", required=True)
    args = parser.parse_args()
    evaluations = args.run_dir / "evaluations"
    paths = sorted(evaluations.glob("*/summary.json"))
    rows = [flatten_summary(path, args.strategy_name) for path in paths]
    rows.sort(key=lambda row: (int(row["step"]), str(row["strategy"])))
    base = next((row for row in rows if row["strategy"] == "base_checkpoint"), None)
    for row in rows:
        if base and row["strategy"] != "base_checkpoint":
            if row["validation_ppl"] is not None and base["validation_ppl"] is not None:
                row["validation_ppl_change_vs_base_percent"] = 100.0 * (
                    float(row["validation_ppl"]) / float(base["validation_ppl"]) - 1.0
                )
            for metric in ["controlled_qa_f1_percent", "longbench_qa_f1_percent"]:
                if row.get(metric) is not None and base.get(metric) is not None:
                    row[f"{metric}_change_vs_base_pp"] = float(row[metric]) - float(base[metric])
    milestone_metadata = {
        path.stem: read_json(path)
        for path in sorted(args.run_dir.glob("milestone_*.json"))
    }
    milestone_by_step = {
        int(value["step"]): value for value in milestone_metadata.values()
    }
    for row in rows:
        milestone = milestone_by_step.get(int(row["step"]))
        if milestone and row["strategy"] != "base_checkpoint":
            row["world_size"] = milestone.get("world_size")
            row["tokens_per_step"] = milestone.get("tokens_per_step")
            row["tokens_seen_nominal"] = milestone.get("tokens_seen_nominal")
            row["training_wall_seconds_last_segment"] = milestone.get("segment_wall_seconds")
    payload = {
        "strategy": args.strategy_name,
        "rows": rows,
        "milestones": milestone_metadata,
        "interpretation": {
            "base_comparison": "Compares a trained checkpoint with the common step-0 initialization; it does not isolate PE.",
            "native_comparison": "Must be added by merge_result_bundles.py before attributing an improvement to PE.",
            "screening_only": True,
        },
    }
    write_json(args.run_dir / "strategy_summary.json", payload)
    write_csv(args.run_dir / "strategy_summary.csv", rows)
    print(args.run_dir / "strategy_summary.json")


if __name__ == "__main__":
    main()
