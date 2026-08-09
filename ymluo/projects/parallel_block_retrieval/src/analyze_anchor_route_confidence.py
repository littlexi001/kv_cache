from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate anchor candidate-count confidence on dev and audit generated states."
    )
    parser.add_argument("--calibration_rows_path", required=True)
    parser.add_argument("--generated_rows_path", required=True)
    parser.add_argument("--bridge_traces_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    calibration = [
        row
        for row in read_jsonl(Path(args.calibration_rows_path))
        if str(row["split"]) == "dev"
        and str(row["step_type"]) == "resolve_answer_from_bridge"
    ]
    if not calibration:
        raise ValueError("no dev answer rows found for calibration")
    calibration_counts = [len(row["anchor_candidates"]) for row in calibration]
    threshold = int(np.quantile(calibration_counts, args.quantile, method="higher"))
    generated = read_jsonl(Path(args.generated_rows_path))
    traces = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.bridge_traces_path))
    }
    details = []
    for row in generated:
        query_id = int(row["query_id"])
        candidate_count = len(row["anchor_candidates"])
        accepted = 0 < candidate_count <= threshold
        details.append(
            {
                "query_id": query_id,
                "candidate_count": candidate_count,
                "accepted": accepted,
                "bridge_state_hit": bool(traces[query_id]["bridge_target_hit"]),
                "target_top3": 0 < int(row["anchor_rank"]) <= 3,
                "target_top512": 0 < int(row["anchor_rank"]) <= 512,
            }
        )
    accepted_rows = [row for row in details if row["accepted"]]
    payload = {
        "source": "candidate-count confidence calibrated only on valid dev states",
        "selection_uses_test_gold": False,
        "calibration_split": "dev",
        "quantile": args.quantile,
        "threshold": threshold,
        "calibration_candidate_counts": calibration_counts,
        "generated_test_steps": len(details),
        "acceptance_rate": len(accepted_rows) / len(details),
        "accepted_bridge_state_precision": (
            sum(row["bridge_state_hit"] for row in accepted_rows) / len(accepted_rows)
            if accepted_rows
            else 0.0
        ),
        "accepted_target_top3_rate": (
            sum(row["target_top3"] for row in accepted_rows) / len(accepted_rows)
            if accepted_rows
            else 0.0
        ),
        "rejected_steps": sum(not row["accepted"] for row in details),
        "details": details,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
