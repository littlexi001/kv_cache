from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep generated answer steps accepted by a frozen anchor confidence audit."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--confidence_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence = json.loads(Path(args.confidence_path).read_text(encoding="utf-8"))
    accepted_ids = {
        int(row["query_id"]) for row in confidence["details"] if bool(row["accepted"])
    }
    steps = read_jsonl(Path(args.step_queries_path))
    candidates = read_jsonl(Path(args.candidate_rows_path))
    accepted_steps = [row for row in steps if int(row["query_id"]) in accepted_ids]
    accepted_candidates = [
        row for row in candidates if int(row["query_id"]) in accepted_ids
    ]
    step_keys = {(int(row["query_id"]), int(row["step_index"])) for row in accepted_steps}
    candidate_keys = {
        (int(row["query_id"]), int(row["step_index"])) for row in accepted_candidates
    }
    if step_keys != candidate_keys:
        raise ValueError("accepted steps and candidate rows do not cover the same keys")
    write_jsonl(output_dir / "steps.jsonl", accepted_steps)
    write_jsonl(output_dir / "candidate_rows.jsonl", accepted_candidates)
    summary = {
        "source": "frozen anchor candidate-count confidence gate",
        "selection_uses_test_gold": False,
        "threshold": int(confidence["threshold"]),
        "input_steps": len(steps),
        "accepted_steps": len(accepted_steps),
        "acceptance_rate": len(accepted_steps) / len(steps),
        "steps_path": str(output_dir / "steps.jsonl"),
        "candidate_rows_path": str(output_dir / "candidate_rows.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
