from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group top sentence spans into one compact generation branch per block."
    )
    parser.add_argument("--rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--branch_blocks", type=int, default=3)
    parser.add_argument("--spans_per_block", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in read_jsonl(Path(args.rows_path)):
        grouped: dict[int, list[dict[str, Any]]] = {}
        block_order = []
        for candidate in row["branch_candidates"]:
            block_id = int(candidate["block_id"])
            if block_id not in grouped:
                grouped[block_id] = []
                block_order.append(block_id)
            if len(grouped[block_id]) < args.spans_per_block:
                grouped[block_id].append(candidate)
        branches = []
        for block_rank, block_id in enumerate(block_order[: args.branch_blocks], start=1):
            spans = sorted(
                {
                    (int(candidate["start"]), int(candidate["end"]))
                    for candidate in grouped[block_id]
                }
            )
            branches.append(
                {
                    "rank": block_rank,
                    "block_rank": block_rank,
                    "span_rank": 1,
                    "block_id": block_id,
                    "start": min(start for start, _end in spans),
                    "end": max(end for _start, end in spans),
                    "segments": [[start, end] for start, end in spans],
                    "score": max(float(item["score"]) for item in grouped[block_id]),
                    "target_overlap": max(
                        float(item["target_overlap"]) for item in grouped[block_id]
                    ),
                }
            )
        output_rows.append({**row, "branch_candidates": branches})
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.rows_path),
        "selection_uses_gold": False,
        "branch_blocks": args.branch_blocks,
        "spans_per_block": args.spans_per_block,
        "steps": len(output_rows),
        "mean_branches": statistics.fmean(
            len(row["branch_candidates"]) for row in output_rows
        ),
        "branch_evidence_recall": statistics.fmean(
            any(float(item["target_overlap"]) >= 0.8 for item in row["branch_candidates"])
            for row in output_rows
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
