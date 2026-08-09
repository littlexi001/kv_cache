from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all-head block rankings into full-block generation branches."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--allhead_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="selected")
    parser.add_argument(
        "--ranking_field",
        default="",
        help="Optional explicit ranked block field, such as lexical_candidates.",
    )
    parser.add_argument("--branch_blocks", type=int, default=3)
    parser.add_argument("--block_tokens", type=int, default=256)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_rows(
    steps: Sequence[dict[str, Any]],
    allhead_rows: Sequence[dict[str, Any]],
    *,
    method: str,
    ranking_field: str,
    branch_blocks: int,
    block_tokens: int,
) -> list[dict[str, Any]]:
    step_by_key = {
        (int(step["query_id"]), int(step["step_index"])): step for step in steps
    }
    rows = []
    for source in allhead_rows:
        key = (int(source["query_id"]), int(source["step_index"]))
        step = step_by_key[key]
        source_field = ranking_field or f"{method}_top16"
        ranked = [int(item) for item in source[source_field]]
        target = int(step["target_block_ids"][0])
        candidates = []
        for rank, block_id in enumerate(ranked[:branch_blocks], start=1):
            candidates.append(
                {
                    "rank": rank,
                    "block_rank": rank,
                    "span_rank": 1,
                    "block_id": block_id,
                    "start": 0,
                    "end": block_tokens,
                    "score": float(branch_blocks - rank + 1),
                    "target_overlap": 1.0 if block_id == target else 0.0,
                }
            )
        rows.append(
            {
                "query_id": key[0],
                "step_index": key[1],
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "selection_uses_gold": False,
                "candidate_hit": target in ranked,
                "candidate_target_rank": ranked.index(target) + 1 if target in ranked else 0,
                "branch_target_span_rank": (
                    next(
                        (
                            rank
                            for rank, block_id in enumerate(
                                ranked[:branch_blocks], start=1
                            )
                            if block_id == target
                        ),
                        0,
                    )
                ),
                "branch_candidates": candidates,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.branch_blocks <= 0 or args.block_tokens <= 0:
        raise ValueError("branch and block sizes must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(
        read_jsonl(Path(args.step_queries_path)),
        read_jsonl(Path(args.allhead_rows_path)),
        method=args.method,
        ranking_field=args.ranking_field,
        branch_blocks=args.branch_blocks,
        block_tokens=args.block_tokens,
    )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups = []
    for split, step_type in sorted(
        {(row["split"], row["step_type"]) for row in rows}
    ):
        group = [
            row for row in rows if row["split"] == split and row["step_type"] == step_type
        ]
        groups.append(
            {
                "split": split,
                "step_type": step_type,
                "steps": len(group),
                "candidate_recall_at_16": statistics.fmean(
                    0 < row["candidate_target_rank"] <= 16 for row in group
                ),
                "branch_block_recall": statistics.fmean(
                    row["branch_target_span_rank"] > 0 for row in group
                ),
            }
        )
    summary = {
        "source": "full-block branches from frozen all-head candidate ranking",
        "selection_uses_gold": False,
        "method": args.method,
        "ranking_field": args.ranking_field or f"{args.method}_top16",
        "branch_blocks": args.branch_blocks,
        "steps": len(rows),
        "summaries": groups,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
