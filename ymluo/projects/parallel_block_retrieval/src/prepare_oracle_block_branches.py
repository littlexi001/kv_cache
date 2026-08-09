from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one gold-support block branch to measure reader upper bounds."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument(
        "--support_paragraph_only",
        action="store_true",
        help="Restrict the oracle branch to the aligned official support paragraph.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in read_jsonl(Path(args.step_queries_path)):
        block_id = int(step["target_block_ids"][0])
        end = (
            min(args.block_tokens, int(step["support_paragraph_tokens"]))
            if args.support_paragraph_only
            else args.block_tokens
        )
        rows.append(
            {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "selection_uses_gold": True,
                "branch_candidates": [
                    {
                        "rank": 1,
                        "block_rank": 1,
                        "span_rank": 1,
                        "block_id": block_id,
                        "start": 0,
                        "end": end,
                        "score": 1.0,
                        "target_overlap": 1.0,
                    }
                ],
            }
        )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": (
            "official gold support-paragraph reader upper bound"
            if args.support_paragraph_only
            else "official gold supporting block reader upper bound"
        ),
        "selection_uses_gold": True,
        "support_paragraph_only": args.support_paragraph_only,
        "steps": len(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
