from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine parallel bridge-state candidate pools for sparse K profiling."
    )
    parser.add_argument("--rows_paths", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_limit", type=int, default=16)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    paths = [Path(item.strip()) for item in args.rows_paths.split(",") if item.strip()]
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append(
                {
                    "query_id": int(row["query_id"]),
                    "step_index": 1,
                    "split": str(row.get("split", "unknown")),
                    "step_type": "resolve_answer_from_bridge",
                    "selection_uses_gold": False,
                    "parallel_candidates": [
                        int(item)
                        for item in row["round_robin_candidates"][
                            : args.candidate_limit
                        ]
                    ],
                }
            )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output_path": str(output_path)}))


if __name__ == "__main__":
    main()
