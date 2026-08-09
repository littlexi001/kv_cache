from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bridge-controller JSONL outputs to retrieval CSV."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Method name and results path as name=results.jsonl; may be repeated.",
    )
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    rows = []
    for specification in args.run:
        method, separator, raw_path = specification.partition("=")
        if not separator or not method or not raw_path:
            raise ValueError(f"invalid run specification: {specification!r}")
        for result in read_jsonl(Path(raw_path)):
            trace = result.get("search_trace", [])
            selected = (
                [int(item) for item in trace[-1]["selected_blocks"]]
                if trace
                else [int(item) for item in result["hop1_selected"]]
            )
            rows.append(
                {
                    "method": method,
                    "query_id": int(result["query_id"]),
                    "dataset": str(result["dataset"]),
                    "source_record_recall": float(
                        bool(result.get("any_search_record_hit"))
                    ),
                    "record_top1_recall": "",
                    "answer_block_recall": float(
                        bool(result.get("any_search_gold_hit"))
                    ),
                    "answer_block_mrr": "",
                    "gold_block_count": len(result["gold_block_ids"]),
                    "record_margin": "",
                    "selected_block_ids": json.dumps(selected),
                    "ranked_block_ids": json.dumps(selected),
                }
            )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
