from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deeper block branches only for low-confidence transitions."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--selection_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="heuristic_structured")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--start_rank", type=int, default=4)
    parser.add_argument("--end_rank", type=int, default=6)
    parser.add_argument("--block_tokens", type=int, default=256)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def should_expand(selection: dict[str, Any], method: str, threshold: float) -> bool:
    return max(float(item) for item in selection[f"{method}_scores"]) <= threshold


def main() -> None:
    args = parse_args()
    if not 1 <= args.start_rank <= args.end_rank or args.block_tokens <= 0:
        raise ValueError("invalid extension rank range or block size")
    steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.step_queries_path))
    }
    retrieval = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.retrieval_rows_path))
    }
    selections = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.selection_rows_path))
    }
    if set(retrieval) != set(selections):
        raise ValueError("retrieval and selection rows do not align")
    expanded_ids = [
        query_id
        for query_id in sorted(selections)
        if should_expand(selections[query_id], args.method, args.threshold)
    ]
    rows = []
    for query_id in expanded_ids:
        step = steps[query_id]
        source = retrieval[query_id]
        ranked = [int(item) for item in source["lexical_candidates"]]
        target = int(step["target_block_ids"][0])
        candidates = []
        for rank in range(args.start_rank, args.end_rank + 1):
            block_id = ranked[rank - 1]
            candidates.append(
                {
                    "rank": rank,
                    "block_rank": rank,
                    "span_rank": 1,
                    "block_id": block_id,
                    "start": 0,
                    "end": args.block_tokens,
                    "score": float(args.end_rank - rank + 1),
                    "target_overlap": 1.0 if block_id == target else 0.0,
                }
            )
        rows.append(
            {
                "query_id": query_id,
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "selection_uses_gold": False,
                "branch_candidates": candidates,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    excluded = sorted(set(selections) - set(expanded_ids))
    (output_dir / "excluded_query_ids.txt").write_text(
        ",".join(str(item) for item in excluded), encoding="utf-8"
    )
    summary = {
        "source": "confidence-gated deeper lexical block branches",
        "selection_uses_gold": False,
        "method": args.method,
        "threshold": args.threshold,
        "queries": len(selections),
        "expanded_queries": len(expanded_ids),
        "expansion_fraction": len(expanded_ids) / len(selections),
        "start_rank": args.start_rank,
        "end_rank": args.end_rank,
        "extra_blocks_per_expanded_query": args.end_rank - args.start_rank + 1,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
