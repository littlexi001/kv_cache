from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop complete queries with any failed block-contained target span audit."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--queries_path", required=True)
    parser.add_argument("--audit_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit_path).read_text(encoding="utf-8"))
    excluded = {int(row["query_id"]) for row in audit["failures"]}
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if int(row["query_id"]) not in excluded
    ]
    queries = [
        row
        for row in read_jsonl(Path(args.queries_path))
        if int(row["query_id"]) not in excluded
    ]
    if len(steps) != 2 * len(queries):
        raise RuntimeError("audited two-hop query set is incomplete")
    write_jsonl(output_dir / "step_queries.jsonl", steps)
    write_jsonl(output_dir / "queries.jsonl", queries)
    split_counts = Counter(str(row["split"]) for row in queries)
    summary = {
        "source_step_queries": args.step_queries_path,
        "source_queries": args.queries_path,
        "audit_path": args.audit_path,
        "excluded_query_ids": sorted(excluded),
        "excluded_queries": len(excluded),
        "queries": len(queries),
        "steps": len(steps),
        "split_queries": dict(split_counts),
        "target_span_failures_after_filter": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
