from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover every eligible query stored in an existing real corpus."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--datasets", default="2wikimqa")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def materialize_query_pool(
    records: Sequence[dict[str, Any]], datasets: set[str]
) -> list[dict[str, Any]]:
    source_cache: dict[str, list[dict[str, Any]]] = {}
    queries = []
    for record in records:
        if str(record["dataset"]) not in datasets:
            continue
        if not record.get("question") or not record.get("answers"):
            continue
        if not record.get("gold_block_ids"):
            continue
        source_file = str(record["source_file"])
        if source_file not in source_cache:
            source_cache[source_file] = read_jsonl(Path(source_file))
        source_index = int(record["source_index"])
        source_row = source_cache[source_file][source_index]
        context = str(source_row.get("context", ""))
        if not context:
            continue
        queries.append({**record, "query_id": len(queries), "context": context})
    return queries


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    queries = materialize_query_pool(read_jsonl(corpus_dir / "records.jsonl"), datasets)
    with output_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")
    summary = {
        "source": str(corpus_dir / "records.jsonl"),
        "datasets": sorted(datasets),
        "queries": len(queries),
        "selection_uses_answer": False,
        "eligibility_uses_answer_occurrence_for_evaluation": True,
        "output_path": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
