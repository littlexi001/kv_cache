from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired complementarity of matched BM25 and dynamic KV retrieval."
    )
    parser.add_argument("--bm25_rows", required=True)
    parser.add_argument("--kv_rows", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bm25_blocks", type=int, default=3)
    parser.add_argument("--kv_blocks", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    bm25_rows = [
        row for row in read_jsonl(args.bm25_rows) if str(row["split"]) == args.split
    ]
    kv_rows = read_jsonl(args.kv_rows)
    kv_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row for row in kv_rows
    }
    if len(kv_by_key) != len(kv_rows):
        raise ValueError("duplicate KV query/step keys")

    rows = []
    for bm25 in bm25_rows:
        key = (int(bm25["query_id"]), int(bm25["step_index"]))
        if key not in kv_by_key:
            raise KeyError(f"KV rows have no matched key {key}")
        kv = kv_by_key[key]
        gold = {int(item) for item in kv["gold_block_ids"]}
        bm25_top3 = int(bm25["branch_target_span_rank"]) > 0
        bm25_top16 = int(bm25["candidate_target_rank"]) > 0
        kv_top4 = bool(gold & set(int(item) for item in kv["final_blocks"][: args.kv_blocks]))
        kv_top39 = bool(kv["rerank_hit39"])
        rows.append(
            {
                "query_id": key[0],
                "step_index": key[1],
                "bm25_top3": bm25_top3,
                "bm25_top16": bm25_top16,
                "kv_top4": kv_top4,
                "kv_top39": kv_top39,
                "hybrid_top3_top4": bm25_top3 or kv_top4,
                "hybrid_top16_top4": bm25_top16 or kv_top4,
            }
        )
    if len(rows) != len(kv_rows):
        raise ValueError("matched BM25/KV row counts differ")

    step_metrics = {}
    for step_index in sorted({row["step_index"] for row in rows}):
        subset = [row for row in rows if row["step_index"] == step_index]
        step_metrics[str(step_index)] = {
            "steps": len(subset),
            "bm25_top3_recall": mean(row["bm25_top3"] for row in subset),
            "bm25_top16_recall": mean(row["bm25_top16"] for row in subset),
            "kv_top4_recall": mean(row["kv_top4"] for row in subset),
            "kv_top39_recall": mean(row["kv_top39"] for row in subset),
            "hybrid_top3_top4_recall": mean(
                row["hybrid_top3_top4"] for row in subset
            ),
            "hybrid_top16_top4_recall": mean(
                row["hybrid_top16_top4"] for row in subset
            ),
            "kv_top4_rescues_bm25_top3": mean(
                row["kv_top4"] and not row["bm25_top3"] for row in subset
            ),
            "bm25_top3_rescues_kv_top4": mean(
                row["bm25_top3"] and not row["kv_top4"] for row in subset
            ),
            "bm25_top3_and_kv_top4": mean(
                row["bm25_top3"] and row["kv_top4"] for row in subset
            ),
        }

    by_query: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_query[row["query_id"]][row["step_index"]] = row
    if any(set(steps) != {0, 1} for steps in by_query.values()):
        raise ValueError("every query must have exactly two steps")
    both_steps = {
        method: mean(
            query[0][method] and query[1][method] for query in by_query.values()
        )
        for method in (
            "bm25_top3",
            "bm25_top16",
            "kv_top4",
            "kv_top39",
            "hybrid_top3_top4",
            "hybrid_top16_top4",
        )
    }
    summary = {
        "source": "paired matched-corpus BM25 and dynamic KV retrieval complementarity",
        "selection_uses_test_gold": False,
        "split": args.split,
        "steps": len(rows),
        "queries": len(by_query),
        "bm25_blocks": args.bm25_blocks,
        "kv_blocks": args.kv_blocks,
        "hybrid_max_blocks": args.bm25_blocks + args.kv_blocks,
        "hybrid_max_tokens_at_256": 256 * (args.bm25_blocks + args.kv_blocks),
        "step_metrics": step_metrics,
        "both_steps": both_steps,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
