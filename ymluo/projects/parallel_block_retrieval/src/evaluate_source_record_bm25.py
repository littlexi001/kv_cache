from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from profile_real_qk import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 within the oracle source record.")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--block_scores", required=True)
    parser.add_argument("--budgets", default="1,2,3,8,16,39")
    args = parser.parse_args()

    queries = read_jsonl(Path(args.corpus_dir) / "queries.jsonl")
    scores = np.load(args.block_scores, mmap_mode="r")
    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    rows: list[dict[str, float | int]] = []
    for budget in budgets:
        any_hits: list[float] = []
        all_hits: list[float] = []
        fractions: list[float] = []
        reciprocal_ranks: list[float] = []
        for query_index, query in enumerate(queries):
            start = int(query["block_start"])
            end = start + int(query["block_count"])
            source_ids = np.arange(start, end, dtype=np.int64)
            order = np.lexsort((source_ids, -np.asarray(scores[query_index, start:end])))
            selected = source_ids[order[:budget]].tolist()
            gold = {int(item) for item in query.get("gold_block_ids", [])}
            hit_count = len(set(selected) & gold)
            ranks = [rank + 1 for rank, block_id in enumerate(selected) if block_id in gold]
            any_hits.append(float(hit_count > 0))
            all_hits.append(float(gold <= set(selected)))
            fractions.append(hit_count / max(1, len(gold)))
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        rows.append(
            {
                "budget": budget,
                "queries": len(queries),
                "any_evidence_recall": statistics.fmean(any_hits),
                "all_evidence_recall": statistics.fmean(all_hits),
                "evidence_fraction": statistics.fmean(fractions),
                "evidence_mrr": statistics.fmean(reciprocal_ranks),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
