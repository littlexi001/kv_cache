from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired BM25 versus E5 LongMemEval analysis.")
    parser.add_argument("--bm25_summary", required=True)
    parser.add_argument("--bm25_rows", required=True)
    parser.add_argument("--e5_summary", required=True)
    parser.add_argument("--e5_rows", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def paired_binary(
    before: dict[int, dict[str, Any]],
    after: dict[int, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    query_ids = sorted(set(before) & set(after))
    wins = sum(not bool(before[qid][metric]) and bool(after[qid][metric]) for qid in query_ids)
    losses = sum(bool(before[qid][metric]) and not bool(after[qid][metric]) for qid in query_ids)
    return {
        "queries": len(query_ids),
        "e5_wins": wins,
        "e5_losses": losses,
        "ties": len(query_ids) - wins - losses,
        "two_sided_binomial_p": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    bm25_summary = read_json(args.bm25_summary)
    e5_summary = read_json(args.e5_summary)
    bm25_rows = read_jsonl(args.bm25_rows)
    e5_rows = read_jsonl(args.e5_rows)
    bm25_by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    e5_by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in bm25_rows:
        bm25_by_method[str(row["method"])].append(row)
    for row in e5_rows:
        e5_by_method[str(row["method"])].append(row)
    bm25_quality = {row["method"]: row for row in bm25_summary["quality"]}
    e5_quality = {row["method"]: row for row in e5_summary["quality"]}

    pairs = [("global_block_bm25", "e5_global_block")]
    for depth in (1, 3, 8, 16, 32):
        pairs.extend(
            [
                (
                    f"session_router{depth}_block_bm25",
                    f"e5_global_session{depth}_block",
                ),
                (
                    f"owner_metadata_session{depth}_block_bm25",
                    f"e5_owner_metadata_session{depth}_block",
                ),
            ]
        )

    comparisons = []
    by_type = []
    for bm25_method, e5_method in pairs:
        if bm25_method not in bm25_by_method or e5_method not in e5_by_method:
            continue
        bm25_positive = {
            int(row["query_id"]): row
            for row in bm25_by_method[bm25_method]
            if not row["is_abstention"]
        }
        e5_positive = {
            int(row["query_id"]): row
            for row in e5_by_method[e5_method]
            if not row["is_abstention"]
        }
        ids = sorted(set(bm25_positive) & set(e5_positive))
        bm25_item = bm25_quality[bm25_method]
        e5_item = e5_quality[e5_method]
        comparisons.append(
            {
                "bm25_method": bm25_method,
                "e5_method": e5_method,
                "queries": len(ids),
                "bm25_mean_candidate_blocks": bm25_item["mean_candidate_blocks"],
                "e5_mean_candidate_blocks": e5_item["mean_candidate_blocks"],
                "bm25_mean_query_seconds": bm25_item["mean_query_seconds"],
                "e5_mean_query_seconds": e5_item["mean_query_seconds"],
                "bm25_exact_block_any_at_8": bm25_item["exact_block_any_at_8"],
                "e5_exact_block_any_at_8": e5_item["exact_block_any_at_8"],
                "exact_block_any_at_8_delta": (
                    e5_item["exact_block_any_at_8"]
                    - bm25_item["exact_block_any_at_8"]
                ),
                "bm25_latest_exact_block_any_at_8": bm25_item[
                    "latest_exact_block_any_at_8"
                ],
                "e5_latest_exact_block_any_at_8": e5_item[
                    "latest_exact_block_any_at_8"
                ],
                "latest_exact_block_any_at_8_delta": (
                    e5_item["latest_exact_block_any_at_8"]
                    - bm25_item["latest_exact_block_any_at_8"]
                ),
                "bm25_all_evidence_sessions_at_8": bm25_item[
                    "all_evidence_sessions_at_8"
                ],
                "e5_all_evidence_sessions_at_8": e5_item[
                    "all_evidence_sessions_at_8"
                ],
                "all_evidence_sessions_at_8_delta": (
                    e5_item["all_evidence_sessions_at_8"]
                    - bm25_item["all_evidence_sessions_at_8"]
                ),
                "paired_exact_block_any_at_8": paired_binary(
                    bm25_positive, e5_positive, "exact_block_any_at_8"
                ),
                "paired_latest_exact_block_any_at_8": paired_binary(
                    bm25_positive, e5_positive, "latest_exact_block_any_at_8"
                ),
                "paired_all_evidence_sessions_at_8": paired_binary(
                    bm25_positive, e5_positive, "all_evidence_sessions_at_8"
                ),
                "mean_evidence_session_recall_at_8_delta": mean(
                    float(e5_positive[qid]["evidence_session_recall_at_8"])
                    - float(bm25_positive[qid]["evidence_session_recall_at_8"])
                    for qid in ids
                ),
            }
        )
        for question_type in sorted(
            {str(bm25_positive[qid]["question_type"]) for qid in ids}
        ):
            typed_ids = [
                qid
                for qid in ids
                if bm25_positive[qid]["question_type"] == question_type
            ]
            by_type.append(
                {
                    "bm25_method": bm25_method,
                    "e5_method": e5_method,
                    "question_type": question_type,
                    "queries": len(typed_ids),
                    "bm25_exact_block_any_at_8": mean(
                        float(bm25_positive[qid]["exact_block_any_at_8"])
                        for qid in typed_ids
                    ),
                    "e5_exact_block_any_at_8": mean(
                        float(e5_positive[qid]["exact_block_any_at_8"])
                        for qid in typed_ids
                    ),
                }
            )

    output = {
        "source": "paired LongMemEval shared-10M BM25 versus E5 dense-RAG analysis",
        "protocol_notes": {
            "positive_queries": 48,
            "final_top_blocks": 8,
            "final_working_set_tokens": 512,
            "e5_model": e5_summary["embedding_model"],
            "e5_session_max_length": 512,
            "e5_is_lightweight_baseline_not_strongest_rag": True,
        },
        "e5_offline_indexing": e5_summary["offline_indexing"],
        "comparisons": comparisons,
        "exact_block_quality_by_type": by_type,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
