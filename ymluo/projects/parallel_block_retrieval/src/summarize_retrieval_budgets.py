from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from profile_real_qk import read_jsonl
from run_lexical_block_retrieval import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generic retrieval CSVs by budget.")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--retrieval_result",
        action="append",
        required=True,
        help="Named query_results.csv as label=/path; may be repeated.",
    )
    parser.add_argument("--budgets", default="1,2,3")
    return parser.parse_args()


def named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("retrieval_result must be label=/path")
    label, path = spec.split("=", maxsplit=1)
    return label, Path(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.corpus_dir) / "queries.jsonl")
    }
    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    rows: list[dict[str, Any]] = []
    for spec in args.retrieval_result:
        label, path = named_path(spec)
        with path.open("r", encoding="utf-8", newline="") as f:
            for result in csv.DictReader(f):
                query = queries[int(result["query_id"])]
                ranked = [int(item) for item in json.loads(result["ranked_block_ids"])]
                gold = {int(item) for item in query.get("gold_block_ids", [])}
                for budget in budgets:
                    selected = ranked[:budget]
                    hits = [rank + 1 for rank, block_id in enumerate(selected) if block_id in gold]
                    start = int(query["block_start"])
                    end = start + int(query["block_count"])
                    rows.append(
                        {
                            "method": f"{label}:{result['method']}",
                            "budget": budget,
                            "query_id": int(query["query_id"]),
                            "selected_blocks": len(selected),
                            "gold_blocks": len(gold),
                            "evidence_hits": len(set(selected) & gold),
                            "any_evidence_recall": float(bool(hits)),
                            "all_evidence_recall": float(gold <= set(selected)),
                            "evidence_fraction": len(set(selected) & gold) / max(1, len(gold)),
                            "evidence_mrr": 1.0 / min(hits) if hits else 0.0,
                            "source_record_recall": float(
                                any(start <= block_id < end for block_id in selected)
                            ),
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        for budget in budgets:
            group = [
                row for row in rows if row["method"] == method and row["budget"] == budget
            ]
            if not group:
                continue
            summary_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "queries": len(group),
                    "mean_selected_blocks": statistics.fmean(
                        int(row["selected_blocks"]) for row in group
                    ),
                    "any_evidence_recall": statistics.fmean(
                        float(row["any_evidence_recall"]) for row in group
                    ),
                    "all_evidence_recall": statistics.fmean(
                        float(row["all_evidence_recall"]) for row in group
                    ),
                    "evidence_fraction": statistics.fmean(
                        float(row["evidence_fraction"]) for row in group
                    ),
                    "evidence_mrr": statistics.fmean(
                        float(row["evidence_mrr"]) for row in group
                    ),
                    "source_record_recall": statistics.fmean(
                        float(row["source_record_recall"]) for row in group
                    ),
                }
            )
    write_csv(output_dir / "query_metrics.csv", rows, list(rows[0]))
    write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
