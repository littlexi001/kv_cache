from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from profile_real_qk import read_jsonl
from run_lexical_block_retrieval import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate controlled synthetic evidence and hard-negative retrieval metrics."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--retrieval_result",
        action="append",
        default=[],
        help="Named CSV in the form label=/path/to/query_results.csv; may be repeated.",
    )
    parser.add_argument("--allhead_topk_npz")
    parser.add_argument("--target_blocks", type=int, default=39)
    parser.add_argument(
        "--budgets",
        default="",
        help="Optional comma-separated final budgets, for example 1,4,8,16,39.",
    )
    return parser.parse_args()


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("retrieval_result must use label=/path/to/query_results.csv")
    label, path = spec.split("=", maxsplit=1)
    return label.strip(), Path(path)


def evaluate_ids(
    *,
    method: str,
    query: dict[str, Any],
    selected_ids: list[int],
    result_kind: str,
) -> dict[str, Any]:
    selected = set(selected_ids)
    gold = set(int(item) for item in query["gold_block_ids"])
    negatives = set(int(item) for item in query.get("hard_negative_block_ids", []))
    evidence_hits = len(selected & gold)
    negative_hits = len(selected & negatives)
    start = int(query["block_start"])
    end = start + int(query["block_count"])
    return {
        "method": method,
        "result_kind": result_kind,
        "query_id": int(query["query_id"]),
        "task_type": str(query["task_type"]),
        "split": str(query["split"]),
        "selected_blocks": len(selected),
        "gold_blocks": len(gold),
        "evidence_hits": evidence_hits,
        "any_evidence_recall": float(evidence_hits > 0),
        "all_evidence_recall": float(evidence_hits == len(gold)),
        "evidence_fraction": evidence_hits / len(gold),
        "hard_negative_hits": negative_hits,
        "hard_negative_hit_rate": float(negative_hits > 0),
        "hard_negative_fraction": negative_hits / max(1, len(negatives)),
        "source_record_recall": float(any(start <= item < end for item in selected)),
        "question_evidence_lexical_jaccard": float(
            query["question_evidence_lexical_jaccard"]
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        tasks = ["all", *sorted({str(row["task_type"]) for row in method_rows})]
        splits = ["all", *sorted({str(row["split"]) for row in method_rows})]
        for task in tasks:
            for split in splits:
                group = [
                    row
                    for row in method_rows
                    if (task == "all" or row["task_type"] == task)
                    and (split == "all" or row["split"] == split)
                ]
                if not group:
                    continue
                output.append(
                    {
                        "method": method,
                        "result_kind": group[0]["result_kind"],
                        "task_type": task,
                        "split": split,
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
                        "hard_negative_hit_rate": statistics.fmean(
                            float(row["hard_negative_hit_rate"]) for row in group
                        ),
                        "mean_hard_negative_hits": statistics.fmean(
                            int(row["hard_negative_hits"]) for row in group
                        ),
                        "source_record_recall": statistics.fmean(
                            float(row["source_record_recall"]) for row in group
                        ),
                        "mean_lexical_jaccard": statistics.fmean(
                            float(row["question_evidence_lexical_jaccard"])
                            for row in group
                        ),
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    query_by_id = {int(row["query_id"]): row for row in queries}
    budgets = (
        sorted({int(item) for item in args.budgets.split(",") if item.strip()})
        if args.budgets
        else [args.target_blocks]
    )
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must be positive")
    rows: list[dict[str, Any]] = []

    for spec in args.retrieval_result:
        label, path = parse_named_path(spec)
        with path.open("r", encoding="utf-8", newline="") as f:
            for result in csv.DictReader(f):
                query = query_by_id[int(result["query_id"])]
                ranked = [int(item) for item in json.loads(result["ranked_block_ids"])]
                for budget in budgets:
                    rows.append(
                        evaluate_ids(
                            method=f"{label}:{result['method']}@{budget}",
                            query=query,
                            selected_ids=ranked[:budget],
                            result_kind="final_budget",
                        )
                    )

    if args.allhead_topk_npz:
        payload = np.load(args.allhead_topk_npz)
        block_ids = payload["block_ids"]
        if int(block_ids.shape[0]) != len(queries):
            raise ValueError("all-head query count does not match corpus")
        limits = sorted({1, 2, 4, 8, int(block_ids.shape[3])})
        for query_index, query in enumerate(queries):
            for limit in limits:
                candidates = sorted(
                    {
                        int(item)
                        for item in block_ids[query_index, :, :, :limit].reshape(-1)
                        if int(item) >= 0
                    }
                )
                rows.append(
                    evaluate_ids(
                        method=f"allhead:candidate_union_top{limit}",
                        query=query,
                        selected_ids=candidates,
                        result_kind="candidate_oracle",
                    )
                )

    if not rows:
        raise ValueError("no retrieval results were provided")
    summaries = aggregate(rows)
    write_csv(output_dir / "query_metrics.csv", rows, list(rows[0]))
    write_csv(output_dir / "summary.csv", summaries, list(summaries[0]))
    primary = [
        row for row in summaries if row["task_type"] == "all" and row["split"] == "all"
    ]
    summary = {
        "source": "controlled synthetic evidence evaluation",
        "queries": len(queries),
        "target_block_budgets": budgets,
        "methods": primary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
