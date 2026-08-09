from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from profile_real_qk import read_jsonl
from run_lexical_block_retrieval import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage-wise model-guided route failures.")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--failure_examples", type=int, default=12)
    return parser.parse_args()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.corpus_dir) / "queries.jsonl")
    }
    diagnostics = read_jsonl(Path(args.diagnostics))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for route in diagnostics:
        query = queries[int(route["query_id"])]
        gold = {int(item) for item in query["gold_block_ids"]}
        first_candidates = {
            int(item["block_id"]) for item in route.get("first_candidates", [])
        }
        ordered_first_candidates = [
            int(item["block_id"]) for item in route.get("first_candidates", [])
        ]
        first_gold_ranks = [
            rank + 1
            for rank, block_id in enumerate(ordered_first_candidates)
            if block_id in gold
        ]
        first_scores = [
            float(item["combined_score"]) for item in route.get("first_candidates", [])
        ]
        second_candidates = {
            int(item["block_id"]) for item in route.get("second_candidates", [])
        }
        candidate_union = first_candidates | second_candidates
        first = int(route["first_condition_block"])
        second = int(route["second_condition_block"])
        selected = {item for item in (first, second) if item >= 0}
        remaining_after_first = gold - {first}
        row = {
            "query_id": int(query["query_id"]),
            "task_type": str(query.get("task_type", "")),
            "split": str(query.get("split", "")),
            "gold_blocks": len(gold),
            "first_candidate_oracle": float(bool(first_candidates & gold)),
            "first_candidate_all_oracle": float(gold <= first_candidates),
            "first_top2_oracle": float(bool(set(ordered_first_candidates[:2]) & gold)),
            "first_top3_oracle": float(bool(set(ordered_first_candidates[:3]) & gold)),
            "first_gold_rank": min(first_gold_ranks) if first_gold_ranks else 0,
            "first_score_margin": (
                first_scores[0] - first_scores[1] if len(first_scores) > 1 else float("inf")
            ),
            "first_selected_gold": float(first in gold),
            "completion_margin": float(route["completion_margin"]),
            "followup_nonempty": float(bool(str(route.get("followup_query", "")).strip())),
            "second_candidate_oracle": float(
                bool(remaining_after_first) and bool(second_candidates & remaining_after_first)
            ),
            "candidate_union_oracle": float(bool(candidate_union & gold)),
            "candidate_union_all_oracle": float(gold <= candidate_union),
            "second_selected_remaining_gold": float(second in remaining_after_first),
            "route_any_evidence": float(bool(selected & gold)),
            "route_all_evidence": float(gold <= selected),
        }
        rows.append(row)
        if not row["route_all_evidence"] and len(failures) < args.failure_examples:
            failures.append(
                {
                    **row,
                    "question": query["question"],
                    "evidence_texts": query.get("evidence_texts", []),
                    "gold_block_ids": sorted(gold),
                    "first_premise": route.get("first_premise", ""),
                    "followup_query": route.get("followup_query", ""),
                    "first_condition_block": first,
                    "second_condition_block": second,
                    "first_candidate_blocks": sorted(first_candidates),
                    "second_candidate_blocks": sorted(second_candidates),
                }
            )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("all", "all")].append(row)
        groups[(str(row["task_type"]), "all")].append(row)
        groups[("all", str(row["split"]))].append(row)
        groups[(str(row["task_type"]), str(row["split"]))].append(row)
    metric_keys = [
        "first_candidate_oracle",
        "first_candidate_all_oracle",
        "first_top2_oracle",
        "first_top3_oracle",
        "first_gold_rank",
        "first_score_margin",
        "first_selected_gold",
        "completion_margin",
        "followup_nonempty",
        "second_candidate_oracle",
        "candidate_union_oracle",
        "candidate_union_all_oracle",
        "second_selected_remaining_gold",
        "route_any_evidence",
        "route_all_evidence",
    ]
    summary_rows = [
        {
            "task_type": task,
            "split": split,
            "queries": len(group),
            **{key: mean(group, key) for key in metric_keys},
        }
        for (task, split), group in sorted(groups.items())
    ]
    write_csv(output_dir / "route_rows.csv", rows, list(rows[0]))
    write_csv(output_dir / "route_summary.csv", summary_rows, list(summary_rows[0]))
    with (output_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    primary = [row for row in summary_rows if row["split"] == "all"]
    print(json.dumps(primary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
