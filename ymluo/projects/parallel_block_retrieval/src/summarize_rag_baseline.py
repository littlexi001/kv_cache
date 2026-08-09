from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strict-chain RAG retrieval and compare final answers."
    )
    parser.add_argument("--first_retrieval_rows_path", required=True)
    parser.add_argument("--bridge_generation_rows_path", required=True)
    parser.add_argument("--second_retrieval_rows_path", required=True)
    parser.add_argument("--support_rows_path", required=True)
    parser.add_argument("--baseline_support_rows_path", default="")
    parser.add_argument("--ranking_prefix", default="hybrid_rrf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[bool]) -> float:
    return statistics.fmean(values) if values else float("nan")


def selected_answer_hit(row: dict[str, Any]) -> bool:
    return bool(row["branch_target_hits"][int(row["yes_no_index"])])


def two_sided_exact_binomial(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def main() -> None:
    args = parse_args()
    first_rows = [
        row
        for row in read_jsonl(args.first_retrieval_rows_path)
        if str(row["split"]) == args.split and str(row["step_type"]) == "resolve_bridge"
    ]
    bridge_rows = read_jsonl(args.bridge_generation_rows_path)
    second_rows = read_jsonl(args.second_retrieval_rows_path)
    support_rows = read_jsonl(args.support_rows_path)
    if not (len(first_rows) == len(bridge_rows) == len(second_rows) == len(support_rows)):
        raise ValueError("strict-chain inputs do not contain the same number of queries")

    first = {int(row["query_id"]): row for row in first_rows}
    bridges = {int(row["query_id"]): row for row in bridge_rows}
    second = {int(row["query_id"]): row for row in second_rows}
    support = {int(row["query_id"]): row for row in support_rows}
    query_ids = sorted(first)
    if not all(set(group) == set(query_ids) for group in (bridges, second, support)):
        raise ValueError("strict-chain query IDs do not align")

    rank_field = f"{args.ranking_prefix}_rank"
    bridge_correct = {query_id: bool(bridges[query_id]["target_hit"]) for query_id in query_ids}
    second_hit = {
        query_id: 0 < int(second[query_id][rank_field]) <= 16 for query_id in query_ids
    }
    final_hit = {query_id: selected_answer_hit(support[query_id]) for query_id in query_ids}
    oracle_hit = {
        query_id: any(support[query_id]["branch_target_hits"]) for query_id in query_ids
    }

    summary: dict[str, Any] = {
        "queries": len(query_ids),
        "first_retrieval_top3_recall": mean(
            [0 < int(first[query_id][rank_field]) <= 3 for query_id in query_ids]
        ),
        "bridge_accuracy": mean([bridge_correct[query_id] for query_id in query_ids]),
        "second_retrieval_top16_recall": mean(
            [second_hit[query_id] for query_id in query_ids]
        ),
        "second_top16_given_bridge_correct": mean(
            [second_hit[q] for q in query_ids if bridge_correct[q]]
        ),
        "second_top16_given_bridge_wrong": mean(
            [second_hit[q] for q in query_ids if not bridge_correct[q]]
        ),
        "candidate_oracle_answer_accuracy": mean(
            [oracle_hit[query_id] for query_id in query_ids]
        ),
        "verifier_final_answer_accuracy": mean(
            [final_hit[query_id] for query_id in query_ids]
        ),
        "final_given_second_retrieval_hit": mean(
            [final_hit[q] for q in query_ids if second_hit[q]]
        ),
        "final_given_second_retrieval_miss": mean(
            [final_hit[q] for q in query_ids if not second_hit[q]]
        ),
    }

    if args.baseline_support_rows_path:
        baseline_rows = {
            int(row["query_id"]): row
            for row in read_jsonl(args.baseline_support_rows_path)
        }
        if not set(query_ids).issubset(baseline_rows):
            raise ValueError("baseline is missing strict-chain query IDs")
        baseline_hit = {
            query_id: selected_answer_hit(baseline_rows[query_id])
            for query_id in query_ids
        }
        wins = sum(final_hit[q] and not baseline_hit[q] for q in query_ids)
        losses = sum(baseline_hit[q] and not final_hit[q] for q in query_ids)
        summary["paired_baseline"] = {
            "baseline_accuracy": mean([baseline_hit[q] for q in query_ids]),
            "new_accuracy": mean([final_hit[q] for q in query_ids]),
            "absolute_delta": mean([final_hit[q] for q in query_ids])
            - mean([baseline_hit[q] for q in query_ids]),
            "wins": wins,
            "losses": losses,
            "ties": len(query_ids) - wins - losses,
            "mcnemar_exact_p": two_sided_exact_binomial(wins, losses),
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
