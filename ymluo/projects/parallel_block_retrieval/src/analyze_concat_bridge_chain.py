from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize concat bridge generation and conditional second-hop recall."
    )
    parser.add_argument("--generation_rows_path", required=True)
    parser.add_argument("--bridge_traces_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--baseline_generation_rows_path", default="")
    parser.add_argument("--baseline_retrieval_rows_path", default="")
    parser.add_argument("--baseline_selection_rows_path", default="")
    parser.add_argument("--baseline_selection_field", default="heuristic_structured_index")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_or_zero(row: dict[str, Any]) -> int:
    for key in ("lexical_rank", "candidate_rank"):
        if key in row:
            return int(row[key])
    raise KeyError("retrieval row has no lexical/candidate rank")


def recall_summary(ranks: list[int]) -> dict[str, float | int]:
    return {
        "queries": len(ranks),
        "recall_at_1": statistics.fmean(0 < rank <= 1 for rank in ranks),
        "recall_at_3": statistics.fmean(0 < rank <= 3 for rank in ranks),
        "recall_at_16": statistics.fmean(0 < rank <= 16 for rank in ranks),
    }


def selected_hits(
    generation_rows: list[dict[str, Any]],
    selection_by_query: dict[int, int] | None = None,
) -> dict[int, bool]:
    output = {}
    for row in generation_rows:
        query_id = int(row["query_id"])
        index = selection_by_query[query_id] if selection_by_query else 0
        output[query_id] = bool(row["branches"][index]["target_hit"])
    return output


def main() -> None:
    args = parse_args()
    generations = read_jsonl(Path(args.generation_rows_path))
    traces = {
        int(row["query_id"]): bool(row["bridge_target_hit"])
        for row in read_jsonl(Path(args.bridge_traces_path))
    }
    retrieval = {
        int(row["query_id"]): rank_or_zero(row)
        for row in read_jsonl(Path(args.retrieval_rows_path))
    }
    query_ids = sorted(traces)
    if set(query_ids) != set(retrieval):
        raise ValueError("bridge traces and second-hop retrieval do not align")
    bridge_hits = [traces[query_id] for query_id in query_ids]
    correct_ranks = [retrieval[q] for q in query_ids if traces[q]]
    incorrect_ranks = [retrieval[q] for q in query_ids if not traces[q]]
    payload: dict[str, Any] = {
        "source": "Top3 concatenated bridge reader with leak-free second-hop retrieval",
        "queries": len(query_ids),
        "first_retrieval_recall_at_3": statistics.fmean(
            bool(row["retrieval_target_span_hit_at_k"]) for row in generations
        ),
        "concat_bridge_accuracy": statistics.fmean(bridge_hits),
        "second_hop": {
            "overall": recall_summary([retrieval[q] for q in query_ids]),
            "given_correct_bridge": recall_summary(correct_ranks),
            "given_incorrect_bridge": recall_summary(incorrect_ranks),
        },
    }
    if args.baseline_generation_rows_path:
        baseline_rows = read_jsonl(Path(args.baseline_generation_rows_path))
        if args.baseline_selection_rows_path:
            selection = {
                int(row["query_id"]): int(row[args.baseline_selection_field])
                for row in read_jsonl(Path(args.baseline_selection_rows_path))
            }
        else:
            selection = None
        baseline = selected_hits(baseline_rows, selection)
        concat = selected_hits(generations)
        if set(baseline) != set(concat):
            raise ValueError("baseline and concat generation queries do not align")
        wins = sum(concat[q] and not baseline[q] for q in concat)
        losses = sum(baseline[q] and not concat[q] for q in concat)
        payload["paired_bridge_vs_baseline"] = {
            "baseline_accuracy": statistics.fmean(baseline.values()),
            "concat_accuracy": statistics.fmean(concat.values()),
            "wins": wins,
            "losses": losses,
            "mcnemar_p": mcnemar_exact_p(wins, losses),
        }
    if args.baseline_retrieval_rows_path:
        baseline_ranks = {
            int(row["query_id"]): rank_or_zero(row)
            for row in read_jsonl(Path(args.baseline_retrieval_rows_path))
        }
        if set(baseline_ranks) != set(retrieval):
            raise ValueError("baseline and concat retrieval queries do not align")
        paired_retrieval = {}
        for budget in (1, 3, 16):
            baseline_hits = {
                query_id: 0 < rank <= budget
                for query_id, rank in baseline_ranks.items()
            }
            concat_hits = {
                query_id: 0 < rank <= budget
                for query_id, rank in retrieval.items()
            }
            wins = sum(
                concat_hits[query_id] and not baseline_hits[query_id]
                for query_id in retrieval
            )
            losses = sum(
                baseline_hits[query_id] and not concat_hits[query_id]
                for query_id in retrieval
            )
            paired_retrieval[f"recall_at_{budget}"] = {
                "baseline": statistics.fmean(baseline_hits.values()),
                "concat": statistics.fmean(concat_hits.values()),
                "wins": wins,
                "losses": losses,
                "mcnemar_p": mcnemar_exact_p(wins, losses),
            }
        payload["paired_second_hop_vs_baseline"] = paired_retrieval
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
