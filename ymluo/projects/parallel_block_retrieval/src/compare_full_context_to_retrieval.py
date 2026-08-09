from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair full-context answer hits with the frozen 10M support selector."
    )
    parser.add_argument("--full_context_rows_path", required=True)
    parser.add_argument("--retrieval_scores_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * (0.5**discordant)
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    full_rows = read_jsonl(Path(args.full_context_rows_path))
    retrieval_rows = read_jsonl(Path(args.retrieval_scores_path))
    retrieval_hits = {}
    for row in retrieval_rows:
        selected_index = int(row["yes_no_index"])
        retrieval_hits[int(row["query_id"])] = bool(
            row["branch_target_hits"][selected_index]
        )

    full_by_length = {}
    for row in full_rows:
        full_by_length.setdefault(int(row["context_tokens"]), {})[
            int(row["query_id"])
        ] = bool(row["answer_hit"])
    expected_ids = set(retrieval_hits)
    comparisons = []
    for context_length, full_hits in sorted(full_by_length.items()):
        if set(full_hits) != expected_ids:
            raise ValueError(f"query IDs do not align for context {context_length}")
        retrieval_wins = sum(
            retrieval_hits[query_id] and not full_hits[query_id]
            for query_id in expected_ids
        )
        retrieval_losses = sum(
            full_hits[query_id] and not retrieval_hits[query_id]
            for query_id in expected_ids
        )
        both_correct = sum(
            full_hits[query_id] and retrieval_hits[query_id]
            for query_id in expected_ids
        )
        both_wrong = len(expected_ids) - retrieval_wins - retrieval_losses - both_correct
        comparisons.append(
            {
                "context_tokens": context_length,
                "queries": len(expected_ids),
                "full_context_accuracy": sum(full_hits.values()) / len(expected_ids),
                "retrieval_accuracy": sum(retrieval_hits.values()) / len(expected_ids),
                "retrieval_minus_full_pp": 100.0
                * (sum(retrieval_hits.values()) - sum(full_hits.values()))
                / len(expected_ids),
                "retrieval_wins": retrieval_wins,
                "retrieval_losses": retrieval_losses,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "mcnemar_exact_p": exact_mcnemar_pvalue(
                    retrieval_wins, retrieval_losses
                ),
            }
        )
    payload = {
        "source": "paired full-context versus 10M retrieval support selector",
        "retrieval_selection_uses_gold": False,
        "comparisons": comparisons,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
