from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired Top3 and Top16 concatenated reader outputs by gold rank."
    )
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--top3_rows_path", required=True)
    parser.add_argument("--top16_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_rows(path: str) -> dict[int, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {
            int(row["query_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def target_hit(row: dict[str, Any]) -> bool:
    return bool(row["branches"][0]["target_hit"])


def paired_group(
    query_ids: list[int],
    top3: dict[int, dict[str, Any]],
    top16: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    old = {query_id: target_hit(top3[query_id]) for query_id in query_ids}
    new = {query_id: target_hit(top16[query_id]) for query_id in query_ids}
    wins = sum(new[q] and not old[q] for q in query_ids)
    losses = sum(old[q] and not new[q] for q in query_ids)
    return {
        "queries": len(query_ids),
        "top3_correct": sum(old.values()),
        "top3_accuracy": statistics.fmean(old.values()),
        "top16_correct": sum(new.values()),
        "top16_accuracy": statistics.fmean(new.values()),
        "wins": wins,
        "losses": losses,
        "mcnemar_p": mcnemar_exact_p(wins, losses),
    }


def main() -> None:
    args = parse_args()
    retrieval = read_rows(args.retrieval_rows_path)
    top3 = read_rows(args.top3_rows_path)
    top16 = read_rows(args.top16_rows_path)
    if not set(retrieval) == set(top3) == set(top16):
        raise ValueError("retrieval and generation rows do not align")
    categories: dict[str, Callable[[int], bool]] = {
        "gold_rank_1_3": lambda rank: 0 < rank <= 3,
        "gold_rank_4_16": lambda rank: 4 <= rank <= 16,
        "gold_rank_after_16_or_missing": lambda rank: not (0 < rank <= 16),
    }
    payload = {
        "source": "paired 8B Top3 versus Top16 concatenated final reader",
        "overall": paired_group(sorted(retrieval), top3, top16),
        "by_gold_rank": {
            name: paired_group(
                [
                    query_id
                    for query_id, row in retrieval.items()
                    if predicate(int(row["lexical_rank"]))
                ],
                top3,
                top16,
            )
            for name, predicate in categories.items()
        },
        "runtime": {
            "top3_mean_generation_seconds": statistics.fmean(
                float(row["branches"][0]["generation_seconds"])
                for row in top3.values()
            ),
            "top16_mean_generation_seconds": statistics.fmean(
                float(row["branches"][0]["generation_seconds"])
                for row in top16.values()
            ),
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
