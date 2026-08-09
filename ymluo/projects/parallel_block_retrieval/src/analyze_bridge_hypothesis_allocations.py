from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select bridge-hypothesis block allocations on dev and freeze on test."
    )
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--budgets", default="3,6,9,16")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def allocated_candidates(
    rankings: Sequence[Sequence[int]],
    branch_order: Sequence[int],
    allocation: Sequence[int],
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for score_rank, count in enumerate(allocation):
        if count <= 0:
            continue
        branch_index = int(branch_order[score_rank])
        added = 0
        for value in rankings[branch_index]:
            block_id = int(value)
            if block_id in seen:
                continue
            seen.add(block_id)
            output.append(block_id)
            added += 1
            if added >= count:
                break
    return output


def monotonic_allocations(budget: int, branches: int = 3) -> list[tuple[int, ...]]:
    return [
        values
        for values in product(range(budget + 1), repeat=branches)
        if sum(values) == budget
        and all(values[index] >= values[index + 1] for index in range(branches - 1))
    ]


def hits_for_allocation(
    rows: Sequence[dict[str, Any]], allocation: Sequence[int]
) -> list[bool]:
    hits = []
    for row in rows:
        candidates = allocated_candidates(
            row["hypothesis_candidates"], row["branch_order"], allocation
        )
        hits.append(int(row["target_block_id"]) in candidates)
    return hits


def mean(values: Sequence[bool]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    args = parse_args()
    dev_rows = read_jsonl(Path(args.dev_rows_path))
    test_rows = read_jsonl(Path(args.test_rows_path))
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    results = []
    for budget in budgets:
        candidates = monotonic_allocations(budget)
        scored = []
        for allocation in candidates:
            dev_hits = hits_for_allocation(dev_rows, allocation)
            scored.append((mean(dev_hits), allocation, dev_hits))
        # Prefer concentrating work when dev recall ties exactly.
        dev_recall, allocation, _dev_hits = max(
            scored,
            key=lambda item: (
                item[0],
                item[1][0],
                item[1][1],
                item[1][2],
            ),
        )
        test_hits = hits_for_allocation(test_rows, allocation)
        selected_hits = [
            0 < int(row["selected_rank"]) <= budget for row in test_rows
        ]
        round_robin_hits = [
            0 < int(row["round_robin_rank"]) <= budget for row in test_rows
        ]
        wins = sum(new and not old for new, old in zip(test_hits, selected_hits))
        losses = sum(old and not new for new, old in zip(test_hits, selected_hits))
        results.append(
            {
                "budget": budget,
                "candidate_allocations": len(candidates),
                "dev_selected_recall": mean(
                    [0 < int(row["selected_rank"]) <= budget for row in dev_rows]
                ),
                "dev_round_robin_recall": mean(
                    [0 < int(row["round_robin_rank"]) <= budget for row in dev_rows]
                ),
                "dev_selected_allocation_recall": dev_recall,
                "selected_allocation": list(allocation),
                "test_selected_recall": mean(selected_hits),
                "test_round_robin_recall": mean(round_robin_hits),
                "test_selected_allocation_recall": mean(test_hits),
                "test_wins_losses_vs_selected": [wins, losses],
                "test_mcnemar_p_vs_selected": mcnemar_exact_p(wins, losses),
            }
        )
    payload = {
        "source": "dev-selected monotonic block allocation over verifier-ranked states",
        "selection_uses_test_labels": False,
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "results": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
