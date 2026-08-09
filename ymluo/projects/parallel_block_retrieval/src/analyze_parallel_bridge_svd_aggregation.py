from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose a parallel-Q aggregation on dev and evaluate it on test."
    )
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall(rows: Sequence[dict[str, Any]], method: str, budget: int) -> float:
    return sum(
        0 < int(row["method_ranks"][method]) <= budget for row in rows
    ) / max(1, len(rows))


def conditional_mrr(rows: Sequence[dict[str, Any]], method: str) -> float:
    ranks = [int(row["method_ranks"][method]) for row in rows]
    reachable = [rank for rank in ranks if rank > 0]
    return sum(1.0 / rank for rank in reachable) / max(1, len(reachable))


def main() -> None:
    args = parse_args()
    dev_rows = read_jsonl(Path(args.dev_rows_path))
    test_rows = read_jsonl(Path(args.test_rows_path))
    methods = sorted(dev_rows[0]["method_ranks"])
    selected_method = max(
        methods,
        key=lambda method: (
            recall(dev_rows, method, 3),
            conditional_mrr(dev_rows, method),
            method,
        ),
    )
    baseline_hits = [
        0 < int(row["selected_bm25_rank"]) <= 3 for row in test_rows
    ]
    selected_hits = [
        0 < int(row["method_ranks"][selected_method]) <= 3 for row in test_rows
    ]
    wins = sum(new and not old for new, old in zip(selected_hits, baseline_hits))
    losses = sum(old and not new for new, old in zip(selected_hits, baseline_hits))
    payload = {
        "source": "dev-selected parallel bridge-state SVD aggregation",
        "selection_uses_test_labels": False,
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "dev_methods": {
            method: {
                "recall_at_3": recall(dev_rows, method, 3),
                "conditional_mrr": conditional_mrr(dev_rows, method),
            }
            for method in methods
        },
        "selected_method": selected_method,
        "test_selected_bm25_recall_at_3": sum(baseline_hits) / len(test_rows),
        "test_parallel_svd_recall_at_3": sum(selected_hits) / len(test_rows),
        "test_wins_losses": [wins, losses],
        "test_mcnemar_p": mcnemar_exact_p(wins, losses),
        "test_candidate_pool_recall_at_16": sum(
            0 < int(row["candidate_pool_rank"]) <= 16 for row in test_rows
        )
        / len(test_rows),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
