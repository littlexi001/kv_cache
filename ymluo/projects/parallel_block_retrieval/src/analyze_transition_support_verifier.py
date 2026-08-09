from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select transition-support score fusion on dev and freeze on test."
    )
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / max(float(array.std()), 1.0e-8)


def fused_index(row: dict[str, Any], yes_weight: float, ll_weight: float) -> int:
    score = zscore(row["heuristic_scores"])
    score += yes_weight * zscore(row["yes_no_scores"])
    score += ll_weight * zscore(row["answer_logprob_scores"])
    return int(np.argmax(score))


def accuracy(
    rows: Sequence[dict[str, Any]], yes_weight: float, ll_weight: float
) -> float:
    return sum(
        bool(row["branch_target_hits"][fused_index(row, yes_weight, ll_weight)])
        for row in rows
    ) / max(1, len(rows))


def main() -> None:
    args = parse_args()
    dev_rows = read_jsonl(Path(args.dev_rows_path))
    test_rows = read_jsonl(Path(args.test_rows_path))
    weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    candidates = []
    for yes_weight, ll_weight in product(weights, repeat=2):
        candidates.append(
            (accuracy(dev_rows, yes_weight, ll_weight), yes_weight, ll_weight)
        )
    dev_accuracy, yes_weight, ll_weight = max(
        candidates,
        key=lambda item: (item[0], -item[1] - item[2], -item[1], -item[2]),
    )
    baseline_hits = [
        bool(row["branch_target_hits"][int(row["heuristic_index"])])
        for row in test_rows
    ]
    fused_hits = [
        bool(
            row["branch_target_hits"][fused_index(row, yes_weight, ll_weight)]
        )
        for row in test_rows
    ]
    wins = sum(new and not old for new, old in zip(fused_hits, baseline_hits))
    losses = sum(old and not new for new, old in zip(fused_hits, baseline_hits))
    payload = {
        "source": "dev-selected frozen transition-support score fusion",
        "selection_uses_test_labels": False,
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "selected_yes_weight": yes_weight,
        "selected_answer_logprob_weight": ll_weight,
        "dev_fused_accuracy": dev_accuracy,
        "test_heuristic_accuracy": sum(baseline_hits) / len(test_rows),
        "test_fused_accuracy": sum(fused_hits) / len(test_rows),
        "test_wins_losses": [wins, losses],
        "test_mcnemar_p": mcnemar_exact_p(wins, losses),
        "test_any_branch_accuracy": sum(row["any_branch_hit"] for row in test_rows)
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
