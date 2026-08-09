from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a confidence-gated extension with the frozen Top3 policy."
    )
    parser.add_argument("--base_generation_rows_path", required=True)
    parser.add_argument("--base_selection_rows_path", required=True)
    parser.add_argument("--adaptive_generation_rows_path", required=True)
    parser.add_argument("--adaptive_selection_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--method", default="heuristic_structured")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["query_id"]): row for row in read_jsonl(Path(path))}


def selected_hits(
    generation: dict[int, dict[str, Any]],
    selections: dict[int, dict[str, Any]],
    ids: Sequence[int],
    method: str,
) -> list[bool]:
    field = f"{method}_index"
    return [
        bool(generation[q]["branches"][int(selections[q][field])]["target_hit"])
        for q in ids
    ]


def main() -> None:
    args = parse_args()
    base_generation = keyed(args.base_generation_rows_path)
    base_selections = keyed(args.base_selection_rows_path)
    adaptive_generation = keyed(args.adaptive_generation_rows_path)
    adaptive_selections = keyed(args.adaptive_selection_rows_path)
    ids = sorted(base_generation)
    for rows in (base_selections, adaptive_generation, adaptive_selections):
        if set(rows) != set(ids):
            raise ValueError("adaptive comparison inputs do not align")
    base_hits = selected_hits(
        base_generation, base_selections, ids, args.method
    )
    adaptive_hits = selected_hits(
        adaptive_generation, adaptive_selections, ids, args.method
    )
    wins = sum(new and not old for old, new in zip(base_hits, adaptive_hits))
    losses = sum(old and not new for old, new in zip(base_hits, adaptive_hits))
    payload = {
        "source": "paired confidence-gated extension evaluation",
        "selection_uses_gold": False,
        "evaluation_uses_gold": True,
        "queries": len(ids),
        "method": args.method,
        "base_mean_blocks": sum(len(base_generation[q]["branches"]) for q in ids)
        / len(ids),
        "adaptive_mean_blocks": sum(
            len(adaptive_generation[q]["branches"]) for q in ids
        )
        / len(ids),
        "expanded_queries": sum(
            len(adaptive_generation[q]["branches"])
            > len(base_generation[q]["branches"])
            for q in ids
        ),
        "base_selected_accuracy": sum(base_hits) / len(ids),
        "adaptive_selected_accuracy": sum(adaptive_hits) / len(ids),
        "base_any_branch_accuracy": sum(
            bool(base_generation[q]["any_branch_target_hit"]) for q in ids
        )
        / len(ids),
        "adaptive_any_branch_accuracy": sum(
            bool(adaptive_generation[q]["any_branch_target_hit"]) for q in ids
        )
        / len(ids),
        "wins_losses": [wins, losses],
        "mcnemar_p": mcnemar_exact_p(wins, losses),
        "adaptive_mean_parallel_critical_seconds": sum(
            float(adaptive_generation[q]["parallel_branch_critical_seconds"])
            for q in ids
        )
        / len(ids),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
