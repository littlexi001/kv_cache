from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a frozen-selector, leakage-free strict reasoning chain."
    )
    parser.add_argument("--bridge_generation_rows_path", required=True)
    parser.add_argument("--bridge_selection_rows_path", required=True)
    parser.add_argument("--bridge_traces_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--answer_generation_rows_path", required=True)
    parser.add_argument("--answer_selection_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bridge_selection_field", default="heuristic_structured_index")
    parser.add_argument("--answer_selection_field", default="heuristic_structured_index")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["query_id"]): row for row in read_jsonl(Path(path))}


def paired_summary(
    baseline: Sequence[bool], selected: Sequence[bool]
) -> dict[str, Any]:
    wins = sum(new and not old for old, new in zip(baseline, selected))
    losses = sum(old and not new for old, new in zip(baseline, selected))
    return {
        "baseline_hits": sum(baseline),
        "selected_hits": sum(selected),
        "wins_losses": [wins, losses],
        "mcnemar_p": mcnemar_exact_p(wins, losses),
    }


def rate(ids: Sequence[int], predicate: Callable[[int], bool]) -> float:
    return sum(bool(predicate(query_id)) for query_id in ids) / max(1, len(ids))


def main() -> None:
    args = parse_args()
    bridge_generation = keyed(args.bridge_generation_rows_path)
    bridge_selection = keyed(args.bridge_selection_rows_path)
    bridge_traces = keyed(args.bridge_traces_path)
    retrieval = keyed(args.retrieval_rows_path)
    answer_generation = keyed(args.answer_generation_rows_path)
    answer_selection = keyed(args.answer_selection_rows_path)
    ids = sorted(answer_generation)
    sources = {
        "bridge_generation": bridge_generation,
        "bridge_selection": bridge_selection,
        "bridge_traces": bridge_traces,
        "retrieval": retrieval,
        "answer_selection": answer_selection,
    }
    for name, rows in sources.items():
        if set(rows) != set(ids):
            raise ValueError(f"{name} does not exactly cover answer queries")

    bridge_top1 = [
        bool(bridge_generation[q]["branches"][0]["target_hit"]) for q in ids
    ]
    bridge_heuristic = [
        bool(
            bridge_generation[q]["branches"][
                int(bridge_selection[q]["heuristic_index"])
            ]["target_hit"]
        )
        for q in ids
    ]
    bridge_head = [bool(bridge_traces[q]["bridge_target_hit"]) for q in ids]
    answer_top1 = [
        bool(answer_generation[q]["branches"][0]["target_hit"]) for q in ids
    ]
    answer_heuristic = [
        bool(
            answer_generation[q]["branches"][
                int(answer_selection[q]["heuristic_index"])
            ]["target_hit"]
        )
        for q in ids
    ]
    answer_head = [
        bool(
            answer_generation[q]["branches"][
                int(answer_selection[q][args.answer_selection_field])
            ]["target_hit"]
        )
        for q in ids
    ]
    answer_any = [
        bool(answer_generation[q]["any_branch_target_hit"]) for q in ids
    ]

    payload = {
        "source": "offline evaluation of deployment-only selector outputs",
        "selection_uses_gold": False,
        "evaluation_uses_gold": True,
        "queries": len(ids),
        "bridge": {
            "top1_accuracy": sum(bridge_top1) / len(ids),
            "heuristic_accuracy": sum(bridge_heuristic) / len(ids),
            "head_accuracy": sum(bridge_head) / len(ids),
            "head_vs_top1": paired_summary(bridge_top1, bridge_head),
            "head_vs_heuristic": paired_summary(bridge_heuristic, bridge_head),
        },
        "retrieval": {
            "recall_at_1": rate(ids, lambda q: 0 < int(retrieval[q]["lexical_rank"]) <= 1),
            "recall_at_3": rate(ids, lambda q: 0 < int(retrieval[q]["lexical_rank"]) <= 3),
            "recall_at_16": rate(ids, lambda q: 0 < int(retrieval[q]["lexical_rank"]) <= 16),
            "recall_at_3_given_bridge_hit": rate(
                [q for q in ids if bridge_traces[q]["bridge_target_hit"]],
                lambda q: 0 < int(retrieval[q]["lexical_rank"]) <= 3,
            ),
            "recall_at_16_given_bridge_hit": rate(
                [q for q in ids if bridge_traces[q]["bridge_target_hit"]],
                lambda q: 0 < int(retrieval[q]["lexical_rank"]) <= 16,
            ),
        },
        "answer": {
            "top1_accuracy": sum(answer_top1) / len(ids),
            "heuristic_accuracy": sum(answer_heuristic) / len(ids),
            "head_accuracy": sum(answer_head) / len(ids),
            "any_branch_accuracy": sum(answer_any) / len(ids),
            "head_vs_top1": paired_summary(answer_top1, answer_head),
            "head_vs_heuristic": paired_summary(answer_heuristic, answer_head),
            "joint_bridge_and_head_accuracy": sum(
                bridge and answer for bridge, answer in zip(bridge_head, answer_head)
            )
            / len(ids),
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
