from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_stepwise_set_utility import mcnemar_exact_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bridge-source and final-reader capacity in a 2x2 design."
    )
    parser.add_argument("--small_bridge_small_reader", required=True)
    parser.add_argument("--small_bridge_large_reader", required=True)
    parser.add_argument("--large_bridge_small_reader", required=True)
    parser.add_argument("--large_bridge_large_reader", required=True)
    parser.add_argument("--small_bridge_traces", required=True)
    parser.add_argument("--large_bridge_traces", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generation_rows(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["query_id"]): row for row in read_jsonl(Path(path))}


def bridge_hits(path: str) -> dict[int, bool]:
    return {
        int(row["query_id"]): bool(row["bridge_target_hit"])
        for row in read_jsonl(Path(path))
    }


def summarize(
    rows: dict[int, dict[str, Any]], bridges: dict[int, bool]
) -> dict[str, Any]:
    if set(rows) != set(bridges):
        raise ValueError("generation and bridge trace queries do not align")
    hits = {
        query_id: bool(row["branches"][0]["target_hit"])
        for query_id, row in rows.items()
    }
    retrieved = {
        query_id: bool(row["retrieval_target_span_hit_at_k"])
        for query_id, row in rows.items()
    }

    def conditional(mask: dict[int, bool], value: bool) -> dict[str, Any]:
        query_ids = [query_id for query_id in rows if mask[query_id] == value]
        correct = sum(hits[query_id] for query_id in query_ids)
        return {
            "queries": len(query_ids),
            "correct": correct,
            "accuracy": correct / len(query_ids),
        }

    return {
        "queries": len(rows),
        "correct": sum(hits.values()),
        "accuracy": statistics.fmean(hits.values()),
        "given_second_hop_retrieved": conditional(retrieved, True),
        "given_second_hop_not_retrieved": conditional(retrieved, False),
        "given_bridge_correct": conditional(bridges, True),
        "given_bridge_incorrect": conditional(bridges, False),
        "joint_bridge_and_answer_correct": sum(
            bridges[query_id] and hits[query_id] for query_id in rows
        ),
        "mean_generation_seconds": statistics.fmean(
            float(row["branches"][0]["generation_seconds"])
            for row in rows.values()
        ),
    }


def paired(
    baseline: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("paired runs do not align")
    old = {
        query_id: bool(row["branches"][0]["target_hit"])
        for query_id, row in baseline.items()
    }
    new = {
        query_id: bool(row["branches"][0]["target_hit"])
        for query_id, row in candidate.items()
    }
    wins = sum(new[q] and not old[q] for q in old)
    losses = sum(old[q] and not new[q] for q in old)
    return {
        "baseline_accuracy": statistics.fmean(old.values()),
        "candidate_accuracy": statistics.fmean(new.values()),
        "wins": wins,
        "losses": losses,
        "mcnemar_p": mcnemar_exact_p(wins, losses),
    }


def main() -> None:
    args = parse_args()
    runs = {
        "small_bridge_small_reader": generation_rows(args.small_bridge_small_reader),
        "small_bridge_large_reader": generation_rows(args.small_bridge_large_reader),
        "large_bridge_small_reader": generation_rows(args.large_bridge_small_reader),
        "large_bridge_large_reader": generation_rows(args.large_bridge_large_reader),
    }
    traces = {
        "small": bridge_hits(args.small_bridge_traces),
        "large": bridge_hits(args.large_bridge_traces),
    }
    payload = {
        "source": "strict 2x2 bridge-source and final-reader capacity comparison",
        "runs": {
            name: summarize(rows, traces["small" if name.startswith("small") else "large"])
            for name, rows in runs.items()
        },
        "paired_effects": {
            "reader_8b_given_small_bridge": paired(
                runs["small_bridge_small_reader"],
                runs["small_bridge_large_reader"],
            ),
            "reader_8b_given_large_bridge": paired(
                runs["large_bridge_small_reader"],
                runs["large_bridge_large_reader"],
            ),
            "bridge_8b_given_small_reader": paired(
                runs["small_bridge_small_reader"],
                runs["large_bridge_small_reader"],
            ),
            "bridge_8b_given_large_reader": paired(
                runs["small_bridge_large_reader"],
                runs["large_bridge_large_reader"],
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
