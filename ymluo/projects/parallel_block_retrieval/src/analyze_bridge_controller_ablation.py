from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from run_dynamic_kv_multisample import token_f1


BINARY_FIELDS = (
    "hop1_record_hit",
    "hop1_gold_hit",
    "any_search_record_hit",
    "any_search_gold_hit",
    "answer_hit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired analysis for global bridge-controller retrieval runs."
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated name=results.jsonl entries.",
    )
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    if "answer_f1" not in enriched:
        enriched["answer_f1"] = max(
            token_f1(str(enriched.get("answer_text", "")), str(reference))
            for reference in enriched["answers"]
        )
    if "online_seconds" not in enriched:
        enriched["online_seconds"] = sum(
            float(enriched.get(field) or 0.0)
            for field in (
                "qk_capture_seconds",
                "initial_retrieval_seconds",
                "qk_retrieval_seconds",
                "bridge_generation_seconds",
                "bridge_bm25_seconds",
                "answer_generation_seconds",
            )
        )
    return enriched


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in rows if bool(row.get("any_search_gold_hit"))]
    no_gold_rows = [row for row in rows if not bool(row.get("any_search_gold_hit"))]
    return {
        "queries": len(rows),
        **{
            f"{field}_rate": sum(bool(row.get(field)) for row in rows) / len(rows)
            for field in BINARY_FIELDS
        },
        "mean_answer_f1": statistics.fmean(float(row["answer_f1"]) for row in rows),
        "median_online_seconds": statistics.median(
            float(row["online_seconds"]) for row in rows
        ),
        "bridge_gold_queries": len(gold_rows),
        "answer_hit_given_bridge_gold": (
            sum(bool(row.get("answer_hit")) for row in gold_rows) / len(gold_rows)
            if gold_rows
            else 0.0
        ),
        "answer_hit_without_bridge_gold": (
            sum(bool(row.get("answer_hit")) for row in no_gold_rows)
            / len(no_gold_rows)
            if no_gold_rows
            else 0.0
        ),
    }


def paired_binary(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    wins = [
        int(right["query_id"])
        for left, right in zip(baseline, candidate, strict=True)
        if not bool(left.get(field)) and bool(right.get(field))
    ]
    losses = [
        int(right["query_id"])
        for left, right in zip(baseline, candidate, strict=True)
        if bool(left.get(field)) and not bool(right.get(field))
    ]
    discordant = len(wins) + len(losses)
    smaller = min(len(wins), len(losses))
    exact_p = (
        min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(smaller + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "wins": wins,
        "losses": losses,
        "net": len(wins) - len(losses),
        "rate_delta": (len(wins) - len(losses)) / len(baseline),
        "mcnemar_exact_p": exact_p,
    }


def paired_continuous(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    field: str,
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            float(right[field]) - float(left[field])
            for left, right in zip(baseline, candidate, strict=True)
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(0)
    indices = generator.integers(
        0, deltas.shape[0], size=(bootstrap_samples, deltas.shape[0])
    )
    means = deltas[indices].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "wins": int((deltas > 0).sum()),
        "losses": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()),
        "bootstrap_95_ci": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def main() -> None:
    args = parse_args()
    run_specs = []
    for spec in args.runs.split(","):
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid run specification: {spec!r}")
        run_specs.append((name, Path(path)))
    if len(run_specs) < 2:
        raise ValueError("at least two runs are required")

    runs: dict[str, list[dict[str, Any]]] = {}
    expected_ids = None
    for name, path in run_specs:
        rows = sorted(
            (enrich_row(row) for row in read_jsonl(path)),
            key=lambda row: int(row["query_id"]),
        )
        query_ids = [int(row["query_id"]) for row in rows]
        if expected_ids is None:
            expected_ids = query_ids
        elif query_ids != expected_ids:
            raise ValueError(f"query IDs for {name} do not align with the baseline")
        runs[name] = rows

    baseline_name = run_specs[0][0]
    comparisons = {}
    for baseline_index, (left_name, _left_path) in enumerate(run_specs):
        for right_name, _right_path in run_specs[baseline_index + 1 :]:
            comparisons[f"{right_name}_vs_{left_name}"] = {
                **{
                    field: paired_binary(runs[left_name], runs[right_name], field)
                    for field in BINARY_FIELDS
                },
                "answer_f1": paired_continuous(
                    runs[left_name], runs[right_name], "answer_f1"
                ),
            }
    payload = {
        "baseline": baseline_name,
        "query_ids": expected_ids,
        "summaries": {name: summarize(rows) for name, rows in runs.items()},
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
