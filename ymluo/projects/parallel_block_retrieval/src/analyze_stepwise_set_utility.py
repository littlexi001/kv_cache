from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PAIR_SPECS = (
    ("resolve_bridge", "fact_only", "target_only", "fact_to_target_block"),
    ("resolve_bridge", "target_only", "target_span", "target_block_to_auto_span"),
    ("resolve_bridge", "fact_only", "target_span", "oracle_fact_to_auto_span"),
    (
        "resolve_bridge",
        "target_only",
        "target_plus_negative",
        "add_negative_block",
    ),
    ("resolve_bridge", "target_only", "negative_only", "remove_target_block"),
    (
        "resolve_answer_from_bridge",
        "fact_no_state",
        "fact_with_state",
        "add_compact_state_to_fact",
    ),
    (
        "resolve_answer_from_bridge",
        "fact_with_state",
        "fact_with_full_state",
        "replace_compact_with_full_state_on_fact",
    ),
    (
        "resolve_answer_from_bridge",
        "target_no_state",
        "target_with_state",
        "add_compact_state_to_target_block",
    ),
    (
        "resolve_answer_from_bridge",
        "target_no_state",
        "target_span_no_state",
        "target_block_to_auto_span_no_state",
    ),
    (
        "resolve_answer_from_bridge",
        "target_with_state",
        "target_span_with_state",
        "target_block_to_auto_span_with_state",
    ),
    (
        "resolve_answer_from_bridge",
        "fact_with_state",
        "target_span_with_state",
        "oracle_fact_to_auto_span_with_state",
    ),
    (
        "resolve_answer_from_bridge",
        "target_with_state",
        "target_with_full_state",
        "replace_compact_with_full_state_on_target_block",
    ),
    (
        "resolve_answer_from_bridge",
        "target_with_state",
        "target_plus_negative_with_state",
        "add_negative_block_with_state",
    ),
    (
        "resolve_answer_from_bridge",
        "target_with_state",
        "target_plus_previous_with_state",
        "add_previous_block_with_state",
    ),
    (
        "resolve_answer_from_bridge",
        "target_with_state",
        "previous_with_state",
        "replace_target_with_previous_block",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired analysis of stepwise set utility.")
    parser.add_argument(
        "--rows_path",
        required=True,
        help="One or more comma-separated rows.jsonl paths.",
    )
    parser.add_argument("--splits", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_mean_ci(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 1_000
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        indices = rng.integers(0, array.size, size=(count, array.size))
        means[start : start + count] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_modes(
    rows: Sequence[dict[str, Any]],
    step_type: str,
    baseline_mode: str,
    candidate_mode: str,
    name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    relevant = [row for row in rows if row["step_type"] == step_type]
    by_key = {
        (int(row["query_id"]), int(row["step_index"]), str(row["mode"])): row
        for row in relevant
    }
    keys = sorted(
        (query_id, step_index)
        for query_id, step_index, mode in by_key
        if mode == baseline_mode
        and (query_id, step_index, candidate_mode) in by_key
    )
    if not keys:
        return None
    baseline = [by_key[(*key, baseline_mode)] for key in keys]
    candidate = [by_key[(*key, candidate_mode)] for key in keys]
    nll_delta = [
        float(candidate_row["target_nll"]) - float(baseline_row["target_nll"])
        for baseline_row, candidate_row in zip(baseline, candidate)
    ]
    hit_delta = [
        float(candidate_row["target_hit"]) - float(baseline_row["target_hit"])
        for baseline_row, candidate_row in zip(baseline, candidate)
    ]
    f1_delta = [
        float(candidate_row["target_f1"]) - float(baseline_row["target_f1"])
        for baseline_row, candidate_row in zip(baseline, candidate)
    ]
    hit_wins = sum(delta > 0 for delta in hit_delta)
    hit_losses = sum(delta < 0 for delta in hit_delta)
    nll_ci = bootstrap_mean_ci(nll_delta, bootstrap_samples, seed)
    hit_ci = bootstrap_mean_ci(hit_delta, bootstrap_samples, seed + 1)
    f1_ci = bootstrap_mean_ci(f1_delta, bootstrap_samples, seed + 2)
    return {
        "name": name,
        "step_type": step_type,
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "pairs": len(keys),
        "mean_baseline_nll": float(np.mean([row["target_nll"] for row in baseline])),
        "mean_candidate_nll": float(np.mean([row["target_nll"] for row in candidate])),
        "mean_nll_delta_candidate_minus_baseline": float(np.mean(nll_delta)),
        "nll_delta_bootstrap_95ci": list(nll_ci),
        "nll_candidate_wins_losses_ties": [
            sum(delta < 0 for delta in nll_delta),
            sum(delta > 0 for delta in nll_delta),
            sum(delta == 0 for delta in nll_delta),
        ],
        "baseline_hit_rate": float(np.mean([row["target_hit"] for row in baseline])),
        "candidate_hit_rate": float(np.mean([row["target_hit"] for row in candidate])),
        "mean_hit_delta_candidate_minus_baseline": float(np.mean(hit_delta)),
        "hit_delta_bootstrap_95ci": list(hit_ci),
        "hit_wins_losses_ties": [
            hit_wins,
            hit_losses,
            len(keys) - hit_wins - hit_losses,
        ],
        "mcnemar_exact_p": mcnemar_exact_p(hit_wins, hit_losses),
        "mean_f1_delta_candidate_minus_baseline": float(np.mean(f1_delta)),
        "f1_delta_bootstrap_95ci": list(f1_ci),
    }


def main() -> None:
    args = parse_args()
    rows_paths = [Path(item.strip()) for item in args.rows_path.split(",") if item.strip()]
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    rows = [row for path in rows_paths for row in read_jsonl(path)]
    if allowed_splits:
        rows = [row for row in rows if str(row["split"]) in allowed_splits]
    comparisons = []
    for offset, spec in enumerate(PAIR_SPECS):
        comparison = compare_modes(
            rows,
            *spec,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + offset * 11,
        )
        if comparison is not None:
            comparisons.append(comparison)
    payload = {
        "source": [str(path) for path in rows_paths],
        "splits": sorted(allowed_splits),
        "rows": len(rows),
        "queries": len({int(row["query_id"]) for row in rows}),
        "bootstrap_samples": args.bootstrap_samples,
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
