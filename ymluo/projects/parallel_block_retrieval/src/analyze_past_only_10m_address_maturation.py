#!/usr/bin/env python3
"""Quantify address maturation across generation states in a real 10M memory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "doc" / "1b_context_search_research_exploration" / "evidence"
METHOD = "multilevel_bm25_book8_segment32"
METRICS = [
    "true_scope_rank",
    "book_router_hit",
    "same_scope_any_at_8",
    "same_scope_fraction_at_8",
    "same_scope_within_4k_any_at_8",
    "candidate_blocks",
    "query_seconds",
    "scope_query_features",
    "scope_top8_positive_share",
    "scope_score_normalized_entropy",
    "scope_top1_z",
]
BINARY = {
    "book_router_hit",
    "same_scope_any_at_8",
    "same_scope_within_4k_any_at_8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_all_states_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "pg19_past_only_10m_address_maturation_20260715.json",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_change(
    early: np.ndarray,
    late: np.ndarray,
    *,
    binary: bool,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    delta = late - early
    output: dict[str, Any] = {
        "definition": "late(512) minus early(64)",
        "mean_delta": float(delta.mean()),
        "query_bootstrap95": bootstrap_ci(delta, samples, rng),
    }
    if binary:
        gains = int(np.sum((early == 0) & (late == 1)))
        losses = int(np.sum((early == 1) & (late == 0)))
        output.update(
            {
                "discordant_gains": gains,
                "discordant_losses": losses,
                "exact_two_sided_mcnemar_p": (
                    float(binomtest(gains, gains + losses, p=0.5).pvalue)
                    if gains + losses
                    else 1.0
                ),
            }
        )
    return output


def monotonic_fraction(matrix: np.ndarray, direction: str) -> float:
    differences = np.diff(matrix, axis=1)
    if direction == "nondecreasing":
        return float(np.mean(np.all(differences >= -1e-12, axis=1)))
    if direction == "nonincreasing":
        return float(np.mean(np.all(differences <= 1e-12, axis=1)))
    raise ValueError(direction)


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.rows) if row["method"] == METHOD]
    states = sorted({int(row["prefix_tokens"]) for row in rows})
    query_ids = sorted({int(row["query_id"]) for row in rows})
    lookup = {(int(row["query_id"]), int(row["prefix_tokens"])): row for row in rows}
    if len(lookup) != len(states) * len(query_ids):
        raise ValueError("incomplete query-by-state matrix")
    rng = np.random.default_rng(args.seed)

    matrices = {
        metric: np.asarray(
            [
                [float(lookup[(query_id, state)][metric]) for state in states]
                for query_id in query_ids
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    by_state: dict[str, Any] = {}
    for state_index, state in enumerate(states):
        by_state[str(state)] = {
            metric: float(matrix[:, state_index].mean())
            for metric, matrix in matrices.items()
        }

    paired_64_to_512 = {
        metric: paired_change(
            matrix[:, 0],
            matrix[:, -1],
            binary=metric in BINARY,
            samples=args.bootstrap_samples,
            rng=rng,
        )
        for metric, matrix in matrices.items()
    }
    within_query_trajectory = {
        "true_scope_rank_nonincreasing": monotonic_fraction(
            matrices["true_scope_rank"], "nonincreasing"
        ),
        "top8_hit_nondecreasing": monotonic_fraction(
            matrices["same_scope_any_at_8"], "nondecreasing"
        ),
        "top8_purity_nondecreasing": monotonic_fraction(
            matrices["same_scope_fraction_at_8"], "nondecreasing"
        ),
        "score_entropy_nondecreasing": monotonic_fraction(
            matrices["scope_score_normalized_entropy"], "nondecreasing"
        ),
    }

    transition_rows: dict[str, list[float]] = defaultdict(list)
    for query_index in range(len(query_ids)):
        for state_index in range(1, len(states)):
            transition_rows["rank_improvement"].append(
                matrices["true_scope_rank"][query_index, state_index - 1]
                - matrices["true_scope_rank"][query_index, state_index]
            )
            for metric in (
                "scope_query_features",
                "scope_top8_positive_share",
                "scope_score_normalized_entropy",
                "scope_top1_z",
            ):
                transition_rows[metric].append(
                    matrices[metric][query_index, state_index]
                    - matrices[metric][query_index, state_index - 1]
                )
    transition_correlations: dict[str, Any] = {}
    rank_improvement = transition_rows["rank_improvement"]
    for metric, values in transition_rows.items():
        if metric == "rank_improvement":
            continue
        correlation = spearmanr(values, rank_improvement)
        transition_correlations[metric] = {
            "rho_with_true_rank_improvement": float(correlation.statistic),
            "p_descriptive": float(correlation.pvalue),
        }

    payload = {
        "source": "real strict past-only PG19 9.9M address maturation",
        "protocol": {
            "method": METHOD,
            "queries": len(query_ids),
            "states": states,
            "final_reader_budget_tokens": 512,
            "selection_uses_target": False,
            "offline_scope_labels_used_only_for_evaluation": True,
        },
        "state_means": by_state,
        "paired_64_to_512": paired_64_to_512,
        "within_query_trajectory": within_query_trajectory,
        "transition_correlations": transition_correlations,
        "interpretation": {
            "address_matures": (
                "true scope rank and Top-8 retrieval improve as generated state grows while the "
                "candidate domain and final 512-token reader budget stay nearly fixed"
            ),
            "confidence_is_not_concentration": (
                "normalized score entropy rises and Top-8 score share falls, so a sharper score "
                "distribution is not a valid general proxy for address correctness"
            ),
            "controller_implication": (
                "trajectory/change features are required; static margin or entropy gates are insufficient"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"state_means": by_state, "trajectory": within_query_trajectory}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
