from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired address-maturation analysis for past-only retrieval."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.rows)
    methods = sorted({str(row["method"]) for row in rows})
    suffixes = sorted({int(row["prefix_tokens"]) for row in rows})
    lookup = {
        (str(row["method"]), int(row["query_id"]), int(row["prefix_tokens"])): row
        for row in rows
    }
    query_ids = sorted({int(row["query_id"]) for row in rows})
    trajectories = []
    comparisons = []
    for method_index, method in enumerate(methods):
        per_query_correlations = []
        for query_id in query_ids:
            fractions = [
                float(lookup[(method, query_id, suffix)][f"same_scope_fraction_at_{args.topk}"])
                for suffix in suffixes
            ]
            corr = spearmanr(suffixes, fractions)
            if np.isfinite(corr.statistic):
                per_query_correlations.append(float(corr.statistic))
        trajectories.append(
            {
                "method": method,
                "suffixes": suffixes,
                "mean_per_query_spearman_suffix_vs_same_scope_fraction": float(
                    np.mean(per_query_correlations)
                ),
                "positive_trajectory_rate": float(
                    np.mean(np.asarray(per_query_correlations) > 0)
                ),
            }
        )
        for left, right in zip(suffixes, suffixes[1:]):
            any_changes = []
            fraction_changes = []
            near4_changes = []
            jaccards = []
            for query_id in query_ids:
                left_row = lookup[(method, query_id, left)]
                right_row = lookup[(method, query_id, right)]
                any_changes.append(
                    int(right_row[f"same_scope_any_at_{args.topk}"])
                    - int(left_row[f"same_scope_any_at_{args.topk}"])
                )
                fraction_changes.append(
                    float(right_row[f"same_scope_fraction_at_{args.topk}"])
                    - float(left_row[f"same_scope_fraction_at_{args.topk}"])
                )
                near4_changes.append(
                    int(right_row[f"same_scope_within_4k_any_at_{args.topk}"])
                    - int(left_row[f"same_scope_within_4k_any_at_{args.topk}"])
                )
                left_set = set(int(item) for item in left_row["top_block_ids"][: args.topk])
                right_set = set(int(item) for item in right_row["top_block_ids"][: args.topk])
                jaccards.append(len(left_set & right_set) / len(left_set | right_set))
            wins = sum(value > 0 for value in any_changes)
            losses = sum(value < 0 for value in any_changes)
            comparisons.append(
                {
                    "method": method,
                    "transition": f"{left}->{right}",
                    "same_scope_any_wins": wins,
                    "same_scope_any_losses": losses,
                    "same_scope_any_exact_p": (
                        float(binomtest(wins, wins + losses, 0.5).pvalue)
                        if wins + losses
                        else 1.0
                    ),
                    "mean_same_scope_fraction_change": float(np.mean(fraction_changes)),
                    "same_scope_fraction_change_bootstrap95": bootstrap_ci(
                        fraction_changes,
                        samples=args.bootstrap_samples,
                        seed=args.seed + method_index * 10 + left,
                    ),
                    "near4k_any_wins": sum(value > 0 for value in near4_changes),
                    "near4k_any_losses": sum(value < 0 for value in near4_changes),
                    "mean_topk_jaccard": float(np.mean(jaccards)),
                }
            )
        left, right = suffixes[0], suffixes[-1]
        any_changes = []
        fraction_changes = []
        for query_id in query_ids:
            left_row = lookup[(method, query_id, left)]
            right_row = lookup[(method, query_id, right)]
            any_changes.append(
                int(right_row[f"same_scope_any_at_{args.topk}"])
                - int(left_row[f"same_scope_any_at_{args.topk}"])
            )
            fraction_changes.append(
                float(right_row[f"same_scope_fraction_at_{args.topk}"])
                - float(left_row[f"same_scope_fraction_at_{args.topk}"])
            )
        wins = sum(value > 0 for value in any_changes)
        losses = sum(value < 0 for value in any_changes)
        comparisons.append(
            {
                "method": method,
                "transition": f"{left}->{right}",
                "same_scope_any_wins": wins,
                "same_scope_any_losses": losses,
                "same_scope_any_exact_p": (
                    float(binomtest(wins, wins + losses, 0.5).pvalue)
                    if wins + losses
                    else 1.0
                ),
                "mean_same_scope_fraction_change": float(np.mean(fraction_changes)),
                "same_scope_fraction_change_bootstrap95": bootstrap_ci(
                    fraction_changes,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 + method_index,
                ),
            }
        )

    output = {
        "source": "paired dynamic properties on PG19 past-only 10M retrieval",
        "protocol": {
            "past_only": True,
            "selection_uses_target": False,
            "state_uses_recent_suffix": True,
            "topk": args.topk,
        },
        "queries": len(query_ids),
        "suffixes": suffixes,
        "trajectories": trajectories,
        "paired_comparisons": comparisons,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
