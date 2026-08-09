from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate address maturity to marginal retrieval value across state lengths."
    )
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument(
        "--ppl_rows",
        required=True,
        help="Comma-separated suffix:path pairs.",
    )
    parser.add_argument("--retrieval_method", default="bm25_e5_rrf")
    parser.add_argument("--reader_method", default="bm25_e5_rrf_local_2x4")
    parser.add_argument("--output_path", required=True)
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
    ppl_paths = {}
    for item in args.ppl_rows.split(","):
        suffix, path = item.split(":", 1)
        ppl_paths[int(suffix)] = path
    retrieval_rows = [
        row
        for row in read_jsonl(args.retrieval_rows)
        if row["method"] == args.retrieval_method
    ]
    retrieval_lookup = {
        (int(row["prefix_tokens"]), int(row["query_id"])): row
        for row in retrieval_rows
    }
    state_rows = []
    per_state_improvements: dict[int, dict[int, float]] = {}
    for state in sorted(ppl_paths):
        rows = read_jsonl(ppl_paths[state])
        query_only = {
            int(row["query_id"]): float(row["mean_nll"])
            for row in rows
            if row["method"] == "query_only"
        }
        reader = {
            int(row["query_id"]): row
            for row in rows
            if row["method"] == args.reader_method
        }
        random_rows = {
            int(row["query_id"]): row
            for row in rows
            if row["method"] == "random512"
        }
        improvements = {
            query_id: query_only[query_id] - float(reader[query_id]["mean_nll"])
            for query_id in sorted(query_only)
        }
        random_improvements = [
            query_only[query_id] - float(random_rows[query_id]["mean_nll"])
            for query_id in sorted(query_only)
        ]
        per_state_improvements[state] = improvements
        retrieval_group = [
            retrieval_lookup[(state, query_id)] for query_id in sorted(query_only)
        ]
        query_nll = float(np.mean(list(query_only.values())))
        reader_nll = float(np.mean([float(row["mean_nll"]) for row in reader.values()]))
        state_rows.append(
            {
                "state_suffix_tokens": state,
                "query_only_mean_nll": query_nll,
                "query_only_ppl": math.exp(query_nll),
                "reader_mean_nll": reader_nll,
                "reader_ppl": math.exp(reader_nll),
                "mean_retrieval_nll_improvement": float(np.mean(list(improvements.values()))),
                "retrieval_improvement_bootstrap95": bootstrap_ci(
                    list(improvements.values()),
                    samples=args.bootstrap_samples,
                    seed=args.seed + state,
                ),
                "mean_random_nll_improvement": float(np.mean(random_improvements)),
                "same_scope_any_at_8": float(
                    np.mean([float(row["same_scope_any_at_8"]) for row in retrieval_group])
                ),
                "mean_same_scope_fraction_at_8": float(
                    np.mean(
                        [float(row["same_scope_fraction_at_8"]) for row in retrieval_group]
                    )
                ),
            }
        )

    comparisons = []
    states = sorted(per_state_improvements)
    for left, right in zip(states, states[1:]):
        differences = [
            per_state_improvements[right][query_id]
            - per_state_improvements[left][query_id]
            for query_id in sorted(per_state_improvements[left])
        ]
        comparisons.append(
            {
                "transition": f"{left}->{right}",
                "mean_change_in_retrieval_improvement": float(np.mean(differences)),
                "change_bootstrap95": bootstrap_ci(
                    differences,
                    samples=args.bootstrap_samples,
                    seed=args.seed + left + right,
                ),
                "later_state_wins": sum(value > 0 for value in differences),
                "earlier_state_wins": sum(value < 0 for value in differences),
            }
        )
    peak_state = max(
        state_rows, key=lambda row: float(row["mean_retrieval_nll_improvement"])
    )["state_suffix_tokens"]
    peak_to_last = [
        per_state_improvements[states[-1]][query_id]
        - per_state_improvements[int(peak_state)][query_id]
        for query_id in sorted(per_state_improvements[int(peak_state)])
    ]
    output = {
        "source": "PG19 past-only state maturity versus marginal retrieval value",
        "protocol": {
            "past_only": True,
            "retrieval_method": args.retrieval_method,
            "reader_method": args.reader_method,
            "retrieved_tokens": 512,
            "selection_uses_target": False,
        },
        "states": state_rows,
        "adjacent_state_comparisons": comparisons,
        "peak_retrieval_value_state": peak_state,
        "peak_to_longest_state_change": {
            "transition": f"{peak_state}->{states[-1]}",
            "mean_change_in_retrieval_improvement": float(np.mean(peak_to_last)),
            "change_bootstrap95": bootstrap_ci(
                peak_to_last, samples=args.bootstrap_samples, seed=args.seed + 999
            ),
            "longest_state_wins": sum(value > 0 for value in peak_to_last),
            "peak_state_wins": sum(value < 0 for value in peak_to_last),
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
