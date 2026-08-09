from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from merge_sparse_attention_reference_nll import paired_bootstrap_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a per-query oracle over strict-token-budget retrieval actions."
    )
    parser.add_argument("--input", action="append", required=True, help="label=rows.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--token_limit", type=int, default=1000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("input must use label=path")
    label, raw_path = spec.split("=", maxsplit=1)
    return label, Path(raw_path)


def load_actions(
    specs: list[str], token_limit: int
) -> tuple[dict[int, float], dict[int, dict[str, dict[str, Any]]]]:
    full_by_query: dict[int, float] = {}
    actions_by_query: dict[int, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        label, path = named_path(spec)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                query_id = int(raw["query_id"])
                action = str(raw["action"])
                nll = float(raw["nll"])
                if action == "full":
                    previous = full_by_query.setdefault(query_id, nll)
                    if not np.isclose(previous, nll, atol=1.0e-6):
                        raise ValueError(f"inconsistent full NLL for query {query_id}")
                    continue
                max_tokens = int(float(raw["max_layer_global_tokens"]))
                violation_rate = float(raw["strict_1000_token_violation_rate"])
                if max_tokens > token_limit or violation_rate > 0.0:
                    continue
                name = f"{label}:{action}"
                actions_by_query.setdefault(query_id, {})[name] = {
                    "nll": nll,
                    "delta_nll_vs_full": float(raw["delta_nll_vs_full"]),
                    "max_layer_global_tokens": max_tokens,
                }
    return full_by_query, actions_by_query


def summarize_values(
    action: str, deltas: np.ndarray, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    ci_low, ci_high = paired_bootstrap_ci(deltas, bootstrap_samples, seed)
    return {
        "action": action,
        "queries": len(deltas),
        "mean_delta_nll_vs_full": float(deltas.mean()),
        "delta_nll_ci95_low": ci_low,
        "delta_nll_ci95_high": ci_high,
        "median_delta_nll_vs_full": float(np.median(deltas)),
        "p95_abs_delta_nll": float(np.quantile(np.abs(deltas), 0.95)),
        "fraction_nll_not_worse": float(np.mean(deltas <= 0.0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_by_query, actions_by_query = load_actions(args.input, args.token_limit)
    common_queries = sorted(set(full_by_query) & set(actions_by_query))
    if not common_queries:
        raise ValueError("no strict-budget actions found")

    action_names = sorted(
        set.intersection(*(set(actions_by_query[item]) for item in common_queries))
    )
    if not action_names:
        raise ValueError("inputs have no action shared by every query")
    summary_rows: list[dict[str, Any]] = []
    for action in action_names:
        deltas = np.asarray(
            [actions_by_query[item][action]["delta_nll_vs_full"] for item in common_queries]
        )
        summary_rows.append(
            summarize_values(
                action, deltas, args.bootstrap_samples, args.seed
            )
        )

    oracle_rows: list[dict[str, Any]] = []
    for query_id in common_queries:
        available = actions_by_query[query_id]
        chosen, row = min(
            available.items(), key=lambda item: (float(item[1]["nll"]), item[0])
        )
        oracle_rows.append(
            {
                "query_id": query_id,
                "full_nll": full_by_query[query_id],
                "chosen_action": chosen,
                **row,
            }
        )
    oracle_deltas = np.asarray(
        [float(row["delta_nll_vs_full"]) for row in oracle_rows]
    )
    oracle_summary = summarize_values(
        "per_query_oracle", oracle_deltas, args.bootstrap_samples, args.seed
    )
    oracle_summary["action_counts"] = dict(
        sorted(Counter(str(row["chosen_action"]) for row in oracle_rows).items())
    )
    summary_rows.append(oracle_summary)

    with (output_dir / "query_oracle.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oracle_rows[0]))
        writer.writeheader()
        writer.writerows(oracle_rows)
    with (output_dir / "action_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = sorted(set().union(*(row.keys() for row in summary_rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "queries": len(common_queries),
        "token_limit": args.token_limit,
        "actions": action_names,
        "action_summary": summary_rows,
        "interpretation": (
            "The oracle chooses only among actions whose measured layer-global context "
            "never exceeds the token limit. It is an upper bound, not a deployable router."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
