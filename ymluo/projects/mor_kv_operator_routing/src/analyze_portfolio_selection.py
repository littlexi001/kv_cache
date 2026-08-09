from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_natural_operator_library import cluster_bootstrap_ci, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stratified two-fold audit for fixed-quota specialist/deep KV portfolios."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as alias=mode=/path/to/answer_nll_rows.csv.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_alias", default="deep")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def load_candidate(spec: str) -> tuple[str, dict[int, dict[str, Any]]]:
    pieces = spec.split("=", 2)
    if len(pieces) != 3:
        raise ValueError("candidate must be alias=mode=/path/to/csv")
    alias, mode, raw_path = pieces
    rows: dict[int, dict[str, Any]] = {}
    with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["mode"] != mode:
                continue
            rows[int(raw["query_id"])] = {
                "dataset": raw["dataset"],
                "answer_nll": float(raw["answer_nll"]),
                "context_blocks": int(raw["context_blocks"]),
                "context_tokens": int(raw["context_tokens"]),
            }
    if not rows:
        raise ValueError(f"no rows for {mode} in {raw_path}")
    return alias, rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [load_candidate(spec) for spec in args.candidate]
    aliases = [alias for alias, _ in payloads]
    if len(set(aliases)) != len(aliases):
        raise ValueError("candidate aliases must be unique")
    if args.baseline_alias not in aliases:
        raise ValueError("baseline alias is not a candidate")
    query_sets = [set(rows) for _, rows in payloads]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("candidates do not cover identical queries")
    query_ids = sorted(query_sets[0])
    dataset = np.asarray([payloads[0][1][query_id]["dataset"] for query_id in query_ids])
    nll = np.asarray(
        [[rows[query_id]["answer_nll"] for _, rows in payloads] for query_id in query_ids],
        dtype=np.float64,
    )
    blocks = {
        alias: sorted({rows[query_id]["context_blocks"] for query_id in query_ids})
        for alias, rows in payloads
    }
    tokens = {
        alias: sorted({rows[query_id]["context_tokens"] for query_id in query_ids})
        for alias, rows in payloads
    }
    if len({tuple(value) for value in blocks.values()}) != 1:
        raise ValueError(f"unequal block counts: {blocks}")
    if len({tuple(value) for value in tokens.values()}) != 1:
        raise ValueError(f"unequal context tokens: {tokens}")

    # Assign alternating parity independently inside each dataset.
    parity = np.zeros(len(query_ids), dtype=np.int64)
    for task in sorted(set(dataset.tolist())):
        indices = np.flatnonzero(dataset == task)
        parity[indices] = np.arange(len(indices)) % 2
    routed = np.zeros(len(query_ids), dtype=np.float64)
    selected_actions = np.zeros(len(query_ids), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    for test_parity in [0, 1]:
        train = parity != test_parity
        test = ~train
        train_means = nll[train].mean(axis=0)
        selected = int(np.argmin(train_means))
        routed[test] = nll[test, selected]
        selected_actions[test] = selected
        fold_rows.append(
            {
                "test_parity": test_parity,
                "calibration_queries": int(train.sum()),
                "heldout_queries": int(test.sum()),
                "selected_action": aliases[selected],
                "calibration_mean_nll": float(train_means[selected]),
                "heldout_selected_mean_nll": float(nll[test, selected].mean()),
                "heldout_baseline_mean_nll": float(
                    nll[test, aliases.index(args.baseline_alias)].mean()
                ),
            }
        )

    baseline_index = aliases.index(args.baseline_alias)
    baseline = nll[:, baseline_index]
    delta = routed - baseline
    rng = np.random.default_rng(args.seed)
    cluster_ci = cluster_bootstrap_ci(
        delta, dataset, args.bootstrap_samples, rng
    )
    query_bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
    for sample_index in range(args.bootstrap_samples):
        sample = rng.integers(0, len(delta), len(delta))
        query_bootstrap[sample_index] = delta[sample].mean()
    query_ci = [
        float(np.quantile(query_bootstrap, 0.025)),
        float(np.quantile(query_bootstrap, 0.975)),
    ]
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        rows.append(
            {
                "query_id": query_id,
                "dataset": dataset[index],
                "parity": int(parity[index]),
                "selected_action": aliases[selected_actions[index]],
                "selected_nll": routed[index],
                "baseline_nll": baseline[index],
                "delta_vs_baseline": delta[index],
            }
        )
    candidate_rows = [
        {
            "action": alias,
            "mean_nll": float(nll[:, index].mean()),
            "median_nll": float(np.median(nll[:, index])),
        }
        for index, alias in enumerate(aliases)
    ]
    summary = {
        "source": "stratified two-fold fixed-quota portfolio selection",
        "queries": len(query_ids),
        "datasets": dict(sorted(Counter(dataset.tolist()).items())),
        "physical_budget": {
            "context_blocks": next(iter(blocks.values())),
            "context_tokens": next(iter(tokens.values())),
        },
        "baseline": args.baseline_alias,
        "baseline_mean_nll": float(baseline.mean()),
        "out_of_fold_selected_mean_nll": float(routed.mean()),
        "mean_delta_vs_baseline": float(delta.mean()),
        "query_bootstrap_ci95": query_ci,
        "dataset_cluster_bootstrap_ci95": list(cluster_ci),
        "selected_action_counts": dict(
            sorted(Counter(aliases[index] for index in selected_actions).items())
        ),
        "folds": fold_rows,
        "interpretation": (
            "Each held-out query is evaluated under a single fixed portfolio chosen only by "
            "the opposite parity calibration split. This tests quota selection, not per-query routing."
        ),
    }
    write_csv(output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(output_dir / "folds.csv", fold_rows)
    write_csv(output_dir / "oof_rows.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

