from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact


BUDGETS = (1, 2, 4, 8, 16, 32, 64)
DISTANCE_BINS = (
    (0, 4096, "0-4K"),
    (4096, 16384, "4K-16K"),
    (16384, 65536, "16K-64K"),
    (65536, float("inf"), ">64K"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze causal scope/distance utility and bounded online probes."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--method", default="bm25_e5_rrf")
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query = defaultdict(list)
    for row in rows:
        by_query[int(row["query_id"])].append(row)
    return {
        "candidate_windows": len(rows),
        "queries": len(by_query),
        "future_event_rate": mean([float(row["future_event"]) for row in rows]),
        "query_macro_future_event_rate": mean(
            [
                float(np.mean([float(row["future_event"]) for row in group]))
                for group in by_query.values()
            ]
        ),
        "positive_future_utility_rate": mean(
            [float(row["delta_nll_b"] > 0) for row in rows]
        ),
        "mean_delta_nll_a": mean([float(row["delta_nll_a"]) for row in rows]),
        "mean_delta_nll_b": mean([float(row["delta_nll_b"]) for row in rows]),
        "query_macro_mean_delta_nll_b": mean(
            [
                float(np.mean([float(row["delta_nll_b"]) for row in group]))
                for group in by_query.values()
            ]
        ),
    }


def selection_quality(selected: list[dict[str, Any]]) -> dict[str, Any]:
    nll = [float(row["mean_nll_b"]) for row in selected]
    baseline = [float(row["baseline_nll_b"]) for row in selected]
    improvement = [left - right for left, right in zip(baseline, nll)]
    mean_nll = float(np.mean(nll))
    return {
        "queries": len(selected),
        "mean_nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 20.0)),
        "baseline_ppl": math.exp(min(float(np.mean(baseline)), 20.0)),
        "mean_nll_improvement_vs_query_only": float(np.mean(improvement)),
        "positive_future_utility_rate": float(np.mean(np.asarray(improvement) > 0)),
        "same_scope_rate": float(np.mean([row["same_scope"] for row in selected])),
        "mean_distance_tokens_for_same_scope": mean(
            [float(row["distance_tokens"]) for row in selected if row["same_scope"]]
        ),
    }


def evaluate_policy(
    groups: dict[int, list[dict[str, Any]]],
    *,
    ordering: str,
    budget: int,
    threshold: float | None,
    static: dict[int, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = []
    probes = []
    for query_id, group in sorted(groups.items()):
        def ordering_key(row: dict[str, Any]) -> tuple[float, ...]:
            rank = int(row["origins"]["bm25_e5_rrf"])
            if ordering == "global_rank":
                return (rank, int(row["window_start"]))
            if ordering == "same_scope_then_rank":
                return (-int(row["same_scope"]), rank, int(row["window_start"]))
            if ordering == "same_scope_then_distance_then_rank":
                distance = (
                    float(row["distance_tokens"])
                    if row["distance_tokens"] is not None
                    else float("inf")
                )
                return (
                    -int(row["same_scope"]),
                    distance,
                    rank,
                    int(row["window_start"]),
                )
            raise ValueError(f"unknown ordering: {ordering}")

        if ordering == "same_scope_multiscale_round_robin":
            rank_key = lambda row: (
                int(row["origins"]["bm25_e5_rrf"]), int(row["window_start"])
            )
            distance_labels = [label for _, _, label in DISTANCE_BINS]
            buckets = [
                sorted(
                    [
                        row
                        for row in group
                        if row["same_scope"] and row["distance_bin"] == label
                    ],
                    key=rank_key,
                )
                for label in distance_labels
            ]
            buckets.append(
                sorted([row for row in group if not row["same_scope"]], key=rank_key)
            )
            ordered = []
            depth = 0
            while len(ordered) < len(group):
                added = False
                for bucket in buckets:
                    if depth < len(bucket):
                        ordered.append(bucket[depth])
                        added = True
                if not added:
                    break
                depth += 1
            ranked = ordered[:budget]
        else:
            ranked = sorted(group, key=ordering_key)[:budget]
        best = ranked[0]
        used = 0
        for row in ranked:
            used += 1
            if float(row["delta_nll_a"]) > float(best["delta_nll_a"]):
                best = row
            if threshold is not None and float(row["delta_nll_a"]) >= threshold:
                best = row
                break
        selected.append(best)
        probes.append(used)
    future_improvement = [
        float(static[int(row["query_id"])]["mean_nll_b"]) - float(row["mean_nll_b"])
        for row in selected
    ]
    return {
        "candidate_ordering": ordering,
        "policy": (
            "best_observed_A_within_budget"
            if threshold is None
            else f"first_delta_A_ge_{threshold:g}_else_best_seen"
        ),
        "max_probe_budget": budget,
        "mean_A_probes": float(np.mean(probes)),
        "selection_uses_future_B": False,
        "quality_on_future_B": selection_quality(selected),
        "mean_nll_improvement_over_static_top1": float(np.mean(future_improvement)),
        "improvement_over_static_bootstrap95": bootstrap_ci(
            future_improvement, samples=bootstrap_samples, seed=seed
        ),
        "wins_over_static_top1": sum(value > 0 for value in future_improvement),
        "losses_to_static_top1": sum(value < 0 for value in future_improvement),
        "ties_with_static_top1": sum(value == 0 for value in future_improvement),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("past_only") or summary.get("query_book_future_block_violations") != 0:
        raise ValueError("verified past-only data required")
    metadata = {
        int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")
    }
    scope_ids = np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r")
    original_centers = np.load(
        data_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    rows = read_jsonl(args.rows)
    by_query_all = defaultdict(list)
    for row in rows:
        by_query_all[int(row["query_id"])].append(row)
    random95 = {
        query_id: float(
            np.quantile(
                [
                    float(row["delta_nll_b"])
                    for row in group
                    if "random" in row["origins"]
                ],
                0.95,
            )
        )
        for query_id, group in by_query_all.items()
    }

    enriched = []
    for row in rows:
        if args.method not in row["origins"]:
            continue
        query_id = int(row["query_id"])
        block_ids = np.asarray(row["block_ids"], dtype=np.int64)
        scopes = np.asarray(scope_ids[block_ids], dtype=np.int64)
        valid = scopes[scopes >= 0]
        if not len(valid):
            scope = -1
        else:
            unique, counts = np.unique(valid, return_counts=True)
            scope = int(unique[int(np.argmax(counts))])
        query_scope = int(metadata[query_id]["book_index"])
        same_scope = scope == query_scope
        distance = None
        distance_label = None
        if same_scope:
            positions = np.asarray(original_centers[block_ids], dtype=np.int64)
            positions = positions[(scopes == query_scope) & (positions >= 0)]
            center = float(np.median(positions))
            local_start = int(metadata[query_id]["local_context_start_token"])
            if center >= local_start:
                raise RuntimeError("future block found in past-only memory")
            distance = local_start - center
            for low, high, label in DISTANCE_BINS:
                if low <= distance < high:
                    distance_label = label
                    break
        item = dict(row)
        item.update(
            {
                "candidate_scope": scope,
                "same_scope": same_scope,
                "distance_tokens": distance,
                "distance_bin": distance_label,
                "future_event": float(row["delta_nll_b"]) > random95[query_id],
            }
        )
        enriched.append(item)

    same = [row for row in enriched if row["same_scope"]]
    other = [row for row in enriched if not row["same_scope"]]
    table = np.asarray(
        [
            [sum(row["future_event"] for row in same), sum(not row["future_event"] for row in same)],
            [sum(row["future_event"] for row in other), sum(not row["future_event"] for row in other)],
        ],
        dtype=np.int64,
    )
    odds = fisher_exact(table)
    distance_groups = {
        label: [row for row in same if row["distance_bin"] == label]
        for _, _, label in DISTANCE_BINS
    }
    policy_groups = defaultdict(list)
    for row in enriched:
        policy_groups[int(row["query_id"])].append(row)
    static = {
        query_id: min(
            group,
            key=lambda row: (
                int(row["origins"][args.method]), int(row["window_start"])
            ),
        )
        for query_id, group in policy_groups.items()
    }
    policies = []
    policy_index = 0
    for ordering in (
        "global_rank",
        "same_scope_then_rank",
        "same_scope_then_distance_then_rank",
        "same_scope_multiscale_round_robin",
    ):
        for budget in BUDGETS:
            for threshold in (None, 0.0, 0.02, 0.05, 0.1):
                policies.append(
                    evaluate_policy(
                        policy_groups,
                        ordering=ordering,
                        budget=budget,
                        threshold=threshold,
                        static=static,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + policy_index,
                    )
                )
                policy_index += 1

    output = {
        "source": "PG19 past-only causal utility structure and bounded probes",
        "protocol": {
            "past_only": True,
            "predefined_source": False,
            "query_book_future_block_violations": 0,
            "future_B_is_never_used_for_selection": True,
            "same_scope_is_observed_metadata_for_diagnosis": True,
            "distance_is_to_the_start_of_the_512_token_local_context": True,
        },
        "retrieval_method": args.method,
        "candidate_rows": len(enriched),
        "scope_utility": {
            "same_scope": summarize_rows(same),
            "other_scope": summarize_rows(other),
            "future_event_contingency": table.tolist(),
            "future_event_odds_ratio": float(odds.statistic),
            "future_event_fisher_p": float(odds.pvalue),
        },
        "same_scope_by_past_distance": {
            label: summarize_rows(group) for label, group in distance_groups.items()
        },
        "static_top1": selection_quality(list(static.values())),
        "bounded_probe_policies": policies,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
