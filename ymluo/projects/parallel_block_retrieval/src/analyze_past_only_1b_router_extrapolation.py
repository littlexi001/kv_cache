from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import betabinom, spearmanr
from sklearn.metrics import brier_score_loss


DEPTHS = (1, 3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a nested-scope exceedance model on real 10M-100M PG19 "
            "and extrapolate the upper router to 1B without claiming a real 1B run."
        )
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_memory_tokens", type=int, default=1_000_000_000)
    parser.add_argument("--frontier_blocks", type=int, default=512)
    parser.add_argument("--event_refresh_rate", type=float, default=0.20)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def fit_beta_binomial_prior(counts: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)

    def objective(log_parameters: np.ndarray) -> float:
        alpha, beta = np.exp(log_parameters)
        likelihood = betaln(counts + alpha, totals - counts + beta) - betaln(
            alpha, beta
        )
        return -float(likelihood.sum())

    observed = float((counts.sum() + 0.5) / (totals.sum() + 1.0))
    concentration = 20.0
    initial = np.log(
        [max(observed * concentration, 1.0e-3), max((1.0 - observed) * concentration, 1.0e-3)]
    )
    result = minimize(objective, initial, method="L-BFGS-B", bounds=[(-12, 16), (-12, 16)])
    if not result.success:
        raise RuntimeError(f"beta-binomial prior fit failed: {result.message}")
    alpha, beta = np.exp(result.x)
    return float(alpha), float(beta)


def predictive_topd(
    current_count: np.ndarray,
    current_total: np.ndarray,
    future_total: np.ndarray,
    *,
    depth: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    additions = future_total - current_total
    remaining_budget = depth - 1 - current_count
    probabilities = np.zeros(len(current_count), dtype=np.float64)
    possible = remaining_budget >= 0
    probabilities[possible] = betabinom.cdf(
        remaining_budget[possible],
        additions[possible],
        current_count[possible] + alpha,
        current_total[possible] - current_count[possible] + beta,
    )
    return probabilities


def predictive_count_mean(
    current_count: np.ndarray,
    current_total: np.ndarray,
    future_total: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    posterior_mean = (current_count + alpha) / (current_total + alpha + beta)
    return current_count + (future_total - current_total) * posterior_mean


def cluster_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    unique = np.unique(groups)
    by_group = np.asarray([values[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(samples, len(unique)))
    return interval(by_group[indices].mean(axis=1))


def router_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        if row["method"] != "hier_bm25_scope1":
            continue
        key = (int(row["query_id"]), int(row["prefix_tokens"]), int(row["memory_tokens"]))
        if key in seen:
            raise ValueError(f"duplicate router geometry for {key}")
        seen.add(key)
        active = int(row["active_scopes"])
        rank = int(row["true_scope_rank"])
        records.append(
            {
                "query_id": key[0],
                "prefix_tokens": key[1],
                "memory_tokens": key[2],
                "competitor_scopes": rank - 1,
                "distractor_scopes": active - 1,
            }
        )
    return records


def nested_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["query_id"], row["prefix_tokens"])].append(row)
    violations = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: row["memory_tokens"])
        for previous, current in zip(ordered, ordered[1:]):
            if current["distractor_scopes"] < previous["distractor_scopes"] or current[
                "competitor_scopes"
            ] < previous["competitor_scopes"]:
                violations.append(
                    {
                        "query_id": key[0],
                        "prefix_tokens": key[1],
                        "previous": previous,
                        "current": current,
                    }
                )
    return {
        "query_state_trajectories": len(grouped),
        "nested_monotonic_violations": len(violations),
        "examples": violations[:10],
    }


def validation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["query_id"], row["prefix_tokens"], row["memory_tokens"]): row
        for row in records
    }
    scales = sorted({row["memory_tokens"] for row in records})
    output = []
    for current_scale, future_scale in zip(scales, scales[1:]):
        current = [row for row in records if row["memory_tokens"] == current_scale]
        future = [
            by_key[(row["query_id"], row["prefix_tokens"], future_scale)]
            for row in current
        ]
        current_count = np.asarray([row["competitor_scopes"] for row in current])
        current_total = np.asarray([row["distractor_scopes"] for row in current])
        future_count = np.asarray([row["competitor_scopes"] for row in future])
        future_total = np.asarray([row["distractor_scopes"] for row in future])
        alpha, beta = fit_beta_binomial_prior(current_count, current_total)
        predicted_count = predictive_count_mean(
            current_count,
            current_total,
            future_total,
            alpha=alpha,
            beta=beta,
        )
        correlation = spearmanr(future_count, predicted_count).statistic
        item: dict[str, Any] = {
            "current_memory_tokens": current_scale,
            "future_memory_tokens": future_scale,
            "events": len(current),
            "prior_alpha": alpha,
            "prior_beta": beta,
            "actual_mean_competitors": float(future_count.mean()),
            "predicted_mean_competitors": float(predicted_count.mean()),
            "competitor_count_mae": float(np.abs(future_count - predicted_count).mean()),
            "competitor_count_spearman": float(correlation),
            "topd": {},
        }
        for depth in DEPTHS:
            probability = predictive_topd(
                current_count,
                current_total,
                future_total,
                depth=depth,
                alpha=alpha,
                beta=beta,
            )
            actual = (future_count < depth).astype(np.float64)
            item["topd"][str(depth)] = {
                "predicted_recall": float(probability.mean()),
                "actual_recall": float(actual.mean()),
                "brier": float(brier_score_loss(actual, probability)),
            }
        output.append(item)
    return output


def observed_candidate_costs(
    rows: list[dict[str, Any]],
    *,
    max_scale: int,
    target_blocks: int,
    frontier_blocks: int,
    event_refresh_rate: float,
) -> list[dict[str, Any]]:
    output = []
    observed_depths = (1, 3, 8, 16, 32)
    observed: dict[int, np.ndarray] = {}
    for depth in observed_depths:
        selected = [
            row
            for row in rows
            if int(row["memory_tokens"]) == max_scale
            and row["method"] == f"hier_bm25_scope{depth}"
        ]
        observed[depth] = np.asarray(
            [float(row["candidate_blocks"]) for row in selected]
        )
    large_depth_blocks_per_scope = float(
        np.mean(
            np.concatenate(
                [observed[depth] / depth for depth in (8, 16, 32)]
            )
        )
    )
    for depth in DEPTHS:
        if depth in observed:
            candidate_blocks = observed[depth]
            mean_candidates = float(candidate_blocks.mean())
            p95_candidates = float(np.quantile(candidate_blocks, 0.95))
            basis = "observed at 100M"
        else:
            mean_candidates = large_depth_blocks_per_scope * depth
            p95_candidates = None
            basis = "projected from observed D8/D16/D32 blocks per selected scope"
        event_access = frontier_blocks + event_refresh_rate * mean_candidates
        output.append(
            {
                "scope_depth": depth,
                "observed_100m_mean_candidate_blocks": mean_candidates
                if depth in observed
                else None,
                "observed_100m_p95_candidate_blocks": p95_candidates,
                "candidate_blocks_basis": basis,
                "projected_blocks_per_selected_scope": large_depth_blocks_per_scope,
                "mean_candidate_blocks_using_100m_scope_sizes": mean_candidates,
                "projected_1b_candidate_fraction_if_scope_sizes_stable": mean_candidates
                / target_blocks,
                "projected_1b_access_reduction_every_hierarchical_refresh": target_blocks
                / mean_candidates,
                "projected_1b_mean_blocks_with_event_refresh": event_access,
                "projected_1b_access_reduction_with_event_refresh": target_blocks
                / event_access,
            }
        )
    return output


def extrapolate(
    records: list[dict[str, Any]],
    *,
    target_tokens: int,
    block_tokens: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    max_scale = max(row["memory_tokens"] for row in records)
    current = [row for row in records if row["memory_tokens"] == max_scale]
    groups = np.asarray([row["query_id"] for row in current])
    prefixes = np.asarray([row["prefix_tokens"] for row in current])
    counts = np.asarray([row["competitor_scopes"] for row in current])
    totals = np.asarray([row["distractor_scopes"] for row in current])
    scale_ratio = target_tokens / float(max_scale)
    target_totals = np.maximum(
        totals.astype(np.int64),
        np.rint((totals.astype(np.float64) + 1.0) * scale_ratio).astype(np.int64)
        - 1,
    )
    alpha, beta = fit_beta_binomial_prior(counts, totals)
    expected_count = predictive_count_mean(
        counts, totals, target_totals, alpha=alpha, beta=beta
    )
    output: dict[str, Any] = {
        "source_memory_tokens": max_scale,
        "target_memory_tokens": target_tokens,
        "target_blocks": target_tokens // block_tokens,
        "events": len(current),
        "query_groups": len(np.unique(groups)),
        "observed_mean_active_scopes_at_100m": float((totals + 1).mean()),
        "projected_mean_active_scopes_at_1b": float((target_totals + 1).mean()),
        "prior_alpha": alpha,
        "prior_beta": beta,
        "predicted_mean_wrong_scopes_above_true_scope": float(expected_count.mean()),
        "topd": {},
    }
    for index, depth in enumerate(DEPTHS):
        probability = predictive_topd(
            counts,
            totals,
            target_totals,
            depth=depth,
            alpha=alpha,
            beta=beta,
        )
        per_prefix = {}
        for prefix in sorted(set(prefixes.tolist())):
            selected = prefixes == prefix
            per_prefix[str(prefix)] = {
                "predicted_recall": float(probability[selected].mean()),
                "query_cluster_bootstrap95": cluster_bootstrap(
                    probability[selected],
                    groups[selected],
                    samples=bootstrap_samples,
                    seed=seed + 1000 * index + prefix,
                ),
            }
        output["topd"][str(depth)] = {
            "predicted_recall": float(probability.mean()),
            "query_cluster_bootstrap95": cluster_bootstrap(
                probability,
                groups,
                samples=bootstrap_samples,
                seed=seed + 1000 * index,
            ),
            "by_prefix": per_prefix,
        }
    output["minimum_depth_for_mean_predicted_recall"] = {}
    output["minimum_depth_by_prefix_for_predicted_recall"] = {}
    for threshold in (0.80, 0.90, 0.95):
        matching = [
            depth
            for depth in DEPTHS
            if output["topd"][str(depth)]["predicted_recall"] >= threshold
        ]
        output["minimum_depth_for_mean_predicted_recall"][str(threshold)] = (
            min(matching) if matching else None
        )
        output["minimum_depth_by_prefix_for_predicted_recall"][str(threshold)] = {}
        for prefix in sorted(set(prefixes.tolist())):
            prefix_matching = [
                depth
                for depth in DEPTHS
                if output["topd"][str(depth)]["by_prefix"][str(prefix)][
                    "predicted_recall"
                ]
                >= threshold
            ]
            output["minimum_depth_by_prefix_for_predicted_recall"][str(threshold)][
                str(prefix)
            ] = min(prefix_matching) if prefix_matching else None
    return output


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.rows)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    records = router_records(rows)
    audit = nested_audit(records)
    if audit["nested_monotonic_violations"]:
        raise ValueError("scope competitor counts are not nested and monotonic")
    block_tokens = int(summary["data_summary"]["block_tokens"])
    max_scale = max(row["memory_tokens"] for row in records)
    target_blocks = args.target_memory_tokens // block_tokens
    storage_multiplier = args.target_memory_tokens / max_scale
    output = {
        "source": "posterior 1B scope-router extrapolation from real nested PG19 10M-100M",
        "claim_boundary": (
            "This is a validated statistical extrapolation, not a real 1B index run. "
            "It assumes future PG19 book scopes are exchangeable and scope sizes stay bounded."
        ),
        "protocol": {
            "contains_synthetic_text": False,
            "contains_repeated_distractor_text": False,
            "past_only": True,
            "predefined_source": False,
            "selection_uses_target": False,
            "fixed_100m_idf_across_nested_validation_scales": True,
            "target_memory_tokens": args.target_memory_tokens,
            "block_tokens": block_tokens,
            "frontier_blocks": args.frontier_blocks,
            "event_refresh_rate": args.event_refresh_rate,
            "beta_binomial_empirical_bayes": True,
            "bootstrap_unit": "query",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "nested_audit": audit,
        "nested_scale_validation": validation(records),
        "one_billion_extrapolation": extrapolate(
            records,
            target_tokens=args.target_memory_tokens,
            block_tokens=block_tokens,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "candidate_access_projection": observed_candidate_costs(
            rows,
            max_scale=max_scale,
            target_blocks=target_blocks,
            frontier_blocks=args.frontier_blocks,
            event_refresh_rate=args.event_refresh_rate,
        ),
        "storage_projection": {
            "observed_100m_block_index_bytes": int(summary["block_index_bytes"]),
            "observed_100m_scope_index_bytes": int(summary["scope_index_bytes"]),
            "linear_1b_block_index_bytes": int(
                round(summary["block_index_bytes"] * storage_multiplier)
            ),
            "linear_1b_scope_index_bytes": int(
                round(summary["scope_index_bytes"] * storage_multiplier)
            ),
            "note": "Linear storage only; it is not measured SSD or RAM residency.",
        },
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
