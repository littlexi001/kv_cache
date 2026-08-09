#!/usr/bin/env python3
"""Paired analysis for the real 10M-token PG19 multilevel retrieval run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "doc" / "1b_context_search_research_exploration" / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_tuning_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--reader_rows",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_ppl_s512_rows_20260715.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "pg19_past_only_multilevel_10m_analysis_20260715.json",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_effect(
    candidate: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    query_ids = sorted(set(candidate) & set(baseline))
    if not query_ids:
        raise ValueError("paired methods have no shared query IDs")
    deltas = np.asarray(
        [candidate[qid]["mean_nll"] - baseline[qid]["mean_nll"] for qid in query_ids],
        dtype=np.float64,
    )
    indices = rng.integers(0, deltas.size, size=(samples, deltas.size))
    bootstrap = deltas[indices].mean(axis=1)
    tolerance = 1e-12
    wins = int(np.sum(deltas < -tolerance))
    losses = int(np.sum(deltas > tolerance))
    ties = int(deltas.size - wins - losses)
    sign_p = (
        float(binomtest(wins, wins + losses, p=0.5, alternative="two-sided").pvalue)
        if wins + losses
        else 1.0
    )
    mean_delta = float(deltas.mean())
    return {
        "queries": len(query_ids),
        "delta_definition": "candidate_mean_nll_minus_baseline_mean_nll; negative favors candidate",
        "mean_delta_nll": mean_delta,
        "paired_query_bootstrap95": interval(bootstrap),
        "geometric_mean_ppl_ratio": float(math.exp(mean_delta)),
        "wins_lower_nll": wins,
        "losses_higher_nll": losses,
        "ties": ties,
        "exact_two_sided_sign_test_p": sign_p,
    }


def safe_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return {"n": len(left), "rho": None, "p": None}
    result = spearmanr(left, right)
    return {"n": len(left), "rho": float(result.statistic), "p": float(result.pvalue)}


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> None:
    args = parse_args()
    retrieval_rows = load_jsonl(args.retrieval_rows)
    reader_rows = load_jsonl(args.reader_rows)

    retrieval_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in retrieval_rows:
        if int(row["prefix_tokens"]) != 512:
            continue
        method = str(row["method"])
        retrieval_by_method[method].append(row)
        retrieval_by_key[(int(row["query_id"]), method)] = row

    reader_by_method: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in reader_rows:
        reader_by_method[str(row["method"])][int(row["query_id"])] = row

    method_summary: dict[str, dict[str, Any]] = {}
    query_only = reader_by_method["query_only"]
    for method, rows_by_query in sorted(reader_by_method.items()):
        rows = list(rows_by_query.values())
        nll = np.asarray([float(row["mean_nll"]) for row in rows], dtype=np.float64)
        forward = np.asarray([float(row["forward_seconds"]) for row in rows], dtype=np.float64)
        summary: dict[str, Any] = {
            "queries": len(rows),
            "retrieved_tokens": int(rows[0]["retrieved_tokens"]),
            "mean_nll": float(nll.mean()),
            "ppl": float(math.exp(nll.mean())),
            "mean_reader_forward_seconds": float(forward.mean()),
        }
        retrieval = retrieval_by_method.get(method)
        if retrieval:
            summary.update(
                {
                    "mean_retrieval_seconds": mean(retrieval, "query_seconds"),
                    "mean_candidate_blocks": mean(retrieval, "candidate_blocks"),
                    "mean_same_scope_fraction_at_8": mean(retrieval, "same_scope_fraction_at_8"),
                    "same_scope_within_4k_any_at_8": mean(
                        retrieval, "same_scope_within_4k_any_at_8"
                    ),
                    "prototype_sequential_seconds": mean(retrieval, "query_seconds")
                    + float(forward.mean()),
                }
            )
        if method != "query_only":
            shared = sorted(set(rows_by_query) & set(query_only))
            delta = np.asarray(
                [rows_by_query[q]["mean_nll"] - query_only[q]["mean_nll"] for q in shared],
                dtype=np.float64,
            )
            summary["mean_delta_nll_vs_query_only"] = float(delta.mean())
        method_summary[method] = summary

    best = "multilevel_bm25_book8_segment32"
    baselines = [
        "query_only",
        "random512",
        "global_bm25_unigram",
        "flat_book_bm25_depth8",
        "multilevel_bm25_book8_segment8",
        "multilevel_bm25_book8_segment128",
        "multilevel_bm25_book32_segment32",
    ]
    rng = np.random.default_rng(args.seed)
    paired = {
        baseline: paired_effect(
            reader_by_method[best],
            reader_by_method[baseline],
            samples=args.bootstrap_samples,
            rng=rng,
        )
        for baseline in baselines
    }

    quality_correlations: dict[str, Any] = {}
    pooled: dict[str, list[float]] = defaultdict(list)
    for method, rows_by_query in sorted(reader_by_method.items()):
        if method in {"query_only", "random512"} or method not in retrieval_by_method:
            continue
        utility: list[float] = []
        purity: list[float] = []
        local4k: list[float] = []
        for query_id, reader_row in rows_by_query.items():
            retrieval_row = retrieval_by_key[(query_id, method)]
            gain = float(query_only[query_id]["mean_nll"] - reader_row["mean_nll"])
            utility.append(gain)
            purity.append(float(retrieval_row["same_scope_fraction_at_8"]))
            local4k.append(float(retrieval_row["same_scope_within_4k_any_at_8"]))
            pooled["utility"].append(gain)
            pooled["purity"].append(purity[-1])
            pooled["local4k"].append(local4k[-1])
        quality_correlations[method] = {
            "utility_definition": "query_only_mean_nll_minus_method_mean_nll; positive is useful",
            "purity_vs_reader_utility": safe_spearman(purity, utility),
            "local4k_hit_vs_reader_utility": safe_spearman(local4k, utility),
        }
    quality_correlations["pooled_method_query_rows"] = {
        "note": "descriptive pooled correlation; method/query observations are not independent",
        "purity_vs_reader_utility": safe_spearman(pooled["purity"], pooled["utility"]),
        "local4k_hit_vs_reader_utility": safe_spearman(pooled["local4k"], pooled["utility"]),
    }

    budget_methods = [
        "multilevel_bm25_book8_segment8",
        "multilevel_bm25_book8_segment32",
        "multilevel_bm25_book8_segment128",
    ]
    budget_curve = {method: method_summary[method] for method in budget_methods}
    budget_curve["paired_segment32_vs_segment8"] = paired[
        "multilevel_bm25_book8_segment8"
    ]
    budget_curve["paired_segment32_vs_segment128"] = paired[
        "multilevel_bm25_book8_segment128"
    ]

    payload = {
        "source": "real strict past-only PG19 10M multilevel BM25 plus Qwen3-0.6B reader",
        "protocol": {
            "memory_tokens": 9_900_032,
            "memory_blocks": 154_688,
            "block_tokens": 64,
            "segment_max_blocks": 64,
            "segment_max_tokens": 4_096,
            "state_suffix_tokens": 512,
            "reader_retrieval_tokens": 512,
            "reader_future_target_tokens": 128,
            "selection_uses_target": False,
            "predefined_source": False,
            "contains_synthetic_text": False,
            "query_unit_for_uncertainty": True,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "method_summary": method_summary,
        "paired_best_book8_segment32": paired,
        "intermediate_segment_budget_curve": budget_curve,
        "retrieval_quality_vs_reader_utility": quality_correlations,
        "timing_caveat": (
            "retrieval CPU time and distributed per-batch reader forward time were measured in "
            "different execution stages; prototype_sequential_seconds is only an additive system estimate"
        ),
        "interpretation": {
            "best_observed_method": best,
            "candidate_domain_reduction_vs_global": float(
                method_summary["global_bm25_unigram"]["mean_candidate_blocks"]
                / method_summary[best]["mean_candidate_blocks"]
            ),
            "best_is_significantly_better_than_query_only_by_bootstrap95": bool(
                paired["query_only"]["paired_query_bootstrap95"][1] < 0.0
            ),
            "segment32_beats_adjacent_budgets_significantly": {
                "segment8": bool(
                    paired["multilevel_bm25_book8_segment8"]["paired_query_bootstrap95"][1]
                    < 0.0
                ),
                "segment128": bool(
                    paired["multilevel_bm25_book8_segment128"]["paired_query_bootstrap95"][1]
                    < 0.0
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["interpretation"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
