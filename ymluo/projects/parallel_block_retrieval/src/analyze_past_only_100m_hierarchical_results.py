from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate paired retrieval and reader results for the 100M study."
    )
    parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--ppl128_rows", required=True)
    parser.add_argument("--ppl512_rows", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def ppl_quality(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        micro_nll = sum(float(row["total_nll"]) for row in group) / sum(
            int(row["target_tokens"]) for row in group
        )
        output[method] = {
            "ppl": math.exp(micro_nll),
            "micro_nll": micro_nll,
            "same_scope_any": statistics.fmean(
                float(row["same_scope_any"]) for row in group
            ),
            "same_scope_fraction": statistics.fmean(
                float(row["same_scope_fraction"]) for row in group
            ),
        }
    return output


def paired_nll(
    rows: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    lookup = {
        (int(row["query_id"]), str(row["method"])): float(row["mean_nll"])
        for row in rows
    }


def purity_utility_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in rows
    }
    query_only = {
        int(row["query_id"]): float(row["mean_nll"])
        for row in rows
        if row["method"] == "query_only"
    }
    methods = (
        "global_bm25_unigram",
        "hier_bm25_scope3",
        "hier_bm25_scope8",
        "oracle_scope_bm25",
    )
    per_method = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        purity = [float(row["same_scope_fraction"]) for row in group]
        utility = [
            query_only[int(row["query_id"])] - float(row["mean_nll"])
            for row in group
        ]
        correlation = (
            spearmanr(purity, utility)
            if len(set(purity)) > 1
            else None
        )
        hit_utility = [
            value
            for row, value in zip(group, utility)
            if bool(row["same_scope_any"])
        ]
        miss_utility = [
            value
            for row, value in zip(group, utility)
            if not bool(row["same_scope_any"])
        ]
        per_method[method] = {
            "spearman_purity_vs_reader_utility": finite_or_none(
                correlation.statistic if correlation is not None else math.nan
            ),
            "spearman_pvalue": finite_or_none(
                correlation.pvalue if correlation is not None else math.nan
            ),
            "mean_utility_scope_hit": (
                statistics.fmean(hit_utility) if hit_utility else None
            ),
            "mean_utility_scope_miss": (
                statistics.fmean(miss_utility) if miss_utility else None
            ),
            "scope_hits": len(hit_utility),
            "scope_misses": len(miss_utility),
        }

    paired = {}
    for method in ("hier_bm25_scope3", "hier_bm25_scope8"):
        purity_gain = []
        reader_gain = []
        for query_id in sorted(query_only):
            hierarchy = lookup[(query_id, method)]
            global_row = lookup[(query_id, "global_bm25_unigram")]
            purity_gain.append(
                float(hierarchy["same_scope_fraction"])
                - float(global_row["same_scope_fraction"])
            )
            reader_gain.append(
                float(global_row["mean_nll"]) - float(hierarchy["mean_nll"])
            )
        correlation = spearmanr(purity_gain, reader_gain)
        paired[method] = {
            "spearman_purity_gain_vs_reader_gain": float(correlation.statistic),
            "spearman_pvalue": float(correlation.pvalue),
            "mean_purity_gain": statistics.fmean(purity_gain),
            "mean_reader_nll_gain": statistics.fmean(reader_gain),
        }
    return {"per_method": per_method, "hierarchy_vs_global": paired}
    query_ids = sorted(
        query_id
        for query_id, method in lookup
        if method == method_a and (query_id, method_b) in lookup
    )
    differences = [
        lookup[(query_id, method_a)] - lookup[(query_id, method_b)]
        for query_id in query_ids
    ]
    return {
        "method_a": method_a,
        "method_b": method_b,
        "meaning": "negative favors method_a",
        "queries": len(differences),
        "mean_nll_a_minus_b": statistics.fmean(differences),
        "bootstrap95": bootstrap_mean_ci(differences, samples=samples, seed=seed),
        "a_wins": sum(value < 0 for value in differences),
        "b_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def retrieval_row(
    summary_rows: list[dict[str, Any]], memory_tokens: int, suffix: int, method: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summary_rows
        if int(row["memory_tokens"]) == memory_tokens
        and int(row["state_suffix_tokens"]) == suffix
        and str(row["method"]) == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {(memory_tokens, suffix, method)}")
    return matches[0]


def recall_transitions(
    rows: list[dict[str, Any]], suffix: int, method_a: str, method_b: str
) -> dict[str, int]:
    filtered = [
        row
        for row in rows
        if int(row["memory_tokens"]) == 100_000_000
        and int(row["prefix_tokens"]) == suffix
        and row["method"] in {method_a, method_b}
    ]
    lookup = {
        (int(row["query_id"]), str(row["method"])): bool(
            row["same_scope_any_at_8"]
        )
        for row in filtered
    }
    counts = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for query_id in range(30):
        a = lookup[(query_id, method_a)]
        b = lookup[(query_id, method_b)]
        if a and b:
            counts["both"] += 1
        elif a:
            counts["a_only"] += 1
        elif b:
            counts["b_only"] += 1
        else:
            counts["neither"] += 1
    return counts


def main() -> None:
    args = parse_args()
    retrieval_summary = json.loads(
        Path(args.retrieval_summary).read_text(encoding="utf-8")
    )
    summary_rows = retrieval_summary["retrieval_quality"]
    retrieval_rows = read_jsonl(args.retrieval_rows)
    ppl_rows = {
        "128": read_jsonl(args.ppl128_rows),
        "512": read_jsonl(args.ppl512_rows),
    }
    methods = [
        "global_bm25_unigram",
        "hier_bm25_scope3",
        "hier_bm25_scope8",
    ]
    scale_curves = {}
    for suffix in (128, 512):
        scale_curves[str(suffix)] = [
            {
                "memory_tokens": scale,
                **{
                    method: retrieval_row(summary_rows, scale, suffix, method)
                    for method in methods
                },
            }
            for scale in (9_900_032, 20_000_000, 50_000_000, 100_000_000)
        ]

    selected_100m = {}
    for suffix in (128, 512):
        global_row = retrieval_row(
            summary_rows, 100_000_000, suffix, "global_bm25_unigram"
        )
        selected_100m[str(suffix)] = {}
        for method in methods[1:]:
            row = retrieval_row(summary_rows, 100_000_000, suffix, method)
            selected_100m[str(suffix)][method] = {
                **row,
                "candidate_reduction_x": 1.0 / float(row["mean_candidate_fraction"]),
                "measured_query_speedup_x": float(global_row["mean_query_seconds"])
                / float(row["mean_query_seconds"]),
                "top8_any_retention_vs_global": float(row["same_scope_any_at_8"])
                / float(global_row["same_scope_any_at_8"]),
            }

    comparisons = {}
    for suffix_index, suffix in enumerate((128, 512)):
        rows = ppl_rows[str(suffix)]
        comparisons[str(suffix)] = [
            paired_nll(
                rows,
                method_a,
                method_b,
                samples=args.bootstrap_samples,
                seed=args.seed + suffix_index * 10 + pair_index,
            )
            for pair_index, (method_a, method_b) in enumerate(
                (
                    ("hier_bm25_scope3", "global_bm25_unigram"),
                    ("hier_bm25_scope8", "global_bm25_unigram"),
                    ("hier_bm25_scope3", "hier_bm25_scope8"),
                    ("oracle_scope_bm25", "hier_bm25_scope3"),
                )
            )
        ]

    output = {
        "source": "paired analysis of strict past-only PG19 100M hierarchy",
        "protocol": retrieval_summary["protocol"],
        "offline_index": {
            key: retrieval_summary[key]
            for key in (
                "decode_seconds",
                "block_index_seconds",
                "scope_index_seconds",
                "block_index_bytes",
                "scope_index_bytes",
                "block_features",
                "scope_features",
            )
        },
        "scale_curves": scale_curves,
        "selected_100m": selected_100m,
        "top8_recall_transitions_100m": {
            str(suffix): {
                method: recall_transitions(
                    retrieval_rows, suffix, "global_bm25_unigram", method
                )
                for method in methods[1:]
            }
            for suffix in (128, 512)
        },
        "reader_quality": {
            suffix: ppl_quality(rows) for suffix, rows in ppl_rows.items()
        },
        "paired_reader_comparisons": comparisons,
        "purity_utility_diagnostics": {
            suffix: purity_utility_diagnostics(rows)
            for suffix, rows in ppl_rows.items()
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
