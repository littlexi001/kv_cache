from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest, fisher_exact, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prefix maturity, scale survival, and retrieval complementarity."
    )
    parser.add_argument("--text_rows", required=True)
    parser.add_argument("--qk_rows")
    parser.add_argument("--ppl_rows")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--source_name", default="real XSum continuation")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def paired_binary(before: list[bool], after: list[bool]) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("paired arrays differ")
    wins = sum((not left) and right for left, right in zip(before, after))
    losses = sum(left and (not right) for left, right in zip(before, after))
    discordant = wins + losses
    p_value = (
        float(binomtest(min(wins, losses), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "before_rate": mean(before),
        "after_rate": mean(after),
        "absolute_gain": mean(after) - mean(before),
        "wins": wins,
        "losses": losses,
        "two_sided_exact_p": p_value,
    }


def paired_mean(
    before: list[float],
    after: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(after, dtype=np.float64) - np.asarray(
        before, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "before_mean": mean(before),
        "after_mean": mean(after),
        "mean_gain": float(differences.mean()),
        "bootstrap95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def jaccard(left: list[int], right: list[int], topk: int) -> float:
    left_set = set(left[:topk])
    right_set = set(right[:topk])
    return len(left_set & right_set) / len(left_set | right_set)


def normalized_scale_ids(row: dict[str, Any], topk: int) -> list[int]:
    base_blocks = int(row["base_blocks"])
    output = []
    for block_id in row["top_block_ids"][:topk]:
        block_id = int(block_id)
        output.append(block_id if block_id < base_blocks else -1 - (block_id - base_blocks))
    return output


def analyze_rows(
    rows: list[dict[str, Any]], *, topk: int, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    hit_key = f"source_any_at_{topk}"
    recall_key = f"source_recall_at_{topk}"
    memories = sorted({int(row["memory_tokens"]) for row in rows})
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    query_ids = sorted({int(row["query_id"]) for row in rows})
    lookup = {
        (
            str(row["method"]),
            int(row["memory_tokens"]),
            int(row["prefix_tokens"]),
            int(row["query_id"]),
        ): row
        for row in rows
    }

    prefix_maturity = []
    prefix_transitions = []
    for memory in memories:
        for method in methods:
            before_rows = [lookup[(method, memory, prefixes[0], q)] for q in query_ids]
            after_rows = [lookup[(method, memory, prefixes[-1], q)] for q in query_ids]
            binary = paired_binary(
                [bool(row[hit_key]) for row in before_rows],
                [bool(row[hit_key]) for row in after_rows],
            )
            continuous = paired_mean(
                [float(row[recall_key]) for row in before_rows],
                [float(row[recall_key]) for row in after_rows],
                samples=bootstrap_samples,
                seed=seed + memory + len(prefix_maturity),
            )
            first_hits = []
            for query_id in query_ids:
                first_hits.append(
                    next(
                        (
                            prefix
                            for prefix in prefixes
                            if bool(lookup[(method, memory, prefix, query_id)][hit_key])
                        ),
                        None,
                    )
                )
            prefix_maturity.append(
                {
                    "memory_tokens": memory,
                    "method": method,
                    "prefix_comparison": f"{prefixes[0]}->{prefixes[-1]}",
                    "any_hit": binary,
                    "source_recall": continuous,
                    "earliest_topk_hit": {
                        str(prefix): sum(item == prefix for item in first_hits) / len(first_hits)
                        for prefix in prefixes
                    },
                    "never_hit_rate": sum(item is None for item in first_hits) / len(first_hits),
                }
            )
            for left, right in zip(prefixes, prefixes[1:]):
                left_rows = [lookup[(method, memory, left, q)] for q in query_ids]
                right_rows = [lookup[(method, memory, right, q)] for q in query_ids]
                left_hits = [bool(row[hit_key]) for row in left_rows]
                right_hits = [bool(row[hit_key]) for row in right_rows]
                hit_count = sum(left_hits)
                miss_count = len(left_hits) - hit_count
                prefix_transitions.append(
                    {
                        "memory_tokens": memory,
                        "method": method,
                        "transition": f"{left}->{right}",
                        "hit_persistence": (
                            sum(a and b for a, b in zip(left_hits, right_hits)) / hit_count
                            if hit_count
                            else None
                        ),
                        "new_discovery_given_miss": (
                            sum((not a) and b for a, b in zip(left_hits, right_hits))
                            / miss_count
                            if miss_count
                            else None
                        ),
                        "loss_given_hit": (
                            sum(a and (not b) for a, b in zip(left_hits, right_hits))
                            / hit_count
                            if hit_count
                            else None
                        ),
                        "topk_jaccard": mean(
                            jaccard(a["top_block_ids"], b["top_block_ids"], topk)
                            for a, b in zip(left_rows, right_rows)
                        ),
                    }
                )

    scale_survival = []
    for prefix in prefixes:
        for method in methods:
            small_rows = [lookup[(method, memories[0], prefix, q)] for q in query_ids]
            large_rows = [lookup[(method, memories[-1], prefix, q)] for q in query_ids]
            small_hits = [bool(row[hit_key]) for row in small_rows]
            large_hits = [bool(row[hit_key]) for row in large_rows]
            small_count = sum(small_hits)
            scale_survival.append(
                {
                    "method": method,
                    "prefix_tokens": prefix,
                    "scale_comparison": f"{memories[0]}->{memories[-1]}",
                    "paired_hit_change": paired_binary(small_hits, large_hits),
                    "survival_given_small_hit": (
                        sum(a and b for a, b in zip(small_hits, large_hits)) / small_count
                        if small_count
                        else None
                    ),
                    "topk_jaccard": mean(
                        jaccard(
                            normalized_scale_ids(a, topk),
                            normalized_scale_ids(b, topk),
                            topk,
                        )
                        for a, b in zip(small_rows, large_rows)
                    ),
                }
            )

    rank_maturity = []
    for memory in memories:
        for method in methods:
            rank_rows = [
                row
                for row in rows
                if int(row["memory_tokens"]) == memory
                and str(row["method"]) == method
                and row.get("best_gold_rank") is not None
            ]
            if not rank_rows:
                continue
            correlation = spearmanr(
                [int(row["prefix_tokens"]) for row in rank_rows],
                [math.log1p(int(row["best_gold_rank"])) for row in rank_rows],
            )
            rank_maturity.append(
                {
                    "memory_tokens": memory,
                    "method": method,
                    "prefix_vs_log_best_gold_rank_spearman": float(correlation.statistic),
                    "p_value": float(correlation.pvalue),
                }
            )
    return {
        "memory_tokens": memories,
        "prefix_tokens": prefixes,
        "methods": methods,
        "queries": len(query_ids),
        "topk": topk,
        "prefix_maturity": prefix_maturity,
        "prefix_transitions": prefix_transitions,
        "scale_survival": scale_survival,
        "rank_maturity": rank_maturity,
    }


def complementarity(
    text_rows: list[dict[str, Any]], qk_rows: list[dict[str, Any]], topk: int
) -> list[dict[str, Any]]:
    hit_key = f"source_any_at_{topk}"
    text_lookup = {
        (
            int(row["query_id"]),
            int(row["memory_tokens"]),
            int(row["prefix_tokens"]),
            str(row["method"]),
        ): row
        for row in text_rows
    }
    qk_lookup = {
        (
            int(row["query_id"]),
            int(row["memory_tokens"]),
            int(row["prefix_tokens"]),
            str(row["method"]),
        ): row
        for row in qk_rows
    }
    memories = sorted({int(row["memory_tokens"]) for row in qk_rows})
    prefixes = sorted({int(row["prefix_tokens"]) for row in qk_rows})
    query_ids = sorted({int(row["query_id"]) for row in qk_rows})
    text_methods = sorted({str(row["method"]) for row in text_rows})
    qk_methods = sorted({str(row["method"]) for row in qk_rows})
    rows = []
    for memory in memories:
        for prefix in prefixes:
            for text_method in text_methods:
                for qk_method in qk_methods:
                    text_hits = [
                        bool(text_lookup[(q, memory, prefix, text_method)][hit_key])
                        for q in query_ids
                    ]
                    qk_hits = [
                        bool(qk_lookup[(q, memory, prefix, qk_method)][hit_key])
                        for q in query_ids
                    ]
                    rows.append(
                        {
                            "memory_tokens": memory,
                            "prefix_tokens": prefix,
                            "text_method": text_method,
                            "qk_method": qk_method,
                            "text_hit": mean(text_hits),
                            "qk_hit": mean(qk_hits),
                            "union_hit": mean(a or b for a, b in zip(text_hits, qk_hits)),
                            "qk_rescue_text_miss": mean(
                                (not a) and b for a, b in zip(text_hits, qk_hits)
                            ),
                        }
                    )
    return rows


def qk_event_gates(
    text_rows: list[dict[str, Any]],
    qk_rows: list[dict[str, Any]],
    ppl_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    memory = max(int(row["memory_tokens"]) for row in qk_rows)
    prefixes = sorted({int(row["prefix_tokens"]) for row in qk_rows})
    prefix = prefixes[-1]
    previous = prefixes[-2]
    qk_method = "qk_raw_max"
    text_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in text_rows
        if int(row["memory_tokens"]) == memory
    }
    qk_lookup = {
        (int(row["query_id"]), int(row["prefix_tokens"]), str(row["method"])): row
        for row in qk_rows
        if int(row["memory_tokens"]) == memory
    }
    ppl_lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in ppl_rows
    }
    query_ids = sorted({int(row["query_id"]) for row in ppl_rows})
    feature_values: dict[str, list[int]] = {
        "qk_top8_overlap_e5_top8": [],
        "qk_top8_overlap_hybrid_top8": [],
        "qk_top8_recurrence_in_previous_top512": [],
    }
    hits = []
    qk_deltas = []
    for query_id in query_ids:
        qk_top8 = set(
            qk_lookup[(query_id, prefix, qk_method)]["top_block_ids"][:8]
        )
        feature_values["qk_top8_overlap_e5_top8"].append(
            len(qk_top8 & set(text_lookup[(query_id, prefix, "e5")]["top_block_ids"][:8]))
        )
        feature_values["qk_top8_overlap_hybrid_top8"].append(
            len(
                qk_top8
                & set(
                    text_lookup[(query_id, prefix, "bm25_e5_rrf")][
                        "top_block_ids"
                    ][:8]
                )
            )
        )
        feature_values["qk_top8_recurrence_in_previous_top512"].append(
            len(
                qk_top8
                & set(
                    qk_lookup[(query_id, previous, qk_method)]["top_block_ids"][:512]
                )
            )
        )
        hits.append(bool(ppl_lookup[(query_id, qk_method)]["source_any_hit"]))
        qk_deltas.append(
            float(ppl_lookup[(query_id, qk_method)]["mean_nll"])
            - float(ppl_lookup[(query_id, "query_only")]["mean_nll"])
        )

    output = []
    for feature, values in feature_values.items():
        for threshold in (1, 2):
            gate = [value >= threshold for value in values]
            triggered = sum(gate)
            if not triggered:
                continue
            trigger_hits = sum(flag and hit for flag, hit in zip(gate, hits))
            other_hits = sum((not flag) and hit for flag, hit in zip(gate, hits))
            table = [
                [trigger_hits, triggered - trigger_hits],
                [other_hits, len(gate) - triggered - other_hits],
            ]
            fisher = fisher_exact(table)
            odds_ratio = float(fisher.statistic)
            gate_item: dict[str, Any] = {
                "feature": feature,
                "threshold": threshold,
                "triggered_queries": triggered,
                "source_hit_given_trigger": trigger_hits / triggered,
                "source_hit_without_trigger": (
                    other_hits / (len(gate) - triggered)
                    if triggered < len(gate)
                    else None
                ),
                "fisher_exact_odds_ratio": (
                    odds_ratio if math.isfinite(odds_ratio) else None
                ),
                "fisher_exact_two_sided_p": float(fisher.pvalue),
                "mean_qk_delta_nll_given_trigger": mean(
                    delta for delta, flag in zip(qk_deltas, gate) if flag
                ),
                "mean_qk_delta_nll_without_trigger": mean(
                    delta for delta, flag in zip(qk_deltas, gate) if not flag
                ),
                "gated_policies": [],
            }
            for fallback in ("e5", "bm25_e5_rrf"):
                gated = [
                    float(
                        ppl_lookup[
                            (query_id, qk_method if flag else fallback)
                        ]["mean_nll"]
                    )
                    for query_id, flag in zip(query_ids, gate)
                ]
                fallback_nll = [
                    float(ppl_lookup[(query_id, fallback)]["mean_nll"])
                    for query_id in query_ids
                ]
                comparison = paired_mean(
                    fallback_nll,
                    gated,
                    samples=bootstrap_samples,
                    seed=seed + len(output) * 10 + threshold,
                )
                gate_item["gated_policies"].append(
                    {
                        "policy": f"qk_if_gate_else_{fallback}",
                        "mean_nll": mean(gated),
                        "ppl": math.exp(mean(gated)),
                        "paired_delta_vs_fallback": comparison,
                    }
                )
            output.append(gate_item)
    return output


def main() -> None:
    args = parse_args()
    text_rows = read_jsonl(args.text_rows)
    output: dict[str, Any] = {
        "source": f"paired dynamic and scale properties on {args.source_name}",
        "contains_synthetic_text": False,
        "selection_uses_target": False,
        "text": analyze_rows(
            text_rows,
            topk=args.topk,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    if args.qk_rows:
        qk_rows = read_jsonl(args.qk_rows)
        output["qk"] = analyze_rows(
            qk_rows,
            topk=args.topk,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1,
        )
        output["text_qk_complementarity"] = complementarity(
            text_rows, qk_rows, args.topk
        )
        if args.ppl_rows:
            output["qk_event_gates"] = qk_event_gates(
                text_rows,
                qk_rows,
                read_jsonl(args.ppl_rows),
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 2,
            )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
