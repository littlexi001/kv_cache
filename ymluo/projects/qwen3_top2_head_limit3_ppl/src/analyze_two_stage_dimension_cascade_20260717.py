from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import torch

from analyze_residual_certified_pca_20260717 import (
    covariance_basis,
    exact_rerank,
    quantize_dequantize_int4,
    selection_metrics,
    summarize,
)
from analyze_verify_then_expand_pca_20260717 import load_records, parse_float_list


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not values:
        raise ValueError("expected at least one prefix rank")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage prefix-to-PCA64 candidate funnels."
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix_ranks", default="16,32,48")
    parser.add_argument("--stage_fractions", default="0.08,0.12,0.2,0.3,0.4,0.6")
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.06,0.08")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--sample_stride", type=int, default=32)
    return parser.parse_args()


def two_stage_candidates(
    prefix_scores: torch.Tensor,
    full_scores: torch.Tensor,
    stage_fraction: float,
    candidate_fraction: float,
) -> torch.Tensor:
    history_count = int(prefix_scores.numel())
    stage_count = min(history_count, max(1, math.ceil(stage_fraction * history_count)))
    candidate_count = min(
        stage_count, max(1, math.ceil(candidate_fraction * history_count))
    )
    if candidate_fraction > stage_fraction:
        raise ValueError("candidate fraction cannot exceed stage fraction")
    stage = torch.topk(prefix_scores, k=stage_count).indices
    local = torch.topk(full_scores[stage], k=candidate_count).indices
    return stage[local]


def normalized_work(prefix_rank: int, stage_fraction: float) -> float:
    return (prefix_rank + (64 - prefix_rank) * stage_fraction) / 64.0


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    prefix_ranks = parse_int_list(args.prefix_ranks)
    stage_fractions = parse_float_list(args.stage_fractions)
    candidate_fractions = parse_float_list(args.candidate_fractions)
    configurations = tuple(
        (rank, stage, candidate)
        for rank, stage, candidate in product(
            prefix_ranks, stage_fractions, candidate_fractions
        )
        if 0 < rank < 64 and args.top_fraction <= candidate <= stage <= 1.0
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = load_records(args.trace_path)
    grouped: dict[tuple[int, float, float], list[dict[str, float]]] = defaultdict(list)

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0, :]
        all_key = record["key"].to(device).float()[0]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(all_key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(all_key.shape[1]) - 1
        key = all_key[:, :history_count]
        top_count = max(1, math.ceil(args.top_fraction * history_count))
        bases = torch.stack(
            [
                covariance_basis(head_key[:: args.sample_stride], 64).flip(-1)
                for head_key in key
            ]
        )
        projected_key = torch.einsum("hkd,hdm->hkm", key, bases)
        indexed_key = quantize_dequantize_int4(projected_key)
        grouped_query = query.reshape(kv_heads, groups, query.shape[-1])
        projected_query = torch.einsum("hgd,hdm->hgm", grouped_query, bases)

        for query_head in range(query_heads):
            kv_head = query_head // groups
            group_head = query_head % groups
            exact = key[kv_head] @ query[query_head] * scaling
            current_score = (
                all_key[kv_head, -1] @ query[query_head] * scaling
            ).view(1)
            attention = torch.softmax(
                torch.cat((exact, current_score)), dim=-1
            )[:history_count]
            true_indices = torch.topk(exact, k=top_count).indices
            oracle_mass = float(attention[true_indices].sum().item())
            full_scores = (
                indexed_key[kv_head] @ projected_query[kv_head, group_head]
            ) * scaling
            prefix_scores = {
                rank: (
                    indexed_key[kv_head, :, :rank]
                    @ projected_query[kv_head, group_head, :rank]
                )
                * scaling
                for rank in prefix_ranks
            }
            for rank, stage_fraction, candidate_fraction in configurations:
                candidates = two_stage_candidates(
                    prefix_scores[rank],
                    full_scores,
                    stage_fraction,
                    candidate_fraction,
                )
                selected = exact_rerank(exact, candidates, top_count)
                grouped[(rank, stage_fraction, candidate_fraction)].append(
                    selection_metrics(
                        selected, true_indices, attention, oracle_mass
                    )
                )
        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={layer}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries: list[dict[str, Any]] = []
    for (rank, stage_fraction, candidate_fraction), items in grouped.items():
        summaries.append(
            {
                "prefix_rank": rank,
                "stage_fraction": stage_fraction,
                "candidate_fraction": candidate_fraction,
                "normalized_index_work": normalized_work(rank, stage_fraction),
                "cases": len(items),
                "top2_recall": summarize(float(item["top2_recall"]) for item in items),
                "top2_attention_mass_recall": summarize(
                    float(item["top2_attention_mass_recall"]) for item in items
                ),
            }
        )
    summaries.sort(
        key=lambda item: (
            float(item["normalized_index_work"]),
            -float(item["top2_attention_mass_recall"]["mean"]),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "two_stage_dimension_cascade",
        "topic": args.topic,
        "top_fraction": args.top_fraction,
        "configurations": len(configurations),
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
