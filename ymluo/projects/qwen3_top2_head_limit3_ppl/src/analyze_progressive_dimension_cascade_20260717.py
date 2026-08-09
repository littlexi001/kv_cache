from __future__ import annotations

import argparse
import csv
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
        raise ValueError("expected at least one rank")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 16-to-32-to-64 dimensional PCA screening cascade."
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--stage1_fractions", default="0.1,0.2,0.3,0.4,0.6")
    parser.add_argument("--stage2_fractions", default="0.04,0.06,0.08,0.12,0.2")
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.06,0.08")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cascade_candidates(
    stage_scores: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    stage_fractions: tuple[float, float, float],
) -> torch.Tensor:
    history_count = int(stage_scores[0].numel())
    if any(score.numel() != history_count for score in stage_scores):
        raise ValueError("all stage scores must cover the same token history")
    first_count, second_count, final_count = (
        min(history_count, max(1, math.ceil(fraction * history_count)))
        for fraction in stage_fractions
    )
    if not final_count <= second_count <= first_count:
        raise ValueError("cascade candidate counts must be monotonically decreasing")
    first = torch.topk(stage_scores[0], k=first_count).indices
    second_local = torch.topk(stage_scores[1][first], k=second_count).indices
    second = first[second_local]
    final_local = torch.topk(stage_scores[2][second], k=final_count).indices
    return second[final_local]


def normalized_index_work(
    ranks: tuple[int, int, int], stage_fractions: tuple[float, float, float]
) -> float:
    first_rank, second_rank, final_rank = ranks
    first_fraction, second_fraction, _ = stage_fractions
    work = (
        first_rank
        + (second_rank - first_rank) * first_fraction
        + (final_rank - second_rank) * second_fraction
    )
    return work / final_rank


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ranks = parse_int_list(args.ranks)
    if len(ranks) != 3 or not ranks[0] < ranks[1] < ranks[2]:
        raise ValueError("ranks must contain three strictly increasing values")
    stage1_fractions = parse_float_list(args.stage1_fractions)
    stage2_fractions = parse_float_list(args.stage2_fractions)
    candidate_fractions = parse_float_list(args.candidate_fractions)
    configurations = tuple(
        (first, second, final)
        for first, second, final in product(
            stage1_fractions, stage2_fractions, candidate_fractions
        )
        if args.top_fraction <= final <= second <= first <= 1.0
    )
    if not configurations:
        raise ValueError("no valid cascade configurations")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = load_records(args.trace_path)
    rows: list[dict[str, Any]] = []

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
                covariance_basis(head_key[:: args.sample_stride], ranks[-1]).flip(-1)
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
            head_query = query[query_head]
            exact = key[kv_head] @ head_query * scaling
            current_score = (
                all_key[kv_head, -1] @ head_query * scaling
            ).view(1)
            attention = torch.softmax(
                torch.cat((exact, current_score)), dim=-1
            )[:history_count]
            true_indices = torch.topk(exact, k=top_count).indices
            oracle_mass = float(attention[true_indices].sum().item())
            stage_scores = tuple(
                (
                    indexed_key[kv_head, :, :rank]
                    @ projected_query[kv_head, group_head, :rank]
                )
                * scaling
                for rank in ranks
            )
            for fractions in configurations:
                candidates = cascade_candidates(stage_scores, fractions)
                selected = exact_rerank(exact, candidates, top_count)
                rows.append(
                    {
                        "topic": args.topic,
                        "record_index": record_index,
                        "layer": layer,
                        "query_head": query_head,
                        "stage1_fraction": fractions[0],
                        "stage2_fraction": fractions[1],
                        "candidate_fraction": fractions[2],
                        "normalized_index_work": normalized_index_work(
                            ranks, fractions
                        ),
                        "candidate_ratio": candidates.numel() / history_count,
                        **selection_metrics(
                            selected,
                            true_indices,
                            attention,
                            oracle_mass,
                        ),
                    }
                )
        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={layer}",
            flush=True,
        )
        del query, all_key, key, bases, projected_key, indexed_key, projected_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["stage1_fraction"]),
                float(row["stage2_fraction"]),
                float(row["candidate_fraction"]),
            )
        ].append(row)
    summaries = []
    for fractions, items in grouped.items():
        summaries.append(
            {
                "stage1_fraction": fractions[0],
                "stage2_fraction": fractions[1],
                "candidate_fraction": fractions[2],
                "normalized_index_work": normalized_index_work(ranks, fractions),
                "cases": len(items),
                "top2_recall": summarize(float(row["top2_recall"]) for row in items),
                "top2_attention_mass_recall": summarize(
                    float(row["top2_attention_mass_recall"]) for row in items
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
    write_csv(args.output_dir / "per_head_query.csv", rows)
    report = {
        "method": "progressive_dimension_cascade",
        "topic": args.topic,
        "ranks": ranks,
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
