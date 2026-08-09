#!/usr/bin/env python
"""Mechanism audit for proxy-weighted block-Value tail reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected at least one float")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def normalized_sparse_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected_scores = scores[indices]
    return torch.sum(
        torch.softmax(selected_scores.float(), dim=0).unsqueeze(-1)
        * values[indices].float(),
        dim=0,
    )


def tail_block_means(
    values: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count, head_dim = values.shape
    block_count = math.ceil(token_count / block_size)
    padded = F.pad(
        values.float(),
        (0, 0, 0, block_count * block_size - token_count),
    )
    total_sums = padded.reshape(block_count, block_size, head_dim).sum(dim=1)
    total_counts = torch.full(
        (block_count,),
        block_size,
        dtype=torch.float32,
        device=values.device,
    )
    total_counts[-1] = token_count - (block_count - 1) * block_size
    selected_sums = torch.zeros_like(total_sums)
    selected_counts = torch.zeros_like(total_counts)
    block_ids = indices // block_size
    selected_sums.scatter_add_(
        0,
        block_ids.unsqueeze(-1).expand(-1, head_dim),
        values[indices].float(),
    )
    selected_counts.scatter_add_(0, block_ids, torch.ones_like(block_ids).float())
    counts = total_counts - selected_counts
    means = (total_sums - selected_sums) / counts.clamp_min(1.0).unsqueeze(-1)
    means[counts <= 0.0] = 0.0
    return means, counts


def block_corrected_output(
    exact_scores: torch.Tensor,
    mass_scores: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, float]:
    probabilities = torch.softmax(mass_scores.float(), dim=0)
    selected_mass = probabilities[indices].sum()
    sparse_output = normalized_sparse_output(exact_scores, values, indices)
    block_count = math.ceil(values.shape[0] / block_size)
    block_mass = F.pad(
        probabilities,
        (0, block_count * block_size - values.shape[0]),
    ).reshape(block_count, block_size).sum(dim=1)
    selected_block_mass = torch.zeros_like(block_mass)
    selected_block_mass.scatter_add_(0, indices // block_size, probabilities[indices])
    tail_mass = (block_mass - selected_block_mass).clamp_min(0.0)
    tail_means, _ = tail_block_means(values, indices, block_size)
    output = selected_mass * sparse_output + torch.sum(
        tail_mass.unsqueeze(-1) * tail_means,
        dim=0,
    )
    return output, float(selected_mass)


def output_metrics(output: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(output - full)
            / torch.linalg.vector_norm(full).clamp_min(1.0e-12)
        ),
        "cosine": float(
            F.cosine_similarity(output.float(), full.float(), dim=0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--rate_budget", type=int, default=15)
    parser.add_argument("--fractions", default="0.01,0.02,0.04,0.06")
    parser.add_argument("--block_sizes", default="32,64,128,256,512,1024")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in args.traces.split(",") if x.strip())
    fractions = parse_floats(args.fractions)
    block_sizes = parse_ints(args.block_sizes)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    for trace in traces:
        payload = torch.load(trace, map_location="cpu", weights_only=False)
        label = trace.stem
        for record in payload["records"]:
            layer = int(record["layer"])
            query = record["query"].to(device).float()[0, :, 0, :]
            key = record["key"].to(device).float()[0]
            value = record["value"].to(device).float()[0]
            scaling = float(record["scaling"])
            kv_heads, token_count, head_dim = key.shape
            query_heads = query.shape[0]
            groups = query_heads // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                calibration = query[
                    kv_head * groups : (kv_head + 1) * groups
                ]
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.sample_stride],
                    calibration,
                    args.query_shrinkage,
                )
                coefficients = head_key @ key_factor
                projected_calibration = calibration @ query_factor
                bands = quantized_bands(coefficients, projected_calibration)
                key_distortion, _ = distortion_table(
                    coefficients,
                    projected_calibration,
                    ZERO_BIT_LEVELS,
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                reconstruction = reconstruct(bands, allocation)

                for group in range(groups):
                    query_head = kv_head * groups + group
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (head_key @ query[query_head]) * scaling
                    proxy_scores = (
                        reconstruction.float() @ approximate_query.float()
                    ) * scaling
                    full_probability = torch.softmax(exact_scores.float(), dim=0)
                    full_output = torch.sum(
                        full_probability.unsqueeze(-1) * head_value.float(),
                        dim=0,
                    )
                    global_mean = head_value.float().mean(dim=0)

                    for fraction in fractions:
                        keep = min(token_count, max(1, math.ceil(fraction * token_count)))
                        exact_indices = torch.topk(exact_scores, k=keep).indices
                        proxy_indices = torch.topk(proxy_scores, k=keep).indices
                        proxy_sparse = normalized_sparse_output(
                            exact_scores,
                            head_value,
                            proxy_indices,
                        )
                        proxy_probability = torch.softmax(proxy_scores.float(), dim=0)
                        proxy_mass = proxy_probability[proxy_indices].sum()
                        global_corrected = (
                            proxy_mass * proxy_sparse
                            + (1.0 - proxy_mass) * global_mean
                        )
                        methods = {
                            "exact_topk": normalized_sparse_output(
                                exact_scores,
                                head_value,
                                exact_indices,
                            ),
                            "proxy_topk": proxy_sparse,
                            "proxy_global_mean": global_corrected,
                        }
                        for method, output in methods.items():
                            rows.append(
                                {
                                    "trace": label,
                                    "layer": layer,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "fraction": fraction,
                                    "block_size": 0,
                                    "method": method,
                                    "allocation": "-".join(map(str, allocation)),
                                    "exact_selected_mass": float(
                                        full_probability[
                                            exact_indices if method == "exact_topk" else proxy_indices
                                        ].sum()
                                    ),
                                    "proxy_selected_mass": float(proxy_mass),
                                    **output_metrics(output, full_output),
                                }
                            )
                        for block_size in block_sizes:
                            proxy_block, proxy_block_mass = block_corrected_output(
                                exact_scores,
                                proxy_scores,
                                head_value,
                                proxy_indices,
                                block_size,
                            )
                            oracle_block, oracle_block_mass = block_corrected_output(
                                exact_scores,
                                exact_scores,
                                head_value,
                                exact_indices,
                                block_size,
                            )
                            for method, output, estimated_mass, indices in (
                                (
                                    "proxy_block_mean",
                                    proxy_block,
                                    proxy_block_mass,
                                    proxy_indices,
                                ),
                                (
                                    "oracle_block_mean",
                                    oracle_block,
                                    oracle_block_mass,
                                    exact_indices,
                                ),
                            ):
                                rows.append(
                                    {
                                        "trace": label,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "fraction": fraction,
                                        "block_size": block_size,
                                        "method": method,
                                        "allocation": "-".join(map(str, allocation)),
                                        "exact_selected_mass": float(
                                            full_probability[indices].sum()
                                        ),
                                        "proxy_selected_mass": estimated_mass,
                                        **output_metrics(output, full_output),
                                    }
                                )
            print(json.dumps({"trace": label, "layer": layer, "rows": len(rows)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["fraction"], row["block_size"])].append(row)
    summary = []
    for (method, fraction, block_size), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "fraction": fraction,
            "block_size": block_size,
            "cases": len(items),
        }
        for metric in (
            "relative_l2",
            "cosine",
            "exact_selected_mass",
            "proxy_selected_mass",
        ):
            for statistic, value in summarize(float(x[metric]) for x in items).items():
                result[f"{metric}_{statistic}"] = value
        summary.append(result)
    report = {
        "schema": "qksieve_block_tail_reconstruction_v1",
        "traces": [str(x) for x in traces],
        "quality_boundary": (
            "Offline real-QKV mechanism audit. Proxy block mass and exact "
            "Value centroids are evaluated directly; no CUDA latency claim."
        ),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
