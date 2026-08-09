#!/usr/bin/env python
"""Measure safe block pruning for QKSieve proxy scores on real Q/K traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_fier_qksieve_retrieval_fair_20260728 import (
    mapped_queries,
    qksieve_reconstruct,
    second_moment,
)
from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--evaluation_steps", type=int, default=24)
    parser.add_argument("--key_stride", type=int, default=64)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--block_sizes", default="8,16,32,64")
    parser.add_argument("--maximum_tokens", type=int, default=1280)
    return parser.parse_args()


def grouped_query(record: dict[str, Any], kv_heads: int) -> torch.Tensor:
    query = record["query"].float()[..., 0, :]
    batch, query_heads, head_dim = query.shape
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return query.reshape(batch, kv_heads, query_heads // kv_heads, head_dim)


def padded_blocks(
    values: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    token_count = int(values.shape[-2])
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count > token_count:
        padding = values[..., -1:, :].expand(
            *values.shape[:-2],
            padded_count - token_count,
            values.shape[-1],
        )
        values = torch.cat((values, padding), dim=-2)
    return values.reshape(
        *values.shape[:-2],
        block_count,
        block_size,
        values.shape[-1],
    ), token_count


def block_statistics(
    proxy_key: torch.Tensor,
    block_size: int,
) -> dict[str, torch.Tensor | int]:
    blocks, token_count = padded_blocks(proxy_key, block_size)
    center = blocks.mean(dim=-2)
    minimum = blocks.amin(dim=-2)
    maximum = blocks.amax(dim=-2)
    residual = blocks - center.unsqueeze(-2)
    radius_l2 = residual.norm(dim=-1).amax(dim=-1)
    band_residual = residual.reshape(
        *residual.shape[:-1], 8, 16
    )
    band_radius = band_residual.norm(dim=-1).amax(dim=-2)
    return {
        "blocks": blocks,
        "center": center,
        "minimum": minimum,
        "maximum": maximum,
        "radius_l2": radius_l2,
        "band_radius": band_radius,
        "token_count": token_count,
    }


def bound_values(
    projected_query: torch.Tensor,
    statistics: dict[str, torch.Tensor | int],
) -> dict[str, torch.Tensor]:
    center = statistics["center"]
    minimum = statistics["minimum"]
    maximum = statistics["maximum"]
    radius_l2 = statistics["radius_l2"]
    band_radius = statistics["band_radius"]
    blocks = statistics["blocks"]
    assert isinstance(center, torch.Tensor)
    assert isinstance(minimum, torch.Tensor)
    assert isinstance(maximum, torch.Tensor)
    assert isinstance(radius_l2, torch.Tensor)
    assert isinstance(band_radius, torch.Tensor)
    assert isinstance(blocks, torch.Tensor)

    center_score = torch.einsum(
        "bhgd,bhkd->bhgk", projected_query, center
    )
    coordinate = torch.einsum(
        "bhgd,bhkd->bhgk",
        projected_query.clamp_min(0),
        maximum,
    ) + torch.einsum(
        "bhgd,bhkd->bhgk",
        projected_query.clamp_max(0),
        minimum,
    )
    l2 = center_score + (
        projected_query.norm(dim=-1).unsqueeze(-1)
        * radius_l2.unsqueeze(2)
    )
    query_bands = projected_query.reshape(
        *projected_query.shape[:-1], 8, 16
    )
    band_l2 = center_score + torch.einsum(
        "bhgr,bhkr->bhgk",
        query_bands.norm(dim=-1),
        band_radius,
    )
    exact_block_max = torch.einsum(
        "bhgd,bhksd->bhgks", projected_query, blocks
    ).amax(dim=-1)
    return {
        "coordinate_box": coordinate,
        "global_l2": l2,
        "band_l2": band_l2,
        "oracle_block_max": exact_block_max,
    }


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "median": float(tensor.median().item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "maximum": float(tensor.max().item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    block_sizes = sorted(
        {int(item) for item in args.block_sizes.split(",") if item}
    )
    trace = torch.load(
        args.trace, map_location="cpu", weights_only=False
    )
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    layer_keys: dict[int, torch.Tensor] = {}
    for record in trace["records"]:
        layer = int(record["layer"])
        by_layer[layer].append(record)
        if isinstance(record.get("key"), torch.Tensor):
            layer_keys[layer] = record["key"]

    detail: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for layer, records in sorted(by_layer.items()):
        records.sort(key=lambda item: int(item.get("step", 0)))
        calibration = records[: args.calibration_steps]
        evaluation = records[
            args.calibration_steps :
            args.calibration_steps + args.evaluation_steps
        ]
        if len(calibration) < args.calibration_steps or not evaluation:
            raise ValueError(f"layer {layer} has insufficient Query records")
        key = layer_keys[layer].to(device=device).float()
        kv_heads = int(key.shape[1])
        sampled_key = key[..., :: args.key_stride, :].contiguous()
        calibration_query = mapped_queries(calibration).to(device)
        key_covariance = second_moment(sampled_key)
        query_covariance = second_moment(calibration_query)
        query_basis, key_basis = _qk_metric_projection_factors(
            key_covariance,
            query_covariance,
            projection_dim=key.shape[-1],
            query_shrinkage=args.query_shrinkage,
        )
        projected_sample = torch.einsum(
            "bhnd,bhdm->bhnm", sampled_key, key_basis
        )
        projected_calibration_query = torch.einsum(
            "bhnd,bhdm->bhnm", calibration_query, query_basis
        )
        allocation = _hierarchical_qmse_rate_allocation(
            projected_sample,
            projected_calibration_query,
            bit_budget_per_coordinate=15,
            allow_zero_bits=True,
            include_scale_metadata=True,
        )
        projected_key = torch.einsum(
            "bhnd,bhdm->bhnm", key, key_basis
        )
        proxy_key = qksieve_reconstruct(projected_key, allocation)
        active_bands = (allocation[0] > 0).sum(dim=-1)
        for head in range(kv_heads):
            allocation_rows.append(
                {
                    "topic": args.topic,
                    "layer": layer,
                    "kv_head": head,
                    "allocation": [
                        int(value)
                        for value in allocation[0, head].tolist()
                    ],
                    "active_bands": int(active_bands[head].item()),
                    "active_dimensions": int(
                        16 * active_bands[head].item()
                    ),
                }
            )

        statistics_by_size = {
            block_size: block_statistics(proxy_key, block_size)
            for block_size in block_sizes
        }
        token_count = int(proxy_key.shape[-2])
        target_count = min(
            token_count,
            max(1, int(args.maximum_tokens)),
        )
        for record in evaluation:
            query = grouped_query(record, kv_heads).to(device)
            projected_query = torch.einsum(
                "bhgd,bhdm->bhgm", query, query_basis
            )
            proxy_scores = torch.einsum(
                "bhgd,bhnd->bhgn", projected_query, proxy_key
            )
            threshold = torch.topk(
                proxy_scores,
                k=target_count,
                dim=-1,
                sorted=False,
            ).values.amin(dim=-1)
            for block_size, statistics in statistics_by_size.items():
                bounds = bound_values(projected_query, statistics)
                block_count = int(bounds["oracle_block_max"].shape[-1])
                selected_blocks = (
                    bounds["oracle_block_max"]
                    >= threshold.unsqueeze(-1)
                )
                for method, upper in bounds.items():
                    kept_blocks = upper >= threshold.unsqueeze(-1)
                    false_negative_blocks = selected_blocks & ~kept_blocks
                    pruned_fraction = (~kept_blocks).float()
                    false_negative_fraction = (
                        false_negative_blocks.float().sum(dim=-1)
                        / selected_blocks.float().sum(dim=-1).clamp_min(1)
                    )
                    for batch in range(query.shape[0]):
                        for head in range(kv_heads):
                            for group in range(query.shape[2]):
                                active_dimensions = int(
                                    16 * active_bands[head].item()
                                )
                                if method == "coordinate_box":
                                    metadata_values = 2 * active_dimensions
                                elif method == "global_l2":
                                    metadata_values = active_dimensions + 1
                                elif method == "band_l2":
                                    metadata_values = (
                                        active_dimensions
                                        + int(active_bands[head].item())
                                    )
                                else:
                                    metadata_values = 0
                                detail.append(
                                    {
                                        "topic": args.topic,
                                        "layer": layer,
                                        "step": int(record.get("step", 0)),
                                        "kv_head": head,
                                        "query_group": group,
                                        "block_size": block_size,
                                        "method": method,
                                        "token_count": token_count,
                                        "target_count": target_count,
                                        "block_count": block_count,
                                        "pruned_block_fraction": float(
                                            pruned_fraction[
                                                batch, head, group
                                            ].mean().item()
                                        ),
                                        "scanned_token_fraction": float(
                                            kept_blocks[
                                                batch, head, group
                                            ].float().mean().item()
                                        ),
                                        "selected_block_false_negative_fraction": float(
                                            false_negative_fraction[
                                                batch, head, group
                                            ].item()
                                        ),
                                        "metadata_fp16_bytes_per_token_head": (
                                            2.0
                                            * metadata_values
                                            / block_size
                                        ),
                                        "metadata_int8_bytes_per_token_head": (
                                            metadata_values / block_size
                                        ),
                                    }
                                )
            del proxy_scores, projected_query, query
        del (
            key,
            sampled_key,
            calibration_query,
            projected_sample,
            projected_calibration_query,
            projected_key,
            proxy_key,
        )
        torch.cuda.empty_cache()

    aggregate = []
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        groups[(row["block_size"], row["method"])].append(row)
    for (block_size, method), rows in sorted(groups.items()):
        aggregate.append(
            {
                "block_size": block_size,
                "method": method,
                "observations": len(rows),
                "pruned_block_fraction": summarize(
                    [row["pruned_block_fraction"] for row in rows]
                ),
                "scanned_token_fraction": summarize(
                    [row["scanned_token_fraction"] for row in rows]
                ),
                "selected_block_false_negative_fraction": summarize(
                    [
                        row["selected_block_false_negative_fraction"]
                        for row in rows
                    ]
                ),
                "metadata_fp16_bytes_per_token_head_mean": sum(
                    row["metadata_fp16_bytes_per_token_head"]
                    for row in rows
                )
                / len(rows),
                "metadata_int8_bytes_per_token_head_mean": sum(
                    row["metadata_int8_bytes_per_token_head"]
                    for row in rows
                )
                / len(rows),
            }
        )
    output = {
        "config": vars(args) | {
            "trace": str(args.trace),
            "output": str(args.output),
        },
        "scope": (
            "Safe pruning is measured against the exact QKSieve proxy-score "
            "top-k boundary. A block is skipped only when its computed upper "
            "bound is below that boundary."
        ),
        "aggregate": aggregate,
        "allocations": allocation_rows,
        "detail": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
