#!/usr/bin/env python
"""Evaluate strict temporal block bounds for QKSieve proxy-score scans."""

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
from analyze_qksieve_safe_block_bounds_20260729 import (
    grouped_query,
    padded_blocks,
    summarize,
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
    parser.add_argument("--block_sizes", default="16,32,64")
    parser.add_argument("--maximum_tokens", type=int, default=1280)
    return parser.parse_args()


def sampled_thresholds(
    scores: torch.Tensor,
    selected_fraction: float,
) -> tuple[torch.Tensor, int]:
    history_count = int(scores.shape[-1])
    sample_count = min(
        2048,
        max(256, math.ceil(16.0 / selected_fraction)),
    )
    rank = max(
        1,
        min(
            sample_count,
            int(round(selected_fraction * (sample_count + 1))),
        ),
    )
    flat = scores.reshape(-1, history_count)
    sample = torch.arange(
        sample_count, device=scores.device, dtype=torch.long
    )
    centered = ((2 * sample + 1) * history_count) // (2 * sample_count)
    segment = max(1, history_count // sample_count)
    rows = torch.arange(
        flat.shape[0], device=scores.device, dtype=torch.long
    )
    phase = (rows * 131 + 17) % segment
    sample_indices = (
        centered.unsqueeze(0) + phase.unsqueeze(1)
    ) % history_count
    sampled = flat.gather(1, sample_indices)
    thresholds = torch.topk(
        sampled, k=rank, dim=-1, sorted=False
    ).values.amin(dim=-1)
    return thresholds.reshape(scores.shape[:-1]), sample_count


def temporal_statistics(
    proxy_key: torch.Tensor,
    block_size: int,
) -> dict[str, torch.Tensor]:
    blocks, _ = padded_blocks(proxy_key, block_size)
    center = blocks.mean(dim=-2)
    residual = blocks - center.unsqueeze(-2)
    return {
        "blocks": blocks,
        "center": center,
        "minimum": blocks.amin(dim=-2),
        "maximum": blocks.amax(dim=-2),
        "key_norm": blocks.norm(dim=-1).amax(dim=-1),
        "radius_l2": residual.norm(dim=-1).amax(dim=-1),
        "band_radius": residual.reshape(
            *residual.shape[:-1], 8, 16
        ).norm(dim=-1).amax(dim=-2),
    }


def temporal_upper_bounds(
    previous_block_max: torch.Tensor,
    delta_query: torch.Tensor,
    statistics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    center = statistics["center"]
    minimum = statistics["minimum"]
    maximum = statistics["maximum"]
    key_norm = statistics["key_norm"]
    radius_l2 = statistics["radius_l2"]
    band_radius = statistics["band_radius"]
    blocks = statistics["blocks"]
    center_delta = torch.einsum(
        "bhgd,bhkd->bhgk", delta_query, center
    )
    return {
        "previous_plus_global_norm": previous_block_max
        + delta_query.norm(dim=-1).unsqueeze(-1)
        * key_norm.unsqueeze(2),
        "previous_plus_center_radius": previous_block_max
        + center_delta
        + delta_query.norm(dim=-1).unsqueeze(-1)
        * radius_l2.unsqueeze(2),
        "previous_plus_band_radius": previous_block_max
        + center_delta
        + torch.einsum(
            "bhgr,bhkr->bhgk",
            delta_query.reshape(
                *delta_query.shape[:-1], 8, 16
            ).norm(dim=-1),
            band_radius,
        ),
        "previous_plus_coordinate_box": previous_block_max
        + torch.einsum(
            "bhgd,bhkd->bhgk",
            delta_query.clamp_min(0),
            maximum,
        )
        + torch.einsum(
            "bhgd,bhkd->bhgk",
            delta_query.clamp_max(0),
            minimum,
        ),
        "previous_plus_exact_delta": previous_block_max
        + torch.einsum(
            "bhgd,bhksd->bhgks", delta_query, blocks
        ).amax(dim=-1),
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
        active_bands = (allocation[0] > 0).sum(dim=-1)
        projected_key = torch.einsum(
            "bhnd,bhdm->bhnm", key, key_basis
        )
        proxy_key = qksieve_reconstruct(projected_key, allocation)
        statistics_by_size = {
            size: temporal_statistics(proxy_key, size)
            for size in block_sizes
        }
        previous_query = grouped_query(
            calibration[-1], kv_heads
        ).to(device)
        previous_projected_query = torch.einsum(
            "bhgd,bhdm->bhgm", previous_query, query_basis
        )
        previous_scores = torch.einsum(
            "bhgd,bhnd->bhgn", previous_projected_query, proxy_key
        )
        previous_block_max_by_size = {}
        for size in block_sizes:
            score_blocks, _ = padded_blocks(
                previous_scores.unsqueeze(-1), size
            )
            previous_block_max_by_size[size] = (
                score_blocks.squeeze(-1).amax(dim=-1)
            )

        token_count = int(proxy_key.shape[-2])
        target_count = min(token_count, max(1, args.maximum_tokens))
        selected_fraction = target_count / token_count
        for record in evaluation:
            query = grouped_query(record, kv_heads).to(device)
            projected_query = torch.einsum(
                "bhgd,bhdm->bhgm", query, query_basis
            )
            scores = torch.einsum(
                "bhgd,bhnd->bhgn", projected_query, proxy_key
            )
            exact_threshold = torch.topk(
                scores,
                k=target_count,
                dim=-1,
                sorted=False,
            ).values.amin(dim=-1)
            sampled_threshold, sample_count = sampled_thresholds(
                scores, selected_fraction
            )
            delta_query = projected_query - previous_projected_query
            for size in block_sizes:
                score_blocks, _ = padded_blocks(
                    scores.unsqueeze(-1), size
                )
                current_block_max = score_blocks.squeeze(-1).amax(dim=-1)
                bounds = temporal_upper_bounds(
                    previous_block_max_by_size[size],
                    delta_query,
                    statistics_by_size[size],
                )
                for threshold_mode, threshold in (
                    ("exact_topk", exact_threshold),
                    ("sampled_quantile", sampled_threshold),
                ):
                    selected_blocks = (
                        current_block_max >= threshold.unsqueeze(-1)
                    )
                    for method, upper in bounds.items():
                        kept = upper >= threshold.unsqueeze(-1)
                        false_negative = selected_blocks & ~kept
                        for batch in range(query.shape[0]):
                            for head in range(kv_heads):
                                active_dimensions = int(
                                    16 * active_bands[head].item()
                                )
                                for group in range(query.shape[2]):
                                    if method == "previous_plus_global_norm":
                                        key_values = 1
                                    elif method == "previous_plus_center_radius":
                                        key_values = active_dimensions + 1
                                    elif method == "previous_plus_band_radius":
                                        key_values = (
                                            active_dimensions
                                            + int(active_bands[head].item())
                                        )
                                    elif method == "previous_plus_coordinate_box":
                                        key_values = 2 * active_dimensions
                                    else:
                                        key_values = 0
                                    directory_values = (
                                        key_values + query.shape[2]
                                    )
                                    detail.append(
                                        {
                                            "topic": args.topic,
                                            "layer": layer,
                                            "step": int(
                                                record.get("step", 0)
                                            ),
                                            "kv_head": head,
                                            "query_group": group,
                                            "block_size": size,
                                            "threshold_mode": threshold_mode,
                                            "sample_count": (
                                                sample_count
                                                if threshold_mode
                                                == "sampled_quantile"
                                                else 0
                                            ),
                                            "method": method,
                                            "pruned_block_fraction": float(
                                                (~kept)[
                                                    batch, head, group
                                                ].float().mean().item()
                                            ),
                                            "scanned_token_fraction": float(
                                                kept[
                                                    batch, head, group
                                                ].float().mean().item()
                                            ),
                                            "selected_block_false_negative_fraction": float(
                                                false_negative[
                                                    batch, head, group
                                                ].float().sum().item()
                                                / max(
                                                    1.0,
                                                    selected_blocks[
                                                        batch, head, group
                                                    ].float().sum().item(),
                                                )
                                            ),
                                            "directory_fp16_bytes_per_token_kv_head": (
                                                2.0
                                                * directory_values
                                                / size
                                            ),
                                            "directory_int8_bytes_per_token_kv_head": (
                                                directory_values / size
                                            ),
                                        }
                                    )
                previous_block_max_by_size[size] = current_block_max
            previous_projected_query = projected_query
            del scores, query
        del (
            key,
            sampled_key,
            calibration_query,
            projected_sample,
            projected_calibration_query,
            projected_key,
            proxy_key,
            previous_scores,
        )
        torch.cuda.empty_cache()

    aggregate = []
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        groups[
            (
                row["block_size"],
                row["threshold_mode"],
                row["method"],
            )
        ].append(row)
    for (size, threshold_mode, method), rows in sorted(groups.items()):
        aggregate.append(
            {
                "block_size": size,
                "threshold_mode": threshold_mode,
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
                "directory_fp16_bytes_per_token_kv_head_mean": sum(
                    row["directory_fp16_bytes_per_token_kv_head"]
                    for row in rows
                )
                / len(rows),
                "directory_int8_bytes_per_token_kv_head_mean": sum(
                    row["directory_int8_bytes_per_token_kv_head"]
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
            "Each current block is rescanned unless a strict upper bound from "
            "the previous block maximum and projected Query delta proves that "
            "the block cannot cross the current threshold."
        ),
        "aggregate": aggregate,
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
