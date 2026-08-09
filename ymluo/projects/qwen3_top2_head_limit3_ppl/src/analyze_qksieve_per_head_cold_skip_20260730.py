#!/usr/bin/env python
"""Evaluate causal per-KV-head cold-token skipping on QKSieve traces.

The script separates a token's historical retrieval frequency from the current
query.  Calibration queries build one hot set per KV head.  Held-out queries
then scan only:

* the calibrated hot set;
* an optional recent window;
* one optional rotating shard of the remaining cold blocks; and
* the previous restricted selection, when causal carry-over is enabled.

Exact scores are used only for evaluation.  The selector itself uses QKSieve
proxy scores and past selections, so no future-query or oracle information
enters the candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _hierarchical_quantize_band,
    _qk_metric_projection_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_stride", type=int, default=32)
    parser.add_argument("--frequency_fraction", type=float, default=0.02)
    parser.add_argument("--max_fraction", type=float, default=0.06)
    parser.add_argument("--min_tokens", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1280)
    parser.add_argument(
        "--hot_fractions",
        default="0.05,0.10,0.15,0.25,0.40,0.60",
    )
    parser.add_argument("--recent_tokens", default="0,256,512")
    parser.add_argument(
        "--cold_shards",
        default="0,8,16,32",
        help="0 disables cold probing; S>0 scans one of S cold block shards.",
    )
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument(
        "--carry_previous",
        default="0,1",
        help="Whether to retain the preceding restricted selection.",
    )
    return parser.parse_args()


def parse_numbers(text: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(value) for value in text.split(",") if value.strip())


def second_moment(values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhnd,bhne->bhde", values, values) / max(
        1, values.shape[-2]
    )


def mapped_queries(records: list[dict[str, Any]]) -> torch.Tensor:
    queries = torch.cat(
        [record["query"].float() for record in records], dim=2
    )
    batch, query_heads, steps, head_dim = queries.shape
    kv_heads = int(records[0]["key"].shape[1])
    if batch != 1 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    return (
        queries.reshape(batch, kv_heads, groups, steps, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, steps * groups, head_dim)
        .contiguous()
    )


def flatten_query_heads(
    query: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    batch, query_heads, steps, head_dim = query.shape
    if batch != 1 or steps != 1 or query_heads % kv_heads:
        raise ValueError("unsupported Query trace shape")
    return query[:, :, 0, :].reshape(
        kv_heads, query_heads // kv_heads, head_dim
    )


def qksieve_reconstruct(
    projected_key: torch.Tensor,
    allocation: torch.Tensor,
) -> torch.Tensor:
    reconstructed = torch.empty_like(projected_key)
    for head in range(projected_key.shape[1]):
        for band in range(8):
            start = 16 * band
            bits = int(allocation[0, head, band].item())
            reconstructed[:, head, :, start : start + 16] = (
                _hierarchical_quantize_band(
                    projected_key[:, head, :, start : start + 16],
                    bits,
                )
            )
    return reconstructed


def target_count(args: argparse.Namespace, history_tokens: int) -> int:
    return min(
        history_tokens,
        args.max_tokens,
        max(args.min_tokens, math.ceil(args.max_fraction * history_tokens)),
    )


def score_record(
    record: dict[str, Any],
    key: torch.Tensor,
    reconstructed_key: torch.Tensor,
    query_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads = int(key.shape[1])
    grouped_query = flatten_query_heads(
        record["query"].to(device=key.device).float(),
        kv_heads,
    )
    projected_query = torch.einsum(
        "hgd,hdm->hgm", grouped_query, query_basis[0]
    )
    scaling = float(record["scaling"])
    exact = (
        torch.einsum("hgd,hnd->hgn", grouped_query, key[0])
        .reshape(-1, key.shape[-2])
        .mul_(scaling)
    )
    proxy = (
        torch.einsum(
            "hgd,hnd->hgn",
            projected_query,
            reconstructed_key[0],
        )
        .reshape_as(exact)
        .mul_(scaling)
    )
    return exact, proxy


def build_frequency_priority(
    calibration: list[dict[str, Any]],
    key: torch.Tensor,
    reconstructed_key: torch.Tensor,
    query_basis: torch.Tensor,
    frequency_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads = int(key.shape[1])
    query_heads = int(calibration[0]["query"].shape[1])
    groups = query_heads // kv_heads
    history_tokens = int(key.shape[-2])
    frequency = torch.zeros(
        kv_heads,
        history_tokens,
        dtype=torch.float32,
        device=key.device,
    )
    maximum_score = torch.full_like(frequency, -torch.inf)
    for record in calibration:
        _, proxy = score_record(
            record, key, reconstructed_key, query_basis
        )
        selected = torch.topk(
            proxy, k=frequency_count, dim=1, sorted=False
        ).indices.reshape(kv_heads, groups * frequency_count)
        frequency.scatter_add_(
            1,
            selected,
            torch.ones_like(selected, dtype=frequency.dtype),
        )
        torch.maximum(
            maximum_score,
            proxy.reshape(kv_heads, groups, history_tokens).amax(dim=1),
            out=maximum_score,
        )
    score_scale = maximum_score.std(dim=1, keepdim=True).clamp_min(1.0e-8)
    priority = frequency + 1.0e-4 * maximum_score / score_scale
    return frequency, priority


def hot_mask_from_priority(
    priority: torch.Tensor,
    hot_fraction: float,
) -> torch.Tensor:
    hot_count = min(
        priority.shape[1],
        max(1, math.ceil(hot_fraction * priority.shape[1])),
    )
    indices = torch.topk(
        priority, k=hot_count, dim=1, sorted=False
    ).indices
    mask = torch.zeros_like(priority, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def make_pool(
    hot_mask: torch.Tensor,
    recent_tokens: int,
    cold_shards: int,
    block_size: int,
    evaluation_index: int,
    previous: torch.Tensor | None,
    groups: int,
) -> torch.Tensor:
    pool = hot_mask.clone()
    history_tokens = int(pool.shape[1])
    if recent_tokens > 0:
        pool[:, max(0, history_tokens - recent_tokens) :] = True
    if cold_shards > 0:
        block_ids = (
            torch.arange(history_tokens, device=pool.device) // block_size
        )
        head_ids = torch.arange(pool.shape[0], device=pool.device)
        phases = (evaluation_index + 5 * head_ids) % cold_shards
        cold_probe = (
            block_ids[None, :] % cold_shards == phases[:, None]
        )
        pool |= cold_probe & ~hot_mask
    query_pool = pool.repeat_interleave(groups, dim=0)
    if previous is not None:
        query_pool.scatter_(1, previous, True)
    return query_pool


def selected_metrics(
    probabilities: torch.Tensor,
    selected: torch.Tensor,
    exact_top_mask: torch.Tensor,
    baseline_mask: torch.Tensor,
    output_count: int,
) -> dict[str, float]:
    return {
        "attention_mass": float(
            probabilities.gather(1, selected).sum(dim=1).mean().item()
        ),
        "oracle_topk_recall": float(
            exact_top_mask.gather(1, selected)
            .float()
            .sum(dim=1)
            .mean()
            .item()
            / output_count
        ),
        "baseline_selection_recall": float(
            baseline_mask.gather(1, selected)
            .float()
            .sum(dim=1)
            .mean()
            .item()
            / output_count
        ),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(group_fields, key))
        item["conditions"] = len(group)
        for field in metric_fields:
            values = np.asarray(
                [float(row[field]) for row in group], dtype=np.float64
            )
            item[field] = float(values.mean())
            if field in {"mass_retention", "pool_fraction"}:
                item[f"{field}_p05"] = float(np.quantile(values, 0.05))
                item[f"{field}_min"] = float(values.min())
        item["scan_speedup_upper_bound"] = 1.0 / item["pool_fraction"]
        output.append(item)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    hot_fractions = parse_numbers(args.hot_fractions, float)
    recent_options = parse_numbers(args.recent_tokens, int)
    shard_options = parse_numbers(args.cold_shards, int)
    carry_options = tuple(
        bool(value) for value in parse_numbers(args.carry_previous, int)
    )
    if args.block_size <= 0:
        raise ValueError("block_size must be positive")
    if any(value < 0 for value in shard_options):
        raise ValueError("cold_shards must be non-negative")

    step_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        payload = torch.load(
            path_text, map_location="cpu", weights_only=False
        )
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            by_layer[int(record["layer"])].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            calibration = records[: args.calibration_steps]
            evaluation = records[args.calibration_steps :]
            if not calibration or not evaluation:
                raise ValueError(f"layer {layer} lacks held-out records")

            key = calibration[0]["key"].to(device=device).float()
            _, kv_heads, history_tokens, head_dim = key.shape
            query_heads = int(calibration[0]["query"].shape[1])
            groups = query_heads // kv_heads
            sampled_key = key[..., :: args.key_stride, :].contiguous()
            calibration_query = mapped_queries(calibration).to(device)
            query_basis, key_basis = _qk_metric_projection_factors(
                second_moment(sampled_key),
                second_moment(calibration_query),
                projection_dim=head_dim,
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
            reconstructed_key = qksieve_reconstruct(
                projected_key, allocation
            )
            output_count = target_count(args, history_tokens)
            frequency_count = min(
                history_tokens,
                max(1, math.ceil(args.frequency_fraction * history_tokens)),
            )
            frequency, priority = build_frequency_priority(
                calibration,
                key,
                reconstructed_key,
                query_basis,
                frequency_count,
            )
            frequency_rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "history_tokens": history_tokens,
                    "kv_heads": kv_heads,
                    "query_heads": query_heads,
                    "frequency_fraction": args.frequency_fraction,
                    "ever_selected_fraction": float(
                        (frequency > 0).float().mean().item()
                    ),
                    "never_selected_fraction": float(
                        (frequency == 0).float().mean().item()
                    ),
                }
            )
            hot_masks = {
                fraction: hot_mask_from_priority(priority, fraction)
                for fraction in hot_fractions
            }
            configurations = [
                (hot_fraction, recent, shards, carry)
                for hot_fraction in hot_fractions
                for recent in recent_options
                for shards in shard_options
                for carry in carry_options
            ]
            previous_by_config: dict[
                tuple[float, int, int, bool], torch.Tensor | None
            ] = {configuration: None for configuration in configurations}

            for evaluation_index, record in enumerate(evaluation):
                exact_scores, proxy_scores = score_record(
                    record, key, reconstructed_key, query_basis
                )
                probabilities = torch.softmax(exact_scores, dim=-1)
                exact_top = torch.topk(
                    exact_scores,
                    k=output_count,
                    dim=1,
                    sorted=False,
                ).indices
                baseline_selected = torch.topk(
                    proxy_scores,
                    k=output_count,
                    dim=1,
                    sorted=False,
                ).indices
                exact_top_mask = torch.zeros_like(
                    exact_scores, dtype=torch.bool
                )
                exact_top_mask.scatter_(1, exact_top, True)
                baseline_mask = torch.zeros_like(exact_top_mask)
                baseline_mask.scatter_(1, baseline_selected, True)
                baseline_mass = float(
                    probabilities.gather(1, baseline_selected)
                    .sum(dim=1)
                    .mean()
                    .item()
                )

                for configuration in configurations:
                    hot_fraction, recent, shards, carry = configuration
                    previous = (
                        previous_by_config[configuration] if carry else None
                    )
                    pool = make_pool(
                        hot_masks[hot_fraction],
                        recent,
                        shards,
                        args.block_size,
                        evaluation_index,
                        previous,
                        groups,
                    )
                    available = int(pool.sum(dim=1).min().item())
                    if available < output_count:
                        raise RuntimeError(
                            f"pool {available} smaller than output "
                            f"budget {output_count}"
                        )
                    selected = torch.topk(
                        proxy_scores.masked_fill(~pool, -torch.inf),
                        k=output_count,
                        dim=1,
                        sorted=False,
                    ).indices
                    metrics = selected_metrics(
                        probabilities,
                        selected,
                        exact_top_mask,
                        baseline_mask,
                        output_count,
                    )
                    mass_retention = (
                        metrics["attention_mass"] / baseline_mass
                        if baseline_mass > 0.0
                        else 1.0
                    )
                    step_rows.append(
                        {
                            "topic": topic,
                            "layer": layer,
                            "step": int(record["step"]),
                            "hot_fraction": hot_fraction,
                            "recent_tokens": recent,
                            "cold_shards": shards,
                            "carry_previous": int(carry),
                            "output_tokens": output_count,
                            "pool_fraction": float(
                                pool.float().mean().item()
                            ),
                            "attention_mass": metrics["attention_mass"],
                            "baseline_qksieve_mass": baseline_mass,
                            "mass_retention": mass_retention,
                            "oracle_topk_recall": metrics[
                                "oracle_topk_recall"
                            ],
                            "baseline_selection_recall": metrics[
                                "baseline_selection_recall"
                            ],
                        }
                    )
                    if carry:
                        previous_by_config[configuration] = selected

            del (
                key,
                sampled_key,
                calibration_query,
                projected_sample,
                projected_calibration_query,
                projected_key,
                reconstructed_key,
                frequency,
                priority,
            )
            torch.cuda.empty_cache()

    with (args.output_dir / "step_detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    with (args.output_dir / "frequency_detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frequency_rows[0]))
        writer.writeheader()
        writer.writerows(frequency_rows)

    aggregate = aggregate_rows(
        step_rows,
        (
            "hot_fraction",
            "recent_tokens",
            "cold_shards",
            "carry_previous",
        ),
        (
            "pool_fraction",
            "attention_mass",
            "baseline_qksieve_mass",
            "mass_retention",
            "oracle_topk_recall",
            "baseline_selection_recall",
        ),
    )
    feasible = [
        row
        for row in aggregate
        if row["mass_retention"] >= 0.995
        and row["mass_retention_p05"] >= 0.98
    ]
    feasible.sort(
        key=lambda row: (
            row["pool_fraction"],
            -row["mass_retention"],
        )
    )
    summary = {
        "schema": "qksieve_per_head_cold_skip_v1",
        "traces": args.trace,
        "frequency_definition": (
            "per-KV-head counts of calibration-query QKSieve selections"
        ),
        "selection_is_causal": True,
        "exact_scores_are_evaluation_only": True,
        "calibration_steps": args.calibration_steps,
        "frequency_fraction": args.frequency_fraction,
        "output_budget": {
            "min_tokens": args.min_tokens,
            "max_fraction": args.max_fraction,
            "max_tokens": args.max_tokens,
        },
        "block_size": args.block_size,
        "frequency_summary": {
            "ever_selected_fraction": float(
                np.mean(
                    [
                        row["ever_selected_fraction"]
                        for row in frequency_rows
                    ]
                )
            ),
            "never_selected_fraction": float(
                np.mean(
                    [
                        row["never_selected_fraction"]
                        for row in frequency_rows
                    ]
                )
            ),
        },
        "aggregate": aggregate,
        "best_feasible_mass_ge_99_5_p05_ge_98": (
            feasible[0] if feasible else None
        ),
        "caveat": (
            "Scan speedup is the reciprocal pool fraction and is only an "
            "upper bound. Surviving configurations require a compact-index "
            "CUDA benchmark and held-out PPL validation."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ALL_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
