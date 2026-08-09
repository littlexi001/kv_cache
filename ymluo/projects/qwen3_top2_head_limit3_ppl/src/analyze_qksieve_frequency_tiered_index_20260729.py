#!/usr/bin/env python
"""Evaluate a causal frequency-tiered QKSieve index.

Prompt-tail calibration Queries identify frequently retrieved tokens per KV
head.  Those tokens retain the standard 240-bit QKSieve representation while
all other tokens use a lower-rate QK-balanced representation.  Every held-out
Query still scans every token, so the experiment tests precision allocation,
not temporal candidate reuse.
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

from analyze_qksieve_frequency_hotset_20260729 import (
    flatten_query_heads,
    mapped_queries,
    qksieve_reconstruct,
    second_moment,
)
from run_head_top2_targeted_ppl_20260714 import (
    _hierarchical_qmse_rate_allocation,
    _qk_metric_projection_factors,
)

FIXED_HARDWARE_PROFILES = {
    "fixed_80": (8, 0, 0, 0, 0, 0, 0, 0),
    "fixed_81": (8, 1, 0, 0, 0, 0, 0, 0),
    "fixed_82": (8, 2, 0, 0, 0, 0, 0, 0),
    "fixed_44": (4, 4, 0, 0, 0, 0, 0, 0),
    "fixed_421": (4, 2, 1, 0, 0, 0, 0, 0),
    "fixed_441": (4, 4, 1, 0, 0, 0, 0, 0),
    "fixed_811": (8, 1, 1, 0, 0, 0, 0, 0),
    "fixed_442": (4, 4, 2, 0, 0, 0, 0, 0),
    "fixed_84": (8, 4, 0, 0, 0, 0, 0, 0),
    "fixed_4421": (4, 4, 2, 1, 0, 0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_stride", type=int, default=32)
    parser.add_argument("--retrieval_fraction", type=float, default=0.02)
    parser.add_argument(
        "--cold_budgets",
        default="7,9,11",
        help="Physical bit units per 16 coordinates; 15 is QKSieve.",
    )
    parser.add_argument(
        "--hot_fractions",
        default="0.01,0.02,0.04,0.08",
    )
    parser.add_argument(
        "--block_sizes",
        default="",
        help=(
            "Optional comma-separated block sizes for frequency-block "
            "tiering. Empty disables block variants."
        ),
    )
    parser.add_argument(
        "--prior_alphas",
        default="",
        help=(
            "Optional dimensionless frequency-prior strengths for fixed "
            "hardware profiles. Empty disables frequency-prior variants."
        ),
    )
    parser.add_argument(
        "--block_cold_profiles",
        default="",
        help=(
            "Optional fixed hardware profiles used as cold-block "
            "representations in frequency block variants."
        ),
    )
    return parser.parse_args()


def physical_bytes(allocation: torch.Tensor) -> float:
    units = allocation.float() + (allocation > 0).float()
    return float((16.0 * units.sum(dim=-1).mean() / 8.0).item())


def hardware_codebook_allocation(
    projected_sample: torch.Tensor,
    projected_queries: torch.Tensor,
    high_allocation: torch.Tensor,
    target_bytes_per_head: int,
) -> tuple[torch.Tensor, list[str], float]:
    """Allocate whole-band execution profiles under one layer-wide byte cap."""
    _, head_count, _, _ = projected_sample.shape
    reference_scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        projected_queries.float(),
        projected_sample.float(),
    )
    reference_scale = reference_scores.square().mean(
        dim=(-1, -2)
    ).clamp_min(1.0e-12)

    candidates: dict[str, torch.Tensor] = {}
    for name, profile in FIXED_HARDWARE_PROFILES.items():
        candidates[name] = (
            torch.tensor(
                profile,
                dtype=torch.int16,
                device=projected_sample.device,
            )
            .reshape(1, 1, 8)
            .expand(1, head_count, 8)
            .contiguous()
        )
    candidates["auto15"] = high_allocation

    errors: dict[str, torch.Tensor] = {}
    costs: dict[str, torch.Tensor] = {}
    for name, allocation in candidates.items():
        reconstructed = qksieve_reconstruct(
            projected_sample,
            allocation,
        )
        approximate_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            projected_queries.float(),
            reconstructed.float(),
        )
        errors[name] = (
            (approximate_scores - reference_scores)
            .square()
            .mean(dim=(-1, -2))
            / reference_scale
        )[0]
        costs[name] = (
            2
            * (
                allocation[0].to(torch.int64).sum(dim=-1)
                + (allocation[0] > 0).sum(dim=-1)
            )
        )

    maximum_cost = target_bytes_per_head * head_count
    states: dict[int, tuple[float, list[str]]] = {0: (0.0, [])}
    for head in range(head_count):
        next_states: dict[int, tuple[float, list[str]]] = {}
        for accumulated_cost, (accumulated_error, selected) in states.items():
            for name in candidates:
                candidate_cost = accumulated_cost + int(
                    costs[name][head].item()
                )
                if candidate_cost > maximum_cost:
                    continue
                candidate_error = accumulated_error + float(
                    errors[name][head].item()
                )
                previous = next_states.get(candidate_cost)
                if previous is None or candidate_error < previous[0]:
                    next_states[candidate_cost] = (
                        candidate_error,
                        [*selected, name],
                    )
        if not next_states:
            raise RuntimeError("hardware codebook has no feasible allocation")
        states = next_states

    used_cost, (_, selected_names) = min(
        states.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    selected_allocation = torch.empty_like(high_allocation)
    for head, name in enumerate(selected_names):
        selected_allocation[:, head] = candidates[name][:, head]
    return (
        selected_allocation,
        selected_names,
        used_cost / head_count,
    )


def calibration_frequency(
    calibration: list[dict[str, Any]],
    key: torch.Tensor,
    reconstructed_key: torch.Tensor,
    query_basis: torch.Tensor,
    retrieval_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads = int(key.shape[1])
    token_count = int(key.shape[-2])
    query_heads = int(calibration[0]["query"].shape[1])
    groups = query_heads // kv_heads
    frequency = torch.zeros(
        kv_heads,
        token_count,
        dtype=torch.float32,
        device=key.device,
    )
    maximum_score = torch.full_like(frequency, -torch.inf)
    for record in calibration:
        grouped_query = flatten_query_heads(
            record["query"].to(device=key.device).float(),
            kv_heads,
        )
        projected_query = torch.einsum(
            "hgd,hdm->hgm", grouped_query, query_basis[0]
        )
        scores = (
            torch.einsum(
                "hgd,hnd->hgn",
                projected_query,
                reconstructed_key[0],
            )
            * float(record["scaling"])
        )
        selected = torch.topk(
            scores,
            k=retrieval_count,
            dim=-1,
            sorted=False,
        ).indices
        frequency.scatter_add_(
            1,
            selected.reshape(kv_heads, groups * retrieval_count),
            torch.ones(
                kv_heads,
                groups * retrieval_count,
                dtype=frequency.dtype,
                device=frequency.device,
            ),
        )
        maximum_score = torch.maximum(maximum_score, scores.amax(dim=1))
    finite_score = torch.where(
        torch.isfinite(maximum_score),
        maximum_score,
        torch.zeros_like(maximum_score),
    )
    score_scale = finite_score.std(dim=1, keepdim=True).clamp_min(1.0e-8)
    priority = frequency + 1.0e-4 * finite_score / score_scale
    return frequency, priority


def crossing_priority(
    calibration: list[dict[str, Any]],
    high_key: torch.Tensor,
    low_key: torch.Tensor,
    query_basis: torch.Tensor,
    retrieval_count: int,
) -> torch.Tensor:
    """Estimate per-token low-rate top-k crossing risk causally."""
    kv_heads = int(high_key.shape[1])
    token_count = int(high_key.shape[-2])
    priority = torch.zeros(
        kv_heads,
        token_count,
        dtype=torch.float32,
        device=high_key.device,
    )
    for record in calibration:
        grouped_query = flatten_query_heads(
            record["query"].to(device=high_key.device).float(),
            kv_heads,
        )
        projected_query = torch.einsum(
            "hgd,hdm->hgm", grouped_query, query_basis[0]
        )
        high_scores = (
            torch.einsum(
                "hgd,hnd->hgn", projected_query, high_key[0]
            )
            * float(record["scaling"])
        )
        low_scores = (
            torch.einsum(
                "hgd,hnd->hgn", projected_query, low_key[0]
            )
            * float(record["scaling"])
        )
        high_values, high_top = torch.topk(
            high_scores,
            k=retrieval_count,
            dim=-1,
            sorted=False,
        )
        low_top = torch.topk(
            low_scores,
            k=retrieval_count,
            dim=-1,
            sorted=False,
        ).indices
        high_mask = torch.zeros_like(high_scores, dtype=torch.bool)
        low_mask = torch.zeros_like(low_scores, dtype=torch.bool)
        high_mask.scatter_(-1, high_top, True)
        low_mask.scatter_(-1, low_top, True)
        disagreement = (high_mask ^ low_mask).float()
        threshold = high_values.amin(dim=-1, keepdim=True)
        margin = (high_scores - threshold).abs()
        delta = (high_scores - low_scores).abs()
        scale = high_scores.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        soft_crossing = torch.relu(delta - margin) / scale
        risk = disagreement + 0.25 * soft_crossing.clamp_max(4.0)
        priority += risk.sum(dim=1)
    return priority


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=1)
    ranks = torch.empty_like(order)
    source = torch.arange(
        values.shape[1], device=values.device, dtype=order.dtype
    )[None, :].expand_as(order)
    ranks.scatter_(1, order, source)
    return ranks.float() / max(1, values.shape[1] - 1)


def block_hot_mask(
    priority: torch.Tensor,
    hot_fraction: float,
    block_size: int,
) -> tuple[torch.Tensor, float]:
    token_count = int(priority.shape[-1])
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count > token_count:
        padding = torch.zeros(
            *priority.shape[:-1],
            padded_count - token_count,
            dtype=priority.dtype,
            device=priority.device,
        )
        working = torch.cat((priority, padding), dim=-1)
    else:
        working = priority
    block_priority = working.reshape(
        *working.shape[:-1],
        block_count,
        block_size,
    ).sum(dim=-1)
    selected_blocks = min(
        block_count,
        max(1, math.ceil(block_count * hot_fraction)),
    )
    hot_blocks = torch.topk(
        block_priority,
        k=selected_blocks,
        dim=-1,
        sorted=False,
    ).indices
    block_mask = torch.zeros_like(block_priority, dtype=torch.bool)
    block_mask.scatter_(-1, hot_blocks, True)
    token_mask = (
        block_mask[..., :, None]
        .expand(*block_mask.shape, block_size)
        .reshape(*priority.shape[:-1], padded_count)[..., :token_count]
        .contiguous()
    )
    actual_fraction = float(token_mask.float().mean().item())
    return token_mask, actual_fraction


def exact_reference(
    exact_scaled: torch.Tensor,
    retrieval_count: int,
    value: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    exact_top = torch.topk(
        exact_scaled, k=retrieval_count, dim=1, sorted=False
    ).indices
    exact_mask = torch.zeros_like(exact_scaled, dtype=torch.bool)
    exact_mask.scatter_(1, exact_top, True)
    exact_centered = exact_scaled - exact_scaled.mean(dim=1, keepdim=True)
    reference = {
        "mask": exact_mask,
        "probabilities": torch.softmax(exact_scaled, dim=1),
        "top1": exact_scaled.argmax(dim=1),
        "centered": exact_centered,
        "centered_rms": exact_centered.square().mean(dim=1).sqrt(),
        "scores": exact_scaled,
    }
    if value is not None:
        kv_heads = int(value.shape[1])
        query_heads = int(exact_scaled.shape[0])
        groups = query_heads // kv_heads
        reference["full_output"] = torch.einsum(
            "hgn,hnd->hgd",
            reference["probabilities"].reshape(
                kv_heads,
                groups,
                exact_scaled.shape[1],
            ),
            value[0].float(),
        ).reshape(query_heads, value.shape[-1])
        reference["value"] = value[0].float()
    return reference


def retrieval_metrics(
    proxy_scaled: torch.Tensor,
    reference: dict[str, torch.Tensor],
    retrieval_count: int,
    proxy_top: torch.Tensor | None = None,
) -> dict[str, float]:
    if proxy_top is None:
        proxy_top = torch.topk(
            proxy_scaled, k=retrieval_count, dim=1, sorted=False
        ).indices
    proxy_centered = proxy_scaled - proxy_scaled.mean(dim=1, keepdim=True)
    correlation = (
        (reference["centered"] * proxy_centered).mean(dim=1)
        / (
            reference["centered_rms"]
            * proxy_centered.square().mean(dim=1).sqrt()
        ).clamp_min(1.0e-12)
    )
    output = {
        "top2_recall": float(
            reference["mask"].gather(1, proxy_top).float().mean().item()
        ),
        "attention_mass": float(
            reference["probabilities"]
            .gather(1, proxy_top)
            .sum(dim=1)
            .mean()
            .item()
        ),
        "top1_recall": float(
            (
                reference["top1"] == proxy_scaled.argmax(dim=1)
            )
            .float()
            .mean()
            .item()
        ),
        "score_correlation": float(correlation.mean().item()),
    }
    if "value" in reference:
        query_heads = int(proxy_top.shape[0])
        kv_heads = int(reference["value"].shape[0])
        groups = query_heads // kv_heads
        kv_head_ids = (
            torch.arange(query_heads, device=proxy_top.device) // groups
        )
        selected_value = reference["value"][
            kv_head_ids[:, None],
            proxy_top,
        ]
        selected_score = reference["scores"].gather(1, proxy_top)
        selected_weight = torch.softmax(selected_score, dim=1)
        sparse_output = torch.einsum(
            "hk,hkd->hd", selected_weight, selected_value
        )
        full_output = reference["full_output"]
        residual_norm = (sparse_output - full_output).norm(dim=1)
        full_norm = full_output.norm(dim=1).clamp_min(1.0e-8)
        cosine = torch.nn.functional.cosine_similarity(
            sparse_output, full_output, dim=1
        )
        output["output_relative_l2"] = float(
            (residual_norm / full_norm).mean().item()
        )
        output["output_cosine"] = float(cosine.mean().item())
    else:
        output["output_relative_l2"] = float("nan")
        output["output_cosine"] = float("nan")
    return output


def block_local_global_topk(
    scores: torch.Tensor,
    retrieval_count: int,
    block_size: int,
    hot_mask: torch.Tensor,
    hot_extra: int = 1,
) -> torch.Tensor:
    """Deterministic warp-local pool followed by a small global top-k."""
    query_heads, token_count = scores.shape
    kv_heads = int(hot_mask.shape[0])
    groups = query_heads // kv_heads
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count > token_count:
        scores = torch.cat(
            (
                scores,
                torch.full(
                    (query_heads, padded_count - token_count),
                    -torch.inf,
                    dtype=scores.dtype,
                    device=scores.device,
                ),
            ),
            dim=1,
        )
        hot_mask = torch.cat(
            (
                hot_mask,
                torch.zeros(
                    (kv_heads, padded_count - token_count),
                    dtype=hot_mask.dtype,
                    device=hot_mask.device,
                ),
            ),
            dim=1,
        )
    base_local = max(1, math.ceil(retrieval_count / block_count))
    maximum_local = min(block_size, base_local + hot_extra)
    block_scores = scores.reshape(query_heads, block_count, block_size)
    local_values, local_indices = torch.topk(
        block_scores,
        k=maximum_local,
        dim=-1,
        sorted=False,
    )
    hot_blocks = hot_mask.reshape(
        kv_heads, block_count, block_size
    )[..., 0]
    query_hot_blocks = hot_blocks.repeat_interleave(groups, dim=0)
    local_budget = (
        base_local + hot_extra * query_hot_blocks.to(torch.int64)
    ).clamp_max(maximum_local)
    local_rank = torch.arange(
        maximum_local,
        device=scores.device,
    ).reshape(1, 1, maximum_local)
    valid = local_rank < local_budget[..., None]
    local_values = torch.where(valid, local_values, -torch.inf)
    block_offsets = (
        torch.arange(block_count, device=scores.device)
        .reshape(1, block_count, 1)
        * block_size
    )
    local_indices = local_indices + block_offsets
    pool_values = local_values.reshape(query_heads, -1)
    pool_indices = local_indices.reshape(query_heads, -1)
    pool_top = torch.topk(
        pool_values,
        k=retrieval_count,
        dim=1,
        sorted=False,
    ).indices
    return pool_indices.gather(1, pool_top)


def aggregate(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                row["cold_budget"],
                row["hot_fraction"],
            )
        ].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = dict(
            zip(
                ("method", "cold_budget", "hot_fraction"),
                key,
            )
        )
        item["conditions"] = len(group)
        item["index_bytes"] = float(
            np.mean([float(row["index_bytes"]) for row in group])
        )
        for field in (
            "top2_recall",
            "attention_mass",
            "top1_recall",
            "score_correlation",
            "calibration_selection_share_in_hotset",
            "output_relative_l2",
            "output_cosine",
        ):
            item[field] = float(np.nanmean(
                [float(row[field]) for row in group]
            ))
        output.append(item)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cold_budgets = tuple(
        int(item) for item in args.cold_budgets.split(",") if item.strip()
    )
    hot_fractions = tuple(
        float(item) for item in args.hot_fractions.split(",") if item.strip()
    )
    block_sizes = tuple(
        int(item) for item in args.block_sizes.split(",") if item.strip()
    )
    prior_alphas = tuple(
        float(item) for item in args.prior_alphas.split(",") if item.strip()
    )
    block_cold_profiles = tuple(
        item
        for item in args.block_cold_profiles.split(",")
        if item.strip()
    )
    unknown_profiles = set(block_cold_profiles) - set(
        FIXED_HARDWARE_PROFILES
    )
    if unknown_profiles:
        raise ValueError(
            f"unknown block cold profiles: {sorted(unknown_profiles)}"
        )
    detail_rows: list[dict[str, Any]] = []
    codebook_selection_rows: list[dict[str, Any]] = []

    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        payload = torch.load(path_text, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            by_layer[int(record["layer"])].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            calibration = records[: args.calibration_steps]
            evaluation = records[args.calibration_steps :]
            key = calibration[0]["key"].to(device=device).float()
            _, kv_heads, token_count, head_dim = key.shape
            retrieval_count = max(
                1, math.ceil(token_count * args.retrieval_fraction)
            )
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
            projected_key = torch.einsum(
                "bhnd,bhdm->bhnm", key, key_basis
            )
            high_allocation = _hierarchical_qmse_rate_allocation(
                projected_sample,
                projected_calibration_query,
                bit_budget_per_coordinate=15,
                allow_zero_bits=True,
                include_scale_metadata=True,
            )
            high_key = qksieve_reconstruct(
                projected_key, high_allocation
            )
            high_bytes = physical_bytes(high_allocation)
            frequency, priority = calibration_frequency(
                calibration,
                key,
                high_key,
                query_basis,
                retrieval_count,
            )
            log_frequency = torch.log1p(frequency.float())
            frequency_prior = (
                log_frequency
                - log_frequency.mean(dim=1, keepdim=True)
            ) / log_frequency.std(dim=1, keepdim=True).clamp_min(1e-6)
            total_calibration_events = (
                len(calibration)
                * (calibration[0]["query"].shape[1] // kv_heads)
                * retrieval_count
            )

            methods: list[dict[str, Any]] = [
                {
                    "method": "qksieve_240",
                    "cold_budget": 15,
                    "hot_fraction": 1.0,
                    "index_bytes": high_bytes,
                    "key": high_key,
                    "calibration_selection_share_in_hotset": 1.0,
                }
            ]
            for target_bytes in (22, 24):
                (
                    codebook_allocation,
                    selected_profiles,
                    actual_bytes,
                ) = hardware_codebook_allocation(
                    projected_sample,
                    projected_calibration_query,
                    high_allocation,
                    target_bytes,
                )
                methods.append(
                    {
                        "method": f"hardware_codebook_b{target_bytes}",
                        "cold_budget": target_bytes,
                        "hot_fraction": 0.0,
                        "index_bytes": actual_bytes,
                        "key": qksieve_reconstruct(
                            projected_key,
                            codebook_allocation,
                        ),
                        "calibration_selection_share_in_hotset": 0.0,
                    }
                )
                codebook_selection_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "target_bytes": target_bytes,
                        "actual_bytes": actual_bytes,
                        "profiles": selected_profiles,
                        "allocations": codebook_allocation[0].tolist(),
                    }
                )
            fixed_candidates: dict[str, dict[str, Any]] = {}
            for profile_name, profile in FIXED_HARDWARE_PROFILES.items():
                fixed_allocation = (
                    torch.tensor(
                        profile,
                        dtype=torch.int16,
                        device=device,
                    )
                    .reshape(1, 1, 8)
                    .expand(1, kv_heads, 8)
                    .contiguous()
                )
                fixed_key = qksieve_reconstruct(
                    projected_key,
                    fixed_allocation,
                )
                fixed_candidates[profile_name] = {
                    "key": fixed_key,
                    "bytes": physical_bytes(fixed_allocation),
                    "budget": sum(profile),
                }
                methods.append(
                    {
                        "method": profile_name,
                        "cold_budget": sum(profile),
                        "hot_fraction": 0.0,
                        "index_bytes": physical_bytes(fixed_allocation),
                        "key": fixed_key,
                        "calibration_selection_share_in_hotset": 0.0,
                    }
                )
                if profile_name in {"fixed_84", "fixed_811", "fixed_441"}:
                    for alpha in prior_alphas:
                        methods.append(
                            {
                                "method": (
                                    f"{profile_name}_freqprior_a"
                                    f"{alpha:g}"
                                ),
                                "cold_budget": sum(profile),
                                "hot_fraction": 0.0,
                                "index_bytes": (
                                    physical_bytes(fixed_allocation) + 0.5
                                ),
                                "key": fixed_key,
                                "score_prior": frequency_prior,
                                "score_prior_alpha": alpha,
                                "calibration_selection_share_in_hotset": 0.0,
                            }
                        )
            for cold_budget in cold_budgets:
                low_allocation = _hierarchical_qmse_rate_allocation(
                    projected_sample,
                    projected_calibration_query,
                    bit_budget_per_coordinate=cold_budget,
                    allow_zero_bits=True,
                    include_scale_metadata=True,
                )
                low_key = qksieve_reconstruct(
                    projected_key, low_allocation
                )
                low_bytes = physical_bytes(low_allocation)
                crossing = crossing_priority(
                    calibration,
                    high_key,
                    low_key,
                    query_basis,
                    retrieval_count,
                )
                hybrid_priority = (
                    0.5 * rank_normalize(priority)
                    + 0.5 * rank_normalize(crossing)
                )
                methods.append(
                    {
                        "method": "uniform_low_rate",
                        "cold_budget": cold_budget,
                        "hot_fraction": 0.0,
                        "index_bytes": low_bytes,
                        "key": low_key,
                        "calibration_selection_share_in_hotset": 0.0,
                    }
                )
                for selector_name, selector_priority in (
                    ("frequency_tiered", priority),
                    ("crossing_tiered", crossing),
                    ("hybrid_tiered", hybrid_priority),
                ):
                    for hot_fraction in hot_fractions:
                        hot_count = max(
                            1, math.ceil(token_count * hot_fraction)
                        )
                        hot = torch.topk(
                            selector_priority,
                            k=hot_count,
                            dim=1,
                            sorted=False,
                        ).indices
                        hot_mask = torch.zeros(
                            kv_heads,
                            token_count,
                            dtype=torch.bool,
                            device=device,
                        )
                        hot_mask.scatter_(1, hot, True)
                        mixed_key = torch.where(
                            hot_mask[None, :, :, None],
                            high_key,
                            low_key,
                        )
                        selection_share = float(
                            frequency.gather(1, hot)
                            .sum(dim=1)
                            .mean()
                            .item()
                            / max(1, total_calibration_events)
                        )
                        directory_bytes = 4.0 * hot_fraction
                        methods.append(
                            {
                                "method": selector_name,
                                "cold_budget": cold_budget,
                                "hot_fraction": hot_fraction,
                                "index_bytes": (
                                    hot_fraction * high_bytes
                                    + (1.0 - hot_fraction) * low_bytes
                                    + directory_bytes
                                ),
                                "key": mixed_key,
                                "calibration_selection_share_in_hotset": (
                                    selection_share
                                ),
                            }
                        )
                for block_size in block_sizes:
                    for hot_fraction in hot_fractions:
                        hot_mask, actual_hot_fraction = block_hot_mask(
                            priority,
                            hot_fraction,
                            block_size,
                        )
                        mixed_key = torch.where(
                            hot_mask[None, :, :, None],
                            high_key,
                            low_key,
                        )
                        selection_share = float(
                            (frequency * hot_mask.float())
                            .sum(dim=1)
                            .mean()
                            .item()
                            / max(1, total_calibration_events)
                        )
                        block_metadata_bytes = 1.0 / (8.0 * block_size)
                        methods.append(
                            {
                                "method": (
                                    f"frequency_block{block_size}_tiered"
                                ),
                                "cold_budget": cold_budget,
                                "hot_fraction": actual_hot_fraction,
                                "index_bytes": (
                                    actual_hot_fraction * high_bytes
                                    + (1.0 - actual_hot_fraction)
                                    * low_bytes
                                    + block_metadata_bytes
                                ),
                                "key": mixed_key,
                                "calibration_selection_share_in_hotset": (
                                    selection_share
                                ),
                            }
                        )
                        methods.append(
                            {
                                "method": (
                                    f"frequency_block{block_size}_"
                                    "tiered_localpool"
                                ),
                                "cold_budget": cold_budget,
                                "hot_fraction": actual_hot_fraction,
                                "index_bytes": (
                                    actual_hot_fraction * high_bytes
                                    + (1.0 - actual_hot_fraction)
                                    * low_bytes
                                    + block_metadata_bytes
                                ),
                                "key": mixed_key,
                                "calibration_selection_share_in_hotset": (
                                    selection_share
                                ),
                                "local_pool_block_size": block_size,
                                "local_pool_hot_mask": hot_mask,
                            }
                        )

            for profile_name in block_cold_profiles:
                fixed_candidate = fixed_candidates[profile_name]
                for block_size in block_sizes:
                    for hot_fraction in hot_fractions:
                        hot_mask, actual_hot_fraction = block_hot_mask(
                            priority,
                            hot_fraction,
                            block_size,
                        )
                        mixed_key = torch.where(
                            hot_mask[None, :, :, None],
                            high_key,
                            fixed_candidate["key"],
                        )
                        selection_share = float(
                            (frequency * hot_mask.float())
                            .sum(dim=1)
                            .mean()
                            .item()
                            / max(1, total_calibration_events)
                        )
                        methods.append(
                            {
                                "method": (
                                    f"frequency_block{block_size}_cold_"
                                    f"{profile_name}"
                                ),
                                "cold_budget": fixed_candidate["budget"],
                                "hot_fraction": actual_hot_fraction,
                                "index_bytes": (
                                    actual_hot_fraction * high_bytes
                                    + (1.0 - actual_hot_fraction)
                                    * fixed_candidate["bytes"]
                                    + 2.0 / block_size
                                ),
                                "key": mixed_key,
                                "calibration_selection_share_in_hotset": (
                                    selection_share
                                ),
                            }
                        )

            query_heads = int(calibration[0]["query"].shape[1])
            layer_value = (
                calibration[0]["value"].to(device=device)
                if isinstance(calibration[0].get("value"), torch.Tensor)
                else None
            )
            for record in evaluation:
                grouped_query = flatten_query_heads(
                    record["query"].to(device=device).float(),
                    kv_heads,
                )
                projected_query = torch.einsum(
                    "hgd,hdm->hgm", grouped_query, query_basis[0]
                )
                exact_scaled = (
                    torch.einsum(
                        "hgd,hnd->hgn", grouped_query, key[0]
                    ).reshape(query_heads, token_count)
                    * float(record["scaling"])
                )
                reference = exact_reference(
                    exact_scaled,
                    retrieval_count,
                    layer_value,
                )
                for method in methods:
                    proxy_scaled = (
                        torch.einsum(
                            "hgd,hnd->hgn",
                            projected_query,
                            method["key"][0],
                        ).reshape(query_heads, token_count)
                        * float(record["scaling"])
                    )
                    if "score_prior" in method:
                        query_prior = method[
                            "score_prior"
                        ].repeat_interleave(
                            query_heads // kv_heads,
                            dim=0,
                        )
                        score_scale = proxy_scaled.std(
                            dim=1,
                            keepdim=True,
                            unbiased=False,
                        ).clamp_min(1e-6)
                        proxy_scaled = (
                            proxy_scaled
                            + float(method["score_prior_alpha"])
                            * score_scale
                            * query_prior
                        )
                    proxy_top = None
                    if "local_pool_block_size" in method:
                        proxy_top = block_local_global_topk(
                            proxy_scaled,
                            retrieval_count,
                            int(method["local_pool_block_size"]),
                            method["local_pool_hot_mask"],
                        )
                    detail_rows.append(
                        {
                            "topic": topic,
                            "layer": layer,
                            "step": int(record["step"]),
                            "method": method["method"],
                            "cold_budget": method["cold_budget"],
                            "hot_fraction": method["hot_fraction"],
                            "index_bytes": method["index_bytes"],
                            "calibration_selection_share_in_hotset": method[
                                "calibration_selection_share_in_hotset"
                            ],
                            **retrieval_metrics(
                                proxy_scaled,
                                reference,
                                retrieval_count,
                                proxy_top,
                            ),
                        }
                    )

            del methods, key, projected_key, high_key
            torch.cuda.empty_cache()

    with (args.output_dir / "detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    aggregate_rows = aggregate(detail_rows)
    baseline = next(
        row for row in aggregate_rows if row["method"] == "qksieve_240"
    )
    for row in aggregate_rows:
        row["mass_retention_vs_qksieve"] = (
            row["attention_mass"] / baseline["attention_mass"]
        )
        row["index_ratio_vs_qksieve"] = (
            row["index_bytes"] / baseline["index_bytes"]
        )
    qualified = [
        row
        for row in aggregate_rows
        if row["mass_retention_vs_qksieve"] >= 0.995
    ]
    qualified.sort(key=lambda row: row["index_bytes"])
    summary = {
        "schema": "qksieve_frequency_tiered_index_v2",
        "traces": args.trace,
        "calibration_steps": args.calibration_steps,
        "retrieval_fraction": args.retrieval_fraction,
        "frequency_is_causal": True,
        "selectors": {
            "frequency_tiered": "top calibration retrieval frequency",
            "crossing_tiered": (
                "top low-rate/high-rate membership disagreement plus "
                "positive score-error-minus-margin risk"
            ),
            "hybrid_tiered": (
                "equal-weight percentile ranks of frequency and crossing risk"
            ),
        },
        "baseline": baseline,
        "aggregate": aggregate_rows,
        "hardware_codebook_selections": codebook_selection_rows,
        "best_mass_retention_ge_99_5pct": (
            qualified[0] if qualified else None
        ),
        "storage_accounting": (
            "mixed high/low packed-index bytes plus 32-bit hot-token IDs; "
            "exact K/V storage is unchanged in this trace audit"
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
