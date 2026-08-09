from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from evaluate_pca_coselection_hybrid import grouped_scores, record_candidate_quality, summarize
from evaluate_spectral_error_feedback import (
    record_top2_output_quality,
    select_energy_band,
    update_query_state,
)


def binary_auc(scores: list[float], labels: list[float]) -> float:
    score = torch.tensor(scores, dtype=torch.float64)
    label = torch.tensor(labels, dtype=torch.float64)
    positive = int(label.sum().item())
    negative = int(label.numel() - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(score, descending=True)
    true_positive = torch.cumsum(label[order], dim=0) / positive
    false_positive = torch.cumsum(1.0 - label[order], dim=0) / negative
    true_positive = torch.cat((torch.zeros(1, dtype=true_positive.dtype), true_positive))
    false_positive = torch.cat((torch.zeros(1, dtype=false_positive.dtype), false_positive))
    return float(torch.trapz(true_positive, false_positive).item())


def candidate_contains(
    candidates: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return (targets.unsqueeze(-1) == candidates.unsqueeze(-2)).any(dim=-1)


def fixed_pca_rank_scores(
    projected_query: torch.Tensor,
    projected_key: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Score with the largest-variance `rank` dimensions of the PCA64 basis."""
    if projected_query.ndim != 2 or projected_key.ndim != 2:
        raise ValueError("query and key must both be matrices")
    if projected_query.shape[-1] != projected_key.shape[-1]:
        raise ValueError("query and key projection dimensions must match")
    if not 0 < rank <= projected_query.shape[-1]:
        raise ValueError("rank must be in (0, projection_dim]")
    return projected_query[:, -rank:] @ projected_key[:, -rank:].T


def quantize_dequantize_projected_int4(projected_key: torch.Tensor) -> torch.Tensor:
    """Emulate the deployed symmetric per-token INT4 PCA key representation."""
    if projected_key.shape[-1] % 2 != 0:
        raise ValueError("INT4 projection dimension must be even")
    scale = projected_key.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(projected_key.float() / scale).clamp(-7, 7)
    return codes * scale


def gaussian_expected_outside_crossings(
    approximate_scores: torch.Tensor,
    residual_sigma: torch.Tensor,
    keep_count: int,
    candidate_count: int,
) -> torch.Tensor:
    """Expected outside-pool tokens crossing the proxy top-k floor."""
    if approximate_scores.ndim != 2 or residual_sigma.ndim != 1:
        raise ValueError("scores must be [heads, tokens] and sigma must be [heads]")
    if approximate_scores.shape[0] != residual_sigma.shape[0]:
        raise ValueError("score and sigma head counts must match")
    top_values, top_indices = torch.topk(
        approximate_scores, k=candidate_count, dim=-1
    )
    target_floor = top_values[:, keep_count - 1]
    gap = (target_floor.unsqueeze(-1) - approximate_scores).clamp_min(0.0)
    crossing_probability = 0.5 * torch.erfc(
        gap / (2.0 * residual_sigma.unsqueeze(-1).clamp_min(1.0e-12))
    )
    inside = torch.zeros_like(approximate_scores, dtype=torch.bool)
    inside.scatter_(1, top_indices, True)
    return crossing_probability.masked_fill(inside, 0.0).sum(dim=-1)


def gaussian_tail_density_crossings(
    approximate_scores: torch.Tensor,
    residual_sigma: torch.Tensor,
    keep_count: int,
    candidate_count: int,
) -> torch.Tensor:
    """Estimate outside crossings from the empirical score density at the pool edge.

    The local density uses the lower half of the observed keep-to-candidate rank
    interval. Integrating the Gaussian crossing probability below the candidate
    boundary then has a closed form, avoiding a token-wise erfc scan.
    """
    if approximate_scores.ndim != 2 or residual_sigma.ndim != 1:
        raise ValueError("scores must be [heads, tokens] and sigma must be [heads]")
    if approximate_scores.shape[0] != residual_sigma.shape[0]:
        raise ValueError("score and sigma head counts must match")
    if not 0 < keep_count < candidate_count <= approximate_scores.shape[-1]:
        raise ValueError("counts must satisfy 0 < keep < candidate <= tokens")
    top_values = torch.topk(approximate_scores, k=candidate_count, dim=-1).values
    return gaussian_tail_density_crossings_from_top_values(
        top_values,
        residual_sigma,
        keep_count,
        approximate_scores.shape[-1],
    )


def gaussian_tail_density_crossings_from_top_values(
    top_values: torch.Tensor,
    residual_sigma: torch.Tensor,
    keep_count: int,
    total_token_count: int,
) -> torch.Tensor:
    """Closed-form tail crossings when descending candidate values are available."""
    candidate_count = int(top_values.shape[-1])
    if top_values.ndim != 2 or residual_sigma.ndim != 1:
        raise ValueError("top values must be [heads, candidates] and sigma [heads]")
    if top_values.shape[0] != residual_sigma.shape[0]:
        raise ValueError("top values and sigma head counts must match")
    if not 0 < keep_count < candidate_count <= total_token_count:
        raise ValueError("counts must satisfy 0 < keep < candidate <= tokens")
    density_start_rank = keep_count + (candidate_count - keep_count) // 2
    target_floor = top_values[:, keep_count - 1]
    density_ceiling = top_values[:, density_start_rank - 1]
    candidate_floor = top_values[:, candidate_count - 1]
    local_density = float(candidate_count - density_start_rank) / (
        density_ceiling - candidate_floor
    ).clamp_min(1.0e-12)

    sigma = residual_sigma.clamp_min(1.0e-12)
    boundary_gap = (target_floor - candidate_floor).clamp_min(0.0)
    normalized_gap = boundary_gap / (math.sqrt(2.0) * sigma)
    gaussian_pdf = torch.exp(-0.5 * normalized_gap.square()) / math.sqrt(
        2.0 * math.pi
    )
    gaussian_tail = 0.5 * torch.erfc(normalized_gap / math.sqrt(2.0))
    integrated_tail = math.sqrt(2.0) * sigma * (
        gaussian_pdf - normalized_gap * gaussian_tail
    ).clamp_min(0.0)
    outside_count = total_token_count - candidate_count
    return (local_density * integrated_tail).clamp(max=float(outside_count))


def gaussian_boundary_recall_estimate(
    anchor: torch.Tensor,
    current: torch.Tensor,
    spectral_weights: torch.Tensor,
    target_fraction: float,
    candidate_fraction: float,
) -> torch.Tensor:
    """Boundary-conditioned Gaussian estimate of target-in-candidate recall."""
    if anchor.shape != current.shape or anchor.ndim != 2:
        raise ValueError("anchor and current must have matching [heads, dims] shapes")
    if spectral_weights.ndim != 1 or spectral_weights.shape[0] != anchor.shape[1]:
        raise ValueError("spectral weights must match the query dimensions")
    if not 0.0 < target_fraction < candidate_fraction < 1.0:
        raise ValueError("fractions must satisfy 0 < target < candidate < 1")
    weighted_anchor = anchor * spectral_weights.unsqueeze(0)
    anchor_variance = (weighted_anchor * anchor).sum(dim=-1).clamp_min(1.0e-12)
    current_variance = (
        current.square() * spectral_weights.unsqueeze(0)
    ).sum(dim=-1).clamp_min(1.0e-12)
    covariance = (weighted_anchor * current).sum(dim=-1)
    correlation = (
        covariance / torch.sqrt(anchor_variance * current_variance)
    ).clamp(-0.999999, 0.999999)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=anchor.device),
        torch.tensor(1.0, device=anchor.device),
    )
    target_quantile = normal.icdf(
        torch.tensor(1.0 - target_fraction, device=anchor.device)
    )
    candidate_quantile = normal.icdf(
        torch.tensor(1.0 - candidate_fraction, device=anchor.device)
    )
    conditional_z = (
        correlation * target_quantile - candidate_quantile
    ) / torch.sqrt((1.0 - correlation.square()).clamp_min(1.0e-12))
    return torch.special.ndtr(conditional_z)


def evaluate_trace(
    path: Path,
    *,
    projection_dim: int,
    band_size: int,
    candidate_fraction: float,
    fixed_ranks: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    diagnostics: dict[str, list[float]] = defaultdict(list)
    transport_missed = 0
    pca_missed = 0
    transport_missed_second_band_alignment = 0
    total_target_positions = 0

    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        if len(records) < 2:
            continue
        key_record = next((record for record in records if record.get("key") is not None), None)
        if key_record is None:
            raise ValueError(f"layer {layer} has no stored key tensor")
        key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        value = value[:, :history_count]
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        kv_heads = int(key.shape[0])
        query_heads = int(queries.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(keep_count, math.ceil(candidate_fraction * history_count))

        exact_scores = grouped_scores(key, queries, group_size)
        exact_true = torch.topk(exact_scores, k=keep_count, dim=-1).indices
        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key) / float(
            sampled_key.shape[1]
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        retained_eigenvalues = eigenvalues[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        projected_key_int4 = quantize_dequantize_projected_int4(projected_key)
        grouped_query = queries.reshape(len(records), kv_heads, group_size, queries.shape[-1])
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)

        state = projected_query[0].clone()
        always_two_state = projected_query[0].clone()
        margin_state = projected_query[0].clone()
        extreme_margin_state = projected_query[0].clone()
        gaussian_risk_state = projected_query[0].clone()
        density_risk_state = projected_query[0].clone()
        one_shot_density_state = projected_query[0].clone()
        one_shot_density_int4_state = projected_query[0].clone()
        correlation_state = projected_query[0].clone()
        band_count = projection_dim // band_size
        for step in range(1, len(records)):
            full_scores_by_head: list[torch.Tensor] = []
            one_band_scores_by_head: list[torch.Tensor] = []
            two_band_scores_by_head: list[torch.Tensor] = []
            triggered_scores_by_head: list[torch.Tensor] = []
            extreme_margin_scores_by_head: list[torch.Tensor] = []
            gaussian_risk_scores_by_head: list[torch.Tensor] = []
            density_risk_scores_by_head: list[torch.Tensor] = []
            one_shot_density_scores_by_head: list[torch.Tensor] = []
            one_shot_density_int4_scores_by_head: list[torch.Tensor] = []
            full_int4_scores_by_head: list[torch.Tensor] = []
            correlation_scores_by_head: list[torch.Tensor] = []
            fixed_rank_scores_by_rank: dict[int, list[torch.Tensor]] = {
                rank: [] for rank in fixed_ranks
            }
            fixed_rank_int4_scores_by_rank: dict[int, list[torch.Tensor]] = {
                rank: [] for rank in fixed_ranks
            }
            trigger_by_kv_head: list[bool] = []
            second_band_by_kv_head: list[int] = []
            residual_after_first_by_kv_head: list[torch.Tensor] = []

            for kv_head in range(kv_heads):
                current = projected_query[step, kv_head]
                first_residual = current - state[kv_head]
                first_band = select_energy_band(
                    first_residual, retained_eigenvalues[kv_head], band_size
                )
                first_state = update_query_state(
                    state[kv_head], current, first_band, band_size
                )
                residual_after_first = current - first_state
                second_band = select_energy_band(
                    residual_after_first, retained_eigenvalues[kv_head], band_size
                )
                second_state = update_query_state(
                    first_state, current, second_band, band_size
                )
                keys = projected_key[kv_head]
                keys_int4 = projected_key_int4[kv_head]
                full_scores = current @ keys.T
                full_int4_scores = current @ keys_int4.T
                one_band_scores = first_state @ keys.T
                for rank in fixed_ranks:
                    fixed_rank_scores_by_rank[rank].append(
                        fixed_pca_rank_scores(current, keys, rank)
                    )
                    fixed_rank_int4_scores_by_rank[rank].append(
                        fixed_pca_rank_scores(current, keys_int4, rank)
                    )

                always_first_residual = current - always_two_state[kv_head]
                always_first_band = select_energy_band(
                    always_first_residual,
                    retained_eigenvalues[kv_head],
                    band_size,
                )
                always_first_state = update_query_state(
                    always_two_state[kv_head],
                    current,
                    always_first_band,
                    band_size,
                )
                always_second_residual = current - always_first_state
                always_second_band = select_energy_band(
                    always_second_residual,
                    retained_eigenvalues[kv_head],
                    band_size,
                )
                always_two_state[kv_head] = update_query_state(
                    always_first_state,
                    current,
                    always_second_band,
                    band_size,
                )
                two_band_scores = always_two_state[kv_head] @ keys.T

                margin_first_residual = current - margin_state[kv_head]
                margin_first_band = select_energy_band(
                    margin_first_residual,
                    retained_eigenvalues[kv_head],
                    band_size,
                )
                margin_first_state = update_query_state(
                    margin_state[kv_head],
                    current,
                    margin_first_band,
                    band_size,
                )
                margin_residual = current - margin_first_state
                margin_one_band_scores = margin_first_state @ keys.T

                predicted_sigma = torch.sqrt(
                    (
                        margin_residual.square()
                        * retained_eigenvalues[kv_head].unsqueeze(0)
                    ).sum(dim=-1)
                )
                top_values = torch.topk(
                    margin_one_band_scores, k=candidate_count, dim=-1
                ).values
                target_floor = top_values[:, keep_count - 1]
                candidate_floor = top_values[:, candidate_count - 1]
                candidate_buffer = (target_floor - candidate_floor).clamp_min(1.0e-12)
                danger_ratio = predicted_sigma / candidate_buffer
                trigger = bool((danger_ratio >= 1.0).any().item())

                if trigger:
                    margin_second_band = select_energy_band(
                        margin_residual,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    margin_state[kv_head] = update_query_state(
                        margin_first_state,
                        current,
                        margin_second_band,
                        band_size,
                    )
                else:
                    margin_state[kv_head] = margin_first_state
                triggered_scores = margin_state[kv_head] @ keys.T

                extreme_scan_count = 0
                extreme_ratio = float("inf")
                extreme_scores = extreme_margin_state[kv_head] @ keys.T
                while extreme_scan_count < band_count:
                    extreme_residual = current - extreme_margin_state[kv_head]
                    extreme_band = select_energy_band(
                        extreme_residual,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    extreme_margin_state[kv_head] = update_query_state(
                        extreme_margin_state[kv_head],
                        current,
                        extreme_band,
                        band_size,
                    )
                    extreme_scan_count += 1
                    extreme_scores = extreme_margin_state[kv_head] @ keys.T
                    remaining_residual = current - extreme_margin_state[kv_head]
                    remaining_sigma = torch.sqrt(
                        (
                            remaining_residual.square()
                            * retained_eigenvalues[kv_head].unsqueeze(0)
                        ).sum(dim=-1)
                    )
                    extreme_top_values = torch.topk(
                        extreme_scores, k=candidate_count, dim=-1
                    ).values
                    extreme_buffer = (
                        extreme_top_values[:, keep_count - 1]
                        - extreme_top_values[:, candidate_count - 1]
                    ).clamp_min(1.0e-12)
                    extreme_ratio = float(
                        (
                            remaining_sigma
                            * math.sqrt(2.0 * math.log(history_count))
                            / extreme_buffer
                        )
                        .max()
                        .item()
                    )
                    if extreme_ratio < 1.0:
                        break

                gaussian_scan_count = 0
                expected_crossings = torch.full(
                    (group_size,),
                    float("inf"),
                    dtype=torch.float32,
                    device=device,
                )
                gaussian_scores = gaussian_risk_state[kv_head] @ keys.T
                allowed_misses = 0.05 * keep_count
                while gaussian_scan_count < band_count:
                    gaussian_residual = current - gaussian_risk_state[kv_head]
                    gaussian_band = select_energy_band(
                        gaussian_residual,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    gaussian_risk_state[kv_head] = update_query_state(
                        gaussian_risk_state[kv_head],
                        current,
                        gaussian_band,
                        band_size,
                    )
                    gaussian_scan_count += 1
                    gaussian_scores = gaussian_risk_state[kv_head] @ keys.T
                    gaussian_remaining = current - gaussian_risk_state[kv_head]
                    gaussian_sigma = torch.sqrt(
                        (
                            gaussian_remaining.square()
                            * retained_eigenvalues[kv_head].unsqueeze(0)
                        ).sum(dim=-1)
                    )
                    expected_crossings = gaussian_expected_outside_crossings(
                        gaussian_scores,
                        gaussian_sigma,
                        keep_count,
                        candidate_count,
                    )
                    if bool((expected_crossings <= allowed_misses).all().item()):
                        break

                density_scan_count = 0
                density_expected_crossings = torch.full(
                    (group_size,),
                    float("inf"),
                    dtype=torch.float32,
                    device=device,
                )
                density_scores = density_risk_state[kv_head] @ keys.T
                while density_scan_count < band_count:
                    density_residual = current - density_risk_state[kv_head]
                    density_band = select_energy_band(
                        density_residual,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    density_risk_state[kv_head] = update_query_state(
                        density_risk_state[kv_head],
                        current,
                        density_band,
                        band_size,
                    )
                    density_scan_count += 1
                    density_scores = density_risk_state[kv_head] @ keys.T
                    density_remaining = current - density_risk_state[kv_head]
                    density_sigma = torch.sqrt(
                        (
                            density_remaining.square()
                            * retained_eigenvalues[kv_head].unsqueeze(0)
                        ).sum(dim=-1)
                    )
                    density_expected_crossings = gaussian_tail_density_crossings(
                        density_scores,
                        density_sigma,
                        keep_count,
                        candidate_count,
                    )
                    if bool(
                        (density_expected_crossings <= allowed_misses).all().item()
                    ):
                        break

                one_shot_residual = current - one_shot_density_state[kv_head]
                one_shot_first_band = select_energy_band(
                    one_shot_residual,
                    retained_eigenvalues[kv_head],
                    band_size,
                )
                one_shot_density_state[kv_head] = update_query_state(
                    one_shot_density_state[kv_head],
                    current,
                    one_shot_first_band,
                    band_size,
                )
                one_shot_scan_count = 1
                one_shot_scores = one_shot_density_state[kv_head] @ keys.T

                one_shot_int4_residual = (
                    current - one_shot_density_int4_state[kv_head]
                )
                one_shot_int4_first_band = select_energy_band(
                    one_shot_int4_residual,
                    retained_eigenvalues[kv_head],
                    band_size,
                )
                one_shot_density_int4_state[kv_head] = update_query_state(
                    one_shot_density_int4_state[kv_head],
                    current,
                    one_shot_int4_first_band,
                    band_size,
                )
                one_shot_int4_scan_count = 1
                one_shot_int4_scores = (
                    one_shot_density_int4_state[kv_head] @ keys_int4.T
                )
                one_shot_int4_top_values = torch.topk(
                    one_shot_int4_scores, k=candidate_count, dim=-1
                ).values
                one_shot_int4_expected_crossings = torch.full(
                    (group_size,),
                    float("inf"),
                    dtype=torch.float32,
                    device=device,
                )
                while True:
                    one_shot_int4_remaining = (
                        current - one_shot_density_int4_state[kv_head]
                    )
                    one_shot_int4_sigma = torch.sqrt(
                        (
                            one_shot_int4_remaining.square()
                            * retained_eigenvalues[kv_head].unsqueeze(0)
                        ).sum(dim=-1)
                    )
                    one_shot_int4_expected_crossings = (
                        gaussian_tail_density_crossings_from_top_values(
                            one_shot_int4_top_values,
                            one_shot_int4_sigma,
                            keep_count,
                            history_count,
                        )
                    )
                    if bool(
                        (one_shot_int4_expected_crossings <= allowed_misses)
                        .all()
                        .item()
                    ) or one_shot_int4_scan_count >= band_count:
                        break
                    one_shot_int4_next_band = select_energy_band(
                        one_shot_int4_remaining,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    one_shot_density_int4_state[kv_head] = update_query_state(
                        one_shot_density_int4_state[kv_head],
                        current,
                        one_shot_int4_next_band,
                        band_size,
                    )
                    one_shot_int4_scan_count += 1
                one_shot_int4_scores = (
                    one_shot_density_int4_state[kv_head] @ keys_int4.T
                )
                one_shot_top_values = torch.topk(
                    one_shot_scores, k=candidate_count, dim=-1
                ).values
                one_shot_expected_crossings = torch.full(
                    (group_size,),
                    float("inf"),
                    dtype=torch.float32,
                    device=device,
                )
                while True:
                    one_shot_remaining = (
                        current - one_shot_density_state[kv_head]
                    )
                    one_shot_sigma = torch.sqrt(
                        (
                            one_shot_remaining.square()
                            * retained_eigenvalues[kv_head].unsqueeze(0)
                        ).sum(dim=-1)
                    )
                    one_shot_expected_crossings = (
                        gaussian_tail_density_crossings_from_top_values(
                            one_shot_top_values,
                            one_shot_sigma,
                            keep_count,
                            history_count,
                        )
                    )
                    if bool(
                        (one_shot_expected_crossings <= allowed_misses)
                        .all()
                        .item()
                    ) or one_shot_scan_count >= band_count:
                        break
                    next_band = select_energy_band(
                        one_shot_remaining,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    one_shot_density_state[kv_head] = update_query_state(
                        one_shot_density_state[kv_head],
                        current,
                        next_band,
                        band_size,
                    )
                    one_shot_scan_count += 1
                one_shot_scores = one_shot_density_state[kv_head] @ keys.T

                correlation_scan_count = 0
                estimated_recall = torch.zeros(
                    (group_size,), dtype=torch.float32, device=device
                )
                correlation_scores = correlation_state[kv_head] @ keys.T
                while correlation_scan_count < band_count:
                    correlation_residual = current - correlation_state[kv_head]
                    correlation_band = select_energy_band(
                        correlation_residual,
                        retained_eigenvalues[kv_head],
                        band_size,
                    )
                    correlation_state[kv_head] = update_query_state(
                        correlation_state[kv_head],
                        current,
                        correlation_band,
                        band_size,
                    )
                    correlation_scan_count += 1
                    estimated_recall = gaussian_boundary_recall_estimate(
                        correlation_state[kv_head],
                        current,
                        retained_eigenvalues[kv_head],
                        target_fraction=0.02,
                        candidate_fraction=candidate_fraction,
                    )
                    if bool((estimated_recall >= 0.95).all().item()):
                        break
                correlation_scores = correlation_state[kv_head] @ keys.T

                full_pca_true = torch.topk(full_scores, k=keep_count, dim=-1).indices
                one_band_candidate = torch.topk(
                    margin_one_band_scores, k=candidate_count, dim=-1
                ).indices
                actual_danger = bool(
                    (~candidate_contains(one_band_candidate, full_pca_true)).any().item()
                )
                diagnostics["danger_ratio"].append(float(danger_ratio.max().item()))
                diagnostics["actual_transport_danger"].append(float(actual_danger))
                diagnostics["margin_trigger"].append(float(trigger))
                diagnostics["predicted_sigma"].extend(predicted_sigma.cpu().tolist())
                actual_error_std = (full_scores - margin_one_band_scores).std(dim=-1)
                diagnostics["actual_score_error_std"].extend(
                    actual_error_std.cpu().tolist()
                )
                diagnostics["sigma_to_actual_error_std"].extend(
                    (predicted_sigma / actual_error_std.clamp_min(1.0e-12)).cpu().tolist()
                )
                diagnostics["candidate_buffer"].extend(candidate_buffer.cpu().tolist())

                state[kv_head] = first_state
                full_scores_by_head.append(full_scores)
                full_int4_scores_by_head.append(full_int4_scores)
                one_band_scores_by_head.append(one_band_scores)
                two_band_scores_by_head.append(two_band_scores)
                triggered_scores_by_head.append(triggered_scores)
                extreme_margin_scores_by_head.append(extreme_scores)
                gaussian_risk_scores_by_head.append(gaussian_scores)
                density_risk_scores_by_head.append(density_scores)
                one_shot_density_scores_by_head.append(one_shot_scores)
                one_shot_density_int4_scores_by_head.append(one_shot_int4_scores)
                correlation_scores_by_head.append(correlation_scores)
                trigger_by_kv_head.append(trigger)
                second_band_by_kv_head.append(second_band)
                residual_after_first_by_kv_head.append(residual_after_first)
                diagnostics["extreme_margin_scanned_bands"].append(
                    float(extreme_scan_count)
                )
                diagnostics["extreme_margin_final_ratio"].append(extreme_ratio)
                diagnostics["gaussian_risk_scanned_bands"].append(
                    float(gaussian_scan_count)
                )
                diagnostics["gaussian_expected_crossings"].extend(
                    expected_crossings.cpu().tolist()
                )
                diagnostics["density_risk_scanned_bands"].append(
                    float(density_scan_count)
                )
                diagnostics["density_expected_crossings"].extend(
                    density_expected_crossings.cpu().tolist()
                )
                diagnostics["one_shot_density_scanned_bands"].append(
                    float(one_shot_scan_count)
                )
                diagnostics["one_shot_density_expected_crossings"].extend(
                    one_shot_expected_crossings.cpu().tolist()
                )
                diagnostics["one_shot_density_int4_scanned_bands"].append(
                    float(one_shot_int4_scan_count)
                )
                diagnostics["one_shot_density_int4_expected_crossings"].extend(
                    one_shot_int4_expected_crossings.cpu().tolist()
                )
                diagnostics["correlation_scanned_bands"].append(
                    float(correlation_scan_count)
                )
                diagnostics["correlation_estimated_recall"].extend(
                    estimated_recall.cpu().tolist()
                )

            method_scores = {
                "full_pca64": torch.cat(full_scores_by_head, dim=0),
                "full_pca64_int4": torch.cat(full_int4_scores_by_head, dim=0),
                "bandef16": torch.cat(one_band_scores_by_head, dim=0),
                "bandef32_always": torch.cat(two_band_scores_by_head, dim=0),
                "bandef_margin_trigger": torch.cat(triggered_scores_by_head, dim=0),
                "bandef_extreme_margin": torch.cat(
                    extreme_margin_scores_by_head, dim=0
                ),
                "bandef_gaussian_risk95": torch.cat(
                    gaussian_risk_scores_by_head, dim=0
                ),
                "bandef_density_risk95": torch.cat(
                    density_risk_scores_by_head, dim=0
                ),
                "bandef_one_shot_density95": torch.cat(
                    one_shot_density_scores_by_head, dim=0
                ),
                "bandef_one_shot_density95_int4": torch.cat(
                    one_shot_density_int4_scores_by_head, dim=0
                ),
                "bandef_correlation95": torch.cat(
                    correlation_scores_by_head, dim=0
                ),
            }
            method_scores.update(
                {
                    f"fixed_pca_rank{rank}": torch.cat(scores, dim=0)
                    for rank, scores in fixed_rank_scores_by_rank.items()
                }
            )
            method_scores.update(
                {
                    f"fixed_pca_rank{rank}_int4": torch.cat(scores, dim=0)
                    for rank, scores in fixed_rank_int4_scores_by_rank.items()
                }
            )
            for head in range(query_heads):
                kv_head = head // group_size
                candidates = {
                    method: torch.topk(scores[head], k=candidate_count).indices
                    for method, scores in method_scores.items()
                }
                for method, candidate in candidates.items():
                    record_candidate_quality(
                        metrics,
                        method,
                        candidate,
                        exact_scores[step, head],
                        exact_true[step, head],
                        keep_count,
                    )
                    record_top2_output_quality(
                        metrics,
                        method,
                        candidate,
                        exact_scores[step, head],
                        exact_true[step, head],
                        value[kv_head],
                        keep_count,
                        scaling=1.0 / math.sqrt(key.shape[-1]),
                    )

                full_pca_true = torch.topk(
                    method_scores["full_pca64"][head], k=keep_count
                ).indices
                full_candidate = candidates["full_pca64"]
                band_candidate = candidates["bandef16"]
                full_contains_exact = candidate_contains(full_candidate, exact_true[step, head])
                band_contains_exact = candidate_contains(band_candidate, exact_true[step, head])
                pca_missed += int((~full_contains_exact).sum().item())
                transport_mask = full_contains_exact & ~band_contains_exact
                transport_missed += int(transport_mask.sum().item())
                total_target_positions += int(keep_count)

                missed_transport_tokens = exact_true[step, head][transport_mask]
                if missed_transport_tokens.numel() > 0:
                    residual = residual_after_first_by_kv_head[kv_head][head % group_size]
                    residual_chunks = residual.reshape(-1, band_size)
                    token_key = projected_key[kv_head].index_select(
                        0, missed_transport_tokens
                    ).reshape(missed_transport_tokens.numel(), -1, band_size)
                    contributions = torch.einsum(
                        "bd,nbd->nb", residual_chunks, token_key
                    )
                    needed_band = contributions.argmax(dim=-1)
                    transport_missed_second_band_alignment += int(
                        (needed_band == second_band_by_kv_head[kv_head]).sum().item()
                    )

            diagnostics["kv_head_trigger_rate"].extend(
                float(value) for value in trigger_by_kv_head
            )

        del key, value, queries, exact_scores, exact_true, sampled_key, second_moment
        del eigenvalues, eigenvectors, basis, projected_key, projected_key_int4
        del projected_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    quality = {
        method: {name: summarize(values) for name, values in values_by_metric.items()}
        for method, values_by_metric in metrics.items()
    }
    trigger = torch.tensor(diagnostics["margin_trigger"], dtype=torch.bool)
    danger = torch.tensor(diagnostics["actual_transport_danger"], dtype=torch.bool)
    true_positive = int((trigger & danger).sum().item())
    false_positive = int((trigger & ~danger).sum().item())
    false_negative = int((~trigger & danger).sum().item())
    trigger_precision = true_positive / max(1, true_positive + false_positive)
    trigger_recall = true_positive / max(1, true_positive + false_negative)
    return {
        "path": str(path),
        "fixed_ranks": list(fixed_ranks),
        "quality": quality,
        "numerics": {
            name: summarize(values)
            for name, values in diagnostics.items()
            if name not in {"actual_transport_danger", "margin_trigger"}
        },
        "margin_rule": {
            "definition": "scan the second energy band iff predicted residual score std >= proxy top2-to-candidate-pool score buffer",
            "danger_auc": binary_auc(
                diagnostics["danger_ratio"], diagnostics["actual_transport_danger"]
            ),
            "trigger_rate": float(trigger.float().mean().item()),
            "danger_rate": float(danger.float().mean().item()),
            "precision": trigger_precision,
            "recall": trigger_recall,
            "average_scanned_dimensions": band_size
            * (1.0 + float(trigger.float().mean().item())),
        },
        "extreme_margin_rule": {
            "definition": "greedily scan bands until sigma * sqrt(2 log N) is below the proxy top2-to-candidate-pool score buffer",
            "average_scanned_bands": summarize(
                diagnostics["extreme_margin_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["extreme_margin_scanned_bands"])
            / max(1, len(diagnostics["extreme_margin_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["extreme_margin_scanned_bands"]
            )
            / max(1, len(diagnostics["extreme_margin_scanned_bands"])),
        },
        "gaussian_risk95_rule": {
            "definition": "scan bands until the Gaussian expected outside-pool crossings are at most 5% of the top2 target size",
            "average_scanned_bands": summarize(
                diagnostics["gaussian_risk_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["gaussian_risk_scanned_bands"])
            / max(1, len(diagnostics["gaussian_risk_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["gaussian_risk_scanned_bands"]
            )
            / max(1, len(diagnostics["gaussian_risk_scanned_bands"])),
        },
        "density_risk95_rule": {
            "definition": "fit the score density at the candidate edge and integrate the Gaussian crossing tail in closed form",
            "average_scanned_bands": summarize(
                diagnostics["density_risk_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["density_risk_scanned_bands"])
            / max(1, len(diagnostics["density_risk_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["density_risk_scanned_bands"]
            )
            / max(1, len(diagnostics["density_risk_scanned_bands"])),
        },
        "one_shot_density95_rule": {
            "definition": "use one initial top-k boundary and residual band energies to plan all remaining bands without repeated top-k",
            "average_scanned_bands": summarize(
                diagnostics["one_shot_density_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["one_shot_density_scanned_bands"])
            / max(1, len(diagnostics["one_shot_density_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["one_shot_density_scanned_bands"]
            )
            / max(1, len(diagnostics["one_shot_density_scanned_bands"])),
        },
        "one_shot_density95_int4_rule": {
            "definition": "same one-shot policy with deployed per-token symmetric INT4 PCA keys",
            "average_scanned_bands": summarize(
                diagnostics["one_shot_density_int4_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["one_shot_density_int4_scanned_bands"])
            / max(1, len(diagnostics["one_shot_density_int4_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["one_shot_density_int4_scanned_bands"]
            )
            / max(1, len(diagnostics["one_shot_density_int4_scanned_bands"])),
        },
        "correlation95_rule": {
            "definition": "scan bands until the Gaussian boundary-conditioned target-in-candidate recall estimate reaches 95%",
            "average_scanned_bands": summarize(
                diagnostics["correlation_scanned_bands"]
            ),
            "average_scanned_dimensions": band_size
            * sum(diagnostics["correlation_scanned_bands"])
            / max(1, len(diagnostics["correlation_scanned_bands"])),
            "full64_rate": sum(
                value == band_count
                for value in diagnostics["correlation_scanned_bands"]
            )
            / max(1, len(diagnostics["correlation_scanned_bands"])),
        },
        "failure_decomposition": {
            "target_positions": total_target_positions,
            "pca64_candidate_misses": pca_missed,
            "band_transport_candidate_misses_after_pca_success": transport_missed,
            "transport_miss_second_energy_band_alignment": (
                transport_missed_second_band_alignment / max(1, transport_missed)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerical diagnostics for BandEF.")
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--band_size", type=int, default=16)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--fixed_ranks", default="35,36")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fixed_ranks = tuple(
        sorted({int(value.strip()) for value in args.fixed_ranks.split(",") if value.strip()})
    )
    if not fixed_ranks or fixed_ranks[0] <= 0 or fixed_ranks[-1] > args.projection_dim:
        raise ValueError("fixed ranks must be in (0, projection_dim]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {
        "method": "BandEF numerical failure analysis and analytic margin trigger",
        "traces": [
            evaluate_trace(
                path,
                projection_dim=args.projection_dim,
                band_size=args.band_size,
                candidate_fraction=args.candidate_fraction,
                fixed_ranks=fixed_ranks,
                device=device,
            )
            for path in args.trace_paths
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trace in report["traces"]:
        print(trace["path"])
        for method, values in trace["quality"].items():
            print(
                method,
                f"recall={100.0 * values['top2_position_recall']['mean']:.2f}%",
                f"mass={100.0 * values['attention_mass']['mean']:.2f}%",
                f"output_l2={100.0 * values['oracle_top2_output_relative_l2']['mean']:.2f}%",
            )
        print("margin_rule", trace["margin_rule"])
        print("failure_decomposition", trace["failure_decomposition"])


if __name__ == "__main__":
    main()
