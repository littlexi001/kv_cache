#!/usr/bin/env python
"""Cross-text audit of a persistent query-score-metric binary KV index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from analyze_qaware_binarypc_blockmean_layer0_20260802 import (
    assert_numeric_backend_sane,
    binary_proxy_scores,
    encode_binary_principal,
    encode_joint_kv_residual_codebook,
    encode_product_kv_residual_codebook,
    encode_residual_codebook,
    evenly_spaced_indices,
    fit_binary_principal_projection,
    fit_joint_kv_residual_codebook,
    fit_product_kv_residual_codebook,
    fit_residual_codebook,
    quantize_blockwise_affine,
    quantize_log_error_norms,
    query_metric_factors,
)
from analyze_qksieve_block_coreset_20260802 import (
    block_coreset_tail_statistics,
    fit_block_coreset,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
    symmetric_quantize,
)
from analyze_qksieve_control_variate_layer0_probe_20260802 import (
    load_layer0_activations,
    output_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train_texts", type=Path, nargs="+", required=True)
    parser.add_argument("--test_texts", type=Path, nargs="+", required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--query_offsets", default="0,64,256,1024")
    parser.add_argument("--query_tokens", type=int, default=4)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--query_samples_per_text", type=int, default=256)
    parser.add_argument("--key_samples_per_text", type=int, default=1024)
    parser.add_argument("--local_key_sample_count", type=int, default=4096)
    parser.add_argument("--fraction", type=float, default=0.04)
    parser.add_argument("--eas_ratio", type=float, default=0.10)
    parser.add_argument(
        "--adaptive_coverages",
        default="0.50,0.60,0.70,0.80,0.90,0.95",
    )
    parser.add_argument("--binary_bits", type=int, default=64)
    parser.add_argument("--projection_iterations", type=int, default=6)
    parser.add_argument("--residual_vq_bits", type=int, default=4)
    parser.add_argument("--residual_vq_iterations", type=int, default=6)
    parser.add_argument(
        "--residual_binary_bits",
        default="",
        help="Comma-separated second-stage binary widths after the joint K ID.",
    )
    parser.add_argument("--residual_binary_iterations", type=int, default=6)
    parser.add_argument(
        "--residual_binary_candidate_fractions",
        default="",
        help="Comma-separated fractions that read the second binary stage.",
    )
    parser.add_argument(
        "--joint_rvq_weights",
        default="",
        help="Comma-separated normalized Value weights for joint K/V residual IDs.",
    )
    parser.add_argument(
        "--additive_value_bits",
        default="",
        help=(
            "Comma-separated residual Value-code widths. Each code is additive "
            "to the joint K/V ID, so its histogram can be accumulated separately."
        ),
    )
    parser.add_argument("--additive_value_iterations", type=int, default=6)
    parser.add_argument("--additive_refit_iterations", type=int, default=2)
    parser.add_argument(
        "--additive_block_sizes",
        default="",
        help=(
            "Comma-separated implicit block IDs for additive request-local "
            "K/V residual correction."
        ),
    )
    parser.add_argument(
        "--cv_samples",
        default="",
        help="Comma-separated proxy-mass systematic sample counts.",
    )
    parser.add_argument(
        "--cv_correction",
        choices=("raw", "mass", "mass_shrink", "joint_shrink"),
        default="raw",
    )
    parser.add_argument(
        "--tail_calibration_counts",
        default="",
        help="Comma-separated exact samples for affine proxy-score calibration.",
    )
    parser.add_argument(
        "--adaptive_error_tolerances",
        default="",
        help="Comma-separated relative RMS output-error tolerances.",
    )
    parser.add_argument(
        "--product_rvq_key_bits",
        default="",
        help="Comma-separated K-bit allocations within the residual VQ ID.",
    )
    parser.add_argument(
        "--projection_weighting",
        choices=("uniform", "value_jacobian"),
        default="uniform",
    )
    parser.add_argument("--risk_lambda", type=float, default=1.0)
    parser.add_argument("--risk_error_bits", type=int, default=4)
    parser.add_argument("--risk_error_block_size", type=int, default=256)
    parser.add_argument("--metric_shrinkage", default="oas")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--key_mean_bits", type=int, default=8)
    parser.add_argument("--value_mean_bits", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def value_jacobian_weights(
    metric_keys: torch.Tensor,
    values: torch.Tensor,
    metric_queries: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Regularized output-Jacobian weights for score reconstruction."""
    scores = metric_queries.float() @ metric_keys.float().T * scale
    attention = torch.softmax(scores, dim=-1)
    outputs = attention @ values.float()
    value_norm = values.float().square().sum(dim=-1)
    output_norm = outputs.square().sum(dim=-1, keepdim=True)
    cross = outputs @ values.float().T
    value_delta_sq = (output_norm + value_norm[None, :] - 2.0 * cross).clamp_min(0)
    influence = (attention.square() * value_delta_sq).mean(dim=0)
    normalized = influence / influence.mean().clamp_min(1.0e-12)
    return 1.0 + normalized


def build_rabitq_index(keys: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    """Reference RaBitQ index using the official centering/alpha equations."""
    generator = torch.Generator(device=keys.device).manual_seed(seed)
    random_matrix = torch.randn(
        keys.shape[-1],
        keys.shape[-1],
        generator=generator,
        device=keys.device,
        dtype=torch.float32,
    )
    rotation, _ = torch.linalg.qr(random_matrix)
    key_centroid = keys.float().mean(dim=0)
    centered = keys.float() - key_centroid
    norms = centered.norm(dim=-1).clamp_min(1.0e-8)
    normalized = centered / norms[:, None]
    rotated = centered @ rotation.T
    codes = torch.where(rotated >= 0, 1.0, -1.0)
    cube = codes / math.sqrt(keys.shape[-1])
    reconstructed_unit = cube @ rotation
    alpha = (normalized * reconstructed_unit).sum(dim=-1).clamp_min(1.0e-8)
    return {
        "rotation": rotation,
        "key_centroid": key_centroid,
        "norms": norms,
        "codes": codes,
        "alpha": alpha,
    }


def rabitq_proxy_scores(
    index: dict[str, torch.Tensor],
    keys: torch.Tensor,
    query: torch.Tensor,
    query_centroid: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference FP-query RaBitQ score reconstruction before adaptive Top-p."""
    rotation = index["rotation"]
    key_centroid = index["key_centroid"]
    norms = index["norms"]
    codes = index["codes"]
    alpha = index["alpha"]
    centered_query = query.float() - query_centroid.float()
    rotated_query = centered_query @ rotation.T
    binary_inner = (codes @ rotated_query) / math.sqrt(query.numel())
    centered_inner = norms * binary_inner / alpha
    centroid_completion = (
        query.float() @ key_centroid
        + keys.float() @ query_centroid.float()
        - query_centroid.float() @ key_centroid
    )
    return (centered_inner + centroid_completion) * scale


def fit_rvq_value_centroids(
    values: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    bits: int,
) -> tuple[torch.Tensor, float]:
    """Fit request-local Value means using an existing residual-code ID."""
    fallback = values.float().mean(dim=0)
    centroids = []
    for cluster in range(clusters):
        members = assignments == cluster
        centroids.append(values[members].float().mean(dim=0) if bool(members.any()) else fallback)
    centroids_tensor = symmetric_quantize(torch.stack(centroids), bits, (1,))
    stored_bits = clusters * (bits * values.shape[-1] + 16)
    return centroids_tensor, stored_bits / values.shape[0]


def cluster_means(
    values: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Return dense cluster means while giving empty IDs a stable fallback."""
    means = []
    for cluster in range(clusters):
        members = assignments == cluster
        means.append(
            values[members].float().mean(dim=0)
            if bool(members.any())
            else fallback.float()
        )
    return torch.stack(means)


def fit_additive_value_residual_model(
    values: torch.Tensor,
    primary_assignments: torch.Tensor,
    primary_clusters: int,
    residual_bits: int,
    iterations: int,
) -> dict[str, Any]:
    """Fit an offline residual code V ~= mu[joint_id] + nu[residual_id]."""
    if residual_bits < 1:
        raise ValueError("additive Value residual width must be positive")
    global_mean = values.float().mean(dim=0)
    primary_centroids = cluster_means(
        values,
        primary_assignments,
        primary_clusters,
        global_mean,
    )
    residual = values.float() - primary_centroids.index_select(
        0, primary_assignments
    )
    residual_scale = residual.square().sum(dim=-1).mean().sqrt().clamp_min(1.0e-8)
    return {
        "bits": int(residual_bits),
        "primary_centroids": primary_centroids,
        "residual_scale": float(residual_scale),
        "residual_codebook": fit_residual_codebook(
            residual / residual_scale,
            1 << residual_bits,
            iterations,
        ),
    }


def encode_additive_value_residual(
    values: torch.Tensor,
    primary_assignments: torch.Tensor,
    model: dict[str, Any],
) -> torch.Tensor:
    """Assign a second, independent ID to residual Value geometry."""
    primary_centroids = model["primary_centroids"]
    residual_codebook = model["residual_codebook"]
    if not isinstance(primary_centroids, torch.Tensor) or not isinstance(
        residual_codebook, torch.Tensor
    ):
        raise TypeError("additive Value model tensors are malformed")
    residual = values.float() - primary_centroids.float().index_select(
        0, primary_assignments
    )
    normalized = residual / float(model["residual_scale"])
    return torch.cdist(normalized, residual_codebook.float()).argmin(dim=-1)


def fit_additive_request_centroids(
    values: torch.Tensor,
    primary_assignments: torch.Tensor,
    residual_assignments: torch.Tensor,
    primary_clusters: int,
    residual_clusters: int,
    bits: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Refit two additive codebooks from request-local sufficient statistics."""
    if iterations < 1:
        raise ValueError("additive refit iterations must be positive")
    values = values.float()
    zero = torch.zeros(values.shape[-1], dtype=torch.float32, device=values.device)
    residual_centroids = torch.zeros(
        residual_clusters,
        values.shape[-1],
        dtype=torch.float32,
        device=values.device,
    )
    primary_centroids = torch.zeros(
        primary_clusters,
        values.shape[-1],
        dtype=torch.float32,
        device=values.device,
    )
    for _ in range(iterations):
        primary_centroids = cluster_means(
            values - residual_centroids.index_select(0, residual_assignments),
            primary_assignments,
            primary_clusters,
            zero,
        )
        residual_centroids = cluster_means(
            values - primary_centroids.index_select(0, primary_assignments),
            residual_assignments,
            residual_clusters,
            zero,
        )
    primary_centroids = symmetric_quantize(primary_centroids, bits, (1,))
    residual_centroids = symmetric_quantize(residual_centroids, bits, (1,))
    stored_bits = (primary_clusters + residual_clusters) * (
        bits * values.shape[-1] + 16
    )
    return (
        primary_centroids,
        residual_centroids,
        stored_bits / values.shape[0],
    )


def rvq_tail_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    assignments: torch.Tensor,
    value_centroids: torch.Tensor,
    calibration_count: int = 0,
    selected_conditioned: bool = False,
) -> torch.Tensor:
    """Complete omitted attention with masses accumulated by residual-code ID."""
    calibrated_proxy = affine_calibrated_proxy_scores(
        exact_scores, proxy_scores, calibration_count
    )

    reference = exact_scores.index_select(0, selected).amin()
    selected_exp = (exact_scores.index_select(0, selected) - reference).exp()
    selected_numerator = (
        selected_exp[:, None] * values.float().index_select(0, selected)
    ).sum(dim=0)
    omitted = torch.ones(
        exact_scores.numel(), dtype=torch.bool, device=exact_scores.device
    )
    omitted[selected] = False
    omitted_weights = (calibrated_proxy[omitted] - reference).exp()
    omitted_assignments = assignments[omitted]
    cluster_mass = torch.zeros(
        value_centroids.shape[0],
        dtype=torch.float32,
        device=exact_scores.device,
    )
    cluster_mass.scatter_add_(0, omitted_assignments, omitted_weights)
    effective_centroids = value_centroids.float()
    if selected_conditioned:
        cluster_counts = torch.bincount(
            assignments, minlength=value_centroids.shape[0]
        ).float()
        selected_assignments = assignments.index_select(0, selected)
        selected_counts = torch.bincount(
            selected_assignments, minlength=value_centroids.shape[0]
        ).float()
        selected_value_sum = torch.zeros_like(effective_centroids)
        selected_value_sum.index_add_(
            0,
            selected_assignments,
            values.float().index_select(0, selected),
        )
        omitted_counts = cluster_counts - selected_counts
        effective_centroids = (
            effective_centroids * cluster_counts[:, None] - selected_value_sum
        ) / omitted_counts.clamp_min(1.0)[:, None]
        effective_centroids = torch.where(
            omitted_counts[:, None] > 0,
            effective_centroids,
            torch.zeros_like(effective_centroids),
        )
    tail_numerator = cluster_mass @ effective_centroids
    return (selected_numerator + tail_numerator) / (
        selected_exp.sum() + cluster_mass.sum()
    ).clamp_min(1.0e-12)


def affine_calibrated_proxy_scores(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Fit one query-local affine score map from fixed, evenly spaced samples."""
    calibrated_proxy = proxy_scores.float()
    if sample_count <= 1:
        return calibrated_proxy
    sample = evenly_spaced_indices(
        exact_scores.numel(), sample_count, exact_scores.device
    )
    sample_proxy = calibrated_proxy.index_select(0, sample)
    sample_exact = exact_scores.float().index_select(0, sample)
    centered_proxy = sample_proxy - sample_proxy.mean()
    slope = (
        (centered_proxy * (sample_exact - sample_exact.mean())).sum()
        / centered_proxy.square().sum().clamp_min(1.0e-12)
    ).clamp_min(0.0)
    intercept = sample_exact.mean() - slope * sample_proxy.mean()
    return slope * calibrated_proxy + intercept


def additive_rvq_tail_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    primary_assignments: torch.Tensor,
    primary_centroids: torch.Tensor,
    residual_assignments: torch.Tensor,
    residual_centroids: torch.Tensor,
) -> torch.Tensor:
    """Complete omitted output with two separable code histograms."""
    reference = exact_scores.index_select(0, selected).amin()
    selected_exp = (exact_scores.index_select(0, selected) - reference).exp()
    selected_numerator = (
        selected_exp[:, None] * values.float().index_select(0, selected)
    ).sum(dim=0)
    omitted = torch.ones(
        exact_scores.numel(), dtype=torch.bool, device=exact_scores.device
    )
    omitted[selected] = False
    omitted_weights = (proxy_scores.float()[omitted] - reference).exp()
    primary_mass = torch.zeros(
        primary_centroids.shape[0], dtype=torch.float32, device=values.device
    )
    residual_mass = torch.zeros(
        residual_centroids.shape[0], dtype=torch.float32, device=values.device
    )
    primary_mass.scatter_add_(0, primary_assignments[omitted], omitted_weights)
    residual_mass.scatter_add_(0, residual_assignments[omitted], omitted_weights)
    tail_numerator = (
        primary_mass @ primary_centroids.float()
        + residual_mass @ residual_centroids.float()
    )
    return (selected_numerator + tail_numerator) / (
        selected_exp.sum() + omitted_weights.sum()
    ).clamp_min(1.0e-12)


def proxy_mass_control_variate_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    approximate_values: torch.Tensor,
    selected: torch.Tensor,
    sample_count: int,
    phase: float = 0.5,
    correction_mode: str = "raw",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Correct proxy tail totals by low-variance systematic importance samples.

    The proxy scan supplies a normalized proposal and an inexpensive output
    baseline. Exact K/V reads are required only for sampled tail tokens. The
    denominator estimate is positive by construction, unlike an additive
    Horvitz--Thompson denominator residual.
    """
    if sample_count < 1:
        raise ValueError("control-variate sample count must be positive")
    if not 0.0 <= phase < 1.0:
        raise ValueError("systematic-sampling phase must lie in [0, 1)")
    token_count = exact_scores.numel()
    if values.shape != approximate_values.shape or values.shape[0] != token_count:
        raise ValueError("control-variate tensors have incompatible shapes")
    omitted = torch.ones(
        token_count, dtype=torch.bool, device=exact_scores.device
    )
    omitted[selected] = False
    omitted_indices = torch.nonzero(omitted, as_tuple=False).flatten()
    if omitted_indices.numel() == 0:
        weights = torch.softmax(exact_scores.index_select(0, selected), dim=0)
        return (
            weights @ values.float().index_select(0, selected),
            {"sample_tokens": 0.0, "sample_unique_tokens": 0.0},
        )

    reference = exact_scores.index_select(0, selected).amin()
    selected_weights = (
        exact_scores.index_select(0, selected).float() - reference
    ).exp()
    selected_numerator = (
        selected_weights[:, None]
        * values.float().index_select(0, selected)
    ).sum(dim=0)
    tail_proxy = proxy_scores.float().index_select(0, omitted_indices)
    tail_proxy_weights = (tail_proxy - reference).exp()
    proxy_partition = tail_proxy_weights.sum().clamp_min(1.0e-20)
    tail_approximate_values = approximate_values.float().index_select(
        0, omitted_indices
    )
    base_numerator = (
        tail_proxy_weights[:, None] * tail_approximate_values
    ).sum(dim=0)

    positions = (
        torch.arange(
            sample_count,
            dtype=torch.float32,
            device=exact_scores.device,
        )
        + float(phase)
    ) * (proxy_partition / float(sample_count))
    cumulative = tail_proxy_weights.cumsum(dim=0)
    local_samples = torch.searchsorted(cumulative, positions).clamp_max(
        omitted_indices.numel() - 1
    )
    sampled = omitted_indices.index_select(0, local_samples)
    sampled_proxy = proxy_scores.float().index_select(0, sampled)
    sampled_exact = exact_scores.float().index_select(0, sampled)
    likelihood_ratio = (sampled_exact - sampled_proxy).exp()
    sampled_values = values.float().index_select(0, sampled)
    sampled_approximation = approximate_values.float().index_select(0, sampled)
    ratio_mean = likelihood_ratio.mean()
    denominator_delta = likelihood_ratio - 1.0
    denominator_signal = denominator_delta.mean().square()
    denominator_noise = denominator_delta.var(unbiased=False) / float(sample_count)
    denominator_shrinkage = denominator_signal / (
        denominator_signal + denominator_noise + 1.0e-20
    )
    numerator_delta = (
        likelihood_ratio[:, None] * sampled_values - sampled_approximation
    )
    numerator_mean = numerator_delta.mean(dim=0)
    numerator_signal = numerator_mean.square().sum()
    numerator_noise = (
        numerator_delta.var(dim=0, unbiased=False).sum() / float(sample_count)
    )
    numerator_shrinkage = numerator_signal / (
        numerator_signal + numerator_noise + 1.0e-20
    )
    if correction_mode == "raw":
        corrected_partition = proxy_partition * ratio_mean
        corrected_numerator = base_numerator + proxy_partition * numerator_mean
    elif correction_mode == "mass":
        corrected_partition = proxy_partition * ratio_mean
        corrected_numerator = base_numerator * ratio_mean
    elif correction_mode == "mass_shrink":
        corrected_ratio = (
            1.0
            + denominator_shrinkage
            * (ratio_mean - 1.0)
        ).clamp_min(1.0e-4)
        corrected_partition = proxy_partition * corrected_ratio
        corrected_numerator = base_numerator * corrected_ratio
    elif correction_mode == "joint_shrink":
        corrected_ratio = (
            1.0
            + denominator_shrinkage
            * (ratio_mean - 1.0)
        ).clamp_min(1.0e-4)
        corrected_partition = proxy_partition * corrected_ratio
        corrected_numerator = (
            base_numerator
            + proxy_partition * numerator_shrinkage * numerator_mean
        )
    else:
        raise ValueError(f"unknown control-variate correction: {correction_mode}")
    output = (selected_numerator + corrected_numerator) / (
        selected_weights.sum() + corrected_partition
    ).clamp_min(1.0e-20)
    return output, {
        "sample_tokens": float(sample_count),
        "sample_unique_tokens": float(sampled.unique().numel()),
        "likelihood_ratio_mean": float(likelihood_ratio.mean()),
        "likelihood_ratio_p90": float(torch.quantile(likelihood_ratio, 0.90)),
        "likelihood_ratio_max": float(likelihood_ratio.max()),
        "denominator_shrinkage": float(denominator_shrinkage),
        "numerator_shrinkage": float(numerator_shrinkage),
        "correction_mode": correction_mode,
    }


def select_by_output_rms_bound(
    proxy_scores: torch.Tensor,
    score_uncertainty: torch.Tensor,
    approximate_values: torch.Tensor,
    value_errors: torch.Tensor,
    relative_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select the minimum risk prefix meeting an RMS output-error target.

    Centroid residuals are zero-mean by construction. Under the corresponding
    independent residual model, squared token contribution risks add, yielding
    a length-agnostic budget rule rather than a fixed percentage or cap.
    """
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative output-error tolerance must lie in (0, 1)")
    if proxy_scores.shape != score_uncertainty.shape:
        raise ValueError("proxy scores and uncertainties must align")
    if value_errors.shape != proxy_scores.shape:
        raise ValueError("Value errors must align with scores")
    if approximate_values.shape[0] != proxy_scores.numel():
        raise ValueError("approximate Values must align with scores")
    maximum = proxy_scores.float().max()
    weights = (proxy_scores.float() - maximum).exp()
    partition = weights.sum().clamp_min(1.0e-20)
    approximate_output = (
        weights[:, None] * approximate_values.float()
    ).sum(dim=0) / partition
    output_norm = approximate_output.norm().clamp_min(1.0e-6)
    value_norm = approximate_values.float().norm(dim=-1)
    sensitivity = (
        value_errors.float()
        + score_uncertainty.float()
        * (value_norm + value_errors.float())
    ).clamp_min(1.0e-12)
    contribution_risk = weights * sensitivity
    order = torch.argsort(contribution_risk, descending=True)
    ordered_square = contribution_risk.index_select(0, order).square()
    residual_square = (
        ordered_square.sum() - ordered_square.cumsum(dim=0)
    ).clamp_min(0.0)
    target_square = (
        relative_tolerance * partition * output_norm
    ).square()
    feasible = torch.nonzero(
        residual_square <= target_square, as_tuple=False
    ).flatten()
    keep = int(feasible[0]) + 1 if feasible.numel() else proxy_scores.numel()
    selected = order[:keep]
    initial_estimate = ordered_square.sum().sqrt() / (partition * output_norm)
    residual_estimate = residual_square[keep - 1].sqrt() / (
        partition * output_norm
    )
    return selected, {
        "predicted_relative_error_before_selection": float(initial_estimate),
        "predicted_relative_error_after_selection": float(residual_estimate),
        "relative_tolerance": float(relative_tolerance),
    }


def rms_standardized_error_scale(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    score_uncertainty: torch.Tensor,
    sample_indices: torch.Tensor,
) -> float:
    """Estimate a query-local RMS multiplier for heteroskedastic score risk."""
    if not (
        exact_scores.shape == proxy_scores.shape == score_uncertainty.shape
    ):
        raise ValueError("score calibration vectors must align")
    if sample_indices.numel() == 0:
        raise ValueError("score calibration requires at least one sample")
    observed = (
        exact_scores.index_select(0, sample_indices)
        - proxy_scores.index_select(0, sample_indices)
    ).abs()
    standardized = observed / score_uncertainty.index_select(
        0, sample_indices
    ).clamp_min(1.0e-6)
    return float(standardized.square().mean().sqrt().clamp_min(1.0e-3))


def solve_three_action_rms_budget(
    proxy_scores: torch.Tensor,
    base_score_uncertainty: torch.Tensor,
    refined_score_uncertainty: torch.Tensor,
    approximate_values: torch.Tensor,
    value_errors: torch.Tensor,
    relative_tolerance: float,
    refinement_cost: float,
    exact_cost: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Minimize modeled read cost under an additive output-MSE constraint.

    Each token independently takes one of three actions: keep the base score,
    read a progressive residual score code, or read exact K/V. A scalar
    Lagrange multiplier makes the discrete per-token choices separable; binary
    search finds the least-cost feasible operating point without a learned or
    length-specific budget rule.
    """
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative output-error tolerance must lie in (0, 1)")
    if refinement_cost <= 0.0 or exact_cost <= refinement_cost:
        raise ValueError("action costs must satisfy 0 < refinement < exact")
    if not (
        proxy_scores.shape
        == base_score_uncertainty.shape
        == refined_score_uncertainty.shape
        == value_errors.shape
    ):
        raise ValueError("score, uncertainty, and Value-error vectors must align")
    if approximate_values.shape[0] != proxy_scores.numel():
        raise ValueError("approximate Values must align with scores")

    maximum = proxy_scores.float().max()
    weights = (proxy_scores.float() - maximum).exp()
    partition = weights.sum().clamp_min(1.0e-20)
    approximate_output = (
        weights[:, None] * approximate_values.float()
    ).sum(dim=0) / partition
    output_norm = approximate_output.norm().clamp_min(1.0e-6)
    value_norm = approximate_values.float().norm(dim=-1)
    value_scale = value_norm + value_errors.float()
    base_sensitivity = (
        value_errors.float() + base_score_uncertainty.float() * value_scale
    ).clamp_min(1.0e-12)
    refined_sensitivity = (
        value_errors.float() + refined_score_uncertainty.float() * value_scale
    ).clamp_min(1.0e-12)
    base_square = (weights * base_sensitivity).square()
    refined_square = torch.minimum(
        base_square,
        (weights * refined_sensitivity).square(),
    )
    target_square = (relative_tolerance * partition * output_norm).square()

    def actions_for(multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
        objectives = torch.stack(
            (
                base_square * multiplier,
                refined_square * multiplier + refinement_cost,
                torch.full_like(base_square, exact_cost),
            ),
            dim=0,
        )
        actions = objectives.argmin(dim=0)
        residual_square = torch.where(
            actions == 0,
            base_square,
            torch.where(actions == 1, refined_square, 0.0),
        ).sum()
        return actions, residual_square

    initial_square = base_square.sum()
    if initial_square <= target_square:
        actions = torch.zeros_like(base_square, dtype=torch.long)
        residual_square = initial_square
        multiplier = 0.0
    else:
        low = 0.0
        high = 1.0
        actions, residual_square = actions_for(high)
        for _ in range(96):
            if residual_square <= target_square:
                break
            high *= 2.0
            actions, residual_square = actions_for(high)
        else:
            raise RuntimeError("failed to find a feasible three-action multiplier")
        # Twenty steps give sub-ppm relative precision in the multiplier; the
        # discrete action set changes only at much coarser floating thresholds.
        for _ in range(20):
            middle = 0.5 * (low + high)
            candidate_actions, candidate_square = actions_for(middle)
            if candidate_square <= target_square:
                high = middle
                actions = candidate_actions
                residual_square = candidate_square
            else:
                low = middle
        multiplier = high

    refined = torch.nonzero(actions == 1, as_tuple=False).flatten()
    exact = torch.nonzero(actions == 2, as_tuple=False).flatten()
    predicted_relative_error = residual_square.sqrt() / (
        partition * output_norm
    )
    action_cost = refinement_cost * refined.numel() + exact_cost * exact.numel()
    return refined, exact, {
        "predicted_relative_error_before_actions": float(
            initial_square.sqrt() / (partition * output_norm)
        ),
        "predicted_relative_error_after_actions": float(predicted_relative_error),
        "relative_tolerance": float(relative_tolerance),
        "lagrange_multiplier": float(multiplier),
        "refined_tokens": float(refined.numel()),
        "exact_tokens": float(exact.numel()),
        "modeled_variable_cost": float(action_cost),
    }


def replan_exact_after_refinement(
    base_proxy_scores: torch.Tensor,
    refined_proxy_scores: torch.Tensor,
    base_score_uncertainty: torch.Tensor,
    refined_score_uncertainty: torch.Tensor,
    approximate_values: torch.Tensor,
    value_errors: torch.Tensor,
    relative_tolerance: float,
    refined_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Recompute the exact-KV action after consuming progressive score bits.

    The three-action relaxation decides where an extra score code is worth its
    read cost before that code is observed. Once read, both its point estimate
    and uncertainty are known more accurately. Replanning closes that feedback
    loop instead of selecting exact K/V from the stale base proxy.
    """
    if not (
        base_proxy_scores.shape
        == refined_proxy_scores.shape
        == base_score_uncertainty.shape
        == refined_score_uncertainty.shape
        == value_errors.shape
    ):
        raise ValueError("score, uncertainty, and Value-error vectors must align")
    if approximate_values.shape[0] != base_proxy_scores.numel():
        raise ValueError("approximate Values must align with scores")
    if refined_indices.ndim != 1:
        raise ValueError("refined indices must be one-dimensional")

    hybrid_proxy = base_proxy_scores.clone()
    hybrid_uncertainty = base_score_uncertainty.clone()
    if refined_indices.numel():
        hybrid_proxy[refined_indices] = refined_proxy_scores.index_select(
            0, refined_indices
        )
        hybrid_uncertainty[refined_indices] = (
            refined_score_uncertainty.index_select(0, refined_indices)
        )
    exact, diagnostics = select_by_output_rms_bound(
        hybrid_proxy,
        hybrid_uncertainty,
        approximate_values,
        value_errors,
        relative_tolerance,
    )
    return exact, hybrid_proxy, hybrid_uncertainty, diagnostics


def rerank_exact_after_refinement(
    base_proxy_scores: torch.Tensor,
    refined_proxy_scores: torch.Tensor,
    base_score_uncertainty: torch.Tensor,
    refined_score_uncertainty: torch.Tensor,
    approximate_values: torch.Tensor,
    value_errors: torch.Tensor,
    refined_indices: torch.Tensor,
    exact_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Minimize residual output risk at the solver's fixed exact-KV cost.

    After progressive bits are consumed, selecting the largest squared hybrid
    contribution risks is the exact cardinality-constrained minimizer of the
    additive output-MSE model. This updates token identities without silently
    increasing the exact-KV budget chosen by the three-action solver.
    """
    if not (
        base_proxy_scores.shape
        == refined_proxy_scores.shape
        == base_score_uncertainty.shape
        == refined_score_uncertainty.shape
        == value_errors.shape
    ):
        raise ValueError("score, uncertainty, and Value-error vectors must align")
    if approximate_values.shape[0] != base_proxy_scores.numel():
        raise ValueError("approximate Values must align with scores")
    if refined_indices.ndim != 1:
        raise ValueError("refined indices must be one-dimensional")
    if not 0 <= exact_count <= base_proxy_scores.numel():
        raise ValueError("exact count must lie in [0, token count]")

    hybrid_proxy = base_proxy_scores.clone()
    hybrid_uncertainty = base_score_uncertainty.clone()
    if refined_indices.numel():
        hybrid_proxy[refined_indices] = refined_proxy_scores.index_select(
            0, refined_indices
        )
        hybrid_uncertainty[refined_indices] = (
            refined_score_uncertainty.index_select(0, refined_indices)
        )

    maximum = hybrid_proxy.float().max()
    weights = (hybrid_proxy.float() - maximum).exp()
    partition = weights.sum().clamp_min(1.0e-20)
    approximate_output = (
        weights[:, None] * approximate_values.float()
    ).sum(dim=0) / partition
    output_norm = approximate_output.norm().clamp_min(1.0e-6)
    value_scale = approximate_values.float().norm(dim=-1) + value_errors.float()
    sensitivity = (
        value_errors.float() + hybrid_uncertainty.float() * value_scale
    ).clamp_min(1.0e-12)
    squared_risk = (weights * sensitivity).square()
    exact = torch.topk(squared_risk, exact_count, sorted=False).indices
    removed = (
        squared_risk.index_select(0, exact).sum()
        if exact_count
        else squared_risk.new_zeros(())
    )
    residual = (squared_risk.sum() - removed).clamp_min(0.0)
    return exact, hybrid_proxy, hybrid_uncertainty, {
        "fixed_exact_tokens": float(exact_count),
        "predicted_relative_error_after_fixed_cost_rerank": float(
            residual.sqrt() / (partition * output_norm)
        ),
    }


def token_ids_for_text(
    tokenizer: Any,
    path: Path,
    needed: int,
) -> torch.Tensor:
    text = path.read_text(encoding="utf-8", errors="ignore")
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=needed,
    ).input_ids[0, :needed]
    if token_ids.numel() < needed:
        raise ValueError(f"{path} contains fewer than {needed} tokens")
    return token_ids


def fit_global_codebooks(
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_samples: list[list[torch.Tensor]] | None = None
    query_samples: list[list[torch.Tensor]] | None = None
    value_samples: list[list[torch.Tensor]] | None = None
    metadata: dict[str, Any] | None = None
    for text_path in args.train_texts:
        token_ids = token_ids_for_text(tokenizer, text_path, args.history_tokens)
        query, key, value, current_metadata = load_layer0_activations(
            args.model, token_ids
        )
        if metadata is None:
            metadata = current_metadata
            key_samples = [[] for _ in range(int(metadata["kv_heads"]))]
            query_samples = [[] for _ in range(int(metadata["kv_heads"]))]
            value_samples = [[] for _ in range(int(metadata["kv_heads"]))]
        elif current_metadata != metadata:
            raise ValueError("training activation metadata changed across texts")
        assert (
            key_samples is not None
            and query_samples is not None
            and value_samples is not None
        )
        key_indices = evenly_spaced_indices(
            args.history_tokens,
            args.key_samples_per_text,
            key.device,
        )
        query_indices = evenly_spaced_indices(
            args.history_tokens,
            args.query_samples_per_text,
            query.device,
        )
        group_size = int(metadata["gqa_groups"])
        head_dim = int(metadata["head_dim"])
        for kv_head in range(int(metadata["kv_heads"])):
            key_samples[kv_head].append(
                key[:, kv_head].index_select(0, key_indices).contiguous()
            )
            value_samples[kv_head].append(
                value[:, kv_head].index_select(0, key_indices).contiguous()
            )
            query_samples[kv_head].append(
                query.index_select(0, query_indices)[
                    :,
                    kv_head * group_size : (kv_head + 1) * group_size,
                ].reshape(-1, head_dim)
            )
    if (
        metadata is None
        or key_samples is None
        or query_samples is None
        or value_samples is None
    ):
        raise ValueError("no training texts")

    codebooks: list[dict[str, Any]] = []
    for kv_head, (head_keys, head_queries, head_values) in enumerate(
        zip(key_samples, query_samples, value_samples, strict=True)
    ):
        training_keys = torch.cat(head_keys)
        training_queries = torch.cat(head_queries)
        training_values = torch.cat(head_values)
        metric_query_factor, metric_key_factor, shrinkage = query_metric_factors(
            training_queries, args.metric_shrinkage
        )
        metric_query_mean = training_queries.mean(dim=0) @ metric_query_factor
        metric_training_keys = training_keys @ metric_key_factor
        projection_weights = None
        weighting_diagnostics: dict[str, float] | None = None
        if args.projection_weighting == "value_jacobian":
            per_text_weights = []
            scale = float(metadata["head_dim"] ** -0.5)
            for text_keys, text_queries, text_values in zip(
                head_keys, head_queries, head_values, strict=True
            ):
                per_text_weights.append(
                    value_jacobian_weights(
                        text_keys @ metric_key_factor,
                        text_values,
                        text_queries @ metric_query_factor,
                        scale,
                    )
                )
            projection_weights = torch.cat(per_text_weights)
            weighting_diagnostics = {
                "minimum": float(projection_weights.min()),
                "median": float(projection_weights.median()),
                "p90": float(torch.quantile(projection_weights, 0.90)),
                "p99": float(torch.quantile(projection_weights, 0.99)),
                "maximum": float(projection_weights.max()),
            }
        metric_projection = fit_binary_principal_projection(
            metric_training_keys,
            args.binary_bits,
            args.projection_iterations,
            seed=11000 + kv_head,
            initialization="random",
            sample_weights=projection_weights,
        )
        training_metric_codes, _ = encode_binary_principal(
            metric_training_keys, metric_projection
        )
        training_metric_residual = (
            metric_training_keys.float()
            - training_metric_codes.float() @ metric_projection.float()
        )
        residual_codebook = fit_residual_codebook(
            training_metric_residual,
            clusters=1 << args.residual_vq_bits,
            iterations=args.residual_vq_iterations,
        )
        value_mean = training_values.float().mean(dim=0)
        centered_training_values = training_values.float() - value_mean
        value_scale = (
            centered_training_values.square()
            .sum(dim=-1)
            .mean()
            .sqrt()
            .clamp_min(1.0e-8)
        )
        value_codebook = fit_residual_codebook(
            centered_training_values / value_scale,
            clusters=1 << args.residual_vq_bits,
            iterations=args.residual_vq_iterations,
        )
        joint_residual_codebooks = []
        for value_weight in (
            float(item) for item in args.joint_rvq_weights.split(",") if item
        ):
            joint_model = fit_joint_kv_residual_codebook(
                training_metric_residual,
                training_values,
                clusters=1 << args.residual_vq_bits,
                iterations=args.residual_vq_iterations,
                value_weight=value_weight,
            )
            (
                training_joint_assignments,
                _,
                training_joint_key_centroids,
            ) = encode_joint_kv_residual_codebook(
                training_metric_residual,
                training_values,
                joint_model,
            )
            training_second_residual = (
                training_metric_residual.float()
                - training_joint_key_centroids.float().index_select(
                    0, training_joint_assignments
                )
            )
            joint_model["residual_binary_models"] = [
                {
                    "bits": int(bits),
                    "projection": fit_binary_principal_projection(
                        training_second_residual,
                        int(bits),
                        args.residual_binary_iterations,
                        seed=21000 + 257 * kv_head + int(bits),
                        initialization="random",
                    ),
                }
                for bits in args.residual_binary_bits.split(",")
                if bits
            ]
            joint_model["additive_value_models"] = [
                fit_additive_value_residual_model(
                    training_values,
                    training_joint_assignments,
                    1 << args.residual_vq_bits,
                    int(bits),
                    args.additive_value_iterations,
                )
                for bits in args.additive_value_bits.split(",")
                if bits
            ]
            joint_residual_codebooks.append(joint_model)
        product_residual_codebooks = [
            fit_product_kv_residual_codebook(
                training_metric_residual,
                training_values,
                total_bits=args.residual_vq_bits,
                key_bits=int(key_bits),
                iterations=args.residual_vq_iterations,
            )
            for key_bits in args.product_rvq_key_bits.split(",")
            if key_bits
        ]
        raw_projection = fit_binary_principal_projection(
            training_keys,
            args.binary_bits,
            args.projection_iterations,
            seed=12000 + kv_head,
            initialization="random",
        )
        codebooks.append(
            {
                "metric_query_factor": metric_query_factor,
                "metric_key_factor": metric_key_factor,
                "metric_projection": metric_projection,
                "residual_codebook": residual_codebook,
                "value_codebook": value_codebook,
                "value_mean": value_mean,
                "value_scale": float(value_scale),
                "joint_residual_codebooks": joint_residual_codebooks,
                "product_residual_codebooks": product_residual_codebooks,
                "metric_query_mean": metric_query_mean,
                "raw_projection": raw_projection,
                "resolved_shrinkage": shrinkage,
                "weighting_diagnostics": weighting_diagnostics,
            }
        )
    return codebooks, metadata


def tail_output_and_metrics(
    exact_scores: torch.Tensor,
    full_weights: torch.Tensor,
    full_output: torch.Tensor,
    values: torch.Tensor,
    selected: torch.Tensor,
    metric_coordinates: torch.Tensor,
    metric_query: torch.Tensor,
    raw_coordinates: torch.Tensor,
    raw_query: torch.Tensor,
    coreset: dict[str, Any],
    scale: float,
) -> tuple[dict[str, float], float]:
    reference = exact_scores.index_select(0, selected).amin()
    tail_z, tail_y, diagnostics = block_coreset_tail_statistics(
        metric_coordinates,
        values,
        metric_query * scale,
        selected,
        reference,
        coreset,
        selected_conditioned=False,
        full_score_coordinates=raw_coordinates,
        full_score_direction=raw_query * scale,
    )
    output = combine_selected_and_tail(
        exact_scores,
        exact_scores,
        values,
        selected,
        tail_y,
        tail_z,
        1.0,
    )
    return (
        {
            **output_metrics(output, full_output),
            "selected_mass": float(full_weights.index_select(0, selected).sum()),
            "top1_recall": float(
                torch.isin(torch.argmax(exact_scores)[None], selected)[0]
            ),
        },
        float(diagnostics["bits_per_token"]),
    )


def evaluate_test_text(
    tokenizer: Any,
    text_path: Path,
    codebooks: list[dict[str, Any]],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    offsets = [int(item) for item in args.query_offsets.split(",")]
    adaptive_coverages = [
        float(item) for item in args.adaptive_coverages.split(",") if item
    ]
    cv_samples = [int(item) for item in args.cv_samples.split(",") if item]
    if any(sample_count < 1 for sample_count in cv_samples):
        raise ValueError("control-variate sample counts must be positive")
    tail_calibration_counts = [
        int(item) for item in args.tail_calibration_counts.split(",") if item
    ]
    if any(sample_count < 2 for sample_count in tail_calibration_counts):
        raise ValueError("tail calibration requires at least two samples")
    adaptive_error_tolerances = [
        float(item)
        for item in args.adaptive_error_tolerances.split(",")
        if item
    ]
    if any(
        not 0.0 < tolerance < 1.0
        for tolerance in adaptive_error_tolerances
    ):
        raise ValueError("adaptive error tolerances must lie in (0, 1)")
    additive_block_sizes = [
        int(item) for item in args.additive_block_sizes.split(",") if item
    ]
    if any(block_size < 1 for block_size in additive_block_sizes):
        raise ValueError("additive block sizes must be positive")
    residual_binary_candidate_fractions = [
        float(item)
        for item in args.residual_binary_candidate_fractions.split(",")
        if item
    ]
    if any(
        not 0.0 < fraction <= 1.0
        for fraction in residual_binary_candidate_fractions
    ):
        raise ValueError("residual binary candidate fractions must lie in (0, 1]")
    if any(not 0.0 < coverage < 1.0 for coverage in adaptive_coverages):
        raise ValueError("adaptive coverages must lie strictly between zero and one")
    needed = args.history_tokens + max(offsets) + args.query_tokens
    token_ids = token_ids_for_text(tokenizer, text_path, needed)
    query, key, value, current_metadata = load_layer0_activations(
        args.model, token_ids
    )
    if current_metadata != metadata:
        raise ValueError("test activation metadata differs from training")
    history_count = args.history_tokens
    group_size = int(metadata["gqa_groups"])
    head_dim = int(metadata["head_dim"])
    scale = float(head_dim**-0.5)
    keep = max(1, math.ceil(history_count * args.fraction))
    rows: list[dict[str, Any]] = []

    for kv_head, codebook in enumerate(codebooks):
        head_key = key[:history_count, kv_head].contiguous()
        head_value = value[:history_count, kv_head].contiguous()
        rabitq_index = build_rabitq_index(head_key, seed=17000 + kv_head)
        metric_key_factor = codebook["metric_key_factor"]
        metric_query_factor = codebook["metric_query_factor"]
        metric_projection = codebook["metric_projection"]
        metric_query_mean = codebook["metric_query_mean"]
        residual_codebook = codebook["residual_codebook"]
        value_codebook = codebook["value_codebook"]
        value_mean = codebook["value_mean"]
        value_scale = float(codebook["value_scale"])
        joint_residual_codebooks = codebook["joint_residual_codebooks"]
        product_residual_codebooks = codebook["product_residual_codebooks"]
        raw_projection = codebook["raw_projection"]
        assert isinstance(metric_key_factor, torch.Tensor)
        assert isinstance(metric_query_factor, torch.Tensor)
        assert isinstance(metric_projection, torch.Tensor)
        assert isinstance(metric_query_mean, torch.Tensor)
        assert isinstance(residual_codebook, torch.Tensor)
        assert isinstance(value_codebook, torch.Tensor)
        assert isinstance(value_mean, torch.Tensor)
        assert isinstance(joint_residual_codebooks, list)
        assert isinstance(product_residual_codebooks, list)
        assert isinstance(raw_projection, torch.Tensor)

        metric_coordinates = head_key @ metric_key_factor
        global_metric_codes, global_metric_errors = encode_binary_principal(
            metric_coordinates, metric_projection
        )
        quantized_global_errors, risk_aux_bits = quantize_log_error_norms(
            global_metric_errors,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        metric_reconstruction = (
            global_metric_codes.float() @ metric_projection.float()
        )
        metric_residual = metric_coordinates.float() - metric_reconstruction
        residual_assignments, residual_vq_errors = encode_residual_codebook(
            metric_residual, residual_codebook
        )
        rvq_value_centroids, rvq_tail_bits = fit_rvq_value_centroids(
            head_value,
            residual_assignments,
            residual_codebook.shape[0],
            args.value_mean_bits,
        )
        value_assignments = torch.cdist(
            (head_value.float() - value_mean.float()) / value_scale,
            value_codebook.float(),
        ).argmin(dim=-1)
        dual_value_centroids, dual_tail_bits = fit_rvq_value_centroids(
            head_value,
            value_assignments,
            value_codebook.shape[0],
            args.value_mean_bits,
        )
        quantized_residual_vq_errors, residual_vq_risk_bits = (
            quantize_log_error_norms(
                residual_vq_errors,
                args.risk_error_bits,
                args.risk_error_block_size,
            )
        )
        joint_indexes = []
        joint_block_indexes = []
        joint_binary_indexes = []
        for joint_model in joint_residual_codebooks:
            joint_assignments, joint_errors, joint_key_centroids = (
                encode_joint_kv_residual_codebook(
                    metric_residual,
                    head_value,
                    joint_model,
                )
            )
            quantized_joint_errors, joint_risk_bits = quantize_log_error_norms(
                joint_errors,
                args.risk_error_bits,
                args.risk_error_block_size,
            )
            joint_value_centroids, joint_tail_bits = fit_rvq_value_centroids(
                head_value,
                joint_assignments,
                joint_key_centroids.shape[0],
                args.value_mean_bits,
            )
            refit_key_centroids, refit_key_bits = fit_rvq_value_centroids(
                metric_residual,
                joint_assignments,
                joint_key_centroids.shape[0],
                args.key_mean_bits,
            )
            refit_key_errors = (
                metric_residual.float()
                - refit_key_centroids.float().index_select(0, joint_assignments)
            ).norm(dim=-1)
            quantized_refit_key_errors, refit_key_risk_bits = (
                quantize_log_error_norms(
                    refit_key_errors,
                    args.risk_error_bits,
                    args.risk_error_block_size,
                )
            )
            joint_value_errors = (
                head_value.float()
                - joint_value_centroids.float().index_select(0, joint_assignments)
            ).norm(dim=-1)
            quantized_joint_value_errors, joint_value_risk_bits = (
                quantize_log_error_norms(
                    joint_value_errors,
                    args.risk_error_bits,
                    args.risk_error_block_size,
                )
            )
            additive_value_states = []
            for additive_model in joint_model.get("additive_value_models", []):
                residual_value_assignments = encode_additive_value_residual(
                    head_value,
                    joint_assignments,
                    additive_model,
                )
                residual_bits = int(additive_model["bits"])
                (
                    additive_primary_centroids,
                    additive_residual_centroids,
                    additive_tail_bits,
                ) = fit_additive_request_centroids(
                    head_value,
                    joint_assignments,
                    residual_value_assignments,
                    joint_key_centroids.shape[0],
                    1 << residual_bits,
                    args.value_mean_bits,
                    args.additive_refit_iterations,
                )
                additive_reconstruction = (
                    additive_primary_centroids.float().index_select(
                        0, joint_assignments
                    )
                    + additive_residual_centroids.float().index_select(
                        0, residual_value_assignments
                    )
                )
                additive_value_errors = (
                    head_value.float() - additive_reconstruction
                ).norm(dim=-1)
                (
                    quantized_additive_value_errors,
                    additive_value_risk_bits,
                ) = quantize_log_error_norms(
                    additive_value_errors,
                    args.risk_error_bits,
                    args.risk_error_block_size,
                )
                additive_value_states.append(
                    {
                        "name": (
                            f"addv{residual_bits}i{args.additive_refit_iterations}"
                        ),
                        "residual_assignments": residual_value_assignments,
                        "primary_centroids": additive_primary_centroids,
                        "residual_centroids": additive_residual_centroids,
                        "value_errors": quantized_additive_value_errors,
                        "value_risk_bits": additive_value_risk_bits,
                        "tail_bits": additive_tail_bits,
                        "id_bits": float(residual_bits),
                    }
                )
            value_weight = float(joint_model["value_weight"])
            weight_name = f"{value_weight:.3g}".replace(".", "p")
            second_residual = (
                metric_residual.float()
                - refit_key_centroids.float().index_select(
                    0, joint_assignments
                )
            )
            for residual_binary_model in joint_model.get(
                "residual_binary_models", []
            ):
                residual_projection = residual_binary_model["projection"]
                if not isinstance(residual_projection, torch.Tensor):
                    raise TypeError("residual binary projection is malformed")
                residual_binary_bits = int(residual_binary_model["bits"])
                residual_codes, residual_errors = encode_binary_principal(
                    second_residual,
                    residual_projection,
                )
                quantized_residual_errors, residual_binary_risk_bits = (
                    quantize_log_error_norms(
                        residual_errors,
                        args.risk_error_bits,
                        args.risk_error_block_size,
                    )
                )
                joint_binary_indexes.append(
                    {
                        "name": (
                            f"global_qmetric_jointrvq{args.residual_vq_bits}"
                            f"_w{weight_name}_refitk{args.key_mean_bits}_"
                            f"binres{residual_binary_bits}_risk1"
                        ),
                        "assignments": joint_assignments,
                        "residual_codes": residual_codes,
                        "residual_projection": residual_projection,
                        "errors": quantized_residual_errors,
                        "risk_bits": residual_binary_risk_bits,
                        "base_errors": quantized_refit_key_errors,
                        "base_risk_bits": refit_key_risk_bits,
                        "key_centroids": refit_key_centroids,
                        "value_centroids": joint_value_centroids,
                        "value_errors": quantized_joint_value_errors,
                        "value_risk_bits": joint_value_risk_bits,
                        "tail_bits": joint_tail_bits,
                        "fixed_key_bits": refit_key_bits,
                        "residual_binary_bits": float(residual_binary_bits),
                    }
                )
            for additive_block_size in additive_block_sizes:
                block_assignments = (
                    torch.arange(history_count, device=head_value.device)
                    // additive_block_size
                )
                block_clusters = int(block_assignments[-1]) + 1
                (
                    block_key_primary,
                    block_key_residual,
                    block_key_bits,
                ) = fit_additive_request_centroids(
                    metric_residual,
                    joint_assignments,
                    block_assignments,
                    joint_key_centroids.shape[0],
                    block_clusters,
                    args.key_mean_bits,
                    args.additive_refit_iterations,
                )
                block_key_reconstruction = (
                    block_key_primary.float().index_select(
                        0, joint_assignments
                    )
                    + block_key_residual.float().index_select(
                        0, block_assignments
                    )
                )
                block_key_errors = (
                    metric_residual.float() - block_key_reconstruction
                ).norm(dim=-1)
                quantized_block_key_errors, block_key_risk_bits = (
                    quantize_log_error_norms(
                        block_key_errors,
                        args.risk_error_bits,
                        args.risk_error_block_size,
                    )
                )
                (
                    block_value_primary,
                    block_value_residual,
                    block_value_bits,
                ) = fit_additive_request_centroids(
                    head_value,
                    joint_assignments,
                    block_assignments,
                    joint_key_centroids.shape[0],
                    block_clusters,
                    args.value_mean_bits,
                    args.additive_refit_iterations,
                )
                block_value_reconstruction = (
                    block_value_primary.float().index_select(
                        0, joint_assignments
                    )
                    + block_value_residual.float().index_select(
                        0, block_assignments
                    )
                )
                block_value_errors = (
                    head_value.float() - block_value_reconstruction
                ).norm(dim=-1)
                quantized_block_value_errors, block_value_risk_bits = (
                    quantize_log_error_norms(
                        block_value_errors,
                        args.risk_error_bits,
                        args.risk_error_block_size,
                    )
                )
                joint_block_indexes.append(
                    {
                        "name": (
                            f"global_qmetric_jointrvq{args.residual_vq_bits}"
                            f"_w{weight_name}_block{additive_block_size}_risk1"
                        ),
                        "assignments": joint_assignments,
                        "block_assignments": block_assignments,
                        "errors": quantized_block_key_errors,
                        "risk_bits": block_key_risk_bits,
                        "key_centroids": block_key_primary,
                        "block_key_centroids": block_key_residual,
                        "value_centroids": block_value_primary,
                        "block_value_centroids": block_value_residual,
                        "value_errors": quantized_block_value_errors,
                        "value_risk_bits": block_value_risk_bits,
                        "fixed_key_bits": block_key_bits,
                        "tail_bits": block_value_bits,
                    }
                )
            joint_indexes.append(
                {
                    "name": (
                        f"global_qmetric_jointrvq{args.residual_vq_bits}"
                        f"_w{weight_name}_risk1"
                    ),
                    "assignments": joint_assignments,
                    "errors": quantized_joint_errors,
                    "risk_bits": joint_risk_bits,
                    "key_centroids": joint_key_centroids,
                    "value_centroids": joint_value_centroids,
                    "value_errors": quantized_joint_value_errors,
                    "value_risk_bits": joint_value_risk_bits,
                    "tail_bits": joint_tail_bits,
                    "fixed_key_bits": 0.0,
                    "additive_value_states": additive_value_states,
                }
            )
            joint_indexes.append(
                {
                    "name": (
                        f"global_qmetric_jointrvq{args.residual_vq_bits}"
                        f"_w{weight_name}_refitk{args.key_mean_bits}_risk1"
                    ),
                    "assignments": joint_assignments,
                    "errors": quantized_refit_key_errors,
                    "risk_bits": refit_key_risk_bits,
                    "key_centroids": refit_key_centroids,
                    "value_centroids": joint_value_centroids,
                    "value_errors": quantized_joint_value_errors,
                    "value_risk_bits": joint_value_risk_bits,
                    "tail_bits": joint_tail_bits,
                    "fixed_key_bits": refit_key_bits,
                    "additive_value_states": additive_value_states,
                }
            )
        product_indexes = []
        for product_model in product_residual_codebooks:
            product_assignments, product_errors, product_key_centroids = (
                encode_product_kv_residual_codebook(
                    metric_residual,
                    head_value,
                    product_model,
                )
            )
            quantized_product_errors, product_risk_bits = quantize_log_error_norms(
                product_errors,
                args.risk_error_bits,
                args.risk_error_block_size,
            )
            product_value_centroids, product_tail_bits = fit_rvq_value_centroids(
                head_value,
                product_assignments,
                1 << args.residual_vq_bits,
                args.value_mean_bits,
            )
            key_bits = int(product_model["key_bits"])
            value_bits = int(product_model["value_bits"])
            product_indexes.append(
                {
                    "name": (
                        f"global_qmetric_productrvq{args.residual_vq_bits}"
                        f"_k{key_bits}v{value_bits}_risk1"
                    ),
                    "assignments": product_assignments,
                    "errors": quantized_product_errors,
                    "risk_bits": product_risk_bits,
                    "key_centroids": product_key_centroids,
                    "value_centroids": product_value_centroids,
                    "tail_bits": product_tail_bits,
                    "value_bits": value_bits,
                }
            )
        mean_residual_bias = metric_residual @ metric_query_mean.float() * scale
        quantized_mean_residual_bias, mean_bias_aux_bits = (
            quantize_blockwise_affine(
                mean_residual_bias,
                args.risk_error_bits,
                args.risk_error_block_size,
            )
        )
        radial_scale = (
            (metric_coordinates.float() * metric_reconstruction).sum(dim=-1)
            / metric_reconstruction.square().sum(dim=-1).clamp_min(1.0e-12)
        )
        quantized_radial_scale, radial_aux_bits = quantize_blockwise_affine(
            radial_scale,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        corrected_metric_residual = (
            metric_coordinates.float()
            - quantized_radial_scale[:, None] * metric_reconstruction
        ).norm(dim=-1)
        quantized_corrected_errors, corrected_risk_aux_bits = (
            quantize_log_error_norms(
                corrected_metric_residual,
                args.risk_error_bits,
                args.risk_error_block_size,
            )
        )
        raw_codes, raw_errors = encode_binary_principal(head_key, raw_projection)
        eas_count = min(keep, max(1, math.ceil(keep * args.eas_ratio)))
        raw_eas_indices = torch.topk(raw_errors, eas_count, sorted=False).indices
        metric_eas_indices = torch.topk(
            global_metric_errors, eas_count, sorted=False
        ).indices
        eas_index_bits = (
            eas_count * math.ceil(math.log2(max(2, history_count))) / history_count
        )

        local_calibration = query[
            history_count - args.calibration_tokens : history_count,
            kv_head * group_size : (kv_head + 1) * group_size,
        ].reshape(-1, head_dim)
        local_query_factor, local_key_factor, _ = query_metric_factors(
            local_calibration, args.metric_shrinkage
        )
        local_coordinates = head_key @ local_key_factor
        local_indices = evenly_spaced_indices(
            history_count, args.local_key_sample_count, head_key.device
        )
        local_projection = fit_binary_principal_projection(
            local_coordinates.index_select(0, local_indices),
            args.binary_bits,
            args.projection_iterations,
            seed=13000 + kv_head,
            initialization="random",
        )
        local_codes, local_errors = encode_binary_principal(
            local_coordinates, local_projection
        )
        quantized_local_errors, _ = quantize_log_error_norms(
            local_errors,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        coreset = fit_block_coreset(
            metric_coordinates,
            head_value,
            args.block_size,
            cluster_count=1,
            moment_bits=2,
            iterations=1,
            full_score_coordinates=head_key,
            value_moment_bits=args.value_mean_bits,
            full_score_moment_bits=args.key_mean_bits,
        )
        block_ids = torch.arange(history_count, device=head_value.device).div(
            args.block_size, rounding_mode="floor"
        )
        mean_v = coreset["mean_v"]
        assert isinstance(mean_v, torch.Tensor)
        block_value_tail_bits = (
            mean_v.shape[0]
            * (args.value_mean_bits * head_dim + 16)
            / history_count
        )
        value_deviation = (
            head_value.float() - mean_v[block_ids, 0].float()
        ).norm(dim=-1)
        quantized_value_deviation, value_aux_bits = quantize_log_error_norms(
            value_deviation,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        log_value_deviation = quantized_value_deviation.clamp_min(1.0e-8).log()

        for query_offset in offsets:
            heldout = query[
                history_count
                + query_offset : history_count
                + query_offset
                + args.query_tokens
            ]
            for token_offset in range(args.query_tokens):
                for group_offset in range(group_size):
                    query_head = kv_head * group_size + group_offset
                    current_query = heldout[token_offset, query_head]
                    request_query_centroid = query[:history_count, query_head].mean(
                        dim=0
                    )
                    exact_scores = head_key @ current_query * scale
                    full_weights = torch.softmax(exact_scores, dim=0)
                    full_output = full_weights @ head_value
                    global_metric_query = current_query @ metric_query_factor
                    local_metric_query = current_query @ local_query_factor
                    global_proxy = binary_proxy_scores(
                        global_metric_codes,
                        metric_projection,
                        global_metric_query,
                        scale,
                    )
                    residual_vq_table = residual_codebook.float() @ global_metric_query * scale
                    residual_vq_proxy = global_proxy + residual_vq_table.index_select(
                        0, residual_assignments
                    )
                    residual_vq_uncertainty = (
                        quantized_residual_vq_errors
                        * global_metric_query.norm()
                        / float(head_dim)
                    )
                    joint_query_states = []
                    for joint_index in joint_indexes:
                        joint_key_centroids = joint_index["key_centroids"]
                        joint_assignments = joint_index["assignments"]
                        joint_errors = joint_index["errors"]
                        joint_value_errors = joint_index["value_errors"]
                        joint_value_centroids = joint_index["value_centroids"]
                        assert isinstance(joint_key_centroids, torch.Tensor)
                        assert isinstance(joint_assignments, torch.Tensor)
                        assert isinstance(joint_errors, torch.Tensor)
                        assert isinstance(joint_value_errors, torch.Tensor)
                        assert isinstance(joint_value_centroids, torch.Tensor)
                        joint_score_table = (
                            joint_key_centroids.float() @ global_metric_query * scale
                        )
                        joint_proxy = global_proxy + joint_score_table.index_select(
                            0, joint_assignments
                        )
                        joint_uncertainty = (
                            joint_errors * global_metric_query.norm() / float(head_dim)
                        )
                        centroid_norms = joint_value_centroids.float().norm(dim=-1)
                        value_sensitivity = (
                            joint_value_errors
                            + joint_uncertainty
                            * (
                                centroid_norms.index_select(0, joint_assignments)
                                + joint_value_errors
                            )
                        ).clamp_min(1.0e-8)
                        joint_query_states.append(
                            {
                                **joint_index,
                                "proxy": joint_proxy,
                                "uncertainty": joint_uncertainty,
                                "priority": (
                                    joint_proxy
                                    + args.risk_lambda * joint_uncertainty
                                ),
                                "output_priority": (
                                    joint_proxy + value_sensitivity.log()
                                ),
                            }
                        )
                    joint_binary_query_states = []
                    for binary_index in joint_binary_indexes:
                        binary_key_centroids = binary_index["key_centroids"]
                        binary_assignments = binary_index["assignments"]
                        residual_codes = binary_index["residual_codes"]
                        residual_projection = binary_index[
                            "residual_projection"
                        ]
                        binary_errors = binary_index["errors"]
                        base_errors = binary_index["base_errors"]
                        binary_value_errors = binary_index["value_errors"]
                        binary_value_centroids = binary_index[
                            "value_centroids"
                        ]
                        assert isinstance(binary_key_centroids, torch.Tensor)
                        assert isinstance(binary_assignments, torch.Tensor)
                        assert isinstance(residual_codes, torch.Tensor)
                        assert isinstance(residual_projection, torch.Tensor)
                        assert isinstance(binary_errors, torch.Tensor)
                        assert isinstance(base_errors, torch.Tensor)
                        assert isinstance(binary_value_errors, torch.Tensor)
                        assert isinstance(binary_value_centroids, torch.Tensor)
                        binary_key_table = (
                            binary_key_centroids.float()
                            @ global_metric_query
                            * scale
                        )
                        residual_binary_score = binary_proxy_scores(
                            residual_codes,
                            residual_projection,
                            global_metric_query,
                            scale,
                        )
                        base_joint_proxy = (
                            global_proxy
                            + binary_key_table.index_select(
                                0, binary_assignments
                            )
                        )
                        binary_proxy = base_joint_proxy + residual_binary_score
                        base_uncertainty = (
                            base_errors
                            * global_metric_query.norm()
                            / float(head_dim)
                        )
                        binary_uncertainty = (
                            binary_errors
                            * global_metric_query.norm()
                            / float(head_dim)
                        )
                        centroid_norms = binary_value_centroids.float().norm(
                            dim=-1
                        )
                        value_sensitivity = (
                            binary_value_errors
                            + binary_uncertainty
                            * (
                                centroid_norms.index_select(
                                    0, binary_assignments
                                )
                                + binary_value_errors
                            )
                        ).clamp_min(1.0e-8)
                        base_value_sensitivity = (
                            binary_value_errors
                            + base_uncertainty
                            * (
                                centroid_norms.index_select(
                                    0, binary_assignments
                                )
                                + binary_value_errors
                            )
                        ).clamp_min(1.0e-8)
                        joint_binary_query_states.append(
                            {
                                **binary_index,
                                "proxy": binary_proxy,
                                "base_proxy": base_joint_proxy,
                                "uncertainty": binary_uncertainty,
                                "base_uncertainty": base_uncertainty,
                                "priority": (
                                    binary_proxy
                                    + args.risk_lambda * binary_uncertainty
                                ),
                                "output_priority": (
                                    binary_proxy + value_sensitivity.log()
                                ),
                                "base_output_priority": (
                                    base_joint_proxy
                                    + base_value_sensitivity.log()
                                ),
                            }
                        )
                    joint_block_query_states = []
                    for block_index in joint_block_indexes:
                        primary_key_centroids = block_index["key_centroids"]
                        block_key_centroids = block_index[
                            "block_key_centroids"
                        ]
                        primary_assignments = block_index["assignments"]
                        block_assignments = block_index["block_assignments"]
                        key_errors = block_index["errors"]
                        value_errors = block_index["value_errors"]
                        primary_value_centroids = block_index[
                            "value_centroids"
                        ]
                        block_value_centroids = block_index[
                            "block_value_centroids"
                        ]
                        assert isinstance(primary_key_centroids, torch.Tensor)
                        assert isinstance(block_key_centroids, torch.Tensor)
                        assert isinstance(primary_assignments, torch.Tensor)
                        assert isinstance(block_assignments, torch.Tensor)
                        assert isinstance(key_errors, torch.Tensor)
                        assert isinstance(value_errors, torch.Tensor)
                        assert isinstance(primary_value_centroids, torch.Tensor)
                        assert isinstance(block_value_centroids, torch.Tensor)
                        primary_score_table = (
                            primary_key_centroids.float()
                            @ global_metric_query
                            * scale
                        )
                        block_score_table = (
                            block_key_centroids.float()
                            @ global_metric_query
                            * scale
                        )
                        block_proxy = (
                            global_proxy
                            + primary_score_table.index_select(
                                0, primary_assignments
                            )
                            + block_score_table.index_select(
                                0, block_assignments
                            )
                        )
                        block_uncertainty = (
                            key_errors * global_metric_query.norm() / float(head_dim)
                        )
                        approximate_values = (
                            primary_value_centroids.float().index_select(
                                0, primary_assignments
                            )
                            + block_value_centroids.float().index_select(
                                0, block_assignments
                            )
                        )
                        approximate_value_norm = approximate_values.norm(dim=-1)
                        value_sensitivity = (
                            value_errors
                            + block_uncertainty
                            * (approximate_value_norm + value_errors)
                        ).clamp_min(1.0e-8)
                        joint_block_query_states.append(
                            {
                                **block_index,
                                "proxy": block_proxy,
                                "uncertainty": block_uncertainty,
                                "priority": (
                                    block_proxy
                                    + args.risk_lambda * block_uncertainty
                                ),
                                "output_priority": (
                                    block_proxy + value_sensitivity.log()
                                ),
                                "approximate_values": approximate_values,
                            }
                        )
                    product_query_states = []
                    for product_index in product_indexes:
                        product_key_centroids = product_index["key_centroids"]
                        product_assignments = product_index["assignments"]
                        product_errors = product_index["errors"]
                        assert isinstance(product_key_centroids, torch.Tensor)
                        assert isinstance(product_assignments, torch.Tensor)
                        assert isinstance(product_errors, torch.Tensor)
                        product_score_table = (
                            product_key_centroids.float()
                            @ global_metric_query
                            * scale
                        )
                        product_key_assignments = product_assignments >> int(
                            product_index["value_bits"]
                        )
                        product_proxy = global_proxy + product_score_table.index_select(
                            0, product_key_assignments
                        )
                        product_uncertainty = (
                            product_errors
                            * global_metric_query.norm()
                            / float(head_dim)
                        )
                        product_query_states.append(
                            {
                                **product_index,
                                "proxy": product_proxy,
                                "uncertainty": product_uncertainty,
                                "priority": (
                                    product_proxy
                                    + args.risk_lambda * product_uncertainty
                                ),
                            }
                        )
                    local_proxy = binary_proxy_scores(
                        local_codes,
                        local_projection,
                        local_metric_query,
                        scale,
                    )
                    rabitq_proxy = rabitq_proxy_scores(
                        rabitq_index,
                        head_key,
                        current_query,
                        request_query_centroid,
                        scale,
                    )
                    global_uncertainty = (
                        quantized_global_errors
                        * global_metric_query.norm()
                        / float(head_dim)
                    )
                    mean_corrected_global_proxy = (
                        global_proxy + quantized_mean_residual_bias
                    )
                    centered_global_uncertainty = (
                        quantized_global_errors
                        * (global_metric_query - metric_query_mean).norm()
                        / float(head_dim)
                    )
                    scaled_global_proxy = global_proxy * quantized_radial_scale
                    scaled_global_uncertainty = (
                        quantized_corrected_errors
                        * global_metric_query.norm()
                        / float(head_dim)
                    )
                    local_uncertainty = (
                        quantized_local_errors
                        * local_metric_query.norm()
                        / float(head_dim)
                    )
                    selectors = {
                        "rabitq_fp_query_topk": (
                            rabitq_proxy,
                            metric_coordinates,
                            global_metric_query,
                            176.0 - float(args.binary_bits),
                        ),
                        "global_keymse": (
                            binary_proxy_scores(
                                raw_codes, raw_projection, current_query, scale
                            ),
                            metric_coordinates,
                            global_metric_query,
                            0.0,
                        ),
                        "global_qmetric": (
                            global_proxy,
                            metric_coordinates,
                            global_metric_query,
                            0.0,
                        ),
                        "global_keymse_eas10": (
                            binary_proxy_scores(
                                raw_codes, raw_projection, current_query, scale
                            ).scatter(0, raw_eas_indices, float("inf")),
                            metric_coordinates,
                            global_metric_query,
                            eas_index_bits,
                        ),
                        "global_qmetric_eas10": (
                            global_proxy.clone().scatter(
                                0, metric_eas_indices, float("inf")
                            ),
                            metric_coordinates,
                            global_metric_query,
                            eas_index_bits,
                        ),
                        "global_qmetric_risk1": (
                            global_proxy + args.risk_lambda * global_uncertainty,
                            metric_coordinates,
                            global_metric_query,
                            risk_aux_bits,
                        ),
                        "global_qmetric_value": (
                            global_proxy + log_value_deviation,
                            metric_coordinates,
                            global_metric_query,
                            value_aux_bits,
                        ),
                        "global_qmetric_output_risk1": (
                            global_proxy
                            + args.risk_lambda * global_uncertainty
                            + log_value_deviation,
                            metric_coordinates,
                            global_metric_query,
                            risk_aux_bits + value_aux_bits,
                        ),
                        "global_qmetric_radial4": (
                            scaled_global_proxy,
                            metric_coordinates,
                            global_metric_query,
                            radial_aux_bits,
                        ),
                        "global_qmetric_radial4_risk1": (
                            scaled_global_proxy
                            + args.risk_lambda * scaled_global_uncertainty,
                            metric_coordinates,
                            global_metric_query,
                            radial_aux_bits + corrected_risk_aux_bits,
                        ),
                        "global_qmetric_radial4_output_risk1": (
                            scaled_global_proxy
                            + args.risk_lambda * scaled_global_uncertainty
                            + log_value_deviation,
                            metric_coordinates,
                            global_metric_query,
                            radial_aux_bits
                            + corrected_risk_aux_bits
                            + value_aux_bits,
                        ),
                        "global_qmetric_mean4": (
                            mean_corrected_global_proxy,
                            metric_coordinates,
                            global_metric_query,
                            mean_bias_aux_bits,
                        ),
                        "global_qmetric_mean4_risk1": (
                            mean_corrected_global_proxy
                            + args.risk_lambda * centered_global_uncertainty,
                            metric_coordinates,
                            global_metric_query,
                            mean_bias_aux_bits + risk_aux_bits,
                        ),
                        "global_qmetric_mean4_output_risk1": (
                            mean_corrected_global_proxy
                            + args.risk_lambda * centered_global_uncertainty
                            + log_value_deviation,
                            metric_coordinates,
                            global_metric_query,
                            mean_bias_aux_bits + risk_aux_bits + value_aux_bits,
                        ),
                        f"global_qmetric_rvq{args.residual_vq_bits}": (
                            residual_vq_proxy,
                            metric_coordinates,
                            global_metric_query,
                            float(args.residual_vq_bits),
                        ),
                        f"global_qmetric_rvq{args.residual_vq_bits}_risk1": (
                            residual_vq_proxy
                            + args.risk_lambda * residual_vq_uncertainty,
                            metric_coordinates,
                            global_metric_query,
                            float(args.residual_vq_bits) + residual_vq_risk_bits,
                        ),
                        f"global_qmetric_rvq{args.residual_vq_bits}_output_risk1": (
                            residual_vq_proxy
                            + args.risk_lambda * residual_vq_uncertainty
                            + log_value_deviation,
                            metric_coordinates,
                            global_metric_query,
                            float(args.residual_vq_bits)
                            + residual_vq_risk_bits
                            + value_aux_bits,
                        ),
                        "local_qmetric": (
                            local_proxy,
                            local_coordinates,
                            local_metric_query,
                            0.0,
                        ),
                        "local_qmetric_risk1": (
                            local_proxy + args.risk_lambda * local_uncertainty,
                            local_coordinates,
                            local_metric_query,
                            risk_aux_bits,
                        ),
                        "exact_qk_oracle": (
                            exact_scores,
                            metric_coordinates,
                            global_metric_query,
                            0.0,
                        ),
                        "exact_output_oracle": (
                            exact_scores + value_deviation.clamp_min(1.0e-8).log(),
                            metric_coordinates,
                            global_metric_query,
                            0.0,
                        ),
                    }
                    for joint_state in joint_query_states:
                        selectors[str(joint_state["name"])] = (
                            joint_state["priority"],
                            metric_coordinates,
                            global_metric_query,
                            float(args.residual_vq_bits)
                            + float(joint_state["risk_bits"])
                            + float(joint_state["fixed_key_bits"]),
                        )
                    for binary_state in joint_binary_query_states:
                        binary_proxy = binary_state["proxy"]
                        base_proxy = binary_state["base_proxy"]
                        binary_uncertainty = binary_state["uncertainty"]
                        binary_output_priority = binary_state["output_priority"]
                        base_output_priority = binary_state[
                            "base_output_priority"
                        ]
                        binary_assignments = binary_state["assignments"]
                        binary_value_centroids = binary_state[
                            "value_centroids"
                        ]
                        binary_value_errors = binary_state["value_errors"]
                        assert isinstance(binary_proxy, torch.Tensor)
                        assert isinstance(base_proxy, torch.Tensor)
                        assert isinstance(binary_uncertainty, torch.Tensor)
                        assert isinstance(binary_output_priority, torch.Tensor)
                        assert isinstance(base_output_priority, torch.Tensor)
                        assert isinstance(binary_assignments, torch.Tensor)
                        assert isinstance(binary_value_centroids, torch.Tensor)
                        assert isinstance(binary_value_errors, torch.Tensor)
                        binary_selected = torch.topk(
                            binary_output_priority, keep, sorted=False
                        ).indices
                        for tail_name, tail_proxy in (
                            ("raw", binary_proxy),
                            ("exactmass_oracle", exact_scores),
                        ):
                            binary_output = rvq_tail_output(
                                exact_scores,
                                tail_proxy,
                                head_value,
                                binary_selected,
                                binary_assignments,
                                binary_value_centroids,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{binary_state['name']}_boundrisk_"
                                        f"sharedtail_{tail_name}"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(
                                            binary_state[
                                                "residual_binary_bits"
                                            ]
                                        )
                                        + float(binary_state["risk_bits"])
                                        + float(binary_state["value_risk_bits"])
                                        + float(binary_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        binary_state["tail_bits"]
                                    ),
                                    **output_metrics(binary_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, binary_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            binary_selected,
                                        )[0]
                                    ),
                                }
                            )
                        for candidate_fraction in (
                            residual_binary_candidate_fractions
                        ):
                            candidate_count = min(
                                history_count,
                                max(
                                    keep,
                                    math.ceil(
                                        history_count * candidate_fraction
                                    ),
                                ),
                            )
                            candidates = torch.topk(
                                base_output_priority,
                                candidate_count,
                                sorted=False,
                            ).indices
                            selected_within = torch.topk(
                                binary_output_priority.index_select(
                                    0, candidates
                                ),
                                keep,
                                sorted=False,
                            ).indices
                            progressive_selected = candidates.index_select(
                                0, selected_within
                            )
                            mixed_proxy = base_proxy.clone()
                            mixed_proxy[candidates] = binary_proxy.index_select(
                                0, candidates
                            )
                            progressive_output = rvq_tail_output(
                                exact_scores,
                                mixed_proxy,
                                head_value,
                                progressive_selected,
                                binary_assignments,
                                binary_value_centroids,
                            )
                            fraction_name = int(
                                round(1000 * candidate_fraction)
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{binary_state['name']}_progressive_"
                                        f"cand{fraction_name:03d}_sharedtail_raw"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "candidate_tokens": candidate_count,
                                    "candidate_fraction": (
                                        candidate_count / history_count
                                    ),
                                    "scan_bits_per_token": (
                                        args.binary_bits
                                        + float(
                                            binary_state[
                                                "residual_binary_bits"
                                            ]
                                        )
                                        * candidate_count
                                        / history_count
                                    ),
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(
                                            binary_state[
                                                "residual_binary_bits"
                                            ]
                                        )
                                        + float(binary_state["risk_bits"])
                                        + float(binary_state["value_risk_bits"])
                                        + float(binary_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        binary_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        progressive_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, progressive_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            progressive_selected,
                                        )[0]
                                    ),
                                }
                            )
                        approximate_values = (
                            binary_value_centroids.float().index_select(
                                0, binary_assignments
                            )
                        )
                        for tolerance in adaptive_error_tolerances:
                            adaptive_selected, adaptive_diagnostics = (
                                select_by_output_rms_bound(
                                    binary_proxy,
                                    binary_uncertainty,
                                    approximate_values,
                                    binary_value_errors,
                                    tolerance,
                                )
                            )
                            adaptive_output = rvq_tail_output(
                                exact_scores,
                                binary_proxy,
                                head_value,
                                adaptive_selected,
                                binary_assignments,
                                binary_value_centroids,
                            )
                            tolerance_name = int(round(1000 * tolerance))
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{binary_state['name']}_adaptive_"
                                        f"rmstol{tolerance_name:03d}_sharedtail"
                                    ),
                                    "selected_tokens": int(
                                        adaptive_selected.numel()
                                    ),
                                    "selected_fraction": (
                                        adaptive_selected.numel() / history_count
                                    ),
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(
                                            binary_state[
                                                "residual_binary_bits"
                                            ]
                                        )
                                        + float(binary_state["risk_bits"])
                                        + float(binary_state["value_risk_bits"])
                                        + float(binary_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        binary_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        adaptive_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, adaptive_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            adaptive_selected,
                                        )[0]
                                    ),
                                    "adaptive_diagnostics": adaptive_diagnostics,
                                }
                            )
                    for block_state in joint_block_query_states:
                        block_proxy = block_state["proxy"]
                        block_uncertainty = block_state["uncertainty"]
                        block_output_priority = block_state["output_priority"]
                        primary_assignments = block_state["assignments"]
                        block_assignments = block_state["block_assignments"]
                        primary_value_centroids = block_state[
                            "value_centroids"
                        ]
                        block_value_centroids = block_state[
                            "block_value_centroids"
                        ]
                        approximate_values = block_state[
                            "approximate_values"
                        ]
                        value_errors = block_state["value_errors"]
                        assert isinstance(block_proxy, torch.Tensor)
                        assert isinstance(block_uncertainty, torch.Tensor)
                        assert isinstance(block_output_priority, torch.Tensor)
                        assert isinstance(primary_assignments, torch.Tensor)
                        assert isinstance(block_assignments, torch.Tensor)
                        assert isinstance(primary_value_centroids, torch.Tensor)
                        assert isinstance(block_value_centroids, torch.Tensor)
                        assert isinstance(approximate_values, torch.Tensor)
                        assert isinstance(value_errors, torch.Tensor)
                        block_selected = torch.topk(
                            block_output_priority, keep, sorted=False
                        ).indices
                        for tail_name, tail_proxy in (
                            ("raw", block_proxy),
                            ("exactmass_oracle", exact_scores),
                        ):
                            block_output = additive_rvq_tail_output(
                                exact_scores,
                                tail_proxy,
                                head_value,
                                block_selected,
                                primary_assignments,
                                primary_value_centroids,
                                block_assignments,
                                block_value_centroids,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{block_state['name']}_boundrisk_"
                                        f"additiveblocktail_{tail_name}"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(block_state["risk_bits"])
                                        + float(block_state["value_risk_bits"])
                                        + float(block_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        block_state["tail_bits"]
                                    ),
                                    **output_metrics(block_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, block_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            block_selected,
                                        )[0]
                                    ),
                                }
                            )
                        for tolerance in adaptive_error_tolerances:
                            adaptive_selected, adaptive_diagnostics = (
                                select_by_output_rms_bound(
                                    block_proxy,
                                    block_uncertainty,
                                    approximate_values,
                                    value_errors,
                                    tolerance,
                                )
                            )
                            adaptive_output = additive_rvq_tail_output(
                                exact_scores,
                                block_proxy,
                                head_value,
                                adaptive_selected,
                                primary_assignments,
                                primary_value_centroids,
                                block_assignments,
                                block_value_centroids,
                            )
                            tolerance_name = int(round(1000 * tolerance))
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{block_state['name']}_adaptive_"
                                        f"rmstol{tolerance_name:03d}_"
                                        "additiveblocktail"
                                    ),
                                    "selected_tokens": int(
                                        adaptive_selected.numel()
                                    ),
                                    "selected_fraction": (
                                        adaptive_selected.numel() / history_count
                                    ),
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(block_state["risk_bits"])
                                        + float(block_state["value_risk_bits"])
                                        + float(block_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        block_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        adaptive_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, adaptive_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            adaptive_selected,
                                        )[0]
                                    ),
                                    "adaptive_diagnostics": adaptive_diagnostics,
                                }
                            )
                    for product_state in product_query_states:
                        selectors[str(product_state["name"])] = (
                            product_state["priority"],
                            metric_coordinates,
                            global_metric_query,
                            float(args.residual_vq_bits)
                            + float(product_state["risk_bits"]),
                        )
                    for selector, (
                        priority,
                        selector_coordinates,
                        selector_query,
                        selector_aux_bits,
                    ) in selectors.items():
                        selected = torch.topk(priority, keep, sorted=False).indices
                        metrics, tail_bits = tail_output_and_metrics(
                            exact_scores,
                            full_weights,
                            full_output,
                            head_value,
                            selected,
                            selector_coordinates,
                            selector_query,
                            head_key,
                            current_query,
                            coreset,
                            scale,
                        )
                        rows.append(
                            {
                                "text": text_path.stem,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "token_offset": token_offset,
                                "selector": selector,
                                "selected_tokens": keep,
                                "selected_fraction": keep / history_count,
                                "selector_bits_per_token": (
                                    args.binary_bits + selector_aux_bits
                                ),
                                "tail_bits_per_token": tail_bits,
                                **metrics,
                            }
                        )
                    rvq_priority = (
                        residual_vq_proxy
                        + args.risk_lambda * residual_vq_uncertainty
                    )
                    rvq_selected = torch.topk(
                        rvq_priority, keep, sorted=False
                    ).indices
                    for calibration_count in (0, 256):
                        rvq_output = rvq_tail_output(
                            exact_scores,
                            residual_vq_proxy,
                            head_value,
                            rvq_selected,
                            residual_assignments,
                            rvq_value_centroids,
                            calibration_count=calibration_count,
                        )
                        rvq_metrics = {
                            **output_metrics(rvq_output, full_output),
                            "selected_mass": float(
                                full_weights.index_select(0, rvq_selected).sum()
                            ),
                            "top1_recall": float(
                                torch.isin(
                                    torch.argmax(exact_scores)[None], rvq_selected
                                )[0]
                            ),
                        }
                        calibration_name = (
                            "raw" if calibration_count == 0 else "sample256"
                        )
                        rows.append(
                            {
                                "text": text_path.stem,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "token_offset": token_offset,
                                "selector": (
                                    f"global_qmetric_rvq{args.residual_vq_bits}"
                                    f"_risk1_rvqtail_{calibration_name}"
                                ),
                                "selected_tokens": keep,
                                "selected_fraction": keep / history_count,
                                "selector_bits_per_token": (
                                    args.binary_bits
                                    + args.residual_vq_bits
                                    + residual_vq_risk_bits
                                ),
                                "tail_bits_per_token": rvq_tail_bits,
                                **rvq_metrics,
                            }
                        )
                        blockmass_output = rvq_tail_output(
                            exact_scores,
                            residual_vq_proxy,
                            head_value,
                            rvq_selected,
                            block_ids,
                            mean_v[:, 0],
                            calibration_count=calibration_count,
                        )
                        blockmass_metrics = {
                            **output_metrics(blockmass_output, full_output),
                            "selected_mass": float(
                                full_weights.index_select(0, rvq_selected).sum()
                            ),
                            "top1_recall": float(
                                torch.isin(
                                    torch.argmax(exact_scores)[None], rvq_selected
                                )[0]
                            ),
                        }
                        rows.append(
                            {
                                "text": text_path.stem,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "token_offset": token_offset,
                                "selector": (
                                    f"global_qmetric_rvq{args.residual_vq_bits}"
                                    f"_risk1_blockmass_{calibration_name}"
                                ),
                                "selected_tokens": keep,
                                "selected_fraction": keep / history_count,
                                "selector_bits_per_token": (
                                    args.binary_bits
                                    + args.residual_vq_bits
                                    + residual_vq_risk_bits
                                ),
                                "tail_bits_per_token": block_value_tail_bits,
                                **blockmass_metrics,
                            }
                        )
                    lognormal_output = rvq_tail_output(
                        exact_scores,
                        residual_vq_proxy
                        + 0.5 * residual_vq_uncertainty.square(),
                        head_value,
                        rvq_selected,
                        residual_assignments,
                        rvq_value_centroids,
                    )
                    rows.append(
                        {
                            "text": text_path.stem,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "query_offset": query_offset,
                            "token_offset": token_offset,
                            "selector": (
                                f"global_qmetric_rvq{args.residual_vq_bits}"
                                "_risk1_rvqtail_lognormal"
                            ),
                            "selected_tokens": keep,
                            "selected_fraction": keep / history_count,
                            "selector_bits_per_token": (
                                args.binary_bits
                                + args.residual_vq_bits
                                + residual_vq_risk_bits
                            ),
                            "tail_bits_per_token": rvq_tail_bits,
                            **output_metrics(lognormal_output, full_output),
                            "selected_mass": float(
                                full_weights.index_select(0, rvq_selected).sum()
                            ),
                            "top1_recall": float(
                                torch.isin(
                                    torch.argmax(exact_scores)[None], rvq_selected
                                )[0]
                            ),
                        }
                    )
                    for tail_name, tail_proxy, tail_assignments, tail_centroids, extra_bits, tail_bits in (
                        (
                            "rvqtail_exactmass_oracle",
                            exact_scores,
                            residual_assignments,
                            rvq_value_centroids,
                            0.0,
                            rvq_tail_bits,
                        ),
                        (
                            "dualvaluetail_raw",
                            residual_vq_proxy,
                            value_assignments,
                            dual_value_centroids,
                            float(args.residual_vq_bits),
                            dual_tail_bits,
                        ),
                        (
                            "dualvaluetail_exactmass_oracle",
                            exact_scores,
                            value_assignments,
                            dual_value_centroids,
                            float(args.residual_vq_bits),
                            dual_tail_bits,
                        ),
                    ):
                        tail_output = rvq_tail_output(
                            exact_scores,
                            tail_proxy,
                            head_value,
                            rvq_selected,
                            tail_assignments,
                            tail_centroids,
                        )
                        rows.append(
                            {
                                "text": text_path.stem,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "token_offset": token_offset,
                                "selector": (
                                    f"global_qmetric_rvq{args.residual_vq_bits}"
                                    f"_risk1_{tail_name}"
                                ),
                                "selected_tokens": keep,
                                "selected_fraction": keep / history_count,
                                "selector_bits_per_token": (
                                    args.binary_bits
                                    + args.residual_vq_bits
                                    + residual_vq_risk_bits
                                    + extra_bits
                                ),
                                "tail_bits_per_token": tail_bits,
                                **output_metrics(tail_output, full_output),
                                "selected_mass": float(
                                    full_weights.index_select(0, rvq_selected).sum()
                                ),
                                "top1_recall": float(
                                    torch.isin(
                                        torch.argmax(exact_scores)[None], rvq_selected
                                    )[0]
                                ),
                            }
                        )
                    for joint_state in joint_query_states:
                        joint_priority = joint_state["priority"]
                        joint_output_priority = joint_state["output_priority"]
                        joint_proxy = joint_state["proxy"]
                        joint_uncertainty = joint_state["uncertainty"]
                        joint_assignments = joint_state["assignments"]
                        joint_value_centroids = joint_state["value_centroids"]
                        assert isinstance(joint_priority, torch.Tensor)
                        assert isinstance(joint_output_priority, torch.Tensor)
                        assert isinstance(joint_proxy, torch.Tensor)
                        assert isinstance(joint_uncertainty, torch.Tensor)
                        assert isinstance(joint_assignments, torch.Tensor)
                        assert isinstance(joint_value_centroids, torch.Tensor)
                        joint_selected = torch.topk(
                            joint_priority, keep, sorted=False
                        ).indices
                        for tail_name, tail_proxy, selected_conditioned in (
                            ("raw", joint_proxy, False),
                            ("conditioned_raw", joint_proxy, True),
                            (
                                "lognormal",
                                joint_proxy + 0.5 * joint_uncertainty.square(),
                                False,
                            ),
                            ("exactmass_oracle", exact_scores, False),
                        ):
                            joint_output = rvq_tail_output(
                                exact_scores,
                                tail_proxy,
                                head_value,
                                joint_selected,
                                joint_assignments,
                                joint_value_centroids,
                                selected_conditioned=selected_conditioned,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_sharedtail_"
                                        f"{tail_name}"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(joint_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, joint_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            joint_selected,
                                        )[0]
                                    ),
                                }
                            )
                        for additive_state in joint_state[
                            "additive_value_states"
                        ]:
                            additive_output = additive_rvq_tail_output(
                                exact_scores,
                                joint_proxy,
                                head_value,
                                joint_selected,
                                joint_assignments,
                                additive_state["primary_centroids"],
                                additive_state["residual_assignments"],
                                additive_state["residual_centroids"],
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_sharedtail_"
                                        f"{additive_state['name']}_raw"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(additive_state["id_bits"])
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        additive_state["tail_bits"]
                                    ),
                                    **output_metrics(additive_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, joint_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            joint_selected,
                                        )[0]
                                    ),
                                }
                            )
                        output_selected = torch.topk(
                            joint_output_priority, keep, sorted=False
                        ).indices
                        for tail_name, tail_proxy, selected_conditioned in (
                            ("raw", joint_proxy, False),
                            ("conditioned_raw", joint_proxy, True),
                            ("exactmass_oracle", exact_scores, False),
                        ):
                            output_risk_output = rvq_tail_output(
                                exact_scores,
                                tail_proxy,
                                head_value,
                                output_selected,
                                joint_assignments,
                                joint_value_centroids,
                                selected_conditioned=selected_conditioned,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_boundrisk_"
                                        f"sharedtail_{tail_name}"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["value_risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        output_risk_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, output_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            output_selected,
                                        )[0]
                                    ),
                                }
                            )
                        for calibration_count in tail_calibration_counts:
                            calibrated_output = rvq_tail_output(
                                exact_scores,
                                joint_proxy,
                                head_value,
                                output_selected,
                                joint_assignments,
                                joint_value_centroids,
                                calibration_count=calibration_count,
                            )
                            calibration_indices = evenly_spaced_indices(
                                history_count,
                                calibration_count,
                                exact_scores.device,
                            )
                            calibration_overlap = int(
                                torch.isin(
                                    calibration_indices, output_selected
                                ).sum()
                            )
                            exact_reads = (
                                keep + calibration_count - calibration_overlap
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_boundrisk_"
                                        f"sharedtail_affine{calibration_count}"
                                    ),
                                    "selected_tokens": exact_reads,
                                    "selected_fraction": exact_reads / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["value_risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        calibrated_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, output_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            output_selected,
                                        )[0]
                                    ),
                                }
                            )
                        joint_approximate_values = (
                            joint_value_centroids.float().index_select(
                                0, joint_assignments
                            )
                        )
                        for tolerance in adaptive_error_tolerances:
                            adaptive_selected, adaptive_diagnostics = (
                                select_by_output_rms_bound(
                                    joint_proxy,
                                    joint_uncertainty,
                                    joint_approximate_values,
                                    joint_state["value_errors"],
                                    tolerance,
                                )
                            )
                            adaptive_output = rvq_tail_output(
                                exact_scores,
                                joint_proxy,
                                head_value,
                                adaptive_selected,
                                joint_assignments,
                                joint_value_centroids,
                            )
                            tolerance_name = int(round(1000 * tolerance))
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_adaptive_"
                                        f"rmstol{tolerance_name:03d}_sharedtail"
                                    ),
                                    "selected_tokens": int(
                                        adaptive_selected.numel()
                                    ),
                                    "selected_fraction": (
                                        adaptive_selected.numel() / history_count
                                    ),
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["value_risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(
                                        adaptive_output, full_output
                                    ),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, adaptive_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            adaptive_selected,
                                        )[0]
                                    ),
                                    "adaptive_diagnostics": adaptive_diagnostics,
                                }
                            )
                        phase = (
                            (
                                131 * kv_head
                                + 17 * query_head
                                + 7 * query_offset
                                + token_offset
                            )
                            % 997
                            + 0.5
                        ) / 997.0
                        for sample_count in cv_samples:
                            cv_output, cv_diagnostics = (
                                proxy_mass_control_variate_output(
                                    exact_scores,
                                    joint_proxy,
                                    head_value,
                                    joint_approximate_values,
                                    output_selected,
                                    sample_count,
                                    phase,
                                    args.cv_correction,
                                )
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_boundrisk_"
                                        f"sharedtail_proxycv{sample_count}_"
                                        f"{args.cv_correction}"
                                    ),
                                    "selected_tokens": (
                                        keep
                                        + int(
                                            cv_diagnostics[
                                                "sample_unique_tokens"
                                            ]
                                        )
                                    ),
                                    "selected_fraction": (
                                        keep
                                        + cv_diagnostics[
                                            "sample_unique_tokens"
                                        ]
                                    )
                                    / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["value_risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(cv_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, output_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            output_selected,
                                        )[0]
                                    ),
                                    "cv_diagnostics": cv_diagnostics,
                                }
                            )
                            matched_keep = min(
                                history_count, keep + sample_count
                            )
                            matched_selected = torch.topk(
                                joint_output_priority,
                                matched_keep,
                                sorted=False,
                            ).indices
                            matched_output = rvq_tail_output(
                                exact_scores,
                                joint_proxy,
                                head_value,
                                matched_selected,
                                joint_assignments,
                                joint_value_centroids,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{joint_state['name']}_boundrisk_"
                                        f"sharedtail_overfetch{sample_count}"
                                    ),
                                    "selected_tokens": matched_keep,
                                    "selected_fraction": matched_keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(joint_state["risk_bits"])
                                        + float(joint_state["value_risk_bits"])
                                        + float(joint_state["fixed_key_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        joint_state["tail_bits"]
                                    ),
                                    **output_metrics(matched_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, matched_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            matched_selected,
                                        )[0]
                                    ),
                                }
                            )
                        for additive_state in joint_state[
                            "additive_value_states"
                        ]:
                            additive_primary = additive_state["primary_centroids"]
                            additive_residual = additive_state["residual_centroids"]
                            additive_assignments = additive_state[
                                "residual_assignments"
                            ]
                            additive_value_errors = additive_state["value_errors"]
                            assert isinstance(additive_primary, torch.Tensor)
                            assert isinstance(additive_residual, torch.Tensor)
                            assert isinstance(additive_assignments, torch.Tensor)
                            assert isinstance(additive_value_errors, torch.Tensor)
                            reconstruction_norm = (
                                additive_primary.float().index_select(
                                    0, joint_assignments
                                )
                                + additive_residual.float().index_select(
                                    0, additive_assignments
                                )
                            ).norm(dim=-1)
                            additive_sensitivity = (
                                additive_value_errors
                                + joint_uncertainty
                                * (reconstruction_norm + additive_value_errors)
                            ).clamp_min(1.0e-8)
                            additive_priority = (
                                joint_proxy + additive_sensitivity.log()
                            )
                            additive_selected = torch.topk(
                                additive_priority, keep, sorted=False
                            ).indices
                            for tail_name, tail_proxy in (
                                ("raw", joint_proxy),
                                ("exactmass_oracle", exact_scores),
                            ):
                                additive_output = additive_rvq_tail_output(
                                    exact_scores,
                                    tail_proxy,
                                    head_value,
                                    additive_selected,
                                    joint_assignments,
                                    additive_primary,
                                    additive_assignments,
                                    additive_residual,
                                )
                                rows.append(
                                    {
                                        "text": text_path.stem,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "query_offset": query_offset,
                                        "token_offset": token_offset,
                                        "selector": (
                                            f"{joint_state['name']}_boundrisk_"
                                            f"{additive_state['name']}_"
                                            f"additivetail_{tail_name}"
                                        ),
                                        "selected_tokens": keep,
                                        "selected_fraction": keep / history_count,
                                        "selector_bits_per_token": (
                                            args.binary_bits
                                            + args.residual_vq_bits
                                            + float(additive_state["id_bits"])
                                            + float(joint_state["risk_bits"])
                                            + float(
                                                additive_state["value_risk_bits"]
                                            )
                                            + float(joint_state["fixed_key_bits"])
                                        ),
                                        "tail_bits_per_token": float(
                                            additive_state["tail_bits"]
                                        ),
                                        **output_metrics(
                                            additive_output, full_output
                                        ),
                                        "selected_mass": float(
                                            full_weights.index_select(
                                                0, additive_selected
                                            ).sum()
                                        ),
                                        "top1_recall": float(
                                            torch.isin(
                                                torch.argmax(exact_scores)[None],
                                                additive_selected,
                                            )[0]
                                        ),
                                    }
                                )
                            for calibration_count in tail_calibration_counts:
                                calibrated_proxy = affine_calibrated_proxy_scores(
                                    exact_scores,
                                    joint_proxy,
                                    calibration_count,
                                )
                                calibrated_output = additive_rvq_tail_output(
                                    exact_scores,
                                    calibrated_proxy,
                                    head_value,
                                    additive_selected,
                                    joint_assignments,
                                    additive_primary,
                                    additive_assignments,
                                    additive_residual,
                                )
                                calibration_indices = evenly_spaced_indices(
                                    history_count,
                                    calibration_count,
                                    exact_scores.device,
                                )
                                calibration_overlap = int(
                                    torch.isin(
                                        calibration_indices,
                                        additive_selected,
                                    ).sum()
                                )
                                exact_reads = (
                                    keep
                                    + calibration_count
                                    - calibration_overlap
                                )
                                rows.append(
                                    {
                                        "text": text_path.stem,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "query_offset": query_offset,
                                        "token_offset": token_offset,
                                        "selector": (
                                            f"{joint_state['name']}_boundrisk_"
                                            f"{additive_state['name']}_"
                                            f"additivetail_affine{calibration_count}"
                                        ),
                                        "selected_tokens": exact_reads,
                                        "selected_fraction": exact_reads / history_count,
                                        "selector_bits_per_token": (
                                            args.binary_bits
                                            + args.residual_vq_bits
                                            + float(additive_state["id_bits"])
                                            + float(joint_state["risk_bits"])
                                            + float(
                                                additive_state[
                                                    "value_risk_bits"
                                                ]
                                            )
                                            + float(joint_state["fixed_key_bits"])
                                        ),
                                        "tail_bits_per_token": float(
                                            additive_state["tail_bits"]
                                        ),
                                        **output_metrics(
                                            calibrated_output, full_output
                                        ),
                                        "selected_mass": float(
                                            full_weights.index_select(
                                                0, additive_selected
                                            ).sum()
                                        ),
                                        "top1_recall": float(
                                            torch.isin(
                                                torch.argmax(exact_scores)[None],
                                                additive_selected,
                                            )[0]
                                        ),
                                    }
                                )
                            additive_approximate_values = (
                                additive_primary.float().index_select(
                                    0, joint_assignments
                                )
                                + additive_residual.float().index_select(
                                    0, additive_assignments
                                )
                            )
                            for tolerance in adaptive_error_tolerances:
                                adaptive_selected, adaptive_diagnostics = (
                                    select_by_output_rms_bound(
                                        joint_proxy,
                                        joint_uncertainty,
                                        additive_approximate_values,
                                        additive_value_errors,
                                        tolerance,
                                    )
                                )
                                adaptive_output = additive_rvq_tail_output(
                                    exact_scores,
                                    joint_proxy,
                                    head_value,
                                    adaptive_selected,
                                    joint_assignments,
                                    additive_primary,
                                    additive_assignments,
                                    additive_residual,
                                )
                                tolerance_name = int(round(1000 * tolerance))
                                rows.append(
                                    {
                                        "text": text_path.stem,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "query_offset": query_offset,
                                        "token_offset": token_offset,
                                        "selector": (
                                            f"{joint_state['name']}_adaptive_"
                                            f"rmstol{tolerance_name:03d}_"
                                            f"{additive_state['name']}_additivetail"
                                        ),
                                        "selected_tokens": int(
                                            adaptive_selected.numel()
                                        ),
                                        "selected_fraction": (
                                            adaptive_selected.numel()
                                            / history_count
                                        ),
                                        "selector_bits_per_token": (
                                            args.binary_bits
                                            + args.residual_vq_bits
                                            + float(additive_state["id_bits"])
                                            + float(joint_state["risk_bits"])
                                            + float(
                                                additive_state[
                                                    "value_risk_bits"
                                                ]
                                            )
                                            + float(joint_state["fixed_key_bits"])
                                        ),
                                        "tail_bits_per_token": float(
                                            additive_state["tail_bits"]
                                        ),
                                        **output_metrics(
                                            adaptive_output, full_output
                                        ),
                                        "selected_mass": float(
                                            full_weights.index_select(
                                                0, adaptive_selected
                                            ).sum()
                                        ),
                                        "top1_recall": float(
                                            torch.isin(
                                                torch.argmax(exact_scores)[None],
                                                adaptive_selected,
                                            )[0]
                                        ),
                                        "adaptive_diagnostics": (
                                            adaptive_diagnostics
                                        ),
                                    }
                                )
                            for sample_count in cv_samples:
                                cv_output, cv_diagnostics = (
                                    proxy_mass_control_variate_output(
                                        exact_scores,
                                        joint_proxy,
                                        head_value,
                                        additive_approximate_values,
                                        additive_selected,
                                        sample_count,
                                        phase,
                                        args.cv_correction,
                                    )
                                )
                                rows.append(
                                    {
                                        "text": text_path.stem,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "query_offset": query_offset,
                                        "token_offset": token_offset,
                                        "selector": (
                                            f"{joint_state['name']}_boundrisk_"
                                            f"{additive_state['name']}_"
                                            f"additivetail_proxycv{sample_count}_"
                                            f"{args.cv_correction}"
                                        ),
                                        "selected_tokens": (
                                            keep
                                            + int(
                                                cv_diagnostics[
                                                    "sample_unique_tokens"
                                                ]
                                            )
                                        ),
                                        "selected_fraction": (
                                            keep
                                            + cv_diagnostics[
                                                "sample_unique_tokens"
                                            ]
                                        )
                                        / history_count,
                                        "selector_bits_per_token": (
                                            args.binary_bits
                                            + args.residual_vq_bits
                                            + float(additive_state["id_bits"])
                                            + float(joint_state["risk_bits"])
                                            + float(
                                                additive_state[
                                                    "value_risk_bits"
                                                ]
                                            )
                                            + float(joint_state["fixed_key_bits"])
                                        ),
                                        "tail_bits_per_token": float(
                                            additive_state["tail_bits"]
                                        ),
                                        **output_metrics(cv_output, full_output),
                                        "selected_mass": float(
                                            full_weights.index_select(
                                                0, additive_selected
                                            ).sum()
                                        ),
                                        "top1_recall": float(
                                            torch.isin(
                                                torch.argmax(exact_scores)[None],
                                                additive_selected,
                                            )[0]
                                        ),
                                        "cv_diagnostics": cv_diagnostics,
                                    }
                                )
                                matched_keep = min(
                                    history_count, keep + sample_count
                                )
                                matched_selected = torch.topk(
                                    additive_priority,
                                    matched_keep,
                                    sorted=False,
                                ).indices
                                matched_output = additive_rvq_tail_output(
                                    exact_scores,
                                    joint_proxy,
                                    head_value,
                                    matched_selected,
                                    joint_assignments,
                                    additive_primary,
                                    additive_assignments,
                                    additive_residual,
                                )
                                rows.append(
                                    {
                                        "text": text_path.stem,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "query_offset": query_offset,
                                        "token_offset": token_offset,
                                        "selector": (
                                            f"{joint_state['name']}_boundrisk_"
                                            f"{additive_state['name']}_"
                                            f"additivetail_overfetch{sample_count}"
                                        ),
                                        "selected_tokens": matched_keep,
                                        "selected_fraction": matched_keep / history_count,
                                        "selector_bits_per_token": (
                                            args.binary_bits
                                            + args.residual_vq_bits
                                            + float(additive_state["id_bits"])
                                            + float(joint_state["risk_bits"])
                                            + float(
                                                additive_state[
                                                    "value_risk_bits"
                                                ]
                                            )
                                            + float(joint_state["fixed_key_bits"])
                                        ),
                                        "tail_bits_per_token": float(
                                            additive_state["tail_bits"]
                                        ),
                                        **output_metrics(
                                            matched_output, full_output
                                        ),
                                        "selected_mass": float(
                                            full_weights.index_select(
                                                0, matched_selected
                                            ).sum()
                                        ),
                                        "top1_recall": float(
                                            torch.isin(
                                                torch.argmax(exact_scores)[None],
                                                matched_selected,
                                            )[0]
                                        ),
                                    }
                                )
                    for product_state in product_query_states:
                        product_priority = product_state["priority"]
                        product_proxy = product_state["proxy"]
                        product_assignments = product_state["assignments"]
                        product_value_centroids = product_state["value_centroids"]
                        assert isinstance(product_priority, torch.Tensor)
                        assert isinstance(product_proxy, torch.Tensor)
                        assert isinstance(product_assignments, torch.Tensor)
                        assert isinstance(product_value_centroids, torch.Tensor)
                        product_selected = torch.topk(
                            product_priority, keep, sorted=False
                        ).indices
                        for tail_name, tail_proxy in (
                            ("raw", product_proxy),
                            ("exactmass_oracle", exact_scores),
                        ):
                            product_output = rvq_tail_output(
                                exact_scores,
                                tail_proxy,
                                head_value,
                                product_selected,
                                product_assignments,
                                product_value_centroids,
                            )
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "token_offset": token_offset,
                                    "selector": (
                                        f"{product_state['name']}_sharedtail_"
                                        f"{tail_name}"
                                    ),
                                    "selected_tokens": keep,
                                    "selected_fraction": keep / history_count,
                                    "selector_bits_per_token": (
                                        args.binary_bits
                                        + args.residual_vq_bits
                                        + float(product_state["risk_bits"])
                                    ),
                                    "tail_bits_per_token": float(
                                        product_state["tail_bits"]
                                    ),
                                    **output_metrics(product_output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(
                                            0, product_selected
                                        ).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None],
                                            product_selected,
                                        )[0]
                                    ),
                                }
                            )
                    output_risk_priority = (
                        global_proxy
                        + args.risk_lambda * global_uncertainty
                        + log_value_deviation
                    )
                    contribution_order = torch.argsort(
                        output_risk_priority, descending=True
                    )
                    contribution_mass = torch.softmax(
                        output_risk_priority.index_select(0, contribution_order),
                        dim=0,
                    ).cumsum(dim=0)
                    for coverage in adaptive_coverages:
                        adaptive_keep = int(
                            torch.searchsorted(
                                contribution_mass,
                                torch.tensor(
                                    coverage,
                                    device=contribution_mass.device,
                                    dtype=contribution_mass.dtype,
                                ),
                            ).item()
                        ) + 1
                        selected = contribution_order[:adaptive_keep]
                        metrics, tail_bits = tail_output_and_metrics(
                            exact_scores,
                            full_weights,
                            full_output,
                            head_value,
                            selected,
                            metric_coordinates,
                            global_metric_query,
                            head_key,
                            current_query,
                            coreset,
                            scale,
                        )
                        coverage_name = int(round(100 * coverage))
                        rows.append(
                            {
                                "text": text_path.stem,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "query_offset": query_offset,
                                "token_offset": token_offset,
                                "selector": (
                                    "global_qmetric_output_mass"
                                    f"{coverage_name:02d}"
                                ),
                                "selected_tokens": adaptive_keep,
                                "selected_fraction": adaptive_keep / history_count,
                                "selector_bits_per_token": (
                                    args.binary_bits
                                    + risk_aux_bits
                                    + value_aux_bits
                                ),
                                "tail_bits_per_token": tail_bits,
                                **metrics,
                            }
                        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["selector"], row["query_offset"])].append(row)
    output = []
    for (selector, offset), group in sorted(groups.items()):
        errors = torch.tensor([row["relative_l2"] for row in group])
        output.append(
            {
                "selector": selector,
                "query_offset": offset,
                "conditions": len(group),
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_p90": float(torch.quantile(errors, 0.9)),
                "relative_l2_worst": float(errors.max()),
                "cosine_mean": sum(row["cosine"] for row in group) / len(group),
                "selected_mass_mean": sum(row["selected_mass"] for row in group)
                / len(group),
                "top1_recall_mean": sum(row["top1_recall"] for row in group)
                / len(group),
                "selected_tokens_mean": sum(
                    row["selected_tokens"] for row in group
                )
                / len(group),
                "selected_fraction_mean": sum(
                    row["selected_fraction"] for row in group
                )
                / len(group),
                "selector_bits_per_token": group[0]["selector_bits_per_token"],
                "tail_bits_per_token": group[0]["tail_bits_per_token"],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    assert_numeric_backend_sane()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    codebooks, metadata = fit_global_codebooks(tokenizer, args)
    rows: list[dict[str, Any]] = []
    for text_path in args.test_texts:
        rows.extend(evaluate_test_text(tokenizer, text_path, codebooks, metadata, args))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "schema": "qmetric_global_holdout_layer0_v1",
        "contract": {
            "scope": "cross-text real layer-0 mechanism audit; not end-to-end quality",
            "source_sha256": source_hash,
            "model": str(args.model),
            "train_texts": [str(path) for path in args.train_texts],
            "test_texts": [str(path) for path in args.test_texts],
            "history_tokens": args.history_tokens,
            "query_offsets": args.query_offsets,
            "query_tokens": args.query_tokens,
            "calibration_tokens": args.calibration_tokens,
            "query_samples_per_text": args.query_samples_per_text,
            "key_samples_per_text": args.key_samples_per_text,
            "local_key_sample_count": args.local_key_sample_count,
            "fraction": args.fraction,
            "eas_ratio": args.eas_ratio,
            "adaptive_coverages": args.adaptive_coverages,
            "binary_bits": args.binary_bits,
            "projection_weighting": args.projection_weighting,
            "residual_vq_bits": args.residual_vq_bits,
            "residual_vq_iterations": args.residual_vq_iterations,
            "residual_binary_bits": args.residual_binary_bits,
            "residual_binary_iterations": args.residual_binary_iterations,
            "residual_binary_candidate_fractions": (
                args.residual_binary_candidate_fractions
            ),
            "joint_rvq_weights": args.joint_rvq_weights,
            "additive_value_bits": args.additive_value_bits,
            "additive_value_iterations": args.additive_value_iterations,
            "additive_refit_iterations": args.additive_refit_iterations,
            "additive_block_sizes": args.additive_block_sizes,
            "cv_samples": args.cv_samples,
            "cv_correction": args.cv_correction,
            "tail_calibration_counts": args.tail_calibration_counts,
            "adaptive_error_tolerances": args.adaptive_error_tolerances,
            "product_rvq_key_bits": args.product_rvq_key_bits,
            "risk_lambda": args.risk_lambda,
            "risk_error_bits": args.risk_error_bits,
            "risk_error_block_size": args.risk_error_block_size,
            "metric_shrinkage": args.metric_shrinkage,
            "block_size": args.block_size,
            "key_mean_bits": args.key_mean_bits,
            "value_mean_bits": args.value_mean_bits,
            "model_metadata": metadata,
            "resolved_metric_shrinkages": [
                codebook["resolved_shrinkage"] for codebook in codebooks
            ],
            "projection_weighting_diagnostics": [
                codebook["weighting_diagnostics"] for codebook in codebooks
            ],
        },
        "aggregate": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
