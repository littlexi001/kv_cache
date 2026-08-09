#!/usr/bin/env python
"""Evaluate a length-free, output-risk coverage budget on real Q/K/V traces.

For an omitted token i, the Value-sketch contribution error is bounded by
attention_weight_i * ||W_o (v_i - vhat_i)||.  This script ranks tokens by the
proxy version of that quantity and chooses the smallest set whose cumulative
proxy risk reaches a requested fraction.  No task label or context-length
threshold is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_prefill_tail_calibration_20260803 import (
    metric_basis,
    output_group_gram,
    quantized_log_risk,
)
from analyze_qksieve_tail_partition_calibration_20260803 import (
    load_output_projection,
)
from analyze_qksieve_value_sketch_residual_20260801 import (
    block_affine_quantize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed_top_k", type=int, default=1280)
    parser.add_argument(
        "--fixed_top_ks",
        default="",
        help="Optional comma-separated fixed-budget sweep.",
    )
    parser.add_argument(
        "--global_top_ks",
        default="",
        help=(
            "Optional per-head-equivalent budgets for global head-token "
            "allocation, for example 1280,2560."
        ),
    )
    parser.add_argument(
        "--global_floor_fractions",
        default="",
        help=(
            "Optional comma-separated fractions of each equivalent per-head "
            "budget reserved for that head's highest proxy logits before "
            "globally allocating the remaining slots."
        ),
    )
    parser.add_argument(
        "--global_priority_names",
        default="",
        help="Optional comma-separated subset of global allocation names.",
    )
    parser.add_argument("--coverage_targets", default="0.90,0.95,0.975,0.99")
    parser.add_argument(
        "--sampled_mass_samples",
        default="",
        help="Optional comma-separated sample counts for proxy-mass thresholds.",
    )
    parser.add_argument(
        "--sampled_mass_aggregations",
        default="minimum,median",
        help="Replica-boundary aggregations: minimum and/or median.",
    )
    parser.add_argument(
        "--gaussian_mass_samples",
        default="",
        help=(
            "Optional comma-separated sample counts for the analytic "
            "exponentially tilted Gaussian softmax-mass boundary."
        ),
    )
    parser.add_argument(
        "--mass_ladder_samples",
        default="",
        help=(
            "Optional comma-separated sample counts for sampled-rank "
            "threshold ladders whose full proxy mass is measured."
        ),
    )
    parser.add_argument(
        "--mass_ladder_growth",
        type=float,
        default=2.0,
        help="Multiplicative candidate-count growth between ladder rungs.",
    )
    parser.add_argument(
        "--interval_mass_samples",
        default="",
        help=(
            "Optional sample counts for score-error interval-certified "
            "proxy-mass ladders."
        ),
    )
    parser.add_argument(
        "--coverage_histogram_bins",
        default="",
        help=(
            "Optional comma-separated logit-histogram sizes for a linear-scan "
            "approximation to each attention coverage target."
        ),
    )
    parser.add_argument(
        "--relative_risk_thresholds",
        default="",
        help=(
            "Optional comma-separated per-head tail-risk tolerances relative "
            "to the projected proxy output norm."
        ),
    )
    parser.add_argument(
        "--rss_relative_tolerances",
        default="",
        help=(
            "Optional comma-separated relative RMS tail-risk tolerances. "
            "Unlike the L1 certificate, this models cancellation."
        ),
    )
    parser.add_argument(
        "--rss_safety_factors",
        default="1,2,4",
        help="Comma-separated multipliers applied to the RMS tail risk.",
    )
    parser.add_argument(
        "--global_rss_tolerances",
        default="",
        help=(
            "Optional layer-level RMS tail tolerances with a fixed per-head "
            "floor and globally allocated extra slots."
        ),
    )
    parser.add_argument("--global_rss_floor_k", type=int, default=1280)
    parser.add_argument(
        "--balanced_rss_tolerances",
        default="",
        help=(
            "Optional per-head joint QK/Value RSS tolerances. The head error "
            "scales share the layer root-sum-square output energy, so no "
            "context-length threshold is required."
        ),
    )
    parser.add_argument(
        "--scalar_rss_tolerances",
        default="",
        help=(
            "Optional comma-separated relative RMS tail tolerances for the "
            "sort-free scalar Value-residual bound."
        ),
    )
    parser.add_argument(
        "--scalar_rss_statistics",
        default="rms,p90,maximum",
        help="Value-residual scale summaries: rms, p90, and/or maximum.",
    )
    parser.add_argument("--minimum_top_k", type=int, default=1)
    parser.add_argument("--maximum_top_k", type=int, default=0)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument(
        "--query_factor_source",
        choices=("decode", "prefill", "prefill_decode"),
        default="decode",
        help=(
            "Queries used to construct the request-local QK-balanced basis. "
            "The production-compatible choice is prefill; decode is retained "
            "only for reproducing earlier diagnostic runs."
        ),
    )
    parser.add_argument(
        "--query_factor_prefill_tokens",
        type=int,
        default=0,
        help=(
            "Number of final prefill Query positions used by the QK basis. "
            "Zero uses the trace's query_calibration_tokens declaration."
        ),
    )
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument(
        "--key_bit_levels",
        default="0,1,2,4,8",
        help="Allowed per-band Key quantization widths.",
    )
    parser.add_argument(
        "--key_refinement_rate_budget",
        type=int,
        default=0,
        help=(
            "Optional higher progressive Key-index rate. Zero disables "
            "error-balanced boundary refinement."
        ),
    )
    parser.add_argument(
        "--progressive_refinement_rounds",
        type=int,
        default=2,
        help="Maximum low-rate-to-high-rate boundary refinement rounds.",
    )
    parser.add_argument(
        "--key_quantizer",
        choices=("plain", "metric"),
        default="metric",
        help="Key reconstruction used by both allocation and proxy execution.",
    )
    parser.add_argument(
        "--key_allocation_objective",
        choices=(
            "key_mse",
            "qk_mse",
            "oas_qk_mse",
            "balanced_qk_mse",
            "robust_qk_mse",
        ),
        default="key_mse",
        help="Distortion minimized by the per-head spectral bit allocator.",
    )
    parser.add_argument(
        "--key_allocation_query_source",
        choices=("basis", "decode_first", "prefill_decode_first"),
        default="basis",
        help=(
            "Queries used only for bit-plane allocation. decode_first uses "
            "the first legal decode Query and evaluates later steps without "
            "future-query leakage."
        ),
    )
    parser.add_argument("--value_rank", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument(
        "--value_residual_samples",
        default="",
        help=(
            "Optional comma-separated systematic sample counts for an "
            "unbiased Value-tail residual numerator correction."
        ),
    )
    parser.add_argument(
        "--affine_bound_tolerances",
        default="",
        help=(
            "Optional comma-separated relative output-error bounds for the "
            "query-directed affine residual-moment ladder."
        ),
    )
    parser.add_argument("--risk_bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--value_scale_block", type=int, default=256)
    parser.add_argument("--score_calibration_samples", type=int, default=256)
    parser.add_argument(
        "--focus_block_risk",
        action="store_true",
        help="Evaluate only block-residual RSS masks and key output paths.",
    )
    parser.add_argument(
        "--focus_global_floor_rss",
        action="store_true",
        help="Evaluate only layer RSS allocation with a fixed per-head floor.",
    )
    parser.add_argument(
        "--focus_sampled_mass",
        action="store_true",
        help="Evaluate only exact and sampled proxy-mass prefixes.",
    )
    parser.add_argument(
        "--focus_scalar_rss",
        action="store_true",
        help="Evaluate only the sort-free scalar Value-residual RSS rule.",
    )
    parser.add_argument(
        "--focus_gaussian_mass",
        action="store_true",
        help="Evaluate only exact and Gaussian-moment mass prefixes.",
    )
    parser.add_argument(
        "--focus_mass_ladder",
        action="store_true",
        help="Evaluate only exact and measured sampled-rank mass ladders.",
    )
    parser.add_argument(
        "--focus_interval_mass",
        action="store_true",
        help="Evaluate only score-error interval-certified mass ladders.",
    )
    parser.add_argument(
        "--focus_balanced_rss",
        action="store_true",
        help="Evaluate only the balanced per-head joint QK/Value RSS rule.",
    )
    parser.add_argument(
        "--focus_progressive_balanced_rss",
        action="store_true",
        help=(
            "Evaluate only error-balanced progressive bit-plane refinement "
            "followed by the balanced joint QK/Value RSS rule."
        ),
    )
    parser.add_argument(
        "--focus_rss_calibration",
        action="store_true",
        help=(
            "Keep a small mask/output set and report whether the diagonal "
            "Value-tail RSS predicts the realized layer-output error."
        ),
    )
    return parser.parse_args()


def qk_calibration_queries(
    decode_queries: torch.Tensor,
    prefill_queries: torch.Tensor | None,
    source: str,
) -> torch.Tensor:
    """Flatten the legal request-local Queries used to build the QK basis."""

    if decode_queries.ndim != 3:
        raise ValueError("decode_queries must have shape [steps, groups, dim]")
    decode_flat = decode_queries.reshape(-1, decode_queries.shape[-1])
    if source == "decode":
        return decode_flat
    if prefill_queries is None:
        raise ValueError(f"query_factor_source={source} requires prefill Queries")
    if prefill_queries.ndim != 3:
        raise ValueError("prefill_queries must have shape [tokens, groups, dim]")
    prefill_flat = prefill_queries.reshape(-1, prefill_queries.shape[-1])
    if source == "prefill":
        return prefill_flat
    if source == "prefill_decode":
        return torch.cat((prefill_flat, decode_flat), dim=0)
    raise ValueError(f"unsupported query_factor_source: {source}")


def key_quantization_candidates(
    key_coordinates: torch.Tensor,
    calibration_queries: torch.Tensor,
    quantizer: str,
    bit_levels: tuple[int, ...] = ZERO_BIT_LEVELS,
) -> list[dict[int, torch.Tensor]]:
    """Build actual per-band reconstructions used by the experiment."""

    if quantizer == "metric":
        if bit_levels != ZERO_BIT_LEVELS:
            raise ValueError(
                "custom bit levels are currently supported by plain "
                "quantization only"
            )
        return quantized_bands(key_coordinates, calibration_queries)
    if quantizer != "plain":
        raise ValueError(f"unsupported key quantizer: {quantizer}")
    bands: list[dict[int, torch.Tensor]] = []
    for band_index in range(8):
        start = band_index * 16
        stop = start + 16
        key_band = key_coordinates[:, start:stop]
        bands.append(
            {
                bits: plain_quantize_band(key_band, bits)
                for bits in bit_levels
            }
        )
    return bands


def plain_quantize_band(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-token 16-value quantization at any width up to INT8."""

    if bits in ZERO_BIT_LEVELS:
        return quantize_band(values, bits)
    if bits <= 1 or bits > 8 or values.shape[-1] != 16:
        raise ValueError("custom plain Key quantization expects 2..8 bits")
    maximum_code = (1 << (bits - 1)) - 1
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    scale = scale / float(maximum_code)
    codes = torch.round(values.float() / scale).clamp(
        -maximum_code, maximum_code
    )
    return codes * scale


def key_allocation_distortion(
    key_coordinates: torch.Tensor,
    calibration_queries: torch.Tensor,
    candidates: list[dict[int, torch.Tensor]],
    objective: str,
    balanced_singular_values: torch.Tensor | None = None,
) -> list[dict[int, torch.Tensor]]:
    """Evaluate bit costs with the same reconstruction used by execution."""

    if objective not in {
        "key_mse",
        "qk_mse",
        "oas_qk_mse",
        "balanced_qk_mse",
        "robust_qk_mse",
    }:
        raise ValueError(f"unsupported allocation objective: {objective}")
    if objective in {"balanced_qk_mse", "robust_qk_mse"}:
        if balanced_singular_values is None:
            raise ValueError("balanced_qk_mse requires singular values")
        if balanced_singular_values.numel() != key_coordinates.shape[-1]:
            raise ValueError("balanced singular-value dimension mismatch")
    if objective == "oas_qk_mse":
        oas_alpha, isotropic_query_variance = oas_query_metric_parameters(
            calibration_queries
        )
    else:
        oas_alpha = calibration_queries.new_zeros(())
        isotropic_query_variance = calibration_queries.new_zeros(())
    table: list[dict[int, torch.Tensor]] = []
    for band_index, reconstructions in enumerate(candidates):
        start = band_index * 16
        stop = start + 16
        exact_band = key_coordinates[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs: dict[int, torch.Tensor] = {}
        for bits, reconstructed in reconstructions.items():
            residual = exact_band - reconstructed
            if objective == "key_mse":
                costs[bits] = residual.float().square().mean()
            elif objective in {"qk_mse", "oas_qk_mse"}:
                score_error = query_band.float() @ residual.float().T
                empirical_cost = score_error.square().mean()
                if objective == "oas_qk_mse":
                    isotropic_cost = (
                        residual.float().square().sum(dim=-1).mean()
                        * isotropic_query_variance
                    )
                    costs[bits] = (
                        (1.0 - oas_alpha) * empirical_cost
                        + oas_alpha * isotropic_cost
                    )
                else:
                    costs[bits] = empirical_cost
            else:
                weights = balanced_singular_values[start:stop].float()
                balanced_cost = (
                    residual.float().square() * weights[None, :]
                ).sum(dim=-1).mean()
                if objective == "balanced_qk_mse":
                    costs[bits] = balanced_cost
                else:
                    score_error = query_band.float() @ residual.float().T
                    empirical_cost = score_error.square().mean()
                    costs[bits] = torch.maximum(
                        empirical_cost,
                        balanced_cost,
                    )
        table.append(costs)
    return table


def oas_query_metric_parameters(
    calibration_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytic OAS shrinkage and its isotropic second moment.

    The empirical query second moment is rank deficient when only a few
    prompt-tail Queries are available. OAS shrinks that matrix toward
    ``trace(S) / d * I`` without task labels or trained parameters.
    """

    if calibration_queries.ndim != 2:
        raise ValueError("calibration queries must have shape [samples, dim]")
    sample_count, dimensions = calibration_queries.shape
    if sample_count <= 0 or dimensions <= 0:
        raise ValueError("OAS query metric requires non-empty queries")
    queries = calibration_queries.float()
    second_moment = queries.T @ queries / float(sample_count)
    trace = second_moment.trace()
    trace_square = second_moment.square().sum()
    numerator = (
        (1.0 - 2.0 / float(dimensions)) * trace_square + trace.square()
    )
    denominator = (
        sample_count + 1.0 - 2.0 / float(dimensions)
    ) * (trace_square - trace.square() / float(dimensions))
    alpha = torch.where(
        denominator <= 1.0e-20,
        torch.ones_like(denominator),
        torch.clamp(numerator / denominator, 0.0, 1.0),
    )
    return alpha, trace / float(dimensions)


def balanced_head_output_scales(
    head_output_scales: torch.Tensor,
) -> torch.Tensor:
    """Allocate one layer RSS error budget without starving quiet heads.

    For head energies ``a_h^2``, define

        b_h^2 = (a_h^2 + mean_j(a_j^2)) / 2.

    Then ``sum_h b_h^2 = sum_h a_h^2`` exactly. Thus equal relative RSS
    tolerances on ``b_h`` preserve the layer root-sum-square budget while
    avoiding the singular behavior of normalizing a quiet head by ``a_h``.
    """

    if head_output_scales.ndim != 2:
        raise ValueError("head output scales must have shape [steps, heads]")
    squared = head_output_scales.float().square()
    mean_squared = squared.mean(dim=-1, keepdim=True)
    return torch.sqrt(0.5 * (squared + mean_squared)).clamp_min(1.0e-12)


def joint_qk_value_rss_risk(
    value_residual_risk: torch.Tensor,
    score_rmse: torch.Tensor,
    value_deviation: torch.Tensor,
) -> torch.Tensor:
    """First-order output risk from independent QK and Value errors.

    Softmax linearization gives a score-error contribution proportional to
    ``delta_score * (value - output)``. Combining that term in quadrature
    with Value-sketch residual risk yields a cancellation-aware per-token
    scale rather than the previous linear worst-case sum.
    """

    if value_residual_risk.shape != value_deviation.shape:
        raise ValueError("Value residual risk and deviation must align")
    if score_rmse.shape != value_residual_risk.shape[:-1]:
        raise ValueError("score RMSE must have one value per query head")
    return torch.sqrt(
        value_residual_risk.float().square()
        + score_rmse.float().square()[:, None]
        * value_deviation.float().square()
    ).clamp_min(1.0e-30)


def conformal_score_uncertainty(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    raw_uncertainty: torch.Tensor,
    sample_count: int = 256,
    miscoverage: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a deterministic per-token proxy by a split-conformal quantile."""

    if exact_scores.shape != proxy_scores.shape or (
        raw_uncertainty.shape != exact_scores.shape
    ):
        raise ValueError("score uncertainty tensors must align")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie inside (0, 1)")
    history_count = int(exact_scores.shape[-1])
    active_samples = min(max(2, sample_count), history_count)
    sample_indices = stratified_jittered_sample_indices(
        history_count, active_samples, exact_scores.device
    )
    residual = (
        exact_scores.index_select(1, sample_indices).float()
        - proxy_scores.index_select(1, sample_indices).float()
    ).abs()
    sampled_uncertainty = raw_uncertainty.index_select(
        1, sample_indices
    ).float().clamp_min(1.0e-12)
    ratios = residual / sampled_uncertainty
    conformal_rank = min(
        active_samples,
        max(1, math.ceil((active_samples + 1) * (1.0 - miscoverage))),
    )
    scale = torch.kthvalue(ratios, conformal_rank, dim=-1).values
    return raw_uncertainty.float() * scale[:, None], scale


def sampled_score_output_error(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    reconstructed_value: torch.Tensor,
    head_grams: torch.Tensor,
    query_groups: int,
    sample_count: int = 256,
    top_count: int = 128,
) -> dict[str, torch.Tensor]:
    """Estimate score-induced output error from top-risk and tail probes.

    The highest proxy-score tokens are evaluated exactly. The remaining
    contribution is estimated from a deterministic stratified sample and
    expanded by the tail population. Full exact scores are used only for the
    two diagnostics returned alongside the legal sampled estimate.
    """

    if exact_scores.shape != proxy_scores.shape or exact_scores.ndim != 2:
        raise ValueError("exact and proxy scores must be [query, token]")
    if reconstructed_value.ndim != 2:
        raise ValueError("reconstructed Value must be [token, dimension]")
    if head_grams.shape[0] != query_groups:
        raise ValueError("one output Gram matrix is required per query group")
    if sample_count <= 1 or top_count < 0:
        raise ValueError("invalid score-output probe counts")

    history_count = int(exact_scores.shape[-1])
    active_top = min(top_count, history_count)
    active_total = min(max(active_top, sample_count), history_count)
    requested_tail = active_total - active_top
    estimate_values: list[torch.Tensor] = []
    standard_error_values: list[torch.Tensor] = []
    first_order_values: list[torch.Tensor] = []
    actual_values: list[torch.Tensor] = []

    for row in range(exact_scores.shape[0]):
        gram = head_grams[row % query_groups].float()
        proxy_probability = torch.softmax(proxy_scores[row].float(), dim=-1)
        proxy_output = proxy_probability @ reconstructed_value.float()
        difference = reconstructed_value.float() - proxy_output[None, :]
        score_delta = exact_scores[row].float() - proxy_scores[row].float()
        contribution = (
            proxy_probability[:, None]
            * score_delta[:, None]
            * difference
        )

        top_indices = torch.topk(
            proxy_scores[row], k=active_top, sorted=False
        ).indices
        top_mask = torch.zeros(
            history_count,
            dtype=torch.bool,
            device=exact_scores.device,
        )
        top_mask[top_indices] = True
        top_sum = contribution.index_select(0, top_indices).sum(dim=0)
        tail_indices = torch.nonzero(~top_mask, as_tuple=False).flatten()
        tail_population = int(tail_indices.numel())
        active_tail = min(requested_tail, tail_population)
        if active_tail:
            positions = stratified_jittered_sample_indices(
                tail_population,
                active_tail,
                exact_scores.device,
            )
            sampled_indices = tail_indices.index_select(0, positions)
            sampled = contribution.index_select(0, sampled_indices)
            sampled_mean = sampled.mean(dim=0)
            estimate = top_sum + tail_population * sampled_mean
            if active_tail > 1:
                centered = sampled - sampled_mean[None, :]
                metric_variance = torch.einsum(
                    "nd,de,ne->n", centered, gram, centered
                ).mean()
                finite_population_correction = max(
                    0.0, 1.0 - active_tail / tail_population
                )
                standard_error = tail_population * torch.sqrt(
                    metric_variance.clamp_min(0.0)
                    * finite_population_correction
                    / active_tail
                )
            else:
                standard_error = estimate.new_tensor(float("inf"))
        else:
            estimate = top_sum
            standard_error = estimate.new_zeros(())

        first_order = contribution.sum(dim=0)
        exact_output = (
            torch.softmax(exact_scores[row].float(), dim=-1)
            @ reconstructed_value.float()
        )
        actual = exact_output - proxy_output

        def metric_norm(vector: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(
                torch.einsum("d,de,e->", vector, gram, vector)
                .clamp_min(0.0)
            )

        estimate_values.append(metric_norm(estimate))
        standard_error_values.append(standard_error)
        first_order_values.append(metric_norm(first_order))
        actual_values.append(metric_norm(actual))

    return {
        "estimate": torch.stack(estimate_values),
        "standard_error": torch.stack(standard_error_values),
        "first_order": torch.stack(first_order_values),
        "actual": torch.stack(actual_values),
    }


def sampled_tail_score_output_error(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    value: torch.Tensor,
    reconstructed_value: torch.Tensor,
    selected_mask: torch.Tensor,
    head_grams: torch.Tensor,
    query_groups: int,
    sample_count: int = 256,
    top_count: int = 128,
    score_uncertainty: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate only the proxy-score error left after exact sparse attention."""

    if exact_scores.shape != proxy_scores.shape or (
        selected_mask.shape != exact_scores.shape
    ):
        raise ValueError("tail score tensors must align by query/token")
    if value.shape != reconstructed_value.shape:
        raise ValueError("exact and reconstructed Value tensors must align")
    if head_grams.shape[0] != query_groups:
        raise ValueError("one output Gram matrix is required per query group")
    if score_uncertainty is not None and (
        score_uncertainty.shape != exact_scores.shape
    ):
        raise ValueError("score uncertainty must align with score tensors")

    estimate_values: list[torch.Tensor] = []
    standard_error_values: list[torch.Tensor] = []
    first_order_values: list[torch.Tensor] = []
    actual_values: list[torch.Tensor] = []
    history_count = int(exact_scores.shape[-1])

    for row in range(exact_scores.shape[0]):
        gram = head_grams[row % query_groups].float()
        active = selected_mask[row]
        tail = ~active
        tail_indices = torch.nonzero(tail, as_tuple=False).flatten()
        tail_population = int(tail_indices.numel())
        maximum = torch.maximum(
            exact_scores[row].amax(), proxy_scores[row].amax()
        )
        exact_weight = torch.exp(exact_scores[row].float() - maximum)
        proxy_weight = torch.exp(proxy_scores[row].float() - maximum)
        selected_exact_weight = exact_weight * active.float()
        proxy_tail_weight = proxy_weight * tail.float()
        exact_tail_weight = exact_weight * tail.float()
        selected_numerator = selected_exact_weight @ value.float()
        selected_partition = selected_exact_weight.sum()
        proxy_denominator = (
            selected_partition + proxy_tail_weight.sum()
        ).clamp_min(1.0e-20)
        exact_denominator = (
            selected_partition + exact_tail_weight.sum()
        ).clamp_min(1.0e-20)
        proxy_output = (
            selected_numerator
            + proxy_tail_weight @ reconstructed_value.float()
        ) / proxy_denominator
        exact_output = (
            selected_numerator
            + exact_tail_weight @ reconstructed_value.float()
        ) / exact_denominator
        difference = reconstructed_value.float() - proxy_output[None, :]
        score_delta = exact_scores[row].float() - proxy_scores[row].float()
        normalized_tail_weight = proxy_tail_weight / proxy_denominator
        contribution = (
            normalized_tail_weight[:, None]
            * score_delta[:, None]
            * difference
        )
        value_deviation = torch.sqrt(
            torch.einsum(
                "nd,de,ne->n", difference, gram, difference
            ).clamp_min(0.0)
        )
        log_value_deviation = torch.log(value_deviation.clamp_min(1.0e-30))
        proxy_probe_priority = (
            proxy_scores[row].float() + log_value_deviation
        ).masked_fill(active, -torch.inf)
        if score_uncertainty is None:
            optimistic_probe_priority = proxy_probe_priority
        else:
            optimistic_probe_priority = (
                proxy_scores[row].float()
                + score_uncertainty[row].float()
                + log_value_deviation
            ).masked_fill(active, -torch.inf)

        active_top = min(top_count, tail_population)
        active_total = min(max(active_top, sample_count), tail_population)
        requested_tail = active_total - active_top
        if active_top:
            proxy_top_count = (
                active_top
                if score_uncertainty is None
                else active_top // 2
            )
            optimistic_top_count = active_top - proxy_top_count
            proxy_top_indices = torch.topk(
                proxy_probe_priority,
                k=proxy_top_count,
                sorted=False,
            ).indices
            probe_mask = active.clone()
            probe_mask[proxy_top_indices] = True
            if optimistic_top_count:
                optimistic_top_indices = torch.topk(
                    optimistic_probe_priority.masked_fill(
                        probe_mask, -torch.inf
                    ),
                    k=optimistic_top_count,
                    sorted=False,
                ).indices
                top_indices = torch.cat(
                    (proxy_top_indices, optimistic_top_indices)
                )
            else:
                top_indices = proxy_top_indices
            top_sum = contribution.index_select(0, top_indices).sum(dim=0)
            probe_mask[top_indices] = True
        else:
            top_sum = torch.zeros_like(proxy_output)
            probe_mask = active.clone()

        sample_population_indices = torch.nonzero(
            ~probe_mask, as_tuple=False
        ).flatten()
        sample_population = int(sample_population_indices.numel())
        active_sample = min(requested_tail, sample_population)
        if active_sample:
            positions = stratified_jittered_sample_indices(
                sample_population,
                active_sample,
                exact_scores.device,
            )
            sampled_indices = sample_population_indices.index_select(
                0, positions
            )
            sampled = contribution.index_select(0, sampled_indices)
            sampled_mean = sampled.mean(dim=0)
            estimate = top_sum + sample_population * sampled_mean
            if active_sample > 1:
                centered = sampled - sampled_mean[None, :]
                metric_variance = torch.einsum(
                    "nd,de,ne->n", centered, gram, centered
                ).mean()
                finite_population_correction = max(
                    0.0, 1.0 - active_sample / sample_population
                )
                standard_error = sample_population * torch.sqrt(
                    metric_variance.clamp_min(0.0)
                    * finite_population_correction
                    / active_sample
                )
            else:
                standard_error = estimate.new_tensor(float("inf"))
        else:
            estimate = top_sum
            standard_error = estimate.new_zeros(())

        def metric_norm(vector: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(
                torch.einsum("d,de,e->", vector, gram, vector)
                .clamp_min(0.0)
            )

        estimate_values.append(metric_norm(estimate))
        standard_error_values.append(standard_error)
        first_order_values.append(metric_norm(contribution.sum(dim=0)))
        actual_values.append(metric_norm(exact_output - proxy_output))

    return {
        "estimate": torch.stack(estimate_values),
        "standard_error": torch.stack(standard_error_values),
        "first_order": torch.stack(first_order_values),
        "actual": torch.stack(actual_values),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must be non-empty and aligned")
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1.0e-20:
        return 0.0
    return float((x * y).sum() / denominator)


def sampled_mass_prefix_mask(
    scores: torch.Tensor,
    target_mass: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
    aggregation: str,
    replica_size: int = 256,
) -> torch.Tensor:
    """Estimate a softmax-mass score prefix from systematic samples."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, tokens]")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0, 1)")
    if sample_count <= 0 or replica_size <= 0:
        raise ValueError("sample counts must be positive")
    if aggregation not in {"minimum", "median"}:
        raise ValueError("aggregation must be minimum or median")
    history_count = int(scores.shape[-1])
    active_samples = min(sample_count, history_count)
    replicas = math.ceil(active_samples / replica_size)
    boundaries = []
    for replica in range(replicas):
        sample_ids = torch.arange(
            replica,
            active_samples,
            replicas,
            device=scores.device,
        )
        token_ids = (
            (2 * sample_ids + 1) * history_count
        ) // (2 * active_samples)
        sampled = scores.index_select(-1, token_ids)
        sorted_sample = torch.sort(sampled.float(), dim=-1, descending=True).values
        weights = torch.exp(sorted_sample - sorted_sample[..., :1])
        cumulative = torch.cumsum(weights, dim=-1)
        counts = torch.sum(
            cumulative < target_mass * cumulative[..., -1:], dim=-1
        ) + 1
        boundaries.append(
            torch.gather(
                sorted_sample, -1, (counts - 1).unsqueeze(-1)
            ).squeeze(-1)
        )
    stacked_boundaries = torch.stack(boundaries, dim=-1)
    boundary = (
        stacked_boundaries.amin(dim=-1)
        if aggregation == "minimum"
        else stacked_boundaries.median(dim=-1).values
    )
    mask = scores.float() >= boundary.unsqueeze(-1)
    active_minimum = min(max(1, minimum_top_k), history_count)
    floor_indices = torch.topk(
        scores.float(), k=active_minimum, dim=-1, sorted=False
    ).indices
    mask.scatter_(1, floor_indices, True)
    if maximum_top_k > 0:
        active_maximum = min(maximum_top_k, history_count)
        top_indices = torch.topk(
            scores.float(), k=active_maximum, dim=-1, sorted=False
        ).indices
        maximum_mask = torch.zeros_like(mask)
        maximum_mask.scatter_(1, top_indices, True)
        mask &= maximum_mask
    return mask


def gaussian_mass_prefix_mask(
    scores: torch.Tensor,
    target_mass: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
) -> torch.Tensor:
    """Estimate a softmax-mass prefix from sampled logit mean and variance.

    For S distributed as Normal(mu, sigma^2), exponential tilting gives
    P_exp(S >= t) = 1 - Phi((t - mu - sigma^2) / sigma).  Solving this
    expression for the requested mass gives a threshold without observing
    the rare maximum logits directly.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, tokens]")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0, 1)")
    if sample_count <= 1:
        raise ValueError("sample_count must exceed one")
    history_count = int(scores.shape[-1])
    active_samples = min(sample_count, history_count)
    sample_ids = torch.arange(
        active_samples, device=scores.device, dtype=torch.long
    )
    token_ids = (
        (2 * sample_ids + 1) * history_count
    ) // (2 * active_samples)
    sampled = scores.index_select(-1, token_ids).float()
    mean = sampled.mean(dim=-1)
    variance = sampled.var(dim=-1, unbiased=False).clamp_min(1.0e-12)
    standard_deviation = variance.sqrt()
    normal_quantile = torch.special.ndtri(
        torch.tensor(
            1.0 - target_mass,
            dtype=torch.float32,
            device=scores.device,
        )
    )
    boundary = (
        mean + variance + standard_deviation * normal_quantile
    )
    mask = scores.float() >= boundary[:, None]

    minimum = min(max(1, minimum_top_k), history_count)
    mask |= fixed_mask(scores, minimum)
    upper = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    if upper < history_count:
        over_limit = mask.sum(dim=-1) > upper
        if torch.any(over_limit):
            maximum_mask = fixed_mask(scores, upper)
            mask = torch.where(over_limit[:, None], maximum_mask, mask)
    return mask


def sampled_rank_mass_ladder_mask(
    scores: torch.Tensor,
    target_mass: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
    growth: float,
) -> torch.Tensor:
    """Choose the first sampled-rank threshold with measured proxy mass.

    Sampling estimates only score quantiles. The complete proxy scan measures
    softmax mass at every geometric candidate rung, so rare high logits cannot
    bias the mass estimate itself.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, tokens]")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0, 1)")
    if sample_count <= 1 or growth <= 1.0:
        raise ValueError("sample_count and growth must exceed one")
    history_count = int(scores.shape[-1])
    active_samples = min(sample_count, history_count)
    sample_ids = torch.arange(
        active_samples, device=scores.device, dtype=torch.long
    )
    token_ids = (
        (2 * sample_ids + 1) * history_count
    ) // (2 * active_samples)
    sorted_sample = torch.sort(
        scores.index_select(-1, token_ids).float(),
        dim=-1,
        descending=True,
    ).values
    maximum = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    rung_counts: list[int] = []
    count = min(max(1, minimum_top_k), maximum)
    while True:
        rung_counts.append(count)
        if count >= maximum:
            break
        next_count = min(maximum, max(count + 1, math.ceil(count * growth)))
        count = next_count

    proxy_weights = torch.softmax(scores.float(), dim=-1)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    unresolved = torch.ones(
        scores.shape[0], dtype=torch.bool, device=scores.device
    )
    for rung_count in rung_counts:
        sampled_keep = min(
            active_samples,
            max(1, math.ceil(rung_count * active_samples / history_count)),
        )
        boundary = sorted_sample[:, sampled_keep - 1]
        rung_mask = (
            torch.ones_like(scores, dtype=torch.bool)
            if rung_count >= history_count
            else scores.float() >= boundary[:, None]
        )
        measured_mass = proxy_weights.masked_fill(~rung_mask, 0.0).sum(dim=-1)
        accept = unresolved & (
            (measured_mass >= target_mass) | (rung_count == maximum)
        )
        selected = torch.where(accept[:, None], rung_mask, selected)
        unresolved &= ~accept
        if not bool(torch.any(unresolved)):
            break
    return selected


def affine_residual_bound_ladder_mask(
    proxy_scores: torch.Tensor,
    residual_risk: torch.Tensor,
    output_scale: torch.Tensor,
    tolerance: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
    growth: float,
) -> torch.Tensor:
    """Choose the first rank rung satisfying the affine-tail Cauchy bound.

    For each proposed tail ``T``, the proxy exponential weights are projected
    onto ``span{1, score}``.  The product of the unexplained weight norm and
    the post-output-projection Value-residual Frobenius norm bounds the
    remaining affine-closure error.  Sampling proposes rank thresholds only;
    the complete proxy scan evaluates the bound.
    """

    if proxy_scores.shape != residual_risk.shape or proxy_scores.ndim != 2:
        raise ValueError("scores and residual risk must share [rows, tokens]")
    if output_scale.shape != proxy_scores.shape[:-1]:
        raise ValueError("output_scale must have one value per score row")
    if tolerance <= 0.0 or sample_count <= 1 or growth <= 1.0:
        raise ValueError("tolerance must be positive and ladder settings valid")

    history_count = int(proxy_scores.shape[-1])
    active_samples = min(sample_count, history_count)
    sample_ids = torch.arange(
        active_samples, device=proxy_scores.device, dtype=torch.long
    )
    token_ids = (
        (2 * sample_ids + 1) * history_count
    ) // (2 * active_samples)
    sorted_sample = torch.sort(
        proxy_scores.index_select(-1, token_ids).float(),
        dim=-1,
        descending=True,
    ).values
    maximum = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    rung_counts: list[int] = []
    count = min(max(1, minimum_top_k), maximum)
    while True:
        rung_counts.append(count)
        if count >= maximum:
            break
        count = min(maximum, max(count + 1, math.ceil(count * growth)))

    shifted_scores = proxy_scores.float() - proxy_scores.float().amax(
        dim=-1, keepdim=True
    )
    proxy_weights = torch.exp(shifted_scores)
    proxy_partition = proxy_weights.sum(dim=-1).clamp_min(1.0e-20)
    selected = torch.zeros_like(proxy_scores, dtype=torch.bool)
    unresolved = torch.ones(
        proxy_scores.shape[0], dtype=torch.bool, device=proxy_scores.device
    )
    last_mask = selected
    for rung_count in rung_counts:
        sampled_keep = min(
            active_samples,
            max(1, math.ceil(rung_count * active_samples / history_count)),
        )
        boundary = sorted_sample[:, sampled_keep - 1]
        rung_mask = (
            torch.ones_like(proxy_scores, dtype=torch.bool)
            if rung_count >= history_count
            else proxy_scores.float() >= boundary[:, None]
        )
        tail = (~rung_mask).float()
        tail_count = tail.sum(dim=-1).clamp_min(1.0)
        score_sum = (tail * shifted_scores).sum(dim=-1)
        score_mean = score_sum / tail_count
        centered_scores = (shifted_scores - score_mean[:, None]) * tail
        tail_weights = proxy_weights * tail
        weight_mean = tail_weights.sum(dim=-1) / tail_count
        weight_slope = (
            (centered_scores * tail_weights).sum(dim=-1)
            / centered_scores.square().sum(dim=-1).clamp_min(1.0e-20)
        )
        unexplained_weights = tail * (
            proxy_weights
            - weight_mean[:, None]
            - weight_slope[:, None] * centered_scores
        )
        unexplained_weight_norm = torch.linalg.vector_norm(
            unexplained_weights, dim=-1
        )
        residual_energy = torch.linalg.vector_norm(
            residual_risk.float() * tail, dim=-1
        )
        relative_bound = (
            unexplained_weight_norm
            * residual_energy
            / proxy_partition
            / output_scale.float().clamp_min(1.0e-12)
        )
        accept = unresolved & (
            (relative_bound <= tolerance) | (rung_count == maximum)
        )
        selected = torch.where(accept[:, None], rung_mask, selected)
        unresolved &= ~accept
        last_mask = rung_mask
        if not bool(torch.any(unresolved)):
            break
    if torch.any(unresolved):
        selected = torch.where(unresolved[:, None], last_mask, selected)
    return selected


def exact_prefix_tail_ratio_mass_ladder_mask(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    target_mass: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
    growth: float,
) -> torch.Tensor:
    """Choose a proxy prefix using exact-prefix and sampled-tail mass.

    For a candidate prefix ``S``, exact QK is already needed by sparse
    attention, so its partition ``Z_S`` is available.  A systematic sample
    from the complement estimates the finite-population ratio

        R_T = sum_sample exp(exact) / sum_sample exp(proxy).

    Multiplying the full proxy tail partition by ``R_T`` corrects both an
    affine temperature error and the exponential moment of the residual.
    Only sampled tail scores and exact scores inside the largest tested
    prefix are consumed; no exact full-history scan or length threshold is
    used.
    """

    if exact_scores.shape != proxy_scores.shape or exact_scores.ndim != 2:
        raise ValueError("exact_scores and proxy_scores must share [rows, tokens]")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0, 1)")
    if sample_count <= 1 or growth <= 1.0:
        raise ValueError("sample_count and growth must exceed one")

    history_count = int(proxy_scores.shape[-1])
    active_samples = min(sample_count, history_count)
    sample_ids = torch.arange(
        active_samples, device=proxy_scores.device, dtype=torch.long
    )
    token_ids = (
        (2 * sample_ids + 1) * history_count
    ) // (2 * active_samples)
    sampled_proxy = proxy_scores.index_select(-1, token_ids).float()
    sampled_exact = exact_scores.index_select(-1, token_ids).float()
    sorted_sample = torch.sort(
        sampled_proxy, dim=-1, descending=True
    ).values

    maximum = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    rung_counts: list[int] = []
    count = min(max(1, minimum_top_k), maximum)
    while True:
        rung_counts.append(count)
        if count >= maximum:
            break
        count = min(maximum, max(count + 1, math.ceil(count * growth)))

    proxy_maximum = proxy_scores.float().amax(dim=-1, keepdim=True)
    proxy_weights = torch.exp(proxy_scores.float() - proxy_maximum)
    selected = torch.zeros_like(proxy_scores, dtype=torch.bool)
    unresolved = torch.ones(
        proxy_scores.shape[0], dtype=torch.bool, device=proxy_scores.device
    )
    last_mask = selected
    for rung_count in rung_counts:
        sampled_keep = min(
            active_samples,
            max(1, math.ceil(rung_count * active_samples / history_count)),
        )
        boundary = sorted_sample[:, sampled_keep - 1]
        rung_mask = (
            torch.ones_like(proxy_scores, dtype=torch.bool)
            if rung_count >= history_count
            else proxy_scores.float() >= boundary[:, None]
        )
        selected_log_partition = torch.logsumexp(
            exact_scores.float().masked_fill(~rung_mask, -torch.inf), dim=-1
        )
        proxy_tail_partition = (
            proxy_weights * (~rung_mask).float()
        ).sum(dim=-1)
        sampled_tail = ~rung_mask.index_select(-1, token_ids)
        sampled_tail_count = sampled_tail.sum(dim=-1)
        sampled_exact_tail_log_partition = torch.logsumexp(
            sampled_exact.masked_fill(~sampled_tail, -torch.inf), dim=-1
        )
        sampled_proxy_tail_log_partition = torch.logsumexp(
            sampled_proxy.masked_fill(~sampled_tail, -torch.inf), dim=-1
        )
        tail_log_ratio = (
            sampled_exact_tail_log_partition
            - sampled_proxy_tail_log_partition
        )
        estimated_tail_log_partition = (
            torch.log(proxy_tail_partition.clamp_min(1.0e-30))
            + proxy_maximum.squeeze(-1)
            + tail_log_ratio
        )
        estimated_mass = torch.sigmoid(
            selected_log_partition - estimated_tail_log_partition
        )
        estimated_mass = torch.where(
            (sampled_tail_count == 0) | (proxy_tail_partition == 0),
            torch.ones_like(estimated_mass),
            estimated_mass,
        )
        accept = unresolved & (
            (estimated_mass >= target_mass) | (rung_count == maximum)
        )
        selected = torch.where(accept[:, None], rung_mask, selected)
        unresolved &= ~accept
        last_mask = rung_mask
        if not bool(torch.any(unresolved)):
            break
    if torch.any(unresolved):
        selected = torch.where(unresolved[:, None], last_mask, selected)
    return selected


def interval_certified_mass_ladder_mask(
    scores: torch.Tensor,
    error_bound: torch.Tensor,
    target_mass: float,
    minimum_top_k: int,
    maximum_top_k: int,
    *,
    sample_count: int,
    growth: float,
) -> torch.Tensor:
    """Choose a proxy-score prefix with a lower bound on true softmax mass.

    If ``abs(true_score - scores) <= error_bound`` elementwise, then for a
    prefix A the returned certificate is

        sum_A exp(score - error) /
        (sum_A exp(score - error) + sum_not_A exp(score + error)).

    The first sampled-rank ladder rung whose certificate reaches the target
    is selected. The ordering remains a pure proxy-score prefix; uncertainty
    changes only its size.
    """

    if scores.ndim != 2 or error_bound.shape != scores.shape:
        raise ValueError("scores and error_bound must share [rows, tokens]")
    if torch.any(error_bound < 0):
        raise ValueError("score error bounds must be nonnegative")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target mass must lie in (0, 1)")
    if sample_count <= 1 or growth <= 1.0:
        raise ValueError("sample_count and growth must exceed one")

    history_count = int(scores.shape[-1])
    active_samples = min(sample_count, history_count)
    sample_ids = torch.arange(
        active_samples, device=scores.device, dtype=torch.long
    )
    token_ids = (
        (2 * sample_ids + 1) * history_count
    ) // (2 * active_samples)
    sorted_sample = torch.sort(
        scores.index_select(-1, token_ids).float(),
        dim=-1,
        descending=True,
    ).values
    maximum = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    count = min(max(1, minimum_top_k), maximum)
    rung_counts: list[int] = []
    while True:
        rung_counts.append(count)
        if count >= maximum:
            break
        count = min(maximum, max(count + 1, math.ceil(count * growth)))

    scores32 = scores.float()
    bound32 = error_bound.float()
    reference = (scores32 + bound32).amax(dim=-1, keepdim=True)
    lower_weights = torch.exp(scores32 - bound32 - reference)
    upper_weights = torch.exp(scores32 + bound32 - reference)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    unresolved = torch.ones(
        scores.shape[0], dtype=torch.bool, device=scores.device
    )
    last_mask = selected
    for rung_count in rung_counts:
        sampled_keep = min(
            active_samples,
            max(1, math.ceil(rung_count * active_samples / history_count)),
        )
        boundary = sorted_sample[:, sampled_keep - 1]
        rung_mask = scores32 >= boundary[:, None]
        if rung_count == maximum:
            rung_mask = fixed_mask(scores32, maximum)
        lower_selected = (lower_weights * rung_mask).sum(dim=-1)
        upper_tail = (upper_weights * (~rung_mask)).sum(dim=-1)
        certified_mass = lower_selected / (
            lower_selected + upper_tail
        ).clamp_min(1.0e-30)
        accepted = unresolved & (certified_mass >= target_mass)
        selected = torch.where(accepted[:, None], rung_mask, selected)
        unresolved &= ~accepted
        last_mask = rung_mask
        if not torch.any(unresolved):
            break
    if torch.any(unresolved):
        selected = torch.where(unresolved[:, None], last_mask, selected)
    return selected


def stratified_jittered_sample_indices(
    history_count: int,
    sample_count: int,
    device: torch.device,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Return one deterministic, non-periodic sample inside each stratum."""

    active_sample_count = min(max(1, sample_count), history_count)
    sample = torch.arange(
        active_sample_count,
        device=device,
        dtype=torch.long,
    )
    starts = torch.div(
        sample * history_count,
        active_sample_count,
        rounding_mode="floor",
    )
    stops = torch.div(
        (sample + 1) * history_count,
        active_sample_count,
        rounding_mode="floor",
    )
    widths = (stops - starts).clamp_min(1)
    hashed = (
        sample * 1_103_515_245 + int(seed) + 1_013_904_223
    ).remainder(2_147_483_647)
    return starts + hashed.remainder(widths)


def affine_calibrate_scores(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a positive per-query affine map on a stratified token sample."""

    history_count = int(exact_scores.shape[-1])
    active_sample_count = min(max(2, sample_count), history_count)
    sample_indices = stratified_jittered_sample_indices(
        history_count,
        active_sample_count,
        exact_scores.device,
    )
    proxy_sample = proxy_scores.index_select(1, sample_indices).float()
    exact_sample = exact_scores.index_select(1, sample_indices).float()
    proxy_mean = proxy_sample.mean(dim=-1, keepdim=True)
    exact_mean = exact_sample.mean(dim=-1, keepdim=True)
    centered_proxy = proxy_sample - proxy_mean
    centered_exact = exact_sample - exact_mean
    slope = (
        (centered_proxy * centered_exact).sum(dim=-1, keepdim=True)
        / centered_proxy.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    ).clamp(0.25, 4.0)
    intercept = exact_mean - slope * proxy_mean
    calibrated = proxy_scores.float() * slope + intercept
    return calibrated, slope.squeeze(-1), intercept.squeeze(-1)


def crossfit_affine_score_rmse(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate affine proxy error from one stratified exact-QK sample.

    Even sample positions fit an affine map evaluated on odd positions, and
    vice versa.  The resulting RMSE is observable at decode time and avoids
    measuring calibration quality on the same points used to fit the map.
    """

    history_count = int(exact_scores.shape[-1])
    active_sample_count = min(max(4, sample_count), history_count)
    sample_indices = stratified_jittered_sample_indices(
        history_count,
        active_sample_count,
        exact_scores.device,
    )
    proxy_sample = proxy_scores.index_select(1, sample_indices).float()
    exact_sample = exact_scores.index_select(1, sample_indices).float()
    residual_parts: list[torch.Tensor] = []
    for heldout_parity in (0, 1):
        heldout = torch.arange(
            heldout_parity,
            active_sample_count,
            2,
            device=exact_scores.device,
        )
        fit = torch.arange(
            1 - heldout_parity,
            active_sample_count,
            2,
            device=exact_scores.device,
        )
        fit_proxy = proxy_sample.index_select(1, fit)
        fit_exact = exact_sample.index_select(1, fit)
        proxy_mean = fit_proxy.mean(dim=-1, keepdim=True)
        exact_mean = fit_exact.mean(dim=-1, keepdim=True)
        slope = (
            (
                (fit_proxy - proxy_mean)
                * (fit_exact - exact_mean)
            ).sum(dim=-1, keepdim=True)
            / (fit_proxy - proxy_mean)
            .square()
            .sum(dim=-1, keepdim=True)
            .clamp_min(1.0e-12)
        ).clamp(0.25, 4.0)
        intercept = exact_mean - slope * proxy_mean
        residual_parts.append(
            exact_sample.index_select(1, heldout)
            - (
                slope * proxy_sample.index_select(1, heldout)
                + intercept
            )
        )
    residual = torch.cat(residual_parts, dim=-1)
    rmse = residual.square().mean(dim=-1).sqrt()
    exact_std = (
        exact_sample - exact_sample.mean(dim=-1, keepdim=True)
    ).square().mean(dim=-1).sqrt()
    return rmse, exact_std


def crossfit_affine_softmax_kl(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Estimate softmax KL without fitting and scoring on the same probes."""

    if exact_scores.shape != proxy_scores.shape or exact_scores.ndim != 2:
        raise ValueError("exact and proxy scores must share shape [rows, tokens]")
    history_count = int(exact_scores.shape[-1])
    active_sample_count = min(max(4, sample_count), history_count)
    sample_indices = stratified_jittered_sample_indices(
        history_count,
        active_sample_count,
        exact_scores.device,
    )
    proxy_sample = proxy_scores.index_select(1, sample_indices).float()
    exact_sample = exact_scores.index_select(1, sample_indices).float()
    folds: list[torch.Tensor] = []
    for heldout_parity in (0, 1):
        heldout = torch.arange(
            heldout_parity,
            active_sample_count,
            2,
            device=exact_scores.device,
        )
        fit = torch.arange(
            1 - heldout_parity,
            active_sample_count,
            2,
            device=exact_scores.device,
        )
        fit_proxy = proxy_sample.index_select(1, fit)
        fit_exact = exact_sample.index_select(1, fit)
        proxy_mean = fit_proxy.mean(dim=-1, keepdim=True)
        exact_mean = fit_exact.mean(dim=-1, keepdim=True)
        centered_proxy = fit_proxy - proxy_mean
        slope = (
            (
                centered_proxy
                * (fit_exact - exact_mean)
            ).sum(dim=-1, keepdim=True)
            / centered_proxy.square().sum(dim=-1, keepdim=True).clamp_min(
                1.0e-12
            )
        ).clamp(0.25, 4.0)
        intercept = exact_mean - slope * proxy_mean
        heldout_exact = exact_sample.index_select(1, heldout)
        heldout_proxy = (
            slope * proxy_sample.index_select(1, heldout) + intercept
        )
        exact_log_probability = torch.log_softmax(heldout_exact, dim=-1)
        proxy_log_probability = torch.log_softmax(heldout_proxy, dim=-1)
        folds.append(
            (
                exact_log_probability.exp()
                * (exact_log_probability - proxy_log_probability)
            ).sum(dim=-1).clamp_min(0.0)
        )
    return torch.stack(folds, dim=-1).mean(dim=-1)


def fixed_mask(priority: torch.Tensor, top_k: int) -> torch.Tensor:
    active_top_k = min(max(1, top_k), int(priority.shape[-1]))
    indices = torch.topk(
        priority, k=active_top_k, dim=-1, sorted=False
    ).indices
    mask = torch.zeros_like(priority, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def coverage_mask(
    priority: torch.Tensor,
    target: float,
    minimum_top_k: int,
    maximum_top_k: int,
) -> torch.Tensor:
    """Choose the smallest prefix of descending priority mass per query."""

    sorted_priority, _ = torch.sort(priority, dim=-1, descending=True)
    shifted = sorted_priority - sorted_priority[:, :1]
    sorted_mass = torch.exp(shifted)
    cumulative = torch.cumsum(sorted_mass, dim=-1)
    required = target * cumulative[:, -1:]
    counts = torch.sum(cumulative < required, dim=-1) + 1
    upper = int(priority.shape[-1]) if maximum_top_k <= 0 else min(
        int(priority.shape[-1]), maximum_top_k
    )
    counts = counts.clamp(min=max(1, minimum_top_k), max=upper)
    thresholds = sorted_priority.gather(1, (counts - 1)[:, None])
    return priority >= thresholds


def histogram_coverage_mask(
    priority: torch.Tensor,
    target: float,
    minimum_top_k: int,
    maximum_top_k: int,
    bins: int,
    logit_range: float = 32.0,
) -> torch.Tensor:
    """Cover proxy softmax mass by selecting complete descending logit bins."""

    if priority.ndim != 2:
        raise ValueError("priority must have shape [queries, tokens]")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly inside (0, 1)")
    if bins <= 1 or logit_range <= 0.0:
        raise ValueError("histogram dimensions must be positive")
    history_count = int(priority.shape[-1])
    shifted = priority.float() - priority.float().amax(
        dim=-1, keepdim=True
    )
    weights = torch.exp(shifted)
    bin_width = logit_range / bins
    bin_ids = torch.floor(
        (-shifted).clamp(min=0.0, max=logit_range - 1.0e-6)
        / bin_width
    ).long()
    histogram = torch.zeros(
        priority.shape[0],
        bins,
        dtype=torch.float32,
        device=priority.device,
    )
    histogram.scatter_add_(1, bin_ids, weights)
    cumulative = torch.cumsum(histogram, dim=-1)
    required = target * cumulative[:, -1:]
    boundary_bins = torch.sum(cumulative < required, dim=-1)
    mask = bin_ids <= boundary_bins[:, None]

    minimum = min(max(1, minimum_top_k), history_count)
    if minimum > 1:
        minimum_mask = fixed_mask(priority, minimum)
        mask |= minimum_mask
    upper = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    if upper < history_count:
        over_limit = mask.sum(dim=-1) > upper
        if torch.any(over_limit):
            maximum_mask = fixed_mask(priority, upper)
            mask = torch.where(over_limit[:, None], maximum_mask, mask)
    return mask


def relative_tail_risk_mask(
    priority: torch.Tensor,
    proxy_scores: torch.Tensor,
    output_scale: torch.Tensor,
    tolerance: float,
    minimum_top_k: int,
    maximum_top_k: int,
) -> torch.Tensor:
    """Select the smallest per-head prefix with bounded proxy tail risk."""

    sorted_priority, _ = torch.sort(priority, dim=-1, descending=True)
    log_partition = torch.logsumexp(proxy_scores.float(), dim=-1)
    maximum = sorted_priority[:, 0]
    sorted_risk = torch.exp(sorted_priority - maximum[:, None])
    cumulative = torch.cumsum(sorted_risk, dim=-1)
    total = cumulative[:, -1]
    allowed = (
        tolerance
        * output_scale.float().clamp_min(1.0e-12)
        * torch.exp(log_partition - maximum)
    )
    required = (total - allowed).clamp_min(0.0)
    counts = torch.sum(cumulative < required[:, None], dim=-1) + 1
    upper = int(priority.shape[-1]) if maximum_top_k <= 0 else min(
        int(priority.shape[-1]), maximum_top_k
    )
    counts = counts.clamp(min=max(1, minimum_top_k), max=upper)
    thresholds = sorted_priority.gather(1, (counts - 1)[:, None])
    return priority >= thresholds


def relative_tail_rss_mask(
    priority: torch.Tensor,
    proxy_scores: torch.Tensor,
    output_scale: torch.Tensor,
    tolerance: float,
    safety_factor: float,
    minimum_top_k: int,
    maximum_top_k: int,
) -> torch.Tensor:
    """Select a prefix whose predicted RMS Value-tail error is small."""

    if priority.shape != proxy_scores.shape or priority.ndim != 2:
        raise ValueError("priority and proxy scores must align by query/token")
    if output_scale.shape != priority.shape[:-1]:
        raise ValueError("output scale must have one value per query")
    if tolerance <= 0.0 or safety_factor <= 0.0:
        raise ValueError("RSS tolerance and safety factor must be positive")
    normalized_log_risk = (
        priority.float()
        - torch.logsumexp(proxy_scores.float(), dim=-1, keepdim=True)
    )
    sorted_log_risk, sorted_indices = torch.sort(
        normalized_log_risk, dim=-1, descending=True
    )
    squared_contributions = torch.exp(2.0 * sorted_log_risk)
    cumulative = torch.cumsum(squared_contributions, dim=-1)
    tail_squared = (cumulative[:, -1:] - cumulative).clamp_min(0.0)
    allowed_squared = (
        tolerance * output_scale.float() / safety_factor
    ).square()[:, None]
    counts = torch.sum(tail_squared > allowed_squared, dim=-1) + 1
    history_count = int(priority.shape[-1])
    upper = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    counts = counts.clamp(
        min=min(max(1, minimum_top_k), history_count), max=upper
    )
    ranks = (
        torch.arange(history_count, device=priority.device)[None, :]
        < counts[:, None]
    )
    mask = torch.zeros_like(priority, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, ranks)
    return mask


def boundary_crossing_probability(
    priority: torch.Tensor,
    selected_mask: torch.Tensor,
    score_rmse: torch.Tensor,
) -> torch.Tensor:
    """Approximate each token's probability of crossing the prefix boundary.

    Independent score errors with standard deviation ``score_rmse`` give a
    pairwise score-difference deviation of ``sqrt(2) * score_rmse``.  The
    Gaussian tail therefore reduces to ``0.5 * erfc(gap / (2 * rmse))``.
    This is a numerical decision rule, not a length- or task-specific router.
    """

    if priority.ndim != 2 or selected_mask.shape != priority.shape:
        raise ValueError("priority and selected mask must align")
    if score_rmse.shape == priority.shape[:-1]:
        token_rmse = score_rmse.float()[:, None]
    elif score_rmse.shape == priority.shape:
        token_rmse = score_rmse.float()
    else:
        raise ValueError(
            "score RMSE must have one value per query or query/token"
        )
    selected_priority = priority.float().masked_fill(
        ~selected_mask, torch.inf
    )
    threshold = selected_priority.amin(dim=-1, keepdim=True)
    gap = (priority.float() - threshold).abs()
    denominator = 2.0 * token_rmse.clamp_min(1.0e-12)
    probability = 0.5 * torch.erfc(gap / denominator)
    return torch.where(
        token_rmse > 0.0,
        probability,
        torch.zeros_like(probability),
    )


def minimum_refinement_mask(
    squared_crossing_contribution: torch.Tensor,
    allowed_squared: torch.Tensor,
) -> torch.Tensor:
    """Refine the smallest largest-risk prefix meeting a crossing budget."""

    if squared_crossing_contribution.ndim != 2:
        raise ValueError("crossing contributions must be [query, token]")
    if allowed_squared.shape != squared_crossing_contribution.shape[:-1]:
        raise ValueError("allowed squared error must have one value per query")
    contribution = squared_crossing_contribution.float().clamp_min(0.0)
    sorted_contribution, sorted_indices = torch.sort(
        contribution, dim=-1, descending=True
    )
    cumulative = torch.cumsum(sorted_contribution, dim=-1)
    total = cumulative[:, -1]
    tail_after = (total[:, None] - cumulative).clamp_min(0.0)
    needs_refinement = total > allowed_squared.float()
    counts = torch.sum(
        tail_after > allowed_squared.float()[:, None], dim=-1
    ) + 1
    counts = torch.where(needs_refinement, counts, torch.zeros_like(counts))
    ranks = (
        torch.arange(
            contribution.shape[-1], device=contribution.device
        )[None, :]
        < counts[:, None]
    )
    mask = torch.zeros_like(contribution, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, ranks)
    return mask


def progressive_error_balanced_masks(
    base_scores: torch.Tensor,
    refined_scores: torch.Tensor,
    base_joint_risk: torch.Tensor,
    refined_joint_risk: torch.Tensor,
    base_score_rmse: torch.Tensor,
    refined_score_rmse: torch.Tensor,
    output_scale: torch.Tensor,
    tolerance: float,
    safety_factor: float,
    minimum_top_k: int,
    maximum_top_k: int,
    rounds: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly spend error on omission and score-boundary uncertainty.

    The squared layer budget is split equally between omitted Value-tail risk
    and proxy-induced boundary crossings.  A low-rate index scans every token;
    higher-rate residual bit-planes are fetched only for tokens whose expected
    crossing contribution is needed to satisfy the second half of the budget.
    """

    tensors = (
        refined_scores,
        base_joint_risk,
        refined_joint_risk,
    )
    if base_scores.ndim != 2 or any(
        tensor.shape != base_scores.shape for tensor in tensors
    ):
        raise ValueError("progressive score and risk tensors must align")
    if base_score_rmse.shape != base_scores.shape[:-1] or (
        refined_score_rmse.shape != base_scores.shape[:-1]
    ):
        raise ValueError("score RMSE must have one value per query")
    if output_scale.shape != base_scores.shape[:-1]:
        raise ValueError("output scale must have one value per query")
    if rounds <= 0:
        raise ValueError("progressive refinement rounds must be positive")

    component_tolerance = tolerance / math.sqrt(2.0)
    allowed_squared = (
        component_tolerance
        * output_scale.float().clamp_min(1.0e-12)
        / safety_factor
    ).square()
    refinement_mask = torch.zeros_like(base_scores, dtype=torch.bool)
    selection_mask = torch.zeros_like(base_scores, dtype=torch.bool)

    for _ in range(rounds):
        mixed_scores = torch.where(
            refinement_mask, refined_scores, base_scores
        )
        mixed_risk = torch.where(
            refinement_mask, refined_joint_risk, base_joint_risk
        )
        mixed_priority = mixed_scores + torch.log(
            mixed_risk.float().clamp_min(1.0e-30)
        )
        selection_mask = relative_tail_rss_mask(
            mixed_priority,
            mixed_scores,
            output_scale,
            component_tolerance,
            safety_factor,
            minimum_top_k,
            maximum_top_k,
        )
        crossing_probability = boundary_crossing_probability(
            mixed_priority,
            selection_mask,
            torch.where(
                refinement_mask,
                refined_score_rmse[:, None],
                base_score_rmse[:, None],
            ),
        )
        weighted_risk = (
            torch.softmax(mixed_scores.float(), dim=-1) * mixed_risk.float()
        )
        crossing_squared = (
            crossing_probability * weighted_risk.square()
        ).masked_fill(refinement_mask, 0.0)
        additional = minimum_refinement_mask(
            crossing_squared, allowed_squared
        )
        new_refinement = refinement_mask | additional
        if torch.equal(new_refinement, refinement_mask):
            break
        refinement_mask = new_refinement

    mixed_scores = torch.where(refinement_mask, refined_scores, base_scores)
    mixed_risk = torch.where(
        refinement_mask, refined_joint_risk, base_joint_risk
    )
    mixed_priority = mixed_scores + torch.log(
        mixed_risk.float().clamp_min(1.0e-30)
    )
    selection_mask = relative_tail_rss_mask(
        mixed_priority,
        mixed_scores,
        output_scale,
        component_tolerance,
        safety_factor,
        minimum_top_k,
        maximum_top_k,
    )
    unresolved_crossing = boundary_crossing_probability(
        mixed_priority,
        selection_mask,
        torch.where(
            refinement_mask,
            refined_score_rmse[:, None],
            base_score_rmse[:, None],
        ),
    )
    unresolved_crossing_rss = torch.sqrt(
        (
            unresolved_crossing
            * (
                torch.softmax(mixed_scores.float(), dim=-1)
                * mixed_risk.float()
            ).square()
        ).sum(dim=-1)
    )
    return selection_mask, refinement_mask, unresolved_crossing_rss


def scalar_residual_rss_mask(
    proxy_scores: torch.Tensor,
    residual_risk: torch.Tensor,
    output_scale: torch.Tensor,
    tolerance: float,
    safety_factor: float,
    minimum_top_k: int,
    maximum_top_k: int,
    statistic: str,
) -> torch.Tensor:
    """Select tokens by a sort-free bound on expected Value-tail error.

    If every omitted proxy probability is at most lambda, then
    sum(omitted_probability ** 2) <= lambda. Multiplying by one per-head
    residual scale gives an interpretable RMS output-error bound.
    """

    if proxy_scores.shape != residual_risk.shape or proxy_scores.ndim != 2:
        raise ValueError("scores and residual risk must align by query/token")
    if output_scale.shape != proxy_scores.shape[:-1]:
        raise ValueError("output scale must have one value per query")
    if tolerance <= 0.0 or safety_factor <= 0.0:
        raise ValueError("scalar RSS parameters must be positive")
    if statistic == "rms":
        residual_scale = residual_risk.float().square().mean(dim=-1).sqrt()
    elif statistic == "p90":
        residual_scale = torch.quantile(
            residual_risk.float(), 0.90, dim=-1
        )
    elif statistic == "maximum":
        residual_scale = residual_risk.float().amax(dim=-1)
    else:
        raise ValueError("statistic must be rms, p90, or maximum")

    probabilities = torch.softmax(proxy_scores.float(), dim=-1)
    probability_limit = (
        tolerance
        * output_scale.float().clamp_min(1.0e-12)
        / (safety_factor * residual_scale.clamp_min(1.0e-12))
    ).square()
    mask = probabilities > probability_limit[:, None]

    history_count = int(proxy_scores.shape[-1])
    minimum = min(max(1, minimum_top_k), history_count)
    mask |= fixed_mask(proxy_scores, minimum)
    upper = history_count if maximum_top_k <= 0 else min(
        history_count, maximum_top_k
    )
    if upper < history_count:
        over_limit = mask.sum(dim=-1) > upper
        if torch.any(over_limit):
            maximum_mask = fixed_mask(proxy_scores, upper)
            mask = torch.where(over_limit[:, None], maximum_mask, mask)
    return mask


def piecewise_mean_value(
    value: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact block means and their token-aligned reconstruction."""

    if value.ndim != 2 or value.shape[0] <= 0:
        raise ValueError("value must have shape [tokens, dimensions]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    history_count = int(value.shape[0])
    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    value_padding = padded_count - history_count
    padded_value = torch.nn.functional.pad(
        value.float(), (0, 0, 0, value_padding)
    )
    blocks = padded_value.reshape(block_count, block_size, value.shape[-1])
    valid_counts = torch.full(
        (block_count,),
        block_size,
        dtype=blocks.dtype,
        device=blocks.device,
    )
    if value_padding:
        valid_counts[-1] -= value_padding
    means = blocks.sum(dim=1) / valid_counts[:, None]
    reconstructed = means.repeat_interleave(block_size, dim=0)[:history_count]
    return means, reconstructed


def global_floor_rss_mask(
    risk_priority: torch.Tensor,
    proxy_scores: torch.Tensor,
    layer_output_scale: torch.Tensor,
    *,
    floor_k: int,
    tolerance: float,
    safety_factor: float,
) -> torch.Tensor:
    """Minimize extra slots under a layer RSS risk budget and head floor."""

    if risk_priority.shape != proxy_scores.shape or risk_priority.ndim != 3:
        raise ValueError("risk priority and proxy scores must be [steps,H,N]")
    if layer_output_scale.shape != risk_priority.shape[:1]:
        raise ValueError("layer output scale must have shape [steps]")
    if floor_k <= 0 or tolerance <= 0.0 or safety_factor <= 0.0:
        raise ValueError("global RSS parameters must be positive")
    step_count, head_count, history_count = risk_priority.shape
    active_floor = min(floor_k, history_count)
    floor_indices = torch.topk(
        proxy_scores, k=active_floor, dim=-1, sorted=False
    ).indices
    base_mask = torch.zeros_like(proxy_scores, dtype=torch.bool)
    base_mask.scatter_(2, floor_indices, True)

    normalized_log_risk = risk_priority.float() - torch.logsumexp(
        proxy_scores.float(), dim=-1, keepdim=True
    )
    squared_contributions = torch.exp(2.0 * normalized_log_risk)
    flat_tail = squared_contributions.masked_fill(base_mask, 0.0).reshape(
        step_count, head_count * history_count
    )
    sorted_squared, sorted_indices = torch.sort(
        flat_tail, dim=-1, descending=True
    )
    cumulative = torch.cumsum(sorted_squared, dim=-1)
    total_tail = cumulative[:, -1]
    allowed = (
        tolerance * layer_output_scale.float() / safety_factor
    ).square()
    required_removal = (total_tail - allowed).clamp_min(0.0)
    extra_counts = torch.sum(
        cumulative < required_removal[:, None], dim=-1
    ) + (required_removal > 0.0).to(torch.long)
    extra_ranks = torch.arange(
        flat_tail.shape[-1], device=flat_tail.device
    )[None, :] < extra_counts[:, None]
    flat_extra_mask = torch.zeros_like(flat_tail, dtype=torch.bool)
    flat_extra_mask.scatter_(1, sorted_indices, extra_ranks)
    return (
        base_mask.reshape(step_count, -1) | flat_extra_mask
    ).reshape_as(base_mask)


def sampled_tail_partition_scale(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    selected_mask: torch.Tensor,
    sample_count: int = 256,
    top_count: int = 128,
) -> torch.Tensor:
    """Estimate exact/proxy tail partition ratio with top plus tail probes."""

    if exact_scores.shape != proxy_scores.shape or (
        selected_mask.shape != exact_scores.shape
    ):
        raise ValueError("tail partition tensors must align")
    scales = []
    history_count = int(exact_scores.shape[-1])
    for row in range(exact_scores.shape[0]):
        tail = ~selected_mask[row]
        tail_indices = torch.nonzero(tail, as_tuple=False).flatten()
        tail_population = int(tail_indices.numel())
        if tail_population == 0:
            scales.append(exact_scores.new_ones(()))
            continue
        maximum = torch.maximum(
            exact_scores[row].amax(), proxy_scores[row].amax()
        )
        exact_weight = torch.exp(exact_scores[row].float() - maximum)
        proxy_weight = torch.exp(proxy_scores[row].float() - maximum)
        active_top = min(top_count, tail_population)
        active_total = min(max(active_top, sample_count), tail_population)
        requested_sample = active_total - active_top
        top_indices = torch.topk(
            proxy_scores[row].masked_fill(~tail, -torch.inf),
            k=active_top,
            sorted=False,
        ).indices
        probe_mask = ~tail
        probe_mask = probe_mask.clone()
        probe_mask[top_indices] = True
        exact_partition_estimate = exact_weight.index_select(
            0, top_indices
        ).sum()
        remaining_indices = torch.nonzero(
            ~probe_mask, as_tuple=False
        ).flatten()
        remaining_population = int(remaining_indices.numel())
        active_sample = min(requested_sample, remaining_population)
        if active_sample:
            positions = stratified_jittered_sample_indices(
                remaining_population,
                active_sample,
                exact_scores.device,
            )
            sample_indices = remaining_indices.index_select(0, positions)
            exact_partition_estimate = (
                exact_partition_estimate
                + remaining_population
                * exact_weight.index_select(0, sample_indices).mean()
            )
        proxy_partition = (proxy_weight * tail.float()).sum().clamp_min(
            1.0e-20
        )
        scales.append(exact_partition_estimate / proxy_partition)
    return torch.stack(scales)


def approximate_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    value: torch.Tensor,
    reconstructed_value: torch.Tensor,
    mask: torch.Tensor,
    residual_sample_counts: tuple[int, ...] = (),
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = torch.maximum(
        exact_scores.amax(dim=-1), proxy_scores.amax(dim=-1)
    )
    exact_weights = torch.exp(exact_scores - maximum[:, None])
    proxy_weights = torch.exp(proxy_scores - maximum[:, None])
    full_output = (
        exact_weights @ value.float()
        / exact_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    )
    active = mask.to(exact_weights.dtype)
    selected_exact_weights = exact_weights * active
    selected_proxy_weights = proxy_weights * active
    selected_numerator = selected_exact_weights @ value.float()
    selected_partition = selected_exact_weights.sum(dim=-1)
    selected_exact_only = selected_numerator / selected_partition[
        :, None
    ].clamp_min(1.0e-20)
    proxy_numerator = proxy_weights @ reconstructed_value.float()
    proxy_partition = proxy_weights.sum(dim=-1)
    proxy_selected_reconstructed_numerator = (
        selected_proxy_weights @ reconstructed_value.float()
    )
    proxy_selected_partition = selected_proxy_weights.sum(dim=-1)
    proxy_tail_weights = proxy_weights * (1.0 - active)
    proxy_tail_partition = proxy_tail_weights.sum(dim=-1)
    exact_tail_partition = (
        exact_weights * (1.0 - active)
    ).sum(dim=-1)
    oracle_tail_partition_scale = (
        exact_tail_partition / proxy_tail_partition.clamp_min(1.0e-20)
    )
    sampled_tail_scale = sampled_tail_partition_scale(
        exact_scores,
        proxy_scores,
        mask,
        sample_count=256,
        top_count=128,
    )
    hybrid_denominator = (
        selected_partition + proxy_tail_partition
    )[:, None].clamp_min(1.0e-20)
    value_residual = value.float() - reconstructed_value.float()
    sketch_base_numerator = (
        selected_numerator
        + proxy_numerator
        - proxy_selected_reconstructed_numerator
    )
    proxy_tail_reconstructed_numerator = (
        proxy_numerator - proxy_selected_reconstructed_numerator
    )
    hybrid_sketch_partition_oracle = (
        selected_numerator
        + oracle_tail_partition_scale[:, None]
        * proxy_tail_reconstructed_numerator
    ) / (
        selected_partition
        + oracle_tail_partition_scale * proxy_tail_partition
    )[:, None].clamp_min(1.0e-20)
    hybrid_sketch_partition_sample256 = (
        selected_numerator
        + sampled_tail_scale[:, None]
        * proxy_tail_reconstructed_numerator
    ) / (
        selected_partition + sampled_tail_scale * proxy_tail_partition
    )[:, None].clamp_min(1.0e-20)
    global_value_mean = value.float().mean(dim=0)
    hybrid_global_mean = (
        selected_numerator
        + proxy_tail_partition[:, None] * global_value_mean[None, :]
    ) / hybrid_denominator
    block_mean_outputs: dict[str, torch.Tensor] = {}
    history_count = int(value.shape[0])
    for block_size in (64, 256):
        block_means, _ = piecewise_mean_value(value, block_size)
        block_count = int(block_means.shape[0])
        value_padding = block_count * block_size - history_count
        if value_padding:
            padded_tail_weights = torch.nn.functional.pad(
                proxy_tail_weights, (0, value_padding)
            )
        else:
            padded_tail_weights = proxy_tail_weights
        tail_block_weights = padded_tail_weights.reshape(
            proxy_tail_weights.shape[0], block_count, block_size
        ).sum(dim=-1)
        block_mean_outputs[f"hybrid_blockmean{block_size}"] = (
            selected_numerator + tail_block_weights @ block_means
        ) / hybrid_denominator
        tail_token_mask = 1.0 - active
        if value_padding:
            padded_tail_mask = torch.nn.functional.pad(
                tail_token_mask, (0, value_padding)
            )
            padded_residual = torch.nn.functional.pad(
                value_residual, (0, 0, 0, value_padding)
            )
        else:
            padded_tail_mask = tail_token_mask
            padded_residual = value_residual
        tail_mask_blocks = padded_tail_mask.reshape(
            proxy_tail_weights.shape[0], block_count, block_size
        )
        residual_blocks = padded_residual.reshape(
            block_count, block_size, value_residual.shape[-1]
        )
        tail_residual_sums = torch.einsum(
            "rnb,nbd->rnd", tail_mask_blocks, residual_blocks
        )
        tail_block_counts = tail_mask_blocks.sum(dim=-1).clamp_min(1.0)
        tail_block_mean_weights = tail_block_weights / tail_block_counts
        block_residual_correction = torch.einsum(
            "rn,rnd->rd", tail_block_mean_weights, tail_residual_sums
        )
        block_mean_outputs[
            f"hybrid_sketch_blockresidual{block_size}"
        ] = (
            sketch_base_numerator + block_residual_correction
        ) / hybrid_denominator
    hybrid_sketch = (
        selected_numerator
        + proxy_numerator
        - proxy_selected_reconstructed_numerator
    ) / (
        selected_partition + proxy_partition - proxy_selected_partition
    )[:, None].clamp_min(1.0e-20)
    exact_tail_numerator = (
        (exact_weights * (1.0 - active)) @ reconstructed_value.float()
    )
    exact_weight_sketch = (
        selected_numerator + exact_tail_numerator
    ) / exact_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    hybrid_full_value = (
        selected_numerator
        + (proxy_weights * (1.0 - active)) @ value.float()
    ) / (
        selected_partition + proxy_partition - proxy_selected_partition
    )[:, None].clamp_min(1.0e-20)
    proxy_selected_exact_numerator = selected_proxy_weights @ value.float()
    coherent_proxy_sketch = (
        proxy_selected_exact_numerator
        + proxy_numerator
        - proxy_selected_reconstructed_numerator
    ) / proxy_partition[:, None].clamp_min(1.0e-20)
    coherent_proxy_full_value = (
        proxy_weights @ value.float()
    ) / proxy_partition[:, None].clamp_min(1.0e-20)
    base_numerator = sketch_base_numerator
    outputs = {
        "selected_exact_only": selected_exact_only,
        "hybrid_sketch_partition_oracle": (
            hybrid_sketch_partition_oracle
        ),
        "hybrid_sketch_partition_sample256": (
            hybrid_sketch_partition_sample256
        ),
        "hybrid_global_mean": hybrid_global_mean,
        **block_mean_outputs,
        "hybrid_sketch": hybrid_sketch,
        "exact_weight_sketch": exact_weight_sketch,
        "hybrid_full_value": hybrid_full_value,
        "coherent_proxy_sketch": coherent_proxy_sketch,
        "coherent_proxy_full_value": coherent_proxy_full_value,
    }
    tail_count = (1.0 - active).sum(dim=-1).clamp_min(1.0)
    tail_mean_weight = proxy_tail_partition / tail_count
    global_residual_sum = value_residual.sum(dim=0)
    selected_residual_sum = active @ value_residual
    centered_residual_correction = tail_mean_weight[:, None] * (
        global_residual_sum[None, :] - selected_residual_sum
    )
    outputs["hybrid_sketch_centered_residual"] = (
        base_numerator + centered_residual_correction
    ) / hybrid_denominator
    tail_scores = proxy_scores.float() * (1.0 - active)
    tail_score_mean = tail_scores.sum(dim=-1) / tail_count
    centered_tail_scores = (
        proxy_scores.float() - tail_score_mean[:, None]
    ) * (1.0 - active)
    tail_score_variance_sum = centered_tail_scores.square().sum(dim=-1)
    affine_weight_slope = (
        (centered_tail_scores * proxy_tail_weights).sum(dim=-1)
        / tail_score_variance_sum.clamp_min(1.0e-20)
    )
    tail_residual_sum = (
        global_residual_sum[None, :] - selected_residual_sum
    )
    centered_score_residual_sum = centered_tail_scores @ value_residual
    affine_residual_correction = (
        tail_mean_weight[:, None] * tail_residual_sum
        + affine_weight_slope[:, None] * centered_score_residual_sum
    )
    outputs["hybrid_sketch_affine_residual"] = (
        base_numerator + affine_residual_correction
    ) / hybrid_denominator
    for requested_samples in residual_sample_counts:
        active_samples = min(max(1, requested_samples), history_count)
        sample_ids = torch.arange(
            active_samples, device=value.device, dtype=torch.long
        )
        token_ids = (
            (2 * sample_ids + 1) * history_count
        ) // (2 * active_samples)
        sampled_tail = (1.0 - active.index_select(-1, token_ids))
        sampled_residual = value_residual.index_select(0, token_ids)
        expansion = history_count / active_samples
        proxy_residual_numerator = expansion * torch.einsum(
            "rs,sd->rd",
            proxy_weights.index_select(-1, token_ids) * sampled_tail,
            sampled_residual,
        )
        exact_residual_numerator = expansion * torch.einsum(
            "rs,sd->rd",
            exact_weights.index_select(-1, token_ids) * sampled_tail,
            sampled_residual,
        )
        outputs[
            f"hybrid_sketch_proxyresidualsample{requested_samples}"
        ] = (
            (base_numerator + proxy_residual_numerator)
            / hybrid_denominator
        )
        outputs[
            f"hybrid_sketch_exactresidualsample{requested_samples}"
        ] = (
            (base_numerator + exact_residual_numerator)
            / hybrid_denominator
        )
    attention_mass = selected_exact_weights.sum(dim=-1) / exact_weights.sum(
        dim=-1
    ).clamp_min(1.0e-20)
    return (
        outputs,
        full_output,
        attention_mass,
        exact_weights,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    key_bit_levels = tuple(
        sorted(
            {
                int(item)
                for item in args.key_bit_levels.split(",")
                if item.strip()
            }
        )
    )
    if not key_bit_levels or key_bit_levels[0] != 0:
        raise ValueError("Key bit levels must include zero")
    if any(bits < 0 or bits > 8 for bits in key_bit_levels):
        raise ValueError("Key bit levels must lie in [0, 8]")
    targets = tuple(
        sorted({float(item) for item in args.coverage_targets.split(",")})
    )
    sampled_mass_samples = tuple(
        int(item)
        for item in args.sampled_mass_samples.split(",")
        if item.strip()
    )
    sampled_mass_aggregations = tuple(
        item.strip()
        for item in args.sampled_mass_aggregations.split(",")
        if item.strip()
    )
    if any(item <= 0 for item in sampled_mass_samples):
        raise ValueError("sampled mass sample counts must be positive")
    if any(
        item not in {"minimum", "median"}
        for item in sampled_mass_aggregations
    ):
        raise ValueError("invalid sampled mass aggregation")
    gaussian_mass_samples = tuple(
        int(item)
        for item in args.gaussian_mass_samples.split(",")
        if item.strip()
    )
    if any(item <= 1 for item in gaussian_mass_samples):
        raise ValueError("Gaussian mass sample counts must exceed one")
    mass_ladder_samples = tuple(
        int(item)
        for item in args.mass_ladder_samples.split(",")
        if item.strip()
    )
    if any(item <= 1 for item in mass_ladder_samples):
        raise ValueError("mass ladder sample counts must exceed one")
    if args.mass_ladder_growth <= 1.0:
        raise ValueError("mass ladder growth must exceed one")
    interval_mass_samples = tuple(
        int(item)
        for item in args.interval_mass_samples.split(",")
        if item.strip()
    )
    if any(item <= 1 for item in interval_mass_samples):
        raise ValueError("interval mass sample counts must exceed one")
    if not targets or targets[0] <= 0.0 or targets[-1] >= 1.0:
        raise ValueError("coverage targets must lie strictly inside (0, 1)")
    coverage_histogram_bins = tuple(
        sorted(
            {
                int(item)
                for item in args.coverage_histogram_bins.split(",")
                if item.strip()
            }
        )
    )
    if any(item <= 1 for item in coverage_histogram_bins):
        raise ValueError("coverage histogram sizes must exceed one")
    relative_risk_thresholds = tuple(
        sorted(
            {
                float(item)
                for item in args.relative_risk_thresholds.split(",")
                if item.strip()
            }
        )
    )
    if any(item <= 0.0 for item in relative_risk_thresholds):
        raise ValueError("relative risk thresholds must be positive")
    rss_relative_tolerances = tuple(
        sorted(
            {
                float(item)
                for item in args.rss_relative_tolerances.split(",")
                if item.strip()
            }
        )
    )
    rss_safety_factors = tuple(
        sorted(
            {
                float(item)
                for item in args.rss_safety_factors.split(",")
                if item.strip()
            }
        )
    )
    global_rss_tolerances = tuple(
        sorted(
            {
                float(item)
                for item in args.global_rss_tolerances.split(",")
                if item.strip()
            }
        )
    )
    balanced_rss_tolerances = tuple(
        sorted(
            {
                float(item)
                for item in args.balanced_rss_tolerances.split(",")
                if item.strip()
            }
        )
    )
    scalar_rss_tolerances = tuple(
        sorted(
            {
                float(item)
                for item in args.scalar_rss_tolerances.split(",")
                if item.strip()
            }
        )
    )
    affine_bound_tolerances = tuple(
        sorted(
            {
                float(item)
                for item in args.affine_bound_tolerances.split(",")
                if item.strip()
            }
        )
    )
    if any(item <= 0.0 for item in affine_bound_tolerances):
        raise ValueError("affine bound tolerances must be positive")
    scalar_rss_statistics = tuple(
        item.strip()
        for item in args.scalar_rss_statistics.split(",")
        if item.strip()
    )
    if any(item <= 0.0 for item in rss_relative_tolerances):
        raise ValueError("RSS relative tolerances must be positive")
    if not rss_safety_factors or any(
        item <= 0.0 for item in rss_safety_factors
    ):
        raise ValueError("RSS safety factors must be positive")
    if any(item <= 0.0 for item in global_rss_tolerances):
        raise ValueError("global RSS tolerances must be positive")
    if any(item <= 0.0 for item in balanced_rss_tolerances):
        raise ValueError("balanced RSS tolerances must be positive")
    if any(item <= 0.0 for item in scalar_rss_tolerances):
        raise ValueError("scalar RSS tolerances must be positive")
    if any(
        item not in {"rms", "p90", "maximum"}
        for item in scalar_rss_statistics
    ):
        raise ValueError("invalid scalar RSS statistic")
    if args.global_rss_floor_k <= 0:
        raise ValueError("global RSS floor must be positive")
    if args.fixed_top_k <= 0 or args.minimum_top_k <= 0:
        raise ValueError("top-k values must be positive")
    if args.score_calibration_samples <= 1:
        raise ValueError("score_calibration_samples must exceed one")
    if args.key_refinement_rate_budget < 0:
        raise ValueError("Key refinement rate budget cannot be negative")
    if args.key_refinement_rate_budget and (
        args.key_refinement_rate_budget <= args.key_rate_budget
    ):
        raise ValueError(
            "Key refinement rate budget must exceed the base rate budget"
        )
    if args.progressive_refinement_rounds <= 0:
        raise ValueError("progressive refinement rounds must be positive")
    if args.focus_progressive_balanced_rss and not (
        args.key_refinement_rate_budget
    ):
        raise ValueError(
            "progressive balanced RSS requires a Key refinement rate"
        )
    residual_sample_counts = tuple(
        sorted(
            {
                int(item)
                for item in args.value_residual_samples.split(",")
                if item.strip()
            }
        )
    )
    if any(item <= 0 for item in residual_sample_counts):
        raise ValueError("Value residual sample counts must be positive")
    fixed_top_ks = tuple(
        sorted(
            {
                args.fixed_top_k,
                *(
                    int(item)
                    for item in args.fixed_top_ks.split(",")
                    if item.strip()
                ),
            }
        )
    )
    if fixed_top_ks[0] <= 0:
        raise ValueError("fixed top-k values must be positive")
    global_top_ks = tuple(
        sorted(
            {
                int(item)
                for item in args.global_top_ks.split(",")
                if item.strip()
            }
        )
    )
    if global_top_ks and global_top_ks[0] <= 0:
        raise ValueError("global top-k values must be positive")
    global_floor_fractions = tuple(
        sorted(
            {
                float(item)
                for item in args.global_floor_fractions.split(",")
                if item.strip()
            }
        )
    )
    if any(not 0.0 < item < 1.0 for item in global_floor_fractions):
        raise ValueError("global floor fractions must lie strictly inside (0, 1)")
    requested_global_priorities = {
        item.strip()
        for item in args.global_priority_names.split(",")
        if item.strip()
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace, map_location="cpu", weights_only=False, mmap=True)
    raw_prefill_queries = payload.get("prefill_query_tail", {})
    if not raw_prefill_queries:
        raw_prefill_queries = payload.get("prefill_queries", {})
    prefill_queries_by_layer = {
        int(layer): query for layer, query in raw_prefill_queries.items()
    }
    declared_prefill_tokens = int(
        payload.get("query_calibration_tokens", 0)
        or payload.get("config", {}).get("prefill_query_tail_tokens", 0)
    )
    requested_prefill_tokens = (
        int(args.query_factor_prefill_tokens)
        if args.query_factor_prefill_tokens > 0
        else declared_prefill_tokens
    )
    if args.query_factor_source != "decode" and requested_prefill_tokens <= 0:
        raise ValueError(
            "prefill-based QK factors require --query_factor_prefill_tokens "
            "or a positive trace query_calibration_tokens declaration"
        )
    records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    state_by_layer: dict[int, dict[str, Any]] = {}
    for record in payload["records"]:
        layer = int(record["layer"])
        records_by_layer[layer].append(record)
        if record.get("key") is not None and record.get("value") is not None:
            state_by_layer.setdefault(layer, record)

    model_root = str(
        args.model_name_or_path
        or payload.get("config", {}).get("model_name_or_path", "")
    )
    topic = str(payload.get("config", {}).get("topic", args.trace.stem))
    detail_rows: list[dict[str, Any]] = []
    score_calibration_rows: list[dict[str, Any]] = []
    key_allocation_rows: list[dict[str, Any]] = []
    oas_alpha_values: list[float] = []

    for layer in sorted(records_by_layer):
        if layer not in state_by_layer:
            continue
        records = sorted(records_by_layer[layer], key=lambda row: int(row["step"]))
        state_record = state_by_layer[layer]
        key_all = state_record["key"].to(device).float()[0]
        value_all = state_record["value"].to(device).float()[0]
        scaling = float(state_record["scaling"])
        decode_query = torch.stack(
            [record["query"].to(device).float()[0, :, 0, :] for record in records],
            dim=0,
        )
        prefill_query: torch.Tensor | None = None
        if args.query_factor_source != "decode":
            if layer not in prefill_queries_by_layer:
                raise ValueError(f"trace has no prefill Query tail for layer {layer}")
            raw_prefill = prefill_queries_by_layer[layer]
            if (
                raw_prefill.ndim != 4
                or raw_prefill.shape[0] != 1
                or raw_prefill.shape[-1] != key_all.shape[-1]
            ):
                raise ValueError(
                    f"invalid prefill Query shape for layer {layer}: "
                    f"{tuple(raw_prefill.shape)}"
                )
            if raw_prefill.shape[-2] < requested_prefill_tokens:
                raise ValueError(
                    f"layer {layer} has only {raw_prefill.shape[-2]} prefill "
                    f"Queries, requested {requested_prefill_tokens}"
                )
            prefill_query = (
                raw_prefill[0, :, -requested_prefill_tokens:, :]
                .permute(1, 0, 2)
                .contiguous()
                .to(device)
                .float()
            )
        query_head_count = int(decode_query.shape[1])
        kv_head_count = int(key_all.shape[0])
        query_groups = query_head_count // kv_head_count
        history_count = int(key_all.shape[1])
        projection = load_output_projection(model_root, layer, device)

        condition_outputs: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_full: list[torch.Tensor] = []
        condition_counts: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_refinement_counts: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_unresolved_crossing_rss: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_tail_score_estimate: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_tail_score_standard_error: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_tail_score_actual: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_tail_score_first_order: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_attention_mass: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_risk_mass: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_residual_rss: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_proxy_residual_rss_absolute: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        condition_exact_residual_rss_absolute: dict[
            tuple[str, str], list[torch.Tensor]
        ] = defaultdict(list)
        head_states: list[dict[str, torch.Tensor]] = []

        for kv_head in range(kv_head_count):
            key = key_all[kv_head]
            value = value_all[kv_head]
            head_slice = slice(
                kv_head * query_groups, (kv_head + 1) * query_groups
            )
            queries = decode_query[:, head_slice]
            flat_queries = queries.reshape(-1, queries.shape[-1])
            head_prefill_queries = (
                None if prefill_query is None else prefill_query[:, head_slice]
            )
            calibration_queries = qk_calibration_queries(
                queries,
                head_prefill_queries,
                args.query_factor_source,
            )
            query_factor, key_factor, balanced_singular_values = qk_balanced_factors(
                key[:: args.key_sample_stride],
                calibration_queries,
                args.query_shrinkage,
            )
            key_coordinates = key @ key_factor
            projected_calibration_queries = calibration_queries @ query_factor
            projected_queries = flat_queries @ query_factor
            if args.key_allocation_query_source == "basis":
                projected_allocation_queries = (
                    projected_calibration_queries
                )
            else:
                first_decode_queries = queries[:1].reshape(
                    -1, queries.shape[-1]
                )
                if args.key_allocation_query_source == "decode_first":
                    allocation_queries = first_decode_queries
                else:
                    if head_prefill_queries is None:
                        raise ValueError(
                            "prefill_decode_first allocation requires "
                            "prefill Queries"
                        )
                    allocation_queries = torch.cat(
                        (
                            head_prefill_queries.reshape(
                                -1, head_prefill_queries.shape[-1]
                            ),
                            first_decode_queries,
                        ),
                        dim=0,
                    )
                projected_allocation_queries = (
                    allocation_queries @ query_factor
                )
            if args.key_allocation_objective == "oas_qk_mse":
                oas_alpha_values.append(
                    float(
                        oas_query_metric_parameters(
                            projected_allocation_queries
                        )[0]
                    )
                )
            bands = key_quantization_candidates(
                key_coordinates,
                projected_allocation_queries,
                args.key_quantizer,
                key_bit_levels,
            )
            key_distortion = key_allocation_distortion(
                key_coordinates,
                projected_allocation_queries,
                bands,
                args.key_allocation_objective,
                balanced_singular_values,
            )
            allocation = allocate_bits(
                key_distortion,
                args.key_rate_budget,
                key_bit_levels,
                include_scale_metadata=True,
            )
            key_allocation_rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "rate_budget": args.key_rate_budget,
                    "allocation": "-".join(str(bits) for bits in allocation),
                    "payload_bits_per_token": 16 * sum(allocation),
                }
            )
            reconstructed_key = reconstruct(bands, allocation)
            approximate_queries = torch.stack(
                [query_int8(query) for query in projected_queries], dim=0
            )
            exact_scores = flat_queries @ key.T * scaling
            proxy_scores = approximate_queries.float() @ reconstructed_key.T * scaling
            exact_score_error_bound = (exact_scores - proxy_scores).abs()
            key_residual_norm = torch.linalg.vector_norm(
                key_coordinates.float() - reconstructed_key.float(), dim=-1
            )
            reconstructed_key_norm = torch.linalg.vector_norm(
                reconstructed_key.float(), dim=-1
            )
            query_norm = torch.linalg.vector_norm(
                projected_queries.float(), dim=-1
            )
            query_quantization_norm = torch.linalg.vector_norm(
                projected_queries.float() - approximate_queries.float(),
                dim=-1,
            )
            cauchy_score_error_bound = scaling * (
                query_norm[:, None] * key_residual_norm[None, :]
                + query_quantization_norm[:, None]
                * reconstructed_key_norm[None, :]
            )
            calibrated_proxy_scores, calibration_slope, calibration_intercept = (
                affine_calibrate_scores(
                    exact_scores,
                    proxy_scores,
                    args.score_calibration_samples,
                )
            )
            uncalibrated_rmse = torch.sqrt(
                torch.mean((proxy_scores - exact_scores).square(), dim=-1)
            )
            calibrated_rmse = torch.sqrt(
                torch.mean(
                    (calibrated_proxy_scores - exact_scores).square(), dim=-1
                )
            )
            sampled_crossfit_rmse, sampled_exact_score_std = (
                crossfit_affine_score_rmse(
                    exact_scores,
                    proxy_scores,
                    args.score_calibration_samples,
                )
            )
            sampled_crossfit_softmax_kl = crossfit_affine_softmax_kl(
                exact_scores,
                proxy_scores,
                args.score_calibration_samples,
            )
            calibrated_score_uncertainty, conformal_uncertainty_scale = (
                conformal_score_uncertainty(
                    exact_scores,
                    calibrated_proxy_scores,
                    cauchy_score_error_bound,
                    args.score_calibration_samples,
                    miscoverage=0.01,
                )
            )
            refined_calibrated_proxy_scores: torch.Tensor | None = None
            refined_sampled_crossfit_rmse: torch.Tensor | None = None
            if args.key_refinement_rate_budget:
                refined_allocation = allocate_bits(
                    key_distortion,
                    args.key_refinement_rate_budget,
                    key_bit_levels,
                    include_scale_metadata=True,
                )
                key_allocation_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "rate_budget": args.key_refinement_rate_budget,
                        "allocation": "-".join(
                            str(bits) for bits in refined_allocation
                        ),
                        "payload_bits_per_token": 16
                        * sum(refined_allocation),
                    }
                )
                refined_reconstructed_key = reconstruct(
                    bands, refined_allocation
                )
                refined_proxy_scores = (
                    approximate_queries.float()
                    @ refined_reconstructed_key.T
                    * scaling
                )
                refined_calibrated_proxy_scores, _, _ = (
                    affine_calibrate_scores(
                        exact_scores,
                        refined_proxy_scores,
                        args.score_calibration_samples,
                    )
                )
                refined_sampled_crossfit_rmse, _ = (
                    crossfit_affine_score_rmse(
                        exact_scores,
                        refined_proxy_scores,
                        args.score_calibration_samples,
                    )
                )
            exact_score_std = (
                exact_scores - exact_scores.mean(dim=-1, keepdim=True)
            ).square().mean(dim=-1).sqrt()
            exact_log_probabilities = torch.log_softmax(
                exact_scores.float(), dim=-1
            )
            calibrated_log_probabilities = torch.log_softmax(
                calibrated_proxy_scores.float(), dim=-1
            )
            exact_probabilities = exact_log_probabilities.exp()
            calibrated_score_delta = (
                calibrated_proxy_scores.float() - exact_scores.float()
            )
            attention_weighted_delta_mean = (
                exact_probabilities * calibrated_score_delta
            ).sum(dim=-1)
            fisher_score_distortion = (
                exact_probabilities * calibrated_score_delta.square()
            ).sum(dim=-1) - attention_weighted_delta_mean.square()
            exact_softmax_kl = (
                exact_probabilities
                * (
                    exact_log_probabilities
                    - calibrated_log_probabilities
                )
            ).sum(dim=-1)
            active_sample_count = min(
                max(2, args.score_calibration_samples), history_count
            )
            score_sample_indices = stratified_jittered_sample_indices(
                history_count,
                active_sample_count,
                exact_scores.device,
            )
            exact_score_sample = exact_scores.index_select(
                1, score_sample_indices
            ).float()
            calibrated_score_sample = calibrated_proxy_scores.index_select(
                1, score_sample_indices
            ).float()
            exact_sample_log_probabilities = torch.log_softmax(
                exact_score_sample, dim=-1
            )
            calibrated_sample_log_probabilities = torch.log_softmax(
                calibrated_score_sample, dim=-1
            )
            exact_sample_probabilities = (
                exact_sample_log_probabilities.exp()
            )
            sample_delta = calibrated_score_sample - exact_score_sample
            sample_delta_mean = (
                exact_sample_probabilities * sample_delta
            ).sum(dim=-1)
            sampled_fisher_score_distortion = (
                exact_sample_probabilities * sample_delta.square()
            ).sum(dim=-1) - sample_delta_mean.square()
            sampled_softmax_kl = (
                exact_sample_probabilities
                * (
                    exact_sample_log_probabilities
                    - calibrated_sample_log_probabilities
                )
            ).sum(dim=-1)
            for row in range(flat_queries.shape[0]):
                score_calibration_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "query_row": row,
                        "slope": float(calibration_slope[row]),
                        "intercept": float(calibration_intercept[row]),
                        "uncalibrated_rmse": float(uncalibrated_rmse[row]),
                        "calibrated_rmse": float(calibrated_rmse[row]),
                        "sampled_crossfit_rmse": float(
                            sampled_crossfit_rmse[row]
                        ),
                        "sampled_exact_score_std": float(
                            sampled_exact_score_std[row]
                        ),
                        "sampled_normalized_crossfit_rmse": float(
                            sampled_crossfit_rmse[row]
                            / sampled_exact_score_std[row].clamp_min(1.0e-12)
                        ),
                        "exact_score_std": float(exact_score_std[row]),
                        "normalized_calibrated_rmse": float(
                            calibrated_rmse[row]
                            / exact_score_std[row].clamp_min(1.0e-12)
                        ),
                        "fisher_score_distortion": float(
                            fisher_score_distortion[row].clamp_min(0.0)
                        ),
                        "exact_softmax_kl": float(
                            exact_softmax_kl[row].clamp_min(0.0)
                        ),
                        "sampled_fisher_score_distortion": float(
                            sampled_fisher_score_distortion[row].clamp_min(0.0)
                        ),
                        "sampled_softmax_kl": float(
                            sampled_softmax_kl[row].clamp_min(0.0)
                        ),
                        "sampled_crossfit_softmax_kl": float(
                            sampled_crossfit_softmax_kl[row].clamp_min(0.0)
                        ),
                        "absolute_score_error_mean": float(
                            exact_score_error_bound[row].mean()
                        ),
                        "absolute_score_error_maximum": float(
                            exact_score_error_bound[row].max()
                        ),
                        "cauchy_score_bound_mean": float(
                            cauchy_score_error_bound[row].mean()
                        ),
                        "cauchy_score_bound_maximum": float(
                            cauchy_score_error_bound[row].max()
                        ),
                        "conformal_uncertainty_scale": float(
                            conformal_uncertainty_scale[row]
                        ),
                        "conformal_uncertainty_coverage": float(
                            (
                                (
                                    exact_scores[row]
                                    - calibrated_proxy_scores[row]
                                ).abs()
                                <= calibrated_score_uncertainty[row]
                            ).float().mean()
                        ),
                    }
                )

            gram = output_group_gram(
                projection,
                kv_head,
                query_groups,
                int(value.shape[-1]),
            )
            value_mean, value_vectors, value_coordinates, _ = metric_basis(
                value,
                gram,
                args.value_rank,
                args.value_sample_stride,
                "wo_group",
            )
            quantized_coordinates = block_affine_quantize(
                value_coordinates[:, : args.value_rank],
                bits=args.value_bits,
                block_size=args.value_scale_block,
            )
            reconstructed_value = (
                value_mean
                + quantized_coordinates @ value_vectors[:, : args.value_rank].T
            )
            residual = value.float() - reconstructed_value.float()
            residual_risk = torch.sqrt(
                torch.einsum("nd,de,ne->n", residual, gram, residual)
                .clamp_min(1.0e-30)
            )
            log_risk = quantized_log_risk(
                torch.log(residual_risk), args.risk_bits
            )
            risk_priority = proxy_scores + log_risk[None, :]
            calibrated_risk_priority = (
                calibrated_proxy_scores + log_risk[None, :]
            )
            per_query_head_log_risk = []
            per_query_head_risk = []
            per_query_head_gram = []
            for group in range(query_groups):
                query_head = kv_head * query_groups + group
                start = query_head * int(value.shape[-1])
                output_block = projection[
                    :, start : start + int(value.shape[-1])
                ]
                head_gram = output_block.T @ output_block
                head_risk = torch.sqrt(
                    torch.einsum(
                        "nd,de,ne->n", residual, head_gram, residual
                    ).clamp_min(1.0e-30)
                )
                per_query_head_log_risk.append(
                    quantized_log_risk(
                        torch.log(head_risk), args.risk_bits
                    )
                )
                per_query_head_risk.append(head_risk)
                per_query_head_gram.append(head_gram)
            head_risk_values = torch.stack(per_query_head_risk, dim=0)
            head_gram_values = torch.stack(per_query_head_gram, dim=0)
            head_log_risk_values = torch.stack(
                per_query_head_log_risk, dim=0
            )
            head_log_risk = head_log_risk_values
            head_log_risk = (
                head_log_risk[None, :, :]
                .expand(len(records), -1, -1)
                .reshape(-1, history_count)
            )
            head_risk_priority = proxy_scores + head_log_risk
            head_rss_risk_priority = 2.0 * head_risk_priority
            calibrated_head_risk_priority = (
                calibrated_proxy_scores + head_log_risk
            )
            head_value_deviation = []
            head_output_scales = []
            for group in range(query_groups):
                rows = torch.arange(
                    group,
                    len(records) * query_groups,
                    query_groups,
                    device=device,
                )
                group_weights = torch.softmax(
                    proxy_scores.index_select(0, rows).float(), dim=-1
                )
                group_output = group_weights @ reconstructed_value.float()
                head_output_scales.append(
                    torch.sqrt(
                        torch.einsum(
                            "sd,de,se->s",
                            group_output,
                            head_gram_values[group],
                            group_output,
                        ).clamp_min(1.0e-30)
                    )
                )
                difference = (
                    reconstructed_value.float()[None, :, :]
                    - group_output[:, None, :]
                )
                deviation = torch.sqrt(
                    torch.einsum(
                        "snd,de,sne->sn",
                        difference,
                        head_gram_values[group],
                        difference,
                    ).clamp_min(1.0e-30)
                )
                head_value_deviation.append(deviation)
            head_value_deviation = torch.stack(
                head_value_deviation, dim=1
            ).reshape(-1, history_count)
            head_output_scale = torch.stack(
                head_output_scales, dim=1
            ).reshape(-1)
            score_output_diagnostic = sampled_score_output_error(
                exact_scores,
                calibrated_proxy_scores,
                reconstructed_value,
                head_gram_values,
                query_groups,
                sample_count=args.score_calibration_samples,
                top_count=min(128, args.score_calibration_samples // 2),
            )
            score_row_start = len(score_calibration_rows) - int(
                flat_queries.shape[0]
            )
            for row in range(flat_queries.shape[0]):
                scale = head_output_scale[row].clamp_min(1.0e-12)
                estimate = score_output_diagnostic["estimate"][row] / scale
                standard_error = (
                    score_output_diagnostic["standard_error"][row] / scale
                )
                score_calibration_rows[score_row_start + row].update(
                    {
                        "sampled_score_output_relative_estimate": float(
                            estimate
                        ),
                        "sampled_score_output_relative_standard_error": float(
                            standard_error
                        ),
                        "sampled_score_output_relative_ucb95": float(
                            estimate + 2.0 * standard_error
                        ),
                        "first_order_score_output_relative": float(
                            score_output_diagnostic["first_order"][row]
                            / scale
                        ),
                        "actual_score_output_relative": float(
                            score_output_diagnostic["actual"][row] / scale
                        ),
                    }
                )
            block_risk_priorities: dict[int, torch.Tensor] = {}
            block_output_scales: dict[int, torch.Tensor] = {}
            for block_size in (() if args.focus_global_floor_rss else (64, 256)):
                _, block_reconstructed_value = piecewise_mean_value(
                    value, block_size
                )
                block_residual = (
                    value.float() - block_reconstructed_value.float()
                )
                block_residual_risk = torch.sqrt(
                    torch.einsum(
                        "nd,de,ne->n",
                        block_residual,
                        gram,
                        block_residual,
                    ).clamp_min(1.0e-30)
                )
                block_log_risk = quantized_log_risk(
                    torch.log(block_residual_risk), args.risk_bits
                )
                block_risk_priorities[block_size] = (
                    proxy_scores + block_log_risk[None, :]
                )
                output_scales = []
                for group in range(query_groups):
                    rows = torch.arange(
                        group,
                        len(records) * query_groups,
                        query_groups,
                        device=device,
                    )
                    group_weights = torch.softmax(
                        proxy_scores.index_select(0, rows).float(), dim=-1
                    )
                    group_output = (
                        group_weights @ block_reconstructed_value.float()
                    )
                    output_scales.append(
                        torch.sqrt(
                            torch.einsum(
                                "sd,de,se->s",
                                group_output,
                                head_gram_values[group],
                                group_output,
                            ).clamp_min(1.0e-30)
                        )
                    )
                block_output_scales[block_size] = torch.stack(
                    output_scales, dim=1
                ).reshape(-1)
            expanded_head_risk = (
                head_risk_values[None, :, :]
                .expand(len(records), -1, -1)
                .reshape(-1, history_count)
            )
            joint_rmse_risk = (
                expanded_head_risk
                + uncalibrated_rmse[:, None] * head_value_deviation
            ).clamp_min(1.0e-30)
            joint_calibrated_rss_risk = joint_qk_value_rss_risk(
                expanded_head_risk,
                sampled_crossfit_rmse,
                head_value_deviation,
            )
            refined_joint_calibrated_rss_risk: torch.Tensor | None = None
            if refined_sampled_crossfit_rmse is not None:
                refined_joint_calibrated_rss_risk = (
                    joint_qk_value_rss_risk(
                        expanded_head_risk,
                        refined_sampled_crossfit_rmse,
                        head_value_deviation,
                    )
                )
            joint_oracle_risk = (
                expanded_head_risk
                + (exact_scores - proxy_scores).abs()
                * head_value_deviation
            ).clamp_min(1.0e-30)
            joint_rmse_priority = proxy_scores + torch.log(joint_rmse_risk)
            joint_calibrated_rss_priority = (
                calibrated_proxy_scores
                + torch.log(joint_calibrated_rss_risk)
            )
            joint_oracle_priority = proxy_scores + torch.log(
                joint_oracle_risk
            )
            head_states.append(
                {
                    "exact_scores": exact_scores,
                    "proxy_scores": proxy_scores,
                    "calibrated_proxy_scores": calibrated_proxy_scores,
                    "calibrated_score_uncertainty": (
                        calibrated_score_uncertainty
                    ),
                    "raw_cauchy_score_uncertainty": (
                        cauchy_score_error_bound
                    ),
                    "group_risk_priority": risk_priority,
                    "calibrated_group_risk_priority": (
                        calibrated_risk_priority
                    ),
                    "head_risk_priority": head_risk_priority,
                    "head_rss_risk_priority": head_rss_risk_priority,
                    "calibrated_head_risk_priority": (
                        calibrated_head_risk_priority
                    ),
                    "joint_rmse_priority": joint_rmse_priority,
                    "joint_calibrated_rss_priority": (
                        joint_calibrated_rss_priority
                    ),
                    "joint_oracle_priority": joint_oracle_priority,
                    "joint_rmse_risk": joint_rmse_risk,
                    "joint_calibrated_rss_risk": joint_calibrated_rss_risk,
                    "sampled_crossfit_rmse": sampled_crossfit_rmse,
                    "refined_calibrated_proxy_scores": (
                        refined_calibrated_proxy_scores
                    ),
                    "refined_sampled_crossfit_rmse": (
                        refined_sampled_crossfit_rmse
                    ),
                    "refined_joint_calibrated_rss_risk": (
                        refined_joint_calibrated_rss_risk
                    ),
                    "joint_oracle_risk": joint_oracle_risk,
                    "head_output_scale": head_output_scale,
                    "group_residual_risk": residual_risk,
                    "head_residual_risk": head_risk_values,
                    "head_gram_values": head_gram_values,
                    "value": value,
                    "reconstructed_value": reconstructed_value,
                }
            )

            masks: dict[str, torch.Tensor] = {}
            for fixed_top_k in fixed_top_ks:
                masks[f"fixed_proxy_k{fixed_top_k}"] = fixed_mask(
                    proxy_scores, fixed_top_k
                )
                masks[
                    f"fixed_risk{args.risk_bits}_k{fixed_top_k}"
                ] = fixed_mask(risk_priority, fixed_top_k)
                masks[
                    f"fixed_headrisk{args.risk_bits}_k{fixed_top_k}"
                ] = fixed_mask(head_risk_priority, fixed_top_k)
                masks[
                    f"fixed_calibrated_risk{args.risk_bits}_k{fixed_top_k}"
                ] = fixed_mask(calibrated_risk_priority, fixed_top_k)
                masks[
                    f"fixed_calibrated_headrisk{args.risk_bits}_k{fixed_top_k}"
                ] = fixed_mask(
                    calibrated_head_risk_priority, fixed_top_k
                )
            for target in targets:
                suffix = str(target).replace(".", "p")
                masks[f"attention_coverage_{suffix}"] = coverage_mask(
                    proxy_scores,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                masks[f"calibrated_attention_coverage_{suffix}"] = coverage_mask(
                    calibrated_proxy_scores,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                masks[f"exact_attention_coverage_{suffix}"] = coverage_mask(
                    exact_scores,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                for sampled_count in sampled_mass_samples:
                    for aggregation in sampled_mass_aggregations:
                        masks[
                            f"sampledmass{sampled_count}_{aggregation}_"
                            f"coverage_{suffix}"
                        ] = sampled_mass_prefix_mask(
                            proxy_scores,
                            target,
                            args.minimum_top_k,
                            args.maximum_top_k,
                            sample_count=sampled_count,
                            aggregation=aggregation,
                        )
                for sampled_count in gaussian_mass_samples:
                    masks[
                        f"gaussianmass{sampled_count}_coverage_{suffix}"
                    ] = gaussian_mass_prefix_mask(
                        proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                    )
                    masks[
                        f"calibrated_gaussianmass{sampled_count}_"
                        f"coverage_{suffix}"
                    ] = gaussian_mass_prefix_mask(
                        calibrated_proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                    )
                for sampled_count in mass_ladder_samples:
                    masks[
                        f"massladder{sampled_count}_coverage_{suffix}"
                    ] = sampled_rank_mass_ladder_mask(
                        proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                        growth=args.mass_ladder_growth,
                    )
                    masks[
                        f"calibrated_massladder{sampled_count}_"
                        f"coverage_{suffix}"
                    ] = sampled_rank_mass_ladder_mask(
                        calibrated_proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                        growth=args.mass_ladder_growth,
                    )
                    masks[
                        f"exactprefix_tailratio{sampled_count}_"
                        f"coverage_{suffix}"
                    ] = exact_prefix_tail_ratio_mass_ladder_mask(
                        exact_scores,
                        proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                        growth=args.mass_ladder_growth,
                    )
                for sampled_count in interval_mass_samples:
                    masks[
                        f"intervaloracle{sampled_count}_coverage_{suffix}"
                    ] = interval_certified_mass_ladder_mask(
                        proxy_scores,
                        exact_score_error_bound,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                        growth=args.mass_ladder_growth,
                    )
                    masks[
                        f"intervalcauchy{sampled_count}_coverage_{suffix}"
                    ] = interval_certified_mass_ladder_mask(
                        proxy_scores,
                        cauchy_score_error_bound,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=sampled_count,
                        growth=args.mass_ladder_growth,
                    )
                for bins in coverage_histogram_bins:
                    attention_histogram_mask = histogram_coverage_mask(
                        calibrated_proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        bins,
                    )
                    masks[
                        f"attention_hist{bins}_coverage_{suffix}"
                    ] = histogram_coverage_mask(
                        proxy_scores,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        bins,
                    )
                    masks[
                        f"calibrated_attention_hist{bins}_coverage_{suffix}"
                    ] = attention_histogram_mask
                    headrisk_histogram_mask = histogram_coverage_mask(
                        calibrated_head_risk_priority,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        bins,
                    )
                    masks[
                        f"calibrated_headrisk_hist{bins}_coverage_{suffix}"
                    ] = headrisk_histogram_mask
                    masks[
                        f"calibrated_union_attention_headrisk_hist{bins}_"
                        f"coverage_{suffix}"
                    ] = attention_histogram_mask | headrisk_histogram_mask
                    grouprisk_histogram_mask = histogram_coverage_mask(
                        calibrated_risk_priority,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        bins,
                    )
                    masks[
                        f"calibrated_grouprisk_hist{bins}_coverage_{suffix}"
                    ] = grouprisk_histogram_mask
                    masks[
                        f"calibrated_union_attention_grouprisk_hist{bins}_"
                        f"coverage_{suffix}"
                    ] = attention_histogram_mask | grouprisk_histogram_mask
                masks[f"risk{args.risk_bits}_coverage_{suffix}"] = coverage_mask(
                    risk_priority,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                masks[
                    f"headrisk{args.risk_bits}_coverage_{suffix}"
                ] = coverage_mask(
                    head_risk_priority,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                masks[
                    f"rssrisk{args.risk_bits}_coverage_{suffix}"
                ] = coverage_mask(
                    head_rss_risk_priority,
                    target,
                    args.minimum_top_k,
                    args.maximum_top_k,
                )
                masks[f"jointrmse{args.risk_bits}_coverage_{suffix}"] = (
                    coverage_mask(
                        joint_rmse_priority,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                )
                masks[f"jointoracle{args.risk_bits}_coverage_{suffix}"] = (
                    coverage_mask(
                        joint_oracle_priority,
                        target,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                )
            for tolerance in affine_bound_tolerances:
                suffix = str(tolerance).replace(".", "p")
                masks[f"affinebound_reltail_{suffix}"] = (
                    affine_residual_bound_ladder_mask(
                        proxy_scores,
                        expanded_head_risk,
                        head_output_scale,
                        tolerance,
                        args.minimum_top_k,
                        args.maximum_top_k,
                        sample_count=(
                            mass_ladder_samples[0]
                            if mass_ladder_samples
                            else 1024
                        ),
                        growth=args.mass_ladder_growth,
                    )
                )
            for tolerance in relative_risk_thresholds:
                suffix = str(tolerance).replace(".", "p")
                masks[f"jointrmse{args.risk_bits}_reltail_{suffix}"] = (
                    relative_tail_risk_mask(
                        joint_rmse_priority,
                        proxy_scores,
                        head_output_scale,
                        tolerance,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                )
                masks[f"jointoracle{args.risk_bits}_reltail_{suffix}"] = (
                    relative_tail_risk_mask(
                        joint_oracle_priority,
                        proxy_scores,
                        head_output_scale,
                        tolerance,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                )
            for tolerance in rss_relative_tolerances:
                tolerance_suffix = str(tolerance).replace(".", "p")
                for safety_factor in rss_safety_factors:
                    safety_suffix = str(safety_factor).replace(".", "p")
                    masks[
                        f"rssrisk{args.risk_bits}_reltail_"
                        f"{tolerance_suffix}_s{safety_suffix}"
                    ] = relative_tail_rss_mask(
                        head_risk_priority,
                        proxy_scores,
                        head_output_scale,
                        tolerance,
                        safety_factor,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                    masks[
                        f"rssgrouprisk{args.risk_bits}_reltail_"
                        f"{tolerance_suffix}_s{safety_suffix}"
                    ] = relative_tail_rss_mask(
                        risk_priority,
                        proxy_scores,
                        head_output_scale,
                        tolerance,
                        safety_factor,
                        args.minimum_top_k,
                        args.maximum_top_k,
                    )
                    for block_size in (64, 256):
                        masks[
                            f"rssblock{block_size}risk{args.risk_bits}_"
                            f"reltail_{tolerance_suffix}_s{safety_suffix}"
                        ] = relative_tail_rss_mask(
                            block_risk_priorities[block_size],
                            proxy_scores,
                            block_output_scales[block_size],
                            tolerance,
                            safety_factor,
                            args.minimum_top_k,
                            args.maximum_top_k,
                        )

            for tolerance in scalar_rss_tolerances:
                tolerance_suffix = str(tolerance).replace(".", "p")
                for safety_factor in rss_safety_factors:
                    safety_suffix = str(safety_factor).replace(".", "p")
                    for statistic in scalar_rss_statistics:
                        condition = (
                            f"scalarrss_{statistic}_reltail_"
                            f"{tolerance_suffix}_s{safety_suffix}"
                        )
                        masks[condition] = scalar_residual_rss_mask(
                            proxy_scores,
                            expanded_head_risk,
                            head_output_scale,
                            tolerance,
                            safety_factor,
                            args.minimum_top_k,
                            args.maximum_top_k,
                            statistic,
                        )
                        masks[f"calibrated_{condition}"] = (
                            scalar_residual_rss_mask(
                                calibrated_proxy_scores,
                                expanded_head_risk,
                                head_output_scale,
                                tolerance,
                                safety_factor,
                                args.minimum_top_k,
                                args.maximum_top_k,
                                statistic,
                            )
                        )

            if args.focus_block_risk:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith("rssblock")
                }
            elif args.focus_global_floor_rss:
                masks = {}
            elif args.focus_sampled_mass:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(("attention_coverage_", "sampledmass"))
                }
            elif args.focus_scalar_rss:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(("scalarrss_", "calibrated_scalarrss_"))
                }
            elif args.focus_gaussian_mass:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(
                        (
                            "attention_coverage_",
                            "gaussianmass",
                            "calibrated_gaussianmass",
                        )
                    )
                }
            elif args.focus_mass_ladder:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(
                        (
                            "attention_coverage_",
                            "calibrated_attention_coverage_",
                            "exact_attention_coverage_",
                            "massladder",
                            "calibrated_massladder",
                            "exactprefix_tailratio",
                            "attention_hist",
                            "calibrated_attention_hist",
                            "calibrated_headrisk_hist",
                            "calibrated_union_attention_headrisk_hist",
                            "calibrated_grouprisk_hist",
                            "calibrated_union_attention_grouprisk_hist",
                            "affinebound",
                        )
                    )
                }
            elif args.focus_interval_mass:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(
                        (
                            "attention_coverage_",
                            "massladder",
                            "intervaloracle",
                            "intervalcauchy",
                        )
                    )
                }
            elif (
                args.focus_balanced_rss
                or args.focus_progressive_balanced_rss
            ):
                masks = {}
            elif args.focus_rss_calibration:
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name.startswith(
                        (
                            "fixed_proxy_",
                            "exact_attention_coverage_",
                            "calibrated_attention_coverage_",
                        )
                    )
                }

            full_for_head: torch.Tensor | None = None
            exact_risk_mass = torch.exp(
                exact_scores - exact_scores.amax(dim=-1, keepdim=True)
            ) * residual_risk[None, :]
            for condition, mask in masks.items():
                outputs, full_output, attention_mass, _ = approximate_output(
                    exact_scores,
                    proxy_scores,
                    value,
                    reconstructed_value,
                    mask,
                    residual_sample_counts,
                )
                if args.focus_block_risk:
                    outputs = {
                        path: output
                        for path, output in outputs.items()
                        if path
                        in {
                            "selected_exact_only",
                            "hybrid_blockmean64",
                            "hybrid_blockmean256",
                            "hybrid_sketch",
                        }
                    }
                else:
                    calibrated_outputs, _, _, _ = approximate_output(
                        exact_scores,
                        calibrated_proxy_scores,
                        value,
                        reconstructed_value,
                        mask,
                        residual_sample_counts,
                    )
                    outputs.update(
                        {
                            f"calibrated_{path}": output
                            for path, output in calibrated_outputs.items()
                        }
                    )
                    if (
                        args.focus_scalar_rss
                        or args.focus_gaussian_mass
                        or args.focus_mass_ladder
                        or args.focus_interval_mass
                        or args.focus_rss_calibration
                    ):
                        outputs = {
                            path: output
                            for path, output in outputs.items()
                            if path
                            in {
                                "selected_exact_only",
                                "hybrid_sketch",
                                "hybrid_sketch_partition_oracle",
                                "hybrid_sketch_partition_sample256",
                                "hybrid_full_value",
                                "hybrid_sketch_centered_residual",
                                "hybrid_sketch_affine_residual",
                                "calibrated_selected_exact_only",
                                "calibrated_hybrid_sketch",
                                "calibrated_hybrid_full_value",
                                "calibrated_hybrid_sketch_centered_residual",
                                "calibrated_hybrid_sketch_affine_residual",
                                "exact_weight_sketch",
                                "hybrid_full_value",
                                "calibrated_hybrid_full_value",
                                *(
                                    f"hybrid_sketch_proxyresidualsample{count}"
                                    for count in residual_sample_counts
                                ),
                                *(
                                    f"hybrid_sketch_exactresidualsample{count}"
                                    for count in residual_sample_counts
                                ),
                                *(
                                    "calibrated_hybrid_sketch_"
                                    f"proxyresidualsample{count}"
                                    for count in residual_sample_counts
                                ),
                                *(
                                    "calibrated_hybrid_sketch_"
                                    f"exactresidualsample{count}"
                                    for count in residual_sample_counts
                                ),
                            }
                        }
                if full_for_head is None:
                    full_for_head = full_output
                risk_coverage = (
                    (exact_risk_mass * mask).sum(dim=-1)
                    / exact_risk_mass.sum(dim=-1).clamp_min(1.0e-20)
                )
                proxy_probability = torch.softmax(
                    proxy_scores.float(), dim=-1
                )
                tail_residual_rss = torch.sqrt(
                    (
                        proxy_probability
                        * expanded_head_risk
                        * (~mask).float()
                    )
                    .square()
                    .sum(dim=-1)
                )
                exact_probability = torch.softmax(
                    exact_scores.float(), dim=-1
                )
                exact_tail_residual_rss = torch.sqrt(
                    (
                        exact_probability
                        * expanded_head_risk
                        * (~mask).float()
                    )
                    .square()
                    .sum(dim=-1)
                )
                relative_tail_residual_rss = (
                    tail_residual_rss
                    / head_output_scale.clamp_min(1.0e-12)
                )
                for approximation_path, output in outputs.items():
                    key = (condition, approximation_path)
                    condition_outputs[key].append(
                        output.reshape(len(records), query_groups, -1)
                    )
                    condition_counts[key].append(
                        mask.sum(dim=-1).reshape(len(records), query_groups)
                    )
                    condition_attention_mass[key].append(
                        attention_mass.reshape(len(records), query_groups)
                    )
                    condition_risk_mass[key].append(
                        risk_coverage.reshape(len(records), query_groups)
                    )
                    condition_residual_rss[key].append(
                        relative_tail_residual_rss.reshape(
                            len(records), query_groups
                        )
                    )
                    condition_proxy_residual_rss_absolute[key].append(
                        tail_residual_rss.reshape(
                            len(records), query_groups
                        )
                    )
                    condition_exact_residual_rss_absolute[key].append(
                        exact_tail_residual_rss.reshape(
                            len(records), query_groups
                        )
                    )
            if full_for_head is None:
                _, full_for_head, _, _ = approximate_output(
                    exact_scores,
                    proxy_scores,
                    value,
                    reconstructed_value,
                    fixed_mask(proxy_scores, min(1, history_count)),
                )
            condition_full.append(
                full_for_head.reshape(len(records), query_groups, -1)
            )

        if global_top_ks or global_rss_tolerances or balanced_rss_tolerances:
            proxy_cube = torch.cat(
                [
                    state["proxy_scores"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            calibrated_proxy_cube = torch.cat(
                [
                    state["calibrated_proxy_scores"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            group_risk_cube = torch.cat(
                [
                    state["group_risk_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            head_risk_cube = torch.cat(
                [
                    state["head_risk_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            head_output_scale_cube = torch.cat(
                [
                    state["head_output_scale"].reshape(
                        len(records), query_groups
                    )
                    for state in head_states
                ],
                dim=1,
            )
            calibrated_group_risk_cube = torch.cat(
                [
                    state["calibrated_group_risk_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            calibrated_head_risk_cube = torch.cat(
                [
                    state["calibrated_head_risk_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            joint_rmse_cube = torch.cat(
                [
                    state["joint_rmse_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            joint_oracle_cube = torch.cat(
                [
                    state["joint_oracle_priority"].reshape(
                        len(records), query_groups, history_count
                    )
                    for state in head_states
                ],
                dim=1,
            )
            proxy_log_partition = torch.logsumexp(
                proxy_cube, dim=-1, keepdim=True
            )
            calibrated_proxy_log_partition = torch.logsumexp(
                calibrated_proxy_cube, dim=-1, keepdim=True
            )
            global_priorities = {
                "attention": proxy_cube - proxy_log_partition,
                f"grouprisk{args.risk_bits}": (
                    group_risk_cube - proxy_log_partition
                ),
                f"headrisk{args.risk_bits}": (
                    head_risk_cube - proxy_log_partition
                ),
                f"jointrmse{args.risk_bits}": (
                    joint_rmse_cube - proxy_log_partition
                ),
                f"jointoracle{args.risk_bits}": (
                    joint_oracle_cube - proxy_log_partition
                ),
                "calibrated_attention": (
                    calibrated_proxy_cube
                    - calibrated_proxy_log_partition
                ),
                f"calibrated_grouprisk{args.risk_bits}": (
                    calibrated_group_risk_cube
                    - calibrated_proxy_log_partition
                ),
                f"calibrated_headrisk{args.risk_bits}": (
                    calibrated_head_risk_cube
                    - calibrated_proxy_log_partition
                ),
            }
            if requested_global_priorities:
                missing = requested_global_priorities - set(global_priorities)
                if missing:
                    raise ValueError(
                        "Unknown global priorities: " + ", ".join(sorted(missing))
                    )
                global_priorities = {
                    name: priority
                    for name, priority in global_priorities.items()
                    if name in requested_global_priorities
                }
            total_slots = query_head_count * history_count
            for global_top_k in global_top_ks:
                active_slots = min(
                    total_slots, query_head_count * global_top_k
                )
                for allocation_name, priority_cube in global_priorities.items():
                    floor_options: tuple[float | None, ...] = (
                        None,
                        *global_floor_fractions,
                    )
                    for floor_fraction in floor_options:
                        global_indices: torch.Tensor | None = None
                        flat_priority = priority_cube.reshape(
                            len(records), total_slots
                        )
                        if floor_fraction is None:
                            global_indices = torch.topk(
                                flat_priority,
                                k=active_slots,
                                dim=-1,
                                sorted=False,
                            ).indices
                            flat_global_mask = torch.zeros_like(
                                flat_priority, dtype=torch.bool
                            )
                            flat_global_mask.scatter_(1, global_indices, True)
                            condition = (
                                f"global_{allocation_name}_equivk{global_top_k}"
                            )
                        else:
                            floor_k = min(
                                history_count,
                                max(1, round(global_top_k * floor_fraction)),
                            )
                            floor_indices = torch.topk(
                                proxy_cube,
                                k=floor_k,
                                dim=-1,
                                sorted=False,
                            ).indices
                            base_mask = torch.zeros_like(
                                proxy_cube, dtype=torch.bool
                            )
                            base_mask.scatter_(2, floor_indices, True)
                            flat_global_mask = base_mask.reshape(
                                len(records), total_slots
                            )
                            remaining_slots = active_slots - (
                                query_head_count * floor_k
                            )
                            if remaining_slots > 0:
                                eligible_priority = flat_priority.masked_fill(
                                    flat_global_mask, -torch.inf
                                )
                                global_indices = torch.topk(
                                    eligible_priority,
                                    k=remaining_slots,
                                    dim=-1,
                                    sorted=False,
                                ).indices
                                flat_global_mask.scatter_(
                                    1, global_indices, True
                                )
                            floor_suffix = str(floor_fraction).replace(".", "p")
                            condition = (
                                f"global_{allocation_name}_proxyfloor"
                                f"{floor_suffix}_equivk{global_top_k}"
                            )
                        global_mask = flat_global_mask.reshape(
                            len(records),
                            query_head_count,
                            history_count,
                        )
                        for kv_head, state in enumerate(head_states):
                            head_start = kv_head * query_groups
                            head_stop = head_start + query_groups
                            mask = global_mask[
                                :, head_start:head_stop, :
                            ].reshape(-1, history_count)
                            outputs, _, attention_mass, exact_weights = (
                                approximate_output(
                                    state["exact_scores"],
                                    state["proxy_scores"],
                                    state["value"],
                                    state["reconstructed_value"],
                                    mask,
                                )
                            )
                            calibrated_outputs, _, _, _ = approximate_output(
                                state["exact_scores"],
                                state["calibrated_proxy_scores"],
                                state["value"],
                                state["reconstructed_value"],
                                mask,
                            )
                            outputs.update(
                                {
                                    f"calibrated_{path}": output
                                    for path, output in calibrated_outputs.items()
                                }
                            )
                            if allocation_name.startswith(
                                ("headrisk", "jointrmse", "jointoracle")
                            ):
                                active_residual_risk = state[
                                    "head_residual_risk"
                                ][None, :, :].expand(len(records), -1, -1)
                            else:
                                active_residual_risk = state[
                                    "group_residual_risk"
                                ][None, None, :].expand(
                                    len(records), query_groups, -1
                                )
                            active_residual_risk = active_residual_risk.reshape(
                                -1, history_count
                            )
                            exact_risk_mass = exact_weights * active_residual_risk
                            risk_coverage = (
                                (exact_risk_mass * mask).sum(dim=-1)
                                / exact_risk_mass.sum(dim=-1).clamp_min(1.0e-20)
                            )
                            head_residual_risk = state[
                                "head_residual_risk"
                            ][None, :, :].expand(
                                len(records), -1, -1
                            ).reshape(-1, history_count)
                            tail_residual_rss = torch.sqrt(
                                (
                                    torch.softmax(
                                        state["proxy_scores"].float(), dim=-1
                                    )
                                    * head_residual_risk
                                    * (~mask).float()
                                ).square().sum(dim=-1)
                            ) / state["head_output_scale"].clamp_min(1.0e-12)
                            for approximation_path, output in outputs.items():
                                key = (condition, approximation_path)
                                condition_outputs[key].append(
                                    output.reshape(
                                        len(records), query_groups, -1
                                    )
                                )
                                condition_counts[key].append(
                                    mask.sum(dim=-1).reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_attention_mass[key].append(
                                    attention_mass.reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_risk_mass[key].append(
                                    risk_coverage.reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_residual_rss[key].append(
                                    tail_residual_rss.reshape(
                                        len(records), query_groups
                                    )
                                )
                        del global_mask, flat_global_mask
                        if global_indices is not None:
                            del global_indices

            layer_output_scale = torch.sqrt(
                head_output_scale_cube.float().square().sum(dim=1)
            )
            balanced_output_scale_cube = balanced_head_output_scales(
                head_output_scale_cube
            )
            for tolerance in (
                ()
                if args.focus_progressive_balanced_rss
                else balanced_rss_tolerances
            ):
                tolerance_suffix = str(tolerance).replace(".", "p")
                for safety_factor in rss_safety_factors:
                    safety_suffix = str(safety_factor).replace(".", "p")
                    condition = (
                        f"balancedjointrss{args.risk_bits}_"
                        f"rel{tolerance_suffix}_s{safety_suffix}_"
                        f"floor{args.minimum_top_k}"
                    )
                    for kv_head, state in enumerate(head_states):
                        head_start = kv_head * query_groups
                        head_stop = head_start + query_groups
                        head_scale = balanced_output_scale_cube[
                            :, head_start:head_stop
                        ].reshape(-1)
                        mask = relative_tail_rss_mask(
                            state["joint_calibrated_rss_priority"],
                            state["calibrated_proxy_scores"],
                            head_scale,
                            tolerance,
                            safety_factor,
                            args.minimum_top_k,
                            args.maximum_top_k,
                        )
                        tail_score_diagnostic = (
                            sampled_tail_score_output_error(
                                state["exact_scores"],
                                state["calibrated_proxy_scores"],
                                state["value"],
                                state["reconstructed_value"],
                                mask,
                                state["head_gram_values"],
                                query_groups,
                                sample_count=args.score_calibration_samples,
                                top_count=min(
                                    128,
                                    args.score_calibration_samples // 2,
                                ),
                                score_uncertainty=state[
                                    "raw_cauchy_score_uncertainty"
                                ],
                            )
                        )
                        outputs, _, attention_mass, exact_weights = (
                            approximate_output(
                                state["exact_scores"],
                                state["calibrated_proxy_scores"],
                                state["value"],
                                state["reconstructed_value"],
                                mask,
                            )
                        )
                        outputs = {
                            path: output
                            for path, output in outputs.items()
                            if path
                            in {
                                "selected_exact_only",
                                "hybrid_sketch",
                                "hybrid_sketch_partition_oracle",
                                "hybrid_sketch_partition_sample256",
                                "hybrid_sketch_centered_residual",
                                "hybrid_sketch_affine_residual",
                                "hybrid_sketch_blockresidual64",
                                "hybrid_sketch_blockresidual256",
                                "hybrid_full_value",
                                "exact_weight_sketch",
                            }
                        }
                        head_residual_risk = state[
                            "head_residual_risk"
                        ][None, :, :].expand(
                            len(records), -1, -1
                        ).reshape(-1, history_count)
                        exact_risk_mass = exact_weights * head_residual_risk
                        risk_coverage = (
                            (exact_risk_mass * mask).sum(dim=-1)
                            / exact_risk_mass.sum(dim=-1).clamp_min(1.0e-20)
                        )
                        proxy_probability = torch.softmax(
                            state["calibrated_proxy_scores"].float(), dim=-1
                        )
                        exact_probability = torch.softmax(
                            state["exact_scores"].float(), dim=-1
                        )
                        tail = (~mask).float()
                        proxy_tail_residual_rss = torch.sqrt(
                            (
                                proxy_probability
                                * head_residual_risk
                                * tail
                            ).square().sum(dim=-1)
                        )
                        exact_tail_residual_rss = torch.sqrt(
                            (
                                exact_probability
                                * head_residual_risk
                                * tail
                            ).square().sum(dim=-1)
                        )
                        for approximation_path, output in outputs.items():
                            key = (condition, approximation_path)
                            condition_outputs[key].append(
                                output.reshape(
                                    len(records), query_groups, -1
                                )
                            )
                            condition_counts[key].append(
                                mask.sum(dim=-1).reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_tail_score_estimate[key].append(
                                tail_score_diagnostic["estimate"].reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_tail_score_standard_error[key].append(
                                tail_score_diagnostic[
                                    "standard_error"
                                ].reshape(len(records), query_groups)
                            )
                            condition_tail_score_actual[key].append(
                                tail_score_diagnostic["actual"].reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_tail_score_first_order[key].append(
                                tail_score_diagnostic[
                                    "first_order"
                                ].reshape(len(records), query_groups)
                            )
                            condition_attention_mass[key].append(
                                attention_mass.reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_risk_mass[key].append(
                                risk_coverage.reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_residual_rss[key].append(
                                (
                                    proxy_tail_residual_rss
                                    / head_scale.clamp_min(1.0e-12)
                                ).reshape(len(records), query_groups)
                            )
                            condition_proxy_residual_rss_absolute[key].append(
                                proxy_tail_residual_rss.reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_exact_residual_rss_absolute[key].append(
                                exact_tail_residual_rss.reshape(
                                    len(records), query_groups
                                )
                            )
            if args.key_refinement_rate_budget:
                for tolerance in balanced_rss_tolerances:
                    tolerance_suffix = str(tolerance).replace(".", "p")
                    for safety_factor in rss_safety_factors:
                        safety_suffix = str(safety_factor).replace(".", "p")
                        condition = (
                            f"progressivebalancedjointrss{args.risk_bits}_"
                            f"base{args.key_rate_budget}_"
                            f"refine{args.key_refinement_rate_budget}_"
                            f"rel{tolerance_suffix}_s{safety_suffix}_"
                            f"floor{args.minimum_top_k}"
                        )
                        for kv_head, state in enumerate(head_states):
                            refined_scores = state[
                                "refined_calibrated_proxy_scores"
                            ]
                            refined_score_rmse = state[
                                "refined_sampled_crossfit_rmse"
                            ]
                            refined_joint_risk = state[
                                "refined_joint_calibrated_rss_risk"
                            ]
                            if (
                                refined_scores is None
                                or refined_score_rmse is None
                                or refined_joint_risk is None
                            ):
                                raise RuntimeError(
                                    "progressive state is incomplete"
                                )
                            head_start = kv_head * query_groups
                            head_stop = head_start + query_groups
                            head_scale = balanced_output_scale_cube[
                                :, head_start:head_stop
                            ].reshape(-1)
                            mask, refinement_mask, crossing_rss = (
                                progressive_error_balanced_masks(
                                    state["calibrated_proxy_scores"],
                                    refined_scores,
                                    state["joint_calibrated_rss_risk"],
                                    refined_joint_risk,
                                    state["sampled_crossfit_rmse"],
                                    refined_score_rmse,
                                    head_scale,
                                    tolerance,
                                    safety_factor,
                                    args.minimum_top_k,
                                    args.maximum_top_k,
                                    rounds=args.progressive_refinement_rounds,
                                )
                            )
                            mixed_scores = torch.where(
                                refinement_mask,
                                refined_scores,
                                state["calibrated_proxy_scores"],
                            )
                            outputs, _, attention_mass, exact_weights = (
                                approximate_output(
                                    state["exact_scores"],
                                    mixed_scores,
                                    state["value"],
                                    state["reconstructed_value"],
                                    mask,
                                )
                            )
                            outputs = {
                                path: output
                                for path, output in outputs.items()
                                if path
                                in {
                                    "selected_exact_only",
                                    "hybrid_sketch",
                                    "hybrid_sketch_partition_oracle",
                                    "hybrid_sketch_partition_sample256",
                                    "hybrid_sketch_centered_residual",
                                    "hybrid_sketch_affine_residual",
                                    "hybrid_sketch_blockresidual64",
                                    "hybrid_sketch_blockresidual256",
                                    "hybrid_full_value",
                                    "exact_weight_sketch",
                                }
                            }
                            head_residual_risk = state[
                                "head_residual_risk"
                            ][None, :, :].expand(
                                len(records), -1, -1
                            ).reshape(-1, history_count)
                            exact_risk_mass = (
                                exact_weights * head_residual_risk
                            )
                            risk_coverage = (
                                (exact_risk_mass * mask).sum(dim=-1)
                                / exact_risk_mass.sum(dim=-1).clamp_min(
                                    1.0e-20
                                )
                            )
                            proxy_probability = torch.softmax(
                                mixed_scores.float(), dim=-1
                            )
                            exact_probability = torch.softmax(
                                state["exact_scores"].float(), dim=-1
                            )
                            tail = (~mask).float()
                            proxy_tail_residual_rss = torch.sqrt(
                                (
                                    proxy_probability
                                    * head_residual_risk
                                    * tail
                                ).square().sum(dim=-1)
                            )
                            exact_tail_residual_rss = torch.sqrt(
                                (
                                    exact_probability
                                    * head_residual_risk
                                    * tail
                                ).square().sum(dim=-1)
                            )
                            for approximation_path, output in outputs.items():
                                key = (condition, approximation_path)
                                condition_outputs[key].append(
                                    output.reshape(
                                        len(records), query_groups, -1
                                    )
                                )
                                condition_counts[key].append(
                                    mask.sum(dim=-1).reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_refinement_counts[key].append(
                                    refinement_mask.sum(dim=-1).reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_unresolved_crossing_rss[key].append(
                                    (
                                        crossing_rss
                                        / head_scale.clamp_min(1.0e-12)
                                    ).reshape(len(records), query_groups)
                                )
                                condition_attention_mass[key].append(
                                    attention_mass.reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_risk_mass[key].append(
                                    risk_coverage.reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_residual_rss[key].append(
                                    (
                                        proxy_tail_residual_rss
                                        / head_scale.clamp_min(1.0e-12)
                                    ).reshape(len(records), query_groups)
                                )
                                condition_proxy_residual_rss_absolute[
                                    key
                                ].append(
                                    proxy_tail_residual_rss.reshape(
                                        len(records), query_groups
                                    )
                                )
                                condition_exact_residual_rss_absolute[
                                    key
                                ].append(
                                    exact_tail_residual_rss.reshape(
                                        len(records), query_groups
                                    )
                                )
            for tolerance in global_rss_tolerances:
                tolerance_suffix = str(tolerance).replace(".", "p")
                for safety_factor in rss_safety_factors:
                    safety_suffix = str(safety_factor).replace(".", "p")
                    global_mask = global_floor_rss_mask(
                        group_risk_cube,
                        proxy_cube,
                        layer_output_scale,
                        floor_k=args.global_rss_floor_k,
                        tolerance=tolerance,
                        safety_factor=safety_factor,
                    )
                    condition = (
                        f"globalfloorrss_grouprisk{args.risk_bits}_"
                        f"floor{args.global_rss_floor_k}_"
                        f"rel{tolerance_suffix}_s{safety_suffix}"
                    )
                    for kv_head, state in enumerate(head_states):
                        head_start = kv_head * query_groups
                        head_stop = head_start + query_groups
                        mask = global_mask[
                            :, head_start:head_stop, :
                        ].reshape(-1, history_count)
                        outputs, _, attention_mass, exact_weights = (
                            approximate_output(
                                state["exact_scores"],
                                state["proxy_scores"],
                                state["value"],
                                state["reconstructed_value"],
                                mask,
                            )
                        )
                        outputs = {
                            path: output
                            for path, output in outputs.items()
                            if path in {"hybrid_sketch", "selected_exact_only"}
                        }
                        active_residual_risk = state[
                            "group_residual_risk"
                        ][None, None, :].expand(
                            len(records), query_groups, -1
                        ).reshape(-1, history_count)
                        exact_risk_mass = (
                            exact_weights * active_residual_risk
                        )
                        risk_coverage = (
                            (exact_risk_mass * mask).sum(dim=-1)
                            / exact_risk_mass.sum(dim=-1).clamp_min(1.0e-20)
                        )
                        head_residual_risk = state[
                            "head_residual_risk"
                        ][None, :, :].expand(
                            len(records), -1, -1
                        ).reshape(-1, history_count)
                        tail_residual_rss = torch.sqrt(
                            (
                                torch.softmax(
                                    state["proxy_scores"].float(), dim=-1
                                )
                                * head_residual_risk
                                * (~mask).float()
                            ).square().sum(dim=-1)
                        ) / state["head_output_scale"].clamp_min(1.0e-12)
                        for approximation_path, output in outputs.items():
                            key = (condition, approximation_path)
                            condition_outputs[key].append(
                                output.reshape(
                                    len(records), query_groups, -1
                                )
                            )
                            condition_counts[key].append(
                                mask.sum(dim=-1).reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_attention_mass[key].append(
                                attention_mass.reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_risk_mass[key].append(
                                risk_coverage.reshape(
                                    len(records), query_groups
                                )
                            )
                            condition_residual_rss[key].append(
                                tail_residual_rss.reshape(
                                    len(records), query_groups
                                )
                            )
                    del global_mask

        full_heads = torch.cat(condition_full, dim=1)
        full_projected = full_heads.reshape(len(records), -1) @ projection.T
        denominator = torch.linalg.vector_norm(
            full_projected, dim=-1
        ).clamp_min(1.0e-12)
        for key in sorted(condition_outputs):
            condition, approximation_path = key
            approximate_heads = torch.cat(condition_outputs[key], dim=1)
            approximate_projected = (
                approximate_heads.reshape(len(records), -1) @ projection.T
            )
            relative_l2 = (
                torch.linalg.vector_norm(
                    approximate_projected - full_projected, dim=-1
                )
                / denominator
            )
            counts = torch.cat(condition_counts[key], dim=1)
            has_progressive_diagnostic = key in condition_refinement_counts
            if has_progressive_diagnostic:
                refinement_counts = torch.cat(
                    condition_refinement_counts[key], dim=1
                )
                unresolved_crossing_rss = torch.cat(
                    condition_unresolved_crossing_rss[key], dim=1
                )
            else:
                refinement_counts = torch.zeros_like(counts)
                unresolved_crossing_rss = torch.zeros_like(
                    counts, dtype=torch.float32
                )
            has_tail_score_diagnostic = key in condition_tail_score_estimate
            if has_tail_score_diagnostic:
                tail_score_estimate = torch.cat(
                    condition_tail_score_estimate[key], dim=1
                )
                tail_score_standard_error = torch.cat(
                    condition_tail_score_standard_error[key], dim=1
                )
                tail_score_actual = torch.cat(
                    condition_tail_score_actual[key], dim=1
                )
                tail_score_first_order = torch.cat(
                    condition_tail_score_first_order[key], dim=1
                )
                layer_tail_score_estimate = torch.sqrt(
                    tail_score_estimate.float().square().sum(dim=1)
                ) / denominator
                layer_tail_score_standard_error = torch.sqrt(
                    tail_score_standard_error.float().square().sum(dim=1)
                ) / denominator
                layer_tail_score_actual = torch.sqrt(
                    tail_score_actual.float().square().sum(dim=1)
                ) / denominator
                layer_tail_score_first_order = torch.sqrt(
                    tail_score_first_order.float().square().sum(dim=1)
                ) / denominator
            else:
                layer_tail_score_estimate = torch.zeros_like(relative_l2)
                layer_tail_score_standard_error = torch.zeros_like(
                    relative_l2
                )
                layer_tail_score_actual = torch.zeros_like(relative_l2)
                layer_tail_score_first_order = torch.zeros_like(relative_l2)
            attention_mass = torch.cat(
                condition_attention_mass[key], dim=1
            )
            risk_mass = torch.cat(condition_risk_mass[key], dim=1)
            residual_rss = torch.cat(
                condition_residual_rss[key], dim=1
            )
            has_value_rss_diagnostic = (
                key in condition_proxy_residual_rss_absolute
            )
            if has_value_rss_diagnostic:
                proxy_absolute_rss = torch.cat(
                    condition_proxy_residual_rss_absolute[key], dim=1
                )
                exact_absolute_rss = torch.cat(
                    condition_exact_residual_rss_absolute[key], dim=1
                )
                predicted_layer_proxy_rss = torch.sqrt(
                    proxy_absolute_rss.float().square().sum(dim=1)
                ) / denominator
                predicted_layer_exact_rss = torch.sqrt(
                    exact_absolute_rss.float().square().sum(dim=1)
                ) / denominator
            else:
                predicted_layer_proxy_rss = torch.zeros_like(relative_l2)
                predicted_layer_exact_rss = torch.zeros_like(relative_l2)
            for step in range(len(records)):
                step_counts = counts[step].float()
                step_refinement_counts = refinement_counts[step].float()
                detail_rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "step": step,
                        "condition": condition,
                        "approximation_path": approximation_path,
                        "relative_l2": float(relative_l2[step]),
                        "selected_tokens_mean": float(step_counts.mean()),
                        "selected_tokens_min": int(step_counts.min()),
                        "selected_tokens_p10": float(
                            torch.quantile(step_counts, 0.10)
                        ),
                        "selected_tokens_p50": float(
                            torch.quantile(step_counts, 0.50)
                        ),
                        "selected_tokens_p90": float(
                            torch.quantile(step_counts, 0.90)
                        ),
                        "selected_tokens_max": int(step_counts.max()),
                        "selected_ratio_mean": float(
                            step_counts.mean() / history_count
                        ),
                        "refined_tokens_mean": float(
                            step_refinement_counts.mean()
                        ),
                        "refined_ratio_mean": float(
                            step_refinement_counts.mean() / history_count
                        ),
                        "unresolved_crossing_rss_mean": float(
                            unresolved_crossing_rss[step].mean()
                        ),
                        "unresolved_crossing_rss_p90": float(
                            torch.quantile(
                                unresolved_crossing_rss[step].float(), 0.90
                            )
                        ),
                        "progressive_diagnostic_available": int(
                            has_progressive_diagnostic
                        ),
                        "sampled_tail_score_output_relative_estimate": float(
                            layer_tail_score_estimate[step]
                        ),
                        "sampled_tail_score_output_relative_standard_error": float(
                            layer_tail_score_standard_error[step]
                        ),
                        "sampled_tail_score_output_relative_ucb95": float(
                            layer_tail_score_estimate[step]
                            + 2.0 * layer_tail_score_standard_error[step]
                        ),
                        "actual_tail_score_output_relative": float(
                            layer_tail_score_actual[step]
                        ),
                        "first_order_tail_score_output_relative": float(
                            layer_tail_score_first_order[step]
                        ),
                        "tail_score_diagnostic_available": int(
                            has_tail_score_diagnostic
                        ),
                        "exact_attention_mass_mean": float(
                            attention_mass[step].mean()
                        ),
                        "exact_residual_risk_mass_mean": float(
                            risk_mass[step].mean()
                        ),
                        "predicted_tail_residual_rss_mean": float(
                            residual_rss[step].mean()
                        ),
                        "predicted_tail_residual_rss_p90": float(
                            torch.quantile(residual_rss[step], 0.90)
                        ),
                        "predicted_tail_residual_rss_max": float(
                            residual_rss[step].max()
                        ),
                        "predicted_layer_proxy_value_rss": float(
                            predicted_layer_proxy_rss[step]
                        ),
                        "predicted_layer_exact_value_rss": float(
                            predicted_layer_exact_rss[step]
                        ),
                        "exact_value_rss_calibration_ratio": float(
                            relative_l2[step]
                            / predicted_layer_exact_rss[step].clamp_min(
                                1.0e-12
                            )
                        ),
                        "value_rss_diagnostic_available": int(
                            has_value_rss_diagnostic
                        ),
                    }
                )
        torch.cuda.empty_cache()
        print(json.dumps({"topic": topic, "layer": layer}), flush=True)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[
            (str(row["condition"]), str(row["approximation_path"]))
        ].append(row)
    summary_rows = []
    for (condition, approximation_path), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "condition": condition,
                "approximation_path": approximation_path,
                "cases": len(rows),
                **{
                    f"relative_l2_{name}": value
                    for name, value in quantiles(
                        [float(row["relative_l2"]) for row in rows]
                    ).items()
                },
                **{
                    f"selected_tokens_{name}": value
                    for name, value in quantiles(
                        [float(row["selected_tokens_mean"]) for row in rows]
                    ).items()
                },
                "selected_ratio_mean": sum(
                    float(row["selected_ratio_mean"]) for row in rows
                )
                / len(rows),
                "refined_tokens_mean": sum(
                    float(row["refined_tokens_mean"]) for row in rows
                )
                / len(rows),
                "refined_ratio_mean": sum(
                    float(row["refined_ratio_mean"]) for row in rows
                )
                / len(rows),
                **{
                    f"unresolved_crossing_rss_{name}": value
                    for name, value in quantiles(
                        [
                            float(row["unresolved_crossing_rss_mean"])
                            for row in rows
                        ]
                    ).items()
                },
                "progressive_diagnostic_available": bool(
                    rows[0]["progressive_diagnostic_available"]
                ),
                **{
                    f"sampled_tail_score_output_relative_estimate_{name}": value
                    for name, value in quantiles(
                        [
                            float(
                                row[
                                    "sampled_tail_score_output_relative_estimate"
                                ]
                            )
                            for row in rows
                        ]
                    ).items()
                },
                **{
                    f"actual_tail_score_output_relative_{name}": value
                    for name, value in quantiles(
                        [
                            float(row["actual_tail_score_output_relative"])
                            for row in rows
                        ]
                    ).items()
                },
                "tail_score_probe_ucb95_coverage": sum(
                    float(
                        row["actual_tail_score_output_relative"]
                        <= row["sampled_tail_score_output_relative_ucb95"]
                    )
                    for row in rows
                )
                / len(rows),
                "tail_score_diagnostic_available": bool(
                    rows[0]["tail_score_diagnostic_available"]
                ),
                "exact_attention_mass_mean": sum(
                    float(row["exact_attention_mass_mean"]) for row in rows
                )
                / len(rows),
                "exact_residual_risk_mass_mean": sum(
                    float(row["exact_residual_risk_mass_mean"]) for row in rows
                )
                / len(rows),
                **{
                    f"predicted_layer_exact_value_rss_{name}": value
                    for name, value in quantiles(
                        [
                            float(row["predicted_layer_exact_value_rss"])
                            for row in rows
                        ]
                    ).items()
                },
                **{
                    f"exact_value_rss_calibration_ratio_{name}": value
                    for name, value in quantiles(
                        [
                            float(row["exact_value_rss_calibration_ratio"])
                            for row in rows
                        ]
                    ).items()
                },
                "actual_vs_exact_value_rss_pearson": pearson_correlation(
                    [float(row["relative_l2"]) for row in rows],
                    [
                        float(row["predicted_layer_exact_value_rss"])
                        for row in rows
                    ],
                ),
                "value_rss_diagnostic_available": bool(
                    rows[0]["value_rss_diagnostic_available"]
                ),
            }
        )
    score_calibration_summary = {
        name: quantiles(
            [float(row[name]) for row in score_calibration_rows]
        )
        for name in (
            "slope",
            "intercept",
            "uncalibrated_rmse",
            "calibrated_rmse",
            "sampled_crossfit_rmse",
            "sampled_exact_score_std",
            "sampled_normalized_crossfit_rmse",
            "exact_score_std",
            "normalized_calibrated_rmse",
            "fisher_score_distortion",
            "exact_softmax_kl",
            "sampled_fisher_score_distortion",
            "sampled_softmax_kl",
            "sampled_crossfit_softmax_kl",
            "sampled_score_output_relative_estimate",
            "sampled_score_output_relative_standard_error",
            "sampled_score_output_relative_ucb95",
            "first_order_score_output_relative",
            "actual_score_output_relative",
            "conformal_uncertainty_scale",
            "conformal_uncertainty_coverage",
        )
    }

    report = {
        "schema": "qksieve_output_risk_budget_v1",
        "setup": {
            "trace": str(args.trace),
            "topic": topic,
            "model_name_or_path": model_root,
            "history_tokens": int(
                next(iter(state_by_layer.values()))["key"].shape[2]
            ),
            "layers": sorted(state_by_layer),
            "decode_steps": max(len(rows) for rows in records_by_layer.values()),
            "fixed_top_ks": fixed_top_ks,
            "global_top_ks": global_top_ks,
            "global_rss_tolerances": global_rss_tolerances,
            "global_rss_floor_k": args.global_rss_floor_k,
            "balanced_rss_tolerances": balanced_rss_tolerances,
            "scalar_rss_tolerances": scalar_rss_tolerances,
            "affine_bound_tolerances": affine_bound_tolerances,
            "scalar_rss_statistics": scalar_rss_statistics,
            "rss_safety_factors": rss_safety_factors,
            "coverage_targets": targets,
            "gaussian_mass_samples": gaussian_mass_samples,
            "mass_ladder_samples": mass_ladder_samples,
            "mass_ladder_growth": args.mass_ladder_growth,
            "relative_risk_thresholds": relative_risk_thresholds,
            "minimum_top_k": args.minimum_top_k,
            "maximum_top_k": args.maximum_top_k,
            "key_rate_budget": args.key_rate_budget,
            "key_bit_levels": key_bit_levels,
            "key_refinement_rate_budget": (
                args.key_refinement_rate_budget or None
            ),
            "progressive_refinement_rounds": (
                args.progressive_refinement_rounds
            ),
            "key_quantizer": args.key_quantizer,
            "key_allocation_objective": args.key_allocation_objective,
            "key_allocation_query_source": (
                args.key_allocation_query_source
            ),
            "oas_alpha": (
                quantiles(oas_alpha_values) if oas_alpha_values else None
            ),
            "value_rank": args.value_rank,
            "value_bits": args.value_bits,
            "risk_bits": args.risk_bits,
            "score_calibration_samples": args.score_calibration_samples,
            "query_factor_source": args.query_factor_source,
            "query_factor_prefill_tokens": requested_prefill_tokens,
        },
        "algorithm": (
            "Evaluate attention-mass and output-risk selectors. The balanced "
            "joint RSS rule combines calibrated QK score uncertainty and "
            "Value-tail residual risk in quadrature, then gives every head a "
            "share of the same layer root-sum-square error budget."
        ),
        "claim_boundary": (
            "Real-QKV local layer-output diagnostic. It is not model-level PPL "
            "and uses a full sort only to test the numerical decision rule."
        ),
        "score_calibration": score_calibration_summary,
        "summary": summary_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    with (args.output_dir / "score_calibration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(score_calibration_rows[0])
        )
        writer.writeheader()
        writer.writerows(score_calibration_rows)
    with (args.output_dir / "key_allocations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(key_allocation_rows[0])
        )
        writer.writeheader()
        writer.writerows(key_allocation_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
