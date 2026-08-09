#!/usr/bin/env python
"""Query-aware 64-bit principal coding with bounded block-tail completion."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qksieve_block_coreset_20260802 import (
    block_coreset_tail_statistics,
    fit_block_coreset,
)
from analyze_qksieve_conditional_value_moments_20260802 import (
    combine_selected_and_tail,
)
from analyze_qksieve_control_variate_layer0_probe_20260802 import (
    load_layer0_activations,
    output_metrics,
)


def fit_binary_principal_projection(
    samples: torch.Tensor,
    bits: int,
    iterations: int,
    seed: int,
    initialization: str = "random",
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit BinaryPC's greedy signed residual dictionary on one head."""
    residual = samples.float().clone()
    if sample_weights is None:
        weights = torch.ones(
            residual.shape[0], device=residual.device, dtype=residual.dtype
        )
    else:
        weights = sample_weights.to(device=residual.device, dtype=residual.dtype)
        if weights.shape != residual.shape[:1]:
            raise ValueError("sample weights must have one value per sample")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("sample weights must be finite and strictly positive")
    weights = weights / weights.mean().clamp_min(1.0e-12)
    weight_sum = weights.sum().clamp_min(1.0e-12)
    generator = torch.Generator(device=samples.device).manual_seed(seed)
    vectors = []
    for _ in range(bits):
        if initialization == "random":
            vector = torch.randn(
                residual.shape[-1], generator=generator, device=residual.device
            )
        elif initialization == "spectral":
            variances = residual.square().sum(dim=0)
            vector = torch.zeros_like(variances)
            vector[int(torch.argmax(variances))] = 1.0
            for _ in range(max(2, iterations)):
                vector = residual.T @ (residual @ vector)
                vector = vector / vector.norm().clamp_min(1.0e-12)
        else:
            raise ValueError(f"unknown principal-code initialization: {initialization}")
        for _ in range(iterations):
            signs = (residual @ vector).sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            vector = (
                weights[:, None] * signs[:, None] * residual
            ).sum(dim=0) / weight_sum
        signs = (residual @ vector).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        # Refit the centroid for the final assignment.  Reusing the centroid
        # from the previous assignment can increase the residual and, over
        # many bits, overflow even though each exact alternating step is safe.
        vector = (
            weights[:, None] * signs[:, None] * residual
        ).sum(dim=0) / weight_sum
        residual = residual - signs[:, None] * vector
        vectors.append(vector)
    return torch.stack(vectors)


def encode_binary_principal(
    coordinates: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = coordinates.float().clone()
    codes = []
    for vector in projection.float():
        signs = (residual @ vector).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        residual = residual - signs[:, None] * vector
        codes.append(signs)
    return torch.stack(codes, dim=-1), residual.norm(dim=-1)


def binary_proxy_scores(
    codes: torch.Tensor,
    projection: torch.Tensor,
    query: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    probe = projection.float() @ query.float()
    quantization_scale = 127.0 / probe.abs().max().clamp_min(1.0e-6)
    probe = torch.round(probe * quantization_scale) / quantization_scale
    return codes.float() @ probe * scale


def evenly_spaced_indices(
    length: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def assert_numeric_backend_sane() -> None:
    """Fail closed when a broken CPU reduction backend corrupts experiments."""
    probe = torch.ones((2048, 64), dtype=torch.float32)
    observed = float(probe.mean())
    if abs(observed - 1.0) > 1.0e-6:
        raise RuntimeError(
            "PyTorch reduction self-test failed: mean(ones)="
            f"{observed}; use a working backend or one CPU thread"
        )


def quantize_log_error_norms(
    errors: torch.Tensor,
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, float]:
    """Quantize positive reconstruction norms with blockwise log scales."""
    if bits < 1:
        raise ValueError("error quantization needs at least one bit")
    levels = float((1 << bits) - 1)
    reconstructed = torch.empty_like(errors, dtype=torch.float32)
    for start in range(0, errors.numel(), block_size):
        stop = min(errors.numel(), start + block_size)
        block = errors[start:stop].float().clamp_min(1.0e-12).log2()
        lower = block.amin()
        upper = block.amax()
        step = (upper - lower) / levels
        if step <= 1.0e-12:
            reconstructed[start:stop] = lower.exp2()
        else:
            quantized = torch.round((block - lower) / step).clamp(0.0, levels)
            reconstructed[start:stop] = (lower + quantized * step).exp2()
    scale_bits = 32 * math.ceil(errors.numel() / block_size)
    return reconstructed, bits + scale_bits / errors.numel()


def quantize_blockwise_affine(
    values: torch.Tensor,
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, float]:
    """Blockwise min/max scalar quantization with explicit metadata cost."""
    if bits < 1 or block_size < 1:
        raise ValueError("bits and block size must be positive")
    levels = float((1 << bits) - 1)
    reconstructed = torch.empty_like(values, dtype=torch.float32)
    block_count = math.ceil(values.numel() / block_size)
    for block in range(block_count):
        start = block * block_size
        stop = min(values.numel(), start + block_size)
        current = values[start:stop].float()
        minimum = current.min()
        span = (current.max() - minimum).clamp_min(1.0e-12)
        codes = torch.round((current - minimum) * levels / span).clamp(0, levels)
        reconstructed[start:stop] = minimum + codes * span / levels
    bits_per_token = bits + 32.0 * block_count / values.numel()
    return reconstructed, bits_per_token


def fit_residual_codebook(
    residuals: torch.Tensor,
    clusters: int,
    iterations: int,
) -> torch.Tensor:
    """Fit a deterministic small vector codebook to principal-code residuals."""
    if clusters < 1 or clusters > residuals.shape[0] or iterations < 1:
        raise ValueError("invalid residual codebook shape or iteration count")
    samples = residuals.float()
    centroids = [samples[samples.square().sum(dim=-1).argmax()]]
    minimum_distance = (samples - centroids[0]).square().sum(dim=-1)
    for _ in range(1, clusters):
        next_centroid = samples[minimum_distance.argmax()]
        centroids.append(next_centroid)
        minimum_distance = torch.minimum(
            minimum_distance,
            (samples - next_centroid).square().sum(dim=-1),
        )
    codebook = torch.stack(centroids)
    for _ in range(iterations):
        assignments = torch.cdist(samples, codebook).argmin(dim=-1)
        updated = []
        for cluster in range(clusters):
            members = assignments == cluster
            updated.append(samples[members].mean(dim=0) if bool(members.any()) else codebook[cluster])
        next_codebook = torch.stack(updated)
        if torch.equal(next_codebook, codebook):
            break
        codebook = next_codebook
    return codebook


def encode_residual_codebook(
    residuals: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = torch.cdist(residuals.float(), codebook.float()).argmin(dim=-1)
    remainder = residuals.float() - codebook.float().index_select(0, assignments)
    return assignments, remainder.norm(dim=-1)


def fit_joint_kv_residual_codebook(
    key_residuals: torch.Tensor,
    values: torch.Tensor,
    clusters: int,
    iterations: int,
    value_weight: float = 1.0,
) -> dict[str, torch.Tensor | float]:
    """Fit one residual ID that jointly represents score-space K and Value."""
    if key_residuals.shape != values.shape:
        raise ValueError("joint K/V residual coding requires aligned equal-width tensors")
    if value_weight <= 0.0:
        raise ValueError("joint Value weight must be positive")

    key_residuals = key_residuals.float()
    values = values.float()
    value_mean = values.mean(dim=0)
    centered_values = values - value_mean
    key_scale = key_residuals.square().sum(dim=-1).mean().sqrt().clamp_min(1.0e-8)
    value_scale = centered_values.square().sum(dim=-1).mean().sqrt().clamp_min(1.0e-8)
    normalized = torch.cat(
        (
            key_residuals / key_scale,
            value_weight * centered_values / value_scale,
        ),
        dim=-1,
    )
    return {
        "codebook": fit_residual_codebook(normalized, clusters, iterations),
        "key_scale": float(key_scale),
        "value_scale": float(value_scale),
        "value_mean": value_mean,
        "value_weight": float(value_weight),
        "head_dim": int(key_residuals.shape[-1]),
    }


def encode_joint_kv_residual_codebook(
    key_residuals: torch.Tensor,
    values: torch.Tensor,
    model: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode aligned K/V and return IDs, score-space residual norms, K centroids."""
    codebook = model["codebook"]
    value_mean = model["value_mean"]
    if not isinstance(codebook, torch.Tensor) or not isinstance(value_mean, torch.Tensor):
        raise TypeError("joint residual model tensors are malformed")
    head_dim = int(model["head_dim"])
    key_scale = float(model["key_scale"])
    value_scale = float(model["value_scale"])
    value_weight = float(model["value_weight"])
    normalized = torch.cat(
        (
            key_residuals.float() / key_scale,
            value_weight * (values.float() - value_mean.float()) / value_scale,
        ),
        dim=-1,
    )
    assignments = torch.cdist(normalized, codebook.float()).argmin(dim=-1)
    key_centroids = codebook[:, :head_dim].float() * key_scale
    remainder = key_residuals.float() - key_centroids.index_select(0, assignments)
    return assignments, remainder.norm(dim=-1), key_centroids


def fit_product_kv_residual_codebook(
    key_residuals: torch.Tensor,
    values: torch.Tensor,
    total_bits: int,
    key_bits: int,
    iterations: int,
) -> dict[str, torch.Tensor | int | float]:
    """Split one packed residual ID into independent K-score and Value subcodes."""
    if key_residuals.shape != values.shape:
        raise ValueError("product K/V coding requires aligned equal-width tensors")
    value_bits = total_bits - key_bits
    if key_bits < 1 or value_bits < 1:
        raise ValueError("product coding requires at least one K bit and one Value bit")

    values = values.float()
    value_mean = values.mean(dim=0)
    centered_values = values - value_mean
    value_scale = centered_values.square().sum(dim=-1).mean().sqrt().clamp_min(1.0e-8)
    return {
        "key_codebook": fit_residual_codebook(
            key_residuals.float(), 1 << key_bits, iterations
        ),
        "value_codebook": fit_residual_codebook(
            centered_values / value_scale, 1 << value_bits, iterations
        ),
        "value_mean": value_mean,
        "value_scale": float(value_scale),
        "key_bits": int(key_bits),
        "value_bits": int(value_bits),
        "total_bits": int(total_bits),
    }


def encode_product_kv_residual_codebook(
    key_residuals: torch.Tensor,
    values: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a packed product ID and return its score-space K residual."""
    key_codebook = model["key_codebook"]
    value_codebook = model["value_codebook"]
    value_mean = model["value_mean"]
    if not all(
        isinstance(item, torch.Tensor)
        for item in (key_codebook, value_codebook, value_mean)
    ):
        raise TypeError("product residual model tensors are malformed")
    assert isinstance(key_codebook, torch.Tensor)
    assert isinstance(value_codebook, torch.Tensor)
    assert isinstance(value_mean, torch.Tensor)
    value_scale = float(model["value_scale"])
    value_bits = int(model["value_bits"])
    key_assignments = torch.cdist(
        key_residuals.float(), key_codebook.float()
    ).argmin(dim=-1)
    value_assignments = torch.cdist(
        (values.float() - value_mean.float()) / value_scale,
        value_codebook.float(),
    ).argmin(dim=-1)
    packed_assignments = (key_assignments << value_bits) | value_assignments
    remainder = key_residuals.float() - key_codebook.float().index_select(
        0, key_assignments
    )
    return packed_assignments, remainder.norm(dim=-1), key_codebook.float()


def query_windows(
    query: torch.Tensor,
    history_tokens: int,
    calibration_tokens: int,
    query_tokens: int,
    calibration_source: str,
    heldout_gap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split prior-turn calibration queries from held-out next-turn queries."""
    if calibration_source == "post_history":
        calibration_start = history_tokens
        heldout_start = history_tokens + calibration_tokens + heldout_gap
    elif calibration_source == "history_tail":
        calibration_start = history_tokens - calibration_tokens
        heldout_start = history_tokens + heldout_gap
    else:
        raise ValueError(f"unknown calibration source: {calibration_source}")
    if calibration_start < 0:
        raise ValueError("calibration window starts before the sequence")
    calibration = query[
        calibration_start : calibration_start + calibration_tokens
    ]
    heldout = query[heldout_start : heldout_start + query_tokens]
    if calibration.shape[0] != calibration_tokens or heldout.shape[0] != query_tokens:
        raise ValueError("query tensor does not cover requested calibration windows")
    return calibration, heldout


def query_metric_factors(
    calibration_queries: torch.Tensor,
    shrinkage: float | str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return exact dual factors whose Key norm is expected score MSE."""
    dimension = calibration_queries.shape[-1]
    covariance = (
        calibration_queries.float().T @ calibration_queries.float()
    ) / float(calibration_queries.shape[0])
    if isinstance(shrinkage, str):
        if shrinkage != "oas":
            raise ValueError(f"unknown metric shrinkage: {shrinkage}")
        shrinkage = oas_second_moment_shrinkage(
            covariance, calibration_queries.shape[0]
        )
    shrinkage = float(shrinkage)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("metric shrinkage must be in [0, 1] or 'oas'")
    isotropic = torch.trace(covariance) / float(dimension)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * isotropic * torch.eye(
        dimension, device=covariance.device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = isotropic.clamp_min(1.0e-8) * 1.0e-4
    eigenvalues = eigenvalues.clamp_min(floor)
    # Symmetric factors are invariant to arbitrary rotations inside repeated
    # eigenspaces.  A bare U sqrt(Lambda) coordinate factor is mathematically
    # valid for dot products but makes the downstream binary code basis-sensitive.
    key_factor = (
        eigenvectors * eigenvalues.sqrt()[None, :]
    ) @ eigenvectors.T
    query_factor = (
        eigenvectors * eigenvalues.rsqrt()[None, :]
    ) @ eigenvectors.T
    return query_factor, key_factor, shrinkage


def oas_second_moment_shrinkage(
    second_moment: torch.Tensor,
    sample_count: int,
) -> float:
    """OAS-style isotropic shrinkage for the uncentered query second moment."""
    dimension = second_moment.shape[0]
    trace = torch.trace(second_moment)
    trace_square = second_moment.square().sum()
    numerator = (1.0 - 2.0 / dimension) * trace_square + trace.square()
    denominator = (sample_count + 1.0 - 2.0 / dimension) * (
        trace_square - trace.square() / dimension
    )
    if denominator <= 0:
        return 1.0
    return float((numerator / denominator).clamp(0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--texts", type=Path, nargs="+", required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--calibration_tokens", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=4)
    parser.add_argument(
        "--calibration_source",
        choices=("post_history", "history_tail"),
        default="post_history",
    )
    parser.add_argument("--heldout_gap", type=int, default=0)
    parser.add_argument("--fractions", default="0.01,0.02,0.06")
    parser.add_argument("--binary_bits", type=int, default=64)
    parser.add_argument("--projection_iterations", type=int, default=4)
    parser.add_argument("--projection_sample_stride", type=int, default=8)
    parser.add_argument(
        "--projection_sample_count",
        type=int,
        default=4096,
        help="Length-independent reservoir size; <=0 falls back to stride.",
    )
    parser.add_argument(
        "--projection_initialization",
        choices=("random", "spectral"),
        default="spectral",
    )
    parser.add_argument("--error_rescue_ratio", type=float, default=0.1)
    parser.add_argument(
        "--risk_lambdas",
        default="",
        help="Comma-separated query-dependent UCB multipliers; empty disables.",
    )
    parser.add_argument("--risk_error_bits", type=int, default=4)
    parser.add_argument("--risk_error_block_size", type=int, default=256)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument(
        "--metric_shrinkage",
        default="oas",
        help="Float in [0,1], or 'oas' for analytic query-metric shrinkage.",
    )
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--key_mean_bits", type=int, default=8)
    parser.add_argument("--value_mean_bits", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_with_error_rescue(
    scores: torch.Tensor,
    errors: torch.Tensor,
    keep: int,
    rescue_ratio: float,
) -> torch.Tensor:
    if rescue_ratio <= 0.0:
        return torch.topk(scores, keep, sorted=False).indices
    rescue_count = min(keep, max(1, int(keep * rescue_ratio)))
    rescue = torch.topk(errors, rescue_count, sorted=False).indices
    rescued_scores = scores.clone()
    rescued_scores[rescue] = scores.max() + 1.0
    return torch.topk(rescued_scores, keep, sorted=False).indices


def evaluate_text(
    text_path: Path,
    token_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query, key, value, metadata = load_layer0_activations(args.model, token_ids)
    history_count = args.history_tokens
    history_key = key[:history_count]
    history_value = value[:history_count]
    calibration, heldout = query_windows(
        query,
        history_count,
        args.calibration_tokens,
        args.query_tokens,
        args.calibration_source,
        args.heldout_gap,
    )
    group_size = int(metadata["gqa_groups"])
    scale = float(metadata["head_dim"] ** -0.5)
    fractions = [float(item) for item in args.fractions.split(",")]
    risk_lambdas = [
        float(item) for item in args.risk_lambdas.split(",") if item.strip()
    ]
    rows: list[dict[str, Any]] = []
    resolved_metric_shrinkages: list[float] = []

    for kv_head in range(int(metadata["kv_heads"])):
        head_key = history_key[:, kv_head].contiguous()
        head_value = history_value[:, kv_head].contiguous()
        if args.projection_sample_count > 0:
            projection_indices = evenly_spaced_indices(
                history_count,
                args.projection_sample_count,
                head_key.device,
            )
            projection_key = head_key.index_select(0, projection_indices)
        else:
            projection_key = head_key[:: args.projection_sample_stride]
        head_calibration = calibration[
            :, kv_head * group_size : (kv_head + 1) * group_size
        ].reshape(-1, int(metadata["head_dim"]))
        query_factor, key_factor, _ = qk_balanced_factors(
            projection_key,
            head_calibration,
            args.query_shrinkage,
        )
        balanced_coordinates = head_key @ key_factor
        raw_projection = fit_binary_principal_projection(
            projection_key,
            args.binary_bits,
            args.projection_iterations,
            seed=1000 + kv_head,
            initialization=args.projection_initialization,
        )
        balanced_projection = fit_binary_principal_projection(
            balanced_coordinates.index_select(0, projection_indices)
            if args.projection_sample_count > 0
            else balanced_coordinates[:: args.projection_sample_stride],
            args.binary_bits,
            args.projection_iterations,
            seed=2000 + kv_head,
            initialization=args.projection_initialization,
        )
        raw_codes, raw_errors = encode_binary_principal(head_key, raw_projection)
        balanced_codes, balanced_errors = encode_binary_principal(
            balanced_coordinates, balanced_projection
        )
        (
            metric_query_factor,
            metric_key_factor,
            resolved_metric_shrinkage,
        ) = query_metric_factors(
            head_calibration, args.metric_shrinkage
        )
        resolved_metric_shrinkages.append(resolved_metric_shrinkage)
        metric_coordinates = head_key @ metric_key_factor
        metric_projection = fit_binary_principal_projection(
            metric_coordinates.index_select(0, projection_indices)
            if args.projection_sample_count > 0
            else metric_coordinates[:: args.projection_sample_stride],
            args.binary_bits,
            args.projection_iterations,
            seed=3000 + kv_head,
            initialization=args.projection_initialization,
        )
        metric_codes, metric_errors = encode_binary_principal(
            metric_coordinates, metric_projection
        )
        quantized_metric_errors, risk_aux_bits = quantize_log_error_norms(
            metric_errors,
            args.risk_error_bits,
            args.risk_error_block_size,
        )
        coreset = fit_block_coreset(
            balanced_coordinates,
            head_value,
            args.block_size,
            cluster_count=1,
            moment_bits=2,
            iterations=1,
            full_score_coordinates=head_key,
            value_moment_bits=args.value_mean_bits,
            full_score_moment_bits=args.key_mean_bits,
        )

        for query_offset in range(args.query_tokens):
            for group_offset in range(group_size):
                query_head = kv_head * group_size + group_offset
                current_query = heldout[query_offset, query_head]
                exact_scores = head_key @ current_query * scale
                full_weights = torch.softmax(exact_scores, dim=0)
                full_output = full_weights @ head_value
                balanced_query = current_query @ query_factor
                metric_query = current_query @ metric_query_factor
                raw_proxy = binary_proxy_scores(
                    raw_codes, raw_projection, current_query, scale
                )
                balanced_proxy = binary_proxy_scores(
                    balanced_codes,
                    balanced_projection,
                    balanced_query,
                    scale,
                )
                metric_proxy = binary_proxy_scores(
                    metric_codes,
                    metric_projection,
                    metric_query,
                    scale,
                )
                selector_states = {
                    f"binarypc_keymse{args.binary_bits}": (
                        raw_proxy,
                        raw_errors,
                        head_key,
                        current_query,
                        0.0,
                    ),
                    f"qaware_binarypc{args.binary_bits}": (
                        balanced_proxy,
                        balanced_errors,
                        balanced_coordinates,
                        balanced_query,
                        0.0,
                    ),
                    f"qmetric_binarypc{args.binary_bits}": (
                        metric_proxy,
                        metric_errors,
                        metric_coordinates,
                        metric_query,
                        0.0,
                    ),
                }
                score_uncertainty = (
                    quantized_metric_errors
                    * metric_query.float().norm()
                    / float(metadata["head_dim"])
                )
                for risk_lambda in risk_lambdas:
                    selector_states[
                        f"qmetric_ucb{risk_lambda:g}_int{args.risk_error_bits}_"
                        f"{args.binary_bits}"
                    ] = (
                        metric_proxy + risk_lambda * score_uncertainty,
                        metric_errors,
                        metric_coordinates,
                        metric_query,
                        risk_aux_bits,
                    )
                for selector, (
                    proxy_scores,
                    errors,
                    selector_coordinates,
                    selector_query,
                    selector_aux_bits,
                ) in selector_states.items():
                    for fraction in fractions:
                        keep = max(1, math.ceil(history_count * fraction))
                        selected = selected_with_error_rescue(
                            proxy_scores,
                            errors,
                            keep,
                            args.error_rescue_ratio,
                        )
                        selected_output = torch.softmax(
                            exact_scores.index_select(0, selected), dim=0
                        ) @ head_value.index_select(0, selected)
                        reference = exact_scores.index_select(0, selected).amin()
                        tail_z, tail_y, diagnostics = (
                            block_coreset_tail_statistics(
                                selector_coordinates,
                                head_value,
                                selector_query * scale,
                                selected,
                                reference,
                                coreset,
                                selected_conditioned=False,
                                full_score_coordinates=head_key,
                                full_score_direction=current_query * scale,
                            )
                        )
                        tail_output = combine_selected_and_tail(
                            exact_scores,
                            exact_scores,
                            head_value,
                            selected,
                            tail_y,
                            tail_z,
                            1.0,
                        )
                        for method, output, tail_bits in (
                            ("selected_only", selected_output, 0.0),
                            (
                                "blockmean_tail",
                                tail_output,
                                float(diagnostics["bits_per_token"]),
                            ),
                        ):
                            rows.append(
                                {
                                    "text": text_path.stem,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "query_offset": query_offset,
                                    "selector": selector,
                                    "selector_bits_per_token": (
                                        args.binary_bits + selector_aux_bits
                                    ),
                                    "tail_bits_per_token": tail_bits,
                                    "fraction": fraction,
                                    "selected_tokens": keep,
                                    "method": method,
                                    **output_metrics(output, full_output),
                                    "selected_mass": float(
                                        full_weights.index_select(0, selected).sum()
                                    ),
                                    "top1_recall": float(
                                        torch.isin(
                                            torch.argmax(exact_scores)[None], selected
                                        )[0]
                                    ),
                                    "resolved_metric_shrinkage": (
                                        resolved_metric_shrinkage
                                    ),
                                }
                            )
    metadata = dict(metadata)
    metadata["resolved_metric_shrinkages"] = resolved_metric_shrinkages
    return rows, metadata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["selector"], row["fraction"], row["method"])].append(row)
    result = []
    for (selector, fraction, method), group in sorted(groups.items()):
        errors = torch.tensor([row["relative_l2"] for row in group])
        result.append(
            {
                "selector": selector,
                "fraction": fraction,
                "method": method,
                "conditions": len(group),
                "relative_l2_mean": float(errors.mean()),
                "relative_l2_p90": float(torch.quantile(errors, 0.9)),
                "relative_l2_worst": float(errors.max()),
                "cosine_mean": sum(row["cosine"] for row in group) / len(group),
                "selected_mass_mean": sum(
                    row["selected_mass"] for row in group
                )
                / len(group),
                "top1_recall_mean": sum(row["top1_recall"] for row in group)
                / len(group),
                "selector_bits_per_token": group[0]["selector_bits_per_token"],
                "tail_bits_per_token": group[0]["tail_bits_per_token"],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    assert_numeric_backend_sane()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    needed = args.history_tokens + args.heldout_gap + args.query_tokens
    if args.calibration_source == "post_history":
        needed += args.calibration_tokens
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for text_path in args.texts:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=needed,
        ).input_ids[0, :needed]
        if token_ids.numel() < needed:
            raise ValueError(f"{text_path} contains fewer than {needed} tokens")
        current_rows, metadata = evaluate_text(text_path, token_ids, args)
        rows.extend(current_rows)
    payload = {
        "schema": "qaware_binarypc_blockmean_layer0_v1",
        "contract": {
            "scope": "real Qwen3 layer-0 mechanism audit; not end-to-end quality",
            "model": str(args.model),
            "history_tokens": args.history_tokens,
            "calibration_tokens": args.calibration_tokens,
            "calibration_source": args.calibration_source,
            "heldout_gap": args.heldout_gap,
            "query_tokens": args.query_tokens,
            "binary_bits": args.binary_bits,
            "projection_sample_stride": args.projection_sample_stride,
            "projection_sample_count": args.projection_sample_count,
            "projection_initialization": args.projection_initialization,
            "error_rescue_ratio": args.error_rescue_ratio,
            "metric_shrinkage": args.metric_shrinkage,
            "model_metadata": metadata,
        },
        "aggregate": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
