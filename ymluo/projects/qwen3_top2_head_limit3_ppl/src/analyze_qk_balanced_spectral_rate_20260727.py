from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
    reconstruct,
)
from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    query_int8,
    selection_metrics,
)


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


def parse_named_allocations(value: str) -> dict[str, list[int]]:
    allocations: dict[str, list[int]] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "fixed allocations must use name=bit-bit-... syntax"
            )
        name, spec = item.split("=", 1)
        allocation = [int(bits) for bits in spec.split("-")]
        if len(allocation) != GROUP_COUNT:
            raise ValueError(
                f"{name} must specify exactly {GROUP_COUNT} spectral bands"
            )
        if any(bits not in ZERO_BIT_LEVELS for bits in allocation):
            raise ValueError(
                f"{name} uses bits outside {tuple(ZERO_BIT_LEVELS)}"
            )
        allocations[f"qk_fixed_{name.strip()}"] = allocation
    return allocations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def covariance(values: torch.Tensor) -> torch.Tensor:
    return values.transpose(0, 1) @ values / max(1, values.shape[0])


def symmetric_covariance_factors(
    covariance_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.float())
    floor = eigenvalues.amax() * 1.0e-8 + 1.0e-12
    eigenvalues = eigenvalues.clamp_min(floor)
    square_root = (
        eigenvectors
        @ torch.diag(eigenvalues.sqrt())
        @ eigenvectors.transpose(0, 1)
    )
    inverse_square_root = (
        eigenvectors
        @ torch.diag(eigenvalues.rsqrt())
        @ eigenvectors.transpose(0, 1)
    )
    return square_root, inverse_square_root


def qk_balanced_factors(
    sampled_key: torch.Tensor,
    calibration_queries: torch.Tensor,
    query_shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_covariance = covariance(sampled_key.float())
    query_covariance = covariance(calibration_queries.float())
    isotropic_scale = query_covariance.diagonal().mean()
    regularized_query = (
        (1.0 - query_shrinkage) * query_covariance
        + query_shrinkage
        * isotropic_scale
        * torch.eye(
            query_covariance.shape[0],
            device=query_covariance.device,
        )
    )
    key_sqrt, key_inverse_sqrt = symmetric_covariance_factors(
        key_covariance
    )
    query_sqrt, query_inverse_sqrt = symmetric_covariance_factors(
        regularized_query
    )
    left, singular_values, right_h = torch.linalg.svd(
        query_sqrt @ key_sqrt,
        full_matrices=False,
    )
    scale = singular_values.sqrt().unsqueeze(0)
    query_factor = (query_inverse_sqrt @ left) * scale
    key_factor = (
        key_inverse_sqrt @ right_h.transpose(0, 1)
    ) * scale
    return query_factor, key_factor, singular_values


def distortion_table(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    sampled_coefficients = coefficients[::32]
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = sampled_coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs = {}
        for bits in ZERO_BIT_LEVELS:
            residual = key_band - quantize_band(key_band, bits)
            score_error = query_band @ residual.transpose(0, 1)
            costs[bits] = score_error.square().mean()
        output.append(costs)
    return output


def metric_scale_quantize_band(
    values: torch.Tensor,
    bits: int,
    calibration_queries: torch.Tensor,
    metric_mode: str = "full",
) -> torch.Tensor:
    if metric_mode not in {"identity", "diagonal", "full"}:
        raise ValueError(f"unsupported scale metric: {metric_mode}")
    if bits == 0:
        return torch.zeros_like(values)
    working = values.float()
    if bits == 1:
        codes = torch.where(working >= 0.0, 1.0, -1.0)
    else:
        maximum_code = (1 << (bits - 1)) - 1
        initial_scale = (
            working.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
            / float(maximum_code)
        )
        codes = torch.round(working / initial_scale).clamp(
            -maximum_code,
            maximum_code,
        )
    if metric_mode == "identity":
        weighted_codes = codes
    elif metric_mode == "diagonal":
        variances = calibration_queries.float().square().mean(dim=0)
        weighted_codes = codes * variances
    else:
        metric = (
            calibration_queries.float().transpose(0, 1)
            @ calibration_queries.float()
        )
        metric /= max(1, calibration_queries.shape[0])
        weighted_codes = codes @ metric
    numerator = (weighted_codes * working).sum(dim=-1, keepdim=True)
    denominator = (
        (weighted_codes * codes).sum(dim=-1, keepdim=True)
        .clamp_min(1.0e-12)
    )
    scale = (numerator / denominator).clamp_min(0.0)
    return (codes * scale).to(values.dtype)


def topk_boundary_weights(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    top_fraction: float,
    include_global_floor: bool,
) -> torch.Tensor:
    scores = calibration_queries.float() @ coefficients.float().transpose(
        0,
        1,
    )
    token_count = int(scores.shape[-1])
    top_count = min(
        token_count,
        max(1, math.ceil(top_fraction * token_count)),
    )
    lower_rank = min(token_count, max(top_count, 2 * top_count))
    upper_rank = max(1, math.ceil(top_count / 2))
    ranked = torch.topk(scores, k=lower_rank, dim=-1).values
    threshold = ranked[:, top_count - 1]
    upper_score = ranked[:, upper_rank - 1]
    lower_score = ranked[:, lower_rank - 1]
    score_scale = scores.std(dim=-1).clamp_min(1.0e-6)
    bandwidth = (
        0.5 * (upper_score - lower_score).abs()
    ).clamp_min(1.0e-3 * score_scale)
    normalized_distance = (
        scores - threshold.unsqueeze(-1)
    ) / bandwidth.unsqueeze(-1)
    boundary_kernel = torch.exp(
        -0.5 * normalized_distance.square()
    )
    normalized_kernel = boundary_kernel / boundary_kernel.mean(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    return (
        1.0 + normalized_kernel
        if include_global_floor
        else normalized_kernel
    )


def boundary_scale_quantized_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    boundary_weights: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[:, start:stop].float()
        query_band = calibration_queries[:, start:stop].float()
        exact_band_scores = query_band @ key_band.transpose(0, 1)
        quantized = {}
        for bits in ZERO_BIT_LEVELS:
            if bits == 0:
                quantized[bits] = torch.zeros_like(key_band)
                continue
            if bits == 1:
                codes = torch.where(key_band >= 0.0, 1.0, -1.0)
            else:
                maximum_code = (1 << (bits - 1)) - 1
                initial_scale = (
                    key_band.abs().amax(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1.0e-8)
                    / float(maximum_code)
                )
                codes = torch.round(key_band / initial_scale).clamp(
                    -maximum_code,
                    maximum_code,
                )
            code_scores = query_band @ codes.transpose(0, 1)
            numerator = (
                boundary_weights * code_scores * exact_band_scores
            ).sum(dim=0)
            denominator = (
                boundary_weights * code_scores.square()
            ).sum(dim=0).clamp_min(1.0e-12)
            scales = (numerator / denominator).clamp_min(0.0)
            quantized[bits] = codes * scales.unsqueeze(-1)
        output.append(quantized)
    return output


def fit_token_score_affine(
    coefficients: torch.Tensor,
    reconstructed: torch.Tensor,
    calibration_queries: torch.Tensor,
    ridge: float,
    fit_bias: bool,
    fit_gain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    quantized_queries = torch.stack(
        [query_int8(query) for query in calibration_queries],
        dim=0,
    )
    exact_scores = (
        calibration_queries.float() @ coefficients.float().transpose(0, 1)
    )
    proxy_scores = (
        quantized_queries.float() @ reconstructed.float().transpose(0, 1)
    )
    if fit_bias and fit_gain:
        exact_mean = exact_scores.mean(dim=0)
        proxy_mean = proxy_scores.mean(dim=0)
        exact_working = exact_scores - exact_mean
        proxy_working = proxy_scores - proxy_mean
    else:
        exact_mean = torch.zeros_like(exact_scores[0])
        proxy_mean = torch.zeros_like(proxy_scores[0])
        exact_working = exact_scores
        proxy_working = proxy_scores
    if fit_gain:
        denominator = proxy_working.square().sum(dim=0).clamp_min(1.0e-12)
        numerator = (proxy_working * exact_working).sum(dim=0)
        gain = (numerator / denominator + ridge) / (1.0 + ridge)
        gain = gain.clamp(0.25, 4.0)
        bias = (
            exact_mean - gain * proxy_mean
            if fit_bias
            else torch.zeros_like(gain)
        )
    else:
        gain = torch.ones_like(exact_scores[0])
        bias = (
            (exact_scores - proxy_scores).mean(dim=0) / (1.0 + ridge)
            if fit_bias
            else torch.zeros_like(gain)
        )
    return gain, bias


def quantize_token_score_metadata(
    gain: torch.Tensor,
    bias: torch.Tensor,
    bits: int,
    include_gain: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits not in (4, 8):
        raise ValueError("metadata bits must be 4 or 8")
    maximum_code = (1 << (bits - 1)) - 1

    def quantize(values: torch.Tensor) -> torch.Tensor:
        scale = values.abs().amax().clamp_min(1.0e-8) / maximum_code
        return (
            torch.round(values / scale)
            .clamp(-maximum_code, maximum_code)
            * scale
        )

    quantized_gain = (
        1.0 + quantize(gain - 1.0)
        if include_gain
        else torch.ones_like(gain)
    )
    return quantized_gain, quantize(bias)


def fit_empirical_bayes_score_bias(
    coefficients: torch.Tensor,
    reconstructed: torch.Tensor,
    calibration_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    quantized_queries = torch.stack(
        [query_int8(query) for query in calibration_queries],
        dim=0,
    )
    residual_scores = (
        calibration_queries.float() @ coefficients.float().transpose(0, 1)
        - quantized_queries.float()
        @ reconstructed.float().transpose(0, 1)
    )
    residual_mean = residual_scores.mean(dim=0)
    mean_noise = residual_scores.var(
        dim=0,
        unbiased=True,
    ).mean() / residual_scores.shape[0]
    observed_mean_power = residual_mean.square().mean().clamp_min(1.0e-12)
    shrinkage = float(
        (1.0 - mean_noise / observed_mean_power)
        .clamp(0.0, 1.0)
        .item()
    )
    return (
        torch.ones_like(residual_mean),
        residual_mean * shrinkage,
        shrinkage,
    )


def distortion_table_from_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    quantized_bands: list[dict[int, torch.Tensor]],
) -> list[dict[int, torch.Tensor]]:
    sampled_indices = torch.arange(
        0,
        coefficients.shape[0],
        32,
        device=coefficients.device,
    )
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[sampled_indices, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs = {}
        for bits in ZERO_BIT_LEVELS:
            reconstructed = quantized_bands[group_index][bits][
                sampled_indices
            ]
            score_error = (
                query_band
                @ (key_band - reconstructed).transpose(0, 1)
            )
            costs[bits] = score_error.square().mean()
        output.append(costs)
    return output


def boundary_distortion_table_from_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    quantized_bands: list[dict[int, torch.Tensor]],
    boundary_weights: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    sampled_indices = torch.arange(
        0,
        coefficients.shape[0],
        32,
        device=coefficients.device,
    )
    sampled_weights = boundary_weights[:, sampled_indices]
    weight_denominator = sampled_weights.sum().clamp_min(1.0e-12)
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[sampled_indices, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs = {}
        for bits in ZERO_BIT_LEVELS:
            reconstructed = quantized_bands[group_index][bits][
                sampled_indices
            ]
            score_error = (
                query_band
                @ (key_band - reconstructed).transpose(0, 1)
            )
            costs[bits] = (
                sampled_weights * score_error.square()
            ).sum() / weight_denominator
        output.append(costs)
    return output


def softmax_fisher_cost(
    score_error: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    mean_error = (attention * score_error).sum(dim=-1)
    second_moment = (attention * score_error.square()).sum(dim=-1)
    return (second_moment - mean_error.square()).clamp_min(0.0).mean()


def fisher_distortion_table(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    scaling: float,
) -> list[dict[int, torch.Tensor]]:
    sampled_coefficients = coefficients[::32]
    exact_scores = (
        calibration_queries @ sampled_coefficients.transpose(0, 1)
    ) * scaling
    attention = torch.softmax(exact_scores, dim=-1)
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = sampled_coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs = {}
        for bits in ZERO_BIT_LEVELS:
            residual = key_band - quantize_band(key_band, bits)
            score_error = (
                query_band @ residual.transpose(0, 1)
            ) * scaling
            costs[bits] = softmax_fisher_cost(score_error, attention)
        output.append(costs)
    return output


@lru_cache(maxsize=None)
def feasible_allocations(
    total_rate_budget: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        allocation
        for allocation in itertools.product(
            ZERO_BIT_LEVELS,
            repeat=GROUP_COUNT,
        )
        if sum(
            bits + int(bits > 0) for bits in allocation
        )
        <= total_rate_budget
    )


def joint_qmse_allocation(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    total_rate_budget: int,
) -> tuple[int, ...]:
    sampled_coefficients = coefficients[::32]
    error_options = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = sampled_coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        error_options.append(
            torch.stack(
                [
                    query_band
                    @ (
                        key_band - quantize_band(key_band, bits)
                    ).transpose(0, 1)
                    for bits in ZERO_BIT_LEVELS
                ],
                dim=0,
            )
        )
    flattened = torch.stack(error_options, dim=0).flatten(2)
    option_count = GROUP_COUNT * len(ZERO_BIT_LEVELS)
    vectors = flattened.reshape(option_count, -1)
    gram = vectors @ vectors.transpose(0, 1)
    gram /= max(1, vectors.shape[-1])

    allocations = feasible_allocations(total_rate_budget)
    level_index = {
        bits: index for index, bits in enumerate(ZERO_BIT_LEVELS)
    }
    option_indices = torch.tensor(
        [
            [
                group * len(ZERO_BIT_LEVELS) + level_index[bits]
                for group, bits in enumerate(allocation)
            ]
            for allocation in allocations
        ],
        dtype=torch.long,
        device=coefficients.device,
    )
    costs = torch.zeros(
        len(allocations),
        dtype=gram.dtype,
        device=gram.device,
    )
    for left in range(GROUP_COUNT):
        left_index = option_indices[:, left]
        costs += gram[left_index, left_index]
        for right in range(left + 1, GROUP_COUNT):
            costs += 2.0 * gram[
                left_index,
                option_indices[:, right],
            ]
    selected = int(torch.argmin(costs).item())
    return allocations[selected]


def aggregate(
    rows: list[dict[str, Any]],
    allocation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allocations_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows:
        allocations_by_method[str(row["method"])].append(row)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    output = []
    for (method, fraction), items in sorted(grouped.items()):
        method_allocations = allocations_by_method[method]
        total_bits = sum(
            int(row["total_index_bits"]) for row in method_allocations
        ) / len(method_allocations)
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "total_index_bits_mean": total_bits,
            "index_ratio_of_full_kv": total_bits / FULL_KV_BITS,
            "factor_identity_error_max": max(
                float(row["factor_identity_error"])
                for row in method_allocations
            ),
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
            "score_rmse",
        ):
            for statistic, value in summarize(
                float(row[field]) for row in items
            ).items():
                result[f"{field}_{statistic}"] = value
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Key-PCA and QK-balanced biorthogonal coordinates under "
            "the same metadata-aware variable-bit spectral rate."
        )
    )
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument(
        "--calibration_source",
        choices=("decode_prefix", "prefill_tail"),
        default="decode_prefix",
        help=(
            "Use the first decode records, or the captured final prefill "
            "Queries. The frozen QKSieve paper path uses prefill_tail."
        ),
    )
    parser.add_argument("--total_rate_budget", type=int, default=15)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--selected_fractions", default="0.01,0.02,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.01)
    parser.add_argument(
        "--include_key_centered_qk",
        action="store_true",
        help=(
            "Also evaluate QK-balanced coordinates after removing the "
            "per-head Key mean, which is a score-ranking invariant."
        ),
    )
    parser.add_argument(
        "--qk_fixed_allocations",
        default="",
        help=(
            "Optional comma-separated named fixed allocations, for example "
            "'444=4-4-4-0-0-0-0-0,822=8-2-2-0-0-0-0-0'."
        ),
    )
    return parser.parse_args()


def resolve_calibration_and_evaluation(
    payload: dict[str, Any],
    layer: int,
    layer_records: list[dict[str, Any]],
    calibration_steps: int,
    calibration_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]], int]:
    if calibration_source == "decode_prefix":
        if len(layer_records) <= calibration_steps:
            raise ValueError(f"layer {layer} has no held-out queries")
        calibration = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[:calibration_steps]
            ],
            dim=0,
        )
        return calibration, layer_records[calibration_steps:], calibration_steps

    if calibration_source != "prefill_tail":
        raise ValueError(f"unknown calibration source: {calibration_source}")
    prefill_queries = payload.get("prefill_queries")
    if not isinstance(prefill_queries, dict):
        raise ValueError("trace has no captured prefill Queries")
    raw = prefill_queries.get(layer, prefill_queries.get(str(layer)))
    if raw is None:
        raise ValueError(f"trace has no prefill Queries for layer {layer}")
    query = raw.to(device).float()
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            f"layer {layer} prefill Query shape must be [1,H,N,D], got {query.shape}"
        )
    if query.shape[2] < calibration_steps:
        raise ValueError(
            f"layer {layer} has only {query.shape[2]} prefill Queries; "
            f"need {calibration_steps}"
        )
    calibration = query[0, :, -calibration_steps:, :].permute(1, 0, 2)
    return calibration.contiguous(), layer_records, 0


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    selected_fractions = parse_floats(args.selected_fractions)
    fixed_qk_allocations = parse_named_allocations(args.qk_fixed_allocations)
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows = []
    allocation_rows = []
    spectrum_rows = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda row: int(row["step"]))
        calibration, evaluation_records, evaluation_start = (
            resolve_calibration_and_evaluation(
                payload,
                layer,
                layer_records,
                args.calibration_steps,
                args.calibration_source,
                device,
            )
        )
        raw_key = next(
            (
                record.get("key")
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no key tensor")
        key = raw_key.to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        kv_head_count = int(key.shape[0])
        query_head_count = int(layer_records[0]["query"].shape[1])
        groups = query_head_count // kv_head_count
        calibration_scaling = float(layer_records[0]["scaling"])

        prepared = []
        for kv_head in range(kv_head_count):
            head_key = key[kv_head]
            head_calibration = calibration[
                :, kv_head * groups : (kv_head + 1) * groups
            ].reshape(-1, head_key.shape[-1])
            key_pca_basis, key_eigenvalues = covariance_basis(
                head_key[:: args.sample_stride]
            )
            qk_query_factor, qk_key_factor, singular_values = (
                qk_balanced_factors(
                    head_key[:: args.sample_stride],
                    head_calibration,
                    args.query_shrinkage,
                )
            )
            transforms = {
                "key_pca": (
                    key_pca_basis,
                    key_pca_basis,
                    head_key,
                    None,
                ),
                "qk_balanced": (
                    qk_query_factor,
                    qk_key_factor,
                    head_key,
                    None,
                ),
            }
            centered_singular_values = None
            if args.include_key_centered_qk:
                key_mean = head_key.mean(dim=0)
                centered_key = head_key - key_mean
                (
                    centered_query_factor,
                    centered_key_factor,
                    centered_singular_values,
                ) = qk_balanced_factors(
                    centered_key[:: args.sample_stride],
                    head_calibration,
                    args.query_shrinkage,
                )
                transforms["qk_balanced_keycentered"] = (
                    centered_query_factor,
                    centered_key_factor,
                    centered_key,
                    key_mean,
                )
            head_states = {}
            for method, (
                query_factor,
                key_factor,
                index_key,
                score_offset_key,
            ) in transforms.items():
                coefficients = index_key @ key_factor
                projected_calibration = head_calibration @ query_factor
                allocation = allocate_bits(
                    distortion_table(
                        coefficients,
                        projected_calibration,
                    ),
                    args.total_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                quantized_bands = []
                for group_index in range(GROUP_COUNT):
                    start = group_index * GROUP_SIZE
                    band = coefficients[:, start : start + GROUP_SIZE]
                    quantized_bands.append(
                        {
                            bits: quantize_band(band, bits)
                            for bits in ZERO_BIT_LEVELS
                        }
                    )
                reconstructed = reconstruct(quantized_bands, allocation)
                factor_identity_error = float(
                    (
                        query_factor
                        @ key_factor.transpose(0, 1)
                        - torch.eye(128, device=device)
                    )
                    .abs()
                    .max()
                    .item()
                )
                code_bits = GROUP_SIZE * sum(allocation)
                metadata_bits = GROUP_SIZE * sum(
                    bits > 0 for bits in allocation
                )
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(map(str, allocation)),
                        "code_bits": code_bits,
                        "metadata_bits": metadata_bits,
                        "total_index_bits": code_bits + metadata_bits,
                        "factor_identity_error": factor_identity_error,
                    }
                )
                head_states[method] = {
                    "query_factor": query_factor,
                    "reconstructed": reconstructed,
                    "score_offset_key": score_offset_key,
                }
                if method == "qk_balanced":
                    normalized_boundary_weights = topk_boundary_weights(
                        coefficients,
                        projected_calibration,
                        args.top_fraction,
                        include_global_floor=False,
                    )
                    for boundary_suffix, boundary_weights in (
                        (
                            "boundary_scale",
                            1.0 + normalized_boundary_weights,
                        ),
                        (
                            "boundary_only_scale",
                            normalized_boundary_weights,
                        ),
                    ):
                        boundary_bands = boundary_scale_quantized_bands(
                            coefficients,
                            projected_calibration,
                            boundary_weights,
                        )
                        boundary_allocation = allocate_bits(
                            boundary_distortion_table_from_bands(
                                coefficients,
                                projected_calibration,
                                boundary_bands,
                                boundary_weights,
                            ),
                            args.total_rate_budget,
                            ZERO_BIT_LEVELS,
                            include_scale_metadata=True,
                        )
                        boundary_method = (
                            f"qk_balanced_{boundary_suffix}"
                        )
                        boundary_reconstructed = reconstruct(
                            boundary_bands,
                            boundary_allocation,
                        )
                        boundary_code_bits = GROUP_SIZE * sum(
                            boundary_allocation
                        )
                        boundary_metadata_bits = GROUP_SIZE * sum(
                            bits > 0 for bits in boundary_allocation
                        )
                        allocation_rows.append(
                            {
                                "label": args.label,
                                "layer": layer,
                                "kv_head": kv_head,
                                "method": boundary_method,
                                "allocation": "-".join(
                                    map(str, boundary_allocation)
                                ),
                                "code_bits": boundary_code_bits,
                                "metadata_bits": (
                                    boundary_metadata_bits
                                ),
                                "total_index_bits": (
                                    boundary_code_bits
                                    + boundary_metadata_bits
                                ),
                                "factor_identity_error": (
                                    factor_identity_error
                                ),
                            }
                        )
                        head_states[boundary_method] = {
                            "query_factor": query_factor,
                            "reconstructed": boundary_reconstructed,
                            "score_offset_key": score_offset_key,
                        }
                    for metric_suffix, metric_mode in (
                        ("lsq_scale", "identity"),
                        ("diag_metric_scale", "diagonal"),
                        ("metric_scale", "full"),
                    ):
                        metric_scale_bands = []
                        for group_index in range(GROUP_COUNT):
                            start = group_index * GROUP_SIZE
                            stop = start + GROUP_SIZE
                            band = coefficients[:, start:stop]
                            query_band = projected_calibration[
                                :, start:stop
                            ]
                            metric_scale_bands.append(
                                {
                                    bits: metric_scale_quantize_band(
                                        band,
                                        bits,
                                        query_band,
                                        metric_mode,
                                    )
                                    for bits in ZERO_BIT_LEVELS
                                }
                            )
                        metric_scale_allocation = allocate_bits(
                            distortion_table_from_bands(
                                coefficients,
                                projected_calibration,
                                metric_scale_bands,
                            ),
                            args.total_rate_budget,
                            ZERO_BIT_LEVELS,
                            include_scale_metadata=True,
                        )
                        metric_scale_method = (
                            f"qk_balanced_{metric_suffix}"
                        )
                        metric_scale_reconstructed = reconstruct(
                            metric_scale_bands,
                            metric_scale_allocation,
                        )
                        metric_scale_code_bits = GROUP_SIZE * sum(
                            metric_scale_allocation
                        )
                        metric_scale_metadata_bits = GROUP_SIZE * sum(
                            bits > 0
                            for bits in metric_scale_allocation
                        )
                        allocation_rows.append(
                            {
                                "label": args.label,
                                "layer": layer,
                                "kv_head": kv_head,
                                "method": metric_scale_method,
                                "allocation": "-".join(
                                    map(str, metric_scale_allocation)
                                ),
                                "code_bits": metric_scale_code_bits,
                                "metadata_bits": (
                                    metric_scale_metadata_bits
                                ),
                                "total_index_bits": (
                                    metric_scale_code_bits
                                    + metric_scale_metadata_bits
                                ),
                                "factor_identity_error": (
                                    factor_identity_error
                                ),
                            }
                        )
                        head_states[metric_scale_method] = {
                            "query_factor": query_factor,
                            "reconstructed": metric_scale_reconstructed,
                            "score_offset_key": score_offset_key,
                        }
                        if metric_mode == "full":
                            for affine_suffix, ridge, fit_bias in (
                                ("gain_ridge1", 1.0, False),
                                ("affine_ridge1", 1.0, True),
                                ("affine_ridge0p1", 0.1, True),
                            ):
                                gain, bias = fit_token_score_affine(
                                    coefficients,
                                    metric_scale_reconstructed,
                                    projected_calibration,
                                    ridge,
                                    fit_bias,
                                )
                                affine_method = (
                                    "qk_balanced_metric_scale_"
                                    f"{affine_suffix}"
                                )
                                extra_metadata_bits = (
                                    32 if fit_bias else 16
                                )
                                allocation_rows.append(
                                    {
                                        "label": args.label,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "method": affine_method,
                                        "allocation": "-".join(
                                            map(
                                                str,
                                                metric_scale_allocation,
                                            )
                                        ),
                                        "code_bits": (
                                            metric_scale_code_bits
                                        ),
                                        "metadata_bits": (
                                            metric_scale_metadata_bits
                                            + extra_metadata_bits
                                        ),
                                        "total_index_bits": (
                                            metric_scale_code_bits
                                            + metric_scale_metadata_bits
                                            + extra_metadata_bits
                                        ),
                                        "factor_identity_error": (
                                            factor_identity_error
                                        ),
                                    }
                                )
                                head_states[affine_method] = {
                                    "query_factor": query_factor,
                                    "reconstructed": (
                                        metric_scale_reconstructed
                                    ),
                                    "score_offset_key": score_offset_key,
                                    "score_gain": gain,
                                    "score_bias": bias,
                                }
                            affine_gain, affine_bias = (
                                fit_token_score_affine(
                                    coefficients,
                                    metric_scale_reconstructed,
                                    projected_calibration,
                                    ridge=1.0,
                                    fit_bias=True,
                                )
                            )
                            bias_gain, bias_only = (
                                fit_token_score_affine(
                                    coefficients,
                                    metric_scale_reconstructed,
                                    projected_calibration,
                                    ridge=1.0,
                                    fit_bias=True,
                                    fit_gain=False,
                                )
                            )
                            eb_gain, eb_bias, _ = (
                                fit_empirical_bayes_score_bias(
                                    coefficients,
                                    metric_scale_reconstructed,
                                    projected_calibration,
                                )
                            )
                            metadata_variants = [
                                (
                                    "affine_int8",
                                    *quantize_token_score_metadata(
                                        affine_gain,
                                        affine_bias,
                                        bits=8,
                                        include_gain=True,
                                    ),
                                    16,
                                ),
                                (
                                    "affine_int4",
                                    *quantize_token_score_metadata(
                                        affine_gain,
                                        affine_bias,
                                        bits=4,
                                        include_gain=True,
                                    ),
                                    8,
                                ),
                                (
                                    "bias_fp16",
                                    bias_gain,
                                    bias_only,
                                    16,
                                ),
                                (
                                    "bias_int8",
                                    *quantize_token_score_metadata(
                                        bias_gain,
                                        bias_only,
                                        bits=8,
                                        include_gain=False,
                                    ),
                                    8,
                                ),
                                (
                                    "bias_int4",
                                    *quantize_token_score_metadata(
                                        bias_gain,
                                        bias_only,
                                        bits=4,
                                        include_gain=False,
                                    ),
                                    4,
                                ),
                                (
                                    "bias_eb_fp16",
                                    eb_gain,
                                    eb_bias,
                                    16,
                                ),
                                (
                                    "bias_eb_int8",
                                    *quantize_token_score_metadata(
                                        eb_gain,
                                        eb_bias,
                                        bits=8,
                                        include_gain=False,
                                    ),
                                    8,
                                ),
                                (
                                    "bias_eb_int4",
                                    *quantize_token_score_metadata(
                                        eb_gain,
                                        eb_bias,
                                        bits=4,
                                        include_gain=False,
                                    ),
                                    4,
                                ),
                            ]
                            for (
                                metadata_suffix,
                                metadata_gain,
                                metadata_bias,
                                extra_metadata_bits,
                            ) in metadata_variants:
                                metadata_method = (
                                    "qk_balanced_metric_scale_"
                                    f"{metadata_suffix}"
                                )
                                allocation_rows.append(
                                    {
                                        "label": args.label,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "method": metadata_method,
                                        "allocation": "-".join(
                                            map(
                                                str,
                                                metric_scale_allocation,
                                            )
                                        ),
                                        "code_bits": (
                                            metric_scale_code_bits
                                        ),
                                        "metadata_bits": (
                                            metric_scale_metadata_bits
                                            + extra_metadata_bits
                                        ),
                                        "total_index_bits": (
                                            metric_scale_code_bits
                                            + metric_scale_metadata_bits
                                            + extra_metadata_bits
                                        ),
                                        "factor_identity_error": (
                                            factor_identity_error
                                        ),
                                    }
                                )
                                head_states[metadata_method] = {
                                    "query_factor": query_factor,
                                    "reconstructed": (
                                        metric_scale_reconstructed
                                    ),
                                    "score_offset_key": score_offset_key,
                                    "score_gain": metadata_gain,
                                    "score_bias": metadata_bias,
                                }
                    joint_allocation = joint_qmse_allocation(
                        coefficients,
                        projected_calibration,
                        args.total_rate_budget,
                    )
                    joint_method = "qk_balanced_joint_qmse"
                    joint_reconstructed = reconstruct(
                        quantized_bands,
                        joint_allocation,
                    )
                    joint_code_bits = GROUP_SIZE * sum(
                        joint_allocation
                    )
                    joint_metadata_bits = GROUP_SIZE * sum(
                        bits > 0 for bits in joint_allocation
                    )
                    allocation_rows.append(
                        {
                            "label": args.label,
                            "layer": layer,
                            "kv_head": kv_head,
                            "method": joint_method,
                            "allocation": "-".join(
                                map(str, joint_allocation)
                            ),
                            "code_bits": joint_code_bits,
                            "metadata_bits": joint_metadata_bits,
                            "total_index_bits": (
                                joint_code_bits + joint_metadata_bits
                            ),
                            "factor_identity_error": (
                                factor_identity_error
                            ),
                        }
                    )
                    head_states[joint_method] = {
                        "query_factor": query_factor,
                        "reconstructed": joint_reconstructed,
                        "score_offset_key": score_offset_key,
                    }
                    fisher_allocation = allocate_bits(
                        fisher_distortion_table(
                            coefficients,
                            projected_calibration,
                            calibration_scaling,
                        ),
                        args.total_rate_budget,
                        ZERO_BIT_LEVELS,
                        include_scale_metadata=True,
                    )
                    fisher_method = "qk_balanced_fisher"
                    fisher_reconstructed = reconstruct(
                        quantized_bands,
                        fisher_allocation,
                    )
                    fisher_code_bits = GROUP_SIZE * sum(
                        fisher_allocation
                    )
                    fisher_metadata_bits = GROUP_SIZE * sum(
                        bits > 0 for bits in fisher_allocation
                    )
                    allocation_rows.append(
                        {
                            "label": args.label,
                            "layer": layer,
                            "kv_head": kv_head,
                            "method": fisher_method,
                            "allocation": "-".join(
                                map(str, fisher_allocation)
                            ),
                            "code_bits": fisher_code_bits,
                            "metadata_bits": fisher_metadata_bits,
                            "total_index_bits": (
                                fisher_code_bits + fisher_metadata_bits
                            ),
                            "factor_identity_error": (
                                factor_identity_error
                            ),
                        }
                    )
                    head_states[fisher_method] = {
                        "query_factor": query_factor,
                        "reconstructed": fisher_reconstructed,
                        "score_offset_key": score_offset_key,
                    }
                if method == "qk_balanced":
                    for fixed_method, fixed_allocation in (
                        fixed_qk_allocations.items()
                    ):
                        fixed_reconstructed = reconstruct(
                            quantized_bands,
                            fixed_allocation,
                        )
                        fixed_code_bits = GROUP_SIZE * sum(
                            fixed_allocation
                        )
                        fixed_metadata_bits = GROUP_SIZE * sum(
                            bits > 0 for bits in fixed_allocation
                        )
                        allocation_rows.append(
                            {
                                "label": args.label,
                                "layer": layer,
                                "kv_head": kv_head,
                                "method": fixed_method,
                                "allocation": "-".join(
                                    map(str, fixed_allocation)
                                ),
                                "code_bits": fixed_code_bits,
                                "metadata_bits": fixed_metadata_bits,
                                "total_index_bits": (
                                    fixed_code_bits + fixed_metadata_bits
                                ),
                                "factor_identity_error": (
                                    factor_identity_error
                                ),
                            }
                        )
                        head_states[fixed_method] = {
                            "query_factor": query_factor,
                            "reconstructed": fixed_reconstructed,
                            "score_offset_key": score_offset_key,
                        }
            spectrum_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "key_pca_top16_energy": float(
                        key_eigenvalues[:16].sum()
                        / key_eigenvalues.sum().clamp_min(1.0e-12)
                    ),
                    "qk_top16_score_energy": float(
                        singular_values[:16].square().sum()
                        / singular_values.square().sum().clamp_min(1.0e-12)
                    ),
                    "qk_top48_score_energy": float(
                        singular_values[:48].square().sum()
                        / singular_values.square().sum().clamp_min(1.0e-12)
                    ),
                    "qk_centered_top16_score_energy": (
                        float(
                            centered_singular_values[:16].square().sum()
                            / centered_singular_values.square().sum().clamp_min(
                                1.0e-12
                            )
                        )
                        if centered_singular_values is not None
                        else math.nan
                    ),
                    "qk_centered_top48_score_energy": (
                        float(
                            centered_singular_values[:48].square().sum()
                            / centered_singular_values.square().sum().clamp_min(
                                1.0e-12
                            )
                        )
                        if centered_singular_values is not None
                        else math.nan
                    ),
                }
            )
            prepared.append(
                {
                    "head_key": head_key,
                    "methods": head_states,
                }
            )

        top_count = max(1, math.ceil(args.top_fraction * history_count))
        for heldout_index, record in enumerate(
            evaluation_records,
            start=evaluation_start,
        ):
            query = record["query"].to(device).float()[0, :, 0, :]
            scaling = float(record["scaling"])
            for kv_head, state in enumerate(prepared):
                for group in range(groups):
                    query_head = kv_head * groups + group
                    head_query = query[query_head]
                    exact_scores = state["head_key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(
                        exact_scores, k=top_count
                    ).indices
                    for method, method_state in state["methods"].items():
                        projected_query = query_int8(
                            head_query @ method_state["query_factor"]
                        )
                        approximate_unscaled = (
                            method_state["reconstructed"]
                            @ projected_query
                        )
                        score_offset_key = method_state["score_offset_key"]
                        if score_offset_key is not None:
                            approximate_unscaled = (
                                approximate_unscaled
                                + score_offset_key @ head_query
                            )
                        score_gain = method_state.get("score_gain")
                        if score_gain is not None:
                            approximate_unscaled = (
                                approximate_unscaled * score_gain
                                + method_state["score_bias"]
                            )
                        approximate_scores = approximate_unscaled * scaling
                        for fraction in selected_fractions:
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "heldout_step": heldout_index,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "method": method,
                                    "selected_fraction_target": fraction,
                                    **selection_metrics(
                                        exact_scores,
                                        attention,
                                        approximate_scores,
                                        true_top,
                                        fraction,
                                    ),
                                }
                            )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "layer": layer,
                    "layers": len(by_layer),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows, allocation_rows)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "sample_stride": args.sample_stride,
            "calibration_steps": args.calibration_steps,
            "calibration_source": args.calibration_source,
            "total_rate_budget": args.total_rate_budget,
            "query_shrinkage": args.query_shrinkage,
            "selected_fractions": selected_fractions,
            "top_fraction": args.top_fraction,
            "include_key_centered_qk": args.include_key_centered_qk,
            "qk_fixed_allocations": fixed_qk_allocations,
        },
        "allocation_histograms": {
            method: dict(
                Counter(
                    str(row["allocation"])
                    for row in allocation_rows
                    if row["method"] == method
                ).most_common()
            )
            for method in sorted(
                {str(row["method"]) for row in allocation_rows}
            )
        },
        "spectrum": {
            field: summarize(float(row[field]) for row in spectrum_rows)
            for field in (
                "key_pca_top16_energy",
                "qk_top16_score_energy",
                "qk_top48_score_energy",
                "qk_centered_top16_score_energy",
                "qk_centered_top48_score_energy",
            )
        },
        "methods": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocation_rows)
    write_csv(args.output_dir / "spectrum.csv", spectrum_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
