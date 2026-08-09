#!/usr/bin/env python
"""Test a compact JL sketch of the Key error left by QKSieve's main code."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import allocate_bits
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import reconstruct
from analyze_qksieve_output_risk_budget_20260803 import (
    affine_calibrate_scores,
    conformal_score_uncertainty,
    key_allocation_distortion,
    key_quantization_candidates,
    qk_calibration_queries,
)


KEY_BIT_LEVELS = (0, 1, 2, 4, 8)
FULL_KV_BITS_PER_TOKEN = 2 * 128 * 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rates", default="15,23")
    parser.add_argument("--sketch_ranks", default="8,16,32")
    parser.add_argument("--sketch_bits", default="1,2")
    parser.add_argument("--sketch_seeds", default="0")
    parser.add_argument("--skip_sketch", action="store_true")
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--prefill_query_tokens", type=int, default=8)
    parser.add_argument("--score_calibration_samples", type=int, default=256)
    parser.add_argument("--risk_probe_counts", default="32,64,128,256")
    parser.add_argument("--crossing_failure_probability", type=float, default=0.01)
    return parser.parse_args()


def hadamard_matrix(dimensions: int, device: torch.device) -> torch.Tensor:
    if dimensions <= 0 or dimensions & (dimensions - 1):
        raise ValueError("Hadamard dimensions must be a positive power of two")
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < dimensions:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(float(dimensions))


def deterministic_jl_matrix(
    dimensions: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0 < rank <= dimensions:
        raise ValueError("JL rank must fit the source dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (dimensions,), generator=generator)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0).to(device)
    columns = torch.randperm(dimensions, generator=generator)[:rank].to(device)
    return signs[:, None] * hadamard_matrix(dimensions, device)[:, columns]


def quantize_sketch_per_token(
    values: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    if bits == 1:
        scale = values.float().abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
        signs = torch.where(values >= 0.0, 1.0, -1.0)
        return signs * scale
    if not 1 < bits <= 8:
        raise ValueError("sketch bits must lie in [1, 8]")
    maximum_code = (1 << (bits - 1)) - 1
    scale = (
        values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        / float(maximum_code)
    )
    codes = torch.round(values.float() / scale).clamp(
        -maximum_code, maximum_code
    )
    return codes * scale


def crossfit_wiener_correction(
    exact_scores: torch.Tensor,
    base_scores: torch.Tensor,
    correction: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a sketch only when both probe halves validate its linear gain."""

    history_count = exact_scores.shape[-1]
    active_samples = min(sample_count, history_count)
    sample_ids = torch.div(
        torch.arange(active_samples, device=exact_scores.device)
        * history_count,
        active_samples,
        rounding_mode="floor",
    ).long()
    target = (
        exact_scores.index_select(1, sample_ids).float()
        - base_scores.index_select(1, sample_ids).float()
    )
    feature = correction.index_select(1, sample_ids).float()

    def fit(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_x = feature[:, mask]
        local_y = target[:, mask]
        x_mean = local_x.mean(dim=-1, keepdim=True)
        y_mean = local_y.mean(dim=-1, keepdim=True)
        centered_x = local_x - x_mean
        beta = (
            (centered_x * (local_y - y_mean)).sum(dim=-1, keepdim=True)
            / centered_x.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        ).clamp(0.0, 1.0)
        intercept = y_mean - beta * x_mean
        return beta, intercept

    even = torch.arange(active_samples, device=exact_scores.device) % 2 == 0
    odd = ~even
    beta_even, intercept_even = fit(even)
    beta_odd, intercept_odd = fit(odd)
    prediction_on_odd = beta_even * feature[:, odd] + intercept_even
    prediction_on_even = beta_odd * feature[:, even] + intercept_odd
    improves_odd = (
        (target[:, odd] - prediction_on_odd).square().mean(dim=-1)
        < target[:, odd].square().mean(dim=-1)
    )
    improves_even = (
        (target[:, even] - prediction_on_even).square().mean(dim=-1)
        < target[:, even].square().mean(dim=-1)
    )
    accepted = improves_even & improves_odd
    full_mask = torch.ones(active_samples, dtype=torch.bool, device=exact_scores.device)
    beta, intercept = fit(full_mask)
    beta = torch.where(accepted[:, None], beta, torch.zeros_like(beta))
    intercept = torch.where(
        accepted[:, None], intercept, torch.zeros_like(intercept)
    )
    corrected = base_scores.float() + beta * correction.float() + intercept
    return corrected, beta.squeeze(-1), accepted


def block_log_upper_quantize(
    values: torch.Tensor,
    bits: int = 4,
    block_size: int = 256,
) -> torch.Tensor:
    """Quantize positive norms upward so the decoded value stays an upper bound."""

    if values.ndim != 1 or torch.any(values < 0.0):
        raise ValueError("risk norms must be a non-negative vector")
    levels = (1 << bits) - 1
    logs = torch.log(values.float().clamp_min(1.0e-20))
    output = torch.empty_like(logs)
    for start in range(0, logs.numel(), block_size):
        stop = min(logs.numel(), start + block_size)
        block = logs[start:stop]
        minimum = block.amin()
        scale = ((block.amax() - minimum) / float(levels)).clamp_min(1.0e-12)
        codes = torch.ceil((block - minimum) / scale).clamp(0, levels)
        output[start:stop] = minimum + codes * scale
    return torch.exp(output)


def sampled_heteroscedastic_sigma(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    raw_scale: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Fit one request-local RMS multiplier for tokenwise residual scales."""

    history_count = exact_scores.shape[-1]
    active_samples = min(sample_count, history_count)
    sample_ids = torch.div(
        torch.arange(active_samples, device=exact_scores.device)
        * history_count,
        active_samples,
        rounding_mode="floor",
    ).long()
    normalized = (
        exact_scores.index_select(1, sample_ids).float()
        - proxy_scores.index_select(1, sample_ids).float()
    ) / raw_scale.index_select(1, sample_ids).float().clamp_min(1.0e-12)
    multiplier = normalized.square().mean(dim=-1).sqrt().clamp_min(1.0e-6)
    return raw_scale.float() * multiplier[:, None]


def empirical_crossing_probability(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    raw_scale: torch.Tensor,
    boundary: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate one-sided crossing risk from request-local exact probes.

    The add-one survival estimate has a non-zero probability floor, so unseen
    tails become increasingly expensive as the history grows instead of being
    silently treated as impossible.
    """

    history_count = exact_scores.shape[-1]
    active_samples = min(sample_count, history_count)
    sample_ids = torch.div(
        torch.arange(active_samples, device=exact_scores.device)
        * history_count,
        active_samples,
        rounding_mode="floor",
    ).long()
    normalized = (
        exact_scores.index_select(1, sample_ids).float()
        - proxy_scores.index_select(1, sample_ids).float()
    ) / raw_scale.index_select(1, sample_ids).float().clamp_min(1.0e-12)
    sorted_normalized = torch.sort(normalized, dim=-1).values.contiguous()
    normalized_gap = (
        boundary - proxy_scores.float()
    ) / raw_scale.float().clamp_min(1.0e-12)
    insertion = torch.searchsorted(
        sorted_normalized,
        normalized_gap.contiguous(),
        right=False,
    )
    probability = (
        active_samples - insertion + 1
    ).float() / float(active_samples + 1)

    centered = normalized - normalized.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1).clamp_min(1.0e-12)
    kurtosis = centered.pow(4).mean(dim=-1) / variance.square()
    q99 = torch.quantile(normalized, 0.99, dim=-1)
    return probability, kurtosis, q99


def bernstein_rescue_counts(
    crossing_probability: torch.Tensor,
    failure_probability: float,
    maximum_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_crossings = crossing_probability.sum(dim=-1)
    log_inverse_failure = math.log(1.0 / failure_probability)
    counts = torch.ceil(
        expected_crossings
        + torch.sqrt(2.0 * expected_crossings * log_inverse_failure)
        + (2.0 / 3.0) * log_inverse_failure
    ).long().clamp(0, maximum_count)
    return counts, expected_crossings


def ragged_rescue_diagnostics(
    target_mask: torch.Tensor,
    priority: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[float, float, float]:
    recalls = []
    target_counts = []
    for row in range(target_mask.shape[0]):
        target_count = int(target_mask[row].sum().item())
        target_counts.append(float(target_count))
        if target_count == 0:
            recalls.append(1.0)
            continue
        count = int(counts[row].item())
        if count == 0:
            recalls.append(0.0)
            continue
        indices = torch.topk(priority[row], count, sorted=False).indices
        recalls.append(
            float(target_mask[row].gather(0, indices).sum().item())
            / float(target_count)
        )
    return (
        fmean(recalls),
        fmean(target_counts),
        sum(recall < 1.0 for recall in recalls) / len(recalls),
    )


def rerank_union(
    exact_scores: torch.Tensor,
    base_indices: torch.Tensor,
    rescue_priority: torch.Tensor,
    rescue_counts: torch.Tensor,
) -> torch.Tensor:
    """Exact-rerank a ragged rescue set while returning a dense top-k tensor."""

    output = []
    active_k = base_indices.shape[-1]
    for row in range(exact_scores.shape[0]):
        rescue_count = int(rescue_counts[row].item())
        if rescue_count:
            rescue_indices = torch.topk(
                rescue_priority[row], rescue_count, sorted=False
            ).indices
            union = torch.cat((base_indices[row], rescue_indices))
        else:
            union = base_indices[row]
        union_exact = exact_scores[row].gather(0, union)
        positions = torch.topk(union_exact, active_k, sorted=False).indices
        output.append(union.gather(0, positions))
    return torch.stack(output)


def ragged_union_mask(
    base_indices: torch.Tensor,
    rescue_priority: torch.Tensor,
    rescue_counts: torch.Tensor,
) -> torch.Tensor:
    """Return the deduplicated base-plus-rescue set for each query row."""

    if base_indices.ndim != 2 or rescue_priority.ndim != 2:
        raise ValueError("base indices and rescue priority must be matrices")
    if base_indices.shape[0] != rescue_priority.shape[0] or (
        rescue_counts.shape != (base_indices.shape[0],)
    ):
        raise ValueError("ragged union inputs must agree on query rows")
    output = torch.zeros_like(rescue_priority, dtype=torch.bool)
    output.scatter_(1, base_indices, True)
    for row in range(rescue_priority.shape[0]):
        rescue_count = int(rescue_counts[row].item())
        if rescue_count:
            rescue_indices = torch.topk(
                rescue_priority[row], rescue_count, sorted=False
            ).indices
            output[row].scatter_(0, rescue_indices, True)
    return output


def selection_metrics_from_mask(
    exact_scores: torch.Tensor,
    selected_mask: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, float]:
    """Measure exact sparse attention for a different budget in every row."""

    if exact_scores.shape != selected_mask.shape or selected_mask.ndim != 2:
        raise ValueError("selected mask must align with exact scores")
    full_probability = torch.softmax(exact_scores.float(), dim=-1)
    selected_mass = (full_probability * selected_mask.float()).sum(dim=-1)
    masked_scores = exact_scores.float().masked_fill(~selected_mask, -torch.inf)
    selected_probability = torch.softmax(masked_scores, dim=-1)
    sparse_output = selected_probability @ values.float()
    full_output = full_probability @ values.float()
    output_relative = torch.linalg.vector_norm(
        sparse_output - full_output, dim=-1
    ) / torch.linalg.vector_norm(full_output, dim=-1).clamp_min(1.0e-12)

    recalls: list[torch.Tensor] = []
    selected_counts = selected_mask.sum(dim=-1)
    for row in range(exact_scores.shape[0]):
        active_k = int(selected_counts[row].item())
        exact_indices = torch.topk(
            exact_scores[row], active_k, sorted=False
        ).indices
        recalls.append(selected_mask[row].gather(0, exact_indices).float().mean())
    recall = torch.stack(recalls)
    return {
        "topk_recall_mean": float(recall.mean().item()),
        "topk_recall_minimum": float(recall.min().item()),
        "attention_mass_mean": float(selected_mass.mean().item()),
        "attention_mass_minimum": float(selected_mass.min().item()),
        "output_relative_l2_mean": float(output_relative.mean().item()),
        "output_relative_l2_maximum": float(output_relative.max().item()),
        "score_rmse_mean": 0.0,
        "selected_tokens_mean": float(selected_counts.float().mean().item()),
        "selected_tokens_maximum": int(selected_counts.max().item()),
        "selected_ratio_mean": float(
            selected_counts.float().mean().item() / exact_scores.shape[-1]
        ),
    }


def selection_metrics_from_indices(
    exact_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, float]:
    active_k = selected_indices.shape[-1]
    exact_indices = torch.topk(exact_scores, active_k, dim=-1, sorted=False).indices
    exact_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    exact_mask.scatter_(1, exact_indices, True)
    recall = exact_mask.gather(1, selected_indices).float().mean(dim=-1)
    full_probability = torch.softmax(exact_scores.float(), dim=-1)
    selected_mass = full_probability.gather(1, selected_indices).sum(dim=-1)
    selected_exact_scores = exact_scores.gather(1, selected_indices).float()
    selected_probability = torch.softmax(selected_exact_scores, dim=-1)
    selected_values = values[selected_indices]
    sparse_output = torch.einsum(
        "rk,rkd->rd", selected_probability, selected_values.float()
    )
    full_output = full_probability @ values.float()
    output_relative = torch.linalg.vector_norm(
        sparse_output - full_output, dim=-1
    ) / torch.linalg.vector_norm(full_output, dim=-1).clamp_min(1.0e-12)
    return {
        "topk_recall_mean": float(recall.mean().item()),
        "topk_recall_minimum": float(recall.min().item()),
        "attention_mass_mean": float(selected_mass.mean().item()),
        "attention_mass_minimum": float(selected_mass.min().item()),
        "output_relative_l2_mean": float(output_relative.mean().item()),
        "output_relative_l2_maximum": float(output_relative.max().item()),
        "score_rmse_mean": 0.0,
        "selected_tokens_mean": float(active_k),
        "selected_tokens_maximum": int(active_k),
        "selected_ratio_mean": float(active_k / exact_scores.shape[-1]),
    }


def selection_metrics(
    exact_scores: torch.Tensor,
    priority_scores: torch.Tensor,
    values: torch.Tensor,
    top_k: int,
) -> dict[str, float]:
    active_k = min(top_k, exact_scores.shape[-1])
    exact_indices = torch.topk(exact_scores, active_k, dim=-1, sorted=False).indices
    selected_indices = torch.topk(
        priority_scores, active_k, dim=-1, sorted=False
    ).indices
    exact_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    exact_mask.scatter_(1, exact_indices, True)
    recall = exact_mask.gather(1, selected_indices).float().mean(dim=-1)

    full_probability = torch.softmax(exact_scores.float(), dim=-1)
    selected_mass = full_probability.gather(1, selected_indices).sum(dim=-1)
    selected_exact_scores = exact_scores.gather(1, selected_indices).float()
    selected_probability = torch.softmax(selected_exact_scores, dim=-1)
    selected_values = values[selected_indices]
    sparse_output = torch.einsum(
        "rk,rkd->rd", selected_probability, selected_values.float()
    )
    full_output = full_probability @ values.float()
    output_relative = torch.linalg.vector_norm(
        sparse_output - full_output, dim=-1
    ) / torch.linalg.vector_norm(full_output, dim=-1).clamp_min(1.0e-12)
    score_rmse = torch.mean(
        (priority_scores.float() - exact_scores.float()).square(), dim=-1
    ).sqrt()
    return {
        "topk_recall_mean": float(recall.mean().item()),
        "topk_recall_minimum": float(recall.min().item()),
        "attention_mass_mean": float(selected_mass.mean().item()),
        "attention_mass_minimum": float(selected_mass.min().item()),
        "output_relative_l2_mean": float(output_relative.mean().item()),
        "output_relative_l2_maximum": float(output_relative.max().item()),
        "score_rmse_mean": float(score_rmse.mean().item()),
        "selected_tokens_mean": float(active_k),
        "selected_tokens_maximum": int(active_k),
        "selected_ratio_mean": float(active_k / exact_scores.shape[-1]),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output = []
    for method, items in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "conditions": len(items),
                "key_bits_per_token_mean": fmean(
                    float(item["key_bits_per_token"]) for item in items
                ),
                "index_ratio_of_full_kv_mean": fmean(
                    float(item["index_ratio_of_full_kv"]) for item in items
                ),
                "topk_recall_mean": fmean(
                    float(item["topk_recall_mean"]) for item in items
                ),
                "topk_recall_worst": min(
                    float(item["topk_recall_minimum"]) for item in items
                ),
                "attention_mass_mean": fmean(
                    float(item["attention_mass_mean"]) for item in items
                ),
                "attention_mass_worst": min(
                    float(item["attention_mass_minimum"]) for item in items
                ),
                "output_relative_l2_mean": fmean(
                    float(item["output_relative_l2_mean"]) for item in items
                ),
                "output_relative_l2_worst": max(
                    float(item["output_relative_l2_maximum"]) for item in items
                ),
                "score_rmse_mean": fmean(
                    float(item["score_rmse_mean"]) for item in items
                ),
                "wiener_beta_mean": fmean(
                    float(item.get("wiener_beta_mean", 0.0)) for item in items
                ),
                "wiener_acceptance_mean": fmean(
                    float(item.get("wiener_acceptance", 0.0)) for item in items
                ),
                "risk_missed_topk_recall_mean": fmean(
                    float(item.get("risk_missed_topk_recall", 0.0))
                    for item in items
                ),
                "rescue_tokens_mean": fmean(
                    float(item.get("rescue_tokens_mean", 0.0)) for item in items
                ),
                "predicted_crossings_mean": fmean(
                    float(item.get("predicted_crossings_mean", 0.0))
                    for item in items
                ),
                "actual_missed_topk_mean": fmean(
                    float(item.get("actual_missed_topk_mean", 0.0))
                    for item in items
                ),
                "rescue_failure_rate_mean": fmean(
                    float(item.get("rescue_failure_rate", 0.0))
                    for item in items
                ),
                "selected_tokens_mean": fmean(
                    float(item.get("selected_tokens_mean", 0.0))
                    for item in items
                ),
                "selected_tokens_maximum": max(
                    int(item.get("selected_tokens_maximum", 0))
                    for item in items
                ),
                "selected_ratio_mean": fmean(
                    float(item.get("selected_ratio_mean", 0.0))
                    for item in items
                ),
                "normalized_residual_kurtosis_mean": fmean(
                    float(item.get("normalized_residual_kurtosis_mean", 0.0))
                    for item in items
                ),
                "normalized_residual_q99_mean": fmean(
                    float(item.get("normalized_residual_q99_mean", 0.0))
                    for item in items
                ),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rates = tuple(sorted({int(item) for item in args.rates.split(",") if item}))
    sketch_ranks = tuple(
        sorted({int(item) for item in args.sketch_ranks.split(",") if item})
    )
    sketch_bits = tuple(
        sorted({int(item) for item in args.sketch_bits.split(",") if item})
    )
    sketch_seeds = tuple(
        sorted({int(item) for item in args.sketch_seeds.split(",") if item})
    )
    risk_probe_counts = tuple(
        sorted({int(item) for item in args.risk_probe_counts.split(",") if item})
    )
    if not rates or rates[0] <= 0 or args.top_k <= 0:
        raise ValueError("rates and top_k must be positive")
    if not 0.0 < args.crossing_failure_probability < 1.0:
        raise ValueError("crossing failure probability must lie in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace, map_location="cpu", weights_only=False, mmap=True)
    raw_prefill = payload.get("prefill_queries") or payload.get(
        "prefill_query_tail", {}
    )
    prefill_by_layer = {int(layer): value for layer, value in raw_prefill.items()}
    records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    state_by_layer: dict[int, dict[str, Any]] = {}
    for record in payload["records"]:
        layer = int(record["layer"])
        records_by_layer[layer].append(record)
        if record.get("key") is not None and record.get("value") is not None:
            state_by_layer.setdefault(layer, record)

    topic = str(payload.get("config", {}).get("topic", args.trace.stem))
    rows: list[dict[str, Any]] = []
    for layer in sorted(records_by_layer):
        if layer not in state_by_layer or layer not in prefill_by_layer:
            continue
        records = sorted(records_by_layer[layer], key=lambda item: int(item["step"]))
        state_record = state_by_layer[layer]
        key_all = state_record["key"].to(device).float()[0]
        value_all = state_record["value"].to(device).float()[0]
        scaling = float(state_record["scaling"])
        decode_query = torch.stack(
            [record["query"].to(device).float()[0, :, 0, :] for record in records]
        )
        raw_query = prefill_by_layer[layer]
        prefill_query = (
            raw_query[0, :, -args.prefill_query_tokens :, :]
            .permute(1, 0, 2)
            .contiguous()
            .to(device)
            .float()
        )
        query_groups = decode_query.shape[1] // key_all.shape[0]
        for kv_head in range(key_all.shape[0]):
            key = key_all[kv_head]
            value = value_all[kv_head]
            head_slice = slice(kv_head * query_groups, (kv_head + 1) * query_groups)
            queries = decode_query[:, head_slice]
            flat_queries = queries.reshape(-1, queries.shape[-1])
            head_prefill = prefill_query[:, head_slice]
            calibration_queries = qk_calibration_queries(
                queries, head_prefill, "prefill"
            )
            query_factor, key_factor, singular_values = qk_balanced_factors(
                key[:: args.key_sample_stride],
                calibration_queries,
                args.query_shrinkage,
            )
            key_coordinates = key @ key_factor
            projected_calibration = calibration_queries @ query_factor
            projected_queries = flat_queries @ query_factor
            approximate_queries = torch.stack(
                [query_int8(query) for query in projected_queries]
            )
            exact_scores = flat_queries @ key.T * scaling
            bands = key_quantization_candidates(
                key_coordinates,
                projected_calibration,
                "plain",
                KEY_BIT_LEVELS,
            )
            distortion = key_allocation_distortion(
                key_coordinates,
                projected_calibration,
                bands,
                "oas_qk_mse",
                singular_values,
            )
            reconstructions: dict[int, torch.Tensor] = {}
            allocations: dict[int, list[int]] = {}
            for rate in rates:
                allocation = allocate_bits(
                    distortion,
                    rate,
                    KEY_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                allocations[rate] = allocation
                reconstructions[rate] = reconstruct(bands, allocation)

            base_rate = rates[0]
            base_key = reconstructions[base_rate]
            base_proxy = approximate_queries.float() @ base_key.T * scaling
            base_proxy, _, _ = affine_calibrate_scores(
                exact_scores,
                base_proxy,
                args.score_calibration_samples,
            )
            base_bits = 16 * sum(allocations[base_rate])
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": f"rate{base_rate}",
                    "key_bits_per_token": base_bits,
                    "index_ratio_of_full_kv": base_bits / FULL_KV_BITS_PER_TOKEN,
                    "wiener_beta_mean": 0.0,
                    "wiener_acceptance": 0.0,
                    "risk_missed_topk_recall": 0.0,
                    "rescue_tokens_mean": 0.0,
                    **selection_metrics(
                        exact_scores, base_proxy, value, args.top_k
                    ),
                }
            )
            residual_norm_upper = block_log_upper_quantize(
                torch.linalg.vector_norm(
                    key_coordinates.float() - base_key.float(), dim=-1
                )
            )
            query_norm = torch.linalg.vector_norm(
                projected_queries.float(), dim=-1
            )
            raw_score_uncertainty = (
                scaling * query_norm[:, None] * residual_norm_upper[None, :]
            )
            score_uncertainty, _ = conformal_score_uncertainty(
                exact_scores,
                base_proxy,
                raw_score_uncertainty,
                args.score_calibration_samples,
                miscoverage=0.01,
            )
            score_sigma = sampled_heteroscedastic_sigma(
                exact_scores,
                base_proxy,
                raw_score_uncertainty,
                args.score_calibration_samples,
            )
            active_k = min(args.top_k, exact_scores.shape[-1])
            base_indices = torch.topk(
                base_proxy, active_k, dim=-1, sorted=False
            ).indices
            base_mask = torch.zeros_like(base_proxy, dtype=torch.bool)
            base_mask.scatter_(1, base_indices, True)
            optimistic_priority = (base_proxy + score_uncertainty).masked_fill(
                base_mask, -torch.inf
            )
            boundary = base_proxy.gather(1, base_indices).amin(
                dim=-1, keepdim=True
            )
            crossing_z = (base_proxy - boundary) / score_sigma.clamp_min(1.0e-8)
            crossing_probability = (
                0.5 * torch.erfc(-crossing_z / math.sqrt(2.0))
            ).masked_fill(base_mask, 0.0)
            crossing_priority = crossing_probability.masked_fill(
                base_mask, -torch.inf
            )
            exact_top_mask = torch.zeros_like(base_mask)
            exact_top_mask.scatter_(
                1,
                torch.topk(
                    exact_scores, active_k, dim=-1, sorted=False
                ).indices,
                True,
            )
            missed_exact = exact_top_mask & ~base_mask
            missed_count = missed_exact.sum(dim=-1).clamp_min(1)
            for risk_count in risk_probe_counts:
                active_risk = min(
                    risk_count, exact_scores.shape[-1] - active_k
                )
                risk_indices = torch.topk(
                    optimistic_priority,
                    active_risk,
                    dim=-1,
                    sorted=False,
                ).indices
                risk_recall = (
                    missed_exact.gather(1, risk_indices).sum(dim=-1)
                    / missed_count
                )
                union_indices = torch.cat((base_indices, risk_indices), dim=-1)
                union_exact = exact_scores.gather(1, union_indices)
                rerank_positions = torch.topk(
                    union_exact, active_k, dim=-1, sorted=False
                ).indices
                reranked_indices = union_indices.gather(1, rerank_positions)
                risk_bits = 4
                rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": f"rate{base_rate}_ucb_rescue{risk_count}",
                        "key_bits_per_token": base_bits + risk_bits,
                        "index_ratio_of_full_kv": (
                            (base_bits + risk_bits) / FULL_KV_BITS_PER_TOKEN
                        ),
                        "wiener_beta_mean": 0.0,
                        "wiener_acceptance": 0.0,
                        "risk_missed_topk_recall": float(
                            risk_recall.mean().item()
                        ),
                        "rescue_tokens_mean": float(active_risk),
                        **selection_metrics_from_indices(
                            exact_scores, reranked_indices, value
                        ),
                    }
                )
                crossing_indices = torch.topk(
                    crossing_priority,
                    active_risk,
                    dim=-1,
                    sorted=False,
                ).indices
                crossing_recall = (
                    missed_exact.gather(1, crossing_indices).sum(dim=-1)
                    / missed_count
                )
                crossing_union = torch.cat((base_indices, crossing_indices), dim=-1)
                crossing_exact = exact_scores.gather(1, crossing_union)
                crossing_positions = torch.topk(
                    crossing_exact, active_k, dim=-1, sorted=False
                ).indices
                crossing_reranked = crossing_union.gather(
                    1, crossing_positions
                )
                rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": f"rate{base_rate}_crossing_rescue{risk_count}",
                        "key_bits_per_token": base_bits + risk_bits,
                        "index_ratio_of_full_kv": (
                            (base_bits + risk_bits) / FULL_KV_BITS_PER_TOKEN
                        ),
                        "wiener_beta_mean": 0.0,
                        "wiener_acceptance": 0.0,
                        "risk_missed_topk_recall": float(
                            crossing_recall.mean().item()
                        ),
                        "rescue_tokens_mean": float(active_risk),
                        **selection_metrics_from_indices(
                            exact_scores, crossing_reranked, value
                        ),
                    }
                )
            adaptive_counts, crossing_mean = bernstein_rescue_counts(
                crossing_probability,
                args.crossing_failure_probability,
                exact_scores.shape[-1] - active_k,
            )
            adaptive_reranked = rerank_union(
                exact_scores,
                base_indices,
                crossing_priority,
                adaptive_counts,
            )
            (
                gaussian_rescue_recall,
                actual_missed_mean,
                gaussian_rescue_failure,
            ) = ragged_rescue_diagnostics(
                missed_exact,
                crossing_priority,
                adaptive_counts,
            )
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": f"rate{base_rate}_crossing_bernstein",
                    "key_bits_per_token": base_bits + 4,
                    "index_ratio_of_full_kv": (
                        (base_bits + 4) / FULL_KV_BITS_PER_TOKEN
                    ),
                    "wiener_beta_mean": 0.0,
                    "wiener_acceptance": 0.0,
                    "risk_missed_topk_recall": gaussian_rescue_recall,
                    "rescue_tokens_mean": float(
                        adaptive_counts.float().mean().item()
                    ),
                    "predicted_crossings_mean": float(
                        crossing_mean.mean().item()
                    ),
                    "actual_missed_topk_mean": actual_missed_mean,
                    "rescue_failure_rate": gaussian_rescue_failure,
                    **selection_metrics_from_indices(
                        exact_scores, adaptive_reranked, value
                    ),
                }
            )
            (
                empirical_probability,
                normalized_kurtosis,
                normalized_q99,
            ) = empirical_crossing_probability(
                exact_scores,
                base_proxy,
                raw_score_uncertainty,
                boundary,
                args.score_calibration_samples,
            )
            empirical_probability.masked_fill_(base_mask, 0.0)
            empirical_priority = empirical_probability.masked_fill(
                base_mask, -torch.inf
            )
            empirical_counts, empirical_mean = bernstein_rescue_counts(
                empirical_probability,
                args.crossing_failure_probability,
                exact_scores.shape[-1] - active_k,
            )
            empirical_reranked = rerank_union(
                exact_scores,
                base_indices,
                empirical_priority,
                empirical_counts,
            )
            (
                empirical_rescue_recall,
                actual_missed_mean,
                empirical_rescue_failure,
            ) = ragged_rescue_diagnostics(
                missed_exact,
                empirical_priority,
                empirical_counts,
            )
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": f"rate{base_rate}_empirical_crossing_bernstein",
                    "key_bits_per_token": base_bits + 4,
                    "index_ratio_of_full_kv": (
                        (base_bits + 4) / FULL_KV_BITS_PER_TOKEN
                    ),
                    "wiener_beta_mean": 0.0,
                    "wiener_acceptance": 0.0,
                    "risk_missed_topk_recall": empirical_rescue_recall,
                    "rescue_tokens_mean": float(
                        empirical_counts.float().mean().item()
                    ),
                    "predicted_crossings_mean": float(
                        empirical_mean.mean().item()
                    ),
                    "actual_missed_topk_mean": actual_missed_mean,
                    "rescue_failure_rate": empirical_rescue_failure,
                    "normalized_residual_kurtosis_mean": float(
                        normalized_kurtosis.mean().item()
                    ),
                    "normalized_residual_q99_mean": float(
                        normalized_q99.mean().item()
                    ),
                    **selection_metrics_from_indices(
                        exact_scores, empirical_reranked, value
                    ),
                }
            )
            empirical_union_mask = ragged_union_mask(
                base_indices,
                empirical_priority,
                empirical_counts,
            )
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": (
                        f"rate{base_rate}_empirical_crossing_keep_union"
                    ),
                    "key_bits_per_token": base_bits + 4,
                    "index_ratio_of_full_kv": (
                        (base_bits + 4) / FULL_KV_BITS_PER_TOKEN
                    ),
                    "wiener_beta_mean": 0.0,
                    "wiener_acceptance": 0.0,
                    "risk_missed_topk_recall": empirical_rescue_recall,
                    "rescue_tokens_mean": float(
                        empirical_counts.float().mean().item()
                    ),
                    "predicted_crossings_mean": float(
                        empirical_mean.mean().item()
                    ),
                    "actual_missed_topk_mean": actual_missed_mean,
                    "rescue_failure_rate": empirical_rescue_failure,
                    "normalized_residual_kurtosis_mean": float(
                        normalized_kurtosis.mean().item()
                    ),
                    "normalized_residual_q99_mean": float(
                        normalized_q99.mean().item()
                    ),
                    **selection_metrics_from_mask(
                        exact_scores, empirical_union_mask, value
                    ),
                }
            )
            for rate in rates[1:]:
                proxy = (
                    approximate_queries.float() @ reconstructions[rate].T * scaling
                )
                proxy, _, _ = affine_calibrate_scores(
                    exact_scores,
                    proxy,
                    args.score_calibration_samples,
                )
                payload_bits = 16 * sum(allocations[rate])
                rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": f"rate{rate}",
                        "key_bits_per_token": payload_bits,
                        "index_ratio_of_full_kv": (
                            payload_bits / FULL_KV_BITS_PER_TOKEN
                        ),
                        "wiener_beta_mean": 0.0,
                        "wiener_acceptance": 0.0,
                        "risk_missed_topk_recall": 0.0,
                        "rescue_tokens_mean": 0.0,
                        **selection_metrics(
                            exact_scores, proxy, value, args.top_k
                        ),
                    }
                )

            residual = key_coordinates - base_key
            dimensions = residual.shape[-1]
            active_sketch_ranks = () if args.skip_sketch else sketch_ranks
            for rank in active_sketch_ranks:
                for bits in sketch_bits:
                    for sketch_seed in sketch_seeds:
                        projection = deterministic_jl_matrix(
                            dimensions,
                            rank,
                            1000003 * layer + 1009 * kv_head + sketch_seed,
                            device,
                        )
                        residual_sketch = residual @ projection
                        quantized_sketch = quantize_sketch_per_token(
                            residual_sketch, bits
                        )
                        query_sketch = approximate_queries.float() @ projection
                        correction = (
                            float(dimensions) / float(rank)
                        ) * (query_sketch @ quantized_sketch.T) * scaling
                        corrected_proxy, wiener_beta, correction_accepted = (
                            crossfit_wiener_correction(
                                exact_scores,
                                base_proxy,
                                correction,
                                args.score_calibration_samples,
                            )
                        )
                        sketch_payload_bits = rank * bits + 16
                        total_bits = base_bits + sketch_payload_bits
                        rows.append(
                            {
                                "topic": topic,
                                "layer": layer,
                                "kv_head": kv_head,
                                "method": (
                                    f"rate{base_rate}_jl{rank}_i{bits}_s{sketch_seed}"
                                ),
                                "key_bits_per_token": total_bits,
                                "index_ratio_of_full_kv": (
                                    total_bits / FULL_KV_BITS_PER_TOKEN
                                ),
                                "wiener_beta_mean": float(
                                    wiener_beta.mean().item()
                                ),
                                "wiener_acceptance": float(
                                    correction_accepted.float().mean().item()
                                ),
                                "risk_missed_topk_recall": 0.0,
                                "rescue_tokens_mean": 0.0,
                                **selection_metrics(
                                    exact_scores,
                                    corrected_proxy,
                                    value,
                                    args.top_k,
                                ),
                            }
                        )
        del key_all, value_all, decode_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(
            dict.fromkeys(key for row in rows for key in row)
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "qksieve_complementary_residual_sketch_v1",
        "parameters": {**vars(args), "trace": str(args.trace), "output_dir": str(args.output_dir)},
        "summary": summary,
        "claim_boundary": (
            "Offline exact-QKV selector diagnostic. The JL correction is a "
            "high-probability estimator, not a deterministic worst-case certificate."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
