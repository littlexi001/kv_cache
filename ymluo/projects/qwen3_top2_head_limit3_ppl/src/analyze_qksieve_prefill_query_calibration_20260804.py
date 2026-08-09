#!/usr/bin/env python
"""Test request-local prefill calibration of QKSieve Value-tail moments.

The calibration target is the attention output already produced during dense
prefill.  A single clipped least-squares gain is fit per layer from the final
prefill Queries and is evaluated only on later decode Queries.  This is a
mechanism audit on captured Q/K/V tensors, not a model-level PPL claim.
"""

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

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_tail_partition_calibration_20260803 import (
    clipped_wiener_gain,
    load_output_projection,
    metric_value_basis,
    output_metrics,
)
from analyze_qksieve_value_sketch_residual_20260801 import block_affine_quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--prefill_query_tokens", type=int, default=8)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--conditional_fit_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--value_rank", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument("--value_scale_block", type=int, default=256)
    return parser.parse_args()


def fit_conditional_map(
    coordinates: torch.Tensor,
    residual: torch.Tensor,
    *,
    fit_stride: int,
) -> torch.Tensor:
    """Fit residual = mean + A (coordinate - mean) by ridge regression."""

    if coordinates.ndim != 2 or residual.ndim != 2:
        raise ValueError("coordinates and residual must be matrices")
    if coordinates.shape[0] != residual.shape[0]:
        raise ValueError("coordinates and residual must share token count")
    if fit_stride <= 0:
        raise ValueError("fit_stride must be positive")
    indices = torch.arange(
        0, coordinates.shape[0], fit_stride, device=coordinates.device
    )
    sampled_coordinates = coordinates.index_select(0, indices).float()
    sampled_residual = residual.index_select(0, indices).float()
    centered_coordinates = (
        sampled_coordinates - sampled_coordinates.mean(dim=0, keepdim=True)
    )
    centered_residual = sampled_residual - sampled_residual.mean(
        dim=0, keepdim=True
    )
    count = max(1, int(indices.numel()))
    covariance = centered_coordinates.T @ centered_coordinates / float(count)
    cross_covariance = centered_residual.T @ centered_coordinates / float(count)
    ridge = covariance.diagonal().mean().clamp_min(1.0e-8) * 1.0e-3
    return torch.linalg.solve(
        covariance
        + ridge
        * torch.eye(
            coordinates.shape[-1],
            dtype=torch.float32,
            device=coordinates.device,
        ),
        cross_covariance.T,
    ).T


def clipped_noninferior_gain(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a gain that cannot worsen any calibration query's squared error."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must be aligned query batches")
    least_squares_gain, _ = clipped_wiener_gain(prediction, target)
    flat_prediction = prediction.float().reshape(prediction.shape[0], -1)
    flat_target = target.float().reshape(target.shape[0], -1)
    denominator = flat_prediction.square().sum(dim=-1)
    correlation = (flat_prediction * flat_target).sum(dim=-1)
    safe_upper = torch.where(
        denominator <= epsilon,
        torch.ones_like(denominator),
        torch.where(
            correlation > 0.0,
            (2.0 * correlation / denominator.clamp_min(epsilon)).clamp(0.0, 1.0),
            torch.zeros_like(correlation),
        ),
    )
    gain = torch.minimum(least_squares_gain, safe_upper.amin())
    baseline_error = flat_target.square().sum()
    if float(baseline_error) <= epsilon:
        return gain, torch.zeros_like(gain)
    residual_error = (
        flat_target - gain * flat_prediction
    ).square().sum()
    return gain, 1.0 - residual_error / baseline_error


def evaluate_tail_outputs(
    queries: torch.Tensor,
    causal_lengths: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    reconstructed_value: torch.Tensor,
    coordinates: torch.Tensor,
    linear_map: torch.Tensor,
    query_factor: torch.Tensor,
    *,
    scaling: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Evaluate exact, ValueSketch, residual-mean, and conditional outputs."""

    if queries.ndim != 3:
        raise ValueError("queries must have shape [tokens, query_groups, dim]")
    if causal_lengths.shape != (queries.shape[0],):
        raise ValueError("one causal length is required per query token")
    residual = value.float() - reconstructed_value.float()
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for token_index, query_groups in enumerate(queries):
        length = int(causal_lengths[token_index])
        if not 0 < length <= key.shape[0]:
            raise ValueError("causal length lies outside the captured cache")
        active_k = min(top_k, length)
        active_key = key[:length].float()
        active_value = value[:length].float()
        active_reconstructed_key = reconstructed_key[:length].float()
        active_reconstructed_value = reconstructed_value[:length].float()
        active_coordinates = coordinates[:length].float()
        active_residual = residual[:length]

        exact_scores = query_groups.float() @ active_key.T * float(scaling)
        projected_query = query_groups.float() @ query_factor.float()
        approximate_query = torch.stack(
            [query_int8(item) for item in projected_query]
        ).float()
        proxy_scores = (
            approximate_query @ active_reconstructed_key.T * float(scaling)
        )
        candidate_indices = torch.topk(
            proxy_scores, active_k, dim=-1, sorted=False
        ).indices
        selected_mask = torch.zeros_like(proxy_scores, dtype=torch.bool)
        selected_mask.scatter_(1, candidate_indices, True)

        anchor = torch.maximum(
            exact_scores.amax(dim=-1), proxy_scores.amax(dim=-1)
        )
        exact_weights = torch.exp(exact_scores - anchor[:, None])
        proxy_weights = torch.exp(proxy_scores - anchor[:, None])
        selected_exact_weights = exact_weights.gather(1, candidate_indices)
        selected_values = active_value[candidate_indices]
        selected_numerator = torch.sum(
            selected_exact_weights[..., None] * selected_values, dim=1
        )
        selected_denominator = selected_exact_weights.sum(dim=-1)

        tail_weights = proxy_weights.masked_fill(selected_mask, 0.0)
        tail_mass = tail_weights.sum(dim=-1)
        tail_reconstructed_numerator = tail_weights @ active_reconstructed_value
        denominator = (selected_denominator + tail_mass).clamp_min(1.0e-20)
        valuesketch = (
            selected_numerator + tail_reconstructed_numerator
        ) / denominator[:, None]

        selected_residual = active_residual[candidate_indices]
        selected_coordinate = active_coordinates[candidate_indices]
        tail_count = max(1, length - active_k)
        tail_residual_mean = (
            active_residual.sum(dim=0)[None, :] - selected_residual.sum(dim=1)
        ) / float(tail_count)
        tail_coordinate_mean = (
            active_coordinates.sum(dim=0)[None, :]
            - selected_coordinate.sum(dim=1)
        ) / float(tail_count)
        residual_mean_correction = tail_mass[:, None] * tail_residual_mean
        residual_mean_output = (
            selected_numerator
            + tail_reconstructed_numerator
            + residual_mean_correction
        ) / denominator[:, None]

        weighted_coordinates = tail_weights @ active_coordinates
        centered_weighted_coordinates = (
            weighted_coordinates - tail_mass[:, None] * tail_coordinate_mean
        )
        conditional_direction_numerator = (
            centered_weighted_coordinates @ linear_map.T
        )
        conditional_output = (
            selected_numerator
            + tail_reconstructed_numerator
            + residual_mean_correction
            + conditional_direction_numerator
        ) / denominator[:, None]
        full_output = (exact_weights @ active_value) / exact_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-20)

        outputs["full"].append(full_output)
        outputs["valuesketch"].append(valuesketch)
        outputs["residual_mean"].append(residual_mean_output)
        outputs["conditional"].append(conditional_output)
    return {name: torch.stack(items) for name, items in outputs.items()}


def projected_sequence(
    output: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    if output.ndim != 3:
        raise ValueError("output must have shape [tokens, query_heads, dim]")
    return output.reshape(output.shape[0], -1).float() @ projection.float().T


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.prefill_query_tokens < 2:
        raise ValueError("top_k must be positive and at least two prefill Queries are required")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(
        args.trace, map_location="cpu", weights_only=False, mmap=True
    )
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

    model_root = str(
        args.model_name_or_path
        or payload.get("config", {}).get("model_name_or_path", "")
    )
    topic = str(payload.get("config", {}).get("topic", args.trace.stem))
    declared_history = int(payload.get("config", {}).get("history_tokens", 0))
    rows: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for layer in sorted(records_by_layer):
        if layer not in state_by_layer or layer not in prefill_by_layer:
            continue
        state_record = state_by_layer[layer]
        key_all = state_record["key"].to(device).float()[0]
        value_all = state_record["value"].to(device).float()[0]
        scaling = float(state_record["scaling"])
        records = sorted(records_by_layer[layer], key=lambda item: int(item["step"]))
        decode_queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0, :] for record in records]
        )
        raw_prefill_queries = prefill_by_layer[layer]
        prefill_queries = (
            raw_prefill_queries[0, :, -args.prefill_query_tokens :, :]
            .permute(1, 0, 2)
            .contiguous()
            .to(device)
            .float()
        )
        query_head_count = int(decode_queries.shape[1])
        kv_head_count = int(key_all.shape[0])
        query_groups = query_head_count // kv_head_count
        if query_head_count % kv_head_count:
            raise ValueError("query heads must be divisible by KV heads")
        projection = load_output_projection(model_root, layer, device)
        token_count = int(key_all.shape[1])
        prefix_count = min(
            token_count,
            declared_history - 1 if declared_history > 1 else token_count - 1,
        )
        prefill_count = int(prefill_queries.shape[0])
        if prefix_count < prefill_count:
            raise ValueError("captured prefill tail exceeds the prompt prefix")
        prefill_lengths = torch.arange(
            prefix_count - prefill_count + 1,
            prefix_count + 1,
            dtype=torch.long,
            device=device,
        )
        decode_lengths = torch.full(
            (decode_queries.shape[0],), token_count, dtype=torch.long, device=device
        )
        prefill_outputs: dict[str, torch.Tensor] = {
            name: torch.empty(
                prefill_count,
                query_head_count,
                value_all.shape[-1],
                dtype=torch.float32,
                device=device,
            )
            for name in ("full", "valuesketch", "residual_mean", "conditional")
        }
        decode_outputs: dict[str, torch.Tensor] = {
            name: torch.empty(
                decode_queries.shape[0],
                query_head_count,
                value_all.shape[-1],
                dtype=torch.float32,
                device=device,
            )
            for name in ("full", "valuesketch", "residual_mean", "conditional")
        }

        for kv_head in range(kv_head_count):
            key = key_all[kv_head]
            value = value_all[kv_head]
            head_slice = slice(
                kv_head * query_groups, (kv_head + 1) * query_groups
            )
            head_prefill = prefill_queries[:, head_slice]
            calibration_queries = head_prefill.reshape(
                -1, head_prefill.shape[-1]
            )
            query_factor, key_factor, _ = qk_balanced_factors(
                key[:: args.key_sample_stride],
                calibration_queries,
                args.query_shrinkage,
            )
            key_coordinates = key @ key_factor
            projected_calibration = calibration_queries @ query_factor
            bands = quantized_bands(key_coordinates, projected_calibration)
            distortion, _ = distortion_table(
                key_coordinates, projected_calibration, ZERO_BIT_LEVELS
            )
            allocation = allocate_bits(
                distortion,
                args.key_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            reconstructed_key = reconstruct(bands, allocation)

            head_dimension = int(value.shape[-1])
            group_gram = torch.zeros(
                head_dimension,
                head_dimension,
                dtype=torch.float32,
                device=device,
            )
            for query_head in range(head_slice.start, head_slice.stop):
                start = query_head * head_dimension
                block = projection[:, start : start + head_dimension].float()
                group_gram.add_(block.T @ block)
            value_mean, value_vectors, value_coefficients, _ = metric_value_basis(
                value,
                group_gram,
                sample_stride=args.value_sample_stride,
                maximum_rank=args.value_rank,
            )
            quantized_coefficients = block_affine_quantize(
                value_coefficients[:, : args.value_rank],
                bits=args.value_bits,
                block_size=args.value_scale_block,
            )
            reconstructed_value = (
                value_mean
                + quantized_coefficients @ value_vectors[:, : args.value_rank].T
            )
            coordinates = reconstructed_key[:, :8].float()
            linear_map = fit_conditional_map(
                coordinates,
                value.float() - reconstructed_value.float(),
                fit_stride=args.conditional_fit_stride,
            )
            head_prefill_outputs = evaluate_tail_outputs(
                head_prefill,
                prefill_lengths,
                key,
                value,
                reconstructed_key,
                reconstructed_value,
                coordinates,
                linear_map,
                query_factor,
                scaling=scaling,
                top_k=args.top_k,
            )
            head_decode_outputs = evaluate_tail_outputs(
                decode_queries[:, head_slice],
                decode_lengths,
                key,
                value,
                reconstructed_key,
                reconstructed_value,
                coordinates,
                linear_map,
                query_factor,
                scaling=scaling,
                top_k=args.top_k,
            )
            for name in prefill_outputs:
                prefill_outputs[name][:, head_slice] = head_prefill_outputs[name]
                decode_outputs[name][:, head_slice] = head_decode_outputs[name]

        projected_prefill_full = projected_sequence(prefill_outputs["full"], projection)
        projected_prefill_baseline = projected_sequence(
            prefill_outputs["residual_mean"], projection
        )
        projected_prefill_conditional = projected_sequence(
            prefill_outputs["conditional"], projection
        )
        linear_gain, linear_reduction = clipped_wiener_gain(
            projected_prefill_conditional - projected_prefill_baseline,
            projected_prefill_full - projected_prefill_baseline,
        )
        projected_prefill_valuesketch = projected_sequence(
            prefill_outputs["valuesketch"], projection
        )
        total_gain, total_reduction = clipped_wiener_gain(
            projected_prefill_conditional - projected_prefill_valuesketch,
            projected_prefill_full - projected_prefill_valuesketch,
        )
        safe_total_gain, safe_total_reduction = clipped_noninferior_gain(
            projected_prefill_conditional - projected_prefill_valuesketch,
            projected_prefill_full - projected_prefill_valuesketch,
        )
        decode_outputs["prefill_calibrated_linear"] = (
            decode_outputs["residual_mean"]
            + linear_gain
            * (decode_outputs["conditional"] - decode_outputs["residual_mean"])
        )
        decode_outputs["prefill_calibrated_total"] = (
            decode_outputs["valuesketch"]
            + total_gain
            * (decode_outputs["conditional"] - decode_outputs["valuesketch"])
        )
        decode_outputs["prefill_safe_total"] = (
            decode_outputs["valuesketch"]
            + safe_total_gain
            * (decode_outputs["conditional"] - decode_outputs["valuesketch"])
        )
        gains.append(
            {
                "topic": topic,
                "layer": layer,
                "linear_gain": float(linear_gain),
                "linear_prefill_error_reduction": float(linear_reduction),
                "total_gain": float(total_gain),
                "total_prefill_error_reduction": float(total_reduction),
                "safe_total_gain": float(safe_total_gain),
                "safe_total_prefill_error_reduction": float(
                    safe_total_reduction
                ),
                "prefill_queries": prefill_count,
            }
        )

        projected_full = projected_sequence(decode_outputs["full"], projection)
        for method in (
            "valuesketch",
            "residual_mean",
            "conditional",
            "prefill_calibrated_linear",
            "prefill_calibrated_total",
            "prefill_safe_total",
        ):
            projected = projected_sequence(decode_outputs[method], projection)
            for query_index in range(projected.shape[0]):
                rows.append(
                    {
                        "topic": topic,
                        "layer": layer,
                        "query_index": query_index,
                        "method": method,
                        "history_tokens": token_count,
                        "top_k": min(args.top_k, token_count),
                        "gain": (
                            float(linear_gain)
                            if method == "prefill_calibrated_linear"
                            else float(total_gain)
                            if method == "prefill_calibrated_total"
                            else float(safe_total_gain)
                            if method == "prefill_safe_total"
                            else math.nan
                        ),
                        **output_metrics(
                            projected[query_index], projected_full[query_index]
                        ),
                    }
                )
        del key_all, value_all, projection
        torch.cuda.empty_cache()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary = {
        method: {
            "rows": len(items),
            "projected_relative_l2_mean": fmean(
                float(item["relative_l2"]) for item in items
            ),
            "projected_relative_l2_max": max(
                float(item["relative_l2"]) for item in items
            ),
            "projected_cosine_mean": fmean(
                float(item["cosine"]) for item in items
            ),
        }
        for method, items in sorted(grouped.items())
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_layer_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "gains.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gains[0]))
        writer.writeheader()
        writer.writerows(gains)
    report = {
        "trace": str(args.trace),
        "model_name_or_path": model_root,
        "config": vars(args) | {"output_dir": str(args.output_dir), "trace": str(args.trace)},
        "summary": summary,
        "linear_gain_mean": fmean(float(item["linear_gain"]) for item in gains),
        "linear_gain_min": min(float(item["linear_gain"]) for item in gains),
        "linear_gain_max": max(float(item["linear_gain"]) for item in gains),
        "total_gain_mean": fmean(float(item["total_gain"]) for item in gains),
        "total_gain_min": min(float(item["total_gain"]) for item in gains),
        "total_gain_max": max(float(item["total_gain"]) for item in gains),
        "safe_total_gain_mean": fmean(
            float(item["safe_total_gain"]) for item in gains
        ),
        "safe_total_gain_min": min(
            float(item["safe_total_gain"]) for item in gains
        ),
        "safe_total_gain_max": max(
            float(item["safe_total_gain"]) for item in gains
        ),
        "linear_prefill_error_reduction_mean": fmean(
            float(item["linear_prefill_error_reduction"]) for item in gains
        ),
        "total_prefill_error_reduction_mean": fmean(
            float(item["total_prefill_error_reduction"]) for item in gains
        ),
        "safe_total_prefill_error_reduction_mean": fmean(
            float(item["safe_total_prefill_error_reduction"])
            for item in gains
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "ALL_COMPLETE").touch()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
