#!/usr/bin/env python
"""Decompose and calibrate QKSieve's omitted softmax/Value tail.

This is a mechanism audit on captured real Q/K/V tensors.  It separates
candidate selection, tail partition estimation, and low-rank Value error.
No result from this script is a model-level PPL or speed claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

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
from analyze_qksieve_value_sketch_residual_20260801 import (
    block_affine_quantize,
    value_basis,
)


def parse_csv(specification: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(sorted({cast(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_name_or_path",
        default="",
        help=(
            "Optional model root used to project concatenated head errors "
            "through each layer's o_proj. The trace config is used by default."
        ),
    )
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--sample_counts", default="64,128,256,512")
    parser.add_argument("--block_sizes", default="128,256,512,1024")
    parser.add_argument("--conditional_dims", default="8,16,32")
    parser.add_argument(
        "--conditional_fit_stride",
        type=int,
        default=1,
        help=(
            "Fit the closed-form Key-to-Value residual map on every Nth "
            "prefill token. Use 32 to match the low-cost runtime prototype."
        ),
    )
    parser.add_argument(
        "--tail_sampling",
        choices=("random", "systematic"),
        default="random",
        help=(
            "How exact control-variate probes are drawn from the unselected "
            "tail. Random sampling is unbiased; systematic sampling is a "
            "coalesced-memory deployment proxy."
        ),
    )
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--value_rank", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument("--value_scale_block", type=int, default=256)
    parser.add_argument(
        "--block_moment_bits",
        type=int,
        choices=(4, 8, 16),
        default=16,
        help=(
            "Simulated storage precision for block residual means, block "
            "Key-coordinate means, and the conditional map."
        ),
    )
    parser.add_argument(
        "--risk_delta",
        type=float,
        default=0.01,
        help=(
            "Failure probability for the vector-Bernstein residual radius. "
            "This is a probabilistic diagnostic, not a deterministic bound."
        ),
    )
    parser.add_argument(
        "--value_metric",
        choices=("raw", "wo_group"),
        default="raw",
        help=(
            "Use ordinary Value PCA or PCA under the summed W_o Gram metric "
            "of all query heads sharing the KV head."
        ),
    )
    parser.add_argument(
        "--max_records_per_trace",
        type=int,
        default=0,
        help="Process only this many leading records per trace; 0 keeps all.",
    )
    return parser.parse_args()


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def quantize_block_moment(
    tensor: torch.Tensor,
    bits: int,
    reduce_dims: tuple[int, ...],
) -> torch.Tensor:
    if bits == 16:
        return tensor.to(torch.float16).float()
    maximum_code = float((1 << (bits - 1)) - 1)
    scale = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1.0e-12)
    return (
        torch.round(tensor / scale * maximum_code)
        .clamp(-maximum_code, maximum_code)
        * (scale / maximum_code)
    )


def vector_bernstein_radius(
    variance: torch.Tensor,
    maximum_weighted_norm: torch.Tensor,
    *,
    failure_probability: float,
    dimension: int,
) -> torch.Tensor:
    """Return a coordinate-union vector Bernstein confidence radius."""

    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure probability must lie in (0, 1)")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    log_factor = math.log(2.0 * dimension / failure_probability)
    return torch.sqrt(2.0 * variance.clamp_min(0.0) * log_factor) + (
        (2.0 / 3.0) * maximum_weighted_norm.clamp_min(0.0) * log_factor
    )


def clipped_wiener_gain(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one held-out scalar gain and report relative squared-error reduction."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=prediction.device)
        return zero, zero
    prediction_float = prediction.float()
    target_float = target.float()
    denominator = prediction_float.square().sum()
    if float(denominator) <= epsilon:
        zero = torch.zeros((), dtype=torch.float32, device=prediction.device)
        return zero, zero
    gain = (
        (prediction_float * target_float).sum() / denominator
    ).clamp(0.0, 1.0)
    baseline_error = target_float.square().sum()
    if float(baseline_error) <= epsilon:
        return gain, torch.zeros_like(gain)
    residual_error = (target_float - gain * prediction_float).square().sum()
    relative_reduction = 1.0 - residual_error / baseline_error
    return gain, relative_reduction


def append_query_crossfit_conditional_rows(
    rows: list[dict[str, Any]],
) -> int:
    """Calibrate conditional corrections on other prefill queries only.

    The conditional direction is measured relative to the block residual-mean
    baseline.  For every held-out query, one scalar gain is fit from the other
    queries that share its request, layer, head, and selection configuration.
    """

    baseline_prefix = "block_residual_mean_"
    conditional_prefix = "block_conditional_residual_"
    originals = list(rows)

    def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["model_name_or_path"],
            row["trace"],
            row["history_tokens"],
            row["token_count"],
            row["layer"],
            row["kv_head"],
            row["query_head"],
            row["query_head_count"],
            row["candidate_mode"],
            row["top_k"],
            row["sample_count"],
            row["block_size"],
        )

    def record_key(row: dict[str, Any]) -> tuple[int, int]:
        return int(row["record_index"]), int(row["step"])

    baselines: dict[
        tuple[tuple[Any, ...], tuple[int, int], str], dict[str, Any]
    ] = {}
    conditionals: dict[
        tuple[tuple[Any, ...], str], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)

    for row in originals:
        method = str(row["method"])
        if method.startswith(baseline_prefix):
            score_mode = method[len(baseline_prefix) :]
            baselines[(group_key(row), record_key(row), score_mode)] = row

    for row in originals:
        method = str(row["method"])
        if not method.startswith(conditional_prefix):
            continue
        suffix = method[len(conditional_prefix) :]
        score_mode, separator, dimension = suffix.rpartition("_d")
        if not separator or not dimension.isdigit():
            continue
        baseline = baselines.get((group_key(row), record_key(row), score_mode))
        if baseline is None:
            continue
        conditionals[(group_key(row), suffix)].append((baseline, row))

    appended = 0
    for (_, suffix), pairs in sorted(conditionals.items(), key=lambda item: str(item[0])):
        pairs.sort(key=lambda pair: record_key(pair[0]))
        if len(pairs) < 2:
            continue
        directions = torch.stack(
            [
                conditional["_output_tensor"] - baseline["_output_tensor"]
                for baseline, conditional in pairs
            ]
        )
        targets = torch.stack(
            [
                baseline["_full_output_tensor"] - baseline["_output_tensor"]
                for baseline, _ in pairs
            ]
        )
        output_method = f"block_conditional_query_crossfit_residual_{suffix}"
        for index, (baseline, _) in enumerate(pairs):
            train_mask = torch.ones(
                len(pairs), dtype=torch.bool, device=directions.device
            )
            train_mask[index] = False
            gain, reduction = clipped_wiener_gain(
                directions[train_mask], targets[train_mask]
            )
            output = baseline["_output_tensor"] + gain * directions[index]
            full_output = baseline["_full_output_tensor"]
            row = dict(baseline)
            row.update(
                {
                    "method": output_method,
                    "residual_risk_absolute": math.nan,
                    "residual_risk_relative": math.nan,
                    "residual_risk_range_absolute": math.nan,
                    "residual_risk_bernstein_absolute": math.nan,
                    "residual_risk_bernstein_relative": math.nan,
                    "tail_correction_l2": float(
                        torch.linalg.vector_norm(gain * directions[index])
                    ),
                    "conditional_gain": float(gain),
                    "conditional_holdout_error_reduction": float(reduction),
                    **output_metrics(output, full_output),
                    "_output_tensor": output.detach().float().clone(),
                    "_full_output_tensor": full_output.detach().float().clone(),
                }
            )
            rows.append(row)
            appended += 1
    return appended


def systematic_indices(
    token_count: int,
    sample_count: int,
    phase_seed: int,
    device: torch.device,
) -> torch.Tensor:
    sample_count = min(token_count, max(1, sample_count))
    positions = torch.arange(sample_count, device=device, dtype=torch.long)
    centered = ((2 * positions + 1) * token_count) // (2 * sample_count)
    segment = max(1, token_count // sample_count)
    return (centered + phase_seed % segment) % token_count


def metric_value_basis(
    value: torch.Tensor,
    gram: torch.Tensor,
    *,
    sample_stride: int,
    maximum_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank Value basis minimizing error after a positive-definite metric."""

    value_float = value.float()
    mean = value_float.mean(dim=0)
    centered = value_float - mean
    dimension = int(value.shape[-1])
    regularizer = (
        gram.diagonal().mean().clamp_min(1.0e-8) * 1.0e-5
    )
    cholesky = torch.linalg.cholesky(
        gram.float()
        + regularizer
        * torch.eye(dimension, dtype=torch.float32, device=value.device)
    )
    transformed_sample = centered[::sample_stride] @ cholesky
    _, singular_values, right_vectors_h = torch.linalg.svd(
        transformed_sample,
        full_matrices=False,
    )
    active_rank = min(maximum_rank, dimension, right_vectors_h.shape[0])
    transformed_vectors = right_vectors_h[:active_rank].T.contiguous()
    coordinates = centered @ cholesky @ transformed_vectors
    value_vectors = torch.linalg.solve(
        cholesky.T,
        transformed_vectors,
    )
    energy = singular_values.square()
    explained = energy / energy.sum().clamp_min(1.0e-20)
    return mean, value_vectors, coordinates, explained


def output_metrics(output: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    absolute_l2 = torch.linalg.vector_norm(output - full)
    full_l2 = torch.linalg.vector_norm(full).clamp_min(1.0e-12)
    return {
        "absolute_l2": float(absolute_l2),
        "full_output_l2": float(full_l2),
        "relative_l2": float(absolute_l2 / full_l2),
        "cosine": float(F.cosine_similarity(output.float(), full.float(), dim=0)),
    }


def normalized_output(
    selected_scores: torch.Tensor,
    selected_values: torch.Tensor,
    tail_scores: torch.Tensor | None = None,
    tail_values: torch.Tensor | None = None,
    *,
    tail_alpha: float = 1.0,
) -> torch.Tensor:
    maximum = selected_scores.max()
    if tail_scores is not None and tail_scores.numel():
        maximum = torch.maximum(maximum, tail_scores.max())
    selected_weights = torch.exp(selected_scores.float() - maximum)
    numerator = torch.sum(
        selected_weights.unsqueeze(-1) * selected_values.float(), dim=0
    )
    denominator = selected_weights.sum()
    if tail_scores is not None and tail_values is not None and tail_scores.numel():
        tail_weights = torch.exp(tail_scores.float() - maximum)
        numerator = numerator + float(tail_alpha) * torch.sum(
            tail_weights.unsqueeze(-1) * tail_values.float(), dim=0
        )
        denominator = denominator + float(tail_alpha) * tail_weights.sum()
    return numerator / denominator.clamp_min(1.0e-20)


def append_row(
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
    method: str,
    output: torch.Tensor,
    full_output: torch.Tensor,
    *,
    sample_count: int,
    true_tail_partition: torch.Tensor,
    estimated_tail_partition: torch.Tensor,
    alpha: float,
    affine_slope: float = math.nan,
    affine_residual_std: float = math.nan,
    block_size: int = 0,
    residual_risk_absolute: float = math.nan,
    residual_risk_relative: float = math.nan,
    residual_risk_range_absolute: float = math.nan,
    residual_risk_bernstein_absolute: float = math.nan,
    residual_risk_bernstein_relative: float = math.nan,
    tail_correction_l2: float = math.nan,
    tail_effective_tokens: float = math.nan,
    proxy_selected_mass: float = math.nan,
    conditional_gain: float = math.nan,
    conditional_holdout_error_reduction: float = math.nan,
) -> None:
    tail_relative_error = float(
        torch.abs(estimated_tail_partition - true_tail_partition)
        / true_tail_partition.clamp_min(1.0e-20)
    )
    rows.append(
        {
            **identity,
            "method": method,
            "sample_count": sample_count,
            "block_size": block_size,
            "tail_partition_relative_error": tail_relative_error,
            "alpha": alpha,
            "affine_slope": affine_slope,
            "affine_residual_std": affine_residual_std,
            "residual_risk_absolute": residual_risk_absolute,
            "residual_risk_relative": residual_risk_relative,
            "residual_risk_range_absolute": residual_risk_range_absolute,
            "residual_risk_bernstein_absolute": (
                residual_risk_bernstein_absolute
            ),
            "residual_risk_bernstein_relative": (
                residual_risk_bernstein_relative
            ),
            "tail_correction_l2": tail_correction_l2,
            "tail_effective_tokens": tail_effective_tokens,
            "proxy_selected_mass": proxy_selected_mass,
            "conditional_gain": conditional_gain,
            "conditional_holdout_error_reduction": (
                conditional_holdout_error_reduction
            ),
            **output_metrics(output, full_output),
            "_output_tensor": output.detach().float().clone(),
            "_full_output_tensor": full_output.detach().float().clone(),
        }
    )


def load_output_projection(
    model_root: str, layer: int, device: torch.device
) -> torch.Tensor:
    from safetensors import safe_open

    root = Path(model_root)
    tensor_name = f"model.layers.{layer}.self_attn.o_proj.weight"
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index["weight_map"][tensor_name]
        candidates = (root / shard_name,)
    else:
        candidates = tuple(sorted(root.glob("*.safetensors")))
    for candidate in candidates:
        with safe_open(candidate, framework="pt", device="cpu") as handle:
            if tensor_name in handle.keys():
                return handle.get_tensor(tensor_name).to(device).float()
    raise KeyError(f"cannot find {tensor_name} under {root}")


def projected_layer_rows(
    rows: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model_name_or_path"],
                row["trace"],
                row["record_index"],
                row["history_tokens"],
                row["layer"],
                row["step"],
                row["candidate_mode"],
                row["method"],
                row["sample_count"],
                row["block_size"],
            )
        ].append(row)

    by_model_layer: dict[tuple[str, int], list[tuple[Any, ...]]] = defaultdict(
        list
    )
    for key in grouped:
        by_model_layer[(str(key[0]), int(key[4]))].append(key)

    output_rows: list[dict[str, Any]] = []
    for (model_root, layer), keys in sorted(by_model_layer.items()):
        if not model_root or not Path(model_root).exists():
            continue
        projection = load_output_projection(model_root, layer, device)
        for key in keys:
            items = sorted(grouped[key], key=lambda item: int(item["query_head"]))
            query_head_count = int(items[0]["query_head_count"])
            if len(items) != query_head_count:
                raise RuntimeError(
                    f"incomplete layer output for {key}: "
                    f"{len(items)} of {query_head_count} heads"
                )
            approximate = torch.cat(
                [item["_output_tensor"] for item in items], dim=0
            )
            full = torch.cat(
                [item["_full_output_tensor"] for item in items], dim=0
            )
            if projection.shape[1] != approximate.numel():
                raise RuntimeError(
                    f"o_proj input {projection.shape[1]} does not match "
                    f"concatenated heads {approximate.numel()}"
                )
            projected_approximate = projection @ approximate
            projected_full = projection @ full
            metrics = output_metrics(projected_approximate, projected_full)

            risks = [float(item["residual_risk_absolute"]) for item in items]
            predicted_absolute = math.nan
            predicted_relative = math.nan
            if all(math.isfinite(value) for value in risks):
                head_dim = int(items[0]["_output_tensor"].numel())
                predicted_variance = torch.zeros((), device=device)
                for query_head, risk in enumerate(risks):
                    column_start = query_head * head_dim
                    column_stop = column_start + head_dim
                    block_frobenius_squared = projection[
                        :, column_start:column_stop
                    ].square().sum()
                    predicted_variance.add_(
                        (risk * risk / head_dim) * block_frobenius_squared
                    )
                predicted_absolute = float(torch.sqrt(predicted_variance))
                predicted_relative = predicted_absolute / max(
                    float(torch.linalg.vector_norm(projected_full)), 1.0e-12
                )

            output_rows.append(
                {
                    "model_name_or_path": model_root,
                    "trace": key[1],
                    "record_index": key[2],
                    "history_tokens": key[3],
                    "layer": layer,
                    "step": key[5],
                    "candidate_mode": key[6],
                    "method": key[7],
                    "sample_count": key[8],
                    "block_size": key[9],
                    "query_heads": query_head_count,
                    "projected_absolute_l2": metrics["absolute_l2"],
                    "projected_full_output_l2": metrics["full_output_l2"],
                    "projected_relative_l2": metrics["relative_l2"],
                    "projected_cosine": metrics["cosine"],
                    "predicted_projected_risk_absolute": predicted_absolute,
                    "predicted_projected_risk_relative": predicted_relative,
                }
            )
        del projection
        torch.cuda.empty_cache()
    return output_rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(path) for path in args.traces.split(",") if path.strip())
    sample_counts = parse_csv(args.sample_counts, int)
    block_sizes = parse_csv(args.block_sizes, int)
    conditional_dims = parse_csv(args.conditional_dims, int)
    if (
        args.top_k <= 0
        or min(sample_counts) <= 0
        or args.conditional_fit_stride <= 0
    ):
        raise ValueError("top_k and sample counts must be positive")
    if not 0.0 < args.risk_delta < 1.0:
        raise ValueError("risk_delta must lie in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    projection_cache: dict[tuple[str, int], torch.Tensor] = {}

    for trace_path in traces:
        payload = torch.load(
            trace_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        history_tokens = int(payload.get("config", {}).get("history_tokens", 0))
        topic = str(payload.get("config", {}).get("topic", trace_path.stem))
        records = payload["records"]
        if args.max_records_per_trace > 0:
            records = records[: args.max_records_per_trace]
        cache_state_by_layer: dict[int, dict[str, Any]] = {}
        for state_record in payload["records"]:
            state_layer = int(state_record["layer"])
            if (
                state_layer not in cache_state_by_layer
                and state_record.get("key") is not None
                and state_record.get("value") is not None
            ):
                cache_state_by_layer[state_layer] = state_record
        model_name_or_path = str(
            args.model_name_or_path
            or payload.get("config", {}).get("model_name_or_path", "")
        )
        for record_index, record in enumerate(records):
            layer = int(record["layer"])
            state_record = (
                record
                if record.get("key") is not None
                and record.get("value") is not None
                else cache_state_by_layer.get(layer)
            )
            if state_record is None:
                raise ValueError(f"trace has no cached K/V state for layer {layer}")
            query = record["query"].to(device).float()[0, :, 0, :]
            key = state_record["key"].to(device).float()[0]
            value = state_record["value"].to(device).float()[0]
            scaling = float(record["scaling"])
            kv_head_count, token_count, _ = key.shape
            query_groups = query.shape[0] // kv_head_count
            keep_count = min(token_count, args.top_k)
            output_projection = None
            if args.value_metric == "wo_group":
                projection_key = (model_name_or_path, layer)
                if projection_key not in projection_cache:
                    projection_cache[projection_key] = load_output_projection(
                        model_name_or_path, layer, device
                    )
                output_projection = projection_cache[projection_key]

            for kv_head in range(kv_head_count):
                head_key = key[kv_head]
                head_value = value[kv_head]
                calibration_queries = query[
                    kv_head * query_groups : (kv_head + 1) * query_groups
                ]
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.key_sample_stride],
                    calibration_queries,
                    args.query_shrinkage,
                )
                key_coordinates = head_key @ key_factor
                projected_calibration = calibration_queries @ query_factor
                bands = quantized_bands(key_coordinates, projected_calibration)
                key_distortion, _ = distortion_table(
                    key_coordinates,
                    projected_calibration,
                    ZERO_BIT_LEVELS,
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.key_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                reconstructed_key = reconstruct(bands, allocation)

                if output_projection is None:
                    (
                        value_mean,
                        value_vectors,
                        value_coordinates,
                        explained,
                    ) = value_basis(
                        head_value,
                        sample_stride=args.value_sample_stride,
                        maximum_rank=args.value_rank,
                    )
                else:
                    head_dimension = int(head_value.shape[-1])
                    group_gram = torch.zeros(
                        head_dimension,
                        head_dimension,
                        dtype=torch.float32,
                        device=device,
                    )
                    for group in range(query_groups):
                        query_head = kv_head * query_groups + group
                        start = query_head * head_dimension
                        block = output_projection[
                            :, start : start + head_dimension
                        ].float()
                        group_gram.add_(block.T @ block)
                    (
                        value_mean,
                        value_vectors,
                        value_coordinates,
                        explained,
                    ) = metric_value_basis(
                        head_value,
                        group_gram,
                        sample_stride=args.value_sample_stride,
                        maximum_rank=args.value_rank,
                    )
                quantized_value_coordinates = block_affine_quantize(
                    value_coordinates[:, : args.value_rank],
                    bits=args.value_bits,
                    block_size=args.value_scale_block,
                )
                reconstructed_value = (
                    value_mean
                    + quantized_value_coordinates
                    @ value_vectors[:, : args.value_rank].T
                )
                residual = head_value.float() - reconstructed_value.float()
                block_models: dict[
                    int,
                    tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                    ],
                ] = {}
                for block_size in block_sizes:
                    block_ids = torch.arange(token_count, device=device) // block_size
                    block_count = int(block_ids[-1]) + 1
                    block_counts = torch.zeros(
                        block_count, dtype=torch.float32, device=device
                    )
                    block_counts.index_add_(
                        0,
                        block_ids,
                        torch.ones(token_count, dtype=torch.float32, device=device),
                    )
                    block_value_sums = torch.zeros(
                        block_count,
                        head_value.shape[-1],
                        dtype=torch.float32,
                        device=device,
                    )
                    block_value_sums.index_add_(0, block_ids, head_value.float())
                    block_reconstructed_value_sums = torch.zeros_like(
                        block_value_sums
                    )
                    block_reconstructed_value_sums.index_add_(
                        0,
                        block_ids,
                        reconstructed_value.float(),
                    )
                    block_residual_square_sums = torch.zeros(
                        block_count, dtype=torch.float32, device=device
                    )
                    block_residual_square_sums.index_add_(
                        0,
                        block_ids,
                        residual.square().sum(dim=-1),
                    )
                    block_residual_sums = (
                        block_value_sums - block_reconstructed_value_sums
                    )
                    block_residual_means = quantize_block_moment(
                        block_residual_sums
                        / block_counts[:, None].clamp_min(1.0),
                        args.block_moment_bits,
                        (1,),
                    )
                    block_residual_sums = (
                        block_residual_means * block_counts[:, None]
                    )
                    block_models[block_size] = (
                        block_ids,
                        block_counts,
                        block_value_sums,
                        block_residual_sums,
                        block_residual_square_sums,
                    )
                conditional_models: dict[
                    tuple[int, int], dict[str, torch.Tensor]
                ] = {}
                for block_size, block_model in block_models.items():
                    block_ids, block_counts, _, block_residual_sums, _ = (
                        block_model
                    )
                    block_residual_means = (
                        block_residual_sums
                        / block_counts[:, None].clamp_min(1.0)
                    )
                    for conditional_dim in conditional_dims:
                        active_dim = min(
                            conditional_dim, reconstructed_key.shape[-1]
                        )
                        coordinates = reconstructed_key[:, :active_dim].float()
                        block_coordinate_sums = torch.zeros(
                            block_counts.numel(),
                            active_dim,
                            dtype=torch.float32,
                            device=device,
                        )
                        block_coordinate_sums.index_add_(
                            0, block_ids, coordinates
                        )
                        block_coordinate_means = (
                            block_coordinate_sums
                            / block_counts[:, None].clamp_min(1.0)
                        )
                        block_coordinate_means = quantize_block_moment(
                            block_coordinate_means,
                            args.block_moment_bits,
                            (1,),
                        )
                        block_coordinate_sums = (
                            block_coordinate_means * block_counts[:, None]
                        )
                        fit_indices = torch.arange(
                            0,
                            token_count,
                            args.conditional_fit_stride,
                            device=device,
                        )
                        fit_block_ids = block_ids[fit_indices]
                        fit_block_counts = torch.zeros_like(block_counts)
                        fit_block_counts.index_add_(
                            0,
                            fit_block_ids,
                            torch.ones_like(fit_block_ids, dtype=torch.float32),
                        )
                        fit_coordinate_sums = torch.zeros_like(
                            block_coordinate_sums
                        )
                        fit_coordinate_sums.index_add_(
                            0, fit_block_ids, coordinates[fit_indices]
                        )
                        fit_residual_sums = torch.zeros_like(block_residual_sums)
                        fit_residual_sums.index_add_(
                            0, fit_block_ids, residual[fit_indices]
                        )
                        fit_coordinate_means = fit_coordinate_sums / (
                            fit_block_counts[:, None].clamp_min(1.0)
                        )
                        fit_residual_means = fit_residual_sums / (
                            fit_block_counts[:, None].clamp_min(1.0)
                        )
                        centered_fit_coordinates = (
                            coordinates[fit_indices]
                            - fit_coordinate_means[fit_block_ids]
                        )
                        centered_fit_residual = (
                            residual[fit_indices]
                            - fit_residual_means[fit_block_ids]
                        )
                        fit_count = int(fit_indices.numel())
                        covariance = (
                            centered_fit_coordinates.T
                            @ centered_fit_coordinates
                        ) / float(fit_count)
                        cross_covariance = (
                            centered_fit_residual.T
                            @ centered_fit_coordinates
                        ) / float(fit_count)
                        ridge = (
                            covariance.diagonal().mean().clamp_min(1.0e-8)
                            * 1.0e-3
                        )
                        linear_map = torch.linalg.solve(
                            covariance
                            + ridge
                            * torch.eye(
                                active_dim,
                                dtype=torch.float32,
                                device=device,
                            ),
                            cross_covariance.T,
                        ).T
                        linear_map = quantize_block_moment(
                            linear_map,
                            args.block_moment_bits,
                            (0, 1),
                        )
                        train_indices = fit_indices[::2]
                        calibration_indices = fit_indices[1::2]
                        if train_indices.numel() == 0:
                            train_indices = fit_indices
                        train_block_ids = block_ids[train_indices]
                        train_block_counts = torch.zeros_like(block_counts)
                        train_block_counts.index_add_(
                            0,
                            train_block_ids,
                            torch.ones_like(
                                train_block_ids, dtype=torch.float32
                            ),
                        )
                        train_coordinate_sums = torch.zeros_like(
                            block_coordinate_sums
                        )
                        train_coordinate_sums.index_add_(
                            0, train_block_ids, coordinates[train_indices]
                        )
                        train_residual_sums = torch.zeros_like(
                            block_residual_sums
                        )
                        train_residual_sums.index_add_(
                            0, train_block_ids, residual[train_indices]
                        )
                        train_coordinate_means = train_coordinate_sums / (
                            train_block_counts[:, None].clamp_min(1.0)
                        )
                        train_residual_means = train_residual_sums / (
                            train_block_counts[:, None].clamp_min(1.0)
                        )
                        centered_train_coordinates = (
                            coordinates[train_indices]
                            - train_coordinate_means[train_block_ids]
                        )
                        centered_train_residual = (
                            residual[train_indices]
                            - train_residual_means[train_block_ids]
                        )
                        train_count = max(1, int(train_indices.numel()))
                        train_covariance = (
                            centered_train_coordinates.T
                            @ centered_train_coordinates
                        ) / float(train_count)
                        train_cross_covariance = (
                            centered_train_residual.T
                            @ centered_train_coordinates
                        ) / float(train_count)
                        train_ridge = (
                            train_covariance.diagonal()
                            .mean()
                            .clamp_min(1.0e-8)
                            * 1.0e-3
                        )
                        train_linear_map = torch.linalg.solve(
                            train_covariance
                            + train_ridge
                            * torch.eye(
                                active_dim,
                                dtype=torch.float32,
                                device=device,
                            ),
                            train_cross_covariance.T,
                        ).T
                        if calibration_indices.numel() > 0:
                            calibration_block_ids = block_ids[
                                calibration_indices
                            ]
                            valid_calibration = (
                                train_block_counts[calibration_block_ids] > 0
                            )
                            calibration_indices = calibration_indices[
                                valid_calibration
                            ]
                            calibration_block_ids = calibration_block_ids[
                                valid_calibration
                            ]
                        if calibration_indices.numel() > 0:
                            calibration_prediction = (
                                coordinates[calibration_indices]
                                - train_coordinate_means[
                                    calibration_block_ids
                                ]
                            ) @ train_linear_map.T
                            calibration_target = (
                                residual[calibration_indices]
                                - train_residual_means[
                                    calibration_block_ids
                                ]
                            )
                            wiener_gain, wiener_holdout_reduction = (
                                clipped_wiener_gain(
                                    calibration_prediction,
                                    calibration_target,
                                )
                            )
                        else:
                            wiener_gain = torch.zeros(
                                (), dtype=torch.float32, device=device
                            )
                            wiener_holdout_reduction = torch.zeros_like(
                                wiener_gain
                            )
                        wiener_linear_map = quantize_block_moment(
                            wiener_gain * train_linear_map,
                            args.block_moment_bits,
                            (0, 1),
                        )
                        conditional_error = (
                            residual - block_residual_means[block_ids]
                        ) - (
                            (coordinates - block_coordinate_means[block_ids])
                            @ linear_map.T
                        )
                        block_conditional_error_sums = torch.zeros_like(
                            block_residual_sums
                        )
                        block_conditional_error_sums.index_add_(
                            0, block_ids, conditional_error
                        )
                        block_conditional_error_square_sums = torch.zeros(
                            block_counts.numel(),
                            dtype=torch.float32,
                            device=device,
                        )
                        block_conditional_error_square_sums.index_add_(
                            0,
                            block_ids,
                            conditional_error.square().sum(dim=-1),
                        )
                        block_conditional_error_max_norms = torch.zeros(
                            block_counts.numel(),
                            dtype=torch.float32,
                            device=device,
                        )
                        block_conditional_error_max_norms.scatter_reduce_(
                            0,
                            block_ids,
                            torch.linalg.vector_norm(
                                conditional_error, dim=-1
                            ),
                            reduce="amax",
                            include_self=True,
                        )
                        conditional_models[(block_size, active_dim)] = {
                            "coordinates": coordinates,
                            "block_coordinate_sums": block_coordinate_sums,
                            "block_coordinate_means": block_coordinate_means,
                            "block_residual_means": block_residual_means,
                            "linear_map": linear_map,
                            "wiener_linear_map": wiener_linear_map,
                            "wiener_gain": wiener_gain,
                            "wiener_holdout_error_reduction": (
                                wiener_holdout_reduction
                            ),
                            "block_error_sums": block_conditional_error_sums,
                            "block_error_square_sums": (
                                block_conditional_error_square_sums
                            ),
                            "block_error_max_norms": (
                                block_conditional_error_max_norms
                            ),
                        }

                for group in range(query_groups):
                    query_head = kv_head * query_groups + group
                    exact_scores = head_key @ query[query_head] * scaling
                    approximate_query = query_int8(
                        query[query_head] @ query_factor
                    )
                    proxy_scores = (
                        reconstructed_key.float() @ approximate_query.float()
                    ) * scaling
                    common_center = torch.maximum(
                        exact_scores.max(), proxy_scores.max()
                    )
                    exact_weights = torch.exp(exact_scores - common_center)
                    proxy_weights = torch.exp(proxy_scores - common_center)
                    exact_indices = torch.topk(
                        exact_scores, k=keep_count, sorted=False
                    ).indices
                    proxy_indices = torch.topk(
                        proxy_scores, k=keep_count, sorted=False
                    ).indices
                    full_output = torch.sum(
                        exact_weights.unsqueeze(-1) * head_value, dim=0
                    ) / exact_weights.sum().clamp_min(1.0e-20)

                    for candidate_mode, candidate_indices in (
                        ("exact", exact_indices),
                        ("proxy", proxy_indices),
                    ):
                        selected = torch.zeros(
                            token_count,
                            dtype=torch.bool,
                            device=device,
                        )
                        selected[candidate_indices] = True
                        tail = ~selected
                        selected_exact_scores = exact_scores[candidate_indices]
                        selected_values = head_value[candidate_indices]
                        selected_exact_weights = exact_weights[candidate_indices]
                        selected_partition = selected_exact_weights.sum()
                        selected_numerator = torch.sum(
                            selected_exact_weights.unsqueeze(-1) * selected_values,
                            dim=0,
                        )
                        exact_tail_partition = exact_weights[tail].sum()
                        proxy_tail_partition = proxy_weights[tail].sum()
                        proxy_tail_numerator = torch.sum(
                            proxy_weights[tail].unsqueeze(-1)
                            * reconstructed_value[tail],
                            dim=0,
                        )
                        exact_sketch_tail_numerator = torch.sum(
                            exact_weights[tail].unsqueeze(-1)
                            * reconstructed_value[tail],
                            dim=0,
                        )
                        identity = {
                            "model_name_or_path": model_name_or_path,
                            "trace": trace_path.stem,
                            "record_index": record_index,
                            "step": int(record.get("step", -1)),
                            "topic": topic,
                            "history_tokens": history_tokens or token_count,
                            "token_count": token_count,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "query_head_count": int(query.shape[0]),
                            "candidate_mode": candidate_mode,
                            "top_k": keep_count,
                            "selected_mass": float(
                                selected_partition
                                / exact_weights.sum().clamp_min(1.0e-20)
                            ),
                            "value_explained_variance": float(
                                explained[: args.value_rank].sum()
                            ),
                        }

                        selected_only = selected_numerator / selected_partition.clamp_min(
                            1.0e-20
                        )
                        append_row(
                            rows,
                            identity,
                            "selected_only",
                            selected_only,
                            full_output,
                            sample_count=0,
                            true_tail_partition=exact_tail_partition,
                            estimated_tail_partition=torch.zeros_like(
                                exact_tail_partition
                            ),
                            alpha=0.0,
                        )
                        for fixed_alpha in (0.5, 1.0):
                            denominator = (
                                selected_partition
                                + fixed_alpha * proxy_tail_partition
                            )
                            output = (
                                selected_numerator
                                + fixed_alpha * proxy_tail_numerator
                            ) / denominator.clamp_min(1.0e-20)
                            append_row(
                                rows,
                                identity,
                                f"fixed_alpha_{fixed_alpha:g}",
                                output,
                                full_output,
                                sample_count=0,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=(
                                    fixed_alpha * proxy_tail_partition
                                ),
                                alpha=fixed_alpha,
                            )

                        # Treat the rank-r Value-tail contribution as a noisy
                        # correction to selected-only attention.  The residual
                        # second moment is request-local metadata; the query
                        # contributes only tail sum(w^2).  Positive-part SURE
                        # then gives a continuous, length-free shrinkage rule.
                        full_tail_denominator = (
                            selected_partition + proxy_tail_partition
                        ).clamp_min(1.0e-20)
                        full_tail_output = (
                            selected_numerator + proxy_tail_numerator
                        ) / full_tail_denominator
                        tail_residual = residual[tail]
                        tail_residual_centered = (
                            tail_residual
                            - tail_residual.mean(dim=0, keepdim=True)
                        )
                        tail_residual_variance = (
                            tail_residual_centered.square().sum()
                            / max(1, int(tail_residual.shape[0]) - 1)
                        )
                        tail_noise_squared = (
                            proxy_weights[tail].square().sum()
                            * tail_residual_variance
                            / full_tail_denominator.square()
                        )
                        correction = full_tail_output - selected_only
                        correction_squared = correction.square().sum().clamp_min(
                            1.0e-20
                        )
                        tail_weight_square_sum = proxy_weights[tail].square().sum()
                        tail_effective_tokens = float(
                            proxy_tail_partition.square()
                            / tail_weight_square_sum.clamp_min(1.0e-20)
                        )
                        proxy_selected_mass = float(
                            selected_partition / full_tail_denominator
                        )
                        sure_alpha = float(
                            (
                                1.0
                                - tail_noise_squared / correction_squared
                            ).clamp(0.0, 1.0)
                        )
                        ridge_alpha = float(
                            correction_squared
                            / (correction_squared + tail_noise_squared).clamp_min(
                                1.0e-20
                            )
                        )
                        for method, shrinkage in (
                            ("sure_tail_shrinkage", sure_alpha),
                            ("ridge_tail_shrinkage", ridge_alpha),
                        ):
                            shrinkage_output = selected_only + shrinkage * correction
                            append_row(
                                rows,
                                identity,
                                method,
                                shrinkage_output,
                                full_output,
                                sample_count=0,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=(
                                    shrinkage * proxy_tail_partition
                                ),
                                alpha=shrinkage,
                                residual_risk_absolute=float(
                                    torch.sqrt(tail_noise_squared)
                                ),
                                residual_risk_relative=float(
                                    torch.sqrt(tail_noise_squared)
                                    / torch.linalg.vector_norm(
                                        full_output
                                    ).clamp_min(1.0e-12)
                                ),
                                tail_correction_l2=float(
                                    torch.sqrt(correction_squared)
                                ),
                                tail_effective_tokens=tail_effective_tokens,
                                proxy_selected_mass=proxy_selected_mass,
                            )

                        oracle_lambda_unclipped = float(
                            torch.dot(full_output - selected_only, correction)
                            / correction_squared
                        )
                        oracle_lambda = min(
                            1.0, max(0.0, oracle_lambda_unclipped)
                        )
                        oracle_shrinkage_output = (
                            selected_only + oracle_lambda * correction
                        )
                        append_row(
                            rows,
                            identity,
                            "oracle_tail_shrinkage",
                            oracle_shrinkage_output,
                            full_output,
                            sample_count=0,
                            true_tail_partition=exact_tail_partition,
                            estimated_tail_partition=(
                                oracle_lambda * proxy_tail_partition
                            ),
                            alpha=oracle_lambda,
                            affine_slope=oracle_lambda_unclipped,
                            residual_risk_absolute=float(
                                torch.sqrt(tail_noise_squared)
                            ),
                            residual_risk_relative=float(
                                torch.sqrt(tail_noise_squared)
                                / torch.linalg.vector_norm(
                                    full_output
                                ).clamp_min(1.0e-12)
                            ),
                            tail_correction_l2=float(
                                torch.sqrt(correction_squared)
                            ),
                            tail_effective_tokens=tail_effective_tokens,
                            proxy_selected_mass=proxy_selected_mass,
                        )

                        oracle_alpha = float(
                            exact_tail_partition
                            / proxy_tail_partition.clamp_min(1.0e-20)
                        )
                        oracle_mass_output = (
                            selected_numerator
                            + oracle_alpha * proxy_tail_numerator
                        ) / (
                            selected_partition + exact_tail_partition
                        ).clamp_min(1.0e-20)
                        append_row(
                            rows,
                            identity,
                            "oracle_tail_mass_proxy_value",
                            oracle_mass_output,
                            full_output,
                            sample_count=0,
                            true_tail_partition=exact_tail_partition,
                            estimated_tail_partition=exact_tail_partition,
                            alpha=oracle_alpha,
                        )
                        exact_score_sketch_output = (
                            selected_numerator + exact_sketch_tail_numerator
                        ) / (
                            selected_partition + exact_tail_partition
                        ).clamp_min(1.0e-20)
                        append_row(
                            rows,
                            identity,
                            "exact_score_value_sketch",
                            exact_score_sketch_output,
                            full_output,
                            sample_count=0,
                            true_tail_partition=exact_tail_partition,
                            estimated_tail_partition=exact_tail_partition,
                            alpha=1.0,
                        )

                        conditional_tail_states: dict[
                            tuple[int, int], dict[str, torch.Tensor]
                        ] = {}
                        for block_size, block_model in block_models.items():
                            (
                                block_ids,
                                block_counts,
                                block_value_sums,
                                block_residual_sums,
                                block_residual_square_sums,
                            ) = block_model
                            selected_blocks = block_ids[candidate_indices]
                            selected_counts = torch.zeros_like(block_counts)
                            selected_counts.index_add_(
                                0,
                                selected_blocks,
                                torch.ones_like(
                                    selected_blocks, dtype=torch.float32
                                ),
                            )
                            selected_value_sums = torch.zeros_like(block_value_sums)
                            selected_value_sums.index_add_(
                                0,
                                selected_blocks,
                                selected_values.float(),
                            )
                            selected_residual_sums = torch.zeros_like(
                                block_residual_sums
                            )
                            selected_residual_sums.index_add_(
                                0,
                                selected_blocks,
                                selected_values.float()
                                - reconstructed_value[candidate_indices].float(),
                            )
                            selected_residual_square_sums = torch.zeros_like(
                                block_residual_square_sums
                            )
                            selected_residual_square_sums.index_add_(
                                0,
                                selected_blocks,
                                (
                                    selected_values.float()
                                    - reconstructed_value[
                                        candidate_indices
                                    ].float()
                                ).square().sum(dim=-1),
                            )
                            tail_counts = block_counts - selected_counts
                            tail_means = (
                                block_value_sums - selected_value_sums
                            ) / tail_counts[:, None].clamp_min(1.0)

                            for score_mode, weights in (
                                ("exact", exact_weights),
                                ("proxy", proxy_weights),
                            ):
                                tail_block_weights = torch.zeros_like(block_counts)
                                tail_block_weights.index_add_(
                                    0,
                                    block_ids[tail],
                                    weights[tail],
                                )
                                tail_block_squared_weights = torch.zeros_like(
                                    block_counts
                                )
                                tail_block_squared_weights.index_add_(
                                    0,
                                    block_ids[tail],
                                    weights[tail].square(),
                                )
                                tail_numerator = torch.sum(
                                    tail_block_weights[:, None] * tail_means,
                                    dim=0,
                                )
                                tail_partition = tail_block_weights.sum()
                                block_output = (
                                    selected_numerator + tail_numerator
                                ) / (
                                    selected_partition + tail_partition
                                ).clamp_min(1.0e-20)
                                append_row(
                                    rows,
                                    identity,
                                    f"block_tail_mean_{score_mode}",
                                    block_output,
                                    full_output,
                                    sample_count=0,
                                    true_tail_partition=exact_tail_partition,
                                    estimated_tail_partition=tail_partition,
                                    alpha=float(
                                        tail_partition
                                        / proxy_tail_partition.clamp_min(1.0e-20)
                                    ),
                                    block_size=block_size,
                                )

                                tail_residual_means = (
                                    block_residual_sums - selected_residual_sums
                                ) / tail_counts[:, None].clamp_min(1.0)
                                residual_correction = torch.sum(
                                    tail_block_weights[:, None]
                                    * tail_residual_means,
                                    dim=0,
                                )
                                residual_output = (
                                    selected_numerator
                                    + (
                                        exact_sketch_tail_numerator
                                        if score_mode == "exact"
                                        else proxy_tail_numerator
                                    )
                                    + residual_correction
                                ) / (
                                    selected_partition + tail_partition
                                ).clamp_min(1.0e-20)
                                tail_residual_mean_square = (
                                    block_residual_square_sums
                                    - selected_residual_square_sums
                                ) / tail_counts.clamp_min(1.0)
                                tail_residual_variance = (
                                    tail_residual_mean_square
                                    - tail_residual_means.square().sum(dim=-1)
                                ).clamp_min(0.0)
                                residual_risk_absolute = torch.sqrt(
                                    torch.sum(
                                        tail_block_squared_weights
                                        * tail_residual_variance
                                    )
                                ) / (
                                    selected_partition + tail_partition
                                ).clamp_min(1.0e-20)
                                residual_risk_relative = (
                                    residual_risk_absolute
                                    / torch.linalg.vector_norm(
                                        full_output
                                    ).clamp_min(1.0e-12)
                                )
                                append_row(
                                    rows,
                                    identity,
                                    f"block_residual_mean_{score_mode}",
                                    residual_output,
                                    full_output,
                                    sample_count=0,
                                    true_tail_partition=exact_tail_partition,
                                    estimated_tail_partition=tail_partition,
                                    alpha=float(
                                        tail_partition
                                        / proxy_tail_partition.clamp_min(1.0e-20)
                                    ),
                                    block_size=block_size,
                                    residual_risk_absolute=float(
                                        residual_risk_absolute
                                    ),
                                    residual_risk_relative=float(
                                        residual_risk_relative
                                    ),
                                )

                                for conditional_dim in conditional_dims:
                                    active_dim = min(
                                        conditional_dim,
                                        reconstructed_key.shape[-1],
                                    )
                                    conditional_model = conditional_models[
                                        (block_size, active_dim)
                                    ]
                                    coordinates = conditional_model[
                                        "coordinates"
                                    ]
                                    block_coordinate_sums = conditional_model[
                                        "block_coordinate_sums"
                                    ]
                                    block_coordinate_means = conditional_model[
                                        "block_coordinate_means"
                                    ]
                                    block_residual_means = conditional_model[
                                        "block_residual_means"
                                    ]
                                    linear_map = conditional_model["linear_map"]
                                    selected_coordinate_sums = torch.zeros_like(
                                        block_coordinate_sums
                                    )
                                    selected_coordinate_sums.index_add_(
                                        0,
                                        selected_blocks,
                                        coordinates[candidate_indices],
                                    )
                                    tail_coordinate_means = (
                                        block_coordinate_sums
                                        - selected_coordinate_sums
                                    ) / tail_counts[:, None].clamp_min(1.0)
                                    tail_weighted_coordinates = torch.zeros_like(
                                        block_coordinate_sums
                                    )
                                    tail_weighted_coordinates.index_add_(
                                        0,
                                        block_ids[tail],
                                        weights[tail].unsqueeze(-1)
                                        * coordinates[tail],
                                    )
                                    weighted_coordinate_means = (
                                        tail_weighted_coordinates
                                        / tail_block_weights[:, None].clamp_min(
                                            1.0e-20
                                        )
                                    )
                                    conditional_residual_means = (
                                        tail_residual_means
                                        + torch.einsum(
                                            "di,bi->bd",
                                            linear_map,
                                            weighted_coordinate_means
                                            - tail_coordinate_means,
                                        )
                                    )
                                    conditional_correction = torch.sum(
                                        tail_block_weights[:, None]
                                        * conditional_residual_means,
                                        dim=0,
                                    )
                                    selected_conditional_error = (
                                        selected_values.float()
                                        - reconstructed_value[
                                            candidate_indices
                                        ].float()
                                        - block_residual_means[selected_blocks]
                                        - (
                                            coordinates[candidate_indices]
                                            - block_coordinate_means[
                                                selected_blocks
                                            ]
                                        )
                                        @ linear_map.T
                                    )
                                    selected_conditional_error_sums = (
                                        torch.zeros_like(block_residual_sums)
                                    )
                                    selected_conditional_error_sums.index_add_(
                                        0,
                                        selected_blocks,
                                        selected_conditional_error,
                                    )
                                    selected_conditional_error_square_sums = (
                                        torch.zeros_like(block_counts)
                                    )
                                    selected_conditional_error_square_sums.index_add_(
                                        0,
                                        selected_blocks,
                                        selected_conditional_error.square().sum(
                                            dim=-1
                                        ),
                                    )
                                    tail_conditional_error_means = (
                                        conditional_model["block_error_sums"]
                                        - selected_conditional_error_sums
                                    ) / tail_counts[:, None].clamp_min(1.0)
                                    tail_conditional_error_mean_square = (
                                        conditional_model[
                                            "block_error_square_sums"
                                        ]
                                        - selected_conditional_error_square_sums
                                    ) / tail_counts.clamp_min(1.0)
                                    tail_conditional_error_variance = (
                                        tail_conditional_error_mean_square
                                        - tail_conditional_error_means.square().sum(
                                            dim=-1
                                        )
                                    ).clamp_min(0.0)
                                    conditional_variance_numerator = torch.sum(
                                        tail_block_squared_weights
                                        * tail_conditional_error_variance
                                    )
                                    output_denominator = (
                                        selected_partition + tail_partition
                                    ).clamp_min(1.0e-20)
                                    conditional_risk_absolute = torch.sqrt(
                                        conditional_variance_numerator
                                    ) / output_denominator
                                    tail_block_max_weights = torch.zeros_like(
                                        block_counts
                                    )
                                    tail_block_max_weights.scatter_reduce_(
                                        0,
                                        block_ids[tail],
                                        weights[tail],
                                        reduce="amax",
                                        include_self=True,
                                    )
                                    tail_error_max_norms = (
                                        conditional_model[
                                            "block_error_max_norms"
                                        ]
                                        + torch.linalg.vector_norm(
                                            tail_conditional_error_means,
                                            dim=-1,
                                        )
                                    )
                                    maximum_weighted_residual = torch.max(
                                        tail_block_max_weights
                                        * tail_error_max_norms
                                    )
                                    bernstein_numerator = (
                                        vector_bernstein_radius(
                                            conditional_variance_numerator,
                                            maximum_weighted_residual,
                                            failure_probability=(
                                                args.risk_delta
                                            ),
                                            dimension=int(
                                                head_value.shape[-1]
                                            ),
                                        )
                                    )
                                    bernstein_absolute = (
                                        bernstein_numerator
                                        / output_denominator
                                    )
                                    full_output_norm = (
                                        torch.linalg.vector_norm(full_output)
                                        .clamp_min(1.0e-12)
                                    )
                                    conditional_output = (
                                        selected_numerator
                                        + (
                                            exact_sketch_tail_numerator
                                            if score_mode == "exact"
                                            else proxy_tail_numerator
                                        )
                                        + conditional_correction
                                    ) / (
                                        selected_partition + tail_partition
                                    ).clamp_min(1.0e-20)
                                    append_row(
                                        rows,
                                        identity,
                                        (
                                            "block_conditional_residual_"
                                            f"{score_mode}_d{active_dim}"
                                        ),
                                        conditional_output,
                                        full_output,
                                        sample_count=0,
                                        true_tail_partition=(
                                            exact_tail_partition
                                        ),
                                        estimated_tail_partition=(
                                            tail_partition
                                        ),
                                        alpha=float(
                                            tail_partition
                                            / proxy_tail_partition.clamp_min(
                                                1.0e-20
                                            )
                                        ),
                                        block_size=block_size,
                                        residual_risk_absolute=float(
                                            conditional_risk_absolute
                                        ),
                                        residual_risk_relative=float(
                                            conditional_risk_absolute
                                            / full_output_norm
                                        ),
                                        residual_risk_range_absolute=float(
                                            maximum_weighted_residual
                                            / output_denominator
                                        ),
                                        residual_risk_bernstein_absolute=float(
                                            bernstein_absolute
                                        ),
                                        residual_risk_bernstein_relative=float(
                                            bernstein_absolute
                                            / full_output_norm
                                        ),
                                    )
                                    wiener_linear_map = conditional_model[
                                        "wiener_linear_map"
                                    ]
                                    wiener_conditional_residual_means = (
                                        tail_residual_means
                                        + torch.einsum(
                                            "di,bi->bd",
                                            wiener_linear_map,
                                            weighted_coordinate_means
                                            - tail_coordinate_means,
                                        )
                                    )
                                    wiener_conditional_correction = torch.sum(
                                        tail_block_weights[:, None]
                                        * wiener_conditional_residual_means,
                                        dim=0,
                                    )
                                    wiener_conditional_output = (
                                        selected_numerator
                                        + (
                                            exact_sketch_tail_numerator
                                            if score_mode == "exact"
                                            else proxy_tail_numerator
                                        )
                                        + wiener_conditional_correction
                                    ) / output_denominator
                                    append_row(
                                        rows,
                                        identity,
                                        (
                                            "block_conditional_wiener_"
                                            f"residual_{score_mode}_d{active_dim}"
                                        ),
                                        wiener_conditional_output,
                                        full_output,
                                        sample_count=0,
                                        true_tail_partition=(
                                            exact_tail_partition
                                        ),
                                        estimated_tail_partition=(
                                            tail_partition
                                        ),
                                        alpha=float(
                                            tail_partition
                                            / proxy_tail_partition.clamp_min(
                                                1.0e-20
                                            )
                                        ),
                                        block_size=block_size,
                                        conditional_gain=float(
                                            conditional_model["wiener_gain"]
                                        ),
                                        conditional_holdout_error_reduction=float(
                                            conditional_model[
                                                "wiener_holdout_error_reduction"
                                            ]
                                        ),
                                    )
                                    if score_mode == "proxy":
                                        conditional_tail_states[
                                            (block_size, active_dim)
                                        ] = {
                                            "coordinates": coordinates,
                                            "block_ids": block_ids,
                                            "tail_coordinate_means": (
                                                tail_coordinate_means
                                            ),
                                            "tail_residual_means": (
                                                tail_residual_means
                                            ),
                                            "linear_map": linear_map,
                                            "tail_numerator": (
                                                proxy_tail_numerator
                                                + conditional_correction
                                            ),
                                        }

                        for sample_count in sample_counts:
                            all_tail_indices = torch.nonzero(
                                tail, as_tuple=False
                            ).flatten()
                            tail_count = int(all_tail_indices.numel())
                            if tail_count == 0:
                                continue
                            active_sample_count = min(sample_count, tail_count)
                            if args.tail_sampling == "random":
                                generator = torch.Generator(device=device)
                                generator.manual_seed(
                                    1000003 * layer
                                    + 1009 * query_head
                                    + 17 * sample_count
                                )
                                sampled_tail_offsets = torch.randint(
                                    0,
                                    tail_count,
                                    (active_sample_count,),
                                    generator=generator,
                                    device=device,
                                )
                            else:
                                sampled_tail_offsets = systematic_indices(
                                    tail_count,
                                    active_sample_count,
                                    1009 * layer + 131 * query_head,
                                    device,
                                )
                            tail_sample_indices = all_tail_indices[
                                sampled_tail_offsets
                            ]
                            expansion = float(tail_count) / float(
                                active_sample_count
                            )
                            sampled_exact_weights = exact_weights[
                                tail_sample_indices
                            ]
                            sampled_proxy_weights = proxy_weights[
                                tail_sample_indices
                            ]
                            sampled_values = reconstructed_value[
                                tail_sample_indices
                            ]
                            sampled_true_values = head_value[tail_sample_indices]
                            weight_difference = (
                                sampled_exact_weights - sampled_proxy_weights
                            )
                            cv_tail_partition = (
                                proxy_tail_partition
                                + expansion * weight_difference.sum()
                            ).clamp_min(0.0)

                            for (
                                conditional_block_size,
                                conditional_dim,
                            ), conditional_state in conditional_tail_states.items():
                                sample_blocks = conditional_state["block_ids"][
                                    tail_sample_indices
                                ]
                                sample_prediction = (
                                    sampled_values
                                    + conditional_state[
                                        "tail_residual_means"
                                    ][sample_blocks]
                                    + (
                                        conditional_state["coordinates"][
                                            tail_sample_indices
                                        ]
                                        - conditional_state[
                                            "tail_coordinate_means"
                                        ][sample_blocks]
                                    )
                                    @ conditional_state["linear_map"].T
                                )
                                conditional_joint_residual = (
                                    sampled_exact_weights.unsqueeze(-1)
                                    * sampled_true_values
                                    - sampled_proxy_weights.unsqueeze(-1)
                                    * sample_prediction
                                )
                                conditional_tail_numerator = (
                                    conditional_state["tail_numerator"]
                                    + expansion
                                    * conditional_joint_residual.sum(dim=0)
                                )
                                conditional_cv_output = (
                                    selected_numerator
                                    + conditional_tail_numerator
                                ) / (
                                    selected_partition + cv_tail_partition
                                ).clamp_min(1.0e-20)
                                sample_size = int(tail_sample_indices.numel())
                                if sample_size > 1:
                                    centered_joint_residual = (
                                        conditional_joint_residual
                                        - conditional_joint_residual.mean(
                                            dim=0, keepdim=True
                                        )
                                    )
                                    finite_population = (
                                        max(
                                            0.0,
                                            1.0
                                            - sample_size
                                            / max(1.0, float(tail_count)),
                                        )
                                        if args.tail_sampling == "systematic"
                                        else 1.0
                                    )
                                    numerator_standard_error = float(tail.sum()) * torch.sqrt(
                                        finite_population
                                        * centered_joint_residual.square().sum()
                                        / float(sample_size * (sample_size - 1))
                                    )
                                    output_standard_error = (
                                        numerator_standard_error
                                        / (
                                            selected_partition
                                            + cv_tail_partition
                                        ).clamp_min(1.0e-20)
                                    )
                                else:
                                    output_standard_error = torch.full(
                                        (),
                                        float("inf"),
                                        device=device,
                                    )
                                append_row(
                                    rows,
                                    identity,
                                    (
                                        "sample_conditional_joint_cv_proxy_"
                                        f"d{conditional_dim}"
                                    ),
                                    conditional_cv_output,
                                    full_output,
                                    sample_count=sample_count,
                                    true_tail_partition=exact_tail_partition,
                                    estimated_tail_partition=cv_tail_partition,
                                    alpha=float(
                                        cv_tail_partition
                                        / proxy_tail_partition.clamp_min(
                                            1.0e-20
                                        )
                                    ),
                                    block_size=conditional_block_size,
                                    residual_risk_absolute=float(
                                        output_standard_error
                                    ),
                                    residual_risk_relative=float(
                                        output_standard_error
                                        / torch.linalg.vector_norm(
                                            conditional_cv_output
                                        ).clamp_min(1.0e-12)
                                    ),
                                )

                            ratio_alpha = float(
                                sampled_exact_weights.sum()
                                / sampled_proxy_weights.sum().clamp_min(1.0e-20)
                            )
                            ratio_output = (
                                selected_numerator
                                + ratio_alpha * proxy_tail_numerator
                            ) / (
                                selected_partition
                                + ratio_alpha * proxy_tail_partition
                            ).clamp_min(1.0e-20)
                            append_row(
                                rows,
                                identity,
                                "sample_ratio_alpha",
                                ratio_output,
                                full_output,
                                sample_count=sample_count,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=(
                                    ratio_alpha * proxy_tail_partition
                                ),
                                alpha=ratio_alpha,
                            )

                            cv_tail_numerator = proxy_tail_numerator + expansion * torch.sum(
                                weight_difference.unsqueeze(-1) * sampled_values,
                                dim=0,
                            )
                            cv_output = (
                                selected_numerator + cv_tail_numerator
                            ) / (
                                selected_partition + cv_tail_partition
                            ).clamp_min(1.0e-20)
                            append_row(
                                rows,
                                identity,
                                "sample_score_control_variate",
                                cv_output,
                                full_output,
                                sample_count=sample_count,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=cv_tail_partition,
                                alpha=float(
                                    cv_tail_partition
                                    / proxy_tail_partition.clamp_min(1.0e-20)
                                ),
                            )

                            direct_tail_partition = (
                                expansion * sampled_exact_weights.sum()
                            )
                            direct_tail_numerator = expansion * torch.sum(
                                sampled_exact_weights.unsqueeze(-1)
                                * sampled_true_values,
                                dim=0,
                            )
                            direct_output = (
                                selected_numerator + direct_tail_numerator
                            ) / (
                                selected_partition + direct_tail_partition
                            ).clamp_min(1.0e-20)
                            append_row(
                                rows,
                                identity,
                                "sample_direct_horvitz_thompson",
                                direct_output,
                                full_output,
                                sample_count=sample_count,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=direct_tail_partition,
                                alpha=float(
                                    direct_tail_partition
                                    / proxy_tail_partition.clamp_min(1.0e-20)
                                ),
                            )

                            joint_residual = (
                                sampled_exact_weights.unsqueeze(-1)
                                * sampled_true_values
                                - sampled_proxy_weights.unsqueeze(-1)
                                * sampled_values
                            )
                            joint_cv_tail_numerator = (
                                proxy_tail_numerator
                                + expansion * joint_residual.sum(dim=0)
                            )
                            joint_cv_output = (
                                selected_numerator + joint_cv_tail_numerator
                            ) / (
                                selected_partition + cv_tail_partition
                            ).clamp_min(1.0e-20)
                            append_row(
                                rows,
                                identity,
                                "sample_joint_control_variate",
                                joint_cv_output,
                                full_output,
                                sample_count=sample_count,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=cv_tail_partition,
                                alpha=float(
                                    cv_tail_partition
                                    / proxy_tail_partition.clamp_min(1.0e-20)
                                ),
                            )

                            sample_proxy_scores = proxy_scores[tail_sample_indices]
                            sample_exact_scores = exact_scores[tail_sample_indices]
                            proxy_centered = (
                                sample_proxy_scores - sample_proxy_scores.mean()
                            )
                            exact_centered = (
                                sample_exact_scores - sample_exact_scores.mean()
                            )
                            slope = (
                                (proxy_centered * exact_centered).mean()
                                / proxy_centered.square().mean().clamp_min(1.0e-8)
                            ).clamp_min(0.0)
                            intercept = (
                                sample_exact_scores.mean()
                                - slope * sample_proxy_scores.mean()
                            )
                            calibrated_tail_scores = (
                                slope * proxy_scores[tail] + intercept
                            )
                            affine_output = normalized_output(
                                selected_exact_scores,
                                selected_values,
                                calibrated_tail_scores,
                                reconstructed_value[tail],
                            )
                            estimated_affine_tail_partition = torch.exp(
                                calibrated_tail_scores - common_center
                            ).sum()
                            affine_residual = (
                                sample_exact_scores
                                - (slope * sample_proxy_scores + intercept)
                            )
                            append_row(
                                rows,
                                identity,
                                "sample_affine_scores",
                                affine_output,
                                full_output,
                                sample_count=sample_count,
                                true_tail_partition=exact_tail_partition,
                                estimated_tail_partition=(
                                    estimated_affine_tail_partition
                                ),
                                alpha=float(
                                    estimated_affine_tail_partition
                                    / proxy_tail_partition.clamp_min(1.0e-20)
                                ),
                                affine_slope=float(slope),
                                affine_residual_std=float(
                                    affine_residual.std(unbiased=False)
                                ),
                            )

            print(
                json.dumps(
                    {
                        "trace": trace_path.stem,
                        "layer": layer,
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
            del query, key, value
            torch.cuda.empty_cache()

    query_crossfit_rows = append_query_crossfit_conditional_rows(rows)
    print(
        json.dumps({"query_crossfit_rows": query_crossfit_rows}),
        flush=True,
    )
    layer_rows = projected_layer_rows(rows, device)
    for row in rows:
        del row["_output_tensor"]
        del row["_full_output_tensor"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if layer_rows:
        with (args.output_dir / "per_layer_output.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["history_tokens"],
                row["candidate_mode"],
                row["method"],
                row["sample_count"],
                row["block_size"],
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for identity, items in sorted(grouped.items()):
        history_tokens, candidate_mode, method, sample_count, block_size = identity
        result: dict[str, Any] = {
            "history_tokens": history_tokens,
            "candidate_mode": candidate_mode,
            "method": method,
            "sample_count": sample_count,
            "block_size": block_size,
            "cases": len(items),
        }
        for metric in (
            "selected_mass",
            "tail_partition_relative_error",
            "relative_l2",
            "cosine",
            "alpha",
            "affine_slope",
            "affine_residual_std",
            "absolute_l2",
            "full_output_l2",
            "residual_risk_absolute",
            "residual_risk_relative",
            "residual_risk_range_absolute",
            "residual_risk_bernstein_absolute",
            "residual_risk_bernstein_relative",
            "tail_correction_l2",
            "tail_effective_tokens",
            "proxy_selected_mass",
            "value_explained_variance",
            "conditional_gain",
            "conditional_holdout_error_reduction",
        ):
            finite_values = [
                float(item[metric])
                for item in items
                if math.isfinite(float(item[metric]))
            ]
            if finite_values:
                for statistic, value in summarize(finite_values).items():
                    result[f"{metric}_{statistic}"] = value
        summary_rows.append(result)

    report = {
        "schema": "qksieve_tail_partition_calibration_v1",
        "config": {
            "traces": [str(path) for path in traces],
            "top_k": args.top_k,
            "sample_counts": sample_counts,
            "block_sizes": block_sizes,
            "conditional_dims": conditional_dims,
            "conditional_fit_stride": args.conditional_fit_stride,
            "tail_sampling": args.tail_sampling,
            "key_rate_budget": args.key_rate_budget,
            "value_rank": args.value_rank,
            "value_bits": args.value_bits,
            "block_moment_bits": args.block_moment_bits,
            "value_metric": args.value_metric,
            "risk_delta": args.risk_delta,
            "max_records_per_trace": args.max_records_per_trace,
            "output_projection_rows": len(layer_rows),
        },
        "claim_boundary": (
            "Real-QKV local-output audit only; not a model-level quality or "
            "runtime claim. Oracle rows isolate error sources and are not methods."
        ),
        "summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
