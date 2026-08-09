from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    covariance,
    qk_balanced_factors,
    symmetric_covariance_factors,
)


TRACE_SCHEMA = "qksieve_generation_drift_trace_v1"
TEACHER_FORCED_TRACE_SCHEMA = "qksieve_teacher_forced_drift_trace_v1"
FROZEN_METHOD = "qksieve_fullprompt_auto_plain_fulltopk"
FROZEN_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
)
HEAD_DIM = GROUP_COUNT * GROUP_SIZE
PHYSICAL_INDEX_BITS = 240
DEFAULT_SAMPLE_COUNTS = (1, 4, 8, 16, 32)
DEFAULT_POSITION_BUCKETS = (
    (0, 63, "0000-0063"),
    (64, 255, "0064-0255"),
    (256, 1023, "0256-1023"),
    (1024, 4095, "1024-4095"),
    (4096, None, "4096+"),
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(
        sorted({int(item) for item in value.split(",") if item.strip()})
    )
    if not result or result[0] <= 0:
        raise ValueError("expected positive comma-separated integers")
    return result


def position_bucket(step: int) -> str:
    if step < 0:
        raise ValueError("decode step must be non-negative")
    for lower, upper, name in DEFAULT_POSITION_BUCKETS:
        if step >= lower and (upper is None or step <= upper):
            return name
    raise AssertionError("position bucket table is incomplete")


def direct_target_count(history_tokens: int) -> int:
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def true_top_count(history_tokens: int, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("true_top_fraction must lie in (0, 1]")
    return min(history_tokens, max(1, math.ceil(fraction * history_tokens)))


def physical_index_bits(allocation: Iterable[int]) -> int:
    values = tuple(int(bits) for bits in allocation)
    if len(values) != GROUP_COUNT:
        raise ValueError(f"allocation must contain {GROUP_COUNT} bands")
    if any(bits not in ZERO_BIT_LEVELS for bits in values):
        raise ValueError("allocation contains an unsupported bit level")
    return GROUP_SIZE * (
        sum(values) + sum(int(bits > 0) for bits in values)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value.detach().cpu()


def _normalize_layer_dict(value: Any, name: str) -> dict[int, torch.Tensor]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    output: dict[int, torch.Tensor] = {}
    for layer, tensor in value.items():
        layer_index = int(layer)
        if layer_index in output:
            raise ValueError(f"{name} contains duplicate layer {layer_index}")
        output[layer_index] = _as_tensor(
            tensor,
            f"{name}[{layer_index}]",
        )
    return output


def validate_trace(
    payload: Any,
    *,
    trace_path: Path,
    sample_counts: tuple[int, ...],
    expected_method: str = FROZEN_METHOD,
    expected_score_mode: str = FROZEN_SCORE_MODE,
    expected_shrinkage: float = 0.75,
    allow_experimental_method: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{trace_path}: trace payload must be a dictionary")
    schema = str(payload.get("schema", ""))
    if schema not in {TRACE_SCHEMA, TEACHER_FORCED_TRACE_SCHEMA}:
        raise ValueError(f"{trace_path}: unsupported trace schema")
    trace_kind = (
        "free_generation"
        if schema == TRACE_SCHEMA
        else "teacher_forced_corpus_continuation"
    )
    declared_trace_kind = str(payload.get("trace_kind", trace_kind))
    if declared_trace_kind != trace_kind:
        raise ValueError(f"{trace_path}: trace_kind contradicts schema")
    method = str(payload.get("method", ""))
    score_mode = str(payload.get("score_mode", ""))
    if not allow_experimental_method:
        if method != expected_method:
            raise ValueError(
                f"{trace_path}: expected frozen method {expected_method}, "
                f"got {method}"
            )
        if score_mode != expected_score_mode:
            raise ValueError(
                f"{trace_path}: expected frozen score mode "
                f"{expected_score_mode}, got {score_mode}"
            )
    shrinkage = float(payload.get("qk_metric_query_shrinkage", -1.0))
    if not math.isclose(
        shrinkage,
        expected_shrinkage,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{trace_path}: shrinkage {shrinkage} does not match "
            f"{expected_shrinkage}"
        )
    production_query_count = int(
        payload.get("query_calibration_tokens", 0)
    )
    if production_query_count <= 0:
        raise ValueError(
            f"{trace_path}: query_calibration_tokens must be positive"
        )

    trace_layers = tuple(int(item) for item in payload.get("trace_layers", ()))
    trace_steps = tuple(int(item) for item in payload.get("trace_steps", ()))
    if (
        not trace_layers
        or tuple(sorted(set(trace_layers))) != trace_layers
        or trace_layers[0] < 0
    ):
        raise ValueError(f"{trace_path}: trace_layers must be unique and sorted")
    if (
        not trace_steps
        or tuple(sorted(set(trace_steps))) != trace_steps
        or trace_steps[0] != 0
    ):
        raise ValueError(
            f"{trace_path}: trace_steps must be sorted and start at zero"
        )

    prompt_tail = _normalize_layer_dict(
        payload.get("prefill_query_tail"),
        "prefill_query_tail",
    )
    missing_prompt_tail_layers = set(trace_layers) - set(prompt_tail)
    if missing_prompt_tail_layers:
        raise ValueError(
            f"{trace_path}: prompt-tail data is missing trace layers "
            f"{sorted(missing_prompt_tail_layers)}"
        )
    maximum_sample_count = max(sample_counts)
    recorded_tail_count = int(
        payload.get(
            "recorded_prefill_query_tail_tokens",
            maximum_sample_count,
        )
    )
    if recorded_tail_count < maximum_sample_count:
        raise ValueError(
            f"{trace_path}: declared prompt-tail window is too short"
        )

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{trace_path}: records must be a non-empty list")
    records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"{trace_path}: record {index} is not a dictionary")
        layer = int(record.get("layer", -1))
        if layer not in trace_layers:
            raise ValueError(f"{trace_path}: record uses unregistered layer")
        step = int(record.get("step", -1))
        if step not in trace_steps:
            raise ValueError(f"{trace_path}: record uses unregistered step")
        query = _as_tensor(record.get("query"), f"record[{index}].query")
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError(f"{trace_path}: invalid Query trace shape")
        if query.shape[-2:] != (1, HEAD_DIM):
            raise ValueError(f"{trace_path}: Query must be [1,H,1,128]")
        normalized = dict(record)
        normalized["layer"] = layer
        normalized["step"] = step
        normalized["query"] = query
        if record.get("key") is not None:
            normalized["key"] = _as_tensor(
                record["key"],
                f"record[{index}].key",
            )
        records_by_layer[layer].append(normalized)

    observed_steps: tuple[int, ...] | None = None
    prompt_tokens = int(payload.get("prompt_tokens", 0))
    if prompt_tokens <= 0:
        raise ValueError(f"{trace_path}: prompt_tokens must be positive")
    for layer in trace_layers:
        tail = prompt_tail[layer]
        if (
            tail.ndim != 4
            or tail.shape[0] != 1
            or tail.shape[-1] != HEAD_DIM
            or tail.shape[-2] < maximum_sample_count
        ):
            raise ValueError(
                f"{trace_path}: layer {layer} prompt tail cannot support "
                f"{maximum_sample_count} samples"
            )
        layer_records = records_by_layer.get(layer, [])
        steps = tuple(int(record["step"]) for record in layer_records)
        if (
            not steps
            or steps[0] != 0
            or tuple(sorted(set(steps))) != steps
        ):
            raise ValueError(
                f"{trace_path}: layer {layer} records must be unique, "
                "sorted, and include step zero"
            )
        if observed_steps is None:
            observed_steps = steps
        elif steps != observed_steps:
            raise ValueError(
                f"{trace_path}: traced layers have different step coverage"
            )
        first_key = layer_records[0].get("key")
        if not isinstance(first_key, torch.Tensor):
            raise ValueError(
                f"{trace_path}: layer {layer} step zero must contain Key state"
            )
        if (
            first_key.ndim != 4
            or first_key.shape[0] != 1
            or first_key.shape[-1] != HEAD_DIM
            or first_key.shape[-2] != prompt_tokens + 1
        ):
            raise ValueError(
                f"{trace_path}: layer {layer} Key state must contain "
                "prompt plus current decode token"
            )
        query_heads = int(tail.shape[1])
        key_heads = int(first_key.shape[1])
        if query_heads % key_heads != 0:
            raise ValueError(f"{trace_path}: invalid GQA head ratio")
        for record in layer_records:
            if int(record["query"].shape[1]) != query_heads:
                raise ValueError(
                    f"{trace_path}: Query-head count changes within layer"
                )

    sequence_ids = payload.get(
        "generated_ids" if trace_kind == "free_generation" else "sequence_ids",
        [],
    )
    if isinstance(sequence_ids, torch.Tensor):
        generated_count = int(sequence_ids.numel())
    elif isinstance(sequence_ids, (list, tuple)):
        generated_count = len(sequence_ids)
    else:
        raise ValueError(f"{trace_path}: continuation ids have invalid type")
    if generated_count <= 0:
        raise ValueError(f"{trace_path}: trace has no continuation tokens")
    if max(observed_steps or (0,)) >= generated_count:
        raise ValueError(
            f"{trace_path}: observed step exceeds continuation length"
        )

    return {
        "path": trace_path,
        "payload": payload,
        "method": method,
        "score_mode": score_mode,
        "trace_kind": trace_kind,
        "shrinkage": shrinkage,
        "production_query_count": production_query_count,
        "recorded_tail_count": recorded_tail_count,
        "trace_layers": trace_layers,
        "requested_steps": trace_steps,
        "observed_steps": observed_steps or (),
        "prompt_tail": prompt_tail,
        "records_by_layer": records_by_layer,
        "prompt_tokens": prompt_tokens,
        "generated_count": generated_count,
    }


def _floored_construction_moments(
    sampled_key: torch.Tensor,
    calibration_queries: torch.Tensor,
    query_shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key_moment = covariance(sampled_key.float())
    raw_query_moment = covariance(calibration_queries.float())
    isotropic_scale = raw_query_moment.diagonal().mean()
    query_moment = (
        (1.0 - query_shrinkage) * raw_query_moment
        + query_shrinkage
        * isotropic_scale
        * torch.eye(HEAD_DIM, device=raw_query_moment.device)
    )
    key_sqrt, _ = symmetric_covariance_factors(key_moment)
    query_sqrt, _ = symmetric_covariance_factors(query_moment)
    floored_key = key_sqrt @ key_sqrt
    floored_query = query_sqrt @ query_sqrt
    left, singular_values, right_h = torch.linalg.svd(
        query_sqrt @ key_sqrt,
        full_matrices=False,
    )
    return (
        floored_query,
        floored_key,
        left,
        right_h.transpose(0, 1),
        singular_values,
    )


def _distortion_table(
    sampled_coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    table: list[dict[int, torch.Tensor]] = []
    for group in range(GROUP_COUNT):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = sampled_coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs: dict[int, torch.Tensor] = {}
        for bits in ZERO_BIT_LEVELS:
            residual = key_band - quantize_band(key_band, bits)
            costs[bits] = (
                query_band @ residual.transpose(0, 1)
            ).square().mean()
        table.append(costs)
    return table


def _allocate(
    sampled_coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    total_rate_budget: int,
) -> tuple[tuple[int, ...], list[dict[int, torch.Tensor]]]:
    table = _distortion_table(sampled_coefficients, calibration_queries)
    allocation = allocate_bits(
        table,
        total_rate_budget,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    bits = physical_index_bits(allocation)
    expected = GROUP_SIZE * total_rate_budget
    if bits > expected:
        raise AssertionError(
            f"allocator exceeded the {expected}-bit physical budget with {bits}"
        )
    return allocation, table


def _reconstruct(
    coefficients: torch.Tensor,
    allocation: tuple[int, ...],
) -> torch.Tensor:
    bands = []
    for group, bits in enumerate(allocation):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        bands.append(quantize_band(coefficients[:, start:stop], bits))
    return torch.cat(bands, dim=-1)


def _diag_qmse(
    queries: torch.Tensor,
    exact_keys: torch.Tensor,
    approximate_keys: torch.Tensor,
) -> float:
    total = torch.zeros((), dtype=torch.float32, device=queries.device)
    residual = exact_keys - approximate_keys
    for group in range(GROUP_COUNT):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        errors = (
            queries[:, start:stop]
            @ residual[:, start:stop].transpose(0, 1)
        )
        total = total + errors.square().mean()
    return float(total.item())


def _full_qmse(
    queries: torch.Tensor,
    exact_keys: torch.Tensor,
    approximate_keys: torch.Tensor,
) -> float:
    errors = queries @ (exact_keys - approximate_keys).transpose(0, 1)
    return float(errors.square().mean().item())


def _operator_norm(matrix: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(matrix.float(), ord=2).item())


def _subspace_sine(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    cosines = torch.linalg.svdvals(
        left[:, :rank].transpose(0, 1) @ right[:, :rank]
    )
    minimum = cosines.amin().clamp(0.0, 1.0)
    return float(torch.sqrt((1.0 - minimum.square()).clamp_min(0.0)).item())


def _moment_product_bound(
    reference_query: torch.Tensor,
    current_query: torch.Tensor,
    reference_key: torch.Tensor,
    current_key: torch.Tensor,
) -> float:
    query_error = _operator_norm(current_query - reference_query)
    key_error = _operator_norm(current_key - reference_key)
    query_denominator = (
        math.sqrt(
            max(
                float(torch.linalg.eigvalsh(reference_query).amin().item()),
                1.0e-30,
            )
        )
        + math.sqrt(
            max(
                float(torch.linalg.eigvalsh(current_query).amin().item()),
                1.0e-30,
            )
        )
    )
    key_denominator = (
        math.sqrt(
            max(
                float(torch.linalg.eigvalsh(reference_key).amin().item()),
                1.0e-30,
            )
        )
        + math.sqrt(
            max(
                float(torch.linalg.eigvalsh(current_key).amin().item()),
                1.0e-30,
            )
        )
    )
    query_root_error = query_error / query_denominator
    key_root_error = key_error / key_denominator
    return (
        query_root_error * math.sqrt(_operator_norm(current_key))
        + math.sqrt(_operator_norm(reference_query)) * key_root_error
    )


def _second_moment_drift(
    calibration: torch.Tensor,
    heldout: torch.Tensor,
) -> tuple[float, float, float, float]:
    calibration_moment = covariance(calibration.float())
    heldout_moment = covariance(heldout.float())
    difference = heldout_moment - calibration_moment
    op = _operator_norm(difference)
    fro = float(torch.linalg.matrix_norm(difference, ord="fro").item())
    op_reference = max(_operator_norm(calibration_moment), 1.0e-12)
    fro_reference = max(
        float(torch.linalg.matrix_norm(calibration_moment, ord="fro").item()),
        1.0e-12,
    )
    return op, op / op_reference, fro, fro / fro_reference


def _selection_row(
    query: torch.Tensor,
    projected_query: torch.Tensor,
    exact_key: torch.Tensor,
    approximate_key: torch.Tensor,
    scaling: float,
    true_fraction: float,
) -> dict[str, float | int]:
    exact_scores = (exact_key @ query.float()) * scaling
    proxy_query = query_int8(projected_query.float())
    approximate_scores = (approximate_key @ proxy_query) * scaling
    count = int(exact_scores.numel())
    selected_count = direct_target_count(count)
    true_count = true_top_count(count, true_fraction)
    selected = torch.topk(
        approximate_scores,
        k=selected_count,
        sorted=False,
    ).indices
    oracle_active = torch.topk(
        exact_scores,
        k=selected_count,
        sorted=False,
    ).indices
    true_indices = torch.topk(
        exact_scores,
        k=true_count,
        sorted=False,
    ).indices
    selected_mask = torch.zeros(count, dtype=torch.bool, device=query.device)
    selected_mask[selected] = True
    attention = torch.softmax(exact_scores.float(), dim=0)
    selected_mass = min(
        1.0,
        max(0.0, float(attention[selected].sum().item())),
    )
    oracle_active_mass = min(
        1.0,
        max(0.0, float(attention[oracle_active].sum().item())),
    )
    true_mass = min(
        1.0,
        max(0.0, float(attention[true_indices].sum().item())),
    )
    exact_centered = exact_scores - exact_scores.mean()
    approximate_centered = approximate_scores - approximate_scores.mean()
    denominator = (
        torch.linalg.vector_norm(exact_centered)
        * torch.linalg.vector_norm(approximate_centered)
    )
    pearson = (
        float(
            (
                exact_centered @ approximate_centered / denominator
            ).item()
        )
        if float(denominator.item()) > 0.0
        else 0.0
    )
    error = exact_scores - approximate_scores
    centered_error = error - error.mean()
    rmse = float(error.square().mean().sqrt().item())
    centered_rmse = float(centered_error.square().mean().sqrt().item())
    exact_scale = max(
        float(exact_centered.square().mean().sqrt().item()),
        1.0e-12,
    )
    error_range = float((error.amax() - error.amin()).item())
    top_values = torch.topk(
        exact_scores,
        k=min(count, selected_count + 1),
        sorted=True,
    ).values
    boundary_gap = (
        float((top_values[selected_count - 1] - top_values[selected_count]).item())
        if selected_count < count
        else math.inf
    )
    oracle_omitted = max(0.0, 1.0 - oracle_active_mass)
    proxy_omitted = max(0.0, 1.0 - selected_mass)
    return {
        "history_tokens": count,
        "active_tokens": selected_count,
        "active_fraction": selected_count / count,
        "true_top_tokens": true_count,
        "true_top_fraction": true_count / count,
        "top2_recall": (
            int(selected_mask[true_indices].sum().item()) / true_count
        ),
        "selected_attention_mass": selected_mass,
        "oracle_active_attention_mass": oracle_active_mass,
        "oracle_top2_attention_mass": true_mass,
        "top2_attention_mass_recall": (
            selected_mass / true_mass if true_mass > 0.0 else 0.0
        ),
        "proxy_omitted_mass": proxy_omitted,
        "oracle_omitted_mass": oracle_omitted,
        "omitted_mass_inflation": (
            proxy_omitted / oracle_omitted
            if oracle_omitted > 1.0e-12
            else (1.0 if proxy_omitted <= 1.0e-12 else math.inf)
        ),
        "score_pearson": pearson,
        "score_rmse": rmse,
        "centered_score_rmse": centered_rmse,
        "normalized_centered_score_rmse": centered_rmse / exact_scale,
        "score_error_range": error_range,
        "active_boundary_gap": boundary_gap,
    }


def _trace_identifier(trace: dict[str, Any]) -> str:
    payload = trace["payload"]
    return (
        f"{payload.get('task', '')}::{payload.get('sample_id', '')}::"
        f"{trace['path'].name}"
    )


def analyze_trace(
    trace: dict[str, Any],
    *,
    sample_counts: tuple[int, ...],
    sample_stride: int,
    total_rate_budget: int,
    true_top_fraction: float,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if total_rate_budget <= 0:
        raise ValueError("total_rate_budget must be positive")
    payload = trace["payload"]
    trace_id = _trace_identifier(trace)
    per_query_rows: list[dict[str, Any]] = []
    per_bucket_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    for layer in trace["trace_layers"]:
        tail = trace["prompt_tail"][layer][0].to(device=device).float()
        records = trace["records_by_layer"][layer]
        first_key_state = records[0]["key"][0].to(device=device).float()
        exact_prompt_key = first_key_state[:, :-1, :]
        query_head_count = int(tail.shape[0])
        key_head_count = int(exact_prompt_key.shape[0])
        groups_per_key = query_head_count // key_head_count

        for key_head in range(key_head_count):
            exact_key = exact_prompt_key[key_head]
            sampled_key = exact_key[::sample_stride]
            query_start = key_head * groups_per_key
            query_stop = query_start + groups_per_key
            head_tail = tail[query_start:query_stop]
            transforms: dict[int, dict[str, Any]] = {}
            for sample_count in sample_counts:
                calibration = (
                    head_tail[:, -sample_count:, :]
                    .reshape(-1, HEAD_DIM)
                    .contiguous()
                )
                query_factor, key_factor, singular_values = (
                    qk_balanced_factors(
                        sampled_key,
                        calibration,
                        trace["shrinkage"],
                    )
                )
                projected_sample = sampled_key @ key_factor
                projected_calibration = calibration @ query_factor
                allocation, calibration_table = _allocate(
                    projected_sample,
                    projected_calibration,
                    total_rate_budget,
                )
                (
                    query_moment,
                    key_moment,
                    left,
                    right,
                    construction_singular_values,
                ) = _floored_construction_moments(
                    sampled_key,
                    calibration,
                    trace["shrinkage"],
                )
                ad_error = float(
                    (
                        query_factor @ key_factor.transpose(0, 1)
                        - torch.eye(HEAD_DIM, device=device)
                    )
                    .abs()
                    .amax()
                    .item()
                )
                transforms[sample_count] = {
                    "calibration": calibration,
                    "query_factor": query_factor,
                    "key_factor": key_factor,
                    "singular_values": singular_values,
                    "projected_sample": projected_sample,
                    "projected_calibration": projected_calibration,
                    "allocation": allocation,
                    "calibration_table": calibration_table,
                    "query_moment": query_moment,
                    "key_moment": key_moment,
                    "left": left,
                    "right": right,
                    "construction_singular_values": (
                        construction_singular_values
                    ),
                    "ad_error": ad_error,
                }

            reference_count = max(sample_counts)
            reference = transforms[reference_count]
            for sample_count in sample_counts:
                transform = transforms[sample_count]
                epsilon_m = _moment_product_bound(
                    reference["query_moment"],
                    transform["query_moment"],
                    reference["key_moment"],
                    transform["key_moment"],
                )
                gaps: dict[str, float] = {}
                left_angles: dict[str, float] = {}
                right_angles: dict[str, float] = {}
                certified_bounds: dict[str, float | None] = {}
                reference_singular = reference[
                    "construction_singular_values"
                ]
                for rank in range(GROUP_SIZE, HEAD_DIM, GROUP_SIZE):
                    gap = float(
                        (
                            reference_singular[rank - 1]
                            - reference_singular[rank]
                        ).item()
                    )
                    gaps[str(rank)] = gap
                    left_angles[str(rank)] = _subspace_sine(
                        reference["left"],
                        transform["left"],
                        rank,
                    )
                    right_angles[str(rank)] = _subspace_sine(
                        reference["right"],
                        transform["right"],
                        rank,
                    )
                    certified_bounds[str(rank)] = (
                        2.0 * math.sqrt(2.0) * epsilon_m / gap
                        if gap > 0.0 and epsilon_m < gap / 2.0
                        else None
                    )
                allocation = transform["allocation"]
                allocation_rows.append(
                    {
                        "trace_id": trace_id,
                        "task": str(payload.get("task", "")),
                        "sample_id": str(payload.get("sample_id", "")),
                        "trace_kind": trace["trace_kind"],
                        "layer": layer,
                        "kv_head": key_head,
                        "query_sample_count": sample_count,
                        "reference_query_sample_count": reference_count,
                        "calibration_query_vectors": int(
                            transform["calibration"].shape[0]
                        ),
                        "sampled_key_vectors": int(sampled_key.shape[0]),
                        "allocation": "-".join(map(str, allocation)),
                        "allocated_index_bits": physical_index_bits(allocation),
                        "reserved_index_bits": (
                            GROUP_SIZE * total_rate_budget
                        ),
                        "rate_budget_saturated": (
                            physical_index_bits(allocation)
                            == GROUP_SIZE * total_rate_budget
                        ),
                        "ad_identity_max_abs": transform["ad_error"],
                        "query_moment_op_error_to_reference": _operator_norm(
                            transform["query_moment"]
                            - reference["query_moment"]
                        ),
                        "moment_product_error_bound": epsilon_m,
                        "band_boundary_gaps_json": json.dumps(
                            gaps,
                            sort_keys=True,
                        ),
                        "left_subspace_sines_json": json.dumps(
                            left_angles,
                            sort_keys=True,
                        ),
                        "right_subspace_sines_json": json.dumps(
                            right_angles,
                            sort_keys=True,
                        ),
                        "certified_subspace_bounds_json": json.dumps(
                            certified_bounds,
                            sort_keys=True,
                        ),
                    }
                )

                projected_key = exact_key @ transform["key_factor"]
                approximate_key = _reconstruct(projected_key, allocation)
                heldout_by_bucket: dict[str, list[torch.Tensor]] = defaultdict(
                    list
                )
                for record in records:
                    step = int(record["step"])
                    bucket = position_bucket(step)
                    scaling = float(record.get("scaling", HEAD_DIM**-0.5))
                    query_state = record["query"][0].to(device=device).float()
                    for query_head in range(query_start, query_stop):
                        query = query_state[query_head, 0]
                        projected_query = query @ transform["query_factor"]
                        heldout_by_bucket[bucket].append(query)
                        metrics = _selection_row(
                            query,
                            projected_query,
                            exact_key,
                            approximate_key,
                            scaling,
                            true_top_fraction,
                        )
                        per_query_rows.append(
                            {
                                "trace_id": trace_id,
                                "task": str(payload.get("task", "")),
                                "sample_id": str(
                                    payload.get("sample_id", "")
                                ),
                                "trace_kind": trace["trace_kind"],
                                "layer": layer,
                                "kv_head": key_head,
                                "query_head": query_head,
                                "query_sample_count": sample_count,
                                "production_query_sample_count": trace[
                                    "production_query_count"
                                ],
                                "decode_step": step,
                                "position_bucket": bucket,
                                "scaling": scaling,
                                "allocation": "-".join(map(str, allocation)),
                                "allocated_index_bits": physical_index_bits(
                                    allocation
                                ),
                                "reserved_index_bits": (
                                    GROUP_SIZE * total_rate_budget
                                ),
                                **metrics,
                            }
                        )

                for bucket, heldout_values in sorted(heldout_by_bucket.items()):
                    heldout = torch.stack(heldout_values)
                    heldout_projected = (
                        heldout @ transform["query_factor"]
                    )
                    raw_drift = _second_moment_drift(
                        transform["calibration"],
                        heldout,
                    )
                    projected_drift = _second_moment_drift(
                        transform["projected_calibration"],
                        heldout_projected,
                    )
                    heldout_table = _distortion_table(
                        transform["projected_sample"],
                        heldout_projected,
                    )
                    oracle_allocation = allocate_bits(
                        heldout_table,
                        total_rate_budget,
                        ZERO_BIT_LEVELS,
                        include_scale_metadata=True,
                    )
                    if physical_index_bits(oracle_allocation) > (
                        GROUP_SIZE * total_rate_budget
                    ):
                        raise AssertionError(
                            "held-out oracle exceeded physical rate"
                        )
                    oracle_key = _reconstruct(
                        projected_key,
                        oracle_allocation,
                    )
                    frozen_sampled_diag = sum(
                        float(heldout_table[group][bits].item())
                        for group, bits in enumerate(allocation)
                    )
                    oracle_sampled_diag = sum(
                        float(heldout_table[group][bits].item())
                        for group, bits in enumerate(oracle_allocation)
                    )
                    frozen_prompt_diag = _diag_qmse(
                        heldout_projected,
                        projected_key,
                        approximate_key,
                    )
                    oracle_prompt_diag = _diag_qmse(
                        heldout_projected,
                        projected_key,
                        oracle_key,
                    )
                    per_bucket_rows.append(
                        {
                            "trace_id": trace_id,
                            "task": str(payload.get("task", "")),
                            "sample_id": str(payload.get("sample_id", "")),
                            "trace_kind": trace["trace_kind"],
                            "layer": layer,
                            "kv_head": key_head,
                            "query_sample_count": sample_count,
                            "position_bucket": bucket,
                            "heldout_query_vectors": int(heldout.shape[0]),
                            "allocation": "-".join(map(str, allocation)),
                            "oracle_allocation": "-".join(
                                map(str, oracle_allocation)
                            ),
                            "allocation_band_agreement": sum(
                                int(left == right)
                                for left, right in zip(
                                    allocation,
                                    oracle_allocation,
                                )
                            )
                            / GROUP_COUNT,
                            "raw_covariance_drift_op": raw_drift[0],
                            "raw_covariance_drift_op_relative": raw_drift[1],
                            "raw_covariance_drift_fro": raw_drift[2],
                            "raw_covariance_drift_fro_relative": raw_drift[3],
                            "projected_covariance_drift_op": (
                                projected_drift[0]
                            ),
                            "projected_covariance_drift_op_relative": (
                                projected_drift[1]
                            ),
                            "projected_covariance_drift_fro": (
                                projected_drift[2]
                            ),
                            "projected_covariance_drift_fro_relative": (
                                projected_drift[3]
                            ),
                            "frozen_heldout_sampled_qmse_diag": (
                                frozen_sampled_diag
                            ),
                            "oracle_heldout_sampled_qmse_diag": (
                                oracle_sampled_diag
                            ),
                            "sampled_allocation_regret_diag": max(
                                0.0,
                                frozen_sampled_diag - oracle_sampled_diag,
                            ),
                            "sampled_allocation_regret_ratio": (
                                frozen_sampled_diag / oracle_sampled_diag
                                if oracle_sampled_diag > 0.0
                                else (
                                    1.0
                                    if frozen_sampled_diag == 0.0
                                    else math.inf
                                )
                            ),
                            "frozen_heldout_prompt_qmse_diag": (
                                frozen_prompt_diag
                            ),
                            "oracle_heldout_prompt_qmse_diag": (
                                oracle_prompt_diag
                            ),
                            "frozen_heldout_qmse_full": _full_qmse(
                                heldout_projected,
                                projected_key,
                                approximate_key,
                            ),
                            "oracle_heldout_qmse_full": _full_qmse(
                                heldout_projected,
                                projected_key,
                                oracle_key,
                            ),
                        }
                    )
                del projected_key, approximate_key

    return per_query_rows, per_bucket_rows, allocation_rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        aggregate = dict(zip(group_fields, key))
        aggregate["count"] = len(items)
        for metric in metric_fields:
            finite = [
                float(item[metric])
                for item in items
                if math.isfinite(float(item[metric]))
            ]
            aggregate[metric] = (
                summarize(finite) if finite else {"count": 0}
            )
        output.append(aggregate)
    return output


def analyze_payloads(
    payloads: list[tuple[Path, Any]],
    *,
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS,
    production_query_samples: int = 8,
    sample_stride: int = 32,
    total_rate_budget: int = 15,
    query_shrinkage: float = 0.75,
    true_top_fraction: float = 0.02,
    device: torch.device = torch.device("cpu"),
    allow_experimental_method: bool = False,
) -> dict[str, Any]:
    if production_query_samples not in sample_counts:
        raise ValueError(
            "production_query_samples must occur in query_sample_counts"
        )
    traces = [
        validate_trace(
            payload,
            trace_path=path,
            sample_counts=sample_counts,
            expected_shrinkage=query_shrinkage,
            allow_experimental_method=allow_experimental_method,
        )
        for path, payload in payloads
    ]
    if any(
        trace["production_query_count"] != production_query_samples
        for trace in traces
    ):
        raise ValueError(
            "trace production Query count does not match frozen protocol"
        )

    per_query_rows: list[dict[str, Any]] = []
    per_bucket_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for trace in traces:
        query_rows, bucket_rows, trace_allocations = analyze_trace(
            trace,
            sample_counts=sample_counts,
            sample_stride=sample_stride,
            total_rate_budget=total_rate_budget,
            true_top_fraction=true_top_fraction,
            device=device,
        )
        per_query_rows.extend(query_rows)
        per_bucket_rows.extend(bucket_rows)
        allocation_rows.extend(trace_allocations)

    max_observed_step = max(
        max(trace["observed_steps"]) for trace in traces
    )
    requested_steps = sorted(
        {
            step
            for trace in traces
            for step in trace["requested_steps"]
        }
    )
    observed_steps = sorted(
        {
            step
            for trace in traces
            for step in trace["observed_steps"]
        }
    )
    observed_buckets = sorted(
        {position_bucket(step) for step in observed_steps}
    )
    trace_kinds = sorted({trace["trace_kind"] for trace in traces})
    query_metrics = (
        "top2_recall",
        "selected_attention_mass",
        "oracle_active_attention_mass",
        "proxy_omitted_mass",
        "oracle_omitted_mass",
        "omitted_mass_inflation",
        "score_pearson",
        "centered_score_rmse",
        "normalized_centered_score_rmse",
        "score_error_range",
    )
    bucket_metrics = (
        "raw_covariance_drift_op_relative",
        "projected_covariance_drift_op_relative",
        "allocation_band_agreement",
        "sampled_allocation_regret_ratio",
        "frozen_heldout_sampled_qmse_diag",
        "oracle_heldout_sampled_qmse_diag",
    )
    summary = {
        "schema": "qksieve_query_drift_analysis_v1",
        "protocol": {
            "method": FROZEN_METHOD,
            "score_mode": FROZEN_SCORE_MODE,
            "query_sample_counts": list(sample_counts),
            "production_query_samples": production_query_samples,
            "sample_stride": sample_stride,
            "total_rate_budget": total_rate_budget,
            "reserved_physical_index_bits": GROUP_SIZE * total_rate_budget,
            "query_shrinkage": query_shrinkage,
            "true_top_fraction": true_top_fraction,
            "fixed_prompt_key_reference": True,
            "query_int8_in_selection_metrics": True,
            "exact_kv_after_selection": True,
            "no_rerank_router_recent_sink_or_full_fallback": True,
        },
        "coverage": {
            "trace_count": len(traces),
            "tasks": sorted(
                {str(trace["payload"].get("task", "")) for trace in traces}
            ),
            "requested_steps": requested_steps,
            "observed_steps": observed_steps,
            "max_observed_step": max_observed_step,
            "observed_position_buckets": observed_buckets,
            "generated_tokens_min": min(
                trace["generated_count"] for trace in traces
            ),
            "generated_tokens_max": max(
                trace["generated_count"] for trace in traces
            ),
            "covers_1k_decode_query": max_observed_step >= 1023,
            "covers_2k_decode_query": max_observed_step >= 2047,
            "covers_4k_decode_query": max_observed_step >= 4095,
            "trace_kinds": trace_kinds,
            "contains_free_generation": "free_generation" in trace_kinds,
            "contains_teacher_forced_continuation": (
                "teacher_forced_corpus_continuation" in trace_kinds
            ),
            "coverage_warning": (
                "Long-output coverage is false unless the corresponding "
                "decode step was actually observed before EOS."
            ),
        },
        "counts": {
            "per_query_rows": len(per_query_rows),
            "per_head_bucket_rows": len(per_bucket_rows),
            "allocation_rows": len(allocation_rows),
        },
        "by_query_sample_count": _aggregate_rows(
            per_query_rows,
            ("query_sample_count",),
            query_metrics,
        ),
        "by_query_sample_count_and_position": _aggregate_rows(
            per_query_rows,
            ("query_sample_count", "position_bucket"),
            query_metrics,
        ),
        "drift_by_query_sample_count_and_position": _aggregate_rows(
            per_bucket_rows,
            ("query_sample_count", "position_bucket"),
            bucket_metrics,
        ),
        "limitations": [
            "Later Queries are evaluated against the fixed prompt Key state "
            "captured at decode step zero; newly generated Keys are excluded.",
            "The largest prompt-tail sample is an empirical reference, not a "
            "population covariance oracle.",
            "Free-generation traces do not establish 1K-4K behavior when EOS "
            "occurs before those registered positions.",
        ],
    }
    return {
        "per_query": per_query_rows,
        "per_head_bucket": per_bucket_rows,
        "allocations": allocation_rows,
        "summary": summary,
    }


def resolve_trace_paths(specs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for spec in specs:
        matches = [Path(item) for item in glob.glob(spec, recursive=True)]
        if not matches and Path(spec).is_file():
            matches = [Path(spec)]
        for path in matches:
            if path.is_file():
                paths.add(path.resolve())
    if not paths:
        raise ValueError("no QKSieve trace files matched --trace")
    return sorted(paths)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    source_paths = (
        project_root / "src/analyze_qksieve_query_drift_20260728.py",
        project_root / "src/collect_qksieve_teacher_forced_drift_20260728.py",
        project_root / "src/run_sample_calibrated_longbench_20260717.py",
        project_root / "src/run_head_top2_targeted_ppl_20260714.py",
        project_root / "src/analyze_qk_balanced_spectral_rate_20260727.py",
        project_root
        / "src/analyze_automatic_spectral_rate_allocation_20260727.py",
        project_root
        / "src/analyze_hierarchical_spectral_quantization_20260727.py",
    )
    return {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Trace file or glob; repeat for multiple inputs.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--query_sample_counts",
        default="1,4,8,16,32",
    )
    parser.add_argument("--production_query_samples", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--total_rate_budget", type=int, default=15)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--true_top_fraction", type=float, default=0.02)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--allow_experimental_method", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_counts = parse_positive_ints(args.query_sample_counts)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = resolve_trace_paths(args.trace)
    payloads = [
        (
            path,
            torch.load(path, map_location="cpu", weights_only=False),
        )
        for path in paths
    ]
    result = analyze_payloads(
        payloads,
        sample_counts=sample_counts,
        production_query_samples=args.production_query_samples,
        sample_stride=args.sample_stride,
        total_rate_budget=args.total_rate_budget,
        query_shrinkage=args.query_shrinkage,
        true_top_fraction=args.true_top_fraction,
        device=device,
        allow_experimental_method=args.allow_experimental_method,
    )
    result["summary"]["source_sha256"] = source_hashes()
    result["summary"]["trace_sha256"] = {
        str(path): sha256(path) for path in paths
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", result["per_query"])
    write_csv(
        args.output_dir / "per_head_bucket.csv",
        result["per_head_bucket"],
    )
    write_csv(
        args.output_dir / "allocations.csv",
        result["allocations"],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
