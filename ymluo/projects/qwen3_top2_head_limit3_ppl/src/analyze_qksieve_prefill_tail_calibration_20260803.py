#!/usr/bin/env python
"""Test whether prefill Queries can calibrate QKSieve's Value-tail weight.

The experiment is deliberately local: it uses real post-RoPE Q/K/V tensors,
fits closed-form coefficients only on the final prefill Queries, and evaluates
them on later decode Queries from the same request.  It is not a model-level
PPL or runtime benchmark.
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
    distortion_table,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)
from analyze_qksieve_tail_partition_calibration_20260803 import (
    load_output_projection,
    metric_value_basis,
)
from analyze_qksieve_value_sketch_residual_20260801 import (
    block_affine_quantize,
    value_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=1280)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--value_rank", type=int, default=16)
    parser.add_argument("--value_bits", type=int, default=4)
    parser.add_argument("--value_scale_block", type=int, default=256)
    parser.add_argument(
        "--value_metrics",
        default="raw,wo_group",
        help="Comma-separated Value basis objectives: raw,wo_group.",
    )
    parser.add_argument(
        "--candidate_modes",
        default="exact,proxy",
        help="Comma-separated candidate selectors: exact,proxy.",
    )
    parser.add_argument(
        "--calibration_counts",
        default="1,2,4,8",
        help="Numbers of final-prefill Queries used by the closed-form fit.",
    )
    return parser.parse_args()


def parse_choices(specification: str, allowed: set[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(x.strip() for x in specification.split(",") if x.strip()))
    if not values or not set(values) <= allowed:
        raise ValueError(f"expected non-empty subset of {sorted(allowed)}, got {values}")
    return values


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def metric_basis(
    value: torch.Tensor,
    gram: torch.Tensor,
    rank: int,
    sample_stride: int,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if metric == "raw":
        return value_basis(
            value,
            sample_stride=sample_stride,
            maximum_rank=rank,
        )
    return metric_value_basis(
        value,
        gram,
        sample_stride=sample_stride,
        maximum_rank=rank,
    )


def output_group_gram(
    projection: torch.Tensor,
    kv_head: int,
    query_groups: int,
    head_dimension: int,
) -> torch.Tensor:
    gram = torch.zeros(
        head_dimension,
        head_dimension,
        dtype=torch.float32,
        device=projection.device,
    )
    for group in range(query_groups):
        query_head = kv_head * query_groups + group
        start = query_head * head_dimension
        block = projection[:, start : start + head_dimension]
        gram.add_(block.T @ block)
    return gram


def quantized_log_risk(log_risk: torch.Tensor, bits: int) -> torch.Tensor:
    levels = float((1 << bits) - 1)
    lower = log_risk.amin()
    upper = log_risk.amax()
    scale = ((upper - lower) / levels).clamp_min(1.0e-12)
    codes = torch.round((log_risk - lower) / scale).clamp(0.0, levels)
    return codes * scale + lower


def approximate_outputs(
    queries: torch.Tensor,
    prefix_counts: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    reconstructed_value: torch.Tensor,
    query_factor: torch.Tensor,
    scaling: float,
    top_k: int,
    candidate_modes: tuple[str, ...],
    residual_log_priorities: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return selected-only, correction, and Full outputs for each Query."""

    exact_scores = queries.float() @ key.float().T * scaling
    projected_queries = queries.float() @ query_factor
    approximate_queries = torch.stack(
        [query_int8(projected_query) for projected_query in projected_queries],
        dim=0,
    )
    proxy_scores = approximate_queries.float() @ reconstructed_key.float().T * scaling
    positions = torch.arange(key.shape[0], device=key.device)
    valid = positions[None, :] < prefix_counts[:, None]
    exact_scores = exact_scores.masked_fill(~valid, -torch.inf)
    proxy_scores = proxy_scores.masked_fill(~valid, -torch.inf)
    maximum = torch.maximum(
        exact_scores.amax(dim=-1), proxy_scores.amax(dim=-1)
    )
    exact_weights = torch.exp(exact_scores - maximum[:, None])
    proxy_weights = torch.exp(proxy_scores - maximum[:, None])
    full_output = (
        exact_weights @ value.float()
        / exact_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    )
    proxy_numerator = proxy_weights @ reconstructed_value.float()
    proxy_partition = proxy_weights.sum(dim=-1)
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for mode in candidate_modes:
        if mode == "exact":
            ranking_scores = exact_scores
        elif mode == "proxy":
            ranking_scores = proxy_scores
        elif mode == "residualfp_exact":
            ranking_scores = exact_scores + residual_log_priorities["fp"][None, :]
        elif mode.startswith("residual") and mode.endswith("_proxy"):
            precision = mode[len("residual") : -len("_proxy")]
            ranking_scores = (
                proxy_scores + residual_log_priorities[precision][None, :]
            )
        else:
            raise ValueError(f"unsupported candidate mode: {mode}")
        active_top_k = min(top_k, int(prefix_counts.min()))
        indices = torch.topk(
            ranking_scores, k=active_top_k, dim=-1, sorted=False
        ).indices
        selected_exact_weights = exact_weights.gather(1, indices)
        selected_proxy_weights = proxy_weights.gather(1, indices)
        selected_values = value.float()[indices]
        selected_reconstructed_values = reconstructed_value.float()[indices]
        selected_numerator = torch.sum(
            selected_exact_weights[..., None] * selected_values, dim=1
        )
        selected_partition = selected_exact_weights.sum(dim=-1)
        proxy_selected_numerator = torch.sum(
            selected_proxy_weights[..., None] * selected_reconstructed_values,
            dim=1,
        )
        proxy_selected_partition = selected_proxy_weights.sum(dim=-1)
        selected_only = selected_numerator / selected_partition[:, None].clamp_min(
            1.0e-20
        )
        tail_numerator = proxy_numerator - proxy_selected_numerator
        tail_partition = proxy_partition - proxy_selected_partition
        tail_complete = (selected_numerator + tail_numerator) / (
            selected_partition + tail_partition
        )[:, None].clamp_min(1.0e-20)
        outputs[mode] = (
            selected_only,
            tail_complete - selected_only,
            full_output,
        )
    return outputs


def projected_components(
    projection: torch.Tensor,
    selected: torch.Tensor,
    correction: torch.Tensor,
    full: torch.Tensor,
    kv_head_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_count, query_head_count, head_dimension = selected.shape
    query_groups = query_head_count // kv_head_count
    selected_error = (selected - full).reshape(query_count, -1)
    projected_error = selected_error @ projection.T
    projected_correction = correction.reshape(query_count, -1) @ projection.T
    group_corrections = []
    for kv_head in range(kv_head_count):
        masked = torch.zeros_like(correction)
        start = kv_head * query_groups
        masked[:, start : start + query_groups] = correction[
            :, start : start + query_groups
        ]
        group_corrections.append(masked.reshape(query_count, -1) @ projection.T)
    return projected_error, projected_correction, torch.stack(group_corrections, dim=1)


def scalar_fit(error: torch.Tensor, correction: torch.Tensor) -> float:
    denominator = correction.square().sum().clamp_min(1.0e-20)
    alpha = -torch.sum(error * correction) / denominator
    return float(alpha.clamp(0.0, 1.0))


def vector_fit(error: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
    # corrections: [samples, groups, hidden]
    gram = torch.einsum("sgh,skh->gk", corrections, corrections)
    right = -torch.einsum("sh,sgh->g", error, corrections)
    regularizer = gram.diagonal().mean().clamp_min(1.0e-12) * 1.0e-4
    coefficients = torch.linalg.solve(
        gram
        + regularizer
        * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
        right,
    )
    return coefficients.clamp(0.0, 1.0)


def relative_errors(
    error: torch.Tensor,
    full_projected: torch.Tensor,
    correction: torch.Tensor | None = None,
    alpha: float = 0.0,
    group_corrections: torch.Tensor | None = None,
    group_alpha: torch.Tensor | None = None,
) -> list[float]:
    active = error
    if correction is not None:
        active = active + float(alpha) * correction
    if group_corrections is not None and group_alpha is not None:
        active = active + torch.einsum("sgh,g->sh", group_corrections, group_alpha)
    denominator = torch.linalg.vector_norm(full_projected, dim=-1).clamp_min(1.0e-12)
    return (torch.linalg.vector_norm(active, dim=-1) / denominator).cpu().tolist()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.value_rank <= 0:
        raise ValueError("top_k and value_rank must be positive")
    value_metrics = parse_choices(args.value_metrics, {"raw", "wo_group"})
    candidate_modes = parse_choices(
        args.candidate_modes,
        {
            "exact",
            "proxy",
            "residualfp_exact",
            "residualfp_proxy",
            "residual8_proxy",
            "residual4_proxy",
        },
    )
    calibration_counts = tuple(
        sorted({int(x) for x in args.calibration_counts.split(",") if x.strip()})
    )
    if not calibration_counts or calibration_counts[0] <= 0:
        raise ValueError("calibration_counts must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace, map_location="cpu", weights_only=False, mmap=True)
    prefill_queries_by_layer = payload.get("prefill_queries", {})
    if not prefill_queries_by_layer:
        raise ValueError(
            "trace has no prefill_queries; recapture it with "
            "--prefill_query_tail_tokens"
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
    alpha_rows: list[dict[str, Any]] = []

    for layer in sorted(records_by_layer):
        if layer not in state_by_layer or layer not in prefill_queries_by_layer:
            continue
        records = sorted(records_by_layer[layer], key=lambda x: int(x["step"]))
        state_record = state_by_layer[layer]
        key_all = state_record["key"].to(device).float()[0]
        value_all = state_record["value"].to(device).float()[0]
        scaling = float(state_record["scaling"])
        prefill_query = prefill_queries_by_layer[layer].to(device).float()[0]
        prefill_query = prefill_query.permute(1, 0, 2).contiguous()
        decode_query = torch.stack(
            [record["query"].to(device).float()[0, :, 0, :] for record in records],
            dim=0,
        )
        query_head_count = int(prefill_query.shape[1])
        kv_head_count = int(key_all.shape[0])
        query_groups = query_head_count // kv_head_count
        history_count = int(key_all.shape[1])
        prefill_count = int(prefill_query.shape[0])
        prefill_prefix = torch.arange(
            history_count - prefill_count + 1,
            history_count + 1,
            device=device,
            dtype=torch.long,
        )
        decode_prefix = torch.full(
            (decode_query.shape[0],), history_count, device=device, dtype=torch.long
        )
        projection = load_output_projection(model_root, layer, device)

        for value_metric in value_metrics:
            stores: dict[str, dict[str, list[torch.Tensor]]] = {
                split: {
                    mode: []
                    for mode in candidate_modes
                }
                for split in ("prefill", "decode")
            }
            for kv_head in range(kv_head_count):
                key = key_all[kv_head]
                value = value_all[kv_head]
                head_slice = slice(
                    kv_head * query_groups, (kv_head + 1) * query_groups
                )
                calibration_queries = prefill_query[:, head_slice].reshape(-1, key.shape[-1])
                query_factor, key_factor, _ = qk_balanced_factors(
                    key[:: args.key_sample_stride],
                    calibration_queries,
                    args.query_shrinkage,
                )
                key_coordinates = key @ key_factor
                projected_calibration = calibration_queries @ query_factor
                bands = quantized_bands(key_coordinates, projected_calibration)
                key_distortion, _ = distortion_table(
                    key_coordinates, projected_calibration, ZERO_BIT_LEVELS
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.key_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                reconstructed_key = reconstruct(bands, allocation)
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
                    value_metric,
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
                log_risk = torch.log(residual_risk)
                residual_log_priorities = {
                    "fp": log_risk,
                    "8": quantized_log_risk(log_risk, 8),
                    "4": quantized_log_risk(log_risk, 4),
                }
                for split, queries, prefixes in (
                    ("prefill", prefill_query[:, head_slice], prefill_prefix),
                    ("decode", decode_query[:, head_slice], decode_prefix),
                ):
                    flat_queries = queries.reshape(-1, queries.shape[-1])
                    flat_prefixes = prefixes.repeat_interleave(query_groups)
                    outputs = approximate_outputs(
                        flat_queries,
                        flat_prefixes,
                        key,
                        value,
                        reconstructed_key,
                        reconstructed_value,
                        query_factor,
                        scaling,
                        args.top_k,
                        candidate_modes,
                        residual_log_priorities,
                    )
                    for mode, tensors in outputs.items():
                        stores[split][mode].append(
                            tuple(
                                tensor.reshape(
                                    queries.shape[0], query_groups, -1
                                )
                                for tensor in tensors
                            )
                        )

            for candidate_mode in candidate_modes:
                components: dict[str, tuple[torch.Tensor, ...]] = {}
                for split in ("prefill", "decode"):
                    head_chunks = stores[split][candidate_mode]
                    selected = torch.cat([chunk[0] for chunk in head_chunks], dim=1)
                    correction = torch.cat([chunk[1] for chunk in head_chunks], dim=1)
                    full = torch.cat([chunk[2] for chunk in head_chunks], dim=1)
                    error, projected_correction, group_correction = projected_components(
                        projection,
                        selected,
                        correction,
                        full,
                        kv_head_count,
                    )
                    full_projected = full.reshape(full.shape[0], -1) @ projection.T
                    components[split] = (
                        error,
                        projected_correction,
                        group_correction,
                        full_projected,
                    )
                test_error, test_correction, test_groups, test_full = components["decode"]
                for method, alpha in (
                    ("selected_only", 0.0),
                    ("fixed_alpha_0.5", 0.5),
                    ("fixed_alpha_1", 1.0),
                ):
                    for step, relative in enumerate(
                        relative_errors(
                            test_error,
                            test_full,
                            correction=test_correction,
                            alpha=alpha,
                        )
                    ):
                        detail_rows.append(
                            {
                                "topic": topic,
                                "layer": layer,
                                "step": step,
                                "value_metric": value_metric,
                                "candidate_mode": candidate_mode,
                                "method": method,
                                "calibration_count": 0,
                                "relative_l2": relative,
                            }
                        )
                calibration_error, calibration_correction, calibration_groups, _ = components[
                    "prefill"
                ]
                for requested_count in calibration_counts:
                    count = min(requested_count, calibration_error.shape[0])
                    active_slice = slice(calibration_error.shape[0] - count, None)
                    layer_alpha = scalar_fit(
                        calibration_error[active_slice],
                        calibration_correction[active_slice],
                    )
                    group_alpha = vector_fit(
                        calibration_error[active_slice],
                        calibration_groups[active_slice],
                    )
                    alpha_rows.append(
                        {
                            "topic": topic,
                            "layer": layer,
                            "value_metric": value_metric,
                            "candidate_mode": candidate_mode,
                            "calibration_count": count,
                            "layer_alpha": layer_alpha,
                            "kv_alpha_mean": float(group_alpha.mean()),
                            "kv_alpha_min": float(group_alpha.min()),
                            "kv_alpha_max": float(group_alpha.max()),
                        }
                    )
                    methods = {
                        f"prefill_layer_scalar_t{count}": relative_errors(
                            test_error,
                            test_full,
                            correction=test_correction,
                            alpha=layer_alpha,
                        ),
                        f"prefill_kv_vector_t{count}": relative_errors(
                            test_error,
                            test_full,
                            group_corrections=test_groups,
                            group_alpha=group_alpha,
                        ),
                    }
                    for method, values in methods.items():
                        for step, relative in enumerate(values):
                            detail_rows.append(
                                {
                                    "topic": topic,
                                    "layer": layer,
                                    "step": step,
                                    "value_metric": value_metric,
                                    "candidate_mode": candidate_mode,
                                    "method": method,
                                    "calibration_count": count,
                                    "relative_l2": relative,
                                }
                            )
                oracle_layer_alpha = scalar_fit(test_error, test_correction)
                oracle_group_alpha = vector_fit(test_error, test_groups)
                for method, values in {
                    "decode_oracle_layer_scalar": relative_errors(
                        test_error,
                        test_full,
                        correction=test_correction,
                        alpha=oracle_layer_alpha,
                    ),
                    "decode_oracle_kv_vector": relative_errors(
                        test_error,
                        test_full,
                        group_corrections=test_groups,
                        group_alpha=oracle_group_alpha,
                    ),
                }.items():
                    for step, relative in enumerate(values):
                        detail_rows.append(
                            {
                                "topic": topic,
                                "layer": layer,
                                "step": step,
                                "value_metric": value_metric,
                                "candidate_mode": candidate_mode,
                                "method": method,
                                "calibration_count": -1,
                                "relative_l2": relative,
                            }
                        )
        del key_all, value_all, projection
        torch.cuda.empty_cache()
        print(json.dumps({"topic": topic, "layer": layer}), flush=True)

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["value_metric"], row["candidate_mode"], row["method"])].append(
            float(row["relative_l2"])
        )
    summary_rows = []
    for (value_metric, candidate_mode, method), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "value_metric": value_metric,
                "candidate_mode": candidate_mode,
                "method": method,
                "cases": len(values),
                **{f"relative_l2_{key}": value for key, value in quantiles(values).items()},
            }
        )
    report = {
        "schema": "qksieve_prefill_tail_calibration_v1",
        "trace": str(args.trace),
        "topic": topic,
        "model_name_or_path": model_root,
        "config": {
            "top_k": args.top_k,
            "key_sample_stride": args.key_sample_stride,
            "query_shrinkage": args.query_shrinkage,
            "key_rate_budget": args.key_rate_budget,
            "value_rank": args.value_rank,
            "value_bits": args.value_bits,
            "value_metrics": value_metrics,
            "candidate_modes": candidate_modes,
            "calibration_counts": calibration_counts,
            "layers": sorted(records_by_layer),
            "decode_steps": max(len(rows) for rows in records_by_layer.values()),
        },
        "algorithm": (
            "Fit clipped least-squares Value-tail coefficients on final prefill "
            "Queries after W_o, then hold them fixed for later decode Queries."
        ),
        "claim_boundary": (
            "Real-QKV local-output temporal-transfer audit; decode-oracle rows are "
            "diagnostics and no row is a model-level PPL or speed claim."
        ),
        "summary": summary_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_case.csv", detail_rows), ("alphas.csv", alpha_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
