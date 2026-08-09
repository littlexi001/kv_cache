from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from analyze_bandef_numerics import (
    gaussian_tail_density_crossings_from_top_values,
)
from evaluate_spectral_error_feedback import (
    record_top2_output_quality,
    select_energy_band,
    update_query_state,
)


def grouped_scores(
    key: torch.Tensor, query: torch.Tensor, group_size: int
) -> torch.Tensor:
    kv_heads, tokens, key_dimensions = key.shape
    if query.ndim == 3:
        steps, query_heads, dimensions = query.shape
        grouped_query = query.reshape(steps, kv_heads, group_size, dimensions)
    elif query.ndim == 4:
        steps, grouped_heads, grouped_size, dimensions = query.shape
        query_heads = grouped_heads * grouped_size
        grouped_query = query
    else:
        raise ValueError("query must have three or four dimensions")
    if (
        dimensions != key_dimensions
        or query_heads != kv_heads * group_size
        or grouped_query.shape[1:3] != (kv_heads, group_size)
    ):
        raise ValueError("incompatible grouped-query shapes")
    return torch.einsum("thgd,hnd->thgn", grouped_query, key).reshape(
        steps, query_heads, tokens
    )


def quantize_per_token_int4(value: torch.Tensor) -> torch.Tensor:
    scale = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    return torch.round(value.float() / scale).clamp(-7, 7) * scale


def quantize_per_token_int8(value: torch.Tensor) -> torch.Tensor:
    scale = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    return torch.round(value.float() / scale).clamp(-127, 127) * scale


def quantize_per_band_int4(value: torch.Tensor, band_size: int) -> torch.Tensor:
    if value.shape[-1] % band_size:
        raise ValueError("the dimension must be divisible by band_size")
    bands = value.float().reshape(*value.shape[:-1], -1, band_size)
    scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    return (torch.round(bands / scale).clamp(-7, 7) * scale).flatten(-2)


def quantize_per_band_logscale_int4(
    value: torch.Tensor, band_size: int, exponent_step: float
) -> torch.Tensor:
    if value.shape[-1] % band_size:
        raise ValueError("the dimension must be divisible by band_size")
    bands = value.float().reshape(*value.shape[:-1], -1, band_size)
    exact_scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    base_scale = exact_scale.amax(dim=-2, keepdim=True)
    exponent = torch.round(
        torch.log2(base_scale / exact_scale).clamp_min(0.0) / exponent_step
    ).clamp(0, 15)
    scale = base_scale * torch.exp2(-exponent_step * exponent)
    return (torch.round(bands / scale).clamp(-7, 7) * scale).flatten(-2)


def normalized_hadamard(value: torch.Tensor) -> torch.Tensor:
    dimensions = value.shape[-1]
    if dimensions <= 0 or dimensions & (dimensions - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    output = value.float().clone()
    width = 1
    while width < dimensions:
        grouped = output.reshape(*output.shape[:-1], -1, 2, width)
        left = grouped[..., 0, :].clone()
        right = grouped[..., 1, :].clone()
        grouped[..., 0, :] = left + right
        grouped[..., 1, :] = left - right
        width *= 2
    return output / math.sqrt(dimensions)


def block_hadamard(value: torch.Tensor, band_size: int) -> torch.Tensor:
    if value.shape[-1] % band_size:
        raise ValueError("the dimension must be divisible by band_size")
    bands = value.reshape(*value.shape[:-1], -1, band_size)
    return normalized_hadamard(bands).flatten(-2)


def candidate_recall(
    approximate_scores: torch.Tensor,
    exact_top: torch.Tensor,
    candidate_count: int,
) -> torch.Tensor:
    candidate = torch.topk(
        approximate_scores, candidate_count, dim=-1, sorted=False
    ).indices
    candidate = candidate.sort(dim=-1).values
    locations = torch.searchsorted(candidate, exact_top)
    locations = locations.clamp_max(candidate_count - 1)
    found = torch.gather(candidate, -1, locations) == exact_top
    return found.float().mean(dim=-1)


def summarize(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p50": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "minimum": float(tensor.min()),
        "count": int(tensor.numel()),
    }


def dynamic_one_shot_scores(
    projected_key: torch.Tensor,
    projected_query: torch.Tensor,
    spectral_weights: torch.Tensor,
    *,
    keep_count: int,
    candidate_count: int,
    band_size: int,
) -> tuple[torch.Tensor, list[float]]:
    steps, kv_heads, group_size, dimensions = projected_query.shape
    band_count = dimensions // band_size
    states = projected_query[0].clone()
    step_scores: list[torch.Tensor] = []
    scanned_dimensions: list[float] = []
    allowed_misses = 0.05 * keep_count
    for step in range(1, steps):
        scores_by_head: list[torch.Tensor] = []
        for kv_head in range(kv_heads):
            current = projected_query[step, kv_head]
            residual = current - states[kv_head]
            band = select_energy_band(
                residual, spectral_weights[kv_head], band_size
            )
            states[kv_head] = update_query_state(
                states[kv_head], current, band, band_size
            )
            scan_count = 1
            first_scores = states[kv_head] @ projected_key[kv_head].T
            top_values = torch.topk(
                first_scores, k=candidate_count, dim=-1
            ).values
            while scan_count < band_count:
                remaining = current - states[kv_head]
                sigma = torch.sqrt(
                    (
                        remaining.square()
                        * spectral_weights[kv_head].unsqueeze(0)
                    ).sum(dim=-1)
                )
                crossings = gaussian_tail_density_crossings_from_top_values(
                    top_values, sigma, keep_count, projected_key.shape[1]
                )
                if bool((crossings <= allowed_misses).all().item()):
                    break
                next_band = select_energy_band(
                    remaining, spectral_weights[kv_head], band_size
                )
                states[kv_head] = update_query_state(
                    states[kv_head], current, next_band, band_size
                )
                scan_count += 1
            scores_by_head.append(states[kv_head] @ projected_key[kv_head].T)
            scanned_dimensions.extend(
                [float(scan_count * band_size)] * group_size
            )
        step_scores.append(torch.cat(scores_by_head, dim=0))
    return torch.stack(step_scores), scanned_dimensions


def record_output_batch(
    metrics: dict[str, dict[str, list[float]]],
    method: str,
    approximate_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    exact_top: torch.Tensor,
    value: torch.Tensor,
    *,
    candidate_count: int,
    keep_count: int,
    group_size: int,
    scaling: float,
) -> None:
    candidates = torch.topk(
        approximate_scores, candidate_count, dim=-1, sorted=False
    ).indices
    for step in range(approximate_scores.shape[0]):
        for head in range(approximate_scores.shape[1]):
            record_top2_output_quality(
                metrics,
                method,
                candidates[step, head],
                exact_scores[step, head],
                exact_top[step, head],
                value[head // group_size],
                keep_count,
                scaling,
            )


def evaluate_trace(
    path: Path,
    *,
    projection_dim: int,
    fixed_rank: int,
    candidate_fraction: float,
    band_size: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    recall: dict[str, list[float]] = defaultdict(list)
    score_error: dict[str, list[float]] = defaultdict(list)
    key_error: dict[str, list[float]] = defaultdict(list)
    scan_dimensions: dict[str, list[float]] = defaultdict(list)
    output_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        if len(records) < 2:
            continue
        key_record = next((row for row in records if row.get("key") is not None), None)
        if key_record is None:
            continue
        key = key_record["key"].to(device).float()[0]
        value = key_record["value"].to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        value = value[:, :history_count]
        queries = torch.stack(
            [row["query"].to(device).float()[0, :, 0] for row in records]
        )
        kv_heads = int(key.shape[0])
        query_heads = int(queries.shape[1])
        group_size = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))
        candidate_count = max(
            keep_count, math.ceil(candidate_fraction * history_count)
        )

        exact_scores = grouped_scores(key, queries, group_size)
        exact_top = torch.topk(
            exact_scores[1:], keep_count, dim=-1, sorted=False
        ).indices
        sampled_key = key[:, ::32]
        second_moment = torch.einsum(
            "hnd,hne->hde", sampled_key, sampled_key
        ) / float(sampled_key.shape[1])
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        retained = eigenvalues[..., -projection_dim:].clamp_min(1.0e-6)
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        grouped_query = queries.reshape(
            len(records), kv_heads, group_size, queries.shape[-1]
        )
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)
        projected_query_int8 = quantize_per_token_int8(projected_query)

        root_variance = retained.sqrt()
        balanced_key = projected_key / root_variance.unsqueeze(1)
        balanced_query = projected_query * root_variance.unsqueeze(0).unsqueeze(2)
        balanced_h_key = block_hadamard(balanced_key, band_size)
        balanced_h_query = block_hadamard(balanced_query, band_size)

        key_variants = {
            "pca64_fp": projected_key,
            "pca64_int4_token": quantize_per_token_int4(projected_key),
            "pca64_int4_token_qint8": quantize_per_token_int4(projected_key),
            "pca64_int4_band": quantize_per_band_int4(projected_key, band_size),
            "pca64_int4_band_qint8": quantize_per_band_int4(
                projected_key, band_size
            ),
            "pca64_int4_band_logscale_q025": quantize_per_band_logscale_int4(
                projected_key, band_size, 0.25
            ),
            "pca64_int4_band_logscale_q025_qint8": (
                quantize_per_band_logscale_int4(projected_key, band_size, 0.25)
            ),
            "pca64_int4_band_logscale_q050": quantize_per_band_logscale_int4(
                projected_key, band_size, 0.50
            ),
            "pca64_int4_band_logscale_q100": quantize_per_band_logscale_int4(
                projected_key, band_size, 1.00
            ),
            "balanced_pca64_int4_token": quantize_per_token_int4(balanced_key),
            "balanced_h16_pca64_int4_token": quantize_per_token_int4(
                balanced_h_key
            ),
            "balanced_h16_pca64_int4_band": quantize_per_band_int4(
                balanced_h_key, band_size
            ),
        }
        query_variants = {
            "pca64_fp": projected_query,
            "pca64_int4_token": projected_query,
            "pca64_int4_token_qint8": projected_query_int8,
            "pca64_int4_band": projected_query,
            "pca64_int4_band_qint8": projected_query_int8,
            "pca64_int4_band_logscale_q025": projected_query,
            "pca64_int4_band_logscale_q025_qint8": projected_query_int8,
            "pca64_int4_band_logscale_q050": projected_query,
            "pca64_int4_band_logscale_q100": projected_query,
            "balanced_pca64_int4_token": balanced_query,
            "balanced_h16_pca64_int4_token": balanced_h_query,
            "balanced_h16_pca64_int4_band": balanced_h_query,
        }

        fp_scores = grouped_scores(projected_key, projected_query, group_size)[1:]
        fp_norm = torch.linalg.vector_norm(fp_scores, dim=-1).clamp_min(1.0e-12)
        for method, method_key in key_variants.items():
            method_query = query_variants[method]
            method_scores = grouped_scores(method_key, method_query, group_size)[1:]
            method_recall = candidate_recall(
                method_scores, exact_top, candidate_count
            )
            recall[method].extend(method_recall.flatten().cpu().tolist())
            relative_score_error = (
                torch.linalg.vector_norm(method_scores - fp_scores, dim=-1)
                / fp_norm
            )
            score_error[method].extend(
                relative_score_error.flatten().cpu().tolist()
            )

            fp_key = balanced_h_key if "balanced_h16" in method else (
                balanced_key if "balanced_" in method else projected_key
            )
            relative_key_error = (
                torch.linalg.vector_norm(method_key - fp_key, dim=-1)
                / torch.linalg.vector_norm(fp_key, dim=-1).clamp_min(1.0e-12)
            )
            key_error[method].extend(relative_key_error.flatten().cpu().tolist())

            if method in {
                "pca64_fp",
                "pca64_int4_token",
                "pca64_int4_band_logscale_q025_qint8",
            }:
                record_output_batch(
                    output_metrics,
                    method,
                    method_scores,
                    exact_scores[1:],
                    exact_top,
                    value,
                    candidate_count=candidate_count,
                    keep_count=keep_count,
                    group_size=group_size,
                    scaling=1.0 / math.sqrt(key.shape[-1]),
                )

            rank_method = f"{method}_rank{fixed_rank}"
            rank_scores = grouped_scores(
                method_key[..., -fixed_rank:],
                method_query[..., -fixed_rank:],
                group_size,
            )[1:]
            rank_recall = candidate_recall(rank_scores, exact_top, candidate_count)
            recall[rank_method].extend(rank_recall.flatten().cpu().tolist())

        dynamic_methods = {
            "dynamic_token_int4": (
                key_variants["pca64_int4_token"],
                retained,
            ),
            "dynamic_logscale16_int4": (
                key_variants["pca64_int4_band_logscale_q025"],
                retained,
            ),
            "dynamic_logscale16_int4_quantized_risk": (
                key_variants["pca64_int4_band_logscale_q025"],
                key_variants["pca64_int4_band_logscale_q025"]
                .square()
                .mean(dim=1),
            ),
        }
        for method, (method_key, risk_weights) in dynamic_methods.items():
            method_scores, method_scan_dimensions = dynamic_one_shot_scores(
                method_key,
                projected_query,
                risk_weights,
                keep_count=keep_count,
                candidate_count=candidate_count,
                band_size=band_size,
            )
            method_recall = candidate_recall(
                method_scores, exact_top, candidate_count
            )
            recall[method].extend(method_recall.flatten().cpu().tolist())
            relative_score_error = (
                torch.linalg.vector_norm(method_scores - fp_scores, dim=-1)
                / fp_norm
            )
            score_error[method].extend(
                relative_score_error.flatten().cpu().tolist()
            )
            scan_dimensions[method].extend(method_scan_dimensions)
            record_output_batch(
                output_metrics,
                method,
                method_scores,
                exact_scores[1:],
                exact_top,
                value,
                candidate_count=candidate_count,
                keep_count=keep_count,
                group_size=group_size,
                scaling=1.0 / math.sqrt(key.shape[-1]),
            )

        del key, queries, exact_scores, exact_top, sampled_key, second_moment
        del eigenvalues, eigenvectors, projected_key, projected_query
        del balanced_key, balanced_query, balanced_h_key, balanced_h_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    methods = sorted(recall)
    return {
        "path": str(path),
        "projection_dim": projection_dim,
        "fixed_rank": fixed_rank,
        "candidate_fraction": candidate_fraction,
        "methods": {
            method: {
                "top2_position_recall": summarize(recall[method]),
                **(
                    {"score_relative_l2": summarize(score_error[method])}
                    if method in score_error
                    else {}
                ),
                **(
                    {"key_relative_l2": summarize(key_error[method])}
                    if method in key_error
                    else {}
                ),
                **(
                    {"scanned_dimensions": summarize(scan_dimensions[method])}
                    if method in scan_dimensions
                    else {}
                ),
                **{
                    name: summarize(values)
                    for name, values in output_metrics[method].items()
                },
            }
            for method in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--fixed_rank", type=int, default=48)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--band_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "dual-scaled variance-balanced PCA INT4",
        "traces": [
            evaluate_trace(
                path,
                projection_dim=args.projection_dim,
                fixed_rank=args.fixed_rank,
                candidate_fraction=args.candidate_fraction,
                band_size=args.band_size,
                device=torch.device(args.device),
            )
            for path in args.trace_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for trace in report["traces"]:
        print(trace["path"])
        for method, metrics in trace["methods"].items():
            print(
                method,
                "recall=",
                f'{100.0 * metrics["top2_position_recall"]["mean"]:.4f}%'
            )


if __name__ == "__main__":
    main()
