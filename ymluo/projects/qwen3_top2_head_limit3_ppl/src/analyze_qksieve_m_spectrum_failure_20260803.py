#!/usr/bin/env python
"""Relate QK-balanced spectrum/drift to held-out proxy failures.

This is a real-QKV mechanism probe, not a model-level quality benchmark.  It
constructs the production request-local basis from final-prefill Queries and
evaluates later decode Queries without using them to choose the basis or bits.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import allocate_bits
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    covariance,
    qk_balanced_factors,
    symmetric_covariance_factors,
)
from analyze_qk_progressive_refinement_20260727 import reconstruct
from analyze_qksieve_output_risk_budget_20260803 import (
    affine_calibrate_scores,
    histogram_coverage_mask,
    key_allocation_distortion,
    key_quantization_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query_tokens", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--key_rate_budget", type=int, default=19)
    parser.add_argument(
        "--key_quantizer", choices=("plain", "metric"), default="metric"
    )
    parser.add_argument(
        "--key_allocation_objective",
        choices=("key_mse", "qk_mse"),
        default="key_mse",
    )
    parser.add_argument("--score_calibration_samples", type=int, default=256)
    parser.add_argument("--coverage_target", type=float, default=0.95)
    parser.add_argument("--coverage_histogram_bins", type=int, default=256)
    parser.add_argument("--minimum_top_k", type=int, default=256)
    return parser.parse_args()


def resolve_paths(specifications: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for specification in specifications:
        matches = glob.glob(specification)
        if not matches and Path(specification).is_file():
            matches = [specification]
        paths.update(Path(match).resolve() for match in matches)
    if not paths:
        raise FileNotFoundError("no traces matched")
    return sorted(paths)


def regularized_qk_spectrum(
    sampled_key: torch.Tensor,
    queries: torch.Tensor,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key_covariance = covariance(sampled_key.float())
    raw_query_covariance = covariance(queries.float())
    isotropic_scale = raw_query_covariance.diagonal().mean()
    regularized_query = (
        (1.0 - shrinkage) * raw_query_covariance
        + shrinkage
        * isotropic_scale
        * torch.eye(
            raw_query_covariance.shape[0],
            device=raw_query_covariance.device,
        )
    )
    query_sqrt, _ = symmetric_covariance_factors(regularized_query)
    key_sqrt, _ = symmetric_covariance_factors(key_covariance)
    left, singular_values, right_h = torch.linalg.svd(
        query_sqrt @ key_sqrt,
        full_matrices=False,
    )
    return regularized_query, singular_values, left, right_h.transpose(0, 1)


def effective_rank(singular_values: torch.Tensor) -> float:
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1.0e-30)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()
    return float(entropy.exp())


def energy_fraction(singular_values: torch.Tensor, rank: int) -> float:
    energy = singular_values.square()
    return float(energy[:rank].sum() / energy.sum().clamp_min(1.0e-30))


def subspace_sine(first: torch.Tensor, second: torch.Tensor, rank: int) -> float:
    cosines = torch.linalg.svdvals(first[:, :rank].T @ second[:, :rank])
    minimum = cosines.amin().clamp(0.0, 1.0)
    return float(torch.sqrt((1.0 - minimum.square()).clamp_min(0.0)))


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "maximum": float(tensor.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def analyze_trace(path: Path, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    prefill_by_layer = {
        int(layer): query for layer, query in payload["prefill_query_tail"].items()
    }
    records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)
    rows: list[dict[str, Any]] = []
    for layer, raw_records in sorted(records_by_layer.items()):
        records = sorted(raw_records, key=lambda row: int(row["step"]))
        state = next(row for row in records if row.get("key") is not None)
        key_all = state["key"][0].to(device).float()
        scaling = float(state["scaling"])
        decode = torch.stack(
            [record["query"][0, :, 0, :].to(device).float() for record in records]
        )
        raw_prefill = prefill_by_layer[layer]
        if raw_prefill.shape[-2] < args.query_tokens:
            raise ValueError(f"{path}: layer {layer} has too few prefill Queries")
        prefill = (
            raw_prefill[0, :, -args.query_tokens :, :]
            .permute(1, 0, 2)
            .contiguous()
            .to(device)
            .float()
        )
        query_groups = int(decode.shape[1] // key_all.shape[0])
        for kv_head in range(int(key_all.shape[0])):
            head_slice = slice(kv_head * query_groups, (kv_head + 1) * query_groups)
            key = key_all[kv_head]
            sampled_key = key[:: args.key_sample_stride]
            prefill_queries = prefill[:, head_slice].reshape(-1, key.shape[-1])
            decode_queries = decode[:, head_slice].reshape(-1, key.shape[-1])
            (
                prefill_covariance,
                prefill_singular,
                prefill_left,
                prefill_right,
            ) = regularized_qk_spectrum(
                sampled_key,
                prefill_queries,
                args.query_shrinkage,
            )
            (
                decode_covariance,
                decode_singular,
                decode_left,
                decode_right,
            ) = regularized_qk_spectrum(
                sampled_key,
                decode_queries,
                args.query_shrinkage,
            )
            query_factor, key_factor, _ = qk_balanced_factors(
                sampled_key,
                prefill_queries,
                args.query_shrinkage,
            )
            key_coordinates = key @ key_factor
            projected_prefill = prefill_queries @ query_factor
            bands = key_quantization_candidates(
                key_coordinates,
                projected_prefill,
                args.key_quantizer,
            )
            distortion = key_allocation_distortion(
                key_coordinates,
                projected_prefill,
                bands,
                args.key_allocation_objective,
            )
            allocation = allocate_bits(
                distortion,
                args.key_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            approximate_key = reconstruct(bands, allocation)
            projected_decode = decode_queries @ query_factor
            approximate_query = torch.stack(
                [query_int8(query) for query in projected_decode]
            )
            exact_scores = decode_queries @ key.T * scaling
            proxy_scores = approximate_query.float() @ approximate_key.T * scaling
            calibrated, slopes, _ = affine_calibrate_scores(
                exact_scores,
                proxy_scores,
                args.score_calibration_samples,
            )
            residual = calibrated - exact_scores
            exact_std = exact_scores.std(dim=-1, correction=0).clamp_min(1.0e-12)
            normalized_rmse = residual.square().mean(dim=-1).sqrt() / exact_std
            exact_log = torch.log_softmax(exact_scores, dim=-1)
            proxy_log = torch.log_softmax(calibrated, dim=-1)
            exact_probability = exact_log.exp()
            kl = (exact_probability * (exact_log - proxy_log)).sum(dim=-1)
            selected = histogram_coverage_mask(
                calibrated,
                args.coverage_target,
                args.minimum_top_k,
                0,
                args.coverage_histogram_bins,
            )
            selected_mass = (exact_probability * selected).sum(dim=-1)
            covariance_drift = torch.linalg.matrix_norm(
                decode_covariance - prefill_covariance,
                ord=2,
            ) / torch.linalg.matrix_norm(decode_covariance, ord=2).clamp_min(1.0e-30)
            row: dict[str, Any] = {
                "trace": str(path),
                "task": str(payload.get("task", "")),
                "sample_id": str(payload.get("sample_id", "")),
                "history_tokens": int(key.shape[0]),
                "layer": layer,
                "kv_head": kv_head,
                "allocation": "-".join(map(str, allocation)),
                "prefill_effective_rank": effective_rank(prefill_singular),
                "decode_effective_rank": effective_rank(decode_singular),
                "query_covariance_drift": float(covariance_drift),
                "calibrated_slope_mean": float(slopes.mean()),
                "normalized_score_rmse_mean": float(normalized_rmse.mean()),
                "softmax_kl_mean": float(kl.mean()),
                "selected_ratio_mean": float(selected.float().mean()),
                "exact_selected_mass_mean": float(selected_mass.mean()),
            }
            for rank in (16, 32, 48, 64):
                row[f"prefill_energy_r{rank}"] = energy_fraction(
                    prefill_singular, rank
                )
                row[f"decode_energy_r{rank}"] = energy_fraction(
                    decode_singular, rank
                )
                row[f"left_drift_r{rank}"] = subspace_sine(
                    prefill_left, decode_left, rank
                )
                row[f"right_drift_r{rank}"] = subspace_sine(
                    prefill_right, decode_right, rank
                )
            row["prefill_sigma48_to_sigma1"] = float(
                prefill_singular[47] / prefill_singular[0].clamp_min(1.0e-30)
            )
            row["decode_sigma48_to_sigma1"] = float(
                decode_singular[47] / decode_singular[0].clamp_min(1.0e-30)
            )
            rows.append(row)
        torch.cuda.empty_cache()
        print(json.dumps({"trace": path.name, "layer": layer}), flush=True)
    return rows


def pearson(rows: list[dict[str, Any]], first: str, second: str) -> float:
    values = torch.tensor(
        [[float(row[first]), float(row[second])] for row in rows],
        dtype=torch.float64,
    )
    if values.shape[0] < 2:
        return math.nan
    return float(torch.corrcoef(values.T)[0, 1])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = resolve_paths(args.trace)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(analyze_trace(path, args, device))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trace"])].append(row)
    metrics = (
        "prefill_effective_rank",
        "decode_effective_rank",
        "query_covariance_drift",
        "left_drift_r16",
        "right_drift_r16",
        "left_drift_r48",
        "right_drift_r48",
        "normalized_score_rmse_mean",
        "softmax_kl_mean",
        "selected_ratio_mean",
        "exact_selected_mass_mean",
    )
    traces = []
    for trace, trace_rows in grouped.items():
        traces.append(
            {
                "trace": trace,
                "sample_id": trace_rows[0]["sample_id"],
                "history_tokens": trace_rows[0]["history_tokens"],
                **{
                    metric: quantiles([float(row[metric]) for row in trace_rows])
                    for metric in metrics
                },
            }
        )
    correlations = {
        metric: {
            "vs_normalized_score_rmse": pearson(
                rows, metric, "normalized_score_rmse_mean"
            ),
            "vs_softmax_kl": pearson(rows, metric, "softmax_kl_mean"),
        }
        for metric in (
            "prefill_effective_rank",
            "decode_effective_rank",
            "query_covariance_drift",
            "left_drift_r16",
            "right_drift_r16",
            "left_drift_r48",
            "right_drift_r48",
            "prefill_energy_r48",
            "decode_energy_r48",
        )
    }
    report = {
        "schema": "qksieve_m_spectrum_failure_v1",
        "setup": vars(args) | {"output_dir": str(args.output_dir)},
        "claim_boundary": (
            "Real-QKV held-out mechanism analysis; no downstream model score "
            "or runtime claim is made."
        ),
        "traces": traces,
        "head_level_correlations": correlations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
