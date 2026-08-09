from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    exact_rerank,
    grouped_scores,
    logscale16_int4_dequantize,
    parse_floats,
    quantize_query_int8,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import pca_factors, qk_metric_factors


def weighted_key_covariance(
    sampled_key: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    normalized = weights.float() / weights.float().mean(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    return torch.einsum(
        "hn,hnd,hne->hde", normalized, sampled_key, sampled_key
    ) / float(sampled_key.shape[1])


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--candidate_fractions", default="0.04,0.05,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    if not layers or min(layers) < 0:
        raise ValueError("layers must contain nonnegative integers")
    if args.rank % 16 or args.rank > 128:
        raise ValueError("rank must be a multiple of 16 and no larger than 128")
    if args.test_start_step < args.train_steps:
        raise ValueError("test queries must not overlap calibration queries")
    candidate_fractions = parse_floats(args.candidate_fractions)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    head_cases = 0

    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for _, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            needed = args.test_start_step + args.test_steps
            if len(records) < needed:
                raise ValueError(f"{trace_path} has only {len(records)} query steps")
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            head_count, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // head_count
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            sampled_key = key[:, :: args.key_sample_stride]
            global_key_covariance = torch.einsum(
                "hnd,hne->hde", sampled_key, sampled_key
            ) / float(sampled_key.shape[1])
            train_query = torch.stack(
                [
                    row["query"].to(device).float()[0, :, 0]
                    for row in records[: args.train_steps]
                ]
            ).reshape(args.train_steps, head_count, groups, head_dim)
            query_covariance = torch.einsum(
                "thgd,thge->hde", train_query, train_query
            ) / float(args.train_steps * groups)
            isotropic_scale = query_covariance.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1)
            identity = torch.eye(head_dim, device=device).unsqueeze(0)
            regularized_query_covariance = (
                (1.0 - args.query_shrinkage) * query_covariance
                + args.query_shrinkage
                * isotropic_scale[:, None, None]
                * identity
            )

            calibration_scores = torch.einsum(
                "thgd,hnd->thgn", train_query, sampled_key
            )
            score_variance = calibration_scores.square().mean(dim=(0, 2))
            std_key_covariance = weighted_key_covariance(
                sampled_key, score_variance.clamp_min(1.0e-12).sqrt()
            )
            variance_key_covariance = weighted_key_covariance(
                sampled_key, score_variance
            )
            factor_sets = {
                "pca": pca_factors(global_key_covariance, args.rank),
                "qkmetric_global": qk_metric_factors(
                    global_key_covariance,
                    regularized_query_covariance,
                    args.rank,
                ),
                "qkmetric_extreme_std": qk_metric_factors(
                    std_key_covariance,
                    regularized_query_covariance,
                    args.rank,
                ),
                "qkmetric_extreme_variance": qk_metric_factors(
                    variance_key_covariance,
                    regularized_query_covariance,
                    args.rank,
                ),
                "qkmetric_extreme_std_blend50": qk_metric_factors(
                    0.5 * (global_key_covariance + std_key_covariance),
                    regularized_query_covariance,
                    args.rank,
                ),
            }

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            for method, (query_factor, key_factor) in factor_sets.items():
                indexed_key = logscale16_int4_dequantize(
                    torch.einsum("hnd,hdr->hnr", key, key_factor)
                )
                for record in test_records:
                    query = record["query"].to(device).float()[0, :, 0]
                    exact_scores = grouped_scores(query, key) * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    oracle = torch.topk(
                        exact_scores, keep_count, dim=-1, sorted=False
                    ).indices
                    oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)
                    projected_query = torch.einsum(
                        "hgd,hdr->hgr",
                        query.reshape(head_count, groups, head_dim),
                        query_factor,
                    ).reshape(query_heads, args.rank)
                    proxy_scores = grouped_scores(
                        quantize_query_int8(projected_query), indexed_key
                    ) * scaling
                    for fraction in candidate_fractions:
                        candidate_count = max(
                            keep_count, math.ceil(fraction * history_count)
                        )
                        candidates = torch.topk(
                            proxy_scores,
                            candidate_count,
                            dim=-1,
                            sorted=False,
                        ).indices
                        selected = exact_rerank(
                            exact_scores, candidates, keep_count
                        )
                        recall, mass = selected_quality(
                            selected, oracle, attention, oracle_mass
                        )
                        name = f"{method}_candidate{fraction:g}"
                        metrics[name]["top2_recall"].extend(recall.cpu().tolist())
                        metrics[name]["top2_attention_mass_recall"].extend(
                            mass.cpu().tolist()
                        )
                del indexed_key
            head_cases += args.test_steps * query_heads
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "candidate_fractions": candidate_fractions,
        },
        "head_cases": head_cases,
        "retrieval": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(metric_values.items())
            }
            for method, metric_values in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
