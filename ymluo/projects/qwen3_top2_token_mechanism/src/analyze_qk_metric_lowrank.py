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
    parse_ints,
    quantize_query_int8,
    selected_quality,
    summarize,
)


def symmetric_factors(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    floor = eigenvalues.amax(dim=-1, keepdim=True) * 1.0e-8 + 1.0e-12
    eigenvalues = eigenvalues.clamp_min(floor)
    square_root = torch.einsum(
        "hdi,hi,hei->hde", eigenvectors, eigenvalues.sqrt(), eigenvectors
    )
    inverse_square_root = torch.einsum(
        "hdi,hi,hei->hde", eigenvectors, eigenvalues.rsqrt(), eigenvectors
    )
    return square_root, inverse_square_root


def qk_metric_factors(
    key_covariance: torch.Tensor,
    query_covariance: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_sqrt, key_inverse_sqrt = symmetric_factors(key_covariance)
    query_sqrt, query_inverse_sqrt = symmetric_factors(query_covariance)
    metric = query_sqrt @ key_sqrt
    left, singular, right_h = torch.linalg.svd(metric, full_matrices=False)
    scale = singular[..., :rank].sqrt().unsqueeze(-2)
    query_factor = (
        query_inverse_sqrt @ left[..., :rank]
    ) * scale
    key_factor = (
        key_inverse_sqrt @ right_h.transpose(-1, -2)[..., :rank]
    ) * scale
    return query_factor.contiguous(), key_factor.contiguous()


def pca_factors(
    key_covariance: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _, eigenvectors = torch.linalg.eigh(key_covariance.float())
    basis = eigenvectors[..., -rank:].contiguous()
    return basis, basis


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--test_start_step", type=int)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkages", default="0.25,0.5,0.75,0.9,0.97,1.0")
    parser.add_argument("--candidate_fractions", default="0.04,0.05,0.06,0.08")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    layers = {int(item) for item in args.layers.split(",") if item.strip()}
    if not layers or min(layers) < 0:
        raise ValueError("layers must be nonnegative integers")
    shrinkages = parse_floats(args.query_shrinkages)
    candidate_fractions = parse_floats(args.candidate_fractions)
    if args.rank % 16 or args.rank > 128:
        raise ValueError("rank must be a multiple of 16 and no larger than 128")
    if any(shrinkage > 1.0 for shrinkage in shrinkages):
        raise ValueError("query shrinkages must be no larger than one")
    if any(fraction <= args.top_fraction for fraction in candidate_fractions):
        raise ValueError("candidate fractions must exceed the top fraction")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    case_count = 0
    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            test_start_step = (
                args.train_steps
                if args.test_start_step is None
                else args.test_start_step
            )
            if test_start_step < args.train_steps:
                raise ValueError("test start must not overlap query calibration steps")
            needed = test_start_step + args.test_steps
            if len(records) < needed:
                raise ValueError(
                    f"trace {trace_path} layer {layer} has only {len(records)} steps"
                )
            train_records = records[: args.train_steps]
            test_records = records[test_start_step:needed]
            key_record = next(row for row in records if row.get("key") is not None)
            all_key = key_record["key"].to(device).float()[0]
            history_count = all_key.shape[1] - 1
            key = all_key[:, :history_count]
            scaling = float(key_record["scaling"])
            kv_heads, _, head_dim = key.shape
            first_query = train_records[0]["query"]
            query_heads = int(first_query.shape[1])
            groups = query_heads // kv_heads
            keep_count = max(1, math.ceil(args.top_fraction * history_count))

            sampled_key = key[:, :: args.key_sample_stride]
            key_covariance = torch.einsum(
                "hnd,hne->hde", sampled_key, sampled_key
            ) / float(sampled_key.shape[1])
            train_query = torch.stack(
                [row["query"].to(device).float()[0, :, 0, :] for row in train_records]
            ).reshape(args.train_steps, kv_heads, groups, head_dim)
            query_covariance = torch.einsum(
                "thgd,thge->hde", train_query, train_query
            ) / float(args.train_steps * groups)
            isotropic_scale = query_covariance.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1)
            identity = torch.eye(head_dim, device=device).unsqueeze(0)

            factor_sets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "pca": pca_factors(key_covariance, args.rank)
            }
            for shrinkage in shrinkages:
                regularized_query_covariance = (
                    (1.0 - shrinkage) * query_covariance
                    + shrinkage * isotropic_scale[:, None, None] * identity
                )
                factor_sets[f"qkmetric_s{shrinkage:g}"] = qk_metric_factors(
                    key_covariance, regularized_query_covariance, args.rank
                )

            for method, (query_factor, key_factor) in factor_sets.items():
                projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
                indexed_key = logscale16_int4_dequantize(projected_key)
                for record in test_records:
                    query = record["query"].to(device).float()[0, :, 0, :]
                    exact_scores = grouped_scores(query, key) * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    oracle = torch.topk(
                        exact_scores, keep_count, dim=-1, sorted=False
                    ).indices
                    oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)
                    projected_query = torch.einsum(
                        "hgd,hdr->hgr",
                        query.reshape(kv_heads, groups, head_dim),
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
                del projected_key, indexed_key
            case_count += args.test_steps * query_heads
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args) | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "query_shrinkages": shrinkages,
            "candidate_fractions": candidate_fractions,
        },
        "head_cases": case_count,
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
