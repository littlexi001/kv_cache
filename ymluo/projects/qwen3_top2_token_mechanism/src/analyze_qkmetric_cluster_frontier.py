from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    grouped_scores,
    parse_floats,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors


def assign_clusters(
    value: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    centroid_norm = centroids.square().sum(dim=-1)
    assignments = []
    for start in range(0, value.shape[0], batch_size):
        chunk = value[start : start + batch_size]
        distance = (
            chunk.square().sum(dim=-1, keepdim=True)
            + centroid_norm.unsqueeze(0)
            - 2.0 * chunk @ centroids.T
        )
        assignments.append(distance.argmin(dim=-1))
    return torch.cat(assignments)


def recompute_centroids(
    value: torch.Tensor,
    assignments: torch.Tensor,
    previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_count, dimensions = previous.shape
    sums = torch.zeros(
        cluster_count, dimensions, dtype=torch.float32, device=value.device
    )
    counts = torch.bincount(assignments, minlength=cluster_count).float()
    sums.index_add_(0, assignments, value.float())
    centroids = sums / counts.clamp_min(1.0).unsqueeze(-1)
    centroids = torch.where(
        (counts > 0).unsqueeze(-1), centroids, previous.float()
    )
    return centroids, counts


def train_kmeans(
    value: torch.Tensor,
    cluster_count: int,
    sample_count: int,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = min(sample_count, value.shape[0])
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    sample_indices = torch.randperm(
        value.shape[0], generator=generator, device=value.device
    )[:sample_count]
    sample = value[sample_indices].float()
    initial_indices = torch.linspace(
        0, sample_count - 1, cluster_count, device=value.device
    ).round().long()
    centroids = sample[initial_indices].clone()
    for _ in range(iterations):
        assignment = assign_clusters(sample, centroids, batch_size)
        centroids, _ = recompute_centroids(sample, assignment, centroids)

    full_assignment = assign_clusters(value.float(), centroids, batch_size)
    centroids, counts = recompute_centroids(
        value.float(), full_assignment, centroids
    )
    full_assignment = assign_clusters(value.float(), centroids, batch_size)
    centroids, counts = recompute_centroids(
        value.float(), full_assignment, centroids
    )
    return centroids.contiguous(), full_assignment.contiguous(), counts


def cluster_band_radii(
    value: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    band_size: int,
) -> torch.Tensor:
    residual = value.float() - centroids[assignments]
    band_norm = residual.reshape(value.shape[0], -1, band_size).norm(dim=-1)
    radii = torch.zeros(
        centroids.shape[0], band_norm.shape[-1],
        dtype=torch.float32,
        device=value.device,
    )
    expanded = assignments.unsqueeze(-1).expand_as(band_norm)
    radii.scatter_reduce_(0, expanded, band_norm, reduce="amax", include_self=True)
    return radii


def cluster_diagonal_variance(
    value: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    residual_square = (value.float() - centroids[assignments]).square()
    variance = torch.zeros_like(centroids, dtype=torch.float32)
    variance.index_add_(0, assignments, residual_square)
    return variance / counts.clamp_min(1.0).unsqueeze(-1)


def gaussian_expected_maximum_multiplier(counts: torch.Tensor) -> torch.Tensor:
    """Leading-order expected maximum of `counts` standard-normal samples."""
    return torch.sqrt(2.0 * counts.clamp_min(2.0).log())


def cluster_prefix_for_budget(
    cluster_scores: torch.Tensor,
    counts: torch.Tensor,
    target_count: int,
) -> tuple[torch.Tensor, int]:
    order = cluster_scores.argsort(descending=True)
    cumulative = counts[order].cumsum(dim=0)
    probe_count = int(
        torch.searchsorted(
            cumulative, torch.tensor(float(target_count), device=counts.device)
        ).item()
    ) + 1
    probe_count = min(probe_count, order.numel())
    return order[:probe_count], probe_count


def exact_rerank_scanned(
    exact_scores: torch.Tensor,
    scanned: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    scanned_scores = exact_scores[scanned]
    local_count = min(keep_count, scanned.numel())
    local = torch.topk(
        scanned_scores, local_count, dim=-1, sorted=False
    ).indices
    return scanned[local]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--cluster_count", type=int, default=256)
    parser.add_argument("--kmeans_sample_count", type=int, default=8192)
    parser.add_argument("--kmeans_iterations", type=int, default=8)
    parser.add_argument("--assignment_batch_size", type=int, default=4096)
    parser.add_argument("--scan_fractions", default="0.03,0.04,0.05,0.06,0.08,0.10")
    parser.add_argument(
        "--norm_reserve_fractions", default="0.0025,0.005,0.01,0.02"
    )
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    layers = {int(item) for item in args.layers.split(",") if item.strip()}
    scan_fractions = parse_floats(args.scan_fractions)
    norm_reserve_fractions = parse_floats(args.norm_reserve_fractions)
    if args.rank % 16 or args.rank > 128:
        raise ValueError("rank must be a multiple of 16 and at most 128")
    if args.cluster_count <= 1 or args.cluster_count > 65536:
        raise ValueError("cluster count must be in [2, 65536]")
    if max(norm_reserve_fractions) >= min(scan_fractions):
        raise ValueError("norm reserve fractions must be below every scan fraction")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    build_rows: list[dict[str, Any]] = []

    for trace_index, trace_path in enumerate(args.trace_paths):
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            needed = args.test_start_step + args.test_steps
            if len(records) < needed:
                raise ValueError(f"{trace_path} layer {layer} has too few queries")
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            head_count, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // head_count
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            sampled_key = key[:, :: args.key_sample_stride]
            key_covariance = torch.einsum(
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
            query_factor, key_factor = qk_metric_factors(
                key_covariance, regularized_query_covariance, args.rank
            )
            projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
            norm_reserves = {
                fraction: torch.topk(
                    projected_key.norm(dim=-1),
                    max(1, math.ceil(fraction * history_count)),
                    dim=-1,
                    sorted=False,
                ).indices
                for fraction in norm_reserve_fractions
            }

            centroids_by_head = []
            assignments_by_head = []
            counts_by_head = []
            radii_by_head = []
            variance_by_head = []
            if device.type == "cuda":
                torch.cuda.synchronize()
            build_start = time.perf_counter()
            for head in range(head_count):
                centroids, assignment, counts = train_kmeans(
                    projected_key[head],
                    args.cluster_count,
                    args.kmeans_sample_count,
                    args.kmeans_iterations,
                    args.assignment_batch_size,
                    args.seed + trace_index * 1000 + layer * 17 + head,
                )
                centroids_by_head.append(centroids)
                assignments_by_head.append(assignment)
                counts_by_head.append(counts)
                radii_by_head.append(
                    cluster_band_radii(
                        projected_key[head], centroids, assignment, 16
                    )
                )
                variance_by_head.append(
                    cluster_diagonal_variance(
                        projected_key[head], centroids, assignment, counts
                    )
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            build_seconds = time.perf_counter() - build_start
            centroids = torch.stack(centroids_by_head)
            assignments = torch.stack(assignments_by_head)
            counts = torch.stack(counts_by_head)
            radii = torch.stack(radii_by_head)
            variance = torch.stack(variance_by_head)
            build_rows.append(
                {
                    "trace": str(trace_path),
                    "layer": layer,
                    "history_tokens": history_count,
                    "build_seconds": build_seconds,
                    "empty_cluster_fraction": float((counts == 0).float().mean()),
                    "cluster_size_mean": float(counts.mean()),
                    "cluster_size_max": float(counts.max()),
                }
            )

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            for record in test_records:
                query = record["query"].to(device).float()[0, :, 0]
                grouped_query = query.reshape(head_count, groups, head_dim)
                projected_query = torch.einsum(
                    "hgd,hdr->hgr", grouped_query, query_factor
                )
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                for head in range(head_count):
                    for group in range(groups):
                        query_head = head * groups + group
                        center_scores = (
                            centroids[head] @ projected_query[head, group]
                        ) * scaling
                        query_bands = projected_query[head, group].reshape(-1, 16)
                        upper_scores = center_scores + (
                            radii[head] * query_bands.norm(dim=-1).unsqueeze(0)
                        ).sum(dim=-1) * scaling
                        directional_sigma = torch.sqrt(
                            variance[head]
                            @ projected_query[head, group].square()
                        ).clamp_min(0.0)
                        expected_max_scores = center_scores + (
                            gaussian_expected_maximum_multiplier(counts[head])
                            * directional_sigma
                            * scaling
                        )
                        score_sets = [
                            ("center", center_scores, 0.0),
                            ("band_upper", upper_scores, 0.0),
                            ("expected_max", expected_max_scores, 0.0),
                        ]
                        score_sets.extend(
                            (
                                f"center_norm{reserve_fraction:g}",
                                center_scores,
                                reserve_fraction,
                            )
                            for reserve_fraction in norm_reserve_fractions
                        )
                        for score_name, cluster_scores, reserve_fraction in score_sets:
                            for fraction in scan_fractions:
                                cluster_target_count = max(
                                    1,
                                    math.ceil(
                                        (fraction - reserve_fraction)
                                        * history_count
                                    ),
                                )
                                selected_clusters, probe_count = (
                                    cluster_prefix_for_budget(
                                        cluster_scores,
                                        counts[head],
                                        cluster_target_count,
                                    )
                                )
                                membership = torch.zeros(
                                    args.cluster_count,
                                    dtype=torch.bool,
                                    device=device,
                                )
                                membership[selected_clusters] = True
                                scanned = torch.nonzero(
                                    membership[assignments[head]], as_tuple=False
                                ).flatten()
                                if reserve_fraction:
                                    scanned = torch.unique(
                                        torch.cat(
                                            [
                                                scanned,
                                                norm_reserves[reserve_fraction][head],
                                            ]
                                        )
                                    )
                                if scanned.numel() < keep_count:
                                    continue
                                selected = exact_rerank_scanned(
                                    exact_scores[query_head], scanned, keep_count
                                )
                                recall, mass = selected_quality(
                                    selected.unsqueeze(0),
                                    oracle[query_head].unsqueeze(0),
                                    attention[query_head].unsqueeze(0),
                                    oracle_mass[query_head].unsqueeze(0),
                                )
                                name = f"{score_name}_scan{fraction:g}"
                                metrics[name]["top2_recall"].append(
                                    float(recall.item())
                                )
                                metrics[name]["top2_attention_mass_recall"].append(
                                    float(mass.item())
                                )
                                metrics[name]["scanned_fraction"].append(
                                    scanned.numel() / history_count
                                )
                                metrics[name]["probe_count"].append(probe_count)

            del key, projected_key, centroids, assignments, counts, radii, variance
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "scan_fractions": scan_fractions,
            "norm_reserve_fractions": norm_reserve_fractions,
        },
        "build": build_rows,
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
