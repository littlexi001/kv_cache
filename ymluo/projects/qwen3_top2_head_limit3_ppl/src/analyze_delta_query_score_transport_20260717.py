from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def evaluate_trace(
    trace_path: Path,
    device: torch.device,
    projection_dim: int,
    delta_ranks: tuple[int, ...],
    refresh_intervals: tuple[int, ...],
    candidate_fraction: float,
    keep_fraction: float,
    metrics: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)
    layer_steps = {}

    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        if len(records) < 2:
            continue
        first_key = records[0].get("key")
        if first_key is None:
            raise RuntimeError("the first record of each layer must contain K")
        key = first_key.to(device).float()[0]
        scaling = float(records[0]["scaling"])
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0, :] for record in records]
        )
        query_head_count = int(queries.shape[1])
        kv_head_count = int(key.shape[0])
        groups = query_head_count // kv_head_count
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        candidate_count = max(1, math.ceil(candidate_fraction * history_count))
        keep_count = max(1, math.ceil(keep_fraction * history_count))

        sampled_key = key[:, ::32]
        second_moment = torch.einsum(
            "hkd,hke->hde", sampled_key, sampled_key
        ) / float(sampled_key.shape[1])
        _, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        projected_key = torch.einsum("hkd,hdm->hkm", key, basis)
        projected_queries = torch.stack(
            [
                torch.stack(
                    [
                        torch.mv(basis[head // groups].transpose(0, 1), step_query[head])
                        for head in range(query_head_count)
                    ]
                )
                for step_query in queries
            ]
        )
        recursive_scores = {}
        for head in range(query_head_count):
            kv_head = head // groups
            initial_scores = torch.mv(
                projected_key[kv_head], projected_queries[0, head]
            )
            for rank in delta_ranks:
                for interval in refresh_intervals:
                    recursive_scores[(head, rank, interval)] = initial_scores.clone()
                    recursive_scores[
                        (head, rank, interval, "shared")
                    ] = initial_scores.clone()
                    recursive_scores[
                        (head, rank, interval, "pca_tail")
                    ] = initial_scores.clone()

        for step in range(1, len(records)):
            step_delta = projected_queries[step] - projected_queries[step - 1]
            shared_dimensions = {}
            for kv_head in range(kv_head_count):
                grouped_delta = step_delta[
                    kv_head * groups : (kv_head + 1) * groups
                ]
                group_energy = grouped_delta.square().sum(dim=0)
                for rank in delta_ranks:
                    shared_dimensions[(kv_head, rank)] = torch.topk(
                        group_energy, k=rank
                    ).indices
            for head in range(query_head_count):
                kv_head = head // groups
                head_projected_key = projected_key[kv_head]
                previous_query = projected_queries[step - 1, head]
                current_query = projected_queries[step, head]
                delta = current_query - previous_query
                delta_energy = delta.square().sum().clamp_min(1.0e-12)
                previous_scores = torch.mv(head_projected_key, previous_query)
                current_scores = torch.mv(head_projected_key, current_query)
                exact_scores = torch.mv(key[kv_head], queries[step, head]) * scaling
                exact_probabilities = torch.softmax(exact_scores, dim=-1)
                true_indices = torch.topk(exact_scores, k=keep_count).indices

                transported_scores = {
                    "stale": previous_scores,
                    "current_pca64": current_scores,
                }
                delta_corrections = {}
                for rank in delta_ranks:
                    dimensions = torch.topk(delta.abs(), k=rank).indices
                    truncated_delta = torch.zeros_like(delta)
                    truncated_delta[dimensions] = delta[dimensions]
                    delta_correction = torch.mv(
                        head_projected_key, truncated_delta
                    )
                    delta_corrections[rank] = delta_correction
                    transported_scores[f"delta{rank}"] = (
                        previous_scores + delta_correction
                    )
                    metrics[f"delta{rank}"]["delta_energy_retained"].append(
                        float(
                            (
                                truncated_delta.square().sum() / delta_energy
                            ).item()
                        )
                    )
                    shared_truncated_delta = torch.zeros_like(delta)
                    shared_indices = shared_dimensions[(kv_head, rank)]
                    shared_truncated_delta[shared_indices] = delta[shared_indices]
                    shared_correction = torch.mv(
                        head_projected_key, shared_truncated_delta
                    )
                    transported_scores[f"delta{rank}_shared"] = (
                        previous_scores + shared_correction
                    )
                    metrics[f"delta{rank}_shared"][
                        "delta_energy_retained"
                    ].append(
                        float(
                            (
                                shared_truncated_delta.square().sum()
                                / delta_energy
                            ).item()
                        )
                    )
                    fixed_indices = torch.arange(
                        projection_dim - rank,
                        projection_dim,
                        device=device,
                    )
                    fixed_truncated_delta = torch.zeros_like(delta)
                    fixed_truncated_delta[fixed_indices] = delta[fixed_indices]
                    fixed_correction = torch.mv(
                        head_projected_key, fixed_truncated_delta
                    )
                    transported_scores[f"delta{rank}_pca_tail"] = (
                        previous_scores + fixed_correction
                    )
                    metrics[f"delta{rank}_pca_tail"][
                        "delta_energy_retained"
                    ].append(
                        float(
                            (
                                fixed_truncated_delta.square().sum()
                                / delta_energy
                            ).item()
                        )
                    )
                    for interval in refresh_intervals:
                        cache_key = (head, rank, interval)
                        if step % interval == 0:
                            recursive_scores[cache_key] = current_scores.clone()
                        else:
                            recursive_scores[cache_key] = (
                                recursive_scores[cache_key] + delta_correction
                            )
                        transported_scores[
                            f"delta{rank}_refresh{interval}"
                        ] = recursive_scores[cache_key]
                        shared_cache_key = (head, rank, interval, "shared")
                        if step % interval == 0:
                            recursive_scores[shared_cache_key] = (
                                current_scores.clone()
                            )
                        else:
                            recursive_scores[shared_cache_key] = (
                                recursive_scores[shared_cache_key]
                                + shared_correction
                            )
                        transported_scores[
                            f"delta{rank}_shared_refresh{interval}"
                        ] = recursive_scores[shared_cache_key]
                        fixed_cache_key = (head, rank, interval, "pca_tail")
                        if step % interval == 0:
                            recursive_scores[fixed_cache_key] = (
                                current_scores.clone()
                            )
                        else:
                            recursive_scores[fixed_cache_key] = (
                                recursive_scores[fixed_cache_key]
                                + fixed_correction
                            )
                        transported_scores[
                            f"delta{rank}_pca_tail_refresh{interval}"
                        ] = recursive_scores[fixed_cache_key]

                score_scale = current_scores.std().clamp_min(1.0e-8)
                for method, approximate_scores in transported_scores.items():
                    normalized_rmse = (
                        (approximate_scores - current_scores).square().mean().sqrt()
                        / score_scale
                    )
                    candidate_indices = torch.topk(
                        approximate_scores, k=candidate_count
                    ).indices
                    candidate_mask = torch.zeros(
                        history_count, dtype=torch.bool, device=device
                    )
                    candidate_mask[candidate_indices] = True
                    candidate_recall = candidate_mask[true_indices].float().mean()
                    candidate_exact_scores = exact_scores.index_select(
                        0, candidate_indices
                    )
                    selected_local = torch.topk(
                        candidate_exact_scores, k=keep_count
                    ).indices
                    selected_indices = candidate_indices.index_select(
                        0, selected_local
                    )
                    selected_mask = torch.zeros_like(candidate_mask)
                    selected_mask[selected_indices] = True
                    topk_recall = selected_mask[true_indices].float().mean()
                    retained_mass = exact_probabilities.index_select(
                        0, selected_indices
                    ).sum()
                    metrics[method]["normalized_score_rmse"].append(
                        float(normalized_rmse.item())
                    )
                    metrics[method]["candidate_recall_exact_topk"].append(
                        float(candidate_recall.item())
                    )
                    metrics[method]["reranked_topk_recall"].append(
                        float(topk_recall.item())
                    )
                    metrics[method]["retained_attention_mass"].append(
                        float(retained_mass.item())
                    )
        del key, queries, projected_key, projected_queries

    return {"path": str(trace_path), "layer_steps": layer_steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--delta_ranks", default="4,8,16,32")
    parser.add_argument("--refresh_intervals", default="2,4,8,16")
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--keep_fraction", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ranks = tuple(int(item) for item in args.delta_ranks.split(",") if item)
    refresh_intervals = tuple(
        int(item) for item in args.refresh_intervals.split(",") if item
    )
    if not ranks or min(ranks) < 1 or max(ranks) > args.projection_dim:
        raise ValueError("delta ranks must be within the projection dimension")
    if not refresh_intervals or min(refresh_intervals) < 2:
        raise ValueError("refresh intervals must be at least two")
    if not 0.0 < args.keep_fraction <= args.candidate_fraction <= 1.0:
        raise ValueError("fractions must satisfy 0 < keep <= candidate <= 1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    traces = [
        evaluate_trace(
            path,
            device,
            args.projection_dim,
            ranks,
            refresh_intervals,
            args.candidate_fraction,
            args.keep_fraction,
            metrics,
        )
        for path in args.trace_paths
    ]
    report = {
        "traces": traces,
        "projection_dim": args.projection_dim,
        "delta_ranks": ranks,
        "refresh_intervals": refresh_intervals,
        "transport_scan_estimates": {
            f"delta{rank}_refresh{interval}": {
                "average_projection_dimensions": (
                    args.projection_dim + (interval - 1) * rank
                )
                / interval,
                "projection_scan_speedup": args.projection_dim
                / (
                    (args.projection_dim + (interval - 1) * rank)
                    / interval
                ),
            }
            for rank in ranks
            for interval in refresh_intervals
        },
        "candidate_fraction": args.candidate_fraction,
        "keep_fraction": args.keep_fraction,
        "note": "FP32 PCA transport ceiling; recursive accumulation is included, INT4 is not.",
        "metrics": {
            method: {
                metric: summarize(values)
                for metric, values in sorted(method_metrics.items())
            }
            for method, method_metrics in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
