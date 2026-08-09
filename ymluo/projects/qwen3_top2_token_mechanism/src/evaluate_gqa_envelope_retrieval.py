from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_pca_coselection_hybrid import summarize


def group_ball_upper_bound(
    projected_key: torch.Tensor, normalized_queries: torch.Tensor
) -> tuple[torch.Tensor, float]:
    if projected_key.ndim != 2 or normalized_queries.ndim != 2:
        raise ValueError("key and queries must both be matrices")
    if projected_key.shape[1] != normalized_queries.shape[1]:
        raise ValueError("key and query dimensions must match")
    center = normalized_queries.mean(dim=0)
    radius = torch.linalg.vector_norm(normalized_queries - center, dim=-1).max()
    upper = projected_key @ center + torch.linalg.vector_norm(
        projected_key, dim=-1
    ) * radius
    return upper, float(radius.item())


def certified_prefix_size(
    upper_bound: torch.Tensor, exact_scores: torch.Tensor, top_count: int
) -> int:
    if exact_scores.ndim != 2:
        raise ValueError("exact_scores must have shape [group_heads, tokens]")
    if not 0 < top_count <= exact_scores.shape[1]:
        raise ValueError("top_count must fit within the token count")
    thresholds = torch.topk(exact_scores, k=top_count, dim=-1).values[:, -1]
    shared_threshold = thresholds.min()
    return int((upper_bound >= shared_threshold).sum().item())


def evaluate_trace(
    path: Path,
    *,
    projection_dim: int,
    candidate_fraction: float,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    metrics: dict[str, list[float]] = defaultdict(list)
    layer_steps: dict[int, int] = {}
    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        key_record = next((record for record in records if record.get("key") is not None), None)
        if key_record is None:
            raise ValueError(f"layer {layer} has no stored key tensor")
        key = key_record["key"].to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        kv_heads = int(key.shape[0])
        query_heads = int(queries.shape[1])
        group_size = query_heads // kv_heads
        candidate_count = max(1, math.ceil(candidate_fraction * history_count))

        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key) / float(
            sampled_key.shape[1]
        )
        _, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis)
        grouped_query = queries.reshape(len(records), kv_heads, group_size, queries.shape[-1])
        projected_query = torch.einsum("thgd,hdm->thgm", grouped_query, basis)

        for step in range(len(records)):
            for kv_head in range(kv_heads):
                normalized_query = torch.nn.functional.normalize(
                    projected_query[step, kv_head], dim=-1
                )
                keys = projected_key[kv_head]
                exact_scores = normalized_query @ keys.T
                upper, radius = group_ball_upper_bound(keys, normalized_query)
                prefix = certified_prefix_size(upper, exact_scores, candidate_count)
                prefix_fraction = prefix / history_count
                scan_ratio = 1.0 / group_size + prefix_fraction
                metrics["query_ball_radius"].append(radius)
                metrics["certified_prefix_fraction"].append(prefix_fraction)
                metrics["scan_ratio_vs_independent_heads"].append(scan_ratio)
                metrics["ideal_index_speedup"].append(1.0 / scan_ratio)

        del key, queries, sampled_key, second_moment, eigenvectors, basis
        del projected_key, projected_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "path": str(path),
        "projection_dim": projection_dim,
        "candidate_fraction_per_query_head": candidate_fraction,
        "layers": len(records_by_layer),
        "layer_steps": layer_steps,
        "test_contract": "query-independent PCA basis; no learned router or task label",
        "quality_contract": "ball bound certifies the exact same per-head PCA candidate sets",
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate certified shared GQA envelope retrieval.")
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 < args.candidate_fraction < 1.0:
        raise ValueError("candidate_fraction must be in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {
        "method": "certified GQA query-ball envelope",
        "traces": [
            evaluate_trace(
                path,
                projection_dim=args.projection_dim,
                candidate_fraction=args.candidate_fraction,
                device=device,
            )
            for path in args.trace_paths
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trace in report["traces"]:
        metrics = trace["metrics"]
        print(trace["path"])
        print(
            f"prefix={100.0 * metrics['certified_prefix_fraction']['mean']:.2f}%",
            f"scan_ratio={100.0 * metrics['scan_ratio_vs_independent_heads']['mean']:.2f}%",
            f"ideal_speedup={metrics['ideal_index_speedup']['mean']:.3f}x",
            f"radius={metrics['query_ball_radius']['mean']:.3f}",
        )


if __name__ == "__main__":
    main()
