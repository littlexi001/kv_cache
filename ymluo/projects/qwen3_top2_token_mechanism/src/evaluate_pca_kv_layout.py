from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_pca_coselection_hybrid import grouped_scores, summarize


def balanced_kd_positions(vectors: np.ndarray, page_size: int) -> np.ndarray:
    """Return a balanced, query-independent physical order in vector space."""
    values = np.asarray(vectors)
    if values.ndim != 2:
        raise ValueError("vectors must have shape [tokens, dimensions]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    token_count = values.shape[0]
    order: list[int] = []
    stack = [np.arange(token_count, dtype=np.int32)]
    while stack:
        indices = stack.pop()
        if indices.size <= page_size:
            order.extend(sorted(indices.tolist()))
            continue
        subset = values[indices]
        split_dimension = int(np.argmax(np.var(subset, axis=0, dtype=np.float64)))
        middle = indices.size // 2
        partition = np.argpartition(subset[:, split_dimension], middle)
        left = indices[partition[:middle]]
        right = indices[partition[middle:]]
        stack.append(right)
        stack.append(left)
    if len(order) != token_count or len(set(order)) != token_count:
        raise RuntimeError("balanced partition did not produce a token permutation")
    positions = np.empty(token_count, dtype=np.int32)
    positions[np.asarray(order, dtype=np.int32)] = np.arange(token_count, dtype=np.int32)
    return positions


def symmetric_int4(values: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(values), axis=0, keepdims=True)
    scale = np.maximum(maximum / 7.0, 1.0e-8)
    return np.clip(np.rint(values / scale), -7, 7).astype(np.int8)


def page_count(tokens: np.ndarray, positions: np.ndarray, page_size: int) -> int:
    return int(np.unique(positions[tokens] // page_size).size)


def evaluate_trace(
    path: Path,
    *,
    page_size: int,
    projection_dim: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records_by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in payload["records"]:
        records_by_layer[int(record["layer"])].append(record)

    page_metrics: dict[str, list[float]] = defaultdict(list)
    build_seconds: dict[str, float] = defaultdict(float)
    layer_steps: dict[int, int] = {}
    for layer, records in sorted(records_by_layer.items()):
        records.sort(key=lambda row: int(row.get("step", 0)))
        layer_steps[layer] = len(records)
        key_record = next((record for record in records if record.get("key") is not None), None)
        if key_record is None:
            raise ValueError(f"layer {layer} has no stored key tensor")
        key = key_record["key"].to(device).float()[0]
        queries = torch.stack(
            [record["query"].to(device).float()[0, :, 0] for record in records]
        )
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        kv_heads = int(key.shape[0])
        query_heads = int(queries.shape[1])
        groups = query_heads // kv_heads
        keep_count = max(1, math.ceil(0.02 * history_count))

        exact_scores = grouped_scores(key, queries, groups)
        exact_indices = torch.topk(exact_scores, k=keep_count, dim=-1).indices.cpu().numpy()
        sampled_key = key[:, ::32]
        second_moment = torch.einsum("hnd,hne->hde", sampled_key, sampled_key) / float(
            sampled_key.shape[1]
        )
        _, eigenvectors = torch.linalg.eigh(second_moment)
        basis = eigenvectors[..., -projection_dim:]
        projected_key = torch.einsum("hnd,hdm->hnm", key, basis).cpu().numpy()
        del exact_scores, key, queries, sampled_key, second_moment, eigenvectors, basis

        chronological = np.arange(history_count, dtype=np.int32)
        for kv_head in range(kv_heads):
            layouts: dict[str, np.ndarray] = {"chronological": chronological}
            for name, vectors in (
                ("pca64_fp32", projected_key[kv_head]),
                ("pca64_int4", symmetric_int4(projected_key[kv_head])),
            ):
                started = time.perf_counter()
                layouts[name] = balanced_kd_positions(vectors, page_size)
                build_seconds[name] += time.perf_counter() - started

            head_start = kv_head * groups
            head_end = head_start + groups
            for step in range(len(records)):
                selected = np.unique(exact_indices[step, head_start:head_end].reshape(-1))
                for name, positions in layouts.items():
                    page_metrics[name].append(float(page_count(selected, positions, page_size)))
                page_metrics["selected_tokens"].append(float(selected.size))

        del exact_indices, projected_key
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = {name: summarize(values) for name, values in page_metrics.items()}
    baseline = summaries["chronological"]["mean"]
    for name in ("pca64_fp32", "pca64_int4"):
        summaries[name]["page_reduction_fraction"] = 1.0 - summaries[name]["mean"] / baseline
        summaries[name]["ideal_page_read_speedup"] = baseline / summaries[name]["mean"]
        summaries[name]["layout_build_seconds_python"] = build_seconds[name]
    return {
        "path": str(path),
        "page_tokens": page_size,
        "projection_dim": projection_dim,
        "layers": len(records_by_layer),
        "layer_steps": layer_steps,
        "test_contract": "layout uses prefill K only; every decode query is held out",
        "quality_contract": "exact selected token IDs are unchanged",
        "metrics": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query-independent PCA-K physical KV layouts.")
    parser.add_argument("--trace_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    traces = [
        evaluate_trace(
            path,
            page_size=args.page_size,
            projection_dim=args.projection_dim,
            device=device,
        )
        for path in args.trace_paths
    ]
    report = {
        "method": "balanced PCA-K page partition",
        "traces": traces,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trace in traces:
        print(trace["path"])
        baseline = trace["metrics"]["chronological"]["mean"]
        for name in ("pca64_fp32", "pca64_int4"):
            metric = trace["metrics"][name]
            print(
                name,
                f"pages={baseline:.3f}->{metric['mean']:.3f}",
                f"reduction={100.0 * metric['page_reduction_fraction']:.2f}%",
                f"ideal_speedup={metric['ideal_page_read_speedup']:.3f}x",
            )


if __name__ == "__main__":
    main()
