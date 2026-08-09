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
    quantize_query_int8,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors, symmetric_factors


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--coverage_thresholds", default="0.90,0.95,0.97,0.98,0.99")
    args = parser.parse_args()

    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    thresholds = [
        float(value) for value in args.coverage_thresholds.split(",") if value.strip()
    ]
    if not layers or min(layers) < 0:
        raise ValueError("layers must contain nonnegative integers")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layer_rows: list[dict[str, Any]] = []

    for trace_path in args.trace_paths:
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        records_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            layer = int(record["layer"])
            if layer in layers:
                records_by_layer[layer].append(record)

        for layer, records in sorted(records_by_layer.items()):
            records.sort(key=lambda row: int(row.get("step", 0)))
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            head_count, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // head_count
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
            key_sqrt, _ = symmetric_factors(key_covariance)
            query_sqrt, _ = symmetric_factors(regularized_query_covariance)
            singular = torch.linalg.svdvals(query_sqrt @ key_sqrt)
            energy = singular.square()
            coverage48 = energy[..., :48].sum(dim=-1) / energy.sum(dim=-1)
            coverage64 = energy[..., :64].sum(dim=-1) / energy.sum(dim=-1)

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            rank_metrics: dict[int, dict[str, list[float]]] = {
                48: defaultdict(list),
                64: defaultdict(list),
            }
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            candidate_count = max(
                keep_count, math.ceil(args.candidate_fraction * history_count)
            )
            for rank in (48, 64):
                query_factor, key_factor = qk_metric_factors(
                    key_covariance, regularized_query_covariance, rank
                )
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
                    ).reshape(query_heads, rank)
                    proxy_scores = grouped_scores(
                        quantize_query_int8(projected_query), indexed_key
                    ) * scaling
                    candidates = torch.topk(
                        proxy_scores, candidate_count, dim=-1, sorted=False
                    ).indices
                    selected = exact_rerank(
                        exact_scores, candidates, keep_count
                    )
                    recall, mass = selected_quality(
                        selected, oracle, attention, oracle_mass
                    )
                    rank_metrics[rank]["top2_recall"].extend(recall.cpu().tolist())
                    rank_metrics[rank]["top2_attention_mass_recall"].extend(
                        mass.cpu().tolist()
                    )

            layer_rows.append(
                {
                    "trace": str(trace_path),
                    "layer": layer,
                    "coverage48_mean": float(coverage48.mean()),
                    "coverage48_min": float(coverage48.min()),
                    "coverage64_mean": float(coverage64.mean()),
                    "coverage64_min": float(coverage64.min()),
                    "rank48": {
                        name: values for name, values in rank_metrics[48].items()
                    },
                    "rank64": {
                        name: values for name, values in rank_metrics[64].items()
                    },
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    methods: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    selected_ranks: dict[str, list[int]] = defaultdict(list)
    for row in layer_rows:
        for rank in (48, 64):
            name = f"fixed_rank{rank}"
            for metric, values in row[f"rank{rank}"].items():
                methods[name][metric].extend(values)
        for threshold in thresholds:
            rank = 48 if row["coverage48_min"] >= threshold else 64
            name = f"coverage_gate_{threshold:g}"
            selected_ranks[name].append(rank)
            for metric, values in row[f"rank{rank}"].items():
                methods[name][metric].extend(values)

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "coverage_thresholds": thresholds,
        },
        "layers": [
            {
                key: value
                for key, value in row.items()
                if key not in {"rank48", "rank64"}
            }
            | {
                "rank48_summary": {
                    metric: summarize(values)
                    for metric, values in row["rank48"].items()
                },
                "rank64_summary": {
                    metric: summarize(values)
                    for metric, values in row["rank64"].items()
                },
            }
            for row in layer_rows
        ],
        "methods": {
            name: {
                "mean_rank": (
                    sum(selected_ranks[name]) / len(selected_ranks[name])
                    if name in selected_ranks
                    else int(name.removeprefix("fixed_rank"))
                ),
                "rank64_layer_fraction": (
                    sum(rank == 64 for rank in selected_ranks[name])
                    / len(selected_ranks[name])
                    if name in selected_ranks
                    else float(name == "fixed_rank64")
                ),
                "metrics": {
                    metric: summarize(values)
                    for metric, values in metric_values.items()
                },
            }
            for name, metric_values in methods.items()
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
