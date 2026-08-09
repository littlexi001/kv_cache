from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_numeric_pruning_frontier import (
    grouped_scores,
    logscale16_int4_dequantize,
    parse_floats,
    quantize_query_int8,
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors


def local_quota_candidates(
    scores: torch.Tensor,
    block_size: int,
    quota: int,
    candidate_count: int,
) -> tuple[torch.Tensor, int]:
    token_count = int(scores.numel())
    block_count = math.ceil(token_count / block_size)
    padded_count = block_count * block_size
    if padded_count != token_count:
        scores = torch.cat(
            (
                scores,
                torch.full(
                    (padded_count - token_count,),
                    -torch.inf,
                    dtype=scores.dtype,
                    device=scores.device,
                ),
            )
        )
    blocks = scores.reshape(block_count, block_size)
    local_scores, local_offsets = torch.topk(
        blocks, min(quota, block_size), dim=-1, sorted=False
    )
    block_offsets = (
        torch.arange(block_count, device=scores.device).unsqueeze(-1) * block_size
    )
    local_indices = (local_offsets + block_offsets).flatten()
    local_scores = local_scores.flatten()
    valid = local_indices < token_count
    local_indices = local_indices[valid]
    local_scores = local_scores[valid]
    if local_indices.numel() > candidate_count:
        final_local = torch.topk(
            local_scores, candidate_count, sorted=False
        ).indices
        local_indices = local_indices[final_local]
    return local_indices, int(valid.sum().item())


def record_quality(
    bucket: dict[str, list[float]],
    selected: torch.Tensor,
    oracle: torch.Tensor,
    attention: torch.Tensor,
    oracle_mass: torch.Tensor,
) -> None:
    recall, mass = selected_quality(
        selected.unsqueeze(0),
        oracle.unsqueeze(0),
        attention.unsqueeze(0),
        oracle_mass.unsqueeze(0),
    )
    bucket["top2_recall"].append(float(recall.item()))
    bucket["top2_attention_mass_recall"].append(float(mass.item()))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline quality frontier for fused block-local QK selection."
    )
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--block_sizes", default="128,256,512")
    parser.add_argument("--quota_fractions", default="0.0625,0.09375,0.125")
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    block_sizes = [
        int(value) for value in args.block_sizes.split(",") if value.strip()
    ]
    quota_fractions = parse_floats(args.quota_fractions)
    if min(quota_fractions) < args.candidate_fraction:
        raise ValueError("local quotas must cover the global candidate fraction")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    build_rows: list[dict[str, Any]] = []

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
            kv_heads, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // kv_heads
            keep_count = max(1, math.ceil(args.top_fraction * history_count))
            candidate_count = max(
                keep_count, math.ceil(args.candidate_fraction * history_count)
            )

            sampled_key = key[:, :: args.key_sample_stride]
            key_covariance = torch.einsum(
                "hnd,hne->hde", sampled_key, sampled_key
            ) / float(sampled_key.shape[1])
            train_query = torch.stack(
                [
                    row["query"].to(device).float()[0, :, 0]
                    for row in records[: args.train_steps]
                ]
            ).reshape(args.train_steps, kv_heads, groups, head_dim)
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
            indexed_key = logscale16_int4_dequantize(
                torch.einsum("hnd,hdr->hnr", key, key_factor)
            )
            build_rows.append(
                {
                    "trace": str(trace_path),
                    "layer": layer,
                    "history_tokens": history_count,
                    "rank": args.rank,
                }
            )

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            for record in test_records:
                query = record["query"].to(device).float()[0, :, 0]
                grouped_query = query.reshape(kv_heads, groups, head_dim)
                projected_query = quantize_query_int8(
                    torch.einsum("hgd,hdr->hgr", grouped_query, query_factor)
                )
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                for head in range(kv_heads):
                    for group in range(groups):
                        query_head = head * groups + group
                        proxy_scores = indexed_key[head] @ projected_query[head, group]
                        global_candidates = torch.topk(
                            proxy_scores, candidate_count, sorted=False
                        ).indices
                        global_selected_local = torch.topk(
                            exact_scores[query_head, global_candidates],
                            keep_count,
                            sorted=False,
                        ).indices
                        global_selected = global_candidates[global_selected_local]
                        record_quality(
                            metrics["global_c6"],
                            global_selected,
                            oracle[query_head],
                            attention[query_head],
                            oracle_mass[query_head],
                        )

                        for block_size in block_sizes:
                            for quota_fraction in quota_fractions:
                                quota = max(1, math.ceil(quota_fraction * block_size))
                                candidates, emitted_count = local_quota_candidates(
                                    proxy_scores,
                                    block_size,
                                    quota,
                                    candidate_count,
                                )
                                selected_local = torch.topk(
                                    exact_scores[query_head, candidates],
                                    keep_count,
                                    sorted=False,
                                ).indices
                                selected = candidates[selected_local]
                                name = (
                                    f"local_b{block_size}_q{quota}"
                                    f"_c{args.candidate_fraction:g}"
                                )
                                record_quality(
                                    metrics[name],
                                    selected,
                                    oracle[query_head],
                                    attention[query_head],
                                    oracle_mass[query_head],
                                )
                                metrics[name]["local_emit_fraction"].append(
                                    emitted_count / history_count
                                )
                                metrics[name]["global_topk_input_fraction"].append(
                                    emitted_count / history_count
                                )

    config = dict(vars(args))
    config["trace_paths"] = [str(path) for path in args.trace_paths]
    config["output_path"] = str(args.output_path)
    result = {
        "config": config,
        "build": build_rows,
        "methods": {
            name: {metric: summarize(values) for metric, values in bucket.items()}
            for name, bucket in sorted(metrics.items())
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
