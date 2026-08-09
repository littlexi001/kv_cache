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
    selected_quality,
    summarize,
)
from analyze_qk_metric_lowrank import qk_metric_factors
from analyze_qkmetric_microblock_frontier import (
    block_candidates,
    microblock_summaries,
)


def quantized_summary(summary: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mean_scale = (summary["mean"].abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(
        1.0e-8
    )
    variance_scale = (summary["variance"].amax(dim=-1, keepdim=True) / 255.0).clamp_min(
        1.0e-8
    )
    mean = torch.round(summary["mean"] / mean_scale).clamp(-127, 127) * mean_scale
    variance = (
        torch.round(summary["variance"] / variance_scale).clamp(0, 255)
        * variance_scale
    )
    return mean, variance


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
        description="Offline quality frontier for a 32D-to-48D QK-Metric cascade."
    )
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="16")
    parser.add_argument("--base_rank", type=int, default=32)
    parser.add_argument("--full_rank", type=int, default=48)
    parser.add_argument(
        "--summary_rank",
        type=int,
        default=0,
        help="Dimensions used by block summaries; zero means full_rank.",
    )
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--outer_fractions", default="0.16,0.20,0.24")
    parser.add_argument("--middle_fractions", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--candidate_fractions", default="0.03,0.04,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    if args.base_rank >= args.full_rank:
        raise ValueError("base_rank must be smaller than full_rank")
    if args.base_rank % 16 or args.full_rank % 16:
        raise ValueError("ranks must be multiples of 16")
    summary_rank = args.full_rank if args.summary_rank == 0 else args.summary_rank
    if summary_rank not in (args.base_rank, args.full_rank):
        raise ValueError("summary_rank must equal base_rank or full_rank")
    layers = {int(item) for item in args.layers.split(",") if item.strip()}
    outer_fractions = parse_floats(args.outer_fractions)
    middle_fractions = parse_floats(args.middle_fractions)
    candidate_fractions = parse_floats(args.candidate_fractions)
    if max(middle_fractions) >= min(outer_fractions):
        raise ValueError("middle fractions must be smaller than every outer fraction")
    if max(candidate_fractions) > min(middle_fractions):
        raise ValueError("candidate fractions must not exceed middle fractions")

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
            needed = args.test_start_step + args.test_steps
            if len(records) < needed:
                raise ValueError(f"{trace_path} layer {layer} has too few queries")
            key_record = next(row for row in records if row.get("key") is not None)
            key = key_record["key"].to(device).float()[0, :, :-1]
            scaling = float(key_record["scaling"])
            kv_heads, history_count, head_dim = key.shape
            query_heads = int(records[0]["query"].shape[1])
            groups = query_heads // kv_heads
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
                key_covariance, regularized_query_covariance, args.full_rank
            )
            projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
            indexed_key = logscale16_int4_dequantize(projected_key)
            summary = microblock_summaries(
                projected_key[..., :summary_rank],
                args.block_size,
                history_count,
            )
            mean_q8, variance_q8 = quantized_summary(summary)
            expected_multiplier = torch.sqrt(
                2.0 * summary["lengths"].float().clamp_min(2.0).log()
            )
            block_count = int(summary["lengths"].numel())
            full_index_bytes = history_count * args.full_rank / 2 + history_count * 4
            summary_bytes = block_count * (2 * summary_rank + 4)
            build_rows.append(
                {
                    "trace": str(trace_path),
                    "layer": layer,
                    "history_tokens": history_count,
                    "block_count": block_count,
                    "int4_index_fraction_of_full_kv_bf16": full_index_bytes
                    / (history_count * head_dim * 4),
                    "q8_summary_fraction_of_full_kv_bf16": summary_bytes
                    / (history_count * head_dim * 4),
                }
            )

            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]
            for record in test_records:
                query = record["query"].to(device).float()[0, :, 0]
                grouped_query = query.reshape(kv_heads, groups, head_dim)
                projected_query = torch.einsum(
                    "hgd,hdr->hgr", grouped_query, query_factor
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
                        projected = projected_query[head, group]
                        base_query = projected[: args.base_rank]
                        summary_query = projected[:summary_rank]
                        center = mean_q8[head] @ summary_query
                        sigma = torch.sqrt(
                            variance_q8[head] @ summary_query.square()
                        ).clamp_min(0.0)
                        block_scores = center + expected_multiplier * sigma
                        base_scores = indexed_key[head, :, : args.base_rank] @ base_query
                        full_scores = indexed_key[head] @ projected

                        for outer_fraction in outer_fractions:
                            outer, _ = block_candidates(
                                block_scores,
                                summary["lengths"],
                                math.ceil(outer_fraction * history_count),
                                args.block_size,
                                history_count,
                            )
                            for middle_fraction in middle_fractions:
                                middle_count = math.ceil(
                                    middle_fraction * history_count
                                )
                                middle_local = torch.topk(
                                    base_scores[outer],
                                    min(middle_count, outer.numel()),
                                    sorted=False,
                                ).indices
                                middle = outer[middle_local]
                                for candidate_fraction in candidate_fractions:
                                    candidate_count = max(
                                        keep_count,
                                        math.ceil(candidate_fraction * history_count),
                                    )
                                    candidate_local = torch.topk(
                                        full_scores[middle],
                                        min(candidate_count, middle.numel()),
                                        sorted=False,
                                    ).indices
                                    candidates = middle[candidate_local]
                                    selected_local = torch.topk(
                                        exact_scores[query_head, candidates],
                                        min(keep_count, candidates.numel()),
                                        sorted=False,
                                    ).indices
                                    selected = candidates[selected_local]
                                    name = (
                                        f"q8_b{args.block_size}_r{args.base_rank}to"
                                        f"{args.full_rank}_o{outer_fraction:g}"
                                        f"_m{middle_fraction:g}_c{candidate_fraction:g}"
                                    )
                                    record_quality(
                                        metrics[name],
                                        selected,
                                        oracle[query_head],
                                        attention[query_head],
                                        oracle_mass[query_head],
                                    )
                                    metrics[name]["outer_fraction"].append(
                                        outer.numel() / history_count
                                    )
                                    metrics[name]["middle_fraction"].append(
                                        middle.numel() / history_count
                                    )
                                    metrics[name]["candidate_fraction"].append(
                                        candidates.numel() / history_count
                                    )
                                    dimension_token_cost = (
                                        args.base_rank * outer.numel()
                                        + (args.full_rank - args.base_rank)
                                        * middle.numel()
                                        + head_dim * candidates.numel()
                                    ) / history_count
                                    metrics[name]["dimension_token_cost"].append(
                                        float(dimension_token_cost)
                                    )

    config = dict(vars(args))
    config["trace_paths"] = [str(path) for path in args.trace_paths]
    config["output_path"] = str(args.output_path)
    result = {
        "config": config,
        "build": build_rows,
        "methods": {
            name: {metric: summarize(values) for metric, values in bucket.items()}
            for name, bucket in metrics.items()
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
