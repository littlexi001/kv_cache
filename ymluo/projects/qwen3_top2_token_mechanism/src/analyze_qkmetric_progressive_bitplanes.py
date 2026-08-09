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


def symmetric_int2_dequantize(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 16:
        raise ValueError("INT2 rank must be divisible by 16")
    shape = values.shape
    bands = values.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 3.0
    normalized = bands / scale
    levels = torch.tensor(
        (-3.0, -1.0, 1.0, 3.0), device=values.device, dtype=torch.float32
    )
    nearest = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return (levels[nearest] * scale).reshape(shape)


def logscale16_int4_codes(
    projected_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = projected_key.shape
    bands = projected_key.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    exact_scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    base_scale = exact_scale.amax(dim=-2, keepdim=True)
    exponent = torch.round(
        torch.log2(base_scale / exact_scale).clamp_min(0.0) / 0.25
    ).clamp(0, 15)
    scale = base_scale * torch.exp2(-0.25 * exponent)
    codes = torch.round(bands / scale).clamp(-7, 7)
    return codes, scale


def nested_coarse_dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    short_bin: int,
) -> torch.Tensor:
    if not 0 <= short_bin < 4:
        raise ValueError("short_bin must identify one of four ordered code bins")
    sizes = [4, 4, 4, 4]
    sizes[short_bin] = 3
    boundaries = [-7]
    for size in sizes:
        boundaries.append(boundaries[-1] + size)
    coarse = torch.empty_like(codes)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        representative = 0.5 * (lower + upper - 1)
        coarse = torch.where(
            (codes >= lower) & (codes < upper),
            torch.full_like(coarse, representative),
            coarse,
        )
    return (coarse * scale).reshape(*codes.shape[:-2], codes.shape[-2] * 16)


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


def exact_select(
    exact_scores: torch.Tensor,
    candidates: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    local = torch.topk(
        exact_scores[candidates], min(keep_count, candidates.numel()), sorted=False
    ).indices
    return candidates[local]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality frontier for progressive 2+2-bit QK-Metric retrieval."
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
    parser.add_argument("--outer_fractions", default="0.08,0.10,0.12,0.16")
    parser.add_argument("--candidate_fraction", type=float, default=0.06)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--residual_storage_fractions", default="0.25,0.50,1.0")
    args = parser.parse_args()

    if args.rank % 16:
        raise ValueError("rank must be divisible by 16")
    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    outer_fractions = parse_floats(args.outer_fractions)
    residual_storage_fractions = parse_floats(args.residual_storage_fractions)
    if min(outer_fractions) < args.candidate_fraction:
        raise ValueError("outer fractions must cover the final candidate fraction")
    if not all(0.0 < value <= 1.0 for value in residual_storage_fractions):
        raise ValueError("residual storage fractions must be in (0, 1]")

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
            projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
            int4_codes, int4_scale = logscale16_int4_codes(projected_key)
            int4_key = (int4_codes * int4_scale).reshape_as(projected_key)
            coarse_key = symmetric_int2_dequantize(projected_key)
            residual_key = symmetric_int2_dequantize(projected_key - coarse_key)
            reconstructed_key = coarse_key + residual_key
            nested_coarse_keys = {
                short_bin: nested_coarse_dequantize(
                    int4_codes, int4_scale, short_bin
                )
                for short_bin in range(4)
            }

            residual_energy = residual_key.square().sum(dim=-1)
            projected_energy = projected_key.square().sum(dim=-1)
            masks: dict[str, dict[float, torch.Tensor]] = defaultdict(dict)
            for priority_name, priority in (
                ("residual", residual_energy),
                ("norm", projected_energy),
            ):
                for storage_fraction in residual_storage_fractions:
                    stored_count = max(1, math.ceil(storage_fraction * history_count))
                    stored = torch.topk(
                        priority, stored_count, dim=-1, sorted=False
                    ).indices
                    mask = torch.zeros(
                        kv_heads,
                        history_count,
                        dtype=torch.bool,
                        device=device,
                    )
                    mask.scatter_(1, stored, True)
                    masks[priority_name][storage_fraction] = mask

            full_kv_bytes = history_count * head_dim * 4
            scale_bytes_per_token = 4
            int4_bytes = history_count * (args.rank / 2 + scale_bytes_per_token)
            coarse_bytes = history_count * (args.rank / 4 + scale_bytes_per_token)
            build_rows.append(
                {
                    "trace": str(trace_path),
                    "layer": layer,
                    "history_tokens": history_count,
                    "int4_fraction_of_full_kv_bf16": int4_bytes / full_kv_bytes,
                    "int2_fraction_of_full_kv_bf16": coarse_bytes / full_kv_bytes,
                    "progressive_full_residual_fraction_of_full_kv_bf16": (
                        2 * coarse_bytes / full_kv_bytes
                    ),
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
                projected_query = quantize_query_int8(projected_query)
                exact_scores = grouped_scores(query, key) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                oracle = torch.topk(
                    exact_scores, keep_count, dim=-1, sorted=False
                ).indices
                oracle_mass = torch.gather(attention, -1, oracle).sum(dim=-1)

                for head in range(kv_heads):
                    for group in range(groups):
                        query_head = head * groups + group
                        q = projected_query[head, group]
                        coarse_scores = coarse_key[head] @ q
                        int4_scores = int4_key[head] @ q
                        reconstructed_scores = reconstructed_key[head] @ q

                        for name, scores in (
                            ("int4", int4_scores),
                            ("int2", coarse_scores),
                            ("int2_plus_int2", reconstructed_scores),
                        ):
                            candidates = torch.topk(
                                scores, candidate_count, sorted=False
                            ).indices
                            selected = exact_select(
                                exact_scores[query_head], candidates, keep_count
                            )
                            record_quality(
                                metrics[f"{name}_c{args.candidate_fraction:g}"],
                                selected,
                                oracle[query_head],
                                attention[query_head],
                                oracle_mass[query_head],
                            )

                        for short_bin, nested_key in nested_coarse_keys.items():
                            nested_scores = nested_key[head] @ q
                            for outer_fraction in outer_fractions:
                                outer_count = max(
                                    candidate_count,
                                    math.ceil(outer_fraction * history_count),
                                )
                                outer = torch.topk(
                                    nested_scores, outer_count, sorted=False
                                ).indices
                                direct_selected = exact_select(
                                    exact_scores[query_head], outer, keep_count
                                )
                                direct_name = (
                                    f"nested_direct_shortbin{short_bin}"
                                    f"_c{outer_fraction:g}"
                                )
                                record_quality(
                                    metrics[direct_name],
                                    direct_selected,
                                    oracle[query_head],
                                    attention[query_head],
                                    oracle_mass[query_head],
                                )
                                metrics[direct_name]["normalized_bit_scan"].append(
                                    0.5
                                )
                                metrics[direct_name]["exact_qk_fraction"].append(
                                    outer.numel() / history_count
                                )
                                refined_scores = int4_scores[outer]
                                local = torch.topk(
                                    refined_scores,
                                    min(candidate_count, outer.numel()),
                                    sorted=False,
                                ).indices
                                candidates = outer[local]
                                selected = exact_select(
                                    exact_scores[query_head], candidates, keep_count
                                )
                                name = (
                                    f"nested_int4_shortbin{short_bin}"
                                    f"_o{outer_fraction:g}"
                                    f"_c{args.candidate_fraction:g}"
                                )
                                record_quality(
                                    metrics[name],
                                    selected,
                                    oracle[query_head],
                                    attention[query_head],
                                    oracle_mass[query_head],
                                )
                                metrics[name]["normalized_bit_scan"].append(
                                    0.5 + 0.5 * outer.numel() / history_count
                                )
                                metrics[name]["stored_bits_per_dimension"].append(
                                    4.0
                                )

                        for outer_fraction in outer_fractions:
                            outer_count = max(
                                candidate_count,
                                math.ceil(outer_fraction * history_count),
                            )
                            outer = torch.topk(
                                coarse_scores, outer_count, sorted=False
                            ).indices
                            for priority_name, priority_masks in masks.items():
                                for storage_fraction, mask in priority_masks.items():
                                    refined_key = coarse_key[head, outer] + (
                                        residual_key[head, outer]
                                        * mask[head, outer].unsqueeze(-1)
                                    )
                                    refined_scores = refined_key @ q
                                    local = torch.topk(
                                        refined_scores,
                                        min(candidate_count, outer.numel()),
                                        sorted=False,
                                    ).indices
                                    candidates = outer[local]
                                    selected = exact_select(
                                        exact_scores[query_head], candidates, keep_count
                                    )
                                    name = (
                                        f"progressive_{priority_name}_store"
                                        f"{storage_fraction:g}_o{outer_fraction:g}"
                                        f"_c{args.candidate_fraction:g}"
                                    )
                                    record_quality(
                                        metrics[name],
                                        selected,
                                        oracle[query_head],
                                        attention[query_head],
                                        oracle_mass[query_head],
                                    )
                                    metrics[name]["normalized_bit_scan"].append(
                                        0.5 + outer.numel() / history_count * storage_fraction * 0.5
                                    )
                                    metrics[name]["stored_bits_per_dimension"].append(
                                        2.0 + 2.0 * storage_fraction
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
