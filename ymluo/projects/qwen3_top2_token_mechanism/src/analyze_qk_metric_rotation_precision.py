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
from analyze_qk_metric_lowrank import qk_metric_factors


def normalized_hadamard(dimension: int, device: torch.device) -> torch.Tensor:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    matrix = torch.ones(1, 1, device=device)
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(dimension)


def band_rotation(
    rank: int,
    device: torch.device,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    if rank % 16:
        raise ValueError("rank must be a multiple of 16")
    hadamard = normalized_hadamard(16, device)
    blocks = []
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    for _ in range(rank // 16):
        if seed is None:
            blocks.append(hadamard)
        else:
            signs = torch.randint(
                0, 2, (16,), device=device, generator=generator
            ).float().mul_(2.0).sub_(1.0)
            blocks.append(signs.diag() @ hadamard)
    return torch.block_diag(*blocks)


def uniform4_int2_dequantize(value: torch.Tensor) -> torch.Tensor:
    bands = value.float().reshape(*value.shape[:-1], -1, 16)
    scale = bands.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    code = torch.round((bands / scale + 1.0) * 1.5).clamp(0, 3)
    return ((code / 1.5 - 1.0) * scale).flatten(-2)


def mixed_precision_dequantize(
    value: torch.Tensor, int4_dimensions: int
) -> torch.Tensor:
    if not 0 < int4_dimensions < value.shape[-1] or int4_dimensions % 16:
        raise ValueError("INT4 prefix must be chunk aligned and within the rank")
    return torch.cat(
        (
            logscale16_int4_dequantize(value[..., :int4_dimensions]),
            uniform4_int2_dequantize(value[..., int4_dimensions:]),
        ),
        dim=-1,
    )


def choose_head_rotations(
    projected_key: torch.Tensor,
    projected_query: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    head_count, _, rank = projected_key.shape
    exact = torch.einsum("thgr,hnr->thgn", projected_query, projected_key)
    losses = []
    for rotation in rotations:
        rotated_key = projected_key @ rotation
        rotated_query = projected_query @ rotation
        indexed_key = logscale16_int4_dequantize(rotated_key)
        indexed_query = quantize_query_int8(rotated_query)
        approximate = torch.einsum(
            "thgr,hnr->thgn", indexed_query, indexed_key
        )
        losses.append((approximate - exact).square().mean(dim=(0, 2, 3)))
    choices = torch.stack(losses).argmin(dim=0)
    return rotations[choices].reshape(head_count, rank, rank)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--ranks", default="32,48,64")
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--test_steps", type=int, default=8)
    parser.add_argument("--test_start_step", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--rotation_seeds", type=int, default=8)
    parser.add_argument("--candidate_fractions", default="0.04,0.05,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    args = parser.parse_args()

    layers = {int(value) for value in args.layers.split(",") if value.strip()}
    if not layers or min(layers) < 0:
        raise ValueError("layers must contain nonnegative integers")
    ranks = parse_ints(args.ranks)
    candidate_fractions = parse_floats(args.candidate_fractions)
    if any(rank % 16 or rank > 128 for rank in ranks):
        raise ValueError("ranks must be multiples of 16 and no larger than 128")
    if args.test_start_step < args.train_steps:
        raise ValueError("test queries must not overlap calibration queries")
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
            test_records = records[
                args.test_start_step : args.test_start_step + args.test_steps
            ]

            for rank in ranks:
                query_factor, key_factor = qk_metric_factors(
                    key_covariance, regularized_query_covariance, rank
                )
                projected_key = torch.einsum("hnd,hdr->hnr", key, key_factor)
                projected_train_query = torch.einsum(
                    "thgd,hdr->thgr", train_query, query_factor
                )
                base_rotations = torch.stack(
                    [
                        band_rotation(rank, device, seed=seed)
                        for seed in range(args.rotation_seeds)
                    ]
                )
                selected_rotation = choose_head_rotations(
                    projected_key[:, :: args.key_sample_stride],
                    projected_train_query,
                    base_rotations,
                )
                transforms = {
                    "identity": torch.eye(rank, device=device)
                    .expand(head_count, -1, -1),
                    "hadamard": band_rotation(rank, device).expand(
                        head_count, -1, -1
                    ),
                    f"calibrated_hadamard{args.rotation_seeds}": selected_rotation,
                }
                for transform_name, transform in transforms.items():
                    transformed_key = torch.einsum(
                        "hnr,hrs->hns", projected_key, transform
                    )
                    key_variants = {
                        "int4": logscale16_int4_dequantize(transformed_key)
                    }
                    if rank >= 32:
                        key_variants["int4prefix16_int2tail"] = (
                            mixed_precision_dequantize(transformed_key, 16)
                        )
                    if rank >= 48:
                        key_variants["int4prefix32_int2tail"] = (
                            mixed_precision_dequantize(transformed_key, 32)
                        )

                    for record in test_records:
                        query = record["query"].to(device).float()[0, :, 0]
                        exact_scores = grouped_scores(query, key) * scaling
                        attention = torch.softmax(exact_scores, dim=-1)
                        oracle = torch.topk(
                            exact_scores, keep_count, dim=-1, sorted=False
                        ).indices
                        oracle_mass = torch.gather(
                            attention, -1, oracle
                        ).sum(dim=-1)
                        projected_query = torch.einsum(
                            "hgd,hdr->hgr",
                            query.reshape(head_count, groups, head_dim),
                            query_factor,
                        )
                        transformed_query = torch.einsum(
                            "hgr,hrs->hgs", projected_query, transform
                        ).reshape(query_heads, rank)
                        indexed_query = quantize_query_int8(transformed_query)
                        for precision, indexed_key in key_variants.items():
                            proxy_scores = grouped_scores(
                                indexed_query, indexed_key
                            ) * scaling
                            for fraction in candidate_fractions:
                                candidate_count = max(
                                    keep_count,
                                    math.ceil(fraction * history_count),
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
                                name = (
                                    f"qkmetric_r{rank}_{transform_name}_{precision}"
                                    f"_candidate{fraction:g}"
                                )
                                metrics[name]["top2_recall"].extend(
                                    recall.cpu().tolist()
                                )
                                metrics[name][
                                    "top2_attention_mass_recall"
                                ].extend(mass.cpu().tolist())
                    del transformed_key, key_variants
                del projected_key
            head_cases += args.test_steps * query_heads
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "config": vars(args)
        | {
            "trace_paths": [str(path) for path in args.trace_paths],
            "layers": sorted(layers),
            "ranks": ranks,
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
