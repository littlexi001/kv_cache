#!/usr/bin/env python
"""Real-QKV audit of low-rank Value sketches for QKSieve tail recovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)


def parse_csv(specification: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(sorted({cast(x) for x in specification.split(",") if x.strip()}))
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def output_metrics(output: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(output - full)
            / torch.linalg.vector_norm(full).clamp_min(1.0e-12)
        ),
        "cosine": float(F.cosine_similarity(output.float(), full.float(), dim=0)),
    }


def value_basis(
    values: torch.Tensor,
    *,
    sample_stride: int,
    maximum_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = values[::sample_stride].float()
    mean = sample.mean(dim=0)
    centered = sample - mean
    covariance = centered.T @ centered / max(1, sample.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors[:, order[:maximum_rank]].contiguous()
    coefficients = (values.float() - mean) @ basis
    explained = eigenvalues[order[:maximum_rank]].clamp_min(0.0)
    total = eigenvalues.clamp_min(0.0).sum().clamp_min(1.0e-20)
    return mean, basis, coefficients, explained / total


def block_affine_quantize(
    coefficients: torch.Tensor,
    *,
    bits: int,
    block_size: int,
) -> torch.Tensor:
    if bits >= 16:
        return coefficients.to(torch.float16).float()
    if bits not in {2, 4, 8}:
        raise ValueError("Value sketch bits must be one of 2, 4, 8, 16")
    token_count, rank = coefficients.shape
    block_count = math.ceil(token_count / block_size)
    padded = F.pad(
        coefficients.float(),
        (0, 0, 0, block_count * block_size - token_count),
    ).reshape(block_count, block_size, rank)
    minimum = padded.amin(dim=1, keepdim=True)
    maximum = padded.amax(dim=1, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((maximum - minimum) / levels).clamp_min(1.0e-12)
    codes = torch.round((padded - minimum) / scale).clamp(0, levels)
    return (codes * scale + minimum).reshape(-1, rank)[:token_count]


def normalized_sparse_output(
    exact_scores: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    weights = torch.softmax(exact_scores[indices].float(), dim=0)
    return torch.sum(weights.unsqueeze(-1) * values[indices].float(), dim=0)


def residual_output(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    values: torch.Tensor,
    reconstructed_values: torch.Tensor,
    indices: torch.Tensor,
    *,
    shared_normalizer: bool,
) -> tuple[torch.Tensor, float]:
    selected = torch.zeros(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    selected[indices] = True
    tail = ~selected
    if shared_normalizer:
        selected_logits = exact_scores[indices].float()
        tail_logits = proxy_scores[tail].float()
        anchor = torch.maximum(selected_logits.max(), tail_logits.max())
        selected_weight = torch.exp(selected_logits - anchor)
        tail_weight = torch.exp(tail_logits - anchor)
        denominator = selected_weight.sum() + tail_weight.sum()
        output = (
            torch.sum(selected_weight.unsqueeze(-1) * values[indices].float(), dim=0)
            + torch.sum(
                tail_weight.unsqueeze(-1) * reconstructed_values[tail].float(),
                dim=0,
            )
        ) / denominator.clamp_min(1.0e-20)
        return output, float(selected_weight.sum() / denominator)

    probability = torch.softmax(proxy_scores.float(), dim=0)
    selected_mass = probability[indices].sum()
    sparse = normalized_sparse_output(exact_scores, values, indices)
    tail_output = torch.sum(
        probability[tail].unsqueeze(-1) * reconstructed_values[tail].float(),
        dim=0,
    )
    return selected_mass * sparse + tail_output, float(selected_mass)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--key_sample_stride", type=int, default=32)
    parser.add_argument("--value_sample_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--key_rate_budget", type=int, default=15)
    parser.add_argument("--fractions", default="0.01,0.02,0.04,0.06")
    parser.add_argument("--value_ranks", default="4,8,16,32")
    parser.add_argument("--value_bits", default="2,4,8,16")
    parser.add_argument("--value_scale_block", type=int, default=256)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    traces = tuple(Path(x) for x in args.traces.split(",") if x.strip())
    fractions = parse_csv(args.fractions, float)
    ranks = parse_csv(args.value_ranks, int)
    value_bits = parse_csv(args.value_bits, int)
    maximum_rank = max(ranks)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    for trace in traces:
        payload = torch.load(trace, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            layer = int(record["layer"])
            query = record["query"].to(device).float()[0, :, 0, :]
            key = record["key"].to(device).float()[0]
            value = record["value"].to(device).float()[0]
            scaling = float(record["scaling"])
            kv_heads, token_count, _ = key.shape
            groups = query.shape[0] // kv_heads

            for kv_head in range(kv_heads):
                head_key = key[kv_head]
                head_value = value[kv_head]
                calibration = query[kv_head * groups : (kv_head + 1) * groups]
                query_factor, key_factor, _ = qk_balanced_factors(
                    head_key[:: args.key_sample_stride],
                    calibration,
                    args.query_shrinkage,
                )
                key_coefficients = head_key @ key_factor
                projected_calibration = calibration @ query_factor
                bands = quantized_bands(key_coefficients, projected_calibration)
                key_distortion, _ = distortion_table(
                    key_coefficients,
                    projected_calibration,
                    ZERO_BIT_LEVELS,
                )
                allocation = allocate_bits(
                    key_distortion,
                    args.key_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                key_reconstruction = reconstruct(bands, allocation)
                value_mean, value_vectors, value_coefficients, explained = value_basis(
                    head_value,
                    sample_stride=args.value_sample_stride,
                    maximum_rank=maximum_rank,
                )
                reconstructions: dict[tuple[int, int], torch.Tensor] = {}
                for rank in ranks:
                    for bits in value_bits:
                        quantized = block_affine_quantize(
                            value_coefficients[:, :rank],
                            bits=bits,
                            block_size=args.value_scale_block,
                        )
                        reconstructions[(rank, bits)] = (
                            value_mean
                            + quantized @ value_vectors[:, :rank].T
                        )

                for group in range(groups):
                    query_head = kv_head * groups + group
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (head_key @ query[query_head]) * scaling
                    proxy_scores = (
                        key_reconstruction.float() @ approximate_query.float()
                    ) * scaling
                    full_probability = torch.softmax(exact_scores.float(), dim=0)
                    full_output = torch.sum(
                        full_probability.unsqueeze(-1) * head_value.float(),
                        dim=0,
                    )

                    for fraction in fractions:
                        keep = min(
                            token_count,
                            max(1, math.ceil(fraction * token_count)),
                        )
                        indices = torch.topk(proxy_scores, k=keep).indices
                        baseline = normalized_sparse_output(
                            exact_scores,
                            head_value,
                            indices,
                        )
                        rows.append(
                            {
                                "trace": trace.stem,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "fraction": fraction,
                                "method": "proxy_topk",
                                "value_rank": 0,
                                "value_bits": 0,
                                "value_explained_variance": 0.0,
                                "index_ratio_of_full_fp16_kv": 240.0 / 4096.0,
                                "selected_mass": float(
                                    full_probability[indices].sum()
                                ),
                                **output_metrics(baseline, full_output),
                            }
                        )
                        for rank in ranks:
                            for bits in value_bits:
                                reconstructed = reconstructions[(rank, bits)]
                                value_scale_bits = (
                                    32.0 * rank / args.value_scale_block
                                )
                                ratio = (
                                    240.0 + rank * bits + value_scale_bits
                                ) / 4096.0
                                for shared in (False, True):
                                    output, selected_mass = residual_output(
                                        exact_scores,
                                        proxy_scores,
                                        head_value,
                                        reconstructed,
                                        indices,
                                        shared_normalizer=shared,
                                    )
                                    rows.append(
                                        {
                                            "trace": trace.stem,
                                            "layer": layer,
                                            "kv_head": kv_head,
                                            "query_head": query_head,
                                            "fraction": fraction,
                                            "method": (
                                                "value_sketch_shared"
                                                if shared
                                                else "value_sketch_blend"
                                            ),
                                            "value_rank": rank,
                                            "value_bits": bits,
                                            "value_explained_variance": float(
                                                explained[:rank].sum()
                                            ),
                                            "index_ratio_of_full_fp16_kv": ratio,
                                            "selected_mass": selected_mass,
                                            **output_metrics(output, full_output),
                                        }
                                    )
            print(
                json.dumps(
                    {"trace": trace.stem, "layer": layer, "rows": len(rows)}
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_head.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                row["fraction"],
                row["value_rank"],
                row["value_bits"],
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for identity, items in sorted(grouped.items()):
        method, fraction, rank, bits = identity
        result: dict[str, Any] = {
            "method": method,
            "fraction": fraction,
            "value_rank": rank,
            "value_bits": bits,
            "cases": len(items),
        }
        for metric in (
            "relative_l2",
            "cosine",
            "selected_mass",
            "value_explained_variance",
            "index_ratio_of_full_fp16_kv",
        ):
            for statistic, value in summarize(
                float(item[metric]) for item in items
            ).items():
                result[f"{metric}_{statistic}"] = value
        summary.append(result)
    report = {
        "schema": "qksieve_value_sketch_residual_v1",
        "traces": [str(x) for x in traces],
        "quality_boundary": (
            "Offline real-QKV mechanism audit only. Value projection, "
            "quantization, and residual accumulation have no CUDA timing claim."
        ),
        "key_index_bits_per_token_per_kv_head": 240,
        "value_scale_metadata_bits_per_block_component": 32,
        "value_scale_block": args.value_scale_block,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
