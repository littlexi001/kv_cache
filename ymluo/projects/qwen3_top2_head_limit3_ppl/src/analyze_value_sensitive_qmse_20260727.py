from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from analyze_automatic_spectral_rate_allocation_20260727 import (
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    reconstruct,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    distortion_table_from_bands,
    metric_scale_quantize_band,
    qk_balanced_factors,
    softmax_fisher_cost,
)


METHODS = ("qscore_qmse", "softmax_fisher", "value_jacobian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare score-, Fisher-, and value-Jacobian-aware spectral "
            "rate allocation on held-out Q/K/V traces."
        )
    )
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--total_rate_budget", type=int, default=15)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--selected_fractions", default="0.01,0.04")
    return parser.parse_args()


def parse_fractions(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {
                float(item)
                for item in specification.split(",")
                if item.strip()
            }
        )
    )
    if not values or values[0] <= 0.0 or values[-1] >= 1.0:
        raise ValueError("selected fractions must be in (0, 1)")
    return values


def metric_quantized_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    bands: list[dict[int, torch.Tensor]] = []
    for band_index in range(GROUP_COUNT):
        start = band_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        bands.append(
            {
                bits: metric_scale_quantize_band(
                    key_band,
                    bits,
                    query_band,
                    metric_mode="full",
                )
                for bits in ZERO_BIT_LEVELS
            }
        )
    return bands


def full_attention_statistics(
    projected_queries: torch.Tensor,
    coefficients: torch.Tensor,
    values: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = (
        projected_queries.float() @ coefficients.float().transpose(0, 1)
    ) * scaling
    attention = torch.softmax(scores, dim=-1)
    output = attention @ values.float()
    value_square = values.float().square().sum(dim=-1).unsqueeze(0)
    output_square = output.square().sum(dim=-1, keepdim=True)
    cross = output @ values.float().transpose(0, 1)
    centered_value_square = (
        value_square + output_square - 2.0 * cross
    ).clamp_min(0.0)
    return attention, output, centered_value_square


def fisher_distortion_from_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    quantized_bands: list[dict[int, torch.Tensor]],
    attention: torch.Tensor,
    scaling: float,
    sample_stride: int,
) -> list[dict[int, torch.Tensor]]:
    sampled_indices = torch.arange(
        0,
        coefficients.shape[0],
        sample_stride,
        device=coefficients.device,
    )
    # Normalization is local to the sampled proxy objective.  Clone so the
    # full-attention tensor used by the other objectives remains unchanged.
    sampled_attention = attention[:, sampled_indices].clone()
    sampled_attention /= sampled_attention.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)
    table: list[dict[int, torch.Tensor]] = []
    for band_index in range(GROUP_COUNT):
        start = band_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[sampled_indices, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs: dict[int, torch.Tensor] = {}
        for bits in ZERO_BIT_LEVELS:
            reconstructed = quantized_bands[band_index][bits][
                sampled_indices
            ]
            score_error = (
                query_band
                @ (key_band - reconstructed).float().transpose(0, 1)
            ) * scaling
            costs[bits] = softmax_fisher_cost(
                score_error,
                sampled_attention,
            )
        table.append(costs)
    return table


def value_jacobian_distortion_from_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    quantized_bands: list[dict[int, torch.Tensor]],
    attention: torch.Tensor,
    centered_value_square: torch.Tensor,
    scaling: float,
    sample_stride: int,
) -> list[dict[int, torch.Tensor]]:
    sampled_indices = torch.arange(
        0,
        coefficients.shape[0],
        sample_stride,
        device=coefficients.device,
    )
    influence = (
        attention[:, sampled_indices].square()
        * centered_value_square[:, sampled_indices]
    )
    denominator = influence.sum().clamp_min(1.0e-20)
    table: list[dict[int, torch.Tensor]] = []
    for band_index in range(GROUP_COUNT):
        start = band_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_band = coefficients[sampled_indices, start:stop]
        query_band = calibration_queries[:, start:stop]
        costs: dict[int, torch.Tensor] = {}
        for bits in ZERO_BIT_LEVELS:
            reconstructed = quantized_bands[band_index][bits][
                sampled_indices
            ]
            score_error = (
                query_band
                @ (key_band - reconstructed).float().transpose(0, 1)
            ) * scaling
            costs[bits] = (
                influence * score_error.square()
            ).sum() / denominator
        table.append(costs)
    return table


def evaluate_candidate_output(
    exact_scores: torch.Tensor,
    approximate_scores: torch.Tensor,
    values: torch.Tensor,
    selected_fraction: float,
) -> dict[str, float]:
    token_count = int(exact_scores.numel())
    selected_count = min(
        token_count,
        max(1, math.ceil(selected_fraction * token_count)),
    )
    selected = torch.topk(
        approximate_scores,
        k=selected_count,
    ).indices
    exact_attention = torch.softmax(exact_scores.float(), dim=-1)
    full_output = exact_attention @ values.float()
    selected_scores = exact_scores[selected].float()
    selected_attention = torch.softmax(selected_scores, dim=-1)
    sparse_output = selected_attention @ values[selected].float()
    output_error = sparse_output - full_output
    full_norm = full_output.norm().clamp_min(1.0e-12)
    true_top = torch.topk(exact_scores, k=selected_count).indices
    selected_mask = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=exact_scores.device,
    )
    selected_mask[selected] = True
    return {
        "selected_count": selected_count,
        "topk_recall": float(
            selected_mask[true_top].float().mean().item()
        ),
        "attention_mass": float(exact_attention[selected].sum().item()),
        "output_relative_l2": float(
            (output_error.norm() / full_norm).item()
        ),
        "output_cosine": float(
            F.cosine_similarity(
                sparse_output.unsqueeze(0),
                full_output.unsqueeze(0),
            ).item()
        ),
        "output_mse": float(output_error.square().mean().item()),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction"]))
        ].append(row)
    summary = []
    for (method, fraction), items in sorted(grouped.items()):
        output: dict[str, Any] = {
            "method": method,
            "selected_fraction": fraction,
            "cases": len(items),
        }
        for field in (
            "topk_recall",
            "attention_mass",
            "output_relative_l2",
            "output_cosine",
            "output_mse",
        ):
            tensor = torch.tensor(
                [float(item[field]) for item in items],
                dtype=torch.float64,
            )
            output[f"{field}_mean"] = float(tensor.mean().item())
            output[f"{field}_p50"] = float(
                torch.quantile(tensor, 0.50).item()
            )
            output[f"{field}_p90"] = float(
                torch.quantile(tensor, 0.90).item()
            )
        summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.selected_fractions)
    payload = torch.load(
        args.trace_path,
        map_location="cpu",
        weights_only=False,
    )
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda record: int(record["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(f"layer {layer} has no held-out query")
        state_record = next(
            (
                record
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if state_record is None or state_record.get("value") is None:
            raise ValueError(
                f"layer {layer} requires captured K and V tensors"
            )
        key = state_record["key"].to(device).float()[0, :, :-1]
        value = state_record["value"].to(device).float()[0, :, :-1]
        scaling = float(state_record["scaling"])
        calibration = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )
        kv_head_count = int(key.shape[0])
        query_head_count = int(calibration.shape[1])
        if query_head_count % kv_head_count != 0:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_head_count // kv_head_count

        for kv_head in range(kv_head_count):
            head_key = key[kv_head]
            head_value = value[kv_head]
            head_calibration = calibration[
                :,
                kv_head * groups : (kv_head + 1) * groups,
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, _ = qk_balanced_factors(
                head_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            quantized_bands = metric_quantized_bands(
                coefficients,
                projected_calibration,
            )
            attention, _, centered_value_square = (
                full_attention_statistics(
                    projected_calibration,
                    coefficients,
                    head_value,
                    scaling,
                )
            )
            distortion_tables = {
                "qscore_qmse": distortion_table_from_bands(
                    coefficients,
                    projected_calibration,
                    quantized_bands,
                ),
                "softmax_fisher": fisher_distortion_from_bands(
                    coefficients,
                    projected_calibration,
                    quantized_bands,
                    attention,
                    scaling,
                    args.sample_stride,
                ),
                "value_jacobian": value_jacobian_distortion_from_bands(
                    coefficients,
                    projected_calibration,
                    quantized_bands,
                    attention,
                    centered_value_square,
                    scaling,
                    args.sample_stride,
                ),
            }
            reconstructions: dict[str, torch.Tensor] = {}
            for method, table in distortion_tables.items():
                allocation = allocate_bits(
                    table,
                    args.total_rate_budget,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                reconstructions[method] = reconstruct(
                    quantized_bands,
                    allocation,
                )
                allocations.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(
                            str(bits) for bits in allocation
                        ),
                        "code_bits": 16 * sum(allocation),
                        "scale_bits": 16
                        * sum(bits > 0 for bits in allocation),
                    }
                )

            for record in layer_records[args.calibration_steps :]:
                step = int(record["step"])
                heldout = record["query"].to(device).float()[
                    0,
                    kv_head * groups : (kv_head + 1) * groups,
                    0,
                    :,
                ]
                for query_group, raw_query in enumerate(heldout):
                    projected_query = raw_query @ query_factor
                    proxy_query = query_int8(projected_query)
                    exact_scores = (
                        projected_query @ coefficients.transpose(0, 1)
                    ) * scaling
                    for method in METHODS:
                        approximate_scores = (
                            proxy_query
                            @ reconstructions[method].transpose(0, 1)
                        ) * scaling
                        for fraction in fractions:
                            metrics = evaluate_candidate_output(
                                exact_scores,
                                approximate_scores,
                                head_value,
                                fraction,
                            )
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "heldout_step": step,
                                    "kv_head": kv_head,
                                    "query_group": query_group,
                                    "method": method,
                                    "selected_fraction": fraction,
                                    **metrics,
                                }
                            )

    summary_rows = summarize_rows(rows)
    allocation_histograms: dict[str, dict[str, int]] = {}
    for method in METHODS:
        allocation_histograms[method] = dict(
            Counter(
                str(row["allocation"])
                for row in allocations
                if row["method"] == method
            )
        )
    summary = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "sample_stride": args.sample_stride,
            "calibration_steps": args.calibration_steps,
            "total_rate_budget": args.total_rate_budget,
            "query_shrinkage": args.query_shrinkage,
            "selected_fractions": fractions,
        },
        "allocation_histograms": allocation_histograms,
        "summary": summary_rows,
        "row_count": len(rows),
        "allocation_count": len(allocations),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocations)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
