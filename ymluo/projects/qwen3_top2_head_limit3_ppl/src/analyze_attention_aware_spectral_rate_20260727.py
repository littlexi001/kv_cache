from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
    reconstruct,
)
from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    query_int8,
    selection_metrics,
)


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(
        0, length - 1, steps=count, device=device
    ).round().long().unique()


def oas_shrinkage(values: torch.Tensor) -> float:
    """Closed-form Oracle Approximating Shrinkage for a second moment."""
    sample_count, dimensions = values.shape
    second_moment = values.transpose(0, 1) @ values / max(1, sample_count)
    trace = second_moment.trace()
    trace_square = second_moment.square().sum()
    numerator = (
        (1.0 - 2.0 / dimensions) * trace_square
        + trace.square()
    )
    denominator = (
        sample_count + 1.0 - 2.0 / dimensions
    ) * (
        trace_square
        - trace.square() / dimensions
    )
    if float(denominator.item()) <= 1.0e-20:
        return 1.0
    return float(
        torch.clamp(numerator / denominator, 0.0, 1.0).item()
    )


def attention_aware_distortion(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    calibration_scaling: torch.Tensor,
    sample_index: torch.Tensor,
) -> tuple[
    list[dict[int, torch.Tensor]],
    list[dict[int, torch.Tensor]],
    list[dict[int, torch.Tensor]],
]:
    sampled_coefficients = coefficients.index_select(0, sample_index)
    exact_scores = (
        calibration_queries @ sampled_coefficients.transpose(0, 1)
    ) * calibration_scaling[:, None]
    probabilities = torch.softmax(exact_scores, dim=-1)
    fisher_tables: list[dict[int, torch.Tensor]] = []
    attention_mse_tables: list[dict[int, torch.Tensor]] = []
    boundary_tables: list[dict[int, torch.Tensor]] = []

    candidate_count = max(1, math.ceil(0.06 * sample_index.numel()))
    boundary = torch.topk(
        exact_scores, k=candidate_count, dim=-1
    ).values[:, -1:]
    centered = exact_scores - boundary
    local_scale = torch.quantile(
        centered.abs(), 0.10, dim=-1, keepdim=True
    ).clamp_min(1.0e-4)
    sigmoid = torch.sigmoid(centered / local_scale)
    boundary_weight = sigmoid * (1.0 - sigmoid)
    boundary_weight /= boundary_weight.sum(dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )

    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        sampled_band = sampled_coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        fisher_costs: dict[int, torch.Tensor] = {}
        attention_mse_costs: dict[int, torch.Tensor] = {}
        boundary_costs: dict[int, torch.Tensor] = {}
        for bits in ZERO_BIT_LEVELS:
            residual = sampled_band - quantize_band(sampled_band, bits)
            score_error = (
                query_band @ residual.transpose(0, 1)
            ) * calibration_scaling[:, None]
            weighted_mean = (probabilities * score_error).sum(
                dim=-1, keepdim=True
            )
            fisher_costs[bits] = (
                probabilities * (score_error - weighted_mean).square()
            ).sum(dim=-1).mean()
            attention_mse_costs[bits] = (
                probabilities * score_error.square()
            ).sum(dim=-1).mean()
            boundary_costs[bits] = (
                boundary_weight * score_error.square()
            ).sum(dim=-1).mean()
        fisher_tables.append(fisher_costs)
        attention_mse_tables.append(attention_mse_costs)
        boundary_tables.append(boundary_costs)
    return fisher_tables, attention_mse_tables, boundary_tables


def aggregate(
    rows: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allocation_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allocations:
        allocation_by_method[str(row["method"])].append(row)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    output = []
    for (method, fraction), items in sorted(grouped.items()):
        method_allocations = allocation_by_method[method]
        total_bits = sum(
            int(item["total_index_bits"]) for item in method_allocations
        ) / len(method_allocations)
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "total_index_bits_mean": total_bits,
            "index_ratio_of_full_kv": total_bits / FULL_KV_BITS,
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
        ):
            for name, value in summarize(
                float(item[field]) for item in items
            ).items():
                result[f"{field}_{name}"] = value
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare query-MSE, softmax-Fisher, attention-weighted, and "
            "top-k-boundary spectral rate allocation on held-out Q/K traces."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_key_count", type=int, default=256)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--total_rate_budgets", default="12,13,14,15,16")
    parser.add_argument("--selected_fractions", default="0.02,0.04,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    budgets = parse_ints(args.total_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    shrinkage_rows: list[dict[str, Any]] = []
    calibration_seconds = 0.0
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda item: int(item["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(f"layer {layer} has no held-out query")
        raw_key = next(
            (
                record.get("key")
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no key tensor")
        all_key = raw_key.to(device).float()[0]
        history_count = int(all_key.shape[1]) - 1
        key = all_key[:, :history_count]
        kv_heads = int(key.shape[0])
        query_heads = int(layer_records[0]["query"].shape[1])
        groups = query_heads // kv_heads
        calibration_queries = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )
        calibration_scaling = torch.tensor(
            [
                float(record["scaling"])
                for record in layer_records[: args.calibration_steps]
                for _ in range(groups)
            ],
            device=device,
        )
        prepared = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        layer_start = time.perf_counter()
        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            basis, _ = covariance_basis(
                head_key[:: args.sample_stride]
            )
            coefficients = head_key @ basis
            projected_calibration = (
                calibration_queries[
                    :, kv_head * groups : (kv_head + 1) * groups
                ]
                @ basis
            ).reshape(-1, coefficients.shape[-1])
            sample_index = sampled_indices(
                history_count, args.calibration_key_count, device
            )
            (
                fisher_distortion,
                attention_mse_distortion,
                boundary_distortion,
            ) = attention_aware_distortion(
                coefficients,
                projected_calibration,
                calibration_scaling,
                sample_index,
            )
            # Raw QK MSE is retained as the deployment baseline.
            qmse_distortion = []
            oas_distortion = []
            rank_ridge_distortion = []
            oas_alpha = oas_shrinkage(projected_calibration)
            rank_ridge_alpha = coefficients.shape[-1] / (
                coefficients.shape[-1]
                + projected_calibration.shape[0]
            )
            isotropic_query_variance = float(
                projected_calibration.square().mean().item()
            )
            for group_index in range(GROUP_COUNT):
                start = group_index * GROUP_SIZE
                stop = start + GROUP_SIZE
                sampled_band = coefficients.index_select(
                    0, sample_index
                )[:, start:stop]
                query_band = projected_calibration[:, start:stop]
                costs = {}
                oas_costs = {}
                ridge_costs = {}
                for bits in ZERO_BIT_LEVELS:
                    residual = sampled_band - quantize_band(
                        sampled_band, bits
                    )
                    error = query_band @ residual.transpose(0, 1)
                    qmse_cost = error.square().mean()
                    isotropic_cost = (
                        residual.square().sum(dim=-1).mean()
                        * isotropic_query_variance
                    )
                    costs[bits] = qmse_cost
                    oas_costs[bits] = (
                        (1.0 - oas_alpha) * qmse_cost
                        + oas_alpha * isotropic_cost
                    )
                    ridge_costs[bits] = (
                        (1.0 - rank_ridge_alpha) * qmse_cost
                        + rank_ridge_alpha * isotropic_cost
                    )
                qmse_distortion.append(costs)
                oas_distortion.append(oas_costs)
                rank_ridge_distortion.append(ridge_costs)
            shrinkage_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "query_samples": projected_calibration.shape[0],
                    "dimensions": projected_calibration.shape[1],
                    "oas_alpha": oas_alpha,
                    "rank_ridge_alpha": rank_ridge_alpha,
                    "isotropic_query_variance": (
                        isotropic_query_variance
                    ),
                }
            )

            distortions = {
                "qmse": qmse_distortion,
                "qmse_oas": oas_distortion,
                "qmse_rankridge": rank_ridge_distortion,
                "fisher": fisher_distortion,
                "attnmse": attention_mse_distortion,
                "boundary": boundary_distortion,
            }
            quantized_bands = []
            for group_index in range(GROUP_COUNT):
                start = group_index * GROUP_SIZE
                band = coefficients[:, start : start + GROUP_SIZE]
                quantized_bands.append(
                    {
                        bits: quantize_band(band, bits)
                        for bits in ZERO_BIT_LEVELS
                    }
                )
            allocations: dict[str, tuple[int, ...]] = {}
            for objective, table in distortions.items():
                for budget in budgets:
                    allocations[
                        f"auto_{objective}_total_b{budget}"
                    ] = allocate_bits(
                        table,
                        budget,
                        ZERO_BIT_LEVELS,
                        include_scale_metadata=True,
                    )
            reconstructed = {
                method: reconstruct(quantized_bands, allocation)
                for method, allocation in allocations.items()
            }
            for method, allocation in allocations.items():
                code_bits = GROUP_SIZE * sum(allocation)
                metadata_bits = GROUP_SIZE * sum(
                    bits > 0 for bits in allocation
                )
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(map(str, allocation)),
                        "code_bits": code_bits,
                        "metadata_bits": metadata_bits,
                        "total_index_bits": code_bits + metadata_bits,
                    }
                )
            prepared.append(
                {
                    "head_key": head_key,
                    "basis": basis,
                    "reconstructed": reconstructed,
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        calibration_seconds += time.perf_counter() - layer_start

        top_count = max(1, math.ceil(args.top_fraction * history_count))
        for heldout_index, record in enumerate(
            layer_records[args.calibration_steps :],
            start=args.calibration_steps,
        ):
            query = record["query"].to(device).float()[0, :, 0, :]
            scaling = float(record["scaling"])
            for kv_head, state in enumerate(prepared):
                for group in range(groups):
                    query_head = kv_head * groups + group
                    head_query = query[query_head]
                    projected_query = query_int8(
                        head_query @ state["basis"]
                    )
                    exact_scores = state["head_key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(
                        exact_scores, k=top_count
                    ).indices
                    for method, reconstructed_key in state[
                        "reconstructed"
                    ].items():
                        approximate_scores = (
                            reconstructed_key @ projected_query
                        ) * scaling
                        for fraction in selected_fractions:
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "heldout_step": heldout_index,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "method": method,
                                    "selected_fraction_target": fraction,
                                    **selection_metrics(
                                        exact_scores,
                                        attention,
                                        approximate_scores,
                                        true_top,
                                        fraction,
                                    ),
                                }
                            )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "layer": layer,
                    "layers": len(by_layer),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows, allocation_rows)
    allocation_histograms = {}
    for method in sorted({str(row["method"]) for row in allocation_rows}):
        allocation_histograms[method] = dict(
            Counter(
                str(row["allocation"])
                for row in allocation_rows
                if row["method"] == method
            ).most_common()
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocation_rows)
    write_csv(args.output_dir / "shrinkage.csv", shrinkage_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "sample_stride": args.sample_stride,
            "calibration_key_count": args.calibration_key_count,
            "calibration_steps": args.calibration_steps,
            "total_rate_budgets": budgets,
            "selected_fractions": selected_fractions,
            "top_fraction": args.top_fraction,
        },
        "calibration_seconds": calibration_seconds,
        "shrinkage": {
            "oas_alpha_mean": sum(
                float(row["oas_alpha"]) for row in shrinkage_rows
            )
            / len(shrinkage_rows),
            "rank_ridge_alpha_mean": sum(
                float(row["rank_ridge_alpha"])
                for row in shrinkage_rows
            )
            / len(shrinkage_rows),
        },
        "allocation_histograms": allocation_histograms,
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
