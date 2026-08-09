from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
    quantize_band,
    reconstruct,
)
from analyze_hierarchical_spectral_quantization_20260727 import (
    covariance_basis,
    query_int8,
    selection_metrics,
)


HEAD_DIM = 128
GROUP_SIZE = 16
GROUP_COUNT = HEAD_DIM // GROUP_SIZE
FULL_KV_BITS = 2 * HEAD_DIM * 16


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


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    left_tensor = torch.tensor(list(left), dtype=torch.float64)
    right_tensor = torch.tensor(list(right), dtype=torch.float64)
    left_tensor -= left_tensor.mean()
    right_tensor -= right_tensor.mean()
    denominator = (
        torch.linalg.vector_norm(left_tensor)
        * torch.linalg.vector_norm(right_tensor)
    )
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((left_tensor @ right_tensor / denominator).item())


def rate_fields(allocation: tuple[int, ...]) -> dict[str, float | int]:
    code_bits = GROUP_SIZE * sum(allocation)
    metadata_bits = 16 * sum(bits > 0 for bits in allocation)
    return {
        "allocation": "-".join(map(str, allocation)),
        "code_bits": code_bits,
        "metadata_bits": metadata_bits,
        "total_index_bits": code_bits + metadata_bits,
        "index_ratio_of_full_kv": (
            code_bits + metadata_bits
        )
        / FULL_KV_BITS,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output = []
    fields = (
        "calibration_qmse_objective",
        "heldout_key_error_mse",
        "diagonal_band_mse",
        "cross_band_mse",
        "absolute_cross_over_diagonal",
        "actual_over_diagonal",
        "query_quantization_mse",
        "production_score_mse",
        "normalized_boundary_gap",
        "retrieval_safety_margin",
        "cantelli_crossing_bound",
        "top2_recall",
        "selected_attention_mass",
        "score_pearson",
    )
    for method, items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "cases": len(items),
            "total_index_bits_mean": sum(
                int(item["total_index_bits"]) for item in items
            )
            / len(items),
            "index_ratio_of_full_kv_mean": sum(
                float(item["index_ratio_of_full_kv"]) for item in items
            )
            / len(items),
        }
        for field in fields:
            statistics = summarize(float(item[field]) for item in items)
            result.update(
                {
                    f"{field}_{name}": value
                    for name, value in statistics.items()
                }
            )
        result["log_objective_vs_log_heldout_mse_pearson"] = pearson(
            [
                math.log(
                    max(
                        1.0e-20,
                        float(item["calibration_qmse_objective"]),
                    )
                )
                for item in items
            ],
            [
                math.log(
                    max(1.0e-20, float(item["heldout_key_error_mse"]))
                )
                for item in items
            ],
        )
        result["safety_margin_vs_top2_recall_pearson"] = pearson(
            [
                float(item["retrieval_safety_margin"])
                for item in items
            ],
            [float(item["top2_recall"]) for item in items],
        )
        result["crossing_bound_vs_top2_recall_pearson"] = pearson(
            [
                float(item["cantelli_crossing_bound"])
                for item in items
            ],
            [float(item["top2_recall"]) for item in items],
        )
        output.append(result)
    return output


def margin_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    for method, items in sorted(grouped.items()):
        margins = torch.tensor(
            [float(item["retrieval_safety_margin"]) for item in items],
            dtype=torch.float64,
        )
        boundaries = [
            -math.inf,
            float(torch.quantile(margins, 0.25).item()),
            float(torch.quantile(margins, 0.50).item()),
            float(torch.quantile(margins, 0.75).item()),
            math.inf,
        ]
        for bin_index in range(4):
            selected = [
                item
                for item in items
                if boundaries[bin_index]
                <= float(item["retrieval_safety_margin"])
                and (
                    float(item["retrieval_safety_margin"])
                    < boundaries[bin_index + 1]
                    if bin_index < 3
                    else float(item["retrieval_safety_margin"])
                    <= boundaries[bin_index + 1]
                )
            ]
            output.append(
                {
                    "method": method,
                    "safety_margin_quartile": bin_index + 1,
                    "lower": boundaries[bin_index],
                    "upper": boundaries[bin_index + 1],
                    "cases": len(selected),
                    "top2_recall_mean": sum(
                        float(item["top2_recall"]) for item in selected
                    )
                    / max(1, len(selected)),
                    "selected_attention_mass_mean": sum(
                        float(item["selected_attention_mass"])
                        for item in selected
                    )
                    / max(1, len(selected)),
                    "cantelli_crossing_bound_mean": sum(
                        float(item["cantelli_crossing_bound"])
                        for item in selected
                    )
                    / max(1, len(selected)),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the block-diagonal QK-error objective, cross-band terms, "
            "and top-k margin behavior on held-out Q/K traces."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--selected_fraction", type=float, default=0.06)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda record: int(record["step"]))
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
        prepared = []
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
            ).reshape(-1, HEAD_DIM)
            key_distortion, query_distortion = distortion_table(
                coefficients,
                projected_calibration,
                ZERO_BIT_LEVELS,
            )
            allocations = {
                "auto_key_b10": allocate_bits(
                    key_distortion,
                    10,
                    ZERO_BIT_LEVELS,
                ),
                "auto_qmse_b10": allocate_bits(
                    query_distortion,
                    10,
                    ZERO_BIT_LEVELS,
                ),
                "auto_qmse_total_b14": allocate_bits(
                    query_distortion,
                    14,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                ),
                "auto_qmse_total_b15": allocate_bits(
                    query_distortion,
                    15,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                ),
            }
            bands = []
            for group_index in range(GROUP_COUNT):
                start = group_index * GROUP_SIZE
                band = coefficients[:, start : start + GROUP_SIZE]
                bands.append(
                    {
                        bits: quantize_band(band, bits)
                        for bits in ZERO_BIT_LEVELS
                    }
                )
            prepared.append(
                {
                    "head_key": head_key,
                    "basis": basis,
                    "coefficients": coefficients,
                    "allocations": allocations,
                    "reconstructed": {
                        method: reconstruct(bands, allocation)
                        for method, allocation in allocations.items()
                    },
                    "calibration_objective": {
                        method: sum(
                            float(query_distortion[index][bits].item())
                            for index, bits in enumerate(allocation)
                        )
                        for method, allocation in allocations.items()
                    },
                }
            )

        top_count = max(1, math.ceil(args.top_fraction * history_count))
        selected_count = max(
            top_count,
            math.ceil(args.selected_fraction * history_count),
        )
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
                    projected_query = head_query @ state["basis"]
                    quantized_query = query_int8(projected_query)
                    exact_scores = state["head_key"] @ head_query
                    scaled_exact_scores = exact_scores * scaling
                    attention = torch.softmax(scaled_exact_scores, dim=-1)
                    top_values, top_indices = torch.topk(
                        scaled_exact_scores,
                        k=min(history_count, selected_count + 1),
                    )
                    boundary_gap = (
                        float((top_values[top_count - 1] - top_values[top_count]).item())
                        if top_values.numel() > top_count
                        else math.inf
                    )
                    normalized_gap = boundary_gap / max(
                        1.0e-8,
                        float(scaled_exact_scores.std().item()),
                    )
                    true_top = top_indices[:top_count]
                    retrieval_gap = (
                        float(
                            (
                                top_values[top_count - 1]
                                - top_values[selected_count]
                            ).item()
                        )
                        if top_values.numel() > selected_count
                        else math.inf
                    )
                    for method, reconstructed in state[
                        "reconstructed"
                    ].items():
                        residual = state["coefficients"] - reconstructed
                        band_errors = torch.stack(
                            [
                                (
                                    residual[
                                        :,
                                        index * GROUP_SIZE : (index + 1)
                                        * GROUP_SIZE,
                                    ]
                                    @ projected_query[
                                        index * GROUP_SIZE : (index + 1)
                                        * GROUP_SIZE
                                    ]
                                )
                                for index in range(GROUP_COUNT)
                            ],
                            dim=-1,
                        )
                        key_error = band_errors.sum(dim=-1)
                        actual_mse = float(key_error.square().mean().item())
                        diagonal_mse = float(
                            band_errors.square().sum(dim=-1).mean().item()
                        )
                        cross_mse = actual_mse - diagonal_mse
                        approximate_raw = reconstructed @ quantized_query
                        approximate_scores = approximate_raw * scaling
                        production_error = exact_scores - approximate_raw
                        query_only_error = (
                            reconstructed
                            @ (projected_query - quantized_query)
                        )
                        metrics = selection_metrics(
                            scaled_exact_scores,
                            attention,
                            approximate_scores,
                            true_top,
                            args.selected_fraction,
                        )
                        pairwise_error_std = (
                            math.sqrt(
                                2.0
                                * max(
                                    1.0e-20,
                                    float(
                                        production_error.square()
                                        .mean()
                                        .item()
                                    ),
                                )
                            )
                            * abs(scaling)
                        )
                        safety_margin = retrieval_gap / max(
                            1.0e-12, pairwise_error_std
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "layer": layer,
                                "heldout_step": heldout_index,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "method": method,
                                **rate_fields(
                                    state["allocations"][method]
                                ),
                                "calibration_qmse_objective": state[
                                    "calibration_objective"
                                ][method],
                                "heldout_key_error_mse": actual_mse,
                                "diagonal_band_mse": diagonal_mse,
                                "cross_band_mse": cross_mse,
                                "absolute_cross_over_diagonal": (
                                    abs(cross_mse)
                                    / max(1.0e-20, diagonal_mse)
                                ),
                                "actual_over_diagonal": (
                                    actual_mse
                                    / max(1.0e-20, diagonal_mse)
                                ),
                                "query_quantization_mse": float(
                                    query_only_error.square().mean().item()
                                ),
                                "production_score_mse": float(
                                    production_error.square().mean().item()
                                ),
                                "normalized_boundary_gap": normalized_gap,
                                "retrieval_safety_margin": safety_margin,
                                "cantelli_crossing_bound": (
                                    1.0
                                    / (1.0 + safety_margin * safety_margin)
                                ),
                                **metrics,
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

    summary = aggregate(rows)
    bins = margin_bins(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cases.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "margin_bins.csv", bins)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "sample_stride": args.sample_stride,
            "calibration_steps": args.calibration_steps,
            "selected_fraction": args.selected_fraction,
            "top_fraction": args.top_fraction,
        },
        "methods": summary,
        "margin_bins": bins,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
