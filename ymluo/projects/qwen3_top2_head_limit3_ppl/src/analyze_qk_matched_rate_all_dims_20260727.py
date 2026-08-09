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
    FULL_KV_BITS,
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
)
from analyze_hierarchical_spectral_quantization_20260727 import (
    query_int8,
    selection_metrics,
)
from analyze_qk_balanced_spectral_rate_20260727 import (
    distortion_table_from_bands,
    metric_scale_quantize_band,
    qk_balanced_factors,
)


MATCHED_CONFIGS = {
    "fixed_spectral_4421": ((4, 4, 2, 1, 0, 0, 0, 0), 15),
    "fixed_spectral_4440": ((4, 4, 4, 0, 0, 0, 0, 0), 15),
    "fixed_spectral_8111": ((8, 1, 1, 1, 0, 0, 0, 0), 15),
    "uniform_all_dims_int1": ((1,) * GROUP_COUNT, 16),
    "uniform_all_dims_int2": ((2,) * GROUP_COUNT, 24),
    "uniform_all_dims_int4": ((4,) * GROUP_COUNT, 40),
    "manual_head8_mid4_tail0": ((8, 4, 4, 0, 0, 0, 0, 0), 19),
    "manual_head8_mid4_tail1": ((8, 4, 4, 1, 1, 1, 1, 1), 29),
    "manual_head8_mid4_tail2": ((8, 4, 4, 2, 2, 2, 2, 2), 34),
}


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {
                float(item)
                for item in specification.split(",")
                if item.strip()
            }
        )
    )
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("fractions must be in (0, 1)")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def allocation_rate(allocation: Iterable[int]) -> int:
    return sum(int(bits) + int(int(bits) > 0) for bits in allocation)


def reconstruct(
    bands: list[dict[int, torch.Tensor]],
    allocation: Iterable[int],
) -> torch.Tensor:
    return torch.cat(
        [
            bands[index][int(bits)]
            for index, bits in enumerate(allocation)
        ],
        dim=-1,
    )


def quantized_bands(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
) -> list[dict[int, torch.Tensor]]:
    output: list[dict[int, torch.Tensor]] = []
    for band_index in range(GROUP_COUNT):
        start = band_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        output.append(
            {
                bits: metric_scale_quantize_band(
                    coefficients[:, start:stop],
                    bits,
                    calibration_queries[:, start:stop],
                    metric_mode="full",
                )
                for bits in ZERO_BIT_LEVELS
            }
        )
    return output


def diagonal_covariance_samples(
    diagonal: torch.Tensor,
) -> torch.Tensor:
    """Construct samples whose uncentered covariance is exactly diag(diagonal)."""
    if diagonal.ndim != 1:
        raise ValueError("covariance diagonal must be one-dimensional")
    if bool((diagonal < 0.0).any().item()):
        raise ValueError("covariance diagonal must be non-negative")
    dimension = int(diagonal.numel())
    return torch.diag((diagonal.float() * dimension).sqrt())


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


def aggregate(
    rows: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allocation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allocations:
        allocation_groups[str(row["method"])].append(row)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (method, fraction), items in sorted(groups.items()):
        method_allocations = allocation_groups[method]
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "index_bits_mean": sum(
                int(row["index_bits"]) for row in method_allocations
            )
            / len(method_allocations),
        }
        result["index_ratio_of_full_kv"] = (
            result["index_bits_mean"] / FULL_KV_BITS
        )
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
            "score_rmse",
        ):
            result.update(
                {
                    f"{field}_{statistic}": value
                    for statistic, value in summarize(
                        float(item[field]) for item in items
                    ).items()
                }
            )
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "At exactly matched physical index rates, compare all-dimension "
            "uniform low-bit QK indices with qMSE-allocated mixed precision."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--top_fraction", type=float, default=0.01)
    parser.add_argument("--selected_fractions", default="0.01,0.04")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    selected_fractions = parse_floats(args.selected_fractions)
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if not 0.0 <= args.query_shrinkage <= 1.0:
        raise ValueError("query_shrinkage must be in [0, 1]")
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
        if state_record is None:
            raise ValueError(f"layer {layer} has no captured Key")
        key = state_record["key"].to(device).float()[0, :, :-1]
        scaling = float(state_record["scaling"])
        calibration = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )
        kv_heads = int(key.shape[0])
        query_heads = int(calibration.shape[1])
        if query_heads % kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_heads // kv_heads

        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            head_calibration = calibration[
                :,
                kv_head * groups : (kv_head + 1) * groups,
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, singular_values = qk_balanced_factors(
                head_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            empirical_bands = quantized_bands(
                coefficients,
                projected_calibration,
            )
            empirical_distortion = distortion_table_from_bands(
                coefficients,
                projected_calibration,
                empirical_bands,
            )
            regularized_queries = diagonal_covariance_samples(
                singular_values
            )
            regularized_bands = quantized_bands(
                coefficients,
                regularized_queries,
            )
            regularized_distortion = distortion_table_from_bands(
                coefficients,
                regularized_queries,
                regularized_bands,
            )
            method_allocations: dict[str, tuple[int, ...]] = {}
            method_bands: dict[
                str, list[dict[int, torch.Tensor]]
            ] = {}
            for uniform_method, (uniform_allocation, rate) in (
                MATCHED_CONFIGS.items()
            ):
                method_allocations[uniform_method] = uniform_allocation
                method_bands[uniform_method] = empirical_bands
                empirical_method = f"auto_qmse_rate{rate}"
                method_allocations[empirical_method] = allocate_bits(
                    empirical_distortion,
                    rate,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                method_bands[empirical_method] = empirical_bands
                regularized_method = f"auto_regularized_qmse_rate{rate}"
                method_allocations[regularized_method] = allocate_bits(
                    regularized_distortion,
                    rate,
                    ZERO_BIT_LEVELS,
                    include_scale_metadata=True,
                )
                method_bands[regularized_method] = regularized_bands
            reconstructions = {
                method: reconstruct(method_bands[method], allocation)
                for method, allocation in method_allocations.items()
            }
            for method, allocation in method_allocations.items():
                allocations.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation_objective": (
                            "regularized_diagonal_qmse"
                            if method.startswith(
                                "auto_regularized_qmse_"
                            )
                            else "empirical_band_qmse"
                            if method.startswith("auto_qmse_")
                            else "fixed_schedule"
                        ),
                        "allocation": "-".join(
                            str(bits) for bits in allocation
                        ),
                        "index_bits": (
                            GROUP_SIZE * allocation_rate(allocation)
                        ),
                    }
                )

            for record in layer_records[args.calibration_steps :]:
                step = int(record["step"])
                query = record["query"].to(device).float()[0, :, 0, :]
                for query_head in range(
                    kv_head * groups,
                    (kv_head + 1) * groups,
                ):
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (
                        coefficients.float() @ projected_query.float()
                    ) * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    top_count = min(
                        exact_scores.numel(),
                        max(
                            1,
                            math.ceil(
                                args.top_fraction * exact_scores.numel()
                            ),
                        ),
                    )
                    true_top = torch.topk(
                        exact_scores,
                        k=top_count,
                    ).indices
                    for method, reconstructed in reconstructions.items():
                        approximate_scores = (
                            reconstructed.float()
                            @ approximate_query.float()
                        ) * scaling
                        for fraction in selected_fractions:
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "step": step,
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
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocations)
    summary = aggregate(rows, allocations)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
