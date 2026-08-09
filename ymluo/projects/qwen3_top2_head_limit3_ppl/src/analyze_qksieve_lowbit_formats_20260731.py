from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    query_int8,
    selection_metrics,
)
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors


HEAD_DIM = 128
BAND_SIZE = 16
BAND_COUNT = HEAD_DIM // BAND_SIZE
FULL_KV_BITS = 2 * HEAD_DIM * 16
SCALE_BITS_PER_BAND = 16

FAMILY_BITS = {
    "int_maxabs_native": (0, 1, 2, 4, 8),
    "int_lsq_native": (0, 1, 2, 4, 8),
    "int_lsq_3bit": (0, 1, 2, 3, 4, 8),
    "minifloat_maxabs": (0, 1, 2, 3, 4, 8),
    "minifloat_lsq": (0, 1, 2, 3, 4, 8),
}


def parse_ints(value: str) -> list[int]:
    result = sorted({int(part) for part in value.split(",") if part.strip()})
    if not result:
        raise ValueError("expected at least one integer")
    return result


def parse_floats(value: str) -> list[float]:
    result = sorted({float(part) for part in value.split(",") if part.strip()})
    if not result:
        raise ValueError("expected at least one float")
    return result


def parse_strings(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("expected at least one value")
    unknown = sorted(set(result) - set(FAMILY_BITS))
    if unknown:
        raise ValueError(f"unknown quantizer families: {unknown}")
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


def integer_codes(bits: int, device: torch.device) -> torch.Tensor:
    if bits == 1:
        return torch.tensor([-1.0, 1.0], device=device)
    maximum = (1 << (bits - 1)) - 1
    return torch.arange(-maximum, maximum + 1, device=device).float()


def fp8_e4m3_levels() -> list[float]:
    positive = {0.0}
    exponent_bits = 4
    mantissa_bits = 3
    bias = 7
    for mantissa in range(1, 1 << mantissa_bits):
        positive.add(
            (mantissa / (1 << mantissa_bits)) * 2.0 ** (1 - bias)
        )
    for exponent in range(1, 1 << exponent_bits):
        for mantissa in range(0, 1 << mantissa_bits):
            positive.add(
                (1.0 + mantissa / (1 << mantissa_bits))
                * 2.0 ** (exponent - bias)
            )
    maximum = max(positive)
    normalized = sorted(value / maximum for value in positive)
    return sorted({-value for value in normalized} | set(normalized))


def minifloat_codes(bits: int, device: torch.device) -> torch.Tensor:
    if bits == 1:
        levels = [-1.0, 1.0]
    elif bits == 2:
        levels = [-1.0, 0.0, 1.0]
    elif bits == 3:
        # E2M0-style logarithmic levels, with one unused bit pattern.
        positive = [0.0, 0.25, 0.5, 1.0]
        levels = sorted({-value for value in positive} | set(positive))
    elif bits == 4:
        # OCP-like E2M1 finite values, normalized to unit maximum.
        positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        positive = [value / 6.0 for value in positive]
        levels = sorted({-value for value in positive} | set(positive))
    elif bits == 8:
        levels = fp8_e4m3_levels()
    else:
        raise ValueError(f"unsupported minifloat width: {bits}")
    return torch.tensor(levels, dtype=torch.float32, device=device)


def nearest_codes(
    normalized: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    insertion = torch.searchsorted(levels, normalized.contiguous())
    upper_index = insertion.clamp(max=levels.numel() - 1)
    lower_index = (insertion - 1).clamp(min=0)
    upper = levels[upper_index]
    lower = levels[lower_index]
    return torch.where(
        (normalized - lower).abs() <= (upper - normalized).abs(),
        lower,
        upper,
    )


def quantize_with_codes(
    values: torch.Tensor,
    levels: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    working = values.float()
    maximum_level = levels.abs().amax().clamp_min(1.0e-12)
    if levels.numel() == 2 and bool(torch.all(levels.abs() == maximum_level)):
        codes = torch.where(working >= 0.0, maximum_level, -maximum_level)
        scale = working.abs().mean(dim=-1, keepdim=True) / maximum_level
        return (codes * scale).to(values.dtype)

    scale = (
        working.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        / maximum_level
    )
    codes = torch.zeros_like(working)
    for _ in range(iterations):
        codes = nearest_codes(working / scale, levels)
        denominator = codes.square().sum(dim=-1, keepdim=True)
        scale = (
            (codes * working).sum(dim=-1, keepdim=True)
            / denominator.clamp_min(1.0e-12)
        ).abs()
        scale = scale.clamp_min(1.0e-12)
    codes = nearest_codes(working / scale, levels)
    return (codes * scale).to(values.dtype)


def quantize_with_maxabs_codes(
    values: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    working = values.float()
    maximum_level = levels.abs().amax().clamp_min(1.0e-12)
    if levels.numel() == 2 and bool(torch.all(levels.abs() == maximum_level)):
        codes = torch.where(working >= 0.0, maximum_level, -maximum_level)
        scale = working.abs().mean(dim=-1, keepdim=True) / maximum_level
        return (codes * scale).to(values.dtype)
    scale = (
        working.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        / maximum_level
    )
    codes = nearest_codes(working / scale, levels)
    return (codes * scale).to(values.dtype)


def quantize_band(
    values: torch.Tensor,
    bits: int,
    family: str,
) -> torch.Tensor:
    if bits == 0:
        return torch.zeros_like(values)
    if family == "int_maxabs_native":
        if bits == 1:
            levels = integer_codes(bits, values.device)
            return quantize_with_codes(values, levels, iterations=1)
        maximum = (1 << (bits - 1)) - 1
        scale = (
            values.float()
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-12)
            / maximum
        )
        codes = torch.round(values.float() / scale).clamp(-maximum, maximum)
        return (codes * scale).to(values.dtype)
    if family in {"int_lsq_native", "int_lsq_3bit"}:
        return quantize_with_codes(
            values,
            integer_codes(bits, values.device),
            iterations=3,
        )
    if family == "minifloat_maxabs":
        return quantize_with_maxabs_codes(
            values,
            minifloat_codes(bits, values.device),
        )
    if family == "minifloat_lsq":
        return quantize_with_codes(
            values,
            minifloat_codes(bits, values.device),
            iterations=3,
        )
    raise ValueError(f"unknown family: {family}")


def allocation_cost(bits: int) -> int:
    return BAND_SIZE * bits + (SCALE_BITS_PER_BAND if bits > 0 else 0)


def allocate_formats(
    distortion: list[dict[int, float]],
    physical_budget: int,
    bit_levels: tuple[int, ...],
) -> tuple[int, ...]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for band_costs in distortion:
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_bits, (total_cost, allocation) in states.items():
            for bits in bit_levels:
                new_bits = used_bits + allocation_cost(bits)
                if new_bits > physical_budget:
                    continue
                candidate = total_cost + band_costs[bits]
                current = updated.get(new_bits)
                if current is None or candidate < current[0]:
                    updated[new_bits] = (candidate, allocation + (bits,))
        states = updated
    if not states:
        raise RuntimeError(f"no allocation fits {physical_budget} bits")
    candidates = [
        (cost, -used_bits, allocation)
        for used_bits, (cost, allocation) in states.items()
    ]
    return min(candidates)[2]


def aggregate(
    rows: list[dict[str, Any]],
    allocation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allocation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows:
        allocation_groups[str(row["method"])].append(row)
    metric_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        metric_groups[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)

    output = []
    for (method, fraction), items in sorted(metric_groups.items()):
        allocations = allocation_groups[method]
        physical_bits = sum(
            int(row["physical_bits"]) for row in allocations
        ) / len(allocations)
        result: dict[str, Any] = {
            "method": method,
            "family": str(items[0]["family"]),
            "budget_bits": int(items[0]["budget_bits"]),
            "selected_fraction_target": fraction,
            "cases": len(items),
            "physical_bits_mean": physical_bits,
            "index_ratio_of_full_kv": physical_bits / FULL_KV_BITS,
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
            "score_rmse",
        ):
            for statistic, value in summarize(
                float(item[field]) for item in items
            ).items():
                result[f"{field}_{statistic}"] = value
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare low-bit integer and scaled minifloat Key indices in "
            "QKSieve's QK-balanced coordinate system."
        )
    )
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--families",
        default=(
            "int_maxabs_native,int_lsq_native,"
            "int_lsq_3bit,minifloat_maxabs,minifloat_lsq"
        ),
    )
    parser.add_argument("--budgets", default="128,160,192,240")
    parser.add_argument("--selected_fractions", default="0.02")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.5)
    parser.add_argument("--max_layers", type=int, default=0)
    parser.add_argument("--max_heldout_steps", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    families = parse_strings(args.families)
    budgets = parse_ints(args.budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)
    layer_items = sorted(by_layer.items())
    if args.max_layers > 0:
        layer_items = layer_items[: args.max_layers]

    rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for layer_index, (layer, layer_records) in enumerate(layer_items, start=1):
        layer_records.sort(key=lambda row: int(row["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(f"layer {layer} has no held-out queries")
        raw_key = next(
            (
                record.get("key")
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no Key tensor")

        key = raw_key.to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        kv_head_count = int(key.shape[0])
        query_head_count = int(layer_records[0]["query"].shape[1])
        groups = query_head_count // kv_head_count
        calibration = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )

        prepared = []
        for kv_head in range(kv_head_count):
            head_key = key[kv_head]
            head_calibration = calibration[
                :, kv_head * groups : (kv_head + 1) * groups
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, _ = qk_balanced_factors(
                head_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            sampled_coefficients = coefficients[:: args.sample_stride]

            family_bands: dict[str, list[dict[int, torch.Tensor]]] = {}
            for family in families:
                quantized_bands: list[dict[int, torch.Tensor]] = []
                for band_index in range(BAND_COUNT):
                    start = band_index * BAND_SIZE
                    stop = start + BAND_SIZE
                    band = coefficients[:, start:stop]
                    quantized_bands.append(
                        {
                            bits: quantize_band(band, bits, family)
                            for bits in FAMILY_BITS[family]
                        }
                    )
                family_bands[family] = quantized_bands

            methods: dict[str, dict[str, Any]] = {}
            for family in families:
                quantized_bands = family_bands[family]
                distortion: list[dict[int, float]] = []
                for band_index in range(BAND_COUNT):
                    start = band_index * BAND_SIZE
                    stop = start + BAND_SIZE
                    exact_band = sampled_coefficients[:, start:stop]
                    query_band = projected_calibration[:, start:stop]
                    band_costs = {}
                    for bits in FAMILY_BITS[family]:
                        reconstructed_band = quantized_bands[band_index][bits][
                            :: args.sample_stride
                        ]
                        score_error = query_band @ (
                            exact_band - reconstructed_band
                        ).transpose(0, 1)
                        band_costs[bits] = float(
                            score_error.square().mean().item()
                        )
                    distortion.append(band_costs)

                for budget in budgets:
                    allocation = allocate_formats(
                        distortion,
                        budget,
                        FAMILY_BITS[family],
                    )
                    reconstructed = torch.cat(
                        [
                            quantized_bands[index][bits]
                            for index, bits in enumerate(allocation)
                        ],
                        dim=-1,
                    )
                    physical_bits = sum(
                        allocation_cost(bits) for bits in allocation
                    )
                    method = f"{family}_b{budget}"
                    methods[method] = {
                        "family": family,
                        "budget": budget,
                        "query_factor": query_factor,
                        "reconstructed": reconstructed,
                    }
                    allocation_rows.append(
                        {
                            "label": args.label,
                            "layer": layer,
                            "kv_head": kv_head,
                            "method": method,
                            "family": family,
                            "budget_bits": budget,
                            "allocation": "-".join(map(str, allocation)),
                            "code_bits": BAND_SIZE * sum(allocation),
                            "scale_bits": SCALE_BITS_PER_BAND
                            * sum(bits > 0 for bits in allocation),
                            "physical_bits": physical_bits,
                        }
                    )
            prepared.append({"head_key": head_key, "methods": methods})

        heldout_records = layer_records[args.calibration_steps :]
        if args.max_heldout_steps > 0:
            heldout_records = heldout_records[: args.max_heldout_steps]
        top_count = max(1, math.ceil(args.top_fraction * history_count))
        for heldout_offset, record in enumerate(heldout_records):
            heldout_step = args.calibration_steps + heldout_offset
            query = record["query"].to(device).float()[0, :, 0, :]
            scaling = float(record["scaling"])
            for kv_head, state in enumerate(prepared):
                for group in range(groups):
                    query_head = kv_head * groups + group
                    head_query = query[query_head]
                    exact_scores = state["head_key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(exact_scores, k=top_count).indices
                    for method, method_state in state["methods"].items():
                        projected_query = query_int8(
                            head_query @ method_state["query_factor"]
                        )
                        approximate_scores = (
                            method_state["reconstructed"] @ projected_query
                        ) * scaling
                        for fraction in selected_fractions:
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "heldout_step": heldout_step,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "method": method,
                                    "family": method_state["family"],
                                    "budget_bits": method_state["budget"],
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
                    "layer_progress": f"{layer_index}/{len(layer_items)}",
                    "rows": len(rows),
                }
            ),
            flush=True,
        )
        del prepared, key, calibration
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate(rows, allocation_rows)
    output = {
        "config": {
            "trace_path": str(args.trace_path),
            "label": args.label,
            "families": families,
            "budgets": budgets,
            "selected_fractions": selected_fractions,
            "top_fraction": args.top_fraction,
            "sample_stride": args.sample_stride,
            "calibration_steps": args.calibration_steps,
            "query_shrinkage": args.query_shrinkage,
            "layers": len(layer_items),
        },
        "allocation_histograms": {
            method: dict(
                Counter(
                    str(row["allocation"])
                    for row in allocation_rows
                    if row["method"] == method
                ).most_common()
            )
            for method in sorted(
                {str(row["method"]) for row in allocation_rows}
            )
        },
        "methods": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocation_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
