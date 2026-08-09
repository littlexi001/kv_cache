from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_COUNT,
    GROUP_SIZE,
    quantize_band,
)
from analyze_hierarchical_spectral_quantization_20260727 import (
    quantize_groupwise,
    query_int8,
    selection_metrics,
)
from analyze_qk_balanced_spectral_rate_20260727 import (
    qk_balanced_factors,
    summarize,
)


STANDARD_OPTIONS = (
    ("drop", 0, 0),
    ("int1", 32, 1),
    ("int2", 48, 2),
    ("int4", 80, 4),
    ("int8", 144, 8),
)
SHARED_SIGN_WIDTHS = (2, 4, 8, 16)
JL_RATE_OPTIONS = (
    (8, 8),
    (16, 4),
    (32, 2),
    (8, 4),
    (16, 2),
)


def parse_floats(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("expected at least one floating-point value")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shared_envelope_parameters(
    sampled_coefficients: torch.Tensor,
    envelope_start: int = 0,
    envelope_stop: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate_rms = sampled_coefficients.square().mean(dim=0).sqrt().clamp_min(
        1.0e-8
    )
    normalized = sampled_coefficients / coordinate_rms
    envelope = (
        normalized[:, envelope_start:envelope_stop]
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
    )
    denominator = envelope.square().sum().clamp_min(1.0e-12)
    coordinate_amplitude = (
        envelope * sampled_coefficients.abs()
    ).sum(dim=0) / denominator
    return coordinate_rms, coordinate_amplitude


def shared_envelope(
    coefficients: torch.Tensor,
    coordinate_rms: torch.Tensor,
    envelope_start: int = 0,
    envelope_stop: int = 128,
) -> torch.Tensor:
    normalized = coefficients / coordinate_rms
    # The single FP16 value is the only per-token scale used by all sign bands.
    return (
        normalized[:, envelope_start:envelope_stop]
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
        .half()
        .float()
    )


def shared_sign_reconstruction(
    band: torch.Tensor,
    envelope: torch.Tensor,
    coordinate_amplitude: torch.Tensor,
    width: int,
) -> torch.Tensor:
    reconstructed = torch.zeros_like(band)
    reconstructed[:, :width] = (
        torch.where(
            band[:, :width] >= 0.0,
            torch.ones_like(band[:, :width]),
            -torch.ones_like(band[:, :width]),
        )
        * envelope
        * coordinate_amplitude[:width]
    )
    return reconstructed


def rademacher_projection(
    input_dimension: int,
    output_dimension: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (input_dimension, output_dimension),
        generator=generator,
        dtype=torch.int8,
    )
    return (
        signs.float().mul_(2.0).sub_(1.0)
        / math.sqrt(output_dimension)
    ).to(device)


def quantize_projection(
    values: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    group_size = min(16, int(values.shape[-1]))
    return quantize_groupwise(values, bits, group_size=group_size)


def prepare_jl_tail(
    coefficients: torch.Tensor,
    sampled_coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    output_dimension: int,
    bits: int,
    seed: int,
) -> dict[str, torch.Tensor | float | int]:
    projection = rademacher_projection(
        coefficients.shape[-1] - 32,
        output_dimension,
        seed,
        coefficients.device,
    )
    sampled_tail = sampled_coefficients[:, 32:]
    query_tail = calibration_queries[:, 32:]
    sampled_sketch = quantize_projection(sampled_tail @ projection, bits)
    projected_query = quantize_projection(query_tail @ projection, 8)
    exact_tail_scores = query_tail @ sampled_tail.transpose(0, 1)
    approximate_tail_scores = (
        projected_query @ sampled_sketch.transpose(0, 1)
    )
    calibration_qmse = float(
        (exact_tail_scores - approximate_tail_scores)
        .square()
        .mean()
        .item()
    )
    return {
        "projection": projection,
        "reconstructed_sketch": quantize_projection(
            coefficients[:, 32:] @ projection,
            bits,
        ),
        "calibration_qmse": calibration_qmse,
        "seed": seed,
    }


def make_band_options(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    coordinate_rms: torch.Tensor,
    coordinate_amplitude: torch.Tensor,
    envelope_start: int = 0,
    envelope_stop: int = 128,
) -> list[dict[str, tuple[int, bool, torch.Tensor, float]]]:
    sampled_coefficients = coefficients[::32]
    sampled_envelope = shared_envelope(
        sampled_coefficients,
        coordinate_rms,
        envelope_start,
        envelope_stop,
    )
    full_envelope = shared_envelope(
        coefficients,
        coordinate_rms,
        envelope_start,
        envelope_stop,
    )
    output = []
    for group_index in range(GROUP_COUNT):
        start = group_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        sampled_band = sampled_coefficients[:, start:stop]
        full_band = coefficients[:, start:stop]
        query_band = calibration_queries[:, start:stop]
        group_options: dict[str, tuple[int, bool, torch.Tensor, float]] = {}
        for name, physical_bits, quantization_bits in STANDARD_OPTIONS:
            reconstructed = quantize_band(full_band, quantization_bits)
            sampled_reconstructed = quantize_band(
                sampled_band, quantization_bits
            )
            residual = sampled_band - sampled_reconstructed
            cost = float(
                (query_band @ residual.transpose(0, 1))
                .square()
                .mean()
                .item()
            )
            group_options[name] = (
                physical_bits,
                False,
                reconstructed,
                cost,
            )
        amplitude = coordinate_amplitude[start:stop]
        for width in SHARED_SIGN_WIDTHS:
            name = f"shared_sign{width}"
            reconstructed = shared_sign_reconstruction(
                full_band,
                full_envelope,
                amplitude,
                width,
            )
            sampled_reconstructed = shared_sign_reconstruction(
                sampled_band,
                sampled_envelope,
                amplitude,
                width,
            )
            residual = sampled_band - sampled_reconstructed
            cost = float(
                (query_band @ residual.transpose(0, 1))
                .square()
                .mean()
                .item()
            )
            group_options[name] = (
                width,
                True,
                reconstructed,
                cost,
            )
        output.append(group_options)
    return output


def allocate_options(
    band_options: list[dict[str, tuple[int, bool, torch.Tensor, float]]],
    total_bit_budget: int,
    allow_shared_sign: bool,
) -> tuple[list[str], int]:
    states: dict[
        tuple[int, bool],
        tuple[float, tuple[str, ...]],
    ] = {(0, False): (0.0, ())}
    for options in band_options:
        updated: dict[
            tuple[int, bool],
            tuple[float, tuple[str, ...]],
        ] = {}
        for (used_bits, shared_scale_used), (cost, allocation) in states.items():
            for name, (option_bits, uses_shared_scale, _, option_cost) in (
                options.items()
            ):
                if uses_shared_scale and not allow_shared_sign:
                    continue
                new_shared_scale_used = shared_scale_used or uses_shared_scale
                new_bits = (
                    used_bits
                    + option_bits
                    + (
                        16
                        if uses_shared_scale and not shared_scale_used
                        else 0
                    )
                )
                if new_bits > total_bit_budget:
                    continue
                key = (new_bits, new_shared_scale_used)
                candidate = (cost + option_cost, allocation + (name,))
                current = updated.get(key)
                if current is None or candidate[0] < current[0]:
                    updated[key] = candidate
        states = updated
    feasible = [
        (cost, -used_bits, allocation, used_bits)
        for (used_bits, _), (cost, allocation) in states.items()
    ]
    if not feasible:
        raise RuntimeError("no feasible fractional-tail allocation")
    _, _, allocation, used_bits = min(feasible)
    return list(allocation), used_bits


def reconstruct(
    band_options: list[dict[str, tuple[int, bool, torch.Tensor, float]]],
    allocation: list[str],
) -> torch.Tensor:
    return torch.cat(
        [
            options[name][2]
            for options, name in zip(band_options, allocation, strict=True)
        ],
        dim=-1,
    )


def allocation_physical_bits(
    band_options: list[dict[str, tuple[int, bool, torch.Tensor, float]]],
    allocation: list[str],
) -> int:
    used_bits = sum(
        options[name][0]
        for options, name in zip(band_options, allocation, strict=True)
    )
    if any(
        options[name][1]
        for options, name in zip(band_options, allocation, strict=True)
    ):
        used_bits += 16
    return used_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test an ultra-low-rate shared-envelope sign sketch on the tail "
            "of QK-balanced spectral coordinates."
        )
    )
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--total_bit_budget", type=int, default=240)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--selected_fractions", default="0.01,0.02,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.01)
    parser.add_argument("--center_keys", action="store_true")
    parser.add_argument("--jl_seed_trials", type=int, default=8)
    parser.add_argument("--jl_seed", type=int, default=20260727)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    selected_fractions = parse_floats(args.selected_fractions)
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows = []
    allocation_rows = []
    for layer, layer_records in sorted(by_layer.items()):
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
            raise ValueError(f"layer {layer} has no key tensor")
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
            key_mean = (
                head_key.mean(dim=0)
                if args.center_keys
                else torch.zeros_like(head_key[0])
            )
            index_key = head_key - key_mean
            head_calibration = calibration[
                :, kv_head * groups : (kv_head + 1) * groups
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, _ = qk_balanced_factors(
                index_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = index_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            coordinate_rms, coordinate_amplitude = shared_envelope_parameters(
                coefficients[:: args.sample_stride]
            )
            options = make_band_options(
                coefficients,
                projected_calibration,
                coordinate_rms,
                coordinate_amplitude,
            )
            (
                tail_coordinate_rms,
                tail_coordinate_amplitude,
            ) = shared_envelope_parameters(
                coefficients[:: args.sample_stride],
                32,
                96,
            )
            tail_options = make_band_options(
                coefficients,
                projected_calibration,
                tail_coordinate_rms,
                tail_coordinate_amplitude,
                32,
                96,
            )
            spread_options = {}
            for envelope_start, envelope_stop in (
                (0, 128),
                (16, 80),
                (16, 96),
                (16, 112),
                (16, 128),
                (32, 128),
                (48, 128),
            ):
                (
                    spread_coordinate_rms,
                    spread_coordinate_amplitude,
                ) = shared_envelope_parameters(
                    coefficients[:: args.sample_stride],
                    envelope_start,
                    envelope_stop,
                )
                spread_options[
                    (envelope_start, envelope_stop)
                ] = make_band_options(
                    coefficients,
                    projected_calibration,
                    spread_coordinate_rms,
                    spread_coordinate_amplitude,
                    envelope_start,
                    envelope_stop,
                )
            core_reconstruction = torch.cat(
                (
                    quantize_band(coefficients[:, :16], 4),
                    quantize_band(coefficients[:, 16:32], 4),
                ),
                dim=-1,
            )
            jl_methods: dict[
                str, dict[str, torch.Tensor | float | int]
            ] = {}
            for output_dimension, bits in JL_RATE_OPTIONS:
                fixed_seed = (
                    args.jl_seed
                    + layer * 100003
                    + kv_head * 1009
                    + output_dimension * 31
                    + bits
                )
                fixed = prepare_jl_tail(
                    coefficients,
                    coefficients[:: args.sample_stride],
                    projected_calibration,
                    output_dimension,
                    bits,
                    fixed_seed,
                )
                fixed_name = f"qk_core44_jl{output_dimension}_int{bits}_fixed"
                jl_methods[fixed_name] = fixed

                trials = [
                    prepare_jl_tail(
                        coefficients,
                        coefficients[:: args.sample_stride],
                        projected_calibration,
                        output_dimension,
                        bits,
                        fixed_seed + trial * 104729,
                    )
                    for trial in range(args.jl_seed_trials)
                ]
                selected = min(
                    trials,
                    key=lambda state: float(state["calibration_qmse"]),
                )
                selected_name = (
                    f"qk_core44_jl{output_dimension}_int{bits}_select"
                    f"{args.jl_seed_trials}"
                )
                jl_methods[selected_name] = selected
                total_bits = (
                    2 * (GROUP_SIZE * 4 + 16)
                    + output_dimension * bits
                    + 16
                )
                for method, state in (
                    (fixed_name, fixed),
                    (selected_name, selected),
                ):
                    allocation_rows.append(
                        {
                            "label": args.label,
                            "layer": layer,
                            "kv_head": kv_head,
                            "method": method,
                            "allocation": (
                                f"int4-int4-jl{output_dimension}xint{bits}"
                            ),
                            "total_index_bits": total_bits,
                            "index_ratio_of_full_kv": (
                                total_bits / FULL_KV_BITS
                            ),
                            "uses_shared_envelope": False,
                            "jl_seed": int(state["seed"]),
                            "jl_calibration_qmse": float(
                                state["calibration_qmse"]
                            ),
                        }
                    )
            head_methods = {}
            for method, allow_shared_sign in (
                ("qk_standard_240b", False),
                ("qk_fractional_tail_240b", True),
            ):
                allocation, used_bits = allocate_options(
                    options,
                    args.total_bit_budget,
                    allow_shared_sign,
                )
                head_methods[method] = reconstruct(options, allocation)
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(allocation),
                        "total_index_bits": used_bits,
                        "index_ratio_of_full_kv": used_bits / FULL_KV_BITS,
                        "uses_shared_envelope": any(
                            name.startswith("shared_sign")
                            for name in allocation
                        ),
                    }
                )
            full_sign_options = [
                {
                    name: option
                    for name, option in options.items()
                    if not name.startswith("shared_sign")
                    or name == "shared_sign16"
                }
                for options in options
            ]
            full_sign_allocation, full_sign_bits = allocate_options(
                full_sign_options,
                args.total_bit_budget,
                allow_shared_sign=True,
            )
            head_methods["qk_fractional_fullsign_240b"] = reconstruct(
                full_sign_options,
                full_sign_allocation,
            )
            allocation_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": "qk_fractional_fullsign_240b",
                    "allocation": "-".join(full_sign_allocation),
                    "total_index_bits": full_sign_bits,
                    "index_ratio_of_full_kv": (
                        full_sign_bits / FULL_KV_BITS
                    ),
                    "uses_shared_envelope": any(
                        name.startswith("shared_sign")
                        for name in full_sign_allocation
                    ),
                }
            )
            fixed_allocations = {
                "qk_fixed_44_tail1shared_240b": (
                    ["int4", "int4"]
                    + ["shared_sign16"] * 4
                    + ["drop"] * 2
                ),
                "qk_fixed_44_tail1shared_dp4a_240b": (
                    ["int4", "int4"]
                    + ["shared_sign16"] * 4
                    + ["drop"] * 2
                ),
                "qk_fixed_8_44_drop_tail_304b": (
                    ["int8", "int4", "int4"] + ["drop"] * 5
                ),
                "qk_fixed_8_44_tail0125_330b": (
                    ["int8", "int4", "int4"] + ["shared_sign2"] * 5
                ),
                "qk_fixed_8_44_tail0500_360b": (
                    ["int8", "int4", "int4"] + ["shared_sign8"] * 5
                ),
            }
            for method, allocation in fixed_allocations.items():
                used_bits = allocation_physical_bits(options, allocation)
                head_methods[method] = reconstruct(options, allocation)
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(allocation),
                        "total_index_bits": used_bits,
                        "index_ratio_of_full_kv": used_bits / FULL_KV_BITS,
                        "uses_shared_envelope": any(
                            name.startswith("shared_sign")
                            for name in allocation
                        ),
                    }
                )
            tail_envelope_allocation = (
                ["int4", "int4"]
                + ["shared_sign16"] * 4
                + ["drop"] * 2
            )
            for method in (
                "qk_fixed_44_tail1shared_tailenv_240b",
                "qk_fixed_44_tail1shared_tailenv_dp4a_240b",
            ):
                used_bits = allocation_physical_bits(
                    tail_options,
                    tail_envelope_allocation,
                )
                head_methods[method] = reconstruct(
                    tail_options,
                    tail_envelope_allocation,
                )
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(tail_envelope_allocation),
                        "total_index_bits": used_bits,
                        "index_ratio_of_full_kv": used_bits / FULL_KV_BITS,
                        "uses_shared_envelope": True,
                    }
                )
            spread_allocations = {
                "qk_spread_allsign_144b": (
                    (0, 128),
                    ["shared_sign16"] * 8,
                ),
                "qk_spread_core4_tail1_208b": (
                    (16, 128),
                    ["int4"] + ["shared_sign16"] * 7,
                ),
                "qk_spread_core42_tail1_240b": (
                    (32, 128),
                    ["int4", "int2"] + ["shared_sign16"] * 6,
                ),
                "qk_spread_core222_tail1_240b": (
                    (48, 128),
                    ["int2", "int2", "int2"] + ["shared_sign16"] * 5,
                ),
                "qk_spread_core8_tail1_272b": (
                    (16, 128),
                    ["int8"] + ["shared_sign16"] * 7,
                ),
                "qk_spread_core44_tail1_272b": (
                    (32, 128),
                    ["int4", "int4"] + ["shared_sign16"] * 6,
                ),
                "qk_spread_core8_tail1_224b": (
                    (16, 80),
                    ["int8"]
                    + ["shared_sign16"] * 4
                    + ["drop"] * 3,
                ),
                "qk_spread_core8_tail1_240b": (
                    (16, 96),
                    ["int8"]
                    + ["shared_sign16"] * 5
                    + ["drop"] * 2,
                ),
                "qk_spread_core8_tail1_256b": (
                    (16, 112),
                    ["int8"]
                    + ["shared_sign16"] * 6
                    + ["drop"],
                ),
            }
            for method, (
                envelope_region,
                allocation,
            ) in spread_allocations.items():
                method_options = spread_options[envelope_region]
                used_bits = allocation_physical_bits(
                    method_options,
                    allocation,
                )
                head_methods[method] = reconstruct(
                    method_options,
                    allocation,
                )
                allocation_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": method,
                        "allocation": "-".join(allocation),
                        "total_index_bits": used_bits,
                        "index_ratio_of_full_kv": (
                            used_bits / FULL_KV_BITS
                        ),
                        "uses_shared_envelope": True,
                    }
                )
            all_int1_allocation = ["int1"] * GROUP_COUNT
            all_int1_bits = allocation_physical_bits(
                options,
                all_int1_allocation,
            )
            head_methods["qk_spread_allint1_bandscale_256b"] = reconstruct(
                options,
                all_int1_allocation,
            )
            allocation_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "method": "qk_spread_allint1_bandscale_256b",
                    "allocation": "-".join(all_int1_allocation),
                    "total_index_bits": all_int1_bits,
                    "index_ratio_of_full_kv": all_int1_bits / FULL_KV_BITS,
                    "uses_shared_envelope": False,
                }
            )
            prepared.append(
                {
                    "key": head_key,
                    "key_mean": key_mean,
                    "query_factor": query_factor,
                    "coordinate_amplitude": coordinate_amplitude,
                    "tail_coordinate_amplitude": (
                        tail_coordinate_amplitude
                    ),
                    "core_reconstruction": core_reconstruction,
                    "jl_methods": jl_methods,
                    "methods": head_methods,
                }
            )

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
                    exact_scores = state["key"] @ head_query * scaling
                    attention = torch.softmax(exact_scores, dim=-1)
                    true_top = torch.topk(
                        exact_scores,
                        k=top_count,
                    ).indices
                    raw_projected_query = (
                        head_query @ state["query_factor"]
                    )
                    projected_query = query_int8(raw_projected_query)
                    for method, reconstructed in state["methods"].items():
                        effective_query = projected_query
                        if method in {
                            "qk_fixed_44_tail1shared_dp4a_240b",
                            "qk_fixed_44_tail1shared_tailenv_dp4a_240b",
                        }:
                            amplitude = (
                                state["tail_coordinate_amplitude"]
                                if "tailenv" in method
                                else state["coordinate_amplitude"]
                            )
                            effective_query = projected_query.clone()
                            effective_query[32:96] = (
                                query_int8(
                                    projected_query[32:96]
                                    * amplitude[32:96]
                                )
                                / amplitude[32:96].clamp_min(1.0e-8)
                            )
                        approximate_scores = (
                            reconstructed @ effective_query
                        ) * scaling
                        if args.center_keys:
                            approximate_scores = approximate_scores + (
                                state["key_mean"] @ head_query
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
                    projected_core_query = query_int8(
                        raw_projected_query[:32]
                    )
                    core_scores = (
                        state["core_reconstruction"] @ projected_core_query
                    )
                    for method, jl_state in state["jl_methods"].items():
                        tail_query = quantize_projection(
                            raw_projected_query[32:]
                            @ jl_state["projection"],
                            8,
                        )
                        approximate_scores = (
                            core_scores
                            + jl_state["reconstructed_sketch"] @ tail_query
                        ) * scaling
                        if args.center_keys:
                            approximate_scores = approximate_scores + (
                                state["key_mean"] @ head_query
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

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    summary = []
    for (method, fraction), items in sorted(grouped.items()):
        method_allocations = [
            row for row in allocation_rows if row["method"] == method
        ]
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "cases": len(items),
            "total_index_bits_mean": sum(
                int(row["total_index_bits"]) for row in method_allocations
            )
            / len(method_allocations),
            "index_ratio_of_full_kv": sum(
                float(row["index_ratio_of_full_kv"])
                for row in method_allocations
            )
            / len(method_allocations),
        }
        for field in (
            "top2_recall",
            "selected_attention_mass",
            "top2_attention_mass_recall",
            "score_pearson",
            "score_rmse",
        ):
            for statistic, value in summarize(
                float(row[field]) for row in items
            ).items():
                result[f"{field}_{statistic}"] = value
        summary.append(result)

    output = {
        "config": {
            **vars(args),
            "trace_path": str(args.trace_path),
            "output_dir": str(args.output_dir),
            "selected_fractions": selected_fractions,
            "shared_sign_widths": SHARED_SIGN_WIDTHS,
            "shared_envelope_bits_per_token": 16,
            "jl_rate_options": JL_RATE_OPTIONS,
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
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
