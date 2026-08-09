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
from analyze_hierarchical_spectral_quantization_20260727 import query_int8
from analyze_qk_balanced_spectral_rate_20260727 import (
    distortion_table_from_bands,
    qk_balanced_factors,
)
from analyze_qk_progressive_refinement_20260727 import (
    allocation_rate,
    exact_nested_base_allocation,
    interval_candidates,
    parse_floats,
    parse_ints,
    quantized_bands,
    reconstruct,
    selection_statistics,
    summarize,
)


def conservative_log_quantize(
    values: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Quantize nonnegative values upward, reserving code zero for zero."""

    if bits <= 0:
        raise ValueError("bits must be positive")
    if torch.any(values < 0):
        raise ValueError("norm values must be nonnegative")
    output = torch.zeros_like(values)
    positive = values > 0
    if not bool(positive.any()):
        return output
    positive_values = values[positive].float()
    minimum = positive_values.min()
    maximum = positive_values.max()
    positive_levels = (1 << bits) - 1
    if positive_levels == 1 or bool(minimum == maximum):
        output[positive] = maximum.to(output.dtype)
        return output

    log_minimum = torch.log2(minimum)
    log_maximum = torch.log2(maximum)
    step = (log_maximum - log_minimum) / (positive_levels - 1)
    indices = torch.ceil(
        (torch.log2(positive_values) - log_minimum) / step
    ).clamp_(0, positive_levels - 1)
    reconstructed = torch.exp2(log_minimum + indices * step)
    reconstructed.mul_(1.0 + 8.0 * torch.finfo(reconstructed.dtype).eps)
    reconstructed = torch.nextafter(
        reconstructed,
        torch.full_like(reconstructed, float("inf")),
    )
    output[positive] = reconstructed.to(output.dtype)
    if not bool(torch.all(output[positive] >= values[positive])):
        raise RuntimeError("conservative norm quantization rounded downward")
    return output


def residual_norm_bound(
    residual: torch.Tensor,
    query: torch.Tensor,
    differing_bands: tuple[int, ...],
    bits: int,
    mode: str,
) -> tuple[torch.Tensor, int, int]:
    """Return a conservative Cauchy bound and its physical storage cost."""

    token_count = int(residual.shape[0])
    if mode == "global":
        mask = torch.zeros(
            residual.shape[-1],
            dtype=torch.bool,
            device=residual.device,
        )
        for band in differing_bands:
            start = band * GROUP_SIZE
            mask[start : start + GROUP_SIZE] = True
        key_norm = conservative_log_quantize(
            torch.linalg.vector_norm(residual[:, mask], dim=-1),
            bits,
        )
        query_norm = torch.linalg.vector_norm(query[mask])
        bound = key_norm * query_norm
        scalar_count = 1
    elif mode == "per_band":
        bound = torch.zeros(
            token_count,
            device=residual.device,
            dtype=torch.float32,
        )
        for band in differing_bands:
            start = band * GROUP_SIZE
            stop = start + GROUP_SIZE
            key_norm = conservative_log_quantize(
                torch.linalg.vector_norm(
                    residual[:, start:stop],
                    dim=-1,
                ),
                bits,
            )
            query_norm = torch.linalg.vector_norm(query[start:stop])
            bound.add_(key_norm * query_norm)
        scalar_count = len(differing_bands)
    else:
        raise ValueError(f"unknown norm mode: {mode}")

    code_bits_per_token = scalar_count * bits
    # Each scalar stream stores two FP16 log endpoints per head.
    metadata_bits = scalar_count * 32
    return bound, code_bits_per_token, metadata_bits


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int, int, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["norm_mode"]),
                int(row["norm_bits"]),
                int(row["base_rate_budget"]),
                float(row["selected_fraction"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "topk_recall",
        "selected_attention_mass",
        "oracle_mass_recall",
        "full_proxy_topk_containment",
        "bound_token_coverage",
        "refinement_ratio",
        "bound_to_error_mean_ratio",
        "scan_bits_per_token",
        "scan_ratio_of_full_index",
        "scan_ratio_of_full_kv",
    )
    for key, items in sorted(grouped.items()):
        norm_mode, norm_bits, base_budget, selected_fraction = key
        result: dict[str, Any] = {
            "norm_mode": norm_mode,
            "norm_bits": norm_bits,
            "base_rate_budget": base_budget,
            "selected_fraction": selected_fraction,
            "cases": len(items),
            "full_rate_units_mean": sum(
                float(item["full_rate_units"]) for item in items
            )
            / len(items),
            "base_rate_units_mean": sum(
                float(item["base_rate_units"]) for item in items
            )
            / len(items),
            "residual_band_count_mean": sum(
                float(item["residual_band_count"]) for item in items
            )
            / len(items),
        }
        for metric in metrics:
            result.update(
                {
                    f"{metric}_{statistic}": value
                    for statistic, value in summarize(
                        float(item[metric]) for item in items
                    ).items()
                }
            )
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use conservatively quantized residual norms to certify which "
            "tokens can cross the top-k boundary before reading finer QK "
            "spectral bands."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--full_rate_budget", type=int, default=15)
    parser.add_argument("--base_rate_budgets", default="5,7,9,11,13")
    parser.add_argument("--selected_fractions", default="0.01,0.04")
    parser.add_argument("--norm_bits", default="2,4,8")
    parser.add_argument("--norm_modes", default="global,per_band")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    base_budgets = parse_ints(args.base_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    norm_bits = parse_ints(args.norm_bits)
    norm_modes = tuple(
        item.strip()
        for item in args.norm_modes.split(",")
        if item.strip()
    )
    if min(base_budgets) < 0 or max(base_budgets) >= args.full_rate_budget:
        raise ValueError("base budgets must be below the full rate")
    if min(norm_bits) <= 0 or max(norm_bits) > 16:
        raise ValueError("norm bits must be in [1, 16]")
    if not norm_modes or any(
        mode not in {"global", "per_band"} for mode in norm_modes
    ):
        raise ValueError("norm modes must be global and/or per_band")
    if any(not 0.0 < value < 1.0 for value in selected_fractions):
        raise ValueError("selected fractions must be in (0, 1)")

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
        layer_records.sort(key=lambda item: int(item["step"]))
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
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        query_groups = query_heads // kv_heads
        token_count = int(key.shape[-2])

        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            head_calibration = calibration[
                :,
                kv_head * query_groups : (kv_head + 1) * query_groups,
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, _ = qk_balanced_factors(
                head_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            bands = quantized_bands(
                coefficients,
                projected_calibration,
            )
            full_allocation = allocate_bits(
                distortion_table_from_bands(
                    coefficients,
                    projected_calibration,
                    bands,
                ),
                args.full_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            full_reconstruction = reconstruct(bands, full_allocation)
            full_rate = allocation_rate(full_allocation)
            base_states: dict[
                int,
                tuple[
                    tuple[int, ...],
                    torch.Tensor,
                    torch.Tensor,
                    tuple[int, ...],
                ],
            ] = {}
            for base_budget in base_budgets:
                base_allocation = exact_nested_base_allocation(
                    coefficients,
                    projected_calibration,
                    bands,
                    full_allocation,
                    base_budget,
                    args.sample_stride,
                )
                base_reconstruction = reconstruct(
                    bands,
                    base_allocation,
                )
                residual = full_reconstruction - base_reconstruction
                differing_bands = tuple(
                    index
                    for index, (full_bits, base_bits) in enumerate(
                        zip(full_allocation, base_allocation)
                    )
                    if full_bits != base_bits
                )
                base_states[base_budget] = (
                    base_allocation,
                    base_reconstruction,
                    residual,
                    differing_bands,
                )
                allocations.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "full_allocation": "-".join(
                            str(bits) for bits in full_allocation
                        ),
                        "full_rate_units": full_rate,
                        "base_rate_budget": base_budget,
                        "base_allocation": "-".join(
                            str(bits) for bits in base_allocation
                        ),
                        "base_rate_units": allocation_rate(
                            base_allocation
                        ),
                        "residual_bands": "-".join(
                            str(index) for index in differing_bands
                        ),
                    }
                )

            for record in layer_records[args.calibration_steps :]:
                step = int(record["step"])
                query = record["query"].to(device).float()[0, :, 0, :]
                for query_head in range(
                    kv_head * query_groups,
                    (kv_head + 1) * query_groups,
                ):
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (
                        coefficients.float() @ projected_query.float()
                    ) * scaling
                    full_scores = (
                        full_reconstruction.float()
                        @ approximate_query.float()
                    ) * scaling

                    for selected_fraction in selected_fractions:
                        top_count = min(
                            token_count,
                            max(
                                1,
                                math.ceil(
                                    selected_fraction * token_count
                                ),
                            ),
                        )
                        full_selected = torch.topk(
                            full_scores,
                            k=top_count,
                        ).indices
                        for base_budget, (
                            base_allocation,
                            base_reconstruction,
                            residual,
                            differing_bands,
                        ) in base_states.items():
                            base_rate = allocation_rate(base_allocation)
                            base_scores = (
                                base_reconstruction.float()
                                @ approximate_query.float()
                            ) * scaling
                            absolute_error = (
                                full_scores - base_scores
                            ).abs()
                            missing_rate = full_rate - base_rate

                            for mode in norm_modes:
                                for bits in norm_bits:
                                    bound, code_bits, metadata_bits = (
                                        residual_norm_bound(
                                            residual,
                                            approximate_query,
                                            differing_bands,
                                            bits,
                                            mode,
                                        )
                                    )
                                    bound.mul_(scaling)
                                    candidate, _ = interval_candidates(
                                        base_scores,
                                        bound,
                                        top_count,
                                    )
                                    candidate_indices = torch.nonzero(
                                        candidate,
                                        as_tuple=False,
                                    ).flatten()
                                    selected_local = torch.topk(
                                        full_scores[candidate_indices],
                                        k=top_count,
                                    ).indices
                                    selected = candidate_indices[
                                        selected_local
                                    ]
                                    statistics = selection_statistics(
                                        exact_scores,
                                        selected,
                                        top_count,
                                    )
                                    refinement_ratio = float(
                                        candidate.float().mean().item()
                                    )
                                    scan_bits = (
                                        GROUP_SIZE * base_rate
                                        + code_bits
                                        + metadata_bits / token_count
                                        + GROUP_SIZE
                                        * missing_rate
                                        * refinement_ratio
                                    )
                                    denominator = (
                                        absolute_error.mean().item()
                                    )
                                    rows.append(
                                        {
                                            "label": args.label,
                                            "layer": layer,
                                            "kv_head": kv_head,
                                            "query_head": query_head,
                                            "step": step,
                                            "norm_mode": mode,
                                            "norm_bits": bits,
                                            "base_rate_budget": base_budget,
                                            "selected_fraction": (
                                                selected_fraction
                                            ),
                                            "full_rate_units": full_rate,
                                            "base_rate_units": base_rate,
                                            "residual_band_count": len(
                                                differing_bands
                                            ),
                                            "topk_recall": statistics[
                                                "topk_recall"
                                            ],
                                            "selected_attention_mass": (
                                                statistics[
                                                    "selected_attention_mass"
                                                ]
                                            ),
                                            "oracle_mass_recall": (
                                                statistics[
                                                    "oracle_mass_recall"
                                                ]
                                            ),
                                            "full_proxy_topk_containment": (
                                                candidate[full_selected]
                                                .float()
                                                .mean()
                                                .item()
                                            ),
                                            "bound_token_coverage": (
                                                (
                                                    absolute_error
                                                    <= bound + 1.0e-6
                                                )
                                                .float()
                                                .mean()
                                                .item()
                                            ),
                                            "refinement_ratio": (
                                                refinement_ratio
                                            ),
                                            "bound_to_error_mean_ratio": (
                                                bound.mean().item()
                                                / max(
                                                    denominator,
                                                    1.0e-12,
                                                )
                                            ),
                                            "scan_bits_per_token": scan_bits,
                                            "scan_ratio_of_full_index": (
                                                scan_bits
                                                / (GROUP_SIZE * full_rate)
                                            ),
                                            "scan_ratio_of_full_kv": (
                                                scan_bits / FULL_KV_BITS
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
    summary = aggregate(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
