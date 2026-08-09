from __future__ import annotations

import argparse
import csv
import itertools
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
    metric_scale_quantize_band,
    qk_balanced_factors,
)


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(
        sorted(
            {
                int(item)
                for item in specification.split(",")
                if item.strip()
            }
        )
    )
    if not values:
        raise ValueError("expected at least one integer")
    return values


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
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantized_bands(
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


def allocation_rate(allocation: Iterable[int]) -> int:
    return sum(int(bits) + int(int(bits) > 0) for bits in allocation)


def exact_nested_base_allocation(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    bands: list[dict[int, torch.Tensor]],
    full_allocation: tuple[int, ...],
    base_rate_budget: int,
    sample_stride: int,
) -> tuple[int, ...]:
    """Select complete full-index bands under a smaller scan budget.

    The base representation is physically nested in the full representation:
    a band is either read at its full stored precision or not read at all.
    There are at most eight active bands, so exhaustive subset search gives
    the exact calibration qMSE optimum, including cross-band error terms.
    """

    active = [
        index for index, bits in enumerate(full_allocation) if bits > 0
    ]
    sampled_coefficients = coefficients[::sample_stride]
    best: tuple[float, int, tuple[int, ...]] | None = None
    for enabled_mask in itertools.product((False, True), repeat=len(active)):
        allocation = [0] * GROUP_COUNT
        for enabled, band_index in zip(enabled_mask, active):
            if enabled:
                allocation[band_index] = full_allocation[band_index]
        rate = allocation_rate(allocation)
        if rate > base_rate_budget:
            continue
        approximate = reconstruct(bands, allocation)[::sample_stride]
        score_error = (
            calibration_queries.float()
            @ (sampled_coefficients - approximate)
            .float()
            .transpose(0, 1)
        )
        candidate = (
            float(score_error.square().mean().item()),
            -rate,
            tuple(allocation),
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("no nested base allocation fits the rate budget")
    return best[2]


def systematic_sample_indices(
    token_count: int,
    sample_count: int,
    phase: int,
    device: torch.device,
) -> torch.Tensor:
    actual = min(token_count, sample_count)
    stride = max(1, token_count // actual)
    while math.gcd(stride, token_count) != 1:
        stride += 1
    offsets = torch.arange(actual, device=device)
    return (phase + offsets * stride) % token_count


def conformal_radius(errors: torch.Tensor, alpha: float) -> torch.Tensor:
    count = int(errors.numel())
    rank = min(
        count,
        max(1, math.ceil((count + 1) * (1.0 - alpha))),
    )
    return torch.kthvalue(errors.flatten(), rank).values


def interval_candidates(
    approximate_scores: torch.Tensor,
    radius: torch.Tensor | float,
    top_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = approximate_scores - radius
    upper = approximate_scores + radius
    lower_threshold = torch.topk(lower, k=top_count).values[-1]
    candidate = upper >= lower_threshold
    if int(candidate.sum().item()) < top_count:
        raise RuntimeError("interval set cannot contain the requested top-k")
    return candidate, lower_threshold


def refine_and_select(
    base_scores: torch.Tensor,
    full_scores: torch.Tensor,
    radius: torch.Tensor | float,
    top_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate, threshold = interval_candidates(
        base_scores,
        radius,
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
    selected = candidate_indices[selected_local]
    return selected, candidate


def selection_statistics(
    exact_scores: torch.Tensor,
    selected: torch.Tensor,
    top_count: int,
) -> dict[str, float]:
    token_count = int(exact_scores.numel())
    true_top = torch.topk(exact_scores, k=top_count).indices
    selected_mask = torch.zeros(
        token_count,
        dtype=torch.bool,
        device=exact_scores.device,
    )
    selected_mask[selected] = True
    attention = torch.softmax(exact_scores.float(), dim=-1)
    selected_mass = attention[selected].sum()
    oracle_mass = attention[true_top].sum().clamp_min(1.0e-12)
    return {
        "topk_recall": float(
            selected_mask[true_top].float().mean().item()
        ),
        "selected_attention_mass": float(selected_mass.item()),
        "oracle_mass_recall": float(
            (selected_mass / oracle_mass).item()
        ),
    }


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, int, float, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["method"]),
                int(row["base_rate_budget"]),
                float(row["selected_fraction"]),
                float(row["alpha"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "topk_recall",
        "selected_attention_mass",
        "oracle_mass_recall",
        "refinement_ratio",
        "interval_full_proxy_topk_recall",
        "sampled_radius_token_coverage",
        "access_rate_units",
        "access_ratio_of_full_index",
        "access_ratio_of_full_kv",
    )
    for key, items in sorted(grouped.items()):
        method, base_budget, fraction, alpha = key
        result: dict[str, Any] = {
            "method": method,
            "base_rate_budget": base_budget,
            "selected_fraction": fraction,
            "alpha": alpha,
            "cases": len(items),
            "full_rate_units_mean": sum(
                float(item["full_rate_units"]) for item in items
            )
            / len(items),
            "base_rate_units_mean": sum(
                float(item["base_rate_units"]) for item in items
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
            "Evaluate QK-balanced progressive spectral-band reads. A nested "
            "base index scans every token; sampled score-error intervals "
            "trigger full packed-index reads only near the top-k boundary."
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
    parser.add_argument("--alphas", default="0.05,0.02,0.01,0.005")
    parser.add_argument("--current_query_samples", type=int, default=256)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    base_budgets = parse_ints(args.base_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    alphas = parse_floats(args.alphas)
    if min(base_budgets) < 0:
        raise ValueError("base rate budgets cannot be negative")
    if max(base_budgets) >= args.full_rate_budget:
        raise ValueError("base rate budgets must be below the full rate")
    if any(not 0.0 < value < 1.0 for value in selected_fractions):
        raise ValueError("selected fractions must be in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("alphas must be in (0, 1)")
    if args.current_query_samples <= 0:
        raise ValueError("current-query sample count must be positive")

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
        if query_heads % kv_heads != 0:
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
                tuple[tuple[int, ...], torch.Tensor],
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
                base_states[base_budget] = (
                    base_allocation,
                    base_reconstruction,
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
                    phase = (
                        step * 1009
                        + layer * 131
                        + kv_head * 31
                        + query_head * 17
                    ) % token_count
                    sample = systematic_sample_indices(
                        token_count,
                        args.current_query_samples,
                        phase,
                        device,
                    )

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
                        full_statistics = selection_statistics(
                            exact_scores,
                            full_selected,
                            top_count,
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "step": step,
                                "method": "full_packed_proxy",
                                "base_rate_budget": full_rate,
                                "selected_fraction": selected_fraction,
                                "alpha": 0.0,
                                "full_rate_units": full_rate,
                                "base_rate_units": full_rate,
                                "refinement_ratio": 0.0,
                                "interval_full_proxy_topk_recall": 1.0,
                                "sampled_radius_token_coverage": 1.0,
                                "access_rate_units": float(full_rate),
                                "access_ratio_of_full_index": 1.0,
                                "access_ratio_of_full_kv": (
                                    GROUP_SIZE * full_rate / FULL_KV_BITS
                                ),
                                **full_statistics,
                            }
                        )

                        for base_budget, (
                            base_allocation,
                            base_reconstruction,
                        ) in base_states.items():
                            base_rate = allocation_rate(base_allocation)
                            base_scores = (
                                base_reconstruction.float()
                                @ approximate_query.float()
                            ) * scaling
                            base_selected = torch.topk(
                                base_scores,
                                k=top_count,
                            ).indices
                            base_statistics = selection_statistics(
                                exact_scores,
                                base_selected,
                                top_count,
                            )
                            rows.append(
                                {
                                    "label": args.label,
                                    "layer": layer,
                                    "kv_head": kv_head,
                                    "query_head": query_head,
                                    "step": step,
                                    "method": "base_only",
                                    "base_rate_budget": base_budget,
                                    "selected_fraction": selected_fraction,
                                    "alpha": 0.0,
                                    "full_rate_units": full_rate,
                                    "base_rate_units": base_rate,
                                    "refinement_ratio": 0.0,
                                    "interval_full_proxy_topk_recall": float(
                                        torch.isin(
                                            full_selected,
                                            base_selected,
                                        )
                                        .float()
                                        .mean()
                                        .item()
                                    ),
                                    "sampled_radius_token_coverage": 0.0,
                                    "access_rate_units": float(base_rate),
                                    "access_ratio_of_full_index": (
                                        base_rate / full_rate
                                    ),
                                    "access_ratio_of_full_kv": (
                                        GROUP_SIZE
                                        * base_rate
                                        / FULL_KV_BITS
                                    ),
                                    **base_statistics,
                                }
                            )

                            score_error = (
                                full_scores - base_scores
                            ).abs()
                            sampled_error = score_error[sample]
                            extra_rate = full_rate - base_rate
                            for alpha in alphas:
                                radius = conformal_radius(
                                    sampled_error,
                                    alpha,
                                )
                                selected, candidate = refine_and_select(
                                    base_scores,
                                    full_scores,
                                    radius,
                                    top_count,
                                )
                                statistics = selection_statistics(
                                    exact_scores,
                                    selected,
                                    top_count,
                                )
                                candidate_mask = candidate
                                full_selected_recall = float(
                                    candidate_mask[full_selected]
                                    .float()
                                    .mean()
                                    .item()
                                )
                                refinement_ratio = float(
                                    candidate.float().mean().item()
                                )
                                # Current-query calibration reads the full
                                # code for a small systematic sample. Count it
                                # conservatively even when sample/candidate
                                # accesses overlap.
                                sample_ratio = min(
                                    1.0,
                                    args.current_query_samples
                                    / token_count,
                                )
                                access_rate = (
                                    base_rate
                                    + extra_rate
                                    * (refinement_ratio + sample_ratio)
                                )
                                rows.append(
                                    {
                                        "label": args.label,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "step": step,
                                        "method": "progressive_interval",
                                        "base_rate_budget": base_budget,
                                        "selected_fraction": selected_fraction,
                                        "alpha": alpha,
                                        "full_rate_units": full_rate,
                                        "base_rate_units": base_rate,
                                        "refinement_ratio": (
                                            refinement_ratio
                                        ),
                                        "interval_full_proxy_topk_recall": (
                                            full_selected_recall
                                        ),
                                        "sampled_radius_token_coverage": float(
                                            (
                                                score_error <= radius
                                            )
                                            .float()
                                            .mean()
                                            .item()
                                        ),
                                        "access_rate_units": access_rate,
                                        "access_ratio_of_full_index": (
                                            access_rate / full_rate
                                        ),
                                        "access_ratio_of_full_kv": (
                                            GROUP_SIZE
                                            * access_rate
                                            / FULL_KV_BITS
                                        ),
                                        **statistics,
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
