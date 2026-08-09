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
    quantized_bands,
    reconstruct,
    selection_statistics,
)


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(
        sorted({int(item) for item in specification.split(",") if item.strip()})
    )
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(item) for item in specification.split(",") if item.strip()})
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


def upward_block_quantize(
    values: torch.Tensor,
    *,
    bits: int,
    block_size: int,
) -> torch.Tensor:
    """Quantize nonnegative norms upward, preserving a valid upper bound."""

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if bool((values < 0).any()):
        raise ValueError("upper-bound values must be nonnegative")

    token_count = int(values.numel())
    levels = float((1 << bits) - 1)
    output = torch.empty_like(values)
    infinity = torch.full((), float("inf"), device=values.device)
    for start in range(0, token_count, block_size):
        stop = min(token_count, start + block_size)
        block = values[start:stop]
        maximum = block.max()
        if float(maximum.item()) == 0.0:
            output[start:stop] = 0.0
            continue
        scale = maximum / levels
        code = torch.ceil(block / scale).clamp_(0.0, levels)
        reconstructed = code * scale
        # nextafter protects the analytical certificate from FP roundoff.
        reconstructed = torch.where(
            code == 0,
            torch.zeros_like(reconstructed),
            torch.nextafter(reconstructed, infinity),
        )
        output[start:stop] = torch.maximum(reconstructed, block)
    return output


def interval_candidate_mask(
    base_scores: torch.Tensor,
    bound: torch.Tensor,
    top_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = base_scores - bound
    upper = base_scores + bound
    lower_threshold = torch.topk(lower, k=top_count).values[-1]
    candidate = upper >= lower_threshold
    if int(candidate.sum().item()) < top_count:
        raise RuntimeError("certified interval contains fewer than top-k tokens")
    return candidate, lower_threshold


def certified_bound(
    residual: torch.Tensor,
    query: torch.Tensor,
    omitted_bands: tuple[int, ...],
    *,
    mode: str,
    norm_bits: int,
    norm_block_size: int,
) -> tuple[torch.Tensor, int]:
    if mode == "global":
        key_norm = torch.linalg.vector_norm(residual.float(), dim=-1)
        key_upper = upward_block_quantize(
            key_norm,
            bits=norm_bits,
            block_size=norm_block_size,
        )
        query_norm = torch.linalg.vector_norm(query.float())
        return key_upper * query_norm, 1

    if mode != "bandwise":
        raise ValueError(f"unknown bound mode: {mode}")
    bound = torch.zeros(residual.shape[0], device=residual.device)
    for band_index in omitted_bands:
        start = band_index * GROUP_SIZE
        stop = start + GROUP_SIZE
        key_norm = torch.linalg.vector_norm(
            residual[:, start:stop].float(), dim=-1
        )
        key_upper = upward_block_quantize(
            key_norm,
            bits=norm_bits,
            block_size=norm_block_size,
        )
        query_norm = torch.linalg.vector_norm(query[start:stop].float())
        bound.add_(key_upper * query_norm)
    return bound, len(omitted_bands)


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
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                int(row["base_rate_budget"]),
                float(row["selected_fraction"]),
                row["bound_mode"],
                int(row["norm_bits"]),
                int(row["norm_block_size"]),
            )
        ].append(row)

    metrics = (
        "candidate_ratio",
        "full_proxy_topk_containment",
        "refined_vs_full_proxy_topk_recall",
        "topk_recall",
        "selected_attention_mass",
        "oracle_mass_recall",
        "scan_bits_per_token",
        "access_bits_per_token",
        "access_ratio_of_full_index",
        "access_ratio_of_full_kv",
        "stored_index_ratio_of_full_kv",
        "bound_slack_mean",
    )
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        method, base_budget, fraction, mode, norm_bits, block_size = key
        result: dict[str, Any] = {
            "method": method,
            "base_rate_budget": base_budget,
            "selected_fraction": fraction,
            "bound_mode": mode,
            "norm_bits": norm_bits,
            "norm_block_size": block_size,
            "cases": len(items),
        }
        for metric in metrics:
            for statistic, value in summarize(
                float(item[metric]) for item in items
            ).items():
                result[f"{metric}_{statistic}"] = value
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic top-k boundary refinement. A nested "
            "low-rate index scans every token; upward-quantized residual "
            "norms certify which tokens may cross the full-proxy boundary."
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
    parser.add_argument("--base_rate_budgets", default="5,7,9,11")
    parser.add_argument("--selected_fractions", default="0.04")
    parser.add_argument("--bound_modes", default="global,bandwise")
    parser.add_argument("--norm_bits", default="4,8")
    parser.add_argument("--norm_block_sizes", default="256,1024,32768")
    parser.add_argument(
        "--layers",
        default="",
        help="Optional comma-separated layer subset for fast mechanism screening.",
    )
    parser.add_argument(
        "--max_heldout_steps",
        type=int,
        default=0,
        help="Limit held-out decode steps per layer; zero keeps every step.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    base_budgets = parse_ints(args.base_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    norm_bits_values = parse_ints(args.norm_bits)
    norm_block_sizes = parse_ints(args.norm_block_sizes)
    bound_modes = tuple(
        item.strip() for item in args.bound_modes.split(",") if item.strip()
    )
    requested_layers = (
        set(parse_ints(args.layers)) if args.layers.strip() else None
    )
    if not bound_modes or any(mode not in {"global", "bandwise"} for mode in bound_modes):
        raise ValueError("bound_modes must contain global and/or bandwise")
    if max(base_budgets) >= args.full_rate_budget:
        raise ValueError("base rate budgets must be smaller than full rate")
    if any(not 0.0 < value < 1.0 for value in selected_fractions):
        raise ValueError("selected fractions must lie in (0, 1)")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        layer = int(record["layer"])
        if requested_layers is None or layer in requested_layers:
            by_layer[layer].append(record)
    if not by_layer:
        raise ValueError("the requested layer subset is absent from the trace")

    rows: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda item: int(item["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(f"layer {layer} has no held-out query")
        state_record = next(
            (record for record in layer_records if record.get("key") is not None),
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
                :, kv_head * query_groups : (kv_head + 1) * query_groups
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, _ = qk_balanced_factors(
                head_key[:: args.sample_stride],
                head_calibration,
                args.query_shrinkage,
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            bands = quantized_bands(coefficients, projected_calibration)
            full_allocation = allocate_bits(
                distortion_table_from_bands(
                    coefficients, projected_calibration, bands
                ),
                args.full_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            full_reconstruction = reconstruct(bands, full_allocation)
            full_rate = allocation_rate(full_allocation)
            full_bits = GROUP_SIZE * full_rate

            base_states: dict[int, tuple[tuple[int, ...], torch.Tensor]] = {}
            for base_budget in base_budgets:
                base_allocation = exact_nested_base_allocation(
                    coefficients,
                    projected_calibration,
                    bands,
                    full_allocation,
                    base_budget,
                    args.sample_stride,
                )
                base_reconstruction = reconstruct(bands, base_allocation)
                base_states[base_budget] = (base_allocation, base_reconstruction)
                allocations.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "full_allocation": "-".join(map(str, full_allocation)),
                        "full_rate_units": full_rate,
                        "base_rate_budget": base_budget,
                        "base_allocation": "-".join(map(str, base_allocation)),
                        "base_rate_units": allocation_rate(base_allocation),
                    }
                )

            heldout_records = layer_records[args.calibration_steps :]
            if args.max_heldout_steps > 0:
                heldout_records = heldout_records[: args.max_heldout_steps]
            for record in heldout_records:
                step = int(record["step"])
                query = record["query"].to(device).float()[0, :, 0, :]
                for query_head in range(
                    kv_head * query_groups, (kv_head + 1) * query_groups
                ):
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (
                        coefficients.float() @ projected_query.float()
                    ) * scaling
                    full_scores = (
                        full_reconstruction.float() @ approximate_query.float()
                    ) * scaling

                    for selected_fraction in selected_fractions:
                        top_count = min(
                            token_count,
                            max(1, math.ceil(selected_fraction * token_count)),
                        )
                        full_selected = torch.topk(full_scores, k=top_count).indices
                        for base_budget, (
                            base_allocation,
                            base_reconstruction,
                        ) in base_states.items():
                            base_rate = allocation_rate(base_allocation)
                            base_bits = GROUP_SIZE * base_rate
                            residual = full_reconstruction - base_reconstruction
                            omitted_bands = tuple(
                                index
                                for index, (full_value, base_value) in enumerate(
                                    zip(full_allocation, base_allocation)
                                )
                                if full_value > 0 and base_value == 0
                            )
                            base_scores = (
                                base_reconstruction.float()
                                @ approximate_query.float()
                            ) * scaling
                            true_residual = (full_scores - base_scores).abs()

                            for mode in bound_modes:
                                for norm_bits in norm_bits_values:
                                    for norm_block_size in norm_block_sizes:
                                        bound, norm_streams = certified_bound(
                                            residual,
                                            approximate_query,
                                            omitted_bands,
                                            mode=mode,
                                            norm_bits=norm_bits,
                                            norm_block_size=norm_block_size,
                                        )
                                        bound.mul_(scaling)
                                        if bool((bound + 1.0e-6 < true_residual).any()):
                                            raise AssertionError(
                                                "quantized Cauchy bound was violated"
                                            )
                                        candidate, _ = interval_candidate_mask(
                                            base_scores, bound, top_count
                                        )
                                        candidate_indices = torch.nonzero(
                                            candidate, as_tuple=False
                                        ).flatten()
                                        selected_local = torch.topk(
                                            full_scores[candidate_indices],
                                            k=top_count,
                                        ).indices
                                        selected = candidate_indices[selected_local]
                                        containment = float(
                                            candidate[full_selected]
                                            .float()
                                            .mean()
                                            .item()
                                        )
                                        refined_recall = float(
                                            torch.isin(full_selected, selected)
                                            .float()
                                            .mean()
                                            .item()
                                        )
                                        if containment < 1.0 or refined_recall < 1.0:
                                            raise AssertionError(
                                                "certified refinement lost full-proxy top-k"
                                            )

                                        candidate_ratio = float(
                                            candidate.float().mean().item()
                                        )
                                        blocks = math.ceil(
                                            token_count / norm_block_size
                                        )
                                        scale_bits_per_token = (
                                            16.0
                                            * norm_streams
                                            * blocks
                                            / token_count
                                        )
                                        norm_code_bits = norm_bits * norm_streams
                                        scan_bits = (
                                            base_bits
                                            + norm_code_bits
                                            + scale_bits_per_token
                                        )
                                        access_bits = scan_bits + candidate_ratio * (
                                            full_bits - base_bits
                                        )
                                        stored_bits = (
                                            full_bits
                                            + norm_code_bits
                                            + scale_bits_per_token
                                        )
                                        slack = (bound - true_residual).clamp_min(0)
                                        statistics = selection_statistics(
                                            exact_scores, selected, top_count
                                        )
                                        rows.append(
                                            {
                                                "label": args.label,
                                                "layer": layer,
                                                "kv_head": kv_head,
                                                "query_head": query_head,
                                                "step": step,
                                                "method": "certified_band_refine",
                                                "base_rate_budget": base_budget,
                                                "base_rate_units": base_rate,
                                                "full_rate_units": full_rate,
                                                "selected_fraction": selected_fraction,
                                                "bound_mode": mode,
                                                "norm_bits": norm_bits,
                                                "norm_block_size": norm_block_size,
                                                "norm_streams": norm_streams,
                                                "candidate_ratio": candidate_ratio,
                                                "full_proxy_topk_containment": containment,
                                                "refined_vs_full_proxy_topk_recall": refined_recall,
                                                "scan_bits_per_token": scan_bits,
                                                "access_bits_per_token": access_bits,
                                                "access_ratio_of_full_index": access_bits
                                                / full_bits,
                                                "access_ratio_of_full_kv": access_bits
                                                / FULL_KV_BITS,
                                                "stored_index_ratio_of_full_kv": stored_bits
                                                / FULL_KV_BITS,
                                                "bound_slack_mean": float(
                                                    slack.mean().item()
                                                ),
                                                **statistics,
                                            }
                                        )

        print(
            json.dumps(
                {"label": args.label, "layer": layer, "rows": len(rows)}
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocations)
    summary = aggregate(rows)
    safe = [
        row
        for row in summary
        if row["full_proxy_topk_containment_minimum"] == 1.0
        and row["refined_vs_full_proxy_topk_recall_minimum"] == 1.0
    ]
    best = min(
        safe,
        key=lambda row: (
            row["access_ratio_of_full_index_mean"],
            row["candidate_ratio_mean"],
        ),
    )
    payload = {
        "schema": "qksieve_certified_band_refinement_v1",
        "trace_path": str(args.trace_path),
        "layers": sorted(by_layer),
        "max_heldout_steps": args.max_heldout_steps,
        "certificate": (
            "For omitted bands, |delta_i| <= sum_g ||q_g||_2 "
            "||k_i,g||_2. Norms are blockwise upward-quantized. "
            "Tokens whose upper score can cross the kth-largest lower "
            "score are refined with the complete packed proxy."
        ),
        "rows": len(rows),
        "best_safe": best,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
