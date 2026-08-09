#!/usr/bin/env python
"""Evaluate nested low-rate scan plus high-rate candidate refinement."""

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
    values = tuple(sorted({int(item) for item in specification.split(",") if item.strip()}))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item) for item in specification.split(",") if item.strip()}))
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["base_rate_budget"]),
                float(row["selected_fraction"]),
                float(row["overfetch_factor"]),
            )
        ].append(row)
    metrics = (
        "candidate_ratio",
        "full_proxy_topk_containment",
        "refined_vs_full_proxy_topk_recall",
        "topk_recall",
        "selected_attention_mass",
        "oracle_mass_recall",
        "mass_retention_vs_full_proxy",
        "scan_bits_per_token",
        "access_bits_per_token",
        "access_ratio_of_full_index",
        "access_ratio_of_full_kv",
        "stored_index_ratio_of_full_kv",
    )
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        base_budget, selected_fraction, overfetch = key
        result: dict[str, Any] = {
            "method": "nested_overfetch_refine",
            "base_rate_budget": base_budget,
            "selected_fraction": selected_fraction,
            "overfetch_factor": overfetch,
            "cases": len(items),
        }
        for metric in metrics:
            for statistic, value in summarize(float(item[metric]) for item in items).items():
                result[f"{metric}_{statistic}"] = value
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a nested low-rate QKSieve proxy, overfetch candidates, "
            "then read the remaining packed bands only for those candidates."
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
    parser.add_argument("--overfetch_factors", default="1.5,2,3,4,6,8")
    parser.add_argument("--layers", default="")
    parser.add_argument("--max_heldout_steps", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    base_budgets = parse_ints(args.base_rate_budgets)
    selected_fractions = parse_floats(args.selected_fractions)
    overfetch_factors = parse_floats(args.overfetch_factors)
    requested_layers = set(parse_ints(args.layers)) if args.layers.strip() else None
    if max(base_budgets) >= args.full_rate_budget:
        raise ValueError("base rate budgets must be smaller than full rate")
    if any(not 0.0 < value < 1.0 for value in selected_fractions):
        raise ValueError("selected fractions must lie in (0, 1)")
    if any(value < 1.0 for value in overfetch_factors):
        raise ValueError("overfetch factors must be at least one")

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
        state_record = next((record for record in layer_records if record.get("key") is not None), None)
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
                head_key[:: args.sample_stride], head_calibration, args.query_shrinkage
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            bands = quantized_bands(coefficients, projected_calibration)
            full_allocation = allocate_bits(
                distortion_table_from_bands(coefficients, projected_calibration, bands),
                args.full_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            full_reconstruction = reconstruct(bands, full_allocation)
            full_bits = GROUP_SIZE * allocation_rate(full_allocation)
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
                        "full_bits_per_token": full_bits,
                        "base_rate_budget": base_budget,
                        "base_allocation": "-".join(map(str, base_allocation)),
                        "base_bits_per_token": GROUP_SIZE * allocation_rate(base_allocation),
                    }
                )

            heldout_records = layer_records[args.calibration_steps :]
            if args.max_heldout_steps > 0:
                heldout_records = heldout_records[: args.max_heldout_steps]
            for record in heldout_records:
                step = int(record["step"])
                query = record["query"].to(device).float()[0, :, 0, :]
                for query_head in range(kv_head * query_groups, (kv_head + 1) * query_groups):
                    projected_query = query[query_head] @ query_factor
                    approximate_query = query_int8(projected_query)
                    exact_scores = (coefficients.float() @ projected_query.float()) * scaling
                    full_scores = (full_reconstruction.float() @ approximate_query.float()) * scaling

                    for selected_fraction in selected_fractions:
                        top_count = min(token_count, max(1, math.ceil(selected_fraction * token_count)))
                        full_selected = torch.topk(full_scores, k=top_count).indices
                        full_stats = selection_statistics(exact_scores, full_selected, top_count)
                        full_mass = max(float(full_stats["selected_attention_mass"]), 1.0e-30)
                        for base_budget, (base_allocation, base_reconstruction) in base_states.items():
                            base_bits = GROUP_SIZE * allocation_rate(base_allocation)
                            base_scores = (base_reconstruction.float() @ approximate_query.float()) * scaling
                            for overfetch in overfetch_factors:
                                candidate_count = min(
                                    token_count,
                                    max(top_count, math.ceil(overfetch * top_count)),
                                )
                                candidate = torch.topk(base_scores, k=candidate_count).indices
                                selected_local = torch.topk(full_scores[candidate], k=top_count).indices
                                selected = candidate[selected_local]
                                containment = float(
                                    torch.isin(full_selected, candidate).float().mean().item()
                                )
                                refined_recall = float(
                                    torch.isin(full_selected, selected).float().mean().item()
                                )
                                candidate_ratio = candidate_count / token_count
                                access_bits = base_bits + candidate_ratio * (full_bits - base_bits)
                                statistics = selection_statistics(exact_scores, selected, top_count)
                                rows.append(
                                    {
                                        "label": args.label,
                                        "layer": layer,
                                        "kv_head": kv_head,
                                        "query_head": query_head,
                                        "step": step,
                                        "method": "nested_overfetch_refine",
                                        "base_rate_budget": base_budget,
                                        "base_allocation": "-".join(map(str, base_allocation)),
                                        "full_allocation": "-".join(map(str, full_allocation)),
                                        "selected_fraction": selected_fraction,
                                        "overfetch_factor": overfetch,
                                        "candidate_ratio": candidate_ratio,
                                        "full_proxy_topk_containment": containment,
                                        "refined_vs_full_proxy_topk_recall": refined_recall,
                                        "mass_retention_vs_full_proxy": float(statistics["selected_attention_mass"]) / full_mass,
                                        "scan_bits_per_token": base_bits,
                                        "access_bits_per_token": access_bits,
                                        "access_ratio_of_full_index": access_bits / full_bits,
                                        "access_ratio_of_full_kv": access_bits / FULL_KV_BITS,
                                        "stored_index_ratio_of_full_kv": full_bits / FULL_KV_BITS,
                                        **statistics,
                                    }
                                )
        print(json.dumps({"label": args.label, "layer": layer, "rows": len(rows)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "allocations.csv", allocations)
    summary = aggregate(rows)
    near_lossless = [
        row
        for row in summary
        if row["full_proxy_topk_containment_mean"] >= 0.995
        and row["mass_retention_vs_full_proxy_mean"] >= 0.999
    ]
    best = min(
        near_lossless,
        key=lambda row: (row["access_ratio_of_full_index_mean"], row["candidate_ratio_mean"]),
    ) if near_lossless else None
    report = {
        "schema": "qksieve_nested_overfetch_refinement_v1",
        "trace_path": str(args.trace_path),
        "layers": sorted(by_layer),
        "max_heldout_steps": args.max_heldout_steps,
        "quality_boundary": (
            "Candidate selection uses exact top-k over the low-rate proxy; "
            "the final selection uses exact top-k over the complete proxy "
            "inside candidates. This is a mechanism frontier, not a timed kernel."
        ),
        "best_near_lossless": best,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
