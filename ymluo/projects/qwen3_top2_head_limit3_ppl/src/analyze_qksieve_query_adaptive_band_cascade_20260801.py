#!/usr/bin/env python
"""Audit a training-free, query-adaptive QKSieve band cascade.

The full mixed-bit index is stored in band-major form.  At each decode step,
small per-band second-moment matrices rank bands without scanning history.  A
coarse score scans only the bands that explain a requested fraction of the
current Query's estimated score energy.  The complete proxy is then evaluated
only for an overfetched candidate set.

This script measures the mechanism frontier.  It does not claim CUDA latency;
that requires a fused band-major scan/refinement kernel.
"""

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
    quantized_bands,
    reconstruct,
    selection_statistics,
)


def parse_floats(specification: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(item) for item in specification.split(",") if item.strip()})
    )
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def parse_ints(specification: str) -> tuple[int, ...]:
    values = tuple(
        sorted({int(item) for item in specification.split(",") if item.strip()})
    )
    if not values:
        raise ValueError("expected at least one integer")
    return values


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["energy_coverage_target"]),
                float(row["selected_fraction"]),
                float(row["overfetch_factor"]),
            )
        ].append(row)
    metrics = (
        "selected_band_count",
        "selected_band_rate_units",
        "selected_band_bits_per_token",
        "realized_energy_coverage",
        "candidate_ratio",
        "full_proxy_topk_containment",
        "refined_vs_full_proxy_topk_recall",
        "topk_recall",
        "selected_attention_mass",
        "oracle_mass_recall",
        "mass_retention_vs_full_proxy",
        "access_bits_per_token",
        "access_ratio_of_full_index",
        "access_ratio_of_full_kv",
    )
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        coverage, selected_fraction, overfetch = key
        result: dict[str, Any] = {
            "method": "query_adaptive_band_cascade",
            "energy_coverage_target": coverage,
            "selected_fraction": selected_fraction,
            "overfetch_factor": overfetch,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--full_rate_budget", type=int, default=15)
    parser.add_argument(
        "--energy_coverages", default="0.8,0.9,0.95,0.98,0.99,0.995"
    )
    parser.add_argument("--selected_fractions", default="0.04")
    parser.add_argument("--overfetch_factors", default="1.5,2,3,4,6")
    parser.add_argument("--layers", default="")
    parser.add_argument("--max_heldout_steps", type=int, default=0)
    return parser.parse_args()


def selected_band_mask(
    second_moments: list[torch.Tensor],
    query: torch.Tensor,
    allocation: tuple[int, ...],
    target_coverage: float,
) -> tuple[list[int], float, list[float]]:
    energies: list[float] = []
    active: list[int] = []
    for band, bits in enumerate(allocation):
        if bits <= 0:
            energies.append(0.0)
            continue
        start = band * GROUP_SIZE
        stop = start + GROUP_SIZE
        query_band = query[start:stop].float()
        energy = float(
            (query_band @ second_moments[band] @ query_band)
            .clamp_min(0.0)
            .item()
        )
        energies.append(energy)
        active.append(band)
    if not active:
        raise RuntimeError("the full allocation contains no active band")
    total = max(sum(energies), 1.0e-30)
    ranked = sorted(active, key=lambda band: (-energies[band], band))
    chosen: list[int] = []
    retained = 0.0
    for band in ranked:
        chosen.append(band)
        retained += energies[band]
        if retained / total >= target_coverage:
            break
    return sorted(chosen), retained / total, energies


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    coverages = parse_floats(args.energy_coverages)
    selected_fractions = parse_floats(args.selected_fractions)
    overfetch_factors = parse_floats(args.overfetch_factors)
    requested_layers = set(parse_ints(args.layers)) if args.layers.strip() else None
    if any(not 0.0 < value <= 1.0 for value in coverages):
        raise ValueError("energy coverages must lie in (0, 1]")
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
    head_rows: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda item: int(item["step"]))
        if len(layer_records) <= args.calibration_steps:
            raise ValueError(f"layer {layer} has no held-out query")
        state_record = next(
            (record for record in layer_records if record.get("key") is not None), None
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
            full_reconstruction = reconstruct(bands, full_allocation).float()
            full_rate_units = allocation_rate(full_allocation)
            full_bits = GROUP_SIZE * full_rate_units
            second_moments: list[torch.Tensor] = []
            band_score_components: list[torch.Tensor] = []
            for band in range(len(full_allocation)):
                start = band * GROUP_SIZE
                stop = start + GROUP_SIZE
                values = full_reconstruction[:, start:stop]
                second_moments.append(
                    (values.transpose(0, 1) @ values) / max(1, token_count)
                )
                band_score_components.append(values)
            head_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "full_allocation": "-".join(map(str, full_allocation)),
                    "full_rate_units": full_rate_units,
                    "full_bits_per_token": full_bits,
                    "stored_index_ratio_of_full_kv": full_bits / FULL_KV_BITS,
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
                    approximate_query = query_int8(projected_query).float()
                    exact_scores = (
                        coefficients.float() @ projected_query.float()
                    ) * scaling
                    per_band_scores = []
                    for band, values in enumerate(band_score_components):
                        start = band * GROUP_SIZE
                        stop = start + GROUP_SIZE
                        per_band_scores.append(
                            (values @ approximate_query[start:stop]) * scaling
                        )
                    full_scores = torch.stack(per_band_scores, dim=0).sum(dim=0)

                    for selected_fraction in selected_fractions:
                        top_count = min(
                            token_count,
                            max(1, math.ceil(selected_fraction * token_count)),
                        )
                        full_selected = torch.topk(full_scores, k=top_count).indices
                        full_statistics = selection_statistics(
                            exact_scores, full_selected, top_count
                        )
                        full_mass = max(
                            float(full_statistics["selected_attention_mass"]), 1.0e-30
                        )
                        for coverage in coverages:
                            chosen, realized_coverage, energies = selected_band_mask(
                                second_moments,
                                approximate_query,
                                full_allocation,
                                coverage,
                            )
                            chosen_set = set(chosen)
                            base_allocation = tuple(
                                bits if band in chosen_set else 0
                                for band, bits in enumerate(full_allocation)
                            )
                            base_rate_units = allocation_rate(base_allocation)
                            base_bits = GROUP_SIZE * base_rate_units
                            base_scores = torch.stack(
                                [per_band_scores[band] for band in chosen], dim=0
                            ).sum(dim=0)
                            for overfetch in overfetch_factors:
                                candidate_count = min(
                                    token_count,
                                    max(top_count, math.ceil(overfetch * top_count)),
                                )
                                candidate = torch.topk(
                                    base_scores, k=candidate_count
                                ).indices
                                selected_local = torch.topk(
                                    full_scores[candidate], k=top_count
                                ).indices
                                selected = candidate[selected_local]
                                containment = float(
                                    torch.isin(full_selected, candidate)
                                    .float()
                                    .mean()
                                )
                                refined_recall = float(
                                    torch.isin(full_selected, selected).float().mean()
                                )
                                candidate_ratio = candidate_count / token_count
                                access_bits = base_bits + candidate_ratio * (
                                    full_bits - base_bits
                                )
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
                                        "method": "query_adaptive_band_cascade",
                                        "full_allocation": "-".join(
                                            map(str, full_allocation)
                                        ),
                                        "selected_bands": "-".join(map(str, chosen)),
                                        "band_energies": "-".join(
                                            f"{value:.8g}" for value in energies
                                        ),
                                        "energy_coverage_target": coverage,
                                        "realized_energy_coverage": realized_coverage,
                                        "selected_band_count": len(chosen),
                                        "selected_band_rate_units": base_rate_units,
                                        "selected_band_bits_per_token": base_bits,
                                        "selected_fraction": selected_fraction,
                                        "overfetch_factor": overfetch,
                                        "candidate_ratio": candidate_ratio,
                                        "full_proxy_topk_containment": containment,
                                        "refined_vs_full_proxy_topk_recall": refined_recall,
                                        "mass_retention_vs_full_proxy": float(
                                            statistics["selected_attention_mass"]
                                        )
                                        / full_mass,
                                        "access_bits_per_token": access_bits,
                                        "access_ratio_of_full_index": access_bits
                                        / full_bits,
                                        "access_ratio_of_full_kv": access_bits
                                        / FULL_KV_BITS,
                                        **statistics,
                                    }
                                )
        print(
            json.dumps({"label": args.label, "layer": layer, "rows": len(rows)}),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query.csv", rows)
    write_csv(args.output_dir / "heads.csv", head_rows)
    summary = aggregate(rows)
    near_lossless = [
        row
        for row in summary
        if row["full_proxy_topk_containment_mean"] >= 0.995
        and row["mass_retention_vs_full_proxy_mean"] >= 0.999
    ]
    best = (
        min(
            near_lossless,
            key=lambda row: (
                row["access_ratio_of_full_index_mean"],
                row["candidate_ratio_mean"],
            ),
        )
        if near_lossless
        else None
    )
    report = {
        "schema": "qksieve_query_adaptive_band_cascade_v1",
        "trace_path": str(args.trace_path),
        "layers": sorted(by_layer),
        "max_heldout_steps": args.max_heldout_steps,
        "quality_boundary": (
            "Band choice uses only current Query and per-band Key second moments. "
            "Candidate and final selection use exact top-k in this mechanism audit; "
            "latency requires a fused band-major CUDA implementation."
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
