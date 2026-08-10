#!/usr/bin/env python
"""Evaluate only the registered QKSieve shrinkage grid.

This analyzer is numerically equivalent to the ``qk_balanced`` branch of
``analyze_qk_balanced_spectral_rate_20260727.py``.  It deliberately omits the
dozens of unrelated method variants in that exploratory script and reuses
exact scores across shrinkage values.  Inputs, qMSE allocation, quantizers,
held-out conditions, metric formulas, and output row keys remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

import analyze_qk_balanced_spectral_rate_20260727 as reference
from analyze_automatic_spectral_rate_allocation_20260727 import (
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    quantize_band,
)
from analyze_hierarchical_spectral_quantization_20260727 import query_int8


METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "oracle_top2_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item) for item in text.split(",") if item}))
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in text.split(",") if item}))


def lambda_tag(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_reconstruction(
    coefficients: torch.Tensor,
    calibration_queries: torch.Tensor,
    total_rate_budget: int,
) -> tuple[torch.Tensor, list[int]]:
    """Match the exploratory analyzer's qMSE allocation and reconstruction."""

    allocation = allocate_bits(
        reference.distortion_table(coefficients, calibration_queries),
        total_rate_budget,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    reconstructed = torch.cat(
        [
            quantize_band(
                coefficients[
                    :, group_index * GROUP_SIZE : (group_index + 1) * GROUP_SIZE
                ],
                bits,
            )
            for group_index, bits in enumerate(allocation)
        ],
        dim=-1,
    )
    return reconstructed, allocation


def batched_selection_metrics(
    exact_scores: torch.Tensor,
    approximate_scores: torch.Tensor,
    true_top_indices: torch.Tensor,
    fractions: tuple[float, ...],
) -> dict[float, dict[str, torch.Tensor]]:
    """Vectorized form of ``selection_metrics`` for one query batch."""

    if exact_scores.shape != approximate_scores.shape:
        raise ValueError("exact and approximate score shapes differ")
    if exact_scores.ndim != 2 or true_top_indices.ndim != 2:
        raise ValueError("batched metrics require rank-two tensors")
    if not torch.isfinite(exact_scores).all() or not torch.isfinite(
        approximate_scores
    ).all():
        raise AssertionError("non-finite selector scores")

    attention = torch.softmax(exact_scores, dim=-1)
    true_mass = attention.gather(1, true_top_indices).sum(dim=-1)
    centered_exact = exact_scores - exact_scores.mean(dim=-1, keepdim=True)
    centered_approximate = approximate_scores - approximate_scores.mean(
        dim=-1, keepdim=True
    )
    denominator = (
        torch.linalg.vector_norm(centered_exact, dim=-1)
        * torch.linalg.vector_norm(centered_approximate, dim=-1)
    )
    pearson = torch.where(
        denominator > 0.0,
        (centered_exact * centered_approximate).sum(dim=-1) / denominator,
        torch.zeros_like(denominator),
    )
    rmse = (exact_scores - approximate_scores).square().mean(dim=-1).sqrt()

    output: dict[float, dict[str, torch.Tensor]] = {}
    token_count = exact_scores.shape[-1]
    for fraction in fractions:
        selected_count = min(token_count, max(1, math.ceil(fraction * token_count)))
        selected = torch.topk(
            approximate_scores, k=selected_count, dim=-1
        ).indices
        selected_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
        selected_mask.scatter_(1, selected, True)
        hits = selected_mask.gather(1, true_top_indices).sum(dim=-1)
        selected_mass = attention.gather(1, selected).sum(dim=-1)
        output[fraction] = {
            "selected_fraction": torch.full_like(
                pearson, selected_count / token_count
            ),
            "selected_count": torch.full_like(
                hits, selected_count, dtype=torch.long
            ),
            "top2_recall": hits.float() / true_top_indices.shape[-1],
            "selected_attention_mass": selected_mass,
            "oracle_top2_attention_mass": true_mass,
            "top2_attention_mass_recall": selected_mass
            / true_mass.clamp_min(1.0e-30),
            "score_pearson": pearson,
            "score_rmse": rmse,
        }
    return output


def append_metric_rows(
    rows: list[dict[str, Any]],
    metrics: dict[float, dict[str, torch.Tensor]],
    *,
    label: str,
    layer: int,
    evaluation_start: int,
    kv_head: int,
    query_head_start: int,
    groups: int,
) -> None:
    batch_size = next(iter(metrics.values()))["top2_recall"].numel()
    expected_batch = batch_size // groups
    for fraction, fields in metrics.items():
        cpu_fields = {
            name: value.detach().cpu().tolist() for name, value in fields.items()
        }
        for batch_index in range(batch_size):
            step = evaluation_start + batch_index // groups
            query_head = query_head_start + batch_index % groups
            rows.append(
                {
                    "label": label,
                    "layer": layer,
                    "heldout_step": step,
                    "kv_head": kv_head,
                    "query_head": query_head,
                    "method": "qk_balanced",
                    "selected_fraction_target": fraction,
                    **{
                        name: values[batch_index]
                        for name, values in cpu_fields.items()
                    },
                }
            )
    if expected_batch <= 0:
        raise AssertionError("empty held-out query batch")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["selected_fraction_target"])].append(row)
    output = []
    for fraction, items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": "qk_balanced",
            "selected_fraction_target": fraction,
            "conditions": len(items),
        }
        for metric in METRICS:
            values = torch.tensor(
                [float(item[metric]) for item in items], dtype=torch.float64
            )
            result[metric] = float(values.mean())
        output.append(result)
    return output


@torch.inference_mode()
def analyze_grid(
    trace_path: Path,
    output_root: Path,
    *,
    label: str,
    shrinkages: tuple[float, ...],
    fractions: tuple[float, ...],
    sample_stride: int,
    calibration_steps: int,
    calibration_source: str,
    total_rate_budget: int,
    top_fraction: float,
    device: torch.device,
    layers: tuple[int, ...] = (),
) -> dict[str, Any]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        layer = int(record["layer"])
        if not layers or layer in layers:
            by_layer[layer].append(record)
    if layers and set(by_layer) != set(layers):
        raise AssertionError("requested layer is absent from trace")

    rows_by_shrinkage = {value: [] for value in shrinkages}
    allocations_by_shrinkage = {value: [] for value in shrinkages}
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda row: int(row["step"]))
        calibration, evaluation_records, evaluation_start = (
            reference.resolve_calibration_and_evaluation(
                payload,
                layer,
                layer_records,
                calibration_steps,
                calibration_source,
                device,
            )
        )
        raw_key = next(
            (row.get("key") for row in layer_records if row.get("key") is not None),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} has no key tensor")
        key = raw_key.to(device).float()[0]
        history_count = int(key.shape[1]) - 1
        key = key[:, :history_count]
        kv_heads = int(key.shape[0])
        query_heads = int(layer_records[0]["query"].shape[1])
        groups = query_heads // kv_heads
        if groups * kv_heads != query_heads:
            raise AssertionError("query heads are not divisible by KV heads")
        queries = torch.stack(
            [row["query"].to(device).float()[0, :, 0, :] for row in evaluation_records]
        )
        scalings = torch.tensor(
            [float(row["scaling"]) for row in evaluation_records],
            dtype=torch.float32,
            device=device,
        )
        top_count = max(1, math.ceil(top_fraction * history_count))

        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            head_calibration = calibration[
                :, kv_head * groups : (kv_head + 1) * groups
            ].reshape(-1, head_key.shape[-1])
            states: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
            for shrinkage in shrinkages:
                query_factor, key_factor, _ = reference.qk_balanced_factors(
                    head_key[::sample_stride], head_calibration, shrinkage
                )
                coefficients = head_key @ key_factor
                projected_calibration = head_calibration @ query_factor
                reconstructed, allocation = selected_reconstruction(
                    coefficients, projected_calibration, total_rate_budget
                )
                states[shrinkage] = (query_factor, reconstructed)
                allocations_by_shrinkage[shrinkage].append(
                    {
                        "label": label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "method": "qk_balanced",
                        "allocation": "-".join(map(str, allocation)),
                        "code_bits": GROUP_SIZE * sum(allocation),
                        "metadata_bits": GROUP_SIZE
                        * sum(bits > 0 for bits in allocation),
                    }
                )

            query_start = kv_head * groups
            head_queries = queries[:, query_start : query_start + groups].reshape(
                -1, head_key.shape[-1]
            )
            row_scalings = scalings[:, None].expand(-1, groups).reshape(-1)
            exact_scores = (head_queries @ head_key.transpose(0, 1)) * row_scalings[
                :, None
            ]
            true_top = torch.topk(exact_scores, k=top_count, dim=-1).indices

            for shrinkage, (query_factor, reconstructed) in states.items():
                projected_queries = torch.stack(
                    [query_int8(query @ query_factor) for query in head_queries]
                )
                approximate_scores = (
                    projected_queries @ reconstructed.transpose(0, 1)
                ) * row_scalings[:, None]
                metrics = batched_selection_metrics(
                    exact_scores, approximate_scores, true_top, fractions
                )
                append_metric_rows(
                    rows_by_shrinkage[shrinkage],
                    metrics,
                    label=label,
                    layer=layer,
                    evaluation_start=evaluation_start,
                    kv_head=kv_head,
                    query_head_start=query_start,
                    groups=groups,
                )
                del approximate_scores, projected_queries
            del exact_scores, true_top, states

        print(
            json.dumps(
                {
                    "label": label,
                    "layer": layer,
                    "layers": len(by_layer),
                    "rows_per_lambda": {
                        str(value): len(rows_by_shrinkage[value])
                        for value in shrinkages
                    },
                }
            ),
            flush=True,
        )

    trace_digest = sha256(trace_path)
    source_digest = sha256(Path(__file__))
    for shrinkage in shrinkages:
        output = output_root / lambda_tag(shrinkage)
        rows = rows_by_shrinkage[shrinkage]
        allocations = allocations_by_shrinkage[shrinkage]
        write_csv(output / "per_head.csv", rows)
        write_csv(output / "allocations.csv", allocations)
        summary = {
            "schema": "qksieve_shrinkage_grid_cell_v1",
            "config": {
                "trace_path": str(trace_path),
                "label": label,
                "sample_stride": sample_stride,
                "calibration_steps": calibration_steps,
                "calibration_source": calibration_source,
                "total_rate_budget": total_rate_budget,
                "query_shrinkage": shrinkage,
                "selected_fractions": list(fractions),
                "top_fraction": top_fraction,
                "layers": sorted(by_layer),
            },
            "conditions": len(rows),
            "allocation_histogram": dict(
                Counter(row["allocation"] for row in allocations).most_common()
            ),
            "metrics": aggregate_rows(rows),
            "provenance": {
                "trace_sha256": trace_digest,
                "analyzer_sha256": source_digest,
                "equivalence_contract": (
                    "qk_balanced branch; exact scores reused across lambdas"
                ),
            },
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    grid = {
        "schema": "qksieve_shrinkage_grid_v1",
        "label": label,
        "trace_sha256": trace_digest,
        "analyzer_sha256": source_digest,
        "shrinkages": list(shrinkages),
        "fractions": list(fractions),
        "layers": sorted(by_layer),
        "conditions_per_lambda": {
            str(value): len(rows_by_shrinkage[value]) for value in shrinkages
        },
    }
    (output_root / "grid_summary.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "ALL_COMPLETE").touch()
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--query_shrinkages", default="0,0.25,0.5,0.75,0.9")
    parser.add_argument("--selected_fractions", default="0.01,0.02,0.04")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--calibration_source", default="prefill_tail")
    parser.add_argument("--total_rate_budget", type=int, default=15)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shrinkages = parse_float_list(args.query_shrinkages)
    fractions = parse_float_list(args.selected_fractions)
    if any(not 0.0 <= value <= 1.0 for value in shrinkages):
        raise ValueError("shrinkage values must lie in [0, 1]")
    if any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("selected fractions must lie in (0, 1]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = analyze_grid(
        args.trace_path,
        args.output_root,
        label=args.label,
        shrinkages=shrinkages,
        fractions=fractions,
        sample_stride=args.sample_stride,
        calibration_steps=args.calibration_steps,
        calibration_source=args.calibration_source,
        total_rate_budget=args.total_rate_budget,
        top_fraction=args.top_fraction,
        device=device,
        layers=parse_int_list(args.layers),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
