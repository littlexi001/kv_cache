#!/usr/bin/env python
"""Measure value and score distributions across QKSieve's eight 16D bands."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_qksieve_frequency_hotset_20260729 import (
    mapped_queries,
    second_moment,
)
from run_head_top2_targeted_ppl_20260714 import (
    _qk_metric_projection_factors,
)


BAND_COUNT = 8
BAND_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--key_stride", type=int, default=32)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    return parser.parse_args()


def band_value_metrics(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return [heads, bands] statistics for [heads, samples, 128]."""
    heads = int(values.shape[0])
    banded = values.reshape(heads, values.shape[1], BAND_COUNT, BAND_SIZE)
    flattened = banded.permute(0, 2, 1, 3).reshape(
        heads, BAND_COUNT, -1
    )
    absolute = flattened.abs()
    square_mean = flattened.square().mean(dim=-1)
    energy = square_mean * flattened.shape[-1]
    return {
        "mean_abs": absolute.mean(dim=-1),
        "rms": square_mean.sqrt(),
        "median_abs": torch.quantile(absolute, 0.50, dim=-1),
        "p90_abs": torch.quantile(absolute, 0.90, dim=-1),
        "p99_abs": torch.quantile(absolute, 0.99, dim=-1),
        "max_abs": absolute.amax(dim=-1),
        "energy_share": energy / energy.sum(dim=-1, keepdim=True).clamp_min(
            1e-30
        ),
    }


def score_contribution_metrics(
    projected_query: torch.Tensor,
    projected_key: torch.Tensor,
    top_fraction: float,
) -> dict[str, torch.Tensor]:
    """Return [heads, bands] QK contribution statistics."""
    heads, query_count, _ = projected_query.shape
    key_count = int(projected_key.shape[1])
    contributions = []
    for band in range(BAND_COUNT):
        start = band * BAND_SIZE
        stop = start + BAND_SIZE
        contributions.append(
            torch.einsum(
                "hqd,hkd->hqk",
                projected_query[..., start:stop],
                projected_key[..., start:stop],
            )
        )
    stacked = torch.stack(contributions, dim=-1)
    mean_abs = stacked.abs().mean(dim=(1, 2))
    square_mean = stacked.square().mean(dim=(1, 2))
    absolute_share = mean_abs / mean_abs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-30)
    variance_share = square_mean / square_mean.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-30)

    total_score = stacked.sum(dim=-1)
    top_count = min(
        key_count,
        max(1, math.ceil(top_fraction * key_count)),
    )
    selected = torch.topk(
        total_score,
        k=top_count,
        dim=-1,
        sorted=False,
    ).indices
    selected_expanded = selected[..., None].expand(
        heads, query_count, top_count, BAND_COUNT
    )
    selected_contribution = stacked.gather(2, selected_expanded)
    selected_mean_abs = selected_contribution.abs().mean(dim=(1, 2))
    selected_absolute_share = selected_mean_abs / selected_mean_abs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-30)
    return {
        "score_mean_abs": mean_abs,
        "score_rms": square_mean.sqrt(),
        "score_abs_share": absolute_share,
        "score_variance_share": variance_share,
        "top_score_mean_abs": selected_mean_abs,
        "top_score_abs_share": selected_absolute_share,
    }


def add_rows(
    rows: list[dict[str, Any]],
    topic: str,
    layer: int,
    coordinate_system: str,
    key_metrics: dict[str, torch.Tensor],
    query_metrics: dict[str, torch.Tensor],
    score_metrics: dict[str, torch.Tensor],
) -> None:
    heads = int(key_metrics["rms"].shape[0])
    for head in range(heads):
        for band in range(BAND_COUNT):
            rows.append(
                {
                    "topic": topic,
                    "layer": layer,
                    "head": head,
                    "coordinate_system": coordinate_system,
                    "band": band + 1,
                    "dimension_start": band * BAND_SIZE,
                    "dimension_stop": (band + 1) * BAND_SIZE,
                    **{
                        f"key_{name}": float(value[head, band].item())
                        for name, value in key_metrics.items()
                    },
                    **{
                        f"query_{name}": float(value[head, band].item())
                        for name, value in query_metrics.items()
                    },
                    **{
                        name: float(value[head, band].item())
                        for name, value in score_metrics.items()
                    },
                }
            )


def aggregate_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["coordinate_system"], row["band"])].append(row)
    metric_names = [
        key
        for key in rows[0]
        if key
        not in {
            "topic",
            "layer",
            "head",
            "coordinate_system",
            "band",
            "dimension_start",
            "dimension_stop",
        }
    ]
    output = []
    for (coordinate_system, band), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "coordinate_system": coordinate_system,
            "band": band,
            "conditions": len(group),
        }
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in group],
                dtype=np.float64,
            )
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_median"] = float(np.median(values))
            item[f"{metric}_p10"] = float(np.quantile(values, 0.10))
            item[f"{metric}_p90"] = float(np.quantile(values, 0.90))
        output.append(item)
    return output


def trend_summary(
    rows: list[dict[str, Any]],
    coordinate_system: str,
    metric: str,
) -> dict[str, float]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row["coordinate_system"] == coordinate_system:
            grouped[(row["topic"], row["layer"], row["head"])].append(row)
    monotonic = []
    correlations = []
    tail_ratios = []
    first3_shares = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item["band"])
        values = np.asarray(
            [max(1e-30, float(item[metric])) for item in ordered]
        )
        monotonic.append(float(np.all(values[:-1] >= values[1:])))
        correlations.append(
            float(np.corrcoef(np.arange(BAND_COUNT), np.log(values))[0, 1])
        )
        tail_ratios.append(float(values[-1] / values[0]))
        if metric.endswith("_share"):
            first3_shares.append(float(values[:3].sum()))
    result = {
        "strictly_monotonic_fraction": float(np.mean(monotonic)),
        "band_index_log_metric_correlation_mean": float(
            np.mean(correlations)
        ),
        "band8_over_band1_median": float(np.median(tail_ratios)),
        "band8_over_band1_p90": float(np.quantile(tail_ratios, 0.90)),
    }
    if first3_shares:
        result["first3_bands_share_mean"] = float(np.mean(first3_shares))
    return result


def save_plot(
    aggregate: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        by_system[row["coordinate_system"]].append(row)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    panels = (
        ("key_rms_mean", "K RMS / band 1", True),
        ("query_rms_mean", "Held-out Q RMS / band 1", True),
        ("score_abs_share_mean", "Absolute QK contribution share", False),
    )
    for axis, (metric, title, normalize) in zip(axes, panels):
        for coordinate_system, system_rows in sorted(by_system.items()):
            ordered = sorted(system_rows, key=lambda item: item["band"])
            values = np.asarray([float(item[metric]) for item in ordered])
            if normalize:
                values = values / max(1e-30, values[0])
            axis.plot(
                np.arange(1, BAND_COUNT + 1),
                values,
                marker="o",
                label=coordinate_system,
            )
        axis.set_title(title)
        axis.set_xlabel("16D band")
        axis.grid(alpha=0.25)
        if normalize:
            axis.set_yscale("log")
    axes[0].set_ylabel("Value")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    detail_rows: list[dict[str, Any]] = []

    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        payload = torch.load(path_text, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload["records"]:
            by_layer[int(record["layer"])].append(record)

        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            calibration = records[: args.calibration_steps]
            evaluation = records[args.calibration_steps :]
            key = calibration[0]["key"].to(device=device).float()
            _, kv_heads, _, head_dim = key.shape
            if head_dim != BAND_COUNT * BAND_SIZE:
                raise ValueError(f"expected head dimension 128, got {head_dim}")
            sampled_key = key[..., :: args.key_stride, :].contiguous()
            calibration_query = mapped_queries(calibration).to(device)
            evaluation_query = mapped_queries(evaluation).to(device)
            query_basis, key_basis = _qk_metric_projection_factors(
                second_moment(sampled_key),
                second_moment(calibration_query),
                projection_dim=head_dim,
                query_shrinkage=args.query_shrinkage,
            )
            projected_key = torch.einsum(
                "bhnd,bhdm->bhnm",
                key,
                key_basis,
            )[0]
            projected_evaluation_query = torch.einsum(
                "bhnd,bhdm->bhnm",
                evaluation_query,
                query_basis,
            )[0]
            score_key = projected_key[:, :: args.key_stride, :]

            add_rows(
                detail_rows,
                topic,
                layer,
                "qk_balanced",
                band_value_metrics(projected_key),
                band_value_metrics(projected_evaluation_query),
                score_contribution_metrics(
                    projected_evaluation_query,
                    score_key,
                    args.top_fraction,
                ),
            )

            raw_evaluation_query = evaluation_query[0]
            raw_key = key[0]
            add_rows(
                detail_rows,
                topic,
                layer,
                "raw_dimension_chunks",
                band_value_metrics(raw_key),
                band_value_metrics(raw_evaluation_query),
                score_contribution_metrics(
                    raw_evaluation_query,
                    raw_key[:, :: args.key_stride, :],
                    args.top_fraction,
                ),
            )
            del (
                key,
                sampled_key,
                calibration_query,
                evaluation_query,
                query_basis,
                key_basis,
                projected_key,
                projected_evaluation_query,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    detail_path = args.output_dir / "band_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    aggregate = aggregate_rows(detail_rows)
    aggregate_path = args.output_dir / "band_aggregate.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    trends = {}
    for system in ("qk_balanced", "raw_dimension_chunks"):
        trends[system] = {
            metric: trend_summary(detail_rows, system, metric)
            for metric in (
                "key_rms",
                "query_rms",
                "score_rms",
                "score_abs_share",
                "top_score_abs_share",
            )
        }
    summary = {
        "schema": "qksieve_band_distribution_v1",
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
        },
        "conditions_per_coordinate_system": len(detail_rows)
        // (2 * BAND_COUNT),
        "aggregate": aggregate,
        "trends": trends,
        "interpretation": (
            "QK-balanced bands are ordered by a query-conditioned singular "
            "spectrum. Early-band value RMS is expected to be larger, but "
            "score contribution rather than K magnitude alone determines "
            "whether a tail band is removable."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    save_plot(aggregate, args.output_dir / "band_distributions.png")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
