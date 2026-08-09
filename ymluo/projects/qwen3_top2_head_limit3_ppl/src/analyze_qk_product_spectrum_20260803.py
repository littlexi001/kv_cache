#!/usr/bin/env python
"""Measure the singular spectrum of Cq_tilde^(1/2) Ck^(1/2)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkages", default="0,0.75")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def second_moment(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return values.transpose(0, 1) @ values / max(1, values.shape[0])


def psd_square_root(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix.float())
    return (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())
        @ eigenvectors.transpose(0, 1)
    )


def energy_rank(energy: torch.Tensor, threshold: float) -> int:
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1.0e-30)
    return int(torch.searchsorted(cumulative, threshold).item()) + 1


def spectrum_metric_row(singular_values: torch.Tensor) -> dict[str, float | int]:
    energy = singular_values.square()
    probability = energy / energy.sum().clamp_min(1.0e-30)
    normalized = singular_values / singular_values[0].clamp_min(1.0e-30)
    return {
        "stable_rank": float(
            energy.sum() / energy[0].clamp_min(1.0e-30)
        ),
        "entropy_effective_rank": float(
            torch.exp(
                -(
                    probability
                    * probability.clamp_min(1.0e-30).log()
                ).sum()
            )
        ),
        "rank90": energy_rank(energy, 0.90),
        "rank95": energy_rank(energy, 0.95),
        "rank99": energy_rank(energy, 0.99),
        "rank999": energy_rank(energy, 0.999),
        "top16_energy": float(energy[:16].sum() / energy.sum()),
        "top32_energy": float(energy[:32].sum() / energy.sum()),
        "top48_energy": float(energy[:48].sum() / energy.sum()),
        "top64_energy": float(energy[:64].sum() / energy.sum()),
        "sigma16_over_sigma1": float(normalized[15]),
        "sigma32_over_sigma1": float(normalized[31]),
        "sigma48_over_sigma1": float(normalized[47]),
        "sigma64_over_sigma1": float(normalized[63]),
        "sigma128_over_sigma1": float(normalized[-1]),
    }


SPECTRUM_METRICS = (
    "stable_rank",
    "entropy_effective_rank",
    "rank90",
    "rank95",
    "rank99",
    "rank999",
    "top16_energy",
    "top32_energy",
    "top48_energy",
    "top64_energy",
    "sigma16_over_sigma1",
    "sigma32_over_sigma1",
    "sigma48_over_sigma1",
    "sigma64_over_sigma1",
    "sigma128_over_sigma1",
)


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


@torch.inference_mode()
def analyze_cache_qk_product_spectrum(
    cache: Any,
    prefill_queries: dict[int, torch.Tensor],
    sample_stride: int = 32,
    query_shrinkages: tuple[float, ...] = (0.75,),
) -> dict[str, Any]:
    """Analyze the exact request-local cache used by a quality run."""
    key_cache = getattr(cache, "key_cache", None)
    if key_cache is None:
        raise TypeError("cache does not expose key_cache")
    rows: list[dict[str, Any]] = []
    for layer, query in sorted(prefill_queries.items()):
        if layer >= len(key_cache):
            raise ValueError(f"missing cache layer {layer}")
        key = key_cache[layer]
        if not isinstance(key, torch.Tensor) or key.numel() == 0:
            raise ValueError(f"cache layer {layer} has no Key tensor")
        get_seq_length = getattr(cache, "get_seq_length", None)
        active_count = (
            int(get_seq_length(layer))
            if callable(get_seq_length)
            else int(key.shape[-2])
        )
        key = key[..., :active_count, :]
        query = query.to(device=key.device, dtype=torch.float32)
        batch_count, query_heads, query_tokens, head_dim = query.shape
        _, kv_heads, _, key_head_dim = key.shape
        if batch_count != 1 or key_head_dim != head_dim:
            raise ValueError("unsupported cache/query shape")
        if query_heads % kv_heads != 0:
            raise ValueError("query heads are not divisible by KV heads")
        groups = query_heads // kv_heads
        grouped_query = (
            query.reshape(
                batch_count,
                kv_heads,
                groups,
                query_tokens,
                head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_count, kv_heads, query_tokens * groups, head_dim)
        )
        for kv_head in range(kv_heads):
            sampled_key = key[0, kv_head, ::sample_stride].float()
            head_query = grouped_query[0, kv_head]
            key_moment = second_moment(sampled_key)
            query_moment = second_moment(head_query)
            isotropic_scale = query_moment.diagonal().mean()
            key_root = psd_square_root(key_moment)
            identity = torch.eye(head_dim, device=key.device)
            for shrinkage in query_shrinkages:
                regularized_query = (
                    (1.0 - shrinkage) * query_moment
                    + shrinkage * isotropic_scale * identity
                )
                matrix = psd_square_root(regularized_query) @ key_root
                singular_values = torch.linalg.svdvals(matrix).sort(
                    descending=True
                ).values
                rows.append(
                    {
                        "layer": int(layer),
                        "kv_head": int(kv_head),
                        "query_shrinkage": float(shrinkage),
                        "history_tokens": active_count,
                        "key_samples": int(sampled_key.shape[0]),
                        "query_samples": int(head_query.shape[0]),
                        **spectrum_metric_row(singular_values),
                    }
                )
    by_shrinkage = {
        str(shrinkage): {
            metric: quantiles(
                [
                    float(row[metric])
                    for row in rows
                    if float(row["query_shrinkage"]) == shrinkage
                ]
            )
            for metric in SPECTRUM_METRICS
        }
        for shrinkage in query_shrinkages
    }
    return {
        "schema": "request_local_qk_product_spectrum_v1",
        "sample_stride": sample_stride,
        "layer_head_count": len(rows) // max(1, len(query_shrinkages)),
        "by_query_shrinkage": by_shrinkage,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectra(
    output_path: Path,
    normalized_spectra: dict[float, list[list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.0),
        constrained_layout=True,
    )
    ranks = torch.arange(1, 129)
    for shrinkage, spectra in sorted(normalized_spectra.items()):
        values = torch.tensor(spectra, dtype=torch.float64)
        median = torch.quantile(values, 0.50, dim=0)
        low = torch.quantile(values, 0.10, dim=0)
        high = torch.quantile(values, 0.90, dim=0)
        label = f"query shrinkage={shrinkage:g}"
        axes[0].plot(ranks, median, label=label)
        axes[0].fill_between(ranks, low, high, alpha=0.16)

        energy = values.square()
        cumulative = torch.cumsum(energy, dim=1) / energy.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-30)
        cumulative_median = torch.quantile(cumulative, 0.50, dim=0)
        cumulative_low = torch.quantile(cumulative, 0.10, dim=0)
        cumulative_high = torch.quantile(cumulative, 0.90, dim=0)
        axes[1].plot(ranks, cumulative_median, label=label)
        axes[1].fill_between(
            ranks, cumulative_low, cumulative_high, alpha=0.16
        )

    axes[0].set_title("Normalized singular-value spectrum")
    axes[0].set_xlabel("Singular-value rank")
    axes[0].set_ylabel("sigma_i / sigma_1")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_title("Cumulative squared singular-value energy")
    axes[1].set_xlabel("Retained rank")
    axes[1].set_ylabel("Cumulative energy fraction")
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.99, color="black", linestyle=":", linewidth=0.8)
    axes[1].axvline(48, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(0.80, 1.005)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.suptitle(
        "M = Cq_tilde^(1/2) Ck^(1/2); median and 10-90% over layer-heads"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.calibration_steps < 1:
        raise ValueError("sample_stride and calibration_steps must be positive")
    shrinkages = sorted(
        {float(value) for value in args.query_shrinkages.split(",")}
    )
    if any(value < 0.0 or value > 1.0 for value in shrinkages):
        raise ValueError("query shrinkage must lie in [0, 1]")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("trace contains no records")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    head_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    normalized_spectra: dict[float, list[list[float]]] = defaultdict(list)

    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda row: int(row["step"]))
        calibration_records = layer_records[: args.calibration_steps]
        if len(calibration_records) < args.calibration_steps:
            raise ValueError(f"layer {layer} lacks calibration queries")
        raw_key = next(
            (
                record.get("key")
                for record in layer_records
                if record.get("key") is not None
            ),
            None,
        )
        if raw_key is None:
            raise ValueError(f"layer {layer} lacks a key tensor")
        key = raw_key.to(device=device, dtype=torch.float32)[0]
        query = torch.stack(
            [
                record["query"].to(device=device, dtype=torch.float32)[0, :, 0]
                for record in calibration_records
            ],
            dim=0,
        )
        kv_heads = int(key.shape[0])
        query_heads = int(query.shape[1])
        groups = query_heads // kv_heads
        if groups * kv_heads != query_heads:
            raise ValueError("query heads are not divisible by KV heads")

        for kv_head in range(kv_heads):
            sampled_key = key[kv_head, :: args.sample_stride]
            head_query = query[
                :, kv_head * groups : (kv_head + 1) * groups
            ].reshape(-1, key.shape[-1])
            key_moment = second_moment(sampled_key)
            query_moment = second_moment(head_query)
            isotropic_scale = query_moment.diagonal().mean()
            key_root = psd_square_root(key_moment)

            for shrinkage in shrinkages:
                regularized_query = (
                    (1.0 - shrinkage) * query_moment
                    + shrinkage
                    * isotropic_scale
                    * torch.eye(query_moment.shape[0], device=device)
                )
                matrix = psd_square_root(regularized_query) @ key_root
                singular_values = torch.linalg.svdvals(matrix).sort(
                    descending=True
                ).values
                normalized = singular_values / singular_values[0].clamp_min(1.0e-30)
                normalized_spectra[shrinkage].append(normalized.cpu().tolist())
                head_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "query_shrinkage": shrinkage,
                        "key_samples": int(sampled_key.shape[0]),
                        "query_samples": int(head_query.shape[0]),
                        **spectrum_metric_row(singular_values),
                    }
                )
                for rank, (sigma, ratio) in enumerate(
                    zip(singular_values.tolist(), normalized.tolist()), start=1
                ):
                    spectrum_rows.append(
                        {
                            "label": args.label,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_shrinkage": shrinkage,
                            "rank": rank,
                            "singular_value": sigma,
                            "sigma_over_sigma1": ratio,
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", head_rows)
    write_csv(args.output_dir / "singular_values.csv", spectrum_rows)
    plot_spectra(args.output_dir / "spectrum_curves.png", normalized_spectra)

    summary = {
        "schema": "qk_product_spectrum_v1",
        "label": args.label,
        "trace_path": str(args.trace_path),
        "matrix_definition": "Cq_tilde^(1/2) @ Ck^(1/2)",
        "second_moment_centered": False,
        "sample_stride": args.sample_stride,
        "calibration_steps": args.calibration_steps,
        "layer_count": len(by_layer),
        "layer_head_count": len(head_rows) // len(shrinkages),
        "by_query_shrinkage": {
            str(shrinkage): {
                metric: quantiles(
                    [
                        float(row[metric])
                        for row in head_rows
                        if float(row["query_shrinkage"]) == shrinkage
                    ]
                )
                for metric in SPECTRUM_METRICS
            }
            for shrinkage in shrinkages
        },
        "claim_boundary": (
            "One 4K sports request. Results establish the within-request "
            "layer-head spectrum, not cross-topic or cross-length universality."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
