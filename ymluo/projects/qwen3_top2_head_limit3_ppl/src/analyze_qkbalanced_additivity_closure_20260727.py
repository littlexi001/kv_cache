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
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
)
from analyze_qk_balanced_spectral_rate_20260727 import (
    covariance,
    distortion_table_from_bands,
    qk_balanced_factors,
)
from analyze_qk_progressive_refinement_20260727 import (
    quantized_bands,
    reconstruct,
)


def block_offdiagonal_ratio(
    matrix: torch.Tensor,
    block_size: int = GROUP_SIZE,
) -> float:
    mask = torch.ones_like(matrix, dtype=torch.bool)
    for start in range(0, matrix.shape[0], block_size):
        mask[start : start + block_size, start : start + block_size] = False
    denominator = torch.linalg.matrix_norm(matrix.float()).clamp_min(1.0e-20)
    return float(
        (
            torch.linalg.vector_norm(matrix.float()[mask])
            / denominator
        ).item()
    )


def band_error_decomposition(
    residual: torch.Tensor,
    query: torch.Tensor,
) -> tuple[float, float, float]:
    band_errors = torch.stack(
        [
            residual[
                :,
                index * GROUP_SIZE : (index + 1) * GROUP_SIZE,
            ]
            @ query[
                index * GROUP_SIZE : (index + 1) * GROUP_SIZE
            ]
            for index in range(GROUP_COUNT)
        ],
        dim=-1,
    )
    actual = float(band_errors.sum(dim=-1).square().mean().item())
    diagonal = float(band_errors.square().sum(dim=-1).mean().item())
    return actual, diagonal, actual - diagonal


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


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = torch.tensor(list(left), dtype=torch.float64)
    y = torch.tensor(list(right), dtype=torch.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = (
        torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    )
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((x @ y / denominator).item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate simultaneous covariance diagonalization and the "
            "additive band-qMSE objective of the frozen QK-balanced index."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--total_rate_budget", type=int, default=15)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
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

    head_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        layer_records.sort(key=lambda row: int(row["step"]))
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
            raise ValueError(f"layer {layer} has no key")
        key = state_record["key"].to(device).float()[0, :, :-1]
        calibration = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[: args.calibration_steps]
            ],
            dim=0,
        )
        heldout = torch.stack(
            [
                record["query"].to(device).float()[0, :, 0, :]
                for record in layer_records[args.calibration_steps :]
            ],
            dim=0,
        )
        kv_heads = int(key.shape[0])
        query_heads = int(calibration.shape[1])
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_heads // kv_heads

        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            sampled_key = head_key[:: args.sample_stride]
            head_calibration = calibration[
                :,
                kv_head * groups : (kv_head + 1) * groups,
            ].reshape(-1, head_key.shape[-1])
            head_heldout = heldout[
                :,
                kv_head * groups : (kv_head + 1) * groups,
            ].reshape(-1, head_key.shape[-1])
            query_factor, key_factor, singular_values = (
                qk_balanced_factors(
                    sampled_key,
                    head_calibration,
                    args.query_shrinkage,
                )
            )
            coefficients = head_key @ key_factor
            projected_calibration = head_calibration @ query_factor
            projected_heldout = head_heldout @ query_factor

            query_covariance = covariance(head_calibration)
            isotropic = query_covariance.diagonal().mean()
            regularized_query_covariance = (
                (1.0 - args.query_shrinkage) * query_covariance
                + args.query_shrinkage
                * isotropic
                * torch.eye(
                    query_covariance.shape[0],
                    device=device,
                )
            )
            transformed_query_covariance = (
                query_factor.transpose(0, 1)
                @ regularized_query_covariance
                @ query_factor
            )
            transformed_key_covariance = (
                key_factor.transpose(0, 1)
                @ covariance(sampled_key)
                @ key_factor
            )
            target_covariance = torch.diag(singular_values)
            heldout_query_covariance = covariance(projected_heldout)

            bands = quantized_bands(
                coefficients,
                projected_calibration,
            )
            distortion = distortion_table_from_bands(
                coefficients,
                projected_calibration,
                bands,
            )
            allocation = allocate_bits(
                distortion,
                args.total_rate_budget,
                ZERO_BIT_LEVELS,
                include_scale_metadata=True,
            )
            reconstruction = reconstruct(bands, allocation)
            residual = coefficients - reconstruction
            calibration_objective = sum(
                float(distortion[index][bits].item())
                for index, bits in enumerate(allocation)
            )
            head_rows.append(
                {
                    "label": args.label,
                    "layer": layer,
                    "kv_head": kv_head,
                    "allocation": "-".join(map(str, allocation)),
                    "regularized_query_offblock_ratio": (
                        block_offdiagonal_ratio(
                            transformed_query_covariance
                        )
                    ),
                    "sampled_key_offblock_ratio": (
                        block_offdiagonal_ratio(
                            transformed_key_covariance
                        )
                    ),
                    "heldout_query_offblock_ratio": (
                        block_offdiagonal_ratio(
                            heldout_query_covariance
                        )
                    ),
                    "regularized_query_target_relative_error": float(
                        (
                            torch.linalg.matrix_norm(
                                transformed_query_covariance
                                - target_covariance
                            )
                            / torch.linalg.matrix_norm(
                                target_covariance
                            ).clamp_min(1.0e-20)
                        ).item()
                    ),
                    "sampled_key_target_relative_error": float(
                        (
                            torch.linalg.matrix_norm(
                                transformed_key_covariance
                                - target_covariance
                            )
                            / torch.linalg.matrix_norm(
                                target_covariance
                            ).clamp_min(1.0e-20)
                        ).item()
                    ),
                    "calibration_additive_qmse": calibration_objective,
                }
            )

            for heldout_index, projected_query in enumerate(
                projected_heldout
            ):
                actual, diagonal, cross = band_error_decomposition(
                    residual,
                    projected_query,
                )
                query_rows.append(
                    {
                        "label": args.label,
                        "layer": layer,
                        "kv_head": kv_head,
                        "heldout_query": heldout_index,
                        "calibration_additive_qmse": (
                            calibration_objective
                        ),
                        "heldout_actual_qmse": actual,
                        "heldout_additive_qmse": diagonal,
                        "heldout_cross_qmse": cross,
                        "absolute_cross_over_additive": (
                            abs(cross) / max(diagonal, 1.0e-20)
                        ),
                        "actual_over_additive": (
                            actual / max(diagonal, 1.0e-20)
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "layer": layer,
                    "head_rows": len(head_rows),
                    "query_rows": len(query_rows),
                }
            ),
            flush=True,
        )

    summary = {
        "label": args.label,
        "heads": len(head_rows),
        "heldout_queries": len(query_rows),
        "head_metrics": {
            field: summarize(float(row[field]) for row in head_rows)
            for field in (
                "regularized_query_offblock_ratio",
                "sampled_key_offblock_ratio",
                "heldout_query_offblock_ratio",
                "regularized_query_target_relative_error",
                "sampled_key_target_relative_error",
            )
        },
        "query_metrics": {
            field: summarize(float(row[field]) for row in query_rows)
            for field in (
                "heldout_actual_qmse",
                "heldout_additive_qmse",
                "heldout_cross_qmse",
                "absolute_cross_over_additive",
                "actual_over_additive",
            )
        },
        "log_calibration_vs_log_heldout_actual_pearson": pearson(
            (
                math.log(
                    max(1.0e-20, float(row["calibration_additive_qmse"]))
                )
                for row in query_rows
            ),
            (
                math.log(
                    max(1.0e-20, float(row["heldout_actual_qmse"]))
                )
                for row in query_rows
            ),
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", head_rows)
    write_csv(args.output_dir / "per_query.csv", query_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
