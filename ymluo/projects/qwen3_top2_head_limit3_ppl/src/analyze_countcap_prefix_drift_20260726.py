from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def top_eigenspace(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    order = torch.arange(
        eigenvalues.numel() - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    return (
        eigenvalues.index_select(0, order),
        eigenvectors.index_select(1, order)[:, :rank].contiguous(),
    )


def score_operator_inner(
    query_gram: torch.Tensor,
    key_gram: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Frobenius inner product of Q left K^T and Q right K^T."""

    return torch.trace(
        left.transpose(0, 1) @ query_gram @ right @ key_gram
    )


def symmetric_spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(matrix).abs().amax()


def prefix_drift_metrics(
    key: torch.Tensor,
    queries: torch.Tensor,
    rank: int,
    prefix_tokens: int,
    sample_stride: int,
) -> dict[str, float]:
    key = key.float()
    queries = queries.float()
    history_tokens, head_dim = key.shape
    if queries.shape[-1] != head_dim:
        raise ValueError("K and Q dimensions differ")
    if not 0 < rank < head_dim:
        raise ValueError("rank must be in [1, head_dim)")
    if prefix_tokens <= 0 or sample_stride <= 0:
        raise ValueError("prefix_tokens and sample_stride must be positive")

    identity = torch.eye(head_dim, dtype=key.dtype, device=key.device)
    centered_key = key - key.mean(dim=0, keepdim=True)
    centered_key_gram = centered_key.transpose(0, 1) @ centered_key
    query_gram = queries.transpose(0, 1) @ queries
    full_energy = torch.trace(query_gram @ centered_key_gram).clamp_min(1.0e-30)

    full_second_moment = key.transpose(0, 1) @ key / float(history_tokens)
    full_eigenvalues, full_basis = top_eigenspace(full_second_moment, rank)
    full_projector = full_basis @ full_basis.transpose(0, 1)

    prefix_count = min(history_tokens, prefix_tokens)
    sampled_prefix = key[:prefix_count:sample_stride]
    prefix_second_moment = (
        sampled_prefix.transpose(0, 1) @ sampled_prefix
        / float(sampled_prefix.shape[0])
    )
    _, prefix_basis = top_eigenspace(prefix_second_moment, rank)
    prefix_projector = prefix_basis @ prefix_basis.transpose(0, 1)

    intrinsic_operator = identity - full_projector
    drift_operator = full_projector - prefix_projector
    prefix_operator = identity - prefix_projector
    intrinsic_energy = score_operator_inner(
        query_gram,
        centered_key_gram,
        intrinsic_operator,
        intrinsic_operator,
    ).clamp_min(0.0)
    drift_energy = score_operator_inner(
        query_gram,
        centered_key_gram,
        drift_operator,
        drift_operator,
    ).clamp_min(0.0)
    cross_inner = score_operator_inner(
        query_gram,
        centered_key_gram,
        intrinsic_operator,
        drift_operator,
    )
    prefix_residual_energy = score_operator_inner(
        query_gram,
        centered_key_gram,
        prefix_operator,
        prefix_operator,
    ).clamp_min(0.0)
    decomposition_energy = intrinsic_energy + drift_energy + 2.0 * cross_inner

    principal_cosines = torch.linalg.svdvals(
        full_basis.transpose(0, 1) @ prefix_basis
    ).clamp(0.0, 1.0)
    projector_distance = torch.sqrt(
        (1.0 - principal_cosines.square()).clamp_min(0.0)
    ).amax()
    subspace_overlap = principal_cosines.square().mean()

    covariance_delta = symmetric_spectral_norm(
        prefix_second_moment - full_second_moment
    )
    full_key_energy = torch.trace(full_second_moment).clamp_min(1.0e-30)
    full_key_residual = torch.trace(
        intrinsic_operator @ full_second_moment
    ).clamp_min(0.0)
    prefix_key_residual = torch.trace(
        prefix_operator @ full_second_moment
    ).clamp_min(0.0)
    key_excess_risk = (
        prefix_key_residual - full_key_residual
    ).clamp_min(0.0)
    gap_free_key_excess_bound = 2.0 * rank * covariance_delta

    centered_key_energy = torch.trace(centered_key_gram).clamp_min(1.0e-30)
    prefix_centered_key_residual = torch.trace(
        prefix_operator @ centered_key_gram
    ).clamp_min(0.0)
    query_blind_score_bound = (
        torch.linalg.eigvalsh(query_gram).clamp_min(0.0).amax()
        * prefix_centered_key_residual
    )
    eigengap = (
        full_eigenvalues[rank - 1] - full_eigenvalues[rank]
    ).clamp_min(0.0)
    dk_rhs_uncapped = (
        2.0 * covariance_delta / eigengap.clamp_min(1.0e-30)
    )
    dk_rhs = dk_rhs_uncapped.clamp_max(1.0)

    query_spectral_norm = torch.sqrt(
        torch.linalg.eigvalsh(query_gram).clamp_min(0.0).amax()
    )
    score_norm = torch.sqrt(full_energy)
    condition_factor = (
        query_spectral_norm
        * centered_key.norm()
        / score_norm.clamp_min(1.0e-30)
    )
    covariance_bound = condition_factor * dk_rhs

    intrinsic_root = torch.sqrt(intrinsic_energy / full_energy)
    direct_drift_root = torch.sqrt(drift_energy / full_energy)
    prefix_root = torch.sqrt(prefix_residual_energy / full_energy)
    triangle_rhs = intrinsic_root + direct_drift_root
    covariance_triangle_rhs = intrinsic_root + covariance_bound
    cross_correlation = cross_inner / torch.sqrt(
        (intrinsic_energy * drift_energy).clamp_min(1.0e-30)
    )
    decomposition_relative_error = (
        (prefix_residual_energy - decomposition_energy).abs()
        / prefix_residual_energy.clamp_min(1.0e-30)
    )

    return {
        "history_tokens": float(history_tokens),
        "query_count": float(queries.shape[0]),
        "rank": float(rank),
        "prefix_tokens": float(prefix_count),
        "prefix_samples": float(sampled_prefix.shape[0]),
        "full_covariance_norm": float(
            symmetric_spectral_norm(full_second_moment).item()
        ),
        "covariance_drift": float(covariance_delta.item()),
        "relative_covariance_drift": float(
            (
                covariance_delta
                / symmetric_spectral_norm(full_second_moment).clamp_min(1.0e-30)
            ).item()
        ),
        "full_key_reconstruction_fidelity": float(
            (1.0 - full_key_residual / full_key_energy).item()
        ),
        "prefix_key_reconstruction_fidelity": float(
            (1.0 - prefix_key_residual / full_key_energy).item()
        ),
        "key_excess_risk_ratio": float(
            (key_excess_risk / full_key_energy).item()
        ),
        "gap_free_key_excess_bound_ratio": float(
            (gap_free_key_excess_bound / full_key_energy).item()
        ),
        "gap_free_key_excess_bound_satisfied": float(
            key_excess_risk <= gap_free_key_excess_bound + 1.0e-5
        ),
        "prefix_centered_key_reconstruction_fidelity": float(
            (
                1.0
                - prefix_centered_key_residual / centered_key_energy
            ).item()
        ),
        "query_blind_score_bound_ratio": float(
            (query_blind_score_bound / full_energy).item()
        ),
        "rank_boundary_eigengap": float(eigengap.item()),
        "relative_rank_boundary_eigengap": float(
            (
                eigengap
                / full_eigenvalues[0].abs().clamp_min(1.0e-30)
            ).item()
        ),
        "davis_kahan_rhs_uncapped": float(dk_rhs_uncapped.item()),
        "davis_kahan_rhs": float(dk_rhs.item()),
        "davis_kahan_informative": float(dk_rhs_uncapped < 1.0),
        "projector_spectral_distance": float(projector_distance.item()),
        "subspace_overlap": float(subspace_overlap.item()),
        "condition_factor": float(condition_factor.item()),
        "intrinsic_qk_root_relative_error": float(intrinsic_root.item()),
        "direct_qk_drift_root_relative_error": float(
            direct_drift_root.item()
        ),
        "prefix_qk_root_relative_error": float(prefix_root.item()),
        "prefix_qk_fidelity": float(
            (1.0 - prefix_root.square()).item()
        ),
        "intrinsic_qk_fidelity": float(
            (1.0 - intrinsic_root.square()).item()
        ),
        "drift_intrinsic_cross_correlation": float(
            cross_correlation.clamp(-1.0, 1.0).item()
        ),
        "direct_triangle_rhs": float(triangle_rhs.item()),
        "direct_triangle_slack": float(
            (triangle_rhs - prefix_root).clamp_min(0.0).item()
        ),
        "covariance_triangle_rhs": float(covariance_triangle_rhs.item()),
        "covariance_bound_informative": float(covariance_triangle_rhs < 1.0),
        "decomposition_relative_error": float(
            decomposition_relative_error.item()
        ),
    }


def quantiles(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def aggregate(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    excluded = {
        "model",
        "topic",
        "trace_path",
        "layer",
        "kv_head",
    }
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        record = {field: value for field, value in zip(group_fields, key)}
        record["cases"] = len(items)
        numeric_fields = [
            field
            for field, value in items[0].items()
            if field not in excluded
            and field not in group_fields
            and isinstance(value, (int, float))
        ]
        for field in numeric_fields:
            stats = quantiles(float(item[field]) for item in items)
            for statistic, value in stats.items():
                record[f"{field}_{statistic}"] = value
        output.append(record)
    return output


def parse_trace(specification: str) -> tuple[str, str, Path]:
    parts = specification.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--trace must be MODEL=TOPIC=PATH")
    return parts[0], parts[1], Path(parts[2])


@torch.inference_mode()
def analyze_trace(
    model: str,
    topic: str,
    path: Path,
    rank: int,
    prefix_lengths: tuple[int, ...],
    sample_stride: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"{path} has no records")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows: list[dict[str, Any]] = []
    for layer, layer_records in sorted(by_layer.items()):
        state_record = next(
            (
                record
                for record in layer_records
                if isinstance(record.get("key"), torch.Tensor)
            ),
            None,
        )
        if state_record is None:
            raise ValueError(f"layer {layer} has no captured Key state")
        all_key = state_record["key"][0].to(device)
        history_tokens = int(all_key.shape[-2]) - 1
        history_key = all_key[:, :history_tokens]
        query_head_count = int(layer_records[0]["query"].shape[1])
        kv_head_count = int(history_key.shape[0])
        groups = query_head_count // kv_head_count

        for kv_head in range(kv_head_count):
            query_parts = []
            start = kv_head * groups
            stop = start + groups
            for record in layer_records:
                query_parts.append(record["query"][0, start:stop, 0])
            queries = torch.cat(query_parts, dim=0).to(device)
            for prefix_tokens in prefix_lengths:
                metrics = prefix_drift_metrics(
                    history_key[kv_head],
                    queries,
                    rank,
                    prefix_tokens,
                    sample_stride,
                )
                rows.append(
                    {
                        "model": model,
                        "topic": topic,
                        "trace_path": str(path),
                        "layer": layer,
                        "kv_head": kv_head,
                        **metrics,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure covariance, projector, and query-weighted score drift "
            "between CountCap's prefix PCA basis and a full-history basis."
        )
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="MODEL=TOPIC=/path/to/trace.pt; may be supplied more than once",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument(
        "--prefix_lengths",
        default="512,1024,2048,4096,8192",
    )
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prefix_lengths = tuple(
        sorted(
            {
                int(value)
                for value in args.prefix_lengths.split(",")
                if value.strip()
            }
        )
    )
    if not prefix_lengths or min(prefix_lengths) <= 0:
        raise ValueError("prefix_lengths must contain positive integers")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for specification in args.trace:
        model, topic, path = parse_trace(specification)
        rows.extend(
            analyze_trace(
                model,
                topic,
                path,
                args.rank,
                prefix_lengths,
                args.sample_stride,
                device,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prefix_drift_rows.csv", rows)
    report = {
        "config": {
            "traces": args.trace,
            "rank": args.rank,
            "prefix_lengths": prefix_lengths,
            "sample_stride": args.sample_stride,
            "device": str(device),
            "score_space": "row-centered QK",
            "production_basis": "uncentered sampled prefix Key PCA",
            "reference_basis": "uncentered full-history Key PCA",
        },
        "overall_by_prefix": aggregate(rows, ("prefix_tokens",)),
        "by_model_prefix": aggregate(rows, ("model", "prefix_tokens")),
        "by_model_topic_prefix": aggregate(
            rows,
            ("model", "topic", "prefix_tokens"),
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "models": sorted({row["model"] for row in rows}),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
