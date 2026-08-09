from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def effective_rank(energy: torch.Tensor) -> float:
    probability = energy.clamp_min(0.0)
    probability = probability / probability.sum().clamp_min(1.0e-30)
    entropy = -(probability * probability.clamp_min(1.0e-30).log()).sum()
    return float(entropy.exp().item())


def projection_score_metrics(
    key_gram: torch.Tensor,
    query_gram: torch.Tensor,
    basis: torch.Tensor,
    head_dim: int,
) -> dict[str, float]:
    projector = basis @ basis.transpose(0, 1)
    identity = torch.eye(
        head_dim,
        dtype=projector.dtype,
        device=projector.device,
    )
    residual = identity - projector
    full_energy = torch.trace(query_gram @ key_gram) / float(head_dim)
    approximate_energy = (
        torch.trace(query_gram @ projector @ key_gram @ projector)
        / float(head_dim)
    )
    error_energy = (
        torch.trace(query_gram @ residual @ key_gram @ residual)
        / float(head_dim)
    )
    inner_product = (
        torch.trace(query_gram @ key_gram @ projector)
        / float(head_dim)
    )
    denominator = torch.sqrt(
        full_energy.clamp_min(1.0e-30)
        * approximate_energy.clamp_min(1.0e-30)
    )
    return {
        "qk_fidelity": float(
            (1.0 - error_energy / full_energy.clamp_min(1.0e-30)).item()
        ),
        "qk_relative_mse": float(
            (error_energy / full_energy.clamp_min(1.0e-30)).item()
        ),
        "qk_score_cosine": float(
            (inner_product / denominator.clamp_min(1.0e-30)).item()
        ),
    }


def spectral_metrics(
    key: torch.Tensor,
    queries: torch.Tensor,
    rank: int,
    basis_prefix_tokens: int = 2048,
    basis_sample_stride: int = 32,
) -> dict[str, float]:
    """Compute K and QK spectra without materializing the long QK matrix."""

    key = key.float()
    queries = queries.float()
    head_dim = int(key.shape[-1])
    if queries.shape[-1] != head_dim:
        raise ValueError("K and Q dimensions differ")
    if not 0 < rank <= head_dim:
        raise ValueError("rank must be in [1, head_dim]")
    if basis_prefix_tokens <= 0 or basis_sample_stride <= 0:
        raise ValueError("basis prefix and stride must be positive")

    key_gram = key.transpose(0, 1) @ key
    key_eigenvalues, key_basis = torch.linalg.eigh(key_gram)
    key_eigenvalues = key_eigenvalues.clamp_min(0.0)
    order = torch.arange(
        head_dim - 1,
        -1,
        -1,
        device=key.device,
    )
    key_energy = key_eigenvalues.index_select(0, order)
    key_basis = key_basis.index_select(1, order)

    query_coordinates = queries @ key_basis
    query_direction_energy = query_coordinates.square().sum(dim=0)
    qk_direction_energy = key_energy * query_direction_energy
    qk_total_energy = qk_direction_energy.sum().clamp_min(1.0e-30)

    # The nonzero squared singular values of QK^T / sqrt(d) equal the
    # eigenvalues of Q(K^T K)Q^T / d.
    qk_row_gram = queries @ key_gram @ queries.transpose(0, 1)
    qk_row_gram /= float(head_dim)
    qk_singular_energy, qk_left_vectors = torch.linalg.eigh(qk_row_gram)
    qk_singular_energy = qk_singular_energy.clamp_min(0.0).flip(0)
    qk_left_vectors = qk_left_vectors.flip(1)
    qk_singular_energy_sum = qk_singular_energy.sum().clamp_min(1.0e-30)
    score_sum_over_tokens = (
        queries @ key.sum(dim=0) / math.sqrt(float(head_dim))
    )
    row_mean_energy_fraction = (
        score_sum_over_tokens.square().sum()
        / (
            float(key.shape[0])
            * qk_singular_energy_sum
        )
    ).clamp(0.0, 1.0)
    top_left_vector = qk_left_vectors[:, 0]
    top_right_constant_alignment = (
        top_left_vector.dot(score_sum_over_tokens).square()
        / (
            float(key.shape[0])
            * qk_singular_energy[0].clamp_min(1.0e-30)
        )
    ).clamp(0.0, 1.0)

    # Softmax is invariant to one additive constant per query. Row-centering
    # QK scores is exactly equivalent to centering K across history positions.
    centered_key = key - key.mean(dim=0, keepdim=True)
    centered_key_gram = centered_key.transpose(0, 1) @ centered_key
    centered_qk_row_gram = (
        queries @ centered_key_gram @ queries.transpose(0, 1)
    )
    centered_qk_row_gram /= float(head_dim)
    centered_qk_singular_energy = (
        torch.linalg.eigvalsh(centered_qk_row_gram).clamp_min(0.0).flip(0)
    )
    centered_qk_energy_sum = (
        centered_qk_singular_energy.sum().clamp_min(1.0e-30)
    )

    key_pca_qk_retained = qk_direction_energy[:rank].sum() / qk_total_energy
    qk_optimal_retained = (
        qk_singular_energy[:rank].sum() / qk_singular_energy_sum
    )
    query_energy = queries.square().sum().clamp_min(1.0e-30)
    query_energy_retained = (
        query_coordinates[:, :rank].square().sum() / query_energy
    )
    key_energy_retained = (
        key_energy[:rank].sum() / key_energy.sum().clamp_min(1.0e-30)
    )

    query_covariance = (
        queries.transpose(0, 1) @ queries / float(queries.shape[0])
    )
    query_gram = queries.transpose(0, 1) @ queries
    key_covariance = key_gram / float(key.shape[0])
    commutator = key_covariance @ query_covariance - query_covariance @ key_covariance
    commutator_ratio = (
        commutator.norm()
        / (
            key_covariance.norm()
            * query_covariance.norm()
        ).clamp_min(1.0e-30)
    )
    centered_key_covariance = centered_key_gram / float(key.shape[0])
    centered_commutator = (
        centered_key_covariance @ query_covariance
        - query_covariance @ centered_key_covariance
    )
    centered_commutator_ratio = (
        centered_commutator.norm()
        / (
            centered_key_covariance.norm()
            * query_covariance.norm()
        ).clamp_min(1.0e-30)
    )

    tail_error = qk_direction_energy[rank:].sum() / float(head_dim)
    spectral_tail_upper_bound = (
        key_energy[rank]
        * queries.square().sum()
        / float(head_dim)
        if rank < head_dim
        else torch.zeros((), device=key.device)
    )
    first = qk_singular_energy[0].clamp_min(1.0e-30)
    boundary = qk_singular_energy[min(rank - 1, qk_singular_energy.numel() - 1)]

    full_basis = key_basis[:, :rank]
    _, centered_key_basis = torch.linalg.eigh(centered_key_gram)
    centered_key_basis = centered_key_basis[:, -rank:]
    sampled_full_key = key[::basis_sample_stride]
    sampled_full_gram = sampled_full_key.transpose(0, 1) @ sampled_full_key
    _, sampled_full_basis = torch.linalg.eigh(sampled_full_gram)
    sampled_full_basis = sampled_full_basis[:, -rank:]

    production_prefix_count = min(int(key.shape[0]), basis_prefix_tokens)
    production_sampled_key = key[
        :production_prefix_count:basis_sample_stride
    ]
    production_gram = (
        production_sampled_key.transpose(0, 1) @ production_sampled_key
    )
    _, production_basis = torch.linalg.eigh(production_gram)
    production_basis = production_basis[:, -rank:]

    full_projection = projection_score_metrics(
        key_gram,
        query_gram,
        full_basis,
        head_dim,
    )
    sampled_full_projection = projection_score_metrics(
        key_gram,
        query_gram,
        sampled_full_basis,
        head_dim,
    )
    production_projection = projection_score_metrics(
        key_gram,
        query_gram,
        production_basis,
        head_dim,
    )
    centered_full_projection = projection_score_metrics(
        centered_key_gram,
        query_gram,
        full_basis,
        head_dim,
    )
    centered_key_projection = projection_score_metrics(
        centered_key_gram,
        query_gram,
        centered_key_basis,
        head_dim,
    )
    centered_sampled_full_projection = projection_score_metrics(
        centered_key_gram,
        query_gram,
        sampled_full_basis,
        head_dim,
    )
    centered_production_projection = projection_score_metrics(
        centered_key_gram,
        query_gram,
        production_basis,
        head_dim,
    )
    sampled_full_overlap = (
        (full_basis.transpose(0, 1) @ sampled_full_basis)
        .square()
        .sum()
        / float(rank)
    )
    production_overlap = (
        (full_basis.transpose(0, 1) @ production_basis)
        .square()
        .sum()
        / float(rank)
    )

    output = {
        "key_effective_rank": effective_rank(key_energy),
        "qk_effective_rank": effective_rank(qk_singular_energy),
        "centered_qk_effective_rank": effective_rank(
            centered_qk_singular_energy
        ),
        "qk_rank1_energy_fraction": float(
            (qk_singular_energy[0] / qk_singular_energy_sum).item()
        ),
        "centered_qk_rank1_energy_fraction": float(
            (
                centered_qk_singular_energy[0]
                / centered_qk_energy_sum
            ).item()
        ),
        "softmax_invariant_row_mean_energy_fraction": float(
            row_mean_energy_fraction.item()
        ),
        "qk_top_right_vector_constant_alignment": float(
            top_right_constant_alignment.item()
        ),
        "key_energy_retained_rank48": float(key_energy_retained.item()),
        "query_energy_in_key_pca48": float(query_energy_retained.item()),
        "qk_energy_retained_key_pca48": float(key_pca_qk_retained.item()),
        "qk_energy_retained_optimal_rank48": float(qk_optimal_retained.item()),
        "qk_optimality_gap": float(
            (qk_optimal_retained - key_pca_qk_retained).clamp_min(0.0).item()
        ),
        "centered_qk_energy_retained_optimal_rank48": float(
            (
                centered_qk_singular_energy[:rank].sum()
                / centered_qk_energy_sum
            ).item()
        ),
        "centered_qk_energy_retained_uncentered_key_pca48": (
            centered_full_projection["qk_fidelity"]
        ),
        "centered_qk_energy_retained_centered_key_pca48": (
            centered_key_projection["qk_fidelity"]
        ),
        "centered_qk_uncentered_key_pca_optimality_gap": float(
            max(
                0.0,
                (
                    centered_qk_singular_energy[:rank].sum()
                    / centered_qk_energy_sum
                ).item()
                - centered_full_projection["qk_fidelity"],
            )
        ),
        "qk_sigma1_over_sigma48": float(
            torch.sqrt(first / boundary.clamp_min(1.0e-30)).item()
        ),
        "key_query_covariance_commutator_ratio": float(
            commutator_ratio.item()
        ),
        "centered_key_query_covariance_commutator_ratio": float(
            centered_commutator_ratio.item()
        ),
        "qk_tail_frobenius_error_squared": float(tail_error.item()),
        "qk_tail_spectral_upper_bound_squared": float(
            spectral_tail_upper_bound.item()
        ),
        "qk_tail_bound_satisfied": float(
            tail_error <= spectral_tail_upper_bound + 1.0e-3
        ),
        "full_key_pca_qk_fidelity": full_projection["qk_fidelity"],
        "full_key_pca_qk_score_cosine": full_projection[
            "qk_score_cosine"
        ],
        "sampled_full_pca_qk_fidelity": sampled_full_projection[
            "qk_fidelity"
        ],
        "sampled_full_pca_qk_score_cosine": sampled_full_projection[
            "qk_score_cosine"
        ],
        "sampled_full_pca_subspace_overlap": float(
            sampled_full_overlap.item()
        ),
        "production_prefix_tokens": production_prefix_count,
        "production_prefix_samples": int(production_sampled_key.shape[0]),
        "production_prefix_pca_qk_fidelity": production_projection[
            "qk_fidelity"
        ],
        "production_prefix_pca_qk_relative_mse": production_projection[
            "qk_relative_mse"
        ],
        "production_prefix_pca_qk_score_cosine": production_projection[
            "qk_score_cosine"
        ],
        "production_prefix_pca_subspace_overlap": float(
            production_overlap.item()
        ),
        "centered_full_key_pca_qk_fidelity": centered_full_projection[
            "qk_fidelity"
        ],
        "centered_full_key_pca_qk_score_cosine": centered_full_projection[
            "qk_score_cosine"
        ],
        "centered_key_pca_qk_fidelity": centered_key_projection[
            "qk_fidelity"
        ],
        "centered_sampled_full_pca_qk_fidelity": (
            centered_sampled_full_projection["qk_fidelity"]
        ),
        "centered_sampled_full_pca_qk_score_cosine": (
            centered_sampled_full_projection["qk_score_cosine"]
        ),
        "centered_production_prefix_pca_qk_fidelity": (
            centered_production_projection["qk_fidelity"]
        ),
        "centered_production_prefix_pca_qk_score_cosine": (
            centered_production_projection["qk_score_cosine"]
        ),
    }
    for cutoff in (8, 16, 24, 32, 48, 64):
        if cutoff > head_dim:
            continue
        output[f"key_energy_retained_rank{cutoff}"] = float(
            (
                key_energy[:cutoff].sum()
                / key_energy.sum().clamp_min(1.0e-30)
            ).item()
        )
        output[f"qk_energy_retained_optimal_rank{cutoff}"] = float(
            (
                qk_singular_energy[:cutoff].sum()
                / qk_singular_energy_sum
            ).item()
        )
        output[f"centered_qk_energy_retained_optimal_rank{cutoff}"] = float(
            (
                centered_qk_singular_energy[:cutoff].sum()
                / centered_qk_energy_sum
            ).item()
        )
    if rank < qk_singular_energy.numel():
        output["qk_lambda_rank_over_next_ratio"] = float(
            (
                qk_singular_energy[rank - 1]
                / qk_singular_energy[rank].clamp_min(1.0e-30)
            ).item()
        )
    else:
        output["qk_lambda_rank_over_next_ratio"] = float("inf")

    for prefix_limit in (512, 1024, 2048, 4096, 8192):
        prefix_count = min(int(key.shape[0]), prefix_limit)
        if (
            prefix_count == production_prefix_count
            and basis_sample_stride == 32
        ):
            prefix_projection = production_projection
            prefix_overlap = production_overlap
            prefix_samples = int(production_sampled_key.shape[0])
        else:
            prefix_key = key[:prefix_count:basis_sample_stride]
            prefix_gram = prefix_key.transpose(0, 1) @ prefix_key
            _, prefix_basis = torch.linalg.eigh(prefix_gram)
            prefix_basis = prefix_basis[:, -rank:]
            prefix_projection = projection_score_metrics(
                key_gram,
                query_gram,
                prefix_basis,
                head_dim,
            )
            prefix_overlap = (
                (full_basis.transpose(0, 1) @ prefix_basis)
                .square()
                .sum()
                / float(rank)
            )
            prefix_samples = int(prefix_key.shape[0])
        centered_prefix_projection = projection_score_metrics(
            centered_key_gram,
            query_gram,
            (
                production_basis
                if prefix_count == production_prefix_count
                and basis_sample_stride == 32
                else prefix_basis
            ),
            head_dim,
        )
        stem = f"prefix{prefix_limit}_pca"
        output[f"{stem}_qk_fidelity"] = prefix_projection["qk_fidelity"]
        output[f"{stem}_qk_score_cosine"] = prefix_projection[
            "qk_score_cosine"
        ]
        output[f"{stem}_subspace_overlap"] = float(prefix_overlap.item())
        output[f"{stem}_sample_count"] = prefix_samples
        output[f"{stem}_centered_qk_fidelity"] = (
            centered_prefix_projection["qk_fidelity"]
        )
        output[f"{stem}_centered_qk_score_cosine"] = (
            centered_prefix_projection["qk_score_cosine"]
        )
    return output


def percentile(values: list[float], fraction: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, fraction).item())


def aggregate(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    summaries = []
    excluded = {
        "model",
        "topic",
        "trace_path",
        "layer",
        "kv_head",
        "history_tokens",
        "query_count",
        "rank",
    }
    for key, subset in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["cases"] = len(subset)
        numeric_fields = [
            field
            for field, value in subset[0].items()
            if field not in excluded and isinstance(value, (int, float))
        ]
        for field in numeric_fields:
            values = [float(row[field]) for row in subset]
            summary[f"{field}_mean"] = sum(values) / len(values)
            summary[f"{field}_p10"] = percentile(values, 0.10)
            summary[f"{field}_p50"] = percentile(values, 0.50)
            summary[f"{field}_p90"] = percentile(values, 0.90)
            summary[f"{field}_min"] = min(values)
            summary[f"{field}_max"] = max(values)
        summaries.append(summary)
    return summaries


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
    device: torch.device,
    basis_prefix_tokens: int,
    basis_sample_stride: int,
) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"{path} has no records")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

    rows = []
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
            metrics = spectral_metrics(
                history_key[kv_head],
                queries,
                rank,
                basis_prefix_tokens,
                basis_sample_stride,
            )
            rows.append(
                {
                    "model": model,
                    "topic": topic,
                    "trace_path": str(path),
                    "layer": layer,
                    "kv_head": kv_head,
                    "history_tokens": history_tokens,
                    "query_count": int(queries.shape[0]),
                    "rank": rank,
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
        description="Analyze stacked decode-query QK singular spectra."
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="MODEL=TOPIC=/path/to/trace.pt",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--basis_prefix_tokens", type=int, default=2048)
    parser.add_argument("--basis_sample_stride", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []
    for specification in args.trace:
        model, topic, path = parse_trace(specification)
        rows.extend(
            analyze_trace(
                model,
                topic,
                path,
                args.rank,
                device,
                args.basis_prefix_tokens,
                args.basis_sample_stride,
            )
        )
    if not rows:
        raise RuntimeError("no spectrum rows")

    report = {
        "config": {
            "traces": args.trace,
            "rank": args.rank,
            "basis_prefix_tokens": args.basis_prefix_tokens,
            "basis_sample_stride": args.basis_sample_stride,
            "device": str(device),
        },
        "overall": aggregate(rows, ()),
        "by_model": aggregate(rows, ("model",)),
        "by_model_topic": aggregate(rows, ("model", "topic")),
        "by_model_layer": aggregate(rows, ("model", "layer")),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "qk_spectrum_rows.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
