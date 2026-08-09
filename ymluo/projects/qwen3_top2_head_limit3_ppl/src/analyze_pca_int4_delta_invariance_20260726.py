from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def covariance_basis(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ascending eigenvalues and the leading uncentered-PCA directions."""

    working = matrix.float()
    second_moment = working.transpose(0, 1) @ working
    second_moment /= float(working.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvalues, eigenvectors[:, -rank:].contiguous()


def production_int4_dequantize(
    projected_key: torch.Tensor, group_size: int = 16
) -> torch.Tensor:
    """Emulate the grouped log-scale INT4 representation used by the CUDA path."""

    if projected_key.shape[-1] % group_size:
        raise ValueError("projection dimension must be divisible by group_size")
    original_shape = projected_key.shape
    grouped = projected_key.float().reshape(
        *original_shape[:-1], original_shape[-1] // group_size, group_size
    )
    exact_scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-8) / 7.0
    base_scales = exact_scales.amax(dim=-1, keepdim=True)
    exponents = torch.round(torch.log2(base_scales / exact_scales) * 4.0)
    exponents = exponents.clamp(0.0, 15.0)
    scales = base_scales * torch.pow(2.0, -exponents / 4.0)
    codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(-7.0, 7.0)
    return (codes * scales.unsqueeze(-1)).reshape(original_shape)


def production_int8_dequantize(projected_query: torch.Tensor) -> torch.Tensor:
    scale = projected_query.float().abs().amax().clamp_min(1.0e-8) / 127.0
    codes = torch.round(projected_query.float() / scale).clamp(-127.0, 127.0)
    return codes * scale


def centered_error_metrics(
    exact: torch.Tensor,
    approximate: torch.Tensor,
    attention: torch.Tensor,
) -> dict[str, float]:
    exact_float = exact.float()
    approximate_float = approximate.float()
    exact_centered = exact_float - exact_float.mean()
    approximate_centered = approximate_float - approximate_float.mean()
    delta = exact_centered - approximate_centered
    denominator = exact_centered.square().mean().sqrt().clamp_min(1.0e-12)
    correlation_denominator = (
        exact_centered.norm() * approximate_centered.norm()
    ).clamp_min(1.0e-12)
    attention = attention.float()
    attention_sum = attention.sum().clamp_min(1.0e-12)
    normalized_attention = attention / attention_sum
    weighted_delta = exact_float - approximate_float
    weighted_delta = weighted_delta - (normalized_attention * weighted_delta).sum()
    return {
        "centered_nrmse": float(
            (delta.square().mean().sqrt() / denominator).item()
        ),
        "pearson": float(
            ((exact_centered * approximate_centered).sum() / correlation_denominator).item()
        ),
        "attention_weighted_abs_error": float(
            (normalized_attention * weighted_delta.abs()).sum().item()
        ),
        "attention_weighted_rmse": float(
            (normalized_attention * weighted_delta.square()).sum().sqrt().item()
        ),
    }


def candidate_metrics(
    proxy_scores: torch.Tensor,
    exact_history_scores: torch.Tensor,
    current_score: torch.Tensor,
    full_attention: torch.Tensor,
    history_value: torch.Tensor | None,
    current_value: torch.Tensor | None,
    fraction: float,
) -> dict[str, float]:
    history_count = int(exact_history_scores.numel())
    keep_count = min(history_count, max(1, math.ceil(fraction * history_count)))
    selected = torch.topk(proxy_scores, k=keep_count, sorted=False).indices
    exact_top = torch.topk(exact_history_scores, k=keep_count, sorted=False).indices
    selected_mask = torch.zeros(
        history_count, dtype=torch.bool, device=exact_history_scores.device
    )
    selected_mask[selected] = True
    exact_top_mask = torch.zeros_like(selected_mask)
    exact_top_mask[exact_top] = True
    topk_recall = selected_mask[exact_top].float().mean()
    history_attention = full_attention[:-1]
    current_mass = full_attention[-1]
    retained_history_mass = history_attention[selected].sum()
    retained_mass = retained_history_mass + current_mass
    optimal_history_mass = history_attention[exact_top].sum()
    optimal_retained_mass = optimal_history_mass + current_mass
    exact_top_intersection_mass = history_attention[
        exact_top[selected_mask[exact_top]]
    ].sum()
    missed_exact_top_mass = optimal_history_mass - exact_top_intersection_mass
    selected_extra_mass = history_attention[selected[~exact_top_mask[selected]]].sum()
    score_error_linf = (proxy_scores - exact_history_scores).abs().amax()
    deterministic_retained_mass_lower_bound = (
        current_mass
        + torch.exp(-2.0 * score_error_linf) * optimal_history_mass
    )
    result = {
        "fraction": float(fraction),
        "keep_count": float(keep_count),
        "topk_recall": float(topk_recall.item()),
        "attention_mass_weighted_topk_recall": float(
            (
                exact_top_intersection_mass
                / optimal_history_mass.clamp_min(1.0e-12)
            ).item()
        ),
        "missed_exact_top_attention_mass": float(
            missed_exact_top_mass.item()
        ),
        "selected_extra_attention_mass": float(selected_extra_mass.item()),
        "optimal_retained_attention_mass": float(
            optimal_retained_mass.item()
        ),
        "retained_attention_mass": float(retained_mass.item()),
        "retained_attention_mass_regret": float(
            (optimal_retained_mass - retained_mass).clamp_min(0.0).item()
        ),
        "dropped_attention_mass": float((1.0 - retained_mass).clamp_min(0.0).item()),
        "proxy_score_error_linf": float(score_error_linf.item()),
        "deterministic_retained_mass_lower_bound": float(
            deterministic_retained_mass_lower_bound.item()
        ),
        "deterministic_mass_bound_satisfied": float(
            retained_mass + 1.0e-6
            >= deterministic_retained_mass_lower_bound
        ),
    }
    if history_value is None or current_value is None:
        return result

    support_scores = torch.cat((exact_history_scores[selected], current_score.view(1)))
    support_values = torch.cat(
        (history_value.index_select(0, selected), current_value.view(1, -1)),
        dim=0,
    )
    sparse_output = torch.softmax(support_scores, dim=0) @ support_values.float()
    full_values = torch.cat((history_value, current_value.view(1, -1)), dim=0)
    full_output = full_attention @ full_values.float()
    difference = sparse_output - full_output
    output_norm = full_output.norm().clamp_min(1.0e-12)
    value_norm_max = full_values.norm(dim=-1).amax()
    theoretical_bound = 2.0 * (1.0 - retained_mass).clamp_min(0.0) * value_norm_max
    cosine_denominator = (sparse_output.norm() * full_output.norm()).clamp_min(1.0e-12)
    result.update(
        {
            "attention_output_relative_l2": float((difference.norm() / output_norm).item()),
            "attention_output_cosine": float(
                ((sparse_output * full_output).sum() / cosine_denominator).item()
            ),
            "attention_output_absolute_l2": float(difference.norm().item()),
            "attention_output_bound": float(theoretical_bound.item()),
            "output_bound_satisfied": float(
                difference.norm() <= theoretical_bound + 1.0e-5
            ),
        }
    )
    return result


def sampled_quantile_metrics(
    proxy_scores: torch.Tensor,
    exact_history_scores: torch.Tensor,
    current_score: torch.Tensor,
    full_attention: torch.Tensor,
    history_value: torch.Tensor | None,
    current_value: torch.Tensor | None,
    fraction: float,
    sample_count: int,
    capacity_fraction: float,
) -> dict[str, float]:
    """Diagnose the deterministic midpoint threshold used by production."""

    history_count = int(proxy_scores.numel())
    sample_count = min(history_count, max(1, sample_count))
    sample_index = torch.arange(sample_count, device=proxy_scores.device)
    sampled_token = torch.div(
        (2 * sample_index + 1) * history_count,
        2 * sample_count,
        rounding_mode="floor",
    ).clamp_max(history_count - 1)
    sample_keep = min(
        sample_count,
        max(1, math.ceil(fraction * sample_count)),
    )
    sampled_threshold = torch.topk(
        proxy_scores.index_select(0, sampled_token),
        k=sample_keep,
        sorted=True,
    ).values[-1]
    target_count = min(
        history_count,
        max(1, math.ceil(fraction * history_count)),
    )
    exact_proxy_threshold = torch.topk(
        proxy_scores,
        k=target_count,
        sorted=True,
    ).values[-1]
    selected = torch.nonzero(
        proxy_scores >= sampled_threshold,
        as_tuple=False,
    ).flatten()
    capacity = min(
        history_count,
        max(
            target_count,
            math.ceil(capacity_fraction * history_count),
        ),
    )
    overflow = int(selected.numel()) > capacity

    selected_mask = torch.zeros(
        history_count,
        dtype=torch.bool,
        device=proxy_scores.device,
    )
    selected_mask[selected] = True
    exact_top = torch.topk(
        exact_history_scores,
        k=target_count,
        sorted=False,
    ).indices
    exact_top_mask = torch.zeros_like(selected_mask)
    exact_top_mask[exact_top] = True
    history_attention = full_attention[:-1]
    current_mass = full_attention[-1]
    retained_mass = history_attention[selected].sum() + current_mass
    optimal_history_mass = history_attention[exact_top].sum()
    optimal_retained_mass = optimal_history_mass + current_mass
    exact_top_intersection_mass = history_attention[
        exact_top[selected_mask[exact_top]]
    ].sum()

    result = {
        "fraction": float(fraction),
        "keep_count": float(target_count),
        "sample_count": float(sample_count),
        "sampled_selected_count": float(selected.numel()),
        "sampled_selected_fraction": float(selected.numel() / history_count),
        "candidate_capacity": float(capacity),
        "candidate_capacity_fraction": float(capacity / history_count),
        "sampled_candidate_overflow": float(overflow),
        "sampled_threshold": float(sampled_threshold.item()),
        "exact_proxy_threshold": float(exact_proxy_threshold.item()),
        "sampled_threshold_absolute_error": float(
            (sampled_threshold - exact_proxy_threshold).abs().item()
        ),
        "topk_recall": float(
            selected_mask[exact_top].float().mean().item()
        ),
        "attention_mass_weighted_topk_recall": float(
            (
                exact_top_intersection_mass
                / optimal_history_mass.clamp_min(1.0e-12)
            ).item()
        ),
        "missed_exact_top_attention_mass": float(
            (optimal_history_mass - exact_top_intersection_mass).item()
        ),
        "selected_extra_attention_mass": float(
            history_attention[selected[~exact_top_mask[selected]]].sum().item()
        ),
        "optimal_retained_attention_mass": float(
            optimal_retained_mass.item()
        ),
        "retained_attention_mass": float(retained_mass.item()),
        "retained_attention_mass_regret": float(
            (optimal_retained_mass - retained_mass).clamp_min(0.0).item()
        ),
        "dropped_attention_mass": float(
            (1.0 - retained_mass).clamp_min(0.0).item()
        ),
    }
    if history_value is None or current_value is None:
        return result

    support_scores = torch.cat(
        (exact_history_scores[selected], current_score.view(1))
    )
    support_values = torch.cat(
        (
            history_value.index_select(0, selected),
            current_value.view(1, -1),
        ),
        dim=0,
    )
    sparse_output = torch.softmax(support_scores, dim=0) @ support_values.float()
    full_values = torch.cat(
        (history_value, current_value.view(1, -1)),
        dim=0,
    )
    full_output = full_attention @ full_values.float()
    difference = sparse_output - full_output
    output_norm = full_output.norm().clamp_min(1.0e-12)
    cosine_denominator = (
        sparse_output.norm() * full_output.norm()
    ).clamp_min(1.0e-12)
    result.update(
        {
            "attention_output_relative_l2": float(
                (difference.norm() / output_norm).item()
            ),
            "attention_output_cosine": float(
                (
                    (sparse_output * full_output).sum()
                    / cosine_denominator
                ).item()
            ),
        }
    )
    return result


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
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        record = {field: value for field, value in zip(group_fields, key)}
        record["cases"] = len(items)
        numeric_fields = [
            field
            for field, value in items[0].items()
            if field not in group_fields
            and field
            not in {
                "topic",
                "method",
                "stage",
                "trace_path",
            }
            and isinstance(value, (int, float))
        ]
        for field in numeric_fields:
            stats = quantiles(float(item[field]) for item in items)
            for statistic, value in stats.items():
                record[f"{field}_{statistic}"] = value
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_trace_spec(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        path = Path(specification)
        return path.stem, path
    topic, raw_path = specification.split("=", 1)
    return topic, Path(raw_path)


@torch.inference_mode()
def analyze_trace(
    topic: str,
    trace_path: Path,
    rank: int,
    sample_stride: int,
    basis_prefix_tokens: int,
    quantile_sample_count: int,
    fractions: tuple[float, ...],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"{trace_path} contains no records")
    spectrum_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0]
        all_key = record["key"].to(device).float()[0]
        all_value = (
            record["value"].to(device).float()[0]
            if "value" in record
            else None
        )
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(all_key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(all_key.shape[1]) - 1
        history_key = all_key[:, :history_count]

        sampled_full_bases: list[torch.Tensor] = []
        production_bases: list[torch.Tensor] = []
        full_bases: list[torch.Tensor] = []
        full_projected_history_keys: list[torch.Tensor] = []
        sampled_full_projected_history_keys: list[torch.Tensor] = []
        production_projected_history_keys: list[torch.Tensor] = []
        production_quantized_projected_keys: list[torch.Tensor] = []
        for kv_head in range(kv_heads):
            key = history_key[kv_head]
            full_eigenvalues, full_basis = covariance_basis(key, rank)
            sampled_full_eigenvalues, sampled_full_basis = covariance_basis(
                key[::sample_stride], rank
            )
            prefix_count = min(history_count, basis_prefix_tokens)
            production_eigenvalues, production_basis = covariance_basis(
                key[:prefix_count:sample_stride],
                rank,
            )
            sampled_full_bases.append(sampled_full_basis)
            production_bases.append(production_basis)
            full_bases.append(full_basis)
            full_projected_history_keys.append(key @ full_basis)
            sampled_full_projected_history_keys.append(
                key @ sampled_full_basis
            )
            production_projected_key = key @ production_basis
            production_projected_history_keys.append(
                production_projected_key
            )
            production_quantized_projected_keys.append(
                production_int4_dequantize(production_projected_key)
            )

            full_energy = full_eigenvalues.clamp_min(0.0)
            retained_energy = full_energy[-rank:].sum() / full_energy.sum().clamp_min(
                1.0e-12
            )
            singular_values = full_energy.clamp_min(0.0).sqrt()
            normalized_energy = full_energy / full_energy.sum().clamp_min(1.0e-12)
            entropy = -(
                normalized_energy
                * normalized_energy.clamp_min(1.0e-30).log()
            ).sum()
            principal_cosines = torch.linalg.svdvals(
                full_basis.transpose(0, 1) @ sampled_full_basis
            )
            production_principal_cosines = torch.linalg.svdvals(
                full_basis.transpose(0, 1) @ production_basis
            )
            boundary_ratio = full_energy[-rank] / full_energy[-rank - 1].clamp_min(
                1.0e-12
            )
            spectrum_rows.append(
                {
                    "topic": topic,
                    "trace_path": str(trace_path),
                    "record_index": record_index,
                    "layer": layer,
                    "kv_head": kv_head,
                    "history_tokens": history_count,
                    "rank": rank,
                    "key_energy_retained": float(retained_energy.item()),
                    "key_energy_dropped": float((1.0 - retained_energy).item()),
                    "effective_rank": float(torch.exp(entropy).item()),
                    "sigma1_over_sigma48": float(
                        (
                            singular_values[-1]
                            / singular_values[-rank].clamp_min(1.0e-12)
                        ).item()
                    ),
                    "lambda48_over_lambda49": float(boundary_ratio.item()),
                    "sampled_full_subspace_overlap": float(
                        principal_cosines.square().mean().item()
                    ),
                    "sampled_full_min_principal_cosine": float(
                        principal_cosines.min().item()
                    ),
                    "sampled_covariance_energy_retained": float(
                        (
                            sampled_full_eigenvalues[-rank:]
                            .clamp_min(0.0)
                            .sum()
                            / sampled_full_eigenvalues.clamp_min(0.0)
                            .sum()
                            .clamp_min(1.0e-12)
                        ).item()
                    ),
                    "production_prefix_tokens": prefix_count,
                    "production_prefix_samples": int(
                        key[:prefix_count:sample_stride].shape[0]
                    ),
                    "production_sampled_covariance_energy_retained": float(
                        (
                            production_eigenvalues[-rank:]
                            .clamp_min(0.0)
                            .sum()
                            / production_eigenvalues.clamp_min(0.0)
                            .sum()
                            .clamp_min(1.0e-12)
                        ).item()
                    ),
                    "production_full_subspace_overlap": float(
                        production_principal_cosines.square().mean().item()
                    ),
                    "production_full_min_principal_cosine": float(
                        production_principal_cosines.min().item()
                    ),
                }
            )

        for query_head in range(query_heads):
            kv_head = query_head // groups
            q = query[query_head]
            key = history_key[kv_head]
            current_key = all_key[kv_head, -1]
            value = all_value[kv_head, :history_count] if all_value is not None else None
            current_value = all_value[kv_head, -1] if all_value is not None else None
            sampled_full_basis = sampled_full_bases[kv_head]
            production_basis = production_bases[kv_head]
            full_basis = full_bases[kv_head]
            full_projected_key = full_projected_history_keys[kv_head]
            sampled_full_projected_key = (
                sampled_full_projected_history_keys[kv_head]
            )
            production_projected_key = (
                production_projected_history_keys[kv_head]
            )
            production_quantized_key = (
                production_quantized_projected_keys[kv_head]
            )
            sampled_full_projected_query = q @ sampled_full_basis
            production_projected_query = q @ production_basis
            production_quantized_query = production_int8_dequantize(
                production_projected_query
            )

            exact_scores = (key @ q) * scaling
            current_score = (current_key @ q) * scaling
            full_svd_scores = (full_projected_key @ (q @ full_basis)) * scaling
            sampled_full_pca_scores = (
                sampled_full_projected_key @ sampled_full_projected_query
            ) * scaling
            production_pca_scores = (
                production_projected_key @ production_projected_query
            ) * scaling
            production_int4_scores = (
                production_quantized_key @ production_projected_query
            ) * scaling
            production_scores = (
                production_quantized_key @ production_quantized_query
            ) * scaling
            full_attention = torch.softmax(
                torch.cat((exact_scores, current_score.view(1))), dim=0
            )
            history_attention = full_attention[:-1]
            exact_centered_energy = (
                exact_scores.float() - exact_scores.float().mean()
            ).square().sum().clamp_min(1.0e-12)
            pca_tail = exact_scores - production_pca_scores
            int4_delta = production_pca_scores - production_int4_scores
            centered_pca_tail = pca_tail - pca_tail.mean()
            centered_int4_delta = int4_delta - int4_delta.mean()
            delta_cosine_denominator = (
                centered_pca_tail.norm() * centered_int4_delta.norm()
            ).clamp_min(1.0e-12)
            query_energy_retained = (
                production_projected_query.square().sum()
                / q.square().sum().clamp_min(1.0e-12)
            )

            stages = {
                "exact_qk": exact_scores,
                "full_svd48_fp32": full_svd_scores,
                "sampled_full_pca48_fp32": sampled_full_pca_scores,
                "production_prefix_pca48_fp32": production_pca_scores,
                "production_prefix_pca48_int4_key": production_int4_scores,
                "production_pca48_int4k_int8q": production_scores,
            }
            common = {
                "topic": topic,
                "trace_path": str(trace_path),
                "record_index": record_index,
                "layer": layer,
                "query_head": query_head,
                "kv_head": kv_head,
                "history_tokens": history_count,
                "rank": rank,
                "query_energy_retained": float(query_energy_retained.item()),
                "pca_tail_score_energy_ratio": float(
                    (
                        (pca_tail - pca_tail.mean()).square().sum()
                        / exact_centered_energy
                    ).item()
                ),
                "int4_score_energy_ratio": float(
                    (
                        centered_int4_delta.square().sum()
                        / exact_centered_energy
                    ).item()
                ),
                "int4_to_pca_tail_energy_ratio": float(
                    (
                        centered_int4_delta.square().sum()
                        / centered_pca_tail.square().sum().clamp_min(1.0e-12)
                    ).item()
                ),
                "pca_tail_int4_delta_cosine": float(
                    (
                        (centered_pca_tail * centered_int4_delta).sum()
                        / delta_cosine_denominator
                    ).item()
                ),
            }
            for stage, proxy_scores in stages.items():
                score_rows.append(
                    {
                        **common,
                        "stage": stage,
                        **centered_error_metrics(
                            exact_scores, proxy_scores, history_attention
                        ),
                    }
                )
                for fraction in fractions:
                    candidate_rows.append(
                        {
                            **common,
                            "method": stage,
                            **candidate_metrics(
                                proxy_scores,
                                exact_scores,
                                current_score,
                                full_attention,
                                value,
                                current_value,
                                fraction,
                            ),
                        }
                    )
            for fraction in fractions:
                capacity_fraction = min(
                    1.0,
                    max(2.0 * fraction, fraction + 0.04),
                )
                candidate_rows.append(
                    {
                        **common,
                        "method": (
                            "production_pca48_int4k_int8q_"
                            "sampled_quantile_uncapped"
                        ),
                        **sampled_quantile_metrics(
                            production_scores,
                            exact_scores,
                            current_score,
                            full_attention,
                            value,
                            current_value,
                            fraction,
                            quantile_sample_count,
                            capacity_fraction,
                        ),
                    }
                )

    return spectrum_rows, score_rows, candidate_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how PCA spectral truncation and production INT4/INT8 "
            "perturbations propagate from QK scores to attention mass and outputs."
        )
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="TOPIC=/path/to/trace.pt; may be supplied more than once",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--basis_prefix_tokens", type=int, default=2048)
    parser.add_argument("--quantile_sample_count", type=int, default=256)
    parser.add_argument("--fractions", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fractions = tuple(
        sorted({float(value) for value in args.fractions.split(",") if value})
    )
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must contain values in (0, 1]")
    if (
        args.rank <= 0
        or args.sample_stride <= 0
        or args.basis_prefix_tokens <= 0
        or args.quantile_sample_count <= 0
    ):
        raise ValueError("rank, prefix, stride, and sample count must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    spectrum_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for specification in args.trace:
        topic, trace_path = parse_trace_spec(specification)
        current_spectrum, current_scores, current_candidates = analyze_trace(
            topic,
            trace_path,
            args.rank,
            args.sample_stride,
            args.basis_prefix_tokens,
            args.quantile_sample_count,
            fractions,
            device,
        )
        spectrum_rows.extend(current_spectrum)
        score_rows.extend(current_scores)
        candidate_rows.extend(current_candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "spectrum_rows.csv", spectrum_rows)
    write_csv(args.output_dir / "score_rows.csv", score_rows)
    write_csv(args.output_dir / "candidate_rows.csv", candidate_rows)
    report = {
        "config": {
            "traces": args.trace,
            "rank": args.rank,
            "sample_stride": args.sample_stride,
            "basis_prefix_tokens": args.basis_prefix_tokens,
            "quantile_sample_count": args.quantile_sample_count,
            "fractions": fractions,
            "device": str(device),
            "candidate_attention": (
                "proxy selects candidates; original exact Q/K/V compute attention"
            ),
        },
        "spectrum_overall": aggregate(spectrum_rows, ()),
        "spectrum_by_topic": aggregate(spectrum_rows, ("topic",)),
        "score_overall": aggregate(score_rows, ("stage",)),
        "score_by_topic": aggregate(score_rows, ("topic", "stage")),
        "candidate_overall": aggregate(candidate_rows, ("method", "fraction")),
        "candidate_by_topic": aggregate(
            candidate_rows, ("topic", "method", "fraction")
        ),
        "candidate_by_layer": aggregate(
            candidate_rows, ("layer", "method", "fraction")
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
