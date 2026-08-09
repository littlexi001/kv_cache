from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def covariance_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    working = matrix.float()
    gram = working.transpose(0, 1) @ working
    _, eigenvectors = torch.linalg.eigh(gram)
    return eigenvectors[:, -rank:].contiguous()


def production_int4_dequantize(
    projected_key: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    if projected_key.shape[-1] % group_size:
        raise ValueError("projection dimension must be divisible by group_size")
    original_shape = projected_key.shape
    grouped = projected_key.float().reshape(
        *original_shape[:-1],
        original_shape[-1] // group_size,
        group_size,
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


def midpoint_sample_indices(
    history_count: int,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    count = min(history_count, max(1, sample_count))
    sample_index = torch.arange(count, device=device)
    return torch.div(
        (2 * sample_index + 1) * history_count,
        2 * count,
        rounding_mode="floor",
    ).clamp_max(history_count - 1)


def top_mass(probabilities: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or probabilities.numel() == 0:
        return probabilities.new_zeros(())
    count = min(count, int(probabilities.numel()))
    return torch.topk(probabilities, k=count, sorted=False).values.sum()


def ranking_certificate_metrics(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    full_attention: torch.Tensor,
    fraction: float,
    gamma_grid: tuple[float, ...],
    score_error_upper_bound: torch.Tensor | None = None,
) -> dict[str, float]:
    history_count = int(exact_scores.numel())
    keep_count = min(history_count, max(1, math.ceil(fraction * history_count)))
    exact_values, exact_top = torch.topk(
        exact_scores,
        k=keep_count,
        sorted=True,
    )
    proxy_values, proxy_top = torch.topk(
        proxy_scores,
        k=keep_count,
        sorted=True,
    )
    exact_threshold = exact_values[-1]
    proxy_threshold = proxy_values[-1]
    exact_mask = torch.zeros(
        history_count,
        dtype=torch.bool,
        device=exact_scores.device,
    )
    proxy_mask = torch.zeros_like(exact_mask)
    exact_mask[exact_top] = True
    proxy_mask[proxy_top] = True

    history_attention = full_attention[:-1].float()
    current_mass = full_attention[-1].float()
    exact_top_mass = history_attention[exact_mask].sum()
    proxy_top_mass = history_attention[proxy_mask].sum()
    intersection_mass = history_attention[exact_mask & proxy_mask].sum()

    score_error = proxy_scores.float() - exact_scores.float()
    error_range = score_error.amax() - score_error.amin()
    mean_centered_error = score_error - score_error.mean()
    error_l2_squared = mean_centered_error.square().sum()
    robust_core = exact_mask & (
        exact_scores > exact_threshold + error_range
    )
    robust_core_mass = history_attention[robust_core].sum()
    robust_core_satisfied = (
        (~robust_core) | proxy_mask
    ).all()

    outside_mask = ~exact_mask
    error_midrange = 0.5 * (score_error.amax() + score_error.amin())
    tight_tokenwise_bound = (score_error - error_midrange).abs()

    def tokenwise_core(bound: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outside_upper = (
            (exact_scores[outside_mask] + bound[outside_mask]).amax()
            if outside_mask.any()
            else exact_scores.new_tensor(float("-inf"))
        )
        core = exact_mask & (exact_scores - bound > outside_upper)
        return core, history_attention[core].sum()

    tight_core, tight_core_mass = tokenwise_core(tight_tokenwise_bound)
    tight_core_satisfied = ((~tight_core) | proxy_mask).all()

    best_bound = exact_top_mass
    best_gamma = exact_scores.new_zeros(())
    best_boundary_mass = exact_top_mass
    best_large_error_count = torch.tensor(
        keep_count,
        dtype=torch.int64,
        device=exact_scores.device,
    )
    for gamma in gamma_grid:
        boundary = exact_mask & (
            exact_scores <= exact_threshold + 2.0 * gamma
        )
        boundary_mass = history_attention[boundary].sum()
        far_probabilities = history_attention[exact_mask & ~boundary]
        large_error_count = min(
            keep_count,
            int(math.ceil(float(error_l2_squared.item()) / (gamma * gamma))),
        )
        candidate_bound = (
            boundary_mass
            + top_mass(far_probabilities, large_error_count)
        ).clamp_max(exact_top_mass)
        if candidate_bound < best_bound:
            best_bound = candidate_bound
            best_gamma = exact_scores.new_tensor(gamma)
            best_boundary_mass = boundary_mass
            best_large_error_count = torch.tensor(
                large_error_count,
                dtype=torch.int64,
                device=exact_scores.device,
            )

    additional_omitted_mass = (
        exact_top_mass - proxy_top_mass
    ).clamp_min(0.0)
    result = {
        "fraction": float(fraction),
        "keep_count": float(keep_count),
        "exact_top_threshold": float(exact_threshold.item()),
        "proxy_top_threshold": float(proxy_threshold.item()),
        "score_error_range": float(error_range.item()),
        "score_error_l2_squared_mean_centered": float(
            error_l2_squared.item()
        ),
        "topk_recall": float(
            proxy_mask[exact_top].float().mean().item()
        ),
        "exact_top_attention_mass": float(exact_top_mass.item()),
        "proxy_top_attention_mass": float(proxy_top_mass.item()),
        "mass_weighted_topk_recall": float(
            (
                intersection_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        ),
        "additional_omitted_attention_mass": float(
            additional_omitted_mass.item()
        ),
        "robust_core_attention_mass": float(robust_core_mass.item()),
        "robust_core_fraction_of_exact_top_mass": float(
            (
                robust_core_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        ),
        "robust_core_inclusion_satisfied": float(robust_core_satisfied.item()),
        "tight_tokenwise_core_attention_mass": float(
            tight_core_mass.item()
        ),
        "tight_tokenwise_core_fraction_of_exact_top_mass": float(
            (
                tight_core_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        ),
        "tight_tokenwise_core_inclusion_satisfied": float(
            tight_core_satisfied.item()
        ),
        "uniform_margin_retained_mass_lower_bound": float(
            (current_mass + robust_core_mass).item()
        ),
        "actual_proxy_retained_mass": float(
            (current_mass + proxy_top_mass).item()
        ),
        "best_l2_mass_bound_gamma": float(best_gamma.item()),
        "best_l2_false_negative_mass_upper_bound": float(best_bound.item()),
        "best_l2_boundary_mass": float(best_boundary_mass.item()),
        "best_l2_large_error_count_bound": float(
            best_large_error_count.item()
        ),
        "best_l2_bound_satisfied": float(
            additional_omitted_mass <= best_bound + 1.0e-6
        ),
    }
    for gamma in gamma_grid:
        boundary = exact_mask & (
            exact_scores <= exact_threshold + 2.0 * gamma
        )
        boundary_mass = history_attention[boundary].sum()
        stem = f"gamma{str(gamma).replace('.', 'p')}"
        result[f"{stem}_top_boundary_mass"] = float(boundary_mass.item())
        result[f"{stem}_top_boundary_mass_fraction"] = float(
            (
                boundary_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        )
    if score_error_upper_bound is not None:
        upper_bound = score_error_upper_bound.float()
        if upper_bound.shape != exact_scores.shape:
            raise ValueError("score_error_upper_bound shape mismatch")
        bounded_error = (exact_scores.float() - proxy_scores.float()).abs()
        bound_satisfied = bounded_error <= upper_bound + 2.0e-5
        norm_core, norm_core_mass = tokenwise_core(upper_bound)
        norm_core_satisfied = ((~norm_core) | proxy_mask).all()
        result.update(
            {
                "norm_error_bound_satisfied": float(
                    bound_satisfied.all().item()
                ),
                "norm_error_bound_max_violation": float(
                    (bounded_error - upper_bound).clamp_min(0.0).amax().item()
                ),
                "norm_error_bound_mean": float(upper_bound.mean().item()),
                "norm_error_bound_p90": float(
                    torch.quantile(upper_bound, 0.90).item()
                ),
                "norm_tokenwise_core_attention_mass": float(
                    norm_core_mass.item()
                ),
                "norm_tokenwise_core_fraction_of_exact_top_mass": float(
                    (
                        norm_core_mass
                        / exact_top_mass.clamp_min(1.0e-30)
                    ).item()
                ),
                "norm_tokenwise_core_inclusion_satisfied": float(
                    norm_core_satisfied.item()
                ),
            }
        )
    return result


def sampled_threshold_metrics(
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    full_attention: torch.Tensor,
    fraction: float,
    sample_count: int,
    score_error_upper_bound: torch.Tensor | None = None,
) -> dict[str, float]:
    history_count = int(exact_scores.numel())
    keep_count = min(history_count, max(1, math.ceil(fraction * history_count)))
    sample_index = midpoint_sample_indices(
        history_count,
        sample_count,
        exact_scores.device,
    )
    sample_keep = min(
        int(sample_index.numel()),
        max(1, math.ceil(fraction * int(sample_index.numel()))),
    )
    sampled_threshold = torch.topk(
        proxy_scores.index_select(0, sample_index),
        k=sample_keep,
        sorted=True,
    ).values[-1]
    exact_proxy_threshold = torch.topk(
        proxy_scores,
        k=keep_count,
        sorted=True,
    ).values[-1]
    exact_threshold = torch.topk(
        exact_scores,
        k=keep_count,
        sorted=True,
    ).values[-1]
    threshold_error = (sampled_threshold - exact_proxy_threshold).abs()
    score_error = proxy_scores.float() - exact_scores.float()
    error_range = score_error.amax() - score_error.amin()

    selected = proxy_scores >= sampled_threshold
    exact_top = torch.topk(
        exact_scores,
        k=keep_count,
        sorted=False,
    ).indices
    exact_mask = torch.zeros_like(selected)
    exact_mask[exact_top] = True
    robust_core = exact_mask & (
        exact_scores
        > exact_threshold + error_range + threshold_error
    )
    robust_core_satisfied = ((~robust_core) | selected).all()
    history_attention = full_attention[:-1].float()
    current_mass = full_attention[-1].float()
    exact_top_mass = history_attention[exact_mask].sum()
    intersection_mass = history_attention[exact_mask & selected].sum()
    selected_mass = history_attention[selected].sum()
    robust_core_mass = history_attention[robust_core].sum()

    result = {
        "sample_count": float(sample_index.numel()),
        "sampled_selected_count": float(selected.sum().item()),
        "sampled_selected_fraction": float(
            selected.float().mean().item()
        ),
        "sampled_threshold": float(sampled_threshold.item()),
        "exact_proxy_threshold": float(exact_proxy_threshold.item()),
        "sampled_threshold_absolute_error": float(threshold_error.item()),
        "sampled_topk_recall": float(
            selected[exact_top].float().mean().item()
        ),
        "sampled_mass_weighted_topk_recall": float(
            (
                intersection_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        ),
        "sampled_retained_attention_mass": float(
            (current_mass + selected_mass).item()
        ),
        "sampled_additional_omitted_attention_mass": float(
            (exact_top_mass - selected_mass).clamp_min(0.0).item()
        ),
        "sampled_robust_core_attention_mass": float(
            robust_core_mass.item()
        ),
        "sampled_robust_core_fraction_of_exact_top_mass": float(
            (
                robust_core_mass
                / exact_top_mass.clamp_min(1.0e-30)
            ).item()
        ),
        "sampled_robust_core_inclusion_satisfied": float(
            robust_core_satisfied.item()
        ),
        "sampled_uniform_margin_retained_mass_lower_bound": float(
            (current_mass + robust_core_mass).item()
        ),
    }
    if score_error_upper_bound is not None:
        upper_bound = score_error_upper_bound.float()
        norm_core = exact_mask & (
            exact_scores - upper_bound > sampled_threshold
        )
        norm_core_mass = history_attention[norm_core].sum()
        result.update(
            {
                "sampled_norm_tokenwise_core_attention_mass": float(
                    norm_core_mass.item()
                ),
                "sampled_norm_tokenwise_core_fraction_of_exact_top_mass": float(
                    (
                        norm_core_mass
                        / exact_top_mass.clamp_min(1.0e-30)
                    ).item()
                ),
                "sampled_norm_tokenwise_core_inclusion_satisfied": float(
                    (((~norm_core) | selected).all()).item()
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
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    excluded = {
        "model",
        "topic",
        "trace_path",
        "record_index",
        "layer",
        "query_head",
        "kv_head",
        "method",
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
    trace_path: Path,
    rank: int,
    sample_stride: int,
    basis_prefix_tokens: int,
    quantile_sample_count: int,
    fractions: tuple[float, ...],
    gamma_grid: tuple[float, ...],
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"{trace_path} contains no records")
    rows: list[dict[str, Any]] = []
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)

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
        all_key = state_record["key"].to(device).float()[0]
        query_heads = int(layer_records[0]["query"].shape[1])
        kv_heads = int(all_key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(all_key.shape[1]) - 1
        history_key = all_key[:, :history_count]

        bases: list[torch.Tensor] = []
        projected_keys: list[torch.Tensor] = []
        quantized_keys: list[torch.Tensor] = []
        prefix_count = min(history_count, basis_prefix_tokens)
        for kv_head in range(kv_heads):
            key = history_key[kv_head]
            basis = covariance_basis(
                key[:prefix_count:sample_stride],
                rank,
            )
            projected_key = key @ basis
            bases.append(basis)
            projected_keys.append(projected_key)
            quantized_keys.append(
                production_int4_dequantize(projected_key)
            )

        for record_index, record in enumerate(layer_records):
            query = record["query"].to(device).float()[0, :, 0]
            scaling = float(record["scaling"])
            for query_head in range(query_heads):
                kv_head = query_head // groups
                q = query[query_head]
                key = history_key[kv_head]
                current_key = all_key[kv_head, -1]
                basis = bases[kv_head]
                projected_query = q @ basis
                quantized_query = production_int8_dequantize(
                    projected_query
                )
                exact_scores = (key @ q) * scaling
                current_score = (current_key @ q) * scaling
                fp32_scores = (
                    projected_keys[kv_head] @ projected_query
                ) * scaling
                int4_key_scores = (
                    quantized_keys[kv_head] @ projected_query
                ) * scaling
                production_scores = (
                    quantized_keys[kv_head] @ quantized_query
                ) * scaling
                projected_key = projected_keys[kv_head]
                quantized_key = quantized_keys[kv_head]
                query_tail_norm = torch.sqrt(
                    (
                        q.square().sum()
                        - projected_query.square().sum()
                    ).clamp_min(0.0)
                )
                key_tail_norm = torch.sqrt(
                    (
                        key.square().sum(dim=-1)
                        - projected_key.square().sum(dim=-1)
                    ).clamp_min(0.0)
                )
                pca_error_bound = query_tail_norm * key_tail_norm
                int4_key_error_bound = (
                    pca_error_bound
                    + projected_query.norm()
                    * (projected_key - quantized_key).norm(dim=-1)
                )
                production_error_bound = (
                    pca_error_bound
                    + (projected_query - quantized_query).norm()
                    * projected_key.norm(dim=-1)
                    + quantized_query.norm()
                    * (projected_key - quantized_key).norm(dim=-1)
                )
                absolute_scaling = abs(scaling)
                stage_error_bounds = {
                    "prefix_pca48_fp32": (
                        pca_error_bound * absolute_scaling
                    ),
                    "prefix_pca48_int4k": (
                        int4_key_error_bound * absolute_scaling
                    ),
                    "prefix_pca48_int4k_int8q": (
                        production_error_bound * absolute_scaling
                    ),
                }
                full_attention = torch.softmax(
                    torch.cat((exact_scores, current_score.view(1))),
                    dim=0,
                )
                stages = {
                    "prefix_pca48_fp32": fp32_scores,
                    "prefix_pca48_int4k": int4_key_scores,
                    "prefix_pca48_int4k_int8q": production_scores,
                }
                common = {
                    "model": model,
                    "topic": topic,
                    "trace_path": str(trace_path),
                    "record_index": record_index,
                    "layer": layer,
                    "query_head": query_head,
                    "kv_head": kv_head,
                    "history_tokens": history_count,
                    "rank": rank,
                }
                for method, scores in stages.items():
                    for fraction in fractions:
                        row = {
                            **common,
                            "method": method,
                            **ranking_certificate_metrics(
                                exact_scores,
                                scores,
                                full_attention,
                                fraction,
                                gamma_grid,
                                stage_error_bounds[method],
                            ),
                        }
                        if method == "prefix_pca48_int4k_int8q":
                            row.update(
                                sampled_threshold_metrics(
                                    exact_scores,
                                    scores,
                                    full_attention,
                                    fraction,
                                    quantile_sample_count,
                                    stage_error_bounds[method],
                                )
                            )
                        rows.append(row)
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
            "Measure margin-conditioned top-k and attention-mass certificates "
            "for CountCap's production score proxy."
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
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--basis_prefix_tokens", type=int, default=2048)
    parser.add_argument("--quantile_sample_count", type=int, default=256)
    parser.add_argument("--fractions", default="0.02,0.04,0.06")
    parser.add_argument(
        "--gamma_grid",
        default="0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fractions = tuple(
        sorted({float(value) for value in args.fractions.split(",") if value})
    )
    gamma_grid = tuple(
        sorted({float(value) for value in args.gamma_grid.split(",") if value})
    )
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must contain values in (0, 1]")
    if not gamma_grid or min(gamma_grid) <= 0.0:
        raise ValueError("gamma_grid must contain positive values")
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
                args.sample_stride,
                args.basis_prefix_tokens,
                args.quantile_sample_count,
                fractions,
                gamma_grid,
                device,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "margin_certificate_rows.csv", rows)
    report = {
        "config": {
            "traces": args.trace,
            "rank": args.rank,
            "sample_stride": args.sample_stride,
            "basis_prefix_tokens": args.basis_prefix_tokens,
            "quantile_sample_count": args.quantile_sample_count,
            "fractions": fractions,
            "gamma_grid": gamma_grid,
            "device": str(device),
        },
        "overall": aggregate(rows, ("method", "fraction")),
        "by_model": aggregate(rows, ("model", "method", "fraction")),
        "by_model_topic": aggregate(
            rows,
            ("model", "topic", "method", "fraction"),
        ),
        "by_model_layer": aggregate(
            rows,
            ("model", "layer", "method", "fraction"),
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
