from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def covariance_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    working = matrix.float()
    second_moment = working.transpose(0, 1) @ working
    second_moment /= float(working.shape[0])
    _, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvectors[:, -rank:].contiguous()


def quantize_dequantize_int4(values: torch.Tensor) -> torch.Tensor:
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(values.float() / scale).clamp(-7, 7)
    return codes * scale


def quantize_dequantize_positive(
    values: torch.Tensor, bits: int
) -> torch.Tensor:
    if not 1 <= bits <= 16:
        raise ValueError("bits must be in [1, 16]")
    maximum_code = (1 << bits) - 1
    scale = values.float().amax().clamp_min(1.0e-12) / maximum_code
    return torch.ceil(values.float() / scale).clamp(0, maximum_code) * scale


def projection_error_terms(
    query: torch.Tensor,
    key: torch.Tensor,
    projection: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact/approximate scores and valid one-/two-scalar error bounds.

    The two-part bound separates PCA residual error from INT4 projection error:

        |q k - (q U) (Q4(k U))| <=
            ||q_perp|| ||k_perp|| + ||q U|| ||k U - Q4(k U)||.

    The single-scalar bound stores only ||k - U Q4(k U)|| and is looser.
    """

    projected_key = key.float() @ projection.float()
    indexed_key = quantize_dequantize_int4(projected_key)
    projected_query = query.float() @ projection.float()
    approximate = (indexed_key @ projected_query) * float(scaling)
    exact = (key.float() @ query.float()) * float(scaling)

    key_energy = key.float().square().sum(dim=-1)
    projected_key_energy = projected_key.square().sum(dim=-1)
    key_residual_norm = (key_energy - projected_key_energy).clamp_min(0.0).sqrt()
    quantization_error_norm = torch.linalg.vector_norm(
        projected_key - indexed_key, dim=-1
    )

    query_energy = query.float().square().sum()
    projected_query_energy = projected_query.square().sum()
    query_residual_norm = (query_energy - projected_query_energy).clamp_min(0.0).sqrt()
    projected_query_norm = projected_query_energy.clamp_min(0.0).sqrt()
    query_norm = query_energy.clamp_min(0.0).sqrt()

    two_part_bound = (
        query_residual_norm * key_residual_norm
        + projected_query_norm * quantization_error_norm
    ) * float(scaling)
    reconstructed_error_norm = (
        key_residual_norm.square() + quantization_error_norm.square()
    ).sqrt()
    single_bound = query_norm * reconstructed_error_norm * float(scaling)
    return exact, approximate, two_part_bound, single_bound, reconstructed_error_norm


def interval_candidate_mask(
    approximate: torch.Tensor, bound: torch.Tensor, top_count: int
) -> torch.Tensor:
    """Return a certified superset of the exact top-k score positions."""

    if not 0 < top_count <= approximate.numel():
        raise ValueError("top_count must be in (0, number of tokens]")
    lower = approximate - bound
    upper = approximate + bound
    kth_lower = torch.topk(lower, k=top_count).values[-1]
    return upper >= kth_lower


def exact_rerank(
    exact: torch.Tensor, candidate_indices: torch.Tensor, top_count: int
) -> torch.Tensor:
    if candidate_indices.numel() < top_count:
        raise ValueError("candidate set must contain at least top_count positions")
    local = torch.topk(exact[candidate_indices], k=top_count).indices
    return candidate_indices[local]


def fixed_budget_residual_rescue(
    approximate: torch.Tensor,
    key_error_norm: torch.Tensor,
    top_count: int,
    rescue_count: int,
) -> torch.Tensor:
    """Reserve part of a fixed budget for keys with the largest index error."""

    if not 0 <= rescue_count <= top_count:
        raise ValueError("rescue_count must be in [0, top_count]")
    base_count = top_count - rescue_count
    selected = torch.zeros_like(approximate, dtype=torch.bool)
    if base_count:
        selected[torch.topk(approximate, k=base_count).indices] = True
    if rescue_count:
        selected[torch.topk(key_error_norm, k=rescue_count).indices] = True
    missing = top_count - int(selected.sum().item())
    if missing:
        remaining_scores = approximate.masked_fill(selected, -torch.inf)
        selected[torch.topk(remaining_scores, k=missing).indices] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def union_rescue_candidates(
    approximate: torch.Tensor,
    rescue_score: torch.Tensor,
    primary_count: int,
    total_count: int,
) -> torch.Tensor:
    """Keep every primary candidate, then fill a small residual rescue tier."""

    if approximate.shape != rescue_score.shape:
        raise ValueError("approximate and rescue_score must have the same shape")
    if not 0 < primary_count <= total_count <= approximate.numel():
        raise ValueError("candidate counts must satisfy 0 < primary <= total <= N")
    selected = torch.zeros_like(approximate, dtype=torch.bool)
    primary = torch.topk(approximate, k=primary_count).indices
    selected[primary] = True
    rescue_needed = total_count - primary_count
    if rescue_needed:
        available_rescue = rescue_score.masked_fill(selected, -torch.inf)
        selected[torch.topk(available_rescue, k=rescue_needed).indices] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def selection_metrics(
    selected: torch.Tensor,
    true_indices: torch.Tensor,
    attention: torch.Tensor,
    oracle_mass: float,
) -> dict[str, float]:
    selected_mask = torch.zeros_like(attention, dtype=torch.bool)
    selected_mask[selected] = True
    hits = int(selected_mask[true_indices].sum().item())
    selected_mass = float(attention[selected].sum().item())
    return {
        "top2_recall": hits / max(1, int(true_indices.numel())),
        "selected_attention_mass": selected_mass,
        "oracle_top2_attention_mass": oracle_mass,
        "top2_attention_mass_recall": (
            selected_mass / oracle_mass if oracle_mass > 0.0 else 0.0
        ),
    }


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


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "top2_recall",
        "selected_attention_mass",
        "oracle_top2_attention_mass",
        "top2_attention_mass_recall",
        "candidate_ratio",
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method, items in sorted(groups.items()):
        result: dict[str, Any] = {"method": method, "cases": len(items)}
        for field in metric_fields:
            stats = summarize(float(item[field]) for item in items)
            result.update({f"{field}_{name}": value for name, value in stats.items()})
        output.append(result)
    return output


def pearson(left: list[float], right: list[float]) -> float:
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((x @ y / denominator).item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe residual-certified PCA64 INT4 retrieval on real per-head Q/K traces."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--ucb_lambdas", default="0,0.01,0.025,0.05,0.1,0.25,0.5,1")
    parser.add_argument("--rescue_fractions", default="0.0025,0.005,0.01")
    parser.add_argument("--candidate_fractions", default="0.02,0.03,0.04,0.06,0.08,0.12,0.16")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if args.rank <= 0 or args.sample_stride <= 0:
        raise ValueError("rank and sample_stride must be positive")
    ucb_lambdas = parse_float_list(args.ucb_lambdas)
    rescue_fractions = [
        value
        for value in parse_float_list(args.rescue_fractions)
        if 0.0 <= value <= args.top_fraction
    ]
    candidate_fractions = [
        value
        for value in parse_float_list(args.candidate_fractions)
        if args.top_fraction <= value <= 1.0
    ]
    if not candidate_fractions:
        raise ValueError("candidate_fractions must include a value >= top_fraction")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError("trace contains no records")
    head_dim = int(records[0]["query"].shape[-1])
    if args.rank > head_dim:
        raise ValueError("rank cannot exceed head dimension")

    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    bound_violations = {"two_part": 0, "single": 0}
    bound_values = 0

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0, :]
        all_key = record["key"].to(device).float()[0]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(all_key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(all_key.shape[1]) - 1
        if query_heads % kv_heads != 0 or history_count <= 0:
            raise ValueError("invalid GQA trace shape")
        key = all_key[:, :history_count]
        top_count = max(1, math.ceil(args.top_fraction * history_count))

        projections = torch.stack(
            [
                covariance_basis(head_key[:: args.sample_stride], args.rank)
                for head_key in key
            ]
        )

        for query_head in range(query_heads):
            kv_head = query_head // groups
            head_query = query[query_head]
            head_key = key[kv_head]
            exact, approximate, two_bound, single_bound, key_error_norm = (
                projection_error_terms(
                    head_query, head_key, projections[kv_head], scaling
                )
            )
            all_scores = torch.cat(
                (
                    exact,
                    (all_key[kv_head, -1] @ head_query * scaling).view(1),
                )
            )
            attention = torch.softmax(all_scores, dim=-1)[:history_count]
            true_indices = torch.topk(exact, k=top_count).indices
            oracle_mass = float(attention[true_indices].sum().item())

            actual_error = (exact - approximate).abs()
            tolerance = 2.0e-4 + 1.0e-5 * exact.abs()
            bound_violations["two_part"] += int(
                (actual_error > two_bound + tolerance).sum().item()
            )
            bound_violations["single"] += int(
                (actual_error > single_bound + tolerance).sum().item()
            )
            bound_values += history_count

            base = {
                "topic": args.topic,
                "record_index": record_index,
                "layer": layer,
                "query_head": query_head,
                "kv_head": kv_head,
                "history_tokens": history_count,
                "top2_tokens": top_count,
            }

            baseline_selected = torch.topk(approximate, k=top_count).indices
            baseline_metrics = selection_metrics(
                baseline_selected, true_indices, attention, oracle_mass
            )
            rows.append(
                {
                    **base,
                    "method": "approx_top2",
                    "candidate_ratio": top_count / history_count,
                    **baseline_metrics,
                }
            )

            for value in ucb_lambdas:
                selected = torch.topk(
                    approximate + value * two_bound, k=top_count
                ).indices
                rows.append(
                    {
                        **base,
                        "method": f"ucb_lambda_{value:g}",
                        "candidate_ratio": top_count / history_count,
                        **selection_metrics(
                            selected, true_indices, attention, oracle_mass
                        ),
                    }
                )

            for fraction in rescue_fractions:
                rescue_count = min(top_count, math.ceil(fraction * history_count))
                selected = fixed_budget_residual_rescue(
                    approximate, key_error_norm, top_count, rescue_count
                )
                rows.append(
                    {
                        **base,
                        "method": f"residual_rescue_{fraction:g}",
                        "candidate_ratio": top_count / history_count,
                        **selection_metrics(
                            selected, true_indices, attention, oracle_mass
                        ),
                    }
                )

            strict_ratios: dict[str, float] = {}
            for name, bound in (
                ("strict_two_part", two_bound),
                ("strict_single", single_bound),
            ):
                candidate_mask = interval_candidate_mask(
                    approximate, bound, top_count
                )
                candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                selected = exact_rerank(exact, candidates, top_count)
                strict_ratios[name] = candidates.numel() / history_count
                rows.append(
                    {
                        **base,
                        "method": name,
                        "candidate_ratio": strict_ratios[name],
                        **selection_metrics(
                            selected, true_indices, attention, oracle_mass
                        ),
                    }
                )

            upper = approximate + two_bound
            query_norm = head_query.float().norm()
            single_upper = approximate + single_bound
            single_upper_u8 = approximate + (
                query_norm
                * quantize_dequantize_positive(key_error_norm, bits=8)
                * scaling
            )
            single_upper_u4 = approximate + (
                query_norm
                * quantize_dequantize_positive(key_error_norm, bits=4)
                * scaling
            )
            for fraction in candidate_fractions:
                candidate_count = max(top_count, math.ceil(fraction * history_count))
                candidate_count = min(history_count, candidate_count)
                approximate_candidates = torch.topk(
                    approximate, k=candidate_count
                ).indices
                approximate_selected = exact_rerank(
                    exact, approximate_candidates, top_count
                )
                rows.append(
                    {
                        **base,
                        "method": f"base_overfetch_approx_{fraction:g}",
                        "candidate_ratio": candidate_count / history_count,
                        **selection_metrics(
                            approximate_selected,
                            true_indices,
                            attention,
                            oracle_mass,
                        ),
                    }
                )
                candidates = torch.topk(upper, k=candidate_count).indices
                selected = exact_rerank(exact, candidates, top_count)
                rows.append(
                    {
                        **base,
                        "method": f"progressive_upper_{fraction:g}",
                        "candidate_ratio": candidate_count / history_count,
                        **selection_metrics(
                            selected, true_indices, attention, oracle_mass
                        ),
                    }
                )
                for rescue_name, rescue_score in (
                    ("error_norm", key_error_norm),
                    ("bound", two_bound),
                    ("upper", upper),
                    ("single_upper", single_upper),
                    ("single_upper_u8", single_upper_u8),
                    ("single_upper_u4", single_upper_u4),
                ):
                    union_candidates = union_rescue_candidates(
                        approximate,
                        rescue_score,
                        top_count,
                        candidate_count,
                    )
                    union_selected = exact_rerank(
                        exact, union_candidates, top_count
                    )
                    rows.append(
                        {
                            **base,
                            "method": f"base_union_{rescue_name}_{fraction:g}",
                            "candidate_ratio": (
                                union_candidates.numel() / history_count
                            ),
                            **selection_metrics(
                                union_selected,
                                true_indices,
                                attention,
                                oracle_mass,
                            ),
                        }
                    )

            query_projected = head_query @ projections[kv_head]
            q_energy = float(head_query.square().sum().item())
            q_projected_energy = float(query_projected.square().sum().item())
            risk_rows.append(
                {
                    **base,
                    "query_residual_energy_fraction": max(
                        0.0, 1.0 - q_projected_energy / max(q_energy, 1.0e-12)
                    ),
                    "baseline_top2_recall": baseline_metrics["top2_recall"],
                    "baseline_mass_recall": baseline_metrics[
                        "top2_attention_mass_recall"
                    ],
                    "two_part_bound_mean": float(two_bound.mean().item()),
                    "two_part_bound_p90": float(
                        torch.quantile(two_bound, 0.9).item()
                    ),
                    "error_to_bound_mean": float(
                        (actual_error / two_bound.clamp_min(1.0e-12)).mean().item()
                    ),
                    "error_to_bound_max": float(
                        (actual_error / two_bound.clamp_min(1.0e-12)).max().item()
                    ),
                    "strict_two_part_candidate_ratio": strict_ratios[
                        "strict_two_part"
                    ],
                    "strict_single_candidate_ratio": strict_ratios[
                        "strict_single"
                    ],
                }
            )

        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={layer} history={history_count}",
            flush=True,
        )
        del query, all_key, key, projections
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head_query.csv", rows)
    write_csv(args.output_dir / "risk_signals.csv", risk_rows)
    overall = aggregate_rows(rows)
    write_csv(args.output_dir / "summary_overall.csv", overall)

    query_risk = [float(row["query_residual_energy_fraction"]) for row in risk_rows]
    baseline_miss = [1.0 - float(row["baseline_mass_recall"]) for row in risk_rows]
    strict_size = [
        float(row["strict_two_part_candidate_ratio"]) for row in risk_rows
    ]
    index_bits = args.rank * 4 + 16
    full_kv_bits = 2 * head_dim * 16
    report = {
        "topic": args.topic,
        "trace_path": str(args.trace_path),
        "records": len(records),
        "rank": args.rank,
        "sample_stride": args.sample_stride,
        "top_fraction": args.top_fraction,
        "overall": overall,
        "bound_validation": {
            "values": bound_values,
            "two_part_violations": bound_violations["two_part"],
            "single_violations": bound_violations["single"],
        },
        "risk_correlations": {
            "query_residual_vs_baseline_mass_miss": pearson(
                query_risk, baseline_miss
            ),
            "query_residual_vs_strict_candidate_ratio": pearson(
                query_risk, strict_size
            ),
        },
        "index_storage_ratio_vs_fp16_kv": {
            "pca64_int4_plus_fp16_scale": index_bits / full_kv_bits,
            "plus_one_int8_error_norm": (index_bits + 8) / full_kv_bits,
            "plus_two_int8_error_norms": (index_bits + 16) / full_kv_bits,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
