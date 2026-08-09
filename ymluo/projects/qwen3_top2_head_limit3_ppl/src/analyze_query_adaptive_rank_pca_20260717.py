from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_residual_certified_pca_20260717 import (
    parse_float_list,
    quantize_dequantize_int4,
    selection_metrics,
    summarize,
)
from analyze_svd_index_recall_20260716 import (
    quantize_dequantize_int2_ternary,
    quantize_dequantize_int2_uniform4,
)


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(part) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one integer")
    return values


def covariance_eigensystem(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    working = matrix.float()
    second_moment = working.transpose(0, 1) @ working
    second_moment /= float(working.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    return (
        eigenvectors.flip(dims=(1,)).contiguous(),
        eigenvalues.flip(dims=(0,)).clamp_min(0.0).contiguous(),
    )


def covariance_eigenbasis(matrix: torch.Tensor) -> torch.Tensor:
    basis, _ = covariance_eigensystem(matrix)
    return basis


def quantize_dequantize_binary_sign(matrix: torch.Tensor) -> torch.Tensor:
    """One bit per value with a shared per-dimension reconstruction scale."""

    working = matrix.float()
    scale = working.abs().mean(dim=0, keepdim=True)
    sign = torch.where(working >= 0, 1.0, -1.0)
    return sign * scale


def hierarchical_projected_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    basis_descending: torch.Tensor,
    ranks: list[int],
    base_rank: int,
    group_size: int,
    scaling: float,
    residual_precision: str = "int4",
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Build cumulative scores from a base PCA index plus residual rank groups."""

    if ranks[0] != base_rank:
        raise ValueError("the first rank must equal base_rank")
    if any(rank > basis_descending.shape[1] for rank in ranks):
        raise ValueError("rank exceeds basis dimension")
    if any(rank < base_rank or (rank - base_rank) % group_size for rank in ranks):
        raise ValueError("ranks must align to residual group boundaries")

    maximum_rank = ranks[-1]
    projected_key = key.float() @ basis_descending[:, :maximum_rank]
    projected_query = query.float() @ basis_descending[:, :maximum_rank]
    query_energy = query.float().square().sum().clamp_min(1.0e-12)
    scores: dict[int, torch.Tensor] = {}
    coverage: dict[int, float] = {}
    cumulative = torch.zeros(key.shape[0], dtype=torch.float32, device=key.device)

    starts = [0] + list(range(base_rank, maximum_rank, group_size))
    ends = [base_rank] + [
        min(maximum_rank, start + group_size) for start in starts[1:]
    ]
    rank_set = set(ranks)
    for start, end in zip(starts, ends):
        group = projected_key[:, start:end]
        if start == 0 or residual_precision == "int4":
            indexed = quantize_dequantize_int4(group)
        elif residual_precision == "int2_uniform4":
            indexed = quantize_dequantize_int2_uniform4(group)
        elif residual_precision == "int2_ternary":
            indexed = quantize_dequantize_int2_ternary(group)
        elif residual_precision == "binary_sign":
            indexed = quantize_dequantize_binary_sign(group)
        else:
            raise ValueError("unsupported residual precision")
        cumulative = cumulative + indexed @ projected_query[start:end]
        if end in rank_set:
            scores[end] = cumulative * float(scaling)
            coverage[end] = float(
                (
                    projected_query[:end].square().sum() / query_energy
                ).clamp(0.0, 1.0).item()
            )
    if set(scores) != rank_set:
        raise RuntimeError("failed to construct every requested rank")
    return scores, coverage


def choose_rank(
    coverage: dict[int, float], threshold: float, maximum_rank: int
) -> int:
    allowed = [rank for rank in sorted(coverage) if rank <= maximum_rank]
    if not allowed:
        raise ValueError("maximum_rank excludes every available rank")
    for rank in allowed:
        if coverage[rank] >= threshold:
            return rank
    return allowed[-1]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "top2_recall",
        "selected_attention_mass",
        "oracle_top2_attention_mass",
        "top2_attention_mass_recall",
        "selected_rank",
        "query_energy_coverage",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_storage_ratio(
    rank: int,
    head_dim: int,
    base_rank: int,
    group_size: int,
    residual_precision: str,
) -> float:
    residual_groups = max(0, math.ceil((rank - base_rank) / group_size))
    residual_bits = {
        "int4": 4,
        "int2_uniform4": 2,
        "int2_ternary": 2,
        "binary_sign": 1,
    }[residual_precision]
    per_token_residual_scale_bits = 0 if residual_precision == "binary_sign" else 16
    index_bits = (
        base_rank * 4
        + 16
        + (rank - base_rank) * residual_bits
        + residual_groups * per_token_residual_scale_bits
    )
    return index_bits / (2 * head_dim * 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate training-free query-energy adaptive PCA rank on real Q/K traces."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--ranks", default="64,80,96,112,128")
    parser.add_argument("--base_rank", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--energy_thresholds", default="0.7,0.75,0.8,0.85,0.9,0.95")
    parser.add_argument("--rank_caps", default="80,96,112,128")
    parser.add_argument(
        "--residual_precision",
        choices=["int4", "int2_uniform4", "int2_ternary", "binary_sign"],
        default="int4",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ranks = parse_int_list(args.ranks)
    thresholds = parse_float_list(args.energy_thresholds)
    rank_caps = parse_int_list(args.rank_caps)
    if ranks[0] != args.base_rank:
        raise ValueError("ranks must start with base_rank")
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if args.sample_stride <= 0 or args.group_size <= 0:
        raise ValueError("sample_stride and group_size must be positive")
    if any(not 0.0 < threshold <= 1.0 for threshold in thresholds):
        raise ValueError("energy thresholds must be in (0, 1]")
    if any(cap not in ranks for cap in rank_caps):
        raise ValueError("every rank cap must be present in ranks")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError("trace contains no records")
    head_dim = int(records[0]["query"].shape[-1])
    if ranks[-1] > head_dim:
        raise ValueError("rank cannot exceed head dimension")

    rows: list[dict[str, Any]] = []
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
        eigensystems = [
            covariance_eigensystem(head_key[:: args.sample_stride])
            for head_key in key
        ]
        eigenbases = torch.stack([item[0] for item in eigensystems])
        eigenvalues = torch.stack([item[1] for item in eigensystems])

        for query_head in range(query_heads):
            kv_head = query_head // groups
            head_query = query[query_head]
            head_key = key[kv_head]
            exact = (head_key @ head_query) * scaling
            current_score = (all_key[kv_head, -1] @ head_query * scaling).view(1)
            attention = torch.softmax(torch.cat((exact, current_score)), dim=-1)[
                :history_count
            ]
            true_indices = torch.topk(exact, k=top_count).indices
            oracle_mass = float(attention[true_indices].sum().item())
            scores, coverage = hierarchical_projected_scores(
                head_query,
                head_key,
                eigenbases[kv_head],
                ranks,
                args.base_rank,
                args.group_size,
                scaling,
                args.residual_precision,
            )
            base_score = scores[args.base_rank]
            query_in_basis = head_query.float() @ eigenbases[kv_head]
            tail_score_std = float(
                (
                    query_in_basis[args.base_rank :].square()
                    * eigenvalues[kv_head, args.base_rank :]
                )
                .sum()
                .sqrt()
                .mul(scaling)
                .item()
            )
            ranked_base = torch.topk(
                base_score, k=min(history_count, 4 * top_count), sorted=True
            ).values
            boundary = ranked_base[top_count - 1]
            proxy_attention = torch.softmax(
                torch.cat((base_score, current_score)), dim=-1
            )[:history_count]

            def normalized_margin(multiplier: int) -> float:
                index = min(history_count, multiplier * top_count) - 1
                margin = float((boundary - ranked_base[index]).item())
                return margin / max(tail_score_std, 1.0e-12)

            base = {
                "topic": args.topic,
                "record_index": record_index,
                "layer": layer,
                "query_head": query_head,
                "kv_head": kv_head,
                "history_tokens": history_count,
                "top2_tokens": top_count,
                "candidate_ratio": top_count / history_count,
                "base_proxy_top2_mass": float(
                    proxy_attention[torch.topk(base_score, k=top_count).indices]
                    .sum()
                    .item()
                ),
                "base_tail_score_std": tail_score_std,
                "base_margin_to_2k_sigma": normalized_margin(2),
                "base_margin_to_4k_sigma": normalized_margin(4),
                "base_boundary_band_1sigma_ratio": float(
                    (
                        (base_score - boundary).abs()
                        <= max(tail_score_std, 1.0e-12)
                    )
                    .float()
                    .mean()
                    .item()
                ),
            }

            metrics_by_rank: dict[int, dict[str, float]] = {}
            for rank in ranks:
                selected = torch.topk(scores[rank], k=top_count).indices
                metrics = selection_metrics(
                    selected, true_indices, attention, oracle_mass
                )
                metrics_by_rank[rank] = metrics
                rows.append(
                    {
                        **base,
                        "method": f"fixed_rank_{rank}",
                        "selected_rank": rank,
                        "query_energy_coverage": coverage[rank],
                        **metrics,
                    }
                )

            for cap in rank_caps:
                for threshold in thresholds:
                    rank = choose_rank(coverage, threshold, cap)
                    rows.append(
                        {
                            **base,
                            "method": f"adaptive_energy_{threshold:g}_cap{cap}",
                            "selected_rank": rank,
                            "query_energy_coverage": coverage[rank],
                            **metrics_by_rank[rank],
                        }
                    )

        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={layer} history={history_count}",
            flush=True,
        )
        del query, all_key, key, eigenbases, eigenvalues
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head_query.csv", rows)
    overall = aggregate_rows(rows)
    write_csv(args.output_dir / "summary_overall.csv", overall)
    report = {
        "topic": args.topic,
        "trace_path": str(args.trace_path),
        "records": len(records),
        "ranks": ranks,
        "energy_thresholds": thresholds,
        "rank_caps": rank_caps,
        "residual_precision": args.residual_precision,
        "top_fraction": args.top_fraction,
        "overall": overall,
        "maximum_index_storage_ratio_vs_fp16_kv": {
            str(rank): index_storage_ratio(
                rank,
                head_dim,
                args.base_rank,
                args.group_size,
                args.residual_precision,
            )
            for rank in ranks
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
