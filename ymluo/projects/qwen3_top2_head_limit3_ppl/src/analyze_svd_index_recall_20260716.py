from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_positive_ints(value: str, maximum: int) -> list[int]:
    values = sorted({int(part) for part in value.split(",") if part.strip()})
    values = [value for value in values if 0 < value <= maximum]
    if not values:
        raise ValueError("ranks must contain a value in (0, head_dim]")
    return values


def quantize_dequantize_int4(values: torch.Tensor) -> torch.Tensor:
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    codes = torch.round(values.float() / scale).clamp(-7, 7)
    return codes * scale


def quantize_dequantize_int2_ternary(values: torch.Tensor) -> torch.Tensor:
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    codes = torch.round(values.float() / scale).clamp(-1, 1)
    return codes * scale


def quantize_dequantize_int2_uniform4(values: torch.Tensor) -> torch.Tensor:
    scale = values.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    normalized = (values.float() / scale).clamp(-1, 1)
    codes = torch.round((normalized + 1.0) * 1.5).clamp(0, 3)
    return (codes / 1.5 - 1.0) * scale


def quantize_dequantize_int2_grouped(
    values: torch.Tensor, group_size: int
) -> torch.Tensor:
    if values.shape[-1] % group_size != 0:
        raise ValueError("INT2 group size must divide the projection dimension")
    grouped = values.reshape(*values.shape[:-1], -1, group_size)
    quantized = quantize_dequantize_int2_uniform4(grouped)
    return quantized.reshape_as(values)


def covariance_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    working = matrix.float()
    second_moment = working.transpose(0, 1) @ working
    second_moment /= float(working.shape[0])
    _, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvectors[:, -rank:].contiguous()


def svd_basis(matrix: torch.Tensor, rank: int, center: bool) -> torch.Tensor:
    working = matrix.float()
    if center:
        working = working - working.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(working, full_matrices=False)
    return vh[:rank].transpose(0, 1).contiguous()


def subspace_diagnostics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    singular_values = torch.linalg.svdvals(left.transpose(0, 1) @ right).clamp(0, 1)
    left_projector = left @ left.transpose(0, 1)
    right_projector = right @ right.transpose(0, 1)
    return {
        "principal_cos_mean": float(singular_values.mean().item()),
        "principal_cos_min": float(singular_values.min().item()),
        "projector_max_abs_error": float(
            (left_projector - right_projector).abs().max().item()
        ),
    }


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "minimum": float(tensor.min().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict[str, Any]], group_fields: list[str]
) -> list[dict[str, Any]]:
    metric_fields = (
        "top2_recall",
        "selected_attention_mass",
        "oracle_top2_attention_mass",
        "top2_attention_mass_recall",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        result: dict[str, Any] = dict(zip(group_fields, key))
        result["cases"] = len(items)
        for field in metric_fields:
            stats = summarize([float(item[field]) for item in items])
            result.update({f"{field}_{name}": value for name, value in stats.items()})
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current sampled PCA against sampled/full, centered/uncentered "
            "SVD using true RoPE-aware per-query-head top-2% attention."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--ranks", default="32,48,64")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    if args.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError("trace contains no records")

    head_dim = int(records[0]["query"].shape[-1])
    ranks = parse_positive_ints(args.ranks, head_dim)
    max_rank = max(ranks)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        query = record["query"].to(device).float()[0, :, 0, :]
        all_key = record["key"].to(device).float()[0]
        scaling = float(record["scaling"])
        query_heads = int(query.shape[0])
        kv_heads = int(all_key.shape[0])
        groups = query_heads // kv_heads
        history_count = int(all_key.shape[1]) - 1
        if groups <= 0 or query_heads % kv_heads != 0 or history_count <= 0:
            raise ValueError("invalid GQA trace shape")
        key = all_key[:, :history_count]
        true_count = max(1, math.ceil(args.top_fraction * history_count))

        exact_scores = torch.empty(
            (query_heads, history_count), dtype=torch.float32, device=device
        )
        history_attention = torch.empty_like(exact_scores)
        true_indices = torch.empty(
            (query_heads, true_count), dtype=torch.long, device=device
        )
        oracle_mass = torch.empty(query_heads, dtype=torch.float32, device=device)
        for query_head in range(query_heads):
            kv_head = query_head // groups
            all_scores = torch.matmul(all_key[kv_head], query[query_head]) * scaling
            exact_scores[query_head] = all_scores[:history_count]
            history_attention[query_head] = torch.softmax(all_scores, dim=-1)[
                :history_count
            ]
            true_indices[query_head] = torch.topk(
                exact_scores[query_head], k=true_count
            ).indices
            oracle_mass[query_head] = history_attention[
                query_head, true_indices[query_head]
            ].sum()

        bases: dict[str, list[torch.Tensor]] = defaultdict(list)
        for kv_head in range(kv_heads):
            head_key = key[kv_head]
            sampled = head_key[:: args.sample_stride]
            current = covariance_basis(sampled, max_rank)
            sampled_svd = svd_basis(sampled, max_rank, center=False)
            bases["current_sampled_pca"].append(current)
            bases["sampled_svd_uncentered"].append(sampled_svd)
            bases["sampled_svd_centered"].append(
                svd_basis(sampled, max_rank, center=True)
            )
            bases["full_svd_uncentered"].append(
                svd_basis(head_key, max_rank, center=False)
            )
            bases["full_svd_centered"].append(
                svd_basis(head_key, max_rank, center=True)
            )
            for rank in ranks:
                diagnostics = subspace_diagnostics(
                    current[:, -rank:], sampled_svd[:, :rank]
                )
                basis_rows.append(
                    {
                        "topic": args.topic,
                        "record_index": record_index,
                        "layer": layer,
                        "kv_head": kv_head,
                        "rank": rank,
                        **diagnostics,
                    }
                )

        for scheme, head_bases in bases.items():
            for rank in ranks:
                if scheme == "current_sampled_pca":
                    projection = torch.stack(
                        [basis[:, -rank:] for basis in head_bases]
                    )
                else:
                    projection = torch.stack(
                        [basis[:, :rank] for basis in head_bases]
                    )
                projected_key = torch.einsum("hnd,hdr->hnr", key, projection)
                grouped_query = query.reshape(kv_heads, groups, head_dim)
                projected_query = torch.einsum(
                    "hgd,hdr->hgr", grouped_query, projection
                )
                for precision in (
                    "fp16",
                    "int4",
                    "int2_ternary",
                    "int2_uniform4",
                    "int2_uniform4_g16",
                    "int2_uniform4_g8",
                ):
                    if precision == "fp16":
                        indexed_key = projected_key
                    elif precision == "int4":
                        indexed_key = quantize_dequantize_int4(projected_key)
                    elif precision == "int2_ternary":
                        indexed_key = quantize_dequantize_int2_ternary(
                            projected_key
                        )
                    elif precision == "int2_uniform4":
                        indexed_key = quantize_dequantize_int2_uniform4(
                            projected_key
                        )
                    elif precision == "int2_uniform4_g16":
                        indexed_key = quantize_dequantize_int2_grouped(
                            projected_key, 16
                        )
                    else:
                        indexed_key = quantize_dequantize_int2_grouped(
                            projected_key, 8
                        )
                    approximate_scores = torch.einsum(
                        "hnr,hgr->hgn", indexed_key, projected_query
                    ).reshape(query_heads, history_count)
                    selected = torch.topk(
                        approximate_scores, k=true_count, dim=-1
                    ).indices
                    for query_head in range(query_heads):
                        selected_mask = torch.zeros(
                            history_count, dtype=torch.bool, device=device
                        )
                        selected_mask[selected[query_head]] = True
                        hits = int(
                            selected_mask[true_indices[query_head]].sum().item()
                        )
                        selected_mass = float(
                            history_attention[query_head, selected[query_head]]
                            .sum()
                            .item()
                        )
                        oracle = float(oracle_mass[query_head].item())
                        rows.append(
                            {
                                "topic": args.topic,
                                "record_index": record_index,
                                "layer": layer,
                                "query_head": query_head,
                                "kv_head": query_head // groups,
                                "scheme": scheme,
                                "rank": rank,
                                "precision": precision,
                                "history_tokens": history_count,
                                "top2_tokens": true_count,
                                "top2_recall": hits / true_count,
                                "selected_attention_mass": selected_mass,
                                "oracle_top2_attention_mass": oracle,
                                "top2_attention_mass_recall": (
                                    selected_mass / oracle if oracle > 0.0 else 0.0
                                ),
                            }
                        )
                del projected_key, projected_query
        print(
            f"topic={args.topic} record={record_index + 1}/{len(records)} "
            f"layer={layer} history={history_count}",
            flush=True,
        )
        del query, all_key, key, exact_scores, history_attention, true_indices
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_fields = list(rows[0])
    write_csv(args.output_dir / "per_head_query.csv", rows, row_fields)
    basis_fields = list(basis_rows[0])
    write_csv(args.output_dir / "basis_equivalence.csv", basis_rows, basis_fields)
    overall = aggregate_rows(rows, ["scheme", "rank", "precision"])
    by_layer_head = aggregate_rows(
        rows, ["layer", "query_head", "scheme", "rank", "precision"]
    )
    write_csv(args.output_dir / "summary_overall.csv", overall, list(overall[0]))
    write_csv(
        args.output_dir / "summary_by_layer_head.csv",
        by_layer_head,
        list(by_layer_head[0]),
    )
    report = {
        "topic": args.topic,
        "trace_path": str(args.trace_path),
        "records": len(records),
        "top_fraction": args.top_fraction,
        "sample_stride": args.sample_stride,
        "ranks": ranks,
        "overall": overall,
        "sampled_pca_svd_equivalence": {
            field: summarize([float(row[field]) for row in basis_rows])
            for field in (
                "principal_cos_mean",
                "principal_cos_min",
                "projector_max_abs_error",
            )
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
