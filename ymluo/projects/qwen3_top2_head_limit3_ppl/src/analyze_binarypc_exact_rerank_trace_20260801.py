"""Held-out exact-QK audit for BinaryPC overfetch and reranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binarypc_repo", type=Path, required=True)
    parser.add_argument("--projection_path", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--overfetch_factors", default="1,1.25,1.5,2,3,4")
    parser.add_argument("--max_steps_per_layer", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def selected_count(history_tokens: int) -> int:
    return min(
        history_tokens,
        1280,
        max(256, math.ceil(0.06 * history_tokens)),
    )


def parse_factors(spec: str) -> list[float]:
    values = sorted({float(item) for item in spec.split(",") if item.strip()})
    if not values or any(value < 1.0 for value in values):
        raise ValueError("overfetch factors must be at least one")
    return values


def evaluate_selection(
    exact_scaled: torch.Tensor,
    candidate_kv_indices: torch.Tensor,
    target_count: int,
) -> dict[str, float]:
    group_count = exact_scaled.shape[2]
    candidates = candidate_kv_indices.unsqueeze(2).expand(
        -1, -1, group_count, -1
    )
    candidate_exact = torch.gather(exact_scaled, -1, candidates)
    selected_scores, selected_local = torch.topk(
        candidate_exact,
        k=target_count,
        dim=-1,
        sorted=False,
    )
    selected = torch.gather(candidates, -1, selected_local)
    exact_top = torch.topk(
        exact_scaled, k=target_count, dim=-1, sorted=False
    ).indices
    exact_mask = torch.zeros_like(exact_scaled, dtype=torch.bool)
    exact_mask.scatter_(-1, exact_top, True)
    candidate_recall = (
        exact_mask.gather(-1, candidates).float().sum(dim=-1)
        / target_count
    )
    selected_recall = exact_mask.gather(-1, selected).float().mean(dim=-1)
    probabilities = torch.softmax(exact_scaled.float(), dim=-1)
    selected_mass = probabilities.gather(-1, selected).sum(dim=-1)
    oracle_mass = probabilities.gather(-1, exact_top).sum(dim=-1)
    selected_top_token = torch.gather(
        selected,
        -1,
        selected_scores.argmax(dim=-1, keepdim=True),
    ).squeeze(-1)
    top1 = (exact_scaled.argmax(dim=-1) == selected_top_token).float()
    return {
        "candidate_exact_topk_recall": float(candidate_recall.mean().item()),
        "exact_topk_recall_after_rerank": float(selected_recall.mean().item()),
        "attention_mass_after_rerank": float(selected_mass.mean().item()),
        "oracle_topk_mass": float(oracle_mass.mean().item()),
        "top1_recall_after_rerank": float(top1.mean().item()),
        "query_heads": int(selected_recall.numel()),
    }


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(int(row["query_heads"]) for row in rows)
    return sum(
        float(row[field]) * int(row["query_heads"]) for row in rows
    ) / total


def reference_key_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the newest record that materializes the shared historical Key."""
    available = [
        record
        for record in records
        if isinstance(record.get("key"), torch.Tensor)
    ]
    if not available:
        raise ValueError("trace layer has no materialized historical Key")
    return max(available, key=lambda record: int(record["step"]))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.binarypc_repo))
    from bpc.bpc.bpc import (  # noqa: PLC0415
        _load_cuda_kernel,
        binary_project_cuda,
        compute_errors_cuda,
        compute_hashscores_cuda,
    )

    if not _load_cuda_kernel(0):
        raise RuntimeError("official BinaryPC CUDA extension failed to load")
    projections = torch.load(
        args.projection_path, map_location="cpu", weights_only=False
    )
    if not isinstance(projections, dict):
        raise TypeError("BinaryPC projection checkpoint must be a dictionary")
    factors = parse_factors(args.overfetch_factors)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []

    for trace_spec in args.trace:
        topic, path_text = trace_spec.split("=", 1)
        trace = torch.load(path_text, map_location="cpu", weights_only=False)
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in trace["records"]:
            by_layer[int(record["layer"])].append(record)
        for layer, records in sorted(by_layer.items()):
            records.sort(key=lambda item: int(item["step"]))
            selected_records = records[-args.max_steps_per_layer :]
            projection = projections[layer].to(
                device=device, dtype=torch.bfloat16
            )
            key_record = reference_key_record(records)
            key = key_record["key"].to(
                device=device, dtype=torch.bfloat16
            )
            history_tokens = int(key.shape[-2])
            if history_tokens % 256:
                raise ValueError("BinaryPC trace length must be a multiple of 256")
            target_count = selected_count(history_tokens)
            error_count = max(1, int(0.1 * target_count))
            hashcodes = binary_project_cuda(key, projection)
            errors = compute_errors_cuda(key, hashcodes, projection)
            error_indices = torch.topk(
                errors, k=error_count, dim=-1, sorted=False
            ).indices
            projection_t = projection.transpose(-1, -2).contiguous()

            for record in selected_records:
                query = record["query"].to(
                    device=device, dtype=torch.bfloat16
                )[..., 0, :]
                grouped_query = query.reshape(1, 8, 4, 128)
                probe = (grouped_query @ projection_t).contiguous()
                hash_scores = compute_hashscores_cuda(
                    probe, hashcodes, history_tokens, history_tokens
                )
                hash_scores.scatter_(2, error_indices, 10000)
                exact_scaled = torch.einsum(
                    "bhgd,bhnd->bhgn",
                    grouped_query.float(),
                    key.float(),
                ) * float(record["scaling"])

                for factor in factors:
                    coarse_count = min(
                        history_tokens,
                        max(target_count, math.ceil(factor * target_count)),
                    )
                    candidates = torch.topk(
                        hash_scores,
                        k=coarse_count,
                        dim=-1,
                        sorted=False,
                    ).indices
                    metrics = evaluate_selection(
                        exact_scaled, candidates, target_count
                    )
                    rows.append(
                        {
                            "topic": topic,
                            "layer": layer,
                            "step": int(record["step"]),
                            "history_tokens": history_tokens,
                            "target_tokens_per_query_head": target_count,
                            "overfetch_factor": factor,
                            "coarse_tokens_per_kv_head": coarse_count,
                            "logical_index_bits_per_token_per_kv_head": 64,
                            **metrics,
                        }
                    )
            del key, projection, hashcodes, errors, error_indices
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"topic": topic, "layer": layer, "rows": len(rows)}
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates: list[dict[str, Any]] = []
    for factor in factors:
        current = [row for row in rows if row["overfetch_factor"] == factor]
        aggregates.append(
            {
                "overfetch_factor": factor,
                "conditions": len(current),
                "query_heads": sum(int(row["query_heads"]) for row in current),
                "coarse_tokens_per_kv_head": current[0][
                    "coarse_tokens_per_kv_head"
                ],
                **{
                    field: weighted_mean(current, field)
                    for field in (
                        "candidate_exact_topk_recall",
                        "exact_topk_recall_after_rerank",
                        "attention_mass_after_rerank",
                        "oracle_topk_mass",
                        "top1_recall_after_rerank",
                    )
                },
                "worst_condition_topk_recall": min(
                    float(row["exact_topk_recall_after_rerank"])
                    for row in current
                ),
                "worst_condition_attention_mass": min(
                    float(row["attention_mass_after_rerank"])
                    for row in current
                ),
            }
        )
    result = {
        "schema": "binarypc_exact_rerank_trace_audit_v1",
        "contract": {
            "model": "Llama-3.1-8B-Instruct",
            "trace_length": 32768,
            "topics": [item.split("=", 1)[0] for item in args.trace],
            "heldout_steps_per_layer": args.max_steps_per_layer,
            "key_snapshot_policy": (
                "newest materialized per-layer Key snapshot; held-out Queries "
                "may omit duplicate Keys"
            ),
            "coarse_selector": "released BinaryPC-64 plus 10% error rescue",
            "refinement": "exact QK within GQA-shared overfetch candidates",
            "full_fallback": False,
            "router": False,
        },
        "aggregate": aggregates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
