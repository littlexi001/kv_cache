#!/usr/bin/env python
"""Replay real QK traces to test versioned QKSieve coordinates under drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_automatic_spectral_rate_allocation_20260727 import (
    FULL_KV_BITS,
    GROUP_COUNT,
    GROUP_SIZE,
    ZERO_BIT_LEVELS,
    allocate_bits,
    distortion_table,
    quantize_band,
    reconstruct,
)
from analyze_qk_balanced_spectral_rate_20260727 import qk_balanced_factors


METHODS = (
    "frozen_epoch1",
    "frozen_epoch1_reallocated",
    "global_rebuild",
    "versioned_two_epoch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch1_trace", type=Path, required=True)
    parser.add_argument("--epoch2_trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--evaluation_steps", type=int, default=4)
    parser.add_argument("--query_shrinkage", type=float, default=0.75)
    parser.add_argument("--rate_units", type=int, default=15)
    parser.add_argument("--topk", type=int, default=1280)
    return parser.parse_args()


def records_by_layer(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in payload.get("records", []):
        output[int(record["layer"])].append(record)
    for records in output.values():
        records.sort(key=lambda item: int(item["step"]))
    if not output:
        raise ValueError(f"{path} contains no trace records")
    return dict(output)


def layer_key(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    raw = next((item.get("key") for item in records if item.get("key") is not None), None)
    if raw is None:
        raise ValueError("layer trace has no Key tensor")
    key = raw.to(device=device, dtype=torch.float32)[0]
    return key[:, :-1].contiguous()


def layer_queries(
    records: list[dict[str, Any]],
    device: torch.device,
    start: int,
    count: int,
) -> torch.Tensor:
    selected = records[start : start + count]
    if len(selected) != count:
        raise ValueError("trace has too few Query steps")
    return torch.stack(
        [item["query"].to(device=device, dtype=torch.float32)[0, :, 0] for item in selected],
        dim=0,
    )


def fit_index(
    key: torch.Tensor,
    calibration_queries: torch.Tensor,
    *,
    sample_stride: int,
    query_shrinkage: float,
    rate_units: int,
) -> dict[str, Any]:
    sampled_key = key[::sample_stride]
    query_factor, key_factor, singular_values = qk_balanced_factors(
        sampled_key,
        calibration_queries,
        query_shrinkage,
    )
    sampled_coefficients = sampled_key @ key_factor
    key_costs, _ = distortion_table(
        sampled_coefficients,
        calibration_queries @ query_factor,
        ZERO_BIT_LEVELS,
    )
    allocation = allocate_bits(
        key_costs,
        rate_units,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    return {
        "query_factor": query_factor,
        "key_factor": key_factor,
        "allocation": tuple(int(value) for value in allocation),
        "singular_values": singular_values,
    }


def reallocate_index(
    state: dict[str, Any],
    keys: list[torch.Tensor],
    calibration_queries: torch.Tensor,
    *,
    sample_stride: int,
    rate_units: int,
) -> dict[str, Any]:
    sampled = torch.cat([key[::sample_stride] for key in keys], dim=0)
    coefficients = sampled @ state["key_factor"]
    key_costs, _ = distortion_table(
        coefficients,
        calibration_queries @ state["query_factor"],
        ZERO_BIT_LEVELS,
    )
    output = dict(state)
    output["allocation"] = allocate_bits(
        key_costs,
        rate_units,
        ZERO_BIT_LEVELS,
        include_scale_metadata=True,
    )
    return output


def reconstruct_keys(key: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    coefficients = key @ state["key_factor"]
    bands = []
    for band in range(GROUP_COUNT):
        start = band * GROUP_SIZE
        stop = start + GROUP_SIZE
        bits = int(state["allocation"][band])
        bands.append({bits: quantize_band(coefficients[:, start:stop], bits)})
    return reconstruct(bands, state["allocation"])


def proxy_scores(
    queries: torch.Tensor,
    reconstructed_segments: list[torch.Tensor],
    states: list[dict[str, Any]],
    scaling: float,
) -> torch.Tensor:
    parts = []
    for reconstructed, state in zip(reconstructed_segments, states, strict=True):
        projected_query = queries @ state["query_factor"]
        parts.append(projected_query @ reconstructed.transpose(0, 1))
    return torch.cat(parts, dim=-1) * scaling


def selection_rows(
    exact_scores: torch.Tensor,
    method_scores: dict[str, torch.Tensor],
    topk: int,
) -> list[dict[str, float | str]]:
    count = min(topk, exact_scores.shape[-1])
    exact_indices = torch.topk(exact_scores, count, dim=-1, sorted=False).indices
    exact_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    exact_mask.scatter_(1, exact_indices, True)
    probabilities = torch.softmax(exact_scores.float(), dim=-1)
    oracle_mass = probabilities.gather(1, exact_indices).sum(dim=-1)
    rows: list[dict[str, float | str]] = []
    for method, scores in method_scores.items():
        indices = torch.topk(scores, count, dim=-1, sorted=False).indices
        recall = exact_mask.gather(1, indices).float().mean(dim=-1)
        mass = probabilities.gather(1, indices).sum(dim=-1)
        ratio = mass / oracle_mass.clamp_min(1.0e-12)
        for query_index in range(exact_scores.shape[0]):
            rows.append(
                {
                    "method": method,
                    "query_index": query_index,
                    "topk_recall": float(recall[query_index].item()),
                    "selected_attention_mass": float(mass[query_index].item()),
                    "oracle_attention_mass": float(oracle_mass[query_index].item()),
                    "selected_to_oracle_mass": float(ratio[query_index].item()),
                }
            )
    return rows


def covariance_drift(first: torch.Tensor, second: torch.Tensor) -> float:
    first_cov = first.transpose(0, 1) @ first / max(1, first.shape[0])
    second_cov = second.transpose(0, 1) @ second / max(1, second.shape[0])
    return float(
        ((second_cov - first_cov).norm() / first_cov.norm().clamp_min(1.0e-12)).item()
    )


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "minimum": float(tensor.min().item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    first = records_by_layer(args.epoch1_trace)
    second = records_by_layer(args.epoch2_trace)
    layers = sorted(set(first) & set(second))
    if not layers:
        raise ValueError("traces have no common layers")
    detail: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    combined_history_tokens: int | None = None

    for layer in layers:
        key1 = layer_key(first[layer], device)
        key2 = layer_key(second[layer], device)
        if combined_history_tokens is None:
            combined_history_tokens = int(key1.shape[1] + key2.shape[1])
        qcal1 = layer_queries(first[layer], device, 0, args.calibration_steps)
        qcal2 = layer_queries(second[layer], device, 0, args.calibration_steps)
        qeval1 = layer_queries(
            first[layer], device, args.calibration_steps, args.evaluation_steps
        )
        qeval2 = layer_queries(
            second[layer], device, args.calibration_steps, args.evaluation_steps
        )
        scaling = float(first[layer][0]["scaling"])
        kv_heads = int(key1.shape[0])
        groups = int(qcal1.shape[1]) // kv_heads

        for kv_head in range(kv_heads):
            head_key1 = key1[kv_head]
            head_key2 = key2[kv_head]
            head_qcal1 = qcal1[:, kv_head * groups : (kv_head + 1) * groups].reshape(-1, 128)
            head_qcal2 = qcal2[:, kv_head * groups : (kv_head + 1) * groups].reshape(-1, 128)
            head_qeval1 = qeval1[:, kv_head * groups : (kv_head + 1) * groups].reshape(-1, 128)
            head_qeval2 = qeval2[:, kv_head * groups : (kv_head + 1) * groups].reshape(-1, 128)
            evaluation_queries = torch.cat([head_qeval1, head_qeval2], dim=0)
            combined_calibration = torch.cat([head_qcal1, head_qcal2], dim=0)

            state1 = fit_index(
                head_key1,
                head_qcal1,
                sample_stride=args.sample_stride,
                query_shrinkage=args.query_shrinkage,
                rate_units=args.rate_units,
            )
            state2 = fit_index(
                head_key2,
                head_qcal2,
                sample_stride=args.sample_stride,
                query_shrinkage=args.query_shrinkage,
                rate_units=args.rate_units,
            )
            global_state = fit_index(
                torch.cat([head_key1, head_key2], dim=0),
                combined_calibration,
                sample_stride=args.sample_stride,
                query_shrinkage=args.query_shrinkage,
                rate_units=args.rate_units,
            )
            reallocated = reallocate_index(
                state1,
                [head_key1, head_key2],
                combined_calibration,
                sample_stride=args.sample_stride,
                rate_units=args.rate_units,
            )
            cross_drift = covariance_drift(
                head_key1[:: args.sample_stride],
                head_key2[:: args.sample_stride],
            )
            midpoint1 = int(head_key1.shape[0] // 2)
            midpoint2 = int(head_key2.shape[0] // 2)
            within_epoch1_drift = covariance_drift(
                head_key1[:midpoint1: args.sample_stride],
                head_key1[midpoint1:: args.sample_stride],
            )
            within_epoch2_drift = covariance_drift(
                head_key2[:midpoint2: args.sample_stride],
                head_key2[midpoint2:: args.sample_stride],
            )

            reconstructed1_state1 = reconstruct_keys(head_key1, state1)
            reconstructed2_state1 = reconstruct_keys(head_key2, state1)
            reconstructed1_reallocated = reconstruct_keys(head_key1, reallocated)
            reconstructed2_reallocated = reconstruct_keys(head_key2, reallocated)
            reconstructed1_global = reconstruct_keys(head_key1, global_state)
            reconstructed2_global = reconstruct_keys(head_key2, global_state)
            reconstructed1_epoch = reconstructed1_state1
            reconstructed2_epoch = reconstruct_keys(head_key2, state2)

            exact_scores = torch.cat(
                [
                    evaluation_queries @ head_key1.transpose(0, 1),
                    evaluation_queries @ head_key2.transpose(0, 1),
                ],
                dim=-1,
            ) * scaling
            method_scores = {
                "frozen_epoch1": proxy_scores(
                    evaluation_queries,
                    [reconstructed1_state1, reconstructed2_state1],
                    [state1, state1],
                    scaling,
                ),
                "frozen_epoch1_reallocated": proxy_scores(
                    evaluation_queries,
                    [reconstructed1_reallocated, reconstructed2_reallocated],
                    [reallocated, reallocated],
                    scaling,
                ),
                "global_rebuild": proxy_scores(
                    evaluation_queries,
                    [reconstructed1_global, reconstructed2_global],
                    [global_state, global_state],
                    scaling,
                ),
                "versioned_two_epoch": proxy_scores(
                    evaluation_queries,
                    [reconstructed1_epoch, reconstructed2_epoch],
                    [state1, state2],
                    scaling,
                ),
            }
            rows = selection_rows(exact_scores, method_scores, args.topk)
            queries_per_domain = int(head_qeval1.shape[0])
            for row in rows:
                row.update(
                    {
                        "layer": layer,
                        "kv_head": kv_head,
                        "query_domain": (
                            "epoch1"
                            if int(row["query_index"]) < queries_per_domain
                            else "epoch2"
                        ),
                    }
                )
                detail.append(row)
            profiles.extend(
                {
                    "layer": layer,
                    "kv_head": kv_head,
                    "profile": name,
                    "allocation": "-".join(map(str, state["allocation"])),
                    "index_bits": GROUP_SIZE
                    * sum(bits + int(bits > 0) for bits in state["allocation"]),
                    "key_covariance_cross_epoch_drift": cross_drift,
                    "key_covariance_within_epoch1_drift": within_epoch1_drift,
                    "key_covariance_within_epoch2_drift": within_epoch2_drift,
                }
                for name, state in (
                    ("epoch1", state1),
                    ("epoch2", state2),
                    ("global_rebuild", global_state),
                    ("frozen_reallocated", reallocated),
                )
            )
            print(f"layer={layer} kv_head={kv_head} complete", flush=True)
            del exact_scores, method_scores
            torch.cuda.empty_cache()

        del key1, key2, qcal1, qcal2, qeval1, qeval2
        torch.cuda.empty_cache()

    aggregate: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in detail if row["method"] == method]
        aggregate[method] = {
            "queries": len(method_rows),
            "topk_recall": summarize([float(row["topk_recall"]) for row in method_rows]),
            "selected_attention_mass": summarize(
                [float(row["selected_attention_mass"]) for row in method_rows]
            ),
            "selected_to_oracle_mass": summarize(
                [float(row["selected_to_oracle_mass"]) for row in method_rows]
            ),
            "by_query_domain": {
                domain: {
                    "topk_recall_mean": sum(
                        float(row["topk_recall"])
                        for row in method_rows
                        if row["query_domain"] == domain
                    )
                    / sum(row["query_domain"] == domain for row in method_rows),
                    "selected_to_oracle_mass_mean": sum(
                        float(row["selected_to_oracle_mass"])
                        for row in method_rows
                        if row["query_domain"] == domain
                    )
                    / sum(row["query_domain"] == domain for row in method_rows),
                }
                for domain in ("epoch1", "epoch2")
            },
        }
    result = {
        "schema": "qksieve_versioned_coordinate_drift_trace_v1",
        "epoch1_trace": str(args.epoch1_trace),
        "epoch2_trace": str(args.epoch2_trace),
        "history_tokens": combined_history_tokens,
        "topk": args.topk,
        "sample_stride": args.sample_stride,
        "calibration_steps": args.calibration_steps,
        "evaluation_steps_per_domain": args.evaluation_steps,
        "rate_units": args.rate_units,
        "logical_index_bits_per_token_kv_head": args.rate_units * GROUP_SIZE,
        "index_ratio_of_full_fp16_kv": args.rate_units * GROUP_SIZE / FULL_KV_BITS,
        "additional_basis_bytes_per_layer_kv_head_per_epoch": 65536,
        "extra_second_epoch_basis_ratio_of_index": (
            65536 / (combined_history_tokens * args.rate_units * 2)
            if combined_history_tokens
            else None
        ),
        "methods": aggregate,
        "allocation_histograms": {
            profile: dict(
                Counter(
                    row["allocation"]
                    for row in profiles
                    if row["profile"] == profile
                )
            )
            for profile in {row["profile"] for row in profiles}
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "detail.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    with (args.output_dir / "profiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in profiles for key in row}))
        writer.writeheader()
        writer.writerows(profiles)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
