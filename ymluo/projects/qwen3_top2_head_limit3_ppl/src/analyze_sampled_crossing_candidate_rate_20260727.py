from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from analyze_hierarchical_spectral_quantization_20260727 import (
    FULL_KV_BITS_PER_TOKEN,
    covariance_basis,
    quantize_groupwise,
    query_int8,
    sign_reconstruct,
)


CORE_CODE_BITS = 16 * 8 + 32 * 4
TAIL_CODE_BITS = 80


def parse_floats(value: str) -> list[float]:
    values = sorted({float(part) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def systematic_sample_indices(
    count: int,
    sample_count: int,
    phase: int,
    device: torch.device,
) -> torch.Tensor:
    actual = min(count, sample_count)
    stride = max(1, count // actual)
    while math.gcd(stride, count) != 1:
        stride += 1
    offsets = torch.arange(actual, device=device)
    return (phase + offsets * stride) % count


def candidate_mask_from_count(
    coarse_scores: torch.Tensor,
    candidate_count: int,
) -> torch.Tensor:
    indices = torch.topk(coarse_scores, k=candidate_count).indices
    mask = torch.zeros_like(coarse_scores, dtype=torch.bool)
    mask[indices] = True
    return mask


def sampled_crossing_candidates(
    coarse_scores: torch.Tensor,
    tail_scores: torch.Tensor,
    sample: torch.Tensor,
    selected_count: int,
    upper_quantile: float,
    max_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    upper_tail = torch.quantile(tail_scores[sample].float(), upper_quantile)
    candidate_count = selected_count
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        candidate = candidate_mask_from_count(coarse_scores, candidate_count)
        refined_candidate_scores = (
            coarse_scores[candidate] + tail_scores[candidate]
        )
        threshold = torch.topk(
            refined_candidate_scores, k=selected_count
        ).values[-1]
        required = int(
            ((coarse_scores + upper_tail) >= threshold).sum().item()
        )
        required = min(
            coarse_scores.numel(),
            max(selected_count, required),
        )
        if required <= candidate_count:
            break
        candidate_count = required

    candidate = candidate_mask_from_count(coarse_scores, candidate_count)
    refined = torch.full_like(coarse_scores, -torch.inf)
    refined[candidate] = coarse_scores[candidate] + tail_scores[candidate]
    return candidate, refined, float(upper_tail.item()), iterations


def oracle_candidate_count(
    coarse_scores: torch.Tensor,
    full_scores: torch.Tensor,
    selected_count: int,
) -> tuple[int, torch.Tensor]:
    full_top = torch.topk(full_scores, k=selected_count).indices
    coarse_order = torch.argsort(coarse_scores, descending=True)
    ranks = torch.empty_like(coarse_order)
    ranks[coarse_order] = torch.arange(
        coarse_order.numel(), device=coarse_scores.device
    )
    count = int(ranks[full_top].max().item()) + 1
    return count, full_top


def selection_metrics(
    exact_scores: torch.Tensor,
    attention: torch.Tensor,
    refined_scores: torch.Tensor,
    candidate: torch.Tensor,
    exact_top: torch.Tensor,
    full_proxy_top: torch.Tensor,
    selected_count: int,
) -> dict[str, float]:
    selected = torch.topk(refined_scores, k=selected_count).indices
    selected_mask = torch.zeros_like(candidate)
    selected_mask[selected] = True
    return {
        "candidate_ratio": float(candidate.float().mean().item()),
        "candidate_exact_top2_recall": float(
            candidate[exact_top].float().mean().item()
        ),
        "candidate_proxy_topk_recall": float(
            candidate[full_proxy_top].float().mean().item()
        ),
        "selected_exact_top2_recall": float(
            selected_mask[exact_top].float().mean().item()
        ),
        "selected_proxy_topk_recall": float(
            selected_mask[full_proxy_top].float().mean().item()
        ),
        "selected_attention_mass": float(attention[selected].sum().item()),
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)

    metrics = (
        "candidate_ratio",
        "candidate_exact_top2_recall",
        "candidate_proxy_topk_recall",
        "selected_exact_top2_recall",
        "selected_proxy_topk_recall",
        "selected_attention_mass",
        "upper_tail_quantile",
        "iterations",
        "mean_scan_code_bits",
    )
    output: list[dict[str, Any]] = []
    for (method, selected_fraction), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": selected_fraction,
            "cases": len(items),
            "stored_code_bits": CORE_CODE_BITS + TAIL_CODE_BITS,
            "stored_code_ratio_of_full_kv": (
                CORE_CODE_BITS + TAIL_CODE_BITS
            )
            / FULL_KV_BITS_PER_TOKEN,
        }
        for metric in metrics:
            stats = summarize(float(item[metric]) for item in items)
            result.update(
                {f"{metric}_{stat}": value for stat, value in stats.items()}
            )
        result["mean_scan_code_ratio_of_full_kv"] = (
            result["mean_scan_code_bits_mean"] / FULL_KV_BITS_PER_TOKEN
        )
        output.append(result)
    return output


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
            "Estimate a per-head progressive tail candidate rate from 256 "
            "sampled tail corrections and the observed coarse-score boundary."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_samples", type=int, default=256)
    parser.add_argument("--upper_quantiles", default="0.95,0.975,0.99,0.995,1")
    parser.add_argument("--selected_fractions", default="0.02,0.03,0.04,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_iterations", type=int, default=4)
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    selected_fractions = parse_floats(args.selected_fractions)
    upper_quantiles = parse_floats(args.upper_quantiles)
    if any(not 0.0 < value <= 1.0 for value in upper_quantiles):
        raise ValueError("upper quantiles must be in (0, 1]")
    if any(value < args.top_fraction for value in selected_fractions):
        raise ValueError("selected fractions cannot be below top_fraction")

    payload = torch.load(args.trace_path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", []))
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError("trace contains no records")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    states: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        raw_key = record.get("key")
        if layer not in states:
            if raw_key is None:
                raise RuntimeError(f"layer {layer} is missing its initial key")
            all_key = raw_key.to(device).float()[0]
            history_count = int(all_key.shape[1]) - 1
            key = all_key[:, :history_count]
            prepared_heads: list[dict[str, torch.Tensor]] = []
            for kv_head in range(key.shape[0]):
                head_key = key[kv_head]
                basis, _ = covariance_basis(head_key[:: args.sample_stride])
                coefficients = head_key @ basis
                mixed_core_key = torch.cat(
                    (
                        quantize_groupwise(coefficients[:, :16], 8),
                        quantize_groupwise(coefficients[:, 16:48], 4),
                    ),
                    dim=-1,
                )
                tail_sign_key = sign_reconstruct(coefficients[:, 48:])
                prepared_heads.append(
                    {
                        "head_key": head_key,
                        "basis": basis,
                        "mixed_core_key": mixed_core_key,
                        "tail_sign_key": tail_sign_key,
                    }
                )
            states[layer] = {
                "history_count": history_count,
                "kv_heads": int(key.shape[0]),
                "prepared_heads": prepared_heads,
            }

        state = states[layer]
        history_count = int(state["history_count"])
        kv_heads = int(state["kv_heads"])
        query = record["query"].to(device).float()[0, :, 0, :]
        query_heads = int(query.shape[0])
        if query_heads % kv_heads:
            raise RuntimeError("query heads must be divisible by KV heads")
        groups = query_heads // kv_heads
        scaling = float(record["scaling"])
        exact_top_count = max(1, math.ceil(args.top_fraction * history_count))

        for kv_head in range(kv_heads):
            prepared = state["prepared_heads"][kv_head]
            for group in range(groups):
                query_head = kv_head * groups + group
                head_query = query[query_head]
                projected_query = head_query @ prepared["basis"]
                projected_query_q8 = query_int8(projected_query)
                exact_scores = (prepared["head_key"] @ head_query) * scaling
                attention = torch.softmax(exact_scores, dim=-1)
                exact_top = torch.topk(
                    exact_scores, k=exact_top_count
                ).indices
                coarse_scores = (
                    prepared["mixed_core_key"] @ projected_query_q8[:48]
                ) * scaling
                tail_scores = (
                    prepared["tail_sign_key"] @ projected_query_q8[48:]
                ) * scaling
                full_proxy_scores = coarse_scores + tail_scores
                phase = (
                    record_index * 1009
                    + layer * 131
                    + kv_head * 31
                    + query_head * 17
                ) % history_count
                sample = systematic_sample_indices(
                    history_count,
                    args.calibration_samples,
                    phase,
                    device,
                )

                for selected_fraction in selected_fractions:
                    selected_count = max(
                        1, math.ceil(selected_fraction * history_count)
                    )
                    oracle_count, full_proxy_top = oracle_candidate_count(
                        coarse_scores,
                        full_proxy_scores,
                        selected_count,
                    )
                    oracle_candidate = candidate_mask_from_count(
                        coarse_scores, oracle_count
                    )
                    oracle_refined = torch.full_like(coarse_scores, -torch.inf)
                    oracle_refined[oracle_candidate] = full_proxy_scores[
                        oracle_candidate
                    ]
                    oracle_metrics = selection_metrics(
                        exact_scores,
                        attention,
                        oracle_refined,
                        oracle_candidate,
                        exact_top,
                        full_proxy_top,
                        selected_count,
                    )
                    oracle_ratio = oracle_count / history_count
                    rows.append(
                        {
                            "label": args.label,
                            "record": record_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "history_count": history_count,
                            "method": "oracle_min_candidate",
                            "selected_fraction_target": selected_fraction,
                            "upper_tail_quantile": 0.0,
                            "iterations": 1,
                            "mean_scan_code_bits": (
                                CORE_CODE_BITS
                                + TAIL_CODE_BITS * oracle_ratio
                            ),
                            **oracle_metrics,
                        }
                    )

                    for upper_quantile in upper_quantiles:
                        (
                            candidate,
                            refined,
                            upper_tail,
                            iterations,
                        ) = sampled_crossing_candidates(
                            coarse_scores,
                            tail_scores,
                            sample,
                            selected_count,
                            upper_quantile,
                            args.max_iterations,
                        )
                        metrics = selection_metrics(
                            exact_scores,
                            attention,
                            refined,
                            candidate,
                            exact_top,
                            full_proxy_top,
                            selected_count,
                        )
                        candidate_ratio = metrics["candidate_ratio"]
                        sample_ratio = min(
                            1.0,
                            args.calibration_samples / history_count,
                        )
                        rows.append(
                            {
                                "label": args.label,
                                "record": record_index,
                                "layer": layer,
                                "kv_head": kv_head,
                                "query_head": query_head,
                                "history_count": history_count,
                                "method": (
                                    f"sampled_crossing_q{upper_quantile:g}"
                                ),
                                "selected_fraction_target": selected_fraction,
                                "upper_tail_quantile": upper_tail,
                                "iterations": iterations,
                                "mean_scan_code_bits": (
                                    CORE_CODE_BITS
                                    + TAIL_CODE_BITS
                                    * min(1.0, candidate_ratio + sample_ratio)
                                ),
                                **metrics,
                            }
                        )

        print(
            json.dumps(
                {
                    "label": args.label,
                    "record": record_index + 1,
                    "records": len(records),
                    "rows": len(rows),
                }
            ),
            flush=True,
        )

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    report = {
        "config": {
            **vars(args),
            "trace_path": str(args.trace_path),
            "output_dir": str(args.output_dir),
            "selected_fractions": selected_fractions,
            "upper_quantiles": upper_quantiles,
            "code_bit_accounting": {
                "core": CORE_CODE_BITS,
                "tail": TAIL_CODE_BITS,
                "full_kv": FULL_KV_BITS_PER_TOKEN,
            },
        },
        "records": len(records),
        "methods": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
