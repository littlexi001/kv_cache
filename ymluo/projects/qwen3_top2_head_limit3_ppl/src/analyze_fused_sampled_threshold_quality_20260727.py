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


def kernel_sample_indices(
    count: int,
    sample_count: int,
    query_head: int,
    device: torch.device,
) -> torch.Tensor:
    """Match hier841_sample_thresholds_kernel's deterministic samples."""
    actual = min(count, sample_count)
    segment = max(1, count // actual)
    phase = (query_head * 131 + 17) % segment
    thread = torch.arange(actual, device=device, dtype=torch.long)
    centered = ((2 * thread + 1) * count) // (2 * actual)
    return (centered + phase) % count


def mask_recall(mask: torch.Tensor, target: torch.Tensor) -> float:
    return float(mask[target].float().mean().item())


def selected_metrics(
    selected: torch.Tensor,
    candidate: torch.Tensor,
    exact_top: torch.Tensor,
    proxy_top: torch.Tensor,
    attention: torch.Tensor,
) -> dict[str, float]:
    return {
        "candidate_ratio": float(candidate.float().mean().item()),
        "selected_ratio": float(selected.float().mean().item()),
        "candidate_exact_top2_recall": mask_recall(candidate, exact_top),
        "candidate_proxy_topk_recall": mask_recall(candidate, proxy_top),
        "selected_exact_top2_recall": mask_recall(selected, exact_top),
        "selected_proxy_topk_recall": mask_recall(selected, proxy_top),
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
        "selected_ratio",
        "candidate_exact_top2_recall",
        "candidate_proxy_topk_recall",
        "selected_exact_top2_recall",
        "selected_proxy_topk_recall",
        "selected_attention_mass",
        "capacity_fraction",
        "capacity_overflow",
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
            "Evaluate the exact variable-cardinality semantics used by the "
            "fused sampled-threshold hierarchical 8/4/1 CUDA kernel."
        )
    )
    parser.add_argument("--trace_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--calibration_samples", type=int, default=256)
    parser.add_argument("--candidate_fractions", default="0.2,0.3,0.4")
    parser.add_argument("--selected_fractions", default="0.02,0.04,0.06")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--capacity_multiplier", type=float, default=2.0)
    parser.add_argument("--minimum_capacity_fraction", type=float, default=0.04)
    parser.add_argument("--max_records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    candidate_fractions = parse_floats(args.candidate_fractions)
    selected_fractions = parse_floats(args.selected_fractions)
    if any(not 0.0 < value < 1.0 for value in candidate_fractions):
        raise ValueError("candidate fractions must be in (0, 1)")
    if any(not 0.0 < value <= 1.0 for value in selected_fractions):
        raise ValueError("selected fractions must be in (0, 1]")
    if max(selected_fractions) > min(candidate_fractions):
        raise ValueError("selected fraction cannot exceed candidate fraction")

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
                full_proxy_scores = coarse_scores + (
                    prepared["tail_sign_key"] @ projected_query_q8[48:]
                ) * scaling
                sample = kernel_sample_indices(
                    history_count,
                    args.calibration_samples,
                    query_head,
                    device,
                )
                sample_ratio = sample.numel() / history_count

                for selected_fraction in selected_fractions:
                    selected_count = max(
                        1, math.ceil(selected_fraction * history_count)
                    )
                    proxy_top = torch.topk(
                        full_proxy_scores, k=selected_count
                    ).indices
                    fixed_selected = torch.zeros(
                        history_count, dtype=torch.bool, device=device
                    )
                    fixed_selected[proxy_top] = True
                    fixed_metrics = selected_metrics(
                        fixed_selected,
                        torch.ones_like(fixed_selected),
                        exact_top,
                        proxy_top,
                        attention,
                    )
                    capacity_fraction = min(
                        1.0,
                        max(
                            args.minimum_capacity_fraction,
                            args.capacity_multiplier * selected_fraction,
                        ),
                    )
                    rows.append(
                        {
                            "label": args.label,
                            "record": record_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "query_head": query_head,
                            "history_count": history_count,
                            "method": "fixed_full841_topk",
                            "selected_fraction_target": selected_fraction,
                            "capacity_fraction": capacity_fraction,
                            "capacity_overflow": 0.0,
                            "mean_scan_code_bits": (
                                CORE_CODE_BITS + TAIL_CODE_BITS
                            ),
                            **fixed_metrics,
                        }
                    )

                    selected_keep = max(
                        1,
                        math.ceil(selected_fraction * sample.numel()),
                    )
                    selected_threshold = torch.topk(
                        full_proxy_scores[sample], k=selected_keep
                    ).values[-1]

                    for candidate_fraction in candidate_fractions:
                        candidate_keep = max(
                            1,
                            math.ceil(candidate_fraction * sample.numel()),
                        )
                        candidate_threshold = torch.topk(
                            coarse_scores[sample], k=candidate_keep
                        ).values[-1]
                        candidate = coarse_scores >= candidate_threshold
                        selected = candidate & (
                            full_proxy_scores >= selected_threshold
                        )
                        metrics = selected_metrics(
                            selected,
                            candidate,
                            exact_top,
                            proxy_top,
                            attention,
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
                                    "fused_sampled_threshold"
                                    f"_c{candidate_fraction:g}"
                                ),
                                "selected_fraction_target": selected_fraction,
                                "capacity_fraction": capacity_fraction,
                                "capacity_overflow": float(
                                    metrics["selected_ratio"]
                                    > capacity_fraction
                                ),
                                "mean_scan_code_bits": (
                                    CORE_CODE_BITS * (1.0 + sample_ratio)
                                    + TAIL_CODE_BITS
                                    * (
                                        sample_ratio
                                        + metrics["candidate_ratio"]
                                    )
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
            "candidate_fractions": candidate_fractions,
            "selected_fractions": selected_fractions,
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
