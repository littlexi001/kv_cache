from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from analyze_residual_certified_pca_20260717 import (
    covariance_basis,
    exact_rerank,
    projection_error_terms,
    selection_metrics,
    summarize,
)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(sorted({float(part) for part in value.split(",") if part.strip()}))
    if not values:
        raise ValueError("expected at least one candidate fraction")
    return values


def conformal_upper_quantile(values: list[float], coverage: float) -> float:
    if not values:
        raise ValueError("cannot calibrate from an empty sample")
    ordered = sorted(values)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
    return ordered[rank - 1]


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
            "Cross-topic evaluation of calibrated verify-then-expand PCA retrieval."
        )
    )
    parser.add_argument("--calibration_trace", required=True, type=Path)
    parser.add_argument("--test_trace", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--calibration_topic", required=True)
    parser.add_argument("--test_topic", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--sample_stride", type=int, default=32)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument(
        "--candidate_fractions", default="0.02,0.0225,0.025,0.03,0.04,0.06,0.08"
    )
    parser.add_argument("--coverages", default="0.8,0.9,0.95,0.99")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"trace has no records: {path}")
    return records


def record_head_terms(
    record: dict[str, Any],
    rank: int,
    sample_stride: int,
    device: torch.device,
):
    layer = int(record["layer"])
    query = record["query"].to(device).float()[0, :, 0, :]
    all_key = record["key"].to(device).float()[0]
    scaling = float(record["scaling"])
    query_heads = int(query.shape[0])
    kv_heads = int(all_key.shape[0])
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    history_count = int(all_key.shape[1]) - 1
    if history_count <= 0:
        raise ValueError("trace contains no history")
    key = all_key[:, :history_count]
    groups = query_heads // kv_heads
    projections = torch.stack(
        [
            covariance_basis(head_key[::sample_stride], rank)
            for head_key in key
        ]
    )
    for query_head in range(query_heads):
        kv_head = query_head // groups
        exact, approximate, _, single_bound, _ = projection_error_terms(
            query[query_head], key[kv_head], projections[kv_head], scaling
        )
        current_score = (
            all_key[kv_head, -1] @ query[query_head] * scaling
        ).view(1)
        attention = torch.softmax(
            torch.cat((exact, current_score)), dim=-1
        )[:history_count]
        yield {
            "layer": layer,
            "query_head": query_head,
            "history_count": history_count,
            "exact": exact,
            "approximate": approximate,
            "bound": single_bound,
            "attention": attention,
        }


@torch.inference_mode()
def calibrate(
    records: list[dict[str, Any]],
    rank: int,
    sample_stride: int,
    device: torch.device,
) -> tuple[dict[int, list[float]], list[float]]:
    by_layer: dict[int, list[float]] = defaultdict(list)
    global_values: list[float] = []
    for record_index, record in enumerate(records):
        for terms in record_head_terms(record, rank, sample_stride, device):
            normalized_positive_error = (
                (terms["exact"] - terms["approximate"]).clamp_min(0.0)
                / terms["bound"].clamp_min(1.0e-6)
            )
            sample_max = float(normalized_positive_error.max().item())
            by_layer[int(terms["layer"])].append(sample_max)
            global_values.append(sample_max)
        print(
            f"calibration record={record_index + 1}/{len(records)} "
            f"layer={int(record['layer'])}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return by_layer, global_values


def evaluate_selection(
    exact: torch.Tensor,
    approximate: torch.Tensor,
    bound: torch.Tensor,
    attention: torch.Tensor,
    top_count: int,
    fractions: tuple[float, ...],
    alpha: float,
) -> dict[str, Any]:
    history_count = int(exact.numel())
    true_indices = torch.topk(exact, k=top_count).indices
    oracle_mass = float(attention[true_indices].sum().item())
    approximate_order = torch.argsort(approximate, descending=True)
    calibrated_upper = approximate + float(alpha) * bound
    selected = None
    chosen_fraction = fractions[-1]
    stopped_by_bound = False
    for level_index, fraction in enumerate(fractions):
        candidate_count = min(
            history_count, max(top_count, math.ceil(fraction * history_count))
        )
        candidates = approximate_order[:candidate_count]
        selected = exact_rerank(exact, candidates, top_count)
        if level_index == len(fractions) - 1:
            chosen_fraction = candidate_count / history_count
            break
        threshold = exact[selected].min()
        outside = approximate_order[candidate_count:]
        outside_upper = (
            calibrated_upper[outside].max()
            if outside.numel()
            else torch.tensor(-torch.inf, device=exact.device)
        )
        if bool(outside_upper <= threshold):
            chosen_fraction = candidate_count / history_count
            stopped_by_bound = True
            break
    assert selected is not None

    oracle_fraction = fractions[-1]
    for fraction in fractions:
        candidate_count = min(
            history_count, max(top_count, math.ceil(fraction * history_count))
        )
        candidates = approximate_order[:candidate_count]
        candidate_mask = torch.zeros_like(exact, dtype=torch.bool)
        candidate_mask[candidates] = True
        if bool(candidate_mask[true_indices].all()):
            oracle_fraction = candidate_count / history_count
            break
    metrics = selection_metrics(selected, true_indices, attention, oracle_mass)
    return {
        "candidate_ratio": chosen_fraction,
        "oracle_candidate_ratio": oracle_fraction,
        "stopped_by_bound": float(stopped_by_bound),
        **metrics,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    fractions = parse_float_list(args.candidate_fractions)
    coverages = parse_float_list(args.coverages)
    if fractions[0] < args.top_fraction or fractions[-1] > 1.0:
        raise ValueError("candidate fractions must lie in [top_fraction, 1]")
    if any(not 0.0 < coverage < 1.0 for coverage in coverages):
        raise ValueError("coverages must lie in (0, 1)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    calibration_records = load_records(args.calibration_trace)
    test_records = load_records(args.test_trace)
    layer_risk, global_risk = calibrate(
        calibration_records, args.rank, args.sample_stride, device
    )
    global_alpha = {
        coverage: conformal_upper_quantile(global_risk, coverage)
        for coverage in coverages
    }
    layer_alpha = {
        layer: {
            coverage: conformal_upper_quantile(values, coverage)
            for coverage in coverages
        }
        for layer, values in layer_risk.items()
    }

    rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(test_records):
        for terms in record_head_terms(
            record, args.rank, args.sample_stride, device
        ):
            history_count = int(terms["history_count"])
            top_count = max(1, math.ceil(args.top_fraction * history_count))
            true_indices = torch.topk(terms["exact"], k=top_count).indices
            oracle_mass = float(terms["attention"][true_indices].sum().item())
            for fraction in fractions:
                candidate_count = min(
                    history_count,
                    max(top_count, math.ceil(fraction * history_count)),
                )
                candidates = torch.topk(
                    terms["approximate"], k=candidate_count
                ).indices
                selected = exact_rerank(terms["exact"], candidates, top_count)
                fixed_rows.append(
                    {
                        "test_topic": args.test_topic,
                        "record_index": record_index,
                        "layer": int(terms["layer"]),
                        "query_head": int(terms["query_head"]),
                        "candidate_ratio": candidate_count / history_count,
                        **selection_metrics(
                            selected,
                            true_indices,
                            terms["attention"],
                            oracle_mass,
                        ),
                    }
                )
            for coverage in coverages:
                alpha = layer_alpha.get(int(terms["layer"]), {}).get(
                    coverage, global_alpha[coverage]
                )
                metrics = evaluate_selection(
                    terms["exact"],
                    terms["approximate"],
                    terms["bound"],
                    terms["attention"],
                    top_count,
                    fractions,
                    alpha,
                )
                rows.append(
                    {
                        "calibration_topic": args.calibration_topic,
                        "test_topic": args.test_topic,
                        "record_index": record_index,
                        "layer": int(terms["layer"]),
                        "query_head": int(terms["query_head"]),
                        "coverage": coverage,
                        "alpha": alpha,
                        **metrics,
                    }
                )
        print(
            f"test record={record_index + 1}/{len(test_records)} "
            f"layer={int(record['layer'])}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = []
    for coverage in coverages:
        items = [row for row in rows if row["coverage"] == coverage]
        candidate_values = [float(row["candidate_ratio"]) for row in items]
        recall_values = [float(row["top2_recall"]) for row in items]
        mass_values = [float(row["top2_attention_mass_recall"]) for row in items]
        stop_counts = Counter(round(value, 6) for value in candidate_values)
        summaries.append(
            {
                "coverage": coverage,
                "cases": len(items),
                "global_alpha": global_alpha[coverage],
                "candidate_ratio": summarize(candidate_values),
                "top2_recall": summarize(recall_values),
                "top2_attention_mass_recall": summarize(mass_values),
                "exact_top2_set_rate": sum(value == 1.0 for value in recall_values)
                / len(recall_values),
                "bound_stop_rate": sum(
                    float(row["stopped_by_bound"]) for row in items
                )
                / len(items),
                "candidate_histogram": {
                    str(key): value for key, value in sorted(stop_counts.items())
                },
                "oracle_candidate_ratio_mean": sum(
                    float(row["oracle_candidate_ratio"]) for row in items
                )
                / len(items),
            }
        )

    fixed_summaries = []
    for fraction in fractions:
        items = [
            row
            for row in fixed_rows
            if abs(float(row["candidate_ratio"]) - fraction) < 1.0e-4
        ]
        recall_values = [float(row["top2_recall"]) for row in items]
        mass_values = [float(row["top2_attention_mass_recall"]) for row in items]
        fixed_summaries.append(
            {
                "candidate_fraction": fraction,
                "cases": len(items),
                "top2_recall": summarize(recall_values),
                "top2_attention_mass_recall": summarize(mass_values),
                "exact_top2_set_rate": sum(value == 1.0 for value in recall_values)
                / len(recall_values),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_head_query.csv", rows)
    write_csv(args.output_dir / "fixed_candidate_per_head_query.csv", fixed_rows)
    report = {
        "method": "calibrated_verify_then_expand",
        "calibration_topic": args.calibration_topic,
        "test_topic": args.test_topic,
        "rank": args.rank,
        "top_fraction": args.top_fraction,
        "candidate_fractions": fractions,
        "calibration_cases": len(global_risk),
        "layer_calibration_cases": {
            str(layer): len(values) for layer, values in layer_risk.items()
        },
        "global_alpha": global_alpha,
        "layer_alpha": layer_alpha,
        "fixed_candidate_summaries": fixed_summaries,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
