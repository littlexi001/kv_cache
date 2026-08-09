from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


GUARD_LABELS = (8, 10, 12, 16, 20, 25)
EXPECTED_CROSSING_LIMITS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze strict discrete top-k temporal certificates."
    )
    parser.add_argument("--trace_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--gqa_group_size", type=int, default=4)
    return parser.parse_args()


def tensor(record: dict[str, Any], name: str) -> torch.Tensor:
    return torch.as_tensor(record[name]).float()


def quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.float().reshape(-1)
    if value.numel() == 0:
        return {}
    points = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        dtype=torch.float32,
    )
    result = torch.quantile(value, points)
    return {
        name: float(item)
        for name, item in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "p99", "max"),
            result,
        )
    }


def conditional_rate(mask: torch.Tensor, event: torch.Tensor) -> float | None:
    selected = mask.bool()
    if not selected.any():
        return None
    return float(event.bool()[selected].float().mean())


def summarize_guard(
    records: list[dict[str, Any]],
    label: int,
    gqa_group_size: int,
) -> dict[str, Any] | None:
    prefix = f"temporal_guard{label}_"
    records = [
        record
        for record in records
        if prefix + "actual_miss_count" in record
    ]
    if not records:
        return None

    recall_rows = []
    miss_count_rows = []
    miss_fraction_rows = []
    strict_rows = []
    expected_rows = []
    old_fraction_rows = []
    grouped_miss_rows = []
    grouped_strict_rows = []
    grouped_expected_rows = []
    for record in records:
        recall = tensor(record, prefix + "recall")
        miss_count = tensor(record, prefix + "actual_miss_count")
        miss_fraction = tensor(record, prefix + "actual_miss_fraction")
        strict = tensor(record, prefix + "strict_safe").bool()
        expected = tensor(record, prefix + "expected_crossings")
        old_fraction = tensor(record, prefix + "old_fraction")
        if recall.shape[-1] % gqa_group_size != 0:
            raise RuntimeError("query heads are not divisible by GQA group size")
        group_shape = (
            *recall.shape[:-1],
            recall.shape[-1] // gqa_group_size,
            gqa_group_size,
        )
        grouped_miss = (miss_count.reshape(group_shape) > 0.0).any(dim=-1)
        grouped_strict = strict.reshape(group_shape).all(dim=-1)
        grouped_expected = expected.reshape(group_shape).amax(dim=-1)

        recall_rows.append(recall.reshape(-1))
        miss_count_rows.append(miss_count.reshape(-1))
        miss_fraction_rows.append(miss_fraction.reshape(-1))
        strict_rows.append(strict.reshape(-1))
        expected_rows.append(expected.reshape(-1))
        old_fraction_rows.append(old_fraction.reshape(-1))
        grouped_miss_rows.append(grouped_miss.reshape(-1))
        grouped_strict_rows.append(grouped_strict.reshape(-1))
        grouped_expected_rows.append(grouped_expected.reshape(-1))

    recall = torch.cat(recall_rows)
    miss_count = torch.cat(miss_count_rows)
    miss_fraction = torch.cat(miss_fraction_rows)
    strict = torch.cat(strict_rows)
    expected = torch.cat(expected_rows)
    old_fraction = torch.cat(old_fraction_rows)
    grouped_miss = torch.cat(grouped_miss_rows)
    grouped_strict = torch.cat(grouped_strict_rows)
    grouped_expected = torch.cat(grouped_expected_rows)
    head_miss = miss_count > 0.0

    gates = []
    for limit in EXPECTED_CROSSING_LIMITS:
        accepted = expected <= limit
        grouped_accepted = grouped_expected <= limit
        gates.append(
            {
                "max_expected_crossings": limit,
                "head_accept_rate": float(accepted.float().mean()),
                "head_miss_rate_when_accepted": conditional_rate(
                    accepted,
                    head_miss,
                ),
                "head_mean_miss_fraction_when_accepted": (
                    float(miss_fraction[accepted].mean())
                    if accepted.any()
                    else None
                ),
                "gqa_group_accept_rate": float(
                    grouped_accepted.float().mean()
                ),
                "gqa_group_miss_rate_when_accepted": conditional_rate(
                    grouped_accepted,
                    grouped_miss,
                ),
            }
        )

    return {
        "guard_percent": label,
        "records": len(records),
        "old_history_fraction_mean": float(old_fraction.mean()),
        "candidate_recall_mean": float(recall.mean()),
        "head_exact_coverage_rate": float((~head_miss).float().mean()),
        "gqa_group_exact_coverage_rate": float(
            (~grouped_miss).float().mean()
        ),
        "strict_head_accept_rate": float(strict.float().mean()),
        "strict_head_miss_rate_when_accepted": conditional_rate(
            strict,
            head_miss,
        ),
        "strict_gqa_group_accept_rate": float(
            grouped_strict.float().mean()
        ),
        "strict_gqa_group_miss_rate_when_accepted": conditional_rate(
            grouped_strict,
            grouped_miss,
        ),
        "actual_miss_count": quantiles(miss_count),
        "actual_miss_fraction": quantiles(miss_fraction),
        "expected_crossings": quantiles(expected),
        "expected_crossing_gates": gates,
    }


def summarize(
    records: list[dict[str, Any]],
    gqa_group_size: int,
) -> dict[str, Any]:
    records = [
        record
        for record in records
        if float(
            tensor(record, "temporal_trace_available").mean().item()
        )
        > 0.0
    ]
    if not records:
        raise RuntimeError("trace contains no adjacent-token temporal records")
    if gqa_group_size <= 0:
        raise ValueError("gqa_group_size must be positive")

    def flat(name: str) -> torch.Tensor:
        values = [
            tensor(record, name).reshape(-1)
            for record in records
            if name in record
        ]
        return torch.cat(values) if values else torch.empty(0)

    safe_rows: list[torch.Tensor] = []
    recall_rows: list[torch.Tensor] = []
    gqa_safe_rows: list[torch.Tensor] = []
    gqa_miss_rows: list[torch.Tensor] = []
    for record in records:
        safe = tensor(record, "temporal_certificate_safe").bool()
        recall = tensor(
            record,
            "temporal_candidate_recall_from_previous_with_new_tokens",
        )
        if safe.shape != recall.shape:
            raise RuntimeError("certificate and recall shapes do not match")
        if safe.shape[-1] % gqa_group_size != 0:
            raise RuntimeError("query heads are not divisible by GQA group size")
        grouped_safe = safe.reshape(
            *safe.shape[:-1],
            safe.shape[-1] // gqa_group_size,
            gqa_group_size,
        ).all(dim=-1)
        grouped_miss = (
            recall.reshape(
                *recall.shape[:-1],
                recall.shape[-1] // gqa_group_size,
                gqa_group_size,
            )
            < 1.0 - 1.0e-6
        ).any(dim=-1)
        safe_rows.append(safe.reshape(-1))
        recall_rows.append(recall.reshape(-1))
        gqa_safe_rows.append(grouped_safe.reshape(-1))
        gqa_miss_rows.append((grouped_safe & grouped_miss).reshape(-1))

    safe_all = torch.cat(safe_rows)
    recall_all = torch.cat(recall_rows)
    gqa_safe_all = torch.cat(gqa_safe_rows)
    gqa_miss_all = torch.cat(gqa_miss_rows)
    certified_miss = safe_all & (recall_all < 1.0 - 1.0e-6)
    margin = flat("temporal_boundary_margin")
    threshold_slack = flat("temporal_threshold_slack")
    score_bound = flat("temporal_score_change_bound")
    margin_ratio = flat("temporal_certificate_margin_ratio")

    by_layer_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer_records[int(record["layer"])].append(record)
    by_layer = []
    for layer, subset in sorted(by_layer_records.items()):
        layer_safe = torch.cat(
            [
                tensor(record, "temporal_certificate_safe").reshape(-1)
                for record in subset
            ]
        )
        layer_recall = torch.cat(
            [
                tensor(
                    record,
                    "temporal_candidate_recall_from_previous_with_new_tokens",
                ).reshape(-1)
                for record in subset
            ]
        )
        by_layer.append(
            {
                "layer": layer,
                "steps": len(subset),
                "candidate_jaccard": float(
                    torch.cat(
                        [
                            tensor(
                                record,
                                "temporal_candidate_jaccard",
                            ).reshape(-1)
                            for record in subset
                        ]
                    ).mean()
                ),
                "certificate_head_rate": float(layer_safe.mean()),
                "certified_miss_rate": float(
                    (
                        layer_safe.bool()
                        & (layer_recall < 1.0 - 1.0e-6)
                    ).float().mean()
                ),
            }
        )

    guards = []
    for label in GUARD_LABELS:
        guard = summarize_guard(records, label, gqa_group_size)
        if guard is not None:
            guards.append(guard)

    return {
        "records": len(records),
        "head_observations": int(safe_all.numel()),
        "gqa_group_observations": int(gqa_safe_all.numel()),
        "candidate_jaccard_mean": float(
            flat("temporal_candidate_jaccard").mean()
        ),
        "candidate_recall_mean": float(
            flat("temporal_candidate_recall_from_previous").mean()
        ),
        "candidate_recall_with_new_tokens_mean": float(recall_all.mean()),
        "query_delta_norm": quantiles(flat("temporal_query_delta_norm")),
        "score_change_bound": quantiles(score_bound),
        "threshold_slack": quantiles(threshold_slack),
        "threshold_slack_positive_rate": float(
            (threshold_slack > 0.0).float().mean()
        ),
        "discrete_boundary_margin": quantiles(margin),
        "discrete_boundary_margin_positive_rate": float(
            (margin > 0.0).float().mean()
        ),
        "certificate_margin_ratio": quantiles(margin_ratio),
        "certificate_head_rate": float(safe_all.float().mean()),
        "certificate_layer_rate": float(
            flat("temporal_certificate_layer_safe").mean()
        ),
        "certificate_gqa_group_rate": float(
            gqa_safe_all.float().mean()
        ),
        "certified_head_miss_rate": float(
            certified_miss.float().mean()
        ),
        "certified_gqa_group_miss_rate": float(
            gqa_miss_all.float().mean()
        ),
        "guards": guards,
        "layers": by_layer,
    }


def main() -> None:
    args = parse_args()
    records = torch.load(
        args.trace_path,
        map_location="cpu",
        weights_only=False,
    )
    output = summarize(records, args.gqa_group_size)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
