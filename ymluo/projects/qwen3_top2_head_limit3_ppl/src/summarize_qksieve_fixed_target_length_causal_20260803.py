#!/usr/bin/env python
"""Summarize the fixed-target QKSieve length-causal experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


EXACT = "exact_qk_oracle_k1280"
PROXY_FULL = "qksieve_keymse_requestlocal_fulltopk_k1280"
PROXY_SAMPLED = "qksieve_keymse_requestlocal_sampled_k1280_c32"
VALUE_TAIL = "qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def optional_float(value: Any) -> float:
    return float(value) if value is not None else math.nan


def pearson(rows: list[dict[str, Any]], x: str, y: str) -> float:
    pairs = [
        (float(row[x]), float(row[y]))
        for row in rows
        if math.isfinite(float(row.get(x, math.nan)))
        and math.isfinite(float(row.get(y, math.nan)))
    ]
    if len(pairs) < 3:
        return math.nan
    values = torch.tensor(pairs, dtype=torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(centered.square().sum(dim=0).prod())
    if float(denominator) == 0.0:
        return math.nan
    return float((centered[:, 0] * centered[:, 1]).sum() / denominator)


def method(rows: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return rows.get(name)


def quality(full_nll: float, candidate: dict[str, Any] | None) -> float:
    if candidate is None:
        return math.nan
    return math.exp(full_nll - float(candidate["nll"]))


def token_metric(
    payload: dict[str, Any],
    variant: str,
    name: str,
) -> float:
    values = [
        float(row[name])
        for row in payload.get("token_rows", {}).get(variant, [])
        if name in row
    ]
    return sum(values) / len(values) if values else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    """Replace non-finite diagnostics with JSON null for strict readers."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    target_hashes: dict[str, set[str]] = {}
    recent_hashes: dict[str, set[str]] = {}

    for path in sorted(args.input_root.glob("*_h*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = {
            str(row["variant"]): row
            for row in payload.get("rows", [])
            if row.get("variant")
        }
        full = method(rows, "full_attention")
        if full is None:
            continue
        topic = str(payload["topic"])
        target_hashes.setdefault(topic, set()).add(
            str(payload.get("target_token_ids_sha256", ""))
        )
        recent_hashes.setdefault(topic, set()).add(
            str(payload.get("recent_256_token_ids_sha256", ""))
        )
        full_nll = float(full["nll"])
        exact = method(rows, EXACT)
        proxy_full = method(rows, PROXY_FULL)
        sampled = method(rows, PROXY_SAMPLED)
        value_tail = method(rows, VALUE_TAIL)
        spectrum = (
            payload.get("qk_product_spectrum", {})
            .get("by_query_shrinkage", {})
            .get("0.75", {})
        )

        def delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
            if left is None or right is None:
                return math.nan
            return float(left["nll"]) - float(right["nll"])

        exact_mass = optional_float(
            exact.get("oracle_retained_mass_mean") if exact else None
        )
        case_rows.append(
            {
                "topic": topic,
                "seed": int(payload["seed"]),
                "history_tokens": int(payload["history_tokens"]),
                "eval_tokens": int(payload["eval_tokens"]),
                "method_count": len(rows),
                "full_nll": full_nll,
                "exact_quality": quality(full_nll, exact),
                "proxy_full_quality": quality(full_nll, proxy_full),
                "sampled_quality": quality(full_nll, sampled),
                "value_tail_quality": quality(full_nll, value_tail),
                "budget_delta_nll": delta(exact, full),
                "proxy_ranking_delta_nll": delta(proxy_full, exact),
                "sampling_delta_nll": delta(sampled, proxy_full),
                "value_correction_delta_nll": delta(value_tail, sampled),
                "residual_delta_nll": delta(value_tail, full),
                "exact_selected_mass": exact_mass,
                "exact_omitted_mass": (
                    1.0 - exact_mass if math.isfinite(exact_mass) else math.nan
                ),
                "exact_boundary_gap_over_score_std": token_metric(
                    payload,
                    EXACT,
                    "exact_boundary_gap_over_score_std_mean",
                ),
                "exact_value_selected_tail_gap_l2": token_metric(
                    payload,
                    EXACT,
                    "value_selected_tail_gap_l2_mean",
                ),
                "exact_value_sparse_full_relative_error": token_metric(
                    payload,
                    EXACT,
                    "value_sparse_full_relative_error_mean",
                ),
                "exact_value_identity_relative_residual": token_metric(
                    payload,
                    EXACT,
                    "value_identity_relative_residual_mean",
                ),
                "proxy_full_topk_recall": optional_float(
                    proxy_full.get("exact_topk_recall_mean")
                    if proxy_full
                    else None
                ),
                "proxy_full_selected_mass": optional_float(
                    proxy_full.get("selected_attention_mass_mean")
                    if proxy_full
                    else None
                ),
                "sampled_topk_recall": optional_float(
                    sampled.get("exact_topk_recall_mean") if sampled else None
                ),
                "sampled_selected_mass": optional_float(
                    sampled.get("selected_attention_mass_mean")
                    if sampled
                    else None
                ),
                "exact_kl": optional_float(
                    exact.get("kl_full_to_sparse_mean") if exact else None
                ),
                "sampled_kl": optional_float(
                    sampled.get("kl_full_to_sparse_mean") if sampled else None
                ),
                "value_tail_kl": optional_float(
                    value_tail.get("kl_full_to_sparse_mean")
                    if value_tail
                    else None
                ),
                "spectrum_rank95_p50": optional_float(
                    spectrum.get("rank95", {}).get("p50")
                ),
                "spectrum_rank95_p90": optional_float(
                    spectrum.get("rank95", {}).get("p90")
                ),
                "spectrum_rank99_p50": optional_float(
                    spectrum.get("rank99", {}).get("p50")
                ),
                "spectrum_top48_energy_p50": optional_float(
                    spectrum.get("top48_energy", {}).get("p50")
                ),
                "spectrum_top48_energy_p10": optional_float(
                    spectrum.get("top48_energy", {}).get("p10")
                ),
                "target_hash": str(payload.get("target_token_ids_sha256", "")),
                "recent_hash": str(
                    payload.get("recent_256_token_ids_sha256", "")
                ),
                "path": str(path),
            }
        )

    case_rows.sort(key=lambda row: (row["history_tokens"], row["topic"]))
    write_csv(args.output_dir / "case_decomposition.csv", case_rows)

    complete_rows = [row for row in case_rows if int(row["method_count"]) == 5]
    topic_count = len({str(row["topic"]) for row in complete_rows})
    eval_token_counts = sorted({int(row["eval_tokens"]) for row in complete_rows})
    correlations = {
        "budget_loss_vs_exact_omitted_mass": pearson(
            complete_rows, "budget_delta_nll", "exact_omitted_mass"
        ),
        "budget_loss_vs_spectrum_rank95": pearson(
            complete_rows, "budget_delta_nll", "spectrum_rank95_p50"
        ),
        "budget_loss_vs_value_sparse_full_relative_error": pearson(
            complete_rows,
            "budget_delta_nll",
            "exact_value_sparse_full_relative_error",
        ),
        "omitted_mass_vs_value_sparse_full_relative_error": pearson(
            complete_rows,
            "exact_omitted_mass",
            "exact_value_sparse_full_relative_error",
        ),
        "proxy_loss_vs_one_minus_recall": pearson(
            [
                {
                    **row,
                    "one_minus_proxy_recall": 1.0
                    - float(row["proxy_full_topk_recall"]),
                }
                for row in complete_rows
                if math.isfinite(float(row["proxy_full_topk_recall"]))
            ],
            "proxy_ranking_delta_nll",
            "one_minus_proxy_recall",
        ),
        "value_gain_vs_exact_omitted_mass": pearson(
            complete_rows, "value_correction_delta_nll", "exact_omitted_mass"
        ),
    }
    summary = {
        "schema": "qksieve_fixed_target_length_causal_summary_v1",
        "input_root": str(args.input_root),
        "case_count": len(case_rows),
        "complete_case_count": len(complete_rows),
        "topic_count": topic_count,
        "eval_token_counts": eval_token_counts,
        "target_fixed_within_topic": {
            topic: len(hashes - {""}) == 1
            for topic, hashes in target_hashes.items()
        },
        "recent_256_fixed_within_topic": {
            topic: len(hashes - {""}) == 1
            for topic, hashes in recent_hashes.items()
        },
        "correlations": correlations,
        "claim_boundary": (
            "Correlations are diagnostic only. "
            f"{topic_count} topics with per-case target-token counts "
            f"{eval_token_counts} do not establish population-level causality."
        ),
    }
    strict_summary = json_safe(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(strict_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(strict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
