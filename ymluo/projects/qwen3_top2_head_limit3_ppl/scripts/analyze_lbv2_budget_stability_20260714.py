#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def index_method(paths: list[Path], method: str) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_rows(paths) if row.get("method") == method}


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def prediction(row: dict[str, str]) -> str:
    return row.get("longbench_v2_pred", "").strip().upper()


def summarize_source(rows: list[dict[str, str]], full: list[dict[str, str]]) -> dict[str, Any]:
    count = max(1, len(rows))
    return {
        "samples": len(rows),
        "score": sum(value(row, "score") for row in rows) / count,
        "prediction_agreement_with_full": sum(
            prediction(row) == prediction(full_row) for row, full_row in zip(rows, full)
        )
        / count,
        "mean_kv_ratio": sum(value(row, "keep_fraction") for row in rows) / count,
        "online_seconds": sum(value(row, "online_seconds") for row in rows),
    }


def summarize_policy(records: list[dict[str, Any]], full_online: float) -> dict[str, Any]:
    count = max(1, len(records))
    online = sum(item["online_seconds"] for item in records)
    return {
        "samples": len(records),
        "score": sum(item["score"] for item in records) / count,
        "prediction_agreement_with_full": sum(item["prediction_agreement"] for item in records) / count,
        "mean_peak_kv_ratio": sum(item["peak_kv_ratio"] for item in records) / count,
        "online_seconds": online,
        "online_speed_vs_full": full_online / online if online > 0 else None,
        "selected_source_counts": dict(Counter(item["selected_source"] for item in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True, type=Path)
    parser.add_argument("--b2048", nargs="+", required=True, type=Path)
    parser.add_argument("--b3072", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    full_index = index_method(args.base, "full_kv")
    base_index = index_method(args.base, "ours_page_gather")
    b2048_index = index_method(args.b2048, "ours_page_gather")
    b3072_index = index_method(args.b3072, "ours_page_gather")
    sample_ids = sorted(full_index.keys() & base_index.keys() & b2048_index.keys() & b3072_index.keys())
    full_rows = [full_index[sample_id] for sample_id in sample_ids]
    base_rows = [base_index[sample_id] for sample_id in sample_ids]
    b2048_rows = [b2048_index[sample_id] for sample_id in sample_ids]
    b3072_rows = [b3072_index[sample_id] for sample_id in sample_ids]
    full_summary = summarize_source(full_rows, full_rows)

    labels: list[dict[str, Any]] = []
    two_rung_records: list[dict[str, Any]] = []
    three_rung_records: list[dict[str, Any]] = []
    for sample_id, full_row, base_row, row2, row3 in zip(
        sample_ids, full_rows, base_rows, b2048_rows, b3072_rows
    ):
        pred_full = prediction(full_row)
        pred_base = prediction(base_row)
        pred2 = prediction(row2)
        pred3 = prediction(row3)
        stable_base2 = bool(pred_base) and pred_base == pred2
        stable_23 = bool(pred2) and pred2 == pred3
        match = {
            "base": pred_base == pred_full,
            "b2048": pred2 == pred_full,
            "b3072": pred3 == pred_full,
        }
        correct = {
            "base": value(base_row, "score") > 0.5,
            "b2048": value(row2, "score") > 0.5,
            "b3072": value(row3, "score") > 0.5,
            "full": value(full_row, "score") > 0.5,
        }
        min_match_full = next((name for name in ("base", "b2048", "b3072") if match[name]), "full")
        min_correct = next((name for name in ("base", "b2048", "b3072", "full") if correct[name]), "none")

        common_online = value(base_row, "online_seconds") + value(row2, "online_seconds")
        if stable_base2:
            two_selected_name = "base_stable_b2048"
            two_selected_row = base_row
        else:
            two_selected_name = "full_fallback"
            two_selected_row = full_row
        two_rung_records.append(
            {
                "selected_source": two_selected_name,
                "score": value(two_selected_row, "score"),
                "prediction_agreement": prediction(two_selected_row) == pred_full,
                "peak_kv_ratio": max(
                    value(base_row, "keep_fraction"),
                    value(row2, "keep_fraction"),
                    value(two_selected_row, "keep_fraction"),
                ),
                "online_seconds": common_online
                + (value(full_row, "online_seconds") if not stable_base2 else 0.0),
            }
        )

        three_online = common_online
        if stable_base2:
            three_selected_name = "base_stable_b2048"
            three_selected_row = base_row
        else:
            three_online += value(row3, "online_seconds")
            if stable_23:
                three_selected_name = "b2048_stable_b3072"
                three_selected_row = row2
            else:
                three_selected_name = "full_fallback"
                three_selected_row = full_row
                three_online += value(full_row, "online_seconds")
        three_rung_records.append(
            {
                "selected_source": three_selected_name,
                "score": value(three_selected_row, "score"),
                "prediction_agreement": prediction(three_selected_row) == pred_full,
                "peak_kv_ratio": max(
                    value(base_row, "keep_fraction"),
                    value(row2, "keep_fraction"),
                    value(row3, "keep_fraction") if not stable_base2 else 0.0,
                    value(three_selected_row, "keep_fraction"),
                ),
                "online_seconds": three_online,
            }
        )

        labels.append(
            {
                "sample_id": sample_id,
                "domain": full_row.get("domain", ""),
                "sub_domain": full_row.get("sub_domain", ""),
                "operator_mode": base_row.get("ours_operator_mode", ""),
                "base_budget": int(value(base_row, "budget_tokens")),
                "pred_base": pred_base,
                "pred_b2048": pred2,
                "pred_b3072": pred3,
                "pred_full": pred_full,
                "stable_base_b2048": int(stable_base2),
                "stable_b2048_b3072": int(stable_23),
                "base_matches_full": int(match["base"]),
                "b2048_matches_full": int(match["b2048"]),
                "b3072_matches_full": int(match["b3072"]),
                "min_match_full": min_match_full,
                "min_correct": min_correct,
                "base_score": value(base_row, "score"),
                "b2048_score": value(row2, "score"),
                "b3072_score": value(row3, "score"),
                "full_score": value(full_row, "score"),
                "base_kv_ratio": value(base_row, "keep_fraction"),
                "b2048_kv_ratio": value(row2, "keep_fraction"),
                "b3072_kv_ratio": value(row3, "keep_fraction"),
                "ours_score_gap2": value(base_row, "ours_score_gap2"),
                "ours_score_gap3": value(base_row, "ours_score_gap3"),
                "ours_score_entropy": value(base_row, "ours_score_entropy"),
                "ours_score_max": value(base_row, "ours_score_max"),
                "ours_query_coverage_recall": value(base_row, "ours_query_coverage_recall"),
            }
        )

    full_online = full_summary["online_seconds"]
    payload = {
        "matched_samples": len(sample_ids),
        "sources": {
            "full": full_summary,
            "base": summarize_source(base_rows, full_rows),
            "b2048": summarize_source(b2048_rows, full_rows),
            "b3072": summarize_source(b3072_rows, full_rows),
        },
        "stability": {
            "base_b2048_rate": sum(item["stable_base_b2048"] for item in labels) / max(1, len(labels)),
            "base_b2048_precision_to_full": sum(
                item["base_matches_full"] for item in labels if item["stable_base_b2048"]
            )
            / max(1, sum(item["stable_base_b2048"] for item in labels)),
            "b2048_b3072_rate": sum(item["stable_b2048_b3072"] for item in labels) / max(1, len(labels)),
            "b2048_b3072_precision_to_full": sum(
                item["b2048_matches_full"] for item in labels if item["stable_b2048_b3072"]
            )
            / max(1, sum(item["stable_b2048_b3072"] for item in labels)),
        },
        "label_counts": {
            "min_match_full": dict(Counter(item["min_match_full"] for item in labels)),
            "min_correct": dict(Counter(item["min_correct"] for item in labels)),
        },
        "policies": {
            "two_rung_stability_then_full": summarize_policy(two_rung_records, full_online),
            "three_rung_stability_then_full": summarize_policy(three_rung_records, full_online),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]) if labels else ["sample_id"])
        writer.writeheader()
        writer.writerows(labels)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
