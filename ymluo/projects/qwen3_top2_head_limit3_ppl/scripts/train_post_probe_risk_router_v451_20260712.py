#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import train_operator_risk_router_v450_20260712 as v450
import train_source_router_v437_20260712 as v437


POST_FEATURES = [
    "post_chars",
    "post_words",
    "post_unique_word_ratio",
    "post_digit_fraction",
    "post_newlines",
    "post_generated_tokens",
    "post_refusal",
    "post_uncertainty",
    "post_empty",
    "post_exact_grounded",
    "post_content_overlap",
]

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "with", "i", "you", "we", "they",
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_\-']*", text.lower())


def add_post_features(record: dict[str, Any], sparse_row: dict[str, str]) -> None:
    prediction = str(sparse_row.get("prediction", "") or "").strip()
    request_text = str(record.get("text", ""))
    pred_words = words(prediction)
    content_words = [word for word in pred_words if word not in STOP and len(word) >= 3]
    request_set = set(words(request_text))
    normalized_prediction = " ".join(pred_words)
    normalized_request = " ".join(words(request_text))
    lowered = prediction.lower()
    refusal_terms = [
        "cannot answer", "can't answer", "cannot provide", "can't provide", "not enough information",
        "no information", "unable to", "not able to", "cannot verify", "i don't know",
    ]
    uncertainty_terms = ["maybe", "possibly", "it seems", "likely", "appears", "unclear", "not sure"]
    record.update(
        {
            "post_chars": float(len(prediction)),
            "post_words": float(len(pred_words)),
            "post_unique_word_ratio": len(set(pred_words)) / max(1, len(pred_words)),
            "post_digit_fraction": sum(char.isdigit() for char in prediction) / max(1, len(prediction)),
            "post_newlines": float(prediction.count("\n")),
            "post_generated_tokens": v450.fnum(sparse_row, "generated_tokens"),
            "post_refusal": float(any(term in lowered for term in refusal_terms)),
            "post_uncertainty": float(any(term in lowered for term in uncertainty_terms)),
            "post_empty": float(not pred_words or normalized_prediction in {"none", "unknown", "n a"}),
            "post_exact_grounded": float(
                len(normalized_prediction) >= 3 and normalized_prediction in normalized_request
            ),
            "post_content_overlap": sum(word in request_set for word in content_words) / max(1, len(content_words)),
        }
    )
    record["text"] = request_text + "\nSPARSE_OUTPUT:\n" + prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(v437.ROOT))
    parser.add_argument(
        "--frontier",
        default="outputs/riskkv_v19_v450_operator_risk_router_20260712/matched_frontier.csv",
    )
    parser.add_argument(
        "--sparse-results",
        default="outputs/riskkv_v19_v440_true_pure_source_router_v440_m20_20260712_m20_bDyn_pDyn/task_results.csv",
    )
    parser.add_argument("--max-false-safe", type=float, default=0.10)
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v451_post_probe_risk_router_20260712")
    args = parser.parse_args()

    root = Path(args.root)
    records: list[dict[str, Any]] = [dict(row) for row in v450.read_csv(root / args.frontier)]
    sparse_table = v450.by_key(v450.read_csv(root / args.sparse_results))
    matched: list[dict[str, Any]] = []
    for record in records:
        key = (str(record["task"]), str(record["sample_id"]))
        sparse_row = sparse_table.get(key)
        if sparse_row is None:
            continue
        add_post_features(record, sparse_row)
        matched.append(record)
    records = matched
    v450.NUMERIC_FEATURES = [*v450.NUMERIC_FEATURES, *POST_FEATURES]

    scenarios: list[tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = []
    scenarios.append(
        (
            "sample_holdout",
            "sample",
            "fold0",
            [row for row in records if int(row["fold"]) not in {0, 1}],
            [row for row in records if int(row["fold"]) == 1],
            [row for row in records if int(row["fold"]) == 0],
        )
    )
    for task in v437.TASKS:
        scenarios.append(
            (
                f"loto_{task}",
                "task",
                task,
                [row for row in records if row["task"] != task and int(row["fold"]) != 1],
                [row for row in records if row["task"] != task and int(row["fold"]) == 1],
                [row for row in records if row["task"] == task],
            )
        )
    for family in sorted({str(row["family"]) for row in records}):
        scenarios.append(
            (
                f"lofo_{family}",
                "family",
                family,
                [row for row in records if row["family"] != family and int(row["fold"]) != 1],
                [row for row in records if row["family"] != family and int(row["fold"]) == 1],
                [row for row in records if row["family"] == family],
            )
        )

    predictions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for scenario, holdout_type, holdout, train_rows, cal_rows, test_rows in scenarios:
        if not train_rows or not cal_rows or not test_rows:
            continue
        pred, summary, cal = v450.run_scenario(
            records,
            train_rows,
            cal_rows,
            test_rows,
            scenario,
            holdout_type,
            holdout,
            args.max_false_safe,
        )
        predictions.extend(pred)
        summaries.append(summary)
        calibration.extend(cal)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    v450.write_csv(output_dir / "router_predictions.csv", predictions)
    v450.write_csv(output_dir / "router_summary.csv", summaries)
    v450.write_csv(output_dir / "calibration.csv", calibration)
    metadata = {
        "router": "v451_post_probe_risk_feasibility",
        "diagnostic_only": True,
        "uses_task_name_as_feature": False,
        "uses_prompt_template": False,
        "post_features": POST_FEATURES,
        "probe_execution_cost_included_in_reported_speed": False,
        "samples": len(records),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
