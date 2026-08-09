#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

import train_lbv2_budget_change_router_20260714 as router_lib


def make_records(
    dataset_path: Path,
    base_paths: list[Path],
    b2048_paths: list[Path],
    full_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    raw_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_index = {str(row.get("_id", "")): row for row in raw_rows if isinstance(row, dict)}
    base = router_lib.read_method(base_paths, "ours_page_gather")
    b2048 = router_lib.read_method(b2048_paths, "ours_page_gather")
    full = router_lib.read_method(full_paths or [], "full_kv")
    sample_ids = sorted(raw_index.keys() & base.keys() & b2048.keys())
    if full_paths:
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in full]
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        raw = raw_index[sample_id]
        base_row = base[sample_id]
        record = {
            "sample_id": sample_id,
            "text": router_lib.query_text(raw),
            "categorical": {
                f"domain={raw.get('domain', '')}": 1.0,
                f"sub_domain={raw.get('sub_domain', '')}": 1.0,
                f"operator={base_row.get('ours_operator_mode', '')}": 1.0,
            },
            "numeric": router_lib.numeric_features(base_row, raw),
            "budget_changes_prediction": int(
                router_lib.prediction(base_row) != router_lib.prediction(b2048[sample_id])
            ),
            "base": base_row,
            "b2048": b2048[sample_id],
        }
        if full_paths:
            record["full"] = full[sample_id]
        records.append(record)
    return records


def bootstrap_score_ratio(
    records: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
    samples: int = 2000,
) -> dict[str, float]:
    choose_b2048 = probabilities >= threshold
    selected_scores = np.asarray(
        [
            router_lib.number(record["b2048"] if use_b2048 else record["base"], "score")
            for record, use_b2048 in zip(records, choose_b2048)
        ],
        dtype=np.float64,
    )
    full_scores = np.asarray([router_lib.number(record["full"], "score") for record in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(records), size=len(records))
        denominator = float(full_scores[indices].mean())
        if denominator > 0:
            ratios.append(float(selected_scores[indices].mean()) / denominator)
    if not ratios:
        return {"p05": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "p05": float(np.quantile(ratios, 0.05)),
        "median": float(np.quantile(ratios, 0.50)),
        "p95": float(np.quantile(ratios, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--train-base", nargs="+", required=True, type=Path)
    parser.add_argument("--train-b2048", nargs="+", required=True, type=Path)
    parser.add_argument("--calibration-dataset", required=True, type=Path)
    parser.add_argument("--calibration-basefull", nargs="+", required=True, type=Path)
    parser.add_argument("--calibration-b2048", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    args = parser.parse_args()

    train_records = make_records(args.train_dataset, args.train_base, args.train_b2048)
    calibration_records = make_records(
        args.calibration_dataset,
        args.calibration_basefull,
        args.calibration_b2048,
        args.calibration_basefull,
    )
    if not train_records or not calibration_records:
        raise RuntimeError(
            f"Missing frozen records: train={len(train_records)} calibration={len(calibration_records)}"
        )
    train_labels = np.asarray(
        [record["budget_changes_prediction"] for record in train_records], dtype=np.int64
    )
    if len(np.unique(train_labels)) < 2:
        raise RuntimeError("Training budget-change labels contain only one class")

    train_indices = np.arange(len(train_records))
    calibration_indices = np.arange(len(calibration_records))
    parts = router_lib.fit_feature_parts(train_records, train_indices)
    x_train = router_lib.transform(train_records, train_indices, parts)
    x_calibration = router_lib.transform(calibration_records, calibration_indices, parts)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=args.seed,
    )
    model.fit(x_train, train_labels)
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_calibration)[:, class_index]

    thresholds = [round(value, 2) for value in np.linspace(0.01, 0.99, 99)]
    frontier: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = router_lib.evaluate_threshold(calibration_records, probabilities, threshold)
        row["bootstrap_score_over_full"] = bootstrap_score_ratio(
            calibration_records, probabilities, threshold, args.seed + int(threshold * 1000)
        )
        frontier.append(row)
    feasible = [row for row in frontier if row["score_over_full"] + 1e-12 >= args.quality_ratio]
    selected = min(feasible, key=lambda row: (row["mean_kv_ratio"], -row["score"])) if feasible else max(
        frontier, key=lambda row: (row["score_over_full"], -row["mean_kv_ratio"])
    )

    bundle = {
        "schema": "lbv2_budget_change_router_v1",
        "feature_parts": parts,
        "model": model,
        "threshold": selected["threshold"],
        "base_budget_action": "base",
        "expanded_budget_action": "b2048",
        "numeric_keys": list(router_lib.NUMERIC_KEYS),
        "training_samples": len(train_records),
        "calibration_samples": len(calibration_records),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "router.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    payload = {
        "protocol": "frozen router_train fit and router_calibration threshold selection",
        "training_samples": len(train_records),
        "training_change_rate": float(train_labels.mean()),
        "calibration_samples": len(calibration_records),
        "calibration_change_rate": float(
            np.mean([record["budget_changes_prediction"] for record in calibration_records])
        ),
        "quality_ratio_constraint": args.quality_ratio,
        "selected_operating_point": selected,
        "frontier": frontier,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "calibration_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "budget_changes_prediction", "change_probability", "domain", "sub_domain"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, probability in zip(calibration_records, probabilities):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "budget_changes_prediction": record["budget_changes_prediction"],
                    "change_probability": probability,
                    "domain": record["full"].get("domain", ""),
                    "sub_domain": record["full"].get("sub_domain", ""),
                }
            )
    with (args.output_dir / "frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_rows = [
            {
                **{key: value for key, value in row.items() if key != "bootstrap_score_over_full"},
                **{
                    f"bootstrap_ratio_{key}": value
                    for key, value in row["bootstrap_score_over_full"].items()
                },
            }
            for row in frontier
        ]
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "frontier"}, indent=2))


if __name__ == "__main__":
    main()
