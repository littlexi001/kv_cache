from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from train_shifted_physical_budget_router_20260715 import (
    FEATURE_NAMES,
    load_json,
    load_training_rows,
    make_regressor,
    parse_triplet,
    shifted_feature_vector,
    regressor_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one ordinal score and calibrate a three-action budget lattice."
    )
    parser.add_argument("--triplet", action="append", required=True)
    parser.add_argument(
        "--calibration_quartet",
        action="append",
        required=True,
        metavar="LOW,MID,HIGH,FULL",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target_retention", type=float, default=0.95)
    parser.add_argument("--low_fraction", type=float, default=0.01)
    parser.add_argument("--mid_fraction", type=float, default=0.015)
    parser.add_argument("--high_fraction", type=float, default=0.02)
    parser.add_argument("--low_stream_group_size", type=int, default=2)
    parser.add_argument("--mid_stream_group_size", type=int, default=2)
    parser.add_argument("--high_stream_group_size", type=int, default=1)
    parser.add_argument("--mid_cost", type=float, default=1.0)
    parser.add_argument("--high_cost", type=float, default=3.0)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument(
        "--model_type", choices=("extra_trees", "ridge", "mlp"), default="extra_trees"
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def parse_quartet(spec: str) -> tuple[Path, Path, Path, Path]:
    paths = tuple(Path(value.strip()) for value in spec.split(","))
    if len(paths) != 4:
        raise ValueError("each calibration quartet must be LOW,MID,HIGH,FULL")
    return paths  # type: ignore[return-value]


def load_calibration_rows(
    quartets: list[tuple[Path, Path, Path, Path]],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    features: list[list[float]] = []
    metadata: list[dict[str, float | int]] = []
    for window_index, (low_path, mid_path, high_path, full_path) in enumerate(
        quartets
    ):
        low = load_json(low_path)
        mid = load_json(mid_path)
        high = load_json(high_path)
        full = load_json(full_path)
        arrays = [
            list(map(float, payload["token_nll"]))
            for payload in (low, mid, high, full)
        ]
        causal = low.get("causal_router_features", [])
        if len({len(values) for values in [*arrays, causal]}) != 1:
            raise ValueError(f"unaligned calibration arrays in {low_path}")
        low_nll, mid_nll, high_nll, full_nll = arrays
        for target_index in range(1, len(low_nll)):
            features.append(
                shifted_feature_vector(
                    causal[target_index - 1],
                    causal[target_index],
                    target_index,
                    len(low_nll),
                )
            )
            metadata.append(
                {
                    "window_index": window_index,
                    "target_index": target_index,
                    "low_nll": low_nll[target_index],
                    "mid_nll": mid_nll[target_index],
                    "high_nll": high_nll[target_index],
                    "full_nll": full_nll[target_index],
                }
            )
    return np.asarray(features, dtype=np.float32), metadata


def calibrated_actions(
    predictions: np.ndarray,
    mid_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    actions = np.zeros(predictions.shape, dtype=np.int8)
    actions[predictions >= mid_threshold] = 1
    actions[predictions >= high_threshold] = 2
    return actions


def calibrate_two_thresholds(
    predictions: np.ndarray,
    metadata: list[dict[str, float | int]],
    target_retention: float,
    mid_cost: float,
    high_cost: float,
) -> tuple[float, float, dict[str, float | int]]:
    unique = sorted(set(map(float, predictions)), reverse=True)
    thresholds = [math.inf, *unique]
    full_nll = np.asarray([float(row["full_nll"]) for row in metadata])
    best: tuple[float, float, dict[str, float | int]] | None = None
    best_cost = math.inf
    for mid_index, mid_threshold in enumerate(thresholds):
        for high_threshold in thresholds[: mid_index + 1]:
            actions = calibrated_actions(
                predictions, mid_threshold, high_threshold
            )
            mixed_nll = np.asarray(
                [
                    float(row[("low_nll", "mid_nll", "high_nll")[action]])
                    for row, action in zip(metadata, actions)
                ]
            )
            retention = math.exp(float(full_nll.mean() - mixed_nll.mean()))
            if retention < target_retention:
                continue
            mid_count = int((actions == 1).sum())
            high_count = int((actions == 2).sum())
            cost = mid_cost * mid_count + high_cost * high_count
            if cost >= best_cost:
                continue
            best_cost = cost
            best = (
                float(mid_threshold),
                float(high_threshold),
                {
                    "low_count": int((actions == 0).sum()),
                    "mid_count": mid_count,
                    "high_count": high_count,
                    "mid_rate": mid_count / len(actions),
                    "high_rate": high_count / len(actions),
                    "quality_retention": retention,
                    "mixed_ppl": math.exp(float(mixed_nll.mean())),
                    "full_ppl": math.exp(float(full_nll.mean())),
                    "routing_cost": float(cost),
                },
            )
    if best is None:
        raise RuntimeError("the three-action lattice cannot meet target retention")
    return best


def main() -> None:
    args = parse_args()
    if not 0 < args.low_fraction < args.mid_fraction < args.high_fraction <= 1:
        raise ValueError(
            "expected 0 < low_fraction < mid_fraction < high_fraction <= 1"
        )
    if min(
        args.low_stream_group_size,
        args.mid_stream_group_size,
        args.high_stream_group_size,
    ) < 1:
        raise ValueError("stream group sizes must be positive")
    training_triplets = [parse_triplet(spec) for spec in args.triplet]
    calibration_quartets = [
        parse_quartet(spec) for spec in args.calibration_quartet
    ]
    features, gains, _ = load_training_rows(training_triplets)
    model = make_regressor(args)
    model.fit(features, gains)
    calibration_features, calibration_metadata = load_calibration_rows(
        calibration_quartets
    )
    calibration_predictions = model.predict(calibration_features)
    mid_threshold, high_threshold, calibration = calibrate_two_thresholds(
        calibration_predictions,
        calibration_metadata,
        args.target_retention,
        args.mid_cost,
        args.high_cost,
    )
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "shifted_causal": True,
        "ordinal_three_action": True,
        "low_fraction": args.low_fraction,
        "mid_fraction": args.mid_fraction,
        "high_fraction": args.high_fraction,
        "low_stream_group_size": args.low_stream_group_size,
        "mid_stream_group_size": args.mid_stream_group_size,
        "high_stream_group_size": args.high_stream_group_size,
        "mid_threshold": mid_threshold,
        "high_threshold": high_threshold,
        "action_lattice": {
            "low_fraction": args.low_fraction,
            "mid_fraction": args.mid_fraction,
            "high_fraction": args.high_fraction,
            "low_stream_group_size": args.low_stream_group_size,
            "mid_stream_group_size": args.mid_stream_group_size,
            "high_stream_group_size": args.high_stream_group_size,
        },
        "target_retention": args.target_retention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report: dict[str, Any] = {
        "status": "trained_computer_calibrated_space_not_independent_test",
        "training_samples": int(features.shape[0]),
        "model_type": args.model_type,
        "calibration_samples": len(calibration_metadata),
        "train_gain_positive_rate": float((gains > 0).mean()),
        "mid_threshold": mid_threshold,
        "high_threshold": high_threshold,
        "calibration": calibration,
        "feature_importance": regressor_feature_importance(model),
        "training_triplets": [
            [str(path) for path in paths] for paths in training_triplets
        ],
        "calibration_quartets": [
            [str(path) for path in paths] for paths in calibration_quartets
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
