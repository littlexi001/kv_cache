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
    make_regressor,
    regressor_feature_importance,
)
from train_shifted_three_action_router_20260715 import (
    calibrate_two_thresholds,
    calibrated_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ordinal router from on-policy counterfactual probes."
    )
    parser.add_argument("--train", action="append", required=True, type=Path)
    parser.add_argument("--calibration", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--target_kind",
        choices=("low_high_gain", "teacher_action"),
        default="low_high_gain",
    )
    parser.add_argument("--target_retention_to_high", type=float, default=0.995)
    parser.add_argument(
        "--calibration_objective",
        choices=("teacher_recall", "nll_retention", "joint_safety"),
        default="teacher_recall",
    )
    parser.add_argument("--minimum_required_action_recall", type=float, default=0.95)
    parser.add_argument("--require_full_reference_labels", action="store_true")
    parser.add_argument("--mid_cost", type=float, default=1.0)
    parser.add_argument("--high_cost", type=float, default=3.0)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument(
        "--model_type", choices=("extra_trees", "ridge", "mlp"), default="mlp"
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_counterfactual_rows(
    paths: list[Path],
    target_kind: str,
    require_full_reference_labels: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, float | int]],
    dict[str, Any],
]:
    features: list[list[float]] = []
    targets: list[float] = []
    metadata: list[dict[str, float | int]] = []
    action_config: dict[str, Any] | None = None
    for window_index, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            require_full_reference_labels
            and payload.get("teacher_reference") != "full_kv_nll"
        ):
            raise ValueError(f"counterfactual labels are not FullKV-referenced: {path}")
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError(f"feature schema mismatch in {path}")
        current_config = {
            "action_fractions": list(map(float, payload["action_fractions"])),
            "action_stream_group_sizes": list(
                map(int, payload["action_stream_group_sizes"])
            ),
        }
        if action_config is None:
            action_config = current_config
        elif current_config != action_config:
            raise ValueError("all counterfactual files must use one action lattice")
        vectors = payload["feature_vectors"]
        losses = payload["counterfactual_nll"]
        reference_losses = payload.get(
            "counterfactual_reference_nll",
            [float(action_nll[2]) for action_nll in losses],
        )
        teacher_actions = payload["teacher_actions"]
        if not len(vectors) == len(losses) == len(reference_losses) == len(teacher_actions):
            raise ValueError(f"unaligned counterfactual arrays in {path}")
        for target_index, (vector, action_nll, reference_nll, teacher_action) in enumerate(
            zip(vectors, losses, reference_losses, teacher_actions), start=1
        ):
            if len(vector) != len(FEATURE_NAMES) or len(action_nll) != 3:
                raise ValueError(f"invalid counterfactual row in {path}")
            low_nll, mid_nll, high_nll = map(float, action_nll)
            features.append(list(map(float, vector)))
            targets.append(
                low_nll - high_nll
                if target_kind == "low_high_gain"
                else float(teacher_action)
            )
            metadata.append(
                {
                    "window_index": window_index,
                    "target_index": target_index,
                    "low_nll": low_nll,
                    "mid_nll": mid_nll,
                    "high_nll": high_nll,
                    "full_nll": float(reference_nll),
                    "teacher_action": int(teacher_action),
                }
            )
    if action_config is None:
        raise ValueError("at least one counterfactual file is required")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
        metadata,
        action_config,
    )


def calibrate_teacher_recall(
    predictions: np.ndarray,
    metadata: list[dict[str, float | int]],
    minimum_recall: float,
    mid_cost: float,
    high_cost: float,
    minimum_retention: float | None = None,
) -> tuple[float, float, dict[str, float | int]]:
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    teacher = np.asarray([int(row["teacher_action"]) for row in metadata])
    unique = sorted(set(map(float, predictions)), reverse=True)
    thresholds = [math.inf, *unique]
    best: tuple[float, float, dict[str, float | int]] | None = None
    best_cost = math.inf
    requires_mid = teacher >= 1
    requires_high = teacher == 2
    full_nll = np.asarray([float(row["full_nll"]) for row in metadata])
    for mid_index, mid_threshold in enumerate(thresholds):
        for high_threshold in thresholds[: mid_index + 1]:
            actions = calibrated_actions(
                predictions, mid_threshold, high_threshold
            )
            mid_recall = (
                float((actions[requires_mid] >= 1).mean())
                if bool(requires_mid.any())
                else 1.0
            )
            high_recall = (
                float((actions[requires_high] == 2).mean())
                if bool(requires_high.any())
                else 1.0
            )
            if min(mid_recall, high_recall) < minimum_recall:
                continue
            mixed_nll = np.asarray(
                [
                    float(row[("low_nll", "mid_nll", "high_nll")[action]])
                    for row, action in zip(metadata, actions)
                ]
            )
            quality_retention = math.exp(
                float(full_nll.mean() - mixed_nll.mean())
            )
            if (
                minimum_retention is not None
                and quality_retention < minimum_retention
            ):
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
                    "required_mid_recall": mid_recall,
                    "required_high_recall": high_recall,
                    "exact_action_accuracy": float((actions == teacher).mean()),
                    "quality_retention": quality_retention,
                    "full_ppl": math.exp(float(full_nll.mean())),
                    "mixed_ppl": math.exp(float(mixed_nll.mean())),
                    "routing_cost": float(cost),
                },
            )
    if best is None:
        raise RuntimeError("no thresholds satisfy required-action recall")
    return best


def main() -> None:
    args = parse_args()
    train_features, train_targets, train_metadata, action_config = (
        load_counterfactual_rows(
            args.train,
            args.target_kind,
            args.require_full_reference_labels,
        )
    )
    model = make_regressor(args)
    model.fit(train_features, train_targets)
    train_predictions = model.predict(train_features)

    calibration_paths = args.calibration or args.train
    calibration_features, _, calibration_metadata, calibration_config = (
        load_counterfactual_rows(
            calibration_paths,
            args.target_kind,
            args.require_full_reference_labels,
        )
    )
    if calibration_config != action_config:
        raise ValueError("training and calibration action lattices differ")
    calibration_predictions = model.predict(calibration_features)
    if args.calibration_objective in {"teacher_recall", "joint_safety"}:
        mid_threshold, high_threshold, calibration = calibrate_teacher_recall(
            calibration_predictions,
            calibration_metadata,
            args.minimum_required_action_recall,
            args.mid_cost,
            args.high_cost,
            minimum_retention=(
                args.target_retention_to_high
                if args.calibration_objective == "joint_safety"
                else None
            ),
        )
    else:
        mid_threshold, high_threshold, calibration = calibrate_two_thresholds(
            calibration_predictions,
            calibration_metadata,
            args.target_retention_to_high,
            args.mid_cost,
            args.high_cost,
        )
    fractions = action_config["action_fractions"]
    group_sizes = action_config["action_stream_group_sizes"]
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "shifted_causal": True,
        "ordinal_three_action": True,
        "counterfactual_onpolicy_teacher": True,
        "teacher_reference": (
            "full_kv_nll"
            if args.require_full_reference_labels
            else "not_enforced"
        ),
        "low_fraction": fractions[0],
        "mid_fraction": fractions[1],
        "high_fraction": fractions[2],
        "low_stream_group_size": group_sizes[0],
        "mid_stream_group_size": group_sizes[1],
        "high_stream_group_size": group_sizes[2],
        "mid_threshold": mid_threshold,
        "high_threshold": high_threshold,
        "target_retention_to_high": args.target_retention_to_high,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)

    teacher_actions = np.asarray(
        [int(row["teacher_action"]) for row in train_metadata]
    )
    report = {
        "status": (
            "counterfactual_train_with_heldout_calibration_not_independent_test"
            if args.calibration
            else "counterfactual_training_fit_only_not_independent_validation"
        ),
        "model_type": args.model_type,
        "target_kind": args.target_kind,
        "calibration_objective": args.calibration_objective,
        "teacher_reference": (
            "full_kv_nll"
            if args.require_full_reference_labels
            else "not_enforced"
        ),
        "training_samples": int(len(train_metadata)),
        "calibration_samples": int(len(calibration_metadata)),
        "train_prediction_correlation": float(
            np.corrcoef(train_targets, train_predictions)[0, 1]
        ),
        "teacher_action_rates": {
            name: float((teacher_actions == action).mean())
            for action, name in enumerate(("low", "mid", "high"))
        },
        "action_lattice": action_config,
        "mid_threshold": mid_threshold,
        "high_threshold": high_threshold,
        "calibration": calibration,
        "feature_importance": regressor_feature_importance(model),
        "training_files": [str(path) for path in args.train],
        "calibration_files": [str(path) for path in calibration_paths],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
