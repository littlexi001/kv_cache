from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "previous_top1_probability",
    "previous_top1_top2_logit_margin",
    "previous_entropy",
    "previous_normalized_entropy",
    "previous_retrieval_feature_valid",
    "previous_retrieval_score_spread",
    "previous_retrieval_candidate_stability",
    "previous_retrieval_refreshed_fraction",
    "previous_log1p_top1_history_frequency",
    "previous_top1_is_digit_token",
    "previous_top1_is_number_word",
    "previous_top1_is_numeric_token",
    "previous_top1_is_alpha_token",
    "previous_top1_is_punctuation_token",
    "current_log1p_input_history_frequency",
    "current_input_is_digit_token",
    "current_input_is_number_word",
    "current_input_is_numeric_token",
    "current_input_is_alpha_token",
    "current_input_is_punctuation_token",
    "relative_target_position",
    "log1p_history_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shifted-causal 1%/1.5% physical KV budget router."
    )
    parser.add_argument(
        "--triplet",
        action="append",
        required=True,
        metavar="LOW,HIGH,FULL",
        help="May be repeated; each entry contains aligned result JSON files.",
    )
    parser.add_argument(
        "--calibration_triplet",
        action="append",
        metavar="LOW,HIGH,FULL",
        help="Held-out windows used only to calibrate the routing threshold.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target_retention", type=float, default=0.95)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument(
        "--model_type", choices=("extra_trees", "ridge", "mlp"), default="extra_trees"
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_triplet(spec: str) -> tuple[Path, Path, Path]:
    paths = [Path(value.strip()) for value in spec.split(",")]
    if len(paths) != 3:
        raise ValueError("each --triplet must be LOW,HIGH,FULL")
    return paths[0], paths[1], paths[2]


def shifted_feature_vector(
    previous: dict[str, Any],
    current: dict[str, Any],
    target_index: int,
    target_count: int,
) -> list[float]:
    values = [
        float(previous["top1_probability"]),
        float(previous["top1_top2_logit_margin"]),
        float(previous["entropy"]),
        float(previous["normalized_entropy"]),
        float(previous.get("retrieval_feature_valid", 0.0)),
        float(previous.get("retrieval_score_spread", 0.0)),
        float(previous.get("retrieval_candidate_stability", 0.0)),
        float(previous.get("retrieval_refreshed_fraction", 0.0)),
        math.log1p(int(previous["top1_history_frequency"])),
        float(previous["top1_is_digit_token"]),
        float(previous["top1_is_number_word"]),
        float(previous["top1_is_numeric_token"]),
        float(previous["top1_is_alpha_token"]),
        float(previous["top1_is_punctuation_token"]),
        math.log1p(int(current["input_history_frequency"])),
        float(current["input_is_digit_token"]),
        float(current["input_is_number_word"]),
        float(current["input_is_numeric_token"]),
        float(current["input_is_alpha_token"]),
        float(current["input_is_punctuation_token"]),
        float(target_index) / max(1, target_count - 1),
        math.log1p(int(current["history_tokens"])),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("shifted feature schema length mismatch")
    return values


def load_training_rows(
    triplets: list[tuple[Path, Path, Path]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[list[float]] = []
    gains: list[float] = []
    metadata: list[dict[str, Any]] = []
    for window_index, (low_path, high_path, full_path) in enumerate(triplets):
        low = load_json(low_path)
        high = load_json(high_path)
        full = load_json(full_path)
        low_nll = list(map(float, low["token_nll"]))
        high_nll = list(map(float, high["token_nll"]))
        full_nll = list(map(float, full["token_nll"]))
        causal = low.get("causal_router_features", [])
        if not (
            len(low_nll)
            == len(high_nll)
            == len(full_nll)
            == len(causal)
        ):
            raise ValueError(
                f"unaligned token arrays or missing router features in {low_path}"
            )
        for target_index in range(1, len(low_nll)):
            features.append(
                shifted_feature_vector(
                    causal[target_index - 1],
                    causal[target_index],
                    target_index,
                    len(low_nll),
                )
            )
            gains.append(low_nll[target_index] - high_nll[target_index])
            metadata.append(
                {
                    "window_index": window_index,
                    "target_index": target_index,
                    "low_nll": low_nll[target_index],
                    "high_nll": high_nll[target_index],
                    "full_nll": full_nll[target_index],
                }
            )
    return np.asarray(features, dtype=np.float32), np.asarray(gains), metadata


def calibrate_threshold(
    predictions: np.ndarray,
    metadata: list[dict[str, Any]],
    target_retention: float,
) -> tuple[float, dict[str, float | int]]:
    thresholds = [math.inf, *sorted(set(map(float, predictions)), reverse=True)]
    best: tuple[float, dict[str, float | int]] | None = None
    for threshold in thresholds:
        use_high = predictions >= threshold
        mixed_nll = np.asarray(
            [
                row["high_nll"] if high else row["low_nll"]
                for row, high in zip(metadata, use_high)
            ]
        )
        full_nll = np.asarray([row["full_nll"] for row in metadata])
        retention = math.exp(float(full_nll.mean() - mixed_nll.mean()))
        if retention < target_retention:
            continue
        candidate = {
            "high_positions": int(use_high.sum()),
            "high_rate": float(use_high.mean()),
            "mixed_ppl": math.exp(float(mixed_nll.mean())),
            "full_ppl": math.exp(float(full_nll.mean())),
            "quality_retention": retention,
        }
        if best is None or candidate["high_positions"] < best[1]["high_positions"]:
            best = (float(threshold), candidate)
    if best is None:
        raise RuntimeError("the high action cannot satisfy target retention")
    return best


def make_regressor(args: argparse.Namespace) -> Any:
    if args.model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if args.model_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="lbfgs",
                alpha=0.01,
                max_iter=2000,
                random_state=args.seed,
            ),
        )
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )


def regressor_feature_importance(model: Any) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
        values = np.abs(np.asarray(model.named_steps["ridge"].coef_))
    elif hasattr(model, "named_steps") and "mlpregressor" in model.named_steps:
        values = np.abs(model.named_steps["mlpregressor"].coefs_[0]).sum(axis=1)
    else:
        return {}
    values = values / max(float(values.sum()), 1.0e-12)
    return {
        name: float(importance)
        for name, importance in sorted(
            zip(FEATURE_NAMES, values), key=lambda item: item[1], reverse=True
        )
    }


def main() -> None:
    args = parse_args()
    triplets = [parse_triplet(spec) for spec in args.triplet]
    calibration_triplets = [
        parse_triplet(spec) for spec in (args.calibration_triplet or [])
    ]
    features, gains, metadata = load_training_rows(triplets)
    model = make_regressor(args)
    model.fit(features, gains)
    predictions = model.predict(features)
    if calibration_triplets:
        calibration_features, _, calibration_metadata = load_training_rows(
            calibration_triplets
        )
        calibration_predictions = model.predict(calibration_features)
    else:
        calibration_metadata = metadata
        calibration_predictions = predictions
    threshold, calibration = calibrate_threshold(
        calibration_predictions, calibration_metadata, args.target_retention
    )
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "low_fraction": 0.01,
        "high_fraction": 0.015,
        "stream_group_size": 2,
        "target_retention": args.target_retention,
        "shifted_causal": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        "status": (
            "held_out_threshold_calibration_not_independent_test"
            if calibration_triplets
            else "training_fit_only_not_independent_validation"
        ),
        "sample_count": int(features.shape[0]),
        "model_type": args.model_type,
        "calibration_sample_count": int(len(calibration_metadata)),
        "feature_names": FEATURE_NAMES,
        "gain_mean": float(gains.mean()),
        "gain_positive_rate": float((gains > 0).mean()),
        "train_prediction_correlation": float(np.corrcoef(gains, predictions)[0, 1]),
        "feature_importance": regressor_feature_importance(model),
        "threshold": threshold,
        "calibration": calibration,
        "triplets": [[str(path) for path in paths] for paths in triplets],
        "calibration_triplets": [
            [str(path) for path in paths] for paths in calibration_triplets
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
