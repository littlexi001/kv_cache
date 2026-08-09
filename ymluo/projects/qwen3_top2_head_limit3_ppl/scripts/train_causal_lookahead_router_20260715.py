from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from train_provisional_criticality_router_20260715 import (
    ACTIONS,
    load_dataset,
    summarize_policy,
)


def key(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["topic"]), int(row["window"]), int(row["target_index"])


def load_hidden_metadata(
    root: Path,
) -> dict[tuple[str, int, int], tuple[np.ndarray, dict[str, object]]]:
    output = {}
    for path in sorted(root.glob("*/hidden_states.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hidden = payload["hidden_states"].float().numpy()
        for row, value in zip(payload["metadata"], hidden, strict=True):
            output[key(row)] = (value, row)
    if not output:
        raise RuntimeError(f"no hidden state files under {root}")
    return output


def lexical_features(text: str) -> list[float]:
    stripped = text.strip()
    lowered = stripped.lower()
    function_words = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return [
        float(any(character.isdigit() for character in stripped)),
        float(bool(stripped) and not any(character.isalnum() for character in stripped)),
        float(stripped[:1].isupper()),
        float(lowered in function_words),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--hidden_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--hidden_layer", type=int, default=32)
    parser.add_argument("--hidden_layers", default="8,16,24,32")
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--conformal_levels", default="0.5,0.8,0.9")
    parser.add_argument("--safe_delta_nll", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    layer_numbers = [int(item) for item in args.hidden_layers.split(",") if item]
    layer_offset = layer_numbers.index(args.hidden_layer)
    conformal_levels = [float(item) for item in args.conformal_levels.split(",") if item]
    all_records = load_dataset(args.scan_root, args.reference_root)
    record_by_key = {record["key"]: record for record in all_records}
    hidden_metadata = load_hidden_metadata(args.hidden_root)
    records = []
    previous_hidden = []
    scalar_features = []
    for record in all_records:
        topic, window, target_index = record["key"]
        if target_index < args.lookback:
            continue
        lag_keys = [
            (topic, window, target_index - lag)
            for lag in range(args.lookback, 0, -1)
        ]
        lag_payloads = [hidden_metadata[lag_key] for lag_key in lag_keys]
        lag_records = [record_by_key[lag_key] for lag_key in lag_keys]
        hidden, previous = lag_payloads[-1]
        _, current = hidden_metadata[record["key"]]
        lag_hidden = np.asarray(
            [payload[0][layer_offset] for payload in lag_payloads], dtype=np.float32
        )
        previous_hidden.append(
            np.concatenate((hidden[layer_offset], lag_hidden.mean(axis=0)), axis=0)
        )
        temporal_scalars = []
        for (_, lag_metadata), lag_record in zip(
            lag_payloads, lag_records, strict=True
        ):
            temporal_scalars.extend(
                [
                    float(lag_metadata["top1_probability"]),
                    float(lag_metadata["normalized_entropy"]),
                    float(lag_metadata["top1_top2_logit_margin"]),
                    float(lag_metadata["retained_mass_mean"]),
                    float(lag_record["action_nll"]["0.005"]),
                ]
            )
        scalar_features.append(
            [
                *temporal_scalars,
                float(previous["top1_probability"]),
                float(previous["normalized_entropy"]),
                float(previous["top1_top2_logit_margin"]),
                float(previous["retained_mass_mean"]),
                math.log1p(float(previous["top1_history_frequency"])),
                float(int(previous["top1_id"]) == int(current["input_id"])),
                math.log1p(float(current["input_history_frequency"])),
                target_index / 255.0,
                *lexical_features(str(current["input_text"])),
            ]
        )
        records.append(record)

    previous_hidden_array = np.asarray(previous_hidden, dtype=np.float32)
    scalar_array = np.asarray(scalar_features, dtype=np.float32)
    windows = np.asarray([record["key"][1] for record in records])
    train_mask = windows == 0
    calibration_mask = windows == 1
    test_mask = windows == 2
    hidden_scaler = StandardScaler().fit(previous_hidden_array[train_mask])
    hidden_scaled = hidden_scaler.transform(previous_hidden_array)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=args.seed)
    pca.fit(hidden_scaled[train_mask])
    hidden_projected = pca.transform(hidden_scaled)
    scalar_scaler = StandardScaler().fit(scalar_array[train_mask])
    features = np.concatenate(
        (hidden_projected, scalar_scaler.transform(scalar_array)), axis=1
    )
    calibration_records = [
        record for record, selected in zip(records, calibration_mask, strict=True) if selected
    ]
    test_records = [
        record for record, selected in zip(records, test_mask, strict=True) if selected
    ]
    predictions = {}
    selected_models = {}
    for action in ACTIONS:
        target = np.asarray([record["deltas"][action] for record in records])
        candidates = []
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(features[train_mask], target[train_mask])
            candidates.append((f"ridge_a{alpha:g}", model))
        for leaf_size in (8, 16, 32):
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=leaf_size,
                max_features=0.8,
                random_state=args.seed + leaf_size + int(float(action) * 10000),
                n_jobs=-1,
            )
            model.fit(features[train_mask], target[train_mask])
            candidates.append((f"extra_leaf{leaf_size}", model))
        name, model = min(
            candidates,
            key=lambda item: float(
                np.mean(
                    np.abs(
                        target[calibration_mask]
                        - item[1].predict(features[calibration_mask])
                    )
                )
            ),
        )
        predictions[action] = model.predict(features)
        selected_models[action] = name

    level_reports = []
    for level in conformal_levels:
        offsets = {}
        for action in ACTIONS:
            target = np.asarray([record["deltas"][action] for record in records])
            residual = target[calibration_mask] - predictions[action][calibration_mask]
            offsets[action] = float(np.quantile(residual, level, method="higher"))

        def route(mask: np.ndarray) -> list[str]:
            actions = []
            for row_index in np.flatnonzero(mask):
                selected = "rerank"
                for action in ACTIONS:
                    if predictions[action][row_index] + offsets[action] <= args.safe_delta_nll:
                        selected = action
                        break
                actions.append(selected)
            return actions

        calibration_actions = route(calibration_mask)
        test_actions = route(test_mask)
        calibration_no_rerank = [
            "0.02" if action == "rerank" else action
            for action in calibration_actions
        ]
        test_no_rerank = [
            "0.02" if action == "rerank" else action for action in test_actions
        ]
        level_reports.append(
            {
                "conformal_level": level,
                "offsets": offsets,
                "calibration": summarize_policy(
                    calibration_records, calibration_actions
                ),
                "independent_test": summarize_policy(test_records, test_actions),
                "no_rerank_fallback_2pct": {
                    "calibration": summarize_policy(
                        calibration_records, calibration_no_rerank
                    ),
                    "independent_test": summarize_policy(
                        test_records, test_no_rerank
                    ),
                },
            }
        )

    report = {
        "protocol": {
            "train_windows": [0],
            "calibration_windows": [1],
            "test_windows": [2],
            "feature_timing": (
                "All features are available after token t-1 and before token t enters layer 1."
            ),
            "target_leakage": False,
            "lost_prefix_tokens_per_window": args.lookback,
            "lookback": args.lookback,
            "hidden_layer": args.hidden_layer,
            "safe_delta_nll": args.safe_delta_nll,
        },
        "selected_models": selected_models,
        "levels": level_reports,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
