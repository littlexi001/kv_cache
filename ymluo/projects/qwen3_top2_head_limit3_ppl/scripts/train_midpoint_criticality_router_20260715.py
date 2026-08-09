from __future__ import annotations

import argparse
import json
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


def metadata_key(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["topic"]), int(row["window"]), int(row["target_index"])


def load_hidden(root: Path) -> dict[tuple[str, int, int], np.ndarray]:
    output = {}
    for path in sorted(root.glob("*/hidden_states.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = list(payload["metadata"])
        hidden = payload["hidden_states"].float().numpy()
        if hidden.ndim != 3 or hidden.shape[0] != len(metadata):
            raise RuntimeError(f"expected [tokens, layers, hidden] in {path}, got {hidden.shape}")
        for row, value in zip(metadata, hidden, strict=True):
            key = metadata_key(row)
            if key in output:
                raise RuntimeError(f"duplicate hidden key: {key}")
            output[key] = value
    if not output:
        raise RuntimeError(f"no hidden_states.pt under {root}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--hidden_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--hidden_layers", default="8,16,24,32")
    parser.add_argument("--conformal_level", type=float, default=0.95)
    parser.add_argument("--safe_delta_nll", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    layer_numbers = [int(item) for item in args.hidden_layers.split(",") if item.strip()]
    records = load_dataset(args.scan_root, args.reference_root)
    hidden_by_key = load_hidden(args.hidden_root)
    hidden = np.asarray([hidden_by_key[record["key"]] for record in records], dtype=np.float32)
    if hidden.shape[1] != len(layer_numbers):
        raise RuntimeError("hidden layer count does not match --hidden_layers")
    windows = np.asarray([record["key"][1] for record in records])
    train_mask = windows == 0
    calibration_mask = windows == 1
    test_mask = windows == 2
    calibration_records = [record for record, selected in zip(records, calibration_mask, strict=True) if selected]
    test_records = [record for record, selected in zip(records, test_mask, strict=True) if selected]
    reports = []

    for layer_index, layer_number in enumerate(layer_numbers):
        layer_hidden = hidden[:, layer_index, :]
        scaler = StandardScaler().fit(layer_hidden[train_mask])
        scaled = scaler.transform(layer_hidden)
        component_count = min(64, int(train_mask.sum()) - 1, scaled.shape[1])
        pca = PCA(n_components=component_count, svd_solver="randomized", random_state=args.seed)
        projected = pca.fit_transform(scaled[train_mask])
        projected_all = pca.transform(scaled)
        action_models = {}
        action_offsets = {}
        selected_model_names = {}
        for action in ACTIONS:
            target = np.asarray([record["deltas"][action] for record in records], dtype=np.float64)
            candidates = []
            for alpha in (10.0, 100.0, 1000.0, 10000.0):
                model = Ridge(alpha=alpha, solver="lsqr")
                model.fit(scaled[train_mask], target[train_mask])
                candidates.append((f"ridge_a{alpha:g}", model, scaled))
            extra = ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=args.seed + layer_number + int(float(action) * 10000),
                n_jobs=-1,
            )
            extra.fit(projected, target[train_mask])
            candidates.append(("pca64_extra", extra, projected_all))
            name, model, model_features = min(
                candidates,
                key=lambda candidate: float(
                    np.mean(
                        np.abs(
                            target[calibration_mask]
                            - candidate[1].predict(candidate[2][calibration_mask])
                        )
                    )
                ),
            )
            calibration_prediction = model.predict(model_features[calibration_mask])
            residual = target[calibration_mask] - calibration_prediction
            action_models[action] = (model, model_features)
            action_offsets[action] = float(
                np.quantile(residual, args.conformal_level, method="higher")
            )
            selected_model_names[action] = name

        def route(mask: np.ndarray) -> list[str]:
            indices = np.flatnonzero(mask)
            upper = {
                action: model.predict(features[indices]) + action_offsets[action]
                for action, (model, features) in action_models.items()
            }
            selected_actions = []
            for row in range(len(indices)):
                selected = "rerank"
                for action in ACTIONS:
                    if upper[action][row] <= args.safe_delta_nll:
                        selected = action
                        break
                selected_actions.append(selected)
            return selected_actions

        calibration_actions = route(calibration_mask)
        test_actions = route(test_mask)
        reports.append(
            {
                "gate_after_layer": layer_number,
                "selected_models": selected_model_names,
                "conformal_offsets": action_offsets,
                "calibration": summarize_policy(calibration_records, calibration_actions),
                "independent_test": summarize_policy(test_records, test_actions),
            }
        )

    output = {
        "protocol": {
            "train_windows": [0],
            "calibration_windows": [1],
            "test_windows": [2],
            "hidden_layers": layer_numbers,
            "conformal_level": args.conformal_level,
            "safe_delta_nll": args.safe_delta_nll,
            "note": "Offline action composition is a predictability upper bound; layer-mixed execution remains to be measured.",
        },
        "layers": reports,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
