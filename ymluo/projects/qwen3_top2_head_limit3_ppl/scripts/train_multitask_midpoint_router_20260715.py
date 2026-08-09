from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train_provisional_criticality_router_20260715 import (
    ACTIONS,
    load_dataset,
    summarize_policy,
)


PRIVILEGED_TARGETS = (
    "top1_probability",
    "normalized_entropy",
    "top1_top2_logit_margin",
    "retained_mass_mean",
)


class EvidenceRouter(torch.nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width // 2),
            torch.nn.GELU(),
            torch.nn.Linear(width // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def metadata_key(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["topic"]), int(row["window"]), int(row["target_index"])


def load_hidden_and_teacher(
    root: Path,
) -> dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]]:
    output = {}
    for path in sorted(root.glob("*/hidden_states.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hidden = payload["hidden_states"].float().numpy()
        metadata = list(payload["metadata"])
        for row, value in zip(metadata, hidden, strict=True):
            teacher = np.asarray(
                [float(row[name]) for name in PRIVILEGED_TARGETS], dtype=np.float32
            )
            output[metadata_key(row)] = (value, teacher)
    if not output:
        raise RuntimeError(f"no hidden states under {root}")
    return output


def train_candidate(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    width: int,
    auxiliary: bool,
    seed: int,
    max_epochs: int,
) -> tuple[EvidenceRouter, float, int]:
    torch.manual_seed(seed)
    output_dim = targets.shape[1] if auxiliary else len(ACTIONS)
    model = EvidenceRouter(features.shape[1], width, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    target_tensor = torch.from_numpy(targets.astype(np.float32))
    best_state = None
    best_error = float("inf")
    best_epoch = 0
    patience = 60
    for epoch in range(max_epochs):
        model.train()
        prediction = model(feature_tensor[train_mask])
        main_loss = torch.nn.functional.smooth_l1_loss(
            prediction[:, : len(ACTIONS)],
            target_tensor[train_mask, : len(ACTIONS)],
        )
        loss = main_loss
        if auxiliary:
            auxiliary_loss = torch.nn.functional.smooth_l1_loss(
                prediction[:, len(ACTIONS) :],
                target_tensor[train_mask, len(ACTIONS) :],
            )
            loss = loss + 0.25 * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            calibration_prediction = model(feature_tensor[calibration_mask])[
                :, : len(ACTIONS)
            ]
            calibration_error = float(
                torch.mean(
                    torch.abs(
                        calibration_prediction
                        - target_tensor[calibration_mask, : len(ACTIONS)]
                    )
                ).item()
            )
        if calibration_error < best_error - 1.0e-5:
            best_error = calibration_error
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone() for name, parameter in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("router training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_error, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--hidden_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hidden_layers", default="8,16,24,32")
    parser.add_argument("--conformal_levels", default="0.5,0.8,0.9")
    parser.add_argument("--safe_delta_nll", type=float, default=0.05)
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    layer_numbers = [int(item) for item in args.hidden_layers.split(",") if item]
    conformal_levels = [float(item) for item in args.conformal_levels.split(",") if item]
    records = load_dataset(args.scan_root, args.reference_root)
    payloads = load_hidden_and_teacher(args.hidden_root)
    hidden = np.asarray([payloads[record["key"]][0] for record in records], dtype=np.float32)
    teacher = np.asarray([payloads[record["key"]][1] for record in records], dtype=np.float32)
    action_targets = np.asarray(
        [[record["deltas"][action] for action in ACTIONS] for record in records],
        dtype=np.float32,
    )
    windows = np.asarray([record["key"][1] for record in records])
    train_mask = windows == 0
    calibration_mask = windows == 1
    test_mask = windows == 2
    calibration_records = [
        record for record, selected in zip(records, calibration_mask, strict=True) if selected
    ]
    test_records = [
        record for record, selected in zip(records, test_mask, strict=True) if selected
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for layer_offset, layer_number in enumerate(layer_numbers):
        scaler = StandardScaler().fit(hidden[train_mask, layer_offset])
        scaled = scaler.transform(hidden[:, layer_offset])
        component_count = min(
            args.pca_components, int(train_mask.sum()) - 1, scaled.shape[1]
        )
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=args.seed + layer_number,
        )
        pca.fit(scaled[train_mask])
        features = pca.transform(scaled).astype(np.float32)
        raw_targets = np.concatenate((action_targets, teacher), axis=1)
        target_mean = raw_targets[train_mask].mean(axis=0)
        target_std = raw_targets[train_mask].std(axis=0).clip(min=1.0e-4)
        normalized_targets = (raw_targets - target_mean) / target_std

        candidates = []
        for auxiliary in (False, True):
            for width in (32, 64, 128):
                for seed_offset in (0, 1):
                    model, error, epoch = train_candidate(
                        features,
                        normalized_targets,
                        train_mask,
                        calibration_mask,
                        width,
                        auxiliary,
                        args.seed + 1000 * layer_number + 17 * seed_offset + width,
                        args.max_epochs,
                    )
                    candidates.append((error, auxiliary, width, seed_offset, epoch, model))
        error, auxiliary, width, seed_offset, epoch, model = min(
            candidates, key=lambda row: row[0]
        )
        model.eval()
        with torch.no_grad():
            normalized_prediction = model(torch.from_numpy(features)).numpy()
        prediction = (
            normalized_prediction[:, : len(ACTIONS)]
            * target_std[: len(ACTIONS)]
            + target_mean[: len(ACTIONS)]
        )

        level_reports = []
        for level in conformal_levels:
            offsets = []
            for action_index in range(len(ACTIONS)):
                residual = (
                    action_targets[calibration_mask, action_index]
                    - prediction[calibration_mask, action_index]
                )
                offsets.append(float(np.quantile(residual, level, method="higher")))

            def route(mask: np.ndarray) -> list[str]:
                selected_actions = []
                for row_index in np.flatnonzero(mask):
                    selected = "rerank"
                    for action_index, action in enumerate(ACTIONS):
                        upper = prediction[row_index, action_index] + offsets[action_index]
                        if upper <= args.safe_delta_nll:
                            selected = action
                            break
                    selected_actions.append(selected)
                return selected_actions

            level_reports.append(
                {
                    "conformal_level": level,
                    "offsets": dict(zip(ACTIONS, offsets, strict=True)),
                    "calibration": summarize_policy(
                        calibration_records, route(calibration_mask)
                    ),
                    "independent_test": summarize_policy(test_records, route(test_mask)),
                }
            )

        torch.save(
            {
                "layer": layer_number,
                "pca_mean": pca.mean_.astype(np.float32),
                "pca_components": pca.components_.astype(np.float32),
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
                "target_mean": target_mean.astype(np.float32),
                "target_std": target_std.astype(np.float32),
                "width": width,
                "auxiliary": auxiliary,
                "state_dict": model.state_dict(),
            },
            args.output_dir / f"router_layer{layer_number}.pt",
        )
        reports.append(
            {
                "gate_after_layer": layer_number,
                "selected_candidate": {
                    "auxiliary_distillation": auxiliary,
                    "width": width,
                    "seed_offset": seed_offset,
                    "best_epoch": epoch,
                    "calibration_normalized_mae": error,
                },
                "levels": level_reports,
            }
        )

    report = {
        "protocol": {
            "train_windows": [0],
            "calibration_windows": [1],
            "test_windows": [2],
            "inference_inputs": "midpoint hidden state only",
            "privileged_training_targets": list(PRIVILEGED_TARGETS),
            "safe_delta_nll": args.safe_delta_nll,
            "note": "Offline full-action labels remain an upper bound until layer-mixed execution is measured.",
        },
        "layers": reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
