from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from critical_position_router import ATTENTION_MASS_FEATURES, build_feature_vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final-hidden-state critical-position routers.")
    parser.add_argument("--budget_result_root", required=True, type=Path)
    parser.add_argument("--hidden_result_root", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--train_windows", default="0,1")
    parser.add_argument("--test_windows", default="2")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["topic"]), int(row["window"]), int(row["target_index"])


def read_budget(root: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    result: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for path in sorted(root.glob("*/token_results.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                result[key(row)][str(row["method"])] = float(row["nll"])
    return result


def read_hidden(root: Path) -> tuple[list[tuple[str, int, int]], list[dict[str, Any]], torch.Tensor]:
    keys: list[tuple[str, int, int]] = []
    metadata: list[dict[str, Any]] = []
    hidden: list[torch.Tensor] = []
    for path in sorted(root.glob("*/hidden_states.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        shard_metadata = list(payload["metadata"])
        shard_hidden = payload["hidden_states"].float()
        if len(shard_metadata) != int(shard_hidden.shape[0]):
            raise RuntimeError(f"metadata/hidden mismatch in {path}")
        keys.extend(key(row) for row in shard_metadata)
        metadata.extend(shard_metadata)
        hidden.append(shard_hidden)
    if not hidden:
        raise RuntimeError(f"no hidden_states.pt under {root}")
    return keys, metadata, torch.cat(hidden, dim=0)


def int_set(spec: str) -> set[int]:
    return {int(item.strip()) for item in spec.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    train_windows = int_set(args.train_windows)
    test_windows = int_set(args.test_windows)
    losses = read_budget(args.budget_result_root)
    keys, metadata, hidden_tensor = read_hidden(args.hidden_result_root)
    required = {"full_attention", "head_top1pct", "head_top2pct", "head_top4pct"}
    selected = [index for index, item in enumerate(keys) if required <= set(losses.get(item, {}))]
    keys = [keys[index] for index in selected]
    metadata = [metadata[index] for index in selected]
    hidden = hidden_tensor[selected].numpy().astype(np.float32)

    scalar: list[list[float]] = []
    for item, row in zip(keys, metadata):
        attention = {name: float(row[name]) for name in ATTENTION_MASS_FEATURES}
        scalar.append(
            build_feature_vector(
                logit_features={
                    "top1_probability": float(row["top1_probability"]),
                    "top1_top2_logit_margin": float(row["top1_top2_logit_margin"]),
                    "entropy": float(row["entropy"]),
                },
                attention_features=attention,
                top1_text=str(row["top1_text"]),
                top1_history_frequency=int(row["top1_history_frequency"]),
                prediction_index=int(row["target_index"]),
                prediction_horizon=256,
                topic=str(row["topic"]),
                input_text=str(row["input_text"]),
                input_history_frequency=int(row["input_history_frequency"]),
            )
        )
    scalar_array = np.asarray(scalar, dtype=np.float32)
    train = np.asarray([item[1] in train_windows for item in keys], dtype=bool)
    test = np.asarray([item[1] in test_windows for item in keys], dtype=bool)
    low = np.asarray([losses[item]["head_top1pct"] for item in keys])
    fixed = np.asarray([losses[item]["head_top2pct"] for item in keys])
    high = np.asarray([losses[item]["head_top4pct"] for item in keys])
    full = np.asarray([losses[item]["full_attention"] for item in keys])
    target = low - high

    scalar_scaler = StandardScaler().fit(scalar_array[train])
    scalar_scaled = scalar_scaler.transform(scalar_array)
    hidden_scaler = StandardScaler().fit(hidden[train])
    hidden_scaled = hidden_scaler.transform(hidden)

    candidates: list[tuple[str, Any, np.ndarray]] = []
    for alpha in [10.0, 100.0, 1000.0, 10000.0]:
        candidates.append((f"hidden_ridge_a{alpha:g}", Ridge(alpha=alpha, solver="lsqr"), hidden_scaled))
    for components in [32, 64, 128, 256]:
        pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed)
        pca.fit(hidden_scaled[train])
        projected = pca.transform(hidden_scaled)
        combined = np.concatenate((projected, scalar_scaled), axis=1)
        candidates.append(
            (
                f"pca{components}_extra",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=args.seed,
                ),
                combined,
            )
        )
        candidates.append(
            (
                f"pca{components}_gbr",
                GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=2,
                    min_samples_leaf=10,
                    learning_rate=0.03,
                    loss="huber",
                    random_state=args.seed,
                ),
                combined,
            )
        )

    reports: list[dict[str, Any]] = []
    for name, model, model_x in candidates:
        model.fit(model_x[train], target[train])
        prediction = model.predict(model_x)
        report: dict[str, Any] = {
            "model": name,
            "test_auc_benefit_gt_0p1": float(roc_auc_score(target[test] > 0.1, prediction[test])),
            "test_correlation": float(np.corrcoef(target[test], prediction[test])[0, 1]),
            "rates": [],
        }
        for rate in [0.10, 0.20, 0.30, 0.40, 0.50]:
            threshold = float(np.quantile(prediction[train], 1.0 - rate))
            use_high = prediction[test] >= threshold
            routed = np.where(use_high, high[test], low[test])
            actual_rate = float(use_high.mean())
            committed = 0.01 * (1.0 - actual_rate) + 0.04 * actual_rate
            report["rates"].append(
                {
                    "calibration_rate": rate,
                    "actual_test_high_rate": actual_rate,
                    "mean_committed_fraction": committed,
                    "offline_routed_ppl": math.exp(float(routed.mean())),
                    "ppl_ratio_vs_full": math.exp(float(routed.mean() - full[test].mean())),
                }
            )
        reports.append(report)

    output = {
        "train_tokens": int(train.sum()),
        "test_tokens": int(test.sum()),
        "test_full_ppl": math.exp(float(full[test].mean())),
        "test_fixed1_ppl": math.exp(float(low[test].mean())),
        "test_fixed2_ppl": math.exp(float(fixed[test].mean())),
        "test_fixed4_ppl": math.exp(float(high[test].mean())),
        "models": reports,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
