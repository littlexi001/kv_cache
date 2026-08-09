from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from critical_position_router import FEATURE_NAMES, build_feature_vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal low/high critical-position budget router.")
    parser.add_argument("--budget_result_root", required=True, type=Path)
    parser.add_argument("--attention_stats_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_windows", default="0,1")
    parser.add_argument("--test_windows", default="2")
    parser.add_argument("--low_method", default="head_top1pct")
    parser.add_argument("--high_method", default="head_top4pct")
    parser.add_argument("--full_method", default="full_attention")
    parser.add_argument("--fixed_reference_method", default="head_top2pct")
    parser.add_argument("--low_fraction", type=float, default=0.01)
    parser.add_argument("--high_fraction", type=float, default=0.04)
    parser.add_argument("--calibration_high_rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def int_set(spec: str) -> set[int]:
    return {int(item.strip()) for item in spec.split(",") if item.strip()}


def read_sharded_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/token_results.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no token_results.csv rows found under {root}")
    return rows


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return str(row["topic"]), int(row["window"]), int(row["target_index"])


def ppl(values: np.ndarray) -> float:
    return math.exp(float(values.mean()))


def main() -> None:
    args = parse_args()
    train_windows = int_set(args.train_windows)
    test_windows = int_set(args.test_windows)
    if train_windows & test_windows:
        raise ValueError("train_windows and test_windows must be disjoint")
    if not (0.0 < args.calibration_high_rate < 1.0):
        raise ValueError("calibration_high_rate must be in (0, 1)")

    budget_rows = read_sharded_rows(args.budget_result_root)
    stats_rows = read_sharded_rows(args.attention_stats_root)
    losses: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    stats_by_key: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in budget_rows:
        losses[key(row)][str(row["method"])] = float(row["nll"])
    for row in stats_rows:
        stats_by_key[key(row)] = row

    keys = sorted(set(losses) & set(stats_by_key))
    required = {args.low_method, args.high_method, args.full_method, args.fixed_reference_method}
    keys = [item for item in keys if required <= set(losses[item])]
    features: list[list[float]] = []
    labels: list[float] = []
    for item in keys:
        topic, window, target_index = item
        row = stats_by_key[item]
        previous = stats_by_key.get((topic, window, target_index - 1))
        input_text = "" if previous is None else str(previous["token_text"])
        input_frequency = 0 if previous is None else int(previous["history_token_frequency"])
        logit_values = {
            "top1_probability": float(row["top1_probability"]),
            "top1_top2_logit_margin": float(row["top1_top2_logit_margin"]),
            "entropy": float(row["entropy"]),
        }
        attention_values = {name: float(row[name]) for name in FEATURE_NAMES if name.startswith("retained_")}
        attention_values["top1_attention_mass_mean"] = float(row["top1_attention_mass_mean"])
        features.append(
            build_feature_vector(
                logit_features=logit_values,
                attention_features=attention_values,
                top1_text=str(row["top1_text"]),
                top1_history_frequency=int(row["top1_history_frequency"]),
                prediction_index=target_index,
                prediction_horizon=256,
                topic=topic,
                input_text=input_text,
                input_history_frequency=input_frequency,
            )
        )
        labels.append(losses[item][args.low_method] - losses[item][args.high_method])

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    train_mask = np.asarray([item[1] in train_windows for item in keys], dtype=bool)
    test_mask = np.asarray([item[1] in test_windows for item in keys], dtype=bool)
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("empty train or test split")

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x[train_mask], y[train_mask])
    predictions = model.predict(x)
    threshold = float(np.quantile(predictions[train_mask], 1.0 - args.calibration_high_rate))
    selected_high = predictions >= threshold

    low = np.asarray([losses[item][args.low_method] for item in keys])
    high = np.asarray([losses[item][args.high_method] for item in keys])
    full = np.asarray([losses[item][args.full_method] for item in keys])
    fixed = np.asarray([losses[item][args.fixed_reference_method] for item in keys])

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        chosen = np.where(selected_high[mask], high[mask], low[mask])
        high_rate = float(selected_high[mask].mean())
        committed_fraction = args.low_fraction * (1.0 - high_rate) + args.high_fraction * high_rate
        executed_fraction = args.low_fraction + args.high_fraction * high_rate
        return {
            "tokens": int(mask.sum()),
            "full_ppl": ppl(full[mask]),
            "low_ppl": ppl(low[mask]),
            "fixed_reference_ppl": ppl(fixed[mask]),
            "high_ppl": ppl(high[mask]),
            "offline_routed_ppl": ppl(chosen),
            "offline_routed_ppl_ratio_vs_full": math.exp(float(chosen.mean() - full[mask].mean())),
            "selected_high_rate": high_rate,
            "mean_committed_fraction": committed_fraction,
            "mean_executed_fraction": executed_fraction,
        }

    report = {
        "feature_names": FEATURE_NAMES,
        "train_windows": sorted(train_windows),
        "test_windows": sorted(test_windows),
        "low_method": args.low_method,
        "high_method": args.high_method,
        "fixed_reference_method": args.fixed_reference_method,
        "threshold": threshold,
        "calibration_high_rate": args.calibration_high_rate,
        "train": metrics(train_mask),
        "test": metrics(test_mask),
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(importance)}
                for name, importance in zip(FEATURE_NAMES, model.feature_importances_)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        ),
    }
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "low_fraction": args.low_fraction,
        "high_fraction": args.high_fraction,
        "prediction_horizon": 256,
        "training_report": report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "critical_position_router.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
