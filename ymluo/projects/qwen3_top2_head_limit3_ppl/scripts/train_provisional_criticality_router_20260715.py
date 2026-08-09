from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


ACTIONS = ("0.005", "0.01", "0.02")
FEATURE_NAMES = (
    "target_position_fraction",
    "logit_entropy",
    "logit_entropy_normalized",
    "logit_top1_probability",
    "logit_top2_probability",
    "logit_top1_top2_margin",
    "logit_top5_mass",
    "retained_mass_mean",
    "retained_mass_min",
    "retained_mass_p10",
    "retained_mass_p25",
    "retained_mass_lt90_fraction",
    "retained_mass_lt95_fraction",
    "retained_mass_lt99_fraction",
    "retained_mass_early_mean",
    "retained_mass_middle_mean",
    "retained_mass_late_mean",
    "top1_attention_mass_mean",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["topic"], int(row["window"]), int(row["target_index"])


def feature_vector(row: dict[str, str]) -> list[float]:
    values = []
    for name in FEATURE_NAMES:
        if name == "target_position_fraction":
            values.append(int(row["target_index"]) / 255.0)
        else:
            values.append(float(row[name]))
    return values


def load_dataset(scan_root: Path, reference_root: Path) -> list[dict[str, object]]:
    records = []
    for topic in ("sports", "medicine"):
        for window in (0, 1, 2):
            action_rows = {}
            for action in ACTIONS:
                path = scan_root / f"{topic}_w{window}_f{action}" / "token_results.csv"
                action_rows[action] = {row_key(row): row for row in read_rows(path)}
            reference_path = reference_root / f"{topic}_w{window}" / "token_results.csv"
            reference_rows = {row_key(row): row for row in read_rows(reference_path)}
            keys = sorted(reference_rows)
            for key in keys:
                if any(key not in action_rows[action] for action in ACTIONS):
                    raise RuntimeError(f"missing aligned action row for {key}")
                provisional = action_rows["0.005"][key]
                reference = reference_rows[key]
                if int(provisional["token_id"]) != int(reference["token_id"]):
                    raise RuntimeError(f"token alignment mismatch for {key}")
                action_nll = {action: float(action_rows[action][key]["nll"]) for action in ACTIONS}
                reference_nll = float(reference["nll"])
                records.append(
                    {
                        "key": key,
                        "features": feature_vector(provisional),
                        "action_nll": action_nll,
                        "reference_nll": reference_nll,
                        "deltas": {action: action_nll[action] - reference_nll for action in ACTIONS},
                    }
                )
    return records


def split_arrays(records: list[dict[str, object]], window: int) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    selected = [record for record in records if record["key"][1] == window]
    x = np.asarray([record["features"] for record in selected], dtype=np.float64)
    deltas = {
        action: np.asarray([record["deltas"][action] for record in selected], dtype=np.float64)
        for action in ACTIONS
    }
    return x, deltas, selected


def summarize_policy(
    records: list[dict[str, object]],
    selected_actions: list[str],
) -> dict[str, object]:
    policy_nll = []
    reference_nll = []
    for record, action in zip(records, selected_actions, strict=True):
        reference = float(record["reference_nll"])
        reference_nll.append(reference)
        policy_nll.append(reference if action == "rerank" else float(record["action_nll"][action]))
    mean_policy = float(np.mean(policy_nll))
    mean_reference = float(np.mean(reference_nll))
    counts = Counter(selected_actions)
    return {
        "tokens": len(records),
        "mean_nll": mean_policy,
        "reference_mean_nll": mean_reference,
        "delta_nll_vs_rerank": mean_policy - mean_reference,
        "ppl": math.exp(mean_policy),
        "reference_ppl": math.exp(mean_reference),
        "ppl_ratio_vs_rerank": math.exp(mean_policy - mean_reference),
        "action_rates": {action: counts[action] / len(records) for action in (*ACTIONS, "rerank")},
        "mean_scan_fraction": sum(
            (float(action) if action != "rerank" else 0.008650079035647509)
            for action in selected_actions
        )
        / len(selected_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--conformal_level", type=float, default=0.95)
    parser.add_argument("--safe_delta_nll", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    records = load_dataset(args.scan_root, args.reference_root)
    train_x, train_delta, _ = split_arrays(records, 0)
    calibration_x, calibration_delta, calibration_records = split_arrays(records, 1)
    test_x, test_delta, test_records = split_arrays(records, 2)
    models = {}
    conformal_offsets = {}
    feature_importance = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for action in ACTIONS:
        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=args.seed + int(float(action) * 10000),
            n_jobs=-1,
        )
        model.fit(train_x, train_delta[action])
        calibration_prediction = model.predict(calibration_x)
        residual = calibration_delta[action] - calibration_prediction
        conformal_offsets[action] = float(np.quantile(residual, args.conformal_level, method="higher"))
        models[action] = model
        feature_importance += model.feature_importances_
    feature_importance /= len(ACTIONS)

    def route(features: np.ndarray) -> list[str]:
        upper_bounds = {
            action: models[action].predict(features) + conformal_offsets[action]
            for action in ACTIONS
        }
        selected = []
        for index in range(features.shape[0]):
            action = "rerank"
            for candidate in ACTIONS:
                if upper_bounds[candidate][index] <= args.safe_delta_nll:
                    action = candidate
                    break
            selected.append(action)
        return selected

    calibration_actions = route(calibration_x)
    test_actions = route(test_x)
    fixed_test = {
        action: summarize_policy(test_records, [action] * len(test_records)) for action in ACTIONS
    }
    report = {
        "protocol": {
            "train_windows": [0],
            "calibration_windows": [1],
            "test_windows": [2],
            "features": list(FEATURE_NAMES),
            "target_leakage_features_excluded": ["token_id", "nll", "topic", "window"],
            "conformal_level": args.conformal_level,
            "safe_delta_nll": args.safe_delta_nll,
        },
        "conformal_offsets": conformal_offsets,
        "calibration": summarize_policy(calibration_records, calibration_actions),
        "independent_test": summarize_policy(test_records, test_actions),
        "fixed_action_independent_test": fixed_test,
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(importance)}
                for name, importance in zip(FEATURE_NAMES, feature_importance, strict=True)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "independent_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["topic", "window", "target_index", "selected_action", "reference_nll", "policy_nll"],
        )
        writer.writeheader()
        for record, action in zip(test_records, test_actions, strict=True):
            topic, window, target_index = record["key"]
            reference_nll = float(record["reference_nll"])
            writer.writerow(
                {
                    "topic": topic,
                    "window": window,
                    "target_index": target_index,
                    "selected_action": action,
                    "reference_nll": reference_nll,
                    "policy_nll": reference_nll if action == "rerank" else float(record["action_nll"][action]),
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
