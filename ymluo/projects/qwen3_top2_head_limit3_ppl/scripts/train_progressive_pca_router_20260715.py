from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ACTIONS = ("0.005", "0.01", "0.02")
FEATURE_NAMES = (
    "target_position_fraction",
    "logit_entropy",
    "logit_entropy_normalized",
    "logit_top1_probability",
    "logit_top2_probability",
    "logit_top1_top2_margin",
    "logit_top5_mass",
    "candidate_top_gap_mean",
    "candidate_boundary_gap_mean",
    "candidate_temporal_stability_mean",
)
ACTION_LATENCY_MS = {"0.005": 0.4728525, "0.01": 0.6304461, "0.02": 0.9554893}
FULL_LATENCY_MS = 2.628


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["topic"], int(row["window"]), int(row["target_index"])


def feature_vector(row: dict[str, str]) -> list[float]:
    defaults = {"candidate_temporal_stability_mean": 1.0}
    return [
        int(row["target_index"]) / 255.0,
        *(float(row.get(name, defaults.get(name, 0.0))) for name in FEATURE_NAMES[1:]),
    ]


def load_records(scan_root: Path, full_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for topic in ("sports", "medicine"):
        for window in (0, 1, 2):
            action_rows = {}
            for action in ACTIONS:
                path = scan_root / f"{topic}_w{window}_f{action}" / "token_results.csv"
                action_rows[action] = {row_key(row): row for row in read_csv(path)}
            full_rows = {
                row_key(row): row
                for row in read_csv(full_root / f"{topic}_w{window}" / "token_results.csv")
                if row["method"] == "full_attention"
            }
            for key in sorted(full_rows):
                if any(key not in action_rows[action] for action in ACTIONS):
                    raise RuntimeError(f"missing aligned PCA action for {key}")
                provisional = action_rows["0.005"][key]
                full = full_rows[key]
                if int(provisional["token_id"]) != int(full["token_id"]):
                    raise RuntimeError(f"token alignment mismatch for {key}")
                action_nll = {
                    action: float(action_rows[action][key]["nll"]) for action in ACTIONS
                }
                full_nll = float(full["nll"])
                records.append(
                    {
                        "key": key,
                        "features": feature_vector(provisional),
                        "action_nll": action_nll,
                        "full_nll": full_nll,
                        "deltas": {
                            action: action_nll[action] - full_nll for action in ACTIONS
                        },
                    }
                )
    return records


def mask_for_window(records: list[dict[str, object]], window: int) -> np.ndarray:
    return np.asarray([record["key"][1] == window for record in records], dtype=bool)


def summarize(
    records: list[dict[str, object]], actions: list[str]
) -> dict[str, object]:
    policy_nll = np.asarray(
        [record["action_nll"][action] for record, action in zip(records, actions, strict=True)],
        dtype=np.float64,
    )
    full_nll = np.asarray([record["full_nll"] for record in records], dtype=np.float64)
    counts = Counter(actions)
    mean_policy = float(policy_nll.mean())
    mean_full = float(full_nll.mean())
    progressive_latency = float(
        np.mean(
            [
                ACTION_LATENCY_MS["0.005"]
                + (0.0 if action == "0.005" else ACTION_LATENCY_MS[action])
                for action in actions
            ]
        )
    )
    mean_final_fraction = float(np.mean([float(action) for action in actions]))
    return {
        "tokens": len(records),
        "mean_nll": mean_policy,
        "full_mean_nll": mean_full,
        "delta_nll_vs_full": mean_policy - mean_full,
        "ppl": math.exp(mean_policy),
        "full_ppl": math.exp(mean_full),
        "ppl_ratio_vs_full": math.exp(mean_policy - mean_full),
        "action_rates": {action: counts[action] / len(records) for action in ACTIONS},
        "mean_final_attention_fraction": mean_final_fraction,
        "logical_index_plus_exact_kv_fraction": 0.06640625 + mean_final_fraction,
        "progressive_attention_latency_ms_128k": progressive_latency,
        "estimated_attention_speedup_128k": FULL_LATENCY_MS / progressive_latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--quality_limit", type=float, default=1.05)
    parser.add_argument("--model_type", choices=["extra_trees", "ridge"], default="extra_trees")
    parser.add_argument("--distill_mlp", action="store_true")
    parser.add_argument("--force_conformal_level", type=float)
    parser.add_argument("--force_safe_delta", type=float)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    records = load_records(args.scan_root, args.full_root)
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    train_mask = mask_for_window(records, 0)
    calibration_mask = mask_for_window(records, 1)
    test_mask = mask_for_window(records, 2)
    calibration_records = [
        record for record, selected in zip(records, calibration_mask, strict=True) if selected
    ]
    test_records = [record for record, selected in zip(records, test_mask, strict=True) if selected]

    models = {}
    predictions = {}
    calibration_targets = {}
    for action in ACTIONS:
        target = np.asarray([record["deltas"][action] for record in records])
        if args.model_type == "ridge":
            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)),
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=12,
                max_features=1.0,
                random_state=args.seed + int(float(action) * 10000),
                n_jobs=-1,
            )
        model.fit(features[train_mask], target[train_mask])
        models[action] = model
        predictions[action] = model.predict(features)
        calibration_targets[action] = target[calibration_mask]

    candidate_policies = []

    def route(
        mask: np.ndarray, offsets: dict[str, float], safe_delta: float
    ) -> list[str]:
        selected_actions = []
        for index in np.flatnonzero(mask):
            selected_action = "0.02"
            for action in ACTIONS[:-1]:
                if predictions[action][index] + offsets[action] <= safe_delta:
                    selected_action = action
                    break
            selected_actions.append(selected_action)
        return selected_actions

    for conformal_level in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        offsets = {}
        for action in ACTIONS:
            residual = calibration_targets[action] - predictions[action][calibration_mask]
            offsets[action] = float(
                np.quantile(residual, conformal_level, method="higher")
            )
        for safe_delta in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            calibration_actions = route(calibration_mask, offsets, safe_delta)
            calibration_summary = summarize(calibration_records, calibration_actions)
            candidate_policies.append(
                {
                    "conformal_level": conformal_level,
                    "safe_delta_nll": safe_delta,
                    "offsets": offsets,
                    "calibration_actions": calibration_actions,
                    "calibration": calibration_summary,
                }
            )

    feasible = [
        row
        for row in candidate_policies
        if row["calibration"]["ppl_ratio_vs_full"] <= args.quality_limit
    ]
    if args.force_conformal_level is not None or args.force_safe_delta is not None:
        if args.force_conformal_level is None or args.force_safe_delta is None:
            raise ValueError("both forced router hyperparameters must be provided")
        selected = next(
            row
            for row in candidate_policies
            if row["conformal_level"] == args.force_conformal_level
            and row["safe_delta_nll"] == args.force_safe_delta
        )
    else:
        selected = min(
            feasible or candidate_policies,
            key=lambda row: (
                row["calibration"]["progressive_attention_latency_ms_128k"]
                if feasible
                else row["calibration"]["ppl_ratio_vs_full"],
                row["calibration"]["ppl_ratio_vs_full"],
            ),
        )
    test_actions = route(
        test_mask,
        selected["offsets"],
        float(selected["safe_delta_nll"]),
    )
    fixed_test = {
        action: summarize(test_records, [action] * len(test_records)) for action in ACTIONS
    }
    distilled = None
    if args.distill_mlp:
        fit_mask = train_mask | calibration_mask
        teacher_actions = route(
            fit_mask,
            selected["offsets"],
            float(selected["safe_delta_nll"]),
        )
        teacher_labels = np.asarray([ACTIONS.index(action) for action in teacher_actions])
        fit_features = features[fit_mask]
        rng = np.random.default_rng(args.seed)
        class_indices = [np.flatnonzero(teacher_labels == index) for index in range(len(ACTIONS))]
        max_count = max(len(indices) for indices in class_indices)
        balanced_indices = np.concatenate(
            [rng.choice(indices, size=max_count, replace=True) for indices in class_indices]
        )
        rng.shuffle(balanced_indices)
        student_scaler = StandardScaler().fit(fit_features)
        student = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            alpha=0.01,
            max_iter=3000,
            random_state=args.seed,
        )
        student.fit(
            student_scaler.transform(fit_features[balanced_indices]),
            teacher_labels[balanced_indices],
        )
        student_test_labels = student.predict(student_scaler.transform(features[test_mask]))
        student_test_actions = [ACTIONS[int(label)] for label in student_test_labels]
        student_calibration_labels = student.predict(
            student_scaler.transform(features[calibration_mask])
        )
        student_calibration_actions = [
            ACTIONS[int(label)] for label in student_calibration_labels
        ]
        distilled = {
            "scaler": student_scaler,
            "model": student,
            "calibration": summarize(calibration_records, student_calibration_actions),
            "independent_test": summarize(test_records, student_test_actions),
            "teacher_agreement_test": float(
                np.mean(student_test_labels == np.asarray([ACTIONS.index(a) for a in test_actions]))
            ),
        }
    report = {
        "protocol": {
            "train_windows": [0],
            "calibration_windows": [1],
            "independent_test_windows": [2],
            "features": list(FEATURE_NAMES),
            "target_leakage_features_excluded": [
                "token_id",
                "nll",
                "topic",
                "window",
                "true_next_token_type",
            ],
            "progressive_execution": "always 0.5%, then rerun current position at 1% or 2%",
            "quality_limit_selected_on_calibration": args.quality_limit,
            "physical_full_kv_still_retained_in_current_harness": True,
            "model_type": args.model_type,
        },
        "selected_hyperparameters": {
            "conformal_level": selected["conformal_level"],
            "safe_delta_nll": selected["safe_delta_nll"],
            "offsets": selected["offsets"],
        },
        "calibration": selected["calibration"],
        "independent_test": summarize(test_records, test_actions),
        "fixed_action_independent_test": fixed_test,
        "distilled_mlp": None
        if distilled is None
        else {
            "calibration": distilled["calibration"],
            "independent_test": distilled["independent_test"],
            "teacher_agreement_test": distilled["teacher_agreement_test"],
        },
        "feature_importance": sorted(
            [
                {
                    "feature": name,
                    "importance": float(
                        np.mean(
                            [
                                model.feature_importances_[index]
                                if hasattr(model, "feature_importances_")
                                else abs(model[-1].coef_[index])
                                for model in models.values()
                            ]
                        )
                    ),
                }
                for index, name in enumerate(FEATURE_NAMES)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        ),
        "calibration_frontier": [
            {
                "conformal_level": row["conformal_level"],
                "safe_delta_nll": row["safe_delta_nll"],
                **row["calibration"],
            }
            for row in candidate_policies
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    artifact = {
            "models": models,
            "feature_names": FEATURE_NAMES,
            "offsets": selected["offsets"],
            "safe_delta_nll": selected["safe_delta_nll"],
            "actions": ACTIONS,
            "protocol": report["protocol"],
        }
    if args.model_type == "ridge":
        effective_weights = []
        effective_biases = []
        for action in ACTIONS:
            scaler = models[action][0]
            ridge = models[action][1]
            weight = np.asarray(ridge.coef_, dtype=np.float64) / scaler.scale_
            bias = float(ridge.intercept_ - np.dot(weight, scaler.mean_))
            effective_weights.append(weight)
            effective_biases.append(bias)
        artifact["linear_router"] = {
            "weights": np.asarray(effective_weights),
            "biases": np.asarray(effective_biases),
        }
    if distilled is not None:
        student_scaler = distilled["scaler"]
        student = distilled["model"]
        artifact["mlp_action_router"] = {
            "feature_mean": student_scaler.mean_,
            "feature_scale": student_scaler.scale_,
            "input_weight": student.coefs_[0],
            "input_bias": student.intercepts_[0],
            "output_weight": student.coefs_[1],
            "output_bias": student.intercepts_[1],
            "classes": student.classes_,
        }
    joblib.dump(artifact, args.output_dir / "router.joblib")
    with (args.output_dir / "independent_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topic",
                "window",
                "target_index",
                "selected_action",
                "full_nll",
                "policy_nll",
            ],
        )
        writer.writeheader()
        for record, action in zip(test_records, test_actions, strict=True):
            topic, window, target_index = record["key"]
            writer.writerow(
                {
                    "topic": topic,
                    "window": window,
                    "target_index": target_index,
                    "selected_action": action,
                    "full_nll": record["full_nll"],
                    "policy_nll": record["action_nll"][action],
                }
            )
    print(json.dumps(report["independent_test"], indent=2))


if __name__ == "__main__":
    main()
