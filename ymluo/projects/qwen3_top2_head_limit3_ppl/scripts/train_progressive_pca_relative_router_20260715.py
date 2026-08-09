from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_progressive_pca_router_20260715 import (  # noqa: E402
    ACTIONS,
    FEATURE_NAMES,
    load_records,
    mask_for_window,
    summarize,
)


def relative_summary(
    records: list[dict[str, object]], actions: list[str]
) -> dict[str, object]:
    output = summarize(records, actions)
    policy_nll = np.asarray(
        [record["action_nll"][action] for record, action in zip(records, actions, strict=True)],
        dtype=np.float64,
    )
    max_action_nll = np.asarray(
        [record["action_nll"]["0.02"] for record in records], dtype=np.float64
    )
    output.update(
        {
            "delta_nll_vs_2pct": float((policy_nll - max_action_nll).mean()),
            "ppl_ratio_vs_2pct": math.exp(float((policy_nll - max_action_nll).mean())),
            "logical_pca_index_plus_shared_gqa_kv_fraction": (
                0.06640625 + float(output["mean_final_attention_fraction"])
            ),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_extra_ratio_vs_2pct", type=float, default=1.005)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    records = load_records(args.scan_root, args.full_root)
    features = np.asarray([record["features"] for record in records], dtype=np.float64)
    train_mask = mask_for_window(records, 0)
    calibration_mask = mask_for_window(records, 1)
    test_mask = mask_for_window(records, 2)
    calibration_records = [
        record for record, keep in zip(records, calibration_mask, strict=True) if keep
    ]
    test_records = [record for record, keep in zip(records, test_mask, strict=True) if keep]

    models: dict[str, ExtraTreesRegressor] = {}
    predictions: dict[str, np.ndarray] = {}
    calibration_targets: dict[str, np.ndarray] = {}
    for action in ACTIONS[:-1]:
        target = np.asarray(
            [
                record["action_nll"][action] - record["action_nll"]["0.02"]
                for record in records
            ],
            dtype=np.float64,
        )
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

    def route(mask: np.ndarray, offsets: dict[str, float], safe_regret: float) -> list[str]:
        actions = []
        for index in np.flatnonzero(mask):
            selected = "0.02"
            for action in ACTIONS[:-1]:
                if predictions[action][index] + offsets[action] <= safe_regret:
                    selected = action
                    break
            actions.append(selected)
        return actions

    candidates = []
    for conformal_level in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99):
        offsets = {
            action: float(
                np.quantile(
                    calibration_targets[action] - predictions[action][calibration_mask],
                    conformal_level,
                    method="higher",
                )
            )
            for action in ACTIONS[:-1]
        }
        for safe_regret in (-0.02, -0.01, 0.0, 0.0025, 0.005, 0.01, 0.02, 0.04):
            actions = route(calibration_mask, offsets, safe_regret)
            candidates.append(
                {
                    "conformal_level": conformal_level,
                    "safe_regret": safe_regret,
                    "offsets": offsets,
                    "actions": actions,
                    "calibration": relative_summary(calibration_records, actions),
                }
            )
    feasible = [
        row
        for row in candidates
        if row["calibration"]["ppl_ratio_vs_2pct"] <= args.max_extra_ratio_vs_2pct
    ]
    selected = min(
        feasible or candidates,
        key=lambda row: (
            row["calibration"]["progressive_attention_latency_ms_128k"]
            if feasible
            else row["calibration"]["ppl_ratio_vs_2pct"],
            row["calibration"]["ppl_ratio_vs_2pct"],
        ),
    )
    teacher_test_actions = route(
        test_mask, selected["offsets"], float(selected["safe_regret"])
    )

    fit_mask = train_mask | calibration_mask
    teacher_fit_actions = route(
        fit_mask, selected["offsets"], float(selected["safe_regret"])
    )
    teacher_labels = np.asarray([ACTIONS.index(action) for action in teacher_fit_actions])
    fit_features = features[fit_mask]
    scaler = StandardScaler().fit(fit_features)
    rng = np.random.default_rng(args.seed)
    class_indices = [np.flatnonzero(teacher_labels == index) for index in range(len(ACTIONS))]
    max_count = max(len(indices) for indices in class_indices)
    balanced_indices = np.concatenate(
        [rng.choice(indices, size=max_count, replace=True) for indices in class_indices]
    )
    rng.shuffle(balanced_indices)
    student = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=0.01,
        max_iter=3000,
        random_state=args.seed,
    )
    student.fit(
        scaler.transform(fit_features[balanced_indices]), teacher_labels[balanced_indices]
    )

    def student_actions(mask: np.ndarray, escalation_bias: float) -> list[str]:
        logits = student.predict_log_proba(scaler.transform(features[mask]))
        adjusted = logits.copy()
        adjusted[:, 0] -= escalation_bias
        adjusted[:, 1] -= 0.5 * escalation_bias
        adjusted[:, 2] += escalation_bias
        return [ACTIONS[int(index)] for index in np.argmax(adjusted, axis=1)]

    student_candidates = []
    for bias in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        actions = student_actions(calibration_mask, bias)
        student_candidates.append(
            {
                "escalation_bias": bias,
                "actions": actions,
                "calibration": relative_summary(calibration_records, actions),
            }
        )
    feasible_students = [
        row
        for row in student_candidates
        if row["calibration"]["ppl_ratio_vs_2pct"] <= args.max_extra_ratio_vs_2pct
    ]
    selected_student = min(
        feasible_students or student_candidates,
        key=lambda row: (
            row["calibration"]["progressive_attention_latency_ms_128k"]
            if feasible_students
            else row["calibration"]["ppl_ratio_vs_2pct"],
            row["calibration"]["ppl_ratio_vs_2pct"],
        ),
    )
    test_student_actions = student_actions(
        test_mask, float(selected_student["escalation_bias"])
    )

    adjusted_output_bias = student.intercepts_[1].copy()
    bias = float(selected_student["escalation_bias"])
    adjusted_output_bias[0] -= bias
    adjusted_output_bias[1] -= 0.5 * bias
    adjusted_output_bias[2] += bias
    report = {
        "protocol": {
            "target": "action NLL minus shared-2pct NLL",
            "train_windows": [0],
            "calibration_windows": [1],
            "independent_test_windows": [2],
            "features": list(FEATURE_NAMES),
            "max_extra_ratio_vs_2pct_selected_on_calibration": args.max_extra_ratio_vs_2pct,
            "gqa_candidate_mode": "shared_mean",
        },
        "selected_teacher": {
            "conformal_level": selected["conformal_level"],
            "safe_regret": selected["safe_regret"],
            "offsets": selected["offsets"],
            "calibration": selected["calibration"],
            "independent_test": relative_summary(test_records, teacher_test_actions),
        },
        "selected_mlp": {
            "escalation_bias": selected_student["escalation_bias"],
            "calibration": selected_student["calibration"],
            "independent_test": relative_summary(test_records, test_student_actions),
        },
        "fixed_2pct_test": relative_summary(test_records, ["0.02"] * len(test_records)),
        "teacher_calibration_frontier": [
            {
                "conformal_level": row["conformal_level"],
                "safe_regret": row["safe_regret"],
                **row["calibration"],
            }
            for row in candidates
        ],
        "student_calibration_frontier": [
            {"escalation_bias": row["escalation_bias"], **row["calibration"]}
            for row in student_candidates
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    artifact = {
        "feature_names": FEATURE_NAMES,
        "actions": ACTIONS,
        "protocol": report["protocol"],
        "mlp_action_router": {
            "feature_mean": scaler.mean_,
            "feature_scale": scaler.scale_,
            "input_weight": student.coefs_[0],
            "input_bias": student.intercepts_[0],
            "output_weight": student.coefs_[1],
            "output_bias": adjusted_output_bias,
            "classes": student.classes_,
        },
    }
    joblib.dump(artifact, args.output_dir / "router.joblib")
    print(json.dumps(report["selected_mlp"]["independent_test"], indent=2))


if __name__ == "__main__":
    main()
