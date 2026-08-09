from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CONDITION = "attention_hist256_coverage_0p95"
PATH = "calibrated_hybrid_sketch_affine_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate_roots",
        required=True,
        help="Comma-separated RATE=RESULT_ROOT entries.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def parse_rate_roots(value: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for item in value.split(","):
        rate, root = item.split("=", 1)
        output[int(rate)] = Path(root)
    if len(output) < 2:
        raise ValueError("at least two rate roots are required")
    return dict(sorted(output.items()))


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def load_rate_rows(rate: int, root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    output: dict[tuple[str, int, int], dict[str, Any]] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = case_dir / "summary.json"
        score_path = case_dir / "score_calibration.csv"
        detail_path = case_dir / "per_case.csv"
        if not (summary_path.exists() and score_path.exists() and detail_path.exists()):
            continue
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        history_count = int(report["setup"]["history_tokens"])
        with detail_path.open(encoding="utf-8", newline="") as handle:
            details = [
                row
                for row in csv.DictReader(handle)
                if row["condition"] == CONDITION
                and row["approximation_path"] == PATH
            ]
        steps_by_layer: dict[int, int] = defaultdict(int)
        for row in details:
            layer = int(row["layer"])
            steps_by_layer[layer] = max(
                steps_by_layer[layer], int(row["step"]) + 1
            )
        with score_path.open(encoding="utf-8", newline="") as handle:
            score_rows = list(csv.DictReader(handle))
        max_query_row: dict[int, int] = defaultdict(int)
        for row in score_rows:
            layer = int(row["layer"])
            max_query_row[layer] = max(
                max_query_row[layer], int(row["query_row"]) + 1
            )
        query_groups = {
            layer: max_query_row[layer] // step_count
            for layer, step_count in steps_by_layer.items()
        }
        risk_by_condition: dict[tuple[int, int], list[tuple[float, float]]] = (
            defaultdict(list)
        )
        for row in score_rows:
            layer = int(row["layer"])
            step = int(row["query_row"]) // query_groups[layer]
            risk_by_condition[(layer, step)].append(
                (
                    float(row["sampled_crossfit_rmse"]),
                    float(row["sampled_normalized_crossfit_rmse"]),
                )
            )
        for row in details:
            layer = int(row["layer"])
            step = int(row["step"])
            risks = np.asarray(
                risk_by_condition[(layer, step)], dtype=np.float64
            )
            key = (case_dir.name, layer, step)
            output[key] = {
                "case": case_dir.name,
                "history_tokens": history_count,
                "layer": layer,
                "step": step,
                "rate": rate,
                "output_relative_l2": float(row["relative_l2"]),
                "selected_ratio": float(row["selected_ratio_mean"]),
                "exact_attention_mass": float(
                    row["exact_attention_mass_mean"]
                ),
                "sample_rmse_mean": float(risks[:, 0].mean()),
                "sample_rmse_p90": float(np.quantile(risks[:, 0], 0.90)),
                "sample_rmse_maximum": float(risks[:, 0].max()),
                "sample_normalized_rmse_mean": float(risks[:, 1].mean()),
                "sample_normalized_rmse_p90": float(
                    np.quantile(risks[:, 1], 0.90)
                ),
                "sample_normalized_rmse_maximum": float(risks[:, 1].max()),
                "extreme_risk_p90": float(
                    np.quantile(risks[:, 1], 0.90)
                    * math.sqrt(2.0 * math.log(history_count))
                ),
            }
    return output


def summarize_rows(
    name: str,
    rows: list[dict[str, Any]],
    *,
    case: str = "all",
) -> dict[str, Any]:
    errors = [float(row["output_relative_l2"]) for row in rows]
    rates = [int(row["rate"]) for row in rows]
    return {
        "policy": name,
        "case": case,
        "conditions": len(rows),
        "rate_mean": float(np.mean(rates)),
        "rate15_fraction": rates.count(15) / len(rates),
        "rate19_fraction": rates.count(19) / len(rates),
        "rate23_fraction": rates.count(23) / len(rates),
        "selected_ratio_mean": float(
            np.mean([float(row["selected_ratio"]) for row in rows])
        ),
        "exact_attention_mass_mean": float(
            np.mean([float(row["exact_attention_mass"]) for row in rows])
        ),
        "output_relative_l2_mean": float(np.mean(errors)),
        "output_relative_l2_p90": percentile(errors, 0.90),
        "output_relative_l2_p99": percentile(errors, 0.99),
        "output_relative_l2_maximum": max(errors),
    }


def main() -> None:
    args = parse_args()
    rate_roots = parse_rate_roots(args.rate_roots)
    by_rate = {
        rate: load_rate_rows(rate, root) for rate, root in rate_roots.items()
    }
    common = set.intersection(*(set(rows) for rows in by_rate.values()))
    if not common:
        raise RuntimeError("rate roots do not contain paired conditions")
    rates = sorted(by_rate)
    policies: dict[str, list[dict[str, Any]]] = {
        f"fixed_rate{rate}": [by_rate[rate][key] for key in sorted(common)]
        for rate in rates
    }
    threshold_grid = {
        "sample_rmse_p90": (0.30, 0.40, 0.50, 0.60, 0.70),
        "sample_normalized_rmse_p90": (0.10, 0.15, 0.20, 0.25, 0.30),
        "extreme_risk_p90": (0.50, 0.75, 1.00, 1.25, 1.50),
    }
    decision_rows: list[dict[str, Any]] = []
    for feature, thresholds in threshold_grid.items():
        for threshold in thresholds:
            name = f"minrate_{feature}_le_{threshold:g}"
            selected_rows = []
            for key in sorted(common):
                chosen = by_rate[rates[-1]][key]
                for rate in rates:
                    candidate = by_rate[rate][key]
                    if float(candidate[feature]) <= threshold:
                        chosen = candidate
                        break
                selected_rows.append(chosen)
                decision_rows.append(
                    {
                        "policy": name,
                        "threshold_feature": feature,
                        "threshold": threshold,
                        **chosen,
                    }
                )
            policies[name] = selected_rows

    summary_rows: list[dict[str, Any]] = []
    for name, rows in policies.items():
        summary_rows.append(summarize_rows(name, rows))
        cases = sorted({str(row["case"]) for row in rows})
        for case in cases:
            summary_rows.append(
                summarize_rows(
                    name,
                    [row for row in rows if row["case"] == case],
                    case=case,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "controller_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "controller_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    report = {
        "schema": "qksieve_sampled_rate_controller_v1",
        "condition": CONDITION,
        "approximation_path": PATH,
        "rate_roots": {str(rate): str(root) for rate, root in rate_roots.items()},
        "paired_conditions": len(common),
        "decision_rule": (
            "At each layer and decode step, choose the smallest index rate "
            "whose 256-point stratified-jittered cross-fit score error is "
            "below the stated threshold; otherwise use the highest rate."
        ),
        "claim_boundary": (
            "This is a discovery-set controller simulation on real-QKV "
            "layer outputs. Thresholds require a frozen held-out test."
        ),
        "summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
