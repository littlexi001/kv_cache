from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


CONDITION_PREFIX = "balancedjointrss"
OUTPUT_PATH = "hybrid_sketch_blockresidual64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--targets", default="0.001,0.0025,0.005,0.01,0.02,0.04"
    )
    return parser.parse_args()


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty list")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rate(
    case: str,
    rate: int,
    root: Path,
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], list[dict[str, Any]]],
]:
    details = [
        row
        for row in read_csv(root / "per_case.csv")
        if row["condition"].startswith(CONDITION_PREFIX)
        and row["approximation_path"] == OUTPUT_PATH
    ]
    score_rows = read_csv(root / "score_calibration.csv")
    steps_by_layer: dict[int, int] = defaultdict(int)
    query_rows_by_layer: dict[int, int] = defaultdict(int)
    for row in details:
        layer = int(row["layer"])
        steps_by_layer[layer] = max(steps_by_layer[layer], int(row["step"]) + 1)
    for row in score_rows:
        layer = int(row["layer"])
        query_rows_by_layer[layer] = max(
            query_rows_by_layer[layer], int(row["query_row"]) + 1
        )

    detail_by_condition: dict[tuple[int, int], dict[str, Any]] = {}
    for row in details:
        key = (int(row["layer"]), int(row["step"]))
        detail_by_condition[key] = {
            "case": case,
            "rate": rate,
            "layer": key[0],
            "step": key[1],
            "output_relative_l2": float(row["relative_l2"]),
            "selected_ratio": float(row["selected_ratio_mean"]),
            "selected_tokens": float(row["selected_tokens_mean"]),
            "tail_score_estimate": float(
                row["sampled_tail_score_output_relative_estimate"]
            ),
            "tail_score_standard_error": float(
                row["sampled_tail_score_output_relative_standard_error"]
            ),
            "tail_score_ucb95": float(
                row["sampled_tail_score_output_relative_ucb95"]
            ),
            "tail_score_actual": float(
                row["actual_tail_score_output_relative"]
            ),
        }

    scores_by_condition: dict[
        tuple[int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in score_rows:
        layer = int(row["layer"])
        step_count = steps_by_layer[layer]
        query_groups = query_rows_by_layer[layer] // step_count
        step = int(row["query_row"]) // query_groups
        scores_by_condition[(layer, step)].append(
            {
                "case": case,
                "rate": rate,
                "layer": layer,
                "step": step,
                "kv_head": int(row["kv_head"]),
                "query_row": int(row["query_row"]),
                "estimate": float(
                    row["sampled_score_output_relative_estimate"]
                ),
                "standard_error": float(
                    row["sampled_score_output_relative_standard_error"]
                ),
                "ucb95": float(row["sampled_score_output_relative_ucb95"]),
                "first_order": float(
                    row["first_order_score_output_relative"]
                ),
                "actual": float(row["actual_score_output_relative"]),
                "sampled_softmax_kl": float(row["sampled_softmax_kl"]),
                "sampled_crossfit_softmax_kl": float(
                    row.get("sampled_crossfit_softmax_kl", "nan")
                ),
            }
        )
    return detail_by_condition, scores_by_condition


def summarize_policy(
    policy: str,
    chosen: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rates = [int(row["rate"]) for row in chosen]
    output_errors = [float(row["output_relative_l2"]) for row in chosen]
    actual_score_errors = [float(row["actual"]) for row in score_rows]
    actual_tail_score_errors = [
        float(row["tail_score_actual"]) for row in chosen
    ]
    return {
        "policy": policy,
        "conditions": len(chosen),
        "rate_mean": fmean(rates),
        **{
            f"rate{rate}_fraction": rates.count(rate) / len(rates)
            for rate in sorted(set(rates))
        },
        "selected_ratio_mean": fmean(
            float(row["selected_ratio"]) for row in chosen
        ),
        "output_relative_l2_mean": fmean(output_errors),
        "output_relative_l2_p90": quantile(output_errors, 0.90),
        "output_relative_l2_p99": quantile(output_errors, 0.99),
        "output_relative_l2_maximum": max(output_errors),
        "actual_score_output_relative_mean": fmean(actual_score_errors),
        "actual_score_output_relative_p90": quantile(
            actual_score_errors, 0.90
        ),
        "actual_tail_score_output_relative_mean": fmean(
            actual_tail_score_errors
        ),
        "actual_tail_score_output_relative_p90": quantile(
            actual_tail_score_errors, 0.90
        ),
        "tail_score_probe_ucb95_coverage": fmean(
            float(row["tail_score_actual"] <= row["tail_score_ucb95"])
            for row in chosen
        ),
        "probe_ucb95_coverage": fmean(
            float(row["actual"] <= row["ucb95"]) for row in score_rows
        ),
    }


def main() -> None:
    args = parse_args()
    targets = sorted(
        {float(item) for item in args.targets.split(",") if item.strip()}
    )
    cases = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    by_rate_detail: dict[int, dict[tuple[str, int, int], dict[str, Any]]] = (
        defaultdict(dict)
    )
    by_rate_scores: dict[
        int, dict[tuple[str, int, int], list[dict[str, Any]]]
    ] = defaultdict(dict)
    for case_dir in cases:
        for rate_dir in sorted(case_dir.glob("rate*")):
            rate_suffix = rate_dir.name.removeprefix("rate")
            if not rate_dir.is_dir() or not rate_suffix.isdigit():
                continue
            rate = int(rate_suffix)
            detail, scores = load_rate(case_dir.name, rate, rate_dir)
            for (layer, step), row in detail.items():
                by_rate_detail[rate][(case_dir.name, layer, step)] = row
            for (layer, step), rows in scores.items():
                by_rate_scores[rate][(case_dir.name, layer, step)] = rows

    rates = sorted(by_rate_detail)
    common = set.intersection(
        *(set(by_rate_detail[rate]) for rate in rates),
        *(set(by_rate_scores[rate]) for rate in rates),
    )
    if not common:
        raise RuntimeError("no paired layer-step conditions")

    summaries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for rate in rates:
        chosen = [by_rate_detail[rate][key] for key in sorted(common)]
        score_rows = [
            row
            for key in sorted(common)
            for row in by_rate_scores[rate][key]
        ]
        summaries.append(summarize_policy(f"fixed_rate{rate}", chosen, score_rows))

    for target in targets:
        chosen_details = []
        chosen_scores = []
        for key in sorted(common):
            chosen_rate = rates[-1]
            for rate in rates:
                if (
                    float(by_rate_detail[rate][key]["tail_score_ucb95"])
                    <= target
                ):
                    chosen_rate = rate
                    break
            chosen_details.append(by_rate_detail[chosen_rate][key])
            chosen_scores.extend(by_rate_scores[chosen_rate][key])
            decisions.append(
                {
                    "target": target,
                    "case": key[0],
                    "layer": key[1],
                    "step": key[2],
                    "chosen_rate": chosen_rate,
                    "chosen_tail_score_ucb95": float(
                        by_rate_detail[chosen_rate][key][
                            "tail_score_ucb95"
                        ]
                    ),
                    "output_relative_l2": float(
                        by_rate_detail[chosen_rate][key]["output_relative_l2"]
                    ),
                    "selected_ratio": float(
                        by_rate_detail[chosen_rate][key]["selected_ratio"]
                    ),
                }
            )
        summaries.append(
            summarize_policy(
                f"tail_probe_ucb95_le_{target:g}",
                chosen_details,
                chosen_scores,
            )
        )

    kl_targets = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    for target in kl_targets:
        chosen_details = []
        chosen_scores = []
        for key in sorted(common):
            chosen_rate = rates[-1]
            for rate in rates:
                layer_kl = quantile(
                    [
                        float(row["sampled_softmax_kl"])
                        for row in by_rate_scores[rate][key]
                    ],
                    0.90,
                )
                if layer_kl <= target:
                    chosen_rate = rate
                    break
            chosen_details.append(by_rate_detail[chosen_rate][key])
            chosen_scores.extend(by_rate_scores[chosen_rate][key])
        summaries.append(
            summarize_policy(
                f"sampled_softmax_kl_p90_le_{target:g}",
                chosen_details,
                chosen_scores,
            )
        )

    for target in kl_targets:
        chosen_details = []
        chosen_scores = []
        for key in sorted(common):
            chosen_rate = rates[-1]
            for rate in rates:
                values = [
                    float(row["sampled_crossfit_softmax_kl"])
                    for row in by_rate_scores[rate][key]
                ]
                if all(value == value for value in values) and (
                    quantile(values, 0.90) <= target
                ):
                    chosen_rate = rate
                    break
            chosen_details.append(by_rate_detail[chosen_rate][key])
            chosen_scores.extend(by_rate_scores[chosen_rate][key])
        summaries.append(
            summarize_policy(
                f"crossfit_softmax_kl_p90_le_{target:g}",
                chosen_details,
                chosen_scores,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "controller_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "controller_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    report = {
        "schema": "qksieve_output_probe_rate_controller_v1",
        "input_root": str(args.input_root),
        "rates": rates,
        "targets": targets,
        "sampled_softmax_kl_targets": kl_targets,
        "paired_layer_steps": len(common),
        "decision_rule": (
            "Choose the smallest stable prefill-derived bit-plane rate whose "
            "per-layer 95%-UCB from 128 exact top-risk plus 128 "
            "stratified-tail probes is below the residual score-output "
            "error target after exact sparse attention."
        ),
        "claim_boundary": (
            "Discovery diagnostic on real-QKV local layer outputs. A target "
            "must be frozen before held-out closed-loop evaluation."
        ),
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
