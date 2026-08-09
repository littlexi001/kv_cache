from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alternating within-dataset calibration/test audit for real LongBench NLL routing."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in the form mode=/path/to/answer_nll_rows.csv",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def read_candidate(spec: str) -> tuple[str, list[dict[str, str]]]:
    if "=" not in spec:
        raise ValueError("candidate must be mode=/path/to/csv")
    mode, raw_path = spec.split("=", 1)
    with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mode"] == mode]
    return mode, rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [read_candidate(spec) for spec in args.candidate]
    modes = [mode for mode, _ in payloads]
    nll: dict[tuple[int, str], float] = {}
    dataset: dict[int, str] = {}
    for mode, rows in payloads:
        for row in rows:
            query_id = int(row["query_id"])
            nll[(query_id, mode)] = float(row["answer_nll"])
            dataset[query_id] = row["dataset"]
    query_ids = sorted(dataset)
    if any((query_id, mode) not in nll for query_id in query_ids for mode in modes):
        raise ValueError("Candidate files do not cover identical query IDs")

    calibration: list[int] = []
    test: list[int] = []
    policy_rows: list[dict[str, Any]] = []
    policy: dict[str, str] = {}
    for task in sorted(set(dataset.values())):
        ids = [query_id for query_id in query_ids if dataset[query_id] == task]
        task_calibration, task_test = ids[::2], ids[1::2]
        calibration.extend(task_calibration)
        test.extend(task_test)
        scores = {
            mode: statistics.fmean(nll[(query_id, mode)] for query_id in task_calibration)
            for mode in modes
        }
        policy[task] = min(scores, key=scores.get)
        for mode, mean_nll in scores.items():
            policy_rows.append(
                {
                    "dataset": task,
                    "calibration_queries": len(task_calibration),
                    "test_queries": len(task_test),
                    "candidate": mode,
                    "calibration_mean_nll": mean_nll,
                    "selected": float(mode == policy[task]),
                }
            )

    routed_rows: list[dict[str, Any]] = []
    for query_id in test:
        selected = policy[dataset[query_id]]
        routed_rows.append(
            {
                "query_id": query_id,
                "dataset": dataset[query_id],
                "selected_candidate": selected,
                "answer_nll": nll[(query_id, selected)],
                **{f"nll_{mode}": nll[(query_id, mode)] for mode in modes},
            }
        )

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, Any]] = []
    routed = np.asarray([float(row["answer_nll"]) for row in routed_rows])
    for mode in modes:
        reference = np.asarray([float(row[f"nll_{mode}"]) for row in routed_rows])
        delta = routed - reference
        bootstrap = np.empty(args.bootstrap_samples)
        for index in range(args.bootstrap_samples):
            sample = rng.integers(0, len(delta), len(delta))
            bootstrap[index] = delta[sample].mean()
        comparisons.append(
            {
                "reference": mode,
                "reference_test_mean_nll": float(reference.mean()),
                "routed_test_mean_nll": float(routed.mean()),
                "mean_delta": float(delta.mean()),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "win_rate": float(np.mean(delta < 0.0)),
            }
        )

    oracle_choices: list[str] = []
    oracle_values: list[float] = []
    for query_id in test:
        choice = min(modes, key=lambda mode: nll[(query_id, mode)])
        oracle_choices.append(choice)
        oracle_values.append(nll[(query_id, choice)])
    candidate_test_means = {
        mode: statistics.fmean(nll[(query_id, mode)] for query_id in test) for mode in modes
    }
    best_global_mode = min(candidate_test_means, key=candidate_test_means.get)
    oracle_mean = statistics.fmean(oracle_values)
    oracle_diagnostic = {
        "best_global_mode": best_global_mode,
        "best_global_mean_nll": candidate_test_means[best_global_mode],
        "per_query_oracle_mean_nll": oracle_mean,
        "oracle_headroom": candidate_test_means[best_global_mode] - oracle_mean,
        "oracle_action_counts": dict(
            sorted({mode: oracle_choices.count(mode) for mode in modes}.items())
        ),
    }

    write_csv(output_dir / "calibration_policy.csv", policy_rows, list(policy_rows[0]))
    write_csv(output_dir / "heldout_rows.csv", routed_rows, list(routed_rows[0]))
    write_csv(output_dir / "comparisons.csv", comparisons, list(comparisons[0]))
    summary = {
        "source": "real LongBench alternating within-dataset NLL route audit",
        "calibration_queries": len(calibration),
        "heldout_queries": len(test),
        "policy": policy,
        "comparisons": comparisons,
        "oracle_diagnostic": oracle_diagnostic,
        "conclusion": (
            "This small natural-data route does not beat the strongest global candidate; "
            "dataset-level calibration is not sufficient evidence for MoR-KV generalization."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
