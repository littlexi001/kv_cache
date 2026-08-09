from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge causal sparse-reference NLL shards.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--router_bundle_test_only",
        help="Optional deployment bundle; retain only its query-disjoint test IDs.",
    )
    return parser.parse_args()


def paired_bootstrap_ci(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for raw_path in args.input:
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        **raw,
                        "query_id": int(raw["query_id"]),
                        "context_tokens": int(raw["context_tokens"]),
                        "target_token_id": int(raw["target_token_id"]),
                        "nll": float(raw["nll"]),
                        "delta_nll_vs_full": float(raw["delta_nll_vs_full"]),
                        "mean_selected_blocks": float(raw["mean_selected_blocks"]),
                        "mean_relative_output_l2": float(raw["mean_relative_output_l2"]),
                        "p95_relative_output_l2": float(raw["p95_relative_output_l2"]),
                        "violation_rate": float(raw["violation_rate"]),
                        "mean_physical_gqa_blocks": float(
                            raw.get("mean_physical_gqa_blocks", raw["mean_selected_blocks"])
                        ),
                        "physical_gqa_saving_rate": float(
                            raw.get("physical_gqa_saving_rate", 0.0)
                        ),
                        "mean_layer_global_blocks": float(
                            raw.get("mean_layer_global_blocks", raw["mean_selected_blocks"])
                        ),
                        "max_layer_global_blocks": int(
                            float(raw.get("max_layer_global_blocks", raw["mean_selected_blocks"]))
                        ),
                        "mean_layer_global_tokens": float(
                            raw.get(
                                "mean_layer_global_tokens",
                                256.0 * float(raw["mean_selected_blocks"]),
                            )
                        ),
                        "max_layer_global_tokens": int(
                            float(
                                raw.get(
                                    "max_layer_global_tokens",
                                    256.0 * float(raw["mean_selected_blocks"]),
                                )
                            )
                        ),
                        "strict_1000_token_violation_rate": float(
                            raw.get("strict_1000_token_violation_rate", 0.0)
                        ),
                        "mean_router_upper_bound": float(
                            raw.get("mean_router_upper_bound", 0.0)
                        ),
                        "p95_router_upper_bound": float(
                            raw.get("p95_router_upper_bound", 0.0)
                        ),
                        "max_router_upper_bound": float(
                            raw.get("max_router_upper_bound", 0.0)
                        ),
                        "router_near_threshold_fraction": float(
                            raw.get("router_near_threshold_fraction", 0.0)
                        ),
                    }
                )
    if args.router_bundle_test_only:
        bundle = json.loads(
            Path(args.router_bundle_test_only).read_text(encoding="utf-8")
        )
        allowed = set(int(item) for item in bundle["test_query_ids"])
        rows = [row for row in rows if int(row["query_id"]) in allowed]
    rows.sort(key=lambda row: (row["query_id"], row["action"]))
    with (output_dir / "reference_nll_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    action_summary: list[dict[str, Any]] = []
    for action in sorted({str(row["action"]) for row in rows}):
        group = [row for row in rows if row["action"] == action]
        deltas = np.asarray([float(row["delta_nll_vs_full"]) for row in group])
        ci_low, ci_high = paired_bootstrap_ci(
            deltas, args.bootstrap_samples, args.seed
        )
        action_summary.append(
            {
                "action": action,
                "queries": len(group),
                "mean_nll": statistics.fmean(float(row["nll"]) for row in group),
                "mean_delta_nll_vs_full": float(deltas.mean()),
                "delta_nll_ci95_low": ci_low,
                "delta_nll_ci95_high": ci_high,
                "median_delta_nll_vs_full": float(np.median(deltas)),
                "p95_abs_delta_nll": float(np.quantile(np.abs(deltas), 0.95)),
                "fraction_nll_not_worse": float(np.mean(deltas <= 0.0)),
                "mean_selected_blocks": statistics.fmean(
                    float(row["mean_selected_blocks"]) for row in group
                ),
                "mean_head_relative_output_l2": statistics.fmean(
                    float(row["mean_relative_output_l2"]) for row in group
                ),
                "mean_head_violation_rate": statistics.fmean(
                    float(row["violation_rate"]) for row in group
                ),
                "mean_physical_gqa_blocks": statistics.fmean(
                    float(row["mean_physical_gqa_blocks"]) for row in group
                ),
                "mean_physical_gqa_saving_rate": statistics.fmean(
                    float(row["physical_gqa_saving_rate"]) for row in group
                ),
                "mean_layer_global_blocks": statistics.fmean(
                    float(row["mean_layer_global_blocks"]) for row in group
                ),
                "max_layer_global_blocks": max(
                    int(row["max_layer_global_blocks"]) for row in group
                ),
                "mean_layer_global_tokens": statistics.fmean(
                    float(row["mean_layer_global_tokens"]) for row in group
                ),
                "max_layer_global_tokens": max(
                    int(row["max_layer_global_tokens"]) for row in group
                ),
                "strict_1000_token_violation_rate": statistics.fmean(
                    float(row["strict_1000_token_violation_rate"]) for row in group
                ),
                "mean_router_upper_bound": statistics.fmean(
                    float(row["mean_router_upper_bound"]) for row in group
                ),
                "mean_router_near_threshold_fraction": statistics.fmean(
                    float(row["router_near_threshold_fraction"]) for row in group
                ),
            }
        )
    with (output_dir / "action_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_summary[0]))
        writer.writeheader()
        writer.writerows(action_summary)

    nonfull = [row for row in rows if row["action"] != "full"]
    errors = np.asarray([float(row["mean_relative_output_l2"]) for row in nonfull])
    deltas = np.asarray([float(row["delta_nll_vs_full"]) for row in nonfull])
    summary = {
        "queries": len({int(row["query_id"]) for row in rows}),
        "rows": len(rows),
        "action_summary": action_summary,
        "mean_head_error_delta_nll_pearson": float(
            np.corrcoef(errors, deltas)[0, 1]
        ),
        "interpretation": (
            "Paired deltas compare the same query under a causal final-prompt-token "
            "attention-output intervention versus full attention."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
