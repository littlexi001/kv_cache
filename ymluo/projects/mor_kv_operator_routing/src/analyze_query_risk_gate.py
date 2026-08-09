from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASE_RISK_SIGNALS = (
    "mean_router_upper_bound",
    "p95_router_upper_bound",
    "max_router_upper_bound",
    "router_near_threshold_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deployable query-level fallback gates for a sparse KV policy."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--action", default="learned_conformal")
    parser.add_argument(
        "--fallback_fractions",
        default="0,0.05,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated fractions of highest-risk queries sent to full attention.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--layer_weights",
        help=(
            "Optional comma-separated inclusive ranges and calibration-frozen weights, "
            "for example 0-6:0.214,7-13:0.344,21-27:0.348."
        ),
    )
    return parser.parse_args()


def parse_layer_weights(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    weights: dict[int, float] = {}
    for item in raw.split(","):
        layer_range, raw_weight = item.split(":", maxsplit=1)
        start, end = [int(value) for value in layer_range.split("-", maxsplit=1)]
        for layer in range(start, end + 1):
            weights[layer] = float(raw_weight)
    return weights


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = [float(item) for item in args.fallback_fractions.split(",")]
    layer_weights = parse_layer_weights(args.layer_weights)

    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["action"] == args.action]
    if not rows:
        raise ValueError(f"No rows found for action={args.action!r}")

    delta = np.asarray([float(row["delta_nll_vs_full"]) for row in rows])
    saving = np.asarray([float(row["physical_gqa_saving_rate"]) for row in rows])
    signals = {
        signal_name: np.asarray([float(row[signal_name]) for row in rows])
        for signal_name in BASE_RISK_SIGNALS
    }
    if layer_weights:
        layer_risk = []
        for row in rows:
            by_layer = {
                int(layer): float(value)
                for layer, value in json.loads(
                    row["mean_router_upper_bound_by_layer"]
                ).items()
            }
            layer_risk.append(
                sum(
                    layer_weights.get(layer, 0.0) * value
                    for layer, value in by_layer.items()
                )
            )
        signals["amplification_weighted_router_upper_bound"] = np.asarray(layer_risk)
    rng = np.random.default_rng(args.seed)
    correlations: list[dict[str, float | str]] = []
    gate_rows: list[dict[str, float | int | str]] = []

    for signal_name, signal in signals.items():
        signal_ranks = average_ranks(signal)
        correlations.append(
            {
                "signal": signal_name,
                "pearson_signed_delta": correlation(signal, delta),
                "spearman_signed_delta": correlation(signal_ranks, average_ranks(delta)),
                "pearson_abs_delta": correlation(signal, np.abs(delta)),
                "spearman_abs_delta": correlation(
                    signal_ranks, average_ranks(np.abs(delta))
                ),
            }
        )
        descending = np.argsort(-signal, kind="mergesort")
        for fraction in fractions:
            fallback_count = min(len(rows), int(np.ceil(fraction * len(rows))))
            fallback = np.zeros(len(rows), dtype=bool)
            fallback[descending[:fallback_count]] = True
            deployed_delta = np.where(fallback, 0.0, delta)
            deployed_saving = np.where(fallback, 0.0, saving)
            ci_low, ci_high = bootstrap_mean_ci(
                deployed_delta, args.bootstrap_samples, rng
            )
            gate_rows.append(
                {
                    "signal": signal_name,
                    "requested_fallback_fraction": fraction,
                    "fallback_queries": fallback_count,
                    "actual_fallback_fraction": fallback_count / len(rows),
                    "mean_physical_gqa_saving_rate": float(deployed_saving.mean()),
                    "mean_delta_nll_vs_full": float(deployed_delta.mean()),
                    "delta_nll_ci95_low": ci_low,
                    "delta_nll_ci95_high": ci_high,
                    "median_delta_nll_vs_full": float(np.median(deployed_delta)),
                    "p95_abs_delta_nll": float(np.quantile(np.abs(deployed_delta), 0.95)),
                }
            )

    with (output_dir / "risk_gate_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)

    summary = {
        "action": args.action,
        "queries": len(rows),
        "baseline_mean_physical_gqa_saving_rate": float(saving.mean()),
        "baseline_mean_delta_nll_vs_full": float(delta.mean()),
        "baseline_p95_abs_delta_nll": float(np.quantile(np.abs(delta), 0.95)),
        "correlations": correlations,
        "interpretation": (
            "Each gate falls back the highest deployable risk-signal values to full "
            "attention. This is post-hoc diagnostic evidence and requires a fresh "
            "holdout before it can be reported as a validated policy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
