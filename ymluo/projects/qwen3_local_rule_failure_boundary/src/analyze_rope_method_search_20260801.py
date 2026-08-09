from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


METRICS = (
    "gold_evidence_token_recall",
    "gold_evidence_line_hit_rate",
    "gold_chain_complete_rate",
    "gold_evidence_attention_mass",
    "next_token_correct",
    "query_seconds",
    "phase_rescue_trigger_fraction",
    "phase_rescue_score_lift_mean",
    "phase_rescue_shift_rms_mean",
    "phase_rescue_active_planes_mean",
    "phase_rescue_realized_lift_ratio_mean",
    "phase_rescue_negative_lift_fraction",
    "gold_phase_trigger_fraction",
    "selected_gold_phase_trigger_fraction",
    "gold_phase_score_lift_mean",
    # Strict sparse/trust-region phase-rescue diagnostics.
    "strict_phase_solver_calls",
    "strict_phase_feasible_fraction",
    "strict_phase_target_lift_mean",
    "strict_phase_solver_lift_mean",
    "strict_phase_applied_lift_mean",
    "strict_phase_support_mean",
    "strict_phase_support_max",
    "strict_phase_budget_mean",
    "strict_phase_cap_mean",
    "strict_phase_shift_abs_max",
    "strict_phase_nontrigger_noop_max",
    "strict_phase_random_support",
    "strict_phase_exact_pre_selector",
    # Native Phase Envelope / coherent-distance rollback diagnostics.
    "npe_support_count",
    "npe_remote_support_count",
    "npe_certificate_trigger_count",
    "npe_certificate_trigger_fraction",
    "npe_applied_count",
    "npe_applied_fraction",
    "npe_search_success_fraction",
    "npe_suppression_gap_mean",
    "npe_local_anchor_median_mean",
    "npe_local_anchor_mad_mean",
    "npe_score_lift_mean",
    "npe_rollback_tokens_mean",
    "npe_rollback_tokens_median",
    "npe_rollback_tokens_p90",
    "npe_rollback_tokens_p95",
    "npe_rollback_tokens_max",
    "npe_effective_distance_mean",
    "npe_gold_certificate_trigger_fraction",
    "npe_unmodified_native_max_error",
    # Block-transport diagnostics, retained so negative screens stay auditable.
    "block_selected_count",
    "block_trigger_count",
    "block_trigger_fraction",
    "random_matched_trigger_fraction",
    "block_relative_distance_error_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="full_rope")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(root.rglob("rows.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rows.jsonl under {root}")
    seen: set[tuple[int, int, str]] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                int(row["target_context_tokens"]),
                int(row["seed"]),
                str(row["variant"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else float("nan")


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    index = probability * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(
    rows: Sequence[dict[str, Any]],
    baseline: str,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        length = int(row["target_context_tokens"])
        variant = str(row["variant"])
        seed = int(row["seed"])
        grouped[(length, variant)].append(row)
        lookup[(length, seed, variant)] = row

    output: list[dict[str, Any]] = []
    for (length, variant), items in sorted(grouped.items()):
        mean_nll = mean(float(item["gold_nll"]) for item in items)
        summary: dict[str, Any] = {
            "target_context_tokens": length,
            "variant": variant,
            "sample_count": len(items),
            "mean_gold_nll": mean_nll,
            "gold_ppl": math.exp(mean_nll),
        }
        for metric in METRICS:
            available = [float(item[metric]) for item in items if metric in item]
            if available:
                summary[metric] = mean(available)

        paired = []
        accuracy_delta = []
        for item in items:
            base = lookup.get((length, int(item["seed"]), baseline))
            if base is None:
                continue
            paired.append(float(item["gold_nll"]) - float(base["gold_nll"]))
            accuracy_delta.append(
                float(item["next_token_correct"])
                - float(base["next_token_correct"])
            )
        if paired:
            delta = mean(paired)
            low, high = bootstrap_mean_ci(
                paired,
                bootstrap_samples,
                seed=20260801 + length + sum(ord(char) for char in variant),
            )
            summary.update(
                {
                    "paired_count_vs_baseline": len(paired),
                    "delta_nll_vs_baseline": delta,
                    "delta_nll_ci95_low": low,
                    "delta_nll_ci95_high": high,
                    "ppl_ratio_vs_baseline": math.exp(delta),
                    "nll_win_rate_vs_baseline": mean(value < 0.0 for value in paired),
                    "delta_accuracy_vs_baseline": mean(accuracy_delta),
                }
            )
        output.append(summary)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: Sequence[dict[str, Any]], baseline: str) -> str:
    lines = [
        "# RoPE method screen",
        "",
        f"Paired baseline: `{baseline}`. PPL is `exp(mean NLL)` across matched seeds.",
        "",
    ]
    for length in sorted({int(row["target_context_tokens"]) for row in summary}):
        rows = [row for row in summary if int(row["target_context_tokens"]) == length]
        rows.sort(key=lambda row: float(row["mean_gold_nll"]))
        lines.extend(
            [
                f"## {length:,} tokens",
                "",
                f"| Variant | n | Gold PPL | Accuracy | Evidence recall | Evidence mass | ΔNLL vs `{baseline}` [95% CI] |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            delta = float(row.get("delta_nll_vs_baseline", float("nan")))
            low = float(row.get("delta_nll_ci95_low", float("nan")))
            high = float(row.get("delta_nll_ci95_high", float("nan")))
            lines.append(
                "| {variant} | {n} | {ppl:.3f} | {acc:.1%} | {recall:.1%} | "
                "{mass:.3%} | {delta:+.3f} [{low:+.3f}, {high:+.3f}] |".format(
                    variant=row["variant"],
                    n=row["sample_count"],
                    ppl=float(row["gold_ppl"]),
                    acc=float(row.get("next_token_correct", float("nan"))),
                    recall=float(
                        row.get("gold_evidence_token_recall", float("nan"))
                    ),
                    mass=float(
                        row.get("gold_evidence_attention_mass", float("nan"))
                    ),
                    delta=delta,
                    low=low,
                    high=high,
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir)
    summary = summarize(rows, args.baseline, args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.md").write_text(
        markdown(summary, args.baseline),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
