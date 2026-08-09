from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


BASELINE_VARIANTS = (
    "native_full",
    "rope_top2",
    "local_global_postscore",
)
METHOD_VARIANT = "local_global_rephase05"
TUNING_SAMPLES = {
    "niah_multikey_3_32768_0",
    "niah_multivalue_32768_0",
}
ALPHA = {
    "local_global_postscore": 0.0,
    "local_global_rephase02": 0.02,
    "local_global_rephase05": 0.05,
    "local_global_rephase10": 0.10,
    "local_global_rephase15": 0.15,
    "local_global_rephase25": 0.25,
    "local_global_rephase50": 0.50,
    "local_global_rephase75": 0.75,
    "local_global_rephase100": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--rephase-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--smoke-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = path / "rows.jsonl" if path.is_dir() else path
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        result.setdefault((str(row["sample_id"]), str(row["variant"])), row)
    return list(result.values())


def optional_mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def paired_bootstrap(
    deltas: Sequence[float], resamples: int, seed: int
) -> list[float] | None:
    if not deltas:
        return None
    generator = random.Random(seed)
    draws = [
        mean(generator.choice(deltas) for _ in deltas)
        for _ in range(resamples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def task_macro(rows: Sequence[dict[str, Any]], variant: str) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["variant"] == variant:
            grouped[str(row["task"])].append(float(row["official_score"]))
    return mean(mean(values) for values in grouped.values()) if grouped else None


def summarize_split(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = (*BASELINE_VARIANTS, METHOD_VARIANT)
    overall: dict[str, Any] = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        niah = [
            row
            for row in subset
            if str(row["task"]).startswith("niah_")
            and int(row.get("answer_evidence_span_count", 0)) > 0
        ]
        overall[variant] = {
            "samples": len(subset),
            "tasks": len({str(row["task"]) for row in subset}),
            "macro_official_score_percent": (
                100.0 * float(task_macro(rows, variant) or 0.0)
            ),
            "sample_mean_official_score_percent": (
                100.0 * optional_mean(float(row["official_score"]) for row in subset)
                if subset
                else None
            ),
            "mean_first_answer_nll": optional_mean(
                float(row["first_answer_next_token_nll"]) for row in subset
            ),
            "mean_query_seconds": optional_mean(
                row.get("query_seconds") for row in subset
            ),
            "mean_generation_seconds": optional_mean(
                row.get("generation_seconds") for row in subset
            ),
            "mean_niah_evidence_recall": optional_mean(
                row.get("gold_evidence_token_recall") for row in niah
            ),
            "mean_niah_evidence_mass": optional_mean(
                row.get("gold_evidence_attention_mass") for row in niah
            ),
            "mean_rephase_gold_qk_delta": optional_mean(
                row.get("rephase_gold_score_delta_mean") for row in subset
                if "rephase" in str(row["variant"])
            ),
            "mean_rephase_nongold_qk_delta": optional_mean(
                row.get("rephase_nongold_score_delta_mean") for row in subset
                if "rephase" in str(row["variant"])
            ),
            "mean_rephase_gold_minus_nongold_delta": optional_mean(
                row.get("rephase_gold_minus_nongold_delta") for row in subset
                if "rephase" in str(row["variant"])
            ),
            "mean_rephase_abs_position_shift": optional_mean(
                row.get("rephase_mean_abs_position_shift") for row in subset
                if "rephase" in str(row["variant"])
            ),
            "mean_rephase_effective_distance": optional_mean(
                row.get("rephase_mean_effective_distance") for row in subset
                if "rephase" in str(row["variant"])
            ),
            "support_budget_violation_fraction": optional_mean(
                row.get("support_budget_violation_fraction") for row in subset
            ),
            "duplicate_support_violation_fraction": optional_mean(
                row.get("duplicate_support_violation_fraction") for row in subset
            ),
        }
    return overall


def comparisons(
    rows: Sequence[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    lookup = {
        (str(row["sample_id"]), str(row["variant"])): row for row in rows
    }
    output: dict[str, Any] = {}
    method_ids = {
        sample for sample, variant in lookup if variant == METHOD_VARIANT
    }
    for reference in BASELINE_VARIANTS:
        paired = sorted(
            method_ids
            & {sample for sample, variant in lookup if variant == reference}
        )
        score_deltas = [
            float(lookup[(sample, METHOD_VARIANT)]["official_score"])
            - float(lookup[(sample, reference)]["official_score"])
            for sample in paired
        ]
        nll_deltas = [
            float(lookup[(sample, METHOD_VARIANT)]["first_answer_next_token_nll"])
            - float(lookup[(sample, reference)]["first_answer_next_token_nll"])
            for sample in paired
        ]
        interval = paired_bootstrap(score_deltas, resamples, seed)
        output[f"{METHOD_VARIANT}_minus_{reference}"] = {
            "paired_samples": len(paired),
            "official_score_delta_points": 100.0 * mean(score_deltas),
            "official_score_delta_95ci_points": (
                [100.0 * value for value in interval] if interval else None
            ),
            "first_answer_nll_delta": mean(nll_deltas),
            "rescues": sum(delta > 1e-12 for delta in score_deltas),
            "harms": sum(delta < -1e-12 for delta in score_deltas),
            "unchanged": sum(abs(delta) <= 1e-12 for delta in score_deltas),
        }
    return output


def task_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["task"]))].append(
            float(row["official_score"])
        )
    return [
        {
            "variant": variant,
            "task": task,
            "samples": len(values),
            "official_score_percent": 100.0 * mean(values),
        }
        for (variant, task), values in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_smoke(smoke: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for sample in sorted({str(row["sample_id"]) for row in smoke}):
        points = sorted(
            (
                ALPHA[str(row["variant"])],
                float(row["official_score"]),
                float(row["first_answer_next_token_nll"]),
            )
            for row in smoke
            if str(row["sample_id"]) == sample
            and str(row["variant"]) in ALPHA
        )
        axes[0].plot(
            [point[0] for point in points],
            [100.0 * point[1] for point in points],
            marker="o",
            label=sample.replace("_32768_0", ""),
        )
        axes[1].plot(
            [point[0] for point in points],
            [point[2] for point in points],
            marker="o",
            label=sample.replace("_32768_0", ""),
        )
    axes[0].set_ylabel("Official score (%)")
    axes[1].set_ylabel("First-answer NLL")
    for axis in axes:
        axis.set_xlabel("Position interpolation alpha")
        axis.axvline(0.05, color="#8b5cf6", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("RULER-32K smoke sweep: position repair is non-monotonic")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_smoke_mechanism(
    smoke: Sequence[dict[str, Any]], output: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for sample in sorted({str(row["sample_id"]) for row in smoke}):
        points = sorted(
            (
                ALPHA[str(row["variant"])],
                100.0 * float(row.get("gold_evidence_attention_mass", 0.0)),
                float(row.get("rephase_gold_minus_nongold_delta", 0.0)),
            )
            for row in smoke
            if str(row["sample_id"]) == sample
            and str(row["variant"]) in ALPHA
        )
        label = sample.replace("_32768_0", "")
        axes[0].plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [point[0] for point in points],
            [point[2] for point in points],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Gold evidence attention mass (%)")
    axes[1].set_ylabel("Gold minus non-gold QK delta")
    axes[1].axhline(0, color="#ef4444", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xlabel("Position interpolation alpha")
        axis.axvline(0.05, color="#8b5cf6", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Smoke mechanism: moving closer does not selectively favor gold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_tasks(task_data: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    variants = (*BASELINE_VARIANTS, METHOD_VARIANT)
    tasks = sorted({str(row["task"]) for row in task_data})
    lookup = {
        (str(row["variant"]), str(row["task"])): float(
            row["official_score_percent"]
        )
        for row in task_data
    }
    width = 0.8 / len(variants)
    xs = list(range(len(tasks)))
    colors = ["#64748b", "#f59e0b", "#14b8a6", "#8b5cf6"]
    fig, axis = plt.subplots(figsize=(16, 5.8))
    for index, variant in enumerate(variants):
        offsets = [x - 0.4 + width / 2 + index * width for x in xs]
        axis.bar(
            offsets,
            [lookup.get((variant, task), 0.0) for task in tasks],
            width,
            label=variant,
            color=colors[index],
        )
    axis.set_xticks(xs)
    axis.set_xticklabels(tasks, rotation=35, ha="right")
    axis.set_ylabel("Official RULER score (%)")
    axis.set_title("Held-out RULER-32K samples (two tuning samples excluded)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_qk(rows: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    method = [
        row
        for row in rows
        if row["variant"] == METHOD_VARIANT
        and str(row["task"]).startswith("niah_")
        and int(row.get("answer_evidence_span_count", 0)) > 0
    ]
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    xs = [float(row["rephase_nongold_score_delta_mean"]) for row in method]
    ys = [float(row["rephase_gold_score_delta_mean"]) for row in method]
    scores = [float(row["official_score"]) for row in method]
    scatter = axis.scatter(xs, ys, c=scores, cmap="viridis", vmin=0, vmax=1, s=65)
    low = min([0.0, *xs, *ys])
    high = max([0.0, *xs, *ys])
    axis.plot([low, high], [low, high], linestyle="--", color="#ef4444", label="gold = non-gold")
    axis.axhline(0, color="#94a3b8", linewidth=0.8)
    axis.axvline(0, color="#94a3b8", linewidth=0.8)
    axis.set_xlabel("Mean non-gold QK delta")
    axis.set_ylabel("Mean gold QK delta")
    axis.set_title("5% repair: does phase movement favor evidence?")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.colorbar(scatter, ax=axis, label="Official score")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline = [
        row
        for row in read_rows(args.baseline_inputs)
        if str(row["variant"]) in BASELINE_VARIANTS
    ]
    method = [
        row
        for row in read_rows(args.rephase_inputs)
        if str(row["variant"]) == METHOD_VARIANT
    ]
    combined = unique_rows([*baseline, *method])
    heldout = [
        row for row in combined if str(row["sample_id"]) not in TUNING_SAMPLES
    ]
    smoke = unique_rows(read_rows(args.smoke_inputs))
    summary = {
        "protocol": {
            "selected_alpha": 0.05,
            "tuning_samples": sorted(TUNING_SAMPLES),
            "primary_split": "all paired samples except the two tuning samples",
        },
        "all_26": {
            "overall": summarize_split(combined),
            "comparisons": comparisons(
                combined, args.bootstrap_resamples, args.seed
            ),
        },
        "heldout_24": {
            "overall": summarize_split(heldout),
            "comparisons": comparisons(
                heldout, args.bootstrap_resamples, args.seed + 100
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    heldout_tasks = task_rows(heldout)
    write_csv(args.output_dir / "heldout_task_scores.csv", heldout_tasks)
    write_csv(args.output_dir / "all_rows.csv", combined)
    write_csv(args.output_dir / "smoke_rows.csv", smoke)
    plot_smoke(smoke, args.output_dir / "smoke_alpha_sweep.png")
    plot_smoke_mechanism(smoke, args.output_dir / "smoke_mass_qk.png")
    plot_tasks(heldout_tasks, args.output_dir / "heldout_task_scores.png")
    plot_qk(heldout, args.output_dir / "heldout_niah_qk_delta.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
