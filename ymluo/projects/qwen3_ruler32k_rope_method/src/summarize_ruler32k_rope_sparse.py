from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", default="rope_top2")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = path / "rows.jsonl" if path.is_dir() else path
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["variant"]))
        if key in unique and unique[key] != row:
            raise RuntimeError(f"conflicting duplicate row: {key}")
        unique[key] = row
    return list(unique.values())


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def paired_bootstrap(deltas: Sequence[float], resamples: int, seed: int) -> list[float] | None:
    if not deltas or resamples <= 0:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draws.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict[str, Any]], reference: str, resamples: int, seed: int) -> dict[str, Any]:
    by_key = {(str(row["sample_id"]), str(row["variant"])): row for row in rows}
    variants = sorted({str(row["variant"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    task_rows: list[dict[str, Any]] = []
    overall: dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        task_means = []
        for task in tasks:
            subset = [row for row in variant_rows if row["task"] == task]
            if not subset:
                continue
            task_score = mean([float(row["official_score"]) for row in subset])
            task_means.append(float(task_score))
            task_rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "samples": len(subset),
                    "official_score_percent": 100.0 * float(task_score),
                    "mean_first_answer_nll": mean([float(row["first_answer_next_token_nll"]) for row in subset]),
                    "mean_evidence_recall": mean([
                        float(row["gold_evidence_token_recall"])
                        for row in subset
                        if int(row.get("answer_evidence_span_count", 0)) > 0 and row.get("gold_evidence_token_recall") is not None
                    ]),
                    "mean_evidence_mass": mean([
                        float(row["gold_evidence_attention_mass"])
                        for row in subset
                        if int(row.get("answer_evidence_span_count", 0)) > 0 and row.get("gold_evidence_attention_mass") is not None
                    ]),
                }
            )
        overall[variant] = {
            "samples": len(variant_rows),
            "tasks": len(task_means),
            "macro_official_score_percent": 100.0 * float(mean(task_means) or 0.0),
            "mean_first_answer_nll": mean([float(row["first_answer_next_token_nll"]) for row in variant_rows]),
            "mean_query_seconds": mean([float(row["query_seconds"]) for row in variant_rows]),
            "mean_generation_seconds": mean([float(row["generation_seconds"]) for row in variant_rows]),
            "mean_niah_answer_evidence_recall": mean([
                float(row["gold_evidence_token_recall"])
                for row in variant_rows
                if str(row["task"]).startswith("niah_")
                and int(row.get("answer_evidence_span_count", 0)) > 0
                and row.get("gold_evidence_token_recall") is not None
            ]),
            "mean_niah_answer_evidence_mass": mean([
                float(row["gold_evidence_attention_mass"])
                for row in variant_rows
                if str(row["task"]).startswith("niah_")
                and int(row.get("answer_evidence_span_count", 0)) > 0
                and row.get("gold_evidence_attention_mass") is not None
            ]),
            "support_budget_violation_fraction": mean([
                float(row["support_budget_violation_fraction"])
                for row in variant_rows if row.get("support_budget_violation_fraction") is not None
            ]),
            "duplicate_support_violation_fraction": mean([
                float(row["duplicate_support_violation_fraction"])
                for row in variant_rows if row.get("duplicate_support_violation_fraction") is not None
            ]),
        }

    comparisons: dict[str, Any] = {}
    reference_ids = {sample for sample, variant in by_key if variant == reference}
    for variant in variants:
        if variant == reference:
            continue
        paired_ids = sorted(reference_ids & {sample for sample, method in by_key if method == variant})
        score_deltas = [
            float(by_key[(sample, variant)]["official_score"])
            - float(by_key[(sample, reference)]["official_score"])
            for sample in paired_ids
        ]
        nll_deltas = [
            float(by_key[(sample, variant)]["first_answer_next_token_nll"])
            - float(by_key[(sample, reference)]["first_answer_next_token_nll"])
            for sample in paired_ids
        ]
        score_interval = paired_bootstrap(score_deltas, resamples, seed)
        comparisons[f"{variant}_minus_{reference}"] = {
            "paired_samples": len(paired_ids),
            "official_score_delta_points": 100.0 * float(mean(score_deltas) or 0.0),
            "official_score_delta_95ci_points": (
                [100.0 * value for value in score_interval]
                if score_interval is not None else None
            ),
            "first_answer_nll_delta": mean(nll_deltas),
            "first_answer_nll_delta_95ci": paired_bootstrap(nll_deltas, resamples, seed + 1),
            "rescues": sum(
                float(by_key[(sample, reference)]["official_score"]) == 0.0
                and float(by_key[(sample, variant)]["official_score"]) > 0.0
                for sample in paired_ids
            ),
            "harms": sum(
                float(by_key[(sample, reference)]["official_score"]) > 0.0
                and float(by_key[(sample, variant)]["official_score"]) == 0.0
                for sample in paired_ids
            ),
            "improved_samples": sum(delta > 1e-12 for delta in score_deltas),
            "worsened_samples": sum(delta < -1e-12 for delta in score_deltas),
            "unchanged_samples": sum(abs(delta) <= 1e-12 for delta in score_deltas),
        }
    return {"rows": len(rows), "variants": variants, "tasks": tasks, "overall": overall, "comparisons": comparisons, "task_rows": task_rows}


def plot_task_scores(task_rows: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    variants = ["native_full", "rope_top2", "local_global_postscore", "local_global_blend25"]
    present = [variant for variant in variants if any(row["variant"] == variant for row in task_rows)]
    tasks = sorted({str(row["task"]) for row in task_rows})
    lookup = {(row["variant"], row["task"]): float(row["official_score_percent"]) for row in task_rows}
    width = 0.8 / max(1, len(present))
    xs = list(range(len(tasks)))
    fig, axis = plt.subplots(figsize=(16, 6))
    colors = ["#64748b", "#f59e0b", "#14b8a6", "#8b5cf6"]
    for index, variant in enumerate(present):
        offsets = [x - 0.4 + width / 2 + index * width for x in xs]
        axis.bar(offsets, [lookup.get((variant, task), 0.0) for task in tasks], width, label=variant, color=colors[index])
    axis.set_ylabel("Official RULER score (%)")
    sample_count = sum(int(row["samples"]) for row in task_rows if row["variant"] == present[0])
    samples_per_task = sample_count // max(1, len(tasks))
    axis.set_title(
        f"Qwen3-8B, RULER-32K pilot — {samples_per_task} samples/task "
        f"({sample_count} total), paired 2% support"
    )
    axis.set_xticks(xs)
    axis.set_xticklabels(tasks, rotation=35, ha="right")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    summary = summarize(rows, args.reference, args.bootstrap_resamples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "task_scores.csv", summary["task_rows"])
    write_csv(args.output_dir / "rows.csv", rows)
    plot_task_scores(summary["task_rows"], args.output_dir / "task_scores.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
