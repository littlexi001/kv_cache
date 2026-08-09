from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("shard*/rows.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], seed: int = 20260807, draws: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_two_sided_sign_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    extreme = min(positive, negative)
    tail = sum(math.comb(n, k) for k in range(extreme + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_values(
    by_sample: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    variant: str,
    transform: Callable[[dict[str, Any], dict[str, Any]], float],
) -> list[float]:
    return [
        transform(variants[baseline], variants[variant])
        for _, variants in sorted(by_sample.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="native_rope")
    parser.add_argument("--baseline-run-dir", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.run_dir)
    if args.baseline_run_dir is not None:
        rows.extend(
            row for row in read_rows(args.baseline_run_dir)
            if str(row.get("variant")) == args.baseline
        )
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row["sample_id"])][str(row["variant"])] = row
    variants = sorted({str(row["variant"]) for row in rows})
    expected = set(variants)
    incomplete = [sample for sample, values in by_sample.items() if set(values) != expected]
    if incomplete:
        raise RuntimeError(f"incomplete samples: {incomplete}")

    output: dict[str, Any] = {"samples": len(by_sample), "baseline": args.baseline}
    comparisons: list[dict[str, Any]] = []
    for variant in variants:
        if variant == args.baseline:
            continue
        score_delta = paired_values(
            by_sample,
            args.baseline,
            variant,
            lambda native, current: float(current["official_score"])
            - float(native["official_score"]),
        )
        nll_improvement = paired_values(
            by_sample,
            args.baseline,
            variant,
            lambda native, current: float(native["gold_answer_mean_nll"])
            - float(current["gold_answer_mean_nll"]),
        )
        score_positive = sum(value > 1e-12 for value in score_delta)
        score_negative = sum(value < -1e-12 for value in score_delta)
        nll_positive = sum(value > 0 for value in nll_improvement)
        nll_negative = sum(value < 0 for value in nll_improvement)
        comparisons.append(
            {
                "variant": variant,
                "score_delta_mean": mean(score_delta),
                "score_delta_bootstrap_95ci": bootstrap_mean_ci(score_delta),
                "score_improved_tied_degraded": [
                    score_positive,
                    len(score_delta) - score_positive - score_negative,
                    score_negative,
                ],
                "score_sign_test_p": exact_two_sided_sign_p(
                    score_positive, score_negative
                ),
                "nll_improvement_mean": mean(nll_improvement),
                "nll_improvement_median": median(nll_improvement),
                "nll_improvement_bootstrap_95ci": bootstrap_mean_ci(
                    nll_improvement, seed=20260808
                ),
                "nll_improved_degraded": [nll_positive, nll_negative],
                "nll_sign_test_p": exact_two_sided_sign_p(nll_positive, nll_negative),
            }
        )
    output["comparisons"] = comparisons

    task_rows: list[dict[str, Any]] = []
    tasks = sorted({str(row["task"]) for row in rows})
    for task in tasks:
        sample_ids = sorted(
            sample_id
            for sample_id, values in by_sample.items()
            if str(values[args.baseline]["task"]) == task
        )
        for variant in variants:
            selected = [by_sample[sample_id][variant] for sample_id in sample_ids]
            task_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "samples": len(selected),
                    "official_score_mean": mean(
                        float(row["official_score"]) for row in selected
                    ),
                    "gold_nll_mean": mean(
                        float(row["gold_answer_mean_nll"]) for row in selected
                    ),
                }
            )
    output["by_task"] = task_rows

    (args.run_dir / "paired_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 26-sample paired validation",
        "",
        "| Variant | Δ official (95% bootstrap CI) | improved/tied/degraded | Δ Gold NLL (95% bootstrap CI) | NLL +/- | sign p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        score_ci = row["score_delta_bootstrap_95ci"]
        nll_ci = row["nll_improvement_bootstrap_95ci"]
        score_counts = "/".join(map(str, row["score_improved_tied_degraded"]))
        nll_counts = "/".join(map(str, row["nll_improved_degraded"]))
        lines.append(
            f"| {row['variant']} | {row['score_delta_mean']:+.4f} "
            f"[{score_ci[0]:+.4f}, {score_ci[1]:+.4f}] | {score_counts} | "
            f"{row['nll_improvement_mean']:+.4f} [{nll_ci[0]:+.4f}, {nll_ci[1]:+.4f}] | "
            f"{nll_counts} | {row['nll_sign_test_p']:.3g} |"
        )
    (args.run_dir / "paired_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
