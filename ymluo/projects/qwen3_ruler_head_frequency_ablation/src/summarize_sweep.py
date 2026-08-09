from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = set(run_dir.glob("shard*/rows.jsonl")) | set(run_dir.glob("spec*/rows.jsonl"))
    for path in sorted(paths):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="native_rope")
    parser.add_argument("--baseline-run-dir", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.run_dir)
    if args.baseline_run_dir is not None:
        rows.extend(
            row for row in read_rows(args.baseline_run_dir)
            if str(row.get("variant")) == args.baseline
        )
    if not rows:
        raise RuntimeError("no rows")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_sample[row["sample_id"]][row["variant"]] = row
    if args.baseline not in by_variant:
        raise RuntimeError(f"missing baseline {args.baseline}")
    expected = {row["sample_id"] for row in by_variant[args.baseline]}
    summaries: list[dict[str, Any]] = []
    for variant, selected in by_variant.items():
        if {row["sample_id"] for row in selected} != expected:
            raise RuntimeError(f"incomplete samples for {variant}")
        score_deltas: list[float] = []
        nll_improvements: list[float] = []
        first_deltas: list[float] = []
        if variant != args.baseline:
            for sample_id in sorted(expected):
                native = by_sample[sample_id][args.baseline]
                current = by_sample[sample_id][variant]
                score_deltas.append(float(current["official_score"]) - float(native["official_score"]))
                nll_improvements.append(float(native["gold_answer_mean_nll"]) - float(current["gold_answer_mean_nll"]))
                first_deltas.append(float(current["first_answer_next_token_correct"]) - float(native["first_answer_next_token_correct"]))
        clipped = [max(-2.0, min(2.0, value)) for value in nll_improvements]
        official_mean = mean(float(row["official_score"]) for row in selected)
        mean_nll = mean(float(row["gold_answer_mean_nll"]) for row in selected)
        score_delta = mean(score_deltas) if score_deltas else 0.0
        clipped_nll = mean(clipped) if clipped else 0.0
        summaries.append({
            "variant": variant,
            "samples": len(selected),
            "spec": selected[0]["spec"],
            "official_score_mean": official_mean,
            "first_token_accuracy": mean(float(row["first_answer_next_token_correct"]) for row in selected),
            "gold_answer_mean_nll": mean_nll,
            "gold_answer_ppl_from_mean_nll": math.exp(min(mean_nll, 30.0)),
            "paired_official_delta": score_delta,
            "mean_nll_improvement": mean(nll_improvements) if nll_improvements else 0.0,
            "median_nll_improvement": median(nll_improvements) if nll_improvements else 0.0,
            "mean_clipped_nll_improvement": clipped_nll,
            "first_token_accuracy_delta": mean(first_deltas) if first_deltas else 0.0,
            "utility": score_delta + 0.05 * clipped_nll,
            "improved_score_samples": sum(value > 1e-12 for value in score_deltas),
            "degraded_score_samples": sum(value < -1e-12 for value in score_deltas),
            "improved_nll_samples": sum(value > 0 for value in nll_improvements),
            "degraded_nll_samples": sum(value < 0 for value in nll_improvements),
            "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in selected),
        })
    summaries.sort(key=lambda row: (row["utility"], row["official_score_mean"]), reverse=True)
    (args.run_dir / "merged_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat = [{key: value for key, value in row.items() if key != "spec"} for row in summaries]
    with (args.run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# RoPE head-group × frequency sweep",
        "",
        "| Rank | Variant | Official | Δ score | Gold NLL | clipped ΔNLL | Utility | score improved/degraded |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summaries[:30], start=1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['official_score_mean']:.4f} | "
            f"{row['paired_official_delta']:+.4f} | {row['gold_answer_mean_nll']:.3f} | "
            f"{row['mean_clipped_nll_improvement']:+.3f} | {row['utility']:+.4f} | "
            f"{row['improved_score_samples']}/{row['degraded_score_samples']} |"
        )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
