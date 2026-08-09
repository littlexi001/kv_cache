from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("shard*/rows.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def bootstrap_ci(values: list[float], repeats: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(20260804)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return samples[int(0.025 * repeats)], samples[int(0.975 * repeats)]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.run_dir)
    if not rows:
        raise RuntimeError("no rows")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_sample[row["sample_id"]][row["variant"]] = row

    native = by_variant.get("native_rope", [])
    expected = {row["sample_id"] for row in native}
    summaries: list[dict[str, Any]] = []
    for variant, selected in sorted(by_variant.items()):
        paired = []
        for sample_id in sorted(expected):
            pair = by_sample[sample_id]
            if variant in pair and "native_rope" in pair:
                paired.append(float(pair[variant]["official_score"]) - float(pair["native_rope"]["official_score"]))
        lo, hi = bootstrap_ci(paired)
        task_means = defaultdict(list)
        for row in selected:
            task_means[row["task"]].append(float(row["official_score"]))
        mean_nll = mean(float(row["gold_answer_mean_nll"]) for row in selected)
        summaries.append({
            "variant": variant,
            "samples": len(selected),
            "official_score_mean": mean(float(row["official_score"]) for row in selected),
            "task_macro_score": mean(mean(values) for values in task_means.values()),
            "first_token_accuracy": mean(float(row["first_answer_next_token_correct"]) for row in selected),
            "first_token_mean_nll": mean(float(row["first_answer_next_token_nll"]) for row in selected),
            "gold_answer_mean_nll": mean_nll,
            "gold_answer_ppl_from_mean_nll": math.exp(min(mean_nll, 30.0)),
            "paired_score_delta_vs_native": mean(paired) if paired else None,
            "paired_delta_ci95_low": lo if paired else None,
            "paired_delta_ci95_high": hi if paired else None,
            "improved": sum(value > 1e-12 for value in paired),
            "degraded": sum(value < -1e-12 for value in paired),
            "tied": sum(abs(value) <= 1e-12 for value in paired),
            "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in selected),
        })

    (args.run_dir / "merged_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = ["# 推理时 RNoPE：RULER-32K 结果", "", "| 条件 | 样本 | 官方分数 | 首 token 准确率 | Gold PPL | 相对原生差值 | 95% CI | 改善/退化/持平 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        delta = row["paired_score_delta_vs_native"]
        ci = "—" if delta is None else f"[{row['paired_delta_ci95_low']:.4f}, {row['paired_delta_ci95_high']:.4f}]"
        lines.append(
            f"| {row['variant']} | {row['samples']} | {row['official_score_mean']:.4f} | "
            f"{row['first_token_accuracy']:.4f} | {row['gold_answer_ppl_from_mean_nll']:.3f} | "
            f"{'—' if delta is None else f'{delta:+.4f}'} | {ci} | "
            f"{row['improved']}/{row['degraded']}/{row['tied']} |"
        )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

