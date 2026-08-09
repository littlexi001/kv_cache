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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile_interval(values: list[float], seed: int, repeats: int = 20000) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [draws[int(0.025 * repeats)], draws[int(0.975 * repeats) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True, help="SEED=RUN_DIR")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="native_rope")
    args = parser.parse_args()
    all_rows: list[dict[str, Any]] = []
    for value in args.seed_run:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        for row in read_jsonl(Path(path_text) / "merged_rows.jsonl"):
            all_rows.append({**row, "evaluation_seed": seed})
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        by_variant[row["variant"]].append(row)
        by_case[(row["evaluation_seed"], row["sample_id"])][row["variant"]] = row
    if args.baseline not in by_variant:
        raise RuntimeError(f"missing baseline: {args.baseline}")
    expected = set(by_case)
    summaries: list[dict[str, Any]] = []
    for variant, rows in by_variant.items():
        cases = {(row["evaluation_seed"], row["sample_id"]) for row in rows}
        if cases != expected:
            raise RuntimeError(f"incomplete cases for {variant}: {len(cases)} vs {len(expected)}")
        score_deltas: list[float] = []
        nll_improvements: list[float] = []
        per_seed: list[dict[str, Any]] = []
        for seed in sorted({case[0] for case in expected}):
            seed_cases = sorted(case for case in expected if case[0] == seed)
            seed_score_deltas = []
            seed_nll_improvements = []
            for case in seed_cases:
                native = by_case[case][args.baseline]
                current = by_case[case][variant]
                seed_score_deltas.append(float(current["official_score"]) - float(native["official_score"]))
                seed_nll_improvements.append(float(native["gold_answer_mean_nll"]) - float(current["gold_answer_mean_nll"]))
            score_deltas.extend(seed_score_deltas)
            nll_improvements.extend(seed_nll_improvements)
            per_seed.append({
                "seed": seed,
                "samples": len(seed_cases),
                "official_delta": mean(seed_score_deltas),
                "gold_nll_improvement": mean(seed_nll_improvements),
                "nll_improved": sum(value > 0 for value in seed_nll_improvements),
                "nll_degraded": sum(value < 0 for value in seed_nll_improvements),
            })
        mean_nll = mean(float(row["gold_answer_mean_nll"]) for row in rows)
        summaries.append({
            "variant": variant,
            "spec": rows[0]["spec"],
            "seeds": len(per_seed),
            "samples": len(rows),
            "official_score_mean": mean(float(row["official_score"]) for row in rows),
            "paired_official_delta": mean(score_deltas),
            "official_delta_ci95": percentile_interval(score_deltas, 20260806),
            "min_seed_official_delta": min(row["official_delta"] for row in per_seed),
            "official_improved": sum(value > 0 for value in score_deltas),
            "official_degraded": sum(value < 0 for value in score_deltas),
            "gold_answer_mean_nll": mean_nll,
            "gold_answer_ppl": math.exp(min(mean_nll, 30.0)),
            "mean_gold_nll_improvement": mean(nll_improvements),
            "gold_nll_improvement_ci95": percentile_interval(nll_improvements, 20260807),
            "nll_improved": sum(value > 0 for value in nll_improvements),
            "nll_degraded": sum(value < 0 for value in nll_improvements),
            "per_seed": per_seed,
        })
    summaries.sort(
        key=lambda row: (row["min_seed_official_delta"] >= 0, row["mean_gold_nll_improvement"]),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat = [{key: value for key, value in row.items() if key not in {"spec", "per_seed"}} for row in summaries]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
