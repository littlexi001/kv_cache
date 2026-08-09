from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rows_path in path.rglob("rows.jsonl"):
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True, help="SEED=RUN_DIR")
    parser.add_argument("--baseline", default="native_rope")
    args = parser.parse_args()

    by_case: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    variants: set[str] = set()
    for value in args.seed_run:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        for row in read_rows(Path(path_text)):
            variant = str(row["variant"])
            variants.add(variant)
            by_case[(seed, str(row["sample_id"]))][variant] = row

    summaries = []
    for variant in sorted(variants):
        if variant == args.baseline:
            continue
        paired = [
            values
            for values in by_case.values()
            if args.baseline in values and variant in values
        ]
        if not paired:
            continue
        score_deltas = [
            float(values[variant]["official_score"])
            - float(values[args.baseline]["official_score"])
            for values in paired
        ]
        nll_improvements = [
            float(values[args.baseline]["gold_answer_mean_nll"])
            - float(values[variant]["gold_answer_mean_nll"])
            for values in paired
        ]
        summaries.append(
            {
                "variant": variant,
                "paired_samples": len(paired),
                "official_delta": mean(score_deltas),
                "official_improved": sum(value > 0 for value in score_deltas),
                "official_degraded": sum(value < 0 for value in score_deltas),
                "mean_gold_nll_improvement": mean(nll_improvements),
                "nll_improved": sum(value > 0 for value in nll_improvements),
                "nll_degraded": sum(value < 0 for value in nll_improvements),
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
