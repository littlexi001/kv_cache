from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import run_longbench_frequency_scaling as longbench_eval  # noqa: E402
import run_longbench_e_panel_frequency_scaling as longbench_e_eval  # noqa: E402
import run_pg19_frequency_ppl as pg19_eval  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty merge")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 20000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [draws[int(0.025 * repeats)], draws[int(0.975 * repeats) - 1]]


def paired_longbench(rows: Sequence[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_case = {(str(row["variant"]), str(row["sample_id"])): row for row in rows}
    sample_ids = sorted({str(row["sample_id"]) for row in rows if row["variant"] == baseline})
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        score_delta = []
        nll_improvement = []
        for sample_id in sample_ids:
            native = by_case[(baseline, sample_id)]
            current = by_case[(variant, sample_id)]
            score_delta.append(float(current["official_qa_f1"]) - float(native["official_qa_f1"]))
            nll_improvement.append(
                float(native["gold_answer_mean_nll"]) - float(current["gold_answer_mean_nll"])
            )
        output.append(
            {
                "variant": variant,
                "sample_count": len(sample_ids),
                "paired_qa_f1_delta_percent": 100.0 * mean(score_delta),
                "qa_f1_delta_ci95_percent": [100.0 * value for value in bootstrap_ci(score_delta, 20260808)],
                "paired_gold_nll_improvement": mean(nll_improvement),
                "gold_nll_improvement_ci95": bootstrap_ci(nll_improvement, 20260809),
                "nll_improved": sum(value > 0 for value in nll_improvement),
                "nll_degraded": sum(value < 0 for value in nll_improvement),
            }
        )
    return output


def paired_pg19(rows: Sequence[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_case = {(str(row["variant"]), str(row["case_id"])): row for row in rows}
    case_ids = sorted({str(row["case_id"]) for row in rows if row["variant"] == baseline})
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for length in sorted({int(row["context_length"]) for row in rows}):
            selected = [
                case_id for case_id in case_ids
                if int(by_case[(baseline, case_id)]["context_length"]) == length
            ]
            improvements = []
            relative_ppl = []
            for case_id in selected:
                native = by_case[(baseline, case_id)]
                current = by_case[(variant, case_id)]
                improvements.append(float(native["mean_nll"]) - float(current["mean_nll"]))
                relative_ppl.append(
                    100.0 * (float(current["perplexity"]) / float(native["perplexity"]) - 1.0)
                )
            output.append(
                {
                    "variant": variant,
                    "context_length": length,
                    "case_count": len(selected),
                    "paired_nll_improvement": mean(improvements),
                    "nll_improvement_ci95": bootstrap_ci(improvements, 20260810 + length),
                    "mean_relative_ppl_change_percent": mean(relative_ppl),
                    "relative_ppl_change_ci95_percent": bootstrap_ci(relative_ppl, 20260811 + length),
                }
            )
    return output


def paired_longbench_e(rows: Sequence[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_case = {
        (str(row["variant"]), str(row["dataset"]), str(row["sample_id"])): row
        for row in rows
    }
    cases = sorted(
        {
            (str(row["dataset"]), str(row["sample_id"]))
            for row in rows if row["variant"] == baseline
        }
    )
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        score_delta = []
        nll_improvement = []
        for dataset, sample_id in cases:
            native = by_case[(baseline, dataset, sample_id)]
            current = by_case[(variant, dataset, sample_id)]
            score_delta.append(float(current["official_score"]) - float(native["official_score"]))
            nll_improvement.append(
                float(native["gold_answer_mean_nll"]) - float(current["gold_answer_mean_nll"])
            )
        output.append(
            {
                "variant": variant,
                "sample_count": len(cases),
                "paired_score_delta_percent": 100.0 * mean(score_delta),
                "score_delta_ci95_percent": [100.0 * value for value in bootstrap_ci(score_delta, 20260812)],
                "paired_gold_nll_improvement": mean(nll_improvement),
                "gold_nll_improvement_ci95": bootstrap_ci(nll_improvement, 20260813),
                "official_improved": sum(value > 0 for value in score_delta),
                "official_degraded": sum(value < 0 for value in score_delta),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("longbench", "longbench_e", "pg19"))
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("shard*/rows.jsonl"))
    if not paths:
        raise RuntimeError(f"no shard rows under {args.run_dir}")
    rows = [row for path in paths for row in read_jsonl(path)]
    if args.mode == "longbench":
        key: Callable[[dict[str, Any]], tuple[Any, ...]] = lambda row: (
            row["variant"], row["sample_id"]
        )
        summary = longbench_eval.summarize(rows)
        paired = paired_longbench(rows, "native_rope")
    elif args.mode == "longbench_e":
        key = lambda row: (row["variant"], row["dataset"], row["sample_id"])
        summary = longbench_e_eval.summarize(rows)
        paired = paired_longbench_e(rows, "native_rope")
    else:
        key = lambda row: (row["variant"], row["case_id"])
        summary = pg19_eval.summarize(rows)
        paired = paired_pg19(rows, "native_rope")
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate benchmark cases across shards")
    rows.sort(key=key)
    write_csv(args.run_dir / "merged_rows.csv", rows)
    with (args.run_dir / "merged_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(
        args.run_dir / "summary.csv",
        [
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in summary
        ],
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        args.run_dir / "paired_summary.csv",
        [
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in paired
        ],
    )
    (args.run_dir / "paired_summary.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "merge.done").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
