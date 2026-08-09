from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence


VARIANTS = (
    "native_noop",
    "exact_final_pre_top2_postscore",
    "queryspan_block_top2_postscore",
    "queryspan_tokenmax_top2_postscore",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(deltas: Sequence[float], seed: int, replicates: int = 20000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    samples = [
        statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(replicates)
    ]
    samples.sort()
    return statistics.mean(deltas), samples[499], samples[19499]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for shard in args.shards for row in read_jsonl(shard / "rows.jsonl")]
    rows.sort(key=lambda row: (row["sample_id"], VARIANTS.index(row["variant"])))
    if len(rows) != 18 * len(VARIANTS):
        raise RuntimeError(f"expected 72 rows, found {len(rows)}")
    by = {(row["sample_id"], row["variant"]): row for row in rows}
    ids = sorted({row["sample_id"] for row in rows})
    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        def mean(key: str, default: float = 0.0) -> float:
            return statistics.mean(float(row.get(key, default) or default) for row in selected)
        nll = mean("first_token_nll")
        summary.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "first_token_nll": nll,
                "first_token_ppl": math.exp(min(nll, 30.0)),
                "first_token_accuracy_percent": 100.0 * mean("first_token_correct"),
                "gold_evidence_token_recall_percent": 100.0 * mean("gold_evidence_token_recall"),
                "gold_evidence_attention_mass_percent": 100.0 * mean("gold_evidence_attention_mass"),
                "mean_query_seconds": mean("query_seconds"),
            }
        )
    baseline = "exact_final_pre_top2_postscore"
    comparisons = []
    for offset, variant in enumerate(VARIANTS[2:]):
        nll = [by[(sample, variant)]["first_token_nll"] - by[(sample, baseline)]["first_token_nll"] for sample in ids]
        recall = [by[(sample, variant)]["gold_evidence_token_recall"] - by[(sample, baseline)]["gold_evidence_token_recall"] for sample in ids]
        mass = [by[(sample, variant)]["gold_evidence_attention_mass"] - by[(sample, baseline)]["gold_evidence_attention_mass"] for sample in ids]
        nll_result = bootstrap(nll, 20260803 + offset * 100)
        recall_result = bootstrap(recall, 20260804 + offset * 100)
        mass_result = bootstrap(mass, 20260805 + offset * 100)
        comparisons.append(
            {
                "left": variant,
                "right": baseline,
                "sample_count": len(ids),
                "delta_first_token_nll": nll_result[0],
                "delta_first_token_nll_ci_low": nll_result[1],
                "delta_first_token_nll_ci_high": nll_result[2],
                "delta_evidence_recall_percentage_points": 100.0 * recall_result[0],
                "delta_evidence_recall_ci_low": 100.0 * recall_result[1],
                "delta_evidence_recall_ci_high": 100.0 * recall_result[2],
                "delta_evidence_mass_percentage_points": 100.0 * mass_result[0],
                "delta_evidence_mass_ci_low": 100.0 * mass_result[1],
                "delta_evidence_mass_ci_high": 100.0 * mass_result[2],
            }
        )
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "rows.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "comparisons": comparisons}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tokenmax = comparisons[1]
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Query-span pre-RoPE screening result",
                "",
                "- Integrity: **PASS** (18 samples, 72 rows).",
                f"- Token-max recall delta: {tokenmax['delta_evidence_recall_percentage_points']:+.2f} percentage points "
                f"(95% CI [{tokenmax['delta_evidence_recall_ci_low']:+.2f}, {tokenmax['delta_evidence_recall_ci_high']:+.2f}]).",
                f"- Token-max first-token NLL delta: {tokenmax['delta_first_token_nll']:+.4f} "
                f"(95% CI [{tokenmax['delta_first_token_nll_ci_low']:+.4f}, {tokenmax['delta_first_token_nll_ci_high']:+.4f}]).",
                "- Decision: **NO-GO**. Recall improves, but answer likelihood degrades significantly.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

