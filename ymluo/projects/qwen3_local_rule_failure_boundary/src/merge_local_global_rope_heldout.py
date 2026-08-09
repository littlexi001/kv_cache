from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


LENGTHS = (8192, 16384, 32768, 65536)
SEEDS = tuple(range(8, 32))
VARIANTS = (
    "full_rope",
    "rope_top2",
    "semantic_top2_postscore",
    "local_global_raw",
    "local_global_calibrated",
    "local_global_blend25",
    "local_global_blend50",
    "dual_max_blend25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["target_context_tokens"]),
        int(row["seed"]),
        str(row["variant"]),
    )


def comparable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in row.items()
        if name not in {"prefill_seconds", "query_seconds"}
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for length in LENGTHS:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if int(row["target_context_tokens"]) == length
                and str(row["variant"]) == variant
            ]
            mean_nll = statistics.fmean(
                float(row["gold_nll"]) for row in selected
            )
            output.append(
                {
                    "target_context_tokens": length,
                    "variant": variant,
                    "sample_count": len(selected),
                    "gold_evidence_token_recall": statistics.fmean(
                        float(row["gold_evidence_token_recall"])
                        for row in selected
                    ),
                    "gold_evidence_line_hit_rate": statistics.fmean(
                        float(row["gold_evidence_line_hit_rate"])
                        for row in selected
                    ),
                    "gold_chain_complete_rate": statistics.fmean(
                        float(row["gold_chain_complete_rate"])
                        for row in selected
                    ),
                    "gold_evidence_attention_mass": statistics.fmean(
                        float(row["gold_evidence_attention_mass"])
                        for row in selected
                    ),
                    "mean_gold_nll": mean_nll,
                    "gold_ppl": math.exp(mean_nll),
                    "next_token_accuracy": statistics.fmean(
                        int(row["next_token_correct"]) for row in selected
                    ),
                    "mean_query_seconds": statistics.fmean(
                        float(row["query_seconds"]) for row in selected
                    ),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    duplicate_count = 0
    for source_text in args.input_jsonl:
        source = Path(source_text)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_key = key(row)
            if row_key in merged:
                duplicate_count += 1
                if comparable(merged[row_key]) != comparable(row):
                    raise RuntimeError(
                        f"non-timing duplicate mismatch for {row_key}: {source}"
                    )
                continue
            merged[row_key] = row

    expected = {
        (length, seed, variant)
        for length in LENGTHS
        for seed in SEEDS
        for variant in VARIANTS
    }
    actual = set(merged)
    if actual != expected:
        raise RuntimeError(
            "coverage mismatch: "
            f"missing={sorted(expected - actual)[:20]} "
            f"unexpected={sorted(actual - expected)[:20]}"
        )

    rows = [merged[row_key] for row_key in sorted(merged)]
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "rows.csv", rows)
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "source_files": [str(Path(item)) for item in args.input_jsonl],
        "duplicate_count": duplicate_count,
        "row_count": len(rows),
        "lengths": list(LENGTHS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "expected_row_count": len(expected),
    }
    write_json(output_dir / "merge_manifest.json", manifest)
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
