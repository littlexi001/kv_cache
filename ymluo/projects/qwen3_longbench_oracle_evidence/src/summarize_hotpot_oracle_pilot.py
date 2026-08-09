#!/usr/bin/env python3
"""Merge HotpotQA oracle-pilot shards and write paired summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260802)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    deltas: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(rng.choice(deltas) for _ in deltas))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def aggregate_random(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"].startswith("random_document_seed"):
            grouped[row["sample_id"]].append(row)
    output = []
    numeric = (
        "official_qa_f1",
        "normalized_exact_match",
        "prediction_contains_answer",
        "selected_context_contains_answer",
        "context_tokens",
        "prompt_tokens",
        "compression_ratio",
        "gold_answer_mean_nll",
        "gold_answer_ppl",
        "generation_seconds",
        "nll_seconds",
    )
    for sample_id, values in grouped.items():
        row = {"sample_id": sample_id, "condition": "random_document_mean"}
        for key in numeric:
            finite = [float(item[key]) for item in values if math.isfinite(float(item[key]))]
            row[key] = mean(finite) if finite else float("nan")
        row["replicates"] = len(values)
        output.append(row)
    return output


def condition_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    output = []
    for condition, values in sorted(grouped.items()):
        nll_values = [
            float(row["gold_answer_mean_nll"])
            for row in values
            if math.isfinite(float(row["gold_answer_mean_nll"]))
        ]
        output.append(
            {
                "condition": condition,
                "samples": len(values),
                "qa_f1_percent": 100.0 * mean(float(row["official_qa_f1"]) for row in values),
                "exact_match_percent": 100.0
                * mean(float(row["normalized_exact_match"]) for row in values),
                "prediction_contains_answer_percent": 100.0
                * mean(float(row["prediction_contains_answer"]) for row in values),
                "selected_context_contains_answer_percent": 100.0
                * mean(float(row["selected_context_contains_answer"]) for row in values),
                "mean_gold_answer_nll": mean(nll_values) if nll_values else float("nan"),
                "mean_gold_answer_ppl": math.exp(mean(nll_values)) if nll_values else float("nan"),
                "mean_context_tokens": mean(float(row["context_tokens"]) for row in values),
                "mean_prompt_tokens": mean(float(row["prompt_tokens"]) for row in values),
                "mean_compression_ratio_percent": 100.0
                * mean(float(row["compression_ratio"]) for row in values),
                "mean_generation_seconds": mean(float(row["generation_seconds"]) for row in values),
            }
        )
    return output


def paired_summaries(
    rows: Sequence[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["condition"]] = row
    output = []
    targets = [
        "oracle_sentence",
        "oracle_document",
        "bm25_document",
        "random_document_mean",
        "query_only",
    ]
    for target in targets:
        pairs = [
            (conditions["full"], conditions[target])
            for conditions in by_sample.values()
            if "full" in conditions and target in conditions
        ]
        f1_deltas = [
            float(target_row["official_qa_f1"]) - float(full_row["official_qa_f1"])
            for full_row, target_row in pairs
        ]
        low, high = paired_bootstrap_ci(
            f1_deltas, bootstrap_samples, bootstrap_seed + len(output)
        )
        full_wrong = [pair for pair in pairs if not bool(pair[0]["normalized_exact_match"])]
        full_correct = [pair for pair in pairs if bool(pair[0]["normalized_exact_match"])]
        rescue = (
            mean(float(target_row["normalized_exact_match"]) for _, target_row in full_wrong)
            if full_wrong
            else float("nan")
        )
        harm = (
            mean(1.0 - float(target_row["normalized_exact_match"]) for _, target_row in full_correct)
            if full_correct
            else float("nan")
        )
        nll_deltas = [
            float(target_row["gold_answer_mean_nll"])
            - float(full_row["gold_answer_mean_nll"])
            for full_row, target_row in pairs
            if math.isfinite(float(target_row["gold_answer_mean_nll"]))
            and math.isfinite(float(full_row["gold_answer_mean_nll"]))
        ]
        output.append(
            {
                "comparison": f"{target}_minus_full",
                "paired_samples": len(pairs),
                "mean_f1_delta_points": 100.0 * mean(f1_deltas) if f1_deltas else float("nan"),
                "f1_delta_ci95_low_points": 100.0 * low,
                "f1_delta_ci95_high_points": 100.0 * high,
                "mean_gold_nll_delta": mean(nll_deltas) if nll_deltas else float("nan"),
                "rescue_rate_percent": 100.0 * rescue if math.isfinite(rescue) else float("nan"),
                "harm_rate_percent": 100.0 * harm if math.isfinite(harm) else float("nan"),
                "full_wrong_count": len(full_wrong),
                "full_correct_count": len(full_correct),
            }
        )
    # The direct content-vs-shortening comparison is as important as vs Full.
    oracle_random_pairs = [
        (conditions["random_document_mean"], conditions["oracle_document"])
        for conditions in by_sample.values()
        if "random_document_mean" in conditions and "oracle_document" in conditions
    ]
    deltas = [
        float(oracle["official_qa_f1"]) - float(random_row["official_qa_f1"])
        for random_row, oracle in oracle_random_pairs
    ]
    low, high = paired_bootstrap_ci(deltas, bootstrap_samples, bootstrap_seed + 97)
    output.append(
        {
            "comparison": "oracle_document_minus_random_document_mean",
            "paired_samples": len(oracle_random_pairs),
            "mean_f1_delta_points": 100.0 * mean(deltas) if deltas else float("nan"),
            "f1_delta_ci95_low_points": 100.0 * low,
            "f1_delta_ci95_high_points": 100.0 * high,
            "mean_gold_nll_delta": mean(
                float(oracle["gold_answer_mean_nll"])
                - float(random_row["gold_answer_mean_nll"])
                for random_row, oracle in oracle_random_pairs
            )
            if oracle_random_pairs
            else float("nan"),
            "rescue_rate_percent": float("nan"),
            "harm_rate_percent": float("nan"),
            "full_wrong_count": float("nan"),
            "full_correct_count": float("nan"),
        }
    )
    return output


def markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    labels = {
        "condition": "Condition",
        "samples": "n",
        "qa_f1_percent": "QA-F1",
        "exact_match_percent": "EM",
        "mean_gold_answer_ppl": "Gold PPL",
        "mean_context_tokens": "Context tokens",
        "mean_compression_ratio_percent": "Prompt/full",
    }
    header = "| " + " | ".join(labels.get(field, field) for field in fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, separator]
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                if field in {"qa_f1_percent", "exact_match_percent", "mean_compression_ratio_percent"}:
                    cells.append(f"{value:.2f}%")
                elif field == "mean_context_tokens":
                    cells.append(f"{value:.0f}")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_result_doc(
    path: Path,
    summaries: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    manifest: Sequence[dict[str, Any]],
    run_args: dict[str, Any],
) -> None:
    summary_by_name = {row["condition"]: row for row in summaries}
    pair_by_name = {row["comparison"]: row for row in paired}
    oracle_gain = pair_by_name.get("oracle_document_minus_full", {})
    random_gap = pair_by_name.get("oracle_document_minus_random_document_mean", {})
    full = summary_by_name.get("full", {})
    oracle = summary_by_name.get("oracle_document", {})
    alignment_mode = str(run_args.get("alignment_mode", "unknown"))
    lines = [
        "# Results: LongBench HotpotQA evidence-selection diagnostic",
        "",
        f"Status: **frozen {len(manifest)}-sample run; descriptive evidence only**.",
        "",
        f"Evidence alignment mode: `{alignment_mode}`.",
        "",
        "## Result first",
        "",
    ]
    if full and oracle and oracle_gain and random_gap:
        lines.extend(
            [
                (
                    f"On the frozen {len(manifest)}-sample pilot, Full context reached "
                    f"{full['qa_f1_percent']:.2f} QA-F1 / {full['exact_match_percent']:.2f}% EM, "
                    f"while the complete gold-support documents reached "
                    f"{oracle['qa_f1_percent']:.2f} QA-F1 / {oracle['exact_match_percent']:.2f}% EM."
                ),
                "",
                (
                    f"The paired Oracle-document gain over Full was "
                    f"{oracle_gain['mean_f1_delta_points']:+.2f} F1 points "
                    f"(descriptive paired-bootstrap 95% CI "
                    f"[{oracle_gain['f1_delta_ci95_low_points']:+.2f}, "
                    f"{oracle_gain['f1_delta_ci95_high_points']:+.2f}]). "
                    f"Against equal-budget random documents, the gap was "
                    f"{random_gap['mean_f1_delta_points']:+.2f} F1 points."
                ),
                "",
            ]
        )
    fields = (
        "condition",
        "samples",
        "qa_f1_percent",
        "exact_match_percent",
        "mean_gold_answer_ppl",
        "mean_context_tokens",
        "mean_compression_ratio_percent",
    )
    lines.extend([markdown_table(summaries, fields), "", "## Paired comparisons", ""])
    lines.append(
        "| Comparison | n | Mean F1 delta | 95% CI | Rescue | Harm | Mean gold-NLL delta |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in paired:
        rescue = (
            f"{row['rescue_rate_percent']:.1f}%"
            if math.isfinite(float(row["rescue_rate_percent"]))
            else "N/A"
        )
        harm = (
            f"{row['harm_rate_percent']:.1f}%"
            if math.isfinite(float(row["harm_rate_percent"]))
            else "N/A"
        )
        lines.append(
            f"| `{row['comparison']}` | {row['paired_samples']} | "
            f"{row['mean_f1_delta_points']:+.2f} | "
            f"[{row['f1_delta_ci95_low_points']:+.2f}, {row['f1_delta_ci95_high_points']:+.2f}] | "
            f"{rescue} | {harm} | {row['mean_gold_nll_delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen sample audit",
            "",
            "| LongBench ID | HotpotQA ID | Prompt tokens | Evidence position | Type | Level |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in manifest:
        lines.append(
            f"| `{row['sample_id']}` | `{row['original_hotpot_id']}` | "
            f"{row['full_prompt_tokens']} | {row['evidence_position_fraction']:.3f} "
            f"({row['evidence_position_bin']}) | {row['hotpot_type']} | {row['hotpot_level']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"This is an evidence **compression and re-encoding** diagnostic. It does not preserve the original KV positions, and it jointly reduces distance and softmax competition. The {len(manifest)} frozen examples are a strictly aligned subset rather than a LongBench-wide estimate.",
            "",
            "Artifacts: `sample_manifest.jsonl`, `evidence_mapping.jsonl`, `predictions.jsonl`, `condition_summary.csv`, `paired_summary.csv`, and `summary.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = []
    prediction_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    manifests = []
    metas = []
    for shard_dir in args.shard_dirs:
        meta = json.loads((shard_dir / "run_meta.json").read_text(encoding="utf-8"))
        metas.append(meta)
        manifest_hashes.append(meta["manifest_sha256"])
        manifests.append(read_jsonl(shard_dir / "sample_manifest.jsonl"))
        prediction_rows.extend(read_jsonl(shard_dir / "predictions.jsonl"))
        evidence_rows.extend(read_jsonl(shard_dir / "evidence_mapping.jsonl"))
    if len(set(manifest_hashes)) != 1:
        raise RuntimeError(f"shard manifests disagree: {manifest_hashes}")
    manifest = manifests[0]
    run_args = dict(metas[0].get("args", {}))
    expected_ids = {row["sample_id"] for row in manifest}
    observed_ids = {row["sample_id"] for row in prediction_rows}
    if expected_ids != observed_ids:
        raise RuntimeError(
            f"prediction sample mismatch: missing={expected_ids-observed_ids}, extra={observed_ids-expected_ids}"
        )
    keys = [(row["sample_id"], row["condition"]) for row in prediction_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate sample-condition predictions across shards")
    evidence_ids = {row["sample_id"] for row in evidence_rows}
    if evidence_ids != expected_ids:
        raise RuntimeError(
            f"evidence sample mismatch: missing={expected_ids-evidence_ids}, extra={evidence_ids-expected_ids}"
        )
    random_budget_errors = [
        float(choice["relative_budget_error"])
        for row in evidence_rows
        for choice in row.get("random_choices", {}).values()
        if "relative_budget_error" in choice
    ]
    if random_budget_errors and max(random_budget_errors) > 0.05 + 1.0e-12:
        raise RuntimeError(
            f"random control token-budget error exceeds 5%: {max(random_budget_errors):.6f}"
        )
    expected_conditions = {
        "full",
        "oracle_sentence",
        "oracle_document",
        "bm25_document",
        "query_only",
        "random_document_seed0",
        "random_document_seed1",
        "random_document_seed2",
    }
    for sample_id in expected_ids:
        conditions = {row["condition"] for row in prediction_rows if row["sample_id"] == sample_id}
        if conditions != expected_conditions:
            raise RuntimeError(f"condition mismatch for {sample_id}: {conditions}")
    random_rows = aggregate_random(prediction_rows)
    analysis_rows = prediction_rows + random_rows
    summaries = condition_summary(analysis_rows)
    paired = paired_summaries(
        analysis_rows, args.bootstrap_samples, args.bootstrap_seed
    )
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "evidence_mapping.jsonl", evidence_rows)
    write_jsonl(args.output_dir / "sample_manifest.jsonl", manifest)
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "paired_summary.csv", paired)
    payload = {
        "manifest_sha256": manifest_hashes[0],
        "sample_count": len(manifest),
        "prediction_rows": len(prediction_rows),
        "alignment_mode": run_args.get("alignment_mode", "unknown"),
        "max_random_budget_error": max(random_budget_errors)
        if random_budget_errors
        else float("nan"),
        "conditions": summaries,
        "paired": paired,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_result_doc(
        args.output_dir / "visualization_results.md",
        summaries,
        paired,
        manifest,
        run_args,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
