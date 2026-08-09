#!/usr/bin/env python3
"""Add source-provenance and random-answer sensitivity audits to a frozen run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from run_hotpot_oracle_pilot import normalize_title, parse_longbench_passages
from summarize_hotpot_oracle_pilot import paired_bootstrap_ci


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sample_condition_metrics(
    predictions: Sequence[dict[str, Any]], sample_ids: set[str]
) -> dict[str, dict[str, float]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in predictions:
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            by_sample.setdefault(sample_id, {})[str(row["condition"])] = row
    output: dict[str, dict[str, float]] = {}
    for condition in ("full", "oracle_document"):
        rows = [conditions[condition] for conditions in by_sample.values()]
        output[condition] = {
            "samples": float(len(rows)),
            "qa_f1_percent": 100.0 * mean(float(row["official_qa_f1"]) for row in rows),
            "exact_match_percent": 100.0
            * mean(float(row["normalized_exact_match"]) for row in rows),
            "mean_gold_nll": mean(float(row["gold_answer_mean_nll"]) for row in rows),
        }
    random_by_sample: dict[str, float] = {}
    for sample_id, conditions in by_sample.items():
        random_rows = [
            row
            for name, row in conditions.items()
            if name.startswith("random_document_seed")
        ]
        random_by_sample[sample_id] = mean(
            float(row["official_qa_f1"]) for row in random_rows
        )
    output["random_document_mean"] = {
        "samples": float(len(random_by_sample)),
        "qa_f1_percent": 100.0 * mean(random_by_sample.values()),
    }
    full_deltas = [
        float(conditions["oracle_document"]["official_qa_f1"])
        - float(conditions["full"]["official_qa_f1"])
        for conditions in by_sample.values()
    ]
    random_deltas = [
        float(conditions["oracle_document"]["official_qa_f1"])
        - random_by_sample[sample_id]
        for sample_id, conditions in by_sample.items()
    ]
    for name, deltas, seed in (
        # Match summarize_hotpot_oracle_pilot.py: oracle_document is the
        # second target and therefore uses bootstrap_seed + 1.
        ("oracle_document_minus_full", full_deltas, 20260803),
        ("oracle_document_minus_random", random_deltas, 20260899),
    ):
        low, high = paired_bootstrap_ci(deltas, 100000, seed)
        output[name] = {
            "samples": float(len(deltas)),
            "mean_f1_delta_points": 100.0 * mean(deltas),
            "ci95_low_points": 100.0 * low,
            "ci95_high_points": 100.0 * high,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench_jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence_mapping", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(args.manifest)
    evidence_rows = read_jsonl(args.evidence_mapping)
    predictions = read_jsonl(args.predictions)
    longbench_rows = read_jsonl(args.longbench_jsonl)
    longbench_by_id = {
        str(row.get("_id", index)): row for index, row in enumerate(longbench_rows)
    }
    expected_ids = {str(row["sample_id"]) for row in manifest}
    if {str(row["sample_id"]) for row in evidence_rows} != expected_ids:
        raise RuntimeError("evidence mapping does not cover the frozen manifest")

    provenance_rows: list[dict[str, Any]] = []
    for evidence in evidence_rows:
        sample_id = str(evidence["sample_id"])
        source_row = longbench_by_id[sample_id]
        documents_by_title: dict[str, list[Any]] = {}
        for document in parse_longbench_passages(str(source_row["context"])):
            documents_by_title.setdefault(normalize_title(document.title), []).append(document)
        for alignment in evidence["support_alignment_records"]:
            matches = documents_by_title.get(normalize_title(alignment["title"]), [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"support title is not unique in source row: {sample_id} {alignment['title']}"
                )
            document = matches[0]
            body = " ".join(document.sentences)
            start = int(alignment["matched_start"])
            end = int(alignment["matched_end"])
            matched_text = str(alignment["matched_text"])
            verified = body[start:end] == matched_text
            if not verified:
                raise RuntimeError(
                    f"matched span does not reproduce source passage: {sample_id} {alignment['title']}"
                )
            provenance_rows.append(
                {
                    "sample_id": sample_id,
                    "title": document.title,
                    "source_sentence_id": alignment["source_sentence_id"],
                    "match_type": alignment["match_type"],
                    "passage_sha256": sha256_bytes(body.encode("utf-8")),
                    "matched_span_sha256": sha256_bytes(matched_text.encode("utf-8")),
                    "matched_start": start,
                    "matched_end": end,
                    "span_verified_against_source": verified,
                }
            )
    write_jsonl(args.output_dir / "passage_provenance.jsonl", provenance_rows)

    contaminated_ids = {
        str(row["sample_id"])
        for row in evidence_rows
        if any(
            bool(choice.get("contains_answer"))
            for choice in row.get("random_choices", {}).values()
        )
    }
    clean_ids = expected_ids - contaminated_ids
    sensitivity = {
        "all_samples": sample_condition_metrics(predictions, expected_ids),
        "random_answer_free_samples": sample_condition_metrics(predictions, clean_ids),
        "random_answer_contaminated_sample_ids": sorted(contaminated_ids),
    }
    (args.output_dir / "random_answer_sensitivity.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "longbench_jsonl_sha256": sha256_bytes(args.longbench_jsonl.read_bytes()),
        "manifest_sha256": sha256_bytes(args.manifest.read_bytes()),
        "sample_count": len(expected_ids),
        "support_span_count": len(provenance_rows),
        "verified_support_span_count": sum(
            int(row["span_verified_against_source"]) for row in provenance_rows
        ),
        "random_answer_contaminated_sample_count": len(contaminated_ids),
        "sensitivity": sensitivity,
    }
    (args.output_dir / "provenance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
