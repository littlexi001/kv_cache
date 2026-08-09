from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


CANONICAL_COLUMNS = (
    "experiment_id",
    "model",
    "sample_id",
    "sample_group",
    "condition",
    "placement",
    "length",
    "length_unit",
    "evidence_scope",
    "top1_correct",
    "candidate_correct",
    "candidate_prediction",
    "candidate_margin",
    "gold_ppl",
    "gold_probability",
    "answer_margin",
    "strongest_wrong_token",
    "evidence_mass",
    "start_key_mass",
    "hop1_result_mass",
    "hop2_input_mass",
    "hop2_result_mass",
    "evidence_label_mass",
    "conflict_target_mass",
    "conflict_block_mass",
    "conflict_label_mass",
    "ordinary_background_mass",
    "other_token_mass",
    "outside_top20_mass",
    "attention_entropy",
    "effective_tokens",
    "evidence_qk_logit",
    "evidence_qk_cosine",
    "evidence_rank",
    "evidence_logsumexp",
    "non_evidence_logsumexp",
    "softmax_logsumexp",
    "evidence_log_odds",
    "source_file",
)

DERIVED_COLUMNS = (
    "evidence_mass_delta_from_short",
    "evidence_mass_ratio_from_short",
    "other_token_mass_delta_from_short",
    "outside_top20_mass_delta_from_short",
    "answer_margin_delta_from_short",
    "evidence_qk_logit_delta_from_short",
    "softmax_logsumexp_delta_from_short",
    "failure_transition",
    "recovery_transition",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    return float(value)


def optional_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    return int(float(value))


def blank_row() -> dict[str, Any]:
    return {column: None for column in CANONICAL_COLUMNS}


def load_clean_trace(path: Path) -> list[dict[str, Any]]:
    result = []
    for source in read_csv(path):
        row = blank_row()
        row.update(
            {
                "experiment_id": "clean_middle_english_single_token_128k",
                "model": "Qwen3-8B",
                "sample_id": "river_window_basket",
                "sample_group": "clean_two_hop",
                "condition": "clean",
                "placement": "middle",
                "length": optional_int(source, "length"),
                "length_unit": "filler_tokens",
                "evidence_scope": "four_atomic_positions",
                "top1_correct": optional_int(source, "top1_correct"),
                "candidate_correct": optional_int(
                    source,
                    "candidate_correct"
                    if "candidate_correct" in source
                    else "top1_correct",
                ),
                "candidate_prediction": source.get("candidate_prediction"),
                "candidate_margin": optional_float(source, "candidate_margin"),
                "gold_ppl": optional_float(source, "gold_ppl"),
                "gold_probability": optional_float(source, "gold_probability"),
                "answer_margin": optional_float(source, "signed_answer_margin"),
                "strongest_wrong_token": source.get("strongest_wrong_token"),
                "evidence_mass": optional_float(source, "atomic_evidence_mass"),
                "start_key_mass": optional_float(source, "start_key_mass"),
                "hop1_result_mass": optional_float(source, "hop1_result_mass"),
                "hop2_input_mass": optional_float(source, "hop2_input_mass"),
                "hop2_result_mass": optional_float(source, "hop2_result_mass"),
                "other_token_mass": optional_float(source, "other_token_mass"),
                "outside_top20_mass": optional_float(source, "outside_top20_mass"),
                "attention_entropy": optional_float(source, "attention_entropy"),
                "effective_tokens": optional_float(source, "effective_tokens"),
                "evidence_qk_logit": optional_float(source, "mean_hop2_result_logit"),
                "evidence_qk_cosine": optional_float(source, "mean_hop2_result_cosine"),
                "evidence_rank": optional_float(source, "mean_hop2_result_rank"),
                "softmax_logsumexp": optional_float(source, "mean_head_logsumexp"),
                "evidence_log_odds": optional_float(
                    source, "mean_hop2_result_log_odds_proxy"
                ),
                "source_file": str(path),
            }
        )
        result.append(row)
    return result


def load_refined_clean_trace(path: Path) -> list[dict[str, Any]]:
    rows = load_clean_trace(path)
    for row in rows:
        row["experiment_id"] = "clean_middle_boundary_refine_34_500"
    return rows


def load_candidate_dense_trace(path: Path) -> list[dict[str, Any]]:
    rows = load_clean_trace(path)
    for row in rows:
        row["experiment_id"] = "clean_middle_candidate_margin_dense_34_100"
    return rows


def load_semantic_conditions(path: Path) -> list[dict[str, Any]]:
    result = []
    for source in read_csv(path):
        evidence_mass = optional_float(source, "evidence_mass")
        row = blank_row()
        row.update(
            {
                "experiment_id": "semantic_catalog_plain_distractor_conflict",
                "model": "Qwen3-8B",
                "sample_id": source.get("concept_id"),
                "sample_group": source.get("bin"),
                "condition": source.get("condition"),
                "placement": source.get("placement"),
                "length": optional_int(source, "filler_length"),
                "length_unit": "filler_tokens",
                "evidence_scope": "verified_catalog_entry",
                "top1_correct": optional_int(source, "greedy_correct"),
                "candidate_correct": optional_int(source, "candidate_correct"),
                "gold_ppl": optional_float(source, "gold_ppl"),
                "answer_margin": optional_float(source, "candidate_margin"),
                "evidence_mass": evidence_mass,
                "evidence_label_mass": optional_float(source, "evidence_label_mass"),
                "conflict_target_mass": optional_float(source, "target_special_mass"),
                "conflict_block_mass": optional_float(source, "special_mass"),
                "conflict_label_mass": optional_float(source, "special_label_mass"),
                "ordinary_background_mass": optional_float(
                    source, "ordinary_background_mass"
                ),
                "other_token_mass": None
                if evidence_mass is None
                else 1.0 - evidence_mass,
                "outside_top20_mass": optional_float(source, "outside_top20_mass"),
                "evidence_logsumexp": optional_float(source, "evidence_logsumexp"),
                "non_evidence_logsumexp": optional_float(
                    source, "non_evidence_logsumexp"
                ),
                "evidence_log_odds": optional_float(
                    source, "evidence_vs_non_evidence"
                ),
                "source_file": str(path),
            }
        )
        result.append(row)
    return result


def load_fixed_relative(path: Path) -> list[dict[str, Any]]:
    result = []
    for source in read_csv(path):
        evidence_mass = optional_float(source, "mean_evidence_mass")
        row = blank_row()
        row.update(
            {
                "experiment_id": "clean_fixed_relative_distance_328_128k",
                "model": "Qwen3-8B",
                "sample_id": "river_window_basket",
                "sample_group": "clean_two_hop",
                "condition": "clean",
                "placement": "fixed_relative_328",
                "length": optional_int(source, "filler_tokens"),
                "length_unit": "filler_tokens",
                "evidence_scope": "hop2_result_only",
                "gold_ppl": optional_float(source, "gold_ppl"),
                "gold_probability": optional_float(source, "gold_probability"),
                "evidence_mass": evidence_mass,
                "hop2_result_mass": evidence_mass,
                "other_token_mass": None
                if evidence_mass is None
                else 1.0 - evidence_mass,
                "evidence_qk_logit": optional_float(
                    source, "mean_evidence_logit"
                ),
                "evidence_qk_cosine": optional_float(
                    source, "mean_evidence_cosine"
                ),
                "evidence_rank": optional_float(source, "mean_evidence_rank"),
                "softmax_logsumexp": optional_float(
                    source, "mean_head_logsumexp"
                ),
                "source_file": str(path),
            }
        )
        qk_logit = row["evidence_qk_logit"]
        denominator = row["softmax_logsumexp"]
        if qk_logit is not None and denominator is not None:
            row["evidence_log_odds"] = qk_logit - denominator
        result.append(row)
    return result


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def add_deltas(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_id"],
            row["sample_id"],
            row["condition"],
            row["placement"],
            row["evidence_scope"],
        )
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["length"]))
        baseline = group[0]
        previous = None
        for row in group:
            for field, output in (
                ("evidence_mass", "evidence_mass_delta_from_short"),
                ("other_token_mass", "other_token_mass_delta_from_short"),
                ("outside_top20_mass", "outside_top20_mass_delta_from_short"),
                ("answer_margin", "answer_margin_delta_from_short"),
                ("evidence_qk_logit", "evidence_qk_logit_delta_from_short"),
                ("softmax_logsumexp", "softmax_logsumexp_delta_from_short"),
            ):
                if finite(row[field]) and finite(baseline[field]):
                    row[output] = float(row[field]) - float(baseline[field])
                else:
                    row[output] = None
            if finite(row["evidence_mass"]) and finite(baseline["evidence_mass"]):
                denominator = float(baseline["evidence_mass"])
                row["evidence_mass_ratio_from_short"] = (
                    float(row["evidence_mass"]) / denominator
                    if denominator != 0.0
                    else None
                )
            else:
                row["evidence_mass_ratio_from_short"] = None
            row["failure_transition"] = 0
            row["recovery_transition"] = 0
            if (
                previous is not None
                and previous["top1_correct"] is not None
                and row["top1_correct"] is not None
            ):
                if int(previous["top1_correct"]) == 1 and int(row["top1_correct"]) == 0:
                    row["failure_transition"] = 1
                if int(previous["top1_correct"]) == 0 and int(row["top1_correct"]) == 1:
                    row["recovery_transition"] = 1
            previous = row


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def boundary_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_id"],
            row["sample_id"],
            row["sample_group"],
            row["condition"],
            row["placement"],
            row["evidence_scope"],
        )
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        group.sort(key=lambda row: int(row["length"]))
        failures = [
            row
            for row in group
            if row["top1_correct"] is not None and int(row["top1_correct"]) == 0
        ]
        margin_crossings = [
            row for row in group if finite(row["answer_margin"]) and float(row["answer_margin"]) <= 0.0
        ]
        first = group[0]
        last = group[-1]
        result.append(
            {
                "experiment_id": key[0],
                "sample_id": key[1],
                "sample_group": key[2],
                "condition": key[3],
                "placement": key[4],
                "evidence_scope": key[5],
                "minimum_length": first["length"],
                "maximum_length": last["length"],
                "points": len(group),
                "first_observed_failure": failures[0]["length"] if failures else None,
                "first_nonpositive_margin": margin_crossings[0]["length"]
                if margin_crossings
                else None,
                "endpoint_evidence_mass_ratio": last.get(
                    "evidence_mass_ratio_from_short"
                ),
                "endpoint_answer_margin": last["answer_margin"],
                "endpoint_gold_ppl": last["gold_ppl"],
                "failure_transitions": sum(
                    int(row.get("failure_transition", 0)) for row in group
                ),
                "recovery_transitions": sum(
                    int(row.get("recovery_transition", 0)) for row in group
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    project = Path(args.project_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[Path, Callable[[Path], list[dict[str, Any]]]]] = [
        (
            project
            / "artifacts/20260723_single_sample_failure_trace_from_128k"
            / "single_sample_trace.csv",
            load_clean_trace,
        ),
        (
            project
            / "artifacts/20260723_single_sample_margin_boundary_refine"
            / "single_sample_trace.csv",
            load_refined_clean_trace,
        ),
        (
            project
            / "artifacts/20260724_candidate_margin_dense_34_100"
            / "single_sample_trace.csv",
            load_candidate_dense_trace,
        ),
        (
            project
            / "artifacts/20260723_local_softmax_boundary_qwen3_8b"
            / "needle_trajectories.csv",
            load_semantic_conditions,
        ),
        (
            project
            / "artifacts/20260719_fixed_relative_328_128k"
            / "analysis/fixed_relative_rows.csv",
            load_fixed_relative,
        ),
    ]
    rows: list[dict[str, Any]] = []
    loaded_sources = []
    for path, loader in sources:
        if not path.exists():
            continue
        loaded = loader(path)
        rows.extend(loaded)
        loaded_sources.append({"path": str(path), "rows": len(loaded)})
    if not rows:
        raise FileNotFoundError("none of the configured registry sources exist")
    add_deltas(rows)
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            str(row["sample_id"]),
            str(row["condition"]),
            str(row["placement"]),
            int(row["length"]),
        )
    )
    write_csv(
        output / "attention_failure_registry.csv",
        rows,
        CANONICAL_COLUMNS + DERIVED_COLUMNS,
    )
    summaries = boundary_summary(rows)
    write_csv(output / "boundary_summary.csv", summaries, summaries[0].keys())
    (output / "registry_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": len(rows),
                "groups": len(summaries),
                "sources": loaded_sources,
                "important_note": (
                    "Compare evidence_mass only when evidence_scope matches; "
                    "different scopes have different numerators."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows across {len(summaries)} groups -> {output}")


if __name__ == "__main__":
    main()
