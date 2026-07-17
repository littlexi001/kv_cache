from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import run_length_causal_mechanism_20260717 as causal


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def sem_binary(value: float, n: int) -> float:
    return math.sqrt(value * (1.0 - value) / max(1, n))


def exact_paired_binary_p(delta_positive: int, delta_negative: int) -> float:
    discordant = delta_positive + delta_negative
    if discordant == 0:
        return 1.0
    tail = min(delta_positive, delta_negative)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def index_rows(
    rows: Iterable[dict[str, Any]], query_mode: str
) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    output: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["query_mode"] != query_mode:
            continue
        key = (
            int(row["seed"]),
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        if key in output:
            raise ValueError(f"duplicate result key: {key}, query={query_mode}")
        output[key] = row
    return output


def paired_effect_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full = index_rows(rows, "full2")
    lengths = sorted({key[1] for key in full})
    placements = sorted({key[2] for key in full})
    conditions = sorted({key[3] for key in full if key[3] != "clean"})
    output: list[dict[str, Any]] = []
    for length in lengths:
        for placement in placements:
            for condition in conditions:
                pairs = []
                seeds = sorted(
                    key[0]
                    for key in full
                    if key[1:] == (length, placement, condition)
                    and (key[0], length, placement, "clean") in full
                )
                for seed in seeds:
                    clean = full[(seed, length, placement, "clean")]
                    treatment = full[(seed, length, placement, condition)]
                    pairs.append((clean, treatment))
                if not pairs:
                    continue
                gains = sum(
                    int(treatment["generation_final_correct"])
                    > int(clean["generation_final_correct"])
                    for clean, treatment in pairs
                )
                losses = sum(
                    int(treatment["generation_final_correct"])
                    < int(clean["generation_final_correct"])
                    for clean, treatment in pairs
                )
                output.append(
                    {
                        "target_context_tokens": length,
                        "placement": placement,
                        "condition_vs_clean": condition,
                        "pair_count": len(pairs),
                        "delta_generation_final_accuracy": statistics.mean(
                            int(treatment["generation_final_correct"])
                            - int(clean["generation_final_correct"])
                            for clean, treatment in pairs
                        ),
                        "generation_gains": gains,
                        "generation_losses": losses,
                        "generation_mcnemar_exact_p": exact_paired_binary_p(gains, losses),
                        "delta_start_excluded_candidate_accuracy": statistics.mean(
                            int(treatment["start_excluded_candidate_correct"])
                            - int(clean["start_excluded_candidate_correct"])
                            for clean, treatment in pairs
                        ),
                        "delta_gold_mean_nll": statistics.mean(
                            float(treatment["gold_candidate_mean_nll"])
                            - float(clean["gold_candidate_mean_nll"])
                            for clean, treatment in pairs
                        ),
                        "delta_candidate_margin": statistics.mean(
                            float(treatment["candidate_margin"])
                            - float(clean["candidate_margin"])
                            for clean, treatment in pairs
                        ),
                    }
                )
    return output


def core_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["query_mode"] == "full2":
            buckets[
                (
                    int(row["target_context_tokens"]),
                    str(row["placement"]),
                    str(row["condition"]),
                )
            ].append(row)
    output: list[dict[str, Any]] = []
    for (length, placement, condition), selected in sorted(buckets.items()):
        generation_accuracy = mean(selected, "generation_final_correct")
        strict_candidate = mean(selected, "candidate_correct")
        conditioned_candidate = mean(selected, "start_excluded_candidate_correct")
        roles = Counter(str(row["candidate_prediction_role"]) for row in selected)
        output.append(
            {
                "target_context_tokens": length,
                "placement": placement,
                "condition": condition,
                "sample_count": len(selected),
                "generation_final_accuracy": generation_accuracy,
                "generation_final_sem": sem_binary(generation_accuracy, len(selected)),
                "generation_contains_gold_rate": mean(selected, "generation_contains_gold"),
                "strict_candidate_accuracy": strict_candidate,
                "start_excluded_candidate_accuracy": conditioned_candidate,
                "mean_gold_nll": mean(selected, "gold_candidate_mean_nll"),
                "mean_gold_ppl": statistics.mean(
                    float(row["gold_candidate_ppl"]) for row in selected
                ),
                "mean_candidate_margin": mean(selected, "candidate_margin"),
                "mean_candidate_entropy": mean(selected, "candidate_entropy"),
                "prediction_roles": json.dumps(roles, ensure_ascii=False, sort_keys=True),
            }
        )
    return output


def probe_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["query_mode"] in {"hop1", "oracle_hop2"}:
            buckets[
                (
                    int(row["target_context_tokens"]),
                    str(row["placement"]),
                    str(row["condition"]),
                    str(row["query_mode"]),
                )
            ].append(row)
    output = []
    for (length, placement, condition, query_mode), selected in sorted(buckets.items()):
        output.append(
            {
                "target_context_tokens": length,
                "placement": placement,
                "condition": condition,
                "query_mode": query_mode,
                "sample_count": len(selected),
                "strict_accuracy": mean(selected, "candidate_correct"),
                "input_excluded_accuracy": mean(
                    selected, "start_excluded_candidate_correct"
                ),
                "mean_gold_nll": mean(selected, "gold_candidate_mean_nll"),
                "mean_margin": mean(selected, "candidate_margin"),
                "prediction_roles": json.dumps(
                    Counter(str(row["candidate_prediction_role"]) for row in selected),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return output


def final_only_summary(scores: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        if row["query_mode"] != "full2":
            continue
        if row["candidate_role"] in {
            "gold_start",
            "gold_intermediate",
            "conflict_intermediate",
        }:
            continue
        key = (
            int(row["seed"]),
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        cases[key].append(row)
    buckets: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (seed, length, placement, condition), selected in cases.items():
        best = min(selected, key=lambda row: int(row["rank"]))
        buckets[(length, placement, condition)].append(
            {
                "seed": seed,
                "correct": int(best["candidate_role"] == "gold_final"),
                "prediction_role": str(best["candidate_role"]),
            }
        )
    output = []
    for (length, placement, condition), selected in sorted(buckets.items()):
        output.append(
            {
                "target_context_tokens": length,
                "placement": placement,
                "condition": condition,
                "sample_count": len(selected),
                "final_only_candidate_accuracy": statistics.mean(
                    row["correct"] for row in selected
                ),
                "prediction_roles": json.dumps(
                    Counter(str(row["prediction_role"]) for row in selected),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return output


def latent_mechanism_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[tuple[int, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        by_case[key][str(row["query_mode"])] = row
    detailed = []
    for (seed, length, placement, condition), modes in sorted(by_case.items()):
        if not {"full2", "hop1", "oracle_hop2"}.issubset(modes):
            continue
        full = int(modes["full2"]["start_excluded_candidate_correct"])
        hop1 = int(modes["hop1"]["start_excluded_candidate_correct"])
        oracle = int(modes["oracle_hop2"]["start_excluded_candidate_correct"])
        if full:
            label = "latent_success"
        elif not hop1:
            label = "first_hop_access_or_binding_failure"
        elif not oracle:
            label = "second_rule_access_failure"
        else:
            label = "composition_or_state_update_failure"
        detailed.append(
            {
                "seed": seed,
                "target_context_tokens": length,
                "placement": placement,
                "condition": condition,
                "full2_start_excluded_correct": full,
                "hop1_input_excluded_correct": hop1,
                "oracle_hop2_input_excluded_correct": oracle,
                "latent_failure_mechanism": label,
            }
        )
    grouped: Counter[tuple[int, str, str, str]] = Counter()
    totals: Counter[tuple[int, str, str]] = Counter()
    for row in detailed:
        base_key = (
            int(row["target_context_tokens"]),
            str(row["placement"]),
            str(row["condition"]),
        )
        grouped[(*base_key, str(row["latent_failure_mechanism"]))] += 1
        totals[base_key] += 1
    summary = [
        {
            "target_context_tokens": length,
            "placement": placement,
            "condition": condition,
            "latent_failure_mechanism": mechanism,
            "count": count,
            "fraction": count / totals[(length, placement, condition)],
            "sample_count": totals[(length, placement, condition)],
        }
        for (length, placement, condition, mechanism), count in sorted(grouped.items())
    ]
    return detailed, summary


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and summarize length-causal shards.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_seed_count", type=int, default=16)
    parser.add_argument("--expected_lengths", type=int, default=5)
    parser.add_argument("--expected_conditions", type=int, default=6)
    parser.add_argument("--expected_placements", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files = sorted(input_dir.glob("shard*/results.jsonl"))
    score_files = sorted(input_dir.glob("shard*/candidate_scores.jsonl"))
    case_files = sorted(input_dir.glob("shard*/cases.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"no completed shards under {input_dir}")

    rows = [row for path in result_files for row in read_jsonl(path)]
    scores = [row for path in score_files for row in read_jsonl(path)]
    cases = [row for path in case_files for row in read_jsonl(path)]
    expected_bodies = (
        args.expected_seed_count
        * args.expected_lengths
        * args.expected_conditions
        * args.expected_placements
    )
    expected_results = expected_bodies * len(causal.QUERY_SPECS)
    validation = {
        "completed_shard_count": len(result_files),
        "result_count": len(rows),
        "expected_result_count": expected_results,
        "case_count": len(cases),
        "expected_case_count": expected_bodies,
        "candidate_score_count": len(scores),
        "seed_values": sorted({int(row["seed"]) for row in rows}),
        "complete": len(rows) == expected_results and len(cases) == expected_bodies,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not validation["complete"]:
        raise RuntimeError(f"incomplete aggregate: {validation}")

    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "candidate_scores.csv", scores)
    core = core_summary(rows)
    probes = probe_summary(rows)
    final_only = final_only_summary(scores)
    paired = paired_effect_rows(rows)
    mechanism_detail, mechanisms = causal.mechanism_rows(rows)
    latent_detail, latent_mechanisms = latent_mechanism_rows(rows)
    write_csv(output_dir / "core_summary.csv", core)
    write_csv(output_dir / "probe_summary.csv", probes)
    write_csv(output_dir / "final_only_summary.csv", final_only)
    write_csv(output_dir / "paired_effects.csv", paired)
    write_csv(output_dir / "mechanism_detail.csv", mechanism_detail)
    write_csv(output_dir / "mechanism_summary.csv", mechanisms)
    write_csv(output_dir / "latent_mechanism_detail.csv", latent_detail)
    write_csv(output_dir / "latent_mechanism_summary.csv", latent_mechanisms)

    clean = [row for row in core if row["condition"] == "clean"]
    conflict = [row for row in core if row["condition"] == "conflict1"]
    report = [
        "# Length-causal mechanism aggregate",
        "",
        f"Validated {len(rows)} query rows from {len(result_files)} shards and "
        f"{args.expected_seed_count} paired seeds.",
        "",
        "## Clean length control",
        "",
        markdown_table(
            clean,
            (
                "target_context_tokens",
                "sample_count",
                "generation_final_accuracy",
                "start_excluded_candidate_accuracy",
                "mean_gold_ppl",
                "mean_candidate_margin",
            ),
        ),
        "",
        "## Conflict length control",
        "",
        markdown_table(
            conflict,
            (
                "target_context_tokens",
                "sample_count",
                "generation_final_accuracy",
                "start_excluded_candidate_accuracy",
                "mean_gold_ppl",
                "mean_candidate_margin",
            ),
        ),
        "",
        "Primary candidate accuracy includes the input/start code. "
        "The start-excluded column is retained only for comparison with the historical protocol.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
