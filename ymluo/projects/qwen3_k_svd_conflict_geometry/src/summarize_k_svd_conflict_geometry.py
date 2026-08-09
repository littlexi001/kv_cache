from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GEOMETRY_METRICS = (
    "spectral_top_energy_fraction",
    "q_conflict_top_energy_fraction",
    "q_clean_top_energy_fraction",
    "gold_vs_conflict_rule_cos_raw",
    "gold_vs_conflict_rule_pre_rope_cos_raw",
    "gold_vs_conflict_rule_cos_top",
    "gold_vs_conflict_rule_cos_tail",
    "gold_vs_conflict_rule_a_top_energy_fraction",
    "gold_vs_conflict_rule_b_top_energy_fraction",
    "gold_vs_conflict_rule_delta_top_energy_fraction",
    "gold_vs_conflict_rule_relative_delta",
    "gold_vs_conflict_code_cos_raw",
    "gold_vs_conflict_code_pre_rope_cos_raw",
    "gold_vs_conflict_code_cos_top",
    "gold_vs_conflict_code_cos_tail",
    "gold_vs_conflict_code_a_top_energy_fraction",
    "gold_vs_conflict_code_b_top_energy_fraction",
    "gold_vs_conflict_code_delta_top_energy_fraction",
    "gold_vs_conflict_code_relative_delta",
    "query_clean_vs_conflict_cos_raw",
    "query_clean_vs_conflict_cos_top",
    "query_clean_vs_conflict_cos_tail",
    "query_clean_vs_conflict_relative_delta",
    "query_clean_vs_conflict_delta_top_energy_fraction",
    "gold_rule_clean_vs_conflict_cos_raw",
    "gold_rule_clean_vs_conflict_pre_rope_cos_raw",
    "gold_rule_clean_vs_conflict_cos_top",
    "gold_rule_clean_vs_conflict_cos_tail",
    "gold_rule_clean_vs_conflict_relative_delta",
    "gold_rule_clean_vs_conflict_delta_top_energy_fraction",
    "gold_code_clean_vs_conflict_cos_raw",
    "gold_code_clean_vs_conflict_pre_rope_cos_raw",
    "gold_code_clean_vs_conflict_cos_top",
    "gold_code_clean_vs_conflict_cos_tail",
    "gold_code_clean_vs_conflict_relative_delta",
    "gold_code_clean_vs_conflict_delta_top_energy_fraction",
    "rule_qk_gold_dot_top",
    "rule_qk_gold_dot_tail",
    "rule_qk_conflict_dot_top",
    "rule_qk_conflict_dot_tail",
    "rule_qk_gold_minus_conflict_dot_top",
    "rule_qk_gold_minus_conflict_dot_tail",
    "rule_qk_delta_q_cos_top",
    "rule_qk_delta_q_cos_tail",
    "code_qk_gold_dot_top",
    "code_qk_gold_dot_tail",
    "code_qk_conflict_dot_top",
    "code_qk_conflict_dot_tail",
    "code_qk_gold_minus_conflict_dot_top",
    "code_qk_gold_minus_conflict_dot_tail",
    "code_qk_delta_q_cos_top",
    "code_qk_delta_q_cos_tail",
    "query_delta_top_energy_fraction",
    "query_delta_relative_norm",
    "query_delta_vs_rule_delta_cos_top",
    "query_delta_vs_rule_delta_cos_tail",
    "shared_code_gold_vs_conflict_cos_raw",
    "shared_code_gold_vs_conflict_pre_rope_cos_raw",
    "shared_code_gold_vs_conflict_cos_top",
    "shared_code_gold_vs_conflict_cos_tail",
    "shared_code_gold_vs_conflict_a_top_energy_fraction",
    "shared_code_gold_vs_conflict_b_top_energy_fraction",
    "shared_code_gold_vs_conflict_delta_top_energy_fraction",
    "shared_code_gold_vs_conflict_relative_delta",
    "shared_code_qk_gold_dot_top",
    "shared_code_qk_gold_dot_tail",
    "shared_code_qk_conflict_dot_top",
    "shared_code_qk_conflict_dot_tail",
    "shared_code_qk_gold_minus_conflict_dot_top",
    "shared_code_qk_gold_minus_conflict_dot_tail",
    "shared_code_qk_delta_q_cos_top",
    "shared_code_qk_delta_q_cos_tail",
    "shared_numeric_gold_vs_conflict_cos_raw",
    "shared_numeric_gold_vs_conflict_pre_rope_cos_raw",
    "shared_numeric_gold_vs_conflict_cos_top",
    "shared_numeric_gold_vs_conflict_cos_tail",
    "shared_numeric_gold_vs_conflict_a_top_energy_fraction",
    "shared_numeric_gold_vs_conflict_b_top_energy_fraction",
    "shared_numeric_gold_vs_conflict_delta_top_energy_fraction",
    "shared_numeric_gold_vs_conflict_relative_delta",
    "shared_numeric_qk_gold_dot_top",
    "shared_numeric_qk_gold_dot_tail",
    "shared_numeric_qk_conflict_dot_top",
    "shared_numeric_qk_conflict_dot_tail",
    "shared_numeric_qk_gold_minus_conflict_dot_top",
    "shared_numeric_qk_gold_minus_conflict_dot_tail",
    "shared_numeric_qk_delta_q_cos_top",
    "shared_numeric_qk_delta_q_cos_tail",
)

TOKEN_METRICS = (
    "token_gold_vs_conflict_cos_raw",
    "token_gold_vs_conflict_pre_rope_cos_raw",
    "token_gold_vs_conflict_cos_top",
    "token_gold_vs_conflict_cos_tail",
    "token_gold_vs_conflict_a_top_energy_fraction",
    "token_gold_vs_conflict_b_top_energy_fraction",
    "token_gold_vs_conflict_delta_top_energy_fraction",
    "token_gold_vs_conflict_relative_delta",
    "token_qk_gold_dot_top",
    "token_qk_gold_dot_tail",
    "token_qk_conflict_dot_top",
    "token_qk_conflict_dot_tail",
    "token_qk_gold_minus_conflict_dot_top",
    "token_qk_gold_minus_conflict_dot_tail",
    "token_qk_delta_q_cos_top",
    "token_qk_delta_q_cos_tail",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def update(
    groups: dict[tuple[Any, ...], dict[str, list[float]]],
    key: tuple[Any, ...],
    row: dict[str, str],
    metrics: Iterable[str],
) -> None:
    state = groups.setdefault(key, {})
    for metric in metrics:
        number = parse_number(row.get(metric))
        if number is None:
            continue
        pair = state.setdefault(metric, [0.0, 0.0])
        pair[0] += number
        pair[1] += 1.0


def per_seed_means(
    groups: dict[tuple[Any, ...], dict[str, list[float]]]
) -> dict[tuple[Any, ...], dict[str, float]]:
    return {
        key: {metric: value[0] / value[1] for metric, value in state.items() if value[1]}
        for key, state in groups.items()
    }


def aggregate_seed_means(
    seed_means: dict[tuple[Any, ...], dict[str, float]],
    key_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    # The final key element is always seed.  SEM is therefore computed over independent prompts,
    # not over the hundreds of correlated layer/head rows.
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for full_key, metrics in seed_means.items():
        base_key = full_key[:-1]
        for metric, value in metrics.items():
            grouped[base_key][metric].append(value)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        row: dict[str, Any] = dict(zip(key_names, key))
        metric_values = grouped[key]
        all_counts: list[int] = []
        for metric, values in metric_values.items():
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            row[metric] = mean
            row[f"{metric}_sem"] = math.sqrt(variance / len(values))
            all_counts.append(len(values))
        # Some derived quantities are undefined for a mathematically zero delta (for
        # example, gold K when a conflict is causally later).  Report the number of
        # independent seed rows represented by the group; metric-specific SEMs still
        # use only their finite values.
        row["seed_count"] = max(all_counts) if all_counts else 0
        output.append(row)
    return output


def locate(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate K-SVD geometry shards")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    overall_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    layer_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    head_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    order_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    geometry_files = sorted(input_dir.glob("shard_*/geometry_rows.csv"))
    if not geometry_files:
        raise FileNotFoundError(f"No geometry shard files under {input_dir}")

    for path in geometry_files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                context = row["pair_context"]
                rank = int(row["rank"])
                layer = int(row["layer"])
                q_head = int(row["q_head"])
                seed = int(row["seed"])
                order = row["conflict_order"]
                update(overall_groups, (context, rank, seed), row, GEOMETRY_METRICS)
                if rank == 16:
                    update(layer_groups, (context, layer, seed), row, GEOMETRY_METRICS)
                    update(head_groups, (context, layer, q_head, seed), row, GEOMETRY_METRICS)
                    update(order_groups, (context, order, seed), row, GEOMETRY_METRICS)

    overall_seed_means = per_seed_means(overall_groups)
    overall_rows = aggregate_seed_means(
        overall_seed_means, ("pair_context", "rank")
    )
    layer_rows = aggregate_seed_means(
        per_seed_means(layer_groups), ("pair_context", "layer")
    )
    head_rows = aggregate_seed_means(
        per_seed_means(head_groups), ("pair_context", "layer", "q_head")
    )
    for row in head_rows:
        row["kv_head"] = int(row["q_head"]) // 2
    order_rows = aggregate_seed_means(
        per_seed_means(order_groups), ("pair_context", "conflict_order")
    )
    write_csv(output_dir / "overall_rank_summary.csv", overall_rows)
    overall_seed_rows: list[dict[str, Any]] = []
    for (context, rank, seed), metrics in sorted(overall_seed_means.items()):
        overall_seed_rows.append(
            {"pair_context": context, "rank": rank, "seed": seed, **metrics}
        )
    write_csv(output_dir / "overall_rank_seed_means.csv", overall_seed_rows)
    write_csv(output_dir / "layer_summary_r16.csv", layer_rows)
    write_csv(output_dir / "head_summary_r16.csv", head_rows)
    write_csv(output_dir / "conflict_order_summary_r16.csv", order_rows)

    token_overall_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    token_layer_groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    token_files = sorted(input_dir.glob("shard_*/shared_code_token_rows_r16.csv"))
    for path in token_files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                context = row["pair_context"]
                index = int(row["subtoken_index"])
                decoded = row["decoded"]
                is_numeric = int(row["is_numeric"])
                layer = int(row["layer"])
                seed = int(row["seed"])
                update(
                    token_overall_groups,
                    (context, index, decoded, is_numeric, seed),
                    row,
                    TOKEN_METRICS,
                )
                update(
                    token_layer_groups,
                    (context, layer, index, decoded, is_numeric, seed),
                    row,
                    TOKEN_METRICS,
                )
    token_overall_rows = aggregate_seed_means(
        per_seed_means(token_overall_groups),
        ("pair_context", "subtoken_index", "decoded", "is_numeric"),
    )
    token_layer_rows = aggregate_seed_means(
        per_seed_means(token_layer_groups),
        ("pair_context", "layer", "subtoken_index", "decoded", "is_numeric"),
    )
    write_csv(output_dir / "shared_token_summary_r16.csv", token_overall_rows)
    write_csv(output_dir / "shared_token_layer_summary_r16.csv", token_layer_rows)

    main_row = locate(overall_rows, pair_context="filler_8k", rank=16)
    main_seed_rows = [
        row
        for row in overall_seed_rows
        if row["pair_context"] == "filler_8k" and row["rank"] == 16
    ]
    before = locate(order_rows, pair_context="filler_8k", conflict_order="before_gold")
    after = locate(order_rows, pair_context="filler_8k", conflict_order="after_gold")
    filler_heads = [row for row in head_rows if row["pair_context"] == "filler_8k"]
    for row in filler_heads:
        row["shared_numeric_qk_total_advantage"] = (
            float(row["shared_numeric_qk_gold_minus_conflict_dot_top"])
            + float(row["shared_numeric_qk_gold_minus_conflict_dot_tail"])
        )
        row["code_qk_total_advantage"] = (
            float(row["code_qk_gold_minus_conflict_dot_top"])
            + float(row["code_qk_gold_minus_conflict_dot_tail"])
        )
    most_conflict = sorted(filler_heads, key=lambda row: row["code_qk_total_advantage"])[:12]
    most_gold = sorted(filler_heads, key=lambda row: row["code_qk_total_advantage"], reverse=True)[:12]
    write_csv(output_dir / "most_conflict_favoring_heads_r16.csv", most_conflict)
    write_csv(output_dir / "most_gold_favoring_heads_r16.csv", most_gold)

    findings = {
        "input_shards": len(geometry_files),
        "main_context": "filler_8k",
        "main_rank": 16,
        "seed_count": main_row["seed_count"],
        "spectral_top16_energy_fraction": main_row["spectral_top_energy_fraction"],
        "query_top16_energy_fraction": main_row["q_conflict_top_energy_fraction"],
        "gold_vs_conflict_code_cosine": {
            "raw": main_row["gold_vs_conflict_code_cos_raw"],
            "pre_rope_raw": main_row["gold_vs_conflict_code_pre_rope_cos_raw"],
            "top16": main_row["gold_vs_conflict_code_cos_top"],
            "tail": main_row["gold_vs_conflict_code_cos_tail"],
        },
        "gold_vs_conflict_code_delta_top16_energy_fraction": main_row[
            "gold_vs_conflict_code_delta_top_energy_fraction"
        ],
        "code_qk_gold_minus_conflict": {
            "top16": main_row["code_qk_gold_minus_conflict_dot_top"],
            "tail": main_row["code_qk_gold_minus_conflict_dot_tail"],
            "total": main_row["code_qk_gold_minus_conflict_dot_top"]
            + main_row["code_qk_gold_minus_conflict_dot_tail"],
        },
        "shared_numeric_cosine": {
            "raw": main_row["shared_numeric_gold_vs_conflict_cos_raw"],
            "pre_rope_raw": main_row["shared_numeric_gold_vs_conflict_pre_rope_cos_raw"],
            "top16": main_row["shared_numeric_gold_vs_conflict_cos_top"],
            "tail": main_row["shared_numeric_gold_vs_conflict_cos_tail"],
        },
        "shared_numeric_delta_top16_energy_fraction": main_row[
            "shared_numeric_gold_vs_conflict_delta_top_energy_fraction"
        ],
        "shared_numeric_qk_gold_minus_conflict": {
            "top16": main_row["shared_numeric_qk_gold_minus_conflict_dot_top"],
            "tail": main_row["shared_numeric_qk_gold_minus_conflict_dot_tail"],
            "total": main_row["shared_numeric_qk_gold_minus_conflict_dot_top"]
            + main_row["shared_numeric_qk_gold_minus_conflict_dot_tail"],
        },
        "seed_sign_consistency": {
            "code_top16_conflict_favoring_count": sum(
                row["code_qk_gold_minus_conflict_dot_top"] < 0 for row in main_seed_rows
            ),
            "code_tail_gold_favoring_count": sum(
                row["code_qk_gold_minus_conflict_dot_tail"] > 0 for row in main_seed_rows
            ),
            "shared_numeric_top16_conflict_favoring_count": sum(
                row["shared_numeric_qk_gold_minus_conflict_dot_top"] < 0
                for row in main_seed_rows
            ),
            "shared_numeric_tail_gold_favoring_count": sum(
                row["shared_numeric_qk_gold_minus_conflict_dot_tail"] > 0
                for row in main_seed_rows
            ),
            "denominator": len(main_seed_rows),
        },
        "query_clean_vs_conflict": {
            "raw_cosine": main_row["query_clean_vs_conflict_cos_raw"],
            "top16_cosine": main_row["query_clean_vs_conflict_cos_top"],
            "tail_cosine": main_row["query_clean_vs_conflict_cos_tail"],
            "relative_delta": main_row["query_clean_vs_conflict_relative_delta"],
            "delta_top16_energy_fraction": main_row["query_delta_top_energy_fraction"],
        },
        "causal_order_check_gold_code_relative_delta": {
            "conflict_before_gold": before["gold_code_clean_vs_conflict_relative_delta"],
            "conflict_after_gold": after["gold_code_clean_vs_conflict_relative_delta"],
        },
        "most_conflict_favoring_code_heads": [
            {
                "layer": row["layer"],
                "q_head": row["q_head"],
                "kv_head": row["kv_head"],
                "qk_gold_minus_conflict": row["code_qk_total_advantage"],
            }
            for row in most_conflict
        ],
        "most_gold_favoring_code_heads": [
            {
                "layer": row["layer"],
                "q_head": row["q_head"],
                "kv_head": row["kv_head"],
                "qk_gold_minus_conflict": row["code_qk_total_advantage"],
            }
            for row in most_gold
        ],
    }
    (output_dir / "key_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
