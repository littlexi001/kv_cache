from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


HEAD_METRICS = [
    "gold_rule_mass",
    "decoy_rule_mass",
    "competitor_rule_mass",
    "non_gold_rule_mass",
    "gold_rule_selectivity",
    "gold_uniform_enrichment",
    "gold_vs_decoy_log2_density_ratio",
    "gold_vs_background_density_ratio",
    "gold_top2_token_recall",
    "gold_top2_token_precision",
    "gold_top2_mass_recall",
    "gold_best_token_rank",
    "gold_mean_token_rank",
    "gold_step_0_mass",
    "gold_step_1_mass",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row[field])
        return value if math.isfinite(value) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def aggregate(
    rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
    metric_fields: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(key_fields, key))
        item["sample_count"] = len(group)
        item["pair_count"] = len({row.get("pair_id", "") for row in group})
        for field in metric_fields:
            values = [number(row, field) for row in group]
            values = [value for value in values if math.isfinite(value)]
            item[f"mean_{field}"] = mean(values) if values else float("nan")
            item[f"std_{field}"] = stdev(values) if len(values) > 1 else float("nan")
            item[f"sem_{field}"] = (
                item[f"std_{field}"] / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            )
        output.append(item)
    return output


def add_all_competitor_scope(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    copied = list(rows)
    for row in rows:
        cloned = dict(row)
        cloned["competitor_count"] = "all"
        copied.append(cloned)
    return copied


def paired_effects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["pair_id"], row["layer"], row["head"])
        indexed[key][row["condition"]] = row
    deltas: list[dict[str, str]] = []
    for (pair_id, layer, head), variants in indexed.items():
        if set(variants) != {"conflict", "nonconflict"}:
            continue
        conflict = variants["conflict"]
        nonconflict = variants["nonconflict"]
        item: dict[str, Any] = {
            "pair_id": pair_id,
            "competitor_count": conflict["competitor_count"],
            "layer": layer,
            "head": head,
        }
        for metric in HEAD_METRICS:
            item[f"delta_{metric}"] = number(conflict, metric) - number(nonconflict, metric)
        deltas.append({key: str(value) for key, value in item.items()})
    delta_metrics = [f"delta_{metric}" for metric in HEAD_METRICS]
    return aggregate(
        add_all_competitor_scope(deltas),
        ("competitor_count", "layer", "head"),
        delta_metrics,
    )


def case_paired_effects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        indexed[row["pair_id"]][row["condition"]] = row
    output: list[dict[str, Any]] = []
    for pair_id, variants in sorted(indexed.items()):
        if set(variants) != {"conflict", "nonconflict"}:
            continue
        conflict = variants["conflict"]
        nonconflict = variants["nonconflict"]
        output.append(
            {
                "pair_id": pair_id,
                "competitor_count": conflict["competitor_count"],
                "conflict_correct": int(number(conflict, "candidate_correct")),
                "nonconflict_correct": int(number(nonconflict, "candidate_correct")),
                "delta_correct": number(conflict, "candidate_correct")
                - number(nonconflict, "candidate_correct"),
                "delta_candidate_margin": number(conflict, "candidate_margin")
                - number(nonconflict, "candidate_margin"),
                "delta_mean_gold_rule_mass": number(conflict, "mean_gold_rule_mass")
                - number(nonconflict, "mean_gold_rule_mass"),
                "delta_mean_gold_rule_selectivity": number(
                    conflict, "mean_gold_rule_selectivity"
                )
                - number(nonconflict, "mean_gold_rule_selectivity"),
            }
        )
    return output


def top_heads(head_summary: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in head_summary:
        groups[(str(row["condition"]), str(row["competitor_count"]))].append(row)
    output: list[dict[str, Any]] = []
    for (condition, competitors), group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda row: (
                float(row["mean_gold_rule_mass"]),
                float(row["mean_gold_rule_selectivity"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked[:top_k], start=1):
            output.append({"rank": rank, **row})
    return output


def heatmap(
    rows: list[dict[str, Any]],
    value_field: str,
    output_path: Path,
    title: str,
    centered: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted({int(row["layer"]) for row in rows})
    heads = sorted({int(row["head"]) for row in rows})
    matrix = np.full((len(layers), len(heads)), np.nan)
    layer_index = {value: idx for idx, value in enumerate(layers)}
    head_index = {value: idx for idx, value in enumerate(heads)}
    for row in rows:
        matrix[layer_index[int(row["layer"])], head_index[int(row["head"])]] = float(
            row[value_field]
        )
    kwargs: dict[str, Any] = {}
    if centered:
        limit = float(np.nanmax(np.abs(matrix)))
        kwargs = {"vmin": -limit, "vmax": limit, "cmap": "coolwarm"}
    fig, ax = plt.subplots(figsize=(max(7, len(heads) * 0.55), max(5, len(layers) * 0.32)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)), heads)
    if len(layers) <= 36:
        ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired per-head evidence attention runs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    directory = Path(args.input_dir)
    head_rows = read_csv(directory / "head_attention.csv")
    case_rows = read_csv(directory / "case_results.csv")

    scoped_head_rows = add_all_competitor_scope(head_rows)
    head_summary = aggregate(
        scoped_head_rows,
        ("condition", "competitor_count", "layer", "head"),
        HEAD_METRICS,
    )
    condition_summary = aggregate(
        add_all_competitor_scope(case_rows),
        ("condition", "competitor_count"),
        [
            "candidate_correct",
            "candidate_margin",
            "gold_candidate_mean_nll",
            "mean_gold_rule_mass",
            "mean_gold_rule_selectivity",
            "mean_gold_uniform_enrichment",
            "mean_gold_top2_token_recall",
        ],
    )
    correctness_summary = aggregate(
        head_rows,
        ("condition", "candidate_correct"),
        HEAD_METRICS,
    )
    effects = paired_effects(head_rows)
    case_effects = case_paired_effects(case_rows)
    ranked = top_heads(head_summary, args.top_k)

    write_csv(directory / "head_summary_by_condition.csv", head_summary)
    write_csv(directory / "condition_summary.csv", condition_summary)
    write_csv(directory / "correctness_summary.csv", correctness_summary)
    write_csv(directory / "paired_conflict_effect_by_head.csv", effects)
    write_csv(directory / "paired_conflict_effect_by_case.csv", case_effects)
    write_csv(directory / "top_heads.csv", ranked)

    for condition in ("nonconflict", "conflict"):
        selected = [
            row
            for row in head_summary
            if row["condition"] == condition and row["competitor_count"] == "all"
        ]
        heatmap(
            selected,
            "mean_gold_rule_mass",
            directory / f"heatmap_{condition}_gold_mass.png",
            f"{condition}: mean attention mass on gold rules",
        )
        heatmap(
            selected,
            "mean_gold_rule_selectivity",
            directory / f"heatmap_{condition}_gold_selectivity.png",
            f"{condition}: gold / (gold + non-gold rule) attention",
        )
    selected_effects = [row for row in effects if row["competitor_count"] == "all"]
    heatmap(
        selected_effects,
        "mean_delta_gold_rule_mass",
        directory / "heatmap_conflict_delta_gold_mass.png",
        "Conflict - nonconflict: attention mass on gold rules",
        centered=True,
    )
    heatmap(
        selected_effects,
        "mean_delta_gold_rule_selectivity",
        directory / "heatmap_conflict_delta_gold_selectivity.png",
        "Conflict - nonconflict: gold-rule selectivity",
        centered=True,
    )

    payload = {
        "condition_summary": condition_summary,
        "top_5_heads": [row for row in ranked if row["rank"] <= 5],
        "case_paired_effects": case_effects,
        "artifacts": sorted(path.name for path in directory.iterdir() if path.is_file()),
    }
    (directory / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["condition_summary"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
