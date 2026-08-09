from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def token_category(token_text: str, token_piece: str = "") -> str:
    if not token_text:
        return "special_or_empty"
    if token_text.isspace():
        return "newline" if "\n" in token_text or "\r" in token_text else "whitespace"
    stripped = token_text.strip()
    if not stripped:
        return "whitespace"
    if all(unicodedata.category(char).startswith("P") for char in stripped):
        return "punctuation"
    if all(unicodedata.category(char).startswith("S") for char in stripped):
        return "symbol"
    normalized = stripped.replace(",", "").replace(".", "")
    if normalized.isdigit():
        return "number"
    if stripped.isalpha():
        return "word"
    if stripped.isalnum():
        return "alphanumeric"
    if token_piece.startswith("<") and token_piece.endswith(">"):
        return "special"
    return "mixed_or_subword"


def safe_float(value: str | int | float | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def summarize_tokens(rows: list[dict[str, str]], top_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    category_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    role_totals = {"sink": 0.0, "recent": 0.0, "remote": 0.0}
    total_events = sum(safe_float(row.get("top2_selected_count")) for row in rows)
    total_exposure = sum(safe_float(row.get("eligible_event_count")) for row in rows)
    total_mass = sum(safe_float(row.get("top2_attention_mass_sum")) for row in rows)

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        category = token_category(row.get("token_text", ""), row.get("token_piece", ""))
        events = safe_float(row.get("top2_selected_count"))
        exposure = safe_float(row.get("eligible_event_count"))
        mass = safe_float(row.get("top2_attention_mass_sum"))
        category_totals[category]["token_count"] += 1
        category_totals[category]["selected_events"] += events
        category_totals[category]["eligible_events"] += exposure
        category_totals[category]["attention_mass"] += mass
        role_totals["sink"] += safe_float(row.get("sink_role_count"))
        role_totals["recent"] += safe_float(row.get("recent_role_count"))
        role_totals["remote"] += safe_float(row.get("remote_role_count"))
        enriched = dict(row)
        enriched["token_category"] = category
        enriched["selected_event_share"] = events / total_events if total_events else 0.0
        enriched["attention_mass_share"] = mass / total_mass if total_mass else 0.0
        enriched_rows.append(enriched)

    category_rows: list[dict[str, Any]] = []
    for category, totals in sorted(category_totals.items()):
        selected_share = totals["selected_events"] / total_events if total_events else 0.0
        exposure_share = totals["eligible_events"] / total_exposure if total_exposure else 0.0
        category_rows.append(
            {
                "token_category": category,
                "token_count": int(totals["token_count"]),
                "selected_events": int(totals["selected_events"]),
                "selected_event_share": selected_share,
                "eligible_event_share": exposure_share,
                "selection_enrichment_vs_exposure": (
                    selected_share / exposure_share if exposure_share else ""
                ),
                "attention_mass_sum": totals["attention_mass"],
                "attention_mass_share": totals["attention_mass"] / total_mass if total_mass else 0.0,
            }
        )

    role_sum = sum(role_totals.values())
    role_rows = [
        {
            "position_role": role,
            "selected_events": int(events),
            "selected_event_share": events / role_sum if role_sum else 0.0,
        }
        for role, events in role_totals.items()
    ]

    top_rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for rank, row in enumerate(
        sorted(enriched_rows, key=lambda item: safe_float(item.get("top2_selected_count")), reverse=True)[:top_n],
        start=1,
    ):
        cumulative += safe_float(row.get("selected_event_share"))
        top_rows.append({"rank": rank, **row, "cumulative_selected_event_share": cumulative})
    return category_rows, role_rows, top_rows


def summarize_overlap(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sink_tokens"])].append(row)
    output: list[dict[str, Any]] = []
    for sink_tokens, group in sorted(grouped.items()):
        query_count = sum(safe_float(row.get("query_count")) for row in group)
        top_events = sum(safe_float(row.get("top2_selected_events")) for row in group)
        overlap_events = sum(safe_float(row.get("overlap_events")) for row in group)
        top_mass = sum(safe_float(row.get("top2_attention_mass_sum")) for row in group)
        overlap_mass = sum(safe_float(row.get("overlap_attention_mass_sum")) for row in group)
        weighted_full_mass = sum(
            safe_float(row.get("mean_sink_recent_full_attention_mass")) * safe_float(row.get("query_count"))
            for row in group
        )
        weighted_cosine = sum(
            safe_float(row.get("mean_pruned_distribution_cosine")) * safe_float(row.get("query_count"))
            for row in group
        )
        output.append(
            {
                "sink_tokens": sink_tokens,
                "query_head_rows": int(query_count),
                "top2_selected_events": int(top_events),
                "overlap_events": int(overlap_events),
                "overlap_event_recall": overlap_events / top_events if top_events else "",
                "overlap_top2_mass_recall": overlap_mass / top_mass if top_mass else "",
                "mean_sink_recent_full_attention_mass": weighted_full_mass / query_count if query_count else "",
                "mean_pruned_distribution_cosine": weighted_cosine / query_count if query_count else "",
            }
        )
    return output


def weighted_metric(rows: Iterable[dict[str, str]], field: str) -> float | str:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        weight = safe_float(row.get("query_count"))
        numerator += float(value) * weight
        denominator += weight
    return numerator / denominator if denominator else ""


def summarize_concentration(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metric_fields = [field for field in rows[0] if field.startswith("mean_")] if rows else []
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, int | str, list[dict[str, str]]]] = [("overall", "", rows)]
    layers = sorted({int(row["layer"]) for row in rows})
    scopes.extend(("layer", layer, [row for row in rows if int(row["layer"]) == layer]) for layer in layers)
    for scope, layer, group in scopes:
        result: dict[str, Any] = {
            "scope": scope,
            "layer": layer,
            "query_head_rows": int(sum(safe_float(row.get("query_count")) for row in group)),
        }
        for field in metric_fields:
            result[field] = weighted_metric(group, field)
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Top-2% token, concentration, and sink+recent diagnostics.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    token_rows = read_csv(run_dir / "top2_token_events.csv")
    overlap_rows = read_csv(run_dir / "sink_recent_overlap_by_layer_head.csv")
    concentration_rows = read_csv(run_dir / "top2_concentration_by_layer_head.csv")

    category_summary, role_summary, top_tokens = summarize_tokens(token_rows, args.top_n)
    overlap_summary = summarize_overlap(overlap_rows)
    concentration_summary = summarize_concentration(concentration_rows)
    write_csv(output_dir / "token_category_summary.csv", category_summary, list(category_summary[0]))
    write_csv(output_dir / "position_role_summary.csv", role_summary, list(role_summary[0]))
    write_csv(output_dir / "top_tokens.csv", top_tokens, list(top_tokens[0]))
    write_csv(output_dir / "sink_recent_overlap_summary.csv", overlap_summary, list(overlap_summary[0]))
    write_csv(
        output_dir / "attention_concentration_summary.csv",
        concentration_summary,
        list(concentration_summary[0]),
    )

    best_overlap = max(
        overlap_summary,
        key=lambda row: safe_float(row.get("mean_pruned_distribution_cosine")),
    )
    total_selected = sum(safe_float(row.get("top2_selected_count")) for row in token_rows)
    role_selected = sum(safe_float(row.get("selected_events")) for row in role_summary)
    summary = {
        "definitions": {
            "selection_enrichment_vs_exposure": (
                "category share of Top-2% selection events divided by category share of eligible events"
            ),
            "overlap_event_recall": "fraction of oracle Top-2% positions also present in equal-budget sink+recent",
            "overlap_top2_mass_recall": "fraction of oracle Top-2% full-attention mass covered by sink+recent",
        },
        "sanity_checks": {
            "role_counts_equal_selected_events": math.isclose(total_selected, role_selected),
            "selected_events": total_selected,
            "role_events": role_selected,
        },
        "best_sink_allocation_by_distribution_cosine": best_overlap,
        "paths": {
            "token_category_summary": str(output_dir / "token_category_summary.csv"),
            "position_role_summary": str(output_dir / "position_role_summary.csv"),
            "top_tokens": str(output_dir / "top_tokens.csv"),
            "sink_recent_overlap_summary": str(output_dir / "sink_recent_overlap_summary.csv"),
            "attention_concentration_summary": str(output_dir / "attention_concentration_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["sanity_checks"], indent=2), flush=True)


if __name__ == "__main__":
    main()

