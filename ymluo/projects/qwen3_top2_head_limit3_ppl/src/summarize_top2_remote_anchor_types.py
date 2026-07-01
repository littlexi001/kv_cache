from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "he",
    "she",
    "it",
    "they",
    "we",
    "you",
    "i",
    "his",
    "her",
    "him",
    "them",
    "that",
    "this",
    "was",
    "is",
    "had",
    "have",
    "not",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize remote true-top2 token counts by structural/semantic/other anchor type."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--page_size", type=int, default=64)
    parser.add_argument("--top_tokens_per_type", type=int, default=100)
    return parser.parse_args()


def unescape_text(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")


def fine_category(token_text: str) -> str:
    raw = unescape_text(token_text)
    stripped = raw.strip()
    if not stripped:
        return "whitespace/newline"
    if "\n" in raw:
        if any(ch in raw for ch in '.?!:;,"\u201c\u201d\u2019'):
            return "sentence/dialogue boundary"
        return "newline"
    if re.fullmatch(r"[\.,!?;:\u2014\-\)\(\[\]\"'\u201c\u201d\u2019]+", stripped):
        return "punctuation"
    if re.fullmatch(r"\d+", stripped):
        return "number"
    if stripped.lower() in FUNCTION_WORDS:
        return "function word/pronoun"
    if stripped[:1].isupper():
        return "capitalized/name-like"
    return "content word/subword"


def anchor_type(token_text: str) -> str:
    category = fine_category(token_text)
    if category in {"sentence/dialogue boundary", "newline", "whitespace/newline", "punctuation"}:
        return "structural"
    if category in {"content word/subword", "capitalized/name-like", "number"}:
        return "semantic"
    return "other"


@dataclass
class Accumulator:
    selected_events: int = 0
    selected_attention_mass_sum: float = 0.0
    token_indices: set[int] | None = None

    def add(self, token_index: int, events: int, mass: float) -> None:
        self.selected_events += events
        self.selected_attention_mass_sum += mass
        if self.token_indices is None:
            self.token_indices = set()
        self.token_indices.add(token_index)

    @property
    def unique_tokens(self) -> int:
        return len(self.token_indices or ())


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["token_index"] = int(row["token_index"])
            row["selected_count"] = int(row["selected_count"])
            row["selected_attention_mass_sum"] = float(row.get("selected_attention_mass_sum") or 0.0)
            row["anchor_type"] = anchor_type(row.get("token_text", ""))
            row["fine_category"] = fine_category(row.get("token_text", ""))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accumulator_rows(grouped: dict[tuple[Any, ...], dict[str, Accumulator]], scope_fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope_key, by_type in sorted(grouped.items()):
        total_events = sum(acc.selected_events for acc in by_type.values())
        total_mass = sum(acc.selected_attention_mass_sum for acc in by_type.values())
        for anchor in ["structural", "semantic", "other"]:
            acc = by_type.get(anchor, Accumulator())
            row = {field: value for field, value in zip(scope_fields, scope_key)}
            row.update(
                {
                    "anchor_type": anchor,
                    "unique_tokens": acc.unique_tokens,
                    "selected_events": acc.selected_events,
                    "selected_event_fraction": acc.selected_events / total_events if total_events else 0.0,
                    "selected_attention_mass_sum": acc.selected_attention_mass_sum,
                    "selected_attention_mass_fraction": (
                        acc.selected_attention_mass_sum / total_mass if total_mass else 0.0
                    ),
                    "mean_attention_mass_per_event": (
                        acc.selected_attention_mass_sum / acc.selected_events if acc.selected_events else 0.0
                    ),
                }
            )
            rows.append(row)
    return rows


def group_by_anchor(rows: list[dict[str, Any]], scope_fields: list[str]) -> dict[tuple[Any, ...], dict[str, Accumulator]]:
    grouped: dict[tuple[Any, ...], dict[str, Accumulator]] = defaultdict(lambda: defaultdict(Accumulator))
    for row in rows:
        scope_key = tuple(row[field] for field in scope_fields)
        grouped[scope_key][row["anchor_type"]].add(
            int(row["token_index"]),
            int(row["selected_count"]),
            float(row["selected_attention_mass_sum"]),
        )
    return grouped


def page_proxy_rows(rows: list[dict[str, Any]], scope_fields: list[str], page_size: int) -> list[dict[str, Any]]:
    by_scope: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope[tuple(row[field] for field in scope_fields)].append(row)

    output: list[dict[str, Any]] = []
    for scope_key, scope_rows in sorted(by_scope.items()):
        structural_pages = {
            int(row["token_index"]) // page_size for row in scope_rows if row["anchor_type"] == "structural"
        }
        semantic_rows = [row for row in scope_rows if row["anchor_type"] == "semantic"]
        semantic_events = sum(int(row["selected_count"]) for row in semantic_rows)
        semantic_mass = sum(float(row["selected_attention_mass_sum"]) for row in semantic_rows)
        semantic_events_on_structural_pages = sum(
            int(row["selected_count"])
            for row in semantic_rows
            if int(row["token_index"]) // page_size in structural_pages
        )
        semantic_mass_on_structural_pages = sum(
            float(row["selected_attention_mass_sum"])
            for row in semantic_rows
            if int(row["token_index"]) // page_size in structural_pages
        )
        row = {field: value for field, value in zip(scope_fields, scope_key)}
        row.update(
            {
                "page_size": page_size,
                "structural_pages": len(structural_pages),
                "semantic_events": semantic_events,
                "semantic_events_on_structural_pages": semantic_events_on_structural_pages,
                "semantic_event_fraction_on_structural_pages": (
                    semantic_events_on_structural_pages / semantic_events if semantic_events else 0.0
                ),
                "semantic_attention_mass_sum": semantic_mass,
                "semantic_attention_mass_on_structural_pages": semantic_mass_on_structural_pages,
                "semantic_attention_mass_fraction_on_structural_pages": (
                    semantic_mass_on_structural_pages / semantic_mass if semantic_mass else 0.0
                ),
            }
        )
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    if args.page_size <= 0:
        raise ValueError("--page_size must be positive.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    token_rows = read_rows(input_dir / "token_selection_counts.csv")
    layer_rows = read_rows(input_dir / "token_selection_counts_by_layer.csv")
    layer_head_rows = read_rows(input_dir / "token_selection_counts_by_layer_head.csv")
    for row in layer_rows:
        row["layer"] = int(row["layer"])
    for row in layer_head_rows:
        row["layer"] = int(row["layer"])
        row["head"] = int(row["head"])

    overall_rows = accumulator_rows(group_by_anchor(token_rows, []), [])
    write_csv(
        output_dir / "typed_anchor_summary.csv",
        overall_rows,
        [
            "anchor_type",
            "unique_tokens",
            "selected_events",
            "selected_event_fraction",
            "selected_attention_mass_sum",
            "selected_attention_mass_fraction",
            "mean_attention_mass_per_event",
        ],
    )

    by_layer_rows = accumulator_rows(group_by_anchor(layer_rows, ["layer"]), ["layer"])
    write_csv(
        output_dir / "typed_anchor_by_layer.csv",
        by_layer_rows,
        [
            "layer",
            "anchor_type",
            "unique_tokens",
            "selected_events",
            "selected_event_fraction",
            "selected_attention_mass_sum",
            "selected_attention_mass_fraction",
            "mean_attention_mass_per_event",
        ],
    )

    by_layer_head_rows = accumulator_rows(group_by_anchor(layer_head_rows, ["layer", "head"]), ["layer", "head"])
    write_csv(
        output_dir / "typed_anchor_by_layer_head.csv",
        by_layer_head_rows,
        [
            "layer",
            "head",
            "anchor_type",
            "unique_tokens",
            "selected_events",
            "selected_event_fraction",
            "selected_attention_mass_sum",
            "selected_attention_mass_fraction",
            "mean_attention_mass_per_event",
        ],
    )

    top_rows: list[dict[str, Any]] = []
    for anchor in ["structural", "semantic", "other"]:
        candidates = [row for row in token_rows if row["anchor_type"] == anchor and int(row["selected_count"]) > 0]
        candidates.sort(key=lambda row: (int(row["selected_count"]), float(row["selected_attention_mass_sum"])), reverse=True)
        for rank, row in enumerate(candidates[: max(0, args.top_tokens_per_type)], start=1):
            top_rows.append(
                {
                    "anchor_type": anchor,
                    "rank": rank,
                    "token_index": row["token_index"],
                    "token_id": row.get("token_id", ""),
                    "token_text": row.get("token_text", ""),
                    "fine_category": row["fine_category"],
                    "eligible_queries": row.get("eligible_queries", ""),
                    "selected_count": row["selected_count"],
                    "selection_rate": row.get("selection_rate", ""),
                    "selected_attention_mass_sum": row["selected_attention_mass_sum"],
                }
            )
    write_csv(
        output_dir / "top_tokens_by_anchor_type.csv",
        top_rows,
        [
            "anchor_type",
            "rank",
            "token_index",
            "token_id",
            "token_text",
            "fine_category",
            "eligible_queries",
            "selected_count",
            "selection_rate",
            "selected_attention_mass_sum",
        ],
    )

    page_proxy = page_proxy_rows(token_rows, [], args.page_size)
    write_csv(
        output_dir / "typed_anchor_page_proxy.csv",
        page_proxy,
        [
            "page_size",
            "structural_pages",
            "semantic_events",
            "semantic_events_on_structural_pages",
            "semantic_event_fraction_on_structural_pages",
            "semantic_attention_mass_sum",
            "semantic_attention_mass_on_structural_pages",
            "semantic_attention_mass_fraction_on_structural_pages",
        ],
    )
    page_proxy_layer_head = page_proxy_rows(layer_head_rows, ["layer", "head"], args.page_size)
    write_csv(
        output_dir / "typed_anchor_page_proxy_by_layer_head.csv",
        page_proxy_layer_head,
        [
            "layer",
            "head",
            "page_size",
            "structural_pages",
            "semantic_events",
            "semantic_events_on_structural_pages",
            "semantic_event_fraction_on_structural_pages",
            "semantic_attention_mass_sum",
            "semantic_attention_mass_on_structural_pages",
            "semantic_attention_mass_fraction_on_structural_pages",
        ],
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "page_size": args.page_size,
        "anchor_type_definition": {
            "structural": "punctuation, sentence/dialogue boundaries, newline/whitespace tokens",
            "semantic": "content/subword, capitalized/name-like, and numeric tokens",
            "other": "mostly function words and pronouns",
        },
        "paths": {
            "typed_anchor_summary": str(output_dir / "typed_anchor_summary.csv"),
            "typed_anchor_by_layer": str(output_dir / "typed_anchor_by_layer.csv"),
            "typed_anchor_by_layer_head": str(output_dir / "typed_anchor_by_layer_head.csv"),
            "top_tokens_by_anchor_type": str(output_dir / "top_tokens_by_anchor_type.csv"),
            "typed_anchor_page_proxy": str(output_dir / "typed_anchor_page_proxy.csv"),
            "typed_anchor_page_proxy_by_layer_head": str(output_dir / "typed_anchor_page_proxy_by_layer_head.csv"),
        },
    }
    (output_dir / "typed_anchor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
