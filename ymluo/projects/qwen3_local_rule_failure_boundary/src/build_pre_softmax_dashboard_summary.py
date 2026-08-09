#!/usr/bin/env python3
"""Build a compact browser payload for the saved pre-softmax Q/K diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROLES = ("hop1_result", "hop2_input", "hop2_result")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_summary", type=Path, required=True)
    parser.add_argument("--role_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis_rows = read_csv(args.analysis_summary)
    role_rows = read_csv(args.role_summary)
    role_by_key = {
        (int(row["length"]), row["role"]): row
        for row in role_rows
        if row["role"] in ROLES
    }

    rows: list[dict[str, object]] = []
    for row in analysis_rows:
        length = int(row["length"])
        roles: dict[str, dict[str, float]] = {}
        for role in ROLES:
            detail = role_by_key[(length, role)]
            roles[role] = {
                "mean_logit": number(detail, "mean_logit"),
                "mean_cosine": number(detail, "mean_cosine"),
                "mean_rank": number(detail, "mean_rank"),
                "mean_rank_percentile": number(detail, "mean_rank_percentile"),
                "top2pct_head_fraction": number(detail, "top2pct_head_fraction"),
                "top100_head_fraction": number(detail, "top100_head_fraction"),
                "mean_max_logit_gap": number(detail, "mean_max_logit_gap"),
            }
        rows.append(
            {
                "length": length,
                "prompt_tokens": int(row["prompt_tokens"]),
                "key_length": int(row["key_length"]),
                "top2pct_budget": int(row["top2pct_budget"]),
                "gold_ppl": number(row, "gold_ppl"),
                "mean_head_logsumexp": number(row, "mean_head_logsumexp"),
                "mean_head_max_logit": number(row, "mean_head_max_logit"),
                "mean_query_norm": number(row, "mean_query_norm"),
                "roles": roles,
            }
        )

    payload = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "condition": "clean_two_hop_english_single_token_middle_full2",
        "aggregation": "mean_over_36_layers_x_32_query_heads",
        "role_order": list(ROLES),
        "rows": rows,
        "limitations": {
            "full_token_logits_saved": False,
            "description": (
                "The experiment saved exact pre-softmax diagnostics for the marked evidence roles, "
                "plus per-head maxima/logsumexp. It did not persist every token's full QK-logit vector."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
