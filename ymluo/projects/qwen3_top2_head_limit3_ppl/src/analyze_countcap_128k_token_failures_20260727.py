from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired token-level CountCap NLL failures."
    )
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--repair_root", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def row_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        row["topic"],
        int(row["window"]),
        int(row["target_index"]),
    )


def load_baseline(
    root: Path,
) -> tuple[
    dict[tuple[str, int, int], dict[str, str]],
    dict[tuple[str, int, int], dict[str, str]],
]:
    full: dict[tuple[str, int, int], dict[str, str]] = {}
    current: dict[tuple[str, int, int], dict[str, str]] = {}
    for path in sorted(root.glob("length128000_*/token_results.csv")):
        for row in load_csv(path):
            key = row_key(row)
            if row["method"] == "full_attention":
                full[key] = row
            elif row["method"] == "direct_countcap":
                current[key] = row
    if not full or set(current) != set(full):
        raise ValueError("baseline Full and CountCap token rows are not paired")
    return full, current


def load_variants(
    root: Path,
) -> dict[str, dict[tuple[str, int, int], dict[str, str]]]:
    variants: dict[str, dict[tuple[str, int, int], dict[str, str]]] = {}
    for path in sorted(root.glob("*/token_results.csv")):
        rows = {row_key(row): row for row in load_csv(path)}
        if rows:
            variants[path.parent.name] = rows
    return variants


def token_category(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation"
    if any(character.isdigit() for character in stripped):
        return "number_or_mixed"
    if stripped.isalpha():
        return "alphabetic"
    return "other"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    full, current = load_baseline(args.baseline_root)
    variants = {"sampled_k1280_current": current}
    variants.update(load_variants(args.repair_root))
    token_counts = Counter(int(row["token_id"]) for row in full.values())

    summaries: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    for label, sparse in sorted(variants.items()):
        if set(sparse) != set(full):
            continue
        records = []
        by_category: dict[str, list[float]] = defaultdict(list)
        for key in sorted(full):
            token_id = int(full[key]["token_id"])
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            delta = float(sparse[key]["nll"]) - float(full[key]["nll"])
            category = token_category(text)
            by_category[category].append(delta)
            records.append(
                {
                    "variant": label,
                    "topic": key[0],
                    "window": key[1],
                    "target_index": key[2],
                    "token_id": token_id,
                    "token": repr(text),
                    "target_frequency": token_counts[token_id],
                    "category": category,
                    "full_nll": float(full[key]["nll"]),
                    "sparse_nll": float(sparse[key]["nll"]),
                    "delta_nll": delta,
                }
            )
        deltas = [record["delta_nll"] for record in records]
        positive = sorted((max(0.0, value) for value in deltas), reverse=True)
        positive_sum = sum(positive)
        summary = {
            "variant": label,
            "tokens": len(records),
            "mean_delta_nll": sum(deltas) / len(deltas),
            "median_delta_nll": percentile(deltas, 0.5),
            "p90_delta_nll": percentile(deltas, 0.9),
            "p99_delta_nll": percentile(deltas, 0.99),
            "fraction_delta_gt_0p1": sum(value > 0.1 for value in deltas)
            / len(deltas),
            "fraction_delta_gt_1": sum(value > 1.0 for value in deltas)
            / len(deltas),
        }
        for fraction in (0.01, 0.05, 0.10, 0.20):
            count = max(1, math.ceil(fraction * len(positive)))
            summary[f"positive_loss_share_top_{int(100 * fraction)}pct"] = (
                sum(positive[:count]) / positive_sum if positive_sum else 0.0
            )
        summaries.append(summary)
        for category, values in sorted(by_category.items()):
            category_rows.append(
                {
                    "variant": label,
                    "category": category,
                    "tokens": len(values),
                    "mean_delta_nll": sum(values) / len(values),
                    "fraction_delta_gt_0p1": sum(value > 0.1 for value in values)
                    / len(values),
                }
            )
        worst_rows.extend(
            sorted(records, key=lambda record: record["delta_nll"], reverse=True)[:40]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    for name, rows in (
        ("category_breakdown.csv", category_rows),
        ("worst_tokens.csv", worst_rows),
    ):
        if not rows:
            continue
        with (args.output_dir / name).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
