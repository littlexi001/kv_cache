from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROLE_INDEX = {
    "start_key": 0,
    "hop1_result": 1,
    "hop2_input": 2,
    "hop2_result": 3,
}


def load_mode(output_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in (output_dir / "data").glob("length_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows[int(payload["target_context_tokens"])] = payload
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def role_mass(payload: dict[str, Any], role: str) -> float:
    return float(payload["attention"]["overall_role_mass"][ROLE_INDEX[role]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched multi-token and single-token sweeps.")
    parser.add_argument("--legacy_dir", required=True)
    parser.add_argument("--single_token_dir", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    legacy_dir = Path(args.legacy_dir)
    single_dir = Path(args.single_token_dir)
    output_dir = Path(args.output_dir) if args.output_dir else single_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = load_mode(legacy_dir)
    single = load_mode(single_dir)
    lengths = sorted(set(legacy) & set(single))
    if not lengths:
        raise SystemExit("no common completed lengths")

    rows: list[dict[str, Any]] = []
    legacy_base = float(legacy[lengths[0]]["answer"]["gold_ppl"])
    single_base = float(single[lengths[0]]["answer"]["gold_ppl"])
    for length in lengths:
        old = legacy[length]
        new = single[length]
        old_ppl = float(old["answer"]["gold_ppl"])
        new_ppl = float(new["answer"]["gold_ppl"])
        row: dict[str, Any] = {
            "length": length,
            "legacy_ppl": old_ppl,
            "single_token_ppl": new_ppl,
            "legacy_relative_ppl": old_ppl / legacy_base,
            "single_token_relative_ppl": new_ppl / single_base,
            "legacy_entropy": float(old["attention"]["overall_entropy"]),
            "single_token_entropy": float(new["attention"]["overall_entropy"]),
        }
        for role in ("hop1_result", "hop2_input", "hop2_result"):
            row[f"legacy_{role}_mass"] = role_mass(old, role)
            row[f"single_token_{role}_mass"] = role_mass(new, role)
        rows.append(row)

    csv_path = output_dir / "code_mode_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    xs = [float(length) for length in lengths]
    old_ppl = [float(row["legacy_ppl"]) for row in rows]
    new_ppl = [float(row["single_token_ppl"]) for row in rows]
    old_relative = [float(row["legacy_relative_ppl"]) for row in rows]
    new_relative = [float(row["single_token_relative_ppl"]) for row in rows]
    anchors = [length for length in (0, 1000, 8000, 32000, 64000) if length in legacy and length in single]
    lines = [
        "# Multi-token vs single-token evidence-code comparison",
        "",
        f"Common lengths: {len(lengths)} ({lengths[0]} to {lengths[-1]}).",
        "",
        "Important: absolute PPL is not directly calibrated across modes because the legacy answer",
        "contains multiple tokenizer tokens whereas the control answer contains one token. Compare",
        "within-mode trends and PPL relative to each mode's length-0 baseline.",
        "",
        "## Trend summary",
        "",
        f"- Legacy length/PPL Pearson: {pearson(xs, old_ppl):.4f}",
        f"- Single-token length/PPL Pearson: {pearson(xs, new_ppl):.4f}",
        f"- Legacy length/relative-PPL Pearson: {pearson(xs, old_relative):.4f}",
        f"- Single-token length/relative-PPL Pearson: {pearson(xs, new_relative):.4f}",
        "",
        "## Anchor lengths",
        "",
        "| Length | Legacy PPL | Legacy / short | Single-token PPL | Single / short |",
        "|---:|---:|---:|---:|---:|",
    ]
    lookup = {int(row["length"]): row for row in rows}
    for length in anchors:
        row = lookup[length]
        lines.append(
            f"| {length} | {row['legacy_ppl']:.4f} | {row['legacy_relative_ppl']:.3f}x | "
            f"{row['single_token_ppl']:.4f} | {row['single_token_relative_ppl']:.3f}x |"
        )
    lines.append("")
    (output_dir / "code_mode_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} and {output_dir / 'code_mode_comparison.md'}")


if __name__ == "__main__":
    main()
