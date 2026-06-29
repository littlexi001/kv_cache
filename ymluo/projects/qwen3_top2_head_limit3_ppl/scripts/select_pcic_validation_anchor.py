from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def combo_tag(combo: str) -> str:
    return combo.replace(",", "_")


def read_eval_rows(output_name: str) -> list[dict[str, str]]:
    path = OUTPUTS / output_name / "pcic_r_blockwise_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "pcic_r_eval"]


def summarize_combo(output_name: str) -> dict[str, float]:
    rows = read_eval_rows(output_name)
    avg_delta_ppl = sum(float(row["delta_ppl"]) for row in rows) / len(rows)
    avg_delta_loss = sum(float(row.get("delta_loss") or 0.0) for row in rows) / len(rows)
    worst_delta_ppl = max(float(row["delta_ppl"]) for row in rows)
    return {
        "avg_delta_ppl": avg_delta_ppl,
        "avg_delta_loss": avg_delta_loss,
        "worst_delta_ppl": worst_delta_ppl,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combos", required=True, help="Semicolon- or space-separated combos.")
    parser.add_argument("--fixed_pattern", required=True, help="Output pattern containing {combo_tag}.")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument(
        "--score",
        choices=["avg_delta_ppl", "avg_delta_loss", "worst_delta_ppl"],
        default="avg_delta_ppl",
    )
    parser.add_argument("--print_table", action="store_true")
    return parser.parse_args()


def split_combos(raw: str) -> list[str]:
    chunks = raw.replace(";", " ").split()
    combos: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        combo = chunk.strip()
        if combo and combo not in seen:
            combos.append(combo)
            seen.add(combo)
    return combos


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for combo in split_combos(args.combos):
        output_name = args.fixed_pattern.format(combo_tag=combo_tag(combo))
        summary = summarize_combo(output_name)
        rows.append({"combo": combo, "output": output_name, **summary})
    rows.sort(key=lambda row: (float(row[args.score]), float(row["avg_delta_loss"])))
    if args.print_table:
        print("| combo | avg_delta_ppl | avg_delta_loss | worst_delta_ppl | output |")
        print("| --- | ---: | ---: | ---: | --- |")
        for row in rows:
            print(
                f"| `{row['combo']}` | {float(row['avg_delta_ppl']):.6f} | "
                f"{float(row['avg_delta_loss']):.6f} | {float(row['worst_delta_ppl']):.6f} | "
                f"`{row['output']}` |"
            )
        return
    print(";".join(str(row["combo"]) for row in rows[: args.topk]))


if __name__ == "__main__":
    main()
