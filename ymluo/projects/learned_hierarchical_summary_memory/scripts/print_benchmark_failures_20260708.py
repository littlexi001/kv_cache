from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs"))
    parser.add_argument("--names", required=True, help="Comma-separated output directory names.")
    parser.add_argument("--max_pred_chars", type=int, default=180)
    args = parser.parse_args()

    for name in [item.strip() for item in args.names.split(",") if item.strip()]:
        path = args.base / name / "trials.csv"
        if not path.exists():
            path = args.base / name / "trials.partial.csv"
        print(f"\n{name} source={path.name}")
        if not path.exists():
            print("missing")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    score = float(row["score"])
                except (KeyError, ValueError):
                    continue
                if row.get("method") == "full_raw" or score >= 1.0:
                    continue
                pred = row.get("prediction", "").replace("\n", " ")[: args.max_pred_chars]
                print(
                    f"{row.get('benchmark')} {row.get('task')} case={row.get('case_id')} "
                    f"{row.get('method')} score={score:.4f} pred={pred} ans={row.get('answers')}"
                )


if __name__ == "__main__":
    main()
