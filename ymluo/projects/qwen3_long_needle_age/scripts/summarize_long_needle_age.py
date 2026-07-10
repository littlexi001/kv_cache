from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print long needle age summary.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = read_rows(output_dir / "summary_by_length.csv")
    if not rows:
        raise SystemExit(f"no summary found under {output_dir}")
    print("| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['target_length']} | {row['depth_percent']} | {row['cases']} | "
            f"{row['accuracy']} | {row['miss_rate']} | {row['wrong_rate']} | "
            f"{row['mean_answer_ppl']} | {row['mean_evidence_mass']} |"
        )


if __name__ == "__main__":
    main()
