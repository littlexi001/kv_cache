from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine fixed and adaptive exact-attention frontiers.")
    parser.add_argument("--fixed_root", required=True, type=Path)
    parser.add_argument("--adaptive_root", required=True, type=Path, nargs="+")
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fixed_fraction(method: str) -> float:
    if method == "full_attention":
        return 1.0
    value = method.removeprefix("head_top").removesuffix("pct").replace("p", ".")
    return float(value) / 100.0


def collect_fixed(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_w*/summary.csv")):
        case = path.parent.name
        topic, window_spec = case.rsplit("_w", 1)
        for row in read_csv(path):
            if row["topic"] != "all":
                continue
            rows.append(
                {
                    "family": "fixed",
                    "topic": topic,
                    "window": int(window_spec),
                    "method": row["method"],
                    "nll": float(row["nll"]),
                    "ppl": float(row["ppl"]),
                    "attention_link_ratio": fixed_fraction(row["method"]),
                    "online_seconds": float(row["mean_online_seconds_per_case"]),
                }
            )
    return rows


def collect_adaptive(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_w*/summary.csv")):
        for row in read_csv(path):
            converted: dict[str, Any] = {
                "family": "adaptive_mass",
                "topic": row["topic"],
                "window": int(row["window"]),
                "method": row["method"],
                "nll": float(row["nll"]),
                "ppl": float(row["ppl"]),
                "attention_link_ratio": float(row["attention_link_ratio"]),
                "online_seconds": float(row["online_seconds"]),
            }
            for key, value in row.items():
                if key.startswith("budget_") and key.endswith("_rate") and value:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def split_name(window: int, requested: str) -> bool:
    if requested == "all":
        return True
    if requested == "calibration_w01":
        return window in {0, 1}
    if requested == "independent_w2":
        return window == 2
    raise ValueError(requested)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ["calibration_w01", "independent_w2", "all"]:
        split_rows = [row for row in rows if split_name(int(row["window"]), split)]
        full_rows = [row for row in split_rows if row["method"] == "full_attention"]
        full_nll = sum(float(row["nll"]) for row in full_rows) / max(1, len(full_rows))
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            groups[(str(row["family"]), str(row["method"]))].append(row)
        for (family, method), subset in sorted(groups.items()):
            nll = sum(float(row["nll"]) for row in subset) / len(subset)
            link_ratio = sum(float(row["attention_link_ratio"]) for row in subset) / len(subset)
            result: dict[str, Any] = {
                "split": split,
                "family": family,
                "method": method,
                "cases": len(subset),
                "nll": nll,
                "ppl": math.exp(min(20.0, nll)),
                "delta_nll_vs_full": nll - full_nll,
                "ppl_ratio_vs_full": math.exp(min(20.0, nll - full_nll)),
                "attention_link_ratio": link_ratio,
                "attention_v_upper_bound": 1.0 / max(1e-12, link_ratio),
                "mean_online_seconds_full_qk_diagnostic": sum(
                    float(row["online_seconds"]) for row in subset
                )
                / len(subset),
            }
            budget_keys = sorted(
                {key for row in subset for key in row if key.startswith("budget_") and key.endswith("_rate")}
            )
            for key in budget_keys:
                result[key] = sum(float(row.get(key, 0.0)) for row in subset) / len(subset)
            output.append(result)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_fixed(args.fixed_root)
    for root in args.adaptive_root:
        rows.extend(collect_adaptive(root))
    summary = aggregate(rows)
    write_csv(args.output_dir / "frontier.csv", summary)
    (args.output_dir / "frontier.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
