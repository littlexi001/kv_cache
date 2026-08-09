from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISCOVERY_CONTROL = "l25_g3_f46_a0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--control", default=DISCOVERY_CONTROL)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    native = next(row for row in rows if row["variant"] == "native_rope")["spec"]
    eligible = [
        row for row in rows
        if row["variant"] != "native_rope"
        and float(row["min_seed_official_delta"]) >= 0.0
        and int(row.get("official_degraded", 0)) == 0
        and float(row["mean_gold_nll_improvement"]) > 0.0
    ]
    eligible.sort(key=lambda row: float(row["mean_gold_nll_improvement"]), reverse=True)
    selected = eligible[: args.limit]
    if args.control:
        control = next(row for row in rows if row["variant"] == args.control)
        if all(row["variant"] != args.control for row in selected):
            selected.append(control)
    specs = [native] + [row["spec"] for row in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "selection_rule": "no per-sample official degradation, nonnegative mean official delta on every validation seed, then maximum Gold NLL improvement",
                "source": str(args.validation_summary),
                "specs": specs,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": [spec["name"] for spec in specs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
