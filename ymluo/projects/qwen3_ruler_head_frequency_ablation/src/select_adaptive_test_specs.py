from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIXED_CONTROL = "l25_g3_f46_relative_s8192_a0.25_b1_fixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixed-control", default=FIXED_CONTROL)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = json.loads(
        args.validation_summary.read_text(encoding="utf-8")
    )
    native = next(row for row in rows if row["variant"] == "native_rope")
    fixed = next(row for row in rows if row["variant"] == args.fixed_control)
    eligible = [
        row
        for row in rows
        if "_adaptive_" in row["variant"]
        and float(row["min_seed_official_delta"]) >= 0.0
        and int(row.get("official_degraded", 0)) == 0
        and float(row["mean_gold_nll_improvement"]) > 0.0
    ]
    eligible.sort(
        key=lambda row: (
            int(row.get("nll_degraded", 0)),
            -float(row["mean_gold_nll_improvement"]),
        )
    )
    if not eligible:
        raise SystemExit("No adaptive candidate passed the validation constraints")

    selected = eligible[0]
    payload = {
        "selection_rule": (
            "adaptive candidates only; no per-sample official degradation; "
            "nonnegative official delta on every validation seed; positive mean "
            "Gold NLL improvement; then minimize NLL-degraded samples and maximize "
            "mean Gold NLL improvement"
        ),
        "source": str(args.validation_summary),
        "selected_adaptive": selected["variant"],
        "specs": [native["spec"], selected["spec"], fixed["spec"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_adaptive": selected["variant"],
                "nll_degraded": selected.get("nll_degraded", 0),
                "mean_gold_nll_improvement": selected["mean_gold_nll_improvement"],
                "test_specs": [spec["name"] for spec in payload["specs"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
