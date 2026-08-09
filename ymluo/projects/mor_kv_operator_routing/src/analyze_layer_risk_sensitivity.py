from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare end-to-end NLL amplification across sparse layer groups."
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="LABEL=merged action_summary.csv; may be repeated.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.group:
        label, path_raw = item.split("=", 1)
        with Path(path_raw).open("r", encoding="utf-8", newline="") as handle:
            actions = list(csv.DictReader(handle))
        full = next(row for row in actions if row["action"] == "full")
        learned = next(
            row for row in actions if row["action"] == "learned_conformal"
        )
        full_logical = float(full["mean_selected_blocks"])
        learned_logical = float(learned["mean_selected_blocks"])
        physical_saving = float(learned["mean_physical_gqa_saving_rate"])
        delta_nll = float(learned["mean_delta_nll_vs_full"])
        rows.append(
            {
                "layer_group": label,
                "logical_saving_rate": 1.0 - learned_logical / full_logical,
                "physical_gqa_saving_rate": physical_saving,
                "mean_delta_nll": delta_nll,
                "delta_nll_ci95_low": float(learned["delta_nll_ci95_low"]),
                "delta_nll_ci95_high": float(learned["delta_nll_ci95_high"]),
                "p95_abs_delta_nll": float(learned["p95_abs_delta_nll"]),
                "nll_per_1pct_physical_saving": delta_nll
                / max(physical_saving * 100.0, 1.0e-12),
            }
        )
    rows.sort(key=lambda row: (row["nll_per_1pct_physical_saving"], row["layer_group"]))
    with (output_dir / "layer_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "groups": rows,
        "recommended_low_amplification_order": [row["layer_group"] for row in rows],
        "note": (
            "NLL per one percentage point of physical GQA saving is an empirical "
            "cross-layer amplification score, not a formal Lipschitz bound."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
