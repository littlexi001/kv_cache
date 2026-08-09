from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "top2_recall_mean",
    "selected_attention_mass_mean",
    "top2_attention_mass_recall_mean",
    "score_pearson_mean",
    "score_rmse_mean",
    "index_ratio_of_full_kv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine QK-balanced versus Key-PCA trace analyses."
    )
    parser.add_argument(
        "--input_dir",
        action="append",
        type=Path,
        required=True,
        help="May be repeated. Each directory is recursively searched.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = sorted(
        {
            path.resolve()
            for root in args.input_dir
            for path in root.rglob("summary.json")
        }
    )
    per_trace = []
    spectral = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "methods" not in payload or "spectrum" not in payload:
            continue
        label = str(payload["config"]["label"])
        spectral.append(
            {
                "label": label,
                "key_pca_top16_energy_mean": payload["spectrum"][
                    "key_pca_top16_energy"
                ]["mean"],
                "qk_top16_score_energy_mean": payload["spectrum"][
                    "qk_top16_score_energy"
                ]["mean"],
                "qk_top48_score_energy_mean": payload["spectrum"][
                    "qk_top48_score_energy"
                ]["mean"],
            }
        )
        for row in payload["methods"]:
            per_trace.append({"label": label, **row})
    if not per_trace:
        raise ValueError("no compatible summary.json files found")

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_trace:
        grouped[
            (str(row["method"]), float(row["selected_fraction_target"]))
        ].append(row)
    weighted = []
    for (method, fraction), rows in sorted(grouped.items()):
        cases = sum(int(row["cases"]) for row in rows)
        output: dict[str, Any] = {
            "method": method,
            "selected_fraction_target": fraction,
            "traces": len(rows),
            "cases": cases,
        }
        for metric in METRICS:
            output[metric] = sum(
                float(row[metric]) * int(row["cases"]) for row in rows
            ) / cases
        weighted.append(output)

    paired_improvements = []
    lookup = {
        (str(row["method"]), float(row["selected_fraction_target"])): row
        for row in weighted
    }
    fractions = sorted(
        {
            float(row["selected_fraction_target"])
            for row in weighted
            if row["method"] == "qk_balanced"
        }
    )
    for fraction in fractions:
        key = lookup[("key_pca", fraction)]
        balanced = lookup[("qk_balanced", fraction)]
        output = {"selected_fraction_target": fraction}
        for metric in METRICS:
            output[f"key_pca_{metric}"] = key[metric]
            output[f"qk_balanced_{metric}"] = balanced[metric]
            output[f"qk_minus_key_{metric}"] = (
                balanced[metric] - key[metric]
            )
        paired_improvements.append(output)

    summary = {
        "inputs": [str(path) for path in paths],
        "trace_count": len(spectral),
        "weighted_methods": weighted,
        "paired_improvements": paired_improvements,
        "spectral_concentration": {
            "key_pca_top16_energy_mean": sum(
                float(row["key_pca_top16_energy_mean"]) for row in spectral
            )
            / len(spectral),
            "qk_top16_score_energy_mean": sum(
                float(row["qk_top16_score_energy_mean"]) for row in spectral
            )
            / len(spectral),
            "qk_top48_score_energy_mean": sum(
                float(row["qk_top48_score_energy_mean"]) for row in spectral
            )
            / len(spectral),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_trace.csv", per_trace)
    write_csv(args.output_dir / "spectral.csv", spectral)
    write_csv(args.output_dir / "weighted_methods.csv", weighted)
    write_csv(args.output_dir / "paired_improvements.csv", paired_improvements)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
