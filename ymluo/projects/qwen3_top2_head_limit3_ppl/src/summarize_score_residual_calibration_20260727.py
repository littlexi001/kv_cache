from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


BASE_METHOD = "qk_balanced_metric_scale"
VARIANTS = (
    "qk_balanced_metric_scale_bias_fp16",
    "qk_balanced_metric_scale_bias_int4",
    "qk_balanced_metric_scale_affine_int8",
    "qk_balanced_metric_scale_bias_eb_int4",
)
METRICS = (
    "top2_recall",
    "selected_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
)
PAIR_FIELDS = (
    "label",
    "layer",
    "heldout_step",
    "kv_head",
    "query_head",
    "selected_fraction_target",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize held-out score-residual calibration across traces."
        )
    )
    parser.add_argument(
        "--input_roots",
        required=True,
        help="Comma-separated roots containing trace/per_head.csv files.",
    )
    parser.add_argument("--selected_fraction", type=float, default=0.01)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_rows(
    roots: list[Path],
    selected_fraction: float,
) -> dict[tuple[str, ...], dict[str, dict[str, float]]]:
    paired: dict[
        tuple[str, ...],
        dict[str, dict[str, float]],
    ] = defaultdict(dict)
    methods = {BASE_METHOD, *VARIANTS}
    paths: list[Path] = []
    for root in roots:
        paths.extend(sorted(root.glob("*/per_head.csv")))
    if not paths:
        raise FileNotFoundError("no trace/per_head.csv files found")
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["method"]
                if method not in methods:
                    continue
                if (
                    abs(
                        float(row["selected_fraction_target"])
                        - selected_fraction
                    )
                    > 1.0e-9
                ):
                    continue
                key = tuple(row[field] for field in PAIR_FIELDS)
                paired[key][method] = {
                    metric: float(row[metric]) for metric in METRICS
                }
    return paired


def aggregate_pairs(
    paired: dict[tuple[str, ...], dict[str, dict[str, float]]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    complete = {
        key: methods
        for key, methods in paired.items()
        if BASE_METHOD in methods
        and all(variant in methods for variant in VARIANTS)
    }
    if not complete:
        raise RuntimeError("no strict complete method pairs")
    labels = sorted({key[0] for key in complete})
    per_label: list[dict[str, object]] = []
    overall: dict[str, object] = {}
    block_values: dict[
        tuple[str, str, str, str],
        list[float],
    ] = defaultdict(list)

    for variant in VARIANTS:
        overall_pairs = [
            (methods[BASE_METHOD], methods[variant])
            for methods in complete.values()
        ]
        variant_summary: dict[str, object] = {
            "strict_pair_count": len(overall_pairs),
        }
        for metric in METRICS:
            base_mean = mean(pair[0][metric] for pair in overall_pairs)
            variant_mean = mean(pair[1][metric] for pair in overall_pairs)
            variant_summary[metric] = {
                "base_mean": base_mean,
                "variant_mean": variant_mean,
                "variant_minus_base": variant_mean - base_mean,
            }
        overall[variant] = variant_summary

        for label in labels:
            label_pairs = [
                (methods[BASE_METHOD], methods[variant])
                for key, methods in complete.items()
                if key[0] == label
            ]
            row: dict[str, object] = {
                "label": label,
                "variant": variant,
                "strict_pair_count": len(label_pairs),
            }
            for metric in METRICS:
                base_mean = mean(pair[0][metric] for pair in label_pairs)
                variant_mean = mean(pair[1][metric] for pair in label_pairs)
                row[f"{metric}_base"] = base_mean
                row[f"{metric}_variant"] = variant_mean
                row[f"{metric}_delta"] = variant_mean - base_mean
            per_label.append(row)

        for key, methods in complete.items():
            block_key = (key[0], key[1], key[2], variant)
            block_values[block_key].append(
                methods[variant]["top2_recall"]
                - methods[BASE_METHOD]["top2_recall"]
            )

    paired_blocks = []
    for (label, layer, heldout_step, variant), values in sorted(
        block_values.items()
    ):
        paired_blocks.append(
            {
                "label": label,
                "layer": int(layer),
                "heldout_step": int(heldout_step),
                "variant": variant,
                "head_pair_count": len(values),
                "mean_top2_recall_delta": mean(values),
            }
        )

    for variant in VARIANTS:
        deltas = [
            float(row["mean_top2_recall_delta"])
            for row in paired_blocks
            if row["variant"] == variant
        ]
        summary = overall[variant]
        assert isinstance(summary, dict)
        summary["paired_block_count"] = len(deltas)
        summary["positive_block_fraction"] = mean(
            1.0 if value > 0.0 else 0.0 for value in deltas
        )
        summary["nonnegative_block_fraction"] = mean(
            1.0 if value >= 0.0 else 0.0 for value in deltas
        )

    summary = {
        "base_method": BASE_METHOD,
        "variants": list(VARIANTS),
        "strict_complete_pair_count": len(complete),
        "label_count": len(labels),
        "labels": labels,
        "overall": overall,
    }
    return summary, per_label, paired_blocks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.input_roots.split(",") if item]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = read_rows(roots, args.selected_fraction)
    summary, per_label, paired_blocks = aggregate_pairs(paired)
    summary["config"] = {
        "input_roots": [str(root) for root in roots],
        "selected_fraction": args.selected_fraction,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "per_label.csv", per_label)
    write_csv(output_dir / "paired_blocks.csv", paired_blocks)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
