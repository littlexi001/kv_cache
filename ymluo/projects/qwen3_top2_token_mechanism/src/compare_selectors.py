from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def block_bootstrap_ci(
    values: list[float],
    block_size: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    blocks = [values[start : start + block_size] for start in range(0, len(values), block_size)]
    if repetitions <= 0 or len(blocks) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repetitions):
        selected = [blocks[rng.randrange(len(blocks))] for _ in range(len(blocks))]
        flattened_sum = sum(sum(block) for block in selected)
        flattened_count = sum(len(block) for block in selected)
        samples.append(flattened_sum / flattened_count)
    return mean, percentile(samples, 0.025), percentile(samples, 0.975)


def group_nll(rows: list[dict[str, str]]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], {})[int(row["token_index"])] = float(row["nll"])
    return grouped


def paired_delta(left: dict[int, float], right: dict[int, float]) -> list[float]:
    common = sorted(set(left) & set(right))
    return [left[index] - right[index] for index in common]


def find_reference_mode(ppl_rows: list[dict[str, str]], target_ratio: float) -> str:
    matches = [
        row["mode"]
        for row in ppl_rows
        if row["selector"] == "top_attention"
        and row.get("ratio", "") != ""
        and math.isclose(float(row["ratio"]), target_ratio)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Top-attention reference at ratio={target_ratio}, got {matches}")
    return matches[0]


def make_plot(output_dir: Path, rows: list[dict[str, Any]], dpi: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["mode"]) for row in rows]
    means = [float(row["delta_nll_vs_top2_mean"]) for row in rows]
    lower = [float(row["delta_nll_vs_top2_ci_low"]) for row in rows]
    upper = [float(row["delta_nll_vs_top2_ci_high"]) for row in rows]
    errors = [[mean - low for mean, low in zip(means, lower)], [high - mean for mean, high in zip(means, upper)]]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.55), 5), dpi=dpi)
    ax.errorbar(range(len(rows)), means, yerr=errors, fmt="o", capsize=3, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Paired delta NLL vs Top-2% (nats/token)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "paired_delta_nll_vs_top2.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of Top-2% and equal-budget controls.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--target_ratio", type=float, default=0.02)
    parser.add_argument("--reference_mode", default="")
    parser.add_argument("--equivalence_nll_margin", type=float, default=0.01)
    parser.add_argument("--equivalence_ppl_relative_margin", type=float, default=0.01)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--make_plot", action="store_true")
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    ppl_rows = read_csv(run_dir / "ppl_by_selector.csv")
    nll_rows = read_csv(run_dir / "token_nll_by_selector.csv")
    nll_by_mode = group_nll(nll_rows)
    ppl_by_mode = {row["mode"]: row for row in ppl_rows}
    full_mode = "full_attention"
    reference_mode = args.reference_mode or find_reference_mode(ppl_rows, args.target_ratio)
    if full_mode not in nll_by_mode or reference_mode not in nll_by_mode:
        raise ValueError("Full-attention or Top-attention reference NLL rows are missing.")

    output_rows: list[dict[str, Any]] = []
    for mode, values in nll_by_mode.items():
        delta_full = paired_delta(values, nll_by_mode[full_mode])
        delta_top2 = paired_delta(values, nll_by_mode[reference_mode])
        mean_full, low_full, high_full = block_bootstrap_ci(
            delta_full, args.block_size, args.bootstrap_repetitions, args.seed
        )
        mean_top2, low_top2, high_top2 = block_bootstrap_ci(
            delta_top2, args.block_size, args.bootstrap_repetitions, args.seed + 1
        )
        ppl = float(ppl_by_mode[mode]["ppl"])
        full_ppl = float(ppl_by_mode[full_mode]["ppl"])
        top2_ppl = float(ppl_by_mode[reference_mode]["ppl"])
        within_ppl_margin = abs(ppl / top2_ppl - 1.0) <= args.equivalence_ppl_relative_margin
        equivalence_established = (
            low_top2 >= -args.equivalence_nll_margin
            and high_top2 <= args.equivalence_nll_margin
        )
        output_rows.append(
            {
                "mode": mode,
                "selector": ppl_by_mode[mode]["selector"],
                "sink_tokens": ppl_by_mode[mode].get("sink_tokens", ""),
                "ratio": ppl_by_mode[mode].get("ratio", ""),
                "token_count": len(delta_top2),
                "ppl": ppl,
                "ppl_ratio_vs_full": ppl / full_ppl,
                "ppl_ratio_vs_top2": ppl / top2_ppl,
                "delta_nll_vs_full_mean": mean_full,
                "delta_nll_vs_full_ci_low": low_full,
                "delta_nll_vs_full_ci_high": high_full,
                "delta_nll_vs_top2_mean": mean_top2,
                "delta_nll_vs_top2_ci_low": low_top2,
                "delta_nll_vs_top2_ci_high": high_top2,
                "within_top2_relative_ppl_margin": within_ppl_margin,
                "top2_nll_equivalence_established": equivalence_established,
            }
        )

    output_rows.sort(key=lambda row: float(row["ppl"]))
    output_path = output_dir / "paired_selector_comparison.csv"
    write_csv(output_path, output_rows, list(output_rows[0]))
    curve_rows = [
        row
        for row in ppl_rows
        if row["selector"] == "top_attention" and row.get("ratio", "") != ""
    ]
    best_curve = min(curve_rows, key=lambda row: float(row["ppl"]))
    sink_recent_rows = [row for row in output_rows if row["selector"] in {"sink_recent", "recent"}]
    best_sink_recent = min(sink_recent_rows, key=lambda row: float(row["ppl"]))
    plot_path = make_plot(output_dir, output_rows, args.plot_dpi) if args.make_plot else None
    summary = {
        "reference_mode": reference_mode,
        "equivalence_rule": {
            "nll_margin_nats_per_token": args.equivalence_nll_margin,
            "decision": "95% block-bootstrap CI for candidate minus Top-2% must lie inside +/- margin",
            "relative_ppl_margin_descriptive": args.equivalence_ppl_relative_margin,
        },
        "best_top_attention_curve_point": best_curve,
        "best_sink_recent_control": best_sink_recent,
        "sink_recent_equivalent_to_top2": bool(best_sink_recent["top2_nll_equivalence_established"]),
        "paths": {
            "paired_selector_comparison": str(output_path),
            "paired_plot": plot_path,
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

