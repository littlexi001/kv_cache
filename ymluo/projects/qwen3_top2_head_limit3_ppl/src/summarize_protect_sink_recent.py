from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sink/recent protection sweep.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ppl_rows = read_csv(run_dir / "ppl_by_mode.csv")
    load_rows = read_csv(run_dir / "limit_load_by_head.csv")

    load_by_mode: dict[str, list[dict[str, str]]] = {}
    for row in load_rows:
        load_by_mode.setdefault(row["mode"], []).append(row)

    top2_ppl = next(float(row["ppl"]) for row in ppl_rows if row["mode"] == "top2")
    baseline_row = next((row for row in ppl_rows if row["mode"] == "baseline"), None)
    baseline_ppl = float(baseline_row["ppl"]) if baseline_row is not None else float("nan")
    summary_rows: list[dict[str, Any]] = []
    for row in ppl_rows:
        mode = row["mode"]
        group = load_by_mode.get(mode, [])
        summary_rows.append(
            {
                "mode": mode,
                "ppl": float(row["ppl"]),
                "loss": float(row["loss"]),
                "ppl_ratio_to_top2": float(row["ppl"]) / top2_ppl,
                "ppl_ratio_to_baseline": float(row["ppl"]) / baseline_ppl if math.isfinite(baseline_ppl) else "",
                "limit_strategy": row.get("limit_strategy", ""),
                "protected_sink_tokens": row.get("protected_sink_tokens", ""),
                "protected_recent_fraction": row.get("protected_recent_fraction", ""),
                "mean_kept_fraction": mean(
                    [float(item["kept_fraction_of_original_top2"]) for item in group]
                )
                if group
                else "",
                "mean_removed_per_query": mean([float(item["removed_per_query_mean"]) for item in group])
                if group
                else "",
            }
        )

    fields = [
        "mode",
        "ppl",
        "loss",
        "ppl_ratio_to_top2",
        "ppl_ratio_to_baseline",
        "limit_strategy",
        "protected_sink_tokens",
        "protected_recent_fraction",
        "mean_kept_fraction",
        "mean_removed_per_query",
    ]
    write_csv(output_dir / "protect_sink_recent_summary.csv", summary_rows, fields)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in summary_rows if row["mode"] != "baseline"]
    labels = [row["mode"] for row in plot_rows]
    ppls = [float(row["ppl"]) for row in plot_rows]
    colors = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
    ]

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=args.plot_dpi)
    bars = ax.bar(labels, ppls, color=[colors[index % len(colors)] for index in range(len(labels))])
    ax.axhline(top2_ppl, color="black", linestyle="--", linewidth=1.2, label=f"top2 PPL {top2_ppl:.2f}")
    ax.set_yscale("log")
    ax.set_title("PPL for sink/recent protected top3-head rules")
    ax.set_xlabel("Attention selection mode")
    ax.set_ylabel("Perplexity on evaluation tokens, log scale")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25, which="both")
    ax.legend()
    for bar, value in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "protect_rules_ppl_logy.png")
    plt.close(fig)

    scatter_rows = [row for row in summary_rows if row["mean_kept_fraction"] != ""]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.plot_dpi)
    for row in scatter_rows:
        kept = float(row["mean_kept_fraction"])
        ppl = float(row["ppl"])
        ax.scatter(kept, ppl, s=60)
        ax.text(kept, ppl, row["mode"].replace("top2limit3", ""), fontsize=8, ha="left", va="bottom")
    ax.axhline(top2_ppl, color="black", linestyle="--", linewidth=1.2, label=f"top2 PPL {top2_ppl:.2f}")
    ax.set_yscale("log")
    ax.set_title("PPL versus retained original top2-link fraction")
    ax.set_xlabel("Mean final kept / original top2 kept")
    ax.set_ylabel("Perplexity on evaluation tokens, log scale")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "protect_kept_fraction_vs_ppl_logy.png")
    plt.close(fig)

    metadata = {
        "run_dir": str(run_dir),
        "top2_ppl": top2_ppl,
        "baseline_ppl": baseline_ppl,
        "rows": len(summary_rows),
    }
    with (output_dir / "summary_metadata.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
