from __future__ import annotations

import argparse
import csv
import re
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


def parse_mode(mode: str) -> tuple[int | None, float | None]:
    match = re.fullmatch(r"top2limit3protects(\d+)r([0-9]+(?:p[0-9]+)?)", mode)
    if not match:
        return None, None
    return int(match.group(1)), float(match.group(2).replace("p", "."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine sink/recent protection summaries.")
    parser.add_argument("--summary_csv", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    by_mode: dict[str, dict[str, Any]] = {}
    for csv_path in args.summary_csv:
        for row in read_csv(Path(csv_path)):
            mode = row["mode"]
            if mode in by_mode:
                continue
            sink, recent_percent = parse_mode(mode)
            merged: dict[str, Any] = dict(row)
            merged["sink_tokens"] = "" if sink is None else sink
            merged["recent_percent"] = "" if recent_percent is None else recent_percent
            by_mode[mode] = merged

    rows = list(by_mode.values())
    rows.sort(key=lambda row: (float(row["ppl"]), row["mode"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "ppl",
        "loss",
        "ppl_ratio_to_top2",
        "ppl_ratio_to_baseline",
        "limit_strategy",
        "protected_sink_tokens",
        "protected_recent_fraction",
        "sink_tokens",
        "recent_percent",
        "mean_kept_fraction",
        "mean_removed_per_query",
    ]
    write_csv(output_dir / "combined_protect_sink_recent_summary.csv", rows, fields)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top2_ppl = float(next(row["ppl"] for row in rows if row["mode"] == "top2"))
    gap_row = next((row for row in rows if row["mode"] == "top2limit3gap8p0"), None)
    gap_ppl = float(gap_row["ppl"]) if gap_row is not None else None

    protect_rows = [row for row in rows if row["mode"].startswith("top2limit3protects")]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.plot_dpi)
    for sink in sorted({int(row["sink_tokens"]) for row in protect_rows if row["sink_tokens"] != ""}):
        series = [row for row in protect_rows if row["sink_tokens"] == sink]
        series.sort(key=lambda row: float(row["recent_percent"]))
        ax.plot(
            [float(row["recent_percent"]) for row in series],
            [float(row["ppl"]) for row in series],
            marker="o",
            label=f"sink {sink}",
        )
    ax.axhline(top2_ppl, color="black", linestyle="--", linewidth=1.2, label=f"top2 {top2_ppl:.2f}")
    if gap_ppl is not None:
        ax.axhline(gap_ppl, color="#666666", linestyle=":", linewidth=1.2, label=f"gap8 {gap_ppl:.2f}")
    ax.set_yscale("log")
    ax.set_title("Effect of protecting recent tokens before top3-head limiting")
    ax.set_xlabel("Protected recent historical tokens (%)")
    ax.set_ylabel("Perplexity on evaluation tokens, log scale")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recent_percent_threshold_ppl_logy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.plot_dpi)
    scatter_rows = [row for row in rows if row["mean_kept_fraction"] != ""]
    for row in scatter_rows:
        kept = float(row["mean_kept_fraction"])
        ppl = float(row["ppl"])
        label = row["mode"].replace("top2limit3", "")
        ax.scatter(kept, ppl, s=60)
        ax.text(kept, ppl, label, fontsize=7, ha="left", va="bottom")
    ax.axhline(top2_ppl, color="black", linestyle="--", linewidth=1.2, label=f"top2 {top2_ppl:.2f}")
    if gap_ppl is not None:
        ax.axhline(gap_ppl, color="#666666", linestyle=":", linewidth=1.2, label=f"gap8 {gap_ppl:.2f}")
    ax.set_yscale("log")
    ax.set_title("PPL versus retained original top2-link fraction")
    ax.set_xlabel("Mean final kept / original top2 kept")
    ax.set_ylabel("Perplexity on evaluation tokens, log scale")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "combined_kept_fraction_vs_ppl_logy.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
