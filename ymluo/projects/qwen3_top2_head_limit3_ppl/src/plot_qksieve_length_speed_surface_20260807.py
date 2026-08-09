#!/usr/bin/env python
"""Create paper-ready plots for QKSieve lifecycle speed measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTEXT_LABELS = {
    8192: "8K",
    16384: "16K",
    32768: "32K",
    65536: "64K",
    131072: "128K",
}
GENERATED_TOKENS = (8, 16, 32, 64, 128, 256, 512)


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(
            output / f"{name}.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_latency(rows: list[dict], output: Path) -> None:
    context = np.array([row["context_tokens"] for row in rows])
    full = np.array([row["full_step_ms_median"] for row in rows])
    sparse = np.array([row["sparse_step_ms_median"] for row in rows])
    speedup = np.array([row["steady_speedup_median"] for row in rows])
    full_low = np.array([row["full_step_ms_p05"] for row in rows])
    full_high = np.array([row["full_step_ms_p95"] for row in rows])
    sparse_low = np.array([row["sparse_step_ms_p05"] for row in rows])
    sparse_high = np.array([row["sparse_step_ms_p95"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))
    left, right = axes
    left.plot(context, full, "o-", color="#2f3e46", label="Full attention")
    left.fill_between(context, full_low, full_high, color="#2f3e46", alpha=0.14)
    left.plot(context, sparse, "s-", color="#c24135", label="QKSieve")
    left.fill_between(context, sparse_low, sparse_high, color="#c24135", alpha=0.14)
    left.set_xscale("log", base=2)
    left.set_xticks(context, [CONTEXT_LABELS[int(item)] for item in context])
    left.set_xlabel("Cached context length")
    left.set_ylabel("Decode latency (ms/token)")
    left.grid(axis="y", color="#d4d4d4", linewidth=0.6)
    left.legend(frameon=False, fontsize=8)

    right.plot(context, speedup, "o-", color="#0f766e", linewidth=2)
    right.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    right.fill_between(context, 0.0, 1.0, color="#c24135", alpha=0.07)
    right.set_xscale("log", base=2)
    right.set_xticks(context, [CONTEXT_LABELS[int(item)] for item in context])
    right.set_ylim(0.45, max(3.05, float(speedup.max()) + 0.2))
    right.set_xlabel("Cached context length")
    right.set_ylabel("Steady-state speedup")
    right.grid(axis="y", color="#d4d4d4", linewidth=0.6)
    for x, y in zip(context, speedup):
        right.text(x, y + 0.08, f"{y:.2f}x", ha="center", fontsize=7)
    figure.tight_layout()
    save_figure(figure, output, "decode_latency_and_speedup")


def plot_surface(rows: list[dict], output: Path, lifecycle: str) -> None:
    matrix = np.array(
        [
            [row[f"{lifecycle}_g{generated}_median"] for generated in GENERATED_TOKENS]
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.35))
    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=0.25,
        vmax=3.0,
        aspect="auto",
    )
    axis.set_xticks(range(len(GENERATED_TOKENS)), GENERATED_TOKENS)
    axis.set_yticks(
        range(len(rows)),
        [CONTEXT_LABELS[int(row["context_tokens"])] for row in rows],
    )
    axis.set_xlabel("Generated tokens per query")
    axis.set_ylabel("Cached context length")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value < 0.55 or value > 2.35 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}x",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Total decode speedup")
    figure.tight_layout()
    save_figure(figure, output, f"{lifecycle}_speed_surface")


def plot_break_even(rows: list[dict], output: Path) -> None:
    context = np.array([row["context_tokens"] for row in rows])
    warm = np.array([row["warm_break_even_tokens_median"] for row in rows])
    cold = np.array([row["cold_break_even_tokens_median"] for row in rows])
    finite = np.concatenate((warm[np.isfinite(warm)], cold[np.isfinite(cold)]))
    ceiling = max(32.0, float(finite.max()) * 1.15)
    warm_plot = np.where(np.isfinite(warm), warm, np.nan)
    cold_plot = np.where(np.isfinite(cold), cold, np.nan)

    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    axis.plot(context, warm_plot, "o-", color="#0f766e", label="Warm prefix")
    axis.plot(context, cold_plot, "s-", color="#c24135", label="Cold index")
    axis.set_xscale("log", base=2)
    axis.set_xticks(context, [CONTEXT_LABELS[int(item)] for item in context])
    axis.set_ylim(0.0, ceiling * 1.08)
    axis.set_xlabel("Cached context length")
    axis.set_ylabel("Break-even generated tokens")
    axis.grid(axis="y", color="#d4d4d4", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    for x, warm_value, cold_value in zip(context, warm, cold):
        if not np.isfinite(warm_value) and not np.isfinite(cold_value):
            axis.text(x, ceiling * 0.94, "none", ha="center", fontsize=8)
            continue
        if np.isfinite(warm_value):
            axis.text(x, warm_value + 0.8, f"{warm_value:.1f}", ha="center", fontsize=7)
        if np.isfinite(cold_value):
            axis.text(x, cold_value + 0.8, f"{cold_value:.1f}", ha="center", fontsize=7)
    figure.tight_layout()
    save_figure(figure, output, "break_even_tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.surface_json.read_text(encoding="utf-8"))
    rows = payload["summaries"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_latency(rows, args.output_dir)
    plot_surface(rows, args.output_dir, "warm")
    plot_surface(rows, args.output_dir, "cold")
    plot_break_even(rows, args.output_dir)


if __name__ == "__main__":
    main()
