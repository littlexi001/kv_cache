from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen3_4b": "Qwen3-4B",
}
MODEL_COLORS = {
    "llama31_8b": "#0B6E4F",
    "qwen25_7b": "#C44E52",
    "qwen3_4b": "#4C72B0",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_prefix(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = payload["by_model_prefix"]
    for model in MODEL_LABELS:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: float(row["prefix_tokens"]),
        )
        x = [float(row["prefix_tokens"]) for row in selected]
        overlap = [float(row["subspace_overlap_mean"]) for row in selected]
        fidelity = [float(row["prefix_qk_fidelity_mean"]) for row in selected]
        color = MODEL_COLORS[model]
        ax.plot(
            x,
            overlap,
            marker="o",
            color=color,
            label=f"{MODEL_LABELS[model]} overlap",
        )
        ax.plot(
            x,
            fidelity,
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.78,
            label=f"{MODEL_LABELS[model]} QK fidelity",
        )
    ax.axvline(2048, color="#333333", linestyle=":", linewidth=1.2)
    ax.text(2160, 0.45, "frozen 2K", fontsize=8, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["0.5K", "1K", "2K", "4K", "8K"])
    ax.set_ylim(0.4, 0.82)
    ax.set_xlabel("Prefix tokens used to estimate PCA basis")
    ax.set_ylabel("Mean fraction")
    ax.set_title("(a) Prefix basis drift")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2, loc="lower right")


def plot_margin(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = {row["method"]: row for row in payload["overall"]}
    methods = [
        "prefix_pca48_fp32",
        "prefix_pca48_int4k",
        "prefix_pca48_int4k_int8q",
    ]
    labels = ["PCA48 FP32", "+ INT4 K", "+ INT8 Q"]
    set_recall = [float(rows[name]["topk_recall_mean"]) for name in methods]
    mass_recall = [
        float(rows[name]["mass_weighted_topk_recall_mean"])
        for name in methods
    ]
    x = list(range(len(methods)))
    width = 0.34
    ax.bar(
        [value - width / 2 for value in x],
        set_recall,
        width,
        label="Token-set recall",
        color="#8172B2",
    )
    ax.bar(
        [value + width / 2 for value in x],
        mass_recall,
        width,
        label="Attention-mass recall",
        color="#CCB974",
    )
    production = rows["prefix_pca48_int4k_int8q"]
    sampled_mass = float(production["sampled_mass_weighted_topk_recall_mean"])
    certified_mass = float(
        production[
            "sampled_norm_tokenwise_core_fraction_of_exact_top_mass_mean"
        ]
    )
    ax.axhline(
        sampled_mass,
        color="#0B6E4F",
        linestyle="--",
        linewidth=1.4,
        label="Production sampled mass recall",
    )
    ax.axhline(
        certified_mass,
        color="#C44E52",
        linestyle=":",
        linewidth=1.6,
        label="Strictly certified exact-top mass",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall relative to exact top-4%")
    ax.set_title("(b) Margin and mass preservation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7, loc="lower left")


def plot_cost(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = payload["measurements"]
    x = [float(row["history_tokens"]) / 1000.0 for row in rows]
    full = [float(row["full_steady_ms_per_token"]) for row in rows]
    sparse = [float(row["countcap_steady_ms_per_token"]) for row in rows]
    amortized = [float(row["countcap_ms_per_token"]) for row in rows]
    ax.plot(
        x,
        full,
        marker="o",
        color="#C44E52",
        label="Full KV steady",
    )
    ax.plot(
        x,
        sparse,
        marker="s",
        color="#0B6E4F",
        label="CountCap steady",
    )
    ax.plot(
        x,
        amortized,
        marker="^",
        linestyle="--",
        color="#4C72B0",
        label="CountCap, lazy index / 255 steps",
    )
    crossover = float(payload["predicted_decode_crossover_tokens"]) / 1000.0
    ax.axvline(crossover, color="#333333", linestyle=":", linewidth=1.2)
    ax.text(
        crossover + 0.5,
        69,
        f"steady fit: {crossover:.1f}K",
        fontsize=8,
        color="#333333",
    )
    ax.set_xlim(1.5, 33)
    ax.set_xlabel("History length (K tokens)")
    ax.set_ylabel("Model-forward latency (ms/token)")
    ax.set_title("(c) Steady and amortized cost")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_summary", type=Path, required=True)
    parser.add_argument("--margin_summary", type=Path, required=True)
    parser.add_argument("--cost_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.35))
    plot_prefix(axes[0], load_json(args.prefix_summary))
    plot_margin(axes[1], load_json(args.margin_summary))
    plot_cost(axes[2], load_json(args.cost_summary))
    figure.suptitle(
        "CountCap theory closure: prefix drift, ranking stability, and cost",
        fontsize=13,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
