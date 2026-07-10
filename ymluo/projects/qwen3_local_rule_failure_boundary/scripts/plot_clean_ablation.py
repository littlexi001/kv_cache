from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIM_ORDER = ["low", "high", "conflict"]
COUNT_ORDER = [0, 4, 16, 64]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def bucket(rows: Iterable[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return out


def cand_acc(rows: list[dict[str, str]]) -> float:
    return mean(int(float(row["candidate_correct"])) for row in rows)


def gen_acc(rows: list[dict[str, str]]) -> float:
    return mean(int(float(row["generation_correct"])) for row in rows)


def margin(rows: list[dict[str, str]]) -> float:
    return mean(float(row["candidate_margin"]) for row in rows)


def selectivity(rows: list[dict[str, str]]) -> float:
    vals = [
        float(row["rule_attention_selectivity"])
        for row in rows
        if row.get("rule_attention_selectivity", "") not in {"", "nan"}
    ]
    return mean(vals)


def setup_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_clean_length(q06: list[dict[str, str]], q8: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for rows, label, marker in [(q06, "Qwen3-0.6B", "o"), (q8, "Qwen3-8B", "s")]:
        groups = bucket(rows, ["target_context_tokens"])
        lengths = sorted(int(k[0]) for k in groups)
        accs = [cand_acc(groups[(str(length),)]) for length in lengths]
        margins = [margin(groups[(str(length),)]) for length in lengths]
        axes[0].plot(lengths, accs, marker=marker, label=label)
        axes[1].plot(lengths, margins, marker=marker, label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.legend(frameon=False)
    setup_ax(axes[0], "Clean length: candidate accuracy", "accuracy")
    setup_ax(axes[1], "Clean length: gold margin", "mean margin")
    axes[0].set_ylim(0, 1.05)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_gap(rows: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    groups = bucket(rows, ["target_context_tokens", "rule_gap_tokens"])
    for length in sorted({int(row["target_context_tokens"]) for row in rows}):
        gaps = sorted({int(row["rule_gap_tokens"]) for row in rows if int(row["target_context_tokens"]) == length})
        accs = [cand_acc(groups[(str(length), str(gap))]) for gap in gaps]
        margins = [margin(groups[(str(length), str(gap))]) for gap in gaps]
        axes[0].plot(gaps, accs, marker="o", label=f"{length // 1024}k")
        axes[1].plot(gaps, margins, marker="o", label=f"{length // 1024}k")
    for ax in axes:
        ax.set_xlabel("requested rule gap (tokens)")
        ax.legend(frameon=False)
    setup_ax(axes[0], "Clean gap: candidate accuracy", "accuracy")
    setup_ax(axes[1], "Clean gap: gold margin", "mean margin")
    axes[0].set_ylim(0, 1.05)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def heat_values(rows: list[dict[str, str]], length: int, competitor: int) -> list[list[float]]:
    groups = bucket(
        [
            row
            for row in rows
            if int(row["target_context_tokens"]) == length and int(row["competitor_count"]) == competitor
        ],
        ["distractor_similarity", "distractor_count"],
    )
    matrix: list[list[float]] = []
    for sim in SIM_ORDER:
        matrix.append([cand_acc(groups.get((sim, str(count)), [])) for count in COUNT_ORDER])
    return matrix


def plot_q06_interference_heatmap(rows: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row_idx, length in enumerate([8192, 32768]):
        for col_idx, comp in enumerate([0, 4]):
            ax = axes[row_idx][col_idx]
            data = heat_values(rows, length, comp)
            im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"0.6B {length // 1024}k, competitors={comp}", fontsize=10)
            ax.set_xticks(range(len(COUNT_ORDER)), [str(v) for v in COUNT_ORDER])
            ax.set_yticks(range(len(SIM_ORDER)), SIM_ORDER)
            ax.set_xlabel("distractor count")
            if col_idx == 0:
                ax.set_ylabel("similarity")
            for y, vals in enumerate(data):
                for x, value in enumerate(vals):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="candidate accuracy")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_model_size_interference(q06: list[dict[str, str]], q8: list[dict[str, str]], out: Path) -> None:
    rows_by_model = {
        "0.6B": [
            row
            for row in q06
            if int(row["target_context_tokens"]) == 8192 and int(row["seed"]) in {0, 1, 2}
        ],
        "8B": q8,
    }
    labels = []
    values_by_model: dict[str, list[float]] = {"0.6B": [], "8B": []}
    for comp in [0, 4]:
        for sim in SIM_ORDER:
            labels.append(f"c{comp}-{sim}")
            for model, model_rows in rows_by_model.items():
                groups = bucket(
                    [
                        row
                        for row in model_rows
                        if int(row["competitor_count"]) == comp and row["distractor_similarity"] == sim
                    ],
                    ["competitor_count", "distractor_similarity"],
                )
                values_by_model[model].append(cand_acc(next(iter(groups.values()), [])))
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar([v - width / 2 for v in x], values_by_model["0.6B"], width=width, label="Qwen3-0.6B")
    ax.bar([v + width / 2 for v in x], values_by_model["8B"], width=width, label="Qwen3-8B")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    setup_ax(ax, "8k clean-interference: model size comparison", "candidate accuracy")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_attention_selectivity(rows: list[dict[str, str]], out: Path) -> None:
    rows = [row for row in rows if int(row["target_context_tokens"]) == 8192]
    groups = bucket(rows, ["competitor_count", "distractor_similarity"])
    labels = []
    values = []
    for comp in [0, 4]:
        for sim in SIM_ORDER:
            labels.append(f"c{comp}-{sim}")
            values.append(selectivity(groups[(str(comp), sim)]))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"])
    ax.set_ylim(0, max(0.5, max(values) * 1.2))
    setup_ax(ax, "0.6B 8k: attention selectivity under interference", "gold-rule selectivity")
    ax.set_xlabel("competitors-similarity")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clean local-rule ablation figures.")
    parser.add_argument("--project", type=Path, default=Path("."))
    args = parser.parse_args()
    project = args.project
    outputs = project / "outputs"
    fig_dir = project / "figures" / "clean_ablation_20260710"
    fig_dir.mkdir(parents=True, exist_ok=True)

    q06_length = read_csv(outputs / "clean_length_qwen06_20260710" / "results.csv")
    q06_gap = read_csv(outputs / "clean_gap_qwen06_20260710" / "results.csv")
    q06_interference = read_csv(outputs / "clean_interference_qwen06_20260710" / "results.csv")
    q8_length = read_csv(outputs / "clean_length_qwen8b_20260710" / "results.csv")
    q8_interference = read_csv(outputs / "clean_interference_qwen8b_20260710" / "results.csv")

    plot_clean_length(q06_length, q8_length, fig_dir / "clean_length_acc_margin.png")
    plot_gap(q06_gap, fig_dir / "clean_gap_acc_margin.png")
    plot_q06_interference_heatmap(q06_interference, fig_dir / "qwen06_interference_heatmap.png")
    plot_model_size_interference(q06_interference, q8_interference, fig_dir / "model_size_interference_8k.png")
    plot_attention_selectivity(q06_interference, fig_dir / "qwen06_attention_selectivity_8k.png")
    print(fig_dir)


if __name__ == "__main__":
    main()

