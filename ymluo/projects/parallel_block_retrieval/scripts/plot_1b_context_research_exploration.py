from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "doc" / "1b_context_search_research_exploration"
EVIDENCE_DIR = REPORT_DIR / "evidence"
ASSET_DIR = REPORT_DIR / "assets"


def load_json(name: str) -> dict:
    with (EVIDENCE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_kmean_failure() -> None:
    summary = load_json("musique_kmean_retrieval_summary.json")
    rows = {
        row["step_type"]: row
        for row in summary["summaries"]
        if row["split"] == "test"
    }

    def best(row: dict, budget: int) -> float:
        suffix = f"_recall_at_{budget}"
        return max(float(value) for key, value in row.items() if key.endswith(suffix))

    labels = ["Bridge block", "Answer block"]
    ordered = [rows["resolve_bridge"], rows["resolve_answer_from_bridge"]]
    recall16 = [100.0 * best(row, 16) for row in ordered]
    recall512 = [100.0 * best(row, 512) for row in ordered]
    x = np.arange(len(labels))
    width = 0.34

    plt.figure(figsize=(6.4, 3.8))
    plt.bar(x - width / 2, recall16, width, label="Best Recall@16")
    plt.bar(x + width / 2, recall512, width, label="Best Recall@512")
    plt.xticks(x, labels)
    plt.ylabel("Gold evidence recall (%)")
    plt.ylim(0, max(recall512) * 1.25)
    plt.title("Direct Q-to-K-mean retrieval does not locate evidence")
    plt.legend(frameon=False)
    for index, value in enumerate(recall16):
        plt.text(index - width / 2, value + 0.3, f"{value:.1f}", ha="center")
    for index, value in enumerate(recall512):
        plt.text(index + width / 2, value + 0.3, f"{value:.1f}", ha="center")
    save_figure("kmean_global_retrieval_failure.png")


def plot_low_rank() -> None:
    summary = load_json("longbench_recordcontext_subspace.json")
    profiles = summary["profiles"]
    labels = [f"L{row['layer']}/H{row['query_head']}" for row in profiles]
    rank8 = [100.0 * row["residual_energy_rank8"]["mean"] for row in profiles]
    rank16 = [100.0 * row["residual_energy_rank16"]["mean"] for row in profiles]
    rank90 = [row["residual_rank90"]["mean"] for row in profiles]
    x = np.arange(len(labels))
    width = 0.34

    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.bar(x - width / 2, rank8, width, label="Rank-8 energy")
    axis.bar(x + width / 2, rank16, width, label="Rank-16 energy")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Residual spectral energy retained (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Block residual K is medium-low rank in full-record prefill")
    axis.legend(frameon=False, loc="lower right")
    for index, value in enumerate(rank90):
        axis.text(index, 99, f"r90={value:.1f}", ha="center", va="top", fontsize=8)
    save_figure("residual_k_low_rank.png")


def plot_record_locality() -> None:
    summary = load_json("longbench_record_locality.json")
    rows = summary["datasets"]
    labels = [row["dataset"] for row in rows]
    adjacent = [row["mean_adjacent_cosine_across_profiles"] for row in rows]
    cross_record = [row["mean_different_record_random_cosine_across_profiles"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9.2, 4.2))
    plt.bar(x - width / 2, adjacent, width, label="Adjacent blocks in one record")
    plt.bar(x + width / 2, cross_record, width, label="Random blocks across records")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Global-centered K centroid cosine")
    plt.ylim(0, 0.9)
    plt.title("Real records contain measurable local K continuity")
    plt.legend(frameon=False)
    save_figure("record_locality.png")


def plot_hierarchy_tradeoff() -> None:
    series = [
        (
            "LongBench full-record K",
            "longbench_recordcontext_two_level_search.json",
        ),
        (
            "LongBench block-local K",
            "longbench_blocklocal_two_level_search.json",
        ),
        ("Shuffled MuSiQue control", "musique_two_level_search.json"),
    ]
    matched_configs = {
        (0.05, 0.005),
        (0.10, 0.010),
        (0.20, 0.020),
        (0.50, 0.050),
    }
    plt.figure(figsize=(7.2, 4.5))
    for label, filename in series:
        rows = [
            row
            for row in load_json(filename)["experiments"]
            if (
                row["parent_fraction"],
                row["requested_block_scan_fraction"],
            )
            in matched_configs
        ]
        speedup = [row["estimated_dot_product_speedup"] for row in rows]
        recall = [100.0 * row["mean_exact_neighbor_recall"] for row in rows]
        order = np.argsort(speedup)
        plt.plot(
            np.asarray(speedup)[order],
            np.asarray(recall)[order],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=label,
        )
    plt.xlabel("Estimated dot-product reduction (x)")
    plt.ylabel("Exact centroid Top-10 neighbor recall (%)")
    plt.title("Two-level search works only when position groups are coherent")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    save_figure("hierarchy_speed_recall_tradeoff.png")


def plot_fps_failure() -> None:
    summary = load_json("musique_fps16_summary.json")
    rows = {
        row["step_type"]: row
        for row in summary["summaries"]
        if row["split"] == "test"
    }
    budgets = summary["prototype_budgets"]
    labels = [
        ("Bridge query", rows["resolve_bridge"]),
        ("Answer query", rows["resolve_answer_from_bridge"]),
    ]

    plt.figure(figsize=(6.6, 4.0))
    for label, row in labels:
        values = [100.0 * row[f"fps{budget}_exact_top1_agreement"] for budget in budgets]
        plt.plot(budgets, values, marker="o", label=label)
    plt.xticks(budgets)
    plt.xlabel("Real K prototypes retained per block")
    plt.ylabel("Agreement with exact max-QK Top-1 (%)")
    plt.ylim(0, 30)
    plt.title("Unsupervised farthest-point prototypes miss max-attention")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    save_figure("fps_max_qk_failure.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_kmean_failure()
    plot_low_rank()
    plot_record_locality()
    plot_hierarchy_tradeoff()
    plot_fps_failure()


if __name__ == "__main__":
    main()
