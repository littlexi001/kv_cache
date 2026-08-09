from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


YMLUO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = YMLUO_ROOT / "projects" / "parallel_block_retrieval"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ASSET_ROOT = (
    YMLUO_ROOT
    / "doc"
    / "assets"
    / "technical_report_10m_iterative_kv_retrieval_20260713"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ASSET_ROOT / name, dpi=180, bbox_inches="tight")
    plt.close()


def load_system_results() -> tuple[dict, dict]:
    svd_dir = OUTPUT_ROOT / "report_svd32_strict_chain500_v1"
    first = load_json(svd_dir / "first_step_summary.json")["summaries"][0]
    bridge = load_json(svd_dir / "bridge_summary.json")
    second = load_json(svd_dir / "second_retrieval_summary.json")["summaries"][0]
    final = load_json(svd_dir / "final_selector_summary.json")
    svd = {
        "first": first["retrieval_target_span_recall_at_k"],
        "bridge": bridge["bridge_state_hit_rate"],
        "second": second["lexical_recall_at_16"],
        "oracle": final["oracle_any_branch_accuracy"],
        "final": final["yes_no_accuracy"],
    }
    rag = load_json(
        OUTPUT_ROOT / "rag_e5_hybrid_strict_chain500_v1" / "summary.json"
    )
    return svd, rag


def plot_pipeline_funnel() -> None:
    svd, rag = load_system_results()
    stages = [
        ("First Top-3", "first", "first_retrieval_top3_recall"),
        ("Bridge correct", "bridge", "bridge_accuracy"),
        ("Second Top-16", "second", "second_retrieval_top16_recall"),
        ("Candidate oracle", "oracle", "candidate_oracle_answer_accuracy"),
        ("Verifier final", "final", "verifier_final_answer_accuracy"),
    ]
    svd_values = [100.0 * svd[key] for _, key, _ in stages]
    rag_values = [100.0 * rag[key] for _, _, key in stages]
    x = np.arange(len(stages))
    width = 0.36

    plt.figure(figsize=(9.0, 4.6))
    plt.bar(x - width / 2, svd_values, width, label="SVD32 pipeline")
    plt.bar(x + width / 2, rag_values, width, label="Hybrid-RAG pipeline")
    plt.xticks(x, [label for label, _, _ in stages], rotation=15, ha="right")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 100)
    plt.title("Same reader and verifier: retriever quality propagates through the chain")
    plt.legend(frameon=False)
    for index, value in enumerate(svd_values):
        plt.text(index - width / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(rag_values):
        plt.text(index + width / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
    save("retrieval_pipeline_funnel.png")


def plot_quality_comparison() -> None:
    full = load_json(
        OUTPUT_ROOT / "musique_fullcontext_10k20k40k_test500_v3" / "summary.json"
    )
    _, rag = load_system_results()
    labels = ["Full 10K", "Full 20K", "Full 40K", "SVD32 10M", "Hybrid-RAG 10M"]
    values = [100.0 * row["answer_hit_rate"] for row in full["by_length"]]
    values.extend([36.4, 100.0 * rag["verifier_final_answer_accuracy"]])
    colors = ["#9aa0a6", "#9aa0a6", "#9aa0a6", "#2b6cb0", "#d97706"]

    plt.figure(figsize=(8.2, 4.5))
    bars = plt.bar(np.arange(len(labels)), values, color=colors)
    plt.xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("Strict Answer Hit (%)")
    plt.ylim(0, 50)
    plt.title("End-to-end quality on the same 500 MuSiQue test questions")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center")
    save("quality_vs_full_attention.png")


def plot_gpu_scaling() -> None:
    scaling = load_json(
        OUTPUT_ROOT / "musique_verifier_system_scaling_30q_v1" / "scaling.json"
    )["scaling"]
    gpu = np.asarray([row["world_size"] for row in scaling])
    bridge = np.asarray([row["stages"]["bridge"]["mean_seconds"] for row in scaling])
    answer = np.asarray(
        [row["stages"]["answer_generation"]["mean_seconds"] for row in scaling]
    )
    verifier = np.asarray([row["stages"]["verifier"]["mean_seconds"] for row in scaling])

    plt.figure(figsize=(7.3, 4.5))
    plt.bar(gpu, bridge, label="Bridge (serial)")
    plt.bar(gpu, answer, bottom=bridge, label="16 answer branches")
    plt.bar(gpu, verifier, bottom=bridge + answer, label="16 verifiers")
    plt.xticks(gpu)
    plt.xlabel("GPUs per request")
    plt.ylabel("Mean model wall-clock (s/query)")
    plt.title("Parallel branches reduce latency; the bridge remains serial")
    plt.legend(frameon=False)
    save("gpu_stage_scaling.png")


def plot_bridge_conditioned_recall() -> None:
    _, rag = load_system_results()
    labels = ["Bridge correct", "Bridge wrong"]
    svd = [89.29, 29.88]
    rag_values = [
        100.0 * rag["second_top16_given_bridge_correct"],
        100.0 * rag["second_top16_given_bridge_wrong"],
    ]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(6.8, 4.3))
    plt.bar(x - width / 2, svd, width, label="SVD32 pipeline")
    plt.bar(x + width / 2, rag_values, width, label="Hybrid-RAG pipeline")
    plt.xticks(x, labels)
    plt.ylabel("Dynamic second-hop Recall@16 (%)")
    plt.ylim(0, 100)
    plt.title("Bridge errors cause query drift; Hybrid-RAG degrades less")
    plt.legend(frameon=False)
    for index, value in enumerate(svd):
        plt.text(index - width / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(rag_values):
        plt.text(index + width / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
    save("bridge_conditioned_second_recall.png")


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_pipeline_funnel()
    plot_quality_comparison()
    plot_gpu_scaling()
    plot_bridge_conditioned_recall()


if __name__ == "__main__":
    main()
