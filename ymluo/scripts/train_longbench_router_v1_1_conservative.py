#!/usr/bin/env python3
"""Train a conservative dynamic LongBench router.

v1 labels individual examples by "smallest action close to the per-example
best", which is noisy when all sparse actions fail. This conservative variant
uses task-level average labels and an explicit fallback flag for high-risk
tasks. It is a better first router for quality-priority experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_longbench_router_v1 import (
    TASK_FAMILY,
    build_budget_dataset,
    build_page_dataset,
    load_longbench_rows,
    train_classifier,
)


OUTPUTS_ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs")
LONGBENCH_ZIP = (
    OUTPUTS_ROOT
    / "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2"
    / "longbench_data.zip"
)
OUTPUT_DIR = OUTPUTS_ROOT / "longbench_dynamic_router_v1_1_conservative_20260709"
SEED = 2026070911


NUMERIC_COLS = [
    "context_length_field",
    "raw_prefix_tokens",
    "raw_prompt_tokens",
    "log_raw_prefix_tokens",
    "query_chars",
    "query_words",
    "query_content_words",
    "query_numbers",
    "query_entities",
    "query_has_question",
    "context_chars",
    "context_words_20k",
    "context_numbers_20k",
    "query_context_overlap",
    "query_context_overlap_ratio",
    "answer_count",
]
CATEGORICAL_COLS = ["task", "family", "metric"]
PAGE_CANDIDATES = [32, 64, 128, 256, 512]
BUDGET_CANDIDATES = [128, 256, 512, 1024]
FALLBACK_TASKS = {"passage_retrieval_en", "passage_count"}


def choose_task_page_labels(page_df: pd.DataFrame, eps: float = 0.005) -> dict[str, int]:
    labels = {}
    for task, group in page_df.groupby("task"):
        means = {p: float(group[f"score_p{p}"].mean()) for p in PAGE_CANDIDATES if f"score_p{p}" in group}
        best = max(means.values())
        safe = [p for p, score in means.items() if score >= best - eps]
        labels[str(task)] = min(safe)
    return labels


def choose_task_budget_labels(budget_df: pd.DataFrame, eps: float = 0.01) -> dict[str, int]:
    labels = {}
    for task, group in budget_df.groupby("task"):
        means = {b: float(group[f"score_b{b}"].mean()) for b in BUDGET_CANDIDATES if f"score_b{b}" in group}
        best = max(means.values())
        safe = [b for b, score in means.items() if score >= best - eps]
        labels[str(task)] = min(safe)
    return labels


def simulate_task_policy(df: pd.DataFrame, label_col: str, kind: str) -> dict[str, float]:
    scores = []
    oracle = []
    costs = []
    if kind == "page":
        for _, row in df.iterrows():
            page = int(row[label_col])
            scores.append(float(row[f"score_p{page}"]))
            oracle.append(float(row["oracle_page_score"]))
            costs.append(page)
        return {
            "policy_score": float(np.mean(scores)),
            "oracle_score": float(np.mean(oracle)),
            "mean_page_tokens": float(np.mean(costs)),
        }
    for _, row in df.iterrows():
        if int(row["fallback_label"]):
            continue
        budget = int(row[label_col])
        if f"score_b{budget}" in row:
            scores.append(float(row[f"score_b{budget}"]))
            oracle.append(float(row["oracle_budget_score"]))
            costs.append(budget)
    return {
        "policy_score_sparse_only": float(np.mean(scores)),
        "oracle_score_sparse_only": float(np.mean(oracle)),
        "mean_budget_sparse_only": float(np.mean(costs)),
        "fallback_rate": float(df["fallback_label"].mean()),
    }


def write_report(
    page_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    page_task_labels: dict[str, int],
    budget_task_labels: dict[str, int],
    page_metrics: dict,
    budget_metrics: dict,
    fallback_metrics: dict,
    page_sim: dict,
    budget_sim: dict,
) -> None:
    lines = [
        "# LongBench Dynamic Router v1.1 Conservative",
        "",
        "This router uses task-level calibrated labels and an explicit high-risk fallback head.",
        "",
        "## Task Labels",
        "",
        "### Page Tokens",
        "",
    ]
    for task in sorted(page_task_labels):
        lines.append(f"- {task}: page_tokens={page_task_labels[task]}")
    lines += ["", "### Budget", ""]
    for task in sorted(budget_task_labels):
        fb = " + fallback" if task in FALLBACK_TASKS else ""
        lines.append(f"- {task}: budget={budget_task_labels[task]}{fb}")
    lines += [
        "",
        "## Metrics",
        "",
        f"- Page router accuracy: {page_metrics['accuracy']:.4f}",
        f"- Budget router accuracy: {budget_metrics['accuracy']:.4f}",
        f"- Fallback router accuracy: {fallback_metrics['accuracy']:.4f}",
        "",
        "## Simulated Policy",
        "",
        f"- Page policy score: {page_sim['policy_score']:.4f}",
        f"- Page oracle score: {page_sim['oracle_score']:.4f}",
        f"- Mean page_tokens: {page_sim['mean_page_tokens']:.1f}",
        f"- Sparse budget policy score: {budget_sim['policy_score_sparse_only']:.4f}",
        f"- Sparse budget oracle score: {budget_sim['oracle_score_sparse_only']:.4f}",
        f"- Mean sparse budget: {budget_sim['mean_budget_sparse_only']:.1f}",
        f"- Fallback rate: {budget_sim['fallback_rate']:.4f}",
        "",
        "## Classification Reports",
        "",
        "### Page",
        "```text",
        page_metrics["classification_report"],
        "```",
        "",
        "### Budget",
        "```text",
        budget_metrics["classification_report"],
        "```",
        "",
        "### Fallback",
        "```text",
        fallback_metrics["classification_report"],
        "```",
    ]
    (OUTPUT_DIR / "router_v1_1_conservative_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lb_rows = load_longbench_rows(LONGBENCH_ZIP)
    page_df = build_page_dataset(OUTPUTS_ROOT, lb_rows, eps=0.005)
    budget_df = build_budget_dataset(OUTPUTS_ROOT, lb_rows, eps=0.01)

    page_task_labels = choose_task_page_labels(page_df)
    budget_task_labels = choose_task_budget_labels(budget_df)
    page_df["label_page_tokens_conservative"] = page_df["task"].map(page_task_labels).astype(int)
    budget_df["label_budget_conservative"] = budget_df["task"].map(budget_task_labels).astype(int)
    budget_df["fallback_label"] = budget_df["task"].isin(FALLBACK_TASKS).astype(int)

    page_model, page_metrics, page_pred = train_classifier(
        page_df,
        "label_page_tokens_conservative",
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        SEED,
    )
    budget_model, budget_metrics, budget_pred = train_classifier(
        budget_df,
        "label_budget_conservative",
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        SEED + 1,
    )
    fallback_model, fallback_metrics, fallback_pred = train_classifier(
        budget_df,
        "fallback_label",
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        SEED + 2,
    )

    page_sim = simulate_task_policy(page_df, "label_page_tokens_conservative", "page")
    budget_sim = simulate_task_policy(budget_df, "label_budget_conservative", "budget")

    page_df.to_csv(OUTPUT_DIR / "page_router_dataset_conservative.csv", index=False)
    budget_df.to_csv(OUTPUT_DIR / "budget_router_dataset_conservative.csv", index=False)
    page_pred.to_csv(OUTPUT_DIR / "page_holdout_predictions_conservative.csv", index=False)
    budget_pred.to_csv(OUTPUT_DIR / "budget_holdout_predictions_conservative.csv", index=False)
    fallback_pred.to_csv(OUTPUT_DIR / "fallback_holdout_predictions_conservative.csv", index=False)

    artifact = {
        "page_model": page_model,
        "budget_model": budget_model,
        "fallback_model": fallback_model,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "task_family": TASK_FAMILY,
        "page_task_labels": page_task_labels,
        "budget_task_labels": budget_task_labels,
        "fallback_tasks": sorted(FALLBACK_TASKS),
        "page_metrics": page_metrics,
        "budget_metrics": budget_metrics,
        "fallback_metrics": fallback_metrics,
        "page_sim": page_sim,
        "budget_sim": budget_sim,
    }
    joblib.dump(artifact, OUTPUT_DIR / "longbench_dynamic_router_v1_1_conservative.joblib")
    write_report(
        page_df,
        budget_df,
        page_task_labels,
        budget_task_labels,
        page_metrics,
        budget_metrics,
        fallback_metrics,
        page_sim,
        budget_sim,
    )
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "page_accuracy": page_metrics["accuracy"],
        "budget_accuracy": budget_metrics["accuracy"],
        "fallback_accuracy": fallback_metrics["accuracy"],
        "page_sim": page_sim,
        "budget_sim": budget_sim,
        "page_task_labels": page_task_labels,
        "budget_task_labels": budget_task_labels,
        "fallback_tasks": sorted(FALLBACK_TASKS),
    }, indent=2))


if __name__ == "__main__":
    main()
