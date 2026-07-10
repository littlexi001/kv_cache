#!/usr/bin/env python3
"""Train a first-pass dynamic LongBench router from completed RiskKV sweeps.

This script distills existing sweep results into two lightweight router heads:

1. page_router: choose page/block size from page_tokens sweeps at B=512.
2. budget_router: choose the smallest safe budget from fixed-budget sweeps.

The labels are pseudo-oracle labels from already completed experiments. This is
intended as a practical v1 router, not as the final learned router with online
retriever-gap features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK_FAMILY = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "few_shot",
    "triviaqa": "few_shot",
    "samsum": "few_shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "are",
    "was",
    "were",
    "will",
    "would",
    "should",
    "could",
    "into",
    "about",
    "after",
    "before",
    "between",
    "there",
    "their",
    "have",
    "has",
    "had",
    "does",
    "did",
    "not",
    "you",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_root",
        default="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs",
    )
    parser.add_argument(
        "--longbench_zip",
        default=(
            "/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/"
            "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/"
            "longbench_dynamic_router_v1_20260709"
        ),
    )
    parser.add_argument("--page_eps", type=float, default=0.01)
    parser.add_argument("--budget_eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_longbench_rows(zip_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".jsonl"):
                continue
            task = Path(name).stem
            text = archive.read(name).decode("utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("_id", ""))
                if sample_id:
                    rows[(task, sample_id)] = row
    return rows


def words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS}


def number_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def entity_count(text: str) -> int:
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text)
    uppers = re.findall(r"\b[A-Z0-9_]{3,}\b", text)
    return len(caps) + len(uppers)


def family_for(task: str) -> str:
    return TASK_FAMILY.get(task, "other")


def base_features(row: pd.Series, lb_row: dict[str, Any] | None) -> dict[str, Any]:
    task = str(row["task"])
    query = str((lb_row or {}).get("input", ""))
    context = str((lb_row or {}).get("context", ""))
    answers = (lb_row or {}).get("answers") or []
    q_words = words(query)
    q_content = content_words(query)
    c_content = content_words(context[:20000])
    overlap = len(q_content & c_content)
    overlap_ratio = overlap / max(1, len(q_content))
    raw_prefix = float(row.get("raw_prefix_tokens", 0) or 0)
    ctx_len = float(row.get("context_length_field", 0) or 0)
    return {
        "task": task,
        "family": family_for(task),
        "metric": str(row.get("metric", "")),
        "sample_id": str(row["sample_id"]),
        "context_length_field": ctx_len,
        "raw_prefix_tokens": raw_prefix,
        "raw_prompt_tokens": float(row.get("raw_prompt_tokens", 0) or 0),
        "log_raw_prefix_tokens": math.log1p(max(0.0, raw_prefix)),
        "query_chars": len(query),
        "query_words": len(q_words),
        "query_content_words": len(q_content),
        "query_numbers": number_count(query),
        "query_entities": entity_count(query),
        "query_has_question": int("?" in query),
        "context_chars": len(context),
        "context_words_20k": len(words(context[:20000])),
        "context_numbers_20k": number_count(context[:20000]),
        "query_context_overlap": overlap,
        "query_context_overlap_ratio": overlap_ratio,
        "answer_count": len(answers) if isinstance(answers, list) else 0,
    }


def build_page_dataset(outputs_root: Path, lb_rows: dict[tuple[str, str], dict[str, Any]], eps: float) -> pd.DataFrame:
    page_dirs = {
        32: "table5_question_aware_riskkv_20260708_llama_blocksize_p32_b512_m20/riskkv_question_aware_b512",
        64: "table5_question_aware_riskkv_20260708_llama_blocksize_p64_b512_m20/riskkv_question_aware_b512",
        128: "table5_question_aware_riskkv_20260708_llama_blocksize_p128_b512_m20/riskkv_question_aware_b512",
        256: "table5_question_aware_riskkv_20260708_llama_blocksize_p256_b512_m20/riskkv_question_aware_b512",
        512: "table5_question_aware_riskkv_20260708_llama_blocksize_p512_b512_m20/riskkv_question_aware_b512",
    }
    by_key: dict[tuple[str, str], dict[int, pd.Series]] = defaultdict(dict)
    for page, rel in page_dirs.items():
        path = outputs_root / rel / "task_results.csv"
        if not path.exists():
            continue
        df = read_results(path)
        for _, row in df.iterrows():
            by_key[(str(row["task"]), str(row["sample_id"]))][page] = row

    out = []
    for key, variants in by_key.items():
        if len(variants) < 2:
            continue
        scores = {page: float(row["score"]) for page, row in variants.items()}
        best = max(scores.values())
        safe_pages = [page for page, score in scores.items() if score >= best - eps]
        label = min(safe_pages)
        ref = variants[128] if 128 in variants else next(iter(variants.values()))
        features = base_features(ref, lb_rows.get(key))
        for page, score in scores.items():
            features[f"score_p{page}"] = score
        features["label_page_tokens"] = int(label)
        features["oracle_page_tokens"] = int(max(scores, key=scores.get))
        features["oracle_page_score"] = float(best)
        features["label_page_score"] = float(scores[label])
        features["available_pages"] = ",".join(str(x) for x in sorted(scores))
        out.append(features)
    return pd.DataFrame(out)


def build_budget_dataset(outputs_root: Path, lb_rows: dict[tuple[str, str], dict[str, Any]], eps: float) -> pd.DataFrame:
    budget_dirs = {
        128: "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b128",
        256: "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b256",
        512: "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b512",
        1024: "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b1024",
        2048: "table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b2048",
    }
    by_key: dict[tuple[str, str], dict[int, pd.Series]] = defaultdict(dict)
    for budget, rel in budget_dirs.items():
        path = outputs_root / rel / "task_results.csv"
        if not path.exists():
            continue
        df = read_results(path)
        for _, row in df.iterrows():
            by_key[(str(row["task"]), str(row["sample_id"]))][budget] = row

    out = []
    for key, variants in by_key.items():
        if len(variants) < 2:
            continue
        scores = {budget: float(row["score"]) for budget, row in variants.items()}
        best = max(scores.values())
        safe_budgets = [budget for budget, score in scores.items() if score >= best - eps]
        label = min(safe_budgets)
        ref = variants[512] if 512 in variants else next(iter(variants.values()))
        features = base_features(ref, lb_rows.get(key))
        for budget, score in scores.items():
            features[f"score_b{budget}"] = score
        features["label_budget"] = int(label)
        features["oracle_budget"] = int(max(scores, key=scores.get))
        features["oracle_budget_score"] = float(best)
        features["label_budget_score"] = float(scores[label])
        features["budget_spread"] = float(max(scores.values()) - min(scores.values()))
        features["danger_label"] = int(label >= 1024 or (scores.get(1024, best) - scores.get(512, best)) > 0.05)
        features["available_budgets"] = ",".join(str(x) for x in sorted(scores))
        out.append(features)
    return pd.DataFrame(out)


def train_classifier(
    df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    work = df.dropna(subset=[label_col]).copy()
    y = work[label_col].astype(str)
    X = work[numeric_cols + categorical_cols]
    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        work.index.to_numpy(),
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", pre), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "n": int(len(work)),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "label_counts": dict(Counter(y)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
        "labels": sorted(y.unique()),
    }
    pred_df = work.loc[idx_test].copy()
    pred_df[f"pred_{label_col}"] = pred
    return pipe, metrics, pred_df


def simulate_page(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    scores_pred = []
    scores_p128 = []
    scores_p32 = []
    scores_oracle = []
    for _, row in df.iterrows():
        pred = int(row[pred_col])
        scores_pred.append(float(row.get(f"score_p{pred}", np.nan)))
        scores_p128.append(float(row.get("score_p128", np.nan)))
        scores_p32.append(float(row.get("score_p32", np.nan)))
        scores_oracle.append(float(row["oracle_page_score"]))
    return {
        "pred_score": float(np.nanmean(scores_pred)),
        "p128_score": float(np.nanmean(scores_p128)),
        "p32_score": float(np.nanmean(scores_p32)),
        "oracle_score": float(np.nanmean(scores_oracle)),
    }


def simulate_budget(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    scores_pred = []
    scores_b512 = []
    scores_b1024 = []
    scores_oracle = []
    budgets = []
    for _, row in df.iterrows():
        pred = int(row[pred_col])
        budgets.append(pred)
        scores_pred.append(float(row.get(f"score_b{pred}", np.nan)))
        scores_b512.append(float(row.get("score_b512", np.nan)))
        scores_b1024.append(float(row.get("score_b1024", np.nan)))
        scores_oracle.append(float(row["oracle_budget_score"]))
    return {
        "pred_score": float(np.nanmean(scores_pred)),
        "b512_score": float(np.nanmean(scores_b512)),
        "b1024_score": float(np.nanmean(scores_b1024)),
        "oracle_score": float(np.nanmean(scores_oracle)),
        "pred_mean_budget": float(np.mean(budgets)),
    }


def write_report(
    output_dir: Path,
    page_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    page_metrics: dict[str, Any],
    budget_metrics: dict[str, Any],
    page_sim: dict[str, float],
    budget_sim: dict[str, float],
) -> None:
    lines = [
        "# LongBench Dynamic Router v1",
        "",
        "This is a first-pass router distilled from completed RiskKV sweeps.",
        "",
        "## Datasets",
        "",
        f"- Page router samples: {len(page_df)}",
        f"- Budget router samples: {len(budget_df)}",
        "",
        "## Page Router",
        "",
        f"- Accuracy: {page_metrics['accuracy']:.4f}",
        f"- Label counts: `{page_metrics['label_counts']}`",
        "",
        "Simulation on page-router held-out rows:",
        "",
        f"- Predicted policy score: {page_sim['pred_score']:.4f}",
        f"- Fixed page=128 score: {page_sim['p128_score']:.4f}",
        f"- Fixed page=32 score: {page_sim['p32_score']:.4f}",
        f"- Oracle page score: {page_sim['oracle_score']:.4f}",
        "",
        "```text",
        page_metrics["classification_report"],
        "```",
        "",
        "## Budget Router",
        "",
        f"- Accuracy: {budget_metrics['accuracy']:.4f}",
        f"- Label counts: `{budget_metrics['label_counts']}`",
        "",
        "Simulation on budget-router held-out rows:",
        "",
        f"- Predicted policy score: {budget_sim['pred_score']:.4f}",
        f"- Fixed B=512 score: {budget_sim['b512_score']:.4f}",
        f"- Fixed B=1024 score: {budget_sim['b1024_score']:.4f}",
        f"- Oracle budget score: {budget_sim['oracle_score']:.4f}",
        f"- Predicted mean budget: {budget_sim['pred_mean_budget']:.1f}",
        "",
        "```text",
        budget_metrics["classification_report"],
        "```",
    ]
    (output_dir / "router_v1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lb_rows = load_longbench_rows(Path(args.longbench_zip))

    page_df = build_page_dataset(outputs_root, lb_rows, args.page_eps)
    budget_df = build_budget_dataset(outputs_root, lb_rows, args.budget_eps)
    if page_df.empty:
        raise SystemExit("No page-router training data found.")
    if budget_df.empty:
        raise SystemExit("No budget-router training data found.")

    numeric_cols = [
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
    categorical_cols = ["task", "family", "metric"]

    page_model, page_metrics, page_pred = train_classifier(
        page_df,
        "label_page_tokens",
        numeric_cols,
        categorical_cols,
        args.seed,
    )
    budget_model, budget_metrics, budget_pred = train_classifier(
        budget_df,
        "label_budget",
        numeric_cols,
        categorical_cols,
        args.seed + 1,
    )

    page_sim = simulate_page(page_pred, "pred_label_page_tokens")
    budget_sim = simulate_budget(budget_pred, "pred_label_budget")

    page_df.to_csv(output_dir / "page_router_dataset.csv", index=False)
    budget_df.to_csv(output_dir / "budget_router_dataset.csv", index=False)
    page_pred.to_csv(output_dir / "page_router_holdout_predictions.csv", index=False)
    budget_pred.to_csv(output_dir / "budget_router_holdout_predictions.csv", index=False)
    joblib.dump(
        {
            "page_model": page_model,
            "budget_model": budget_model,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "task_family": TASK_FAMILY,
            "page_metrics": page_metrics,
            "budget_metrics": budget_metrics,
            "page_sim": page_sim,
            "budget_sim": budget_sim,
            "page_eps": args.page_eps,
            "budget_eps": args.budget_eps,
        },
        output_dir / "longbench_dynamic_router_v1.joblib",
    )
    write_report(output_dir, page_df, budget_df, page_metrics, budget_metrics, page_sim, budget_sim)
    print(json.dumps({
        "output_dir": str(output_dir),
        "page_samples": len(page_df),
        "budget_samples": len(budget_df),
        "page_accuracy": page_metrics["accuracy"],
        "budget_accuracy": budget_metrics["accuracy"],
        "page_sim": page_sim,
        "budget_sim": budget_sim,
    }, indent=2))


if __name__ == "__main__":
    main()
