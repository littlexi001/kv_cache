from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION_TERMS = re.compile(
    r"\b(latest|most recent|recently|current|currently|now|newest|last time|"
    r"still|anymore|these days|changed?|updated?|switched?|started?|stopped?|"
    r"moved?|became|replaced?)\b",
    re.IGNORECASE,
)
ORDER_TERMS = re.compile(
    r"\b(before|after|first|last|earlier|later|when|how long|since|until|date|time)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict whether a LongMemEval query needs version-chain handling using "
            "only question/state text and retrieved-page metadata."
        )
    )
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--selection_pattern", required=True)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_corr(values: np.ndarray) -> float:
    if len(values) < 2 or float(np.std(values)) == 0.0:
        return 0.0
    return float(np.corrcoef(np.arange(len(values)), values)[0, 1])


def cue_features(text: str) -> list[float]:
    lower = text.lower()
    return [
        float(bool(VERSION_TERMS.search(text))),
        float(bool(ORDER_TERMS.search(text))),
        float("most recent" in lower or "latest" in lower or "currently" in lower),
        float("now" in lower or "current" in lower),
        float("change" in lower or "switch" in lower or "update" in lower),
        float("still" in lower or "anymore" in lower),
        float(len(re.findall(r"\b\d+\b", lower))),
        float(len(re.findall(r"\b(?:day|week|month|year)s?\b", lower))),
        float(len(re.findall(r"\b\w+\b", lower))),
    ]


def page_features(
    block_ids: list[int],
    block_dates: np.ndarray,
    block_sessions: np.ndarray,
    question_date: int,
) -> list[float]:
    dates = np.asarray([int(block_dates[index]) for index in block_ids], dtype=np.float64)
    sessions = np.asarray([int(block_sessions[index]) for index in block_ids])
    valid = dates > 0
    dates = dates[valid]
    if not len(dates):
        return [0.0] * 12
    ages_days = (float(question_date) - dates) / (60.0 * 24.0)
    span_days = float((dates.max() - dates.min()) / (60.0 * 24.0))
    sorted_dates = np.sort(np.unique(dates))
    max_gap_days = (
        float(np.diff(sorted_dates).max() / (60.0 * 24.0))
        if len(sorted_dates) > 1
        else 0.0
    )
    return [
        float(len(np.unique(sessions))),
        float(len(np.unique(dates))),
        span_days,
        max_gap_days,
        float(np.mean(ages_days)),
        float(np.min(ages_days)),
        float(np.max(ages_days)),
        float(np.std(ages_days)),
        float(np.mean(ages_days <= 7.0)),
        float(np.mean(ages_days <= 30.0)),
        float(np.mean(ages_days <= 90.0)),
        safe_corr(dates),
    ]


def evaluate_predictions(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predicted = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "threshold_0_5": {
            "selection_rate": float(predicted.mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive": int((predicted & labels).sum()),
            "false_positive": int((predicted & ~labels).sum()),
            "false_negative": int((~predicted & labels).sum()),
        },
    }


def fit_text_fold(
    train_text: list[str],
    test_text: list[str],
    train_labels: np.ndarray,
    *,
    seed: int,
) -> tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    return train_matrix.tocsr(), test_matrix.tocsr(), vectorizer


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for partition in range(args.partitions):
        data_dir = Path(args.data_pattern.format(partition=partition))
        selection_dir = Path(args.selection_pattern.format(partition=partition))
        queries = {int(row["query_id"]): row for row in read_jsonl(data_dir / "queries.jsonl")}
        states = {
            int(row["query_id"]): row
            for row in read_jsonl(selection_dir / "states.jsonl")
        }
        static = {
            int(row["query_id"]): row
            for row in read_jsonl(selection_dir / "rows.jsonl")
            if row["method"] == "static_top12"
        }
        block_dates = np.load(data_dir / "base_block_date_minutes.npy", mmap_mode="r")
        block_sessions = np.load(data_dir / "base_block_session_rows.npy", mmap_mode="r")
        if set(queries) != set(states) or set(queries) != set(static):
            raise RuntimeError(f"partition {partition} rows do not align")
        for query_id, query in queries.items():
            question = str(query["question"])
            state = str(states[query_id]["state_text"])
            block_ids = list(map(int, static[query_id]["top_block_ids"]))
            records.append(
                {
                    "partition": partition,
                    "query_id": query_id,
                    "question_id": str(query["question_id"]),
                    "question_type": str(query["question_type"]),
                    "is_abstention": bool(query["is_abstention"]),
                    "question": question,
                    "state": state,
                    "question_state": question + " [STATE] " + state,
                    "dense_question": cue_features(question)
                    + page_features(
                        block_ids,
                        block_dates,
                        block_sessions,
                        int(query["question_date_minutes"]),
                    ),
                    "dense_question_state": cue_features(question)
                    + cue_features(state)
                    + page_features(
                        block_ids,
                        block_dates,
                        block_sessions,
                        int(query["question_date_minutes"]),
                    ),
                }
            )
    records.sort(key=lambda row: (row["partition"], row["question_id"]))
    if len(records) != 500:
        raise RuntimeError(f"expected 500 records, got {len(records)}")

    labels = np.asarray(
        [record["question_type"] == "knowledge-update" for record in records],
        dtype=bool,
    )
    groups = np.asarray([record["partition"] for record in records])
    variants = {
        "question_text": ("question", "dense_question", False),
        "question_text_plus_page_dates": ("question", "dense_question", True),
        "question_state_plus_page_dates": (
            "question_state",
            "dense_question_state",
            True,
        ),
    }
    predictions = {name: np.zeros(len(records), dtype=np.float64) for name in variants}
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = groups == held_out
        for variant_index, (name, (text_key, dense_key, include_dense)) in enumerate(
            variants.items()
        ):
            train_text = [records[index][text_key] for index in np.flatnonzero(train)]
            test_text = [records[index][text_key] for index in np.flatnonzero(test)]
            train_x, test_x, _ = fit_text_fold(
                train_text,
                test_text,
                labels[train],
                seed=args.seed + variant_index,
            )
            if include_dense:
                scaler = StandardScaler()
                train_dense = scaler.fit_transform(
                    np.asarray([records[index][dense_key] for index in np.flatnonzero(train)])
                )
                test_dense = scaler.transform(
                    np.asarray([records[index][dense_key] for index in np.flatnonzero(test)])
                )
                train_x = hstack([train_x, csr_matrix(train_dense)], format="csr")
                test_x = hstack([test_x, csr_matrix(test_dense)], format="csr")
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                solver="liblinear",
                random_state=args.seed + variant_index,
            )
            model.fit(train_x, labels[train])
            predictions[name][test] = model.predict_proba(test_x)[:, 1]

    heuristic = np.asarray(
        [float(bool(VERSION_TERMS.search(record["question"]))) for record in records]
    )
    output = {
        "protocol": {
            "memory_scope": "eight independent real 10M-token shards",
            "target_for_analysis_only": "knowledge-update question type",
            "test_time_features": "question/state text and static retrieved-page dates only",
            "answers_or_gold_evidence_used_at_test": False,
            "outer_validation": "leave-one-10M-shard-out",
        },
        "queries": len(records),
        "version_queries": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "heuristic_version_terms": evaluate_predictions(labels, heuristic),
        "models": {
            name: evaluate_predictions(labels, scores)
            for name, scores in predictions.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_path = output_path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "partition": record["partition"],
                        "question_id": record["question_id"],
                        "question_type": record["question_type"],
                        "is_version_target": bool(labels[index]),
                        "heuristic_version_term": bool(heuristic[index]),
                        **{
                            f"{name}_probability": float(scores[index])
                            for name, scores in predictions.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
