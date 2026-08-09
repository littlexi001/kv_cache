from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


NUMERIC_FEATURES = [
    "question_nll_good_z",
    "block_bm25_z",
    "record_bm25_z",
    "global_nll_rank_fraction",
    "global_bm25_rank_fraction",
    "record_rank_fraction",
    "within_record_nll_rank_fraction",
    "within_record_bm25_rank_fraction",
    "record_top3_nll_good_z",
    "record_top8_nll_good_z",
    "record_ql_rank_fraction",
    "block_position_fraction",
    "log_record_blocks",
    "nll_bm25_interaction",
    "record_block_interaction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a query-disjoint candidate block reranker."
    )
    parser.add_argument("--train", action="append", required=True, help="CSV glob")
    parser.add_argument("--test", action="append", required=True, help="CSV glob")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_blocks", type=int, default=39)
    return parser.parse_args()


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError(f"no candidate score files matched {patterns}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            source = Path(path).parent.name
            rows.extend({**row, "_source": source} for row in csv.DictReader(handle))
    return rows


def standardized(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1.0e-6)


def average_ranks(values: np.ndarray, largest: bool = False) -> np.ndarray:
    ids = np.arange(len(values), dtype=np.int64)
    order = np.lexsort((ids, -values if largest else values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def build_query_features(
    rows: list[dict[str, Any]], dataset_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    count = len(rows)
    qnll = np.asarray([float(row["question_nll"]) for row in rows])
    block_bm25 = np.asarray([float(row["block_bm25_score"]) for row in rows])
    record_bm25 = np.asarray([float(row["record_bm25_score"]) for row in rows])
    record_ids = np.asarray([int(row["record_id"]) for row in rows])
    unique_records = sorted(set(record_ids.tolist()))
    record_top3: dict[int, float] = {}
    record_top8: dict[int, float] = {}
    within_nll_rank = np.empty(count, dtype=np.float64)
    within_bm25_rank = np.empty(count, dtype=np.float64)
    record_size = np.empty(count, dtype=np.float64)
    for record_id in unique_records:
        positions = np.flatnonzero(record_ids == record_id)
        values = np.sort(qnll[positions])
        record_top3[record_id] = float(values[: min(3, len(values))].mean())
        record_top8[record_id] = float(values[: min(8, len(values))].mean())
        within_nll_rank[positions] = average_ranks(qnll[positions]) / max(
            len(positions) - 1, 1
        )
        within_bm25_rank[positions] = average_ranks(
            block_bm25[positions], largest=True
        ) / max(len(positions) - 1, 1)
        record_size[positions] = float(rows[int(positions[0])]["record_blocks"])
    record_top3_values = np.asarray([record_top3[item] for item in record_ids])
    record_top8_values = np.asarray([record_top8[item] for item in record_ids])
    unique_top3 = np.asarray([record_top3[item] for item in unique_records])
    unique_record_rank = average_ranks(unique_top3)
    rank_by_record = dict(zip(unique_records, unique_record_rank.tolist()))
    record_ql_rank = np.asarray([rank_by_record[item] for item in record_ids]) / max(
        len(unique_records) - 1, 1
    )

    qnll_good_z = standardized(-qnll)
    block_bm25_z = standardized(block_bm25)
    record_bm25_z = standardized(record_bm25)
    matrix = np.column_stack(
        [
            qnll_good_z,
            block_bm25_z,
            record_bm25_z,
            average_ranks(qnll) / max(count - 1, 1),
            average_ranks(block_bm25, largest=True) / max(count - 1, 1),
            np.asarray([float(row["record_bm25_rank"]) for row in rows])
            / max(max(float(row["record_bm25_rank"]) for row in rows), 1.0),
            within_nll_rank,
            within_bm25_rank,
            standardized(-record_top3_values),
            standardized(-record_top8_values),
            record_ql_rank,
            np.asarray([float(row["block_position_fraction"]) for row in rows]),
            np.log1p(record_size),
            qnll_good_z * block_bm25_z,
            record_bm25_z * block_bm25_z,
        ]
    ).astype(np.float32)
    dataset_matrix = np.zeros((count, len(dataset_names)), dtype=np.float32)
    dataset_to_column = {name: index for index, name in enumerate(dataset_names)}
    for index, row in enumerate(rows):
        column = dataset_to_column.get(str(row["dataset"]))
        if column is not None:
            dataset_matrix[index, column] = 1.0
    matrix = np.column_stack([matrix, dataset_matrix]).astype(np.float32)
    labels = np.asarray([float(row["is_gold"]) for row in rows], dtype=np.float32)
    block_ids = [int(row["block_id"]) for row in rows]
    record_id_list = [int(row["record_id"]) for row in rows]
    feature_names = NUMERIC_FEATURES + [f"dataset={name}" for name in dataset_names]
    if matrix.shape[1] != len(feature_names) or not np.all(np.isfinite(matrix)):
        raise ValueError("candidate feature matrix is invalid")
    return matrix, labels, block_ids, record_id_list, feature_names


def grouped_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[(str(row["_source"]), int(row["query_id"]))].append(row)
    return dict(sorted(output.items()))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_groups = grouped_rows(load_rows(args.train))
    test_groups = grouped_rows(load_rows(args.test))
    dataset_names = sorted(
        {str(row["dataset"]) for rows in train_groups.values() for row in rows}
    )

    train_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    train_query_ids: list[tuple[str, int]] = []
    for query_key, rows in train_groups.items():
        matrix, labels, _, _, _ = build_query_features(rows, dataset_names)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            continue
        weights = np.where(
            labels > 0,
            0.5 / positives,
            0.5 / negatives,
        ).astype(np.float32)
        train_parts.append(matrix)
        label_parts.append(labels)
        weight_parts.append(weights)
        train_query_ids.append(query_key)
    train_x = np.concatenate(train_parts)
    train_y = np.concatenate(label_parts)
    train_weight = np.concatenate(weight_parts)

    configs = [
        {"max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 3.0},
        {"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 10.0},
    ]
    models = []
    for config in configs:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"],
            random_state=20260711,
        )
        model.fit(train_x, train_y, sample_weight=train_weight)
        models.append(model)

    query_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, float]] = []
    for query_key, rows in test_groups.items():
        query_id = query_key[1]
        matrix, labels, block_ids, record_ids, feature_names = build_query_features(
            rows, dataset_names
        )
        prediction = np.mean(
            [model.predict_proba(matrix)[:, 1] for model in models], axis=0
        )
        ids = np.asarray(block_ids, dtype=np.int64)
        order = np.lexsort((ids, -prediction))
        ranked = ids[order].tolist()
        selected = ranked[: args.target_blocks]
        gold = {block_ids[index] for index in np.flatnonzero(labels > 0)}
        selected_set = set(selected)
        source_records = {
            int(row["record_id"])
            for row in rows
            if float(row["is_source_record"]) > 0
        }
        metrics.append(
            {
                "any_evidence_recall": float(bool(selected_set & gold)),
                "all_evidence_recall": float(gold <= selected_set),
                "evidence_fraction": len(selected_set & gold) / max(len(gold), 1),
                "source_record_recall": float(
                    bool(source_records & {record_ids[index] for index in order[: args.target_blocks]})
                ),
            }
        )
        query_rows.append(
            {
                "method": "learned_candidate_reranker",
                "query_id": query_id,
                "dataset": rows[0]["dataset"],
                "selected_block_ids": json.dumps(selected),
                "ranked_block_ids": json.dumps(ranked),
                **metrics[-1],
            }
        )

    fields = list(query_rows[0])
    with (output_dir / "query_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_rows)
    bundle = {
        "models": models,
        "feature_names": feature_names,
        "dataset_names": dataset_names,
        "configs": configs,
    }
    joblib.dump(bundle, output_dir / "reranker.joblib", compress=3)
    summary = {
        "train_queries": len(train_query_ids),
        "test_queries": len(test_groups),
        "train_candidates": len(train_x),
        "target_blocks": args.target_blocks,
        "feature_names": feature_names,
        "any_evidence_recall": float(
            np.mean([row["any_evidence_recall"] for row in metrics])
        ),
        "all_evidence_recall": float(
            np.mean([row["all_evidence_recall"] for row in metrics])
        ),
        "evidence_fraction": float(
            np.mean([row["evidence_fraction"] for row in metrics])
        ),
        "source_record_recall": float(
            np.mean([row["source_record_recall"] for row in metrics])
        ),
        "note": (
            "Gold labels are used only for training candidate rows. Test ranking uses "
            "answer-free features and a query/record-disjoint split."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
