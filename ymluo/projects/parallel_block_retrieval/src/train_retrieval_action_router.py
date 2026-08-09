from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEEP_METHOD = "deep_ql_record39_svd32"
HYBRID_METHOD = "hybrid_record30_svd32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a corpus-disjoint router between two strict-39 retrieval actions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deep_method", default=DEEP_METHOD)
    parser.add_argument("--hybrid_method", default=HYBRID_METHOD)
    parser.add_argument("--target_blocks", type=int, default=39)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in json.loads(value)]


def entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(values)
    return float(-(probabilities * np.log(probabilities)).sum())


def overlap_fraction(left: Sequence[int], right: Sequence[int], top_k: int) -> float:
    left_set = set(left[:top_k])
    right_set = set(right[:top_k])
    return len(left_set & right_set) / max(min(len(left_set), len(right_set)), 1)


def record_distribution_features(
    block_ids: Sequence[int], block_to_record: dict[int, int]
) -> tuple[float, float, float]:
    record_ids = [block_to_record[int(block_id)] for block_id in block_ids]
    counts = Counter(record_ids)
    return (
        float(len(counts)),
        max(counts.values(), default=0) / max(len(record_ids), 1),
        entropy(record_ids),
    )


def question_features(question: str) -> list[float]:
    lowered = question.lower()
    words = question.split()
    return [
        math.log1p(len(question)),
        math.log1p(len(words)),
        sum(character.isdigit() for character in question) / max(len(question), 1),
        float("?" in question),
        float(any(token in lowered for token in ("who", "whose"))),
        float(any(token in lowered for token in ("when", "year", "date"))),
        float(any(token in lowered for token in ("where", "country", "city"))),
        float(any(token in lowered for token in ("why", "how"))),
        float(any(token in lowered for token in ("compare", "both", "difference"))),
    ]


FEATURE_NAMES = [
    "log_question_chars",
    "log_question_words",
    "question_digit_fraction",
    "has_question_mark",
    "asks_person",
    "asks_time",
    "asks_place",
    "asks_explanation",
    "asks_comparison",
    "relative_bm25_margin",
    "record_router_used_likelihood",
    "record_router_disagreement",
    "record_bm25_relative_gap",
    "record_question_nll_gap",
    "record_bm25_ql_agree",
    "candidate_log_count",
    "candidate_mean_question_nll",
    "candidate_min_question_nll",
    "candidate_question_nll_spread",
    "selected_overlap_5",
    "selected_overlap_10",
    "selected_overlap_20",
    "selected_overlap_39",
    "ranked_overlap_39",
    "deep_record_count",
    "deep_top_record_fraction",
    "deep_record_entropy",
    "hybrid_record_count",
    "hybrid_top_record_fraction",
    "hybrid_record_entropy",
    "record_count_difference",
    "top_record_fraction_difference",
]


def build_feature_vector(
    *,
    question: str,
    deep_row: dict[str, str],
    hybrid_row: dict[str, str],
    routing_row: dict[str, str],
    record_score_rows: Sequence[dict[str, str]],
    candidate_row: dict[str, str],
    block_to_record: dict[int, int],
) -> np.ndarray:
    deep_selected = parse_ids(deep_row["selected_block_ids"])
    hybrid_selected = parse_ids(hybrid_row["selected_block_ids"])
    deep_ranked = parse_ids(deep_row["ranked_block_ids"])
    hybrid_ranked = parse_ids(hybrid_row["ranked_block_ids"])

    bm25_rows = sorted(record_score_rows, key=lambda row: int(row["bm25_rank"]))
    ql_rows = sorted(record_score_rows, key=lambda row: float(row["question_nll"]))
    bm25_top = float(bm25_rows[0]["bm25_score"])
    bm25_second = float(bm25_rows[min(1, len(bm25_rows) - 1)]["bm25_score"])
    ql_top = float(ql_rows[0]["question_nll"])
    ql_second = float(ql_rows[min(1, len(ql_rows) - 1)]["question_nll"])
    deep_record = record_distribution_features(deep_selected, block_to_record)
    hybrid_record = record_distribution_features(hybrid_selected, block_to_record)
    candidate_mean = float(candidate_row["mean_block_question_nll"])
    candidate_min = float(candidate_row["min_block_question_nll"])

    values = question_features(question) + [
        float(routing_row["relative_bm25_margin"]),
        float(str(routing_row["used_likelihood"]).lower() == "true"),
        float(routing_row["bm25_record"] != routing_row["likelihood_record"]),
        (bm25_top - bm25_second) / max(abs(bm25_top), 1.0e-6),
        ql_second - ql_top,
        float(bm25_rows[0]["record_id"] == ql_rows[0]["record_id"]),
        math.log1p(float(candidate_row["candidate_blocks"])),
        candidate_mean,
        candidate_min,
        candidate_mean - candidate_min,
        overlap_fraction(deep_selected, hybrid_selected, 5),
        overlap_fraction(deep_selected, hybrid_selected, 10),
        overlap_fraction(deep_selected, hybrid_selected, 20),
        overlap_fraction(deep_selected, hybrid_selected, 39),
        overlap_fraction(deep_ranked, hybrid_ranked, 39),
        *deep_record,
        *hybrid_record,
        deep_record[0] - hybrid_record[0],
        deep_record[1] - hybrid_record[1],
    ]
    output = np.asarray(values, dtype=np.float64)
    if output.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(output)):
        raise ValueError("invalid answer-free action-router feature vector")
    return output


def load_bundle(
    spec: dict[str, str],
    config_dir: Path,
    deep_method: str,
    hybrid_method: str,
) -> dict[str, Any]:
    corpus_dir = resolve(config_dir, spec["corpus_dir"])
    queries = {int(row["query_id"]): row for row in read_jsonl(corpus_dir / "queries.jsonl")}
    records = read_jsonl(corpus_dir / "records.jsonl")
    block_to_record: dict[int, int] = {}
    for record_id, record in enumerate(records):
        start = int(record["block_start"])
        for block_id in range(start, start + int(record["block_count"])):
            block_to_record[block_id] = record_id

    retrieval: dict[tuple[str, int], dict[str, str]] = {}
    for row in read_csv(resolve(config_dir, spec["retrieval_csv"])):
        if row["method"] in {deep_method, hybrid_method}:
            retrieval[(row["method"], int(row["query_id"]))] = row
    routing = {
        int(row["query_id"]): row
        for row in read_csv(resolve(config_dir, spec["record_routing_csv"]))
    }
    record_scores: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(resolve(config_dir, spec["record_scores_csv"])):
        record_scores[int(row["query_id"])].append(row)
    candidate = {
        int(row["query_id"]): row
        for row in read_csv(resolve(config_dir, spec["candidate_diagnostics_csv"]))
    }
    nll: dict[tuple[str, int], float] = {}
    if spec.get("answer_nll_csv"):
        for row in read_csv(resolve(config_dir, spec["answer_nll_csv"])):
            nll[(row["mode"], int(row["query_id"]))] = float(row["answer_nll"])

    rows = []
    for query_id in sorted(queries):
        deep_row = retrieval[(deep_method, query_id)]
        hybrid_row = retrieval[(hybrid_method, query_id)]
        rows.append(
            {
                "query_id": query_id,
                "dataset": str(queries[query_id]["dataset"]),
                "features": build_feature_vector(
                    question=str(queries[query_id]["question"]),
                    deep_row=deep_row,
                    hybrid_row=hybrid_row,
                    routing_row=routing[query_id],
                    record_score_rows=record_scores[query_id],
                    candidate_row=candidate[query_id],
                    block_to_record=block_to_record,
                ),
                "deep_row": deep_row,
                "hybrid_row": hybrid_row,
                "deep_nll": nll.get((deep_method, query_id)),
                "hybrid_nll": nll.get((hybrid_method, query_id)),
            }
        )
    return {"name": spec["name"], "rows": rows}


def make_model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.stack([row["features"] for row in rows]).astype(np.float64)


def targets(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    if any(row["deep_nll"] is None or row["hybrid_nll"] is None for row in rows):
        raise ValueError("training bundles require both action NLL labels")
    return np.asarray(
        [float(row["hybrid_nll"]) - float(row["deep_nll"]) for row in rows],
        dtype=np.float64,
    )


def routed_nll(rows: Sequence[dict[str, Any]], prediction: np.ndarray, threshold: float) -> float:
    values = [
        float(row["hybrid_nll"] if estimate < -threshold else row["deep_nll"])
        for row, estimate in zip(rows, prediction)
    ]
    return float(np.mean(values))


def select_model(train_bundles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(train_bundles) < 2:
        raise ValueError("at least two train bundles are required for corpus-disjoint selection")
    alphas = [0.1, 1.0, 10.0, 100.0]
    thresholds = [0.0, 0.02, 0.05, 0.1, 0.2]
    candidates = []
    for alpha in alphas:
        heldout_rows: list[dict[str, Any]] = []
        heldout_prediction: list[float] = []
        for heldout_index, heldout in enumerate(train_bundles):
            fit_rows = [
                row
                for index, bundle in enumerate(train_bundles)
                if index != heldout_index
                for row in bundle["rows"]
            ]
            model = make_model(alpha)
            model.fit(matrix(fit_rows), targets(fit_rows))
            prediction = model.predict(matrix(heldout["rows"]))
            heldout_rows.extend(heldout["rows"])
            heldout_prediction.extend(prediction.tolist())
        prediction_array = np.asarray(heldout_prediction)
        for threshold in thresholds:
            candidates.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "oof_mean_nll": routed_nll(
                        heldout_rows, prediction_array, threshold
                    ),
                    "oof_switch_rate": float(np.mean(prediction_array < -threshold)),
                }
            )
    all_rows = [row for bundle in train_bundles for row in bundle["rows"]]
    always_deep_nll = float(np.mean([float(row["deep_nll"]) for row in all_rows]))
    best = min(candidates, key=lambda row: (row["oof_mean_nll"], row["oof_switch_rate"]))
    if best["oof_mean_nll"] >= always_deep_nll:
        return {
            "enabled": False,
            "alpha": best["alpha"],
            "threshold": float("inf"),
            "oof_mean_nll": always_deep_nll,
            "oof_switch_rate": 0.0,
            "always_deep_nll": always_deep_nll,
            "candidates": candidates,
        }
    return {
        "enabled": True,
        **best,
        "always_deep_nll": always_deep_nll,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_bundles = [
        load_bundle(item, config_path.parent, args.deep_method, args.hybrid_method)
        for item in config["train"]
    ]
    test_bundle = load_bundle(
        config["test"], config_path.parent, args.deep_method, args.hybrid_method
    )
    selection = select_model(train_bundles)
    train_rows = [row for bundle in train_bundles for row in bundle["rows"]]
    model = make_model(float(selection["alpha"]))
    model.fit(matrix(train_rows), targets(train_rows))
    prediction = model.predict(matrix(test_bundle["rows"]))
    threshold = float(selection["threshold"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    selected_test_nll = []
    deep_test_nll = []
    hybrid_test_nll = []
    for row, estimate in zip(test_bundle["rows"], prediction):
        use_hybrid = bool(estimate < -threshold)
        source = row["hybrid_row"] if use_hybrid else row["deep_row"]
        output_rows.append(
            {
                "method": "frozen_action_router",
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "selected_action": args.hybrid_method if use_hybrid else args.deep_method,
                "predicted_hybrid_minus_deep_nll": float(estimate),
                "selected_block_ids": source["selected_block_ids"],
                "ranked_block_ids": source["ranked_block_ids"],
            }
        )
        if row["deep_nll"] is not None and row["hybrid_nll"] is not None:
            deep_test_nll.append(float(row["deep_nll"]))
            hybrid_test_nll.append(float(row["hybrid_nll"]))
            selected_test_nll.append(
                float(row["hybrid_nll"] if use_hybrid else row["deep_nll"])
            )
    with (output_dir / "query_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary: dict[str, Any] = {
        "train_bundles": [bundle["name"] for bundle in train_bundles],
        "test_bundle": test_bundle["name"],
        "train_queries": len(train_rows),
        "test_queries": len(test_bundle["rows"]),
        "feature_names": FEATURE_NAMES,
        "selection_uses_corpus_disjoint_oof_only": True,
        "test_answers_used_for_routing_or_selection": False,
        "router_enabled": selection["enabled"],
        "alpha": selection["alpha"],
        "threshold": None if not np.isfinite(threshold) else threshold,
        "oof_mean_nll": selection["oof_mean_nll"],
        "oof_always_deep_nll": selection["always_deep_nll"],
        "oof_switch_rate": selection["oof_switch_rate"],
        "test_switch_rate": float(np.mean(prediction < -threshold)),
    }
    if selected_test_nll:
        summary.update(
            {
                "test_deep_mean_nll": float(np.mean(deep_test_nll)),
                "test_hybrid_mean_nll": float(np.mean(hybrid_test_nll)),
                "test_router_mean_nll": float(np.mean(selected_test_nll)),
                "test_oracle_action_mean_nll": float(
                    np.mean(np.minimum(deep_test_nll, hybrid_test_nll))
                ),
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "selection": selection,
            "deep_method": args.deep_method,
            "hybrid_method": args.hybrid_method,
        },
        output_dir / "router.joblib",
        compress=3,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
