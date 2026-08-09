from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METHOD_PATTERN = re.compile(r"^hier_bm25_scope(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether scope-score geometry supports train-only adaptive Top-D."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_adaptive_rows", required=True)
    parser.add_argument("--ppl128_rows")
    parser.add_argument("--ppl512_rows")
    parser.add_argument("--depths", default="1,3,8,16,32")
    parser.add_argument("--confidence_thresholds", default="0.70,0.80,0.90")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("depths must be positive")
    return values


def parse_floats(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0 or max(values) >= 1:
        raise ValueError("confidence thresholds must lie in (0, 1)")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def reader_analysis(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    lookup = {
        (int(row["query_id"]), str(row["method"])): row for row in rows
    }
    methods = sorted({str(row["method"]) for row in rows})
    quality = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        micro_nll = sum(float(row["total_nll"]) for row in group) / sum(
            int(row["target_tokens"]) for row in group
        )
        quality[method] = {
            "ppl": math.exp(micro_nll),
            "micro_nll": micro_nll,
            "same_scope_any": mean(
                [float(row["same_scope_any"]) for row in group]
            ),
            "same_scope_fraction": mean(
                [float(row["same_scope_fraction"]) for row in group]
            ),
        }

    pairs = (
        ("adaptive_scope_geometry_c70", "hier_bm25_scope3"),
        ("adaptive_scope_geometry_c70", "hier_bm25_scope8"),
        ("adaptive_scope_geometry_c80", "hier_bm25_scope8"),
        ("adaptive_scope_geometry_c90", "hier_bm25_scope8"),
        ("hier_bm25_scope16", "hier_bm25_scope8"),
        ("hier_bm25_scope32", "hier_bm25_scope8"),
        ("adaptive_scope_geometry_c70", "global_bm25_unigram"),
        ("adaptive_scope_geometry_c80", "global_bm25_unigram"),
    )
    comparisons = []
    for pair_index, (method_a, method_b) in enumerate(pairs):
        query_ids = sorted(
            query_id
            for query_id, method in lookup
            if method == method_a and (query_id, method_b) in lookup
        )
        differences = [
            float(lookup[(query_id, method_a)]["mean_nll"])
            - float(lookup[(query_id, method_b)]["mean_nll"])
            for query_id in query_ids
        ]
        comparisons.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "meaning": "negative favors method_a",
                "queries": len(differences),
                "mean_nll_a_minus_b": mean(differences),
                "bootstrap95": bootstrap_mean_ci(
                    differences, samples=50_000, seed=seed + pair_index
                ),
                "a_wins": sum(value < 0 for value in differences),
                "b_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return {"quality": quality, "paired_comparisons": comparisons}


def structural_features(row: dict[str, Any]) -> list[float]:
    return [
        math.log(float(row["memory_tokens"])),
        math.log(float(row["prefix_tokens"])),
        math.log1p(float(row["scope_query_features"])),
        math.log1p(float(row["active_scopes"])),
    ]


def geometry_features(row: dict[str, Any]) -> list[float]:
    top1 = max(abs(float(row["scope_top1_score"])), 1.0e-8)
    return [
        *structural_features(row),
        math.log1p(float(row["positive_scope_scores"])),
        float(row["scope_top1_score"]) / max(float(row["scope_query_features"]), 1.0),
        float(row["scope_normalized_margin_1_2"]),
        float(row["scope_margin_3_4"]) / top1,
        float(row["scope_margin_8_9"]) / top1,
        float(row["scope_margin_16_17"]) / top1,
        float(row["scope_margin_32_33"]) / top1,
        float(row["scope_top1_positive_share"]),
        float(row["scope_top3_positive_share"]),
        float(row["scope_top8_positive_share"]),
        float(row["scope_top16_positive_share"]),
        float(row["scope_top32_positive_share"]),
        float(row["scope_score_normalized_entropy"]),
        float(row["scope_score_hhi"]),
        float(row["scope_top1_z"]),
    ]


def fit_predict_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    classes = np.unique(y_train)
    if len(classes) == 1:
        return np.full(len(x_test), float(classes[0]), dtype=np.float64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def evaluate_predictions(
    examples: list[dict[str, Any]],
    probabilities: np.ndarray,
    depths: list[int],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(examples), dtype=bool)
    true_ranks = np.asarray([int(row["true_scope_rank"]) for row in examples])
    output = {}
    for depth_index, depth in enumerate(depths):
        labels = (true_ranks <= depth).astype(np.int64)[mask]
        scores = probabilities[mask, depth_index]
        output[str(depth)] = {
            "examples": len(labels),
            "positive_rate": float(labels.mean()),
            "auc": safe_auc(labels, scores),
            "brier": float(brier_score_loss(labels, scores)),
        }
    aucs = [item["auc"] for item in output.values() if item["auc"] is not None]
    return {"by_depth": output, "macro_auc": mean([float(item) for item in aucs])}


def summarize_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (int(row["memory_tokens"]), int(row["prefix_tokens"]), str(row["method"]))
            for row in rows
        }
    )
    for memory_tokens, suffix, method in keys:
        group = [
            row
            for row in rows
            if int(row["memory_tokens"]) == memory_tokens
            and int(row["prefix_tokens"]) == suffix
            and str(row["method"]) == method
        ]
        depth_counts = Counter(int(row["chosen_scope_depth"]) for row in group)
        output.append(
            {
                "memory_tokens": memory_tokens,
                "state_suffix_tokens": suffix,
                "method": method,
                "queries": len(group),
                "mean_chosen_scope_depth": mean(
                    [float(row["chosen_scope_depth"]) for row in group]
                ),
                "chosen_depth_counts": {
                    str(depth): depth_counts[depth] for depth in sorted(depth_counts)
                },
                "scope_router_recall": mean(
                    [float(row["scope_router_hit"]) for row in group]
                ),
                "same_scope_any_at_8": mean(
                    [float(row["same_scope_any_at_8"]) for row in group]
                ),
                "same_scope_fraction_at_8": mean(
                    [float(row["same_scope_fraction_at_8"]) for row in group]
                ),
                "mean_candidate_blocks": mean(
                    [float(row["candidate_blocks"]) for row in group]
                ),
                "mean_candidate_fraction": mean(
                    [float(row["candidate_fraction"]) for row in group]
                ),
                "mean_query_seconds": mean(
                    [float(row["query_seconds"]) for row in group]
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    depths = parse_ints(args.depths)
    thresholds = parse_floats(args.confidence_thresholds)
    rows = read_jsonl(args.rows)
    depth_rows: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        match = METHOD_PATTERN.match(str(row["method"]))
        if not match:
            continue
        depth = int(match.group(1))
        if depth not in depths:
            continue
        key = (
            int(row["query_id"]),
            int(row["memory_tokens"]),
            int(row["prefix_tokens"]),
            depth,
        )
        depth_rows[key] = row

    examples = []
    for query_id, memory_tokens, suffix, depth in sorted(depth_rows):
        if depth != depths[0]:
            continue
        row = depth_rows[(query_id, memory_tokens, suffix, depth)]
        if row.get("true_scope_rank") is None:
            continue
        examples.append(row)
    expected = 30 * 4 * 4
    if len(examples) != expected:
        raise RuntimeError(f"expected {expected} examples, found {len(examples)}")

    groups = np.asarray([int(row["query_id"]) for row in examples], dtype=np.int64)
    true_ranks = np.asarray([int(row["true_scope_rank"]) for row in examples])
    feature_sets = {
        "structural": np.asarray(
            [structural_features(row) for row in examples], dtype=np.float64
        ),
        "score_geometry": np.asarray(
            [geometry_features(row) for row in examples], dtype=np.float64
        ),
    }
    predictions = {
        name: np.zeros((len(examples), len(depths)), dtype=np.float64)
        for name in feature_sets
    }
    fold_assignments = np.full(len(examples), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)
    dummy = np.zeros(len(examples))
    for fold, (train_indices, test_indices) in enumerate(splitter.split(dummy, groups=groups)):
        fold_assignments[test_indices] = fold
        for feature_name, features in feature_sets.items():
            for depth_index, depth in enumerate(depths):
                labels = (true_ranks <= depth).astype(np.int64)
                predictions[feature_name][test_indices, depth_index] = fit_predict_binary(
                    features[train_indices],
                    labels[train_indices],
                    features[test_indices],
                    seed=args.seed + fold * 100 + depth_index,
                )
    if np.any(fold_assignments < 0):
        raise RuntimeError("some examples received no out-of-fold prediction")
    for feature_name in predictions:
        predictions[feature_name] = np.maximum.accumulate(
            predictions[feature_name], axis=1
        )

    masks = {
        "all_scales": np.ones(len(examples), dtype=bool),
        "100m": np.asarray(
            [int(row["memory_tokens"]) == 100_000_000 for row in examples]
        ),
    }
    calibration = {
        feature_name: {
            mask_name: evaluate_predictions(
                examples, probabilities, depths, mask=mask
            )
            for mask_name, mask in masks.items()
        }
        for feature_name, probabilities in predictions.items()
    }

    adaptive_rows = []
    geometry_probabilities = predictions["score_geometry"]
    for example_index, example in enumerate(examples):
        query_id = int(example["query_id"])
        memory_tokens = int(example["memory_tokens"])
        suffix = int(example["prefix_tokens"])
        probabilities = geometry_probabilities[example_index]
        for threshold in thresholds:
            eligible = [
                depth
                for depth, probability in zip(depths, probabilities)
                if float(probability) >= threshold
            ]
            chosen_depth = eligible[0] if eligible else depths[-1]
            source = depth_rows[(query_id, memory_tokens, suffix, chosen_depth)]
            method = f"adaptive_scope_geometry_c{int(round(threshold * 100)):02d}"
            adaptive_rows.append(
                {
                    **source,
                    "method": method,
                    "chosen_scope_depth": chosen_depth,
                    "predicted_scope_hit_probabilities": {
                        str(depth): float(probability)
                        for depth, probability in zip(depths, probabilities)
                    },
                    "confidence_threshold": threshold,
                    "out_of_fold": True,
                    "grouped_by_query_id": True,
                    "train_uses_target": False,
                    "selection_uses_target": False,
                }
            )

    fixed_summary = []
    for depth in depths:
        method = f"hier_bm25_scope{depth}"
        fixed_rows = [row for row in rows if row["method"] == method]
        for row in fixed_rows:
            row["chosen_scope_depth"] = depth
        fixed_summary.extend(summarize_policy(fixed_rows))

    output_summary = {
        "source": "grouped out-of-fold scope confidence and adaptive Top-D",
        "protocol": {
            "examples": len(examples),
            "query_groups": len(set(groups.tolist())),
            "group_folds": args.folds,
            "same_query_never_crosses_train_test_within_fold": True,
            "memory_scales": sorted(
                {int(row["memory_tokens"]) for row in examples}
            ),
            "state_suffix_tokens": sorted(
                {int(row["prefix_tokens"]) for row in examples}
            ),
            "depths": depths,
            "confidence_thresholds": thresholds,
            "features_use_target": False,
            "labels_use_train_query_scope_only": True,
            "selection_uses_target": False,
            "final_reader_blocks": 8,
        },
        "calibration": calibration,
        "fixed_depth_quality": fixed_summary,
        "adaptive_quality": summarize_policy(adaptive_rows),
    }
    if bool(args.ppl128_rows) != bool(args.ppl512_rows):
        raise ValueError("ppl128_rows and ppl512_rows must be provided together")
    if args.ppl128_rows and args.ppl512_rows:
        output_summary["reader"] = {
            "128": reader_analysis(read_jsonl(args.ppl128_rows), seed=args.seed + 1000),
            "512": reader_analysis(read_jsonl(args.ppl512_rows), seed=args.seed + 2000),
        }
    output_summary_path = Path(args.output_summary)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    adaptive_path = Path(args.output_adaptive_rows)
    adaptive_path.parent.mkdir(parents=True, exist_ok=True)
    with adaptive_path.open("w", encoding="utf-8") as handle:
        for row in adaptive_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
