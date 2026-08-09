from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_natural_operator_library import (
    cluster_bootstrap_ci,
    jaccard,
    overlap_fraction,
    question_features,
    ridge_predict,
    standardize_apply,
    standardize_fit,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a specialist tail-regret gate from per-head QK confidence and freeze it "
            "on a disjoint-query holdout."
        )
    )
    parser.add_argument("--calibration_per_head_topk", required=True)
    parser.add_argument("--calibration_queries_jsonl", required=True)
    parser.add_argument("--calibration_policies", required=True)
    parser.add_argument("--calibration_specialist_results", required=True)
    parser.add_argument("--calibration_deep_results", required=True)
    parser.add_argument("--calibration_nll_rows", action="append", required=True)
    parser.add_argument("--calibration_specialist_mode", required=True)
    parser.add_argument("--calibration_deep_mode", required=True)
    parser.add_argument("--target_per_head_topk", required=True)
    parser.add_argument("--target_queries_jsonl", required=True)
    parser.add_argument("--target_policy_summary", required=True)
    parser.add_argument("--target_specialist_results", required=True)
    parser.add_argument("--target_deep_results", required=True)
    parser.add_argument("--target_nll_rows", required=True)
    parser.add_argument("--target_specialist_mode", required=True)
    parser.add_argument("--target_deep_mode", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphas", default="0,0.1,1,10,100,1000")
    parser.add_argument("--thresholds", default="0,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--risk_zs", default="0,0.5,1,1.5,2")
    parser.add_argument("--tail_weight", type=float, default=0.2)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_nll(paths: Sequence[str], mode: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for raw_path in paths:
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["mode"] == mode:
                    values[int(row["query_id"])] = float(row["answer_nll"])
    if not values:
        raise ValueError(f"no NLL rows for {mode}")
    return values


def read_rankings(path: Path, mode: str) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == mode:
                rankings[int(row["query_id"])] = [
                    int(item) for item in json.loads(row["ranked_block_ids"])
                ]
    if not rankings:
        raise ValueError(f"no rankings for {mode} in {path}")
    return rankings


def read_calibration_policies(path: Path, method: str) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            policies[row["heldout_dataset"]] = {
                "head_count": int(row["head_count"]),
                "depth": int(row["depth"]),
                "bm25_quota": int(row["bm25_quota"]),
                "aggregation": row["aggregation"],
                "top_specialists": json.loads(row["top_specialists"]),
            }
    if not policies:
        raise ValueError(f"no policies for {method}")
    return policies


def aggregate(values: Sequence[float], stem: str) -> tuple[list[str], list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return (
        [f"{stem}:mean", f"{stem}:min", f"{stem}:max", f"{stem}:std"],
        [float(array.mean()), float(array.min()), float(array.max()), float(array.std())],
    )


def head_policy_features(
    query_id: int,
    query: dict[str, Any],
    scores: np.ndarray,
    block_ids: np.ndarray,
    policy: dict[str, Any],
    specialist_ranking: Sequence[int],
    deep_ranking: Sequence[int],
) -> tuple[list[str], list[float]]:
    names, values = question_features(str(query["question"]))
    selected = policy["top_specialists"][: int(policy["head_count"])]
    depth = min(int(policy["depth"]), scores.shape[-1])
    metric_values: dict[str, list[float]] = {
        "top1": [],
        "margin12": [],
        "margin14": [],
        "spread": [],
        "normalized_margin12": [],
        "entropy": [],
    }
    head_sets: list[set[int]] = []
    top1_blocks: list[int] = []
    for specialist in selected:
        layer = int(specialist["layer"])
        head = int(specialist["query_head"])
        current = np.asarray(scores[query_id, layer, head], dtype=np.float64)
        shifted = current - current.max()
        probabilities = np.exp(shifted)
        probabilities /= max(float(probabilities.sum()), 1.0e-30)
        top1 = float(current[0])
        margin12 = float(current[0] - current[min(1, len(current) - 1)])
        margin14 = float(current[0] - current[min(3, len(current) - 1)])
        spread = float(current.std())
        metric_values["top1"].append(top1)
        metric_values["margin12"].append(margin12)
        metric_values["margin14"].append(margin14)
        metric_values["spread"].append(spread)
        metric_values["normalized_margin12"].append(margin12 / max(abs(top1), 1.0e-6))
        metric_values["entropy"].append(
            float(-(probabilities * np.log(probabilities + 1.0e-30)).sum())
        )
        current_blocks = [
            int(item) for item in block_ids[query_id, layer, head, :depth].tolist()
        ]
        head_sets.append(set(current_blocks))
        top1_blocks.append(current_blocks[0])
    for metric, current_values in metric_values.items():
        current_names, current_features = aggregate(current_values, f"head:{metric}")
        names.extend(current_names)
        values.extend(current_features)
    pairwise = [
        jaccard(head_sets[left], head_sets[right])
        for left in range(len(head_sets))
        for right in range(left + 1, len(head_sets))
    ] or [1.0]
    current_names, current_features = aggregate(pairwise, "head:pairwise_jaccard")
    names.extend(current_names)
    values.extend(current_features)
    names.extend(
        [
            "head:top1_unique_ratio",
            "policy:head_count",
            "policy:depth",
            "policy:bm25_quota",
            "policy:is_minority_max",
            "retrieval:selected_jaccard",
            "retrieval:top5_overlap",
            "retrieval:top10_overlap",
            "retrieval:top39_overlap",
            "retrieval:top1_agree",
        ]
    )
    values.extend(
        [
            len(set(top1_blocks)) / max(len(top1_blocks), 1),
            float(policy["head_count"]),
            float(policy["depth"]),
            float(policy["bm25_quota"]),
            float(policy["aggregation"] == "minority_max"),
            jaccard(specialist_ranking, deep_ranking),
            overlap_fraction(specialist_ranking, deep_ranking, 5),
            overlap_fraction(specialist_ranking, deep_ranking, 10),
            overlap_fraction(specialist_ranking, deep_ranking, 39),
            float(specialist_ranking[:1] == deep_ranking[:1]),
        ]
    )
    return names, values


def build_matrix(
    queries: list[dict[str, Any]],
    scores: np.ndarray,
    block_ids: np.ndarray,
    policy_for_dataset: dict[str, dict[str, Any]] | None,
    fixed_policy: dict[str, Any] | None,
    specialist_rankings: dict[int, list[int]],
    deep_rankings: dict[int, list[int]],
) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for query_id, query in enumerate(queries):
        policy = (
            fixed_policy
            if fixed_policy is not None
            else policy_for_dataset[str(query["dataset"])]
        )
        names, values = head_policy_features(
            query_id,
            query,
            scores,
            block_ids,
            policy,
            specialist_rankings[query_id],
            deep_rankings[query_id],
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("feature layout changed")
        rows.append(values)
    if feature_names is None:
        raise ValueError("no queries")
    return np.asarray(rows, dtype=np.float64), feature_names


def tune_gate(
    matrix: np.ndarray,
    regret: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    thresholds: Sequence[float],
    risk_zs: Sequence[float],
    tail_weight: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for alpha in alphas:
        for threshold in thresholds:
            for risk_z in risk_zs:
                selected_regret = np.zeros(len(regret), dtype=np.float64)
                switches = np.zeros(len(regret), dtype=bool)
                for heldout in sorted(set(groups.tolist())):
                    train = groups != heldout
                    test = ~train
                    train_x, mean, scale = standardize_fit(matrix[train])
                    test_x = standardize_apply(matrix[test], mean, scale)
                    prediction, residual = ridge_predict(
                        train_x, regret[train, None], test_x, alpha
                    )
                    conservative = prediction[:, 0] + risk_z * residual[0]
                    current_switch = conservative <= -threshold
                    test_rows = np.flatnonzero(test)
                    switches[test_rows] = current_switch
                    selected_regret[test_rows[current_switch]] = regret[
                        test_rows[current_switch]
                    ]
                positive_tail = max(float(np.quantile(selected_regret, 0.95)), 0.0)
                objective = float(selected_regret.mean() + tail_weight * positive_tail)
                row = {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "risk_z": float(risk_z),
                    "objective": objective,
                    "mean_regret": float(selected_regret.mean()),
                    "p95_regret": float(np.quantile(selected_regret, 0.95)),
                    "switch_rate": float(switches.mean()),
                }
                if best is None or (
                    row["objective"], -row["switch_rate"], row["risk_z"], row["alpha"]
                ) < (
                    best["objective"],
                    -best["switch_rate"],
                    best["risk_z"],
                    best["alpha"],
                ):
                    best = row
    if best is None:
        raise ValueError("empty gate grid")
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_queries = read_jsonl(Path(args.calibration_queries_jsonl))
    target_queries = read_jsonl(Path(args.target_queries_jsonl))
    calibration_uids = {str(query["record_uid"]) for query in calibration_queries}
    target_uids = {str(query["record_uid"]) for query in target_queries}
    if calibration_uids & target_uids:
        raise ValueError("calibration and target queries overlap")
    calibration_payload = np.load(args.calibration_per_head_topk)
    target_payload = np.load(args.target_per_head_topk)
    calibration_policies = read_calibration_policies(
        Path(args.calibration_policies), args.calibration_specialist_mode
    )
    target_summary = json.loads(Path(args.target_policy_summary).read_text(encoding="utf-8"))
    target_policy = {
        **target_summary["action"],
        "top_specialists": target_summary["top_specialists"],
    }
    calibration_specialist_rankings = read_rankings(
        Path(args.calibration_specialist_results), args.calibration_specialist_mode
    )
    calibration_deep_rankings = read_rankings(
        Path(args.calibration_deep_results), args.calibration_deep_mode
    )
    target_specialist_rankings = read_rankings(
        Path(args.target_specialist_results), args.target_specialist_mode
    )
    target_deep_rankings = read_rankings(
        Path(args.target_deep_results), args.target_deep_mode
    )
    calibration_x, feature_names = build_matrix(
        calibration_queries,
        calibration_payload["scores"],
        calibration_payload["block_ids"],
        calibration_policies,
        None,
        calibration_specialist_rankings,
        calibration_deep_rankings,
    )
    target_x, target_feature_names = build_matrix(
        target_queries,
        target_payload["scores"],
        target_payload["block_ids"],
        None,
        target_policy,
        target_specialist_rankings,
        target_deep_rankings,
    )
    if feature_names != target_feature_names:
        raise AssertionError("calibration/target feature layouts disagree")
    calibration_specialist_nll = read_nll(
        args.calibration_nll_rows, args.calibration_specialist_mode
    )
    calibration_deep_nll = read_nll(
        args.calibration_nll_rows, args.calibration_deep_mode
    )
    target_specialist_nll = read_nll(
        [args.target_nll_rows], args.target_specialist_mode
    )
    target_deep_nll = read_nll([args.target_nll_rows], args.target_deep_mode)
    calibration_regret = np.asarray(
        [
            calibration_specialist_nll[index] - calibration_deep_nll[index]
            for index in range(len(calibration_queries))
        ]
    )
    target_regret = np.asarray(
        [
            target_specialist_nll[index] - target_deep_nll[index]
            for index in range(len(target_queries))
        ]
    )
    groups = np.asarray([str(query["dataset"]) for query in calibration_queries])
    target_groups = np.asarray([str(query["dataset"]) for query in target_queries])
    hyper = tune_gate(
        calibration_x,
        calibration_regret,
        groups,
        parse_floats(args.alphas),
        parse_floats(args.thresholds),
        parse_floats(args.risk_zs),
        args.tail_weight,
    )
    train_x, mean, scale = standardize_fit(calibration_x)
    test_x = standardize_apply(target_x, mean, scale)
    prediction, residual = ridge_predict(
        train_x, calibration_regret[:, None], test_x, hyper["alpha"]
    )
    conservative = prediction[:, 0] + hyper["risk_z"] * residual[0]
    switches = conservative <= -hyper["threshold"]
    selected_regret = np.where(switches, target_regret, 0.0)
    deep = np.asarray([target_deep_nll[index] for index in range(len(target_queries))])
    selected_nll = deep + selected_regret
    rng = np.random.default_rng(args.seed)
    query_bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
    for index in range(args.bootstrap_samples):
        sample = rng.integers(0, len(selected_regret), len(selected_regret))
        query_bootstrap[index] = selected_regret[sample].mean()
    cluster_ci = cluster_bootstrap_ci(
        selected_regret, target_groups, args.bootstrap_samples, rng
    )
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(target_queries):
        rows.append(
            {
                "query_id": index,
                "dataset": query["dataset"],
                "predicted_regret": float(prediction[index, 0]),
                "conservative_regret": float(conservative[index]),
                "switched_to_specialist": bool(switches[index]),
                "actual_specialist_regret": float(target_regret[index]),
                "selected_nll": float(selected_nll[index]),
                "deep_nll": float(deep[index]),
            }
        )
    summary = {
        "source": "frozen head-QK confidence tail-regret gate",
        "calibration_queries": len(calibration_queries),
        "target_queries": len(target_queries),
        "query_uid_overlap": 0,
        "feature_count": len(feature_names),
        "features": feature_names,
        "hyperparameters": hyper,
        "target": {
            "deep_mean_nll": float(deep.mean()),
            "selected_mean_nll": float(selected_nll.mean()),
            "mean_delta_vs_deep": float(selected_regret.mean()),
            "query_ci95": [
                float(np.quantile(query_bootstrap, 0.025)),
                float(np.quantile(query_bootstrap, 0.975)),
            ],
            "dataset_cluster_ci95": list(cluster_ci),
            "switches": int(switches.sum()),
            "switch_rate": float(switches.mean()),
            "switch_win_rate": float(np.mean(target_regret[switches] < 0.0))
            if switches.any()
            else 0.0,
            "p95_selected_regret": float(np.quantile(selected_regret, 0.95)),
        },
        "interpretation": (
            "The gate uses only query text, selected-head score shapes/agreement, policy size, "
            "and specialist/deep block disagreement. It never sees a target answer or target NLL."
        ),
    }
    write_csv(output_dir / "target_rows.csv", rows)
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

