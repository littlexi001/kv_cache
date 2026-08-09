from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit natural-task operator complementarity and run a strict "
            "leave-one-dataset-out, query-only regret router."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as alias=mode=/path/to/answer_nll_rows.csv.",
    )
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument(
        "--retrieval_results",
        action="append",
        default=[],
        help="query_results.csv used only for pre-gather disagreement features.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--thresholds", default="0,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--risk_zs", default="0,0.5,1")
    parser.add_argument("--tail_weight", type=float, default=0.1)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--make_plots", type=parse_bool, default=True)
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_candidate(spec: str) -> tuple[str, str, Path]:
    pieces = spec.split("=", 2)
    if len(pieces) != 3:
        raise ValueError("candidate must be alias=mode=/path/to/csv")
    alias, mode, raw_path = pieces
    if not alias or not mode or not raw_path:
        raise ValueError("candidate alias, mode and path must be non-empty")
    return alias, mode, Path(raw_path)


def load_candidate(spec: str) -> tuple[str, str, dict[int, dict[str, Any]]]:
    alias, mode, path = parse_candidate(spec)
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["mode"] != mode:
                continue
            query_id = int(raw["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate query {query_id} for {alias}")
            rows[query_id] = {
                "query_id": query_id,
                "dataset": raw["dataset"],
                "answer_nll": float(raw["answer_nll"]),
                "context_blocks": int(raw["context_blocks"]),
                "context_tokens": int(raw["context_tokens"]),
            }
    if not rows:
        raise ValueError(f"no rows for mode {mode} in {path}")
    return alias, mode, rows


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in json.loads(raw)]


def load_retrieval_rows(paths: Sequence[str]) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_path in paths:
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                key = (int(raw["query_id"]), raw["method"])
                row = {
                    "record_margin": float(raw.get("record_margin", 0.0) or 0.0),
                    "selected": parse_int_list(raw["selected_block_ids"]),
                    "ranked": parse_int_list(raw["ranked_block_ids"]),
                }
                if key in rows:
                    previous = rows[key]
                    if previous["selected"] != row["selected"]:
                        raise ValueError(f"conflicting retrieval rows for {key}")
                    continue
                rows[key] = row
    return rows


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def question_features(question: str) -> tuple[list[str], list[float]]:
    words = TOKEN_RE.findall(question)
    lower = [word.lower() for word in words]
    chars = max(len(question), 1)
    word_count = max(len(words), 1)
    unique_ratio = len(set(lower)) / word_count
    avg_word_length = sum(len(word) for word in words) / word_count
    digit_chars = sum(character.isdigit() for character in question)
    upper_tokens = sum(word[:1].isupper() for word in words)
    punctuation = sum(not character.isalnum() and not character.isspace() for character in question)
    numbers = sum(any(character.isdigit() for character in word) for word in words)
    connector_words = {"and", "or", "before", "after", "between", "both", "same", "first", "last"}
    comparison_words = {"more", "less", "higher", "lower", "earlier", "later", "difference", "compare"}
    names = [
        "q_log_words",
        "q_log_chars",
        "q_unique_ratio",
        "q_avg_word_length",
        "q_digit_ratio",
        "q_upper_token_ratio",
        "q_punctuation_ratio",
        "q_number_token_ratio",
        "q_connector_ratio",
        "q_comparison_ratio",
        "q_has_quotes",
        "q_has_parentheses",
    ]
    values = [
        math.log1p(len(words)),
        math.log1p(len(question)),
        unique_ratio,
        avg_word_length,
        digit_chars / chars,
        upper_tokens / word_count,
        punctuation / chars,
        numbers / word_count,
        sum(word in connector_words for word in lower) / word_count,
        sum(word in comparison_words for word in lower) / word_count,
        float(any(character in question for character in {'"', "'", "“", "”"})),
        float("(" in question or ")" in question),
    ]
    for cue in ["what", "who", "where", "when", "why", "how", "which", "is", "are", "does", "did"]:
        names.append(f"q_starts_{cue}")
        values.append(float(bool(lower) and lower[0] == cue))
    return names, values


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def overlap_fraction(left: Sequence[int], right: Sequence[int], depth: int) -> float:
    left_prefix = set(left[:depth])
    right_prefix = set(right[:depth])
    denominator = max(min(depth, len(left), len(right)), 1)
    return len(left_prefix & right_prefix) / denominator


def build_features(
    query_ids: Sequence[int],
    queries: dict[int, dict[str, Any]],
    actions: Sequence[str],
    retrieval_rows: dict[tuple[int, str], dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    feature_names: list[str] | None = None
    matrix: list[list[float]] = []
    for query_id in query_ids:
        q_names, q_values = question_features(str(queries[query_id]["question"]))
        names = list(q_names)
        values = list(q_values)
        available = [action for action in actions if (query_id, action) in retrieval_rows]
        if len(available) != len(actions):
            missing = sorted(set(actions) - set(available))
            raise ValueError(f"missing retrieval features for query {query_id}: {missing}")
        for action in actions:
            row = retrieval_rows[(query_id, action)]
            margin = float(row["record_margin"])
            names.extend([f"{action}:margin", f"{action}:log_abs_margin"])
            values.extend([margin, math.log1p(abs(margin))])
        for left_index, left in enumerate(actions):
            left_row = retrieval_rows[(query_id, left)]
            for right in actions[left_index + 1 :]:
                right_row = retrieval_rows[(query_id, right)]
                stem = f"{left}|{right}"
                names.extend(
                    [
                        f"{stem}:selected_jaccard",
                        f"{stem}:top5_overlap",
                        f"{stem}:top10_overlap",
                        f"{stem}:top1_agree",
                    ]
                )
                values.extend(
                    [
                        jaccard(left_row["selected"], right_row["selected"]),
                        overlap_fraction(left_row["ranked"], right_row["ranked"], 5),
                        overlap_fraction(left_row["ranked"], right_row["ranked"], 10),
                        float(left_row["ranked"][:1] == right_row["ranked"][:1]),
                    ]
                )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("feature layout changed across queries")
        matrix.append(values)
    if feature_names is None:
        raise ValueError("no queries")
    return np.asarray(matrix, dtype=np.float64), feature_names


def standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return (matrix - mean) / scale, mean, scale


def standardize_apply(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix - mean) / scale


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y
    prediction = test_design @ weights
    residual = train_y - design @ weights
    residual_scale = np.sqrt(np.mean(np.square(residual), axis=0))
    return prediction, residual_scale


def route_from_predictions(
    predicted_delta: np.ndarray,
    residual_scale: np.ndarray,
    baseline_index: int,
    threshold: float,
    risk_z: float,
) -> np.ndarray:
    conservative = predicted_delta + risk_z * residual_scale[None, :]
    conservative[:, baseline_index] = 0.0
    best = np.argmin(conservative, axis=1)
    routed = np.full(len(predicted_delta), baseline_index, dtype=np.int64)
    improvement = -conservative[np.arange(len(predicted_delta)), best]
    accepted = (best != baseline_index) & (improvement >= threshold)
    routed[accepted] = best[accepted]
    return routed


def inner_predictions(
    matrix: np.ndarray,
    nll: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    threshold: float,
    risk_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routed_actions = np.zeros(len(matrix), dtype=np.int64)
    routed_nll = np.zeros(len(matrix), dtype=np.float64)
    baseline_nll = np.zeros(len(matrix), dtype=np.float64)
    for heldout_group in sorted(set(groups.tolist())):
        train = groups != heldout_group
        test = ~train
        baseline_index = int(np.argmin(nll[train].mean(axis=0)))
        train_x, mean, scale = standardize_fit(matrix[train])
        test_x = standardize_apply(matrix[test], mean, scale)
        target = nll[train] - nll[train, baseline_index][:, None]
        prediction, residual_scale = ridge_predict(train_x, target, test_x, alpha)
        actions = route_from_predictions(
            prediction, residual_scale, baseline_index, threshold, risk_z
        )
        test_rows = np.flatnonzero(test)
        routed_actions[test_rows] = actions
        routed_nll[test_rows] = nll[test_rows, actions]
        baseline_nll[test_rows] = nll[test_rows, baseline_index]
    return routed_actions, routed_nll, baseline_nll


def tune_router(
    matrix: np.ndarray,
    nll: np.ndarray,
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
                _, routed, baseline = inner_predictions(
                    matrix, nll, groups, alpha, threshold, risk_z
                )
                regret = routed - baseline
                positive_tail = max(float(np.quantile(regret, 0.95)), 0.0)
                objective = float(routed.mean() + tail_weight * positive_tail)
                row = {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "risk_z": float(risk_z),
                    "objective": objective,
                    "mean_nll": float(routed.mean()),
                    "p95_regret": float(np.quantile(regret, 0.95)),
                }
                if best is None or (
                    row["objective"], row["threshold"], row["risk_z"], row["alpha"]
                ) < (
                    best["objective"],
                    best["threshold"],
                    best["risk_z"],
                    best["alpha"],
                ):
                    best = row
    if best is None:
        raise ValueError("empty hyperparameter grid")
    return best


def greedy_oracle_curve(nll: np.ndarray, actions: Sequence[str]) -> list[dict[str, Any]]:
    means = nll.mean(axis=0)
    selected = [int(np.argmin(means))]
    rows: list[dict[str, Any]] = []
    while True:
        oracle = nll[:, selected].min(axis=1)
        rows.append(
            {
                "library_size": len(selected),
                "added_action": actions[selected[-1]],
                "selected_actions": json.dumps([actions[index] for index in selected]),
                "oracle_mean_nll": float(oracle.mean()),
                "headroom_vs_best_global": float(means.min() - oracle.mean()),
            }
        )
        remaining = [index for index in range(len(actions)) if index not in selected]
        if not remaining:
            break
        next_index = min(
            remaining,
            key=lambda index: float(np.minimum(oracle, nll[:, index]).mean()),
        )
        selected.append(next_index)
    return rows


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        pieces = [values[groups == group] for group in sampled_groups]
        bootstrap[index] = np.concatenate(pieces).mean()
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def make_plots(
    output_dir: Path,
    oracle_curve: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        [row["library_size"] for row in oracle_curve],
        [row["oracle_mean_nll"] for row in oracle_curve],
        marker="o",
    )
    axis.set_xlabel("Greedy operator-library size")
    axis.set_ylabel("Per-query oracle mean answer NLL")
    axis.set_title("Natural-task operator complementarity")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = plot_dir / "oracle_headroom_by_library_size.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    labels = [row["method"] for row in comparison_rows]
    values = [row["mean_nll"] for row in comparison_rows]
    colors = ["#4c78a8" if label != "query_regret_router" else "#e45756" for label in labels]
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Mean answer NLL")
    axis.set_title("Leave-one-dataset-out natural routing")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = plot_dir / "lodo_router_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [load_candidate(spec) for spec in args.candidate]
    aliases = [alias for alias, _, _ in payloads]
    modes = [mode for _, mode, _ in payloads]
    if len(set(aliases)) != len(aliases):
        raise ValueError("candidate aliases must be unique")
    query_id_sets = [set(rows) for _, _, rows in payloads]
    if any(query_ids != query_id_sets[0] for query_ids in query_id_sets[1:]):
        raise ValueError("candidate files do not cover identical query IDs")
    query_ids = sorted(query_id_sets[0])
    queries = {int(row["query_id"]): row for row in read_jsonl(Path(args.queries_jsonl))}
    if any(query_id not in queries for query_id in query_ids):
        raise ValueError("queries_jsonl does not cover all candidate query IDs")

    dataset = np.asarray([payloads[0][2][query_id]["dataset"] for query_id in query_ids])
    for _, _, rows in payloads[1:]:
        if any(rows[query_id]["dataset"] != dataset[index] for index, query_id in enumerate(query_ids)):
            raise ValueError("candidate dataset labels disagree")
    nll = np.asarray(
        [[rows[query_id]["answer_nll"] for _, _, rows in payloads] for query_id in query_ids],
        dtype=np.float64,
    )
    context_blocks = {
        alias: sorted({rows[query_id]["context_blocks"] for query_id in query_ids})
        for alias, _, rows in payloads
    }
    context_tokens = {
        alias: sorted({rows[query_id]["context_tokens"] for query_id in query_ids})
        for alias, _, rows in payloads
    }
    if len({tuple(value) for value in context_blocks.values()}) != 1:
        raise ValueError(f"candidates use unequal physical block counts: {context_blocks}")
    if len({tuple(value) for value in context_tokens.values()}) != 1:
        raise ValueError(f"candidates use unequal context token counts: {context_tokens}")

    retrieval_rows = load_retrieval_rows(args.retrieval_results)
    matrix, feature_names = build_features(query_ids, queries, modes, retrieval_rows)
    oracle_curve = greedy_oracle_curve(nll, aliases)
    candidate_rows: list[dict[str, Any]] = []
    for index, alias in enumerate(aliases):
        candidate_rows.append(
            {
                "action": alias,
                "mode": modes[index],
                "queries": len(query_ids),
                "mean_nll": float(nll[:, index].mean()),
                "median_nll": float(np.median(nll[:, index])),
                "wins": int(np.sum(nll[:, index] == nll.min(axis=1))),
                "context_blocks": json.dumps(context_blocks[alias]),
                "context_tokens": json.dumps(context_tokens[alias]),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for left in range(len(aliases)):
        for right in range(left + 1, len(aliases)):
            oracle_pair = np.minimum(nll[:, left], nll[:, right])
            best_single = min(float(nll[:, left].mean()), float(nll[:, right].mean()))
            pairwise_rows.append(
                {
                    "left": aliases[left],
                    "right": aliases[right],
                    "mean_delta_left_minus_right": float((nll[:, left] - nll[:, right]).mean()),
                    "left_win_rate": float(np.mean(nll[:, left] < nll[:, right])),
                    "tie_rate": float(np.mean(nll[:, left] == nll[:, right])),
                    "pair_oracle_mean_nll": float(oracle_pair.mean()),
                    "pair_oracle_headroom": float(best_single - oracle_pair.mean()),
                }
            )

    alphas = parse_float_list(args.alphas)
    thresholds = parse_float_list(args.thresholds)
    risk_zs = parse_float_list(args.risk_zs)
    routed_actions = np.zeros(len(query_ids), dtype=np.int64)
    routed_nll = np.zeros(len(query_ids), dtype=np.float64)
    fold_baseline_actions = np.zeros(len(query_ids), dtype=np.int64)
    fold_baseline_nll = np.zeros(len(query_ids), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for heldout_dataset in sorted(set(dataset.tolist())):
        train = dataset != heldout_dataset
        test = ~train
        hyper = tune_router(
            matrix[train],
            nll[train],
            dataset[train],
            alphas,
            thresholds,
            risk_zs,
            args.tail_weight,
        )
        baseline_index = int(np.argmin(nll[train].mean(axis=0)))
        train_x, mean, scale = standardize_fit(matrix[train])
        test_x = standardize_apply(matrix[test], mean, scale)
        target = nll[train] - nll[train, baseline_index][:, None]
        prediction, residual_scale = ridge_predict(
            train_x, target, test_x, hyper["alpha"]
        )
        actions = route_from_predictions(
            prediction,
            residual_scale,
            baseline_index,
            hyper["threshold"],
            hyper["risk_z"],
        )
        test_rows = np.flatnonzero(test)
        routed_actions[test_rows] = actions
        routed_nll[test_rows] = nll[test_rows, actions]
        fold_baseline_actions[test_rows] = baseline_index
        fold_baseline_nll[test_rows] = nll[test_rows, baseline_index]
        fold_rows.append(
            {
                "heldout_dataset": heldout_dataset,
                "train_queries": int(train.sum()),
                "test_queries": int(test.sum()),
                "baseline_action": aliases[baseline_index],
                "alpha": hyper["alpha"],
                "threshold": hyper["threshold"],
                "risk_z": hyper["risk_z"],
                "inner_objective": hyper["objective"],
                "test_router_mean_nll": float(routed_nll[test].mean()),
                "test_baseline_mean_nll": float(fold_baseline_nll[test].mean()),
                "test_mean_delta": float((routed_nll[test] - fold_baseline_nll[test]).mean()),
                "fallback_rate": float(np.mean(actions == baseline_index)),
            }
        )

    best_global_index = int(np.argmin(nll.mean(axis=0)))
    global_best = nll[:, best_global_index]
    oracle_actions = np.argmin(nll, axis=1)
    lodo_rows: list[dict[str, Any]] = []
    for row_index, query_id in enumerate(query_ids):
        lodo_rows.append(
            {
                "query_id": query_id,
                "dataset": dataset[row_index],
                "selected_action": aliases[routed_actions[row_index]],
                "baseline_action": aliases[fold_baseline_actions[row_index]],
                "oracle_action": aliases[oracle_actions[row_index]],
                "routed_nll": routed_nll[row_index],
                "fold_baseline_nll": fold_baseline_nll[row_index],
                "global_best_nll": global_best[row_index],
                "oracle_nll": nll[row_index, oracle_actions[row_index]],
                "router_regret_vs_fold_baseline": routed_nll[row_index]
                - fold_baseline_nll[row_index],
                "router_regret_vs_oracle": routed_nll[row_index]
                - nll[row_index, oracle_actions[row_index]],
            }
        )

    oracle_nll = nll.min(axis=1)
    comparison_rows = [
        {"method": alias, "mean_nll": float(nll[:, index].mean())}
        for index, alias in enumerate(aliases)
    ]
    comparison_rows.extend(
        [
            {"method": "fold_static_baseline", "mean_nll": float(fold_baseline_nll.mean())},
            {"method": "query_regret_router", "mean_nll": float(routed_nll.mean())},
            {"method": "per_query_oracle", "mean_nll": float(oracle_nll.mean())},
        ]
    )
    rng = np.random.default_rng(args.seed)
    delta_fold = routed_nll - fold_baseline_nll
    delta_global = routed_nll - global_best
    fold_ci = cluster_bootstrap_ci(
        delta_fold, dataset, args.bootstrap_samples, rng
    )
    global_ci = cluster_bootstrap_ci(
        delta_global, dataset, args.bootstrap_samples, rng
    )
    plot_paths = make_plots(output_dir, oracle_curve, comparison_rows) if args.make_plots else []

    summary = {
        "source": "natural LongBench equal-budget operator complementarity and LODO query routing",
        "queries": len(query_ids),
        "datasets": dict(sorted(Counter(dataset.tolist()).items())),
        "actions": aliases,
        "modes": modes,
        "feature_count": len(feature_names),
        "feature_policy": "query text plus pre-gather retrieval disagreement; no dataset label or gold feature",
        "best_global_action": aliases[best_global_index],
        "best_global_mean_nll": float(global_best.mean()),
        "full_library_oracle_mean_nll": float(oracle_nll.mean()),
        "full_library_oracle_headroom": float(global_best.mean() - oracle_nll.mean()),
        "oracle_action_counts": dict(sorted(Counter(aliases[index] for index in oracle_actions).items())),
        "lodo_router": {
            "mean_nll": float(routed_nll.mean()),
            "fold_static_baseline_mean_nll": float(fold_baseline_nll.mean()),
            "mean_delta_vs_fold_baseline": float(delta_fold.mean()),
            "dataset_cluster_ci95_vs_fold_baseline": list(fold_ci),
            "mean_delta_vs_global_best": float(delta_global.mean()),
            "dataset_cluster_ci95_vs_global_best": list(global_ci),
            "oracle_action_accuracy": float(np.mean(routed_actions == oracle_actions)),
            "selected_action_counts": dict(
                sorted(Counter(aliases[index] for index in routed_actions).items())
            ),
        },
        "interpretation": (
            "Oracle headroom measures operator complementarity. The LODO router is a strict "
            "cross-dataset test and does not use dataset identity or gold retrieval features."
        ),
        "plot_paths": plot_paths,
    }
    write_csv(output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(output_dir / "pairwise_complementarity.csv", pairwise_rows)
    write_csv(output_dir / "oracle_subset_curve.csv", oracle_curve)
    write_csv(output_dir / "lodo_folds.csv", fold_rows)
    write_csv(output_dir / "lodo_rows.csv", lodo_rows)
    write_csv(output_dir / "method_comparison.csv", comparison_rows)
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

