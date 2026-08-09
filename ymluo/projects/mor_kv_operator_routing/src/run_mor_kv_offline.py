from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


LEGACY_SELECTED_HEADS = ((3, 10), (21, 8), (6, 7), (16, 14))


@dataclass(frozen=True)
class Action:
    head_count: int
    depth: int
    mode: str
    bm25_quota: int

    @property
    def name(self) -> str:
        return f"h{self.head_count}_d{self.depth}_{self.mode}_bm{self.bm25_quota}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out evaluation of query-conditioned mixture-of-retrievers for KV block access."
        )
    )
    parser.add_argument("--allhead_topk_npz", required=True)
    parser.add_argument("--queries_jsonl", required=True)
    parser.add_argument("--bm25_scores", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head_profiles")
    parser.add_argument("--budgets", default="1,4,8,16,39")
    parser.add_argument("--head_counts", default="1,2,4,8,16,32")
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--modes", default="weighted_rrf,minority_max")
    parser.add_argument("--submodular_temperatures", default="0.25,0.5,1,2,4")
    parser.add_argument("--negative_penalty", type=float, default=0.25)
    parser.add_argument("--gqa_group_size", type=int, default=2)
    parser.add_argument("--gqa_deduplicate", default="true")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--make_plots", default="true")
    return parser.parse_args()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def parse_ints(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item.strip()})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_roles(path: str | None) -> dict[tuple[int, int], str]:
    if not path:
        return {}
    roles: dict[tuple[int, int], str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels = [item for item in row["multi_label_functions"].split(";") if item]
            primary = row["primary_function"]
            roles[(int(row["layer"]), int(row["head"]))] = (
                "+".join(labels) if labels else primary
            )
    return roles


class ScoreSignatureRouter:
    """Nearest-centroid router over score statistics available before KV gather."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.centroids: np.ndarray | None = None

    @staticmethod
    def features(scores: np.ndarray) -> np.ndarray:
        # scores: [queries, layers, query_heads, candidates]
        if scores.ndim != 4 or scores.shape[-1] < 4:
            raise ValueError(f"Expected [Q,L,H,K>=4] scores, got {scores.shape}")
        top = scores[..., 0]
        margin4 = scores[..., 0] - scores[..., 3]
        spread = scores.std(axis=-1)
        return np.concatenate(
            [
                top.reshape(scores.shape[0], -1),
                margin4.reshape(scores.shape[0], -1),
                spread.reshape(scores.shape[0], -1),
            ],
            axis=1,
        ).astype(np.float64)

    def fit(self, scores: np.ndarray, labels: Sequence[str], train_mask: np.ndarray) -> None:
        features = self.features(scores)
        train = features[train_mask]
        self.mean = train.mean(axis=0)
        self.scale = np.maximum(train.std(axis=0), 1e-6)
        normalized = (features - self.mean) / self.scale
        self.labels = sorted(set(labels))
        label_array = np.asarray(labels)
        self.centroids = np.stack(
            [normalized[train_mask & (label_array == label)].mean(axis=0) for label in self.labels]
        )

    def predict(self, scores: np.ndarray) -> list[str]:
        if self.mean is None or self.scale is None or self.centroids is None:
            raise RuntimeError("Router must be fitted before predict")
        normalized = (self.features(scores) - self.mean) / self.scale
        distances = ((normalized[:, None, :] - self.centroids[None, :, :]) ** 2).mean(axis=-1)
        return [self.labels[index] for index in distances.argmin(axis=1)]


def gqa_key(layer: int, query_head: int, group_size: int) -> tuple[int, int]:
    return layer, query_head // group_size


def head_quality(
    block_ids: np.ndarray,
    queries: Sequence[dict[str, Any]],
    query_indices: Sequence[int],
    *,
    gqa_group_size: int,
    gqa_deduplicate: bool,
) -> list[tuple[float, int, int]]:
    qualities: list[tuple[float, int, int]] = []
    for layer in range(block_ids.shape[1]):
        for head in range(block_ids.shape[2]):
            reciprocal_ranks: list[float] = []
            for query_index in query_indices:
                gold = {int(item) for item in queries[query_index]["gold_block_ids"]}
                ranks = [
                    rank
                    for rank, block_id in enumerate(block_ids[query_index, layer, head], start=1)
                    if int(block_id) in gold
                ]
                reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
            qualities.append((statistics.fmean(reciprocal_ranks), layer, head))
    qualities.sort(reverse=True)
    if not gqa_deduplicate:
        return qualities
    selected: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in qualities:
        key = gqa_key(item[1], item[2], gqa_group_size)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def bm25_ranking(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"Expected 2-D BM25 scores, got {scores.shape}")
    return np.argsort(-scores, axis=1, kind="stable")


def qk_ranking(
    query_index: int,
    block_ids: np.ndarray,
    quality: Sequence[tuple[float, int, int]],
    action: Action,
    max_blocks: int | None = None,
) -> list[int]:
    if action.mode.startswith("submodular"):
        temperature = 1.0
        if "_t" in action.mode:
            temperature = float(action.mode.rsplit("_t", 1)[1])
        if temperature <= 0.0:
            raise ValueError("submodular temperature must be positive")
        selected_heads = list(quality[: action.head_count])
        candidates: list[int] = []
        candidate_to_col: dict[int, int] = {}
        for _, layer, head in selected_heads:
            for block_id in block_ids[query_index, layer, head, : action.depth]:
                block = int(block_id)
                if block >= 0 and block not in candidate_to_col:
                    candidate_to_col[block] = len(candidates)
                    candidates.append(block)
        if not candidates:
            return []
        utility = np.zeros((len(selected_heads), len(candidates)), dtype=np.float64)
        for row, (weight, layer, head) in enumerate(selected_heads):
            safe_weight = max(float(weight), 1e-4)
            for rank, block_id in enumerate(
                block_ids[query_index, layer, head, : action.depth], start=1
            ):
                block = int(block_id)
                if block >= 0:
                    utility[row, candidate_to_col[block]] = max(
                        utility[row, candidate_to_col[block]], safe_weight / rank
                    )
        limit = min(len(candidates), max_blocks or len(candidates))
        coverage = np.zeros(len(selected_heads), dtype=np.float64)
        available = np.ones(len(candidates), dtype=bool)
        output: list[int] = []
        for _ in range(limit):
            gains = (
                np.exp(-coverage / temperature)[:, None]
                * (1.0 - np.exp(-utility / temperature))
            ).sum(axis=0)
            gains[~available] = -np.inf
            column = int(np.argmax(gains))
            if not math.isfinite(float(gains[column])) or gains[column] <= 0.0:
                break
            output.append(candidates[column])
            coverage += utility[:, column]
            available[column] = False
        return output

    values: dict[int, float] = {}
    for weight, layer, head in quality[: action.head_count]:
        safe_weight = max(float(weight), 1e-4)
        for rank, block_id in enumerate(
            block_ids[query_index, layer, head, : action.depth], start=1
        ):
            block = int(block_id)
            if block < 0:
                continue
            if action.mode == "weighted_rrf":
                values[block] = values.get(block, 0.0) + safe_weight / (60.0 + rank)
            elif action.mode == "minority_max":
                values[block] = max(values.get(block, 0.0), safe_weight / rank)
            else:
                raise ValueError(f"Unknown aggregation mode: {action.mode}")
    return [block for block, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def fixed_head_ranking(
    query_index: int,
    block_ids: np.ndarray,
    heads: Sequence[tuple[int, int]],
    *,
    depth: int,
) -> list[int]:
    values: dict[int, float] = {}
    for layer, head in heads:
        for rank, block_id in enumerate(block_ids[query_index, layer, head, :depth], start=1):
            block = int(block_id)
            values[block] = values.get(block, 0.0) + 1.0 / (60.0 + rank)
    return [block for block, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def allhead_ranking(query_index: int, block_ids: np.ndarray, depth: int) -> list[int]:
    values: dict[int, float] = {}
    _, layers, heads, _ = block_ids.shape
    for layer in range(layers):
        for head in range(heads):
            for rank, block_id in enumerate(
                block_ids[query_index, layer, head, :depth], start=1
            ):
                block = int(block_id)
                values[block] = values.get(block, 0.0) + 1.0 / (60.0 + rank)
    return [block for block, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def combine_rankings(
    bm25_ids: Sequence[int], qk_ids: Sequence[int], *, budget: int, bm25_quota: int
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for item in bm25_ids[:bm25_quota]:
        block = int(item)
        if block not in seen:
            output.append(block)
            seen.add(block)
    for item in qk_ids:
        block = int(item)
        if block not in seen:
            output.append(block)
            seen.add(block)
        if len(output) >= budget:
            return output[:budget]
    for item in bm25_ids[bm25_quota:]:
        block = int(item)
        if block not in seen:
            output.append(block)
            seen.add(block)
        if len(output) >= budget:
            break
    return output[:budget]


def query_metric(
    query: dict[str, Any], selected_ids: Sequence[int], negative_penalty: float
) -> dict[str, float]:
    selected = {int(item) for item in selected_ids}
    gold = {int(item) for item in query["gold_block_ids"]}
    negatives = {int(item) for item in query.get("hard_negative_block_ids", [])}
    evidence_hits = len(selected & gold)
    negative_hits = len(selected & negatives)
    fraction = evidence_hits / max(1, len(gold))
    negative_rate = float(negative_hits > 0)
    ranks = [rank for rank, item in enumerate(selected_ids, start=1) if int(item) in gold]
    return {
        "any_evidence_recall": float(evidence_hits > 0),
        "all_evidence_recall": float(evidence_hits == len(gold)),
        "evidence_fraction": fraction,
        "hard_negative_hit_rate": negative_rate,
        "hard_negative_hits": float(negative_hits),
        "evidence_mrr": 1.0 / min(ranks) if ranks else 0.0,
        "utility": fraction - negative_penalty * negative_rate,
    }


def mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric set")
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def candidate_bm25_quotas(budget: int) -> list[int]:
    return sorted(
        {
            0,
            1,
            budget // 4,
            budget // 2,
            (3 * budget) // 4,
            max(0, budget - 1),
            budget,
        }
    )


def action_space(
    budget: int,
    head_counts: Sequence[int],
    depths: Sequence[int],
    modes: Sequence[str],
    *,
    allow_bm25: bool,
) -> list[Action]:
    quotas = candidate_bm25_quotas(budget) if allow_bm25 else [0]
    return [
        Action(head_count, depth, mode, quota)
        for head_count in head_counts
        for depth in depths
        for mode in modes
        for quota in quotas
    ]


def evaluate_action(
    action: Action,
    query_indices: Sequence[int],
    *,
    budget: int,
    block_ids: np.ndarray,
    bm25_ids: np.ndarray,
    queries: Sequence[dict[str, Any]],
    quality: Sequence[tuple[float, int, int]],
    negative_penalty: float,
) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for query_index in query_indices:
        qk_ids = qk_ranking(query_index, block_ids, quality, action, max_blocks=budget)
        selected = combine_rankings(
            bm25_ids[query_index],
            qk_ids,
            budget=budget,
            bm25_quota=action.bm25_quota,
        )
        metrics.append(query_metric(queries[query_index], selected, negative_penalty))
    return mean_metrics(metrics)


def tune_action(
    actions: Sequence[Action],
    dev_indices: Sequence[int],
    **kwargs: Any,
) -> tuple[Action, dict[str, float]]:
    best: tuple[tuple[float, ...], Action, dict[str, float]] | None = None
    for action in actions:
        metrics = evaluate_action(action, dev_indices, **kwargs)
        key = (
            metrics["utility"],
            metrics["evidence_fraction"],
            metrics["all_evidence_recall"],
            metrics["any_evidence_recall"],
            -metrics["hard_negative_hit_rate"],
            -float(action.head_count),
            -float(action.depth),
        )
        if best is None or key > best[0]:
            best = (key, action, metrics)
    if best is None:
        raise ValueError("No actions were provided")
    return best[1], best[2]


def split_indices(queries: Sequence[dict[str, Any]], split: str) -> list[int]:
    return [index for index, query in enumerate(queries) if query.get("split") == split]


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for split in ["all", *sorted({str(row["split"]) for row in method_rows})]:
            for task in ["all", *sorted({str(row["task_type"]) for row in method_rows})]:
                group = [
                    row
                    for row in method_rows
                    if (split == "all" or row["split"] == split)
                    and (task == "all" or row["task_type"] == task)
                ]
                if not group:
                    continue
                summary = {
                    "method": method,
                    "split": split,
                    "task_type": task,
                    "queries": len(group),
                    "mean_selected_blocks": statistics.fmean(
                        float(row["selected_blocks"]) for row in group
                    ),
                }
                for key in (
                    "any_evidence_recall",
                    "all_evidence_recall",
                    "evidence_fraction",
                    "hard_negative_hit_rate",
                    "hard_negative_hits",
                    "evidence_mrr",
                    "utility",
                ):
                    summary[key] = statistics.fmean(float(row[key]) for row in group)
                output.append(summary)
    return output


def policy_heads(
    quality: Sequence[tuple[float, int, int]], action: Action
) -> list[tuple[float, int, int]]:
    return list(quality[: action.head_count])


def plot_results(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    router_confusion: np.ndarray,
    task_labels: Sequence[str],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    test_all = [
        row for row in summaries if row["split"] == "test" and row["task_type"] == "all"
    ]
    chosen_prefixes = (
        "bm25_b",
        "global_qk_b",
        "routed_qk_b",
        "single_hybrid_b",
        "mor_kv_b",
        "mor_kv_submodular_b",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for prefix in chosen_prefixes:
        selected = sorted(
            [row for row in test_all if str(row["method"]).startswith(prefix)],
            key=lambda row: float(row["mean_selected_blocks"]),
        )
        if not selected:
            continue
        ax.plot(
            [row["mean_selected_blocks"] for row in selected],
            [row["utility"] for row in selected],
            marker="o",
            label=prefix[:-2],
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Retrieved KV blocks")
    ax.set_ylabel("Evidence utility (fraction - penalty × distractor hit)")
    ax.set_title("Held-out retrieval utility versus KV block budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "heldout_utility_by_budget.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    budget4 = [row for row in test_all if int(float(row["mean_selected_blocks"])) == 4]
    fig, ax = plt.subplots(figsize=(8, 6))
    annotation_offsets = ((5, 4), (5, -14), (-85, 5), (5, 16), (-70, -16), (5, 28))
    plotted_index = 0
    for row in budget4:
        if not any(str(row["method"]).startswith(prefix) for prefix in chosen_prefixes):
            continue
        ax.scatter(row["hard_negative_hit_rate"], row["evidence_fraction"], s=70)
        ax.annotate(
            str(row["method"]).replace("_b4", ""),
            (row["hard_negative_hit_rate"], row["evidence_fraction"]),
            xytext=annotation_offsets[plotted_index % len(annotation_offsets)],
            textcoords="offset points",
        )
        plotted_index += 1
    ax.set_xlabel("Hard-negative hit rate (lower is better)")
    ax.set_ylabel("Evidence fraction (higher is better)")
    ax.set_title("Held-out budget-4 retrieval Pareto plane")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "budget4_evidence_vs_distractor.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(router_confusion, cmap="Blues")
    ax.set_xticks(range(len(task_labels)), task_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(task_labels)), task_labels)
    ax.set_xlabel("Predicted operator family")
    ax.set_ylabel("True task family")
    ax.set_title("Held-out score-signature router")
    for row in range(router_confusion.shape[0]):
        for col in range(router_confusion.shape[1]):
            ax.text(col, row, str(int(router_confusion[row, col])), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = plot_dir / "router_confusion_test.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.allhead_topk_npz)
    block_ids = payload["block_ids"]
    scores = payload["scores"]
    queries = read_jsonl(Path(args.queries_jsonl))
    bm25_scores = np.load(args.bm25_scores)
    bm25_ids = bm25_ranking(bm25_scores)
    if block_ids.shape[0] != len(queries) or scores.shape[0] != len(queries):
        raise ValueError("All-head arrays and query count differ")
    if bm25_scores.shape[0] != len(queries):
        raise ValueError("BM25 scores and query count differ")

    budgets = parse_ints(args.budgets)
    head_counts = parse_ints(args.head_counts)
    depths = parse_ints(args.depths)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    submodular_temperatures = [
        float(item) for item in args.submodular_temperatures.split(",") if item.strip()
    ]
    gqa_deduplicate = str2bool(args.gqa_deduplicate)
    make_plots = str2bool(args.make_plots)
    roles = load_roles(args.head_profiles)
    task_labels = sorted({str(query["task_type"]) for query in queries})
    split_array = np.asarray([str(query["split"]) for query in queries])
    true_tasks = [str(query["task_type"]) for query in queries]
    train_mask = split_array == "train"

    router = ScoreSignatureRouter()
    router.fit(scores, true_tasks, train_mask)
    predicted_tasks = router.predict(scores)
    router_rows = [
        {
            "query_id": int(query["query_id"]),
            "split": str(query["split"]),
            "true_task": true_tasks[index],
            "predicted_task": predicted_tasks[index],
            "correct": float(true_tasks[index] == predicted_tasks[index]),
        }
        for index, query in enumerate(queries)
    ]
    router_accuracy = {
        split: statistics.fmean(
            row["correct"] for row in router_rows if split == "all" or row["split"] == split
        )
        for split in ("all", "train", "dev", "test")
    }

    train_indices = split_indices(queries, "train")
    dev_indices = split_indices(queries, "dev")
    task_train = {
        task: [index for index in train_indices if true_tasks[index] == task]
        for task in task_labels
    }
    task_dev = {
        task: [index for index in dev_indices if true_tasks[index] == task]
        for task in task_labels
    }
    quality_global = head_quality(
        block_ids,
        queries,
        train_indices,
        gqa_group_size=args.gqa_group_size,
        gqa_deduplicate=gqa_deduplicate,
    )
    quality_task = {
        task: head_quality(
            block_ids,
            queries,
            indices,
            gqa_group_size=args.gqa_group_size,
            gqa_deduplicate=gqa_deduplicate,
        )
        for task, indices in task_train.items()
    }

    policies: dict[str, Any] = {
        "global": {},
        "task_qk": defaultdict(dict),
        "task_mor": defaultdict(dict),
    }
    policy_rows: list[dict[str, Any]] = []
    for budget in budgets:
        common_kwargs = {
            "budget": budget,
            "block_ids": block_ids,
            "bm25_ids": bm25_ids,
            "queries": queries,
            "negative_penalty": args.negative_penalty,
        }
        global_qk, global_qk_dev = tune_action(
            action_space(budget, head_counts, depths, modes, allow_bm25=False),
            dev_indices,
            quality=quality_global,
            **common_kwargs,
        )
        global_mor, global_mor_dev = tune_action(
            action_space(budget, head_counts, depths, modes, allow_bm25=True),
            dev_indices,
            quality=quality_global,
            **common_kwargs,
        )
        policies["global"][str(budget)] = {
            "qk": asdict(global_qk),
            "qk_dev": global_qk_dev,
            "mor": asdict(global_mor),
            "mor_dev": global_mor_dev,
        }
        for kind, action, quality in (
            ("global_qk", global_qk, quality_global),
            ("global_mor", global_mor, quality_global),
        ):
            for rank, (score, layer, head) in enumerate(policy_heads(quality, action), start=1):
                policy_rows.append(
                    {
                        "scope": "global",
                        "budget": budget,
                        "policy_kind": kind,
                        "rank": rank,
                        "layer": layer,
                        "query_head": head,
                        "kv_head": head // args.gqa_group_size,
                        "train_mrr": score,
                        "controlled_role": roles.get((layer, head), "unknown"),
                    }
                )
        for task in task_labels:
            task_qk, task_qk_dev = tune_action(
                action_space(budget, head_counts, depths, modes, allow_bm25=False),
                task_dev[task],
                quality=quality_task[task],
                **common_kwargs,
            )
            task_mor, task_mor_dev = tune_action(
                action_space(budget, head_counts, depths, modes, allow_bm25=True),
                task_dev[task],
                quality=quality_task[task],
                **common_kwargs,
            )
            policies["task_qk"][task][str(budget)] = {
                "action": asdict(task_qk),
                "dev": task_qk_dev,
            }
            policies["task_mor"][task][str(budget)] = {
                "action": asdict(task_mor),
                "dev": task_mor_dev,
            }
            for kind, action in (("task_qk", task_qk), ("task_mor", task_mor)):
                for rank, (score, layer, head) in enumerate(
                    policy_heads(quality_task[task], action), start=1
                ):
                    policy_rows.append(
                        {
                            "scope": task,
                            "budget": budget,
                            "policy_kind": kind,
                            "rank": rank,
                            "layer": layer,
                            "query_head": head,
                            "kv_head": head // args.gqa_group_size,
                            "train_mrr": score,
                            "controlled_role": roles.get((layer, head), "unknown"),
                        }
                    )

    submodular_tuning: dict[str, Any] = {}
    for budget in budgets:
        candidates: list[tuple[tuple[float, ...], float, dict[str, float]]] = []
        for temperature in submodular_temperatures:
            rows: list[dict[str, float]] = []
            for query_index in dev_indices:
                routed_task = predicted_tasks[query_index]
                base = Action(
                    **policies["task_mor"][routed_task][str(budget)]["action"]
                )
                action = Action(
                    head_count=base.head_count,
                    depth=base.depth,
                    mode=f"submodular_t{temperature:g}",
                    bm25_quota=base.bm25_quota,
                )
                ids = qk_ranking(
                    query_index,
                    block_ids,
                    quality_task[routed_task],
                    action,
                    max_blocks=budget,
                )
                selected = combine_rankings(
                    bm25_ids[query_index],
                    ids,
                    budget=budget,
                    bm25_quota=base.bm25_quota,
                )
                rows.append(query_metric(queries[query_index], selected, args.negative_penalty))
            metrics = mean_metrics(rows)
            key = (
                metrics["utility"],
                metrics["evidence_fraction"],
                -metrics["hard_negative_hit_rate"],
            )
            candidates.append((key, temperature, metrics))
        _, selected_temperature, selected_metrics = max(candidates, key=lambda item: item[0])
        submodular_tuning[str(budget)] = {
            "temperature": selected_temperature,
            "dev": selected_metrics,
        }

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    random_mapping = task_labels[1:] + task_labels[:1]
    random_task = dict(zip(task_labels, random_mapping))
    allhead_cache: dict[tuple[int, int], list[int]] = {}
    legacy_cache: dict[tuple[int, int], list[int]] = {}

    def record(method: str, query_index: int, selected: list[int], budget: int) -> None:
        query = queries[query_index]
        metric = query_metric(query, selected, args.negative_penalty)
        metric_rows.append(
            {
                "method": method,
                "query_id": int(query["query_id"]),
                "split": str(query["split"]),
                "task_type": str(query["task_type"]),
                "selected_blocks": len(selected),
                **metric,
            }
        )
        retrieval_rows.append(
            {
                "method": method,
                "query_id": int(query["query_id"]),
                "dataset": str(query["dataset"]),
                "split": str(query["split"]),
                "task_type": str(query["task_type"]),
                "selected_block_ids": json.dumps(selected),
                "ranked_block_ids": json.dumps(selected),
            }
        )

    for budget in budgets:
        global_qk = Action(**policies["global"][str(budget)]["qk"])
        global_mor = Action(**policies["global"][str(budget)]["mor"])
        for query_index, query in enumerate(queries):
            bm_selected = [int(item) for item in bm25_ids[query_index, :budget]]
            record(f"bm25_b{budget}", query_index, bm_selected, budget)

            cache_key = (query_index, min(16, budget))
            if cache_key not in allhead_cache:
                allhead_cache[cache_key] = allhead_ranking(
                    query_index, block_ids, depth=min(16, max(1, budget))
                )
                legacy_cache[cache_key] = fixed_head_ranking(
                    query_index,
                    block_ids,
                    LEGACY_SELECTED_HEADS,
                    depth=min(16, max(1, budget)),
                )
            record(
                f"allhead_rrf_b{budget}",
                query_index,
                allhead_cache[cache_key][:budget],
                budget,
            )
            record(
                f"legacy4_rrf_b{budget}",
                query_index,
                legacy_cache[cache_key][:budget],
                budget,
            )

            global_qk_ids = qk_ranking(
                query_index, block_ids, quality_global, global_qk, max_blocks=budget
            )
            record(
                f"global_qk_b{budget}",
                query_index,
                combine_rankings(
                    bm25_ids[query_index], global_qk_ids, budget=budget, bm25_quota=0
                ),
                budget,
            )
            global_mor_ids = qk_ranking(
                query_index, block_ids, quality_global, global_mor, max_blocks=budget
            )
            record(
                f"single_hybrid_b{budget}",
                query_index,
                combine_rankings(
                    bm25_ids[query_index],
                    global_mor_ids,
                    budget=budget,
                    bm25_quota=global_mor.bm25_quota,
                ),
                budget,
            )

            true_task = true_tasks[query_index]
            predicted_task = predicted_tasks[query_index]
            wrong_task = random_task[predicted_task]
            for prefix, routed_task in (
                ("oracle_task_qk", true_task),
                ("routed_qk", predicted_task),
            ):
                action = Action(**policies["task_qk"][routed_task][str(budget)]["action"])
                ids = qk_ranking(
                    query_index,
                    block_ids,
                    quality_task[routed_task],
                    action,
                    max_blocks=budget,
                )
                record(
                    f"{prefix}_b{budget}",
                    query_index,
                    combine_rankings(
                        bm25_ids[query_index], ids, budget=budget, bm25_quota=0
                    ),
                    budget,
                )
            for prefix, routed_task in (
                ("oracle_task_mor", true_task),
                ("mor_kv", predicted_task),
                ("wrong_router_mor", wrong_task),
            ):
                action = Action(**policies["task_mor"][routed_task][str(budget)]["action"])
                ids = qk_ranking(
                    query_index,
                    block_ids,
                    quality_task[routed_task],
                    action,
                    max_blocks=budget,
                )
                record(
                    f"{prefix}_b{budget}",
                    query_index,
                    combine_rankings(
                        bm25_ids[query_index],
                        ids,
                        budget=budget,
                        bm25_quota=action.bm25_quota,
                    ),
                    budget,
                )
                if prefix == "mor_kv":
                    temperature = float(submodular_tuning[str(budget)]["temperature"])
                    submodular_action = Action(
                        head_count=action.head_count,
                        depth=action.depth,
                        mode=f"submodular_t{temperature:g}",
                        bm25_quota=action.bm25_quota,
                    )
                    submodular_ids = qk_ranking(
                        query_index,
                        block_ids,
                        quality_task[routed_task],
                        submodular_action,
                        max_blocks=budget,
                    )
                    record(
                        f"mor_kv_submodular_b{budget}",
                        query_index,
                        combine_rankings(
                            bm25_ids[query_index],
                            submodular_ids,
                            budget=budget,
                            bm25_quota=action.bm25_quota,
                        ),
                        budget,
                    )

    summaries = aggregate_rows(metric_rows)
    test_label_to_index = {label: index for index, label in enumerate(task_labels)}
    confusion = np.zeros((len(task_labels), len(task_labels)), dtype=np.int64)
    for row in router_rows:
        if row["split"] != "test":
            continue
        confusion[
            test_label_to_index[row["true_task"]], test_label_to_index[row["predicted_task"]]
        ] += 1
    plot_paths = plot_results(output_dir, summaries, confusion, task_labels) if make_plots else []

    write_csv(output_dir / "router_predictions.csv", router_rows, list(router_rows[0]))
    write_csv(output_dir / "query_metrics.csv", metric_rows, list(metric_rows[0]))
    write_csv(output_dir / "query_results.csv", retrieval_rows, list(retrieval_rows[0]))
    write_csv(output_dir / "summary.csv", summaries, list(summaries[0]))
    write_csv(
        output_dir / "head_portfolios.csv",
        policy_rows,
        (
            "scope",
            "budget",
            "policy_kind",
            "rank",
            "layer",
            "query_head",
            "kv_head",
            "train_mrr",
            "controlled_role",
        ),
    )
    with (output_dir / "policies.json").open("w", encoding="utf-8") as handle:
        json.dump(policies, handle, ensure_ascii=False, indent=2)

    test_primary = [
        row
        for row in summaries
        if row["split"] == "test" and row["task_type"] == "all"
    ]
    summary = {
        "source": "held-out MoR-KV operator-routing experiment",
        "queries": len(queries),
        "splits": dict(Counter(query["split"] for query in queries)),
        "tasks": dict(Counter(true_tasks)),
        "budgets": budgets,
        "negative_penalty": args.negative_penalty,
        "gqa_group_size": args.gqa_group_size,
        "gqa_deduplicate": gqa_deduplicate,
        "router_accuracy": router_accuracy,
        "submodular_tuning": submodular_tuning,
        "test_methods": test_primary,
        "plot_paths": plot_paths,
        "interpretation": (
            "This experiment evaluates block nomination and distractor exposure from real Q/K "
            "profiles. End-to-end sparse-attention kernels and downstream generation remain separate gates."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
