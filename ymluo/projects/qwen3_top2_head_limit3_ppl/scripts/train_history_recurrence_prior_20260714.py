from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import HistGradientBoostingClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_multitopic_lpcm_ppl_20260714 import AutoTokenizer  # noqa: E402


FEATURE_NAMES = [
    "position",
    "distance_to_query",
    "rarity_mean",
    "rare2_fraction",
    "rare8_fraction",
    "unique_fraction",
    "ngram_recurrence",
    "token_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a history-only prior for future recurrence sources.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_model", required=True, type=Path)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--windows_per_category", type=int, default=10)
    parser.add_argument("--window_stride_tokens", type=int, default=8192)
    parser.add_argument("--span_tokens", type=int, default=480)
    parser.add_argument("--span_stride_tokens", type=int, default=64)
    parser.add_argument("--label_ngram_tokens", type=int, default=16)
    parser.add_argument("--feature_ngram_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def encode_category_stream(
    tokenizer: Any,
    documents: list[str],
    required_tokens: int,
    seed: int,
) -> list[int]:
    usable = [text.strip() for text in documents if len(text.strip()) >= 200]
    random.Random(seed).shuffle(usable)
    stream: list[int] = []
    for document in usable:
        stream.extend(tokenizer("\n\n---\n\n" + document, add_special_tokens=False)["input_ids"])
        if len(stream) >= required_tokens:
            break
    return stream


def mine_recurrence_episodes(
    remote_ids: list[int],
    target_ids: list[int],
    ngram_tokens: int,
) -> list[dict[str, int]]:
    index: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for start in range(len(remote_ids) - ngram_tokens + 1):
        index[tuple(remote_ids[start : start + ngram_tokens])].append(start)
    matches: list[tuple[int, int]] = []
    for target_start in range(len(target_ids) - ngram_tokens + 1):
        starts = index.get(tuple(target_ids[target_start : target_start + ngram_tokens]))
        if starts:
            matches.append((target_start, starts[-1]))
    episodes: list[dict[str, int]] = []
    for target_start, source_start in matches:
        if episodes and (
            target_start == episodes[-1]["last_target_start"] + 1
            and source_start == episodes[-1]["last_source_start"] + 1
        ):
            episodes[-1]["last_target_start"] = target_start
            episodes[-1]["last_source_start"] = source_start
            episodes[-1]["matched_tokens"] += 1
            continue
        episodes.append(
            {
                "target_start": target_start,
                "source_start": source_start,
                "last_target_start": target_start,
                "last_source_start": source_start,
                "matched_tokens": ngram_tokens,
            }
        )
    return episodes


def prefix_sum(values: list[float]) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=np.float64))))


def span_feature_rows(
    remote_ids: list[int],
    span_tokens: int,
    stride_tokens: int,
    ngram_tokens: int,
) -> tuple[list[int], np.ndarray]:
    token_counts = Counter(remote_ids)
    rarity = [math.log((len(remote_ids) + 1) / token_counts[token]) for token in remote_ids]
    rare2 = [float(token_counts[token] <= 2) for token in remote_ids]
    rare8 = [float(token_counts[token] <= 8) for token in remote_ids]
    ngrams = [tuple(remote_ids[i : i + ngram_tokens]) for i in range(len(remote_ids) - ngram_tokens + 1)]
    ngram_counts = Counter(ngrams)
    recurrence = [math.log1p(ngram_counts[gram] - 1) for gram in ngrams]
    rarity_prefix = prefix_sum(rarity)
    rare2_prefix = prefix_sum(rare2)
    rare8_prefix = prefix_sum(rare8)
    recurrence_prefix = prefix_sum(recurrence)
    starts = list(range(0, len(remote_ids) - span_tokens + 1, stride_tokens))
    rows: list[list[float]] = []
    for start in starts:
        end = start + span_tokens
        ids = remote_ids[start:end]
        counts = Counter(ids)
        entropy = -sum((count / span_tokens) * math.log(count / span_tokens) for count in counts.values())
        ngram_end = max(start, end - ngram_tokens + 1)
        ngram_count = max(1, ngram_end - start)
        rows.append(
            [
                start / len(remote_ids),
                (len(remote_ids) - start) / len(remote_ids),
                (rarity_prefix[end] - rarity_prefix[start]) / span_tokens,
                (rare2_prefix[end] - rare2_prefix[start]) / span_tokens,
                (rare8_prefix[end] - rare8_prefix[start]) / span_tokens,
                len(counts) / span_tokens,
                (recurrence_prefix[ngram_end] - recurrence_prefix[start]) / ngram_count,
                entropy,
            ]
        )
    return starts, np.asarray(rows, dtype=np.float32)


def label_candidate_starts(
    episodes: list[dict[str, int]],
    candidate_starts: list[int],
    span_tokens: int,
) -> set[int]:
    positives: set[int] = set()
    for episode in episodes:
        source = episode["source_start"]
        eligible = [start for start in candidate_starts if start <= source < start + span_tokens]
        if eligible:
            positives.add(max(eligible))
    return positives


def ranking_metrics(
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["window_id"]].append(index)
    positive_groups = 0
    hits = {1: 0, 4: 0, 8: 0, 16: 0}
    reciprocal_ranks: list[float] = []
    details: list[dict[str, Any]] = []
    for window_id, indices in groups.items():
        positives = {index for index in indices if records[index]["label"] == 1}
        if not positives:
            continue
        positive_groups += 1
        ranked = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
        best_rank = min(ranked.index(index) + 1 for index in positives)
        reciprocal_ranks.append(1.0 / best_rank)
        for k in hits:
            hits[k] += int(best_rank <= k)
        details.append(
            {
                "window_id": window_id,
                "best_positive_rank": best_rank,
                "candidate_count": len(indices),
            }
        )
    return {
        "positive_windows": positive_groups,
        "hit_at": {str(k): hits[k] / max(1, positive_groups) for k in hits},
        "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
        "details": details,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        data_home=args.dataset_cache_dir,
        shuffle=False,
    )
    documents_by_category: dict[str, list[str]] = defaultdict(list)
    for text, target in zip(dataset.data, dataset.target):
        documents_by_category[dataset.target_names[int(target)]].append(text)

    test_categories = {"rec.sport.baseball", "sci.med", "comp.graphics"}
    remaining = sorted(set(dataset.target_names) - test_categories)
    dev_categories = set(remaining[-3:])
    train_categories = set(remaining[:-3])
    required_tokens = (
        (args.windows_per_category - 1) * args.window_stride_tokens
        + args.history_tokens
        + args.eval_tokens
    )
    records: list[dict[str, Any]] = []
    feature_batches: list[np.ndarray] = []
    skipped_categories: list[str] = []
    for category in sorted(dataset.target_names):
        stream = encode_category_stream(
            tokenizer,
            documents_by_category[category],
            required_tokens,
            args.seed + dataset.target_names.index(category),
        )
        if len(stream) < required_tokens:
            skipped_categories.append(category)
            continue
        split = "test" if category in test_categories else "dev" if category in dev_categories else "train"
        for window in range(args.windows_per_category):
            start = window * args.window_stride_tokens
            remote_ids = stream[start : start + args.history_tokens]
            target_ids = stream[
                start + args.history_tokens : start + args.history_tokens + args.eval_tokens
            ]
            episodes = mine_recurrence_episodes(remote_ids, target_ids, args.label_ngram_tokens)
            candidate_starts, features = span_feature_rows(
                remote_ids,
                args.span_tokens,
                args.span_stride_tokens,
                args.feature_ngram_tokens,
            )
            positives = label_candidate_starts(episodes, candidate_starts, args.span_tokens)
            window_id = f"{category}:{window}"
            feature_batches.append(features)
            records.extend(
                {
                    "window_id": window_id,
                    "category": category,
                    "split": split,
                    "candidate_start": candidate_start,
                    "label": int(candidate_start in positives),
                    "episode_count": len(episodes),
                }
                for candidate_start in candidate_starts
            )
    x = np.concatenate(feature_batches, axis=0)
    y = np.asarray([record["label"] for record in records], dtype=np.int64)
    train_mask = np.asarray([record["split"] == "train" for record in records])
    positive_weight = max(1.0, float((train_mask & (y == 0)).sum()) / max(1, (train_mask & (y == 1)).sum()))
    sample_weight = np.where(y[train_mask] == 1, positive_weight, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    model.fit(x[train_mask], y[train_mask], sample_weight=sample_weight)
    scores = model.predict_proba(x)[:, 1]
    split_metrics: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        indices = [index for index, record in enumerate(records) if record["split"] == split]
        split_metrics[split] = ranking_metrics(
            [records[index] for index in indices], scores[np.asarray(indices)]
        )
    output = {
        "feature_names": FEATURE_NAMES,
        "train_categories": sorted(train_categories),
        "dev_categories": sorted(dev_categories),
        "test_categories": sorted(test_categories),
        "skipped_categories": skipped_categories,
        "candidate_rows": len(records),
        "positive_rows": int(y.sum()),
        "positive_weight": positive_weight,
        "metrics": split_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "config": vars(args)}, args.output_model)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
