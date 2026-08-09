from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank within-block spans with leakage-free step Q and real SVD32 K."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_query_profiles", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--window_tokens", type=int, default=32)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--span_mode", choices=["fixed", "sentence"], default="fixed")
    parser.add_argument("--specialist_heads", type=int, default=8)
    parser.add_argument("--exclude_query_ids", default="375")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_ids(*groups: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(item) for group in groups for item in group))


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> tuple[int, int]:
    values = list(int(item) for item in sequence)
    target = list(int(item) for item in pattern)
    if not target:
        raise ValueError("target fact tokenization is empty")
    for start in range(len(values) - len(target) + 1):
        if values[start : start + len(target)] == target:
            return start, start + len(target)
    raise ValueError("target fact token sequence was not found in its declared block")


def find_text_subsequence(
    sequence: Sequence[int], text: str, tokenizer: Any
) -> tuple[int, int]:
    patterns = []
    for prefix in ("", " ", "\n", "\n\n"):
        token_ids = tokenizer(prefix + text, add_special_tokens=False)["input_ids"]
        if token_ids and token_ids not in patterns:
            patterns.append(token_ids)
    for pattern in patterns:
        try:
            return find_subsequence(sequence, pattern)
        except ValueError:
            continue
    raise ValueError(f"target fact was not found under whitespace variants: {text!r}")


def window_starts(block_tokens: int, window_tokens: int, stride: int) -> list[int]:
    if window_tokens <= 0 or stride <= 0 or window_tokens > block_tokens:
        raise ValueError("invalid window configuration")
    starts = list(range(0, block_tokens - window_tokens + 1, stride))
    final_start = block_tokens - window_tokens
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def sentence_token_spans(
    block_token_ids: Sequence[int], tokenizer: Any
) -> list[tuple[int, int]]:
    text = tokenizer.decode(block_token_ids, skip_special_tokens=True)
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if item.strip()
    ]
    spans = []
    for sentence in sentences:
        try:
            span = find_text_subsequence(block_token_ids, sentence, tokenizer)
        except ValueError:
            continue
        if span not in spans:
            spans.append(span)
    if not spans:
        raise ValueError("block did not produce any token-aligned sentence spans")
    return spans


class SVDKIndex:
    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = profile_dir
        self.summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
        if self.summary.get("contains_synthetic_vectors"):
            raise ValueError("KV span retrieval requires real Q/K profiles")
        self.layers = [int(item) for item in self.summary["layers"]]
        self.shards = sorted(self.summary["shards"], key=lambda row: int(row["block_start"]))
        self.arrays: dict[tuple[int, int], np.ndarray] = {}
        for shard_index, shard in enumerate(self.shards):
            for layer in self.layers:
                path = profile_dir / Path(shard["layer_k_paths"][str(layer)]).name
                self.arrays[(shard_index, layer)] = np.load(path, mmap_mode="r")

    def block(self, layer: int, block_id: int) -> np.ndarray:
        for shard_index, shard in enumerate(self.shards):
            start = int(shard["block_start"])
            end = int(shard["block_end"])
            if start <= block_id < end:
                return np.asarray(self.arrays[(shard_index, layer)][block_id - start])
        raise IndexError(f"block {block_id} is outside all profile shards")


def candidate_blocks(step: dict[str, Any], scenario: str) -> list[int]:
    target = [int(item) for item in step["target_block_ids"]]
    negative = [int(item) for item in step["hard_negative_block_ids"][:1]]
    if scenario == "target_block":
        return target
    if scenario == "target_plus_negative":
        return unique_ids(target, negative)
    raise ValueError(f"unknown scenario: {scenario}")


def step_window_features(
    *,
    index: SVDKIndex,
    query: np.ndarray,
    query_mask: np.ndarray,
    block_ids: Sequence[int],
    window_tokens: int,
    stride: int,
    num_kv_heads: int,
    spans_by_block: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    query_tokens, num_layers, num_query_heads, _rank = query.shape
    repeat_groups = num_query_heads // num_kv_heads
    kv_map = np.arange(num_query_heads, dtype=np.int64) // repeat_groups
    fixed_starts = window_starts(
        int(index.summary["block_tokens"]), window_tokens, stride
    )
    ranges_by_block = {
        int(block_id): (
            spans_by_block[int(block_id)]
            if spans_by_block is not None
            else [(start, start + window_tokens) for start in fixed_starts]
        )
        for block_id in block_ids
    }
    metadata = [
        {"block_id": int(block_id), "start": int(start), "end": int(end)}
        for block_id in block_ids
        for start, end in ranges_by_block[int(block_id)]
    ]
    features = np.empty((len(metadata), num_layers * num_query_heads), dtype=np.float32)
    valid = query_mask.astype(bool)
    if not valid.any():
        raise ValueError("step query profile has no valid query tokens")
    for layer_index, layer in enumerate(index.layers):
        q = query[:, layer_index].astype(np.float32, copy=False)
        layer_offset = layer_index * num_query_heads
        row_offset = 0
        for block_id in block_ids:
            keys = index.block(layer, int(block_id)).astype(np.float32, copy=False)
            mapped_keys = keys[:, kv_map]
            similarities = np.einsum("qhd,thd->qht", q, mapped_keys, optimize=True)
            for start, end in ranges_by_block[int(block_id)]:
                pooled = similarities[:, :, start:end].max(axis=-1)
                features[row_offset, layer_offset : layer_offset + num_query_heads] = (
                    pooled[valid].mean(axis=0)
                )
                row_offset += 1
    return features, metadata


def target_overlap(
    metadata: Sequence[dict[str, int]], target_block: int, target_span: tuple[int, int]
) -> np.ndarray:
    target_start, target_end = target_span
    target_length = max(1, target_end - target_start)
    return np.asarray(
        [
            (
                max(0, min(item["end"], target_end) - max(item["start"], target_start))
                / target_length
                if int(item["block_id"]) == target_block
                else 0.0
            )
            for item in metadata
        ],
        dtype=np.float32,
    )


def specialist_heads(
    examples: Sequence[dict[str, Any]], count: int
) -> dict[str, dict[str, Any]]:
    by_step: dict[str, list[np.ndarray]] = defaultdict(list)
    for example in examples:
        if example["split"] != "train" or example["scenario"] != "target_plus_negative":
            continue
        features = example["features"]
        positive = example["overlap"] >= 0.999
        if not positive.any() or positive.all():
            continue
        margin = features[positive].max(axis=0) - features[~positive].max(axis=0)
        by_step[example["step_type"]].append(margin)
    output = {}
    for step_type, margins in by_step.items():
        mean_margin = np.mean(np.stack(margins), axis=0)
        selected = np.argsort(-mean_margin)[:count]
        positive_weights = np.maximum(mean_margin[selected], 0.0) + 1.0e-4
        positive_weights /= positive_weights.sum()
        output[step_type] = {
            "indices": selected,
            "weights": positive_weights,
            "mean_margin": mean_margin,
        }
    return output


def method_scores(
    features: np.ndarray,
    specialist: dict[str, Any],
) -> dict[str, np.ndarray]:
    top_count = min(8, features.shape[1])
    dynamic_top = np.partition(features, -top_count, axis=1)[:, -top_count:]
    indices = specialist["indices"]
    weights = specialist["weights"]
    return {
        "all_heads_mean": features.mean(axis=1),
        "dynamic_top8_mean": dynamic_top.mean(axis=1),
        "operator_specialist_mean": features[:, indices].mean(axis=1),
        "operator_specialist_weighted": features[:, indices] @ weights,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (row["split"], row["step_type"], row["scenario"], row["method"])
            for row in rows
        }
    )
    output = []
    for key in keys:
        group = [
            row
            for row in rows
            if (row["split"], row["step_type"], row["scenario"], row["method"])
            == key
        ]
        output.append(
            {
                "split": key[0],
                "step_type": key[1],
                "scenario": key[2],
                "method": key[3],
                "steps": len(group),
                "target_block_recall_at_1": statistics.fmean(
                    row["target_block_hit_at_1"] for row in group
                ),
                "target_span_recall_at_1": statistics.fmean(
                    row["target_span_hit_at_1"] for row in group
                ),
                "target_span_80pct_recall_at_1": statistics.fmean(
                    row["target_span_hit80_at_1"] for row in group
                ),
                "target_span_recall_at_2": statistics.fmean(
                    row["target_span_hit_at_2"] for row in group
                ),
                "mean_target_span_mrr": statistics.fmean(
                    row["target_span_mrr"] for row in group
                ),
                "mean_selected_tokens": statistics.fmean(
                    row["selected_tokens"] for row in group
                ),
                "mean_top1_margin": statistics.fmean(row["top1_margin"] for row in group),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.specialist_heads <= 0:
        raise ValueError("specialist_heads must be positive")
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    index = SVDKIndex(Path(args.profile_dir))
    query_payload = torch.load(
        args.step_query_profiles, map_location="cpu", weights_only=False
    )
    if query_payload["profile_space"] != "pre_rope_step_state_q":
        raise ValueError("expected leakage-free step-state Q profiles")
    steps = query_payload["steps"]
    query_vectors = query_payload["svd_q"].numpy()
    query_mask = query_payload["mask"].numpy()
    num_kv_heads = int(query_payload["num_kv_heads"])
    spans_by_block = None
    if args.span_mode == "sentence":
        spans_by_block = {
            block_id: sentence_token_spans(blocks[block_id].tolist(), tokenizer)
            for block_id in range(int(blocks.shape[0]))
        }

    examples = []
    feature_started = time.perf_counter()
    for step_index, step in enumerate(steps):
        if int(step["query_id"]) in excluded_ids:
            continue
        target_block = int(step["target_block_ids"][0])
        fact_span = find_text_subsequence(
            blocks[target_block].tolist(), str(step["target_fact"]), tokenizer
        )
        for scenario in ("target_block", "target_plus_negative"):
            features, metadata = step_window_features(
                index=index,
                query=query_vectors[step_index],
                query_mask=query_mask[step_index],
                block_ids=candidate_blocks(step, scenario),
                window_tokens=args.window_tokens,
                stride=args.window_stride,
                num_kv_heads=num_kv_heads,
                spans_by_block=spans_by_block,
            )
            examples.append(
                {
                    "query_id": int(step["query_id"]),
                    "step_index": int(step["step_index"]),
                    "split": str(step["split"]),
                    "step_type": str(step["step_type"]),
                    "scenario": scenario,
                    "target_block": target_block,
                    "features": features,
                    "metadata": metadata,
                    "overlap": target_overlap(metadata, target_block, fact_span),
                }
            )
    feature_seconds = time.perf_counter() - feature_started
    specialists = specialist_heads(examples, args.specialist_heads)

    rows = []
    ranking_started = time.perf_counter()
    for example in examples:
        specialist = specialists[example["step_type"]]
        for method, scores in method_scores(example["features"], specialist).items():
            order = np.argsort(-scores)
            overlap = example["overlap"]
            positive_ranks = np.flatnonzero(overlap[order] >= 0.999)
            best_rank = int(positive_ranks[0]) + 1 if positive_ranks.size else 0
            top = example["metadata"][int(order[0])]
            top_candidates = [
                {
                    "rank": rank + 1,
                    "block_id": int(example["metadata"][int(item)]["block_id"]),
                    "start": int(example["metadata"][int(item)]["start"]),
                    "end": int(example["metadata"][int(item)]["end"]),
                    "score": float(scores[item]),
                    "target_overlap": float(overlap[item]),
                }
                for rank, item in enumerate(order[:4])
            ]
            rows.append(
                {
                    "query_id": example["query_id"],
                    "step_index": example["step_index"],
                    "split": example["split"],
                    "step_type": example["step_type"],
                    "scenario": example["scenario"],
                    "method": method,
                    "target_block_hit_at_1": int(top["block_id"])
                    == example["target_block"],
                    "target_span_overlap_at_1": float(overlap[order[0]]),
                    "target_span_hit_at_1": bool(overlap[order[0]] >= 0.999),
                    "target_span_hit80_at_1": bool(overlap[order[0]] >= 0.8),
                    "target_span_hit_at_2": bool((overlap[order[:2]] >= 0.999).any()),
                    "target_span_mrr": 1.0 / best_rank if best_rank else 0.0,
                    "top1_margin": float(scores[order[0]] - scores[order[1]])
                    if len(order) > 1
                    else float("inf"),
                    "selected_block": int(top["block_id"]),
                    "selected_start": int(top["start"]),
                    "selected_end": int(top["end"]),
                    "selected_tokens": int(top["end"] - top["start"]),
                    "best_target_rank": best_rank,
                    "top_candidates": top_candidates,
                }
            )
    ranking_seconds = time.perf_counter() - ranking_started

    specialist_payload = {}
    num_query_heads = int(query_payload["num_query_heads"])
    for step_type, values in specialists.items():
        selected = [int(item) for item in values["indices"]]
        specialist_payload[step_type] = [
            {
                "flat_index": index_value,
                "layer": index.layers[index_value // num_query_heads],
                "query_head": index_value % num_query_heads,
                "train_margin": float(values["mean_margin"][index_value]),
                "weight": float(values["weights"][offset]),
            }
            for offset, index_value in enumerate(selected)
        ]
    payload = {
        "source": "real leakage-free step Q and real pre-RoPE SVD32 K",
        "contains_synthetic_vectors": False,
        "selection_uses_test_gold": False,
        "specialists_fit_split": "train",
        "excluded_query_ids": sorted(excluded_ids),
        "layers": index.layers,
        "span_mode": args.span_mode,
        "window_tokens": args.window_tokens,
        "window_stride": args.window_stride,
        "specialist_heads": args.specialist_heads,
        "examples": len(examples),
        "feature_seconds": feature_seconds,
        "mean_feature_milliseconds_per_example": feature_seconds / len(examples) * 1e3,
        "ranking_seconds": ranking_seconds,
        "specialist_head_details": specialist_payload,
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
