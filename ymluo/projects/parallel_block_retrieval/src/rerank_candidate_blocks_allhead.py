from __future__ import annotations

import argparse
import bisect
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank lexical candidates with frozen train-selected all-head Q/K channels."
    )
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_query_profiles", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_field", default="lexical_candidates")
    parser.add_argument("--candidate_limit", type=int, default=64)
    parser.add_argument("--selected_channels", type=int, default=8)
    parser.add_argument("--rrf_top_blocks", type=int, default=16)
    parser.add_argument("--rrf_constant", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LayerSvdKIndex:
    def __init__(self, profile_dir: Path, summary: dict[str, Any]) -> None:
        self.profile_dir = profile_dir
        self.shards = sorted(summary["shards"], key=lambda item: int(item["block_start"]))
        self.starts = [int(item["block_start"]) for item in self.shards]
        self.ends = [int(item["block_end"]) for item in self.shards]

    def blocks(self, layer: int, block_ids: Sequence[int]) -> np.ndarray:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, block_id_value in enumerate(block_ids):
            block_id = int(block_id_value)
            shard_index = bisect.bisect_right(self.starts, block_id) - 1
            if shard_index < 0 or block_id >= self.ends[shard_index]:
                raise IndexError(f"block {block_id} is outside the all-head index")
            grouped.setdefault(shard_index, []).append((output_index, block_id))
        output = None
        for shard_index, items in grouped.items():
            shard = self.shards[shard_index]
            path = self.profile_dir / Path(shard["layer_k_paths"][str(layer)]).name
            array = np.load(path, mmap_mode="r")
            offsets = [block_id - self.starts[shard_index] for _, block_id in items]
            values = np.asarray(array[offsets])
            if output is None:
                output = np.empty((len(block_ids), *values.shape[1:]), dtype=values.dtype)
            output_indices = np.asarray([item[0] for item in items], dtype=np.int64)
            output[output_indices] = values
        if output is None:
            raise ValueError("candidate block list is empty")
        return output


def rank_target(scores: np.ndarray, target_index: int) -> int:
    block_ids = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((block_ids, -scores))
    return int(np.flatnonzero(order == target_index)[0]) + 1


def ranked_candidate_ids(
    scores: np.ndarray, candidate_ids: Sequence[int], limit: int
) -> list[int]:
    tie_break_ids = np.asarray(candidate_ids, dtype=np.int64)
    order = np.lexsort((tie_break_ids, -scores))[:limit]
    return [int(candidate_ids[index]) for index in order]


def select_channels(
    score_cube: np.ndarray,
    rows: Sequence[dict[str, Any]],
    target_positions: Sequence[int],
    step_type: str,
    count: int,
) -> list[int]:
    train_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "train"
        and row["step_type"] == step_type
        and target_positions[index] >= 0
    ]
    channel_count = score_cube.shape[1]
    metrics = []
    for channel in range(channel_count):
        ranks = [
            rank_target(score_cube[index, channel], target_positions[index])
            for index in train_indices
        ]
        recall3 = statistics.fmean(rank <= 3 for rank in ranks) if ranks else 0.0
        mrr = statistics.fmean(1.0 / rank for rank in ranks) if ranks else 0.0
        metrics.append((recall3, mrr, channel))
    metrics.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in metrics[:count]]


def rrf_scores(
    channel_scores: np.ndarray, top_blocks: int, constant: float
) -> np.ndarray:
    candidate_count = channel_scores.shape[1]
    output = np.zeros(candidate_count, dtype=np.float32)
    block_ids = np.arange(candidate_count, dtype=np.int64)
    for scores in channel_scores:
        order = np.lexsort((block_ids, -scores))[:top_blocks]
        for rank, block_id in enumerate(order, start=1):
            output[block_id] += 1.0 / (constant + rank)
    return output


def summarize(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["split"], row["step_type"]) for row in rows})
    for split, step_type in keys:
        group = [
            row for row in rows if row["split"] == split and row["step_type"] == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "candidate_recall": statistics.fmean(row["candidate_hit"] for row in group),
            "mean_score_seconds": statistics.fmean(row["score_seconds"] for row in group),
        }
        for method in methods:
            valid = [row[f"{method}_rank"] for row in group if row[f"{method}_rank"] > 0]
            for budget in (1, 3, 16):
                item[f"{method}_recall_at_{budget}"] = statistics.fmean(
                    0 < row[f"{method}_rank"] <= budget for row in group
                )
            item[f"{method}_mrr"] = statistics.fmean(
                1.0 / row[f"{method}_rank"] if row[f"{method}_rank"] > 0 else 0.0
                for row in group
            )
            item[f"{method}_conditional_mrr"] = (
                statistics.fmean(1.0 / rank for rank in valid) if valid else 0.0
            )
        output.append(item)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.candidate_limit <= 0 or args.selected_channels <= 0:
        raise ValueError("candidate and channel counts must be positive")
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    query_payload = torch.load(
        Path(args.step_query_profiles), map_location="cpu", weights_only=False
    )
    steps = [dict(item) for item in query_payload["steps"]]
    candidates_by_key = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.candidate_rows_path))
    }
    layers = [int(item) for item in summary["layers"]]
    num_query_heads = int(summary["num_query_heads"])
    num_kv_heads = int(summary["num_kv_heads"])
    repeat_groups = num_query_heads // num_kv_heads
    channel_count = len(layers) * num_query_heads
    device = torch.device(args.device)
    index = LayerSvdKIndex(profile_dir, summary)
    max_candidates = args.candidate_limit
    score_cube = np.full(
        (len(steps), channel_count, max_candidates), -np.inf, dtype=np.float32
    )
    target_positions = []
    candidate_lists = []
    score_seconds = [0.0 for _ in steps]
    query_vectors = query_payload["svd_q"]
    query_masks = query_payload["mask"]
    for step_index, step in enumerate(steps):
        key = (int(step["query_id"]), int(step["step_index"]))
        candidate_ids = [
            int(item)
            for item in candidates_by_key[key][args.candidate_field][:max_candidates]
        ]
        candidate_lists.append(candidate_ids)
        target = int(step["target_block_ids"][0])
        target_positions.append(candidate_ids.index(target) if target in candidate_ids else -1)
    union_candidate_ids = sorted(
        {block_id for candidate_ids in candidate_lists for block_id in candidate_ids}
    )
    union_offsets = {
        block_id: offset for offset, block_id in enumerate(union_candidate_ids)
    }
    candidate_offsets = [
        torch.tensor(
            [union_offsets[block_id] for block_id in candidate_ids],
            dtype=torch.long,
            device=device,
        )
        for candidate_ids in candidate_lists
    ]
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        layer_keys_np = index.blocks(layer, union_candidate_ids)
        layer_keys = torch.from_numpy(layer_keys_np).to(
            device=device, dtype=torch.float16
        )
        for step_index, candidate_ids in enumerate(candidate_lists):
            step_started = time.perf_counter()
            keys = layer_keys.index_select(0, candidate_offsets[step_index])
            query = query_vectors[step_index, :, layer_index].to(
                device=device, dtype=torch.float16
            )
            mask = query_masks[step_index].to(device=device)
            grouped_query = query.reshape(
                query.shape[0], num_kv_heads, repeat_groups, query.shape[-1]
            )
            similarities = torch.einsum("igpd,btgd->igpbt", grouped_query, keys)
            per_token = similarities.amax(dim=-1).float()
            valid = mask.sum().clamp_min(1).float()
            block_scores = (per_token * mask[:, None, None, None]).sum(dim=0) / valid
            block_scores = block_scores.reshape(num_query_heads, len(candidate_ids))
            channel_start = layer_index * num_query_heads
            score_cube[
                step_index,
                channel_start : channel_start + num_query_heads,
                : len(candidate_ids),
            ] = block_scores.cpu().numpy()
            del keys, similarities, per_token, block_scores
            score_seconds[step_index] += time.perf_counter() - step_started
        del layer_keys
        print(
            json.dumps(
                {
                    "layer": layer,
                    "layer_index": layer_index + 1,
                    "layers": len(layers),
                    "union_candidate_blocks": len(union_candidate_ids),
                    "seconds": time.perf_counter() - layer_started,
                }
            ),
            flush=True,
        )

    selected = {
        step_type: select_channels(
            score_cube,
            steps,
            target_positions,
            step_type,
            args.selected_channels,
        )
        for step_type in sorted({str(step["step_type"]) for step in steps})
    }
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    fixed_pairs = [(3, 10), (21, 8), (6, 7), (16, 14)]
    fixed_channels = [
        layer_to_index[layer] * num_query_heads + head
        for layer, head in fixed_pairs
        if layer in layer_to_index and head < num_query_heads
    ]
    methods = ["candidate", "fixed4", "all_mean", "all_max", "all_rrf", "selected"]
    rows = []
    for index_value, step in enumerate(steps):
        candidate_ids = candidate_lists[index_value]
        count = len(candidate_ids)
        target_position = target_positions[index_value]
        scores = score_cube[index_value, :, :count]
        method_scores = {
            "fixed4": scores[fixed_channels].mean(axis=0),
            "all_mean": scores.mean(axis=0),
            "all_max": scores.max(axis=0),
            "all_rrf": rrf_scores(scores, args.rrf_top_blocks, args.rrf_constant),
            "selected": scores[selected[str(step["step_type"])]].mean(axis=0),
        }
        row = {
            "query_id": int(step["query_id"]),
            "step_index": int(step["step_index"]),
            "split": str(step["split"]),
            "step_type": str(step["step_type"]),
            "target_block_id": int(step["target_block_ids"][0]),
            "candidate_blocks": count,
            "candidate_hit": target_position >= 0,
            "candidate_rank": target_position + 1 if target_position >= 0 else 0,
            "score_seconds": score_seconds[index_value],
            "selection_uses_gold": False,
            "channel_selection_uses_train_labels_only": True,
            "candidate_top16": candidate_ids[:16],
        }
        for method, values in method_scores.items():
            row[f"{method}_rank"] = (
                rank_target(values, target_position) if target_position >= 0 else 0
            )
            row[f"{method}_top16"] = ranked_candidate_ids(
                values, candidate_ids, 16
            )
        rows.append(row)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_specs = {
        step_type: [
            {
                "channel": channel,
                "layer": layers[channel // num_query_heads],
                "query_head": channel % num_query_heads,
                "kv_head": (channel % num_query_heads) // repeat_groups,
            }
            for channel in channels
        ]
        for step_type, channels in selected.items()
    }
    payload = {
        "source": "all-head SVD32 Q/K reranking within lexical candidates",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "channel_selection_uses_train_labels_only": True,
        "steps": len(rows),
        "candidate_field": args.candidate_field,
        "candidate_limit": args.candidate_limit,
        "selected_channels": selected_specs,
        "fixed_pairs": fixed_pairs,
        "mean_score_seconds": statistics.fmean(score_seconds),
        "summaries": summarize(rows, methods),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
