from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from evaluate_xsum_10m_retrieval_ppl import resolve_dtype, target_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score fixed real-10M retrieval pages in score order versus within-scope "
            "causal and reverse-causal order."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--selection_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="bm25_e5_rrf")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ordered_unique(values: Iterable[int]) -> list[int]:
    output = []
    seen = set()
    for value in values:
        value = int(value)
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def page_scope_and_position(
    block_id: int,
    query_id: int,
    *,
    base_count: int,
    base_scope_ids: np.ndarray | None,
    base_centers: np.ndarray | None,
) -> tuple[int, float]:
    if block_id >= base_count:
        return 1_000_000_000 + query_id, float(block_id - base_count)
    scope = int(base_scope_ids[block_id]) if base_scope_ids is not None else 0
    position = float(base_centers[block_id]) if base_centers is not None else float(block_id)
    return scope, position


def scope_order(
    retrieval_ids: list[int],
    query_id: int,
    *,
    base_count: int,
    base_scope_ids: np.ndarray | None,
    base_centers: np.ndarray | None,
    reverse: bool,
) -> list[int]:
    scope_sequence = []
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for block_id in retrieval_ids:
        scope, position = page_scope_and_position(
            block_id,
            query_id,
            base_count=base_count,
            base_scope_ids=base_scope_ids,
            base_centers=base_centers,
        )
        if scope not in grouped:
            scope_sequence.append(scope)
        grouped[scope].append((position, block_id))
    return [
        block_id
        for scope in scope_sequence
        for _, block_id in sorted(grouped[scope], reverse=reverse)
    ]


def materialize_context(
    ordered_ids: list[int],
    query_id: int,
    *,
    base_blocks: np.ndarray,
    source_blocks: np.ndarray | None,
) -> np.ndarray:
    base_count = len(base_blocks)
    pieces = []
    for block_id in ordered_ids:
        if block_id < base_count:
            pieces.append(np.asarray(base_blocks[block_id], dtype=np.int32))
        elif source_blocks is not None:
            pieces.append(
                np.asarray(source_blocks[query_id, block_id - base_count], dtype=np.int32)
            )
        else:
            raise IndexError(f"block {block_id} exceeds base memory without source blocks")
    return np.stack(pieces).reshape(-1) if pieces else np.empty(0, dtype=np.int32)


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_path = data_dir / "source_blocks.npy"
    source_blocks = np.load(source_path, mmap_mode="r") if source_path.exists() else None
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    scope_path = data_dir / "base_block_scope_ids.npy"
    base_scope_ids = np.load(scope_path, mmap_mode="r") if scope_path.exists() else None
    centers_path = data_dir / "base_block_original_centers.npy"
    base_centers = np.load(centers_path, mmap_mode="r") if centers_path.exists() else None
    selections = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.selection_rows))
        if str(row["method"]) == args.method
    }
    query_ids = sorted(selections)
    if args.max_queries > 0:
        query_ids = query_ids[: args.max_queries]
    local_ids = [query_id for query_id in query_ids if query_id % args.world_size == args.rank]

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    rows = []
    for index, query_id in enumerate(local_ids):
        selection = selections[query_id]
        id_key = (
            "selected_block_ids"
            if "selected_block_ids" in selection
            else "top_block_ids"
        )
        retrieval_order = ordered_unique(selection[id_key])
        orders = {
            "retrieval_score_order": retrieval_order,
            "scope_old_to_new": scope_order(
                retrieval_order,
                query_id,
                base_count=len(base_blocks),
                base_scope_ids=base_scope_ids,
                base_centers=base_centers,
                reverse=False,
            ),
            "scope_new_to_old": scope_order(
                retrieval_order,
                query_id,
                base_count=len(base_blocks),
                base_scope_ids=base_scope_ids,
                base_centers=base_centers,
                reverse=True,
            ),
        }
        for order_name, block_ids in orders.items():
            context = materialize_context(
                block_ids,
                query_id,
                base_blocks=base_blocks,
                source_blocks=source_blocks,
            )
            mean_nll, total_nll, target_tokens, seconds, input_tokens = target_nll(
                model,
                context,
                np.asarray(queries[query_id], dtype=np.int32),
                np.asarray(targets[query_id], dtype=np.int32),
                device,
            )
            rows.append(
                {
                    "query_id": query_id,
                    "method": args.method,
                    "order": order_name,
                    "selected_block_ids": block_ids,
                    "mean_nll": mean_nll,
                    "total_nll": total_nll,
                    "target_tokens": target_tokens,
                    "forward_seconds": seconds,
                    "model_input_tokens": input_tokens,
                    "selection_uses_target": False,
                }
            )
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "completed": index + 1,
                    "queries": len(local_ids),
                    "query_id": query_id,
                }
            ),
            flush=True,
        )
    with (output_dir / f"rows_rank{args.rank:03d}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
