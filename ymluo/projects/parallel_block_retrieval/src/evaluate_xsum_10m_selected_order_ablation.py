from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from evaluate_xsum_10m_retrieval_ppl import resolve_dtype, target_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score fixed XSum 10M retrieval pages under different page orders."
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


def materialize_context(
    ordered_ids: list[int],
    *,
    base_blocks: np.ndarray,
    source_blocks: np.ndarray,
    base_count: int,
) -> np.ndarray:
    pieces = []
    for block_id in ordered_ids:
        if block_id < base_count:
            pieces.append(np.asarray(base_blocks[block_id], dtype=np.int32))
        else:
            pieces.append(
                np.asarray(source_blocks[block_id - base_count], dtype=np.int32)
            )
    return np.stack(pieces).reshape(-1) if pieces else np.empty(0, dtype=np.int32)


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    source_count = int(summary["source_blocks"])
    total_blocks = 10_000_000 // int(summary["block_tokens"])
    base_count = total_blocks - source_count
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
        retrieval_order = ordered_unique(selections[query_id]["selected_block_ids"])
        orders = {
            "original_old_to_new": sorted(retrieval_order),
            "reverse_new_to_old": sorted(retrieval_order, reverse=True),
            "retrieval_score_order": retrieval_order,
        }
        for order_name, block_ids in orders.items():
            context = materialize_context(
                block_ids,
                base_blocks=base_blocks,
                source_blocks=source_blocks[query_id],
                base_count=base_count,
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
