from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize query-specific virtual source blocks from the LongBench code "
            "candidate-utility landscape into one sparse sidecar corpus."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--utility_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--memory_tokens", type=int, default=10_000_000)
    parser.add_argument("--observed_target_tokens", type=int, default=64)
    parser.add_argument(
        "--candidate_origins",
        default="bm25,e5,bm25_e5_rrf",
        help="Keep windows proposed by at least one named retrieval channel.",
    )
    parser.add_argument(
        "--max_retriever_rank",
        type=int,
        default=0,
        help="If positive, keep windows entering this depth in at least one named channel.",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    block_tokens = int(summary["block_tokens"])
    source_count = int(summary["source_blocks"])
    total_blocks = args.memory_tokens // block_tokens
    base_count = total_blocks - source_count
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    base_scope_ids = np.asarray(
        np.load(data_dir / "base_block_scope_ids.npy", mmap_mode="r"), dtype=np.int64
    )
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    metadata = {int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")}
    candidate_origins = {
        item.strip() for item in args.candidate_origins.split(",") if item.strip()
    }
    utility_rows = sorted(
        [
            row
            for row in read_jsonl(args.utility_rows)
            if any(
                origin in candidate_origins
                and (
                    args.max_retriever_rank <= 0
                    or int(rank) <= args.max_retriever_rank
                )
                for origin, rank in row["origins"].items()
            )
        ],
        key=lambda row: (int(row["query_id"]), int(row["window_start"])),
    )
    if any(
        bool(row["candidate_retrieval_uses_target"])
        or bool(row["selection_uses_future_segment_b"])
        for row in utility_rows
    ):
        raise RuntimeError("utility landscape contains target selection leakage")

    key_to_local: dict[tuple[Any, ...], int] = {}
    materialized_blocks: list[np.ndarray] = []
    materialized_scopes: list[int] = []

    def local_block(query_id: int, virtual_block_id: int) -> int:
        if virtual_block_id < base_count:
            key = ("base", virtual_block_id)
            tokens = base_blocks[virtual_block_id]
            scope = int(base_scope_ids[virtual_block_id])
        else:
            source_index = virtual_block_id - base_count
            if not 0 <= source_index < source_count:
                raise ValueError("virtual source block lies outside the query source")
            key = ("source", query_id, source_index)
            tokens = source_blocks[query_id, source_index]
            scope = int(metadata[query_id]["repo_index"])
        if key not in key_to_local:
            key_to_local[key] = len(materialized_blocks)
            materialized_blocks.append(np.asarray(tokens, dtype=np.int32))
            materialized_scopes.append(scope)
        return key_to_local[key]

    candidate_rows = []
    for row in utility_rows:
        query_id = int(row["query_id"])
        virtual_ids = [int(item) for item in row["block_ids"]]
        local_ids = [local_block(query_id, item) for item in virtual_ids]
        query_scope = int(metadata[query_id]["repo_index"])
        local_scopes = [materialized_scopes[item] for item in local_ids]
        origins = {str(key): int(value) for key, value in row["origins"].items()}
        candidate_rows.append(
            {
                "query_id": query_id,
                "candidate_id": int(row["window_start"]),
                "state_suffix_tokens": int(queries.shape[1] + args.observed_target_tokens),
                "previous_depth": 0,
                "expanded_depth": len(local_ids),
                "previous_block_ids": [],
                "expanded_block_ids": local_ids,
                "virtual_block_ids": virtual_ids,
                "origins": origins,
                "best_retriever_rank": min(origins.values()) if origins else None,
                "retriever_count": len(origins),
                "query_scope_id": query_scope,
                "candidate_scope_ids": local_scopes,
                "same_scope_fraction": sum(scope == query_scope for scope in local_scopes)
                / len(local_scopes),
                "same_scope_any": any(scope == query_scope for scope in local_scopes),
                "source_overlap": int(row["source_overlap"]),
                "delta_nll_observed_a": float(row["delta_nll_a"]),
                "delta_nll_future_b": float(row["delta_nll_b"]),
                "candidate_retrieval_uses_target": False,
                "segment_a_is_observed": True,
                "future_target_used": False,
                "reader_forward_used": False,
                "expanded_workset_reader_forward_used": False,
                "selection_uses_target": False,
            }
        )

    state_queries = np.concatenate(
        [
            np.asarray(queries, dtype=np.int32),
            np.asarray(targets[:, : args.observed_target_tokens], dtype=np.int32),
        ],
        axis=1,
    )
    np.save(output_dir / "base_blocks.npy", np.stack(materialized_blocks))
    np.save(
        output_dir / "base_block_scope_ids.npy",
        np.asarray(materialized_scopes, dtype=np.int32),
    )
    np.save(output_dir / "queries.npy", state_queries)
    with (output_dir / "candidate_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    output_summary = {
        "source": "real LongBench-v2 code utility candidates materialized for K/V sidecars",
        "source_data_dir": str(data_dir),
        "source_utility_rows": args.utility_rows,
        "query_samples": len(queries),
        "candidate_rows": len(candidate_rows),
        "candidate_origins": sorted(candidate_origins),
        "max_retriever_rank": args.max_retriever_rank,
        "materialized_candidate_blocks": len(materialized_blocks),
        "materialized_candidate_tokens": len(materialized_blocks) * block_tokens,
        "block_tokens": block_tokens,
        "state_tokens": int(state_queries.shape[1]),
        "observed_segment_a_tokens": args.observed_target_tokens,
        "future_segment_b_tokens": int(targets.shape[1] - args.observed_target_tokens),
        "scope_type": "repository_context",
        "contains_synthetic_text": False,
        "candidate_retrieval_uses_target": False,
        "future_target_used": False,
        "selection_uses_target": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output_summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
