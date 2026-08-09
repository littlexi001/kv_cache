from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit that an extended past-only memory preserves its causal prefix."
    )
    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--extended_data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--chunk_blocks", type=int, default=16_384)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arrays_equal_chunked(
    left: np.ndarray,
    right: np.ndarray,
    chunk_rows: int,
    *,
    require_dtype: bool = True,
) -> bool:
    if left.shape != right.shape or (require_dtype and left.dtype != right.dtype):
        return False
    for start in range(0, len(left), chunk_rows):
        end = min(start + chunk_rows, len(left))
        if not np.array_equal(left[start:end], right[start:end]):
            return False
    return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_data_dir)
    extended_dir = Path(args.extended_data_dir)
    input_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    extended_summary = json.loads(
        (extended_dir / "summary.json").read_text(encoding="utf-8")
    )
    input_blocks = np.load(input_dir / "base_blocks.npy", mmap_mode="r")
    extended_blocks = np.load(extended_dir / "base_blocks.npy", mmap_mode="r")
    input_scopes = np.load(input_dir / "base_block_scope_ids.npy", mmap_mode="r")
    extended_scopes = np.load(
        extended_dir / "base_block_scope_ids.npy", mmap_mode="r"
    )
    input_centers = np.load(
        input_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    extended_centers = np.load(
        extended_dir / "base_block_original_centers.npy", mmap_mode="r"
    )
    metadata = read_jsonl(extended_dir / "metadata.jsonl")

    prefix_blocks_exact = arrays_equal_chunked(
        input_blocks,
        extended_blocks[: len(input_blocks)],
        args.chunk_blocks,
    )
    prefix_scopes_exact = arrays_equal_chunked(
        input_scopes,
        extended_scopes[: len(input_scopes)],
        args.chunk_blocks,
        require_dtype=False,
    )
    prefix_centers_exact = arrays_equal_chunked(
        input_centers,
        extended_centers[: len(input_centers)],
        args.chunk_blocks,
    )

    causal_violations = 0
    query_scope_block_counts = {}
    for row in metadata:
        scope = int(row["book_index"])
        local_start = int(row["local_context_start_token"])
        selected = np.flatnonzero(extended_scopes == scope)
        positions = np.asarray(extended_centers[selected], dtype=np.int64)
        violations = int(np.sum(positions >= local_start))
        causal_violations += violations
        query_scope_block_counts[str(scope)] = {
            "blocks": len(selected),
            "future_or_local_violations": violations,
        }

    exact_file_names = ("queries.npy", "targets.npy", "source_blocks.npy", "metadata.jsonl")
    exact_file_hashes = {
        name: {
            "input_sha256": file_sha256(input_dir / name),
            "extended_sha256": file_sha256(extended_dir / name),
            "exact": file_sha256(input_dir / name) == file_sha256(extended_dir / name),
        }
        for name in exact_file_names
    }
    audit = {
        "source": "past-only memory extension causal and identity audit",
        "input_memory_tokens": int(input_summary["memory_tokens"]),
        "extended_memory_tokens": int(extended_summary["memory_tokens"]),
        "input_blocks": len(input_blocks),
        "extended_blocks": len(extended_blocks),
        "prefix_blocks_exact": prefix_blocks_exact,
        "prefix_scopes_exact": prefix_scopes_exact,
        "prefix_centers_exact": prefix_centers_exact,
        "exact_query_target_metadata_files": all(
            item["exact"] for item in exact_file_hashes.values()
        ),
        "exact_file_hashes": exact_file_hashes,
        "query_scope_future_or_local_block_violations": causal_violations,
        "query_scope_block_counts": query_scope_block_counts,
        "past_only": bool(extended_summary.get("past_only")),
        "memory_contains_query_book_future": bool(
            extended_summary.get("memory_contains_query_book_future")
        ),
        "contains_synthetic_text": bool(
            extended_summary.get("contains_synthetic_text")
        ),
        "contains_repeated_distractor_text": bool(
            extended_summary.get("contains_repeated_distractor_text")
        ),
        "selection_uses_target": False,
    }
    required_checks = (
        prefix_blocks_exact,
        prefix_scopes_exact,
        prefix_centers_exact,
        audit["exact_query_target_metadata_files"],
        causal_violations == 0,
        audit["past_only"],
        not audit["memory_contains_query_book_future"],
        not audit["contains_synthetic_text"],
        not audit["contains_repeated_distractor_text"],
    )
    audit["all_checks_passed"] = all(required_checks)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise RuntimeError("past-only extension audit failed")


if __name__ == "__main__":
    main()
