from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from profile_real_qk import read_jsonl
from profile_sparse_candidate_k import candidate_block_ids
from run_step_state_kv_span_retrieval import sentence_token_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sentence token boundaries only for retrieved candidate blocks."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="anchor_candidates")
    parser.add_argument("--candidate_limit", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    rows = read_jsonl(Path(args.candidate_rows_path))
    block_ids = candidate_block_ids(rows, args.candidate_field, args.candidate_limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    offsets = np.zeros(len(block_ids) + 1, dtype=np.int64)
    spans = []
    failed = []
    started = time.perf_counter()
    for offset, block_id in enumerate(block_ids):
        try:
            local_spans = sentence_token_spans(blocks[block_id].tolist(), tokenizer)
        except ValueError:
            local_spans = [(0, int(blocks.shape[1]))]
            failed.append(block_id)
        spans.extend(local_spans)
        offsets[offset + 1] = len(spans)
    span_array = np.asarray(spans, dtype=np.int16)
    np.save(output_dir / "block_ids.npy", np.asarray(block_ids, dtype=np.int32))
    np.save(output_dir / "block_span_offsets.npy", offsets)
    np.save(output_dir / "sentence_spans.npy", span_array)
    elapsed = time.perf_counter() - started
    summary = {
        "source": "sparse token-boundary-only sentence sidecar",
        "contains_synthetic_vectors": False,
        "sparse": True,
        "blocks": len(block_ids),
        "block_tokens": int(blocks.shape[1]),
        "sentence_spans": len(spans),
        "failed_blocks_with_full_block_fallback": failed,
        "build_seconds": elapsed,
        "sidecar_bytes": int(
            np.asarray(block_ids, dtype=np.int32).nbytes
            + offsets.nbytes
            + span_array.nbytes
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
