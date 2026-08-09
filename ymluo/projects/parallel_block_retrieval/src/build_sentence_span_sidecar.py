from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from run_step_state_kv_span_retrieval import sentence_token_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a token-boundary-only sentence sidecar for block-local KV scoring."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--log_every", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    offsets = np.zeros(int(blocks.shape[0]) + 1, dtype=np.int64)
    spans: list[tuple[int, int]] = []
    failed_blocks = []
    started = time.perf_counter()
    for block_id in range(int(blocks.shape[0])):
        try:
            local_spans = sentence_token_spans(blocks[block_id].tolist(), tokenizer)
        except ValueError:
            local_spans = [(0, int(blocks.shape[1]))]
            failed_blocks.append(block_id)
        spans.extend(local_spans)
        offsets[block_id + 1] = len(spans)
        if (block_id + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "blocks": block_id + 1,
                        "total_blocks": int(blocks.shape[0]),
                        "spans": len(spans),
                    }
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - started
    span_array = np.asarray(spans, dtype=np.int16)
    np.save(output_dir / "block_span_offsets.npy", offsets)
    np.save(output_dir / "sentence_spans.npy", span_array)
    lengths = span_array[:, 1].astype(np.int32) - span_array[:, 0].astype(np.int32)
    summary = {
        "source": "token boundaries only; no embeddings or synthetic vectors",
        "contains_synthetic_vectors": False,
        "corpus_dir": str(corpus_dir),
        "blocks": int(blocks.shape[0]),
        "block_tokens": int(blocks.shape[1]),
        "sentence_spans": int(span_array.shape[0]),
        "mean_spans_per_block": float(span_array.shape[0] / blocks.shape[0]),
        "mean_span_tokens": float(lengths.mean()),
        "median_span_tokens": float(np.median(lengths)),
        "p95_span_tokens": float(np.quantile(lengths, 0.95)),
        "failed_blocks_with_full_block_fallback": failed_blocks,
        "build_seconds": elapsed,
        "sidecar_bytes": int(offsets.nbytes + span_array.nbytes),
        "offsets_path": str(output_dir / "block_span_offsets.npy"),
        "spans_path": str(output_dir / "sentence_spans.npy"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
