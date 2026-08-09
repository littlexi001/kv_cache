from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from evaluate_pg19_candidate_utility_landscape import context_for_window
from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched short counterfactual utility probes."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch_sizes", default="1,2,4,8,16")
    parser.add_argument("--query_id", type=int, default=0)
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--target_tokens", type=int, default=64)
    parser.add_argument("--window_blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.inference_mode()
def probe_batch(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    *,
    prompt_tokens: int,
    target_tokens: int,
) -> torch.Tensor:
    hidden = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    positions = torch.arange(
        prompt_tokens - 1,
        prompt_tokens + target_tokens - 1,
        device=input_ids.device,
    )
    selected = hidden.index_select(1, positions)
    logits = model.lm_head(selected).float()
    targets = input_ids[:, prompt_tokens : prompt_tokens + target_tokens]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
    )


def main() -> None:
    args = parse_args()
    batch_sizes = sorted(
        {int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()}
    )
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("batch_sizes must be positive")
    data_dir = Path(args.data_dir)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    all_source_blocks = np.load(data_dir / "source_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")
    base_count = min(
        len(base_blocks),
        (10_000_000 - int(summary["source_tokens"])) // int(summary["block_tokens"]),
    )
    source_blocks = np.asarray(all_source_blocks[args.query_id])
    source_count = len(source_blocks)
    rows = [
        row
        for row in read_jsonl(args.rows)
        if int(row["query_id"]) == args.query_id
        and any(method in row["origins"] for method in ("bm25", "e5", "bm25_e5_rrf"))
    ]
    if not rows:
        raise RuntimeError("no retrieval candidates found for query")
    contexts = []
    for row in rows:
        contexts.append(
            context_for_window(
                int(row["window_start"]),
                window_blocks=args.window_blocks,
                base_blocks=base_blocks,
                source_blocks=source_blocks,
                base_count=base_count,
            )
        )
    while len(contexts) < max(batch_sizes):
        contexts.extend(contexts)

    query = np.asarray(queries[args.query_id, -args.prefix_tokens :], dtype=np.int64)
    target = np.asarray(targets[args.query_id, : args.target_tokens], dtype=np.int64)
    prompt_tokens = len(contexts[0]) + len(query)
    sequence_tokens = prompt_tokens + len(target)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    results = []
    for batch_size in batch_sizes:
        array = np.stack(
            [np.concatenate([context, query, target]) for context in contexts[:batch_size]]
        )
        input_ids = torch.from_numpy(array).to(device)
        for _ in range(args.warmup):
            probe_batch(
                model,
                input_ids,
                prompt_tokens=prompt_tokens,
                target_tokens=len(target),
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            probe_batch(
                model,
                input_ids,
                prompt_tokens=prompt_tokens,
                target_tokens=len(target),
            )
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000.0)
        median_ms = float(np.median(latencies))
        results.append(
            {
                "batch_size": batch_size,
                "median_batch_ms": median_ms,
                "p95_batch_ms": float(np.quantile(latencies, 0.95)),
                "median_ms_per_candidate": median_ms / batch_size,
                "candidate_probes_per_second": 1000.0 * batch_size / median_ms,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )

    output = {
        "source": "real LongBench code candidate utility probe microbenchmark",
        "model": args.model_name_or_path,
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "query_id": args.query_id,
        "window_tokens": len(contexts[0]),
        "query_tokens": len(query),
        "observed_target_tokens": len(target),
        "sequence_tokens_per_candidate": sequence_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "reader-forward and LM-head only; excludes 10M retrieval, I/O, and generation",
        "results": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
