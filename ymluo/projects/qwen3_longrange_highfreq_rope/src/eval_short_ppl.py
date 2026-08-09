from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
ABLATION_SRC = PROJECTS / "qwen3_ruler_head_frequency_ablation" / "src"
RNOPE_SRC = PROJECTS / "qwen3_inference_rnope" / "src"
RULER_SRC = PROJECTS / "qwen3_ruler32k_rope_method" / "src"
LONG_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
for directory in (ABLATION_SRC, RNOPE_SRC, RULER_SRC, LONG_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from head_frequency_intervention import HeadFrequencyIntervention  # noqa: E402
import run_inference_rnope_ruler as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--specs-json", type=Path, required=True)
    parser.add_argument("--variants", default="native_rope,late_f00_07_delete")
    parser.add_argument("--sequence-lengths", default="2048,4096")
    parser.add_argument("--sequences-per-length", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_text(path: Path) -> str:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["text"])
    rows = [str(value) for value in table.column("text").to_pylist() if str(value).strip()]
    if not rows:
        raise RuntimeError("parquet contains no non-empty text")
    return "\n\n".join(rows)


def token_windows(token_ids: list[int], length: int, count: int) -> list[list[int]]:
    width = length + 1
    windows = [token_ids[index * width : (index + 1) * width] for index in range(count)]
    if any(len(window) != width for window in windows):
        raise ValueError(
            f"need at least {width * count} tokens for {count} windows of length {length}"
        )
    return windows


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 256) -> float:
    total = 0.0
    tokens = int(labels.numel())
    for start in range(0, tokens, chunk):
        stop = min(tokens, start + chunk)
        total += float(
            F.cross_entropy(
                logits[:, start:stop, :].float().reshape(-1, logits.shape[-1]),
                labels[:, start:stop].reshape(-1),
                reduction="sum",
            ).item()
        )
    return total


def main() -> None:
    args = parse_args()
    raw_specs = json.loads(args.specs_json.read_text(encoding="utf-8"))
    by_name = {str(spec["name"]): spec for spec in raw_specs["specs"]}
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    specs = [by_name[variant] for variant in variants]
    lengths = [int(value) for value in args.sequence_lengths.split(",")]

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype="bfloat16",
        attn_implementation="sdpa",
        original_max_position_embeddings=40960,
        global_max_position=40960,
        load_in_4bit=True,
        device_map="auto",
    )
    model, tokenizer = base.core.local_global.load_model(load_args)
    intervention = HeadFrequencyIntervention(model)
    text = read_text(args.parquet)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []

    for length in lengths:
        windows = token_windows(token_ids, length, args.sequences_per_length)
        for spec in specs:
            total_nll = 0.0
            total_tokens = 0
            started = time.perf_counter()
            with intervention.activate(spec), torch.inference_mode():
                for window in windows:
                    values = torch.tensor(window, dtype=torch.long, device=device)[None, :]
                    logits = model(input_ids=values[:, :-1], use_cache=False).logits
                    labels = values[:, 1:]
                    total_nll += sequence_nll(logits, labels)
                    total_tokens += int(labels.numel())
                    del values, logits, labels
                    base.core.local_global.clear_allocator()
            mean_nll = total_nll / total_tokens
            results.append(
                {
                    "variant": spec["name"],
                    "sequence_length": length,
                    "sequences": len(windows),
                    "tokens": total_tokens,
                    "mean_nll": mean_nll,
                    "ppl": math.exp(mean_nll),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(json.dumps(results[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset": str(args.parquet),
                "model": args.model_name_or_path,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
