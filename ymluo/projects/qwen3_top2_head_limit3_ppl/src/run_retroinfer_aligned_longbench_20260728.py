#!/usr/bin/env python3
"""Run official RetroInfer and its Full-Flash backend on aligned LongBench."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover - Windows-only static validation path
    resource = None

import run_controlled_public_kv_benchmark_v1 as lb
import run_sample_calibrated_longbench_20260717 as qksieve_lb
from audit_retroinfer_official_checkout_20260728 import OFFICIAL_COMMIT


FULL_METHOD = "retroinfer_stack_full_flash"
RETROINFER_METHOD = "retroinfer_official_aligned"
METHOD_TO_ATTENTION = {
    FULL_METHOD: "Full_Flash_Attn",
    RETROINFER_METHOD: "RetroInfer",
}
DEFAULT_TASKS = (
    "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,"
    "qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,"
    "gov_report,multi_news,lcc,repobench-p"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_checkout", required=True, type=Path)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--official_config_model_name",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument(
        "--methods",
        default=f"{FULL_METHOD},{RETROINFER_METHOD}",
    )
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--sample_offset_per_task", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_prompt_tokens", type=int, default=7500)
    parser.add_argument(
        "--prompt_truncation_mode",
        choices=("official_middle",),
        default="official_middle",
    )
    parser.add_argument("--prompt_wrapper", choices=("llama3",), default="llama3")
    parser.add_argument("--official_query_tail_tokens", type=int, default=8)
    parser.add_argument("--max_new_tokens_override", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=130_000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--retrieval_budget", type=float, default=0.018)
    parser.add_argument("--estimation_budget", type=float, default=0.232)
    parser.add_argument("--cache_ratio", type=float, default=0.05)
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--use_cuda_graph", action="store_true")
    return parser.parse_args()


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_TO_ATTENTION))
    if not methods or unknown:
        raise ValueError(f"unsupported RetroInfer methods: {unknown}")
    if len(set(methods)) != len(methods):
        raise ValueError("RetroInfer methods must not contain duplicates")
    return methods


def _git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_official_checkout(checkout: Path) -> str:
    commit = _git_output(checkout, "rev-parse", "HEAD")
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"RetroInfer checkout must be {OFFICIAL_COMMIT}, got {commit}"
        )
    dirty = _git_output(
        checkout,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if dirty:
        raise RuntimeError("RetroInfer checkout has tracked modifications")
    return commit


def import_official_runtime(checkout: Path) -> tuple[Any, Any, Any]:
    checkout_text = str(checkout.resolve())
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    model_hub = importlib.import_module("model_hub")
    config = importlib.import_module("config")
    return model_hub.LlamaModel, model_hub.QwenModel, config.generate_config


def load_official_model(
    args: argparse.Namespace,
    llama_model: Any,
    qwen_model: Any,
) -> Any:
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model_class = (
        qwen_model
        if "qwen" in args.official_config_model_name.lower()
        else llama_model
    )
    model = model_class(
        args.model_name_or_path,
        max_length=args.model_max_length,
        dtype=dtype,
        device_map=args.device_map,
    )
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    return model


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_cuda() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def _peak_cuda() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "gpu_peak_allocated_bytes": 0,
            "gpu_peak_reserved_bytes": 0,
        }
    return {
        "gpu_peak_allocated_bytes": sum(
            int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        ),
        "gpu_peak_reserved_bytes": sum(
            int(torch.cuda.max_memory_reserved(index))
            for index in range(torch.cuda.device_count())
        ),
    }


def _max_rss_bytes() -> int:
    # Linux reports KiB. This runner is only supported on the official Linux
    # CUDA stack, so keep the conversion explicit in the output contract.
    if resource is None:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


@torch.inference_mode()
def generate_aligned(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    attention_type: str,
    attention_config: dict[str, Any],
    max_new_tokens: int,
    stop_token_ids: set[int],
) -> dict[str, Any]:
    """Mirror official generation while applying the aligned stop policy."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("aligned RetroInfer evaluation requires batch size 1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model.attention_type = attention_type
    model.batch_size = 1
    model.input_length = int(input_ids.shape[1])
    model.max_new_length = int(max_new_tokens)
    model.prefill_bsz = 1
    model.prefill_method = "full"
    if model.input_length + max_new_tokens > model.max_length:
        raise ValueError("prompt plus output exceeds official model max length")

    _reset_peak_cuda()
    valid_start = np.zeros((1,), dtype=np.int64)
    _sync_cuda()
    started = time.perf_counter()
    model.init_kv_cache(valid_start, attention_config)
    _sync_cuda()
    cache_init_seconds = time.perf_counter() - started

    _sync_cuda()
    started = time.perf_counter()
    logits = model.prefill_forward(inputs_ids=input_ids)
    output_ids = model.sampling(logits, do_sample=False)
    _sync_cuda()
    prefill_seconds = time.perf_counter() - started

    _sync_cuda()
    started = time.perf_counter()
    model.move()
    _sync_cuda()
    cache_prepare_seconds = time.perf_counter() - started

    graph_capture_seconds = 0.0
    if attention_type == "RetroInfer":
        _sync_cuda()
        started = time.perf_counter()
        model.kv_cache.capture_cuda_graph()
        _sync_cuda()
        graph_capture_seconds = time.perf_counter() - started

    generated_ids: list[int] = []
    first_id = int(output_ids.reshape(-1)[0].item())
    if first_id not in stop_token_ids:
        generated_ids.append(first_id)

    decode_steps = 0
    _sync_cuda()
    started = time.perf_counter()
    while generated_ids and len(generated_ids) < max_new_tokens:
        logits = model.decode_forward(inputs_ids=output_ids)
        output_ids = model.sampling(logits, do_sample=False)
        decode_steps += 1
        next_id = int(output_ids.reshape(-1)[0].item())
        if next_id in stop_token_ids:
            break
        generated_ids.append(next_id)
    _sync_cuda()
    decode_seconds = time.perf_counter() - started

    prediction = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    total_seconds = (
        cache_init_seconds
        + prefill_seconds
        + cache_prepare_seconds
        + graph_capture_seconds
        + decode_seconds
    )
    return {
        "prediction": prediction,
        "generated_ids": generated_ids,
        "cache_init_seconds": cache_init_seconds,
        "prefill_seconds": prefill_seconds,
        "cache_prepare_seconds": cache_prepare_seconds,
        "graph_capture_seconds": graph_capture_seconds,
        "decode_seconds": decode_seconds,
        "decode_steps": decode_steps,
        "decode_tpot_seconds": (
            decode_seconds / decode_steps if decode_steps else None
        ),
        "total_seconds": total_seconds,
        "cpu_peak_rss_bytes": _max_rss_bytes(),
        **_peak_cuda(),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.is_file() or path.stat().st_size == 0
    fieldnames = list(row)
    if not needs_header:
        with path.open(encoding="utf-8", newline="") as handle:
            existing = csv.DictReader(handle).fieldnames
        if not existing:
            raise RuntimeError("existing RetroInfer CSV has no header")
        fieldnames = existing
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _example_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        tasks=args.tasks,
        longbench_data_dir=args.longbench_data_dir,
        max_samples_per_task=args.max_samples_per_task,
        sample_offset_per_task=args.sample_offset_per_task,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_new_tokens_override=args.max_new_tokens_override,
        max_prompt_tokens=args.max_prompt_tokens,
        prompt_truncation_mode=args.prompt_truncation_mode,
        prompt_wrapper=args.prompt_wrapper,
        official_query_tail_tokens=args.official_query_tail_tokens,
        max_context_tokens=0,
    )


def _prompt_sha256(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(
        input_ids.detach().cpu().to(torch.int64).numpy().tobytes()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    methods = parse_methods(args.methods)
    official_commit = verify_official_checkout(args.official_checkout)
    llama_model, qwen_model, generate_config = import_official_runtime(
        args.official_checkout
    )
    model = load_official_model(args, llama_model, qwen_model)
    tokenizer = model.tokenizer
    examples = qksieve_lb.load_examples(_example_args(args))

    output_path = args.output_dir / "sample_results.csv"
    completed = {
        (row["task"], row["sample_id"], row["method"])
        for row in _read_rows(output_path)
    }
    for index, example in enumerate(examples, 1):
        bundle = qksieve_lb.build_bundle(
            tokenizer,
            example,
            _example_args(args),
        )
        input_ids = bundle.input_ids.to(model.layers[0].device)
        stop_ids = qksieve_lb.longbench_stop_token_ids(
            tokenizer,
            example.task,
        )
        prompt_hash = _prompt_sha256(bundle.input_ids)
        print(
            f"[{index}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prompt={input_ids.shape[1]}",
            flush=True,
        )
        for method in methods:
            key = (example.task, example.sample_id, method)
            if key in completed:
                print(f"  {method}: already complete", flush=True)
                continue
            attention_type = METHOD_TO_ATTENTION[method]
            config = generate_config(
                args.official_config_model_name,
                int(input_ids.shape[1]),
                attention_type,
                retrieval_budget=args.retrieval_budget,
                estimation_budget=args.estimation_budget,
                cache_ratio=args.cache_ratio,
                use_cuda_graph=args.use_cuda_graph,
                gpu_only=args.gpu_only,
            )
            result = generate_aligned(
                model,
                tokenizer,
                input_ids,
                attention_type=attention_type,
                attention_config=config,
                max_new_tokens=example.max_new_tokens,
                stop_token_ids=stop_ids,
            )
            score = lb.score_prediction(
                example.metric,
                result["prediction"],
                example.answers,
                example.all_classes,
                task=example.task,
            )
            row = {
                "task": example.task,
                "sample_id": example.sample_id,
                "method": method,
                "executed_path": method,
                "attention_type": attention_type,
                "protocol": "qksieve_aligned_longbench_v1",
                "official_repository_commit": official_commit,
                "model_name_or_path": args.model_name_or_path,
                "official_config_model_name": (
                    args.official_config_model_name
                ),
                "dtype": args.dtype,
                "prompt_tokens": int(input_ids.shape[1]),
                "prompt_sha256": prompt_hash,
                "prompt_truncation_mode": args.prompt_truncation_mode,
                "prompt_wrapper": args.prompt_wrapper,
                "stop_token_ids": json.dumps(sorted(stop_ids)),
                "max_new_tokens": example.max_new_tokens,
                "generated_tokens": len(result["generated_ids"]),
                "decode_steps": result["decode_steps"],
                "prediction": result["prediction"],
                "answers": json.dumps(
                    example.answers,
                    ensure_ascii=False,
                ),
                "score": score,
                "retrieval_budget": (
                    args.retrieval_budget
                    if method == RETROINFER_METHOD
                    else 1.0
                ),
                "estimation_budget": (
                    args.estimation_budget
                    if method == RETROINFER_METHOD
                    else 0.0
                ),
                "cache_ratio": (
                    args.cache_ratio
                    if method == RETROINFER_METHOD
                    else 1.0
                ),
                "gpu_only": int(args.gpu_only),
                "use_cuda_graph": int(args.use_cuda_graph),
                **{
                    name: value
                    for name, value in result.items()
                    if name not in {"prediction", "generated_ids"}
                },
            }
            _append_row(output_path, row)
            completed.add(key)
            print(
                f"  {method}: score={score:.6f} "
                f"total={result['total_seconds']:.4f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
