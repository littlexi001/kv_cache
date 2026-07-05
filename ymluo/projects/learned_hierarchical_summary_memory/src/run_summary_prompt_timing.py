from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_classic_kv_retrieval_summary_benchmark import (
    Config as ClassicConfig,
    Case,
    adaptive_level,
    build_cases,
)


MEMORY_LEVELS = ("summary10", "summary100", "summary1000", "raw")


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    tasks_per_variant: int
    distractor_records: int
    seed: int
    raw_context_tokens: int
    summary10_tokens: int
    summary100_tokens: int
    summary1000_tokens: int
    methods: tuple[str, ...]
    device: str
    dtype: str
    attn_implementation: str
    repeats: int
    warmup: int
    warmup_tokens: int
    prompt_overhead_tokens: int
    max_input_tokens: int
    pad_context_to_budget: bool
    skip_model: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Wall-clock timing for summary-memory prompt prefill.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--tasks_per_variant", type=int, default=6)
    parser.add_argument("--distractor_records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026070304)
    parser.add_argument("--raw_context_tokens", type=int, default=10_000)
    parser.add_argument("--summary10_tokens", type=int, default=10)
    parser.add_argument("--summary100_tokens", type=int, default=100)
    parser.add_argument("--summary1000_tokens", type=int, default=1_000)
    parser.add_argument("--methods", default="full_raw,adaptive_no_raw,adaptive_with_raw,summary1000_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup_tokens", type=int, default=128)
    parser.add_argument("--prompt_overhead_tokens", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument(
        "--pad_context_to_budget",
        action="store_true",
        help="Pad tokenized prompt to selected memory budget plus prompt_overhead_tokens.",
    )
    parser.add_argument("--skip_model", action="store_true", help="Only measure context building and estimate tokens.")
    args = parser.parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    return Config(**{**vars(args), "methods": methods})


def classic_config(config: Config) -> ClassicConfig:
    return ClassicConfig(
        output_dir=config.output_dir,
        tasks_per_variant=config.tasks_per_variant,
        distractor_records=config.distractor_records,
        seed=config.seed,
        raw_context_tokens=config.raw_context_tokens,
        summary10_tokens=config.summary10_tokens,
        summary100_tokens=config.summary100_tokens,
        summary1000_tokens=config.summary1000_tokens,
    )


def level_for_method(method: str, case: Case) -> str:
    if method == "full_raw":
        return "raw"
    if method == "summary10_only":
        return "summary10"
    if method == "summary100_only":
        return "summary100"
    if method == "summary1000_only":
        return "summary1000"
    if method == "adaptive_no_raw":
        return adaptive_level(case, allow_raw=False)
    if method == "adaptive_with_raw":
        return adaptive_level(case, allow_raw=True)
    raise ValueError(f"unknown method: {method}")


def level_budget(config: Config, level: str) -> int:
    return {
        "summary10": config.summary10_tokens,
        "summary100": config.summary100_tokens,
        "summary1000": config.summary1000_tokens,
        "raw": config.raw_context_tokens,
    }[level]


def context_for_level(case: Case, level: str, config: Config) -> str:
    if level == "summary10":
        return case.summary10
    if level == "summary100":
        return case.summary100
    if level == "summary1000":
        return case.summary1000
    if level == "raw":
        return case.raw_context
    raise ValueError(level)


def query_for_case(case: Case) -> str:
    prompts = {
        "passkey": "What is the secret passkey?",
        "needle": "What answer label is mapped by the needle key?",
        "kv_lookup": "What label should be returned for the target key?",
        "conflict_latest": "What is the latest value after updates?",
        "multihop": "Follow the project to artifact to action chain. What is the final action?",
        "exact_code": "What is the exact code string?",
    }
    return prompts.get(case.variant, "Return the answer stored in the context.")


def build_prompt(case: Case, method: str, config: Config) -> tuple[str, str]:
    level = level_for_method(method, case)
    context = context_for_level(case, level, config)
    prompt = (
        "You are answering a memory retrieval question. Use only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query_for_case(case)}\n"
        "Answer only with the final value.\nAnswer:"
    )
    return level, prompt


def route_mix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    mixes: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        total = len(items)
        counts = Counter(row["memory_level"] for row in items)
        mix: dict[str, Any] = {"method": method, "cases": total}
        for level in MEMORY_LEVELS:
            name = "full_attention" if level == "raw" else level
            mix[f"{name}_count"] = counts[level]
            mix[f"{name}_ratio"] = counts[level] / max(1, total)
        mixes.append(mix)
    return mixes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[dtype_name]


def synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    full_items = grouped.get("full_raw", [])
    full_avg_tokens = statistics.mean(row["input_tokens"] for row in full_items) if full_items else 0.0
    full_avg_prefill = statistics.mean(row["prefill_seconds"] for row in full_items) if full_items else 0.0
    full_avg_pipeline = statistics.mean(row["pipeline_seconds"] for row in full_items) if full_items else 0.0

    summary: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        input_tokens = [row["input_tokens"] for row in items]
        prefill = [row["prefill_seconds"] for row in items]
        pipeline = [row["pipeline_seconds"] for row in items]
        tokenizer = [row["tokenize_seconds"] for row in items]
        build = [row["build_seconds"] for row in items]
        peak_memory = [row["peak_memory_mb"] for row in items if row["peak_memory_mb"] is not None]
        avg_prefill = statistics.mean(prefill)
        avg_pipeline = statistics.mean(pipeline)
        avg_tokens = statistics.mean(input_tokens)
        summary.append(
            {
                "method": method,
                "cases": len(items),
                "avg_input_tokens": avg_tokens,
                "token_ratio_vs_full_raw": avg_tokens / full_avg_tokens if full_avg_tokens else 0.0,
                "avg_build_seconds": statistics.mean(build),
                "avg_tokenize_seconds": statistics.mean(tokenizer),
                "avg_prefill_seconds": avg_prefill,
                "p50_prefill_seconds": statistics.median(prefill),
                "p90_prefill_seconds": percentile(prefill, 0.9),
                "avg_pipeline_seconds": avg_pipeline,
                "time_ratio_vs_full_raw_prefill": avg_prefill / full_avg_prefill if full_avg_prefill else 0.0,
                "speedup_vs_full_raw_prefill": full_avg_prefill / avg_prefill if avg_prefill > 0 and full_avg_prefill else 0.0,
                "time_ratio_vs_full_raw_pipeline": avg_pipeline / full_avg_pipeline if full_avg_pipeline else 0.0,
                "speedup_vs_full_raw_pipeline": full_avg_pipeline / avg_pipeline if avg_pipeline > 0 and full_avg_pipeline else 0.0,
                "avg_peak_memory_mb": statistics.mean(peak_memory) if peak_memory else None,
            }
        )
    return summary


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def run_skip_model(config: Config, cases: list[Case]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in config.methods:
        for case in cases:
            start = time.perf_counter()
            level, prompt = build_prompt(case, method, config)
            build_seconds = time.perf_counter() - start
            input_tokens = estimate_tokens(prompt)
            if config.pad_context_to_budget:
                input_tokens = max(input_tokens, level_budget(config, level) + config.prompt_overhead_tokens)
            rows.append(
                {
                    "method": method,
                    "case_id": case.case_id,
                    "variant": case.variant,
                    "memory_level": level,
                    "input_tokens": input_tokens,
                    "build_seconds": build_seconds,
                    "tokenize_seconds": 0.0,
                    "prefill_seconds": 0.0,
                    "pipeline_seconds": build_seconds,
                    "tokens_per_second": 0.0,
                    "peak_memory_mb": None,
                }
            )
    return rows


def run_hf_model(config: Config, cases: list[Case]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    input_device = next(model.parameters()).device

    rows: list[dict[str, Any]] = []
    warmed = False
    repeats = max(1, config.repeats)
    for method in config.methods:
        for case in cases:
            pipeline_start = time.perf_counter()
            build_start = time.perf_counter()
            level, prompt = build_prompt(case, method, config)
            build_seconds = time.perf_counter() - build_start

            tokenize_start = time.perf_counter()
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if config.pad_context_to_budget:
                target_tokens = level_budget(config, level) + config.prompt_overhead_tokens
                if input_ids.shape[1] < target_tokens:
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    pad = input_ids.new_full((input_ids.shape[0], target_tokens - input_ids.shape[1]), pad_token_id)
                    input_ids = torch.cat([input_ids, pad], dim=1)
            if config.max_input_tokens > 0 and input_ids.shape[1] > config.max_input_tokens:
                input_ids = input_ids[:, -config.max_input_tokens :]
            input_ids = input_ids.to(input_device)
            tokenize_seconds = time.perf_counter() - tokenize_start

            if not warmed and config.warmup > 0:
                warm_len = min(input_ids.shape[1], max(1, config.warmup_tokens))
                warm_ids = input_ids[:, :warm_len]
                with torch.inference_mode():
                    for _ in range(config.warmup):
                        _ = model(input_ids=warm_ids, use_cache=True)
                synchronize(torch, input_device)
                warmed = True

            if input_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(input_device)

            timings: list[float] = []
            with torch.inference_mode():
                for _ in range(repeats):
                    synchronize(torch, input_device)
                    forward_start = time.perf_counter()
                    outputs = model(input_ids=input_ids, use_cache=True)
                    synchronize(torch, input_device)
                    timings.append(time.perf_counter() - forward_start)
                    del outputs

            prefill_seconds = statistics.mean(timings)
            peak_memory_mb = None
            if input_device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(input_device) / (1024 * 1024)
            pipeline_seconds = time.perf_counter() - pipeline_start
            input_tokens = int(input_ids.shape[1])
            rows.append(
                {
                    "method": method,
                    "case_id": case.case_id,
                    "variant": case.variant,
                    "memory_level": level,
                    "input_tokens": input_tokens,
                    "build_seconds": build_seconds,
                    "tokenize_seconds": tokenize_seconds,
                    "prefill_seconds": prefill_seconds,
                    "pipeline_seconds": pipeline_seconds,
                    "tokens_per_second": input_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
                    "peak_memory_mb": peak_memory_mb,
                }
            )
    return rows


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(classic_config(config))
    rows = run_skip_model(config, cases) if config.skip_model else run_hf_model(config, cases)
    timing_summary = summarize_timing(rows)
    mixes = route_mix(rows)

    write_csv(output_dir / "timing_rows.csv", rows)
    write_csv(output_dir / "timing_summary.csv", timing_summary)
    write_csv(output_dir / "route_mix.csv", mixes)
    payload = {
        "config": asdict(config),
        "cases": len(cases),
        "timing_summary": timing_summary,
        "route_mix": mixes,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("method,cases,avg_input_tokens,avg_prefill_seconds,speedup_vs_full_raw_prefill")
    for row in timing_summary:
        print(
            f"{row['method']},{row['cases']},{row['avg_input_tokens']:.1f},"
            f"{row['avg_prefill_seconds']:.6f},{row['speedup_vs_full_raw_prefill']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
