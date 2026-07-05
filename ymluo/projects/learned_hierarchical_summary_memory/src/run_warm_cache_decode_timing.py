from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config as BenchConfig,
    build_memory_for_action,
    build_prompt,
    load_longbench_cases,
    load_ruler_cases,
    parse_csv_tuple,
    parse_int_tuple,
    resolve_action,
)
from run_static_summary_ppl_speed import resolve_dtype  # noqa: E402

try:
    from memory_policy_router_runtime import load_router  # noqa: E402
except Exception:
    load_router = None


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    longbench_data_dir: str
    ruler_data_dir: str
    longbench_tasks: tuple[str, ...]
    ruler_tasks: tuple[str, ...]
    ruler_context_lengths: tuple[int, ...]
    methods: tuple[str, ...]
    max_examples_per_task: int
    block_tokens: int
    recent_tokens: int
    max_input_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    decode_steps: int
    warmup_steps: int
    dtype: str
    attn_implementation: str
    device_map: str
    router_path: str
    case_id_filter: str
    max_cases: int
    seed: int


@dataclass
class TimingRow:
    benchmark: str
    task: str
    case_id: str
    method: str
    routed_action: str
    prompt_tokens: int
    token_ratio_vs_full_raw: float
    cache_build_seconds: float
    decode_steps: int
    decode_seconds: float
    total_seconds: float
    decode_tokens_per_second: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Warm-cache 1k decode timing for summary memory policies.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--longbench_data_dir", default="ymluo/external/KVCache-Factory/data/LongBench")
    parser.add_argument("--ruler_data_dir", default="ymluo/external/KVCache-Factory/data/RULER")
    parser.add_argument("--longbench_tasks", default="passage_count,passage_retrieval_en")
    parser.add_argument("--ruler_tasks", default="niah_single_1,niah_multiquery,cwe,vt")
    parser.add_argument("--ruler_context_lengths", default="8192,16384")
    parser.add_argument("--methods", default="full_raw,summary1_8,summary1_4,retrieval_raw_k2,router")
    parser.add_argument("--max_examples_per_task", type=int, default=1)
    parser.add_argument("--block_tokens", type=int, default=1024)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=12000)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--decode_steps", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument("--router_path", default="")
    parser.add_argument("--case_id_filter", default="")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026070406)
    args = parser.parse_args()
    return Config(
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        longbench_data_dir=args.longbench_data_dir,
        ruler_data_dir=args.ruler_data_dir,
        longbench_tasks=parse_csv_tuple(args.longbench_tasks),
        ruler_tasks=parse_csv_tuple(args.ruler_tasks),
        ruler_context_lengths=parse_int_tuple(args.ruler_context_lengths),
        methods=parse_csv_tuple(args.methods),
        max_examples_per_task=args.max_examples_per_task,
        block_tokens=args.block_tokens,
        recent_tokens=args.recent_tokens,
        max_input_tokens=args.max_input_tokens,
        summary10_words=args.summary10_words,
        summary100_words=args.summary100_words,
        summary1000_words=args.summary1000_words,
        decode_steps=args.decode_steps,
        warmup_steps=args.warmup_steps,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        router_path=args.router_path,
        case_id_filter=args.case_id_filter,
        max_cases=args.max_cases,
        seed=args.seed,
    )


def bench_config(config: Config) -> BenchConfig:
    return BenchConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        longbench_data_dir=config.longbench_data_dir,
        ruler_data_dir=config.ruler_data_dir,
        longbench_tasks=config.longbench_tasks,
        ruler_tasks=config.ruler_tasks,
        ruler_context_lengths=config.ruler_context_lengths,
        methods=config.methods,
        max_examples_per_task=config.max_examples_per_task,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        max_input_tokens=config.max_input_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_new_tokens_exact=config.decode_steps,
        max_new_tokens_summary=config.decode_steps,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
        cuda_visible_devices="",
        router_path=config.router_path,
        seed=config.seed,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prefill_cache(model: Any, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor, float]:
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    synchronize()
    return out.past_key_values, out.logits[:, -1, :], time.perf_counter() - start


def decode_from_cache(model: Any, logits: torch.Tensor, past_key_values: Any, steps: int) -> float:
    if steps <= 0:
        return 0.0
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(max(0, steps - 1)):
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    synchronize()
    return time.perf_counter() - start


def measure_prompt(model: Any, tokenizer: Any, prompt: str, decode_steps: int, warmup_steps: int) -> tuple[int, float, float]:
    device = next(param.device for param in model.parameters() if param.device.type != "meta")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)

    if warmup_steps > 0:
        past, logits, _ = prefill_cache(model, input_ids[:, -min(input_ids.shape[1], 64) :])
        _ = decode_from_cache(model, logits, past, warmup_steps)

    past, logits, cache_seconds = prefill_cache(model, input_ids)
    decode_seconds = decode_from_cache(model, logits, past, decode_steps)
    return int(input_ids.shape[1]), cache_seconds, decode_seconds


def summarize(rows: list[TimingRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[TimingRow]] = {}
    for row in rows:
        groups.setdefault((row.benchmark, row.task, row.method), []).append(row)
        groups.setdefault(("__overall__", "__overall__", row.method), []).append(row)

    full_by_group: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for (bench, task, method), items in groups.items():
        if method == "full_raw":
            full_by_group[(bench, task)] = (
                statistics.mean(row.prompt_tokens for row in items),
                statistics.mean(row.cache_build_seconds for row in items),
                statistics.mean(row.decode_seconds for row in items),
                statistics.mean(row.total_seconds for row in items),
            )

    out = []
    for (bench, task, method), items in sorted(groups.items()):
        key = (bench, task)
        avg_prompt = statistics.mean(row.prompt_tokens for row in items)
        avg_cache = statistics.mean(row.cache_build_seconds for row in items)
        avg_decode = statistics.mean(row.decode_seconds for row in items)
        avg_total = statistics.mean(row.total_seconds for row in items)
        avg_tps = statistics.mean(row.decode_tokens_per_second for row in items)
        full_prompt, full_cache, full_decode, full_total = full_by_group.get(
            key, (avg_prompt, avg_cache, avg_decode, avg_total)
        )
        out.append(
            {
                "benchmark": bench,
                "task": task,
                "method": method,
                "samples": len(items),
                "avg_prompt_tokens": avg_prompt,
                "token_ratio_vs_full_raw": avg_prompt / full_prompt if full_prompt else 0.0,
                "avg_cache_build_seconds": avg_cache,
                "cache_build_speedup_vs_full_raw": full_cache / avg_cache if avg_cache else 0.0,
                "avg_decode_seconds": avg_decode,
                "decode_speedup_vs_full_raw": full_decode / avg_decode if avg_decode else 0.0,
                "avg_total_seconds": avg_total,
                "total_speedup_vs_full_raw": full_total / avg_total if avg_total else 0.0,
                "avg_decode_tokens_per_second": avg_tps,
            }
        )
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bconfig = bench_config(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(config.dtype, torch),
    }
    if config.device_map not in {"", "none", "cuda"}:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if config.device_map in {"", "none", "cuda"} and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    router = None
    if load_router is not None and config.router_path and Path(config.router_path).exists():
        router = load_router(config.router_path, conservative_generation_upgrade="summary100")

    cases = load_longbench_cases(bconfig) + load_ruler_cases(bconfig)
    if config.case_id_filter:
        cases = [case for case in cases if case.case_id == config.case_id_filter]
    if config.max_cases > 0:
        cases = cases[: config.max_cases]
    if not cases:
        raise ValueError("no cases selected")
    rows: list[TimingRow] = []
    for case_idx, case in enumerate(cases):
        full_prompt = build_prompt(tokenizer, case, build_memory_for_action("full_raw", tokenizer, case, bconfig), bconfig)
        full_prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
        for method in config.methods:
            action = resolve_action(method, tokenizer, case, bconfig, router)
            memory = build_memory_for_action(action, tokenizer, case, bconfig)
            prompt = build_prompt(tokenizer, case, memory, bconfig)
            prompt_tokens, cache_seconds, decode_seconds = measure_prompt(
                model, tokenizer, prompt, config.decode_steps, config.warmup_steps
            )
            row = TimingRow(
                benchmark=case.benchmark,
                task=case.task,
                case_id=case.case_id,
                method=method,
                routed_action=action,
                prompt_tokens=prompt_tokens,
                token_ratio_vs_full_raw=prompt_tokens / full_prompt_tokens if full_prompt_tokens else 0.0,
                cache_build_seconds=cache_seconds,
                decode_steps=config.decode_steps,
                decode_seconds=decode_seconds,
                total_seconds=cache_seconds + decode_seconds,
                decode_tokens_per_second=config.decode_steps / decode_seconds if decode_seconds else 0.0,
            )
            rows.append(row)
            write_csv(output_dir / "timing.partial.csv", [asdict(item) for item in rows])
        print(f"finished case {case_idx + 1}/{len(cases)} {case.benchmark}/{case.task}/{case.case_id}", flush=True)

    summary = summarize(rows)
    write_csv(output_dir / "timing.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "num_cases": len(cases), "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("benchmark,task,method,samples,tokens_vs_full,cache_speedup,decode_speedup,total_speedup")
    for row in summary:
        print(
            f"{row['benchmark']},{row['task']},{row['method']},{row['samples']},"
            f"{row['token_ratio_vs_full_raw']:.4f},"
            f"{row['cache_build_speedup_vs_full_raw']:.3f},"
            f"{row['decode_speedup_vs_full_raw']:.3f},"
            f"{row['total_speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
