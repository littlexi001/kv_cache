from __future__ import annotations

import argparse
import csv
import json
import math
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

from run_static_summary_ppl_speed import (  # noqa: E402
    Config as PplContextConfig,
    LearnedSummaryScorer,
    context_for_method,
    resolve_dtype,
    score_target,
    synchronize,
    train_learned_summary_scorer,
    write_csv,
)
from run_teacher_summary_compare_eval import (  # noqa: E402
    Config as CompareConfig,
    TeacherSummaryCache,
    context_for_compare_method,
)


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    adapter_path: str
    teacher_model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    methods: tuple[str, ...]
    samples_per_dataset: int
    sample_stride_tokens: int
    prefill_tokens: int
    eval_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    eval_start_tokens: int
    teacher_max_new_tokens: int
    teacher_temperature: float
    teacher_cache_path: str
    device: str
    dtype: str
    teacher_dtype: str
    attn_implementation: str
    learned_summary_train_tokens: int
    learned_summary_epochs: int
    learned_summary_hidden_dim: int
    learned_summary_lr: float
    learned_summary_max_sentences: int
    learned_summary_seed: int


@dataclass
class TimingRow:
    dataset: str
    sample_id: int
    start_token: int
    method: str
    prompt_tokens: int
    eval_tokens: int
    total_input_tokens: int
    token_ratio_vs_full_raw: float
    compression_seconds: float
    prompt_tokenize_seconds: float
    forward_seconds: float
    total_online_seconds: float
    total_cached_seconds: float
    nll: float
    ppl: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Measure compression time and inference time separately.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/compression_inference_timing")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--teacher_model_name_or_path", default="/home/fdong/models/Qwen3-4B-Instruct")
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--methods", default="full_raw,learned_static_hier")
    parser.add_argument("--samples_per_dataset", type=int, default=2)
    parser.add_argument("--sample_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=100_000)
    parser.add_argument("--eval_start_tokens", type=int, default=40_000)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=520)
    parser.add_argument("--teacher_temperature", type=float, default=0.0)
    parser.add_argument("--teacher_cache_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--teacher_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070313)
    args = parser.parse_args()

    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(**{**vars(args), "text_paths": text_paths, "dataset_names": dataset_names, "methods": methods})


def ppl_config(config: Config, methods: tuple[str, ...]) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=methods,
        samples_per_dataset=config.samples_per_dataset,
        sample_stride_tokens=config.sample_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.eval_tokens,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_text_tokens=config.max_text_tokens,
        device=config.device,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        summary_backend="learned",
        learned_summary_train_tokens=config.learned_summary_train_tokens,
        learned_summary_epochs=config.learned_summary_epochs,
        learned_summary_hidden_dim=config.learned_summary_hidden_dim,
        learned_summary_lr=config.learned_summary_lr,
        learned_summary_max_sentences=config.learned_summary_max_sentences,
        learned_summary_seed=config.learned_summary_seed,
    )


def compare_config(config: Config, methods: tuple[str, ...]) -> CompareConfig:
    return CompareConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        teacher_model_name_or_path=config.teacher_model_name_or_path,
        adapter_path=config.adapter_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=methods,
        samples_per_dataset=config.samples_per_dataset,
        sample_stride_tokens=config.sample_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.eval_tokens,
        block_tokens=config.block_tokens,
        recent_tokens=config.recent_tokens,
        summary10_words=config.summary10_words,
        summary100_words=config.summary100_words,
        summary1000_words=config.summary1000_words,
        max_text_tokens=config.max_text_tokens,
        eval_start_tokens=config.eval_start_tokens,
        teacher_max_new_tokens=config.teacher_max_new_tokens,
        teacher_temperature=config.teacher_temperature,
        teacher_cache_path=config.teacher_cache_path,
        device=config.device,
        dtype=config.dtype,
        teacher_dtype=config.teacher_dtype,
        attn_implementation=config.attn_implementation,
        summary_backend_for_learned="learned",
        learned_summary_train_tokens=config.learned_summary_train_tokens,
        learned_summary_epochs=config.learned_summary_epochs,
        learned_summary_hidden_dim=config.learned_summary_hidden_dim,
        learned_summary_lr=config.learned_summary_lr,
        learned_summary_max_sentences=config.learned_summary_max_sentences,
        learned_summary_seed=config.learned_summary_seed,
    )


def load_token_ids(tokenizer: Any, config: Config) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, path in zip(config.dataset_names, config.text_paths):
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        out[name] = tokenizer(text, add_special_tokens=False)["input_ids"][: config.max_text_tokens]
    return out


def sample_starts(ids: list[int], config: Config) -> list[int]:
    needed = config.prefill_tokens + config.eval_tokens
    max_start = len(ids) - needed
    if max_start < 0:
        return []
    first = min(config.eval_start_tokens, max_start)
    starts = [min(max_start, first + idx * config.sample_stride_tokens) for idx in range(config.samples_per_dataset)]
    return list(dict.fromkeys(starts))


def summarize(rows: list[TimingRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TimingRow]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row)
    full_rows = grouped.get("full_raw", [])
    full_forward = statistics.mean(row.forward_seconds for row in full_rows) if full_rows else 0.0
    full_online = statistics.mean(row.total_online_seconds for row in full_rows) if full_rows else full_forward

    out: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        total_eval = sum(row.eval_tokens for row in items)
        mean_nll = sum(row.nll * row.eval_tokens for row in items) / max(1, total_eval)
        compression = statistics.mean(row.compression_seconds for row in items)
        tokenization = statistics.mean(row.prompt_tokenize_seconds for row in items)
        forward = statistics.mean(row.forward_seconds for row in items)
        online = statistics.mean(row.total_online_seconds for row in items)
        cached = statistics.mean(row.total_cached_seconds for row in items)
        out.append(
            {
                "method": method,
                "samples": len(items),
                "eval_tokens": total_eval,
                "ppl": math.exp(min(mean_nll, 80.0)),
                "avg_prompt_tokens": statistics.mean(row.prompt_tokens for row in items),
                "avg_total_input_tokens": statistics.mean(row.total_input_tokens for row in items),
                "avg_token_ratio_vs_full_raw": statistics.mean(row.token_ratio_vs_full_raw for row in items),
                "avg_compression_seconds": compression,
                "avg_prompt_tokenize_seconds": tokenization,
                "avg_forward_seconds": forward,
                "avg_total_online_seconds": online,
                "avg_total_cached_seconds": cached,
                "forward_speedup_vs_full_forward": full_forward / forward if forward > 0 else 0.0,
                "online_speedup_vs_full_forward": full_forward / online if online > 0 else 0.0,
                "cached_speedup_vs_full_forward": full_forward / cached if cached > 0 else 0.0,
                "online_speedup_vs_full_online": full_online / online if online > 0 else 0.0,
            }
        )
    return out


def time_one_method(
    config: Config,
    method: str,
    model: torch.nn.Module,
    tokenizer: Any,
    prefix_ids: list[int],
    target_ids: list[int],
    device: torch.device,
    summary_scorer: LearnedSummaryScorer | None,
    teacher_model: Any,
    teacher_tokenizer: Any,
    teacher_cache: TeacherSummaryCache | None,
) -> tuple[int, int, float, float, float, float, float, float]:
    if method == "full_raw":
        prompt_ids = prefix_ids
        compression_seconds = 0.0
        prompt_tokenize_seconds = 0.0
    elif method == "learned_static_hier":
        started = time.perf_counter()
        prompt_text = context_for_method(
            ppl_config(config, ("static_hier",)),
            tokenizer,
            prefix_ids,
            "static_hier",
            summary_scorer=summary_scorer,
        )
        compression_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
        prompt_tokenize_seconds = time.perf_counter() - started
    elif method == "teacher_static_hier":
        if teacher_cache is None:
            raise ValueError("teacher_static_hier requires a teacher cache")
        started = time.perf_counter()
        prompt_text = context_for_compare_method(
            compare_config(config, ("teacher_static_hier",)),
            tokenizer,
            prefix_ids,
            "teacher_static_hier",
            summary_scorer,
            teacher_model,
            teacher_tokenizer,
            teacher_cache,
            next(teacher_model.parameters()).device if teacher_model is not None else device,
        )
        compression_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
        prompt_tokenize_seconds = time.perf_counter() - started
    else:
        raise ValueError(method)

    input_tensor = torch.tensor(prompt_ids + target_ids, dtype=torch.long, device=device).view(1, -1)
    synchronize(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        nll, ppl = score_target(model, input_tensor, len(prompt_ids), len(target_ids))
    synchronize(torch, device)
    forward_seconds = time.perf_counter() - started
    total_online_seconds = compression_seconds + prompt_tokenize_seconds + forward_seconds
    total_cached_seconds = prompt_tokenize_seconds + forward_seconds
    total_tokens = int(input_tensor.shape[1])
    del input_tensor
    return (
        len(prompt_ids),
        total_tokens,
        compression_seconds,
        prompt_tokenize_seconds,
        forward_seconds,
        total_online_seconds,
        total_cached_seconds,
        nll,
    )


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset = load_token_ids(tokenizer, config)

    summary_scorer = None
    summary_scorer_train_seconds = 0.0
    if "learned_static_hier" in config.methods:
        started = time.perf_counter()
        summary_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, ppl_config(config, ("static_hier",)))
        summary_scorer_train_seconds = time.perf_counter() - started

    teacher_model = None
    teacher_tokenizer = None
    teacher_cache = None
    if "teacher_static_hier" in config.methods:
        cache_path = Path(config.teacher_cache_path) if config.teacher_cache_path else output_dir / "teacher_summary_cache.jsonl"
        teacher_cache = TeacherSummaryCache(cache_path)
        teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name_or_path, trust_remote_code=True)
        teacher_load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": resolve_dtype(config.teacher_dtype, torch),
        }
        if config.attn_implementation:
            teacher_load_kwargs["attn_implementation"] = config.attn_implementation
        teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name_or_path, **teacher_load_kwargs)
        if not hasattr(teacher_model, "hf_device_map"):
            teacher_model = teacher_model.to(requested_device)
        teacher_model.eval()

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)
    model.eval()
    device = next(model.parameters()).device

    rows: list[TimingRow] = []
    for dataset in config.dataset_names:
        ids = token_ids_by_dataset[dataset]
        for sample_id, start in enumerate(sample_starts(ids, config)):
            prefix_ids = ids[start : start + config.prefill_tokens]
            target_ids = ids[start + config.prefill_tokens : start + config.prefill_tokens + config.eval_tokens]
            if len(prefix_ids) < config.prefill_tokens or len(target_ids) < config.eval_tokens:
                continue
            full_prompt_tokens = len(prefix_ids)
            for method in config.methods:
                (
                    prompt_tokens,
                    total_tokens,
                    compression_seconds,
                    prompt_tokenize_seconds,
                    forward_seconds,
                    total_online_seconds,
                    total_cached_seconds,
                    nll,
                ) = time_one_method(
                    config,
                    method,
                    model,
                    tokenizer,
                    prefix_ids,
                    target_ids,
                    device,
                    summary_scorer,
                    teacher_model,
                    teacher_tokenizer,
                    teacher_cache,
                )
                rows.append(
                    TimingRow(
                        dataset=dataset,
                        sample_id=sample_id,
                        start_token=start,
                        method=method,
                        prompt_tokens=prompt_tokens,
                        eval_tokens=len(target_ids),
                        total_input_tokens=total_tokens,
                        token_ratio_vs_full_raw=prompt_tokens / max(1, full_prompt_tokens),
                        compression_seconds=compression_seconds,
                        prompt_tokenize_seconds=prompt_tokenize_seconds,
                        forward_seconds=forward_seconds,
                        total_online_seconds=total_online_seconds,
                        total_cached_seconds=total_cached_seconds,
                        nll=nll,
                        ppl=math.exp(min(nll, 80.0)),
                    )
                )

    summary = summarize(rows)
    write_csv(output_dir / "timing_rows.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "summary_scorer_train_seconds": summary_scorer_train_seconds,
                "summary_scorer": summary_scorer.metadata if summary_scorer else None,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "method,samples,ppl,avg_total_input_tokens,avg_compression_seconds,"
        "avg_prompt_tokenize_seconds,avg_forward_seconds,avg_total_online_seconds,"
        "forward_speedup_vs_full_forward,online_speedup_vs_full_forward,cached_speedup_vs_full_forward"
    )
    for row in summary:
        print(
            f"{row['method']},{row['samples']},{row['ppl']:.4f},"
            f"{row['avg_total_input_tokens']:.1f},{row['avg_compression_seconds']:.6f},"
            f"{row['avg_prompt_tokenize_seconds']:.6f},{row['avg_forward_seconds']:.6f},"
            f"{row['avg_total_online_seconds']:.6f},{row['forward_speedup_vs_full_forward']:.3f},"
            f"{row['online_speedup_vs_full_forward']:.3f},{row['cached_speedup_vs_full_forward']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
