from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
    Block,
    Config as PplContextConfig,
    LearnedSummaryScorer,
    StaticSummary,
    build_blocks,
    build_static_summaries,
    fit_word_budget,
    resolve_dtype,
    score_target,
    synchronize,
    train_learned_summary_scorer,
    write_csv,
)


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    teacher_model_name_or_path: str
    adapter_path: str
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
    summary_backend_for_learned: str
    learned_summary_train_tokens: int
    learned_summary_epochs: int
    learned_summary_hidden_dim: int
    learned_summary_lr: float
    learned_summary_max_sentences: int
    learned_summary_seed: int


@dataclass
class EvalRow:
    phase: str
    dataset: str
    sample_id: int
    start_token: int
    method: str
    prompt_tokens: int
    eval_tokens: int
    total_input_tokens: int
    nll: float
    ppl: float
    forward_seconds: float
    tokens_per_second: float


class TeacherSummaryCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, dict[str, str]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.items[row["key"]] = row["summary"]

    def get(self, key: str) -> dict[str, str] | None:
        return self.items.get(key)

    def add(self, key: str, summary: dict[str, str]) -> None:
        self.items[key] = summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "summary": summary}, ensure_ascii=False) + "\n")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Compare heuristic/learned/teacher generated static summaries.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/teacher_summary_compare")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--teacher_model_name_or_path", default="/home/fdong/models/Qwen3-4B-Instruct")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--methods", default="full_raw,heuristic_static_hier,learned_static_hier,teacher_static_hier")
    parser.add_argument("--samples_per_dataset", type=int, default=2)
    parser.add_argument("--sample_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=120_000)
    parser.add_argument("--eval_start_tokens", type=int, default=40_000)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=520)
    parser.add_argument("--teacher_temperature", type=float, default=0.0)
    parser.add_argument("--teacher_cache_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--teacher_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--summary_backend_for_learned", choices=["learned"], default="learned")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070311)
    args = parser.parse_args()

    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(**{**vars(args), "text_paths": text_paths, "dataset_names": dataset_names, "methods": methods})


def ppl_config(config: Config, backend: str) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=("static_hier",),
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
        summary_backend=backend,
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


def compose_static_hier(summaries: list[StaticSummary]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(summaries):
        distance_from_recent = len(summaries) - idx
        if distance_from_recent == 1:
            parts.append(item.summary1000)
        elif distance_from_recent <= 3:
            parts.append(item.summary100)
        else:
            parts.append(item.summary10)
    return "\n".join(part for part in parts if part.strip())


def parse_teacher_output(text: str) -> dict[str, str]:
    def grab(label: str, next_labels: list[str]) -> str:
        pattern = rf"{label}\s*:\s*(.*)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        value = match.group(1)
        for next_label in next_labels:
            split = re.split(rf"\n\s*{next_label}\s*:", value, maxsplit=1, flags=re.IGNORECASE)
            if len(split) > 1:
                value = split[0]
        return value.strip()

    s10 = grab("S10", ["S100", "S1000"])
    s100 = grab("S100", ["S1000"])
    s1000 = grab("S1000", [])
    if not s1000:
        s1000 = text.strip()
    if not s100:
        s100 = s1000
    if not s10:
        s10 = s100
    return {"summary10": s10, "summary100": s100, "summary1000": s1000}


def teacher_key(config: Config, block: Block) -> str:
    payload = {
        "teacher": config.teacher_model_name_or_path,
        "prompt_version": "s10_s100_s1000_dense_memory_v1_250_words",
        "teacher_max_new_tokens": config.teacher_max_new_tokens,
        "block_tokens": config.block_tokens,
        "text": block.text,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def generate_teacher_summary(
    config: Config,
    teacher_model: Any,
    teacher_tokenizer: Any,
    block: Block,
    cache: TeacherSummaryCache,
    device: torch.device,
) -> StaticSummary:
    key = teacher_key(config, block)
    cached = cache.get(key)
    if cached is None:
        prompt = (
            "You are compressing a long-context memory block for future next-token prediction.\n"
            "Preserve concrete entities, events, chronology, causal links, technical terms, and numbers.\n"
            "Do not answer a question. Do not add facts. Write dense neutral English.\n"
            "Return exactly three fields:\n"
            "S10: <=10 words keywords\n"
            "S100: <=100 words dense summary\n"
            "S1000: <=250 words detailed memory\n\n"
            f"TEXT:\n{block.text}\n\n"
            "SUMMARY:\n"
        )
        inputs = teacher_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.teacher_max_new_tokens,
            "do_sample": config.teacher_temperature > 0,
            "temperature": config.teacher_temperature if config.teacher_temperature > 0 else None,
            "pad_token_id": teacher_tokenizer.eos_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.inference_mode():
            output = teacher_model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = teacher_tokenizer.decode(generated, skip_special_tokens=True)
        cached = parse_teacher_output(text)
        cache.add(key, cached)
        del inputs, output, generated
    return StaticSummary(
        block_id=block.block_id,
        summary10=fit_word_budget([cached.get("summary10", "")], config.summary10_words),
        summary100=fit_word_budget([cached.get("summary100", "")], config.summary100_words),
        summary1000=fit_word_budget([cached.get("summary1000", "")], config.summary1000_words),
    )


def context_for_compare_method(
    config: Config,
    tokenizer: Any,
    prefix_ids: list[int],
    method: str,
    learned_scorer: LearnedSummaryScorer | None,
    teacher_model: Any,
    teacher_tokenizer: Any,
    teacher_cache: TeacherSummaryCache,
    device: torch.device,
) -> str:
    if method == "full_raw":
        return tokenizer.decode(prefix_ids, skip_special_tokens=True)
    if method == "recent_only":
        return tokenizer.decode(prefix_ids[-config.recent_tokens :], skip_special_tokens=True)

    recent_ids = prefix_ids[-config.recent_tokens :] if config.recent_tokens > 0 else []
    older_ids = prefix_ids[: max(0, len(prefix_ids) - len(recent_ids))]
    recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)
    blocks = build_blocks(tokenizer, older_ids, config.block_tokens)

    if method == "heuristic_static_hier":
        summaries = build_static_summaries(ppl_config(config, "heuristic"), blocks)
    elif method == "learned_static_hier":
        summaries = build_static_summaries(ppl_config(config, "learned"), blocks, summary_scorer=learned_scorer)
    elif method == "teacher_static_hier":
        summaries = [
            generate_teacher_summary(config, teacher_model, teacher_tokenizer, block, teacher_cache, device)
            for block in blocks
        ]
    else:
        raise ValueError(method)

    return f"Static memory summaries:\n{compose_static_hier(summaries)}\n\nRecent raw text:\n{recent_text}"


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
    phase: str,
    device: torch.device,
    learned_scorer: LearnedSummaryScorer | None,
    teacher_model: Any,
    teacher_tokenizer: Any,
    teacher_cache: TeacherSummaryCache,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    model.eval()
    for dataset in config.dataset_names:
        ids = token_ids_by_dataset[dataset]
        for sample_id, start in enumerate(sample_starts(ids, config)):
            prefix = ids[start : start + config.prefill_tokens]
            target = ids[start + config.prefill_tokens : start + config.prefill_tokens + config.eval_tokens]
            if len(prefix) < config.prefill_tokens or len(target) < config.eval_tokens:
                continue
            for method in config.methods:
                prompt = context_for_compare_method(
                    config,
                    tokenizer,
                    prefix,
                    method,
                    learned_scorer,
                    teacher_model,
                    teacher_tokenizer,
                    teacher_cache,
                    device,
                )
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
                input_tensor = torch.tensor(prompt_ids + target, dtype=torch.long, device=device).view(1, -1)
                synchronize(torch, device)
                started = time.perf_counter()
                with torch.inference_mode():
                    nll, ppl = score_target(model, input_tensor, len(prompt_ids), len(target))
                synchronize(torch, device)
                elapsed = time.perf_counter() - started
                total_tokens = int(input_tensor.shape[1])
                rows.append(
                    EvalRow(
                        phase=phase,
                        dataset=dataset,
                        sample_id=sample_id,
                        start_token=start,
                        method=method,
                        prompt_tokens=len(prompt_ids),
                        eval_tokens=len(target),
                        total_input_tokens=total_tokens,
                        nll=nll,
                        ppl=ppl,
                        forward_seconds=elapsed,
                        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
                    )
                )
                del input_tensor
    return rows


def summarize(rows: list[EvalRow], by_dataset: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[EvalRow]] = {}
    for row in rows:
        key = (row.phase, row.dataset, row.method) if by_dataset else (row.phase, row.method)
        grouped.setdefault(key, []).append(row)

    full_ref: dict[tuple[str, ...], dict[str, float]] = {}
    for key, items in grouped.items():
        if key[-1] == "full_raw":
            full_ref[key[:-1]] = {
                "tokens": statistics.mean(row.total_input_tokens for row in items),
                "seconds": statistics.mean(row.forward_seconds for row in items),
            }

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        total_eval = sum(row.eval_tokens for row in items)
        mean_nll = sum(row.nll * row.eval_tokens for row in items) / max(1, total_eval)
        avg_tokens = statistics.mean(row.total_input_tokens for row in items)
        avg_seconds = statistics.mean(row.forward_seconds for row in items)
        full = full_ref.get(key[:-1], {"tokens": avg_tokens, "seconds": avg_seconds})
        row_out: dict[str, Any] = {
            "phase": key[0],
            "method": key[-1],
            "samples": len(items),
            "eval_tokens": total_eval,
            "mean_nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 80.0)),
            "avg_prompt_tokens": statistics.mean(row.prompt_tokens for row in items),
            "avg_total_input_tokens": avg_tokens,
            "token_ratio_vs_full_raw": avg_tokens / full["tokens"] if full["tokens"] else 0.0,
            "avg_forward_seconds": avg_seconds,
            "speedup_vs_full_raw": full["seconds"] / avg_seconds if avg_seconds > 0 else 0.0,
        }
        if by_dataset:
            row_out["dataset"] = key[1]
        out.append(row_out)
    return out


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    teacher_load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.teacher_dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
        teacher_load_kwargs["attn_implementation"] = config.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset = load_token_ids(tokenizer, config)
    learned_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, ppl_config(config, "learned"))

    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name_or_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name_or_path, **teacher_load_kwargs)
    if not hasattr(teacher_model, "hf_device_map"):
        teacher_model = teacher_model.to(requested_device)
    teacher_device = next(teacher_model.parameters()).device
    teacher_model.eval()

    cache_path = Path(config.teacher_cache_path) if config.teacher_cache_path else output_dir / "teacher_summary_cache.jsonl"
    teacher_cache = TeacherSummaryCache(cache_path)

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    device = next(model.parameters()).device

    all_rows = evaluate(
        model,
        tokenizer,
        token_ids_by_dataset,
        config,
        "base",
        device,
        learned_scorer,
        teacher_model,
        teacher_tokenizer,
        teacher_cache,
    )
    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)
        all_rows.extend(
            evaluate(
                model,
                tokenizer,
                token_ids_by_dataset,
                config,
                "adapted",
                device,
                learned_scorer,
                teacher_model,
                teacher_tokenizer,
                teacher_cache,
            )
        )

    summary_overall = summarize(all_rows, by_dataset=False)
    summary_by_dataset = summarize(all_rows, by_dataset=True)
    write_csv(output_dir / "eval_rows.csv", [asdict(row) for row in all_rows])
    write_csv(output_dir / "summary_overall.csv", summary_overall)
    write_csv(output_dir / "summary_by_dataset.csv", summary_by_dataset)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "token_counts": {name: len(ids) for name, ids in token_ids_by_dataset.items()},
                "learned_scorer": learned_scorer.metadata if learned_scorer else None,
                "summary_overall": summary_overall,
                "summary_by_dataset": summary_by_dataset,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("phase,method,samples,ppl,avg_total_input_tokens,speedup_vs_full_raw")
    for row in summary_overall:
        print(
            f"{row['phase']},{row['method']},{row['samples']},{row['ppl']:.4f},"
            f"{row['avg_total_input_tokens']:.1f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
