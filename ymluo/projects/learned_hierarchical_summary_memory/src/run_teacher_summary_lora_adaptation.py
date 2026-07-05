from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_static_summary_lora_adaptation import summarize_eval  # noqa: E402
from run_static_summary_ppl_speed import (  # noqa: E402
    Config as PplContextConfig,
    LearnedSummaryScorer,
    resolve_dtype,
    train_learned_summary_scorer,
    write_csv,
)
from run_teacher_summary_compare_eval import (  # noqa: E402
    Config as CompareConfig,
    TeacherSummaryCache,
    context_for_compare_method,
    evaluate as compare_evaluate,
    load_token_ids,
    sample_starts,
    summarize as summarize_compare,
)


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    teacher_model_name_or_path: str
    reference_adapter_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    eval_methods: tuple[str, ...]
    train_method: str
    train_steps: int
    learning_rate: float
    grad_accum_steps: int
    train_examples_per_dataset: int
    train_start_tokens: int
    train_stride_tokens: int
    prefill_tokens: int
    target_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    eval_start_tokens: int
    eval_samples_per_dataset: int
    eval_stride_tokens: int
    teacher_max_new_tokens: int
    teacher_temperature: float
    teacher_cache_path: str
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
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
class EncodedExample:
    dataset: str
    start_token: int
    input_ids: list[int]
    labels: list[int]
    prompt_tokens: int
    target_tokens: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on teacher-generated static summary memory.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/teacher_summary_lora_adapt")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--teacher_model_name_or_path", default="/home/fdong/models/Qwen3-4B-Instruct")
    parser.add_argument("--reference_adapter_path", default="")
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--eval_methods", default="full_raw,learned_static_hier,teacher_static_hier")
    parser.add_argument("--train_method", default="teacher_static_hier")
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--train_examples_per_dataset", type=int, default=4)
    parser.add_argument("--train_start_tokens", type=int, default=0)
    parser.add_argument("--train_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=90_000)
    parser.add_argument("--eval_start_tokens", type=int, default=40_000)
    parser.add_argument("--eval_samples_per_dataset", type=int, default=2)
    parser.add_argument("--eval_stride_tokens", type=int, default=2048)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=520)
    parser.add_argument("--teacher_temperature", type=float, default=0.0)
    parser.add_argument("--teacher_cache_path", default="")
    parser.add_argument("--seed", type=int, default=2026070312)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--teacher_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070311)
    args = parser.parse_args()

    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    eval_methods = tuple(item.strip() for item in args.eval_methods.split(",") if item.strip())
    lora_target_modules = tuple(item.strip() for item in args.lora_target_modules.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(
        **{
            **vars(args),
            "text_paths": text_paths,
            "dataset_names": dataset_names,
            "eval_methods": eval_methods,
            "lora_target_modules": lora_target_modules,
        }
    )


def compare_config(config: Config, methods: tuple[str, ...] | None = None) -> CompareConfig:
    return CompareConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        teacher_model_name_or_path=config.teacher_model_name_or_path,
        adapter_path="",
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=methods or config.eval_methods,
        samples_per_dataset=config.eval_samples_per_dataset,
        sample_stride_tokens=config.eval_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.target_tokens,
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


def ppl_config(config: Config) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=("static_hier",),
        samples_per_dataset=config.eval_samples_per_dataset,
        sample_stride_tokens=config.eval_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=config.target_tokens,
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


def train_starts(ids: list[int], config: Config) -> list[int]:
    needed = config.prefill_tokens + config.target_tokens
    max_start = min(len(ids) - needed, config.eval_start_tokens - needed)
    if max_start < config.train_start_tokens:
        return []
    starts = [
        config.train_start_tokens + idx * config.train_stride_tokens
        for idx in range(config.train_examples_per_dataset)
    ]
    return [start for start in starts if start <= max_start]


def precompute_teacher_eval_cache(
    config: Config,
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    learned_scorer: LearnedSummaryScorer | None,
    teacher_model: Any,
    teacher_tokenizer: Any,
    teacher_cache: TeacherSummaryCache,
    teacher_device: torch.device,
) -> None:
    cfg = compare_config(config, ("teacher_static_hier",))
    for dataset in config.dataset_names:
        ids = token_ids_by_dataset[dataset]
        for start in sample_starts(ids, cfg):
            prefix = ids[start : start + config.prefill_tokens]
            context_for_compare_method(
                cfg,
                tokenizer,
                prefix,
                "teacher_static_hier",
                learned_scorer,
                teacher_model,
                teacher_tokenizer,
                teacher_cache,
                teacher_device,
            )


def build_train_examples(
    config: Config,
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    learned_scorer: LearnedSummaryScorer | None,
    teacher_model: Any,
    teacher_tokenizer: Any,
    teacher_cache: TeacherSummaryCache,
    teacher_device: torch.device,
) -> list[EncodedExample]:
    cfg = compare_config(config, (config.train_method,))
    examples: list[EncodedExample] = []
    for dataset in config.dataset_names:
        ids = token_ids_by_dataset[dataset]
        for start in train_starts(ids, config):
            prefix = ids[start : start + config.prefill_tokens]
            target = ids[start + config.prefill_tokens : start + config.prefill_tokens + config.target_tokens]
            if len(prefix) < config.prefill_tokens or len(target) < config.target_tokens:
                continue
            prompt = context_for_compare_method(
                cfg,
                tokenizer,
                prefix,
                config.train_method,
                learned_scorer,
                teacher_model,
                teacher_tokenizer,
                teacher_cache,
                teacher_device,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
            input_ids = prompt_ids + target
            labels = [-100] * len(prompt_ids) + target
            examples.append(
                EncodedExample(
                    dataset=dataset,
                    start_token=start,
                    input_ids=input_ids,
                    labels=labels,
                    prompt_tokens=len(prompt_ids),
                    target_tokens=len(target),
                )
            )
    if not examples:
        raise ValueError("no train examples were constructed")
    return examples


def train_lora_on_encoded_examples(
    model: torch.nn.Module,
    examples: list[EncodedExample],
    config: Config,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.train()

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=config.learning_rate)
    rng = random.Random(config.seed)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []

    for step in range(1, config.train_steps + 1):
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            example = rng.choice(examples)
            input_tensor = torch.tensor(example.input_ids, dtype=torch.long, device=device).view(1, -1)
            label_tensor = torch.tensor(example.labels, dtype=torch.long, device=device).view(1, -1)
            outputs = model(input_ids=input_tensor, labels=label_tensor, use_cache=False)
            loss = outputs.loss / max(1, config.grad_accum_steps)
            loss.backward()
            accum_loss += float(loss.detach().cpu()) * max(1, config.grad_accum_steps)
            del input_tensor, label_tensor, outputs, loss
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == config.train_steps:
            mean_loss = accum_loss / max(1, config.grad_accum_steps)
            row = {
                "step": step,
                "train_loss": mean_loss,
                "train_ppl": math.exp(min(mean_loss, 80.0)),
                "elapsed_seconds": time.perf_counter() - started,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_fraction": trainable / max(1, total),
            }
            history.append(row)
            print(
                f"step {step}/{config.train_steps} loss={row['train_loss']:.4f} "
                f"ppl={row['train_ppl']:.2f} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return model, history


def load_base_model(config: Config, torch_module: Any, requested_device: torch.device) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch_module)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    return model


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset = load_token_ids(tokenizer, compare_config(config))
    learned_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, ppl_config(config))

    teacher_cache_path = Path(config.teacher_cache_path) if config.teacher_cache_path else output_dir / "teacher_summary_cache.jsonl"
    teacher_cache = TeacherSummaryCache(teacher_cache_path)
    teacher_load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(config.teacher_dtype, torch),
    }
    if config.attn_implementation:
        teacher_load_kwargs["attn_implementation"] = config.attn_implementation
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name_or_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name_or_path, **teacher_load_kwargs)
    if not hasattr(teacher_model, "hf_device_map"):
        teacher_model = teacher_model.to(requested_device)
    teacher_device = next(teacher_model.parameters()).device
    teacher_model.eval()

    print("precomputing teacher-summary train examples", flush=True)
    train_examples = build_train_examples(
        config,
        tokenizer,
        token_ids_by_dataset,
        learned_scorer,
        teacher_model,
        teacher_tokenizer,
        teacher_cache,
        teacher_device,
    )
    print(f"built {len(train_examples)} train examples", flush=True)
    precompute_teacher_eval_cache(
        config,
        tokenizer,
        token_ids_by_dataset,
        learned_scorer,
        teacher_model,
        teacher_tokenizer,
        teacher_cache,
        teacher_device,
    )
    del teacher_model, teacher_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_rows: list[Any] = []
    model = load_base_model(config, torch, requested_device)
    device = next(model.parameters()).device
    all_rows.extend(
        compare_evaluate(
            model,
            tokenizer,
            token_ids_by_dataset,
            compare_config(config),
            "base",
            device,
            learned_scorer,
            None,
            None,
            teacher_cache,
        )
    )

    if config.reference_adapter_path:
        ref_model = PeftModel.from_pretrained(model, config.reference_adapter_path)
        all_rows.extend(
            compare_evaluate(
                ref_model,
                tokenizer,
                token_ids_by_dataset,
                compare_config(config),
                "reference_learned_adapter",
                device,
                learned_scorer,
                None,
                None,
                teacher_cache,
            )
        )
        del ref_model, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = load_base_model(config, torch, requested_device)
    device = next(model.parameters()).device
    model, train_history = train_lora_on_encoded_examples(model, train_examples, config, device)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    all_rows.extend(
        compare_evaluate(
            model,
            tokenizer,
            token_ids_by_dataset,
            compare_config(config),
            "teacher_summary_adapter",
            device,
            learned_scorer,
            None,
            None,
            teacher_cache,
        )
    )

    summary_overall = summarize_compare(all_rows, by_dataset=False)
    summary_by_dataset = summarize_compare(all_rows, by_dataset=True)
    write_csv(output_dir / "eval_rows.csv", [asdict(row) for row in all_rows])
    write_csv(output_dir / "summary_overall.csv", summary_overall)
    write_csv(output_dir / "summary_by_dataset.csv", summary_by_dataset)
    write_csv(output_dir / "train_examples.csv", [asdict(row) for row in train_examples])
    write_csv(output_dir / "train_history.csv", train_history)
    legacy_summary = summarize_eval(all_rows)
    write_csv(output_dir / "summary_legacy_phase_method.csv", legacy_summary)
    payload = {
        "config": asdict(config),
        "token_counts": {name: len(ids) for name, ids in token_ids_by_dataset.items()},
        "learned_scorer": learned_scorer.metadata if learned_scorer is not None else None,
        "train_examples": {
            "count": len(train_examples),
            "avg_prompt_tokens": sum(row.prompt_tokens for row in train_examples) / max(1, len(train_examples)),
            "avg_total_tokens": sum(len(row.input_ids) for row in train_examples) / max(1, len(train_examples)),
        },
        "summary_overall": summary_overall,
        "summary_by_dataset": summary_by_dataset,
        "train_history_tail": train_history[-10:],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("phase,method,samples,ppl,avg_total_input_tokens,speedup_vs_full_raw")
    for row in summary_overall:
        print(
            f"{row['phase']},{row['method']},{row['samples']},{row['ppl']:.4f},"
            f"{row['avg_total_input_tokens']:.1f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
