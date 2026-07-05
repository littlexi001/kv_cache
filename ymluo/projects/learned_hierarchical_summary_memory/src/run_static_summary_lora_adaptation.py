from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    eval_methods: tuple[str, ...]
    train_method: str
    train_methods: tuple[str, ...]
    train_steps: int
    learning_rate: float
    grad_accum_steps: int
    prefill_tokens: int
    target_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    train_start_tokens: int
    train_span_tokens: int
    eval_start_tokens: int
    eval_samples_per_dataset: int
    eval_stride_tokens: int
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    device: str
    dtype: str
    attn_implementation: str
    merge_adapter_for_eval: bool
    summary_backend: str = "heuristic"
    learned_summary_train_tokens: int = 60_000
    learned_summary_epochs: int = 8
    learned_summary_hidden_dim: int = 32
    learned_summary_lr: float = 3e-3
    learned_summary_max_sentences: int = 20_000
    learned_summary_seed: int = 2026070307


@dataclass
class EvalRow:
    phase: str
    dataset: str
    sample_id: int
    method: str
    prompt_tokens: int
    eval_tokens: int
    total_input_tokens: int
    nll: float
    ppl: float
    forward_seconds: float
    tokens_per_second: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="LoRA adaptation for static summary memory + recent raw LM format.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_adapt")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument(
        "--text_paths",
        default=(
            "ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt,"
            "ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt"
        ),
    )
    parser.add_argument("--dataset_names", default="warpeace,montecristo")
    parser.add_argument("--eval_methods", default="full_raw,recent_only,static_hier,static_sum100,static_sum1000")
    parser.add_argument("--train_method", default="static_hier")
    parser.add_argument("--train_methods", default="")
    parser.add_argument("--train_steps", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--target_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=220_000)
    parser.add_argument("--train_start_tokens", type=int, default=0)
    parser.add_argument("--train_span_tokens", type=int, default=120_000)
    parser.add_argument("--eval_start_tokens", type=int, default=150_000)
    parser.add_argument("--eval_samples_per_dataset", type=int, default=4)
    parser.add_argument("--eval_stride_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026070306)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--merge_adapter_for_eval", action="store_true")
    parser.add_argument("--summary_backend", choices=["heuristic", "learned"], default="heuristic")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070307)
    args = parser.parse_args()
    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    eval_methods = tuple(item.strip() for item in args.eval_methods.split(",") if item.strip())
    train_methods = tuple(item.strip() for item in args.train_methods.split(",") if item.strip()) or (args.train_method,)
    lora_target_modules = tuple(item.strip() for item in args.lora_target_modules.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(
        **{
            **vars(args),
            "text_paths": text_paths,
            "dataset_names": dataset_names,
            "eval_methods": eval_methods,
            "train_methods": train_methods,
            "lora_target_modules": lora_target_modules,
        }
    )


def context_config(config: Config, methods: tuple[str, ...]) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=methods,
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
        summary_backend=config.summary_backend,
        learned_summary_train_tokens=config.learned_summary_train_tokens,
        learned_summary_epochs=config.learned_summary_epochs,
        learned_summary_hidden_dim=config.learned_summary_hidden_dim,
        learned_summary_lr=config.learned_summary_lr,
        learned_summary_max_sentences=config.learned_summary_max_sentences,
        learned_summary_seed=config.learned_summary_seed,
    )


def load_token_ids(tokenizer: Any, config: Config) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, text_path in zip(config.dataset_names, config.text_paths):
        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        out[name] = tokenizer(text, add_special_tokens=False)["input_ids"][: config.max_text_tokens]
    return out


def make_lm_example(
    tokenizer: Any,
    ids: list[int],
    start: int,
    config: Config,
    method: str,
    device: torch.device,
    summary_scorer: LearnedSummaryScorer | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prefix = ids[start : start + config.prefill_tokens]
    target = ids[start + config.prefill_tokens : start + config.prefill_tokens + config.target_tokens]
    if len(prefix) < config.prefill_tokens or len(target) < config.target_tokens:
        raise ValueError("not enough tokens for example")
    prompt = context_for_method(context_config(config, (method,)), tokenizer, prefix, method, summary_scorer=summary_scorer)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id or 0]
    input_ids = torch.tensor(prompt_ids + target, dtype=torch.long, device=device).view(1, -1)
    labels = torch.full_like(input_ids, -100)
    labels[:, len(prompt_ids) :] = torch.tensor(target, dtype=torch.long, device=device).view(1, -1)
    return input_ids, labels, len(prompt_ids)


def evaluate_methods(
    model: torch.nn.Module,
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
    phase: str,
    device: torch.device,
    summary_scorer: LearnedSummaryScorer | None,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    model.eval()
    for dataset in config.dataset_names:
        ids = token_ids_by_dataset[dataset]
        for sample_id in range(config.eval_samples_per_dataset):
            start = config.eval_start_tokens + sample_id * config.eval_stride_tokens
            if start + config.prefill_tokens + config.target_tokens > len(ids):
                continue
            prefix = ids[start : start + config.prefill_tokens]
            target = ids[start + config.prefill_tokens : start + config.prefill_tokens + config.target_tokens]
            for method in config.eval_methods:
                prompt = context_for_method(
                    context_config(config, (method,)),
                    tokenizer,
                    prefix,
                    method,
                    summary_scorer=summary_scorer,
                )
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                if not prompt_ids:
                    prompt_ids = [tokenizer.eos_token_id or 0]
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


def summarize_eval(rows: list[EvalRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EvalRow]] = {}
    for row in rows:
        grouped.setdefault((row.phase, row.method), []).append(row)

    full_ref: dict[str, dict[str, float]] = {}
    for phase in sorted({row.phase for row in rows}):
        items = [row for row in rows if row.phase == phase and row.method == "full_raw"]
        if items:
            full_ref[phase] = {
                "tokens": statistics.mean(row.total_input_tokens for row in items),
                "seconds": statistics.mean(row.forward_seconds for row in items),
            }

    out: list[dict[str, Any]] = []
    for (phase, method), items in sorted(grouped.items()):
        total_eval = sum(row.eval_tokens for row in items)
        mean_nll = sum(row.nll * row.eval_tokens for row in items) / max(1, total_eval)
        avg_tokens = statistics.mean(row.total_input_tokens for row in items)
        avg_seconds = statistics.mean(row.forward_seconds for row in items)
        full = full_ref.get(phase, {"tokens": avg_tokens, "seconds": avg_seconds})
        out.append(
            {
                "phase": phase,
                "method": method,
                "samples": len(items),
                "eval_tokens": total_eval,
                "mean_nll": mean_nll,
                "ppl": math.exp(min(mean_nll, 80.0)),
                "avg_prompt_tokens": statistics.mean(row.prompt_tokens for row in items),
                "avg_total_input_tokens": avg_tokens,
                "token_ratio_vs_full_raw": avg_tokens / full["tokens"] if full["tokens"] else 0.0,
                "avg_forward_seconds": avg_seconds,
                "time_ratio_vs_full_raw": avg_seconds / full["seconds"] if full["seconds"] else 0.0,
                "speedup_vs_full_raw": full["seconds"] / avg_seconds if avg_seconds > 0 else 0.0,
            }
        )
    return out


def train_lora(
    model: torch.nn.Module,
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
    device: torch.device,
    summary_scorer: LearnedSummaryScorer | None,
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
    datasets = list(config.dataset_names)
    needed = config.prefill_tokens + config.target_tokens
    train_history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(1, config.train_steps + 1):
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            dataset = rng.choice(datasets)
            ids = token_ids_by_dataset[dataset]
            low = config.train_start_tokens
            high = min(len(ids) - needed, config.train_start_tokens + config.train_span_tokens - needed)
            if high <= low:
                raise ValueError(f"not enough training tokens for {dataset}")
            start = rng.randint(low, high)
            input_ids, labels, prompt_len = make_lm_example(
                tokenizer,
                ids,
                start,
                config,
                rng.choice(config.train_methods),
                device,
                summary_scorer,
            )
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss / max(1, config.grad_accum_steps)
            loss.backward()
            accum_loss += float(loss.detach().cpu()) * max(1, config.grad_accum_steps)
            del input_ids, labels, outputs, loss
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == config.train_steps:
            row = {
                "step": step,
                "train_loss": accum_loss / max(1, config.grad_accum_steps),
                "train_ppl": math.exp(min(accum_loss / max(1, config.grad_accum_steps), 80.0)),
                "elapsed_seconds": time.perf_counter() - started,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_fraction": trainable / max(1, total),
            }
            train_history.append(row)
            print(
                f"step {step}/{config.train_steps} loss={row['train_loss']:.4f} "
                f"ppl={row['train_ppl']:.2f} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return model, train_history


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset = load_token_ids(tokenizer, config)
    summary_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, context_config(config, config.eval_methods))
    if summary_scorer is not None:
        torch.save(
            {
                "state_dict": summary_scorer.model.state_dict(),
                "mean": summary_scorer.mean,
                "std": summary_scorer.std,
                "metadata": summary_scorer.metadata,
            },
            output_dir / "learned_summary_scorer.pt",
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    device = next(model.parameters()).device
    model.eval()

    base_rows = evaluate_methods(model, tokenizer, token_ids_by_dataset, config, "base", device, summary_scorer)
    model, train_history = train_lora(model, tokenizer, token_ids_by_dataset, config, device, summary_scorer)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    if config.merge_adapter_for_eval and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        model = model.to(device)
    adapted_rows = evaluate_methods(model, tokenizer, token_ids_by_dataset, config, "adapted", device, summary_scorer)
    all_rows = base_rows + adapted_rows
    summary = summarize_eval(all_rows)

    write_csv(output_dir / "eval_rows.csv", [asdict(row) for row in all_rows])
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "train_history.csv", train_history)
    payload = {
        "config": asdict(config),
        "summary_scorer": summary_scorer.metadata if summary_scorer is not None else None,
        "summary": summary,
        "train_history_tail": train_history[-10:],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("phase,method,samples,ppl,avg_total_input_tokens,avg_forward_seconds,speedup_vs_full_raw")
    for row in summary:
        print(
            f"{row['phase']},{row['method']},{row['samples']},{row['ppl']:.4f},"
            f"{row['avg_total_input_tokens']:.1f},{row['avg_forward_seconds']:.4f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
