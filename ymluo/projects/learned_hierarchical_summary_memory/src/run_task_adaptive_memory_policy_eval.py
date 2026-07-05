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

from run_full_information_stress_eval import (  # noqa: E402
    StressCase,
    build_stress_cases,
    candidate_answers,
    context_with_raw_retrieval,
    score_answer_choice,
)
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
    adapter_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    generation_methods: tuple[str, ...]
    exact_methods: tuple[str, ...]
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
    num_choices: int
    generation_nll_slack: float
    device: str
    dtype: str
    attn_implementation: str
    learned_summary_train_tokens: int
    learned_summary_epochs: int
    learned_summary_hidden_dim: int
    learned_summary_lr: float
    learned_summary_max_sentences: int
    learned_summary_seed: int
    seed: int

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.generation_methods + self.exact_methods)))


@dataclass(frozen=True)
class GenerationCase:
    dataset: str
    sample_id: int
    start_token: int
    prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass
class MethodRow:
    task_family: str
    dataset: str
    sample_id: int
    start_token: int
    method: str
    prompt_tokens: int
    total_input_tokens: int
    token_ratio_vs_full_raw: float
    forward_seconds: float
    nll: float | None
    ppl: float | None
    full_raw_nll: float | None
    nll_delta_vs_full_raw: float | None
    choice_correct: int | None
    answer_retained: int | None
    selected_answer: str
    correct_answer: str
    success: int


@dataclass
class OracleRow:
    task_family: str
    dataset: str
    sample_id: int
    start_token: int
    selected_method: str
    prompt_tokens: int
    token_ratio_vs_full_raw: float
    forward_seconds: float
    success: int
    fallback_used: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate an oracle task-adaptive summary/raw memory policy.")
    parser.add_argument(
        "--output_dir",
        default="ymluo/projects/learned_hierarchical_summary_memory/outputs/task_adaptive_memory_policy",
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument("--generation_methods", default="summary10,summary100,summary1000,static_hier,full_raw")
    parser.add_argument(
        "--exact_methods",
        default="summary10,summary100,summary1000,static_hier,retrieval_raw_k1,retrieval_raw_k2,full_raw",
    )
    parser.add_argument("--samples_per_dataset", type=int, default=4)
    parser.add_argument("--sample_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=160_000)
    parser.add_argument("--eval_start_tokens", type=int, default=40_000)
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--generation_nll_slack", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070401)
    parser.add_argument("--seed", type=int, default=2026070402)
    args = parser.parse_args()

    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    generation_methods = tuple(item.strip() for item in args.generation_methods.split(",") if item.strip())
    exact_methods = tuple(item.strip() for item in args.exact_methods.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    return Config(
        **{
            **vars(args),
            "text_paths": text_paths,
            "dataset_names": dataset_names,
            "generation_methods": generation_methods,
            "exact_methods": exact_methods,
        }
    )


def context_config(config: Config) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=tuple(sorted(set(config.generation_methods + config.exact_methods))),
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
    deduped: list[int] = []
    for start in starts:
        if start not in deduped:
            deduped.append(start)
    return deduped


def build_generation_cases(
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
) -> list[GenerationCase]:
    cases: list[GenerationCase] = []
    for dataset, all_ids in token_ids_by_dataset.items():
        for sample_id, start in enumerate(sample_starts(all_ids, config)):
            prefix_ids = all_ids[start : start + config.prefill_tokens]
            target_ids = all_ids[start + config.prefill_tokens : start + config.prefill_tokens + config.eval_tokens]
            if len(prefix_ids) < config.prefill_tokens or len(target_ids) < config.eval_tokens:
                continue
            cases.append(
                GenerationCase(
                    dataset=dataset,
                    sample_id=sample_id,
                    start_token=start,
                    prefix_ids=tuple(prefix_ids),
                    target_ids=tuple(target_ids),
                )
            )
    return cases


def summary_method_to_static(method: str) -> str:
    mapping = {
        "summary10": "static_sum10",
        "summary100": "static_sum100",
        "summary1000": "static_sum1000",
        "summary1_8": "summary1_8",
        "summary1_4": "summary1_4",
        "summary1_2": "summary1_2",
        "static_hier": "static_hier",
        "full_raw": "full_raw",
        "recent_only": "recent_only",
    }
    if method not in mapping:
        raise ValueError(method)
    return mapping[method]


def make_generation_prompt(
    config: Config,
    tokenizer: Any,
    case: GenerationCase,
    method: str,
    summary_scorer: LearnedSummaryScorer | None,
) -> str:
    return context_for_method(
        context_config(config),
        tokenizer,
        list(case.prefix_ids),
        summary_method_to_static(method),
        summary_scorer=summary_scorer,
    )


def make_exact_prompt(
    config: Config,
    tokenizer: Any,
    case: StressCase,
    method: str,
    summary_scorer: LearnedSummaryScorer | None,
) -> str:
    if method.startswith("retrieval_raw_k"):
        top_k = int(method.removeprefix("retrieval_raw_k"))
        memory = context_with_raw_retrieval(config, tokenizer, case, top_k=top_k, summary_scorer=summary_scorer)
    else:
        memory = context_for_method(
            context_config(config),
            tokenizer,
            list(case.prefix_ids),
            summary_method_to_static(method),
            summary_scorer=summary_scorer,
        )
    return f"{memory}\n\n{case.question}"


def evaluate_generation_case(
    model: torch.nn.Module,
    tokenizer: Any,
    case: GenerationCase,
    config: Config,
    summary_scorer: LearnedSummaryScorer | None,
    device: torch.device,
) -> list[MethodRow]:
    measured: dict[str, tuple[int, int, float, float, float]] = {}
    for method in config.generation_methods:
        prompt = make_generation_prompt(config, tokenizer, case, method, summary_scorer)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id or 0]
        input_ids = torch.tensor(prompt_ids + list(case.target_ids), dtype=torch.long, device=device).view(1, -1)
        synchronize(torch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            nll, ppl = score_target(model, input_ids, len(prompt_ids), len(case.target_ids))
        synchronize(torch, device)
        elapsed = time.perf_counter() - started
        measured[method] = (len(prompt_ids), int(input_ids.shape[1]), nll, ppl, elapsed)
        del input_ids

    if "full_raw" not in measured:
        raise ValueError("generation_methods must include full_raw for oracle labeling")
    full_prompt_tokens, _, full_nll, _, _ = measured["full_raw"]

    rows: list[MethodRow] = []
    for method, (prompt_tokens, total_input_tokens, nll, ppl, elapsed) in measured.items():
        success = int(nll <= full_nll + config.generation_nll_slack)
        rows.append(
            MethodRow(
                task_family="generation",
                dataset=case.dataset,
                sample_id=case.sample_id,
                start_token=case.start_token,
                method=method,
                prompt_tokens=prompt_tokens,
                total_input_tokens=total_input_tokens,
                token_ratio_vs_full_raw=prompt_tokens / full_prompt_tokens if full_prompt_tokens else 0.0,
                forward_seconds=elapsed,
                nll=nll,
                ppl=ppl,
                full_raw_nll=full_nll,
                nll_delta_vs_full_raw=nll - full_nll,
                choice_correct=None,
                answer_retained=None,
                selected_answer="",
                correct_answer="",
                success=success,
            )
        )
    return rows


def evaluate_exact_case(
    model: torch.nn.Module,
    tokenizer: Any,
    case: StressCase,
    case_idx: int,
    all_cases: list[StressCase],
    config: Config,
    summary_scorer: LearnedSummaryScorer | None,
    device: torch.device,
) -> list[MethodRow]:
    choices = candidate_answers(all_cases, case_idx, config)
    full_prompt = make_exact_prompt(config, tokenizer, case, "full_raw", summary_scorer)
    full_prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])

    rows: list[MethodRow] = []
    for method in config.exact_methods:
        prompt = make_exact_prompt(config, tokenizer, case, method, summary_scorer)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        synchronize(torch, device)
        started = time.perf_counter()
        scores: list[tuple[str, float, int]] = []
        for choice in choices:
            nll, total_tokens = score_answer_choice(model, tokenizer, prompt, choice, device)
            scores.append((choice, nll, total_tokens))
        synchronize(torch, device)
        elapsed = time.perf_counter() - started
        scores.sort(key=lambda item: item[1])
        selected_answer = scores[0][0]
        correct = int(selected_answer == case.answer)
        answer_retained = int(case.answer in prompt)
        rows.append(
            MethodRow(
                task_family="exact",
                dataset=case.dataset,
                sample_id=case.sample_id,
                start_token=case.start_token,
                method=method,
                prompt_tokens=prompt_tokens,
                total_input_tokens=max(total_tokens for _, _, total_tokens in scores),
                token_ratio_vs_full_raw=prompt_tokens / full_prompt_tokens if full_prompt_tokens else 0.0,
                forward_seconds=elapsed,
                nll=None,
                ppl=None,
                full_raw_nll=None,
                nll_delta_vs_full_raw=None,
                choice_correct=correct,
                answer_retained=answer_retained,
                selected_answer=selected_answer,
                correct_answer=case.answer,
                success=int(correct and answer_retained),
            )
        )
    return rows


def choose_oracle(rows: list[MethodRow]) -> OracleRow:
    successful = [row for row in rows if row.success]
    if successful:
        selected = min(successful, key=lambda row: (row.prompt_tokens, row.forward_seconds))
    else:
        selected = min(rows, key=lambda row: (row.method != "full_raw", row.prompt_tokens))
    return OracleRow(
        task_family=selected.task_family,
        dataset=selected.dataset,
        sample_id=selected.sample_id,
        start_token=selected.start_token,
        selected_method=selected.method,
        prompt_tokens=selected.prompt_tokens,
        token_ratio_vs_full_raw=selected.token_ratio_vs_full_raw,
        forward_seconds=selected.forward_seconds,
        success=selected.success,
        fallback_used=int(selected.method == "full_raw"),
    )


def summarize_methods(rows: list[MethodRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[MethodRow]] = {}
    for row in rows:
        grouped.setdefault((row.task_family, row.method), []).append(row)
    out: list[dict[str, Any]] = []
    for (task_family, method), items in sorted(grouped.items()):
        nll_items = [item for item in items if item.nll is not None]
        exact_items = [item for item in items if item.choice_correct is not None]
        out.append(
            {
                "task_family": task_family,
                "method": method,
                "samples": len(items),
                "success_rate": statistics.mean(item.success for item in items),
                "choice_accuracy": statistics.mean(item.choice_correct for item in exact_items) if exact_items else "",
                "answer_retained_rate": statistics.mean(item.answer_retained for item in exact_items) if exact_items else "",
                "mean_nll": statistics.mean(item.nll for item in nll_items) if nll_items else "",
                "mean_ppl": math.exp(min(statistics.mean(item.nll for item in nll_items), 80.0)) if nll_items else "",
                "mean_nll_delta_vs_full_raw": statistics.mean(item.nll_delta_vs_full_raw for item in nll_items)
                if nll_items
                else "",
                "avg_prompt_tokens": statistics.mean(item.prompt_tokens for item in items),
                "avg_token_ratio_vs_full_raw": statistics.mean(item.token_ratio_vs_full_raw for item in items),
                "avg_forward_seconds": statistics.mean(item.forward_seconds for item in items),
            }
        )
    return out


def summarize_oracle(rows: list[OracleRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[OracleRow]] = {}
    for row in rows:
        grouped.setdefault(row.task_family, []).append(row)
        grouped.setdefault("__overall__", []).append(row)

    out: list[dict[str, Any]] = []
    for task_family, items in sorted(grouped.items()):
        counts: dict[str, int] = {}
        for item in items:
            counts[item.selected_method] = counts.get(item.selected_method, 0) + 1
        payload: dict[str, Any] = {
            "task_family": task_family,
            "samples": len(items),
            "success_rate": statistics.mean(item.success for item in items),
            "avg_prompt_tokens": statistics.mean(item.prompt_tokens for item in items),
            "avg_token_ratio_vs_full_raw": statistics.mean(item.token_ratio_vs_full_raw for item in items),
            "avg_forward_seconds": statistics.mean(item.forward_seconds for item in items),
            "fallback_rate": statistics.mean(item.fallback_used for item in items),
        }
        for method, count in sorted(counts.items()):
            payload[f"select_{method}"] = count
            payload[f"select_{method}_rate"] = count / len(items)
        out.append(payload)
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset = load_token_ids(tokenizer, config)
    summary_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, context_config(config))
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
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    model.eval()
    input_device = next(model.parameters()).device

    generation_cases = build_generation_cases(token_ids_by_dataset, config)
    exact_cases = build_stress_cases(tokenizer, token_ids_by_dataset, config)
    method_rows: list[MethodRow] = []
    oracle_rows: list[OracleRow] = []

    for case in generation_cases:
        rows = evaluate_generation_case(model, tokenizer, case, config, summary_scorer, input_device)
        method_rows.extend(rows)
        oracle_rows.append(choose_oracle(rows))

    for case_idx, case in enumerate(exact_cases):
        rows = evaluate_exact_case(model, tokenizer, case, case_idx, exact_cases, config, summary_scorer, input_device)
        method_rows.extend(rows)
        oracle_rows.append(choose_oracle(rows))

    method_summary = summarize_methods(method_rows)
    oracle_summary = summarize_oracle(oracle_rows)

    write_csv(output_dir / "method_rows.csv", [asdict(row) for row in method_rows])
    write_csv(output_dir / "oracle_rows.csv", [asdict(row) for row in oracle_rows])
    write_csv(output_dir / "method_summary.csv", method_summary)
    write_csv(output_dir / "oracle_summary.csv", oracle_summary)
    payload = {
        "config": asdict(config),
        "summary_scorer": summary_scorer.metadata if summary_scorer is not None else None,
        "method_summary": method_summary,
        "oracle_summary": oracle_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("ORACLE")
    print("task_family,samples,success_rate,avg_token_ratio_vs_full_raw,avg_forward_seconds,fallback_rate,selection")
    for row in oracle_summary:
        selection = {
            key.removeprefix("select_").removesuffix("_rate"): value
            for key, value in row.items()
            if key.startswith("select_") and key.endswith("_rate")
        }
        print(
            f"{row['task_family']},{row['samples']},{row['success_rate']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f},{row['avg_forward_seconds']:.4f},"
            f"{row['fallback_rate']:.4f},{selection}"
        )
    print("METHODS")
    print("task_family,method,samples,success_rate,avg_token_ratio_vs_full_raw,avg_forward_seconds")
    for row in method_summary:
        print(
            f"{row['task_family']},{row['method']},{row['samples']},{row['success_rate']:.4f},"
            f"{row['avg_token_ratio_vs_full_raw']:.4f},{row['avg_forward_seconds']:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
