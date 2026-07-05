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
    build_blocks,
    build_static_summaries,
    content_words,
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
    methods: tuple[str, ...]
    samples_per_dataset: int
    sample_stride_tokens: int
    prefill_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    eval_start_tokens: int
    num_choices: int
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


@dataclass(frozen=True)
class StressCase:
    dataset: str
    sample_id: int
    start_token: int
    task_type: str
    key: str
    answer: str
    question: str
    prefix_ids: tuple[int, ...]
    evidence_block_id: int


@dataclass
class EvalRow:
    dataset: str
    sample_id: int
    start_token: int
    task_type: str
    method: str
    evidence_block_id: int
    prompt_tokens: int
    total_choice_tokens: int
    token_ratio_vs_full_raw: float
    answer_retained: int
    selected_answer: str
    correct_answer: str
    choice_correct: int
    correct_choice_nll: float
    best_choice_nll: float
    nll_margin_best_minus_correct: float
    forward_seconds: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Stress-test summary memory on tasks that need exact raw evidence."
    )
    parser.add_argument(
        "--output_dir",
        default="ymluo/projects/learned_hierarchical_summary_memory/outputs/full_information_stress_eval",
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--text_paths", required=True)
    parser.add_argument("--dataset_names", required=True)
    parser.add_argument(
        "--methods",
        default=(
            "full_raw,learned_static_hier,learned_static_sum1000,"
            "retrieval_raw_k1,retrieval_raw_k2,risk_gate_full_on_exact"
        ),
    )
    parser.add_argument("--samples_per_dataset", type=int, default=4)
    parser.add_argument("--sample_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=160_000)
    parser.add_argument("--eval_start_tokens", type=int, default=40_000)
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070314)
    parser.add_argument("--seed", type=int, default=2026070315)
    args = parser.parse_args()

    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if len(text_paths) != len(dataset_names):
        raise ValueError("--text_paths and --dataset_names must have the same count")
    if args.num_choices < 2:
        raise ValueError("--num_choices must be at least 2")
    return Config(**{**vars(args), "text_paths": text_paths, "dataset_names": dataset_names, "methods": methods})


def context_config(config: Config) -> PplContextConfig:
    return PplContextConfig(
        output_dir=config.output_dir,
        model_name_or_path=config.model_name_or_path,
        text_paths=config.text_paths,
        dataset_names=config.dataset_names,
        methods=config.methods,
        samples_per_dataset=config.samples_per_dataset,
        sample_stride_tokens=config.sample_stride_tokens,
        prefill_tokens=config.prefill_tokens,
        eval_tokens=1,
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
    max_start = len(ids) - config.prefill_tokens
    if max_start < 0:
        return []
    first = min(config.eval_start_tokens, max_start)
    starts = [min(max_start, first + idx * config.sample_stride_tokens) for idx in range(config.samples_per_dataset)]
    deduped: list[int] = []
    for start in starts:
        if start not in deduped:
            deduped.append(start)
    return deduped


def make_answer(dataset: str, sample_id: int, rng: random.Random) -> tuple[str, str]:
    prefixes = ["lumen", "atlas", "cedar", "violet", "ember", "quartz", "harbor", "silver"]
    suffixes = ["keystone", "cipher", "signal", "archive", "vector", "ledger", "anchor", "relay"]
    key = f"{dataset.upper()}-{sample_id:03d}-{rng.randrange(100, 999)}"
    answer = f"{rng.choice(prefixes)}-{rng.randrange(100, 999)}-{rng.choice(suffixes)}"
    return key, answer


def build_stress_cases(
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
) -> list[StressCase]:
    rng = random.Random(config.seed)
    cases: list[StressCase] = []
    for dataset, all_ids in token_ids_by_dataset.items():
        for sample_id, start in enumerate(sample_starts(all_ids, config)):
            raw_prefix_ids = all_ids[start : start + config.prefill_tokens]
            raw_blocks = [
                raw_prefix_ids[idx : idx + config.block_tokens]
                for idx in range(0, len(raw_prefix_ids), config.block_tokens)
            ]
            if len(raw_blocks) < 2:
                continue

            recent_block_start = max(0, len(raw_prefix_ids) - config.recent_tokens)
            older_token_count = recent_block_start
            older_block_count = max(1, math.ceil(older_token_count / config.block_tokens))
            evidence_block_id = sample_id % max(1, older_block_count - 1)

            key, answer = make_answer(dataset, sample_id, rng)
            record = (
                "\n\nSPECIAL MEMORY RECORD. "
                f"The private access code for {key} is {answer}. "
                "This value must be recalled exactly.\n\n"
            )

            block_texts = [tokenizer.decode(block, skip_special_tokens=True) for block in raw_blocks]
            block_texts[evidence_block_id] = record + block_texts[evidence_block_id]
            prefix_text = "\n".join(block_texts)
            prefix_ids = tuple(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
            question = (
                "Use the memory above to answer the exact-evidence question.\n"
                f"Question: What is the private access code for {key}?\n"
                "Answer with only the code.\n"
                "Answer:"
            )
            cases.append(
                StressCase(
                    dataset=dataset,
                    sample_id=sample_id,
                    start_token=start,
                    task_type="exact_key_value",
                    key=key,
                    answer=answer,
                    question=question,
                    prefix_ids=prefix_ids,
                    evidence_block_id=evidence_block_id,
                )
            )
    return cases


def static_hier_summary_text(summaries: list[Any]) -> str:
    parts: list[str] = []
    for idx, item in enumerate(summaries):
        distance_from_recent = len(summaries) - idx
        if distance_from_recent == 1:
            parts.append(item.summary1000)
        elif distance_from_recent <= 3:
            parts.append(item.summary100)
        else:
            parts.append(item.summary10)
    return "\n".join(parts)


def lexical_retrieve_blocks(question: str, blocks: list[Any], top_k: int) -> list[Any]:
    query_terms = set(content_words(question))
    scored: list[tuple[int, int, Any]] = []
    for block in blocks:
        block_terms = set(content_words(block.text))
        overlap = len(query_terms & block_terms)
        scored.append((overlap, -block.block_id, block))
    scored.sort(reverse=True)
    return [block for overlap, _, block in scored[:top_k] if overlap > 0]


def context_with_raw_retrieval(
    config: Config,
    tokenizer: Any,
    case: StressCase,
    top_k: int,
    summary_scorer: LearnedSummaryScorer | None,
) -> str:
    prefix_ids = list(case.prefix_ids)
    recent_ids = prefix_ids[-config.recent_tokens :] if config.recent_tokens > 0 else []
    older_ids = prefix_ids[: max(0, len(prefix_ids) - len(recent_ids))]
    recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)
    blocks = build_blocks(tokenizer, older_ids, config.block_tokens)
    summaries = build_static_summaries(context_config(config), blocks, summary_scorer=summary_scorer)
    retrieved = lexical_retrieve_blocks(case.question, blocks, top_k=top_k)
    retrieved_text = "\n\n".join(f"[raw block {block.block_id}]\n{block.text}" for block in retrieved)
    return (
        f"Static memory summaries:\n{static_hier_summary_text(summaries)}\n\n"
        f"Retrieved raw evidence blocks:\n{retrieved_text}\n\n"
        f"Recent raw text:\n{recent_text}"
    )


def make_prompt(
    config: Config,
    tokenizer: Any,
    case: StressCase,
    method: str,
    summary_scorer: LearnedSummaryScorer | None,
) -> str:
    prefix_ids = list(case.prefix_ids)
    if method == "full_raw":
        memory = context_for_method(context_config(config), tokenizer, prefix_ids, "full_raw", summary_scorer=summary_scorer)
    elif method == "learned_static_hier":
        memory = context_for_method(context_config(config), tokenizer, prefix_ids, "static_hier", summary_scorer=summary_scorer)
    elif method == "learned_static_sum1000":
        memory = context_for_method(context_config(config), tokenizer, prefix_ids, "static_sum1000", summary_scorer=summary_scorer)
    elif method.startswith("retrieval_raw_k"):
        top_k = int(method.removeprefix("retrieval_raw_k"))
        memory = context_with_raw_retrieval(config, tokenizer, case, top_k=top_k, summary_scorer=summary_scorer)
    elif method == "risk_gate_full_on_exact":
        memory = context_for_method(context_config(config), tokenizer, prefix_ids, "full_raw", summary_scorer=summary_scorer)
    else:
        raise ValueError(method)
    return f"{memory}\n\n{case.question}"


def candidate_answers(cases: list[StressCase], idx: int, config: Config) -> list[str]:
    correct = cases[idx].answer
    distractors = [case.answer for case in cases if case.answer != correct]
    if len(distractors) < config.num_choices - 1:
        rng = random.Random(config.seed + idx)
        while len(distractors) < config.num_choices - 1:
            _, answer = make_answer("distractor", len(distractors), rng)
            if answer != correct:
                distractors.append(answer)
    rng = random.Random(config.seed + 10_000 + idx)
    chosen = rng.sample(distractors, config.num_choices - 1)
    choices = [correct] + chosen
    rng.shuffle(choices)
    return choices


def score_answer_choice(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
    device: torch.device,
) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id or 0]
    target_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long, device=device).view(1, -1)
    with torch.inference_mode():
        nll, _ = score_target(model, input_ids, len(prompt_ids), len(target_ids))
    total_tokens = int(input_ids.shape[1])
    del input_ids
    return nll, total_tokens


def summarize(rows: list[EvalRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EvalRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.method), []).append(row)

    full_tokens_by_dataset: dict[str, float] = {}
    full_seconds_by_dataset: dict[str, float] = {}
    for (dataset, method), items in grouped.items():
        if method == "full_raw":
            full_tokens_by_dataset[dataset] = statistics.mean(item.prompt_tokens for item in items)
            full_seconds_by_dataset[dataset] = statistics.mean(item.forward_seconds for item in items)

    summary: list[dict[str, Any]] = []
    for (dataset, method), items in sorted(grouped.items()):
        avg_prompt_tokens = statistics.mean(item.prompt_tokens for item in items)
        avg_seconds = statistics.mean(item.forward_seconds for item in items)
        full_tokens = full_tokens_by_dataset.get(dataset, avg_prompt_tokens)
        full_seconds = full_seconds_by_dataset.get(dataset, avg_seconds)
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "samples": len(items),
                "answer_retained_rate": statistics.mean(item.answer_retained for item in items),
                "choice_accuracy": statistics.mean(item.choice_correct for item in items),
                "avg_prompt_tokens": avg_prompt_tokens,
                "token_ratio_vs_full_raw": avg_prompt_tokens / full_tokens if full_tokens else 0.0,
                "avg_forward_seconds": avg_seconds,
                "speedup_vs_full_raw": full_seconds / avg_seconds if avg_seconds else 0.0,
                "avg_correct_choice_nll": statistics.mean(item.correct_choice_nll for item in items),
                "avg_nll_margin_best_minus_correct": statistics.mean(
                    item.nll_margin_best_minus_correct for item in items
                ),
            }
        )

    overall_grouped: dict[str, list[EvalRow]] = {}
    for row in rows:
        overall_grouped.setdefault(row.method, []).append(row)
    for method, items in sorted(overall_grouped.items()):
        full_items = [row for row in rows if row.method == "full_raw" and row.dataset in {item.dataset for item in items}]
        full_prompt = statistics.mean(item.prompt_tokens for item in full_items) if full_items else statistics.mean(
            item.prompt_tokens for item in items
        )
        full_seconds = statistics.mean(item.forward_seconds for item in full_items) if full_items else statistics.mean(
            item.forward_seconds for item in items
        )
        avg_prompt = statistics.mean(item.prompt_tokens for item in items)
        avg_seconds = statistics.mean(item.forward_seconds for item in items)
        summary.append(
            {
                "dataset": "__overall__",
                "method": method,
                "samples": len(items),
                "answer_retained_rate": statistics.mean(item.answer_retained for item in items),
                "choice_accuracy": statistics.mean(item.choice_correct for item in items),
                "avg_prompt_tokens": avg_prompt,
                "token_ratio_vs_full_raw": avg_prompt / full_prompt if full_prompt else 0.0,
                "avg_forward_seconds": avg_seconds,
                "speedup_vs_full_raw": full_seconds / avg_seconds if avg_seconds else 0.0,
                "avg_correct_choice_nll": statistics.mean(item.correct_choice_nll for item in items),
                "avg_nll_margin_best_minus_correct": statistics.mean(
                    item.nll_margin_best_minus_correct for item in items
                ),
            }
        )
    return summary


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

    cases = build_stress_cases(tokenizer, token_ids_by_dataset, config)
    if not cases:
        raise ValueError("no stress cases were created")

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    model.eval()
    input_device = next(model.parameters()).device

    rows: list[EvalRow] = []
    case_payloads: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        case_payloads.append(
            {
                "dataset": case.dataset,
                "sample_id": case.sample_id,
                "start_token": case.start_token,
                "task_type": case.task_type,
                "key": case.key,
                "answer": case.answer,
                "question": case.question,
                "evidence_block_id": case.evidence_block_id,
            }
        )
        choices = candidate_answers(cases, case_idx, config)
        full_prompt = make_prompt(config, tokenizer, case, "full_raw", summary_scorer=summary_scorer)
        full_prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
        for method in config.methods:
            prompt = make_prompt(config, tokenizer, case, method, summary_scorer=summary_scorer)
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

            synchronize(torch, input_device)
            started = time.perf_counter()
            scores: list[tuple[str, float, int]] = []
            for choice in choices:
                nll, total_tokens = score_answer_choice(model, tokenizer, prompt, choice, input_device)
                scores.append((choice, nll, total_tokens))
            synchronize(torch, input_device)
            elapsed = time.perf_counter() - started

            scores.sort(key=lambda item: item[1])
            selected_answer, best_nll, _ = scores[0]
            correct_nll = next(nll for answer, nll, _ in scores if answer == case.answer)
            total_choice_tokens = sum(total_tokens for _, _, total_tokens in scores)
            denom_tokens = full_prompt_tokens if full_prompt_tokens else prompt_tokens
            rows.append(
                EvalRow(
                    dataset=case.dataset,
                    sample_id=case.sample_id,
                    start_token=case.start_token,
                    task_type=case.task_type,
                    method=method,
                    evidence_block_id=case.evidence_block_id,
                    prompt_tokens=prompt_tokens,
                    total_choice_tokens=total_choice_tokens,
                    token_ratio_vs_full_raw=prompt_tokens / denom_tokens if denom_tokens else 0.0,
                    answer_retained=int(case.answer in prompt),
                    selected_answer=selected_answer,
                    correct_answer=case.answer,
                    choice_correct=int(selected_answer == case.answer),
                    correct_choice_nll=correct_nll,
                    best_choice_nll=best_nll,
                    nll_margin_best_minus_correct=best_nll - correct_nll,
                    forward_seconds=elapsed,
                )
            )

    summary = summarize(rows)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "cases.json").write_text(json.dumps(case_payloads, indent=2), encoding="utf-8")
    payload = {
        "config": asdict(config),
        "summary_scorer": summary_scorer.metadata if summary_scorer is not None else None,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("dataset,method,samples,answer_retained_rate,choice_accuracy,avg_prompt_tokens,token_ratio_vs_full_raw,avg_forward_seconds,speedup_vs_full_raw")
    for row in summary:
        print(
            f"{row['dataset']},{row['method']},{row['samples']},"
            f"{row['answer_retained_rate']:.4f},{row['choice_accuracy']:.4f},"
            f"{row['avg_prompt_tokens']:.1f},{row['token_ratio_vs_full_raw']:.4f},"
            f"{row['avg_forward_seconds']:.4f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
