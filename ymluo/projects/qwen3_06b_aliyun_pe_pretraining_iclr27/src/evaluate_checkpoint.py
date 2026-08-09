from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import string
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from data_pipeline import take_validation_blocks
from io_utils import append_jsonl, utc_timestamp, write_csv, write_json
from model_utils import input_device, load_model, load_tokenizer
from pe_strategies import load_strategy, save_strategy_profile


LONG_BENCH_PROMPTS = {
    "multifieldqa_en": (
        "Read the following text and answer the question briefly.\n\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give the answer.\n\n"
        "{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "qasper": (
        "Read the scientific article and answer the question. If it cannot be answered, "
        "reply unanswerable.\n\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    common = collections.Counter(prediction_tokens) & collections.Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_prediction(prediction: str, answers: list[str]) -> dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_answers = [normalize_answer(answer) for answer in answers]
    return {
        "qa_f1": max(token_f1(prediction, answer) for answer in answers),
        "exact_match": float(normalized_prediction in normalized_answers),
        "contains_answer": float(
            any(answer and answer in normalized_prediction for answer in normalized_answers)
        ),
    }


def truncate_middle(tokens: list[int], limit: int) -> list[int]:
    if len(tokens) <= limit:
        return tokens
    first = limit // 2
    return tokens[:first] + tokens[-(limit - first) :]


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> tuple[str, list[int], float]:
    device = input_device(model)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    started = time.perf_counter()
    generated = model.generate(
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    new_ids = generated[0, prompt.shape[1] :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), new_ids, elapsed


@torch.inference_mode()
def answer_nll(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    answers: list[str],
) -> tuple[float, int, float, int]:
    device = input_device(model)
    candidates: list[list[int]] = []
    for answer in answers[:4]:
        for text in [answer, " " + answer]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in candidates:
                candidates.append(ids)
    if not candidates:
        return float("nan"), 0, float("nan"), 0
    best = (float("inf"), 0, float("inf"), 0)
    for candidate in candidates:
        all_ids = prompt_ids + candidate
        tokens = torch.tensor(all_ids, dtype=torch.long, device=device).unsqueeze(0)
        output = model(input_ids=tokens, use_cache=False)
        start = len(prompt_ids) - 1
        logits = output.logits[0, start : start + len(candidate), :].float()
        labels = torch.tensor(candidate, dtype=torch.long, device=device)
        losses = F.cross_entropy(logits, labels, reduction="none")
        first_accuracy = int(int(logits[0].argmax().item()) == candidate[0])
        mean_nll = float(losses.mean().item())
        first_nll = float(losses[0].item())
        if mean_nll < best[0]:
            best = (mean_nll, len(candidate), first_nll, first_accuracy)
        del output, tokens, logits, losses
    return best


def distribute_filler(total: int, slots: int, filler_unit: list[int]) -> list[list[int]]:
    if total < 0:
        raise ValueError("target length is smaller than fixed prompt content")
    stream = (filler_unit * (math.ceil(total / max(1, len(filler_unit))) + 1))[:total]
    sizes = [total // slots + (1 if index < total % slots else 0) for index in range(slots)]
    output: list[list[int]] = []
    cursor = 0
    for size in sizes:
        output.append(stream[cursor : cursor + size])
        cursor += size
    return output


def controlled_case(tokenizer: Any, task: str, target_length: int, seed: int) -> tuple[list[int], list[str], dict[str, Any]]:
    rng = random.Random(seed)
    header = tokenizer.encode(
        "Use only the records below. Return only the requested value.\n",
        add_special_tokens=False,
    )
    query: str
    answer: str
    facts: list[str]
    if task == "niah_single":
        keys = ["amber", "cedar", "falcon", "harbor", "lunar", "maple", "quartz", "river"]
        values = ["cobalt", "violet", "tulip", "silver", "marble", "saffron", "willow", "coral"]
        target = seed % len(keys)
        facts = [f"Record {key} has code {value}.\n" for key, value in zip(keys, values)]
        answer = values[target]
        query = f"\nQuestion: What is the code for record {keys[target]}?\nAnswer:"
    elif task == "niah_multi":
        keys = ["atlas", "basil", "comet", "delta", "ember", "fjord"]
        values = ["plum", "ivory", "moss", "pearl", "rose", "teal"]
        first = seed % len(keys)
        second = (first + 3) % len(keys)
        facts = [f"Record {key} has code {value}.\n" for key, value in zip(keys, values)]
        answer = f"{values[first]}, {values[second]}"
        query = (
            f"\nQuestion: Give the codes for {keys[first]} and {keys[second]} in that order, "
            "separated by a comma.\nAnswer:"
        )
    elif task == "variable_tracking":
        names = ["Kestrel", "Mira", "Orion", "Pavo", "Rhea"]
        value = ["bronze", "cyan", "jade", "lilac"][seed % 4]
        facts = [f"Variable {names[-1]} stores {value}.\n"]
        for index in range(len(names) - 2, -1, -1):
            facts.append(f"Variable {names[index]} stores the value in {names[index + 1]}.\n")
        answer = value
        query = f"\nQuestion: What value is ultimately stored in {names[0]}?\nAnswer:"
    else:
        raise ValueError(task)
    rng.shuffle(facts)
    fact_ids = [tokenizer.encode(fact, add_special_tokens=False) for fact in facts]
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    filler_unit = tokenizer.encode(
        " The archive contains an ordinary note about weather, meals, and daily routines.",
        add_special_tokens=False,
    )
    fixed = len(header) + len(query_ids) + sum(len(value) for value in fact_ids)
    filler = distribute_filler(target_length - fixed, len(fact_ids) + 1, filler_unit)
    prompt_ids = list(header)
    for index, fact in enumerate(fact_ids):
        prompt_ids.extend(filler[index])
        prompt_ids.extend(fact)
    prompt_ids.extend(filler[-1])
    prompt_ids.extend(query_ids)
    if len(prompt_ids) != target_length:
        raise AssertionError((len(prompt_ids), target_length))
    return prompt_ids, [answer], {"task": task, "seed": seed, "facts": facts, "query": query}


@torch.inference_mode()
def evaluate_ppl(
    model: Any,
    tokenizer: Any,
    validation_manifest: Path,
    sequence_length: int,
    blocks: int,
    seed: int,
) -> dict[str, Any]:
    device = input_device(model)
    losses: list[float] = []
    tokens_total = 0
    started = time.perf_counter()
    for block in take_validation_blocks(
        validation_manifest, tokenizer, sequence_length, blocks, seed
    ):
        tokens = block.to(device).unsqueeze(0)
        output = model(input_ids=tokens, labels=tokens, use_cache=False)
        losses.append(float(output.loss.item()))
        tokens_total += int(tokens.numel() - 1)
        del output, tokens
    nll = mean(losses)
    return {
        "status": "complete",
        "blocks": len(losses),
        "tokens": tokens_total,
        "sequence_length": sequence_length,
        "mean_nll": nll,
        "ppl": math.exp(min(nll, 30.0)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_controlled(
    model: Any,
    tokenizer: Any,
    output_path: Path,
    lengths: list[int],
    samples_per_task: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tasks = ["niah_single", "niah_multi", "variable_tracking"]
    for length in lengths:
        for task_index, task in enumerate(tasks):
            for sample in range(samples_per_task):
                case_seed = seed + length * 100 + task_index * 10_000 + sample
                try:
                    prompt_ids, answers, metadata = controlled_case(
                        tokenizer, task, length, case_seed
                    )
                    prediction, generated_ids, generation_seconds = generate_text(
                        model, tokenizer, prompt_ids, max_new_tokens
                    )
                    nll, answer_tokens, first_nll, first_accuracy = answer_nll(
                        model, tokenizer, prompt_ids, answers
                    )
                    scores = score_prediction(prediction, answers)
                    row = {
                        "benchmark": "controlled_ruler_style",
                        "length": length,
                        "task": task,
                        "sample": sample,
                        "case_seed": case_seed,
                        "answers": answers,
                        "prediction": prediction,
                        "generated_token_ids": generated_ids,
                        "gold_answer_mean_nll": nll,
                        "gold_answer_ppl": math.exp(min(nll, 30.0)),
                        "gold_answer_tokens": answer_tokens,
                        "first_token_nll": first_nll,
                        "first_token_accuracy": first_accuracy,
                        "generation_seconds": generation_seconds,
                        **scores,
                        "case": metadata,
                    }
                    rows.append(row)
                    append_jsonl(output_path, row)
                except Exception as error:
                    failure = {
                        "stage": "controlled_ruler_style",
                        "length": length,
                        "task": task,
                        "sample": sample,
                        "error": repr(error),
                    }
                    failures.append(failure)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return rows, failures


def load_longbench(task: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    return [dict(row) for row in dataset]


def evaluate_longbench(
    model: Any,
    tokenizer: Any,
    output_path: Path,
    tasks: list[str],
    samples_per_task: int,
    max_prompt_length: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        try:
            source = load_longbench(task)
        except Exception as error:
            failures.append({"stage": "longbench_download", "task": task, "error": repr(error)})
            continue
        indices = list(range(len(source)))
        random.Random(seed + task_index).shuffle(indices)
        for sample_index, source_index in enumerate(indices[:samples_per_task]):
            row = source[source_index]
            try:
                if task not in LONG_BENCH_PROMPTS:
                    raise ValueError(f"no prompt template for LongBench task {task}")
                prompt = LONG_BENCH_PROMPTS[task].format(
                    context=str(row["context"]), input=str(row["input"])
                )
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = truncate_middle(prompt_ids, max_prompt_length)
                answers = [str(answer) for answer in row["answers"]]
                prediction, generated_ids, generation_seconds = generate_text(
                    model, tokenizer, prompt_ids, max_new_tokens
                )
                nll, answer_tokens, first_nll, first_accuracy = answer_nll(
                    model, tokenizer, prompt_ids, answers
                )
                scores = score_prediction(prediction, answers)
                result = {
                    "benchmark": "longbench_v1",
                    "task": task,
                    "sample": sample_index,
                    "source_index": source_index,
                    "prompt_tokens": len(prompt_ids),
                    "answers": answers,
                    "prediction": prediction,
                    "generated_token_ids": generated_ids,
                    "gold_answer_mean_nll": nll,
                    "gold_answer_ppl": math.exp(min(nll, 30.0)),
                    "gold_answer_tokens": answer_tokens,
                    "first_token_nll": first_nll,
                    "first_token_accuracy": first_accuracy,
                    "generation_seconds": generation_seconds,
                    **scores,
                }
                rows.append(result)
                append_jsonl(output_path, result)
            except Exception as error:
                failures.append(
                    {
                        "stage": "longbench_sample",
                        "task": task,
                        "sample": sample_index,
                        "source_index": source_index,
                        "error": repr(error),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows, failures


def aggregate_rows(rows: Iterable[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        nll = mean(float(row["gold_answer_mean_nll"]) for row in selected)
        output.append(
            {
                **dict(zip(group_fields, key)),
                "samples": len(selected),
                "qa_f1_percent": 100.0 * mean(float(row["qa_f1"]) for row in selected),
                "exact_match_percent": 100.0 * mean(float(row["exact_match"]) for row in selected),
                "contains_answer_percent": 100.0 * mean(float(row["contains_answer"]) for row in selected),
                "first_token_accuracy_percent": 100.0 * mean(float(row["first_token_accuracy"]) for row in selected),
                "gold_answer_mean_nll": nll,
                "gold_answer_ppl": math.exp(min(nll, 30.0)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--eval-lengths", default="2048,4096,8192")
    parser.add_argument("--ruler-samples-per-task", type=int, default=4)
    parser.add_argument("--ppl-blocks", type=int, default=16)
    parser.add_argument("--ppl-sequence-length", type=int, default=2048)
    parser.add_argument("--run-longbench", type=int, choices=[0, 1], default=1)
    parser.add_argument("--longbench-tasks", default="hotpotqa,2wikimqa,multifieldqa_en")
    parser.add_argument("--longbench-samples-per-task", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument(
        "--initialization", choices=["checkpoint", "from_scratch"], default="checkpoint"
    )
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_status = args.output_dir / "status.json"
    existing_summary = args.output_dir / "summary.json"
    existing_complete = bool(
        existing_status.exists()
        and json.loads(existing_status.read_text(encoding="utf-8")).get("complete", False)
    )
    if args.run_longbench:
        existing_complete = bool(
            existing_complete
            and existing_summary.exists()
            and json.loads(existing_summary.read_text(encoding="utf-8")).get("longbench_status")
            == "complete"
        )
    if not existing_complete:
        for filename in ["controlled_rows.jsonl", "longbench_rows.jsonl", "failures.jsonl"]:
            (args.output_dir / filename).unlink(missing_ok=True)
    failures_path = args.output_dir / "failures.jsonl"
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    strategy = load_strategy(args.strategy)
    tokenizer = load_tokenizer(args.tokenizer_path)
    write_json(
        args.output_dir / "evaluation_contract.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "controlled_tasks": ["niah_single", "niah_multi", "variable_tracking"],
            "longbench_prompt_templates": LONG_BENCH_PROMPTS,
            "metrics": {
                "qa_f1": "maximum normalized whitespace-token overlap F1 against accepted answers",
                "exact_match": "normalized generated text exactly equals an accepted answer",
                "contains_answer": "normalized generated text contains an accepted normalized answer",
                "gold_answer_mean_nll": "minimum mean teacher-forced answer-token NLL over accepted answers and leading-space variants",
                "validation_ppl": "exp(mean next-token NLL) over fixed-size held-out DCLM blocks",
            },
            "protocol_boundary": {
                "controlled_ruler_style": "self-contained diagnostic, not an official RULER leaderboard score",
                "longbench_v1": "official data subset with package-local prompts, not a full official leaderboard reproduction",
            },
        },
    )
    model, chosen_attention = load_model(
        args.model_path,
        strategy,
        args.dtype,
        args.attention_implementation,
        for_training=False,
        initialization=args.initialization,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation requires a CUDA GPU")
    model.to(torch.device("cuda:0"))
    model.eval()
    save_strategy_profile(model, strategy, args.output_dir / "strategy_profile.json")
    failures: list[dict[str, Any]] = []
    try:
        ppl = evaluate_ppl(
            model,
            tokenizer,
            args.validation_manifest,
            args.ppl_sequence_length,
            args.ppl_blocks,
            args.seed + 71,
        )
    except Exception as error:
        ppl = {"status": "failed", "error": repr(error)}
        failures.append({"stage": "validation_ppl", "error": repr(error)})
        torch.cuda.empty_cache()

    lengths = [int(value) for value in args.eval_lengths.split(",") if value.strip()]
    controlled_rows, controlled_failures = evaluate_controlled(
        model,
        tokenizer,
        args.output_dir / "controlled_rows.jsonl",
        lengths,
        args.ruler_samples_per_task,
        args.max_new_tokens,
        args.seed,
    )
    failures.extend(controlled_failures)
    longbench_rows: list[dict[str, Any]] = []
    if args.run_longbench:
        longbench_tasks = [value.strip() for value in args.longbench_tasks.split(",") if value.strip()]
        longbench_rows, longbench_failures = evaluate_longbench(
            model,
            tokenizer,
            args.output_dir / "longbench_rows.jsonl",
            longbench_tasks,
            args.longbench_samples_per_task,
            max(lengths) - args.max_new_tokens,
            args.max_new_tokens,
            args.seed + 101,
        )
        failures.extend(longbench_failures)

    for failure in failures:
        append_jsonl(failures_path, {"timestamp": utc_timestamp(), **failure})
    controlled_summary = aggregate_rows(controlled_rows, ["length", "task"])
    longbench_summary = aggregate_rows(longbench_rows, ["task"])
    write_csv(args.output_dir / "controlled_summary.csv", controlled_summary)
    write_csv(args.output_dir / "longbench_summary.csv", longbench_summary)
    critical_complete = (
        ppl.get("status") == "complete"
        and len(controlled_rows) == len(lengths) * 3 * args.ruler_samples_per_task
    )
    summary = {
        "label": args.label,
        "strategy": strategy.name,
        "step": args.step,
        "model_path": str(args.model_path),
        "attention_implementation": chosen_attention,
        "validation_ppl": ppl,
        "controlled": controlled_summary,
        "longbench": longbench_summary,
        "longbench_status": (
            "disabled" if not args.run_longbench else ("complete" if longbench_rows else "unavailable")
        ),
        "failure_count": len(failures),
        "critical_complete": critical_complete,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "status.json",
        {
            "complete": critical_complete,
            "controlled_rows": len(controlled_rows),
            "longbench_rows": len(longbench_rows),
            "failures": len(failures),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
