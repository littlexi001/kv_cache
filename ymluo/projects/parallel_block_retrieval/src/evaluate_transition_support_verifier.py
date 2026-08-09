from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_branch_transition_verifier import choose_branch
from profile_real_qk import resolve_dtype
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated state transitions with a frozen Qwen support verifier."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--generation_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--step_type",
        choices=["resolve_bridge", "resolve_answer_from_bridge"],
        default="resolve_bridge",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def support_prompt(question: str, evidence: str, proposed_answer: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Proposed answer:\n"
        f"{proposed_answer}\n\n"
        "Does the evidence directly support the proposed answer as an answer to the "
        "question? Reply with only Yes or No."
    )


def answer_likelihood_prompt(question: str, evidence: str) -> str:
    return (
        "Evidence:\n"
        f"{evidence}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Return only the shortest exact answer supported by the evidence.\n"
        "Answer:"
    )


def logit_group_margin(
    logits: torch.Tensor, positive_ids: Sequence[int], negative_ids: Sequence[int]
) -> torch.Tensor:
    positive = torch.logsumexp(logits[:, list(positive_ids)].float(), dim=1)
    negative = torch.logsumexp(logits[:, list(negative_ids)].float(), dim=1)
    return positive - negative


def make_chat_prefix(tokenizer: Any, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def score_yes_no(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    positive_ids: Sequence[int],
    negative_ids: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts[start : start + batch_size]
        ]
        encoded = tokenizer(
            batch_prompts, padding=True, return_tensors="pt", add_special_tokens=False
        ).to(device)
        logits = model(**encoded, use_cache=False).logits[:, -1, :]
        scores.extend(
            float(item)
            for item in logit_group_margin(
                logits, positive_ids, negative_ids
            ).cpu().tolist()
        )
    return scores


@torch.inference_mode()
def score_answer_logprob(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    output = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(prompts), batch_size):
        records = []
        for prompt, answer in zip(
            prompts[start : start + batch_size], answers[start : start + batch_size]
        ):
            prefix = make_chat_prefix(tokenizer, prompt)
            answer_ids = tokenizer(
                answer, add_special_tokens=False
            )["input_ids"]
            if not answer_ids:
                answer_ids = [int(tokenizer.eos_token_id)]
            records.append((prefix, answer_ids))
        max_length = max(len(prefix) + len(answer) for prefix, answer in records)
        input_ids = []
        attention_mask = []
        answer_masks = []
        for prefix, answer_ids in records:
            sequence = prefix + answer_ids
            padding = max_length - len(sequence)
            input_ids.append([pad_id] * padding + sequence)
            attention_mask.append([0] * padding + [1] * len(sequence))
            answer_masks.append(
                [0] * (padding + len(prefix)) + [1] * len(answer_ids)
            )
        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        answer_mask = torch.tensor(answer_masks, dtype=torch.bool, device=device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        token_logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(
            2, ids[:, 1:].unsqueeze(-1)
        )[:, :, 0]
        shifted_mask = answer_mask[:, 1:]
        sums = (token_logprobs * shifted_mask).sum(dim=1)
        counts = shifted_mask.sum(dim=1).clamp_min(1)
        output.extend(float(item) for item in (sums / counts).cpu().tolist())
    return output


def selection_accuracy(rows: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.fmean(
        bool(row["branch_target_hits"][int(row[field])]) for row in rows
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    steps = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) == args.split and row["step_type"] == args.step_type
    }
    generations = [
        row
        for row in read_jsonl(Path(args.generation_rows_path))
        if str(row["split"]) == args.split and row["step_type"] == args.step_type
    ]
    generations.sort(key=lambda row: int(row["query_id"]))
    if args.max_queries:
        generations = generations[: args.max_queries]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    positive_ids = list(
        dict.fromkeys(
            tokenizer(value, add_special_tokens=False)["input_ids"][0]
            for value in ("Yes", " Yes", "yes")
        )
    )
    negative_ids = list(
        dict.fromkeys(
            tokenizer(value, add_special_tokens=False)["input_ids"][0]
            for value in ("No", " No", "no")
        )
    )

    support_prompts = []
    likelihood_prompts = []
    answers = []
    metadata = []
    for generation in generations:
        query_id = int(generation["query_id"])
        step = steps[query_id]
        heuristic_index, heuristic_trace = choose_branch(
            step, generation["branches"]
        )
        branch_hits = []
        for branch in generation["branches"]:
            answer = str(branch["generated_text"]).strip()
            evidence = str(branch["memory_text"])
            question = str(step["step_question"])
            support_prompts.append(support_prompt(question, evidence, answer))
            likelihood_prompts.append(answer_likelihood_prompt(question, evidence))
            answers.append(answer)
            branch_hits.append(answer_hit(answer, [str(step["target_output"])]))
        metadata.append(
            {
                "query_id": query_id,
                "heuristic_index": heuristic_index,
                "heuristic_scores": [float(item["score"]) for item in heuristic_trace],
                "heuristic_trace": heuristic_trace,
                "branch_target_hits": branch_hits,
                "branch_retrieval_ranks": [
                    int(branch["rank"]) for branch in generation["branches"]
                ],
                "branch_generated_tokens": [
                    int(branch["generated_tokens"]) for branch in generation["branches"]
                ],
                "generated_answers": [
                    str(branch["generated_text"]) for branch in generation["branches"]
                ],
            }
        )

    started = time.perf_counter()
    yes_no_scores = score_yes_no(
        model,
        tokenizer,
        support_prompts,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        batch_size=args.batch_size,
        device=device,
    )
    yes_no_seconds = time.perf_counter() - started
    started = time.perf_counter()
    answer_logprobs = score_answer_logprob(
        model,
        tokenizer,
        likelihood_prompts,
        answers,
        batch_size=args.batch_size,
        device=device,
    )
    answer_logprob_seconds = time.perf_counter() - started

    branch_count = len(generations[0]["branches"]) if generations else 0
    rows = []
    for query_offset, item in enumerate(metadata):
        start = query_offset * branch_count
        end = start + branch_count
        yes_scores = yes_no_scores[start:end]
        likelihood_scores = answer_logprobs[start:end]
        rows.append(
            {
                **item,
                "yes_no_scores": yes_scores,
                "answer_logprob_scores": likelihood_scores,
                "yes_no_index": max(range(branch_count), key=yes_scores.__getitem__),
                "answer_logprob_index": max(
                    range(branch_count), key=likelihood_scores.__getitem__
                ),
                "any_branch_hit": any(item["branch_target_hits"]),
            }
        )

    summary = {
        "source": "frozen-Qwen transition support and answer-likelihood verifier",
        "selection_uses_gold": False,
        "split": args.split,
        "step_type": args.step_type,
        "queries": len(rows),
        "branches_per_query": branch_count,
        "heuristic_accuracy": selection_accuracy(rows, "heuristic_index"),
        "yes_no_accuracy": selection_accuracy(rows, "yes_no_index"),
        "answer_logprob_accuracy": selection_accuracy(rows, "answer_logprob_index"),
        "any_branch_accuracy": statistics.fmean(row["any_branch_hit"] for row in rows),
        "yes_no_seconds": yes_no_seconds,
        "answer_logprob_seconds": answer_logprob_seconds,
        "mean_yes_no_ms_per_query": 1000.0 * yes_no_seconds / max(1, len(rows)),
        "mean_answer_logprob_ms_per_query": 1000.0
        * answer_logprob_seconds
        / max(1, len(rows)),
        "positive_token_ids": positive_ids,
        "negative_token_ids": negative_ids,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
