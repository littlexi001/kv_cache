from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
PROJECTS = HERE.parents[2]
LONG_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
for directory in (PROJECT / "src", LONG_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from head_frequency_intervention import HeadFrequencyIntervention  # noqa: E402
import run_longbench_rope_sparse as longbench  # noqa: E402


PROMPTS = {
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
}

MAX_NEW_TOKENS = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
}

NO_CHAT = {"lcc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small frozen LongBench-E multi-task panel.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--longbench-dir", required=True, type=Path)
    parser.add_argument("--longbench-code-root", required=True, type=Path)
    parser.add_argument("--specs-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", default=",".join(PROMPTS))
    parser.add_argument("--samples-per-task", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=20260806)
    parser.add_argument("--max-prompt-tokens", type=int, default=39000)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_specs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value["specs"] if isinstance(value, dict) else value
    if not isinstance(specs, list) or not specs:
        raise ValueError("specs-json must contain a non-empty specs list")
    return specs


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def length_bin(length: int) -> str:
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def deterministic_key(dataset: str, sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{sample_id}".encode()).hexdigest()


def select_cases(
    longbench_dir: Path,
    datasets: Sequence[str],
    samples_per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = read_jsonl(longbench_dir / f"{dataset}_e.jsonl")
        groups: dict[str, list[dict[str, Any]]] = {"0-4k": [], "4-8k": [], "8k+": []}
        for index, row in enumerate(rows):
            sample_id = str(row.get("_id", index))
            row = {**row, "sample_id": sample_id, "dataset_name": dataset}
            groups[length_bin(int(row["length"]))].append(row)
        for group in groups.values():
            group.sort(key=lambda row: deterministic_key(dataset, row["sample_id"], seed))
        selected: list[dict[str, Any]] = []
        quota = max(1, samples_per_task // 3)
        for name in ("0-4k", "4-8k", "8k+"):
            selected.extend(groups[name][:quota])
        if len(selected) < samples_per_task:
            selected_ids = {row["sample_id"] for row in selected}
            remaining = [
                row for name in ("0-4k", "4-8k", "8k+") for row in groups[name]
                if row["sample_id"] not in selected_ids
            ]
            remaining.sort(key=lambda row: deterministic_key(dataset, row["sample_id"], seed + 1))
            selected.extend(remaining[: samples_per_task - len(selected)])
        if len(selected) != samples_per_task:
            raise RuntimeError(f"could not select {samples_per_task} cases for {dataset}")
        output.extend(selected)
    return output


def make_prompt_ids(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> list[int]:
    dataset = str(row["dataset_name"])
    content = PROMPTS[dataset].format(**row)
    if dataset in NO_CHAT:
        ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_tokens:
        left = max_tokens // 2
        ids = ids[:left] + ids[-(max_tokens - left) :]
    return [int(value) for value in ids]


def metric_functions(code_root: Path) -> dict[str, Any]:
    del code_root

    def normalize_answer(value: str) -> str:
        value = value.lower()
        value = "".join(character for character in value if character not in set(string.punctuation))
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split())

    def qa_f1(prediction: str, ground_truth: str, **_: Any) -> float:
        predicted = normalize_answer(prediction).split()
        expected = normalize_answer(ground_truth).split()
        common = Counter(predicted) & Counter(expected)
        same = sum(common.values())
        if same == 0:
            return 0.0
        precision = same / len(predicted)
        recall = same / len(expected)
        return 2.0 * precision * recall / (precision + recall)

    def retrieval(prediction: str, ground_truth: str, **_: Any) -> float:
        matches = re.findall(r"Paragraph (\d+)", ground_truth)
        if not matches:
            return 0.0
        expected = matches[0]
        numbers = re.findall(r"\d+", prediction)
        return 0.0 if not numbers else sum(value == expected for value in numbers) / len(numbers)

    def code_similarity(prediction: str, ground_truth: str, **_: Any) -> float:
        from difflib import SequenceMatcher

        candidate = ""
        for line in prediction.lstrip("\n").split("\n"):
            if "`" not in line and "#" not in line and "//" not in line:
                candidate = line
                break
        return round(100.0 * SequenceMatcher(None, candidate, ground_truth).ratio()) / 100.0

    return {
        "qasper": qa_f1,
        "multifieldqa_en": qa_f1,
        "hotpotqa": qa_f1,
        "2wikimqa": qa_f1,
        "passage_retrieval_en": retrieval,
        "lcc": code_similarity,
    }


def official_score(row: dict[str, Any], prediction: str, metric: Any) -> float:
    scores = [
        float(metric(prediction, answer, all_classes=row.get("all_classes")))
        for answer in row["answers"]
    ]
    return max(scores) if scores else 0.0


def run_last(model: Any, prompt: torch.Tensor, cache: Any, prefix_length: int) -> tuple[torch.Tensor, Any]:
    output = longbench.cache_runner.forward_with_cache(
        model,
        prompt[:, -1:].to(longbench.cache_runner.input_device(model)),
        cache,
        prefix_length,
    )
    return output.logits[:, -1, :].float(), output.past_key_values


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    cache: Any,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    generated: list[int] = []
    stops = longbench.eos_ids(tokenizer)
    past_length = prompt_length
    for step in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in stops:
            break
        generated.append(token_id)
        if step + 1 == max_new_tokens:
            break
        token = torch.tensor(
            [[token_id]], dtype=torch.long, device=longbench.cache_runner.input_device(model)
        )
        output = longbench.cache_runner.forward_with_cache(model, token, cache, past_length)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        past_length += 1
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated


@torch.inference_mode()
def gold_nll(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
    answer: str,
) -> tuple[float, int]:
    ids = longbench.oracle.token_ids(tokenizer, answer)
    if not ids:
        return float("nan"), 0
    logits, cache = run_last(model, prompt, cache, prefix_length)
    losses = []
    past_length = int(prompt.shape[1])
    for index, token_id in enumerate(ids):
        losses.append(-float(F.log_softmax(logits, dim=-1)[0, token_id].item()))
        if index + 1 < len(ids):
            token = torch.tensor(
                [[token_id]], dtype=torch.long, device=longbench.cache_runner.input_device(model)
            )
            output = longbench.cache_runner.forward_with_cache(model, token, cache, past_length)
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            past_length += 1
    return mean(losses), len(ids)


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        dataset_scores = {
            dataset: 100.0 * mean(float(row["official_score"]) for row in selected if row["dataset"] == dataset)
            for dataset in sorted({str(row["dataset"]) for row in selected})
        }
        mean_nll = mean(float(row["gold_answer_mean_nll"]) for row in selected)
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "longbench_macro_score": mean(dataset_scores.values()),
                "dataset_scores": dataset_scores,
                "gold_answer_mean_nll": mean_nll,
                "gold_answer_ppl": math.exp(min(mean_nll, 30.0)),
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = sorted(set(datasets) - set(PROMPTS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    specs = read_specs(args.specs_json)
    cases = select_cases(
        args.longbench_dir, datasets, args.samples_per_task, args.selection_seed
    )
    pairs = [(spec, case) for spec in specs for case in cases]
    pairs = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    if not pairs:
        raise RuntimeError("this shard has no cases")
    metrics = metric_functions(args.longbench_code_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model, tokenizer = longbench.local_global.load_model(load_args)
    intervention = HeadFrequencyIntervention(model)
    write_json(
        args.output_dir / "config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "specs": specs,
            "selected_cases": [
                {
                    "dataset": row["dataset_name"],
                    "sample_id": row["sample_id"],
                    "length": row["length"],
                    "length_bin": length_bin(int(row["length"])),
                }
                for row in cases
            ],
            "pair_count": len(pairs),
            "weights_frozen": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )
    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {
        (str(row["variant"]), str(row["dataset"]), str(row["sample_id"])) for row in existing
    }
    for index, (spec, case) in enumerate(pairs, start=1):
        variant = str(spec["name"])
        dataset = str(case["dataset_name"])
        sample_id = str(case["sample_id"])
        if (variant, dataset, sample_id) in completed:
            continue
        prompt_ids = make_prompt_ids(tokenizer, case, args.max_prompt_tokens)
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prefix_length = len(prompt_ids) - 1
        with intervention.activate(spec):
            started = time.perf_counter()
            legacy, prefill_seconds = longbench.cache_runner.prefill_sequence(
                model, prompt[:, :-1], args.prefill_chunk_size
            )
            cache = longbench.cache_runner.cache_from_legacy(legacy)
            del legacy
            logits, cache = run_last(model, prompt, cache, prefix_length)
            prediction, generated_ids = generate(
                model,
                tokenizer,
                logits,
                cache,
                len(prompt_ids),
                MAX_NEW_TOKENS[dataset],
            )
            longbench.rope_repair.reset_dynamic_cache(cache, prefix_length)
            nll, answer_tokens = gold_nll(
                model, tokenizer, prompt, cache, prefix_length, str(case["answers"][0])
            )
            longbench.rope_repair.reset_dynamic_cache(cache, prefix_length)
            elapsed = time.perf_counter() - started
        row = {
            "variant": variant,
            "spec": spec,
            "dataset": dataset,
            "sample_id": sample_id,
            "length": int(case["length"]),
            "length_bin": length_bin(int(case["length"])),
            "prompt_tokens": len(prompt_ids),
            "answers": case["answers"],
            "prediction": prediction,
            "generated_token_ids": generated_ids,
            "official_score": official_score(case, prediction, metrics[dataset]),
            "gold_answer_mean_nll": nll,
            "gold_answer_ppl": math.exp(min(nll, 30.0)),
            "gold_answer_tokens": answer_tokens,
            "prefill_seconds": prefill_seconds,
            "elapsed_seconds": elapsed,
        }
        append_jsonl(rows_path, [row])
        completed.add((variant, dataset, sample_id))
        print(
            f"[{index}/{len(pairs)}] {variant} {dataset}/{sample_id[:8]} "
            f"score={row['official_score']:.3f} nll={nll:.3f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del cache, prompt
        longbench.local_global.clear_allocator()
    all_rows = read_jsonl(rows_path)
    write_csv(args.output_dir / "rows.csv", all_rows)
    summary = summarize(all_rows)
    write_csv(
        args.output_dir / "summary.csv",
        [
            {key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()}
            for row in summary
        ],
    )
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
