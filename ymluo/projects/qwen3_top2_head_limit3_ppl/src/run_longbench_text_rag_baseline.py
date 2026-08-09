from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from run_controlled_public_kv_benchmark_v1 import (
    LONG_BENCH_PROMPTS,
    make_pages,
    score_prediction,
    trim_context_to_tokens,
)


ALL_TASKS = ",".join(LONG_BENCH_PROMPTS)
ALL_METHODS = (
    "recent_1024",
    "bm25_1024",
    "e5_1024",
    "hybrid_rrf_1024",
    "hybrid_recent_1024",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ordinary text RAG on LongBench with official task metrics."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--embedding_model_name_or_path", default="intfloat/e5-base-v2")
    parser.add_argument("--longbench_zip_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default=ALL_TASKS)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--max_samples_per_task", type=int, default=20)
    parser.add_argument("--max_context_tokens", type=int, default=7500)
    parser.add_argument("--budget_tokens", type=int, default=1024)
    parser.add_argument("--page_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens_override", type=int, default=0)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--embedding_max_length", type=int, default=192)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--prompt_wrapper", choices=["none", "llama3"], default="llama3")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--load_stagger_seconds", type=float, default=0.0)
    parser.add_argument("--full_baseline_csv", default="")
    parser.add_argument("--kv_baseline_csv", default="")
    return parser.parse_args()


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("LongBench RAG generation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = parse_list(args.tasks)
    examples: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.longbench_zip_path) as archive:
        for task in tasks:
            if task not in LONG_BENCH_PROMPTS:
                raise ValueError(f"unsupported LongBench task: {task}")
            path = f"data/{task}.jsonl"
            rows = [
                json.loads(line)
                for line in archive.open(path).read().decode("utf-8").splitlines()
                if line.strip()
            ]
            rows = rows[: args.max_samples_per_task]
            rows.sort(key=lambda row: str(row.get("_id", "")))
            info = LONG_BENCH_PROMPTS[task]
            for row in rows:
                max_new_tokens = int(info["max_new_tokens"])
                if args.max_new_tokens_override > 0:
                    max_new_tokens = min(max_new_tokens, args.max_new_tokens_override)
                examples.append(
                    {
                        "task": task,
                        "sample_id": str(row.get("_id", len(examples))),
                        "context": str(row["context"]),
                        "query": str(row["input"]),
                        "answers": [str(item) for item in row["answers"]],
                        "all_classes": [str(item) for item in (row.get("all_classes") or [])],
                        "metric": str(info["metric"]),
                        "prefix_template": str(info["prefix"]),
                        "suffix_template": str(info["suffix"]),
                        "no_chat": bool(info.get("no_chat", False)),
                        "global_task": bool(info.get("global_task", False)),
                        "max_new_tokens": max_new_tokens,
                    }
                )
    return examples


class BM25Index:
    def __init__(self, documents: Sequence[str]) -> None:
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            dtype=np.float32,
        )
        self.empty = False
        try:
            counts = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        except ValueError:
            self.empty = True
            self.document_count = len(documents)
            return
        document_count = int(counts.shape[0])
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = max(float(lengths.mean()), 1.0e-6)
        row_ids = np.repeat(np.arange(document_count), np.diff(counts.indptr))
        frequencies = counts.data
        denominator = frequencies + 1.2 * (
            1.0 - 0.75 + 0.75 * lengths[row_ids] / average_length
        )
        counts.data = (
            inverse_document_frequency[counts.indices]
            * frequencies
            * 2.2
            / denominator
        )
        self.weighted_documents = counts
        self.document_count = document_count

    def score(self, query: str) -> np.ndarray:
        if self.empty:
            return np.zeros(self.document_count, dtype=np.float32)
        query_counts = self.vectorizer.transform([query]).tocsr().astype(np.float32)
        query_counts.data.fill(1.0)
        return np.asarray(
            (query_counts @ self.weighted_documents.transpose()).toarray()[0],
            dtype=np.float32,
        )


def stable_ranking(scores: np.ndarray) -> list[int]:
    ids = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((ids, -np.asarray(scores, dtype=np.float64))).tolist()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], rrf_k: float
) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, page_id_value in enumerate(ranking, start=1):
            page_id = int(page_id_value)
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[page_id] = min(best_rank.get(page_id, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))


@torch.inference_mode()
def encode_e5(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    prefix: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            [prefix + text for text in texts[start : start + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output.append(F.normalize(embeddings.float(), dim=1))
    return torch.cat(output, dim=0)


def page_token_counts(tokenizer: Any, page_texts: Sequence[str]) -> list[int]:
    return [
        len(tokenizer(text + "\n\n", add_special_tokens=False)["input_ids"])
        for text in page_texts
    ]


def take_ranked_pages(
    ranking: Sequence[int],
    counts: Sequence[int],
    budget: int,
    *,
    excluded: set[int] | None = None,
) -> list[int]:
    excluded = excluded or set()
    selected: list[int] = []
    used = 0
    for page_id_value in ranking:
        page_id = int(page_id_value)
        if page_id in excluded:
            continue
        count = int(counts[page_id])
        if used + count > budget:
            continue
        selected.append(page_id)
        used += count
        if used >= budget:
            break
    return selected


def selected_context(
    tokenizer: Any,
    page_texts: Sequence[str],
    page_ids: Sequence[int],
    budget_tokens: int,
) -> tuple[str, int]:
    text = "\n\n".join(page_texts[item] for item in sorted(set(page_ids)))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > budget_tokens:
        ids = ids[:budget_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, len(ids)


def retrieval_query(example: dict[str, Any]) -> str:
    query = str(example["query"]).strip()
    if query:
        return query
    return str(example["suffix_template"]).format(input="").strip()


def build_prompt_ids(
    tokenizer: Any,
    example: dict[str, Any],
    context: str,
    prompt_wrapper: str,
) -> list[int]:
    prefix = str(example["prefix_template"])
    suffix = str(example["suffix_template"]).format(input=example["query"])
    if prompt_wrapper == "llama3" and not bool(example["no_chat"]):
        prefix = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prefix
        )
        suffix += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return (
        tokenizer(prefix, add_special_tokens=False)["input_ids"]
        + tokenizer(context, add_special_tokens=False)["input_ids"]
        + tokenizer(suffix, add_special_tokens=False)["input_ids"]
    )


@torch.inference_mode()
def greedy_generate(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int], float, float]:
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    synchronize(device)
    started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
        cache_position=torch.arange(input_ids.shape[1], device=device),
    )
    synchronize(device)
    prefill_seconds = time.perf_counter() - started
    cache = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    generated: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    synchronize(device)
    started = time.perf_counter()
    for step in range(max_new_tokens):
        next_id = int(torch.argmax(logits.float(), dim=-1).item())
        if eos_token_id is not None and next_id == int(eos_token_id):
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor(
                [len(prompt_ids) + step], dtype=torch.long, device=device
            ),
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
    synchronize(device)
    decode_seconds = time.perf_counter() - started
    return (
        tokenizer.decode(generated, skip_special_tokens=True),
        generated,
        prefill_seconds,
        decode_seconds,
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        for task in ["ALL"] + sorted({str(row["task"]) for row in rows}):
            group = [
                row
                for row in rows
                if str(row["method"]) == method
                and (task == "ALL" or str(row["task"]) == task)
            ]
            if not group:
                continue
            count = len(group)
            output.append(
                {
                    "task": task,
                    "method": method,
                    "samples": count,
                    "score": sum(float(row["score"]) for row in group) / count,
                    "mean_total_seconds": sum(float(row["total_seconds"]) for row in group)
                    / count,
                    "mean_retrieval_seconds": sum(
                        float(row["retrieval_seconds"]) for row in group
                    )
                    / count,
                    "mean_prefill_seconds": sum(float(row["prefill_seconds"]) for row in group)
                    / count,
                    "mean_decode_seconds": sum(float(row["decode_seconds"]) for row in group)
                    / count,
                    "mean_raw_context_tokens": sum(
                        int(row["raw_context_tokens"]) for row in group
                    )
                    / count,
                    "mean_retrieved_context_tokens": sum(
                        int(row["retrieved_context_tokens"]) for row in group
                    )
                    / count,
                    "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in group)
                    / count,
                }
            )
    return output


def baseline_summary(
    path: str,
    examples: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    if not path:
        return None
    expected = {(str(item["task"]), str(item["sample_id"])) for item in examples}
    rows = [
        row
        for row in read_csv(Path(path))
        if (str(row["task"]), str(row["sample_id"])) in expected
    ]
    matched = {(str(row["task"]), str(row["sample_id"])) for row in rows}
    if not matched:
        raise ValueError(f"{label} baseline does not overlap selected samples")
    return {
        "label": label,
        "samples": len(rows),
        "selected_samples": len(expected),
        "coverage": len(matched) / len(expected),
        "complete_coverage": matched == expected,
        "score": sum(float(row["score"]) for row in rows) / len(rows),
        "mean_total_seconds": sum(float(row["total_seconds"]) for row in rows)
        / len(rows),
        "mean_online_seconds": sum(float(row["online_seconds"]) for row in rows)
        / len(rows),
        "mean_context_tokens": sum(float(row["kept_context_tokens"]) for row in rows)
        / len(rows),
        "task_scores": {
            task: sum(float(row["score"]) for row in rows if row["task"] == task)
            / sum(1 for row in rows if row["task"] == task)
            for task in sorted({str(row["task"]) for row in rows})
        },
    }


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    methods = parse_list(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if min(
        args.max_samples_per_task,
        args.max_context_tokens,
        args.budget_tokens,
        args.page_tokens,
    ) <= 0:
        raise ValueError("sample and token counts must be positive")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    barrier(world_size)
    examples = load_examples(args)
    local_examples = [item for index, item in enumerate(examples) if index % world_size == rank]

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    if world_size > 1 and args.load_stagger_seconds > 0:
        time.sleep(local_rank * args.load_stagger_seconds)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True

    needs_e5 = any(method in {"e5_1024", "hybrid_rrf_1024", "hybrid_recent_1024"} for method in methods)
    embedding_tokenizer = None
    embedding_model = None
    if needs_e5:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            args.embedding_model_name_or_path, use_fast=True
        )
        embedding_model = AutoModel.from_pretrained(
            args.embedding_model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        embedding_model.eval()

    barrier(world_size)

    local_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for local_index, example in enumerate(local_examples, start=1):
        context = trim_context_to_tokens(
            tokenizer, str(example["context"]), args.max_context_tokens
        )
        raw_context_tokens = len(
            tokenizer(context, add_special_tokens=False)["input_ids"]
        )
        segmentation_started = time.perf_counter()
        pages = make_pages(tokenizer, context, 0, args.page_tokens)
        page_texts = [str(page.text) for page in pages]
        counts = page_token_counts(tokenizer, page_texts)
        segmentation_seconds = time.perf_counter() - segmentation_started
        query = retrieval_query(example)

        bm25_started = time.perf_counter()
        bm25 = BM25Index(page_texts)
        bm25_scores = bm25.score(query)
        bm25_ranking = stable_ranking(bm25_scores)
        bm25_core_seconds = time.perf_counter() - bm25_started
        bm25_seconds = segmentation_seconds + bm25_core_seconds

        e5_ranking: list[int] = []
        e5_seconds = 0.0
        if embedding_model is not None and embedding_tokenizer is not None:
            synchronize(device)
            e5_started = time.perf_counter()
            passage_embeddings = encode_e5(
                embedding_model,
                embedding_tokenizer,
                page_texts,
                prefix="passage: ",
                batch_size=args.embedding_batch_size,
                max_length=args.embedding_max_length,
                device=device,
            )
            query_embedding = encode_e5(
                embedding_model,
                embedding_tokenizer,
                [query],
                prefix="query: ",
                batch_size=1,
                max_length=args.embedding_max_length,
                device=device,
            )
            e5_scores = (query_embedding @ passage_embeddings.transpose(0, 1))[0]
            synchronize(device)
            e5_core_seconds = time.perf_counter() - e5_started
            e5_seconds = segmentation_seconds + e5_core_seconds
            e5_ranking = stable_ranking(e5_scores.float().cpu().numpy())
            del passage_embeddings, query_embedding, e5_scores

        hybrid_ranking = (
            reciprocal_rank_fusion([bm25_ranking, e5_ranking], args.rrf_k)
            if e5_ranking
            else bm25_ranking
        )
        recent_ranking = list(reversed(range(len(pages))))
        selections: dict[str, list[int]] = {}
        retrieval_seconds: dict[str, float] = {}
        if "recent_1024" in methods:
            selections["recent_1024"] = take_ranked_pages(
                recent_ranking, counts, args.budget_tokens
            )
            retrieval_seconds["recent_1024"] = segmentation_seconds
        if "bm25_1024" in methods:
            selections["bm25_1024"] = take_ranked_pages(
                bm25_ranking, counts, args.budget_tokens
            )
            retrieval_seconds["bm25_1024"] = bm25_seconds
        if "e5_1024" in methods:
            selections["e5_1024"] = take_ranked_pages(
                e5_ranking, counts, args.budget_tokens
            )
            retrieval_seconds["e5_1024"] = e5_seconds
        if "hybrid_rrf_1024" in methods:
            selections["hybrid_rrf_1024"] = take_ranked_pages(
                hybrid_ranking, counts, args.budget_tokens
            )
            retrieval_seconds["hybrid_rrf_1024"] = (
                segmentation_seconds + bm25_core_seconds + e5_core_seconds
            )
        if "hybrid_recent_1024" in methods:
            recent = take_ranked_pages(
                recent_ranking, counts, args.budget_tokens // 2
            )
            remote = take_ranked_pages(
                hybrid_ranking,
                counts,
                args.budget_tokens - sum(counts[item] for item in recent),
                excluded=set(recent),
            )
            selections["hybrid_recent_1024"] = recent + remote
            retrieval_seconds["hybrid_recent_1024"] = (
                segmentation_seconds + bm25_core_seconds + e5_core_seconds
            )

        for method in methods:
            selected_text, retrieved_tokens = selected_context(
                tokenizer,
                page_texts,
                selections[method],
                args.budget_tokens,
            )
            prompt_ids = build_prompt_ids(
                tokenizer, example, selected_text, args.prompt_wrapper
            )
            prediction, generated_ids, prefill_seconds, decode_seconds = greedy_generate(
                model,
                tokenizer,
                prompt_ids,
                int(example["max_new_tokens"]),
                device,
            )
            score = score_prediction(
                str(example["metric"]),
                prediction,
                list(example["answers"]),
                list(example["all_classes"]),
                task=str(example["task"]),
            )
            total_seconds = (
                retrieval_seconds[method] + prefill_seconds + decode_seconds
            )
            local_rows.append(
                {
                    "benchmark": "longbench",
                    "task": str(example["task"]),
                    "sample_id": str(example["sample_id"]),
                    "method": method,
                    "metric": str(example["metric"]),
                    "score": score,
                    "prediction": prediction.replace("\n", "\\n")[:500],
                    "answers": json.dumps(example["answers"], ensure_ascii=False),
                    "generated_tokens": len(generated_ids),
                    "retrieval_seconds": retrieval_seconds[method],
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                    "total_seconds": total_seconds,
                    "raw_context_tokens": raw_context_tokens,
                    "retrieved_context_tokens": retrieved_tokens,
                    "prompt_tokens": len(prompt_ids),
                    "selected_pages": json.dumps(sorted(selections[method])),
                    "page_count": len(pages),
                    "budget_tokens": args.budget_tokens,
                    "page_tokens": args.page_tokens,
                    "selection_uses_answers": False,
                }
            )
        if local_index % args.log_every == 0 or local_index == len(local_examples):
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed": local_index,
                        "local_total": len(local_examples),
                        "task": example["task"],
                        "sample_id": example["sample_id"],
                    }
                ),
                flush=True,
            )

    write_csv(output_dir / f"rows_rank{rank:03d}.csv", local_rows)
    rank_metadata = {
        "rank": rank,
        "samples": len(local_examples),
        "method_rows": len(local_rows),
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    (output_dir / f"metadata_rank{rank:03d}.json").write_text(
        json.dumps(rank_metadata, indent=2), encoding="utf-8"
    )
    barrier(world_size)

    if rank == 0:
        rows = [
            row
            for shard in range(world_size)
            for row in read_csv(output_dir / f"rows_rank{shard:03d}.csv")
        ]
        rows.sort(key=lambda row: (str(row["task"]), str(row["sample_id"]), str(row["method"])))
        write_csv(output_dir / "task_results.csv", rows)
        summary = summarize(rows)
        write_csv(output_dir / "summary.csv", summary)
        baselines = [
            item
            for item in (
                baseline_summary(args.full_baseline_csv, examples, "full_kv"),
                baseline_summary(args.kv_baseline_csv, examples, "kv_baseline"),
            )
            if item is not None
        ]
        payload = {
            "source": "ordinary text BM25/E5 RAG on official LongBench",
            "model": args.model_name_or_path,
            "embedding_model": args.embedding_model_name_or_path,
            "samples": len(examples),
            "tasks": parse_list(args.tasks),
            "methods": methods,
            "budget_tokens": args.budget_tokens,
            "page_tokens": args.page_tokens,
            "max_context_tokens": args.max_context_tokens,
            "world_size": world_size,
            "selection_uses_answers": False,
            "summary": summary,
            "cached_matched_baselines": baselines,
            "rank_metadata": [
                json.loads(
                    (output_dir / f"metadata_rank{shard:03d}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for shard in range(world_size)
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
