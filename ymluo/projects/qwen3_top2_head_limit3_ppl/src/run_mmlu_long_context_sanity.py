from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    prefill_cache,
    resolve_dtype,
)
from run_qabs_downstream_kv_retrieval import LABELS  # noqa: E402
from run_vertical_memory_v1_downstream import MemoryPage, detect_status, extract_entities, extract_keywords, idf_for_pages  # noqa: E402


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    data_dir: str
    output_dir: str
    subjects: str
    split: str
    max_per_subject: int
    fewshot: int
    distractor_pages: int
    seed: int
    chunk_size: int
    dtype: str
    device: str
    device_map: str
    attn_implementation: str


DISTRACTOR_TOPICS = [
    "astronomy telescope calibration and spectral drift",
    "clinical trial dosage response and patient monitoring",
    "liquidity buffer risk exposure and portfolio review",
    "contract clause appeal deadline and court filing",
    "river survey habitat recovery and species count",
    "robotics navigation stack sensor fusion and field test",
    "public relations crisis planning and media response",
    "ancient history trade routes and political legitimacy",
]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MMLU long-context sanity test for page routing.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_dir", default="/home/fdong/zx_workspace/moe-llava/moellava/eval/mmlu_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--subjects",
        default="computer_security,high_school_geography,philosophy,public_relations,high_school_statistics",
    )
    parser.add_argument("--split", choices=["dev", "test", "val"], default="test")
    parser.add_argument("--max_per_subject", type=int, default=20)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--distractor_pages", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026070202)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    return Config(**vars(parser.parse_args()))


@torch.inference_mode()
def run_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    text: str,
) -> tuple[Any, torch.Tensor]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return past_key_values, prev_logits


@torch.inference_mode()
def score_label(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    label: str,
) -> tuple[float, int]:
    ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    loss = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        loss += float(F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
        outputs = model_forward(
            model,
            {
                "input_ids": token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
            },
        )
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
    return loss, int(ids.shape[-1])


def read_subject(data_dir: Path, subject: str, split: str) -> list[dict[str, str]]:
    path = data_dir / split / f"{subject}_{split}.csv"
    if not path.exists() and split == "val":
        path = data_dir / "validation" / f"{subject}_val.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            rows.append(
                {
                    "question": row[0],
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                    "answer": row[5].strip().upper(),
                }
            )
    return rows


def format_mmlu_question(subject: str, row: dict[str, str]) -> str:
    subject_text = subject.replace("_", " ")
    return (
        f"The following is a multiple choice question about {subject_text}.\n"
        f"Question: {row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
        "Answer:"
    )


def format_fewshot(subject: str, examples: list[dict[str, str]]) -> str:
    parts = []
    for row in examples:
        parts.append(format_mmlu_question(subject, row) + f" {row['answer']}\n")
    return "\n".join(parts)


def distractor_context(rng: random.Random, pages: int) -> str:
    out = []
    for idx in range(pages):
        topic = rng.choice(DISTRACTOR_TOPICS)
        status = "Background page" if idx % 5 else "Current reference page"
        out.append(
            f"{status} {idx}. This page discusses {topic}. "
            f"It contains definitions, examples, and caveats for archival review. "
            f"It is not written as an answer key for the later multiple choice question."
        )
    return "\n\n".join(out) + "\n\n"


def build_pages(context: str) -> list[MemoryPage]:
    pages = []
    for idx, text in enumerate(block.strip() for block in re.split(r"\n\s*\n+", context) if block.strip()):
        pages.append(
            MemoryPage(
                page_id=idx,
                text=text,
                entities=extract_entities(text),
                keywords=extract_keywords(text),
                label="",
                status=detect_status(text),
                structural_score=text.count(".") + text.count(":"),
            )
        )
    return pages


def routed_context(context: str, query: str, max_pages: int = 1) -> tuple[str, dict[str, Any]]:
    pages = build_pages(context)
    idf = idf_for_pages(pages)
    query_entities = extract_entities(query)
    query_keywords = extract_keywords(query)
    scored = []
    for page in pages:
        score = 12.0 * len(page.entities & query_entities)
        score += sum(min(count, page.keywords.get(word, 0)) * idf.get(word, 1.0) for word, count in query_keywords.items())
        if page.status == "current":
            score += 0.5
        scored.append((score, page))
    scored.sort(key=lambda item: (item[0], item[1].page_id), reverse=True)
    selected = [page for _, page in scored[:max_pages]]
    text = "\nRouted background page:\n"
    for page in selected:
        text += re.sub(r"\s+", " ", page.text).strip() + "\n"
    return text, {
        "routed_pages": len(selected),
        "page_count": len(pages),
        "top_page_score": scored[0][0] if scored else 0.0,
    }


def evaluate_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    context_cache: Any,
    context_prev: torch.Tensor,
    prompt: str,
    answer: str,
) -> tuple[str, dict[str, float], float, float, int, float]:
    started = time.perf_counter()
    query_cache, query_prev = run_prefix(
        model,
        tokenizer,
        input_device,
        clone_past_key_values(context_cache) if context_cache is not None else None,
        context_prev.detach().clone() if context_prev.numel() else context_prev,
        prompt,
    )
    scores = {}
    gold_loss = 0.0
    gold_tokens = 0
    for label in LABELS:
        loss, tokens = score_label(
            model,
            tokenizer,
            input_device,
            clone_past_key_values(query_cache),
            query_prev.detach().clone(),
            label,
        )
        scores[label] = -loss
        if label == answer:
            gold_loss = loss
            gold_tokens = tokens
    elapsed = time.perf_counter() - started
    mean_loss = gold_loss / max(1, gold_tokens)
    return max(scores, key=scores.get), scores, elapsed, mean_loss, gold_tokens, math.exp(min(mean_loss, 80.0))


def empty_cache(input_device: torch.device) -> tuple[Any, torch.Tensor]:
    return None, torch.empty((1, 0), device=input_device)


def append_row(rows: list[dict[str, Any]], base: dict[str, Any], mode: str, pred: str, scores: dict[str, float], elapsed: float, prefill: float, tokens: int, loss: float, loss_tokens: int, ppl: float, meta: dict[str, Any]) -> None:
    rows.append(
        {
            **base,
            "mode": mode,
            "pred": pred,
            "correct": int(pred == base["answer"]),
            "eval_seconds": elapsed,
            "prefill_seconds": prefill,
            "total_seconds": elapsed + prefill,
            "visible_tokens": tokens,
            "gold_label_loss": loss,
            "gold_label_tokens": loss_tokens,
            "gold_label_ppl": ppl,
            "routed_pages": int(meta.get("routed_pages", 0)),
            "page_count": int(meta.get("page_count", 0)),
            "top_page_score": float(meta.get("top_page_score", 0.0)),
            **{f"score_{label}": scores[label] for label in LABELS},
        }
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["mode"])].append(row)
        grouped[("ALL", row["mode"])].append(row)
    out = []
    for (subject, mode), subset in sorted(grouped.items()):
        total_tokens = sum(int(row["gold_label_tokens"]) for row in subset)
        total_loss = sum(float(row["gold_label_loss"]) * int(row["gold_label_tokens"]) for row in subset)
        mean_loss = total_loss / max(1, total_tokens)
        n = max(1, len(subset))
        out.append(
            {
                "subject": subject,
                "mode": mode,
                "tasks": len(subset),
                "accuracy": sum(int(row["correct"]) for row in subset) / n,
                "gold_label_loss": mean_loss,
                "gold_label_ppl": math.exp(min(mean_loss, 80.0)),
                "mean_eval_seconds": sum(float(row["eval_seconds"]) for row in subset) / n,
                "mean_total_seconds": sum(float(row["total_seconds"]) for row in subset) / n,
                "mean_visible_tokens": sum(int(row["visible_tokens"]) for row in subset) / n,
                "mean_routed_pages": sum(int(row["routed_pages"]) for row in subset) / n,
            }
        )
    return out


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = torch.device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, device)

    rng = random.Random(config.seed)
    data_dir = Path(config.data_dir)
    rows: list[dict[str, Any]] = []
    for subject in [part.strip() for part in config.subjects.split(",") if part.strip()]:
        examples = read_subject(data_dir, subject, config.split)
        rng.shuffle(examples)
        examples = examples[: config.max_per_subject]
        dev_examples = read_subject(data_dir, subject, "dev")[: config.fewshot] if config.fewshot else []
        fewshot = format_fewshot(subject, dev_examples)
        for idx, row in enumerate(examples):
            print(f"{subject} {idx + 1}/{len(examples)}", flush=True)
            query = fewshot + format_mmlu_question(subject, row)
            noise = distractor_context(rng, config.distractor_pages)
            base = {
                "subject": subject,
                "task_id": idx,
                "answer": row["answer"],
                "question": row["question"],
            }

            cache, prev = empty_cache(input_device)
            prompt_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False)["input_ids"]
            pred, scores, elapsed, loss, loss_tokens, ppl = evaluate_prompt(
                model, tokenizer, input_device, cache, prev, query, row["answer"]
            )
            append_row(rows, base, "mmlu_direct", pred, scores, elapsed, 0.0, int(prompt_ids.shape[-1]), loss, loss_tokens, ppl, {})

            noise_ids = tokenizer(noise, return_tensors="pt", add_special_tokens=False)["input_ids"]
            started = time.perf_counter()
            noise_cache, noise_prev = prefill_cache(model, noise_ids, noise_ids.shape[-1], config.chunk_size, input_device)
            prefill = time.perf_counter() - started
            pred, scores, elapsed, loss, loss_tokens, ppl = evaluate_prompt(
                model, tokenizer, input_device, noise_cache, noise_prev, query, row["answer"]
            )
            append_row(rows, base, "mmlu_long_full_noise", pred, scores, elapsed, prefill, int(noise_ids.shape[-1]) + int(prompt_ids.shape[-1]), loss, loss_tokens, ppl, {})

            route_text, meta = routed_context(noise, query, max_pages=1)
            routed_prompt = route_text + query
            routed_ids = tokenizer(routed_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            cache, prev = empty_cache(input_device)
            pred, scores, elapsed, loss, loss_tokens, ppl = evaluate_prompt(
                model, tokenizer, input_device, cache, prev, routed_prompt, row["answer"]
            )
            append_row(rows, base, "mmlu_routed_noise_top1", pred, scores, elapsed, 0.0, int(routed_ids.shape[-1]), loss, loss_tokens, ppl, meta)

            oracle_prompt = f"\nTyped memory summary: ANSWER_LABEL={row['answer']}; status=current.\n" + query
            oracle_ids = tokenizer(oracle_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            cache, prev = empty_cache(input_device)
            pred, scores, elapsed, loss, loss_tokens, ppl = evaluate_prompt(
                model, tokenizer, input_device, cache, prev, oracle_prompt, row["answer"]
            )
            append_row(rows, base, "oracle_answerline_upper_bound", pred, scores, elapsed, 0.0, int(oracle_ids.shape[-1]), loss, loss_tokens, ppl, {})

    fields = [
        "subject",
        "task_id",
        "mode",
        "answer",
        "pred",
        "correct",
        "eval_seconds",
        "prefill_seconds",
        "total_seconds",
        "visible_tokens",
        "gold_label_loss",
        "gold_label_tokens",
        "gold_label_ppl",
        "routed_pages",
        "page_count",
        "top_page_score",
        "question",
    ] + [f"score_{label}" for label in LABELS]
    with (output_dir / "mmlu_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = summarize(rows)
    with (output_dir / "mmlu_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary_rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
