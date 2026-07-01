from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_hierarchical_book_index_recall import SparseTfidfIndex, joined, selected_page_set  # noqa: E402
from analyze_longrange_book_index_semantic_retrieval import (  # noqa: E402
    LABELS,
    GeneratedTask,
    authority_score,
    build_indexes,
    build_task,
    overlap_page,
)
from analyze_typed_anchor_page_recall import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
    str2bool,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-pruned downstream QA eval for long-range hierarchical book-index routing."
    )
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_tokens", default="10000,20000")
    parser.add_argument("--tasks_per_length", type=int, default=4)
    parser.add_argument("--eval_tokens", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--paragraph_min_tokens", type=int, default=64)
    parser.add_argument("--paragraph_max_tokens", type=int, default=192)
    parser.add_argument("--section_max_paragraphs", type=int, default=8)
    parser.add_argument("--query_window_tokens", type=int, default=256)
    parser.add_argument(
        "--schemes",
        default=(
            "full_context,recent_only,sink_recent,remote_tail_p4,remote_tail_p8,remote_tail_p16,"
            "book_flat_p4,book_flat_p8,book_auth_flat_p4,book_auth_flat_p8,book_auth_flat_p16,"
            "book_hier_s4_p2,book_auth_hier_s4_p2,hybrid_tail4_authflat4,hybrid_tail4_authhier_s4_p2"
        ),
    )
    parser.add_argument("--add_page_markers", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2026070101)
    return parser.parse_args()


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


def page_text(task: GeneratedTask, page_id: int, pages: list[Any], add_marker: bool) -> str:
    page = pages[page_id]
    text = joined(task.token_texts, page.start, page.end)
    if not add_marker:
        return text
    return f"\n[PAGE {page_id} tokens={page.start}-{page.end}]\n{text}\n[/PAGE {page_id}]\n"


def auth_flat_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    page_count: int,
) -> set[int]:
    query_vec = index.query_vector(query_text)
    scored = []
    for page_id in candidate_ids:
        score = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id])
        score += authority_score(joined(task.token_texts, pages[page_id].start, pages[page_id].end))
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return {page_id for page_id, _ in scored[:page_count]}


def auth_hier_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    section_count: int,
    pages_per_section: int,
) -> set[int]:
    page_query = page_index.query_vector(query_text)
    section_query = section_index.query_vector(query_text)
    section_scored = []
    for section_id in candidate_sections:
        score = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        score += authority_score(joined(task.token_texts, sections[section_id].start, sections[section_id].end))
        section_scored.append((section_id, score))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    candidate_set = set(candidate_pages)
    selected: set[int] = set()
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            score = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            score += authority_score(joined(task.token_texts, pages[page_id].start, pages[page_id].end))
            page_scored.append((page_id, score))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        selected.update(page_id for page_id, _ in page_scored[:pages_per_section])
    return selected


def selected_pages_for_scheme(
    scheme: str,
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    sink_tokens: int,
    recent_tokens: int,
    query_window_tokens: int,
) -> set[int]:
    remote_end = max(0, task.prefill_tokens - recent_tokens)
    candidate_pages = [page.unit_id for page in pages if page.end > sink_tokens and page.start < remote_end]
    candidate_sections = [section.unit_id for section in sections if section.end > sink_tokens and section.start < remote_end]
    query_start = max(0, task.query_start - query_window_tokens)
    query_text = joined(task.token_texts, query_start, task.query_start) + "\n" + task.query_text

    if scheme.startswith("hybrid_tail4_authflat"):
        count = int(scheme.removeprefix("hybrid_tail4_authflat"))
        tail = set(sorted(candidate_pages, key=lambda page_id: pages[page_id].end)[-4:])
        return tail | auth_flat_pages(task, pages, page_index, query_text, candidate_pages, count)
    match = re.fullmatch(r"hybrid_tail4_authhier_s(\d+)_p(\d+)", scheme)
    if match:
        tail = set(sorted(candidate_pages, key=lambda page_id: pages[page_id].end)[-4:])
        return tail | auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
    if scheme.startswith("book_auth_flat_p"):
        count = int(scheme.removeprefix("book_auth_flat_p"))
        return auth_flat_pages(task, pages, page_index, query_text, candidate_pages, count)
    match = re.fullmatch(r"book_auth_hier_s(\d+)_p(\d+)", scheme)
    if match:
        return auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
    selected, _ = selected_page_set(
        scheme,
        task.query_start,
        task.token_texts,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        sink_tokens,
        remote_end,
        query_window_tokens,
    )
    return selected


def build_prompt(
    tokenizer: Any,
    task: GeneratedTask,
    scheme: str,
    selected_pages: set[int],
    pages: list[Any],
    sink_tokens: int,
    recent_tokens: int,
    add_page_markers: bool,
) -> tuple[str, int, bool, bool]:
    evidence_page = overlap_page(pages, task.evidence_span)
    decoy_page = overlap_page(pages, task.decoy_span)
    evidence_hit = False
    decoy_hit = False
    if scheme == "full_context":
        context = decode_tokens(tokenizer, task.token_ids[: task.prefill_tokens])
        evidence_hit = True
        decoy_hit = True
    elif scheme == "recent_only":
        start = max(0, task.prefill_tokens - recent_tokens)
        context = decode_tokens(tokenizer, task.token_ids[start : task.prefill_tokens])
        evidence_hit = task.evidence_span.start >= start
        decoy_hit = task.decoy_span.start >= start
    elif scheme == "sink_recent":
        sink = decode_tokens(tokenizer, task.token_ids[: min(sink_tokens, task.prefill_tokens)])
        start = max(0, task.prefill_tokens - recent_tokens)
        recent = decode_tokens(tokenizer, task.token_ids[start : task.prefill_tokens])
        context = sink + "\n[... middle context omitted ...]\n" + recent
        evidence_hit = task.evidence_span.start < sink_tokens or task.evidence_span.start >= start
        decoy_hit = task.decoy_span.start < sink_tokens or task.decoy_span.start >= start
    else:
        sink = decode_tokens(tokenizer, task.token_ids[: min(sink_tokens, task.prefill_tokens)])
        start = max(0, task.prefill_tokens - recent_tokens)
        recent = decode_tokens(tokenizer, task.token_ids[start : task.prefill_tokens])
        page_parts = [page_text(task, page_id, pages, add_page_markers) for page_id in sorted(selected_pages)]
        context = sink + "\n[SELECTED REMOTE PAGES]\n" + "".join(page_parts)
        context += "\n[RECENT CONTEXT]\n" + recent
        evidence_hit = evidence_page in selected_pages if evidence_page is not None else False
        decoy_hit = decoy_page in selected_pages if decoy_page is not None else False
    prompt = context + "\n" + task.query_text
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, prompt_tokens, evidence_hit, decoy_hit


@torch.inference_mode()
def prefill_prompt(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
    past_key_values = None
    prev_logits: torch.Tensor | None = None
    total = input_ids.shape[-1]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
        del outputs
    if prev_logits is None:
        raise ValueError("Cannot score an empty prompt.")
    return past_key_values, prev_logits


@torch.inference_mode()
def score_option(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
) -> tuple[float, int]:
    ids = tokenizer(" " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
    total = 0.0
    for pos in range(ids.shape[-1]):
        token = ids[:, pos : pos + 1]
        total += float(-F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
        kwargs = {
            "input_ids": token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        prev_logits = outputs.logits[:, -1, :].detach()
        del outputs
    return total, int(ids.numel())


def clone_past(past_key_values: Any) -> Any:
    if hasattr(past_key_values, "to_legacy_cache"):
        return copy.deepcopy(past_key_values)
    return tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in past_key_values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (int(row["context_tokens"]), str(row["scheme"]))
        group = grouped[key]
        group["tasks"] += 1
        for field in [
            "correct",
            "calibrated_correct",
            "decoy_pred",
            "calibrated_decoy_pred",
            "evidence_hit",
            "decoy_hit",
            "prompt_tokens",
            "full_context_tokens",
            "selected_pages",
            "correct_label_nll",
            "margin_true_minus_decoy",
            "calibrated_margin_true_minus_decoy",
        ]:
            group[field] += float(row[field])
    out = []
    for (context_tokens, scheme), group in sorted(grouped.items()):
        tasks = group["tasks"]
        out.append(
            {
                "context_tokens": context_tokens,
                "scheme": scheme,
                "tasks": int(tasks),
                "accuracy": group["correct"] / tasks if tasks else 0.0,
                "calibrated_accuracy": group["calibrated_correct"] / tasks if tasks else 0.0,
                "decoy_pred_rate": group["decoy_pred"] / tasks if tasks else 0.0,
                "calibrated_decoy_pred_rate": group["calibrated_decoy_pred"] / tasks if tasks else 0.0,
                "evidence_hit_rate": group["evidence_hit"] / tasks if tasks else 0.0,
                "decoy_hit_rate": group["decoy_hit"] / tasks if tasks else 0.0,
                "mean_prompt_tokens": group["prompt_tokens"] / tasks if tasks else 0.0,
                "mean_selected_pages": group["selected_pages"] / tasks if tasks else 0.0,
                "token_ratio": group["prompt_tokens"] / group["full_context_tokens"]
                if group["full_context_tokens"]
                else 0.0,
                "mean_correct_label_nll": group["correct_label_nll"] / tasks if tasks else 0.0,
                "mean_margin_true_minus_decoy": group["margin_true_minus_decoy"] / tasks if tasks else 0.0,
                "mean_calibrated_margin_true_minus_decoy": (
                    group["calibrated_margin_true_minus_decoy"] / tasks if tasks else 0.0
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = [int(part) for part in args.context_tokens.split(",") if part.strip()]
    schemes = [part.strip() for part in args.schemes.split(",") if part.strip()]
    rng = random.Random(args.seed)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    for context_tokens in context_lengths:
        for task_index in range(args.tasks_per_length):
            task_id = context_tokens * 100 + task_index
            task = build_task(tokenizer, rng, task_id, context_tokens, args.eval_tokens)
            pages, _, page_index, sections, section_index, section_to_pages = build_indexes(task, args)
            evidence_page = overlap_page(pages, task.evidence_span)
            decoy_page = overlap_page(pages, task.decoy_span)
            full_context_tokens = task.prefill_tokens + len(tokenizer(task.query_text, add_special_tokens=False)["input_ids"])
            manifest.append(
                {
                    "context_tokens": context_tokens,
                    "task_id": task.task_id,
                    "target_key": task.target_key,
                    "target_label": task.target_label,
                    "decoy_label": task.decoy_label,
                    "evidence_page": evidence_page,
                    "decoy_page": decoy_page,
                    "paragraph_count": len(pages),
                    "section_count": len(sections),
                }
            )
            print(
                f"context={context_tokens} task={task_index + 1}/{args.tasks_per_length} "
                f"target={task.target_label} decoy={task.decoy_label} evidence_page={evidence_page} decoy_page={decoy_page}",
                flush=True,
            )
            prior_ids = tokenizer(task.query_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            prior_past, prior_prev_logits = prefill_prompt(model, prior_ids, args.chunk_size, input_device)
            prior_scores: dict[str, float] = {}
            for label in LABELS:
                score, _ = score_option(
                    model,
                    tokenizer,
                    input_device,
                    clone_past(prior_past),
                    prior_prev_logits.detach().clone(),
                    label,
                )
                prior_scores[label] = score
            del prior_past, prior_prev_logits, prior_ids
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
            for scheme in schemes:
                if scheme in {"full_context", "recent_only", "sink_recent"}:
                    selected_pages: set[int] = set()
                else:
                    selected_pages = selected_pages_for_scheme(
                        scheme,
                        task,
                        pages,
                        page_index,
                        sections,
                        section_index,
                        section_to_pages,
                        args.sink_tokens,
                        args.recent_tokens,
                        args.query_window_tokens,
                    )
                prompt, prompt_tokens, evidence_hit, decoy_hit = build_prompt(
                    tokenizer,
                    task,
                    scheme,
                    selected_pages,
                    pages,
                    args.sink_tokens,
                    args.recent_tokens,
                    args.add_page_markers,
                )
                prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                past, prev_logits = prefill_prompt(model, prompt_ids, args.chunk_size, input_device)
                scores: dict[str, float] = {}
                lengths: dict[str, int] = {}
                for label in LABELS:
                    score, length = score_option(
                        model,
                        tokenizer,
                        input_device,
                        clone_past(past),
                        prev_logits.detach().clone(),
                        label,
                    )
                    scores[label] = score
                    lengths[label] = length
                pred = max(scores, key=scores.get)
                calibrated_scores = {label: scores[label] - prior_scores[label] for label in LABELS}
                calibrated_pred = max(calibrated_scores, key=calibrated_scores.get)
                true_score = scores[task.target_label]
                decoy_score = scores[task.decoy_label]
                rows.append(
                    {
                        "context_tokens": context_tokens,
                        "task_id": task.task_id,
                        "scheme": scheme,
                        "target_label": task.target_label,
                        "decoy_label": task.decoy_label,
                        "pred_label": pred,
                        "calibrated_pred_label": calibrated_pred,
                        "correct": int(pred == task.target_label),
                        "calibrated_correct": int(calibrated_pred == task.target_label),
                        "decoy_pred": int(pred == task.decoy_label),
                        "calibrated_decoy_pred": int(calibrated_pred == task.decoy_label),
                        "evidence_hit": int(evidence_hit),
                        "decoy_hit": int(decoy_hit),
                        "selected_pages": len(selected_pages),
                        "prompt_tokens": prompt_tokens,
                        "full_context_tokens": full_context_tokens,
                        "token_ratio": prompt_tokens / full_context_tokens if full_context_tokens else 0.0,
                        "correct_label_nll": -true_score / max(1, lengths[task.target_label]),
                        "margin_true_minus_decoy": true_score - decoy_score,
                        "calibrated_margin_true_minus_decoy": (
                            calibrated_scores[task.target_label] - calibrated_scores[task.decoy_label]
                        ),
                        **{f"score_{label}": scores[label] for label in LABELS},
                        **{f"prior_score_{label}": prior_scores[label] for label in LABELS},
                        **{f"calibrated_score_{label}": calibrated_scores[label] for label in LABELS},
                    }
                )
                del past, prev_logits, prompt_ids
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

    row_fields = [
        "context_tokens",
        "task_id",
        "scheme",
        "target_label",
        "decoy_label",
        "pred_label",
        "calibrated_pred_label",
        "correct",
        "calibrated_correct",
        "decoy_pred",
        "calibrated_decoy_pred",
        "evidence_hit",
        "decoy_hit",
        "selected_pages",
        "prompt_tokens",
        "full_context_tokens",
        "token_ratio",
        "correct_label_nll",
        "margin_true_minus_decoy",
        "calibrated_margin_true_minus_decoy",
        "score_A",
        "score_B",
        "score_C",
        "score_D",
        "prior_score_A",
        "prior_score_B",
        "prior_score_C",
        "prior_score_D",
        "calibrated_score_A",
        "calibrated_score_B",
        "calibrated_score_C",
        "calibrated_score_D",
    ]
    write_csv(output_dir / "downstream_rows.csv", rows, row_fields)
    summary_rows = summarize(rows)
    write_csv(
        output_dir / "downstream_summary.csv",
        summary_rows,
        [
            "context_tokens",
            "scheme",
            "tasks",
            "accuracy",
            "calibrated_accuracy",
            "decoy_pred_rate",
            "calibrated_decoy_pred_rate",
            "evidence_hit_rate",
            "decoy_hit_rate",
            "mean_prompt_tokens",
            "mean_selected_pages",
            "token_ratio",
            "mean_correct_label_nll",
            "mean_margin_true_minus_decoy",
            "mean_calibrated_margin_true_minus_decoy",
        ],
    )
    write_csv(
        output_dir / "downstream_manifest.csv",
        manifest,
        [
            "context_tokens",
            "task_id",
            "target_key",
            "target_label",
            "decoy_label",
            "evidence_page",
            "decoy_page",
            "paragraph_count",
            "section_count",
        ],
    )
    summary = {
        "args": vars(args),
        "resolved": {
            "context_lengths": context_lengths,
            "tasks": len(manifest),
            "schemes": schemes,
            "seconds": time.perf_counter() - started,
        },
        "paths": {
            "summary": str(output_dir / "downstream_summary.csv"),
            "rows": str(output_dir / "downstream_rows.csv"),
            "manifest": str(output_dir / "downstream_manifest.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": summary["resolved"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
