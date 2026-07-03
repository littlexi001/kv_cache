from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_hierarchical_book_index_recall import (  # noqa: E402
    SparseTfidfIndex,
    TextUnit,
    assign_parents,
    build_paragraphs,
    build_sections,
    build_sentences,
    joined,
)
from book_page_router import adaptive_section_fanout  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    pick_input_device,
    resolve_dtype,
)


@dataclass(frozen=True)
class RoutedMemory:
    selected_pages: list[int]
    prompt_ids: list[int]
    summary_text: str
    route_seconds: float
    summary_token_count: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt chain_typedhier_role_auto_p1 + answerline_summary to plain LM continuation: "
            "natural pages -> hierarchical lexical routing -> typed continuation memory -> next-token PPL."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", default="data/war_and_peace_pg2600.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=20_000)
    parser.add_argument("--eval_tokens", type=int, default=5_000)
    parser.add_argument("--eval_window_tokens", type=int, default=256)
    parser.add_argument("--recent_tokens", type=int, default=1024)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--query_window_tokens", type=int, default=256)
    parser.add_argument("--min_sentence_tokens", type=int, default=8)
    parser.add_argument("--paragraph_min_tokens", type=int, default=64)
    parser.add_argument("--paragraph_max_tokens", type=int, default=192)
    parser.add_argument("--section_max_paragraphs", type=int, default=8)
    parser.add_argument("--pages_per_section", type=int, default=1)
    parser.add_argument("--seed_pages", type=int, default=2)
    parser.add_argument("--tail_pages", type=int, default=1)
    parser.add_argument("--section_count", default="auto")
    parser.add_argument("--adjacent_radius", type=int, default=0)
    parser.add_argument("--summary_excerpt_words", type=int, default=48)
    parser.add_argument("--max_summary_tokens", type=int, default=1536)
    parser.add_argument(
        "--modes",
        default="full,recent,typedhier_summary,typedhier_raw,typedhier_plain_summary,typedhier_plain_raw",
    )
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--write_windows", type=str2bool, default=True)
    parser.add_argument("--write_examples", type=int, default=3)
    return parser.parse_args()


def read_text(path: str, max_chars: int) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")[:max_chars]


def encode_text(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    return tokenizer(text, add_special_tokens=add_special_tokens, return_tensors=None)["input_ids"]


def decode_ids(tokenizer: Any, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [decode_ids(tokenizer, [int(token_id)]) for token_id in token_ids]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_words(text: str, word_limit: int) -> str:
    words = compact_text(text).split()
    return " ".join(words[:word_limit])


def build_indexes(
    tokenizer: Any,
    token_ids: list[int],
    prefill_tokens: int,
    min_sentence_tokens: int,
    paragraph_min_tokens: int,
    paragraph_max_tokens: int,
    section_max_paragraphs: int,
) -> tuple[list[str], list[TextUnit], SparseTfidfIndex, list[TextUnit], SparseTfidfIndex, dict[int, list[int]]]:
    prefix_ids = token_ids[:prefill_tokens]
    texts = token_texts(tokenizer, prefix_ids)
    sentences = build_sentences(texts, min_sentence_tokens)
    paragraphs = build_paragraphs(texts, sentences, paragraph_min_tokens, paragraph_max_tokens)
    sections = build_sections(paragraphs, section_max_paragraphs)
    paragraphs = assign_parents(paragraphs, sections)
    paragraph_docs = [joined(texts, unit.start, unit.end) for unit in paragraphs]
    section_docs = [joined(texts, unit.start, unit.end) for unit in sections]
    page_index = SparseTfidfIndex(paragraph_docs)
    section_index = SparseTfidfIndex(section_docs)
    section_to_pages: dict[int, list[int]] = defaultdict(list)
    for paragraph in paragraphs:
        if paragraph.parent_id is not None:
            section_to_pages[int(paragraph.parent_id)].append(paragraph.unit_id)
    return texts, paragraphs, page_index, sections, section_index, dict(section_to_pages)


def eligible_pages(
    pages: list[TextUnit],
    sink_tokens: int,
    remote_end: int,
) -> list[int]:
    return [
        unit.unit_id
        for unit in pages
        if unit.end > sink_tokens and unit.start < remote_end and unit.length > 0
    ]


def eligible_sections(
    sections: list[TextUnit],
    sink_tokens: int,
    remote_end: int,
) -> list[int]:
    return [
        unit.unit_id
        for unit in sections
        if unit.end > sink_tokens and unit.start < remote_end and unit.length > 0
    ]


def expand_adjacent(selected: set[int], candidate_pages: list[int], radius: int) -> set[int]:
    if radius <= 0:
        return selected
    candidate_set = set(candidate_pages)
    expanded = set(selected)
    for page_id in list(selected):
        for delta in range(1, radius + 1):
            if page_id - delta in candidate_set:
                expanded.add(page_id - delta)
            if page_id + delta in candidate_set:
                expanded.add(page_id + delta)
    return expanded


def select_typedhier_pages(
    query_text: str,
    pages: list[TextUnit],
    page_index: SparseTfidfIndex,
    sections: list[TextUnit],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    sink_tokens: int,
    remote_end: int,
    section_count_spec: str,
    pages_per_section: int,
    seed_pages: int,
    adjacent_radius: int,
) -> list[int]:
    candidate_pages = eligible_pages(pages, sink_tokens, remote_end)
    candidate_sections = eligible_sections(sections, sink_tokens, remote_end)
    if not candidate_pages:
        return []
    query_vec = page_index.query_vector(query_text)
    seeds = {page_id for page_id, _ in page_index.topk(query_vec, candidate_pages, seed_pages)}
    if section_count_spec == "auto":
        section_count = adaptive_section_fanout(section_to_pages, candidate_sections)
    else:
        section_count = max(1, int(section_count_spec))
    selected = set(seeds)
    section_query = section_index.query_vector(query_text)
    for section_id, _ in section_index.topk(section_query, candidate_sections, section_count):
        page_ids = [page_id for page_id in section_to_pages.get(section_id, []) if page_id in candidate_pages]
        selected.update(page_id for page_id, _ in page_index.topk(query_vec, page_ids, pages_per_section))
    selected = expand_adjacent(selected, candidate_pages, adjacent_radius)
    return sorted(selected, key=lambda page_id: (pages[page_id].start, page_id))


def add_tail_pages(
    selected_pages: list[int],
    pages: list[TextUnit],
    sink_tokens: int,
    remote_end: int,
    tail_pages: int,
) -> list[int]:
    if tail_pages <= 0:
        return selected_pages
    candidates = eligible_pages(pages, sink_tokens, remote_end)
    tail = sorted(candidates, key=lambda page_id: (pages[page_id].end, page_id))[-tail_pages:]
    return sorted(set(selected_pages) | set(tail), key=lambda page_id: (pages[page_id].start, page_id))


def build_typed_summary_text(
    tokenizer: Any,
    token_texts_prefix: list[str],
    pages: list[TextUnit],
    page_index: SparseTfidfIndex,
    selected_pages: list[int],
    query_text: str,
    excerpt_words: int,
    max_summary_tokens: int,
    raw: bool,
) -> tuple[str, int]:
    lines = [
        "Typed continuation memory: status=remote_context.",
        f"CURRENT_QUERY={compact_text(query_text)[-500:]}",
        "Use routed remote pages as background; continue from RECENT_CONTEXT.",
    ]
    for rank, page_id in enumerate(selected_pages, start=1):
        page = pages[page_id]
        text = joined(token_texts_prefix, page.start, page.end)
        terms = ",".join(page_index.summary_terms(page_id, 10))
        if raw:
            payload = compact_text(text)
        else:
            payload = first_words(text, excerpt_words)
        lines.append(
            f"ROUTED_PAGE_{rank}: page_id={page_id}; token_range={page.start}-{page.end}; "
            f"terms={terms}; excerpt={payload}"
        )
    text = "\n".join(lines) + "\n"
    ids = encode_text(tokenizer, text, add_special_tokens=False)
    if len(ids) <= max_summary_tokens:
        return text, len(ids)
    trimmed_ids = ids[:max_summary_tokens]
    trimmed_text = decode_ids(tokenizer, trimmed_ids)
    return trimmed_text + "\n", len(trimmed_ids)


def build_routed_memory_prompt(
    tokenizer: Any,
    all_token_ids: list[int],
    token_texts_prefix: list[str],
    pages: list[TextUnit],
    page_index: SparseTfidfIndex,
    sections: list[TextUnit],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    history_end: int,
    args: argparse.Namespace,
    raw: bool,
) -> RoutedMemory:
    route_start = time.perf_counter()
    recent_start = max(0, history_end - args.query_window_tokens)
    query_text = decode_ids(tokenizer, all_token_ids[recent_start:history_end])
    remote_end = max(args.sink_tokens, min(args.prefill_tokens, history_end - args.recent_tokens))
    selected_pages = select_typedhier_pages(
        query_text,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        args.sink_tokens,
        remote_end,
        args.section_count,
        args.pages_per_section,
        args.seed_pages,
        args.adjacent_radius,
    )
    summary_text, summary_tokens = build_typed_summary_text(
        tokenizer,
        token_texts_prefix,
        pages,
        page_index,
        selected_pages,
        query_text,
        args.summary_excerpt_words,
        args.max_summary_tokens,
        raw=raw,
    )
    sink_ids = all_token_ids[: args.sink_tokens]
    recent_ids = all_token_ids[max(0, history_end - args.recent_tokens) : history_end]
    header_ids = encode_text(tokenizer, "\n\n[MEMORY]\n", add_special_tokens=False)
    summary_ids = encode_text(tokenizer, summary_text, add_special_tokens=False)
    recent_header_ids = encode_text(tokenizer, "\n[RECENT_CONTEXT]\n", add_special_tokens=False)
    prompt_ids = sink_ids + header_ids + summary_ids + recent_header_ids + recent_ids
    return RoutedMemory(
        selected_pages=selected_pages,
        prompt_ids=prompt_ids,
        summary_text=summary_text,
        route_seconds=time.perf_counter() - route_start,
        summary_token_count=summary_tokens,
    )


def build_plain_routed_memory_prompt(
    tokenizer: Any,
    all_token_ids: list[int],
    token_texts_prefix: list[str],
    pages: list[TextUnit],
    page_index: SparseTfidfIndex,
    sections: list[TextUnit],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    history_end: int,
    args: argparse.Namespace,
    raw: bool,
    include_tail: bool,
) -> RoutedMemory:
    route_start = time.perf_counter()
    recent_start = max(0, history_end - args.query_window_tokens)
    query_text = decode_ids(tokenizer, all_token_ids[recent_start:history_end])
    remote_end = max(args.sink_tokens, min(args.prefill_tokens, history_end - args.recent_tokens))
    selected_pages = select_typedhier_pages(
        query_text,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        args.sink_tokens,
        remote_end,
        args.section_count,
        args.pages_per_section,
        args.seed_pages,
        args.adjacent_radius,
    )
    if include_tail:
        selected_pages = add_tail_pages(
            selected_pages,
            pages,
            args.sink_tokens,
            remote_end,
            args.tail_pages,
        )
    if raw:
        remote_ids: list[int] = []
        for page_id in selected_pages:
            page = pages[page_id]
            remote_ids.extend(all_token_ids[page.start : page.end])
            remote_ids.extend(encode_text(tokenizer, "\n\n", add_special_tokens=False))
        if len(remote_ids) > args.max_summary_tokens:
            remote_ids = remote_ids[: args.max_summary_tokens]
        summary_text = decode_ids(tokenizer, remote_ids)
        summary_tokens = len(remote_ids)
    else:
        chunks = []
        for page_id in selected_pages:
            page = pages[page_id]
            text = joined(token_texts_prefix, page.start, page.end)
            chunks.append(first_words(text, args.summary_excerpt_words))
        summary_text = "\n\n".join(chunk for chunk in chunks if chunk).strip() + "\n"
        remote_ids = encode_text(tokenizer, summary_text, add_special_tokens=False)
        if len(remote_ids) > args.max_summary_tokens:
            remote_ids = remote_ids[: args.max_summary_tokens]
            summary_text = decode_ids(tokenizer, remote_ids)
        summary_tokens = len(remote_ids)
    sink_ids = all_token_ids[: args.sink_tokens]
    recent_ids = all_token_ids[max(0, history_end - args.recent_tokens) : history_end]
    sep_ids = encode_text(tokenizer, "\n\n", add_special_tokens=False)
    prompt_ids = sink_ids + sep_ids + remote_ids + sep_ids + recent_ids
    return RoutedMemory(
        selected_pages=selected_pages,
        prompt_ids=prompt_ids,
        summary_text=summary_text,
        route_seconds=time.perf_counter() - route_start,
        summary_token_count=summary_tokens,
    )


def build_prompt_ids(
    mode: str,
    tokenizer: Any,
    all_token_ids: list[int],
    token_texts_prefix: list[str],
    pages: list[TextUnit],
    page_index: SparseTfidfIndex,
    sections: list[TextUnit],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    history_end: int,
    args: argparse.Namespace,
) -> RoutedMemory:
    if mode == "full":
        return RoutedMemory([], all_token_ids[:history_end], "", 0.0, 0)
    if mode == "recent":
        sink_ids = all_token_ids[: args.sink_tokens]
        recent_ids = all_token_ids[max(0, history_end - args.recent_tokens) : history_end]
        joiner = encode_text(tokenizer, "\n\n[RECENT_CONTEXT]\n", add_special_tokens=False)
        return RoutedMemory([], sink_ids + joiner + recent_ids, "", 0.0, 0)
    if mode == "typedhier_summary":
        return build_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=False,
        )
    if mode == "typedhier_raw":
        return build_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=True,
        )
    if mode == "typedhier_plain_summary":
        return build_plain_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=False,
            include_tail=False,
        )
    if mode == "typedhier_plain_raw":
        return build_plain_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=True,
            include_tail=False,
        )
    if mode == "typedhier_tail_summary":
        return build_plain_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=False,
            include_tail=True,
        )
    if mode == "typedhier_tail_raw":
        return build_plain_routed_memory_prompt(
            tokenizer,
            all_token_ids,
            token_texts_prefix,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            history_end,
            args,
            raw=True,
            include_tail=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def score_target(
    model: Any,
    input_device: torch.device,
    prompt_ids: list[int],
    target_ids: list[int],
) -> tuple[float, int, float]:
    if not target_ids:
        return 0.0, 0, 0.0
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=input_device)
    prompt_len = len(prompt_ids)
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        labels = input_ids[:, prompt_len:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            labels.view(-1),
            reduction="sum",
        )
    if input_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return float(losses.item()), int(labels.numel()), seconds


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text(args.text_path, args.max_chars)
    token_ids = encode_text(tokenizer, text, args.add_special_tokens)
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    required = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < required:
        raise RuntimeError(f"Need {required} tokens, got {len(token_ids)}")
    token_ids = token_ids[:required]

    print("building natural page hierarchy...", flush=True)
    (
        token_texts_prefix,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
    ) = build_indexes(
        tokenizer,
        token_ids,
        args.prefill_tokens,
        args.min_sentence_tokens,
        args.paragraph_min_tokens,
        args.paragraph_max_tokens,
        args.section_max_paragraphs,
    )

    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()
    input_device = pick_input_device(model, args.device)

    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for mode in modes:
        total_loss = 0.0
        total_tokens = 0
        total_forward_seconds = 0.0
        total_route_seconds = 0.0
        prompt_lens: list[float] = []
        page_counts: list[float] = []
        summary_lens: list[float] = []
        mode_start = time.perf_counter()
        window_count = math.ceil(args.eval_tokens / args.eval_window_tokens)
        for window_idx, start in enumerate(
            range(args.prefill_tokens, args.prefill_tokens + args.eval_tokens, args.eval_window_tokens),
            start=1,
        ):
            end = min(args.prefill_tokens + args.eval_tokens, start + args.eval_window_tokens)
            target_ids = token_ids[start:end]
            routed = build_prompt_ids(
                mode,
                tokenizer,
                token_ids,
                token_texts_prefix,
                pages,
                page_index,
                sections,
                section_index,
                section_to_pages,
                start,
                args,
            )
            loss, count, forward_seconds = score_target(model, input_device, routed.prompt_ids, target_ids)
            total_loss += loss
            total_tokens += count
            total_forward_seconds += forward_seconds
            total_route_seconds += routed.route_seconds
            prompt_lens.append(float(len(routed.prompt_ids)))
            page_counts.append(float(len(routed.selected_pages)))
            summary_lens.append(float(routed.summary_token_count))
            ppl = math.exp(total_loss / max(1, total_tokens))
            print(
                f"{mode} window {window_idx}/{window_count}: tokens {start}-{end - 1} "
                f"ppl_so_far={ppl:.4f} prompt={len(routed.prompt_ids)} pages={len(routed.selected_pages)}",
                flush=True,
            )
            if args.write_windows:
                window_rows.append(
                    {
                        "mode": mode,
                        "window_index": window_idx,
                        "target_start": start,
                        "target_end": end,
                        "loss": loss,
                        "token_count": count,
                        "ppl": math.exp(loss / max(1, count)),
                        "prompt_tokens": len(routed.prompt_ids),
                        "selected_pages": " ".join(str(page_id) for page_id in routed.selected_pages),
                        "selected_page_count": len(routed.selected_pages),
                        "summary_tokens": routed.summary_token_count,
                        "route_seconds": routed.route_seconds,
                        "forward_seconds": forward_seconds,
                    }
                )
            if len(examples) < args.write_examples and routed.summary_text:
                examples.append(
                    {
                        "mode": mode,
                        "window_index": window_idx,
                        "target_start": start,
                        "selected_pages": routed.selected_pages,
                        "summary_text": routed.summary_text,
                        "target_preview": decode_ids(tokenizer, target_ids[:80]),
                    }
                )
        total_seconds = time.perf_counter() - mode_start
        rows.append(
            {
                "mode": mode,
                "loss": total_loss / max(1, total_tokens),
                "ppl": math.exp(total_loss / max(1, total_tokens)),
                "token_count": total_tokens,
                "total_seconds": total_seconds,
                "forward_seconds": total_forward_seconds,
                "route_seconds": total_route_seconds,
                "avg_prompt_tokens": mean(prompt_lens),
                "avg_selected_pages": mean(page_counts),
                "avg_summary_tokens": mean(summary_lens),
                "prefill_tokens": args.prefill_tokens,
                "eval_tokens": args.eval_tokens,
                "eval_window_tokens": args.eval_window_tokens,
                "recent_tokens": args.recent_tokens,
                "sink_tokens": args.sink_tokens,
                "query_window_tokens": args.query_window_tokens,
                "paragraph_min_tokens": args.paragraph_min_tokens,
                "paragraph_max_tokens": args.paragraph_max_tokens,
                "section_max_paragraphs": args.section_max_paragraphs,
                "section_count": args.section_count,
                "pages_per_section": args.pages_per_section,
                "seed_pages": args.seed_pages,
                "adjacent_radius": args.adjacent_radius,
                "summary_excerpt_words": args.summary_excerpt_words,
                "max_summary_tokens": args.max_summary_tokens,
            }
        )

    with (output_dir / "ppl_by_mode.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.write_windows:
        with (output_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()))
            writer.writeheader()
            writer.writerows(window_rows)
    (output_dir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_used": len(token_ids),
            "paragraph_count": len(pages),
            "section_count": len(sections),
            "paragraph_mean_tokens": mean([float(page.length) for page in pages]),
            "modes": modes,
            "method_note": (
                "Plain-LM adaptation of chain_typedhier_role_auto_p1 + answerline_summary: "
                "natural paragraph pages, hierarchical section/page routing from recent query text, "
                "typed continuation memory summaries, then teacher-forced continuation PPL."
            ),
        },
        "paths": {
            "ppl_by_mode": str(output_dir / "ppl_by_mode.csv"),
            "windows": str(output_dir / "windows.csv") if args.write_windows else None,
            "examples": str(output_dir / "examples.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
