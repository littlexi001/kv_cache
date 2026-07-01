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
    build_token_to_page,
    joined,
    selected_page_set,
)
from analyze_typed_anchor_page_recall import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    RecallAccumulator,
    active_collector,
    anchor_type,
    install_qwen3_attention_patch,
    model_forward,
    pick_input_device,
    resolve_dtype,
    str2bool,
    token_text,
    write_csv,
)


LABELS = ["A", "B", "C", "D"]
TOPICS = [
    "maritime insurance arbitration",
    "Byzantine manuscript catalog",
    "lunar habitat oxygen audit",
    "Renaissance trade ledger",
    "compiler regression memo",
    "rare-disease trial registry",
    "hydroelectric dam inspection",
    "ceramic isotope survey",
]
NAMES = [
    "Amara",
    "Benoit",
    "Celeste",
    "Dorian",
    "Elian",
    "Farah",
    "Gideon",
    "Helena",
    "Ilya",
    "Jun",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    page_id: int | None = None


@dataclass
class GeneratedTask:
    task_id: int
    token_ids: list[int]
    token_texts: list[str]
    prefill_tokens: int
    query_start: int
    evidence_span: Span
    decoy_span: Span
    target_key: str
    target_label: str
    decoy_label: str
    query_text: str
    evidence_spans: list[Span] | None = None
    decoy_spans: list[Span] | None = None
    task_variant: str = "single"
    bridge_key: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-range semantic retrieval diagnostic for hierarchical book-index page routing."
    )
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_tokens", default="10000,20000")
    parser.add_argument("--tasks_per_length", type=int, default=4)
    parser.add_argument("--eval_tokens", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--exclude_sink_tokens", type=int, default=64)
    parser.add_argument("--exclude_recent_tokens", type=int, default=512)
    parser.add_argument("--fixed_page_size", type=int, default=64)
    parser.add_argument("--paragraph_min_tokens", type=int, default=64)
    parser.add_argument("--paragraph_max_tokens", type=int, default=192)
    parser.add_argument("--section_max_paragraphs", type=int, default=8)
    parser.add_argument("--query_window_tokens", type=int, default=256)
    parser.add_argument("--flat_page_counts", default="4,8,16,32")
    parser.add_argument("--hier_section_counts", default="1,2,4,8")
    parser.add_argument("--hier_pages_per_section", default="2,4")
    parser.add_argument("--observe_query_tokens", default="last16", help="all, last16, or comma-separated offsets.")
    parser.add_argument("--write_per_query", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2026070101)
    return parser.parse_args()


def make_key(rng: random.Random, task_id: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return f"LR{task_id:03d}-" + "".join(rng.choice(alphabet) for _ in range(8))


def filler_paragraph(rng: random.Random, idx: int) -> str:
    topic = rng.choice(TOPICS)
    name = rng.choice(NAMES)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    number = rng.randrange(1000, 9999)
    return (
        f"\nFiller note {idx}: {name} reviewed the {topic} packet under reference {code}-{number}. "
        f"The note mentions schedules, materials, witnesses, invoices, and routine cross-checks. "
        "It is background material and does not define the requested answer label.\n"
    )


def encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def append_segment(
    token_ids: list[int],
    tokenizer: Any,
    text: str,
    limit: int | None = None,
) -> Span:
    ids = encode(tokenizer, text)
    if limit is not None:
        ids = ids[: max(0, limit)]
    start = len(token_ids)
    token_ids.extend(ids)
    return Span(start=start, end=len(token_ids))


def fill_to(token_ids: list[int], tokenizer: Any, rng: random.Random, target_len: int, filler_idx: int) -> int:
    while len(token_ids) < target_len:
        remaining = target_len - len(token_ids)
        text = filler_paragraph(rng, filler_idx)
        append_segment(token_ids, tokenizer, text, remaining)
        filler_idx += 1
    return filler_idx


def build_task(tokenizer: Any, rng: random.Random, task_id: int, context_tokens: int, eval_tokens: int) -> GeneratedTask:
    target_key = make_key(rng, task_id)
    target_label = rng.choice(LABELS)
    decoy_label = rng.choice([label for label in LABELS if label != target_label])
    topic = rng.choice(TOPICS)
    witness = rng.choice(NAMES)
    true_text = (
        f"\nAUTHORITATIVE EVIDENCE PAGE. In the {topic}, the verified answer label for lookup key "
        f"{target_key} is ANSWER_LABEL={target_label}. The certifying witness is {witness}. "
        "This authoritative page overrides later reminders, summaries, and decoys.\n"
    )
    decoy_text = (
        f"\nNEAR-TAIL DECOY PAGE. This page repeats lookup key {target_key}, but it is explicitly marked "
        f"as obsolete and non-authoritative. It misleadingly lists ANSWER_LABEL={decoy_label}. "
        "Do not use this decoy when the query asks for the authoritative evidence page.\n"
    )
    query_text = (
        f"\nQuestion: According to the AUTHORITATIVE EVIDENCE PAGE, what is the ANSWER_LABEL for lookup key "
        f"{target_key}? Return only one letter. ANSWER_LABEL:"
    )
    query_ids = encode(tokenizer, query_text)
    if len(query_ids) > eval_tokens:
        query_ids = query_ids[-eval_tokens:]
    token_ids: list[int] = []
    filler_idx = 0
    filler_idx = fill_to(token_ids, tokenizer, rng, min(512, max(96, context_tokens // 20)), filler_idx)
    evidence_span = append_segment(token_ids, tokenizer, true_text)
    decoy_ids = encode(tokenizer, decoy_text)
    decoy_target = max(evidence_span.end + 128, context_tokens - 512 - len(decoy_ids) - 96)
    filler_idx = fill_to(token_ids, tokenizer, rng, decoy_target, filler_idx)
    decoy_span = append_segment(token_ids, tokenizer, decoy_text)
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens, filler_idx)
    if len(token_ids) != context_tokens:
        token_ids = token_ids[:context_tokens]
    query_start = len(token_ids)
    token_ids.extend(query_ids[:eval_tokens])
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens + eval_tokens, filler_idx)
    token_texts = [token_text(tokenizer, token_id) for token_id in token_ids]
    return GeneratedTask(
        task_id=task_id,
        token_ids=token_ids,
        token_texts=token_texts,
        prefill_tokens=context_tokens,
        query_start=query_start,
        evidence_span=evidence_span,
        decoy_span=decoy_span,
        target_key=target_key,
        target_label=target_label,
        decoy_label=decoy_label,
        query_text=query_text,
        evidence_spans=[evidence_span],
        decoy_spans=[decoy_span],
        task_variant="single",
    )


def overlap_page(units: list[TextUnit], span: Span) -> int | None:
    best_page: int | None = None
    best_overlap = 0
    for unit in units:
        overlap = max(0, min(unit.end, span.end) - max(unit.start, span.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = unit.unit_id
    return best_page


def observe_query_positions(prefill_tokens: int, eval_tokens: int, spec: str) -> set[int]:
    if spec == "all":
        return set(range(prefill_tokens, prefill_tokens + eval_tokens))
    if spec == "last16":
        start = prefill_tokens + max(0, eval_tokens - 16)
        return set(range(start, prefill_tokens + eval_tokens))
    offsets = [int(part) for part in spec.split(",") if part.strip()]
    return {prefill_tokens + offset for offset in offsets if 0 <= offset < eval_tokens}


def authority_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for pattern in ["authoritative evidence page", "verified answer label", "certifying witness", "overrides"]:
        if pattern in lowered:
            score += 0.35
    for pattern in ["near-tail decoy", "obsolete", "non-authoritative", "misleadingly", "do not use this decoy"]:
        if pattern in lowered:
            score -= 0.45
    return score


class LongRangeBookIndexCollector:
    def __init__(
        self,
        task: GeneratedTask,
        query_tokens: set[int],
        top_fraction: float,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        fixed_page_size: int,
        anchor_types: list[str],
        paragraph_units: list[TextUnit],
        paragraph_page_ids: list[int],
        paragraph_index: SparseTfidfIndex,
        section_units: list[TextUnit],
        section_index: SparseTfidfIndex,
        section_to_pages: dict[int, list[int]],
        retrieval_schemes: list[str],
        query_window_tokens: int,
        write_per_query: bool,
    ) -> None:
        self.task = task
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.fixed_page_size = fixed_page_size
        self.anchor_types = anchor_types
        self.paragraph_units = paragraph_units
        self.paragraph_page_ids = paragraph_page_ids
        self.paragraph_index = paragraph_index
        self.section_units = section_units
        self.section_index = section_index
        self.section_to_pages = section_to_pages
        self.retrieval_schemes = retrieval_schemes
        self.query_window_tokens = query_window_tokens
        self.write_per_query = write_per_query
        self.evidence_page = overlap_page(paragraph_units, task.evidence_span)
        self.decoy_page = overlap_page(paragraph_units, task.decoy_span)
        self.paragraph_authority_scores = [
            authority_score(joined(task.token_texts, unit.start, unit.end)) for unit in paragraph_units
        ]
        self.section_authority_scores = [
            authority_score(joined(task.token_texts, unit.start, unit.end)) for unit in section_units
        ]
        self.recall_by_scheme: dict[str, RecallAccumulator] = defaultdict(RecallAccumulator)
        self.evidence_hits: Counter[str] = Counter()
        self.decoy_hits: Counter[str] = Counter()
        self.cases: Counter[str] = Counter()
        self.evidence_top2_events: Counter[str] = Counter()
        self.decoy_top2_events: Counter[str] = Counter()
        self.per_query_rows: list[dict[str, Any]] = []

    def _fixed_page(self, token_index: int) -> int:
        return token_index // self.fixed_page_size

    def _paragraph_page(self, token_index: int) -> int:
        return self.paragraph_page_ids[token_index]

    def _runtime_pages(self, query_token: int, remote_end: int) -> dict[str, set[int]]:
        pages: dict[str, set[int]] = {}
        query_start = max(0, query_token - self.query_window_tokens)
        query_text = joined(self.task.token_texts, query_start, query_token)
        paragraph_query = self.paragraph_index.query_vector(query_text)
        section_query = self.section_index.query_vector(query_text)
        paragraph_candidates = [
            unit.unit_id
            for unit in self.paragraph_units
            if unit.end > self.exclude_sink_tokens and unit.start < remote_end
        ]
        section_candidates = [
            unit.unit_id
            for unit in self.section_units
            if unit.end > self.exclude_sink_tokens and unit.start < remote_end
        ]
        for scheme in self.retrieval_schemes:
            if scheme.startswith("book_auth_flat_p"):
                page_count = int(scheme.removeprefix("book_auth_flat_p"))
                scored = [
                    (
                        unit_id,
                        SparseTfidfIndex.cosine(paragraph_query, self.paragraph_index.vectors[unit_id])
                        + self.paragraph_authority_scores[unit_id],
                    )
                    for unit_id in paragraph_candidates
                ]
                scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
                pages[scheme] = {unit_id for unit_id, _ in scored[:page_count]}
                continue
            match = re.fullmatch(r"book_auth_hier_s(\d+)_p(\d+)", scheme)
            if match:
                section_count = int(match.group(1))
                pages_per_section = int(match.group(2))
                section_scored = [
                    (
                        unit_id,
                        SparseTfidfIndex.cosine(section_query, self.section_index.vectors[unit_id])
                        + self.section_authority_scores[unit_id],
                    )
                    for unit_id in section_candidates
                ]
                section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
                selected_pages: set[int] = set()
                for section_id, _ in section_scored[:section_count]:
                    page_candidates = [
                        page_id
                        for page_id in self.section_to_pages.get(section_id, [])
                        if self.paragraph_units[page_id].end > self.exclude_sink_tokens
                        and self.paragraph_units[page_id].start < remote_end
                    ]
                    page_scored = [
                        (
                            page_id,
                            SparseTfidfIndex.cosine(paragraph_query, self.paragraph_index.vectors[page_id])
                            + self.paragraph_authority_scores[page_id],
                        )
                        for page_id in page_candidates
                    ]
                    page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
                    selected_pages.update(page_id for page_id, _ in page_scored[:pages_per_section])
                pages[scheme] = selected_pages
                continue
            selected, _ = selected_page_set(
                scheme,
                query_token,
                self.task.token_texts,
                self.paragraph_units,
                self.paragraph_index,
                self.section_units,
                self.section_index,
                self.section_to_pages,
                self.exclude_sink_tokens,
                remote_end,
                self.query_window_tokens,
            )
            pages[scheme] = selected
        return pages

    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = min(int(finite[0, 0].sum().item()), query_token + 1)
        if valid_count <= 1:
            return
        history_count = valid_count - 1
        remote_end = max(0, history_count - self.exclude_recent_tokens)
        if remote_end <= self.exclude_sink_tokens:
            return
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        top_masses = torch.gather(attention_weights, dim=-1, index=top_indices)
        remote_mask = (top_indices >= self.exclude_sink_tokens) & (top_indices < remote_end)
        runtime_pages = self._runtime_pages(query_token, remote_end)

        for head in range(top_indices.shape[0]):
            head_indices = top_indices[head, remote_mask[head]].detach().cpu().tolist()
            head_masses = top_masses[head, remote_mask[head]].detach().cpu().tolist()
            if not head_indices:
                continue
            structural_tokens: list[int] = []
            semantic_tokens: list[tuple[int, float]] = []
            evidence_top2 = 0
            decoy_top2 = 0
            for token_index, mass in zip(head_indices, head_masses):
                if self.task.evidence_span.start <= int(token_index) < self.task.evidence_span.end:
                    evidence_top2 += 1
                if self.task.decoy_span.start <= int(token_index) < self.task.decoy_span.end:
                    decoy_top2 += 1
                anchor = self.anchor_types[int(token_index)]
                if anchor == "structural":
                    structural_tokens.append(int(token_index))
                elif anchor == "semantic":
                    semantic_tokens.append((int(token_index), float(mass)))
            semantic_events = len(semantic_tokens)
            semantic_mass = sum(mass for _, mass in semantic_tokens)
            anchor_pages = {
                "fixed_anchor": {self._fixed_page(token_index) for token_index in structural_tokens},
                "paragraph_anchor": {self._paragraph_page(token_index) for token_index in structural_tokens},
            }
            schemes = {**anchor_pages, **runtime_pages}
            for scheme, pages in schemes.items():
                page_of = self._fixed_page if scheme == "fixed_anchor" else self._paragraph_page
                covered = [(idx, mass) for idx, mass in semantic_tokens if page_of(idx) in pages]
                covered_events = len(covered)
                covered_mass = sum(mass for _, mass in covered)
                self.recall_by_scheme[scheme].add(
                    len(pages),
                    len(pages),
                    semantic_events,
                    semantic_mass,
                    covered_events,
                    covered_mass,
                )
                self.cases[scheme] += 1
                if self.evidence_page is not None and self.evidence_page in pages:
                    self.evidence_hits[scheme] += 1
                if self.decoy_page is not None and self.decoy_page in pages:
                    self.decoy_hits[scheme] += 1
                self.evidence_top2_events[scheme] += evidence_top2
                self.decoy_top2_events[scheme] += decoy_top2
                if self.write_per_query:
                    self.per_query_rows.append(
                        {
                            "task_id": self.task.task_id,
                            "query_token": query_token,
                            "layer": layer,
                            "head": head,
                            "scheme": scheme,
                            "selected_pages": len(pages),
                            "evidence_hit": int(self.evidence_page is not None and self.evidence_page in pages),
                            "decoy_hit": int(self.decoy_page is not None and self.decoy_page in pages),
                            "semantic_events": semantic_events,
                            "semantic_mass": semantic_mass,
                            "covered_semantic_events": covered_events,
                            "covered_semantic_mass": covered_mass,
                            "semantic_mass_recall": covered_mass / semantic_mass if semantic_mass else 0.0,
                            "evidence_top2_events": evidence_top2,
                            "decoy_top2_events": decoy_top2,
                        }
                    )

    def rows(self, context_tokens: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scheme, acc in sorted(self.recall_by_scheme.items()):
            base = acc.row({"context_tokens": context_tokens, "task_id": self.task.task_id, "scheme": scheme})
            cases = self.cases[scheme]
            base.update(
                {
                    "evidence_page": self.evidence_page,
                    "decoy_page": self.decoy_page,
                    "evidence_hit_rate": self.evidence_hits[scheme] / cases if cases else 0.0,
                    "decoy_hit_rate": self.decoy_hits[scheme] / cases if cases else 0.0,
                    "evidence_top2_events": self.evidence_top2_events[scheme],
                    "decoy_top2_events": self.decoy_top2_events[scheme],
                }
            )
            rows.append(base)
        return rows


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: LongRangeBookIndexCollector,
) -> None:
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_idx, start in enumerate(range(prefill_tokens, prefill_tokens + eval_tokens, chunk_size), start=1):
            end = min(start + chunk_size, prefill_tokens + eval_tokens)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end].to(input_device),
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()


def build_indexes(task: GeneratedTask, args: argparse.Namespace) -> tuple[
    list[TextUnit],
    list[int],
    SparseTfidfIndex,
    list[TextUnit],
    SparseTfidfIndex,
    dict[int, list[int]],
]:
    sentences = build_sentences(task.token_texts[: task.prefill_tokens], min_sentence_tokens=8)
    paragraphs = build_paragraphs(
        task.token_texts[: task.prefill_tokens],
        sentences,
        args.paragraph_min_tokens,
        args.paragraph_max_tokens,
    )
    sections = build_sections(paragraphs, args.section_max_paragraphs)
    paragraphs = assign_parents(paragraphs, sections)
    paragraph_page_ids = build_token_to_page(paragraphs, len(task.token_ids))
    paragraph_docs = [joined(task.token_texts, unit.start, unit.end) for unit in paragraphs]
    section_docs = [joined(task.token_texts, unit.start, unit.end) for unit in sections]
    paragraph_index = SparseTfidfIndex(paragraph_docs)
    section_index = SparseTfidfIndex(section_docs)
    section_to_pages: dict[int, list[int]] = defaultdict(list)
    for paragraph in paragraphs:
        if paragraph.parent_id is not None:
            section_to_pages[paragraph.parent_id].append(paragraph.unit_id)
    return paragraphs, paragraph_page_ids, paragraph_index, sections, section_index, section_to_pages


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (int(row["context_tokens"]), str(row["scheme"]))
        group = grouped[key]
        cases = float(row.get("cases", 0.0))
        group["anchor_pages"] += float(row.get("mean_anchor_pages", 0.0)) * cases
        for name in [
            "cases",
            "anchor_events",
            "semantic_events",
            "semantic_mass",
            "covered_semantic_events",
            "covered_semantic_mass",
            "evidence_hit_rate",
            "decoy_hit_rate",
            "evidence_top2_events",
            "decoy_top2_events",
        ]:
            if name in row:
                group[name] += float(row[name])
        group["tasks"] += 1.0
    out = []
    for (context_tokens, scheme), group in sorted(grouped.items()):
        cases = group["cases"]
        tasks = group["tasks"]
        out.append(
            {
                "context_tokens": context_tokens,
                "scheme": scheme,
                "tasks": int(tasks),
                "cases": int(cases),
                "mean_pages": group["anchor_pages"] / cases if cases else 0.0,
                "semantic_events": int(group["semantic_events"]),
                "semantic_mass": group["semantic_mass"],
                "covered_semantic_events": int(group["covered_semantic_events"]),
                "covered_semantic_mass": group["covered_semantic_mass"],
                "semantic_event_recall": (
                    group["covered_semantic_events"] / group["semantic_events"] if group["semantic_events"] else 0.0
                ),
                "semantic_mass_recall": (
                    group["covered_semantic_mass"] / group["semantic_mass"] if group["semantic_mass"] else 0.0
                ),
                "evidence_hit_rate": group["evidence_hit_rate"] / tasks if tasks else 0.0,
                "decoy_hit_rate": group["decoy_hit_rate"] / tasks if tasks else 0.0,
                "evidence_top2_events": int(group["evidence_top2_events"]),
                "decoy_top2_events": int(group["decoy_top2_events"]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = [int(part) for part in args.context_tokens.split(",") if part.strip()]
    flat_counts = [int(part) for part in args.flat_page_counts.split(",") if part.strip()]
    section_counts = [int(part) for part in args.hier_section_counts.split(",") if part.strip()]
    pages_per_section = [int(part) for part in args.hier_pages_per_section.split(",") if part.strip()]
    retrieval_schemes = [f"remote_tail_p{count}" for count in flat_counts]
    retrieval_schemes += [f"book_flat_p{count}" for count in flat_counts]
    retrieval_schemes += [f"book_auth_flat_p{count}" for count in flat_counts]
    retrieval_schemes += [f"book_hier_s{s}_p{p}" for s in section_counts for p in pages_per_section]
    retrieval_schemes += [f"book_auth_hier_s{s}_p{p}" for s in section_counts for p in pages_per_section]

    rng = random.Random(args.seed)
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    task_rows: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    for context_tokens in context_lengths:
        for task_index in range(args.tasks_per_length):
            task_id = context_tokens * 100 + task_index
            task = build_task(tokenizer, rng, task_id, context_tokens, args.eval_tokens)
            paragraphs, paragraph_page_ids, paragraph_index, sections, section_index, section_to_pages = build_indexes(
                task,
                args,
            )
            evidence_page = overlap_page(paragraphs, task.evidence_span)
            decoy_page = overlap_page(paragraphs, task.decoy_span)
            manifest.append(
                {
                    "context_tokens": context_tokens,
                    "task_id": task.task_id,
                    "target_key": task.target_key,
                    "target_label": task.target_label,
                    "decoy_label": task.decoy_label,
                    "evidence_start": task.evidence_span.start,
                    "evidence_end": task.evidence_span.end,
                    "decoy_start": task.decoy_span.start,
                    "decoy_end": task.decoy_span.end,
                    "paragraph_count": len(paragraphs),
                    "section_count": len(sections),
                    "paragraph_mean_tokens": (
                        sum(unit.length for unit in paragraphs) / len(paragraphs) if paragraphs else 0.0
                    ),
                    "evidence_page": evidence_page,
                    "decoy_page": decoy_page,
                }
            )
            print(
                f"context={context_tokens} task={task_index + 1}/{args.tasks_per_length} "
                f"evidence_page={evidence_page} decoy_page={decoy_page}",
                flush=True,
            )
            anchor_types = [anchor_type(text_piece) for text_piece in task.token_texts]
            collector = LongRangeBookIndexCollector(
                task=task,
                query_tokens=observe_query_positions(task.prefill_tokens, args.eval_tokens, args.observe_query_tokens),
                top_fraction=args.top_fraction,
                exclude_sink_tokens=args.exclude_sink_tokens,
                exclude_recent_tokens=args.exclude_recent_tokens,
                fixed_page_size=args.fixed_page_size,
                anchor_types=anchor_types,
                paragraph_units=paragraphs,
                paragraph_page_ids=paragraph_page_ids,
                paragraph_index=paragraph_index,
                section_units=sections,
                section_index=section_index,
                section_to_pages=section_to_pages,
                retrieval_schemes=retrieval_schemes,
                query_window_tokens=args.query_window_tokens,
                write_per_query=args.write_per_query,
            )
            input_ids = torch.tensor(task.token_ids, dtype=torch.long).view(1, -1)
            past = prefill_cache(model, input_ids, task.prefill_tokens, args.chunk_size, input_device)
            run_eval(
                model,
                input_ids,
                past,
                task.prefill_tokens,
                args.eval_tokens,
                args.chunk_size,
                input_device,
                collector,
            )
            task_rows.extend(collector.rows(context_tokens))
            all_per_query.extend(collector.per_query_rows)
            del past
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    task_fields = [
        "context_tokens",
        "task_id",
        "scheme",
        "cases",
        "anchor_events",
        "mean_anchor_events",
        "mean_anchor_pages",
        "semantic_events",
        "semantic_mass",
        "covered_semantic_events",
        "covered_semantic_mass",
        "semantic_event_recall",
        "semantic_mass_recall",
        "evidence_page",
        "decoy_page",
        "evidence_hit_rate",
        "decoy_hit_rate",
        "evidence_top2_events",
        "decoy_top2_events",
    ]
    write_csv(output_dir / "longrange_task_recall.csv", task_rows, task_fields)
    summary_rows = aggregate(task_rows)
    write_csv(
        output_dir / "longrange_summary.csv",
        summary_rows,
        [
            "context_tokens",
            "scheme",
            "tasks",
            "cases",
            "mean_pages",
            "semantic_events",
            "semantic_mass",
            "covered_semantic_events",
            "covered_semantic_mass",
            "semantic_event_recall",
            "semantic_mass_recall",
            "evidence_hit_rate",
            "decoy_hit_rate",
            "evidence_top2_events",
            "decoy_top2_events",
        ],
    )
    if args.write_per_query:
        write_csv(
            output_dir / "longrange_per_query.csv",
            all_per_query,
            [
                "task_id",
                "query_token",
                "layer",
                "head",
                "scheme",
                "selected_pages",
                "evidence_hit",
                "decoy_hit",
                "semantic_events",
                "semantic_mass",
                "covered_semantic_events",
                "covered_semantic_mass",
                "semantic_mass_recall",
                "evidence_top2_events",
                "decoy_top2_events",
            ],
        )
    write_csv(
        output_dir / "longrange_manifest.csv",
        manifest,
        [
            "context_tokens",
            "task_id",
            "target_key",
            "target_label",
            "decoy_label",
            "evidence_start",
            "evidence_end",
            "decoy_start",
            "decoy_end",
            "paragraph_count",
            "section_count",
            "paragraph_mean_tokens",
            "evidence_page",
            "decoy_page",
        ],
    )
    summary = {
        "args": vars(args),
        "resolved": {
            "context_lengths": context_lengths,
            "tasks": len(manifest),
            "retrieval_schemes": retrieval_schemes,
            "seconds": time.perf_counter() - started,
        },
        "paths": {
            "summary": str(output_dir / "longrange_summary.csv"),
            "task_recall": str(output_dir / "longrange_task_recall.csv"),
            "manifest": str(output_dir / "longrange_manifest.csv"),
            "per_query": str(output_dir / "longrange_per_query.csv") if args.write_per_query else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": summary["resolved"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
