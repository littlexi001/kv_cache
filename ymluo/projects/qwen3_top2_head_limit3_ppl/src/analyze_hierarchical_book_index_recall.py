from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from analyze_typed_anchor_page_recall import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RecallAccumulator,
    active_collector,
    anchor_type,
    build_query_samples,
    install_qwen3_attention_patch,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
    str2bool,
    token_text,
    write_csv,
)


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "he",
    "she",
    "it",
    "they",
    "we",
    "you",
    "i",
    "his",
    "her",
    "him",
    "them",
    "that",
    "this",
    "was",
    "is",
    "had",
    "have",
    "not",
    "be",
    "been",
    "are",
    "were",
    "will",
    "would",
    "could",
    "should",
    "there",
    "their",
    "what",
    "when",
    "where",
    "which",
    "who",
    "said",
}


@dataclass(frozen=True)
class TextUnit:
    unit_id: int
    level: str
    start: int
    end: int
    parent_id: int | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


class SparseTfidfIndex:
    def __init__(self, docs: list[str]) -> None:
        self.doc_tokens = [self._words(doc) for doc in docs]
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        doc_count = max(1, len(docs))
        self.idf = {word: math.log((doc_count + 1) / (count + 1)) + 1.0 for word, count in df.items()}
        self.vectors = [self._vector_from_tokens(tokens) for tokens in self.doc_tokens]

    @staticmethod
    def _words(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}|\d+", text.lower())
        return [word for word in words if word not in STOPWORDS and len(word) > 1]

    def _vector_from_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec = {word: float(count) * self.idf.get(word, 1.0) for word, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vec.values()))
        if norm > 0:
            vec = {word: value / norm for word, value in vec.items()}
        return vec

    def query_vector(self, text: str) -> dict[str, float]:
        return self._vector_from_tokens(self._words(text))

    @staticmethod
    def cosine(query: dict[str, float], doc: dict[str, float]) -> float:
        if len(query) > len(doc):
            query, doc = doc, query
        return sum(value * doc.get(word, 0.0) for word, value in query.items())

    def topk(self, query: dict[str, float], candidate_ids: list[int], k: int) -> list[tuple[int, float]]:
        scored = [(unit_id, self.cosine(query, self.vectors[unit_id])) for unit_id in candidate_ids]
        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return scored[: max(0, k)]

    def summary_terms(self, unit_id: int, limit: int) -> list[str]:
        items = sorted(self.vectors[unit_id].items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in items[:limit]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical book-like lexical index recall against true remote top2 semantic tokens."
    )
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument("--text_path", default="data/war_and_peace_pg2600.txt")
    parser.add_argument("--output_dir", default="outputs/hierarchical_book_index_recall")
    parser.add_argument("--total_tokens", type=int, default=4160)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--eval_tokens", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--exclude_sink_tokens", type=int, default=64)
    parser.add_argument("--exclude_recent_tokens", type=int, default=512)
    parser.add_argument("--fixed_page_size", type=int, default=64)
    parser.add_argument("--min_sentence_tokens", type=int, default=8)
    parser.add_argument("--paragraph_min_tokens", type=int, default=32)
    parser.add_argument("--paragraph_max_tokens", type=int, default=128)
    parser.add_argument("--section_max_paragraphs", type=int, default=8)
    parser.add_argument("--query_window_tokens", type=int, default=64)
    parser.add_argument("--flat_page_counts", default="4,8,16")
    parser.add_argument("--hier_section_counts", default="1,2,4")
    parser.add_argument("--hier_pages_per_section", default="2,4")
    parser.add_argument("--max_query_samples", type=int, default=64)
    parser.add_argument("--query_stride", type=int, default=0)
    parser.add_argument("--write_per_query", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def joined(token_texts: list[str], start: int, end: int) -> str:
    return "".join(token_texts[start:end])


def is_sentence_end(text: str) -> bool:
    stripped = text.strip()
    return "\n\n" in text or stripped in {".", "?", "!", ":", ";"} or any(ch in text for ch in ".?!\n")


def build_sentences(token_texts: list[str], min_sentence_tokens: int) -> list[TextUnit]:
    units: list[TextUnit] = []
    start = 0
    for idx, text in enumerate(token_texts):
        if idx - start + 1 >= min_sentence_tokens and is_sentence_end(text):
            units.append(TextUnit(len(units), "sentence", start, idx + 1))
            start = idx + 1
    if start < len(token_texts):
        units.append(TextUnit(len(units), "sentence", start, len(token_texts)))
    return units


def build_paragraphs(
    token_texts: list[str],
    sentences: list[TextUnit],
    min_tokens: int,
    max_tokens: int,
) -> list[TextUnit]:
    paragraphs: list[TextUnit] = []
    if not sentences:
        return paragraphs
    start_sentence = 0
    current_start = sentences[0].start
    current_end = sentences[0].end
    for pos, sentence in enumerate(sentences):
        if pos == start_sentence:
            continue
        previous_text = token_texts[sentences[pos - 1].end - 1]
        candidate_len = sentence.end - current_start
        boundary_hint = "\n\n" in previous_text or "\n" in previous_text
        boundary = boundary_hint and candidate_len >= min_tokens
        too_long = candidate_len > max_tokens and (current_end - current_start) >= min_tokens
        if boundary or too_long:
            paragraphs.append(TextUnit(len(paragraphs), "paragraph", current_start, current_end))
            start_sentence = pos
            current_start = sentence.start
        current_end = sentence.end
    paragraphs.append(TextUnit(len(paragraphs), "paragraph", current_start, current_end))
    return paragraphs


def build_sections(paragraphs: list[TextUnit], max_paragraphs: int) -> list[TextUnit]:
    sections: list[TextUnit] = []
    for start in range(0, len(paragraphs), max(1, max_paragraphs)):
        group = paragraphs[start : start + max(1, max_paragraphs)]
        if not group:
            continue
        sections.append(TextUnit(len(sections), "section", group[0].start, group[-1].end))
    return sections


def assign_parents(paragraphs: list[TextUnit], sections: list[TextUnit]) -> list[TextUnit]:
    assigned: list[TextUnit] = []
    section_idx = 0
    for paragraph in paragraphs:
        while section_idx + 1 < len(sections) and paragraph.start >= sections[section_idx].end:
            section_idx += 1
        parent = sections[section_idx].unit_id if sections else None
        assigned.append(TextUnit(paragraph.unit_id, paragraph.level, paragraph.start, paragraph.end, parent))
    return assigned


def build_token_to_page(units: list[TextUnit], total_tokens: int) -> list[int]:
    ids = [0] * total_tokens
    for unit in units:
        for idx in range(unit.start, min(unit.end, total_tokens)):
            ids[idx] = unit.unit_id
    return ids


def eligible_pages(units: list[TextUnit], sink: int, remote_end: int) -> list[int]:
    return [unit.unit_id for unit in units if unit.end > sink and unit.start < remote_end]


def selected_page_set(
    scheme: str,
    query_token: int,
    token_texts: list[str],
    paragraph_units: list[TextUnit],
    paragraph_index: SparseTfidfIndex,
    section_units: list[TextUnit],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    sink: int,
    remote_end: int,
    query_window: int,
) -> tuple[set[int], float]:
    query_start = max(0, query_token - query_window)
    query_text = joined(token_texts, query_start, query_token)
    paragraph_query = paragraph_index.query_vector(query_text)
    section_query = section_index.query_vector(query_text)
    candidates = eligible_pages(paragraph_units, sink, remote_end)
    if scheme.startswith("remote_tail_p"):
        page_count = int(scheme.removeprefix("remote_tail_p"))
        tail = sorted(candidates, key=lambda unit_id: paragraph_units[unit_id].end)[-page_count:]
        return set(tail), 0.0
    if scheme.startswith("book_flat_p"):
        page_count = int(scheme.removeprefix("book_flat_p"))
        return {unit_id for unit_id, _ in paragraph_index.topk(paragraph_query, candidates, page_count)}, 0.0

    match = re.fullmatch(r"book_hier_s(\d+)_p(\d+)", scheme)
    if not match:
        raise ValueError(f"Unknown scheme: {scheme}")
    section_count = int(match.group(1))
    pages_per_section = int(match.group(2))
    section_candidates = [
        section.unit_id for section in section_units if section.end > sink and section.start < remote_end
    ]
    top_sections = section_index.topk(section_query, section_candidates, section_count)
    pages: set[int] = set()
    for section_id, _ in top_sections:
        page_candidates = [
            page_id
            for page_id in section_to_pages.get(section_id, [])
            if paragraph_units[page_id].end > sink and paragraph_units[page_id].start < remote_end
        ]
        pages.update(unit_id for unit_id, _ in paragraph_index.topk(paragraph_query, page_candidates, pages_per_section))
    return pages, float(len(top_sections))


class HierarchicalBookIndexCollector:
    def __init__(
        self,
        query_tokens: set[int],
        top_fraction: float,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        fixed_page_size: int,
        token_texts: list[str],
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
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.fixed_page_size = fixed_page_size
        self.token_texts = token_texts
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
        self.observed_query_tokens: set[int] = set()
        self.recall_by_scope: dict[tuple[str, str, tuple[int, ...]], RecallAccumulator] = defaultdict(
            RecallAccumulator
        )
        self.per_query_rows: list[dict[str, Any]] = []

    def _fixed_page(self, token_index: int) -> int:
        return token_index // self.fixed_page_size

    def _paragraph_page(self, token_index: int) -> int:
        return self.paragraph_page_ids[token_index]

    def _add_recall(
        self,
        scheme: str,
        layer: int,
        head: int,
        selected_pages: int,
        semantic_events: int,
        semantic_mass: float,
        covered_events: int,
        covered_mass: float,
    ) -> None:
        for scope, key in [("overall", ()), ("layer", (layer,)), ("layer_head", (layer, head))]:
            self.recall_by_scope[(scheme, scope, key)].add(
                selected_pages,
                selected_pages,
                semantic_events,
                semantic_mass,
                covered_events,
                covered_mass,
            )

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
        self.observed_query_tokens.add(query_token)

        retrieval_pages_by_scheme = {
            scheme: selected_page_set(
                scheme,
                query_token,
                self.token_texts,
                self.paragraph_units,
                self.paragraph_index,
                self.section_units,
                self.section_index,
                self.section_to_pages,
                self.exclude_sink_tokens,
                remote_end,
                self.query_window_tokens,
            )[0]
            for scheme in self.retrieval_schemes
        }

        for head in range(top_indices.shape[0]):
            head_indices = top_indices[head, remote_mask[head]].detach().cpu().tolist()
            head_masses = top_masses[head, remote_mask[head]].detach().cpu().tolist()
            if not head_indices:
                continue
            structural_tokens: list[int] = []
            semantic_tokens: list[tuple[int, float]] = []
            for token_index, mass in zip(head_indices, head_masses):
                anchor = self.anchor_types[int(token_index)]
                if anchor == "structural":
                    structural_tokens.append(int(token_index))
                elif anchor == "semantic":
                    semantic_tokens.append((int(token_index), float(mass)))

            semantic_events = len(semantic_tokens)
            semantic_mass = sum(mass for _, mass in semantic_tokens)
            baseline_pages = {
                "fixed_anchor": {self._fixed_page(token_index) for token_index in structural_tokens},
                "paragraph_anchor": {self._paragraph_page(token_index) for token_index in structural_tokens},
            }
            for scheme, pages in {**baseline_pages, **retrieval_pages_by_scheme}.items():
                if scheme == "fixed_anchor":
                    page_of = self._fixed_page
                else:
                    page_of = self._paragraph_page
                covered = [(idx, mass) for idx, mass in semantic_tokens if page_of(idx) in pages]
                covered_events = len(covered)
                covered_mass = sum(mass for _, mass in covered)
                self._add_recall(
                    scheme,
                    layer,
                    head,
                    len(pages),
                    semantic_events,
                    semantic_mass,
                    covered_events,
                    covered_mass,
                )
                if self.write_per_query:
                    self.per_query_rows.append(
                        {
                            "query_token": query_token,
                            "layer": layer,
                            "head": head,
                            "scheme": scheme,
                            "selected_pages": len(pages),
                            "semantic_events": semantic_events,
                            "semantic_mass": semantic_mass,
                            "covered_semantic_events": covered_events,
                            "covered_semantic_mass": covered_mass,
                            "semantic_event_recall": covered_events / semantic_events if semantic_events else 0.0,
                            "semantic_mass_recall": covered_mass / semantic_mass if semantic_mass else 0.0,
                        }
                    )

    def recall_rows(self, scope: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (scheme, row_scope, key), acc in sorted(self.recall_by_scope.items()):
            if row_scope != scope:
                continue
            extra: dict[str, Any] = {"scheme": scheme, "scope": row_scope}
            if scope in {"layer", "layer_head"}:
                extra["layer"] = key[0]
            if scope == "layer_head":
                extra["head"] = key[1]
            rows.append(acc.row(extra))
        return rows


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: HierarchicalBookIndexCollector,
) -> None:
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
            end = min(start + chunk_size, eval_end)
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    flat_counts = [int(part) for part in args.flat_page_counts.split(",") if part.strip()]
    section_counts = [int(part) for part in args.hier_section_counts.split(",") if part.strip()]
    pages_per_section = [int(part) for part in args.hier_pages_per_section.split(",") if part.strip()]
    retrieval_schemes = [f"remote_tail_p{count}" for count in flat_counts]
    retrieval_schemes += [f"book_flat_p{count}" for count in flat_counts]
    retrieval_schemes += [f"book_hier_s{s}_p{p}" for s in section_counts for p in pages_per_section]

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if args.require_total_tokens and len(token_ids) < args.total_tokens:
        raise ValueError(f"Need {args.total_tokens} tokens, got {len(token_ids)}.")
    token_ids = token_ids[: args.total_tokens]
    token_texts = [token_text(tokenizer, token_id) for token_id in token_ids]
    anchor_types = [anchor_type(text_piece) for text_piece in token_texts]

    sentences = build_sentences(token_texts, args.min_sentence_tokens)
    paragraphs = build_paragraphs(token_texts, sentences, args.paragraph_min_tokens, args.paragraph_max_tokens)
    sections = build_sections(paragraphs, args.section_max_paragraphs)
    paragraphs = assign_parents(paragraphs, sections)
    paragraph_page_ids = build_token_to_page(paragraphs, len(token_ids))
    paragraph_docs = [joined(token_texts, unit.start, unit.end) for unit in paragraphs]
    section_docs = [joined(token_texts, unit.start, unit.end) for unit in sections]
    paragraph_index = SparseTfidfIndex(paragraph_docs)
    section_index = SparseTfidfIndex(section_docs)
    section_to_pages: dict[int, list[int]] = defaultdict(list)
    for paragraph in paragraphs:
        if paragraph.parent_id is not None:
            section_to_pages[paragraph.parent_id].append(paragraph.unit_id)

    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)
    collector = HierarchicalBookIndexCollector(
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        fixed_page_size=args.fixed_page_size,
        token_texts=token_texts,
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

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    seconds = time.perf_counter() - started

    recall_fields = [
        "scheme",
        "scope",
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
    ]
    write_csv(output_dir / "book_index_recall_summary.csv", collector.recall_rows("overall"), recall_fields)
    write_csv(output_dir / "book_index_recall_by_layer.csv", collector.recall_rows("layer"), ["layer"] + recall_fields)
    write_csv(
        output_dir / "book_index_recall_by_layer_head.csv",
        collector.recall_rows("layer_head"),
        ["layer", "head"] + recall_fields,
    )
    if args.write_per_query:
        write_csv(
            output_dir / "per_query_book_index_recall.csv",
            collector.per_query_rows,
            [
                "query_token",
                "layer",
                "head",
                "scheme",
                "selected_pages",
                "semantic_events",
                "semantic_mass",
                "covered_semantic_events",
                "covered_semantic_mass",
                "semantic_event_recall",
                "semantic_mass_recall",
            ],
        )

    index_rows = []
    for unit in paragraphs[:200]:
        index_rows.append(
            {
                "unit_id": unit.unit_id,
                "level": unit.level,
                "parent_id": unit.parent_id,
                "start": unit.start,
                "end": unit.end,
                "tokens": unit.length,
                "summary_terms": " ".join(paragraph_index.summary_terms(unit.unit_id, 12)),
            }
        )
    write_csv(
        output_dir / "index_sample.csv",
        index_rows,
        ["unit_id", "level", "parent_id", "start", "end", "tokens", "summary_terms"],
    )

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": sorted(collector.observed_query_tokens),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "section_count": len(sections),
            "paragraph_mean_tokens": (
                sum(unit.length for unit in paragraphs) / len(paragraphs) if paragraphs else 0.0
            ),
            "section_mean_tokens": sum(unit.length for unit in sections) / len(sections) if sections else 0.0,
            "retrieval_schemes": retrieval_schemes,
            "seconds": seconds,
            "metric_definitions": {
                "book_flat_pK": "Runtime lexical TF-IDF retrieval of top K paragraph pages from current recent query text.",
                "book_hier_sS_pP": (
                    "Runtime top-down lexical routing: top S sections, then top P paragraph pages per selected section."
                ),
                "fixed_anchor/paragraph_anchor": (
                    "Oracle-style anchor baseline using true selected structural top2 tokens to choose fixed or paragraph pages."
                ),
            },
        },
        "paths": {
            "summary": str(output_dir / "book_index_recall_summary.csv"),
            "by_layer": str(output_dir / "book_index_recall_by_layer.csv"),
            "by_layer_head": str(output_dir / "book_index_recall_by_layer_head.csv"),
            "per_query": str(output_dir / "per_query_book_index_recall.csv") if args.write_per_query else None,
            "index_sample": str(output_dir / "index_sample.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
