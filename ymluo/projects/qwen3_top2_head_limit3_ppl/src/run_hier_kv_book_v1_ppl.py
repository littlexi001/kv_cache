from __future__ import annotations

import argparse
import csv
import json
import math
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
    TextUnit,
    assign_parents,
    build_paragraphs,
    build_sections,
    build_sentences,
    joined,
)
from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    clone_past_key_values,
    model_forward,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_MODE = "baseline"
_ACTIVE_BOOK_STATE: "BookKVState | None" = None
_ACTIVE_SINK_TOKENS = 64
_ACTIVE_RECENT_TOKENS = 512
_ACTIVE_ALWAYS_KEEP_SELF = True


@dataclass(frozen=True)
class BookKVConfig:
    sink_tokens: int
    recent_tokens: int
    top_sections: int
    pages_per_section: int
    seed_pages: int
    tail_pages: int
    route_refresh_tokens: int


class BookKVState:
    def __init__(
        self,
        pages: list[TextUnit],
        sections: list[TextUnit],
        section_to_pages: dict[int, list[int]],
        config: BookKVConfig,
    ) -> None:
        self.pages = pages
        self.sections = sections
        self.section_to_pages = section_to_pages
        self.config = config
        self.page_parent = torch.tensor([int(page.parent_id or 0) for page in pages], dtype=torch.long)
        self.page_starts = [int(page.start) for page in pages]
        self.page_ends = [int(page.end) for page in pages]
        self.section_starts = [int(section.start) for section in sections]
        self.section_ends = [int(section.end) for section in sections]
        self._summaries: dict[tuple[int, str, torch.device], torch.Tensor] = {}
        self._last_selected: dict[tuple[int, int, int], list[int]] = {}
        self.selected_page_counts: list[int] = []
        self.selected_token_counts: list[int] = []
        self.route_compute_calls = 0
        self.route_apply_calls = 0

    def _range_mean(self, key_states: torch.Tensor, ranges: list[tuple[int, int]], history_count: int) -> torch.Tensor:
        parts = []
        for start, end in ranges:
            start = max(0, min(start, history_count))
            end = max(start + 1, min(end, history_count))
            parts.append(key_states[:, :, start:end, :].float().mean(dim=2))
        if not parts:
            batch, heads, _, dim = key_states.shape
            return torch.empty((batch, heads, 0, dim), dtype=torch.float32, device=key_states.device)
        return torch.stack(parts, dim=2)

    def page_summaries(self, layer_idx: int, key_states: torch.Tensor, history_count: int) -> torch.Tensor:
        cache_key = (layer_idx, "page", key_states.device)
        cached = self._summaries.get(cache_key)
        if cached is not None:
            return cached
        ranges = list(zip(self.page_starts, self.page_ends))
        summaries = self._range_mean(key_states, ranges, history_count)
        self._summaries[cache_key] = summaries
        return summaries

    def section_summaries(self, layer_idx: int, key_states: torch.Tensor, history_count: int) -> torch.Tensor:
        cache_key = (layer_idx, "section", key_states.device)
        cached = self._summaries.get(cache_key)
        if cached is not None:
            return cached
        ranges = list(zip(self.section_starts, self.section_ends))
        summaries = self._range_mean(key_states, ranges, history_count)
        self._summaries[cache_key] = summaries
        return summaries

    def _eligible_pages(self, remote_end: int) -> list[int]:
        return [
            page.unit_id
            for page in self.pages
            if page.end > self.config.sink_tokens and page.start < remote_end
        ]

    def _tail_pages(self, remote_end: int) -> list[int]:
        if self.config.tail_pages <= 0:
            return []
        candidates = self._eligible_pages(remote_end)
        return sorted(candidates, key=lambda page_id: (self.pages[page_id].end, page_id))[-self.config.tail_pages :]

    def selected_pages_for_head(
        self,
        layer_idx: int,
        head_idx: int,
        query_token: int,
        page_scores: torch.Tensor,
        section_scores: torch.Tensor,
        remote_end: int,
    ) -> list[int]:
        refresh = max(1, self.config.route_refresh_tokens)
        cache_key = (layer_idx, head_idx, query_token // refresh)
        cached = self._last_selected.get(cache_key)
        if cached is not None:
            return cached
        self.route_compute_calls += 1

        eligible = self._eligible_pages(remote_end)
        if not eligible:
            self._last_selected[cache_key] = []
            return []

        eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=page_scores.device)
        selected: set[int] = set()
        if self.config.seed_pages > 0:
            k = min(self.config.seed_pages, eligible_tensor.numel())
            seed_scores = page_scores.index_select(0, eligible_tensor)
            seed_pos = torch.topk(seed_scores, k=k, largest=True).indices
            selected.update(int(eligible_tensor[pos].item()) for pos in seed_pos)

        section_count = min(max(1, self.config.top_sections), section_scores.numel())
        top_sections = torch.topk(section_scores, k=section_count, largest=True).indices.tolist()
        eligible_set = set(eligible)
        for section_id in top_sections:
            page_ids = [page_id for page_id in self.section_to_pages.get(int(section_id), []) if page_id in eligible_set]
            if not page_ids:
                continue
            page_tensor = torch.tensor(page_ids, dtype=torch.long, device=page_scores.device)
            k = min(max(1, self.config.pages_per_section), page_tensor.numel())
            scores = page_scores.index_select(0, page_tensor)
            pos = torch.topk(scores, k=k, largest=True).indices
            selected.update(int(page_tensor[item].item()) for item in pos)

        selected.update(self._tail_pages(remote_end))
        result = sorted(selected, key=lambda page_id: (self.pages[page_id].start, page_id))
        self._last_selected[cache_key] = result
        return result

    def record_selection(self, selected: list[int]) -> None:
        self.selected_page_counts.append(len(selected))
        self.selected_token_counts.append(sum(self.pages[page_id].length for page_id in selected))
        self.route_apply_calls += 1

    def stats(self) -> dict[str, float]:
        def avg(values: list[int]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return {
            "avg_selected_pages": avg(self.selected_page_counts),
            "avg_selected_page_tokens": avg(self.selected_token_counts),
            "route_compute_calls": float(self.route_compute_calls),
            "route_apply_calls": float(self.route_apply_calls),
        }


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
    parser = argparse.ArgumentParser(description="Hierarchical natural-page KV memory v1 PPL evaluator.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--text_path", default="data/war_and_peace_pg2600.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=20_000)
    parser.add_argument("--eval_tokens", type=int, default=5_000)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--modes", default="recent,bookkv")
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--top_sections", type=int, default=1)
    parser.add_argument("--pages_per_section", type=int, default=1)
    parser.add_argument("--seed_pages", type=int, default=1)
    parser.add_argument("--tail_pages", type=int, default=0)
    parser.add_argument("--route_refresh_tokens", type=int, default=16)
    parser.add_argument("--min_sentence_tokens", type=int, default=8)
    parser.add_argument("--paragraph_min_tokens", type=int, default=64)
    parser.add_argument("--paragraph_max_tokens", type=int, default=192)
    parser.add_argument("--section_max_paragraphs", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--reuse_prefill_cache", type=str2bool, default=True)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def decode_token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def build_book_units(
    tokenizer: Any,
    token_ids: list[int],
    prefill_tokens: int,
    min_sentence_tokens: int,
    paragraph_min_tokens: int,
    paragraph_max_tokens: int,
    section_max_paragraphs: int,
) -> tuple[list[TextUnit], list[TextUnit], dict[int, list[int]], list[str]]:
    prefix_ids = token_ids[:prefill_tokens]
    token_texts = decode_token_texts(tokenizer, prefix_ids)
    sentences = build_sentences(token_texts, min_sentence_tokens)
    pages = build_paragraphs(token_texts, sentences, paragraph_min_tokens, paragraph_max_tokens)
    sections = build_sections(pages, section_max_paragraphs)
    pages = assign_parents(pages, sections)
    section_to_pages: dict[int, list[int]] = defaultdict(list)
    for page in pages:
        if page.parent_id is not None:
            section_to_pages[int(page.parent_id)].append(page.unit_id)
    return pages, sections, dict(section_to_pages), token_texts


def _indices_from_keep_mask(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = keep.sum(dim=-1)
    max_selected = int(counts.max().item()) if counts.numel() else 0
    if max_selected <= 0:
        empty = torch.zeros((*keep.shape[:-1], 0), dtype=torch.long, device=keep.device)
        return empty, torch.zeros_like(empty, dtype=torch.bool)
    indices = torch.zeros((*keep.shape[:-1], max_selected), dtype=torch.long, device=keep.device)
    valid = torch.zeros((*keep.shape[:-1], max_selected), dtype=torch.bool, device=keep.device)
    for idx in torch.nonzero(counts > 0, as_tuple=False).tolist():
        selected = torch.nonzero(keep[tuple(idx)], as_tuple=False).flatten()
        count = selected.numel()
        indices[tuple(idx)][0:count] = selected
        valid[tuple(idx)][0:count] = True
    return indices, valid


def _selected_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    final_keep: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, _, head_dim = query_states.shape
    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=gather)
    selected_values = torch.gather(value_states, dim=2, index=gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        key_count = key_states.shape[-2]
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = torch.sum(weights[:, :, :, None] * selected_values, dim=2)
    return output[:, None, :, :].contiguous(), None


def _selected_attention_from_indices(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    final_indices: torch.Tensor,
    final_valid: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    head_dim = query_states.shape[-1]
    head_count = query_states.shape[1]
    gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=gather)
    selected_values = torch.gather(value_states, dim=2, index=gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        key_count = key_states.shape[-2]
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output = torch.sum(weights[:, :, :, None] * selected_values, dim=2)
    return output[:, None, :, :].contiguous(), None


def _padded_indices_from_lists(index_lists: list[list[list[int]]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count = len(index_lists)
    head_count = len(index_lists[0]) if batch_count else 0
    max_len = max((len(values) for batch in index_lists for values in batch), default=1)
    final_indices = torch.zeros((batch_count, head_count, max_len), dtype=torch.long, device=device)
    final_valid = torch.zeros((batch_count, head_count, max_len), dtype=torch.bool, device=device)
    for batch_idx, batch in enumerate(index_lists):
        for head_idx, values in enumerate(batch):
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.long, device=device)
            final_indices[batch_idx, head_idx, : tensor.numel()] = tensor
            final_valid[batch_idx, head_idx, : tensor.numel()] = True
    return final_indices, final_valid


def _unique_sorted_indices(parts: list[range], self_index: int) -> list[int]:
    seen: set[int] = set()
    values: list[int] = []
    for part in parts:
        for idx in part:
            if idx not in seen:
                seen.add(idx)
                values.append(idx)
    if self_index not in seen:
        values.append(self_index)
    values.sort()
    return values


def _bookkv_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    if _ACTIVE_BOOK_STATE is None:
        raise RuntimeError("bookkv mode requires active BookKVState.")
    if query_states.shape[-2] != 1:
        raise RuntimeError("bookkv requires token-by-token eval; set --eval_chunk_size 1.")
    batch_count, head_count, _, _ = query_states.shape
    key_count = key_states.shape[-2]
    if key_count <= 1:
        return value_states[:, :, -1:, :].transpose(1, 2).contiguous(), None
    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    query_token = history_count
    cfg = _ACTIVE_BOOK_STATE.config
    remote_end = max(cfg.sink_tokens, history_count - cfg.recent_tokens)
    index_lists: list[list[list[int]]] = [[[] for _ in range(head_count)] for _ in range(batch_count)]
    refresh = max(1, cfg.route_refresh_tokens)
    route_bucket = query_token // refresh
    cached_by_head: dict[tuple[int, int], list[int]] = {}
    missing_heads: list[tuple[int, int]] = []
    for batch_idx in range(batch_count):
        for head_idx in range(head_count):
            cached = _ACTIVE_BOOK_STATE._last_selected.get((layer_idx, head_idx, route_bucket))
            if cached is None:
                missing_heads.append((batch_idx, head_idx))
            else:
                cached_by_head[(batch_idx, head_idx)] = cached
    page_scores = None
    section_scores = None
    if missing_heads:
        page_summaries = _ACTIVE_BOOK_STATE.page_summaries(layer_idx, key_states, history_count)
        section_summaries = _ACTIVE_BOOK_STATE.section_summaries(layer_idx, key_states, history_count)
        q = query_states[:, :, 0, :].float()
        page_scores = torch.einsum("bhd,bhpd->bhp", q, page_summaries)
        section_scores = torch.einsum("bhd,bhsd->bhs", q, section_summaries)
    for batch_idx in range(batch_count):
        for head_idx in range(head_count):
            selected_pages = cached_by_head.get((batch_idx, head_idx))
            if selected_pages is None:
                if page_scores is None or section_scores is None:
                    raise RuntimeError("Missing book route scores for uncached head.")
                selected_pages = _ACTIVE_BOOK_STATE.selected_pages_for_head(
                    layer_idx,
                    head_idx,
                    query_token,
                    page_scores[batch_idx, head_idx],
                    section_scores[batch_idx, head_idx],
                    remote_end,
                )
            _ACTIVE_BOOK_STATE.record_selection(selected_pages)
            parts: list[range] = []
            if cfg.sink_tokens > 0:
                parts.append(range(0, min(cfg.sink_tokens, history_count)))
            for page_id in selected_pages:
                page = _ACTIVE_BOOK_STATE.pages[page_id]
                start = max(0, min(int(page.start), history_count))
                end = max(start, min(int(page.end), history_count))
                if start < end:
                    parts.append(range(start, end))
            if cfg.recent_tokens > 0:
                parts.append(range(max(0, history_count - cfg.recent_tokens), history_count))
            index_lists[batch_idx][head_idx] = _unique_sorted_indices(parts, history_count)
    final_indices, final_valid = _padded_indices_from_lists(index_lists, query_states.device)
    return _selected_attention_from_indices(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )


def _recent_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    if query_states.shape[-2] != 1:
        raise RuntimeError("recent mode requires token-by-token eval; set --eval_chunk_size 1.")
    batch_count, head_count, _, _ = query_states.shape
    key_count = key_states.shape[-2]
    history_count = key_count - 1
    parts = []
    if _ACTIVE_SINK_TOKENS > 0:
        parts.append(range(0, min(_ACTIVE_SINK_TOKENS, history_count)))
    if _ACTIVE_RECENT_TOKENS > 0:
        parts.append(range(max(0, history_count - _ACTIVE_RECENT_TOKENS), history_count))
    values = _unique_sorted_indices(parts, history_count)
    final_indices = torch.tensor(values, dtype=torch.long, device=query_states.device).view(1, 1, -1)
    final_indices = final_indices.expand(batch_count, head_count, -1).contiguous()
    final_valid = torch.ones_like(final_indices, dtype=torch.bool)
    return _selected_attention_from_indices(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )


def _hier_book_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    if _ACTIVE_MODE == "bookkv":
        return _bookkv_attention_forward(module, query_states, key_states, value_states, attention_mask, scaling)
    if _ACTIVE_MODE == "recent":
        return _recent_attention_forward(query_states, key_states, value_states, attention_mask, scaling)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        weights = F.dropout(weights, p=dropout, training=True)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _hier_book_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _hier_book_attention_forward


class active_mode:
    def __init__(self, mode: str, book_state: BookKVState | None, sink_tokens: int, recent_tokens: int) -> None:
        self.mode = mode
        self.book_state = book_state
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.previous: tuple[str, BookKVState | None, int, int] | None = None

    def __enter__(self) -> None:
        global _ACTIVE_MODE, _ACTIVE_BOOK_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS
        self.previous = (_ACTIVE_MODE, _ACTIVE_BOOK_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS)
        _ACTIVE_MODE = self.mode
        _ACTIVE_BOOK_STATE = self.book_state
        _ACTIVE_SINK_TOKENS = self.sink_tokens
        _ACTIVE_RECENT_TOKENS = self.recent_tokens

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_MODE, _ACTIVE_BOOK_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS
        if self.previous is not None:
            _ACTIVE_MODE, _ACTIVE_BOOK_STATE, _ACTIVE_SINK_TOKENS, _ACTIVE_RECENT_TOKENS = self.previous


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    eval_chunk_size: int,
    input_device: torch.device,
    mode: str,
    book_state: BookKVState | None,
    sink_tokens: int,
    recent_tokens: int,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
    clone_initial_cache: bool,
    log_every: int,
) -> tuple[float, float, int, float]:
    if clone_initial_cache:
        print(f"cloning shared prefill cache for mode: {mode}", flush=True)
        past_key_values = clone_past_key_values(initial_past_key_values)
        prev_logits = initial_prev_logits.detach().clone()
    else:
        past_key_values = initial_past_key_values
        prev_logits = initial_prev_logits
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / eval_chunk_size)
    started = time.perf_counter()
    with active_mode(mode, book_state, sink_tokens, recent_tokens):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, eval_chunk_size), start=1):
            end = min(start + eval_chunk_size, eval_end)
            chunk = input_ids[:, start:end].to(input_device)
            kwargs: dict[str, Any] = {
                "input_ids": chunk,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            outputs = model_forward(model, kwargs)
            logits = outputs.logits
            shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
                chunk.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_count += int(chunk.numel())
            prev_logits = logits[:, -1, :].detach()
            past_key_values = outputs.past_key_values
            if log_every > 0 and (chunk_idx % log_every == 0 or chunk_idx == total_chunks):
                ppl_so_far = math.exp(total_loss / max(1, total_count))
                print(
                    f"{mode} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1} "
                    f"ppl_so_far={ppl_so_far:.4f}",
                    flush=True,
                )
            del outputs, chunk, logits, shifted_logits, loss
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
    seconds = time.perf_counter() - started
    loss_value = total_loss / max(1, total_count)
    return loss_value, math.exp(loss_value), total_count, seconds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    total_needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < total_needed:
        raise RuntimeError(f"Need {total_needed} tokens, got {len(token_ids)}")
    token_ids = token_ids[:total_needed]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    print("building natural book units...", flush=True)
    pages, sections, section_to_pages, token_texts = build_book_units(
        tokenizer,
        token_ids,
        args.prefill_tokens,
        args.min_sentence_tokens,
        args.paragraph_min_tokens,
        args.paragraph_max_tokens,
        args.section_max_paragraphs,
    )

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    install_attention_patch()
    input_device = pick_input_device(model, requested_device)

    print("starting shared full prefill cache", flush=True)
    with active_mode("baseline", None, args.sink_tokens, args.recent_tokens):
        prefill_start = time.perf_counter()
        shared_past, shared_prev_logits = prefill_cache(
            model,
            input_ids,
            args.prefill_tokens,
            args.chunk_size,
            input_device,
        )
        shared_prefill_seconds = time.perf_counter() - prefill_start
    print(f"shared prefill cache ready: {shared_prefill_seconds:.3f}s", flush=True)

    rows: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for idx, mode in enumerate(modes):
        if mode not in {"baseline", "recent", "bookkv"}:
            raise ValueError(f"Unknown mode: {mode}")
        book_state = None
        if mode == "bookkv":
            book_state = BookKVState(
                pages,
                sections,
                section_to_pages,
                BookKVConfig(
                    sink_tokens=args.sink_tokens,
                    recent_tokens=args.recent_tokens,
                    top_sections=args.top_sections,
                    pages_per_section=args.pages_per_section,
                    seed_pages=args.seed_pages,
                    tail_pages=args.tail_pages,
                    route_refresh_tokens=args.route_refresh_tokens,
                ),
            )
        loss, ppl, token_count, seconds = compute_eval_loss(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.eval_chunk_size,
            input_device,
            mode,
            book_state,
            args.sink_tokens,
            args.recent_tokens,
            shared_past,
            shared_prev_logits,
            clone_initial_cache=len(modes) > 1,
            log_every=args.log_every,
        )
        book_stats = book_state.stats() if book_state is not None else {}
        rows.append(
            {
                "mode": mode,
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "shared_prefill_seconds": shared_prefill_seconds,
                "sink_tokens": args.sink_tokens,
                "recent_tokens": args.recent_tokens,
                "top_sections": args.top_sections if mode == "bookkv" else "",
                "pages_per_section": args.pages_per_section if mode == "bookkv" else "",
                "seed_pages": args.seed_pages if mode == "bookkv" else "",
                "tail_pages": args.tail_pages if mode == "bookkv" else "",
                "route_refresh_tokens": args.route_refresh_tokens if mode == "bookkv" else "",
                "avg_selected_pages": book_stats.get("avg_selected_pages", ""),
                "avg_selected_page_tokens": book_stats.get("avg_selected_page_tokens", ""),
                "route_compute_calls": book_stats.get("route_compute_calls", ""),
                "route_apply_calls": book_stats.get("route_apply_calls", ""),
            }
        )
    write_csv(output_dir / "ppl_by_mode.csv", rows)
    page_examples = [
        {
            "page_id": page.unit_id,
            "start": page.start,
            "end": page.end,
            "parent_section": page.parent_id,
            "text": joined(token_texts, page.start, page.end),
        }
        for page in pages[:10]
    ]
    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_used": len(token_ids),
            "paragraph_pages": len(pages),
            "sections": len(sections),
            "paragraph_mean_tokens": sum(page.length for page in pages) / len(pages) if pages else 0.0,
            "method": (
                "KV-level hierarchical book memory: natural paragraph/section ranges, per-layer/head K mean summaries, "
                "query-to-section/page routing, attention over sink+recent+routed original KV ranges."
            ),
        },
        "paths": {"ppl_by_mode": str(output_dir / "ppl_by_mode.csv")},
        "page_examples": page_examples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
