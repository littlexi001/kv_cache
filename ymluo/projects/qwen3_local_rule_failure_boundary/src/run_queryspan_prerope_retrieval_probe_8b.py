from __future__ import annotations

"""Query-span pre-RoPE retrieval with an unchanged native RoPE consumer.

This is deliberately a narrow, falsifiable probe.  The selector may read only
pre-RoPE queries from the prompt's question span and pre-RoPE history keys.
Gold/conflict labels are passed to the metric recorder *after* support has been
selected.  Selected tokens are always consumed with their original post-RoPE
QK scores and original values.
"""

import argparse
import csv
import gc
import json
import math
import os
import statistics
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

import run_local_global_rope_probe_8b as runner
import run_local_rule_failure_boundary as base
import run_rope_retrieval_repair_8b as rope_repair
import run_suppression_certificate_safety_probe_8b as safety


VARIANTS = (
    "native_noop",
    "exact_final_pre_top2_postscore",
    "queryspan_block_top2_postscore",
    "queryspan_tokenmax_top2_postscore",
)

_ACTIVE_CONTROLLER: "QuerySpanController | None" = None
_CAPTURE_PREFIX = False
_CAPTURE_ANCHOR_POSITIONS: tuple[int, ...] = ()
_PREFIX_KEY_STORAGE = os.environ.get("QUERYSPAN_PREKEY_STORAGE", "cuda").lower()
if _PREFIX_KEY_STORAGE not in {"cuda", "cpu"}:
    raise ValueError("QUERYSPAN_PREKEY_STORAGE must be 'cuda' or 'cpu'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final-query and query-span pre-RoPE selectors while "
            "retaining native post-RoPE attention consumption."
        )
    )
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-shards", default="")
    parser.add_argument("--lengths", default="8192,32768,65536")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--minimum-keep-tokens", type=int, default=0)
    parser.add_argument("--maximum-keep-tokens", type=int, default=0)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--query-anchor-count", type=int, default=16)
    parser.add_argument("--score-chunk-blocks", type=int, default=32)
    parser.add_argument("--class-sample-count", type=int, default=8)
    parser.add_argument("--packet-gap-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=70000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_query_anchor_positions(
    query_span: Sequence[int], prefix_length: int, anchor_count: int
) -> tuple[int, ...]:
    """Choose fixed, label-free anchors evenly across the visible question."""

    if len(query_span) != 2:
        raise ValueError("query_span must be [start, end]")
    start = max(0, int(query_span[0]))
    end = min(int(prefix_length), int(query_span[1]))
    if start >= end:
        raise ValueError(f"empty visible question span: {query_span}, prefix={prefix_length}")
    count = min(end - start, max(1, int(anchor_count)))
    if count == 1:
        return (end - 1,)
    values = [round(start + index * (end - start - 1) / (count - 1)) for index in range(count)]
    return tuple(dict.fromkeys(int(value) for value in values))


@dataclass(frozen=True)
class BudgetLayout:
    key_count: int
    keep_count: int
    sink_count: int
    remote_start: int
    remote_end: int
    remote_count: int
    local_start: int
    local_count: int
    current: int


def compute_budget_layout(
    key_count: int,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
) -> BudgetLayout:
    if key_count <= 0:
        raise ValueError("key_count must be positive")
    keep = min(int(key_count), max(1, int(keep_count)))
    current = int(key_count) - 1
    history_budget = min(current, keep - 1)
    local_count = min(max(0, int(local_window)), history_budget)
    local_start = current - local_count
    remaining = history_budget - local_count
    sink_count = min(max(0, int(sink_tokens)), remaining, max(0, local_start))
    remaining -= sink_count
    remote_start = sink_count
    remote_end = local_start
    remote_available = max(0, remote_end - remote_start)
    remote_count = min(remaining, remote_available)
    if remaining != remote_count:
        # This can only happen for tiny sequences where sink/local exhaust the
        # disjoint history.  Fill from the remote interval by construction.
        raise RuntimeError("budget partition could not allocate every history token")
    return BudgetLayout(
        key_count=int(key_count),
        keep_count=keep,
        sink_count=sink_count,
        remote_start=remote_start,
        remote_end=remote_end,
        remote_count=remote_count,
        local_start=local_start,
        local_count=local_count,
        current=current,
    )


def fixed_positions(layout: BudgetLayout, heads: int, device: torch.device) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    if layout.sink_count:
        pieces.append(
            torch.arange(layout.sink_count, device=device, dtype=torch.long)
            .view(1, -1)
            .expand(heads, -1)
        )
    if layout.local_count:
        pieces.append(
            torch.arange(layout.local_start, layout.current, device=device, dtype=torch.long)
            .view(1, -1)
            .expand(heads, -1)
        )
    pieces.append(torch.full((heads, 1), layout.current, device=device, dtype=torch.long))
    return pieces


def exact_final_pre_selection(
    pre_scores: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, BudgetLayout]:
    """Exact final-query pre-RoPE Top-K under the shared fixed-region budget."""

    if pre_scores.dim() != 2:
        raise ValueError("pre_scores must be [query_heads, keys]")
    heads, key_count = map(int, pre_scores.shape)
    layout = compute_budget_layout(key_count, keep_count, local_window, sink_tokens)
    pieces: list[torch.Tensor] = []
    if layout.sink_count:
        pieces.append(
            torch.arange(layout.sink_count, device=pre_scores.device)
            .view(1, -1)
            .expand(heads, -1)
        )
    if layout.remote_count:
        remote = torch.topk(
            pre_scores[:, layout.remote_start : layout.remote_end].float(),
            k=layout.remote_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices + layout.remote_start
        pieces.append(remote)
    if layout.local_count:
        pieces.append(
            torch.arange(layout.local_start, layout.current, device=pre_scores.device)
            .view(1, -1)
            .expand(heads, -1)
        )
    pieces.append(
        torch.full((heads, 1), layout.current, device=pre_scores.device, dtype=torch.long)
    )
    selected = torch.cat(pieces, dim=-1)
    if int(selected.shape[-1]) != layout.keep_count:
        raise AssertionError("exact selector violated token budget")
    return selected, layout


def _gqa_groups(query: torch.Tensor, key_or_value: torch.Tensor) -> int:
    if query.dim() != 4 or key_or_value.dim() != 4:
        raise ValueError("GQA tensors must be [batch, heads, tokens, dim]")
    query_heads = int(query.shape[1])
    kv_heads = int(key_or_value.shape[1])
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[0] != key_or_value.shape[0] or query.shape[-1] != key_or_value.shape[-1]:
        raise ValueError("incompatible GQA tensors")
    return query_heads // kv_heads


def gqa_query_key_scores(
    query: torch.Tensor, key: torch.Tensor, score_scale: float = 1.0
) -> torch.Tensor:
    groups = _gqa_groups(query, key)
    batch, query_heads, query_tokens, dim = query.shape
    kv_heads = int(key.shape[1])
    grouped_query = query.reshape(batch, kv_heads, groups, query_tokens, dim)
    scores = torch.einsum("bmgqd,bmkd->bmgqk", grouped_query, key)
    return scores.reshape(batch, query_heads, query_tokens, key.shape[-2]) * float(score_scale)


def gather_per_query_head_gqa(
    values: torch.Tensor, positions: torch.Tensor, groups: int
) -> torch.Tensor:
    batch, kv_heads, key_count, dim = values.shape
    query_heads = kv_heads * int(groups)
    if positions.dim() != 2 or int(positions.shape[0]) != query_heads:
        raise ValueError("positions must be [query_heads, selected]")
    selected = int(positions.shape[1])
    expanded = values.unsqueeze(2).expand(batch, kv_heads, groups, key_count, dim)
    index = positions.reshape(kv_heads, groups, selected).view(
        1, kv_heads, groups, selected, 1
    ).expand(batch, kv_heads, groups, selected, dim)
    return expanded.gather(3, index).reshape(batch, query_heads, selected, dim)


def queryspan_block_scores(
    anchor_queries: torch.Tensor,
    key_pre: torch.Tensor,
    remote_start: int,
    remote_end: int,
    block_size: int,
    score_chunk_blocks: int,
) -> tuple[torch.Tensor, list[int]]:
    """ColBERT-style late interaction: mean_a max_token cos(q_a, k)."""

    if int(anchor_queries.shape[0]) != 1 or int(key_pre.shape[0]) != 1:
        raise ValueError("the probe currently supports batch size one")
    if block_size <= 0 or score_chunk_blocks <= 0:
        raise ValueError("block sizes must be positive")
    remote_count = max(0, int(remote_end) - int(remote_start))
    heads = int(anchor_queries.shape[1])
    if remote_count == 0:
        return torch.empty((heads, 0), device=anchor_queries.device), []
    groups = _gqa_groups(anchor_queries, key_pre)
    batch, query_heads, anchors, dim = anchor_queries.shape
    kv_heads = int(key_pre.shape[1])
    query = F.normalize(anchor_queries.float(), dim=-1).reshape(
        batch, kv_heads, groups, anchors, dim
    )
    remote_key = F.normalize(
        key_pre[:, :, int(remote_start) : int(remote_end), :].float(), dim=-1
    )
    block_lengths = [
        min(block_size, remote_count - offset)
        for offset in range(0, remote_count, block_size)
    ]
    block_count = len(block_lengths)
    padding = block_count * block_size - remote_count
    if padding:
        remote_key = F.pad(remote_key, (0, 0, 0, padding))
    blocked = remote_key.reshape(batch, kv_heads, block_count, block_size, dim)
    output: list[torch.Tensor] = []
    for start in range(0, block_count, int(score_chunk_blocks)):
        end = min(block_count, start + int(score_chunk_blocks))
        scores = torch.einsum(
            "zmgad,zmntd->zmgant", query, blocked[:, :, start:end]
        )
        if end == block_count and block_lengths[-1] != block_size:
            valid = torch.arange(block_size, device=scores.device) < block_lengths[-1]
            scores[..., -1, :] = scores[..., -1, :].masked_fill(~valid, -torch.inf)
        block = scores.amax(dim=-1).mean(dim=3)
        output.append(block.reshape(batch, query_heads, end - start)[0])
    return torch.cat(output, dim=-1), block_lengths


def _partial_block_token_scores(
    anchor_queries: torch.Tensor,
    key_pre: torch.Tensor,
    head: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    groups = _gqa_groups(anchor_queries, key_pre)
    kv_head = int(head) // groups
    query = F.normalize(anchor_queries[0, int(head)].float(), dim=-1)
    keys = F.normalize(key_pre[0, kv_head, positions].float(), dim=-1)
    # A boundary block cannot be retained whole.  Keep the tokens that strongly
    # match at least one question facet; no evidence labels enter this rule.
    return torch.matmul(query, keys.transpose(0, 1)).amax(dim=0)


def queryspan_block_selection(
    anchor_queries: torch.Tensor,
    key_pre: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
    block_size: int,
    score_chunk_blocks: int,
) -> tuple[torch.Tensor, BudgetLayout, dict[str, float]]:
    """Rank blocks by multi-vector late interaction, then spend an exact budget."""

    heads = int(anchor_queries.shape[1])
    layout = compute_budget_layout(
        int(key_pre.shape[-2]), keep_count, local_window, sink_tokens
    )
    block_scores, block_lengths = queryspan_block_scores(
        anchor_queries,
        key_pre,
        layout.remote_start,
        layout.remote_end,
        block_size,
        score_chunk_blocks,
    )
    remote_rows: list[torch.Tensor] = []
    selected_block_counts: list[int] = []
    partial_counts: list[int] = []
    if layout.remote_count:
        ranked = torch.argsort(block_scores, dim=-1, descending=True).cpu()
        for head in range(heads):
            remaining = layout.remote_count
            chosen: list[torch.Tensor] = []
            used_blocks = 0
            partial = 0
            for raw_block in ranked[head].tolist():
                length = int(block_lengths[int(raw_block)])
                start = layout.remote_start + int(raw_block) * int(block_size)
                positions = torch.arange(start, start + length, device=key_pre.device)
                if length <= remaining:
                    chosen.append(positions)
                    remaining -= length
                else:
                    token_scores = _partial_block_token_scores(
                        anchor_queries, key_pre, head, positions
                    )
                    chosen.append(
                        positions[
                            torch.topk(token_scores, k=remaining, largest=True, sorted=False).indices
                        ]
                    )
                    partial = remaining
                    remaining = 0
                used_blocks += 1
                if remaining == 0:
                    break
            if remaining:
                raise RuntimeError("block selector exhausted blocks before its token budget")
            remote_rows.append(torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=key_pre.device))
            selected_block_counts.append(used_blocks)
            partial_counts.append(partial)
        remote = torch.stack(remote_rows, dim=0)
    else:
        remote = torch.empty((heads, 0), dtype=torch.long, device=key_pre.device)
        selected_block_counts = [0] * heads
        partial_counts = [0] * heads
    pieces: list[torch.Tensor] = []
    if layout.sink_count:
        pieces.append(
            torch.arange(layout.sink_count, device=key_pre.device)
            .view(1, -1)
            .expand(heads, -1)
        )
    if layout.remote_count:
        pieces.append(remote)
    if layout.local_count:
        pieces.append(
            torch.arange(layout.local_start, layout.current, device=key_pre.device)
            .view(1, -1)
            .expand(heads, -1)
        )
    pieces.append(torch.full((heads, 1), layout.current, device=key_pre.device, dtype=torch.long))
    selected = torch.cat(pieces, dim=-1)
    if int(selected.shape[-1]) != layout.keep_count:
        raise AssertionError("query-span selector violated token budget")
    return selected, layout, {
        "selected_blocks_mean": statistics.fmean(selected_block_counts),
        "partial_block_tokens_mean": statistics.fmean(partial_counts),
        "candidate_block_count": float(len(block_lengths)),
    }


def queryspan_tokenmax_selection(
    anchor_queries: torch.Tensor,
    key_pre: torch.Tensor,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, BudgetLayout]:
    """Minimal ablation: per-token max over question-anchor cosine scores."""

    groups = _gqa_groups(anchor_queries, key_pre)
    batch, heads, anchors, dim = anchor_queries.shape
    kv_heads = int(key_pre.shape[1])
    query = F.normalize(anchor_queries.float(), dim=-1).reshape(
        batch, kv_heads, groups, anchors, dim
    )
    keys = F.normalize(key_pre.float(), dim=-1)
    scores = torch.einsum("bmgad,bmkd->bmgak", query, keys)
    token_scores = scores.amax(dim=3).reshape(batch, heads, key_pre.shape[-2])[0]
    return exact_final_pre_selection(token_scores, keep_count, local_window, sink_tokens)


@dataclass
class QuerySpanMetrics:
    head_rows: int = 0
    selected_gold: int = 0
    gold_total: int = 0
    selected_remote_gold: int = 0
    remote_gold_total: int = 0
    selected_conflict: int = 0
    conflict_total: int = 0
    selected_remote_conflict: int = 0
    remote_conflict_total: int = 0
    selected_lexical: int = 0
    lexical_total: int = 0
    gold_line_hits: int = 0
    conflict_line_hits: int = 0
    gold_mass_sum: float = 0.0
    conflict_mass_sum: float = 0.0
    lexical_mass_sum: float = 0.0
    other_mass_sum: float = 0.0
    budget_violations: int = 0
    duplicate_violations: int = 0
    sink_violations: int = 0
    local_violations: int = 0
    current_violations: int = 0
    remote_slots: int = 0
    selected_slots: int = 0
    exact_overlap_sum: float = 0.0
    exact_overlap_count: int = 0
    selected_blocks_sum: float = 0.0
    partial_tokens_sum: float = 0.0
    block_event_count: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "gold_evidence_token_recall": self.selected_gold / max(1, self.gold_total),
            "remote_gold_evidence_token_recall": (
                self.selected_remote_gold / self.remote_gold_total
                if self.remote_gold_total
                else None
            ),
            "conflict_token_recall": self.selected_conflict / max(1, self.conflict_total),
            "remote_conflict_token_recall": (
                self.selected_remote_conflict / self.remote_conflict_total
                if self.remote_conflict_total
                else None
            ),
            "lexical_distractor_token_recall": self.selected_lexical / max(1, self.lexical_total),
            "gold_evidence_line_hit_rate": self.gold_line_hits / max(1, self.head_rows),
            "conflict_line_hit_rate": self.conflict_line_hits / max(1, self.head_rows),
            "gold_evidence_attention_mass": self.gold_mass_sum / max(1, self.head_rows),
            "conflict_attention_mass": self.conflict_mass_sum / max(1, self.head_rows),
            "lexical_distractor_attention_mass": self.lexical_mass_sum / max(1, self.head_rows),
            "other_attention_mass": self.other_mass_sum / max(1, self.head_rows),
            "token_budget_violation_fraction": self.budget_violations / max(1, self.head_rows),
            "duplicate_support_violation_fraction": self.duplicate_violations / max(1, self.head_rows),
            "sink_coverage_violation_fraction": self.sink_violations / max(1, self.head_rows),
            "local_coverage_violation_fraction": self.local_violations / max(1, self.head_rows),
            "current_coverage_violation_fraction": self.current_violations / max(1, self.head_rows),
            "remote_selected_fraction": self.remote_slots / max(1, self.selected_slots),
            "support_overlap_with_exact_mean": self.exact_overlap_sum / max(1, self.exact_overlap_count),
            "selected_blocks_mean": self.selected_blocks_sum / max(1, self.block_event_count),
            "partial_block_tokens_mean": self.partial_tokens_sum / max(1, self.block_event_count),
            "selector_used_evidence_labels": 0,
        }


@dataclass
class QuerySpanController:
    variant: str
    ratio: float
    minimum_keep_tokens: int
    maximum_keep_tokens: int
    local_window: int
    sink_tokens: int
    block_size: int
    score_chunk_blocks: int
    evaluation_positions: dict[str, tuple[int, ...]]
    record_spans: dict[str, tuple[tuple[int, int], ...]]
    metrics: QuerySpanMetrics = field(default_factory=QuerySpanMetrics)

    def _mask(self, category: str, key_count: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(key_count, dtype=torch.bool, device=device)
        positions = self.evaluation_positions.get(category, ())
        if positions:
            values = torch.tensor(positions, dtype=torch.long, device=device)
            values = values[(values >= 0) & (values < key_count)]
            mask[values] = True
        return mask

    def record(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        layout: BudgetLayout,
        exact_reference: torch.Tensor | None,
        block_diagnostics: dict[str, float] | None,
    ) -> None:
        heads = int(selected.shape[0])
        self.metrics.head_rows += heads
        masks = {
            name: self._mask(name, layout.key_count, selected.device)
            for name in ("gold_evidence", "conflict_evidence", "lexical_format_distractor")
        }
        selected_masks = {name: mask[selected] for name, mask in masks.items()}
        self.metrics.selected_gold += int(selected_masks["gold_evidence"].sum().item())
        self.metrics.gold_total += int(masks["gold_evidence"].sum().item()) * heads
        self.metrics.selected_conflict += int(selected_masks["conflict_evidence"].sum().item())
        self.metrics.conflict_total += int(masks["conflict_evidence"].sum().item()) * heads
        remote_region = torch.zeros(layout.key_count, dtype=torch.bool, device=selected.device)
        remote_region[layout.remote_start : layout.remote_end] = True
        remote_gold = masks["gold_evidence"] & remote_region
        remote_conflict = masks["conflict_evidence"] & remote_region
        self.metrics.selected_remote_gold += int(remote_gold[selected].sum().item())
        self.metrics.remote_gold_total += int(remote_gold.sum().item()) * heads
        self.metrics.selected_remote_conflict += int(remote_conflict[selected].sum().item())
        self.metrics.remote_conflict_total += int(remote_conflict.sum().item()) * heads
        self.metrics.selected_lexical += int(selected_masks["lexical_format_distractor"].sum().item())
        self.metrics.lexical_total += int(masks["lexical_format_distractor"].sum().item()) * heads
        for category, field_name in (
            ("gold_evidence", "gold_line_hits"),
            ("conflict_evidence", "conflict_line_hits"),
        ):
            spans = self.record_spans.get(category, ())
            hit = torch.zeros(heads, dtype=torch.bool, device=selected.device)
            for start, end in spans:
                hit |= ((selected >= int(start)) & (selected < int(end))).any(dim=-1)
            setattr(self.metrics, field_name, getattr(self.metrics, field_name) + int(hit.sum().item()))
        probabilities = weights[0, :, 0].float()
        class_masses: dict[str, torch.Tensor] = {}
        for category, chosen in selected_masks.items():
            class_masses[category] = probabilities.masked_fill(~chosen, 0.0).sum(dim=-1)
        self.metrics.gold_mass_sum += float(class_masses["gold_evidence"].sum().item())
        self.metrics.conflict_mass_sum += float(class_masses["conflict_evidence"].sum().item())
        self.metrics.lexical_mass_sum += float(class_masses["lexical_format_distractor"].sum().item())
        accounted = sum(class_masses.values())
        self.metrics.other_mass_sum += float((1.0 - accounted).sum().item())

        self.metrics.budget_violations += int((selected.shape[-1] != layout.keep_count)) * heads
        sorted_support = selected.sort(dim=-1).values
        duplicates = (sorted_support[:, 1:] == sorted_support[:, :-1]).any(dim=-1)
        self.metrics.duplicate_violations += int(duplicates.sum().item())
        if layout.sink_count:
            sink_ok = (selected < layout.sink_count).sum(dim=-1) == layout.sink_count
            self.metrics.sink_violations += int((~sink_ok).sum().item())
        if layout.local_count:
            local_ok = (
                (selected >= layout.local_start) & (selected < layout.current)
            ).sum(dim=-1) == layout.local_count
            self.metrics.local_violations += int((~local_ok).sum().item())
        current_ok = (selected == layout.current).any(dim=-1)
        self.metrics.current_violations += int((~current_ok).sum().item())
        remote = (selected >= layout.remote_start) & (selected < layout.remote_end)
        self.metrics.remote_slots += int(remote.sum().item())
        self.metrics.selected_slots += int(selected.numel())
        if exact_reference is not None:
            if exact_reference.shape != selected.shape:
                raise ValueError("exact-reference support shape mismatch")
            selected_bitmap = torch.zeros(
                (heads, layout.key_count), dtype=torch.bool, device=selected.device
            )
            exact_bitmap = torch.zeros_like(selected_bitmap)
            selected_bitmap.scatter_(1, selected, True)
            exact_bitmap.scatter_(1, exact_reference.to(selected.device), True)
            intersection = (selected_bitmap & exact_bitmap).sum(dim=-1).float()
            union = (2 * selected.shape[-1] - intersection).clamp_min(1.0)
            self.metrics.exact_overlap_sum += float((intersection / union).sum().item())
            self.metrics.exact_overlap_count += heads
        if block_diagnostics is not None:
            self.metrics.selected_blocks_sum += float(block_diagnostics["selected_blocks_mean"])
            self.metrics.partial_tokens_sum += float(block_diagnostics["partial_block_tokens_mean"])
            self.metrics.block_event_count += 1


@contextmanager
def activate(controller: QuerySpanController | None):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


def read_only_final_query_kv(
    past_key_value: Any | None,
    layer_index: int,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if past_key_value is None:
        return current_key, current_value
    if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
        prefix_key = past_key_value.key_cache[int(layer_index)]
        prefix_value = past_key_value.value_cache[int(layer_index)]
    elif isinstance(past_key_value, (tuple, list)):
        prefix_key, prefix_value = past_key_value[int(layer_index)]
    else:
        raise TypeError("unsupported cache type")
    return (
        torch.cat((prefix_key, current_key), dim=-2),
        torch.cat((prefix_value, current_value), dim=-2),
    )


def _make_key_capture_hook(attention: torch.nn.Module):
    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not _CAPTURE_PREFIX:
            return
        attention._queryspan_pre_key_chunks.append(
            output.detach().transpose(1, 2).contiguous().to(_PREFIX_KEY_STORAGE)
        )

    return hook


def _make_query_capture_hook(attention: torch.nn.Module):
    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not _CAPTURE_PREFIX:
            return
        start = int(attention._queryspan_capture_cursor)
        end = start + int(output.shape[1])
        local = [position - start for position in _CAPTURE_ANCHOR_POSITIONS if start <= position < end]
        if local:
            indices = torch.tensor(local, dtype=torch.long, device=output.device)
            attention._queryspan_anchor_chunks.append(
                output.detach().index_select(1, indices).transpose(1, 2).contiguous().to(_PREFIX_KEY_STORAGE)
            )
            attention._queryspan_anchor_position_chunks.extend(start + value for value in local)
        attention._queryspan_capture_cursor = end

    return hook


def patch_model(model: Any) -> None:
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        module._queryspan_original_forward = module.forward
        module._queryspan_modeling_qwen3 = modeling_qwen3
        module._queryspan_pre_key_chunks = []
        module._queryspan_pre_key_cache = None
        module._queryspan_anchor_chunks = []
        module._queryspan_anchor_cache = None
        module._queryspan_anchor_position_chunks = []
        module._queryspan_capture_cursor = 0
        module._queryspan_exact_support = None
        module._queryspan_key_handle = module.k_norm.register_forward_hook(_make_key_capture_hook(module))
        module._queryspan_query_handle = module.q_norm.register_forward_hook(_make_query_capture_hook(module))
        module.forward = types.MethodType(queryspan_attention_forward, module)
        found += 1
    if not found:
        raise RuntimeError("no Qwen3Attention modules found")


def capture_prefill_sequence(
    model: Any,
    prompt_prefix: torch.Tensor,
    chunk_size: int,
    anchor_positions: Sequence[int],
) -> tuple[Any, float]:
    global _CAPTURE_PREFIX, _CAPTURE_ANCHOR_POSITIONS
    attentions = [module for module in model.modules() if module.__class__.__name__ == "Qwen3Attention"]
    anchors = tuple(sorted(dict.fromkeys(map(int, anchor_positions))))
    for attention in attentions:
        attention._queryspan_pre_key_chunks = []
        attention._queryspan_pre_key_cache = None
        attention._queryspan_anchor_chunks = []
        attention._queryspan_anchor_cache = None
        attention._queryspan_anchor_position_chunks = []
        attention._queryspan_capture_cursor = 0
        attention._queryspan_exact_support = None
    _CAPTURE_ANCHOR_POSITIONS = anchors
    _CAPTURE_PREFIX = True
    try:
        result = base.prefill_sequence(model, prompt_prefix, chunk_size)
    finally:
        _CAPTURE_PREFIX = False
        _CAPTURE_ANCHOR_POSITIONS = ()
    expected = int(prompt_prefix.shape[1])
    for attention in attentions:
        keys = torch.cat(attention._queryspan_pre_key_chunks, dim=2)
        if int(keys.shape[2]) != expected:
            raise RuntimeError(f"layer {attention.layer_idx} captured {keys.shape[2]} keys, expected {expected}")
        captured_positions = tuple(attention._queryspan_anchor_position_chunks)
        if captured_positions != anchors:
            raise RuntimeError(
                f"layer {attention.layer_idx} query anchors {captured_positions}, expected {anchors}"
            )
        queries = torch.cat(attention._queryspan_anchor_chunks, dim=2)
        if int(queries.shape[2]) != len(anchors):
            raise RuntimeError("query-anchor cache size mismatch")
        attention._queryspan_pre_key_cache = keys
        attention._queryspan_anchor_cache = queries
        attention._queryspan_pre_key_chunks = []
        attention._queryspan_anchor_chunks = []
    return result


def queryspan_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = _ACTIVE_CONTROLLER
    if controller is None or int(hidden_states.shape[-2]) != 1:
        return self._queryspan_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_post, current_key_post = self._queryspan_modeling_qwen3.apply_rotary_pos_emb(
        query_pre, current_key_pre, cos.to(query_pre.device), sin.to(query_pre.device)
    )
    key_post, value = read_only_final_query_kv(
        past_key_value, int(self.layer_idx), current_key_post, current_value
    )
    cached_pre = self._queryspan_pre_key_cache
    anchors = self._queryspan_anchor_cache
    if cached_pre is None or anchors is None:
        raise RuntimeError("exact pre-RoPE caches are missing; use capture_prefill_sequence")
    key_pre = torch.cat((cached_pre.to(current_key_pre.device), current_key_pre), dim=2)
    anchors = anchors.to(query_pre.device)
    if int(key_pre.shape[-2]) != int(key_post.shape[-2]):
        raise RuntimeError("pre/post key cache lengths differ")
    scale = float(getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1])))
    post_scores = runner.add_attention_mask(
        gqa_query_key_scores(query_post, key_post, scale), attention_mask
    )
    keep_count = max(1, int(math.ceil(controller.ratio * int(key_post.shape[-2]))))
    if controller.minimum_keep_tokens > 0:
        keep_count = max(keep_count, controller.minimum_keep_tokens)
    if controller.maximum_keep_tokens > 0:
        keep_count = min(keep_count, controller.maximum_keep_tokens)
    keep_count = min(int(key_post.shape[-2]), keep_count)
    diagnostics: dict[str, float] | None = None
    if controller.variant == "exact_final_pre_top2_postscore":
        pre_scores = runner.add_attention_mask(
            gqa_query_key_scores(query_pre, key_pre, scale), attention_mask
        )
        selected, layout = exact_final_pre_selection(
            pre_scores[0, :, 0], keep_count, controller.local_window, controller.sink_tokens
        )
        self._queryspan_exact_support = selected.detach().clone()
    elif controller.variant == "queryspan_block_top2_postscore":
        selected, layout, diagnostics = queryspan_block_selection(
            anchors,
            key_pre,
            keep_count,
            controller.local_window,
            controller.sink_tokens,
            controller.block_size,
            controller.score_chunk_blocks,
        )
    elif controller.variant == "queryspan_tokenmax_top2_postscore":
        selected, layout = queryspan_tokenmax_selection(
            anchors, key_pre, keep_count, controller.local_window, controller.sink_tokens
        )
    else:
        raise ValueError(f"unsupported query-span variant: {controller.variant}")
    sparse_scores = post_scores[0, :, 0].gather(1, selected).unsqueeze(0).unsqueeze(2)
    groups = _gqa_groups(query_post, value)
    selected_value = gather_per_query_head_gqa(value, selected, groups)
    weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
    exact_reference = getattr(self, "_queryspan_exact_support", None)
    if controller.variant == "exact_final_pre_top2_postscore":
        exact_reference = None
    controller.record(selected, weights, layout, exact_reference, diagnostics)
    attention_output = torch.matmul(weights, selected_value)
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(attention_output), weights


def build_case(tokenizer: Any, length: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    case = safety.build_case(
        tokenizer,
        total_tokens=int(length),
        seed=int(seed),
        packet_gap_tokens=int(args.packet_gap_tokens),
        class_sample_count=int(args.class_sample_count),
    )
    spans: dict[str, list[tuple[int, int]]] = {
        "gold_evidence": [],
        "conflict_evidence": [],
        "lexical_format_distractor": [],
    }
    for record in case["records"]:
        category = str(record["category"])
        if category in spans:
            start, end = record["span"]
            spans[category].append((int(start), int(end)))
    case["evaluation_positions"] = {
        name: tuple(map(int, case["class_positions"][name]))
        for name in spans
    }
    case["record_spans"] = {name: tuple(values) for name, values in spans.items()}
    return case


def _mean_present(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(int(row["target_context_tokens"]), str(row["variant"])) for row in rows})
    metric_names = (
        "gold_evidence_token_recall",
        "remote_gold_evidence_token_recall",
        "conflict_token_recall",
        "remote_conflict_token_recall",
        "gold_evidence_line_hit_rate",
        "conflict_line_hit_rate",
        "gold_evidence_attention_mass",
        "conflict_attention_mass",
        "token_budget_violation_fraction",
        "duplicate_support_violation_fraction",
        "sink_coverage_violation_fraction",
        "local_coverage_violation_fraction",
        "current_coverage_violation_fraction",
        "support_overlap_with_exact_mean",
        "mean_query_seconds",
    )
    for length, variant in keys:
        selected = [
            row
            for row in rows
            if int(row["target_context_tokens"]) == length and str(row["variant"]) == variant
        ]
        mean_nll = statistics.fmean(float(row["gold_nll"]) for row in selected)
        item: dict[str, Any] = {
            "target_context_tokens": length,
            "variant": variant,
            "sample_count": len(selected),
            "gold_ppl": math.exp(min(mean_nll, 700.0)),
            "next_token_accuracy": statistics.fmean(int(row["next_token_correct"]) for row in selected),
            "gold_conflict_margin": statistics.fmean(float(row["gold_conflict_margin"]) for row in selected),
        }
        for name in metric_names:
            item[name] = _mean_present(selected, name)
        output.append(item)
    return output


def merge_shards(output_dir: Path, shard_paths: Iterable[str]) -> None:
    rows: list[dict[str, Any]] = []
    for raw in shard_paths:
        path = Path(raw) / "rows.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: (int(row["target_context_tokens"]), int(row["seed"]), str(row["variant"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    write_csv(output_dir / "rows.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> tuple[list[int], list[str]]:
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if args.block_size <= 0 or args.query_anchor_count <= 0 or args.score_chunk_blocks <= 0:
        raise ValueError("block/query-anchor/chunk sizes must be positive")
    if args.minimum_keep_tokens < 0 or args.maximum_keep_tokens < 0:
        raise ValueError("keep-token bounds must be non-negative")
    if (
        args.minimum_keep_tokens > 0
        and args.maximum_keep_tokens > 0
        and args.minimum_keep_tokens > args.maximum_keep_tokens
    ):
        raise ValueError("minimum keep tokens cannot exceed maximum")
    lengths = sorted({int(value) for value in args.lengths.split(",") if value.strip()})
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if not lengths or not variants or unknown:
        raise ValueError(f"invalid lengths or variants; unknown={unknown}")
    sparse = [value for value in variants if value != "native_noop"]
    if "native_noop" not in variants:
        raise ValueError("native_noop is required as the exact untouched baseline")
    if any(value.startswith("queryspan_") for value in sparse) and (
        "exact_final_pre_top2_postscore" not in sparse
    ):
        raise ValueError("query-span arms require the exact-final-pre comparator")
    ordered = [value for value in VARIANTS if value in variants]
    return lengths, ordered


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        merge_shards(output_dir, (value.strip() for value in args.merge_shards.split(",") if value.strip()))
        return
    lengths, variants = _validate_args(args)
    if not args.model_name_or_path and not args.dry_run:
        raise ValueError("--model-name-or-path is required outside merge/dry-run mode")
    config = {
        **vars(args),
        "resolved_lengths": lengths,
        "variants": variants,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "selector_uses_evidence_labels": False,
        "consumer": "native post-RoPE QK + original V + sparse softmax",
        "prefix_query_cache": "question anchors only; never full-sequence Q",
    }
    write_json(output_dir / "config.json", config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    model, tokenizer = runner.load_model(args)
    patch_model(model)
    answer_ids = safety.answer_token_ids(tokenizer)
    rows_path = output_dir / "rows.jsonl"
    completed: set[tuple[int, int]] = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((int(row["target_context_tokens"]), int(row["seed"])))

    for length in lengths:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            if (length, seed) in completed:
                continue
            case = build_case(tokenizer, length, seed, args)
            prompt = torch.tensor(case["prompt_ids"], dtype=torch.long).view(1, -1)
            prefix_length = int(prompt.shape[1]) - 1
            anchor_positions = select_query_anchor_positions(
                case["query_span"], prefix_length, args.query_anchor_count
            )
            anchor_token_ids = [int(prompt[0, position]) for position in anchor_positions]
            anchor_token_texts = [
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).replace("\n", "\\n")
                for token_id in anchor_token_ids
            ]
            legacy, prefill_seconds = capture_prefill_sequence(
                model, prompt[:, :-1], args.prefill_chunk_size, anchor_positions
            )
            cache = base.cache_from_legacy(legacy)
            del legacy
            baseline_answer: dict[str, Any] | None = None
            seed_rows: list[dict[str, Any]] = []
            for variant in variants:
                controller = None
                if variant != "native_noop":
                    controller = QuerySpanController(
                        variant=variant,
                        ratio=args.ratio,
                        minimum_keep_tokens=args.minimum_keep_tokens,
                        maximum_keep_tokens=args.maximum_keep_tokens,
                        local_window=args.local_window,
                        sink_tokens=args.sink_tokens,
                        block_size=args.block_size,
                        score_chunk_blocks=args.score_chunk_blocks,
                        evaluation_positions=case["evaluation_positions"],
                        record_spans=case["record_spans"],
                    )
                base.synchronize()
                started = time.perf_counter()
                with activate(controller), torch.inference_mode():
                    output = base.forward_with_cache(
                        model,
                        prompt[:, -1:].to(base.input_device(model)),
                        cache,
                        prefix_length,
                    )
                base.synchronize()
                query_seconds = time.perf_counter() - started
                answer = safety.answer_metrics(
                    tokenizer, output.logits, answer_ids, case["conflict_answer"]
                )
                if variant == "native_noop":
                    baseline_answer = answer
                if baseline_answer is None:
                    raise RuntimeError("native_noop must run before sparse variants")
                delta = safety.delta_metrics(answer, baseline_answer)
                metric_values: dict[str, Any]
                if controller is None:
                    metric_values = {
                        name: None
                        for name in QuerySpanMetrics().summary()
                    }
                    metric_values.update(
                        {
                            "selector_used_evidence_labels": 0,
                            "native_noop": 1,
                        }
                    )
                else:
                    metric_values = {**controller.metrics.summary(), "native_noop": 0}
                    invalid = {
                        name: metric_values[name]
                        for name in (
                            "token_budget_violation_fraction",
                            "duplicate_support_violation_fraction",
                            "sink_coverage_violation_fraction",
                            "local_coverage_violation_fraction",
                            "current_coverage_violation_fraction",
                        )
                        if float(metric_values[name]) != 0.0
                    }
                    if invalid:
                        raise RuntimeError(
                            f"hard support audit failed for {variant}: {invalid}"
                        )
                keep_count = max(1, int(math.ceil(args.ratio * int(prompt.shape[1]))))
                if args.minimum_keep_tokens > 0:
                    keep_count = max(keep_count, args.minimum_keep_tokens)
                if args.maximum_keep_tokens > 0:
                    keep_count = min(keep_count, args.maximum_keep_tokens)
                audit_layout = compute_budget_layout(
                    int(prompt.shape[1]),
                    keep_count,
                    args.local_window,
                    args.sink_tokens,
                )
                gold_positions = tuple(case["evaluation_positions"]["gold_evidence"])
                conflict_positions = tuple(case["evaluation_positions"]["conflict_evidence"])
                gold_in_sink = sum(position < audit_layout.sink_count for position in gold_positions)
                conflict_in_sink = sum(
                    position < audit_layout.sink_count for position in conflict_positions
                )
                gold_blocks = {
                    (position - audit_layout.remote_start) // args.block_size
                    for position in gold_positions
                    if audit_layout.remote_start <= position < audit_layout.remote_end
                }
                conflict_blocks = {
                    (position - audit_layout.remote_start) // args.block_size
                    for position in conflict_positions
                    if audit_layout.remote_start <= position < audit_layout.remote_end
                }
                seed_rows.append(
                    {
                        "target_context_tokens": int(length),
                        "prompt_tokens": int(prompt.shape[1]),
                        "seed": int(seed),
                        "variant": variant,
                        "gold_answer": case["gold_output"],
                        "conflict_answer": case["conflict_output"],
                        "query_span_start": int(case["query_span"][0]),
                        "query_span_end": int(case["query_span"][1]),
                        "query_anchor_count": len(anchor_positions),
                        "query_anchor_first": anchor_positions[0],
                        "query_anchor_last": anchor_positions[-1],
                        "query_anchor_token_ids": json.dumps(anchor_token_ids),
                        "query_anchor_token_texts": json.dumps(
                            anchor_token_texts, ensure_ascii=False
                        ),
                        "cached_full_prefix_query_tokens": 0,
                        "expected_keep_tokens": min(int(prompt.shape[1]), keep_count),
                        "expected_remote_keep_tokens": audit_layout.remote_count,
                        "gold_tokens_in_sink": gold_in_sink,
                        "gold_fraction_in_sink": gold_in_sink / max(1, len(gold_positions)),
                        "conflict_tokens_in_sink": conflict_in_sink,
                        "conflict_fraction_in_sink": (
                            conflict_in_sink / max(1, len(conflict_positions))
                        ),
                        "gold_remote_block_count": len(gold_blocks),
                        "conflict_remote_block_count": len(conflict_blocks),
                        "gold_conflict_share_remote_block": int(
                            bool(gold_blocks & conflict_blocks)
                        ),
                        "record_order": ",".join(
                            str(record["category"]) for record in case["records"]
                        ),
                        "ratio": float(args.ratio),
                        "local_window": int(args.local_window),
                        "sink_tokens": int(args.sink_tokens),
                        "block_size": int(args.block_size),
                        **metric_values,
                        **answer,
                        **delta,
                        "prefill_seconds": float(prefill_seconds),
                        "query_seconds": float(query_seconds),
                        "mean_query_seconds": float(query_seconds),
                    }
                )
                del output
                rope_repair.reset_dynamic_cache(cache, prefix_length)
            with rows_path.open("a", encoding="utf-8") as handle:
                for row in seed_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            write_csv(output_dir / "rows.csv", all_rows)
            write_csv(output_dir / "summary.csv", summarize(all_rows))
            for row in seed_rows:
                print(
                    f"length={length} seed={seed} {row['variant']}: "
                    f"ppl={row['gold_ppl']:.4f} correct={row['next_token_correct']} "
                    f"gold_recall={row.get('gold_evidence_token_recall')}",
                    flush=True,
                )
            del cache, prompt
            clear_allocator()

    all_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(all_rows)
    write_csv(output_dir / "rows.csv", all_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "design.json",
        {
            "selector_inputs": ["question-span pre-RoPE Q anchors", "pre-RoPE history K"],
            "forbidden_selector_inputs": ["gold span", "conflict span", "answer token", "gradient"],
            "shared_budget": "same sink + remote + local + current token count",
            "consumer": "unmodified post-RoPE QK and V on selected support",
        },
    )
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
