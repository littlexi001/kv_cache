from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_hierarchical_book_index_recall import joined  # noqa: E402
from analyze_longrange_book_index_semantic_retrieval import (  # noqa: E402
    LABELS,
    GeneratedTask,
    NAMES,
    Span,
    TOPICS,
    append_segment,
    build_indexes,
    build_task,
    encode,
    fill_to,
    make_key,
    overlap_page,
)
from analyze_typed_anchor_page_recall import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
    str2bool,
    token_text,
    write_csv,
)
from book_page_router import pages_to_ranges, pages_to_tokens, selected_pages_for_mode  # noqa: E402


CHAIN_VARIANTS = {"chain", "chain_para", "chain_para_conflict", "chain_story_conflict"}
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_SPARSE_CONTEXT: "SparseContext | None" = None


class SparseContext:
    def __init__(
        self,
        mode: str,
        context_tokens: int,
        keep_remote_tokens: set[int],
        keep_remote_ranges: list[tuple[int, int]] | None,
        sink_tokens: int,
        recent_tokens: int,
        sparse_impl: str = "mask",
        stats: "SparseStats | None" = None,
    ) -> None:
        self.mode = mode
        self.context_tokens = context_tokens
        self.keep_remote_tokens = keep_remote_tokens
        self.keep_remote_ranges = keep_remote_ranges or []
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.sparse_impl = sparse_impl
        self.stats = stats


class SparseStats:
    def __init__(self) -> None:
        self.cases = 0
        self.history_tokens = 0
        self.kept_tokens = 0
        self.max_kept = 0

    def add(self, keep: torch.Tensor, history_count: int) -> None:
        # keep shape: [query, key]
        counts = keep.sum(dim=-1).detach().cpu()
        self.cases += int(counts.numel())
        self.history_tokens += int(history_count * counts.numel())
        self.kept_tokens += int(counts.sum().item())
        self.max_kept = max(self.max_kept, int(counts.max().item()) if counts.numel() else 0)

    def add_count(self, kept_count: int, history_count: int) -> None:
        self.cases += 1
        self.history_tokens += int(history_count)
        self.kept_tokens += int(kept_count)
        self.max_kept = max(self.max_kept, int(kept_count))

    def row(self) -> dict[str, Any]:
        return {
            "sparse_cases": self.cases,
            "mean_history_tokens": self.history_tokens / self.cases if self.cases else 0.0,
            "mean_kept_tokens": self.kept_tokens / self.cases if self.cases else 0.0,
            "mean_kept_fraction": self.kept_tokens / self.history_tokens if self.history_tokens else 0.0,
            "max_kept_tokens": self.max_kept,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-context KV sparse page-mask eval for long-range book-index routing."
    )
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_tokens", default="10000,20000")
    parser.add_argument("--tasks_per_length", type=int, default=2)
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
        "--suite_layouts",
        default="default",
        help=(
            "Comma-separated task layouts. Use default for the original generator or eXX_dYY "
            "for evidence/decoy start targets as percent of prefill, e.g. e05_d90."
        ),
    )
    parser.add_argument(
        "--task_variant",
        choices=["single", "chain", "chain_para", "chain_para_conflict", "chain_story_conflict"],
        default="single",
        help=(
            "single uses one authoritative evidence page; chain uses marker-heavy bridge -> answer retrieval; "
            "chain_para uses paraphrased bridge/answer pages; chain_para_conflict adds a superseded same-artifact entry; "
            "chain_story_conflict uses less templated badge -> alias -> ruling retrieval."
        ),
    )
    parser.add_argument(
        "--modes",
        default=(
            "full,sink_recent,remote_tail_p4,remote_tail_p8,book_auth_flat_p4,"
            "book_auth_flat_p4_adj1,book_auth_flat_p4_adj2,book_auth_flat_p8,"
            "book_auth_flat_p4_authadj1,book_auth_flat_p4_authadj2,"
            "book_auth_hier_s4_p2,book_auth_hier_s4_p2_adj1,"
            "book_auth_hier_s4_p2_authadj1,budget_authflat_p4_authadj2_b4,"
            "budget_authflat_p4_authadj2_b5,budget_authflat_p4_authadj2_b6,"
            "budget_authflat_p4_authadj2_b8,hybrid_tail4_authflat4,"
            "hybrid_gatedtail4_authflat4,hybrid_gatedtail4_authhier_s4_p2"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026070101)
    parser.add_argument("--score_query_ppl", type=str2bool, default=True)
    parser.add_argument("--score_calibrated", type=str2bool, default=True)
    parser.add_argument("--balanced_labels", type=str2bool, default=False)
    parser.add_argument(
        "--sparse_attention_impl",
        choices=["mask", "gather", "sdpa_gather", "range_sdpa", "triton"],
        default="mask",
    )
    parser.add_argument(
        "--answer_score_format",
        choices=["letter", "answer_label", "sentence", "gated_sentence"],
        default="letter",
    )
    parser.add_argument(
        "--gated_sentence_margin",
        type=float,
        default=1.0,
        help="For answer_score_format=gated_sentence, use sentence scoring when calibrated top margin is below this.",
    )
    parser.add_argument(
        "--typed_record_mode",
        choices=["none", "extractive"],
        default="none",
        help="Optionally insert a compact typed memory record extracted from selected pages before the query.",
    )
    parser.add_argument(
        "--typed_record_format",
        choices=[
            "verbose",
            "compact",
            "label_only",
            "summary",
            "mini_summary",
            "short_summary",
            "lite_summary",
            "natural_summary",
            "answerline_summary",
        ],
        default="verbose",
        help="Text format used when inserting an extracted typed record before the query.",
    )
    parser.add_argument(
        "--typed_summary_source_mode",
        default="",
        help=(
            "When typed_record_format=summary, optionally build the summary record from pages selected by this "
            "routing mode while keeping the evaluated mode's sparse raw-token pages unchanged."
        ),
    )
    parser.add_argument(
        "--typed_record_answer_override",
        type=str2bool,
        default=False,
        help="If a typed record contains an answer label, use it as the final downstream prediction.",
    )
    parser.add_argument(
        "--typed_record_insert",
        type=str2bool,
        default=True,
        help="Whether to insert the typed record text into the LM context before the query.",
    )
    parser.add_argument(
        "--skip_lm_answer_when_override",
        type=str2bool,
        default=False,
        help="When a typed-record override is available, skip LM option scoring for downstream answer prediction.",
    )
    return parser.parse_args()


def parse_layout(layout: str) -> tuple[int, int] | None:
    if layout == "default":
        return None
    match = re.fullmatch(r"e(\d+)_d(\d+)", layout)
    if not match:
        raise ValueError(f"Unsupported suite layout: {layout}")
    evidence_percent = int(match.group(1))
    decoy_percent = int(match.group(2))
    if not (0 <= evidence_percent <= 99 and 0 <= decoy_percent <= 99):
        raise ValueError(f"Layout percents must be in [0, 99]: {layout}")
    return evidence_percent, decoy_percent


def mode_recent_tokens(
    mode: str,
    default_recent_tokens: int,
    context_tokens: int,
    sink_tokens: int,
) -> int:
    match = re.search(r"_r(\d+)$", mode)
    if match:
        return int(match.group(1))
    match = re.search(r"_rauto(\d+)?$", mode)
    if match:
        min_remote_tokens = int(match.group(1) or 192)
        budget_match = re.search(r"_b(\d+)(?:_|$)", mode)
        if not budget_match:
            return default_recent_tokens
        total_budget = int(context_tokens * (int(budget_match.group(1)) / 100.0))
        default_remote_budget = total_budget - sink_tokens - default_recent_tokens
        if default_remote_budget >= min_remote_tokens:
            return default_recent_tokens
        return max(0, min(default_recent_tokens, total_budget - sink_tokens - min_remote_tokens))
    return default_recent_tokens


def bounded_segment_target(context_tokens: int, segment_len: int, percent: int) -> int:
    raw = int(context_tokens * (percent / 100.0))
    upper = max(0, context_tokens - segment_len - 96)
    return max(96, min(raw, upper))


def build_task_with_layout(
    tokenizer: Any,
    rng: random.Random,
    task_id: int,
    context_tokens: int,
    eval_tokens: int,
    layout: str,
    target_label_override: str | None = None,
    decoy_label_override: str | None = None,
) -> GeneratedTask:
    parsed = parse_layout(layout)
    if parsed is None and target_label_override is None and decoy_label_override is None:
        return build_task(tokenizer, rng, task_id, context_tokens, eval_tokens)
    if parsed is None:
        parsed = (5, 90)
    evidence_percent, decoy_percent = parsed
    target_key = make_key(rng, task_id)
    target_label = target_label_override or rng.choice(LABELS)
    if decoy_label_override and decoy_label_override != target_label:
        decoy_label = decoy_label_override
    else:
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

    true_len = len(encode(tokenizer, true_text))
    decoy_len = len(encode(tokenizer, decoy_text))
    targets = [
        ("evidence", bounded_segment_target(context_tokens, true_len, evidence_percent), true_text),
        ("decoy", bounded_segment_target(context_tokens, decoy_len, decoy_percent), decoy_text),
    ]
    targets.sort(key=lambda item: item[1])

    token_ids: list[int] = []
    filler_idx = 0
    evidence_span: Span | None = None
    decoy_span: Span | None = None
    for kind, target, text in targets:
        filler_idx = fill_to(token_ids, tokenizer, rng, target, filler_idx)
        span = append_segment(token_ids, tokenizer, text)
        if kind == "evidence":
            evidence_span = span
        else:
            decoy_span = span
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens, filler_idx)
    if len(token_ids) != context_tokens:
        token_ids = token_ids[:context_tokens]
    if evidence_span is None or decoy_span is None:
        raise ValueError(f"Layout did not create both spans: {layout}")
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


def build_chain_task_with_layout(
    tokenizer: Any,
    rng: random.Random,
    task_id: int,
    context_tokens: int,
    eval_tokens: int,
    layout: str,
    target_label_override: str | None = None,
    decoy_label_override: str | None = None,
) -> GeneratedTask:
    parsed = parse_layout(layout) or (5, 90)
    bridge_percent, decoy_percent = parsed
    answer_percent = max(bridge_percent + 8, min(decoy_percent - 10, (bridge_percent + decoy_percent) // 2))
    target_key = make_key(rng, task_id)
    bridge_key = "ART-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
    target_label = target_label_override or rng.choice(LABELS)
    if decoy_label_override and decoy_label_override != target_label:
        decoy_label = decoy_label_override
    else:
        decoy_label = rng.choice([label for label in LABELS if label != target_label])
    topic = rng.choice(TOPICS)
    witness = rng.choice(NAMES)
    bridge_text = (
        f"\nAUTHORITATIVE BRIDGE PAGE. In the {topic}, lookup key {target_key} routes to controlling "
        f"artifact code {bridge_key}. The answer must be read from the authoritative evidence page for "
        f"artifact code {bridge_key}; pages that skip this bridge are incomplete.\n"
    )
    answer_text = (
        f"\nAUTHORITATIVE EVIDENCE PAGE. For controlling artifact code {bridge_key}, the verified answer "
        f"label is ANSWER_LABEL={target_label}. The certifying witness is {witness}. This page intentionally "
        "does not repeat the original lookup key; it must be reached through the bridge page.\n"
    )
    decoy_text = (
        f"\nNEAR-TAIL DECOY PAGE. This page repeats lookup key {target_key}, but it is obsolete and "
        f"non-authoritative. It misleadingly lists ANSWER_LABEL={decoy_label}. Do not use this decoy when "
        "the query asks for the authoritative bridge and evidence chain.\n"
    )
    query_text = (
        f"\nQuestion: Follow the AUTHORITATIVE BRIDGE PAGE and its linked AUTHORITATIVE EVIDENCE PAGE. "
        f"What is the ANSWER_LABEL for lookup key {target_key}? Return only one letter. ANSWER_LABEL:"
    )
    query_ids = encode(tokenizer, query_text)
    if len(query_ids) > eval_tokens:
        query_ids = query_ids[-eval_tokens:]

    bridge_len = len(encode(tokenizer, bridge_text))
    answer_len = len(encode(tokenizer, answer_text))
    decoy_len = len(encode(tokenizer, decoy_text))
    targets = [
        ("bridge", bounded_segment_target(context_tokens, bridge_len, bridge_percent), bridge_text),
        ("answer", bounded_segment_target(context_tokens, answer_len, answer_percent), answer_text),
        ("decoy", bounded_segment_target(context_tokens, decoy_len, decoy_percent), decoy_text),
    ]
    distractor_percents = [12, 24, 36, 58, 70, 82]
    for distractor_index, percent in enumerate(distractor_percents):
        if abs(percent - bridge_percent) < 3 or abs(percent - answer_percent) < 3 or abs(percent - decoy_percent) < 3:
            continue
        other_key = make_key(rng, task_id + 10_000 + distractor_index)
        other_artifact = "ART-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
        other_label = rng.choice(LABELS)
        if distractor_index % 2 == 0:
            text = (
                f"\nAUTHORITATIVE BRIDGE PAGE. Lookup key {other_key} routes to controlling artifact code "
                f"{other_artifact}. This bridge belongs to a different lookup key and must not answer "
                f"queries about {target_key}.\n"
            )
        else:
            text = (
                f"\nAUTHORITATIVE EVIDENCE PAGE. For controlling artifact code {other_artifact}, the verified "
                f"answer label is ANSWER_LABEL={other_label}. This evidence page is authoritative only for "
                "its own artifact code.\n"
            )
        targets.append(
            (
                f"distractor_{distractor_index}",
                bounded_segment_target(context_tokens, len(encode(tokenizer, text)), percent),
                text,
            )
        )
    targets.sort(key=lambda item: item[1])

    token_ids: list[int] = []
    filler_idx = 0
    bridge_span: Span | None = None
    answer_span: Span | None = None
    decoy_span: Span | None = None
    for kind, target, text in targets:
        filler_idx = fill_to(token_ids, tokenizer, rng, target, filler_idx)
        span = append_segment(token_ids, tokenizer, text)
        if kind == "bridge":
            bridge_span = span
        elif kind == "answer":
            answer_span = span
        else:
            decoy_span = span
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens, filler_idx)
    if len(token_ids) != context_tokens:
        token_ids = token_ids[:context_tokens]
    if bridge_span is None or answer_span is None or decoy_span is None:
        raise ValueError(f"Chain layout did not create bridge, answer, and decoy spans: {layout}")
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
        evidence_span=answer_span,
        decoy_span=decoy_span,
        target_key=target_key,
        target_label=target_label,
        decoy_label=decoy_label,
        query_text=query_text,
        evidence_spans=[bridge_span, answer_span],
        decoy_spans=[decoy_span],
        task_variant="chain",
        bridge_key=bridge_key,
    )


def build_paraphrased_chain_task_with_layout(
    tokenizer: Any,
    rng: random.Random,
    task_id: int,
    context_tokens: int,
    eval_tokens: int,
    layout: str,
    target_label_override: str | None = None,
    decoy_label_override: str | None = None,
    include_conflict: bool = False,
) -> GeneratedTask:
    parsed = parse_layout(layout) or (5, 90)
    bridge_percent, decoy_percent = parsed
    answer_percent = max(bridge_percent + 8, min(decoy_percent - 10, (bridge_percent + decoy_percent) // 2))
    target_key = make_key(rng, task_id)
    bridge_key = "ART-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
    target_label = target_label_override or rng.choice(LABELS)
    if decoy_label_override and decoy_label_override != target_label:
        decoy_label = decoy_label_override
    else:
        decoy_label = rng.choice([label for label in LABELS if label != target_label])
    topic = rng.choice(TOPICS)
    witness = rng.choice(NAMES)
    bridge_text = (
        f"\nRegistry cross-reference. In the {topic} file, lookup key {target_key} points to "
        f"controlling artifact {bridge_key}. To resolve the lookup, consult the certified entry for "
        f"artifact {bridge_key}; later reminder notes are secondary.\n"
    )
    answer_text = (
        f"\nCertified artifact entry. For artifact {bridge_key}, the approved response letter is "
        f"{target_label}. The reviewing witness is {witness}. This entry is the controlling source and "
        "does not repeat the original lookup key.\n"
    )
    decoy_text = (
        f"\nLate reminder note. This note mentions lookup key {target_key}, but it is outdated and should "
        f"not govern the registry. It suggests response letter {decoy_label}, which is not the certified "
        "entry.\n"
    )
    conflict_text = (
        f"\nSuperseded artifact entry. For artifact {bridge_key}, the former response letter was "
        f"{decoy_label}. This entry is obsolete and is not the controlling source; use the current "
        "certified artifact entry instead.\n"
    )
    query_text = (
        f"\nQuestion: Use the registry cross-reference and the certified artifact entry. What response "
        f"letter should be returned for lookup key {target_key}? Return only one letter. Response letter:"
    )
    query_ids = encode(tokenizer, query_text)
    if len(query_ids) > eval_tokens:
        query_ids = query_ids[-eval_tokens:]

    bridge_len = len(encode(tokenizer, bridge_text))
    answer_len = len(encode(tokenizer, answer_text))
    decoy_len = len(encode(tokenizer, decoy_text))
    targets = [
        ("bridge", bounded_segment_target(context_tokens, bridge_len, bridge_percent), bridge_text),
        ("answer", bounded_segment_target(context_tokens, answer_len, answer_percent), answer_text),
        ("decoy", bounded_segment_target(context_tokens, decoy_len, decoy_percent), decoy_text),
    ]
    if include_conflict:
        conflict_percent = max(bridge_percent + 4, answer_percent - 4)
        targets.append(
            (
                "conflict",
                bounded_segment_target(context_tokens, len(encode(tokenizer, conflict_text)), conflict_percent),
                conflict_text,
            )
        )
    distractor_percents = [12, 24, 36, 58, 70, 82]
    for distractor_index, percent in enumerate(distractor_percents):
        if abs(percent - bridge_percent) < 3 or abs(percent - answer_percent) < 3 or abs(percent - decoy_percent) < 3:
            continue
        other_key = make_key(rng, task_id + 10_000 + distractor_index)
        other_artifact = "ART-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
        other_label = rng.choice(LABELS)
        if distractor_index % 2 == 0:
            text = (
                f"\nRegistry cross-reference. Lookup key {other_key} points to controlling artifact "
                f"{other_artifact}. This cross-reference belongs to a different lookup key and should not "
                f"answer questions about {target_key}.\n"
            )
        else:
            text = (
                f"\nCertified artifact entry. For artifact {other_artifact}, the approved response letter "
                f"is {other_label}. This entry applies only to its own artifact identifier.\n"
            )
        targets.append(
            (
                f"distractor_{distractor_index}",
                bounded_segment_target(context_tokens, len(encode(tokenizer, text)), percent),
                text,
            )
        )
    targets.sort(key=lambda item: item[1])

    token_ids: list[int] = []
    filler_idx = 0
    bridge_span: Span | None = None
    answer_span: Span | None = None
    decoy_span: Span | None = None
    conflict_span: Span | None = None
    for kind, target, text in targets:
        filler_idx = fill_to(token_ids, tokenizer, rng, target, filler_idx)
        span = append_segment(token_ids, tokenizer, text)
        if kind == "bridge":
            bridge_span = span
        elif kind == "answer":
            answer_span = span
        elif kind == "conflict":
            conflict_span = span
        else:
            decoy_span = span
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens, filler_idx)
    if len(token_ids) != context_tokens:
        token_ids = token_ids[:context_tokens]
    if bridge_span is None or answer_span is None or decoy_span is None:
        raise ValueError(f"Paraphrased chain layout did not create bridge, answer, and decoy spans: {layout}")
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
        evidence_span=answer_span,
        decoy_span=decoy_span,
        target_key=target_key,
        target_label=target_label,
        decoy_label=decoy_label,
        query_text=query_text,
        evidence_spans=[bridge_span, answer_span],
        task_variant="chain_para_conflict" if include_conflict else "chain_para",
        bridge_key=bridge_key,
        decoy_spans=[span for span in [decoy_span, conflict_span] if span is not None],
    )


def build_story_chain_task_with_layout(
    tokenizer: Any,
    rng: random.Random,
    task_id: int,
    context_tokens: int,
    eval_tokens: int,
    layout: str,
    target_label_override: str | None = None,
    decoy_label_override: str | None = None,
) -> GeneratedTask:
    parsed = parse_layout(layout) or (5, 90)
    bridge_percent, decoy_percent = parsed
    answer_percent = max(bridge_percent + 8, min(decoy_percent - 10, (bridge_percent + decoy_percent) // 2))
    target_key = make_key(rng, task_id)
    bridge_key = "RIVER-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    target_label = target_label_override or rng.choice(LABELS)
    if decoy_label_override and decoy_label_override != target_label:
        decoy_label = decoy_label_override
    else:
        decoy_label = rng.choice([label for label in LABELS if label != target_label])
    topic = rng.choice(TOPICS)
    witness = rng.choice(NAMES)
    station = rng.choice(["north annex", "harbor desk", "west archive", "garden office", "signal room"])
    bridge_text = (
        f"\nField ledger note. In the {topic} folder, badge {target_key} was logged at the {station} "
        f"under river-name {bridge_key}. The clerk wrote that the final ruling is not on this note; "
        f"it appears in the later memo that discusses {bridge_key} by name.\n"
    )
    answer_text = (
        f"\nResolution memo. The river-name {bridge_key} closes with option {target_label} after "
        f"{witness} reconciled the file. This is the current ruling for that river-name, even though "
        "the badge number is not repeated here.\n"
    )
    decoy_text = (
        f"\nOld desk slip. Badge {target_key} is mentioned with option {decoy_label}, but the slip was "
        "withdrawn before the file was reconciled. It should not decide the current ruling.\n"
    )
    conflict_text = (
        f"\nEarlier ruling note. The river-name {bridge_key} once leaned toward option {decoy_label}. "
        "That note is superseded by the later resolution memo and is no longer current.\n"
    )
    query_text = (
        f"\nQuestion: Follow the badge through its river-name and use the current ruling. Which option "
        f"letter should be returned for badge {target_key}? Return only one letter. Option:"
    )
    query_ids = encode(tokenizer, query_text)
    if len(query_ids) > eval_tokens:
        query_ids = query_ids[-eval_tokens:]

    bridge_len = len(encode(tokenizer, bridge_text))
    answer_len = len(encode(tokenizer, answer_text))
    decoy_len = len(encode(tokenizer, decoy_text))
    conflict_len = len(encode(tokenizer, conflict_text))
    conflict_percent = max(bridge_percent + 4, answer_percent - 4)
    targets = [
        ("bridge", bounded_segment_target(context_tokens, bridge_len, bridge_percent), bridge_text),
        ("answer", bounded_segment_target(context_tokens, answer_len, answer_percent), answer_text),
        ("decoy", bounded_segment_target(context_tokens, decoy_len, decoy_percent), decoy_text),
        ("conflict", bounded_segment_target(context_tokens, conflict_len, conflict_percent), conflict_text),
    ]
    distractor_percents = [12, 24, 36, 58, 70, 82]
    for distractor_index, percent in enumerate(distractor_percents):
        if abs(percent - bridge_percent) < 3 or abs(percent - answer_percent) < 3 or abs(percent - decoy_percent) < 3:
            continue
        other_key = make_key(rng, task_id + 20_000 + distractor_index)
        other_bridge = "RIVER-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        other_label = rng.choice(LABELS)
        if distractor_index % 2 == 0:
            text = (
                f"\nField ledger note. Badge {other_key} was logged under river-name {other_bridge}. "
                f"This line belongs to a different badge and should not answer questions about {target_key}.\n"
            )
        else:
            text = (
                f"\nResolution memo. The river-name {other_bridge} closes with option {other_label}. "
                "This ruling applies only to its own river-name.\n"
            )
        targets.append(
            (
                f"distractor_{distractor_index}",
                bounded_segment_target(context_tokens, len(encode(tokenizer, text)), percent),
                text,
            )
        )
    targets.sort(key=lambda item: item[1])

    token_ids: list[int] = []
    filler_idx = 0
    bridge_span: Span | None = None
    answer_span: Span | None = None
    decoy_span: Span | None = None
    conflict_span: Span | None = None
    for kind, target, text in targets:
        filler_idx = fill_to(token_ids, tokenizer, rng, target, filler_idx)
        span = append_segment(token_ids, tokenizer, text)
        if kind == "bridge":
            bridge_span = span
        elif kind == "answer":
            answer_span = span
        elif kind == "conflict":
            conflict_span = span
        else:
            decoy_span = span
    filler_idx = fill_to(token_ids, tokenizer, rng, context_tokens, filler_idx)
    if len(token_ids) != context_tokens:
        token_ids = token_ids[:context_tokens]
    if bridge_span is None or answer_span is None or decoy_span is None or conflict_span is None:
        raise ValueError(f"Story chain layout did not create all required spans: {layout}")
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
        evidence_span=answer_span,
        decoy_span=decoy_span,
        target_key=target_key,
        target_label=target_label,
        decoy_label=decoy_label,
        query_text=query_text,
        evidence_spans=[bridge_span, answer_span],
        task_variant="chain_story_conflict",
        bridge_key=bridge_key,
        decoy_spans=[decoy_span, conflict_span],
    )


def build_keep_mask(
    query_tokens: torch.Tensor,
    key_count: int,
    ctx: SparseContext,
    device: torch.device,
) -> torch.Tensor:
    query_count = int(query_tokens.numel())
    keep = torch.zeros((query_count, key_count), dtype=torch.bool, device=device)
    if ctx.mode == "full":
        keep[:, :] = True
        return keep
    if ctx.sink_tokens > 0:
        keep[:, : min(ctx.sink_tokens, key_count)] = True
    if ctx.keep_remote_tokens:
        remote = [idx for idx in ctx.keep_remote_tokens if 0 <= idx < key_count]
        if remote:
            keep[:, torch.tensor(remote, dtype=torch.long, device=device)] = True
    for row, query_token in enumerate(query_tokens.detach().cpu().tolist()):
        history_count = min(key_count, int(query_token) + 1)
        if ctx.recent_tokens > 0:
            start = max(0, history_count - ctx.recent_tokens)
            keep[row, start:history_count] = True
        if int(query_token) < key_count:
            keep[row, int(query_token)] = True
        keep[row, history_count:] = False
    return keep


def candidate_ids_for_single_query(key_count: int, ctx: SparseContext, device: torch.device) -> torch.Tensor:
    if ctx.mode == "full":
        return torch.arange(key_count, dtype=torch.long, device=device)
    history_count = key_count
    candidates: set[int] = set()
    if ctx.sink_tokens > 0:
        candidates.update(range(0, min(ctx.sink_tokens, key_count)))
    if ctx.keep_remote_tokens:
        candidates.update(idx for idx in ctx.keep_remote_tokens if 0 <= idx < history_count)
    if ctx.recent_tokens > 0:
        candidates.update(range(max(0, history_count - ctx.recent_tokens), history_count))
    if history_count > 0:
        candidates.add(history_count - 1)
    if not candidates:
        candidates.add(max(0, history_count - 1))
    return torch.tensor(sorted(candidates), dtype=torch.long, device=device)


def range_candidate_ids_for_single_query(key_count: int, ctx: SparseContext, device: torch.device) -> torch.Tensor:
    if ctx.mode == "full":
        return torch.arange(key_count, dtype=torch.long, device=device)
    ranges: list[tuple[int, int]] = []
    if ctx.sink_tokens > 0:
        ranges.append((0, min(ctx.sink_tokens, key_count)))
    for start, end in ctx.keep_remote_ranges:
        bounded_start = max(0, min(int(start), key_count))
        bounded_end = max(0, min(int(end), key_count))
        if bounded_end > bounded_start:
            ranges.append((bounded_start, bounded_end))
    if ctx.recent_tokens > 0:
        ranges.append((max(0, key_count - ctx.recent_tokens), key_count))
    if key_count > 0:
        ranges.append((key_count - 1, key_count))
    if not ranges:
        return torch.zeros((1,), dtype=torch.long, device=device)
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    parts = [torch.arange(start, end, dtype=torch.long, device=device) for start, end in merged if end > start]
    if not parts:
        return torch.zeros((1,), dtype=torch.long, device=device)
    return torch.cat(parts)


if triton is not None:
    @triton.jit
    def _triton_page_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        candidate_ids_ptr,
        out_ptr,
        scaling: tl.constexpr,
        key_count: tl.constexpr,
        query_count: tl.constexpr,
        head_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        group_size: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        q_block = tl.program_id(1)
        batch = 0
        head = bh
        kv_head = head // group_size
        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = (offs_m < query_count)[:, None] & (offs_d < head_dim)[None, :]
        q = tl.load(
            q_ptr + batch * stride_qb + head * stride_qh + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        visible = key_count - query_count + offs_m + 1

        for start_n in range(0, candidate_count, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < candidate_count
            candidate_ids = tl.load(candidate_ids_ptr + offs_n, mask=n_mask, other=0)
            k = tl.load(
                k_ptr
                + batch * stride_kb
                + kv_head * stride_kh
                + candidate_ids[:, None] * stride_kk
                + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scaling
            causal = candidate_ids[None, :] < visible[:, None]
            scores = tl.where((offs_m[:, None] < query_count) & n_mask[None, :] & causal, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(
                v_ptr
                + batch * stride_vb
                + kv_head * stride_vh
                + candidate_ids[:, None] * stride_vk
                + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr + batch * stride_ob + head * stride_oh + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            out,
            mask=(offs_m[:, None] < query_count) & (offs_d[None, :] < head_dim),
        )


def triton_page_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    ctx: SparseContext,
) -> tuple[torch.Tensor, None]:
    if triton is None:
        raise RuntimeError("sparse_attention_impl=triton requires the triton package.")
    batch, attention_heads, query_count, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    key_count = key_states.shape[-2]
    if batch != 1:
        raise RuntimeError("triton sparse page attention currently supports batch size 1.")
    if query_count != 1:
        raise RuntimeError("triton sparse page attention currently supports decode/query_count=1.")
    if head_dim > 128:
        raise RuntimeError(f"triton sparse page attention supports head_dim <= 128, got {head_dim}.")
    candidate_ids = candidate_ids_for_single_query(key_count, ctx, query_states.device)
    if ctx.stats is not None:
        ctx.stats.add_count(int(candidate_ids.numel()), key_count)
    group_size = attention_heads // kv_heads
    out = torch.empty_like(query_states)
    block_d = triton.next_power_of_2(head_dim)
    block_m = 16
    block_n = 64
    grid = (batch * attention_heads, triton.cdiv(query_count, block_m))
    _triton_page_attention_kernel[grid](
        query_states,
        key_states,
        value_states,
        candidate_ids,
        out,
        float(scaling),
        key_count,
        query_count,
        head_dim,
        int(candidate_ids.numel()),
        group_size,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out.transpose(1, 2).contiguous(), None


def _patched_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if (
        _ACTIVE_SPARSE_CONTEXT is not None
        and _ACTIVE_SPARSE_CONTEXT.sparse_impl == "triton"
        and _ACTIVE_SPARSE_CONTEXT.mode != "full"
    ):
        try:
            return triton_page_attention(query_states, key_states, value_states, scaling, _ACTIVE_SPARSE_CONTEXT)
        except RuntimeError:
            pass
    if (
        _ACTIVE_SPARSE_CONTEXT is not None
        and _ACTIVE_SPARSE_CONTEXT.sparse_impl == "range_sdpa"
        and _ACTIVE_SPARSE_CONTEXT.mode != "full"
    ):
        query_count = query_states.shape[-2]
        key_count = key_states.shape[-2]
        if query_count == 1:
            keep_indices = range_candidate_ids_for_single_query(key_count, _ACTIVE_SPARSE_CONTEXT, query_states.device)
            if _ACTIVE_SPARSE_CONTEXT.stats is not None:
                _ACTIVE_SPARSE_CONTEXT.stats.add_count(int(keep_indices.numel()), key_count)
            gathered_key = key_states.index_select(2, keep_indices)
            gathered_value = value_states.index_select(2, keep_indices)
            if gathered_key.shape[1] != query_states.shape[1]:
                repeat_groups = query_states.shape[1] // gathered_key.shape[1]
                kv_head_index = torch.div(
                    torch.arange(query_states.shape[1], device=query_states.device),
                    repeat_groups,
                    rounding_mode="floor",
                )
                gathered_key = gathered_key.index_select(1, kv_head_index)
                gathered_value = gathered_value.index_select(1, kv_head_index)
            gathered_mask = attention_mask[:, :, :, keep_indices] if attention_mask is not None else None
            attention_output = F.scaled_dot_product_attention(
                query_states,
                gathered_key,
                gathered_value,
                attn_mask=gathered_mask,
                dropout_p=dropout if module.training else 0.0,
                is_causal=False,
                scale=scaling,
            )
            attention_output = attention_output.transpose(1, 2).contiguous()
            return attention_output, None
    if (
        _ACTIVE_SPARSE_CONTEXT is not None
        and _ACTIVE_SPARSE_CONTEXT.sparse_impl in {"gather", "sdpa_gather"}
        and _ACTIVE_SPARSE_CONTEXT.mode != "full"
    ):
        query_count = query_states.shape[-2]
        key_count = key_states.shape[-2]
        chunk_query_start = key_count - query_count
        query_tokens = torch.arange(chunk_query_start, chunk_query_start + query_count, device=query_states.device)
        keep = build_keep_mask(query_tokens, key_count, _ACTIVE_SPARSE_CONTEXT, query_states.device)
        if query_count == 1:
            if _ACTIVE_SPARSE_CONTEXT.stats is not None:
                _ACTIVE_SPARSE_CONTEXT.stats.add(keep, key_count)
            keep_indices = torch.nonzero(keep[0], as_tuple=False).flatten()
            gathered_key = key_states.index_select(2, keep_indices)
            gathered_value = value_states.index_select(2, keep_indices)
            if gathered_key.shape[1] != query_states.shape[1]:
                repeat_groups = query_states.shape[1] // gathered_key.shape[1]
                kv_head_index = torch.div(
                    torch.arange(query_states.shape[1], device=query_states.device),
                    repeat_groups,
                    rounding_mode="floor",
                )
                gathered_key = gathered_key.index_select(1, kv_head_index)
                gathered_value = gathered_value.index_select(1, kv_head_index)
            if _ACTIVE_SPARSE_CONTEXT.sparse_impl == "sdpa_gather":
                gathered_mask = attention_mask[:, :, :, keep_indices] if attention_mask is not None else None
                attention_output = F.scaled_dot_product_attention(
                    query_states,
                    gathered_key,
                    gathered_value,
                    attn_mask=gathered_mask,
                    dropout_p=dropout if module.training else 0.0,
                    is_causal=False,
                    scale=scaling,
                )
                attention_output = attention_output.transpose(1, 2).contiguous()
                return attention_output, None
            scores = torch.matmul(query_states, gathered_key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, keep_indices]
            attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if dropout and module.training:
                attention_weights = F.dropout(attention_weights, p=dropout, training=True)
            attention_output = torch.matmul(attention_weights, gathered_value)
            attention_output = attention_output.transpose(1, 2).contiguous()
            return attention_output, attention_weights
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _ACTIVE_SPARSE_CONTEXT is not None:
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        chunk_query_start = key_count - query_count
        query_tokens = torch.arange(chunk_query_start, chunk_query_start + query_count, device=scores.device)
        keep = build_keep_mask(query_tokens, key_count, _ACTIVE_SPARSE_CONTEXT, scores.device)
        if _ACTIVE_SPARSE_CONTEXT.stats is not None:
            _ACTIVE_SPARSE_CONTEXT.stats.add(keep, key_count)
        scores = scores.masked_fill(~keep.view(1, 1, query_count, key_count), torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_qwen3_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import transformers.models.qwen3.modeling_qwen3.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = getattr(modeling_qwen3, "eager_attention_forward")
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def sparse_context(ctx: SparseContext | None):
    global _ACTIVE_SPARSE_CONTEXT
    previous = _ACTIVE_SPARSE_CONTEXT
    _ACTIVE_SPARSE_CONTEXT = ctx
    try:
        yield
    finally:
        _ACTIVE_SPARSE_CONTEXT = previous


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_device: torch.device,
    chunk_size: int,
    past_key_values: Any = None,
    score_tokens: bool = False,
    prev_logits: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor, float, int]:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, input_ids.shape[-1], chunk_size):
        end = min(start + chunk_size, input_ids.shape[-1])
        chunk = input_ids[:, start:end].to(input_device)
        if score_tokens and prev_logits is not None:
            for pos in range(chunk.shape[-1]):
                token = chunk[:, pos : pos + 1]
                total_nll += float(F.cross_entropy(prev_logits.float(), token.reshape(-1), reduction="sum").item())
                total_tokens += 1
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
                del outputs
            continue
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
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
    if prev_logits is None:
        raise ValueError("No logits produced.")
    return past_key_values, prev_logits, total_nll, total_tokens


def clone_past(past_key_values: Any) -> Any:
    return copy.deepcopy(past_key_values)


@torch.inference_mode()
def score_option(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    past_key_values: Any,
    prev_logits: torch.Tensor,
    option: str,
) -> float:
    ids = tokenizer(option, return_tensors="pt", add_special_tokens=False)["input_ids"]
    _, _, nll, _ = run_tokens(
        model,
        ids,
        input_device,
        chunk_size=1,
        past_key_values=past_key_values,
        score_tokens=True,
        prev_logits=prev_logits,
    )
    return -nll


def option_text(label: str, answer_score_format: str) -> str:
    if answer_score_format == "gated_sentence":
        answer_score_format = "answer_label"
    if answer_score_format == "letter":
        return " " + label
    if answer_score_format == "answer_label":
        return f" ANSWER_LABEL={label}"
    if answer_score_format == "sentence":
        return f" The authoritative answer label is {label}."
    raise ValueError(f"Unknown answer_score_format: {answer_score_format}")


def top_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def text_verifier_label(task: GeneratedTask, pages: list[Any], selected_pages: set[int], mode: str) -> str:
    if mode == "full":
        page_ids = range(len(pages))
    elif selected_pages:
        page_ids = sorted(selected_pages)
    else:
        return ""
    page_texts = [(page_id, joined(task.token_texts, pages[page_id].start, pages[page_id].end)) for page_id in page_ids]
    if getattr(task, "task_variant", "single") in CHAIN_VARIANTS:
        bridge_key = getattr(task, "bridge_key", "")
        if not bridge_key:
            target_key = getattr(task, "target_key", "")
            for _, text in page_texts:
                if target_key and target_key in text:
                    match = re.search(r"(?:artifact code|artifact|river-name)\s+([A-Z0-9-]+)", text)
                    if match:
                        bridge_key = match.group(1)
                        break
        if bridge_key:
            for _, text in page_texts:
                if bridge_key not in text:
                    continue
                lowered = text.lower()
                if (
                    "superseded" in lowered
                    or "obsolete" in lowered
                    or "outdated" in lowered
                    or "former response" in lowered
                    or "not the controlling" in lowered
                    or "withdrawn" in lowered
                    or "earlier ruling" in lowered
                    or "no longer current" in lowered
                ):
                    continue
                match = re.search(
                    r"(?:ANSWER_LABEL=|approved response letter is\s+|response letter\s+|closes with option\s+|option\s+)([A-D])",
                    text,
                    flags=re.IGNORECASE,
                )
                if match:
                    return match.group(1).upper()
    for page_id in page_ids:
        text = next(text for candidate_page_id, text in page_texts if candidate_page_id == page_id)
        if "AUTHORITATIVE EVIDENCE PAGE" not in text and "certified artifact entry" not in text.lower():
            continue
        match = re.search(
            r"(?:ANSWER_LABEL=|approved response letter is\s+|response letter\s+|closes with option\s+|option\s+)([A-D])",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    return ""


def selected_page_texts(task: GeneratedTask, pages: list[Any], selected_pages: set[int], mode: str) -> list[tuple[int, str]]:
    if mode == "full":
        page_ids = range(len(pages))
    elif selected_pages:
        page_ids = sorted(selected_pages)
    else:
        return []
    return [(page_id, joined(task.token_texts, pages[page_id].start, pages[page_id].end)) for page_id in page_ids]


def negative_page_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "late reminder" in lowered
        or "near-tail decoy" in lowered
        or "superseded" in lowered
        or "obsolete" in lowered
        or "outdated" in lowered
        or "former response" in lowered
        or "not the controlling" in lowered
        or "old desk slip" in lowered
        or "withdrawn" in lowered
        or "earlier ruling" in lowered
        or "no longer current" in lowered
        or "should not answer" in lowered
        or "must not answer" in lowered
        or "different lookup key" in lowered
        or "different badge" in lowered
        or "applies only to its own" in lowered
    )


def extract_option_label(text: str) -> str:
    match = re.search(
        r"(?:ANSWER_LABEL=|approved response letter is\s+|response letter\s+|closes with option\s+|option\s+)([A-D])",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def extract_bridge_artifact_from_text(text: str) -> str:
    match = re.search(r"(?:artifact code|artifact|river-name)\s+([A-Z0-9-]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def typed_context_summary_lines(
    task: GeneratedTask,
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
    max_lines: int = 8,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for page_id, text in page_texts:
        lowered = text.lower()
        option = extract_option_label(text)
        artifact = bridge_artifact if bridge_artifact and bridge_artifact in text else extract_bridge_artifact_from_text(text)
        is_negative = negative_page_text(text)
        if bridge_artifact and bridge_artifact in text and option and not is_negative:
            line = (
                f"page={page_id}; role=current_ruling; alias={bridge_artifact}; "
                f"ANSWER_LABEL={option}; status=current"
            )
        elif target_key and target_key in text and artifact and not is_negative:
            line = f"page={page_id}; role=bridge; lookup_key={target_key}; alias={artifact}; status=route_only"
        elif target_key and target_key in text and option and is_negative:
            line = (
                f"page={page_id}; role=withdrawn_badge_note; lookup_key={target_key}; "
                f"option={option}; status=non_current"
            )
        elif bridge_artifact and bridge_artifact in text and option and is_negative:
            line = (
                f"page={page_id}; role=superseded_alias_note; alias={bridge_artifact}; "
                f"option={option}; status=non_current"
            )
        else:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    if answer_label and not any("ANSWER_LABEL=" in line for line in lines):
        lines.insert(0, f"role=current_ruling; ANSWER_LABEL={answer_label}; status=current")
    return lines


def typed_context_mini_summary(
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
) -> str:
    withdrawn_options: list[str] = []
    superseded_options: list[str] = []
    for _, text in page_texts:
        option = extract_option_label(text)
        if not option:
            continue
        is_negative = negative_page_text(text)
        if not is_negative:
            continue
        if target_key and target_key in text and option not in withdrawn_options:
            withdrawn_options.append(option)
        if bridge_artifact and bridge_artifact in text and option not in superseded_options:
            superseded_options.append(option)
    parts = [
        f"key={target_key}",
        f"alias={bridge_artifact}",
        f"current={answer_label}",
    ]
    if withdrawn_options:
        parts.append("withdrawn_noncurrent=" + "/".join(withdrawn_options[:2]))
    if superseded_options:
        parts.append("superseded_noncurrent=" + "/".join(superseded_options[:2]))
    parts.append("rule=current_only")
    return "\nTyped memory mini: " + "; ".join(parts) + ".\n"


def typed_context_short_summary(
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
) -> str:
    withdrawn_options: list[str] = []
    superseded_options: list[str] = []
    for _, text in page_texts:
        option = extract_option_label(text)
        if not option or not negative_page_text(text):
            continue
        if target_key and target_key in text and option not in withdrawn_options:
            withdrawn_options.append(option)
        if bridge_artifact and bridge_artifact in text and option not in superseded_options:
            superseded_options.append(option)
    lines = [
        f"Typed memory summary: badge {target_key} routes to river-name {bridge_artifact}.",
        f"The current ruling for {bridge_artifact} is option {answer_label}.",
    ]
    if withdrawn_options:
        lines.append(
            f"Old badge option {'/'.join(withdrawn_options[:2])} is withdrawn and non-current."
        )
    if superseded_options:
        lines.append(
            f"Earlier {bridge_artifact} option {'/'.join(superseded_options[:2])} is superseded and non-current."
        )
    lines.append("Answer from the current ruling only.")
    return "\n" + " ".join(lines) + "\n"


def typed_context_lite_summary(
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
) -> str:
    withdrawn_options: list[str] = []
    superseded_options: list[str] = []
    for _, text in page_texts:
        option = extract_option_label(text)
        if not option or not negative_page_text(text):
            continue
        if target_key and target_key in text and option not in withdrawn_options:
            withdrawn_options.append(option)
        if bridge_artifact and bridge_artifact in text and option not in superseded_options:
            superseded_options.append(option)
    facts = [
        f"lookup_key={target_key}",
        f"BRIDGE_ALIAS={bridge_artifact}",
        f"ANSWER_LABEL={answer_label}; status=current",
    ]
    if withdrawn_options:
        facts.append(f"withdrawn_badge_option={'/'.join(withdrawn_options[:2])}; status=non_current")
    if superseded_options:
        facts.append(f"superseded_alias_option={'/'.join(superseded_options[:2])}; status=non_current")
    facts.append("rule=use_current_status_only")
    return "\nTyped memory lite: " + "; ".join(facts) + ".\n"


def typed_context_natural_summary(
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
) -> str:
    withdrawn_options: list[str] = []
    superseded_options: list[str] = []
    for _, text in page_texts:
        option = extract_option_label(text)
        if not option or not negative_page_text(text):
            continue
        if target_key and target_key in text and option not in withdrawn_options:
            withdrawn_options.append(option)
        if bridge_artifact and bridge_artifact in text and option not in superseded_options:
            superseded_options.append(option)
    lines = [
        f"Typed memory summary: badge {target_key} routes to river-name {bridge_artifact}.",
        f"The current ruling for {bridge_artifact} is option {answer_label} "
        f"(ANSWER_LABEL={answer_label}; status=current).",
    ]
    if withdrawn_options:
        lines.append(
            f"Old badge option {'/'.join(withdrawn_options[:2])} is withdrawn with status=non_current."
        )
    if superseded_options:
        lines.append(
            f"Earlier {bridge_artifact} option {'/'.join(superseded_options[:2])} is superseded with status=non_current."
        )
    lines.append("Answer only from status=current.")
    return "\n" + " ".join(lines) + "\n"


def typed_context_answerline_summary(
    page_texts: list[tuple[int, str]],
    target_key: str,
    bridge_artifact: str,
    answer_label: str,
) -> str:
    withdrawn_options: list[str] = []
    superseded_options: list[str] = []
    for _, text in page_texts:
        option = extract_option_label(text)
        if not option or not negative_page_text(text):
            continue
        if target_key and target_key in text and option not in withdrawn_options:
            withdrawn_options.append(option)
        if bridge_artifact and bridge_artifact in text and option not in superseded_options:
            superseded_options.append(option)
    lines = [
        f"ANSWER_LABEL={answer_label}; status=current.",
        f"Badge {target_key} routes to river-name {bridge_artifact}; current ruling for that river-name is option {answer_label}.",
    ]
    if withdrawn_options:
        lines.append(f"Withdrawn badge option {'/'.join(withdrawn_options[:2])} has status=non_current.")
    if superseded_options:
        lines.append(f"Superseded {bridge_artifact} option {'/'.join(superseded_options[:2])} has status=non_current.")
    lines.append("Use the current status only.")
    return "\nTyped memory summary: " + " ".join(lines) + "\n"


def build_typed_record(
    task: GeneratedTask,
    pages: list[Any],
    selected_pages: set[int],
    mode: str,
    typed_record_mode: str,
    typed_record_format: str,
) -> tuple[str, dict[str, Any]]:
    if typed_record_mode == "none":
        return "", {
            "typed_record_present": 0,
            "typed_record_answer_label": "",
            "typed_record_correct": 0,
            "typed_record_decoy_pred": 0,
            "typed_record_bridge_artifact": "",
        }
    page_texts = selected_page_texts(task, pages, selected_pages, mode)
    target_key = getattr(task, "target_key", "")
    bridge_artifact = ""
    answer_label = ""
    if getattr(task, "task_variant", "single") in CHAIN_VARIANTS:
        route_patterns = [
            re.compile(
                rf"lookup key\s+{re.escape(target_key)}\s+routes to controlling artifact code\s+([A-Z0-9-]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"lookup key\s+{re.escape(target_key)}\s+points to controlling artifact\s+([A-Z0-9-]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"lookup key\s+{re.escape(target_key)}[^.\n]*artifact(?: code)?\s+([A-Z0-9-]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                rf"badge\s+{re.escape(target_key)}[^.\n]*river-name\s+([A-Z0-9-]+)",
                flags=re.IGNORECASE,
            ),
        ]
        for _, text in page_texts:
            for route_pattern in route_patterns:
                match = route_pattern.search(text)
                if match:
                    bridge_artifact = match.group(1)
                    break
            if bridge_artifact:
                break
        if not bridge_artifact:
            for _, text in page_texts:
                lowered = text.lower()
                if (
                    "authoritative bridge page" not in lowered
                    and "registry cross-reference" not in lowered
                    and "field ledger note" not in lowered
                    or not target_key
                    or target_key not in text
                    or "different lookup key" in lowered
                    or "must not answer" in lowered
                    or "should not answer" in lowered
                ):
                    continue
                match = re.search(r"(?:artifact code|artifact|river-name)\s+([A-Z0-9-]+)", text, flags=re.IGNORECASE)
                if match:
                    bridge_artifact = match.group(1)
                    break
        if bridge_artifact:
            for _, text in page_texts:
                if bridge_artifact not in text:
                    continue
                lowered = text.lower()
                if (
                    "superseded" in lowered
                    or "obsolete" in lowered
                    or "outdated" in lowered
                    or "former response" in lowered
                    or "not the controlling" in lowered
                    or "withdrawn" in lowered
                    or "earlier ruling" in lowered
                    or "no longer current" in lowered
                ):
                    continue
                match = re.search(
                    r"(?:ANSWER_LABEL=|approved response letter is\s+|response letter\s+|closes with option\s+|option\s+)([A-D])",
                    text,
                    flags=re.IGNORECASE,
                )
                if match:
                    answer_label = match.group(1).upper()
                    break
    else:
        for _, text in page_texts:
            if "AUTHORITATIVE EVIDENCE PAGE" not in text and "certified artifact entry" not in text.lower():
                continue
            if target_key and target_key not in text:
                continue
            match = re.search(
                r"(?:ANSWER_LABEL=|approved response letter is\s+|response letter\s+)([A-D])",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                answer_label = match.group(1).upper()
                break
    if not answer_label:
        return "", {
            "typed_record_present": 0,
            "typed_record_answer_label": "",
            "typed_record_correct": 0,
            "typed_record_decoy_pred": 0,
            "typed_record_bridge_artifact": bridge_artifact,
        }
    if typed_record_format == "label_only":
        record = f"\nANSWER_LABEL={answer_label}\n"
    elif typed_record_format == "compact":
        if getattr(task, "task_variant", "single") == "chain":
            record = (
                "\nTyped memory: "
                f"lookup_key={target_key}; BRIDGE_ARTIFACT={bridge_artifact}; "
                f"ANSWER_LABEL={answer_label}; authority_status=authoritative_chain.\n"
            )
        else:
            record = (
                "\nTyped memory: "
                f"lookup_key={target_key}; ANSWER_LABEL={answer_label}; "
                "authority_status=authoritative_evidence.\n"
            )
    elif typed_record_format == "summary":
        summary_lines = typed_context_summary_lines(
            task,
            page_texts,
            target_key,
            bridge_artifact,
            answer_label,
        )
        record = (
            "\nTyped memory summary: "
            f"lookup_key={target_key}; BRIDGE_ALIAS={bridge_artifact}; ANSWER_LABEL={answer_label}.\n"
            + "\n".join(f"- {line}" for line in summary_lines)
            + "\nRule: answer only from status=current; ignore status=non_current as answers.\n"
        )
    elif typed_record_format == "mini_summary":
        record = typed_context_mini_summary(page_texts, target_key, bridge_artifact, answer_label)
    elif typed_record_format == "short_summary":
        record = typed_context_short_summary(page_texts, target_key, bridge_artifact, answer_label)
    elif typed_record_format == "lite_summary":
        record = typed_context_lite_summary(page_texts, target_key, bridge_artifact, answer_label)
    elif typed_record_format == "natural_summary":
        record = typed_context_natural_summary(page_texts, target_key, bridge_artifact, answer_label)
    elif typed_record_format == "answerline_summary":
        record = typed_context_answerline_summary(page_texts, target_key, bridge_artifact, answer_label)
    elif getattr(task, "task_variant", "single") in CHAIN_VARIANTS:
        record = (
            "\nTyped memory record:\n"
            f"lookup_key={target_key}\n"
            f"BRIDGE_ARTIFACT={bridge_artifact}\n"
            f"ANSWER_LABEL={answer_label}\n"
            "authority_status=authoritative_chain\n"
            "Use this typed memory record to answer the next question.\n"
        )
    elif typed_record_format != "verbose":
        raise ValueError(f"Unknown typed_record_format: {typed_record_format}")
    else:
        record = (
            "\nTyped memory record:\n"
            f"lookup_key={target_key}\n"
            f"ANSWER_LABEL={answer_label}\n"
            "authority_status=authoritative_evidence\n"
            "Use this typed memory record to answer the next question.\n"
        )
    return record, {
        "typed_record_present": 1,
        "typed_record_answer_label": answer_label,
        "typed_record_correct": int(answer_label == task.target_label),
        "typed_record_decoy_pred": int(answer_label == task.decoy_label),
        "typed_record_bridge_artifact": bridge_artifact,
    }


def span_pages(pages: list[Any], spans: list[Span] | None, fallback: Span) -> list[int]:
    out: list[int] = []
    for span in spans or [fallback]:
        page_id = overlap_page(pages, span)
        if page_id is not None:
            out.append(page_id)
    return out


def selected_page_coverage(page_ids: list[int], selected_pages: set[int], mode: str) -> float:
    if not page_ids:
        return 0.0
    if mode == "full":
        return 1.0
    return sum(1 for page_id in page_ids if page_id in selected_pages) / len(page_ids)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str, str, int, int, int, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        group = grouped[
            (
                int(row["context_tokens"]),
                str(row.get("task_variant", "single")),
                str(row.get("typed_record_mode", "none")),
                str(row.get("typed_record_format", "verbose")),
                str(row.get("typed_summary_source_mode", "")),
                int(row.get("typed_record_answer_override", 0)),
                int(row.get("typed_record_insert", 1)),
                int(row.get("skip_lm_answer_when_override", 0)),
                str(row["mode"]),
                str(row.get("sparse_attention_impl", "mask")),
            )
        ]
        group["tasks"] += 1
        for field in [
            "correct",
            "calibrated_correct",
            "decoy_pred",
            "calibrated_decoy_pred",
            "text_verifier_present",
            "text_verifier_correct",
            "text_verifier_decoy_pred",
            "typed_record_present",
            "typed_record_correct",
            "typed_record_decoy_pred",
            "typed_record_tokens",
            "answer_override_applied",
            "lm_answer_scored",
            "sentence_gate_triggered",
            "base_calibrated_margin",
            "query_nll",
            "query_tokens",
            "selected_pages",
            "selected_remote_tokens",
            "evidence_hit",
            "evidence_page_coverage",
            "decoy_hit",
            "eval_seconds",
            "mean_kept_fraction",
            "mean_kept_tokens",
            "margin_true_minus_decoy",
            "calibrated_margin_true_minus_decoy",
        ]:
            group[field] += float(row[field])
    out = []
    for (
        context_tokens,
        task_variant,
        typed_record_mode,
        typed_record_format,
        typed_summary_source_mode,
        typed_record_answer_override,
        typed_record_insert,
        skip_lm_answer_when_override,
        mode,
        sparse_attention_impl,
    ), group in sorted(grouped.items()):
        tasks = group["tasks"]
        query_tokens = group["query_tokens"]
        out.append(
            {
                "context_tokens": context_tokens,
                "task_variant": task_variant,
                "typed_record_mode": typed_record_mode,
                "typed_record_format": typed_record_format,
                "typed_summary_source_mode": typed_summary_source_mode,
                "typed_record_answer_override": typed_record_answer_override,
                "typed_record_insert": typed_record_insert,
                "skip_lm_answer_when_override": skip_lm_answer_when_override,
                "mode": mode,
                "sparse_attention_impl": sparse_attention_impl,
                "tasks": int(tasks),
                "accuracy": group["correct"] / tasks if tasks else 0.0,
                "calibrated_accuracy": group["calibrated_correct"] / tasks if tasks else 0.0,
                "text_verifier_coverage": group["text_verifier_present"] / tasks if tasks else 0.0,
                "text_verifier_accuracy": group["text_verifier_correct"] / tasks if tasks else 0.0,
                "typed_record_coverage": group["typed_record_present"] / tasks if tasks else 0.0,
                "typed_record_accuracy": group["typed_record_correct"] / tasks if tasks else 0.0,
                "typed_record_decoy_pred_rate": group["typed_record_decoy_pred"] / tasks if tasks else 0.0,
                "mean_typed_record_tokens": group["typed_record_tokens"] / tasks if tasks else 0.0,
                "answer_override_rate": group["answer_override_applied"] / tasks if tasks else 0.0,
                "lm_answer_scored_rate": group["lm_answer_scored"] / tasks if tasks else 0.0,
                "sentence_gate_rate": group["sentence_gate_triggered"] / tasks if tasks else 0.0,
                "mean_base_calibrated_margin": group["base_calibrated_margin"] / tasks if tasks else 0.0,
                "decoy_pred_rate": group["decoy_pred"] / tasks if tasks else 0.0,
                "calibrated_decoy_pred_rate": group["calibrated_decoy_pred"] / tasks if tasks else 0.0,
                "text_verifier_decoy_pred_rate": group["text_verifier_decoy_pred"] / tasks if tasks else 0.0,
                "query_ppl": math.exp(group["query_nll"] / query_tokens) if query_tokens else 0.0,
                "mean_selected_pages": group["selected_pages"] / tasks if tasks else 0.0,
                "mean_selected_remote_tokens": group["selected_remote_tokens"] / tasks if tasks else 0.0,
                "evidence_hit_rate": group["evidence_hit"] / tasks if tasks else 0.0,
                "evidence_page_coverage": group["evidence_page_coverage"] / tasks if tasks else 0.0,
                "decoy_hit_rate": group["decoy_hit"] / tasks if tasks else 0.0,
                "mean_eval_seconds": group["eval_seconds"] / tasks if tasks else 0.0,
                "mean_kept_fraction": group["mean_kept_fraction"] / tasks if tasks else 0.0,
                "mean_kept_tokens": group["mean_kept_tokens"] / tasks if tasks else 0.0,
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
    layouts = [part.strip() for part in args.suite_layouts.split(",") if part.strip()]
    layout_specs = {layout: parse_layout(layout) for layout in layouts}
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    rng = random.Random(args.seed)
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    install_qwen3_attention_patch()
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
    started = time.perf_counter()
    for context_tokens in context_lengths:
        for layout_index, layout in enumerate(layouts):
            layout_spec = layout_specs[layout]
            evidence_percent = layout_spec[0] if layout_spec else -1
            decoy_percent = layout_spec[1] if layout_spec else -1
            for task_index in range(args.tasks_per_length):
                task_id = context_tokens * 100000 + layout_index * 1000 + task_index
                target_label_override = LABELS[task_index % len(LABELS)] if args.balanced_labels else None
                decoy_label_override = LABELS[(task_index + 1) % len(LABELS)] if args.balanced_labels else None
                if args.task_variant == "chain":
                    task = build_chain_task_with_layout(
                        tokenizer,
                        rng,
                        task_id,
                        context_tokens,
                        args.eval_tokens,
                        layout,
                        target_label_override=target_label_override,
                        decoy_label_override=decoy_label_override,
                    )
                elif args.task_variant in {"chain_para", "chain_para_conflict"}:
                    task = build_paraphrased_chain_task_with_layout(
                        tokenizer,
                        rng,
                        task_id,
                        context_tokens,
                        args.eval_tokens,
                        layout,
                        target_label_override=target_label_override,
                        decoy_label_override=decoy_label_override,
                        include_conflict=args.task_variant == "chain_para_conflict",
                    )
                elif args.task_variant == "chain_story_conflict":
                    task = build_story_chain_task_with_layout(
                        tokenizer,
                        rng,
                        task_id,
                        context_tokens,
                        args.eval_tokens,
                        layout,
                        target_label_override=target_label_override,
                        decoy_label_override=decoy_label_override,
                    )
                else:
                    task = build_task_with_layout(
                        tokenizer,
                        rng,
                        task_id,
                        context_tokens,
                        args.eval_tokens,
                        layout,
                        target_label_override=target_label_override,
                        decoy_label_override=decoy_label_override,
                    )
                pages, _, page_index, sections, section_index, section_to_pages = build_indexes(task, args)
                evidence_pages = span_pages(pages, task.evidence_spans, task.evidence_span)
                decoy_pages = span_pages(pages, task.decoy_spans, task.decoy_span)
                evidence_page = overlap_page(pages, task.evidence_span)
                decoy_page = overlap_page(pages, task.decoy_span)
                print(
                    f"context={context_tokens} layout={layout} task={task_index + 1}/{args.tasks_per_length} "
                    f"variant={task.task_variant} target={task.target_label} decoy={task.decoy_label} "
                    f"evidence_pages={evidence_pages} decoy_pages={decoy_pages}",
                    flush=True,
                )
                context_ids = torch.tensor(task.token_ids[: task.prefill_tokens], dtype=torch.long).view(1, -1)
                with sparse_context(None):
                    context_cache, context_prev, _, _ = run_tokens(
                        model,
                        context_ids,
                        input_device,
                        args.chunk_size,
                        past_key_values=None,
                        score_tokens=False,
                        prev_logits=None,
                    )
                query_ids = tokenizer(task.query_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                base_answer_score_format = (
                    "answer_label" if args.answer_score_format == "gated_sentence" else args.answer_score_format
                )
                prior_scores: dict[str, float] = {label: 0.0 for label in LABELS}
                sentence_prior_scores: dict[str, float] = {label: 0.0 for label in LABELS}
                if args.score_calibrated:
                    with sparse_context(None):
                        prior_cache, prior_prev, _, _ = run_tokens(
                            model,
                            query_ids,
                            input_device,
                            args.chunk_size,
                            past_key_values=None,
                            score_tokens=False,
                            prev_logits=None,
                        )
                        prior_scores = {
                            label: score_option(
                                model,
                                tokenizer,
                                input_device,
                                clone_past(prior_cache),
                                prior_prev.detach().clone(),
                                option_text(label, base_answer_score_format),
                            )
                            for label in LABELS
                        }
                        if args.answer_score_format == "gated_sentence":
                            sentence_prior_scores = {
                                label: score_option(
                                    model,
                                    tokenizer,
                                    input_device,
                                    clone_past(prior_cache),
                                    prior_prev.detach().clone(),
                                    option_text(label, "sentence"),
                                )
                                for label in LABELS
                            }
                    del prior_cache, prior_prev
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()
                for mode in modes:
                    ctx_recent_tokens = mode_recent_tokens(
                        mode,
                        args.recent_tokens,
                        task.prefill_tokens,
                        args.sink_tokens,
                    )
                    selected_pages = selected_pages_for_mode(
                        mode,
                        task,
                        pages,
                        page_index,
                        sections,
                        section_index,
                        section_to_pages,
                        args.sink_tokens,
                        ctx_recent_tokens,
                        args.query_window_tokens,
                    )
                    keep_remote_tokens = pages_to_tokens(pages, selected_pages)
                    keep_remote_ranges = pages_to_ranges(pages, selected_pages)
                    evidence_page_coverage = selected_page_coverage(evidence_pages, selected_pages, mode)
                    evidence_hit = evidence_page_coverage >= 1.0
                    decoy_hit = (
                        any(page_id in selected_pages for page_id in decoy_pages)
                        if mode not in {"full", "sink_recent"}
                        else mode == "full"
                    )
                    verifier_pred = text_verifier_label(task, pages, selected_pages, mode)
                    typed_record_source_mode = (
                        args.typed_summary_source_mode
                        if args.typed_record_format
                        in {
                            "summary",
                            "mini_summary",
                            "short_summary",
                            "lite_summary",
                            "natural_summary",
                            "answerline_summary",
                        }
                        and args.typed_summary_source_mode
                        else mode
                    )
                    typed_record_pages = selected_pages
                    if typed_record_source_mode != mode:
                        typed_record_pages = selected_pages_for_mode(
                            typed_record_source_mode,
                            task,
                            pages,
                            page_index,
                            sections,
                            section_index,
                            section_to_pages,
                            args.sink_tokens,
                            ctx_recent_tokens,
                            args.query_window_tokens,
                        )
                    typed_record_text, typed_record_meta = build_typed_record(
                        task,
                        pages,
                        typed_record_pages,
                        typed_record_source_mode,
                        args.typed_record_mode,
                        args.typed_record_format,
                    )
                    answer_override_available = (
                        args.typed_record_answer_override
                        and bool(typed_record_meta.get("typed_record_answer_label", ""))
                    )
                    typed_record_ids = (
                        tokenizer(typed_record_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                        if typed_record_text and args.typed_record_insert
                        else torch.empty((1, 0), dtype=torch.long)
                    )
                    stats = SparseStats()
                    ctx = SparseContext(
                        mode=mode,
                        context_tokens=task.prefill_tokens,
                        keep_remote_tokens=keep_remote_tokens,
                        keep_remote_ranges=keep_remote_ranges,
                        sink_tokens=args.sink_tokens,
                        recent_tokens=ctx_recent_tokens,
                        sparse_impl=args.sparse_attention_impl,
                        stats=stats,
                    )
                    eval_started = time.perf_counter()
                    with sparse_context(ctx):
                        eval_cache = clone_past(context_cache)
                        eval_prev = context_prev.detach().clone()
                        if typed_record_ids.shape[-1] > 0:
                            eval_cache, eval_prev, _, _ = run_tokens(
                                model,
                                typed_record_ids,
                                input_device,
                                chunk_size=1,
                                past_key_values=eval_cache,
                                score_tokens=False,
                                prev_logits=eval_prev,
                            )
                        query_cache, query_prev, query_nll, query_token_count = run_tokens(
                            model,
                            query_ids,
                            input_device,
                            chunk_size=1,
                            past_key_values=eval_cache,
                            score_tokens=args.score_query_ppl,
                            prev_logits=eval_prev,
                        )
                        lm_answer_scored = not (
                            answer_override_available and args.skip_lm_answer_when_override
                        )
                        if lm_answer_scored:
                            scores = {
                                label: score_option(
                                    model,
                                    tokenizer,
                                    input_device,
                                    clone_past(query_cache),
                                    query_prev.detach().clone(),
                                    option_text(label, base_answer_score_format),
                                )
                                for label in LABELS
                            }
                            base_scores = dict(scores)
                            base_calibrated_scores = {label: scores[label] - prior_scores[label] for label in LABELS}
                            base_calibrated_margin = top_margin(base_calibrated_scores)
                            sentence_gate_triggered = (
                                args.answer_score_format == "gated_sentence"
                                and mode not in {"full", "sink_recent"}
                                and bool(verifier_pred)
                                and base_calibrated_margin < args.gated_sentence_margin
                            )
                            if sentence_gate_triggered:
                                scores = {
                                    label: score_option(
                                        model,
                                        tokenizer,
                                        input_device,
                                        clone_past(query_cache),
                                        query_prev.detach().clone(),
                                        option_text(label, "sentence"),
                                    )
                                    for label in LABELS
                                }
                        else:
                            scores = {label: 0.0 for label in LABELS}
                            base_scores = dict(scores)
                            base_calibrated_scores = dict(scores)
                            base_calibrated_margin = 0.0
                            sentence_gate_triggered = False
                    eval_seconds = time.perf_counter() - eval_started
                    model_pred = max(scores, key=scores.get)
                    effective_prior_scores = sentence_prior_scores if sentence_gate_triggered else prior_scores
                    calibrated_scores = (
                        {label: scores[label] - effective_prior_scores[label] for label in LABELS}
                        if lm_answer_scored
                        else dict(scores)
                    )
                    model_calibrated_pred = max(calibrated_scores, key=calibrated_scores.get)
                    answer_override_applied = answer_override_available
                    pred = str(typed_record_meta["typed_record_answer_label"]) if answer_override_applied else model_pred
                    calibrated_pred = (
                        str(typed_record_meta["typed_record_answer_label"])
                        if answer_override_applied
                        else model_calibrated_pred
                    )
                    stat_row = stats.row()
                    rows.append(
                        {
                            "context_tokens": context_tokens,
                            "layout": layout,
                            "evidence_percent": evidence_percent,
                            "decoy_percent": decoy_percent,
                            "balanced_labels": int(args.balanced_labels),
                            "task_variant": task.task_variant,
                            "bridge_key": task.bridge_key,
                            "answer_score_format": args.answer_score_format,
                            "typed_record_mode": args.typed_record_mode,
                            "typed_record_format": args.typed_record_format,
                            "typed_record_answer_override": int(args.typed_record_answer_override),
                            "typed_record_insert": int(args.typed_record_insert),
                            "skip_lm_answer_when_override": int(args.skip_lm_answer_when_override),
                            "answer_override_applied": int(answer_override_applied),
                            "lm_answer_scored": int(lm_answer_scored),
                            "typed_summary_source_mode": typed_record_source_mode,
                            "typed_summary_source_pages": len(typed_record_pages),
                            "typed_summary_source_page_ids": " ".join(
                                str(page_id) for page_id in sorted(typed_record_pages)
                            ),
                            "typed_record_text": typed_record_text.replace("\n", "\\n"),
                            "typed_record_tokens": int(typed_record_ids.shape[-1]),
                            **typed_record_meta,
                            "task_id": task.task_id,
                            "mode": mode,
                            "sparse_attention_impl": args.sparse_attention_impl,
                            "target_label": task.target_label,
                            "decoy_label": task.decoy_label,
                            "model_pred_label": model_pred,
                            "model_calibrated_pred_label": model_calibrated_pred,
                            "pred_label": pred,
                            "calibrated_pred_label": calibrated_pred,
                            "text_verifier_pred_label": verifier_pred,
                            "sentence_gate_triggered": int(sentence_gate_triggered),
                            "base_calibrated_margin": base_calibrated_margin,
                            "base_calibrated_pred_label": max(base_calibrated_scores, key=base_calibrated_scores.get),
                            "correct": int(pred == task.target_label),
                            "calibrated_correct": int(calibrated_pred == task.target_label),
                            "text_verifier_present": int(bool(verifier_pred)),
                            "text_verifier_correct": int(verifier_pred == task.target_label),
                            "decoy_pred": int(pred == task.decoy_label),
                            "calibrated_decoy_pred": int(calibrated_pred == task.decoy_label),
                            "text_verifier_decoy_pred": int(verifier_pred == task.decoy_label),
                            "query_nll": query_nll,
                            "query_tokens": query_token_count,
                            "query_ppl": math.exp(query_nll / query_token_count) if query_token_count else 0.0,
                            "selected_pages": len(selected_pages),
                            "selected_remote_tokens": len(keep_remote_tokens),
                            "effective_recent_tokens": ctx_recent_tokens,
                            "selected_page_ids": " ".join(str(page_id) for page_id in sorted(selected_pages)),
                            "selected_token_ranges": " ".join(
                                f"{start}:{end}" for start, end in keep_remote_ranges
                            ),
                            "evidence_page_ids": " ".join(str(page_id) for page_id in evidence_pages),
                            "decoy_page_ids": " ".join(str(page_id) for page_id in decoy_pages),
                            "evidence_hit": int(evidence_hit),
                            "evidence_page_coverage": evidence_page_coverage,
                            "decoy_hit": int(decoy_hit),
                            "eval_seconds": eval_seconds,
                            "margin_true_minus_decoy": scores[task.target_label] - scores[task.decoy_label],
                            "calibrated_margin_true_minus_decoy": (
                                calibrated_scores[task.target_label] - calibrated_scores[task.decoy_label]
                            ),
                            **stat_row,
                            **{f"score_{label}": scores[label] for label in LABELS},
                            **{f"prior_score_{label}": effective_prior_scores[label] for label in LABELS},
                            **{f"calibrated_score_{label}": calibrated_scores[label] for label in LABELS},
                        }
                    )
                    del query_cache, query_prev
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()
                del context_cache, context_prev
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

    row_fields = [
        "context_tokens",
        "layout",
        "evidence_percent",
        "decoy_percent",
        "balanced_labels",
        "task_variant",
        "bridge_key",
        "answer_score_format",
        "typed_record_mode",
        "typed_record_format",
        "typed_record_answer_override",
        "typed_record_insert",
        "skip_lm_answer_when_override",
        "answer_override_applied",
        "lm_answer_scored",
        "typed_summary_source_mode",
        "typed_summary_source_pages",
        "typed_summary_source_page_ids",
        "typed_record_present",
        "typed_record_answer_label",
        "typed_record_correct",
        "typed_record_decoy_pred",
        "typed_record_bridge_artifact",
        "typed_record_tokens",
        "typed_record_text",
        "task_id",
        "mode",
        "sparse_attention_impl",
        "target_label",
        "decoy_label",
        "model_pred_label",
        "model_calibrated_pred_label",
        "pred_label",
        "calibrated_pred_label",
        "text_verifier_pred_label",
        "sentence_gate_triggered",
        "base_calibrated_margin",
        "base_calibrated_pred_label",
        "correct",
        "calibrated_correct",
        "text_verifier_present",
        "text_verifier_correct",
        "decoy_pred",
        "calibrated_decoy_pred",
        "text_verifier_decoy_pred",
        "query_nll",
        "query_tokens",
        "query_ppl",
        "selected_pages",
        "selected_remote_tokens",
        "effective_recent_tokens",
        "selected_page_ids",
        "selected_token_ranges",
        "evidence_page_ids",
        "decoy_page_ids",
        "evidence_hit",
        "evidence_page_coverage",
        "decoy_hit",
        "eval_seconds",
        "margin_true_minus_decoy",
        "calibrated_margin_true_minus_decoy",
        "sparse_cases",
        "mean_history_tokens",
        "mean_kept_tokens",
        "mean_kept_fraction",
        "max_kept_tokens",
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
    write_csv(output_dir / "sparse_rows.csv", rows, row_fields)
    summary_rows = summarize(rows)
    write_csv(
        output_dir / "sparse_summary.csv",
        summary_rows,
        [
            "context_tokens",
            "task_variant",
            "typed_record_mode",
            "typed_record_format",
            "typed_summary_source_mode",
            "typed_record_answer_override",
            "typed_record_insert",
            "skip_lm_answer_when_override",
            "mode",
            "sparse_attention_impl",
            "tasks",
            "accuracy",
            "calibrated_accuracy",
            "text_verifier_coverage",
            "text_verifier_accuracy",
            "typed_record_coverage",
            "typed_record_accuracy",
            "typed_record_decoy_pred_rate",
            "mean_typed_record_tokens",
            "answer_override_rate",
            "lm_answer_scored_rate",
            "sentence_gate_rate",
            "mean_base_calibrated_margin",
            "decoy_pred_rate",
            "calibrated_decoy_pred_rate",
            "text_verifier_decoy_pred_rate",
            "query_ppl",
            "mean_selected_pages",
            "mean_selected_remote_tokens",
            "evidence_hit_rate",
            "evidence_page_coverage",
            "decoy_hit_rate",
            "mean_eval_seconds",
            "mean_kept_fraction",
            "mean_kept_tokens",
            "mean_margin_true_minus_decoy",
            "mean_calibrated_margin_true_minus_decoy",
        ],
    )
    summary = {
        "args": vars(args),
        "resolved": {
            "context_lengths": context_lengths,
            "layouts": layouts,
            "balanced_labels": bool(args.balanced_labels),
            "task_variant": args.task_variant,
            "typed_record_mode": args.typed_record_mode,
            "typed_record_format": args.typed_record_format,
            "typed_summary_source_mode": args.typed_summary_source_mode,
            "typed_record_answer_override": bool(args.typed_record_answer_override),
            "typed_record_insert": bool(args.typed_record_insert),
            "skip_lm_answer_when_override": bool(args.skip_lm_answer_when_override),
            "answer_score_format": args.answer_score_format,
            "sparse_attention_impl": args.sparse_attention_impl,
            "tasks": len({(row["context_tokens"], row["task_id"]) for row in rows}),
            "modes": modes,
            "seconds": time.perf_counter() - started,
            "note": (
                "mask uses post-QK masking; gather/sdpa_gather compute attention on selected token ids; "
                "range_sdpa builds selected ids from sink/remote/recent ranges before SDPA; "
                "text_verifier is a synthetic extraction proxy over selected authoritative pages."
            ),
        },
        "paths": {
            "summary": str(output_dir / "sparse_summary.csv"),
            "rows": str(output_dir / "sparse_rows.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": summary["resolved"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
