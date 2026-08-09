from __future__ import annotations

"""Safety audit for RoPE suppression certificates on frozen Qwen3-8B.

The prefix is always evaluated by the unmodified pretrained model.  Only the
final query token is instrumented.  A baseline pass records several
counterfactual suppression certificates and freezes one matched intervention
plan per semantic class.  Four subsequent passes move exactly the same number
of class-specific keys to their frozen local-envelope phase and measure the
causal change in the gold answer margin and perplexity.

This is deliberately independent from the phase-coherent and blockwise probe
runners.  It patches Qwen3Attention only while a local controller is active and
does not alter any existing experiment module.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

import run_local_rule_failure_boundary as base


CLASS_ORDER = (
    "gold_evidence",
    "conflict_evidence",
    "lexical_format_distractor",
    "filler",
)

CERTIFICATE_FIELDS = (
    "pre_suppression",
    "anchor_suppression",
    "grid_envelope_suppression",
    "phase_upper_suppression",
)

RAW_SCORE_FIELDS = (
    "post_score",
    "pre_score",
    "anchor_score",
    "grid_envelope_score",
    "phase_upper_score",
)

NUMBER_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

NUMBER_DIGITS = tuple(str(index) for index in range(1, 10))
ANSWER_DIGIT_BY_WORD = dict(zip(NUMBER_WORDS, NUMBER_DIGITS))

TRUTH_STATUS_MARKERS = ("verified", "unverified")

_ACTIVE_CONTROLLER: "CertificateController | None" = None


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{float(value):.{digits}g}")


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def expand_pair_values(values: torch.Tensor, head_dim: int) -> torch.Tensor:
    if int(values.shape[-1]) * 2 != int(head_dim):
        raise ValueError(
            f"pair count {values.shape[-1]} is incompatible with head_dim={head_dim}"
        )
    return torch.cat((values, values), dim=-1)


def attention_scaling(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> float:
    cos, sin = position_embeddings
    value = torch.sqrt(
        cos.float().reshape(-1)[0].square()
        + sin.float().reshape(-1)[0].square()
    )
    return float(value.item())


def _phase_values(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = positions.float().unsqueeze(-1) * inv_freq.float().view(
        *((1,) * positions.dim()), -1
    )
    cos = expand_pair_values(torch.cos(phase), head_dim).to(dtype=dtype)
    sin = expand_pair_values(torch.sin(phase), head_dim).to(dtype=dtype)
    return cos, sin


def invert_selected_rope(
    post_keys: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rope_scale: float,
) -> torch.Tensor:
    """Invert split-half RoPE for shared or per-head selected positions."""

    cos, sin = _phase_values(
        positions.to(device=post_keys.device),
        inv_freq.to(device=post_keys.device),
        int(post_keys.shape[-1]),
        post_keys.dtype,
    )
    while cos.dim() < post_keys.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    scaled = post_keys / float(rope_scale)
    return scaled * cos - rotate_half(scaled) * sin


def move_post_keys(
    post_keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Move already-RoPE'd keys to new virtual positions exactly."""

    delta = new_positions.to(post_keys.device) - old_positions.to(post_keys.device)
    cos, sin = _phase_values(
        delta,
        inv_freq.to(device=post_keys.device),
        int(post_keys.shape[-1]),
        post_keys.dtype,
    )
    while cos.dim() < post_keys.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return post_keys * cos + rotate_half(post_keys) * sin


def relative_score_from_pre(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    distances: torch.Tensor,
    inv_freq: torch.Tensor,
    rope_scale: float,
    score_scale: float,
) -> torch.Tensor:
    """Reconstruct standard RoPE QK from pre-RoPE vectors and distances."""

    phase = distances.to(key_pre.device).float().unsqueeze(-1) * inv_freq.float().to(
        key_pre.device
    ).view(*((1,) * distances.dim()), -1)
    cos = expand_pair_values(torch.cos(phase), int(key_pre.shape[-1])).to(key_pre)
    # Keys precede the query, hence the relative rotation is -distance.
    sin = expand_pair_values(-torch.sin(phase), int(key_pre.shape[-1])).to(key_pre)
    while cos.dim() < key_pre.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    relative_key = key_pre * cos + rotate_half(key_pre) * sin
    scores = (query_pre * relative_key).sum(dim=-1) * float(score_scale)
    return scores * float(rope_scale) ** 2


def phase_upper_scores(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    rope_scale: float,
    score_scale: float,
) -> torch.Tensor:
    """Independent-frequency phase envelope; an upper bound, not one position."""

    half = int(query_pre.shape[-1] // 2)
    qx, qy = query_pre[..., :half], query_pre[..., half:]
    kx, ky = key_pre[..., :half], key_pre[..., half:]
    a = qx.float() * kx.float() + qy.float() * ky.float()
    b = qx.float() * ky.float() - qy.float() * kx.float()
    envelope = torch.sqrt(a.square() + b.square() + 1e-12).sum(dim=-1)
    return envelope * float(score_scale) * float(rope_scale) ** 2


def gather_shared_positions(
    values: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    index = positions.view(1, 1, -1, 1).expand(
        values.shape[0], values.shape[1], positions.numel(), values.shape[-1]
    )
    return values.gather(2, index)


def gather_per_head_positions(
    values: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if positions.dim() != 2 or int(positions.shape[0]) != int(values.shape[1]):
        raise ValueError("positions must be [heads, selected]")
    index = positions.view(1, positions.shape[0], positions.shape[1], 1).expand(
        values.shape[0], positions.shape[0], positions.shape[1], values.shape[-1]
    )
    return values.gather(2, index)


def gqa_group_count(query: torch.Tensor, key_or_value: torch.Tensor) -> int:
    """Validate grouped-query head geometry without expanding KV tensors."""

    if query.dim() != 4 or key_or_value.dim() != 4:
        raise ValueError("GQA tensors must be [batch, heads, tokens, head_dim]")
    if query.shape[0] != key_or_value.shape[0]:
        raise ValueError("query and KV batch sizes differ")
    if query.shape[-1] != key_or_value.shape[-1]:
        raise ValueError("query and KV head dimensions differ")
    query_heads = int(query.shape[1])
    kv_heads = int(key_or_value.shape[1])
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError(
            f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
        )
    return query_heads // kv_heads


def gqa_query_key_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    score_scale: float = 1.0,
) -> torch.Tensor:
    """Exact QK scores using grouped heads without materialising repeated K."""

    groups = gqa_group_count(query, key)
    batch, query_heads, query_tokens, head_dim = query.shape
    kv_heads = int(key.shape[1])
    grouped_query = query.reshape(
        batch, kv_heads, groups, query_tokens, head_dim
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key)
    return scores.reshape(batch, query_heads, query_tokens, key.shape[-2]) * float(
        score_scale
    )


def gqa_attention_output(
    weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Exact AV product using grouped heads without materialising repeated V."""

    if weights.dim() != 4 or value.dim() != 4:
        raise ValueError(
            "attention weights and value cache must be "
            "[batch, heads, query/key tokens, key tokens/head_dim]"
        )
    batch, query_heads, query_tokens, key_count = weights.shape
    if int(value.shape[0]) != int(batch):
        raise ValueError("attention weights and value cache have different batches")
    if int(value.shape[-2]) != int(key_count):
        raise ValueError("attention weights and value cache have different key counts")
    kv_heads = int(value.shape[1])
    if kv_heads <= 0 or int(query_heads) % kv_heads:
        raise ValueError(
            f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
        )
    groups = int(query_heads) // kv_heads
    grouped_weights = weights.reshape(
        batch, kv_heads, groups, query_tokens, key_count
    )
    output = torch.einsum("bhgqk,bhkd->bhgqd", grouped_weights, value)
    return output.reshape(batch, query_heads, query_tokens, value.shape[-1])


def expand_selected_gqa_heads(values: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand only a small gathered subset, never the full KV cache."""

    if groups <= 0:
        raise ValueError("groups must be positive")
    batch, kv_heads, selected, head_dim = values.shape
    return (
        values.unsqueeze(2)
        .expand(batch, kv_heads, groups, selected, head_dim)
        .reshape(batch, kv_heads * groups, selected, head_dim)
    )


def gather_shared_gqa_positions(
    values: torch.Tensor,
    positions: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    selected = gather_shared_positions(values, positions)
    return expand_selected_gqa_heads(selected, groups)


def gather_per_query_head_gqa_positions(
    values: torch.Tensor,
    positions: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Gather per-query-head positions through a broadcast KV-head view."""

    batch, kv_heads, key_count, head_dim = values.shape
    query_heads = kv_heads * int(groups)
    if positions.dim() != 2 or int(positions.shape[0]) != query_heads:
        raise ValueError("positions must be [query_heads, selected]")
    if positions.numel() and (
        int(positions.min()) < 0 or int(positions.max()) >= key_count
    ):
        raise ValueError("positions contain an out-of-range key index")
    selected = int(positions.shape[1])
    grouped_values = values.unsqueeze(2).expand(
        batch, kv_heads, groups, key_count, head_dim
    )
    index = positions.reshape(kv_heads, groups, selected).view(
        1, kv_heads, groups, selected, 1
    ).expand(batch, kv_heads, groups, selected, head_dim)
    gathered = grouped_values.gather(3, index)
    return gathered.reshape(batch, query_heads, selected, head_dim)


def select_evenly(
    positions: Sequence[int], count: int, required: Sequence[int] = ()
) -> list[int]:
    """Deterministically balance a class sample while retaining decisive tokens."""

    pool = sorted({int(item) for item in positions})
    required_unique = [item for item in dict.fromkeys(map(int, required)) if item in pool]
    if count <= 0 or not pool:
        return []
    if len(pool) <= count:
        return pool
    output = required_unique[:count]
    remaining = [item for item in pool if item not in output]
    slots = count - len(output)
    if slots <= 0:
        return sorted(output)
    if slots == 1:
        chosen = [remaining[len(remaining) // 2]]
    else:
        chosen = [
            remaining[round(index * (len(remaining) - 1) / (slots - 1))]
            for index in range(slots)
        ]
    return sorted(dict.fromkeys((*output, *chosen)))


def assemble_case_from_encoded_records(
    records: Sequence[dict[str, Any]],
    suffix_ids: Sequence[int],
    filler_id: int,
    *,
    total_tokens: int,
    seed: int,
    packet_gap_tokens: int,
    class_sample_count: int,
) -> dict[str, Any]:
    """Assemble an exact-length prompt from already-tokenized, auditable records."""

    ordered = [dict(record) for record in records]
    random.Random(2026080107 + int(seed) * 1009).shuffle(ordered)
    prompt_ids: list[int] = []
    class_positions: dict[str, list[int]] = {name: [] for name in CLASS_ORDER}
    decisive_positions: dict[str, list[int]] = {name: [] for name in CLASS_ORDER}
    packet_filler_positions: list[int] = []
    public_records: list[dict[str, Any]] = []

    for record_index, record in enumerate(ordered):
        category = str(record["category"])
        if category not in class_positions or category == "filler":
            raise ValueError(f"invalid record category: {category}")
        start = len(prompt_ids)
        ids = [int(item) for item in record["ids"]]
        prompt_ids.extend(ids)
        end = len(prompt_ids)
        class_positions[category].extend(range(start, end))
        local_decisive = [int(item) for item in record.get("decisive_local", [])]
        decisive_positions[category].extend(start + item for item in local_decisive)
        public_records.append(
            {
                "category": category,
                "text": record.get("text", ""),
                "span": [start, end],
                "decisive_positions": [start + item for item in local_decisive],
            }
        )
        if record_index + 1 < len(ordered) and packet_gap_tokens > 0:
            gap_start = len(prompt_ids)
            prompt_ids.extend([int(filler_id)] * int(packet_gap_tokens))
            gap_end = len(prompt_ids)
            positions = list(range(gap_start, gap_end))
            class_positions["filler"].extend(positions)
            packet_filler_positions.extend(positions)

    suffix_ids = [int(item) for item in suffix_ids]
    tail_count = int(total_tokens) - len(prompt_ids) - len(suffix_ids)
    if tail_count < 0:
        raise ValueError(
            f"fixed records and query need {len(prompt_ids) + len(suffix_ids)} "
            f"tokens, exceeding total_tokens={total_tokens}"
        )
    tail_start = len(prompt_ids)
    prompt_ids.extend([int(filler_id)] * tail_count)
    class_positions["filler"].extend(range(tail_start, tail_start + tail_count))
    query_start = len(prompt_ids)
    prompt_ids.extend(suffix_ids)
    if len(prompt_ids) != int(total_tokens):
        raise AssertionError(f"constructed {len(prompt_ids)}, expected {total_tokens}")

    sample_positions: dict[str, list[int]] = {}
    for category in CLASS_ORDER:
        pool = class_positions[category]
        required = decisive_positions[category]
        if category == "filler" and packet_filler_positions:
            pool = packet_filler_positions
        sample_positions[category] = select_evenly(
            pool, int(class_sample_count), required=required
        )

    return {
        "prompt_ids": prompt_ids,
        "total_tokens": int(total_tokens),
        "seed": int(seed),
        "query_span": [query_start, len(prompt_ids)],
        "class_positions": class_positions,
        "decisive_positions": decisive_positions,
        "sample_positions": sample_positions,
        "records": public_records,
        "filler_count": len(class_positions["filler"]),
        "packet_filler_count": len(packet_filler_positions),
    }


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer(text, add_special_tokens=False)["input_ids"]]


def single_token_id(tokenizer: Any, text: str, label: str) -> int:
    ids = token_ids(tokenizer, text)
    if len(ids) != 1:
        raise RuntimeError(f"{label} must be one token: {text!r} -> {ids}")
    return ids[0]


def encoded_record(
    tokenizer: Any, category: str, text: str, decisive_word: str
) -> dict[str, Any]:
    ids = token_ids(tokenizer, text)
    decisive_id = single_token_id(
        tokenizer, f" {decisive_word}", f"decisive word {decisive_word}"
    )
    locations = [index for index, token_id in enumerate(ids) if token_id == decisive_id]
    if len(locations) != 1:
        raise RuntimeError(
            f"expected one {decisive_word!r} token in {text!r}; got {locations}"
        )
    return {
        "category": category,
        "text": text,
        "ids": ids,
        "decisive_local": locations,
    }


def assert_no_truth_status_leakage(text: str) -> None:
    lowered = text.lower()
    leaked = [marker for marker in TRUTH_STATUS_MARKERS if marker in lowered]
    if leaked:
        raise ValueError(f"truth-status marker leaked into model input: {leaked}")


def build_case(
    tokenizer: Any,
    *,
    total_tokens: int,
    seed: int,
    packet_gap_tokens: int,
    class_sample_count: int,
) -> dict[str, Any]:
    conflict_words = ("four", "six", "eight", "two")
    conflict = conflict_words[int(seed) % len(conflict_words)]
    lexical_facts = (
        ("Alice", "seven"),
        ("Bob", "three"),
        ("Carol", "five"),
        ("David", "one"),
    )
    records = [
        encoded_record(
            tokenizer,
            "gold_evidence",
            "The school register lists Xiaoming's age as nine years.\n",
            "nine",
        ),
        encoded_record(
            tokenizer,
            "conflict_evidence",
            f"A family note lists Xiaoming's age as {conflict} years.\n",
            conflict,
        ),
    ]
    records.extend(
        encoded_record(
            tokenizer,
            "lexical_format_distractor",
            f"The school register lists {name}'s age as {age_word} "
            f"{'year' if age_word == 'one' else 'years'}.\n",
            age_word,
        )
        for name, age_word in lexical_facts
    )
    query_text = (
        "\nQuestion: According to the school register, what is Xiaoming's age? "
        "Reply with exactly one digit and nothing else. Answer: "
    )
    visible_text = "".join(record["text"] for record in records) + query_text
    assert_no_truth_status_leakage(visible_text)
    filler_id = single_token_id(tokenizer, ".", "filler period")
    case = assemble_case_from_encoded_records(
        records,
        token_ids(tokenizer, query_text),
        filler_id,
        total_tokens=total_tokens,
        seed=seed,
        packet_gap_tokens=packet_gap_tokens,
        class_sample_count=class_sample_count,
    )
    case.update(
        {
            "query_text": query_text,
            "gold_answer": "nine",
            "conflict_answer": conflict,
            "gold_output": ANSWER_DIGIT_BY_WORD["nine"],
            "conflict_output": ANSWER_DIGIT_BY_WORD[conflict],
            "filler_token_id": filler_id,
            "filler_token_text": tokenizer.decode(
                [filler_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
    )
    return case


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"prompt_ids", "class_positions"}
    }


def _score_at_moved_positions(
    query_post: torch.Tensor,
    selected_key_post: torch.Tensor,
    old_positions: torch.Tensor,
    anchor_distances: torch.Tensor,
    query_position: int,
    inv_freq: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    new_positions = torch.full_like(old_positions, int(query_position)) - anchor_distances
    moved = move_post_keys(
        selected_key_post, old_positions, new_positions, inv_freq
    )
    return (
        query_post[0, :, 0, :].unsqueeze(1).float() * moved[0].float()
    ).sum(dim=-1) * float(score_scale)


def certificate_bundle(
    query_pre: torch.Tensor,
    query_post: torch.Tensor,
    key_post: torch.Tensor,
    native_scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    query_position: int,
    inv_freq: torch.Tensor,
    rope_scale: float,
    score_scale: float,
    anchor_distances: Sequence[int],
    fixed_anchor_distance: int,
) -> dict[str, torch.Tensor]:
    """Compute raw references and their native-post suppression gaps."""

    positions = positions.to(device=key_post.device, dtype=torch.long)
    # Only the sampled keys are expanded from KV heads to query heads.  At
    # Qwen3-8B's 64K context this is O(H_q * samples * d), rather than the
    # O(H_q * context * d) allocation caused by repeat_kv on the full cache.
    selected_post = gather_shared_gqa_positions(key_post, positions, groups)
    key_pre = invert_selected_rope(selected_post, positions, inv_freq, rope_scale)
    pre_score = (
        query_pre.float() * key_pre.float()
    ).sum(dim=-1)[0] * float(score_scale) * float(rope_scale) ** 2
    post_score = native_scores[0, :, 0, :].index_select(1, positions).float()

    anchor_values = torch.tensor(
        list(anchor_distances),
        dtype=torch.long,
        device=key_post.device,
    ).clamp_max(int(query_position))
    anchor_scores = []
    for distance in anchor_values.tolist():
        distances = torch.full(
            (query_post.shape[1], positions.numel()),
            int(distance),
            dtype=torch.long,
            device=key_post.device,
        )
        old = positions.view(1, -1).expand(query_post.shape[1], -1)
        anchor_scores.append(
            _score_at_moved_positions(
                query_post,
                selected_post,
                old,
                distances,
                query_position,
                inv_freq,
                score_scale,
            )
        )
    anchor_stack = torch.stack(anchor_scores, dim=-1)
    grid_envelope_score, best_index = anchor_stack.max(dim=-1)
    best_anchor_distance = anchor_values[best_index]
    fixed_index = min(
        range(len(anchor_distances)),
        key=lambda index: abs(int(anchor_distances[index]) - int(fixed_anchor_distance)),
    )
    anchor_score = anchor_stack[..., fixed_index]
    phase_upper_score = phase_upper_scores(
        query_pre.expand(-1, -1, positions.numel(), -1),
        key_pre,
        rope_scale,
        score_scale,
    )[0]

    actual_distances = int(query_position) - positions
    reconstructed = relative_score_from_pre(
        query_pre.expand(-1, -1, positions.numel(), -1),
        key_pre,
        actual_distances,
        inv_freq,
        rope_scale,
        score_scale,
    )[0]
    return {
        "post_score": post_score,
        "pre_score": pre_score,
        "anchor_score": anchor_score,
        "grid_envelope_score": grid_envelope_score,
        "phase_upper_score": phase_upper_score,
        "pre_suppression": pre_score - post_score,
        "anchor_suppression": anchor_score - post_score,
        "grid_envelope_suppression": grid_envelope_score - post_score,
        "phase_upper_suppression": phase_upper_score - post_score,
        "best_anchor_distance": best_anchor_distance,
        "reconstruction_error": reconstructed - post_score,
    }


def select_matched_plan(
    positions: torch.Tensor,
    certificate: torch.Tensor,
    best_anchor_distance: torch.Tensor,
    matched_tokens: int,
) -> dict[str, torch.Tensor]:
    """Select exactly k positions per head from a balanced class sample."""

    if certificate.dim() != 2:
        raise ValueError("certificate must be [heads, candidates]")
    keep = min(max(1, int(matched_tokens)), int(certificate.shape[1]))
    top = torch.topk(certificate.float(), k=keep, dim=-1, largest=True).indices
    shared = positions.view(1, -1).expand(certificate.shape[0], -1)
    return {
        "positions": shared.gather(1, top).detach().cpu(),
        "anchor_distances": best_anchor_distance.gather(1, top).detach().cpu(),
        "baseline_certificates": certificate.gather(1, top).detach().cpu(),
    }


def apply_frozen_plan_to_scores(
    native_scores: torch.Tensor,
    query_post: torch.Tensor,
    key_post: torch.Tensor,
    plan: dict[str, torch.Tensor],
    *,
    groups: int,
    query_position: int,
    inv_freq: torch.Tensor,
    score_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply a fixed-count, fixed-position, fixed-anchor phase intervention."""

    positions = plan["positions"].to(device=key_post.device, dtype=torch.long)
    distances = plan["anchor_distances"].to(device=key_post.device, dtype=torch.long)
    selected_keys = gather_per_query_head_gqa_positions(key_post, positions, groups)
    repaired = _score_at_moved_positions(
        query_post,
        selected_keys,
        positions,
        distances,
        query_position,
        inv_freq,
        score_scale,
    )
    native_selected = native_scores[0, :, 0, :].gather(1, positions).float()
    modified = native_scores.clone()
    modified[0, :, 0, :].scatter_(1, positions, repaired.to(modified.dtype))
    delta = repaired.float() - native_selected
    return modified, {
        "applied_count": int(delta.numel()),
        "score_delta_sum": float(delta.sum().item()),
        "positive_score_delta_count": int((delta > 0).sum().item()),
        "score_delta_abs_sum": float(delta.abs().sum().item()),
    }


@dataclass
class CertificateController:
    mode: str
    case: dict[str, Any]
    anchor_distances: tuple[int, ...]
    fixed_anchor_distance: int
    matched_tokens: int
    trigger_threshold: float
    target_class: str | None = None
    plan: dict[str, dict[int, dict[str, torch.Tensor]]] = field(
        default_factory=lambda: {name: {} for name in CLASS_ORDER}
    )
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    reconstruction_error_max: float = 0.0
    applied_count: int = 0
    score_delta_sum: float = 0.0
    score_delta_abs_sum: float = 0.0
    positive_score_delta_count: int = 0

    def plan_layer(
        self,
        layer_index: int,
        query_pre: torch.Tensor,
        query_post: torch.Tensor,
        key_post: torch.Tensor,
        native_scores: torch.Tensor,
        native_weights: torch.Tensor,
        groups: int,
        query_position: int,
        inv_freq: torch.Tensor,
        rope_scale: float,
        score_scale: float,
    ) -> None:
        for category in CLASS_ORDER:
            raw_positions = self.case["sample_positions"][category]
            if not raw_positions:
                raise RuntimeError(f"class {category} has no sampled positions")
            positions = torch.tensor(
                raw_positions, dtype=torch.long, device=key_post.device
            )
            bundle = certificate_bundle(
                query_pre,
                query_post,
                key_post,
                native_scores,
                positions,
                groups=groups,
                query_position=query_position,
                inv_freq=inv_freq,
                rope_scale=rope_scale,
                score_scale=score_scale,
                anchor_distances=self.anchor_distances,
                fixed_anchor_distance=self.fixed_anchor_distance,
            )
            self.reconstruction_error_max = max(
                self.reconstruction_error_max,
                float(bundle["reconstruction_error"].abs().max().item()),
            )
            self.plan[category][int(layer_index)] = select_matched_plan(
                positions,
                bundle["grid_envelope_suppression"],
                bundle["best_anchor_distance"],
                self.matched_tokens,
            )
            decisive = set(map(int, self.case["decisive_positions"][category]))
            class_attention = native_weights[0, :, 0, :].index_select(1, positions)
            for head in range(int(query_post.shape[1])):
                for local_index, position in enumerate(raw_positions):
                    row: dict[str, Any] = {
                        "layer": int(layer_index),
                        "head": int(head),
                        "class": category,
                        "sample_index": int(local_index),
                        "token_position": int(position),
                        "relative_distance": int(query_position - int(position)),
                        "is_decisive_token": int(int(position) in decisive),
                        "native_attention_probability": rounded(
                            float(class_attention[head, local_index].item())
                        ),
                        "best_anchor_distance": int(
                            bundle["best_anchor_distance"][head, local_index].item()
                        ),
                        "reconstruction_error": rounded(
                            float(bundle["reconstruction_error"][head, local_index].item())
                        ),
                    }
                    for field_name in (*RAW_SCORE_FIELDS, *CERTIFICATE_FIELDS):
                        row[field_name] = rounded(
                            float(bundle[field_name][head, local_index].item())
                        )
                    for field_name in CERTIFICATE_FIELDS:
                        row[f"{field_name}_trigger"] = int(
                            float(bundle[field_name][head, local_index].item())
                            > self.trigger_threshold
                        )
                    self.sample_rows.append(row)

    def intervention_layer(
        self,
        layer_index: int,
        native_scores: torch.Tensor,
        query_post: torch.Tensor,
        key_post: torch.Tensor,
        groups: int,
        query_position: int,
        inv_freq: torch.Tensor,
        score_scale: float,
    ) -> torch.Tensor:
        if self.target_class is None:
            raise RuntimeError("intervention controller is missing target_class")
        entry = self.plan[self.target_class][int(layer_index)]
        modified, summary = apply_frozen_plan_to_scores(
            native_scores,
            query_post,
            key_post,
            entry,
            groups=groups,
            query_position=query_position,
            inv_freq=inv_freq,
            score_scale=score_scale,
        )
        self.applied_count += int(summary["applied_count"])
        self.score_delta_sum += float(summary["score_delta_sum"])
        self.score_delta_abs_sum += float(summary["score_delta_abs_sum"])
        self.positive_score_delta_count += int(summary["positive_score_delta_count"])
        return modified

    def intervention_summary(self) -> dict[str, Any]:
        baseline_certificates = []
        if self.target_class is not None:
            for entry in self.plan[self.target_class].values():
                baseline_certificates.extend(
                    float(item)
                    for item in entry["baseline_certificates"].reshape(-1).tolist()
                )
        return {
            "applied_count": int(self.applied_count),
            "mean_score_delta": rounded(
                self.score_delta_sum / max(1, self.applied_count)
            ),
            "mean_abs_score_delta": rounded(
                self.score_delta_abs_sum / max(1, self.applied_count)
            ),
            "positive_score_delta_fraction": rounded(
                self.positive_score_delta_count / max(1, self.applied_count)
            ),
            "selected_baseline_trigger_fraction": rounded(
                sum(value > self.trigger_threshold for value in baseline_certificates)
                / max(1, len(baseline_certificates))
            ),
        }


@contextmanager
def activate(controller: CertificateController | None):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


def add_attention_mask(
    scores: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is None:
        return scores
    return scores + attention_mask[
        :, :, -scores.shape[-2] :, : scores.shape[-1]
    ]


def read_only_final_query_kv(
    past_key_value: Any | None,
    layer_index: int,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one layer's current KV without mutating or retaining it in cache.

    ``DynamicCache.update`` stores each layer's concatenated prefix.  When an
    immutable legacy prefix is also retained for the following intervention
    passes, a one-token query can therefore keep almost a second full cache
    alive by the final layer.  This probe never consumes the returned cache, so
    a layer-local concatenation is both semantically exact and memory bounded.
    """

    if past_key_value is None:
        return current_key, current_value
    prefix_key: torch.Tensor | None = None
    prefix_value: torch.Tensor | None = None
    if hasattr(past_key_value, "key_cache") and hasattr(
        past_key_value, "value_cache"
    ):
        if int(layer_index) < len(past_key_value.key_cache):
            prefix_key = past_key_value.key_cache[int(layer_index)]
            prefix_value = past_key_value.value_cache[int(layer_index)]
    elif isinstance(past_key_value, (tuple, list)):
        if int(layer_index) < len(past_key_value):
            prefix_key, prefix_value = past_key_value[int(layer_index)]
    else:
        raise TypeError("unsupported cache type for read-only final-query probe")
    if prefix_key is None or prefix_value is None or prefix_key.shape[-2] == 0:
        return current_key, current_value
    if prefix_key.shape[:-2] != current_key.shape[:-2] or (
        prefix_key.shape[-1] != current_key.shape[-1]
    ):
        raise ValueError("prefix and current key tensors have incompatible shapes")
    if prefix_value.shape[:-2] != current_value.shape[:-2] or (
        prefix_value.shape[-1] != current_value.shape[-1]
    ):
        raise ValueError("prefix and current value tensors have incompatible shapes")
    return (
        torch.cat((prefix_key, current_key), dim=-2),
        torch.cat((prefix_value, current_value), dim=-2),
    )


def certificate_attention_forward(
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
        return self._suppression_certificate_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._suppression_certificate_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    cos = cos.to(query_pre.device)
    sin = sin.to(query_pre.device)
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre, current_key_pre, cos, sin
    )
    key_post, value = read_only_final_query_kv(
        past_key_value,
        int(self.layer_idx),
        current_key_post,
        current_value,
    )

    groups = gqa_group_count(query_post, key_post)
    if gqa_group_count(query_post, value) != groups:
        raise ValueError("key and value caches use different GQA head geometry")
    key_count = int(key_post.shape[-2])
    query_position = key_count - 1
    score_scale = float(
        getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1]))
    )
    native_scores = gqa_query_key_scores(query_post, key_post, score_scale)
    native_scores = add_attention_mask(native_scores, attention_mask)
    native_weights = F.softmax(native_scores.float(), dim=-1).to(query_post.dtype)
    rotary = self._suppression_certificate_rotary_ref()
    if rotary is None:
        raise RuntimeError("model rotary embedding was released unexpectedly")
    inv_freq = rotary.inv_freq.detach().float().to(query_post.device)

    scores = native_scores
    if controller.mode == "plan":
        controller.plan_layer(
            int(self.layer_idx),
            query_pre,
            query_post,
            key_post,
            native_scores,
            native_weights,
            groups,
            query_position,
            inv_freq,
            attention_scaling((cos, sin)),
            score_scale,
        )
    elif controller.mode == "intervene":
        scores = controller.intervention_layer(
            int(self.layer_idx),
            native_scores,
            query_post,
            key_post,
            groups,
            query_position,
            inv_freq,
            score_scale,
        )
    else:
        raise ValueError(f"unknown controller mode: {controller.mode}")

    weights = F.softmax(scores.float(), dim=-1).to(query_post.dtype)
    attention_output = gqa_attention_output(weights, value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


def patch_model(model: Any) -> None:
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        if not hasattr(module, "_suppression_certificate_original_forward"):
            module._suppression_certificate_original_forward = module.forward
            module._suppression_certificate_modeling_qwen3 = modeling_qwen3
            # A weak reference avoids registering the shared rotary module as a
            # child of every attention layer through nn.Module.__setattr__.
            module._suppression_certificate_rotary_ref = weakref.ref(
                model.model.rotary_emb
            )
            module.forward = types.MethodType(certificate_attention_forward, module)
        found += 1
    if found == 0:
        raise RuntimeError("no Qwen3Attention modules found")


def reset_dynamic_cache(cache: Any, prefix_length: int) -> None:
    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise TypeError("this probe requires a DynamicCache")
    for index in range(len(cache.key_cache)):
        cache.key_cache[index] = cache.key_cache[index][:, :, :prefix_length, :]
        cache.value_cache[index] = cache.value_cache[index][:, :, :prefix_length, :]
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = int(prefix_length)


def resolve_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def load_model(args: argparse.Namespace, maximum_length: int) -> tuple[Any, Any]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    factor = base.rope_factor_for_length(
        maximum_length, args.original_max_position_embeddings
    )
    if factor > 1.0:
        config.max_position_embeddings = int(maximum_length)
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(
                args.original_max_position_embeddings
            ),
        }
    dtype = resolve_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def answer_token_ids(tokenizer: Any) -> dict[str, int]:
    values = {
        # The prompt already ends with a literal space after ``Answer:``.  The
        # next target is therefore the no-leading-space digit token.  Scoring
        # the string ``" " + digit`` here would instead be off by one token
        # because Qwen emits the standalone space token at this boundary.
        word: single_token_id(tokenizer, digit, f"answer {word} ({digit})")
        for word, digit in ANSWER_DIGIT_BY_WORD.items()
    }
    if len(set(values.values())) != len(values):
        raise RuntimeError(f"answer token ids are not unique: {values}")
    return values


def answer_metrics(
    tokenizer: Any,
    logits: torch.Tensor,
    answer_ids: dict[str, int],
    conflict_answer: str,
) -> dict[str, Any]:
    final_logits = logits[0, -1].float()
    log_probs = F.log_softmax(final_logits, dim=-1)
    gold_id = int(answer_ids["nine"])
    conflict_id = int(answer_ids[conflict_answer])
    gold_nll = -float(log_probs[gold_id].item())
    masked = final_logits.clone()
    masked[gold_id] = -torch.inf
    strongest_id = int(masked.argmax().item())
    prediction_id = int(final_logits.argmax().item())
    return {
        "gold_nll": rounded(gold_nll),
        "gold_ppl": rounded(math.exp(min(gold_nll, 700.0))),
        "gold_probability": rounded(math.exp(-gold_nll)),
        "gold_full_vocab_margin": rounded(
            float(final_logits[gold_id].item() - final_logits[strongest_id].item())
        ),
        "gold_conflict_margin": rounded(
            float(final_logits[gold_id].item() - final_logits[conflict_id].item())
        ),
        "prediction_token_id": prediction_id,
        "prediction_text": tokenizer.decode(
            [prediction_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).replace("\n", "\\n"),
        "next_token_correct": int(prediction_id == gold_id),
        "strongest_non_gold_token_id": strongest_id,
        "strongest_non_gold_text": tokenizer.decode(
            [strongest_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).replace("\n", "\\n"),
    }


def delta_metrics(
    answer: dict[str, Any], baseline_answer: dict[str, Any]
) -> dict[str, Any]:
    return {
        "delta_gold_nll": rounded(
            float(answer["gold_nll"]) - float(baseline_answer["gold_nll"])
        ),
        "gold_ppl_ratio": rounded(
            float(answer["gold_ppl"]) / max(float(baseline_answer["gold_ppl"]), 1e-30)
        ),
        "delta_gold_ppl": rounded(
            float(answer["gold_ppl"]) - float(baseline_answer["gold_ppl"])
        ),
        "delta_gold_full_vocab_margin": rounded(
            float(answer["gold_full_vocab_margin"])
            - float(baseline_answer["gold_full_vocab_margin"])
        ),
        "delta_gold_conflict_margin": rounded(
            float(answer["gold_conflict_margin"])
            - float(baseline_answer["gold_conflict_margin"])
        ),
    }


def native_baseline_enabled(
    target_context_tokens: int,
    native_max_context_tokens: int,
) -> bool:
    """Return whether the optional untouched-native calibration is in scope.

    A non-positive limit preserves the historical unlimited behaviour.  The
    production launcher sets 32K explicitly because Transformers' eager GQA
    implementation materialises repeated full-context K/V tensors.
    """

    limit = int(native_max_context_tokens)
    return limit <= 0 or int(target_context_tokens) <= limit


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def binary_auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Tie-aware Mann-Whitney AUROC without a sklearn dependency."""

    if not positive_scores or not negative_scores:
        return float("nan")
    labelled = [(float(value), 1) for value in positive_scores]
    labelled.extend((float(value), 0) for value in negative_scores)
    labelled.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(labelled):
        end = index + 1
        while end < len(labelled) and labelled[end][0] == labelled[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in labelled[index:end]
        )
        index = end
    positive_count = len(positive_scores)
    negative_count = len(negative_scores)
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return statistic / (positive_count * negative_count)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def stable_bootstrap_seed(base_seed: int, *labels: Any) -> int:
    """Derive a process-independent RNG seed for one reported statistic."""

    payload = "\x1f".join((str(int(base_seed)), *(str(label) for label in labels)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def seed_stratified_bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic: Any,
    *,
    replicates: int,
    random_seed: int,
    minimum_seeds_per_stratum: int,
) -> dict[str, Any]:
    """Cluster-bootstrap complete seeds, paired across context lengths.

    Every sampled seed contributes all of its correlated head/layer/token rows.
    Repeated seed draws intentionally duplicate that full cluster.  For an
    all-length report, the same seed draw carries every context length, which
    preserves the paired length trajectory.  An incomplete seed-by-length grid
    is reported as NA instead of silently changing the estimand.
    """

    strata: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        length = int(row["target_context_tokens"])
        seed = int(row["seed"])
        strata.setdefault(length, {}).setdefault(seed, []).append(row)
    counts = {length: len(seed_rows) for length, seed_rows in sorted(strata.items())}
    common: dict[str, Any] = {
        "bootstrap_unit": "seed",
        "bootstrap_stratification": "paired_target_context_tokens",
        "bootstrap_seed_counts": ";".join(
            f"{length}:{count}" for length, count in counts.items()
        ),
        "bootstrap_replicates": int(replicates),
    }
    if not rows or not strata:
        return {
            **common,
            "ci95_low": "NA",
            "ci95_high": "NA",
            "bootstrap_valid_replicates": 0,
            "bootstrap_status": "NA:no_rows",
        }
    insufficient = {
        length: count
        for length, count in counts.items()
        if count < int(minimum_seeds_per_stratum)
    }
    if insufficient:
        detail = ",".join(
            f"{length}:{count}" for length, count in sorted(insufficient.items())
        )
        return {
            **common,
            "ci95_low": "NA",
            "ci95_high": "NA",
            "bootstrap_valid_replicates": 0,
            "bootstrap_status": (
                f"NA:insufficient_seeds[{detail}]"
                f"<minimum_{int(minimum_seeds_per_stratum)}"
            ),
        }
    seed_sets = [set(seed_rows) for seed_rows in strata.values()]
    reference_seeds = seed_sets[0]
    if any(seed_set != reference_seeds for seed_set in seed_sets[1:]):
        return {
            **common,
            "ci95_low": "NA",
            "ci95_high": "NA",
            "bootstrap_valid_replicates": 0,
            "bootstrap_status": "NA:unbalanced_seed_length_grid",
        }
    if int(replicates) < 1:
        return {
            **common,
            "ci95_low": "NA",
            "ci95_high": "NA",
            "bootstrap_valid_replicates": 0,
            "bootstrap_status": "NA:bootstrap_disabled",
        }

    rng = random.Random(int(random_seed))
    estimates: list[float] = []
    seeds = sorted(reference_seeds)
    rows_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for seed_rows in strata.values():
        for seed in seeds:
            rows_by_seed[seed].extend(seed_rows[seed])
    for _ in range(int(replicates)):
        replicate_rows: list[dict[str, Any]] = []
        for _draw in range(len(seeds)):
            sampled_seed = seeds[rng.randrange(len(seeds))]
            replicate_rows.extend(rows_by_seed[sampled_seed])
        try:
            estimate = float(statistic(replicate_rows))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(estimate):
            estimates.append(estimate)
    required_valid = max(1, int(math.ceil(int(replicates) * 0.90)))
    if len(estimates) < required_valid:
        return {
            **common,
            "ci95_low": "NA",
            "ci95_high": "NA",
            "bootstrap_valid_replicates": len(estimates),
            "bootstrap_status": (
                f"NA:only_{len(estimates)}_of_{int(replicates)}_valid_replicates"
            ),
        }
    return {
        **common,
        "ci95_low": rounded(quantile(estimates, 0.025)),
        "ci95_high": rounded(quantile(estimates, 0.975)),
        "bootstrap_valid_replicates": len(estimates),
        "bootstrap_status": "ok",
    }


def certificate_distribution_summary(
    samples: Sequence[dict[str, Any]], trigger_threshold: float
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lengths: list[int | str] = sorted(
        {int(row["target_context_tokens"]) for row in samples}
    )
    lengths.append("all")
    for length in lengths:
        subset = [
            row
            for row in samples
            if length == "all" or int(row["target_context_tokens"]) == length
        ]
        for category in CLASS_ORDER:
            class_rows = [row for row in subset if row["class"] == category]
            for metric in (*RAW_SCORE_FIELDS, *CERTIFICATE_FIELDS):
                values = [float(row[metric]) for row in class_rows]
                if not values:
                    continue
                output.append(
                    {
                        "target_context_tokens": length,
                        "class": category,
                        "metric": metric,
                        "sample_count": len(values),
                        "mean": rounded(statistics.fmean(values)),
                        "std": rounded(
                            statistics.pstdev(values) if len(values) > 1 else 0.0
                        ),
                        "p10": rounded(quantile(values, 0.10)),
                        "p50": rounded(quantile(values, 0.50)),
                        "p90": rounded(quantile(values, 0.90)),
                        "trigger_rate": (
                            rounded(
                                sum(value > trigger_threshold for value in values)
                                / len(values)
                            )
                            if metric in CERTIFICATE_FIELDS
                            else ""
                        ),
                    }
                )
    return output


def certificate_auroc_summary(
    samples: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_bootstrap_seeds: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lengths: list[int | str] = sorted(
        {int(row["target_context_tokens"]) for row in samples}
    )
    lengths.append("all")
    tasks = (
        ("gold_vs_conflict", ("gold_evidence",), ("conflict_evidence",)),
        (
            "gold_vs_lexical_format",
            ("gold_evidence",),
            ("lexical_format_distractor",),
        ),
        ("gold_vs_filler", ("gold_evidence",), ("filler",)),
        (
            "gold_vs_all_nongold",
            ("gold_evidence",),
            ("conflict_evidence", "lexical_format_distractor", "filler"),
        ),
        (
            "semantic_evidence_vs_nonsemantic",
            ("gold_evidence", "conflict_evidence"),
            ("lexical_format_distractor", "filler"),
        ),
    )
    metric_fields = tuple(
        metric
        for metric in (*RAW_SCORE_FIELDS, *CERTIFICATE_FIELDS)
        if any(metric in row for row in samples)
    )
    scopes = ("all_sampled", "decisive_only")
    for length in lengths:
        length_subset = [
            row
            for row in samples
            if length == "all" or int(row["target_context_tokens"]) == length
        ]
        for scope in scopes:
            subset = [
                row
                for row in length_subset
                if scope == "all_sampled"
                or int(row.get("is_decisive_token", 0)) == 1
            ]
            for metric in metric_fields:
                metric_subset = [row for row in subset if metric in row]
                for task, positive_classes, negative_classes in tasks:
                    positives = [
                        float(row[metric])
                        for row in metric_subset
                        if row["class"] in positive_classes
                    ]
                    negatives = [
                        float(row[metric])
                        for row in metric_subset
                        if row["class"] in negative_classes
                    ]
                    by_seed_length: dict[
                        tuple[int, int], list[dict[str, Any]]
                    ] = {}
                    for row in metric_subset:
                        by_seed_length.setdefault(
                            (
                                int(row["target_context_tokens"]),
                                int(row["seed"]),
                            ),
                            [],
                        ).append(row)
                    seed_auc_rows: list[dict[str, Any]] = []
                    for (seed_length, seed), seed_rows in sorted(
                        by_seed_length.items()
                    ):
                        seed_positives = [
                            float(row[metric])
                            for row in seed_rows
                            if row["class"] in positive_classes
                        ]
                        seed_negatives = [
                            float(row[metric])
                            for row in seed_rows
                            if row["class"] in negative_classes
                        ]
                        seed_auc = binary_auroc(seed_positives, seed_negatives)
                        if math.isfinite(seed_auc):
                            seed_auc_rows.append(
                                {
                                    "target_context_tokens": seed_length,
                                    "seed": seed,
                                    "seed_auroc": seed_auc,
                                }
                            )

                    def auc_statistic(
                        replicate_rows: Sequence[dict[str, Any]],
                    ) -> float:
                        values = [
                            float(row["seed_auroc"]) for row in replicate_rows
                        ]
                        if not values:
                            raise ValueError("no valid seed-level AUROC values")
                        return statistics.fmean(values)

                    bootstrap = seed_stratified_bootstrap(
                        seed_auc_rows,
                        auc_statistic,
                        replicates=bootstrap_replicates,
                        random_seed=stable_bootstrap_seed(
                            bootstrap_seed, "auroc", length, scope, metric, task
                        ),
                        minimum_seeds_per_stratum=minimum_bootstrap_seeds,
                    )
                    pooled = binary_auroc(positives, negatives)
                    output.append(
                        {
                            "target_context_tokens": length,
                            "scope": scope,
                            "metric": metric,
                            "task": task,
                            "positive_count": len(positives),
                            "negative_count": len(negatives),
                            "pooled_auroc": (
                                rounded(pooled) if math.isfinite(pooled) else "NA"
                            ),
                            "seed_auroc_count": len(seed_auc_rows),
                            "auroc": (
                                rounded(
                                    statistics.fmean(
                                        float(row["seed_auroc"])
                                        for row in seed_auc_rows
                                    )
                                )
                                if seed_auc_rows
                                else "NA"
                            ),
                            "seed_auroc_std": (
                                rounded(
                                    statistics.pstdev(
                                        float(row["seed_auroc"])
                                        for row in seed_auc_rows
                                    )
                                )
                                if len(seed_auc_rows) > 1
                                else (0.0 if seed_auc_rows else "NA")
                            ),
                            "auroc_ci95_low": bootstrap.pop("ci95_low"),
                            "auroc_ci95_high": bootstrap.pop("ci95_high"),
                            **bootstrap,
                        }
                    )
    return output


def intervention_summary(
    case_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_bootstrap_seeds: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lengths: list[int | str] = sorted(
        {int(row["target_context_tokens"]) for row in case_rows}
    )
    lengths.append("all")
    for length in lengths:
        for intervention in ("native_baseline", "none", *CLASS_ORDER):
            rows = [
                row
                for row in case_rows
                if row["intervention_class"] == intervention
                and (length == "all" or int(row["target_context_tokens"]) == length)
            ]
            if not rows:
                continue
            fields = (
                "gold_nll",
                "gold_ppl",
                "gold_probability",
                "gold_full_vocab_margin",
                "gold_conflict_margin",
                "next_token_correct",
                "delta_gold_nll",
                "gold_ppl_ratio",
                "delta_gold_ppl",
                "delta_gold_full_vocab_margin",
                "delta_gold_conflict_margin",
                "instrumentation_delta_gold_nll",
                "instrumentation_gold_ppl_ratio",
                "instrumentation_delta_gold_ppl",
                "instrumentation_delta_gold_full_vocab_margin",
                "instrumentation_delta_gold_conflict_margin",
            )
            observed_lengths = sorted(
                {int(row["target_context_tokens"]) for row in rows}
            )
            comparisons = sorted(
                {
                    str(row["comparison_baseline"])
                    for row in rows
                    if row.get("comparison_baseline")
                }
            )
            if len(comparisons) > 1:
                raise ValueError(
                    "mixed comparison baselines for "
                    f"{intervention} at {length}: {comparisons}"
                )
            summary: dict[str, Any] = {
                "target_context_tokens": length,
                "intervention_class": intervention,
                "case_count": len(rows),
                "comparison_baseline": comparisons[0] if comparisons else "unspecified",
                "observed_context_lengths": ",".join(map(str, observed_lengths)),
            }
            availability = [
                int(row["native_baseline_available"])
                for row in rows
                if "native_baseline_available" in row
            ]
            if availability:
                summary["native_baseline_measured_case_count"] = sum(availability)
                summary["native_baseline_skipped_case_count"] = (
                    len(availability) - sum(availability)
                )
            for field_name in fields:
                values = [float(row[field_name]) for row in rows if field_name in row]
                if values:
                    summary[f"mean_{field_name}"] = rounded(statistics.fmean(values))
                    summary[f"{field_name}_case_count"] = len(values)
            if "mean_gold_nll" in summary:
                # Primary PPL aggregation follows the other RoPE experiments:
                # average matched-case NLL first, then exponentiate.  The
                # arithmetic mean of per-case PPL remains available above only
                # as an outlier-sensitive descriptive statistic.
                summary["gold_ppl_exp_mean_nll"] = rounded(
                    math.exp(min(float(summary["mean_gold_nll"]), 700.0))
                )
            for field_name in ("delta_gold_nll", "delta_gold_conflict_margin"):
                def mean_statistic(
                    replicate_rows: Sequence[dict[str, Any]],
                    selected_field: str = field_name,
                ) -> float:
                    values = [
                        float(row[selected_field])
                        for row in replicate_rows
                        if selected_field in row
                    ]
                    if not values:
                        raise ValueError(f"no values for {selected_field}")
                    return statistics.fmean(values)

                bootstrap = seed_stratified_bootstrap(
                    rows,
                    mean_statistic,
                    replicates=bootstrap_replicates,
                    random_seed=stable_bootstrap_seed(
                        bootstrap_seed,
                        "intervention",
                        length,
                        intervention,
                        field_name,
                    ),
                    minimum_seeds_per_stratum=minimum_bootstrap_seeds,
                )
                prefix = f"mean_{field_name}"
                summary[f"{prefix}_ci95_low"] = bootstrap.pop("ci95_low")
                summary[f"{prefix}_ci95_high"] = bootstrap.pop("ci95_high")
                # The seed counts/status are identical for both requested
                # intervention metrics; retain field-specific validity too.
                summary[f"{field_name}_bootstrap_valid_replicates"] = bootstrap[
                    "bootstrap_valid_replicates"
                ]
                summary[f"{field_name}_bootstrap_status"] = bootstrap[
                    "bootstrap_status"
                ]
                summary.setdefault("bootstrap_unit", bootstrap["bootstrap_unit"])
                summary.setdefault(
                    "bootstrap_stratification",
                    bootstrap["bootstrap_stratification"],
                )
                summary.setdefault(
                    "bootstrap_seed_counts", bootstrap["bootstrap_seed_counts"]
                )
                summary.setdefault(
                    "bootstrap_replicates", bootstrap["bootstrap_replicates"]
                )
            output.append(summary)
    return output


def native_baseline_coverage(
    case_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe explicitly where untouched-native calibration was measured."""

    baselines = [
        row for row in case_rows if row.get("intervention_class") == "none"
    ]
    lengths: list[int | str] = sorted(
        {int(row["target_context_tokens"]) for row in baselines}
    )
    lengths.append("all")
    output: list[dict[str, Any]] = []
    for length in lengths:
        rows = [
            row
            for row in baselines
            if length == "all" or int(row["target_context_tokens"]) == length
        ]
        known = [row for row in rows if "native_baseline_available" in row]
        measured = sum(int(row["native_baseline_available"]) for row in known)
        statuses = sorted(
            {str(row.get("native_baseline_status", "unspecified")) for row in known}
        )
        output.append(
            {
                "target_context_tokens": length,
                "instrumented_baseline_case_count": len(rows),
                "native_status_known_case_count": len(known),
                "native_baseline_measured_case_count": measured,
                "native_baseline_skipped_case_count": len(known) - measured,
                "native_baseline_coverage_fraction": (
                    rounded(measured / len(known)) if known else "NA"
                ),
                "native_baseline_statuses": ",".join(statuses) if statuses else "NA",
            }
        )
    return output


def collect_raw(
    source_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for source in source_dirs:
        raw = source / "raw"
        for result_path in sorted(raw.glob("*_result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            case_rows.extend(result["case_rows"])
        for sample_path in sorted(raw.glob("*_certificate_samples.jsonl")):
            samples.extend(read_jsonl(sample_path))
    return case_rows, samples


def write_aggregate_outputs(
    output_dir: Path,
    source_dirs: Sequence[Path],
    trigger_threshold: float,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_bootstrap_seeds: int,
) -> None:
    case_rows, samples = collect_raw(source_dirs)
    distributions = certificate_distribution_summary(samples, trigger_threshold)
    aurocs = certificate_auroc_summary(
        samples,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_bootstrap_seeds=minimum_bootstrap_seeds,
    )
    interventions = intervention_summary(
        case_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_bootstrap_seeds=minimum_bootstrap_seeds,
    )
    native_coverage = native_baseline_coverage(case_rows)
    write_csv(output_dir / "case_rows.csv", case_rows)
    write_csv(output_dir / "certificate_samples.csv", samples)
    write_csv(output_dir / "certificate_distributions.csv", distributions)
    write_csv(output_dir / "certificate_aurocs.csv", aurocs)
    write_csv(output_dir / "intervention_summary.csv", interventions)
    write_csv(output_dir / "native_baseline_coverage.csv", native_coverage)
    write_json(
        output_dir / "summary.json",
        {
            "source_dirs": [str(path) for path in source_dirs],
            "case_row_count": len(case_rows),
            "certificate_sample_count": len(samples),
            "bootstrap": {
                "unit": "seed",
                "stratification": "paired_target_context_tokens",
                "replicates": int(bootstrap_replicates),
                "random_seed": int(bootstrap_seed),
                "minimum_seeds_per_stratum": int(minimum_bootstrap_seeds),
            },
            "certificate_distributions": distributions,
            "certificate_aurocs": aurocs,
            "intervention_summary": interventions,
            "native_baseline_coverage": native_coverage,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-final-query suppression-certificate safety probe"
    )
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="8192,32768,65536")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--class-sample-count", type=int, default=8)
    parser.add_argument("--matched-tokens", type=int, default=1)
    parser.add_argument("--packet-gap-tokens", type=int, default=16)
    parser.add_argument("--anchor-distances", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--fixed-anchor-distance", type=int, default=128)
    parser.add_argument("--trigger-threshold", type=float, default=0.0)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--native-max-context-tokens",
        type=int,
        default=0,
        help=(
            "Run the untouched native eager-query calibration only through this "
            "context length; non-positive means unlimited. The instrumented-none "
            "baseline is always run and remains the intervention reference."
        ),
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026080119)
    parser.add_argument("--minimum-bootstrap-seeds", type=int, default=4)
    parser.add_argument("--merge-shards", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _case_file_stem(length: int, seed: int) -> str:
    return f"length_{int(length)}_seed_{int(seed)}"


def run_case(
    model: Any,
    tokenizer: Any,
    answer_ids: dict[str, int],
    case: dict[str, Any],
    args: argparse.Namespace,
    anchor_distances: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = torch.tensor(case["prompt_ids"], dtype=torch.long).view(1, -1)
    prefix_length = int(prompt.shape[1]) - 1
    base.synchronize()
    started = time.perf_counter()
    legacy, prefill_seconds = base.prefill_sequence(
        model, prompt[:, :-1], args.prefill_chunk_size
    )

    # The untouched eager implementation is calibration-only.  It repeats all
    # KV heads to query heads and can exceed memory at 64K even for one query.
    # The grouped instrumented-none path below is the primary baseline at every
    # length, while native calibration remains available at bounded lengths.
    native_is_enabled = native_baseline_enabled(
        int(case["total_tokens"]), args.native_max_context_tokens
    )
    native_answer: dict[str, Any] | None = None
    native_seconds: float | None = None
    if native_is_enabled:
        base.synchronize()
        native_started = time.perf_counter()
        cache = base.cache_from_legacy(legacy)
        with torch.inference_mode():
            native_output = base.forward_with_cache(
                model,
                prompt[:, -1:].to(base.input_device(model)),
                cache,
                prefix_length,
            )
        base.synchronize()
        native_seconds = time.perf_counter() - native_started
        native_answer = answer_metrics(
            tokenizer, native_output.logits, answer_ids, case["conflict_answer"]
        )
        del native_output, cache

    plan_controller = CertificateController(
        mode="plan",
        case=case,
        anchor_distances=anchor_distances,
        fixed_anchor_distance=args.fixed_anchor_distance,
        matched_tokens=args.matched_tokens,
        trigger_threshold=args.trigger_threshold,
    )
    base.synchronize()
    baseline_started = time.perf_counter()
    cache = base.cache_from_legacy(legacy)
    with activate(plan_controller), torch.inference_mode():
        baseline_output = base.forward_with_cache(
            model,
            prompt[:, -1:].to(base.input_device(model)),
            cache,
            prefix_length,
        )
    base.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    instrumented_answer = answer_metrics(
        tokenizer, baseline_output.logits, answer_ids, case["conflict_answer"]
    )
    del baseline_output, cache
    instrumentation_delta = (
        delta_metrics(instrumented_answer, native_answer)
        if native_answer is not None
        else None
    )

    native_status = (
        "measured_untouched_native"
        if native_answer is not None
        else "skipped_context_exceeds_native_max"
    )

    common = {
        "target_context_tokens": int(case["total_tokens"]),
        "seed": int(case["seed"]),
        "gold_answer": "nine",
        "conflict_answer": case["conflict_answer"],
        "gold_output": case["gold_output"],
        "conflict_output": case["conflict_output"],
        "primary_baseline": "instrumented_none",
        "native_baseline_available": int(native_answer is not None),
        "native_baseline_status": native_status,
        "native_max_context_tokens": int(args.native_max_context_tokens),
    }
    zero_delta = {
        "delta_gold_nll": 0.0,
        "gold_ppl_ratio": 1.0,
        "delta_gold_ppl": 0.0,
        "delta_gold_full_vocab_margin": 0.0,
        "delta_gold_conflict_margin": 0.0,
    }
    case_rows: list[dict[str, Any]] = []
    if native_answer is not None:
        case_rows.append(
            {
            **common,
            "intervention_class": "native_baseline",
            "comparison_baseline": "native_baseline",
            **native_answer,
            **zero_delta,
            "prefill_seconds": rounded(prefill_seconds),
            "query_seconds": rounded(native_seconds or 0.0),
            }
        )
    instrumentation_columns = (
        prefixed_metrics(instrumentation_delta, "instrumentation_")
        if instrumentation_delta is not None
        else {}
    )
    case_rows.append(
        {
            **common,
            "intervention_class": "none",
            "comparison_baseline": "instrumented_none",
            **instrumented_answer,
            **zero_delta,
            **instrumentation_columns,
            "prefill_seconds": rounded(prefill_seconds),
            "query_seconds": rounded(baseline_seconds),
        }
    )
    interventions: dict[str, Any] = {}
    for category in CLASS_ORDER:
        controller = CertificateController(
            mode="intervene",
            case=case,
            anchor_distances=anchor_distances,
            fixed_anchor_distance=args.fixed_anchor_distance,
            matched_tokens=args.matched_tokens,
            trigger_threshold=args.trigger_threshold,
            target_class=category,
            plan=plan_controller.plan,
        )
        base.synchronize()
        query_started = time.perf_counter()
        cache = base.cache_from_legacy(legacy)
        with activate(controller), torch.inference_mode():
            output = base.forward_with_cache(
                model,
                prompt[:, -1:].to(base.input_device(model)),
                cache,
                prefix_length,
            )
        base.synchronize()
        query_seconds = time.perf_counter() - query_started
        answer = answer_metrics(
            tokenizer, output.logits, answer_ids, case["conflict_answer"]
        )
        intervention_metrics = controller.intervention_summary()
        delta = delta_metrics(answer, instrumented_answer)
        native_delta = (
            prefixed_metrics(delta_metrics(answer, native_answer), "native_")
            if native_answer is not None
            else None
        )
        interventions[category] = {
            "answer": answer,
            "delta_from_instrumented_baseline": delta,
            "delta_from_native_baseline": native_delta,
            "intervention": intervention_metrics,
            "query_seconds": rounded(query_seconds),
        }
        case_rows.append(
            {
                **common,
                "intervention_class": category,
                "comparison_baseline": "instrumented_none",
                **answer,
                **delta,
                **(native_delta or {}),
                **intervention_metrics,
                "prefill_seconds": rounded(prefill_seconds),
                "query_seconds": rounded(query_seconds),
            }
        )
        del output, cache

    sample_rows = [
        {**common, **row} for row in plan_controller.sample_rows
    ]
    result = {
        "schema_version": 2,
        "experiment": "suppression_certificate_safety_probe_qwen3_8b",
        "case": public_case(case),
        "primary_baseline": "instrumented_none",
        "baseline_answer": instrumented_answer,
        "native_baseline_answer": native_answer,
        "native_baseline_available": native_answer is not None,
        "native_baseline_status": native_status,
        "native_max_context_tokens": int(args.native_max_context_tokens),
        "instrumented_baseline_answer": instrumented_answer,
        "instrumentation_delta_from_native": instrumentation_delta,
        "interventions": interventions,
        "case_rows": case_rows,
        "certificate_reconstruction_error_max": rounded(
            plan_controller.reconstruction_error_max
        ),
        "timing": {
            "prefill_seconds": rounded(prefill_seconds),
            "baseline_query_seconds": rounded(baseline_seconds),
            "instrumented_baseline_query_seconds": rounded(baseline_seconds),
            "native_baseline_query_seconds": (
                rounded(native_seconds) if native_seconds is not None else None
            ),
            "total_seconds": rounded(time.perf_counter() - started),
        },
    }
    del legacy, prompt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, sample_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_replicates < 0 or args.minimum_bootstrap_seeds < 1:
        raise ValueError(
            "bootstrap replicates must be non-negative and minimum seeds positive"
        )
    if args.merge_shards:
        sources = [
            Path(item.strip())
            for item in args.merge_shards.split(",")
            if item.strip()
        ]
        if not sources:
            raise ValueError("--merge-shards supplied no source directories")
        write_aggregate_outputs(
            output_dir,
            sources,
            args.trigger_threshold,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            minimum_bootstrap_seeds=args.minimum_bootstrap_seeds,
        )
        write_json(
            output_dir / "merge_config.json",
            {**vars(args), "resolved_sources": [str(path) for path in sources]},
        )
        print(f"merged {len(sources)} shards into {output_dir}", flush=True)
        return

    lengths = sorted(set(parse_int_csv(args.lengths)))
    anchors = tuple(sorted(set(parse_int_csv(args.anchor_distances))))
    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required unless --merge-shards is used")
    if not lengths or min(lengths) <= 0:
        raise ValueError("lengths must be positive")
    if not anchors or min(anchors) < 0:
        raise ValueError("anchor distances must be non-negative")
    if args.fixed_anchor_distance not in anchors:
        raise ValueError("fixed anchor distance must occur in --anchor-distances")
    if args.class_sample_count < 1 or args.matched_tokens < 1:
        raise ValueError("sample and matched token counts must be positive")
    if args.matched_tokens > args.class_sample_count:
        raise ValueError("matched tokens cannot exceed class sample count")
    if args.prefill_chunk_size < 1:
        raise ValueError("prefill chunk size must be positive")

    config = {
        **vars(args),
        "resolved_lengths": lengths,
        "resolved_anchor_distances": anchors,
        "class_order": CLASS_ORDER,
        "certificate_fields": CERTIFICATE_FIELDS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "frozen_final_query": True,
        "explicit_truth_status_markers_in_prompt": False,
        "instrumented_attention_implementation": "grouped_gqa_einsum_v1",
        "primary_baseline": "instrumented_none",
    }
    write_json(output_dir / "config.json", config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    previews = [
        public_case(
            build_case(
                tokenizer,
                total_tokens=length,
                seed=args.seed_start,
                packet_gap_tokens=args.packet_gap_tokens,
                class_sample_count=args.class_sample_count,
            )
        )
        for length in lengths
    ]
    write_json(output_dir / "design.json", {"config": config, "cases": previews})
    if args.dry_run:
        print(json.dumps({"config": config, "cases": previews}, ensure_ascii=False, indent=2))
        return

    model, tokenizer = load_model(args, max(lengths))
    patch_model(model)
    ids = answer_token_ids(tokenizer)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for length in lengths:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            stem = _case_file_stem(length, seed)
            result_path = raw_dir / f"{stem}_result.json"
            sample_path = raw_dir / f"{stem}_certificate_samples.jsonl"
            if result_path.exists() and sample_path.exists():
                print(f"length={length} seed={seed} already complete", flush=True)
                continue
            case = build_case(
                tokenizer,
                total_tokens=length,
                seed=seed,
                packet_gap_tokens=args.packet_gap_tokens,
                class_sample_count=args.class_sample_count,
            )
            result, samples = run_case(model, tokenizer, ids, case, args, anchors)
            temporary_samples = sample_path.with_suffix(sample_path.suffix + ".tmp")
            if temporary_samples.exists():
                temporary_samples.unlink()
            append_jsonl(temporary_samples, samples)
            temporary_samples.replace(sample_path)
            write_json(result_path, result)
            message = (
                f"length={length} seed={seed} baseline_ppl="
                f"{result['instrumented_baseline_answer']['gold_ppl']:.4f} "
                f"margin={result['instrumented_baseline_answer']['gold_full_vocab_margin']:.4f} "
                f"reconstruction_error="
                f"{result['certificate_reconstruction_error_max']:.6g}"
            )
            if result["instrumentation_delta_from_native"] is None:
                message += " native_calibration=skipped"
            else:
                message += (
                    " native_calibration=measured instrumented_margin_delta="
                    f"{result['instrumentation_delta_from_native']['delta_gold_full_vocab_margin']:.4f}"
                )
            print(message, flush=True)

    write_aggregate_outputs(
        output_dir,
        [output_dir],
        args.trigger_threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        minimum_bootstrap_seeds=args.minimum_bootstrap_seeds,
    )
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
