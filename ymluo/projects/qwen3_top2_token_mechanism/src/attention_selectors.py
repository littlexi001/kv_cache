from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectorSpec:
    """A fixed-budget historical-token selector.

    ``sink_tokens`` is an allocation inside the historical-token budget, not
    an extra budget. The current query token can be kept separately.
    """

    name: str
    sink_tokens: int = 0


SUPPORTED_SELECTORS = {
    "top_attention",
    "bottom_attention",
    "recent",
    "sink",
    "random",
    "top_attention_drop_sink",
    "top_attention_drop_recent",
    "top_attention_drop_remote",
}


def parse_selector(value: str) -> SelectorSpec:
    normalized = value.strip().lower()
    match = re.fullmatch(r"sink_recent_s(\d+)", normalized)
    if match:
        return SelectorSpec("sink_recent", int(match.group(1)))
    if normalized not in SUPPORTED_SELECTORS:
        raise ValueError(f"Unsupported selector: {value}")
    return SelectorSpec(normalized)


def historical_budget(history_tokens: int, ratio: float) -> int:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if history_tokens <= 0:
        return 0
    return min(history_tokens, max(1, math.ceil(ratio * history_tokens)))


def _sink_recent_positions(history_tokens: int, budget: int, sink_tokens: int) -> list[int]:
    if history_tokens <= 0 or budget <= 0:
        return []
    sink_count = min(max(0, sink_tokens), budget, history_tokens)
    selected = list(range(sink_count))
    selected_set = set(selected)
    for position in range(history_tokens - 1, -1, -1):
        if len(selected) >= budget:
            break
        if position not in selected_set:
            selected.append(position)
            selected_set.add(position)
    return sorted(selected)


def _hashed_random_scores(
    history_tokens: int,
    head_count: int,
    query_position: int,
    layer_idx: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(history_tokens, device=device, dtype=torch.int64).view(1, -1)
    heads = torch.arange(head_count, device=device, dtype=torch.int64).view(-1, 1)
    # A stateless control: identical inputs always produce identical samples.
    values = (
        (positions + 1) * 1_103_515_245
        + (heads + 1) * 12_345
        + (query_position + 1) * 2_654_435_761
        + (layer_idx + 1) * 97_531
        + int(seed)
    ) % 2_147_483_647
    return values


def build_keep_mask(
    scores: torch.Tensor,
    selector: SelectorSpec,
    ratio: float,
    *,
    always_keep_self: bool = True,
    role_sink_tokens: int = 4,
    role_recent_tokens: int = 256,
    random_seed: int = 20260714,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build causal masks for ``[batch, heads, queries, keys]`` scores.

    The experiment uses unpadded single-sequence inputs. For a chunked causal
    forward, the current key position is ``key_count - query_count + q``.
    ``history_keep`` never includes the current token; ``keep`` may include it.
    """

    if scores.ndim != 4:
        raise ValueError(f"Expected rank-4 scores, got shape={tuple(scores.shape)}")
    batch_count, head_count, query_count, key_count = scores.shape
    if key_count < query_count:
        raise ValueError("key_count must be >= query_count for causal attention")

    keep = torch.zeros_like(scores, dtype=torch.bool)
    history_keep = torch.zeros_like(scores, dtype=torch.bool)
    past_tokens = key_count - query_count

    for batch_idx in range(batch_count):
        for query_idx in range(query_count):
            current = past_tokens + query_idx
            budget = historical_budget(current, ratio)
            if always_keep_self:
                keep[batch_idx, :, query_idx, current] = True
            if budget == 0:
                continue

            row = scores[batch_idx, :, query_idx, :current]
            chosen = torch.zeros((head_count, current), dtype=torch.bool, device=scores.device)

            if selector.name in {
                "top_attention",
                "top_attention_drop_sink",
                "top_attention_drop_recent",
                "top_attention_drop_remote",
            }:
                indices = torch.topk(row, k=budget, dim=-1, largest=True).indices
                chosen.scatter_(1, indices, True)
                positions = torch.arange(current, device=scores.device)
                is_sink = positions < min(role_sink_tokens, current)
                is_recent = positions >= max(0, current - role_recent_tokens)
                if selector.name == "top_attention_drop_sink":
                    chosen &= ~is_sink.view(1, -1)
                elif selector.name == "top_attention_drop_recent":
                    chosen &= ~is_recent.view(1, -1)
                elif selector.name == "top_attention_drop_remote":
                    chosen &= (is_sink | is_recent).view(1, -1)
            elif selector.name == "bottom_attention":
                indices = torch.topk(row, k=budget, dim=-1, largest=False).indices
                chosen.scatter_(1, indices, True)
            elif selector.name == "recent":
                chosen[:, current - budget : current] = True
            elif selector.name == "sink":
                chosen[:, :budget] = True
            elif selector.name == "sink_recent":
                positions = _sink_recent_positions(current, budget, selector.sink_tokens)
                chosen[:, positions] = True
            elif selector.name == "random":
                random_scores = _hashed_random_scores(
                    current,
                    head_count,
                    current,
                    layer_idx,
                    random_seed,
                    scores.device,
                )
                indices = torch.topk(random_scores, k=budget, dim=-1, largest=True).indices
                chosen.scatter_(1, indices, True)
            else:  # pragma: no cover - parse_selector prevents this path.
                raise ValueError(f"Unsupported selector: {selector.name}")

            history_keep[batch_idx, :, query_idx, :current] = chosen
            keep[batch_idx, :, query_idx, :current] = chosen

    return keep, history_keep


def actual_history_fraction(history_keep: torch.Tensor) -> tuple[int, int]:
    """Return selected and eligible layer-head-query history events."""

    batch_count, head_count, query_count, key_count = history_keep.shape
    past_tokens = key_count - query_count
    eligible = 0
    for query_idx in range(query_count):
        eligible += batch_count * head_count * (past_tokens + query_idx)
    return int(history_keep.sum().item()), eligible
