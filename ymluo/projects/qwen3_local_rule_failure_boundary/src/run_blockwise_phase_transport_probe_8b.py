from __future__ import annotations

"""Blockwise coherent phase transport on frozen Qwen3-8B.

All variants use the same exact pre-RoPE block-max selector.  They differ only
in how a selected remote block is consumed by final-query attention:

* selector-only: native post-RoPE QK;
* clipped consumer: independently clip every selected remote distance;
* coherent transport: move a whole block by one shared distance translation;
* mass-preserving transport: coherent transport plus remote log-partition match;
* random matched-rate: transport the same number of blocks per head at random.

The prefix remains native Qwen3 RoPE and only the final query-token attention is
intervened on.  Values are always the original cached V tensors.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import run_phase_coherent_rope_probe_8b as phase
import run_rope_retrieval_repair_8b as rope_repair


runner = phase.runner

BLOCKWISE_VARIANTS = (
    "block16_selector_only",
    "block16_clipped_consumer",
    "block16_transport",
    "block16_transport_masspreserve",
    "block16_random_matched",
    "block32_selector_only",
    "block32_clipped_consumer",
    "block32_transport",
    "block32_transport_masspreserve",
    "block32_random_matched",
)
runner.VARIANTS = tuple(dict.fromkeys((*runner.VARIANTS, *BLOCKWISE_VARIANTS)))

_DELEGATE_FORWARD = phase.phase_kernel_attention_forward
_BASE_METRIC_SUMMARY = runner.MetricAccumulator.summary


def _block_metric_summary(self: runner.MetricAccumulator) -> dict[str, float]:
    summary = _BASE_METRIC_SUMMARY(self)
    block_count = int(getattr(self, "block_selected_count", 0))
    trigger_count = int(getattr(self, "block_trigger_count", 0))
    random_count = int(getattr(self, "block_random_trigger_count", 0))
    summary.update(
        {
            "selected_remote_blocks_per_head": float(
                getattr(self, "block_selected_per_head_sum", 0.0)
            )
            / max(1, int(getattr(self, "block_head_rows", 0))),
            "block_trigger_fraction": trigger_count / max(1, block_count),
            "random_matched_trigger_fraction": random_count / max(1, block_count),
            "triggered_anchor_suppression_mean": float(
                getattr(self, "block_trigger_gap_sum", 0.0)
            )
            / max(1, trigger_count),
            "triggered_translation_abs_mean": float(
                getattr(self, "block_trigger_tau_sum", 0.0)
            )
            / max(1, trigger_count),
            "selected_token_fraction_actual": float(
                getattr(self, "block_selected_token_sum", 0.0)
            )
            / max(1, int(getattr(self, "block_total_token_sum", 0))),
            "block_relative_distance_error_max": float(
                getattr(self, "block_relative_error_max", 0.0)
            ),
        }
    )
    return summary


runner.MetricAccumulator.summary = _block_metric_summary


def block_size_for_variant(variant: str) -> int:
    if variant.startswith("block16_"):
        return 16
    if variant.startswith("block32_"):
        return 32
    raise ValueError(f"variant does not encode a supported block size: {variant}")


@dataclass(frozen=True)
class BlockSelection:
    positions: torch.Tensor
    remote_mask: torch.Tensor
    selected_block_ids: torch.Tensor
    anchor_positions: torch.Tensor
    block_size: int
    requested_keep_count: int
    actual_keep_count: int
    local_start: int
    remote_start: int
    remote_end: int


def blockwise_premax_selection(
    pre_scores: torch.Tensor,
    *,
    keep_count: int,
    local_window: int,
    sink_tokens: int,
    block_size: int,
) -> BlockSelection:
    """Select fixed remote blocks by their exact pre-RoPE maximum.

    ``pre_scores`` is ``[heads, keys]``.  Current, recent-local, and sink tokens
    are reserved first.  The remaining token budget purchases only complete
    remote blocks, so the realized budget is never above ``keep_count`` and is
    less than it by fewer than ``block_size`` tokens.
    """

    if pre_scores.dim() != 2:
        raise ValueError(f"pre_scores must be [heads, keys], got {pre_scores.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    heads, key_count = map(int, pre_scores.shape)
    if key_count <= 0:
        raise ValueError("at least one key is required")
    requested = min(key_count, max(1, int(keep_count)))
    current = key_count - 1
    local_history = min(max(0, int(local_window)), max(0, requested - 1), current)
    local_start = current - local_history
    remaining_after_local = requested - (local_history + 1)
    sink_count = min(
        max(0, int(sink_tokens)),
        max(0, local_start),
        max(0, remaining_after_local),
    )
    remote_start = sink_count
    remote_token_count = max(0, local_start - remote_start)
    complete_blocks = remote_token_count // block_size
    remote_end = remote_start + complete_blocks * block_size

    base_count = sink_count + local_history + 1
    affordable_blocks = max(0, (requested - base_count) // block_size)
    selected_blocks = min(complete_blocks, affordable_blocks)

    if selected_blocks > 0:
        block_scores = pre_scores[:, remote_start:remote_end].reshape(
            heads, complete_blocks, block_size
        )
        block_max = block_scores.amax(dim=-1)
        block_ids = torch.topk(
            block_max, selected_blocks, dim=-1, largest=True, sorted=False
        ).indices
        block_ids = block_ids.sort(dim=-1).values
        offsets = torch.arange(block_size, device=pre_scores.device)
        remote_positions = (
            remote_start
            + block_ids.unsqueeze(-1) * block_size
            + offsets.view(1, 1, -1)
        ).reshape(heads, -1)
        selected_scores = pre_scores.gather(1, remote_positions).reshape(
            heads, selected_blocks, block_size
        )
        anchor_offsets = selected_scores.argmax(dim=-1)
        anchor_positions = (
            remote_start + block_ids * block_size + anchor_offsets
        )
    else:
        block_ids = torch.empty(
            (heads, 0), dtype=torch.long, device=pre_scores.device
        )
        remote_positions = torch.empty(
            (heads, 0), dtype=torch.long, device=pre_scores.device
        )
        anchor_positions = torch.empty_like(block_ids)

    sink = torch.arange(sink_count, device=pre_scores.device).view(1, -1).expand(
        heads, -1
    )
    local = torch.arange(local_start, key_count, device=pre_scores.device).view(
        1, -1
    ).expand(heads, -1)
    positions = torch.cat((sink, remote_positions, local), dim=-1)
    remote_mask = torch.zeros_like(positions, dtype=torch.bool)
    if remote_positions.shape[1] > 0:
        remote_mask[:, sink_count : sink_count + remote_positions.shape[1]] = True
    actual = int(positions.shape[-1])
    if actual > requested:
        raise AssertionError(f"actual selection {actual} exceeds budget {requested}")
    if selected_blocks and requested - actual >= block_size:
        raise AssertionError("selection left an affordable complete block unused")

    return BlockSelection(
        positions=positions,
        remote_mask=remote_mask,
        selected_block_ids=block_ids,
        anchor_positions=anchor_positions,
        block_size=block_size,
        requested_keep_count=requested,
        actual_keep_count=actual,
        local_start=local_start,
        remote_start=remote_start,
        remote_end=remote_end,
    )


def gather_head_tokens(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather ``[1, heads, keys, dim]`` at per-head positions."""

    index = positions.view(1, positions.shape[0], -1, 1).expand(
        1, positions.shape[0], positions.shape[1], values.shape[-1]
    )
    return values.gather(2, index)


def gather_head_scores(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return scores[0, :, 0, :].gather(1, positions)


def scores_at_head_distances(
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    effective_distance: torch.Tensor,
    inv_freq: torch.Tensor,
    rotate_half: Any,
    attention_scale: float,
    score_scale: float,
) -> torch.Tensor:
    """Evaluate QK for a different effective distance per head and key."""

    if query_pre.shape[0] != 1 or query_pre.shape[-2] != 1:
        raise ValueError("one batch and one query are required")
    if key_pre.shape[:2] != query_pre.shape[:2]:
        raise ValueError("query/key batch and head dimensions must match")
    if effective_distance.shape != key_pre.shape[1:3]:
        raise ValueError(
            f"distance shape {effective_distance.shape} != {key_pre.shape[1:3]}"
        )
    head_dim = int(key_pre.shape[-1])
    pair_phase = (
        effective_distance.float().unsqueeze(-1)
        * inv_freq.float().view(1, 1, -1)
    )
    cos = phase._expand_pair_values(torch.cos(pair_phase), head_dim).to(key_pre.dtype)
    # Cached keys precede the query, hence the relative rotation is -distance.
    sin = phase._expand_pair_values(-torch.sin(pair_phase), head_dim).to(key_pre.dtype)
    rotated_key = key_pre * cos.unsqueeze(0) + rotate_half(key_pre) * sin.unsqueeze(0)
    scores = (
        query_pre[0, :, 0, :].unsqueeze(1) * rotated_key[0]
    ).sum(dim=-1)
    return scores * float(score_scale) * float(attention_scale) ** 2


def counterfactual_transport(
    query_pre: torch.Tensor,
    expanded_key_pre: torch.Tensor,
    native_post_scores: torch.Tensor,
    selection: BlockSelection,
    *,
    current_position: int,
    local_anchor_distance: int,
    inv_freq: torch.Tensor,
    rotate_half: Any,
    attention_scale: float,
    score_scale: float,
) -> dict[str, torch.Tensor]:
    """Build a coherent translation from anchor counterfactual suppression.

    For selected block ``b``, the semantic anchor is its maximum pre-RoPE token.
    If placing that anchor at ``local_anchor_distance`` raises its score, the
    block is triggered and receives ``tau_b = delta_anchor - local_distance``.
    Every token in that block uses ``delta'_j = delta_j - tau_b``.
    """

    heads = int(selection.positions.shape[0])
    blocks = int(selection.selected_block_ids.shape[1])
    native_selected = gather_head_scores(native_post_scores, selection.positions)
    selected_key_pre = gather_head_tokens(expanded_key_pre, selection.positions)
    actual_distance = (
        int(current_position) - selection.positions
    ).clamp_min(0)
    tau_by_token = torch.zeros_like(actual_distance)

    if blocks == 0:
        empty = torch.empty((heads, 0), device=selection.positions.device)
        return {
            "native_selected": native_selected,
            "selected_key_pre": selected_key_pre,
            "actual_distance": actual_distance,
            "transport_distance": actual_distance,
            "trigger": empty.bool(),
            "suppression": empty,
            "tau": empty,
            "tau_by_token": tau_by_token,
        }

    anchor_key_pre = gather_head_tokens(
        expanded_key_pre, selection.anchor_positions
    )
    anchor_native = gather_head_scores(
        native_post_scores, selection.anchor_positions
    )
    local_distance = torch.full(
        (heads, blocks),
        int(local_anchor_distance),
        dtype=torch.long,
        device=selection.positions.device,
    )
    anchor_local = scores_at_head_distances(
        query_pre,
        anchor_key_pre,
        local_distance,
        inv_freq,
        rotate_half,
        attention_scale,
        score_scale,
    )
    suppression = anchor_local.float() - anchor_native.float()
    trigger = suppression > 0.0
    anchor_distance = (
        int(current_position) - selection.anchor_positions
    ).clamp_min(0)
    tau = torch.where(
        trigger,
        anchor_distance - int(local_anchor_distance),
        torch.zeros_like(anchor_distance),
    )

    remote_width = blocks * selection.block_size
    # Positions are laid out as [sink | selected complete blocks | local].
    remote_offset = selection.remote_start
    tau_by_token[
        :, remote_offset : remote_offset + remote_width
    ] = tau.repeat_interleave(selection.block_size, dim=-1)
    transport_distance = actual_distance - tau_by_token

    return {
        "native_selected": native_selected,
        "selected_key_pre": selected_key_pre,
        "actual_distance": actual_distance,
        "transport_distance": transport_distance,
        "trigger": trigger,
        "suppression": suppression,
        "tau": tau,
        "tau_by_token": tau_by_token,
    }


def matched_random_trigger_mask(
    trigger: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    """Choose exactly the same number of blocks per head deterministically."""

    if trigger.shape != block_ids.shape:
        raise ValueError("trigger and block_ids must have identical shapes")
    heads, blocks = map(int, trigger.shape)
    output = torch.zeros_like(trigger)
    if blocks == 0:
        return output
    modulus = 2_147_483_647
    for head in range(heads):
        count = int(trigger[head].sum().item())
        if count == 0:
            continue
        ids = block_ids[head].to(torch.int64)
        hashes = (
            ids * 1_103_515_245
            + (head + 1) * 12_345
            + (int(layer_index) + 1) * 2_654_435_761
        ) % modulus
        chosen = torch.topk(hashes, count, largest=False, sorted=False).indices
        output[head, chosen] = True
    return output


def apply_random_matched_transport(
    transport: dict[str, torch.Tensor],
    selection: BlockSelection,
    random_trigger: torch.Tensor,
    *,
    current_position: int,
    local_anchor_distance: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = int(selection.selected_block_ids.shape[1])
    if blocks == 0:
        return transport["actual_distance"], transport["tau"]
    anchor_distance = (
        int(current_position) - selection.anchor_positions
    ).clamp_min(0)
    random_tau = torch.where(
        random_trigger,
        anchor_distance - int(local_anchor_distance),
        torch.zeros_like(anchor_distance),
    )
    remote_width = blocks * selection.block_size
    tau_by_token = torch.zeros_like(transport["actual_distance"])
    tau_by_token[
        :, selection.remote_start : selection.remote_start + remote_width
    ] = (
        random_tau.repeat_interleave(selection.block_size, dim=-1)
    )
    return transport["actual_distance"] - tau_by_token, random_tau


def maximum_block_relative_error(
    original_distance: torch.Tensor,
    transported_distance: torch.Tensor,
    selection: BlockSelection,
) -> float:
    blocks = int(selection.selected_block_ids.shape[1])
    if blocks == 0:
        return 0.0
    start = selection.remote_start
    end = start + blocks * selection.block_size
    before = original_distance[:, start:end].reshape(
        original_distance.shape[0], blocks, selection.block_size
    )
    after = transported_distance[:, start:end].reshape_as(before)
    if selection.block_size == 1:
        return 0.0
    before_diff = before[..., 1:] - before[..., :-1]
    after_diff = after[..., 1:] - after[..., :-1]
    return float((after_diff - before_diff).abs().max().item())


def _record_block_metrics(
    controller: runner.Controller,
    selection: BlockSelection,
    transport: dict[str, torch.Tensor],
    random_trigger: torch.Tensor | None,
    total_tokens: int,
) -> None:
    metrics = controller.metrics
    trigger = transport["trigger"]
    trigger_count = int(trigger.sum().item())
    block_count = int(trigger.numel())
    metrics.block_head_rows = int(getattr(metrics, "block_head_rows", 0)) + int(
        selection.positions.shape[0]
    )
    metrics.block_selected_per_head_sum = float(
        getattr(metrics, "block_selected_per_head_sum", 0.0)
    ) + float(selection.selected_block_ids.numel())
    metrics.block_selected_count = int(
        getattr(metrics, "block_selected_count", 0)
    ) + block_count
    metrics.block_trigger_count = int(
        getattr(metrics, "block_trigger_count", 0)
    ) + trigger_count
    metrics.block_trigger_gap_sum = float(
        getattr(metrics, "block_trigger_gap_sum", 0.0)
    ) + float(transport["suppression"].masked_select(trigger).sum().item())
    metrics.block_trigger_tau_sum = float(
        getattr(metrics, "block_trigger_tau_sum", 0.0)
    ) + float(transport["tau"].abs().masked_select(trigger).sum().item())
    metrics.block_random_trigger_count = int(
        getattr(metrics, "block_random_trigger_count", 0)
    ) + (0 if random_trigger is None else int(random_trigger.sum().item()))
    metrics.block_selected_token_sum = int(
        getattr(metrics, "block_selected_token_sum", 0)
    ) + int(selection.positions.numel())
    metrics.block_total_token_sum = int(
        getattr(metrics, "block_total_token_sum", 0)
    ) + int(total_tokens * selection.positions.shape[0])


def blockwise_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = runner._ACTIVE_CONTROLLER
    if controller is None or controller.variant not in BLOCKWISE_VARIANTS:
        return _DELEGATE_FORWARD(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
    if hidden_states.shape[-2] != 1:
        return self._local_global_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._local_global_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre,
        current_key_pre,
        cos.to(query_pre.device),
        sin.to(query_pre.device),
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_post, value = past_key_value.update(
            current_key_post, current_value, self.layer_idx, cache_kwargs
        )
    else:
        key_post, value = current_key_post, current_value

    groups = query_post.shape[1] // key_post.shape[1]
    expanded_key_post = runner.repeat_kv(key_post, groups)
    expanded_value = runner.repeat_kv(value, groups)
    key_count = int(expanded_key_post.shape[-2])
    score_scale = float(
        getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1]))
    )
    native_post = torch.matmul(
        query_post, expanded_key_post.transpose(2, 3)
    ) * score_scale
    native_post = runner.add_attention_mask(native_post, attention_mask)

    attention_scale = float(
        getattr(
            self,
            "_phase_attention_scale",
            rope_repair._attention_scaling((cos, sin)),
        )
    )
    cached_key_pre = getattr(self, "_phase_pre_key_cache", None)
    if cached_key_pre is None or int(cached_key_pre.shape[2]) != key_count - 1:
        captured = -1 if cached_key_pre is None else int(cached_key_pre.shape[2])
        raise RuntimeError(
            f"layer {self.layer_idx} exact pre-RoPE cache mismatch: "
            f"captured={captured}, expected={key_count - 1}"
        )
    key_pre = torch.cat(
        (cached_key_pre.to(current_key_pre.device), current_key_pre), dim=2
    )
    expanded_key_pre = runner.repeat_kv(key_pre, groups)
    pre_scores = torch.matmul(
        query_pre, expanded_key_pre.transpose(2, 3)
    ) * score_scale
    pre_scores = runner.add_attention_mask(pre_scores, attention_mask)

    keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
    if controller.minimum_keep_tokens > 0:
        keep_count = max(keep_count, controller.minimum_keep_tokens)
    if controller.maximum_keep_tokens > 0:
        keep_count = min(keep_count, controller.maximum_keep_tokens)
    keep_count = min(key_count, keep_count)
    block_size = block_size_for_variant(controller.variant)
    selection = blockwise_premax_selection(
        pre_scores[0, :, 0, :],
        keep_count=keep_count,
        local_window=controller.local_window,
        sink_tokens=controller.sink_tokens,
        block_size=block_size,
    )
    selected_value = gather_head_tokens(expanded_value, selection.positions)
    transport = counterfactual_transport(
        query_pre,
        expanded_key_pre,
        native_post,
        selection,
        current_position=key_count - 1,
        local_anchor_distance=controller.local_window,
        inv_freq=self._local_global_inv_freq,
        rotate_half=modeling_qwen3.rotate_half,
        attention_scale=attention_scale,
        score_scale=score_scale,
    )

    selected_scores = transport["native_selected"]
    effective_distance_used = transport["actual_distance"]
    random_trigger: torch.Tensor | None = None
    if controller.variant.endswith("_selector_only"):
        pass
    elif controller.variant.endswith("_clipped_consumer"):
        clipped_distance = torch.where(
            selection.remote_mask,
            transport["actual_distance"].clamp_max(controller.local_window),
            transport["actual_distance"],
        )
        repaired = scores_at_head_distances(
            query_pre,
            transport["selected_key_pre"],
            clipped_distance,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
        )
        selected_scores = torch.where(
            selection.remote_mask, repaired, selected_scores
        )
        effective_distance_used = clipped_distance
    elif controller.variant.endswith("_random_matched"):
        random_trigger = matched_random_trigger_mask(
            transport["trigger"],
            selection.selected_block_ids,
            layer_index=int(self.layer_idx),
        )
        random_distance, _ = apply_random_matched_transport(
            transport,
            selection,
            random_trigger,
            current_position=key_count - 1,
            local_anchor_distance=controller.local_window,
        )
        repaired = scores_at_head_distances(
            query_pre,
            transport["selected_key_pre"],
            random_distance,
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
        )
        selected_scores = torch.where(
            selection.remote_mask, repaired, selected_scores
        )
        effective_distance_used = random_distance
    else:
        repaired = scores_at_head_distances(
            query_pre,
            transport["selected_key_pre"],
            transport["transport_distance"],
            self._local_global_inv_freq,
            modeling_qwen3.rotate_half,
            attention_scale,
            score_scale,
        )
        selected_scores = torch.where(
            selection.remote_mask, repaired, selected_scores
        )
        effective_distance_used = transport["transport_distance"]
        if controller.variant.endswith("_transport_masspreserve"):
            selected_scores = phase.preserve_remote_partition(
                selected_scores,
                transport["native_selected"],
                selection.remote_mask,
            )

    _record_block_metrics(
        controller,
        selection,
        transport,
        random_trigger,
        key_count,
    )
    error = maximum_block_relative_error(
        transport["actual_distance"],
        effective_distance_used,
        selection,
    )
    controller.metrics.block_relative_error_max = max(
        float(getattr(controller.metrics, "block_relative_error_max", 0.0)), error
    )

    sparse_scores = selected_scores.unsqueeze(0).unsqueeze(2)
    weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
    controller.record(
        selection.positions, weights, key_count, selection.remote_mask
    )
    attention_output = torch.matmul(weights, selected_value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attention_output), weights


runner.local_global_attention_forward = blockwise_attention_forward


if __name__ == "__main__":
    runner.main()
