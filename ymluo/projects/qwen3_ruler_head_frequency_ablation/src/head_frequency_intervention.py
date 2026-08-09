from __future__ import annotations

import contextlib
import inspect
import math
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def frequency_dimensions(frequency_pairs: Sequence[int], head_dim: int) -> tuple[int, ...]:
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    pairs = tuple(sorted(set(int(value) for value in frequency_pairs)))
    if any(value < 0 or value >= half for value in pairs):
        raise ValueError(f"frequency pairs must lie in [0, {half}), got {pairs}")
    return pairs + tuple(value + half for value in pairs)


def normalize_spec(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, dict[int, float]]]:
    normalized: dict[int, dict[int, dict[int, float]]] = {}
    for atom in spec.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) == "relative_distance":
            continue
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        scale = float(atom.get("frequency_scale", 0.0))
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"frequency_scale must lie in [0, 1], got {scale}")
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {}).setdefault(group, {})
                for frequency in frequencies:
                    if frequency in selected and selected[frequency] != scale:
                        raise ValueError(
                            f"conflicting scales for layer={layer}, group={group}, "
                            f"frequency={frequency}: {selected[frequency]} vs {scale}"
                        )
                    selected[frequency] = scale
    return normalized


def normalize_warp_starts(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, dict[int, float]]]:
    normalized: dict[int, dict[int, dict[int, float]]] = {}
    for atom in spec.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) == "relative_distance":
            continue
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        start = float(atom.get("position_warp_start", 0.0))
        if start < 0.0:
            raise ValueError(f"position_warp_start must be nonnegative, got {start}")
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {}).setdefault(group, {})
                for frequency in frequencies:
                    if frequency in selected and selected[frequency] != start:
                        raise ValueError(
                            f"conflicting warp starts for layer={layer}, group={group}, "
                            f"frequency={frequency}: {selected[frequency]} vs {start}"
                        )
                    selected[frequency] = start
    return normalized


def normalize_relative_warps(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, dict[int, tuple[float, float, float]]]]:
    normalized: dict[int, dict[int, dict[int, tuple[float, float, float]]]] = {}
    for atom in spec.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) != "relative_distance":
            continue
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        scale = float(atom.get("frequency_scale", 0.0))
        start = float(atom.get("position_warp_start", 0.0))
        blend = float(atom.get("score_blend", 1.0))
        if not 0.0 <= scale <= 1.0:
            raise ValueError(f"frequency_scale must lie in [0, 1], got {scale}")
        if start < 0.0:
            raise ValueError(f"position_warp_start must be nonnegative, got {start}")
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"score_blend must lie in [0, 1], got {blend}")
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {}).setdefault(group, {})
                for frequency in frequencies:
                    value = (scale, start, blend)
                    if frequency in selected and selected[frequency] != value:
                        raise ValueError(
                            f"conflicting relative warp for layer={layer}, group={group}, "
                            f"frequency={frequency}: {selected[frequency]} vs {value}"
                        )
                    selected[frequency] = value
    return normalized


def normalize_relative_query_starts(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, float]]:
    """Return the first absolute query position eligible for a relative warp.

    Relative-distance score corrections are pairwise and therefore expensive during
    a long prefill.  ``query_position_start`` permits an inference-time retrieval
    ablation that changes only the query tail while leaving filler-to-filler
    computation native.  Multiple atoms targeting one layer/group must agree.
    """
    normalized: dict[int, dict[int, float]] = {}
    for atom in spec.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) != "relative_distance":
            continue
        query_start = float(atom.get("query_position_start", 0.0))
        if query_start < 0.0:
            raise ValueError(
                f"query_position_start must be nonnegative, got {query_start}"
            )
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {})
                if group in selected and selected[group] != query_start:
                    raise ValueError(
                        f"conflicting query starts for layer={layer}, group={group}: "
                        f"{selected[group]} vs {query_start}"
                    )
                selected[group] = query_start
    return normalized


def normalize_relative_query_ends(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, float]]:
    """Return the exclusive absolute query-position end for relative warps."""
    normalized: dict[int, dict[int, float]] = {}
    for atom in spec.get("atoms", []):
        if str(atom.get("warp_mode", "absolute_position")) != "relative_distance":
            continue
        query_end = float(atom.get("query_position_end", math.inf))
        if query_end < 0.0:
            raise ValueError(f"query_position_end must be nonnegative, got {query_end}")
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {})
                if group in selected and selected[group] != query_end:
                    raise ValueError(
                        f"conflicting query ends for layer={layer}, group={group}: "
                        f"{selected[group]} vs {query_end}"
                    )
                selected[group] = query_end
    return normalized


def normalize_relative_gates(
    spec: Mapping[str, Any],
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[int, dict[int, dict[int, dict[str, float | int | str]]]]:
    normalized: dict[int, dict[int, dict[int, dict[str, float | int | str]]]] = {}
    for atom in spec.get("atoms", []):
        mode = str(atom.get("adaptive_gate", ""))
        if not mode:
            continue
        if str(atom.get("warp_mode", "absolute_position")) != "relative_distance":
            raise ValueError("adaptive gates are supported only for relative-distance warps")
        if mode not in {"remote_concentration", "semantic_topk"}:
            raise ValueError(f"unsupported adaptive gate: {mode}")
        layers = tuple(int(value) for value in atom["layers"])
        groups = tuple(int(value) for value in atom["head_groups"])
        frequencies = tuple(int(value) for value in atom["frequency_pairs"])
        if mode == "remote_concentration":
            remote_mass_scale = float(atom.get("adaptive_remote_mass_scale", 0.1))
            topk = int(atom.get("adaptive_topk", 16))
            topk_mass_scale = float(atom.get("adaptive_topk_mass_scale", 0.5))
            if remote_mass_scale <= 0.0:
                raise ValueError("adaptive_remote_mass_scale must be positive")
            if topk <= 0:
                raise ValueError("adaptive_topk must be positive")
            if topk_mass_scale <= 0.0:
                raise ValueError("adaptive_topk_mass_scale must be positive")
            config: dict[str, float | int | str] = {
                "mode": mode,
                "remote_mass_scale": remote_mass_scale,
                "topk": topk,
                "topk_mass_scale": topk_mass_scale,
            }
        else:
            topk_fraction = float(atom.get("adaptive_topk_fraction", 0.02))
            minimum_topk = int(atom.get("adaptive_minimum_topk", 1))
            replace_full_score = float(atom.get("adaptive_replace_full_score", 0.0))
            semantic_score_blend = float(atom.get("adaptive_semantic_score_blend", 1.0))
            if not 0.0 < topk_fraction <= 1.0:
                raise ValueError("adaptive_topk_fraction must lie in (0, 1]")
            if minimum_topk <= 0:
                raise ValueError("adaptive_minimum_topk must be positive")
            if replace_full_score not in {0.0, 1.0}:
                raise ValueError("adaptive_replace_full_score must be boolean-like")
            if not 0.0 <= semantic_score_blend <= 1.0:
                raise ValueError("adaptive_semantic_score_blend must lie in [0, 1]")
            config = {
                "mode": mode,
                "topk_fraction": topk_fraction,
                "minimum_topk": minimum_topk,
                "replace_full_score": replace_full_score,
                "semantic_score_blend": semantic_score_blend,
            }
        frequency_dimensions(frequencies, head_dim)
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"invalid layer in {layers}")
        if any(group < 0 or group >= num_kv_heads for group in groups):
            raise ValueError(f"invalid head group in {groups}")
        for layer in layers:
            for group in groups:
                selected = normalized.setdefault(layer, {}).setdefault(group, {})
                for frequency in frequencies:
                    if frequency in selected and selected[frequency] != config:
                        raise ValueError(
                            f"conflicting adaptive gate for layer={layer}, group={group}, "
                            f"frequency={frequency}"
                        )
                    selected[frequency] = dict(config)
    return normalized


class HeadFrequencyIntervention:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)
        self.num_query_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(model.config.head_dim)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.query_heads_per_group = self.num_query_heads // self.num_kv_heads
        self.rope_theta = float(getattr(model.config, "rope_theta", 10000.0))
        backbone = getattr(model, "model", None)
        self.rotary_embedding = getattr(backbone, "rotary_emb", None)
        self.active: dict[int, dict[int, dict[int, float]]] = {}
        self.active_warp_starts: dict[int, dict[int, dict[int, float]]] = {}
        self.active_relative: dict[
            int, dict[int, dict[int, tuple[float, float, float]]]
        ] = {}
        self.active_relative_query_starts: dict[int, dict[int, float]] = {}
        self.active_relative_query_ends: dict[int, dict[int, float]] = {}
        self.active_relative_gates: dict[
            int, dict[int, dict[int, dict[str, float | int | str]]]
        ] = {}
        self.bypass_original = False
        self._device_scales: dict[
            tuple[int, str], dict[int, tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        self._patch()

    def _scale_vectors(
        self, layer: int, device: torch.device
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        key = (layer, str(device))
        cached = self._device_scales.get(key)
        if cached is not None:
            return cached
        half = self.head_dim // 2
        cached = {}
        for group, frequency_scales in self.active.get(layer, {}).items():
            values = torch.ones(half, dtype=torch.float32, device=device)
            starts = torch.zeros(half, dtype=torch.float32, device=device)
            for frequency, scale in frequency_scales.items():
                values[frequency] = scale
                starts[frequency] = getattr(self, "active_warp_starts", {}).get(
                    layer, {}
                ).get(group, {}).get(frequency, 0.0)
            cached[group] = (values, starts)
        self._device_scales[key] = cached
        return cached

    def _scaled_rotation(
        self,
        qwen3: Any,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        rotated_q: torch.Tensor,
        rotated_k: torch.Tensor,
        layer: int,
        cache_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = int(query_states.shape[-2])
        if cache_position is None:
            positions = torch.arange(sequence_length, device=query_states.device)
        else:
            positions = cache_position.to(query_states.device).reshape(-1)
            if int(positions.numel()) != sequence_length:
                raise ValueError(
                    f"cache_position has {positions.numel()} values for sequence length {sequence_length}"
                )
        half = self.head_dim // 2
        rotary_embedding = getattr(self, "rotary_embedding", None)
        if rotary_embedding is None:
            inv_freq = self.rope_theta ** (
                -torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                    device=query_states.device,
                )
                / self.head_dim
            )
            attention_scaling: float | torch.Tensor = 1.0
        else:
            # The configured model may use YaRN (or another RoPE extension) at
            # lengths beyond its native window.  Read the live buffers after the
            # model rotary embedding has run so the intervention changes only the
            # requested phase, not the model's frequency schedule or mscale.
            inv_freq = rotary_embedding.inv_freq.to(
                device=query_states.device, dtype=torch.float32
            )
            attention_scaling = rotary_embedding.attention_scaling
            if torch.is_tensor(attention_scaling):
                attention_scaling = attention_scaling.to(
                    device=query_states.device, dtype=torch.float32
                )
        query_output = rotated_q.clone()
        key_output = rotated_k.clone()
        for group, (scales, starts) in self._scale_vectors(layer, query_states.device).items():
            raw_positions = positions.float()[:, None]
            warped_positions = torch.minimum(raw_positions, starts[None, :]) + scales[None, :] * (
                raw_positions - starts[None, :]
            ).clamp_min(0.0)
            angles = warped_positions * inv_freq[None, :]
            doubled = torch.cat((angles, angles), dim=-1)
            custom_cos = (doubled.cos() * attention_scaling).to(query_states.dtype)[
                None, None, :, :
            ]
            custom_sin = (doubled.sin() * attention_scaling).to(query_states.dtype)[
                None, None, :, :
            ]
            selected_pairs = scales != 1.0
            # Preserve the native tensor exactly before the warp threshold.  Recomputing
            # the same mathematical rotation can still change BF16 rounding and would
            # violate the method's short-context identity guarantee.
            warped_pair_positions = (raw_positions > starts[None, :]) & selected_pairs[None, :]
            selected_dimensions = torch.cat(
                (warped_pair_positions, warped_pair_positions), dim=-1
            )[None, None, :, :]
            first = group * self.query_heads_per_group
            last = first + self.query_heads_per_group
            query_group = query_states[:, first:last]
            custom_q = (query_group * custom_cos) + (qwen3.rotate_half(query_group) * custom_sin)
            query_output[:, first:last] = torch.where(
                selected_dimensions, custom_q, query_output[:, first:last]
            )
            key_group = key_states[:, group : group + 1]
            custom_k = (key_group * custom_cos) + (qwen3.rotate_half(key_group) * custom_sin)
            key_output[:, group : group + 1] = torch.where(
                selected_dimensions, custom_k, key_output[:, group : group + 1]
            )
        return query_output, key_output

    def _live_inv_freq(self, device: torch.device) -> torch.Tensor:
        rotary_embedding = getattr(self, "rotary_embedding", None)
        if rotary_embedding is not None:
            return rotary_embedding.inv_freq.to(device=device, dtype=torch.float32)
        return self.rope_theta ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
            / self.head_dim
        )

    def _relative_score_correction(
        self,
        query_group: torch.Tensor,
        key_group: torch.Tensor,
        frequency_warps: Mapping[int, tuple[float, float, float]],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        key_length = int(key_group.shape[-2])
        key_positions = torch.arange(
            key_length, device=query_group.device, dtype=torch.float32
        )
        relative_distance = query_positions.float()[:, None] - key_positions[None, :]
        inv_freq = self._live_inv_freq(query_group.device)
        half = self.head_dim // 2
        correction = torch.zeros(
            (
                query_group.shape[0],
                query_group.shape[1],
                query_group.shape[2],
                key_length,
            ),
            dtype=query_group.dtype,
            device=query_group.device,
        )
        for frequency, (scale, start, blend) in frequency_warps.items():
            remote = relative_distance > start
            phase_delta = (
                (1.0 - scale)
                * (relative_distance - start).clamp_min(0.0)
                * inv_freq[frequency]
            )
            cosine = phase_delta.cos().to(query_group.dtype)[None, None, :, :]
            sine = phase_delta.sin().to(query_group.dtype)[None, None, :, :]
            qx = query_group[..., frequency].unsqueeze(-1)
            qy = query_group[..., frequency + half].unsqueeze(-1)
            kx = key_group[..., frequency].unsqueeze(-2)
            ky = key_group[..., frequency + half].unsqueeze(-2)
            native = qx * kx + qy * ky
            cross = qy * kx - qx * ky
            desired = native * cosine + cross * sine
            pair_correction = torch.where(
                remote[None, None, :, :], blend * (desired - native), 0.0
            )
            correction = correction + pair_correction
        return correction

    def _remote_concentration_gate(
        self,
        query_group: torch.Tensor,
        key_group: torch.Tensor,
        query_positions: torch.Tensor,
        start: float,
        config: Mapping[str, float | int | str],
        scaling: float,
        attention_mask: torch.Tensor | None,
        first_head: int,
        last_head: int,
    ) -> torch.Tensor:
        key_length = int(key_group.shape[-2])
        key_positions = torch.arange(
            key_length, device=query_group.device, dtype=torch.float32
        )
        relative_distance = query_positions.float()[:, None] - key_positions[None, :]
        valid = relative_distance >= 0.0
        remote = relative_distance > float(start)
        logits = torch.matmul(
            query_group.float(), key_group.float().transpose(-1, -2)
        ) * float(scaling)
        if attention_mask is not None:
            causal = attention_mask[:, :, :, :key_length]
            if causal.shape[1] != 1:
                causal = causal[:, first_head:last_head]
            logits = logits + causal.float()
        else:
            logits = logits.masked_fill(
                ~valid[None, None, :, :], torch.finfo(logits.dtype).min
            )
        probabilities = torch.softmax(logits, dim=-1)
        remote_probabilities = probabilities * remote[None, None, :, :]
        remote_mass = remote_probabilities.sum(dim=-1, keepdim=True)
        topk = min(int(config["topk"]), key_length)
        remote_topk_mass = torch.topk(
            remote_probabilities, k=topk, dim=-1, largest=True, sorted=False
        ).values.sum(dim=-1, keepdim=True)
        conditional_topk_mass = remote_topk_mass / remote_mass.clamp_min(1e-12)
        mass_gate = (remote_mass / float(config["remote_mass_scale"])).clamp(0.0, 1.0)
        concentration_gate = (
            conditional_topk_mass / float(config["topk_mass_scale"])
        ).clamp(0.0, 1.0)
        return (mass_gate * concentration_gate).to(query_group.dtype)

    def _semantic_topk_selection(
        self,
        query_group: torch.Tensor,
        key_group: torch.Tensor,
        query_positions: torch.Tensor,
        start: float,
        config: Mapping[str, float | int | str],
        attention_mask: torch.Tensor | None,
        first_head: int,
        last_head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select remote keys by content similarity after analytically undoing RoPE."""
        key_length = int(key_group.shape[-2])
        key_positions = torch.arange(
            key_length, device=query_group.device, dtype=torch.float32
        )
        query_positions_float = query_positions.float()
        relative_distance = query_positions_float[:, None] - key_positions[None, :]
        remote = relative_distance > float(start)
        inv_freq = self._live_inv_freq(query_group.device)
        half = self.head_dim // 2

        query_angles = query_positions_float[:, None] * inv_freq[None, :]
        query_cosine = query_angles.cos().to(query_group.dtype)[None, None]
        query_sine = query_angles.sin().to(query_group.dtype)[None, None]
        query_x, query_y = query_group[..., :half], query_group[..., half:]
        query_pre = torch.cat(
            (
                query_x * query_cosine + query_y * query_sine,
                -query_x * query_sine + query_y * query_cosine,
            ),
            dim=-1,
        )

        key_angles = key_positions[:, None] * inv_freq[None, :]
        key_cosine = key_angles.cos().to(key_group.dtype)[None, None]
        key_sine = key_angles.sin().to(key_group.dtype)[None, None]
        key_x, key_y = key_group[..., :half], key_group[..., half:]
        key_pre = torch.cat(
            (
                key_x * key_cosine + key_y * key_sine,
                -key_x * key_sine + key_y * key_cosine,
            ),
            dim=-1,
        )
        semantic_scores = torch.matmul(
            query_pre.float(), key_pre.float().transpose(-1, -2)
        )
        retrieval_logits = semantic_scores.masked_fill(
            ~remote[None, None, :, :], torch.finfo(semantic_scores.dtype).min
        )
        if attention_mask is not None:
            causal = attention_mask[:, :, :, :key_length]
            if causal.shape[1] != 1:
                causal = causal[:, first_head:last_head]
            retrieval_logits = retrieval_logits + causal.float()

        remote_counts = remote.sum(dim=-1)
        fraction = float(config["topk_fraction"])
        minimum = int(config["minimum_topk"])
        selected_counts = torch.ceil(remote_counts.float() * fraction).to(torch.long)
        selected_counts = torch.where(
            remote_counts > 0,
            selected_counts.clamp_min(minimum),
            torch.zeros_like(selected_counts),
        )
        # Use a static upper bound to avoid a GPU-to-CPU synchronization in every
        # prefill chunk. Per-query counts are still enforced by ``keep_rank``.
        maximum = min(
            key_length,
            max(minimum, int(math.ceil(key_length * fraction))),
        )
        indices = torch.topk(
            retrieval_logits, k=maximum, dim=-1, largest=True, sorted=True
        ).indices
        ranks = torch.arange(maximum, device=query_group.device)
        keep_rank = ranks[None, :] < selected_counts[:, None]
        keep_rank = keep_rank[None, None].expand(
            query_group.shape[0], query_group.shape[1], -1, -1
        )
        gate = torch.zeros_like(retrieval_logits, dtype=query_group.dtype)
        gate.scatter_(-1, indices, keep_rank.to(gate.dtype))
        gate = gate * remote[None, None, :, :].to(gate.dtype)
        return gate, semantic_scores

    def _semantic_topk_gate(
        self,
        query_group: torch.Tensor,
        key_group: torch.Tensor,
        query_positions: torch.Tensor,
        start: float,
        config: Mapping[str, float | int | str],
        attention_mask: torch.Tensor | None,
        first_head: int,
        last_head: int,
    ) -> torch.Tensor:
        return self._semantic_topk_selection(
            query_group,
            key_group,
            query_positions,
            start,
            config,
            attention_mask,
            first_head,
            last_head,
        )[0]

    def _replace_relative_attention_heads(
        self,
        this: Any,
        attn_output: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer: int,
        cache_position: torch.Tensor | None,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        query_length = int(query_states.shape[-2])
        key_length = int(key_states.shape[-2])
        if cache_position is None:
            query_positions = torch.arange(
                key_length - query_length,
                key_length,
                device=query_states.device,
            )
        else:
            query_positions = cache_position.to(query_states.device).reshape(-1)
        updated = attn_output.clone()
        for group, frequency_warps in self.active_relative.get(layer, {}).items():
            smallest_start = min(start for _, start, _ in frequency_warps.values())
            if key_length <= smallest_start:
                continue
            first = group * self.query_heads_per_group
            last = first + self.query_heads_per_group
            query_start = self.active_relative_query_starts.get(layer, {}).get(group, 0.0)
            query_end = self.active_relative_query_ends.get(layer, {}).get(group, math.inf)
            eligible_queries = (query_positions >= query_start) & (query_positions < query_end)
            if not bool(eligible_queries.any()):
                continue
            selected_query_positions = query_positions[eligible_queries]
            query_group = query_states[:, first:last, eligible_queries, :].contiguous()
            key_group = key_states[:, group : group + 1].expand(
                -1, self.query_heads_per_group, -1, -1
            ).contiguous()
            value_group = value_states[:, group : group + 1].expand(
                -1, self.query_heads_per_group, -1, -1
            ).contiguous()
            score_correction = self._relative_score_correction(
                query_group,
                key_group,
                frequency_warps,
                selected_query_positions,
            ) * float(this.scaling)
            gate_configs = self.active_relative_gates.get(layer, {}).get(group, {})
            if gate_configs:
                configs = [gate_configs[frequency] for frequency in frequency_warps]
                if any(config != configs[0] for config in configs[1:]):
                    raise ValueError(
                        "all adaptively gated frequencies in one head group must share a gate"
                    )
                if configs[0]["mode"] == "remote_concentration":
                    gate = self._remote_concentration_gate(
                        query_group,
                        key_group,
                        selected_query_positions,
                        smallest_start,
                        configs[0],
                        float(this.scaling),
                        attention_mask,
                        first,
                        last,
                    )
                else:
                    gate, semantic_scores = self._semantic_topk_selection(
                        query_group,
                        key_group,
                        selected_query_positions,
                        smallest_start,
                        configs[0],
                        attention_mask,
                        first,
                        last,
                    )
                    if float(configs[0].get("replace_full_score", 0.0)):
                        native_scores = torch.matmul(
                            query_group.float(), key_group.float().transpose(-1, -2)
                        )
                        semantic_blend = float(configs[0]["semantic_score_blend"])
                        score_correction = (
                            semantic_blend
                            * (semantic_scores - native_scores)
                            * float(this.scaling)
                        ).to(score_correction.dtype)
                score_correction = score_correction * gate
            if attention_mask is not None:
                causal = attention_mask[:, :, eligible_queries, :key_length]
                if causal.shape[1] != 1:
                    causal = causal[:, first:last]
                score_correction = score_correction + causal.to(score_correction.dtype)
            else:
                valid = torch.arange(key_length, device=query_states.device)[None, :] <= (
                    selected_query_positions[:, None]
                )
                score_correction = score_correction.masked_fill(
                    ~valid[None, None, :, :],
                    torch.finfo(score_correction.dtype).min,
                )
            custom = F.scaled_dot_product_attention(
                query_group,
                key_group,
                value_group,
                attn_mask=score_correction,
                dropout_p=0.0 if not this.training else this.attention_dropout,
                scale=float(this.scaling),
                is_causal=False,
            )
            updated[:, eligible_queries, first:last, :] = custom.transpose(1, 2)
        return updated

    def _patch(self) -> None:
        from transformers.models.qwen3 import modeling_qwen3 as qwen3

        controller = self
        found = 0
        for module in self.model.modules():
            if module.__class__.__name__ != "Qwen3Attention":
                continue
            original = module.forward
            past_keyword = (
                "past_key_values"
                if "past_key_values" in inspect.signature(original).parameters
                else "past_key_value"
            )

            def wrapped_forward(
                this: Any,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_value: Any = None,
                past_key_values: Any = None,
                cache_position: torch.Tensor | None = None,
                _original: Any = original,
                _past_keyword: str = past_keyword,
                **kwargs: Any,
            ) -> Any:
                active_cache = past_key_values if past_key_values is not None else past_key_value
                if controller.bypass_original:
                    return _original(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        **{_past_keyword: active_cache},
                        **kwargs,
                    )
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, this.head_dim)
                query_states = this.q_norm(this.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = this.k_norm(this.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = this.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                rotated_q, rotated_k = qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)
                layer = int(this.layer_idx)
                if layer in controller.active:
                    query_states, key_states = controller._scaled_rotation(
                        qwen3,
                        query_states,
                        key_states,
                        rotated_q,
                        rotated_k,
                        layer,
                        cache_position,
                    )
                else:
                    query_states, key_states = rotated_q, rotated_k
                if active_cache is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = active_cache.update(
                        key_states, value_states, this.layer_idx, cache_kwargs
                    )
                attention_interface = qwen3.eager_attention_forward
                if this.config._attn_implementation != "eager":
                    attention_interface = qwen3.ALL_ATTENTION_FUNCTIONS[this.config._attn_implementation]
                attn_output, attn_weights = attention_interface(
                    this,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not this.training else this.attention_dropout,
                    scaling=this.scaling,
                    sliding_window=this.sliding_window,
                    **kwargs,
                )
                if layer in controller.active_relative:
                    attn_output = controller._replace_relative_attention_heads(
                        this,
                        attn_output,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        layer,
                        cache_position,
                    )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return this.o_proj(attn_output), attn_weights

            module.forward = types.MethodType(wrapped_forward, module)
            found += 1
        if found != self.num_layers:
            raise RuntimeError(f"patched {found} attention modules, expected {self.num_layers}")

    @contextlib.contextmanager
    def activate(self, spec: Mapping[str, Any]) -> Iterable[None]:
        previous = self.active
        previous_warp_starts = self.active_warp_starts
        previous_relative = self.active_relative
        previous_relative_query_starts = self.active_relative_query_starts
        previous_relative_query_ends = self.active_relative_query_ends
        previous_relative_gates = self.active_relative_gates
        previous_bypass = self.bypass_original
        previous_scales = self._device_scales
        self.active = normalize_spec(spec, self.num_layers, self.num_kv_heads, self.head_dim)
        self.active_warp_starts = normalize_warp_starts(
            spec, self.num_layers, self.num_kv_heads, self.head_dim
        )
        self.active_relative = normalize_relative_warps(
            spec, self.num_layers, self.num_kv_heads, self.head_dim
        )
        self.active_relative_query_starts = normalize_relative_query_starts(
            spec, self.num_layers, self.num_kv_heads, self.head_dim
        )
        self.active_relative_query_ends = normalize_relative_query_ends(
            spec, self.num_layers, self.num_kv_heads, self.head_dim
        )
        self.active_relative_gates = normalize_relative_gates(
            spec, self.num_layers, self.num_kv_heads, self.head_dim
        )
        self.bypass_original = bool(spec.get("bypass_original", False))
        self._device_scales = {}
        try:
            yield
        finally:
            self.active = previous
            self.active_warp_starts = previous_warp_starts
            self.active_relative = previous_relative
            self.active_relative_query_starts = previous_relative_query_starts
            self.active_relative_query_ends = previous_relative_query_ends
            self.active_relative_gates = previous_relative_gates
            self.bypass_original = previous_bypass
            self._device_scales = previous_scales
