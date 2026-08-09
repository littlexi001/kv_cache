from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache


class PreallocatedDynamicCache(DynamicCache):
    """DynamicCache semantics backed by fixed-capacity, in-place KV buffers."""

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        self.max_cache_len = int(max_cache_len)
        self._graph_decode_enabled = False
        self._graph_prefix_length = 0
        self.graph_cache_position: torch.Tensor | None = None
        self.graph_active_key_count: torch.Tensor | None = None

    def enable_graph_decode(self, prefix_length: int) -> None:
        """Switch one-token decode to fixed-address, device-indexed cache writes."""
        if not self.key_cache or not self.key_cache[0].numel():
            raise RuntimeError("prefill the cache before enabling graph decode")
        prefix_length = int(prefix_length)
        if prefix_length <= 0 or prefix_length >= self.max_cache_len:
            raise ValueError("graph prefix length is outside cache capacity")
        device = self.key_cache[0].device
        self._graph_decode_enabled = True
        self._graph_prefix_length = prefix_length
        self._seen_tokens = prefix_length
        self.graph_cache_position = torch.tensor(
            [prefix_length], dtype=torch.long, device=device
        )
        self.graph_active_key_count = torch.tensor(
            [prefix_length + 1], dtype=torch.int32, device=device
        )

    def set_graph_position(self, position: int) -> None:
        if not self._graph_decode_enabled:
            raise RuntimeError("graph decode is not enabled")
        if self.graph_cache_position is None or self.graph_active_key_count is None:
            raise RuntimeError("graph decode tensors are not initialized")
        position = int(position)
        if position < self._graph_prefix_length or position >= self.max_cache_len:
            raise ValueError("graph position is outside the suffix capacity")
        self.graph_cache_position.fill_(position)
        self.graph_active_key_count.fill_(position + 1)

    def disable_graph_decode(self, active_length: int | None = None) -> None:
        if active_length is None:
            active_length = self._graph_prefix_length
        self._seen_tokens = max(0, min(int(active_length), self.max_cache_len))
        self._graph_decode_enabled = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states is None or value_states is None:
            raise ValueError("key_states and value_states must be tensors")
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have the same shape")

        token_count = int(key_states.shape[-2])
        if self._graph_decode_enabled:
            if token_count != 1:
                raise RuntimeError("graph decode only supports one token per step")
            if self.graph_cache_position is None or self.graph_active_key_count is None:
                raise RuntimeError("graph decode tensors are not initialized")
            if layer_idx >= len(self.key_cache) or not self.key_cache[layer_idx].numel():
                raise RuntimeError("all graph-decode cache layers must be preallocated")
            self.key_cache[layer_idx].index_copy_(
                -2, self.graph_cache_position, key_states
            )
            self.value_cache[layer_idx].index_copy_(
                -2, self.graph_cache_position, value_states
            )
            if layer_idx == 0:
                self.graph_active_key_count.copy_(
                    self.graph_cache_position.to(torch.int32) + 1
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if layer_idx == 0:
            write_start = int(self._seen_tokens)
            write_end = write_start + token_count
            if write_end > self.max_cache_len:
                raise RuntimeError(
                    f"KV cache capacity exceeded: {write_end} > {self.max_cache_len}"
                )
            self._seen_tokens = write_end
        else:
            write_end = int(self._seen_tokens)
            write_start = write_end - token_count
            if write_start < 0:
                raise RuntimeError("nonzero layer updated before layer zero")

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([]))
            self.value_cache.append(torch.tensor([]))

        if not self.key_cache[layer_idx].numel():
            shape = list(key_states.shape)
            shape[-2] = self.max_cache_len
            self.key_cache[layer_idx] = torch.empty(
                shape,
                dtype=key_states.dtype,
                device=key_states.device,
            )
            self.value_cache[layer_idx] = torch.empty(
                shape,
                dtype=value_states.dtype,
                device=value_states.device,
            )
        else:
            expected_shape = list(key_states.shape)
            expected_shape[-2] = self.max_cache_len
            if list(self.key_cache[layer_idx].shape) != expected_shape:
                raise RuntimeError(
                    "KV shape changed after preallocation: "
                    f"{list(self.key_cache[layer_idx].shape)} != {expected_shape}"
                )

        self.key_cache[layer_idx][..., write_start:write_end, :].copy_(key_states)
        self.value_cache[layer_idx][..., write_start:write_end, :].copy_(value_states)
        return (
            self.key_cache[layer_idx][..., :write_end, :],
            self.value_cache[layer_idx][..., :write_end, :],
        )

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        del layer_idx
        return int(self._seen_tokens)

    def get_max_cache_shape(self) -> None:
        # Returning None preserves DynamicCache causal-mask behavior. The physical
        # capacity is an implementation detail and must not enter the mask shape.
        return None

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        self._seen_tokens = max(0, min(int(max_length), self.get_seq_length()))

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        active_length = self.get_seq_length()
        return tuple(
            (
                key[..., :active_length, :],
                value[..., :active_length, :],
            )
            for key, value in zip(self.key_cache, self.value_cache)
        )
