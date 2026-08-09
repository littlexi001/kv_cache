from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers.cache_utils import Cache


@dataclass
class OffloadedPrefillLayerState:
    host_kv: torch.Tensor
    length: int
    device: torch.device


class OffloadedExactPrefillCache(Cache):
    """Exact chunked-prefill cache with only the active layer materialized on GPU."""

    is_compileable = False

    def __init__(self, capacity: int, pin_memory: bool = True) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.pin_memory = bool(pin_memory)
        self.states: list[OffloadedPrefillLayerState | None] = []
        self._seen_tokens = 0

    def _ensure_state(
        self, layer_idx: int, key_states: torch.Tensor
    ) -> OffloadedPrefillLayerState:
        while len(self.states) <= layer_idx:
            self.states.append(None)
        state = self.states[layer_idx]
        if state is None:
            batch, kv_heads, _, head_dim = key_states.shape
            host_kv = torch.empty(
                (2, batch, kv_heads, self.capacity, head_dim),
                dtype=key_states.dtype,
                device="cpu",
                pin_memory=self.pin_memory and key_states.is_cuda,
            )
            state = OffloadedPrefillLayerState(
                host_kv=host_kv,
                length=0,
                device=key_states.device,
            )
            self.states[layer_idx] = state
        return state

    @torch.inference_mode()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape or key_states.dim() != 4:
            raise ValueError("expected matching [batch, kv_heads, seq, dim] K/V")
        state = self._ensure_state(layer_idx, key_states)
        if key_states.device != state.device:
            raise ValueError("a cache layer cannot move between devices")
        old_length = state.length
        chunk_length = int(key_states.shape[-2])
        new_length = old_length + chunk_length
        if new_length > self.capacity:
            raise ValueError(
                f"prefill cache capacity {self.capacity} is smaller than {new_length}"
            )

        full_shape = (*key_states.shape[:-2], new_length, key_states.shape[-1])
        full_key = torch.empty(full_shape, dtype=key_states.dtype, device=key_states.device)
        full_value = torch.empty_like(full_key)
        if old_length:
            full_key[..., :old_length, :].copy_(
                state.host_kv[0, ..., :old_length, :], non_blocking=key_states.is_cuda
            )
            full_value[..., :old_length, :].copy_(
                state.host_kv[1, ..., :old_length, :], non_blocking=key_states.is_cuda
            )
        full_key[..., old_length:new_length, :].copy_(key_states)
        full_value[..., old_length:new_length, :].copy_(value_states)
        state.host_kv[0, ..., old_length:new_length, :].copy_(
            key_states, non_blocking=key_states.is_cuda
        )
        state.host_kv[1, ..., old_length:new_length, :].copy_(
            value_states, non_blocking=value_states.is_cuda
        )
        state.length = new_length
        if layer_idx == 0:
            self._seen_tokens += chunk_length
        return full_key, full_value

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        index = 0 if layer_idx is None else int(layer_idx)
        if index >= len(self.states) or self.states[index] is None:
            return 0
        state = self.states[index]
        assert state is not None
        return state.length

    def get_max_cache_shape(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for state in self.completed_states():
            yield state.host_kv[0, ..., : state.length, :], state.host_kv[
                1, ..., : state.length, :
            ]

    def completed_states(self) -> list[OffloadedPrefillLayerState]:
        if not self.states or any(state is None for state in self.states):
            raise RuntimeError("offloaded prefill cache has incomplete layer state")
        states = [state for state in self.states if state is not None]
        lengths = {state.length for state in states}
        if len(lengths) != 1:
            raise RuntimeError("offloaded prefill cache layers are not synchronized")
        return states

    def synchronize(self) -> None:
        devices = {
            state.device
            for state in self.completed_states()
            if state.device.type == "cuda"
        }
        for device in devices:
            torch.cuda.synchronize(device)

    def pinned_host_bytes(self) -> int:
        return sum(
            state.host_kv.numel() * state.host_kv.element_size()
            for state in self.completed_states()
        )


@dataclass
class QuantizedOffloadedPrefillLayerState:
    host_kv: torch.Tensor
    quantized_kv: torch.Tensor
    scales: torch.Tensor
    length: int
    device: torch.device
    head_dim: int
    quantization_group_size: int


class QuantizedOffloadedExactPrefillCache(Cache):
    """Keep a transient quantized GPU history while offloading exact K/V once."""

    is_compileable = False

    def __init__(
        self,
        capacity: int,
        bits: int = 4,
        group_size: int | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if bits not in {4, 8}:
            raise ValueError("prefill quantization bits must be 4 or 8")
        self.capacity = int(capacity)
        self.bits = int(bits)
        self.group_size = (
            None if group_size is None or int(group_size) <= 0 else int(group_size)
        )
        self.pin_memory = bool(pin_memory)
        self.states: list[
            QuantizedOffloadedPrefillLayerState | None
        ] = []
        self._seen_tokens = 0

    def _ensure_state(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
    ) -> QuantizedOffloadedPrefillLayerState:
        while len(self.states) <= layer_idx:
            self.states.append(None)
        state = self.states[layer_idx]
        if state is not None:
            return state
        batch, kv_heads, _, head_dim = key_states.shape
        if self.bits == 4 and head_dim % 2:
            raise ValueError("INT4 prefill requires an even head dimension")
        quantization_group_size = self.group_size or head_dim
        if head_dim % quantization_group_size:
            raise ValueError(
                "prefill quantization group size must divide the head dimension"
            )
        host_kv = torch.empty(
            (2, batch, kv_heads, self.capacity, head_dim),
            dtype=key_states.dtype,
            device="cpu",
            pin_memory=self.pin_memory and key_states.is_cuda,
        )
        code_dim = head_dim if self.bits == 8 else head_dim // 2
        code_dtype = torch.int8 if self.bits == 8 else torch.uint8
        quantized_kv = torch.empty(
            (2, batch, kv_heads, self.capacity, code_dim),
            dtype=code_dtype,
            device=key_states.device,
        )
        scales = torch.empty(
            (
                2,
                batch,
                kv_heads,
                self.capacity,
                head_dim // quantization_group_size,
            ),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        state = QuantizedOffloadedPrefillLayerState(
            host_kv=host_kv,
            quantized_kv=quantized_kv,
            scales=scales,
            length=0,
            device=key_states.device,
            head_dim=head_dim,
            quantization_group_size=quantization_group_size,
        )
        self.states[layer_idx] = state
        return state

    def _quantize(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum_code = 127 if self.bits == 8 else 7
        head_dim = int(values.shape[-1])
        group_size = self.group_size or head_dim
        grouped = values.float().reshape(
            *values.shape[:-1],
            head_dim // group_size,
            group_size,
        )
        scales = (
            grouped
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1.0e-8)
            / float(maximum_code)
        )
        codes = (
            torch.round(grouped / scales)
            .clamp(-maximum_code, maximum_code)
            .to(torch.int8)
            .reshape(*values.shape)
        )
        scales = scales.squeeze(-1)
        if self.bits == 8:
            return codes, scales.to(values.dtype)
        unsigned = (codes.to(torch.int16) & 0xF).to(torch.uint8)
        paired = unsigned.reshape(*unsigned.shape[:-1], -1, 2)
        packed = paired[..., 0] | (paired[..., 1] << 4)
        return packed, scales.to(values.dtype)

    def _dequantize(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
        head_dim: int,
    ) -> torch.Tensor:
        if self.bits == 8:
            unpacked = codes.to(dtype)
        else:
            low = (codes & 0xF).to(torch.int16)
            high = ((codes >> 4) & 0xF).to(torch.int16)
            low = torch.where(low >= 8, low - 16, low)
            high = torch.where(high >= 8, high - 16, high)
            unpacked = torch.stack((low, high), dim=-1).reshape(
                *codes.shape[:-1],
                head_dim,
            ).to(dtype)
        group_size = self.group_size or head_dim
        grouped = unpacked.reshape(
            *unpacked.shape[:-1],
            head_dim // group_size,
            group_size,
        )
        return (grouped * scales.unsqueeze(-1)).reshape(*unpacked.shape)

    @torch.inference_mode()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape or key_states.dim() != 4:
            raise ValueError("expected matching [batch, kv_heads, seq, dim] K/V")
        state = self._ensure_state(layer_idx, key_states)
        if key_states.device != state.device:
            raise ValueError("a cache layer cannot move between devices")
        if int(key_states.shape[-1]) != state.head_dim:
            raise ValueError("a cache layer cannot change head dimension")
        old_length = state.length
        chunk_length = int(key_states.shape[-2])
        new_length = old_length + chunk_length
        if new_length > self.capacity:
            raise ValueError(
                f"prefill cache capacity {self.capacity} is smaller than {new_length}"
            )

        full_shape = (*key_states.shape[:-2], new_length, state.head_dim)
        full_key = torch.empty(
            full_shape,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        full_value = torch.empty_like(full_key)
        if old_length:
            history_key = self._dequantize(
                state.quantized_kv[0, ..., :old_length, :],
                state.scales[0, ..., :old_length, :],
                key_states.dtype,
                state.head_dim,
            )
            history_value = self._dequantize(
                state.quantized_kv[1, ..., :old_length, :],
                state.scales[1, ..., :old_length, :],
                value_states.dtype,
                state.head_dim,
            )
            full_key[..., :old_length, :].copy_(history_key)
            full_value[..., :old_length, :].copy_(history_value)
            del history_key, history_value
        full_key[..., old_length:new_length, :].copy_(key_states)
        full_value[..., old_length:new_length, :].copy_(value_states)

        state.host_kv[0, ..., old_length:new_length, :].copy_(
            key_states,
            non_blocking=key_states.is_cuda,
        )
        state.host_kv[1, ..., old_length:new_length, :].copy_(
            value_states,
            non_blocking=value_states.is_cuda,
        )
        key_codes, key_scales = self._quantize(key_states)
        value_codes, value_scales = self._quantize(value_states)
        state.quantized_kv[0, ..., old_length:new_length, :].copy_(key_codes)
        state.quantized_kv[1, ..., old_length:new_length, :].copy_(
            value_codes
        )
        state.scales[0, ..., old_length:new_length, :].copy_(key_scales)
        state.scales[1, ..., old_length:new_length, :].copy_(
            value_scales
        )
        state.length = new_length
        if layer_idx == 0:
            self._seen_tokens += chunk_length
        return full_key, full_value

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        index = 0 if layer_idx is None else int(layer_idx)
        if index >= len(self.states) or self.states[index] is None:
            return 0
        state = self.states[index]
        assert state is not None
        return state.length

    def get_max_cache_shape(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for state in self.completed_states():
            yield state.host_kv[0, ..., : state.length, :], state.host_kv[
                1, ..., : state.length, :
            ]

    def completed_states(
        self,
    ) -> list[QuantizedOffloadedPrefillLayerState]:
        if not self.states or any(state is None for state in self.states):
            raise RuntimeError(
                "quantized offloaded prefill cache has incomplete layer state"
            )
        states = [state for state in self.states if state is not None]
        lengths = {state.length for state in states}
        if len(lengths) != 1:
            raise RuntimeError(
                "quantized offloaded prefill cache layers are not synchronized"
            )
        return states

    def synchronize(self) -> None:
        devices = {
            state.device
            for state in self.completed_states()
            if state.device.type == "cuda"
        }
        for device in devices:
            torch.cuda.synchronize(device)

    def pinned_host_bytes(self) -> int:
        return sum(
            state.host_kv.numel() * state.host_kv.element_size()
            for state in self.completed_states()
        )

    def transient_gpu_bytes(self) -> int:
        return sum(
            state.quantized_kv.numel()
            * state.quantized_kv.element_size()
            + state.scales.numel() * state.scales.element_size()
            for state in self.completed_states()
        )
