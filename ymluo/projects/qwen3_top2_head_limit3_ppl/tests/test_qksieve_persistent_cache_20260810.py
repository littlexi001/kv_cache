from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_head_top2_targeted_ppl_20260714 as qksieve  # noqa: E402


class CropCache:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        self.length = int(length)


def test_rewind_retains_buffers_and_resets_active_lengths(monkeypatch) -> None:
    codes = torch.empty(32, dtype=torch.uint8)
    values = torch.empty(32, dtype=torch.uint8)
    state = {
        "packed_qmse_indexed_count": 12,
        "packed_qmse_index": {
            "indexed_count": 12,
            "packed_codes": codes,
        },
        "qksieve_frozen_value_sketch_runtime": {
            "indexed_count": 12,
            "prefix": "qksieve_value_sketch_r16_i4_b256_wometric",
            "packed_codes": values,
        },
        "qksieve_value_sketch_r16_i4_b256_wometric_indexed_count": 12,
    }
    monkeypatch.setattr(qksieve, "_ACTIVE_QABS_PCA_STATES", {0: state})
    cache = CropCache(12)
    key_pointer = codes.data_ptr()
    value_pointer = values.data_ptr()

    result = qksieve.rewind_active_qksieve_cache(cache, 8)

    assert result == {"active_length": 8, "key_layers": 1, "value_layers": 1}
    assert cache.get_seq_length() == 8
    assert state["packed_qmse_indexed_count"] == 8
    assert state["packed_qmse_index"]["indexed_count"] == 8
    assert state["qksieve_frozen_value_sketch_runtime"]["indexed_count"] == 8
    assert state[
        "qksieve_value_sketch_r16_i4_b256_wometric_indexed_count"
    ] == 8
    assert codes.data_ptr() == key_pointer
    assert values.data_ptr() == value_pointer


def test_rewind_validates_before_mutating_cache(monkeypatch) -> None:
    state = {
        "packed_qmse_indexed_count": 6,
        "packed_qmse_index": {"indexed_count": 6},
    }
    monkeypatch.setattr(qksieve, "_ACTIVE_QABS_PCA_STATES", {0: state})
    cache = CropCache(12)

    with pytest.raises(RuntimeError, match="Key index has only 6 tokens"):
        qksieve.rewind_active_qksieve_cache(cache, 8)

    assert cache.get_seq_length() == 12


def test_rewind_rejects_forward_extension(monkeypatch) -> None:
    monkeypatch.setattr(qksieve, "_ACTIVE_QABS_PCA_STATES", {})
    cache = CropCache(12)

    with pytest.raises(ValueError, match="cannot rewind cache"):
        qksieve.rewind_active_qksieve_cache(cache, 13)

    assert cache.get_seq_length() == 12


def test_persistent_signature_exposes_buffer_identity(monkeypatch) -> None:
    key_codes = torch.empty(8, dtype=torch.uint8)
    key_scales = torch.empty(4, dtype=torch.float16)
    value_codes = torch.empty(8, dtype=torch.uint8)
    state = {
        "packed_qmse_indexed_count": 12,
        "packed_qmse_rebuild_count": 1,
        "packed_qmse_index": {
            "packed_codes": key_codes,
            "key_scales": key_scales,
        },
        "qksieve_frozen_value_sketch_runtime": {
            "indexed_count": 12,
            "packed_codes": value_codes,
        },
    }
    monkeypatch.setattr(qksieve, "_ACTIVE_QABS_PCA_STATES", {3: state})

    signature = qksieve.active_qksieve_persistent_state_signature()

    assert signature["layer_count"] == 1
    layer = signature["layers"][0]
    assert layer["layer"] == 3
    assert layer["key_rebuild_count"] == 1
    assert layer["key_code_ptr"] == key_codes.data_ptr()
    assert layer["key_scale_ptr"] == key_scales.data_ptr()
    assert layer["value_code_ptr"] == value_codes.data_ptr()
