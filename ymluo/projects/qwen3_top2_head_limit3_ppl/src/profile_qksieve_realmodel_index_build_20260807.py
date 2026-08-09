#!/usr/bin/env python
"""Profile QKSieve's warmed request-local index build on a real model run.

The profiler monkey-patches coarse Python entry points instead of changing the
deployment path.  CUDA is synchronized around each measured call so the JSON
contains wall-clock stage costs rather than asynchronous launch latency.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

import run_head_top2_targeted_ppl_20260714 as qksieve
import run_qksieve_coldskip_longcontext_quality_20260730 as experiment


PROFILE_OUTPUT = Path(
    os.environ.get(
        "QKSIEVE_PROFILE_OUTPUT",
        "results/qksieve_realmodel_index_profile.json",
    )
)

_layer_stack: list[int | None] = []
_events: list[dict[str, Any]] = []
_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_per_layer: dict[str, dict[str, float]] = defaultdict(
    lambda: defaultdict(float)
)
_index_hashes: list[dict[str, Any]] = []
_profile_hashes = os.environ.get(
    "QKSIEVE_PROFILE_INDEX_HASHES", ""
).strip().lower() in {"1", "true", "yes"}


def _synchronize() -> None:
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def _history_count(args: tuple[Any, ...]) -> int | None:
    for value in args:
        if isinstance(value, torch.Tensor) and value.ndim >= 3:
            return int(value.shape[-2])
    return None


def _active_layer(args: tuple[Any, ...]) -> int | None:
    for value in args:
        if isinstance(value, dict) and "layer_index" in value:
            return int(value["layer_index"])
    return _layer_stack[-1] if _layer_stack else None


def _record(
    name: str,
    seconds: float,
    args: tuple[Any, ...],
) -> None:
    layer = _active_layer(args)
    history_count = _history_count(args)
    _totals[name] += seconds
    _counts[name] += 1
    if layer is not None:
        _per_layer[name][str(layer)] += seconds
    _events.append(
        {
            "stage": name,
            "seconds": seconds,
            "layer": layer,
            "history_count": history_count,
        }
    )


def _record_index_hashes(state: dict[str, Any]) -> None:
    if not _profile_hashes:
        return
    row: dict[str, Any] = {"layer": int(state.get("layer_index", -1))}
    for name in (
        "packed_qmse_key_second_moment",
        "packed_qmse_query_second_moment",
        "basis",
        "query_basis",
        "packed_qmse_allocation",
        "packed_qmse_scale_metrics",
    ):
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            continue
        raw = value.detach().contiguous().cpu().numpy().tobytes()
        row[name] = hashlib.sha256(raw).hexdigest()
    packed_index = state.get("packed_qmse_index")
    if isinstance(packed_index, dict):
        history_count = int(
            state.get(
                "packed_qmse_indexed_count",
                packed_index.get("indexed_count", 0),
            )
        )

        def active_flat_hash(
            value_name: str,
            base_name: str,
            stride_name: str,
        ) -> str | None:
            value = packed_index.get(value_name)
            bases = packed_index.get(base_name)
            strides = packed_index.get(stride_name)
            if not all(
                isinstance(item, torch.Tensor)
                for item in (value, bases, strides)
            ):
                return None
            digest = hashlib.sha256()
            bases_cpu = bases.detach().cpu()
            strides_cpu = strides.detach().cpu()
            for batch_index in range(int(bases_cpu.shape[0])):
                for head_index in range(int(bases_cpu.shape[1])):
                    base = int(bases_cpu[batch_index, head_index].item())
                    stride = int(
                        strides_cpu[batch_index, head_index].item()
                    )
                    active = value[
                        base : base + history_count * stride
                    ]
                    digest.update(
                        active.detach().contiguous().cpu().numpy().tobytes()
                    )
            return digest.hexdigest()

        packed_code_hash = active_flat_hash(
            "packed_codes", "code_bases", "code_strides"
        )
        key_scale_hash = active_flat_hash(
            "key_scales", "scale_bases", "scale_strides"
        )
        if packed_code_hash is not None:
            row["packed_index.packed_codes_active"] = packed_code_hash
        if key_scale_hash is not None:
            row["packed_index.key_scales_active"] = key_scale_hash
        score_bias = packed_index.get("score_bias")
        if isinstance(score_bias, torch.Tensor) and score_bias.numel():
            active_bias = score_bias[..., :history_count]
            raw = active_bias.detach().contiguous().cpu().numpy().tobytes()
            row["packed_index.score_bias_active"] = hashlib.sha256(
                raw
            ).hexdigest()
    _index_hashes.append(row)


def _record_value_hashes(
    value_history: torch.Tensor,
    state: dict[str, Any],
    prefix: str,
    block_count: int,
) -> None:
    if not _profile_hashes:
        return
    layer = int(state.get("layer_index", -1))
    row = next(
        (item for item in _index_hashes if int(item["layer"]) == layer),
        None,
    )
    if row is None:
        row = {"layer": layer}
        _index_hashes.append(row)
    history_count = int(value_history.shape[-2])
    tensors = {
        "mean": state.get(f"{prefix}_mean"),
        "basis": state.get(f"{prefix}_basis"),
        "encoder": state.get(f"{prefix}_encoder"),
        "explained": state.get(f"{prefix}_explained"),
        "rank8_residual": state.get(f"{prefix}_rank8_residual"),
        "packed_codes": state.get(f"{prefix}_packed_codes"),
        "minimum": state.get(f"{prefix}_minimum"),
        "scale": state.get(f"{prefix}_scale"),
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            continue
        active = value
        if name == "packed_codes":
            active = value[..., :history_count, :]
        elif name in {"minimum", "scale"}:
            active = value[..., :block_count, :]
        raw = active.detach().contiguous().cpu().numpy().tobytes()
        row[f"value_sketch.{name}"] = hashlib.sha256(raw).hexdigest()


def _timed_wrapper(
    name: str,
    function: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        _synchronize()
        _record(name, time.perf_counter() - start, args)
        return result

    wrapped.__name__ = getattr(function, "__name__", name)
    wrapped.__doc__ = getattr(function, "__doc__", None)
    return wrapped


def _install_wrappers() -> None:
    original_initialize = qksieve._packed_qmse_initialize

    def initialize(*args: Any, **kwargs: Any) -> Any:
        layer = _active_layer(args)
        _layer_stack.append(layer)
        try:
            _synchronize()
            start = time.perf_counter()
            result = original_initialize(*args, **kwargs)
            _synchronize()
            _record(
                "packed_qmse_initialize_total",
                time.perf_counter() - start,
                args,
            )
            if len(args) >= 2 and isinstance(args[1], dict):
                _record_index_hashes(args[1])
            return result
        finally:
            _layer_stack.pop()

    qksieve._packed_qmse_initialize = initialize

    original_value_ensure = qksieve._ensure_value_sketch_state

    def value_ensure(*args: Any, **kwargs: Any) -> Any:
        _synchronize()
        start = time.perf_counter()
        result = original_value_ensure(*args, **kwargs)
        _synchronize()
        _record(
            "value_sketch_ensure_total",
            time.perf_counter() - start,
            args,
        )
        if (
            len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], dict)
        ):
            prefix, block_count = result
            _record_value_hashes(
                args[0],
                args[1],
                str(prefix),
                int(block_count),
            )
        return result

    qksieve._ensure_value_sketch_state = value_ensure

    qksieve_functions = {
        "qk_factor_legacy": "_qk_metric_projection_factors_with_key_spectrum",
        "qk_factor_legacy_cuda": (
            "_qk_metric_projection_factors_with_key_spectrum_cuda"
        ),
        "qk_factor_cholesky": "_qk_metric_projection_factors_cholesky",
        "qmse_rate_allocation": "_hierarchical_qmse_rate_allocation",
        "keymse_rate_allocation": "_hierarchical_key_rate_allocation",
        "packed_key_rebuild_total": "_packed_qmse_rebuild_index",
        "value_sketch_block_pack": "_block_affine_pack_int4",
        "small_matrix_eigh": "_small_matrix_eigh",
        "small_matrix_svd": "_small_matrix_svd",
        "small_matrix_cholesky": "_small_matrix_cholesky",
        "small_matrix_solve": "_small_matrix_solve",
        "value_factor_eigh": "_value_sketch_matrix_eigh",
        "value_factor_cholesky": "_value_sketch_matrix_cholesky",
        "value_factor_solve": "_value_sketch_matrix_solve",
    }
    for stage, attribute in qksieve_functions.items():
        if (
            int(os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "0")) > 0
            and stage.startswith("qk_factor_")
        ):
            continue
        original = getattr(qksieve, attribute)
        setattr(qksieve, attribute, _timed_wrapper(stage, original))

    # Importing the extensions here moves JIT/load cost outside the measured
    # request.  This matches a deployed process with AOT-built kernels.
    import variablebit_spectral_cuda_20260727 as variablebit_cuda
    import qksieve_valuesketch_cuda_20260801 as value_sketch_cuda
    import qksieve_query_cuda_20260728 as query_cuda
    import mixedblock_spectral_cuda_20260729 as mixedblock_cuda

    for extension_module in (
        variablebit_cuda,
        value_sketch_cuda,
        query_cuda,
        mixedblock_cuda,
    ):
        extension_module.load_extension()

    extension_functions = {
        "packed_key_allocate": "allocate_packed_index",
        "packed_key_encode": "encode_projected_keys_into",
    }
    for stage, attribute in extension_functions.items():
        original = getattr(variablebit_cuda, attribute)
        setattr(variablebit_cuda, attribute, _timed_wrapper(stage, original))


def _exclusive_summary() -> dict[str, float]:
    initialization = _totals.get("packed_qmse_initialize_total", 0.0)
    qk_factor = (
        _totals.get("qk_factor_legacy", 0.0)
        + _totals.get("qk_factor_legacy_cuda", 0.0)
        + _totals.get("qk_factor_cholesky", 0.0)
    )
    allocation = (
        _totals.get("qmse_rate_allocation", 0.0)
        + _totals.get("keymse_rate_allocation", 0.0)
    )
    key_rebuild = _totals.get("packed_key_rebuild_total", 0.0)
    key_encode = _totals.get("packed_key_encode", 0.0)
    key_allocate = _totals.get("packed_key_allocate", 0.0)
    value_total = _totals.get("value_sketch_ensure_total", 0.0)
    value_pack = _totals.get("value_sketch_block_pack", 0.0)
    value_factor = (
        _totals.get("value_factor_eigh", 0.0)
        + _totals.get("value_factor_cholesky", 0.0)
        + _totals.get("value_factor_solve", 0.0)
    )
    return {
        "qk_initialization_other_seconds": max(
            0.0,
            initialization - qk_factor - allocation - key_rebuild,
        ),
        "packed_key_projection_and_other_seconds": max(
            0.0,
            key_rebuild - key_encode - key_allocate,
        ),
        "value_sketch_projection_and_other_seconds": max(
            0.0,
            value_total - value_pack - value_factor,
        ),
    }


def _write_profile(wall_seconds: float, error: str | None) -> None:
    stages = {}
    for name in sorted(_totals):
        stages[name] = {
            "calls": _counts[name],
            "seconds": _totals[name],
            "mean_seconds": _totals[name] / max(1, _counts[name]),
            "per_layer_seconds": dict(sorted(_per_layer[name].items())),
        }
    payload = {
        "wall_seconds": wall_seconds,
        "error": error,
        "qk_factor_solver": os.environ.get(
            "QKSIEVE_QK_FACTOR_SOLVER", "legacy"
        ),
        "value_factor_solver": os.environ.get(
            "QKSIEVE_VALUE_FACTOR_SOLVER", "legacy"
        ),
        "stages": stages,
        "exclusive_estimates": _exclusive_summary(),
        "index_hashes": _index_hashes,
        "events": _events,
    }
    PROFILE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    _install_wrappers()
    start = time.perf_counter()
    error = None
    try:
        experiment.main()
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        _write_profile(time.perf_counter() - start, error)


if __name__ == "__main__":
    main()
