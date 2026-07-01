from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_qabs_guarded", False):
        return

    def guarded_register_fake(op_name: str, *args: Any, **kwargs: Any):
        decorator = register_fake(op_name, *args, **kwargs)

        def guarded_decorator(fn: Any) -> Any:
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "operator torchvision::" in str(exc) and "does not exist" in str(exc):
                    return fn
                raise

        return guarded_decorator

    guarded_register_fake._qabs_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

_ACTIVE_MODE: str = "baseline"
_ACTIVE_TOP_FRACTION: float = 0.02
_ACTIVE_MAX_HEADS_PER_TOKEN: int = 3
_ACTIVE_ALWAYS_KEEP_SELF: bool = True
_ACTIVE_PROTECT_SINK_TOKENS: int = 0
_ACTIVE_PROTECT_RECENT_TOKENS: int = 0
_ACTIVE_LOAD_STATS: "LoadStats | None" = None
_ACTIVE_OBS_STATE: "ObservationWindowState | None" = None
_ACTIVE_OBS_MASS_STATE: "AllTokenMassObservationState | None" = None
_ACTIVE_OBS_HYBRID_STATE: "HybridObservationState | None" = None
_ACTIVE_REUSE_STATE: "ReuseCandidateState | None" = None
_ACTIVE_KNORM_STATE: "KNormState | None" = None
_ACTIVE_BLOCK_ROUTE_STATE: "BlockRouteState | None" = None
_ACTIVE_KSIGN_STATE: "KSignIndexState | None" = None
_ACTIVE_KDOM_STATE: "KDomIndexState | None" = None
_ACTIVE_BREP_STATE: "BlockRepState | None" = None
_ACTIVE_SPARQ_MEAN_STATE: "SparQMeanState | None" = None
_FULL_HEAD_MAP_CACHE: dict[str, Any] | None = None
_FULL_LAYER_MAP_CACHE: dict[str, Any] | None = None
_LAYER_BUDGET_MAP_CACHE: dict[str, Any] | None = None
_FULL_LAYER_SELECTION_CACHE: dict[tuple[str, int], set[int]] = {}
_ACTIVE_FULL_HEAD_MAP_PATH: str = ""
_ACTIVE_FULL_LAYER_MAP_PATH: str = ""
_ACTIVE_LAYER_BUDGET_MAP_PATH: str = ""
_ACTIVE_CANDIDATE_STATS: "CandidateStats | None" = None
_ACTIVE_REUSE_OVERLAP_STATS: "ReuseOverlapStats | None" = None
_ACTIVE_EVIDENCE_COVERAGE_STATS: "EvidenceSpanCoverageStats | None" = None
_ACTIVE_EVIDENCE_SPANS: dict[str, tuple[int, int]] = {}
_ACTIVE_FORCE_EVIDENCE_SPANS: bool = False
_ACTIVE_QABS_FAST_PATH: bool = False
_ACTIVE_QABS_CUDA_FINAL_KERNEL: bool = False
_ACTIVE_QABS_CUDA_CANDIDATE_KERNEL: bool = False
_ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL: bool = False
_ACTIVE_QABS_PROFILE_STATS: "QabsReuseProfileStats | None" = None
_ACTIVE_QABS_CANDIDATE_SELECTION: str = "topk"
_ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE: int = 256
_QABS_CUDA_FINAL_WARNED: bool = False
_QABS_CUDA_CANDIDATE_WARNED: bool = False
_QABS_CUDA_FULL_SCORE_WARNED: bool = False
_QABS_CUDA_BREP_WARNED: bool = False
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_LAYER_BUDGET_QABS_REUSE_STATE: dict[int, dict[str, Any]] = {}
_BATCH_ROW_INDEX_TENSOR_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


def snapshot_layer_budget_qabs_reuse_state() -> dict[int, dict[str, Any]]:
    return copy.deepcopy(_LAYER_BUDGET_QABS_REUSE_STATE)


def restore_layer_budget_qabs_reuse_state(state: dict[int, dict[str, Any]]) -> None:
    global _LAYER_BUDGET_QABS_REUSE_STATE
    _LAYER_BUDGET_QABS_REUSE_STATE = copy.deepcopy(state)


@contextmanager
def evidence_span_coverage(
    stats: "EvidenceSpanCoverageStats | None",
    spans: dict[str, tuple[int, int]],
    force_spans: bool = False,
):
    global _ACTIVE_EVIDENCE_COVERAGE_STATS, _ACTIVE_EVIDENCE_SPANS, _ACTIVE_FORCE_EVIDENCE_SPANS
    previous = (_ACTIVE_EVIDENCE_COVERAGE_STATS, _ACTIVE_EVIDENCE_SPANS, _ACTIVE_FORCE_EVIDENCE_SPANS)
    _ACTIVE_EVIDENCE_COVERAGE_STATS = stats
    _ACTIVE_EVIDENCE_SPANS = dict(spans)
    _ACTIVE_FORCE_EVIDENCE_SPANS = bool(force_spans)
    try:
        yield
    finally:
        _ACTIVE_EVIDENCE_COVERAGE_STATS, _ACTIVE_EVIDENCE_SPANS, _ACTIVE_FORCE_EVIDENCE_SPANS = previous


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline, top2 historical attention, and top2 with max 3 heads per historical token."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_head_limit3_ppl")
    parser.add_argument("--prefill_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument(
        "--eval_chunk_size",
        type=int,
        default=None,
        help="Eval/decode chunk size. Defaults to --chunk_size. Use 1 for token-by-token decoding.",
    )
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--max_heads_per_token", type=int, default=3)
    parser.add_argument(
        "--protect_sink_tokens",
        type=int,
        default=0,
        help="For compatible sparse modes, keep all heads for the first N historical tokens.",
    )
    parser.add_argument(
        "--protect_recent_tokens",
        type=int,
        default=0,
        help="For compatible sparse modes, keep all heads for the most recent N historical tokens.",
    )
    parser.add_argument(
        "--always_keep_self",
        type=str2bool,
        default=True,
        help="Keep each query token's own key/value even when pruning historical tokens.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--modes", default="baseline,top2,top2limit3score")
    parser.add_argument(
        "--qabs_fast_path",
        type=str2bool,
        default=False,
        help="Use the experimental qabs reuse decode fast path that avoids the initial full QK matmul.",
    )
    parser.add_argument(
        "--qabs_cuda_final_kernel",
        type=str2bool,
        default=False,
        help=(
            "Use a lazy CUDA extension to fuse final sparse QK + softmax + V reduce in qabs fast path. "
            "Falls back to PyTorch if the extension is unavailable."
        ),
    )
    parser.add_argument(
        "--qabs_cuda_candidate_kernel",
        type=str2bool,
        default=False,
        help=(
            "Use a lazy CUDA extension for qabs partial-QK candidate scores in qabs fast path. "
            "Falls back to PyTorch if the extension is unavailable."
        ),
    )
    parser.add_argument(
        "--qabs_cuda_reuse_select_kernel",
        type=str2bool,
        default=False,
        help=(
            "Use the experimental fused CUDA selector for qabs reuse rerank + final index construction. "
            "This is off by default because the current selector uses a serial per-head top-k loop."
        ),
    )
    parser.add_argument(
        "--reuse_prefill_cache",
        type=str2bool,
        default=True,
        help=(
            "Run prefill once, clone the resulting KV cache for each mode, and time only cache clone + eval per mode. "
            "This is valid because sparse modes are applied only during eval/decode."
        ),
    )
    parser.add_argument(
        "--baseline_last",
        type=str2bool,
        default=True,
        help="Move baseline to the end of the mode list so sparse experiments finish first.",
    )
    parser.add_argument(
        "--disable_sparse_stats",
        type=str2bool,
        default=False,
        help="Disable sparse load/candidate CSV stats to avoid GPU sync during timing runs.",
    )
    parser.add_argument(
        "--qabs_overlap_stats",
        type=str2bool,
        default=False,
        help="Write qabs reuse overlap CSVs for current raw, previous raw, and previous final candidate sets.",
    )
    parser.add_argument(
        "--qabs_profile",
        type=str2bool,
        default=False,
        help="Profile qabs reuse fast-path stages and write per-layer timing/token CSVs. Adds CUDA sync overhead.",
    )
    parser.add_argument(
        "--qabs_candidate_selection",
        choices=["topk", "sample_quantile", "previous_threshold"],
        default="topk",
        help=(
            "How qabs fast path converts partial scores to the current raw candidate mask. "
            "topk is exact; sample_quantile estimates the 3% threshold from sampled tokens; "
            "previous_threshold reuses the previous decode step's partial-score threshold per layer/head."
        ),
    )
    parser.add_argument(
        "--qabs_threshold_sample_size",
        type=int,
        default=256,
        help="Sample size per layer/head for --qabs_candidate_selection sample_quantile.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Print eval progress every N chunks. Default 1 preserves previous verbose logging.",
    )
    parser.add_argument("--obs_window_tokens", type=int, default=100)
    parser.add_argument("--obs_recent_tokens", type=int, default=100)
    parser.add_argument("--obs_target_coverage", type=float, default=0.90)
    parser.add_argument("--obs_min_heads", type=int, default=1)
    parser.add_argument("--obs_max_heads", type=int, default=16)
    parser.add_argument(
        "--obs_fallback_all",
        type=str2bool,
        default=True,
        help="If true, tokens not seen in the last observation window keep all original top2-selected heads.",
    )
    parser.add_argument(
        "--full_head_map_path",
        default="",
        help="Optional JSON map for fullh modes. Overrides FULL_HEAD_MAP_PATH when set.",
    )
    parser.add_argument(
        "--full_layer_map_path",
        default="",
        help="Optional JSON map for fulll modes. Overrides FULL_LAYER_MAP_PATH/FULL_HEAD_MAP_PATH when set.",
    )
    parser.add_argument(
        "--layer_budget_map_path",
        default="",
        help=(
            "Optional JSON map for layerbudgetattn. Format: "
            "{'default': {'type':'full'}, 'layers': {'0': {'type':'landmark','recent':1024,'stride':64}}}. "
            "Layer budget types: full, recent, landmark, head_recent, synthetic."
        ),
    )
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if max_chars > 0:
            return handle.read(max_chars)
        return handle.read()


def pick_input_device(model: torch.nn.Module, fallback_device: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_modes(spec: str) -> list[str]:
    modes = [part.strip().lower() for part in spec.split(",") if part.strip()]
    invalid = [
        mode
        for mode in modes
        if parse_mode_config(mode)[0]
        not in {
            "baseline",
            "top2",
            "limit_random",
            "limit_score",
            "limit_score_fill",
            "limit_score_gap",
            "limit_score_protect",
            "top2_union_all",
            "observation_window",
            "observation_all_mass",
            "observation_hybrid",
            "sign_xnor",
            "sign_xnor_knorm",
            "sign_xnor_rerank",
            "recent_window",
            "landmark_recent_attention",
            "full_head_recent_attention",
            "full_layer_recent_attention",
            "full_layer_landmark_attention",
            "layer_budget_attention",
            "adaptive_full_layer_recent_attention",
            "qabs_candidate_keep",
            "qabs_partial_rerank",
            "qabs_reuse_rerank",
            "qabs_candidate_attention",
            "qabs_periodic_candidate_attention",
            "qabs_candidate_top2_attention",
            "qabs_candidate_top2_hlimit_attention",
            "qabs_candidate_top2_global_attention",
            "qabs_shared_candidate_attention",
            "qabs_shared_candidate_top2_attention",
            "qabs_shared_reuse_attention",
            "lagged_reuse_rerank",
            "knorm_reservoir",
            "block_route",
            "ksign_index_rerank",
            "kdom_index_rerank",
            "qabs_block_bound",
            "kdim_posting_rerank",
            "block_rep_rerank",
            "block_group_rep_rerank",
            "block_random_rep_rerank",
            "qabs_block_oracle",
            "synthetic_kv_attention",
            "sparq_attention",
            "sparq_fast_attention",
        }
    ]
    if invalid:
        raise ValueError(
            "Invalid modes: "
            f"{invalid}. Valid examples: baseline, top2, top2limit3, top2limit3score, "
            "top2union, obstop2fullnonmasst80kn2mn1, top2limit3gap1p0, "
            "top2limit3protects16r1p0, synthkv16meanattn, synthkv16massattn."
        )
    if not modes:
        raise ValueError("--modes cannot be empty.")
    return modes


def move_baseline_last(modes: list[str]) -> list[str]:
    baselines = [mode for mode in modes if mode == "baseline"]
    non_baselines = [mode for mode in modes if mode != "baseline"]
    return non_baselines + baselines


def parse_mode_config(mode: str) -> tuple[str, int | None]:
    if mode == "baseline":
        return "baseline", None
    if mode == "top2":
        return "top2", None
    if mode == "top2union":
        return "top2_union_all", None
    if mode == "top2obswin":
        return "observation_window", None
    if mode == "obsallmass":
        return "observation_all_mass", None
    if re.fullmatch(r"signxnor\d+(?:p\d+)?", mode):
        return "sign_xnor", None
    if re.fullmatch(r"signxnorknorm\d+(?:p\d+)?", mode):
        return "sign_xnor_knorm", None
    if re.fullmatch(r"signxnor\d+(?:p\d+)?rerank", mode):
        return "sign_xnor_rerank", None
    if re.fullmatch(r"recent\d+", mode):
        return "recent_window", None
    if re.fullmatch(r"landmarkr\d+s\d+attn", mode):
        return "landmark_recent_attention", None
    if re.fullmatch(r"fullh\d+recent\d+attn", mode):
        return "full_head_recent_attention", None
    if re.fullmatch(r"fulll\d+recent\d+attn", mode):
        return "full_layer_recent_attention", None
    if re.fullmatch(r"fulll\d+landmarkr\d+s\d+attn", mode):
        return "full_layer_landmark_attention", None
    if mode == "layerbudgetattn":
        return "layer_budget_attention", None
    if re.fullmatch(r"fullladapt\d+to\d+m\d+(?:p\d+)?recent\d+attn", mode):
        return "adaptive_full_layer_recent_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?keep", mode):
        return "qabs_candidate_keep", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?rerank", mode):
        return "qabs_partial_rerank", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?reuse(?:2set|final)?(?:blk\d+)?", mode):
        return "qabs_reuse_rerank", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?top2attn", mode):
        return "qabs_candidate_top2_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?top2hlim\d+attn", mode):
        return "qabs_candidate_top2_hlimit_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?top2globalattn", mode):
        return "qabs_candidate_top2_global_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?sharedr\d+attn", mode):
        return "qabs_shared_reuse_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?sharedtop2attn", mode):
        return "qabs_shared_candidate_top2_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?sharedattn", mode):
        return "qabs_shared_candidate_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?r\d+attn", mode):
        return "qabs_periodic_candidate_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?appendattn", mode):
        return "qabs_candidate_attention", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?attn", mode):
        return "qabs_candidate_attention", None
    if re.fullmatch(r"lagtop2qabs\d+cand\d+(?:p\d+)?", mode) or re.fullmatch(r"lagrefresh\d+qabs\d+cand\d+(?:p\d+)?", mode) or re.fullmatch(r"qdrift(?:maj|head|share|cos|coshead)?qabs\d+cand\d+(?:p\d+)?", mode) or re.fullmatch(r"pconf\d+(?:p\d+)?qabs\d+cand\d+(?:p\d+)?", mode):
        return "lagged_reuse_rerank", None
    if mode == "knormtop2":
        return "knorm_reservoir", None
    if re.fullmatch(r"blockroute\d+", mode):
        return "block_route", None
    if re.fullmatch(r"ksign(?:w|pm)?\d+cand\d+(?:p\d+)?", mode):
        return "ksign_index_rerank", None
    if re.fullmatch(r"kdomq\d+k\d+cand\d+(?:p\d+)?", mode):
        return "kdom_index_rerank", None
    if re.fullmatch(r"qbb\d+q\d+s\d+(?:p\d+)?cand\d+(?:p\d+)?", mode):
        return "qabs_block_bound", None
    if re.fullmatch(r"kdimq\d+cand\d+(?:p\d+)?", mode):
        return "kdim_posting_rerank", None
    if re.fullmatch(r"brep\d+r\d+q\d+s\d+(?:p\d+)?cand\d+(?:p\d+)?", mode):
        return "block_rep_rerank", None
    if re.fullmatch(r"bgrp\d+g\d+r\d+q\d+s\d+(?:p\d+)?cand\d+(?:p\d+)?", mode):
        return "block_group_rep_rerank", None
    if re.fullmatch(r"brp\d+p\d+r\d+q\d+s\d+(?:p\d+)?cand\d+(?:p\d+)?", mode):
        return "block_random_rep_rerank", None
    if re.fullmatch(r"qboracle\d+q\d+s\d+(?:p\d+)?cand\d+(?:p\d+)?", mode):
        return "qabs_block_oracle", None
    if re.fullmatch(r"synthkv\d+(?:mean|mass)attn", mode):
        return "synthetic_kv_attention", None
    if re.fullmatch(r"sparq\d+cand\d+(?:p\d+)?", mode):
        return "sparq_attention", None
    if re.fullmatch(r"sparqfast(?:dk)?\d+cand\d+(?:p\d+)?", mode):
        return "sparq_fast_attention", None
    if re.fullmatch(r"obshybridt\d+n\d+(?:kt\d+kn\d+)?", mode):
        return "observation_hybrid", None
    if re.fullmatch(r"obshybridmasst\d+n\d+(?:kt\d+kn\d+)?(?:mt\d+mn\d+)?", mode):
        return "observation_hybrid", None
    if re.fullmatch(r"obstop2fullnonmasst\d+(?:kn\d+)?(?:mn\d+)?", mode):
        return "observation_hybrid", None
    match = re.fullmatch(r"top2limit(\d+)protects(\d+)r([0-9]+(?:p[0-9]+)?)", mode)
    if match:
        return "limit_score_protect", int(match.group(1))
    match = re.fullmatch(r"top2limit(\d+)gap([0-9]+(?:p[0-9]+)?)", mode)
    if match:
        return "limit_score_gap", int(match.group(1))
    match = re.fullmatch(r"top2limit(\d+)(scorefill|score)?", mode)
    if match:
        max_heads = int(match.group(1))
        strategy = (
            "limit_score_fill"
            if match.group(2) == "scorefill"
            else "limit_score"
            if match.group(2) == "score"
            else "limit_random"
        )
        return strategy, max_heads
    return "invalid", None


def parse_gap_margin(mode: str) -> float | None:
    match = re.fullmatch(r"top2limit(\d+)gap([0-9]+(?:p[0-9]+)?)", mode)
    if not match:
        return None
    return float(match.group(2).replace("p", "."))


def parse_sign_xnor_candidate_fraction(mode: str) -> float | None:
    match = re.fullmatch(r"signxnor(?:knorm)?(\d+(?:p\d+)?)(?:rerank)?", mode)
    if not match:
        return None
    return float(match.group(1).replace("p", ".")) / 100.0


def parse_recent_window_tokens(mode: str) -> int | None:
    match = re.fullmatch(r"recent(\d+)", mode)
    if not match:
        return None
    return int(match.group(1))


def parse_landmark_recent_params(mode: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"landmarkr(\d+)s(\d+)attn", mode)
    if not match:
        return None
    recent_tokens = max(0, int(match.group(1)))
    stride = max(1, int(match.group(2)))
    return recent_tokens, stride


def parse_full_head_recent_params(mode: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"fullh(\d+)recent(\d+)attn", mode)
    if not match:
        return None
    full_heads = max(0, int(match.group(1)))
    recent_tokens = max(0, int(match.group(2)))
    return full_heads, recent_tokens


def parse_full_layer_recent_params(mode: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"fulll(\d+)recent(\d+)attn", mode)
    if not match:
        return None
    full_layers = max(0, int(match.group(1)))
    recent_tokens = max(0, int(match.group(2)))
    return full_layers, recent_tokens


def parse_full_layer_landmark_params(mode: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"fulll(\d+)landmarkr(\d+)s(\d+)attn", mode)
    if not match:
        return None
    full_layers = max(0, int(match.group(1)))
    recent_tokens = max(0, int(match.group(2)))
    landmark_stride = max(1, int(match.group(3)))
    return full_layers, recent_tokens, landmark_stride


def parse_adaptive_full_layer_recent_params(mode: str) -> tuple[int, int, float, int] | None:
    match = re.fullmatch(r"fullladapt(\d+)to(\d+)m(\d+(?:p\d+)?)recent(\d+)attn", mode)
    if not match:
        return None
    low_layers = max(0, int(match.group(1)))
    high_layers = max(0, int(match.group(2)))
    margin = float(match.group(3).replace("p", "."))
    recent_tokens = max(0, int(match.group(4)))
    if low_layers > high_layers:
        raise ValueError(f"Adaptive full-layer mode requires low<=high: {mode}")
    return low_layers, high_layers, margin, recent_tokens


def full_head_indices_for_layer(layer_idx: int, full_heads: int, head_count: int, device: torch.device) -> torch.Tensor:
    global _FULL_HEAD_MAP_CACHE
    requested = min(max(0, int(full_heads)), head_count)
    if requested <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    map_path = _ACTIVE_FULL_HEAD_MAP_PATH or os.environ.get("FULL_HEAD_MAP_PATH", "")
    heads: list[int] = []
    if map_path:
        if _FULL_HEAD_MAP_CACHE is None or _FULL_HEAD_MAP_CACHE.get("_path") != map_path:
            with open(map_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and "top_heads_by_layer" in loaded:
                loaded = loaded["top_heads_by_layer"]
            _FULL_HEAD_MAP_CACHE = {"_path": map_path, "data": loaded}
        loaded_data = _FULL_HEAD_MAP_CACHE["data"]
        layer_heads: Any = None
        if isinstance(loaded_data, dict):
            layer_heads = loaded_data.get(str(layer_idx), loaded_data.get(layer_idx))
        elif isinstance(loaded_data, list) and layer_idx < len(loaded_data):
            layer_heads = loaded_data[layer_idx]
        if isinstance(layer_heads, list):
            heads = [int(head) for head in layer_heads if 0 <= int(head) < head_count]
    if len(heads) < requested:
        seen = set(heads)
        heads.extend(head for head in range(head_count) if head not in seen)
    return torch.tensor(heads[:requested], dtype=torch.long, device=device)


def full_layer_indices(full_layers: int) -> set[int]:
    global _FULL_LAYER_MAP_CACHE
    requested = max(0, int(full_layers))
    if requested <= 0:
        return set()
    map_path = _ACTIVE_FULL_LAYER_MAP_PATH or os.environ.get(
        "FULL_LAYER_MAP_PATH",
        _ACTIVE_FULL_HEAD_MAP_PATH or os.environ.get("FULL_HEAD_MAP_PATH", ""),
    )
    cache_key = (map_path, requested)
    cached_selection = _FULL_LAYER_SELECTION_CACHE.get(cache_key)
    if cached_selection is not None:
        return cached_selection
    layers: list[int] = []
    if map_path:
        if _FULL_LAYER_MAP_CACHE is None or _FULL_LAYER_MAP_CACHE.get("_path") != map_path:
            with open(map_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            layer_scores: list[tuple[float, int]] = []
            if isinstance(loaded, dict) and isinstance(loaded.get("top_layers"), list):
                layers = [int(layer) for layer in loaded["top_layers"]]
            elif isinstance(loaded, dict) and isinstance(loaded.get("mean_remote_mass"), list):
                for layer_idx, row in enumerate(loaded["mean_remote_mass"]):
                    if isinstance(row, list):
                        layer_scores.append((float(sum(float(value) for value in row)), layer_idx))
                layers = [layer_idx for _, layer_idx in sorted(layer_scores, reverse=True)]
            _FULL_LAYER_MAP_CACHE = {"_path": map_path, "layers": layers}
        layers = list(_FULL_LAYER_MAP_CACHE.get("layers", []))
    if len(layers) < requested:
        seen = set(layers)
        layers.extend(layer for layer in range(requested) if layer not in seen)
    selected = set(layers[:requested])
    _FULL_LAYER_SELECTION_CACHE[cache_key] = selected
    return selected


def _layer_budget_from_parts(
    default_budget: dict[str, Any],
    layer_budgets: dict[str, Any],
    layer_idx: int,
    map_path: str,
) -> dict[str, Any]:
    budget = layer_budgets.get(str(layer_idx), layer_budgets.get(layer_idx, default_budget))
    if not isinstance(budget, dict):
        raise ValueError(f"Layer budget for layer {layer_idx} must be an object in {map_path}")
    return budget


def layer_budget_for_layer(layer_idx: int) -> dict[str, Any]:
    global _LAYER_BUDGET_MAP_CACHE
    map_path = _ACTIVE_LAYER_BUDGET_MAP_PATH or os.environ.get("LAYER_BUDGET_MAP_PATH", "")
    if not map_path:
        raise RuntimeError("layerbudgetattn requires --layer_budget_map_path or LAYER_BUDGET_MAP_PATH.")
    if _LAYER_BUDGET_MAP_CACHE is None or _LAYER_BUDGET_MAP_CACHE.get("_path") != map_path:
        with open(map_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Layer budget map must be a JSON object: {map_path}")
        default_budget = loaded.get("default", {"type": "full"})
        layer_budgets = loaded.get("layers", {})
        if not isinstance(default_budget, dict) or not isinstance(layer_budgets, dict):
            raise ValueError(f"Layer budget map requires object fields default/layers: {map_path}")
        batch_maps = loaded.get("batch_maps", loaded.get("batch", []))
        if batch_maps is None:
            batch_maps = []
        if not isinstance(batch_maps, list):
            raise ValueError(f"Layer budget map batch_maps must be a list: {map_path}")
        normalized_batch_maps: list[dict[str, Any]] = []
        for row_idx, row_map in enumerate(batch_maps):
            if not isinstance(row_map, dict):
                raise ValueError(f"batch_maps[{row_idx}] must be an object in {map_path}")
            row_default = row_map.get("default", default_budget)
            row_layers = row_map.get("layers", {})
            if not isinstance(row_default, dict) or not isinstance(row_layers, dict):
                raise ValueError(f"batch_maps[{row_idx}] requires object fields default/layers in {map_path}")
            normalized_batch_maps.append({"default": row_default, "layers": row_layers})
        _LAYER_BUDGET_MAP_CACHE = {
            "_path": map_path,
            "default": default_budget,
            "layers": layer_budgets,
            "batch_maps": normalized_batch_maps,
        }
    return _layer_budget_from_parts(
        _LAYER_BUDGET_MAP_CACHE["default"],
        _LAYER_BUDGET_MAP_CACHE["layers"],
        layer_idx,
        map_path,
    )


def layer_budgets_for_batch(layer_idx: int, batch_count: int) -> list[dict[str, Any]] | None:
    layer_budget_for_layer(layer_idx)
    if _LAYER_BUDGET_MAP_CACHE is None:
        return None
    batch_maps = _LAYER_BUDGET_MAP_CACHE.get("batch_maps", [])
    if not batch_maps:
        return None
    if len(batch_maps) != batch_count:
        map_path = str(_LAYER_BUDGET_MAP_CACHE.get("_path", ""))
        raise ValueError(
            f"Layer budget batch_maps length must match batch size: got {len(batch_maps)} maps "
            f"for batch {batch_count} in {map_path}"
        )
    map_path = str(_LAYER_BUDGET_MAP_CACHE.get("_path", ""))
    return [
        _layer_budget_from_parts(row_map["default"], row_map["layers"], layer_idx, map_path)
        for row_map in batch_maps
    ]


def batch_row_index_tensor(row_indices: list[int], device: torch.device) -> torch.Tensor:
    key = (str(device), tuple(row_indices))
    cached = _BATCH_ROW_INDEX_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    index = torch.tensor(row_indices, dtype=torch.long, device=device)
    _BATCH_ROW_INDEX_TENSOR_CACHE[key] = index
    return index


def parse_qabs_partial_rerank_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"(?:qabs|lagtop2qabs|lagrefresh\d+qabs|qdrift(?:maj|head|share|cos|coshead)?qabs|pconf\d+(?:p\d+)?qabs)(\d+)cand(\d+(?:p\d+)?)(?:keep|rerank|reuse(?:2set|final)?(?:blk\d+)?|r\d+attn|sharedr\d+attn|sharedtop2attn|sharedattn|top2(?:hlim\d+|global)?attn|appendattn|attn)?", mode)
    if not match:
        return None
    dim_count = int(match.group(1))
    candidate_fraction = float(match.group(2).replace("p", ".")) / 100.0
    return dim_count, candidate_fraction


def parse_qabs_reuse_block_size(mode: str) -> int | None:
    match = re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?reuse(?:2set|final)?blk(\d+)", mode)
    if not match:
        return None
    return max(1, int(match.group(1)))


def parse_qabs_top2_hlimit(mode: str) -> int | None:
    match = re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?top2hlim(\d+)attn", mode)
    if not match:
        return None
    return int(match.group(1))


def parse_qabs_shared_refresh_interval(mode: str) -> int | None:
    match = re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?sharedr(\d+)attn", mode)
    if not match:
        return None
    return max(1, int(match.group(1)))


def parse_qabs_periodic_refresh_interval(mode: str) -> int | None:
    match = re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?r(\d+)attn", mode)
    if not match:
        return None
    return max(1, int(match.group(1)))




def parse_ksign_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"ksign(?:w|pm)?(\d+)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    dim_count = int(match.group(1))
    candidate_fraction = float(match.group(2).replace("p", ".")) / 100.0
    return dim_count, candidate_fraction




def parse_qboracle_params(mode: str) -> tuple[int, int, float, float] | None:
    match = re.fullmatch(r"qboracle(\d+)q(\d+)s(\d+(?:p\d+)?)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    block_size = max(1, int(match.group(1)))
    dim_count = int(match.group(2))
    scan_fraction = float(match.group(3).replace("p", ".")) / 100.0
    candidate_fraction = float(match.group(4).replace("p", ".")) / 100.0
    return block_size, dim_count, scan_fraction, candidate_fraction


def parse_brp_params(mode: str) -> tuple[int, int, int, int, float, float] | None:
    match = re.fullmatch(r"brp(\d+)p(\d+)r(\d+)q(\d+)s(\d+(?:p\d+)?)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    block_size = max(1, int(match.group(1)))
    projection_count = max(1, int(match.group(2)))
    reps_per_projection = max(1, int(match.group(3)))
    dim_count = int(match.group(4))
    scan_fraction = float(match.group(5).replace("p", ".")) / 100.0
    candidate_fraction = float(match.group(6).replace("p", ".")) / 100.0
    return block_size, projection_count, reps_per_projection, dim_count, scan_fraction, candidate_fraction


def parse_bgrp_params(mode: str) -> tuple[int, int, int, int, float, float] | None:
    match = re.fullmatch(r"bgrp(\d+)g(\d+)r(\d+)q(\d+)s(\d+(?:p\d+)?)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    block_size = max(1, int(match.group(1)))
    group_count = max(1, int(match.group(2)))
    reps_per_group = max(1, int(match.group(3)))
    dim_count = int(match.group(4))
    scan_fraction = float(match.group(5).replace("p", ".")) / 100.0
    candidate_fraction = float(match.group(6).replace("p", ".")) / 100.0
    return block_size, group_count, reps_per_group, dim_count, scan_fraction, candidate_fraction


def parse_brep_params(mode: str) -> tuple[int, int, int, float, float] | None:
    match = re.fullmatch(r"brep(\d+)r(\d+)q(\d+)s(\d+(?:p\d+)?)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    block_size = max(1, int(match.group(1)))
    rep_count = max(1, int(match.group(2)))
    dim_count = int(match.group(3))
    scan_fraction = float(match.group(4).replace("p", ".")) / 100.0
    candidate_fraction = float(match.group(5).replace("p", ".")) / 100.0
    return block_size, rep_count, dim_count, scan_fraction, candidate_fraction


def parse_kdim_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"kdimq(\d+)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    dim_count = int(match.group(1))
    candidate_fraction = float(match.group(2).replace("p", ".")) / 100.0
    return dim_count, candidate_fraction


def parse_qbb_params(mode: str) -> tuple[int, int, float, float] | None:
    match = re.fullmatch(r"qbb(\d+)q(\d+)s(\d+(?:p\d+)?)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    block_size = max(1, int(match.group(1)))
    dim_count = int(match.group(2))
    scan_fraction = float(match.group(3).replace("p", ".")) / 100.0
    candidate_fraction = float(match.group(4).replace("p", ".")) / 100.0
    return block_size, dim_count, scan_fraction, candidate_fraction


def parse_kdom_params(mode: str) -> tuple[int, int, float] | None:
    match = re.fullmatch(r"kdomq(\d+)k(\d+)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    query_dim_count = int(match.group(1))
    key_dim_count = int(match.group(2))
    candidate_fraction = float(match.group(3).replace("p", ".")) / 100.0
    return query_dim_count, key_dim_count, candidate_fraction


def ksign_pm_mode(mode: str) -> bool:
    return re.fullmatch(r"ksignpm\d+cand\d+(?:p\d+)?", mode) is not None


def ksign_weighted_mode(mode: str) -> bool:
    return re.fullmatch(r"ksignw\d+cand\d+(?:p\d+)?", mode) is not None


def parse_block_route_size(mode: str) -> int | None:
    match = re.fullmatch(r"blockroute(\d+)", mode)
    if not match:
        return None
    return max(1, int(match.group(1)))


def parse_sparq_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"sparq(?:fast(?:dk)?)?(\d+)cand(\d+(?:p\d+)?)", mode)
    if not match:
        return None
    dim_count = int(match.group(1))
    candidate_fraction = float(match.group(2).replace("p", ".")) / 100.0
    return dim_count, candidate_fraction


def parse_synthetic_kv_params(mode: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"synthkv(\d+)(mean|mass)attn", mode)
    if not match:
        return None
    return max(1, int(match.group(1))), match.group(2)


def sparq_fast_uses_double_k(mode: str) -> bool:
    return mode.startswith("sparqfastdk")




def pconf_mode(mode: str) -> bool:
    return re.fullmatch(r"pconf\d+(?:p\d+)?qabs\d+cand\d+(?:p\d+)?", mode) is not None


def parse_pconf_threshold(mode: str) -> float | None:
    match = re.fullmatch(r"pconf(\d+(?:p\d+)?)qabs\d+cand\d+(?:p\d+)?", mode)
    if not match:
        return None
    return float(match.group(1).replace("p", ".")) / 100.0

def parse_lag_refresh_interval(mode: str) -> int | None:
    match = re.fullmatch(r"lagrefresh(\d+)qabs\d+cand\d+(?:p\d+)?", mode)
    if not match:
        return None
    return max(1, int(match.group(1)))


def qdrift_mode(mode: str) -> bool:
    return re.fullmatch(r"qdrift(?:maj|head|share|cos|coshead)?qabs\d+cand\d+(?:p\d+)?", mode) is not None






def qdrift_cos_mode(mode: str) -> bool:
    return re.fullmatch(r"qdrift(?:cos|coshead)qabs\d+cand\d+(?:p\d+)?", mode) is not None


def qdrift_share_mode(mode: str) -> bool:
    return re.fullmatch(r"qdriftshareqabs\d+cand\d+(?:p\d+)?", mode) is not None


def qdrift_head_mode(mode: str) -> bool:
    return re.fullmatch(r"qdrift(?:head|coshead)qabs\d+cand\d+(?:p\d+)?", mode) is not None


def qdrift_majority_mode(mode: str) -> bool:
    return re.fullmatch(r"qdriftmajqabs\d+cand\d+(?:p\d+)?", mode) is not None


def qabs_reuse_uses_previous_final(mode: str) -> bool:
    return not mode.endswith("reuse2set")


def qabs_reuse_uses_previous_candidate(mode: str) -> bool:
    return not mode.endswith("reusefinal")


def parse_protect_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"top2limit(\d+)protects(\d+)r([0-9]+(?:p[0-9]+)?)", mode)
    if not match:
        return None
    sink_tokens = int(match.group(2))
    recent_percent = float(match.group(3).replace("p", "."))
    return sink_tokens, recent_percent / 100.0


def parse_hybrid_params(mode: str) -> dict[str, Any]:
    top2_full_match = re.fullmatch(
        r"obstop2fullnonmasst(\d+)(?:kn(\d+))?(?:mn(\d+))?",
        mode,
    )
    if top2_full_match:
        return {
            "top2_full_heads": True,
            "top2_use_mass": True,
            "top2_target": 1.0,
            "non_top2_target": int(top2_full_match.group(1)) / 100.0,
            "top2_max_heads": 16,
            "non_top2_max_heads": int(top2_full_match.group(2)) if top2_full_match.group(2) is not None else 2,
            "top2_min_heads": 16,
            "non_top2_min_heads": int(top2_full_match.group(3)) if top2_full_match.group(3) is not None else 1,
        }
    mass_match = re.fullmatch(
        r"obshybridmasst(\d+)n(\d+)(?:kt(\d+)kn(\d+))?(?:mt(\d+)mn(\d+))?",
        mode,
    )
    if mass_match:
        return {
            "top2_full_heads": False,
            "top2_use_mass": True,
            "top2_target": int(mass_match.group(1)) / 100.0,
            "non_top2_target": int(mass_match.group(2)) / 100.0,
            "top2_max_heads": int(mass_match.group(3)) if mass_match.group(3) is not None else 16,
            "non_top2_max_heads": int(mass_match.group(4)) if mass_match.group(4) is not None else 10,
            "top2_min_heads": int(mass_match.group(5)) if mass_match.group(5) is not None else 1,
            "non_top2_min_heads": int(mass_match.group(6)) if mass_match.group(6) is not None else 1,
        }
    match = re.fullmatch(r"obshybridt(\d+)n(\d+)(?:kt(\d+)kn(\d+))?", mode)
    if not match:
        return {
            "top2_full_heads": False,
            "top2_use_mass": False,
            "top2_target": 0.95,
            "non_top2_target": 0.80,
            "top2_max_heads": 16,
            "non_top2_max_heads": 6,
            "top2_min_heads": 1,
            "non_top2_min_heads": 1,
        }
    return {
        "top2_full_heads": False,
        "top2_use_mass": False,
        "top2_target": int(match.group(1)) / 100.0,
        "non_top2_target": int(match.group(2)) / 100.0,
        "top2_max_heads": int(match.group(3)) if match.group(3) is not None else 16,
        "non_top2_max_heads": int(match.group(4)) if match.group(4) is not None else 6,
        "top2_min_heads": 1,
        "non_top2_min_heads": 1,
    }


@dataclass
class LoadStats:
    layer_count: int
    head_count: int

    def __post_init__(self) -> None:
        shape = (self.layer_count, self.head_count)
        self.query_counts = torch.zeros(shape, dtype=torch.long)
        self.history_token_counts = torch.zeros(shape, dtype=torch.long)
        self.original_kept = torch.zeros(shape, dtype=torch.long)
        self.final_kept = torch.zeros(shape, dtype=torch.long)
        self.removed = torch.zeros(shape, dtype=torch.long)
        self.max_final_kept_per_query = torch.zeros(shape, dtype=torch.long)
        self.max_removed_per_query = torch.zeros(shape, dtype=torch.long)

    def update(
        self,
        layer: int,
        original_keep: torch.Tensor,
        final_keep: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        # Tensors have shape [batch, heads, key].
        original_counts = original_keep.sum(dim=(0, 2)).cpu().to(torch.long)
        final_counts = final_keep.sum(dim=(0, 2)).cpu().to(torch.long)
        removed_counts = (original_keep & ~final_keep).sum(dim=(0, 2)).cpu().to(torch.long)
        history_counts = history_valid.sum(dim=(0, 2)).cpu().to(torch.long)
        final_per_query = final_keep.sum(dim=2).cpu().to(torch.long)
        removed_per_query = (original_keep & ~final_keep).sum(dim=2).cpu().to(torch.long)
        batch_count = int(original_keep.shape[0])
        self.query_counts[layer] += batch_count
        self.history_token_counts[layer] += history_counts
        self.original_kept[layer] += original_counts
        self.final_kept[layer] += final_counts
        self.removed[layer] += removed_counts
        self.max_final_kept_per_query[layer] = torch.maximum(
            self.max_final_kept_per_query[layer],
            final_per_query.max(dim=0).values,
        )
        self.max_removed_per_query[layer] = torch.maximum(
            self.max_removed_per_query[layer],
            removed_per_query.max(dim=0).values,
        )

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                history_count = int(self.history_token_counts[layer, head])
                original = int(self.original_kept[layer, head])
                final = int(self.final_kept[layer, head])
                removed = int(self.removed[layer, head])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_count": query_count,
                        "history_token_cases": history_count,
                        "original_top2_kept": original,
                        "final_kept_after_limit3": final,
                        "removed_by_limit3": removed,
                        "final_kept_per_query_mean": final / query_count if query_count else 0.0,
                        "original_kept_per_query_mean": original / query_count if query_count else 0.0,
                        "removed_per_query_mean": removed / query_count if query_count else 0.0,
                        "kept_fraction_of_original_top2": final / original if original else 0.0,
                        "kept_fraction_of_history_cases": final / history_count if history_count else 0.0,
                        "max_final_kept_per_query": int(self.max_final_kept_per_query[layer, head]),
                        "max_removed_per_query": int(self.max_removed_per_query[layer, head]),
                    }
                )
        return rows


@dataclass
class CandidateStats:
    layer_count: int
    head_count: int

    def __post_init__(self) -> None:
        shape = (self.layer_count, self.head_count)
        self.query_counts = torch.zeros(shape, dtype=torch.long)
        self.history_token_counts = torch.zeros(shape, dtype=torch.long)
        self.candidate_counts = torch.zeros(shape, dtype=torch.long)
        self.max_candidate_per_query = torch.zeros(shape, dtype=torch.long)

    def update(self, layer: int, candidate_keep: torch.Tensor, history_valid: torch.Tensor) -> None:
        candidate_counts = candidate_keep.sum(dim=(0, 2)).cpu().to(torch.long)
        history_counts = history_valid.sum(dim=(0, 2)).cpu().to(torch.long)
        candidate_per_query = candidate_keep.sum(dim=2).cpu().to(torch.long)
        batch_count = int(candidate_keep.shape[0])
        self.query_counts[layer] += batch_count
        self.history_token_counts[layer] += history_counts
        self.candidate_counts[layer] += candidate_counts
        self.max_candidate_per_query[layer] = torch.maximum(
            self.max_candidate_per_query[layer],
            candidate_per_query.max(dim=0).values,
        )

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                history_count = int(self.history_token_counts[layer, head])
                candidate = int(self.candidate_counts[layer, head])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_count": query_count,
                        "history_token_cases": history_count,
                        "candidate_tokens": candidate,
                        "candidate_per_query_mean": candidate / query_count if query_count else 0.0,
                        "candidate_fraction_of_history": candidate / history_count if history_count else 0.0,
                        "max_candidate_per_query": int(self.max_candidate_per_query[layer, head]),
                    }
                )
        return rows


@dataclass
class ReuseOverlapStats:
    layer_count: int
    head_count: int

    def __post_init__(self) -> None:
        shape = (self.layer_count, self.head_count)
        self.query_counts = torch.zeros(shape, dtype=torch.long)
        self.history_token_counts = torch.zeros(shape, dtype=torch.long)
        self.current_raw = torch.zeros(shape, dtype=torch.long)
        self.previous_raw = torch.zeros(shape, dtype=torch.long)
        self.previous_final = torch.zeros(shape, dtype=torch.long)
        self.current_and_previous_raw = torch.zeros(shape, dtype=torch.long)
        self.current_and_previous_final = torch.zeros(shape, dtype=torch.long)
        self.previous_raw_and_previous_final = torch.zeros(shape, dtype=torch.long)
        self.all_three = torch.zeros(shape, dtype=torch.long)
        self.union_all = torch.zeros(shape, dtype=torch.long)
        self.union_current_previous_raw = torch.zeros(shape, dtype=torch.long)
        self.union_current_previous_final = torch.zeros(shape, dtype=torch.long)
        self.union_previous_raw_previous_final = torch.zeros(shape, dtype=torch.long)
        self.previous_final_only = torch.zeros(shape, dtype=torch.long)
        self.previous_raw_only = torch.zeros(shape, dtype=torch.long)
        self.current_raw_only = torch.zeros(shape, dtype=torch.long)
        self.max_union_all_per_query = torch.zeros(shape, dtype=torch.long)

    def update(
        self,
        layer: int,
        current_raw: torch.Tensor,
        previous_raw: torch.Tensor | None,
        previous_final: torch.Tensor | None,
    ) -> None:
        if previous_raw is None:
            previous_raw = torch.zeros_like(current_raw)
        if previous_final is None:
            previous_final = torch.zeros_like(current_raw)
        a = current_raw.bool()
        b = previous_raw.bool()
        c = previous_final.bool()
        ab = a & b
        ac = a & c
        bc = b & c
        abc = ab & c
        union_ab = a | b
        union_ac = a | c
        union_bc = b | c
        union_all = union_ab | c
        batch_count = int(a.shape[0])
        history_count = int(a.shape[-1])
        self.query_counts[layer] += batch_count
        self.history_token_counts[layer] += batch_count * history_count
        self.current_raw[layer] += a.sum(dim=(0, 2)).cpu().to(torch.long)
        self.previous_raw[layer] += b.sum(dim=(0, 2)).cpu().to(torch.long)
        self.previous_final[layer] += c.sum(dim=(0, 2)).cpu().to(torch.long)
        self.current_and_previous_raw[layer] += ab.sum(dim=(0, 2)).cpu().to(torch.long)
        self.current_and_previous_final[layer] += ac.sum(dim=(0, 2)).cpu().to(torch.long)
        self.previous_raw_and_previous_final[layer] += bc.sum(dim=(0, 2)).cpu().to(torch.long)
        self.all_three[layer] += abc.sum(dim=(0, 2)).cpu().to(torch.long)
        self.union_all[layer] += union_all.sum(dim=(0, 2)).cpu().to(torch.long)
        self.union_current_previous_raw[layer] += union_ab.sum(dim=(0, 2)).cpu().to(torch.long)
        self.union_current_previous_final[layer] += union_ac.sum(dim=(0, 2)).cpu().to(torch.long)
        self.union_previous_raw_previous_final[layer] += union_bc.sum(dim=(0, 2)).cpu().to(torch.long)
        self.previous_final_only[layer] += (c & ~union_ab).sum(dim=(0, 2)).cpu().to(torch.long)
        self.previous_raw_only[layer] += (b & ~union_ac).sum(dim=(0, 2)).cpu().to(torch.long)
        self.current_raw_only[layer] += (a & ~union_bc).sum(dim=(0, 2)).cpu().to(torch.long)
        self.max_union_all_per_query[layer] = torch.maximum(
            self.max_union_all_per_query[layer],
            union_all.sum(dim=2).cpu().to(torch.long).max(dim=0).values,
        )

    @staticmethod
    def _ratio(numer: int, denom: int) -> float:
        return numer / denom if denom else 0.0

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                history_count = int(self.history_token_counts[layer, head])
                a = int(self.current_raw[layer, head])
                b = int(self.previous_raw[layer, head])
                c = int(self.previous_final[layer, head])
                ab = int(self.current_and_previous_raw[layer, head])
                ac = int(self.current_and_previous_final[layer, head])
                bc = int(self.previous_raw_and_previous_final[layer, head])
                abc = int(self.all_three[layer, head])
                union_all = int(self.union_all[layer, head])
                union_ab = int(self.union_current_previous_raw[layer, head])
                union_ac = int(self.union_current_previous_final[layer, head])
                union_bc = int(self.union_previous_raw_previous_final[layer, head])
                c_only = int(self.previous_final_only[layer, head])
                b_only = int(self.previous_raw_only[layer, head])
                a_only = int(self.current_raw_only[layer, head])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_count": query_count,
                        "history_token_cases": history_count,
                        "current_raw_tokens": a,
                        "previous_raw_tokens": b,
                        "previous_final_tokens": c,
                        "current_and_previous_raw": ab,
                        "current_and_previous_final": ac,
                        "previous_raw_and_previous_final": bc,
                        "all_three": abc,
                        "union_all": union_all,
                        "union_current_previous_raw": union_ab,
                        "union_current_previous_final": union_ac,
                        "union_previous_raw_previous_final": union_bc,
                        "previous_final_only_vs_current_previous_raw": c_only,
                        "previous_raw_only_vs_current_previous_final": b_only,
                        "current_raw_only_vs_previous_raw_previous_final": a_only,
                        "current_raw_per_query_mean": a / query_count if query_count else 0.0,
                        "previous_raw_per_query_mean": b / query_count if query_count else 0.0,
                        "previous_final_per_query_mean": c / query_count if query_count else 0.0,
                        "union_all_per_query_mean": union_all / query_count if query_count else 0.0,
                        "union_ab_per_query_mean": union_ab / query_count if query_count else 0.0,
                        "previous_final_unique_per_query_mean": c_only / query_count if query_count else 0.0,
                        "previous_raw_unique_per_query_mean": b_only / query_count if query_count else 0.0,
                        "current_raw_unique_per_query_mean": a_only / query_count if query_count else 0.0,
                        "current_previous_raw_jaccard": self._ratio(ab, union_ab),
                        "current_previous_final_jaccard": self._ratio(ac, union_ac),
                        "previous_raw_previous_final_jaccard": self._ratio(bc, union_bc),
                        "all_three_fraction_of_union": self._ratio(abc, union_all),
                        "union_ab_fraction_of_union_all": self._ratio(union_ab, union_all),
                        "union_ac_fraction_of_union_all": self._ratio(union_ac, union_all),
                        "union_bc_fraction_of_union_all": self._ratio(union_bc, union_all),
                        "previous_final_unique_fraction_of_union_all": self._ratio(c_only, union_all),
                        "previous_raw_unique_fraction_of_union_all": self._ratio(b_only, union_all),
                        "current_raw_unique_fraction_of_union_all": self._ratio(a_only, union_all),
                        "previous_final_covered_by_current_previous_raw": 1.0 - self._ratio(c_only, c),
                        "previous_raw_covered_by_current_previous_final": 1.0 - self._ratio(b_only, b),
                        "current_raw_covered_by_previous_raw_previous_final": 1.0 - self._ratio(a_only, a),
                        "max_union_all_per_query": int(self.max_union_all_per_query[layer, head]),
                    }
                )
        return rows

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def build_row(name: str, layer: int | str, selector: Any) -> dict[str, Any]:
            query_count = int(self.query_counts[selector].sum())
            history_count = int(self.history_token_counts[selector].sum())
            a = int(self.current_raw[selector].sum())
            b = int(self.previous_raw[selector].sum())
            c = int(self.previous_final[selector].sum())
            ab = int(self.current_and_previous_raw[selector].sum())
            ac = int(self.current_and_previous_final[selector].sum())
            bc = int(self.previous_raw_and_previous_final[selector].sum())
            abc = int(self.all_three[selector].sum())
            union_all = int(self.union_all[selector].sum())
            union_ab = int(self.union_current_previous_raw[selector].sum())
            union_ac = int(self.union_current_previous_final[selector].sum())
            union_bc = int(self.union_previous_raw_previous_final[selector].sum())
            c_only = int(self.previous_final_only[selector].sum())
            b_only = int(self.previous_raw_only[selector].sum())
            a_only = int(self.current_raw_only[selector].sum())
            return {
                "summary": name,
                "layer": layer,
                "query_count": query_count,
                "history_token_cases": history_count,
                "current_raw_per_query_mean": a / query_count if query_count else 0.0,
                "previous_raw_per_query_mean": b / query_count if query_count else 0.0,
                "previous_final_per_query_mean": c / query_count if query_count else 0.0,
                "union_all_per_query_mean": union_all / query_count if query_count else 0.0,
                "union_ab_per_query_mean": union_ab / query_count if query_count else 0.0,
                "union_ac_per_query_mean": union_ac / query_count if query_count else 0.0,
                "union_bc_per_query_mean": union_bc / query_count if query_count else 0.0,
                "previous_final_unique_per_query_mean": c_only / query_count if query_count else 0.0,
                "previous_raw_unique_per_query_mean": b_only / query_count if query_count else 0.0,
                "current_raw_unique_per_query_mean": a_only / query_count if query_count else 0.0,
                "current_previous_raw_jaccard": self._ratio(ab, union_ab),
                "current_previous_final_jaccard": self._ratio(ac, union_ac),
                "previous_raw_previous_final_jaccard": self._ratio(bc, union_bc),
                "all_three_fraction_of_union": self._ratio(abc, union_all),
                "union_ab_fraction_of_union_all": self._ratio(union_ab, union_all),
                "union_ac_fraction_of_union_all": self._ratio(union_ac, union_all),
                "union_bc_fraction_of_union_all": self._ratio(union_bc, union_all),
                "previous_final_unique_fraction_of_union_all": self._ratio(c_only, union_all),
                "previous_raw_unique_fraction_of_union_all": self._ratio(b_only, union_all),
                "current_raw_unique_fraction_of_union_all": self._ratio(a_only, union_all),
                "previous_final_covered_by_current_previous_raw": 1.0 - self._ratio(c_only, c),
                "previous_raw_covered_by_current_previous_final": 1.0 - self._ratio(b_only, b),
                "current_raw_covered_by_previous_raw_previous_final": 1.0 - self._ratio(a_only, a),
            }

        rows.append(build_row("all_layers_all_heads", "all", (slice(None), slice(None))))
        for layer in range(self.layer_count):
            rows.append(build_row("layer_all_heads", layer, (layer, slice(None))))
        return rows


class QabsReuseProfileStats:
    STAGES = (
        "qdim_topk",
        "previous_state_load",
        "partial_scores",
        "candidate_select",
        "candidate_union",
        "reuse_select_kernel",
        "candidate_full_scores",
        "candidate_index_build",
        "final_topk",
        "final_mask_and_stats",
        "final_attention",
    )

    def __init__(self, layer_count: int, head_count: int, sync_cuda: bool = True) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.sync_cuda = sync_cuda
        self.stage_seconds: dict[tuple[int, str], float] = defaultdict(float)
        self.stage_calls: dict[tuple[int, str], int] = defaultdict(int)
        shape = (layer_count, head_count)
        self.query_counts = torch.zeros(shape, dtype=torch.long)
        self.history_token_counts = torch.zeros(shape, dtype=torch.long)
        self.current_raw_tokens = torch.zeros(shape, dtype=torch.long)
        self.union_tokens = torch.zeros(shape, dtype=torch.long)
        self.final_tokens = torch.zeros(shape, dtype=torch.long)
        self.max_current_raw_per_query = torch.zeros(shape, dtype=torch.long)
        self.max_union_per_query = torch.zeros(shape, dtype=torch.long)
        self.max_final_per_query = torch.zeros(shape, dtype=torch.long)
        self.threshold_sums = torch.zeros(shape, dtype=torch.float64)
        self.threshold_counts = torch.zeros(shape, dtype=torch.long)
        self.selection_counts: Counter[str] = Counter()

    def _sync(self, device: torch.device | None) -> None:
        if self.sync_cuda and device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def time_stage(self, layer: int, stage: str, device: torch.device | None):
        self._sync(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync(device)
            self.stage_seconds[(layer, stage)] += time.perf_counter() - start
            self.stage_calls[(layer, stage)] += 1

    def record_masks(
        self,
        layer: int,
        current_raw: torch.Tensor,
        union_mask: torch.Tensor,
        final_mask: torch.Tensor,
        threshold: torch.Tensor | None,
        selection: str,
    ) -> None:
        current_counts = current_raw.sum(dim=2).detach().cpu().to(torch.long)
        union_counts = union_mask.sum(dim=2).detach().cpu().to(torch.long)
        final_counts = final_mask.sum(dim=2).detach().cpu().to(torch.long)
        batch_count = int(current_raw.shape[0])
        history_count = int(current_raw.shape[-1])
        self.query_counts[layer] += batch_count
        self.history_token_counts[layer] += batch_count * history_count
        self.current_raw_tokens[layer] += current_counts.sum(dim=0)
        self.union_tokens[layer] += union_counts.sum(dim=0)
        self.final_tokens[layer] += final_counts.sum(dim=0)
        self.max_current_raw_per_query[layer] = torch.maximum(self.max_current_raw_per_query[layer], current_counts.max(dim=0).values)
        self.max_union_per_query[layer] = torch.maximum(self.max_union_per_query[layer], union_counts.max(dim=0).values)
        self.max_final_per_query[layer] = torch.maximum(self.max_final_per_query[layer], final_counts.max(dim=0).values)
        if threshold is not None:
            threshold_cpu = threshold.detach().float().cpu()
            if threshold_cpu.ndim == 3 and threshold_cpu.shape[-1] == 1:
                threshold_cpu = threshold_cpu.squeeze(-1)
            self.threshold_sums[layer] += threshold_cpu.double().sum(dim=0)
            self.threshold_counts[layer] += threshold_cpu.isfinite().sum(dim=0).to(torch.long)
        self.selection_counts[selection] += batch_count * int(current_raw.shape[1])
        if _ACTIVE_EVIDENCE_COVERAGE_STATS is not None:
            _ACTIVE_EVIDENCE_COVERAGE_STATS.record_masks(
                layer,
                current_raw,
                union_mask,
                final_mask,
                _ACTIVE_EVIDENCE_SPANS,
            )

    def stage_rows(self, mode: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            layer_total = sum(self.stage_seconds.get((layer, stage), 0.0) for stage in self.STAGES)
            for stage in self.STAGES:
                seconds = self.stage_seconds.get((layer, stage), 0.0)
                calls = self.stage_calls.get((layer, stage), 0)
                rows.append(
                    {
                        "mode": mode,
                        "layer": layer,
                        "stage": stage,
                        "seconds": seconds,
                        "calls": calls,
                        "seconds_per_call": seconds / calls if calls else 0.0,
                        "layer_profiled_seconds": layer_total,
                        "stage_fraction_of_layer": seconds / layer_total if layer_total else 0.0,
                    }
                )
        return rows

    def token_rows(self, mode: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                history_cases = int(self.history_token_counts[layer, head])
                current = int(self.current_raw_tokens[layer, head])
                union = int(self.union_tokens[layer, head])
                final = int(self.final_tokens[layer, head])
                threshold_count = int(self.threshold_counts[layer, head])
                rows.append(
                    {
                        "mode": mode,
                        "layer": layer,
                        "head": head,
                        "query_count": query_count,
                        "history_token_cases": history_cases,
                        "current_raw_tokens": current,
                        "union_tokens": union,
                        "final_tokens": final,
                        "current_raw_per_query_mean": current / query_count if query_count else 0.0,
                        "union_per_query_mean": union / query_count if query_count else 0.0,
                        "final_per_query_mean": final / query_count if query_count else 0.0,
                        "current_raw_fraction_of_history": current / history_cases if history_cases else 0.0,
                        "union_fraction_of_history": union / history_cases if history_cases else 0.0,
                        "final_fraction_of_history": final / history_cases if history_cases else 0.0,
                        "max_current_raw_per_query": int(self.max_current_raw_per_query[layer, head]),
                        "max_union_per_query": int(self.max_union_per_query[layer, head]),
                        "max_final_per_query": int(self.max_final_per_query[layer, head]),
                        "mean_candidate_threshold": float(self.threshold_sums[layer, head] / threshold_count) if threshold_count else "",
                    }
                )
        return rows

    def summary_rows(self, mode: str) -> list[dict[str, Any]]:
        total_seconds = sum(self.stage_seconds.values())
        rows = []
        for stage in self.STAGES:
            seconds = sum(self.stage_seconds.get((layer, stage), 0.0) for layer in range(self.layer_count))
            calls = sum(self.stage_calls.get((layer, stage), 0) for layer in range(self.layer_count))
            rows.append(
                {
                    "mode": mode,
                    "stage": stage,
                    "seconds": seconds,
                    "calls": calls,
                    "seconds_per_call": seconds / calls if calls else 0.0,
                    "fraction_of_profiled_seconds": seconds / total_seconds if total_seconds else 0.0,
                }
            )
        return rows


class EvidenceSpanCoverageStats:
    MASKS = ("current", "union", "final")
    METRICS = ("any", "all")

    def __init__(self, layer_count: int, head_count: int) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.query_counts = torch.zeros((layer_count, head_count), dtype=torch.long)
        self.counts: dict[tuple[str, str, str], torch.Tensor] = {}
        self.span_token_counts: Counter[str] = Counter()

    def _counter(self, mask_name: str, span_name: str, metric: str) -> torch.Tensor:
        key = (mask_name, span_name, metric)
        if key not in self.counts:
            self.counts[key] = torch.zeros((self.layer_count, self.head_count), dtype=torch.long)
        return self.counts[key]

    def record_masks(
        self,
        layer: int,
        current_raw: torch.Tensor,
        union_mask: torch.Tensor,
        final_mask: torch.Tensor,
        spans: dict[str, tuple[int, int]],
    ) -> None:
        if not spans:
            return
        masks = {"current": current_raw, "union": union_mask, "final": final_mask}
        batch_count, head_count, history_count = current_raw.shape
        self.query_counts[layer] += batch_count
        for span_name, (start, end) in spans.items():
            start = max(0, int(start))
            end = min(history_count, int(end))
            if start >= end:
                continue
            token_count = end - start
            self.span_token_counts[span_name] += batch_count * token_count
            for mask_name, mask in masks.items():
                span_mask = mask[:, :, start:end].detach().cpu().bool()
                any_hit = span_mask.any(dim=-1).to(torch.long).sum(dim=0)
                all_hit = span_mask.all(dim=-1).to(torch.long).sum(dim=0)
                self._counter(mask_name, span_name, "any")[layer] += any_hit
                self._counter(mask_name, span_name, "all")[layer] += all_hit

    def rows(self, task_id: int, variant: str, mode: str, correct: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                query_count = int(self.query_counts[layer, head])
                for (mask_name, span_name, metric), values in sorted(self.counts.items()):
                    hit_count = int(values[layer, head])
                    rows.append(
                        {
                            "task_id": task_id,
                            "variant": variant,
                            "mode": mode,
                            "correct": correct,
                            "layer": layer,
                            "head": head,
                            "mask": mask_name,
                            "span": span_name,
                            "metric": metric,
                            "hit_count": hit_count,
                            "query_count": query_count,
                            "coverage": hit_count / query_count if query_count else 0.0,
                        }
                    )
        return rows

    def overall_rows(self, task_id: int, variant: str, mode: str, correct: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total_queries = int(self.query_counts.sum().item())
        for (mask_name, span_name, metric), values in sorted(self.counts.items()):
            hit_count = int(values.sum().item())
            rows.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "mode": mode,
                    "correct": correct,
                    "mask": mask_name,
                    "span": span_name,
                    "metric": metric,
                    "hit_count": hit_count,
                    "query_count": total_queries,
                    "coverage": hit_count / total_queries if total_queries else 0.0,
                }
            )
        return rows


class ReuseCandidateState:
    def __init__(self) -> None:
        self.previous: dict[tuple[int, int], dict[str, Any]] = {}
        self.previous_layer: dict[int, dict[str, Any]] = {}
        self.refresh_head_count = 0
        self.refresh_case_count = 0

    def previous_masks(
        self,
        layer: int,
        head: int,
        query_token: int,
        history_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        previous = self.previous.get((layer, head))
        if previous is None or previous["query_token"] != query_token - 1:
            return None, None
        candidate = previous["candidate"].to(device)
        final = previous["final"].to(device)
        if candidate.numel() > history_count:
            candidate = candidate[:history_count]
        if final.numel() > history_count:
            final = final[:history_count]
        if candidate.numel() < history_count:
            candidate = F.pad(candidate, (0, history_count - candidate.numel()), value=False)
        if final.numel() < history_count:
            final = F.pad(final, (0, history_count - final.numel()), value=False)
        return candidate, final

    def update(
        self,
        layer: int,
        query_token: int,
        candidate_keep: torch.Tensor,
        final_keep: torch.Tensor,
        history_count: int,
    ) -> None:
        for head in range(candidate_keep.shape[1]):
            self.previous[(layer, head)] = {
                "query_token": query_token,
                "candidate": candidate_keep[0, head, :history_count].detach().cpu().bool(),
                "final": final_keep[0, head, :history_count].detach().cpu().bool(),
            }

    def previous_layer_masks(
        self,
        layer: int,
        query_token: int,
        history_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        previous = self.previous_layer.get(layer)
        if previous is None or previous["query_token"] != query_token - 1:
            return None, None
        candidate = previous["candidate"].to(device)
        final = previous["final"].to(device)
        if candidate.shape[-1] > history_count:
            candidate = candidate[..., :history_count]
        if final.shape[-1] > history_count:
            final = final[..., :history_count]
        if candidate.shape[-1] < history_count:
            candidate = F.pad(candidate, (0, history_count - candidate.shape[-1]), value=False)
        if final.shape[-1] < history_count:
            final = F.pad(final, (0, history_count - final.shape[-1]), value=False)
        return candidate, final

    def previous_layer_qdims(
        self,
        layer: int,
        query_token: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        previous = self.previous_layer.get(layer)
        if previous is None or previous["query_token"] != query_token - 1 or "qdim_indices" not in previous:
            return None
        return previous["qdim_indices"].to(device)

    def previous_layer_qvalues(
        self,
        layer: int,
        query_token: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        previous = self.previous_layer.get(layer)
        if previous is None or previous["query_token"] != query_token - 1 or "qdim_values" not in previous:
            return None
        return previous["qdim_values"].to(device)

    def previous_layer_thresholds(
        self,
        layer: int,
        query_token: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        previous = self.previous_layer.get(layer)
        if previous is None or previous["query_token"] != query_token - 1 or "candidate_thresholds" not in previous:
            return None
        return previous["candidate_thresholds"].to(device)

    def previous_layer_indices(
        self,
        layer: int,
        query_token: int,
        history_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        previous = self.previous_layer.get(layer)
        if previous is None or previous["query_token"] != query_token - 1 or "indices" not in previous:
            return None, None
        indices = previous["indices"].to(device)
        valid = previous["valid"].to(device)
        valid = valid & (indices >= 0) & (indices < history_count)
        return indices.clamp_min(0), valid

    def update_layer_indices(
        self,
        layer: int,
        query_token: int,
        indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self.previous_layer[layer] = {
            "query_token": query_token,
            "indices": indices.detach().long(),
            "valid": valid.detach().bool(),
        }

    def update_layer(
        self,
        layer: int,
        query_token: int,
        candidate_keep: torch.Tensor,
        final_keep: torch.Tensor,
        history_count: int,
        qdim_indices: torch.Tensor | None = None,
        qdim_values: torch.Tensor | None = None,
        candidate_thresholds: torch.Tensor | None = None,
    ) -> None:
        record = {
            "query_token": query_token,
            "candidate": candidate_keep[..., :history_count].detach().bool(),
            "final": final_keep[..., :history_count].detach().bool(),
        }
        if qdim_indices is not None:
            record["qdim_indices"] = qdim_indices.detach().cpu().long()
        if qdim_values is not None:
            record["qdim_values"] = qdim_values.detach().cpu().float()
        if candidate_thresholds is not None:
            record["candidate_thresholds"] = candidate_thresholds.detach().cpu().float()
        self.previous_layer[layer] = record

    def record_refresh_heads(self, refresh_heads: torch.Tensor | None, head_count: int) -> None:
        if refresh_heads is None:
            self.refresh_head_count += head_count
            self.refresh_case_count += head_count
            return
        self.refresh_head_count += int(refresh_heads.detach().sum().item())
        self.refresh_case_count += int(refresh_heads.numel())

    def refresh_fraction(self) -> float | None:
        if self.refresh_case_count <= 0:
            return None
        return self.refresh_head_count / self.refresh_case_count


class BlockRouteState:
    def __init__(self) -> None:
        self.previous_layer: dict[int, dict[str, Any]] = {}

    def summaries(self, layer: int, key_states: torch.Tensor, history_count: int, block_size: int) -> torch.Tensor:
        previous = self.previous_layer.get(layer)
        if history_count <= 0:
            return torch.empty((*key_states.shape[:2], 0, key_states.shape[-1]), device=key_states.device, dtype=torch.float32)
        if (
            previous is None
            or previous["block_size"] != block_size
            or previous["summaries"].shape[:2] != key_states.shape[:2]
            or previous["summaries"].device != key_states.device
            or previous["history_count"] > history_count
        ):
            history = key_states[:, :, :history_count, :].float()
            block_count = math.ceil(history_count / block_size)
            summaries = []
            counts = []
            for block_index in range(block_count):
                start = block_index * block_size
                end = min(history_count, start + block_size)
                block = history[:, :, start:end, :]
                summaries.append(block.mean(dim=2))
                counts.append(end - start)
            summary_tensor = torch.stack(summaries, dim=2)
            count_tensor = torch.tensor(counts, device=key_states.device, dtype=torch.long)
        else:
            summary_tensor = previous["summaries"].to(key_states.device)
            count_tensor = previous["counts"].to(key_states.device)
            previous_count = int(previous["history_count"])
            if previous_count == history_count - 1:
                new_key = key_states[:, :, history_count - 1, :].float()
                block_index = (history_count - 1) // block_size
                count_in_block = (history_count - 1) % block_size + 1
                if block_index == summary_tensor.shape[2]:
                    summary_tensor = torch.cat([summary_tensor, new_key[:, :, None, :]], dim=2)
                    count_tensor = torch.cat([count_tensor, torch.tensor([1], device=key_states.device, dtype=torch.long)])
                else:
                    old = summary_tensor[:, :, block_index, :]
                    summary_tensor[:, :, block_index, :] = (old * float(count_in_block - 1) + new_key) / float(count_in_block)
                    count_tensor[block_index] = count_in_block
            elif previous_count != history_count:
                history = key_states[:, :, :history_count, :].float()
                block_count = math.ceil(history_count / block_size)
                summaries = []
                counts = []
                for block_index in range(block_count):
                    start = block_index * block_size
                    end = min(history_count, start + block_size)
                    block = history[:, :, start:end, :]
                    summaries.append(block.mean(dim=2))
                    counts.append(end - start)
                summary_tensor = torch.stack(summaries, dim=2)
                count_tensor = torch.tensor(counts, device=key_states.device, dtype=torch.long)
        self.previous_layer[layer] = {
            "block_size": block_size,
            "history_count": history_count,
            "summaries": summary_tensor.detach(),
            "counts": count_tensor.detach(),
        }
        return summary_tensor


class KSignIndexState:
    def __init__(self) -> None:
        self.previous_layer: dict[int, dict[str, Any]] = {}

    def sign_index(self, layer: int, key_states: torch.Tensor, history_count: int) -> torch.Tensor:
        previous = self.previous_layer.get(layer)
        if history_count <= 0:
            return torch.empty((*key_states.shape[:2], key_states.shape[-1], 0), dtype=torch.bool, device=key_states.device)
        if (
            previous is None
            or previous["signs"].shape[:3] != (*key_states.shape[:2], key_states.shape[-1])
            or previous["signs"].device != key_states.device
            or previous["history_count"] > history_count
        ):
            signs = key_states[:, :, :history_count, :].float().ge(0).transpose(-1, -2).contiguous()
        else:
            signs = previous["signs"].to(key_states.device)
            previous_count = int(previous["history_count"])
            if previous_count == history_count - 1:
                new_sign = key_states[:, :, history_count - 1 : history_count, :].float().ge(0).transpose(-1, -2).contiguous()
                signs = torch.cat([signs, new_sign], dim=-1)
            elif previous_count != history_count:
                signs = key_states[:, :, :history_count, :].float().ge(0).transpose(-1, -2).contiguous()
        self.previous_layer[layer] = {"history_count": history_count, "signs": signs.detach()}
        return signs


class BlockRepState:
    def __init__(self) -> None:
        self.previous_layer: dict[tuple[int, int, int], dict[str, Any]] = {}

    def representatives(
        self,
        layer: int,
        key_states: torch.Tensor,
        history_count: int,
        block_size: int,
        rep_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, block_size, rep_count)
        batch_count, head_count, _, _ = key_states.shape
        if history_count <= 0:
            empty_idx = torch.zeros((batch_count, head_count, 0, 0), dtype=torch.long, device=key_states.device)
            empty_valid = torch.zeros_like(empty_idx, dtype=torch.bool)
            return empty_idx, empty_valid
        rep_count = max(1, min(rep_count, block_size))
        block_count = math.ceil(history_count / block_size)
        previous = self.previous_layer.get(key)
        can_reuse = (
            previous is not None
            and previous["indices"].shape[:2] == (batch_count, head_count)
            and previous["indices"].device == key_states.device
            and previous["history_count"] <= history_count
            and previous["indices"].shape[2] <= block_count
        )
        if can_reuse and int(previous["history_count"]) == history_count:
            return previous["indices"], previous["valid"]

        if can_reuse and int(previous["history_count"]) == history_count - 1:
            indices = previous["indices"].to(key_states.device)
            valid = previous["valid"].to(key_states.device)
            start_block = (history_count - 1) // block_size
            if start_block >= indices.shape[2]:
                pad_idx = torch.zeros((batch_count, head_count, 1, rep_count), dtype=torch.long, device=key_states.device)
                pad_valid = torch.zeros_like(pad_idx, dtype=torch.bool)
                indices = torch.cat([indices, pad_idx], dim=2)
                valid = torch.cat([valid, pad_valid], dim=2)
            blocks_to_update = [start_block]
        else:
            indices = torch.zeros((batch_count, head_count, block_count, rep_count), dtype=torch.long, device=key_states.device)
            valid = torch.zeros((batch_count, head_count, block_count, rep_count), dtype=torch.bool, device=key_states.device)
            blocks_to_update = list(range(block_count))

        for block_index in blocks_to_update:
            start = block_index * block_size
            end = min(history_count, start + block_size)
            if end <= start:
                continue
            block = key_states[:, :, start:end, :].float()
            norms = block.square().sum(dim=-1)
            k = min(rep_count, end - start)
            _, local_pos = torch.topk(norms, k=k, dim=-1, largest=True)
            abs_pos = local_pos + start
            if k < rep_count:
                abs_pos = F.pad(abs_pos, (0, rep_count - k), value=0)
                local_valid = torch.zeros((batch_count, head_count, rep_count), dtype=torch.bool, device=key_states.device)
                local_valid[:, :, :k] = True
            else:
                local_valid = torch.ones((batch_count, head_count, rep_count), dtype=torch.bool, device=key_states.device)
            indices[:, :, block_index, :] = abs_pos
            valid[:, :, block_index, :] = local_valid

        self.previous_layer[key] = {
            "history_count": history_count,
            "indices": indices.detach(),
            "valid": valid.detach(),
        }
        return indices, valid


class KDomIndexState:
    def __init__(self) -> None:
        self.previous_layer: dict[tuple[int, int], dict[str, Any]] = {}

    def dominant_index(self, layer: int, key_states: torch.Tensor, history_count: int, key_dim_count: int) -> torch.Tensor:
        previous_key = (layer, key_dim_count)
        previous = self.previous_layer.get(previous_key)
        batch_count, head_count, _, head_dim = key_states.shape
        if history_count <= 0:
            return torch.empty((batch_count, head_count, head_dim, 0), dtype=torch.bool, device=key_states.device)
        if (
            previous is None
            or previous["index"].shape[:3] != (batch_count, head_count, head_dim)
            or previous["index"].device != key_states.device
            or previous["history_count"] > history_count
        ):
            top_dims = torch.topk(
                key_states[:, :, :history_count, :].float().abs(),
                k=min(max(1, key_dim_count), head_dim),
                dim=-1,
                largest=True,
            ).indices
            index = torch.zeros((batch_count, head_count, head_dim, history_count), dtype=torch.bool, device=key_states.device)
            index.scatter_(dim=2, index=top_dims.permute(0, 1, 3, 2), src=torch.ones((batch_count, head_count, top_dims.shape[-1], history_count), dtype=torch.bool, device=key_states.device))
        else:
            index = previous["index"].to(key_states.device)
            previous_count = int(previous["history_count"])
            if previous_count == history_count - 1:
                new_top_dims = torch.topk(
                    key_states[:, :, history_count - 1 : history_count, :].float().abs(),
                    k=min(max(1, key_dim_count), head_dim),
                    dim=-1,
                    largest=True,
                ).indices
                new_index = torch.zeros((batch_count, head_count, head_dim, 1), dtype=torch.bool, device=key_states.device)
                new_index.scatter_(dim=2, index=new_top_dims.permute(0, 1, 3, 2), src=torch.ones((batch_count, head_count, new_top_dims.shape[-1], 1), dtype=torch.bool, device=key_states.device))
                index = torch.cat([index, new_index], dim=-1)
            elif previous_count != history_count:
                top_dims = torch.topk(
                    key_states[:, :, :history_count, :].float().abs(),
                    k=min(max(1, key_dim_count), head_dim),
                    dim=-1,
                    largest=True,
                ).indices
                index = torch.zeros((batch_count, head_count, head_dim, history_count), dtype=torch.bool, device=key_states.device)
                index.scatter_(dim=2, index=top_dims.permute(0, 1, 3, 2), src=torch.ones((batch_count, head_count, top_dims.shape[-1], history_count), dtype=torch.bool, device=key_states.device))
        self.previous_layer[previous_key] = {"history_count": history_count, "index": index.detach()}
        return index


class KNormState:
    def __init__(self) -> None:
        self.previous_layer: dict[int, dict[str, Any]] = {}

    def key_norms(self, layer: int, key_states: torch.Tensor) -> torch.Tensor:
        key_count = int(key_states.shape[-2])
        previous = self.previous_layer.get(layer)
        if (
            previous is None
            or previous["norms"].shape[:2] != key_states.shape[:2]
            or previous["norms"].device != key_states.device
            or previous["norms"].shape[-1] > key_count
        ):
            norms = key_states.float().square().sum(dim=-1)
        elif previous["norms"].shape[-1] == key_count - 1:
            last_norm = key_states[:, :, -1:, :].float().square().sum(dim=-1)
            norms = torch.cat([previous["norms"].to(key_states.device), last_norm], dim=-1)
        elif previous["norms"].shape[-1] == key_count:
            norms = previous["norms"].to(key_states.device)
        else:
            norms = key_states.float().square().sum(dim=-1)
        self.previous_layer[layer] = {"norms": norms.detach()}
        return norms


class SparQMeanState:
    def __init__(self) -> None:
        self.previous_layer: dict[int, dict[str, Any]] = {}
        self.key_dim_layer: dict[int, dict[str, Any]] = {}

    def value_mean(self, layer: int, value_states: torch.Tensor) -> torch.Tensor:
        key_count = int(value_states.shape[-2])
        previous = self.previous_layer.get(layer)
        if previous is not None and previous["key_count"] == key_count - 1:
            previous_mean = previous["mean"].to(value_states.device)
            new_value = value_states[:, :, -1:, :].float()
            mean = (previous_mean * float(key_count - 1) + new_value) / float(key_count)
        else:
            mean = value_states.float().mean(dim=2, keepdim=True)
        self.previous_layer[layer] = {"key_count": key_count, "mean": mean.detach()}
        return mean.to(value_states.dtype)

    def key_dim_major(self, layer: int, key_states: torch.Tensor) -> torch.Tensor:
        key_count = int(key_states.shape[-2])
        required_shape = (*key_states.shape[:2], key_states.shape[-1])
        previous = self.key_dim_layer.get(layer)
        if (
            previous is None
            or previous["cache"].shape[:3] != required_shape
            or previous["cache"].device != key_states.device
            or previous["cache"].dtype != key_states.dtype
            or previous["cache"].shape[-1] < key_count
        ):
            capacity = max(key_count, min(key_count * 2, key_count + 4096))
            cache = torch.empty((*required_shape, capacity), device=key_states.device, dtype=key_states.dtype)
            cache[..., :key_count] = key_states.transpose(-1, -2).contiguous()
            self.key_dim_layer[layer] = {"key_count": key_count, "cache": cache}
            return cache[..., :key_count]

        cache = previous["cache"]
        previous_count = int(previous["key_count"])
        if previous_count == key_count - 1:
            cache[..., previous_count:key_count] = key_states[:, :, -1:, :].transpose(-1, -2)
        elif previous_count != key_count:
            cache[..., :key_count] = key_states.transpose(-1, -2).contiguous()
        previous["key_count"] = key_count
        return cache[..., :key_count]


@dataclass
class WindowTokenStats:
    event_count: int = 0
    head_counts: list[int] = field(default_factory=list)
    head_set_counts: Counter[int] = field(default_factory=Counter)

    def ensure_heads(self, head_count: int) -> None:
        if not self.head_counts:
            self.head_counts = [0 for _ in range(head_count)]

    def add_event(self, selected_heads: list[int], head_count: int) -> None:
        if not selected_heads:
            return
        self.ensure_heads(head_count)
        self.event_count += 1
        mask = 0
        for head in selected_heads:
            self.head_counts[head] += 1
            mask |= 1 << head
        self.head_set_counts[mask] += 1

    def choose_heads(self, target_coverage: float, min_heads: int, max_heads: int) -> list[int]:
        self.ensure_heads(len(self.head_counts))
        if self.event_count <= 0:
            return []
        head_count = len(self.head_counts)
        ordered = sorted(range(head_count), key=lambda head: self.head_counts[head], reverse=True)
        max_heads = min(max_heads, head_count)
        min_heads = min(max(min_heads, 1), max_heads)
        chosen = ordered[:min_heads]
        for k in range(min_heads, max_heads + 1):
            chosen = ordered[:k]
            chosen_set = set(chosen)
            covered = 0
            for mask, count in self.head_set_counts.items():
                if any(mask & (1 << head) for head in chosen_set):
                    covered += count
            if covered / self.event_count >= target_coverage:
                break
        return chosen


@dataclass
class ObservationWindowState:
    layer_count: int
    head_count: int
    window_tokens: int
    recent_tokens: int
    target_coverage: float
    min_heads: int
    max_heads: int
    fallback_all: bool

    def __post_init__(self) -> None:
        self.window_stats: list[dict[int, WindowTokenStats]] = [
            defaultdict(WindowTokenStats) for _ in range(self.layer_count)
        ]
        self.assignments: list[dict[int, set[int]]] = [dict() for _ in range(self.layer_count)]
        self.last_boundary: list[int] = [-1 for _ in range(self.layer_count)]
        self.allocation_rows: list[dict[str, Any]] = []

    def maybe_reallocate(self, layer: int, query_token: int) -> None:
        if query_token <= 0 or query_token % self.window_tokens != 0:
            return
        if self.last_boundary[layer] == query_token:
            return
        stats = self.window_stats[layer]
        new_assignment: dict[int, set[int]] = {}
        assigned_counts: list[int] = []
        event_counts: list[int] = []
        for key_token, token_stats in stats.items():
            heads = token_stats.choose_heads(self.target_coverage, self.min_heads, self.max_heads)
            if heads:
                new_assignment[key_token] = set(heads)
                assigned_counts.append(len(heads))
                event_counts.append(token_stats.event_count)
        self.assignments[layer] = new_assignment
        self.window_stats[layer] = defaultdict(WindowTokenStats)
        self.last_boundary[layer] = query_token
        if assigned_counts:
            counts = Counter(assigned_counts)
            row = {
                "layer": layer,
                "boundary_query_token": query_token,
                "assigned_token_count": len(assigned_counts),
                "mean_assigned_heads": sum(assigned_counts) / len(assigned_counts),
                "mean_window_events_per_assigned_token": sum(event_counts) / len(event_counts),
                "min_assigned_heads": min(assigned_counts),
                "max_assigned_heads": max(assigned_counts),
            }
            for head_count in range(1, self.head_count + 1):
                row[f"tokens_assigned_{head_count}_heads"] = counts.get(head_count, 0)
        else:
            row = {
                "layer": layer,
                "boundary_query_token": query_token,
                "assigned_token_count": 0,
                "mean_assigned_heads": 0.0,
                "mean_window_events_per_assigned_token": 0.0,
                "min_assigned_heads": 0,
                "max_assigned_heads": 0,
            }
            for head_count in range(1, self.head_count + 1):
                row[f"tokens_assigned_{head_count}_heads"] = 0
        self.allocation_rows.append(row)

    def observe(self, layer: int, original_keep: torch.Tensor) -> None:
        # original_keep has shape [batch=1, heads, key].
        if original_keep.shape[0] != 1:
            raise ValueError("ObservationWindowState currently expects batch size 1.")
        selected_by_key: dict[int, list[int]] = defaultdict(list)
        selected = torch.nonzero(original_keep[0], as_tuple=False)
        for head, key_token in selected.tolist():
            selected_by_key[int(key_token)].append(int(head))
        for key_token, heads in selected_by_key.items():
            self.window_stats[layer][key_token].add_event(heads, self.head_count)

    def apply(self, layer: int, query_token: int, original_keep: torch.Tensor) -> torch.Tensor:
        if original_keep.shape[0] != 1:
            raise ValueError("ObservationWindowState currently expects batch size 1.")
        final_keep = original_keep.clone()
        history_count = max(0, query_token)
        recent_start = max(0, history_count - self.recent_tokens)
        selected = torch.nonzero(original_keep[0], as_tuple=False)
        for head, key_token in selected.tolist():
            key_token = int(key_token)
            head = int(head)
            if key_token >= recent_start:
                continue
            assigned = self.assignments[layer].get(key_token)
            if assigned is None:
                if self.fallback_all:
                    continue
                final_keep[0, head, key_token] = False
            elif head not in assigned:
                final_keep[0, head, key_token] = False
        return final_keep

    def rows(self) -> list[dict[str, Any]]:
        return self.allocation_rows


@dataclass
class AllTokenMassObservationState:
    layer_count: int
    head_count: int
    window_tokens: int
    recent_tokens: int
    target_coverage: float
    min_heads: int
    max_heads: int
    fallback_all: bool

    def __post_init__(self) -> None:
        self.mass: list[torch.Tensor] = [
            torch.zeros((self.head_count, 0), dtype=torch.float32) for _ in range(self.layer_count)
        ]
        self.assignments: list[dict[int, set[int]]] = [dict() for _ in range(self.layer_count)]
        self.last_boundary: list[int] = [-1 for _ in range(self.layer_count)]
        self.allocation_rows: list[dict[str, Any]] = []

    def ensure_tokens(self, layer: int, token_count: int) -> None:
        current = self.mass[layer].shape[1]
        if token_count <= current:
            return
        extra = torch.zeros((self.head_count, token_count - current), dtype=torch.float32)
        self.mass[layer] = torch.cat([self.mass[layer], extra], dim=1)

    def observe(self, layer: int, query_token: int, full_attention_weights: torch.Tensor) -> None:
        # full_attention_weights has shape [batch=1, heads, key].
        history_count = max(0, query_token)
        if history_count <= 0:
            return
        self.ensure_tokens(layer, history_count)
        weights = full_attention_weights[0, :, :history_count].detach().float().cpu()
        self.mass[layer][:, :history_count] += weights

    def maybe_reallocate(self, layer: int, query_token: int) -> None:
        if query_token <= 0 or query_token % self.window_tokens != 0:
            return
        if self.last_boundary[layer] == query_token:
            return
        history_count = max(0, query_token)
        assignable_count = max(0, history_count - self.recent_tokens)
        layer_mass = self.mass[layer]
        new_assignment: dict[int, set[int]] = {}
        assigned_counts: list[int] = []
        mass_values: list[float] = []
        for key_token in range(min(assignable_count, layer_mass.shape[1])):
            per_head = layer_mass[:, key_token]
            total = float(per_head.sum())
            if total <= 0.0:
                continue
            ordered = torch.argsort(per_head, descending=True).tolist()
            max_heads = min(self.max_heads, self.head_count)
            min_heads = min(max(self.min_heads, 1), max_heads)
            chosen = ordered[:min_heads]
            cumulative = float(per_head[chosen].sum())
            for k in range(min_heads, max_heads + 1):
                chosen = ordered[:k]
                cumulative = float(per_head[chosen].sum())
                if cumulative / total >= self.target_coverage:
                    break
            new_assignment[key_token] = set(int(head) for head in chosen)
            assigned_counts.append(len(chosen))
            mass_values.append(total)
        self.assignments[layer] = new_assignment
        self.mass[layer] = torch.zeros_like(layer_mass)
        self.last_boundary[layer] = query_token
        counts = Counter(assigned_counts)
        row = {
            "layer": layer,
            "boundary_query_token": query_token,
            "assigned_token_count": len(assigned_counts),
            "mean_assigned_heads": sum(assigned_counts) / len(assigned_counts) if assigned_counts else 0.0,
            "mean_window_attention_mass_per_assigned_token": sum(mass_values) / len(mass_values) if mass_values else 0.0,
            "min_assigned_heads": min(assigned_counts) if assigned_counts else 0,
            "max_assigned_heads": max(assigned_counts) if assigned_counts else 0,
        }
        for head_count in range(1, self.head_count + 1):
            row[f"tokens_assigned_{head_count}_heads"] = counts.get(head_count, 0)
        self.allocation_rows.append(row)

    def apply(self, layer: int, query_token: int, history_valid: torch.Tensor) -> torch.Tensor:
        # history_valid has shape [batch=1, heads, key].
        final_keep = history_valid.clone()
        history_count = max(0, query_token)
        recent_start = max(0, history_count - self.recent_tokens)
        selected = torch.nonzero(history_valid[0], as_tuple=False)
        for head, key_token in selected.tolist():
            key_token = int(key_token)
            head = int(head)
            if key_token >= recent_start:
                continue
            assigned = self.assignments[layer].get(key_token)
            if assigned is None:
                if self.fallback_all:
                    continue
                final_keep[0, head, key_token] = False
            elif head not in assigned:
                final_keep[0, head, key_token] = False
        return final_keep

    def rows(self) -> list[dict[str, Any]]:
        return self.allocation_rows


@dataclass
class HybridObservationState:
    layer_count: int
    head_count: int
    window_tokens: int
    recent_tokens: int
    top2_use_mass: bool
    top2_target_coverage: float
    non_top2_mass_coverage: float
    top2_min_heads: int
    top2_max_heads: int
    non_top2_min_heads: int
    non_top2_max_heads: int
    top2_full_heads: bool
    sink_tokens: int
    fallback_all: bool

    def __post_init__(self) -> None:
        self.mass: list[torch.Tensor] = [
            torch.zeros((self.head_count, 0), dtype=torch.float32) for _ in range(self.layer_count)
        ]
        self.top2_stats: list[dict[int, WindowTokenStats]] = [
            defaultdict(WindowTokenStats) for _ in range(self.layer_count)
        ]
        self.assignments: list[dict[int, set[int]]] = [dict() for _ in range(self.layer_count)]
        self.assignment_kind: list[dict[int, str]] = [dict() for _ in range(self.layer_count)]
        self.last_boundary: list[int] = [-1 for _ in range(self.layer_count)]
        self.allocation_rows: list[dict[str, Any]] = []

    def ensure_tokens(self, layer: int, token_count: int) -> None:
        current = self.mass[layer].shape[1]
        if token_count <= current:
            return
        extra = torch.zeros((self.head_count, token_count - current), dtype=torch.float32)
        self.mass[layer] = torch.cat([self.mass[layer], extra], dim=1)

    def observe(self, layer: int, query_token: int, full_attention_weights: torch.Tensor, top2_keep: torch.Tensor) -> None:
        history_count = max(0, query_token)
        if history_count <= 0:
            return
        self.ensure_tokens(layer, history_count)
        weights = full_attention_weights[0, :, :history_count].detach().float().cpu()
        self.mass[layer][:, :history_count] += weights

        selected_by_key: dict[int, list[int]] = defaultdict(list)
        selected = torch.nonzero(top2_keep[0], as_tuple=False)
        for head, key_token in selected.tolist():
            selected_by_key[int(key_token)].append(int(head))
        for key_token, heads in selected_by_key.items():
            self.top2_stats[layer][key_token].add_event(heads, self.head_count)

    def choose_mass_heads(self, per_head: torch.Tensor, target: float, min_heads: int, max_heads: int) -> list[int]:
        total = float(per_head.sum())
        if total <= 0.0:
            return []
        ordered = torch.argsort(per_head, descending=True).tolist()
        max_heads = min(max_heads, self.head_count)
        min_heads = min(max(min_heads, 1), max_heads)
        chosen = ordered[:min_heads]
        for k in range(min_heads, max_heads + 1):
            chosen = ordered[:k]
            if float(per_head[chosen].sum()) / total >= target:
                break
        return [int(head) for head in chosen]

    def maybe_reallocate(self, layer: int, query_token: int) -> None:
        if query_token <= 0 or query_token % self.window_tokens != 0:
            return
        if self.last_boundary[layer] == query_token:
            return
        history_count = max(0, query_token)
        assignable_count = max(0, history_count - self.recent_tokens)
        layer_mass = self.mass[layer]
        layer_top2 = self.top2_stats[layer]
        new_assignment: dict[int, set[int]] = {}
        new_kind: dict[int, str] = {}
        assigned_counts: list[int] = []
        top2_assigned = 0
        non_top2_assigned = 0
        for key_token in range(min(assignable_count, layer_mass.shape[1])):
            if key_token < self.sink_tokens:
                continue
            if key_token in layer_top2:
                if self.top2_full_heads:
                    heads = list(range(self.head_count))
                elif self.top2_use_mass:
                    heads = self.choose_mass_heads(
                        layer_mass[:, key_token],
                        self.top2_target_coverage,
                        self.top2_min_heads,
                        self.top2_max_heads,
                    )
                else:
                    heads = layer_top2[key_token].choose_heads(
                        self.top2_target_coverage,
                        self.top2_min_heads,
                        self.top2_max_heads,
                    )
                kind = "top2"
            else:
                heads = self.choose_mass_heads(
                    layer_mass[:, key_token],
                    self.non_top2_mass_coverage,
                    self.non_top2_min_heads,
                    self.non_top2_max_heads,
                )
                kind = "non_top2"
            if not heads:
                continue
            new_assignment[key_token] = set(heads)
            new_kind[key_token] = kind
            assigned_counts.append(len(heads))
            if kind == "top2":
                top2_assigned += 1
            else:
                non_top2_assigned += 1
        self.assignments[layer] = new_assignment
        self.assignment_kind[layer] = new_kind
        self.mass[layer] = torch.zeros_like(layer_mass)
        self.top2_stats[layer] = defaultdict(WindowTokenStats)
        self.last_boundary[layer] = query_token
        counts = Counter(assigned_counts)
        row = {
            "layer": layer,
            "boundary_query_token": query_token,
            "assigned_token_count": len(assigned_counts),
            "top2_assigned_token_count": top2_assigned,
            "non_top2_assigned_token_count": non_top2_assigned,
            "mean_assigned_heads": sum(assigned_counts) / len(assigned_counts) if assigned_counts else 0.0,
            "mean_window_events_per_assigned_token": 0.0,
            "mean_window_attention_mass_per_assigned_token": 0.0,
            "min_assigned_heads": min(assigned_counts) if assigned_counts else 0,
            "max_assigned_heads": max(assigned_counts) if assigned_counts else 0,
        }
        for head_count in range(1, self.head_count + 1):
            row[f"tokens_assigned_{head_count}_heads"] = counts.get(head_count, 0)
        self.allocation_rows.append(row)

    def apply(self, layer: int, query_token: int, history_valid: torch.Tensor) -> torch.Tensor:
        final_keep = history_valid.clone()
        history_count = max(0, query_token)
        recent_start = max(0, history_count - self.recent_tokens)
        selected = torch.nonzero(history_valid[0], as_tuple=False)
        for head, key_token in selected.tolist():
            key_token = int(key_token)
            head = int(head)
            if key_token < self.sink_tokens:
                continue
            if key_token >= recent_start:
                continue
            assigned = self.assignments[layer].get(key_token)
            if assigned is None:
                if self.fallback_all:
                    continue
                final_keep[0, head, key_token] = False
            elif head not in assigned:
                final_keep[0, head, key_token] = False
        return final_keep

    def rows(self) -> list[dict[str, Any]]:
        return self.allocation_rows


def _top2_history_keep_for_query(row_scores: torch.Tensor, finite: torch.Tensor, top_fraction: float) -> torch.Tensor:
    # row_scores and finite: [batch, heads, key].
    history_valid = finite.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return torch.zeros_like(finite, dtype=torch.bool)
    self_index = valid_count - 1
    history_valid[:, :, self_index] = False
    history_count = valid_count - 1
    keep_count = min(history_count, max(1, math.ceil(top_fraction * history_count)))
    history_scores = row_scores.masked_fill(~history_valid, torch.finfo(row_scores.dtype).min)
    _, top_indices = torch.topk(history_scores, k=keep_count, dim=-1, largest=True)
    keep = torch.zeros_like(finite, dtype=torch.bool)
    keep.scatter_(-1, top_indices, True)
    keep &= history_valid
    return keep


def _sign_xnor_history_keep_for_query(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    query_index: int,
    finite: torch.Tensor,
    candidate_fraction: float,
    weight_by_key_norm: bool = False,
) -> torch.Tensor:
    # Keep historical keys whose sign-match popcount is in the requested top
    # candidate bucket for each query head. Optionally weight the popcount by
    # the key vector norm. Boundary ties are kept, matching the recall-analysis
    # script.
    history_valid = finite.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return torch.zeros_like(finite, dtype=torch.bool)
    self_index = valid_count - 1
    history_valid[:, :, self_index] = False
    history_count = valid_count - 1
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))

    q_sign = torch.signbit(query_states[:, :, query_index, :])
    k_sign = torch.signbit(key_states[:, :, :history_count, :])
    candidate_scores = (k_sign == q_sign[:, :, None, :]).sum(dim=-1).float()
    if weight_by_key_norm:
        key_norm = key_states[:, :, :history_count, :].float().norm(dim=-1)
        candidate_scores = candidate_scores * key_norm
    threshold = torch.topk(candidate_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]

    keep = torch.zeros_like(finite, dtype=torch.bool)
    keep[:, :, :history_count] = candidate_scores >= threshold
    keep &= history_valid
    return keep


def _topk_within_candidate_for_query(
    row_scores: torch.Tensor,
    finite: torch.Tensor,
    candidate_keep: torch.Tensor,
    top_fraction: float,
) -> torch.Tensor:
    # Use exact QK score only inside a candidate mask, then keep the requested
    # top fraction of the full historical length for each head.
    history_valid = finite.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return torch.zeros_like(finite, dtype=torch.bool)
    self_index = valid_count - 1
    history_valid[:, :, self_index] = False
    history_count = valid_count - 1
    keep_count = min(history_count, max(1, math.ceil(top_fraction * history_count)))
    candidate_valid = candidate_keep & history_valid
    if int(candidate_valid.sum().item()) == 0:
        return torch.zeros_like(finite, dtype=torch.bool)
    candidate_scores = row_scores.masked_fill(~candidate_valid, torch.finfo(row_scores.dtype).min)
    _, top_indices = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    keep = torch.zeros_like(finite, dtype=torch.bool)
    keep.scatter_(-1, top_indices, True)
    keep &= candidate_valid
    return keep


def _recent_history_keep_for_query(
    finite: torch.Tensor,
    recent_tokens: int,
    sink_tokens: int,
) -> torch.Tensor:
    keep = torch.zeros_like(finite, dtype=torch.bool)
    return _apply_full_head_protection(keep, finite, sink_tokens, max(0, recent_tokens))


def _qabs_partial_candidate_keep_for_query(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    query_index: int,
    finite: torch.Tensor,
    dim_count: int,
    candidate_fraction: float,
) -> torch.Tensor:
    # Approximate QK by using only the query dimensions with largest absolute
    # value for the current query/head, then keep the top candidate bucket.
    history_valid = finite.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return torch.zeros_like(finite, dtype=torch.bool)
    self_index = valid_count - 1
    history_valid[:, :, self_index] = False
    history_count = valid_count - 1
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))

    q = query_states[:, :, query_index, :].float()
    k = key_states[:, :, :history_count, :].float()
    selected_dim_count = min(max(1, dim_count), q.shape[-1])
    dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=dim_indices)
    k_indices = dim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
    k_selected = torch.gather(k, dim=-1, index=k_indices)
    partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
    threshold = torch.topk(partial_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]

    keep = torch.zeros_like(finite, dtype=torch.bool)
    keep[:, :, :history_count] = partial_scores >= threshold
    keep &= history_valid
    return keep


def _sparq_attention_output_for_query(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_index: int,
    scaling: float,
    dim_count: int,
    candidate_fraction: float,
    sink_tokens: int,
    recent_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count, head_count, _, head_dim = query_states.shape
    key_count = key_states.shape[-2]
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, query_index, :].float()
    dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=dim_indices)
    k_indices = dim_indices[:, :, None, :].expand(-1, -1, key_count, -1)
    k_selected = torch.gather(key_states.float(), dim=-1, index=k_indices)
    q_abs_sum = q.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    q_selected_abs_sum = q_selected.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    sparq_scale = torch.sqrt(torch.tensor(float(head_dim), device=q.device) * q_selected_abs_sum / q_abs_sum)
    approx_scores = torch.matmul(q_selected[:, :, None, :], k_selected.transpose(2, 3)).squeeze(2) / sparq_scale
    if attention_mask is not None:
        mask_row = attention_mask[:, :, query_index, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        approx_scores = approx_scores + mask_row
    approx_weights = F.softmax(approx_scores, dim=-1, dtype=torch.float32)

    candidate_count = min(key_count, max(1, math.ceil(candidate_fraction * key_count)))
    _, top_indices = torch.topk(approx_weights, k=candidate_count, dim=-1, largest=True)
    selected_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    selected_keep.scatter_(dim=-1, index=top_indices, src=torch.ones_like(top_indices, dtype=torch.bool))
    if sink_tokens > 0 and key_count > 1:
        selected_keep[:, :, : min(sink_tokens, key_count - 1)] = True
    if recent_tokens > 0 and key_count > 1:
        history_count = key_count - 1
        recent_start = max(0, history_count - recent_tokens)
        selected_keep[:, :, recent_start:history_count] = True
    selected_keep[:, :, key_count - 1] = True

    selected_indices, selected_valid = _indices_from_keep_mask(selected_keep)
    gather_index = selected_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=gather_index)
    selected_values = torch.gather(value_states, dim=2, index=gather_index)
    selected_scores = (
        torch.matmul(query_states[:, :, query_index : query_index + 1, :], selected_keys.transpose(2, 3)).squeeze(2)
        * scaling
    )
    if attention_mask is not None:
        mask_row = attention_mask[:, :, query_index, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=selected_indices)
    selected_scores = selected_scores.masked_fill(~selected_valid, torch.finfo(selected_scores.dtype).min)
    selected_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    exact_selected_output = torch.sum(selected_weights[:, :, :, None] * selected_values, dim=2)
    alpha = torch.gather(approx_weights, dim=-1, index=selected_indices).masked_fill(~selected_valid, 0.0).sum(
        dim=-1, keepdim=True
    )
    value_mean = value_states.mean(dim=2)
    attention_output = alpha.to(query_states.dtype) * exact_selected_output + (1.0 - alpha).to(query_states.dtype) * value_mean
    return attention_output, selected_keep


def _sparq_fast_attention_output_for_query(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_index: int,
    scaling: float,
    dim_count: int,
    candidate_fraction: float,
) -> torch.Tensor:
    batch_count, head_count, _, head_dim = query_states.shape
    key_count = key_states.shape[-2]
    selected_dim_count = min(max(1, dim_count), head_dim)
    candidate_count = min(key_count, max(1, math.ceil(candidate_fraction * key_count)))

    q = query_states[:, :, query_index : query_index + 1, :]
    abs_q = q.float().abs()
    abs_q_hat, dim_indices = torch.topk(abs_q, k=selected_dim_count, dim=-1, largest=True, sorted=False)
    q_selected = torch.gather(q.float(), dim=-1, index=dim_indices)
    layer_idx = int(getattr(module, "layer_idx", 0))
    if _ACTIVE_SPARQ_MEAN_STATE is not None and sparq_fast_uses_double_k(_ACTIVE_MODE):
        key_dim_major = _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, key_states)
        k_dim_indices = dim_indices.transpose(-1, -2).expand(batch_count, head_count, selected_dim_count, key_count)
        k_selected = torch.gather(key_dim_major.float(), dim=2, index=k_dim_indices)
        approx_scores = torch.matmul(q_selected, k_selected)
    else:
        k_dim_indices = dim_indices.expand(batch_count, head_count, key_count, selected_dim_count)
        k_selected = torch.gather(key_states.float(), dim=-1, index=k_dim_indices)
        approx_scores = torch.matmul(q_selected, k_selected.transpose(2, 3))

    q_abs_sum = abs_q.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    q_selected_abs_sum = abs_q_hat.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    divscale = torch.sqrt(torch.tensor(float(head_dim), device=q.device) * q_selected_abs_sum / q_abs_sum)
    approx_scores = approx_scores / divscale
    if attention_mask is not None:
        mask_row = attention_mask[:, :, query_index : query_index + 1, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1, -1)
        approx_scores = approx_scores + mask_row
    approx_weights = F.softmax(approx_scores, dim=-1, dtype=torch.float32)

    approx_selected_mass, top_indices = torch.topk(
        approx_weights,
        k=candidate_count,
        dim=-1,
        largest=True,
        sorted=False,
    )
    gather_index = top_indices[..., None].expand(batch_count, head_count, 1, candidate_count, head_dim).squeeze(2)
    valid_indices = torch.ones_like(top_indices.squeeze(2), dtype=torch.bool)
    cuda_exact_output = _maybe_cuda_final_attention(
        q,
        key_states,
        value_states,
        attention_mask,
        top_indices.squeeze(2),
        valid_indices,
        scaling,
    )
    if cuda_exact_output is not None:
        exact_output = cuda_exact_output.transpose(1, 2)
    else:
        selected_keys = torch.gather(key_states, dim=2, index=gather_index)
        selected_values = torch.gather(value_states, dim=2, index=gather_index)
        exact_scores = torch.matmul(q, selected_keys.transpose(2, 3)) * scaling
        if attention_mask is not None:
            exact_scores = exact_scores + torch.gather(mask_row, dim=-1, index=top_indices)
        exact_weights = F.softmax(exact_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        exact_output = torch.matmul(exact_weights, selected_values)

    if _ACTIVE_SPARQ_MEAN_STATE is not None:
        value_mean = _ACTIVE_SPARQ_MEAN_STATE.value_mean(layer_idx, value_states)
    else:
        value_mean = value_states.float().mean(dim=2, keepdim=True).to(value_states.dtype)
    alpha = approx_selected_mass.sum(dim=-1, keepdim=True).to(q.dtype)
    return torch.lerp(value_mean, exact_output, alpha)


def _expand_selected_tokens_to_all_heads(original_keep: torch.Tensor) -> torch.Tensor:
    token_keep = original_keep.any(dim=1, keepdim=True)
    return token_keep.expand_as(original_keep).clone()


def _apply_full_head_protection(
    keep: torch.Tensor,
    finite: torch.Tensor,
    sink_tokens: int,
    recent_tokens: int,
) -> torch.Tensor:
    if sink_tokens <= 0 and recent_tokens <= 0:
        return keep
    final_keep = keep.clone()
    valid_count = int(finite[0, 0].sum().item())
    if valid_count <= 1:
        return final_keep
    history_count = valid_count - 1
    if sink_tokens > 0:
        sink_end = min(sink_tokens, history_count)
        if sink_end > 0:
            final_keep[:, :, :sink_end] = True
    if recent_tokens > 0:
        recent_start = max(0, history_count - recent_tokens)
        if recent_start < history_count:
            final_keep[:, :, recent_start:history_count] = True
    return final_keep


def _limit_heads_per_token_random(original_keep: torch.Tensor, max_heads: int) -> torch.Tensor:
    # original_keep: [batch, heads, key], only historical keys are true.
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    batch, _, key_count = original_keep.shape
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > max_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        if selected_heads.numel() <= max_heads:
            continue
        perm = torch.randperm(selected_heads.numel(), device=selected_heads.device)
        keep_heads = selected_heads[perm[:max_heads]]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _limit_heads_per_token_by_score(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
) -> torch.Tensor:
    # original_keep and row_scores: [batch, heads, key].
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > max_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        if selected_heads.numel() <= max_heads:
            continue
        selected_scores = row_scores[batch_index, selected_heads, key_index]
        _, order = torch.topk(selected_scores, k=max_heads, largest=True)
        keep_heads = selected_heads[order]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _limit_heads_per_token_by_score_fill(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
    history_valid: torch.Tensor,
) -> torch.Tensor:
    # First enforce the token cap by score, then fill each head back to its original top2 load.
    final_keep = _limit_heads_per_token_by_score(original_keep, row_scores, max_heads)
    batch, head_count, _ = original_keep.shape
    target_per_head = original_keep.sum(dim=2)
    token_counts = final_keep.sum(dim=1)
    fill_scores = row_scores.masked_fill(~history_valid, torch.finfo(row_scores.dtype).min)
    sorted_indices = torch.argsort(fill_scores, dim=-1, descending=True)
    for batch_index in range(batch):
        for head in range(head_count):
            target = int(target_per_head[batch_index, head].item())
            current = int(final_keep[batch_index, head].sum().item())
            if current >= target:
                continue
            for key_index in sorted_indices[batch_index, head].tolist():
                if current >= target:
                    break
                if not bool(history_valid[batch_index, head, key_index]):
                    continue
                if bool(final_keep[batch_index, head, key_index]):
                    continue
                if int(token_counts[batch_index, key_index].item()) >= max_heads:
                    continue
                final_keep[batch_index, head, key_index] = True
                token_counts[batch_index, key_index] += 1
                current += 1
    return final_keep


def _limit_heads_per_token_by_score_gap(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    min_heads: int,
    margin: float,
) -> torch.Tensor:
    # Keep at least min_heads by score. Also keep extra selected heads if their
    # score is within margin of the min_heads-th kept score for that token.
    if min_heads <= 0:
        return torch.zeros_like(original_keep)
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero(counts > min_heads, as_tuple=False)
    for batch_index, key_index in over.tolist():
        selected_heads = torch.nonzero(original_keep[batch_index, :, key_index], as_tuple=False).flatten()
        selected_scores = row_scores[batch_index, selected_heads, key_index]
        sorted_scores, order = torch.sort(selected_scores, descending=True)
        threshold = sorted_scores[min_heads - 1] - margin
        keep_heads = selected_heads[order[sorted_scores >= threshold]]
        final_keep[batch_index, :, key_index] = False
        final_keep[batch_index, keep_heads, key_index] = True
    return final_keep


def _protected_history_keys(
    finite: torch.Tensor,
    sink_tokens: int,
    recent_fraction: float,
) -> torch.Tensor:
    # Output shape: [batch, key]. True means the historical token should not be
    # limited across heads for this query.
    batch, _, key_count = finite.shape
    protected = torch.zeros((batch, key_count), dtype=torch.bool, device=finite.device)
    valid_count = int(finite[0, 0].sum().item())
    history_count = max(0, valid_count - 1)
    if history_count <= 0:
        return protected
    if sink_tokens > 0:
        protected[:, : min(sink_tokens, history_count)] = True
    if recent_fraction > 0:
        recent_count = min(history_count, max(1, math.ceil(recent_fraction * history_count)))
        protected[:, history_count - recent_count : history_count] = True
    return protected


def _limit_heads_per_token_by_score_protected(
    original_keep: torch.Tensor,
    row_scores: torch.Tensor,
    max_heads: int,
    protected_keys: torch.Tensor,
) -> torch.Tensor:
    # Protected historical tokens keep all original top2-selected heads. Other
    # historical tokens keep only the max_heads selected heads with largest score.
    if max_heads <= 0:
        return torch.zeros_like(original_keep)
    head_count = original_keep.shape[1]
    if max_heads >= head_count:
        return original_keep.clone()
    masked_scores = row_scores.masked_fill(~original_keep, torch.finfo(row_scores.dtype).min)
    _, keep_head_indices = torch.topk(masked_scores, k=max_heads, dim=1, largest=True)
    allowed_heads = torch.zeros_like(original_keep)
    allowed_heads.scatter_(dim=1, index=keep_head_indices, src=torch.ones_like(keep_head_indices, dtype=torch.bool))
    limited_keep = original_keep & allowed_heads
    return torch.where(protected_keys[:, None, :], original_keep, limited_keep)


def _indices_from_keep_mask(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert a dense boolean mask [batch, heads, key] into padded indices
    # [batch, heads, max_selected]. This is a PyTorch stand-in for the compact
    # candidate-index output a CUDA/Triton kernel should produce directly.
    selected_counts = keep.sum(dim=-1)
    max_selected = int(selected_counts.max().item()) if selected_counts.numel() else 0
    if max_selected <= 0:
        empty_indices = torch.zeros((*keep.shape[:-1], 0), dtype=torch.long, device=keep.device)
        empty_valid = torch.zeros_like(empty_indices, dtype=torch.bool)
        return empty_indices, empty_valid
    positions = torch.arange(keep.shape[-1], device=keep.device, dtype=torch.long).view(1, 1, -1)
    positions = positions.expand_as(keep)
    masked_positions = torch.where(keep, positions, torch.full_like(positions, -1))
    indices = torch.topk(masked_positions, k=max_selected, dim=-1, largest=True).values
    valid = indices >= 0
    return indices.clamp_min(0), valid


def _expand_history_keep_to_aligned_blocks(keep: torch.Tensor, block_size: int | None) -> torch.Tensor:
    if block_size is None or block_size <= 1 or keep.shape[-1] <= 1:
        return keep
    history_count = int(keep.shape[-1])
    pad_count = (-history_count) % int(block_size)
    padded = F.pad(keep, (0, pad_count), value=False) if pad_count else keep
    block_keep = padded.view(*padded.shape[:-1], -1, int(block_size)).any(dim=-1, keepdim=True)
    expanded = block_keep.expand(*block_keep.shape[:-1], int(block_size)).reshape(*padded.shape[:-1], padded.shape[-1])
    return expanded[..., :history_count]


def _force_evidence_spans_into_history_keep(keep: torch.Tensor, spans: dict[str, tuple[int, int]]) -> torch.Tensor:
    if not spans or keep.shape[-1] <= 0:
        return keep
    history_count = int(keep.shape[-1])
    forced = keep.clone()
    for start, end in spans.values():
        span_start = max(0, int(start))
        span_end = min(history_count, int(end))
        if span_start < span_end:
            forced[:, :, span_start:span_end] = True
    return forced


def _qabs_final_indices_from_selected(
    selected_history_indices: torch.Tensor,
    selected_valid: torch.Tensor,
    history_count: int,
    key_count: int,
    always_keep_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count, head_count, _ = selected_history_indices.shape
    filtered_valid = selected_valid.clone()
    sink_end = min(_ACTIVE_PROTECT_SINK_TOKENS, history_count) if _ACTIVE_PROTECT_SINK_TOKENS > 0 else 0
    recent_start = history_count
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        filtered_valid &= ~((selected_history_indices >= recent_start) & (selected_history_indices < history_count))
    if sink_end > 0:
        filtered_valid &= selected_history_indices >= sink_end

    index_parts = [selected_history_indices]
    valid_parts = [filtered_valid]
    if sink_end > 0:
        sink_indices = torch.arange(sink_end, device=selected_history_indices.device, dtype=torch.long).view(1, 1, -1)
        index_parts.append(sink_indices.expand(batch_count, head_count, -1))
        valid_parts.append(torch.ones((batch_count, head_count, sink_end), dtype=torch.bool, device=selected_history_indices.device))
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0 and recent_start < history_count:
        recent_start_no_overlap = max(recent_start, sink_end)
        if recent_start_no_overlap < history_count:
            recent_indices = torch.arange(
                recent_start_no_overlap,
                history_count,
                device=selected_history_indices.device,
                dtype=torch.long,
            ).view(1, 1, -1)
            index_parts.append(recent_indices.expand(batch_count, head_count, -1))
            valid_parts.append(
                torch.ones(
                    (batch_count, head_count, history_count - recent_start_no_overlap),
                    dtype=torch.bool,
                    device=selected_history_indices.device,
                )
            )
    if always_keep_self:
        self_indices = torch.full((batch_count, head_count, 1), key_count - 1, dtype=torch.long, device=selected_history_indices.device)
        index_parts.append(self_indices)
        valid_parts.append(torch.ones_like(self_indices, dtype=torch.bool))
    return torch.cat(index_parts, dim=-1), torch.cat(valid_parts, dim=-1)


def _qabs_final_indices_append_protected(
    selected_history_indices: torch.Tensor,
    selected_valid: torch.Tensor,
    history_count: int,
    key_count: int,
    always_keep_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count, head_count, _ = selected_history_indices.shape
    index_parts = [selected_history_indices]
    valid_parts = [selected_valid]
    sink_end = min(_ACTIVE_PROTECT_SINK_TOKENS, history_count) if _ACTIVE_PROTECT_SINK_TOKENS > 0 else 0
    if sink_end > 0:
        sink_indices = torch.arange(sink_end, device=selected_history_indices.device, dtype=torch.long).view(1, 1, -1)
        index_parts.append(sink_indices.expand(batch_count, head_count, -1))
        valid_parts.append(torch.ones((batch_count, head_count, sink_end), dtype=torch.bool, device=selected_history_indices.device))
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        recent_start = max(recent_start, sink_end)
        if recent_start < history_count:
            recent_indices = torch.arange(
                recent_start,
                history_count,
                device=selected_history_indices.device,
                dtype=torch.long,
            ).view(1, 1, -1)
            index_parts.append(recent_indices.expand(batch_count, head_count, -1))
            valid_parts.append(
                torch.ones(
                    (batch_count, head_count, history_count - recent_start),
                    dtype=torch.bool,
                    device=selected_history_indices.device,
                )
            )
    if always_keep_self:
        self_indices = torch.full((batch_count, head_count, 1), key_count - 1, dtype=torch.long, device=selected_history_indices.device)
        index_parts.append(self_indices)
        valid_parts.append(torch.ones_like(self_indices, dtype=torch.bool))
    return torch.cat(index_parts, dim=-1), torch.cat(valid_parts, dim=-1)


def _recent_window_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    recent_tokens: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, _ = query_states.shape
    key_count = key_states.shape[-2]
    if query_count != 1:
        raise RuntimeError("recent-window fast path requires token-by-token eval; set --eval_chunk_size 1.")
    history_count = max(0, key_count - 1)
    sink_end = min(max(0, sink_tokens), history_count)
    recent_start = max(0, history_count - max(0, recent_tokens))
    recent_start = max(recent_start, sink_end)

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    if sink_end > 0:
        key_parts.append(key_states[:, :, :sink_end, :])
        value_parts.append(value_states[:, :, :sink_end, :])
        if attention_mask is not None:
            mask_parts.append(attention_mask[:, :, :, :sink_end])
    if recent_start < key_count:
        key_parts.append(key_states[:, :, recent_start:key_count, :])
        value_parts.append(value_states[:, :, recent_start:key_count, :])
        if attention_mask is not None:
            mask_parts.append(attention_mask[:, :, :, recent_start:key_count])
    if not key_parts:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    selected_keys = torch.cat(key_parts, dim=2) if len(key_parts) > 1 else key_parts[0]
    selected_values = torch.cat(value_parts, dim=2) if len(value_parts) > 1 else value_parts[0]
    scores = torch.matmul(query_states, selected_keys.transpose(2, 3)) * scaling
    if attention_mask is not None and mask_parts:
        selected_mask = torch.cat(mask_parts, dim=-1) if len(mask_parts) > 1 else mask_parts[0]
        if selected_mask.shape[1] == 1 and head_count != 1:
            selected_mask = selected_mask.expand(batch_count, head_count, -1, -1)
        scores = scores + selected_mask
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, selected_values)
    return attention_output.transpose(1, 2).contiguous(), None


def _landmark_recent_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    recent_tokens: int,
    landmark_stride: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, _ = query_states.shape
    key_count = key_states.shape[-2]
    if query_count != 1:
        raise RuntimeError("landmark-recent attention requires token-by-token eval; set --eval_chunk_size 1.")
    history_count = max(0, key_count - 1)
    if history_count <= 0:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    sink_end = min(max(0, sink_tokens), history_count)
    recent_start = max(0, history_count - max(0, recent_tokens))
    remote_end = max(sink_end, recent_start)
    index_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []

    if sink_end > 0:
        sink_indices = torch.arange(sink_end, device=query_states.device, dtype=torch.long).view(1, 1, -1)
        index_parts.append(sink_indices.expand(batch_count, head_count, -1))
        valid_parts.append(torch.ones((batch_count, head_count, sink_end), device=query_states.device, dtype=torch.bool))

    if remote_end > sink_end:
        stride = max(1, int(landmark_stride))
        max_landmarks = max(1, math.ceil(remote_end / stride) + 1)
        base = torch.arange(max_landmarks, device=query_states.device, dtype=torch.long).view(1, 1, -1) * stride
        if head_count > 1 and stride > 1:
            offsets = torch.div(
                torch.arange(head_count, device=query_states.device, dtype=torch.long) * stride,
                head_count,
                rounding_mode="floor",
            ).view(1, head_count, 1)
        else:
            offsets = torch.zeros((1, head_count, 1), device=query_states.device, dtype=torch.long)
        landmark_indices = base + offsets
        landmark_valid = (landmark_indices >= sink_end) & (landmark_indices < remote_end)
        index_parts.append(landmark_indices.expand(batch_count, -1, -1))
        valid_parts.append(landmark_valid.expand(batch_count, -1, -1))

    recent_start = max(recent_start, sink_end)
    if recent_start < history_count:
        recent_indices = torch.arange(recent_start, history_count, device=query_states.device, dtype=torch.long).view(1, 1, -1)
        index_parts.append(recent_indices.expand(batch_count, head_count, -1))
        valid_parts.append(
            torch.ones((batch_count, head_count, history_count - recent_start), device=query_states.device, dtype=torch.bool)
        )

    if _ACTIVE_ALWAYS_KEEP_SELF:
        self_indices = torch.full((batch_count, head_count, 1), key_count - 1, device=query_states.device, dtype=torch.long)
        index_parts.append(self_indices)
        valid_parts.append(torch.ones_like(self_indices, dtype=torch.bool))

    final_indices = torch.cat(index_parts, dim=-1)
    final_valid = torch.cat(valid_parts, dim=-1)
    attention_output = _qabs_attention_from_final_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    return attention_output, None


def _full_head_recent_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    full_heads: int,
    recent_tokens: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, _ = query_states.shape
    key_count = key_states.shape[-2]
    if query_count != 1:
        raise RuntimeError("full-head-recent attention requires token-by-token eval; set --eval_chunk_size 1.")
    layer_idx = int(getattr(module, "layer_idx", 0))
    full_head_indices = full_head_indices_for_layer(layer_idx, full_heads, head_count, query_states.device)
    full_head_count = int(full_head_indices.numel())
    full_head_mask = torch.zeros((head_count,), dtype=torch.bool, device=query_states.device)
    if full_head_count > 0:
        full_head_mask.scatter_(0, full_head_indices, True)
    recent_head_indices = torch.arange(head_count, device=query_states.device, dtype=torch.long)[~full_head_mask]
    output = torch.empty_like(query_states)

    if full_head_count > 0:
        full_query = query_states.index_select(dim=1, index=full_head_indices)
        full_key = key_states.index_select(dim=1, index=full_head_indices)
        full_value = value_states.index_select(dim=1, index=full_head_indices)
        full_scores = torch.matmul(full_query, full_key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            full_mask = attention_mask[:, :, :, :key_count]
            if full_mask.shape[1] == 1 and full_head_count != 1:
                full_mask = full_mask.expand(batch_count, full_head_count, -1, -1)
            else:
                full_mask = full_mask.index_select(dim=1, index=full_head_indices)
            full_scores = full_scores + full_mask
        full_weights = F.softmax(full_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        full_output = torch.matmul(full_weights, full_value)
        output.index_copy_(1, full_head_indices, full_output)

    if recent_head_indices.numel() > 0:
        recent_query = query_states.index_select(dim=1, index=recent_head_indices)
        recent_key = key_states.index_select(dim=1, index=recent_head_indices)
        recent_value = value_states.index_select(dim=1, index=recent_head_indices)
        history_count = max(0, key_count - 1)
        sink_end = min(max(0, sink_tokens), history_count)
        recent_start = max(0, history_count - max(0, recent_tokens))
        recent_start = max(recent_start, sink_end)
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        if sink_end > 0:
            key_parts.append(recent_key[:, :, :sink_end, :])
            value_parts.append(recent_value[:, :, :sink_end, :])
            if attention_mask is not None:
                mask_parts.append(attention_mask[:, :, :, :sink_end])
        if recent_start < key_count:
            key_parts.append(recent_key[:, :, recent_start:key_count, :])
            value_parts.append(recent_value[:, :, recent_start:key_count, :])
            if attention_mask is not None:
                mask_parts.append(attention_mask[:, :, :, recent_start:key_count])
        if not key_parts:
            recent_output = recent_value[:, :, -1:, :]
        else:
            selected_keys = torch.cat(key_parts, dim=2) if len(key_parts) > 1 else key_parts[0]
            selected_values = torch.cat(value_parts, dim=2) if len(value_parts) > 1 else value_parts[0]
            recent_scores = torch.matmul(recent_query, selected_keys.transpose(2, 3)) * scaling
            if attention_mask is not None and mask_parts:
                selected_mask = torch.cat(mask_parts, dim=-1) if len(mask_parts) > 1 else mask_parts[0]
                recent_head_count = head_count - full_head_count
                if selected_mask.shape[1] == 1 and recent_head_count != 1:
                    selected_mask = selected_mask.expand(batch_count, recent_head_count, -1, -1)
                else:
                    selected_mask = selected_mask.index_select(dim=1, index=recent_head_indices)
                recent_scores = recent_scores + selected_mask
            recent_weights = F.softmax(recent_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            recent_output = torch.matmul(recent_weights, selected_values)
        output.index_copy_(1, recent_head_indices, recent_output)

    return output.transpose(1, 2).contiguous(), None


def _full_layer_recent_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    full_layers: int,
    recent_tokens: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    layer_idx = int(getattr(module, "layer_idx", 0))
    if layer_idx not in full_layer_indices(full_layers):
        return _recent_window_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            recent_tokens,
            sink_tokens,
        )
    key_count = key_states.shape[-2]
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key_count]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), None


def _full_layer_landmark_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    full_layers: int,
    recent_tokens: int,
    landmark_stride: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    layer_idx = int(getattr(module, "layer_idx", 0))
    if layer_idx not in full_layer_indices(full_layers):
        return _landmark_recent_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            recent_tokens,
            landmark_stride,
            sink_tokens,
        )
    key_count = key_states.shape[-2]
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key_count]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), None


def _dense_decode_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    key_count = key_states.shape[-2]
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key_count]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attention_weights, value_states)


def _pad_previous_history_mask(mask: torch.Tensor | None, history_count: int) -> torch.Tensor | None:
    if mask is None:
        return None
    previous_count = int(mask.shape[-1])
    if previous_count == history_count:
        return mask
    if previous_count == history_count - 1:
        pad = torch.zeros((*mask.shape[:-1], 1), dtype=mask.dtype, device=mask.device)
        return torch.cat([mask, pad], dim=-1)
    return None


def _layer_budget_qabs_reuse_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dim_count: int,
    candidate_fraction: float,
    top_fraction: float,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("layer-budget qabs-reuse requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    history_count = key_count - 1
    if history_count <= 0:
        return value_states[:, :, -1:, :].transpose(1, 2).contiguous(), None

    layer_idx = int(getattr(module, "layer_idx", 0))
    selected_dim_count = min(max(1, int(dim_count)), head_dim)
    candidate_fraction = max(0.0, min(1.0, float(candidate_fraction)))
    top_fraction = max(0.0, min(1.0, float(top_fraction)))
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    final_requested = min(history_count, max(1, math.ceil(top_fraction * history_count)))

    q = query_states[:, :, 0, :].float()
    k_history = key_states[:, :, :history_count, :]
    dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=dim_indices)
    k_dim_indices = dim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
    k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
    partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
    threshold = torch.topk(partial_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]
    current_candidate = partial_scores >= threshold

    previous = _LAYER_BUDGET_QABS_REUSE_STATE.get(layer_idx, {})
    previous_candidate = _pad_previous_history_mask(previous.get("candidate"), history_count)
    previous_final = _pad_previous_history_mask(previous.get("final"), history_count)
    candidate_union = current_candidate.clone()
    if previous_candidate is not None:
        candidate_union |= previous_candidate.to(candidate_union.device)
    if previous_final is not None:
        candidate_union |= previous_final.to(candidate_union.device)

    candidate_indices, candidate_valid = _indices_from_keep_mask(candidate_union)
    if candidate_indices.shape[-1] == 0:
        candidate_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
        candidate_valid = torch.zeros_like(candidate_indices, dtype=torch.bool)
    candidate_gather = candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        candidate_scores = candidate_scores + torch.gather(mask_row[:, :, :history_count], dim=-1, index=candidate_indices)
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    selected_positions = torch.topk(candidate_scores, k=min(final_requested, candidate_scores.shape[-1]), dim=-1, largest=True).indices
    selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_positions)

    history_final = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
    history_final.scatter_(dim=-1, index=selected_history_indices.clamp_max(history_count - 1), src=selected_valid)
    _LAYER_BUDGET_QABS_REUSE_STATE[layer_idx] = {
        "candidate": current_candidate.detach(),
        "final": history_final.detach(),
    }

    final_indices, final_valid = _qabs_final_indices_from_selected(
        selected_history_indices,
        selected_valid,
        history_count,
        key_count,
        _ACTIVE_ALWAYS_KEEP_SELF,
    )
    attention_output = _qabs_attention_from_final_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    return attention_output, None


def _layer_budget_headmix_qabs_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    full_heads: int,
    dim_count: int,
    candidate_fraction: float,
    top_fraction: float,
) -> tuple[torch.Tensor, None]:
    qabs_output, _ = _layer_budget_qabs_reuse_attention_forward(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
        dim_count,
        candidate_fraction,
        top_fraction,
    )
    head_count = query_states.shape[1]
    full_head_indices = full_head_indices_for_layer(
        int(getattr(module, "layer_idx", 0)),
        full_heads,
        head_count,
        query_states.device,
    )
    if full_head_indices.numel() <= 0:
        return qabs_output, None
    output = qabs_output.transpose(1, 2).contiguous()
    full_output = _dense_decode_attention_forward(
        query_states.index_select(dim=1, index=full_head_indices),
        key_states.index_select(dim=1, index=full_head_indices),
        value_states.index_select(dim=1, index=full_head_indices),
        attention_mask,
        scaling,
    )
    output.index_copy_(1, full_head_indices, full_output)
    return output.transpose(1, 2).contiguous(), None


def _layer_budget_attention_forward_with_budget(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    sink_tokens: int,
    budget: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    layer_idx = int(getattr(module, "layer_idx", 0))
    budget_type = str(budget.get("type", budget.get("kind", "full"))).lower()
    if budget_type == "full":
        key_count = key_states.shape[-2]
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key_count]
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.matmul(attention_weights, value_states)
        return attention_output.transpose(1, 2).contiguous(), None
    if budget_type == "recent":
        return _recent_window_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            max(0, int(budget.get("recent", budget.get("recent_tokens", 512)))),
            max(0, int(budget.get("sink", sink_tokens))),
        )
    if budget_type == "landmark":
        return _landmark_recent_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            max(0, int(budget.get("recent", budget.get("recent_tokens", 512)))),
            max(1, int(budget.get("stride", budget.get("landmark_stride", 64)))),
            max(0, int(budget.get("sink", sink_tokens))),
        )
    if budget_type in {"qabs", "qabs_reuse", "qabs8cand3", "qabs8cand3reuse"}:
        return _layer_budget_qabs_reuse_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            max(1, int(budget.get("dims", budget.get("dim_count", 8)))),
            float(budget.get("candidate_fraction", budget.get("cand", 0.03))),
            float(budget.get("top_fraction", budget.get("top", _ACTIVE_TOP_FRACTION))),
        )
    if budget_type in {"headmix_qabs", "headmix_qabs_reuse", "hybrid_qabs", "hybrid_qabs_reuse"}:
        return _layer_budget_headmix_qabs_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            max(0, int(budget.get("full_heads", budget.get("full", 0)))),
            max(1, int(budget.get("dims", budget.get("dim_count", 8)))),
            float(budget.get("candidate_fraction", budget.get("cand", 0.03))),
            float(budget.get("top_fraction", budget.get("top", _ACTIVE_TOP_FRACTION))),
        )
    if budget_type in {"head_recent", "full_head_recent", "headfull_recent"}:
        return _full_head_recent_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            max(0, int(budget.get("full_heads", budget.get("full", 0)))),
            max(0, int(budget.get("recent", budget.get("recent_tokens", 512)))),
            max(0, int(budget.get("sink", sink_tokens))),
        )
    if budget_type in {"synthetic", "synth", "synthetic_kv", "synthkv"}:
        return _synthetic_kv_attention_forward(
            query_states,
            key_states,
            value_states,
            scaling,
            max(1, int(budget.get("prototypes", budget.get("prototype_count", 16)))),
            str(budget.get("method", "mass")).lower(),
            max(0, int(budget.get("sink", sink_tokens))),
            max(0, int(budget.get("recent", budget.get("recent_tokens", 512)))),
        )
    if budget_type in {"landmark_synthetic", "hybrid_synthetic", "hybrid_skv", "pcic_skv"}:
        return _landmark_synthetic_kv_attention_forward(
            query_states,
            key_states,
            value_states,
            scaling,
            max(0, int(budget.get("recent", budget.get("recent_tokens", 512)))),
            max(1, int(budget.get("stride", budget.get("landmark_stride", 64)))),
            max(1, int(budget.get("prototypes", budget.get("prototype_count", 16)))),
            str(budget.get("method", "mean")).lower(),
            max(0, int(budget.get("sink", sink_tokens))),
        )
    raise ValueError(f"Unsupported layer budget type for layer {layer_idx}: {budget_type}")


def _is_mixed_dense_landmark_budget(budget: dict[str, Any]) -> bool:
    budget_type = str(budget.get("type", budget.get("kind", "full"))).lower()
    return budget_type in {"full", "landmark"}


def _mixed_full_landmark_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    sink_tokens: int,
    batch_budgets: list[dict[str, Any]],
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, _ = query_states.shape
    if query_count != 1:
        raise RuntimeError("mixed full/landmark fast path requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    history_count = max(0, key_count - 1)
    if history_count <= 0:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    keep = torch.ones((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    position = torch.arange(key_count, device=query_states.device, dtype=torch.long).view(1, 1, -1)
    head_indices = torch.arange(head_count, device=query_states.device, dtype=torch.long).view(1, head_count, 1)
    for row_idx, budget in enumerate(batch_budgets):
        budget_type = str(budget.get("type", budget.get("kind", "full"))).lower()
        if budget_type == "full":
            continue
        recent_tokens = max(0, int(budget.get("recent", budget.get("recent_tokens", 512))))
        stride = max(1, int(budget.get("stride", budget.get("landmark_stride", 64))))
        row_sink_tokens = max(0, int(budget.get("sink", sink_tokens)))
        sink_end = min(row_sink_tokens, history_count)
        recent_start = max(0, history_count - recent_tokens)
        remote_end = max(sink_end, recent_start)
        row_keep = position < sink_end
        if remote_end > sink_end:
            if head_count > 1 and stride > 1:
                offsets = torch.div(head_indices * stride, head_count, rounding_mode="floor")
            else:
                offsets = torch.zeros((1, head_count, 1), device=query_states.device, dtype=torch.long)
            remote_position = position - offsets
            landmark_keep = (
                (position >= sink_end)
                & (position < remote_end)
                & (remote_position >= 0)
                & (torch.remainder(remote_position, stride) == 0)
            )
            row_keep = row_keep | landmark_keep
        recent_start = max(recent_start, sink_end)
        if recent_start < history_count:
            row_keep = row_keep | ((position >= recent_start) & (position < history_count))
        if _ACTIVE_ALWAYS_KEEP_SELF:
            row_keep = row_keep | (position == key_count - 1)
        keep[row_idx : row_idx + 1] = row_keep

    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key_count]
    scores = scores.masked_fill(~keep.unsqueeze(2), torch.finfo(scores.dtype).min)
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_weights, value_states)
    return attention_output.transpose(1, 2).contiguous(), None


def _layer_budget_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    sink_tokens: int,
) -> tuple[torch.Tensor, None]:
    layer_idx = int(getattr(module, "layer_idx", 0))
    batch_budgets = layer_budgets_for_batch(layer_idx, query_states.shape[0])
    if batch_budgets is None:
        return _layer_budget_attention_forward_with_budget(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            sink_tokens,
            layer_budget_for_layer(layer_idx),
        )
    batch_count = query_states.shape[0]
    grouped_rows: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for row_idx, row_budget in enumerate(batch_budgets):
        budget_key = json.dumps(row_budget, sort_keys=True, separators=(",", ":"))
        if budget_key not in grouped_rows:
            grouped_rows[budget_key] = (row_budget, [])
        grouped_rows[budget_key][1].append(row_idx)
    if len(grouped_rows) == 1:
        row_budget, row_indices = next(iter(grouped_rows.values()))
        if len(row_indices) == batch_count:
            return _layer_budget_attention_forward_with_budget(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling,
                sink_tokens,
                row_budget,
            )
    if str2bool(os.environ.get("LAYER_BUDGET_MIXED_DENSE", "0")) and all(
        _is_mixed_dense_landmark_budget(row_budget) for row_budget in batch_budgets
    ):
        return _mixed_full_landmark_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            sink_tokens,
            batch_budgets,
        )
    group_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row_budget, row_indices in grouped_rows.values():
        index = batch_row_index_tensor(row_indices, query_states.device)
        group_attention_mask = (
            attention_mask.index_select(0, index)
            if attention_mask is not None and attention_mask.shape[0] == batch_count
            else attention_mask
        )
        group_output, _ = _layer_budget_attention_forward_with_budget(
            module,
            query_states.index_select(0, index),
            key_states.index_select(0, index),
            value_states.index_select(0, index),
            group_attention_mask,
            scaling,
            sink_tokens,
            row_budget,
        )
        group_outputs.append((index, group_output))
    output = torch.empty(
        (batch_count, *group_outputs[0][1].shape[1:]),
        dtype=group_outputs[0][1].dtype,
        device=group_outputs[0][1].device,
    )
    for index, group_output in group_outputs:
        output.index_copy_(0, index, group_output)
    return output, None


def _qabs_remote_recent_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    selected_history_indices: torch.Tensor,
    selected_valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    _, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qabs remote-recent attention requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    history_count = key_count - 1
    sink_end = min(_ACTIVE_PROTECT_SINK_TOKENS, history_count) if _ACTIVE_PROTECT_SINK_TOKENS > 0 else 0
    recent_start = history_count
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)

    remote_valid = selected_valid & (selected_history_indices >= 0) & (selected_history_indices < history_count)
    if sink_end > 0:
        remote_valid &= selected_history_indices >= sink_end
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        remote_valid &= ~((selected_history_indices >= recent_start) & (selected_history_indices < history_count))

    q = query_states[:, :, 0, :]
    score_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    if selected_history_indices.shape[-1] > 0:
        remote_indices = selected_history_indices.clamp(0, max(0, history_count - 1))
        remote_gather = remote_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        remote_keys = torch.gather(key_states, dim=2, index=remote_gather)
        remote_values = torch.gather(value_states, dim=2, index=remote_gather)
        remote_scores = torch.matmul(query_states[:, :, 0:1, :], remote_keys.transpose(2, 3)).squeeze(2) * scaling
        if attention_mask is not None:
            mask_row = attention_mask[:, :, 0, :key_count]
            if mask_row.shape[1] == 1 and head_count != 1:
                mask_row = mask_row.expand(-1, head_count, -1)
            remote_scores = remote_scores + torch.gather(mask_row, dim=-1, index=remote_indices)
        remote_scores = remote_scores.masked_fill(~remote_valid, torch.finfo(remote_scores.dtype).min)
        score_parts.append(remote_scores)
        value_parts.append(remote_values)

    def append_contiguous(start: int, end: int) -> None:
        if start >= end:
            return
        part_keys = key_states[:, :, start:end, :]
        part_values = value_states[:, :, start:end, :]
        part_scores = torch.matmul(query_states[:, :, 0:1, :], part_keys.transpose(2, 3)).squeeze(2) * scaling
        if attention_mask is not None:
            part_mask = attention_mask[:, :, 0, start:end]
            if part_mask.shape[1] == 1 and head_count != 1:
                part_mask = part_mask.expand(-1, head_count, -1)
            part_scores = part_scores + part_mask
        score_parts.append(part_scores)
        value_parts.append(part_values)

    append_contiguous(0, sink_end)
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        append_contiguous(max(recent_start, sink_end), history_count)
    if _ACTIVE_ALWAYS_KEEP_SELF:
        append_contiguous(key_count - 1, key_count)

    if not score_parts:
        return value_states[:, :, -1:, :].transpose(1, 2).contiguous()

    selected_scores = torch.cat(score_parts, dim=-1) if len(score_parts) > 1 else score_parts[0]
    selected_values = torch.cat(value_parts, dim=2) if len(value_parts) > 1 else value_parts[0]
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous()


def _maybe_cuda_final_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    final_indices: torch.Tensor,
    final_valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor | None:
    global _QABS_CUDA_FINAL_WARNED
    if not _ACTIVE_QABS_CUDA_FINAL_KERNEL:
        return None
    if not (query_states.is_cuda and key_states.is_cuda and value_states.is_cuda):
        return None
    if query_states.shape[0] != 1 and attention_mask is not None:
        return None
    try:
        from qabs_cuda_kernels import final_attention as qabs_cuda_final_attention

        return qabs_cuda_final_attention(
            query_states[:, :, 0, :].contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            final_indices.contiguous(),
            final_valid.contiguous(),
            float(scaling),
        )
    except Exception as exc:
        if not _QABS_CUDA_FINAL_WARNED:
            print(f"warning: qabs CUDA final kernel unavailable; falling back to PyTorch ({exc})", flush=True)
            _QABS_CUDA_FINAL_WARNED = True
        return None


def _maybe_cuda_partial_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    dim_count: int,
    dim_indices: torch.Tensor | None = None,
    key_dim_major: torch.Tensor | None = None,
) -> torch.Tensor | None:
    global _QABS_CUDA_CANDIDATE_WARNED
    if not _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL:
        return None
    if not (query.is_cuda and key_history.is_cuda):
        return None
    try:
        if dim_indices is not None and key_dim_major is not None:
            from qabs_cuda_kernels import partial_scores_dim_major as qabs_cuda_partial_scores_dim_major

            return qabs_cuda_partial_scores_dim_major(
                query.contiguous(),
                key_dim_major,
                dim_indices.contiguous(),
            )
        from qabs_cuda_kernels import partial_scores as qabs_cuda_partial_scores

        return qabs_cuda_partial_scores(
            query.contiguous(),
            key_history,
            int(dim_count),
        )
    except Exception as exc:
        if not _QABS_CUDA_CANDIDATE_WARNED:
            print(f"warning: qabs CUDA candidate kernel unavailable; falling back to PyTorch ({exc})", flush=True)
            _QABS_CUDA_CANDIDATE_WARNED = True
        return None


def _maybe_cuda_brep_partial_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    rep_indices: torch.Tensor,
    rep_valid: torch.Tensor,
    block_size: int,
    dim_count: int,
    scan_fraction: float,
) -> torch.Tensor | None:
    global _QABS_CUDA_BREP_WARNED
    if not _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL:
        return None
    if not (query.is_cuda and key_history.is_cuda and rep_indices.is_cuda and rep_valid.is_cuda):
        return None
    try:
        from qabs_cuda_kernels import brep_partial_scores as qabs_cuda_brep_partial_scores

        return qabs_cuda_brep_partial_scores(
            query.contiguous(),
            key_history.contiguous(),
            rep_indices.contiguous(),
            rep_valid.contiguous(),
            int(block_size),
            int(dim_count),
            float(scan_fraction),
        )
    except Exception as exc:
        if not _QABS_CUDA_BREP_WARNED:
            print(f"warning: brep CUDA partial-score kernel unavailable; falling back to PyTorch ({exc})", flush=True)
            _QABS_CUDA_BREP_WARNED = True
        return None



def _maybe_cuda_reuse_select(
    query: torch.Tensor,
    key_states: torch.Tensor,
    current_candidate: torch.Tensor,
    previous_candidate: torch.Tensor | None,
    previous_final: torch.Tensor | None,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    top_fraction: float,
    always_keep_self: bool,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global _QABS_CUDA_FULL_SCORE_WARNED
    if not (_ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL and _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL and _ACTIVE_QABS_CUDA_FINAL_KERNEL):
        return None
    if not query.is_cuda or not key_states.is_cuda:
        return None
    if _ACTIVE_CANDIDATE_STATS is not None or _ACTIVE_LOAD_STATS is not None:
        return None
    try:
        from qabs_cuda_kernels import reuse_select as qabs_cuda_reuse_select

        return qabs_cuda_reuse_select(
            query.contiguous(),
            key_states.contiguous(),
            current_candidate.contiguous(),
            previous_candidate.contiguous() if previous_candidate is not None else None,
            previous_final.contiguous() if previous_final is not None else None,
            int(protect_sink_tokens),
            int(protect_recent_tokens),
            float(top_fraction),
            bool(always_keep_self),
            float(scaling),
        )
    except Exception as exc:
        if not _QABS_CUDA_FULL_SCORE_WARNED:
            print(f"warning: qabs CUDA reuse-select kernel unavailable; falling back to PyTorch ({exc})", flush=True)
            _QABS_CUDA_FULL_SCORE_WARNED = True
        return None


def _maybe_cuda_candidate_full_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    current_candidate: torch.Tensor,
    previous_candidate: torch.Tensor | None,
    previous_final: torch.Tensor | None,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    scaling: float,
) -> torch.Tensor | None:
    global _QABS_CUDA_FULL_SCORE_WARNED
    if not _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL:
        return None
    if not (query.is_cuda and key_history.is_cuda and current_candidate.is_cuda):
        return None
    try:
        from qabs_cuda_kernels import candidate_full_scores as qabs_cuda_candidate_full_scores

        return qabs_cuda_candidate_full_scores(
            query.contiguous(),
            key_history,
            current_candidate.contiguous(),
            previous_candidate.contiguous() if previous_candidate is not None else None,
            previous_final.contiguous() if previous_final is not None else None,
            int(protect_sink_tokens),
            int(protect_recent_tokens),
            float(scaling),
        )
    except Exception as exc:
        if not _QABS_CUDA_FULL_SCORE_WARNED:
            print(f"warning: qabs CUDA candidate full-score kernel unavailable; falling back to PyTorch ({exc})", flush=True)
            _QABS_CUDA_FULL_SCORE_WARNED = True
        return None



def _select_qabs_candidate_from_partial_scores(
    partial_scores: torch.Tensor,
    requested: int,
    candidate_fraction: float,
    selection: str,
    previous_thresholds: torch.Tensor | None,
    sample_size: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    history_count = int(partial_scores.shape[-1])
    if selection == "previous_threshold" and previous_thresholds is not None:
        threshold = previous_thresholds.float()
        if threshold.ndim == 2:
            threshold = threshold[:, :, None]
        return partial_scores >= threshold, threshold, "previous_threshold"
    if selection == "sample_quantile" and history_count > 0:
        sample_count = min(history_count, max(1, int(sample_size)))
        if sample_count < history_count:
            sample_indices = torch.linspace(
                0,
                history_count - 1,
                steps=sample_count,
                device=partial_scores.device,
                dtype=torch.float32,
            ).round().long().unique()
            sampled = partial_scores.index_select(dim=-1, index=sample_indices)
        else:
            sampled = partial_scores
        sample_requested = min(sampled.shape[-1], max(1, math.ceil(candidate_fraction * sampled.shape[-1])))
        threshold = torch.topk(sampled, k=sample_requested, dim=-1, largest=True).values[:, :, -1:]
        return partial_scores >= threshold, threshold, "sample_quantile"
    threshold = torch.topk(partial_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]
    return partial_scores >= threshold, threshold, "topk"

def _qabs_reuse_fast_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    # Decode-only qabs-reuse path. It avoids the initial full-history QK matmul
    # and computes exact QK only for candidate/final selected token indices.
    qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
    if qabs_params is None:
        raise RuntimeError(f"Invalid qabs reuse-rerank mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = qabs_params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qabs fast path only supports decode-style query_count=1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    layer_idx = int(getattr(module, "layer_idx", 0))
    query_token = key_count - 1
    history_count = key_count - 1
    q_raw = query_states[:, :, 0, :]
    k_history = key_states[:, :, :history_count, :]
    profile = _ACTIVE_QABS_PROFILE_STATS
    mode_kind, _ = parse_mode_config(_ACTIVE_MODE)
    use_lagged_reuse = mode_kind == "lagged_reuse_rerank"
    lag_refresh_interval = parse_lag_refresh_interval(_ACTIVE_MODE)
    block_size = parse_qabs_reuse_block_size(_ACTIVE_MODE) if mode_kind == "qabs_reuse_rerank" else None
    selected_dim_count = min(max(1, dim_count), head_dim)
    with profile.time_stage(layer_idx, "qdim_topk", query_states.device) if profile is not None else nullcontext():
        current_qdim_indices = torch.topk(q_raw.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices
    lag_refresh_step = lag_refresh_interval is not None and query_token % lag_refresh_interval == 0

    previous_candidate = None
    previous_final = None
    previous_qdim_indices = None
    previous_qdim_values = None
    previous_thresholds = None
    with profile.time_stage(layer_idx, "previous_state_load", query_states.device) if profile is not None else nullcontext():
        if _ACTIVE_REUSE_STATE is not None:
            previous_candidate, previous_final = _ACTIVE_REUSE_STATE.previous_layer_masks(
                layer_idx,
                query_token,
                history_count,
                query_states.device,
            )
            previous_qdim_indices = _ACTIVE_REUSE_STATE.previous_layer_qdims(layer_idx, query_token, query_states.device)
            previous_qdim_values = _ACTIVE_REUSE_STATE.previous_layer_qvalues(layer_idx, query_token, query_states.device)
            previous_thresholds = _ACTIVE_REUSE_STATE.previous_layer_thresholds(layer_idx, query_token, query_states.device)

    drift_heads = None
    if qdrift_mode(_ACTIVE_MODE):
        lag_refresh_step = True
        if previous_qdim_indices is not None and previous_final is not None:
            overlap = (current_qdim_indices[..., :, None] == previous_qdim_indices[..., None, :]).any(dim=-1).sum(dim=-1)
            drift_heads = overlap < max(1, math.ceil(selected_dim_count / 2))
            if qdrift_cos_mode(_ACTIVE_MODE) and previous_qdim_values is not None:
                q_current = q_raw.float()
                current_on_previous = torch.gather(q_current, dim=-1, index=previous_qdim_indices)
                previous_values = previous_qdim_values.float()
                numerator = (current_on_previous * previous_values).sum(dim=-1)
                denominator = current_on_previous.norm(dim=-1) * previous_values.norm(dim=-1)
                cos_values = numerator / denominator.clamp_min(1.0e-6)
                drift_heads = drift_heads | (cos_values < 0.5)
                lag_refresh_step = bool(drift_heads.any().item())
            elif qdrift_head_mode(_ACTIVE_MODE) or qdrift_share_mode(_ACTIVE_MODE):
                lag_refresh_step = bool(drift_heads.any().item())
            elif qdrift_majority_mode(_ACTIVE_MODE):
                lag_refresh_step = bool((drift_heads.float().mean() > 0.5).item())
            else:
                lag_refresh_step = bool(drift_heads.any().item())

    pconf_threshold = parse_pconf_threshold(_ACTIVE_MODE)
    if pconf_threshold is not None:
        lag_refresh_step = True
        if previous_final is not None:
            prev_indices, prev_valid = _indices_from_keep_mask(previous_final)
            if prev_indices.shape[-1] == 0:
                drift_heads = torch.ones((batch_count, head_count), dtype=torch.bool, device=query_states.device)
            else:
                prev_gather = prev_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
                prev_keys = torch.gather(k_history, dim=2, index=prev_gather)
                prev_scores = torch.matmul(query_states[:, :, 0:1, :], prev_keys.transpose(2, 3)).squeeze(2) * scaling
                prev_scores = prev_scores.masked_fill(~prev_valid, torch.finfo(prev_scores.dtype).min)
                prev_weights = F.softmax(prev_scores, dim=-1, dtype=torch.float32)
                prev_weights = prev_weights.masked_fill(~prev_valid, 0.0)
                ess = 1.0 / prev_weights.square().sum(dim=-1).clamp_min(1.0e-12)
                valid_count = prev_valid.sum(dim=-1).float().clamp_min(1.0)
                ess_fraction = ess / valid_count
                drift_heads = ess_fraction > pconf_threshold
            lag_refresh_step = bool(drift_heads.any().item())

    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    candidate_thresholds = None
    candidate_selection_used = "topk"
    if (qdrift_head_mode(_ACTIVE_MODE) or qdrift_share_mode(_ACTIVE_MODE) or pconf_threshold is not None) and previous_final is not None and drift_heads is not None:
        current_candidate_history = torch.zeros(
            (batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device
        )
        candidate_union_history = previous_final.clone()
        q = q_raw.float()
        for batch_index, head_index in torch.nonzero(drift_heads, as_tuple=False).tolist():
            dim_index = current_qdim_indices[batch_index, head_index]
            q_selected = q[batch_index, head_index].gather(dim=-1, index=dim_index)
            k_selected = k_history[batch_index, head_index].float().gather(
                dim=-1,
                index=dim_index[None, :].expand(history_count, selected_dim_count),
            )
            partial_score = (k_selected * q_selected[None, :]).sum(dim=-1)
            threshold = torch.topk(partial_score, k=requested, dim=-1, largest=True).values[-1]
            current_candidate_history[batch_index, head_index] = partial_score >= threshold
            candidate_union_history[batch_index, head_index] |= current_candidate_history[batch_index, head_index]
        if qdrift_share_mode(_ACTIVE_MODE):
            shared_tokens = current_candidate_history.any(dim=1, keepdim=True).expand_as(current_candidate_history)
            candidate_union_history |= shared_tokens
    elif use_lagged_reuse and previous_final is not None and not lag_refresh_step:
        current_candidate_history = torch.zeros(
            (batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device
        )
        candidate_union_history = previous_final.clone()
    else:
        q = q_raw.float()
        with profile.time_stage(layer_idx, "partial_scores", query_states.device) if profile is not None else nullcontext():
            key_dim_major = (
                _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, k_history)
                if _ACTIVE_SPARQ_MEAN_STATE is not None
                else None
            )
            partial_scores = _maybe_cuda_partial_scores(
                q_raw,
                k_history,
                selected_dim_count,
                current_qdim_indices,
                key_dim_major,
            )
            if partial_scores is None:
                dim_indices = current_qdim_indices
                q_selected = torch.gather(q, dim=-1, index=dim_indices)
                k_dim_indices = dim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
                k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
                partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
        with profile.time_stage(layer_idx, "candidate_select", query_states.device) if profile is not None else nullcontext():
            current_candidate_history, candidate_thresholds, candidate_selection_used = _select_qabs_candidate_from_partial_scores(
                partial_scores,
                requested,
                candidate_fraction,
                _ACTIVE_QABS_CANDIDATE_SELECTION,
                previous_thresholds,
                _ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE,
            )
        candidate_union_history = None

    score_previous_candidate = None
    score_previous_final = None
    score_protect_sink_tokens = 0
    score_protect_recent_tokens = 0
    if use_lagged_reuse:
        score_previous_final = previous_final
        score_protect_sink_tokens = _ACTIVE_PROTECT_SINK_TOKENS
        score_protect_recent_tokens = _ACTIVE_PROTECT_RECENT_TOKENS
    else:
        if previous_candidate is not None and qabs_reuse_uses_previous_candidate(_ACTIVE_MODE):
            score_previous_candidate = previous_candidate
        if previous_final is not None and qabs_reuse_uses_previous_final(_ACTIVE_MODE):
            score_previous_final = previous_final

    if candidate_union_history is None and not _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL:
        with profile.time_stage(layer_idx, "candidate_union", query_states.device) if profile is not None else nullcontext():
            candidate_union_history = current_candidate_history.clone()
            if score_previous_candidate is not None:
                candidate_union_history |= score_previous_candidate
            if score_previous_final is not None:
                candidate_union_history |= score_previous_final

    if use_lagged_reuse and candidate_union_history is not None:
        if _ACTIVE_PROTECT_SINK_TOKENS > 0:
            candidate_union_history[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
        if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
            recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
            candidate_union_history[:, :, recent_start:history_count] = True

    if use_lagged_reuse and _ACTIVE_REUSE_STATE is not None:
        _ACTIVE_REUSE_STATE.record_refresh_heads(drift_heads if (qdrift_head_mode(_ACTIVE_MODE) or pconf_threshold is not None) else None if lag_refresh_step else torch.zeros((batch_count, head_count), dtype=torch.bool, device=query_states.device), head_count)

    if _ACTIVE_REUSE_OVERLAP_STATS is not None:
        _ACTIVE_REUSE_OVERLAP_STATS.update(
            layer_idx,
            current_candidate_history,
            previous_candidate,
            previous_final,
        )

    if candidate_union_history is None:
        score_current_candidate = current_candidate_history
        score_previous_candidate_for_kernel = score_previous_candidate
        score_previous_final_for_kernel = score_previous_final
        score_protect_sink_for_kernel = score_protect_sink_tokens
        score_protect_recent_for_kernel = score_protect_recent_tokens
    else:
        score_current_candidate = candidate_union_history
        score_previous_candidate_for_kernel = None
        score_previous_final_for_kernel = None
        score_protect_sink_for_kernel = 0
        score_protect_recent_for_kernel = 0

    with profile.time_stage(layer_idx, "reuse_select_kernel", query_states.device) if profile is not None else nullcontext():
        reuse_select_result = (
            None
            if block_size is not None or _ACTIVE_FORCE_EVIDENCE_SPANS
            else _maybe_cuda_reuse_select(
                q_raw,
                key_states,
                score_current_candidate,
                score_previous_candidate_for_kernel,
                score_previous_final_for_kernel,
                _ACTIVE_PROTECT_SINK_TOKENS,
                _ACTIVE_PROTECT_RECENT_TOKENS,
                _ACTIVE_TOP_FRACTION,
                _ACTIVE_ALWAYS_KEEP_SELF,
                scaling,
            )
        )
    if reuse_select_result is not None:
        final_indices, final_valid, history_final = reuse_select_result
        with profile.time_stage(layer_idx, "final_attention", query_states.device) if profile is not None else nullcontext():
            cuda_attention_output = _maybe_cuda_final_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                final_indices,
                final_valid,
                scaling,
            )
        if cuda_attention_output is not None:
            if profile is not None:
                profile_union_history = score_current_candidate
                if score_previous_candidate_for_kernel is not None:
                    profile_union_history = profile_union_history | score_previous_candidate_for_kernel
                if score_previous_final_for_kernel is not None:
                    profile_union_history = profile_union_history | score_previous_final_for_kernel
                profile.record_masks(
                    layer_idx,
                    current_candidate_history,
                    profile_union_history,
                    history_final,
                    candidate_thresholds,
                    candidate_selection_used,
                )
            if _ACTIVE_REUSE_STATE is not None:
                _ACTIVE_REUSE_STATE.update_layer(
                    layer_idx,
                    query_token,
                    current_candidate_history,
                    history_final,
                    history_count,
                    current_qdim_indices,
                    torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices),
                    candidate_thresholds,
                )
            return cuda_attention_output, None

    with profile.time_stage(layer_idx, "candidate_full_scores", query_states.device) if profile is not None else nullcontext():
        candidate_scores = _maybe_cuda_candidate_full_scores(
            q_raw,
            k_history,
            score_current_candidate,
            score_previous_candidate_for_kernel,
            score_previous_final_for_kernel,
            score_protect_sink_for_kernel,
            score_protect_recent_for_kernel,
            scaling,
        )
    if candidate_scores is not None:
        keep_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
        keep_count = min(keep_count, candidate_scores.shape[-1])
        with profile.time_stage(layer_idx, "final_topk", query_states.device) if profile is not None else nullcontext():
            selected_scores, selected_history_indices = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
            selected_valid = torch.isfinite(selected_scores)
    else:
        if candidate_union_history is None:
            candidate_union_history = current_candidate_history.clone()
            if score_previous_candidate is not None:
                candidate_union_history |= score_previous_candidate
            if score_previous_final is not None:
                candidate_union_history |= score_previous_final
            if use_lagged_reuse:
                if _ACTIVE_PROTECT_SINK_TOKENS > 0:
                    candidate_union_history[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
                if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
                    recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
                    candidate_union_history[:, :, recent_start:history_count] = True
        with profile.time_stage(layer_idx, "candidate_full_scores", query_states.device) if profile is not None else nullcontext():
            candidate_indices, candidate_valid = _indices_from_keep_mask(candidate_union_history)
            if candidate_indices.shape[-1] == 0:
                candidate_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
                candidate_valid = torch.zeros_like(candidate_indices, dtype=torch.bool)
            candidate_gather = candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
            candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
            candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
            candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)

        keep_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
        keep_count = min(keep_count, candidate_scores.shape[-1])
        with profile.time_stage(layer_idx, "final_topk", query_states.device) if profile is not None else nullcontext():
            _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
            selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_candidate_positions)
            selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)
    with profile.time_stage(layer_idx, "final_mask_and_stats", query_states.device) if profile is not None else nullcontext():
        history_final = torch.zeros_like(current_candidate_history, dtype=torch.bool)
        history_final.scatter_(dim=-1, index=selected_history_indices, src=selected_valid)

        if _ACTIVE_PROTECT_SINK_TOKENS > 0:
            history_final[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
        if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
            recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
            history_final[:, :, recent_start:history_count] = True
        if _ACTIVE_FORCE_EVIDENCE_SPANS:
            history_final = _force_evidence_spans_into_history_keep(history_final, _ACTIVE_EVIDENCE_SPANS)
        if block_size is not None:
            history_final = _expand_history_keep_to_aligned_blocks(history_final, block_size)
            selected_history_indices, selected_valid = _indices_from_keep_mask(history_final)
        elif _ACTIVE_FORCE_EVIDENCE_SPANS:
            selected_history_indices, selected_valid = _indices_from_keep_mask(history_final)
        if profile is not None:
            profile_union_history = candidate_union_history
            if profile_union_history is None:
                profile_union_history = current_candidate_history.clone()
                if score_previous_candidate is not None:
                    profile_union_history |= score_previous_candidate
                if score_previous_final is not None:
                    profile_union_history |= score_previous_final
            profile.record_masks(
                layer_idx,
                current_candidate_history,
                profile_union_history,
                history_final,
                candidate_thresholds,
                candidate_selection_used,
            )

    if _ACTIVE_CANDIDATE_STATS is not None:
        if candidate_union_history is None:
            candidate_union_history = current_candidate_history.clone()
            if score_previous_candidate is not None:
                candidate_union_history |= score_previous_candidate
            if score_previous_final is not None:
                candidate_union_history |= score_previous_final
            if use_lagged_reuse:
                if _ACTIVE_PROTECT_SINK_TOKENS > 0:
                    candidate_union_history[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
                if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
                    recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
                    candidate_union_history[:, :, recent_start:history_count] = True
        _ACTIVE_CANDIDATE_STATS.update(layer_idx, candidate_union_history, torch.ones_like(candidate_union_history))
    if _ACTIVE_LOAD_STATS is not None:
        _ACTIVE_LOAD_STATS.update(layer_idx, torch.ones_like(history_final), history_final, torch.ones_like(history_final))
    if _ACTIVE_REUSE_STATE is not None:
        _ACTIVE_REUSE_STATE.update_layer(
            layer_idx,
            query_token,
            current_candidate_history,
            history_final,
            history_count,
            current_qdim_indices,
            torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices),
            candidate_thresholds,
        )

    final_indices, final_valid = _qabs_final_indices_from_selected(
        selected_history_indices,
        selected_valid,
        history_count,
        key_count,
        _ACTIVE_ALWAYS_KEEP_SELF,
    )
    with profile.time_stage(layer_idx, "final_attention", query_states.device) if profile is not None else nullcontext():
        cuda_attention_output = _maybe_cuda_final_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            final_indices,
            final_valid,
            scaling,
        )
        if cuda_attention_output is not None:
            return cuda_attention_output, None

        final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        selected_keys = torch.gather(key_states, dim=2, index=final_gather)
        selected_values = torch.gather(value_states, dim=2, index=final_gather)
        selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
        if attention_mask is not None:
            mask_row = attention_mask[:, :, 0, :key_count]
            if mask_row.shape[1] == 1 and head_count != 1:
                mask_row = mask_row.expand(-1, head_count, -1)
            selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
        selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
        attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
        attention_output = attention_output[:, None, :, :].contiguous()
        return attention_output, None


def _qabs_candidate_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
    if qabs_params is None:
        raise RuntimeError(f"Invalid qabs candidate-attention mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = qabs_params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qabs candidate-attention mode requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    q_raw = query_states[:, :, 0, :]
    k_history = key_states[:, :, :history_count, :]
    profile = _ACTIVE_QABS_PROFILE_STATS
    with profile.time_stage(layer_idx, "qdim_topk", query_states.device) if profile is not None else nullcontext():
        current_qdim_indices = torch.topk(q_raw.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices
    key_dim_major = (
        _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, k_history)
        if _ACTIVE_SPARQ_MEAN_STATE is not None
        else None
    )
    with profile.time_stage(layer_idx, "partial_scores", query_states.device) if profile is not None else nullcontext():
        partial_scores = _maybe_cuda_partial_scores(
            q_raw,
            k_history,
            selected_dim_count,
            current_qdim_indices,
            key_dim_major,
        )
        if partial_scores is None:
            q_selected = torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices)
            k_dim_indices = current_qdim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
            k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
            partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
    with profile.time_stage(layer_idx, "candidate_select", query_states.device) if profile is not None else nullcontext():
        _, candidate_indices = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
        candidate_valid = torch.ones_like(candidate_indices, dtype=torch.bool)
    with profile.time_stage(layer_idx, "candidate_index_build", query_states.device) if profile is not None else nullcontext():
        if _ACTIVE_MODE.endswith("appendattn"):
            final_indices, final_valid = _qabs_final_indices_append_protected(
                candidate_indices,
                candidate_valid,
                history_count,
                key_count,
                _ACTIVE_ALWAYS_KEEP_SELF,
            )
        else:
            final_indices, final_valid = _qabs_final_indices_from_selected(
                candidate_indices,
                candidate_valid,
                history_count,
                key_count,
                _ACTIVE_ALWAYS_KEEP_SELF,
            )
    if profile is not None:
        candidate_mask = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
        candidate_mask.scatter_(dim=-1, index=candidate_indices, src=candidate_valid)
        final_history_mask = torch.zeros_like(candidate_mask)
        final_history_mask.scatter_(
            dim=-1,
            index=final_indices.clamp_max(history_count - 1),
            src=final_valid & (final_indices < history_count),
        )
        profile.record_masks(layer_idx, candidate_mask, candidate_mask, final_history_mask, None, "topk")
    with profile.time_stage(layer_idx, "final_attention", query_states.device) if profile is not None else nullcontext():
        cuda_attention_output = _maybe_cuda_final_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            final_indices,
            final_valid,
            scaling,
        )
        if cuda_attention_output is not None:
            return cuda_attention_output, None

        final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        selected_keys = torch.gather(key_states, dim=2, index=final_gather)
        selected_values = torch.gather(value_states, dim=2, index=final_gather)
        selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
        if attention_mask is not None:
            mask_row = attention_mask[:, :, 0, :key_count]
            if mask_row.shape[1] == 1 and head_count != 1:
                mask_row = mask_row.expand(-1, head_count, -1)
            selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
        selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
        attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _qabs_periodic_candidate_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
    refresh_interval = parse_qabs_periodic_refresh_interval(_ACTIVE_MODE)
    if qabs_params is None or refresh_interval is None:
        raise RuntimeError(f"Invalid qabs periodic candidate-attention mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = qabs_params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qabs periodic candidate-attention mode requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None
    if _ACTIVE_REUSE_STATE is None:
        raise RuntimeError("qabs periodic candidate-attention mode requires an active reuse state.")

    layer_idx = int(getattr(module, "layer_idx", 0))
    query_token = key_count - 1
    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))

    previous_indices: torch.Tensor | None = None
    previous_valid: torch.Tensor | None = None
    if query_token % refresh_interval != 0:
        previous_indices, previous_valid = _ACTIVE_REUSE_STATE.previous_layer_indices(
            layer_idx,
            query_token,
            history_count,
            query_states.device,
        )
    if previous_indices is not None and previous_valid is not None:
        selected_history_indices = previous_indices
        selected_valid = previous_valid
        _ACTIVE_REUSE_STATE.update_layer_indices(
            layer_idx,
            query_token,
            selected_history_indices,
            selected_valid,
        )
    else:
        q_raw = query_states[:, :, 0, :]
        k_history = key_states[:, :, :history_count, :]
        current_qdim_indices = torch.topk(q_raw.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices
        key_dim_major = (
            _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, k_history)
            if _ACTIVE_SPARQ_MEAN_STATE is not None
            else None
        )
        partial_scores = _maybe_cuda_partial_scores(
            q_raw,
            k_history,
            selected_dim_count,
            current_qdim_indices,
            key_dim_major,
        )
        if partial_scores is None:
            q_selected = torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices)
            k_dim_indices = current_qdim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
            k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
            partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
        _, selected_history_indices = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
        selected_valid = torch.ones_like(selected_history_indices, dtype=torch.bool)
        _ACTIVE_REUSE_STATE.update_layer_indices(
            layer_idx,
            query_token,
            selected_history_indices,
            selected_valid,
        )

    if os.environ.get("QABS_PERIODIC_FUSED_REMOTE_RECENT", "false").lower() in {"1", "true", "yes", "on"}:
        attention_output = _qabs_remote_recent_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            selected_history_indices,
            selected_valid,
            scaling,
        )
    else:
        final_indices, final_valid = _qabs_final_indices_from_selected(
            selected_history_indices,
            selected_valid,
            history_count,
            key_count,
            _ACTIVE_ALWAYS_KEEP_SELF,
        )
        attention_output = _qabs_attention_from_final_indices(
            query_states,
            key_states,
            value_states,
            attention_mask,
            final_indices,
            final_valid,
            scaling,
        )
    return attention_output, None


def _qabs_attention_from_final_indices(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    final_indices: torch.Tensor,
    final_valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    if cuda_attention_output is not None:
        return cuda_attention_output

    _, head_count, _, head_dim = query_states.shape
    key_count = key_states.shape[-2]
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous()


def _qabs_candidate_top2_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
    if qabs_params is None:
        raise RuntimeError(f"Invalid qabs candidate-top2 mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = qabs_params
    mode_kind, _ = parse_mode_config(_ACTIVE_MODE)
    head_limit = parse_qabs_top2_hlimit(_ACTIVE_MODE)
    use_head_limit = mode_kind == "qabs_candidate_top2_hlimit_attention"
    use_global_budget = mode_kind == "qabs_candidate_top2_global_attention"

    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qabs candidate-top2 modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    candidate_count = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    final_per_head = min(candidate_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))

    q_raw = query_states[:, :, 0, :]
    k_history = key_states[:, :, :history_count, :]
    current_qdim_indices = torch.topk(q_raw.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices
    key_dim_major = (
        _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, k_history)
        if _ACTIVE_SPARQ_MEAN_STATE is not None
        else None
    )
    partial_scores = _maybe_cuda_partial_scores(
        q_raw,
        k_history,
        selected_dim_count,
        current_qdim_indices,
        key_dim_major,
    )
    if partial_scores is None:
        q_selected = torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices)
        k_dim_indices = current_qdim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
        k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
        partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)

    _, candidate_indices = torch.topk(partial_scores, k=candidate_count, dim=-1, largest=True)
    candidate_gather = candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :history_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        candidate_scores = candidate_scores + torch.gather(mask_row, dim=-1, index=candidate_indices)

    if use_global_budget:
        global_budget = min(candidate_scores.numel() // batch_count, max(1, head_count * final_per_head))
        flat_scores = candidate_scores.reshape(batch_count, -1)
        _, flat_positions = torch.topk(flat_scores, k=global_budget, dim=-1, largest=True)
        flat_keep = torch.zeros_like(flat_scores, dtype=torch.bool)
        flat_keep.scatter_(dim=-1, index=flat_positions, src=torch.ones_like(flat_positions, dtype=torch.bool))
        candidate_keep = flat_keep.view(batch_count, head_count, candidate_count)
        history_final = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
        history_final.scatter_(dim=-1, index=candidate_indices, src=candidate_keep)
        selected_history_indices, selected_valid = _indices_from_keep_mask(history_final)
    else:
        selected_scores, selected_candidate_positions = torch.topk(candidate_scores, k=final_per_head, dim=-1, largest=True)
        selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_candidate_positions)
        selected_valid = torch.ones_like(selected_history_indices, dtype=torch.bool)
        if use_head_limit and head_limit is not None:
            history_final = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
            history_final.scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
            selected_score_history = torch.full(
                (batch_count, head_count, history_count),
                torch.finfo(candidate_scores.dtype).min,
                dtype=candidate_scores.dtype,
                device=query_states.device,
            )
            selected_score_history.scatter_(dim=-1, index=selected_history_indices, src=selected_scores)
            protected_keys = torch.zeros((batch_count, history_count), dtype=torch.bool, device=query_states.device)
            if _ACTIVE_PROTECT_SINK_TOKENS > 0:
                protected_keys[:, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
            if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
                recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
                protected_keys[:, recent_start:history_count] = True
            history_final = _limit_heads_per_token_by_score_protected(
                history_final,
                selected_score_history,
                head_limit,
                protected_keys,
            )
            selected_history_indices, selected_valid = _indices_from_keep_mask(history_final)

    final_indices, final_valid = _qabs_final_indices_from_selected(
        selected_history_indices,
        selected_valid,
        history_count,
        key_count,
        _ACTIVE_ALWAYS_KEEP_SELF,
    )
    attention_output = _qabs_attention_from_final_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    return attention_output, None


def _qabs_shared_candidate_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
    if qabs_params is None:
        raise RuntimeError(f"Invalid shared qabs candidate mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = qabs_params
    mode_kind, _ = parse_mode_config(_ACTIVE_MODE)
    use_top2_rerank = mode_kind == "qabs_shared_candidate_top2_attention"
    refresh_interval = parse_qabs_shared_refresh_interval(_ACTIVE_MODE)

    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("shared qabs candidate modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    layer_idx = int(getattr(module, "layer_idx", 0))
    query_token = key_count - 1
    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    candidate_count = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    final_count = min(candidate_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))

    if refresh_interval is not None and _ACTIVE_REUSE_STATE is not None and query_token % refresh_interval != 0:
        _, previous_final = _ACTIVE_REUSE_STATE.previous_layer_masks(
            layer_idx,
            query_token,
            history_count,
            query_states.device,
        )
        if previous_final is not None:
            if _ACTIVE_PROTECT_SINK_TOKENS > 0:
                previous_final[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
            if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
                recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
                previous_final[:, :, recent_start:history_count] = True
            selected_history_indices, selected_valid = _indices_from_keep_mask(previous_final)
            final_indices, final_valid = _qabs_final_indices_from_selected(
                selected_history_indices,
                selected_valid,
                history_count,
                key_count,
                _ACTIVE_ALWAYS_KEEP_SELF,
            )
            attention_output = _qabs_attention_from_final_indices(
                query_states,
                key_states,
                value_states,
                attention_mask,
                final_indices,
                final_valid,
                scaling,
            )
            _ACTIVE_REUSE_STATE.update_layer(
                layer_idx,
                query_token,
                previous_final,
                previous_final,
                history_count,
            )
            return attention_output, None

    q_raw = query_states[:, :, 0, :]
    k_history = key_states[:, :, :history_count, :]
    current_qdim_indices = torch.topk(q_raw.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices
    key_dim_major = (
        _ACTIVE_SPARQ_MEAN_STATE.key_dim_major(layer_idx, k_history)
        if _ACTIVE_SPARQ_MEAN_STATE is not None
        else None
    )
    partial_scores = _maybe_cuda_partial_scores(
        q_raw,
        k_history,
        selected_dim_count,
        current_qdim_indices,
        key_dim_major,
    )
    if partial_scores is None:
        q_selected = torch.gather(q_raw.float(), dim=-1, index=current_qdim_indices)
        k_dim_indices = current_qdim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
        k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
        partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)

    shared_scores = partial_scores.max(dim=1).values
    _, shared_candidate_indices = torch.topk(shared_scores, k=candidate_count, dim=-1, largest=True)
    shared_candidate_indices = shared_candidate_indices[:, None, :].expand(batch_count, head_count, candidate_count)

    if use_top2_rerank:
        candidate_gather = shared_candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
        candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
        if attention_mask is not None:
            mask_row = attention_mask[:, :, 0, :history_count]
            if mask_row.shape[1] == 1 and head_count != 1:
                mask_row = mask_row.expand(-1, head_count, -1)
            candidate_scores = candidate_scores + torch.gather(mask_row, dim=-1, index=shared_candidate_indices)
        _, selected_candidate_positions = torch.topk(candidate_scores, k=final_count, dim=-1, largest=True)
        selected_history_indices = torch.gather(shared_candidate_indices, dim=-1, index=selected_candidate_positions)
    else:
        selected_history_indices = shared_candidate_indices
    selected_valid = torch.ones_like(selected_history_indices, dtype=torch.bool)
    if refresh_interval is not None and _ACTIVE_REUSE_STATE is not None:
        history_final = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
        history_final.scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
        if _ACTIVE_PROTECT_SINK_TOKENS > 0:
            history_final[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
        if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
            recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
            history_final[:, :, recent_start:history_count] = True
        _ACTIVE_REUSE_STATE.update_layer(
            layer_idx,
            query_token,
            history_final,
            history_final,
            history_count,
        )

    final_indices, final_valid = _qabs_final_indices_from_selected(
        selected_history_indices,
        selected_valid,
        history_count,
        key_count,
        _ACTIVE_ALWAYS_KEEP_SELF,
    )
    attention_output = _qabs_attention_from_final_indices(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    return attention_output, None


def _kdim_posting_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_kdim_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid kdim mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("kdim modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]

    # Prototype for a future posting-list implementation: for each selected q dimension,
    # retrieve tokens with extreme K values in the sign-favorable direction.
    per_dim_count = min(history_count, max(1, math.ceil(candidate_fraction * history_count * 2.0 / selected_dim_count)))
    candidate_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
    for dim_pos in range(selected_dim_count):
        dim_index = q_dim_indices[:, :, dim_pos]
        k_dim_values = torch.gather(
            k_history.float(),
            dim=-1,
            index=dim_index[:, :, None, None].expand(batch_count, head_count, history_count, 1),
        ).squeeze(-1)
        signed_values = torch.where(q_selected[:, :, dim_pos : dim_pos + 1] >= 0, k_dim_values, -k_dim_values)
        _, dim_token_indices = torch.topk(signed_values, k=per_dim_count, dim=-1, largest=True)
        candidate_keep.scatter_(dim=-1, index=dim_token_indices, src=torch.ones_like(dim_token_indices, dtype=torch.bool))

    candidate_indices, candidate_valid = _indices_from_keep_mask(candidate_keep)
    if candidate_indices.shape[-1] == 0:
        candidate_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
        candidate_valid = torch.zeros_like(candidate_indices, dtype=torch.bool)
    candidate_gather = candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)

    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None

    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _fixed_projection_directions(count: int, dim: int, device: torch.device) -> torch.Tensor:
    proj = torch.arange(1, count + 1, device=device, dtype=torch.float32).view(count, 1)
    dims = torch.arange(1, dim + 1, device=device, dtype=torch.float32).view(1, dim)
    hashed = torch.sin(proj * dims * 12.9898 + proj * 78.233 + dims * 37.719)
    return torch.where(hashed >= 0, torch.ones_like(hashed), -torch.ones_like(hashed)) / math.sqrt(float(dim))


def _qabs_block_oracle_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_qboracle_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid qboracle mode: {_ACTIVE_MODE}")
    block_size, dim_count, scan_fraction, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qboracle modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]
    k_dim_indices = q_dim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
    k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
    partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)

    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    if padded_count != history_count:
        padded_scores = F.pad(partial_scores, (0, padded_count - history_count), value=-torch.inf)
    else:
        padded_scores = partial_scores
    block_scores = padded_scores.view(batch_count, head_count, block_count, block_size).amax(dim=-1)
    scan_block_count = min(block_count, max(1, math.ceil(scan_fraction * history_count / block_size)))
    _, selected_blocks = torch.topk(block_scores, k=scan_block_count, dim=-1, largest=True)

    scan_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
    for batch_index in range(batch_count):
        for head_index in range(head_count):
            for block_index in selected_blocks[batch_index, head_index].tolist():
                start = int(block_index) * block_size
                end = min(history_count, start + block_size)
                scan_keep[batch_index, head_index, start:end] = True
    candidate_partial = partial_scores.masked_fill(~scan_keep, torch.finfo(partial_scores.dtype).min)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    candidate_partial_scores, candidate_history_indices = torch.topk(candidate_partial, k=requested, dim=-1, largest=True)
    candidate_valid = torch.isfinite(candidate_partial_scores)

    candidate_gather = candidate_history_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_history_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _block_random_rep_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_brp_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid brp mode: {_ACTIVE_MODE}")
    block_size, projection_count, reps_per_projection, dim_count, scan_fraction, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("brp modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]

    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    if padded_count != history_count:
        pad_tokens = padded_count - history_count
        k_for_blocks = F.pad(k_history.float(), (0, 0, 0, pad_tokens), value=0.0)
        valid_for_blocks = torch.zeros((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)
        valid_for_blocks[:, :, :history_count] = True
    else:
        k_for_blocks = k_history.float()
        valid_for_blocks = torch.ones((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)

    block_k = k_for_blocks.view(batch_count, head_count, block_count, block_size, head_dim)
    block_valid = valid_for_blocks.view(batch_count, head_count, block_count, block_size)
    directions = _fixed_projection_directions(projection_count, head_dim, key_states.device)
    reps_per_projection = min(reps_per_projection, block_size)
    rep_key_parts = []
    rep_valid_parts = []
    for projection_index in range(projection_count):
        direction = directions[projection_index]
        projection_scores = (block_k * direction.view(1, 1, 1, 1, head_dim)).sum(dim=-1).masked_fill(~block_valid, -torch.inf)
        _, rep_positions = torch.topk(projection_scores, k=reps_per_projection, dim=-1, largest=True)
        rep_gather = rep_positions[:, :, :, :, None].expand(-1, -1, -1, -1, head_dim)
        rep_key_parts.append(torch.gather(block_k, dim=3, index=rep_gather))
        rep_valid_parts.append(torch.gather(block_valid, dim=3, index=rep_positions))
    rep_keys = torch.cat(rep_key_parts, dim=3)
    rep_valid = torch.cat(rep_valid_parts, dim=3)
    rep_total = rep_keys.shape[3]

    q_dim_for_reps = q_dim_indices[:, :, None, None, :].expand(-1, -1, block_count, rep_total, -1)
    rep_selected = torch.gather(rep_keys, dim=-1, index=q_dim_for_reps)
    rep_scores = (rep_selected * q_selected[:, :, None, None, :]).sum(dim=-1)
    rep_scores = rep_scores.masked_fill(~rep_valid, torch.finfo(rep_scores.dtype).min)
    block_scores = rep_scores.amax(dim=-1)

    scan_block_count = min(block_count, max(1, math.ceil(scan_fraction * history_count / block_size)))
    _, selected_blocks = torch.topk(block_scores, k=scan_block_count, dim=-1, largest=True)
    scan_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=key_states.device)
    for batch_index in range(batch_count):
        for head_index in range(head_count):
            for block_index in selected_blocks[batch_index, head_index].tolist():
                start = int(block_index) * block_size
                end = min(history_count, start + block_size)
                scan_keep[batch_index, head_index, start:end] = True

    scan_indices, scan_valid = _indices_from_keep_mask(scan_keep)
    if scan_indices.shape[-1] == 0:
        scan_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
        scan_valid = torch.zeros_like(scan_indices, dtype=torch.bool)
    scan_gather = scan_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    scan_keys = torch.gather(k_history, dim=2, index=scan_gather)
    q_dim_for_scan = q_dim_indices[:, :, None, :].expand(-1, -1, scan_indices.shape[-1], -1)
    scan_selected = torch.gather(scan_keys.float(), dim=-1, index=q_dim_for_scan)
    partial_scores = (scan_selected * q_selected[:, :, None, :]).sum(dim=-1)
    partial_scores = partial_scores.masked_fill(~scan_valid, torch.finfo(partial_scores.dtype).min)

    requested = min(scan_indices.shape[-1], max(1, math.ceil(candidate_fraction * history_count)))
    _, candidate_positions = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
    candidate_history_indices = torch.gather(scan_indices, dim=-1, index=candidate_positions)
    candidate_valid = torch.gather(scan_valid, dim=-1, index=candidate_positions)

    candidate_gather = candidate_history_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_history_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _block_group_rep_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_bgrp_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid bgrp mode: {_ACTIVE_MODE}")
    block_size, group_count, reps_per_group, dim_count, scan_fraction, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("bgrp modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]

    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    if padded_count != history_count:
        pad_tokens = padded_count - history_count
        k_for_blocks = F.pad(k_history.float(), (0, 0, 0, pad_tokens), value=0.0)
        valid_for_blocks = torch.zeros((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)
        valid_for_blocks[:, :, :history_count] = True
    else:
        k_for_blocks = k_history.float()
        valid_for_blocks = torch.ones((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)

    block_k = k_for_blocks.view(batch_count, head_count, block_count, block_size, head_dim)
    block_valid = valid_for_blocks.view(batch_count, head_count, block_count, block_size)
    group_count = min(group_count, head_dim)
    reps_per_group = min(reps_per_group, block_size)
    rep_key_parts = []
    rep_valid_parts = []
    for group_index in range(group_count):
        start_dim = (group_index * head_dim) // group_count
        end_dim = ((group_index + 1) * head_dim) // group_count
        if end_dim <= start_dim:
            continue
        group_norms = block_k[..., start_dim:end_dim].square().sum(dim=-1).masked_fill(~block_valid, -torch.inf)
        _, rep_positions = torch.topk(group_norms, k=reps_per_group, dim=-1, largest=True)
        rep_gather = rep_positions[:, :, :, :, None].expand(-1, -1, -1, -1, head_dim)
        rep_key_parts.append(torch.gather(block_k, dim=3, index=rep_gather))
        rep_valid_parts.append(torch.gather(block_valid, dim=3, index=rep_positions))
    if not rep_key_parts:
        rep_keys = block_k[:, :, :, :1, :]
        rep_valid = block_valid[:, :, :, :1]
    else:
        rep_keys = torch.cat(rep_key_parts, dim=3)
        rep_valid = torch.cat(rep_valid_parts, dim=3)
    rep_total = rep_keys.shape[3]

    q_dim_for_reps = q_dim_indices[:, :, None, None, :].expand(-1, -1, block_count, rep_total, -1)
    rep_selected = torch.gather(rep_keys, dim=-1, index=q_dim_for_reps)
    rep_scores = (rep_selected * q_selected[:, :, None, None, :]).sum(dim=-1)
    rep_scores = rep_scores.masked_fill(~rep_valid, torch.finfo(rep_scores.dtype).min)
    block_scores = rep_scores.amax(dim=-1)

    scan_block_count = min(block_count, max(1, math.ceil(scan_fraction * history_count / block_size)))
    _, selected_blocks = torch.topk(block_scores, k=scan_block_count, dim=-1, largest=True)
    scan_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=key_states.device)
    for batch_index in range(batch_count):
        for head_index in range(head_count):
            for block_index in selected_blocks[batch_index, head_index].tolist():
                start = int(block_index) * block_size
                end = min(history_count, start + block_size)
                scan_keep[batch_index, head_index, start:end] = True

    scan_indices, scan_valid = _indices_from_keep_mask(scan_keep)
    if scan_indices.shape[-1] == 0:
        scan_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
        scan_valid = torch.zeros_like(scan_indices, dtype=torch.bool)
    scan_gather = scan_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    scan_keys = torch.gather(k_history, dim=2, index=scan_gather)
    q_dim_for_scan = q_dim_indices[:, :, None, :].expand(-1, -1, scan_indices.shape[-1], -1)
    scan_selected = torch.gather(scan_keys.float(), dim=-1, index=q_dim_for_scan)
    partial_scores = (scan_selected * q_selected[:, :, None, :]).sum(dim=-1)
    partial_scores = partial_scores.masked_fill(~scan_valid, torch.finfo(partial_scores.dtype).min)

    requested = min(scan_indices.shape[-1], max(1, math.ceil(candidate_fraction * history_count)))
    _, candidate_positions = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
    candidate_history_indices = torch.gather(scan_indices, dim=-1, index=candidate_positions)
    candidate_valid = torch.gather(scan_valid, dim=-1, index=candidate_positions)

    candidate_gather = candidate_history_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_history_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _block_rep_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_brep_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid brep mode: {_ACTIVE_MODE}")
    block_size, rep_count, dim_count, scan_fraction, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("brep modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]

    block_count = math.ceil(history_count / block_size)
    reps_per_block = min(rep_count, block_size)
    if _ACTIVE_BREP_STATE is not None:
        rep_indices, rep_valid = _ACTIVE_BREP_STATE.representatives(
            int(getattr(module, "layer_idx", 0)),
            key_states,
            history_count,
            block_size,
            reps_per_block,
        )
        rep_gather = rep_indices.clamp_min(0)[:, :, :, :, None].expand(-1, -1, -1, -1, head_dim)
        rep_keys = torch.gather(k_history.float(), dim=2, index=rep_gather.reshape(batch_count, head_count, -1, head_dim))
        rep_keys = rep_keys.view(batch_count, head_count, block_count, reps_per_block, head_dim)
    else:
        padded_count = block_count * block_size
        if padded_count != history_count:
            pad_tokens = padded_count - history_count
            k_for_blocks = F.pad(k_history.float(), (0, 0, 0, pad_tokens), value=0.0)
            valid_for_blocks = torch.zeros((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)
            valid_for_blocks[:, :, :history_count] = True
        else:
            k_for_blocks = k_history.float()
            valid_for_blocks = torch.ones((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)
        block_k = k_for_blocks.view(batch_count, head_count, block_count, block_size, head_dim)
        block_valid = valid_for_blocks.view(batch_count, head_count, block_count, block_size)
        token_norms = block_k.square().sum(dim=-1).masked_fill(~block_valid, -torch.inf)
        _, rep_positions = torch.topk(token_norms, k=reps_per_block, dim=-1, largest=True)
        rep_gather = rep_positions[:, :, :, :, None].expand(-1, -1, -1, -1, head_dim)
        rep_keys = torch.gather(block_k, dim=3, index=rep_gather)
        rep_valid = torch.gather(block_valid, dim=3, index=rep_positions)

    cuda_partial_scores = None
    if _ACTIVE_BREP_STATE is not None:
        cuda_partial_scores = _maybe_cuda_brep_partial_scores(
            q,
            k_history,
            rep_indices,
            rep_valid,
            block_size,
            selected_dim_count,
            scan_fraction,
        )

    if cuda_partial_scores is not None:
        requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
        candidate_scores_partial, candidate_history_indices = torch.topk(cuda_partial_scores, k=requested, dim=-1, largest=True)
        candidate_valid = torch.isfinite(candidate_scores_partial)
    else:
        q_dim_for_reps = q_dim_indices[:, :, None, None, :].expand(-1, -1, block_count, reps_per_block, -1)
        rep_selected = torch.gather(rep_keys, dim=-1, index=q_dim_for_reps)
        rep_scores = (rep_selected * q_selected[:, :, None, None, :]).sum(dim=-1)
        rep_scores = rep_scores.masked_fill(~rep_valid, torch.finfo(rep_scores.dtype).min)
        block_scores = rep_scores.amax(dim=-1)

        scan_block_count = min(block_count, max(1, math.ceil(scan_fraction * history_count / block_size)))
        _, selected_blocks = torch.topk(block_scores, k=scan_block_count, dim=-1, largest=True)
        scan_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=key_states.device)
        for batch_index in range(batch_count):
            for head_index in range(head_count):
                for block_index in selected_blocks[batch_index, head_index].tolist():
                    start = int(block_index) * block_size
                    end = min(history_count, start + block_size)
                    scan_keep[batch_index, head_index, start:end] = True

        scan_indices, scan_valid = _indices_from_keep_mask(scan_keep)
        if scan_indices.shape[-1] == 0:
            scan_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
            scan_valid = torch.zeros_like(scan_indices, dtype=torch.bool)
        scan_gather = scan_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        scan_keys = torch.gather(k_history, dim=2, index=scan_gather)
        q_dim_for_scan = q_dim_indices[:, :, None, :].expand(-1, -1, scan_indices.shape[-1], -1)
        scan_selected = torch.gather(scan_keys.float(), dim=-1, index=q_dim_for_scan)
        partial_scores = (scan_selected * q_selected[:, :, None, :]).sum(dim=-1)
        partial_scores = partial_scores.masked_fill(~scan_valid, torch.finfo(partial_scores.dtype).min)

        requested = min(scan_indices.shape[-1], max(1, math.ceil(candidate_fraction * history_count)))
        _, candidate_positions = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
        candidate_history_indices = torch.gather(scan_indices, dim=-1, index=candidate_positions)
        candidate_valid = torch.gather(scan_valid, dim=-1, index=candidate_positions)

    candidate_gather = candidate_history_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_history_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _qabs_block_bound_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_qbb_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid qbb mode: {_ACTIVE_MODE}")
    block_size, dim_count, scan_fraction, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("qbb modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=q_dim_indices)
    k_history = key_states[:, :, :history_count, :]

    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    if padded_count != history_count:
        pad_tokens = padded_count - history_count
        k_for_blocks = F.pad(k_history.float(), (0, 0, 0, pad_tokens), value=0.0)
        valid_for_blocks = torch.zeros((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)
        valid_for_blocks[:, :, :history_count] = True
    else:
        k_for_blocks = k_history.float()
        valid_for_blocks = torch.ones((batch_count, head_count, padded_count), dtype=torch.bool, device=key_states.device)

    block_k = k_for_blocks.view(batch_count, head_count, block_count, block_size, head_dim)
    block_valid = valid_for_blocks.view(batch_count, head_count, block_count, block_size)
    q_dim_for_blocks = q_dim_indices[:, :, None, None, :].expand(-1, -1, block_count, block_size, -1)
    block_selected = torch.gather(block_k, dim=-1, index=q_dim_for_blocks)
    block_selected_max = block_selected.masked_fill(~block_valid[:, :, :, :, None], -torch.inf).amax(dim=3)
    block_selected_min = block_selected.masked_fill(~block_valid[:, :, :, :, None], torch.inf).amin(dim=3)
    block_upper = torch.where(q_selected[:, :, None, :] >= 0, q_selected[:, :, None, :] * block_selected_max, q_selected[:, :, None, :] * block_selected_min).sum(dim=-1)

    scan_block_count = min(block_count, max(1, math.ceil(scan_fraction * history_count / block_size)))
    _, selected_blocks = torch.topk(block_upper, k=scan_block_count, dim=-1, largest=True)
    scan_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=key_states.device)
    for batch_index in range(batch_count):
        for head_index in range(head_count):
            for block_index in selected_blocks[batch_index, head_index].tolist():
                start = int(block_index) * block_size
                end = min(history_count, start + block_size)
                scan_keep[batch_index, head_index, start:end] = True

    scan_indices, scan_valid = _indices_from_keep_mask(scan_keep)
    if scan_indices.shape[-1] == 0:
        scan_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query_states.device)
        scan_valid = torch.zeros_like(scan_indices, dtype=torch.bool)
    scan_gather = scan_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    scan_keys = torch.gather(k_history, dim=2, index=scan_gather)
    q_dim_for_scan = q_dim_indices[:, :, None, :].expand(-1, -1, scan_indices.shape[-1], -1)
    scan_selected = torch.gather(scan_keys.float(), dim=-1, index=q_dim_for_scan)
    partial_scores = (scan_selected * q_selected[:, :, None, :]).sum(dim=-1)
    partial_scores = partial_scores.masked_fill(~scan_valid, torch.finfo(partial_scores.dtype).min)

    requested = min(scan_indices.shape[-1], max(1, math.ceil(candidate_fraction * history_count)))
    _, candidate_positions = torch.topk(partial_scores, k=requested, dim=-1, largest=True)
    candidate_history_indices = torch.gather(scan_indices, dim=-1, index=candidate_positions)
    candidate_valid = torch.gather(scan_valid, dim=-1, index=candidate_positions)

    candidate_gather = candidate_history_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(k_history, dim=2, index=candidate_gather)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(candidate_scores.shape[-1], max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_history_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states, key_states, value_states, attention_mask, final_indices, final_valid, scaling
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _kdom_index_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_kdom_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid kdom mode: {_ACTIVE_MODE}")
    query_dim_count, key_dim_count, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("kdom modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None

    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    selected_query_dims = min(max(1, query_dim_count), head_dim)
    selected_key_dims = min(max(1, key_dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    q_dim_indices = torch.topk(q.abs(), k=selected_query_dims, dim=-1, largest=True).indices
    q_weights = torch.gather(q.abs(), dim=-1, index=q_dim_indices)

    if _ACTIVE_KDOM_STATE is not None:
        dominant_index = _ACTIVE_KDOM_STATE.dominant_index(layer_idx, key_states, history_count, selected_key_dims)
    else:
        dominant_index = KDomIndexState().dominant_index(layer_idx, key_states, history_count, selected_key_dims)

    dim_hits = torch.gather(
        dominant_index,
        dim=2,
        index=q_dim_indices[:, :, :, None].expand(batch_count, head_count, selected_query_dims, history_count),
    )
    candidate_rank_scores = (dim_hits.float() * q_weights[:, :, :, None]).sum(dim=2)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    _, candidate_indices = torch.topk(candidate_rank_scores, k=requested, dim=-1, largest=True)
    candidate_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
    candidate_keep.scatter_(dim=-1, index=candidate_indices, src=torch.ones_like(candidate_indices, dtype=torch.bool))

    candidate_padded, candidate_valid = _indices_from_keep_mask(candidate_keep)
    gather_index = candidate_padded[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(key_states[:, :, :history_count, :], dim=2, index=gather_index)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    keep_count = min(keep_count, candidate_scores.shape[-1])
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_padded, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None

    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _ksign_index_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    params = parse_ksign_params(_ACTIVE_MODE)
    if params is None:
        raise RuntimeError(f"Invalid ksign mode: {_ACTIVE_MODE}")
    dim_count, candidate_fraction = params
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("ksign modes require token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None
    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    selected_dim_count = min(max(1, dim_count), head_dim)
    q = query_states[:, :, 0, :].float()
    dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_signs = torch.gather(q.ge(0), dim=-1, index=dim_indices)
    if _ACTIVE_KSIGN_STATE is not None:
        sign_index = _ACTIVE_KSIGN_STATE.sign_index(layer_idx, key_states, history_count)
    else:
        sign_index = KSignIndexState().sign_index(layer_idx, key_states, history_count)
    gathered_signs = torch.gather(
        sign_index,
        dim=2,
        index=dim_indices[:, :, :, None].expand(batch_count, head_count, selected_dim_count, history_count),
    )
    sign_matches = gathered_signs == q_signs[:, :, :, None]
    if ksign_pm_mode(_ACTIVE_MODE):
        q_weights = torch.gather(q.abs(), dim=-1, index=dim_indices)
        signed_matches = torch.where(sign_matches, torch.ones_like(sign_matches, dtype=torch.float32), -torch.ones_like(sign_matches, dtype=torch.float32))
        match_scores = (signed_matches * q_weights[:, :, :, None]).sum(dim=2)
    elif ksign_weighted_mode(_ACTIVE_MODE):
        q_weights = torch.gather(q.abs(), dim=-1, index=dim_indices)
        match_scores = (sign_matches.float() * q_weights[:, :, :, None]).sum(dim=2)
    else:
        match_scores = sign_matches.sum(dim=2)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    _, candidate_indices = torch.topk(match_scores, k=requested, dim=-1, largest=True)
    candidate_keep = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=query_states.device)
    candidate_keep.scatter_(dim=-1, index=candidate_indices, src=torch.ones_like(candidate_indices, dtype=torch.bool))

    candidate_padded, candidate_valid = _indices_from_keep_mask(candidate_keep)
    gather_index = candidate_padded[:, :, :, None].expand(-1, -1, -1, head_dim)
    candidate_keys = torch.gather(key_states[:, :, :history_count, :], dim=2, index=gather_index)
    candidate_scores = torch.matmul(query_states[:, :, 0:1, :], candidate_keys.transpose(2, 3)).squeeze(2) * scaling
    candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
    keep_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    keep_count = min(keep_count, candidate_scores.shape[-1])
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_padded, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=selected_history_indices, src=selected_valid)
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None
    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _block_route_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    block_size = parse_block_route_size(_ACTIVE_MODE)
    if block_size is None:
        raise RuntimeError(f"Invalid block route mode: {_ACTIVE_MODE}")
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("blockroute requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None
    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    if _ACTIVE_BLOCK_ROUTE_STATE is not None:
        summaries = _ACTIVE_BLOCK_ROUTE_STATE.summaries(layer_idx, key_states, history_count, block_size)
    else:
        summaries = BlockRouteState().summaries(layer_idx, key_states, history_count, block_size)
    block_count = summaries.shape[2]
    routed_token_budget = max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count))
    selected_block_count = min(block_count, max(1, math.ceil(routed_token_budget / block_size)))
    block_scores = torch.matmul(query_states[:, :, 0:1, :].float(), summaries.transpose(2, 3)).squeeze(2)
    _, block_indices = torch.topk(block_scores, k=selected_block_count, dim=-1, largest=True)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    for block_offset in range(selected_block_count):
        starts = block_indices[:, :, block_offset] * block_size
        for batch_index in range(batch_count):
            for head_index in range(head_count):
                start = int(starts[batch_index, head_index].item())
                end = min(history_count, start + block_size)
                final_keep[batch_index, head_index, start:end] = True
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None

    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _knorm_reservoir_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, None]:
    batch_count, head_count, query_count, head_dim = query_states.shape
    if query_count != 1:
        raise RuntimeError("knormtop2 requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count <= 1:
        attention_output = value_states[:, :, -1:, :].transpose(1, 2).contiguous()
        return attention_output, None
    layer_idx = int(getattr(module, "layer_idx", 0))
    history_count = key_count - 1
    if _ACTIVE_KNORM_STATE is not None:
        norms = _ACTIVE_KNORM_STATE.key_norms(layer_idx, key_states)
    else:
        norms = key_states.float().square().sum(dim=-1)
    history_norms = norms[:, :, :history_count]
    keep_count = min(history_count, max(1, math.ceil(_ACTIVE_TOP_FRACTION * history_count)))
    _, top_indices = torch.topk(history_norms, k=keep_count, dim=-1, largest=True)
    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count].scatter_(dim=-1, index=top_indices, src=torch.ones_like(top_indices, dtype=torch.bool))
    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        final_keep[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        final_keep[:, :, recent_start:history_count] = True
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
    cuda_attention_output = _maybe_cuda_final_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        final_indices,
        final_valid,
        scaling,
    )
    if cuda_attention_output is not None:
        return cuda_attention_output, None

    final_gather = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states, dim=2, index=final_gather)
    selected_values = torch.gather(value_states, dim=2, index=final_gather)
    selected_scores = torch.matmul(query_states[:, :, 0:1, :], selected_keys.transpose(2, 3)).squeeze(2) * scaling
    if attention_mask is not None:
        mask_row = attention_mask[:, :, 0, :key_count]
        if mask_row.shape[1] == 1 and head_count != 1:
            mask_row = mask_row.expand(-1, head_count, -1)
        selected_scores = selected_scores + torch.gather(mask_row, dim=-1, index=final_indices)
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    attention_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.sum(attention_weights[:, :, :, None] * selected_values, dim=2)
    return attention_output[:, None, :, :].contiguous(), None


def _synthetic_kv_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    prototype_count: int,
    method: str,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if query_states.shape[-2] != 1:
        raise RuntimeError("synthetic KV attention requires token-by-token eval; set --eval_chunk_size 1.")
    key_count = key_states.shape[-2]
    if key_count == 0:
        raise RuntimeError("synthetic KV attention received an empty key cache.")

    sink_end = min(max(0, protect_sink_tokens), key_count)
    recent_start = max(sink_end, key_count - max(0, protect_recent_tokens))
    remote_start = sink_end
    remote_end = recent_start

    if method == "mass":
        score_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if sink_end > 0:
            sink_k = key_states[:, :, :sink_end, :]
            sink_v = value_states[:, :, :sink_end, :]
            score_parts.append(torch.matmul(query_states, sink_k.transpose(2, 3)) * scaling)
            value_parts.append(sink_v[:, :, None, :, :])

        remote_len = remote_end - remote_start
        if remote_len > 0:
            boundaries = torch.linspace(
                0,
                remote_len,
                steps=min(prototype_count, remote_len) + 1,
                device=key_states.device,
                dtype=torch.float32,
            ).round().to(torch.long)
            previous_end = 0
            for index in range(boundaries.numel() - 1):
                start = int(boundaries[index].item())
                end = int(boundaries[index + 1].item())
                start = max(start, previous_end)
                end = max(end, start + 1)
                end = min(end, remote_len)
                if start >= end:
                    continue
                key_chunk = key_states[:, :, remote_start + start : remote_start + end, :]
                value_chunk = value_states[:, :, remote_start + start : remote_start + end, :]
                chunk_scores = torch.matmul(query_states, key_chunk.transpose(2, 3)) * scaling
                chunk_weights = F.softmax(chunk_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                chunk_value = torch.sum(chunk_weights[..., None] * value_chunk[:, :, None, :, :], dim=-2)
                chunk_logit = torch.logsumexp(chunk_scores.float(), dim=-1, keepdim=True).to(query_states.dtype)
                score_parts.append(chunk_logit)
                value_parts.append(chunk_value[:, :, :, None, :])
                previous_end = end

        if recent_start < key_count:
            recent_k = key_states[:, :, recent_start:key_count, :]
            recent_v = value_states[:, :, recent_start:key_count, :]
            score_parts.append(torch.matmul(query_states, recent_k.transpose(2, 3)) * scaling)
            value_parts.append(recent_v[:, :, None, :, :])

        if not score_parts:
            score_parts.append(torch.matmul(query_states, key_states[:, :, -1:, :].transpose(2, 3)) * scaling)
            value_parts.append(value_states[:, :, None, -1:, :])
        scores = torch.cat(score_parts, dim=-1)
        values = torch.cat(value_parts, dim=-2)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.sum(weights[..., None] * values, dim=-2)
        return attention_output.transpose(1, 2).contiguous(), None

    parts_k: list[torch.Tensor] = []
    parts_v: list[torch.Tensor] = []
    if sink_end > 0:
        parts_k.append(key_states[:, :, :sink_end, :])
        parts_v.append(value_states[:, :, :sink_end, :])

    remote_len = remote_end - remote_start
    if remote_len > 0:
        if remote_len <= prototype_count:
            parts_k.append(key_states[:, :, remote_start:remote_end, :])
            parts_v.append(value_states[:, :, remote_start:remote_end, :])
        else:
            boundaries = torch.linspace(
                0,
                remote_len,
                steps=prototype_count + 1,
                device=key_states.device,
                dtype=torch.float32,
            ).round().to(torch.long)
            synth_k: list[torch.Tensor] = []
            synth_v: list[torch.Tensor] = []
            previous_end = 0
            for index in range(prototype_count):
                start = int(boundaries[index].item())
                end = int(boundaries[index + 1].item())
                start = max(start, previous_end)
                end = max(end, start + 1)
                end = min(end, remote_len)
                if start >= end:
                    continue
                key_chunk = key_states[:, :, remote_start + start : remote_start + end, :]
                value_chunk = value_states[:, :, remote_start + start : remote_start + end, :]
                synth_k.append(key_chunk.mean(dim=-2, keepdim=True))
                synth_v.append(value_chunk.mean(dim=-2, keepdim=True))
                previous_end = end
            if synth_k:
                parts_k.append(torch.cat(synth_k, dim=-2))
                parts_v.append(torch.cat(synth_v, dim=-2))

    if recent_start < key_count:
        parts_k.append(key_states[:, :, recent_start:key_count, :])
        parts_v.append(value_states[:, :, recent_start:key_count, :])

    compact_k = torch.cat(parts_k, dim=-2) if parts_k else key_states[:, :, -1:, :]
    compact_v = torch.cat(parts_v, dim=-2) if parts_v else value_states[:, :, -1:, :]
    scores = torch.matmul(query_states, compact_k.transpose(2, 3)) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(weights, compact_v)
    return attention_output.transpose(1, 2).contiguous(), None


def _landmark_synthetic_kv_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    recent_tokens: int,
    landmark_stride: int,
    prototype_count: int,
    method: str,
    sink_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if query_states.shape[-2] != 1:
        raise RuntimeError("landmark+synthetic KV attention requires token-by-token eval; set --eval_chunk_size 1.")
    batch_count, head_count, _, _ = query_states.shape
    key_count = key_states.shape[-2]
    if key_count == 0:
        raise RuntimeError("landmark+synthetic KV attention received an empty key cache.")

    history_count = max(0, key_count - 1)
    if history_count <= 0:
        return value_states[:, :, -1:, :].transpose(1, 2).contiguous(), None

    sink_end = min(max(0, sink_tokens), history_count)
    recent_start = max(sink_end, history_count - max(0, recent_tokens))
    remote_start = sink_end
    remote_end = recent_start

    landmark_indices: torch.Tensor | None = None
    if remote_end > remote_start:
        stride = max(1, int(landmark_stride))
        max_landmarks = max(1, math.ceil(remote_end / stride) + 1)
        base = torch.arange(max_landmarks, device=query_states.device, dtype=torch.long).view(1, 1, -1) * stride
        if head_count > 1 and stride > 1:
            offsets = torch.div(
                torch.arange(head_count, device=query_states.device, dtype=torch.long) * stride,
                head_count,
                rounding_mode="floor",
            ).view(1, head_count, 1)
        else:
            offsets = torch.zeros((1, head_count, 1), device=query_states.device, dtype=torch.long)
        indices = base + offsets
        valid = (indices >= remote_start) & (indices < remote_end)
        if valid.any():
            landmark_indices = indices.clamp(0, max(0, key_count - 1)).expand(batch_count, -1, -1)

    if method == "mass":
        score_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if sink_end > 0:
            sink_k = key_states[:, :, :sink_end, :]
            sink_v = value_states[:, :, :sink_end, :]
            score_parts.append(torch.matmul(query_states, sink_k.transpose(2, 3)) * scaling)
            value_parts.append(sink_v[:, :, None, :, :])

        if landmark_indices is not None:
            gather = landmark_indices[:, :, :, None].expand(-1, -1, -1, key_states.shape[-1])
            landmark_k = torch.gather(key_states, dim=2, index=gather)
            landmark_v = torch.gather(value_states, dim=2, index=gather)
            score_parts.append(torch.matmul(query_states, landmark_k.transpose(2, 3)) * scaling)
            value_parts.append(landmark_v[:, :, None, :, :])

        remote_len = remote_end - remote_start
        if remote_len > 0:
            boundaries = torch.linspace(
                0,
                remote_len,
                steps=min(prototype_count, remote_len) + 1,
                device=key_states.device,
                dtype=torch.float32,
            ).round().to(torch.long)
            previous_end = 0
            for index in range(boundaries.numel() - 1):
                start = int(boundaries[index].item())
                end = int(boundaries[index + 1].item())
                start = max(start, previous_end)
                end = max(end, start + 1)
                end = min(end, remote_len)
                if start >= end:
                    continue
                key_chunk = key_states[:, :, remote_start + start : remote_start + end, :]
                value_chunk = value_states[:, :, remote_start + start : remote_start + end, :]
                chunk_scores = torch.matmul(query_states, key_chunk.transpose(2, 3)) * scaling
                chunk_weights = F.softmax(chunk_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                chunk_value = torch.sum(chunk_weights[..., None] * value_chunk[:, :, None, :, :], dim=-2)
                chunk_logit = torch.logsumexp(chunk_scores.float(), dim=-1, keepdim=True).to(query_states.dtype)
                score_parts.append(chunk_logit)
                value_parts.append(chunk_value[:, :, :, None, :])
                previous_end = end

        if recent_start < history_count:
            recent_k = key_states[:, :, recent_start:history_count, :]
            recent_v = value_states[:, :, recent_start:history_count, :]
            score_parts.append(torch.matmul(query_states, recent_k.transpose(2, 3)) * scaling)
            value_parts.append(recent_v[:, :, None, :, :])

        if _ACTIVE_ALWAYS_KEEP_SELF:
            self_k = key_states[:, :, -1:, :]
            self_v = value_states[:, :, -1:, :]
            score_parts.append(torch.matmul(query_states, self_k.transpose(2, 3)) * scaling)
            value_parts.append(self_v[:, :, None, :, :])

        if not score_parts:
            score_parts.append(torch.matmul(query_states, key_states[:, :, -1:, :].transpose(2, 3)) * scaling)
            value_parts.append(value_states[:, :, None, -1:, :])
        scores = torch.cat(score_parts, dim=-1)
        values = torch.cat(value_parts, dim=-2)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.sum(weights[..., None] * values, dim=-2)
        return attention_output.transpose(1, 2).contiguous(), None

    parts_k: list[torch.Tensor] = []
    parts_v: list[torch.Tensor] = []
    if sink_end > 0:
        parts_k.append(key_states[:, :, :sink_end, :])
        parts_v.append(value_states[:, :, :sink_end, :])

    if landmark_indices is not None:
        gather = landmark_indices[:, :, :, None].expand(-1, -1, -1, key_states.shape[-1])
        parts_k.append(torch.gather(key_states, dim=2, index=gather))
        parts_v.append(torch.gather(value_states, dim=2, index=gather))

    remote_len = remote_end - remote_start
    if remote_len > 0:
        boundaries = torch.linspace(
            0,
            remote_len,
            steps=min(prototype_count, remote_len) + 1,
            device=key_states.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        synth_k: list[torch.Tensor] = []
        synth_v: list[torch.Tensor] = []
        previous_end = 0
        for index in range(boundaries.numel() - 1):
            start = int(boundaries[index].item())
            end = int(boundaries[index + 1].item())
            start = max(start, previous_end)
            end = max(end, start + 1)
            end = min(end, remote_len)
            if start >= end:
                continue
            key_chunk = key_states[:, :, remote_start + start : remote_start + end, :]
            value_chunk = value_states[:, :, remote_start + start : remote_start + end, :]
            synth_k.append(key_chunk.mean(dim=-2, keepdim=True))
            synth_v.append(value_chunk.mean(dim=-2, keepdim=True))
            previous_end = end
        if synth_k:
            parts_k.append(torch.cat(synth_k, dim=-2))
            parts_v.append(torch.cat(synth_v, dim=-2))

    if recent_start < history_count:
        parts_k.append(key_states[:, :, recent_start:history_count, :])
        parts_v.append(value_states[:, :, recent_start:history_count, :])

    if _ACTIVE_ALWAYS_KEEP_SELF:
        parts_k.append(key_states[:, :, -1:, :])
        parts_v.append(value_states[:, :, -1:, :])

    compact_k = torch.cat(parts_k, dim=-2) if parts_k else key_states[:, :, -1:, :]
    compact_v = torch.cat(parts_v, dim=-2) if parts_v else value_states[:, :, -1:, :]
    scores = torch.matmul(query_states, compact_k.transpose(2, 3)) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(weights, compact_v)
    return attention_output.transpose(1, 2).contiguous(), None


def _limited_eager_attention_forward(
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
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    mode_kind, mode_max_heads = parse_mode_config(_ACTIVE_MODE)
    if mode_kind == "ksign_index_rerank":
        return _ksign_index_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "kdom_index_rerank":
        return _kdom_index_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "qabs_block_bound":
        return _qabs_block_bound_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "block_rep_rerank":
        return _block_rep_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "block_group_rep_rerank":
        return _block_group_rep_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "block_random_rep_rerank":
        return _block_random_rep_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "qabs_block_oracle":
        return _qabs_block_oracle_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "kdim_posting_rerank":
        return _kdim_posting_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "block_route":
        return _block_route_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "knorm_reservoir":
        return _knorm_reservoir_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "synthetic_kv_attention":
        synthetic_params = parse_synthetic_kv_params(_ACTIVE_MODE)
        if synthetic_params is None:
            raise RuntimeError(f"Invalid synthetic KV mode: {_ACTIVE_MODE}")
        prototype_count, synthetic_method = synthetic_params
        return _synthetic_kv_attention_forward(
            query_states,
            key_states,
            value_states,
            scaling,
            prototype_count,
            synthetic_method,
            _ACTIVE_PROTECT_SINK_TOKENS,
            _ACTIVE_PROTECT_RECENT_TOKENS,
        )
    if mode_kind == "sparq_fast_attention":
        if query_states.shape[-2] != 1:
            raise RuntimeError("SparQ fast mode requires token-by-token eval; set --eval_chunk_size 1.")
        sparq_params = parse_sparq_params(_ACTIVE_MODE)
        if sparq_params is None:
            raise RuntimeError(f"Invalid SparQ fast mode: {_ACTIVE_MODE}")
        dim_count, candidate_fraction = sparq_params
        outputs = [
            _sparq_fast_attention_output_for_query(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_index,
                scaling,
                dim_count,
                candidate_fraction,
            )
            for query_index in range(query_states.shape[-2])
        ]
        return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None
    if mode_kind == "sparq_attention":
        sparq_params = parse_sparq_params(_ACTIVE_MODE)
        if sparq_params is None:
            raise RuntimeError(f"Invalid SparQ mode: {_ACTIVE_MODE}")
        dim_count, candidate_fraction = sparq_params
        outputs: list[torch.Tensor] = []
        final_keeps: list[torch.Tensor] = []
        for query_index in range(query_states.shape[-2]):
            output, selected_keep = _sparq_attention_output_for_query(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_index,
                scaling,
                dim_count,
                candidate_fraction,
                _ACTIVE_PROTECT_SINK_TOKENS,
                _ACTIVE_PROTECT_RECENT_TOKENS,
            )
            outputs.append(output)
            final_keeps.append(selected_keep)
        if _ACTIVE_LOAD_STATS is not None and final_keeps:
            layer_idx = int(getattr(module, "layer_idx", 0))
            for selected_keep in final_keeps:
                _ACTIVE_LOAD_STATS.update(layer_idx, torch.ones_like(selected_keep), selected_keep, torch.ones_like(selected_keep))
        attention_output = torch.stack(outputs, dim=1).contiguous()
        return attention_output, None
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"}
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _qabs_reuse_fast_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind == "qabs_periodic_candidate_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _qabs_periodic_candidate_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind == "qabs_candidate_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _qabs_candidate_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind
        in {
            "qabs_candidate_top2_attention",
            "qabs_candidate_top2_hlimit_attention",
            "qabs_candidate_top2_global_attention",
        }
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _qabs_candidate_top2_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind
        in {
            "qabs_shared_candidate_attention",
            "qabs_shared_candidate_top2_attention",
            "qabs_shared_reuse_attention",
        }
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _qabs_shared_candidate_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
    if mode_kind == "lagged_reuse_rerank":
        raise RuntimeError("lagged reuse mode requires --qabs_fast_path true and --eval_chunk_size 1.")
    if (
        mode_kind == "landmark_recent_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        landmark_params = parse_landmark_recent_params(_ACTIVE_MODE)
        if landmark_params is None:
            raise RuntimeError(f"Invalid landmark-recent mode: {_ACTIVE_MODE}")
        recent_tokens, landmark_stride = landmark_params
        return _landmark_recent_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            recent_tokens,
            landmark_stride,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    if (
        mode_kind == "full_head_recent_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        full_head_params = parse_full_head_recent_params(_ACTIVE_MODE)
        if full_head_params is None:
            raise RuntimeError(f"Invalid full-head-recent mode: {_ACTIVE_MODE}")
        full_heads, recent_tokens = full_head_params
        return _full_head_recent_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            full_heads,
            recent_tokens,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    if (
        mode_kind == "full_layer_recent_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        full_layer_params = parse_full_layer_recent_params(_ACTIVE_MODE)
        if full_layer_params is None:
            raise RuntimeError(f"Invalid full-layer-recent mode: {_ACTIVE_MODE}")
        full_layers, recent_tokens = full_layer_params
        return _full_layer_recent_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            full_layers,
            recent_tokens,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    if (
        mode_kind == "full_layer_landmark_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        full_layer_landmark_params = parse_full_layer_landmark_params(_ACTIVE_MODE)
        if full_layer_landmark_params is None:
            raise RuntimeError(f"Invalid full-layer-landmark mode: {_ACTIVE_MODE}")
        full_layers, recent_tokens, landmark_stride = full_layer_landmark_params
        return _full_layer_landmark_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            full_layers,
            recent_tokens,
            landmark_stride,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    if (
        mode_kind == "layer_budget_attention"
        and query_states.shape[-2] == 1
        and not bool(kwargs.get("output_attentions", False))
    ):
        return _layer_budget_attention_forward(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    if (
        mode_kind == "recent_window"
        and query_states.shape[-2] == 1
        and _ACTIVE_LOAD_STATS is None
        and not bool(kwargs.get("output_attentions", False))
    ):
        recent_tokens = parse_recent_window_tokens(_ACTIVE_MODE)
        if recent_tokens is None:
            raise RuntimeError(f"Invalid recent-window mode: {_ACTIVE_MODE}")
        return _recent_window_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            recent_tokens,
            _ACTIVE_PROTECT_SINK_TOKENS,
        )
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]

    if mode_kind in {
        "top2",
        "limit_random",
        "limit_score",
        "limit_score_fill",
        "limit_score_gap",
        "limit_score_protect",
        "top2_union_all",
        "observation_window",
        "observation_all_mass",
        "observation_hybrid",
        "sign_xnor",
        "sign_xnor_knorm",
        "sign_xnor_rerank",
        "recent_window",
            "qabs_candidate_keep",
            "qabs_partial_rerank",
            "qabs_reuse_rerank",
            "qabs_candidate_attention",
        }:
        final_keep = torch.zeros_like(scores, dtype=torch.bool)
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        chunk_query_start = key_count - query_count
        for query_index in range(query_count):
            row = scores[:, :, query_index, :]
            finite = torch.isfinite(row)
            history_original = _top2_history_keep_for_query(row, finite, _ACTIVE_TOP_FRACTION)
            if mode_kind in {"sign_xnor", "sign_xnor_knorm", "sign_xnor_rerank"}:
                candidate_fraction = parse_sign_xnor_candidate_fraction(_ACTIVE_MODE)
                if candidate_fraction is None:
                    raise RuntimeError(f"Invalid sign-XNOR mode: {_ACTIVE_MODE}")
                sign_candidate = _sign_xnor_history_keep_for_query(
                    query_states,
                    key_states,
                    query_index,
                    finite,
                    candidate_fraction,
                    weight_by_key_norm=mode_kind == "sign_xnor_knorm",
                )
                if mode_kind == "sign_xnor_rerank":
                    history_final = _topk_within_candidate_for_query(
                        row,
                        finite,
                        sign_candidate,
                        _ACTIVE_TOP_FRACTION,
                    )
                else:
                    history_final = sign_candidate
                history_final = _apply_full_head_protection(
                    history_final,
                    finite,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                    _ACTIVE_PROTECT_RECENT_TOKENS,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
            elif mode_kind == "recent_window":
                recent_tokens = parse_recent_window_tokens(_ACTIVE_MODE)
                if recent_tokens is None:
                    raise RuntimeError(f"Invalid recent-window mode: {_ACTIVE_MODE}")
                history_final = _recent_history_keep_for_query(
                    finite,
                    recent_tokens,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
            elif mode_kind in {"qabs_candidate_keep", "qabs_partial_rerank"}:
                qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
                if qabs_params is None:
                    raise RuntimeError(f"Invalid qabs mode: {_ACTIVE_MODE}")
                dim_count, candidate_fraction = qabs_params
                partial_candidate = _qabs_partial_candidate_keep_for_query(
                    query_states,
                    key_states,
                    query_index,
                    finite,
                    dim_count,
                    candidate_fraction,
                )
                if mode_kind == "qabs_candidate_keep":
                    history_final = partial_candidate
                else:
                    history_final = _topk_within_candidate_for_query(
                        row,
                        finite,
                        partial_candidate,
                        _ACTIVE_TOP_FRACTION,
                    )
                history_final = _apply_full_head_protection(
                    history_final,
                    finite,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                    _ACTIVE_PROTECT_RECENT_TOKENS,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
            elif mode_kind == "qabs_reuse_rerank":
                qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
                if qabs_params is None:
                    raise RuntimeError(f"Invalid qabs reuse-rerank mode: {_ACTIVE_MODE}")
                dim_count, candidate_fraction = qabs_params
                layer_idx = int(getattr(module, "layer_idx", 0))
                valid_count = int(finite[0, 0].sum().item())
                history_count = max(0, valid_count - 1)
                query_token = chunk_query_start + query_index
                current_candidate = _qabs_partial_candidate_keep_for_query(
                    query_states,
                    key_states,
                    query_index,
                    finite,
                    dim_count,
                    candidate_fraction,
                )
                candidate_union = current_candidate.clone()
                if _ACTIVE_REUSE_STATE is not None and history_count > 0:
                    for head in range(candidate_union.shape[1]):
                        previous_candidate, previous_final = _ACTIVE_REUSE_STATE.previous_masks(
                            layer_idx,
                            head,
                            query_token,
                            history_count,
                            candidate_union.device,
                        )
                        if previous_candidate is not None and qabs_reuse_uses_previous_candidate(_ACTIVE_MODE):
                            candidate_union[0, head, :history_count] |= previous_candidate
                        if previous_final is not None and qabs_reuse_uses_previous_final(_ACTIVE_MODE):
                            candidate_union[0, head, :history_count] |= previous_final
                history_final = _topk_within_candidate_for_query(
                    row,
                    finite,
                    candidate_union,
                    _ACTIVE_TOP_FRACTION,
                )
                history_final = _apply_full_head_protection(
                    history_final,
                    finite,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                    _ACTIVE_PROTECT_RECENT_TOKENS,
                )
                if _ACTIVE_CANDIDATE_STATS is not None:
                    history_valid = finite.clone()
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_CANDIDATE_STATS.update(layer_idx, candidate_union, history_valid)
                if _ACTIVE_LOAD_STATS is not None:
                    history_valid = finite.clone()
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
                if _ACTIVE_REUSE_STATE is not None and history_count > 0:
                    _ACTIVE_REUSE_STATE.update(layer_idx, query_token, current_candidate, history_final, history_count)
            elif mode_kind == "top2":
                history_final = _apply_full_head_protection(
                    history_original,
                    finite,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                    _ACTIVE_PROTECT_RECENT_TOKENS,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "top2_union_all":
                history_final = _expand_selected_tokens_to_all_heads(history_original)
                history_final = _apply_full_head_protection(
                    history_final,
                    finite,
                    _ACTIVE_PROTECT_SINK_TOKENS,
                    _ACTIVE_PROTECT_RECENT_TOKENS,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_random":
                history_final = _limit_heads_per_token_random(
                    history_original,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score":
                history_final = _limit_heads_per_token_by_score(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_fill":
                layer_idx = int(getattr(module, "layer_idx", 0))
                history_valid = finite.clone()
                valid_count = int(finite[0, 0].sum().item())
                if valid_count > 0:
                    history_valid[:, :, valid_count - 1] = False
                history_final = _limit_heads_per_token_by_score_fill(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    history_valid,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_gap":
                history_final = _limit_heads_per_token_by_score_gap(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    parse_gap_margin(_ACTIVE_MODE) or 0.0,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "limit_score_protect":
                sink_tokens, recent_fraction = parse_protect_params(_ACTIVE_MODE) or (0, 0.0)
                protected_keys = _protected_history_keys(finite, sink_tokens, recent_fraction)
                history_final = _limit_heads_per_token_by_score_protected(
                    history_original,
                    row,
                    mode_max_heads if mode_max_heads is not None else _ACTIVE_MAX_HEADS_PER_TOKEN,
                    protected_keys,
                )
                if _ACTIVE_LOAD_STATS is not None:
                    layer_idx = int(getattr(module, "layer_idx", 0))
                    history_valid = finite.clone()
                    valid_count = int(finite[0, 0].sum().item())
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "observation_window":
                if _ACTIVE_OBS_STATE is None:
                    raise RuntimeError("Observation-window mode requires an active ObservationWindowState.")
                layer_idx = int(getattr(module, "layer_idx", 0))
                valid_count = int(finite[0, 0].sum().item())
                query_token = chunk_query_start + query_index
                _ACTIVE_OBS_STATE.maybe_reallocate(layer_idx, query_token)
                history_final = _ACTIVE_OBS_STATE.apply(layer_idx, query_token, history_original)
                _ACTIVE_OBS_STATE.observe(layer_idx, history_original)
                if _ACTIVE_LOAD_STATS is not None:
                    history_valid = finite.clone()
                    if valid_count > 0:
                        history_valid[:, :, valid_count - 1] = False
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_original, history_final, history_valid)
            elif mode_kind == "observation_all_mass":
                if _ACTIVE_OBS_MASS_STATE is None:
                    raise RuntimeError("obsallmass mode requires an active AllTokenMassObservationState.")
                layer_idx = int(getattr(module, "layer_idx", 0))
                valid_count = int(finite[0, 0].sum().item())
                query_token = chunk_query_start + query_index
                history_valid = finite.clone()
                if valid_count > 0:
                    history_valid[:, :, valid_count - 1] = False
                _ACTIVE_OBS_MASS_STATE.maybe_reallocate(layer_idx, query_token)
                full_weights = F.softmax(row, dim=-1, dtype=torch.float32)
                _ACTIVE_OBS_MASS_STATE.observe(layer_idx, query_token, full_weights)
                history_final = _ACTIVE_OBS_MASS_STATE.apply(layer_idx, query_token, history_valid)
                if _ACTIVE_LOAD_STATS is not None:
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
            elif mode_kind == "observation_hybrid":
                if _ACTIVE_OBS_HYBRID_STATE is None:
                    raise RuntimeError("obshybrid mode requires an active HybridObservationState.")
                layer_idx = int(getattr(module, "layer_idx", 0))
                valid_count = int(finite[0, 0].sum().item())
                query_token = chunk_query_start + query_index
                history_valid = finite.clone()
                if valid_count > 0:
                    history_valid[:, :, valid_count - 1] = False
                _ACTIVE_OBS_HYBRID_STATE.maybe_reallocate(layer_idx, query_token)
                full_weights = F.softmax(row, dim=-1, dtype=torch.float32)
                _ACTIVE_OBS_HYBRID_STATE.observe(layer_idx, query_token, full_weights, history_original)
                history_final = _ACTIVE_OBS_HYBRID_STATE.apply(layer_idx, query_token, history_valid)
                if _ACTIVE_LOAD_STATS is not None:
                    _ACTIVE_LOAD_STATS.update(layer_idx, history_valid, history_final, history_valid)
            else:
                history_final = history_original
            keep = history_final
            if _ACTIVE_ALWAYS_KEEP_SELF:
                valid_count = int(finite[0, 0].sum().item())
                if valid_count > 0:
                    keep[:, :, valid_count - 1] = True
            final_keep[:, :, query_index, :] = keep
        scores = scores.masked_fill(~final_keep, torch.finfo(scores.dtype).min)

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
        setattr(modeling_qwen3, "eager_attention_forward", _limited_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _limited_eager_attention_forward


@contextmanager
def attention_mode(
    mode: str,
    top_fraction: float,
    max_heads_per_token: int,
    always_keep_self: bool,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    load_stats: LoadStats | None,
    obs_state: ObservationWindowState | None = None,
    obs_mass_state: AllTokenMassObservationState | None = None,
    obs_hybrid_state: HybridObservationState | None = None,
    reuse_state: ReuseCandidateState | None = None,
    knorm_state: KNormState | None = None,
    block_route_state: BlockRouteState | None = None,
    ksign_state: KSignIndexState | None = None,
    kdom_state: KDomIndexState | None = None,
    brep_state: BlockRepState | None = None,
    sparq_mean_state: SparQMeanState | None = None,
    candidate_stats: CandidateStats | None = None,
    reuse_overlap_stats: ReuseOverlapStats | None = None,
    qabs_fast_path: bool = False,
    qabs_cuda_final_kernel: bool = False,
    qabs_cuda_candidate_kernel: bool = False,
    qabs_cuda_reuse_select_kernel: bool = False,
    qabs_profile_stats: QabsReuseProfileStats | None = None,
    qabs_candidate_selection: str = "topk",
    qabs_threshold_sample_size: int = 256,
    full_head_map_path: str = "",
    full_layer_map_path: str = "",
    layer_budget_map_path: str = "",
):
    global _ACTIVE_MODE, _ACTIVE_TOP_FRACTION, _ACTIVE_MAX_HEADS_PER_TOKEN, _ACTIVE_ALWAYS_KEEP_SELF, _ACTIVE_PROTECT_SINK_TOKENS, _ACTIVE_PROTECT_RECENT_TOKENS, _ACTIVE_LOAD_STATS, _ACTIVE_OBS_STATE, _ACTIVE_OBS_MASS_STATE, _ACTIVE_OBS_HYBRID_STATE, _ACTIVE_REUSE_STATE, _ACTIVE_KNORM_STATE, _ACTIVE_BLOCK_ROUTE_STATE, _ACTIVE_KSIGN_STATE, _ACTIVE_KDOM_STATE, _ACTIVE_BREP_STATE, _ACTIVE_SPARQ_MEAN_STATE, _ACTIVE_CANDIDATE_STATS, _ACTIVE_REUSE_OVERLAP_STATS, _ACTIVE_QABS_FAST_PATH, _ACTIVE_QABS_CUDA_FINAL_KERNEL, _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL, _ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL, _ACTIVE_QABS_PROFILE_STATS, _ACTIVE_QABS_CANDIDATE_SELECTION, _ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE, _ACTIVE_FULL_HEAD_MAP_PATH, _ACTIVE_FULL_LAYER_MAP_PATH, _ACTIVE_LAYER_BUDGET_MAP_PATH, _LAYER_BUDGET_QABS_REUSE_STATE
    previous = (
        _ACTIVE_MODE,
        _ACTIVE_TOP_FRACTION,
        _ACTIVE_MAX_HEADS_PER_TOKEN,
        _ACTIVE_ALWAYS_KEEP_SELF,
        _ACTIVE_PROTECT_SINK_TOKENS,
        _ACTIVE_PROTECT_RECENT_TOKENS,
        _ACTIVE_LOAD_STATS,
        _ACTIVE_OBS_STATE,
        _ACTIVE_OBS_MASS_STATE,
        _ACTIVE_OBS_HYBRID_STATE,
        _ACTIVE_REUSE_STATE,
        _ACTIVE_KNORM_STATE,
        _ACTIVE_BLOCK_ROUTE_STATE,
        _ACTIVE_KSIGN_STATE,
        _ACTIVE_KDOM_STATE,
        _ACTIVE_BREP_STATE,
        _ACTIVE_SPARQ_MEAN_STATE,
        _ACTIVE_CANDIDATE_STATS,
        _ACTIVE_REUSE_OVERLAP_STATS,
        _ACTIVE_QABS_FAST_PATH,
        _ACTIVE_QABS_CUDA_FINAL_KERNEL,
        _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL,
        _ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL,
        _ACTIVE_QABS_PROFILE_STATS,
        _ACTIVE_QABS_CANDIDATE_SELECTION,
        _ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE,
        _ACTIVE_FULL_HEAD_MAP_PATH,
        _ACTIVE_FULL_LAYER_MAP_PATH,
        _ACTIVE_LAYER_BUDGET_MAP_PATH,
    )
    _ACTIVE_MODE = mode
    _ACTIVE_TOP_FRACTION = top_fraction
    _ACTIVE_MAX_HEADS_PER_TOKEN = max_heads_per_token
    _ACTIVE_ALWAYS_KEEP_SELF = always_keep_self
    _ACTIVE_PROTECT_SINK_TOKENS = max(0, protect_sink_tokens)
    _ACTIVE_PROTECT_RECENT_TOKENS = max(0, protect_recent_tokens)
    _ACTIVE_LOAD_STATS = load_stats
    _ACTIVE_OBS_STATE = obs_state
    _ACTIVE_OBS_MASS_STATE = obs_mass_state
    _ACTIVE_OBS_HYBRID_STATE = obs_hybrid_state
    _ACTIVE_REUSE_STATE = reuse_state
    _ACTIVE_KNORM_STATE = knorm_state
    _ACTIVE_BLOCK_ROUTE_STATE = block_route_state
    _ACTIVE_KSIGN_STATE = ksign_state
    _ACTIVE_KDOM_STATE = kdom_state
    _ACTIVE_BREP_STATE = brep_state
    _ACTIVE_SPARQ_MEAN_STATE = sparq_mean_state
    _ACTIVE_CANDIDATE_STATS = candidate_stats
    _ACTIVE_REUSE_OVERLAP_STATS = reuse_overlap_stats
    _ACTIVE_QABS_FAST_PATH = qabs_fast_path
    _ACTIVE_QABS_CUDA_FINAL_KERNEL = qabs_cuda_final_kernel
    _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL = qabs_cuda_candidate_kernel
    _ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL = qabs_cuda_reuse_select_kernel
    _ACTIVE_QABS_PROFILE_STATS = qabs_profile_stats
    _ACTIVE_QABS_CANDIDATE_SELECTION = qabs_candidate_selection
    _ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE = max(1, qabs_threshold_sample_size)
    _ACTIVE_FULL_HEAD_MAP_PATH = full_head_map_path
    _ACTIVE_FULL_LAYER_MAP_PATH = full_layer_map_path
    _ACTIVE_LAYER_BUDGET_MAP_PATH = layer_budget_map_path
    if mode == "layerbudgetattn":
        _LAYER_BUDGET_QABS_REUSE_STATE = {}
    try:
        yield
    finally:
        (
            _ACTIVE_MODE,
            _ACTIVE_TOP_FRACTION,
            _ACTIVE_MAX_HEADS_PER_TOKEN,
            _ACTIVE_ALWAYS_KEEP_SELF,
            _ACTIVE_PROTECT_SINK_TOKENS,
            _ACTIVE_PROTECT_RECENT_TOKENS,
            _ACTIVE_LOAD_STATS,
            _ACTIVE_OBS_STATE,
            _ACTIVE_OBS_MASS_STATE,
            _ACTIVE_OBS_HYBRID_STATE,
            _ACTIVE_REUSE_STATE,
            _ACTIVE_KNORM_STATE,
            _ACTIVE_BLOCK_ROUTE_STATE,
            _ACTIVE_KSIGN_STATE,
            _ACTIVE_KDOM_STATE,
            _ACTIVE_BREP_STATE,
            _ACTIVE_SPARQ_MEAN_STATE,
            _ACTIVE_CANDIDATE_STATS,
            _ACTIVE_REUSE_OVERLAP_STATS,
            _ACTIVE_QABS_FAST_PATH,
            _ACTIVE_QABS_CUDA_FINAL_KERNEL,
            _ACTIVE_QABS_CUDA_CANDIDATE_KERNEL,
            _ACTIVE_QABS_CUDA_REUSE_SELECT_KERNEL,
            _ACTIVE_QABS_PROFILE_STATS,
            _ACTIVE_QABS_CANDIDATE_SELECTION,
            _ACTIVE_QABS_THRESHOLD_SAMPLE_SIZE,
            _ACTIVE_FULL_HEAD_MAP_PATH,
            _ACTIVE_FULL_LAYER_MAP_PATH,
            _ACTIVE_LAYER_BUDGET_MAP_PATH,
        ) = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
    past_key_values = None
    last_logits: torch.Tensor | None = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        print(f"prefill chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        del outputs, chunk
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    if last_logits is None:
        raise RuntimeError("Prefill produced no logits.")
    return past_key_values, last_logits


def clone_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if torch.is_tensor(past_key_values):
        return past_key_values.detach().clone()
    if isinstance(past_key_values, tuple):
        return tuple(clone_past_key_values(item) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [clone_past_key_values(item) for item in past_key_values]
    if isinstance(past_key_values, dict):
        return {key: clone_past_key_values(value) for key, value in past_key_values.items()}
    to_legacy_cache = getattr(past_key_values, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(past_key_values), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        legacy_cache = clone_past_key_values(to_legacy_cache())
        return from_legacy_cache(legacy_cache)
    return copy.deepcopy(past_key_values)


@torch.inference_mode()
def compute_eval_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    prefill_chunk_size: int,
    eval_chunk_size: int,
    input_device: torch.device,
    mode: str,
    top_fraction: float,
    max_heads_per_token: int,
    always_keep_self: bool,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    load_stats: LoadStats | None,
    obs_state: ObservationWindowState | None = None,
    obs_mass_state: AllTokenMassObservationState | None = None,
    obs_hybrid_state: HybridObservationState | None = None,
    reuse_state: ReuseCandidateState | None = None,
    knorm_state: KNormState | None = None,
    block_route_state: BlockRouteState | None = None,
    ksign_state: KSignIndexState | None = None,
    kdom_state: KDomIndexState | None = None,
    brep_state: BlockRepState | None = None,
    sparq_mean_state: SparQMeanState | None = None,
    candidate_stats: CandidateStats | None = None,
    reuse_overlap_stats: ReuseOverlapStats | None = None,
    qabs_fast_path: bool = False,
    qabs_cuda_final_kernel: bool = False,
    qabs_cuda_candidate_kernel: bool = False,
    qabs_cuda_reuse_select_kernel: bool = False,
    qabs_profile_stats: QabsReuseProfileStats | None = None,
    qabs_candidate_selection: str = "topk",
    qabs_threshold_sample_size: int = 256,
    full_head_map_path: str = "",
    full_layer_map_path: str = "",
    layer_budget_map_path: str = "",
    adaptive_budget_stats: dict[str, Any] | None = None,
    initial_past_key_values: Any | None = None,
    initial_prev_logits: torch.Tensor | None = None,
    clone_initial_cache: bool = True,
    log_every: int = 1,
) -> tuple[float, float, int, float]:
    print(f"starting mode: {mode}", flush=True)
    started = time.perf_counter()
    if initial_past_key_values is None or initial_prev_logits is None:
        past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, prefill_chunk_size, input_device)
    elif clone_initial_cache:
        print(f"cloning shared prefill cache for mode: {mode}", flush=True)
        past_key_values = clone_past_key_values(initial_past_key_values)
        prev_logits = initial_prev_logits.detach().clone()
    else:
        print(f"using shared prefill cache without clone for single mode: {mode}", flush=True)
        past_key_values = initial_past_key_values
        prev_logits = initial_prev_logits
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / eval_chunk_size)
    adaptive_full_layer_params = parse_adaptive_full_layer_recent_params(mode)
    if adaptive_full_layer_params is not None and adaptive_budget_stats is not None:
        adaptive_budget_stats.clear()
        adaptive_budget_stats.update(
            {
                "low_tokens": 0,
                "high_tokens": 0,
                "margin_sum": 0.0,
                "margin_count": 0,
                "margin_threshold": adaptive_full_layer_params[2],
            }
        )
    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, eval_chunk_size), start=1):
        end = min(start + eval_chunk_size, eval_end)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if log_every <= 1 or chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % log_every == 0:
            print(f"ppl {mode} chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
        attention_eval_mode = mode
        if adaptive_full_layer_params is not None:
            if eval_chunk_size != 1:
                raise RuntimeError("adaptive full-layer mode requires --eval_chunk_size 1.")
            low_layers, high_layers, margin_threshold, recent_tokens = adaptive_full_layer_params
            top2_logits = torch.topk(prev_logits.float(), k=2, dim=-1).values
            margin = float((top2_logits[:, 0] - top2_logits[:, 1]).mean())
            chosen_layers = low_layers if margin >= margin_threshold else high_layers
            if adaptive_budget_stats is not None:
                if chosen_layers == low_layers:
                    adaptive_budget_stats["low_tokens"] += 1
                else:
                    adaptive_budget_stats["high_tokens"] += 1
                adaptive_budget_stats["margin_sum"] += margin
                adaptive_budget_stats["margin_count"] += 1
            attention_eval_mode = f"fulll{chosen_layers}recent{recent_tokens}attn"
        with attention_mode(
            attention_eval_mode,
            top_fraction,
            max_heads_per_token,
            always_keep_self,
            protect_sink_tokens,
            protect_recent_tokens,
            load_stats,
            obs_state,
            obs_mass_state,
            obs_hybrid_state,
            reuse_state,
            knorm_state,
            block_route_state,
            ksign_state,
            kdom_state,
            brep_state,
            sparq_mean_state,
            candidate_stats,
            reuse_overlap_stats,
            qabs_fast_path,
            qabs_cuda_final_kernel,
            qabs_cuda_candidate_kernel,
            qabs_cuda_reuse_select_kernel,
            qabs_profile_stats,
            qabs_candidate_selection,
            qabs_threshold_sample_size,
            full_head_map_path,
            full_layer_map_path,
            layer_budget_map_path,
        ):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([prev_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            chunk.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_count += int(chunk.numel())
        prev_logits = logits[:, -1, :].detach()
        past_key_values = outputs.past_key_values
        del outputs, chunk, logits, shifted_logits, loss
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, math.exp(min(mean_loss, 80.0)), total_count, time.perf_counter() - started


def plot_outputs(output_dir: Path, ppl_rows: list[dict[str, Any]], load_rows: list[dict[str, Any]], dpi: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if ppl_rows:
        labels = [str(row["mode"]) for row in ppl_rows]
        ppls = [float(row["ppl"]) for row in ppl_rows]
        losses = [float(row["loss"]) for row in ppl_rows]
        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
        bars = ax.bar(labels, ppls, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_title("PPL by attention selection mode")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Perplexity on evaluation tokens")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "ppl_by_mode.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        bars = ax.bar(labels, ppls, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_yscale("log")
        ax.set_title("PPL by attention selection mode (log scale)")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Perplexity on evaluation tokens, log scale")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        for bar, value in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "ppl_by_mode_logy.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
        bars = ax.bar(labels, losses, color=[colors[index % len(colors)] for index in range(len(labels))])
        ax.set_title("Mean next-token loss by attention selection mode")
        ax.set_xlabel("Attention selection mode")
        ax.set_ylabel("Mean cross entropy loss")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        path = plot_dir / "loss_by_mode.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    if load_rows:
        focus_mode = "top2limit3score" if any(row.get("mode") == "top2limit3score" for row in load_rows) else str(load_rows[0]["mode"])
        focus_rows = [row for row in load_rows if row.get("mode") == focus_mode]
        layers = sorted({int(row["layer"]) for row in focus_rows})
        heads = sorted({int(row["head"]) for row in focus_rows})
        matrix = torch.zeros((len(layers), len(heads)), dtype=torch.float32)
        kept_fraction = torch.zeros((len(layers), len(heads)), dtype=torch.float32)
        layer_to_pos = {layer: pos for pos, layer in enumerate(layers)}
        head_to_pos = {head: pos for pos, head in enumerate(heads)}
        for row in focus_rows:
            layer = int(row["layer"])
            head = int(row["head"])
            matrix[layer_to_pos[layer], head_to_pos[head]] = float(row["final_kept_per_query_mean"])
            kept_fraction[layer_to_pos[layer], head_to_pos[head]] = float(row["kept_fraction_of_original_top2"])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        image = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"{focus_mode} historical-token load per layer and head")
        ax.set_xlabel("Attention head index")
        ax.set_ylabel("Layer index")
        ax.set_xticks(heads)
        ax.set_yticks(layers)
        fig.colorbar(image, ax=ax, label="Mean kept historical tokens per query")
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_head_load_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        layer_means = matrix.mean(dim=1).tolist()
        layer_mins = matrix.min(dim=1).values.tolist()
        layer_maxs = matrix.max(dim=1).values.tolist()
        fig, ax = plt.subplots(figsize=(11, 4), dpi=dpi)
        ax.plot(layers, layer_means, marker="o", label="mean over heads")
        ax.fill_between(layers, layer_mins, layer_maxs, alpha=0.2, label="min-max over heads")
        ax.set_title(f"{focus_mode} head load by layer")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean kept historical tokens per query")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_head_load_by_layer.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        image = ax.imshow(kept_fraction.numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Fraction of original top2 selections kept after {focus_mode}")
        ax.set_xlabel("Attention head index")
        ax.set_ylabel("Layer index")
        ax.set_xticks(heads)
        ax.set_yticks(layers)
        fig.colorbar(image, ax=ax, label="Final kept / original top2 kept")
        fig.tight_layout()
        path = plot_dir / f"{focus_mode}_kept_fraction_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.prefill_tokens <= 0 or args.eval_tokens <= 0:
        raise ValueError("--prefill_tokens and --eval_tokens must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    eval_chunk_size = args.eval_chunk_size if args.eval_chunk_size is not None else args.chunk_size
    if eval_chunk_size <= 0:
        raise ValueError("--eval_chunk_size must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log_every must be positive.")
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.max_heads_per_token <= 0:
        raise ValueError("--max_heads_per_token must be positive.")
    modes = parse_modes(args.modes)
    if args.baseline_last:
        modes = move_baseline_last(modes)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    total_needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < total_needed:
        raise ValueError(f"Tokenization produced {len(token_ids)} tokens, fewer than {total_needed}.")
    token_ids = token_ids[:total_needed]
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()

    layer_count = int(getattr(model.config, "num_hidden_layers"))
    head_count = int(getattr(model.config, "num_attention_heads"))
    input_device = pick_input_device(model, requested_device)

    shared_prefill_seconds = ""
    shared_past_key_values: Any | None = None
    shared_prev_logits: torch.Tensor | None = None
    if args.reuse_prefill_cache:
        print("starting shared prefill cache", flush=True)
        prefill_started = time.perf_counter()
        shared_past_key_values, shared_prev_logits = prefill_cache(
            model,
            input_ids,
            args.prefill_tokens,
            args.chunk_size,
            input_device,
        )
        shared_prefill_seconds = time.perf_counter() - prefill_started
        print(f"shared prefill cache ready: {shared_prefill_seconds:.3f}s", flush=True)

    ppl_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_kind, mode_max_heads = parse_mode_config(mode)
        protect_params = parse_protect_params(mode)
        sign_xnor_candidate_fraction = (
            parse_sign_xnor_candidate_fraction(mode)
            if mode_kind in {"sign_xnor", "sign_xnor_knorm", "sign_xnor_rerank"}
            else None
        )
        landmark_params = parse_landmark_recent_params(mode) if mode_kind == "landmark_recent_attention" else None
        full_head_params = parse_full_head_recent_params(mode) if mode_kind == "full_head_recent_attention" else None
        full_layer_params = parse_full_layer_recent_params(mode) if mode_kind == "full_layer_recent_attention" else None
        full_layer_landmark_params = (
            parse_full_layer_landmark_params(mode) if mode_kind == "full_layer_landmark_attention" else None
        )
        adaptive_full_layer_params = (
            parse_adaptive_full_layer_recent_params(mode)
            if mode_kind == "adaptive_full_layer_recent_attention"
            else None
        )
        recent_window_tokens = (
            parse_recent_window_tokens(mode)
            if mode_kind == "recent_window"
            else landmark_params[0]
            if landmark_params is not None
            else full_head_params[1]
            if full_head_params is not None
            else full_layer_params[1]
            if full_layer_params is not None
            else full_layer_landmark_params[1]
            if full_layer_landmark_params is not None
            else adaptive_full_layer_params[3]
            if adaptive_full_layer_params is not None
            else None
        )
        kdom_params = parse_kdom_params(mode) if mode_kind == "kdom_index_rerank" else None
        qbb_params = parse_qbb_params(mode) if mode_kind == "qabs_block_bound" else None
        kdim_params = parse_kdim_params(mode) if mode_kind == "kdim_posting_rerank" else None
        brep_params = parse_brep_params(mode) if mode_kind == "block_rep_rerank" else None
        bgrp_params = parse_bgrp_params(mode) if mode_kind == "block_group_rep_rerank" else None
        brp_params = parse_brp_params(mode) if mode_kind == "block_random_rep_rerank" else None
        qboracle_params = parse_qboracle_params(mode) if mode_kind == "qabs_block_oracle" else None
        qabs_params = (
            parse_sparq_params(mode)
            if mode_kind in {"sparq_attention", "sparq_fast_attention"}
            else parse_ksign_params(mode)
            if mode_kind == "ksign_index_rerank"
            else (kdom_params[0], kdom_params[2])
            if kdom_params is not None
            else (qbb_params[1], qbb_params[3])
            if qbb_params is not None
            else kdim_params
            if kdim_params is not None
            else (brep_params[2], brep_params[4])
            if brep_params is not None
            else (bgrp_params[3], bgrp_params[5])
            if bgrp_params is not None
            else (brp_params[3], brp_params[5])
            if brp_params is not None
            else (qboracle_params[1], qboracle_params[3])
            if qboracle_params is not None
            else parse_qabs_partial_rerank_params(mode)
            if mode_kind
            in {
                "qabs_candidate_keep",
                "qabs_partial_rerank",
                "qabs_reuse_rerank",
                "qabs_candidate_attention",
                "qabs_periodic_candidate_attention",
                "qabs_candidate_top2_attention",
                "qabs_candidate_top2_hlimit_attention",
                "qabs_candidate_top2_global_attention",
                "qabs_shared_candidate_attention",
                "qabs_shared_candidate_top2_attention",
                "qabs_shared_reuse_attention",
                "lagged_reuse_rerank",
            }
            else None
        )
        qabs_dim_count = qabs_params[0] if qabs_params else None
        qabs_candidate_fraction = qabs_params[1] if qabs_params else None
        kdom_key_dim_count = kdom_params[1] if kdom_params else None
        stats = (
            LoadStats(layer_count, head_count)
            if not args.disable_sparse_stats
            and mode_kind
            in {
                "top2",
                "top2_union_all",
                "limit_random",
                "limit_score",
                "limit_score_fill",
                "limit_score_gap",
                "limit_score_protect",
                "observation_window",
                "observation_all_mass",
                "observation_hybrid",
                "sign_xnor",
                "sign_xnor_knorm",
                "sign_xnor_rerank",
                "recent_window",
                "qabs_candidate_keep",
                "qabs_partial_rerank",
                "qabs_reuse_rerank",
            "lagged_reuse_rerank",
                "sparq_attention",
                "sparq_fast_attention",
            }
            else None
        )
        candidate_stats = (
            CandidateStats(layer_count, head_count)
            if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"} and not args.disable_sparse_stats
            else None
        )
        reuse_overlap_stats = (
            ReuseOverlapStats(layer_count, head_count) if mode_kind == "qabs_reuse_rerank" and args.qabs_overlap_stats else None
        )
        qabs_profile_stats = (
            QabsReuseProfileStats(layer_count, head_count)
            if args.qabs_profile
            and mode_kind
            in {
                "qabs_reuse_rerank",
                "lagged_reuse_rerank",
                "qabs_candidate_attention",
                "qabs_periodic_candidate_attention",
                "qabs_candidate_top2_attention",
                "qabs_candidate_top2_hlimit_attention",
                "qabs_candidate_top2_global_attention",
                "qabs_shared_candidate_attention",
                "qabs_shared_candidate_top2_attention",
            }
            else None
        )
        reuse_state = (
            ReuseCandidateState()
            if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank", "qabs_shared_reuse_attention", "qabs_periodic_candidate_attention"}
            else None
        )
        knorm_state = KNormState() if mode_kind == "knorm_reservoir" else None
        block_route_state = BlockRouteState() if mode_kind == "block_route" else None
        ksign_state = KSignIndexState() if mode_kind == "ksign_index_rerank" else None
        kdom_state = KDomIndexState() if mode_kind == "kdom_index_rerank" else None
        brep_state = BlockRepState() if mode_kind == "block_rep_rerank" else None
        sparq_mean_state = (
            SparQMeanState()
            if mode_kind == "sparq_fast_attention"
            or (
                mode_kind
                in {
                    "qabs_reuse_rerank",
                    "qabs_candidate_attention",
                    "qabs_periodic_candidate_attention",
                    "qabs_candidate_top2_attention",
                    "qabs_candidate_top2_hlimit_attention",
                    "qabs_candidate_top2_global_attention",
                    "qabs_shared_candidate_attention",
                    "qabs_shared_candidate_top2_attention",
                    "qabs_shared_reuse_attention",
                    "lagged_reuse_rerank",
                }
                and args.qabs_fast_path
                and args.qabs_cuda_candidate_kernel
            )
            else None
        )
        adaptive_budget_stats: dict[str, Any] | None = (
            {} if mode_kind == "adaptive_full_layer_recent_attention" else None
        )
        hybrid_params = parse_hybrid_params(mode) if mode_kind == "observation_hybrid" else None
        obs_state = (
            ObservationWindowState(
                layer_count=layer_count,
                head_count=head_count,
                window_tokens=args.obs_window_tokens,
                recent_tokens=args.obs_recent_tokens,
                target_coverage=args.obs_target_coverage,
                min_heads=args.obs_min_heads,
                max_heads=args.obs_max_heads,
                fallback_all=args.obs_fallback_all,
            )
            if mode_kind == "observation_window"
            else None
        )
        obs_mass_state = (
            AllTokenMassObservationState(
                layer_count=layer_count,
                head_count=head_count,
                window_tokens=args.obs_window_tokens,
                recent_tokens=args.obs_recent_tokens,
                target_coverage=args.obs_target_coverage,
                min_heads=args.obs_min_heads,
                max_heads=args.obs_max_heads,
                fallback_all=args.obs_fallback_all,
            )
            if mode_kind == "observation_all_mass"
            else None
        )
        obs_hybrid_state = (
            HybridObservationState(
                layer_count=layer_count,
                head_count=head_count,
                window_tokens=args.obs_window_tokens,
                recent_tokens=args.obs_recent_tokens,
                top2_use_mass=bool(hybrid_params["top2_use_mass"]) if hybrid_params else False,
                top2_target_coverage=hybrid_params["top2_target"] if hybrid_params else 0.95,
                non_top2_mass_coverage=hybrid_params["non_top2_target"] if hybrid_params else 0.80,
                top2_min_heads=hybrid_params["top2_min_heads"] if hybrid_params else args.obs_min_heads,
                top2_max_heads=hybrid_params["top2_max_heads"] if hybrid_params else 16,
                non_top2_min_heads=hybrid_params["non_top2_min_heads"] if hybrid_params else args.obs_min_heads,
                non_top2_max_heads=hybrid_params["non_top2_max_heads"] if hybrid_params else 6,
                top2_full_heads=bool(hybrid_params["top2_full_heads"]) if hybrid_params else False,
                sink_tokens=args.protect_sink_tokens,
                fallback_all=args.obs_fallback_all,
            )
            if mode_kind == "observation_hybrid"
            else None
        )
        loss, ppl, token_count, seconds = compute_eval_loss(
            model,
            input_ids,
            args.prefill_tokens,
            args.eval_tokens,
            args.chunk_size,
            eval_chunk_size,
            input_device,
            mode,
            args.top_fraction,
            mode_max_heads if mode_max_heads is not None else args.max_heads_per_token,
            args.always_keep_self,
            args.protect_sink_tokens,
            args.protect_recent_tokens,
            stats,
            obs_state,
            obs_mass_state,
            obs_hybrid_state,
            reuse_state,
            knorm_state,
            block_route_state,
            ksign_state,
            kdom_state,
            brep_state,
            sparq_mean_state,
            candidate_stats,
            reuse_overlap_stats,
            args.qabs_fast_path,
            args.qabs_cuda_final_kernel,
            args.qabs_cuda_candidate_kernel,
            args.qabs_cuda_reuse_select_kernel,
            qabs_profile_stats,
            args.qabs_candidate_selection,
            args.qabs_threshold_sample_size,
            args.full_head_map_path,
            args.full_layer_map_path,
            args.layer_budget_map_path,
            adaptive_budget_stats,
            shared_past_key_values,
            shared_prev_logits,
            len(modes) > 1,
            args.log_every,
        )
        adaptive_low_tokens = (
            int(adaptive_budget_stats.get("low_tokens", 0)) if adaptive_budget_stats is not None else ""
        )
        adaptive_high_tokens = (
            int(adaptive_budget_stats.get("high_tokens", 0)) if adaptive_budget_stats is not None else ""
        )
        adaptive_total_tokens = (
            int(adaptive_low_tokens) + int(adaptive_high_tokens)
            if adaptive_budget_stats is not None
            else 0
        )
        adaptive_margin_count = (
            int(adaptive_budget_stats.get("margin_count", 0)) if adaptive_budget_stats is not None else 0
        )
        ppl_rows.append(
            {
                "mode": mode,
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "top_fraction": (
                    args.top_fraction
                    if mode not in {"baseline"}
                    and mode_kind
                    not in {
                        "sign_xnor",
                        "sign_xnor_knorm",
                        "sign_xnor_rerank",
                        "recent_window",
                        "landmark_recent_attention",
                        "full_head_recent_attention",
                        "full_layer_recent_attention",
                        "full_layer_landmark_attention",
                        "layer_budget_attention",
                        "adaptive_full_layer_recent_attention",
                        "qabs_candidate_keep",
                        "qabs_partial_rerank",
                        "qabs_reuse_rerank",
                        "qabs_periodic_candidate_attention",
            "lagged_reuse_rerank",
                        "sparq_attention",
                        "sparq_fast_attention",
                    }
                    else ""
                ),
                "sign_xnor_candidate_fraction": sign_xnor_candidate_fraction if sign_xnor_candidate_fraction else "",
                "recent_window_tokens": recent_window_tokens if recent_window_tokens else "",
                "qabs_dim_count": qabs_dim_count if qabs_dim_count else "",
                "qabs_candidate_fraction": qabs_candidate_fraction if qabs_candidate_fraction else "",
                "kdom_key_dim_count": kdom_key_dim_count if kdom_key_dim_count else "",
                "max_heads_per_token": (
                    mode_max_heads
                    if mode_kind
                    in {"limit_random", "limit_score", "limit_score_fill", "limit_score_gap", "limit_score_protect"}
                    else ""
                ),
                "limit_strategy": (
                    "random"
                    if mode_kind == "limit_random"
                    else "score"
                    if mode_kind == "limit_score"
                    else "score_fill"
                    if mode_kind == "limit_score_fill"
                    else "score_gap"
                    if mode_kind == "limit_score_gap"
                    else "score_protect"
                    if mode_kind == "limit_score_protect"
                    else "top2_union_all"
                    if mode_kind == "top2_union_all"
                    else "observation_window"
                    if mode_kind == "observation_window"
                    else "observation_all_mass"
                    if mode_kind == "observation_all_mass"
                    else "observation_hybrid"
                    if mode_kind == "observation_hybrid"
                    else "sign_xnor"
                    if mode_kind == "sign_xnor"
                    else "sign_xnor_knorm"
                    if mode_kind == "sign_xnor_knorm"
                    else "sign_xnor_rerank"
                    if mode_kind == "sign_xnor_rerank"
                    else "recent_window"
                    if mode_kind == "recent_window"
                    else "adaptive_full_layer_recent_attention"
                    if mode_kind == "adaptive_full_layer_recent_attention"
                    else "full_layer_landmark_attention"
                    if mode_kind == "full_layer_landmark_attention"
                    else "layer_budget_attention"
                    if mode_kind == "layer_budget_attention"
                    else "qabs_candidate_keep"
                    if mode_kind == "qabs_candidate_keep"
                    else "qabs_partial_rerank"
                    if mode_kind == "qabs_partial_rerank"
                    else "qabs_reuse_rerank"
                    if mode_kind == "qabs_reuse_rerank"
                    else "qabs_candidate_attention"
                    if mode_kind == "qabs_candidate_attention"
                    else "qabs_periodic_candidate_attention"
                    if mode_kind == "qabs_periodic_candidate_attention"
                    else "qabs_candidate_top2_attention"
                    if mode_kind == "qabs_candidate_top2_attention"
                    else "qabs_candidate_top2_hlimit_attention"
                    if mode_kind == "qabs_candidate_top2_hlimit_attention"
                    else "qabs_candidate_top2_global_attention"
                    if mode_kind == "qabs_candidate_top2_global_attention"
                    else "qabs_shared_candidate_attention"
                    if mode_kind == "qabs_shared_candidate_attention"
                    else "qabs_shared_candidate_top2_attention"
                    if mode_kind == "qabs_shared_candidate_top2_attention"
                    else "qabs_shared_reuse_attention"
                    if mode_kind == "qabs_shared_reuse_attention"
                    else "lagged_reuse_rerank"
                    if mode_kind == "lagged_reuse_rerank"
                    else "knorm_reservoir"
                    if mode_kind == "knorm_reservoir"
                    else "block_route"
                    if mode_kind == "block_route"
                    else "ksign_index_rerank"
                    if mode_kind == "ksign_index_rerank"
                    else "kdom_index_rerank"
                    if mode_kind == "kdom_index_rerank"
                    else "qabs_block_bound"
                    if mode_kind == "qabs_block_bound"
                    else "kdim_posting_rerank"
                    if mode_kind == "kdim_posting_rerank"
                    else "block_rep_rerank"
                    if mode_kind == "block_rep_rerank"
                    else "block_group_rep_rerank"
                    if mode_kind == "block_group_rep_rerank"
                    else "block_random_rep_rerank"
                    if mode_kind == "block_random_rep_rerank"
                    else "qabs_block_oracle"
                    if mode_kind == "qabs_block_oracle"
                    else "sparq_attention"
                    if mode_kind == "sparq_attention"
                    else "sparq_fast_attention"
                    if mode_kind == "sparq_fast_attention"
                    else ""
                ),
                "protected_sink_tokens": protect_params[0] if protect_params else args.protect_sink_tokens,
                "protected_recent_fraction": protect_params[1] if protect_params else "",
                "protected_recent_tokens": args.protect_recent_tokens,
                "always_keep_self": args.always_keep_self if mode != "baseline" else "",
                "qabs_fast_path": (
                    args.qabs_fast_path
                    if mode_kind
                    in {
                        "qabs_reuse_rerank",
                        "qabs_candidate_attention",
                        "qabs_periodic_candidate_attention",
                        "qabs_candidate_top2_attention",
                        "qabs_candidate_top2_hlimit_attention",
                        "qabs_candidate_top2_global_attention",
                        "qabs_shared_candidate_attention",
                        "qabs_shared_candidate_top2_attention",
                        "qabs_shared_reuse_attention",
                        "lagged_reuse_rerank",
                    }
                    else ""
                ),
                "qabs_cuda_final_kernel": (
                    args.qabs_cuda_final_kernel
                    if mode_kind
                    in {
                        "qabs_reuse_rerank",
                        "qabs_candidate_attention",
                        "qabs_periodic_candidate_attention",
                        "qabs_candidate_top2_attention",
                        "qabs_candidate_top2_hlimit_attention",
                        "qabs_candidate_top2_global_attention",
                        "qabs_shared_candidate_attention",
                        "qabs_shared_candidate_top2_attention",
                        "qabs_shared_reuse_attention",
                        "lagged_reuse_rerank",
                        "sparq_fast_attention",
                    }
                    else ""
                ),
                "qabs_cuda_candidate_kernel": (
                    args.qabs_cuda_candidate_kernel
                    if mode_kind
                    in {
                        "qabs_reuse_rerank",
                        "qabs_candidate_attention",
                        "qabs_periodic_candidate_attention",
                        "qabs_candidate_top2_attention",
                        "qabs_candidate_top2_hlimit_attention",
                        "qabs_candidate_top2_global_attention",
                        "qabs_shared_candidate_attention",
                        "qabs_shared_candidate_top2_attention",
                        "qabs_shared_reuse_attention",
                        "lagged_reuse_rerank",
                    }
                    else ""
                ),
                "qabs_cuda_reuse_select_kernel": (
                    args.qabs_cuda_reuse_select_kernel if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"} else ""
                ),
                "qabs_profile": args.qabs_profile if qabs_profile_stats is not None else "",
                "qabs_candidate_selection": (
                    args.qabs_candidate_selection if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"} else ""
                ),
                "qabs_threshold_sample_size": (
                    args.qabs_threshold_sample_size
                    if mode_kind in {"qabs_reuse_rerank", "lagged_reuse_rerank"}
                    and args.qabs_candidate_selection == "sample_quantile"
                    else ""
                ),
                "lagged_refresh_fraction": (
                    reuse_state.refresh_fraction() if mode_kind == "lagged_reuse_rerank" and reuse_state is not None else ""
                ),
                "full_head_map_path": (
                    args.full_head_map_path if mode_kind == "full_head_recent_attention" else ""
                ),
                "full_layer_map_path": (
                    args.full_layer_map_path
                    if mode_kind
                    in {
                        "full_layer_recent_attention",
                        "full_layer_landmark_attention",
                        "adaptive_full_layer_recent_attention",
                    }
                    else ""
                ),
                "layer_budget_map_path": (
                    args.layer_budget_map_path if mode_kind == "layer_budget_attention" else ""
                ),
                "adaptive_low_tokens": adaptive_low_tokens,
                "adaptive_high_tokens": adaptive_high_tokens,
                "adaptive_low_fraction": (
                    float(adaptive_low_tokens) / adaptive_total_tokens
                    if adaptive_total_tokens > 0
                    else ""
                ),
                "adaptive_mean_margin": (
                    float(adaptive_budget_stats.get("margin_sum", 0.0)) / adaptive_margin_count
                    if adaptive_budget_stats is not None and adaptive_margin_count > 0
                    else ""
                ),
                "adaptive_margin_threshold": (
                    adaptive_budget_stats.get("margin_threshold", "")
                    if adaptive_budget_stats is not None
                    else ""
                ),
                "reuse_prefill_cache": args.reuse_prefill_cache,
                "shared_prefill_seconds": shared_prefill_seconds,
            }
        )
        if stats is not None:
            for row in stats.rows():
                row = dict(row)
                row["mode"] = mode
                row["limit_strategy"] = (
                    "random"
                    if mode_kind == "limit_random"
                    else "score"
                    if mode_kind == "limit_score"
                    else "score_fill"
                    if mode_kind == "limit_score_fill"
                    else "score_gap"
                    if mode_kind == "limit_score_gap"
                    else "score_protect"
                    if mode_kind == "limit_score_protect"
                    else "top2_union_all"
                    if mode_kind == "top2_union_all"
                    else "observation_window"
                    if mode_kind == "observation_window"
                    else "observation_all_mass"
                    if mode_kind == "observation_all_mass"
                    else "observation_hybrid"
                    if mode_kind == "observation_hybrid"
                    else "sign_xnor"
                    if mode_kind == "sign_xnor"
                    else "sign_xnor_knorm"
                    if mode_kind == "sign_xnor_knorm"
                    else "sign_xnor_rerank"
                    if mode_kind == "sign_xnor_rerank"
                    else "recent_window"
                    if mode_kind == "recent_window"
                    else "qabs_candidate_keep"
                    if mode_kind == "qabs_candidate_keep"
                    else "qabs_partial_rerank"
                    if mode_kind == "qabs_partial_rerank"
                    else "qabs_reuse_rerank"
                    if mode_kind == "qabs_reuse_rerank"
                    else "lagged_reuse_rerank"
                    if mode_kind == "lagged_reuse_rerank"
                    else "knorm_reservoir"
                    if mode_kind == "knorm_reservoir"
                    else "block_route"
                    if mode_kind == "block_route"
                    else "ksign_index_rerank"
                    if mode_kind == "ksign_index_rerank"
                    else "sparq_attention"
                    if mode_kind == "sparq_attention"
                    else "sparq_fast_attention"
                    if mode_kind == "sparq_fast_attention"
                    else ""
                )
                row["max_heads_per_token"] = mode_max_heads
                row["protected_sink_tokens"] = protect_params[0] if protect_params else args.protect_sink_tokens
                row["protected_recent_fraction"] = protect_params[1] if protect_params else ""
                row["protected_recent_tokens"] = args.protect_recent_tokens
                load_rows.append(row)
        if candidate_stats is not None:
            write_csv(
                output_dir / f"{mode}_candidate_load_by_head.csv",
                [
                    {
                        **row,
                        "mode": mode,
                        "limit_strategy": "qabs_reuse_rerank",
                        "qabs_dim_count": qabs_dim_count if qabs_dim_count else "",
                        "qabs_candidate_fraction": qabs_candidate_fraction if qabs_candidate_fraction else "",
                    }
                    for row in candidate_stats.rows()
                ],
                [
                    "mode",
                    "limit_strategy",
                    "qabs_dim_count",
                    "qabs_candidate_fraction",
                    "layer",
                    "head",
                    "query_count",
                    "history_token_cases",
                    "candidate_tokens",
                    "candidate_per_query_mean",
                    "candidate_fraction_of_history",
                    "max_candidate_per_query",
                ],
            )
        if reuse_overlap_stats is not None:
            overlap_summary_fields = [
                "mode",
                "limit_strategy",
                "qabs_dim_count",
                "qabs_candidate_fraction",
                "summary",
                "layer",
                "query_count",
                "history_token_cases",
                "current_raw_per_query_mean",
                "previous_raw_per_query_mean",
                "previous_final_per_query_mean",
                "union_all_per_query_mean",
                "union_ab_per_query_mean",
                "union_ac_per_query_mean",
                "union_bc_per_query_mean",
                "previous_final_unique_per_query_mean",
                "previous_raw_unique_per_query_mean",
                "current_raw_unique_per_query_mean",
                "current_previous_raw_jaccard",
                "current_previous_final_jaccard",
                "previous_raw_previous_final_jaccard",
                "all_three_fraction_of_union",
                "union_ab_fraction_of_union_all",
                "union_ac_fraction_of_union_all",
                "union_bc_fraction_of_union_all",
                "previous_final_unique_fraction_of_union_all",
                "previous_raw_unique_fraction_of_union_all",
                "current_raw_unique_fraction_of_union_all",
                "previous_final_covered_by_current_previous_raw",
                "previous_raw_covered_by_current_previous_final",
                "current_raw_covered_by_previous_raw_previous_final",
            ]
            write_csv(
                output_dir / f"{mode}_reuse_overlap_summary.csv",
                [
                    {
                        **row,
                        "mode": mode,
                        "limit_strategy": "qabs_reuse_rerank",
                        "qabs_dim_count": qabs_dim_count if qabs_dim_count else "",
                        "qabs_candidate_fraction": qabs_candidate_fraction if qabs_candidate_fraction else "",
                    }
                    for row in reuse_overlap_stats.summary_rows()
                ],
                overlap_summary_fields,
            )
            write_csv(
                output_dir / f"{mode}_reuse_overlap_by_head.csv",
                [
                    {
                        **row,
                        "mode": mode,
                        "limit_strategy": "qabs_reuse_rerank",
                        "qabs_dim_count": qabs_dim_count if qabs_dim_count else "",
                        "qabs_candidate_fraction": qabs_candidate_fraction if qabs_candidate_fraction else "",
                    }
                    for row in reuse_overlap_stats.rows()
                ],
                [
                    "mode",
                    "limit_strategy",
                    "qabs_dim_count",
                    "qabs_candidate_fraction",
                    "layer",
                    "head",
                    "query_count",
                    "history_token_cases",
                    "current_raw_tokens",
                    "previous_raw_tokens",
                    "previous_final_tokens",
                    "current_and_previous_raw",
                    "current_and_previous_final",
                    "previous_raw_and_previous_final",
                    "all_three",
                    "union_all",
                    "union_current_previous_raw",
                    "union_current_previous_final",
                    "union_previous_raw_previous_final",
                    "previous_final_only_vs_current_previous_raw",
                    "previous_raw_only_vs_current_previous_final",
                    "current_raw_only_vs_previous_raw_previous_final",
                    "current_raw_per_query_mean",
                    "previous_raw_per_query_mean",
                    "previous_final_per_query_mean",
                    "union_all_per_query_mean",
                    "union_ab_per_query_mean",
                    "previous_final_unique_per_query_mean",
                    "previous_raw_unique_per_query_mean",
                    "current_raw_unique_per_query_mean",
                    "current_previous_raw_jaccard",
                    "current_previous_final_jaccard",
                    "previous_raw_previous_final_jaccard",
                    "all_three_fraction_of_union",
                    "union_ab_fraction_of_union_all",
                    "union_ac_fraction_of_union_all",
                    "union_bc_fraction_of_union_all",
                    "previous_final_unique_fraction_of_union_all",
                    "previous_raw_unique_fraction_of_union_all",
                    "current_raw_unique_fraction_of_union_all",
                    "previous_final_covered_by_current_previous_raw",
                    "previous_raw_covered_by_current_previous_final",
                    "current_raw_covered_by_previous_raw_previous_final",
                    "max_union_all_per_query",
                ],
            )
        if qabs_profile_stats is not None:
            write_csv(
                output_dir / f"{mode}_qabs_profile_stage_summary.csv",
                qabs_profile_stats.summary_rows(mode),
                [
                    "mode",
                    "stage",
                    "seconds",
                    "calls",
                    "seconds_per_call",
                    "fraction_of_profiled_seconds",
                ],
            )
            write_csv(
                output_dir / f"{mode}_qabs_profile_stage_by_layer.csv",
                qabs_profile_stats.stage_rows(mode),
                [
                    "mode",
                    "layer",
                    "stage",
                    "seconds",
                    "calls",
                    "seconds_per_call",
                    "layer_profiled_seconds",
                    "stage_fraction_of_layer",
                ],
            )
            write_csv(
                output_dir / f"{mode}_qabs_profile_tokens_by_head.csv",
                qabs_profile_stats.token_rows(mode),
                [
                    "mode",
                    "layer",
                    "head",
                    "query_count",
                    "history_token_cases",
                    "current_raw_tokens",
                    "union_tokens",
                    "final_tokens",
                    "current_raw_per_query_mean",
                    "union_per_query_mean",
                    "final_per_query_mean",
                    "current_raw_fraction_of_history",
                    "union_fraction_of_history",
                    "final_fraction_of_history",
                    "max_current_raw_per_query",
                    "max_union_per_query",
                    "max_final_per_query",
                    "mean_candidate_threshold",
                ],
            )

        if obs_state is not None:
            for row in obs_state.rows():
                row = dict(row)
                row["mode"] = mode
                row["obs_window_tokens"] = args.obs_window_tokens
                row["obs_recent_tokens"] = args.obs_recent_tokens
                row["obs_target_coverage"] = args.obs_target_coverage
                row["hybrid_top2_target_coverage"] = ""
                row["hybrid_top2_full_heads"] = ""
                row["hybrid_non_top2_mass_coverage"] = ""
                row["hybrid_top2_max_heads"] = ""
                row["hybrid_non_top2_max_heads"] = ""
                row["top2_assigned_token_count"] = ""
                row["non_top2_assigned_token_count"] = ""
                row["obs_min_heads"] = args.obs_min_heads
                row["obs_max_heads"] = args.obs_max_heads
                row["obs_fallback_all"] = args.obs_fallback_all
                observation_rows.append(row)
        if obs_mass_state is not None:
            for row in obs_mass_state.rows():
                row = dict(row)
                row["mode"] = mode
                row["obs_window_tokens"] = args.obs_window_tokens
                row["obs_recent_tokens"] = args.obs_recent_tokens
                row["obs_target_coverage"] = args.obs_target_coverage
                row["hybrid_top2_target_coverage"] = ""
                row["hybrid_top2_full_heads"] = ""
                row["hybrid_non_top2_mass_coverage"] = ""
                row["hybrid_top2_max_heads"] = ""
                row["hybrid_non_top2_max_heads"] = ""
                row["top2_assigned_token_count"] = ""
                row["non_top2_assigned_token_count"] = ""
                row["obs_min_heads"] = args.obs_min_heads
                row["obs_max_heads"] = args.obs_max_heads
                row["obs_fallback_all"] = args.obs_fallback_all
                observation_rows.append(row)
        if obs_hybrid_state is not None:
            for row in obs_hybrid_state.rows():
                row = dict(row)
                row["mode"] = mode
                row["obs_window_tokens"] = args.obs_window_tokens
                row["obs_recent_tokens"] = args.obs_recent_tokens
                row["obs_target_coverage"] = args.obs_target_coverage
                row["hybrid_top2_use_mass"] = hybrid_params["top2_use_mass"] if hybrid_params else ""
                row["hybrid_top2_full_heads"] = hybrid_params["top2_full_heads"] if hybrid_params else ""
                row["hybrid_top2_target_coverage"] = hybrid_params["top2_target"] if hybrid_params else ""
                row["hybrid_non_top2_mass_coverage"] = hybrid_params["non_top2_target"] if hybrid_params else ""
                row["hybrid_top2_min_heads"] = hybrid_params["top2_min_heads"] if hybrid_params else ""
                row["hybrid_top2_max_heads"] = hybrid_params["top2_max_heads"] if hybrid_params else ""
                row["hybrid_non_top2_min_heads"] = hybrid_params["non_top2_min_heads"] if hybrid_params else ""
                row["hybrid_non_top2_max_heads"] = hybrid_params["non_top2_max_heads"] if hybrid_params else ""
                row["obs_min_heads"] = args.obs_min_heads
                row["obs_max_heads"] = args.obs_max_heads
                row["obs_fallback_all"] = args.obs_fallback_all
                observation_rows.append(row)
    write_csv(
        output_dir / "ppl_by_mode.csv",
        ppl_rows,
        [
            "mode",
            "loss",
            "ppl",
            "token_count",
            "seconds",
            "top_fraction",
            "sign_xnor_candidate_fraction",
            "recent_window_tokens",
            "qabs_dim_count",
            "qabs_candidate_fraction",
            "kdom_key_dim_count",
            "max_heads_per_token",
            "limit_strategy",
            "protected_sink_tokens",
            "protected_recent_fraction",
            "protected_recent_tokens",
            "always_keep_self",
            "qabs_fast_path",
            "qabs_cuda_final_kernel",
            "qabs_cuda_candidate_kernel",
            "qabs_cuda_reuse_select_kernel",
            "qabs_profile",
            "qabs_candidate_selection",
            "qabs_threshold_sample_size",
            "lagged_refresh_fraction",
            "full_head_map_path",
            "full_layer_map_path",
            "layer_budget_map_path",
            "adaptive_low_tokens",
            "adaptive_high_tokens",
            "adaptive_low_fraction",
            "adaptive_mean_margin",
            "adaptive_margin_threshold",
            "reuse_prefill_cache",
            "shared_prefill_seconds",
        ],
    )

    if load_rows:
        write_csv(
            output_dir / "limit_load_by_head.csv",
            load_rows,
            [
                "mode",
                "limit_strategy",
                "max_heads_per_token",
                "protected_sink_tokens",
                "protected_recent_fraction",
                "protected_recent_tokens",
                "layer",
                "head",
                "query_count",
                "history_token_cases",
                "original_top2_kept",
                "final_kept_after_limit3",
                "removed_by_limit3",
                "final_kept_per_query_mean",
                "original_kept_per_query_mean",
                "removed_per_query_mean",
                "kept_fraction_of_original_top2",
                "kept_fraction_of_history_cases",
                "max_final_kept_per_query",
                "max_removed_per_query",
            ],
        )
        score3_rows = [row for row in load_rows if row["mode"] == "top2limit3score"]
        if score3_rows:
            write_csv(
                output_dir / "top2limit3score_load_by_head.csv",
                score3_rows,
                [
                    "mode",
                    "limit_strategy",
                    "max_heads_per_token",
                    "protected_sink_tokens",
                    "protected_recent_fraction",
                    "protected_recent_tokens",
                    "layer",
                    "head",
                    "query_count",
                    "history_token_cases",
                    "original_top2_kept",
                    "final_kept_after_limit3",
                    "removed_by_limit3",
                    "final_kept_per_query_mean",
                    "original_kept_per_query_mean",
                    "removed_per_query_mean",
                    "kept_fraction_of_original_top2",
                    "kept_fraction_of_history_cases",
                    "max_final_kept_per_query",
                    "max_removed_per_query",
                ],
            )
    if observation_rows:
        write_csv(
            output_dir / "observation_window_allocations.csv",
            observation_rows,
            [
                "mode",
                "obs_window_tokens",
                "obs_recent_tokens",
                "obs_target_coverage",
                "hybrid_top2_use_mass",
                "hybrid_top2_full_heads",
                "hybrid_top2_target_coverage",
                "hybrid_non_top2_mass_coverage",
                "hybrid_top2_min_heads",
                "hybrid_top2_max_heads",
                "hybrid_non_top2_min_heads",
                "hybrid_non_top2_max_heads",
                "obs_min_heads",
                "obs_max_heads",
                "obs_fallback_all",
                "layer",
                "boundary_query_token",
                "assigned_token_count",
                "top2_assigned_token_count",
                "non_top2_assigned_token_count",
                "mean_assigned_heads",
                "mean_window_events_per_assigned_token",
                "mean_window_attention_mass_per_assigned_token",
                "min_assigned_heads",
                "max_assigned_heads",
            ]
            + [f"tokens_assigned_{count}_heads" for count in range(1, head_count + 1)],
        )

    plot_paths = plot_outputs(output_dir, ppl_rows, load_rows, args.plot_dpi) if args.make_plots else []
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "resolved": {
                    "total_tokens_used": int(input_ids.numel()),
                    "prefill_tokens": args.prefill_tokens,
                    "eval_tokens": args.eval_tokens,
                    "shared_prefill_seconds": shared_prefill_seconds,
                    "layer_count": layer_count,
                    "head_count": head_count,
                    "modes": modes,
                    "history_selection_rule": "top ceil(top_fraction * history_token_count) historical keys per head",
                    "limit_rule": (
                        "if a historical key is selected by more than max_heads_per_token heads, "
                        "random modes keep a random subset and score modes keep the highest-score heads"
                    ),
                    "observation_window_rule": (
                        "top2obswin observes full top2 head sets for each layer over fixed query windows, "
                        "then reallocates each observed historical token to the smallest head set whose "
                        "window event coverage reaches obs_target_coverage; recent tokens and unobserved "
                        "tokens are kept according to obs_recent_tokens and obs_fallback_all"
                    ),
                    "self_token_rule": "kept unconditionally when always_keep_self=true",
                },
                "paths": {
                    "ppl_by_mode": str(output_dir / "ppl_by_mode.csv"),
                    "limit_load_by_head": str(output_dir / "limit_load_by_head.csv") if load_rows else None,
                    "top2limit3score_load_by_head": (
                        str(output_dir / "top2limit3score_load_by_head.csv")
                        if any(row["mode"] == "top2limit3score" for row in load_rows)
                        else None
                    ),
                    "observation_window_allocations": (
                        str(output_dir / "observation_window_allocations.csv") if observation_rows else None
                    ),
                    "plots": plot_paths,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
