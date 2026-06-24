from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
_ACTIVE_CANDIDATE_STATS: "CandidateStats | None" = None
_ACTIVE_QABS_FAST_PATH: bool = False
_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None


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
        "--disable_sparse_stats",
        type=str2bool,
        default=False,
        help="Disable sparse load/candidate CSV stats to avoid GPU sync during timing runs.",
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
            "qabs_partial_rerank",
            "qabs_reuse_rerank",
        }
    ]
    if invalid:
        raise ValueError(
            "Invalid modes: "
            f"{invalid}. Valid examples: baseline, top2, top2limit3, top2limit3score, "
            "top2union, obstop2fullnonmasst80kn2mn1, top2limit3gap1p0, "
            "top2limit3protects16r1p0."
        )
    if not modes:
        raise ValueError("--modes cannot be empty.")
    return modes


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
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?rerank", mode):
        return "qabs_partial_rerank", None
    if re.fullmatch(r"qabs\d+cand\d+(?:p\d+)?reuse", mode):
        return "qabs_reuse_rerank", None
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


def parse_qabs_partial_rerank_params(mode: str) -> tuple[int, float] | None:
    match = re.fullmatch(r"qabs(\d+)cand(\d+(?:p\d+)?)(?:rerank|reuse)", mode)
    if not match:
        return None
    dim_count = int(match.group(1))
    candidate_fraction = float(match.group(2).replace("p", ".")) / 100.0
    return dim_count, candidate_fraction


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


class ReuseCandidateState:
    def __init__(self) -> None:
        self.previous: dict[tuple[int, int], dict[str, Any]] = {}
        self.previous_layer: dict[int, dict[str, Any]] = {}

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

    def update_layer(
        self,
        layer: int,
        query_token: int,
        candidate_keep: torch.Tensor,
        final_keep: torch.Tensor,
        history_count: int,
    ) -> None:
        self.previous_layer[layer] = {
            "query_token": query_token,
            "candidate": candidate_keep[..., :history_count].detach().bool(),
            "final": final_keep[..., :history_count].detach().bool(),
        }


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
    final_keep = original_keep.clone()
    counts = original_keep.sum(dim=1)
    over = torch.nonzero((counts > max_heads) & ~protected_keys, as_tuple=False)
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
    q = query_states[:, :, 0, :].float()
    k_history = key_states[:, :, :history_count, :]

    selected_dim_count = min(max(1, dim_count), head_dim)
    dim_indices = torch.topk(q.abs(), k=selected_dim_count, dim=-1, largest=True).indices
    q_selected = torch.gather(q, dim=-1, index=dim_indices)
    k_dim_indices = dim_indices[:, :, None, :].expand(-1, -1, history_count, -1)
    k_selected = torch.gather(k_history.float(), dim=-1, index=k_dim_indices)
    partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)
    requested = min(history_count, max(1, math.ceil(candidate_fraction * history_count)))
    threshold = torch.topk(partial_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]
    current_candidate_history = partial_scores >= threshold

    candidate_union_history = current_candidate_history.clone()
    if _ACTIVE_REUSE_STATE is not None:
        previous_candidate, previous_final = _ACTIVE_REUSE_STATE.previous_layer_masks(
            layer_idx,
            query_token,
            history_count,
            candidate_union_history.device,
        )
        if previous_candidate is not None:
            candidate_union_history |= previous_candidate
        if previous_final is not None:
            candidate_union_history |= previous_final

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
    _, selected_candidate_positions = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
    selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_candidate_positions)
    selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)
    history_final = torch.zeros_like(candidate_union_history, dtype=torch.bool)
    history_final.scatter_(dim=-1, index=selected_history_indices, src=selected_valid)

    if _ACTIVE_PROTECT_SINK_TOKENS > 0:
        history_final[:, :, : min(_ACTIVE_PROTECT_SINK_TOKENS, history_count)] = True
    if _ACTIVE_PROTECT_RECENT_TOKENS > 0:
        recent_start = max(0, history_count - _ACTIVE_PROTECT_RECENT_TOKENS)
        history_final[:, :, recent_start:history_count] = True

    if _ACTIVE_CANDIDATE_STATS is not None:
        _ACTIVE_CANDIDATE_STATS.update(layer_idx, candidate_union_history, torch.ones_like(candidate_union_history))
    if _ACTIVE_LOAD_STATS is not None:
        _ACTIVE_LOAD_STATS.update(layer_idx, torch.ones_like(history_final), history_final, torch.ones_like(history_final))
    if _ACTIVE_REUSE_STATE is not None:
        _ACTIVE_REUSE_STATE.update_layer(layer_idx, query_token, current_candidate_history, history_final, history_count)

    final_keep = torch.zeros((batch_count, head_count, key_count), dtype=torch.bool, device=query_states.device)
    final_keep[:, :, :history_count] = history_final
    if _ACTIVE_ALWAYS_KEEP_SELF:
        final_keep[:, :, history_count] = True

    final_indices, final_valid = _indices_from_keep_mask(final_keep)
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
    if (
        _ACTIVE_QABS_FAST_PATH
        and mode_kind == "qabs_reuse_rerank"
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
        "qabs_partial_rerank",
        "qabs_reuse_rerank",
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
            elif mode_kind == "qabs_partial_rerank":
                qabs_params = parse_qabs_partial_rerank_params(_ACTIVE_MODE)
                if qabs_params is None:
                    raise RuntimeError(f"Invalid qabs partial-rerank mode: {_ACTIVE_MODE}")
                dim_count, candidate_fraction = qabs_params
                partial_candidate = _qabs_partial_candidate_keep_for_query(
                    query_states,
                    key_states,
                    query_index,
                    finite,
                    dim_count,
                    candidate_fraction,
                )
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
                        if previous_candidate is not None:
                            candidate_union[0, head, :history_count] |= previous_candidate
                        if previous_final is not None:
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
    candidate_stats: CandidateStats | None = None,
    qabs_fast_path: bool = False,
):
    global _ACTIVE_MODE, _ACTIVE_TOP_FRACTION, _ACTIVE_MAX_HEADS_PER_TOKEN, _ACTIVE_ALWAYS_KEEP_SELF, _ACTIVE_PROTECT_SINK_TOKENS, _ACTIVE_PROTECT_RECENT_TOKENS, _ACTIVE_LOAD_STATS, _ACTIVE_OBS_STATE, _ACTIVE_OBS_MASS_STATE, _ACTIVE_OBS_HYBRID_STATE, _ACTIVE_REUSE_STATE, _ACTIVE_CANDIDATE_STATS, _ACTIVE_QABS_FAST_PATH
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
        _ACTIVE_CANDIDATE_STATS,
        _ACTIVE_QABS_FAST_PATH,
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
    _ACTIVE_CANDIDATE_STATS = candidate_stats
    _ACTIVE_QABS_FAST_PATH = qabs_fast_path
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
            _ACTIVE_CANDIDATE_STATS,
            _ACTIVE_QABS_FAST_PATH,
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
    candidate_stats: CandidateStats | None = None,
    qabs_fast_path: bool = False,
    log_every: int = 1,
) -> tuple[float, float, int, float]:
    print(f"starting mode: {mode}", flush=True)
    started = time.perf_counter()
    past_key_values, prev_logits = prefill_cache(model, input_ids, prefill_tokens, prefill_chunk_size, input_device)
    total_loss = 0.0
    total_count = 0
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / eval_chunk_size)
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
        with attention_mode(
            mode,
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
            candidate_stats,
            qabs_fast_path,
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
        recent_window_tokens = parse_recent_window_tokens(mode) if mode_kind == "recent_window" else None
        qabs_params = (
            parse_qabs_partial_rerank_params(mode)
            if mode_kind in {"qabs_partial_rerank", "qabs_reuse_rerank"}
            else None
        )
        qabs_dim_count = qabs_params[0] if qabs_params else None
        qabs_candidate_fraction = qabs_params[1] if qabs_params else None
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
                "qabs_partial_rerank",
                "qabs_reuse_rerank",
            }
            else None
        )
        candidate_stats = (
            CandidateStats(layer_count, head_count)
            if mode_kind == "qabs_reuse_rerank" and not args.disable_sparse_stats
            else None
        )
        reuse_state = ReuseCandidateState() if mode_kind == "qabs_reuse_rerank" else None
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
            candidate_stats,
            args.qabs_fast_path,
            args.log_every,
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
                        "qabs_partial_rerank",
                        "qabs_reuse_rerank",
                    }
                    else ""
                ),
                "sign_xnor_candidate_fraction": sign_xnor_candidate_fraction if sign_xnor_candidate_fraction else "",
                "recent_window_tokens": recent_window_tokens if recent_window_tokens else "",
                "qabs_dim_count": qabs_dim_count if qabs_dim_count else "",
                "qabs_candidate_fraction": qabs_candidate_fraction if qabs_candidate_fraction else "",
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
                    else "qabs_partial_rerank"
                    if mode_kind == "qabs_partial_rerank"
                    else "qabs_reuse_rerank"
                    if mode_kind == "qabs_reuse_rerank"
                    else ""
                ),
                "protected_sink_tokens": protect_params[0] if protect_params else args.protect_sink_tokens,
                "protected_recent_fraction": protect_params[1] if protect_params else "",
                "protected_recent_tokens": args.protect_recent_tokens,
                "always_keep_self": args.always_keep_self if mode != "baseline" else "",
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
                    else "qabs_partial_rerank"
                    if mode_kind == "qabs_partial_rerank"
                    else "qabs_reuse_rerank"
                    if mode_kind == "qabs_reuse_rerank"
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
            "max_heads_per_token",
            "limit_strategy",
            "protected_sink_tokens",
            "protected_recent_fraction",
            "protected_recent_tokens",
            "always_keep_self",
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
