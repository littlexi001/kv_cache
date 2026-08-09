from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attention_selectors import (  # noqa: E402
    SelectorSpec,
    actual_history_fraction,
    build_keep_mask,
    historical_budget,
    parse_selector,
)


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_top2_mechanism_guarded", False):
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

    guarded_register_fake._top2_mechanism_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"
DEFAULT_RATIOS = "0.001,0.005,0.01,0.02,0.04,0.08,0.16,0.32,1.0"
DEFAULT_CONTROLS = (
    "sink_recent_s0,sink_recent_s1,sink_recent_s2,sink_recent_s4,sink_recent_s8,"
    "sink_recent_s16,recent,sink,random,bottom_attention,"
    "top_attention_drop_sink,top_attention_drop_recent,top_attention_drop_remote"
)

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_RUN: "ActiveRun | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_float_list(spec: str) -> list[float]:
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("Ratios must be in (0, 1].")
    return values


def parse_int_list(spec: str) -> list[int]:
    values = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not values or any(value < 0 for value in values):
        raise ValueError("Sink sweep values must be non-negative integers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Top-2% attention mechanism and equal-budget sink+recent controls."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/top2_token_mechanism")
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--append_eos", type=str2bool, default=False)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--ratio_grid", default=DEFAULT_RATIOS)
    parser.add_argument("--target_ratio", type=float, default=0.02)
    parser.add_argument("--control_selectors", default=DEFAULT_CONTROLS)
    parser.add_argument("--diagnostic_sink_sweep", default="0,1,2,4,8,16,32")
    parser.add_argument("--always_keep_self", type=str2bool, default=True)
    parser.add_argument("--role_sink_tokens", type=int, default=4)
    parser.add_argument("--role_recent_tokens", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=20260714)
    parser.add_argument("--collect_diagnostics", type=str2bool, default=True)
    parser.add_argument("--write_token_nll", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def pick_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio_slug(ratio: float) -> str:
    return f"{ratio:g}".replace(".", "p")


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


@dataclass(frozen=True)
class ModeSpec:
    label: str
    selector: SelectorSpec | None
    ratio: float | None


@dataclass
class SelectionRunStats:
    selected_history_events: int = 0
    eligible_history_events: int = 0

    def update(self, history_keep: torch.Tensor) -> None:
        selected, eligible = actual_history_fraction(history_keep)
        self.selected_history_events += selected
        self.eligible_history_events += eligible

    @property
    def kept_fraction(self) -> float:
        if self.eligible_history_events == 0:
            return float("nan")
        return self.selected_history_events / self.eligible_history_events


class Top2Diagnostics:
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        total_tokens: int,
        ratio: float,
        sink_sweep: list[int],
        always_keep_self: bool,
        role_sink_tokens: int,
        role_recent_tokens: int,
        random_seed: int,
    ) -> None:
        self.layer_count = layer_count
        self.head_count = head_count
        self.total_tokens = total_tokens
        self.ratio = ratio
        self.sink_sweep = sink_sweep
        self.always_keep_self = always_keep_self
        self.role_sink_tokens = role_sink_tokens
        self.role_recent_tokens = role_recent_tokens
        self.random_seed = random_seed
        self.token_counts = torch.zeros(total_tokens, dtype=torch.int64)
        self.token_mass = torch.zeros(total_tokens, dtype=torch.float64)
        self.token_sink_counts = torch.zeros(total_tokens, dtype=torch.int64)
        self.token_recent_counts = torch.zeros(total_tokens, dtype=torch.int64)
        self.token_remote_counts = torch.zeros(total_tokens, dtype=torch.int64)
        self.observed_queries: set[int] = set()
        self.layer_query_union_rows: list[dict[str, Any]] = []
        self.model_query_union_masks: dict[int, torch.Tensor] = {}
        self.model_query_selected_events: dict[int, int] = {}
        self.model_query_layer_union_sum: dict[int, int] = {}
        self.temporal_layer_union = torch.zeros((layer_count, total_tokens), dtype=torch.bool)
        self.temporal_model_union = torch.zeros(total_tokens, dtype=torch.bool)
        shape = (layer_count, head_count)
        self.query_rows = torch.zeros(shape, dtype=torch.int64)
        self.metric_sums = {
            "entropy": torch.zeros(shape, dtype=torch.float64),
            "effective_support_entropy": torch.zeros(shape, dtype=torch.float64),
            "effective_support_l2": torch.zeros(shape, dtype=torch.float64),
            "normalized_effective_support_l2": torch.zeros(shape, dtype=torch.float64),
            "top_history_mass": torch.zeros(shape, dtype=torch.float64),
            "top_plus_self_mass": torch.zeros(shape, dtype=torch.float64),
            "cutoff_gap": torch.zeros(shape, dtype=torch.float64),
        }
        self.cutoff_gap_rows = torch.zeros(shape, dtype=torch.int64)
        overlap_shape = (len(sink_sweep), layer_count, head_count)
        self.overlap_query_rows = torch.zeros(overlap_shape, dtype=torch.int64)
        self.overlap_sums = {
            "top_events": torch.zeros(overlap_shape, dtype=torch.float64),
            "overlap_events": torch.zeros(overlap_shape, dtype=torch.float64),
            "top_mass": torch.zeros(overlap_shape, dtype=torch.float64),
            "overlap_mass": torch.zeros(overlap_shape, dtype=torch.float64),
            "sink_recent_full_mass": torch.zeros(overlap_shape, dtype=torch.float64),
            "distribution_cosine": torch.zeros(overlap_shape, dtype=torch.float64),
        }

    @torch.inference_mode()
    def update(
        self,
        module: torch.nn.Module,
        scores: torch.Tensor,
        top_keep: torch.Tensor,
        top_history_keep: torch.Tensor,
    ) -> None:
        if scores.shape[0] != 1:
            raise ValueError("Diagnostics currently require batch size 1.")
        layer_idx = int(getattr(module, "layer_idx", 0))
        full_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        _, head_count, query_count, key_count = scores.shape
        if head_count != self.head_count:
            raise ValueError(f"Expected {self.head_count} attention heads, got {head_count}.")
        past_tokens = key_count - query_count

        fixed_masks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for sink_tokens in self.sink_sweep:
            fixed_masks.append(
                build_keep_mask(
                    scores,
                    SelectorSpec("sink_recent", sink_tokens),
                    self.ratio,
                    always_keep_self=self.always_keep_self,
                    role_sink_tokens=self.role_sink_tokens,
                    role_recent_tokens=self.role_recent_tokens,
                    random_seed=self.random_seed,
                    layer_idx=layer_idx,
                )
            )

        for query_idx in range(query_count):
            current = past_tokens + query_idx
            if current <= 0:
                continue
            if layer_idx == 0:
                self.observed_queries.add(current)
            hist_selected = top_history_keep[0, :, query_idx, :current]
            hist_weights = full_weights[0, :, query_idx, :current]
            selected_by_position = hist_selected.sum(dim=0).to("cpu", dtype=torch.int64)
            mass_by_position = (hist_weights * hist_selected).sum(dim=0).to("cpu", dtype=torch.float64)
            self.token_counts[:current] += selected_by_position
            self.token_mass[:current] += mass_by_position

            layer_union = selected_by_position > 0
            layer_union_tokens = int(layer_union.sum())
            selected_events = int(selected_by_position.sum())
            per_head_budget = historical_budget(current, self.ratio)
            all_head_intersection_tokens = int((selected_by_position == self.head_count).sum())
            self.layer_query_union_rows.append(
                {
                    "layer": layer_idx,
                    "query_token": current,
                    "history_tokens": current,
                    "per_head_budget": per_head_budget,
                    "head_count": self.head_count,
                    "selected_head_token_events": selected_events,
                    "union_tokens": layer_union_tokens,
                    "union_fraction_of_history": layer_union_tokens / current,
                    "union_vs_single_head_budget": (
                        layer_union_tokens / per_head_budget if per_head_budget else ""
                    ),
                    "sharing_efficiency": layer_union_tokens / selected_events if selected_events else "",
                    "redundant_event_fraction": (
                        1.0 - layer_union_tokens / selected_events if selected_events else ""
                    ),
                    "all_head_intersection_tokens": all_head_intersection_tokens,
                    "all_head_intersection_fraction_of_budget": (
                        all_head_intersection_tokens / per_head_budget if per_head_budget else ""
                    ),
                }
            )
            self.temporal_layer_union[layer_idx, :current] |= layer_union
            self.temporal_model_union[:current] |= layer_union
            if current not in self.model_query_union_masks:
                self.model_query_union_masks[current] = torch.zeros(self.total_tokens, dtype=torch.bool)
                self.model_query_selected_events[current] = 0
                self.model_query_layer_union_sum[current] = 0
            self.model_query_union_masks[current][:current] |= layer_union
            self.model_query_selected_events[current] += selected_events
            self.model_query_layer_union_sum[current] += layer_union_tokens

            positions = torch.arange(current, device=scores.device)
            is_sink = positions < min(current, self.role_sink_tokens)
            is_recent = (positions >= max(0, current - self.role_recent_tokens)) & ~is_sink
            is_remote = ~(is_sink | is_recent)
            self.token_sink_counts[:current] += (
                hist_selected & is_sink.view(1, -1)
            ).sum(dim=0).to("cpu", dtype=torch.int64)
            self.token_recent_counts[:current] += (
                hist_selected & is_recent.view(1, -1)
            ).sum(dim=0).to("cpu", dtype=torch.int64)
            self.token_remote_counts[:current] += (
                hist_selected & is_remote.view(1, -1)
            ).sum(dim=0).to("cpu", dtype=torch.int64)

            full_row = full_weights[0, :, query_idx, : current + 1]
            top_hist_row = hist_selected
            top_keep_row = top_keep[0, :, query_idx, : current + 1]
            entropy = -(full_row * full_row.clamp_min(1e-30).log()).sum(dim=-1)
            effective_entropy = entropy.exp()
            effective_l2 = 1.0 / full_row.square().sum(dim=-1).clamp_min(1e-30)
            top_history_mass = (full_row[:, :current] * top_hist_row).sum(dim=-1)
            top_plus_self_mass = (full_row * top_keep_row).sum(dim=-1)
            self.query_rows[layer_idx] += 1
            self.metric_sums["entropy"][layer_idx] += entropy.to("cpu", dtype=torch.float64)
            self.metric_sums["effective_support_entropy"][layer_idx] += effective_entropy.to(
                "cpu", dtype=torch.float64
            )
            self.metric_sums["effective_support_l2"][layer_idx] += effective_l2.to("cpu", dtype=torch.float64)
            self.metric_sums["normalized_effective_support_l2"][layer_idx] += (
                effective_l2 / (current + 1)
            ).to("cpu", dtype=torch.float64)
            self.metric_sums["top_history_mass"][layer_idx] += top_history_mass.to(
                "cpu", dtype=torch.float64
            )
            self.metric_sums["top_plus_self_mass"][layer_idx] += top_plus_self_mass.to(
                "cpu", dtype=torch.float64
            )

            budget = historical_budget(current, self.ratio)
            if budget < current:
                boundary = torch.topk(
                    scores[0, :, query_idx, :current], k=budget + 1, dim=-1, largest=True
                ).values
                gap = boundary[:, budget - 1] - boundary[:, budget]
                self.metric_sums["cutoff_gap"][layer_idx] += gap.to("cpu", dtype=torch.float64)
                self.cutoff_gap_rows[layer_idx] += 1

            top_distribution = full_row * top_keep_row
            top_distribution /= top_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            for sweep_idx, (fixed_keep, fixed_history) in enumerate(fixed_masks):
                fixed_hist_row = fixed_history[0, :, query_idx, :current]
                fixed_keep_row = fixed_keep[0, :, query_idx, : current + 1]
                intersection = top_hist_row & fixed_hist_row
                fixed_distribution = full_row * fixed_keep_row
                fixed_distribution /= fixed_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                cosine = F.cosine_similarity(top_distribution, fixed_distribution, dim=-1)
                self.overlap_query_rows[sweep_idx, layer_idx] += 1
                self.overlap_sums["top_events"][sweep_idx, layer_idx] += top_hist_row.sum(
                    dim=-1
                ).to("cpu", dtype=torch.float64)
                self.overlap_sums["overlap_events"][sweep_idx, layer_idx] += intersection.sum(
                    dim=-1
                ).to("cpu", dtype=torch.float64)
                self.overlap_sums["top_mass"][sweep_idx, layer_idx] += top_history_mass.to(
                    "cpu", dtype=torch.float64
                )
                self.overlap_sums["overlap_mass"][sweep_idx, layer_idx] += (
                    full_row[:, :current] * intersection
                ).sum(dim=-1).to("cpu", dtype=torch.float64)
                self.overlap_sums["sink_recent_full_mass"][sweep_idx, layer_idx] += (
                    full_row * fixed_keep_row
                ).sum(dim=-1).to("cpu", dtype=torch.float64)
                self.overlap_sums["distribution_cosine"][sweep_idx, layer_idx] += cosine.to(
                    "cpu", dtype=torch.float64
                )

    def concentration_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            for head in range(self.head_count):
                count = int(self.query_rows[layer, head])
                gap_count = int(self.cutoff_gap_rows[layer, head])
                row: dict[str, Any] = {"layer": layer, "head": head, "query_count": count}
                for name, values in self.metric_sums.items():
                    divisor = gap_count if name == "cutoff_gap" else count
                    row[f"mean_{name}"] = float(values[layer, head]) / divisor if divisor else ""
                rows.append(row)
        return rows

    def overlap_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sweep_idx, sink_tokens in enumerate(self.sink_sweep):
            for layer in range(self.layer_count):
                for head in range(self.head_count):
                    query_count = int(self.overlap_query_rows[sweep_idx, layer, head])
                    top_events = float(self.overlap_sums["top_events"][sweep_idx, layer, head])
                    overlap_events = float(self.overlap_sums["overlap_events"][sweep_idx, layer, head])
                    top_mass = float(self.overlap_sums["top_mass"][sweep_idx, layer, head])
                    overlap_mass = float(self.overlap_sums["overlap_mass"][sweep_idx, layer, head])
                    rows.append(
                        {
                            "sink_tokens": sink_tokens,
                            "layer": layer,
                            "head": head,
                            "query_count": query_count,
                            "top2_selected_events": top_events,
                            "overlap_events": overlap_events,
                            "overlap_event_recall": overlap_events / top_events if top_events else "",
                            "top2_attention_mass_sum": top_mass,
                            "overlap_attention_mass_sum": overlap_mass,
                            "overlap_top2_mass_recall": overlap_mass / top_mass if top_mass else "",
                            "mean_sink_recent_full_attention_mass": (
                                float(
                                    self.overlap_sums["sink_recent_full_mass"][sweep_idx, layer, head]
                                )
                                / query_count
                                if query_count
                                else ""
                            ),
                            "mean_pruned_distribution_cosine": (
                                float(self.overlap_sums["distribution_cosine"][sweep_idx, layer, head])
                                / query_count
                                if query_count
                                else ""
                            ),
                        }
                    )
        return rows

    def token_rows(self, tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        max_query = max(self.observed_queries) if self.observed_queries else 0
        for token_index in range(min(max_query, len(token_ids))):
            eligible_queries = sum(1 for query in self.observed_queries if query > token_index)
            eligible_events = eligible_queries * self.layer_count * self.head_count
            token_id = int(token_ids[token_index])
            rows.append(
                {
                    "token_index": token_index,
                    "token_id": token_id,
                    "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                    "token_text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                    "eligible_event_count": eligible_events,
                    "top2_selected_count": int(self.token_counts[token_index]),
                    "selection_rate": (
                        int(self.token_counts[token_index]) / eligible_events if eligible_events else 0.0
                    ),
                    "top2_attention_mass_sum": float(self.token_mass[token_index]),
                    "sink_role_count": int(self.token_sink_counts[token_index]),
                    "recent_role_count": int(self.token_recent_counts[token_index]),
                    "remote_role_count": int(self.token_remote_counts[token_index]),
                }
            )
        return rows

    def model_union_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_token in sorted(self.model_query_union_masks):
            history_tokens = query_token
            per_head_budget = historical_budget(history_tokens, self.ratio)
            union_tokens = int(self.model_query_union_masks[query_token][:history_tokens].sum())
            selected_events = self.model_query_selected_events[query_token]
            rows.append(
                {
                    "query_token": query_token,
                    "history_tokens": history_tokens,
                    "per_head_budget": per_head_budget,
                    "layer_count": self.layer_count,
                    "head_count_per_layer": self.head_count,
                    "total_head_count": self.layer_count * self.head_count,
                    "selected_head_token_events": selected_events,
                    "sum_of_layer_union_tokens": self.model_query_layer_union_sum[query_token],
                    "union_tokens": union_tokens,
                    "union_fraction_of_history": union_tokens / history_tokens,
                    "union_vs_single_head_budget": (
                        union_tokens / per_head_budget if per_head_budget else ""
                    ),
                    "sharing_efficiency": union_tokens / selected_events if selected_events else "",
                    "redundant_event_fraction": (
                        1.0 - union_tokens / selected_events if selected_events else ""
                    ),
                }
            )
        return rows

    @staticmethod
    def _union_summary_row(
        scope: str,
        rows: list[dict[str, Any]],
        layer: int | str = "",
    ) -> dict[str, Any]:
        unions = [float(row["union_tokens"]) for row in rows]
        return {
            "scope": scope,
            "layer": layer,
            "query_rows": len(rows),
            "mean_history_tokens": sum(float(row["history_tokens"]) for row in rows) / len(rows),
            "mean_per_head_budget": sum(float(row["per_head_budget"]) for row in rows) / len(rows),
            "mean_union_tokens": sum(unions) / len(unions),
            "p50_union_tokens": quantile(unions, 0.50),
            "p95_union_tokens": quantile(unions, 0.95),
            "min_union_tokens": min(unions),
            "max_union_tokens": max(unions),
            "mean_union_fraction_of_history": (
                sum(float(row["union_fraction_of_history"]) for row in rows) / len(rows)
            ),
            "mean_union_vs_single_head_budget": (
                sum(float(row["union_vs_single_head_budget"]) for row in rows) / len(rows)
            ),
            "mean_sharing_efficiency": (
                sum(float(row["sharing_efficiency"]) for row in rows) / len(rows)
            ),
            "mean_redundant_event_fraction": (
                sum(float(row["redundant_event_fraction"]) for row in rows) / len(rows)
            ),
            "mean_all_head_intersection_tokens": (
                sum(float(row.get("all_head_intersection_tokens", 0)) for row in rows) / len(rows)
                if scope != "model_query"
                else ""
            ),
        }

    def union_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            layer_rows = [row for row in self.layer_query_union_rows if int(row["layer"]) == layer]
            rows.append(self._union_summary_row("layer_query", layer_rows, layer))
        rows.append(self._union_summary_row("all_layer_queries", self.layer_query_union_rows))
        rows.append(self._union_summary_row("model_query", self.model_union_rows()))
        return rows

    def temporal_union_rows(self) -> list[dict[str, Any]]:
        max_history_tokens = max(self.observed_queries) if self.observed_queries else 0
        rows: list[dict[str, Any]] = []
        for layer in range(self.layer_count):
            union_tokens = int(self.temporal_layer_union[layer, :max_history_tokens].sum())
            rows.append(
                {
                    "scope": "layer_temporal",
                    "layer": layer,
                    "query_count": len(self.observed_queries),
                    "eligible_history_positions": max_history_tokens,
                    "union_tokens_across_queries": union_tokens,
                    "union_fraction_of_positions": (
                        union_tokens / max_history_tokens if max_history_tokens else ""
                    ),
                }
            )
        model_union_tokens = int(self.temporal_model_union[:max_history_tokens].sum())
        rows.append(
            {
                "scope": "model_temporal",
                "layer": "",
                "query_count": len(self.observed_queries),
                "eligible_history_positions": max_history_tokens,
                "union_tokens_across_queries": model_union_tokens,
                "union_fraction_of_positions": (
                    model_union_tokens / max_history_tokens if max_history_tokens else ""
                ),
            }
        )
        return rows


@dataclass
class ActiveRun:
    selector: SelectorSpec
    ratio: float
    always_keep_self: bool
    role_sink_tokens: int
    role_recent_tokens: int
    random_seed: int
    stats: SelectionRunStats
    diagnostics: Top2Diagnostics | None = None


def _selector_eager_attention_forward(
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
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]

    active = _ACTIVE_RUN
    if active is not None:
        layer_idx = int(getattr(module, "layer_idx", 0))
        keep, history_keep = build_keep_mask(
            scores,
            active.selector,
            active.ratio,
            always_keep_self=active.always_keep_self,
            role_sink_tokens=active.role_sink_tokens,
            role_recent_tokens=active.role_recent_tokens,
            random_seed=active.random_seed,
            layer_idx=layer_idx,
        )
        active.stats.update(history_keep)
        if active.diagnostics is not None:
            active.diagnostics.update(module, scores, keep, history_keep)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)

    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_attention_patch() -> None:
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("Could not import Qwen3 eager attention implementation.") from exc
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is None:
        _ORIGINAL_EAGER_ATTENTION_FORWARD = modeling_qwen3.eager_attention_forward
        modeling_qwen3.eager_attention_forward = _selector_eager_attention_forward
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _selector_eager_attention_forward


@contextmanager
def activate(run: ActiveRun | None):
    global _ACTIVE_RUN
    previous = _ACTIVE_RUN
    _ACTIVE_RUN = run
    try:
        yield
    finally:
        _ACTIVE_RUN = previous


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
    for start in range(0, prefill_tokens, chunk_size):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
    if last_logits is None:
        raise RuntimeError("Prefill returned no logits.")
    return past_key_values, last_logits


@torch.inference_mode()
def evaluate_mode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mode: ModeSpec,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    args: argparse.Namespace,
    diagnostics: Top2Diagnostics | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"starting mode={mode.label}", flush=True)
    past, previous_logits = prefill_cache(model, input_ids, prefill_tokens, chunk_size, input_device)
    stats = SelectionRunStats()
    active = None
    if mode.selector is not None and mode.ratio is not None:
        active = ActiveRun(
            selector=mode.selector,
            ratio=mode.ratio,
            always_keep_self=args.always_keep_self,
            role_sink_tokens=args.role_sink_tokens,
            role_recent_tokens=args.role_recent_tokens,
            random_seed=args.random_seed,
            stats=stats,
            diagnostics=diagnostics,
        )

    total_loss = 0.0
    total_count = 0
    token_rows: list[dict[str, Any]] = []
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
        end = min(start + chunk_size, eval_end)
        chunk = input_ids[:, start:end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        with activate(active):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted_logits = torch.cat([previous_logits.unsqueeze(1), logits[:, :-1, :]], dim=1)
        losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
            chunk.reshape(-1),
            reduction="none",
        ).view_as(chunk)
        total_loss += float(losses.sum())
        total_count += int(losses.numel())
        if args.write_token_nll:
            for local_idx, nll in enumerate(losses[0].tolist()):
                token_index = start + local_idx
                token_rows.append(
                    {
                        "mode": mode.label,
                        "selector": mode.selector.name if mode.selector else "full_attention",
                        "ratio": mode.ratio if mode.ratio is not None else "",
                        "token_index": token_index,
                        "token_id": int(input_ids[0, token_index]),
                        "nll": float(nll),
                    }
                )
        previous_logits = logits[:, -1, :].detach()
        past = outputs.past_key_values
        if args.log_every > 0 and (chunk_idx % args.log_every == 0 or chunk_idx == total_chunks):
            print(f"mode={mode.label} chunk={chunk_idx}/{total_chunks} tokens={start}:{end}", flush=True)

    loss = total_loss / max(1, total_count)
    row = {
        "mode": mode.label,
        "selector": mode.selector.name if mode.selector else "full_attention",
        "sink_tokens": mode.selector.sink_tokens if mode.selector else "",
        "ratio": mode.ratio if mode.ratio is not None else "",
        "kept_percent_nominal": 100.0 * mode.ratio if mode.ratio is not None else 100.0,
        "kept_percent_actual_history": 100.0 * stats.kept_fraction if active is not None else 100.0,
        "always_keep_self": args.always_keep_self,
        "loss": loss,
        "ppl": math.exp(loss),
        "token_count": total_count,
    }
    return row, token_rows


def build_modes(ratios: list[float], controls: list[str], target_ratio: float) -> list[ModeSpec]:
    modes = [ModeSpec("full_attention", None, None)]
    modes.extend(
        ModeSpec(f"top_attention_r{ratio_slug(ratio)}", SelectorSpec("top_attention"), ratio)
        for ratio in ratios
    )
    seen = {mode.label for mode in modes}
    for value in controls:
        selector = parse_selector(value)
        label_selector = value.strip().lower()
        label = f"{label_selector}_r{ratio_slug(target_ratio)}"
        if label not in seen:
            modes.append(ModeSpec(label, selector, target_ratio))
            seen.add(label)
    return modes


def make_plot(output_dir: Path, rows: list[dict[str, Any]], dpi: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["mode"]) for row in rows]
    values = [float(row["ppl"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(rows)), 5), dpi=dpi)
    ax.bar(range(len(rows)), values, color="#4c78a8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Perplexity")
    ax.set_title("Top-attention curve and equal-budget controls")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "plots" / "ppl_by_selector.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def build_sanity_checks(
    ppl_rows: list[dict[str, Any]],
    token_nll_rows: list[dict[str, Any]],
    target_ratio: float,
) -> dict[str, Any]:
    top100_delta = next(
        (
            float(row["delta_loss_vs_full"])
            for row in ppl_rows
            if row["selector"] == "top_attention"
            and row["ratio"] != ""
            and math.isclose(float(row["ratio"]), 1.0)
        ),
        None,
    )
    nll_by_mode: dict[str, dict[int, float]] = {}
    for row in token_nll_rows:
        nll_by_mode.setdefault(str(row["mode"]), {})[int(row["token_index"])] = float(row["nll"])
    token_index_sets = [set(values) for values in nll_by_mode.values()]
    same_scored_indices = all(indices == token_index_sets[0] for indices in token_index_sets[1:]) if token_index_sets else None

    sink0_mode = next((mode for mode in nll_by_mode if mode.startswith("sink_recent_s0_r")), None)
    recent_mode = next((mode for mode in nll_by_mode if mode.startswith("recent_r")), None)
    sink0_recent_max_nll_diff: float | None = None
    if sink0_mode is not None and recent_mode is not None:
        common = sorted(set(nll_by_mode[sink0_mode]) & set(nll_by_mode[recent_mode]))
        sink0_recent_max_nll_diff = max(
            (abs(nll_by_mode[sink0_mode][index] - nll_by_mode[recent_mode][index]) for index in common),
            default=0.0,
        )

    equal_budget_selectors = {
        "top_attention",
        "bottom_attention",
        "recent",
        "sink",
        "sink_recent",
        "random",
    }
    equal_budget_fractions = [
        float(row["kept_percent_actual_history"])
        for row in ppl_rows
        if row["selector"] in equal_budget_selectors
        and row["ratio"] != ""
        and math.isclose(float(row["ratio"]), target_ratio)
    ]
    fraction_spread = (
        max(equal_budget_fractions) - min(equal_budget_fractions) if equal_budget_fractions else None
    )
    return {
        "top_attention_100_matches_full_delta_loss": top100_delta,
        "top_attention_100_matches_full_at_1e_6": (
            abs(top100_delta) <= 1e-6 if top100_delta is not None else None
        ),
        "all_modes_score_same_token_indices": same_scored_indices,
        "sink_recent_s0_vs_recent_max_abs_token_nll_diff": sink0_recent_max_nll_diff,
        "sink_recent_s0_matches_recent_at_1e_7": (
            sink0_recent_max_nll_diff <= 1e-7 if sink0_recent_max_nll_diff is not None else None
        ),
        "equal_budget_actual_kept_percent_spread": fraction_spread,
        "equal_budget_selectors_match_at_1e_9_percent": (
            fraction_spread <= 1e-9 if fraction_spread is not None else None
        ),
    }


def main() -> None:
    args = parse_args()
    ratios = parse_float_list(args.ratio_grid)
    if args.target_ratio not in ratios:
        ratios.append(args.target_ratio)
        ratios.sort()
    control_spec = args.control_selectors.strip()
    controls = (
        []
        if control_spec.lower() in {"", "none"}
        else [part.strip() for part in control_spec.split(",") if part.strip()]
    )
    sink_sweep = parse_int_list(args.diagnostic_sink_sweep)
    modes = build_modes(ratios, controls, args.target_ratio)
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
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.dtype, requested_device),
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    install_attention_patch()
    input_device = pick_input_device(model, requested_device)

    diagnostics = None
    if args.collect_diagnostics:
        diagnostics = Top2Diagnostics(
            layer_count=int(model.config.num_hidden_layers),
            head_count=int(model.config.num_attention_heads),
            total_tokens=int(input_ids.numel()),
            ratio=args.target_ratio,
            sink_sweep=sink_sweep,
            always_keep_self=args.always_keep_self,
            role_sink_tokens=args.role_sink_tokens,
            role_recent_tokens=args.role_recent_tokens,
            random_seed=args.random_seed,
        )

    ppl_rows: list[dict[str, Any]] = []
    token_nll_rows: list[dict[str, Any]] = []
    diagnostic_consumed = False
    for mode in modes:
        collect_for_mode = (
            diagnostics
            if not diagnostic_consumed
            and mode.selector == SelectorSpec("top_attention")
            and mode.ratio is not None
            and math.isclose(mode.ratio, args.target_ratio)
            else None
        )
        row, rows = evaluate_mode(
            model,
            input_ids,
            mode,
            args.prefill_tokens,
            args.eval_tokens,
            args.chunk_size,
            input_device,
            args,
            collect_for_mode,
        )
        if collect_for_mode is not None:
            diagnostic_consumed = True
        ppl_rows.append(row)
        token_nll_rows.extend(rows)

    baseline_loss = float(ppl_rows[0]["loss"])
    baseline_ppl = float(ppl_rows[0]["ppl"])
    for row in ppl_rows:
        row["delta_loss_vs_full"] = float(row["loss"]) - baseline_loss
        row["ppl_ratio_vs_full"] = float(row["ppl"]) / baseline_ppl

    ppl_fields = [
        "mode",
        "selector",
        "sink_tokens",
        "ratio",
        "kept_percent_nominal",
        "kept_percent_actual_history",
        "always_keep_self",
        "loss",
        "ppl",
        "delta_loss_vs_full",
        "ppl_ratio_vs_full",
        "token_count",
    ]
    write_csv(output_dir / "ppl_by_selector.csv", ppl_rows, ppl_fields)
    if args.write_token_nll:
        write_csv(
            output_dir / "token_nll_by_selector.csv",
            token_nll_rows,
            ["mode", "selector", "ratio", "token_index", "token_id", "nll"],
        )

    diagnostic_paths: dict[str, str] = {}
    if diagnostics is not None and diagnostic_consumed:
        concentration_rows = diagnostics.concentration_rows()
        overlap_rows = diagnostics.overlap_rows()
        token_rows = diagnostics.token_rows(tokenizer, token_ids)
        layer_union_rows = diagnostics.layer_query_union_rows
        model_union_rows = diagnostics.model_union_rows()
        union_summary_rows = diagnostics.union_summary_rows()
        temporal_union_rows = diagnostics.temporal_union_rows()
        concentration_path = output_dir / "top2_concentration_by_layer_head.csv"
        overlap_path = output_dir / "sink_recent_overlap_by_layer_head.csv"
        token_path = output_dir / "top2_token_events.csv"
        layer_union_path = output_dir / "top2_union_by_layer_query.csv"
        model_union_path = output_dir / "top2_union_by_model_query.csv"
        union_summary_path = output_dir / "top2_union_summary.csv"
        temporal_union_path = output_dir / "top2_temporal_union.csv"
        write_csv(concentration_path, concentration_rows, list(concentration_rows[0]))
        write_csv(overlap_path, overlap_rows, list(overlap_rows[0]))
        write_csv(token_path, token_rows, list(token_rows[0]))
        write_csv(layer_union_path, layer_union_rows, list(layer_union_rows[0]))
        write_csv(model_union_path, model_union_rows, list(model_union_rows[0]))
        write_csv(union_summary_path, union_summary_rows, list(union_summary_rows[0]))
        write_csv(temporal_union_path, temporal_union_rows, list(temporal_union_rows[0]))
        diagnostic_paths = {
            "top2_concentration_by_layer_head": str(concentration_path),
            "sink_recent_overlap_by_layer_head": str(overlap_path),
            "top2_token_events": str(token_path),
            "top2_union_by_layer_query": str(layer_union_path),
            "top2_union_by_model_query": str(model_union_path),
            "top2_union_summary": str(union_summary_path),
            "top2_temporal_union": str(temporal_union_path),
        }

    plot_path = make_plot(output_dir, ppl_rows, args.plot_dpi) if args.make_plots else None
    sanity_checks = build_sanity_checks(ppl_rows, token_nll_rows, args.target_ratio)
    summary = {
        "args": vars(args),
        "resolved": {
            "ratio_grid": ratios,
            "modes": [mode.label for mode in modes],
            "total_tokens": int(input_ids.numel()),
            "device": str(input_device),
            "diagnostics_collected": diagnostic_consumed,
        },
        "sanity_checks": sanity_checks,
        "paths": {
            "ppl_by_selector": str(output_dir / "ppl_by_selector.csv"),
            "token_nll_by_selector": (
                str(output_dir / "token_nll_by_selector.csv") if args.write_token_nll else None
            ),
            "ppl_plot": plot_path,
            **diagnostic_paths,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["sanity_checks"], indent=2), flush=True)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
