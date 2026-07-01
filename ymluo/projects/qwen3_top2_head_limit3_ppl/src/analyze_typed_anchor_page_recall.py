from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_typed_page_guarded", False):
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

    guarded_register_fake._typed_page_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "ymluo/models/Qwen3-0.6B"
DEFAULT_TEXT_PATH = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt"

FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "he",
    "she",
    "it",
    "they",
    "we",
    "you",
    "i",
    "his",
    "her",
    "him",
    "them",
    "that",
    "this",
    "was",
    "is",
    "had",
    "have",
    "not",
}

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "TypedAnchorPageRecallCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query-level diagnostic for typed structural-anchor page recall over true remote top2 semantic tokens."
        )
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_dir", default="outputs/typed_anchor_page_recall")
    parser.add_argument("--total_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=1536)
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
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--exclude_sink_tokens", type=int, default=64)
    parser.add_argument("--exclude_recent_tokens", type=int, default=512)
    parser.add_argument("--fixed_page_size", type=int, default=64)
    parser.add_argument("--structural_max_page_tokens", type=int, default=128)
    parser.add_argument(
        "--structural_boundary_mode",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="paragraph cuts on newline/dialogue boundaries; sentence also cuts on standalone sentence punctuation.",
    )
    parser.add_argument(
        "--structural_neighbor_radius",
        type=int,
        default=1,
        help="For structural_adjacent recall, include +/- this many structural pages around each structural anchor page.",
    )
    parser.add_argument("--max_query_samples", type=int, default=0, help="Use <=0 to analyze all eval queries.")
    parser.add_argument("--query_stride", type=int, default=0)
    parser.add_argument("--oracle_page_counts", default="1,2,4,8,16")
    parser.add_argument("--coverage_thresholds", default="0.8,0.9,0.95")
    parser.add_argument("--write_per_query", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


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


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def fine_category(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "whitespace/newline"
    if "\n" in text:
        if any(ch in text for ch in '.?!:;,"\u201c\u201d\u2019'):
            return "sentence/dialogue boundary"
        return "newline"
    if re.fullmatch(r"[\.,!?;:\u2014\-\)\(\[\]\"'\u201c\u201d\u2019]+", stripped):
        return "punctuation"
    if re.fullmatch(r"\d+", stripped):
        return "number"
    if stripped.lower() in FUNCTION_WORDS:
        return "function word/pronoun"
    if stripped[:1].isupper():
        return "capitalized/name-like"
    return "content word/subword"


def anchor_type(text: str) -> str:
    category = fine_category(text)
    if category in {"sentence/dialogue boundary", "newline", "whitespace/newline", "punctuation"}:
        return "structural"
    if category in {"content word/subword", "capitalized/name-like", "number"}:
        return "semantic"
    return "other"


def is_strong_structural_boundary(text: str, boundary_mode: str) -> bool:
    stripped = text.strip()
    if "\n\n" in text:
        return True
    if "\n" in text and any(ch in text for ch in ".?!:;\u201d\""):
        return True
    if boundary_mode == "paragraph":
        return False
    return stripped in {".", "?", "!", ":", ";", "\u201d", '"'}


def build_structural_page_ids(
    token_texts: list[str],
    max_page_tokens: int,
    boundary_mode: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    page_ids: list[int] = [0] * len(token_texts)
    pages: list[tuple[int, int]] = []
    page_start = 0
    page_id = 0
    for index, text in enumerate(token_texts):
        page_ids[index] = page_id
        should_close = is_strong_structural_boundary(text, boundary_mode) or (
            index - page_start + 1
        ) >= max_page_tokens
        if should_close:
            pages.append((page_start, index + 1))
            page_start = index + 1
            page_id += 1
    if page_start < len(token_texts):
        for index in range(page_start, len(token_texts)):
            page_ids[index] = page_id
        pages.append((page_start, len(token_texts)))
    return page_ids, pages


def build_query_samples(prefill_tokens: int, eval_tokens: int, query_stride: int, max_query_samples: int) -> list[int]:
    queries = list(range(prefill_tokens, prefill_tokens + eval_tokens))
    if query_stride > 0:
        queries = queries[::query_stride]
    if max_query_samples > 0 and len(queries) > max_query_samples:
        if max_query_samples == 1:
            return [queries[len(queries) // 2]]
        step = (len(queries) - 1) / (max_query_samples - 1)
        indices = sorted({round(i * step) for i in range(max_query_samples)})
        queries = [queries[index] for index in indices]
    return queries


@dataclass
class RecallAccumulator:
    cases: int = 0
    anchor_events: int = 0
    anchor_pages: int = 0
    semantic_events: int = 0
    semantic_mass: float = 0.0
    covered_semantic_events: int = 0
    covered_semantic_mass: float = 0.0

    def add(
        self,
        anchor_events: int,
        anchor_pages: int,
        semantic_events: int,
        semantic_mass: float,
        covered_semantic_events: int,
        covered_semantic_mass: float,
    ) -> None:
        self.cases += 1
        self.anchor_events += anchor_events
        self.anchor_pages += anchor_pages
        self.semantic_events += semantic_events
        self.semantic_mass += semantic_mass
        self.covered_semantic_events += covered_semantic_events
        self.covered_semantic_mass += covered_semantic_mass

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "cases": self.cases,
            "anchor_events": self.anchor_events,
            "mean_anchor_events": self.anchor_events / self.cases if self.cases else 0.0,
            "mean_anchor_pages": self.anchor_pages / self.cases if self.cases else 0.0,
            "semantic_events": self.semantic_events,
            "semantic_mass": self.semantic_mass,
            "covered_semantic_events": self.covered_semantic_events,
            "covered_semantic_mass": self.covered_semantic_mass,
            "semantic_event_recall": (
                self.covered_semantic_events / self.semantic_events if self.semantic_events else 0.0
            ),
            "semantic_mass_recall": self.covered_semantic_mass / self.semantic_mass if self.semantic_mass else 0.0,
        }


@dataclass
class OracleAccumulator:
    cases: int = 0
    semantic_events: int = 0
    semantic_mass: float = 0.0
    covered_events: int = 0
    covered_mass: float = 0.0

    def add(self, semantic_events: int, semantic_mass: float, covered_events: int, covered_mass: float) -> None:
        self.cases += 1
        self.semantic_events += semantic_events
        self.semantic_mass += semantic_mass
        self.covered_events += covered_events
        self.covered_mass += covered_mass

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "cases": self.cases,
            "semantic_events": self.semantic_events,
            "semantic_mass": self.semantic_mass,
            "covered_events": self.covered_events,
            "covered_mass": self.covered_mass,
            "semantic_event_recall": self.covered_events / self.semantic_events if self.semantic_events else 0.0,
            "semantic_mass_recall": self.covered_mass / self.semantic_mass if self.semantic_mass else 0.0,
        }


@dataclass
class ThresholdAccumulator:
    cases: int = 0
    total_pages: int = 0
    hit_cases: int = 0

    def add(self, page_count: int | None) -> None:
        self.cases += 1
        if page_count is not None:
            self.hit_cases += 1
            self.total_pages += page_count

    def row(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "cases": self.cases,
            "hit_cases": self.hit_cases,
            "hit_fraction": self.hit_cases / self.cases if self.cases else 0.0,
            "mean_pages_to_threshold": self.total_pages / self.hit_cases if self.hit_cases else 0.0,
        }


class TypedAnchorPageRecallCollector:
    def __init__(
        self,
        query_tokens: set[int],
        top_fraction: float,
        exclude_sink_tokens: int,
        exclude_recent_tokens: int,
        fixed_page_size: int,
        structural_page_ids: list[int],
        structural_page_count: int,
        structural_neighbor_radius: int,
        anchor_types: list[str],
        oracle_page_counts: list[int],
        coverage_thresholds: list[float],
        write_per_query: bool,
    ) -> None:
        self.query_tokens = query_tokens
        self.top_fraction = top_fraction
        self.exclude_sink_tokens = exclude_sink_tokens
        self.exclude_recent_tokens = exclude_recent_tokens
        self.fixed_page_size = fixed_page_size
        self.structural_page_ids = structural_page_ids
        self.structural_page_count = structural_page_count
        self.structural_neighbor_radius = structural_neighbor_radius
        self.anchor_types = anchor_types
        self.oracle_page_counts = oracle_page_counts
        self.coverage_thresholds = coverage_thresholds
        self.write_per_query = write_per_query
        self.observed_query_tokens: set[int] = set()

        self.recall_by_scope: dict[tuple[str, str, tuple[int, ...]], RecallAccumulator] = defaultdict(
            RecallAccumulator
        )
        self.oracle_by_scope: dict[tuple[str, int, str, tuple[int, ...]], OracleAccumulator] = defaultdict(
            OracleAccumulator
        )
        self.threshold_by_scope: dict[tuple[str, float, str, tuple[int, ...]], ThresholdAccumulator] = defaultdict(
            ThresholdAccumulator
        )
        self.per_query_rows: list[dict[str, Any]] = []

    def _page_id(self, scheme: str, token_index: int) -> int:
        if scheme == "fixed":
            return token_index // self.fixed_page_size
        return self.structural_page_ids[token_index]

    def _anchor_pages(self, scheme: str, structural_tokens: list[int]) -> set[int]:
        pages = {self._page_id(scheme, token_index) for token_index in structural_tokens}
        if scheme != "structural_adjacent":
            return pages
        expanded: set[int] = set()
        for page in pages:
            start = max(0, page - self.structural_neighbor_radius)
            end = min(self.structural_page_count - 1, page + self.structural_neighbor_radius)
            expanded.update(range(start, end + 1))
        return expanded

    def _add_recall(
        self,
        scheme: str,
        layer: int,
        head: int,
        anchor_events: int,
        anchor_pages: int,
        semantic_events: int,
        semantic_mass: float,
        covered_semantic_events: int,
        covered_semantic_mass: float,
    ) -> None:
        scopes = [
            ("overall", ()),
            ("layer", (layer,)),
            ("layer_head", (layer, head)),
        ]
        for scope, key in scopes:
            self.recall_by_scope[(scheme, scope, key)].add(
                anchor_events,
                anchor_pages,
                semantic_events,
                semantic_mass,
                covered_semantic_events,
                covered_semantic_mass,
            )

    def _add_oracle(
        self,
        scheme: str,
        page_count: int,
        layer: int,
        head: int,
        semantic_events: int,
        semantic_mass: float,
        covered_events: int,
        covered_mass: float,
    ) -> None:
        scopes = [
            ("overall", ()),
            ("layer", (layer,)),
            ("layer_head", (layer, head)),
        ]
        for scope, key in scopes:
            self.oracle_by_scope[(scheme, page_count, scope, key)].add(
                semantic_events,
                semantic_mass,
                covered_events,
                covered_mass,
            )

    def _add_threshold(
        self,
        scheme: str,
        threshold: float,
        layer: int,
        head: int,
        pages_to_threshold: int | None,
    ) -> None:
        scopes = [
            ("overall", ()),
            ("layer", (layer,)),
            ("layer_head", (layer, head)),
        ]
        for scope, key in scopes:
            self.threshold_by_scope[(scheme, threshold, scope, key)].add(pages_to_threshold)

    def observe(self, layer: int, query_token: int, scores: torch.Tensor, query_index: int) -> None:
        if query_token not in self.query_tokens:
            return
        finite = torch.isfinite(scores[:, :, query_index, :])
        valid_count = min(int(finite[0, 0].sum().item()), query_token + 1)
        if valid_count <= 1:
            return
        history_count = valid_count - 1
        remote_end = max(0, history_count - self.exclude_recent_tokens)
        if remote_end <= self.exclude_sink_tokens:
            return
        top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
        row_scores = scores[0, :, query_index, :history_count].detach().float()
        top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices
        attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
            :, :history_count
        ]
        top_masses = torch.gather(attention_weights, dim=-1, index=top_indices)
        remote_mask = (top_indices >= self.exclude_sink_tokens) & (top_indices < remote_end)
        self.observed_query_tokens.add(query_token)

        for head in range(top_indices.shape[0]):
            head_indices = top_indices[head, remote_mask[head]].detach().cpu().tolist()
            head_masses = top_masses[head, remote_mask[head]].detach().cpu().tolist()
            if not head_indices:
                continue
            structural_tokens: list[int] = []
            semantic_tokens: list[tuple[int, float]] = []
            for token_index, mass in zip(head_indices, head_masses):
                anchor = self.anchor_types[int(token_index)]
                if anchor == "structural":
                    structural_tokens.append(int(token_index))
                elif anchor == "semantic":
                    semantic_tokens.append((int(token_index), float(mass)))
            semantic_events = len(semantic_tokens)
            semantic_mass = sum(mass for _, mass in semantic_tokens)
            for scheme in ["fixed", "structural", "structural_adjacent"]:
                structural_pages = self._anchor_pages(scheme, structural_tokens)
                covered_semantic = [
                    (token_index, mass)
                    for token_index, mass in semantic_tokens
                    if self._page_id(scheme, token_index) in structural_pages
                ]
                covered_events = len(covered_semantic)
                covered_mass = sum(mass for _, mass in covered_semantic)
                self._add_recall(
                    scheme,
                    layer,
                    head,
                    len(structural_tokens),
                    len(structural_pages),
                    semantic_events,
                    semantic_mass,
                    covered_events,
                    covered_mass,
                )
                if self.write_per_query:
                    self.per_query_rows.append(
                        {
                            "query_token": query_token,
                            "layer": layer,
                            "head": head,
                            "scheme": scheme,
                            "structural_anchor_events": len(structural_tokens),
                            "structural_anchor_pages": len(structural_pages),
                            "semantic_events": semantic_events,
                            "semantic_mass": semantic_mass,
                            "covered_semantic_events": covered_events,
                            "covered_semantic_mass": covered_mass,
                            "semantic_event_recall": covered_events / semantic_events if semantic_events else 0.0,
                            "semantic_mass_recall": covered_mass / semantic_mass if semantic_mass else 0.0,
                        }
                    )

                if scheme != "structural_adjacent":
                    semantic_by_page: dict[int, list[float]] = defaultdict(list)
                    for token_index, mass in semantic_tokens:
                        semantic_by_page[self._page_id(scheme, token_index)].append(mass)
                    page_stats = [
                        (page, len(masses), sum(masses)) for page, masses in semantic_by_page.items()
                    ]
                    page_stats.sort(key=lambda item: item[2], reverse=True)
                    cumulative_events = 0
                    cumulative_mass = 0.0
                    threshold_positions: dict[float, int | None] = {
                        threshold: None for threshold in self.coverage_thresholds
                    }
                    for position, (_, events, mass) in enumerate(page_stats, start=1):
                        cumulative_events += events
                        cumulative_mass += mass
                        for threshold in self.coverage_thresholds:
                            if (
                                threshold_positions[threshold] is None
                                and semantic_mass > 0
                                and cumulative_mass / semantic_mass >= threshold
                            ):
                                threshold_positions[threshold] = position
                    for page_count in self.oracle_page_counts:
                        selected = page_stats[:page_count]
                        covered_events_oracle = sum(item[1] for item in selected)
                        covered_mass_oracle = sum(item[2] for item in selected)
                        self._add_oracle(
                            scheme,
                            page_count,
                            layer,
                            head,
                            semantic_events,
                            semantic_mass,
                            covered_events_oracle,
                            covered_mass_oracle,
                        )
                    for threshold, pages_to_threshold in threshold_positions.items():
                        self._add_threshold(scheme, threshold, layer, head, pages_to_threshold)

    def recall_rows(self, scope: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (scheme, row_scope, key), acc in sorted(self.recall_by_scope.items()):
            if row_scope != scope:
                continue
            extra: dict[str, Any] = {"scheme": scheme, "scope": row_scope}
            if scope in {"layer", "layer_head"}:
                extra["layer"] = key[0]
            if scope == "layer_head":
                extra["head"] = key[1]
            rows.append(acc.row(extra))
        return rows

    def oracle_rows(self, scope: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (scheme, page_count, row_scope, key), acc in sorted(self.oracle_by_scope.items()):
            if row_scope != scope:
                continue
            extra: dict[str, Any] = {"scheme": scheme, "oracle_page_count": page_count, "scope": row_scope}
            if scope in {"layer", "layer_head"}:
                extra["layer"] = key[0]
            if scope == "layer_head":
                extra["head"] = key[1]
            rows.append(acc.row(extra))
        return rows

    def threshold_rows(self, scope: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (scheme, threshold, row_scope, key), acc in sorted(self.threshold_by_scope.items()):
            if row_scope != scope:
                continue
            extra: dict[str, Any] = {"scheme": scheme, "coverage_threshold": threshold, "scope": row_scope}
            if scope in {"layer", "layer_head"}:
                extra["layer"] = key[0]
            if scope == "layer_head":
                extra["head"] = key[1]
            rows.append(acc.row(extra))
        return rows


def _patched_eager_attention_forward(
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

    if _ACTIVE_COLLECTOR is not None:
        layer = int(getattr(module, "layer_idx", 0))
        query_count = scores.shape[-2]
        key_count = scores.shape[-1]
        chunk_query_start = key_count - query_count
        for query_index in range(query_count):
            query_token = chunk_query_start + query_index
            _ACTIVE_COLLECTOR.observe(layer, query_token, scores, query_index)

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
        setattr(modeling_qwen3, "eager_attention_forward", _patched_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_eager_attention_forward


@contextmanager
def active_collector(collector: TypedAnchorPageRecallCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_chunks = math.ceil(prefill_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
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
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return past_key_values


@torch.inference_mode()
def run_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    input_device: torch.device,
    collector: TypedAnchorPageRecallCollector,
) -> None:
    eval_end = prefill_tokens + eval_tokens
    total_chunks = math.ceil(eval_tokens / chunk_size)
    with active_collector(collector):
        for chunk_idx, start in enumerate(range(prefill_tokens, eval_end, chunk_size), start=1):
            end = min(start + chunk_size, eval_end)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end].to(input_device),
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": torch.arange(start, end, device=input_device),
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            print(f"eval chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
            outputs = model_forward(model, kwargs)
            past_key_values = outputs.past_key_values
            del outputs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.prefill_tokens + args.eval_tokens > args.total_tokens:
        raise ValueError("--prefill_tokens + --eval_tokens must be <= --total_tokens.")
    if args.fixed_page_size <= 0 or args.structural_max_page_tokens <= 0:
        raise ValueError("Page sizes must be positive.")
    if args.structural_neighbor_radius < 0:
        raise ValueError("--structural_neighbor_radius must be non-negative.")
    oracle_page_counts = [int(part) for part in args.oracle_page_counts.split(",") if part.strip()]
    coverage_thresholds = [float(part) for part in args.coverage_thresholds.split(",") if part.strip()]
    if not oracle_page_counts or any(count <= 0 for count in oracle_page_counts):
        raise ValueError("--oracle_page_counts must contain positive integers.")
    if not coverage_thresholds or any(not (0.0 < value <= 1.0) for value in coverage_thresholds):
        raise ValueError("--coverage_thresholds must contain values in (0, 1].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    if args.append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if args.require_total_tokens and len(token_ids) < args.total_tokens:
        raise ValueError(f"Need {args.total_tokens} tokens, got {len(token_ids)}.")
    token_ids = token_ids[: args.total_tokens]
    token_texts = [token_text(tokenizer, token_id) for token_id in token_ids]
    anchor_types = [anchor_type(text_piece) for text_piece in token_texts]
    structural_page_ids, structural_pages = build_structural_page_ids(
        token_texts,
        args.structural_max_page_tokens,
        args.structural_boundary_mode,
    )
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)

    query_samples = build_query_samples(args.prefill_tokens, args.eval_tokens, args.query_stride, args.max_query_samples)
    collector = TypedAnchorPageRecallCollector(
        query_tokens=set(query_samples),
        top_fraction=args.top_fraction,
        exclude_sink_tokens=args.exclude_sink_tokens,
        exclude_recent_tokens=args.exclude_recent_tokens,
        fixed_page_size=args.fixed_page_size,
        structural_page_ids=structural_page_ids,
        structural_page_count=len(structural_pages),
        structural_neighbor_radius=args.structural_neighbor_radius,
        anchor_types=anchor_types,
        oracle_page_counts=oracle_page_counts,
        coverage_thresholds=coverage_thresholds,
        write_per_query=args.write_per_query,
    )

    started = time.perf_counter()
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    run_eval(model, input_ids, past, args.prefill_tokens, args.eval_tokens, args.chunk_size, input_device, collector)
    seconds = time.perf_counter() - started

    recall_fields = [
        "scheme",
        "scope",
        "cases",
        "anchor_events",
        "mean_anchor_events",
        "mean_anchor_pages",
        "semantic_events",
        "semantic_mass",
        "covered_semantic_events",
        "covered_semantic_mass",
        "semantic_event_recall",
        "semantic_mass_recall",
    ]
    write_csv(output_dir / "page_recall_summary.csv", collector.recall_rows("overall"), recall_fields)
    write_csv(output_dir / "page_recall_by_layer.csv", collector.recall_rows("layer"), ["layer"] + recall_fields)
    write_csv(
        output_dir / "page_recall_by_layer_head.csv",
        collector.recall_rows("layer_head"),
        ["layer", "head"] + recall_fields,
    )

    oracle_fields = [
        "scheme",
        "oracle_page_count",
        "scope",
        "cases",
        "semantic_events",
        "semantic_mass",
        "covered_events",
        "covered_mass",
        "semantic_event_recall",
        "semantic_mass_recall",
    ]
    write_csv(output_dir / "oracle_page_coverage.csv", collector.oracle_rows("overall"), oracle_fields)
    write_csv(output_dir / "oracle_page_coverage_by_layer.csv", collector.oracle_rows("layer"), ["layer"] + oracle_fields)
    write_csv(
        output_dir / "oracle_page_coverage_by_layer_head.csv",
        collector.oracle_rows("layer_head"),
        ["layer", "head"] + oracle_fields,
    )

    threshold_fields = [
        "scheme",
        "coverage_threshold",
        "scope",
        "cases",
        "hit_cases",
        "hit_fraction",
        "mean_pages_to_threshold",
    ]
    write_csv(output_dir / "oracle_pages_to_threshold.csv", collector.threshold_rows("overall"), threshold_fields)
    write_csv(
        output_dir / "oracle_pages_to_threshold_by_layer.csv",
        collector.threshold_rows("layer"),
        ["layer"] + threshold_fields,
    )
    write_csv(
        output_dir / "oracle_pages_to_threshold_by_layer_head.csv",
        collector.threshold_rows("layer_head"),
        ["layer", "head"] + threshold_fields,
    )

    if args.write_per_query:
        write_csv(
            output_dir / "per_query_page_recall.csv",
            collector.per_query_rows,
            [
                "query_token",
                "layer",
                "head",
                "scheme",
                "structural_anchor_events",
                "structural_anchor_pages",
                "semantic_events",
                "semantic_mass",
                "covered_semantic_events",
                "covered_semantic_mass",
                "semantic_event_recall",
                "semantic_mass_recall",
            ],
        )

    summary = {
        "args": vars(args),
        "resolved": {
            "total_tokens_loaded": int(input_ids.numel()),
            "sampled_query_tokens_requested": query_samples,
            "sampled_query_tokens_observed": sorted(collector.observed_query_tokens),
            "structural_page_count": len(structural_pages),
            "structural_page_mean_tokens": (
                sum(end - start for start, end in structural_pages) / len(structural_pages)
                if structural_pages
                else 0.0
            ),
            "seconds": seconds,
            "metric_definitions": {
                "page_recall_summary": (
                    "Uses pages containing selected structural remote top2 anchors to cover selected semantic remote top2 tokens "
                    "for the same query/layer/head."
                ),
                "oracle_page_coverage": (
                    "Upper bound obtained by sorting pages by selected semantic remote top2 attention mass for the same query/layer/head."
                ),
            },
        },
        "paths": {
            "page_recall_summary": str(output_dir / "page_recall_summary.csv"),
            "page_recall_by_layer": str(output_dir / "page_recall_by_layer.csv"),
            "page_recall_by_layer_head": str(output_dir / "page_recall_by_layer_head.csv"),
            "oracle_page_coverage": str(output_dir / "oracle_page_coverage.csv"),
            "oracle_pages_to_threshold": str(output_dir / "oracle_pages_to_threshold.csv"),
            "per_query_page_recall": str(output_dir / "per_query_page_recall.csv") if args.write_per_query else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
