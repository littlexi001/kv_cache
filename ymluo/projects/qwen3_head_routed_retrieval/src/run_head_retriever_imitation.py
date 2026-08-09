from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


METHODS = (
    "position",
    "lexical",
    "semantic",
    "format",
    "repeat",
    "hybrid_lexical",
    "hybrid_semantic",
    "hybrid_format",
    "hybrid_repeat",
    "random",
)
NON_RANDOM_METHODS = METHODS[:-1]
STAT_NAMES = (
    "oracle_events",
    "hits",
    "oracle_mass",
    "intersection_mass",
    "retriever_mass",
    "history_mass",
    "remote_oracle_events",
    "remote_hits",
)

_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_PROBE: "HeadRetrieverProbe | None" = None


def _install_torchvision_fake_registration_guard() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_head_retrieval_guarded", False):
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

    guarded_register_fake._head_retrieval_guarded = True
    torch.library.register_fake = guarded_register_fake


_install_torchvision_fake_registration_guard()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def historical_budget(history_tokens: int, ratio: float) -> int:
    if history_tokens <= 0:
        return 0
    return min(history_tokens, max(1, int(math.ceil(history_tokens * ratio))))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


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
            retry = dict(kwargs)
            retry.pop("cache_position")
            return model(**retry)
        raise


def token_format_category(text: str) -> int:
    if "\n" in text or "\r" in text:
        return 0
    visible = text.strip()
    if not visible:
        return 1
    if any(char.isdigit() for char in visible) and not any(char.isalpha() for char in visible):
        return 2
    if all(not char.isalnum() for char in visible):
        return 3
    if any(char.isalpha() for char in visible) and visible.upper() == visible:
        return 4
    if any(char.isalpha() for char in visible):
        return 5
    return 6


class RetrieverBank:
    """Content/position retrievers that never inspect attention QK scores."""

    def __init__(
        self,
        token_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        decoded_tokens: list[str],
        *,
        ratio: float,
        query_window: int,
        block_size: int,
        repeat_max_n: int,
        sink_tokens: int,
        hybrid_position_fraction: float,
        random_seed: int,
    ) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if token_embeddings.ndim != 2 or token_embeddings.shape[0] != token_ids.shape[0]:
            raise ValueError("token_embeddings must have shape [tokens, dimensions]")
        self.token_ids = token_ids.to("cpu", dtype=torch.long).contiguous()
        self.token_embeddings = token_embeddings.to("cpu", dtype=torch.float32).contiguous()
        self.ratio = ratio
        self.query_window = query_window
        self.block_size = block_size
        self.repeat_max_n = repeat_max_n
        self.sink_tokens = sink_tokens
        if not (0.0 <= hybrid_position_fraction <= 1.0):
            raise ValueError("hybrid_position_fraction must be in [0, 1]")
        self.hybrid_position_fraction = hybrid_position_fraction
        self.random_seed = random_seed
        self.categories = torch.tensor(
            [token_format_category(value) for value in decoded_tokens], dtype=torch.long
        )
        self.category_count = 7
        self.block_token_sets: list[set[int]] = []
        block_embeddings: list[torch.Tensor] = []
        block_formats: list[torch.Tensor] = []
        for start in range(0, len(self.token_ids), block_size):
            end = min(start + block_size, len(self.token_ids))
            block_ids = self.token_ids[start:end]
            self.block_token_sets.append(set(int(value) for value in block_ids.tolist()))
            block_embeddings.append(self.token_embeddings[start:end].mean(dim=0))
            block_formats.append(
                F.one_hot(self.categories[start:end], num_classes=self.category_count)
                .float()
                .mean(dim=0)
            )
        self.block_embeddings = F.normalize(torch.stack(block_embeddings), dim=-1)
        self.block_formats = F.normalize(torch.stack(block_formats), dim=-1)
        self.postings: dict[int, list[int]] = {}
        for block_idx, token_set in enumerate(self.block_token_sets):
            for token_id in token_set:
                self.postings.setdefault(token_id, []).append(block_idx)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def _valid_block_features(
        self, current: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[set[int]], int]:
        full_blocks = current // self.block_size
        remainder = current % self.block_size
        embeddings = self.block_embeddings[:full_blocks]
        formats = self.block_formats[:full_blocks]
        token_sets = list(self.block_token_sets[:full_blocks])
        if remainder:
            start = full_blocks * self.block_size
            partial_embedding = F.normalize(
                self.token_embeddings[start:current].mean(dim=0), dim=0
            ).unsqueeze(0)
            partial_format = F.normalize(
                F.one_hot(
                    self.categories[start:current], num_classes=self.category_count
                ).float().mean(dim=0),
                dim=0,
            ).unsqueeze(0)
            embeddings = torch.cat([embeddings, partial_embedding], dim=0)
            formats = torch.cat([formats, partial_format], dim=0)
            token_sets.append(set(int(value) for value in self.token_ids[start:current].tolist()))
        return embeddings, formats, token_sets, full_blocks

    def _expand_block_scores(self, block_scores: torch.Tensor, current: int) -> torch.Tensor:
        return block_scores.repeat_interleave(self.block_size)[:current].clone()

    def scores(self, current: int) -> dict[str, torch.Tensor]:
        if current <= 0 or current >= len(self.token_ids):
            raise ValueError(f"current must be in [1, {len(self.token_ids) - 1}], got {current}")
        positions = torch.arange(current, dtype=torch.float32)
        recency = positions / max(1.0, float(current - 1))
        tie_break = recency * 1e-5
        query_start = max(0, current - self.query_window + 1)
        query_ids = self.token_ids[query_start : current + 1]
        query_embedding = F.normalize(
            self.token_embeddings[query_start : current + 1].mean(dim=0), dim=0
        )
        query_format = F.normalize(
            F.one_hot(
                self.categories[query_start : current + 1], num_classes=self.category_count
            ).float().mean(dim=0),
            dim=0,
        )
        block_embeddings, block_formats, block_sets, full_blocks = self._valid_block_features(current)

        position_score = recency.clone()
        position_score[: min(current, self.sink_tokens)] += 2.0

        lexical_blocks = torch.zeros(len(block_sets), dtype=torch.float32)
        unique_query_ids = set(int(value) for value in query_ids.tolist())
        for token_id in unique_query_ids:
            matched_blocks = [idx for idx in self.postings.get(token_id, ()) if idx < full_blocks]
            if len(block_sets) > full_blocks and token_id in block_sets[-1]:
                matched_blocks.append(full_blocks)
            inverse_block_frequency = math.log(
                (1.0 + len(block_sets)) / (1.0 + len(matched_blocks))
            ) + 1.0
            if matched_blocks:
                lexical_blocks[torch.tensor(matched_blocks, dtype=torch.long)] += inverse_block_frequency
        lexical_score = self._expand_block_scores(lexical_blocks, current)
        direct_match = torch.zeros(current, dtype=torch.bool)
        for token_id in unique_query_ids:
            direct_match |= self.token_ids[:current] == token_id
        lexical_score += direct_match.float() * 0.25 + tie_break

        semantic_blocks = block_embeddings @ query_embedding
        semantic_score = self._expand_block_scores(semantic_blocks, current) + tie_break

        format_blocks = block_formats @ query_format
        format_score = self._expand_block_scores(format_blocks, current)
        query_categories = torch.unique(self.categories[query_start : current + 1])
        category_match = torch.zeros(current, dtype=torch.bool)
        for category in query_categories.tolist():
            category_match |= self.categories[:current] == int(category)
        format_score += category_match.float() * 0.05 + tie_break

        repeat_score = torch.zeros(current, dtype=torch.float32)
        for width in range(1, min(self.repeat_max_n, current + 1) + 1):
            target = self.token_ids[current - width + 1 : current + 1]
            candidate_end = torch.arange(width - 1, current)
            matches = torch.ones(len(candidate_end), dtype=torch.bool)
            for offset in range(width):
                matches &= self.token_ids[candidate_end - width + 1 + offset] == target[offset]
            repeat_score[candidate_end[matches]] = float(width)
        repeat_score += tie_break

        modulus = 2_147_483_647
        random_values = (
            (torch.arange(current, dtype=torch.int64) + 1) * 1_103_515_245
            + (current + 1) * 12_345
            + self.random_seed
        ) % modulus
        random_score = random_values.float() / float(modulus)

        return {
            "position": position_score,
            "lexical": lexical_score,
            "semantic": semantic_score,
            "format": format_score,
            "repeat": repeat_score,
            "random": random_score,
        }

    def selections(self, current: int) -> dict[str, torch.Tensor]:
        if current not in self._cache:
            budget = historical_budget(current, self.ratio)
            score_map = self.scores(current)
            selections = {
                method: torch.topk(score, k=budget, largest=True).indices.sort().values
                for method, score in score_map.items()
            }
            position_count = min(
                budget,
                max(
                    min(self.sink_tokens, budget),
                    int(math.ceil(budget * self.hybrid_position_fraction)),
                ),
            )
            position_scaffold = torch.topk(
                score_map["position"], k=position_count, largest=True
            ).indices
            scaffold_mask = torch.zeros(current, dtype=torch.bool)
            scaffold_mask[position_scaffold] = True
            for specialist in ("lexical", "semantic", "format", "repeat"):
                remaining = budget - position_count
                if remaining:
                    specialist_score = score_map[specialist].clone()
                    specialist_score[scaffold_mask] = -torch.inf
                    specialist_indices = torch.topk(
                        specialist_score, k=remaining, largest=True
                    ).indices
                    hybrid = torch.cat([position_scaffold, specialist_indices]).sort().values
                else:
                    hybrid = position_scaffold.sort().values
                selections[f"hybrid_{specialist}"] = hybrid
            self._cache[current] = {method: selections[method] for method in METHODS}
        return self._cache[current]

    def discard_before(self, current: int) -> None:
        stale = [key for key in self._cache if key < current]
        for key in stale:
            del self._cache[key]


class HeadRetrieverProbe:
    def __init__(
        self,
        bank: RetrieverBank,
        *,
        layer_count: int,
        head_count: int,
        kv_head_count: int,
        train_start: int,
        train_end: int,
        test_end: int,
        ratio: float,
        sink_tokens: int,
        recent_tokens: int,
        device: torch.device,
    ) -> None:
        self.bank = bank
        self.layer_count = layer_count
        self.head_count = head_count
        self.kv_head_count = kv_head_count
        self.train_start = train_start
        self.train_end = train_end
        self.test_end = test_end
        self.ratio = ratio
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        shape = (2, layer_count, head_count, len(METHODS))
        self.stats = {
            name: torch.zeros(shape, dtype=torch.float64, device=device) for name in STAT_NAMES
        }
        self.query_rows = torch.zeros((2, layer_count), dtype=torch.int64, device=device)
        self.best_method_indices: torch.Tensor | None = None
        self.balanced_method_indices: torch.Tensor | None = None
        self.gqa_union_tokens = torch.zeros(
            (layer_count, kv_head_count), dtype=torch.float64, device=device
        )
        self.gqa_history_tokens = torch.zeros_like(self.gqa_union_tokens)
        self.gqa_single_budget = torch.zeros_like(self.gqa_union_tokens)
        self.gqa_query_rows = torch.zeros_like(self.gqa_union_tokens)

    def split_index(self, current: int) -> int | None:
        if self.train_start <= current < self.train_end:
            return 0
        if self.train_end <= current < self.test_end:
            return 1
        return None

    def finalize_choices(self) -> None:
        train_hits = self.stats["hits"][0, :, :, : len(NON_RANDOM_METHODS)]
        train_events = self.stats["oracle_events"][0, :, :, : len(NON_RANDOM_METHODS)]
        train_recall = train_hits / train_events.clamp_min(1.0)
        remote_hits = self.stats["remote_hits"][0, :, :, : len(NON_RANDOM_METHODS)]
        remote_events = self.stats["remote_oracle_events"][0, :, :, : len(NON_RANDOM_METHODS)]
        remote_recall = remote_hits / remote_events.clamp_min(1.0)
        self.best_method_indices = train_recall.argmax(dim=-1)
        self.balanced_method_indices = (0.5 * train_recall + 0.5 * remote_recall).argmax(dim=-1)

    @torch.inference_mode()
    def update(
        self,
        module: torch.nn.Module,
        scores: torch.Tensor,
        full_weights: torch.Tensor,
    ) -> None:
        if scores.shape[0] != 1:
            raise ValueError("The pilot currently requires batch size 1.")
        layer_idx = int(getattr(module, "layer_idx", 0))
        _, head_count, query_count, key_count = scores.shape
        if head_count != self.head_count:
            raise ValueError(f"Expected {self.head_count} query heads, got {head_count}.")
        past_tokens = key_count - query_count
        for query_idx in range(query_count):
            current = past_tokens + query_idx
            split_idx = self.split_index(current)
            if split_idx is None or current <= 0:
                continue
            selections_cpu = self.bank.selections(current)
            budget = historical_budget(current, self.ratio)
            history_scores = scores[0, :, query_idx, :current]
            oracle_indices = torch.topk(history_scores, k=budget, dim=-1, largest=True).indices
            history_probs = full_weights[0, :, query_idx, :current].float()
            oracle_prob = history_probs.gather(1, oracle_indices)
            oracle_mass = oracle_prob.sum(dim=-1)
            history_mass = history_probs.sum(dim=-1)
            remote_cutoff = max(self.sink_tokens, current - self.recent_tokens)
            oracle_remote = (oracle_indices >= self.sink_tokens) & (oracle_indices < remote_cutoff)
            self.query_rows[split_idx, layer_idx] += 1
            for method_idx, method in enumerate(METHODS):
                selected = selections_cpu[method].to(scores.device)
                selected_mask = torch.zeros(current, dtype=torch.bool, device=scores.device)
                selected_mask[selected] = True
                oracle_hit = selected_mask[oracle_indices]
                self.stats["oracle_events"][split_idx, layer_idx, :, method_idx] += budget
                self.stats["hits"][split_idx, layer_idx, :, method_idx] += oracle_hit.sum(dim=-1)
                self.stats["oracle_mass"][split_idx, layer_idx, :, method_idx] += oracle_mass
                self.stats["intersection_mass"][split_idx, layer_idx, :, method_idx] += (
                    oracle_prob * oracle_hit
                ).sum(dim=-1)
                self.stats["retriever_mass"][split_idx, layer_idx, :, method_idx] += (
                    history_probs.index_select(1, selected).sum(dim=-1)
                )
                self.stats["history_mass"][split_idx, layer_idx, :, method_idx] += history_mass
                self.stats["remote_oracle_events"][split_idx, layer_idx, :, method_idx] += (
                    oracle_remote.sum(dim=-1)
                )
                self.stats["remote_hits"][split_idx, layer_idx, :, method_idx] += (
                    oracle_hit & oracle_remote
                ).sum(dim=-1)
            if split_idx == 1 and self.best_method_indices is not None:
                self._update_gqa_union(layer_idx, current, budget, selections_cpu, scores.device)
        if layer_idx == self.layer_count - 1 and query_count:
            self.bank.discard_before(past_tokens + query_count)

    def _update_gqa_union(
        self,
        layer_idx: int,
        current: int,
        budget: int,
        selections: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        if self.head_count % self.kv_head_count:
            raise ValueError("Query-head count must be divisible by KV-head count for GQA reporting.")
        heads_per_group = self.head_count // self.kv_head_count
        assert self.best_method_indices is not None
        for group_idx in range(self.kv_head_count):
            start = group_idx * heads_per_group
            end = start + heads_per_group
            union_mask = torch.zeros(current, dtype=torch.bool, device=device)
            for head_idx in range(start, end):
                method_idx = int(self.best_method_indices[layer_idx, head_idx])
                union_mask[selections[METHODS[method_idx]].to(device)] = True
            self.gqa_union_tokens[layer_idx, group_idx] += union_mask.sum()
            self.gqa_history_tokens[layer_idx, group_idx] += current
            self.gqa_single_budget[layer_idx, group_idx] += budget
            self.gqa_query_rows[layer_idx, group_idx] += 1

    def metric_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stats = {name: value.detach().cpu() for name, value in self.stats.items()}
        for split_idx, split_name in enumerate(("train", "test")):
            for layer_idx in range(self.layer_count):
                for head_idx in range(self.head_count):
                    for method_idx, method in enumerate(METHODS):
                        oracle_events = float(stats["oracle_events"][split_idx, layer_idx, head_idx, method_idx])
                        hits = float(stats["hits"][split_idx, layer_idx, head_idx, method_idx])
                        oracle_mass = float(stats["oracle_mass"][split_idx, layer_idx, head_idx, method_idx])
                        intersection_mass = float(
                            stats["intersection_mass"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        retriever_mass = float(
                            stats["retriever_mass"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        history_mass = float(
                            stats["history_mass"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        remote_events = float(
                            stats["remote_oracle_events"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        remote_hits = float(
                            stats["remote_hits"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        recall = safe_div(hits, oracle_events)
                        rows.append(
                            {
                                "split": split_name,
                                "layer": layer_idx,
                                "head": head_idx,
                                "method": method,
                                "oracle_events": int(oracle_events),
                                "hits": int(hits),
                                "position_recall": recall,
                                "equal_budget_jaccard": safe_div(recall, 2.0 - recall),
                                "oracle_mass_recall": safe_div(intersection_mass, oracle_mass),
                                "retriever_history_attention_mass_fraction": safe_div(
                                    retriever_mass, history_mass
                                ),
                                "remote_oracle_events": int(remote_events),
                                "remote_hits": int(remote_hits),
                                "remote_position_recall": safe_div(remote_hits, remote_events),
                            }
                        )
        return rows

    def assignment_rows(self) -> list[dict[str, Any]]:
        if self.best_method_indices is None or self.balanced_method_indices is None:
            raise RuntimeError("finalize_choices must be called before writing assignments")
        stats = {name: value.detach().cpu() for name, value in self.stats.items()}
        best = self.best_method_indices.detach().cpu()
        balanced = self.balanced_method_indices.detach().cpu()
        rows: list[dict[str, Any]] = []
        for layer_idx in range(self.layer_count):
            for head_idx in range(self.head_count):
                method_idx = int(best[layer_idx, head_idx])
                balanced_idx = int(balanced[layer_idx, head_idx])
                test_recall_by_method = (
                    stats["hits"][1, layer_idx, head_idx, : len(NON_RANDOM_METHODS)]
                    / stats["oracle_events"][1, layer_idx, head_idx, : len(NON_RANDOM_METHODS)].clamp_min(1)
                )
                oracle_test_idx = int(test_recall_by_method.argmax())

                def recall(split: int, index: int) -> float:
                    return safe_div(
                        float(stats["hits"][split, layer_idx, head_idx, index]),
                        float(stats["oracle_events"][split, layer_idx, head_idx, index]),
                    )

                def remote_recall(split: int, index: int) -> float:
                    return safe_div(
                        float(stats["remote_hits"][split, layer_idx, head_idx, index]),
                        float(stats["remote_oracle_events"][split, layer_idx, head_idx, index]),
                    )

                rows.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "train_best_method": METHODS[method_idx],
                        "train_position_recall": recall(0, method_idx),
                        "test_position_recall": recall(1, method_idx),
                        "test_remote_position_recall": remote_recall(1, method_idx),
                        "train_balanced_method": METHODS[balanced_idx],
                        "balanced_test_position_recall": recall(1, balanced_idx),
                        "balanced_test_remote_position_recall": remote_recall(1, balanced_idx),
                        "diagnostic_test_oracle_best_method": METHODS[oracle_test_idx],
                        "diagnostic_test_oracle_best_recall": recall(1, oracle_test_idx),
                    }
                )
        return rows

    def aggregate_rows(self) -> list[dict[str, Any]]:
        if self.best_method_indices is None or self.balanced_method_indices is None:
            raise RuntimeError("finalize_choices must be called before aggregation")
        stats = {name: value.detach().cpu() for name, value in self.stats.items()}
        rows: list[dict[str, Any]] = []
        for split_idx, split_name in enumerate(("train", "test")):
            for method_idx, method in enumerate(METHODS):
                hits = float(stats["hits"][split_idx, :, :, method_idx].sum())
                events = float(stats["oracle_events"][split_idx, :, :, method_idx].sum())
                remote_hits = float(stats["remote_hits"][split_idx, :, :, method_idx].sum())
                remote_events = float(
                    stats["remote_oracle_events"][split_idx, :, :, method_idx].sum()
                )
                intersection_mass = float(
                    stats["intersection_mass"][split_idx, :, :, method_idx].sum()
                )
                oracle_mass = float(stats["oracle_mass"][split_idx, :, :, method_idx].sum())
                rows.append(
                    {
                        "split": split_name,
                        "policy": f"homogeneous:{method}",
                        "position_recall": safe_div(hits, events),
                        "remote_position_recall": safe_div(remote_hits, remote_events),
                        "oracle_mass_recall": safe_div(intersection_mass, oracle_mass),
                    }
                )

        for label, choices in (
            ("head_routed:train_best", self.best_method_indices.detach().cpu()),
            ("head_routed:train_balanced", self.balanced_method_indices.detach().cpu()),
        ):
            for split_idx, split_name in enumerate(("train", "test")):
                totals = {name: 0.0 for name in ("hits", "events", "remote_hits", "remote_events", "intersection_mass", "oracle_mass")}
                for layer_idx in range(self.layer_count):
                    for head_idx in range(self.head_count):
                        method_idx = int(choices[layer_idx, head_idx])
                        totals["hits"] += float(stats["hits"][split_idx, layer_idx, head_idx, method_idx])
                        totals["events"] += float(
                            stats["oracle_events"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        totals["remote_hits"] += float(
                            stats["remote_hits"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        totals["remote_events"] += float(
                            stats["remote_oracle_events"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        totals["intersection_mass"] += float(
                            stats["intersection_mass"][split_idx, layer_idx, head_idx, method_idx]
                        )
                        totals["oracle_mass"] += float(
                            stats["oracle_mass"][split_idx, layer_idx, head_idx, method_idx]
                        )
                rows.append(
                    {
                        "split": split_name,
                        "policy": label,
                        "position_recall": safe_div(totals["hits"], totals["events"]),
                        "remote_position_recall": safe_div(
                            totals["remote_hits"], totals["remote_events"]
                        ),
                        "oracle_mass_recall": safe_div(
                            totals["intersection_mass"], totals["oracle_mass"]
                        ),
                    }
                )
        return rows

    def gqa_rows(self) -> list[dict[str, Any]]:
        union = self.gqa_union_tokens.detach().cpu()
        history = self.gqa_history_tokens.detach().cpu()
        budget = self.gqa_single_budget.detach().cpu()
        queries = self.gqa_query_rows.detach().cpu()
        rows: list[dict[str, Any]] = []
        for layer_idx in range(self.layer_count):
            for group_idx in range(self.kv_head_count):
                rows.append(
                    {
                        "layer": layer_idx,
                        "kv_group": group_idx,
                        "query_rows": int(queries[layer_idx, group_idx]),
                        "mean_union_fraction_of_history": safe_div(
                            float(union[layer_idx, group_idx]), float(history[layer_idx, group_idx])
                        ),
                        "union_vs_single_head_budget": safe_div(
                            float(union[layer_idx, group_idx]), float(budget[layer_idx, group_idx])
                        ),
                    }
                )
        return rows


def _probe_eager_attention_forward(
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
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if _ACTIVE_PROBE is not None:
        _ACTIVE_PROBE.update(module, scores, attention_weights)
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
        modeling_qwen3.eager_attention_forward = _probe_eager_attention_forward
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _probe_eager_attention_forward


@contextmanager
def activate_probe(probe: HeadRetrieverProbe | None):
    global _ACTIVE_PROBE
    previous = _ACTIVE_PROBE
    _ACTIVE_PROBE = probe
    try:
        yield
    finally:
        _ACTIVE_PROBE = previous


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    input_device: torch.device,
) -> tuple[Any, torch.Tensor]:
    past = None
    previous_logits: torch.Tensor | None = None
    for start in range(0, prefill_tokens, chunk_size):
        end = min(prefill_tokens, start + chunk_size)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        outputs = model_forward(model, kwargs)
        past = outputs.past_key_values
        previous_logits = outputs.logits[:, -1, :].detach()
    if previous_logits is None:
        raise RuntimeError("Prefill produced no logits")
    return past, previous_logits


@torch.inference_mode()
def evaluate_phase(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past: Any,
    previous_logits: torch.Tensor,
    *,
    start: int,
    end: int,
    chunk_size: int,
    input_device: torch.device,
    probe: HeadRetrieverProbe,
    phase: str,
    log_every: int,
) -> tuple[Any, torch.Tensor, float, int]:
    total_nll = 0.0
    token_count = 0
    chunk_count = math.ceil((end - start) / chunk_size)
    for chunk_idx, chunk_start in enumerate(range(start, end, chunk_size), start=1):
        chunk_end = min(end, chunk_start + chunk_size)
        chunk = input_ids[:, chunk_start:chunk_end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(chunk_start, chunk_end, device=input_device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        with activate_probe(probe):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted = torch.cat([previous_logits.unsqueeze(1), logits[:, :-1]], dim=1)
        losses = F.cross_entropy(
            shifted.reshape(-1, shifted.shape[-1]).float(), chunk.reshape(-1), reduction="sum"
        )
        total_nll += float(losses)
        token_count += int(chunk.numel())
        previous_logits = logits[:, -1].detach()
        past = outputs.past_key_values
        if log_every > 0 and (chunk_idx % log_every == 0 or chunk_idx == chunk_count):
            print(
                f"phase={phase} chunk={chunk_idx}/{chunk_count} tokens={chunk_start}:{chunk_end}",
                flush=True,
            )
    return past, previous_logits, total_nll, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imitate per-head oracle Top-2% with heterogeneous external retrievers."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--train_queries", type=int, default=128)
    parser.add_argument("--test_queries", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--query_window", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--repeat_max_n", type=int, default=4)
    parser.add_argument("--hybrid_position_fraction", type=float, default=0.5)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_tokens", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=20260714)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with Path(args.text_path).open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(args.max_chars) if args.max_chars > 0 else handle.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids_list = tokenizer(text, add_special_tokens=args.add_special_tokens)["input_ids"]
    total_tokens = args.prefill_tokens + args.train_queries + args.test_queries
    if len(token_ids_list) < total_tokens:
        raise ValueError(f"Tokenized text has {len(token_ids_list)} tokens; need {total_tokens}.")
    token_ids_list = token_ids_list[:total_tokens]
    input_ids = torch.tensor(token_ids_list, dtype=torch.long).view(1, -1)

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
    input_device = pick_input_device(model, requested_device)
    install_attention_patch()

    with torch.inference_mode():
        token_embeddings = (
            model.get_input_embeddings()(input_ids.to(input_device))[0].float().cpu()
        )
    decoded_tokens = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids_list
    ]
    bank = RetrieverBank(
        input_ids[0],
        token_embeddings,
        decoded_tokens,
        ratio=args.ratio,
        query_window=args.query_window,
        block_size=args.block_size,
        repeat_max_n=args.repeat_max_n,
        sink_tokens=args.sink_tokens,
        hybrid_position_fraction=args.hybrid_position_fraction,
        random_seed=args.random_seed,
    )
    train_start = args.prefill_tokens
    train_end = train_start + args.train_queries
    test_end = train_end + args.test_queries
    probe = HeadRetrieverProbe(
        bank,
        layer_count=int(model.config.num_hidden_layers),
        head_count=int(model.config.num_attention_heads),
        kv_head_count=int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)),
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
        ratio=args.ratio,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        device=input_device,
    )

    print("prefill starting", flush=True)
    past, previous_logits = prefill_cache(
        model, input_ids, args.prefill_tokens, args.chunk_size, input_device
    )
    print("train-query probe starting", flush=True)
    past, previous_logits, train_nll, train_count = evaluate_phase(
        model,
        input_ids,
        past,
        previous_logits,
        start=train_start,
        end=train_end,
        chunk_size=args.chunk_size,
        input_device=input_device,
        probe=probe,
        phase="train",
        log_every=args.log_every,
    )
    probe.finalize_choices()
    print("test-query probe starting", flush=True)
    past, previous_logits, test_nll, test_count = evaluate_phase(
        model,
        input_ids,
        past,
        previous_logits,
        start=train_end,
        end=test_end,
        chunk_size=args.chunk_size,
        input_device=input_device,
        probe=probe,
        phase="test",
        log_every=args.log_every,
    )

    metric_rows = probe.metric_rows()
    assignment_rows = probe.assignment_rows()
    aggregate_rows = probe.aggregate_rows()
    gqa_rows = probe.gqa_rows()
    write_csv(output_dir / "head_retriever_metrics.csv", metric_rows)
    write_csv(output_dir / "head_assignments.csv", assignment_rows)
    write_csv(output_dir / "aggregate_retriever_metrics.csv", aggregate_rows)
    write_csv(output_dir / "gqa_union_by_layer_group.csv", gqa_rows)

    test_aggregates = [row for row in aggregate_rows if row["split"] == "test"]
    best_test_homogeneous = max(
        (row for row in test_aggregates if str(row["policy"]).startswith("homogeneous:") and row["policy"] != "homogeneous:random"),
        key=lambda row: float(row["position_recall"]),
    )
    train_aggregates = [row for row in aggregate_rows if row["split"] == "train"]
    train_selected_homogeneous = max(
        (
            row
            for row in train_aggregates
            if str(row["policy"]).startswith("homogeneous:")
            and row["policy"] != "homogeneous:random"
        ),
        key=lambda row: float(row["position_recall"]),
    )
    fair_homogeneous_test = next(
        row
        for row in test_aggregates
        if row["policy"] == train_selected_homogeneous["policy"]
    )
    routed = next(row for row in test_aggregates if row["policy"] == "head_routed:train_best")
    assignment_counts: dict[str, int] = {method: 0 for method in NON_RANDOM_METHODS}
    for row in assignment_rows:
        assignment_counts[str(row["train_best_method"])] += 1
    summary = {
        "args": vars(args),
        "resolved": {
            "methods": METHODS,
            "total_tokens": total_tokens,
            "num_layers": probe.layer_count,
            "num_query_heads": probe.head_count,
            "num_kv_heads": probe.kv_head_count,
            "input_device": str(input_device),
        },
        "full_attention_ppl": {
            "train": math.exp(train_nll / max(1, train_count)),
            "test": math.exp(test_nll / max(1, test_count)),
        },
        "headline": {
            "train_selected_homogeneous_test": fair_homogeneous_test,
            "best_test_homogeneous_policy_diagnostic": best_test_homogeneous,
            "head_routed_train_selected_test": routed,
            "absolute_recall_gain_vs_train_selected_homogeneous": float(
                routed["position_recall"]
            )
            - float(fair_homogeneous_test["position_recall"]),
            "absolute_recall_gain_vs_best_test_homogeneous_diagnostic": float(
                routed["position_recall"]
            )
            - float(best_test_homogeneous["position_recall"]),
            "head_assignment_counts": assignment_counts,
            "mean_gqa_union_fraction": safe_div(
                float(probe.gqa_union_tokens.sum()), float(probe.gqa_history_tokens.sum())
            ),
            "mean_gqa_union_vs_single_head_budget": safe_div(
                float(probe.gqa_union_tokens.sum()), float(probe.gqa_single_budget.sum())
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "paths": {
            "head_retriever_metrics": str(output_dir / "head_retriever_metrics.csv"),
            "head_assignments": str(output_dir / "head_assignments.csv"),
            "aggregate_retriever_metrics": str(output_dir / "aggregate_retriever_metrics.csv"),
            "gqa_union_by_layer_group": str(output_dir / "gqa_union_by_layer_group.csv"),
        },
        "limitations": [
            "Within-document temporal split; final evaluation must split documents and tasks.",
            "Imitation metrics only; retrieved masks have not yet been evaluated for NLL/PPL.",
            "Semantic retrieval uses frozen input embeddings, not a separately trained encoder.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
