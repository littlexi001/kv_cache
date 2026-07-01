from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402
from run_qabs_evidence_span_coverage import evidence_spans  # noqa: E402


_ORIGINAL_EAGER_ATTENTION_FORWARD: Any | None = None
_ACTIVE_COLLECTOR: "SeparateQKCollector | None" = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze top2 recovery when q and k use separate low-rank SVD subspaces."
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=2)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063006)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--heads", default="0,4,8,12")
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--sink_tokens", type=int, default=10)
    parser.add_argument("--recent_tokens", type=int, default=16)
    parser.add_argument("--max_query_tokens_per_task", type=int, default=2)
    parser.add_argument("--max_tokens_per_group_per_row", type=int, default=32)
    parser.add_argument("--center_q", type=str2bool, default=True)
    parser.add_argument("--center_k", type=str2bool, default=True)
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(max_count))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid {name} range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    if not selected:
        raise ValueError(f"No {name} selected from spec {spec!r}")
    return sorted(selected)


def parse_ranks(value: str, max_rank: int) -> list[int]:
    ranks = sorted({int(part) for part in value.split(",") if part.strip()})
    ranks = [rank for rank in ranks if 1 <= rank <= max_rank]
    if max_rank not in ranks:
        ranks.append(max_rank)
    return sorted(ranks)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_query_tokens(prefill_tokens: int, query_tokens: int, max_query_tokens: int) -> list[int]:
    positions = list(range(prefill_tokens, prefill_tokens + query_tokens))
    if max_query_tokens <= 0 or len(positions) <= max_query_tokens:
        return positions
    return positions[-max_query_tokens:]


def rank_at_energy(singular_values: torch.Tensor, target: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum().item())
    if total <= 0.0:
        return 0
    cdf = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cdf, torch.tensor(target, device=cdf.device)).item()) + 1


def spectral_entropy_effective_rank(singular_values: torch.Tensor) -> float:
    energy = singular_values.square()
    total = energy.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    prob = energy / total
    entropy = -(prob * torch.log(prob.clamp_min(1e-30))).sum()
    return float(torch.exp(entropy).item())


@dataclass
class MeanAccumulator:
    cases: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, values: dict[str, float]) -> None:
        self.cases += 1
        for key, value in values.items():
            if math.isfinite(value):
                self.sums[key] += float(value)

    def row(self, extra: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        row = {**extra, "cases": self.cases}
        for field_name in fields:
            row[field_name] = self.sums.get(field_name, 0.0) / self.cases if self.cases else 0.0
        return row


class SeparateQKCollector:
    def __init__(
        self,
        selected_layers: list[int],
        selected_heads: list[int],
        ranks: list[int],
        top_fraction: float,
        sink_tokens: int,
        recent_tokens: int,
        max_tokens_per_group_per_row: int,
        center_q: bool,
        center_k: bool,
        svd_device: torch.device,
        svd_dtype: torch.dtype,
    ) -> None:
        self.selected_layers = set(selected_layers)
        self.selected_heads = set(selected_heads)
        self.ranks = ranks
        self.top_fraction = top_fraction
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.max_tokens_per_group_per_row = max_tokens_per_group_per_row
        self.center_q = center_q
        self.center_k = center_k
        self.svd_device = svd_device
        self.svd_dtype = svd_dtype

        self.query_tokens: set[int] = set()
        self.base_groups: dict[str, set[int]] = {}
        self.q_history: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.observed_rows = 0
        self.skipped_q_svd_rows = 0
        self.skipped_k_svd_rows = 0
        self.q_svd_accumulators: dict[tuple[str, int | str, int | str], MeanAccumulator] = defaultdict(MeanAccumulator)
        self.k_svd_accumulators: dict[tuple[str, int | str, int | str], MeanAccumulator] = defaultdict(MeanAccumulator)
        self.group_accumulators: dict[tuple[str, str, int | str], MeanAccumulator] = defaultdict(MeanAccumulator)
        self.recall_accumulators: dict[tuple[str, int | str, int | str, int], MeanAccumulator] = defaultdict(
            MeanAccumulator
        )

    def set_task(self, *, query_tokens: set[int], base_groups: dict[str, set[int]]) -> None:
        self.query_tokens = query_tokens
        self.base_groups = base_groups
        self.q_history.clear()

    def observe_chunk(
        self,
        *,
        layer: int,
        chunk_query_start: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        if layer not in self.selected_layers:
            return
        for query_index in range(query_states.shape[-2]):
            query_token = chunk_query_start + query_index
            if query_token not in self.query_tokens:
                continue
            finite = torch.isfinite(scores[:, :, query_index, :])
            valid_count = int(finite[0, 0].sum().item())
            if valid_count <= 2:
                continue
            history_count = valid_count - 1
            top_count = min(history_count, max(1, math.ceil(self.top_fraction * history_count)))
            row_scores = scores[0, :, query_index, :history_count].detach().float()
            attention_weights = F.softmax(scores[0, :, query_index, :valid_count].detach().float(), dim=-1)[
                :, :history_count
            ]
            top_indices = torch.topk(row_scores, k=top_count, dim=-1, largest=True).indices.detach()

            for head in self.selected_heads:
                q_history = self._q_history_for(layer, head, query_states, query_index)
                self._observe_head(
                    layer=layer,
                    head=head,
                    q_history=q_history,
                    query_vector=query_states[0, head, query_index, :].detach(),
                    key_matrix=key_states[0, head, :history_count, :].detach(),
                    score_row=row_scores[head],
                    attention_row=attention_weights[head],
                    top_tokens=top_indices[head],
                    top_count=top_count,
                )
        self._append_q_chunk(layer, query_states)

    def _q_history_for(self, layer: int, head: int, query_states: torch.Tensor, query_index: int) -> torch.Tensor:
        pieces = list(self.q_history.get((layer, head), []))
        if query_index > 0:
            pieces.append(query_states[0, head, :query_index, :].detach().cpu())
        if not pieces:
            return torch.empty((0, query_states.shape[-1]), dtype=torch.float32)
        return torch.cat(pieces, dim=0)

    def _append_q_chunk(self, layer: int, query_states: torch.Tensor) -> None:
        for head in self.selected_heads:
            self.q_history[(layer, head)].append(query_states[0, head, :, :].detach().cpu())

    def _observe_head(
        self,
        *,
        layer: int,
        head: int,
        q_history: torch.Tensor,
        query_vector: torch.Tensor,
        key_matrix: torch.Tensor,
        score_row: torch.Tensor,
        attention_row: torch.Tensor,
        top_tokens: torch.Tensor,
        top_count: int,
    ) -> None:
        if q_history.shape[0] <= 2 or key_matrix.shape[0] <= 2:
            return
        q_working = q_history.to(device=self.svd_device, dtype=self.svd_dtype)
        k_working = key_matrix.to(device=self.svd_device, dtype=self.svd_dtype)
        q_mean = q_working.mean(dim=0, keepdim=True) if self.center_q else torch.zeros_like(q_working[:1])
        k_mean = k_working.mean(dim=0, keepdim=True) if self.center_k else torch.zeros_like(k_working[:1])
        try:
            _, q_singular_values, q_vh = torch.linalg.svd(q_working - q_mean, full_matrices=False)
        except RuntimeError:
            self.skipped_q_svd_rows += 1
            return
        try:
            _, k_singular_values, k_vh = torch.linalg.svd(k_working - k_mean, full_matrices=False)
        except RuntimeError:
            self.skipped_k_svd_rows += 1
            return
        self.observed_rows += 1

        q_basis = q_vh.transpose(0, 1)
        k_basis = k_vh.transpose(0, 1)
        rank_limit = min(int(q_basis.shape[1]), int(k_basis.shape[1]), max(self.ranks))
        ranks = sorted({rank for rank in self.ranks if rank <= rank_limit})
        if rank_limit not in ranks:
            ranks.append(rank_limit)

        self._add_svd("q", layer, head, q_singular_values, ranks, {"q_history_tokens": float(q_history.shape[0])})
        self._add_svd("k", layer, head, k_singular_values, ranks, {"k_history_tokens": float(key_matrix.shape[0])})

        q_raw = query_vector.to(device=self.svd_device, dtype=self.svd_dtype).view(1, -1)
        k_raw = key_matrix.to(device=self.svd_device, dtype=self.svd_dtype)
        q_centered_coeff = ((q_raw - q_mean) @ q_basis).flatten()
        k_centered_coeff = (k_raw - k_mean) @ k_basis
        q_raw_coeff = (q_raw @ q_basis).flatten()
        k_raw_coeff = k_raw @ k_basis

        self._add_recall(
            layer=layer,
            head=head,
            q_basis=q_basis,
            k_basis=k_basis,
            q_coeff=q_raw_coeff,
            k_coeff=k_raw_coeff,
            top_tokens=top_tokens,
            top_count=top_count,
            attention_row=attention_row,
            ranks=ranks,
        )
        self._add_groups(
            layer=layer,
            head=head,
            q_coeff=q_centered_coeff,
            k_coeff=k_centered_coeff,
            score_row=score_row,
            attention_row=attention_row,
            top_tokens=top_tokens,
            ranks=ranks,
        )

    def _add_svd(
        self,
        which: str,
        layer: int,
        head: int,
        singular_values: torch.Tensor,
        ranks: list[int],
        extra_values: dict[str, float],
    ) -> None:
        energy = singular_values.square()
        total_energy = float(energy.sum().item())
        energy_cdf = torch.cumsum(energy, dim=0) / total_energy if total_energy > 0.0 else torch.zeros_like(energy)
        values = {
            "effective_rank": spectral_entropy_effective_rank(singular_values),
            "rank50": float(rank_at_energy(singular_values, 0.50)),
            "rank80": float(rank_at_energy(singular_values, 0.80)),
            "rank90": float(rank_at_energy(singular_values, 0.90)),
            "rank95": float(rank_at_energy(singular_values, 0.95)),
            **extra_values,
        }
        for rank in ranks:
            values[f"{which}_svd_energy_top{rank}"] = float(energy_cdf[rank - 1].item())
        target = self.q_svd_accumulators if which == "q" else self.k_svd_accumulators
        for key in [("overall", "all", "all"), ("layer", layer, "all"), ("layer_head", layer, head)]:
            target[key].add(values)

    def _add_recall(
        self,
        *,
        layer: int,
        head: int,
        q_basis: torch.Tensor,
        k_basis: torch.Tensor,
        q_coeff: torch.Tensor,
        k_coeff: torch.Tensor,
        top_tokens: torch.Tensor,
        top_count: int,
        attention_row: torch.Tensor,
        ranks: list[int],
    ) -> None:
        true_tokens = top_tokens.detach().to(device=self.svd_device, dtype=torch.long)
        true_mask = torch.zeros(k_coeff.shape[0], device=self.svd_device, dtype=torch.bool)
        true_mask[true_tokens] = True
        attn = attention_row.to(device=self.svd_device, dtype=self.svd_dtype)
        true_mass = float(attn[true_mask].sum().item())
        for rank in ranks:
            q_part = q_coeff[:rank]
            k_part = k_coeff[:, :rank]
            cross = q_basis[:, :rank].transpose(0, 1) @ k_basis[:, :rank]
            approx_scores = (q_part.view(1, -1) @ cross @ k_part.transpose(0, 1)).flatten()
            approx_top = torch.topk(approx_scores.float(), k=top_count, largest=True).indices
            recovered = true_mask[approx_top]
            recall = float(recovered.float().mean().item()) if top_count > 0 else 0.0
            recovered_mass = float(attn[approx_top[recovered]].sum().item()) if bool(recovered.any().item()) else 0.0
            values = {
                "top2_recall": recall,
                "top2_attention_mass_recall": recovered_mass / true_mass if true_mass > 0.0 else 0.0,
            }
            for key in [
                ("overall", "all", "all", rank),
                ("layer", layer, "all", rank),
                ("layer_head", layer, head, rank),
            ]:
                self.recall_accumulators[key].add(values)

    def _add_groups(
        self,
        *,
        layer: int,
        head: int,
        q_coeff: torch.Tensor,
        k_coeff: torch.Tensor,
        score_row: torch.Tensor,
        attention_row: torch.Tensor,
        top_tokens: torch.Tensor,
        ranks: list[int],
    ) -> None:
        q_energy = torch.cumsum(q_coeff.square(), dim=0)
        q_total = float(q_energy[-1].item()) if q_energy.numel() else 0.0
        group_tokens = self._tokens_for_row(int(k_coeff.shape[0]), top_tokens.detach().cpu().tolist())
        for group_name, token_indices in group_tokens.items():
            for token_index in token_indices:
                k_token_coeff = k_coeff[token_index]
                k_energy = torch.cumsum(k_token_coeff.square(), dim=0)
                k_total = float(k_energy[-1].item()) if k_energy.numel() else 0.0
                values = {
                    "full_attention_score": float(score_row[token_index].item()),
                    "attention_mass": float(attention_row[token_index].item()),
                }
                for rank in ranks:
                    values[f"q_own_energy_top{rank}"] = (
                        float(q_energy[rank - 1].item()) / q_total if q_total > 0.0 else 0.0
                    )
                    values[f"k_own_energy_top{rank}"] = (
                        float(k_energy[rank - 1].item()) / k_total if k_total > 0.0 else 0.0
                    )
                for key in [
                    ("overall", group_name, "all"),
                    ("layer", group_name, layer),
                    ("layer_head", group_name, (layer, head)),
                ]:
                    self.group_accumulators[key].add(values)

    def _tokens_for_row(self, history_count: int, top_tokens: list[int]) -> dict[str, list[int]]:
        groups: dict[str, set[int]] = {name: set() for name in self.base_groups}
        for name, tokens in self.base_groups.items():
            groups[name].update(token for token in tokens if 0 <= token < history_count)
        groups["top2_selected"] = {int(token) for token in top_tokens if 0 <= int(token) < history_count}
        capped: dict[str, list[int]] = {}
        cap = max(1, self.max_tokens_per_group_per_row)
        for name, tokens in groups.items():
            ordered = sorted(tokens)
            if name != "top2_selected":
                ordered = ordered[:cap]
            if ordered:
                capped[name] = ordered
        return capped

    def svd_rows(self, which: str, metric_fields: list[str]) -> list[dict[str, Any]]:
        source = self.q_svd_accumulators if which == "q" else self.k_svd_accumulators
        rows = []
        for (scope, layer, head), acc in sorted(source.items(), key=lambda item: str(item[0])):
            rows.append(
                acc.row(
                    {
                        "scope": scope,
                        "layer": "" if layer == "all" else layer,
                        "head": "" if head == "all" else head,
                    },
                    metric_fields,
                )
            )
        return rows

    def recall_rows(self, metric_fields: list[str]) -> list[dict[str, Any]]:
        rows = []
        for (scope, layer, head, rank), acc in sorted(
            self.recall_accumulators.items(), key=lambda item: str(item[0])
        ):
            rows.append(
                acc.row(
                    {
                        "scope": scope,
                        "layer": "" if layer == "all" else layer,
                        "head": "" if head == "all" else head,
                        "rank": rank,
                    },
                    metric_fields,
                )
            )
        return rows

    def group_rows(self, metric_fields: list[str]) -> list[dict[str, Any]]:
        rows = []
        for (scope, group, index), acc in sorted(self.group_accumulators.items(), key=lambda item: str(item[0])):
            row: dict[str, Any] = {"scope": scope, "group": group, "layer": "", "head": ""}
            if scope == "layer":
                row["layer"] = index
            elif scope == "layer_head":
                layer, head = index  # type: ignore[misc]
                row["layer"] = layer
                row["head"] = head
            rows.append(acc.row(row, metric_fields))
        return rows


def _separate_qk_eager_attention_forward(
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
        _ACTIVE_COLLECTOR.observe_chunk(
            layer=layer,
            chunk_query_start=key_count - query_count,
            query_states=query_states,
            key_states=key_states,
            scores=scores,
        )

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
        setattr(modeling_qwen3, "eager_attention_forward", _separate_qk_eager_attention_forward)
        if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
            modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _separate_qk_eager_attention_forward


@contextmanager
def active_collector(collector: SeparateQKCollector):
    global _ACTIVE_COLLECTOR
    previous = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    try:
        yield
    finally:
        _ACTIVE_COLLECTOR = previous


@torch.inference_mode()
def run_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    start: int,
    end: int,
    chunk_size: int,
    input_device: torch.device,
    past_key_values: Any | None = None,
) -> Any:
    total_chunks = math.ceil((end - start) / chunk_size)
    for chunk_idx, pos in enumerate(range(start, end, chunk_size), start=1):
        chunk_end = min(pos + chunk_size, end)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, pos:chunk_end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(pos, chunk_end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"tokens chunk {chunk_idx}/{total_chunks}: tokens {pos}-{chunk_end - 1}", flush=True)
    return past_key_values


def build_base_groups(
    *,
    prefill_tokens: int,
    spans: dict[str, tuple[int, int]],
    sink_tokens: int,
    recent_tokens: int,
) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {
        "sink": set(range(0, min(sink_tokens, prefill_tokens))),
        "recent": set(range(max(0, prefill_tokens - recent_tokens), prefill_tokens)),
    }
    for name, (start, end) in spans.items():
        groups[f"evidence_{name}"] = set(range(max(0, start), min(prefill_tokens, end)))
    evidence_all = set()
    for name, tokens in groups.items():
        if name.startswith("evidence_"):
            evidence_all.update(tokens)
    if evidence_all:
        groups["evidence_any"] = evidence_all
    return {name: tokens for name, tokens in groups.items() if tokens}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    install_qwen3_attention_patch()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.heads, head_count, "heads")
    ranks = parse_ranks(args.ranks, head_dim)
    svd_device = torch.device(args.svd_device if torch.cuda.is_available() else "cpu")
    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32

    collector = SeparateQKCollector(
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        ranks=ranks,
        top_fraction=args.top_fraction,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        max_tokens_per_group_per_row=args.max_tokens_per_group_per_row,
        center_q=args.center_q,
        center_k=args.center_k,
        svd_device=svd_device,
        svd_dtype=svd_dtype,
    )

    task_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            query_ids = tokenizer(task["query"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            if query_ids.shape[-1] <= 0 or context_ids.shape[-1] <= 2:
                continue
            input_ids = torch.cat([context_ids, query_ids], dim=-1)
            prefill_tokens = int(context_ids.shape[-1])
            eval_tokens = int(query_ids.shape[-1])
            spans = evidence_spans(tokenizer, task)
            base_groups = build_base_groups(
                prefill_tokens=prefill_tokens,
                spans=spans,
                sink_tokens=args.sink_tokens,
                recent_tokens=args.recent_tokens,
            )
            query_tokens = set(choose_query_tokens(prefill_tokens, eval_tokens, args.max_query_tokens_per_task))
            collector.set_task(query_tokens=query_tokens, base_groups=base_groups)
            task_rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "prefill_tokens": prefill_tokens,
                    "query_tokens": eval_tokens,
                    "sampled_query_tokens": " ".join(str(token) for token in sorted(query_tokens)),
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                    **{f"{name}_span": f"{span[0]}:{span[1]}" for name, span in spans.items()},
                }
            )
            with active_collector(collector):
                past = run_tokens(model, input_ids, 0, prefill_tokens, args.chunk_size, input_device)
                run_tokens(
                    model,
                    input_ids,
                    prefill_tokens,
                    prefill_tokens + eval_tokens,
                    args.chunk_size,
                    input_device,
                    past_key_values=past,
                )
            del past
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    seconds = time.perf_counter() - started
    q_svd_fields = ["effective_rank", "rank50", "rank80", "rank90", "rank95", "q_history_tokens"] + [
        f"q_svd_energy_top{rank}" for rank in ranks
    ]
    k_svd_fields = ["effective_rank", "rank50", "rank80", "rank90", "rank95", "k_history_tokens"] + [
        f"k_svd_energy_top{rank}" for rank in ranks
    ]
    recall_fields = ["top2_recall", "top2_attention_mass_recall"]
    group_fields = ["full_attention_score", "attention_mass"]
    for rank in ranks:
        group_fields.extend([f"q_own_energy_top{rank}", f"k_own_energy_top{rank}"])

    write_csv(output_dir / "q_svd_energy_stats.csv", collector.svd_rows("q", q_svd_fields), ["scope", "layer", "head", "cases"] + q_svd_fields)
    write_csv(output_dir / "k_svd_energy_stats.csv", collector.svd_rows("k", k_svd_fields), ["scope", "layer", "head", "cases"] + k_svd_fields)
    write_csv(output_dir / "separate_qk_top2_recall_stats.csv", collector.recall_rows(recall_fields), ["scope", "layer", "head", "rank", "cases"] + recall_fields)
    write_csv(output_dir / "group_own_subspace_stats.csv", collector.group_rows(group_fields), ["scope", "group", "layer", "head", "cases"] + group_fields)
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        ["variant", "task_id", "prefill_tokens", "query_tokens", "sampled_query_tokens", "target_key", "target_label", "key_span", "label_span", "record_span"],
    )

    summary = {
        "args": vars(args),
        "resolved": {
            "layer_count": layer_count,
            "head_count": head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "ranks": ranks,
            "observed_svd_rows": collector.observed_rows,
            "skipped_q_svd_rows": collector.skipped_q_svd_rows,
            "skipped_k_svd_rows": collector.skipped_k_svd_rows,
            "tasks": len(task_rows),
            "seconds": seconds,
            "metric_notes": {
                "separate_qk_score": "q is projected to historical Q-SVD basis, k is projected to historical K-SVD basis, and scores use q_q^T (Bq^T Bk) k_k.",
                "rank_full_sanity": "At full head_dim rank, separate_qk_score equals full q-k dot up to numerical error.",
            },
        },
        "paths": {
            "q_svd_energy_stats": str(output_dir / "q_svd_energy_stats.csv"),
            "k_svd_energy_stats": str(output_dir / "k_svd_energy_stats.csv"),
            "separate_qk_top2_recall_stats": str(output_dir / "separate_qk_top2_recall_stats.csv"),
            "group_own_subspace_stats": str(output_dir / "group_own_subspace_stats.csv"),
            "tasks": str(output_dir / "tasks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "seconds": seconds, "observed_svd_rows": collector.observed_rows}, indent=2))


if __name__ == "__main__":
    main()
