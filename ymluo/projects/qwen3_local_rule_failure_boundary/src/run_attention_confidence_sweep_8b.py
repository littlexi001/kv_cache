from __future__ import annotations

import argparse
import base64
import gc
import gzip
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base


ROLE_ORDER = (
    "start_key",
    "hop1_result",
    "hop2_input",
    "hop2_result",
    "rule1_line",
    "rule2_line",
)


def parse_lengths(value: str) -> list[int]:
    output: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            output.append(int(part))
            continue
        fields = [int(item) for item in part.split(":")]
        if len(fields) != 3:
            raise ValueError(f"length range must be start:stop:step, got {part!r}")
        start, stop, step = fields
        if step <= 0:
            raise ValueError("length step must be positive")
        output.extend(range(start, stop + 1, step))
    return sorted(set(output))


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{value:.{digits}g}")


def tensor_f16_base64(values: torch.Tensor) -> str:
    raw = values.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def tensor_f32_base64(values: torch.Tensor) -> str:
    raw = values.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def tensor_u32_base64(values: Sequence[int]) -> str:
    raw = torch.tensor(values, dtype=torch.int32).contiguous().numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def write_gzip_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def write_full_pre_softmax_scope(
    path: Path,
    *,
    scope: str,
    key_length: int,
    logits: torch.Tensor,
    top_positions: Sequence[int],
    layer: int | None = None,
    head: int | None = None,
    probabilities: torch.Tensor | None = None,
    logsumexp: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": scope,
        "layer": layer,
        "head": head,
        "key_length": key_length,
        "storage_dtype": "float16_le_base64",
        "logits_f16_b64": tensor_f16_base64(logits),
        "top_logit_positions": [int(position) for position in top_positions],
        "min_logit": rounded(float(logits.min().item())),
        "max_logit": rounded(float(logits.max().item())),
    }
    if probabilities is not None:
        payload["probabilities_f16_b64"] = tensor_f16_base64(probabilities)
    if logsumexp is not None:
        payload["logsumexp"] = rounded(logsumexp)
    write_gzip_json_atomic(path, payload)


def span_sum(probabilities: torch.Tensor, spans: Iterable[tuple[int, int]]) -> float:
    total = torch.zeros((), dtype=torch.float32, device=probabilities.device)
    for start, end in spans:
        if end > start:
            total = total + probabilities[start:end].sum()
    return float(total.item())


def event_code_span(tokenizer: Any, event: base.RuleEvent, code: str, *, last: bool = False) -> tuple[int, int]:
    char_start = event.text.rfind(code) if last else event.text.find(code)
    if char_start < 0:
        raise ValueError(f"code {code!r} not found in event {event.label}: {event.text!r}")
    char_end = char_start + len(code)
    encoded = tokenizer(
        event.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("fast-tokenizer offset_mapping is required for exact code spans")
    local_indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > char_start and left < char_end
    ]
    if not local_indices:
        raise RuntimeError(f"no tokens overlap code {code!r} in event {event.label}")
    start = event.start_token + min(local_indices)
    end = event.start_token + max(local_indices) + 1
    return start, end


def role_spans(tokenizer: Any, body: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    relevant = sorted(
        (event for event in body["events"] if event.kind == "relevant"),
        key=lambda event: event.step,
    )
    if len(relevant) not in {1, 2}:
        raise ValueError(f"expected one or two clean rules, got {len(relevant)}")
    first = relevant[0]
    spans = {
        "start_key": [event_code_span(tokenizer, first, first.antecedent)],
        "hop1_result": [event_code_span(tokenizer, first, first.consequent, last=True)],
        "hop2_input": [],
        "hop2_result": [],
        "rule1_line": [(first.start_token, first.end_token)],
        "rule2_line": [],
    }
    if len(relevant) == 2:
        second = relevant[1]
        spans.update(
            {
                "hop2_input": [event_code_span(tokenizer, second, second.antecedent)],
                "hop2_result": [
                    event_code_span(tokenizer, second, second.consequent, last=True)
                ],
                "rule2_line": [(second.start_token, second.end_token)],
            }
        )
    return spans


def token_role(position: int, body_tokens: int, spans: dict[str, list[tuple[int, int]]]) -> str:
    # Exact symbolic fields take precedence over their containing rule lines.
    for role in ("hop1_result", "hop2_input", "hop2_result", "start_key"):
        if any(start <= position < end for start, end in spans[role]):
            return role
    for role in ("rule1_line", "rule2_line"):
        if any(start <= position < end for start, end in spans[role]):
            return role
    return "filler" if position < body_tokens else "query"


def capture_query_states(
    model: Any,
    query_cache: Any,
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
) -> tuple[Any, dict[int, torch.Tensor], float]:
    layers = list(getattr(getattr(model, "model", None), "layers", []))
    if not layers:
        raise RuntimeError("cannot locate model.model.layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None:
                raise RuntimeError(f"layer {layer_index}: no hidden states in attention hook")
            projected = module.q_proj(hidden_states)
            batch, query_length, _ = projected.shape
            head_dim = int(getattr(module, "head_dim"))
            num_heads = int(projected.shape[-1] // head_dim)
            query = projected.view(batch, query_length, num_heads, head_dim)
            query_norm = getattr(module, "q_norm", None)
            if query_norm is not None:
                query = query_norm(query)
            query = query.transpose(1, 2)
            query = base.apply_rope_to_q(query, position_embeddings)
            captured[layer_index] = query[:, :, -1, :].detach()

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True))
    base.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = base.forward_with_cache(
                model,
                last_prompt_id.to(base.input_device(model)),
                query_cache,
                prompt_len_minus_one,
            )
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    if len(captured) != len(layers):
        raise RuntimeError(f"captured {len(captured)} query layers, expected {len(layers)}")
    return output, captured, time.perf_counter() - started


@torch.inference_mode()
def summarize_attention(
    model: Any,
    output: Any,
    captured_queries: dict[int, torch.Tensor],
    spans: dict[str, list[tuple[int, int]]],
    max_top: int,
    full_pre_softmax_dir: Path | None = None,
    token_type_pre_softmax_path: Path | None = None,
    prompt_token_ids: Sequence[int] | None = None,
    token_text: dict[str, str] | None = None,
) -> tuple[dict[str, Any], set[int]]:
    layers = list(model.model.layers)
    cache = base.legacy_cache(output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    top_count = min(max_top, key_length)
    top2pct_count = min(key_length, max(1, int(math.ceil(0.02 * key_length))))
    # Keep the model-wide accumulator on CPU so the same collector also works
    # when long cases shard the 36 layers across two 24 GB GPUs.
    overall = torch.zeros(key_length, dtype=torch.float32)
    overall_logits = (
        torch.zeros(key_length, dtype=torch.float32)
        if full_pre_softmax_dir is not None
        else None
    )
    token_type_unique_ids: torch.Tensor | None = None
    token_type_inverse: torch.Tensor | None = None
    token_type_counts: torch.Tensor | None = None
    token_type_mean_logits: torch.Tensor | None = None
    token_type_probability_mass: torch.Tensor | None = None
    if token_type_pre_softmax_path is not None:
        if prompt_token_ids is None or token_text is None:
            raise ValueError("token-type export requires prompt_token_ids and token_text")
        token_id_tensor = torch.tensor(prompt_token_ids, dtype=torch.long)
        token_type_unique_ids, token_type_inverse, token_type_counts = torch.unique(
            token_id_tensor,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        num_query_heads = int(captured_queries[0][0].shape[0])
        token_type_mean_logits = torch.empty(
            (len(token_type_unique_ids), len(layers), num_query_heads),
            dtype=torch.float32,
        )
        token_type_probability_mass = torch.empty_like(token_type_mean_logits)
    head_positions: list[list[list[int]]] = []
    head_scores: list[list[list[float]]] = []
    head_entropy: list[list[float]] = []
    head_effective_tokens: list[list[float]] = []
    head_role_mass: list[list[list[float]]] = []
    head_role_logit_mean: list[list[list[float]]] = []
    head_role_logit_max: list[list[list[float]]] = []
    head_role_best_rank: list[list[list[int]]] = []
    head_role_cosine_mean: list[list[list[float]]] = []
    head_role_cosine_max: list[list[list[float]]] = []
    head_role_key_norm_mean: list[list[list[float]]] = []
    head_query_norm: list[list[float]] = []
    head_max_logit: list[list[float]] = []
    head_logsumexp: list[list[float]] = []
    head_top2pct_kept_mass: list[list[float]] = []
    head_top2pct_role_mass: list[list[list[float]]] = []
    head_recent512_mass: list[list[float]] = []
    head_sink16_mass: list[list[float]] = []
    layer_positions: list[list[int]] = []
    layer_scores: list[list[float]] = []
    layer_entropy: list[float] = []
    layer_role_mass: list[list[float]] = []
    needed_positions: set[int] = set()

    for layer_index, layer in enumerate(layers):
        query = captured_queries[layer_index][0]
        key = cache[layer_index][0][0]
        num_heads = int(query.shape[0])
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        scale = float(getattr(layer.self_attn, "scaling", query.shape[-1] ** -0.5))
        layer_distribution = torch.zeros(key_length, dtype=torch.float32, device=key.device)
        layer_logits = (
            torch.zeros(key_length, dtype=torch.float32, device=key.device)
            if full_pre_softmax_dir is not None
            else None
        )
        layer_head_positions: list[list[int]] = []
        layer_head_scores: list[list[float]] = []
        layer_head_entropy: list[float] = []
        layer_head_effective: list[float] = []
        layer_head_roles: list[list[float]] = []
        layer_head_role_logit_mean: list[list[float]] = []
        layer_head_role_logit_max: list[list[float]] = []
        layer_head_role_best_rank: list[list[int]] = []
        layer_head_role_cosine_mean: list[list[float]] = []
        layer_head_role_cosine_max: list[list[float]] = []
        layer_head_role_key_norm_mean: list[list[float]] = []
        layer_head_query_norm: list[float] = []
        layer_head_max_logit: list[float] = []
        layer_head_logsumexp: list[float] = []
        layer_head_top2pct_kept_mass: list[float] = []
        layer_head_top2pct_role_mass: list[list[float]] = []
        layer_head_recent: list[float] = []
        layer_head_sink: list[float] = []
        role_indices = {
            role: torch.tensor(
                [
                    position
                    for start, end in spans[role]
                    for position in range(start, end)
                ],
                dtype=torch.long,
                device=key.device,
            )
            for role in ROLE_ORDER
        }
        type_inverse_device = (
            token_type_inverse.to(key.device)
            if token_type_inverse is not None
            else None
        )
        type_counts_device = (
            token_type_counts.to(device=key.device, dtype=torch.float32)
            if token_type_counts is not None
            else None
        )

        for head_index in range(num_heads):
            kv_index = min(kv_heads - 1, head_index // groups)
            logits = torch.matmul(key[kv_index].float(), query[head_index].float()) * scale
            probabilities = torch.softmax(logits.float(), dim=-1)
            if (
                type_inverse_device is not None
                and type_counts_device is not None
                and token_type_mean_logits is not None
                and token_type_probability_mass is not None
            ):
                type_logit_sums = torch.bincount(
                    type_inverse_device,
                    weights=logits,
                    minlength=len(type_counts_device),
                )
                type_probability_sums = torch.bincount(
                    type_inverse_device,
                    weights=probabilities,
                    minlength=len(type_counts_device),
                )
                token_type_mean_logits[:, layer_index, head_index].copy_(
                    (type_logit_sums / type_counts_device).cpu()
                )
                token_type_probability_mass[:, layer_index, head_index].copy_(
                    type_probability_sums.cpu()
                )
            values, indices = torch.topk(probabilities, k=top_count, largest=True, sorted=True)
            top2pct_values, top2pct_indices = torch.topk(
                probabilities, k=top2pct_count, largest=True, sorted=False
            )
            top2pct_kept_mass = top2pct_values.sum().clamp_min(1e-30)
            top2pct_mask = torch.zeros(key_length, dtype=torch.bool, device=key.device)
            top2pct_mask.scatter_(0, top2pct_indices, True)
            top2pct_role_masses = []
            for role in ROLE_ORDER:
                indices_for_role = role_indices[role]
                role_probabilities = probabilities.index_select(0, indices_for_role)
                role_is_kept = top2pct_mask.index_select(0, indices_for_role)
                kept_role_mass = role_probabilities.masked_select(role_is_kept).sum()
                top2pct_role_masses.append(
                    rounded(float((kept_role_mass / top2pct_kept_mass).item()))
                )
            positions = [int(value) for value in indices.cpu().tolist()]
            scores = [rounded(float(value)) for value in values.cpu().tolist()]
            needed_positions.update(positions)
            entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()).item())
            masses = [span_sum(probabilities, spans[role]) for role in ROLE_ORDER]
            query_vector = query[head_index].float()
            query_norm = torch.linalg.vector_norm(query_vector).clamp_min(1e-30)
            role_logit_means: list[float] = []
            role_logit_maxima: list[float] = []
            role_best_ranks: list[int] = []
            role_cosine_means: list[float] = []
            role_cosine_maxima: list[float] = []
            role_key_norm_means: list[float] = []
            for role in ROLE_ORDER:
                indices_for_role = role_indices[role]
                if indices_for_role.numel() == 0:
                    role_logit_means.append(0.0)
                    role_logit_maxima.append(0.0)
                    role_best_ranks.append(key_length + 1)
                    role_cosine_means.append(0.0)
                    role_cosine_maxima.append(0.0)
                    role_key_norm_means.append(0.0)
                    continue
                role_logits = logits.index_select(0, indices_for_role)
                role_keys = key[kv_index].float().index_select(0, indices_for_role)
                role_key_norms = torch.linalg.vector_norm(role_keys, dim=-1).clamp_min(1e-30)
                role_cosines = torch.matmul(role_keys, query_vector) / (role_key_norms * query_norm)
                role_maximum = role_logits.max()
                role_logit_means.append(rounded(float(role_logits.mean().item())))
                role_logit_maxima.append(rounded(float(role_maximum.item())))
                role_best_ranks.append(int((logits > role_maximum).sum().item()) + 1)
                role_cosine_means.append(rounded(float(role_cosines.mean().item())))
                role_cosine_maxima.append(rounded(float(role_cosines.max().item())))
                role_key_norm_means.append(rounded(float(role_key_norms.mean().item())))
            layer_distribution.add_(probabilities, alpha=1.0 / num_heads)
            if layer_logits is not None:
                layer_logits.add_(logits, alpha=1.0 / num_heads)
                write_full_pre_softmax_scope(
                    full_pre_softmax_dir
                    / "heads"
                    / f"layer_{layer_index:02d}_head_{head_index:02d}.json.gz",
                    scope="head",
                    layer=layer_index,
                    head=head_index,
                    key_length=key_length,
                    logits=logits,
                    top_positions=positions,
                    logsumexp=float(torch.logsumexp(logits, dim=-1).item()),
                )
            layer_head_positions.append(positions)
            layer_head_scores.append(scores)
            layer_head_entropy.append(rounded(entropy))
            layer_head_effective.append(rounded(math.exp(entropy)))
            layer_head_roles.append([rounded(value) for value in masses])
            layer_head_role_logit_mean.append(role_logit_means)
            layer_head_role_logit_max.append(role_logit_maxima)
            layer_head_role_best_rank.append(role_best_ranks)
            layer_head_role_cosine_mean.append(role_cosine_means)
            layer_head_role_cosine_max.append(role_cosine_maxima)
            layer_head_role_key_norm_mean.append(role_key_norm_means)
            layer_head_query_norm.append(rounded(float(query_norm.item())))
            layer_head_max_logit.append(rounded(float(logits.max().item())))
            layer_head_logsumexp.append(rounded(float(torch.logsumexp(logits, dim=-1).item())))
            layer_head_top2pct_kept_mass.append(rounded(float(top2pct_kept_mass.item())))
            layer_head_top2pct_role_mass.append(top2pct_role_masses)
            layer_head_recent.append(rounded(float(probabilities[max(0, key_length - 512) :].sum().item())))
            layer_head_sink.append(rounded(float(probabilities[: min(16, key_length)].sum().item())))

        layer_values, layer_indices = torch.topk(
            layer_distribution, k=top_count, largest=True, sorted=True
        )
        layer_top_positions = [int(value) for value in layer_indices.cpu().tolist()]
        needed_positions.update(layer_top_positions)
        layer_positions.append(layer_top_positions)
        layer_scores.append([rounded(float(value)) for value in layer_values.cpu().tolist()])
        entropy = float(
            (-(layer_distribution * torch.log(layer_distribution.clamp_min(1e-30))).sum()).item()
        )
        layer_entropy.append(rounded(entropy))
        layer_role_mass.append(
            [rounded(span_sum(layer_distribution, spans[role])) for role in ROLE_ORDER]
        )
        overall.add_(layer_distribution.cpu(), alpha=1.0 / len(layers))
        if layer_logits is not None and overall_logits is not None:
            layer_logit_indices = torch.topk(
                layer_logits, k=top_count, largest=True, sorted=True
            ).indices
            write_full_pre_softmax_scope(
                full_pre_softmax_dir / "layers" / f"layer_{layer_index:02d}.json.gz",
                scope="layer",
                layer=layer_index,
                key_length=key_length,
                logits=layer_logits,
                probabilities=layer_distribution,
                top_positions=[int(value) for value in layer_logit_indices.cpu().tolist()],
            )
            overall_logits.add_(layer_logits.cpu(), alpha=1.0 / len(layers))
        head_positions.append(layer_head_positions)
        head_scores.append(layer_head_scores)
        head_entropy.append(layer_head_entropy)
        head_effective_tokens.append(layer_head_effective)
        head_role_mass.append(layer_head_roles)
        head_role_logit_mean.append(layer_head_role_logit_mean)
        head_role_logit_max.append(layer_head_role_logit_max)
        head_role_best_rank.append(layer_head_role_best_rank)
        head_role_cosine_mean.append(layer_head_role_cosine_mean)
        head_role_cosine_max.append(layer_head_role_cosine_max)
        head_role_key_norm_mean.append(layer_head_role_key_norm_mean)
        head_query_norm.append(layer_head_query_norm)
        head_max_logit.append(layer_head_max_logit)
        head_logsumexp.append(layer_head_logsumexp)
        head_top2pct_kept_mass.append(layer_head_top2pct_kept_mass)
        head_top2pct_role_mass.append(layer_head_top2pct_role_mass)
        head_recent512_mass.append(layer_head_recent)
        head_sink16_mass.append(layer_head_sink)

    overall_values, overall_indices = torch.topk(overall, k=top_count, largest=True, sorted=True)
    overall_positions = [int(value) for value in overall_indices.cpu().tolist()]
    needed_positions.update(overall_positions)
    overall_entropy = float((-(overall * torch.log(overall.clamp_min(1e-30))).sum()).item())
    if overall_logits is not None:
        overall_logit_indices = torch.topk(
            overall_logits, k=top_count, largest=True, sorted=True
        ).indices
        write_full_pre_softmax_scope(
            full_pre_softmax_dir / "overall.json.gz",
            scope="overall",
            key_length=key_length,
            logits=overall_logits,
            probabilities=overall,
            top_positions=[int(value) for value in overall_logit_indices.tolist()],
        )
    if (
        token_type_pre_softmax_path is not None
        and token_type_unique_ids is not None
        and token_type_counts is not None
        and token_type_mean_logits is not None
        and token_type_probability_mass is not None
        and token_text is not None
    ):
        unique_ids = [int(value) for value in token_type_unique_ids.tolist()]
        write_gzip_json_atomic(
            token_type_pre_softmax_path,
            {
                "schema_version": 1,
                "key_length": key_length,
                "num_layers": len(layers),
                "num_attention_heads": int(token_type_mean_logits.shape[2]),
                "shape": list(token_type_mean_logits.shape),
                "aggregation": {
                    "raw_logit": "mean QK/sqrt(d) over every prompt position with the token id, per query head",
                    "share": "sum of exact softmax probability over every prompt position with the token id, per query head",
                },
                "token_ids_u32_b64": tensor_u32_base64(unique_ids),
                "token_counts_u32_b64": tensor_u32_base64(
                    [int(value) for value in token_type_counts.tolist()]
                ),
                "token_text": {
                    str(token_id): token_text[str(token_id)]
                    for token_id in unique_ids
                },
                "mean_logits_f16_b64": tensor_f16_base64(token_type_mean_logits),
                "probability_mass_f32_b64": tensor_f32_base64(token_type_probability_mass),
            },
        )
    attention = {
        "max_top": top_count,
        "key_length": key_length,
        "top2pct_count": top2pct_count,
        "role_order": list(ROLE_ORDER),
        "head_positions": head_positions,
        "head_scores": head_scores,
        "head_entropy": head_entropy,
        "head_effective_tokens": head_effective_tokens,
        "head_role_mass": head_role_mass,
        "head_role_logit_mean": head_role_logit_mean,
        "head_role_logit_max": head_role_logit_max,
        "head_role_best_rank": head_role_best_rank,
        "head_role_cosine_mean": head_role_cosine_mean,
        "head_role_cosine_max": head_role_cosine_max,
        "head_role_key_norm_mean": head_role_key_norm_mean,
        "head_query_norm": head_query_norm,
        "head_max_logit": head_max_logit,
        "head_logsumexp": head_logsumexp,
        "head_top2pct_kept_mass": head_top2pct_kept_mass,
        "head_top2pct_role_mass": head_top2pct_role_mass,
        "overall_mean_head_top2pct_kept_mass": rounded(
            statistics.fmean(value for layer_values in head_top2pct_kept_mass for value in layer_values)
        ),
        "overall_mean_head_top2pct_role_mass": [
            rounded(
                statistics.fmean(
                    head_top2pct_role_mass[layer_index][head_index][role_index]
                    for layer_index in range(len(head_top2pct_role_mass))
                    for head_index in range(len(head_top2pct_role_mass[layer_index]))
                )
            )
            for role_index in range(len(ROLE_ORDER))
        ],
        "head_recent512_mass": head_recent512_mass,
        "head_sink16_mass": head_sink16_mass,
        "layer_positions": layer_positions,
        "layer_scores": layer_scores,
        "layer_entropy": layer_entropy,
        "layer_role_mass": layer_role_mass,
        "overall_positions": overall_positions,
        "overall_scores": [rounded(float(value)) for value in overall_values.cpu().tolist()],
        "overall_entropy": rounded(overall_entropy),
        "overall_effective_tokens": rounded(math.exp(overall_entropy)),
        "overall_role_mass": [rounded(span_sum(overall, spans[role])) for role in ROLE_ORDER],
        "overall_recent512_mass": rounded(float(overall[max(0, key_length - 512) :].sum().item())),
        "overall_sink16_mass": rounded(float(overall[: min(16, key_length)].sum().item())),
    }
    return attention, needed_positions


@torch.inference_mode()
def score_gold_from_query_output(
    model: Any,
    tokenizer: Any,
    query_output: Any,
    prompt_length: int,
    gold_answer: str,
    completion_text: str | None = None,
    candidate_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    device = base.input_device(model)
    scored_text = gold_answer if completion_text is None else completion_text
    answer_ids = tokenizer(
        scored_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    logits = query_output.logits[:, -1, :]
    cache = query_output.past_key_values
    past_length = prompt_length
    token_rows: list[dict[str, Any]] = []
    total_nll = 0.0
    for token_index in range(int(answer_ids.shape[1])):
        target = answer_ids[:, token_index]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        log_probability = float(log_probs.gather(1, target.view(1, 1)).item())
        probability = math.exp(log_probability)
        total_nll -= log_probability
        token_rows.append(
            {
                "index": token_index,
                "token_id": int(target.item()),
                "token": tokenizer.decode(
                    [int(target.item())],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "probability": rounded(probability),
                "nll": rounded(-log_probability),
            }
        )
        if token_index + 1 < int(answer_ids.shape[1]):
            advanced = base.forward_with_cache(model, target.view(1, 1), cache, past_length)
            logits = advanced.logits[:, -1, :]
            cache = advanced.past_key_values
            past_length += 1
    count = max(1, int(answer_ids.shape[1]))
    mean_nll = total_nll / count
    first_distribution = torch.softmax(query_output.logits[:, -1, :].float(), dim=-1)
    first_values, first_indices = torch.topk(first_distribution, k=5, sorted=True)
    result = {
        "gold_answer": gold_answer,
        "gold_scored_text": scored_text,
        "gold_token_count": int(answer_ids.shape[1]),
        "gold_total_nll": rounded(total_nll),
        "gold_mean_nll": rounded(mean_nll),
        "gold_ppl": rounded(math.exp(mean_nll)),
        "gold_token_scores": token_rows,
        "next_token_top5": [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "probability": rounded(float(probability)),
            }
            for probability, token_id in zip(first_values[0].cpu().tolist(), first_indices[0].cpu().tolist())
        ],
    }
    if candidate_texts:
        candidate_rows = []
        gold_candidate_index = None
        gold_token_id = int(answer_ids[0, 0].item())
        for candidate_index, candidate_text in enumerate(candidate_texts):
            candidate_ids = tokenizer(
                candidate_text,
                add_special_tokens=False,
            )["input_ids"]
            if len(candidate_ids) != 1:
                raise RuntimeError(
                    f"candidate must be one token: {candidate_text!r} -> {candidate_ids}"
                )
            token_id = int(candidate_ids[0])
            probability = float(first_distribution[0, token_id].item())
            candidate_rows.append(
                {
                    "text": candidate_text,
                    "token_id": token_id,
                    "token": tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "probability": rounded(probability),
                    "log_probability": rounded(math.log(max(probability, 1e-300))),
                }
            )
            if token_id == gold_token_id:
                gold_candidate_index = candidate_index
        if gold_candidate_index is None:
            raise RuntimeError("gold answer is absent from candidate_texts")
        prediction_index = max(
            range(len(candidate_rows)),
            key=lambda index: float(candidate_rows[index]["probability"]),
        )
        gold_log_probability = float(
            candidate_rows[gold_candidate_index]["log_probability"]
        )
        strongest_wrong_log_probability = max(
            float(row["log_probability"])
            for index, row in enumerate(candidate_rows)
            if index != gold_candidate_index
        )
        result.update(
            {
                "candidate_token_scores": candidate_rows,
                "candidate_prediction": candidate_rows[prediction_index]["text"].strip(),
                "candidate_correct": prediction_index == gold_candidate_index,
                "candidate_margin": rounded(
                    gold_log_probability - strongest_wrong_log_probability
                ),
            }
        )
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def build_case(
    tokenizer: Any,
    length: int,
    seed: int,
    code_mode: str,
    placement: str = "middle",
    evidence_hops: int = 2,
) -> dict[str, Any]:
    if evidence_hops == 1:
        bundle = causal.build_bundle(seed, tokenizer=tokenizer, code_mode=code_mode)
        gold_events = bundle["gold_events"][:1]
        placed: list[base.RuleEvent] = []
        if length == 0:
            body, placed = causal.encode_event_block(tokenizer, gold_events, 0)
        else:
            body = base.build_filler_ids(tokenizer, length, 1_700_000 + seed)
            occupied: list[tuple[int, int]] = []
            gold_ids, _ = causal.encode_event_block(tokenizer, gold_events, 0)
            if placement == "prefix":
                preferred = min(256, max(0, length - len(gold_ids)))
            elif placement == "recent":
                preferred = max(0, length - len(gold_ids) - 256)
            else:
                preferred = length // 2 - len(gold_ids) // 2
            causal.insert_event_block(
                body,
                tokenizer,
                gold_events,
                preferred,
                occupied,
                placed,
            )
        return {
            "seed": seed,
            "condition": "clean",
            "placement": placement,
            "target_context_tokens": length,
            "body_ids": torch.tensor(body, dtype=torch.long).view(1, -1),
            "body_tokens": len(body),
            "events": placed,
            **bundle,
        }
    if evidence_hops != 2:
        raise ValueError(f"evidence_hops must be one or two, got {evidence_hops}")
    return causal.build_body(
        tokenizer,
        seed=seed,
        target_context_tokens=length,
        condition="clean",
        placement=placement,
        code_mode=code_mode,
    )


def build_token_table(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    positions: set[int],
    body_tokens: int,
    spans: dict[str, list[tuple[int, int]]],
) -> list[list[Any]]:
    for role_spans_ in spans.values():
        for start, end in role_spans_:
            positions.update(range(start, end))
    rows: list[list[Any]] = []
    for position in sorted(positions):
        token_id = int(prompt_ids[position])
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rows.append([position, token_id, text, token_role(position, body_tokens, spans)])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-case Qwen3-8B confidence/attention sweep for a clean two-hop chain."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="0:64000:500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--code_mode",
        choices=["legacy", "single_token", "english_single_token"],
        default="legacy",
    )
    parser.add_argument("--placement", choices=list(causal.PLACEMENTS), default="middle")
    parser.add_argument(
        "--query_mode",
        choices=["full2", "hop1", "oracle_hop2"],
        default="full2",
    )
    parser.add_argument(
        "--evidence_hops",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of clean VERIFIED RULE transitions placed in the context.",
    )
    parser.add_argument("--prompt_style", choices=["legacy", "cloze"], default="legacy")
    parser.add_argument("--max_top", type=int, default=100)
    parser.add_argument("--prefill_chunk_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--global_max_position", type=int, default=66000)
    parser.add_argument("--shard_label", default="single")
    parser.add_argument(
        "--export_full_pre_softmax_dir",
        default="",
        help=(
            "Optional root for per-position pre-softmax QK logits. One gzip shard is written "
            "per head and layer so the browser can load a selected scope lazily."
        ),
    )
    parser.add_argument(
        "--export_token_type_pre_softmax_dir",
        default="",
        help=(
            "Optional directory for compact per-token-type raw-logit and softmax-mass "
            "matrices. One gzip file is written per context length."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    if not lengths:
        raise ValueError("no lengths selected")
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    preview = build_case(
        tokenizer,
        lengths[0],
        args.seed,
        args.code_mode,
        args.placement,
        args.evidence_hops,
    )
    preview_spans = role_spans(tokenizer, preview)
    if args.code_mode in {"single_token", "english_single_token"}:
        token_counts = [len(base.token_ids(tokenizer, code)) for code in preview["gold_codes"]]
        if token_counts != [1, 1, 1]:
            raise RuntimeError(f"single-token gold code invariant failed: {token_counts}")
        for role in ("start_key", "hop1_result", "hop2_input", "hop2_result"):
            if any(end - start != 1 for start, end in preview_spans[role]):
                raise RuntimeError(f"single-token role span invariant failed for {role}: {preview_spans[role]}")
    shard_config = {
        **vars(args),
        "resolved_lengths": lengths,
        "gold_codes": preview["gold_codes"],
        "role_spans_at_first_length": preview_spans,
    }
    write_json_atomic(output_dir / f"config_{args.shard_label}.json", shard_config)
    if args.dry_run:
        print(json.dumps(shard_config, ensure_ascii=False, indent=2), flush=True)
        return

    rope_factor = base.rope_factor_for_length(
        args.global_max_position,
        args.original_max_position_embeddings,
    )
    model, tokenizer = base.load_model_and_tokenizer(
        args,
        args.global_max_position,
        rope_factor,
    )
    model_config = {
        "num_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(getattr(model.config, "head_dim", 0)),
        "rope_factor": rope_factor,
    }

    for case_index, length in enumerate(lengths, start=1):
        started = time.perf_counter()
        destination = data_dir / f"length_{length}.json"
        if destination.exists():
            print(
                f"[{case_index}/{len(lengths)}] length={length} already complete; skipping",
                flush=True,
            )
            continue
        body = build_case(
            tokenizer,
            length,
            args.seed,
            args.code_mode,
            args.placement,
            args.evidence_hops,
        )
        spans = role_spans(tokenizer, body)
        start_code, steps, gold_answer = causal.query_parameters(body, args.query_mode)
        suffix = causal.build_suffix(args.prompt_style, start_code, steps, args.query_mode)
        suffix_ids = base.token_ids(tokenizer, suffix)
        prompt_ids = body["body_ids"].view(-1).tolist() + suffix_ids
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prompt_len_minus_one = int(prompt.shape[1]) - 1
        base_cache, prefill_seconds = base.prefill_sequence(
            model,
            prompt[:, :-1],
            args.prefill_chunk_size,
        )
        # DynamicCache.update allocates the new K/V tensors before releasing the
        # old ones.  Keeping the legacy tuple alive here duplicates the entire
        # long-context cache and causes a 24 GB card to OOM around 20K.  Transfer
        # ownership to one mutable cache object before the final query token.
        query_cache = base.cache_from_legacy(base_cache)
        del base_cache
        query_output, captured, query_seconds = capture_query_states(
            model,
            query_cache,
            prompt[:, -1:],
            prompt_len_minus_one,
        )
        full_pre_softmax_dir = (
            Path(args.export_full_pre_softmax_dir) / f"length_{length}"
            if args.export_full_pre_softmax_dir
            else None
        )
        token_type_pre_softmax_path = (
            Path(args.export_token_type_pre_softmax_dir) / f"length_{length}.json.gz"
            if args.export_token_type_pre_softmax_dir
            else None
        )
        unique_prompt_token_ids = sorted(set(int(token_id) for token_id in prompt_ids))
        prompt_token_text = {
            str(token_id): tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in unique_prompt_token_ids
        }
        attention, needed_positions = summarize_attention(
            model,
            query_output,
            captured,
            spans,
            args.max_top,
            full_pre_softmax_dir,
            token_type_pre_softmax_path,
            prompt_ids,
            prompt_token_text,
        )
        answer = score_gold_from_query_output(
            model,
            tokenizer,
            query_output,
            int(prompt.shape[1]),
            gold_answer,
            completion_text=(
                f" {gold_answer}"
                if args.code_mode == "english_single_token"
                else gold_answer
            ),
            candidate_texts=(
                [f" {code}" for code in body["gold_codes"]]
                if args.code_mode == "english_single_token"
                else None
            ),
        )
        if args.code_mode == "english_single_token" and answer["gold_token_count"] != 1:
            raise RuntimeError(
                f"English completion must be one token, got {answer['gold_token_count']}: "
                f"{answer['gold_token_scores']}"
            )
        token_table = build_token_table(
            tokenizer,
            prompt_ids,
            needed_positions,
            int(body["body_tokens"]),
            spans,
        )
        if full_pre_softmax_dir is not None:
            write_gzip_json_atomic(
                full_pre_softmax_dir / "tokens.json.gz",
                {
                    "schema_version": 1,
                    "key_length": int(prompt.shape[1]),
                    "storage_dtype": "uint32_le_base64",
                    "token_ids_u32_b64": tensor_u32_base64(prompt_ids),
                    "token_text": prompt_token_text,
                    "body_tokens": int(body["body_tokens"]),
                    "spans": {
                        role: [list(span) for span in role_spans_]
                        for role, role_spans_ in spans.items()
                    },
                },
            )
            write_json_atomic(
                full_pre_softmax_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "model": "Qwen3-8B",
                    "code_mode": args.code_mode,
                    "placement": args.placement,
                    "query_mode": args.query_mode,
                    "evidence_hops": args.evidence_hops,
                    "target_context_tokens": length,
                    "prompt_tokens": int(prompt.shape[1]),
                    "key_length": attention["key_length"],
                    "num_layers": model_config["num_layers"],
                    "num_attention_heads": model_config["num_attention_heads"],
                    "gold_codes": body["gold_codes"],
                    "role_order": list(ROLE_ORDER),
                    "storage_dtype": "float16_le_base64",
                    "probability_definition": "exp(logit - head_logsumexp) for head scope; exact mean post-softmax probability for layer/overall scope",
                    "files": {
                        "tokens": "tokens.json.gz",
                        "overall": "overall.json.gz",
                        "layer_pattern": "layers/layer_{layer:02d}.json.gz",
                        "head_pattern": "heads/layer_{layer:02d}_head_{head:02d}.json.gz",
                    },
                },
            )
        payload = {
            "schema_version": 1,
            "model": "Qwen3-8B",
            "model_config": model_config,
            "seed": args.seed,
            "condition": "clean",
            "code_mode": args.code_mode,
            "gold_code_token_ids": [
                body.get("code_token_ids", {}).get(code)
                for code in body["gold_codes"]
            ],
            "placement": args.placement,
            "target_context_tokens": length,
            "body_tokens": int(body["body_tokens"]),
            "prompt_tokens": int(prompt.shape[1]),
            "gold_codes": body["gold_codes"],
            "query": {
                "mode": args.query_mode,
                "prompt_style": args.prompt_style,
                "start_code": start_code,
                "required_steps": steps,
                "suffix": suffix,
            },
            "evidence_hops": args.evidence_hops,
            "spans": {role: [list(span) for span in role_spans_] for role, role_spans_ in spans.items()},
            "attention": attention,
            "answer": answer,
            "token_table_columns": ["position", "token_id", "text", "role"],
            "token_table": token_table,
            "timing": {
                "prefill_seconds": rounded(prefill_seconds),
                "query_seconds": rounded(query_seconds),
                "total_seconds": rounded(time.perf_counter() - started),
            },
        }
        write_json_atomic(destination, payload)
        print(
            f"[{case_index}/{len(lengths)}] length={length} prompt={prompt.shape[1]} "
            f"ppl={answer['gold_ppl']:.6f} entropy={attention['overall_entropy']:.4f} "
            f"seconds={payload['timing']['total_seconds']:.2f}",
            flush=True,
        )
        del query_cache, query_output, captured, attention
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / f"done_{args.shard_label}.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
