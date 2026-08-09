from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import statistics
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base


RETRIEVAL_METHODS = ("post_top2", "pre_top2", "envelope_top2")
VARIANTS = (
    ("full_attention", "full", False),
    ("post_top2", "post_top2", False),
    ("post_top2_repair", "post_top2", True),
    ("pre_top2", "pre_top2", False),
    ("pre_top2_repair", "pre_top2", True),
    ("envelope_top2", "envelope_top2", False),
    ("envelope_top2_repair", "envelope_top2", True),
)

_ACTIVE_CONTROLLER: "RetrievalController | None" = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_angles(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.outer(
        positions.to(device=inv_freq.device, dtype=torch.float32),
        inv_freq[: head_dim // 2].to(dtype=torch.float32),
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


def invert_rope(
    post: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    """Recover pre-RoPE split-half vectors from scaled post-RoPE vectors."""

    cos, sin = rope_angles(positions, inv_freq, post.shape[-1], post.dtype)
    view = (1,) * (post.dim() - 2) + cos.shape
    cos = cos.view(view)
    sin = sin.view(view)
    scaled = post / float(attention_scaling)
    return scaled * cos - rotate_half(scaled) * sin


def apply_rope_delta(
    post: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Move already-RoPE'd keys from old positions to new virtual positions."""

    delta = (new_positions - old_positions).to(dtype=torch.float32)
    frequencies = delta.unsqueeze(-1) * inv_freq.to(
        device=post.device, dtype=torch.float32
    ).view(1, 1, -1)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(dtype=post.dtype)
    sin = embedding.sin().to(dtype=post.dtype)
    return post * cos + rotate_half(post) * sin


def envelope_scores(query_pre: torch.Tensor, key_pre: torch.Tensor) -> torch.Tensor:
    """Per-pair phase envelope, summed over split-half RoPE pairs."""

    half = query_pre.shape[-1] // 2
    q0, q1 = query_pre[..., :half], query_pre[..., half:]
    k0, k1 = key_pre[..., :half], key_pre[..., half:]
    dot = q0 * k0 + q1 * k1
    cross = q1 * k0 - q0 * k1
    return torch.sqrt(dot.float().square() + cross.float().square() + 1e-12).sum(dim=-1)


def force_current_topk(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Select a fixed-size history set and always retain the current query token."""

    if scores.dim() != 2:
        raise ValueError(f"scores must be [heads, keys], got {tuple(scores.shape)}")
    key_count = scores.shape[-1]
    keep_count = min(key_count, max(1, int(keep_count)))
    current = key_count - 1
    if keep_count == 1:
        return torch.full(
            (scores.shape[0], 1), current, dtype=torch.long, device=scores.device
        )
    history = scores[:, :current]
    history_count = min(keep_count - 1, history.shape[-1])
    chosen = torch.topk(history.float(), k=history_count, dim=-1, largest=True).indices
    current_column = torch.full(
        (scores.shape[0], 1), current, dtype=torch.long, device=scores.device
    )
    selected = torch.cat((chosen, current_column), dim=-1)
    return selected.sort(dim=-1).values


def virtual_positions(selected: torch.Tensor, query_position: int) -> torch.Tensor:
    """Pack selected history immediately before the query while keeping self fixed."""

    if selected.dim() != 2:
        raise ValueError("selected positions must be [heads, kept_tokens]")
    head_count, kept = selected.shape
    output = selected.clone()
    if kept > 1:
        history = torch.arange(
            query_position - (kept - 1),
            query_position,
            device=selected.device,
            dtype=selected.dtype,
        )
        output[:, :-1] = history.view(1, -1).expand(head_count, -1)
    output[:, -1] = query_position
    return output


def repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    return hidden if groups == 1 else hidden.repeat_interleave(groups, dim=1)


def add_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return scores
    return scores + attention_mask[:, :, -scores.shape[-2] :, : scores.shape[-1]]


@dataclass
class MetricAccumulator:
    head_rows: int = 0
    selected_gold: int = 0
    selectable_gold: int = 0
    line_hits: int = 0
    line_events: int = 0
    chain_hits: int = 0
    attention_mass_sum: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "gold_evidence_token_recall": self.selected_gold / max(1, self.selectable_gold),
            "gold_evidence_line_hit_rate": self.line_hits / max(1, self.line_events),
            "gold_chain_complete_rate": self.chain_hits / max(1, self.head_rows),
            "gold_evidence_attention_mass": self.attention_mass_sum / max(1, self.head_rows),
        }


@dataclass
class RetrievalController:
    phase: str
    method: str = "full"
    repair: bool = False
    ratio: float = 0.02
    evidence_spans: tuple[tuple[int, int], ...] = ()
    selections: dict[str, dict[int, torch.Tensor]] = field(
        default_factory=lambda: {method: {} for method in RETRIEVAL_METHODS}
    )
    metrics: MetricAccumulator = field(default_factory=MetricAccumulator)
    collect_metrics: bool = True
    dynamic_selection: bool = False

    def evidence_mask(self, key_count: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(key_count, dtype=torch.bool, device=device)
        for start, end in self.evidence_spans:
            mask[max(0, start) : min(key_count, end)] = True
        return mask

    def record(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        key_count: int,
    ) -> None:
        if not self.collect_metrics:
            return
        # positions: [heads, selected], weights: [1, heads, 1, selected]
        gold = self.evidence_mask(key_count, positions.device)
        selected_gold = gold[positions]
        head_count = positions.shape[0]
        self.metrics.head_rows += head_count
        self.metrics.selected_gold += int(selected_gold.sum().item())
        self.metrics.selectable_gold += int(gold.sum().item()) * head_count
        for start, end in self.evidence_spans:
            hits = ((positions >= start) & (positions < end)).any(dim=-1)
            self.metrics.line_hits += int(hits.sum().item())
            self.metrics.line_events += head_count
        if self.evidence_spans:
            line_matrix = torch.stack(
                [((positions >= start) & (positions < end)).any(dim=-1) for start, end in self.evidence_spans],
                dim=-1,
            )
            self.metrics.chain_hits += int(line_matrix.all(dim=-1).sum().item())
        mass = weights[0, :, 0, :].float().masked_fill(~selected_gold, 0.0).sum(dim=-1)
        self.metrics.attention_mass_sum += float(mass.sum().item())


@contextmanager
def activate_controller(controller: RetrievalController | None):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


def _attention_scaling(position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> float:
    cos, sin = position_embeddings
    value = torch.sqrt(cos.float().reshape(-1)[0].square() + sin.float().reshape(-1)[0].square())
    return float(value.item())


def retrieval_attention_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_value: Any | None = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    controller = _ACTIVE_CONTROLLER
    if controller is None or hidden_states.shape[-2] != 1:
        return self._rope_retrieval_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._rope_retrieval_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_key_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    current_value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    cos = cos.to(device=query_pre.device)
    sin = sin.to(device=query_pre.device)
    query_post, current_key_post = modeling_qwen3.apply_rotary_pos_emb(
        query_pre, current_key_pre, cos, sin
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_post, value = past_key_value.update(
            current_key_post, current_value, self.layer_idx, cache_kwargs
        )
    else:
        key_post, value = current_key_post, current_value

    groups = query_post.shape[1] // key_post.shape[1]
    expanded_key_post = repeat_kv(key_post, groups)
    expanded_value = repeat_kv(value, groups)
    key_count = int(expanded_key_post.shape[-2])
    query_position = key_count - 1
    scaling = float(getattr(self, "scaling", 1.0 / math.sqrt(query_post.shape[-1])))
    post_scores = torch.matmul(query_post, expanded_key_post.transpose(2, 3)) * scaling
    post_scores = add_attention_mask(post_scores, attention_mask)

    if controller.phase == "plan":
        positions = torch.arange(key_count, device=key_post.device)
        inv_freq = self._rope_retrieval_inv_freq.to(device=key_post.device)
        scale = _attention_scaling((cos, sin))
        key_pre = invert_rope(key_post, positions, inv_freq, scale)
        expanded_key_pre = repeat_kv(key_pre, groups)
        pre_scores = torch.matmul(query_pre, expanded_key_pre.transpose(2, 3)) * scaling
        pre_scores = add_attention_mask(pre_scores, attention_mask)
        amplitude_scores = envelope_scores(
            query_pre.expand(-1, -1, key_count, -1), expanded_key_pre
        ).unsqueeze(2) * scaling
        amplitude_scores = add_attention_mask(amplitude_scores, attention_mask)
        keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
        score_map = {
            "post_top2": post_scores,
            "pre_top2": pre_scores,
            "envelope_top2": amplitude_scores,
        }
        for method, scores in score_map.items():
            controller.selections[method][int(self.layer_idx)] = force_current_topk(
                scores[0, :, 0, :], keep_count
            ).detach().cpu()
        weights = F.softmax(post_scores.float(), dim=-1).to(query_post.dtype)
        all_positions = positions.view(1, -1).expand(query_post.shape[1], -1)
        controller.record(all_positions, weights, key_count)
        attention_output = torch.matmul(weights, expanded_value)
    else:
        if controller.dynamic_selection:
            if controller.method == "post_top2":
                retrieval_scores = post_scores
            else:
                positions = torch.arange(key_count, device=key_post.device)
                inv_freq = self._rope_retrieval_inv_freq.to(device=key_post.device)
                scale = _attention_scaling((cos, sin))
                key_pre = invert_rope(key_post, positions, inv_freq, scale)
                expanded_key_pre = repeat_kv(key_pre, groups)
                if controller.method == "pre_top2":
                    retrieval_scores = torch.matmul(
                        query_pre, expanded_key_pre.transpose(2, 3)
                    ) * scaling
                elif controller.method == "envelope_top2":
                    retrieval_scores = envelope_scores(
                        query_pre.expand(-1, -1, key_count, -1), expanded_key_pre
                    ).unsqueeze(2) * scaling
                else:
                    raise ValueError(f"unknown retrieval method: {controller.method}")
                retrieval_scores = add_attention_mask(retrieval_scores, attention_mask)
            keep_count = max(1, int(math.ceil(controller.ratio * key_count)))
            selected = force_current_topk(
                retrieval_scores[0, :, 0, :], keep_count
            )
        else:
            selected = controller.selections[controller.method][int(self.layer_idx)].to(
                device=key_post.device
            ).clone()
            # The frozen plan reserves its last slot for the planning query token.
            selected[:, -1] = query_position
        gather_index = selected.view(1, selected.shape[0], -1, 1).expand(
            1, selected.shape[0], selected.shape[1], expanded_key_post.shape[-1]
        )
        selected_key = expanded_key_post.gather(2, gather_index)
        selected_value = expanded_value.gather(2, gather_index)
        if controller.repair:
            new_positions = virtual_positions(selected, query_position)
            selected_key = apply_rope_delta(
                selected_key,
                selected,
                new_positions,
                self._rope_retrieval_inv_freq,
            )
        sparse_scores = torch.matmul(query_post, selected_key.transpose(2, 3)) * scaling
        weights = F.softmax(sparse_scores.float(), dim=-1).to(query_post.dtype)
        controller.record(selected, weights, key_count)
        attention_output = torch.matmul(weights, selected_value)

    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = self.o_proj(attention_output)
    return attention_output, weights


def patch_model(model: Any) -> None:
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except Exception as exc:
        raise RuntimeError("transformers Qwen3 implementation is required") from exc
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().float().cpu()
    found = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3Attention":
            continue
        if not hasattr(module, "_rope_retrieval_original_forward"):
            module._rope_retrieval_original_forward = module.forward
            module._rope_retrieval_modeling_qwen3 = modeling_qwen3
            module._rope_retrieval_inv_freq = inv_freq
            module.forward = types.MethodType(retrieval_attention_forward, module)
        found += 1
    if not found:
        raise RuntimeError("no Qwen3Attention modules found")


def reset_dynamic_cache(cache: Any, prefix_length: int) -> None:
    """Remove the one appended query token without retaining a second long cache."""

    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise TypeError("experiment requires a DynamicCache with key_cache/value_cache")
    for index in range(len(cache.key_cache)):
        cache.key_cache[index] = cache.key_cache[index][:, :, :prefix_length, :]
        cache.value_cache[index] = cache.value_cache[index][:, :, :prefix_length, :]
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = prefix_length


def seeded_clean_body(
    tokenizer: Any,
    *,
    seed: int,
    target_context_tokens: int,
    placement: str,
) -> dict[str, Any]:
    words = causal.build_english_single_token_code_pool(tokenizer, required=64)
    rng = random.Random(2026072001 + seed * 1009)
    rng.shuffle(words)
    codes = words[:3]
    events = [
        causal.make_event("relevant", f"T{step}", "VERIFIED RULE", codes[step], codes[step + 1], step)
        for step in range(2)
    ]
    encoded, _ = causal.encode_event_block(tokenizer, events, 0)
    if target_context_tokens <= len(encoded):
        raise ValueError("target context is too short for the evidence block")
    body = base.build_filler_ids(tokenizer, target_context_tokens, 2_100_000 + seed)
    if placement == "prefix":
        preferred = min(256, target_context_tokens - len(encoded))
    elif placement == "middle":
        preferred = target_context_tokens // 2 - len(encoded) // 2
    elif placement == "recent":
        preferred = max(0, target_context_tokens - len(encoded) - 256)
    else:
        raise ValueError(f"unknown placement: {placement}")
    placed: list[base.RuleEvent] = []
    occupied: list[tuple[int, int]] = []
    causal.insert_event_block(body, tokenizer, events, preferred, occupied, placed)
    placed.sort(key=lambda event: event.start_token)
    return {
        "body_ids": torch.tensor(body, dtype=torch.long).view(1, -1),
        "body_tokens": len(body),
        "events": placed,
        "gold_codes": codes,
    }


def answer_metrics(tokenizer: Any, logits: torch.Tensor, gold_answer: str) -> dict[str, Any]:
    ids = tokenizer(f" {gold_answer}", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"gold completion must be one token: {gold_answer!r} -> {ids}")
    gold_id = int(ids[0])
    log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
    nll = -float(log_probs[0, gold_id].item())
    prediction_id = int(logits[:, -1, :].argmax(dim=-1).item())
    return {
        "gold_token_id": gold_id,
        "gold_nll": nll,
        "gold_ppl": math.exp(nll),
        "prediction_token_id": prediction_id,
        "prediction_text": tokenizer.decode(
            [prediction_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "next_token_correct": int(prediction_id == gold_id),
    }


def extract_generation_answer(text: str, valid_codes: Sequence[str]) -> dict[str, Any]:
    canonical = {code.lower(): code for code in valid_codes}
    if not canonical:
        return {
            "generation_mentions": [],
            "generation_final_answer": "",
        }
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(code) for code in sorted(canonical, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    mentions = [canonical[match.group(0).lower()] for match in pattern.finditer(text)]
    return {
        "generation_mentions": mentions,
        "generation_final_answer": mentions[-1] if mentions else "",
    }


@torch.inference_mode()
def greedy_generation(
    model: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    cache: Any,
    prompt_length: int,
    max_new_tokens: int,
    valid_codes: Sequence[str],
    gold_answer: str,
    controller: RetrievalController | None,
) -> dict[str, Any]:
    generated: list[int] = []
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_length = prompt_length
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            break
        generated.append(token_id)
        with activate_controller(controller):
            output = base.forward_with_cache(
                model, next_token.to(base.input_device(model)), cache, past_length
            )
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        past_length += 1
    text = tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    extracted = extract_generation_answer(text, valid_codes)
    final_answer = str(extracted["generation_final_answer"])
    mentions = list(extracted["generation_mentions"])
    return {
        "generated_text": text.replace("\n", "\\n"),
        "generated_tokens": len(generated),
        "generation_mentions": " ".join(mentions),
        "generation_contains_gold": int(gold_answer in mentions),
        "generation_final_answer": final_answer,
        "generation_final_correct": int(final_answer == gold_answer),
        "generation_seconds": time.perf_counter() - started,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant, _, _ in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        mean_nll = statistics.fmean(float(row["gold_nll"]) for row in selected)
        output.append(
            {
                "variant": variant,
                "sample_count": len(selected),
                "gold_evidence_token_recall": statistics.fmean(
                    float(row["gold_evidence_token_recall"]) for row in selected
                ),
                "gold_evidence_line_hit_rate": statistics.fmean(
                    float(row["gold_evidence_line_hit_rate"]) for row in selected
                ),
                "gold_chain_complete_rate": statistics.fmean(
                    float(row["gold_chain_complete_rate"]) for row in selected
                ),
                "gold_evidence_attention_mass": statistics.fmean(
                    float(row["gold_evidence_attention_mass"]) for row in selected
                ),
                "mean_gold_nll": mean_nll,
                "gold_answer_ppl": math.exp(mean_nll),
                "final_answer_accuracy": statistics.fmean(
                    int(row["generation_final_correct"]) for row in selected
                ),
                "next_token_accuracy": statistics.fmean(
                    int(row["next_token_correct"]) for row in selected
                ),
                "generation_contains_gold_rate": statistics.fmean(
                    int(row["generation_contains_gold"]) for row in selected
                ),
                "mean_query_seconds": statistics.fmean(
                    float(row["query_seconds"]) for row in selected
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare post-RoPE Top-2%, RoPE-free retrieval, and virtual-position repair."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_context_tokens", type=int, default=64000)
    parser.add_argument("--placement", choices=("prefix", "middle", "recent"), default="prefix")
    parser.add_argument("--prompt_style", choices=("legacy", "chat_concise"), default="chat_concise")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=16)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--global_max_position", type=int, default=130000)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(config_path.read_text(encoding="utf-8"), flush=True)
        return

    rope_factor = base.rope_factor_for_length(
        args.global_max_position, args.original_max_position_embeddings
    )
    model, tokenizer = base.load_model_and_tokenizer(
        args, args.global_max_position, rope_factor
    )
    patch_model(model)
    valid_codes = causal.build_english_single_token_code_pool(tokenizer, required=64)
    rows: list[dict[str, Any]] = []
    rows_path = output_dir / "rows.jsonl"
    completed = {
        int(json.loads(line)["seed"])
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if rows_path.exists() else set()

    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        if seed in completed:
            print(f"seed={seed} already complete; skipping", flush=True)
            continue
        case_started = time.perf_counter()
        body = seeded_clean_body(
            tokenizer,
            seed=seed,
            target_context_tokens=args.target_context_tokens,
            placement=args.placement,
        )
        if args.prompt_style == "chat_concise":
            suffix = causal.build_suffix(
                "chat_concise", body["gold_codes"][0], 2, "full2"
            )
            wrapper_prefix, wrapper_suffix = causal.chat_wrapper_ids(tokenizer)
            prompt_ids = (
                wrapper_prefix
                + body["body_ids"].view(-1).tolist()
                + base.token_ids(tokenizer, suffix)
                + wrapper_suffix
            )
            evidence_offset = len(wrapper_prefix)
        else:
            suffix = base.build_prompt_suffix(body["gold_codes"][0], 2)
            prompt_ids = body["body_ids"].view(-1).tolist() + base.token_ids(tokenizer, suffix)
            evidence_offset = 0
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        prefix_length = int(prompt.shape[1]) - 1
        evidence_spans = tuple(
            (
                evidence_offset + int(event.start_token),
                evidence_offset + int(event.end_token),
            )
            for event in body["events"]
        )
        prefix_legacy, prefill_seconds = base.prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size
        )
        cache = base.cache_from_legacy(prefix_legacy)
        del prefix_legacy

        base.synchronize()
        full_started = time.perf_counter()
        with torch.inference_mode():
            full_output = base.forward_with_cache(
                model, prompt[:, -1:].to(base.input_device(model)), cache, prefix_length
            )
        base.synchronize()
        full_seconds = time.perf_counter() - full_started
        full_answer = answer_metrics(tokenizer, full_output.logits, body["gold_codes"][-1])
        full_generation = greedy_generation(
            model,
            tokenizer,
            full_output.logits,
            cache,
            int(prompt.shape[1]),
            args.max_new_tokens,
            valid_codes,
            body["gold_codes"][-1],
            None,
        )
        del full_output
        reset_dynamic_cache(cache, prefix_length)

        plan = RetrievalController(
            phase="plan", ratio=args.ratio, evidence_spans=evidence_spans
        )
        base.synchronize()
        query_started = time.perf_counter()
        with activate_controller(plan), torch.inference_mode():
            plan_output = base.forward_with_cache(
                model, prompt[:, -1:].to(base.input_device(model)), cache, prefix_length
            )
        base.synchronize()
        plan_seconds = time.perf_counter() - query_started
        plan_answer = answer_metrics(tokenizer, plan_output.logits, body["gold_codes"][-1])
        plan_row = {
            "seed": seed,
            "variant": "full_attention",
            "retrieval_method": "full",
            "position_repair": 0,
            "target_context_tokens": args.target_context_tokens,
            "prompt_tokens": int(prompt.shape[1]),
            "placement": args.placement,
            "prompt_style": args.prompt_style,
            "gold_chain": " -> ".join(body["gold_codes"]),
            **plan.metrics.summary(),
            **full_answer,
            **full_generation,
            "planner_gold_ppl": plan_answer["gold_ppl"],
            "planner_next_token_correct": plan_answer["next_token_correct"],
            "prefill_seconds": prefill_seconds,
            "query_seconds": full_seconds,
            "planner_seconds": plan_seconds,
        }
        seed_rows = [plan_row]
        del plan_output
        reset_dynamic_cache(cache, prefix_length)

        for variant, method, repair in VARIANTS[1:]:
            controller = RetrievalController(
                phase="sparse",
                method=method,
                repair=repair,
                ratio=args.ratio,
                evidence_spans=evidence_spans,
                selections=plan.selections,
            )
            base.synchronize()
            query_started = time.perf_counter()
            with activate_controller(controller), torch.inference_mode():
                output = base.forward_with_cache(
                    model, prompt[:, -1:].to(base.input_device(model)), cache, prefix_length
                )
            base.synchronize()
            query_seconds = time.perf_counter() - query_started
            sparse_answer = answer_metrics(tokenizer, output.logits, body["gold_codes"][-1])
            controller.collect_metrics = False
            controller.dynamic_selection = True
            sparse_generation = greedy_generation(
                model,
                tokenizer,
                output.logits,
                cache,
                int(prompt.shape[1]),
                args.max_new_tokens,
                valid_codes,
                body["gold_codes"][-1],
                controller,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "retrieval_method": method,
                    "position_repair": int(repair),
                    "target_context_tokens": args.target_context_tokens,
                    "prompt_tokens": int(prompt.shape[1]),
                    "placement": args.placement,
                    "prompt_style": args.prompt_style,
                    "gold_chain": " -> ".join(body["gold_codes"]),
                    **controller.metrics.summary(),
                    **sparse_answer,
                    **sparse_generation,
                    "prefill_seconds": prefill_seconds,
                    "query_seconds": query_seconds,
                }
            )
            del output
            reset_dynamic_cache(cache, prefix_length)

        with rows_path.open("a", encoding="utf-8") as handle:
            for row in seed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.extend(seed_rows)
        summary = summarize(rows)
        write_csv(output_dir / "rows.csv", rows)
        write_csv(output_dir / "summary.csv", summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"seed={seed} chain={'->'.join(body['gold_codes'])} "
            f"prompt={prompt.shape[1]} prefill={prefill_seconds:.2f}s "
            f"case={time.perf_counter() - case_started:.2f}s",
            flush=True,
        )
        for row in seed_rows:
            print(
                f"  {row['variant']}: recall={row['gold_evidence_token_recall']:.4f} "
                f"mass={row['gold_evidence_attention_mass']:.6g} "
                f"ppl={row['gold_ppl']:.4f} correct={row['generation_final_correct']}",
                flush=True,
            )
        del cache, prompt, plan
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rows_path.exists():
        all_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = summarize(all_rows)
        write_csv(output_dir / "rows.csv", all_rows)
        write_csv(output_dir / "summary.csv", summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
