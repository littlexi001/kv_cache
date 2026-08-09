from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
for directory in (
    PROJECTS / "qwen3_ruler_head_frequency_ablation" / "src",
    PROJECTS / "qwen3_inference_rnope" / "src",
    PROJECTS / "qwen3_ruler32k_rope_method" / "src",
    PROJECTS / "qwen3_longbench_rope_method_exploration" / "src",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_frequency_sweep as sweep  # noqa: E402
import run_inference_rnope_ruler as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--target-length", type=int, default=32768)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--high-frequency-end", type=int, default=8)
    return parser.parse_args()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    return tuple(cache)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def apply_rope(
    value: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    angles = positions.float()[:, None] * inv_freq[None, :]
    doubled = torch.cat((angles, angles), dim=-1)
    cosine = doubled.cos().to(value.dtype) * scaling.to(value.dtype)
    sine = doubled.sin().to(value.dtype) * scaling.to(value.dtype)
    while cosine.dim() < value.dim():
        cosine = cosine.unsqueeze(0)
        sine = sine.unsqueeze(0)
    return value * cosine + rotate_half(value) * sine


def invert_rope(
    value: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    angles = positions.float()[:, None] * inv_freq[None, :]
    doubled = torch.cat((angles, angles), dim=-1)
    cosine = doubled.cos().to(value.dtype)
    sine = doubled.sin().to(value.dtype)
    while cosine.dim() < value.dim():
        cosine = cosine.unsqueeze(0)
        sine = sine.unsqueeze(0)
    unscaled = value / scaling.to(value.dtype)
    return unscaled * cosine - rotate_half(unscaled) * sine


def subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    return [index for index in range(len(sequence) - width + 1) if list(sequence[index:index + width]) == list(pattern)]


def gold_positions(tokenizer: Any, prompt_ids: Sequence[int], answers: Sequence[str]) -> list[int]:
    positions: set[int] = set()
    for answer in answers:
        patterns: set[tuple[int, ...]] = set()
        for text in (answer, " " + answer):
            encoded = tuple(tokenizer.encode(text, add_special_tokens=False))
            if encoded:
                patterns.add(encoded)
        for pattern in patterns:
            for start in subsequence_starts(prompt_ids, pattern):
                positions.update(range(start, start + len(pattern)))
    return sorted(position for position in positions if position < len(prompt_ids) - 1)


def mass_and_rank(logits: torch.Tensor, evidence_positions: torch.Tensor) -> tuple[float, int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    mass = float(probabilities[evidence_positions].sum().item())
    best_evidence = logits[evidence_positions].max()
    rank = 1 + int((logits > best_evidence).sum().item())
    return mass, rank


def normalized_pair_energy(query: torch.Tensor) -> torch.Tensor:
    half = query.shape[-1] // 2
    energy = query[:half].float().square() + query[half:].float().square()
    return energy / energy.sum().clamp_min(1e-30)


def counterfactual_high_nope_logits(
    query_post: torch.Tensor,
    key_post: torch.Tensor,
    query_position: int,
    inv_freq: torch.Tensor,
    high_end: int,
    attention_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native logits and logits with F0:high_end relative phase set to zero."""
    native_logits = torch.matmul(key_post.float(), query_post.float()) * attention_scale
    half = int(query_post.shape[-1] // 2)
    key_positions = torch.arange(key_post.shape[-2], device=key_post.device)
    relative_distance = query_position - key_positions
    correction = torch.zeros_like(native_logits)
    for frequency in range(high_end):
        qx = query_post[frequency].float()
        qy = query_post[frequency + half].float()
        kx = key_post[:, frequency].float()
        ky = key_post[:, frequency + half].float()
        native_pair = qx * kx + qy * ky
        cross_pair = qy * kx - qx * ky
        phase_delta = relative_distance.float() * inv_freq[frequency].float()
        desired_pair = native_pair * phase_delta.cos() + cross_pair * phase_delta.sin()
        correction = correction + (desired_pair - native_pair) * attention_scale
    return native_logits, native_logits + correction


def capture_evidence_keys_and_prefill(
    model: Any,
    prompt: torch.Tensor,
    evidence_positions: Sequence[int],
    prefill_chunk_size: int,
):
    captured: dict[int, dict[int, torch.Tensor]] = {
        layer: {} for layer in range(len(model.model.layers))
    }
    wanted = set(map(int, evidence_positions))
    handles = []

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            cache_position = kwargs.get("cache_position")
            position_ids = kwargs.get("position_ids")
            if hidden_states is None:
                raise RuntimeError(f"layer {layer_index}: missing prefill hidden states")
            if cache_position is not None:
                positions = cache_position.reshape(-1).tolist()
            elif position_ids is not None:
                positions = position_ids.reshape(-1).tolist()
            else:
                raise RuntimeError(f"layer {layer_index}: missing prefill positions")
            selected = [(local, int(position)) for local, position in enumerate(positions) if int(position) in wanted]
            if not selected:
                return
            projected = module.k_proj(hidden_states)
            batch, key_length, _ = projected.shape
            head_dim = int(module.head_dim)
            kv_heads = int(projected.shape[-1] // head_dim)
            key = projected.view(batch, key_length, kv_heads, head_dim)
            key_norm = getattr(module, "k_norm", None)
            if key_norm is not None:
                key = key_norm(key)
            for local, position in selected:
                captured[layer_index][position] = key[0, local].detach()

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True))
    try:
        legacy, prefill_seconds = base.core.cache_runner.prefill_sequence(
            model, prompt[:, :-1], prefill_chunk_size
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = {
        layer: sorted(wanted - set(layer_values))
        for layer, layer_values in captured.items()
        if wanted - set(layer_values)
    }
    if missing:
        first_layer = min(missing)
        raise RuntimeError(
            f"missing {len(missing[first_layer])} evidence K positions in layer {first_layer}"
        )
    return legacy, prefill_seconds, captured


def capture_final_query(model: Any, prompt: torch.Tensor, cache: Any, prefix_length: int):
    captured_pre: dict[int, torch.Tensor] = {}
    captured_post: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError(f"layer {layer_index}: missing hook inputs")
            projected = module.q_proj(hidden_states)
            batch, query_length, _ = projected.shape
            head_dim = int(module.head_dim)
            heads = int(projected.shape[-1] // head_dim)
            query = projected.view(batch, query_length, heads, head_dim)
            query_norm = getattr(module, "q_norm", None)
            if query_norm is not None:
                query = query_norm(query)
            query = query.transpose(1, 2)
            cosine, sine = position_embeddings
            if cosine.dim() == 2:
                cosine, sine = cosine.unsqueeze(0), sine.unsqueeze(0)
            cosine = cosine.unsqueeze(1).to(query)
            sine = sine.unsqueeze(1).to(query)
            post = query * cosine + rotate_half(query) * sine
            captured_pre[layer_index] = query[0, :, -1].detach()
            captured_post[layer_index] = post[0, :, -1].detach()

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True))
    try:
        logits, output_cache, query_seconds = base.run_last(model, prompt, cache, prefix_length)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured_pre) != len(model.model.layers):
        raise RuntimeError(f"captured {len(captured_pre)} layers")
    return logits, output_cache, query_seconds, captured_pre, captured_post


@torch.inference_mode()
def profile_example(model: Any, tokenizer: Any, example: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt, _, _ = base.ruler.make_prompt(tokenizer, example)
    prompt_ids = prompt[0].tolist()
    evidence = gold_positions(tokenizer, prompt_ids, example.answers)
    if not evidence:
        return [], {"sample_id": example.sample_id, "status": "no_gold_token_occurrence"}

    prefix_length = int(prompt.shape[-1]) - 1
    prefill_legacy, prefill_seconds, evidence_key_pre = capture_evidence_keys_and_prefill(
        model, prompt, evidence, args.prefill_chunk_size
    )
    cache = base.core.cache_runner.cache_from_legacy(prefill_legacy)
    del prefill_legacy
    logits, cache, query_seconds, query_pre, query_post = capture_final_query(
        model, prompt, cache, prefix_length
    )
    layers_cache = legacy_cache(cache)
    evidence_tensor_cpu = torch.tensor(evidence, dtype=torch.long)
    query_position = int(prompt.shape[-1]) - 1
    high_end = int(args.high_frequency_end)
    rows: list[dict[str, Any]] = []
    max_q_reconstruction_error = 0.0

    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.float()
    attention_scaling = rotary.attention_scaling
    if not torch.is_tensor(attention_scaling):
        attention_scaling = torch.tensor(float(attention_scaling), device=inv_freq.device)

    for layer_index, layer in enumerate(model.model.layers):
        key_post = layers_cache[layer_index][0][0]
        key_length = int(key_post.shape[-2])
        live_inv = inv_freq.to(key_post.device)
        live_scaling = attention_scaling.to(key_post.device)

        q_pre_layer = query_pre[layer_index]
        q_post_layer = query_post[layer_index]
        q_reconstructed = apply_rope(
            q_pre_layer.unsqueeze(-2),
            torch.tensor([query_position], device=q_pre_layer.device),
            live_inv.to(q_pre_layer.device),
            live_scaling.to(q_pre_layer.device),
        )[:, 0]
        q_error = float((q_reconstructed.float() - q_post_layer.float()).abs().max().item())
        max_q_reconstruction_error = max(max_q_reconstruction_error, q_error)

        query_heads = int(q_post_layer.shape[0])
        kv_heads = int(key_post.shape[0])
        group_size = query_heads // kv_heads
        head_dim = int(q_post_layer.shape[-1])
        half = head_dim // 2
        evidence_tensor = evidence_tensor_cpu.to(key_post.device)
        attn_scale = float(getattr(layer.self_attn, "scaling", head_dim ** -0.5))

        for head in range(query_heads):
            kv_head = head // group_size
            q_post_head = q_post_layer[head].to(key_post.device)
            q_pre_head = q_pre_layer[head].to(key_post.device)
            k_post_head = key_post[kv_head]
            native_logits, nope_logits = counterfactual_high_nope_logits(
                q_post_head,
                k_post_head,
                query_position,
                live_inv,
                high_end,
                attn_scale,
            )
            native_mass, native_rank = mass_and_rank(native_logits, evidence_tensor)
            nope_mass, nope_rank = mass_and_rank(nope_logits, evidence_tensor)

            evidence_keys = torch.stack(
                [evidence_key_pre[layer_index][position][kv_head] for position in evidence]
            ).float().to(q_pre_head.device)
            pre_cosine = F.cosine_similarity(
                evidence_keys,
                q_pre_head.float().unsqueeze(0).expand_as(evidence_keys),
                dim=-1,
            )
            energy = normalized_pair_energy(q_pre_head)
            uniform = torch.full_like(energy, 1.0 / len(energy))
            distances = query_position - evidence_tensor_cpu
            rows.append({
                "sample_id": example.sample_id,
                "task": base.ruler.base_task(example.task),
                "layer": layer_index,
                "query_head": head,
                "kv_head": kv_head,
                "prompt_tokens": len(prompt_ids),
                "evidence_token_count": len(evidence),
                "evidence_distance_min": int(distances.min().item()),
                "evidence_distance_max": int(distances.max().item()),
                "pre_rope_cosine_mean": float(pre_cosine.mean().item()),
                "pre_rope_cosine_max": float(pre_cosine.max().item()),
                "q_pair_energy_l1_uniform": float((energy - uniform).abs().sum().item()),
                "q_high8_energy_mass": float(energy[:high_end].sum().item()),
                "native_evidence_logit_mean": float(native_logits[evidence_tensor].mean().item()),
                "nope_high_evidence_logit_mean": float(nope_logits[evidence_tensor].mean().item()),
                "high_phase_contribution_mean": float(
                    (native_logits[evidence_tensor] - nope_logits[evidence_tensor]).mean().item()
                ),
                "native_evidence_mass": native_mass,
                "nope_high_evidence_mass": nope_mass,
                "native_best_evidence_rank": native_rank,
                "nope_high_best_evidence_rank": nope_rank,
                "evidence_mass_delta_nope_minus_native": nope_mass - native_mass,
                "evidence_rank_delta_nope_minus_native": nope_rank - native_rank,
                "q_reconstruction_max_error": q_error,
                "evidence_pre_rope_k_captured": len(evidence),
            })

    first_nll, first_correct = base.first_answer_stats(tokenizer, logits, example.answers[0])
    summary = {
        "sample_id": example.sample_id,
        "status": "complete",
        "prompt_tokens": len(prompt_ids),
        "evidence_token_count": len(evidence),
        "prefill_seconds": prefill_seconds,
        "query_seconds": query_seconds,
        "first_answer_nll": first_nll,
        "first_answer_correct": first_correct,
        "max_q_reconstruction_error": max_q_reconstruction_error,
        "evidence_pre_rope_k_positions_captured_per_layer": len(evidence),
    }
    del cache, prompt
    base.core.local_global.clear_allocator()
    return rows, summary


def main() -> None:
    args = parse_args()
    if not 1 <= args.high_frequency_end <= 64:
        raise ValueError("high-frequency-end must lie in [1, 64]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=40960,
        global_max_position=40960,
        load_in_4bit=bool(args.load_in_4bit),
        device_map="auto",
    )
    model, tokenizer = base.core.local_global.load_model(load_args)
    examples = sweep.select_examples(args.examples_jsonl, args.sample_ids, args.target_length)
    base.write_json(
        args.output_dir / "config.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "selected_examples": [asdict(example) | {"context": "<omitted>"} for example in examples],
            "fixed_hidden_state_counterfactual": True,
            "counterfactual": "replace F0:F(high_frequency_end) native phase contribution with pre-RoPE content dot product",
        },
    )
    rows_path = args.output_dir / "head_rows.jsonl"
    summaries = []
    for example in examples:
        rows, summary = profile_example(model, tokenizer, example, args)
        append_jsonl(rows_path, rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    base.write_json(args.output_dir / "sample_summaries.json", summaries)
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
