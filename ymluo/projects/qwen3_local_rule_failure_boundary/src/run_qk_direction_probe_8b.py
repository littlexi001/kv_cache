from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_attention_confidence_sweep_8b as sweep
import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base


VECTOR_ROLES = ("start_key", "hop1_result", "hop2_input", "hop2_result")


def parse_csv_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def position_slice(position_embeddings: Any, local_position: int) -> tuple[torch.Tensor, torch.Tensor]:
    if position_embeddings is None:
        raise RuntimeError("Q/K direction probe requires rotary position embeddings")
    cos, sin = position_embeddings
    if cos.dim() == 2:
        return cos[local_position].detach().cpu(), sin[local_position].detach().cpu()
    if cos.dim() == 3:
        return cos[0, local_position].detach().cpu(), sin[0, local_position].detach().cpu()
    raise RuntimeError(f"unexpected rotary embedding shape: {tuple(cos.shape)}")


def project_keys(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    projected = module.k_proj(hidden_states)
    batch, sequence_length, _ = projected.shape
    head_dim = int(getattr(module, "head_dim"))
    num_heads = int(projected.shape[-1] // head_dim)
    keys = projected.view(batch, sequence_length, num_heads, head_dim)
    key_norm = getattr(module, "k_norm", None)
    if key_norm is not None:
        keys = key_norm(keys)
    return keys.transpose(1, 2)


def project_queries(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    projected = module.q_proj(hidden_states)
    batch, sequence_length, _ = projected.shape
    head_dim = int(getattr(module, "head_dim"))
    num_heads = int(projected.shape[-1] // head_dim)
    queries = projected.view(batch, sequence_length, num_heads, head_dim)
    query_norm = getattr(module, "q_norm", None)
    if query_norm is not None:
        queries = query_norm(queries)
    return queries.transpose(1, 2)


def prefill_and_capture_keys(
    model: Any,
    prompt_prefix: torch.Tensor,
    chunk_size: int,
    role_positions: dict[str, int],
) -> tuple[Any, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]], dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    layers = list(model.model.layers)
    captured: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
        layer_index: {} for layer_index in range(len(layers))
    }
    position_vectors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    chunk_state = {"start": 0}

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
            chunk_start = int(chunk_state["start"])
            chunk_end = chunk_start + int(hidden_states.shape[1])
            selected = {
                role: position - chunk_start
                for role, position in role_positions.items()
                if chunk_start <= position < chunk_end
            }
            if not selected:
                return
            key_pre = project_keys(module, hidden_states)
            key_post = base.apply_rope_to_q(key_pre, position_embeddings)
            for role, local_position in selected.items():
                captured[layer_index][role] = (
                    key_pre[0, :, local_position, :].detach().cpu(),
                    key_post[0, :, local_position, :].detach().cpu(),
                )
                if layer_index == 0 and role not in position_vectors:
                    position_vectors[role] = position_slice(position_embeddings, local_position)

        return hook

    handles = [
        layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True)
        for layer_index, layer in enumerate(layers)
    ]
    device = base.input_device(model)
    ids = prompt_prefix.to(device)
    past = None
    past_len = 0
    base.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for chunk_start in range(0, int(ids.shape[1]), chunk_size):
                chunk_state["start"] = chunk_start
                chunk = ids[:, chunk_start : chunk_start + chunk_size]
                output = base.forward_with_cache(model, chunk, past, past_len)
                past = output.past_key_values
                past_len += int(chunk.shape[1])
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    missing = [
        (layer_index, role)
        for layer_index in range(len(layers))
        for role in VECTOR_ROLES
        if role not in captured[layer_index]
    ]
    if missing:
        raise RuntimeError(f"missing {len(missing)} captured role keys; first={missing[:5]}")
    return base.legacy_cache(past), captured, position_vectors, time.perf_counter() - started


def query_and_capture_both(
    model: Any,
    query_cache: Any,
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
) -> tuple[Any, dict[int, tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor], float]:
    layers = list(model.model.layers)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    query_position_vectors: tuple[torch.Tensor, torch.Tensor] | None = None

    def make_hook(layer_index: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            nonlocal query_position_vectors
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None:
                raise RuntimeError(f"layer {layer_index}: no hidden states in query hook")
            query_pre = project_queries(module, hidden_states)
            query_post = base.apply_rope_to_q(query_pre, position_embeddings)
            captured[layer_index] = (
                query_pre[0, :, -1, :].detach().cpu(),
                query_post[0, :, -1, :].detach().cpu(),
            )
            if layer_index == 0 and query_position_vectors is None:
                query_position_vectors = position_slice(position_embeddings, int(hidden_states.shape[1]) - 1)

        return hook

    handles = [
        layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True)
        for layer_index, layer in enumerate(layers)
    ]
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
    if len(captured) != len(layers) or query_position_vectors is None:
        raise RuntimeError(f"captured {len(captured)} query layers, expected {len(layers)}")
    return output, captured, query_position_vectors, time.perf_counter() - started


def stack_probe_arrays(
    model: Any,
    captured_keys: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]],
    captured_queries: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, np.ndarray]:
    layers = list(model.model.layers)
    q_pre = torch.stack([captured_queries[index][0] for index in range(len(layers))])
    q_post = torch.stack([captured_queries[index][1] for index in range(len(layers))])
    k_pre = torch.stack(
        [
            torch.stack([captured_keys[index][role][0] for index in range(len(layers))])
            for role in VECTOR_ROLES
        ]
    )
    k_post = torch.stack(
        [
            torch.stack([captured_keys[index][role][1] for index in range(len(layers))])
            for role in VECTOR_ROLES
        ]
    )
    post_logits = []
    post_cosines = []
    pre_logits = []
    pre_cosines = []
    for role_index in range(len(VECTOR_ROLES)):
        role_post_logits = []
        role_post_cosines = []
        role_pre_logits = []
        role_pre_cosines = []
        for layer_index, layer in enumerate(layers):
            num_query_heads = int(q_post[layer_index].shape[0])
            num_kv_heads = int(k_post[role_index, layer_index].shape[0])
            groups = max(1, num_query_heads // num_kv_heads)
            repeated_post = k_post[role_index, layer_index].repeat_interleave(groups, dim=0)
            repeated_pre = k_pre[role_index, layer_index].repeat_interleave(groups, dim=0)
            scale = float(getattr(layer.self_attn, "scaling", q_post.shape[-1] ** -0.5))
            post_dot = (q_post[layer_index].float() * repeated_post.float()).sum(dim=-1)
            pre_dot = (q_pre[layer_index].float() * repeated_pre.float()).sum(dim=-1)
            role_post_logits.append(post_dot * scale)
            role_pre_logits.append(pre_dot * scale)
            role_post_cosines.append(
                torch.nn.functional.cosine_similarity(q_post[layer_index].float(), repeated_post.float(), dim=-1)
            )
            role_pre_cosines.append(
                torch.nn.functional.cosine_similarity(q_pre[layer_index].float(), repeated_pre.float(), dim=-1)
            )
        post_logits.append(torch.stack(role_post_logits))
        post_cosines.append(torch.stack(role_post_cosines))
        pre_logits.append(torch.stack(role_pre_logits))
        pre_cosines.append(torch.stack(role_pre_cosines))
    return {
        "q_pre": q_pre.to(dtype=torch.float16).numpy(),
        "q_post": q_post.to(dtype=torch.float16).numpy(),
        "k_pre": k_pre.to(dtype=torch.float16).numpy(),
        "k_post": k_post.to(dtype=torch.float16).numpy(),
        "post_logits": torch.stack(post_logits).to(dtype=torch.float32).numpy(),
        "post_cosines": torch.stack(post_cosines).to(dtype=torch.float32).numpy(),
        "pre_logits": torch.stack(pre_logits).to(dtype=torch.float32).numpy(),
        "pre_cosines": torch.stack(pre_cosines).to(dtype=torch.float32).numpy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture pre/post-RoPE Q/K vectors for direction decomposition.")
    parser.add_argument(
        "--model_name_or_path",
        default="/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="8000,128000")
    parser.add_argument("--placements", default="middle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--global_max_position", type=int, default=130000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = parse_csv_ints(args.lengths)
    placements = parse_csv_strs(args.placements)
    unknown = sorted(set(placements) - set(causal.PLACEMENTS))
    if unknown:
        raise ValueError(f"unknown placements: {unknown}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rope_factor = base.rope_factor_for_length(
        args.global_max_position,
        args.original_max_position_embeddings,
    )
    model, tokenizer = base.load_model_and_tokenizer(args, args.global_max_position, rope_factor)
    config = {
        **vars(args),
        "resolved_lengths": lengths,
        "resolved_placements": placements,
        "rope_factor": rope_factor,
        "vector_roles": list(VECTOR_ROLES),
    }
    sweep.write_json_atomic(output_dir / "config.json", config)

    cases = [(placement, length) for placement in placements for length in lengths]
    for case_index, (placement, length) in enumerate(cases, start=1):
        stem = f"{placement}_length_{length}"
        vector_path = output_dir / f"{stem}.npz"
        metadata_path = output_dir / f"{stem}.json"
        if vector_path.exists() and metadata_path.exists():
            print(f"[{case_index}/{len(cases)}] {stem} already complete; skipping", flush=True)
            continue
        started = time.perf_counter()
        body = sweep.build_case(tokenizer, length, args.seed, "english_single_token", placement)
        spans = sweep.role_spans(tokenizer, body)
        if any(spans[role][0][1] - spans[role][0][0] != 1 for role in VECTOR_ROLES):
            raise RuntimeError(f"expected one-token role spans: {spans}")
        start_code, steps, gold_answer = causal.query_parameters(body, "full2")
        suffix = causal.build_suffix("legacy", start_code, steps, "full2")
        prompt_ids = body["body_ids"].view(-1).tolist() + base.token_ids(tokenizer, suffix)
        prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
        role_positions = {role: int(spans[role][0][0]) for role in VECTOR_ROLES}
        legacy_cache, captured_keys, key_position_vectors, prefill_seconds = prefill_and_capture_keys(
            model,
            prompt[:, :-1],
            args.prefill_chunk_size,
            role_positions,
        )
        query_cache = base.cache_from_legacy(legacy_cache)
        del legacy_cache
        query_output, captured_queries, query_position_vectors, query_seconds = query_and_capture_both(
            model,
            query_cache,
            prompt[:, -1:],
            int(prompt.shape[1]) - 1,
        )
        arrays = stack_probe_arrays(model, captured_keys, captured_queries)
        arrays["query_cos"] = query_position_vectors[0].to(dtype=torch.float32).numpy()
        arrays["query_sin"] = query_position_vectors[1].to(dtype=torch.float32).numpy()
        arrays["key_cos"] = torch.stack([key_position_vectors[role][0] for role in VECTOR_ROLES]).to(dtype=torch.float32).numpy()
        arrays["key_sin"] = torch.stack([key_position_vectors[role][1] for role in VECTOR_ROLES]).to(dtype=torch.float32).numpy()
        np.savez_compressed(vector_path, **arrays)
        answer = sweep.score_gold_from_query_output(
            model,
            tokenizer,
            query_output,
            int(prompt.shape[1]),
            gold_answer,
            completion_text=f" {gold_answer}",
        )
        hop2_index = VECTOR_ROLES.index("hop2_result")
        metadata = {
            "schema_version": 1,
            "model": "Qwen3-8B",
            "seed": args.seed,
            "placement": placement,
            "target_context_tokens": length,
            "prompt_tokens": int(prompt.shape[1]),
            "query_position": int(prompt.shape[1]) - 1,
            "role_positions": role_positions,
            "relative_distances": {
                role: int(prompt.shape[1]) - 1 - position
                for role, position in role_positions.items()
            },
            "gold_codes": body["gold_codes"],
            "gold_answer": gold_answer,
            "gold_ppl": answer["gold_ppl"],
            "rope_factor": rope_factor,
            "num_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "head_dim": int(getattr(model.config, "head_dim", arrays["q_pre"].shape[-1])),
            "mean_hop2_post_logit": float(arrays["post_logits"][hop2_index].mean()),
            "mean_hop2_post_cosine": float(arrays["post_cosines"][hop2_index].mean()),
            "mean_hop2_pre_logit": float(arrays["pre_logits"][hop2_index].mean()),
            "mean_hop2_pre_cosine": float(arrays["pre_cosines"][hop2_index].mean()),
            "timing": {
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "vector_file": vector_path.name,
        }
        sweep.write_json_atomic(metadata_path, metadata)
        print(
            f"[{case_index}/{len(cases)}] {stem} distance={metadata['relative_distances']['hop2_result']} "
            f"post_cos={metadata['mean_hop2_post_cosine']:.6f} pre_cos={metadata['mean_hop2_pre_cosine']:.6f} "
            f"ppl={metadata['gold_ppl']:.4f} seconds={metadata['timing']['total_seconds']:.1f}",
            flush=True,
        )
        del query_cache, query_output, captured_keys, captured_queries, arrays
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
