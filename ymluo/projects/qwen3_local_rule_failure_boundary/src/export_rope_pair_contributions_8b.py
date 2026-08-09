from __future__ import annotations

import argparse
import base64
import gc
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_attention_confidence_sweep_8b as sweep
import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base
import run_qk_direction_probe_8b as qk_probe


def encode_f16(values: np.ndarray) -> str:
    little_endian = np.asarray(values, dtype="<f2", order="C")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = int(values.shape[-1]) // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def invert_rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Invert the possibly YaRN-scaled split-half RoPE transform."""
    norm_squared = cos.square() + sin.square()
    return (values * cos - rotate_half(values) * sin) / norm_squared.clamp_min(1e-12)


def rope_cos_sin_on_device(
    inv_freq: torch.Tensor,
    attention_scaling: float,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE tables locally, bypassing Accelerate's module output-device hook."""
    positions = torch.arange(key_length, device=device, dtype=torch.float32)
    frequencies = positions[:, None] * inv_freq.to(device=device, dtype=torch.float32)[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().mul(float(attention_scaling)).to(dtype=dtype)
    sin = embedding.sin().mul(float(attention_scaling)).to(dtype=dtype)
    return cos, sin


def mean_by_fixed_bins(values: torch.Tensor, bin_size: int) -> torch.Tensor:
    """Average [batch, tokens, pairs] values into consecutive position bins."""
    batch, token_count, pair_count = values.shape
    full_bins = token_count // bin_size
    chunks: list[torch.Tensor] = []
    if full_bins:
        full = values[:, : full_bins * bin_size]
        chunks.append(full.reshape(batch, full_bins, bin_size, pair_count).float().mean(dim=2))
    if full_bins * bin_size < token_count:
        chunks.append(values[:, full_bins * bin_size :].float().mean(dim=1, keepdim=True))
    if not chunks:
        raise ValueError("cannot bin an empty contribution tensor")
    return torch.cat(chunks, dim=1)


def pair_contributions(
    query: torch.Tensor,
    keys: torch.Tensor,
    scaling: float,
    bin_size: int,
) -> torch.Tensor:
    """Return binned split-half pair logits with shape [heads, bins, head_dim/2]."""
    if query.ndim != 2 or keys.ndim != 3:
        raise ValueError(f"unexpected query/key shapes: {tuple(query.shape)}, {tuple(keys.shape)}")
    if query.shape[0] != keys.shape[0] or query.shape[-1] != keys.shape[-1]:
        raise ValueError(f"query/key head shape mismatch: {tuple(query.shape)}, {tuple(keys.shape)}")
    half = int(query.shape[-1]) // 2
    values = (
        query[:, None, :half] * keys[:, :, :half]
        + query[:, None, half:] * keys[:, :, half:]
    ) * float(scaling)
    return mean_by_fixed_bins(values, bin_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-RoPE-pair QK contributions for one Qwen3-8B long-context query."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--length", type=int, default=64000)
    parser.add_argument("--placement", choices=list(causal.PLACEMENTS), default="middle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bin_size", type=int, default=128)
    parser.add_argument("--head_batch_size", type=int, default=8)
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
    if args.bin_size <= 0 or args.head_batch_size <= 0:
        raise ValueError("bin_size and head_batch_size must be positive")
    output_dir = Path(args.output_dir)
    heads_dir = output_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rope_factor = base.rope_factor_for_length(
        args.global_max_position,
        args.original_max_position_embeddings,
    )
    model, tokenizer = base.load_model_and_tokenizer(args, args.global_max_position, rope_factor)
    body = sweep.build_case(tokenizer, args.length, args.seed, "english_single_token", args.placement)
    spans = sweep.role_spans(tokenizer, body)
    start_code, steps, gold_answer = causal.query_parameters(body, "full2")
    suffix = causal.build_suffix("legacy", start_code, steps, "full2")
    prompt_ids = body["body_ids"].view(-1).tolist() + base.token_ids(tokenizer, suffix)
    prompt = torch.tensor(prompt_ids, dtype=torch.long).view(1, -1)
    query_position = int(prompt.shape[1]) - 1

    print(
        f"loading complete; prefill tokens={query_position:,} query_position={query_position:,}",
        flush=True,
    )
    prefix_cache_dynamic, prefill_seconds = base.prefill_sequence(
        model,
        prompt[:, :-1],
        args.prefill_chunk_size,
    )
    prefix_cache = base.legacy_cache(prefix_cache_dynamic)
    del prefix_cache_dynamic
    query_cache = base.cache_from_legacy(prefix_cache)
    query_output, captured_queries, _, query_seconds = qk_probe.query_and_capture_both(
        model,
        query_cache,
        prompt[:, -1:],
        query_position,
    )
    del query_cache, query_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layers = list(model.model.layers)
    num_layers = len(layers)
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", captured_queries[0][0].shape[-1]))
    pair_count = head_dim // 2
    groups = num_query_heads // num_kv_heads
    key_length = int(prefix_cache[0][0].shape[-2])
    bin_count = math.ceil(key_length / args.bin_size)

    rotary = model.model.rotary_emb
    inv_freq_tensor = rotary.inv_freq.detach().float().cpu()
    inv_freq = inv_freq_tensor.numpy()
    attention_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    role_positions = {
        role: int(entries[0][0])
        for role, entries in spans.items()
        if entries
    }
    manifest = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "condition": "clean English single-token two-hop chain",
        "chain": body["gold_codes"],
        "gold_answer": gold_answer,
        "target_context_tokens": int(args.length),
        "prompt_tokens": int(prompt.shape[1]),
        "query_position": query_position,
        "key_length": key_length,
        "placement": args.placement,
        "role_positions": role_positions,
        "num_layers": num_layers,
        "num_attention_heads": num_query_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "pair_count": pair_count,
        "pair_layout": "split_half: pair i = dimensions (i, i + head_dim/2)",
        "bin_size": int(args.bin_size),
        "bin_count": bin_count,
        "bin_aggregation": "arithmetic mean of per-token pair logit contributions",
        "storage_dtype": "float16 little-endian base64 inside gzip JSON",
        "post_definition": "scaled dot contribution after RoPE; sum over pairs equals binned raw attention logit",
        "pre_definition": "scaled dot contribution after algebraically inverting RoPE from cached keys",
        "delta_definition": "post minus pre, computed client-side",
        "rope_theta": float(model.config.rope_theta),
        "rope_factor_label": float(rope_factor),
        "rope_scaling": getattr(model.config, "rope_scaling", None),
        "attention_scaling": attention_scaling,
        "inv_freq": inv_freq.tolist(),
        "files": {"head_pattern": "heads/layer_{layer:02d}_head_{head:02d}.json.gz"},
    }
    sweep.write_json_atomic(output_dir / "manifest.json", manifest)

    export_started = time.perf_counter()
    for layer_index, layer in enumerate(layers):
        layer_started = time.perf_counter()
        key_post = prefix_cache[layer_index][0][0]
        device = key_post.device
        if tuple(key_post.shape) != (num_kv_heads, key_length, head_dim):
            raise RuntimeError(
                f"layer {layer_index}: unexpected cached key shape {tuple(key_post.shape)}"
            )
        cos, sin = rope_cos_sin_on_device(
            inv_freq_tensor,
            attention_scaling,
            key_length,
            device,
            key_post.dtype,
        )
        key_pre = invert_rope(key_post, cos, sin)
        q_pre_all = captured_queries[layer_index][0].to(device=device, dtype=key_post.dtype)
        q_post_all = captured_queries[layer_index][1].to(device=device, dtype=key_post.dtype)
        scaling = float(getattr(layer.self_attn, "scaling", head_dim ** -0.5))

        layer_post = torch.empty((num_query_heads, bin_count, pair_count), dtype=torch.float16)
        layer_pre = torch.empty_like(layer_post)
        for head_start in range(0, num_query_heads, args.head_batch_size):
            head_end = min(num_query_heads, head_start + args.head_batch_size)
            head_indices = torch.arange(head_start, head_end, device=device)
            kv_indices = torch.div(head_indices, groups, rounding_mode="floor")
            selected_post = key_post.index_select(0, kv_indices)
            selected_pre = key_pre.index_select(0, kv_indices)
            post = pair_contributions(
                q_post_all[head_start:head_end], selected_post, scaling, args.bin_size
            )
            pre = pair_contributions(
                q_pre_all[head_start:head_end], selected_pre, scaling, args.bin_size
            )
            layer_post[head_start:head_end] = post.to(device="cpu", dtype=torch.float16)
            layer_pre[head_start:head_end] = pre.to(device="cpu", dtype=torch.float16)
            del selected_post, selected_pre, post, pre

        for head_index in range(num_query_heads):
            post_values = layer_post[head_index].numpy()
            pre_values = layer_pre[head_index].numpy()
            payload = {
                "schema_version": 1,
                "layer": layer_index,
                "head": head_index,
                "kv_head": head_index // groups,
                "shape": [bin_count, pair_count],
                "post_f16_b64": encode_f16(post_values),
                "pre_f16_b64": encode_f16(pre_values),
                "post_min": float(post_values.min()),
                "post_max": float(post_values.max()),
                "pre_min": float(pre_values.min()),
                "pre_max": float(pre_values.max()),
            }
            write_gzip_json(
                heads_dir / f"layer_{layer_index:02d}_head_{head_index:02d}.json.gz",
                payload,
            )

        del key_post, key_pre, cos, sin, q_pre_all, q_post_all, layer_post, layer_pre
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"layer {layer_index:02d}/{num_layers - 1:02d} exported in "
            f"{time.perf_counter() - layer_started:.1f}s",
            flush=True,
        )

    manifest["timing"] = {
        "prefill_seconds": prefill_seconds,
        "query_seconds": query_seconds,
        "export_seconds": time.perf_counter() - export_started,
        "total_seconds": time.perf_counter() - started,
    }
    sweep.write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / "done.txt").write_text("ok\n", encoding="utf-8")
    print(json.dumps(manifest["timing"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
