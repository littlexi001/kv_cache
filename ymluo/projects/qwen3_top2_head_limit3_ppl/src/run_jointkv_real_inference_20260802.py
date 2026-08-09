#!/usr/bin/env python
"""Run packed JointKV inside real Hugging Face autoregressive decode."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

import jointkv_global_threshold_cuda_20260802 as global_cuda
import qabs_cuda_kernels as sparse_cuda
from jointkv_real_index_20260802 import (
    BASE_BITS,
    BASE_OFFSET,
    JOINT_OFFSET,
    RESIDUAL_BITS,
    RESIDUAL_OFFSET,
    JointKVRealIndex,
    build_real_index,
)


@dataclass
class TimedDecode:
    losses: list[float]
    logits: list[torch.Tensor]
    prefill_gpu_ms: float
    prefill_wall_ms: float
    decode_gpu_ms: list[float]
    decode_wall_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codebook_cache", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--history_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=16)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--overfetch", type=float, default=2.0)
    parser.add_argument("--capacity_multiplier", type=float, default=2.75)
    parser.add_argument("--sparse_split_count", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference_cache", type=Path)
    parser.add_argument("--reference_only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_count(tokens: int) -> int:
    return min(tokens, 1280, max(256, math.ceil(0.06 * tokens)))


def encode_repeated(
    tokenizer: Any,
    text_path: Path,
    needed: int,
) -> torch.Tensor:
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() < needed:
        ids = ids.repeat(math.ceil(needed / max(1, ids.numel())))
    return ids[:needed].unsqueeze(0).contiguous()


def timed_cuda_call(function: Callable[[], Any]) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    stop.record()
    torch.cuda.synchronize()
    return output, float(start.elapsed_time(stop)), (time.perf_counter() - start_wall) * 1e3


def cache_layer_tensors(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        item = cache.layers[layer]
        return item.keys, item.values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    legacy = cache.to_legacy_cache()
    return legacy[layer][0], legacy[layer][1]


@torch.no_grad()
def run_dense_decode(
    model: Any,
    input_ids: torch.Tensor,
    history_tokens: int,
    eval_tokens: int,
) -> tuple[TimedDecode, Any]:
    prefill, prefill_gpu_ms, prefill_wall_ms = timed_cuda_call(
        lambda: model(
            input_ids=input_ids[:, :history_tokens],
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    )
    cache = prefill.past_key_values
    del prefill
    losses: list[float] = []
    logits: list[torch.Tensor] = []
    decode_gpu_ms: list[float] = []
    decode_wall_ms: list[float] = []
    for offset in range(eval_tokens):
        current = input_ids[:, history_tokens + offset : history_tokens + offset + 1]
        target = input_ids[:, history_tokens + offset + 1]
        output, gpu_ms, wall_ms = timed_cuda_call(
            lambda current=current, cache=cache: model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        )
        cache = output.past_key_values
        current_logits = output.logits[:, -1].float()
        losses.append(float(F.cross_entropy(current_logits, target)))
        logits.append(current_logits.cpu())
        decode_gpu_ms.append(gpu_ms)
        decode_wall_ms.append(wall_ms)
        del output
    return (
        TimedDecode(
            losses=losses,
            logits=logits,
            prefill_gpu_ms=prefill_gpu_ms,
            prefill_wall_ms=prefill_wall_ms,
            decode_gpu_ms=decode_gpu_ms,
            decode_wall_ms=decode_wall_ms,
        ),
        cache,
    )


@torch.no_grad()
def build_all_indexes(
    cache: Any,
    codebooks: list[list[dict[str, Any]]],
) -> tuple[list[JointKVRealIndex], float, float]:
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    indexes = []
    for layer, layer_codebooks in enumerate(codebooks):
        key, value = cache_layer_tensors(cache, layer)
        indexes.append(
            build_real_index(
                key,
                value,
                layer_codebooks,
                risk_mode="qk_risk",
                risk_lambda=0.0,
            )
        )
    stop.record()
    torch.cuda.synchronize()
    return (
        indexes,
        float(start.elapsed_time(stop)),
        (time.perf_counter() - start_wall) * 1e3,
    )


def install_jointkv_attention(
    model: Any,
    indexes: list[JointKVRealIndex],
    *,
    eval_tokens: int,
    sample_count: int,
    overfetch: float,
    capacity_multiplier: float,
    split_count: int,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    originals: list[tuple[Any, Any]] = []
    workspaces: list[dict[str, torch.Tensor]] = []
    query_workspaces: list[dict[str, torch.Tensor]] = []
    query_matrices: list[torch.Tensor] = []
    target_counts = []
    capacities = []
    query_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    dtype = next(model.parameters()).dtype
    for index in indexes:
        target = selected_count(index.token_count)
        capacity = min(
            index.token_count + eval_tokens,
            max(
                target + eval_tokens,
                math.ceil(target * capacity_multiplier) + eval_tokens,
            ),
        )
        query_placeholder = torch.empty(
            1,
            query_heads,
            index.query_matrix.shape[1],
            dtype=dtype,
            device=index.base_codes.device,
        )
        query_matrix = index.query_matrix.to(dtype).contiguous()
        query_workspace = global_cuda.allocate_query_workspace(
            query_placeholder, query_matrix
        )
        workspaces.append(
            global_cuda.allocate_workspace(
                query_workspace["packed_query"], capacity
            )
        )
        query_workspaces.append(query_workspace)
        query_matrices.append(query_matrix)
        target_counts.append(target)
        capacities.append(capacity)

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        original = attention.forward
        originals.append((attention, original))
        index = indexes[layer_index]
        workspace = workspaces[layer_index]
        query_workspace = query_workspaces[layer_index]
        query_matrix = query_matrices[layer_index]
        target = target_counts[layer_index]

        def sparse_forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            _original: Any = original,
            _index: JointKVRealIndex = index,
            _workspace: dict[str, torch.Tensor] = workspace,
            _query_workspace: dict[str, torch.Tensor] = query_workspace,
            _query_matrix: torch.Tensor = query_matrix,
            _target: int = target,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if hidden_states.shape[:2] != (1, 1):
                return _original(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            query = query_states[:, :, 0].contiguous()
            packed_query, query_lut = global_cuda.project_query_lut_out(
                query,
                _query_matrix,
                _query_workspace,
                base_offset=BASE_OFFSET,
                residual_offset=RESIDUAL_OFFSET,
                base_chunks=BASE_BITS // 8,
                residual_chunks=RESIDUAL_BITS // 8,
            )
            indices, counts, _, _ = global_cuda.sampled_threshold_compact_out(
                packed_query,
                query_lut,
                _index.base_codes,
                _index.residual_codes,
                _index.joint_ids,
                _workspace,
                sample_count=sample_count,
                selected_fraction=_target / _index.token_count,
                overfetch=overfetch,
                joint_offset=JOINT_OFFSET,
            )
            # Prefix positions [0,index.token_count) are retrieved.  Previous
            # decode tokens are exact; the current final token is added by the
            # self-attention CUDA kernel below.
            global_cuda.append_contiguous_suffix(
                indices,
                counts,
                _index.token_count,
                key_states.shape[-2] - 1,
            )
            attention_output = sparse_cuda.final_attention_ragged_self_split(
                query,
                key_states,
                value_states,
                indices,
                counts,
                self.scaling,
                split_count,
            )
            attention_output = (
                attention_output.unsqueeze(2)
                .transpose(1, 2)
                .reshape(*input_shape, -1)
                .contiguous()
            )
            return self.o_proj(attention_output), None

        attention.forward = types.MethodType(sparse_forward, attention)

    return originals, {
        "target_candidates": target_counts[0],
        "capacity": capacities[0],
        "indexed_tokens": indexes[0].token_count,
        "target_fraction": target_counts[0] / indexes[0].token_count,
        "_workspaces": workspaces,
    }


def restore_attention(originals: list[tuple[Any, Any]]) -> None:
    for attention, forward in originals:
        attention.forward = forward


def quality_metrics(full: TimedDecode, sparse: TimedDecode) -> dict[str, float]:
    full_nll = statistics.mean(full.losses)
    sparse_nll = statistics.mean(sparse.losses)
    agreements = []
    kls = []
    for full_logits, sparse_logits in zip(full.logits, sparse.logits, strict=True):
        agreements.append(float(full_logits.argmax(-1).eq(sparse_logits.argmax(-1))[0]))
        probability = torch.softmax(full_logits, dim=-1)
        kls.append(
            float(
                (
                    probability
                    * (
                        torch.log_softmax(full_logits, dim=-1)
                        - torch.log_softmax(sparse_logits, dim=-1)
                    )
                ).sum()
            )
        )
    return {
        "full_nll": full_nll,
        "sparse_nll": sparse_nll,
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "quality_ratio": math.exp(full_nll - sparse_nll),
        "top1_agreement": statistics.mean(agreements),
        "full_to_sparse_kl": statistics.mean(kls),
    }


def timing_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def save_reference(
    path: Path,
    result: TimedDecode,
    contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": contract,
            "losses": result.losses,
            "logits": result.logits,
            "prefill_gpu_ms": result.prefill_gpu_ms,
            "prefill_wall_ms": result.prefill_wall_ms,
            "decode_gpu_ms": result.decode_gpu_ms,
            "decode_wall_ms": result.decode_wall_ms,
        },
        path,
    )


def load_reference(path: Path, expected_contract: dict[str, Any]) -> TimedDecode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != expected_contract:
        raise ValueError("saved Full reference does not match this run")
    payload = dict(payload)
    del payload["contract"]
    return TimedDecode(**payload)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.history_tokens < 256 or args.eval_tokens < 2:
        raise ValueError("use at least 256 history tokens and two evaluation tokens")
    torch.set_num_threads(8)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    needed = args.history_tokens + args.eval_tokens + 1
    input_ids = encode_repeated(tokenizer, args.text, needed).to(args.device)
    cached = torch.load(args.codebook_cache, map_location="cpu", weights_only=False)
    codebooks = cached["codebooks"]

    # Load both extensions before timing any decode call.
    global_cuda.load_extension()
    sparse_cuda._load_extension()
    with torch.no_grad():
        model(input_ids=input_ids[:, :32], use_cache=False, return_dict=True)
    torch.cuda.synchronize()

    reference_contract = {
        "model": str(args.model),
        "text": str(args.text),
        "history_tokens": args.history_tokens,
        "eval_tokens": args.eval_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    if args.reference_cache is not None and args.reference_cache.exists():
        full = load_reference(args.reference_cache, reference_contract)
    else:
        full, full_cache = run_dense_decode(
            model, input_ids, args.history_tokens, args.eval_tokens
        )
        if args.reference_cache is not None:
            save_reference(args.reference_cache, full, reference_contract)
        del full_cache
        gc.collect()
        torch.cuda.empty_cache()
    if args.reference_only:
        if args.reference_cache is None:
            raise ValueError("--reference_only requires --reference_cache")
        payload = {
            "schema": "jointkv-real-hf-full-reference-v1",
            "setup": {
                "model": str(args.model),
                "text": str(args.text),
                "history_tokens": args.history_tokens,
                "eval_tokens": args.eval_tokens,
                "reference_cache": str(args.reference_cache),
            },
            "quality": {
                "full_nll": statistics.mean(full.losses),
                "full_ppl": math.exp(statistics.mean(full.losses)),
            },
            "speed": {
                "full_prefill_gpu_ms": full.prefill_gpu_ms,
                "full_prefill_wall_ms": full.prefill_wall_ms,
                "full_decode_gpu_ms_per_token": timing_summary(
                    full.decode_gpu_ms
                ),
                "full_decode_wall_ms_per_token": timing_summary(
                    full.decode_wall_ms
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    sparse_prefill, sparse_prefill_gpu_ms, sparse_prefill_wall_ms = timed_cuda_call(
        lambda: model(
            input_ids=input_ids[:, : args.history_tokens],
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    )
    sparse_cache = sparse_prefill.past_key_values
    del sparse_prefill
    indexes, index_gpu_ms, index_wall_ms = build_all_indexes(sparse_cache, codebooks)
    originals, selector_contract = install_jointkv_attention(
        model,
        indexes,
        eval_tokens=args.eval_tokens,
        sample_count=args.sample_count,
        overfetch=args.overfetch,
        capacity_multiplier=args.capacity_multiplier,
        split_count=args.sparse_split_count,
    )
    sparse_losses: list[float] = []
    sparse_logits: list[torch.Tensor] = []
    sparse_gpu_ms: list[float] = []
    sparse_wall_ms: list[float] = []
    try:
        for offset in range(args.eval_tokens):
            current = input_ids[
                :, args.history_tokens + offset : args.history_tokens + offset + 1
            ]
            target = input_ids[:, args.history_tokens + offset + 1]
            output, gpu_ms, wall_ms = timed_cuda_call(
                lambda current=current, cache=sparse_cache: model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            )
            sparse_cache = output.past_key_values
            current_logits = output.logits[:, -1].float()
            sparse_losses.append(float(F.cross_entropy(current_logits, target)))
            sparse_logits.append(current_logits.cpu())
            sparse_gpu_ms.append(gpu_ms)
            sparse_wall_ms.append(wall_ms)
            del output
    finally:
        restore_attention(originals)
    sparse = TimedDecode(
        losses=sparse_losses,
        logits=sparse_logits,
        prefill_gpu_ms=sparse_prefill_gpu_ms,
        prefill_wall_ms=sparse_prefill_wall_ms,
        decode_gpu_ms=sparse_gpu_ms,
        decode_wall_ms=sparse_wall_ms,
    )
    selector_workspaces = selector_contract.pop("_workspaces")
    last_counts = torch.cat(
        [workspace["counts"].flatten() for workspace in selector_workspaces]
    ).float()
    last_overflow = torch.cat(
        [workspace["overflow"].flatten() for workspace in selector_workspaces]
    )
    selector_contract.update(
        {
            "last_decode_candidate_count_mean": float(last_counts.mean()),
            "last_decode_candidate_count_min": int(last_counts.min()),
            "last_decode_candidate_count_max": int(last_counts.max()),
            "last_decode_candidate_fraction_mean": float(
                last_counts.mean() / indexes[0].token_count
            ),
            "last_decode_exact_suffix_tokens": args.eval_tokens - 1,
            "last_decode_retrieved_prefix_candidate_mean": float(
                last_counts.mean() - (args.eval_tokens - 1)
            ),
            "last_decode_retrieved_prefix_fraction_mean": float(
                (last_counts.mean() - (args.eval_tokens - 1))
                / indexes[0].token_count
            ),
            "last_decode_overflow_heads": int(last_overflow.sum()),
        }
    )
    full_decode_gpu = statistics.mean(full.decode_gpu_ms)
    sparse_decode_gpu = statistics.mean(sparse.decode_gpu_ms)
    full_decode_wall = statistics.mean(full.decode_wall_ms)
    sparse_decode_wall = statistics.mean(sparse.decode_wall_ms)
    physical_index_bytes = sum(
        tensor.numel() * tensor.element_size()
        for index in indexes
        for tensor in (
            index.base_codes,
            index.residual_codes,
            index.joint_ids,
            index.risk_codes,
        )
    )
    payload = {
        "schema": "jointkv-real-hf-inference-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup": {
            "model": str(args.model),
            "text": str(args.text),
            "history_tokens": args.history_tokens,
            "eval_tokens": args.eval_tokens,
            "layers": len(indexes),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "sample_count": args.sample_count,
            "overfetch": args.overfetch,
            "capacity_multiplier": args.capacity_multiplier,
            "tail_correction": False,
            "full_fallback": False,
            "dense_prefill": True,
            "exact_generated_suffix": True,
            "actual_hf_model_forward": True,
            **selector_contract,
        },
        "quality": quality_metrics(full, sparse),
        "speed": {
            "full_prefill_gpu_ms": full.prefill_gpu_ms,
            "sparse_prefill_gpu_ms": sparse.prefill_gpu_ms,
            "index_build_gpu_ms": index_gpu_ms,
            "full_prefill_wall_ms": full.prefill_wall_ms,
            "sparse_prefill_wall_ms": sparse.prefill_wall_ms,
            "index_build_wall_ms": index_wall_ms,
            "full_decode_gpu_ms_per_token": timing_summary(full.decode_gpu_ms),
            "sparse_decode_gpu_ms_per_token": timing_summary(sparse.decode_gpu_ms),
            "full_decode_wall_ms_per_token": timing_summary(full.decode_wall_ms),
            "sparse_decode_wall_ms_per_token": timing_summary(sparse.decode_wall_ms),
            "steady_decode_gpu_speedup": full_decode_gpu / sparse_decode_gpu,
            "steady_decode_wall_speedup": full_decode_wall / sparse_decode_wall,
            "full_online_gpu_ms": full.prefill_gpu_ms + sum(full.decode_gpu_ms),
            "sparse_online_gpu_ms": (
                sparse.prefill_gpu_ms + index_gpu_ms + sum(sparse.decode_gpu_ms)
            ),
            "online_gpu_speedup_including_index": (
                (full.prefill_gpu_ms + sum(full.decode_gpu_ms))
                / (sparse.prefill_gpu_ms + index_gpu_ms + sum(sparse.decode_gpu_ms))
            ),
        },
        "index": {
            "physical_primary_index_bytes_all_layers": physical_index_bytes,
            "physical_primary_index_mib_all_layers": physical_index_bytes / 2**20,
            "physical_bytes_per_token_kv_head": indexes[0].physical_bytes_per_token_head,
            "logical_bits_per_token_kv_head": indexes[0].logical_bits_per_token_head,
        },
        "token_losses": {
            "full": full.losses,
            "sparse": sparse.losses,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
