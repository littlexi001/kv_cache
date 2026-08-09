from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_controlled_public_kv_benchmark_v1 as lb
import run_head_top2_targeted_ppl_20260714 as reference_module
from hierarchical_pca_cache_20260715 import (
    HierarchicalPCACache,
    hierarchical_llama_attention_mode,
)
from run_critical_position_budget_probe_20260715 import run_one_token
from run_head_top2_targeted_ppl_20260714 import (
    head_qabs_sampled_mass_mode,
    install_llama_head_top_fraction_patch,
    load_model,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (
    TOPICS,
    encode_topic_stream,
    make_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", choices=sorted(TOPICS), default="religion")
    parser.add_argument("--history_tokens", type=int, default=4096)
    parser.add_argument("--decode_tokens", type=int, default=8)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--candidate_fraction", type=float, default=0.02)
    parser.add_argument("--exact_cache_fraction", type=float, default=0.032)
    parser.add_argument("--recent_fraction", type=float, default=0.0)
    parser.add_argument(
        "--directory_backend", choices=("sorted", "fused"), default="sorted"
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.decode_tokens < 2:
        raise ValueError("decode_tokens must be at least two")
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(
        SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
        )
    )
    stream = encode_topic_stream(
        tokenizer,
        TOPICS[args.topic],
        args.history_tokens + args.decode_tokens + 1,
        args.dataset_cache_dir,
        args.seed,
    )
    history_ids = stream[: args.history_tokens]
    decode_ids = stream[
        args.history_tokens : args.history_tokens + args.decode_tokens + 1
    ]
    bundle, _ = make_bundle(tokenizer, history_ids, page_tokens=16)

    reference_cache, reference_prefill_seconds = lb.prefill_prefix(
        model, bundle, input_device, args.prefill_chunk_tokens
    )
    physical_source_cache, physical_prefill_seconds = lb.prefill_prefix(
        model, bundle, input_device, args.prefill_chunk_tokens
    )
    physical_cache = HierarchicalPCACache.from_dynamic_cache(
        physical_source_cache,
        projection_dim=args.projection_dim,
        candidate_fraction=args.candidate_fraction,
        exact_cache_fraction=args.exact_cache_fraction,
        max_new_tokens=args.decode_tokens + 8,
        directory_backend=args.directory_backend,
        record_traces=True,
        recent_fraction=args.recent_fraction,
    )
    del physical_source_cache
    torch.cuda.empty_cache()
    set_attention_implementation(model, "eager")

    rows = []
    reference_nll = 0.0
    physical_nll = 0.0
    reference_seconds = 0.0
    physical_seconds = 0.0
    max_logit_error = 0.0
    candidate_overlap_rows = []
    index_parity_rows = []
    reference_mode = head_qabs_sampled_mass_mode(
        1.0e-6,
        (args.candidate_fraction,),
        0.0025,
        16,
        args.candidate_fraction,
        True,
        True,
        False,
        0,
        0.005,
        "pca_int8",
        args.projection_dim,
        "shared_mean",
    )
    with reference_mode, hierarchical_llama_attention_mode(model):
        for offset in range(args.decode_tokens):
            token_id = int(decode_ids[offset])
            label_id = int(decode_ids[offset + 1])
            position = args.history_tokens + offset
            reference_cache, reference_logits, elapsed, _ = run_one_token(
                model,
                token_id,
                reference_cache,
                position,
                input_device,
            )
            reference_seconds += elapsed
            physical_cache, physical_logits, elapsed, _ = run_one_token(
                model,
                token_id,
                physical_cache,
                position,
                input_device,
            )
            physical_seconds += elapsed
            reference_states = reference_module._ACTIVE_QABS_PCA_STATES or {}
            layer_overlaps = []
            layer_attention_errors = []
            for layer, physical_state in enumerate(physical_cache.states):
                reference_state = reference_states[layer]
                reference_candidates = reference_state["last_shared_candidates"]
                physical_candidates = physical_state.last_shared_candidates
                if physical_candidates is None:
                    raise RuntimeError("physical candidate trace was not recorded")
                overlap = (
                    reference_candidates.unsqueeze(-1)
                    == physical_candidates.to(torch.long).unsqueeze(-2)
                ).any(dim=-1).float().mean()
                layer_overlaps.append(float(overlap.item()))
                physical_attention_output = physical_state.last_attention_output
                if physical_attention_output is None:
                    raise RuntimeError("physical attention trace was not recorded")
                layer_attention_errors.append(
                    float(
                        (
                            reference_state["last_attention_output"].float()
                            - physical_attention_output.float()
                        ).abs().max().item()
                    )
                )
                if offset == 0:
                    index_parity_rows.append(
                        {
                            "layer": layer,
                            "basis_max_abs_error": float(
                                (
                                    reference_state["basis"].float()
                                    - physical_state.basis.float()
                                ).abs().max().item()
                            ),
                            "quantized_equal_fraction": float(
                                (
                                    reference_state["quantized"]
                                    [..., : args.history_tokens, :]
                                    == physical_state.quantized
                                    [..., : args.history_tokens, :]
                                ).float().mean().item()
                            ),
                            "scale_max_abs_error": float(
                                (
                                    reference_state["scales"]
                                    [..., : args.history_tokens, :].float()
                                    - physical_state.scales
                                    [..., : args.history_tokens, :].float()
                                ).abs().max().item()
                            ),
                        }
                    )
            candidate_overlap_rows.append(
                {
                    "offset": offset,
                    "mean": sum(layer_overlaps) / len(layer_overlaps),
                    "min": min(layer_overlaps),
                    "per_layer": layer_overlaps,
                    "attention_error_per_layer": layer_attention_errors,
                }
            )
            reference_log_probs = F.log_softmax(reference_logits[0].float(), dim=-1)
            physical_log_probs = F.log_softmax(physical_logits[0].float(), dim=-1)
            reference_token_nll = -float(reference_log_probs[label_id].item())
            physical_token_nll = -float(physical_log_probs[label_id].item())
            logit_error = float(
                (reference_logits.float() - physical_logits.float()).abs().max().item()
            )
            reference_nll += reference_token_nll
            physical_nll += physical_token_nll
            max_logit_error = max(max_logit_error, logit_error)
            rows.append(
                {
                    "offset": offset,
                    "token_id": token_id,
                    "label_id": label_id,
                    "reference_nll": reference_token_nll,
                    "physical_nll": physical_token_nll,
                    "nll_delta": physical_token_nll - reference_token_nll,
                    "max_logit_error": logit_error,
                }
            )

    count = len(rows)
    result = {
        "topic": args.topic,
        "history_tokens": args.history_tokens,
        "decode_tokens": count,
        "projection_dim": args.projection_dim,
        "candidate_fraction": args.candidate_fraction,
        "exact_cache_fraction": args.exact_cache_fraction,
        "recent_fraction": args.recent_fraction,
        "directory_backend": args.directory_backend,
        "reference_nll": reference_nll / count,
        "physical_nll": physical_nll / count,
        "physical_over_reference_ppl": math.exp(
            (physical_nll - reference_nll) / count
        ),
        "max_logit_error": max_logit_error,
        "reference_online_seconds": reference_seconds,
        "physical_online_seconds": physical_seconds,
        "reference_prefill_seconds": reference_prefill_seconds,
        "physical_prefill_seconds": physical_prefill_seconds,
        "original_full_gpu_kv_bytes": physical_cache.original_gpu_bytes,
        "hierarchical_persistent_gpu_bytes": physical_cache.persistent_gpu_bytes(),
        "hierarchical_gpu_over_full_kv": (
            physical_cache.persistent_gpu_bytes()
            / physical_cache.original_gpu_bytes
        ),
        "pinned_host_bytes": physical_cache.pinned_host_bytes(),
        "mean_cache_hit_rate": physical_cache.mean_cache_hit_rate(),
        "candidate_overlap": candidate_overlap_rows,
        "index_parity": index_parity_rows,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
