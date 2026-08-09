from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype, stack_queries
from run_real_qk_retrieval import (
    candidate_exact_scores,
    global_topk,
    load_index,
    score_colbert_blocks,
    setup_distributed,
)
from run_single_query_dynamic_kv_generation import (
    HOP1_PREFIX,
    advance_token,
    answer_hit,
    build_chat_prompt_parts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-query online 10M retrieval: distributed SVD32 coarse scan, "
            "raw-K rerank, and full-model KV recomputation for selected blocks."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_id", type=int, default=0)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--target_blocks", type=int, default=3)
    parser.add_argument("--retrieval_interval", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--dynamic_query_tokens", type=int, default=3)
    parser.add_argument("--anchor_initial_query", action="store_true")
    parser.add_argument("--coarse_reserve_blocks", type=int, default=0)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=0)
    parser.add_argument("--block_chunk", type=int, default=256)
    parser.add_argument("--hop1_block", type=int, default=20088)
    parser.add_argument("--include_hop2_probe", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def capture_query_ids(
    model: AutoModelForCausalLM,
    capture: QKCapture,
    pair_specs: Sequence[dict[str, int]],
    input_ids: Sequence[int],
    keep_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    capture.clear()
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        model.model(input_ids=ids, use_cache=False, return_dict=True)
    query = stack_queries(capture, list(pair_specs))[0]
    return query[-min(keep_tokens, query.shape[0]) :].float().contiguous()


def broadcast_query(
    query: torch.Tensor | None,
    *,
    max_tokens: int,
    profile_count: int,
    head_dim: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = torch.tensor(
        [0 if query is None else int(query.shape[0])], dtype=torch.long, device=device
    )
    dist.broadcast(length, src=0)
    count = int(length.item())
    if count <= 0 or count > max_tokens:
        raise ValueError(f"invalid broadcast query length: {count}")
    padded = torch.zeros(
        (1, max_tokens, profile_count, head_dim),
        dtype=torch.float16,
        device=device,
    )
    if rank == 0:
        padded[0, :count].copy_(query.to(device=device, dtype=torch.float16))
    dist.broadcast(padded, src=0)
    mask = torch.zeros((1, max_tokens), dtype=torch.bool, device=device)
    mask[:, :count] = True
    return padded, mask


def mask_scores_to_block_range(
    scores: torch.Tensor,
    block_ids: torch.Tensor,
    allowed_start: int,
    allowed_end: int,
) -> torch.Tensor:
    if allowed_start < 0:
        return scores
    if allowed_end <= allowed_start:
        raise ValueError("allowed block range must be non-empty")
    allowed = (block_ids >= allowed_start) & (block_ids < allowed_end)
    while allowed.ndim < scores.ndim:
        allowed = allowed.unsqueeze(0)
    if allowed.shape[-1] != scores.shape[-1]:
        raise ValueError("block IDs must align with the score tensor's last dimension")
    return scores.masked_fill(~allowed, -torch.inf)


def distributed_retrieve(
    *,
    query: torch.Tensor | None,
    basis: torch.Tensor,
    raw_keys: torch.Tensor,
    svd_keys: torch.Tensor,
    local_block_ids: torch.Tensor,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    gold_block_id: int,
    allowed_block_range: tuple[int, int] | None = None,
) -> tuple[list[int], dict[str, Any] | None]:
    max_query_tokens = (
        args.query_tokens + args.dynamic_query_tokens
        if args.anchor_initial_query
        else max(args.query_tokens, args.dynamic_query_tokens)
    )
    raw_query, query_mask = broadcast_query(
        query,
        max_tokens=max_query_tokens,
        profile_count=raw_keys.shape[2],
        head_dim=raw_keys.shape[3],
        rank=rank,
        device=device,
    )
    projected = torch.einsum(
        "bspd,pdr->bspr",
        raw_query.float(),
        basis[..., : args.svd_rank].float(),
    ).to(torch.float16)
    if world_size > 1:
        dist.barrier()
    started = time.perf_counter()
    local_scores = score_colbert_blocks(
        svd_keys[..., : args.svd_rank],
        projected,
        query_mask,
        query_batch=1,
        block_chunk=args.block_chunk,
        exclude_block_prefix_tokens=args.exclude_block_prefix_tokens,
    )
    allowed_range_tensor = torch.tensor(
        list(allowed_block_range) if rank == 0 and allowed_block_range else [-1, -1],
        dtype=torch.long,
        device=device,
    )
    dist.broadcast(allowed_range_tensor, src=0)
    allowed_start, allowed_end = [int(item) for item in allowed_range_tensor.tolist()]
    if allowed_start >= 0:
        local_scores = mask_scores_to_block_range(
            local_scores, local_block_ids, allowed_start, allowed_end
        )
    coarse_values, candidate_ids = global_topk(
        local_scores,
        local_block_ids,
        args.candidate_blocks,
        world_size,
    )
    exact_scores = candidate_exact_scores(
        raw_keys,
        raw_query,
        query_mask,
        candidate_ids,
        local_block_ids,
        args.exclude_block_prefix_tokens,
        1.0 / math.sqrt(raw_keys.shape[-1]),
        world_size,
    )
    if allowed_start >= 0:
        exact_scores = mask_scores_to_block_range(
            exact_scores, candidate_ids, allowed_start, allowed_end
        )
    adjusted = exact_scores.to(torch.float64) - candidate_ids.to(torch.float64) * 1.0e-12
    exact_ranked_tensor = candidate_ids[0][torch.argsort(adjusted[0], descending=True)]
    reserve_count = min(args.coarse_reserve_blocks, args.target_blocks)
    selected = [
        int(item)
        for item in candidate_ids[0, :reserve_count].tolist()
        if allowed_start < 0 or allowed_start <= int(item) < allowed_end
    ]
    for item in exact_ranked_tensor.tolist():
        block_id = int(item)
        if allowed_start >= 0 and not allowed_start <= block_id < allowed_end:
            continue
        if block_id not in selected:
            selected.append(block_id)
        if len(selected) >= args.target_blocks:
            break
    if world_size > 1:
        dist.barrier()
    elapsed = time.perf_counter() - started
    if rank != 0:
        return [int(item) for item in selected], None

    gold_id = gold_block_id
    coarse_pairs = [
        (int(block_id), float(score))
        for block_id, score in zip(
            candidate_ids[0].tolist(), coarse_values[0].tolist(), strict=True
        )
        if allowed_start < 0 or allowed_start <= int(block_id) < allowed_end
    ]
    coarse_list = [item[0] for item in coarse_pairs]
    coarse_score_list = [item[1] for item in coarse_pairs]
    exact_ranked = [
        int(item)
        for item in exact_ranked_tensor.tolist()
        if allowed_start < 0 or allowed_start <= int(item) < allowed_end
    ]
    event = {
        "query_vectors": int(query_mask.sum().item()),
        "selected_block_ids": [int(item) for item in selected],
        "coarse_reserve_blocks": reserve_count,
        "retrieval_seconds": elapsed,
        "hop1_block_in_candidates": args.hop1_block in coarse_list,
        "hop1_block_coarse_rank": (
            coarse_list.index(args.hop1_block) + 1 if args.hop1_block in coarse_list else None
        ),
        "hop1_block_exact_rank": (
            exact_ranked.index(args.hop1_block) + 1
            if args.hop1_block in exact_ranked
            else None
        ),
        "answer_block_in_candidates": gold_id in coarse_list,
        "answer_block_coarse_rank": (
            coarse_list.index(gold_id) + 1 if gold_id in coarse_list else None
        ),
        "answer_block_exact_rank": (
            exact_ranked.index(gold_id) + 1 if gold_id in exact_ranked else None
        ),
        "coarse_top10": coarse_list[:10],
        "coarse_top10_scores": coarse_score_list[:10],
        "coarse_candidate_ids": coarse_list,
        "exact_ranked_candidate_ids": [int(item) for item in exact_ranked],
        "allowed_block_range": (
            [allowed_start, allowed_end] if allowed_start >= 0 else None
        ),
    }
    return [int(item) for item in selected], event


def selected_memory_ids(
    blocks: np.ndarray,
    selected_ids: Sequence[int],
    separator_ids: Sequence[int],
) -> list[int]:
    output: list[int] = []
    for offset, block_id in enumerate(selected_ids):
        if offset:
            output.extend(separator_ids)
        output.extend(int(item) for item in blocks[int(block_id)].tolist())
    return output


@torch.inference_mode()
def rebuild_generation_cache(
    *,
    model: AutoModelForCausalLM,
    blocks: np.ndarray,
    selected_ids: Sequence[int],
    chat_prefix_ids: Sequence[int],
    chat_suffix_ids: Sequence[int],
    generated_ids: Sequence[int],
    separator_ids: Sequence[int],
    device: torch.device,
) -> tuple[Any, torch.Tensor, int]:
    memory = selected_memory_ids(blocks, selected_ids, separator_ids)
    input_ids = list(chat_prefix_ids) + memory + list(chat_suffix_ids) + list(generated_ids)
    outputs = model(
        input_ids=torch.tensor([input_ids], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, outputs.logits[0, -1].float(), len(input_ids)


def decode_snippet(tokenizer: Any, blocks: np.ndarray, block_id: int) -> str:
    text = tokenizer.decode(blocks[block_id].tolist(), skip_special_tokens=True)
    return " ".join(text.split())[:500]


def generate_mode(
    *,
    mode: str,
    initial_query: torch.Tensor | None,
    initial_generated_ids: list[int],
    anchor_query: torch.Tensor | None,
    model: AutoModelForCausalLM | None,
    capture: QKCapture | None,
    tokenizer: Any | None,
    blocks: np.ndarray,
    chat_prefix_ids: Sequence[int],
    chat_suffix_ids: Sequence[int],
    separator_ids: Sequence[int],
    pair_specs: Sequence[dict[str, int]],
    basis: torch.Tensor,
    raw_keys: torch.Tensor,
    svd_keys: torch.Tensor,
    local_block_ids: torch.Tensor,
    query: dict[str, Any],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    mode_started = time.perf_counter()
    gold_block_ids = [int(item) for item in query["gold_block_ids"]]
    if len(gold_block_ids) != 1:
        raise ValueError("single-query experiment requires exactly one gold block")
    gold_block_id = gold_block_ids[0]
    selected, event = distributed_retrieve(
        query=initial_query,
        basis=basis,
        raw_keys=raw_keys,
        svd_keys=svd_keys,
        local_block_ids=local_block_ids,
        args=args,
        rank=rank,
        world_size=world_size,
        device=device,
        gold_block_id=gold_block_id,
    )
    events = []
    if rank == 0:
        event.update({"refresh": 0, "generated_tokens_before_refresh": len(initial_generated_ids)})
        events.append(event)
        print(
            f"[{mode}] refresh=0 selected={event['selected_block_ids']} "
            f"hop1_exact={event['hop1_block_exact_rank']} "
            f"answer_exact={event['answer_block_exact_rank']} "
            f"retrieval={event['retrieval_seconds']:.3f}s",
            flush=True,
        )

    generated = list(initial_generated_ids)
    seed_token_count = len(initial_generated_ids)
    generation_target = seed_token_count + args.max_new_tokens
    recent_queries: list[torch.Tensor] = []
    first_answer_token = None
    if rank == 0:
        cache, logits, position = rebuild_generation_cache(
            model=model,
            blocks=blocks,
            selected_ids=selected,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
            generated_ids=generated,
            separator_ids=separator_ids,
            device=device,
        )

    refresh = 0
    while True:
        if rank == 0:
            tokens_this_interval = (
                generation_target - len(generated)
                if mode == "static"
                else min(args.retrieval_interval, generation_target - len(generated))
            )
            stop = tokens_this_interval <= 0
            for _ in range(max(tokens_this_interval, 0)):
                token_id = int(torch.argmax(logits).item())
                if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                    stop = True
                    break
                generated.append(token_id)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if first_answer_token is None and answer_hit(text, query["answers"]):
                    first_answer_token = len(generated) - seed_token_count
                capture.clear()
                cache, logits = advance_token(model, token_id, cache, position, device)
                position += 1
                recent_queries.append(stack_queries(capture, list(pair_specs))[0, 0].float())
                recent_queries = recent_queries[-args.dynamic_query_tokens :]
            if mode == "static":
                stop = True
        else:
            stop = False
        stop_tensor = torch.tensor([int(stop)], dtype=torch.long, device=device)
        dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            break

        refresh += 1
        if rank == 0:
            recent_query = torch.stack(recent_queries, dim=0)
            next_query = (
                torch.cat([anchor_query, recent_query], dim=0)
                if args.anchor_initial_query
                else recent_query
            )
        else:
            next_query = None
        selected, event = distributed_retrieve(
            query=next_query,
            basis=basis,
            raw_keys=raw_keys,
            svd_keys=svd_keys,
            local_block_ids=local_block_ids,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            gold_block_id=gold_block_id,
        )
        if rank == 0:
            event.update(
                {"refresh": refresh, "generated_tokens_before_refresh": len(generated)}
            )
            events.append(event)
            print(
                f"[{mode}] refresh={refresh} generated={len(generated) - seed_token_count} "
                f"selected={event['selected_block_ids']} "
                f"hop1_exact={event['hop1_block_exact_rank']} "
                f"answer_exact={event['answer_block_exact_rank']} "
                f"retrieval={event['retrieval_seconds']:.3f}s",
                flush=True,
            )
            cache, logits, position = rebuild_generation_cache(
                model=model,
                blocks=blocks,
                selected_ids=selected,
                chat_prefix_ids=chat_prefix_ids,
                chat_suffix_ids=chat_suffix_ids,
                generated_ids=generated,
                separator_ids=separator_ids,
                device=device,
            )

    if rank != 0:
        return None
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    selected_union = sorted(
        {block_id for item in events for block_id in item["selected_block_ids"]}
    )
    snippets = {
        str(block_id): decode_snippet(tokenizer, blocks, block_id)
        for block_id in selected_union
    }
    return {
        "mode": mode,
        "query_id": int(query["query_id"]),
        "question": query["question"],
        "answers": query["answers"],
        "generated_text": generated_text,
        "generated_tokens": len(generated) - seed_token_count,
        "seed_tokens": seed_token_count,
        "answer_hit_128": answer_hit(generated_text, query["answers"]),
        "first_answer_generation_token": first_answer_token,
        "retrieval_refreshes": len(events),
        "selected_unique_blocks": selected_union,
        "selected_unique_block_count": len(selected_union),
        "answer_block_id": gold_block_id,
        "answer_block_ever_selected": gold_block_id in selected_union,
        "hop1_block_ever_selected": args.hop1_block in selected_union,
        "events": events,
        "block_snippets": snippets,
        "total_seconds": time.perf_counter() - mode_started,
    }


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    if world_size <= 1:
        raise ValueError("This experiment requires distributed index shards")
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if profile_summary["profile_space"] != "pre_rope_record_qk":
        raise ValueError("dynamic global retrieval requires a pre_rope_record_qk index")
    if args.svd_rank > int(profile_summary["svd_rank"]):
        raise ValueError("requested SVD rank exceeds the stored index")
    raw_keys, svd_keys, local_block_ids, _ = load_index(
        profile_dir, profile_summary, rank, world_size, device
    )
    basis_payload = torch.load(
        profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    basis = basis_payload["basis"].to(device=device, dtype=torch.float16)
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    query = next(item for item in queries if int(item["query_id"]) == args.query_id)

    model = None
    tokenizer = None
    capture = None
    chat_prefix_ids: list[int] = []
    chat_suffix_ids: list[int] = []
    separator_ids: list[int] = []
    question_query = None
    seeded_query = None
    hop2_probe_query = None
    seed_ids: list[int] = []
    hop2_probe_ids: list[int] = []
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=resolve_dtype(args.dtype),
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))
        question_ids = tokenizer(str(query["question"]), add_special_tokens=False)["input_ids"]
        question_query = capture_query_ids(
            model,
            capture,
            pair_specs,
            question_ids,
            args.query_tokens,
            device,
        )
        chat_prefix_ids, chat_suffix_ids = build_chat_prompt_parts(
            tokenizer, str(query["question"]), "reasoning_v2"
        )
        separator_ids = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        seed_ids = tokenizer(HOP1_PREFIX, add_special_tokens=False)["input_ids"]
        seeded_state_ids = chat_prefix_ids + chat_suffix_ids + seed_ids
        seeded_query = capture_query_ids(
            model,
            capture,
            pair_specs,
            seeded_state_ids,
            args.dynamic_query_tokens,
            device,
        )
        hop2_probe_text = HOP1_PREFIX + "2. Marion Byron was born in"
        hop2_probe_ids = tokenizer(hop2_probe_text, add_special_tokens=False)["input_ids"]
        hop2_probe_query = capture_query_ids(
            model,
            capture,
            pair_specs,
            chat_prefix_ids + chat_suffix_ids + hop2_probe_ids,
            args.dynamic_query_tokens,
            device,
        )

    results = []
    mode_specs = [
        ("static", question_query, []),
        ("dynamic_free", question_query, []),
        ("dynamic_seeded_hop1", seeded_query, seed_ids if rank == 0 else []),
    ]
    if args.include_hop2_probe:
        mode_specs.append(
            (
                "dynamic_seeded_hop2_probe",
                hop2_probe_query,
                hop2_probe_ids if rank == 0 else [],
            )
        )
    for mode, initial_query, initial_generated in mode_specs:
        result = generate_mode(
            mode="static" if mode == "static" else "dynamic",
            initial_query=initial_query,
            initial_generated_ids=initial_generated,
            anchor_query=question_query,
            model=model,
            capture=capture,
            tokenizer=tokenizer,
            blocks=blocks,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
            separator_ids=separator_ids,
            pair_specs=pair_specs,
            basis=basis,
            raw_keys=raw_keys,
            svd_keys=svd_keys,
            local_block_ids=local_block_ids,
            query=query,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if rank == 0:
            result["mode"] = mode
            results.append(result)
            (output_dir / "result.partial.json").write_text(
                json.dumps({"completed_modes": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        dist.barrier()

    if rank == 0:
        payload = {
            "source": "10M distributed SVD32 coarse retrieval plus raw-K rerank",
            "contains_synthetic_vectors": False,
            "num_blocks": int(profile_summary["num_blocks"]),
            "num_tokens": int(profile_summary["num_tokens"]),
            "world_size": world_size,
            "svd_rank": args.svd_rank,
            "candidate_blocks": args.candidate_blocks,
            "target_blocks": args.target_blocks,
            "anchor_initial_query": args.anchor_initial_query,
            "coarse_reserve_blocks": args.coarse_reserve_blocks,
            "retrieval_interval": args.retrieval_interval,
            "kv_materialization": (
                "Selected block token IDs are re-prefilled to compute full-model K/V; "
                "V is not loaded from a persistent 10M KV store."
            ),
            "results": results,
        }
        (output_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        capture.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
