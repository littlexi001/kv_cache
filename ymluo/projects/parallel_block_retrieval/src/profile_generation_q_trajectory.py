from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_all_head_qk import AllHeadCapture
from profile_real_qk import read_jsonl, resolve_dtype, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bridge entities from real first-hop evidence and save the "
            "all-layer/all-head pre-RoPE Q state after every generated token."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--base_profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument(
        "--prompt_mode", choices=["atomic", "bridge_reasoned"], default="bridge_reasoned"
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def answer_hit(text: str, answer: str) -> bool:
    target = normalize_text(answer)
    return bool(target) and target in normalize_text(text)


def bridge_progress(text: str, target: str) -> float:
    generated = normalize_text(text).split()
    expected = normalize_text(target).split()
    if not expected:
        return 0.0
    best = 0
    for start in range(len(generated)):
        matched = 0
        while (
            matched < len(expected)
            and start + matched < len(generated)
            and generated[start + matched] == expected[matched]
        ):
            matched += 1
        best = max(best, matched)
    return best / len(expected)


def render_prompt(tokenizer: Any, memory: str, step: dict[str, Any], mode: str) -> list[int]:
    if mode == "atomic":
        instruction = (
            "Return only the shortest exact answer supported by the evidence. "
            "Ignore unrelated text."
        )
    else:
        instruction = (
            "Resolve the missing relation using only the evidence. Output exactly two lines:\n"
            "Relevant fact: <copy one shortest evidence sentence that answers the question>\n"
            "Bridge entity: <only the shortest missing entity or value>"
        )
    content = (
        f"Evidence:\n{memory}\n\n"
        f"Question: {step['step_question']}\n"
        f"{instruction}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_bridge(text: str, mode: str) -> str:
    if mode == "atomic":
        return text.strip()
    matches = re.findall(
        r"(?:^|\n)\s*Bridge\s+entity\s*:\s*(.+?)\s*(?=\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    return matches[-1].strip() if matches else ""


def paired_steps(rows: Sequence[dict[str, Any]], splits: set[str]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) not in splits or str(row.get("task_type")) != "multihop":
            continue
        grouped.setdefault(int(row["query_id"]), {})[int(row["step_index"])] = row
    pairs = []
    for query_id in sorted(grouped):
        steps = grouped[query_id]
        if 0 not in steps or 1 not in steps:
            continue
        first, second = steps[0], steps[1]
        if not first.get("target_block_ids") or not second.get("target_block_ids"):
            continue
        pairs.append({"query_id": query_id, "first": first, "second": second})
    return pairs


def project_current_q(
    capture: AllHeadCapture,
    *,
    layers: list[int],
    basis: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    repeat_groups = num_query_heads // num_kv_heads
    kv_map = torch.arange(num_query_heads, device=basis.device) // repeat_groups
    output = []
    for layer_index, layer in enumerate(layers):
        query = capture.q[layer][0, -1].float()
        query_basis = basis[layer_index].index_select(0, kv_map)
        projected = torch.einsum("hd,hdr->hr", query, query_basis)
        output.append(F.normalize(projected, dim=-1).to(dtype=torch.float16, device="cpu"))
    return torch.stack(output)


@torch.inference_mode()
def generate_one(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    capture: AllHeadCapture,
    prompt_ids: list[int],
    target: str,
    layers: list[int],
    basis: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    max_new_tokens: int,
    prompt_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    capture.configure(capture_q=True, capture_k=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    states = [
        project_current_q(
            capture,
            layers=layers,
            basis=basis,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        )
    ]
    metadata = [
        {
            "state_index": 0,
            "generated_tokens": 0,
            "last_token_id": int(prompt_ids[-1]),
            "generated_text": "",
            "bridge_progress": 0.0,
            "bridge_complete": False,
        }
    ]
    cache = outputs.past_key_values
    logits = outputs.logits[0, -1].float()
    generated: list[int] = []
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for generated_index in range(max_new_tokens):
        token_id = int(torch.argmax(logits).item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        capture.configure(capture_q=True, capture_k=False)
        position = len(prompt_ids) + generated_index
        outputs = model(
            input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
            past_key_values=cache,
            cache_position=torch.tensor([position], dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1].float()
        states.append(
            project_current_q(
                capture,
                layers=layers,
                basis=basis,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
            )
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        metadata.append(
            {
                "state_index": generated_index + 1,
                "generated_tokens": generated_index + 1,
                "last_token_id": token_id,
                "generated_text": text,
                "bridge_progress": bridge_progress(text, target),
                "bridge_complete": answer_hit(text, target),
            }
        )
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = extract_bridge(generated_text, prompt_mode)
    result = {
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "parsed_bridge": parsed,
        "raw_target_hit": answer_hit(generated_text, target),
        "parsed_target_hit": answer_hit(parsed, target),
        "first_complete_state": next(
            (int(row["state_index"]) for row in metadata if row["bridge_complete"]), None
        ),
    }
    return torch.stack(states), metadata, result


def merge_shards(
    shard_paths: Sequence[Path], pairs: Sequence[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    sample = payloads[0]
    max_states = max(
        int(item["svd_q"].shape[1]) for item in payloads
    )
    shape = (len(pairs), max_states, *sample["svd_q"].shape[2:])
    svd_q = torch.zeros(shape, dtype=torch.float16)
    mask = torch.zeros(len(pairs), max_states, dtype=torch.bool)
    trajectories: list[dict[str, Any] | None] = [None] * len(pairs)
    seen = set()
    for payload in payloads:
        for local_index, pair_index in enumerate(payload["pair_indices"]):
            pair_index = int(pair_index)
            if pair_index in seen:
                raise RuntimeError(f"duplicate trajectory index {pair_index}")
            seen.add(pair_index)
            count = int(payload["mask"][local_index].sum().item())
            svd_q[pair_index, :count].copy_(payload["svd_q"][local_index, :count])
            mask[pair_index, :count] = True
            trajectories[pair_index] = payload["trajectories"][local_index]
    if seen != set(range(len(pairs))) or any(row is None for row in trajectories):
        raise RuntimeError("trajectory shards are incomplete")
    torch.save(
        {
            "svd_q": svd_q,
            "mask": mask,
            "trajectories": trajectories,
            "layers": sample["layers"],
            "num_query_heads": sample["num_query_heads"],
            "num_kv_heads": sample["num_kv_heads"],
            "profile_space": "pre_rope_generation_q_projected_to_frozen_k_svd",
            "normalized": True,
        },
        output_path,
    )
    return {
        "trajectories": len(pairs),
        "states": int(mask.sum().item()),
        "mean_states": float(mask.sum(dim=1).float().mean().item()),
        "bridge_raw_hit_rate": float(
            np.mean([bool(row["generation"]["raw_target_hit"]) for row in trajectories])
        ),
        "bridge_parsed_hit_rate": float(
            np.mean([bool(row["generation"]["parsed_target_hit"]) for row in trajectories])
        ),
    }


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    pairs = paired_steps(read_jsonl(Path(args.step_queries_path)), splits)
    if args.max_queries > 0:
        pairs = pairs[: args.max_queries]
    if not pairs:
        raise ValueError("no complete two-step trajectories matched the requested splits")

    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    basis_payload = torch.load(
        Path(args.base_profile_dir) / "basis.pt", map_location="cpu", weights_only=False
    )
    layers = [int(item) for item in basis_payload["layers"]]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    capture = AllHeadCapture(model, layers)
    basis = basis_payload["basis"].to(device=device, dtype=torch.float32)

    pair_indices = list(range(rank, len(pairs), world_size))
    local_states: list[torch.Tensor] = []
    local_rows = []
    started = time.perf_counter()
    for local_offset, pair_index in enumerate(pair_indices):
        pair = pairs[pair_index]
        first, second = pair["first"], pair["second"]
        evidence_ids = [int(item) for item in first["target_block_ids"]]
        memory = "\n".join(
            tokenizer.decode(blocks[block_id].tolist(), skip_special_tokens=True)
            for block_id in evidence_ids
        )
        prompt_ids = render_prompt(tokenizer, memory, first, args.prompt_mode)
        states, state_metadata, generation = generate_one(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            prompt_ids=prompt_ids,
            target=str(first["target_output"]),
            layers=layers,
            basis=basis,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
            device=device,
        )
        local_states.append(states)
        local_rows.append(
            {
                "query_id": int(pair["query_id"]),
                "split": str(first["split"]),
                "question": str(first["question"]),
                "step_question": str(first["step_question"]),
                "bridge_target": str(first["target_output"]),
                "hop1_gold_block_ids": evidence_ids,
                "hop2_gold_block_ids": [int(item) for item in second["target_block_ids"]],
                "state_metadata": state_metadata,
                "generation": generation,
            }
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "trajectory": local_offset + 1,
                    "local_trajectories": len(pair_indices),
                    "query_id": int(pair["query_id"]),
                    "states": int(states.shape[0]),
                    "bridge_hit": bool(generation["raw_target_hit"]),
                }
            ),
            flush=True,
        )

    local_max = max((int(item.shape[0]) for item in local_states), default=1)
    if world_size > 1:
        maximum = torch.tensor(local_max, dtype=torch.int64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        max_states = int(maximum.item())
    else:
        max_states = local_max
    shape = (len(local_states), max_states, len(layers), num_query_heads, int(basis.shape[-1]))
    local_q = torch.zeros(shape, dtype=torch.float16)
    local_mask = torch.zeros(len(local_states), max_states, dtype=torch.bool)
    for index, states in enumerate(local_states):
        local_q[index, : states.shape[0]].copy_(states)
        local_mask[index, : states.shape[0]] = True
    shard_path = output_dir / f"trajectory_q_rank{rank:03d}.pt"
    torch.save(
        {
            "svd_q": local_q,
            "mask": local_mask,
            "pair_indices": pair_indices,
            "trajectories": local_rows,
            "layers": layers,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
        },
        shard_path,
    )
    capture.close()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        profile_path = output_dir / "trajectory_q_profiles.pt"
        summary = merge_shards(
            [output_dir / f"trajectory_q_rank{item:03d}.pt" for item in range(world_size)],
            pairs,
            profile_path,
        )
        summary.update(
            {
                "source": "real generated bridge trajectory with oracle first-hop evidence",
                "contains_synthetic_vectors": False,
                "contains_synthetic_text": False,
                "selection_uses_gold_first_hop_evidence": True,
                "selection_uses_gold_second_hop_evidence": False,
                "corpus_dir": args.corpus_dir,
                "step_queries_path": args.step_queries_path,
                "base_profile_dir": args.base_profile_dir,
                "profile_path": str(profile_path),
                "prompt_mode": args.prompt_mode,
                "max_new_tokens": args.max_new_tokens,
                "world_size": world_size,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
