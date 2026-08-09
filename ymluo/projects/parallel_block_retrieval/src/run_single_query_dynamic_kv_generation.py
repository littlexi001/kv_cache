from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


HOP1_PREFIX = " Lou Breslow's wife was Marion Byron.\n"
HOP2_QUERY_PREFIX = (
    " Lou Breslow's wife was Marion Byron.\nMarion Byron was born in"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one LongBench answer while refreshing a small set of real KV "
            "blocks from the current Q state. This is an exact sparse-attention "
            "simulation, not a speed kernel."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--query_id", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["dynamic", "full_source", "question_only"], default="dynamic"
    )
    parser.add_argument("--retrieval_interval", type=int, default=2)
    parser.add_argument("--blocks_per_refresh", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--prefill_chunk", type=int, default=1024)
    parser.add_argument(
        "--prompt_style", choices=["legacy", "reasoning_v2", "longbench"], default="legacy"
    )
    parser.add_argument("--seed_hop1", action="store_true")
    parser.add_argument("--seed_hop2_query", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_hit(text: str, answers: Sequence[str]) -> bool:
    normalized = normalize_answer(text)
    return any(
        normalize_answer(answer) in normalized
        for answer in answers
        if normalize_answer(answer)
    )


def first_answer_end(text: str, answers: Sequence[str]) -> int | None:
    normalized_tokens = normalize_answer(text).split()
    for end in range(1, len(normalized_tokens) + 1):
        prefix = " ".join(normalized_tokens[:end])
        if answer_hit(prefix, answers):
            return end
    return None


def select_shared_qk_blocks(
    logits: torch.Tensor,
    context_length: int,
    block_tokens: int,
    budget_blocks: int,
) -> list[int]:
    if context_length <= 0:
        return []
    context_attention = torch.softmax(logits[:, :context_length], dim=-1)
    block_count = math.ceil(context_length / block_tokens)
    scores = torch.zeros(block_count, dtype=torch.float32, device=logits.device)
    for block_id in range(block_count):
        start = block_id * block_tokens
        end = min(start + block_tokens, context_length)
        scores[block_id] = context_attention[:, start:end].sum()
    ids = torch.arange(block_count, device=logits.device)
    # Stable tie breaking is only relevant for degenerate all-zero test inputs.
    order = sorted(
        range(block_count), key=lambda block_id: (-float(scores[block_id]), int(ids[block_id]))
    )
    return order[: min(budget_blocks, block_count)]


def cache_layer_tensors(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    if hasattr(cache, "layers"):
        cache_layer = cache.layers[layer]
        key = next(
            getattr(cache_layer, name)
            for name in ("keys", "key_cache", "key_states")
            if hasattr(cache_layer, name)
        )
        value = next(
            getattr(cache_layer, name)
            for name in ("values", "value_cache", "value_states")
            if hasattr(cache_layer, name)
        )
        return key, value
    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()[layer]
        return legacy[0], legacy[1]
    if isinstance(cache, (list, tuple)):
        return cache[layer][0], cache[layer][1]
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


class DynamicKVController:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        *,
        context_length: int,
        context_start: int = 0,
        context_block_start: int,
        block_tokens: int,
        blocks_per_refresh: int,
        retrieval_interval: int,
    ) -> None:
        self.context_length = context_length
        self.context_start = context_start
        self.context_end = context_start + context_length
        self.context_block_start = context_block_start
        self.block_tokens = block_tokens
        self.blocks_per_refresh = blocks_per_refresh
        self.retrieval_interval = retrieval_interval
        self.phase = "prompt"
        self.phase_token = 0
        self.force_refresh = True
        self.selected_by_layer: dict[int, list[int]] = {}
        self.events: list[dict[str, Any]] = []
        self.original: dict[int, Any] = {}
        for layer, decoder_layer in enumerate(model.model.layers):
            attention = decoder_layer.self_attn
            self.original[layer] = attention.forward
            attention.forward = MethodType(self._wrapper(layer), attention)

    def set_step(self, phase: str, phase_token: int, force_refresh: bool) -> None:
        self.phase = phase
        self.phase_token = phase_token
        self.force_refresh = force_refresh

    def _keep_mask(self, selected: Sequence[int], total_tokens: int, device: torch.device) -> torch.Tensor:
        keep = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        keep[: self.context_start] = True
        for block_id in selected:
            start = self.context_start + block_id * self.block_tokens
            end = min(start + self.block_tokens, self.context_end)
            keep[start:end] = True
        keep[self.context_end :] = True
        return keep

    def _wrapper(self, layer: int):
        original = self.original[layer]

        def wrapped(
            module: torch.nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_value: Any = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ):
            result = original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
            if past_key_value is None or hidden_states.shape[1] != 1:
                raise ValueError("dynamic KV generation requires one cached token per step")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_current = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(query, key_current, *position_embeddings)
            key_cache, value_cache = cache_layer_tensors(past_key_value, layer)
            repeat_groups = query.shape[1] // key_cache.shape[1]
            keys = repeat_kv(key_cache, repeat_groups)[0].float()
            values = repeat_kv(value_cache, repeat_groups)[0].float()
            q = query[0, :, 0].float()
            logits = torch.einsum("hd,hsd->hs", q, keys) * float(module.scaling)
            full_attention = torch.softmax(logits, dim=-1)
            full_output = torch.einsum("hs,hsd->hd", full_attention, values)

            refresh = self.force_refresh or layer not in self.selected_by_layer
            if refresh:
                selected = select_shared_qk_blocks(
                    logits[:, self.context_start : self.context_end],
                    self.context_length,
                    self.block_tokens,
                    self.blocks_per_refresh,
                )
                self.selected_by_layer[layer] = selected
                self.events.append(
                    {
                        "phase": self.phase,
                        "phase_token": self.phase_token,
                        "layer": layer,
                        "selected_local_block_ids": selected,
                        "selected_global_block_ids": [
                            self.context_block_start + item for item in selected
                        ],
                    }
                )
            selected = self.selected_by_layer[layer]
            keep = self._keep_mask(selected, keys.shape[1], logits.device)
            sparse_attention = torch.softmax(logits.masked_fill(~keep[None], -torch.inf), dim=-1)
            sparse_output = torch.einsum("hs,hsd->hd", sparse_attention, values)
            delta_heads = (sparse_output - full_output).reshape(1, 1, -1).to(hidden_states.dtype)
            projected_delta = F.linear(delta_heads, module.o_proj.weight, bias=None)
            output = list(result)
            output[0] = output[0] + projected_delta
            return tuple(output)

        return wrapped

    def close(self, model: AutoModelForCausalLM) -> None:
        for layer, original in self.original.items():
            model.model.layers[layer].self_attn.forward = original


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def advance_token(
    model: AutoModelForCausalLM,
    token_id: int,
    cache: Any,
    position: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    outputs = model(
        input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
        past_key_values=cache,
        cache_position=torch.tensor([position], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, outputs.logits[0, -1].float()


@torch.inference_mode()
def prefill(
    model: AutoModelForCausalLM,
    input_ids: Sequence[int],
    chunk_size: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    cache = None
    logits = None
    for start in range(0, len(input_ids), chunk_size):
        chunk = input_ids[start : start + chunk_size]
        outputs = model(
            input_ids=torch.tensor([chunk], dtype=torch.long, device=device),
            past_key_values=cache,
            cache_position=torch.arange(start, start + len(chunk), device=device),
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1].float()
    if logits is None:
        raise ValueError("prefill input cannot be empty")
    return cache, logits


def build_prompt(question: str, style: str = "legacy") -> str:
    if style == "longbench":
        return (
            "\nAnswer the question based on the given passages. Only give the answer "
            "and do not output any other words.\n\n"
            f"Question: {question}\nAnswer:"
        )
    if style == "reasoning_v2":
        return (
            f"\nQuestion: {question}\n"
            "Use the memory to solve the relation chain. Write at least two short factual "
            "reasoning steps before the final answer. Track exactly which person each fact "
            "describes. Then write one concise line in the format `Final answer: ...`.\n"
            "Reasoning:\n1."
        )
    return (
        f"\nQuestion: {question}\n"
        "Reason step by step using the relevant facts. End with a concise line in the "
        "format `Final answer: ...`.\nAnswer:"
    )


def build_chat_prompt_parts(
    tokenizer: Any, question: str, style: str
) -> tuple[list[int], list[int]]:
    sentinel = "__DYNAMIC_KV_MEMORY_SENTINEL__"
    if style == "reasoning_v2":
        instruction = (
            f"Question: {question}\n"
            "Use the memory to solve the relation chain. Write at least two short factual "
            "reasoning steps before the final answer. Track exactly which person each fact "
            "describes. Then write one concise line in the format `Final answer: ...`."
        )
    elif style == "longbench":
        instruction = (
            f"Question: {question}\n"
            "Answer the question based on the memory. Only give the answer and do not "
            "output any other words."
        )
    else:
        instruction = (
            f"Question: {question}\nReason step by step using the relevant facts. "
            "End with a concise line in the format `Final answer: ...`."
        )
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Memory:\n{sentinel}\n\n{instruction}",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = rendered.split(sentinel, maxsplit=1)
    return (
        tokenizer(prefix, add_special_tokens=False)["input_ids"],
        tokenizer(suffix, add_special_tokens=False)["input_ids"],
    )


def decode_block(tokenizer: Any, blocks: np.ndarray, global_block_id: int) -> str:
    text = tokenizer.decode(blocks[global_block_id].tolist(), skip_special_tokens=True)
    return " ".join(text.split())[:360]


def main() -> None:
    args = parse_args()
    if args.retrieval_interval < 1 or args.blocks_per_refresh < 1:
        raise ValueError("retrieval interval and blocks per refresh must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    corpus_dir = Path(args.corpus_dir)
    queries = read_jsonl(corpus_dir / "queries.jsonl")
    query = next(row for row in queries if int(row["query_id"]) == args.query_id)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    block_start = int(query["block_start"])
    block_count = int(query["block_count"])
    context_ids = np.asarray(
        blocks[block_start : block_start + block_count], dtype=np.int64
    ).reshape(-1).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if args.use_chat_template:
        chat_prefix_ids, prompt_ids = build_chat_prompt_parts(
            tokenizer, str(query["question"]), args.prompt_style
        )
    else:
        chat_prefix_ids = []
        prompt_ids = tokenizer(
            build_prompt(str(query["question"]), args.prompt_style),
            add_special_tokens=False,
        )["input_ids"]
    if args.seed_hop1 and args.seed_hop2_query:
        raise ValueError("choose at most one seeded reasoning prefix")
    reasoning_prefix = (
        HOP2_QUERY_PREFIX
        if args.seed_hop2_query
        else HOP1_PREFIX if args.seed_hop1 else ""
    )
    prefix_ids = tokenizer(reasoning_prefix, add_special_tokens=False)["input_ids"]
    started = time.perf_counter()
    controller = None
    if args.mode == "question_only":
        initial_ids = chat_prefix_ids + prompt_ids + prefix_ids
        cache, logits = prefill(model, initial_ids, args.prefill_chunk, device)
        position = len(initial_ids)
    else:
        memory_ids = chat_prefix_ids + context_ids
        cache, _ = prefill(model, memory_ids, args.prefill_chunk, device)
        position = len(memory_ids)
        if args.mode == "dynamic":
            controller = DynamicKVController(
                model,
                context_length=len(context_ids),
                context_start=len(chat_prefix_ids),
                context_block_start=block_start,
                block_tokens=args.block_tokens,
                blocks_per_refresh=args.blocks_per_refresh,
                retrieval_interval=args.retrieval_interval,
            )
        logits = torch.empty(0, device=device)
        for prompt_index, token_id in enumerate(prompt_ids + prefix_ids):
            if controller is not None:
                controller.set_step("prompt", prompt_index, force_refresh=True)
            cache, logits = advance_token(model, token_id, cache, position, device)
            position += 1

    target_first_token = int(
        tokenizer(" Dayton", add_special_tokens=False)["input_ids"][0]
    )
    probabilities = torch.softmax(logits, dim=-1)
    target_probability = float(probabilities[target_first_token].item())
    target_rank = int((logits > logits[target_first_token]).sum().item()) + 1
    top_values, top_ids = torch.topk(probabilities, k=10)
    initial_top10 = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "probability": float(probability),
        }
        for token_id, probability in zip(
            top_ids.tolist(), top_values.tolist()
        )
    ]

    generated: list[int] = []
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for generated_index in range(args.max_new_tokens):
        token_id = int(torch.argmax(logits).item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        if controller is not None:
            controller.set_step(
                "generation",
                generated_index,
                force_refresh=(generated_index % args.retrieval_interval == 0),
            )
        cache, logits = advance_token(model, token_id, cache, position, device)
        position += 1
    if controller is not None:
        controller.close(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    answers = [str(item) for item in query["answers"]]
    events = controller.events if controller is not None else []
    unique_local = sorted(
        {
            int(block_id)
            for event in events
            for block_id in event["selected_local_block_ids"]
        }
    )
    refreshes: dict[tuple[str, int], set[int]] = defaultdict(set)
    for event in events:
        refreshes[(str(event["phase"]), int(event["phase_token"]))].update(
            int(item) for item in event["selected_local_block_ids"]
        )
    result = {
        "query_id": args.query_id,
        "dataset": query["dataset"],
        "question": query["question"],
        "answers": answers,
        "mode": args.mode,
        "prompt_style": args.prompt_style,
        "use_chat_template": args.use_chat_template,
        "seed_hop1": args.seed_hop1,
        "seed_hop2_query": args.seed_hop2_query,
        "reasoning_prefix": reasoning_prefix,
        "retrieval_interval": args.retrieval_interval,
        "blocks_per_refresh_per_layer": args.blocks_per_refresh,
        "context_tokens": len(context_ids) if args.mode != "question_only" else 0,
        "context_blocks": block_count if args.mode != "question_only" else 0,
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "dayton_first_token_probability": target_probability,
        "dayton_first_token_rank": target_rank,
        "initial_top10_tokens": initial_top10,
        "answer_hit_128": answer_hit(generated_text, answers),
        "first_answer_end_normalized_token": first_answer_end(generated_text, answers),
        "gold_global_block_ids": query["gold_block_ids"],
        "refresh_events": len(refreshes),
        "layer_refresh_rows": len(events),
        "unique_local_blocks": unique_local,
        "unique_global_blocks": [block_start + item for item in unique_local],
        "unique_block_count": len(unique_local),
        "gold_ever_retrieved": bool(
            set(int(item) for item in query["gold_block_ids"])
            & {block_start + item for item in unique_local}
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "retrieval_events": events,
        "retrieved_block_snippets": {
            str(block_start + item): decode_block(tokenizer, blocks, block_start + item)
            for item in unique_local
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "retrieval_events"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
