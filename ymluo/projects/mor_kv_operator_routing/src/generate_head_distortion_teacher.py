from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exact per-head omitted-mass and attention-output-distortion labels "
            "for candidate KV block operators."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--query_vector_tokens", type=int, default=1)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--budget_blocks", type=int, default=8)
    parser.add_argument("--sink_blocks", type=int, default=1)
    parser.add_argument("--recent_blocks", type=int, default=1)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument(
        "--device_map",
        choices=["none", "auto", "balanced"],
        default="none",
        help="Optionally shard larger models across visible GPUs with Accelerate.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    output: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(value) for value in item.split("-", 1)]
            output.update(range(start, end + 1))
        else:
            output.add(int(item))
    layers = sorted(output)
    if not layers or layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError("invalid layer selection")
    return layers


def block_ids_for_positions(length: int, block_tokens: int) -> torch.Tensor:
    return torch.arange(length, dtype=torch.long) // block_tokens


def mandatory_blocks(
    context_blocks: int, sink_blocks: int, recent_blocks: int
) -> set[int]:
    output = set(range(min(sink_blocks, context_blocks)))
    output.update(
        range(max(context_blocks - recent_blocks, 0), context_blocks)
    )
    return output


def rank_blocks(values: torch.Tensor, largest: bool = True) -> list[int]:
    block_ids = torch.arange(values.numel(), device=values.device)
    # Stable CPU tie-breaking makes teacher generation reproducible.
    pairs = list(zip(values.detach().float().cpu().tolist(), block_ids.cpu().tolist()))
    pairs.sort(key=lambda item: ((-item[0]) if largest else item[0], item[1]))
    return [int(block_id) for _, block_id in pairs]


def block_score_vectors(
    logits: torch.Tensor,
    context_token_ids: torch.Tensor,
    question_token_set: set[int],
    block_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_length = int(context_token_ids.numel())
    context_blocks = math.ceil(context_length / block_tokens)
    qk_scores = torch.full(
        (context_blocks,), -torch.inf, dtype=torch.float32, device=logits.device
    )
    lexical_scores = torch.zeros(
        context_blocks, dtype=torch.float32, device=logits.device
    )
    for block in range(context_blocks):
        start, end = block * block_tokens, min((block + 1) * block_tokens, context_length)
        qk_scores[block] = logits[start:end].amax()
        lexical_scores[block] = sum(
            int(token_id) in question_token_set
            for token_id in context_token_ids[start:end].tolist()
        )
    return qk_scores, lexical_scores


def vector_signature(values: torch.Tensor, stem: str) -> dict[str, float]:
    ordered = torch.sort(values.float(), descending=True).values
    shifted = ordered - ordered[0]
    probabilities = torch.softmax(shifted, dim=0)
    return {
        f"{stem}_top1": float(ordered[0].item()),
        f"{stem}_margin12": float((ordered[0] - ordered[min(1, len(ordered) - 1)]).item()),
        f"{stem}_margin14": float((ordered[0] - ordered[min(3, len(ordered) - 1)]).item()),
        f"{stem}_std": float(ordered.std(unbiased=False).item()),
        f"{stem}_entropy": float(
            (-(probabilities * torch.log(probabilities + 1.0e-30)).sum()).item()
        ),
    }


def select_blocks(
    action: str,
    logits: torch.Tensor,
    full_attention: torch.Tensor,
    context_token_ids: torch.Tensor,
    question_token_set: set[int],
    block_tokens: int,
    budget_blocks: int,
    mandatory: set[int],
) -> list[int]:
    context_length = int(context_token_ids.numel())
    context_blocks = math.ceil(context_length / block_tokens)
    if action == "full":
        return list(range(context_blocks))
    if action == "streaming":
        return sorted(mandatory)
    if action == "uniform":
        candidates = np.linspace(
            0, max(context_blocks - 1, 0), num=min(budget_blocks, context_blocks), dtype=np.int64
        ).tolist()
        order = [int(item) for item in candidates]
    else:
        block_scores = torch.full(
            (context_blocks,), -torch.inf, dtype=torch.float32, device=logits.device
        )
        if action == "qk_top_blocks":
            for block in range(context_blocks):
                start, end = block * block_tokens, min((block + 1) * block_tokens, context_length)
                block_scores[block] = logits[start:end].amax()
        elif action == "mass_oracle_blocks":
            block_scores.zero_()
            for block in range(context_blocks):
                start, end = block * block_tokens, min((block + 1) * block_tokens, context_length)
                block_scores[block] = full_attention[start:end].sum()
        elif action == "lexical_blocks":
            block_scores.zero_()
            for block in range(context_blocks):
                start, end = block * block_tokens, min((block + 1) * block_tokens, context_length)
                block_scores[block] = sum(
                    int(token_id) in question_token_set
                    for token_id in context_token_ids[start:end].tolist()
                )
        else:
            raise ValueError(f"unknown action {action}")
        order = rank_blocks(block_scores)
    selected = list(sorted(mandatory))
    seen = set(selected)
    for block in order:
        if len(selected) >= min(budget_blocks, context_blocks):
            break
        if block not in seen:
            selected.append(block)
            seen.add(block)
    if len(selected) < min(budget_blocks, context_blocks):
        for block in range(context_blocks):
            if len(selected) >= min(budget_blocks, context_blocks):
                break
            if block not in seen:
                selected.append(block)
                seen.add(block)
    return selected


def keep_mask_from_blocks(
    selected_blocks: Sequence[int],
    key_length: int,
    context_length: int,
    block_tokens: int,
    query_position: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(key_length, device=device)
    context_keep = torch.zeros(context_length, dtype=torch.bool, device=device)
    for block in selected_blocks:
        start = int(block) * block_tokens
        end = min((int(block) + 1) * block_tokens, context_length)
        context_keep[start:end] = True
    keep = torch.zeros(key_length, dtype=torch.bool, device=device)
    keep[:context_length] = context_keep
    # Prompt/query tokens are always resident and are not charged to the remote block budget.
    keep[context_length : query_position + 1] = True
    keep &= positions <= query_position
    return keep


def distortion_metrics(
    logits: torch.Tensor,
    values: torch.Tensor,
    keep: torch.Tensor,
) -> dict[str, float]:
    full_attention = torch.softmax(logits.float(), dim=-1)
    full_output = full_attention @ values.float()
    sparse_logits = logits.float().masked_fill(~keep, -torch.inf)
    sparse_attention = torch.softmax(sparse_logits, dim=-1)
    sparse_output = sparse_attention @ values.float()
    omitted_mass = float(full_attention[~keep].sum().item())
    difference = full_output - sparse_output
    l2 = float(torch.linalg.vector_norm(difference).item())
    full_norm = float(torch.linalg.vector_norm(full_output).item())
    cosine = float(
        F.cosine_similarity(full_output[None], sparse_output[None], dim=-1).item()
    )
    max_value_norm = float(torch.linalg.vector_norm(values.float(), dim=-1).max().item())
    bound = 2.0 * max_value_norm * omitted_mass
    return {
        "omitted_mass": omitted_mass,
        "output_l2": l2,
        "relative_output_l2": l2 / max(full_norm, 1.0e-8),
        "output_cosine": cosine,
        "max_value_norm": max_value_norm,
        "mass_bound": bound,
        "bound_satisfied": float(l2 <= bound + 1.0e-5),
    }


class DistortionCapture:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        layers: Sequence[int],
        input_ids: torch.Tensor,
        context_length: int,
        query_positions: Sequence[int],
        question_token_set: set[int],
        block_tokens: int,
        budget_blocks: int,
        sink_blocks: int,
        recent_blocks: int,
        query_id: int,
        dataset: str,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.original: dict[int, Any] = {}
        self.input_ids = input_ids
        self.context_length = context_length
        self.query_positions = list(query_positions)
        self.question_token_set = question_token_set
        self.block_tokens = block_tokens
        self.budget_blocks = budget_blocks
        self.query_id = query_id
        self.dataset = dataset
        context_blocks = math.ceil(context_length / block_tokens)
        self.mandatory = mandatory_blocks(context_blocks, sink_blocks, recent_blocks)
        for layer in layers:
            attention = model.model.layers[layer].self_attn
            self.original[layer] = attention.forward
            attention.forward = MethodType(self._wrapper(layer), attention)

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
            if past_key_value is not None:
                raise ValueError("distortion teacher expects a cache-free full forward")
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, *position_embeddings
            )
            repeat_groups = query_states.shape[1] // key_states.shape[1]
            key_states = repeat_kv(key_states, repeat_groups)
            value_states = repeat_kv(value_states, repeat_groups)
            context_ids = self.input_ids[: self.context_length].detach().cpu()
            actions = [
                "full",
                "streaming",
                "uniform",
                "lexical_blocks",
                "qk_top_blocks",
                "mass_oracle_blocks",
            ]
            for query_position in self.query_positions:
                q = query_states[0, :, query_position].float()
                keys = key_states[0, :, : query_position + 1].float()
                values = value_states[0, :, : query_position + 1].float()
                logits = torch.einsum("hd,hsd->hs", q, keys) * float(module.scaling)
                full_attention = torch.softmax(logits, dim=-1)
                for head in range(q.shape[0]):
                    qk_block_scores, lexical_block_scores = block_score_vectors(
                        logits[head, : self.context_length],
                        context_ids,
                        self.question_token_set,
                        self.block_tokens,
                    )
                    qk_order = rank_blocks(qk_block_scores)
                    lexical_order = rank_blocks(lexical_block_scores)
                    comparison_depth = min(self.budget_blocks, len(qk_order))
                    qk_set = set(qk_order[:comparison_depth])
                    lexical_set = set(lexical_order[:comparison_depth])
                    score_features = {
                        **vector_signature(qk_block_scores, "qk"),
                        **vector_signature(lexical_block_scores, "lexical"),
                        "lexical_nonzero_fraction": float(
                            (lexical_block_scores > 0).float().mean().item()
                        ),
                        "qk_lexical_topk_jaccard": len(qk_set & lexical_set)
                        / max(len(qk_set | lexical_set), 1),
                    }
                    for action in actions:
                        selected_blocks = select_blocks(
                            action,
                            logits[head, : self.context_length],
                            full_attention[head, : self.context_length],
                            context_ids,
                            self.question_token_set,
                            self.block_tokens,
                            self.budget_blocks,
                            self.mandatory,
                        )
                        keep = keep_mask_from_blocks(
                            selected_blocks,
                            query_position + 1,
                            self.context_length,
                            self.block_tokens,
                            query_position,
                            logits.device,
                        )
                        metrics = distortion_metrics(logits[head], values[head], keep)
                        self.rows.append(
                            {
                                "query_id": self.query_id,
                                "dataset": self.dataset,
                                "layer": layer,
                                "query_head": head,
                                "kv_head": head // repeat_groups,
                                "query_position": query_position,
                                "action": action,
                                "selected_blocks": len(selected_blocks),
                                "selected_block_ids": json.dumps(selected_blocks),
                                **score_features,
                                **metrics,
                            }
                        )
            return original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        return wrapped

    def close(self, model: AutoModelForCausalLM) -> None:
        for layer, original in self.original.items():
            model.model.layers[layer].self_attn.forward = original


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("distortion teacher requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    all_queries = read_jsonl(corpus_dir / "queries.jsonl")
    query_end = len(all_queries) if args.max_queries <= 0 else args.query_start + args.max_queries
    queries = all_queries[args.query_start : query_end]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        device_map=None if args.device_map == "none" else args.device_map,
    )
    if args.device_map == "none":
        model = model.to(device)
    model.eval()
    model.config.use_cache = False
    layers = parse_layers(args.layers, len(model.model.layers))
    rows: list[dict[str, Any]] = []

    for query in queries:
        query_id = int(query["query_id"])
        block_start = int(query["block_start"])
        block_count = int(query["block_count"])
        context_array = np.asarray(
            blocks[block_start : block_start + block_count], dtype=np.int64
        ).reshape(-1)
        if args.max_context_tokens > 0:
            context_array = context_array[: args.max_context_tokens]
        context_ids = torch.from_numpy(np.array(context_array, copy=True)).long()
        prefix_ids = torch.tensor(
            tokenizer("\nQuestion: ", add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        question_ids = torch.tensor(
            tokenizer(str(query["question"]), add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        suffix_ids = torch.tensor(
            tokenizer("\nAnswer:", add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        input_ids = torch.cat([context_ids, prefix_ids, question_ids, suffix_ids])
        question_start = len(context_ids) + len(prefix_ids)
        query_positions = list(
            range(
                question_start + max(len(question_ids) - args.query_vector_tokens, 0),
                question_start + len(question_ids),
            )
        )
        capture = DistortionCapture(
            model,
            layers,
            input_ids,
            len(context_ids),
            query_positions,
            set(int(item) for item in question_ids.tolist()),
            args.block_tokens,
            args.budget_blocks,
            args.sink_blocks,
            args.recent_blocks,
            query_id,
            str(query["dataset"]),
        )
        with torch.inference_mode():
            input_device = model.get_input_embeddings().weight.device
            model.model(
                input_ids=input_ids[None].to(input_device),
                attention_mask=torch.ones(
                    (1, len(input_ids)), dtype=torch.long, device=input_device
                ),
                use_cache=False,
                return_dict=True,
            )
        capture.close(model)
        rows.extend(capture.rows)
        print(
            json.dumps(
                {
                    "query_id": query_id,
                    "context_tokens": len(context_ids),
                    "rows": len(capture.rows),
                }
            ),
            flush=True,
        )

    fields = list(rows[0])
    with (output_dir / "distortion_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    action_summary: list[dict[str, Any]] = []
    for action in sorted({row["action"] for row in rows}):
        group = [row for row in rows if row["action"] == action]
        action_summary.append(
            {
                "action": action,
                "rows": len(group),
                "mean_selected_blocks": statistics.fmean(
                    float(row["selected_blocks"]) for row in group
                ),
                "mean_omitted_mass": statistics.fmean(
                    float(row["omitted_mass"]) for row in group
                ),
                "mean_output_l2": statistics.fmean(
                    float(row["output_l2"]) for row in group
                ),
                "mean_relative_output_l2": statistics.fmean(
                    float(row["relative_output_l2"]) for row in group
                ),
                "mean_output_cosine": statistics.fmean(
                    float(row["output_cosine"]) for row in group
                ),
                "bound_satisfaction_rate": statistics.fmean(
                    float(row["bound_satisfied"]) for row in group
                ),
            }
        )
    omitted = np.asarray([float(row["omitted_mass"]) for row in rows])
    output_error = np.asarray([float(row["output_l2"]) for row in rows])
    summary = {
        "source": "exact per-head causal attention-distortion teacher prototype",
        "model": args.model_name_or_path,
        "queries": len(queries),
        "layers": layers,
        "query_vector_tokens": args.query_vector_tokens,
        "block_tokens": args.block_tokens,
        "budget_blocks": args.budget_blocks,
        "actions": action_summary,
        "omitted_mass_output_l2_pearson": float(np.corrcoef(omitted, output_error)[0, 1]),
        "interpretation": (
            "Labels are computed from exact post-RoPE Q/K/V attention for the selected "
            "query positions. The full action is a numerical zero-error control."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
