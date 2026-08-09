from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from generate_head_distortion_teacher import (
    block_score_vectors,
    distortion_metrics,
    keep_mask_from_blocks,
    mandatory_blocks,
    parse_layers,
    resolve_dtype,
    rank_blocks,
    select_blocks,
    vector_signature,
)
from train_head_distortion_router import feature_vector


DEPLOYABLE_ACTIONS = ["streaming", "lexical_blocks", "uniform", "qk_top_blocks"]


def select_shared_mass_blocks(
    full_attention: torch.Tensor,
    context_length: int,
    block_tokens: int,
    budget_blocks: int,
    mandatory: set[int],
    allowed_blocks: set[int] | None = None,
) -> list[int]:
    """Select one block set shared by every query head in the current layer."""
    context_blocks = math.ceil(context_length / block_tokens)
    scores = torch.zeros(
        context_blocks, dtype=torch.float32, device=full_attention.device
    )
    for block in range(context_blocks):
        start = block * block_tokens
        end = min((block + 1) * block_tokens, context_length)
        scores[block] = full_attention[:, start:end].sum()
    selected = list(sorted(mandatory))
    seen = set(selected)
    for block in rank_blocks(scores):
        if len(selected) >= min(budget_blocks, context_blocks):
            break
        if allowed_blocks is not None and block not in allowed_blocks:
            continue
        if block not in seen:
            selected.append(block)
            seen.add(block)
    return selected


def select_shared_output_blocks(
    full_attention: torch.Tensor,
    values: torch.Tensor,
    context_length: int,
    block_tokens: int,
    budget_blocks: int,
    mandatory: set[int],
) -> list[int]:
    """Greedily minimize mean per-head relative attention-output error."""
    context_blocks = math.ceil(context_length / block_tokens)
    heads, _, head_dim = values.shape
    block_mass = torch.zeros(
        (context_blocks, heads), dtype=torch.float32, device=values.device
    )
    block_output = torch.zeros(
        (context_blocks, heads, head_dim), dtype=torch.float32, device=values.device
    )
    for block in range(context_blocks):
        start = block * block_tokens
        end = min((block + 1) * block_tokens, context_length)
        weights = full_attention[:, start:end]
        block_mass[block] = weights.sum(dim=-1)
        block_output[block] = torch.einsum(
            "hs,hsd->hd", weights, values[:, start:end]
        )

    resident_attention = full_attention[:, context_length:]
    resident_values = values[:, context_length:]
    current_mass = resident_attention.sum(dim=-1)
    current_output = torch.einsum(
        "hs,hsd->hd", resident_attention, resident_values
    )
    selected = list(sorted(mandatory))
    seen = set(selected)
    if selected:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=values.device)
        current_mass = current_mass + block_mass[selected_tensor].sum(dim=0)
        current_output = current_output + block_output[selected_tensor].sum(dim=0)

    full_output = torch.einsum("hs,hsd->hd", full_attention, values)
    full_norm = torch.linalg.vector_norm(full_output, dim=-1).clamp_min(1.0e-8)
    while len(selected) < min(budget_blocks, context_blocks):
        candidates = [block for block in range(context_blocks) if block not in seen]
        candidate_tensor = torch.tensor(
            candidates, dtype=torch.long, device=values.device
        )
        mass = current_mass[None] + block_mass[candidate_tensor]
        output = current_output[None] + block_output[candidate_tensor]
        sparse_output = output / mass.clamp_min(1.0e-30)[..., None]
        relative_error = (
            torch.linalg.vector_norm(sparse_output - full_output[None], dim=-1)
            / full_norm[None]
        )
        objective = relative_error.mean(dim=-1)
        best_offset = int(torch.argmin(objective).item())
        best = candidates[best_offset]
        selected.append(best)
        seen.add(best)
        current_mass = current_mass + block_mass[best]
        current_output = current_output + block_output[best]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causally replace the final prompt token's per-head attention output at every "
            "layer and measure first-answer-token NLL. This is an exact reference evaluator, "
            "not a speed kernel."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--budget_blocks", type=int, default=8)
    parser.add_argument("--sink_blocks", type=int, default=1)
    parser.add_argument("--recent_blocks", type=int, default=1)
    parser.add_argument("--risk_threshold", type=float, default=0.05)
    parser.add_argument(
        "--answer_tokens",
        type=int,
        default=1,
        help="Number of gold answer tokens to score; 0 scores the full first answer.",
    )
    parser.add_argument("--router_bundle")
    parser.add_argument("--postrope_basis")
    parser.add_argument("--proposal_multiplier", type=int, default=4)
    parser.add_argument(
        "--router_error_threshold",
        type=float,
        help="Override the deployment bundle's learned-router error threshold.",
    )
    parser.add_argument("--router_test_only", action="store_true")
    parser.add_argument(
        "--sparse_layers",
        default="all",
        help="Layer specification such as all, 0-13, or 0,2,4.",
    )
    parser.add_argument(
        "--actions",
        default="full,streaming,uniform,lexical_blocks,qk_top_blocks,mass_oracle_blocks,risk_oracle",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_answer_token(tokenizer: Any, answers: Sequence[str]) -> int:
    return answer_token_ids(tokenizer, answers)[0]


def answer_token_ids(tokenizer: Any, answers: Sequence[str]) -> list[int]:
    for answer in answers:
        normalized = str(answer).strip()
        if not normalized:
            continue
        # Match the established answer-NLL protocol: the prompt ends in ``Answer:``
        # and the gold continuation starts with one space.
        token_ids = tokenizer(" " + normalized, add_special_tokens=False)["input_ids"]
        if token_ids:
            return [int(item) for item in token_ids]
    raise ValueError("query has no non-empty answer token")


class SparseAttentionReference:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        query_positions: Sequence[int],
        context_length: int,
        question_token_set: set[int],
        action: str,
        block_tokens: int,
        budget_blocks: int,
        sink_blocks: int,
        recent_blocks: int,
        risk_threshold: float,
        router_bundle: dict[str, Any] | None,
        postrope_basis: torch.Tensor | None,
        proposal_multiplier: int,
        sparse_layers: Sequence[int],
    ) -> None:
        self.input_ids = input_ids
        self.context_length = context_length
        self.query_positions = [int(item) for item in query_positions]
        self.question_token_set = question_token_set
        self.action = action
        self.block_tokens = block_tokens
        self.budget_blocks = budget_blocks
        self.risk_threshold = risk_threshold
        self.router_bundle = router_bundle
        self.postrope_basis = postrope_basis
        self.proposal_multiplier = proposal_multiplier
        self.sparse_layers = set(int(item) for item in sparse_layers)
        self.original: dict[int, Any] = {}
        self.head_rows: list[dict[str, Any]] = []
        context_blocks = math.ceil(context_length / block_tokens)
        self.mandatory = mandatory_blocks(context_blocks, sink_blocks, recent_blocks)
        for layer, decoder_layer in enumerate(model.model.layers):
            attention = decoder_layer.self_attn
            self.original[layer] = attention.forward
            attention.forward = MethodType(self._wrapper(layer), attention)

    def _choose(
        self,
        layer: int,
        head: int,
        query_position: int,
        logits: torch.Tensor,
        values: torch.Tensor,
        full_attention: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> tuple[str, list[int], dict[str, float]]:
        if self.action == "learned_conformal":
            if self.router_bundle is None:
                raise ValueError("learned_conformal requires --router_bundle")
            qk_scores, lexical_scores = block_score_vectors(
                logits[: self.context_length],
                context_ids,
                self.question_token_set,
                self.block_tokens,
            )
            qk_order = rank_blocks(qk_scores)
            lexical_order = rank_blocks(lexical_scores)
            depth = min(self.budget_blocks, len(qk_order))
            qk_set = set(qk_order[:depth])
            lexical_set = set(lexical_order[:depth])
            score_features = {
                **vector_signature(qk_scores, "qk"),
                **vector_signature(lexical_scores, "lexical"),
                "lexical_nonzero_fraction": float(
                    (lexical_scores > 0).float().mean().item()
                ),
                "qk_lexical_topk_jaccard": len(qk_set & lexical_set)
                / max(len(qk_set | lexical_set), 1),
            }
            row = {"layer": layer, "query_head": head, **score_features}
            features = feature_vector(
                row,
                int(self.router_bundle["num_layers"]),
                int(self.router_bundle["num_heads"]),
                interaction_mode=str(
                    self.router_bundle.get("interaction_mode", "none")
                ),
            )
            feasible: list[tuple[str, float]] = []
            prediction_by_action: dict[str, float] = {}
            upper_by_action: dict[str, float] = {}
            for candidate in self.router_bundle["deployable_actions"]:
                model = self.router_bundle["models"][candidate]
                mean = np.asarray(model["feature_mean"], dtype=np.float64)
                scale = np.asarray(model["feature_scale"], dtype=np.float64)
                weights = np.asarray(model["weights"], dtype=np.float64)
                standardized = (features - mean) / scale
                prediction = float(weights[0] + standardized @ weights[1:])
                scope = str(self.router_bundle["conformal_scope"])
                if scope == "head":
                    correction_key = f"{layer}:{head}"
                elif scope == "layer":
                    correction_key = str(layer)
                else:
                    correction_key = ""
                correction = float(
                    model["correction_by_group"].get(
                        correction_key, model["global_correction"]
                    )
                )
                # Error is non-negative even when the unconstrained ridge prediction
                # extrapolates below zero.
                upper_bound = max(
                    float(self.router_bundle.get("upper_bound_floor", 0.0)),
                    prediction + correction,
                )
                prediction_by_action[candidate] = prediction
                upper_by_action[candidate] = upper_bound
                if upper_bound <= float(
                    self.router_bundle["relative_error_threshold"]
                ):
                    feasible.append((candidate, upper_bound))
            chosen = min(
                feasible,
                key=lambda item: (
                    len(self.mandatory) if item[0] == "streaming" else self.budget_blocks,
                    item[1],
                    item[0],
                ),
            )[0] if feasible else "full"
            selected = select_blocks(
                chosen,
                logits[: self.context_length],
                full_attention[: self.context_length],
                context_ids,
                self.question_token_set,
                self.block_tokens,
                self.budget_blocks,
                self.mandatory,
            )
            keep = keep_mask_from_blocks(
                selected,
                query_position + 1,
                self.context_length,
                self.block_tokens,
                query_position,
                logits.device,
            )
            metrics = distortion_metrics(logits, values, keep)
            metrics.update(
                {
                    "router_predicted_error": (
                        prediction_by_action[chosen] if chosen != "full" else 0.0
                    ),
                    "router_upper_bound": (
                        upper_by_action[chosen] if chosen != "full" else 0.0
                    ),
                    "router_error_threshold": float(
                        self.router_bundle["relative_error_threshold"]
                    ),
                }
            )
            return chosen, selected, metrics

        if self.action != "risk_oracle":
            selected = select_blocks(
                self.action,
                logits[: self.context_length],
                full_attention[: self.context_length],
                context_ids,
                self.question_token_set,
                self.block_tokens,
                self.budget_blocks,
                self.mandatory,
            )
            keep = keep_mask_from_blocks(
                selected,
                query_position + 1,
                self.context_length,
                self.block_tokens,
                query_position,
                logits.device,
            )
            return self.action, selected, distortion_metrics(logits, values, keep)

        candidates: list[tuple[int, float, str, list[int], dict[str, float]]] = []
        for candidate in DEPLOYABLE_ACTIONS:
            selected = select_blocks(
                candidate,
                logits[: self.context_length],
                full_attention[: self.context_length],
                context_ids,
                self.question_token_set,
                self.block_tokens,
                self.budget_blocks,
                self.mandatory,
            )
            keep = keep_mask_from_blocks(
                selected,
                query_position + 1,
                self.context_length,
                self.block_tokens,
                query_position,
                logits.device,
            )
            metrics = distortion_metrics(logits, values, keep)
            if metrics["relative_output_l2"] <= self.risk_threshold:
                candidates.append(
                    (
                        len(selected),
                        metrics["relative_output_l2"],
                        candidate,
                        selected,
                        metrics,
                    )
                )
        if candidates:
            _, _, chosen, selected, metrics = min(candidates)
            return chosen, selected, metrics
        selected = list(range(math.ceil(self.context_length / self.block_tokens)))
        keep = keep_mask_from_blocks(
            selected,
            query_position + 1,
            self.context_length,
            self.block_tokens,
            query_position,
            logits.device,
        )
        return "full", selected, distortion_metrics(logits, values, keep)

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
                raise ValueError("reference evaluator expects a cache-free full forward")
            original_result = original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
            if layer not in self.sparse_layers:
                query_heads = module.q_proj.out_features // module.head_dim
                kv_heads = module.k_proj.out_features // module.head_dim
                repeat_groups = query_heads // kv_heads
                full_blocks = math.ceil(self.context_length / self.block_tokens)
                for query_position in self.query_positions:
                    for head in range(query_heads):
                        self.head_rows.append(
                            {
                                "layer": layer,
                                "query_head": head,
                                "kv_head": head // repeat_groups,
                                "query_position": query_position,
                                "chosen_action": "full",
                                "selected_blocks": full_blocks,
                                "selected_block_ids": list(range(full_blocks)),
                                "full_blocks": full_blocks,
                                "relative_output_l2": 0.0,
                            }
                        )
                return original_result
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
            output_list = list(original_result)
            attention_output = output_list[0].clone()
            for query_position in self.query_positions:
                q = query_states[0, :, query_position].float()
                keys = key_states[0, :, : query_position + 1].float()
                values = value_states[0, :, : query_position + 1].float()
                logits = torch.einsum("hd,hsd->hs", q, keys) * float(module.scaling)
                full_attention = torch.softmax(logits, dim=-1)
                full_outputs = torch.einsum("hs,hsd->hd", full_attention, values)
                sparse_outputs: list[torch.Tensor] = []
                shared_selected = None
                if self.action == "layer_shared_mass_oracle_blocks":
                    shared_selected = select_shared_mass_blocks(
                        full_attention[:, : self.context_length],
                        self.context_length,
                        self.block_tokens,
                        self.budget_blocks,
                        self.mandatory,
                    )
                elif self.action == "layer_shared_svd_mass_blocks":
                    if self.postrope_basis is None:
                        raise ValueError(
                            "layer_shared_svd_mass_blocks requires --postrope_basis"
                        )
                    basis = self.postrope_basis[layer].repeat_interleave(
                        repeat_groups, dim=0
                    )
                    projected_q = torch.einsum("hd,hdr->hr", q, basis)
                    projected_keys = torch.einsum("hsd,hdr->hsr", keys, basis)
                    approximate_logits = (
                        torch.einsum("hr,hsr->hs", projected_q, projected_keys)
                        * float(module.scaling)
                    )
                    approximate_attention = torch.softmax(approximate_logits, dim=-1)
                    context_blocks = math.ceil(
                        self.context_length / self.block_tokens
                    )
                    proposal_budget = min(
                        context_blocks,
                        self.budget_blocks * self.proposal_multiplier,
                    )
                    proposals = select_shared_mass_blocks(
                        approximate_attention[:, : self.context_length],
                        self.context_length,
                        self.block_tokens,
                        proposal_budget,
                        self.mandatory,
                    )
                    shared_selected = select_shared_mass_blocks(
                        full_attention[:, : self.context_length],
                        self.context_length,
                        self.block_tokens,
                        self.budget_blocks,
                        self.mandatory,
                        allowed_blocks=set(proposals),
                    )
                elif self.action == "layer_shared_output_oracle_blocks":
                    shared_selected = select_shared_output_blocks(
                        full_attention,
                        values,
                        self.context_length,
                        self.block_tokens,
                        self.budget_blocks,
                        self.mandatory,
                    )
                for head in range(q.shape[0]):
                    if shared_selected is not None:
                        chosen = self.action
                        selected = shared_selected
                        keep = keep_mask_from_blocks(
                            selected,
                            query_position + 1,
                            self.context_length,
                            self.block_tokens,
                            query_position,
                            logits.device,
                        )
                        metrics = distortion_metrics(logits[head], values[head], keep)
                    else:
                        chosen, selected, metrics = self._choose(
                            layer,
                            head,
                            query_position,
                            logits[head],
                            values[head],
                            full_attention[head],
                            context_ids,
                        )
                    keep = keep_mask_from_blocks(
                        selected,
                        query_position + 1,
                        self.context_length,
                        self.block_tokens,
                        query_position,
                        logits.device,
                    )
                    sparse_attention = torch.softmax(
                        logits[head].masked_fill(~keep, -torch.inf), dim=-1
                    )
                    sparse_outputs.append(sparse_attention @ values[head])
                    self.head_rows.append(
                        {
                            "layer": layer,
                            "query_head": head,
                            "kv_head": head // repeat_groups,
                            "query_position": query_position,
                            "chosen_action": chosen,
                            "selected_blocks": len(selected),
                            "selected_block_ids": selected,
                            "block_tokens": self.block_tokens,
                            "full_blocks": math.ceil(
                                self.context_length / self.block_tokens
                            ),
                            **metrics,
                        }
                    )
                sparse = torch.stack(sparse_outputs)
                delta_heads = (sparse - full_outputs).reshape(1, -1).to(hidden_states.dtype)
                projected_delta = F.linear(
                    delta_heads, module.o_proj.weight, bias=None
                )[0]
                attention_output[0, query_position] += projected_delta
            output_list[0] = attention_output
            return tuple(output_list)

        return wrapped

    def close(self, model: AutoModelForCausalLM) -> None:
        for layer, original in self.original.items():
            model.model.layers[layer].self_attn.forward = original


def summarize_head_rows(
    rows: Sequence[dict[str, Any]], violation_threshold: float
) -> dict[str, Any]:
    physical_unions: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    layer_unions: dict[tuple[int, int], set[int]] = defaultdict(set)
    full_blocks: dict[tuple[int, int, int], int] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["query_position"]), int(row["kv_head"]))
        physical_unions[key].update(int(item) for item in row["selected_block_ids"])
        layer_unions[(int(row["layer"]), int(row["query_position"]))].update(
            int(item) for item in row["selected_block_ids"]
        )
        full_blocks[key] = int(row["full_blocks"])
    physical_values = np.asarray(
        [len(value) for value in physical_unions.values()], dtype=np.float64
    )
    full_values = np.asarray(
        [full_blocks[key] for key in physical_unions], dtype=np.float64
    )
    layer_values = np.asarray(
        [len(value) for value in layer_unions.values()], dtype=np.float64
    )
    block_tokens = int(rows[0].get("block_tokens", 256))
    rows_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_layer[int(row["layer"])].append(row)
    upper_by_layer = {
        str(layer): statistics.fmean(
            float(row.get("router_upper_bound", 0.0)) for row in layer_rows
        )
        for layer, layer_rows in sorted(rows_by_layer.items())
    }
    sparse_fraction_by_layer = {
        str(layer): statistics.fmean(
            float(str(row["chosen_action"]) != "full") for row in layer_rows
        )
        for layer, layer_rows in sorted(rows_by_layer.items())
    }
    return {
        "mean_selected_blocks": statistics.fmean(float(row["selected_blocks"]) for row in rows),
        "mean_relative_output_l2": statistics.fmean(
            float(row["relative_output_l2"]) for row in rows
        ),
        "p95_relative_output_l2": float(
            np.quantile([float(row["relative_output_l2"]) for row in rows], 0.95)
        ),
        "violation_rate": statistics.fmean(
            float(row["relative_output_l2"] > violation_threshold) for row in rows
        ),
        "action_counts": json.dumps(
            dict(sorted(Counter(str(row["chosen_action"]) for row in rows).items()))
        ),
        "mean_physical_gqa_blocks": float(physical_values.mean()),
        "physical_gqa_saving_rate": float(
            1.0 - physical_values.sum() / full_values.sum()
        ),
        "mean_layer_global_blocks": float(layer_values.mean()),
        "max_layer_global_blocks": int(layer_values.max()),
        "mean_layer_global_tokens": float(layer_values.mean() * block_tokens),
        "max_layer_global_tokens": int(layer_values.max() * block_tokens),
        "strict_1000_token_violation_rate": float(
            np.mean(layer_values * block_tokens > 1000)
        ),
        "mean_router_upper_bound": statistics.fmean(
            float(row.get("router_upper_bound", 0.0)) for row in rows
        ),
        "p95_router_upper_bound": float(
            np.quantile(
                [float(row.get("router_upper_bound", 0.0)) for row in rows], 0.95
            )
        ),
        "max_router_upper_bound": max(
            float(row.get("router_upper_bound", 0.0)) for row in rows
        ),
        "router_near_threshold_fraction": statistics.fmean(
            float(
                float(row.get("router_upper_bound", 0.0))
                >= 0.8 * float(row.get("router_error_threshold", float("inf")))
            )
            for row in rows
        ),
        "mean_router_upper_bound_by_layer": json.dumps(upper_by_layer),
        "router_sparse_fraction_by_layer": json.dumps(sparse_fraction_by_layer),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("reference evaluator requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")
    all_queries = read_jsonl(corpus_dir / "queries.jsonl")
    query_end = len(all_queries) if args.max_queries <= 0 else args.query_start + args.max_queries
    queries = all_queries[args.query_start : query_end]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    router_bundle = (
        json.loads(Path(args.router_bundle).read_text(encoding="utf-8"))
        if args.router_bundle
        else None
    )
    postrope_basis = None
    postrope_basis_metadata: dict[str, Any] = {}
    if args.postrope_basis:
        basis_payload = torch.load(
            args.postrope_basis, map_location="cpu", weights_only=False
        )
        postrope_basis = basis_payload["basis"].float().to(device)
        postrope_basis_metadata = dict(basis_payload.get("metadata", {}))
    if router_bundle is not None and args.router_error_threshold is not None:
        router_bundle["relative_error_threshold"] = args.router_error_threshold
    if args.router_test_only:
        if router_bundle is None:
            raise ValueError("--router_test_only requires --router_bundle")
        test_query_ids = set(int(item) for item in router_bundle["test_query_ids"])
        queries = [query for query in queries if int(query["query_id"]) in test_query_ids]
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    if args.proposal_multiplier < 1:
        raise ValueError("--proposal_multiplier must be positive")
    if postrope_basis is not None:
        expected_shape = (
            len(model.model.layers),
            model.config.num_key_value_heads,
            model.model.layers[0].self_attn.head_dim,
        )
        if tuple(postrope_basis.shape[:3]) != expected_shape:
            raise ValueError(
                f"post-RoPE basis shape {tuple(postrope_basis.shape)} does not match "
                f"model prefix {expected_shape}"
            )
    sparse_layers = parse_layers(args.sparse_layers, len(model.model.layers))
    rows: list[dict[str, Any]] = []

    for query in queries:
        context_array = np.asarray(
            blocks[
                int(query["block_start"]) : int(query["block_start"]) + int(query["block_count"])
            ],
            dtype=np.int64,
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
        prompt_ids = torch.cat([context_ids, prefix_ids, question_ids, suffix_ids])
        gold_ids = answer_token_ids(tokenizer, query.get("answers", []))
        if args.answer_tokens > 0:
            gold_ids = gold_ids[: args.answer_tokens]
        target_ids = torch.tensor(gold_ids, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, target_ids[:-1]])
        query_positions = list(
            range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(target_ids))
        )
        full_nll: float | None = None
        for action in actions:
            intervention = None
            if action != "full":
                intervention = SparseAttentionReference(
                    model,
                    input_ids,
                    query_positions,
                    len(context_ids),
                    set(int(item) for item in question_ids.tolist()),
                    action,
                    args.block_tokens,
                    args.budget_blocks,
                    args.sink_blocks,
                    args.recent_blocks,
                    args.risk_threshold,
                    router_bundle,
                    postrope_basis,
                    args.proposal_multiplier,
                    sparse_layers,
                )
            with torch.inference_mode():
                hidden_states = model.model(
                    input_ids=input_ids[None].to(device),
                    attention_mask=torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[0, query_positions]
                # Avoid materializing [sequence, vocabulary] logits; only the scored
                # query positions need an LM-head projection.
                logits = model.lm_head(hidden_states).float()
            if intervention is not None:
                intervention.close(model)
                effective_threshold = (
                    float(router_bundle["relative_error_threshold"])
                    if action == "learned_conformal" and router_bundle is not None
                    else args.risk_threshold
                )
                head_summary = summarize_head_rows(
                    intervention.head_rows, effective_threshold
                )
            else:
                context_blocks = math.ceil(len(context_ids) / args.block_tokens)
                head_summary = {
                    "mean_selected_blocks": float(context_blocks),
                    "mean_relative_output_l2": 0.0,
                    "p95_relative_output_l2": 0.0,
                    "violation_rate": 0.0,
                    "action_counts": json.dumps(
                        {
                            "full": len(model.model.layers)
                            * model.config.num_attention_heads
                            * len(query_positions)
                        }
                    ),
                    "mean_physical_gqa_blocks": float(context_blocks),
                    "physical_gqa_saving_rate": 0.0,
                    "mean_layer_global_blocks": float(context_blocks),
                    "max_layer_global_blocks": context_blocks,
                    "mean_layer_global_tokens": float(context_blocks * args.block_tokens),
                    "max_layer_global_tokens": context_blocks * args.block_tokens,
                    "strict_1000_token_violation_rate": float(
                        context_blocks * args.block_tokens > 1000
                    ),
                    "mean_router_upper_bound": 0.0,
                    "p95_router_upper_bound": 0.0,
                    "max_router_upper_bound": 0.0,
                    "router_near_threshold_fraction": 0.0,
                    "mean_router_upper_bound_by_layer": json.dumps({}),
                    "router_sparse_fraction_by_layer": json.dumps({}),
                }
            token_nll = -torch.log_softmax(logits, dim=-1).gather(
                1, target_ids.to(device)[:, None]
            )[:, 0]
            nll = float(token_nll.mean().item())
            if action == "full":
                full_nll = nll
            if full_nll is None:
                raise ValueError("full must be the first requested action")
            row = {
                "query_id": int(query["query_id"]),
                "dataset": str(query["dataset"]),
                "action": action,
                "context_tokens": len(context_ids),
                "target_token_id": int(target_ids[0]),
                "target_token": tokenizer.decode([int(target_ids[0])]),
                "target_token_ids": json.dumps(target_ids.tolist()),
                "target_text": tokenizer.decode(target_ids.tolist()),
                "scored_tokens": len(target_ids),
                "nll": nll,
                "delta_nll_vs_full": nll - full_nll,
                **head_summary,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    with (output_dir / "reference_nll_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    action_summary = []
    for action in actions:
        group = [row for row in rows if row["action"] == action]
        action_summary.append(
            {
                "action": action,
                "queries": len(group),
                "mean_nll": statistics.fmean(float(row["nll"]) for row in group),
                "mean_delta_nll_vs_full": statistics.fmean(
                    float(row["delta_nll_vs_full"]) for row in group
                ),
                "mean_selected_blocks": statistics.fmean(
                    float(row["mean_selected_blocks"]) for row in group
                ),
                "mean_head_relative_output_l2": statistics.fmean(
                    float(row["mean_relative_output_l2"]) for row in group
                ),
                "mean_head_violation_rate": statistics.fmean(
                    float(row["violation_rate"]) for row in group
                ),
                "mean_physical_gqa_blocks": statistics.fmean(
                    float(row["mean_physical_gqa_blocks"]) for row in group
                ),
                "mean_physical_gqa_saving_rate": statistics.fmean(
                    float(row["physical_gqa_saving_rate"]) for row in group
                ),
                "mean_layer_global_blocks": statistics.fmean(
                    float(row["mean_layer_global_blocks"]) for row in group
                ),
                "max_layer_global_blocks": max(
                    int(row["max_layer_global_blocks"]) for row in group
                ),
                "mean_layer_global_tokens": statistics.fmean(
                    float(row["mean_layer_global_tokens"]) for row in group
                ),
                "max_layer_global_tokens": max(
                    int(row["max_layer_global_tokens"]) for row in group
                ),
                "strict_1000_token_violation_rate": statistics.fmean(
                    float(row["strict_1000_token_violation_rate"]) for row in group
                ),
                "mean_router_upper_bound": statistics.fmean(
                    float(row["mean_router_upper_bound"]) for row in group
                ),
            }
        )
    nonfull = [row for row in rows if row["action"] != "full"]
    error = np.asarray([float(row["mean_relative_output_l2"]) for row in nonfull])
    delta_nll = np.asarray([float(row["delta_nll_vs_full"]) for row in nonfull])
    summary = {
        "model": args.model_name_or_path,
        "queries": len(queries),
        "risk_threshold": args.risk_threshold,
        "answer_tokens": args.answer_tokens,
        "sparse_layers": sparse_layers,
        "postrope_basis": args.postrope_basis or "",
        "postrope_basis_metadata": postrope_basis_metadata,
        "proposal_multiplier": args.proposal_multiplier,
        "action_summary": action_summary,
        "mean_head_error_delta_nll_pearson": (
            float(np.corrcoef(error, delta_nll)[0, 1]) if len(nonfull) > 1 else None
        ),
        "note": (
            "Every intervention replaces only the scored prompt/answer query positions' "
            "per-head attention output at each layer. It measures causal teacher-forced "
            "answer NLL but is not a sparse speed implementation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
