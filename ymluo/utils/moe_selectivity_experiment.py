from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
if str(FDONG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FDONG_SCRIPTS_DIR))

from models import MyQwen3ForCausalLM  # noqa: E402
from models.myqwen import MyQwen3HeadMoE, MyQwen3MoE  # noqa: E402
from transformers import AutoConfig  # noqa: E402
try:
    from transformers import get_cosine_schedule_with_warmup  # noqa: E402
except ImportError:  # noqa: E402
    try:
        from transformers.optimization import get_cosine_schedule_with_warmup  # noqa: E402
    except ImportError:  # noqa: E402
        def get_cosine_schedule_with_warmup(  # noqa: E402
            optimizer: torch.optim.Optimizer,
            num_warmup_steps: int,
            num_training_steps: int,
        ) -> torch.optim.lr_scheduler.LambdaLR:
            def lr_lambda(current_step: int) -> float:
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
from utils import HierarchicalPatternData  # noqa: E402


EXPERIMENT_DEFAULTS = {
    "negative_gradient": {
        "run_name": "moe-negative-gradient",
        "single_token_update": False,
        "gate_inhibition_weight": 0.05,
        "attention_cluster_weight": 0.0,
        "expert_repulsion_weight": 0.001,
        "orthogonalize_gate": False,
        "orthogonalize_experts": False,
        "forced_warmup_steps": 0,
        "forced_warmup_router_loss_weight": 0.0,
        "eval_forced_warmup_routing": False,
    },
    "single_token_update": {
        "run_name": "moe-single-token-update",
        "single_token_update": True,
        "gate_inhibition_weight": 0.0,
        "attention_cluster_weight": 0.0,
        "expert_repulsion_weight": 0.0,
        "orthogonalize_gate": False,
        "orthogonalize_experts": False,
        "forced_warmup_steps": 0,
        "forced_warmup_router_loss_weight": 0.0,
        "eval_forced_warmup_routing": False,
    },
    "orthogonal_init": {
        "run_name": "moe-orthogonal-init",
        "single_token_update": False,
        "gate_inhibition_weight": 0.0,
        "attention_cluster_weight": 0.0,
        "expert_repulsion_weight": 0.0,
        "orthogonalize_gate": True,
        "orthogonalize_experts": True,
        "forced_warmup_steps": 0,
        "forced_warmup_router_loss_weight": 0.0,
        "eval_forced_warmup_routing": False,
    },
    "forced_warmup": {
        "run_name": "moe-forced-warmup",
        "single_token_update": False,
        "gate_inhibition_weight": 0.0,
        "attention_cluster_weight": 0.0,
        "expert_repulsion_weight": 0.0,
        "orthogonalize_gate": False,
        "orthogonalize_experts": False,
        "forced_warmup_steps": 100,
        "forced_warmup_router_loss_weight": 1.0,
        "eval_forced_warmup_routing": True,
    },
    "attention_cluster": {
        "run_name": "moe-attention-cluster",
        "single_token_update": False,
        "gate_inhibition_weight": 0.0,
        "attention_cluster_weight": 0.05,
        "expert_repulsion_weight": 0.0,
        "orthogonalize_gate": False,
        "orthogonalize_experts": False,
        "forced_warmup_steps": 0,
        "forced_warmup_router_loss_weight": 0.0,
        "eval_forced_warmup_routing": False,
    },
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser(experiment_type: str) -> argparse.ArgumentParser:
    defaults = EXPERIMENT_DEFAULTS[experiment_type]
    parser = argparse.ArgumentParser(description=f"Train fdong-style MoE selectivity experiment: {experiment_type}")
    parser.add_argument("--experiment_type", default=experiment_type, choices=sorted(EXPERIMENT_DEFAULTS))
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "fdong" / "Qwen3-0.6B"))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--run_name", default=defaults["run_name"])
    parser.add_argument("--init_checkpoint", default="")

    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--synthetic_num_samples", type=int, default=200_000)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_pad_token_id", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", type=str2bool, default=True)

    parser.add_argument("--debug_vocab_size", type=int, default=257)
    parser.add_argument("--debug_hidden_size", type=int, default=128)
    parser.add_argument("--debug_intermediate_size", type=int, default=256)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=3)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=32)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)

    parser.add_argument("--use_moe", type=str2bool, default=True)
    parser.add_argument("--moe_num_unique_experts", type=int, default=4)
    parser.add_argument("--moe_num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--moe_use_common_expert", type=str2bool, default=False)
    parser.add_argument("--moe_common_intermediate_size", type=int, default=-1)
    parser.add_argument("--moe_router_bias", type=str2bool, default=False)
    parser.add_argument("--moe_normalize_topk_prob", type=str2bool, default=True)
    parser.add_argument("--moe_router_input", choices=["hidden", "attention_output"], default="attention_output")
    parser.add_argument("--moe_head_level", type=str2bool, default=True)
    parser.add_argument("--moe_load_balance_loss_weight", type=float, default=0.0)

    parser.add_argument("--single_token_update", type=str2bool, default=defaults["single_token_update"])
    parser.add_argument("--single_token_position", choices=["random", "last", "cycle"], default="random")
    parser.add_argument("--gate_inhibition_weight", type=float, default=defaults["gate_inhibition_weight"])
    parser.add_argument("--gate_inhibition_temperature", type=float, default=1.0)
    parser.add_argument("--attention_cluster_weight", type=float, default=defaults["attention_cluster_weight"])
    parser.add_argument("--attention_cluster_temperature", type=float, default=1.0)
    parser.add_argument(
        "--attention_cluster_topk",
        type=int,
        default=4,
        help="Keep only the top-k attended history tokens per query/head. Use 0 to use all non-self attention.",
    )
    parser.add_argument("--attention_cluster_include_self", type=str2bool, default=False)
    parser.add_argument("--attention_cluster_detach_attention", type=str2bool, default=True)
    parser.add_argument("--attention_cluster_negative_weight", type=float, default=0.0)
    parser.add_argument("--attention_cluster_negative_feature_layer", type=int, default=1)
    parser.add_argument("--attention_cluster_negative_history_only", type=str2bool, default=False)
    parser.add_argument("--expert_repulsion_weight", type=float, default=defaults["expert_repulsion_weight"])
    parser.add_argument("--expert_repulsion_margin", type=float, default=0.0)
    parser.add_argument("--orthogonalize_gate", type=str2bool, default=defaults["orthogonalize_gate"])
    parser.add_argument("--orthogonalize_experts", type=str2bool, default=defaults["orthogonalize_experts"])
    parser.add_argument("--orthogonal_init_mode", choices=["preserve_norm", "unit"], default="preserve_norm")
    parser.add_argument(
        "--orthogonalize_after_checkpoint",
        type=str2bool,
        default=False,
        help="If false, loading --init_checkpoint preserves checkpoint weights without re-orthogonalizing them.",
    )
    parser.add_argument("--forced_warmup_steps", type=int, default=defaults["forced_warmup_steps"])
    parser.add_argument(
        "--forced_warmup_higher_unit_len",
        type=int,
        default=-1,
        help="If -1, use synthetic_block_size ** synthetic_num_hierarchy_layers.",
    )
    parser.add_argument(
        "--forced_warmup_router_loss_weight",
        type=float,
        default=defaults["forced_warmup_router_loss_weight"],
        help="CE loss weight used to train the router toward the forced warmup assignment.",
    )
    parser.add_argument("--eval_forced_warmup_routing", type=str2bool, default=defaults["eval_forced_warmup_routing"])
    return parser


def build_eval_parser(experiment_type: str) -> argparse.ArgumentParser:
    defaults = EXPERIMENT_DEFAULTS[experiment_type]
    parser = argparse.ArgumentParser(description=f"Eval fdong-style MoE selectivity checkpoint: {experiment_type}")
    parser.add_argument("--experiment_type", default=experiment_type, choices=sorted(EXPERIMENT_DEFAULTS))
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "fdong" / "Qwen3-0.6B"))
    parser.add_argument("--ckpt_file", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=False)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--synthetic_num_samples", type=int, default=200_000)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_pad_token_id", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.0)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", type=str2bool, default=True)

    parser.add_argument("--debug_vocab_size", type=int, default=257)
    parser.add_argument("--debug_hidden_size", type=int, default=128)
    parser.add_argument("--debug_intermediate_size", type=int, default=256)
    parser.add_argument("--debug_num_hidden_layers", type=int, default=3)
    parser.add_argument("--debug_num_attention_heads", type=int, default=4)
    parser.add_argument("--debug_num_key_value_heads", type=int, default=2)
    parser.add_argument("--debug_head_dim", type=int, default=32)
    parser.add_argument("--debug_max_position_embeddings", type=int, default=256)
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=None)
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)

    parser.add_argument("--use_moe", type=str2bool, default=True)
    parser.add_argument("--moe_num_unique_experts", type=int, default=4)
    parser.add_argument("--moe_num_experts_per_tok", type=int, default=1)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--moe_use_common_expert", type=str2bool, default=False)
    parser.add_argument("--moe_common_intermediate_size", type=int, default=-1)
    parser.add_argument("--moe_router_bias", type=str2bool, default=False)
    parser.add_argument("--moe_normalize_topk_prob", type=str2bool, default=True)
    parser.add_argument("--moe_router_input", choices=["hidden", "attention_output"], default="attention_output")
    parser.add_argument("--moe_head_level", type=str2bool, default=True)

    parser.add_argument("--forced_warmup_steps", type=int, default=defaults["forced_warmup_steps"])
    parser.add_argument("--forced_warmup_higher_unit_len", type=int, default=-1)
    parser.add_argument("--eval_forced_warmup_routing", type=str2bool, default=defaults["eval_forced_warmup_routing"])
    return parser


@dataclass
class Batch:
    source: torch.Tensor
    target: torch.Tensor
    metadata: torch.Tensor


class RouterLogitTracker:
    def __init__(self, model: torch.nn.Module) -> None:
        self.enabled = False
        self.logits: list[torch.Tensor] = []
        self.handles = []
        for module in model.modules():
            if isinstance(module, MyQwen3MoE):
                self.handles.append(module.router.register_forward_hook(self._hook))
            elif isinstance(module, MyQwen3HeadMoE):
                for router in module.routers:
                    self.handles.append(router.register_forward_hook(self._hook))

    def _hook(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if self.enabled:
            self.logits.append(output)

    def clear(self) -> None:
        self.logits.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def gate_inhibition_loss(self, temperature: float) -> torch.Tensor | None:
        losses = []
        temp = max(float(temperature), 1e-6)
        for logits in self.logits:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            if flat_logits.shape[-1] <= 1:
                continue
            winners = flat_logits.detach().argmax(dim=-1)
            losses.append(F.cross_entropy(flat_logits.float() / temp, winners))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def target_ce_loss(self, expert_ids: torch.Tensor) -> torch.Tensor | None:
        losses = []
        flat_targets = expert_ids.reshape(-1).long()
        for logits in self.logits:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            if flat_logits.shape[0] != flat_targets.numel():
                continue
            targets = flat_targets.to(flat_logits.device)
            losses.append(F.cross_entropy(flat_logits.float(), targets))
        if not losses:
            return None
        return torch.stack(losses).mean()


def _forced_topk_from_targets(
    routing_weights: torch.Tensor,
    forced_ids: torch.Tensor,
    num_experts_per_tok: int,
    normalize_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    forced_ids = forced_ids.to(routing_weights.device, dtype=torch.long).reshape(-1)
    if num_experts_per_tok == 1:
        topk_indices = forced_ids[:, None]
        topk_weights = torch.ones_like(topk_indices, dtype=routing_weights.dtype)
        return topk_weights, topk_indices

    masked_weights = routing_weights.clone()
    masked_weights.scatter_(1, forced_ids[:, None], -1.0)
    other_weights, other_indices = torch.topk(
        masked_weights,
        k=num_experts_per_tok - 1,
        dim=-1,
    )
    topk_indices = torch.cat([forced_ids[:, None], other_indices], dim=-1)
    forced_weights = routing_weights.gather(1, forced_ids[:, None])
    topk_weights = torch.cat([forced_weights, other_weights.clamp_min(0.0)], dim=-1)
    if normalize_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return topk_weights, topk_indices


def _patched_moe_forward(
    self,
    hidden_states,
    output_expert_labels: bool = False,
    router_hidden_states: torch.Tensor | None = None,
    **kwargs,
):
    original_shape = hidden_states.shape
    flat_states = hidden_states.reshape(-1, self.hidden_size)
    if router_hidden_states is None:
        router_hidden_states = hidden_states
    flat_router_states = router_hidden_states.reshape(-1, self.hidden_size)

    router_logits = self.router(flat_router_states)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    forced_ids = getattr(self, "_forced_expert_ids", None)
    if forced_ids is None:
        topk_weights, topk_indices = torch.topk(
            routing_weights,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        if self.normalize_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    else:
        topk_weights, topk_indices = _forced_topk_from_targets(
            routing_weights,
            forced_ids,
            self.num_experts_per_tok,
            self.normalize_topk_prob,
        )
    topk_weights = topk_weights.to(flat_states.dtype)

    final_states = torch.zeros_like(flat_states)
    for expert_idx, expert in enumerate(self.experts):
        token_idx, slot_idx = torch.where(topk_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_output = expert(flat_states[token_idx])
        final_states[token_idx] += expert_output * topk_weights[token_idx, slot_idx].unsqueeze(-1)

    if self.common_expert is not None:
        final_states = final_states + self.common_expert(flat_states)

    final_states = final_states.reshape(original_shape)
    if output_expert_labels:
        expert_labels = topk_indices.reshape(*original_shape[:-1], self.num_experts_per_tok)
        return final_states, expert_labels
    return final_states


def _patched_head_moe_forward(
    self,
    hidden_states,
    output_expert_labels: bool = False,
    router_hidden_states: torch.Tensor | None = None,
    **kwargs,
):
    if hidden_states.dim() != 4:
        raise ValueError(f"Head-level MoE expects [batch, seq, heads, head_dim], got {tuple(hidden_states.shape)}.")
    if router_hidden_states is None:
        router_hidden_states = hidden_states
    if router_hidden_states.shape != hidden_states.shape:
        raise ValueError(
            "`router_hidden_states` must match head-level expert input shape, "
            f"got {tuple(router_hidden_states.shape)} vs {tuple(hidden_states.shape)}."
        )

    batch, seq_len, num_heads, head_dim = hidden_states.shape
    if num_heads != self.num_heads or head_dim != self.head_dim:
        raise ValueError(
            f"Expected {self.num_heads} heads with dim {self.head_dim}, got {num_heads} heads with dim {head_dim}."
        )

    forced_ids = getattr(self, "_forced_expert_ids", None)
    head_outputs = []
    head_labels = []
    for head_idx in range(self.num_heads):
        head_states = hidden_states[:, :, head_idx, :].reshape(-1, self.head_dim)
        head_router_states = router_hidden_states[:, :, head_idx, :].reshape(-1, self.head_dim)
        router_logits = self.routers[head_idx](head_router_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        if forced_ids is None:
            topk_weights, topk_indices = torch.topk(
                routing_weights,
                k=self.num_experts_per_tok,
                dim=-1,
            )
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            topk_weights, topk_indices = _forced_topk_from_targets(
                routing_weights,
                forced_ids,
                self.num_experts_per_tok,
                self.normalize_topk_prob,
            )
        topk_weights = topk_weights.to(head_states.dtype)

        final_states = torch.zeros_like(head_states)
        for expert_idx, expert in enumerate(self.experts[head_idx]):
            token_idx, slot_idx = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(head_states[token_idx])
            final_states[token_idx] += expert_output * topk_weights[token_idx, slot_idx].unsqueeze(-1)

        if self.common_experts is not None:
            final_states = final_states + self.common_experts[head_idx](head_states)

        head_outputs.append(final_states.reshape(batch, seq_len, self.head_dim))
        if output_expert_labels:
            head_labels.append(topk_indices.reshape(batch, seq_len, self.num_experts_per_tok))

    output = torch.stack(head_outputs, dim=2)
    if output_expert_labels:
        expert_labels = torch.stack(head_labels, dim=2)
        return output, expert_labels
    return output


def install_forced_routing_patch() -> None:
    if getattr(MyQwen3MoE, "_ymluo_forced_routing_patch", False):
        return
    MyQwen3MoE.forward = _patched_moe_forward
    MyQwen3HeadMoE.forward = _patched_head_moe_forward
    MyQwen3MoE._ymluo_forced_routing_patch = True
    MyQwen3HeadMoE._ymluo_forced_routing_patch = True


def clear_forced_expert_ids(model: torch.nn.Module) -> None:
    for module in iter_moe_modules(model):
        if hasattr(module, "_forced_expert_ids"):
            delattr(module, "_forced_expert_ids")


def set_forced_expert_ids(model: torch.nn.Module, expert_ids: torch.Tensor) -> None:
    for module in iter_moe_modules(model):
        module._forced_expert_ids = expert_ids


def apply_debug_model_overrides(config: Any, args: argparse.Namespace) -> None:
    overrides = {
        "vocab_size": args.debug_vocab_size,
        "hidden_size": args.debug_hidden_size,
        "intermediate_size": args.debug_intermediate_size,
        "num_hidden_layers": args.debug_num_hidden_layers,
        "num_attention_heads": args.debug_num_attention_heads,
        "num_key_value_heads": args.debug_num_key_value_heads,
        "head_dim": args.debug_head_dim,
        "max_position_embeddings": args.debug_max_position_embeddings,
    }
    for name, value in overrides.items():
        if value != -1:
            setattr(config, name, value)
    if args.debug_vocab_size != -1:
        if getattr(config, "pad_token_id", None) is None or config.pad_token_id >= args.debug_vocab_size:
            config.pad_token_id = 0
        if getattr(config, "bos_token_id", None) is not None and config.bos_token_id >= args.debug_vocab_size:
            config.bos_token_id = 1
        if getattr(config, "eos_token_id", None) is not None and config.eos_token_id >= args.debug_vocab_size:
            config.eos_token_id = 2


def apply_moe_overrides(config: Any, args: argparse.Namespace) -> None:
    config.use_moe = bool(args.use_moe)
    config.moe_num_unique_experts = int(args.moe_num_unique_experts)
    config.moe_num_experts_per_tok = int(args.moe_num_experts_per_tok)
    config.moe_intermediate_size = (
        int(args.moe_intermediate_size)
        if args.moe_intermediate_size != -1
        else int(config.intermediate_size)
    )
    config.moe_use_common_expert = bool(args.moe_use_common_expert)
    config.moe_common_intermediate_size = (
        int(args.moe_common_intermediate_size)
        if args.moe_common_intermediate_size != -1
        else int(config.moe_intermediate_size)
    )
    config.moe_router_bias = bool(args.moe_router_bias)
    config.moe_normalize_topk_prob = bool(args.moe_normalize_topk_prob)
    config.moe_router_input = str(args.moe_router_input)
    config.moe_head_level = bool(args.moe_head_level)


def resolve_layer_pattern(pattern: list[int] | None, num_layers: int, default_value: int) -> list[int]:
    if pattern is None:
        return [default_value for _ in range(num_layers)]
    if len(pattern) != num_layers:
        raise ValueError(f"Layer pattern must have length {num_layers}, got {len(pattern)}")
    return [int(value) for value in pattern]


def prepare_dataset(args: argparse.Namespace) -> HierarchicalPatternData:
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.synthetic_num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        pad_token_id=args.synthetic_pad_token_id,
        min_token_id=args.synthetic_min_token_id,
        sampling_distribution=args.synthetic_sampling_distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        return_metadata=True,
    )


def sample_batch(dataset: HierarchicalPatternData, args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> Batch:
    indices = torch.randint(0, len(dataset), (args.batch_size,), generator=generator)
    items = [dataset[int(index)] for index in indices.tolist()]
    source = torch.stack([item[0] for item in items]).to(device)
    target = torch.stack([item[1] for item in items]).to(device)
    metadata = torch.stack([item[3] for item in items]).to(device)
    return Batch(source=source, target=target, metadata=metadata)


def iter_moe_modules(model: torch.nn.Module) -> Iterable[MyQwen3MoE | MyQwen3HeadMoE]:
    for module in model.modules():
        if isinstance(module, (MyQwen3MoE, MyQwen3HeadMoE)):
            yield module


def flatten_module_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1).float() for param in module.parameters()])


def assign_flat_parameters(module: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for param in module.parameters():
        numel = param.numel()
        chunk = vector[offset : offset + numel].reshape_as(param).to(device=param.device, dtype=param.dtype)
        param.data.copy_(chunk)
        offset += numel


def orthogonalize_vectors(vectors: list[torch.Tensor], mode: str) -> list[torch.Tensor]:
    orthogonal: list[torch.Tensor] = []
    norms = [vector.norm().clamp_min(1e-12) for vector in vectors]
    for idx, vector in enumerate(vectors):
        candidate = vector.float().clone()
        for previous in orthogonal:
            candidate = candidate - torch.dot(candidate, previous) * previous
        norm = candidate.norm()
        if float(norm) < 1e-8:
            candidate = torch.randn_like(candidate)
            for previous in orthogonal:
                candidate = candidate - torch.dot(candidate, previous) * previous
            norm = candidate.norm().clamp_min(1e-12)
        candidate = candidate / norm
        if mode == "preserve_norm":
            candidate = candidate * norms[idx]
        orthogonal.append(candidate)
    return orthogonal


def orthogonalize_linear_rows(linear: torch.nn.Linear, mode: str) -> None:
    rows = [row.detach().float().clone() for row in linear.weight.data]
    orthogonal = orthogonalize_vectors(rows, mode)
    for idx, row in enumerate(orthogonal):
        linear.weight.data[idx].copy_(row.to(device=linear.weight.device, dtype=linear.weight.dtype))


def orthogonalize_expert_list(experts: Iterable[torch.nn.Module], mode: str) -> None:
    expert_list = list(experts)
    if len(expert_list) <= 1:
        return
    vectors = [flatten_module_parameters(expert).cpu() for expert in expert_list]
    orthogonal = orthogonalize_vectors(vectors, mode)
    for expert, vector in zip(expert_list, orthogonal):
        assign_flat_parameters(expert, vector)


def apply_orthogonal_initialization(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, int]:
    counts = {"gate_groups": 0, "expert_groups": 0}
    for module in iter_moe_modules(model):
        if isinstance(module, MyQwen3MoE):
            if args.orthogonalize_gate:
                orthogonalize_linear_rows(module.router, args.orthogonal_init_mode)
                counts["gate_groups"] += 1
            if args.orthogonalize_experts:
                orthogonalize_expert_list(module.experts, args.orthogonal_init_mode)
                counts["expert_groups"] += 1
        elif isinstance(module, MyQwen3HeadMoE):
            if args.orthogonalize_gate:
                for router in module.routers:
                    orthogonalize_linear_rows(router, args.orthogonal_init_mode)
                    counts["gate_groups"] += 1
            if args.orthogonalize_experts:
                for per_head_experts in module.experts:
                    orthogonalize_expert_list(per_head_experts, args.orthogonal_init_mode)
                    counts["expert_groups"] += 1
    return counts


def prepare_model(args: argparse.Namespace, device: torch.device) -> MyQwen3ForCausalLM:
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    apply_debug_model_overrides(config, args)
    apply_moe_overrides(config, args)
    config._attn_implementation = args.attn_implementation
    config.attention_stride_pattern = resolve_layer_pattern(
        args.attention_stride_pattern,
        config.num_hidden_layers,
        1,
    )
    config.residual_source_pattern = resolve_layer_pattern(
        args.residual_source_pattern,
        config.num_hidden_layers,
        -1,
    )

    model = MyQwen3ForCausalLM(config).to(device)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    if args.init_checkpoint and not args.orthogonalize_after_checkpoint:
        init_counts = {"gate_groups": 0, "expert_groups": 0}
    else:
        init_counts = apply_orthogonal_initialization(model, args)
    print(
        "model: "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} kv_heads={config.num_key_value_heads} "
        f"vocab={config.vocab_size} use_moe={config.use_moe} "
        f"head_level={config.moe_head_level} router_input={config.moe_router_input}",
        flush=True,
    )
    print(f"attention_stride_pattern={config.attention_stride_pattern}", flush=True)
    print(f"orthogonal_init_counts={init_counts}", flush=True)
    model.train()
    return model


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.use_bf16 and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def choose_single_token_position(args: argparse.Namespace, step: int, seq_len: int, generator: torch.Generator) -> int:
    if args.single_token_position == "last":
        return seq_len - 1
    if args.single_token_position == "cycle":
        return (step - 1) % seq_len
    return int(torch.randint(0, seq_len, (1,), generator=generator).item())


def forced_higher_unit_len(args: argparse.Namespace) -> int:
    if args.forced_warmup_higher_unit_len > 0:
        return int(args.forced_warmup_higher_unit_len)
    return int(args.synthetic_block_size ** args.synthetic_num_hierarchy_layers)


def build_forced_warmup_expert_ids(batch: Batch, args: argparse.Namespace, model: MyQwen3ForCausalLM) -> torch.Tensor:
    higher_unit_len = max(forced_higher_unit_len(args), 1)
    num_experts = int(model.config.moe_num_unique_experts)
    positions = torch.arange(batch.source.shape[1], device=batch.source.device)
    per_position = torch.div(positions, higher_unit_len, rounding_mode="floor").remainder(num_experts)
    return per_position.unsqueeze(0).expand(batch.source.shape[0], -1).contiguous()


def higher_occurrence_ids(batch: Batch, args: argparse.Namespace) -> torch.Tensor:
    higher_unit_len = max(forced_higher_unit_len(args), 1)
    positions = torch.arange(batch.source.shape[1], device=batch.source.device)
    occurrence_ids = torch.div(positions, higher_unit_len, rounding_mode="floor")
    return occurrence_ids.unsqueeze(0).expand(batch.source.shape[0], -1).contiguous()


def compute_prediction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    step: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, int, int]:
    if args.single_token_update:
        batch_index = int(torch.randint(0, target.shape[0], (1,), generator=generator).item())
        position = choose_single_token_position(args, step, target.shape[1], generator)
        token_logits = logits[batch_index, position].unsqueeze(0)
        token_target = target[batch_index, position].unsqueeze(0)
        loss = F.cross_entropy(token_logits.float(), token_target)
        correct = int(token_logits.argmax(dim=-1).eq(token_target).sum().detach().cpu())
        return loss, correct, 1, position

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1))
    correct = int(logits.argmax(dim=-1).eq(target).sum().detach().cpu())
    return loss, correct, int(target.numel()), -1


def expert_repulsion_loss(model: torch.nn.Module, margin: float) -> torch.Tensor | None:
    losses = []
    for module in iter_moe_modules(model):
        groups = []
        if isinstance(module, MyQwen3MoE):
            groups.append(list(module.experts))
        elif isinstance(module, MyQwen3HeadMoE):
            groups.extend(list(per_head) for per_head in module.experts)
        for experts in groups:
            if len(experts) <= 1:
                continue
            vectors = torch.stack([torch.cat([param.reshape(-1).float() for param in expert.parameters()]) for expert in experts])
            vectors = F.normalize(vectors, dim=-1)
            sim = vectors @ vectors.t()
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            losses.append(F.relu(sim[mask] + float(margin)).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def attention_history_mass(attn: torch.Tensor, feature_ids: torch.Tensor) -> float | None:
    batch, heads, seq_len, _ = attn.shape
    history = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn.device), diagonal=-1)
    same = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid = (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    mask = same & valid & history[None, :, :]
    valid_rows = mask.any(dim=-1)
    if not bool(valid_rows.any()):
        return None
    mass = (attn.float() * mask[:, None, :, :].float()).sum(dim=-1)
    row_mask = valid_rows[:, None, :].expand(batch, heads, seq_len)
    return float(mass[row_mask].mean().detach().cpu())


def prepare_attention_cluster_weights(attn: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weights = attn.float()
    if args.attention_cluster_detach_attention:
        weights = weights.detach()
    seq_len = weights.shape[-1]
    if not args.attention_cluster_include_self:
        eye = torch.eye(seq_len, dtype=torch.bool, device=weights.device)
        weights = weights.masked_fill(eye[None, None, :, :], 0.0)
    if args.attention_cluster_topk and args.attention_cluster_topk > 0:
        topk = min(int(args.attention_cluster_topk), seq_len)
        top_values, top_indices = torch.topk(weights, k=topk, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_indices, top_values)
        weights = sparse_weights
    return weights


def attention_weighted_router_consistency_loss(
    router_probs: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor | None:
    if router_probs.dim() != 3:
        raise ValueError(f"router_probs must have shape [batch, seq, experts], got {tuple(router_probs.shape)}")
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[:, None, :, :]
    same_expert_prob = torch.einsum("bqe,bke->bqk", router_probs.float(), router_probs.float())
    same_expert_prob = same_expert_prob[:, None, :, :]
    denom = attention_weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return None
    return -(attention_weights * torch.log(same_expert_prob.clamp_min(1e-8))).sum() / denom.clamp_min(1e-8)


def attention_cluster_loss(
    router_logits: list[torch.Tensor],
    attentions: tuple[torch.Tensor, ...] | None,
    model: MyQwen3ForCausalLM,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if not router_logits or not attentions:
        return None
    losses = []
    num_layers = len(attentions)
    num_heads = int(model.config.num_attention_heads)
    num_experts = int(model.config.moe_num_unique_experts)
    moe_head_level = bool(model.config.moe_head_level)
    temp = max(float(args.attention_cluster_temperature), 1e-6)
    if moe_head_level:
        if len(router_logits) < num_layers * num_heads:
            return None
        logit_idx = 0
        for layer_idx in range(num_layers):
            weights = prepare_attention_cluster_weights(attentions[layer_idx], args)
            batch, _, seq_len, _ = weights.shape
            head_losses = []
            for head_idx in range(num_heads):
                logits = router_logits[logit_idx].reshape(batch, seq_len, num_experts)
                logit_idx += 1
                probs = F.softmax(logits.float() / temp, dim=-1)
                head_loss = attention_weighted_router_consistency_loss(
                    probs,
                    weights[:, head_idx],
                )
                if head_loss is not None:
                    head_losses.append(head_loss)
            if head_losses:
                losses.append(torch.stack(head_losses).mean())
    else:
        if len(router_logits) < num_layers:
            return None
        for layer_idx in range(num_layers):
            weights = prepare_attention_cluster_weights(attentions[layer_idx], args)
            batch, _, seq_len, _ = weights.shape
            logits = router_logits[layer_idx].reshape(batch, seq_len, num_experts)
            probs = F.softmax(logits.float() / temp, dim=-1)
            loss = attention_weighted_router_consistency_loss(
                probs,
                weights,
            )
            if loss is not None:
                losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def negative_router_pair_loss_for_probs(
    router_probs: torch.Tensor,
    feature_ids: torch.Tensor,
    history_only: bool,
) -> torch.Tensor | None:
    if router_probs.dim() != 3:
        raise ValueError(f"router_probs must have shape [batch, seq, experts], got {tuple(router_probs.shape)}")
    same_expert_prob = torch.einsum("bqe,bke->bqk", router_probs.float(), router_probs.float())
    different_feature = feature_ids[:, :, None] != feature_ids[:, None, :]
    valid = (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    not_self = ~torch.eye(feature_ids.shape[1], dtype=torch.bool, device=feature_ids.device)[None, :, :]
    mask = different_feature & valid & not_self
    if history_only:
        history = torch.tril(
            torch.ones(feature_ids.shape[1], feature_ids.shape[1], dtype=torch.bool, device=feature_ids.device),
            diagonal=-1,
        )
        mask = mask & history[None, :, :]
    if not bool(mask.any()):
        return None
    return -torch.log((1.0 - same_expert_prob[mask]).clamp_min(1e-8)).mean()


def attention_cluster_negative_pair_loss(
    router_logits: list[torch.Tensor],
    metadata: torch.Tensor,
    model: MyQwen3ForCausalLM,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if not router_logits:
        return None
    feature_layer = int(args.attention_cluster_negative_feature_layer)
    if feature_layer < 0:
        return None
    feature_layer = min(feature_layer, metadata.shape[-1] - 1)
    feature_ids = metadata[:, :, feature_layer]
    losses = []
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    num_experts = int(model.config.moe_num_unique_experts)
    moe_head_level = bool(model.config.moe_head_level)
    temp = max(float(args.attention_cluster_temperature), 1e-6)
    if moe_head_level:
        if len(router_logits) < num_layers * num_heads:
            return None
        logit_idx = 0
        for _layer_idx in range(num_layers):
            head_losses = []
            batch = metadata.shape[0]
            seq_len = metadata.shape[1]
            for _head_idx in range(num_heads):
                logits = router_logits[logit_idx].reshape(batch, seq_len, num_experts)
                logit_idx += 1
                probs = F.softmax(logits.float() / temp, dim=-1)
                loss = negative_router_pair_loss_for_probs(
                    probs,
                    feature_ids,
                    bool(args.attention_cluster_negative_history_only),
                )
                if loss is not None:
                    head_losses.append(loss)
            if head_losses:
                losses.append(torch.stack(head_losses).mean())
    else:
        if len(router_logits) < num_layers:
            return None
        batch = metadata.shape[0]
        seq_len = metadata.shape[1]
        for layer_idx in range(num_layers):
            logits = router_logits[layer_idx].reshape(batch, seq_len, num_experts)
            probs = F.softmax(logits.float() / temp, dim=-1)
            loss = negative_router_pair_loss_for_probs(
                probs,
                feature_ids,
                bool(args.attention_cluster_negative_history_only),
            )
            if loss is not None:
                losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def same_feature_same_expert_rate(labels: torch.Tensor, feature_ids: torch.Tensor) -> float | None:
    if labels is None:
        return None
    primary = labels[..., 0] if labels.shape[-1] != feature_ids.shape[-1] else labels
    if primary.dim() == 3:
        head_rates = []
        for head_idx in range(primary.shape[2]):
            rate = same_feature_same_expert_rate(primary[:, :, head_idx], feature_ids)
            if rate is not None:
                head_rates.append(rate)
        if not head_rates:
            return None
        return float(sum(head_rates) / len(head_rates))

    same_feature = feature_ids[:, :, None] == feature_ids[:, None, :]
    valid = same_feature & (feature_ids[:, :, None] >= 0) & (feature_ids[:, None, :] >= 0)
    not_self = ~torch.eye(feature_ids.shape[1], dtype=torch.bool, device=feature_ids.device)[None, :, :]
    valid = valid & not_self
    if not bool(valid.any()):
        return None
    same_expert = primary[:, :, None] == primary[:, None, :]
    return float((same_expert & valid).float().sum().detach().cpu() / valid.float().sum().detach().cpu().clamp_min(1.0))


def expert_load_metrics(labels: torch.Tensor, num_experts: int) -> dict[str, Any]:
    if labels is None:
        return {}
    primary = labels[..., 0]
    counts = torch.bincount(primary.reshape(-1).long().cpu(), minlength=num_experts).float()
    total = counts.sum().clamp_min(1.0)
    load = counts / total
    entropy = -(load[load > 0] * load[load > 0].log()).sum()
    normalized_entropy = entropy / math.log(max(num_experts, 2))
    return {
        "expert_load": [float(value) for value in load.tolist()],
        "expert_load_max": float(load.max().item()),
        "expert_load_min": float(load.min().item()),
        "expert_load_entropy": float(normalized_entropy.item()),
        "expert_load_nonzero_fraction": float((counts > 0).float().mean().item()),
    }


def aggregate_eval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_loss_sum = sum(row["loss_sum"] for row in rows)
    total_count = sum(row["count"] for row in rows)
    total_correct = sum(row["correct"] for row in rows)
    layers: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        for layer_idx, layer_metrics in row["layers"].items():
            bucket = layers.setdefault(layer_idx, {})
            for key, value in layer_metrics.items():
                if value is not None:
                    bucket.setdefault(key, []).append(value)
    layer_summary = {}
    for layer_idx, layer_values in layers.items():
        layer_summary[layer_idx] = {}
        for key, values in layer_values.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, list):
                width = len(first)
                layer_summary[layer_idx][key] = [
                    sum(float(value[idx]) for value in values) / len(values)
                    for idx in range(width)
                ]
            else:
                layer_summary[layer_idx][key] = sum(float(value) for value in values) / len(values)
    mean_summary = {}
    for layer_values in layer_summary.values():
        for key, value in layer_values.items():
            mean_summary.setdefault(key, []).append(value)
    averaged_mean_summary = {}
    for key, values in mean_summary.items():
        if not values:
            continue
        first = values[0]
        if isinstance(first, list):
            width = len(first)
            averaged_mean_summary[key] = [
                sum(float(value[idx]) for value in values) / len(values)
                for idx in range(width)
            ]
        else:
            averaged_mean_summary[key] = sum(float(value) for value in values) / len(values)
    loss = total_loss_sum / max(total_count, 1)
    return {
        "loss": loss,
        "accuracy": total_correct / max(total_count, 1),
        "ppl": math.exp(min(loss, 80.0)),
        "count": total_count,
        "layers": layer_summary,
        "mean_layers": averaged_mean_summary,
    }


@torch.no_grad()
def evaluate(
    model: MyQwen3ForCausalLM,
    dataset: HierarchicalPatternData,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    force_warmup_routing: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    rows = []
    for batch_idx in range(args.eval_batches):
        batch = sample_batch(dataset, args, generator, device)
        if force_warmup_routing:
            set_forced_expert_ids(model, build_forced_warmup_expert_ids(batch, args, model))
        else:
            clear_forced_expert_ids(model)
        with autocast_context(args, device):
            output = model(
                batch.source,
                use_cache=False,
                output_attentions=True,
                output_expert_labels=True,
                output_hidden_states=False,
            )
        logits = output.logits
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch.target.reshape(-1),
            reduction="sum",
        )
        correct = int(logits.argmax(dim=-1).eq(batch.target).sum().detach().cpu())
        row = {
            "loss_sum": float(loss_sum.detach().cpu()),
            "correct": correct,
            "count": int(batch.target.numel()),
            "layers": {},
        }
        attentions = output.attentions or ()
        expert_labels = getattr(output, "expert_labels", None) or ()
        occurrence_ids = higher_occurrence_ids(batch, args)
        for layer_idx in range(len(attentions)):
            layer_labels = expert_labels[layer_idx] if layer_idx < len(expert_labels) else None
            layer_row = {
                "same_higher_same_expert": same_feature_same_expert_rate(
                    layer_labels,
                    batch.metadata[:, :, 1] if batch.metadata.shape[-1] > 1 else batch.metadata[:, :, 0],
                ),
                "same_higher_occurrence_same_expert": same_feature_same_expert_rate(layer_labels, occurrence_ids),
                "same_local_same_expert": same_feature_same_expert_rate(layer_labels, batch.metadata[:, :, 0]),
                "local_slot_history_mass": attention_history_mass(attentions[layer_idx], batch.metadata[:, :, 0]),
                "higher_level_history_mass": attention_history_mass(
                    attentions[layer_idx],
                    batch.metadata[:, :, 1] if batch.metadata.shape[-1] > 1 else batch.metadata[:, :, 0],
                ),
            }
            layer_row.update(expert_load_metrics(layer_labels, int(model.config.moe_num_unique_experts)))
            row["layers"][str(layer_idx)] = layer_row
        rows.append(row)
    clear_forced_expert_ids(model)
    if was_training:
        model.train()
    return aggregate_eval_metrics(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_layer_metric(layers: dict[str, dict[str, Any]], key: str) -> str:
    parts = []
    for layer_idx in sorted(layers, key=lambda value: int(value)):
        value = layers[layer_idx].get(key)
        if value is None:
            continue
        if isinstance(value, list):
            formatted = "[" + ",".join(f"{item:.3f}" for item in value) + "]"
        else:
            formatted = f"{float(value):.4f}"
        parts.append(f"L{layer_idx}:{formatted}")
    return " ".join(parts)


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: MyQwen3ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{step}.pth")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / f"{step}.optim.pth",
    )
    runtime_config = {
        "attention_stride_pattern": list(model.model.attention_stride_pattern),
        "residual_source_pattern": list(model.model.residual_source_pattern),
        "experiment_type": args.experiment_type,
        "use_moe": bool(model.config.use_moe),
        "moe_num_unique_experts": int(model.config.moe_num_unique_experts),
        "moe_num_experts_per_tok": int(model.config.moe_num_experts_per_tok),
        "moe_router_input": str(model.config.moe_router_input),
        "moe_head_level": bool(model.config.moe_head_level),
    }
    (ckpt_dir / "runtime_config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {ckpt_dir / f'{step}.pth'}", flush=True)


def run_experiment(experiment_type: str, argv: list[str] | None = None) -> None:
    parser = build_parser(experiment_type)
    args = parser.parse_args(argv)
    if args.output_dir == "":
        args.output_dir = str(REPO_ROOT / "ymluo" / "projects" / f"qwen3_moe_{experiment_type}" / "outputs" / "train")
    if args.total_steps < 1:
        raise ValueError("--total_steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.synthetic_num_hierarchy_layers < 2:
        raise ValueError("These metrics expect at least 2 synthetic hierarchy layers.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    run_dir = Path(args.output_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"experiment_type={args.experiment_type}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(
        "synthetic: "
        f"seq_len={args.seq_len} block={args.synthetic_block_size} "
        f"layers={args.synthetic_num_hierarchy_layers} units={args.synthetic_num_units_per_layer} "
        f"distribution={args.synthetic_sampling_distribution}",
        flush=True,
    )

    dataset = prepare_dataset(args)
    if args.forced_warmup_steps > 0:
        install_forced_routing_patch()
    model = prepare_model(args, device)
    tracker = RouterLogitTracker(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)

    started_at = time.monotonic()
    rolling = {
        "loss_sum": 0.0,
        "prediction_loss_sum": 0.0,
        "gate_inhibition_loss_sum": 0.0,
        "attention_cluster_loss_sum": 0.0,
        "attention_negative_loss_sum": 0.0,
        "load_balance_loss_sum": 0.0,
        "expert_repulsion_loss_sum": 0.0,
        "forced_router_loss_sum": 0.0,
        "correct": 0,
        "count": 0,
        "selected_position_sum": 0,
        "selected_position_count": 0,
    }

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.total_steps + 1):
        for _ in range(args.gradient_accumulation_steps):
            batch = sample_batch(dataset, args, train_generator, device)
            forced_ids = None
            if args.forced_warmup_steps > 0 and step <= args.forced_warmup_steps:
                forced_ids = build_forced_warmup_expert_ids(batch, args, model)
                set_forced_expert_ids(model, forced_ids)
            else:
                clear_forced_expert_ids(model)
            tracker.clear()
            tracker.enabled = (
                args.gate_inhibition_weight > 0.0
                or args.attention_cluster_weight > 0.0
                or args.attention_cluster_negative_weight > 0.0
                or (forced_ids is not None and args.forced_warmup_router_loss_weight > 0.0)
            )
            use_load_balance = args.moe_load_balance_loss_weight > 0.0
            with autocast_context(args, device):
                output = model(
                    batch.source,
                    use_cache=False,
                    output_attentions=args.attention_cluster_weight > 0.0,
                    output_expert_labels=False,
                    output_hidden_states=False,
                    output_router_aux_loss=use_load_balance,
                )
                prediction_loss, correct, count, selected_position = compute_prediction_loss(
                    output.logits,
                    batch.target,
                    args,
                    step,
                    train_generator,
                )
                total_loss = prediction_loss
                gate_loss_value = None
                if args.gate_inhibition_weight > 0.0:
                    gate_loss_value = tracker.gate_inhibition_loss(args.gate_inhibition_temperature)
                    if gate_loss_value is not None:
                        total_loss = total_loss + args.gate_inhibition_weight * gate_loss_value
                forced_router_loss_value = None
                if forced_ids is not None and args.forced_warmup_router_loss_weight > 0.0:
                    forced_router_loss_value = tracker.target_ce_loss(forced_ids)
                    if forced_router_loss_value is not None:
                        total_loss = total_loss + args.forced_warmup_router_loss_weight * forced_router_loss_value
                attention_cluster_loss_value = None
                if args.attention_cluster_weight > 0.0:
                    attention_cluster_loss_value = attention_cluster_loss(
                        tracker.logits,
                        output.attentions,
                        model,
                        args,
                    )
                    if attention_cluster_loss_value is not None:
                        total_loss = total_loss + args.attention_cluster_weight * attention_cluster_loss_value
                attention_negative_loss_value = None
                if args.attention_cluster_negative_weight > 0.0:
                    attention_negative_loss_value = attention_cluster_negative_pair_loss(
                        tracker.logits,
                        batch.metadata,
                        model,
                        args,
                    )
                    if attention_negative_loss_value is not None:
                        total_loss = total_loss + args.attention_cluster_negative_weight * attention_negative_loss_value
                load_balance_loss_value = None
                if use_load_balance:
                    load_balance_loss_value = getattr(output, "moe_load_balance_loss", None)
                    if load_balance_loss_value is not None:
                        total_loss = total_loss + args.moe_load_balance_loss_weight * load_balance_loss_value
                expert_loss_value = None
                if args.expert_repulsion_weight > 0.0:
                    expert_loss_value = expert_repulsion_loss(model, args.expert_repulsion_margin)
                    if expert_loss_value is not None:
                        total_loss = total_loss + args.expert_repulsion_weight * expert_loss_value
            tracker.enabled = False
            (total_loss / args.gradient_accumulation_steps).backward()

            rolling["loss_sum"] += float(total_loss.detach().cpu()) * count
            rolling["prediction_loss_sum"] += float(prediction_loss.detach().cpu()) * count
            if gate_loss_value is not None:
                rolling["gate_inhibition_loss_sum"] += float(gate_loss_value.detach().cpu()) * count
            if forced_router_loss_value is not None:
                rolling["forced_router_loss_sum"] += float(forced_router_loss_value.detach().cpu()) * count
            if attention_cluster_loss_value is not None:
                rolling["attention_cluster_loss_sum"] += float(attention_cluster_loss_value.detach().cpu()) * count
            if attention_negative_loss_value is not None:
                rolling["attention_negative_loss_sum"] += float(attention_negative_loss_value.detach().cpu()) * count
            if load_balance_loss_value is not None:
                rolling["load_balance_loss_sum"] += float(load_balance_loss_value.detach().cpu()) * count
            if expert_loss_value is not None:
                rolling["expert_repulsion_loss_sum"] += float(expert_loss_value.detach().cpu()) * count
            rolling["correct"] += correct
            rolling["count"] += count
            if selected_position >= 0:
                rolling["selected_position_sum"] += selected_position
                rolling["selected_position_count"] += 1

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        clear_forced_expert_ids(model)

        if step % args.log_interval == 0 or step == 1:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            count = max(int(rolling["count"]), 1)
            row = {
                "type": "train",
                "step": step,
                "loss": rolling["loss_sum"] / count,
                "prediction_loss": rolling["prediction_loss_sum"] / count,
                "accuracy": rolling["correct"] / count,
                "gate_inhibition_loss": rolling["gate_inhibition_loss_sum"] / count,
                "attention_cluster_loss": rolling["attention_cluster_loss_sum"] / count,
                "attention_negative_loss": rolling["attention_negative_loss_sum"] / count,
                "load_balance_loss": rolling["load_balance_loss_sum"] / count,
                "forced_router_loss": rolling["forced_router_loss_sum"] / count,
                "expert_repulsion_loss": rolling["expert_repulsion_loss_sum"] / count,
                "count": rolling["count"],
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_second": rolling["count"] / elapsed,
                "mean_selected_position": (
                    None
                    if rolling["selected_position_count"] == 0
                    else rolling["selected_position_sum"] / rolling["selected_position_count"]
                ),
            }
            append_jsonl(metrics_path, row)
            print(
                f"step {step}: loss={row['prediction_loss']:.4f} "
                f"acc={row['accuracy']:.4f} gate_inhib={row['gate_inhibition_loss']:.4f} "
                f"attn_cluster={row['attention_cluster_loss']:.4f} "
                f"attn_neg={row['attention_negative_loss']:.4f} "
                f"load_balance={row['load_balance_loss']:.4f} "
                f"forced_router={row['forced_router_loss']:.4f} "
                f"expert_repulse={row['expert_repulsion_loss']:.4f}",
                flush=True,
            )
            started_at = time.monotonic()
            for key in rolling:
                rolling[key] = 0.0 if key.endswith("_sum") else 0

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_generator_state = eval_generator.get_state()
            eval_metrics = evaluate(model, dataset, args, device, eval_generator)
            forced_eval_metrics = None
            if args.eval_forced_warmup_routing:
                eval_generator.set_state(eval_generator_state)
                forced_eval_metrics = evaluate(
                    model,
                    dataset,
                    args,
                    device,
                    eval_generator,
                    force_warmup_routing=True,
                )
            row = {"type": "eval", "step": step, **eval_metrics}
            if forced_eval_metrics is not None:
                row["forced_oracle"] = forced_eval_metrics
            append_jsonl(metrics_path, row)
            mean_layers = eval_metrics.get("mean_layers", {})
            layers = eval_metrics.get("layers", {})
            print(
                f"step {step}: eval_loss={eval_metrics['loss']:.4f} "
                f"eval_acc={eval_metrics['accuracy']:.4f} "
                f"same_higher_same_expert={mean_layers.get('same_higher_same_expert')} "
                f"higher_mass={mean_layers.get('higher_level_history_mass')} "
                f"expert_load={mean_layers.get('expert_load')}",
                flush=True,
            )
            print(
                f"step {step}: "
                f"same_higher_by_layer={format_layer_metric(layers, 'same_higher_same_expert')} "
                f"same_higher_occurrence_by_layer={format_layer_metric(layers, 'same_higher_occurrence_same_expert')} "
                f"higher_mass_by_layer={format_layer_metric(layers, 'higher_level_history_mass')}",
                flush=True,
            )
            print(
                f"step {step}: expert_load_by_layer={format_layer_metric(layers, 'expert_load')}",
                flush=True,
            )
            if forced_eval_metrics is not None:
                forced_layers = forced_eval_metrics.get("layers", {})
                forced_mean_layers = forced_eval_metrics.get("mean_layers", {})
                print(
                    f"step {step}: forced_oracle_eval_loss={forced_eval_metrics['loss']:.4f} "
                    f"forced_oracle_eval_acc={forced_eval_metrics['accuracy']:.4f} "
                    f"forced_same_higher={forced_mean_layers.get('same_higher_same_expert')} "
                    f"forced_same_higher_occurrence={forced_mean_layers.get('same_higher_occurrence_same_expert')}",
                    flush=True,
                )
                print(
                    f"step {step}: "
                    f"forced_same_higher_by_layer={format_layer_metric(forced_layers, 'same_higher_same_expert')} "
                    f"forced_same_higher_occurrence_by_layer="
                    f"{format_layer_metric(forced_layers, 'same_higher_occurrence_same_expert')}",
                    flush=True,
                )

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, args)

    save_checkpoint(ckpt_dir, args.total_steps, model, optimizer, scheduler, args)
    tracker.close()
    print(f"finished. metrics={metrics_path}", flush=True)


def run_eval_only(experiment_type: str, argv: list[str] | None = None) -> None:
    parser = build_eval_parser(experiment_type)
    args = parser.parse_args(argv)
    args.batch_size = args.eval_batch_size
    args.init_checkpoint = args.ckpt_file
    args.orthogonalize_gate = False
    args.orthogonalize_experts = False
    args.orthogonalize_after_checkpoint = False
    args.single_token_update = False
    args.single_token_position = "random"
    args.gate_inhibition_weight = 0.0
    args.gate_inhibition_temperature = 1.0
    args.expert_repulsion_weight = 0.0
    args.expert_repulsion_margin = 0.0
    args.forced_warmup_router_loss_weight = 0.0
    args.orthogonal_init_mode = "preserve_norm"

    if args.synthetic_num_hierarchy_layers < 2:
        raise ValueError("These metrics expect at least 2 synthetic hierarchy layers.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    if args.eval_forced_warmup_routing:
        install_forced_routing_patch()
    dataset = prepare_dataset(args)
    model = prepare_model(args, device)
    model.eval()
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)

    eval_generator_state = eval_generator.get_state()
    metrics = evaluate(model, dataset, args, device, eval_generator)
    payload: dict[str, Any] = {
        "type": "eval_only",
        "experiment_type": experiment_type,
        "ckpt_file": args.ckpt_file,
        "config": vars(args),
        "metrics": metrics,
    }
    if args.eval_forced_warmup_routing:
        eval_generator.set_state(eval_generator_state)
        payload["forced_oracle"] = evaluate(
            model,
            dataset,
            args,
            device,
            eval_generator,
            force_warmup_routing=True,
        )

    output_path = Path(args.output_path) if args.output_path else Path(args.ckpt_file).with_suffix(".selectivity_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    mean_layers = metrics.get("mean_layers", {})
    print(
        f"eval_only: loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"same_higher={mean_layers.get('same_higher_same_expert')} "
        f"same_higher_occurrence={mean_layers.get('same_higher_occurrence_same_expert')} "
        f"higher_mass={mean_layers.get('higher_level_history_mass')} "
        f"expert_load={mean_layers.get('expert_load')}",
        flush=True,
    )
    print(
        f"eval_only: "
        f"same_higher_by_layer={format_layer_metric(metrics.get('layers', {}), 'same_higher_same_expert')} "
        f"same_higher_occurrence_by_layer="
        f"{format_layer_metric(metrics.get('layers', {}), 'same_higher_occurrence_same_expert')} "
        f"higher_mass_by_layer={format_layer_metric(metrics.get('layers', {}), 'higher_level_history_mass')}",
        flush=True,
    )
    print(
        f"eval_only: expert_load_by_layer={format_layer_metric(metrics.get('layers', {}), 'expert_load')}",
        flush=True,
    )
    if "forced_oracle" in payload:
        forced = payload["forced_oracle"]
        forced_mean = forced.get("mean_layers", {})
        print(
            f"forced_oracle: loss={forced['loss']:.4f} acc={forced['accuracy']:.4f} "
            f"same_higher={forced_mean.get('same_higher_same_expert')} "
            f"same_higher_occurrence={forced_mean.get('same_higher_occurrence_same_expert')}",
            flush=True,
        )
    print(f"wrote eval: {output_path}", flush=True)
