from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise RuntimeError("transformers is required.") from exc


_ORIGINAL_QWEN3_ATTENTION_FORWARD: Any | None = None


@dataclass
class GateConfig:
    target_keep_ratio: float = 0.20
    hard_mode: str = "global_budget"
    threshold: float = 0.5
    temperature: float = 1.0
    sink_tokens_all_heads: int = 64
    z_loss_coef: float = 0.001
    budget_loss_coef: float = 0.05
    load_loss_coef: float = 0.01


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value.detach()
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


def split_globs(globs: str) -> list[str]:
    patterns = [item.strip() for item in globs.split(",") if item.strip()]
    return patterns or ["*.txt"]


def discover_text_files(data_root: Path, globs: str) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"train_data_root does not exist: {data_root}")
    if data_root.is_file():
        return [data_root]
    files: list[Path] = []
    for pattern in split_globs(globs):
        files.extend(path for path in data_root.rglob(pattern) if path.is_file())
    unique = sorted(set(files), key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError(f"No text files found under {data_root} with globs={globs!r}")
    return unique


def iter_text_chunks(path: Path, chunk_chars: int, max_chars: int):
    total_chars = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while max_chars <= 0 or total_chars < max_chars:
            read_size = chunk_chars if max_chars <= 0 else min(chunk_chars, max_chars - total_chars)
            if read_size <= 0:
                break
            text = handle.read(read_size)
            if not text:
                break
            total_chars += len(text)
            yield text


class StreamingTextBatcher:
    def __init__(
        self,
        tokenizer_path: Path,
        files: list[Path],
        seq_len: int,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        chunk_chars: int,
        max_chars_per_file: int,
        shuffle_files: bool,
        max_files_per_rank_epoch: int,
    ) -> None:
        if not files:
            raise ValueError("StreamingTextBatcher requires at least one file.")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = max(1, world_size)
        self.seed = seed
        self.chunk_chars = chunk_chars
        self.max_chars_per_file = max_chars_per_file
        self.shuffle_files = shuffle_files
        self.max_files_per_rank_epoch = max_files_per_rank_epoch
        self.all_rank_files = [path for idx, path in enumerate(files) if idx % self.world_size == rank]
        if not self.all_rank_files:
            raise ValueError(f"rank {rank} received no files from {len(files)} files.")
        self.epoch = 0
        self.files: list[Path] = []
        self.file_index = 0
        self.current_chunks = None
        self.token_buffer: list[int] = []
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        self.files = list(self.all_rank_files)
        if self.shuffle_files:
            rng = random.Random(self.seed + self.rank * 100003 + self.epoch * 9973)
            rng.shuffle(self.files)
        if self.max_files_per_rank_epoch > 0:
            self.files = self.files[: self.max_files_per_rank_epoch]
        if not self.files:
            raise ValueError("No files available after max_files_per_rank_epoch filtering.")
        self.file_index = 0
        self.current_chunks = None
        self.epoch += 1

    def _open_next_file(self) -> None:
        if self.file_index >= len(self.files):
            self._reset_epoch()
        path = self.files[self.file_index]
        self.file_index += 1
        self.current_chunks = iter_text_chunks(path, self.chunk_chars, self.max_chars_per_file)

    def _fill(self, target_tokens: int) -> None:
        while len(self.token_buffer) < target_tokens:
            if self.current_chunks is None:
                self._open_next_file()
            try:
                text = next(self.current_chunks)
            except StopIteration:
                self.current_chunks = None
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                self.token_buffer.extend(int(token_id) for token_id in ids)

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.batch_size * (self.seq_len + 1)
        self._fill(needed)
        flat = self.token_buffer[:needed]
        del self.token_buffer[:needed]
        batch = np.asarray(flat, dtype=np.int64).reshape(self.batch_size, self.seq_len + 1)
        tensor = torch.from_numpy(batch).to(device=device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:]


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def add_kv_gates(model: nn.Module, config: GateConfig) -> None:
    hidden_size = int(model.config.hidden_size)
    kv_heads = int(model.config.num_key_value_heads)
    init_bias = math.log(config.target_keep_ratio / max(1e-6, 1.0 - config.target_keep_ratio))
    for layer in model.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "kv_gate"):
            gate = nn.Linear(hidden_size, kv_heads, bias=True)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, init_bias)
            ref = attn.q_proj.weight
            gate = gate.to(device=ref.device, dtype=ref.dtype)
            attn.add_module("kv_gate", gate)
        attn.kv_gate_config = config
        attn._kv_gate_stats = {}


def gate_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        gate = getattr(module, "kv_gate", None)
        if gate is not None:
            params.extend(list(gate.parameters()))
    return params


def gate_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    return {name: tensor.detach().cpu() for name, tensor in raw.state_dict().items() if ".kv_gate." in name}


def install_qwen3_kv_gate_patch() -> None:
    global _ORIGINAL_QWEN3_ATTENTION_FORWARD
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    if _ORIGINAL_QWEN3_ATTENTION_FORWARD is not None:
        return
    _ORIGINAL_QWEN3_ATTENTION_FORWARD = modeling_qwen3.Qwen3Attention.forward

    def gated_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        if not hasattr(self, "kv_gate"):
            return _ORIGINAL_QWEN3_ATTENTION_FORWARD(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        if past_key_value is not None:
            raise RuntimeError("KV-head gate training currently expects use_cache=False.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cfg: GateConfig = self.kv_gate_config
        gate_logits = self.kv_gate(hidden_states)
        gate_prob = torch.sigmoid(gate_logits / max(cfg.temperature, 1e-6))
        batch_size, seq_len, kv_head_count = gate_prob.shape
        sink_len = min(cfg.sink_tokens_all_heads, seq_len) if cfg.sink_tokens_all_heads > 0 else 0
        if cfg.hard_mode == "global_budget":
            hard = torch.zeros_like(gate_prob, dtype=torch.bool)
            if sink_len > 0:
                hard[:, :sink_len, :] = True
            if sink_len < seq_len:
                non_sink_logits = gate_logits[:, sink_len:, :]
                top1 = non_sink_logits.argmax(dim=-1)
                hard[:, sink_len:, :].scatter_(-1, top1.unsqueeze(-1), True)
                target_slots = int(round(cfg.target_keep_ratio * batch_size * seq_len * kv_head_count))
                target_slots = max(target_slots, int(hard.sum().item()))
                remaining = target_slots - int(hard.sum().item())
                if remaining > 0:
                    candidate_scores = gate_logits.masked_fill(hard, torch.finfo(gate_logits.dtype).min)
                    if sink_len > 0:
                        candidate_scores[:, :sink_len, :] = torch.finfo(gate_logits.dtype).min
                    flat_scores = candidate_scores.reshape(-1)
                    remaining = min(remaining, flat_scores.numel())
                    if remaining > 0:
                        _, flat_idx = torch.topk(flat_scores, k=remaining, largest=True)
                        hard_flat = hard.reshape(-1)
                        hard_flat[flat_idx] = True
                        hard = hard_flat.view_as(hard)
        elif cfg.hard_mode == "threshold":
            hard = gate_prob >= cfg.threshold
            if sink_len > 0:
                hard[:, :sink_len, :] = True
            empty = hard.sum(dim=-1) == 0
            if bool(empty.any()):
                top1 = gate_prob.argmax(dim=-1)
                hard = hard.clone()
                hard.scatter_(-1, top1.unsqueeze(-1), True)
        else:
            raise ValueError(f"unknown gate hard mode: {cfg.hard_mode}")
        gate_st = hard.to(gate_prob.dtype) + gate_prob - gate_prob.detach()
        kv_gate = gate_st.transpose(1, 2).unsqueeze(-1).to(key_states.dtype)
        key_states = key_states * kv_gate
        value_states = value_states * kv_gate

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states_repeated = _repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeated = _repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states_repeated.shape[-2]]
            attn_weights = attn_weights + causal_mask
        key_hard = hard.transpose(1, 2)
        key_hard_repeated = key_hard.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_weights = attn_weights.masked_fill(~key_hard_repeated[:, :, None, :], torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout = 0.0 if not self.training else self.attention_dropout
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_repeated)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        prob_mean = gate_prob.float().mean()
        hard_mean = hard.float().mean()
        head_prob_mean = gate_prob.float().mean(dim=(0, 1))
        target = torch.as_tensor(cfg.target_keep_ratio, device=gate_prob.device, dtype=torch.float32)
        budget_loss = ((prob_mean - target) / target.clamp_min(1e-6)).pow(2)
        load_loss = ((head_prob_mean - target) / target.clamp_min(1e-6)).pow(2).mean()
        z_loss = torch.logsumexp(gate_logits.float(), dim=-1).pow(2).mean()
        self._kv_gate_stats = {
            "budget_loss": budget_loss,
            "load_loss": load_loss,
            "z_loss": z_loss,
            "prob_keep_ratio": prob_mean.detach(),
            "hard_keep_ratio": hard_mean.detach(),
            "hard_heads_per_token": (hard.float().sum(dim=-1).mean()).detach(),
            "head_load": hard.float().mean(dim=(0, 1)).detach(),
        }
        return attn_output, None

    modeling_qwen3.Qwen3Attention.forward = gated_forward


def collect_gate_losses(model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    raw = model.module if isinstance(model, DDP) else model
    budget_losses = []
    load_losses = []
    z_losses = []
    prob_ratios = []
    hard_ratios = []
    heads_per_token = []
    head_loads = []
    for layer in raw.model.layers:
        stats = getattr(layer.self_attn, "_kv_gate_stats", None)
        if not stats:
            continue
        budget_losses.append(stats["budget_loss"])
        load_losses.append(stats["load_loss"])
        z_losses.append(stats["z_loss"])
        prob_ratios.append(stats["prob_keep_ratio"])
        hard_ratios.append(stats["hard_keep_ratio"])
        heads_per_token.append(stats["hard_heads_per_token"])
        head_loads.append(stats["head_load"])
    if not budget_losses:
        zero = torch.zeros((), device=device)
        return {
            "budget_loss": zero,
            "load_loss": zero,
            "z_loss": zero,
            "prob_keep_ratio": zero,
            "hard_keep_ratio": zero,
            "hard_heads_per_token": zero,
            "head_load": torch.zeros((1, 1), device=device),
        }
    return {
        "budget_loss": torch.stack(budget_losses).mean(),
        "load_loss": torch.stack(load_losses).mean(),
        "z_loss": torch.stack(z_losses).mean(),
        "prob_keep_ratio": torch.stack(prob_ratios).mean(),
        "hard_keep_ratio": torch.stack(hard_ratios).mean(),
        "hard_heads_per_token": torch.stack(heads_per_token).mean(),
        "head_load": torch.stack(head_loads),
    }


def cosine_lr(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(output_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace, rank: int) -> None:
    if not is_rank0(rank):
        return
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(raw_model.state_dict(), ckpt_dir / "model_state.pt")
    torch.save(gate_state_dict(raw_model), ckpt_dir / "gate_state.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"step": step, "args": vars(args)}, indent=2), encoding="utf-8")
    latest = output_dir / "latest_checkpoint"
    tmp = output_dir / "latest_checkpoint.tmp"
    tmp.write_text(str(ckpt_dir), encoding="utf-8")
    tmp.replace(latest)
    print(f"saved checkpoint: {ckpt_dir}", flush=True)


def load_checkpoint_if_needed(resume_from: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    if not resume_from:
        return 0
    ckpt_dir = Path(resume_from)
    if ckpt_dir.is_file():
        ckpt_dir = Path(ckpt_dir.read_text(encoding="utf-8").strip())
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(torch.load(ckpt_dir / "model_state.pt", map_location=device), strict=True)
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
    state = json.loads((ckpt_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--train_data_root", default="/mnt/workspace/dclm")
    parser.add_argument("--train_text_glob", default="*.txt")
    parser.add_argument("--output_dir", default="/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_kv_head_gate_training/output/kv_head_gate_runs/run")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--max_train_seconds", type=int, default=72_000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gate_learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_keep_ratio", type=float, default=0.20)
    parser.add_argument("--gate_hard_mode", choices=["global_budget", "threshold"], default="global_budget")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--gate_sink_tokens_all_heads", type=int, default=64)
    parser.add_argument("--budget_loss_coef", type=float, default=0.05)
    parser.add_argument("--load_loss_coef", type=float, default=0.01)
    parser.add_argument("--z_loss_coef", type=float, default=0.001)
    parser.add_argument("--train_base_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream_shuffle_files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream_max_files_per_rank_epoch", type=int, default=0)
    parser.add_argument("--stream_chunk_chars", type=int, default=2_000_000)
    parser.add_argument("--stream_max_chars_per_file", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def tiny_smoke_test() -> None:
    config = GateConfig(target_keep_ratio=0.20)
    logits = torch.randn(2, 32, 8)
    hard = torch.zeros_like(logits, dtype=torch.bool)
    sink_len = 1
    hard[:, :sink_len, :] = True
    top1 = logits[:, sink_len:, :].argmax(dim=-1)
    hard[:, sink_len:, :].scatter_(-1, top1.unsqueeze(-1), True)
    target_slots = int(round(config.target_keep_ratio * hard.numel()))
    remaining = max(0, target_slots - int(hard.sum().item()))
    if remaining:
        candidate_scores = logits.masked_fill(hard, torch.finfo(logits.dtype).min)
        candidate_scores[:, :sink_len, :] = torch.finfo(logits.dtype).min
        _, flat_idx = torch.topk(candidate_scores.reshape(-1), k=remaining)
        hard.reshape(-1)[flat_idx] = True
    assert hard.shape == (2, 32, 8)
    print(json.dumps({"hard_keep_ratio": float(hard.float().mean()), "target": config.target_keep_ratio}, indent=2))


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        tiny_smoke_test()
        return
    install_qwen3_kv_gate_patch()
    rank, local_rank, world_size, device = setup_distributed()
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
            torch.backends.cuda.matmul.allow_tf32 = True

        output_dir = Path(args.output_dir)
        if is_rank0(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        barrier()

        files = discover_text_files(Path(args.train_data_root), args.train_text_glob)
        if is_rank0(rank):
            per_rank_counts = [sum(1 for idx in range(len(files)) if idx % max(1, world_size) == r) for r in range(world_size)]
            (output_dir / "streaming_data_meta.json").write_text(
                json.dumps(
                    {
                        "train_data_root": args.train_data_root,
                        "train_text_glob": args.train_text_glob,
                        "all_file_count": len(files),
                        "world_size": world_size,
                        "per_rank_file_counts": per_rank_counts,
                        "stream_shuffle_files": args.stream_shuffle_files,
                        "stream_max_files_per_rank_epoch": args.stream_max_files_per_rank_epoch,
                        "stream_chunk_chars": args.stream_chunk_chars,
                        "stream_max_chars_per_file": args.stream_max_chars_per_file,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"streaming data: discovered {len(files)} files; per_rank_counts={per_rank_counts}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if is_rank0(rank):
            tokenizer.save_pretrained(output_dir / "tokenizer")

        gate_config = GateConfig(
            target_keep_ratio=args.target_keep_ratio,
            hard_mode=args.gate_hard_mode,
            threshold=args.gate_threshold,
            temperature=args.gate_temperature,
            sink_tokens_all_heads=args.gate_sink_tokens_all_heads,
            z_loss_coef=args.z_loss_coef,
            budget_loss_coef=args.budget_loss_coef,
            load_loss_coef=args.load_loss_coef,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        add_kv_gates(model, gate_config)
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if not args.train_base_model:
            for param in model.parameters():
                param.requires_grad_(False)
            for param in gate_parameters(model):
                param.requires_grad_(True)
        model.to(device)

        gate_param_ids = {id(param) for param in gate_parameters(model)}
        base_params = [param for param in model.parameters() if param.requires_grad and id(param) not in gate_param_ids]
        gate_params = [param for param in model.parameters() if param.requires_grad and id(param) in gate_param_ids]
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": args.learning_rate, "weight_decay": args.weight_decay})
        if gate_params:
            param_groups.append({"params": gate_params, "lr": args.gate_learning_rate, "weight_decay": args.weight_decay})
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        start_step = load_checkpoint_if_needed(args.resume_from, model, optimizer, device)

        batcher = StreamingTextBatcher(
            Path(args.model_name_or_path),
            files,
            args.seq_len,
            args.per_device_batch_size,
            rank,
            world_size,
            args.seed + start_step,
            args.stream_chunk_chars,
            args.stream_max_chars_per_file,
            args.stream_shuffle_files,
            args.stream_max_files_per_rank_epoch,
        )
        writer = SummaryWriter(output_dir / "tensorboard") if (is_rank0(rank) and SummaryWriter is not None) else None
        tokens_per_optim_step = args.seq_len * args.per_device_batch_size * args.gradient_accumulation_steps * world_size
        start_time = time.time()
        if is_rank0(rank):
            print(f"world_size={world_size} device={device} tokens_per_optim_step={tokens_per_optim_step}", flush=True)
            print(f"output_dir={output_dir}", flush=True)
            print(f"train_base_model={args.train_base_model} gate_params={sum(p.numel() for p in gate_params)}", flush=True)

        for step in range(start_step + 1, args.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum = {name: torch.zeros((), device=device) for name in [
                "ce_loss", "budget_loss", "load_loss", "z_loss", "prob_keep_ratio", "hard_keep_ratio", "hard_heads_per_token"
            ]}
            last_head_load = None
            step_started = time.time()
            for micro_step in range(args.gradient_accumulation_steps):
                sync_context = model.no_sync() if isinstance(model, DDP) and micro_step < args.gradient_accumulation_steps - 1 else nullcontext()
                input_ids, labels = batcher.next_batch(device)
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        outputs = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
                        ce_loss = outputs.loss
                        gate_losses = collect_gate_losses(model, device)
                        loss = (
                            ce_loss
                            + args.budget_loss_coef * gate_losses["budget_loss"]
                            + args.load_loss_coef * gate_losses["load_loss"]
                            + args.z_loss_coef * gate_losses["z_loss"]
                        )
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                accum["ce_loss"] += ce_loss.detach() / args.gradient_accumulation_steps
                for key in ["budget_loss", "load_loss", "z_loss", "prob_keep_ratio", "hard_keep_ratio", "hard_heads_per_token"]:
                    accum[key] += gate_losses[key].detach() / args.gradient_accumulation_steps
                last_head_load = gate_losses["head_load"].detach()

            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            lr_scale = cosine_lr(step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                base_lr = args.gate_learning_rate if any(id(p) in gate_param_ids for p in group["params"]) else args.learning_rate
                group["lr"] = base_lr * lr_scale
            optimizer.step()

            if step % args.log_steps == 0 or step == start_step + 1:
                reduced = {name: reduce_mean(value, world_size) for name, value in accum.items()}
                grad_mean = reduce_mean(torch.as_tensor(float(grad_norm), device=device), world_size)
                seconds = time.time() - step_started
                elapsed = time.time() - start_time
                toks_per_sec = tokens_per_optim_step / max(seconds, 1e-6)
                total_loss = (
                    reduced["ce_loss"]
                    + args.budget_loss_coef * reduced["budget_loss"]
                    + args.load_loss_coef * reduced["load_loss"]
                    + args.z_loss_coef * reduced["z_loss"]
                )
                if is_rank0(rank):
                    print(
                        f"step={step} loss={float(total_loss):.4f} ce={float(reduced['ce_loss']):.4f} "
                        f"budget={float(reduced['budget_loss']):.4f} load={float(reduced['load_loss']):.4f} "
                        f"z={float(reduced['z_loss']):.4f} prob_keep={float(reduced['prob_keep_ratio']):.4f} "
                        f"hard_keep={float(reduced['hard_keep_ratio']):.4f} heads_tok={float(reduced['hard_heads_per_token']):.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g} grad={float(grad_mean):.3f} "
                        f"tok/s={toks_per_sec:.1f} elapsed_h={elapsed/3600:.2f}",
                        flush=True,
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", float(total_loss), step)
                        writer.add_scalar("train/ce_loss", float(reduced["ce_loss"]), step)
                        writer.add_scalar("gate/budget_loss", float(reduced["budget_loss"]), step)
                        writer.add_scalar("gate/load_loss", float(reduced["load_loss"]), step)
                        writer.add_scalar("gate/z_loss", float(reduced["z_loss"]), step)
                        writer.add_scalar("gate/prob_keep_ratio", float(reduced["prob_keep_ratio"]), step)
                        writer.add_scalar("gate/hard_keep_ratio", float(reduced["hard_keep_ratio"]), step)
                        writer.add_scalar("gate/hard_heads_per_token", float(reduced["hard_heads_per_token"]), step)
                        writer.add_scalar("train/tokens_per_second", toks_per_sec, step)
                        writer.add_scalar("train/grad_norm", float(grad_mean), step)
                        if last_head_load is not None:
                            layer_mean = last_head_load.float().mean(dim=0).cpu()
                            writer.add_histogram("gate/head_load_mean_over_layers", layer_mean, step)
                            writer.add_scalar("gate/head_load_min", float(layer_mean.min()), step)
                            writer.add_scalar("gate/head_load_max", float(layer_mean.max()), step)
                            writer.add_scalar("gate/head_load_mean", float(layer_mean.mean()), step)
                        writer.flush()

            if step % args.save_steps == 0:
                save_checkpoint(output_dir, model, optimizer, step, args, rank)
                barrier()
            if args.max_train_seconds > 0 and (time.time() - start_time) >= args.max_train_seconds:
                save_checkpoint(output_dir, model, optimizer, step, args, rank)
                barrier()
                if is_rank0(rank):
                    print(f"reached max_train_seconds={args.max_train_seconds}; stopping at step {step}", flush=True)
                break
        if writer is not None:
            writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
