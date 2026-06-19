from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise RuntimeError("transformers is required for tokenizer loading.") from exc


@dataclass
class RoutedQwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    initializer_range: float
    tie_word_embeddings: bool = True
    router_top_k: int = 4
    router_temperature: float = 1.0
    router_noise_std: float = 0.1
    attention_bias: bool = False

    @classmethod
    def from_qwen_config(
        cls,
        path: Path,
        router_top_k: int,
        router_temperature: float,
        router_noise_std: float,
    ) -> "RoutedQwenConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        num_heads = int(data["num_attention_heads"])
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=int(data.get("head_dim", int(data["hidden_size"]) // num_heads)),
            max_position_embeddings=int(data.get("max_position_embeddings", 40960)),
            rope_theta=float(data.get("rope_theta", 1000000.0)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            initializer_range=float(data.get("initializer_range", 0.02)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            router_noise_std=router_noise_std,
            attention_bias=bool(data.get("attention_bias", False)),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.outer(position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoutedMHA(nn.Module):
    def __init__(self, config: RoutedQwenConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attn_dim = self.num_heads * self.head_dim
        self.top_k = config.router_top_k
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.attn_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.attn_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.attn_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.attn_dim, config.hidden_size, bias=config.attention_bias)
        self.gate = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        route_logits = logits
        if self.training and self.config.router_noise_std > 0:
            route_logits = route_logits + torch.randn_like(route_logits) * self.config.router_noise_std
        probs = F.softmax(logits / max(self.config.router_temperature, 1e-6), dim=-1)
        top_indices = torch.topk(route_logits, k=self.top_k, dim=-1).indices
        hard = torch.zeros_like(logits)
        hard.scatter_(-1, top_indices, 1.0)
        route = hard + probs - probs.detach()
        prob_mean = probs.float().mean(dim=(0, 1))
        hard_load = hard.float().mean(dim=(0, 1))
        target_prob = 1.0 / self.num_heads
        load_balance_loss = ((prob_mean - target_prob) ** 2).mean() / (target_prob**2)
        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        entropy = -(probs.float() * probs.float().clamp_min(1e-9).log()).sum(dim=-1).mean()
        return route, hard.bool(), load_balance_loss, z_loss, entropy, hard_load

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        route, hard_mask, load_loss, z_loss, entropy, hard_load = self._route(hidden_states)
        route = route.view(batch_size, seq_len, self.num_heads, 1)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        route_heads = route.transpose(1, 2)
        q = q * route_heads
        k = k * route_heads
        v = v * route_heads
        cos, sin = self.rotary(position_ids, q.dtype, q.device)
        q, k = apply_rotary(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_mask = hard_mask.permute(0, 2, 1)[:, :, None, :]
        query_mask = hard_mask.permute(0, 2, 1)[:, :, :, None]
        causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device).tril()
        valid = causal[None, None, :, :] & key_mask
        eye = torch.eye(seq_len, dtype=torch.bool, device=hidden_states.device)
        valid = valid | ((~query_mask) & eye[None, None, :, :])
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attn_probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output * route_heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attn_dim)
        output = self.o_proj(attn_output)
        return output, load_loss, z_loss, entropy, hard_load


class QwenMLP(nn.Module):
    def __init__(self, config: RoutedQwenConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RoutedQwenBlock(nn.Module):
    def __init__(self, config: RoutedQwenConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = RoutedMHA(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = QwenMLP(config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        attn_output, load_loss, z_loss, entropy, hard_load = self.self_attn(attn_input, position_ids)
        hidden_states = residual + attn_output
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(mlp_input)
        return hidden_states, load_loss, z_loss, entropy, hard_load


class RoutedQwenForCausalLM(nn.Module):
    def __init__(self, config: RoutedQwenConfig, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([RoutedQwenBlock(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids)
        load_losses = []
        z_losses = []
        entropies = []
        hard_loads = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, load_loss, z_loss, entropy, hard_load = checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, load_loss, z_loss, entropy, hard_load = layer(hidden_states, position_ids)
            load_losses.append(load_loss)
            z_losses.append(z_loss)
            entropies.append(entropy)
            hard_loads.append(hard_load)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "router_load_loss": torch.stack(load_losses).mean(),
            "router_z_loss": torch.stack(z_losses).mean(),
            "router_entropy": torch.stack(entropies).mean(),
            "router_hard_load": torch.stack(hard_loads),
        }
        if labels is not None:
            output["ce_loss"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
        return output


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
        raise FileNotFoundError(f"No training text files found under {data_root} with globs={globs!r}")
    return unique


def select_training_files(files: list[Path], sample_files: int, seed: int) -> list[Path]:
    if sample_files <= 0 or sample_files >= len(files):
        return files
    rng = random.Random(seed)
    selected = list(files)
    rng.shuffle(selected)
    return sorted(selected[:sample_files], key=lambda path: path.as_posix())


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
            yield text, total_chars


def build_token_cache(
    tokenizer_path: Path,
    train_data_root: Path | None,
    train_text_path: Path | None,
    train_text_glob: str,
    dataset_sample_files: int,
    dataset_sample_seed: int,
    cache_dir: Path,
    max_chars: int,
    max_chars_per_file: int,
    chunk_chars: int,
    rebuild: bool,
    cache_wait_timeout_seconds: int,
    cache_poll_seconds: int,
    rank: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_bin = cache_dir / "train_tokens.uint32.bin"
    meta_path = cache_dir / "train_tokens_meta.json"
    if token_bin.exists() and meta_path.exists() and not rebuild:
        return token_bin
    if not is_rank0(rank):
        started_wait = time.time()
        while True:
            cache_ready = token_bin.exists() and meta_path.exists()
            if cache_ready and (not rebuild or meta_path.stat().st_mtime >= started_wait - 1):
                return token_bin
            if cache_wait_timeout_seconds > 0 and (time.time() - started_wait) > cache_wait_timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for rank0 to build token cache at {token_bin}. "
                    f"Waited {cache_wait_timeout_seconds} seconds."
                )
            time.sleep(max(1, cache_poll_seconds))
    if token_bin.exists():
        token_bin.unlink()
    tmp_bin = cache_dir / "train_tokens.uint32.tmp"
    if tmp_bin.exists():
        tmp_bin.unlink()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if train_data_root is not None:
        all_files = discover_text_files(train_data_root, train_text_glob)
        selected_files = select_training_files(all_files, dataset_sample_files, dataset_sample_seed)
    elif train_text_path is not None:
        all_files = [train_text_path]
        selected_files = [train_text_path]
    else:
        raise ValueError("Either --train_data_root or --train_text_path must be set.")
    total_chars = 0
    total_tokens = 0
    started = time.time()
    with tmp_bin.open("ab") as out:
        for file_idx, text_path in enumerate(selected_files, start=1):
            if max_chars > 0 and total_chars >= max_chars:
                break
            file_budget = max_chars_per_file
            if max_chars > 0:
                remaining_global = max_chars - total_chars
                file_budget = remaining_global if file_budget <= 0 else min(file_budget, remaining_global)
            file_chars = 0
            for text, file_chars in iter_text_chunks(text_path, chunk_chars, file_budget):
                total_chars += len(text)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if ids:
                    arr = np.asarray(ids, dtype=np.uint32)
                    arr.tofile(out)
                    total_tokens += int(arr.size)
                if total_tokens and total_tokens % 5_000_000 < len(ids):
                    print(f"tokenized {total_tokens} tokens from {total_chars} chars", flush=True)
            if file_idx % 100 == 0:
                print(
                    f"processed {file_idx}/{len(selected_files)} sampled files; "
                    f"chars={total_chars} tokens={total_tokens}",
                    flush=True,
                )
            if file_chars == 0:
                print(f"warning: sampled file had no readable text: {text_path}", flush=True)
    tmp_bin.replace(token_bin)
    meta = {
        "tokenizer_path": str(tokenizer_path),
        "train_data_root": str(train_data_root) if train_data_root is not None else "",
        "train_text_path": str(train_text_path) if train_text_path is not None else "",
        "train_text_glob": train_text_glob,
        "all_file_count": len(all_files),
        "sampled_file_count": len(selected_files),
        "dataset_sample_files": dataset_sample_files,
        "dataset_sample_seed": dataset_sample_seed,
        "max_chars": max_chars,
        "max_chars_per_file": max_chars_per_file,
        "chunk_chars": chunk_chars,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "seconds": time.time() - started,
        "sampled_files": [str(path) for path in selected_files],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote token cache: {token_bin} ({total_tokens} tokens)", flush=True)
    return token_bin


class RandomTokenBatcher:
    def __init__(self, token_bin: Path, seq_len: int, batch_size: int, rank: int, seed: int) -> None:
        self.tokens = np.memmap(token_bin, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed + rank * 100003)
        if len(self.tokens) < seq_len + 2:
            raise ValueError(f"Token cache has {len(self.tokens)} tokens, fewer than seq_len + 2.")

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, size=self.batch_size)
        batch = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int64)
        for row, start in enumerate(starts):
            batch[row] = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        tensor = torch.from_numpy(batch).to(device=device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:]


def cosine_lr(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: RoutedQwenConfig,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_rank0(rank):
        return
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(raw_model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "routed_qwen_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"step": step, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    latest = output_dir / "latest_checkpoint"
    tmp_latest = output_dir / "latest_checkpoint.tmp"
    tmp_latest.write_text(str(ckpt_dir), encoding="utf-8")
    tmp_latest.replace(latest)
    print(f"saved checkpoint: {ckpt_dir}", flush=True)


def load_checkpoint_if_needed(
    resume_from: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not resume_from:
        return 0
    ckpt_dir = Path(resume_from)
    if ckpt_dir.is_file():
        ckpt_dir = Path(ckpt_dir.read_text(encoding="utf-8").strip())
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
    state = json.loads((ckpt_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", default="/mnt/workspace/Qwen3-0.6B/config.json")
    parser.add_argument("--tokenizer_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--train_data_root", default="/mnt/workspace/dclm")
    parser.add_argument("--train_text_path", default="")
    parser.add_argument("--train_text_glob", default="*.txt")
    parser.add_argument("--dataset_sample_files", type=int, default=1024)
    parser.add_argument("--dataset_sample_seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="/mnt/workspace/routed_top4_qwen3_0p6b_runs/run")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--max_train_seconds", type=int, default=72_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--router_top_k", type=int, default=4)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--router_z_loss_coef", type=float, default=0.001)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--router_noise_std", type=float, default=0.1)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--token_cache_dir", default="")
    parser.add_argument("--tokenize_max_chars", type=int, default=200_000_000)
    parser.add_argument("--tokenize_max_chars_per_file", type=int, default=250_000)
    parser.add_argument("--tokenize_chunk_chars", type=int, default=2_000_000)
    parser.add_argument("--cache_wait_timeout_seconds", type=int, default=86_400)
    parser.add_argument("--cache_poll_seconds", type=int, default=5)
    parser.add_argument("--rebuild_token_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def tiny_smoke_test() -> None:
    config = RoutedQwenConfig(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        router_top_k=2,
    )
    model = RoutedQwenForCausalLM(config, gradient_checkpointing=True)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = torch.randint(0, config.vocab_size, (2, 32))
    out = model(input_ids, labels)
    loss = out["ce_loss"] + 0.01 * out["router_load_loss"] + 0.001 * out["router_z_loss"]
    loss.backward()
    print(
        json.dumps(
            {
                "loss": float(loss.detach()),
                "ce_loss": float(out["ce_loss"].detach()),
                "router_load_loss": float(out["router_load_loss"].detach()),
                "router_entropy": float(out["router_entropy"].detach()),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        tiny_smoke_test()
        return
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

        token_cache_dir = Path(args.token_cache_dir) if args.token_cache_dir else output_dir / "token_cache"
        train_data_root = Path(args.train_data_root) if args.train_data_root else None
        train_text_path = Path(args.train_text_path) if args.train_text_path else None
        token_bin = build_token_cache(
            Path(args.tokenizer_path),
            train_data_root,
            train_text_path,
            args.train_text_glob,
            args.dataset_sample_files,
            args.dataset_sample_seed,
            token_cache_dir,
            args.tokenize_max_chars,
            args.tokenize_max_chars_per_file,
            args.tokenize_chunk_chars,
            args.rebuild_token_cache,
            args.cache_wait_timeout_seconds,
            args.cache_poll_seconds,
            rank,
        )
        barrier()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        if is_rank0(rank):
            tokenizer.save_pretrained(output_dir / "tokenizer")

        config = RoutedQwenConfig.from_qwen_config(
            Path(args.model_config_path),
            router_top_k=args.router_top_k,
            router_temperature=args.router_temperature,
            router_noise_std=args.router_noise_std,
        )
        if config.router_top_k <= 0 or config.router_top_k > config.num_attention_heads:
            raise ValueError("--router_top_k must be between 1 and num_attention_heads.")
        if is_rank0(rank):
            (output_dir / "routed_qwen_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

        model = RoutedQwenForCausalLM(config, gradient_checkpointing=not args.no_gradient_checkpointing).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        start_step = load_checkpoint_if_needed(args.resume_from, model, optimizer, device)
        batcher = RandomTokenBatcher(token_bin, args.seq_len, args.per_device_batch_size, rank, args.seed)
        writer = SummaryWriter(output_dir / "tensorboard") if (is_rank0(rank) and SummaryWriter is not None) else None

        start_time = time.time()
        tokens_per_optim_step = args.seq_len * args.per_device_batch_size * args.gradient_accumulation_steps * world_size
        if is_rank0(rank):
            print(f"world_size={world_size} device={device} tokens_per_optim_step={tokens_per_optim_step}", flush=True)
            print(f"output_dir={output_dir}", flush=True)

        for step in range(start_step + 1, args.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_ce = torch.zeros((), device=device)
            accum_load = torch.zeros((), device=device)
            accum_z = torch.zeros((), device=device)
            accum_entropy = torch.zeros((), device=device)
            last_hard_load = None
            step_started = time.time()
            for micro_step in range(args.gradient_accumulation_steps):
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DDP) and micro_step < args.gradient_accumulation_steps - 1
                    else nullcontext()
                )
                input_ids, labels = batcher.next_batch(device)
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        out = model(input_ids, labels)
                        ce_loss = out["ce_loss"]
                        load_loss = out["router_load_loss"]
                        z_loss = out["router_z_loss"]
                        loss = ce_loss + args.router_aux_loss_coef * load_loss + args.router_z_loss_coef * z_loss
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                accum_ce += ce_loss.detach() / args.gradient_accumulation_steps
                accum_load += load_loss.detach() / args.gradient_accumulation_steps
                accum_z += z_loss.detach() / args.gradient_accumulation_steps
                accum_entropy += out["router_entropy"].detach() / args.gradient_accumulation_steps
                last_hard_load = out["router_hard_load"].detach()

            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                grad_norm = torch.zeros((), device=device)
            lr_scale = cosine_lr(step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_scale
            optimizer.step()

            if step % args.log_steps == 0 or step == start_step + 1:
                ce_mean = reduce_mean(accum_ce, world_size)
                load_mean = reduce_mean(accum_load, world_size)
                z_mean = reduce_mean(accum_z, world_size)
                entropy_mean = reduce_mean(accum_entropy, world_size)
                grad_mean = reduce_mean(torch.as_tensor(float(grad_norm), device=device), world_size)
                seconds = time.time() - step_started
                elapsed = time.time() - start_time
                toks_per_sec = tokens_per_optim_step / max(seconds, 1e-6)
                if is_rank0(rank):
                    total_loss_value = ce_mean + args.router_aux_loss_coef * load_mean + args.router_z_loss_coef * z_mean
                    print(
                        f"step={step} loss={float(total_loss_value):.4f} ce={float(ce_mean):.4f} "
                        f"load={float(load_mean):.4f} z={float(z_mean):.4f} entropy={float(entropy_mean):.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g} grad={float(grad_mean):.3f} "
                        f"tok/s={toks_per_sec:.1f} elapsed_h={elapsed/3600:.2f}",
                        flush=True,
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", float(total_loss_value), step)
                        writer.add_scalar("train/ce_loss", float(ce_mean), step)
                        writer.add_scalar("router/load_loss", float(load_mean), step)
                        writer.add_scalar("router/z_loss", float(z_mean), step)
                        writer.add_scalar("router/entropy", float(entropy_mean), step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                        writer.add_scalar("train/grad_norm", float(grad_mean), step)
                        writer.add_scalar("train/tokens_per_second", toks_per_sec, step)
                        if last_hard_load is not None:
                            layer_mean = last_hard_load.float().mean(dim=0).cpu()
                            writer.add_histogram("router/head_load_mean_over_layers", layer_mean, step)
                            writer.add_scalar("router/hard_load_min", float(layer_mean.min()), step)
                            writer.add_scalar("router/hard_load_max", float(layer_mean.max()), step)
                            writer.add_scalar("router/hard_load_mean", float(layer_mean.mean()), step)
                        writer.flush()

            if step % args.save_steps == 0:
                save_checkpoint(output_dir, model, optimizer, step, config, args, rank)
                barrier()
            if args.max_train_seconds > 0 and (time.time() - start_time) >= args.max_train_seconds:
                save_checkpoint(output_dir, model, optimizer, step, config, args, rank)
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
