from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 6
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 3072
    rope_theta: float = 1_000_000.0
    rms_eps: float = 1e-6


SPECIAL = {
    "BOS": 1,
    "DOC": 2,
    "SENT": 3,
    "SEP": 4,
    "FACT": 5,
    "KEY": 6,
    "VALUE": 7,
    "QUERY": 8,
    "ANSWER": 9,
}


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.weight.float()).to(source_dtype)


def rope_pair_scales(
    config: ModelConfig,
    variant: str,
    layer_index: int,
    device: torch.device,
    head_count: int | None = None,
) -> torch.Tensor:
    pairs = config.head_dim // 2
    pair_index = torch.arange(pairs, device=device, dtype=torch.float32)
    if variant == "native":
        return torch.ones(pairs, device=device)
    if variant == "deep_highfreq_drop":
        scales = torch.ones(pairs, device=device)
        if layer_index >= config.num_layers // 2:
            scales[:8] = 0.0
        return scales
    if variant == "slow_rope":
        return torch.full((pairs,), 0.5, device=device)
    if variant == "smooth_layer_frequency":
        alpha_min = 0.25
        layer_center = 7.5
        layer_temperature = 1.5
        frequency_center = 7.5
        frequency_temperature = 1.5
        layer_gate = torch.sigmoid(
            torch.tensor((layer_index - layer_center) / layer_temperature, device=device)
        )
        frequency_gate = torch.sigmoid((frequency_center - pair_index) / frequency_temperature)
        return 1.0 - (1.0 - alpha_min) * layer_gate * frequency_gate
    if variant == "deep_highfreq_taper":
        # Optimized replacement for hard deletion: preserve shallow layers and
        # taper fast frequency pairs continuously in deeper layers.
        alpha_min = 0.25
        layer_gate = torch.sigmoid(
            torch.tensor((layer_index - 5.5) / 1.25, device=device)
        )
        frequency_gate = torch.sigmoid((7.5 - pair_index) / 1.5)
        return 1.0 - (1.0 - alpha_min) * layer_gate * frequency_gate
    if variant == "layerwise_slow_rope":
        # Optimized replacement for globally halving every layer: early layers
        # remain nearly native, while deeper layers smoothly approach 0.5x.
        layer_gate = torch.sigmoid(
            torch.tensor((layer_index - 5.5) / 1.5, device=device)
        )
        return torch.ones(pairs, device=device) * (1.0 - 0.5 * layer_gate)
    if variant == "complementary_smooth":
        if head_count is None:
            raise ValueError("complementary_smooth requires head_count")
        layer_gate = torch.sigmoid(
            torch.tensor((layer_index - 5.5) / 1.5, device=device)
        )
        frequency_gate = torch.sigmoid((11.5 - pair_index) / 3.0)
        remote_scale = 1.0 - 0.75 * layer_gate * frequency_gate
        scales = torch.ones((head_count, pairs), device=device)
        # Qwen-style GQA maps the first half of Q heads to KV group 0 and
        # the second half to KV group 1. Group 0 remains native; group 1 is
        # the smoothly slowed long-range branch.
        branch = torch.arange(head_count, device=device) * config.num_kv_heads // head_count
        scales[branch == 1] = remote_scale
        return scales
    raise ValueError(f"Unknown RoPE variant: {variant}")


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    base_angles = position_ids.float()[:, None] * inv_freq[None, :]
    if scales.ndim == 1:
        angles = base_angles * scales[None, :]
        cos = angles.cos()[None, None, :, :].to(x.dtype)
        sin = angles.sin()[None, None, :, :].to(x.dtype)
    elif scales.ndim == 2:
        angles = base_angles[None, :, :] * scales[:, None, :]
        cos = angles.cos()[None, :, :, :].to(x.dtype)
        sin = angles.sin()[None, :, :, :].to(x.dtype)
    else:
        raise ValueError(f"Unexpected RoPE scale rank: {scales.ndim}")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int, rope_variant: str) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.rope_variant = rope_variant
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_eps)
        inv_freq = config.rope_theta ** (
            -torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if rope_variant == "fade_rope_band8":
            # tau_{l,h} = 2*pi*exp(phase_log_scale_{l,h}); initialized to one turn.
            self.phase_log_scale = nn.Parameter(torch.zeros(config.num_heads))
        else:
            self.register_parameter("phase_log_scale", None)

    def _band8_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        fade: bool,
    ) -> torch.Tensor:
        batch, heads, length, _ = q.shape
        scores = torch.zeros((batch, heads, length, length), device=q.device, dtype=torch.float32)
        distance = (
            torch.arange(length, device=q.device)[:, None]
            - torch.arange(length, device=q.device)[None, :]
        ).abs().float()
        tau = None
        if fade:
            tau = (2.0 * math.pi * self.phase_log_scale.float().exp())[:, None, None]
        for pair_start in range(0, self.config.head_dim // 2, 8):
            pair_end = min(pair_start + 8, self.config.head_dim // 2)
            dim_start = pair_start * 2
            dim_end = pair_end * 2
            post_score = torch.matmul(
                q_rope[..., dim_start:dim_end],
                k_rope[..., dim_start:dim_end].transpose(-1, -2),
            ).float()
            if not fade:
                scores = scores + post_score
                continue
            pre_score = torch.matmul(
                q[..., dim_start:dim_end],
                k[..., dim_start:dim_end].transpose(-1, -2),
            ).float()
            omega = torch.sqrt(self.inv_freq[pair_start] * self.inv_freq[pair_end - 1]).float()
            phase_ratio = distance[None, :, :] * omega / tau
            confidence = 1.0 / (1.0 + phase_ratio.square().square())
            scores = scores + confidence[None, :, :, :] * post_score + (
                1.0 - confidence[None, :, :, :]
            ) * pre_score
        scores = scores / math.sqrt(self.config.head_dim)
        causal_mask = torch.ones((length, length), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.matmul(probabilities, v)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, length, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, length, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q_pre = q
        k_pre = k
        position_ids = torch.arange(length, device=hidden.device)
        manual_band8 = self.rope_variant in {"native_band8_reference", "fade_rope_band8"}
        scale_variant = "native" if manual_band8 else self.rope_variant
        q_scales = rope_pair_scales(
            self.config, scale_variant, self.layer_index, hidden.device, self.config.num_heads
        )
        k_scales = rope_pair_scales(
            self.config, scale_variant, self.layer_index, hidden.device, self.config.num_kv_heads
        )
        q = apply_rope(q, position_ids, self.inv_freq, q_scales)
        k = apply_rope(k, position_ids, self.inv_freq, k_scales)
        repeat = self.config.num_heads // self.config.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        if manual_band8:
            k_pre = k_pre.repeat_interleave(repeat, dim=1)
            output = self._band8_attention(
                q_pre, k_pre, v, q, k, fade=self.rope_variant == "fade_rope_band8"
            )
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int, rope_variant: str) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.attention = Attention(config, layer_index, rope_variant)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.mlp = MLP(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.input_norm(hidden))
        hidden = hidden + self.mlp(self.post_attention_norm(hidden))
        return hidden


class QwenStyleLM(nn.Module):
    def __init__(self, config: ModelConfig, rope_variant: str) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, index, rope_variant) for index in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    def forward(self, tokens: torch.Tensor, loss_weights: torch.Tensor) -> tuple[torch.Tensor, int]:
        hidden = self.forward_hidden(tokens)
        shifted_hidden = hidden[:, :-1, :]
        labels = tokens[:, 1:]
        shifted_weights = loss_weights[:, 1:]
        selected = shifted_weights > 0
        selected_hidden = shifted_hidden[selected]
        selected_labels = labels[selected]
        logits = self.lm_head(selected_hidden)
        token_losses = F.cross_entropy(logits.float(), selected_labels, reduction="none")
        selected_weights = shifted_weights[selected].float()
        loss = (token_losses * selected_weights).sum() / selected_weights.sum()
        phase_parameters = [
            layer.attention.phase_log_scale
            for layer in self.layers
            if layer.attention.phase_log_scale is not None
        ]
        if phase_parameters:
            phase_regularization = torch.stack(
                [parameter.square().mean() for parameter in phase_parameters]
            ).mean()
            loss = loss + 1e-4 * phase_regularization
        return loss, int(selected_labels.numel())

    def answer_logits(self, tokens: torch.Tensor, answer_positions: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(tokens)
        batch_index = torch.arange(tokens.shape[0], device=tokens.device)
        query_hidden = hidden[batch_index, answer_positions - 1]
        return self.lm_head(query_hidden)


def make_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
    target_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if sequence_length < 128:
        raise ValueError("sequence_length must be at least 128")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens = torch.empty((batch_size, sequence_length), dtype=torch.long)
    loss_weights = torch.zeros_like(tokens, dtype=torch.float32)
    answer_positions = torch.empty(batch_size, dtype=torch.long)

    key_base = 100
    key_space = 1024
    value_base = 1400
    value_space = 1024
    topic_base = 3000
    topic_space = 512
    word_base = 5000
    number_of_facts = 16

    for batch_index in range(batch_size):
        row = tokens[batch_index]
        row[0] = SPECIAL["BOS"]
        row[1] = SPECIAL["DOC"]
        cursor = 2
        while cursor < sequence_length:
            topic = int(torch.randint(topic_space, (1,), generator=generator).item())
            block = [
                SPECIAL["SENT"],
                topic_base + topic,
                word_base + (topic * 3) % (vocab_size - word_base),
                word_base + (topic * 3 + 1) % (vocab_size - word_base),
                word_base + (topic * 3 + 2) % (vocab_size - word_base),
                SPECIAL["SEP"],
            ]
            take = min(len(block), sequence_length - cursor)
            row[cursor : cursor + take] = torch.tensor(block[:take])
            cursor += take

        keys = torch.randperm(key_space, generator=generator)[:number_of_facts] + key_base
        values = torch.randperm(value_space, generator=generator)[:number_of_facts] + value_base
        max_fact_start = sequence_length - 48
        fact_starts = torch.linspace(16, max_fact_start, number_of_facts).long()
        for fact_index, start in enumerate(fact_starts.tolist()):
            row[start : start + 5] = torch.tensor(
                [SPECIAL["FACT"], int(keys[fact_index]), SPECIAL["VALUE"], int(values[fact_index]), SPECIAL["SEP"]]
            )

        query_targets = torch.randperm(number_of_facts, generator=generator)[:4].tolist()
        if target_fraction is not None:
            query_targets[-1] = int(
                round(max(0.0, min(1.0, target_fraction)) * (number_of_facts - 1))
            )
        query_region_start = sequence_length - 20
        query_answer_positions: list[int] = []
        for query_offset, target_index in enumerate(query_targets):
            query_start = query_region_start + query_offset * 5
            answer_position = query_start + 3
            row[query_start : query_start + 5] = torch.tensor(
                [
                    SPECIAL["QUERY"],
                    int(keys[target_index]),
                    SPECIAL["ANSWER"],
                    int(values[target_index]),
                    SPECIAL["SEP"],
                ]
            )
            query_answer_positions.append(answer_position)
        answer_positions[batch_index] = query_answer_positions[-1]

        loss_weights[batch_index, row < 10] = 1.0
        loss_weights[batch_index, torch.arange(31, sequence_length, 32)] = 0.25
        loss_weights[batch_index, torch.tensor(query_answer_positions)] = 64.0

    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()[:16]
    return tokens, loss_weights, answer_positions, digest


@torch.no_grad()
def evaluate(
    model: QwenStyleLM,
    config: ModelConfig,
    lengths: Iterable[int],
    samples: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int]]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    fractions = (0.05, 0.25, 0.50, 0.75)
    for length in lengths:
        correct = 0
        total_nll = 0.0
        total_margin = 0.0
        started = time.time()
        for sample_index in range(samples):
            fraction = fractions[sample_index % len(fractions)]
            tokens, _, answer_positions, _ = make_batch(
                1,
                int(length),
                config.vocab_size,
                seed + int(length) * 1000 + sample_index,
                target_fraction=fraction,
            )
            tokens = tokens.to(device)
            answer_positions = answer_positions.to(device)
            logits = model.answer_logits(tokens, answer_positions).float()
            gold = tokens[torch.arange(tokens.shape[0], device=device), answer_positions]
            log_probs = logits.log_softmax(dim=-1)
            nll = -log_probs.gather(1, gold[:, None]).squeeze(1)
            top2 = torch.topk(logits, k=2, dim=-1).values
            gold_logit = logits.gather(1, gold[:, None]).squeeze(1)
            strongest_other = torch.where(
                logits.argmax(dim=-1) == gold,
                top2[:, 1],
                top2[:, 0],
            )
            correct += int((logits.argmax(dim=-1) == gold).sum().item())
            total_nll += float(nll.sum().item())
            total_margin += float((gold_logit - strongest_other).sum().item())
        rows.append(
            {
                "length": int(length),
                "samples": samples,
                "accuracy": correct / samples,
                "gold_nll": total_nll / samples,
                "gold_ppl": math.exp(min(20.0, total_nll / samples)),
                "gold_margin": total_margin / samples,
                "elapsed_seconds": time.time() - started,
            }
        )
    model.train()
    return rows


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_lengths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "native",
            "deep_highfreq_drop",
            "slow_rope",
            "smooth_layer_frequency",
            "deep_highfreq_taper",
            "layerwise_slow_rope",
            "complementary_smooth",
            "native_band8_reference",
            "fade_rope_band8",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-lengths", default="512,1024,2048")
    parser.add_argument("--final-eval-lengths", default="512,1024,2048,4096,8192")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = ModelConfig()
    model = QwenStyleLM(config, args.variant).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 115_000_000 <= parameter_count <= 140_000_000:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    wrapped: nn.Module = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if distributed else model
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    global_tokens_per_step = args.micro_batch * args.sequence_length * world_size * args.grad_accum
    total_steps = math.ceil(args.tokens / global_tokens_per_step)
    output_dir = args.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        args_payload = vars(args).copy()
        args_payload["output_dir"] = str(args.output_dir)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "args": args_payload,
                    "model": asdict(config),
                    "parameter_count": parameter_count,
                    "world_size": world_size,
                    "global_tokens_per_step": global_tokens_per_step,
                    "total_steps": total_steps,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    start_step = 0
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    if checkpoint_path.exists() and not args.skip_checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
    if distributed:
        dist.barrier()

    training_started = time.time()
    micro_step = start_step * args.grad_accum
    for step in range(start_step, total_steps):
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        selected_sum = 0
        data_hash = ""
        step_started = time.time()
        for accumulation_index in range(args.grad_accum):
            batch_seed = args.data_seed + micro_step * world_size + rank
            tokens, loss_weights, _, data_hash = make_batch(
                args.micro_batch, args.sequence_length, config.vocab_size, batch_seed
            )
            tokens = tokens.to(device, non_blocking=True)
            loss_weights = loss_weights.to(device, non_blocking=True)
            sync_context = (
                wrapped.no_sync()
                if distributed and accumulation_index < args.grad_accum - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, selected = wrapped(tokens, loss_weights)
                    scaled_loss = loss / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
                scaled_loss.backward()
            loss_sum += float(loss.detach().item())
            selected_sum += selected
            micro_step += 1

        gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0).item())
        if step < args.warmup_steps:
            learning_rate = args.learning_rate * (step + 1) / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()

        elapsed = time.time() - step_started
        if rank == 0 and (step % args.log_every == 0 or step == total_steps - 1):
            append_jsonl(
                output_dir / "train.jsonl",
                {
                    "step": step,
                    "tokens_seen": min(args.tokens, (step + 1) * global_tokens_per_step),
                    "loss": loss_sum / args.grad_accum,
                    "selected_labels": selected_sum,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "tokens_per_second": global_tokens_per_step / elapsed,
                    "step_seconds": elapsed,
                    "data_hash_rank0_last_microbatch": data_hash,
                    "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                    "wall_seconds": time.time() - training_started,
                },
            )

        should_evaluate = (step + 1) % args.eval_every == 0 or step == total_steps - 1
        if should_evaluate:
            if distributed:
                dist.barrier()
            if rank == 0:
                rows = evaluate(model, config, parse_lengths(args.eval_lengths), args.eval_samples, device, args.seed + step)
                append_jsonl(output_dir / "eval.jsonl", {"step": step, "final": False, "rows": rows})
            if distributed:
                dist.barrier()

        should_save = not args.skip_checkpoints and (
            (step + 1) % args.save_every == 0 or step == total_steps - 1
        )
        if should_save and rank == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                temporary,
            )
            temporary.replace(checkpoint_path)
        if distributed and should_save:
            dist.barrier()

    if distributed:
        dist.barrier()
    if rank == 0:
        final_rows = evaluate(
            model,
            config,
            parse_lengths(args.final_eval_lengths),
            args.eval_samples,
            device,
            args.seed + 999_999,
        )
        append_jsonl(output_dir / "eval.jsonl", {"step": total_steps - 1, "final": True, "rows": final_rows})
        (output_dir / "DONE").write_text("complete\n", encoding="utf-8")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
