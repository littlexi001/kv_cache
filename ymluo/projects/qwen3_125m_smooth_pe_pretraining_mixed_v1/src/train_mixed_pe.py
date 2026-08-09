from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


from modeling_pe import (  # noqa: E402
    ModelConfig,
    QwenStyleLM,
    append_jsonl,
    parse_lengths,
)


VARIANTS = [
    "native",
    "deep_highfreq_taper",
    "layerwise_slow_rope",
    "complementary_smooth",
    "native_band8_reference",
    "fade_rope_band8",
]


def phase_tau_stats(model: QwenStyleLM) -> dict[str, float] | None:
    parameters = [
        layer.attention.phase_log_scale.detach().float()
        for layer in model.layers
        if layer.attention.phase_log_scale is not None
    ]
    if not parameters:
        return None
    tau = 2.0 * math.pi * torch.cat(parameters).exp()
    return {
        "phase_tau_min": float(tau.min().item()),
        "phase_tau_mean": float(tau.mean().item()),
        "phase_tau_max": float(tau.max().item()),
    }


class PackedTokenStream:
    def __init__(self, path: Path, sequence_length: int, seed: int) -> None:
        self.path = path
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.sequence_length = sequence_length
        self.chunk_count = len(self.tokens) // sequence_length
        if self.chunk_count < 2:
            raise ValueError(f"Token stream {path} is too small")
        multiplier = 104_729
        while math.gcd(multiplier, self.chunk_count) != 1:
            multiplier += 2
        self.multiplier = multiplier
        self.offset = seed % self.chunk_count

    def sample(self, global_sample_index: int) -> torch.Tensor:
        chunk_index = (global_sample_index * self.multiplier + self.offset) % self.chunk_count
        start = chunk_index * self.sequence_length
        values = np.asarray(self.tokens[start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(values.copy())


def artifact_hash(path: Path) -> str:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        return str(json.loads(meta_path.read_text(encoding="utf-8"))["sha256"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_special_ids(tokenizer_meta: Path) -> dict[str, int]:
    payload = json.loads(tokenizer_meta.read_text(encoding="utf-8"))
    mapping = payload["special_token_ids"]
    required = ["<bos>", "<fact>", "<key>", "<value>", "<query>", "<answer>", "<sep>"]
    required += ["<k0000>", "<k1023>", "<v0000>", "<v1023>"]
    missing = [token for token in required if token not in mapping]
    if missing:
        raise ValueError(f"Missing reserved tokens: {missing}")
    return {key: int(value) for key, value in mapping.items()}


def is_synthetic(global_sample_index: int, fraction: float, seed: int) -> bool:
    threshold = int(round(fraction * 1_000_000))
    value = (global_sample_index * 6364136223846793005 + seed) % 1_000_000
    return value < threshold


def overlay_retrieval(
    row: torch.Tensor,
    loss_weights: torch.Tensor,
    special: dict[str, int],
    seed: int,
    answer_weight: float,
    target_fraction: float | None = None,
    query_count: int = 4,
) -> list[int]:
    sequence_length = int(row.numel())
    if sequence_length < 128:
        raise ValueError("Synthetic retrieval requires at least 128 tokens")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    fact_count = 8
    key_indices = torch.randperm(1024, generator=generator)[:fact_count].tolist()
    value_indices = torch.randperm(1024, generator=generator)[:fact_count].tolist()
    if query_count * 5 + 80 >= sequence_length:
        raise ValueError(f"{query_count} queries do not fit sequence length {sequence_length}")
    repeated_targets = torch.arange(query_count) % fact_count
    query_targets = repeated_targets[torch.randperm(query_count, generator=generator)].tolist()
    if target_fraction is not None:
        query_targets[-1] = int(round(max(0.0, min(1.0, target_fraction)) * (fact_count - 1)))

    row[0] = special["<bos>"]
    query_region_start = sequence_length - query_count * 5
    fact_starts = torch.linspace(12, query_region_start - 16, fact_count).long().tolist()
    for fact_index, start in enumerate(fact_starts):
        key_id = special[f"<k{key_indices[fact_index]:04d}>"]
        value_id = special[f"<v{value_indices[fact_index]:04d}>"]
        row[start : start + 6] = torch.tensor(
            [special["<fact>"], special["<key>"], key_id, special["<value>"], value_id, special["<sep>"]],
            dtype=torch.long,
        )

    answer_positions: list[int] = []
    for query_offset, target_index in enumerate(query_targets):
        query_start = query_region_start + query_offset * 5
        answer_position = query_start + 3
        key_id = special[f"<k{key_indices[target_index]:04d}>"]
        value_id = special[f"<v{value_indices[target_index]:04d}>"]
        row[query_start : query_start + 5] = torch.tensor(
            [special["<query>"], key_id, special["<answer>"], value_id, special["<sep>"]],
            dtype=torch.long,
        )
        loss_weights[answer_position] = answer_weight
        answer_positions.append(answer_position)
    return answer_positions


def make_training_batch(
    stream: PackedTokenStream,
    batch_size: int,
    first_global_sample: int,
    special: dict[str, int],
    synthetic_fraction: float,
    answer_weight: float,
    training_queries: int,
    data_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    rows: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    synthetic_count = 0
    for row_index in range(batch_size):
        sample_index = first_global_sample + row_index
        row = stream.sample(sample_index)
        loss_weight = torch.ones(row.numel(), dtype=torch.float32)
        if is_synthetic(sample_index, synthetic_fraction, data_seed):
            overlay_retrieval(
                row,
                loss_weight,
                special,
                seed=data_seed + sample_index * 17,
                answer_weight=answer_weight,
                query_count=training_queries,
            )
            synthetic_count += 1
        rows.append(row)
        weights.append(loss_weight)
    tokens = torch.stack(rows)
    loss_weights = torch.stack(weights)
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()[:16]
    return tokens, loss_weights, synthetic_count, digest


@torch.no_grad()
def evaluate_natural(
    model: QwenStyleLM,
    stream: PackedTokenStream,
    samples: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_labels = 0
    started = time.time()
    for sample_index in range(samples):
        tokens = stream.sample(sample_index)[None, :].to(device)
        weights = torch.ones_like(tokens, dtype=torch.float32)
        loss, labels = model(tokens, weights)
        total_nll += float(loss.item()) * labels
        total_labels += labels
    mean_nll = total_nll / total_labels
    model.train()
    return {
        "samples": samples,
        "labels": total_labels,
        "mean_nll": mean_nll,
        "ppl": math.exp(min(20.0, mean_nll)),
        "elapsed_seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate_retrieval(
    model: QwenStyleLM,
    source: PackedTokenStream,
    special: dict[str, int],
    lengths: list[int],
    samples: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int]]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    fractions = (0.05, 0.25, 0.50, 0.75)
    for length in lengths:
        if length > source.sequence_length:
            eval_stream = PackedTokenStream(source.path, length, seed + length)
        else:
            eval_stream = source
        correct = 0
        total_nll = 0.0
        total_margin = 0.0
        started = time.time()
        for sample_index in range(samples):
            base = eval_stream.sample(sample_index + 10_000)
            if base.numel() != length:
                base = base[:length].clone()
            else:
                base = base.clone()
            weights = torch.ones(length, dtype=torch.float32)
            answer_positions = overlay_retrieval(
                base,
                weights,
                special,
                seed=seed + length * 1000 + sample_index,
                answer_weight=1.0,
                target_fraction=fractions[sample_index % len(fractions)],
            )
            tokens = base[None, :].to(device)
            position = torch.tensor([answer_positions[-1]], device=device)
            logits = model.answer_logits(tokens, position).float()
            gold = tokens[:, position.item()]
            log_probs = logits.log_softmax(dim=-1)
            nll = -log_probs.gather(1, gold[:, None]).squeeze(1)
            top2 = torch.topk(logits, k=2, dim=-1).values
            gold_logit = logits.gather(1, gold[:, None]).squeeze(1)
            strongest_other = torch.where(logits.argmax(dim=-1) == gold, top2[:, 1], top2[:, 0])
            correct += int((logits.argmax(dim=-1) == gold).sum().item())
            total_nll += float(nll.sum().item())
            total_margin += float((gold_logit - strongest_other).sum().item())
        rows.append(
            {
                "length": length,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--validation-bin", type=Path, required=True)
    parser.add_argument("--tokenizer-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--synthetic-fraction", type=float, default=0.05)
    parser.add_argument("--answer-weight", type=float, default=16.0)
    parser.add_argument("--training-queries", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-lengths", default="512,1024,2048")
    parser.add_argument("--final-eval-lengths", default="512,1024,2048,4096,8192")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--natural-eval-samples", type=int, default=16)
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.synthetic_fraction <= 1.0:
        raise ValueError("synthetic-fraction must be in [0, 1]")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    special = load_special_ids(args.tokenizer_meta)
    config = ModelConfig(vocab_size=32_000)
    model = QwenStyleLM(config, args.variant).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 115_000_000 <= parameter_count <= 140_000_000:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    wrapped: nn.Module = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if distributed else model
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    train_stream = PackedTokenStream(args.train_bin, args.sequence_length, args.data_seed)
    longest_eval = max(parse_lengths(args.final_eval_lengths))
    natural_validation_stream = PackedTokenStream(
        args.validation_bin, args.sequence_length, args.data_seed + 999
    )
    retrieval_validation_stream = PackedTokenStream(
        args.validation_bin, longest_eval, args.data_seed + 1999
    )
    global_tokens_per_step = args.micro_batch * args.sequence_length * world_size * args.grad_accum
    total_steps = math.ceil(args.tokens / global_tokens_per_step)
    output_dir = args.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = vars(args).copy()
        for key in ("output_dir", "train_bin", "validation_bin", "tokenizer_meta"):
            arguments[key] = str(arguments[key])
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "args": arguments,
                    "model": asdict(config),
                    "parameter_count": parameter_count,
                    "world_size": world_size,
                    "global_tokens_per_step": global_tokens_per_step,
                    "total_steps": total_steps,
                    "train_bin_sha256": artifact_hash(args.train_bin),
                    "validation_bin_sha256": artifact_hash(args.validation_bin),
                    "objective": "full-token causal LM; synthetic answer targets reweighted",
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
        synthetic_sum = 0
        data_hash = ""
        step_started = time.time()
        for accumulation_index in range(args.grad_accum):
            first_global_sample = (micro_step * world_size + rank) * args.micro_batch
            tokens, loss_weights, synthetic_count, data_hash = make_training_batch(
                train_stream,
                args.micro_batch,
                first_global_sample,
                special,
                args.synthetic_fraction,
                args.answer_weight,
                args.training_queries,
                args.data_seed,
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
            synthetic_sum += synthetic_count
            micro_step += 1

        gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0).item())
        if step < args.warmup_steps:
            learning_rate = args.learning_rate * (step + 1) / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()

        elapsed = time.time() - step_started
        if rank == 0 and (step % args.log_every == 0 or step == total_steps - 1):
            tau_stats = phase_tau_stats(model) or {}
            append_jsonl(
                output_dir / "train.jsonl",
                {
                    "step": step,
                    "tokens_seen": min(args.tokens, (step + 1) * global_tokens_per_step),
                    "loss": loss_sum / args.grad_accum,
                    "selected_labels": selected_sum,
                    "synthetic_rows_rank0": synthetic_sum,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "tokens_per_second": global_tokens_per_step / elapsed,
                    "step_seconds": elapsed,
                    "data_hash_rank0_last_microbatch": data_hash,
                    "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                    "wall_seconds": time.time() - training_started,
                    **tau_stats,
                },
            )

        should_evaluate = (step + 1) % args.eval_every == 0 or step == total_steps - 1
        if should_evaluate:
            if distributed:
                dist.barrier()
            if rank == 0:
                natural = evaluate_natural(
                    model, natural_validation_stream, args.natural_eval_samples, device
                )
                retrieval = evaluate_retrieval(
                    model,
                    retrieval_validation_stream,
                    special,
                    parse_lengths(args.eval_lengths),
                    args.eval_samples,
                    device,
                    args.seed + step,
                )
                append_jsonl(
                    output_dir / "eval.jsonl",
                    {"step": step, "final": False, "natural": natural, "retrieval": retrieval},
                )
            if distributed:
                dist.barrier()

        should_save = not args.skip_checkpoints and ((step + 1) % args.save_every == 0 or step == total_steps - 1)
        if should_save and rank == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, temporary)
            temporary.replace(checkpoint_path)
        if distributed and should_save:
            dist.barrier()

    if distributed:
        dist.barrier()
    if rank == 0:
        natural = evaluate_natural(
            model, natural_validation_stream, args.natural_eval_samples, device
        )
        retrieval = evaluate_retrieval(
            model,
            retrieval_validation_stream,
            special,
            parse_lengths(args.final_eval_lengths),
            args.eval_samples,
            device,
            args.seed + 999_999,
        )
        append_jsonl(
            output_dir / "eval.jsonl",
            {"step": total_steps - 1, "final": True, "natural": natural, "retrieval": retrieval},
        )
        (output_dir / "DONE").write_text("complete\n", encoding="utf-8")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
