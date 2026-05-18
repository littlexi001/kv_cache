from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[4]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
if str(FDONG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FDONG_SCRIPTS_DIR))


@dataclass
class IntervalBatch:
    source: torch.Tensor
    interval: torch.Tensor


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train"),
    )
    parser.add_argument("--run_name", default="unet8-interval1-lm")

    parser.add_argument("--total_token", type=int, default=10_000)
    parser.add_argument("--subseq_len", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--intervals", type=parse_int_list, default=[1])
    parser.add_argument(
        "--interval_group_mode",
        choices=["scaled", "bounded"],
        default="scaled",
        help=(
            "scaled keeps total_token/subseq_len candidate groups for every interval, "
            "so interval=2 gives 2500 groups and token ids can reach total_token * 2. "
            "bounded keeps generated content token ids <= total_token."
        ),
    )
    parser.add_argument("--sample_with_replacement", type=str2bool, default=True)
    parser.add_argument("--dump_samples", type=int, default=0)

    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--attention_stride_pattern", type=parse_int_list, default=[1, 1, 4, 4, 4, 4, 1, 1])
    parser.add_argument("--residual_source_pattern", type=parse_int_list, default=None)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument(
        "--auto_resize_vocab",
        type=str2bool,
        default=True,
        help="Increase config.vocab_size when synthetic token ids exceed the base Qwen vocabulary.",
    )
    parser.add_argument(
        "--train_mode",
        choices=["full_sequence_lm"],
        default="full_sequence_lm",
        help="Standard causal LM objective over all next-token positions in the 1024-token sequence.",
    )

    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()

    if not args.intervals:
        raise ValueError("--intervals must contain at least one positive integer.")
    if any(interval <= 0 for interval in args.intervals):
        raise ValueError("All intervals must be positive integers.")
    if args.total_token < 1:
        raise ValueError("--total_token must be >= 1")
    if args.subseq_len < 1:
        raise ValueError("--subseq_len must be >= 1")
    if args.seq_len < args.subseq_len:
        raise ValueError("--seq_len must be >= --subseq_len")
    if args.seq_len % args.subseq_len != 0:
        raise ValueError(
            "--seq_len must be divisible by --subseq_len. "
            "For seq_len=1024 and subseq_len=4 this gives 256 subsequences."
        )
    if args.num_hidden_layers < 1:
        raise ValueError("--num_hidden_layers must be >= 1")
    if args.attention_stride_pattern and len(args.attention_stride_pattern) != args.num_hidden_layers:
        raise ValueError(
            "--attention_stride_pattern length must match --num_hidden_layers. "
            f"got {len(args.attention_stride_pattern)} vs {args.num_hidden_layers}"
        )
    if args.residual_source_pattern and len(args.residual_source_pattern) != args.num_hidden_layers:
        raise ValueError(
            "--residual_source_pattern length must match --num_hidden_layers. "
            f"got {len(args.residual_source_pattern)} vs {args.num_hidden_layers}"
        )
    return args


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    if is_distributed():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), rank, local_rank, world_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    return device, 0, 0, 1


def is_rank0(rank: int) -> bool:
    return rank == 0


def log_rank0(rank: int, message: str) -> None:
    if is_rank0(rank):
        print(message, flush=True)


def content_max_token_id(args: argparse.Namespace) -> int:
    if args.interval_group_mode == "bounded":
        return args.total_token
    return args.total_token * max(args.intervals)


def sample_group_count(args: argparse.Namespace) -> int:
    return args.seq_len // args.subseq_len


def group_count_for_interval(args: argparse.Namespace, interval: int) -> int:
    if args.interval_group_mode == "bounded":
        return args.total_token // (interval * args.subseq_len)
    return args.total_token // args.subseq_len


def validate_vocab(args: argparse.Namespace, vocab_size: int) -> None:
    max_content = content_max_token_id(args)
    max_id = max_content
    min_id = 1
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Generated token ids must fit vocab size {vocab_size}; got [{min_id}, {max_id}]."
        )


def required_vocab_size(args: argparse.Namespace) -> int:
    return content_max_token_id(args) + 1


def pattern_from_group_ids(group_ids: torch.Tensor, interval: int, subseq_len: int) -> torch.Tensor:
    offsets = torch.arange(1, subseq_len + 1, dtype=torch.long)
    return interval * (group_ids.unsqueeze(1) * subseq_len + offsets.unsqueeze(0))


def sample_group_ids(
    args: argparse.Namespace,
    interval: int,
    generator: torch.Generator,
) -> torch.Tensor:
    num_groups = group_count_for_interval(args, interval)
    needed = sample_group_count(args)
    if num_groups < 1:
        raise ValueError(
            f"interval={interval} produces no groups. "
            "Use a smaller interval, larger total_token, or interval_group_mode=scaled."
        )
    if num_groups >= needed:
        return torch.randperm(num_groups, generator=generator, dtype=torch.long)[:needed]
    if not args.sample_with_replacement:
        raise ValueError(
            f"interval={interval} only has {num_groups} groups but {needed} are needed. "
            "Enable --sample_with_replacement true or use a smaller interval."
        )
    return torch.randint(0, num_groups, (needed,), generator=generator, dtype=torch.long)


def generate_batch(
    args: argparse.Namespace,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> IntervalBatch:
    sources = []
    intervals = []

    for _ in range(batch_size):
        interval_idx = int(torch.randint(0, len(args.intervals), (1,), generator=generator).item())
        interval = int(args.intervals[interval_idx])
        group_ids = sample_group_ids(args, interval, generator)
        source = pattern_from_group_ids(group_ids, interval, args.subseq_len).reshape(-1)

        sources.append(source)
        intervals.append(interval)

    return IntervalBatch(
        source=torch.stack(sources).to(device),
        interval=torch.tensor(intervals, dtype=torch.long, device=device),
    )


def prepare_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoConfig

    from models import MyQwen3ForCausalLM

    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    config._attn_implementation = args.attn_implementation
    config.num_hidden_layers = args.num_hidden_layers
    config.attention_stride_pattern = args.attention_stride_pattern or [1] * args.num_hidden_layers
    config.residual_source_pattern = args.residual_source_pattern or [-1] * args.num_hidden_layers
    needed_vocab_size = required_vocab_size(args)
    if needed_vocab_size > config.vocab_size:
        if not args.auto_resize_vocab:
            raise ValueError(
                f"Synthetic data needs vocab_size >= {needed_vocab_size}, "
                f"but config.vocab_size={config.vocab_size}. "
                "Set --auto_resize_vocab true or reduce total_token/intervals."
            )
        print(
            f"Resizing config.vocab_size from {config.vocab_size} to {needed_vocab_size} "
            "for synthetic token ids.",
            flush=True,
        )
        config.vocab_size = needed_vocab_size

    model = MyQwen3ForCausalLM(config).to(device)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    validate_vocab(args, model.config.vocab_size)
    return model


def prepare_optimizer(model, args: argparse.Namespace):
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )
    return optimizer, scheduler


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.use_bf16 and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def next_token_forward(
    model,
    source: torch.Tensor,
    train_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_mode != "full_sequence_lm":
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    output = model(input_ids=source, use_cache=False, output_hidden_states=False)
    logits = output.logits[:, :-1, :]
    labels = source[:, 1:]
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels).sum()
    count = torch.tensor(labels.numel(), device=source.device)
    return loss, correct, count


def reduce_train_stats(values: list[float], device: torch.device, world_size: int) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.detach().cpu().tolist()


@torch.no_grad()
def evaluate(
    model,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    world_size: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    correct = 0.0
    count = 0.0
    interval_counts = {str(interval): 0 for interval in args.intervals}

    for _ in range(args.eval_batches):
        batch = generate_batch(args, args.batch_size, generator, device)
        with autocast_context(args, device):
            loss, correct_tensor, count_tensor = next_token_forward(model, batch.source, args.train_mode)
        batch_count = float(count_tensor.detach().cpu())
        loss_sum += float(loss.detach().cpu()) * batch_count
        correct += float(correct_tensor.detach().cpu())
        count += batch_count
        for interval in args.intervals:
            interval_counts[str(interval)] += int(batch.interval.eq(interval).sum().detach().cpu()) * (args.seq_len - 1)

    reduced = reduce_train_stats([loss_sum, correct, count], device, world_size)
    loss_sum, correct, count = reduced

    interval_values = reduce_train_stats(
        [float(interval_counts[str(interval)]) for interval in args.intervals],
        device,
        world_size,
    )
    interval_counts = {
        str(interval): int(value)
        for interval, value in zip(args.intervals, interval_values)
    }

    if was_training:
        model.train()

    loss = None if count == 0 else loss_sum / count
    return {
        "loss": loss,
        "accuracy": None if count == 0 else correct / count,
        "count": int(count),
        "interval_counts": interval_counts,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_runtime_config(ckpt_dir: Path, model, args: argparse.Namespace) -> None:
    real_model = model.module if isinstance(model, DDP) else model
    payload = {
        "num_hidden_layers": real_model.config.num_hidden_layers,
        "attention_stride_pattern": list(real_model.model.attention_stride_pattern),
        "residual_source_pattern": list(real_model.model.residual_source_pattern),
        "training_objective": "full_sequence_next_token_cross_entropy",
        "train_mode": args.train_mode,
        "data": {
            "total_token": args.total_token,
            "subseq_len": args.subseq_len,
            "seq_len": args.seq_len,
            "intervals": args.intervals,
            "interval_group_mode": args.interval_group_mode,
            "sample_group_count": sample_group_count(args),
            "loss_positions_per_sample": args.seq_len - 1,
        },
    }
    write_json(ckpt_dir / "runtime_config.json", payload)


def save_checkpoint(ckpt_dir: Path, step: int, model, optimizer, scheduler, args: argparse.Namespace) -> None:
    real_model = model.module if isinstance(model, DDP) else model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(real_model.state_dict(), ckpt_dir / f"{step}.pth")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / f"{step}.optim.pth",
    )
    write_runtime_config(ckpt_dir, model, args)


def dump_samples(run_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    if args.dump_samples <= 0:
        return
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    batch = generate_batch(args, args.dump_samples, generator, device)
    rows = []
    for idx in range(args.dump_samples):
        rows.append(
            {
                "source": batch.source[idx].detach().cpu().tolist(),
                "interval": int(batch.interval[idx].detach().cpu()),
            }
        )
    write_json(run_dir / "sample_dump.json", {"samples": rows})


def main() -> None:
    args = parse_args()
    device, rank, local_rank, world_size = setup_distributed(args)
    torch.manual_seed(args.seed + rank)

    run_dir = Path(args.output_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    if is_rank0(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "train_config.json", vars(args))
        log_rank0(rank, "Interval subsequence retrieval training")
        log_rank0(rank, f"run_dir={run_dir}")
        log_rank0(rank, f"world_size={world_size}")
        log_rank0(rank, f"device={device}")
        log_rank0(rank, f"num_hidden_layers={args.num_hidden_layers}")
        log_rank0(rank, f"attention_stride_pattern={args.attention_stride_pattern}")
        log_rank0(rank, f"intervals={args.intervals}")
        log_rank0(rank, f"sample_group_count={sample_group_count(args)}")
        log_rank0(rank, f"loss_positions_per_sample={args.seq_len - 1}")
        log_rank0(rank, f"train_mode={args.train_mode}")
        dump_samples(run_dir, args, device)

    model = prepare_model(args, device)
    model.train()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer, scheduler = prepare_optimizer(model, args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank * 1_000_003)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 99_999 + rank * 1_000_003)

    optimizer.zero_grad(set_to_none=True)
    rolling_loss_sum = 0.0
    rolling_correct = 0.0
    rolling_count = 0.0
    started_at = time.monotonic()

    for step in range(1, args.total_steps + 1):
        for micro_step in range(args.gradient_accumulation_steps):
            batch = generate_batch(args, args.batch_size, generator, device)
            sync_context = (
                model.no_sync()
                if world_size > 1 and micro_step < args.gradient_accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_context(args, device):
                    loss, correct_tensor, count_tensor = next_token_forward(model, batch.source, args.train_mode)
                (loss / args.gradient_accumulation_steps).backward()

            batch_count = float(count_tensor.detach().cpu())
            rolling_loss_sum += float(loss.detach().cpu()) * batch_count
            rolling_correct += float(correct_tensor.detach().cpu())
            rolling_count += batch_count

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_interval == 0:
            loss_sum, correct, count = reduce_train_stats(
                [rolling_loss_sum, rolling_correct, rolling_count],
                device,
                world_size,
            )
            elapsed = max(time.monotonic() - started_at, 1e-6)
            row = {
                "type": "train",
                "step": step,
                "loss": None if count == 0 else loss_sum / count,
                "accuracy": None if count == 0 else correct / count,
                "count": int(count),
                "lr": scheduler.get_last_lr()[0],
                "predictions_per_second": count / elapsed,
            }
            if is_rank0(rank):
                append_jsonl(metrics_path, row)
                print(
                    f"step {step}: train_loss={row['loss']:.6f} "
                    f"train_acc={row['accuracy']:.4f} lr={row['lr']:.3e}",
                    flush=True,
                )
            rolling_loss_sum = 0.0
            rolling_correct = 0.0
            rolling_count = 0.0
            started_at = time.monotonic()

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            metrics = evaluate(model, args, device, eval_generator, world_size)
            row = {"type": "eval", "step": step, **metrics}
            if is_rank0(rank):
                append_jsonl(metrics_path, row)
                print(
                    f"step {step}: eval_loss={metrics['loss']:.6f} "
                    f"eval_acc={metrics['accuracy']:.4f}",
                    flush=True,
                )

        if is_rank0(rank) and args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, args)
            print(f"saved checkpoint: {ckpt_dir / f'{step}.pth'}", flush=True)

    if is_rank0(rank):
        save_checkpoint(ckpt_dir, args.total_steps, model, optimizer, scheduler, args)
        print(f"finished. metrics={metrics_path}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
