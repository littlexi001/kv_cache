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

from random_quadruple_data import (
    QuadrupleTableSpec,
    ensure_quadruple_file,
    load_quadruple_table,
    validate_quadruple_table,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
FDONG_SCRIPTS_DIR = REPO_ROOT / "fdong" / "scripts"
if str(FDONG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FDONG_SCRIPTS_DIR))


@dataclass
class RandomQuadrupleBatch:
    source: torch.Tensor
    quadruple_indices: torch.Tensor


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
        default=str(PROJECT_DIR / "outputs" / "train"),
    )
    parser.add_argument("--run_name", default="unet8-random-quad-lm")

    parser.add_argument("--token_min", type=int, default=1)
    parser.add_argument("--token_max", type=int, default=1000)
    parser.add_argument("--quadruple_len", type=int, default=4)
    parser.add_argument("--num_quadruples", type=int, default=100_000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--quadruple_file",
        default=str(PROJECT_DIR / "data" / "random_quadruples_1000_100000.pt"),
    )
    parser.add_argument("--quadruple_seed", type=int, default=20_260_518)
    parser.add_argument("--regenerate_quadruple_file", type=str2bool, default=False)
    parser.add_argument("--sample_with_replacement", type=str2bool, default=False)
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

    if args.token_min < 0:
        raise ValueError("--token_min must be >= 0")
    if args.token_max < args.token_min:
        raise ValueError("--token_max must be >= --token_min")
    if args.quadruple_len < 1:
        raise ValueError("--quadruple_len must be >= 1")
    if args.num_quadruples < 1:
        raise ValueError("--num_quadruples must be >= 1")
    if args.seq_len < args.quadruple_len:
        raise ValueError("--seq_len must be >= --quadruple_len")
    if args.seq_len % args.quadruple_len != 0:
        raise ValueError(
            "--seq_len must be divisible by --quadruple_len. "
            "For seq_len=1024 and quadruple_len=4 this gives 256 quadruples."
        )
    if not args.sample_with_replacement and args.num_quadruples < sample_quadruple_count(args):
        raise ValueError(
            f"--num_quadruples={args.num_quadruples} is smaller than the "
            f"{sample_quadruple_count(args)} quadruples needed per sequence. "
            "Enable --sample_with_replacement true or increase --num_quadruples."
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

    quadruple_file = Path(args.quadruple_file)
    if not quadruple_file.is_absolute():
        quadruple_file = PROJECT_DIR / quadruple_file
    args.quadruple_file = str(quadruple_file)
    return args


def make_quadruple_spec(args: argparse.Namespace) -> QuadrupleTableSpec:
    return QuadrupleTableSpec(
        token_min=args.token_min,
        token_max=args.token_max,
        quadruple_len=args.quadruple_len,
        num_quadruples=args.num_quadruples,
        seed=args.quadruple_seed,
    )


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


def required_vocab_size(args: argparse.Namespace) -> int:
    return args.token_max + 1


def validate_vocab(args: argparse.Namespace, vocab_size: int) -> None:
    min_id = args.token_min
    max_id = args.token_max
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Generated token ids must fit vocab size {vocab_size}; got [{min_id}, {max_id}]."
        )


def sample_quadruple_count(args: argparse.Namespace) -> int:
    return args.seq_len // args.quadruple_len


def load_or_create_quadruple_table(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    quadruple_file = Path(args.quadruple_file)
    spec = make_quadruple_spec(args)
    if is_rank0(rank):
        if args.regenerate_quadruple_file or not quadruple_file.exists():
            action = "regenerating" if args.regenerate_quadruple_file else "creating"
            print(f"{action} quadruple table: {quadruple_file}", flush=True)
        ensure_quadruple_file(quadruple_file, spec, overwrite=args.regenerate_quadruple_file)

    if world_size > 1:
        dist.barrier()

    table, metadata = load_quadruple_table(quadruple_file)
    validate_quadruple_table(table, spec, metadata)
    return table, metadata


def sample_quadruple_indices(
    args: argparse.Namespace,
    generator: torch.Generator,
    table_size: int,
) -> torch.Tensor:
    needed = sample_quadruple_count(args)
    if args.sample_with_replacement:
        return torch.randint(0, table_size, (needed,), generator=generator, dtype=torch.long)
    return torch.randperm(table_size, generator=generator, dtype=torch.long)[:needed]


def generate_batch(
    args: argparse.Namespace,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    quadruple_table: torch.Tensor,
) -> RandomQuadrupleBatch:
    sources = []
    indices = []
    table_size = int(quadruple_table.shape[0])

    for _ in range(batch_size):
        sample_indices = sample_quadruple_indices(args, generator, table_size)
        source = quadruple_table.index_select(0, sample_indices).reshape(-1)
        sources.append(source)
        indices.append(sample_indices)

    return RandomQuadrupleBatch(
        source=torch.stack(sources).to(device),
        quadruple_indices=torch.stack(indices).to(device),
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
    force_vocab_size = getattr(args, "force_vocab_size", None)
    if force_vocab_size is not None:
        print(
            f"Using checkpoint vocab_size={force_vocab_size} instead of config vocab_size={config.vocab_size}.",
            flush=True,
        )
        config.vocab_size = int(force_vocab_size)
    elif needed_vocab_size > config.vocab_size:
        if not args.auto_resize_vocab:
            raise ValueError(
                f"Synthetic data needs vocab_size >= {needed_vocab_size}, "
                f"but config.vocab_size={config.vocab_size}. "
                "Set --auto_resize_vocab true or reduce token_max."
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
    quadruple_table: torch.Tensor,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    correct = 0.0
    count = 0.0

    for _ in range(args.eval_batches):
        batch = generate_batch(args, args.batch_size, generator, device, quadruple_table)
        with autocast_context(args, device):
            loss, correct_tensor, count_tensor = next_token_forward(model, batch.source, args.train_mode)
        batch_count = float(count_tensor.detach().cpu())
        loss_sum += float(loss.detach().cpu()) * batch_count
        correct += float(correct_tensor.detach().cpu())
        count += batch_count

    reduced = reduce_train_stats([loss_sum, correct, count], device, world_size)
    loss_sum, correct, count = reduced

    if was_training:
        model.train()

    loss = None if count == 0 else loss_sum / count
    return {
        "loss": loss,
        "accuracy": None if count == 0 else correct / count,
        "count": int(count),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_runtime_config(
    ckpt_dir: Path,
    model,
    args: argparse.Namespace,
    quadruple_metadata: dict[str, Any],
) -> None:
    real_model = model.module if isinstance(model, DDP) else model
    payload = {
        "num_hidden_layers": real_model.config.num_hidden_layers,
        "attention_stride_pattern": list(real_model.model.attention_stride_pattern),
        "residual_source_pattern": list(real_model.model.residual_source_pattern),
        "training_objective": "full_sequence_next_token_cross_entropy",
        "train_mode": args.train_mode,
        "data": {
            "token_min": args.token_min,
            "token_max": args.token_max,
            "quadruple_len": args.quadruple_len,
            "num_quadruples": args.num_quadruples,
            "quadruple_file": args.quadruple_file,
            "quadruple_metadata": quadruple_metadata,
            "seq_len": args.seq_len,
            "quadruples_per_sequence": sample_quadruple_count(args),
            "sample_with_replacement": args.sample_with_replacement,
            "loss_positions_per_sample": args.seq_len - 1,
        },
    }
    write_json(ckpt_dir / "runtime_config.json", payload)


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    quadruple_metadata: dict[str, Any],
) -> None:
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
    write_runtime_config(ckpt_dir, model, args, quadruple_metadata)


def dump_samples(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    quadruple_table: torch.Tensor,
) -> None:
    if args.dump_samples <= 0:
        return
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    batch = generate_batch(args, args.dump_samples, generator, device, quadruple_table)
    rows = []
    for idx in range(args.dump_samples):
        rows.append(
            {
                "source": batch.source[idx].detach().cpu().tolist(),
                "quadruple_indices": batch.quadruple_indices[idx].detach().cpu().tolist(),
            }
        )
    write_json(run_dir / "sample_dump.json", {"samples": rows})


def main() -> None:
    args = parse_args()
    device, rank, local_rank, world_size = setup_distributed(args)
    torch.manual_seed(args.seed + rank)

    quadruple_table, quadruple_metadata = load_or_create_quadruple_table(args, rank, world_size)

    run_dir = Path(args.output_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    if is_rank0(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        train_config = dict(vars(args))
        train_config["quadruple_metadata"] = quadruple_metadata
        write_json(run_dir / "train_config.json", train_config)
        log_rank0(rank, "Random quadruple retrieval training")
        log_rank0(rank, f"run_dir={run_dir}")
        log_rank0(rank, f"world_size={world_size}")
        log_rank0(rank, f"device={device}")
        log_rank0(rank, f"num_hidden_layers={args.num_hidden_layers}")
        log_rank0(rank, f"attention_stride_pattern={args.attention_stride_pattern}")
        log_rank0(rank, f"quadruple_file={args.quadruple_file}")
        log_rank0(rank, f"quadruple_table_shape={tuple(quadruple_table.shape)}")
        log_rank0(rank, f"quadruples_per_sequence={sample_quadruple_count(args)}")
        log_rank0(rank, f"loss_positions_per_sample={args.seq_len - 1}")
        log_rank0(rank, f"train_mode={args.train_mode}")
        dump_samples(run_dir, args, device, quadruple_table)

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
            batch = generate_batch(args, args.batch_size, generator, device, quadruple_table)
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
            metrics = evaluate(model, args, device, eval_generator, world_size, quadruple_table)
            row = {"type": "eval", "step": step, **metrics}
            if is_rank0(rank):
                append_jsonl(metrics_path, row)
                print(
                    f"step {step}: eval_loss={metrics['loss']:.6f} "
                    f"eval_acc={metrics['accuracy']:.4f}",
                    flush=True,
                )

        if is_rank0(rank) and args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, args, quadruple_metadata)
            print(f"saved checkpoint: {ckpt_dir / f'{step}.pth'}", flush=True)

    if is_rank0(rank):
        save_checkpoint(ckpt_dir, args.total_steps, model, optimizer, scheduler, args, quadruple_metadata)
        print(f"finished. metrics={metrics_path}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
