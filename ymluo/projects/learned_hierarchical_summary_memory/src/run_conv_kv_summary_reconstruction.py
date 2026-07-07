from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    output_dir: str
    modes: tuple[str, ...]
    seq_len: int
    dim: int
    block_size: int
    slots_per_block: int
    query_count: int
    train_samples: int
    test_samples: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    device: str


@dataclass
class ResultRow:
    mode: str
    method: str
    seq_len: int
    active_kv: int
    active_ratio: float
    block_size: int
    slots_per_block: int
    mse: float
    relative_mse: float
    cosine: float
    train_seconds: float
    final_train_loss: float | None


@dataclass(frozen=True)
class Dataset:
    k_train: torch.Tensor
    v_train: torch.Tensor
    q_train: torch.Tensor
    y_train: torch.Tensor
    k_test: torch.Tensor
    v_test: torch.Tensor
    q_test: torch.Tensor
    y_test: torch.Tensor


def parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a Conv1D K/V summary compressor on attention-output "
            "reconstruction tasks."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--modes", type=parse_modes, default=("smooth_local", "needle_exact"))
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--slots_per_block", type=int, default=1)
    parser.add_argument("--query_count", type=int, default=16)
    parser.add_argument("--train_samples", type=int, default=256)
    parser.add_argument("--test_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026070608)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    return Config(**vars(args))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def mean_pool_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    slots_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size % slots_per_block != 0:
        raise ValueError("block_size must be divisible by slots_per_block for fair pooling baselines")
    batch, seq_len, dim = k.shape
    blocks = seq_len // block_size
    usable = blocks * block_size
    subblock = block_size // slots_per_block
    k_out = (
        k[:, :usable, :]
        .reshape(batch, blocks, slots_per_block, subblock, dim)
        .mean(dim=3)
        .reshape(batch, blocks * slots_per_block, dim)
    )
    v_out = (
        v[:, :usable, :]
        .reshape(batch, blocks, slots_per_block, subblock, dim)
        .mean(dim=3)
        .reshape(batch, blocks * slots_per_block, dim)
    )
    return k_out, v_out


def last_token_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    slots_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size % slots_per_block != 0:
        raise ValueError("block_size must be divisible by slots_per_block for fair pooling baselines")
    batch, seq_len, dim = k.shape
    blocks = seq_len // block_size
    usable = blocks * block_size
    subblock = block_size // slots_per_block
    k_out = (
        k[:, :usable, :]
        .reshape(batch, blocks, slots_per_block, subblock, dim)[:, :, :, -1, :]
        .reshape(batch, blocks * slots_per_block, dim)
    )
    v_out = (
        v[:, :usable, :]
        .reshape(batch, blocks, slots_per_block, subblock, dim)[:, :, :, -1, :]
        .reshape(batch, blocks * slots_per_block, dim)
    )
    return k_out, v_out


class ConvKVCompressor(nn.Module):
    def __init__(self, dim: int, block_size: int, slots_per_block: int) -> None:
        super().__init__()
        if slots_per_block < 1:
            raise ValueError("slots_per_block must be positive")
        self.dim = dim
        self.block_size = block_size
        self.slots_per_block = slots_per_block
        self.conv = nn.Conv1d(
            in_channels=2 * dim,
            out_channels=2 * dim * slots_per_block,
            kernel_size=block_size,
            stride=block_size,
            bias=True,
        )
        self.reset_to_mean_pool()

    def reset_to_mean_pool(self) -> None:
        with torch.no_grad():
            if self.block_size % self.slots_per_block != 0:
                raise ValueError("block_size must be divisible by slots_per_block")
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            subblock = self.block_size // self.slots_per_block
            scale = 1.0 / float(subblock)
            for slot in range(self.slots_per_block):
                start = slot * subblock
                end = start + subblock
                for channel in range(2 * self.dim):
                    out_channel = slot * 2 * self.dim + channel
                    self.conv.weight[out_channel, channel, start:end].fill_(scale)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([k, v], dim=-1).transpose(1, 2)
        y = self.conv(x).transpose(1, 2)
        batch, blocks, _ = y.shape
        y = y.reshape(batch, blocks, self.slots_per_block, 2 * self.dim)
        y = y.reshape(batch, blocks * self.slots_per_block, 2 * self.dim)
        k_out, v_out = y.split(self.dim, dim=-1)
        return k_out, v_out


def smooth_1d(x: torch.Tensor, passes: int = 3) -> torch.Tensor:
    y = x
    for _ in range(passes):
        left = F.pad(y[:, :-1, :], (0, 0, 1, 0))
        right = F.pad(y[:, 1:, :], (0, 0, 0, 1))
        y = 0.25 * left + 0.5 * y + 0.25 * right
    return y


def make_smooth_local(
    samples: int,
    seq_len: int,
    dim: int,
    query_count: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = smooth_1d(torch.randn(samples, seq_len, dim, generator=generator, device=device), passes=5)
    topic = F.normalize(latent + 0.05 * torch.randn_like(latent), dim=-1)
    value_mix = torch.randn(dim, dim, generator=generator, device=device) / math.sqrt(dim)
    v = torch.tanh(latent @ value_mix) + 0.02 * torch.randn(samples, seq_len, dim, generator=generator, device=device)

    blocks = seq_len // block_size
    block_ids = torch.randint(0, blocks, (samples, query_count), generator=generator, device=device)
    # Queries target the later half of a local phrase. This makes uniform mean pooling
    # intentionally imperfect while keeping the task locally compressible.
    offsets = torch.randint(block_size // 2, block_size, (samples, query_count), generator=generator, device=device)
    positions = block_ids * block_size + offsets
    q = topic.gather(1, positions.unsqueeze(-1).expand(-1, -1, dim))
    q = q + 0.08 * torch.randn(samples, query_count, dim, generator=generator, device=device)
    q = F.normalize(q, dim=-1) * math.sqrt(dim)
    k = F.normalize(topic, dim=-1) * math.sqrt(dim)
    return k, v, q


def make_needle_exact(
    samples: int,
    seq_len: int,
    dim: int,
    query_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = F.normalize(torch.randn(samples, seq_len, dim, generator=generator, device=device), dim=-1)
    v = 0.15 * torch.randn(samples, seq_len, dim, generator=generator, device=device)
    needle_pos = torch.randint(0, seq_len, (samples,), generator=generator, device=device)
    needle_key = F.normalize(torch.randn(samples, dim, generator=generator, device=device), dim=-1)
    needle_value = torch.randn(samples, dim, generator=generator, device=device)
    batch_idx = torch.arange(samples, device=device)
    k[batch_idx, needle_pos, :] = needle_key * math.sqrt(dim) * 1.7
    v[batch_idx, needle_pos, :] = needle_value
    q = needle_key[:, None, :].expand(samples, query_count, dim).clone()
    q = q + 0.03 * torch.randn(samples, query_count, dim, generator=generator, device=device)
    q = F.normalize(q, dim=-1) * math.sqrt(dim) * 1.7
    return k, v, q


def make_dataset(mode: str, config: Config, device: torch.device) -> Dataset:
    generator = torch.Generator(device=device)
    mode_seed = {"smooth_local": 11, "needle_exact": 23}.get(mode, 97)
    generator.manual_seed(config.seed + mode_seed)
    total = config.train_samples + config.test_samples
    if mode == "smooth_local":
        k, v, q = make_smooth_local(
            total,
            config.seq_len,
            config.dim,
            config.query_count,
            config.block_size,
            generator,
            device,
        )
    elif mode == "needle_exact":
        k, v, q = make_needle_exact(total, config.seq_len, config.dim, config.query_count, generator, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    y = attention(q, k, v).detach()
    train = slice(0, config.train_samples)
    test = slice(config.train_samples, total)
    return Dataset(
        k_train=k[train],
        v_train=v[train],
        q_train=q[train],
        y_train=y[train],
        k_test=k[test],
        v_test=v[test],
        q_test=q[test],
        y_test=y[test],
    )


def train_conv(
    dataset: Dataset,
    config: Config,
    device: torch.device,
) -> tuple[ConvKVCompressor, float, float]:
    model = ConvKVCompressor(config.dim, config.block_size, config.slots_per_block).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    samples = dataset.k_train.shape[0]
    final_loss = 0.0
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        order = torch.randperm(samples, device=device)
        for start_idx in range(0, samples, config.batch_size):
            idx = order[start_idx : start_idx + config.batch_size]
            k = dataset.k_train.index_select(0, idx)
            v = dataset.v_train.index_select(0, idx)
            q = dataset.q_train.index_select(0, idx)
            target = dataset.y_train.index_select(0, idx)
            k_comp, v_comp = model(k, v)
            pred = attention(q, k_comp, v_comp)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        # Keep runtime short while still making progress on CPU.
        if epoch >= 1 and final_loss < 1e-5:
            break
    synchronize(device)
    train_seconds = time.perf_counter() - start
    return model, train_seconds, final_loss


@torch.no_grad()
def evaluate_method(
    mode: str,
    method: str,
    config: Config,
    dataset: Dataset,
    compressor: ConvKVCompressor | None,
    train_seconds: float,
    final_train_loss: float | None,
) -> ResultRow:
    if method == "mean_pool":
        k_comp, v_comp = mean_pool_kv(
            dataset.k_test,
            dataset.v_test,
            config.block_size,
            config.slots_per_block,
        )
    elif method == "last_token":
        k_comp, v_comp = last_token_kv(
            dataset.k_test,
            dataset.v_test,
            config.block_size,
            config.slots_per_block,
        )
    elif method == "trained_conv":
        if compressor is None:
            raise ValueError("trained_conv requires a compressor")
        k_comp, v_comp = compressor(dataset.k_test, dataset.v_test)
    else:
        raise ValueError(method)

    pred = attention(dataset.q_test, k_comp, v_comp)
    target = dataset.y_test
    mse = float(F.mse_loss(pred, target).cpu())
    denom = float(target.pow(2).mean().clamp_min(1e-12).cpu())
    cosine = float(F.cosine_similarity(pred.reshape(-1, config.dim), target.reshape(-1, config.dim), dim=-1).mean().cpu())
    active_kv = int(k_comp.shape[1])
    return ResultRow(
        mode=mode,
        method=method,
        seq_len=config.seq_len,
        active_kv=active_kv,
        active_ratio=active_kv / float(config.seq_len),
        block_size=config.block_size,
        slots_per_block=config.slots_per_block,
        mse=mse,
        relative_mse=mse / denom,
        cosine=cosine,
        train_seconds=train_seconds,
        final_train_loss=final_train_loss,
    )


def summarize_rows(rows: list[ResultRow]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    by_mode: dict[str, list[ResultRow]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row)
    for mode, items in sorted(by_mode.items()):
        mean_row = next(row for row in items if row.method == "mean_pool")
        conv_row = next(row for row in items if row.method == "trained_conv")
        summary.append(
            {
                "mode": mode,
                "conv_relative_mse": conv_row.relative_mse,
                "mean_relative_mse": mean_row.relative_mse,
                "conv_vs_mean_mse_ratio": conv_row.mse / max(mean_row.mse, 1e-12),
                "conv_cosine": conv_row.cosine,
                "mean_cosine": mean_row.cosine,
                "active_ratio": conv_row.active_ratio,
            }
        )
    return summary


def main() -> None:
    config = parse_args()
    if config.seq_len % config.block_size != 0:
        raise ValueError("--seq_len must be divisible by --block_size")
    if config.block_size % config.slots_per_block != 0:
        raise ValueError("--block_size must be divisible by --slots_per_block")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    torch.manual_seed(config.seed)

    rows: list[ResultRow] = []
    start_wall = time.perf_counter()
    for mode in config.modes:
        dataset = make_dataset(mode, config, device)
        compressor, train_seconds, final_loss = train_conv(dataset, config, device)
        rows.append(evaluate_method(mode, "mean_pool", config, dataset, None, 0.0, None))
        rows.append(evaluate_method(mode, "last_token", config, dataset, None, 0.0, None))
        rows.append(evaluate_method(mode, "trained_conv", config, dataset, compressor, train_seconds, final_loss))
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "conv_kv_reconstruction.csv", [asdict(row) for row in rows])
    summary = summarize_rows(rows)
    write_csv(output_dir / "summary.csv", summary)
    payload = {
        "config": asdict(config),
        "device": str(device),
        "wall_seconds": wall_seconds,
        "rows": [asdict(row) for row in rows],
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("mode,method,active_ratio,relative_mse,cosine,train_seconds")
    for row in rows:
        print(
            f"{row.mode},{row.method},{row.active_ratio:.4f},"
            f"{row.relative_mse:.6f},{row.cosine:.6f},{row.train_seconds:.3f}"
        )
    print("\nmode,conv_vs_mean_mse_ratio,conv_cosine,mean_cosine")
    for row in summary:
        print(
            f"{row['mode']},{row['conv_vs_mean_mse_ratio']:.6f},"
            f"{row['conv_cosine']:.6f},{row['mean_cosine']:.6f}"
        )


if __name__ == "__main__":
    main()
