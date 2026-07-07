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
    latent_dim: int
    query_count: int
    train_samples: int
    test_samples: int
    batch_size: int
    ae_epochs: int
    search_epochs: int
    joint_search_weight: float
    block_search_weight: float
    attention_loss_weight: float
    topk_score_weight: float
    score_temperature: float
    score_topk: int
    rare_recon_weight: float
    rare_token_fraction: float
    lr: float
    seed: int
    device: str


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


@dataclass
class ResultRow:
    mode: str
    seq_len: int
    block_size: int
    blocks: int
    dim: int
    latent_dim: int
    latent_seq_ratio: float
    latent_storage_ratio_vs_kv: float
    recon_k_relative_mse: float
    recon_v_relative_mse: float
    recon_attention_relative_mse: float
    recon_attention_cosine: float
    recon_token_top1_recall: float
    recon_token_top5_recall: float
    mean_block_top1_recall: float
    mean_block_top3_recall: float
    recon_block_top1_recall: float
    recon_block_top3_recall: float
    latent_block_top1_recall: float
    latent_block_top3_recall: float
    joint_search_weight: float
    block_search_weight: float
    attention_loss_weight: float
    topk_score_weight: float
    rare_recon_weight: float
    rare_token_fraction: float
    ae_final_loss: float
    ae_final_recon_loss: float
    ae_final_attention_loss: float
    ae_final_block_loss: float
    ae_final_topk_loss: float
    ae_final_rare_loss: float
    search_final_loss: float
    train_seconds: float


@dataclass
class TrainStats:
    total_loss: float
    recon_loss: float
    attention_loss: float
    block_loss: float
    topk_loss: float
    rare_loss: float


def parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a seq down/up autoencoder and test whether low-MSE reconstruction "
            "preserves attention search/top-k ranking."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--modes", type=parse_modes, default=("smooth_local", "needle_exact"))
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--query_count", type=int, default=16)
    parser.add_argument("--train_samples", type=int, default=256)
    parser.add_argument("--test_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ae_epochs", type=int, default=10)
    parser.add_argument("--search_epochs", type=int, default=6)
    parser.add_argument("--joint_search_weight", type=float, default=0.0)
    parser.add_argument(
        "--block_search_weight",
        type=float,
        default=0.0,
        help="Additional block-search CE weight. Added to --joint_search_weight for backward compatibility.",
    )
    parser.add_argument("--attention_loss_weight", type=float, default=0.0)
    parser.add_argument("--topk_score_weight", type=float, default=0.0)
    parser.add_argument("--score_temperature", type=float, default=2.0)
    parser.add_argument("--score_topk", type=int, default=4)
    parser.add_argument("--rare_recon_weight", type=float, default=0.0)
    parser.add_argument("--rare_token_fraction", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026070610)
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


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    denom = target.pow(2).mean().clamp_min(1e-12)
    return float((mse / denom).detach().cpu())


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
    # Mostly smooth sequence, with one rare high-value needle per sample.
    # This makes global reconstruction MSE look deceptively reasonable while
    # exact search can still fail.
    base = smooth_1d(torch.randn(samples, seq_len, dim, generator=generator, device=device), passes=4)
    k = F.normalize(base + 0.05 * torch.randn_like(base), dim=-1) * math.sqrt(dim)
    value_mix = torch.randn(dim, dim, generator=generator, device=device) / math.sqrt(dim)
    v = torch.tanh(base @ value_mix)
    needle_pos = torch.randint(0, seq_len, (samples,), generator=generator, device=device)
    needle_key = F.normalize(torch.randn(samples, dim, generator=generator, device=device), dim=-1)
    needle_value = torch.randn(samples, dim, generator=generator, device=device)
    batch_idx = torch.arange(samples, device=device)
    k[batch_idx, needle_pos, :] = needle_key * math.sqrt(dim) * 1.8
    v[batch_idx, needle_pos, :] = needle_value
    q = needle_key[:, None, :].expand(samples, query_count, dim).clone()
    q = q + 0.03 * torch.randn(samples, query_count, dim, generator=generator, device=device)
    q = F.normalize(q, dim=-1) * math.sqrt(dim) * 1.8
    return k, v, q


def make_dataset(mode: str, config: Config, device: torch.device) -> Dataset:
    generator = torch.Generator(device=device)
    mode_seed = {"smooth_local": 101, "needle_exact": 211}.get(mode, 307)
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


class SeqAutoencoder(nn.Module):
    def __init__(self, dim: int, block_size: int, latent_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * dim, 2 * dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(2 * dim, latent_dim, kernel_size=block_size, stride=block_size),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 2 * dim, kernel_size=block_size, stride=block_size),
            nn.GELU(),
            nn.Conv1d(2 * dim, 2 * dim, kernel_size=3, padding=1),
        )

    def encode(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([k, v], dim=-1).transpose(1, 2)
        return self.encoder(x).transpose(1, 2)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.decoder(z.transpose(1, 2)).transpose(1, 2)
        k_recon, v_recon = y.split(self.dim, dim=-1)
        return k_recon, v_recon

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(k, v)
        k_recon, v_recon = self.decode(z)
        return z, k_recon, v_recon


class LatentSearcher(nn.Module):
    def __init__(self, dim: int, latent_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, latent_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        q_latent = F.normalize(self.q_proj(q), dim=-1)
        z_norm = F.normalize(z, dim=-1)
        return torch.matmul(q_latent, z_norm.transpose(-1, -2)) * self.logit_scale.exp()


def block_scores_from_token_scores(scores: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, queries, seq_len = scores.shape
    blocks = seq_len // block_size
    return scores[:, :, : blocks * block_size].reshape(batch, queries, blocks, block_size).amax(dim=-1)


def block_targets(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    token_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    return block_scores_from_token_scores(token_scores, block_size).argmax(dim=-1)


def block_scores(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    token_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    return block_scores_from_token_scores(token_scores, block_size)


def rare_reconstruction_loss(
    k_recon: torch.Tensor,
    v_recon: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_fraction: float,
) -> torch.Tensor:
    batch, seq_len, _ = k.shape
    count = max(1, min(seq_len, int(math.ceil(seq_len * token_fraction))))
    token_energy = k.float().norm(dim=-1) + v.float().norm(dim=-1)
    idx = token_energy.topk(count, dim=-1).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, k.shape[-1])
    k_err = (k_recon - k).pow(2).gather(1, gather_idx).mean()
    v_err = (v_recon - v).pow(2).gather(1, gather_idx).mean()
    return k_err + v_err


def topk_score_distill_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    topk = max(1, min(topk, teacher_scores.shape[-1]))
    idx = teacher_scores.topk(topk, dim=-1).indices
    teacher = teacher_scores.gather(-1, idx) / max(temperature, 1e-6)
    student = student_scores.gather(-1, idx) / max(temperature, 1e-6)
    teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    student = student - student.mean(dim=-1, keepdim=True)
    return F.mse_loss(student, teacher.detach())


def search_loss_enabled(config: Config) -> bool:
    return (
        config.joint_search_weight > 0
        or config.block_search_weight > 0
        or config.topk_score_weight > 0
    )


def train_autoencoder(
    dataset: Dataset,
    config: Config,
    device: torch.device,
) -> tuple[SeqAutoencoder, LatentSearcher | None, TrainStats]:
    model = SeqAutoencoder(config.dim, config.block_size, config.latent_dim).to(device)
    joint_searcher = LatentSearcher(config.dim, config.latent_dim).to(device) if search_loss_enabled(config) else None
    params = list(model.parameters())
    if joint_searcher is not None:
        params.extend(joint_searcher.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-4)
    samples = dataset.k_train.shape[0]
    final = TrainStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    block_weight = config.joint_search_weight + config.block_search_weight
    for _ in range(config.ae_epochs):
        order = torch.randperm(samples, device=device)
        for start in range(0, samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            k = dataset.k_train.index_select(0, idx)
            v = dataset.v_train.index_select(0, idx)
            q = dataset.q_train.index_select(0, idx)
            y = dataset.y_train.index_select(0, idx)
            z, k_recon, v_recon = model(k, v)
            recon_loss = F.mse_loss(k_recon, k) + F.mse_loss(v_recon, v)
            loss = recon_loss
            attention_loss = torch.zeros((), device=device)
            block_loss = torch.zeros((), device=device)
            topk_loss = torch.zeros((), device=device)
            rare_loss = torch.zeros((), device=device)
            if config.attention_loss_weight > 0:
                pred_y = attention(q, k_recon, v_recon)
                attention_loss = F.mse_loss(pred_y, y)
                loss = loss + config.attention_loss_weight * attention_loss
            if config.rare_recon_weight > 0:
                rare_loss = rare_reconstruction_loss(
                    k_recon,
                    v_recon,
                    k,
                    v,
                    config.rare_token_fraction,
                )
                loss = loss + config.rare_recon_weight * rare_loss
            if joint_searcher is not None:
                search_scores = joint_searcher(q, z)
                if block_weight > 0:
                    target = block_targets(q, k, config.block_size)
                    block_loss = F.cross_entropy(search_scores.reshape(-1, search_scores.shape[-1]), target.reshape(-1))
                    loss = loss + block_weight * block_loss
                if config.topk_score_weight > 0:
                    teacher_scores = block_scores(q, k, config.block_size)
                    topk_loss = topk_score_distill_loss(
                        search_scores,
                        teacher_scores,
                        config.score_topk,
                        config.score_temperature,
                    )
                    loss = loss + config.topk_score_weight * topk_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final = TrainStats(
                total_loss=float(loss.detach().cpu()),
                recon_loss=float(recon_loss.detach().cpu()),
                attention_loss=float(attention_loss.detach().cpu()),
                block_loss=float(block_loss.detach().cpu()),
                topk_loss=float(topk_loss.detach().cpu()),
                rare_loss=float(rare_loss.detach().cpu()),
            )
    return model, joint_searcher, final


def train_searcher(
    model: SeqAutoencoder,
    dataset: Dataset,
    config: Config,
    device: torch.device,
    searcher: LatentSearcher | None = None,
) -> tuple[LatentSearcher, float]:
    searcher = searcher or LatentSearcher(config.dim, config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(searcher.parameters(), lr=config.lr, weight_decay=1e-4)
    samples = dataset.k_train.shape[0]
    final_loss = 0.0
    model.eval()
    for _ in range(config.search_epochs):
        order = torch.randperm(samples, device=device)
        for start in range(0, samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            k = dataset.k_train.index_select(0, idx)
            v = dataset.v_train.index_select(0, idx)
            q = dataset.q_train.index_select(0, idx)
            target = block_targets(q, k, config.block_size)
            with torch.no_grad():
                z = model.encode(k, v)
            scores = searcher(q, z)
            loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return searcher, final_loss


def topk_recall(reference_scores: torch.Tensor, approx_scores: torch.Tensor, k: int) -> float:
    k = min(k, reference_scores.shape[-1], approx_scores.shape[-1])
    ref_idx = reference_scores.topk(k, dim=-1).indices
    app_idx = approx_scores.topk(k, dim=-1).indices
    hits = (ref_idx.unsqueeze(-1) == app_idx.unsqueeze(-2)).any(dim=-1).float()
    return float(hits.mean().detach().cpu())


def block_recall(reference_block_scores: torch.Tensor, approx_block_scores: torch.Tensor, k: int) -> float:
    k = min(k, reference_block_scores.shape[-1], approx_block_scores.shape[-1])
    target = reference_block_scores.argmax(dim=-1, keepdim=True)
    pred = approx_block_scores.topk(k, dim=-1).indices
    return float((pred == target).any(dim=-1).float().mean().detach().cpu())


def mean_block_k(k: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, seq_len, dim = k.shape
    blocks = seq_len // block_size
    return k[:, : blocks * block_size, :].reshape(batch, blocks, block_size, dim).mean(dim=2)


@torch.no_grad()
def evaluate(
    mode: str,
    model: SeqAutoencoder,
    searcher: LatentSearcher,
    dataset: Dataset,
    config: Config,
    train_stats: TrainStats,
    search_final_loss: float,
    train_seconds: float,
) -> ResultRow:
    model.eval()
    searcher.eval()
    z, k_recon, v_recon = model(dataset.k_test, dataset.v_test)
    y_recon = attention(dataset.q_test, k_recon, v_recon)
    full_token_scores = torch.matmul(dataset.q_test, dataset.k_test.transpose(-1, -2)) / math.sqrt(config.dim)
    recon_token_scores = torch.matmul(dataset.q_test, k_recon.transpose(-1, -2)) / math.sqrt(config.dim)
    full_block_scores = block_scores_from_token_scores(full_token_scores, config.block_size)
    recon_block_scores = block_scores_from_token_scores(recon_token_scores, config.block_size)
    mean_k = mean_block_k(dataset.k_test, config.block_size)
    mean_block_scores = torch.matmul(dataset.q_test, mean_k.transpose(-1, -2)) / math.sqrt(config.dim)
    latent_block_scores = searcher(dataset.q_test, z)
    attn_cosine = float(
        F.cosine_similarity(y_recon.reshape(-1, config.dim), dataset.y_test.reshape(-1, config.dim), dim=-1)
        .mean()
        .detach()
        .cpu()
    )
    blocks = config.seq_len // config.block_size
    return ResultRow(
        mode=mode,
        seq_len=config.seq_len,
        block_size=config.block_size,
        blocks=blocks,
        dim=config.dim,
        latent_dim=config.latent_dim,
        latent_seq_ratio=blocks / float(config.seq_len),
        latent_storage_ratio_vs_kv=(blocks * config.latent_dim) / float(config.seq_len * 2 * config.dim),
        recon_k_relative_mse=relative_mse(k_recon, dataset.k_test),
        recon_v_relative_mse=relative_mse(v_recon, dataset.v_test),
        recon_attention_relative_mse=relative_mse(y_recon, dataset.y_test),
        recon_attention_cosine=attn_cosine,
        recon_token_top1_recall=topk_recall(full_token_scores, recon_token_scores, 1),
        recon_token_top5_recall=topk_recall(full_token_scores, recon_token_scores, 5),
        mean_block_top1_recall=block_recall(full_block_scores, mean_block_scores, 1),
        mean_block_top3_recall=block_recall(full_block_scores, mean_block_scores, 3),
        recon_block_top1_recall=block_recall(full_block_scores, recon_block_scores, 1),
        recon_block_top3_recall=block_recall(full_block_scores, recon_block_scores, 3),
        latent_block_top1_recall=block_recall(full_block_scores, latent_block_scores, 1),
        latent_block_top3_recall=block_recall(full_block_scores, latent_block_scores, 3),
        joint_search_weight=config.joint_search_weight,
        block_search_weight=config.block_search_weight,
        attention_loss_weight=config.attention_loss_weight,
        topk_score_weight=config.topk_score_weight,
        rare_recon_weight=config.rare_recon_weight,
        rare_token_fraction=config.rare_token_fraction,
        ae_final_loss=train_stats.total_loss,
        ae_final_recon_loss=train_stats.recon_loss,
        ae_final_attention_loss=train_stats.attention_loss,
        ae_final_block_loss=train_stats.block_loss,
        ae_final_topk_loss=train_stats.topk_loss,
        ae_final_rare_loss=train_stats.rare_loss,
        search_final_loss=search_final_loss,
        train_seconds=train_seconds,
    )


def main() -> None:
    config = parse_args()
    if config.seq_len % config.block_size != 0:
        raise ValueError("--seq_len must be divisible by --block_size")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    torch.manual_seed(config.seed)
    rows: list[ResultRow] = []

    start_wall = time.perf_counter()
    for mode in config.modes:
        dataset = make_dataset(mode, config, device)
        synchronize(device)
        train_start = time.perf_counter()
        model, joint_searcher, train_stats = train_autoencoder(dataset, config, device)
        searcher, search_loss = train_searcher(model, dataset, config, device, joint_searcher)
        synchronize(device)
        train_seconds = time.perf_counter() - train_start
        rows.append(evaluate(mode, model, searcher, dataset, config, train_stats, search_loss, train_seconds))
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "seq_autoencoder_search.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "device": str(device),
        "wall_seconds": wall_seconds,
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "mode,storage_ratio,k_rel_mse,v_rel_mse,attn_rel_mse,"
        "token_top1,token_top5,mean_block_top1,recon_block_top1,latent_block_top1,latent_block_top3"
    )
    for row in rows:
        print(
            f"{row.mode},{row.latent_storage_ratio_vs_kv:.4f},"
            f"{row.recon_k_relative_mse:.6f},{row.recon_v_relative_mse:.6f},"
            f"{row.recon_attention_relative_mse:.6f},"
            f"{row.recon_token_top1_recall:.4f},{row.recon_token_top5_recall:.4f},"
            f"{row.mean_block_top1_recall:.4f},{row.recon_block_top1_recall:.4f},"
            f"{row.latent_block_top1_recall:.4f},{row.latent_block_top3_recall:.4f}"
        )


if __name__ == "__main__":
    main()
