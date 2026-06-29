from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    output_dir: str
    remote_tokens: int
    train_queries: int
    test_queries: int
    dim: int
    value_dim: int
    latent_clusters: int
    prototypes: int
    seed: int
    kmeans_iters: int
    ridge: float
    joint_steps: int
    joint_lr: float
    key_noise: float
    value_noise: float
    query_noise: float
    key_scale: float
    device: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Smoke test for influence-bounded synthetic KV compression."
    )
    parser.add_argument("--output-dir", default="ymluo/projects/influence_bounded_synthetic_kv/outputs/smoke")
    parser.add_argument("--remote-tokens", type=int, default=2048)
    parser.add_argument("--train-queries", type=int, default=128)
    parser.add_argument("--test-queries", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--latent-clusters", type=int, default=32)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--joint-steps", type=int, default=200)
    parser.add_argument("--joint-lr", type=float, default=3e-2)
    parser.add_argument("--key-noise", type=float, default=0.18)
    parser.add_argument("--value-noise", type=float, default=0.20)
    parser.add_argument("--query-noise", type=float, default=0.25)
    parser.add_argument("--key-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return Config(**vars(args))


def attention_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.T / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1, eps=1e-8).mean().item())


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target**2).clamp_min(1e-12)
    return float((mse / denom).item())


def metrics_row(method: str, split: str, pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    err = pred - target
    return {
        "method": method,
        "split": split,
        "mse": float(torch.mean(err**2).item()),
        "relative_mse": relative_mse(pred, target),
        "mean_l2": float(torch.linalg.vector_norm(err, dim=-1).mean().item()),
        "mean_cosine": cosine_mean(pred, target),
    }


def make_problem(config: Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    centers_k = torch.randn(
        config.latent_clusters, config.dim, generator=generator, device=config.device
    )
    centers_k = F.normalize(centers_k, dim=-1) * config.key_scale
    centers_v = torch.randn(
        config.latent_clusters, config.value_dim, generator=generator, device=config.device
    )

    token_cluster = torch.randint(
        0, config.latent_clusters, (config.remote_tokens,), generator=generator, device=config.device
    )
    k_remote = centers_k[token_cluster] + config.key_noise * torch.randn(
        config.remote_tokens, config.dim, generator=generator, device=config.device
    )
    v_remote = centers_v[token_cluster] + config.value_noise * torch.randn(
        config.remote_tokens, config.value_dim, generator=generator, device=config.device
    )

    def sample_queries(count: int) -> torch.Tensor:
        cluster = torch.randint(
            0, config.latent_clusters, (count,), generator=generator, device=config.device
        )
        return centers_k[cluster] + config.query_noise * torch.randn(
            count, config.dim, generator=generator, device=config.device
        )

    q_train = sample_queries(config.train_queries)
    q_test = sample_queries(config.test_queries)
    y_train = attention_output(q_train, k_remote, v_remote)
    y_test = attention_output(q_test, k_remote, v_remote)

    return {
        "k_remote": k_remote,
        "v_remote": v_remote,
        "q_train": q_train,
        "q_test": q_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def kmeans(data: torch.Tensor, clusters: int, iters: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=data.device).manual_seed(seed)
    init_ids = torch.randperm(data.shape[0], generator=generator, device=data.device)[:clusters]
    centers = data[init_ids].clone()
    for _ in range(iters):
        distances = torch.cdist(data.float(), centers.float())
        labels = torch.argmin(distances, dim=1)
        next_centers = centers.clone()
        for idx in range(clusters):
            mask = labels == idx
            if torch.any(mask):
                next_centers[idx] = data[mask].mean(dim=0)
        if torch.allclose(next_centers, centers, rtol=1e-5, atol=1e-6):
            centers = next_centers
            break
        centers = next_centers
    return centers


def solve_ridge_values(
    q: torch.Tensor,
    k_syn: torch.Tensor,
    y_target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    weights = torch.softmax(q @ k_syn.T / math.sqrt(q.shape[-1]), dim=-1)
    lhs = weights.T @ weights
    eye = torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
    rhs = weights.T @ y_target
    return torch.linalg.solve(lhs + ridge * eye, rhs)


def select_top_mass_tokens(
    q: torch.Tensor,
    k_remote: torch.Tensor,
    count: int,
) -> torch.Tensor:
    weights = torch.softmax(q @ k_remote.T / math.sqrt(q.shape[-1]), dim=-1)
    mass = weights.sum(dim=0)
    return torch.topk(mass, k=count).indices


def clip_rows_(tensor: torch.Tensor, max_norm: float) -> None:
    with torch.no_grad():
        norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(max_norm / norms, max=1.0)
        tensor.mul_(scale)


def fit_joint_kv(
    q_train: torch.Tensor,
    y_train: torch.Tensor,
    k_init: torch.Tensor,
    v_init: torch.Tensor,
    config: Config,
    k_bound: float,
    v_bound: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    k_syn = torch.nn.Parameter(k_init.clone())
    v_syn = torch.nn.Parameter(v_init.clone())
    optimizer = torch.optim.Adam([k_syn, v_syn], lr=config.joint_lr)
    history: list[dict[str, Any]] = []

    for step in range(config.joint_steps + 1):
        pred = attention_output(q_train, k_syn, v_syn)
        mse = torch.mean((pred - y_train) ** 2)
        k_over = torch.relu(torch.linalg.vector_norm(k_syn, dim=-1) - k_bound).pow(2).mean()
        v_over = torch.relu(torch.linalg.vector_norm(v_syn, dim=-1) - v_bound).pow(2).mean()
        loss = mse + 0.01 * (k_over + v_over)

        if step in {0, config.joint_steps} or step % 50 == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "mse": float(mse.item()),
                    "k_max_norm": float(torch.linalg.vector_norm(k_syn, dim=-1).max().item()),
                    "v_max_norm": float(torch.linalg.vector_norm(v_syn, dim=-1).max().item()),
                }
            )

        if step == config.joint_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        clip_rows_(k_syn, k_bound)
        clip_rows_(v_syn, v_bound)

    return k_syn.detach(), v_syn.detach(), history


def evaluate_method(
    method: str,
    q_train: torch.Tensor,
    q_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        metrics_row(method, "train", attention_output(q_train, k, v), y_train),
        metrics_row(method, "test", attention_output(q_test, k, v), y_test),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    if config.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {config.device!r}, but CUDA is not available.")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = make_problem(config)
    k_remote = problem["k_remote"]
    v_remote = problem["v_remote"]
    q_train = problem["q_train"]
    q_test = problem["q_test"]
    y_train = problem["y_train"]
    y_test = problem["y_test"]

    rows: list[dict[str, Any]] = []

    generator = torch.Generator(device=config.device).manual_seed(config.seed + 101)
    random_ids = torch.randperm(config.remote_tokens, generator=generator, device=config.device)[
        : config.prototypes
    ]
    rows.extend(
        evaluate_method(
            "random_real_kv",
            q_train,
            q_test,
            y_train,
            y_test,
            k_remote[random_ids],
            v_remote[random_ids],
        )
    )

    top_ids = select_top_mass_tokens(q_train, k_remote, config.prototypes)
    rows.extend(
        evaluate_method(
            "top_mass_real_kv",
            q_train,
            q_test,
            y_train,
            y_test,
            k_remote[top_ids],
            v_remote[top_ids],
        )
    )

    k_kmeans = kmeans(k_remote, config.prototypes, config.kmeans_iters, config.seed + 202)
    v_ridge = solve_ridge_values(q_train, k_kmeans, y_train, config.ridge)
    rows.extend(
        evaluate_method("kmeans_k_ridge_v", q_train, q_test, y_train, y_test, k_kmeans, v_ridge)
    )

    k_bound = float(torch.quantile(torch.linalg.vector_norm(k_remote, dim=-1), 0.95).item())
    v_bound = float(torch.quantile(torch.linalg.vector_norm(v_remote, dim=-1), 0.95).item())
    k_joint, v_joint, joint_history = fit_joint_kv(
        q_train, y_train, k_kmeans, v_ridge, config, k_bound=k_bound, v_bound=v_bound
    )
    rows.extend(evaluate_method("joint_kv", q_train, q_test, y_train, y_test, k_joint, v_joint))

    write_csv(output_dir / "metrics.csv", rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    best = min(test_rows, key=lambda row: row["relative_mse"])
    summary = {
        "config": asdict(config),
        "k_bound_p95": k_bound,
        "v_bound_p95": v_bound,
        "best_test_method": best,
        "joint_history": joint_history,
        "outputs": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Influence-bounded synthetic KV smoke test complete.")
    print(f"Output dir: {output_dir}")
    print("Test relative MSE by method:")
    for row in test_rows:
        print(
            f"  {row['method']}: relative_mse={row['relative_mse']:.6f}, "
            f"mean_cosine={row['mean_cosine']:.6f}"
        )
    print(f"Best test method: {best['method']}")


if __name__ == "__main__":
    main()
