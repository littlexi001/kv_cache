from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model_loader import load_model_and_tokenizer


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, x: torch.Tensor, labels: torch.Tensor, batch_size: int) -> Dict[str, float]:
    losses: List[torch.Tensor] = []
    correct = 0
    margins: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(model.device, dtype=model.dtype)
            yb = labels[start : start + batch_size].to(model.device)
            logits = model.lm_head(xb).float()
            losses.append(F.cross_entropy(logits, yb, reduction="none").cpu())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            correct_logits = logits.gather(1, yb[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, yb[:, None], -float("inf"))
            margins.append((correct_logits - masked.max(dim=-1).values).cpu())
    loss = torch.cat(losses).mean()
    margin = torch.cat(margins).mean()
    return {
        "loss": float(loss.item()),
        "ppl": float(torch.exp(loss).item()),
        "accuracy": correct / x.shape[0],
        "mean_margin": float(margin.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LM-head sensitivity to uncentered SVD directions.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--artifact-dir", default="fdong_seq_compress/artifacts/output_svd_qwen3_0p6b")
    parser.add_argument("--output-dir", default="fdong_seq_compress/outputs/output_svd_sensitivity")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--direction-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = torch.load(artifact_dir / "sampled_final_hidden.pt", map_location="cpu", weights_only=False)
    svd = torch.load(artifact_dir / "uncentered_svd_basis.pt", map_location="cpu", weights_only=False)
    x_all = samples["hidden_states"].float()
    labels_all = samples["target_ids"].long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    count = min(args.eval_samples, x_all.shape[0])
    indices = torch.randperm(x_all.shape[0], generator=generator)[:count]
    x = x_all[indices].contiguous()
    labels = labels_all[indices].contiguous()
    basis = svd["basis"].float()
    singular_values = svd["singular_values"].float()
    _, model, _ = load_model_and_tokenizer(
        args.model_path, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )

    baseline = evaluate(model, x, labels, args.batch_size)
    direction_indices = sorted(set(range(0, basis.shape[1], args.direction_stride)) | {basis.shape[1] - 1})
    rows: List[Dict] = []
    for direction_idx in direction_indices:
        v = basis[:, direction_idx]
        projection = x @ v
        ablated = x - projection[:, None] * v[None, :]
        metrics = evaluate(model, ablated, labels, args.batch_size)
        row = {
            "mode": "remove_single_direction",
            "direction_index": direction_idx,
            "singular_value": float(singular_values[direction_idx].item()),
            "mean_abs_projection": float(projection.abs().mean().item()),
            **metrics,
            "delta_loss": metrics["loss"] - baseline["loss"],
            "ppl_ratio": metrics["ppl"] / baseline["ppl"],
            "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
            "delta_margin": metrics["mean_margin"] - baseline["mean_margin"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    for fraction in (0.01, 0.05, 0.10, 0.20):
        remove_count = max(1, math.ceil(fraction * basis.shape[1]))
        for side in ("top", "tail"):
            selected = basis[:, :remove_count] if side == "top" else basis[:, -remove_count:]
            ablated = x - (x @ selected) @ selected.T
            metrics = evaluate(model, ablated, labels, args.batch_size)
            rows.append(
                {
                    "mode": f"remove_{side}_band",
                    "direction_index": -1,
                    "singular_value": float("nan"),
                    "mean_abs_projection": float((x @ selected).abs().mean().item()),
                    "band_fraction": fraction,
                    **metrics,
                    "delta_loss": metrics["loss"] - baseline["loss"],
                    "ppl_ratio": metrics["ppl"] / baseline["ppl"],
                    "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
                    "delta_margin": metrics["mean_margin"] - baseline["mean_margin"],
                }
            )

    write_csv(output_dir / "svd_direction_sensitivity.csv", rows)
    singles = [row for row in rows if row["mode"] == "remove_single_direction"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot([row["direction_index"] for row in singles], [row["delta_loss"] for row in singles], marker="o", markersize=3)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Delta loss after removing direction")
    axes[1].plot([row["direction_index"] for row in singles], [row["delta_accuracy"] for row in singles], marker="o", markersize=3)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Delta accuracy")
    axes[1].set_xlabel("Uncentered singular direction index")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "svd_direction_sensitivity.png", dpi=180)
    plt.close(fig)
    (output_dir / "summary.json").write_text(
        json.dumps({"baseline": baseline, "eval_samples": count, "direction_stride": args.direction_stride}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote sensitivity results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
