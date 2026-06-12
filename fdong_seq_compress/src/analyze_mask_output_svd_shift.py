from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers.models.qwen3 import modeling_qwen3

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_loader import load_model_and_tokenizer  # noqa: E402
from run_answer_access_causal import (  # noqa: E402
    build_category_indices,
    locate_source_answer,
    overlapping_token_indices,
)
from run_score_top_task_ablation import OracleTopAttention  # noqa: E402
from run_top_token_category_ablation import build_prompt, load_sample  # noqa: E402


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_condition(
    model,
    input_ids: torch.Tensor,
    target_indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids.to(device), use_cache=False).last_hidden_state[0]
        logits = model.lm_head(hidden).float()
    positions = torch.tensor([idx - 1 for idx in target_indices], device=device)
    targets = input_ids[0, torch.tensor(target_indices)].to(device)
    x = hidden[positions].detach().cpu().float()
    selected_logits = logits[positions]
    losses = F.cross_entropy(selected_logits, targets, reduction="none")
    correct_logits = selected_logits.gather(1, targets[:, None]).squeeze(1)
    competitors = selected_logits.clone()
    competitors.scatter_(1, targets[:, None], -float("inf"))
    competitor_values, competitor_ids = competitors.max(dim=-1)
    metrics = {
        "loss": float(losses.mean().item()),
        "ppl": float(torch.exp(losses.mean()).item()),
        "accuracy": float((selected_logits.argmax(dim=-1) == targets).float().mean().item()),
        "correct_logit_mean": float(correct_logits.mean().item()),
        "competitor_logit_mean": float(competitor_values.mean().item()),
        "margin_mean": float((correct_logits - competitor_values).mean().item()),
    }
    return x, metrics, competitor_ids.detach().cpu()


def attribute_loss_shift_to_svd_directions(
    model,
    full_x: torch.Tensor,
    masked_x: torch.Tensor,
    targets: torch.Tensor,
    basis: torch.Tensor,
    device: torch.device,
    steps: int,
) -> Tuple[torch.Tensor, Dict]:
    """Decompose the full-to-masked loss shift with path integrated gradients."""
    if steps < 2:
        raise ValueError("--ig-steps must be at least 2")

    delta_x = masked_x - full_x
    average_gradient = torch.zeros_like(delta_x)
    targets_device = targets.to(device)
    model_dtype = next(model.parameters()).dtype

    for step, alpha in enumerate(torch.linspace(0.0, 1.0, steps)):
        weight = 0.5 if step in (0, steps - 1) else 1.0
        interpolated = (full_x + float(alpha.item()) * delta_x).to(device=device, dtype=model_dtype)
        interpolated.requires_grad_(True)
        with torch.enable_grad():
            logits = model.lm_head(interpolated).float()
            losses = F.cross_entropy(logits, targets_device, reduction="none")
            gradient = torch.autograd.grad(losses.sum(), interpolated)[0]
        average_gradient.add_(gradient.detach().cpu().float(), alpha=weight / (steps - 1))

    delta_coefficients = delta_x @ basis
    gradient_coefficients = average_gradient @ basis
    direction_attribution = delta_coefficients * gradient_coefficients

    with torch.no_grad():
        full_logits = model.lm_head(full_x.to(device=device, dtype=model_dtype)).float()
        masked_logits = model.lm_head(masked_x.to(device=device, dtype=model_dtype)).float()
        full_losses = F.cross_entropy(full_logits, targets_device, reduction="none").cpu()
        masked_losses = F.cross_entropy(masked_logits, targets_device, reduction="none").cpu()
    exact_delta = masked_losses - full_losses
    attributed_delta = direction_attribution.sum(dim=-1)
    diagnostics = {
        "exact_delta_loss_mean": float(exact_delta.mean().item()),
        "attributed_delta_loss_mean": float(attributed_delta.mean().item()),
        "completeness_error_mean": float((attributed_delta - exact_delta).mean().item()),
        "completeness_abs_error_mean": float((attributed_delta - exact_delta).abs().mean().item()),
    }
    return direction_attribution, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Project final hidden-state mask shifts into a cached SVD basis.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--artifact-dir", default="fdong_seq_compress/artifacts/output_svd_qwen3_0p6b")
    parser.add_argument(
        "--data-path",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl",
    )
    parser.add_argument("--sample-id", default="niah_len2000_depth25")
    parser.add_argument("--output-dir", default="fdong_seq_compress/outputs/mask_output_svd_shift")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--score-ratio", type=float, default=0.02)
    parser.add_argument("--position-ratio", type=float, default=0.01)
    parser.add_argument(
        "--excluded-categories",
        default="answer,front,end,other",
        help="Comma-separated category ablations, or 'none' to run only full and score_top_all.",
    )
    parser.add_argument("--ig-steps", type=int, default=33)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svd = torch.load(Path(args.artifact_dir) / "uncentered_svd_basis.pt", map_location="cpu", weights_only=False)
    basis = svd["basis"].float()
    singular_values = svd["singular_values"].float()
    sample = load_sample(Path(args.data_path), args.sample_id)
    prompt = build_prompt(sample)
    answer = str(sample["expected_answer"])
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )
    full_text = prompt + " " + answer
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded.input_ids
    offsets = [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]
    target_indices = overlapping_token_indices(offsets, len(prompt) + 1, len(full_text))
    prompt_encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    prompt_offsets = [(int(a), int(b)) for a, b in prompt_encoded.offset_mapping[0].tolist()]
    answer_indices = locate_source_answer(prompt, answer, prompt_offsets)
    category_indices = build_category_indices(int(prompt_encoded.input_ids.shape[1]), answer_indices, args.position_ratio)
    controller = OracleTopAttention(args.score_ratio, category_indices)
    original_forward = modeling_qwen3.eager_attention_forward
    modeling_qwen3.eager_attention_forward = controller.forward
    excluded_categories = [] if args.excluded_categories.lower() == "none" else [
        category.strip() for category in args.excluded_categories.split(",") if category.strip()
    ]
    valid_categories = {"answer", "front", "end", "other"}
    unknown_categories = set(excluded_categories) - valid_categories
    if unknown_categories:
        raise ValueError(f"Unknown excluded categories: {sorted(unknown_categories)}")
    conditions: List[Tuple[str, bool, str | None]] = [
        ("full", False, None),
        ("score_top_all", True, None),
        *[(f"score_top_without_{category}", True, category) for category in excluded_categories],
    ]
    condition_x: Dict[str, torch.Tensor] = {}
    metric_rows: List[Dict] = []
    try:
        for name, enabled, excluded in conditions:
            controller.reset(enabled, excluded)
            controller.set_recording(False)
            x, metrics, competitor_ids = evaluate_condition(model, input_ids, target_indices, device)
            condition_x[name] = x
            metric_rows.append({"condition": name, **metrics})
            print(json.dumps(metric_rows[-1]), flush=True)
    finally:
        modeling_qwen3.eager_attention_forward = original_forward

    full_x = condition_x["full"]
    full_coeff = full_x @ basis
    full_metrics = next(row for row in metric_rows if row["condition"] == "full")
    for row in metric_rows:
        x = condition_x[row["condition"]]
        delta_x = x - full_x
        row.update(
            {
                "delta_loss": row["loss"] - full_metrics["loss"],
                "ppl_ratio": row["ppl"] / full_metrics["ppl"],
                "delta_accuracy": row["accuracy"] - full_metrics["accuracy"],
                "delta_correct_logit": row["correct_logit_mean"] - full_metrics["correct_logit_mean"],
                "delta_competitor_logit": row["competitor_logit_mean"] - full_metrics["competitor_logit_mean"],
                "delta_margin": row["margin_mean"] - full_metrics["margin_mean"],
                "x_cosine_to_full_mean": float(F.cosine_similarity(x, full_x, dim=-1).mean().item()),
                "x_relative_l2_to_full_mean": float(
                    (
                        torch.linalg.vector_norm(delta_x, dim=-1)
                        / torch.linalg.vector_norm(full_x, dim=-1).clamp_min(1e-12)
                    ).mean().item()
                ),
            }
        )
    direction_rows: List[Dict] = []
    for name, x in condition_x.items():
        coeff = x @ basis
        delta_x = x - full_x
        delta_coeff = coeff - full_coeff
        relative_l2 = torch.linalg.vector_norm(delta_x, dim=-1) / torch.linalg.vector_norm(full_x, dim=-1).clamp_min(1e-12)
        cosine = F.cosine_similarity(x, full_x, dim=-1)
        for k in range(basis.shape[1]):
            direction_rows.append(
                {
                    "condition": name,
                    "direction_index": k,
                    "singular_value": float(singular_values[k].item()),
                    "full_mean_coefficient": float(full_coeff[:, k].mean().item()),
                    "condition_mean_coefficient": float(coeff[:, k].mean().item()),
                    "mean_delta_coefficient": float(delta_coeff[:, k].mean().item()),
                    "mean_abs_delta_coefficient": float(delta_coeff[:, k].abs().mean().item()),
                    "x_cosine_to_full_mean": float(cosine.mean().item()),
                    "x_relative_l2_to_full_mean": float(relative_l2.mean().item()),
                }
            )
    write_csv(output_dir / "condition_metrics.csv", metric_rows)
    write_csv(output_dir / "svd_projection_shift_by_condition.csv", direction_rows)

    targets = input_ids[0, torch.tensor(target_indices)].cpu()
    attribution_rows: List[Dict] = []
    completeness_rows: List[Dict] = []
    for name, _, _ in conditions[1:]:
        attribution, diagnostics = attribute_loss_shift_to_svd_directions(
            model=model,
            full_x=full_x,
            masked_x=condition_x[name],
            targets=targets,
            basis=basis,
            device=device,
            steps=args.ig_steps,
        )
        delta_coeff = condition_x[name] @ basis - full_coeff
        completeness_rows.append({"condition": name, "ig_steps": args.ig_steps, **diagnostics})
        for k in range(basis.shape[1]):
            attribution_rows.append(
                {
                    "condition": name,
                    "direction_index": k,
                    "singular_value": float(singular_values[k].item()),
                    "mean_delta_coefficient": float(delta_coeff[:, k].mean().item()),
                    "mean_abs_delta_coefficient": float(delta_coeff[:, k].abs().mean().item()),
                    "mean_loss_attribution": float(attribution[:, k].mean().item()),
                    "mean_abs_loss_attribution": float(attribution[:, k].abs().mean().item()),
                }
            )
        print(json.dumps(completeness_rows[-1]), flush=True)
    write_csv(output_dir / "svd_loss_attribution_by_condition.csv", attribution_rows)
    write_csv(output_dir / "ig_completeness.csv", completeness_rows)

    fig, axis = plt.subplots(figsize=(12, 6))
    for name, _, _ in conditions[1:]:
        rows = [row for row in direction_rows if row["condition"] == name]
        axis.plot(
            [row["direction_index"] for row in rows],
            [row["mean_abs_delta_coefficient"] for row in rows],
            label=name,
            linewidth=1.2,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Uncentered singular direction index")
    axis.set_ylabel("Mean absolute coefficient shift")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mask_svd_projection_shift.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name, _, _ in conditions[1:]:
        rows = [row for row in attribution_rows if row["condition"] == name]
        indices = [row["direction_index"] for row in rows]
        axes[0].plot(indices, [row["mean_delta_coefficient"] for row in rows], label=name, linewidth=1.0)
        axes[1].plot(indices, [row["mean_loss_attribution"] for row in rows], label=name, linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Mean signed coefficient shift")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[1].set_xlabel("Uncentered singular direction index")
    axes[1].set_ylabel("Mean signed loss attribution")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mask_svd_signed_loss_attribution.png", dpi=180)
    plt.close(fig)
    print(f"Wrote mask-shift analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
