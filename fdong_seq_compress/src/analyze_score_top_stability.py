from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers.models.qwen3 import modeling_qwen3

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_mask_output_svd_shift import evaluate_condition  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from run_answer_access_causal import (  # noqa: E402
    build_category_indices,
    locate_source_answer,
    overlapping_token_indices,
)
from run_score_top_task_ablation import OracleTopAttention  # noqa: E402
from run_top_token_category_ablation import build_prompt  # noqa: E402


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def load_samples(path: Path, requested_ids: set[str]) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if str(sample.get("sample_id")) in requested_ids:
                samples.append(sample)
    found = {str(sample["sample_id"]) for sample in samples}
    missing = requested_ids - found
    if missing:
        raise ValueError(f"Missing sample IDs: {sorted(missing)}")
    return samples


def parse_sample_ids(value: str) -> List[str]:
    if value.strip().lower() != "default":
        return [item.strip() for item in value.split(",") if item.strip()]
    return [
        f"niah_len{length}_depth{depth}"
        for length in (1000, 2000, 4000)
        for depth in (0, 10, 25, 50, 75, 90, 100)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether oracle score-top loss changes persist across NIAH samples.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--data-path",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl",
    )
    parser.add_argument("--sample-ids", default="default")
    parser.add_argument("--score-ratio", type=float, default=0.02)
    parser.add_argument("--position-ratio", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="fdong_seq_compress/outputs/score_top_stability")
    args = parser.parse_args()
    if not 0 < args.score_ratio <= 1:
        raise ValueError("--score-ratio must be in (0, 1].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = parse_sample_ids(args.sample_ids)
    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in load_samples(Path(args.data_path), set(sample_ids))
    }
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )

    original_forward = modeling_qwen3.eager_attention_forward
    rows: List[Dict] = []
    try:
        for sample_id in sample_ids:
            sample = samples_by_id[sample_id]
            prompt = build_prompt(sample)
            answer = str(sample["expected_answer"])
            full_text = prompt + " " + answer
            encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            input_ids = encoded.input_ids
            offsets = [(int(a), int(b)) for a, b in encoded.offset_mapping[0].tolist()]
            target_indices = overlapping_token_indices(offsets, len(prompt) + 1, len(full_text))
            prompt_encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            prompt_offsets = [(int(a), int(b)) for a, b in prompt_encoded.offset_mapping[0].tolist()]
            answer_indices = locate_source_answer(prompt, answer, prompt_offsets)
            categories = build_category_indices(int(prompt_encoded.input_ids.shape[1]), answer_indices, args.position_ratio)
            controller = OracleTopAttention(args.score_ratio, categories)
            modeling_qwen3.eager_attention_forward = controller.forward

            condition_metrics: Dict[str, Dict] = {}
            condition_x: Dict[str, torch.Tensor] = {}
            for condition, enabled in (("full", False), ("score_top_all", True)):
                controller.reset(enabled, None)
                controller.set_recording(False)
                x, metrics, _ = evaluate_condition(model, input_ids, target_indices, device)
                condition_metrics[condition] = metrics
                condition_x[condition] = x

            full = condition_metrics["full"]
            top = condition_metrics["score_top_all"]
            delta_x = condition_x["score_top_all"] - condition_x["full"]
            row = {
                "sample_id": sample_id,
                "nominal_length": int(sample_id.split("_len", 1)[1].split("_", 1)[0]),
                "needle_depth": int(sample_id.rsplit("depth", 1)[1]),
                "prompt_token_count": int(prompt_encoded.input_ids.shape[1]),
                "answer_token_count": len(target_indices),
                "full_loss": full["loss"],
                "top_loss": top["loss"],
                "delta_loss": top["loss"] - full["loss"],
                "full_ppl": full["ppl"],
                "top_ppl": top["ppl"],
                "ppl_ratio": top["ppl"] / full["ppl"],
                "full_accuracy": full["accuracy"],
                "top_accuracy": top["accuracy"],
                "delta_accuracy": top["accuracy"] - full["accuracy"],
                "full_margin": full["margin_mean"],
                "top_margin": top["margin_mean"],
                "delta_margin": top["margin_mean"] - full["margin_mean"],
                "x_cosine_to_full": float(torch.nn.functional.cosine_similarity(
                    condition_x["score_top_all"], condition_x["full"], dim=-1
                ).mean().item()),
                "x_relative_l2": float((
                    torch.linalg.vector_norm(delta_x, dim=-1)
                    / torch.linalg.vector_norm(condition_x["full"], dim=-1).clamp_min(1e-12)
                ).mean().item()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        modeling_qwen3.eager_attention_forward = original_forward

    write_csv(output_dir / "per_sample_metrics.csv", rows)
    delta_losses = [float(row["delta_loss"]) for row in rows]
    ppl_ratios = [float(row["ppl_ratio"]) for row in rows]
    summary = {
        "sample_count": len(rows),
        "score_ratio": args.score_ratio,
        "loss_improved_count": sum(value < 0 for value in delta_losses),
        "loss_improved_fraction": sum(value < 0 for value in delta_losses) / len(rows),
        "mean_delta_loss": statistics.fmean(delta_losses),
        "median_delta_loss": statistics.median(delta_losses),
        "mean_ppl_ratio": statistics.fmean(ppl_ratios),
        "median_ppl_ratio": statistics.median(ppl_ratios),
        "min_ppl_ratio": min(ppl_ratios),
        "max_ppl_ratio": max(ppl_ratios),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [row["needle_depth"] for row in rows]
    axes[0].scatter([row["prompt_token_count"] for row in rows], delta_losses, c=colors, cmap="viridis")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Prompt token count")
    axes[0].set_ylabel("Top 2% loss - full loss")
    axes[0].grid(alpha=0.25)
    axes[1].hist(ppl_ratios, bins=min(12, len(rows)), edgecolor="black", alpha=0.8)
    axes[1].axvline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("PPL ratio: top 2% / full")
    axes[1].set_ylabel("Sample count")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "score_top_stability.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary), flush=True)
    print(f"Wrote stability analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
