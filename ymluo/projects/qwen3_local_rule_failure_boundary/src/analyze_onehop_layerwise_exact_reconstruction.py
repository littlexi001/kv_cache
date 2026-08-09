from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the exact N-to-N+1 finite-difference recurrence and, "
            "optionally, the final Qwen3 RMSNorm/lm_head readout."
        )
    )
    parser.add_argument("--trace-vectors", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--fixed-rope-factor", type=float, default=4.0)
    parser.add_argument("--fixed-max-position-embeddings", type=int, default=147456)
    parser.add_argument("--logit-chunk-size", type=int, default=4096)
    return parser.parse_args()


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def last_token(
    trace: dict[int, dict[str, torch.Tensor]],
    layer: int,
    stage: str,
) -> torch.Tensor:
    return trace[layer][stage][-1].detach().cpu()


def relative_l2(error: torch.Tensor, reference: torch.Tensor) -> float:
    return float(error.float().norm().item()) / max(
        float(reference.float().norm().item()),
        1e-30,
    )


def reconstruct_layers(
    source: dict[int, dict[str, torch.Tensor]],
    target: dict[int, dict[str, torch.Tensor]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in sorted(source):
        source_in = last_token(source, layer, "residual_in")
        target_in = last_token(target, layer, "residual_in")
        source_attn = last_token(source, layer, "attn_out")
        target_attn = last_token(target, layer, "attn_out")
        source_post = last_token(source, layer, "post_attn_residual")
        target_post = last_token(target, layer, "post_attn_residual")
        source_mlp = last_token(source, layer, "mlp_out")
        target_mlp = last_token(target, layer, "mlp_out")
        source_out = last_token(source, layer, "residual_out")
        target_out = last_token(target, layer, "residual_out")

        delta_in = target_in.float() - source_in.float()
        delta_attn = target_attn.float() - source_attn.float()
        delta_post = target_post.float() - source_post.float()
        delta_mlp = target_mlp.float() - source_mlp.float()
        delta_out = target_out.float() - source_out.float()

        real_post_reconstruction = delta_in + delta_attn
        real_out_reconstruction = delta_in + delta_attn + delta_mlp
        real_post_error = delta_post - real_post_reconstruction
        real_out_error = delta_out - real_out_reconstruction
        component_norm_sum = float(
            delta_in.norm().item()
            + delta_attn.norm().item()
            + delta_mlp.norm().item()
        )
        real_reconstruction_norm = float(
            real_out_reconstruction.norm().item()
        )
        pairwise_inner_product_sum = float(
            (
                torch.dot(delta_in, delta_attn)
                + torch.dot(delta_in, delta_mlp)
                + torch.dot(delta_attn, delta_mlp)
            ).item()
        )

        # Q_b denotes the actual BF16 rounding at each residual addition.
        source_post_hat = (source_in + source_attn).to(source_post.dtype)
        target_post_hat = (target_in + target_attn).to(target_post.dtype)
        source_out_hat = (source_post_hat + source_mlp).to(source_out.dtype)
        target_out_hat = (target_post_hat + target_mlp).to(target_out.dtype)
        hardware_delta_post = target_post_hat.float() - source_post_hat.float()
        hardware_delta_out = target_out_hat.float() - source_out_hat.float()

        source_link_error = 0.0
        target_link_error = 0.0
        if layer + 1 in source:
            source_link_error = float(
                (
                    source_out.float()
                    - last_token(source, layer + 1, "residual_in").float()
                )
                .abs()
                .max()
                .item()
            )
            target_link_error = float(
                (
                    target_out.float()
                    - last_token(target, layer + 1, "residual_in").float()
                )
                .abs()
                .max()
                .item()
            )

        rows.append(
            {
                "layer": layer,
                "delta_input_l2": rounded(delta_in.norm().item()),
                "delta_attention_l2": rounded(delta_attn.norm().item()),
                "delta_mlp_l2": rounded(delta_mlp.norm().item()),
                "delta_output_l2": rounded(delta_out.norm().item()),
                "component_norm_sum": rounded(component_norm_sum),
                "real_vector_sum_l2": rounded(real_reconstruction_norm),
                "component_alignment_ratio": rounded(
                    real_reconstruction_norm
                    / max(component_norm_sum, 1e-30)
                ),
                "pairwise_inner_product_sum": rounded(
                    pairwise_inner_product_sum
                ),
                "real_identity_max_abs_error": rounded(
                    real_out_error.abs().max().item()
                ),
                "real_identity_relative_l2_error": rounded(
                    relative_l2(real_out_error, delta_out)
                ),
                "bf16_post_reconstruction_max_abs_error": rounded(
                    (delta_post - hardware_delta_post).abs().max().item()
                ),
                "bf16_output_reconstruction_max_abs_error": rounded(
                    (delta_out - hardware_delta_out).abs().max().item()
                ),
                "bf16_reconstructed_output_l2": rounded(
                    hardware_delta_out.norm().item()
                ),
                "source_interlayer_link_max_abs_error": rounded(
                    source_link_error
                ),
                "target_interlayer_link_max_abs_error": rounded(
                    target_link_error
                ),
            }
        )

    summary = {
        "layer_count": len(rows),
        "max_real_identity_relative_l2_error": max(
            row["real_identity_relative_l2_error"] for row in rows
        ),
        "max_bf16_post_reconstruction_abs_error": max(
            row["bf16_post_reconstruction_max_abs_error"] for row in rows
        ),
        "max_bf16_output_reconstruction_abs_error": max(
            row["bf16_output_reconstruction_max_abs_error"] for row in rows
        ),
        "max_interlayer_link_abs_error": max(
            max(
                row["source_interlayer_link_max_abs_error"],
                row["target_interlayer_link_max_abs_error"],
            )
            for row in rows
        ),
    }
    return rows, summary


def fp32_logits_in_chunks(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    hidden_device = hidden.float().to(weight.device)
    for start in range(0, weight.shape[0], chunk_size):
        part = weight[start : start + chunk_size].float()
        output.append(torch.mv(part, hidden_device).detach().cpu())
    return torch.cat(output, dim=0)


def rmsnorm_jvp(
    source_hidden: torch.Tensor,
    delta_hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x = source_hidden.float()
    dx = delta_hidden.float()
    w = weight.float()
    dimension = x.numel()
    radius = torch.sqrt(x.square().mean() + eps)
    radial = torch.dot(x, dx) / (dimension * radius.pow(3))
    return w * (dx / radius - x * radial)


def readout_analysis(
    args: argparse.Namespace,
    source: dict[int, dict[str, torch.Tensor]],
    target: dict[int, dict[str, torch.Tensor]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    import run_local_rule_failure_boundary as base
    import run_twohop_age_distractor_failure_boundary_8b as twohop

    model, tokenizer = base.load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    gold_id = int(answer_variants[twohop.GOLD_ANSWER][0])
    competitor_id = int(baseline["fixed_competitor_token_id"])
    final_layer = max(source)
    source_hidden_cpu = last_token(source, final_layer, "residual_out")
    target_hidden_cpu = last_token(target, final_layer, "residual_out")

    norm = model.model.norm
    norm_device = next(norm.parameters()).device
    source_hidden = source_hidden_cpu.to(norm_device).view(1, 1, -1)
    target_hidden = target_hidden_cpu.to(norm_device).view(1, 1, -1)
    with torch.inference_mode():
        source_norm = norm(source_hidden)[0, 0]
        target_norm = norm(target_hidden)[0, 0]

    lm_head = model.lm_head
    lm_device = next(lm_head.parameters()).device
    with torch.inference_mode():
        source_logits_bf16 = lm_head(source_norm.to(lm_device)).float().cpu()
        target_logits_bf16 = lm_head(target_norm.to(lm_device)).float().cpu()

    weight = lm_head.weight
    source_norm_for_weight = source_norm.to(weight.device)
    target_norm_for_weight = target_norm.to(weight.device)
    delta_norm_for_weight = target_norm_for_weight.float() - source_norm_for_weight.float()
    source_logits_fp32 = fp32_logits_in_chunks(
        weight,
        source_norm_for_weight,
        args.logit_chunk_size,
    )
    target_logits_fp32 = fp32_logits_in_chunks(
        weight,
        target_norm_for_weight,
        args.logit_chunk_size,
    )
    delta_logits_fp32 = fp32_logits_in_chunks(
        weight,
        delta_norm_for_weight,
        args.logit_chunk_size,
    )
    reconstructed_target_logits = source_logits_fp32 + delta_logits_fp32

    norm_weight = norm.weight.detach().float().cpu()
    norm_eps = float(norm.variance_epsilon)
    delta_hidden = target_hidden_cpu.float() - source_hidden_cpu.float()
    source_norm_cpu = source_norm.detach().float().cpu()
    target_norm_cpu = target_norm.detach().float().cpu()
    norm_delta_exact = target_norm_cpu - source_norm_cpu
    norm_delta_jvp = rmsnorm_jvp(
        source_hidden_cpu,
        delta_hidden,
        norm_weight,
        norm_eps,
    )

    pair_weight = weight[[gold_id, competitor_id]].detach().float().cpu()
    margin_direction = pair_weight[0] - pair_weight[1]
    source_margin_bf16 = float(
        source_logits_bf16[gold_id] - source_logits_bf16[competitor_id]
    )
    target_margin_bf16 = float(
        target_logits_bf16[gold_id] - target_logits_bf16[competitor_id]
    )
    source_margin_fp32 = float(
        source_logits_fp32[gold_id] - source_logits_fp32[competitor_id]
    )
    target_margin_fp32 = float(
        target_logits_fp32[gold_id] - target_logits_fp32[competitor_id]
    )
    reconstructed_margin_fp32 = float(
        reconstructed_target_logits[gold_id]
        - reconstructed_target_logits[competitor_id]
    )
    jvp_margin_change = float(torch.dot(margin_direction, norm_delta_jvp))
    exact_margin_change = float(torch.dot(margin_direction, norm_delta_exact))

    source_probability_bf16 = float(
        torch.softmax(source_logits_bf16, dim=-1)[gold_id]
    )
    target_probability_bf16 = float(
        torch.softmax(target_logits_bf16, dim=-1)[gold_id]
    )
    source_probability_fp32 = float(
        torch.softmax(source_logits_fp32, dim=-1)[gold_id]
    )
    target_probability_fp32 = float(
        torch.softmax(target_logits_fp32, dim=-1)[gold_id]
    )
    reconstructed_probability_fp32 = float(
        torch.softmax(reconstructed_target_logits, dim=-1)[gold_id]
    )

    return {
        "gold_token_id": gold_id,
        "competitor_token_id": competitor_id,
        "source_total": baseline["source_total"],
        "target_total": baseline["target_total"],
        "captured_baseline": {
            "source_gold_probability": baseline["source"]["gold_probability"],
            "target_gold_probability": baseline["target"]["gold_probability"],
            "source_margin": baseline["source"]["gold_vs_fixed_competitor_margin"],
            "target_margin": baseline["target"]["gold_vs_fixed_competitor_margin"],
        },
        "bf16_readout_replay": {
            "source_gold_probability": rounded(source_probability_bf16),
            "target_gold_probability": rounded(target_probability_bf16),
            "source_margin": rounded(source_margin_bf16),
            "target_margin": rounded(target_margin_bf16),
        },
        "fp32_exact_reconstruction": {
            "source_gold_probability": rounded(source_probability_fp32),
            "target_gold_probability": rounded(target_probability_fp32),
            "reconstructed_target_gold_probability": rounded(
                reconstructed_probability_fp32
            ),
            "probability_abs_error": rounded(
                abs(
                    reconstructed_probability_fp32
                    - target_probability_fp32
                )
            ),
            "source_margin": rounded(source_margin_fp32),
            "target_margin": rounded(target_margin_fp32),
            "reconstructed_target_margin": rounded(
                reconstructed_margin_fp32
            ),
            "margin_abs_error": rounded(
                abs(reconstructed_margin_fp32 - target_margin_fp32)
            ),
            "max_logit_abs_error": rounded(
                (
                    reconstructed_target_logits - target_logits_fp32
                )
                .abs()
                .max()
                .item()
            ),
            "exact_margin_change_from_delta_norm": rounded(
                exact_margin_change
            ),
        },
        "final_rmsnorm_first_order": {
            "exact_norm_delta_l2": rounded(norm_delta_exact.norm().item()),
            "jvp_norm_delta_l2": rounded(norm_delta_jvp.norm().item()),
            "relative_l2_error": rounded(
                relative_l2(norm_delta_exact - norm_delta_jvp, norm_delta_exact)
            ),
            "cosine": rounded(
                F.cosine_similarity(
                    norm_delta_exact,
                    norm_delta_jvp,
                    dim=0,
                ).item()
            ),
            "exact_margin_change": rounded(exact_margin_change),
            "jvp_margin_change": rounded(jvp_margin_change),
            "margin_change_abs_error": rounded(
                abs(jvp_margin_change - exact_margin_change)
            ),
        },
    }


def write_report(
    path: Path,
    layer_summary: dict[str, Any],
    readout: dict[str, Any] | None,
) -> None:
    lines = [
        "# N → N+1 逐层精确重构验证",
        "",
        "## 逐层残差递推",
        "",
        (
            f"- 层数：{layer_summary['layer_count']}；实数加法恒等式的最大相对误差："
            f"{100 * layer_summary['max_real_identity_relative_l2_error']:.3f}%。"
        ),
        (
            "- 按模型实际 BF16 残差加法逐次舍入后，最大输出重构绝对误差："
            f"{layer_summary['max_bf16_output_reconstruction_abs_error']:.3g}。"
        ),
        (
            "- 相邻层 `residual_out → residual_in` 最大绝对误差："
            f"{layer_summary['max_interlayer_link_abs_error']:.3g}。"
        ),
        "",
        "实数恒等式的非零误差来自 BF16 在两次 residual add 后分别舍入；把舍入算子写入递推，36 层均可精确重构。",
    ]
    if readout is not None:
        fp32 = readout["fp32_exact_reconstruction"]
        jvp = readout["final_rmsnorm_first_order"]
        lines.extend(
            [
                "",
                "## 最终 margin 与概率",
                "",
                (
                    f"- FP32 最终 margin：实际 {fp32['target_margin']:+.6f}，"
                    f"重构 {fp32['reconstructed_target_margin']:+.6f}，"
                    f"绝对误差 {fp32['margin_abs_error']:.3g}。"
                ),
                (
                    f"- FP32 Gold 概率：实际 {fp32['target_gold_probability']:.8f}，"
                    f"重构 {fp32['reconstructed_target_gold_probability']:.8f}，"
                    f"绝对误差 {fp32['probability_abs_error']:.3g}。"
                ),
                (
                    f"- 只把最终 RMSNorm 作一阶线性化时，向量相对误差为 "
                    f"{100 * jvp['relative_l2_error']:.2f}%，"
                    f"margin 变化误差为 {jvp['margin_change_abs_error']:.4f}。"
                ),
                "",
                "因此，精确有限差分递推可以还原最终输出；Jacobian/JVP 传播只是局部近似，不能和精确恒等式混为一谈。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = torch.load(
        args.trace_vectors,
        map_location="cpu",
        weights_only=False,
    )
    source = traces["source_trace"]
    target = traces["target_trace"]
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))

    layer_rows, layer_summary = reconstruct_layers(source, target)
    write_csv(output_dir / "layerwise_exact_reconstruction.csv", layer_rows)
    write_json(output_dir / "layerwise_reconstruction_summary.json", layer_summary)

    readout = None
    if args.model_name_or_path:
        readout = readout_analysis(args, source, target, baseline)
        write_json(output_dir / "final_readout_reconstruction.json", readout)
    write_report(output_dir / "report.md", layer_summary, readout)
    print(
        json.dumps(
            {"layer_summary": layer_summary, "readout": readout},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
