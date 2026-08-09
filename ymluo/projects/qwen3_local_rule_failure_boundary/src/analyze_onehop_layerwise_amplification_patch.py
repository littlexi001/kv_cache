from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args()


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def weighted_attention_summary(
    attention: pd.DataFrame,
) -> dict[str, Any]:
    rows = {}
    for condition, frame in attention.groupby("condition"):
        weight = frame["weight"].to_numpy(dtype=float)
        weight /= weight.sum()
        rows[condition] = {
            "gold_attention": float(
                np.sum(weight * frame["gold_attention"])
            ),
            "gold_qk": float(np.sum(weight * frame["gold_qk"])),
            "irrelevant_periods_attention": float(
                np.sum(
                    weight * frame["irrelevant_periods_attention"]
                )
            ),
            "distractor_ages_attention": float(
                np.sum(
                    weight * frame["distractor_ages_attention"]
                )
            ),
            "query_attention": float(
                np.sum(weight * frame["query_attention"])
            ),
            "attention_entropy": float(
                np.sum(weight * frame["attention_entropy"])
            ),
        }
    return rows


def trace_summary(trace: pd.DataFrame) -> dict[str, Any]:
    residual_in = trace[trace.stage == "residual_in"].set_index(
        "layer"
    )
    residual_out = trace[trace.stage == "residual_out"].set_index(
        "layer"
    )
    attn = trace[trace.stage == "attn_out"].set_index("layer")
    mlp = trace[trace.stage == "mlp_out"].set_index("layer")
    q_pre = trace[trace.stage == "q_pre"].set_index("layer")

    rows = []
    for layer in residual_out.index:
        rows.append(
            {
                "layer": int(layer),
                "residual_in_delta": float(
                    residual_in.loc[layer, "delta_norm"]
                ),
                "residual_out_delta": float(
                    residual_out.loc[layer, "delta_norm"]
                ),
                "residual_relative_delta": float(
                    residual_out.loc[layer, "relative_delta"]
                ),
                "q_relative_delta": float(
                    q_pre.loc[layer, "relative_delta"]
                ),
                "q_cosine": float(q_pre.loc[layer, "cosine"]),
                "attention_delta": float(
                    attn.loc[layer, "delta_norm"]
                ),
                "mlp_delta": float(mlp.loc[layer, "delta_norm"]),
                "net_delta_growth": float(
                    residual_out.loc[layer, "delta_norm"]
                    - residual_in.loc[layer, "delta_norm"]
                ),
                "amplification": (
                    float(
                        residual_out.loc[
                            layer,
                            "layer_residual_amplification",
                        ]
                    )
                    if pd.notna(
                        residual_out.loc[
                            layer,
                            "layer_residual_amplification",
                        ]
                    )
                    else None
                ),
            }
        )
    frame = pd.DataFrame(rows)
    first_nonzero = int(
        frame.loc[
            frame.residual_out_delta > 1e-6,
            "layer",
        ].iloc[0]
    )
    return {
        "rows": rows,
        "first_nonzero_layer": first_nonzero,
        "largest_net_growth": (
            frame.sort_values(
                "net_delta_growth",
                ascending=False,
            )
            .head(8)
            .to_dict("records")
        ),
        "largest_q_drift": (
            frame.sort_values(
                "q_relative_delta",
                ascending=False,
            )
            .head(8)
            .to_dict("records")
        ),
    }


def patch_summary(patch: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for kind, frame in patch.groupby("patch_kind"):
        qk_best = frame.sort_values(
            "qk_recovery_fraction",
            ascending=False,
        ).iloc[0]
        margin_best = frame.sort_values(
            "fixed_margin_recovery_fraction",
            ascending=False,
        ).iloc[0]
        gold_layers = frame.loc[
            frame.top_token_id
            == frame.loc[
                frame.gold_probability.idxmax(),
                "top_token_id",
            ]
        ]
        output[kind] = {
            "best_qk_layer": int(qk_best.layer),
            "best_qk_recovery": float(
                qk_best.qk_recovery_fraction
            ),
            "best_margin_layer": int(margin_best.layer),
            "best_margin_recovery": float(
                margin_best.fixed_margin_recovery_fraction
            ),
            "top_qk_layers": (
                frame.sort_values(
                    "qk_recovery_fraction",
                    ascending=False,
                )
                .head(8)[
                    [
                        "layer",
                        "qk_recovery_fraction",
                        "fixed_margin_recovery_fraction",
                        "top_token_label",
                    ]
                ]
                .to_dict("records")
            ),
        }
    return output


def make_trace_plot(
    trace_summary_value: dict[str, Any],
    output: Path,
) -> None:
    frame = pd.DataFrame(trace_summary_value["rows"])
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    axes[0].plot(
        frame.layer,
        frame.residual_relative_delta,
        marker="o",
        label="residual out relative L2",
    )
    axes[0].plot(
        frame.layer,
        frame.q_relative_delta,
        marker="o",
        label="pre-RoPE Q relative L2",
    )
    axes[0].set_ylabel("Relative perturbation")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        frame.layer,
        frame.attention_delta,
        marker="o",
        label="attention-output delta",
    )
    axes[1].plot(
        frame.layer,
        frame.mlp_delta,
        marker="o",
        label="MLP-output delta",
    )
    axes[1].plot(
        frame.layer,
        frame.residual_out_delta,
        marker="o",
        label="residual-output delta",
    )
    axes[1].set_ylabel("L2 delta")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    colors = np.where(frame.net_delta_growth >= 0, "#d95f02", "#1b9e77")
    axes[2].bar(
        frame.layer,
        frame.net_delta_growth,
        color=colors,
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Net residual-delta growth")
    axes[2].set_xlabel("Layer")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Where the 64-token perturbation enters and grows",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def make_patch_plot(patch: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    colors = {
        "residual_in": "#1f77b4",
        "attn_out": "#ff7f0e",
        "mlp_out": "#2ca02c",
    }
    for kind, frame in patch.groupby("patch_kind"):
        frame = frame.sort_values("layer")
        axes[0].plot(
            frame.layer,
            frame.qk_recovery_fraction,
            marker="o",
            label=kind,
            color=colors[kind],
        )
        axes[1].plot(
            frame.layer,
            frame.fixed_margin_recovery_fraction,
            marker="o",
            label=kind,
            color=colors[kind],
        )
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=1)
        axis.axhline(1.0, color="gray", linestyle="--")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Critical-QK recovery fraction")
    axes[1].set_ylabel("Output-margin recovery fraction")
    axes[1].set_xlabel("Patched layer")
    fig.suptitle(
        "Causal patching: transplant 143,424 activations into 143,488",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def make_head_plot(
    heads: pd.DataFrame,
    attention: pd.DataFrame,
    output: Path,
) -> None:
    source = attention[attention.condition == "source"].set_index(
        ["layer", "head"]
    )
    target = attention[attention.condition == "target"].set_index(
        ["layer", "head"]
    )
    heads = heads.copy()
    keys = list(zip(heads["layer"], heads["head"]))
    heads["gold_attention_log_ratio"] = [
        np.log(
            max(float(target.loc[key, "gold_attention"]), 1e-30)
            / max(float(source.loc[key, "gold_attention"]), 1e-30)
        )
        for key in keys
    ]
    heads = heads.sort_values("weighted_qk_change")
    labels = [f"L{r.layer}H{r.head}" for r in heads.itertuples()]
    x = np.arange(len(heads))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes[0].bar(x, heads.weighted_qk_change)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Weighted gold-QK change")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, heads.gold_attention_log_ratio, color="#d95f02")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("log(target/source gold attention)")
    axes[1].set_xticks(x, labels, rotation=70, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Critical heads: evidence score and attention collapse",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_report(
    output: Path,
    baseline: dict[str, Any],
    trace: dict[str, Any],
    patch: dict[str, Any],
    attention: dict[str, Any],
    heads: pd.DataFrame,
) -> None:
    source = baseline["source"]
    target = baseline["target"]
    attn_source = attention["source"]
    attn_target = attention["target"]
    head_drop = heads.sort_values(
        "weighted_qk_change"
    ).head(8)
    growth = trace["largest_net_growth"]

    lines = [
        "# 64-token 微扰如何放大为单跳检索失败",
        "",
        "## 基线变化",
        "",
        (
            f"- 长度：{baseline['source_total']:,} → "
            f"{baseline['target_total']:,}，只增加 64 token。"
        ),
        (
            f"- 关键证据加权 QK：{source['critical_qk']:.3f} → "
            f"{target['critical_qk']:.3f}。"
        ),
        (
            f"- P(nine)：{100 * source['gold_probability']:.2f}% → "
            f"{100 * target['gold_probability']:.2f}%。"
        ),
        (
            f"- nine 相对 `{baseline['fixed_competitor_token_label']}` "
            f"的 margin："
            f"{source['gold_vs_fixed_competitor_margin']:+.3f} → "
            f"{target['gold_vs_fixed_competitor_margin']:+.3f}。"
        ),
        "",
        "## 一步一步的计算链",
        "",
        (
            f"第一处非零残差差异出现在第 "
            f"{trace['first_nonzero_layer']} 层输出。"
        ),
        "",
        "残差差异净增长最大的层：",
        "",
        "| 层 | 输入差异 | 输出差异 | 净增长 | attention 差异 | MLP 差异 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in growth:
        lines.append(
            f"| {int(row['layer'])} | "
            f"{row['residual_in_delta']:.3f} | "
            f"{row['residual_out_delta']:.3f} | "
            f"{row['net_delta_growth']:+.3f} | "
            f"{row['attention_delta']:.3f} | "
            f"{row['mlp_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "关键 head 中真实证据 attention 的加权均值：",
            "",
            (
                f"- {baseline['source_total']:,}："
                f"{100 * attn_source['gold_attention']:.6f}%"
            ),
            (
                f"- {baseline['target_total']:,}："
                f"{100 * attn_target['gold_attention']:.6f}%"
            ),
            "",
            "证据 QK 下降贡献最大的 head：",
            "",
            "| head | Query cosine | QK 变化 | 加权 QK 变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in head_drop.itertuples():
        lines.append(
            f"| L{row.layer}H{row.head} | "
            f"{row.query_cosine:.4f} | "
            f"{row.target_qk - row.source_qk:+.3f} | "
            f"{row.weighted_qk_change:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 因果 patch",
            "",
            "恢复率 1 表示完全恢复到较短序列，0 表示没有恢复。",
            "",
            "| patch 类型 | QK 恢复最强层 | QK 恢复率 | margin 恢复最强层 | margin 恢复率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for kind in ("residual_in", "attn_out", "mlp_out"):
        row = patch[kind]
        lines.append(
            f"| `{kind}` | {row['best_qk_layer']} | "
            f"{row['best_qk_recovery']:.3f} | "
            f"{row['best_margin_layer']} | "
            f"{row['best_margin_recovery']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 公式口径",
            "",
            "每层残差传播为：",
            "",
            "$$",
            "h_{l+1}=h_l+A_l(h_l)+M_l(h_l+A_l(h_l)).",
            "$$",
            "",
            "两种长度的差异满足：",
            "",
            "$$",
            "\\delta h_{l+1}=\\delta h_l+\\delta A_l+\\delta M_l.",
            "$$",
            "",
            "最终 pre-RoPE Query 为：",
            "",
            "$$",
            "q_l=W_Q^{(l)}\\operatorname{RMSNorm}(h_l).",
            "$$",
            "",
            "图表：",
            "",
            "- `layerwise_perturbation_growth.png`",
            "- `activation_patch_recovery.png`",
            "- `critical_head_attention_collapse.png`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    baseline = json.loads(
        (root / "baseline.json").read_text(encoding="utf-8")
    )
    trace_frame = pd.read_csv(root / "layer_stage_trace.csv")
    patch_frame = pd.read_csv(root / "activation_patch.csv")
    attention_frame = pd.read_csv(
        root / "critical_head_attention.csv"
    )
    head_frame = pd.read_csv(root / "critical_head_changes.csv")

    trace = trace_summary(trace_frame)
    patch = patch_summary(patch_frame)
    attention = weighted_attention_summary(attention_frame)
    summary = {
        "baseline": baseline,
        "trace": trace,
        "patch": patch,
        "attention": attention,
    }
    (root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_trace_plot(
        trace,
        root / "layerwise_perturbation_growth.png",
    )
    make_patch_plot(
        patch_frame,
        root / "activation_patch_recovery.png",
    )
    make_head_plot(
        head_frame,
        attention_frame,
        root / "critical_head_attention_collapse.png",
    )
    write_report(
        root / "report.md",
        baseline,
        trace,
        patch,
        attention,
        head_frame,
    )


if __name__ == "__main__":
    main()
